"""
pipeline.py

Orchestrates Stage 0 -> 4 with resumable, per-stage JSONL outputs:

outputs/
  config.json                  resolved config (reproducibility)
  index_stats.json
  candidates/{task}.jsonl      Stage 0
  gt.jsonl / fm.jsonl / plan.jsonl
  conv.json                    accepted conversations: ONE indented JSON list
  conversations/{sample}.json  the same conversations, one indented file each
  rejects.jsonl                every dropped sample + stage + reasons
  sft/{split}.jsonl            training-format dialogues
  llm_logs/*.jsonl             every LLM call
  funnel.json                  per-task acceptance at each stage
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Config
from .crossview import build_clusters
from .index import MemoryIndex
from .llm_client import LLMUnavailable, seed_scope
from .miners import ALL_MINERS
from .prompts import conversation as CP
from .schemas import Candidate, DialoguePlan, FalseMemory, GroundTruth
from .stages import s1_ground_truth, s2_false_memory, s3_plan, s4_conversation
from .stages.context import Ctx
from .utils import append_json_list, append_jsonl, read_conversations, read_jsonl, write_json
from .validation.deterministic import CONCRETE_SLOTS, HEAD_NOUN_SLOTS, lexically_supported
from .validation.verifier import LLMVerifier

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config, index: MemoryIndex, clients: dict):
        self.cfg = cfg
        self.clients = clients
        self._stop = threading.Event()
        for client in clients.values():
            if hasattr(client, "stop_event"):
                client.stop_event = self._stop
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        cfg.dump(self.out / "config.json")
        verifier = LLMVerifier(clients["verifier"], cfg.verify.threshold) if clients.get("verifier") else None
        self.ctx = Ctx(cfg=cfg, index=index, generator=clients["generator"],
                       user_sim=clients.get("user_sim", clients["generator"]), verifier=verifier)
        write_json(self.out / "index_stats.json", dict(index.stats))

    ###########################################################################
    # Stage 0
    ###########################################################################

    def mine(self, tasks=None) -> dict[str, list[Candidate]]:
        result = {}
        for miner_cls in ALL_MINERS:
            if tasks and miner_cls.task.name not in tasks:
                continue
            cands = miner_cls(self.ctx.index, self.cfg, self.ctx.xv).mine()
            path = self.out / "candidates" / f"{miner_cls.task.name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            for c in cands:
                append_jsonl(path, c.model_dump(mode="json"))
            result[miner_cls.task.name] = cands
            log.info("mined %s: %d candidates", miner_cls.task.name, len(cands))
        return result

    def load_candidates(self, task: str) -> list[Candidate]:
        return [Candidate.model_validate(o) for o in read_jsonl(self.out / "candidates" / f"{task}.jsonl")]

    ###########################################################################
    # Stages 1-4
    ###########################################################################

    def preflight(self) -> None:
        """
        Before any sample is attempted, confirm every configured model is actually
        served at its base_url. A wrong model name or swapped port otherwise makes
        every call fail (e.g. HTTP 404) and every candidate lands in rejects.jsonl,
        where resume logic then skips it forever.
        """
        problems, seen = [], set()
        for role, client in self.clients.items():
            key = (client.model, str(client.client.base_url))
            if key in seen:
                continue
            seen.add(key)
            log.info("Preflight: checking %s model=%s at %s", role, client.model, key[1])
            try:
                result = (client._request("models.list", client.client.models.list)
                          if hasattr(client, "_request") else client.client.models.list())
                served = [m.id for m in result.data]
            except LLMUnavailable as e:
                raise SystemExit(f"LLM preflight failed: {e}") from e
            except Exception as e:  # tunnel down, server not running, wrong port
                problems.append(f"{role}: cannot reach {key[1]} ({str(e)[:120]})")
                continue
            # Hosted APIs may list ids with a prefix ("models/gemini-..."); match either form.
            if not any(m == client.model or m.split("/", 1)[-1] == client.model for m in served):
                problems.append(f"{role}: model {client.model!r} not served at {key[1]}; server has {served}")
            elif getattr(client, "gemini", False):
                # Listing a model does not guarantee this project's access to it.
                # Check the native endpoint before spending tokens on samples.
                try:
                    client._count_input([{"role": "user", "content": "Hello"}])
                except LLMUnavailable as e:
                    raise SystemExit(f"LLM preflight failed ({role}): {e}") from e
        if problems:
            msg = "LLM preflight failed -- nothing was generated or recorded:\n  " + "\n  ".join(problems)
            log.error(msg)
            raise SystemExit(msg)
        log.info("LLM preflight OK: %s", ", ".join(f"{m} @ {u}" for m, u in sorted(seen)))

    # Stop a run after this many samples crash (server down, tunnel dropped, ...):
    # crashes are NOT recorded as rejects, so those samples are retried next run.
    MAX_CRASHES = 10

    def _load_saved(self) -> None:
        """
        Accepted stage outputs from earlier runs, by sample id. When a sample is
        retried (e.g. its line was removed from rejects.jsonl after a fix), finished
        stages are reused instead of regenerated, and nothing is appended twice.
        """
        self.saved_gt = {o["gt"]["sample_id"]: (Candidate.model_validate(o["candidate"]),
                                                  GroundTruth.model_validate(o["gt"]))
                         for o in read_jsonl(self.out / "gt.jsonl")}
        self.saved_fm = {o["sample_id"]: FalseMemory.model_validate(o) for o in read_jsonl(self.out / "fm.jsonl")}
        self.saved_plan = {o["sample_id"]: DialoguePlan.model_validate(o)
                           for o in read_jsonl(self.out / "plan.jsonl")}

    def generate(self, tasks=None, limit_per_task: int | None = None, splits=None) -> dict:
        self._stop.clear()
        self.preflight()
        self._load_saved()
        finished = read_conversations(self.out)
        done = {c["sample_id"] for c in finished}
        for c in finished:          # backfill files for conversations from earlier runs
            if not self._conversation_path(c["sample_id"]).exists():
                self._write_conversation_file(c)
        rejected = {o["sample_id"] for o in read_jsonl(self.out / "rejects.jsonl")}
        funnel: dict = defaultdict(Counter)
        self._lock = threading.Lock()
        jobs: list[Candidate] = []
        tasks = tasks or [m.task.name for m in ALL_MINERS]
        for task in tasks:
            all_cands = self.load_candidates(task)
            if not all_cands:
                log.warning("%s: no candidates in %s -- run `mine` first (after changing output_dir "
                            "or split days, candidates must be re-mined)", task, self.out / "candidates")
            cands = [c for c in all_cands if not splits or c.split in splits]
            if all_cands and not cands:
                log.warning("%s: no candidates in splits %s", task, splits)
            # Free pre-filter (no LLM): the mined answer must be stated in the user's own
            # captions for the target episode. Event files were LLM-annotated, so some
            # mined values are paraphrased or unsupported; these would be rejected at
            # Stage 1 anyway. Applied BEFORE --limit so the limit counts usable candidates.
            usable = [c for c in cands if self.true_value_supported(c)]
            funnel[task]["filtered_true_value_not_in_captions"] += len(cands) - len(usable)
            cands = usable
            # --limit N means N NEW samples per task: finished and rejected samples are
            # skipped first (and counted), so a limit never selects only old samples.
            new = []
            for cand in cands:
                sid = cand.candidate_id
                if sid in done:
                    funnel[task]["skipped_already_done"] += 1
                elif sid in rejected:
                    funnel[task]["skipped_already_rejected"] += 1
                else:
                    new.append(cand)
            funnel[task]["candidates"] = len(all_cands)
            for cand in new[: limit_per_task or len(new)]:
                funnel[task]["mined"] += 1
                jobs.append(cand)

        workers = max(1, int(getattr(self.cfg.llm, "concurrency", 1) or 1))
        log.info("generating %d samples with %d worker(s)", len(jobs), workers)
        crashes = 0
        failure = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._run_one, c, funnel[c.task.name]): c for c in jobs}
            for fut in as_completed(futures):
                cand = futures[fut]
                try:
                    fut.result()
                except Exception as e:  # infrastructure failure: not a content reject
                    crashes += 1
                    log.error("sample %s crashed (%s); it will be retried next run", cand.candidate_id, e)
                    with self._lock:
                        funnel[cand.task.name]["crashed"] += 1
                    if isinstance(e, LLMUnavailable) or crashes >= self.MAX_CRASHES:
                        failure = e
                        self._stop.set()
                        log.error("Stopping generation: %s. Completed stages are saved; failed samples are not rejections.", e)
                        for f in futures:
                            f.cancel()
                        break
        write_json(self.out / "funnel.json", {k: dict(v) for k, v in funnel.items()})
        if failure is not None:
            raise SystemExit(f"Generation stopped: {failure}")
        return funnel

    def _run_one(self, cand, counter):
        if self._stop.is_set():
            return
        try:
            self._one(cand, counter)
        except LLMUnavailable:
            # Set before the worker can dequeue another candidate.
            self._stop.set()
            raise

    def _conversation_path(self, sample_id: str) -> Path:
        return self.out / "conversations" / f"{safe_filename(sample_id)}.json"

    def _write_conversation_file(self, conv: dict) -> None:
        """One readable, indented JSON file per accepted conversation (also kept in conv.json)."""
        write_json(self._conversation_path(conv["sample_id"]), conv)

    def _count(self, counter: Counter, key: str) -> None:
        with self._lock:
            counter[key] += 1

    def true_value_supported(self, cand: Candidate) -> bool:
        """Is the candidate's true value lexically present in the user's own target rows?"""
        if cand.primary_slot not in CONCRETE_SLOTS:
            return True                      # derived slots (order, day, ...) are checked later
        idx = self.ctx.index
        if cand.target_episode not in idx.episodes:
            return False
        rows = [r for r in idx.episode_rows(cand.target_episode) if r.participant == cand.participant]
        texts = [r.text for r in rows]
        if cand.primary_slot == "people":
            texts += [r.speaker for r in rows if r.speaker]
        return lexically_supported(cand.true_value, texts, idx.synonyms,
                                   head_noun=cand.primary_slot in HEAD_NOUN_SLOTS)

    def _reject(self, cand: Candidate, stage: str, report: dict) -> None:
        log.warning("REJECT sample=%s stage=%s report=%s", cand.candidate_id, stage, report)
        append_jsonl(self.out / "rejects.jsonl",
                     {"sample_id": cand.candidate_id, "task": cand.task.value, "stage": stage, "report": report})

    def _one(self, cand: Candidate, counter: Counter) -> None:
        ctx = self.ctx
        sid = cand.candidate_id
        log.info("START sample=%s task=%s", sid, cand.task.name)
        with seed_scope(sid):
            if sid in self.saved_gt:
                cand, gt = self.saved_gt[sid]           # saved candidate keeps Stage-1 subtype refinements
                self._count(counter, "gt_reused")
            else:
                gt, rep = s1_ground_truth.run(ctx, cand)
                if gt is None:
                    return self._reject(cand, "ground_truth", rep)
                append_jsonl(self.out / "gt.jsonl",
                             {"candidate": cand.model_dump(mode="json"), "gt": gt.model_dump(mode="json")})
            self._count(counter, "gt")
            log.info("sample=%s ground_truth accepted%s", sid, " (reused)" if sid in self.saved_gt else "")

            if sid in self.saved_fm:
                fm = self.saved_fm[sid]
                self._count(counter, "fm_reused")
            else:
                fm, rep = s2_false_memory.run(ctx, cand, gt)
                if fm is None:
                    return self._reject(cand, "false_memory", rep)
                append_jsonl(self.out / "fm.jsonl", fm.model_dump(mode="json"))
            self._count(counter, "fm")
            log.info("sample=%s false_memory accepted%s", sid, " (reused)" if sid in self.saved_fm else "")

            if sid in self.saved_plan:
                plan = self.saved_plan[sid]
                self._count(counter, "plan_reused")
            else:
                plan, rep = s3_plan.run(ctx, cand, gt, fm)
                if plan is None:
                    return self._reject(cand, "plan", rep)
                append_jsonl(self.out / "plan.jsonl", plan.model_dump(mode="json"))
            self._count(counter, "plan")
            log.info("sample=%s plan accepted%s", sid, " (reused)" if sid in self.saved_plan else "")

            conv, rep = s4_conversation.run(ctx, cand, gt, fm, plan)
            if conv is None:
                return self._reject(cand, "conversation", rep)
            self._count(counter, "conv")
        append_json_list(self.out / "conv.json", conv.model_dump(mode="json"))
        self._write_conversation_file(conv.model_dump(mode="json"))
        system = CP.ASSISTANT_SYSTEM.format(max_corr=self.cfg.dialogue.max_corrections_per_turn)
        append_jsonl(self.out / "sft" / f"{cand.split}.jsonl", s4_conversation.to_training_example(conv, system))
        log.info("ACCEPT sample=%s conversation saved", sid)

    ###########################################################################
    # Loading finished samples (for evaluation)
    ###########################################################################

    def load_samples(self, split: str = "test"):
        """Yields (candidate, gt, fm, opening_text) for accepted conversations."""
        gts = {o["gt"]["sample_id"]: (Candidate.model_validate(o["candidate"]), GroundTruth.model_validate(o["gt"]))
               for o in read_jsonl(self.out / "gt.jsonl")}
        fms = {o["sample_id"]: FalseMemory.model_validate(o) for o in read_jsonl(self.out / "fm.jsonl")}
        for conv in read_conversations(self.out):
            sid = conv["sample_id"]
            if conv["split"] != split or sid not in gts or sid not in fms:
                continue
            cand, gt = gts[sid]
            yield cand, gt, fms[sid], conv["turns"][0]["text"]


def safe_filename(sample_id: str) -> str:
    """Sample ids can contain spaces or '/' (T2 embeds plan text); make a safe file name."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)[:180]


def build_index(cfg: Config, rebuild: bool = False) -> MemoryIndex:
    """Load the saved Part-1 index if present (run `cemr.cli index` first), else build it."""
    saved = Path(cfg.data.index_dir) / "index_meta.json"
    if saved.exists() and not rebuild:
        log.info("loading saved index from %s", cfg.data.index_dir)
        return MemoryIndex.load(cfg.data.index_dir)
    d = cfg.data
    idx = MemoryIndex.build(d.captions_glob, d.events_glob, d.synonyms_path,
                            d.people, d.people_aliases, d.participants)
    build_clusters(idx, cfg.mining.cluster_min_overlap)   # sets episode.cluster_id
    return idx


__all__ = ["Pipeline", "build_index", "DialoguePlan"]
