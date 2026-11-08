"""
Stage 4: conversation generation via two-agent self-play, then verification.

Accepted only if:
  * the user's final belief equals the gold belief (converged),
  * no future value leaked (after retries),
  * every correcting assistant turn carries evidence ids,
  * the LLM conversation verifier passes every rubric criterion.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

from ..dialogue.selfplay import run_generation
from ..llm_client import LLMUnavailable, seed_scope
from ..schemas import Candidate, Conversation, DialoguePlan, FalseMemory, GroundTruth
from ..validation.deterministic import repeated_turns
from .context import Ctx
from .s1_ground_truth import rows_without_summaries


def check_conversation(conv: Conversation) -> list[str]:
    errs = []
    if not conv.converged:
        errs.append(f"did not converge: final={conv.final_belief} gold={conv.gold_belief}")
    if conv.flags.get("leak_unresolved"):
        errs.append(f"leaked future values: {conv.flags['leak_unresolved']}")
    for t in conv.turns:
        if t["role"] == "assistant" and t.get("corrected_attribute") and not t.get("evidence_ids"):
            errs.append("assistant corrected without evidence")
    if conv.turns[-1]["role"] != "user":
        errs.append("conversation must end on a user turn")
    errs += repeated_turns(conv.turns)
    return errs


def _evidence_rows(ctx: Ctx, cand: Candidate, conv: Conversation) -> str:
    """
    Rows the verifier judges against: target/lure episodes (first 30 rows each) PLUS
    every row the conversation actually cites, wherever it sits in the episode.
    Summaries are removed: they are not evidence.
    """
    cited = [e for t in conv.turns if t["role"] == "assistant" for e in (t.get("evidence_ids") or [])]
    cited += [t["selected_anchor"] for t in conv.turns if t["role"] == "assistant" and t.get("selected_anchor")]
    eps = [cand.target_episode, *cand.lure_episodes, *cand.meta.get("verify_episodes", [])]
    rows = list(dict.fromkeys(cited + list(cand.lure_rows) + list(cand.meta.get("verify_rows", []))))
    return rows_without_summaries(ctx.rows_block(rows, eps, 30, user=cand.participant))


def run(ctx: Ctx, cand: Candidate, gt: GroundTruth, fm: FalseMemory, plan: DialoguePlan):
    report = {"attempts": []}
    feedback: list[str] | None = None
    max_regenerations = max(0, min(3, ctx.cfg.verify.max_regenerations))
    for attempt in range(max_regenerations + 1):
        log.info("sample=%s conversation attempt %d/%d; feedback=%s",
                 cand.candidate_id, attempt + 1, max_regenerations + 1, feedback)
        # Own seed scope per attempt: a retry is a real resample, not a replay.
        with seed_scope(cand.candidate_id, "conversation", attempt):
            try:
                conv = run_generation(ctx, cand, gt, fm, plan, feedback)
            except LLMUnavailable:
                raise                   # server problem: never recorded as a reject
            except Exception as e:
                log.warning("sample=%s conversation generation failed: %s", cand.candidate_id, e)
                report["attempts"].append({"error": str(e)[:300]})
                continue
            errs = check_conversation(conv)
            entry = {"deterministic_errors": errs}
            if not errs and ctx.verifier is not None:
                payload = {"task": cand.task.value,
                           "turns": [{k: t[k] for k in ("role", "text")} for t in conv.turns],
                           "hidden_fields": [{k: t.get(k) for k in ("intent", "evidence_ids", "corrected_attribute")}
                                             for t in conv.turns if t["role"] == "assistant"],
                           "gold": conv.gold_belief, "user_initial_memory": fm.user_memory}
                vs = ctx.verifier.score("conversation", payload, _evidence_rows(ctx, cand, conv))
                entry["verifier"] = vs.model_dump()
                failing = vs.failing(ctx.verifier.threshold, ignore=ctx.verifier.advisory("conversation"))
                if failing:
                    log.warning("sample=%s conversation VERIFIER REJECT: scores=%s rationale=%s",
                                cand.candidate_id, failing, vs.rationale)
                    errs = [f"verifier {k}={v:.2f}" for k, v in failing.items()]
                    feedback = [f"{k}: {vs.rationale.get(k, '')}" for k in failing] + vs.suggestions[:4]
            if errs and not feedback:
                feedback = errs
            if errs:
                log.warning("sample=%s conversation rejection: %s", cand.candidate_id, errs)
            report["attempts"].append(entry)
            if not errs:
                report["accepted_on_attempt"] = attempt
                return conv, report
            if entry.get("deterministic_errors"):
                feedback = errs
    report["rejected"] = True
    return None, report


def to_training_example(conv: Conversation, system_prompt: str) -> dict:
    """
    Chat-format SFT example. Assistant targets are JSON carrying the hidden
    supervision fields plus the spoken text, so a model learns the policy
    (intent, cues, anchor, evidence, corrected attribute), not only the words.
    """
    import json
    # The model must know "now" to interpret "yesterday" and to reason about what it
    # can have recorded; the same line is given to models under evaluation.
    now = f"\nCurrent time: {conv.query_time_label}." if conv.query_time_label else ""
    msgs = [{"role": "system", "content": system_prompt + now}]
    for t in conv.turns:
        if t["role"] == "user":
            msgs.append({"role": "user", "content": t["text"]})
        else:
            target = {k: t.get(k) for k in ("intent", "retrieval_cues", "selected_anchor",
                                            "evidence_type", "evidence_ids", "corrected_attribute")}
            target["text"] = t["text"]
            msgs.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return {"sample_id": conv.sample_id, "task": conv.task.value, "query_time": conv.query_time,
            "delay_bucket": conv.delay_bucket, "messages": msgs}
