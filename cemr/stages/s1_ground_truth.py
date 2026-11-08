"""Stage 1: ground-truth extraction, re-grounded in raw rows and verified."""

from __future__ import annotations

import logging

from ..crossview import mark_visibility
from ..prompts import ground_truth as P
from ..schemas import Candidate, GroundTruth
from ..schemas import VerifierScores
from ..validation.deterministic import (check_ground_truth, is_fatal, is_primary,
                                         prune_secondary_facts)
from ..validation.verifier import generate_verified
from .context import Ctx, with_feedback

log = logging.getLogger(__name__)

# Characters models wrap around copied ids: <id>, [id], "id", `id`.
_ID_WRAPPERS = "<>[]\"'` "


def sample_id_of(cand: Candidate) -> str:
    return cand.candidate_id


def normalize_evidence_ids(gt: GroundTruth) -> int:
    """
    Strip wrapper characters from every cited evidence id, in place, and drop
    duplicates/empties. Returns how many ids were changed.

    Only formatting is repaired here. Ids that are still not real rows of this
    episode (e.g. an episode id cited instead of a row id) are left untouched,
    so `check_ground_truth` still rejects them.
    """
    changed = 0
    for fact in gt.facts:
        ids = getattr(fact, "evidence_ids", None)
        if not ids:
            continue
        clean, seen = [], set()
        for raw in ids:
            cid = str(raw).strip(_ID_WRAPPERS)
            if cid != raw:
                changed += 1
            if cid and cid not in seen:
                seen.add(cid)
                clean.append(cid)
        fact.evidence_ids = clean
    return changed


def verifier_payload(gt: GroundTruth) -> dict:
    """
    What the LLM verifier judges: only the model's own claims. Fields the code sets
    itself (`source`, `visible_to_user`, dropped-fact records) are removed; the
    verifier misreads them (e.g. source="caption" as a row type).
    """
    d = gt.model_dump(mode="json")
    # sample_id embeds a truncated plan text for T2 ("...organize de"); the verifier
    # read it as the action itself. The target/lure ids stay.
    d.pop("sample_id", None)
    for f in d.get("facts", []):
        f.pop("source", None)
        f.pop("visible_to_user", None)
    d.get("task_specific", {}).pop("dropped_facts", None)
    return d


def rows_without_summaries(rows_block: str) -> str:
    """
    Episode headers look like "== <id>[ [OTHER VIEW: x]] | <automatic summary>".
    The summary is not evidence, so the verifier must not use it to judge
    completeness; keep only the id (and view tag).
    """
    return "\n".join(line.split(" | ", 1)[0] if line.startswith("== ") else line
                     for line in rows_block.splitlines())


def make_repair(cand: Candidate, index):
    """
    Verifier repair for ground truth: facts the verifier flags as unsupported are
    DROPPED and recorded (task_specific["dropped_facts"]); the sample is rejected
    only if the flagged fact is the primary one. When the verifier flags specific
    facts, dropping them resolves the "grounded" criterion.
    """
    def repair(gt: GroundTruth, vs: VerifierScores):
        flagged = {str(f.get("fact_id")): str(f.get("reason", ""))[:200] for f in vs.flagged}
        if not flagged:
            return gt, set(), []
        known = {f.fact_id: f for f in gt.facts}
        hits = {fid: why for fid, why in flagged.items() if fid in known}
        primary_hit = [fid for fid in hits if is_primary(known[fid], cand, index.synonyms)]
        if primary_hit:
            fid = primary_hit[0]
            return None, set(), [f"primary fact {fid} is not supported by its cited rows: {hits[fid]}. "
                                 f"Cite rows that state it, and keep the statement to what they say."]
        if not hits:   # verifier named facts that do not exist: cannot use its judgment
            return gt, set(), []
        dropped = [{"fact_id": fid, "slot": known[fid].slot, "value": known[fid].value,
                    "reason": f"verifier: {why}"} for fid, why in hits.items()]
        gt.facts = [f for f in gt.facts if f.fact_id not in hits]
        gt.task_specific.setdefault("dropped_facts", []).extend(dropped)
        return gt, {"grounded"}, [f"dropped {fid}: {why}" for fid, why in hits.items()]
    return repair


def run(ctx: Ctx, cand: Candidate) -> tuple[GroundTruth | None, dict]:
    sid = sample_id_of(cand)
    rows = ctx.candidate_rows_block(cand)
    prompt = P.build(sid, cand.task.value, cand.participant, cand, rows,
                     cand.primary_slot, cand.true_value)

    def gen(feedback):
        gt = ctx.generator.generate_json(P.SYSTEM, with_feedback(prompt, feedback), schema=GroundTruth)
        gt.narrator = cand.participant
        n = normalize_evidence_ids(gt)                       # before visibility + checks use the ids
        if n:
            log.debug("%s: normalized %d wrapped evidence ids", sid, n)
        mark_visibility(gt.facts, cand.participant, ctx.index)   # deterministic, never trust the LLM
        # Unsupported SECONDARY facts are dropped (and recorded) instead of rejecting
        # the whole sample; a failing primary fact still rejects it in the check below.
        dropped = prune_secondary_facts(gt, cand, ctx.index)
        if dropped:
            gt.task_specific["dropped_facts"] = dropped
            log.debug("%s: dropped %d unsupported secondary facts", sid, len(dropped))
        return gt

    gt, report = generate_verified(
        generate=gen,
        deterministic_check=lambda g: check_ground_truth(g, cand, ctx.index, ctx.xv),
        verifier=ctx.verifier,
        kind="ground_truth",
        to_payload=verifier_payload,
        evidence_rows=rows_without_summaries(rows),
        max_regenerations=ctx.cfg.verify.max_regenerations,
        repair=make_repair(cand, ctx.index),
        fatal=is_fatal,
        reverify_after_repair=ctx.cfg.verify.reverify_after_repair,
    )
    if gt is not None:
        _refine_subtype(cand, gt)
    return gt, report


def _refine_subtype(cand: Candidate, gt: GroundTruth) -> None:
    """Use Stage-1 labels to refine miner subtypes (reported separately in results)."""
    ts = gt.task_specific
    if cand.task.name == "T4":
        cand.subtype = "report_accurate" if ts.get("report_consistent_with_record", True) else "report_conflicts"
    if cand.task.name == "T2" and cand.subtype == "never_executed" and ts.get("execution_found"):
        cand.meta["stage1_found_execution"] = True