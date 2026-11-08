"""
Stage 2: false memory.

The PRIMARY edit is fixed by the miner (task purity). Secondary edits are
offered as a pool of real lure values; the LLM chooses among them and writes
the user's memory. Nothing false is invented by the LLM.
"""

from __future__ import annotations

from ..prompts import false_memory as P
from ..schemas import BeliefCondition, Candidate, FalseMemory, GroundTruth, PRIMARY_ERROR
from ..utils import contains_value, now_label, short_time, stable_rng
from ..validation.deterministic import (CONCRETE_SLOTS, HEAD_NOUN_SLOTS, check_false_memory, coverage,
                                         is_primary, lexically_supported)
from ..validation.verifier import generate_verified
from .context import Ctx, with_feedback

SECONDARY_TYPE = {
    "people": "person_confusion",
    "objects": "object_substitution",
    "activity": "activity_confusion",
    "location": "location_confusion",
}


def _own_evidence(ctx: Ctx, cand: Candidate, ids: list[str]) -> list[str]:
    """Evidence the assistant may show the user: rows from the user's OWN recording."""
    return [e for e in ids if e in ctx.index.rows and ctx.index.rows[e].participant == cand.participant]


def _primary_edit(ctx: Ctx, cand: Candidate, gt: GroundTruth) -> dict:
    # The primary fact is the one carrying the mined true value (other facts may share its slot).
    fact = next((f for f in gt.facts if f.visible_to_user and is_primary(f, cand, ctx.index.synonyms)),
                None) or next(f for f in gt.facts if f.slot == cand.primary_slot and f.visible_to_user)
    return {
        "slot": cand.primary_slot,
        "true": fact.value,
        "false": cand.lure_value,
        "error_type": PRIMARY_ERROR[cand.task],
        "role": "primary",
        "lure_source": (cand.lure_episodes or cand.lure_rows or ["miner"])[0],
        "true_evidence": _own_evidence(ctx, cand, fact.evidence_ids),
    }


def _stated(ctx: Ctx, value: str, ep, slot: str) -> bool:
    """Is `value` stated in the narrator's own captions of `ep`? (annotations can invent values)"""
    rows = [r for r in ctx.index.episode_rows(ep.episode_id) if r.participant == ep.participant]
    texts = [r.text for r in rows]
    if slot == "people":
        return lexically_supported(value, texts + [r.speaker for r in rows if r.speaker], ctx.index.synonyms)
    if slot in HEAD_NOUN_SLOTS:
        return lexically_supported(value, texts, ctx.index.synonyms, head_noun=True)
    return coverage(value, texts) >= 0.5


def _secondary_pool(ctx: Ctx, cand: Candidate, gt: GroundTruth) -> list[dict]:
    """
    Real alternative values for secondary edits, borrowed from episodes a person could
    plausibly blend with the target:
      * the lure episode(s) first;
      * then the user's OWN episodes within 3 h that share a person or the place
        (previously any episode of ANY wearer that day, which produced unrelated,
        implausible details such as "discussing strategy" in a baking memory).
    A value is only borrowed if the source episode's captions state it, and an
    activity only from an episode that shares both people and place (activities
    change the gist most).
    """
    idx = ctx.index
    target = idx.episodes[cand.target_episode]
    narrator = cand.participant
    t_people = set(target.people) - {narrator}

    def related(e):
        return bool((set(e.people) - {narrator}) & t_people) or (e.location and e.location == target.location)

    lures = [idx.episodes[e] for e in cand.lure_episodes if e in idx.episodes]
    nearby = [e for e in idx.episodes_sorted(narrator)
              if e.episode_id != target.episode_id and e.episode_id not in cand.lure_episodes
              and abs(e.abs_start - target.abs_start) <= 3 * 3600 and related(e)]
    sources = lures + nearby
    pool = {}
    for f in gt.facts:
        if (f.slot not in CONCRETE_SLOTS or f.slot == cand.primary_slot or not f.visible_to_user
                or f.source != "caption" or f.slot in pool):
            continue
        for src in sources:
            if f.slot == "activity" and not (related(src) and (set(src.people) - {narrator}) & t_people
                                             and src.location == target.location):
                continue
            alts = [v for v in src.slot_values(f.slot)
                    if v and v != narrator and v not in target.slot_values(f.slot)
                    and _stated(ctx, v, src, f.slot)]
            if alts:
                pool[f.slot] = {"slot": f.slot, "true": f.value, "false": alts[0],
                                "error_type": SECONDARY_TYPE[f.slot], "role": "secondary",
                                "lure_source": src.episode_id,
                                "true_evidence": _own_evidence(ctx, cand, f.evidence_ids)}
                break
    return list(pool.values())


def _borrowed_from(ctx: Ctx, fm: FalseMemory) -> list[dict]:
    """For the verifier: where each false value comes from, with the rows that state it."""
    idx, out = ctx.index, []
    for e in fm.edits:
        src = e.lure_source
        if src in idx.rows:
            rows = [idx.rows[src]]
        elif src in idx.episodes:
            rows = [r for r in idx.episode_rows(src)
                    if contains_value(r.text + " " + (r.speaker or ""), e.false)
                    or lexically_supported(e.false, [r.text], idx.synonyms, head_noun=True)][:2]
        else:
            rows = []
        out.append({"slot": e.slot, "false_value": e.false, "role": e.role, "borrowed_from": src,
                    "rows": [f"[{short_time(r.abs_start)}] {r.text}" for r in rows]})
    return out


def run(ctx: Ctx, cand: Candidate, gt: GroundTruth) -> tuple[FalseMemory | None, dict]:
    cond = BeliefCondition(cand.meta.get("belief_condition", "misbound"))
    rng = stable_rng("fm", cand.candidate_id)
    primary = _primary_edit(ctx, cand, gt) if cond == BeliefCondition.MISBOUND else None
    pool = _secondary_pool(ctx, cand, gt) if cond == BeliefCondition.MISBOUND else []
    n_secondary = rng.randint(1, min(3, len(pool))) if pool else 0
    omitted = []
    if cond == BeliefCondition.INCOMPLETE:
        present = sorted({f.slot for f in gt.facts if f.slot in CONCRETE_SLOTS and f.visible_to_user})
        omitted = rng.sample(present, k=min(len(present), rng.randint(1, 2))) if present else []

    lure_values = {p["false"].lower() for p in pool} | {cand.lure_value.lower()}
    # Facts the person could remember: their own recording, supported by caption text.
    # (Annotation-only facts, e.g. long people lists from the event file, made the
    # "correct" memories contain people the rows never show.)
    own_facts = [f for f in gt.facts if f.visible_to_user and f.source == "caption"]
    # Words the memory must not contain: the true value of the primary edit (always
    # corrected later). Secondary true values are forbidden only if chosen (see prompt).
    forbidden = [primary["true"]] if primary and primary["slot"] in CONCRETE_SLOTS else []
    target = ctx.index.episodes[cand.target_episode]
    prompt = P.build(gt.sample_id, cand.task.value, cond.value, own_facts, primary, pool, omitted,
                     n_secondary, forbidden, now=now_label(cand.query_time),
                     episode_time=now_label(target.abs_start))

    def gen(feedback):
        fm = ctx.generator.generate_json(P.SYSTEM, with_feedback(prompt, feedback), schema=FalseMemory)
        # The omitted slots are chosen here, not by the model (it sometimes listed a
        # derived slot such as completion_status, which crashed the conversation).
        fm.omitted_slots = list(omitted)
        fm.belief_condition = cond
        return fm

    # Controls (correct / incomplete memories) contain no error by design: judge them
    # with the control rubric, not the error rubric (which rejected every control).
    kind = "false_memory" if cond in (BeliefCondition.MISBOUND, BeliefCondition.UNSUPPORTED) \
        else "false_memory_control"

    def payload(fm):
        d = {"task": cand.task.value, "belief_condition": cond.value,
             "current_time": now_label(cand.query_time),        # so "yesterday" is judged correctly
             "memory": fm.model_dump(mode="json"),
             "true_facts": [{k: f.model_dump()[k] for k in ("slot", "value", "statement")} for f in gt.facts]}
        if kind == "false_memory":
            d["borrowed_from"] = _borrowed_from(ctx, fm)
        return d

    return generate_verified(
        generate=gen,
        deterministic_check=lambda fm: check_false_memory(fm, gt, cand, lure_values, ctx.cfg.verify.min_gist_overlap),
        verifier=ctx.verifier,
        kind=kind,
        to_payload=payload,
        evidence_rows=ctx.candidate_rows_block(cand, limit_per_episode=25),
        max_regenerations=ctx.cfg.verify.max_regenerations,
    )
