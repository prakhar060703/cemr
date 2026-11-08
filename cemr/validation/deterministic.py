"""
validation/deterministic.py

Rule-based checks run after every stage. They return a list of error strings
(empty == pass). Failed samples are DROPPED and the reason is logged; nothing
is silently repaired. (Stage 1 drops unsupported SECONDARY facts and records
them in task_specific["dropped_facts"]; a failing primary fact rejects the sample.)
"""

from __future__ import annotations

from ..index import MemoryIndex
from ..schemas import (BeliefCondition, Candidate, DialoguePlan, FalseMemory,
                       GroundTruth, PRIMARY_ERROR, SECONDARY_ERRORS)
from ..utils import canon, contains_value, tokens

# Slots whose values are concrete nouns that should appear in text.
CONCRETE_SLOTS = {"objects", "people", "location", "activity"}
# Slots whose values are derived relations (checked by evidence ids only).
DERIVED_SLOTS = {"temporal_order", "day", "completion_status", "knowledge_source", "causal_link"}


###############################################################################
# Stage 1
###############################################################################

# Secondary people/location facts may be supported by the episode's EVENT ANNOTATION
# when the captions never state them (names are rarely spoken; nobody narrates
# "I am in the kitchen"). Such facts are labelled source="event_annotation" and are
# never used as a correction target. Set to set() for a strictly caption-only benchmark.
ANNOTATION_SLOTS = {"people", "location"}


def _stem(tok: str) -> str:
    """Tiny plural normaliser so 'scales' supports 'scale' (no NLP dependency)."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("ches", "shes", "sses", "xes")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in tokens(text)}


# Slots whose values are noun phrases: the HEAD noun (last content word) must appear,
# so "small cabinet" is not supported by a row that only says "small".
HEAD_NOUN_SLOTS = {"objects", "location"}


def _head(value: str) -> str | None:
    toks = tokens(value)
    return _stem(toks[-1]) if toks else None


def lexically_supported(value: str, texts, synonyms: dict[str, str] | None = None,
                        head_noun: bool = False) -> bool:
    """
    True if `value` (or a synonym mapping to the same canonical value) is found in
    `texts` after plural normalisation. By default any content word suffices;
    with head_noun=True the value's last content word must be present.
    """
    hay = set()
    for t in texts:
        hay |= _stems(t)
    variants = {value}
    if synonyms:
        c = canon(value, synonyms)
        variants |= {c} | {k for k, v in synonyms.items() if v == c}
    for v in variants:
        if head_noun:
            h = _head(v)
            if h and h in hay:
                return True
        elif _stems(v) & hay:
            return True
    return False


def coverage(value: str, texts) -> float:
    """Fraction of the value's content words (plural-normalised) found in `texts`."""
    want = _stems(value)
    if not want:
        return 0.0
    hay = set()
    for t in texts:
        hay |= _stems(t)
    return len(want & hay) / len(want)


# Values too generic to make a plausible, correctable memory error ("I got food").
GENERIC_VALUES = frozenset(
    "food thing things stuff item items object objects something anything everything "
    "supplies belongings snack snacks drink drinks".split())


def is_generic(value: str) -> bool:
    toks = tokens(value)
    return not toks or all(t in GENERIC_VALUES for t in toks)


def _allowed(cand: Candidate, index: MemoryIndex) -> tuple[set[str], set[str]]:
    allowed_eps = {cand.target_episode, *cand.lure_episodes, *cand.meta.get("verify_episodes", [])}
    allowed_rows = set(cand.lure_rows) | set(cand.meta.get("verify_rows", []))
    for ep_id in allowed_eps:
        if ep_id in index.episodes:
            allowed_rows |= {r.row_id for r in index.episode_rows(ep_id)}
            allowed_rows |= set(index.episodes[ep_id].evidence_ids)
    return allowed_eps, allowed_rows


def _annotation_values(slot: str, eps, index: MemoryIndex) -> set[str]:
    vals = set()
    for ep_id in eps:
        ep = index.episodes.get(ep_id)
        if ep is not None and slot in ANNOTATION_SLOTS:
            vals |= {canon(v, index.synonyms) for v in ep.slot_values(slot) if v}
    return vals


def is_primary(f, cand: Candidate, synonyms: dict[str, str] | None = None) -> bool:
    """
    The primary fact is the one in the primary slot that carries the mined true value.
    Other facts in the same slot (e.g. extra objects in T1) are secondary.
    """
    if f.slot != cand.primary_slot:
        return False
    if f.slot in DERIVED_SLOTS:
        # Derived answers (completion_status, day, temporal_order, knowledge_source,
        # causal_link) are labels decided by the miner; the model phrases them its own
        # way ("no record found", "day 2"), so the slot identifies the primary fact.
        return True
    if canon(f.value, synonyms) == canon(cand.true_value, synonyms):
        return True
    return bool(_stems(f.value) & _stems(cand.true_value))


def assess_fact(f, cand: Candidate, index: MemoryIndex, allowed_eps, allowed_rows) -> str | None:
    """
    Sets f.source deterministically and returns an error string, or None if the
    fact is supported.
    """
    bad = [e for e in f.evidence_ids if e not in index.rows]
    if bad:
        return f"{f.fact_id}: unknown evidence ids {bad}"
    outside = [e for e in f.evidence_ids if e not in allowed_rows]
    if outside:
        return f"{f.fact_id}: evidence outside target/lure episodes {outside}"
    f.source = "caption"
    if f.slot not in CONCRETE_SLOTS:
        return None
    if f.slot != "activity" and ("," in f.value or " and " in f" {f.value} " or ";" in f.value):
        return f"{f.fact_id}: value '{f.value}' lists several values; use one fact per value"
    rows = [index.rows[e] for e in f.evidence_ids]
    if lexically_supported(f.value, [r.text for r in rows], index.synonyms, head_noun=f.slot in HEAD_NOUN_SLOTS):
        return None
    if f.slot == "people":
        v = index.person(f.value)
        if v == canon(cand.participant) or any(r.speaker and index.person(r.speaker) == v for r in rows):
            return None
    if f.slot in ANNOTATION_SLOTS and canon(f.value, index.synonyms) in _annotation_values(f.slot, allowed_eps, index):
        f.source = "event_annotation"
        return None
    return f"{f.fact_id}: value '{f.value}' not lexically supported by cited rows"


def prune_secondary_facts(gt: GroundTruth, cand: Candidate, index: MemoryIndex) -> list[dict]:
    """
    Drop SECONDARY facts (every fact except the primary one, see `is_primary`) that fail their support check and
    return what was dropped, so it is recorded with the sample (not silent). A
    failing PRIMARY fact is kept, so `check_ground_truth` rejects the sample.
    """
    allowed_eps, allowed_rows = _allowed(cand, index)
    kept, dropped = [], []
    for f in gt.facts:
        err = assess_fact(f, cand, index, allowed_eps, allowed_rows)
        if err and not is_primary(f, cand, index.synonyms):
            dropped.append({"fact_id": f.fact_id, "slot": f.slot, "value": f.value, "reason": err})
        else:
            kept.append(f)
    gt.facts = kept
    return dropped


# Errors that cannot change between attempts (the rows are the same every time),
# so retrying only wastes calls.
FATAL_PREFIXES = ("LLM says the rows contradict the miner's true value",)


def is_fatal(errs: list[str]) -> bool:
    return any(e.startswith(FATAL_PREFIXES) for e in errs)


def check_ground_truth(gt: GroundTruth, cand: Candidate, index: MemoryIndex, xv=None) -> list[str]:
    errs: list[str] = []
    allowed_eps, allowed_rows = _allowed(cand, index)

    for f in gt.facts:
        err = assess_fact(f, cand, index, allowed_eps, allowed_rows)
        if err:
            errs.append(err)

    primary = [f for f in gt.facts if is_primary(f, cand, index.synonyms)]
    if not primary:
        errs.append(f"no fact for primary slot '{cand.primary_slot}' with value '{cand.true_value}'")
    elif not any(f.visible_to_user for f in primary):
        # The assistant can only show the user their OWN recordings; a correction
        # whose only evidence is someone else's view cannot be gold.
        errs.append("primary fact is only visible in another person's recording")
    elif not any(f.source == "caption" for f in primary):
        errs.append("primary fact is only supported by the event annotation, not by caption text")
    if gt.task_specific.get("miner_disagrees"):
        errs.append("LLM says the rows contradict the miner's true value")
    return errs


###############################################################################
# Stage 2
###############################################################################


def check_false_memory(fm: FalseMemory, gt: GroundTruth, cand: Candidate,
                       lure_pool_values: set[str], min_gist_overlap: float) -> list[str]:
    errs: list[str] = []
    cond = fm.belief_condition
    true_values = {f.value.lower() for f in gt.facts}

    if cond == BeliefCondition.CORRECT:
        if fm.edits:
            errs.append("correct-belief sample must have no edits")
        return errs
    if cond == BeliefCondition.INCOMPLETE:
        if fm.edits:
            errs.append("incomplete-belief sample must have no edits")
        if not fm.omitted_slots:
            errs.append("incomplete-belief sample must omit at least one slot")
        return errs

    primaries = [e for e in fm.edits if e.role == "primary"]
    if len(primaries) != 1:
        errs.append(f"expected exactly 1 primary edit, got {len(primaries)}")
    else:
        p = primaries[0]
        if p.slot != cand.primary_slot or p.false.lower() != cand.lure_value.lower():
            errs.append("primary edit differs from the mined candidate")
        if p.error_type != PRIMARY_ERROR[cand.task]:
            errs.append(f"primary error_type must be {PRIMARY_ERROR[cand.task]}")
    if not 1 <= len(fm.edits) <= 4:
        errs.append(f"{len(fm.edits)} edits (allowed 1-4; 2-4 preferred)")

    caption_values = {f.value.lower() for f in gt.facts if f.source == "caption"}
    for e in fm.edits:
        if e.true.lower() not in true_values:
            errs.append(f"edit {e.slot}: true value '{e.true}' is not a ground-truth fact")
        elif e.true.lower() not in caption_values:
            errs.append(f"edit {e.slot}: '{e.true}' is only supported by the event annotation; "
                        "it cannot be corrected with evidence the user can see")
        if e.role == "secondary":
            if e.error_type not in SECONDARY_ERRORS:
                errs.append(f"secondary error_type '{e.error_type}' not allowed")
            if e.false.lower() not in lure_pool_values:
                errs.append(f"secondary false value '{e.false}' not from a lure record")
        if e.slot in CONCRETE_SLOTS:
            text = fm.user_memory + " " + fm.user_question
            if e.slot == "activity":
                # Activities are phrases: require their words, not the exact wording
                # ("looking for dipping sauces" ~ "looked for the dipping sauce").
                if coverage(e.false, [text]) < 0.5:
                    errs.append(f"memory does not contain false value '{e.false}'")
                if coverage(e.true, [text]) >= 0.6 and coverage(e.false, [text]) < coverage(e.true, [text]):
                    errs.append(f"memory leaks true value '{e.true}'")
            else:
                if not contains_value(text, e.false) and not lexically_supported(e.false, [text], head_noun=True):
                    errs.append(f"memory does not contain false value '{e.false}'")
                if contains_value(text, e.true):
                    errs.append(f"memory leaks true value '{e.true}'")

    gist_hits = sum(1 for g in fm.preserved_gist if set(tokens(g)) & set(tokens(fm.user_memory)))
    if fm.preserved_gist and gist_hits / len(fm.preserved_gist) < min_gist_overlap:
        errs.append("preserved gist not reflected in the memory")
    return errs


###############################################################################
# Stage 3
###############################################################################


MAX_CUE_WORDS = 5


def drop_repeated_steps(plan: DialoguePlan) -> list[int]:
    """
    Repair instead of reject: remove a step that repeats the previous step's anchor
    or evidence without correcting anything (it adds nothing to the dialogue), then
    renumber. Returns the original numbers of the removed steps.
    """
    kept, removed = [], []
    for s in plan.steps:
        prev = kept[-1] if kept else None
        if prev is not None and not s.corrected_attributes and s.intent != "summarize":
            same_anchor = s.selected_anchor and s.selected_anchor == prev.selected_anchor
            same_ev = s.evidence_ids and set(s.evidence_ids) == set(prev.evidence_ids)
            if same_anchor or same_ev:
                removed.append(s.step)
                continue
        kept.append(s)
    for i, s in enumerate(kept, 1):
        s.step = i
    plan.steps = kept
    return removed


def fix_evidence_types(plan: DialoguePlan, index: MemoryIndex, task: str = "") -> int:
    """
    Repair the evidence_type label from the cited rows: speech rows are
    utterance_quote (reported_speech for T4 reports), caption rows caption_detail.
    Timestamp / landmark / absence labels are kept. Returns how many were changed.
    """
    changed = 0
    for s in plan.steps:
        kinds = {index.rows[i].kind for i in s.evidence_ids if i in index.rows}
        if not kinds or s.evidence_type in ("timestamp", "landmark_event", "absence_of_record"):
            continue
        if kinds == {"utterance"}:
            want = "reported_speech" if task.startswith("T4") else "utterance_quote"
        elif kinds == {"caption"}:
            want = "caption_detail"
        else:
            continue
        if s.evidence_type != want:
            s.evidence_type = want
            changed += 1
    return changed


def repair_plan_leaks(plan: DialoguePlan, targets: list[dict], index: MemoryIndex, max_corr: int) -> list[str]:
    """
    Repair (instead of rejecting) plans that show an answer before correcting it; the
    target episode's rows naturally mention the answer, and the model kept failing:
      * step 1 cites no evidence: its evidence/anchor are cleared;
      * retrieval cues that contain a pending true value are removed;
      * a later non-correcting step whose rows state a pending value BECOMES the
        correction of that slot (when its rows are allowed evidence for it and the
        per-turn limit allows); the later correction of that slot is removed;
      * otherwise the revealing rows are dropped from that step.
    Returns notes describing what was changed.
    """
    notes = []
    allowed = {t["slot"]: set(t.get("true_evidence") or []) for t in targets}
    pending = {t["slot"]: t["true"] for t in targets if t.get("true") and t["slot"] in CONCRETE_SLOTS}
    if plan.steps and (plan.steps[0].evidence_ids or plan.steps[0].selected_anchor):
        plan.steps[0].evidence_ids, plan.steps[0].selected_anchor = [], None
        notes.append("step 1: evidence/anchor cleared")
    for idx_s, s in enumerate(plan.steps):
        for slot in s.corrected_attributes:
            pending.pop(slot, None)
        kept = [c for c in s.retrieval_cues if not any(contains_value(c, v) for v in pending.values())]
        if len(kept) != len(s.retrieval_cues):
            notes.append(f"step {s.step}: removed cues revealing an answer")
            s.retrieval_cues = kept
        ids = list(s.evidence_ids) + ([s.selected_anchor] if s.selected_anchor else [])
        for slot, true in list(pending.items()):
            bad = [i for i in ids if i in index.rows and contains_value(index.rows[i].text, true)]
            if not bad:
                continue
            if idx_s > 0 and set(bad) & allowed.get(slot, set()) and len(s.corrected_attributes) < max_corr \
                    and s.intent != "summarize":
                s.corrected_attributes.append(slot)
                s.intent = "correct"
                for later in plan.steps[idx_s + 1:]:
                    if slot in later.corrected_attributes:
                        later.corrected_attributes.remove(slot)
                        if not later.corrected_attributes and later.intent == "correct":
                            later.intent = "present_evidence"
                pending.pop(slot)
                notes.append(f"step {s.step}: now corrects '{slot}' (its rows state the answer)")
            else:
                s.evidence_ids = [i for i in s.evidence_ids if i not in bad]
                if s.selected_anchor in bad:
                    s.selected_anchor = None
                notes.append(f"step {s.step}: removed rows revealing '{true}'")
    return notes


def check_plan_evidence(plan: DialoguePlan, targets: list[dict], index: MemoryIndex) -> list[str]:
    """
    A step's anchor/evidence rows are shown to the user, so they must not state a
    true value before the step that corrects it (e.g. quoting "I scroll on my
    phone" while 'phone' is still to be corrected). Such plans made the assistant
    leak the value in every conversation attempt.
    """
    errs = []
    pending = {t["slot"]: t["true"] for t in targets if t.get("true") and t["slot"] in CONCRETE_SLOTS}
    for s in plan.steps:
        for slot in s.corrected_attributes:
            pending.pop(slot, None)
        ids = list(s.evidence_ids) + ([s.selected_anchor] if s.selected_anchor else [])
        text = " ".join(index.rows[i].text for i in ids if i in index.rows)
        for slot, true in pending.items():
            if contains_value(text, true):
                errs.append(f"step {s.step}: its anchor/evidence rows state '{true}' before '{slot}' "
                            "is corrected; pick a different anchor")
    return errs


def check_plan(plan: DialoguePlan, fm: FalseMemory, targets: list[dict], max_corr: int) -> list[str]:
    """`targets` = edited slots + omitted slots to fill (see controller.correction_targets)."""
    errs: list[str] = []
    if not plan.steps:
        return ["empty plan"]
    first = plan.steps[0]
    if first.intent != "retrieve_and_clarify" or first.corrected_attributes:
        errs.append("step 1 must retrieve_and_clarify without corrections")
    if plan.steps[-1].intent != "summarize":
        errs.append("plan must end with summarize")

    if first.evidence_ids:
        errs.append("step 1 must not cite evidence yet (evidence_ids must be [])")
    for s in plan.steps:
        for cue in s.retrieval_cues:
            if len(cue.split()) > MAX_CUE_WORDS or cue.strip().endswith("?"):
                errs.append(f"step {s.step}: retrieval cue {cue[:40]!r} must be 1-{MAX_CUE_WORDS} "
                            "search keywords, not a sentence or question")
                break
    for prev, cur in zip(plan.steps, plan.steps[1:]):
        if cur.corrected_attributes:
            continue
        same_anchor = cur.selected_anchor and cur.selected_anchor == prev.selected_anchor
        same_ev = cur.evidence_ids and set(cur.evidence_ids) == set(prev.evidence_ids)
        if same_anchor or same_ev:
            errs.append(f"step {cur.step}: repeats the anchor/evidence of step {prev.step} without correcting")
    first_corr = next((i for i, s in enumerate(plan.steps) if s.corrected_attributes), None)
    if first_corr is not None and not any(
            s.intent in ("retrieve_and_clarify", "suggest_cue") for s in plan.steps[:first_corr]):
        errs.append("ask the person something (retrieve_and_clarify/suggest_cue) before the first correction")

    # Cues are hidden supervision: they must not carry a true value before it is corrected.
    pending = {t["slot"]: t["true"] for t in targets
               if t.get("true") and t["slot"] in CONCRETE_SLOTS}
    for s in plan.steps:
        for slot in s.corrected_attributes:      # the correcting step may name its value
            pending.pop(slot, None)
        cue_text = " ".join(s.retrieval_cues)
        for slot, true in pending.items():
            if contains_value(cue_text, true):
                errs.append(f"step {s.step}: retrieval cues reveal '{true}' before it is corrected")

    allowed = {t["slot"]: set(t["true_evidence"]) for t in targets}
    seen: list[str] = []
    for s in plan.steps:
        if len(s.corrected_attributes) > max_corr:
            errs.append(f"step {s.step}: {len(s.corrected_attributes)} corrections > {max_corr}")
        for slot in s.corrected_attributes:
            if slot not in allowed:
                errs.append(f"step {s.step}: corrects '{slot}' which is not an edited slot")
                continue
            if not set(s.evidence_ids) & allowed[slot]:
                errs.append(f"step {s.step}: correction of '{slot}' lacks allowed evidence")
            seen.append(slot)
    if sorted(seen) != sorted(allowed):
        errs.append(f"corrected {sorted(seen)} but edits are {sorted(allowed)}")
    if fm.belief_condition == BeliefCondition.CORRECT and seen:
        errs.append("correct-belief plan must not correct anything")
    return errs


###############################################################################
# Stage 4 (per assistant turn)
###############################################################################


def leaked_values(text: str, forbidden: list[str]) -> list[str]:
    return [v for v in forbidden if contains_value(text, v)]


def near_duplicate(a: str, b: str, threshold: float = 0.65) -> bool:
    """Two turns are near-duplicates if their content words overlap >= threshold (Jaccard)."""
    ta, tb = set(tokens(a)), set(tokens(b))
    if len(ta) < 3 or len(tb) < 3:
        return a.strip().lower() == b.strip().lower() and bool(a.strip())
    return len(ta & tb) / len(ta | tb) >= threshold


def repeated_turns(turns: list[dict], threshold: float = 0.65) -> list[str]:
    """Descriptions of turns that near-duplicate an earlier turn by the same role."""
    out = []
    for i, t in enumerate(turns):
        for j in range(i):
            u = turns[j]
            if u["role"] == t["role"] and near_duplicate(t.get("text", ""), u.get("text", ""), threshold):
                out.append(f"{t['role']} turn {i + 1} repeats turn {j + 1}")
                break
    return out
