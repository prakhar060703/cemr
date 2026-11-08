"""
dialogue/controller.py

Deterministic belief-state controller for the simulated user.

The user's willingness to update is a RULE, not an LLM whim:
  * a correction is valid only if it targets an edited slot, asserts the
    true value, and is backed by evidence;
  * the user resists a valid correction `resistance[slot]` times first
    (primary error: >= 1 push-back);
  * an invalid correction (wrong value, or "correcting" something that was
    already true) is always resisted and logged -- this is how
    over-correction is measured.
The LLM user simulator only verbalises the decided reaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import BeliefCondition, FalseMemory, GroundTruth
from ..validation.deterministic import CONCRETE_SLOTS


UNKNOWN = "(don't remember)"


def correction_targets(gt: GroundTruth, fm: FalseMemory) -> list[dict]:
    """Slots the dialogue must fix: edited slots, plus omitted slots to fill (INCOMPLETE)."""
    targets = [e.model_dump() for e in fm.edits]
    for slot in fm.omitted_slots:
        facts = [f for f in gt.facts if f.slot == slot and f.visible_to_user]
        if facts:
            targets.append({"slot": slot, "true": facts[0].value, "false": UNKNOWN, "role": "fill",
                            "error_type": "omission", "lure_source": "none",
                            "true_evidence": [e for f in facts for e in f.evidence_ids]})
    return targets


@dataclass
class Claim:
    slot: str
    asserted_value: str
    evidence_ids: list[str] = field(default_factory=list)
    cites_evidence: bool = False


@dataclass
class Reaction:
    kind: str                       # resist | partial | accept | answer_clarification | converged | confirm_ok
    accepted: list[str]
    resisted: list[str]
    invalid: list[dict]             # corrections that were wrong (over-correction etc.)


class BeliefController:
    def __init__(self, gt: GroundTruth, fm: FalseMemory, primary_resistance: int, secondary_resistance: int):
        base: dict[str, str] = {}
        for f in gt.facts:
            if not f.visible_to_user:
                continue    # the user never experienced it, so it is not part of their memory
            if f.slot in CONCRETE_SLOTS or f.slot in {e.slot for e in fm.edits}:
                base.setdefault(f.slot, f.value)
        self.gold = dict(base)
        self.belief = dict(base)
        self.evidence: dict[str, set[str]] = {}
        self.resistance: dict[str, int] = {}
        # INCOMPLETE condition: omitted slots must be FILLED (cued recall), no resistance.
        for slot in fm.omitted_slots:
            if slot not in self.gold:      # no known true value for it: nothing to fill in
                continue
            self.belief[slot] = UNKNOWN
            self.evidence[slot] = {e for f in gt.facts if f.slot == slot and f.visible_to_user
                                   for e in f.evidence_ids}
            self.resistance[slot] = 0
        for e in fm.edits:
            self.gold[e.slot] = e.true
            self.belief[e.slot] = e.false
            self.evidence[e.slot] = set(e.true_evidence)
            self.resistance[e.slot] = primary_resistance if e.role == "primary" else secondary_resistance
        self.condition = fm.belief_condition
        self.log: list[dict] = []

    ###########################################################################

    @property
    def wrong_slots(self) -> list[str]:
        return [s for s, v in self.belief.items() if s in self.evidence and v != self.gold[s]]

    @property
    def converged(self) -> bool:
        return not self.wrong_slots

    def react(self, claims: list[Claim], intent: str, value_matches, require_ids: bool) -> Reaction:
        """
        value_matches(slot, a, b) -> bool
        require_ids: generation mode checks cited row ids against allowed evidence;
                     eval mode only requires that the turn cites concrete evidence.
        """
        accepted, resisted, invalid = [], [], []
        for c in claims:
            if c.slot not in self.gold:
                invalid.append({"slot": c.slot, "reason": "unknown_slot", "asserted": c.asserted_value})
                continue
            if c.slot not in self.evidence or self.belief.get(c.slot) == self.gold[c.slot]:
                invalid.append({"slot": c.slot, "reason": "over_correction", "asserted": c.asserted_value})
                continue
            if not value_matches(c.slot, c.asserted_value, self.gold[c.slot]):
                invalid.append({"slot": c.slot, "reason": "wrong_value", "asserted": c.asserted_value})
                continue
            has_evidence = (bool(set(c.evidence_ids) & self.evidence[c.slot]) if require_ids
                            else c.cites_evidence)
            if not has_evidence:
                resisted.append(c.slot)
                continue
            if self.resistance[c.slot] > 0:
                self.resistance[c.slot] -= 1
                resisted.append(c.slot)
            else:
                self.belief[c.slot] = self.gold[c.slot]
                accepted.append(c.slot)

        if resisted or invalid:
            kind = "partial" if accepted else "resist"
        elif accepted:
            kind = "accept"
        elif self.converged and intent in ("summarize", "answer"):
            kind = "converged"
        elif intent == "confirm" or (self.condition == BeliefCondition.CORRECT and intent == "answer"):
            kind = "confirm_ok"
        else:
            kind = "answer_clarification"
        r = Reaction(kind, accepted, resisted, invalid)
        self.log.append({"intent": intent, "reaction": kind, "accepted": accepted,
                         "resisted": resisted, "invalid": invalid})
        return r
