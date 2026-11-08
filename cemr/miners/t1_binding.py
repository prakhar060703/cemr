"""
T1 Source binding / event blending.

Target episode A, lure episode B with the same gist (shared location or
activity, >= 1 shared person) that differs in a slot. The user's memory
binds B's value into A (memory conjunction error).
"""

from __future__ import annotations

from itertools import combinations

from ..schemas import Candidate, TaskId
from ..utils import jaccard, tokens
from ..validation.deterministic import is_generic
from .base import BaseMiner

BINDABLE_SLOTS = ("objects", "people", "location", "activity")


class BindingMiner(BaseMiner):
    task = TaskId.T1

    def _mine(self) -> list[Candidate]:
        out: list[Candidate] = []
        eps = self.eps
        max_gap = self.m.t1_max_gap_hours * 3600
        for a, b in combinations(eps, 2):
            if abs(b.abs_start - a.abs_start) > max_gap or not self.same_split(a, b):
                continue
            shared_people = (set(a.people) & set(b.people)) - {self.narrator}
            if len(shared_people) < self.m.t1_min_shared_people:
                continue
            same_place = a.location and a.location == b.location
            act_overlap = jaccard(tokens(" ".join(a.activity)), tokens(" ".join(b.activity)))
            if not (same_place or act_overlap >= 0.2):
                continue
            # Try both directions: A is the target, B supplies the lure, and vice versa.
            for tgt, lure in ((a, b), (b, a)):
                for slot in BINDABLE_SLOTS:
                    t_vals = set(tgt.slot_values(slot)) - {self.narrator, ""}
                    l_vals = set(lure.slot_values(slot)) - {self.narrator, ""}
                    only_t, only_l = sorted(t_vals - l_vals), sorted(l_vals - t_vals)
                    # Both values must be specific ("detergent", not "food") and stated in
                    # the captions of their own episode: the true value so ground truth can
                    # cite it, the lure so the assistant can show where the blend came from.
                    only_t = [v for v in only_t if not is_generic(v) and self.stated(v, tgt, slot)]
                    only_l = [v for v in only_l if not is_generic(v) and self.stated(v, lure, slot)]
                    if not only_t or not only_l:
                        continue
                    out.append(Candidate(
                        candidate_id=self.cid(tgt.episode_id, lure.episode_id, slot),
                        task=self.task,
                        split=self.split_of(tgt),
                        target_episode=tgt.episode_id,
                        lure_episodes=[lure.episode_id],
                        primary_slot=slot,
                        true_value=only_t[0],
                        lure_value=only_l[0],
                        subtype=f"blend_{slot}",
                        query_time=max(tgt.abs_end, lure.abs_end) + 3600,
                        meta={"shared_people": sorted(shared_people),
                              "same_place": bool(same_place)},
                    ))
                    break  # one slot per (target, lure) direction
        return out
