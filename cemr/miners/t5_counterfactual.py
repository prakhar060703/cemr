"""
T5 Counterfactual blame.

Action episode A (a decision or state change) followed within a window by an
outcome episode O that shares an object or person. The user asserts a strong
counterfactual ("if I hadn't done X, Y wouldn't have happened"). The model
must separate what was OBSERVED (the sequence A -> O) from what can or
cannot be INFERRED (whether A was necessary for O).

Stage 1 labels each downstream fact as observed / inferable / not_inferable
(TANGLE-style evidence boundary); that label set is the gold.
"""

from __future__ import annotations

from ..schemas import Candidate, TaskId
from .base import BaseMiner


class CounterfactualMiner(BaseMiner):
    task = TaskId.T5

    def _mine(self) -> list[Candidate]:
        out: list[Candidate] = []
        eps = self.eps
        window = self.m.t5_max_gap_hours * 3600
        for i, a in enumerate(eps):
            if not (a.decisions or a.state_changes):
                continue
            if self.narrator not in a.people:
                continue  # blame/regret is about the narrator's own action
            for o in eps[i + 1:]:
                if o.abs_start - a.abs_end > window:
                    break
                if not self.same_split(a, o):
                    continue
                shared_obj = set(a.objects) & set(o.objects)
                if not shared_obj or not o.state_changes:
                    continue
                # The action must be stated in the action episode's own rows and the
                # outcome in the outcome episode's rows (annotations can paraphrase or
                # borrow from neighbouring episodes).
                action = next((x for x in a.decisions + a.state_changes if self.stated(x, a, "action")), None)
                outcome = next((x for x in o.state_changes if self.stated(x, o, "outcome")), None)
                if action is None or outcome is None:
                    continue
                out.append(Candidate(
                    candidate_id=self.cid(a.episode_id, o.episode_id),
                    task=self.task,
                    split=self.split_of(a),
                    target_episode=a.episode_id,
                    lure_episodes=[o.episode_id],
                    primary_slot="causal_link",
                    true_value="sequence observed; necessity not established",
                    lure_value=f"'{action}' caused '{outcome}'",
                    subtype="necessity_claim",
                    query_time=o.abs_end + 1800,
                    meta={"action": action, "outcome": outcome,
                          "shared_objects": sorted(shared_obj)},
                ))
        return out
