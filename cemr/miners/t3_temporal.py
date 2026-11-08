"""
T3 Temporal displacement.

Target episode A and a salient landmark episode L. The user's memory moves
A across L (before <-> after), or onto another day. Evidence is contextual
(objects/state present) plus timestamps, which mirrors how people date
memories by reconstruction.
"""

from __future__ import annotations

from ..schemas import Candidate, Episode, TaskId
from .base import BaseMiner


def _salience(ep: Episode) -> float:
    return len(ep.people) + 0.5 * len(ep.state_changes) + (ep.abs_end - ep.abs_start) / 600


class TemporalMiner(BaseMiner):
    task = TaskId.T3

    def _mine(self) -> list[Candidate]:
        out: list[Candidate] = []
        eps = self.eps
        if not eps:
            return out
        # Landmarks = top 20% most salient episodes of each day.
        by_day: dict[int, list[Episode]] = {}
        for e in eps:
            by_day.setdefault(e.day, []).append(e)
        landmarks = []
        for day_eps in by_day.values():
            k = max(1, len(day_eps) // 5)
            # Events seen by many wearers (meetings, meals) make the best landmarks.
            landmarks += sorted(day_eps, key=lambda e: _salience(e) + self.xv.n_views(e.episode_id),
                                reverse=True)[:k]

        gap = self.m.t3_min_gap_minutes * 60
        for lm in landmarks:
            for a in eps:
                if a.episode_id == lm.episode_id or not self.same_split(a, lm):
                    continue
                if abs(a.abs_start - lm.abs_start) < gap or a.day not in (lm.day, lm.day + 1, lm.day - 1):
                    continue
                if a.day == lm.day:
                    true_rel = "after" if a.abs_start > lm.abs_end else "before"
                    lure_rel = "before" if true_rel == "after" else "after"
                    slot, tv, lv, sub = "temporal_order", f"{true_rel} {lm.episode_id}", \
                        f"{lure_rel} {lm.episode_id}", "order_flip"
                else:
                    slot, tv, lv, sub = "day", f"day {a.day}", f"day {lm.day}", "day_shift"
                out.append(Candidate(
                    candidate_id=self.cid(a.episode_id, lm.episode_id, sub),
                    task=self.task,
                    split=self.split_of(a),
                    target_episode=a.episode_id,
                    lure_episodes=[lm.episode_id],
                    primary_slot=slot,
                    true_value=tv,
                    lure_value=lv,
                    subtype=sub,
                    query_time=max(a.abs_end, lm.abs_end) + 3600,
                    meta={"landmark_summary": lm.summary},
                ))
        return out
