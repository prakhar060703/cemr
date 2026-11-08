"""
T2 Intention vs. action (reality / output monitoring).

Plan mentioned in episode P; search later episodes for a matching execution.
Subtypes:
  * never_executed        -- no execution found; user believes it was done
                             (gold wording: "no record", never "didn't happen")
  * executed_by_other     -- done by someone else; user believes they did it
  * done_believed_pending -- done (by narrator); user believes it is still pending
"""

from __future__ import annotations

from ..schemas import Candidate, Episode, TaskId
from ..utils import tokens
from .base import BaseMiner


def _overlap(plan_toks: list[str], ep: Episode) -> float:
    exec_toks = set(tokens(" ".join(ep.activity + ep.state_changes + ep.objects)))
    p = set(plan_toks)
    return len(p & exec_toks) / len(p) if p else 0.0


class IntentionMiner(BaseMiner):
    task = TaskId.T2

    def _narrator_acted(self, ep: Episode, plan_toks: list[str]) -> bool:
        p = set(plan_toks)
        for r in self.index.episode_rows(ep.episode_id):
            if r.kind != "caption":
                continue
            words = r.text.lower().split()
            if words and words[0] in ("i", "i'm", "i've") and p & set(tokens(r.text)):
                return True
        return False

    def _own_rows_match(self, plan_toks: list[str], start: float, end: float) -> bool:
        p = set(plan_toks)
        for r in self.xv.first_person_rows(self.narrator, start, end):
            if len(p & set(tokens(r.text))) / len(p) >= self.m.t2_exec_overlap_threshold:
                return True
        return False

    def _executed_elsewhere(self, plan_toks: list[str], start: float, end: float) -> tuple[list[str], list[str]]:
        """Other wearers whose own first-person captions show them doing the planned action."""
        who, rows = set(), []
        p = set(plan_toks)
        for other in self.index.participants:
            if other == self.narrator:
                continue
            for r in self.xv.first_person_rows(other, start, end):
                if len(p & set(tokens(r.text))) / len(p) >= self.m.t2_exec_overlap_threshold:
                    who.add(other)
                    rows.append(r.row_id)
        return sorted(who), rows[:5]

    def _mine(self) -> list[Candidate]:
        out: list[Candidate] = []
        eps = self.eps
        window = self.m.t2_search_window_hours * 3600
        for i, p in enumerate(eps):
            for plan in p.plans + [d for d in p.decisions if len(tokens(d)) >= 2]:
                ptoks = tokens(plan)
                if len(ptoks) < 2:
                    continue
                # The intention must be stated in the narrator's own rows of the plan
                # episode (the event annotation may paraphrase or invent it).
                if not self.stated(plan, p, "plans"):
                    continue
                # Decisions carried out inside the same episode are not open intentions.
                if _overlap(ptoks, p) >= self.m.t2_exec_overlap_threshold:
                    continue
                later = [e for e in eps[i + 1:] if e.abs_start - p.abs_end <= window and self.same_split(p, e)]
                best = max(later, key=lambda e: _overlap(ptoks, e), default=None)
                best_score = _overlap(ptoks, best) if best else 0.0
                base = dict(task=self.task, split=self.split_of(p), primary_slot="completion_status",
                            meta={"plan_text": plan, "plan_episode": p.episode_id,
                                  "exec_score": round(best_score, 3)})

                if best is None or best_score < self.m.t2_exec_overlap_threshold:
                    horizon = (later[-1].abs_end if later else p.abs_end) + 60
                    # Events cover only part of each hour, so "no episode matches" is not
                    # "no record": also search the narrator's OWN caption rows in the
                    # window. If any row looks like the action, "no record" is unsafe.
                    if self._own_rows_match(ptoks, p.abs_end, horizon):
                        continue
                    # Not in the user's recordings. Did anyone else record doing it?
                    who, rows = self._executed_elsewhere(ptoks, p.abs_end, horizon)
                    base["meta"].update({"executed_by": who, "verify_rows": rows})
                    out.append(Candidate(
                        candidate_id=self.cid(p.episode_id, "never", plan[:40]),
                        target_episode=p.episode_id,
                        # The USER-FACING truth is the same either way: no record in
                        # the user's own memory. World truth is kept in meta.
                        true_value="no execution recorded",
                        lure_value="completed",
                        lure_episodes=[],
                        subtype="executed_unseen" if who else "never_executed",
                        query_time=horizon, **base))
                    continue

                # Who acted? `people` always contains the narrator (camera wearer), so
                # read it from the rows: a first-person caption mentioning the action
                # means the narrator did it.
                narrator_did = self._narrator_acted(best, ptoks)
                actors = set(best.people) - {self.narrator}
                if not narrator_did and actors:
                    # Verify the actor from THEIR OWN first-person captions (other views).
                    who, rows = self._executed_elsewhere(ptoks, best.abs_start, best.abs_end)
                    if not who:
                        # Only the event annotation says someone did it (often just a plan
                        # or agreement in the rows). An unverified executor is not gold.
                        continue
                    base["meta"].update({"actor_verified": True, "verify_rows": rows})
                    out.append(Candidate(
                        candidate_id=self.cid(p.episode_id, best.episode_id, "other"),
                        target_episode=best.episode_id,
                        lure_episodes=[p.episode_id],
                        true_value=f"done by {', '.join(who)}",
                        lure_value="done by me",
                        subtype="executed_by_other",
                        query_time=best.abs_end + 1800, **base))
                elif narrator_did:
                    out.append(Candidate(
                        candidate_id=self.cid(p.episode_id, best.episode_id, "pending"),
                        target_episode=best.episode_id,
                        lure_episodes=[p.episode_id],
                        true_value="completed",
                        lure_value="still pending",
                        subtype="done_believed_pending",
                        query_time=best.abs_end + 1800, **base))
        return out
