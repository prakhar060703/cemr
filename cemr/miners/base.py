"""
miners/base.py

Common machinery for Stage-0 candidate miners.

Design rules
  * Every false value is MINED from another real record (a lure source),
    never invented by the LLM. Miners decide the task label.
  * Mining is PER PARTICIPANT: each wearer is a separate user whose memory
    is their own recording. Lures come from the user's own episodes
    (memory errors borrow from one's own experience) -- except T4, where
    the source of the error is what another person said.
  * Other wearers' views may be used to VERIFY gold labels (see crossview).
  * One real event seen by several wearers yields at most
    `max_per_cluster` candidates per task (near-duplicate control).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from ..config import Config
from ..crossview import CrossView
from ..index import MemoryIndex
from ..schemas import BeliefCondition, Candidate, Episode, TaskId
from ..utils import stable_rng
from ..validation.deterministic import HEAD_NOUN_SLOTS, coverage, lexically_supported


class BaseMiner(ABC):
    task: TaskId

    def __init__(self, index: MemoryIndex, cfg: Config, crossview: CrossView | None = None):
        self.index = index
        self.cfg = cfg
        self.m = cfg.mining
        self.xv = crossview or CrossView.from_index(index, cfg.mining.cluster_min_overlap)
        self.narrator: str = ""          # set per participant in mine()
        self.eps: list[Episode] = []     # the current participant's own episodes, time-sorted

    @abstractmethod
    def _mine(self) -> list[Candidate]:
        """Mine candidates for self.narrator using self.eps (own memory only)."""

    def mine(self) -> list[Candidate]:
        cands: list[Candidate] = []
        for p in self.index.participants:
            self.narrator = p
            self.eps = self.index.episodes_sorted(p)
            for c in self._mine():
                if c.split is None:          # held-out person, non-test day
                    continue
                c.participant = p
                c.meta.setdefault("n_views", self.xv.n_views(c.target_episode))
                cands.append(c)
        cands = self._cap_per_cluster(cands)
        for c in cands:
            self._assign_query_time(c)
        # Deterministic, order-independent subsample.
        cands.sort(key=lambda c: stable_rng(c.candidate_id).random())
        cands = cands[: self.m.max_candidates_per_task]
        for c in cands:
            c.meta["belief_condition"] = self._assign_condition(c).value
        return cands

    ###########################################################################

    def _cap_per_cluster(self, cands: list[Candidate]) -> list[Candidate]:
        by_cluster: dict[str, list[Candidate]] = defaultdict(list)
        for c in cands:
            cid = self.index.episodes[c.target_episode].cluster_id or c.target_episode
            by_cluster[cid].append(c)
        kept = []
        for cid, group in by_cluster.items():
            # Prefer spreading across participants, then deterministic order.
            group.sort(key=lambda c: stable_rng("cluster", c.candidate_id).random())
            seen_p, first, rest = set(), [], []
            for c in group:
                (rest if c.participant in seen_p else first).append(c)
                seen_p.add(c.participant)
            kept += (first + rest)[: self.m.max_per_cluster]
        return kept

    ###########################################################################
    # Query time: when the user asks
    ###########################################################################

    NO_DELAY_SUBTYPES = ("never_executed", "executed_unseen")

    def _assign_query_time(self, c: Candidate) -> None:
        """
        c.query_time from the miner is the EARLIEST valid time. Move it later by a
        delay bucket chosen deterministically per candidate, clipped to the
        participant's last recording within the candidate's split.
        """
        earliest = c.query_time
        c.meta["query_time_earliest"] = earliest
        if c.subtype in self.NO_DELAY_SUBTYPES:
            c.meta["delay_bucket"] = self._bucket(earliest, c)
            return
        own = [e for e in self.index.episodes_sorted(c.participant)
               if self.index.split_of(e.day, c.participant, self.cfg.data) == c.split]
        latest = max((e.abs_end for e in own), default=earliest) + 1800
        rng = stable_rng("query_time", c.candidate_id)
        weights = self.m.query_delay_weights or {"same_day": 1.0}
        bucket = rng.choices(list(weights), weights=list(weights.values()))[0]
        day0 = int(earliest // 86400)
        if bucket == "same_day":
            # Late events: stay on the same day but not at an artificial 23:59.
            t = min(earliest + rng.uniform(1, 3) * 3600, max(earliest, day0 * 86400 + 23.5 * 3600))
        else:
            days = 1 if bucket == "next_day" else rng.randint(2, max(2, self.m.days_later_max))
            t = (day0 + days) * 86400 + rng.uniform(9, 22) * 3600
        c.query_time = max(earliest, min(t, latest))
        c.meta["delay_bucket"] = self._bucket(earliest, c)

    def _bucket(self, earliest: float, c: Candidate) -> str:
        """Label from the ACTUAL delay (clipping can shorten the chosen one)."""
        days = int(c.query_time // 86400) - int(earliest // 86400)
        return "same_day" if days == 0 else "next_day" if days == 1 else "days_later"

    def _assign_condition(self, c: Candidate) -> BeliefCondition:
        r = stable_rng("condition", c.candidate_id).random()
        if r < self.m.correct_belief_rate:
            return BeliefCondition.CORRECT
        if r < self.m.correct_belief_rate + self.m.incomplete_belief_rate:
            return BeliefCondition.INCOMPLETE
        return BeliefCondition.MISBOUND

    def split_of(self, ep: Episode) -> str | None:
        return self.index.split_of(ep.day, self.narrator, self.cfg.data)

    def same_split(self, *eps: Episode) -> bool:
        return len({ep.day for ep in eps}) == 1 or len({self.split_of(e) for e in eps}) == 1

    ###########################################################################
    # Is a value actually stated in the captions? (Event files are LLM annotations:
    # values that the rows never state make candidates Stage 1 must reject.)
    ###########################################################################

    def own_texts(self, ep: Episode, with_speakers: bool = False) -> list[str]:
        rows = [r for r in self.index.episode_rows(ep.episode_id) if r.participant == ep.participant]
        texts = [r.text for r in rows]
        if with_speakers:
            texts += [r.speaker for r in rows if r.speaker]
        return texts

    def stated(self, value: str, ep: Episode, slot: str, min_coverage: float = 0.5) -> bool:
        """Objects/location: head noun present. People: named or speaking. Phrases: >= min_coverage."""
        if slot in HEAD_NOUN_SLOTS:
            return lexically_supported(value, self.own_texts(ep), self.index.synonyms, head_noun=True)
        if slot == "people":
            return lexically_supported(value, self.own_texts(ep, with_speakers=True), self.index.synonyms)
        return coverage(value, self.own_texts(ep)) >= min_coverage

    def cid(self, *parts) -> str:
        return f"{self.task.name}__" + "__".join(str(p) for p in parts)
