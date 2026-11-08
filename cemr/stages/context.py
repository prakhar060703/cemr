"""Shared run context passed to every stage."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from ..config import Config
from ..crossview import CrossView
from ..index import MemoryIndex
from ..retrieval import BM25Retriever
from ..schemas import Candidate
from ..utils import short_time
from ..validation.verifier import LLMVerifier


@dataclass
class Ctx:
    cfg: Config
    index: MemoryIndex
    generator: object              # LLMClient-like: generate_json(system, user, schema=None)
    user_sim: object
    verifier: Optional[LLMVerifier]
    xv: Optional[CrossView] = None
    _retrievers: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if self.xv is None:
            self.xv = CrossView.from_index(self.index, self.cfg.mining.cluster_min_overlap)

    def retriever_until(self, abs_time: float, participant: str) -> BM25Retriever:
        """
        Ego-centric, causal retriever: ONLY `participant`'s own episodes that ended
        before the query time (cached per participant and hour).
        """
        key = (participant, int(abs_time // 3600))
        with self._lock:            # samples run in parallel threads
            if key not in self._retrievers:
                self._retrievers[key] = BM25Retriever(self.index, self.index.bank_until(abs_time, participant))
            return self._retrievers[key]

    def _fmt(self, r, user: str | None) -> str:
        who = f"{r.speaker}: " if r.speaker else ""
        tag = "" if user is None or r.participant == user else f"[OTHER VIEW: {r.participant}] "
        return f"<{r.row_id}> [{short_time(r.abs_start)}] {tag}({r.kind}) {who}{r.text}"

    def rows_block(self, row_ids=None, episode_ids=(), limit_per_episode: int = 60,
                   user: str | None = None, key_rows=()) -> str:
        """
        Rows grouped by episode. When `user` is given, rows recorded by someone
        else are tagged [OTHER VIEW: name] -- the user did not see or hear them.
        """
        seen, lines = set(), []
        for ep_id in episode_ids:
            if ep_id not in self.index.episodes or ep_id in seen:
                continue
            seen.add(ep_id)
            ep = self.index.episodes[ep_id]
            view = "" if user is None or ep.participant == user else f" [OTHER VIEW: {ep.participant}]"
            lines.append(f"== {ep_id}{view} | {ep.summary}")
            rows = self.index.episode_rows(ep_id)
            # First N rows, plus every KEY row (answer / lure / plan mentions) with one
            # neighbour each side, wherever it sits in a long episode.
            keep = set(range(min(limit_per_episode, len(rows))))
            for i, r in enumerate(rows):
                if r.row_id in key_rows:
                    keep |= {i - 1, i, i + 1}
            prev = -1
            for i in sorted(k for k in keep if 0 <= k < len(rows)):
                if prev >= 0 and i != prev + 1:
                    lines.append("   ...")
                r = rows[i]
                seen.add(r.row_id)
                lines.append(self._fmt(r, user))
                prev = i
        for rid in row_ids or []:
            if rid in seen or rid not in self.index.rows:
                continue
            lines.append(self._fmt(self.index.rows[rid], user))
        return "\n".join(lines)

    # ~4 characters per token: 60k chars ~ 15k tokens, leaving room in a 32k context for
    # the instructions and a 4k answer. (A T4 block with several other views overflowed.)
    MAX_BLOCK_CHARS = 60000

    def key_rows(self, cand: Candidate) -> set[str]:
        """Rows that state the candidate's answer, lure or plan: always shown."""
        from ..validation.deterministic import CONCRETE_SLOTS, coverage, lexically_supported
        # Concrete answers are words to look for; derived ones ("witnessed myself",
        # "completed") are labels, so use the phrases the miner stored instead.
        values = [v for v in (cand.true_value, cand.lure_value) if v] \
            if cand.primary_slot in CONCRETE_SLOTS else []
        phrases = [p for p in (cand.meta.get("plan_text"), cand.meta.get("action"),
                               cand.meta.get("outcome")) if p]
        keys = set(cand.lure_rows)
        for ep_id in [cand.target_episode, *cand.lure_episodes]:
            for r in self.index.episode_rows(ep_id) if ep_id in self.index.episodes else []:
                text = r.text + " " + (r.speaker or "")
                if any(lexically_supported(v, [text], self.index.synonyms, head_noun=True) for v in values) \
                        or any(coverage(p, [r.text]) >= 0.5 for p in phrases):
                    keys.add(r.row_id)
        return keys

    def candidate_rows_block(self, cand: Candidate, limit_per_episode: int = 60) -> str:
        """Target + lure episodes (own view) + verification views/rows (other views)."""
        eps = [cand.target_episode, *cand.lure_episodes, *cand.meta.get("verify_episodes", [])]
        rows = list(cand.lure_rows) + list(cand.meta.get("verify_rows", []))
        keys = self.key_rows(cand)
        limit = limit_per_episode
        while True:
            block = self.rows_block(rows, eps, limit, user=cand.participant, key_rows=keys)
            if len(block) <= self.MAX_BLOCK_CHARS or limit <= 10:
                return block
            limit = max(10, limit // 2)     # shrink the context rows; key rows always stay


def with_feedback(prompt: str, feedback) -> str:
    if not feedback:
        return prompt
    return prompt + "\n\nA previous attempt was rejected for these reasons; fix all of them:\n- " + \
        "\n- ".join(feedback)
