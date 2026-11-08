"""
retrieval.py

Dependency-free BM25 over episodes (summary + slots + in-span rows).
Used (a) by the assistant agent during self-play and (b) by the evaluation
harness as the default memory tool. Swap in a dense retriever by
implementing the same `search()` signature.
"""

from __future__ import annotations

import math
from collections import Counter

from .index import MemoryIndex
from .schemas import Episode
from .utils import fmt_abs, short_time, tokens


class BM25Retriever:
    def __init__(self, index: MemoryIndex, episodes: list[Episode], k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.episodes = episodes
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        for ep in episodes:
            row_text = " ".join(r.text for r in index.episode_rows(ep.episode_id))
            self.docs.append(tokens(ep.search_text() + " " + row_text))
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str | list[str], k: int = 5) -> list[tuple[Episode, float]]:
        q = tokens(" ".join(query) if isinstance(query, list) else query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((self.episodes[i], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def render(self, hits: list[tuple[Episode, float]], rows_per_episode: int = 12) -> str:
        """Text block handed to the assistant LLM: episode header + cited rows with ids."""
        blocks = []
        for ep, _ in hits:
            lines = [
                f"[EPISODE {ep.episode_id}] {fmt_abs(ep.abs_start)}-{fmt_abs(ep.abs_end)}",
                # Automatic annotations help find the right episode but are NOT evidence.
                f"  auto-annotation (not evidence): @ {ep.location or 'unknown'} | "
                f"people: {', '.join(ep.people)} | {ep.summary}",
            ]
            for r in self.index.episode_rows(ep.episode_id)[:rows_per_episode]:
                who = f"{r.speaker}: " if r.speaker else ""
                lines.append(f"  <{r.row_id}> [{short_time(r.abs_start)}] ({r.kind}) {who}{r.text}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "(no matching episodes)"
