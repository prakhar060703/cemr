"""
crossview.py  (Stage 0, Part 1b)

EgoLife has several wearers recording the same household at the same time.
This module links their episodes into EVENT CLUSTERS -- one real-world event
seen from several first-person views -- and answers two questions:

  1. Dedup: which candidates are near-duplicates (same event, different wearer)?
  2. Visibility: is a fact visible in the USER'S OWN recording, or only in
     someone else's?

Rule used throughout the benchmark:
  * the ASSISTANT only ever searches the user's own recordings (ego-centric);
  * the GOLD labels may be verified with every view (world truth);
  * a correction is only gold if its evidence is in the user's own recording.

Linking rule (conservative, time is synchronised across wearers by EgoSync):
two episodes of different wearers are the same event if
    overlap(time) / min(duration) >= cluster_min_overlap
and each wearer is listed as present in the other's episode.
"""

from __future__ import annotations

from collections import defaultdict

from .index import MemoryIndex
from .schemas import Episode, Row


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _overlap_ratio(a: Episode, b: Episode) -> float:
    inter = min(a.abs_end, b.abs_end) - max(a.abs_start, b.abs_start)
    shorter = min(a.abs_end - a.abs_start, b.abs_end - b.abs_start)
    return inter / shorter if inter > 0 and shorter > 0 else 0.0


def build_clusters(index: MemoryIndex, min_overlap: float) -> dict[str, list[str]]:
    """
    Assigns `episode.cluster_id` for every episode and returns
    {cluster_id: [episode_ids]}. Singletons get their own cluster.
    """
    uf = _UnionFind()
    eps = index.episodes_sorted()
    for e in eps:
        uf.find(e.episode_id)
    # Sweep line over time: only compare episodes that can overlap.
    active: list[Episode] = []
    for e in eps:
        active = [a for a in active if a.abs_end > e.abs_start]
        for a in active:
            if a.participant == e.participant:
                continue
            if _overlap_ratio(a, e) < min_overlap:
                continue
            if a.participant in e.people and e.participant in a.people:
                uf.union(a.episode_id, e.episode_id)
        active.append(e)

    clusters: dict[str, list[str]] = defaultdict(list)
    for e in eps:
        root = uf.find(e.episode_id)
        clusters[root].append(e.episode_id)
    named = {}
    for root, members in clusters.items():
        cid = "C_" + root
        for m in members:
            index.episodes[m].cluster_id = cid
        named[cid] = members
    index.stats["event_clusters"] = len(named)
    index.stats["multi_view_clusters"] = sum(1 for m in named.values() if len(m) > 1)
    return named


class CrossView:
    """Query helper over clusters (build once per run)."""

    def __init__(self, index: MemoryIndex, clusters: dict[str, list[str]]):
        self.index = index
        self.clusters = clusters

    @classmethod
    def from_index(cls, index: MemoryIndex, min_overlap: float = 0.3) -> "CrossView":
        """Use cluster ids saved by Part 1; build them if the index has none."""
        if any(e.cluster_id is None for e in index.episodes.values()):
            return cls(index, build_clusters(index, min_overlap))
        clusters: dict[str, list[str]] = defaultdict(list)
        for e in index.episodes_sorted():
            clusters[e.cluster_id].append(e.episode_id)
        return cls(index, dict(clusters))

    def views(self, episode_id: str) -> list[Episode]:
        """All episodes (every wearer) of the same real event, including itself."""
        cid = self.index.episodes[episode_id].cluster_id
        return [self.index.episodes[e] for e in self.clusters.get(cid, [episode_id])]

    def other_views(self, episode_id: str) -> list[Episode]:
        me = self.index.episodes[episode_id].participant
        return [e for e in self.views(episode_id) if e.participant != me]

    def other_view_rows(self, episode_id: str) -> list[Row]:
        rows: list[Row] = []
        for e in self.other_views(episode_id):
            rows += self.index.episode_rows(e.episode_id)
        return rows

    def n_views(self, episode_id: str) -> int:
        return len({e.participant for e in self.views(episode_id)})

    def first_person_rows(self, participant: str, start: float, end: float) -> list[Row]:
        """Captions written in the first person by `participant` within a time window."""
        out = []
        for fid, rows in self.index.rows_by_file.items():
            meta = self.index.files.get(fid, {})
            if meta.get("participant") != participant:
                continue
            # Hour files: skip files that cannot overlap the window (650k rows otherwise).
            if "day" in meta and "offset" in meta:
                f_start = (meta["day"] - 1) * 86400 + meta["offset"]
                if f_start > end or f_start + 3 * 3600 < start:
                    continue
            for r in rows:
                if r.kind == "caption" and start <= r.abs_start <= end:
                    w = r.text.lower().split()
                    if w and w[0] in ("i", "i'm", "i've"):
                        out.append(r)
        return out


def mark_visibility(facts, participant: str, index: MemoryIndex) -> None:
    """Deterministic: a fact is visible to the user iff any evidence row is their own."""
    for f in facts:
        f.visible_to_user = any(
            index.rows[e].participant == participant for e in f.evidence_ids if e in index.rows)
