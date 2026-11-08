"""
T4 Experienced vs. told (source confusion / suggestibility).

A speaker-labelled utterance by someone other than the narrator that reports
a past event ("you left the stove on earlier"). We link it to the earlier
episode it most plausibly refers to. The user's belief is that they
WITNESSED what they were only TOLD.

Whether the report is accurate or conflicts with the recording is decided
in Stage 1 (LLM + evidence), producing subtypes `report_accurate` /
`report_conflicts`; report both separately.

Only utterances with a reliable speaker label are used.
"""

from __future__ import annotations

from ..schemas import Candidate, TaskId
from ..utils import tokens
from .base import BaseMiner


QUESTION_STARTS = ("did ", "do ", "does ", "have you", "has ", "is ", "are ", "was ", "were ",
                   "what ", "where ", "when ", "who ", "why ", "how ", "can ", "could ")
MIN_SHARED = 2


class ToldMiner(BaseMiner):
    task = TaskId.T4

    def _mine(self) -> list[Candidate]:
        out: list[Candidate] = []
        eps = self.eps
        cues = self.m.t4_report_cues
        for row in self.index.rows.values():
            if row.participant != self.narrator:      # only what THIS user heard
                continue
            if row.kind != "utterance" or not row.speaker:
                continue
            if row.speaker == self.narrator:
                continue
            text = row.text.lower().strip()
            if not any(c in text for c in cues):
                continue
            # A question ("did you ...?") asks; it does not TELL the narrator anything.
            if text.endswith("?") or text.startswith(QUESTION_STARTS):
                continue
            rtoks = set(tokens(row.text))
            # Same day only: reports refer to recent events, and splits are by day.
            row_day = int(row.abs_start // 86400) + 1
            earlier = [e for e in eps if e.abs_end < row.abs_start and e.day == row_day]
            if not earlier:
                continue
            # Link the report to the earlier episode whose CAPTIONS (and annotation) share
            # the most content words with it; require >= MIN_SHARED so the report is
            # really about that event, not a passing word like "water".
            def shared(e):
                ann = " ".join(e.objects + e.activity + e.state_changes)
                return len(rtoks & set(tokens(ann + " " + " ".join(self.own_texts(e)))))
            best = max(earlier, key=shared)
            if shared(best) < MIN_SHARED:
                continue
            report_ep = self.index.episode_of_row(row.row_id)
            out.append(Candidate(
                candidate_id=self.cid(best.episode_id, row.row_id.replace("#/", "_")),
                task=self.task,
                split=self.split_of(best),
                target_episode=best.episode_id,
                lure_episodes=[report_ep] if report_ep else [],
                lure_rows=[row.row_id],
                primary_slot="knowledge_source",
                true_value=f"told by {row.speaker}",
                lure_value="witnessed myself",
                subtype="unverified",           # refined in Stage 1
                query_time=row.abs_end + 1800,
                # Other wearers' views of the reported event: Stage 1 uses them to
                # decide whether the report was accurate (world truth).
                meta={"report_text": row.text, "reporter": row.speaker,
                      "verify_episodes": [e.episode_id for e in self.xv.other_views(best.episode_id)]},
            ))
        return out
