"""
index.py  (Stage 0, Part 1)

Builds the week-long memory bank from hour-level caption and event files:

  * Row table     -- every caption/utterance with ABSOLUTE timestamps
  * Episode table -- every event, with canonicalised slots
  * lookups       -- row_id -> Row, episode_id -> Episode, rows per file

Normalisation applied to episodes (each change is recorded in `flags`):
  * each file's NARRATOR is its wearer, read from the file name (A3_TASHA_... -> tasha)
  * the narrator is always added to `people` (it is their camera)
  * person names and speaker labels are mapped to one canonical registry
  * slot values are lower-cased, articles stripped, synonyms merged
  * compound locations ("Workstation/Studio") -> first part, raw value kept

The event JSON is treated as an INDEX, not as ground truth: Stage 1 re-reads
the raw rows through `evidence_ids` before any fact is accepted.

Persist with `save(dir)` / `load(dir)` so Part 2 (mining) and later stages do
not re-parse the raw files; inspect `index_report.json` (see index_report.py).
"""

from __future__ import annotations

import glob
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from .schemas import Episode, Row
from .utils import (canon, parse_hour_file_id, parse_participant, parse_span, read_json,
                    read_jsonl, to_abs, write_json)

log = logging.getLogger(__name__)

# Individual warnings printed per build before switching to a summary count;
# every skipped entry is still written to skipped_times.jsonl by `save`.
_MAX_SKIP_WARNINGS = 20

# Extend via DataConfig.synonyms_path as you find variants in the event files.
DEFAULT_SYNONYMS = {
    "dining table": "table",
    "cellphone": "cell phone",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "harddrive": "hard drive",
    "studio": "workstation",
}


# People-list entries that are not a person: dropped from episode.people.
GROUP_WORDS = frozenset({
    "friends", "friend", "group", "group members", "others", "other", "companions", "participants",
    "team members", "colleagues", "guests", "roommates", "roommate", "staff", "store staff",
    "unknown", "everyone", "people", "someone", "students", "student", "janitors", "cleaner",
})
# Entries that mean the wearer themself.
SELF_WORDS = frozenset({"narrator", "user", "i", "me", "myself", "camera wearer", "wearer"})


class MemoryIndex:
    def __init__(self, synonyms: dict[str, str] | None = None, people=(), people_aliases=None,
                 participants=()):
        self.synonyms = {**DEFAULT_SYNONYMS, **(synonyms or {})}
        self.people = {canon(p) for p in people}
        self.people_aliases = {canon(k): canon(v) for k, v in (people_aliases or {}).items()}
        self.only_participants = {canon(p) for p in participants}
        self.rows: dict[str, Row] = {}
        self.episodes: dict[str, Episode] = {}
        self.rows_by_file: dict[str, list[Row]] = defaultdict(list)
        self.files: dict[str, dict] = {}          # file_id -> {"day", "offset", "has_captions", "has_events"}
        self.stats: Counter = Counter()
        self.skipped: list[dict] = []             # malformed entries: file, id, reason, raw times

    ###########################################################################
    # Building from raw files
    ###########################################################################

    @classmethod
    def build(cls, captions_glob: str, events_glob: str, synonyms_path: str = "",
              people=(), people_aliases=None, participants=()) -> "MemoryIndex":
        synonyms = read_json(synonyms_path) if synonyms_path else None
        idx = cls(synonyms, people, people_aliases, participants)
        cap_paths, ev_paths = sorted(glob.glob(captions_glob)), sorted(glob.glob(events_glob))
        if idx.only_participants:
            keep = lambda p: parse_participant(cls._file_id(Path(p))) in idx.only_participants  # noqa: E731
            cap_paths, ev_paths = [p for p in cap_paths if keep(p)], [p for p in ev_paths if keep(p)]
        if not cap_paths and not ev_paths:
            raise FileNotFoundError(f"No files match {captions_glob!r} or {events_glob!r}")
        for path in cap_paths:
            idx._load_caption_file(Path(path))
        for path in ev_paths:
            idx._load_event_file(Path(path))
        idx._post_checks()
        idx._report()
        return idx

    def _file_meta(self, file_id: str) -> dict:
        if file_id not in self.files:
            day, offset = parse_hour_file_id(file_id)
            self.files[file_id] = {"day": day, "offset": offset, "participant": parse_participant(file_id),
                                   "has_captions": False, "has_events": False}
        return self.files[file_id]

    @staticmethod
    def _file_id(path: Path) -> str:
        # Tolerate prefixes/suffixes such as "top5_events_A1_JAKE_DAY1_11000000.json".
        import re
        m = re.search(r"A\d+_[A-Z]+_DAY\d+_\d{8}", path.stem)
        return m.group(0) if m else path.stem

    def _skip(self, kind: str, path: Path, rec_id, record, err: Exception) -> None:
        """Record a malformed entry instead of crashing the whole build."""
        raw = {k: record.get(k) for k in ("start", "end", "start_time", "end_time") if k in record} \
            if isinstance(record, dict) else {}
        entry = {"kind": kind, "file": str(path), "id": rec_id, "reason": str(err), "raw_times": raw}
        self.skipped.append(entry)
        self.stats[f"{kind}s_bad_time"] += 1
        if len(self.skipped) <= _MAX_SKIP_WARNINGS:
            log.warning("Skipping %s %s in %s: %s %s", kind, rec_id, path.name, err, raw)
        elif len(self.skipped) == _MAX_SKIP_WARNINGS + 1:
            log.warning("Further malformed entries are not printed; see skipped_times.jsonl")

    _CITE_RE = __import__("re").compile(r"\s*\[cite:[^\]]*\]")

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Strip annotation artefacts such as '[cite: 4]' left by the captioning tool."""
        return cls._CITE_RE.sub("", text).strip()

    def _load_caption_file(self, path: Path) -> None:
        file_id = self._file_id(path)
        meta = self._file_meta(file_id)
        meta["has_captions"] = True
        for i, r in enumerate(read_json(path)):
            try:
                start, end, padded = parse_span(r)
            except (KeyError, ValueError, AttributeError) as err:
                self._skip("row", path, f"{file_id}#/{i}", r, err)
                continue
            self.stats["rows_zero_len_padded"] += padded
            speaker = r.get("speaker")
            speaker = self.person(speaker) if isinstance(speaker, str) and speaker.strip() else None
            row = Row(
                row_id=f"{file_id}#/{i}",           # index in the ORIGINAL list, as evidence_ids use
                file_id=file_id,
                participant=meta["participant"],
                idx=i,
                kind=r.get("kind", "caption"),
                text=self._clean_text(r.get("text_en") or ""),
                speaker=speaker,
                abs_start=to_abs(meta["day"], meta["offset"] + start),
                abs_end=to_abs(meta["day"], meta["offset"] + end),
            )
            self.rows[row.row_id] = row
            self.rows_by_file[file_id].append(row)
            self.stats[f"rows_{row.kind}"] += 1
            if row.kind == "utterance":
                self.stats["utt_with_speaker" if row.speaker else "utt_no_speaker"] += 1
                if row.speaker and self.people and row.speaker not in self.people:
                    self.stats["unknown_speaker_names"] += 1

    def _canon_location(self, raw: str) -> str:
        if not raw:
            return ""
        first = raw.replace("\\", "/").split("/")[0]
        return canon(first, self.synonyms)

    def _load_event_file(self, path: Path) -> None:
        file_id = self._file_id(path)
        meta = self._file_meta(file_id)
        meta["has_events"] = True
        data = read_json(path)
        events = self._events_list(data, path)
        narrator = meta["participant"]
        for n, e in enumerate(events):
            ev_id = e.get("event_id") if isinstance(e, dict) else None
            try:
                if not ev_id:
                    raise KeyError("missing event_id")
                start, end, padded = parse_span(e)
            except (KeyError, ValueError, AttributeError) as err:
                self._skip("event", path, ev_id or f"{file_id}#event{n}", e, err)
                continue
            c = lambda xs: sorted({canon(x, self.synonyms) for x in (xs or []) if x})  # noqa: E731
            flags = ["zero_length_padded"] if padded else []
            people = set()
            for x in (e.get("people") or []):
                if not isinstance(x, str) or not x.strip():
                    continue
                p = self.person(x)
                if p in SELF_WORDS:
                    p = narrator                      # "narrator" / "user" / "me" = the wearer
                elif p in GROUP_WORDS:
                    self.stats["people_group_words_dropped"] += 1
                    continue                          # "friends", "group members": not a person
                people.add(p)
            people = sorted(people)
            if narrator not in people:
                people = sorted(people + [narrator])
                flags.append("narrator_added")
            raw_loc = e.get("location") or ""
            ep = Episode(
                episode_id=ev_id,
                file_id=file_id,
                participant=narrator,
                day=meta["day"],
                abs_start=to_abs(meta["day"], meta["offset"] + start),
                abs_end=to_abs(meta["day"], meta["offset"] + end),
                summary=e.get("summary", ""),
                people=people,
                activity=c(e.get("activity")),
                discussion=c(e.get("discussion")),
                plans=c(e.get("plans")),
                decisions=c(e.get("decisions")),
                objects=c(e.get("objects_or_food")),
                location=self._canon_location(raw_loc),
                location_raw=raw_loc,
                state_changes=c(e.get("state_changes")),
                evidence_ids=list(e.get("evidence_ids", [])),
                flags=flags,
            )
            if ep.episode_id in self.episodes:
                self.stats["duplicate_episode_ids"] += 1
            self.episodes[ep.episode_id] = ep
            self.stats["episodes"] += 1
            self.stats["narrator_added"] += "narrator_added" in flags
            self.stats["episodes_zero_len_padded"] += padded

    def _events_list(self, data, path: Path) -> list:
        """
        The events of one file. Usually {"events": [...]}; if that key is missing, use
        the first list of event-like dicts under any key; warn (instead of silently
        loading nothing) when a file yields no events.
        """
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            events = data.get("events")
            if events is None:
                for k, v in data.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict) and \
                            ({"event_id", "start_time"} & set(v[0])):
                        log.info("%s: events found under key %r", path.name, k)
                        self.stats["event_files_other_key"] += 1
                        events = v
                        break
        else:
            events = None
        if not events:
            self.stats["event_files_without_events"] += 1
            keys = sorted(data)[:8] if isinstance(data, dict) else type(data).__name__
            if self.stats["event_files_without_events"] <= 10:
                log.warning("No events loaded from %s (top-level keys: %s)", path.name, keys)
            return []
        return events

    def person(self, name: str) -> str:
        """Canonical person name (registry + aliases); 'I'/'me' is resolved by callers."""
        n = canon(name)
        return self.people_aliases.get(n, n)

    def _post_checks(self) -> None:
        """Checks that need both tables loaded."""
        for ep in self.episodes.values():
            if self.people:
                unknown = [p for p in ep.people if p not in self.people]
                if unknown:
                    ep.flags.append("unknown_people:" + ",".join(unknown))
                    self.stats["episodes_unknown_people"] += 1
            has_caps = self.files[ep.file_id]["has_captions"]
            missing = [x for x in ep.evidence_ids if x not in self.rows]
            if has_caps and missing:
                ep.flags.append("dangling_evidence")
                self.stats["evidence_ids_dangling"] += len(missing)
            if not has_caps:
                ep.flags.append("no_caption_file")
            if ep.abs_end <= ep.abs_start:
                ep.flags.append("bad_span")
                self.stats["episodes_bad_span"] += 1

    def _report(self) -> None:
        s = self.stats
        utt = s["rows_utterance"] or 1
        log.info(
            "Index: %d files, %d episodes, %d captions, %d utterances (%.1f%% with speaker), "
            "%d dangling evidence ids, narrator added to %d episodes",
            len(self.files), s["episodes"], s["rows_caption"], s["rows_utterance"],
            100 * s["utt_with_speaker"] / utt, s["evidence_ids_dangling"], s["narrator_added"],
        )
        log.info(
            "Times: skipped %d rows and %d events with malformed times; "
            "padded %d rows and %d events from zero length to 1 s",
            s["rows_bad_time"], s["events_bad_time"], s["rows_zero_len_padded"], s["episodes_zero_len_padded"],
        )

    ###########################################################################
    # Persistence
    ###########################################################################

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for r in sorted(self.rows.values(), key=lambda r: (r.abs_start, r.row_id)):
                f.write(r.model_dump_json() + "\n")
        with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for e in self.episodes_sorted():
                f.write(e.model_dump_json() + "\n")
        with (out / "skipped_times.jsonl").open("w", encoding="utf-8") as f:
            for entry in self.skipped:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        write_json(out / "index_meta.json", {"synonyms": self.synonyms, "people": sorted(self.people),
                                              "people_aliases": self.people_aliases,
                                              "files": self.files, "stats": dict(self.stats)})

    @classmethod
    def load(cls, in_dir: str | Path) -> "MemoryIndex":
        d = Path(in_dir)
        meta = read_json(d / "index_meta.json")
        idx = cls(meta["synonyms"], meta.get("people", ()), meta.get("people_aliases"))
        idx.files = meta["files"]
        idx.stats = Counter(meta["stats"])
        for o in read_jsonl(d / "rows.jsonl"):
            r = Row.model_validate(o)
            idx.rows[r.row_id] = r
            idx.rows_by_file[r.file_id].append(r)
        for rows in idx.rows_by_file.values():
            rows.sort(key=lambda r: r.idx)
        for o in read_jsonl(d / "episodes.jsonl"):
            e = Episode.model_validate(o)
            idx.episodes[e.episode_id] = e
        return idx

    ###########################################################################
    # Queries
    ###########################################################################

    @property
    def participants(self) -> list[str]:
        return sorted({m["participant"] for m in self.files.values()})

    def episodes_sorted(self, participant: str | None = None) -> list[Episode]:
        eps = self.episodes.values() if participant is None else \
            (e for e in self.episodes.values() if e.participant == participant)
        return sorted(eps, key=lambda e: e.abs_start)

    def rows_in_span(self, file_id: str, start: float, end: float) -> list[Row]:
        return [r for r in self.rows_by_file.get(file_id, []) if r.abs_end >= start and r.abs_start <= end]

    def episode_rows(self, episode_id: str) -> list[Row]:
        """Rows inside the episode span (NOT only the cited evidence ids)."""
        ep = self.episodes[episode_id]
        return self.rows_in_span(ep.file_id, ep.abs_start, ep.abs_end)

    def get_rows(self, ids) -> list[Row]:
        return [self.rows[i] for i in ids if i in self.rows]

    def episode_of_row(self, row_id: str) -> str | None:
        r = self.rows.get(row_id)
        if r is None:
            return None
        for ep in self.episodes.values():
            if ep.file_id == r.file_id and ep.abs_start <= r.abs_start <= ep.abs_end:
                return ep.episode_id
        return None

    def bank_until(self, abs_time: float, participant: str | None) -> list[Episode]:
        """
        Ego-centric, causal memory bank: only the given wearer's OWN episodes that
        ended before the query time. Pass participant=None only for analysis
        (an omniscient "household" bank must never be given to the assistant).
        """
        return [e for e in self.episodes_sorted(participant) if e.abs_end <= abs_time]

    @staticmethod
    def split_of(day: int, participant: str, data_cfg) -> str | None:
        """
        Day-based split, shared by ALL participants (the same event is seen by
        several wearers, so a day must never straddle splits).
        Held-out participants: test-day samples -> "test_heldout_person"; their
        other days are dropped (None) so the held-out person is never trained on.
        """
        heldout = {h.lower() for h in data_cfg.heldout_participants}
        if participant in heldout:
            return "test_heldout_person" if day in data_cfg.test_days else None
        if day in data_cfg.test_days:
            return "test"
        if day in data_cfg.dev_days:
            return "dev"
        return "train"


def json_dumps(obj) -> str:  # small helper for reports
    return json.dumps(obj, indent=2, ensure_ascii=False)