"""
utils.py

Small, dependency-free helpers: time parsing, text normalisation,
JSONL IO, deterministic seeding.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from pathlib import Path
from typing import Iterable, Iterator

###############################################################################
# Time
###############################################################################

_HOUR_FILE_RE = re.compile(r"DAY(\d+)_(\d{2})(\d{2})(\d{2})(\d{2})")
_PARTICIPANT_RE = re.compile(r"A(\d+)_([A-Za-z]+)_DAY")


def parse_participant(file_id: str) -> str:
    """'A1_JAKE_DAY1_11000000' -> 'jake' (the wearer of the glasses)."""
    m = _PARTICIPANT_RE.search(file_id)
    if not m:
        raise ValueError(f"Cannot parse participant from '{file_id}'")
    return m.group(2).lower()


# HH:MM:SS with an optional fraction after '.' (events) or ',' (captions, SRT style).
# The fraction is deliberately DROPPED: the whole pipeline works in whole seconds,
# always truncated (never rounded), so captions and events stay on one clock.
_CLOCK_RE = re.compile(r"^(\d{1,3}):(\d{1,2}):(\d{1,2})(?:[.,]\d*)?$")

_CLOCK_MS4_RE = re.compile(r"^(\d{1,3}):(\d{1,2}):(\d{1,2}):\d{3}$")
_CLOCK_MS3_RE = re.compile(r"^(\d{1,2}):(\d{1,2}):\d{3}$")

# Accepted (start, end) field names, in priority order.
#   captions: "start" / "end"      events: "start_time" / "end_time"
TIME_FIELDS = (("start", "end"), ("start_time", "end_time"))


def parse_clock(ts) -> int:
    """
    The ONE time parser for the pipeline.
    '00:09:43,500' / '00:09:43.500' / '00:09:43' -> 583 (whole seconds, fraction dropped).
    Anything else raises ValueError naming the offending value.
    """
    if not isinstance(ts, str):
        raise ValueError(f"time is not a string: {ts!r}")
    t = ts.strip()
    m = _CLOCK_RE.match(t)
    if m:
        h, mi, s = (int(x) for x in m.groups())
    else:
        # Milliseconds written after a colon (seen in the EgoLife event files). A 3-digit
        # last field cannot be seconds, so these are unambiguous:
        #   '00:00:46:100' -> HH:MM:SS:mmm     '01:10:366' -> MM:SS:mmm
        m4 = _CLOCK_MS4_RE.match(t)
        m3 = _CLOCK_MS3_RE.match(t)
        if m4:
            h, mi, s = (int(x) for x in m4.groups())
        elif m3:
            h, (mi, s) = 0, (int(x) for x in m3.groups())
        else:
            raise ValueError(f"bad time {ts!r}: expected HH:MM:SS (anything after '.' or ',' is ignored)")
    if mi > 59 or s > 59:
        raise ValueError(f"bad time {ts!r}: minutes/seconds out of range")
    return h * 3600 + mi * 60 + s


def parse_span(record: dict) -> tuple[int, int, bool]:
    """
    Read a record's (start, end) from whichever field names it uses and parse
    both with `parse_clock`.

    Returns (start, end, padded):
      * end <  start  -> ValueError (malformed; the caller logs and skips it)
      * end == start  -> end = start + 1 and padded=True (truncation can collapse
                         sub-second items to zero length, which breaks overlap checks)
    Raises KeyError if no known start/end field pair is present.
    """
    for s_key, e_key in TIME_FIELDS:
        if s_key in record and e_key in record:
            start, end = parse_clock(record[s_key]), parse_clock(record[e_key])
            break
    else:
        raise KeyError(f"no start/end fields (expected one of {TIME_FIELDS})")
    if end < start:
        raise ValueError(f"end before start ({record[e_key]!r} < {record[s_key]!r})")
    if end == start:
        return start, start + 1, True
    return start, end, False


def parse_hour_file_id(file_id: str) -> tuple[int, float]:
    """
    'A1_JAKE_DAY1_11000000' -> (day=1, offset_seconds=11*3600).
    The trailing 8 digits are HHMMSSFF.
    """
    m = _HOUR_FILE_RE.search(file_id)
    if not m:
        raise ValueError(f"Cannot parse day/hour from '{file_id}'")
    day, hh, mm, ss, _ff = (int(x) for x in m.groups())
    return day, hh * 3600 + mm * 60 + ss


def to_abs(day: int, seconds_of_day: float) -> float:
    """Absolute seconds since DAY1 00:00."""
    return (day - 1) * 86400 + seconds_of_day


def fmt_abs(abs_seconds: float) -> str:
    """Absolute seconds -> 'D1T11:09:43.5' (human readable, sortable)."""
    day = int(abs_seconds // 86400) + 1
    rem = abs_seconds - (day - 1) * 86400
    h = int(rem // 3600)
    m = int((rem % 3600) // 60)
    s = rem % 60
    return f"D{day}T{h:02d}:{m:02d}:{s:04.1f}"


def now_label(abs_seconds: float) -> str:
    """Absolute seconds -> 'Day 4, 14:30 (afternoon)': the conversation's current time."""
    day = int(abs_seconds // 86400) + 1
    rem = int(abs_seconds - (day - 1) * 86400)
    part = natural_time(abs_seconds).split(", ", 1)[1]
    return f"Day {day}, {rem // 3600:02d}:{(rem % 3600) // 60:02d} ({part})"


def natural_time(abs_seconds: float) -> str:
    """Absolute seconds -> 'Day 1, late morning' (for user-facing text)."""
    day = int(abs_seconds // 86400) + 1
    hour = (abs_seconds % 86400) / 3600
    if hour < 6:
        part = "night"
    elif hour < 10:
        part = "early morning"
    elif hour < 12:
        part = "late morning"
    elif hour < 14:
        part = "around noon"
    elif hour < 17:
        part = "afternoon"
    elif hour < 21:
        part = "evening"
    else:
        part = "late evening"
    return f"Day {day}, {part}"


###############################################################################
# Text
###############################################################################

_WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the and or of to in on at for with from by is are was were be been it "
    "this that these those i you he she we they me my your our their his her "
    "then so just okay ok um uh yeah do did does have has had not no yes".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS]


def canon(value: str, synonyms: dict[str, str] | None = None) -> str:
    """Canonicalise a slot value: lowercase, trim, apply synonym map."""
    v = re.sub(r"\s+", " ", value.strip().lower())
    v = re.sub(r"^(the|a|an) ", "", v)
    if synonyms:
        v = synonyms.get(v, v)
    return v


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def contains_value(text: str, value: str) -> bool:
    """Whole-token containment, case-insensitive (used by leakage checks)."""
    vt = tokens(value)
    if not vt:
        return False
    tt = tokens(text)
    n = len(vt)
    return any(tt[i : i + n] == vt for i in range(len(tt) - n + 1))


###############################################################################
# IO / determinism
###############################################################################


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_APPEND_LOCK = threading.Lock()


def append_jsonl(path: str | Path, obj) -> None:
    """Append one JSON line. Locked, so parallel samples never interleave lines."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(line)


def read_json_list(path: str | Path) -> list:
    """A JSON file holding a list; [] if it does not exist yet."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def append_json_list(path: str | Path, obj) -> None:
    """
    Append one item to a JSON file that holds a list (e.g. conv.json), keeping it
    valid, indented JSON. Locked for parallel samples; written to a temp file and
    renamed, so a crash mid-write can never leave a half-written, unreadable file.
    """
    p = Path(path)
    with _APPEND_LOCK:
        items = read_json_list(p)
        items.append(obj)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


def read_conversations(out_dir: str | Path) -> list[dict]:
    """
    Accepted conversations: conv.json (current format) plus any lines in an old
    conv.jsonl from earlier runs, de-duplicated by sample_id.
    """
    out = Path(out_dir)
    convs = {c["sample_id"]: c for c in read_jsonl(out / "conv.jsonl")}
    convs.update({c["sample_id"]: c for c in read_json_list(out / "conv.json")})
    return list(convs.values())


def short_time(abs_seconds: float) -> str:
    """Absolute seconds -> 'D2 20:03:15' (shown next to rows so day/time claims are checkable)."""
    day = int(abs_seconds // 86400) + 1
    rem = int(abs_seconds - (day - 1) * 86400)
    return f"D{day} {rem // 3600:02d}:{(rem % 3600) // 60:02d}:{rem % 60:02d}"


def stable_rng(*keys) -> random.Random:
    """Per-sample RNG derived from ids, so reruns are reproducible regardless of order."""
    h = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))