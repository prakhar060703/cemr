"""
Tiny SYNTHETIC dataset in the exact EgoLife caption/event format, used only to
test the plumbing. Episodes are designed so every miner fires at least once.
Days 1-3 are identical in structure (train / dev / test).
"""

from __future__ import annotations

import json
from pathlib import Path

HOUR_11 = {
    "rows": [
        ("00:09:45,000", "00:09:47,366", "caption", "I looked at the table and held my cell phone.", None),
        ("00:09:47,100", "00:09:50,766", "utterance", "Okay, then we need a stopwatch.", "Jake"),
        ("00:09:55,000", "00:09:58,000", "caption", "I passed the cell phone to Shure at the table.", None),
        ("00:10:00,000", "00:10:03,000", "utterance", "Everyone mark the timestamp on the phone.", "Jake"),
        ("00:10:05,000", "00:10:08,000", "utterance", "Katrina and Alice, your turn.", "Jake"),
        ("00:10:09,000", "00:10:12,000", "utterance", "We still need to buy decorations for the party.", "Alice"),
        ("00:10:25,000", "00:10:28,000", "caption", "I picked up the tripod at the table.", None),
        ("00:10:30,000", "00:10:34,000", "utterance", "Let's discuss the plan for the last day.", "Jake"),
        ("00:10:56,000", "00:10:59,000", "caption", "I placed the hard drive on the table for Shure and Lucia.", None),
        ("00:11:00,000", "00:11:03,000", "utterance", "This hard drive has all the footage.", "Lucia"),
    ],
    "events": [
        {
            "event_id": "E01", "start_time": "00:09:43.500", "end_time": "00:10:14.133",
            "summary": "Jake coordinates marking timestamps using a phone with everyone.",
            "people": ["Jake", "Shure", "Katrina", "Alice"],
            "activity": ["Operating phone", "Passing phone", "Marking timestamps"],
            "discussion": ["Need for a stopwatch", "Buying decorations"],
            "plans": ["Buy decorations for the party"],
            "decisions": ["Everyone must mark the timestamp"],
            "objects_or_food": ["Cell phone"], "location": "Table",
            "state_changes": ["Timestamp marking completed"],
            "evidence_ids": [0, 1, 2, 3, 5],
        },
        {
            "event_id": "E02", "start_time": "00:10:20.200", "end_time": "00:11:04.533",
            "summary": "Jake leads a meeting and introduces the hard drive and tripod.",
            "people": ["Jake", "Shure", "Lucia"],
            "activity": ["Discussing", "Handling hard drive", "Handling tripod"],
            "discussion": ["Last day plans"], "plans": ["Discussing morning activities"],
            "decisions": ["Proceed with discussing the hard drive"],
            "objects_or_food": ["Hard drive", "Tripod"], "location": "Table",
            "state_changes": ["Hard drive introduced"],
            "evidence_ids": [6, 7, 8, 9],
        },
    ],
}

HOUR_14 = {
    "rows": [
        ("00:05:00,000", "00:05:04,000", "caption", "Alice and Shure carried bags of decorations into the living room.", None),
        ("00:05:05,000", "00:05:08,000", "utterance", "We bought the decorations for the party.", "Alice"),
        ("00:20:00,000", "00:20:03,000", "caption", "I walked into the living room.", None),
        ("00:20:04,000", "00:20:08,000", "utterance", "You left the tripod on the table earlier.", "Katrina"),
        ("00:20:10,000", "00:20:13,000", "caption", "The tripod fell off the table.", None),
        ("00:20:14,000", "00:20:17,000", "utterance", "Oh no, the tripod fell.", "Jake"),
    ],
    "events": [
        {
            "event_id": "E03", "start_time": "00:04:58.000", "end_time": "00:05:10.000",
            "summary": "Alice and Shure bring the decorations they bought.",
            "people": ["Alice", "Shure"], "activity": ["Buying decorations", "Carrying bags"],
            "discussion": ["Party decorations"], "plans": [], "decisions": [],
            "objects_or_food": ["Decorations"], "location": "Living room",
            "state_changes": ["Decorations bought for the party"],
            "evidence_ids": [0, 1],
        },
        {
            "event_id": "E04", "start_time": "00:19:58.000", "end_time": "00:20:18.000",
            "summary": "Katrina mentions the tripod left on the table and it falls.",
            "people": ["Jake", "Katrina"], "activity": ["Talking", "Picking up tripod"],
            "discussion": ["Tripod left on table"], "plans": [], "decisions": [],
            "objects_or_food": ["Tripod"], "location": "Table",
            "state_changes": ["Tripod fell"],
            "evidence_ids": [2, 3, 4, 5],
        },
    ],
}


def _write_hour(root: Path, day: int, hhmm: str, spec: dict, wearer: str = "A1_JAKE") -> None:
    file_id = f"{wearer}_DAY{day}_{hhmm}0000"
    rows = [{"start": s, "end": e, "kind": k, "text_en": t, "speaker": sp}
            for s, e, k, t, sp in spec["rows"]]
    events = []
    for ev in spec["events"]:
        ev = dict(ev)
        ev["event_id"] = f"{file_id}_{ev['event_id']}"
        ev["evidence_ids"] = [f"{file_id}#/{i}" for i in ev["evidence_ids"]]
        events.append(ev)
    (root / "captions").mkdir(parents=True, exist_ok=True)
    (root / "events").mkdir(parents=True, exist_ok=True)
    (root / "captions" / f"{file_id}.json").write_text(json.dumps(rows, indent=1))
    (root / "events" / f"{file_id}.json").write_text(json.dumps({"events": events}, indent=1))


# Alice's glasses, same household, same hours (time-synchronised).
ALICE_11 = {
    "rows": [
        ("00:09:50,000", "00:09:53,000", "caption", "I watched Jake pass the cell phone around the table.", None),
        ("00:10:06,000", "00:10:09,000", "caption", "I marked my timestamp on the cell phone.", None),
        ("00:10:09,000", "00:10:12,000", "utterance", "We still need to buy decorations for the party.", "Alice"),
    ],
    "events": [
        {
            "event_id": "E01", "start_time": "00:09:45.000", "end_time": "00:10:14.000",
            "summary": "Alice marks her timestamp on Jake's phone at the table.",
            "people": ["Alice", "Jake", "Shure", "Katrina"],
            "activity": ["Marking timestamps"], "discussion": ["Buying decorations"],
            "plans": [], "decisions": [], "objects_or_food": ["Cell phone"], "location": "Table",
            "state_changes": [], "evidence_ids": [0, 1, 2],
        },
    ],
}
ALICE_14 = {
    "rows": [
        ("00:04:58,000", "00:05:03,000", "caption", "I bought decorations for the party and carried the bags in with Shure.", None),
    ],
    "events": [
        {
            "event_id": "E01", "start_time": "00:04:55.000", "end_time": "00:05:08.000",
            "summary": "Alice and Shure bring in the party decorations.",
            "people": ["Alice", "Shure"], "activity": ["Buying decorations"], "discussion": [],
            "plans": [], "decisions": [], "objects_or_food": ["Decorations"], "location": "Living room",
            "state_changes": ["Decorations bought"], "evidence_ids": [0],
        },
    ],
}


def build_toy_dataset(root: str | Path, days=(1, 2, 3), with_alice: bool = True) -> Path:
    root = Path(root)
    for d in days:
        _write_hour(root, d, "1100", HOUR_11)
        _write_hour(root, d, "1400", HOUR_14)
        if with_alice:
            _write_hour(root, d, "1100", ALICE_11, wearer="A2_ALICE")
            _write_hour(root, d, "1400", ALICE_14, wearer="A2_ALICE")
    return root
