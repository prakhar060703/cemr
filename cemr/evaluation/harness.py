"""
evaluation/harness.py

Interactive evaluation: the model under test plays the assistant against the
same deterministic-controller user simulator used for generation. Scoring is
on the user's FINAL BELIEF STATE, not on transcript overlap.

Modes
-----
multi_turn   (default) full reconstruction dialogue
single_turn  one assistant reply, no user resistance -- the collapsed
             false-premise baseline. Reporting both shows whether a model that
             can spot a false premise can also carry a reconstruction dialogue.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from ..dialogue.selfplay import run_evaluation
from ..utils import append_jsonl, read_jsonl, write_json
from .metrics import aggregate

log = logging.getLogger(__name__)


def evaluate(pipeline, model_client, model_name: str, split: str = "test",
             mode: str = "multi_turn", limit: int | None = None) -> dict:
    ctx = pipeline.ctx
    if mode == "single_turn":
        ctx = copy.copy(ctx)
        ctx.cfg = copy.deepcopy(ctx.cfg)
        ctx.cfg.dialogue.primary_resistance = 0
        ctx.cfg.dialogue.secondary_resistance = 0
        max_turns = 2
    else:
        max_turns = None

    out_dir = Path(pipeline.cfg.output_dir) / "eval" / model_name.replace("/", "_") / f"{split}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "records.jsonl"
    done = {r["sample_id"] for r in read_jsonl(res_path)}

    for n, (cand, gt, fm, opening) in enumerate(pipeline.load_samples(split)):
        if limit and n >= limit:
            break
        if gt.sample_id in done:
            continue
        try:
            rec = run_evaluation(ctx, cand, gt, fm, model_client, opening_text=opening, max_turns=max_turns)
        except Exception as e:  # keep going; failures are recorded
            log.exception("eval failed for %s", gt.sample_id)
            rec = {"sample_id": gt.sample_id, "error": str(e)[:300]}
        append_jsonl(res_path, rec)

    records = [r for r in read_jsonl(res_path) if "error" not in r]
    report = aggregate(records, pipeline.cfg.dialogue.max_corrections_per_turn)
    report["n_errors"] = sum(1 for r in read_jsonl(res_path) if "error" in r)
    write_json(out_dir / "metrics.json", report)
    return report
