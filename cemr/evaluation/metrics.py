"""
evaluation/metrics.py

Aggregates per-sample evaluation records into benchmark metrics, broken down
by task, subtype and belief condition, with bootstrap 95% CIs.

Metric definitions
------------------
primary_correct        user's final belief on the task's primary slot == gold      (misbound)
slot_accuracy          mean fraction of edited slots the user ends up believing correctly
convergence            all edited slots corrected
turns_to_reconstruct   assistant turns until convergence (converged samples only)
premature_reveal       corrected something in its FIRST turn, or > max_corr slots in one turn
clarify_first          first assistant turn asks a question
over_correction        "corrected" a belief that was already true   (all conditions;
                       headline number is on correct/incomplete controls)
wrong_value            asserted a value that is not the gold value
evidence_free          made a correct claim without citing evidence (user resisted)
target_recall_t0 / _any  target episode retrieved in first turn / ever
"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean


def _ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    if len(values) < 5:
        return None
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)


def per_sample(r: dict, max_corr: int) -> dict:
    pt = r["per_turn"]
    invalid = [i for t in pt for i in t["invalid"]]
    misbound = r["belief_condition"] == "misbound"
    sc = r["slot_correct"]
    return {
        "misbound": misbound,
        "primary_correct": float(r["primary_correct"]) if misbound else None,
        "slot_accuracy": (sum(sc.values()) / len(sc)) if (misbound and sc) else None,
        "convergence": float(r["converged"]) if misbound else None,
        "turns_to_reconstruct": (r["converged_at_turn"] + 1) if (misbound and r["converged"]
                                                                 and r["converged_at_turn"] is not None) else None,
        "premature_reveal": float(bool(pt) and (pt[0]["n_corrections"] > 0
                                                or any(t["n_corrections"] > max_corr for t in pt))),
        "clarify_first": float(bool(pt) and bool(pt[0]["asks_question"])),
        "over_correction": float(any(i["reason"] == "over_correction" for i in invalid)),
        "wrong_value": float(any(i["reason"] == "wrong_value" for i in invalid)),
        "target_recall_t0": float(r["target_retrieved_at_turn"] == 0),
        "target_recall_any": float(r["target_retrieved_at_turn"] is not None),
    }


def aggregate(records: list[dict], max_corr: int) -> dict:
    rows = [(r, per_sample(r, max_corr)) for r in records]

    def summarize(group):
        out = {"n": len(group)}
        keys = group[0][1].keys() if group else []
        for k in keys:
            if k == "misbound":
                continue
            vals = [m[k] for _, m in group if m[k] is not None]
            if vals:
                out[k] = round(mean(vals), 4)
                ci = _ci(vals)
                if ci:
                    out[k + "_ci95"] = ci
        return out

    report = {"overall": summarize(rows)}
    for key in ("task", "subtype", "belief_condition", "participant"):
        groups = defaultdict(list)
        for r, m in rows:
            groups[r.get(key, "")].append((r, m))
        report[f"by_{key}"] = {g: summarize(v) for g, v in sorted(groups.items())}
    # Headline over-correction: controls only.
    controls = [(r, m) for r, m in rows if r["belief_condition"] in ("correct", "incomplete")]
    # Spread across people (report mean +- sd, as per-person results in SuperMemory-VQA).
    pp = [v["primary_correct"] for v in report["by_participant"].values() if "primary_correct" in v]
    if len(pp) > 1:
        mu = mean(pp)
        report["primary_correct_across_participants"] = {
            "mean": round(mu, 4), "sd": round((sum((x - mu) ** 2 for x in pp) / (len(pp) - 1)) ** 0.5, 4)}
    report["over_correction_on_controls"] = summarize(controls).get("over_correction")
    return report
