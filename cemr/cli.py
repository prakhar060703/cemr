"""
cli.py

    python -m cemr.cli index     --config configs/default.json     # Stage 0 Part 1 (no LLM)
    python -m cemr.cli mine      --config configs/default.json
    python -m cemr.cli generate  --config configs/default.json --tasks T1 T2 --limit 50
    python -m cemr.cli evaluate  --config configs/default.json --model Qwen/Qwen2.5-7B-Instruct \
                                 --split test --mode multi_turn
    python -m cemr.cli stats     --config configs/default.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict

from .config import Config
from .llm_client import LLMClient, build_clients
from .pipeline import Pipeline, build_index
from .utils import read_conversations, read_jsonl


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cemr")
    ap.add_argument("command", choices=["check-llm", "index", "mine", "generate", "evaluate", "stats", "export-review"])
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--tasks", nargs="*", default=None, help="T1 T2 T3 T4 T5")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="model under test (evaluate)")
    ap.add_argument("--model-base-url", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="multi_turn", choices=["multi_turn", "single_turn"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.from_file(args.config) if args.config else Config()

    if args.command == "stats":
        return _stats(cfg)
    if args.command == "export-review":
        return _export_review(cfg, args.split)
    if args.command == "index":
        return _index(cfg)
    if args.command == "check-llm":
        return _check_llm(cfg)

    index = build_index(cfg)
    clients = build_clients(cfg, log_dir=f"{cfg.output_dir}/llm_logs")
    pipe = Pipeline(cfg, index, clients)

    if args.command == "mine":
        res = pipe.mine(args.tasks)
        print(json.dumps({k: len(v) for k, v in res.items()}, indent=2))
    elif args.command == "generate":
        funnel = pipe.generate(args.tasks, args.limit, args.splits)
        print(json.dumps({k: dict(v) for k, v in funnel.items()}, indent=2))
    elif args.command == "evaluate":
        if not args.model:
            ap.error("--model is required for evaluate")
        model = LLMClient(args.model, args.model_base_url or cfg.llm.base_url, cfg.llm.api_key,
                          temperature=0.0, seed=cfg.llm.seed,
                          max_retries=cfg.llm.max_retries,
                          enable_thinking=cfg.llm.enable_thinking,
                          max_tokens=cfg.llm.max_tokens or None,
                          timeout=cfg.llm.request_timeout or None,
                          reasoning_effort=cfg.llm.reasoning_effort or None,
                          busy_retries=cfg.llm.busy_retries,
                          busy_backoff=cfg.llm.busy_backoff,
                          model_quotas=cfg.llm.model_quotas,
                          quota_state_path=cfg.llm.quota_state_path,
                          log_path=f"{cfg.output_dir}/llm_logs/eval_{args.model.replace('/', '_')}.jsonl")
        from .evaluation.harness import evaluate
        report = evaluate(pipe, model, args.model, args.split, args.mode, args.limit)
        print(json.dumps(report["overall"], indent=2))


_FACT_PREFIX = re.compile(r"^f\d+:\s*")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"|\[[^\]]*\]")


def _reason_key(msg: str) -> str:
    """'f2: value 'tasha' not lexically supported' -> "value '…' not lexically supported"."""
    msg = _FACT_PREFIX.sub("", str(msg))
    return _QUOTED.sub("'…'", msg)[:90]


def _final_reasons(report: dict, threshold: float | None, advisory=()) -> list[str]:
    """Every reason the LAST attempt failed: rule errors, low verifier scores, crashes."""
    attempts = report.get("attempts") or []
    if not attempts:
        for k in ("error", "exception", "reason"):
            if report.get(k):
                return [f"{k}: {str(report[k])[:80]}"]
        return [f"no attempts recorded (report keys: {sorted(report)})"]
    last = attempts[-1]
    out = [_reason_key(e) for e in (last.get("deterministic_errors") or [])]
    if report.get("stopped_early"):
        out.append(f"stopped early: {report['stopped_early']}")
    for note in last.get("repair") or []:
        if note.startswith("primary fact"):
            out.append("verifier: primary fact unsupported")
    ver = last.get("verifier_after_repair") or last.get("verifier") or {}
    tag = "verifier (after dropping flagged facts)" if last.get("verifier_after_repair") else "verifier"
    for name, score in (ver.get("scores") or {}).items():
        if name in advisory:
            continue
        if threshold is None or score < threshold:
            out.append(f"{tag}: {name} score below threshold")
    for k in ("error", "exception"):
        if last.get(k):
            out.append(f"{k}: {str(last[k])[:80]}")
    return out or ["rejected with no recorded reason"]


def _stats(cfg: Config) -> None:
    """Rejection reasons per stage and task -- report these in the paper."""
    from .prompts.verifier import ADVISORY
    threshold = getattr(getattr(cfg, "verify", None), "threshold", None)
    stages, reasons = Counter(), defaultdict(Counter)
    for r in read_jsonl(f"{cfg.output_dir}/rejects.jsonl"):
        stages[(r["task"], r["stage"])] += 1
        advisory = ADVISORY.get(r["stage"], ())
        for reason in set(_final_reasons(r.get("report") or {}, threshold, advisory)):
            reasons[(r["task"], r["stage"])][reason] += 1
    print("Rejections by (task, stage):")
    for k, v in sorted(stages.items()):
        print(f"  {k}: {v}")
        for reason, n in reasons[k].most_common(6):
            print(f"      {n:4d}  {reason}")


def _check_llm(cfg: Config) -> None:
    """Ping every configured model and confirm it returns parseable JSON."""
    ok = True
    for role, client in build_clients(cfg).items():
        try:
            out = client.generate_json("Return only JSON.", 'Return {"ok": true, "role": "%s"}' % role)
            print(f"  OK   {role:9s} {client.model} @ {client.client.base_url} -> {out}")
        except Exception as e:  # connection refused, wrong model name, bad JSON ...
            ok = False
            print(f"  FAIL {role:9s} {client.model} @ {client.client.base_url}: {str(e)[:200]}")
    if not ok:
        raise SystemExit(1)


def _index(cfg: Config) -> None:
    """Stage 0 Part 1: build + save the week index and a data-quality report. No LLM."""
    from .index_report import build_report, print_summary
    from .utils import write_json
    idx = build_index(cfg, rebuild=True)
    idx.save(cfg.data.index_dir)
    report = build_report(idx)
    write_json(f"{cfg.data.index_dir}/index_report.json", report)
    print_summary(report)
    print(f"\nSaved rows.jsonl, episodes.jsonl, index_meta.json, index_report.json -> {cfg.data.index_dir}")


def _export_review(cfg: Config, split: str) -> None:
    """One CSV row per conversation for human review (review ALL test, sample train)."""
    import csv
    fms = {o["sample_id"]: o for o in read_jsonl(f"{cfg.output_dir}/fm.jsonl")}
    path = f"{cfg.output_dir}/review_{split}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "task", "condition", "user_memory", "gold_belief", "dialogue",
                    "grounded(0/1)", "natural(0/1)", "no_premature_reveal(0/1)", "keep(0/1)", "notes"])
        for c in read_conversations(cfg.output_dir):
            if c["split"] != split:
                continue
            dialogue = "\n".join(f"{t['role'].upper()}: {t['text']}" +
                                 (f"  [{t.get('intent')} | ev={t.get('evidence_ids')}]" if t["role"] == "assistant" else "")
                                 for t in c["turns"])
            w.writerow([c["sample_id"], c["task"], c["belief_condition"],
                        fms.get(c["sample_id"], {}).get("user_memory", ""),
                        json.dumps(c["gold_belief"], ensure_ascii=False), dialogue, "", "", "", "", ""])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
