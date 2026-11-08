"""
index_report.py  (Stage 0, Part 1 -- data audit)

Answers, before any mining: is the data good enough for each task?

  per file     rows, speaker coverage, #episodes, how much of the hour the
               episodes cover, gaps between episodes
  episodes     overlaps, bad spans, narrator added, dangling evidence
  vocabulary   location / people / object values -> spot spelling variants
               to add to the synonym map
  readiness    per task: what the data supports and what blocks it
"""

from __future__ import annotations

from collections import Counter

from .index import MemoryIndex
from .utils import fmt_abs

GAP_SECONDS = 60
DUP_JACCARD = 0.5      # two hour files share >= half their caption timestamps -> copied track
MIN_CU_OVERLAP = 0.5   # caption and speech tracks overlap < half the shorter one -> mismatched


def build_report(idx: MemoryIndex) -> dict:
    per_file = {}
    fingerprints: dict[str, set] = {}   # file_id -> caption start offsets within the hour
    all_gaps = []
    overlaps = []
    for file_id, meta in sorted(idx.files.items()):
        rows = idx.rows_by_file.get(file_id, [])
        eps = sorted((e for e in idx.episodes.values() if e.file_id == file_id), key=lambda e: e.abs_start)
        utt = [r for r in rows if r.kind == "utterance"]
        if rows:
            span_start, span_end = min(r.abs_start for r in rows), max(r.abs_end for r in rows)
        else:  # no caption file: assume the full hour
            span_start = (meta["day"] - 1) * 86400 + meta["offset"]
            span_end = span_start + 3600
        # Union of episode intervals, clipped to the recorded span (overlaps counted once).
        covered, cur_s, cur_e = 0.0, None, None
        for e in eps:
            s0, e0 = max(e.abs_start, span_start), min(e.abs_end, span_end)
            if e0 <= s0:
                continue
            if cur_e is None or s0 > cur_e:
                covered += (cur_e - cur_s) if cur_e is not None else 0.0
                cur_s, cur_e = s0, e0
            else:
                cur_e = max(cur_e, e0)
        covered += (cur_e - cur_s) if cur_e is not None else 0.0
        gaps, prev_end = [], span_start
        for e in eps:
            if e.abs_start - prev_end > GAP_SECONDS:
                gaps.append([fmt_abs(prev_end), fmt_abs(e.abs_start), round(e.abs_start - prev_end)])
            if e.abs_start < prev_end and prev_end != span_start:
                overlaps.append(e.episode_id)
            prev_end = max(prev_end, e.abs_end)
        if span_end - prev_end > GAP_SECONDS:
            gaps.append([fmt_abs(prev_end), fmt_abs(span_end), round(span_end - prev_end)])
        all_gaps += gaps
        caps = [r for r in rows if r.kind == "caption"]
        base = (meta["day"] - 1) * 86400 + meta["offset"]
        cap_span = (min(r.abs_start for r in caps), max(r.abs_end for r in caps)) if caps else None
        utt_span = (min(r.abs_start for r in utt), max(r.abs_end for r in utt)) if utt else None
        # Do the caption track and the speech track describe the same stretch of time?
        cu_overlap = None
        if cap_span and utt_span:
            inter = max(0.0, min(cap_span[1], utt_span[1]) - max(cap_span[0], utt_span[0]))
            cu_overlap = round(inter / max(min(cap_span[1] - cap_span[0], utt_span[1] - utt_span[0]), 1), 3)
        fingerprints[file_id] = {round(r.abs_start - base, 2) for r in caps}
        per_file[file_id] = {
            "caption_span": [fmt_abs(cap_span[0]), fmt_abs(cap_span[1])] if cap_span else None,
            "utterance_span": [fmt_abs(utt_span[0]), fmt_abs(utt_span[1])] if utt_span else None,
            "caption_utterance_overlap": cu_overlap,
            "has_captions": meta["has_captions"],
            "has_events": meta["has_events"],
            "rows": len(rows),
            "utterances": len(utt),
            "speaker_coverage": round(sum(1 for r in utt if r.speaker) / len(utt), 3) if utt else None,
            "episodes": len(eps),
            "episode_coverage": round(covered / max(span_end - span_start, 1), 3),
            "gaps_over_60s": gaps,
        }

    # Integrity: a caption track copied into another hour keeps its exact timestamps even if
    # the text was re-paraphrased, so identical start offsets are a robust duplicate signal.
    dup_tracks = []
    fids = [f for f in sorted(fingerprints) if len(fingerprints[f]) >= 20]
    for i, a in enumerate(fids):
        for b in fids[i + 1:]:
            if idx.files[a]["participant"] != idx.files[b]["participant"]:
                continue
            fa, fb = fingerprints[a], fingerprints[b]
            j = len(fa & fb) / len(fa | fb)
            if j >= DUP_JACCARD:
                dup_tracks.append([a, b, round(j, 3)])
    mismatched = [f for f, v in per_file.items()
                  if v["caption_utterance_overlap"] is not None and v["caption_utterance_overlap"] < MIN_CU_OVERLAP]

    eps = list(idx.episodes.values())
    vocab = {
        "location": Counter(e.location for e in eps).most_common(),
        "location_raw_variants": sorted({(e.location_raw, e.location) for e in eps if e.location_raw
                                         and e.location_raw.lower() != e.location}),
        "people": Counter(p for e in eps for p in e.people).most_common(),
        "objects_top50": Counter(o for e in eps for o in e.objects).most_common(50),
    }
    flags = Counter(f for e in eps for f in e.flags)
    s = idx.stats
    utt_total = s["rows_utterance"]
    speaker_cov = s["utt_with_speaker"] / utt_total if utt_total else 0.0
    mean_cov = (sum(v["episode_coverage"] for v in per_file.values()) / len(per_file)) if per_file else 0.0
    days = sorted({m["day"] for m in idx.files.values()})

    readiness = {
        "T1_source_binding": _ok(len(eps) >= 2, "needs >= 2 episodes per day with shared people/place"),
        "T2_intention_action": _ok(mean_cov >= 0.7 and utt_total > 0,
                                   f"needs dense episode coverage (mean {mean_cov:.0%}) and utterances "
                                   "to find future intentions; with sparse coverage 'no record' is unsafe"),
        "T3_temporal_displacement": _ok(len(eps) >= 10, "needs episodes spread over hours/days for landmarks"),
        "T4_experienced_vs_told": _ok(speaker_cov >= 0.5,
                                      f"needs speaker-labelled utterances (currently {speaker_cov:.0%})"),
        "T5_counterfactual_blame": _ok(mean_cov >= 0.7, "needs dense coverage to link action -> outcome"),
    }
    # Per participant: each wearer is a separate user with their own memory.
    per_participant = {}
    for p in idx.participants:
        files = {f: v for f, v in per_file.items() if idx.files[f]["participant"] == p}
        utt_p = [r for f in files for r in idx.rows_by_file.get(f, []) if r.kind == "utterance"]
        per_participant[p] = {
            "hour_files": len(files),
            "days": sorted({idx.files[f]["day"] for f in files}),
            "episodes": sum(v["episodes"] for v in files.values()),
            "speaker_coverage": round(sum(1 for r in utt_p if r.speaker) / len(utt_p), 3) if utt_p else None,
            "mean_episode_coverage": round(sum(v["episode_coverage"] for v in files.values()) / len(files), 3)
            if files else None,
        }
    # Cross-view event clusters (same real event seen by several wearers).
    cl = Counter(e.cluster_id for e in eps if e.cluster_id)
    views_hist = Counter(len({idx.episodes[x].participant for x in eps_ids})
                         for eps_ids in _group(eps).values())
    clusters = {"clusters": len(cl), "multi_view": sum(c for n, c in views_hist.items() if n > 1),
                "views_per_cluster": dict(sorted(views_hist.items()))}

    return {
        "per_participant": per_participant,
        "cross_view": clusters,
        "summary": {
            "days": days, "hour_files": len(idx.files), "episodes": len(eps),
            "captions": s["rows_caption"], "utterances": utt_total,
            "speaker_coverage": round(speaker_cov, 3),
            "mean_episode_coverage": round(mean_cov, 3),
            "dangling_evidence_ids": s["evidence_ids_dangling"],
            "files_without_captions": [f for f, m in idx.files.items() if not m["has_captions"]],
            "files_without_events": [f for f, m in idx.files.items() if not m["has_events"]],
            "episode_flags": dict(flags),
            "overlapping_episodes": overlaps,
            "total_gap_minutes": round(sum(g[2] for g in all_gaps) / 60, 1),
        },
        "integrity": {
            "duplicate_caption_tracks": dup_tracks,
            "caption_utterance_mismatch": mismatched,
        },
        "readiness": readiness,
        "per_file": per_file,
        "vocabulary": vocab,
    }


def _group(eps) -> dict:
    g: dict = {}
    for e in eps:
        g.setdefault(e.cluster_id or e.episode_id, []).append(e.episode_id)
    return g


def _ok(cond: bool, note: str) -> dict:
    return {"ready": bool(cond), "note": note}


def print_summary(report: dict) -> None:
    s = report["summary"]
    print(f"Days {s['days']} | {s['hour_files']} hour files | {s['episodes']} episodes | "
          f"{s['captions']} captions | {s['utterances']} utterances")
    print(f"Speaker coverage: {s['speaker_coverage']:.0%} | mean episode coverage of each hour: "
          f"{s['mean_episode_coverage']:.0%} | gaps: {s['total_gap_minutes']} min")
    print(f"Dangling evidence ids: {s['dangling_evidence_ids']} | flags: {s['episode_flags']}")
    if s["files_without_captions"]:
        print(f"Files without captions ({len(s['files_without_captions'])}): {s['files_without_captions'][:5]} ...")
    print("Per participant:")
    for p, v in report["per_participant"].items():
        sc = "n/a" if v["speaker_coverage"] is None else f"{v['speaker_coverage']:.0%}"
        print(f"  {p:9s} files={v['hour_files']:3d} days={v['days']} episodes={v['episodes']:4d} "
              f"speaker={sc} coverage={v['mean_episode_coverage']:.0%}")
    g = report.get("integrity", {})
    for a, b, j in g.get("duplicate_caption_tracks", []):
        print(f"  !! duplicate caption track: {a} == {b} (timestamp Jaccard {j:.2f}); one hour's captions were copied")
    for f in g.get("caption_utterance_mismatch", []):
        v = report["per_file"][f]
        print(f"  !! captions and speech cover different times in {f}: captions {v['caption_span']}, "
              f"speech {v['utterance_span']} (overlap {v['caption_utterance_overlap']:.0%})")
    x = report["cross_view"]
    print(f"Cross-view: {x['clusters']} event clusters, {x['multi_view']} seen by >1 wearer, "
          f"views/cluster={x['views_per_cluster']}")
    print("Readiness:")
    for task, r in report["readiness"].items():
        print(f"  {'OK ' if r['ready'] else 'NO '} {task:26s} {r['note']}")
