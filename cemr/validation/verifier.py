"""
validation/verifier.py

Closed-loop LLM verification (Planner -> Verifier -> regenerate with feedback),
the pattern used by SuperMemory-VQA. The verifier model should differ from
the generator.

Extensions:
  * advisory criteria (prompts.verifier.ADVISORY) are scored and stored but
    never reject a sample (e.g. ground-truth completeness);
  * an optional `repair` hook can fix an item from the verifier's per-item
    judgments (e.g. drop individual unsupported facts) instead of rejecting it;
  * an optional `fatal` predicate stops retrying when an error cannot change
    between attempts (e.g. the rows contradict the mined answer);
  * every attempt runs in its own seed scope, so retries are real resamples.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TypeVar

from pydantic import BaseModel

from ..llm_client import LLMUnavailable, seed_scope
from ..prompts import verifier as vp
from ..schemas import VerifierScores

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# repair(item, scores) -> (repaired item or None, criteria it resolved, notes)
RepairFn = Callable[[T, VerifierScores], "tuple[Optional[T], set[str], list[str]]"]


class LLMVerifier:
    def __init__(self, client, threshold: float):
        self.client = client
        self.threshold = threshold

    def score(self, kind: str, payload: dict, evidence_rows: str) -> VerifierScores:
        raw = self.client.generate_json(vp.SYSTEM, vp.build(kind, payload, evidence_rows))
        scores = {}
        for k, v in (raw.get("scores") or {}).items():
            try:
                scores[k] = float(v)
            except (TypeError, ValueError):
                scores[k] = 0.0
        # Missing criteria count as failures.
        for crit in vp.RUBRICS[kind]:
            scores.setdefault(crit, 0.0)
        flagged = [f for f in (raw.get("unsupported_facts") or []) if isinstance(f, dict) and f.get("fact_id")]
        return VerifierScores(scores=scores, rationale=raw.get("rationale") or {},
                              suggestions=raw.get("suggestions") or [], flagged=flagged)

    def advisory(self, kind: str) -> set[str]:
        return set(vp.ADVISORY.get(kind, ()))


def generate_verified(
    generate: Callable[[Optional[list[str]]], T],
    deterministic_check: Callable[[T], list[str]],
    verifier: Optional[LLMVerifier],
    kind: str,
    to_payload: Callable[[T], dict],
    evidence_rows: str,
    max_regenerations: int,
    repair: Optional[RepairFn] = None,
    fatal: Optional[Callable[[list[str]], bool]] = None,
    reverify_after_repair: bool = True,
) -> tuple[Optional[T], dict]:
    """
    Run generate -> deterministic checks -> LLM verifier (-> repair), feeding
    failures back into the next generation. Returns (item or None, report).
    """
    feedback: Optional[list[str]] = None
    report: dict = {"attempts": []}
    max_regenerations = max(0, min(3, max_regenerations))
    for attempt in range(max_regenerations + 1):
        log.info("%s generation attempt %d/%d; feedback=%s", kind, attempt + 1, max_regenerations + 1, feedback)
        with seed_scope(kind, "attempt", attempt):
            try:
                item = generate(feedback)
            except LLMUnavailable:
                raise                   # server problem: not this sample's fault, never a reject
            except Exception as e:  # malformed JSON after client retries, schema errors, ...
                log.warning("%s generation failed: %s", kind, e)
                report["attempts"].append({"error": f"generation failed: {e}"[:300]})
                feedback = [f"Previous output could not be parsed: {str(e)[:300]}"]
                continue
            errs = deterministic_check(item)
            entry: dict = {"deterministic_errors": errs}
            if errs:
                log.warning("%s deterministic rejection: %s", kind, errs)
                report["attempts"].append(entry)
                if fatal and fatal(errs):
                    report["stopped_early"] = "error cannot change between attempts"
                    break
                feedback = errs
                continue
            if verifier is not None:
                vs = verifier.score(kind, to_payload(item), evidence_rows)
                entry["verifier"] = vs.model_dump()
                resolved: set[str] = set()
                if repair is not None:
                    fixed, resolved, notes = repair(item, vs)
                    if notes:
                        log.info("%s verifier repair: %s", kind, notes)
                        entry["repair"] = notes
                    if fixed is None:
                        report["attempts"].append(entry)
                        feedback = notes + [s for s in vs.suggestions][:6]
                        continue
                    item = fixed
                    if resolved and reverify_after_repair:
                        # The repair trusted the verifier's list of bad items; check the
                        # repaired item once more, with no further repair allowed.
                        vs = verifier.score(kind, to_payload(item), evidence_rows)
                        entry["verifier_after_repair"] = vs.model_dump()
                        resolved = set()
                failing = vs.failing(verifier.threshold, ignore=verifier.advisory(kind) | resolved)
                if failing:
                    log.warning("%s VERIFIER REJECT: scores=%s rationale=%s suggestions=%s",
                                kind, failing, vs.rationale, vs.suggestions)
                    report["attempts"].append(entry)
                    feedback = [f"{k}={v:.2f}: {vs.rationale.get(k, '')}" for k, v in failing.items()] \
                        + vs.suggestions[:6]
                    continue
            report["attempts"].append(entry)
            report["accepted_on_attempt"] = attempt
            log.info("%s accepted on attempt %d", kind, attempt + 1)
            return item, report
    report["rejected"] = True
    log.warning("%s rejected; no further regeneration", kind)
    return None, report
