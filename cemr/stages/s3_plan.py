"""Stage 3: dialogue plan (belief-state trajectory the assistant should follow)."""

from __future__ import annotations

from ..dialogue.controller import correction_targets
from ..prompts import plan as P
from ..schemas import Candidate, DialoguePlan, FalseMemory, GroundTruth
from ..validation.deterministic import (check_plan, check_plan_evidence, drop_repeated_steps,
                                         fix_evidence_types, repair_plan_leaks)
from ..validation.verifier import generate_verified
from .context import Ctx, with_feedback


def run(ctx: Ctx, cand: Candidate, gt: GroundTruth, fm: FalseMemory) -> tuple[DialoguePlan | None, dict]:
    max_corr = ctx.cfg.dialogue.max_corrections_per_turn
    hits = ctx.retriever_until(cand.query_time, cand.participant).search(fm.user_memory + " " + fm.user_question, k=3)
    hint = ", ".join(f"{ep.episode_id} ({s:.1f})" for ep, s in hits) or "none"
    targets = correction_targets(gt, fm)
    prompt = P.build(gt.sample_id, cand.task.value, fm.belief_condition.value, fm.user_memory,
                     fm.user_question, targets, [f for f in gt.facts if f.visible_to_user], hint, max_corr)
    system = P.SYSTEM.format(max_corr=max_corr)

    def gen(feedback):
        plan = ctx.generator.generate_json(system, with_feedback(prompt, feedback), schema=DialoguePlan)
        # The model kept re-adding a redundant step despite feedback; remove it instead.
        repair_plan_leaks(plan, targets, ctx.index, max_corr)
        drop_repeated_steps(plan)
        fix_evidence_types(plan, ctx.index, cand.task.value)
        return plan

    # Plans are checked deterministically only; the conversation verifier judges the result.
    return generate_verified(
        generate=gen,
        deterministic_check=lambda p: check_plan(p, fm, targets, max_corr)
        + check_plan_evidence(p, targets, ctx.index),
        verifier=None,
        kind="plan",
        to_payload=lambda p: p.model_dump(mode="json"),
        evidence_rows="",
        max_regenerations=ctx.cfg.verify.max_regenerations,
    )
