"""
dialogue/selfplay.py

Two-agent self-play loops.

  run_generation(...) -> Conversation   (Stage 4: training/benchmark dialogues)
  run_evaluation(...) -> dict           (interactive benchmark evaluation)

Both use the SAME controller and user simulator, so generated data and
evaluation share one definition of "the user updated their belief".
"""

from __future__ import annotations

from ..schemas import (BeliefCondition, Candidate, Conversation, DialoguePlan, FalseMemory,
                       GroundTruth, PlanStep, UserTurn)
from ..utils import now_label
from ..validation.deterministic import CONCRETE_SLOTS
from .agents import (EvalAssistant, PlannedAssistant, UserSimulator, dump,
                     make_value_matcher, parse_eval_turn)
from .controller import BeliefController, Claim


def _user_turn(user: UserSimulator, ctrl: BeliefController, assistant_text: str, reaction, turns) -> dict:
    slots = reaction.accepted or reaction.resisted or [i["slot"] for i in reaction.invalid]
    text = user.reply(ctrl.belief, assistant_text, reaction.kind, slots, turns)
    stance = {"resist": "resisting", "partial": "partial", "accept": "accepting",
              "converged": "converged"}.get(reaction.kind, "confident")
    return dump(UserTurn(text=text, stance=stance, belief_update=reaction.accepted,
                         belief_state=dict(ctrl.belief)))


###############################################################################
# Generation
###############################################################################


def _derived_reveals(slot: str, cand: Candidate) -> list[str]:
    """
    Words that give away a DERIVED answer before its correction step. The value-based
    leak check only knows concrete slots, so "Jake mentioned we still have the burgers"
    (the T4 answer: told by Jake) passed in the very first turn.
    """
    if slot == "knowledge_source" and cand.meta.get("reporter"):
        return [str(cand.meta["reporter"])]
    if slot == "day" and cand.true_value.startswith("day "):
        return [cand.true_value]
    return []


REPRESENT_NOTE = ("The user pushed back on your last point. Show the evidence again in a DIFFERENT "
                  "way: quote the row's words or give its time, and acknowledge their doubt. "
                  "Do not repeat your earlier wording.")
SUMMARY_NOTE = ("This is the summary. Restate only details already established above (corrected, "
                "confirmed or quoted in this conversation). Do not add any new detail, person or place.")


def run_generation(ctx, cand: Candidate, gt: GroundTruth, fm: FalseMemory, plan: DialoguePlan,
                   feedback: list[str] | None = None) -> Conversation:
    """`feedback`: reasons a previous attempt of this conversation was rejected."""
    d = ctx.cfg.dialogue
    ctrl = BeliefController(gt, fm, d.primary_resistance, d.secondary_resistance)
    now = now_label(cand.query_time)
    user = UserSimulator(ctx.user_sim, cand.participant.title(), now)
    assistant = PlannedAssistant(ctx.generator, ctx.index, d.max_corrections_per_turn, now=now)
    retriever = ctx.retriever_until(cand.query_time, cand.participant)
    exact = lambda slot, a, b: a.strip().lower() == b.strip().lower()  # noqa: E731  (plan values are gold)

    opening = dump(UserTurn(text=user.open(fm.user_memory, fm.user_question, fm.omitted_slots),
                            stance="confident", belief_state=dict(ctrl.belief)))
    # The question is asked at the query time: annotate it on the opening turn, so
    # "this morning" / "last night" in the text can be read against it.
    opening.update({"query_time": cand.query_time, "query_time_label": now})
    turns: list[dict] = [opening]
    flags: dict = {"leak_retries": 0, "repeated_steps": 0}
    i, repeats = 0, 0
    while i < len(plan.steps) and len(turns) < d.max_turns:
        step = plan.steps[i]
        if repeats:  # re-presenting evidence after a push-back
            step = PlanStep(**{**step.model_dump(), "intent": "present_evidence",
                               "evidence_ids": sorted({e for s in step.corrected_attributes
                                                       for e in ctrl.evidence.get(s, [])})})
        done = [s for s in ctrl.evidence if ctrl.belief[s] == ctrl.gold[s]]
        allowed = {s: ctrl.gold[s] for s in done + step.corrected_attributes}
        forbidden = [ctrl.gold[s] for s in ctrl.wrong_slots
                     if s not in step.corrected_attributes and s in CONCRETE_SLOTS]
        forbidden += [v for s in ctrl.wrong_slots if s not in step.corrected_attributes
                      for v in _derived_reveals(s, cand)]

        notes = []
        if repeats:
            notes.append(REPRESENT_NOTE)
        if step.intent == "summarize":
            notes.append(SUMMARY_NOTE)
        if feedback:
            notes.append("A previous version of this conversation was rejected for: "
                         + "; ".join(feedback)[:1200] + ". Avoid these problems.")
        a_turn, diag = assistant.step(turns, step, retriever, allowed, forbidden, d.retrieval_top_k, notes)
        flags["leak_retries"] += diag["leak_retries"]
        flags["repeat_retries"] = flags.get("repeat_retries", 0) + diag.get("repeat_retries", 0)
        if diag.get("redacted"):
            flags.setdefault("redacted", []).append({"turn": len(turns) + 1, "values": diag["redacted"]})
        if "leak_unresolved" in diag:
            flags["leak_unresolved"] = diag["leak_unresolved"]
            break
        turns.append(dump(a_turn))

        claims = [Claim(s, ctrl.gold[s], step.evidence_ids) for s in step.corrected_attributes]
        reaction = ctrl.react(claims, step.intent, exact, require_ids=True)
        turns.append(_user_turn(user, ctrl, a_turn.text, reaction, turns))

        if reaction.resisted and repeats < 2:
            repeats += 1
            flags["repeated_steps"] += 1
            continue
        repeats = 0
        i += 1
        if step.intent == "summarize":
            break

    return Conversation(
        sample_id=gt.sample_id, task=cand.task, participant=cand.participant, split=cand.split,
        belief_condition=fm.belief_condition, turns=turns,
        final_belief=dict(ctrl.belief), gold_belief=dict(ctrl.gold),
        converged=ctrl.converged, n_turns=len(turns),
        query_time=cand.query_time, query_time_label=now,
        delay_bucket=cand.meta.get("delay_bucket", ""),
        flags={**flags, "controller_log": ctrl.log, "subtype": cand.subtype},
    )


###############################################################################
# Evaluation (model under test plays the assistant)
###############################################################################


def run_evaluation(ctx, cand: Candidate, gt: GroundTruth, fm: FalseMemory, model_client,
                   opening_text: str | None = None, max_turns: int | None = None) -> dict:
    d = ctx.cfg.dialogue
    max_turns = max_turns or d.max_turns
    ctrl = BeliefController(gt, fm, d.primary_resistance, d.secondary_resistance)
    now = now_label(cand.query_time)
    user = UserSimulator(ctx.user_sim, cand.participant.title(), now)
    assistant = EvalAssistant(model_client, d.max_corrections_per_turn, now)
    parser = ctx.verifier.client if ctx.verifier else ctx.generator
    matcher = make_value_matcher(parser)
    retriever = ctx.retriever_until(cand.query_time, cand.participant)

    opening = opening_text or user.open(fm.user_memory, fm.user_question, fm.omitted_slots)
    turns: list[dict] = [{"role": "user", "text": opening}]
    target_retrieved_at: int | None = None
    per_turn = []
    first_correction_turn: int | None = None
    converged_at: int | None = None

    for t in range(max_turns // 2):
        a = assistant.step(turns, retriever, d.retrieval_top_k)
        if target_retrieved_at is None and cand.target_episode in a.retrieved_ids:
            target_retrieved_at = t
        parsed = parse_eval_turn(parser, ctrl.belief, a.text)
        corrections = parsed.get("corrections", []) or []
        claims = [Claim(c.get("slot", ""), c.get("asserted_value", ""), a.evidence_ids,
                        bool(parsed.get("cites_evidence"))) for c in corrections]
        if corrections and first_correction_turn is None:
            first_correction_turn = t
        reaction = ctrl.react(claims, a.intent, matcher, require_ids=False)
        per_turn.append({"turn": t, "n_corrections": len(corrections), "reaction": reaction.kind,
                         "invalid": reaction.invalid, "asks_question": parsed.get("asks_question", False)})
        turns.append({"role": "assistant", "text": a.text, "retrieved_ids": a.retrieved_ids})
        turns.append(_user_turn(user, ctrl, a.text, reaction, turns))
        if ctrl.converged and converged_at is None:
            converged_at = t
            if fm.belief_condition != BeliefCondition.MISBOUND:
                break
        elif converged_at is not None:
            break  # one extra assistant turn to answer after convergence

    slots = list(ctrl.evidence)
    return {
        "sample_id": gt.sample_id, "task": cand.task.value, "subtype": cand.subtype,
        "query_time": cand.query_time, "query_time_label": now,
        "delay_bucket": cand.meta.get("delay_bucket", ""),
        "participant": cand.participant, "n_views": cand.meta.get("n_views", 1),
        "split": cand.split, "belief_condition": fm.belief_condition.value,
        "final_belief": ctrl.belief, "gold_belief": ctrl.gold,
        "slot_correct": {s: ctrl.belief[s] == ctrl.gold[s] for s in slots},
        "primary_slot": cand.primary_slot,
        "primary_correct": ctrl.belief.get(cand.primary_slot) == ctrl.gold.get(cand.primary_slot),
        "converged": ctrl.converged, "converged_at_turn": converged_at,
        "target_retrieved_at_turn": target_retrieved_at,
        "first_correction_turn": first_correction_turn,
        "per_turn": per_turn, "turns": turns,
    }
