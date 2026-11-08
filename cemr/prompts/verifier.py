"""
Verifier / judge prompts (run on a DIFFERENT model from the generator).
Scores are produced per criterion with a rationale written before the score
(rationale-first scoring, as in SuperMemory-VQA's verifier).
"""

import json

SYSTEM = """You are a strict quality verifier for a research benchmark. For each criterion,
write a one-sentence rationale FIRST, then a score in [0,1]. Be conservative: if you
are unsure, score low. Return ONLY JSON:
{"rationale": {criterion: str}, "scores": {criterion: float}, "suggestions": [str]}
(plus any extra keys the task asks for)."""

RUBRICS = {
    "ground_truth": {
        "grounded": "Every fact's value AND statement is explicitly supported by its cited rows "
                    "(no inference, no added 'I saw'/'I heard'/'we were' unless a row says so).",
        "complete": "The key people/objects/location/activity stated in the rows are covered "
                    "(informational: recorded, never used to reject).",
        "task_label": "task_specific fields are correct given the rows and the task notes.",
    },
    "false_memory": {
        "plausible": "The errors are ones a real person could make by blending in details from the "
                     "real related episode they were borrowed from (see borrowed_from).",
        "gist_preserved": "The memory still clearly refers to the same episode.",
        "no_leak": "The memory does not reveal any corrected true value.",
        "natural": "It sounds like something a person would actually say (no awkward phrases).",
        "single_primary": "The primary (mined) error is the main one; any other changed details are "
                          "small substitutions of a person, object or place.",
    },
    # CORRECT / INCOMPLETE control conditions: the memory must contain NO error, so the
    # error rubric above does not apply (it rejected every control sample).
    "false_memory_control": {
        "no_false_detail": "The memory contains no detail that contradicts the true facts "
                           "(for a vague memory: the listed slots are left out, nothing is made up).",
        "gist_preserved": "The memory clearly refers to the same episode.",
        "natural": "It sounds like something a person would actually say.",
    },
    "conversation": {
        "grounded": "Every assistant claim is supported by the cited rows; no person is named as "
                    "doing something unless a row names them; the summary adds no new details.",
        "no_premature_reveal": "The assistant never states a detail before the plan reaches it.",
        "evidence_before_acceptance": "The user only accepts a correction after evidence was shown.",
        "natural": "Both sides sound like real people; no repeated turns or robotic phrasing.",
        "reconstruction": "The dialogue reads as memory reconstruction, not quiz-style QA.",
    },
}

# Criteria that are scored and stored but never reject a sample.
ADVISORY = {
    "ground_truth": {"complete"},
}

# Shared reading rules, so the verifier judges evidence the same way the generator was told to.
EVIDENCE_NOTES = """Reading the evidence:
- Each row is "<row id> [D<day> HH:MM:SS] (caption|utterance) speaker: text". The day and
  time in brackets ARE evidence: a fact about the day or time is supported by the
  timestamps of its cited rows.
- A caption describes what the narrator (the wearer, "I") saw or did. An utterance is
  speech by the named speaker; it supports "X said ...", not that it happened.
- Rows tagged [OTHER VIEW: name] were recorded by someone else; the narrator did not see them."""

TASK_NOTES = {
    "T1_source_binding": "Facts describe the target episode; the related episode is where a "
                         "blended detail came from.",
    "T2_intention_action": "The target is an intention (a plan or decision). completion_status "
                           "may be 'no execution recorded': that fact is supported when the cited "
                           "row states the intention and no shown row of the narrator's own "
                           "recording shows it being done. It must be phrased as 'no record', "
                           "never as 'it did not happen'.",
    "T3_temporal_displacement": "The day/temporal_order fact is supported by the row timestamps. "
                                "The related episode is the landmark it is dated against.",
    "T4_experienced_vs_told": "The report is an utterance in which someone else tells the narrator "
                              "about the target event. knowledge_source 'told by X' is supported by "
                              "that utterance row.",
    "T5_counterfactual_blame": "The target episode holds the narrator's ACTION; the related episode "
                               "holds the later OUTCOME. Outcome facts should cite the related "
                               "episode's rows. 'Necessity not established' is correct unless a row "
                               "states the causal link.",
}

FALSE_MEMORY_NOTES = """About false memories:
- Each false value is BORROWED from a real related episode (listed in borrowed_from with
  the rows where it appears). That is by design: memory errors blend real experiences.
  Do not penalise a false value for being absent from the target episode.
- A memory may change 1-4 details (one primary, the rest small substitutions)."""

FLAG_INSTRUCTION = """
Also list every fact whose value or statement is NOT explicitly supported by its cited
rows, with a short reason (an empty list if all are supported):
"unsupported_facts": [{"fact_id": "f2", "reason": "row only says Tasha told me to take them"}]"""


def build(kind, payload, evidence_rows):
    rubric = RUBRICS[kind]
    task = payload.get("task") if isinstance(payload, dict) else None
    notes = TASK_NOTES.get(task, "")
    return f"""### TASK: verify_{kind}
CRITERIA:
{json.dumps(rubric, indent=1)}
{EVIDENCE_NOTES}
{("Task notes: " + notes) if notes else ""}
{FALSE_MEMORY_NOTES if kind == "false_memory" else ""}

EVIDENCE ROWS:
{evidence_rows}

ITEM TO VERIFY:
{json.dumps(payload, indent=1, ensure_ascii=False)}
{FLAG_INSTRUCTION if kind == "ground_truth" else ""}"""


TURN_PARSER_SYSTEM = """You annotate one assistant turn from a memory-assistant dialogue.
Given the user's current beliefs (slot -> value) and the assistant's text, report which
slots the assistant says are WRONG and what value it asserts instead, which slots it
CONFIRMS, and whether it cites concrete evidence (a quote, a timestamp, a described
scene). Return ONLY JSON:
{"corrections": [{"slot": str, "asserted_value": str}], "confirmations": [str],
 "cites_evidence": bool, "asks_question": bool}"""


def turn_parser(belief_state, assistant_text):
    return f"""### TASK: parse_turn
USER BELIEFS: {json.dumps(belief_state, ensure_ascii=False)}
ASSISTANT TURN: "{assistant_text}\""""


VALUE_MATCH_SYSTEM = """Decide whether two short descriptions refer to the same value
(e.g. "cell phone" vs "my phone" -> true; "hard drive" vs "phone" -> false).
Return ONLY JSON: {"same": bool}"""


def value_match(slot, a, b):
    return f"### TASK: value_match\nslot: {slot}\nA: {a}\nB: {b}"
