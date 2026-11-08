"""Stage 3 prompt: dialogue planning (a sequence of belief-state moves)."""

import json

SYSTEM = """You plan how a memory assistant helps a person reconstruct an episode from
their own recordings. You are writing the assistant's strategy, not dialogue text.

Rules:
- Step 1 always retrieves and asks a clarification question using a distinctive
  anchor detail; it corrects nothing and cites no evidence yet (evidence_ids: []).
  When timing matters (when something happened, what the person was told vs saw,
  the order of events), step 1 or 2 asks WHEN the person thinks it happened, and a
  later step uses the recording's time as evidence (evidence_type "timestamp").
- At least one step before the first correction asks the person something
  (retrieve_and_clarify or suggest_cue), so they try to recall before being told.
- retrieval_cues are 1-4 SEARCH KEYWORDS each ("fridge", "fake eyelashes", "Tasha"),
  never sentences or questions, and never a true value that has not been corrected yet.
- Do not reuse the same anchor or the same evidence rows in two consecutive steps,
  except when the second step is the correction that the first step's evidence supports.
- Correct at most {max_corr} attributes per step, and only with evidence whose ids are
  listed as allowed for that attribute.
- Correct the primary error last or second-to-last, after rapport is built.
- Anchors are row ids of distinctive details (a quote, a visible object) that help the
  person recall; an anchor must not state an uncorrected true value outright.
- End with one "summarize" step. For a correct memory: confirm, then answer.
- Return ONLY JSON."""


def build(sample_id, task, condition, user_memory, user_question, edits, facts, retrieval_hint, max_corr):
    allowed = {e["slot"]: e["true_evidence"] for e in edits}
    return f"""### TASK: plan
### BENCHMARK TASK: {task}
Belief condition: {condition}
User memory: "{user_memory}"
User question: "{user_question}"

EDITS TO CORRECT (slot: true <- false):
{json.dumps([{k: e[k] for k in ("slot", "true", "false", "role")} for e in edits], ensure_ascii=False)}

ALLOWED EVIDENCE PER SLOT: {json.dumps(allowed)}
ALL FACTS: {json.dumps([f.model_dump() for f in facts], ensure_ascii=False)}
RETRIEVAL HINT (episodes the retriever surfaces for the user's words): {retrieval_hint}

Return:
{{
  "sample_id": "{sample_id}",
  "correction_order": [slot, ...],
  "steps": [{{
    "step": int,
    "intent": "retrieve_and_clarify"|"confirm"|"present_evidence"|"correct"|"suggest_cue"|"answer"|"summarize",
    "retrieval_cues": [str],
    "selected_anchor": row_id|null,
    "evidence_type": "utterance_quote"|"caption_detail"|"timestamp"|"landmark_event"|"absence_of_record"|"reported_speech"|"none",
    "evidence_ids": [row ids],
    "corrected_attributes": [slot]   // <= {max_corr}
  }}]
}}"""
