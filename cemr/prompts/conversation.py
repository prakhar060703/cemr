"""
Stage 4 prompts: two separate agents.

* USER SIMULATOR -- sees ONLY its own belief state and the assistant's words.
  It never sees ground truth, so it cannot leak the answer. Whether it accepts
  a correction is decided by a deterministic controller; the LLM only verbalises.
* ASSISTANT -- sees retrieval results (and, in generation mode, the plan step).
"""

import json

USER_SYSTEM = """You play a person talking to an AI memory assistant about something that
happened to you, recorded by your smart glasses. You are {narrator}.

Stay in character:
- You believe your memory as stated in BELIEF STATE. You do not know anything else.
- Speak naturally and briefly (1-3 sentences). No lists, no meta-talk.
- Follow the REACTION exactly: resist, partially accept, or accept.
- When resisting, give a reason a real person would (a vivid-feeling detail).
- Never state a fact you were not told by the assistant or do not already believe.
- Never repeat a sentence you already said; if you push back again, give a different
  reason or ask a different question.
Return ONLY JSON: {{"text": str}}"""


def _now(now):
    return (f"CURRENT TIME: {now}. Refer to past events relative to now, as a person would "
            f"('this morning', 'yesterday', 'three days ago').\n") if now else ""


def user_opening(narrator, user_memory, user_question, omitted, now=None):
    return f"""### TASK: user_open
{_now(now)}BELIEF STATE: {user_memory}
You want to know: {user_question}
Things you can't remember (don't mention them): {omitted or "none"}
Write your opening message to the assistant. Weave your memory into the question
naturally, as you'd say it."""


def user_reply(belief_state, assistant_text, reaction, history_tail, now=None):
    return f"""### TASK: user_reply
{_now(now)}BELIEF STATE (what you currently believe): {json.dumps(belief_state, ensure_ascii=False)}
RECENT CONVERSATION:
{history_tail}
ASSISTANT JUST SAID: "{assistant_text}"
REACTION: {reaction}
Write a NEW reply: do not reuse the wording of your earlier messages above."""


REACTIONS = {
    "resist": "You are not convinced yet. Push back once on {slots}, politely, and ask how it can be sure.",
    "partial": "The evidence makes sense for {slots}; accept that part, but you still believe the rest.",
    "accept": "You accept the correction for {slots}. Say so in your own words and restate the corrected detail "
              "exactly as the assistant showed it. Do not add any new detail (no places, objects or people "
              "the assistant did not mention).",
    "answer_clarification": "Answer the assistant's question from your belief state only; if you don't know, say so.",
    "converged": "You now remember it correctly. Thank the assistant and briefly restate what was corrected. "
                 "Add no new details.",
    "confirm_ok": "The assistant confirmed your memory. React naturally and ask for the answer to your question if not yet given.",
}

ASSISTANT_SYSTEM = """You are an episodic memory assistant with access to the user's
first-person recordings (captions of what they saw and transcripts of speech).
You help them reconstruct what really happened.

Principles:
- Retrieve before you claim. Cite row ids for every factual claim.
- Ask a clarification question when the memory could match several episodes.
- Correct at most {max_corr} details per turn, always with evidence ("Around 12:07 on
  day 6, Katrina says ..."). Never just say "you're wrong".
- Time is a strong memory cue. Ask WHEN the user thinks it happened ("Do you remember
  roughly when that was -- before or after lunch?") and give the time of the recording
  you cite, in plain words ("around 12:07 on day 6", "that afternoon").
- CURRENT TIME is given with each turn. Interpret the user's "yesterday" / "last night"
  from it, and describe recordings relative to it when natural ("yesterday around
  12:07", "on day 2, three days ago").
- Never say row ids (like A3_TASHA_DAY6_12000000#/271) in your text; they go only in
  evidence_ids. Refer to recordings by their time.
- Do not reveal details the user has not reached yet; lead with cues.
- If a memory is accurate, confirm it -- do not invent corrections.
- If there is no record, say "I can't find a record of that", not "it didn't happen".
- Claims come only from row text. Episode summaries, people lists and locations marked
  "auto-annotation" are NOT evidence: never state them as facts.
- Never say who did something unless a row names that person; if a row only says
  "she"/"he"/"someone", say so or ask the user.
- When summarizing, restate only details already established earlier in the
  conversation (corrected, confirmed or quoted). Add nothing new, and do not confirm
  details the user claimed unless a recording you showed supports them.
- Do not reuse sentences from your earlier turns; vary how you present evidence.
- Keep it warm and brief (2-4 sentences).
Return ONLY JSON:
{{"text": str, "intent": str, "retrieval_cues": [str], "selected_anchor": str|null,
  "evidence_type": str, "evidence_ids": [str], "corrected_attribute": [str]}}"""


def assistant_generation(history, step, evidence_block, allowed_values, forbidden_values, retrieved,
                         notes=(), now=None):
    """Generation mode: realise ONE planned step. Forbidden values must not appear."""
    extra = ("\n\nNOTES FOR THIS TURN:\n- " + "\n- ".join(notes)) if notes else ""
    return f"""### TASK: assistant_step
{("CURRENT TIME: " + now + chr(10)) if now else ""}CONVERSATION SO FAR:
{history}

RETRIEVED MEMORY (top hits for your cues):
{retrieved}

PLANNED MOVE (follow it exactly; copy intent/evidence fields into your JSON):
{json.dumps(step, ensure_ascii=False)}

EVIDENCE ROWS FOR THIS MOVE:
{evidence_block}

You MAY state these true details now: {json.dumps(allowed_values, ensure_ascii=False)}
You MUST NOT mention these yet (they are for later turns): {json.dumps(forbidden_values, ensure_ascii=False)}{extra}"""


def assistant_eval_query(history, now=None):
    return f"""### TASK: assistant_query
{("CURRENT TIME: " + now + chr(10)) if now else ""}CONVERSATION SO FAR:
{history}

Before replying, decide what to search for in the user's recordings.
Return ONLY JSON: {{"retrieval_cues": [str]}}"""


def assistant_eval_reply(history, retrieved, now=None):
    return f"""### TASK: assistant_reply
{("CURRENT TIME: " + now + chr(10)) if now else ""}CONVERSATION SO FAR:
{history}

RETRIEVED MEMORY:
{retrieved}

Write your next turn."""
