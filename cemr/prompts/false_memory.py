"""Stage 2 prompt: false-memory writing, constrained by pre-selected edits."""

import json

SYSTEM = """You write a person's (possibly mistaken) memory of their own day, for a
memory-assistant benchmark.

Rules:
- The memory is first person, casual, 1-3 sentences, like something a person would say.
- It must contain EXACTLY the false values you are given (use their words), keep the
  preserved gist, and must NOT mention any of the true values that were changed --
  not even alongside the false value ("food and detergent" leaks "detergent").
- Never add details that are not in the facts or the edits (no extra people, places or objects).
- Some values are LABELS, not words to copy: completion_status ("no execution recorded"),
  causal_link ("sequence observed; necessity not established" / "'X' caused 'Y'"),
  knowledge_source ("told by X" / "witnessed myself"), temporal_order, day. Express them
  in natural words: "I'm sure I already did it", "that's why it broke", "I saw it myself".
- The question is what the person actually wants to know; it presupposes the memory.
- Return ONLY JSON."""

CONDITION_RULES = {
    "misbound": "The memory contains the listed false values (primary + chosen secondary).",
    "incomplete": "The memory is vague: it omits the listed slots entirely, but says nothing false.",
    "correct": "The memory is accurate: use only true facts. There are no edits.",
    "unsupported": "The memory asserts the listed claim, which the record neither supports nor refutes.",
}


def build(sample_id, task, condition, facts, primary_edit, secondary_pool, omitted_slots, n_secondary,
          forbidden=(), now=None, episode_time=None):
    when = (f"The person recalls this at {now}; the episode was recorded {episode_time}. Any time "
            f"reference must be relative to now ('yesterday', 'this morning', 'on Tuesday').\n") \
        if now and episode_time else ""
    return f"""### TASK: false_memory
### BENCHMARK TASK: {task}
{when}
Belief condition: {condition} -- {CONDITION_RULES[condition]}

TRUE FACTS:
{json.dumps([f.model_dump() for f in facts], indent=1, ensure_ascii=False)}

PRIMARY EDIT (fixed, copy it verbatim into "edits"):
{json.dumps(primary_edit, ensure_ascii=False) if primary_edit else "none"}

SECONDARY EDIT OPTIONS (choose exactly {n_secondary}; each false value comes from a real
lure record -- do not invent new values; copy slot/true/false/lure_source/true_evidence):
{json.dumps(secondary_pool, indent=1, ensure_ascii=False) if secondary_pool else "none"}

Slots to omit (incomplete condition only): {omitted_slots or "none"}

NEVER write these words in user_memory or user_question: {json.dumps(list(forbidden), ensure_ascii=False) if forbidden else "none"}
For every secondary edit you choose, never write its "true" value either.

Return:
{{
  "sample_id": "{sample_id}",
  "belief_condition": "{condition}",
  "user_memory": str,
  "user_question": str,
  "edits": [{{"slot","true","false","error_type","role","lure_source","true_evidence"}}],
  "omitted_slots": {json.dumps(omitted_slots)},
  "preserved_gist": [str]   // 2-4 short phrases shared by memory and truth
}}"""
