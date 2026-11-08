"""Stage 1 prompt: ground-truth extraction, grounded in raw rows."""

import json

SYSTEM = """You extract ground truth for an autobiographical-memory benchmark.

Rules:
- Use ONLY what is explicitly stated in the rows. Never infer feelings,
  intentions or details that are not written.
- Every fact must cite 1-5 row ids. Each row starts with its id in angle brackets,
  e.g. <A1_JAKE_DAY1_11000000#/8>, then its time [D1 11:09:43]. Write the id WITHOUT
  the brackets: "A1_JAKE_DAY1_11000000#/8". Cite only ids you can see; never list a
  range of ids.
- Lines starting with "==" are episode headers: "== <episode id> | <automatic summary>".
  Never cite an episode id, and never copy wording from the summary. The summary is
  not evidence. Take every value from the rows themselves.
- Write each value with the words used in the cited rows, not a paraphrase or
  summary. If a row says "scale", write "scale", not "electronic scale". Keep values
  short: 1-3 words for people, objects and location.
- A person may be cited when a cited row names them or they are the speaker of a
  cited utterance. Do not add people who appear in neither.
- Cite evidence from the episode the fact is about: facts about the target episode
  cite target-episode rows; facts about a related episode cite that episode's rows.
- Rows tagged [OTHER VIEW: name] were recorded by someone else's glasses: the narrator
  did NOT see or hear them. Use them only to decide what really happened (e.g. who did
  something, whether a report was true); every fact about the narrator's own experience
  must cite at least one untagged row.
- Each STATEMENT must stay close to its cited rows' wording and add nothing:
  * a caption supports what it says the narrator saw or did ("I got a bottle of
    detergent" -> "I got a bottle of detergent", not "I picked up detergent");
  * an utterance supports only that the speaker SAID it: "Tasha said to take the
    kitchen wipes", not "Tasha gave me kitchen wipes";
  * do not add "I saw", "I heard", "we were", "together" or a place unless a row says it;
  * a pronoun ("she", "he", "they") stays unresolved unless a row names the person.
- Day and time come ONLY from the [D.. HH:MM:SS] timestamps of the cited rows, never
  from episode ids or the summary.
- If something is absent from the rows (e.g. a plan never carried out), do not write it
  as a fact such as "I did not do X"; record it in task_specific, and phrase any
  completion fact as "no record of X in my recordings".
- Write each statement in first person from the narrator's point of view.
- One fact per slot value. Keep statements short and concrete. Omit a slot rather than
  guess it.
- Return ONLY a JSON object matching the requested schema."""

TASK_SPECIFIC_INSTRUCTIONS = {
    "T1_source_binding": 'task_specific: {}',
    "T2_intention_action": (
        'task_specific: {"execution_found": bool, "execution_seen_by_narrator": bool, '
        '"executor": str|null, "execution_evidence": [row ids], '
        '"wording_rule": "if the narrator\'s own recording shows no execution, the answer to the '
        "user is 'I can't find a record of it in your recordings' -- even if another view shows "
        "someone doing it; never say 'it didn't happen'\"}"
    ),
    "T3_temporal_displacement": (
        'task_specific: {"landmark": str, "true_relation": "before"|"after"|"other day", '
        '"contextual_cues": [{"cue": str, "evidence_ids": [row ids]}]}  '
        "-- contextual cues are details that date the target relative to the landmark "
        "(objects still present, state already changed, etc.)"
    ),
    "T4_experienced_vs_told": (
        'task_specific: {"reporter": str, "report_text": str, "report_row": str, '
        '"narrator_witnessed": bool, "report_consistent_with_record": bool, '
        '"conflicting_detail": str|null, "record_evidence": [row ids]}'
    ),
    "T5_counterfactual_blame": (
        'task_specific: {"action": str, "outcome": str, "consequences": ['
        '{"statement": str, "label": "observed"|"inferable"|"not_inferable", '
        '"evidence_ids": [row ids]}]}  -- "observed" = explicitly recorded; '
        '"inferable" = follows directly from recorded facts; "not_inferable" = would '
        "require knowing what would have happened otherwise"
    ),
}


def build(sample_id, task, narrator, candidate, rows_block, primary_slot, true_value):
    return f"""### TASK: ground_truth
### BENCHMARK TASK: {task}
Narrator: {narrator}
Target episode: {candidate.target_episode}
Related episodes: {', '.join(candidate.lure_episodes) or 'none'}
Miner hints (verify against rows, do not trust blindly): {json.dumps(candidate.meta, ensure_ascii=False)}

ROWS (captions describe what the narrator saw; utterances are speech):
{rows_block}

Produce:
{{
  "sample_id": "{sample_id}",
  "task": "{task}",
  "narrator": "{narrator}",
  "target_episode": "{candidate.target_episode}",
  "lure_episodes": {json.dumps(candidate.lure_episodes)},
  "facts": [{{"fact_id": "f1", "slot": str, "value": str, "statement": str, "evidence_ids": ["A1_JAKE_DAY1_11000000#/8"]}}],
  {TASK_SPECIFIC_INSTRUCTIONS[task]}
}}

Required: include one fact with slot="{primary_slot}" whose value is "{true_value}"
(if the rows contradict this value, set that fact's value to what the rows show and
add "miner_disagrees": true inside task_specific).
Other slots to cover only where the rows state them: people, objects, location,
activity, and any time cues."""