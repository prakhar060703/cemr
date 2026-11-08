# CEMR pipeline changes: fixes from the pilot run

Replace your `cemr/` folder with the one in this zip. Nothing in `data/` or your
config needs to change: every new setting has a default.

## Before rerunning

1. Rename `outputs/rejects.jsonl` (e.g. `rejects_pilot1.jsonl`) so the pilot's
   rejected samples are tried again with the fixes.
2. Run `mine` again: the miners changed (T1, T2, T4, T5), so the candidate
   files need regenerating.
3. Then run `generate` with a small `--limit`, followed by `stats`.

Samples already accepted in `gt.jsonl`, `fm.jsonl` or `plan.jsonl` are reused,
not regenerated. Delete those lines to force a fresh run of a stage.

## What changed

### Speed and reliability
- **Per-call seeds.** One fixed seed made every retry of an identical prompt
  return an identical answer. That is why the 3 conversation attempts failed
  the same way and turns repeated. Each call now derives its seed from the base
  seed, the prompt and the attempt number: runs are still reproducible, but
  retries are real resamples.
- **Thinking mode off** (`enable_thinking: false`), sent through vLLM's chat
  template. This should make the 27B verifier much faster.
- **Parallel samples** (`concurrency: 4`). Set it to 1 for sequential runs.
- **Server failures are never recorded as rejects.** A dropped tunnel or a 404
  stops that sample without writing it to rejects, and it is retried next run.
  After 10 crashes the run stops.
- **`max_tokens: 4096`** per call, and the simulated user runs at
  **temperature 0.9** (`user_sim_temperature`).

### Verifier (most ground-truth rejections)
- The verifier now lists which facts are unsupported. Those facts are dropped
  and recorded in `task_specific.dropped_facts`. The sample is rejected only if
  the **answer fact** itself is unsupported. The repaired ground truth is then
  verified once more (`verify.reverify_after_repair`).
- **Completeness is recorded but can no longer reject a sample.**
- The verifier gets per-task notes (e.g. T5: the outcome lives in the related
  episode; T2: "no record" is supported by absence) and the same evidence
  reading rules as the generator.
- If the model says the rows contradict the mined answer, the sample stops
  after 1 attempt instead of 3.
- The sample id (which contains a truncated T2 plan text) is no longer shown
  to the verifier.

### Ground truth
- Every row now shows its time, e.g. `[D2 20:03:15]`, so day and time facts are
  checkable. They must come from these timestamps, never from episode ids.
- Statement rules: stay close to the row's wording; speech supports only "X
  said ..."; no added "I saw", "I heard" or "we were"; never resolve "she/he"
  to a name; absence is not written as a fact ("no record of X" instead).
- For derived answers (completion_status, day, temporal_order,
  knowledge_source, causal_link) the answer fact is found by its slot.

### False memory
- The prompt names the words the memory must not contain (the leaked
  "detergent" case), including when mentioned next to the false value.

### Plan (new checks)
- Retrieval cues must be 1-5 keywords, not sentences or questions, and must not
  reveal a true value before it is corrected.
- Step 1 cites no evidence.
- The same anchor or evidence may not repeat in back-to-back steps unless the
  second step makes the correction.

### Conversation
- Assistant and user turns that nearly copy an earlier turn are rewritten, and
  a conversation with repeated turns is rejected.
- Rejection reasons from one attempt are fed into the next.
- New rules: claims come only from row text (episode summaries and people lists
  are labelled "auto-annotation, not evidence"); nobody is named as doing
  something unless a row names them; the summary restates only what was
  established.
- The verifier always sees every row the conversation cites.

### Miners (run `mine` again)
- **T1:** the true and lure values must both be stated in their own episode's
  captions, and generic values (food, stuff, snacks, items, ...) are skipped.
- **T2:** the intention must be stated in the plan episode's rows; "done by X"
  is used only when X's own recording shows them doing it.
- **T4:** questions ("did you ...?") are no longer used as reports; the report
  must share at least 2 content words with the event it is about.
- **T5:** the action must be stated in the action episode and the outcome in
  the outcome episode.

### Stats
- `stats` lists every reason per task and stage, including verifier scores,
  early stops and primary-fact failures; advisory scores are skipped.

## New settings (all optional)

```
"llm":    {"user_sim_temperature": 0.9, "enable_thinking": false,
           "max_tokens": 4096, "concurrency": 4},
"verify": {"reverify_after_repair": true}
```

If the 27B server rejects `enable_thinking`, set it to `null`. If a request
fails with "maximum context length", lower `max_tokens`.
