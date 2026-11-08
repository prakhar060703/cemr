# CEMR Repository Knowledge Transfer

## 1. What this repository does

This project builds and evaluates grounded, multi-turn conversations about autobiographical memory from egocentric video data. Its preparation path uses video-derived fused timelines and Gemini-produced event annotations, alongside caption/utterance rows. The CEMR code in this repository indexes those derived sources, mines situations in which a person could confuse details or sources, creates a ground-truth record with evidence, constructs a deliberately imperfect user memory, plans a correction dialogue, simulates both sides, validates the result, and saves accepted dialogues as JSON and supervised fine-tuning (SFT) examples.

The central constraint is evidence grounding: video-derived event JSON is useful for finding candidate situations, but its Gemini-written descriptions are not accepted as verified truth by themselves. Facts and corrections are checked against caption/utterance rows, linked to events with evidence row IDs. Conversation generation stores the evidence row IDs used for each correction. The code also has an evaluation path in which a tested assistant retrieves memories and interacts with a deterministic belief-state controller plus a language-model user simulator.

The Python package is `cemr/`. The principal command entry point is `python -m cemr.cli`. The five benchmark task families are T1–T5, listed in §6.

## 2. Repository map and what to read first

| Path | Responsibility |
|---|---|
| `cemr/cli.py` | CLI commands: index, mine, generate, evaluate, stats, export-review, check-llm. |
| `cemr/config.py` | Dataclass defaults and JSON configuration loader. |
| `cemr/schemas.py` | Pydantic record formats for rows, episodes, candidates, facts, memories, plans, turns, conversations, and verifier results. |
| `cemr/index.py`, `cemr/index_report.py` | Input ingestion, normalization, persistent index and data-integrity report. |
| `cemr/crossview.py` | Links overlapping records of the same real-world event from different wearers; labels evidence visibility. |
| `cemr/miners/` | Deterministic candidate mining for T1–T5. |
| `cemr/stages/` | Stage 1 ground truth, Stage 2 false memory, Stage 3 plan, Stage 4 conversation. |
| `cemr/prompts/` | Prompt text and JSON contracts for generation, user simulation, planning, and verification. |
| `cemr/dialogue/` | Assistant/user agent behavior, belief-state controller, and self-play/evaluation loops. |
| `cemr/retrieval.py` | Dependency-free lexical BM25 episode retrieval. |
| `cemr/validation/` | Deterministic support/leak/structure checks and LLM verifier orchestration. |
| `cemr/llm_client.py`, `cemr/limits.py` | OpenAI-compatible client, JSON parsing, retries, Gemini token preflight and local quota accounting. |
| `cemr/evaluation/` | Test harness and metrics. |
| `data/captions/`, `data/events/` | Video-derived fused timeline rows and Gemini-created event annotations, organized by participant/day/hour. |
| `outputs/`, `cov/gemini/` | Checked-in/generated experiment snapshots, indexes and conversations. These are data products, not the canonical implementation. |
| `tests/` | Toy fixtures and tests for indexing, mining, conversation generation, quota guards, and split behavior. |

The repository has many repeated hour-level source and generated JSON files. Their schemas are described below; individual instance content varies by wearer, day, hour, event, and run. The video acquisition and event-extraction script supplied separately by the project owner is upstream preparation code, not part of the checked-in `cemr/` package. The executable behavior in this checkout is defined by its Python source and active config, not by every old generated example.

## 2.1 Where video enters, and how event annotations were created

The overall project is video-based even though `cemr/index.py` does not decode video. Video preparation happens upstream and produces two structured views that CEMR can consume: fused timeline JSON (dense timestamped caption/utterance entries) and top-5 event JSON (Gemini-generated event summaries linked back to timeline entries). Raw video is the source modality; the CEMR package reads derived representations of that video.

The separately supplied event-extraction script documents a hierarchical Gemini process over fused timeline files. It iterates participant/day directories, opens each `fused_timeline_<clip-id>.json`, formats entries as lines containing a clip row reference, start/end time, kind, optional speaker and text, and divides long timelines into windows of up to 500 lines with 60 lines of overlap. Gemini first extracts a few local candidate activities per window, preserving evidence IDs. A second Gemini call consolidates candidates for the clip into at most five macro-events, assigning IDs `<clip-id>_E01` through `<clip-id>_E05` and returning fields such as summary, people, activities, discussion, plans, decisions, objects/food, location, state changes, evidence IDs, uncertainties, developments, source coverage and boundary notes. The script requests JSON responses, retries across a configured model pool with backoff, and skips existing nonempty event outputs to support resuming.

Those outputs are saved as `top5_events_<clip-id>.json` beneath participant/day folders. This is the origin of the event annotations that later become `Episode` records. Event timestamps belong to the clip/timeline represented by that file; the clip identifier encodes the hour context used by CEMR to align event spans with caption rows. Evidence IDs bridge the event back to dense timeline entries. The extraction code checks JSON parsing and, in the supplied version, truncates an overlong event list to five; it does not independently establish that Gemini's semantic descriptions are true. A prompt asking Gemini to preserve evidence IDs is not a separate human audit.

The project owner's described workflow adds a possible video-grounding verification pass after event extraction: use an event's clip ID and start/end timestamps to retrieve the corresponding segment from the source video, then send both that video segment and the proposed activity description to Gemini and ask whether the description is supported by what is visible/audible. This is a sensible cross-check because timeline-based event extraction and direct video inspection provide different evidence views. For a true verification call, the prompt must include the candidate description and ask for a structured judgment (supported / partially supported / contradicted / unclear), observed evidence, and reasons or timestamps. Keep original event text and verifier result/model/version/clip interval as separate fields. Neither pasted script is that exact compare-a-candidate call: the video batch script produces an activity timeline from video, while the other script produces event proposals from fused timeline text.

There are two upstream scripts in the project materials. The chronological video-analysis script downloads MP4 clips from the Hugging Face EgoLife dataset, naturally sorts them, groups them in batches (default six), measures each clip duration with `ffprobe`, uploads every clip in a batch through Gemini Files API, and asks the configured model (default `gemini-3.1-flash-lite`) for one combined summary and observable activity timeline. Each activity has a 1-based `clip_index` and `start_seconds`/`end_seconds` relative to that clip; `batch_offset_seconds` records the clip's position in the combined batch timeline. It saves a `manifest.json` and `batch_XXXXX.json`, validates response structure and timestamp bounds, records API usage, and can resume completed batches. This script **generates** a video-derived summary/timeline; by itself it does not compare an already-extracted event description against the video.

The separate event-extraction script consumes existing fused timeline JSON and derives top-five event annotations from its text rows. Thus the materials describe both direct video analysis and timeline-based event extraction. A combined project flow can be video download/analysis → fused timeline and/or clip-level video activity timeline → event proposals → (if desired) a separate verification call that supplies each proposal plus its matching video segment and asks whether they agree → CEMR indexing/mining/dialogue generation. Do not call the chronological analysis step a verification pass unless the prompt includes the candidate event description being checked and returns an explicit comparison judgment. The `cemr/` package itself consumes captions and event JSON; neither upstream video script is implemented inside it.

## 3. End-to-end execution

The logical flow is:

```text
raw caption and event JSON
       │
       ├─ index.py: timestamped Row table + normalized Episode table
       │             crossview.py links matching views
       ├─ miners: deterministic Candidate records (T1…T5)
       ├─ Stage 1: evidence-backed GroundTruth facts (LLM + deterministic checks)
       ├─ Stage 2: FalseMemory (LLM + deterministic leak/support checks)
       ├─ Stage 3: DialoguePlan (LLM + plan validation)
       └─ Stage 4: user/assistant self-play, controller, retrieval, validation
                     ├─ accepted Conversation JSON
                     ├─ SFT JSONL
                     └─ rejected sample with stage/reasons
```

### 3.1 Indexing

`MemoryIndex.build()` reads files selected by `data.captions_glob` and `data.events_glob`. The checked-in config uses `data/captions/*/*/*.json` and `data/events/*/*/*.json`. The file basename contains an identifier such as `A1_JAKE_DAY1_11000000`; it encodes participant, day, and hour start. Participant IDs are canonicalized to lowercase names. The hour number is converted to a day-relative offset; row and episode times are represented as absolute seconds so records from multiple days can be ordered.

Each caption input list element becomes a `Row`. Its ID is the source file ID plus `#/` and the original zero-based list index, e.g. `A1_JAKE_DAY1_11000000#/8`. The index deliberately preserves the original position because event evidence IDs point into the original list. A row records file, wearer, original index, kind (`caption` or `utterance`), text, optional speaker, and absolute start/end. Caption citation residue matching `[cite: ...]` is stripped. Malformed timestamps are skipped and recorded; zero-length spans are padded by the parser and counted.

An event becomes an `Episode`. The event JSON usually has an `events` array; code also tolerates a top-level list and searches alternate list keys for event-like records. The episode stores ID, file, wearer, day/span, summary, people, activity, discussion, plans, decisions, objects/food, location, state changes, and `evidence_ids`. Slot text is normalized (lowercase, canonical person aliases and configured synonyms); compound locations keep the first component as canonical location and retain `location_raw`. The wearer is always inserted in `people`, flagged as `narrator_added` if missing. Group labels are discarded from the people set. These event annotations are retrieval/mining metadata, not sufficient evidence for a gold fact.

Evidence IDs support one-based source variants, partial/missing-hour IDs, case differences, and recovery against a row timestamp/span. Missing or malformed IDs are counted and the report flags dangling references. `index_report.py` additionally reports quality issues such as duplicate caption tracks, caption/utterance mismatch, missing evidence, unknown people, and missing speaker labels.

Cross-view clustering (`build_clusters`) associates episodes from different wearers when temporal overlap is high enough and the wearer annotations identify each other. A cluster ID means “likely same real-world event, different recording view”; it is not an assertion that all views contain identical evidence. Facts have `visible_to_user` set according to whether evidence comes from the target participant's own recording. Other participants' rows can help verify world-level details, but must not be shown to a simulated user as if they personally observed them.

`MemoryIndex.save()` writes `rows.jsonl`, `episodes.jsonl`, `skipped_times.jsonl`, and `index_meta.json`. JSONL means one JSON object per line, suitable for streaming/append. `index_meta.json` stores synonyms, people/aliases, file metadata, and aggregate stats. `MemoryIndex.load()` reconstructs the row and episode dictionaries from those files.

### 3.2 Candidate mining and data splits

Miners are deterministic Python, not LLM-generated. `Pipeline.mine()` writes one JSONL file per task under `output_dir/candidates/` plus `candidates/summary.json` and index stats. A `Candidate` records its ID/task, participant, split, target episode, lure episodes and/or rows, primary slot, true/lure values, subtype, query time, and task metadata.

Splits are assigned by day so recordings of the same day/event cannot leak across train/dev/test: current config says days 1–5 train, day 6 dev, day 7 test. Optional held-out people have only test-day examples labeled `test_heldout_person`; their other days are dropped. Mining also bounds candidates and near-duplicates (`max_candidates_per_task`, `max_per_cluster`). The assistant's retrieval bank is ego-centric: only episodes of the candidate participant that ended before the query time are available.

### 3.3 LLM stages and resuming

For each candidate, `Pipeline._one()` executes Stage 1→4. Accepted output from earlier stages is loaded by sample ID from `gt.jsonl`, `fm.jsonl`, and `plan.jsonl`, so reruns resume rather than spend calls regenerating the same artifacts. Accepted conversations and rejects are likewise skipped. Infrastructure failures are not content rejects: the run stops, saves completed work, and allows retry next time. Verifier/content failures can cause regeneration with the last failure feedback; the config default allows two regenerations after the initial attempt.

Generation performs a prefilter before the requested per-task limit: for concrete slots, the mined true value must be lexically supported by the target participant's own rows. This prevents spending model calls on candidates whose annotation value never appears in the user's captions. Derived slots (e.g. temporal order) are checked in later stages.

## 4. Input data and stored formats

### 4.1 Raw captions and utterances

Caption source files are JSON arrays, one file per participant/day/hour. The implementation reads each item’s `start`, `end`, `kind`, `text_en`, and optional `speaker`. `kind` is normally `caption` or `utterance`; utterances with speaker labels are important for T4. `start`/`end` accept the timestamp forms handled by `utils.parse_span` (including comma/dot decimal separators). These are relative to the hour file; the index adds the encoded day/hour offset.

Conceptual example (exact source key shape can vary only where parser permits):

```json
[
  {"start":"00:00:08,000", "end":"00:00:10,000", "kind":"caption",
   "text_en":"I put the cell phone on the table", "speaker":null},
  {"start":"00:00:11,000", "end":"00:00:12,000", "kind":"utterance",
   "text_en":"You left it there earlier", "speaker":"Alice"}
]
```

### 4.2 Video-derived event creation

At project level, events are an intermediate annotation layer derived from egocentric video. The fused timeline captures dense time-aligned descriptions and utterances for each clip. The supplied extraction script does not operate on pixels directly: it serializes each timeline entry with its source ID and timestamp, asks Gemini to identify local candidate activities, then asks Gemini to merge/select the top five clip-level events. This hierarchical chunking lets long timelines be processed in overlapping windows, while the consolidation pass restores a clip-level event view. It writes JSON with the schema CEMR expects. A separate video-verification pass can then inspect the actual timestamp-bounded segment alongside the proposed description. Because both extraction and verification use Gemini, the second pass is a useful modality-based consistency check, not fully independent validation; caption/utterance evidence and human review remain valuable, especially for ambiguous or fast actions.

The chronological video-analysis run writes a manifest of settings and selected clips, while each batch result records clip positions/durations, Gemini summary/activity output, and API usage. For a clip in a combined batch, `batch_offset_seconds` converts a clip-relative activity timestamp to the batch timeline by addition. That batch metadata is provenance for upstream video summarization; CEMR's hour-level row IDs and episode time conversion are a later alignment layer. Do not assume every event `start_time` is already a global batch timestamp: the event-extraction script uses timestamps from its fused timeline, and CEMR adds the hour-file offset.

### 4.3 Raw events

Event files are generally JSON objects with an `events` array. Every usable event requires `event_id`, start/end time, and can include the descriptive slots and `evidence_ids`. Evidence IDs point back to caption-array positions. An event is an indexing hypothesis; Stage 1 checks the relevant row text before accepting it as a factual claim.

### 4.4 Main Pydantic artifacts

The authoritative field definitions are in `cemr/schemas.py`.

| Artifact | Important fields / purpose |
|---|---|
| `Row` | `row_id`, `file_id`, `participant`, `idx`, `kind`, `text`, `speaker`, `abs_start`, `abs_end`. |
| `Episode` | `episode_id`, `file_id`, wearer/day/span, `summary`, structured slots, `evidence_ids`, normalization `flags`, `cluster_id`. `search_text()` combines annotation fields for retrieval. |
| `Candidate` | Mined target/lure and task label; `primary_slot`, `true_value`, `lure_value`, `query_time`, split and `meta`. |
| `Fact` | `fact_id`, `slot`, normalized `value`, first-person `statement`, nonempty `evidence_ids`, `visible_to_user`, and `source` (`caption` or `event_annotation`). |
| `GroundTruth` | Sample/task/narrator/target/lures, list of facts, `task_specific` task metadata. |
| `Edit` | Slot plus true/false values, error type/role, lure source, and true evidence. |
| `FalseMemory` | User's deliberately incorrect/incomplete/correct memory and question, edits, omissions, preserved gist. |
| `PlanStep` | Step number, intent, retrieval cues, selected evidence anchor, evidence type/IDs, correction attributes. |
| `AssistantTurn` | Text plus intent, cues, retrieved episode IDs, evidence IDs, correction labels. |
| `UserTurn` | Text, stance, slots updated, and current slot→value belief snapshot. |
| `Conversation` | ordered turns, task/participant/split/condition, final and gold belief maps, convergence, query-time annotations, flags. |
| `VerifierScores` | rubric score map, rationale, suggestions, and item-level flags. Threshold logic ignores explicitly advisory criteria. |

Enums and task labels: `T1_source_binding`, `T2_intention_action`, `T3_temporal_displacement`, `T4_experienced_vs_told`, `T5_counterfactual_blame`. Belief conditions: `misbound`, `incomplete`, `correct`, `unsupported`. Primary error taxonomy maps the five tasks to event blending, reality monitoring, temporal confusion, source confusion, and causal misattribution.

### 4.5 Output directory and persistence

`output_dir` is experiment-specific. The pipeline writes:

| File/folder | Format and meaning |
|---|---|
| `config.json` | Resolved dataclass config snapshot for reproducibility. |
| `index_stats.json` | Index summary copied into the run output. |
| `candidates/T1.jsonl` … `T5.jsonl` | One candidate object per line. |
| `gt.jsonl` | One wrapper per accepted Stage-1 result: `{ "candidate": ..., "gt": ... }`. |
| `fm.jsonl` | One `FalseMemory` object per accepted sample. |
| `plan.jsonl` | One `DialoguePlan` object per accepted sample. |
| `conv.json` | Combined accepted conversation JSON list. Despite old comments/help sometimes saying `.jsonl`, current code appends JSON objects to a JSON list file. |
| `conversations/<safe-sample-id>.json` | Pretty-printed individual copy of each accepted conversation. |
| `rejects.jsonl` | Sample ID/task/stage and failure report, one JSON object per rejected sample. API outages are not written as content rejects. |
| `sft/<split>.jsonl` | Training examples: JSON objects containing `messages`, normally system/user/assistant message records. |
| `llm_logs/{generator,user_sim,verifier}.jsonl` | One record per completed call: model, elapsed seconds, seed/scope, full messages, output, provider usage, and locally counted input tokens. Treat as sensitive: prompts and generated text are stored. |
| `funnel.json` | Counts by task for the most recent generation run, including skips, filters and stage totals. |
| `eval/<model>/<split>_<mode>/records.jsonl`, `metrics.json` | Evaluation traces/aggregate metrics. |
| `review_<split>.csv` | Human review export with dialogue and blank rating columns. |

JSONL is append-friendly and supports resumption. `conv.json` is a JSON array/list and `write_json` rewrites the formatted file after appending. File names are sanitized/truncated because some sample IDs contain slashes or plan text.

## 5. Conversation construction in depth

### Stage 1 — Ground truth

`s1_ground_truth.run()` builds a prompt from the target/lure episode context and relevant timestamped row evidence, asks the configured generator for structured facts, parses to a `GroundTruth`, and applies deterministic checks plus verifier judgments. Facts are meant to describe what the evidence supports, in first-person form. Concrete slot values need row support; values that exist only in event annotations are marked `event_annotation` and cannot be used as a user-facing correction. For cross-view facts, other-person views can establish world truth while `visible_to_user=false` prevents them from becoming user-observable memory. T2 distinguishes “no execution recorded” from “it did not happen”; T5 explicitly distinguishes observed sequence from causal necessity.

The verifier returns rubric scores and may identify individual unsupported facts. Repairable non-primary facts can be dropped and recorded in `task_specific`; unsupported primary answers reject the sample. The remaining item can be verified again after repair (`reverify_after_repair`).

### Stage 2 — False memory

`FalseMemory` expresses the test user's starting point, question, and desired correction(s). The prompt preserves the true event gist while changing a mined slot using a value from the lure source. It can also produce controls: `correct` (right memory to measure over-correction), `incomplete` (slot omitted, requiring recall), or `unsupported` (claim not justified either way). Deterministic leakage checks prevent the answer from appearing in the question/memory before the assistant earns the correction. `Edit.true_evidence` records which rows support correcting the slot.

### Stage 3 — Plan

The plan gives the assistant a sequence of intents (`retrieve_and_clarify`, `confirm`, `present_evidence`, `correct`, `suggest_cue`, `answer`, `summarize`) and specific retrieval cues/evidence. It constrains how many attributes may be corrected in a turn. Validation checks that cues are short keywords, that a first step does not prematurely cite evidence, that corrections use valid evidence and avoid revealing a future answer, and that anchors are not mechanically repeated. A plan is saved and reused on rerun.

### Stage 4 — Self-play

1. A `BeliefController` builds the gold belief from visible facts, then applies false edits or sets omitted slots to `(don't remember)`. The user simulator sees the false memory/question and produces the opening turn as JSON.
2. The candidate's query time is converted to a readable label. The retriever is restricted to this participant's own episodes ending by that time.
3. For each plan step, `PlannedAssistant` runs BM25 on the cues (or latest user text), then receives conversation history, step instructions, selected evidence rows with timestamps, allowed values, forbidden future values, retrieved episode context, notes, and the current query-time label. The generator returns a JSON response containing the assistant text (and any expected response fields).
4. Local checks detect future-value leakage and near-duplicate assistant text. Generation retries with rewrite instructions; if a forbidden value remains, redaction is attempted and diagnostic flags are retained. Row IDs must not be rendered to the user; readable times and quoted text are used instead.
5. A separate turn parser extracts proposed slot corrections/confirmations and whether evidence is cited. A value matcher compares asserted and gold values, using canonical/token overlap first and an LLM matching call when needed.
6. The deterministic controller decides whether a correction is valid. In generation mode, the assistant must cite an allowed evidence row for that slot. Wrong-value, unknown-slot, or correction of an already-correct belief is invalid. Valid primary errors resist once by default, then accept; secondary errors default to no resistance. Missing evidence never consumes resistance. The user simulator only verbalizes the controller decision; it does not choose whether to believe the assistant.
7. The loop finishes when the plan is exhausted, the combined assistant+user turn count hits `max_turns`, or the belief converges and a final response is produced. Repeated evidence after pushback is presented with a different wording and acknowledgment. Final deterministic checks ensure convergence, evidence grounding, no premature reveal, acceptable repetition, and other task-specific conditions. The LLM verifier adds quality/rubric checks.
8. Accepted conversation is appended to aggregate `conv.json`, written to its individual file, and converted to SFT messages under its split. Rejected samples record the failed stage and detailed report.

The user simulator never receives the gold facts as conversational knowledge; it receives the current controller belief and the assistant's last text/reaction. The deterministic controller owns the actual belief transition. This separation makes acceptance behavior reproducible even though the user wording is generated.

### Evaluation mode

`EvalAssistant` is the model under test, separate from the configured verifier/user simulator. It proposes retrieval cues, receives BM25 results from the same ego-centric memory bank, and writes a response. The verifier model parses its turn into claims; the same controller measures update, evidence use, resistance, and over-correction. The harness has single-turn and multi-turn modes and writes records/metrics by model, split, and mode. Evaluation is only meaningful when the tested assistant and frozen test examples are selected consistently.

## 6. The five mined task types

| ID | Cognitive error tested | Mining idea / ground-truth boundary |
|---|---|---|
| T1 source binding | Details from two similar episodes are blended. | Pair episodes with shared setting/activity and person, but differing concrete slot values. True and lure values must each be present in their own episode's rows. Candidate is one slot substitution. |
| T2 intention/action | A plan is remembered as completed, pending, or done by the wrong person. | Find a plan in the wearer's own rows; search a bounded later window. A lack of match in the user's own recording yields “no execution recorded,” not proof of non-occurrence. Other actors require their own first-person recording to verify. |
| T3 temporal displacement | Event is placed before/after a landmark or on the wrong day. | Select salient episodes as landmarks, require a minimum time gap and same/adjacent day. Evidence combines timestamps, landmark context, and target episode facts. |
| T4 experienced vs told | A report from another speaker is misremembered as personal witnessing. | Use a speaker-labeled utterance from the user's recording, exclude questions, link to an earlier same-day event by shared content words. Stage 1 determines report accuracy/conflict using other views; correct user-facing source is “told by [speaker]”. |
| T5 counterfactual blame | Temporal sequence is mistaken for causal necessity. | Pair a decision/state change and later outcome sharing an object, within the time window. Action and outcome must both occur in their own rows. Gold distinguishes observed order from whether the first event was necessary for the second. |

## 7. Models, token budgets, and why Gemini 3.1 Flash-Lite

### What the checked-in code actually selects

`configs/default.json` sets all three roles—`generator_model`, `user_sim_model`, and `verifier_model`—to `gemini-3.1-flash-lite`, using Google's OpenAI-compatible endpoint `https://generativelanguage.googleapis.com/v1beta/openai/`. It configures `max_tokens: 8192`, `temperature: 0.7`, verifier temperature `0.0`, concurrency `1`, a minimum 30-second request interval, 3 busy retries with 10-second exponential backoff, 300-second timeout, and API key lookup via `GEMINI_API_KEY`.

This is the active config's choice, not a rationale established by an experiment in code. The comments/config show a practical choice to use one available hosted model for all roles, with an explicit per-model RPM entry of 15 and one guarded code path. The repository does **not** contain a controlled benchmark proving Flash-Lite is more accurate, cheaper for this workload, or better than other candidate models. A defensible KT statement is: Flash-Lite is configured as a lightweight single-model provider for generation, user wording, and verification; the choice appears operational (availability/quota/cost/speed tradeoff), while empirical quality must be established by running the evaluation harness against alternatives. Do not present that inferred rationale as a measured result.

The source also allows different models/base URLs per role. `configs/llm_info.txt` describes historical local Qwen/vLLM configurations; it is not the current Gemini config. The checked-in help and README have older model/output names in places and must not override `configs/default.json` plus the implementation.

### What “input tokens” means here

For Gemini requests, `LLMClient.chat()` calls Google's `countTokens` endpoint before sending the generation request. `count_payload()` includes the complete system instruction and all user/assistant history in the provider's `generateContentRequest` representation. Thus input is not just the last user prompt: it includes system rules, prior turns, evidence rows, retrieval output, schemas/examples embedded in prompts, and JSON-generation repair feedback. The count is the native count returned by Google's endpoint, not a local word-to-token estimate.

The client fails closed if counting fails. It refuses to send a request if `counted_input + 1,024 framing margin + max_tokens` exceeds the configured model input limit. For the supported Gemini models in `cemr/limits.py`, the table is 1,048,576 input tokens and 65,536 maximum output tokens. The configured output cap 8,192 is below that maximum. These are context/model limits, not project RPM/TPM/RPD quotas.

### What “output tokens” means here

`max_tokens` is sent to the OpenAI-compatible chat completion request as the maximum completion budget for a single call. It is a cap, not the expected response length. Responses vary by stage: compact JSON for extraction/turn parsing; larger JSON for facts, memories, plans, verifier reports; natural-language JSON-wrapped turns for the user and assistant. A `finish_reason == "length"` is treated as infrastructure failure and not a valid sample. Provider `usage` (when returned) is stored in logs along with `counted_input_tokens`.

One conversation can require many API calls: Stage 1/verification, Stage 2, Stage 3, each assistant turn, each user reply, turn parsing, possible value-matching fallback, and verification/regeneration. The token cap applies separately to each call. `llm_logs/*.jsonl` is the source for actual per-call input/output usage; do not estimate a conversation's cost by multiplying the cap by turns.

### Quota accounting

`QuotaGuard` stores local usage in SQLite (`llm.model_quotas` and `quota_state_path`). Counters are per model and shared by processes using that DB. It can enforce configured RPM, TPM, and rolling 24-hour RPD limits; missing (`null`) limits are unknown and not enforced. Generation calls reserve counted input plus the 1,024-token margin; they do not reserve output tokens. Failed calls/retries count locally. Google `countTokens`, metadata calls, other applications/projects, and separate state files are not included. The database contains usage counters, not API credentials.

The selected config lists RPM 15 for `gemini-3.1-flash-lite`, but TPM and RPD are null. Hence this repository does not enforce those unknown project limits. Be aware that current `MODEL_LIMITS` recognizes exact model strings; aliases/new model IDs must be added with verified capacities before using the Gemini guarded path.

## 8. Configuration and code/config discrepancies

`Config.from_file()` starts from dataclass defaults and assigns keys from each section by attribute name. Unknown sections are assigned as extra attributes and do not override known dataclasses. Therefore the checked-in JSON key `"dialoge"` (typo) does not populate `Config.dialogue`; the dialogue settings under that key are ignored, and dataclass defaults take effect: max turns 14, max corrections 2, primary resistance 1, secondary resistance 0, retrieval top-k 5. This matters because apparent JSON settings cannot be assumed active.

The config also has `data.people`, day splits, source globs, output location, quotas, and mining settings. `Config.dump()` writes resolved dataclass state, which is preferable for reproducing a run. Current values should be checked in the emitted output config before interpreting a result.

The repository includes multiple generations of artifacts: root `outputs/` and `cov/gemini/` snapshots, old logs, and docs that mention older models such as Gemini 3.8 Flash / 3.1 Pro or local Qwen. `help.txt`, `README.md`, `SETUP.md`, `CHANGES.md`, and `configs/llm_info.txt` are not mutually consistent with the checked-in default config and current code. For runtime truth use, in order: current source; config after loading (and any typo/default behavior); saved `output_dir/config.json`; then data snapshots. Treat docs as historical operator notes where they conflict.

Other implementation details worth knowing:

- CLI `index` builds and persists the index without calling an LLM. `mine` is deterministic, though the CLI constructs configured clients before dispatching to it.
- CLI uses `output_dir` to place logs and products; existing output directories are resumable, so changing prompts/model without changing output directory can reuse stale stages.
- `.env` is ignored by Git; `GEMINI_API_KEY` should stay there or in the environment. `api_key` values beginning with `env:` are resolved by the client.
- `requirements.txt` includes OpenAI SDK compatibility, Pydantic, dotenv, httpx and pytest. The actual provider call uses the OpenAI client against Gemini's compatibility endpoint; countTokens uses direct HTTP.
- `monitor.ps1`, some config comments, and portions of old documentation refer to `cov/gemini`/old paths. Confirm the active output directory instead of relying on a monitor script's hard-coded path.

## 9. Validation and quality safeguards

Validation has deterministic and model-assisted parts. Deterministic checks include lexical support of concrete values, evidence ID validity, row/episode time/source consistency, allowed correction slots and evidence, primary/secondary limits per turn, repetition checks, forbidden-value leakage, conversation convergence, and task-specific evidence rules. A verifier model scores criteria and may flag individual facts. `verify.threshold` defaults to 0.6, with advisory criteria excluded from rejection as defined in `prompts/verifier.py`.

An accepted conversation is therefore a sample that passed the current automated rules; it is not a guarantee of perfect naturalness or a substitute for human review. `export-review` creates a CSV with human-scoring columns for groundedness, naturalness, premature reveal, keep/reject, and notes. The repository's tests exercise key invariants with a toy corpus and mock LLM; they do not validate the truth or quality of every checked-in generated record.

## 10. A practical reading/running path

For a new contributor, read `schemas.py`, then `index.py` and `crossview.py`, the miner corresponding to the target task, the four stage modules, the dialogue controller/self-play code, then `llm_client.py` and the prompts. For a new run, inspect the resolved config, build the index after source changes, mine candidates, generate a small pilot, inspect funnel/rejects/logs, export human review, then generate/evaluate held-out test data. Reuse an output directory only when deliberately resuming the same experiment.

Typical commands (check help for this checkout's arguments first):

```powershell
python -m cemr.cli index --config configs/default.json
python -m cemr.cli mine --config configs/default.json
python -m cemr.cli generate --config configs/default.json --tasks T1 T2 T3 T4 T5 --limit 2 --splits train
python -m cemr.cli stats --config configs/default.json
python -m cemr.cli export-review --config configs/default.json --split train
python -m cemr.cli evaluate --config configs/default.json --model gemini-3.1-flash-lite --split test --mode multi_turn --limit 2
```

Before a meaningful evaluation, generate accepted test examples and compare candidate assistant models on the same frozen set. Keep generator, user simulator, verifier, and evaluated assistant roles clear in reports; currently all configured generation roles share Flash-Lite, but the evaluated model is selected separately with `--model`.

## 11. Things not to infer from the repository

- The strings `true_value`, `lure_value`, or `gold_belief` are labels created from mining/evidence rules, not independent proof that the source corpus is correct.
- An event summary/people list is an annotation and may be wrong. Use linked row evidence and `Fact.source`/`visible_to_user`.
- “No execution recorded” means no matching record was found in the bounded memory search. It does not establish that the action never occurred in the world.
- A successful convergence means the controller accepted all required belief updates under the configured evidence rules. It does not alone establish dialogue quality.
- Gemini Flash-Lite's inclusion in config does not prove model-selection superiority. Measure it against alternatives on the same held-out data and report actual logs/metrics.
- The 8,192 output setting is a maximum budget per request, not a fixed generated length and not an entire conversation budget.
