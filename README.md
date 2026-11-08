# CEMR pipeline: Conversational Episodic Memory Reconstruction

The default configuration uses Gemini 3.1 Flash-Lite for generation, user
simulation, and verification. Put `GEMINI_API_KEY=your-key` in the repo-root
`.env`, then run:

```bash
pip install -r requirements.txt
python -m cemr.cli check-llm
python -m cemr.cli mine
python -m cemr.cli generate --limit 2
python -m cemr.cli stats
```

New samples go to `outputs/gemini`; the existing index at `outputs/index` is
reused. Commands default to `configs/default.json`. Local vLLM instructions
below describe the alternative self-hosted setup.

Once test-split conversations have been generated, compare assistants with
`python -m cemr.cli evaluate --model gemini-3.8-flash --split test --mode multi_turn`.
Repeat with `gemini-3.1-flash-lite` and `gemini-3.1-pro-preview`, keeping the
dataset and simulator fixed. Use `--mode single_turn` for the baseline.

This package generates and evaluates conversations in which a user holds a
wrong or incomplete memory of their own day (from EgoLife), and an assistant
corrects it step by step using evidence from the supplied fused timelines.
The CEMR pipeline reads fused timeline JSON and event JSON; it does not load,
download, or inspect raw video.

Five tasks, one primary error each:

| Task | The user gets wrong | Lure (where the false value comes from) |
|---|---|---|
| T1 source binding | **which** episode a detail came from | a similar real episode |
| T2 intention vs action | **whether** a planned action was done / by whom | the plan mention vs. its execution |
| T3 temporal displacement | **when** (before/after a landmark, which day) | a salient landmark episode |
| T4 experienced vs told | **how they know** (witnessed vs. told) | another person's speaker-labelled utterance |
| T5 counterfactual blame | **what follows** without an action | an action → outcome pair |

## The pipeline

The inputs to this repository are prepared before CEMR runs. Caption files
contain fused, timestamped timeline entries (captions and utterances). Event
files are generated separately from those timeline entries: a Gemini
event-extraction script processes the timeline in overlapping chunks, gathers
candidate activities, and consolidates them into up to five events per clip,
preserving evidence IDs that point back to timeline rows. These descriptions
are model-generated annotations, not verified ground truth. This repository
does not use raw video to check them; Stage 1 grounds facts in the timeline
rows cited by the events.

```
Fused timeline JSON ──> Gemini event extraction ──> event JSON
        │                                         │
        └──────────────────┬──────────────────────┘
                           v
Stage 0  index + mine   (code)      candidates/{T1..T5}.jsonl     lure mined from indexed data
Stage 1  ground truth   (LLM+check) gt.jsonl                      checked against timeline rows
Stage 2  false memory   (LLM+check) fm.jsonl                      primary edit fixed by miner
Stage 3  dialogue plan  (LLM+check) plan.jsonl                    belief-state moves
Stage 4  self-play      (2 agents)  conv.jsonl, sft/{split}.jsonl user sim never sees gold
         every stage: deterministic checks -> LLM verifier -> regenerate with feedback
         rejects.jsonl records every dropped sample and why
```

Design choices that come from prior benchmarks:

- **Over-generate, then filter with a verifier loop.** This follows EgoLifeQA,
  which kept under 1% of its candidates, and SuperMemory-VQA, which scores
  each item per criterion and rejects anything below τ = 0.6.
- **One model across all roles by default.** Generator, verifier, and user
  simulator use Gemini 3.1 Flash-Lite (`configs/default.json`); configs can
  assign different models or endpoints per role when needed.
- **A deterministic controller decides the user simulator's reactions.** A
  correction is accepted only if it targets the right slot, states the true
  value, and cites evidence. The primary error is resisted once before it is
  accepted. The LLM only puts the decided reaction into words. The design is
  inspired by Clarinet's oracle user.
- **Matched controls.** Some samples have a correct belief (about 12%) and
  some an incomplete one (about 8%). They measure over-correction, in the
  spirit of MemArena's matched grounded foils.
- **No cross-split leakage.** Splits are by day, and the memory bank is
  causal: it contains only episodes that ended before the query time.

## Multiple wearers (EgoLife A1–A6)

Each hour file belongs to one person: `A3_TASHA_DAY2_14000000` is Tasha's first-person
recording, and in it "I" means Tasha. The pipeline treats each wearer as a separate user.

| Rule | Where it lives |
|---|---|
| The narrator of each file comes from its name; that person is always added to `people` | `index.py` |
| Names and speaker labels are mapped to one list of six people (`data.people`, `data.people_aliases`) | `index.py` |
| **The assistant only searches the user's own recordings, up to the query time** | `index.bank_until(t, participant)`, `Ctx.retriever_until` |
| The same event seen by several wearers is linked into an **event cluster** (their times overlap and each lists the other as present) | `crossview.py` |
| At most `max_per_cluster` candidates per task per event, to avoid near-duplicates | `miners/base.py` |
| Lures come from the user's own episodes. For T4, only utterances the user actually heard are used | `miners/*` |
| Other people's recordings can **check** the gold answer, e.g. T2 "Alice did it" from Alice's own first-person caption, or whether a T4 report was accurate | `miners/t2_intention.py`, `t4_told.py`, Stage 1 |
| After Stage 1, a fact is marked visible to the user only if its evidence is in their own recording (done by code, not by the LLM). Only visible facts can be corrections | `crossview.mark_visibility`, `check_ground_truth` |
| Splits are by day for **all** wearers; `heldout_participants` adds a `test_heldout_person` split made from test days only | `index.split_of` |
| Results are broken down per participant, plus the mean and SD across people | `evaluation/metrics.py` |

T2 "executed_unseen": if another wearer recorded doing the planned action but the user's own
recording shows nothing, the gold answer is still "I can't find a record of it in your
recordings". Who actually did it is kept in `meta`.

## Data layout

```
data/captions/A1_JAKE_DAY1_11000000.json   fused timeline rows: [{start, end, kind, text_en, speaker}, ...]
data/events/A1_JAKE_DAY1_11000000.json     Gemini event annotations: {"events": [{event_id, start_time, ..., evidence_ids}]}
```

The event-extraction script takes the fused timeline rows as input and writes
the event annotations; it does not take raw video as input. Event descriptions
and event timestamps are Gemini-generated. Evidence IDs connect each event to
the fused timeline rows from which it was summarized, allowing later stages to
check support. No separate video-based verification is performed by this
repository. Evidence IDs use the form `"<file_id>#/<row index>"`, which is the
format your event files already use. File names must contain
`DAY<d>_<HHMMSSFF>`; this is how local times are converted to absolute
times across the week.

## Run

```bash
pip install -r requirements.txt
# start vLLM servers (generator / verifier can share one server if both models are loaded)
python -m cemr.cli index    --config configs/default.json            # Stage 0 part 1: index + clusters + report
python -m cemr.cli mine     --config configs/default.json            # Stage 0 part 2: candidates
python -m cemr.cli generate --config configs/default.json --limit 20 # pilot: 20 per task
python -m cemr.cli stats    --config configs/default.json            # rejection reasons
python -m cemr.cli export-review --config configs/default.json --split test   # human review CSV
python -m cemr.cli evaluate --config configs/default.json \
       --model Qwen/Qwen2.5-7B-Instruct --split test --mode multi_turn
python -m cemr.cli evaluate ... --mode single_turn                   # collapsed baseline
pytest -q                                                            # plumbing tests (mock LLM)
```

Every stage can be resumed: finished and rejected samples are skipped when
you run a command again.

## Recommended order of work

1. **Run `mine` first and read `index_stats.json`.** Check the share of
   utterances that have a speaker label. T4 depends entirely on it.
2. **Run a pilot:** `generate --limit 20`, then `stats`. Read about 10
   conversations per task, tighten the prompts, and repeat until the
   rejection reasons stabilise.
3. **Scale up** to the full set.
4. **Human review.** Review every test-split conversation and a sample of
   train with `export-review`, and report the kept fraction.
5. **Validate the simulator.** For about 30 test samples, have a person play
   the user while following the same reaction rules. Compare model scores
   against the simulated-user scores.

## What to report

- **Stage funnel.** Per task, how many samples survive each stage
  (`funnel.json`) and the main rejection reasons (`stats`).
- **Metrics** (`evaluation/metrics.py`):
  - primary-slot correction
  - slot accuracy
  - convergence
  - turns to reconstruction
  - premature reveal
  - clarify-first
  - over-correction on controls
  - wrong-value rate
  - target recall
- **Breakdowns.** Give each metric by task, subtype and belief condition,
  with bootstrap 95% CIs.
- **Single-turn vs multi-turn gap.** This result shows the benchmark
  measures more than false-premise detection.

## Known limitations to state

- **T2 "never executed" is one person's view.** The gold wording is "no
  record found", not "it didn't happen".
- **T4 conflicting reports are rare.** Report the `report_accurate` and
  `report_conflicts` subtypes separately.
- **T5 gold is a label set, not a single answer.** Each consequence is
  labelled observed, inferable or not_inferable, so score T5 with that
  rubric.
- **The event JSON is LLM output derived from fused timelines.** Stage 1 facts must cite timeline rows, and
  concrete values are checked lexically against those rows.
- **The tests only check plumbing.** `tests/` runs with a mock LLM, so it
  says nothing about output quality. Quality comes from the pilot and human
  review.
