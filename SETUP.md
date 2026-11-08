# Setting up and running the CEMR pipeline

## 0. What you need

| Need | Notes |
|---|---|
| Python 3.10+ | |
| vLLM on GPUs | Two models: a **generator** and a different **verifier** |
| EgoLife captions + events | One JSON file per participant per hour, for all 7 days |

GPU guidance (bf16 weights; AWQ roughly quarters the memory):

| Setup | Generator (port 8000) | Verifier (port 8001) |
|---|---|---|
| 8 × 80 GB | Qwen2.5-32B-Instruct, TP=2 | Qwen2.5-72B-Instruct, TP=4 |
| 4 × 80 GB | Qwen2.5-32B-Instruct, TP=2 | Qwen2.5-72B-Instruct-AWQ, TP=2 |
| 2 × 48 GB | Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct-AWQ |

The verifier should be different from the generator (ideally larger), so the pipeline
does not grade its own output. The user simulator can share the generator's server.

## 1. Install

```bash
unzip cemr_pipeline.zip && cd cemr_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # 12 plumbing tests, mock LLM, no GPU needed
```

## 2. Put the data in place

```
data/
  captions/  A1_JAKE_DAY1_11000000.json   A2_ALICE_DAY1_11000000.json   ...
  events/    A1_JAKE_DAY1_11000000.json   A2_ALICE_DAY1_11000000.json   ...
```

- **File names** must contain `A<n>_<NAME>_DAY<d>_<HHMMSSFF>`. This is how the
  wearer, the day and the hour are found. Prefixes are fine (`top5_events_A1_JAKE_...`).
- **Captions file:** a list of `{"start","end","kind","text_en","speaker"}`, where
  `kind` is `caption` or `utterance`.
- **Events file:** `{"events": [ ... ]}`. Its `evidence_ids` use the form
  `"<file_id>#/<row index in the captions list>"`.
- **Use the full event lists**, not only the top-5 events. With sparse events,
  "I can't find a record of it" (T2) cannot be trusted.

If your files live elsewhere, just point the globs in the config at them. Nothing
needs to be copied.

## 3. Write your config

For Gemini authentication, create `.env` in the repository root (copy
`.env.example`) and add `GEMINI_API_KEY=your-key`. Set `llm.api_key` to
`"env:GEMINI_API_KEY"` in your JSON config. All model roles, including evaluation,
read this file automatically. An existing process environment variable takes
precedence. `.env` files are ignored by Git; keys stay out of saved run configs.
This configures authentication only; Gemini model names and endpoint URLs must
also be configured.

```bash
cp configs/default.json configs/my.json
```

Edit at least these fields:

```json
"llm":  { "generator_model": "Qwen/Qwen2.5-32B-Instruct",
          "generator_base_url": "http://localhost:8000/v1",
          "verifier_model":  "Qwen/Qwen2.5-72B-Instruct",
          "verifier_base_url":  "http://localhost:8001/v1",
          "user_sim_model":  "Qwen/Qwen2.5-32B-Instruct",
          "user_sim_base_url":  "http://localhost:8000/v1" },
"data": { "captions_glob": "/path/to/captions/*.json",
          "events_glob":   "/path/to/events/*.json",
          "participants": [],                 // [] = all wearers; e.g. ["jake"] for a first try
          "people": ["jake","alice","tasha","lucia","katrina","shure"],
          "people_aliases": {},               // fill in after step 4
          "train_days": [1,2,3,4,5], "dev_days": [6], "test_days": [7],
          "heldout_participants": [] },
"output_dir": "outputs/run1"
```

Each model name must match exactly what vLLM serves: `--served-model-name`, or the
Hugging Face id if you don't set one.

## 4. Stage 0, Part 1: build the index (no GPU)

```bash
python -m cemr.cli index --config configs/my.json
```

This writes `outputs/index/{rows,episodes}.jsonl`, `index_meta.json` and
`index_report.json`, and prints a summary. Check these before going further:

| Check | Where | What to do |
|---|---|---|
| Unknown names (`unknown_people:`, `unknown_speaker_names`) | report flags | add them to `people_aliases`, e.g. `{"shu re": "shure"}` |
| Location/object variants | `vocabulary` | add a synonyms JSON and set `synonyms_path` |
| Dangling evidence ids | summary | usually a file-name mismatch between captions and events |
| Speaker coverage per participant | `per_participant` | below 50% means T4 isn't feasible for that person |
| Episode coverage per participant | `per_participant` | below 70% makes T2 and T5 unreliable |
| Multi-view clusters | `cross_view` | if close to 0, lower `mining.cluster_min_overlap` |

Rerun `index` after each fix. It is fast and uses no LLM.

## 5. Start the model servers

```bash
# terminal 1: generator + user simulator
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000 --tensor-parallel-size 2 \
     --max-model-len 32768 --gpu-memory-utilization 0.90
# terminal 2: verifier (other GPUs)
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve Qwen/Qwen2.5-72B-Instruct --port 8001 \
     --tensor-parallel-size 4 --max-model-len 32768
```

Then check both servers:

```bash
python -m cemr.cli check-llm --config configs/my.json
#   OK   generator ...   OK   verifier ...   OK   user_sim ...
```

With only one server, give all three roles the same URL and model. It works, but
verification is weaker; say so in the paper.

Qwen3 models also work: `<think>` blocks are removed before JSON parsing. Turning
thinking off is faster.

## 6. Stage 0, Part 2: mine candidates (no GPU)

```bash
python -m cemr.cli mine --config configs/my.json
# -> outputs/run1/candidates/T1..T5.jsonl, prints counts per task
```

Open a few lines of each file. Check that the target, the lure and the true/lure
values make sense, and that a "someone else did it" sample in T2 has
`actor_verified: true`.

## 7. Pilot (do not skip)

```bash
python -m cemr.cli generate --config configs/my.json --limit 20 --splits train
python -m cemr.cli stats    --config configs/my.json
```

- **`funnel.json`** shows how many samples survive each stage per task.
- **`stats`** gives the top rejection reasons.
- **Read about 10 samples per task** in `gt.jsonl`, `fm.jsonl` and `conv.jsonl`.
- **Fix, then rerun on a fresh `output_dir`.** Fix prompts (`cemr/prompts/`) or
  thresholds (`verify.threshold`, the mining settings), and use a new
  `output_dir` so earlier rejects don't hide the change.

Stop iterating when every task keeps a reasonable share of samples (roughly 30–70%
survive Stage 4) and the samples you read look right.

## 8. Full generation

```bash
nohup python -m cemr.cli generate --config configs/my.json > gen.log 2>&1 &
```

The run can be resumed: rerunning the same command skips finished and rejected
samples. Every LLM call is logged in `outputs/run1/llm_logs/`.

Outputs:
- `sft/{train,dev,test}.jsonl`: chat-format training data, with the hidden fields
  (intent, retrieval cues, anchor, evidence, corrected attribute)
- `conv.jsonl`: full conversations, including the user's belief state after each turn
- `rejects.jsonl`: every dropped sample, the stage it failed at, and why

## 9. Human review

```bash
python -m cemr.cli export-review --config configs/my.json --split test
```

This writes a CSV with one row per conversation and columns for
grounded / natural / no premature reveal / keep. Review all of test and a sample of
train. Report the kept fraction and remove rejected ids from `sft/test.jsonl`.

## 10. Evaluate a model

```bash
# serve the model under test (any port), then:
python -m cemr.cli evaluate --config configs/my.json --model Qwen/Qwen2.5-7B-Instruct \
       --model-base-url http://localhost:8002/v1 --split test --mode multi_turn
python -m cemr.cli evaluate ... --mode single_turn      # collapsed false-premise baseline
```

The results go to `outputs/run1/eval/<model>/<split>_<mode>/metrics.json`, broken
down by task, subtype, belief condition and participant, with 95% CIs.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `check-llm` fails with connection error | server not up, or wrong port in `*_base_url` |
| `check-llm` fails with 404 / model not found | model name differs from vLLM's served name |
| Many `invalid JSON` retries | model too small; lower the temperature; use an instruct model |
| Most rejects: "value not lexically supported" | ground-truth facts paraphrase the rows; tighten the prompt or add synonyms |
| Most rejects: "primary fact is only visible in another person's recording" | expected for some T2/T4 samples; if very high, check speaker labels and participant names |
| T4 has 0 candidates | no speaker-labelled utterances, or the report cue words don't match your data (`mining.t4_report_cues`) |
| Few T3 candidates | too few episodes per day; lower `mining.t3_min_gap_minutes` |
| "did not converge" rejects | assistant turns aren't citing the planned evidence; check `plan.jsonl` evidence ids |
