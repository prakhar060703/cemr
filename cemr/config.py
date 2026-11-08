"""
config.py

Every tunable in one place. Load from YAML/JSON with `Config.from_file`,
and dump the resolved config next to every run for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    # Use "env:GEMINI_API_KEY" to read a real key from the environment. Local vLLM
    # servers started without --api-key accept any key, so one key can serve all roles.
    api_key: str = "not-needed"
    # Optional per-role key for providers that use a different credential.
    generator_api_key: str = ""
    # Generator and verifier should be DIFFERENT models (e.g. Qwen 32B vs 72B)
    # so the pipeline is not grading its own output.
    generator_model: str = "Qwen/Qwen2.5-32B-Instruct"
    verifier_model: str = "Qwen/Qwen2.5-72B-Instruct"
    user_sim_model: str = "Qwen/Qwen2.5-32B-Instruct"
    # vLLM serves one model per server: give each role its own URL if they differ.
    # Empty -> fall back to base_url.
    generator_base_url: str = ""
    verifier_base_url: str = ""
    user_sim_base_url: str = ""
    temperature: float = 0.7
    verifier_temperature: float = 0.0
    # The simulated user runs a little hotter so replies do not repeat verbatim.
    user_sim_temperature: float = 0.9
    max_retries: int = 3
    # Base seed. Each call derives its own seed from this + prompt + attempt
    # (see llm_client.seed_scope), so runs are reproducible but retries differ.
    seed: Optional[int] = 13
    # Qwen3-style "thinking" switch sent via vLLM chat_template_kwargs.
    # false = no hidden reasoning before the JSON (much faster verifier calls);
    # null = do not send the switch (for models/servers that reject it).
    enable_thinking: Optional[bool] = False
    # Cap on generated tokens per call (0 = server default). Prompt + this must
    # fit the server's max_model_len (32768 for the 30B server).
    max_tokens: int = 4096
    # Samples processed in parallel by `generate`. vLLM batches concurrent
    # requests, so 4-8 is much faster than 1. Set 1 for strictly sequential runs.
    concurrency: int = 4
    # Seconds to wait for one model answer before treating the server as stuck
    # (temporary failures get at most three retries before stopping the run).
    request_timeout: float = 300.0
    # Minimum interval between requests to the same model (in seconds).
    # Useful for rate-limited APIs like Gemini.
    min_request_interval: float = 0.0
    # Hosted models with a thinking budget (e.g. Gemini via its OpenAI-compatible
    # endpoint): "none" | "low" | "medium" | "high". Empty = do not send.
    reasoning_effort: str = ""
    # When the server says it is busy (HTTP 429/5xx, e.g. Gemini "high demand"),
    # wait busy_backoff seconds, doubling each time, for up to busy_retries tries
    # (3 retries at 10 s = 10/20/40 s waits). Quota exhaustion never retries.
    busy_retries: int = 3
    busy_backoff: float = 10.0
    model_quotas: dict = field(default_factory=dict)
    quota_state_path: str = "outputs/gemini_quota.sqlite3"


@dataclass
class DataConfig:
    # One JSON file per hour, e.g. captions/A1_JAKE_DAY1_11000000.json
    captions_glob: str = "data/captions/*.json"
    # One JSON file per hour with {"events": [...]}
    events_glob: str = "data/events/*.json"
    # Each hour file is one participant's first-person recording; the narrator of
    # a file is read from its name (A1_JAKE_... -> "jake"). Leave empty to use all.
    participants: tuple = ()
    # Canonical names of everyone in the household + spelling variants.
    people: tuple = ("jake", "alice", "tasha", "lucia", "katrina", "shure")
    people_aliases: dict = field(default_factory=dict)   # {"shu re": "shure", ...}
    # Optional JSON {"variant": "canonical"} merged into the built-in synonym map.
    synonyms_path: str = ""
    # Where Part 1 writes rows.jsonl / episodes.jsonl / index_report.json
    index_dir: str = "outputs/index"
    # Split by DAY so lures / episodes never leak across splits.
    train_days: tuple = (1, 2, 3, 4, 5)
    dev_days: tuple = (6,)
    test_days: tuple = (7,)
    # Optional person-generalisation test: these participants' TEST-day samples go
    # to split "test_heldout_person"; their train/dev-day samples are dropped.
    heldout_participants: tuple = ()


@dataclass
class MiningConfig:
    # T1 binding: pairs of episodes that share gist but differ in 1-2 slots
    t1_min_shared_people: int = 1
    t1_max_gap_hours: float = 48.0
    # T2 intention vs action: how far ahead to search for an execution
    t2_search_window_hours: float = 36.0
    t2_exec_overlap_threshold: float = 0.34
    # T3 temporal: min separation between target and landmark
    t3_min_gap_minutes: float = 20.0
    # T4 experienced vs told: cue words that mark reported past events
    t4_report_cues: tuple = (
        "you left", "you forgot", "you said", "you put", "you took",
        "earlier", "yesterday", "this morning", "last night", "i saw",
        "you already", "you didn't",       # "did you" removed: questions are not reports
    )
    # T5 counterfactual: max gap between action and its outcome
    t5_max_gap_hours: float = 12.0
    # Control conditions (fraction of samples per task)
    # Near-duplicate control: one real event seen by several wearers yields at most
    # this many candidates per (task, event cluster).
    max_per_cluster: int = 2
    # Cross-view linking: two episodes of different wearers are the same event if
    # their time overlap / shorter span >= this AND they list each other as present.
    cluster_min_overlap: float = 0.3
    # When the user asks (query time). Each candidate gets a delay bucket after the
    # earliest valid time (target/lure/outcome ended): the same day (+1-3 h), the next
    # day, or 2..days_later_max days later. Longer delays = bigger memory bank and more
    # distractors; report results per bucket. Never past the participant's last
    # recording or into another split. T2 "no record" samples keep the earliest time
    # (a later query could include a real execution the miner never searched for).
    query_delay_weights: dict = field(default_factory=lambda: {"same_day": 0.4, "next_day": 0.3,
                                                               "days_later": 0.3})
    days_later_max: int = 5
    correct_belief_rate: float = 0.12
    incomplete_belief_rate: float = 0.08
    max_candidates_per_task: int = 2000


@dataclass
class DialogueConfig:
    max_turns: int = 14                 # assistant+user turns combined
    max_corrections_per_turn: int = 2
    primary_resistance: int = 1         # user pushes back this many times on primary error
    secondary_resistance: int = 0
    retrieval_top_k: int = 5


@dataclass
class VerifyConfig:
    threshold: float = 0.6              # reject if ANY rubric score < threshold
    max_regenerations: int = 2          # closed-loop verifier -> regenerate with feedback
    # After the verifier's flagged facts are dropped, verify the repaired item once
    # more (costs one extra verifier call, only for repaired samples).
    reverify_after_repair: bool = True
    min_gist_overlap: float = 0.5


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    output_dir: str = "outputs"

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        cfg = cls()
        for section, values in raw.items():
            target = getattr(cfg, section, None)
            if target is None or not isinstance(values, dict):
                setattr(cfg, section, values)
                continue
            for k, v in values.items():
                if isinstance(getattr(target, k, None), tuple):
                    v = tuple(v)
                setattr(target, k, v)
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
