"""
schemas.py

Pydantic models for every artifact the pipeline writes.
Each stage reads the previous stage's model and emits its own; nothing is
passed around as untyped dicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

###############################################################################
# Stage 0: index
###############################################################################


class Row(BaseModel):
    """One caption or utterance line from an hour file."""
    row_id: str                     # "A1_JAKE_DAY1_11000000#/8"
    file_id: str
    participant: str                # whose glasses recorded it ("jake"); "I" in captions = this person
    idx: int
    kind: Literal["caption", "utterance"]
    text: str
    speaker: Optional[str] = None
    abs_start: float
    abs_end: float


class Episode(BaseModel):
    episode_id: str
    file_id: str
    participant: str                # the wearer / narrator of this episode
    day: int
    abs_start: float
    abs_end: float
    summary: str
    people: list[str] = []
    activity: list[str] = []
    discussion: list[str] = []
    plans: list[str] = []
    decisions: list[str] = []
    objects: list[str] = []
    location: str = ""
    location_raw: str = ""                     # before canonicalisation (for auditing)
    state_changes: list[str] = []
    evidence_ids: list[str] = []
    flags: list[str] = []                      # e.g. "narrator_added", "dangling_evidence"
    cluster_id: Optional[str] = None           # same real-world event seen from several views

    def slot_values(self, slot: str) -> list[str]:
        v = getattr(self, slot)
        return [v] if isinstance(v, str) else list(v)

    def search_text(self) -> str:
        return " ".join(
            [self.summary, self.location]
            + self.people + self.activity + self.discussion
            + self.objects + self.plans + self.decisions + self.state_changes
        )


###############################################################################
# Task / error taxonomy
###############################################################################


class TaskId(str, Enum):
    T1 = "T1_source_binding"          # which episode each detail came from
    T2 = "T2_intention_action"        # planned vs. actually performed
    T3 = "T3_temporal_displacement"   # when it happened (relative to a landmark)
    T4 = "T4_experienced_vs_told"     # witnessed vs. reported by someone else
    T5 = "T5_counterfactual_blame"    # what would / would not follow without X


class BeliefCondition(str, Enum):
    MISBOUND = "misbound"             # contains a primary error
    INCOMPLETE = "incomplete"         # missing slots, nothing wrong (control)
    CORRECT = "correct"               # memory is right (over-correction control)
    UNSUPPORTED = "unsupported"       # claim with no evidence either way (T2/T5)


PRIMARY_ERROR = {
    TaskId.T1: "event_blending",
    TaskId.T2: "reality_monitoring",
    TaskId.T3: "temporal_confusion",
    TaskId.T4: "source_confusion",
    TaskId.T5: "causal_misattribution",
}

# Secondary errors are restricted to within-episode substitutions so they never
# look like another task's primary error (keeps per-task scores unconfounded).
SECONDARY_ERRORS = ("person_confusion", "object_substitution",
                    "activity_confusion", "location_confusion")


class Candidate(BaseModel):
    """Output of a Stage-0 miner: a target + a lure mined from real data."""
    candidate_id: str
    task: TaskId
    participant: str = ""              # the user whose memory this is (set by BaseMiner)
    split: Optional[Literal["train", "dev", "test", "test_heldout_person"]]
    target_episode: str
    lure_episodes: list[str] = []
    lure_rows: list[str] = []          # e.g. the reported utterance for T4
    primary_slot: str                  # slot the primary error lives in
    true_value: str
    lure_value: str                    # comes from a lure source, never invented
    subtype: str = ""
    query_time: float                  # absolute seconds; memory bank is cut here
    meta: dict = {}


###############################################################################
# Stage 1: ground truth
###############################################################################


class Fact(BaseModel):
    fact_id: str
    slot: str
    value: str
    statement: str                     # first-person, e.g. "I passed my phone around"
    evidence_ids: list[str] = Field(min_length=1)
    # Set deterministically after Stage 1: True if any evidence row comes from the
    # user's OWN recording. Facts visible only in other people's views are used to
    # verify gold labels but never as evidence the assistant can show the user.
    visible_to_user: bool = True
    # Set deterministically after Stage 1 (never trusted from the LLM):
    #   "caption"          -- value's wording is in the cited rows (or the person is a
    #                         cited row's speaker / the narrator)
    #   "event_annotation" -- value only supported by the episode's event annotation
    #                         (people/location); usable as context, never as a correction
    source: Literal["caption", "event_annotation"] = "caption"


class GroundTruth(BaseModel):
    sample_id: str
    task: TaskId
    narrator: str
    target_episode: str
    lure_episodes: list[str] = []
    facts: list[Fact]
    # T2: whether execution was found; T5: which consequences are observed vs inferred
    task_specific: dict = {}


###############################################################################
# Stage 2: false memory
###############################################################################


class Edit(BaseModel):
    slot: str
    true: str
    false: str
    error_type: str
    role: Literal["primary", "secondary"]
    lure_source: str                   # episode_id or row_id the false value came from
    true_evidence: list[str] = Field(min_length=1)


class FalseMemory(BaseModel):
    sample_id: str
    belief_condition: BeliefCondition
    user_memory: str                   # what the user believes, first person
    user_question: str                 # the underlying question they want answered
    edits: list[Edit] = []
    omitted_slots: list[str] = []      # INCOMPLETE condition
    preserved_gist: list[str] = []


###############################################################################
# Stage 3: dialogue plan
###############################################################################

Intent = Literal[
    "retrieve_and_clarify", "confirm", "present_evidence",
    "correct", "suggest_cue", "answer", "summarize",
]
EvidenceType = Literal[
    "utterance_quote", "caption_detail", "timestamp", "landmark_event",
    "absence_of_record", "reported_speech", "none",
]


class PlanStep(BaseModel):
    step: int
    intent: Intent
    retrieval_cues: list[str] = []
    selected_anchor: Optional[str] = None      # row_id of a distinctive detail
    evidence_type: EvidenceType = "none"
    evidence_ids: list[str] = []
    corrected_attributes: list[str] = []       # <= max_corrections_per_turn


class DialoguePlan(BaseModel):
    sample_id: str
    correction_order: list[str]
    steps: list[PlanStep]


###############################################################################
# Stage 4: conversation
###############################################################################


class AssistantTurn(BaseModel):
    role: Literal["assistant"] = "assistant"
    text: str
    intent: Intent
    retrieval_cues: list[str] = []
    retrieved_ids: list[str] = []              # what the retriever returned (episode ids)
    selected_anchor: Optional[str] = None
    evidence_type: EvidenceType = "none"
    evidence_ids: list[str] = []
    corrected_attribute: list[str] = []


class UserTurn(BaseModel):
    role: Literal["user"] = "user"
    text: str
    stance: Literal["confident", "resisting", "partial", "accepting", "converged"]
    belief_update: list[str] = []              # slots accepted this turn
    belief_state: dict[str, str] = {}          # slot -> value the user now holds


class Conversation(BaseModel):
    sample_id: str
    task: TaskId
    participant: str = ""
    split: str
    belief_condition: BeliefCondition
    turns: list[dict]                          # AssistantTurn | UserTurn dumps, in order
    final_belief: dict[str, str]
    gold_belief: dict[str, str]
    converged: bool
    n_turns: int
    # When the user asks: absolute seconds, a readable label ('Day 4, 14:30 (afternoon)'),
    # and how long after the episode (same_day / next_day / days_later).
    query_time: Optional[float] = None
    query_time_label: str = ""
    delay_bucket: str = ""
    flags: dict = {}                           # leakage / validity diagnostics


###############################################################################
# Verification
###############################################################################


class VerifierScores(BaseModel):
    scores: dict[str, float]
    rationale: dict[str, str] = {}
    suggestions: list[str] = []
    # Per-item judgments (ground truth: facts whose value/statement is unsupported),
    # [{"fact_id": str, "reason": str}]. Used to drop single facts, not whole samples.
    flagged: list[dict] = []

    def failing(self, threshold: float, ignore=()) -> dict[str, float]:
        """Gating criteria below threshold (criteria in `ignore` are advisory)."""
        return {k: v for k, v in self.scores.items() if k not in ignore and v < threshold}

    def passed(self, threshold: float, ignore=()) -> bool:
        return bool(self.scores) and not self.failing(threshold, ignore)