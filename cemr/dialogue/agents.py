"""
dialogue/agents.py

Three agents:
  * UserSimulator        -- verbalises the controller's reaction; never sees gold.
  * PlannedAssistant     -- generation mode; realises plan steps with retrieval.
  * EvalAssistant        -- evaluation mode; the model under test, with only a
                            retrieval tool (query -> results -> reply).
"""

from __future__ import annotations

import json

from ..llm_client import LLMUnavailable
from ..prompts import conversation as P
from ..retrieval import BM25Retriever
from ..schemas import AssistantTurn, PlanStep
from ..utils import short_time
from ..validation.deterministic import leaked_values, near_duplicate


import re

# Internal row / episode ids, e.g. A3_TASHA_DAY6_12000000#/271 or ..._E02
_ROW_ID_RE = re.compile(r"A\d+_[A-Z]+_DAY\d+_\d{8}")


def render_history(turns: list[dict], last: int | None = None) -> str:
    seq = turns[-last:] if last else turns
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in seq) or "(start)"


class UserSimulator:
    def __init__(self, client, narrator: str, now: str | None = None):
        self.client = client
        self.system = P.USER_SYSTEM.format(narrator=narrator)
        self.now = now                  # conversation's current time label

    def open(self, user_memory: str, user_question: str, omitted) -> str:
        out = self.client.generate_json(self.system, P.user_opening(None, user_memory, user_question, omitted,
                                                                    self.now))
        return out["text"]

    def reply(self, belief_state: dict, assistant_text: str, reaction: str, slots: list[str], turns) -> str:
        instruction = P.REACTIONS[reaction].format(slots=", ".join(slots) or "that")
        prompt = P.user_reply(belief_state, assistant_text, instruction, render_history(turns, last=4), self.now)
        earlier = [t["text"] for t in turns if t["role"] == "user"]
        text = ""
        for _ in range(2):   # one rewrite if it copies an earlier user turn
            text = self.client.generate_json(self.system, prompt)["text"]
            copy = next((e for e in earlier if near_duplicate(text, e)), None)
            if copy is None:
                break
            prompt += f'\n\nYou already said: "{copy}". Say it differently this time.'
        return text


class PlannedAssistant:
    """Generation mode: realises one plan step; retries if it leaks future values."""

    def __init__(self, client, index, max_corr: int, max_leak_retries: int = 2, now: str | None = None):
        self.client = client
        self.index = index
        self.now = now
        self.system = P.ASSISTANT_SYSTEM.format(max_corr=max_corr)
        self.max_leak_retries = max_leak_retries

    def step(self, turns, step: PlanStep, retriever: BM25Retriever, allowed: dict, forbidden: list[str],
             k: int, notes=()) -> tuple[AssistantTurn, dict]:
        query = step.retrieval_cues or [turns[-1]["text"]]
        hits = retriever.search(query, k=k)
        ev_ids = list(dict.fromkeys(step.evidence_ids + ([step.selected_anchor] if step.selected_anchor else [])))
        ev_block = "\n".join(f"<{r.row_id}> [{short_time(r.abs_start)}] ({r.kind}) "
                             f"{(r.speaker + ': ') if r.speaker else ''}{r.text}"
                             for r in self.index.get_rows(ev_ids)) or "(none)"
        prompt = P.assistant_generation(render_history(turns), step.model_dump(), ev_block,
                                        allowed, forbidden, retriever.render(hits), notes, self.now)
        earlier = [t["text"] for t in turns if t["role"] == "assistant"]
        diag = {"leak_retries": 0, "repeat_retries": 0}
        for attempt in range(self.max_leak_retries + 1):
            raw = self.client.generate_json(self.system, prompt)
            text = raw.get("text", "")
            leaks = leaked_values(text, forbidden)
            if _ROW_ID_RE.search(text):
                leaks = leaks + ["a row id"]
            copy = next((e for e in earlier if near_duplicate(text, e)), None)
            if not leaks and copy is None:
                break
            if leaks:
                diag["leak_retries"] += 1
                prompt += (f"\n\nYour draft revealed {leaks} too early (row ids never go in the text: "
                           "say the time instead). Rewrite without them.")
            if copy is not None:
                diag["repeat_retries"] += 1
                prompt += (f'\n\nYour draft repeats your earlier turn "{copy[:160]}". '
                           "Present it differently (quote the row, give its time, or ask something new).")
        else:
            if leaks:
                # Still leaking after rewrites: redact instead of failing the whole
                # conversation ("Jake said..." -> "someone said..."). Flagged per turn.
                text = self._redact(text, forbidden)
                raw = {**raw, "text": text}
                diag["redacted"] = leaks
                if leaked_values(text, forbidden) or _ROW_ID_RE.search(text):
                    diag["leak_unresolved"] = leaks
        # Hidden supervision fields come from the verified PLAN (gold), text from the LLM.
        turn = AssistantTurn(
            text=raw.get("text", ""),
            intent=step.intent,
            retrieval_cues=step.retrieval_cues,
            retrieved_ids=[ep.episode_id for ep, _ in hits],
            selected_anchor=step.selected_anchor,
            evidence_type=step.evidence_type,
            evidence_ids=step.evidence_ids,
            corrected_attribute=step.corrected_attributes,
        )
        return turn, diag


    def _redact(self, text: str, forbidden: list[str]) -> str:
        """Replace forbidden values with neutral words, and row ids with 'that recording'."""
        text = re.sub(r"(?:in the recording )?(?:at )?\b" + _ROW_ID_RE.pattern + r"(?:_E\d+|#/\d+)?", "then", text)
        people = getattr(self.index, "people", set()) or set()
        for v in sorted(forbidden, key=len, reverse=True):
            if not v:
                continue
            low = v.strip().lower()
            repl = "that day" if low.startswith("day ") else \
                "someone" if (low in people or self.index.person(low) in people) else "something"
            text = re.sub(r"\b" + re.escape(v) + r"\b", repl, text, flags=re.I)
        return text


class EvalAssistant:
    """The model under test. Receives only the conversation and a retrieval tool."""

    def __init__(self, client, max_corr: int, now: str | None = None):
        self.client = client
        self.system = P.ASSISTANT_SYSTEM.format(max_corr=max_corr)
        self.now = now

    def step(self, turns, retriever: BM25Retriever, k: int) -> AssistantTurn:
        hist = render_history(turns)
        try:
            cues = self.client.generate_json(self.system, P.assistant_eval_query(hist, self.now)).get("retrieval_cues", [])
        except LLMUnavailable:
            raise
        except Exception:
            cues = []
        hits = retriever.search(cues or [turns[-1]["text"]], k=k)
        raw_text = self.client.generate(self.system, P.assistant_eval_reply(hist, retriever.render(hits), self.now))
        try:
            obj = self.client.extract_json(raw_text)
            text, intent = obj.get("text", raw_text), obj.get("intent", "answer")
            ev = obj.get("evidence_ids", [])
        except Exception:           # extract_json is local parsing, no server call
            text, intent, ev = raw_text, "answer", []
        allowed_intents = set(AssistantTurn.model_fields["intent"].annotation.__args__)
        return AssistantTurn(text=text, intent=intent if intent in allowed_intents else "answer",
                             retrieval_cues=cues, retrieved_ids=[e.episode_id for e, _ in hits],
                             evidence_ids=ev if isinstance(ev, list) else [])


def parse_eval_turn(parser_client, belief_state: dict, text: str) -> dict:
    """LLM turn parser (verifier model) -> corrections / confirmations / evidence flag."""
    from ..prompts import verifier as V
    try:
        return parser_client.generate_json(V.TURN_PARSER_SYSTEM, V.turn_parser(belief_state, text))
    except LLMUnavailable:
        raise
    except Exception:
        return {"corrections": [], "confirmations": [], "cites_evidence": False, "asks_question": False}


def make_value_matcher(parser_client):
    from ..prompts import verifier as V
    from ..utils import canon, jaccard, tokens
    cache: dict = {}

    def match(slot, a, b) -> bool:
        if canon(a) == canon(b) or (tokens(b) and jaccard(tokens(a), tokens(b)) >= 0.6):
            return True
        key = (slot, canon(a), canon(b))
        if key not in cache:
            try:
                cache[key] = bool(parser_client.generate_json(
                    V.VALUE_MATCH_SYSTEM, V.value_match(slot, a, b)).get("same", False))
            except LLMUnavailable:
                raise
            except Exception:
                cache[key] = False
        return cache[key]

    return match


def dump(obj) -> dict:
    return json.loads(obj.model_dump_json()) if hasattr(obj, "model_dump_json") else obj
