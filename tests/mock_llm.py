"""
Deterministic mock of LLMClient for plumbing tests. It routes on the
`### TASK:` header and builds well-formed answers from the index, so the
whole pipeline can run without a vLLM server. It says nothing about the
quality of real model outputs.
"""

from __future__ import annotations

import json
import re

from cemr.llm_client import LLMClient
from cemr.utils import tokens

CONCRETE = ("people", "objects", "location", "activity")


def _block(prompt: str, start: str, end: str):
    m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), prompt, re.S)
    if not m:
        return None
    txt = m.group(1).strip()
    if txt in ("none", ""):
        return None
    return json.loads(txt)


class MockLLM:
    extract_json = staticmethod(LLMClient.extract_json)

    def __init__(self, index, narrator="Jake"):
        self.index = index
        self.narrator = narrator.lower()
        self.calls = []

    # API used by the pipeline -------------------------------------------------
    def generate(self, system, user, temperature=None):
        return json.dumps(self._route(user))

    def generate_json(self, system, user, schema=None, temperature=None):
        obj = self._route(user)
        return schema.model_validate(obj) if schema else obj

    # --------------------------------------------------------------------------
    def _route(self, p: str):
        task = re.search(r"### TASK: (\w+)", p).group(1)
        self.calls.append(task)
        return getattr(self, "_" + task)(p)

    def _ground_truth(self, p):
        sid = re.search(r'"sample_id": "(.+?)"', p).group(1)
        task = re.search(r"### BENCHMARK TASK: (\S+)", p).group(1)
        target = re.search(r"Target episode: (\S+)", p).group(1)
        lures = re.search(r"Related episodes: (.+)", p).group(1)
        lures = [] if lures.strip() == "none" else [x.strip() for x in lures.split(",")]
        pslot, pval = re.search(r'slot="(.+?)" whose value is "(.+?)"', p).groups()
        ep = self.index.episodes[target]
        rows = self.index.episode_rows(target)
        facts, n = [], 0

        def support(value):
            vt = set(tokens(value))
            for r in rows:
                if vt & set(tokens(r.text)):
                    return [r.row_id]
            return None

        for slot in CONCRETE:
            for v in ep.slot_values(slot):
                if not v or v == self.narrator:
                    continue
                ev = support(v)
                if ev:
                    n += 1
                    facts.append({"fact_id": f"f{n}", "slot": slot, "value": v,
                                  "statement": f"I recall {v}.", "evidence_ids": ev})
                    break
        if not any(f["slot"] == pslot and f["value"] == pval for f in facts):
            facts = [f for f in facts if f["slot"] != pslot]
            ev = support(pval) or [rows[0].row_id]
            facts.append({"fact_id": "fp", "slot": pslot, "value": pval,
                          "statement": f"It was {pval}.", "evidence_ids": ev})
        ts = {"report_consistent_with_record": True} if task.startswith("T4") else {}
        return {"sample_id": sid, "task": task, "narrator": "Jake", "target_episode": target,
                "lure_episodes": lures, "facts": facts, "task_specific": ts}

    def _false_memory(self, p):
        sid = re.search(r'"sample_id": "(.+?)"', p).group(1)
        cond = re.search(r"Belief condition: (\w+)", p).group(1)
        primary = _block(p, "PRIMARY EDIT (fixed, copy it verbatim into \"edits\"):", "\n\nSECONDARY")
        pool = _block(p, "do not invent new values; copy slot/true/false/lure_source/true_evidence):",
                      "\n\nSlots to omit") or []
        n = int(re.search(r"choose exactly (\d+)", p).group(1))
        omitted = json.loads(re.search(r'"omitted_slots": (\[.*?\])', p).group(1))
        edits = ([primary] if primary else []) + pool[:n]
        falses = " and ".join(e["false"] for e in edits if e["slot"] in CONCRETE)
        memory = f"That time I remember {falses}." if falses else "That time I remember it well."
        return {"sample_id": sid, "belief_condition": cond, "user_memory": memory,
                "user_question": "What happened next that time?", "edits": edits,
                "omitted_slots": omitted, "preserved_gist": ["that time"]}

    def _plan(self, p):
        sid = re.search(r'"sample_id": "(.+?)"', p).group(1)
        edits = json.loads(re.search(r"EDITS TO CORRECT \(slot: true <- false\):\n(.*?)\n", p).group(1))
        allowed = json.loads(re.search(r"ALLOWED EVIDENCE PER SLOT: (.*?)\n", p).group(1))
        order = sorted(edits, key=lambda e: e["role"] == "primary")
        steps = [{"step": 1, "intent": "retrieve_and_clarify", "retrieval_cues": ["table"],
                  "selected_anchor": None, "evidence_type": "none", "evidence_ids": [],
                  "corrected_attributes": []}]
        for e in order:
            steps.append({"step": len(steps) + 1, "intent": "correct", "retrieval_cues": [e["true"]],
                          "selected_anchor": allowed[e["slot"]][0], "evidence_type": "caption_detail",
                          "evidence_ids": allowed[e["slot"]][:1], "corrected_attributes": [e["slot"]]})
        if not edits:
            steps.append({"step": len(steps) + 1, "intent": "confirm", "evidence_ids": [],
                          "corrected_attributes": []})
        steps.append({"step": len(steps) + 1, "intent": "summarize", "evidence_ids": [],
                      "corrected_attributes": []})
        return {"sample_id": sid, "correction_order": [e["slot"] for e in order], "steps": steps}

    def _user_open(self, p):
        return {"text": "Hey, I was trying to remember something from earlier."}

    def _user_reply(self, p):
        return {"text": "Hmm, okay, that makes sense."}

    def _assistant_step(self, p):
        return {"text": "Let me check your recording from around then.", "intent": "retrieve_and_clarify"}

    def _assistant_query(self, p):
        return {"retrieval_cues": ["table", "phone"]}

    def _assistant_reply(self, p):
        return {"text": "I found a matching moment. Do you remember who was at the table?",
                "intent": "retrieve_and_clarify", "evidence_ids": []}

    def _parse_turn(self, p):
        return {"corrections": [], "confirmations": [], "cites_evidence": False, "asks_question": True}

    def _value_match(self, p):
        return {"same": False}

    def _verify(self, p):
        crit = json.loads(re.search(r"CRITERIA:\n(\{.*?\n\})", p, re.S).group(1))
        return {"rationale": {k: "ok" for k in crit}, "scores": {k: 0.9 for k in crit}, "suggestions": []}

    _verify_ground_truth = _verify
    _verify_false_memory = _verify
    _verify_conversation = _verify
