"""Text-request guards and persistent, shared local generation quotas."""
from pathlib import Path
import logging
import sqlite3
import time

log = logging.getLogger(__name__)


class LimitError(RuntimeError):
    pass


# Published model capacities; project quotas must come from AI Studio.
MODEL_LIMITS = {name: (1048576, 65536) for name in (
    "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite",
    "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro")}


def count_payload(model, messages):
    request = {"model": f"models/{model}", "contents": []}
    system = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if not isinstance(content, str) or role not in ("system", "user", "assistant"):
            raise LimitError("Token guard supports only system/user/assistant text messages.")
        if role == "system":
            system.append({"text": content})
        else:
            request["contents"].append({"role": "model" if role == "assistant" else "user",
                                        "parts": [{"text": content}]})
    if system:
        request["systemInstruction"] = {"parts": system}
    return {"generateContentRequest": request}


class QuotaGuard:
    """All processes using this DB share counters. No credentials are stored.

    RPD uses a conservative rolling 24h window rather than Pacific midnight.
    External project traffic and token-count/metadata endpoints are not tracked.
    Failed attempts count too. Missing quotas are explicitly unenforced.
    """
    def __init__(self, path, model, limits):
        self.path, self.model = Path(path), model
        self.limits = {k: limits.get(k) for k in ("rpm", "tpm", "rpd")}
        for key, value in self.limits.items():
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{model}: {key} must be a positive integer or null")

    def reserve(self, tokens):
        if self.limits["tpm"] is not None and tokens > self.limits["tpm"]:
            raise LimitError(f"{self.model}: one request exceeds configured TPM")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            with sqlite3.connect(self.path, timeout=30) as db:
                db.execute("CREATE TABLE IF NOT EXISTS calls (model TEXT, ts REAL, tokens INTEGER)")
                db.execute("CREATE INDEX IF NOT EXISTS calls_model_time ON calls(model, ts)")
                db.execute("BEGIN IMMEDIATE")
                now = time.time()
                db.execute("DELETE FROM calls WHERE ts <= ?", (now - 86400,))
                daily = db.execute("SELECT COUNT(*) FROM calls WHERE model=?", (self.model,)).fetchone()[0]
                if self.limits["rpd"] is not None and daily >= self.limits["rpd"]:
                    raise LimitError(f"{self.model}: local rolling-24h request budget exhausted")
                minute = db.execute("SELECT ts,tokens FROM calls WHERE model=? AND ts>? ORDER BY ts",
                                    (self.model, now - 60)).fetchall()
                full = ((self.limits["rpm"] is not None and len(minute) >= self.limits["rpm"])
                        or (self.limits["tpm"] is not None
                            and sum(t for _, t in minute) + tokens > self.limits["tpm"]))
                if not full:
                    db.execute("INSERT INTO calls VALUES (?,?,?)", (self.model, now, tokens))
                    return
                delay = max(0.01, minute[0][0] + 60 - now + 0.05)
            log.info("%s: local RPM/TPM limit reached; waiting %.1fs", self.model, min(delay, 60))
            time.sleep(min(delay, 60))
