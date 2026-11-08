"""
llm_client.py

Extends the original project client (same constructor and `generate` /
`generate_json` API) with:

  * multi-turn `chat()`
  * JSON extraction that tolerates prose around the object
  * optional pydantic validation with the error fed back on retry
  * a call log (JSONL) so every generation is auditable / reproducible
  * per-call seeds: derived from the base seed, the prompt and the active
    `seed_scope` (e.g. sample id + attempt). Reruns stay reproducible, but a
    retry of the same prompt gets a DIFFERENT seed. (A single fixed seed made
    every retry of an identical prompt return an identical answer.)
  * optional `enable_thinking` switch for Qwen3-style models (sent through the
    vLLM chat template); False avoids long hidden reasoning before the JSON
  * thread-safe logging, so samples can run in parallel
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel, ValidationError
from .limits import MODEL_LIMITS, LimitError, QuotaGuard, count_payload

log = logging.getLogger(__name__)

# Extra key mixed into every per-call seed. Set per sample/stage/attempt with
# `seed_scope(...)`. A ContextVar is thread-local, so parallel samples never mix.
_SEED_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar("cemr_seed_scope", default="")
_LOG_LOCK = threading.Lock()


class LLMUnavailable(RuntimeError):
    """
    The model server could not answer (connection refused, tunnel down, 404 wrong
    model, 5xx, rate limit). This is an infrastructure failure, not a bad sample:
    stages re-raise it so the sample is NOT written to rejects.jsonl and is retried
    on the next run.
    """


class LLMQuotaExhausted(LLMUnavailable):
    """Quota or credits exhausted: never retry this request."""


def _quota_exhausted(error) -> bool:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status or getattr(response, "status_code", None)
    detail = (str(error) + " " + str(getattr(error, "body", ""))
              + " " + str(getattr(response, "text", ""))).lower()
    if status == 402:
        return True
    if re.search(r'(?:limit\s*:\s*0\b|["\x27]quota_?value["\x27]\s*:\s*["\x27]?0\b)', detail):
        return True
    if any(word in detail for word in (
        "insufficient_quota", "quota exhausted", "quota exceeded", "exceeded your current quota",
        "resource_exhausted", "daily", "perday", "per_day", "per day",
        "billing", "credit", "spending limit", "budget exhausted",
    )):
        # Explicit minute limits are transient; daily/billing limits take priority.
        permanent = any(word in detail for word in (
            "daily", "perday", "per_day", "per day", "billing", "credit",
            "insufficient_quota", "spending limit", "budget exhausted",
        ))
        minute = any(word in detail for word in ("perminute", "per_minute", "per minute"))
        return permanent or not minute
    return False


# HTTP statuses that mean "the service cannot answer right now", whatever the request.
_UNAVAILABLE_STATUS = {401, 402, 403, 404, 408, 429, 500, 502, 503, 504}
# Of those, the temporary ones worth waiting out (rate limit, overloaded, gateway).
_BUSY_STATUS = {408, 429, 500, 502, 503, 504}


def _unavailable_errors() -> tuple:
    errs: list = [ConnectionError, TimeoutError]
    try:
        import openai
        errs += [openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                 openai.NotFoundError, openai.AuthenticationError, openai.PermissionDeniedError,
                 openai.RateLimitError]
    except Exception:  # openai missing or older version
        pass
    return tuple(errs)


def resolve_api_key(value: Optional[str]) -> str:
    """
    Resolve env: references from the process environment, then the repo's .env.
    Real keys never need to enter config files (which are copied next to outputs).
    """
    import os
    if value and value.startswith("env:"):
        name = value[4:]
        from dotenv import dotenv_values

        env_path = Path(__file__).resolve().parent.parent / ".env"
        key = os.environ.get(name)
        if key is None:
            key = dotenv_values(env_path, encoding="utf-8-sig", interpolate=False).get(name)
        if not key:
            raise SystemExit(f"API key {name} is not set. Set it in the environment or {env_path}.")
        return key
    return value or "not-needed"


@contextlib.contextmanager
def seed_scope(*keys):
    """Nest a seed scope, e.g. `with seed_scope(sample_id, "conv", attempt): ...`."""
    parent = _SEED_SCOPE.get()
    token = _SEED_SCOPE.set(parent + "|" + "|".join(map(str, keys)))
    try:
        yield
    finally:
        _SEED_SCOPE.reset(token)


class LLMClient:
    def __init__(
        self,
        model,
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        temperature=0.7,
        max_retries=3,
        seed: Optional[int] = None,
        log_path: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        busy_retries: int = 3,
        busy_backoff: float = 10.0,
        model_quotas: Optional[dict] = None,
        quota_state_path: str = "outputs/gemini_quota.sqlite3",
        min_request_interval: float = 0.0,
    ):
        from openai import OpenAI  # imported lazily so tests can run without a server

        self.model = model
        self.temperature = temperature
        self.max_retries = max(1, min(3, int(max_retries)))
        self.seed = seed
        self.enable_thinking = enable_thinking     # None -> do not send the switch
        self.max_tokens = max_tokens               # None -> server default
        self.reasoning_effort = reasoning_effort   # e.g. Gemini: "none"/"low"/"medium"/"high"
        # Hosted APIs answer 429/5xx when overloaded ("high demand"): wait and retry,
        # doubling the wait each time (10 s, 20 s, 40 s, ...) before giving up.
        self.busy_retries = max(0, min(3, int(busy_retries)))
        self.busy_backoff = busy_backoff
        self.stop_event = threading.Event()
        from urllib.parse import urlparse
        self.gemini = urlparse(base_url).hostname == "generativelanguage.googleapis.com"
        self.token_limits = None
        self.quota = None
        if self.gemini:
            self.model = model.removeprefix("models/")
            self.token_limits = MODEL_LIMITS.get(self.model)
            if self.token_limits is None:
                raise ValueError(f"No verified token limits for {self.model}; add its capacities to MODEL_LIMITS first")
            if type(max_tokens) is not int or not 1 <= max_tokens <= self.token_limits[1]:
                raise ValueError(f"{self.model}: max_tokens must be between 1 and {self.token_limits[1]}")
            self.quota = QuotaGuard(quota_state_path, self.model, (model_quotas or {}).get(self.model, {}))
            missing = [k for k, v in self.quota.limits.items() if v is None]
            if missing:
                log.warning("%s: project quotas not configured for %s; those limits cannot be enforced",
                            self.model, ', '.join(missing))
        api_key = resolve_api_key(api_key)
        self.log_path = Path(log_path) if log_path else None
        # timeout: seconds to wait for one answer before giving up (the library
        # default is 600 s with 2 retries, so a hung server stalled a call for 30 min).
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0,
                             **({"timeout": timeout} if timeout else {}))
        # Rate limiting: minimum interval between requests
        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0
        self._request_lock = threading.Lock()

    ###########################################################################
    # Raw generation
    ###########################################################################

    def _check_stopped(self):
        if self.stop_event.is_set():
            raise LLMUnavailable("Run stopped after an API failure; no further calls will be sent")

    def _request(self, operation, call):
        """One retry budget per API operation; SDK retries are disabled."""
        import httpx
        for attempt in range(self.busy_retries + 1):
            self._check_stopped()
            label = f"{self.log_path.stem if self.log_path else 'llm'}/{self.model}"
            log.info("API %s %s attempt %d/%d scope=%s", label, operation,
                     attempt + 1, self.busy_retries + 1, _SEED_SCOPE.get())
            started = time.monotonic()
            try:
                result = call()
                log.info("API %s %s OK (%.2fs)", label, operation, time.monotonic() - started)
                return result
            except Exception as e:
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                if _quota_exhausted(e) or isinstance(e, LimitError):
                    log.error("API %s %s: QUOTA/LIMIT EXHAUSTED; stopping without retry: %s", label, operation, e)
                    self.stop_event.set()
                    raise LLMQuotaExhausted(f"{label}: quota/limit exhausted: {e}") from e
                transient = (status in _BUSY_STATUS or
                             (status is None and isinstance(e, _unavailable_errors() + (httpx.TransportError,))))
                if transient and attempt < self.busy_retries:
                    delay = self.busy_backoff * 2 ** attempt
                    log.warning("API %s %s failed (HTTP %s): %s; retry %d/%d in %.1fs",
                                label, operation, status, e, attempt + 1, self.busy_retries, delay)
                    time.sleep(delay)
                    continue
                log.error("API %s %s FAILED; no more retries: %s", label, operation, e)
                if transient or status in _UNAVAILABLE_STATUS or isinstance(e, LLMUnavailable):
                    self.stop_event.set()
                    raise LLMUnavailable(f"{label} {operation}: {e}") from e
                raise

    def chat(self, messages: list[dict], temperature: Optional[float] = None) -> str:
        self._check_stopped()
        # Rate limiting: wait if necessary to maintain minimum interval between requests
        if self.min_request_interval > 0:
            with self._request_lock:
                elapsed = time.time() - self._last_request_time
                if elapsed < self.min_request_interval:
                    sleep_time = self.min_request_interval - elapsed
                    log.info("%s: request pacing; waiting %.1fs", self.model, sleep_time)
                    time.sleep(sleep_time)
        t0 = time.time()
        input_tokens = None
        if self.gemini:
            input_tokens = self._count_input(messages)
            # Reserve room for compatibility-layer framing and output. This is
            # deliberately stricter than the separately published input limit.
            if input_tokens + 1024 + self.max_tokens > self.token_limits[0]:
                raise LLMUnavailable(f"{self.model}: input ({input_tokens}) plus output budget "
                                     "and framing margin exceeds local token limit")
        kwargs = dict(model=self.model, messages=messages,
                      temperature=self.temperature if temperature is None else temperature)
        seed = self.call_seed(messages)
        if seed is not None:
            kwargs["seed"] = seed
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        def send():
            if self.quota:
                self.quota.reserve(input_tokens + 1024)
            self._check_stopped()
            return self.client.chat.completions.create(**kwargs)
        log.debug("%s request messages: %s", self.model, messages)
        response = self._request("chat.completions", send)
        if not response.choices:
            raise LLMUnavailable(f"{self.model}: empty completion response")
        text = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else None
        log.info("%s response scope=%s usage=%s:\n%s", self.model, _SEED_SCOPE.get(), usage, text)
        self._log(messages, text, time.time() - t0, seed, usage, input_tokens)
        if response.choices[0].finish_reason == "length":
            raise LLMUnavailable(f"{self.model}: output hit max_tokens={self.max_tokens}; "
                                 "increase within model limits or shorten the prompt")
        # Update last request time for rate limiting
        if self.min_request_interval > 0:
            with self._request_lock:
                self._last_request_time = time.time()
        return text

    def _count_input(self, messages):
        """Native tokenizer preflight; fail closed if counting is unavailable."""
        import httpx
        payload = count_payload(self.model, messages)
        def count():
            response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:countTokens",
                headers={"x-goog-api-key": self.client.api_key}, json=payload, timeout=60)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Surface the provider's explanation (e.g. model access restrictions),
                # while retaining the status/response for retry classification.
                try:
                    detail = response.json().get("error", {}).get("message", "")
                except (ValueError, AttributeError):
                    detail = ""
                if detail:
                    detail = str(detail).replace(self.client.api_key, "[REDACTED]")[:1500]
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {detail}",
                        request=exc.request, response=response) from None
                raise
            count = response.json()["totalTokens"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid count")
            return count
        try:
            return self._request("countTokens", count)
        except LLMUnavailable:
            raise
        except Exception as e:
            # Do not include request headers or response bodies in errors.
            raise LLMUnavailable(f"{self.model}: token count failed ({type(e).__name__}); "
                                 "generation was not sent") from None

    def call_seed(self, messages: list[dict]) -> Optional[int]:
        """Deterministic per-call seed: base seed + seed scope + prompt content."""
        if self.seed is None:
            return None
        h = hashlib.sha256(f"{self.seed}|{_SEED_SCOPE.get()}|".encode())
        h.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode())
        return int(h.hexdigest()[:8], 16)

    def generate(self, system_prompt, user_prompt, temperature: Optional[float] = None) -> str:
        return self.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=temperature,
        )

    ###########################################################################
    # JSON generation
    ###########################################################################

    def generate_json(
        self,
        system_prompt,
        user_prompt,
        schema: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
    ):
        """
        Returns a dict (or a validated `schema` instance).
        On a parse/validation failure the error is appended to the prompt and
        the call is retried, up to `max_retries`.
        """
        prompt = user_prompt
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            raw = self.generate(system_prompt, prompt, temperature=temperature)
            try:
                obj = self.extract_json(raw)
                return schema.model_validate(obj) if schema else obj
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_err = e
                log.warning("invalid JSON attempt %d/%d: %s; %s", attempt + 1, self.max_retries,
                            str(e)[:200], "retrying" if attempt + 1 < self.max_retries else "no further JSON attempts")
                prompt = (user_prompt + "\n\nYour previous output was invalid:\n"
                          f"{str(e)[:800]}\nReturn ONLY one valid JSON object.")
        raise last_err  # type: ignore[misc]

    ###########################################################################
    # Helpers
    ###########################################################################

    @staticmethod
    def clean_json(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        return text

    @classmethod
    def extract_json(cls, text: str):
        text = cls.clean_json(text)
        # Qwen reasoning models may prepend <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
            if start < 0:
                raise ValueError("no JSON object found in model output")
            depth, in_str, esc = 0, False, False
            for i, ch in enumerate(text[start:], start):
                if in_str:
                    esc = (ch == "\\") and not esc
                    if ch == '"' and not esc:
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start:i + 1])
            raise

    def _log(self, messages, output, seconds, seed=None, usage=None, input_tokens=None):
        if not self.log_path:
            return
        line = json.dumps({"model": self.model, "t": round(seconds, 2), "seed": seed,
                           "scope": _SEED_SCOPE.get(), "messages": messages, "output": output,
                           "usage": usage, "counted_input_tokens": input_tokens},
                          ensure_ascii=False) + "\n"
        with _LOG_LOCK:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)


def build_clients(cfg, log_dir: Optional[str] = None) -> dict[str, "LLMClient"]:
    """generator / verifier / user_sim clients from config."""
    l = cfg.llm
    mk = lambda model, url, temp, name, api_key=None: LLMClient(  # noqa: E731
        model, url or l.base_url, api_key or l.api_key, temp, l.max_retries, l.seed,
        f"{log_dir}/{name}.jsonl" if log_dir else None,
        enable_thinking=l.enable_thinking, max_tokens=l.max_tokens or None,
        timeout=getattr(l, "request_timeout", None) or None,
        reasoning_effort=getattr(l, "reasoning_effort", None) or None,
        busy_retries=getattr(l, "busy_retries", 3), busy_backoff=getattr(l, "busy_backoff", 10.0),
        model_quotas=l.model_quotas, quota_state_path=l.quota_state_path,
        min_request_interval=l.min_request_interval)
    return {
        "generator": mk(l.generator_model, l.generator_base_url, l.temperature, "generator",
                         getattr(l, "generator_api_key", "")),
        "verifier": mk(l.verifier_model, l.verifier_base_url, l.verifier_temperature, "verifier"),
        "user_sim": mk(l.user_sim_model, l.user_sim_base_url, l.user_sim_temperature, "user_sim"),
    }
