"""
llm.py - the only module that talks to OpenRouter.

  * chat(...)              send a prompt to a chat model and return the reply text.
                           Goes through LiteLLM, so changing the model is just a
                           different model string.
  * parse_json_object(...) read the JSON object from a model reply.

chat() retries automatically on temporary problems (rate limits, network
hiccups, server errors) with an increasing wait between attempts.

USAGE METER: every call records its tokens and cost in `METER`, tagged with a
PURPOSE (extract, judge, ...). The evaluation reads the meter at the end and
writes the usage and cost per purpose into its summary.

Where the cost comes from, in order of preference:
  1. the cost OpenRouter reports in the response (exact, in USD), else
  2. an estimate from prices you set in .env (USD per 1 million tokens):
       QARAG_PRICE_GEN_IN, QARAG_PRICE_GEN_OUT
  3. otherwise the cost is unknown ("n/a"); token counts are always recorded.

This is the part of the indexer's qarag.llm that the evaluation needs.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import litellm

# LiteLLM prints a lot of "Provider List" help text on errors; hide it.
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Usage meter: tokens, calls and cost per purpose
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0          # tokens sent to the model (input)
    completion_tokens: int = 0      # tokens written by the model (output)
    cost_usd: float = 0.0           # sum of known costs
    calls_without_cost: int = 0     # calls whose cost was neither reported nor estimable

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd
        self.calls_without_cost += other.calls_without_cost

    def cost_text(self) -> str:
        """Cost for display: exact/estimated amount, marked if some calls had no cost."""
        if self.calls and self.calls_without_cost == self.calls:
            return "n/a"
        text = money(self.cost_usd)
        return text + (" (partial)" if self.calls_without_cost else "")


def money(value: float) -> str:
    """$0.1234, or more decimals for tiny amounts."""
    return f"${value:.4f}" if value >= 0.01 or value == 0 else f"${value:.6f}"


def _price(name: str) -> Optional[float]:
    """Price in USD per 1M tokens from .env, or None if not set."""
    value = os.getenv(name)
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


class UsageMeter:
    """Thread-safe counter: every update happens under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], Usage] = {}   # (scope, purpose) -> Usage
        self._scope = ""

    def reset(self) -> None:
        with self._lock:
            self._data.clear()
            self._scope = ""

    def record(self, purpose: str, prompt_tokens: int, completion_tokens: int,
               cost: Optional[float]) -> None:
        if cost is None:
            # No cost from OpenRouter: estimate from the prices in .env, if any.
            p_in, p_out = _price("QARAG_PRICE_GEN_IN"), _price("QARAG_PRICE_GEN_OUT")
            cost = (None if p_in is None or p_out is None
                    else (prompt_tokens * p_in + completion_tokens * p_out) / 1e6)
        u = Usage(calls=1, prompt_tokens=int(prompt_tokens or 0),
                  completion_tokens=int(completion_tokens or 0),
                  cost_usd=float(cost or 0.0), calls_without_cost=0 if cost is not None else 1)
        with self._lock:
            self._data.setdefault((self._scope, purpose), Usage()).add(u)

    # ------------------------------------------------------------- reading
    def snapshot(self) -> dict[str, dict[str, dict]]:
        """{scope: {purpose: usage-as-dict}}; the evaluation uses scope ""."""
        with self._lock:
            out: dict[str, dict[str, dict]] = {}
            for (scope, purpose), u in self._data.items():
                out.setdefault(scope, {})[purpose] = asdict(u)
            return out

    def total(self, scope: Optional[str] = None) -> Usage:
        """Sum over all purposes, for one scope or for everything."""
        total = Usage()
        with self._lock:
            for (s, _), u in self._data.items():
                if scope is None or s == scope:
                    total.add(u)
        return total


METER = UsageMeter()

# Whether chat calls ask OpenRouter to include the cost in the response.
# Switched off automatically if a provider rejects the option (see chat()).
_ASK_FOR_COST = True


def _usage_from_response(response: Any) -> tuple[int, int, Optional[float]]:
    """(prompt_tokens, completion_tokens, cost or None) from an API response.
    OpenRouter puts the exact cost in usage.cost when usage accounting is on."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, None
    get = (lambda k: usage.get(k)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
    prompt = get("prompt_tokens") or get("total_tokens") or 0
    completion = get("completion_tokens") or 0
    cost = get("cost")
    if cost is None:
        # LiteLLM sometimes computes the cost itself and keeps it here.
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    try:
        cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost = None
    return int(prompt), int(completion), cost


# ---------------------------------------------------------------------------
# JSON parsing that tolerates typical model formatting habits.
# ---------------------------------------------------------------------------
def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model reply.

    Models often wrap JSON in ```json fences, add a sentence before it, or
    (reasoning models such as Qwen3) include a <think>...</think> section.
    We remove those and parse from the first "{" to the last "}"."""
    if text is None:
        raise ValueError("empty reply")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # drop reasoning blocks
    text = re.sub(r"```(?:json)?", "", text)                         # drop code fences
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start:end + 1])


def _is_retryable(error: Exception) -> bool:
    """Temporary errors worth retrying: rate limits (429), server errors (5xx),
    timeouts and connection problems. Errors like a bad API key are not."""
    message = str(error).lower()
    status = getattr(error, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    return any(word in message for word in ("rate limit", "timeout", "timed out",
                                            "connection", "temporarily", "overloaded"))


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def chat(model: str, messages: list[dict], temperature: float, max_tokens: int,
         network_retries: int = 5, purpose: str = "chat") -> str:
    """One chat completion, retried on temporary errors. Returns the reply text.
    `purpose` labels the call in the usage meter (extract, judge ...)."""
    global _ASK_FOR_COST
    delay = 2.0
    for attempt in range(1, network_retries + 1):
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, timeout=180)
        if _ASK_FOR_COST:
            # Ask OpenRouter to report the exact cost of the call in usage.cost.
            kwargs["extra_body"] = {"usage": {"include": True}}
        try:
            response = litellm.completion(**kwargs)
            METER.record(purpose, *_usage_from_response(response))
            return response.choices[0].message.content or ""
        except Exception as error:  # noqa: BLE001 - we decide below whether to retry
            message = str(error).lower()
            if _ASK_FOR_COST and any(w in message for w in ("extra_body", "usage", "unrecognized",
                                                            "unknown parameter", "unexpected keyword")):
                # Safety net: if the provider ever rejects the cost-reporting option,
                # stop sending it (costs then come from the .env prices instead).
                _ASK_FOR_COST = False
                continue
            if attempt == network_retries or not _is_retryable(error):
                raise
            time.sleep(delay)
            delay *= 2           # 2s, 4s, 8s, 16s ... (exponential backoff)
    return ""  # not reached
