"""
llm.py - one wrapper for every chat-model call (ReAct agent, memory, LLM-S).

Why one wrapper:
  * every call is measured the same way (tokens, cost, time) and handed to
    the run log, so the cost of each role can be reported per question;
  * Qwen3 "<think>...</think>" blocks are removed in one place;
  * dry-run mode swaps the real call for a fake one in one place.

Calls go through LiteLLM with model strings like "openrouter/qwen/qwen3-30b-a3b".
OpenRouter returns the real cost of a call when the request contains
extra_body={"usage": {"include": True}}; we read it if present, otherwise we
fall back to LiteLLM's own price table, otherwise the cost stays unknown (None).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from qplan.config import Settings, openrouter_api_key

# Matches a complete <think>...</think> block (also across new lines).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """
    Remove Qwen3 reasoning blocks from a reply.

    Also handles a reply that was cut off inside an unfinished <think> block
    (no closing tag): everything from <think> onwards is then dropped.
    """
    text = _THINK_BLOCK.sub("", text or "")
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return text.strip()


def extract_json(text: str) -> dict:
    """
    Find and parse the first JSON object in a model reply.

    Models sometimes wrap JSON in ```json fences or add a sentence before
    it, so we look for the first "{" and try every matching "}" from the
    end backwards until json.loads succeeds.

    Raises ValueError if no valid JSON object is found.
    """
    text = strip_think(text)
    # Remove Markdown code fences, keep their content.
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in the reply")
    # Try the longest candidate first, then shorter ones.
    for end in range(len(text), start, -1):
        if text[end - 1] != "}":
            continue
        try:
            value = json.loads(text[start:end])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("the reply contains '{' but no valid JSON object")


# Allowed values of the reasoning settings (see config.py) and what is sent
# to OpenRouter for each. "default" sends nothing: the model's own default.
REASONING_VALUES = ("default", "off", "low", "medium", "high")

# A reply cut off by the token limit is retried ONCE with this factor more
# room, up to MAX_RETRY_TOKENS.
RETRY_FACTOR = 2
MAX_RETRY_TOKENS = 16384


def reasoning_body(setting: str) -> dict | None:
    """
    The OpenRouter "reasoning" request parameter for a reasoning setting.

    OpenRouter docs: max_tokens covers hidden reasoning AND the visible
    answer; {"effort": "none"} switches reasoning off. (A model that cannot
    switch reasoning off rejects "none"; use "low" or "default" for it.)
    """
    if setting == "default":
        return None
    if setting == "off":
        return {"effort": "none"}
    return {"effort": setting}          # low | medium | high


@dataclass
class LLMReply:
    """What one chat call returns, with its measurements."""
    text: str                      # reply with <think> blocks removed
    raw_text: str                  # reply exactly as the model sent it
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None      # USD; None = unknown
    cost_source: str | None = None # "openrouter", "litellm", "prices.json" or None
    ms: int = 0                    # wall-clock time of the call
    fake: bool = False             # True in dry-run mode
    finish_reason: str | None = None   # "stop" = complete; "length" = cut off by max_tokens
    reasoning: str | None = None   # the model's hidden reasoning text, if returned
    max_tokens: int = 0            # the budget this call had
    reasoning_setting: str = "default"
    retry_of_truncated: bool = False   # True = this call repeats a cut-off one

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# Signature of a fake responder used in dry-run mode:
#   fake(role, messages) -> reply text, or a dict {"text": ..., "finish_reason": ...}
#   (the dict form lets tests simulate a reply cut off by the token limit)
FakeResponder = Callable[[str, list], object]


@dataclass
class LLMClient:
    """
    Makes chat calls and reports each one to `on_call` (the run logger).

    on_call(role, messages, reply) is called after every call, including a
    cut-off call that is retried, so nothing is missing from the logs.
    """
    settings: Settings
    on_call: Callable[[str, list, LLMReply], None] | None = None
    fake_responder: FakeResponder | None = None
    _api_key: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # In dry-run mode no key is needed; otherwise fail early and clearly.
        self._api_key = openrouter_api_key(required=not self.settings.dry_run)

    def chat(self, role: str, model: str, messages: list, temperature: float,
             max_tokens: int, reasoning: str = "default") -> LLMReply:
        """
        Send `messages` (OpenAI chat format) to `model` and return the reply.

        role      : "react", "memory" or "synth" - for logging/reporting.
        reasoning : default | off | low | medium | high (hidden "thinking").

        If the reply is cut off by the token limit (finish_reason "length"),
        the call is repeated ONCE with RETRY_FACTOR x more tokens. Both calls
        are logged (both are paid for); the second reply is returned.
        """
        reply = self._one_call(role, model, messages, temperature, max_tokens, reasoning)
        if reply.truncated and max_tokens < MAX_RETRY_TOKENS:
            bigger = min(max_tokens * RETRY_FACTOR, MAX_RETRY_TOKENS)
            reply = self._one_call(role, model, messages, temperature, bigger, reasoning,
                                   retry=True)
        return reply

    def _one_call(self, role, model, messages, temperature, max_tokens, reasoning,
                  retry: bool = False) -> LLMReply:
        start = time.perf_counter()
        if self.settings.dry_run:
            reply = self._fake_call(role, model, messages)
        else:
            reply = self._real_call(model, messages, temperature, max_tokens, reasoning)
        reply.ms = int((time.perf_counter() - start) * 1000)
        reply.max_tokens, reply.reasoning_setting = max_tokens, reasoning
        reply.retry_of_truncated = retry
        if self.on_call:
            self.on_call(role, messages, reply)
        return reply

    # ------------------------------------------------------------------
    def _fake_call(self, role: str, model: str, messages: list) -> LLMReply:
        """Dry-run: produce a deterministic reply without any network call."""
        if self.fake_responder is None:
            raise RuntimeError("dry-run mode needs a fake responder")
        out = self.fake_responder(role, messages)
        if isinstance(out, dict):
            raw, finish = str(out.get("text", "")), out.get("finish_reason", "stop")
        else:
            raw, finish = str(out), "stop"
        # Rough token estimate (4 characters per token) so reports still work.
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return LLMReply(text=strip_think(raw), raw_text=raw, model=f"fake:{model}",
                        prompt_tokens=prompt_chars // 4, completion_tokens=len(raw) // 4,
                        cost=0.0, cost_source="dry-run", fake=True, finish_reason=finish)

    def _real_call(self, model: str, messages: list, temperature: float,
                   max_tokens: int, reasoning: str) -> LLMReply:
        """Real call through LiteLLM -> OpenRouter."""
        # Imported here so that dry-run tests do not need to load LiteLLM
        # (it is slow to import).
        import litellm

        litellm.suppress_debug_info = True
        extra_body = {"usage": {"include": True}}       # ask OpenRouter for the cost
        body = reasoning_body(reasoning)
        if body is not None:
            extra_body["reasoning"] = body
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self._api_key,
            num_retries=2,                              # retry network hiccups
            extra_body=extra_body,
        )
        choice = response.choices[0]
        message = choice.message
        raw = message.content or ""
        # OpenRouter returns hidden reasoning in "reasoning"; LiteLLM exposes
        # it as reasoning_content (name differs between versions).
        hidden = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost, source = _read_cost(response, usage, model)
        return LLMReply(text=strip_think(raw), raw_text=raw, model=model,
                        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        cost=cost, cost_source=source,
                        finish_reason=getattr(choice, "finish_reason", None),
                        reasoning=str(hidden) if hidden else None)


def _read_cost(response, usage, model: str = "") -> tuple[float | None, str | None]:
    """
    Find the cost of a call, best source first:
      1. the real cost reported by OpenRouter. Depending on the LiteLLM
         version it is in usage.cost or in the hidden response headers;
      2. LiteLLM's own estimate from its price table;
      3. an estimate from prices.json in the qplan folder (optional file,
         prices in USD per 1M tokens, see prices.example.json);
      4. unknown (None).
    """
    cost = getattr(usage, "cost", None)
    if cost is None and isinstance(usage, dict):
        cost = usage.get("cost")
    hidden = getattr(response, "_hidden_params", None) or {}
    if cost is None:
        cost = (hidden.get("additional_headers") or {}).get("llm_provider-x-litellm-response-cost")
    if cost is not None:
        try:
            return float(cost), "openrouter"
        except (TypeError, ValueError):
            pass
    if hidden.get("response_cost") is not None:
        return float(hidden["response_cost"]), "litellm"
    estimate = _estimate_from_price_table(model, usage)
    if estimate is not None:
        return estimate, "prices.json"
    return None, None


def _estimate_from_price_table(model: str, usage) -> float | None:
    """Cost estimate from prices.json: {"<model>": {"input": $/1M, "output": $/1M}}."""
    from qplan.config import PROJECT_DIR
    path = PROJECT_DIR / "prices.json"
    if not path.is_file() or usage is None:
        return None
    try:
        prices = json.loads(path.read_text(encoding="utf-8")).get(model)
    except (OSError, json.JSONDecodeError):
        return None
    if not prices:
        return None
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    return (pt * float(prices.get("input", 0)) + ct * float(prices.get("output", 0))) / 1_000_000
