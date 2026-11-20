"""
llm.py - the only module that talks to OpenRouter.

Two kinds of call:
  * chat_json(...)  - send a prompt to a chat model and get a JSON object back.
                      Goes through LiteLLM, so changing the generator model is
                      just a different model string (e.g. another open model).
  * embed_texts(...) - turn texts into vectors. OpenRouter's embeddings
                      endpoint is OpenAI-compatible, so we use the official
                      openai client pointed at OpenRouter (verified in the
                      smoke test).

Both functions retry automatically on temporary problems (rate limits,
network hiccups, server errors) with an increasing wait between attempts.

USAGE METER (P4b): every call records its tokens and cost in `METER`, tagged
with a PURPOSE (generate, merge, answer, embed_pairs, ...) and a SCOPE (the
document being indexed, or "" for work that is not tied to one document).
The indexer and the `ask` command read the meter at the end and save it in
the `runs` table, which the `report` command turns into a readable report.

Where the cost comes from, in order of preference:
  1. the cost OpenRouter reports in the response (exact, in USD), else
  2. an estimate from prices you set in .env (USD per 1 million tokens):
       QARAG_PRICE_GEN_IN, QARAG_PRICE_GEN_OUT   chat models
       QARAG_PRICE_EMBED                         embedding model
  3. otherwise the cost is unknown ("n/a"); token counts are always recorded.

The mock_* functions are used by --dry-run: they imitate the real calls
without network access or cost, so the whole pipeline can be tested offline.
They also record ESTIMATED token counts (characters / 4), so the report can be
tested offline too.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Optional

import litellm

# LiteLLM prints a lot of "Provider List" help text on errors; hide it.
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Usage meter: tokens, calls and cost per purpose and per document
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0          # tokens sent to the model (input)
    completion_tokens: int = 0      # tokens written by the model (output); 0 for embeddings
    cost_usd: float = 0.0           # sum of known costs
    calls_without_cost: int = 0     # calls whose cost was neither reported nor estimable

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd
        self.calls_without_cost += other.calls_without_cost

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_text(self) -> str:
        """Cost for display: exact/estimated amount, marked if some calls had no cost."""
        if self.calls and self.calls_without_cost == self.calls:
            return "n/a"
        text = money(self.cost_usd)
        return text + (" (partial)" if self.calls_without_cost else "")


def money(value: float) -> str:
    """$0.1234, or more decimals for the tiny amounts embeddings cost."""
    return f"${value:.4f}" if value >= 0.01 or value == 0 else f"${value:.6f}"


def _price(name: str) -> Optional[float]:
    """Price in USD per 1M tokens from .env, or None if not set."""
    value = os.getenv(name)
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


class UsageMeter:
    """Thread-safe counter. Generation runs in several threads at once, so
    every update happens under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], Usage] = {}   # (scope, purpose) -> Usage
        self._scope = ""

    def reset(self) -> None:
        with self._lock:
            self._data.clear()
            self._scope = ""

    @contextmanager
    def scope(self, name: str):
        """Attribute every call made inside the `with` block to `name` (a doc_id).
        Documents are processed one after another, so one current scope is enough
        (the worker threads of one document all read the same value)."""
        previous, self._scope = self._scope, name
        try:
            yield
        finally:
            self._scope = previous

    def record(self, purpose: str, prompt_tokens: int, completion_tokens: int,
               cost: Optional[float], kind: str = "chat") -> None:
        if cost is None:
            # No cost from OpenRouter: estimate from the prices in .env, if any.
            if kind == "embed":
                p = _price("QARAG_PRICE_EMBED")
                cost = None if p is None else prompt_tokens * p / 1e6
            else:
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
        """{scope: {purpose: usage-as-dict}} - JSON-ready, saved in the runs table."""
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


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)        # ~4 characters per token for English


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
# Chat (Q&A generation)
# ---------------------------------------------------------------------------
def chat(model: str, messages: list[dict], temperature: float, max_tokens: int,
         network_retries: int = 5, purpose: str = "chat") -> str:
    """One chat completion, retried on temporary errors. Returns the reply text.
    `purpose` labels the call in the usage meter (generate, merge, answer ...)."""
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


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
_client_cache: dict[tuple[str, str], Any] = {}


def _openai_client(base_url: str, api_key: str):
    """Create the openai client once and reuse it (it keeps connections open)."""
    key = (base_url, api_key)
    if key not in _client_cache:
        from openai import OpenAI
        _client_cache[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _client_cache[key]


def embed_texts(texts: list[str], model: str, base_url: str, api_key: str,
                batch_size: int = 100, network_retries: int = 5,
                purpose: str = "embed") -> list[list[float]]:
    """Embed many texts, `batch_size` per request, keeping the input order."""
    client = _openai_client(base_url, api_key)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        delay = 2.0
        for attempt in range(1, network_retries + 1):
            try:
                response = client.embeddings.create(model=model, input=batch)
                prompt, _, cost = _usage_from_response(response)
                METER.record(purpose, prompt or sum(map(_estimate_tokens, batch)), 0, cost, kind="embed")
                # The API returns items with an .index; sort to be safe about order.
                items = sorted(response.data, key=lambda item: item.index)
                vectors.extend(item.embedding for item in items)
                break
            except Exception as error:  # noqa: BLE001
                if attempt == network_retries or not _is_retryable(error):
                    raise
                time.sleep(delay)
                delay *= 2
    return vectors


# ---------------------------------------------------------------------------
# Offline mocks for --dry-run
# ---------------------------------------------------------------------------
def _mock_record(purpose: str, prompt_text: str, reply: Any, kind: str = "chat") -> None:
    """Record ESTIMATED tokens for a fake call (cost 0), so --dry-run exercises the report."""
    out = 0 if kind == "embed" else _estimate_tokens(json.dumps(reply))
    METER.record(purpose, _estimate_tokens(prompt_text), out, 0.0, kind=kind)


def mock_generate(core_sentences: list[tuple[int, str]], style: str = "sentence") -> dict[str, Any]:
    """Pretend to be the generator: one simple pair per sentence. The question
    uses the first words of the sentence, so a sentence repeated elsewhere in
    the document produces a near-identical question - which lets the offline
    test exercise the consolidation (merge) step."""
    pairs = []
    for idx, text in core_sentences:
        words = text.split()
        if len(words) < 4:
            continue
        answer = " ".join(words[-3:]).rstrip(".") if style == "short" else text
        pairs.append({"question": f"What does the document say about {' '.join(words[:6])}?",
                      "answer": answer, "sentence_ids": [idx]})
    result = {"pairs": pairs}
    _mock_record("generate", " ".join(t for _, t in core_sentences) + " " * 8000, result)  # + prompt rules
    return result


def mock_merge(candidates: list[tuple[int, str, str, list[int]]]) -> dict[str, Any]:
    """Pretend to be the merge model: candidates whose questions share most of
    their words are treated as the same fact; the others stay separate."""
    def words(q: str) -> set[str]:
        return set(re.findall(r"\w+", q.lower()))
    groups, used = [], set()
    for number, question, answer, ids in candidates:
        if number in used:
            continue
        members, answers, all_ids = [number], [answer], set(ids)
        for other, q2, a2, ids2 in candidates:
            if other not in used and other != number:
                a, b = words(question), words(q2)
                if len(a & b) / max(len(a | b), 1) >= 0.6:
                    members.append(other)
                    answers.append(a2)
                    all_ids |= set(ids2)
        used.update(members)
        merged_answer = answers[0] if len(set(answers)) == 1 else " ".join(dict.fromkeys(answers))
        groups.append({"members": members, "question": question, "answer": merged_answer,
                       "sentence_ids": sorted(all_ids)})
    result = {"groups": groups}
    _mock_record("merge", " ".join(q + " " + a for _, q, a, _ in candidates) + " " * 4000, result)
    return result


def mock_embed(texts: list[str], dimensions: int = 256, purpose: str = "embed") -> list[list[float]]:
    """Pretend embeddings: a deterministic vector built from word hashes, so
    texts sharing words get similar vectors (good enough to test search)."""
    if texts:
        _mock_record(purpose, " ".join(texts), None, kind="embed")
    vectors = []
    for text in texts:
        vec = [0.0] * dimensions
        for word in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors
