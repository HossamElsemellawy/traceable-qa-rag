"""
embed.py - turn a query into a vector with the SAME model as the database.

The database's `meta` table names the model used at indexing time
(meta.embed_model). Using any other model gives plausible but wrong scores,
a silent failure, so the model name is never configurable here: it always
comes from the database.

Two cases:
  * embed_model == "mock-embedding": a test database built by
    `qarag index ... --dry-run`. The vectors are a word hash; we reproduce
    the exact same function (copied from the database contract, section 6.6).
    Completely offline.
  * any other value, e.g. "openai/text-embedding-3-large": a real call to
    OpenRouter's OpenAI-compatible embeddings endpoint.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass

MOCK_MODEL = "mock-embedding"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def mock_embed(text: str, dimensions: int = 256) -> list[float]:
    """
    The indexer's fake embedding (must stay identical to qarag's version).

    Every word is hashed into one of `dimensions` buckets; the vector is the
    normalised word count per bucket. Similar wording -> similar vectors.
    """
    vec = [0.0] * dimensions
    for word in re.findall(r"\w+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dimensions] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@dataclass
class EmbedResult:
    """One embedding plus its measurements (for the run log)."""
    vector: list[float]
    model: str
    tokens: int = 0
    cost: float | None = None
    ms: int = 0


class Embedder:
    """Embeds queries with the model named in the database."""

    def __init__(self, model: str, api_key: str | None):
        self.model = model
        self._client = None
        if model != MOCK_MODEL:
            if not api_key:
                raise RuntimeError(
                    f"The database uses the real embedding model {model!r}, which needs "
                    "OPENROUTER_API_KEY (even in --dry-run, only the LLM calls are faked)."
                )
            # Imported here so mock-only tests do not need the openai package.
            from openai import OpenAI
            self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    @property
    def is_mock(self) -> bool:
        return self.model == MOCK_MODEL

    def embed(self, text: str) -> EmbedResult:
        """Embed one piece of text."""
        start = time.perf_counter()
        if self.is_mock:
            vec = mock_embed(text)
            return EmbedResult(vector=vec, model=self.model, tokens=0, cost=0.0,
                               ms=int((time.perf_counter() - start) * 1000))
        response = self._client.embeddings.create(
            model=self.model, input=[text],
            extra_body={"usage": {"include": True}},   # ask OpenRouter for cost
        )
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0)
        cost = getattr(usage, "cost", None)
        return EmbedResult(vector=list(response.data[0].embedding), model=self.model,
                           tokens=tokens, cost=float(cost) if cost is not None else None,
                           ms=int((time.perf_counter() - start) * 1000))
