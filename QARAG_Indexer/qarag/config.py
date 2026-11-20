"""
config.py - all tunable settings in one place.

Every setting has a sensible default. You can override any of them in two ways:
  1. In the .env file (or as Windows environment variables), using the names
     shown in the right-hand column below, e.g.  QARAG_GEN_MODEL=...
  2. On the command line for the most common ones (see `python -m qarag index --help`).

Keeping settings here (instead of scattered through the code) means a model or
a chunk size can be changed for an experiment without touching any logic,
which is the "slot-based roles" principle from the framework document.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file in the current folder (F:\qarag) so OPENROUTER_API_KEY
# and any QARAG_* overrides become available through os.getenv().
load_dotenv()


def _env(name: str, default: str) -> str:
    """Return environment variable `name`, or `default` if it is not set or empty."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    # ------------------------------------------------------------------ models
    # Q&A generator (LLM-G). The "openrouter/" prefix tells LiteLLM to send the
    # request to OpenRouter; the rest is the model id from openrouter.ai/models.
    gen_model: str = field(default_factory=lambda: _env("QARAG_GEN_MODEL", "openrouter/qwen/qwen3-30b-a3b"))

    # Embedding model, as named on OpenRouter (no "openrouter/" prefix here,
    # because embeddings are called through OpenRouter's OpenAI-compatible
    # endpoint with the openai client - see llm.py).
    embed_model: str = field(default_factory=lambda: _env("QARAG_EMBED_MODEL", "openai/text-embedding-3-large"))

    # OpenRouter endpoint and key (the key comes from .env).
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY", ""))

    # Generation behaviour. Low temperature = more faithful, less creative output.
    gen_temperature: float = field(default_factory=lambda: float(_env("QARAG_GEN_TEMPERATURE", "0.1")))
    gen_max_tokens: int = field(default_factory=lambda: int(_env("QARAG_GEN_MAX_TOKENS", "6000")))
    gen_retries: int = 3            # attempts per chunk if the model's reply is not valid JSON
    max_pairs_per_chunk: int = 60   # safety cap so a confused model cannot flood the database

    # Answer style of the generated pairs (see prompts.py):
    #   "short"    many narrow questions, answer = the exact fact only
    #   "sentence" one fact per pair, answer = one complete sentence   (default)
    #   "detailed" one topic per pair, answer = 1-3 sentences with related details
    # A database holds ONE style; compare styles in separate --db folders.
    answer_style: str = field(default_factory=lambda: _env("QARAG_ANSWER_STYLE", "sentence"))

    # Coverage pass: after a chunk is processed, content sentences (at least
    # `coverage_min_words` words) that no pair cites are sent back to the model
    # once more, asking for pairs that cover them.
    coverage_pass: bool = field(default_factory=lambda: _env("QARAG_COVERAGE_PASS", "1") == "1")
    coverage_min_words: int = 5

    # ----------------------------------------------------------- consolidation
    # After all chunks of a document are done, candidate pairs whose QUESTIONS
    # are similar (cosine similarity >= merge_threshold, measured with the
    # embedding model) are shown to the model together, which merges true
    # duplicates into one pair citing every place the fact appears.
    # Lower threshold = more groups checked (more merge calls, catches more
    # paraphrases); the model still decides what is really the same fact.
    merge_enabled: bool = field(default_factory=lambda: _env("QARAG_MERGE", "1") == "1")
    merge_threshold: float = field(default_factory=lambda: float(_env("QARAG_MERGE_THRESHOLD", "0.82")))
    merge_max_group: int = 10       # largest group of candidates shown to the model at once
    # Model used for merging; empty = same as the generator.
    merge_model: str = field(default_factory=lambda: _env("QARAG_MERGE_MODEL", ""))

    # ---------------------------------------------------------------- answering
    # Model that writes the final answer in `ask` (the LLM-S role); empty = same
    # as the generator. Low temperature keeps answers close to the evidence.
    answer_model: str = field(default_factory=lambda: _env("QARAG_ANSWER_MODEL", ""))
    answer_temperature: float = 0.1
    answer_max_tokens: int = 2000
    answer_top_k: int = field(default_factory=lambda: int(_env("QARAG_ANSWER_TOP_K", "8")))

    # How many chunks are sent to the model at the same time (parallel requests).
    # Higher = faster, but may hit OpenRouter rate limits. 4 is a safe start.
    max_workers: int = field(default_factory=lambda: int(_env("QARAG_MAX_WORKERS", "4")))

    # ---------------------------------------------------------------- chunking
    # Size of each chunk sent to the generator, in approximate tokens
    # (1 token ~ 4 characters of English). The framework document's default is
    # 6000; we start smaller (1500) because models write more complete Q&A
    # coverage when each request covers less text. Tune it in experiments.
    chunk_tokens: int = field(default_factory=lambda: int(_env("QARAG_CHUNK_TOKENS", "1500")))
    # Overlap: this many tokens from the end of the previous chunk are shown
    # to the model as read-only CONTEXT (so it understands pronouns etc.), but
    # it is told not to write questions about them (avoids duplicate pairs).
    overlap_tokens: int = field(default_factory=lambda: int(_env("QARAG_OVERLAP_TOKENS", "200")))
    # Look-ahead: this many tokens from the START of the next chunk are shown
    # after the core. Documents often give a short statement and explain it in
    # the next paragraph; the model may use (and cite) these sentences to
    # complete an answer about a core sentence. 0 switches it off.
    lookahead_tokens: int = field(default_factory=lambda: int(_env("QARAG_LOOKAHEAD_TOKENS", "300")))
    chars_per_token: int = 4

    # --------------------------------------------------------------- embedding
    # Which text of each pair is turned into a vector:
    #   "qa"   -> one vector of "Question: ... Answer: ..."   (default)
    #   "q"    -> one vector of the question only            (framework's original design)
    #   "dual" -> two vectors per pair, one for the question and one for the answer
    embed_variant: str = field(default_factory=lambda: _env("QARAG_EMBED_VARIANT", "qa"))
    embed_batch_size: int = 100     # texts per embedding request

    # ----------------------------------------------------------------- storage
    # Folder holding both databases: qa.sqlite (the pairs) and chroma/ (vectors).
    db_dir: Path = field(default_factory=lambda: Path(_env("QARAG_DB_DIR", "data/index")))
    collection_name: str = "qa_pairs"

    # Sent with every web request. Wikipedia (and other Wikimedia sites) REJECT
    # requests (403 Forbidden) unless the User-Agent names the tool AND gives a
    # way to contact its operator. Put your own contact in .env, e.g.
    #   QARAG_USER_AGENT=qarag-research/0.1 (PhD research; mailto:you@example.com)
    # See https://meta.wikimedia.org/wiki/User-Agent_policy
    user_agent: str = field(default_factory=lambda: _env(
        "QARAG_USER_AGENT", "qarag-research/0.1 (PhD research prototype; contact not configured)"))

    # Offline test mode: fake generator + fake embeddings, no API calls, no cost.
    # Used to check the pipeline end to end before spending credits.
    dry_run: bool = False

    @property
    def sqlite_path(self) -> Path:
        return self.db_dir / "qa.sqlite"

    @property
    def chroma_path(self) -> Path:
        return self.db_dir / "chroma"

    def effective_embed_model(self) -> str:
        """In dry-run mode the vectors are fake, so they get a different model
        name; this stops fake vectors ever being mixed into a real index."""
        return "mock-embedding" if self.dry_run else self.embed_model

    def effective_gen_model(self) -> str:
        return "mock-generator" if self.dry_run else self.gen_model

    def effective_merge_model(self) -> str:
        return self.merge_model or self.gen_model

    def effective_answer_model(self) -> str:
        return self.answer_model or self.gen_model

    def validate(self) -> None:
        """Fail early, with a clear message, on settings that cannot work."""
        if self.answer_style not in ("short", "sentence", "detailed"):
            raise ValueError(f"answer_style must be 'short', 'sentence' or 'detailed', got {self.answer_style!r}")
        if not 0.0 < self.merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be between 0 and 1")
        if self.embed_variant not in ("qa", "q", "dual"):
            raise ValueError(f"embed_variant must be 'qa', 'q' or 'dual', got {self.embed_variant!r}")
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        if self.lookahead_tokens >= self.chunk_tokens:
            raise ValueError("lookahead_tokens must be smaller than chunk_tokens")
        if not self.dry_run and not self.openrouter_api_key.startswith("sk-or-"):
            raise ValueError("OPENROUTER_API_KEY is missing or invalid in .env "
                             "(or use --dry-run to test without API calls)")
