"""
config.py - the settings used by the answer evaluation.

Every setting has a default and can be overridden in the .env file (or as an
environment variable) with the name shown, e.g.  QARAG_ANSWER_MODEL=...
Command-line options of hotpot_ask_eval.py override them again.

This is the part of the indexer's qarag.config that the scoring phase needs.
The answering phase (--db) runs with the indexer's full qarag package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file in the current folder, so OPENROUTER_API_KEY (read by
# LiteLLM) and any QARAG_* overrides become available through os.getenv().
load_dotenv()


def _env(name: str, default: str) -> str:
    """Return environment variable `name`, or `default` if it is not set or empty."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    # ------------------------------------------------------------------ models
    # Fallback model when no answer model is set. The "openrouter/" prefix tells
    # LiteLLM to send the request to OpenRouter; the rest is the model id from
    # openrouter.ai/models.
    gen_model: str = field(default_factory=lambda: _env("QARAG_GEN_MODEL", "openrouter/qwen/qwen3-30b-a3b"))

    # Model that writes the answer (answering phase) and extracts the short
    # answer (scoring phase); empty = gen_model. --answer-model overrides it.
    answer_model: str = field(default_factory=lambda: _env("QARAG_ANSWER_MODEL", ""))
    answer_temperature: float = 0.1
    answer_max_tokens: int = 2000

    # ----------------------------------------------------------------- storage
    # Knowledge base folder (answering phase only): qa.sqlite and chroma/.
    db_dir: Path = field(default_factory=lambda: Path(_env("QARAG_DB_DIR", "data/index")))

    # Offline test mode: no model calls, no cost.
    dry_run: bool = False

    @property
    def sqlite_path(self) -> Path:
        return self.db_dir / "qa.sqlite"

    def effective_answer_model(self) -> str:
        return self.answer_model or self.gen_model
