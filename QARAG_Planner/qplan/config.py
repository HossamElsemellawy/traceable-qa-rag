"""
config.py - all settings of qplan in one place.

Where a setting comes from (later sources win):
    1. the defaults written in the Settings class below;
    2. the .env file in the qplan folder (keys starting with QPLAN_);
    3. options typed on the command line (e.g. --max-steps 8).

Keeping every knob in one dataclass means a run can store its complete
configuration in the run log (Settings.to_dict()), which is what makes
experiments reproducible and comparable later.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# The qplan program folder (the folder that contains the "qplan" package).
# Example on the student's machine: F:\qplan
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Read F:\qplan\.env (if it exists) into environment variables.
# override=False: a variable already set in the cmd window wins over .env.
load_dotenv(PROJECT_DIR / ".env", override=False)

# Default chat model for every slot. "openrouter/" tells LiteLLM which
# provider to use; the rest is the OpenRouter model id.
DEFAULT_MODEL = "openrouter/qwen/qwen3-30b-a3b"

# Other words accepted for the memory switch, mapped to the official value.
MEMORY_POLICY_ALIASES = {"fallback": "on", "true": "on", "yes": "on", "1": "on",
                         "false": "off", "no": "off", "0": "off"}


def _env(name: str, default: str) -> str:
    """Read one QPLAN_ setting from the environment, with a default."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    """Every setting that affects a run. One object is passed everywhere."""

    # ---- database (the indexer's output, opened read-only) --------------
    # Folder that contains qa.sqlite and chroma\  e.g. F:\qarag\data\wiki_test
    db: str = field(default_factory=lambda: _env("QPLAN_DB", ""))

    # ---- model slots (all via OpenRouter, all swappable) ----------------
    # The ReAct agent: reasons and chooses the next action.
    react_model: str = field(default_factory=lambda: _env("QPLAN_REACT_MODEL", DEFAULT_MODEL))
    # The "memory" model: answers from its own trained knowledge (recall).
    memory_model: str = field(default_factory=lambda: _env("QPLAN_MEMORY_MODEL", DEFAULT_MODEL))
    # LLM-S: writes the final grounded answer with citations.
    synth_model: str = field(default_factory=lambda: _env("QPLAN_SYNTH_MODEL", DEFAULT_MODEL))

    # ---- ReAct loop -----------------------------------------------------
    # Maximum number of Thought/Action cycles before the loop is stopped
    # and LLM-S answers with whatever evidence was collected.
    max_steps: int = field(default_factory=lambda: int(_env("QPLAN_MAX_STEPS", "6")))
    # Temperature of the ReAct agent (low = more repeatable plans).
    react_temperature: float = field(default_factory=lambda: float(_env("QPLAN_REACT_TEMPERATURE", "0.0")))
    # Max tokens per ReAct step reply. On OpenRouter this budget covers the
    # model's hidden reasoning AND its visible JSON. 2048 was too small for
    # Qwen3 (a real step spent all 2048 on reasoning, the JSON was cut off).
    react_max_tokens: int = field(default_factory=lambda: int(_env("QPLAN_REACT_MAX_TOKENS", "4096")))

    # ---- hidden reasoning ("thinking") per model slot -------------------
    #   default : send nothing, the model decides (Qwen3: thinks)
    #   off     : reasoning switched off (OpenRouter effort "none"): cheaper,
    #             faster; the agent still writes its reasoning in "thought"
    #   low / medium / high : reasoning effort
    # QPLAN_REASONING sets all three slots; QPLAN_<SLOT>_REASONING overrides one.
    react_reasoning: str = field(default_factory=lambda: _env("QPLAN_REACT_REASONING", _env("QPLAN_REASONING", "default")))
    memory_reasoning: str = field(default_factory=lambda: _env("QPLAN_MEMORY_REASONING", _env("QPLAN_REASONING", "default")))
    synth_reasoning: str = field(default_factory=lambda: _env("QPLAN_SYNTH_REASONING", _env("QPLAN_REASONING", "default")))

    # ---- retrieval (search_kb action) -----------------------------------
    # How many Q&A pairs one search returns (after dedupe by pair_id).
    top_k: int = field(default_factory=lambda: int(_env("QPLAN_TOP_K", "5")))
    # A hit counts as a "match" only if its cosine score >= min_score.
    # If no hit reaches it, the search is reported as "no match".
    # 0.35 is a starting value: tune it by looking at the scores in the logs.
    min_score: float = field(default_factory=lambda: float(_env("QPLAN_MIN_SCORE", "0.35")))

    # ---- direct-first (qplan 0.1.5) -------------------------------------
    # Before the ReAct loop, search the WHOLE question once (like the Path 1
    # one-shot baseline) and let LLM-S try to answer from those pairs.
    #   status "answered" with a knowledge-base citation -> done, no ReAct;
    #   anything else -> the ReAct loop starts, with these pairs already
    #   collected (the agent sees them as step 0) - this is "escalation".
    # on (default) | off (= ReAct only, as in 0.1.4: for ablation runs).
    direct_first: str = field(default_factory=lambda: _env("QPLAN_DIRECT_FIRST", "on"))
    # Pairs returned by the direct search (8 = the Path 1 baseline's k).
    direct_top_k: int = field(default_factory=lambda: int(_env("QPLAN_DIRECT_TOP_K", "8")))

    # ---- parametric memory (recall action) ------------------------------
    # The switch that decides whether the agent may use the memory model:
    #   on   : (default) the agent may use recall for a fact the knowledge
    #          base does not contain. The code enforces "search first": recall
    #          is refused until at least one search_kb has been made.
    #   off  : knowledge base only. The recall action is removed from the
    #          agent's prompt and refused by the code.
    #   free : advanced/experiments: recall allowed at any time, even
    #          before searching.
    # "fallback" (the 0.1.0 name) is accepted and means "on".
    memory_policy: str = field(default_factory=lambda: _env("QPLAN_MEMORY_POLICY", "on"))
    memory_temperature: float = field(default_factory=lambda: float(_env("QPLAN_MEMORY_TEMPERATURE", "0.0")))
    memory_max_tokens: int = field(default_factory=lambda: int(_env("QPLAN_MEMORY_MAX_TOKENS", "1024")))

    # ---- LLM-S (framework: temperature 0.1, max tokens 512+) ------------
    # 0.1.6: temperature 0 and 4000 tokens, as the Path 1 baseline (base100_v3).
    synth_temperature: float = field(default_factory=lambda: float(_env("QPLAN_SYNTH_TEMPERATURE", "0.0")))
    # Higher than the framework's 512 because Qwen3 may spend tokens thinking.
    synth_max_tokens: int = field(default_factory=lambda: int(_env("QPLAN_SYNTH_MAX_TOKENS", "4000")))

    # ---- governance (version 1: accepted but NOT enforced) --------------
    # Every retrieval already carries the user's roles, so the filter can be
    # added later inside retrieve() in one place.
    roles: list = field(default_factory=lambda: ["public"])

    # ---- run storage ----------------------------------------------------
    # Folder for the run log database and the Markdown traces.
    runs_dir: str = field(default_factory=lambda: _env("QPLAN_RUNS_DIR", str(PROJECT_DIR / "runs")))

    # ---- testing --------------------------------------------------------
    # dry_run = True: every LLM call is replaced by a fake, offline reply.
    # Embeddings: a test DB built with mock-embedding is always offline;
    # a real DB still needs the real embedding model (one cheap call).
    dry_run: bool = False
    # Print every step to the screen while the run is going.
    verbose: bool = True

    def to_dict(self) -> dict:
        """Plain dictionary of all settings, stored with each run."""
        return asdict(self)

    def validate(self) -> None:
        """Stop early with a clear message if a setting is impossible."""
        # Accept any letter case and the old name "fallback" (= on).
        policy = str(self.memory_policy).strip().lower()
        policy = MEMORY_POLICY_ALIASES.get(policy, policy)
        if policy not in ("on", "off", "free"):
            raise ValueError(f"memory_policy must be on, off or free (got {self.memory_policy!r})")
        self.memory_policy = policy
        # direct_first: accept on/off and the usual yes/no words.
        direct = str(self.direct_first).strip().lower()
        direct = {"true": "on", "yes": "on", "1": "on", "false": "off", "no": "off", "0": "off"}.get(direct, direct)
        if direct not in ("on", "off"):
            raise ValueError(f"direct_first must be on or off (got {self.direct_first!r})")
        self.direct_first = direct
        if self.direct_top_k < 1:
            raise ValueError("direct_top_k must be at least 1")
        from qplan.llm import REASONING_VALUES
        for attr in ("react_reasoning", "memory_reasoning", "synth_reasoning"):
            value = str(getattr(self, attr)).strip().lower()
            if value not in REASONING_VALUES:
                raise ValueError(f"{attr} must be one of {', '.join(REASONING_VALUES)} (got {value!r})")
            setattr(self, attr, value)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


def openrouter_api_key(required: bool = True) -> str | None:
    """
    Return the OpenRouter key from .env / the environment.

    required=False is used in dry-run mode, where no key is needed.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if required and not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in F:\\qplan\\.env "
            "(see .env.example), or use --dry-run for an offline test."
        )
    return key
