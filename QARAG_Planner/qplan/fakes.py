"""
fakes.py - fake model replies for --dry-run (no network, no cost).

The fakes follow the same reply formats as the real models, so a dry run
exercises the whole engine: JSON parsing, the ReAct loop, search, the memory
policy, LLM-S, citations, the run log and the trace.

Fake ReAct agent behaviour (deterministic):
    step 1                         -> search_kb(<the question>)
                                      (after a direct-first step 0: "More about: <the question>")
    last search found pairs [E#]   -> finish
    last search found no match     -> recall(<the question>)   (once)
    anything else                  -> finish
"""

from __future__ import annotations

import json
import re


def fake_responder(role: str, messages: list) -> str:
    """Return a fake reply text for the given role and chat messages."""
    if role == "react":
        return _fake_react(messages)
    if role == "memory":
        return json.dumps({"answer": "(dry-run) a fake answer from model memory.",
                           "confidence": "low"})
    if role in ("synth", "synth_direct"):          # synth_direct: the direct-first attempt (0.1.5)
        return _fake_synth(messages)
    raise ValueError(f"no fake for role {role!r}")


def _fake_react(messages: list) -> str:
    first_user = messages[1]["content"]
    question = first_user.split("\n", 1)[0].replace("Question:", "").strip()
    observations = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("Observation")]
    actions = [m["content"] for m in messages if m["role"] == "assistant"]
    # A <think> block is included on purpose: it must be stripped by llm.py.
    think = "<think>dry-run reasoning</think>\n"
    if not observations:
        if "Step 0 was done automatically" in first_user:     # direct-first already searched the question
            return think + json.dumps({"thought": "Step 0 searched the whole question; search the missing fact.",
                                       "action": "search_kb", "input": f"More about: {question}"})
        return think + json.dumps({"thought": "Start by searching the knowledge base for the whole question.",
                                   "action": "search_kb", "input": question})
    last = observations[-1]
    already_recalled = any('"recall"' in a for a in actions)
    recall_available = "- recall:" in messages[0]["content"]      # memory switched on
    if "No match" in last and not already_recalled and recall_available:
        return think + json.dumps({"thought": "The knowledge base has nothing; try model memory.",
                                   "action": "recall", "input": question})
    # react-v3 format: finish declares the missing facts.
    missing = [] if "matching pair" in last or "[M" in last else [question]
    return think + json.dumps({"thought": "I have what I can get.", "action": "finish",
                               "input": "Evidence collected.", "missing": missing})


def _fake_synth(messages: list) -> str:
    """Plain text like the Path 1 answer prompt (synth-v5): cites the first
    numbered items, or the refusal sentence if there are none."""
    evidence_block = messages[1]["content"].split("\nQuestion:")[0]
    numbers = re.findall(r"^\[(\d+)\]", evidence_block, flags=re.MULTILINE)
    if not numbers:
        return "The indexed documents do not contain this information."
    cites = "".join(f"[{n}]" for n in numbers[:3])
    return f"(dry-run) Answer built from the evidence {cites}."
