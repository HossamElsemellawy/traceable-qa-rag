"""
generator.py - turn one chunk into validated CANDIDATE Q&A pairs (LLM-G role).

For each chunk:
  1. GENERATION call: build the prompt for the chosen answer style and call
     the model; if the reply is not valid JSON, tell the model and retry.
  2. VALIDATION: check every pair and DROP the bad ones instead of failing
     the chunk:
       - required fields missing or too short          -> dropped
       - sentence ids that were not shown in the chunk  -> those ids removed;
         pair dropped if no CORE sentence id remains (protects provenance)
       - duplicate questions within the chunk           -> merged (ids combined)
  3. COVERAGE PASS (optional, on by default): find content sentences of the
     core that no pair cites. If any, ask the model once more - in the same
     conversation, so it knows what it already wrote - to cover them.
  4. Return the candidates plus coverage numbers for this chunk.

The candidates are NOT final: consolidate.py later merges repeated facts
across the whole document.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from qarag import llm
from qarag.chunker import is_content_sentence
from qarag.config import Settings
from qarag.models import Chunk, GeneratedPair
from qarag.prompts import (JSON_REPAIR_MESSAGE, build_user_prompt, coverage_message,
                           generation_system_prompt)


@dataclass
class ChunkResult:
    chunk_index: int
    pairs: list[GeneratedPair] = field(default_factory=list)
    dropped: int = 0              # pairs removed by validation
    calls: int = 0                # model calls used (generation + retries + coverage)
    n_content: int = 0            # content sentences in the core
    n_cited: int = 0              # of those, cited by at least one pair
    coverage_added: int = 0       # pairs added by the coverage pass
    error: str | None = None      # set if the chunk failed completely


def _validate_pairs(raw: dict, chunk: Chunk, max_pairs: int,
                    existing: list[GeneratedPair] | None = None) -> tuple[list[GeneratedPair], int]:
    """Check each pair individually; keep the good ones, count the rest.
    `existing` = pairs already accepted for this chunk (used by the coverage
    pass so a repeated question is merged into the earlier pair, not duplicated)."""
    # Ids the model may cite: core, context before and look-ahead after.
    allowed_ids = chunk.core_ids | {s.idx for s in chunk.context} | {s.idx for s in chunk.lookahead}
    items = raw.get("pairs", [])
    if not isinstance(items, list):
        raise ValueError('"pairs" is not a list')

    accepted: list[GeneratedPair] = list(existing or [])
    by_question = {p.question.lower(): i for i, p in enumerate(accepted)}
    dropped = 0
    for item in items[:max_pairs]:
        try:
            pair = GeneratedPair.model_validate(item)
        except (ValidationError, TypeError):
            dropped += 1
            continue
        valid_ids = sorted({i for i in pair.sentence_ids if i in allowed_ids})
        if not any(i in chunk.core_ids for i in valid_ids):   # must rest on a CORE sentence
            dropped += 1
            continue
        key = pair.question.lower()
        if key in by_question:
            # Same question twice: keep one pair, but keep ALL supporting sentences.
            i = by_question[key]
            accepted[i] = accepted[i].model_copy(
                update={"sentence_ids": sorted(set(accepted[i].sentence_ids) | set(valid_ids))})
            continue
        by_question[key] = len(accepted)
        accepted.append(pair.model_copy(update={"sentence_ids": valid_ids}))
    dropped += max(0, len(items) - max_pairs)
    new_pairs = accepted[len(existing or []):]
    if existing is not None:
        existing[:] = accepted[:len(existing)]   # existing pairs may have gained sentence ids
    return new_pairs, dropped


def _uncovered(chunk: Chunk, pairs: list[GeneratedPair], min_words: int):
    cited = {i for p in pairs for i in p.sentence_ids}
    content = [s for s in chunk.core if is_content_sentence(s, min_words)]
    return content, [s for s in content if s.idx not in cited]


def _ask_json(settings: Settings, messages: list[dict], result: ChunkResult,
              purpose: str = "generate") -> dict:
    """Call the model and parse its JSON, retrying with a repair message.
    Raises RuntimeError if the API fails or the reply never becomes valid JSON."""
    last_error = "unknown error"
    for _ in range(settings.gen_retries):
        result.calls += 1
        try:
            reply = llm.chat(settings.gen_model, messages, temperature=settings.gen_temperature,
                             max_tokens=settings.gen_max_tokens, purpose=purpose)
        except Exception as error:  # noqa: BLE001 - network/API failure after llm.chat's own retries
            raise RuntimeError(f"API error: {type(error).__name__}: {str(error)[:300]}") from error
        try:
            parsed = llm.parse_json_object(reply)
            messages.append({"role": "assistant", "content": reply})   # keep the conversation
            return parsed
        except (ValueError, TypeError) as error:
            last_error = f"invalid JSON: {error}"
            messages.append({"role": "assistant", "content": reply[:4000]})
            messages.append({"role": "user", "content": JSON_REPAIR_MESSAGE})
    raise RuntimeError(f"gave up after {settings.gen_retries} attempts ({last_error})")


def generate_for_chunk(chunk: Chunk, doc_title: str, total_chunks: int,
                       settings: Settings) -> ChunkResult:
    """Run the generator (and coverage pass) on one chunk. Never raises:
    failures are reported in ChunkResult.error so one bad chunk cannot stop a
    whole document."""
    result = ChunkResult(chunk_index=chunk.chunk_index)

    # ---- dry run: deterministic fake pairs, no API call
    if settings.dry_run:
        raw = llm.mock_generate([(s.idx, s.text) for s in chunk.core], settings.answer_style)
        result.pairs, result.dropped = _validate_pairs(raw, chunk, settings.max_pairs_per_chunk)
        result.calls = 1
        content, missing = _uncovered(chunk, result.pairs, settings.coverage_min_words)
        result.n_content, result.n_cited = len(content), len(content) - len(missing)
        return result

    messages = [
        {"role": "system", "content": generation_system_prompt(settings.answer_style)},
        {"role": "user", "content": build_user_prompt(doc_title, chunk, total_chunks)},
    ]
    # ---- 1+2. generation and validation
    try:
        raw = _ask_json(settings, messages, result)
        result.pairs, result.dropped = _validate_pairs(raw, chunk, settings.max_pairs_per_chunk)
    except (RuntimeError, ValueError) as error:
        result.error = str(error)
        return result

    # ---- 3. coverage pass for content sentences nobody cited
    content, missing = _uncovered(chunk, result.pairs, settings.coverage_min_words)
    if settings.coverage_pass and missing:
        messages.append({"role": "user", "content": coverage_message(missing)})
        try:
            raw = _ask_json(settings, messages, result, purpose="coverage")
            added, dropped = _validate_pairs(raw, chunk, settings.max_pairs_per_chunk,
                                             existing=result.pairs)
            result.pairs.extend(added)
            result.dropped += dropped
            result.coverage_added = len(added)
        except (RuntimeError, ValueError):
            pass   # coverage pass is best-effort: keep the pairs we already have
        content, missing = _uncovered(chunk, result.pairs, settings.coverage_min_words)

    result.n_content, result.n_cited = len(content), len(content) - len(missing)
    return result
