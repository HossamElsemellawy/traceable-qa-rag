"""
consolidate.py - merge repeated facts across a WHOLE document.

Problem: the generator works chunk by chunk, so a fact stated in the
introduction and again in a later section produces two separate candidate
pairs, and an answer whose parts are spread over several sections (e.g. a
list continued later) produces several partial pairs.

Technique (runs automatically once all chunks of a document are generated):

  1. EMBED each candidate's question (cached in SQLite, so re-running this
     step costs nothing extra).
  2. GROUP candidates whose questions are similar: cosine similarity at least
     `merge_threshold`. Grouping is "seed based": each group is built around
     one candidate and only contains candidates similar to THAT candidate,
     so groups cannot drift through long chains of loosely related questions.
     Groups are capped at `merge_max_group` members.
  3. MERGE: each group with 2+ members is shown to the model together with
     every sentence its members cite (with section names). The model decides
     which members really ask for the same information and writes ONE pair for
     each, combining all places; members that differ stay separate. Similarity
     only proposes candidates - the model makes the final decision, so a high
     similarity between "birth date" and "birth place" questions does not
     merge different facts.
  4. FINAL PAIRS: each final pair records
        sentence_ids  every supporting sentence (all sections, all chunks)
        sections      every sub-section those sentences belong to
        source_pages  every page (PDFs)
        merged_from   the candidate ids it replaces (for auditing)

Merging happens only WITHIN one document, never across documents: documents
can have different owners and, later, different access rules, so a pair must
never mix content from two documents.

Re-run with a different threshold anytime: `python -m qarag consolidate --threshold 0.85`
(uses the stored candidates; no new generation).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pydantic import ValidationError

from qarag import llm
from qarag.config import Settings
from qarag.models import Candidate, MergedGroup, Sentence
from qarag.prompts import JSON_REPAIR_MESSAGE, merge_system_prompt, merge_user_prompt
from qarag.store import QAStore


@dataclass
class ConsolidationReport:
    candidates: int = 0
    final_pairs: int = 0
    groups_checked: int = 0      # groups of 2+ similar candidates sent to the model
    merged_away: int = 0         # candidates absorbed into another pair
    merge_failures: int = 0      # groups whose merge call failed (members kept unmerged)


@dataclass
class FinalPair:
    question: str
    answer: str
    sentence_ids: list[int]
    merged_from: list[str]
    chunk_index: int


# ---------------------------------------------------------------------------
# 1. question embeddings (cached)
# ---------------------------------------------------------------------------
def _ensure_question_embeddings(store: QAStore, candidates: list[Candidate],
                                settings: Settings) -> np.ndarray:
    """Return a matrix with one L2-normalised question vector per candidate,
    embedding only those not cached yet for the current embedding model."""
    model = settings.effective_embed_model()
    missing = [c for c in candidates if c.q_embedding is None]
    if missing:
        texts = [c.question for c in missing]
        if settings.dry_run:
            vectors = llm.mock_embed(texts, purpose="embed_questions")
        else:
            vectors = llm.embed_texts(texts, settings.embed_model, settings.openrouter_base_url,
                                      settings.openrouter_api_key, batch_size=settings.embed_batch_size,
                                      purpose="embed_questions")
        for c, v in zip(missing, vectors):
            c.q_embedding = v
        store.save_candidate_embeddings([(c.cand_id, c.q_embedding) for c in missing], model)
    matrix = np.array([c.q_embedding for c in candidates], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


# ---------------------------------------------------------------------------
# 2. seed-based grouping
# ---------------------------------------------------------------------------
def group_similar(matrix: np.ndarray, threshold: float, max_group: int) -> list[list[int]]:
    """Return groups of candidate positions. Every candidate is in exactly one group."""
    similarity = matrix @ matrix.T                 # cosine similarity of every pair of questions
    n = len(matrix)
    assigned = np.zeros(n, dtype=bool)
    groups: list[list[int]] = []
    for seed in range(n):
        if assigned[seed]:
            continue
        # Unassigned candidates similar enough to the SEED, most similar first.
        others = [j for j in np.argsort(-similarity[seed])
                  if j != seed and not assigned[j] and similarity[seed, j] >= threshold]
        members = [seed] + [int(j) for j in others[:max_group - 1]]
        assigned[members] = True
        groups.append(members)
    return groups


# ---------------------------------------------------------------------------
# 3. merging one group with the model
# ---------------------------------------------------------------------------
def _merge_group(members: list[Candidate], sentences_by_idx: dict[int, Sentence],
                 doc_title: str, settings: Settings) -> list[FinalPair]:
    """Ask the model to merge one group. On any failure, return the members
    unmerged (never lose a fact because a merge call failed)."""
    numbered = [(n, c.question, c.answer, c.sentence_ids) for n, c in enumerate(members, start=1)]
    cited_ids = sorted({i for c in members for i in c.sentence_ids})
    cited = [sentences_by_idx[i] for i in cited_ids if i in sentences_by_idx]

    if settings.dry_run:
        raw = llm.mock_merge(numbered)
    else:
        messages = [{"role": "system", "content": merge_system_prompt(settings.answer_style)},
                    {"role": "user", "content": merge_user_prompt(doc_title, numbered, cited)}]
        raw = None
        for _ in range(settings.gen_retries):
            reply = llm.chat(settings.effective_merge_model(), messages,
                             temperature=settings.gen_temperature, max_tokens=settings.gen_max_tokens,
                             purpose="merge")
            try:
                raw = llm.parse_json_object(reply)
                break
            except (ValueError, TypeError):
                messages += [{"role": "assistant", "content": reply[:4000]},
                             {"role": "user", "content": JSON_REPAIR_MESSAGE}]
        if raw is None:
            raise RuntimeError("merge reply never became valid JSON")

    # ---- validate the groups the model returned
    result: list[FinalPair] = []
    covered: set[int] = set()
    for item in raw.get("groups", []):
        try:
            group = MergedGroup.model_validate(item)
        except (ValidationError, TypeError):
            continue
        nums = [m for m in dict.fromkeys(group.members) if 1 <= m <= len(members) and m not in covered]
        if not nums:
            continue
        covered.update(nums)
        chosen = [members[m - 1] for m in nums]
        member_ids = {i for c in chosen for i in c.sentence_ids}
        # Citations may only use sentences that were shown; never fewer than the members had.
        ids = sorted(({i for i in group.sentence_ids if i in cited_ids}) | member_ids)
        result.append(FinalPair(question=group.question, answer=group.answer, sentence_ids=ids,
                                merged_from=[c.cand_id for c in chosen],
                                chunk_index=min(c.chunk_index for c in chosen)))
    # Any member the model forgot is kept unchanged.
    for n, c in enumerate(members, start=1):
        if n not in covered:
            result.append(FinalPair(c.question, c.answer, list(c.sentence_ids), [c.cand_id], c.chunk_index))
    return result


# ---------------------------------------------------------------------------
# 4. the whole document
# ---------------------------------------------------------------------------
def consolidate_document(store: QAStore, doc_id: str, doc_title: str, settings: Settings,
                         log=print) -> ConsolidationReport:
    """Build the final pairs of one document from its candidates and save them
    (replacing any previous final pairs of that document)."""
    report = ConsolidationReport()
    candidates = store.load_candidates(doc_id, settings.effective_embed_model())
    report.candidates = len(candidates)
    if not candidates:
        store.replace_final_pairs(doc_id, [], settings.effective_gen_model())
        return report

    if settings.merge_enabled and len(candidates) > 1:
        matrix = _ensure_question_embeddings(store, candidates, settings)
        groups = group_similar(matrix, settings.merge_threshold, settings.merge_max_group)
    else:
        groups = [[i] for i in range(len(candidates))]

    sentences_by_idx = {s.idx: s for s in store.load_sentences(doc_id)}
    final: list[FinalPair] = []
    to_merge = [g for g in groups if len(g) > 1]
    for g in groups:
        if len(g) == 1:
            c = candidates[g[0]]
            final.append(FinalPair(c.question, c.answer, list(c.sentence_ids), [c.cand_id], c.chunk_index))
    report.groups_checked = len(to_merge)

    if to_merge:
        log(f"  merging: {len(to_merge)} group(s) of similar questions sent to the model")

        def run(group: list[int]) -> tuple[list[int], list[FinalPair] | None]:
            members = [candidates[i] for i in group]
            try:
                return group, _merge_group(members, sentences_by_idx, doc_title, settings)
            except Exception:  # noqa: BLE001 - keep members unmerged on failure
                return group, None

        with ThreadPoolExecutor(max_workers=max(1, settings.max_workers)) as pool:
            for group, merged in pool.map(run, to_merge):
                if merged is None:
                    report.merge_failures += 1
                    merged = [FinalPair(candidates[i].question, candidates[i].answer,
                                        list(candidates[i].sentence_ids), [candidates[i].cand_id],
                                        candidates[i].chunk_index) for i in group]
                final.extend(merged)

    # Safety net: identical questions left after merging are combined.
    by_question: dict[str, FinalPair] = {}
    for pair in sorted(final, key=lambda p: (min(p.sentence_ids or [0]), p.question)):
        key = pair.question.lower()
        if key in by_question:
            kept = by_question[key]
            kept.sentence_ids = sorted(set(kept.sentence_ids) | set(pair.sentence_ids))
            kept.merged_from += pair.merged_from
            kept.chunk_index = min(kept.chunk_index, pair.chunk_index)
        else:
            by_question[key] = pair
    final = list(by_question.values())

    report.final_pairs = len(final)
    report.merged_away = report.candidates - report.final_pairs
    store.replace_final_pairs(doc_id, final, settings.effective_gen_model())
    store.update_coverage(doc_id, settings.coverage_min_words)
    return report
