"""
answer.py - answer a question from the indexed Q&A pairs (the "LLM-S" role).

This is the simple Stage 0 RAG path, without planner or governance:

    question -> embed -> top-k Q&A pairs from Chroma (retrieval)
             -> the pairs + their exact source sentences are given to a model
             -> the model writes ONE answer using only that evidence,
                with numbered citations [1], [2] pointing to the pairs used

The same function will later be used by the HotpotQA evaluation (compare
the answer with the gold answer), so it returns structured data, and the
command-line `ask` command only formats it.

Grounding rules given to the model:
  * use only the evidence; never outside knowledge;
  * if the evidence does not contain the answer, say so (no guessing);
  * cite the evidence items used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from qarag import llm
from qarag.config import Settings
from qarag.models import QARecord, Sentence
from qarag.store import QAStore
from qarag.vector_store import VectorIndex

ANSWER_SYSTEM_PROMPT = """You answer questions using ONLY the numbered evidence provided.

Each evidence item is a question-and-answer pair extracted from a source document, followed by
the exact source sentences it was extracted from.

Rules:
1. Use only the evidence. Do not add facts from your own knowledge, even if you know them.
2. If the evidence does not contain the answer, reply exactly: "The indexed documents do not
   contain this information." Do not guess.
3. If the evidence is partial or conflicting, give what it supports and say what is missing or
   conflicting.
4. SYNTHESISE: read ALL evidence items and combine every item that is relevant into one
   coherent answer. Different items often hold different parts of the answer (for example one
   gives the date, another the place, a third a later detail) - bring them together, and do
   not repeat the same fact twice. Ignore items that are not relevant to the question.
5. Answer directly and concisely: first the answer itself, then any needed qualification. Keep
   names, numbers, dates and units exactly as in the evidence.
6. After each claim, cite the evidence items it comes from in square brackets, e.g. [1] or [2][3].
   Cite only items you actually used."""


@dataclass
class EvidenceItem:
    number: int                     # [n] used in the answer's citations
    score: float                    # retrieval similarity
    record: QARecord
    sentences: list[Sentence] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    text: str                       # the model's answer (with [n] citations)
    evidence: list[EvidenceItem]    # everything that was given to the model
    cited: list[int]                # evidence numbers the answer actually cites


def retrieve(question: str, store: QAStore, settings: Settings, k: int,
             where: Optional[dict] = None) -> list[EvidenceItem]:
    """Top-k pairs for the question, with their source sentences.
    `where` optionally restricts the search by document metadata (P4a)."""
    model, variant = store.get_meta("embed_model"), store.get_meta("embed_variant")
    if not model:
        raise ValueError("The database is empty - run `python -m qarag index ...` first.")
    # The question MUST be embedded with the same model the index was built with.
    if model == "mock-embedding":
        vector = llm.mock_embed([question], purpose="embed_query")[0]
    else:
        settings.validate()
        vector = llm.embed_texts([question], model, settings.openrouter_base_url,
                                 settings.openrouter_api_key, purpose="embed_query")[0]
    index = VectorIndex(settings.chroma_path, settings.collection_name)
    index.open(model, variant)
    hits = index.search(vector, k=k, where=where)
    records = store.get_pairs([h.pair_id for h in hits])

    items, sentence_cache = [], {}
    for hit in hits:
        rec = records.get(hit.pair_id)
        if rec is None:
            continue
        if rec.doc_id not in sentence_cache:
            sentence_cache[rec.doc_id] = {s.idx: s for s in store.load_sentences(rec.doc_id)}
        sents = [sentence_cache[rec.doc_id][i] for i in rec.sentence_ids if i in sentence_cache[rec.doc_id]]
        items.append(EvidenceItem(number=len(items) + 1, score=hit.score, record=rec, sentences=sents))
    return items


def _evidence_block(items: list[EvidenceItem]) -> str:
    lines = []
    for item in items:
        rec = item.record
        where = " | ".join(rec.sections) if rec.sections else ""
        pages = f" | pages {', '.join(map(str, rec.source_pages))}" if rec.source_pages else ""
        lines.append(f"[{item.number}] Document: {rec.citation()}{' | ' + where if where else ''}{pages}")
        lines.append(f"    Q: {rec.question}")
        lines.append(f"    A: {rec.answer}")
        for s in item.sentences:
            lines.append(f"    Source: {s.text}")
        lines.append("")
    return "\n".join(lines)


def answer_question(question: str, store: QAStore, settings: Settings, k: int = 8,
                    where: Optional[dict] = None) -> Answer:
    """Retrieve evidence and write a grounded answer."""
    items = retrieve(question, store, settings, k, where=where)
    if not items:
        return Answer(question, "The indexed documents do not contain this information.", [], [])

    if store.get_meta("embed_model") == "mock-embedding" or settings.dry_run:
        # Offline test: no model call; "answer" with the best pair, citing it.
        text = f"{items[0].record.answer} [1]"
    else:
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Evidence:\n\n{_evidence_block(items)}\nQuestion: {question}"},
        ]
        text = llm.chat(settings.effective_answer_model(), messages,
                        temperature=settings.answer_temperature, max_tokens=settings.answer_max_tokens,
                        purpose="answer")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()   # drop reasoning blocks

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(items)})
    return Answer(question=question, text=text, evidence=items, cited=cited)
