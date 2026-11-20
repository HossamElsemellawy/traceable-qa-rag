"""
evidence.py - the evidence collected during one run (the "collector").

Two kinds of evidence:
    [E#] knowledge-base pairs found by search_kb (verified by source documents)
    [M#] answers from model memory found by recall (NOT verified)

Collector rules (framework section 5.1, with the agreed correction):
    * merge  : evidence from every step goes into one store;
    * dedupe : by pair_id (NOT chunk_index - one chunk holds many different
               facts). A pair found again keeps its first label [E#], its
               best score, and the list of every search that found it;
    * rank   : for LLM-S, knowledge-base evidence sorted by best score, then
               memory evidence in the order it was obtained.

Every item records which step and which search text found it ("found_by"),
which answers the open question of how sub-query results are labelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from qplan.db import KBHit


@dataclass
class KBEvidence:
    ev_id: str                   # "E1", "E2", ...
    hit: KBHit                   # the pair (score = best score so far)
    found_by: list = field(default_factory=list)   # [{"step": n, "query": "...", "score": s}]


@dataclass
class MemoryEvidence:
    ev_id: str                   # "M1", "M2", ...
    question: str                # what the agent asked the memory model
    answer: str
    confidence: str
    model: str
    step: int


@dataclass
class NoMatch:
    """A search that found nothing above the score threshold."""
    step: int
    query: str
    best_score: float | None


class EvidenceStore:
    """All evidence of one run, deduplicated and labelled."""

    def __init__(self) -> None:
        self.kb: dict[str, KBEvidence] = {}        # pair_id -> evidence
        self.memory: list[MemoryEvidence] = []
        self.no_match: list[NoMatch] = []
        # Facts the agent declared missing at its final finish (react-v3).
        self.missing: list[str] = []

    # ---- knowledge base ---------------------------------------------------
    def add_hit(self, hit: KBHit, step: int, query: str) -> tuple[KBEvidence, bool]:
        """
        Add one search hit. Returns (evidence item, is_new).

        If the pair was already found by an earlier search, the existing
        label is kept, the best score is kept, and this search is added to
        found_by.
        """
        record = {"step": step, "query": query, "score": hit.score}
        existing = self.kb.get(hit.pair_id)
        if existing:
            existing.found_by.append(record)
            if hit.score > existing.hit.score:
                existing.hit.score = hit.score
            return existing, False
        ev = KBEvidence(ev_id=f"E{len(self.kb) + 1}", hit=hit, found_by=[record])
        self.kb[hit.pair_id] = ev
        return ev, True

    def add_no_match(self, step: int, query: str, best_score: float | None) -> None:
        self.no_match.append(NoMatch(step, query, best_score))

    # ---- model memory -----------------------------------------------------
    def add_memory(self, question: str, answer: str, confidence: str,
                   model: str, step: int) -> MemoryEvidence:
        ev = MemoryEvidence(ev_id=f"M{len(self.memory) + 1}", question=question,
                            answer=answer, confidence=confidence, model=model, step=step)
        self.memory.append(ev)
        return ev

    # ---- views ------------------------------------------------------------
    def ranked_kb(self) -> list[KBEvidence]:
        """Knowledge-base evidence, best score first (primacy for LLM-S)."""
        return sorted(self.kb.values(), key=lambda e: -e.hit.score)

    def by_id(self) -> dict:
        """Map every label (E1, M1, ...) to its evidence item."""
        out: dict = {e.ev_id: e for e in self.kb.values()}
        out.update({m.ev_id: m for m in self.memory})
        return out

    def is_empty(self) -> bool:
        return not self.kb and not self.memory

    def to_list(self) -> list[dict]:
        """Plain dictionaries for the run log."""
        items = []
        for e in self.ranked_kb():
            items.append({"ev_id": e.ev_id, "kind": "kb", "pair_id": e.hit.pair_id,
                          "score": e.hit.score, "question": e.hit.question,
                          "answer": e.hit.answer, "title": e.hit.title,
                          "source": e.hit.source, "detail": e.hit.to_dict(),
                          "found_by": e.found_by})
        for m in self.memory:
            items.append({"ev_id": m.ev_id, "kind": "memory", "pair_id": None,
                          "score": None, "question": m.question, "answer": m.answer,
                          "title": None, "source": f"model memory: {m.model}",
                          "detail": asdict(m), "found_by": [{"step": m.step, "query": m.question}]})
        return items
