"""
vector_store.py - the Chroma vector index over the Q&A pairs.

What gets embedded depends on the "variant" (chosen when the index is
created, see config.py):

    variant  records per pair   text embedded
    -------  ----------------   ----------------------------------------
    "qa"     1  (<id>::qa)      "Question: <question>\\nAnswer: <answer>"
    "q"      1  (<id>::q)       "<question>"
    "dual"   2  (<id>::q, ::a)  "<question>" and "<answer>" separately

Whatever the variant, every Chroma record carries the pair_id in its
metadata, and search results are merged back to ONE result per pair (the
best-scoring record). So the Q&A PAIR stays the unit - the property governance
will rely on later.

Document metadata (P4a) copied into every record, for filtered search:
    doc_type  e.g. "wikipedia article", "pdf"       (only if known)
    language  e.g. "en"                              (only if known)
    year      int, from the published date           (only if known)
A filter therefore EXCLUDES pairs whose document lacks that field.

Safety: the embedding model and variant are saved in the collection's
metadata. Appending with a different model or variant is refused, because
vectors from different models are not comparable and would silently ruin
search quality. Use --mode overwrite to rebuild with new settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb

from qarag.models import QARecord


def document_filter_fields(doc_type: Optional[str], language: Optional[str],
                           year: Optional[int]) -> dict:
    """Metadata used for filtering. Chroma cannot store None, so unknown
    fields are simply left out."""
    fields: dict = {}
    if doc_type:
        fields["doc_type"] = doc_type
    if language:
        fields["language"] = language
    if year:
        fields["year"] = int(year)
    return fields


def build_filter(doc_type: Optional[str] = None, language: Optional[str] = None,
                 year_from: Optional[int] = None, year_to: Optional[int] = None) -> Optional[dict]:
    """Translate command-line filters into a Chroma `where` clause, e.g.
    --type "wikipedia article" --since 2020  ->  {"$and": [{"doc_type": ...}, {"year": {"$gte": 2020}}]}"""
    parts = []
    if doc_type:
        parts.append({"doc_type": doc_type})
    if language:
        parts.append({"language": language})
    if year_from:
        parts.append({"year": {"$gte": int(year_from)}})
    if year_to:
        parts.append({"year": {"$lte": int(year_to)}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


@dataclass
class SearchHit:
    pair_id: str
    score: float          # cosine similarity, 1.0 = identical direction
    matched: str          # which record matched: "qa", "q" or "a"


def texts_for_pair(record: QARecord, variant: str) -> list[tuple[str, str]]:
    """Return (record_suffix, text) for the Chroma records of one pair."""
    if variant == "qa":
        return [("qa", f"Question: {record.question}\nAnswer: {record.answer}")]
    if variant == "q":
        return [("q", record.question)]
    if variant == "dual":
        return [("q", record.question), ("a", record.answer)]
    raise ValueError(f"unknown variant {variant!r}")


class VectorIndex:
    def __init__(self, path: Path, collection_name: str):
        path.mkdir(parents=True, exist_ok=True)
        # PersistentClient saves everything under `path` (the data/index/chroma folder).
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection_name
        self.collection = None

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Delete the whole collection (used by --mode overwrite)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # it did not exist yet - nothing to delete
        self.collection = None

    def open(self, embed_model: str, variant: str) -> None:
        """Open the collection, creating it if needed, and check that it was
        built with the same embedding model and variant."""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # cosine distance suits normalised text embeddings; the other two
            # keys are our own bookkeeping, checked below.
            metadata={"hnsw:space": "cosine", "embed_model": embed_model, "variant": variant},
            embedding_function=None,   # we always supply vectors ourselves
        )
        existing = self.collection.metadata or {}
        if existing.get("embed_model") != embed_model or existing.get("variant") != variant:
            raise ValueError(
                f"This index was built with embedding model '{existing.get('embed_model')}' and "
                f"variant '{existing.get('variant')}', but you asked for '{embed_model}' / '{variant}'. "
                "Vectors from different settings cannot be mixed. Run with --mode overwrite to rebuild, "
                "or use `python -m qarag reembed` to re-embed the existing pairs.")

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    # --------------------------------------------------------------- writing
    def add(self, records: list[QARecord], texts: list[tuple[QARecord, str, str]],
            vectors: list[list[float]]) -> None:
        """Store vectors. `texts` and `vectors` are aligned: one entry per
        Chroma record (one or two per pair depending on the variant)."""
        ids, docs, metas = [], [], []
        for record, suffix, text in texts:
            ids.append(f"{record.pair_id}::{suffix}")
            docs.append(text)
            metas.append({
                "pair_id": record.pair_id,
                "doc_id": record.doc_id,
                "kind": suffix,
                "chunk_index": record.chunk_index,
                "source": record.source,
                # Chroma metadata cannot hold lists, so sections are joined into one string.
                "sections": " | ".join(record.sections)[:1000],
                **document_filter_fields(record.doc_type, record.language, record.year),
            })
        # Chroma limits the size of one call; 1000 records per call is well within it.
        for start in range(0, len(ids), 1000):
            end = start + 1000
            # upsert = insert, or replace if the id already exists (safe on re-runs)
            self.collection.upsert(ids=ids[start:end], documents=docs[start:end],
                                   metadatas=metas[start:end], embeddings=vectors[start:end])

    def update_document_fields(self, doc_id: str, fields: dict) -> int:
        """Set the metadata filter fields on every record of one document (after
        its metadata was refreshed) without re-embedding. Returns records updated."""
        if self.collection is None or not fields:
            return 0
        found = self.collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        if not found["ids"]:
            return 0
        metas = [{**m, **fields} for m in found["metadatas"]]
        self.collection.update(ids=found["ids"], metadatas=metas)
        return len(found["ids"])

    def delete_document(self, doc_id: str) -> None:
        """Remove every vector of one document (when it changed and is re-indexed)."""
        if self.collection is not None:
            self.collection.delete(where={"doc_id": doc_id})

    # ------------------------------------------------------------- searching
    def search(self, query_vector: list[float], k: int = 5,
               where: Optional[dict] = None) -> list[SearchHit]:
        """Top-k pairs for a query vector. Asks Chroma for extra results
        because the "dual" variant can return the same pair twice.
        `where` is an optional Chroma metadata filter (see build_filter)."""
        n = min(max(k * 3, k), max(self.count(), 1))
        result = self.collection.query(query_embeddings=[query_vector], n_results=n,
                                       include=["metadatas", "distances"], where=where or None)
        best: dict[str, SearchHit] = {}
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            score = 1.0 - float(distance)        # Chroma returns cosine DISTANCE = 1 - similarity
            pid = meta["pair_id"]
            if pid not in best or score > best[pid].score:
                best[pid] = SearchHit(pair_id=pid, score=score, matched=meta["kind"])
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
