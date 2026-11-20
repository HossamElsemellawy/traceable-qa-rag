"""
db.py - read-only access to one qarag database, and the shared retrieve().

A qarag database is a folder, e.g. F:\\qarag\\data\\wiki_test, containing:
    qa.sqlite   the master record (tables meta, documents, sentences, qa_pairs)
    chroma\\     the vector index (collection "qa_pairs")

Rules from the database contract (planner handoff, section 6) applied here:
    1. read-only: SQLite opened with mode=ro; no Chroma write method is ever
       called; the collection is opened with get_collection (a wrong path
       fails loudly instead of creating an empty collection);
    3. meta.schema_version must be a known version;
    4. columns are always selected by name (never SELECT *);
    5. only the tables meta, documents, sentences, qa_pairs are used;
    6. search hits are deduplicated by pair_id (ask Chroma for ~3k, keep the
       best score per pair_id, cut to k);
    8. ALL Chroma access goes through QADatabase.retrieve(), which already
       takes `roles`, so the governance filter can later be added in one
       place with no route around it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qplan.embed import Embedder

# Schema versions of qa.sqlite this program understands.
SUPPORTED_SCHEMA_VERSIONS = {"3"}
COLLECTION_NAME = "qa_pairs"


@dataclass
class KBHit:
    """One Q&A pair found by a search (one row of a QAResultSet)."""
    pair_id: str
    score: float                 # cosine similarity, 1 - distance (0..1)
    question: str
    answer: str
    doc_id: str
    title: str | None
    source: str | None           # URL or file of the source document
    sections: list               # every section the pair cites
    pages: list                  # PDF pages (empty for web pages)
    sentences: list              # supporting sentences: {idx, text, section, page}
    matched_kind: str | None = None   # which vector matched: qa / q / a
    chunk_index: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Everything one retrieve() call produced, kept for the logs."""
    query: str
    top_k: int
    roles: list
    hits: list = field(default_factory=list)          # list[KBHit], best first
    n_raw: int = 0               # vectors returned by Chroma before dedupe
    best_score: float | None = None
    embed_model: str = ""
    embed_tokens: int = 0
    embed_cost: float | None = None
    ms: int = 0


class QADatabase:
    """A qarag database opened read-only."""

    def __init__(self, db_dir: str | Path):
        self.dir = Path(db_dir)
        sqlite_path = self.dir / "qa.sqlite"
        chroma_path = self.dir / "chroma"
        if not sqlite_path.is_file():
            raise FileNotFoundError(f"qa.sqlite not found in {self.dir}")
        if not chroma_path.is_dir():
            raise FileNotFoundError(f"chroma folder not found in {self.dir}")

        # --- SQLite, read-only (mode=ro). as_posix() gives forward slashes,
        # which the SQLite URI format needs on Windows too.
        uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
        self.con = sqlite3.connect(uri, uri=True)
        self.con.row_factory = sqlite3.Row

        # --- meta table: key/value settings of the database.
        self.meta = {r["key"]: r["value"] for r in self.con.execute("SELECT key, value FROM meta")}
        version = str(self.meta.get("schema_version"))
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported database schema_version {version!r} "
                f"(this qplan supports {sorted(SUPPORTED_SCHEMA_VERSIONS)})."
            )
        self.embed_model = self.meta.get("embed_model", "")
        if not self.embed_model:
            raise ValueError("meta.embed_model is missing: cannot know how to embed queries")

        # Which optional columns exist (new ones are added by later indexer
        # versions, e.g. pair_type with P5). Checked once, used by name.
        self._pair_columns = {r["name"] for r in self.con.execute("PRAGMA table_info(qa_pairs)")}

        # --- Chroma, opened without creating anything.
        import chromadb  # imported here: slow import, only needed with a DB
        try:
            self._chroma = chromadb.PersistentClient(path=str(chroma_path))
            self.collection = self._chroma.get_collection(COLLECTION_NAME)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # Chroma's Rust core "panics" (PanicException, which is not a
            # normal Exception) when the database was written by a chromadb
            # version that this installed version does not understand.
            # Turn that crash into a clear instruction.
            if type(exc).__name__ == "PanicException" or "panicked" in str(exc):
                raise RuntimeError(
                    f"Chroma could not open {chroma_path} with the installed chromadb "
                    f"{getattr(chromadb, '__version__', '?')}.\n"
                    "The database was most likely built with a DIFFERENT chromadb version.\n"
                    "Fix: in the indexer's venv run  pip show chromadb  and install exactly that "
                    "version here:  pip install chromadb==<version>\n"
                    "Never use a NEWER version than the indexer: it would upgrade the indexer's "
                    "database files when opening them."
                ) from exc
            raise

        # The collection's own metadata must agree with the meta table.
        cmeta = self.collection.metadata or {}
        self.warnings: list[str] = []
        if cmeta.get("embed_model") and cmeta.get("embed_model") != self.embed_model:
            self.warnings.append(
                f"Chroma collection says embed_model={cmeta.get('embed_model')!r} but "
                f"meta says {self.embed_model!r}"
            )
        if cmeta.get("hnsw:space") not in (None, "cosine"):
            self.warnings.append(f"Chroma distance is {cmeta.get('hnsw:space')!r}, expected cosine")

        self.embedder: Embedder | None = None   # set by attach_embedder()

    # ------------------------------------------------------------------
    def attach_embedder(self, api_key: str | None) -> None:
        """Create the query embedder with the database's own model."""
        self.embedder = Embedder(self.embed_model, api_key)

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------
    def fingerprint(self) -> str:
        """
        Short ID of the database content, stored with every run.

        Built from the meta table and the row counts, so a run log entry can
        tell whether two runs used the same version of a database.
        """
        n_pairs = self.con.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0]
        n_docs = self.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        payload = json.dumps({"meta": self.meta, "pairs": n_pairs, "docs": n_docs}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def info(self) -> dict:
        """A summary of the database for the `info` command."""
        docs = [dict(r) for r in self.con.execute(
            "SELECT d.doc_id, d.title, d.status, d.doc_type, "
            "  (SELECT COUNT(*) FROM qa_pairs p WHERE p.doc_id = d.doc_id) AS n_pairs "
            "FROM documents d ORDER BY d.title")]
        return {
            "folder": str(self.dir),
            "meta": self.meta,
            "fingerprint": self.fingerprint(),
            "documents": docs,
            "n_documents_complete": sum(1 for d in docs if d["status"] == "complete"),
            "n_pairs": self.con.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0],
            "n_vectors": self.collection.count(),
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------------
    def doc_ids_for_titles(self, titles: list[str]) -> tuple[list[str], list[str]]:
        """
        Map document titles to doc_ids (exact title match).
        Returns (doc_ids found, titles not in the database).
        """
        wanted = [t for t in dict.fromkeys(titles) if t]
        found: dict = {}
        for i in range(0, len(wanted), 500):            # SQLite parameter limit
            part = wanted[i:i + 500]
            for r in self.con.execute(
                    f"SELECT title, doc_id FROM documents WHERE title IN ({','.join('?' * len(part))})", part):
                found.setdefault(r["title"], []).append(r["doc_id"])
        doc_ids = [d for tt in wanted for d in found.get(tt, [])]
        return doc_ids, [tt for tt in wanted if tt not in found]

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int, roles: list | None = None,
                 where: dict | None = None) -> RetrievalResult:
        """
        THE shared retriever. Every search of the database goes through here.

        query : the text to search for (the full question or a sub-query)
        top_k : number of distinct Q&A pairs to return
        roles : the user's roles. VERSION 1: accepted and logged, NOT enforced.
                Governance will add a role condition to `where` HERE, inside
                the vector search (never as a filter after the search).
        where : optional Chroma metadata filter, e.g. {"doc_type": "wikipedia article"}
        """
        if self.embedder is None:
            raise RuntimeError("call attach_embedder() before retrieve()")
        roles = list(roles or ["public"])
        start = time.perf_counter()
        result = RetrievalResult(query=query, top_k=top_k, roles=roles,
                                 embed_model=self.embed_model)

        # Step 1 - embed the query with the database's model.
        emb = self.embedder.embed(query)
        result.embed_tokens, result.embed_cost = emb.tokens, emb.cost

        # --- GOVERNANCE HOOK (stage 2) ------------------------------------
        # The role filter will be merged into `where` at this line, e.g.
        #   where = _and(where, role_filter(roles))
        # Nothing to do in version 1.
        # --------------------------------------------------------------------

        # Step 2 - vector search. Ask for ~3k vectors because one pair may
        # have several vectors (variant "dual", later P3a).
        n_total = self.collection.count()
        if n_total == 0:
            result.ms = int((time.perf_counter() - start) * 1000)
            return result
        n = min(3 * top_k, n_total)
        res = self.collection.query(query_embeddings=[emb.vector], n_results=n, where=where,
                                    include=["metadatas", "distances"])
        metas, dists = res["metadatas"][0], res["distances"][0]
        result.n_raw = len(metas)

        # Dedupe by pair_id: keep the best score (and which vector kind won).
        best: dict[str, tuple[float, str | None]] = {}
        for m, d in zip(metas, dists):
            score = 1.0 - float(d)                     # cosine distance -> similarity
            pid = m["pair_id"]
            if pid not in best or score > best[pid][0]:
                best[pid] = (score, m.get("kind"))
        ranked = sorted(best.items(), key=lambda x: -x[1][0])

        # Step 3 - read the pair details from SQLite (skip unfinished docs),
        # then cut to top_k.
        details = self._pair_details([pid for pid, _ in ranked])
        for pid, (score, kind) in ranked:
            if pid not in details:
                continue                               # pair of an unfinished document
            hit = details[pid]
            hit.score, hit.matched_kind = round(score, 4), kind
            result.hits.append(hit)
            if len(result.hits) >= top_k:
                break

        # (LLM-R rerank would be step 4 here; not part of version 1.)
        result.best_score = result.hits[0].score if result.hits else None
        result.ms = int((time.perf_counter() - start) * 1000)
        return result

    # ------------------------------------------------------------------
    def _pair_details(self, pair_ids: list[str]) -> dict[str, KBHit]:
        """Load question, answer, document and supporting sentences per pair."""
        if not pair_ids:
            return {}
        marks = ",".join("?" * len(pair_ids))
        rows = self.con.execute(
            "SELECT p.pair_id, p.doc_id, p.chunk_index, p.question, p.answer, "
            "       p.sentence_ids, p.sections, p.source_pages, "
            "       d.title, d.source, d.status AS doc_status "
            "FROM qa_pairs p JOIN documents d ON d.doc_id = p.doc_id "
            f"WHERE p.pair_id IN ({marks})", pair_ids).fetchall()
        out: dict[str, KBHit] = {}
        for r in rows:
            if r["doc_status"] != "complete":
                continue                               # contract: skip unfinished documents
            sids = _json_list(r["sentence_ids"])
            sentences = []
            if sids:
                sentences = [dict(s) for s in self.con.execute(
                    "SELECT idx, text, section, page FROM sentences "
                    f"WHERE doc_id = ? AND idx IN ({','.join('?' * len(sids))}) ORDER BY idx",
                    [r["doc_id"], *sids])]
            out[r["pair_id"]] = KBHit(
                pair_id=r["pair_id"], score=0.0, question=r["question"], answer=r["answer"],
                doc_id=r["doc_id"], title=r["title"], source=r["source"],
                sections=_json_list(r["sections"]), pages=_json_list(r["source_pages"]),
                sentences=sentences, chunk_index=r["chunk_index"])
        return out


def _json_list(value) -> list:
    """Parse a JSON list column; NULL or bad JSON gives an empty list."""
    if value in (None, ""):
        return []
    try:
        v = json.loads(value)
        return v if isinstance(v, list) else [v]
    except (TypeError, json.JSONDecodeError):
        return []
