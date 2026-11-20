"""
store.py - the SQLite database that is the "source of truth" for Q&A pairs.

Why SQLite next to the vector database?
  Chroma is good at similarity search but not at bookkeeping. SQLite keeps
  everything needed to manage the pairs: which documents were indexed, their
  sentences with sections and pages (provenance), which chunks are finished
  (so an interrupted run can RESUME instead of paying for generation again),
  the raw candidates (so the merge step can be re-run cheaply), and each final
  pair's lifecycle status. It is a single file (qa.sqlite) - open it with
  "DB Browser for SQLite" to inspect everything by hand.

Tables
  documents      one row per indexed source (URL or file), with coverage numbers
  sentences      every sentence: number, page, section path, text
  chunks         one row per chunk, status 'pending' or 'done'
  qa_candidates  raw pairs from the generator, per chunk (before merging)
  qa_pairs       FINAL pairs after document-level consolidation, status:
                   generated -> embedded      (Stage 0)
                 governance will later add 'under_review', 'approved', 'rejected'
  meta           settings this database was built with (embedding model,
                 variant, answer style, schema version)
  runs           one row per `index` / `consolidate` / `reembed` / `ask` run:
                 settings, duration, model calls, tokens and cost  (P4b)
  run_documents  per run and document: what happened, pairs, coverage,
                 calls, tokens and cost                            (P4b)

Document metadata (P4a): documents.authors, published, modified, language,
doc_type, doi, publisher, description, file_name, plus the full extracted
metadata as JSON in documents.metadata.

Schema versions: 2 = before P4; 3 = metadata + run log. A version-2 database
is upgraded in place (columns and tables are added, nothing is lost); its
documents get their metadata the next time they are indexed in append mode
(an unchanged document is skipped but its metadata is refreshed, at no cost).

Governance-ready columns (unused in Stage 0, defaults keep everything public):
  documents.classification / documents.doc_roles,
  qa_pairs.allowed_roles / qa_pairs.access_mode.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

from qarag.models import Candidate, Chunk, GeneratedPair, QARecord, RawDocument, Sentence

if TYPE_CHECKING:  # only for type hints, avoids a circular import with consolidate.py
    from qarag.consolidate import FinalPair

SCHEMA_VERSION = "3"

# Metadata columns added to `documents` in schema 3 (name, SQL type).
METADATA_COLUMNS = [
    ("authors", "TEXT"),        # JSON list of names
    ("published", "TEXT"),      # YYYY[-MM[-DD]]
    ("modified", "TEXT"),
    ("language", "TEXT"),       # e.g. 'en', 'ar'
    ("doc_type", "TEXT"),       # 'wikipedia article', 'scholarly article', 'web page', 'pdf', ...
    ("doi", "TEXT"),
    ("publisher", "TEXT"),
    ("description", "TEXT"),
    ("file_name", "TEXT"),
    ("metadata", "TEXT"),       # JSON: everything extracted, including fields above
]

RUN_TABLES = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    command         TEXT NOT NULL,         -- index | consolidate | reembed | ask
    started_at      TEXT,
    finished_at     TEXT,
    duration_s      REAL,
    status          TEXT,                  -- running | finished | failed
    settings        TEXT,                  -- JSON: models, style, variant, chunk sizes, flags ...
    detail          TEXT,                  -- JSON: e.g. sources and actions, or the question asked
    usage           TEXT,                  -- JSON: {scope: {purpose: {calls, tokens, cost}}}
    calls           INTEGER,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cost_usd        REAL,
    cost_known      INTEGER                -- 1 if every call had a known cost
);
CREATE TABLE IF NOT EXISTS run_documents (
    run_id          INTEGER NOT NULL,
    doc_id          TEXT NOT NULL,
    title           TEXT,
    action          TEXT,                  -- indexed / reindexed / resumed / skipped / failed / ...
    chunks          INTEGER,
    candidates      INTEGER,
    pairs           INTEGER,
    coverage        TEXT,
    calls           INTEGER,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cost_usd        REAL,
    PRIMARY KEY (run_id, doc_id)
);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,      -- short hash of the source location
    source          TEXT UNIQUE NOT NULL,  -- URL or absolute file path
    kind            TEXT NOT NULL,         -- 'url' or 'file'
    title           TEXT,
    content_hash    TEXT NOT NULL,         -- hash of the extracted text, detects changed documents
    n_sentences     INTEGER,
    n_chunks        INTEGER,
    n_content_sentences INTEGER,           -- sentences long enough to carry a fact
    n_cited_sentences   INTEGER,           -- of those, cited by at least one final pair (coverage)
    n_candidates    INTEGER,               -- pairs before merging
    status          TEXT NOT NULL,         -- 'generating' | 'consolidating' | 'complete'
    gen_model       TEXT,
    prompt_version  TEXT,
    answer_style    TEXT,
    classification  TEXT DEFAULT 'public',     -- governance (later)
    doc_roles       TEXT DEFAULT '["public"]', -- governance (later), JSON list
    created_at      TEXT,
    updated_at      TEXT,
""" + ",\n".join(f"    {name:<15} {sql_type}" for name, sql_type in METADATA_COLUMNS) + """
);
CREATE TABLE IF NOT EXISTS sentences (
    doc_id   TEXT NOT NULL,
    idx      INTEGER NOT NULL,             -- sentence number within the document
    page     INTEGER,                      -- PDF page, NULL otherwise
    section  TEXT,                         -- heading path, e.g. 'Career > Film'
    text     TEXT NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
CREATE TABLE IF NOT EXISTS chunks (
    doc_id          TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    first_sentence  INTEGER,
    last_sentence   INTEGER,
    status          TEXT NOT NULL,         -- 'pending' or 'done'
    n_candidates    INTEGER DEFAULT 0,
    n_dropped       INTEGER DEFAULT 0,
    n_coverage_added INTEGER DEFAULT 0,    -- pairs added by the coverage pass
    n_calls         INTEGER DEFAULT 0,     -- model calls used
    error           TEXT,
    PRIMARY KEY (doc_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS qa_candidates (
    cand_id        TEXT PRIMARY KEY,       -- "<doc_id>-c<chunk>-<n>"
    doc_id         TEXT NOT NULL,
    chunk_index    INTEGER NOT NULL,
    question       TEXT NOT NULL,
    answer         TEXT NOT NULL,
    sentence_ids   TEXT NOT NULL,          -- JSON list
    q_embedding    BLOB,                   -- cached question vector (float32) for grouping
    q_embed_model  TEXT,                   -- model that produced q_embedding
    created_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_cand_doc ON qa_candidates(doc_id);
CREATE TABLE IF NOT EXISTS qa_pairs (
    pair_id        TEXT PRIMARY KEY,       -- "<doc_id>-p<n>"
    doc_id         TEXT NOT NULL,
    chunk_index    INTEGER NOT NULL,       -- first chunk where the fact appears
    question       TEXT NOT NULL,
    answer         TEXT NOT NULL,
    sentence_ids   TEXT NOT NULL,          -- JSON list of ALL supporting sentence numbers
    source_pages   TEXT NOT NULL,          -- JSON list of pages (empty if not a PDF)
    sections       TEXT NOT NULL,          -- JSON list of every sub-section cited
    merged_from    TEXT NOT NULL,          -- JSON list of candidate ids combined here
    allowed_roles  TEXT DEFAULT '["public"]',  -- governance (later)
    access_mode    TEXT DEFAULT 'inherit',     -- governance (later): 'inherit' or 'override'
    status         TEXT NOT NULL,          -- 'generated' or 'embedded'
    gen_model      TEXT,
    created_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_pairs_doc    ON qa_pairs(doc_id);
CREATE INDEX IF NOT EXISTS idx_pairs_status ON qa_pairs(status);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
""" + RUN_TABLES

# Content tables, emptied by --mode overwrite. The run log (runs, run_documents)
# is deliberately NOT in this list: the history of experiments - settings,
# tokens, cost - is kept when the content is rebuilt.
TABLES = ("qa_pairs", "qa_candidates", "chunks", "sentences", "documents", "meta")

# Columns read together with every pair (document title and metadata for citations).
_PAIR_SELECT = ("SELECT p.*, d.source, d.title, d.authors, d.published, d.doc_type, d.language "
                "FROM qa_pairs p JOIN documents d USING(doc_id)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: Optional[bytes]) -> Optional[list[float]]:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32).tolist()


class QAStore:
    """Thin wrapper around the SQLite file. All writes happen in the main
    thread (worker threads only call the model), so no locking is needed."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row      # rows behave like dicts: row["question"]
        self.upgraded = False                    # True if an older database was upgraded just now
        # A database created by an older version of this code has different
        # tables; detect it so the user gets a clear message, not a crash.
        has_tables = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'").fetchone()
        version = None
        if has_tables:
            row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() \
                if self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone() else None
            version = row["value"] if row else None
        if has_tables and version == "2":
            self._upgrade_from_2()               # P4: add metadata columns + run tables, keep all data
            version = SCHEMA_VERSION
        self.outdated = bool(has_tables) and version != SCHEMA_VERSION
        if not self.outdated:
            self._create()

    def _upgrade_from_2(self) -> None:
        """Schema 2 -> 3 in place: new columns are added (empty) and the run
        tables are created. Q&A pairs, sentences and vectors are untouched."""
        existing = {r["name"] for r in self.conn.execute("PRAGMA table_info(documents)")}
        with self.conn:
            for name, sql_type in METADATA_COLUMNS:
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {sql_type}")
            self.conn.executescript(RUN_TABLES)
            self.conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (SCHEMA_VERSION,))
        self.upgraded = True

    def _create(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                          (SCHEMA_VERSION,))
        self.conn.commit()

    def require_current(self) -> None:
        if self.outdated:
            raise ValueError(f"The database at {self.path} was created by an older version of qarag. "
                             "Rebuild it with --mode overwrite (or delete the folder).")

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- overwrite
    def reset(self) -> None:
        """Delete ALL content and recreate the tables (used by --mode overwrite).
        Dropping the tables also upgrades a database from an older version.
        The run log is kept (see TABLES)."""
        for table in TABLES:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.commit()
        self._create()
        self.outdated = False

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()

    # ------------------------------------------------------------- documents
    def get_document(self, source: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM documents WHERE source = ?", (source,)).fetchone()

    def complete_documents(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM documents WHERE status != 'generating' "
                                 "ORDER BY created_at").fetchall()

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and everything derived from it."""
        for table in ("qa_pairs", "qa_candidates", "chunks", "sentences", "documents"):
            self.conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

    def create_document(self, doc_id: str, document: RawDocument, content_hash: str,
                        sentences: list[Sentence], chunks: list[Chunk],
                        gen_model: str, prompt_version: str, answer_style: str) -> None:
        """Insert the document, its sentences and its (pending) chunks in one transaction."""
        now = _now()
        with self.conn:   # 'with' = one transaction: all rows are written, or none
            self.conn.execute(
                "INSERT INTO documents(doc_id, source, kind, title, content_hash, n_sentences, n_chunks, "
                "status, gen_model, prompt_version, answer_style, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, document.source, document.kind, document.title, content_hash, len(sentences),
                 len(chunks), "generating", gen_model, prompt_version, answer_style, now, now))
            self.conn.executemany(
                "INSERT INTO sentences(doc_id, idx, page, section, text) VALUES(?,?,?,?,?)",
                [(doc_id, s.idx, s.page, s.section, s.text) for s in sentences])
            self.conn.executemany(
                "INSERT INTO chunks(doc_id, chunk_index, first_sentence, last_sentence, status) "
                "VALUES(?,?,?,?,'pending')",
                [(doc_id, c.chunk_index, c.core[0].idx, c.core[-1].idx) for c in chunks])
            self._write_metadata(doc_id, document.metadata)

    # ------------------------------------------------------- metadata (P4a)
    @staticmethod
    def _metadata_values(metadata: dict) -> dict[str, Optional[str]]:
        """Column values for the documents table from an extracted metadata dict."""
        values: dict[str, Optional[str]] = {}
        for name, _ in METADATA_COLUMNS:
            if name == "metadata":
                values[name] = json.dumps(metadata, ensure_ascii=False, default=str)
            elif name == "authors":
                values[name] = json.dumps(metadata.get("authors") or [], ensure_ascii=False)
            else:
                value = metadata.get(name)
                values[name] = None if value in (None, "") else str(value)
        return values

    def _write_metadata(self, doc_id: str, metadata: dict) -> None:
        values = self._metadata_values(metadata or {})
        assignments = ", ".join(f"{k} = ?" for k in values)
        self.conn.execute(f"UPDATE documents SET {assignments} WHERE doc_id = ?",
                          [*values.values(), doc_id])

    def update_metadata(self, doc_id: str, metadata: dict, title: Optional[str] = None) -> bool:
        """Refresh a document's metadata (e.g. an unchanged document skipped in
        append mode). Returns True if anything changed."""
        row = self.conn.execute("SELECT metadata, title FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        new = self._metadata_values(metadata or {})["metadata"]
        if row is not None and row["metadata"] == new and (title is None or row["title"] == title):
            return False
        with self.conn:
            self._write_metadata(doc_id, metadata)
            if title:
                self.conn.execute("UPDATE documents SET title = ? WHERE doc_id = ?", (title, doc_id))
        return True

    def document_metadata(self, doc_id: str) -> dict:
        row = self.conn.execute("SELECT metadata FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return json.loads(row["metadata"]) if row and row["metadata"] else {}

    def set_document_status(self, doc_id: str, status: str) -> None:
        self.conn.execute("UPDATE documents SET status = ?, updated_at = ? WHERE doc_id = ?",
                          (status, _now(), doc_id))
        self.conn.commit()

    def load_sentences(self, doc_id: str) -> list[Sentence]:
        rows = self.conn.execute("SELECT idx, page, section, text FROM sentences WHERE doc_id = ? "
                                 "ORDER BY idx", (doc_id,)).fetchall()
        return [Sentence(idx=r["idx"], text=r["text"], page=r["page"], section=r["section"]) for r in rows]

    def chunk_bounds(self, doc_id: str) -> list[tuple[int, int, int]]:
        """(chunk_index, first_sentence, last_sentence) for every chunk, so an
        interrupted document is resumed with EXACTLY the same chunks even if
        the chunk-size settings were changed in between."""
        rows = self.conn.execute("SELECT chunk_index, first_sentence, last_sentence FROM chunks "
                                 "WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)).fetchall()
        return [(r["chunk_index"], r["first_sentence"], r["last_sentence"]) for r in rows]

    def pending_chunk_indexes(self, doc_id: str) -> set[int]:
        rows = self.conn.execute("SELECT chunk_index FROM chunks WHERE doc_id = ? AND status != 'done'",
                                 (doc_id,)).fetchall()
        return {r["chunk_index"] for r in rows}

    # ------------------------------------------------------------ candidates
    def save_chunk_candidates(self, doc_id: str, chunk_index: int, pairs: list[GeneratedPair],
                              dropped: int, coverage_added: int, calls: int) -> None:
        """Store the candidates of one chunk and mark the chunk done, atomically,
        so a crash can never leave a chunk half-saved."""
        now = _now()
        rows = [(f"{doc_id}-c{chunk_index}-{n}", doc_id, chunk_index, p.question, p.answer,
                 json.dumps(p.sentence_ids), now) for n, p in enumerate(pairs)]
        with self.conn:
            self.conn.execute("DELETE FROM qa_candidates WHERE doc_id = ? AND chunk_index = ?",
                              (doc_id, chunk_index))
            self.conn.executemany(
                "INSERT INTO qa_candidates(cand_id, doc_id, chunk_index, question, answer, sentence_ids, "
                "created_at) VALUES(?,?,?,?,?,?,?)", rows)
            self.conn.execute(
                "UPDATE chunks SET status='done', n_candidates=?, n_dropped=?, n_coverage_added=?, "
                "n_calls=?, error=NULL WHERE doc_id=? AND chunk_index=?",
                (len(pairs), dropped, coverage_added, calls, doc_id, chunk_index))

    def record_chunk_error(self, doc_id: str, chunk_index: int, error: str) -> None:
        """The chunk stays 'pending' so the next run in append mode retries it."""
        self.conn.execute("UPDATE chunks SET error = ? WHERE doc_id = ? AND chunk_index = ?",
                          (error, doc_id, chunk_index))
        self.conn.commit()

    def load_candidates(self, doc_id: str, embed_model: str) -> list[Candidate]:
        """Candidates of one document; cached question vectors are returned
        only if they were made with the current embedding model."""
        rows = self.conn.execute("SELECT * FROM qa_candidates WHERE doc_id = ? "
                                 "ORDER BY chunk_index, cand_id", (doc_id,)).fetchall()
        return [Candidate(cand_id=r["cand_id"], doc_id=r["doc_id"], chunk_index=r["chunk_index"],
                          question=r["question"], answer=r["answer"],
                          sentence_ids=json.loads(r["sentence_ids"]),
                          q_embedding=_from_blob(r["q_embedding"]) if r["q_embed_model"] == embed_model else None)
                for r in rows]

    def save_candidate_embeddings(self, items: list[tuple[str, list[float]]], embed_model: str) -> None:
        self.conn.executemany("UPDATE qa_candidates SET q_embedding = ?, q_embed_model = ? WHERE cand_id = ?",
                              [(_to_blob(v), embed_model, cid) for cid, v in items])
        self.conn.commit()

    # ----------------------------------------------------------- final pairs
    def replace_final_pairs(self, doc_id: str, pairs: list["FinalPair"], gen_model: str) -> None:
        """Replace all final pairs of a document (after consolidation). Pages
        and sections are derived from the cited sentences."""
        sentences = {s.idx: s for s in self.load_sentences(doc_id)}
        now = _now()
        rows = []
        ordered = sorted(pairs, key=lambda p: (min(p.sentence_ids or [0]), p.question))
        for n, p in enumerate(ordered):
            cited = [sentences[i] for i in p.sentence_ids if i in sentences]
            pages = sorted({s.page for s in cited if s.page is not None})
            sections = list(dict.fromkeys(s.section for s in cited if s.section))   # unique, in order
            rows.append((f"{doc_id}-p{n:04d}", doc_id, p.chunk_index, p.question, p.answer,
                         json.dumps(sorted(p.sentence_ids)), json.dumps(pages), json.dumps(sections),
                         json.dumps(p.merged_from), "generated", gen_model, now))
        with self.conn:
            self.conn.execute("DELETE FROM qa_pairs WHERE doc_id = ?", (doc_id,))
            self.conn.executemany(
                "INSERT INTO qa_pairs(pair_id, doc_id, chunk_index, question, answer, sentence_ids, "
                "source_pages, sections, merged_from, status, gen_model, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.conn.execute("UPDATE documents SET n_candidates = "
                              "(SELECT COUNT(*) FROM qa_candidates WHERE doc_id = ?) WHERE doc_id = ?",
                              (doc_id, doc_id))

    def update_coverage(self, doc_id: str, min_words: int) -> None:
        """Coverage = share of content sentences cited by at least one final pair."""
        sentences = self.load_sentences(doc_id)
        content = {s.idx for s in sentences if len(s.text.split()) >= min_words}
        cited: set[int] = set()
        for row in self.conn.execute("SELECT sentence_ids FROM qa_pairs WHERE doc_id = ?", (doc_id,)):
            cited.update(json.loads(row["sentence_ids"]))
        self.conn.execute("UPDATE documents SET n_content_sentences = ?, n_cited_sentences = ? "
                          "WHERE doc_id = ?", (len(content), len(content & cited), doc_id))
        self.conn.commit()

    def uncited_sentences(self, doc_id: str, min_words: int) -> list[Sentence]:
        """Content sentences no final pair cites - useful to inspect coverage gaps."""
        cited: set[int] = set()
        for row in self.conn.execute("SELECT sentence_ids FROM qa_pairs WHERE doc_id = ?", (doc_id,)):
            cited.update(json.loads(row["sentence_ids"]))
        return [s for s in self.load_sentences(doc_id)
                if len(s.text.split()) >= min_words and s.idx not in cited]

    def pairs_with_status(self, status: str) -> list[QARecord]:
        rows = self.conn.execute(
            _PAIR_SELECT + " WHERE p.status = ? ORDER BY p.pair_id", (status,)).fetchall()
        return [self._record(r) for r in rows]

    def mark_embedded(self, pair_ids: Iterable[str]) -> None:
        self.conn.executemany("UPDATE qa_pairs SET status = 'embedded' WHERE pair_id = ?",
                              [(pid,) for pid in pair_ids])
        self.conn.commit()

    def reset_embedding_status(self) -> None:
        """Mark every pair as needing embedding again (used by `reembed`)."""
        self.conn.execute("UPDATE qa_pairs SET status = 'generated' WHERE status = 'embedded'")
        self.conn.commit()

    def count_pairs(self, doc_id: str) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM qa_pairs WHERE doc_id = ?", (doc_id,)).fetchone()[0]

    def get_pairs(self, pair_ids: list[str]) -> dict[str, QARecord]:
        if not pair_ids:
            return {}
        marks = ",".join("?" * len(pair_ids))
        rows = self.conn.execute(
            _PAIR_SELECT + f" WHERE p.pair_id IN ({marks})", pair_ids).fetchall()
        return {r["pair_id"]: self._record(r) for r in rows}

    def all_pairs(self) -> list[QARecord]:
        rows = self.conn.execute(
            _PAIR_SELECT + " ORDER BY p.pair_id").fetchall()
        return [self._record(r) for r in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> QARecord:
        return QARecord(pair_id=row["pair_id"], doc_id=row["doc_id"], chunk_index=row["chunk_index"],
                        question=row["question"], answer=row["answer"],
                        sentence_ids=json.loads(row["sentence_ids"]),
                        source_pages=json.loads(row["source_pages"]),
                        sections=json.loads(row["sections"]),
                        merged_from=json.loads(row["merged_from"]),
                        source=row["source"], title=row["title"],
                        authors=json.loads(row["authors"]) if row["authors"] else [],
                        published=row["published"], doc_type=row["doc_type"], language=row["language"])

    # ------------------------------------------------------------- reporting
    def list_documents(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT d.*, (SELECT COUNT(*) FROM qa_pairs p WHERE p.doc_id = d.doc_id) AS n_pairs, "
            "(SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.doc_id AND c.status != 'done') AS pending "
            "FROM documents d ORDER BY d.created_at").fetchall()

    def stats(self) -> dict[str, object]:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        content = q("SELECT COALESCE(SUM(n_content_sentences), 0) FROM documents")
        cited = q("SELECT COALESCE(SUM(n_cited_sentences), 0) FROM documents")
        return {
            "documents": q("SELECT COUNT(*) FROM documents"),
            "documents_incomplete": q("SELECT COUNT(*) FROM documents WHERE status != 'complete'"),
            "sentences": q("SELECT COUNT(*) FROM sentences"),
            "chunks": q("SELECT COUNT(*) FROM chunks"),
            "chunks_failed_or_pending": q("SELECT COUNT(*) FROM chunks WHERE status != 'done'"),
            "candidates (before merging)": q("SELECT COUNT(*) FROM qa_candidates"),
            "pairs (after merging)": q("SELECT COUNT(*) FROM qa_pairs"),
            "pairs merged from 2+ candidates": q("SELECT COUNT(*) FROM qa_pairs WHERE merged_from LIKE '%,%'"),
            "pairs citing 2+ sections": q("SELECT COUNT(*) FROM qa_pairs WHERE sections LIKE '%\", \"%'"),
            "pairs added by coverage pass": q("SELECT COALESCE(SUM(n_coverage_added), 0) FROM chunks"),
            "pairs dropped by validation": q("SELECT COALESCE(SUM(n_dropped), 0) FROM chunks"),
            "pairs embedded": q("SELECT COUNT(*) FROM qa_pairs WHERE status = 'embedded'"),
            "coverage (content sentences cited)": f"{cited}/{content} = {100 * cited / content:.1f}%"
                                                  if content else "n/a",
        }

    # ---------------------------------------------------------- run log (P4b)
    def start_run(self, command: str, settings: dict, detail: Optional[dict] = None) -> int:
        """Open a run record; returns its run_id."""
        cur = self.conn.execute(
            "INSERT INTO runs(command, started_at, status, settings, detail) VALUES(?,?,?,?,?)",
            (command, _now(), "running", json.dumps(settings, default=str),
             json.dumps(detail or {}, ensure_ascii=False, default=str)))
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, usage: dict, total, duration_s: float,
                   detail: Optional[dict] = None) -> None:
        """Close a run record with its usage. `total` is an llm.Usage."""
        fields = ["finished_at = ?", "duration_s = ?", "status = ?", "usage = ?", "calls = ?",
                  "prompt_tokens = ?", "completion_tokens = ?", "cost_usd = ?", "cost_known = ?"]
        values = [_now(), round(duration_s, 2), status, json.dumps(usage), total.calls,
                  total.prompt_tokens, total.completion_tokens, round(total.cost_usd, 6),
                  1 if total.calls_without_cost == 0 else 0]
        if detail is not None:
            fields.append("detail = ?")
            values.append(json.dumps(detail, ensure_ascii=False, default=str))
        self.conn.execute(f"UPDATE runs SET {', '.join(fields)} WHERE run_id = ?", [*values, run_id])
        self.conn.commit()

    def save_run_document(self, run_id: int, doc_id: str, title: str, action: str, chunks: int,
                          candidates: int, pairs: int, coverage: str, usage) -> None:
        """One document's outcome and usage within a run (`usage` is an llm.Usage)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO run_documents(run_id, doc_id, title, action, chunks, candidates, pairs, "
            "coverage, calls, prompt_tokens, completion_tokens, cost_usd) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, doc_id, title, action, chunks, candidates, pairs, coverage, usage.calls,
             usage.prompt_tokens, usage.completion_tokens, round(usage.cost_usd, 6)))
        self.conn.commit()

    def list_runs(self, command: Optional[str] = None, limit: int = 50) -> list[sqlite3.Row]:
        if command:
            return self.conn.execute("SELECT * FROM runs WHERE command = ? ORDER BY run_id DESC LIMIT ?",
                                     (command, limit)).fetchall()
        return self.conn.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()

    def get_run(self, run_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def run_documents(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM run_documents WHERE run_id = ? ORDER BY rowid",
                                 (run_id,)).fetchall()

    def document_rows(self) -> list[sqlite3.Row]:
        """Documents with pair counts, for the report."""
        return self.conn.execute(
            "SELECT d.*, (SELECT COUNT(*) FROM qa_pairs p WHERE p.doc_id = d.doc_id) AS n_pairs, "
            "(SELECT COUNT(*) FROM qa_pairs p WHERE p.doc_id = d.doc_id AND p.merged_from LIKE '%,%') "
            "AS n_merged, "
            "(SELECT COALESCE(SUM(n_dropped),0) FROM chunks c WHERE c.doc_id = d.doc_id) AS n_dropped, "
            "(SELECT COALESCE(SUM(n_coverage_added),0) FROM chunks c WHERE c.doc_id = d.doc_id) AS n_cov_added, "
            "(SELECT COALESCE(SUM(n_calls),0) FROM chunks c WHERE c.doc_id = d.doc_id) AS n_gen_calls "
            "FROM documents d ORDER BY d.created_at").fetchall()
