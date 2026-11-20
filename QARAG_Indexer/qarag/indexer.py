"""
indexer.py - runs the whole Stage 0 pipeline for a list of sources.

    for each source (URL or file):
        load text -> numbered sentences with sections -> chunks
        -> GENERATE candidate pairs per chunk (parallel), incl. coverage pass
        -> CONSOLIDATE: merge repeated facts across the whole document
        -> final pairs saved in SQLite
    then:
        embed every final pair not yet embedded -> store vectors in Chroma

Document status moves:  generating -> consolidating -> complete

Two modes:

  mode="overwrite"  Start from scratch: recreate SQLite tables and delete the
                    Chroma collection, then index the given sources.

  mode="append"     Add the given sources to the existing database. For each
                    source the indexer decides:
                      * new source                        -> index it
                      * already indexed, text unchanged   -> skip (no cost)
                      * already indexed, text CHANGED     -> delete its old pairs
                                                             and vectors, re-index
                      * interrupted earlier (incomplete)  -> resume: only the
                                                             unfinished work is done
                      * --force given                     -> re-index anyway

A database is built with ONE embedding model, embedding variant and answer
style; append runs must use the same ones (checked before any money is spent).

Run log (P4b): every run is recorded in the `runs` table (settings, duration,
model calls, tokens, cost) and every document of the run in `run_documents`.
Per-document usage covers generation, coverage pass, merging and the question
embeddings used for merging; the final pair embeddings are made in one batch
for all documents and appear in the run total (purpose "embed_pairs").

Metadata (P4a): an unchanged document that is skipped still gets its metadata
refreshed (no model cost), which also fills metadata for databases created
before P4.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from qarag import __version__, llm
from qarag.chunker import attach_context, build_chunks, split_sentences
from qarag.config import Settings
from qarag.consolidate import consolidate_document
from qarag.generator import generate_for_chunk
from qarag.models import Chunk, Sentence
from qarag.prompts import PROMPT_VERSION
from qarag.sources import load_document
from qarag.store import QAStore
from qarag.vector_store import VectorIndex, document_filter_fields, texts_for_pair


@dataclass
class SourceReport:
    """What happened to one source, for the summary printed at the end."""
    source: str
    action: str = ""            # indexed / skipped / resumed / reindexed / failed
    title: str = ""
    chunks: int = 0
    candidates: int = 0
    pairs: int = 0
    coverage: str = ""
    chunk_errors: list[str] = field(default_factory=list)
    error: Optional[str] = None
    doc_id: str = ""
    usage: llm.Usage = field(default_factory=llm.Usage)   # calls/tokens/cost for this document


def doc_id_for(source: str) -> str:
    """Stable id from the source location: the same URL/path always gets the same id."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def content_hash_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings_snapshot(settings: Settings) -> dict:
    """The settings that matter for comparing runs, saved with every run."""
    return {
        "qarag_version": __version__, "prompt_version": PROMPT_VERSION,
        "gen_model": settings.effective_gen_model(), "merge_model": settings.effective_merge_model()
        if not settings.dry_run else "mock-generator",
        "embed_model": settings.effective_embed_model(), "embed_variant": settings.embed_variant,
        "answer_style": settings.answer_style, "chunk_tokens": settings.chunk_tokens,
        "overlap_tokens": settings.overlap_tokens, "lookahead_tokens": settings.lookahead_tokens,
        "coverage_pass": settings.coverage_pass, "merge_enabled": settings.merge_enabled,
        "merge_threshold": settings.merge_threshold, "max_workers": settings.max_workers,
        "gen_temperature": settings.gen_temperature, "dry_run": settings.dry_run,
    }


class Indexer:
    def __init__(self, settings: Settings, log: Callable[[str], None] = print):
        settings.validate()
        self.settings = settings
        self.log = log                       # where progress messages go (console by default)
        self.store = QAStore(settings.sqlite_path)
        if self.store.upgraded:
            self.log("Database upgraded to schema 3 (document metadata + run log); existing pairs kept.")
        self.index = VectorIndex(settings.chroma_path, settings.collection_name)
        self.last_run_id: Optional[int] = None
        self.last_usage = llm.Usage()

    def close(self) -> None:
        self.store.close()

    # ------------------------------------------------------------- settings
    def _locked_settings(self) -> dict[str, str]:
        """Settings that must stay the same for the whole database."""
        return {"embed_model": self.settings.effective_embed_model(),
                "embed_variant": self.settings.embed_variant,
                "answer_style": self.settings.answer_style}

    def _check_and_lock_settings(self) -> None:
        for key, wanted in self._locked_settings().items():
            stored = self.store.get_meta(key)
            if stored and stored != wanted:
                raise ValueError(
                    f"This database was built with {key} = '{stored}', but this run uses '{wanted}'. "
                    "A database holds one setting so results stay comparable: use --mode overwrite, "
                    "or a separate database folder with --db (e.g. --db data/style_short).")
        for key, wanted in self._locked_settings().items():
            self.store.set_meta(key, wanted)

    # =================================================================== main
    def run(self, locations: list[str], mode: str = "append", force: bool = False) -> list[SourceReport]:
        if mode not in ("append", "overwrite"):
            raise ValueError("mode must be 'append' or 'overwrite'")
        if mode == "overwrite":
            self.log("Overwrite mode: clearing the existing Q&A database and vector index.")
            self.store.reset()
            self.index.reset()
        else:
            self.store.require_current()
        self._check_and_lock_settings()      # before spending money on generation
        self.index.open(self.settings.effective_embed_model(), self.settings.embed_variant)

        reports: list[SourceReport] = []

        def work():
            for number, location in enumerate(locations, start=1):
                self.log(f"\n[{number}/{len(locations)}] {location}")
                report = self._index_one(location, force)
                reports.append(report)
                if report.doc_id:
                    self.store.save_run_document(self.last_run_id, report.doc_id, report.title,
                                                 report.action, report.chunks, report.candidates,
                                                 report.pairs, report.coverage, report.usage)
            self._embed_pending()
            return {"mode": mode, "force": force,
                    "sources": [{"source": r.source, "action": r.action, "title": r.title,
                                 "error": r.error} for r in reports]}

        self._logged_run("index", work, detail={"mode": mode, "sources": locations})
        return reports

    # ============================================================== run log
    def _logged_run(self, command: str, work: Callable[[], object], detail: Optional[dict] = None):
        """Run `work()` while recording usage; save the run in the runs table
        (also when it fails, with status 'failed', so cost is never lost)."""
        llm.METER.reset()
        started = time.time()
        self.last_run_id = self.store.start_run(command, settings_snapshot(self.settings), detail)
        status, final_detail = "failed", None
        try:
            result = work()
            status = "finished"
            final_detail = result if isinstance(result, dict) else None
            return result
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            self.last_usage = llm.METER.total()
            self.store.finish_run(self.last_run_id, status, llm.METER.snapshot(), self.last_usage,
                                  time.time() - started, detail=final_detail)

    # ============================================================ one source
    def _index_one(self, location: str, force: bool) -> SourceReport:
        report = SourceReport(source=location)
        try:
            document = load_document(location, self.settings.user_agent)
        except Exception as error:  # noqa: BLE001 - report and continue with the next source
            first_line = str(error).splitlines()[0] if str(error) else ""
            report.action, report.error = "failed", f"could not load: {type(error).__name__}: {first_line}"
            self.log(f"  ! {report.error}")
            return report

        report.title = document.title
        doc_id = doc_id_for(document.source)
        report.doc_id = doc_id
        with llm.METER.scope(doc_id):        # attribute this document's model calls to it
            self._index_loaded(document, doc_id, force, report)
        report.usage = llm.METER.total(doc_id)
        return report

    def _refresh_metadata(self, doc_id: str, document) -> None:
        """Update metadata of an already indexed document (no model calls) and
        copy the filter fields onto its vectors."""
        if self.store.update_metadata(doc_id, document.metadata, document.title):
            m = document.metadata
            year = m.get("published") or ""
            fields = document_filter_fields(m.get("doc_type"), m.get("language"),
                                            int(year[:4]) if year[:4].isdigit() else None)
            self.index.update_document_fields(doc_id, fields)
            self.log("  metadata refreshed")

    def _index_loaded(self, document, doc_id: str, force: bool, report: SourceReport) -> SourceReport:
        location = report.source
        content_hash = content_hash_for(document.full_text)
        existing = self.store.get_document(document.source)

        # ---- decide what to do with this source
        if existing is not None:
            unchanged = existing["content_hash"] == content_hash
            if unchanged and not force and existing["status"] == "complete":
                report.action = "skipped"
                self._fill_report(report, doc_id)
                self.log(f"  = already indexed and unchanged - skipped ({existing['title']})")
                self._refresh_metadata(doc_id, document)
                return report
            if unchanged and not force:
                self._refresh_metadata(doc_id, document)
                report.action = "resumed"
                self.log(f"  > incomplete from an earlier run ({existing['status']}) - resuming")
                sentences = self.store.load_sentences(doc_id)
                cores = [[s for s in sentences if first <= s.idx <= last]
                         for _, first, last in self.store.chunk_bounds(doc_id)]
                chunks = attach_context(cores, self.settings.overlap_tokens,
                                        self.settings.lookahead_tokens, self.settings.chars_per_token)
                return self._process(doc_id, document.title, chunks, report)
            reason = "forced re-index" if unchanged else "document changed since last time"
            self.log(f"  ~ {reason} - replacing its old Q&A pairs")
            self.store.delete_document(doc_id)
            self.index.delete_document(doc_id)
            report.action = "reindexed"
        else:
            report.action = "indexed"

        # ---- new (or replaced) document: sentences -> chunks -> database
        sentences = split_sentences(document)
        if not sentences:
            report.action, report.error = "failed", "no usable sentences after text extraction"
            self.log(f"  ! {report.error}")
            return report
        chunks = build_chunks(sentences, self.settings.chunk_tokens, self.settings.overlap_tokens,
                              self.settings.lookahead_tokens, self.settings.chars_per_token)
        self.store.create_document(doc_id, document, content_hash, sentences, chunks,
                                   self.settings.effective_gen_model(), PROMPT_VERSION,
                                   self.settings.answer_style)
        n_sections = len({s.section for s in sentences if s.section})
        self.log(f"  title: {document.title}")
        m = document.metadata
        info = [x for x in [m.get("doc_type"), m.get("language"), m.get("published"),
                            ", ".join(m.get("authors") or [])[:60], m.get("doi")] if x]
        if info:
            self.log(f"  metadata: {' | '.join(info)}")
        self.log(f"  {len(sentences)} sentences in {n_sections} section(s) -> {len(chunks)} chunks")
        return self._process(doc_id, document.title, chunks, report)

    def _process(self, doc_id: str, title: str, chunks: list[Chunk], report: SourceReport) -> SourceReport:
        """Generation for pending chunks, then consolidation when all are done."""
        report.chunks = len(chunks)
        self._generate(doc_id, title, chunks, report)
        if self.store.pending_chunk_indexes(doc_id):
            self.store.set_document_status(doc_id, "generating")
            self.log("  ! some chunks failed - run again with --mode append to retry them "
                     "(the document is consolidated once every chunk is done)")
        else:
            self._consolidate(doc_id, title)
        self._fill_report(report, doc_id)
        return report

    def _fill_report(self, report: SourceReport, doc_id: str) -> None:
        row = self.store.conn.execute("SELECT n_candidates, n_content_sentences, n_cited_sentences, n_chunks "
                                      "FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        report.pairs = self.store.count_pairs(doc_id)
        if row:
            report.chunks = row["n_chunks"] or report.chunks
            report.candidates = row["n_candidates"] or 0
            if row["n_content_sentences"]:
                report.coverage = f"{100 * row['n_cited_sentences'] / row['n_content_sentences']:.0f}%"

    # ========================================================= generation
    def _generate(self, doc_id: str, title: str, chunks: list[Chunk], report: SourceReport) -> None:
        """Generate candidates for every chunk still pending, several at a time."""
        pending = self.store.pending_chunk_indexes(doc_id)
        todo = [c for c in chunks if c.chunk_index in pending]
        if not todo:
            return
        done = 0
        # Threads only CALL THE MODEL; every database write happens here in the
        # main thread as results arrive, which keeps SQLite simple and safe.
        with ThreadPoolExecutor(max_workers=max(1, self.settings.max_workers)) as pool:
            futures = {pool.submit(generate_for_chunk, chunk, title, len(chunks), self.settings): chunk
                       for chunk in todo}
            for future in as_completed(futures):
                r = future.result()
                done += 1
                if r.error:
                    self.store.record_chunk_error(doc_id, r.chunk_index, r.error)
                    report.chunk_errors.append(f"chunk {r.chunk_index}: {r.error}")
                    self.log(f"  ! chunk {r.chunk_index + 1}/{len(chunks)} failed: {r.error}")
                    continue
                self.store.save_chunk_candidates(doc_id, r.chunk_index, r.pairs, r.dropped,
                                                 r.coverage_added, r.calls)
                extra = []
                if r.coverage_added:
                    extra.append(f"{r.coverage_added} from coverage pass")
                if r.dropped:
                    extra.append(f"{r.dropped} dropped")
                cov = f"{r.n_cited}/{r.n_content} sentences cited" if r.n_content else "no content sentences"
                note = f" ({', '.join(extra)})" if extra else ""
                self.log(f"  chunk {r.chunk_index + 1}/{len(chunks)}: {len(r.pairs)} candidates{note}, "
                         f"{cov}   [{done}/{len(todo)}]")

    # ====================================================== consolidation
    def _consolidate(self, doc_id: str, title: str) -> None:
        """Merge repeated facts across the document and save the final pairs.
        Any vectors from an earlier consolidation of this document are removed
        first, because the final pairs are rebuilt from scratch."""
        self.store.set_document_status(doc_id, "consolidating")
        self.index.delete_document(doc_id)
        report = consolidate_document(self.store, doc_id, title, self.settings, log=self.log)
        self.store.set_document_status(doc_id, "complete")
        failures = f", {report.merge_failures} merge call(s) failed (kept unmerged)" if report.merge_failures else ""
        self.log(f"  consolidated: {report.candidates} candidates -> {report.final_pairs} pairs "
                 f"({report.merged_away} merged away){failures}")

    def reconsolidate_all(self) -> int:
        """Re-run consolidation for every finished document from the stored
        candidates (e.g. with a new --threshold). No new generation."""
        self.store.require_current()
        self._check_and_lock_settings()
        self.index.open(self.settings.effective_embed_model(), self.settings.embed_variant)
        documents = self.store.complete_documents()

        def work():
            for row in documents:
                self.log(f"\n{row['title']}")
                with llm.METER.scope(row["doc_id"]):
                    self._consolidate(row["doc_id"], row["title"])
            self._embed_pending()
            return {"documents": len(documents)}

        self._logged_run("consolidate", work)
        return len(documents)

    # ========================================================== embedding
    def _embed_pending(self) -> int:
        """Embed every final pair whose status is still 'generated'. Runs after
        all sources, so the embedding API is called in large efficient batches."""
        records = self.store.pairs_with_status("generated")
        if not records:
            self.log("\nNo new pairs to embed.")
            return 0
        variant = self.settings.embed_variant
        self.log(f"\nEmbedding {len(records)} pairs (variant '{variant}', "
                 f"model '{self.settings.effective_embed_model()}') ...")
        # Slices of pairs: progress is saved as we go, so an interruption never
        # causes the same pairs to be embedded (and paid for) twice.
        slice_size = 250
        done = 0
        for start in range(0, len(records), slice_size):
            part = records[start:start + slice_size]
            # One entry per Chroma record: (pair, suffix, text). "dual" gives two per pair.
            entries = [(r, suffix, text) for r in part for suffix, text in texts_for_pair(r, variant)]
            texts = [text for _, _, text in entries]
            if self.settings.dry_run:
                vectors = llm.mock_embed(texts, purpose="embed_pairs")
            else:
                vectors = llm.embed_texts(texts, self.settings.embed_model,
                                          self.settings.openrouter_base_url,
                                          self.settings.openrouter_api_key,
                                          batch_size=self.settings.embed_batch_size,
                                          purpose="embed_pairs")
            self.index.add(part, entries, vectors)
            self.store.mark_embedded(r.pair_id for r in part)
            done += len(part)
            self.log(f"  {done}/{len(records)} pairs embedded")
        self.log(f"Vector index now holds {self.index.count()} vectors.")
        return len(records)

    def reembed(self) -> int:
        """Rebuild only the vectors (e.g. after changing the embedding model or
        variant) from the final pairs already in SQLite - no generation cost."""
        self.store.require_current()
        self.index.reset()
        self.index.open(self.settings.effective_embed_model(), self.settings.embed_variant)
        self.store.set_meta("embed_model", self.settings.effective_embed_model())
        self.store.set_meta("embed_variant", self.settings.embed_variant)
        self.store.reset_embedding_status()
        result = self._logged_run("reembed", lambda: {"pairs": self._embed_pending()})
        return result["pairs"]
