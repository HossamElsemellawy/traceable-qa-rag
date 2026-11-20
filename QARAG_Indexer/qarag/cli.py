"""
cli.py - command-line interface.

Run from the project folder (F:\\qarag) with the virtual environment active:

    python -m qarag index  <URL-or-path> [more ...]  [--mode append|overwrite]
    python -m qarag stats                       counts, merging and coverage
    python -m qarag docs [--meta]               list indexed documents (with metadata)
    python -m qarag ask "your question"         answer from the documents, with citations
    python -m qarag ask "..." --type "wikipedia article" --since 2020    filtered by metadata
    python -m qarag report [--run N]            Markdown report: runs, tokens, cost, documents, samples
    python -m qarag search "your question"      retrieval check: the closest Q&A pairs
    python -m qarag gaps                        content sentences no pair covers
    python -m qarag export pairs.jsonl          all final pairs for review
    python -m qarag consolidate --threshold 0.85   re-run merging only (no generation)
    python -m qarag reembed --variant dual      rebuild vectors only (no generation)

Add --help to any command for all its options, e.g.  python -m qarag index --help
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import typer

from qarag import llm
from qarag.config import Settings
from qarag.indexer import Indexer
from qarag.sources import expand_locations
from qarag.store import QAStore
from qarag.vector_store import VectorIndex, build_filter

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False,
                  help="Stage 0 Q&A indexer for the Governed RAG framework.")


def _fail(error: Exception) -> None:
    """Print a known problem (bad setting, missing file, mismatched index) as
    one readable line instead of a long Python traceback, then stop."""
    typer.echo(f"\nError: {error}", err=True)
    raise typer.Exit(code=1)


def _settings(db_dir: Optional[Path], dry_run: bool = False, **overrides) -> Settings:
    """Start from config.py defaults (+ .env), then apply command-line overrides
    that were actually given (None means "not given, keep the default")."""
    settings = Settings()
    if db_dir is not None:
        settings.db_dir = db_dir
    settings.dry_run = dry_run
    for name, value in overrides.items():
        if value is not None:
            setattr(settings, name, value)
    return settings


def _open_store(db_dir: Optional[Path]) -> QAStore:
    """Open the database for reading commands, with a clear message if it is
    from an older version of the code."""
    store = QAStore(_settings(db_dir).sqlite_path)
    try:
        store.require_current()
    except ValueError as error:
        store.close()
        _fail(error)
    return store


def _run_indexer(settings: Settings, action):
    """Create the Indexer, run `action(indexer)`, always close the database,
    and turn known problems into one readable error line."""
    try:
        indexer = Indexer(settings)
    except (ValueError, OSError) as error:
        _fail(error)
    try:
        return action(indexer)
    except (ValueError, OSError) as error:
        _fail(error)
    finally:
        indexer.close()


# ---------------------------------------------------------------------------
@app.command()
def index(
    sources: List[str] = typer.Argument(None, help="URLs, file paths or folders to index."),
    from_list: Optional[Path] = typer.Option(None, "--from-list", help="Text file with one URL or path per line."),
    mode: str = typer.Option("append", "--mode", "-m",
                             help="'append' adds to the existing database; 'overwrite' starts from scratch."),
    force: bool = typer.Option(False, "--force", help="Re-index sources even if they are unchanged."),
    answer_style: Optional[str] = typer.Option(None, "--answer-style", "-s",
                                               help="short | sentence (default) | detailed"),
    variant: Optional[str] = typer.Option(None, "--variant", help="What to embed: qa (default), q, or dual."),
    merge_threshold: Optional[float] = typer.Option(None, "--threshold",
                                                    help="Question similarity that triggers a merge check (default 0.82)."),
    no_merge: bool = typer.Option(False, "--no-merge", help="Skip merging of repeated facts."),
    no_coverage: bool = typer.Option(False, "--no-coverage", help="Skip the coverage pass."),
    gen_model: Optional[str] = typer.Option(None, "--gen-model", help="Generator model, e.g. openrouter/qwen/qwen3-30b-a3b"),
    embed_model: Optional[str] = typer.Option(None, "--embed-model", help="Embedding model on OpenRouter."),
    chunk_tokens: Optional[int] = typer.Option(None, "--chunk-tokens", help="Approximate tokens per chunk."),
    overlap_tokens: Optional[int] = typer.Option(None, "--overlap-tokens", help="Context tokens shown from the previous chunk."),
    lookahead_tokens: Optional[int] = typer.Option(None, "--lookahead-tokens", help="Context tokens shown from the next chunk (0 = off)."),
    workers: Optional[int] = typer.Option(None, "--workers", help="Model calls in parallel."),
    db_dir: Optional[Path] = typer.Option(None, "--db", help="Database folder (default data/index)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Test the pipeline offline with fake model calls (no cost)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation in overwrite mode."),
):
    """Convert articles into Q&A pairs and add them to the vector index."""
    locations = expand_locations(sources or [], str(from_list) if from_list else None)
    if not locations:
        typer.echo("Nothing to index: give at least one URL, file, folder or --from-list file.")
        raise typer.Exit(code=1)
    if mode == "overwrite" and not yes:
        typer.confirm("Overwrite mode deletes ALL existing Q&A pairs and vectors. Continue?", abort=True)

    settings = _settings(db_dir, dry_run, answer_style=answer_style, embed_variant=variant,
                         merge_threshold=merge_threshold, gen_model=gen_model, embed_model=embed_model,
                         chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
                         lookahead_tokens=lookahead_tokens, max_workers=workers)
    if no_merge:
        settings.merge_enabled = False
    if no_coverage:
        settings.coverage_pass = False
    typer.echo(f"Sources: {len(locations)}   mode: {mode}   style: {settings.answer_style}   "
               f"generator: {settings.effective_gen_model()}   database: {settings.db_dir}")

    usage = {}

    def action(ix: Indexer):
        result = ix.run(locations, mode=mode, force=force)
        usage["run"], usage["total"] = ix.last_run_id, ix.last_usage
        return result

    reports = _run_indexer(settings, action)

    typer.echo("\nSummary")
    typer.echo("-" * 88)
    typer.echo(f"{'action':<10} {'candidates':>10} {'pairs':>6} {'coverage':>9} {'chunks':>7}  title")
    for r in reports:
        typer.echo(f"{r.action:<10} {r.candidates:>10} {r.pairs:>6} {r.coverage:>9} {r.chunks:>7}  "
                   f"{(r.title or r.source)[:45]}")
        if r.error:
            typer.echo(f"{'':<10} error: {r.error}")
        for message in r.chunk_errors[:3]:
            typer.echo(f"{'':<10} {message[:100]}")
    typer.echo("-" * 88)
    _print_usage(usage.get("total"), usage.get("run"))


def _print_usage(total, run_id) -> None:
    """One line with the run's model usage (P4b)."""
    if total is None:
        return
    typer.echo(f"Run {run_id}: {total.calls} model calls, {total.prompt_tokens:,} tokens in, "
               f"{total.completion_tokens:,} tokens out, cost {total.cost_text()}"
               f"   (details: python -m qarag report --run {run_id})")


# ---------------------------------------------------------------------------
@app.command()
def consolidate(
    merge_threshold: Optional[float] = typer.Option(None, "--threshold", help="Question similarity for merge checks."),
    db_dir: Optional[Path] = typer.Option(None, "--db"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Re-run the merging of repeated facts from the stored candidates (no new generation)."""
    store = _open_store(db_dir)
    style = store.get_meta("answer_style")
    store.close()
    settings = _settings(db_dir, dry_run, merge_threshold=merge_threshold, answer_style=style)
    n = _run_indexer(settings, lambda ix: ix.reconsolidate_all())
    typer.echo(f"\nRe-consolidated {n} document(s).")


@app.command()
def reembed(
    variant: Optional[str] = typer.Option(None, "--variant", help="qa, q or dual"),
    embed_model: Optional[str] = typer.Option(None, "--embed-model"),
    db_dir: Optional[Path] = typer.Option(None, "--db"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Rebuild the vector index from the existing pairs (new model or variant, no regeneration)."""
    store = _open_store(db_dir)
    style = store.get_meta("answer_style")
    store.close()
    settings = _settings(db_dir, dry_run, embed_variant=variant, embed_model=embed_model, answer_style=style)
    n = _run_indexer(settings, lambda ix: ix.reembed())
    typer.echo(f"Re-embedded {n} pairs.")


# ---------------------------------------------------------------------------
@app.command()
def stats(db_dir: Optional[Path] = typer.Option(None, "--db")):
    """Show counts, merging and coverage for the database."""
    store = _open_store(db_dir)
    for key, value in store.stats().items():
        typer.echo(f"{key:<38} {value}")
    for key in ("answer_style", "embed_model", "embed_variant"):
        typer.echo(f"{key:<38} {store.get_meta(key)}")
    store.close()


@app.command()
def docs(meta: bool = typer.Option(False, "--meta", help="Also show each document's metadata."),
         db_dir: Optional[Path] = typer.Option(None, "--db")):
    """List the indexed documents."""
    store = _open_store(db_dir)
    for row in store.list_documents():
        cov = (f"{100 * row['n_cited_sentences'] / row['n_content_sentences']:.0f}%"
               if row["n_content_sentences"] else "-")
        pending = f"  ({row['pending']} chunks pending)" if row["pending"] else ""
        typer.echo(f"{row['doc_id']}  {row['status']:<13} {row['n_candidates'] or 0:>4} cand -> "
                   f"{row['n_pairs']:>4} pairs  coverage {cov:>4}  {(row['title'] or '')[:40]}{pending}")
        typer.echo(f"{'':14}{row['source']}")
        if meta:
            authors = json.loads(row["authors"]) if row["authors"] else []
            fields = [("type", row["doc_type"]), ("language", row["language"]),
                      ("published", row["published"]), ("modified", row["modified"]),
                      ("authors", ", ".join(authors)), ("publisher", row["publisher"]),
                      ("doi", row["doi"]), ("file", row["file_name"])]
            shown = [f"{k}: {v}" for k, v in fields if v]
            typer.echo(f"{'':14}" + (" | ".join(shown) if shown else "(no metadata - run index again to fill it)"))
            if row["description"]:
                typer.echo(f"{'':14}{row['description'][:150]}")
    store.close()


# ---------------------------------------------------------------------------
@app.command()
def search(
    query: str = typer.Argument(..., help="A question to search for."),
    k: int = typer.Option(5, "-k", help="Number of Q&A pairs to return."),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show the supporting sentences."),
    doc_type: Optional[str] = typer.Option(None, "--type", help='Only documents of this type, e.g. "wikipedia article".'),
    language: Optional[str] = typer.Option(None, "--lang", help="Only documents in this language, e.g. en."),
    since: Optional[int] = typer.Option(None, "--since", help="Only documents published in or after this year."),
    until: Optional[int] = typer.Option(None, "--until", help="Only documents published in or before this year."),
    db_dir: Optional[Path] = typer.Option(None, "--db"),
):
    """Retrieval check: find the Q&A pairs closest to a question, with their citations."""
    settings = _settings(db_dir)
    store = _open_store(db_dir)
    model, variant = store.get_meta("embed_model"), store.get_meta("embed_variant")
    if not model:
        typer.echo("The database is empty - run `python -m qarag index ...` first.")
        raise typer.Exit(code=1)

    # The query MUST be embedded with the same model the index was built with.
    if model == "mock-embedding":
        vector = llm.mock_embed([query])[0]
    else:
        try:
            settings.validate()
        except ValueError as error:
            _fail(error)
        vector = llm.embed_texts([query], model, settings.openrouter_base_url, settings.openrouter_api_key)[0]

    index = VectorIndex(settings.chroma_path, settings.collection_name)
    try:
        index.open(model, variant)
    except ValueError as error:
        _fail(error)
    hits = index.search(vector, k=k, where=build_filter(doc_type, language, since, until))
    records = store.get_pairs([h.pair_id for h in hits])
    if not hits:
        typer.echo("No pairs found (check the filters: documents without that metadata are excluded).")

    for rank, hit in enumerate(hits, start=1):
        rec = records.get(hit.pair_id)
        if rec is None:
            continue
        merged = f", merged from {len(rec.merged_from)}" if len(rec.merged_from) > 1 else ""
        typer.echo(f"\n#{rank}  score {hit.score:.3f}  (matched {hit.matched}{merged})  [{rec.citation()}]")
        typer.echo(f"   Q: {rec.question}")
        typer.echo(f"   A: {rec.answer}")
        citation = []
        if rec.sections:
            citation.append("sections: " + " | ".join(rec.sections))
        if rec.source_pages:
            citation.append("pages: " + ", ".join(map(str, rec.source_pages)))
        if citation:
            typer.echo(f"   Cited from {'; '.join(citation)}")
        if show_sources:
            sentences = {s.idx: s for s in store.load_sentences(rec.doc_id)}
            for i in rec.sentence_ids:
                if i in sentences:
                    s = sentences[i]
                    where = " · ".join(x for x in [s.section or "", f"p.{s.page}" if s.page else ""] if x)
                    typer.echo(f"   [S{i}{' · ' + where if where else ''}] {s.text[:150]}")
    store.close()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    k: Optional[int] = typer.Option(None, "-k", help="Q&A pairs retrieved as evidence (default 8)."),
    show_evidence: bool = typer.Option(False, "--show-evidence", "-e",
                                       help="Also print the retrieved pairs given to the model."),
    answer_model: Optional[str] = typer.Option(None, "--answer-model", help="Model that writes the answer."),
    doc_type: Optional[str] = typer.Option(None, "--type", help='Only documents of this type, e.g. "wikipedia article".'),
    language: Optional[str] = typer.Option(None, "--lang", help="Only documents in this language, e.g. en."),
    since: Optional[int] = typer.Option(None, "--since", help="Only documents published in or after this year."),
    until: Optional[int] = typer.Option(None, "--until", help="Only documents published in or before this year."),
    db_dir: Optional[Path] = typer.Option(None, "--db"),
):
    """Answer a question from the indexed documents (retrieval + grounded answer with citations).
    Every question is logged as an `ask` run with its tokens and cost (see `report`)."""
    from qarag.answer import answer_question

    settings = _settings(db_dir, answer_model=answer_model)
    store = _open_store(db_dir)
    k = k or settings.answer_top_k
    filters = {"type": doc_type, "lang": language, "since": since, "until": until}
    where = build_filter(doc_type, language, since, until)

    # Log the question as a run (P4b): retrieval + answer tokens and cost.
    llm.METER.reset()
    started = time.time()
    run_id = store.start_run("ask", {"answer_model": settings.effective_answer_model(),
                                     "embed_model": store.get_meta("embed_model"),
                                     "embed_variant": store.get_meta("embed_variant"), "k": k},
                             {"question": question, "k": k, "filters": {a: b for a, b in filters.items() if b}})
    try:
        result = answer_question(question, store, settings, k=k, where=where)
    except ValueError as error:
        store.finish_run(run_id, "failed", llm.METER.snapshot(), llm.METER.total(), time.time() - started)
        store.close()
        _fail(error)
    total = llm.METER.total()
    store.finish_run(run_id, "finished", llm.METER.snapshot(), total, time.time() - started, detail={
        "question": question, "k": k, "filters": {a: b for a, b in filters.items() if b},
        "answer": result.text, "cited": result.cited,
        "evidence": [item.record.pair_id for item in result.evidence],
        "not_found": "do not contain this information" in result.text})

    typer.echo(f"\nQ: {result.question}\n")
    typer.echo(result.text)

    # Short source list: only the evidence the answer actually cites.
    if result.cited:
        typer.echo("\nSources:")
        for item in result.evidence:
            if item.number in result.cited:
                rec = item.record
                where = " | ".join(rec.sections) if rec.sections else "no section"
                pages = f", p. {', '.join(map(str, rec.source_pages))}" if rec.source_pages else ""
                typer.echo(f"  [{item.number}] {rec.citation()} - {where}{pages}")

    if show_evidence:
        typer.echo("\nEvidence given to the model:")
        for item in result.evidence:
            mark = "*" if item.number in result.cited else " "
            typer.echo(f" {mark}[{item.number}] score {item.score:.3f}  Q: {item.record.question}")
            typer.echo(f"        A: {item.record.answer}")
    typer.echo(f"\n(run {run_id}: {total.calls} calls, {total.prompt_tokens:,} tokens in, "
               f"{total.completion_tokens:,} tokens out, cost {total.cost_text()})")
    store.close()


@app.command()
def gaps(
    limit: int = typer.Option(20, "--limit", help="Sentences to show per document."),
    db_dir: Optional[Path] = typer.Option(None, "--db"),
):
    """Show content sentences that no Q&A pair cites (coverage gaps), per document."""
    store = _open_store(db_dir)
    min_words = Settings().coverage_min_words
    for row in store.list_documents():
        missing = store.uncited_sentences(row["doc_id"], min_words)
        typer.echo(f"\n{row['title']}: {len(missing)} uncited content sentence(s)")
        for s in missing[:limit]:
            typer.echo(f"   [S{s.idx}{' · ' + s.section if s.section else ''}] {s.text[:150]}")
    store.close()


@app.command()
def export(
    output: Path = typer.Argument(Path("pairs.jsonl"),
                                  help="Output file: .csv (opens in Excel, for review) or .jsonl (for code)."),
    db_dir: Optional[Path] = typer.Option(None, "--db"),
):
    """Export all final Q&A pairs with their evidence sentences and citations."""
    import csv

    store = _open_store(db_dir)
    records = store.all_pairs()
    sentence_cache: dict[str, dict] = {}

    def evidence(rec) -> str:
        """The exact source sentences behind a pair, each with its section/page."""
        if rec.doc_id not in sentence_cache:
            sentence_cache[rec.doc_id] = {s.idx: s for s in store.load_sentences(rec.doc_id)}
        parts = []
        for i in rec.sentence_ids:
            s = sentence_cache[rec.doc_id].get(i)
            if s:
                where = " · ".join(x for x in [s.section or "", f"p.{s.page}" if s.page else ""] if x)
                parts.append(f"[S{i}{' | ' + where if where else ''}] {s.text}")
        return "\n".join(parts)

    if output.suffix.lower() == ".csv":
        # utf-8-sig = UTF-8 with a BOM, so Excel shows accented characters correctly.
        with output.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "document", "doc_type", "published", "authors", "question",
                             "answer", "sections", "pages", "merged_from_n", "evidence",
                             "your_rating", "your_comment"])
            for r in records:
                writer.writerow([r.pair_id, r.title, r.doc_type or "", r.published or "",
                                 "; ".join(r.authors), r.question, r.answer, " | ".join(r.sections),
                                 ", ".join(map(str, r.source_pages)), len(r.merged_from),
                                 evidence(r), "", ""])
    else:
        with output.open("w", encoding="utf-8") as f:
            for r in records:
                row = dict(r.__dict__)
                row["evidence"] = evidence(r)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"Wrote {len(records)} pairs to {output}")
    store.close()


# ---------------------------------------------------------------------------
@app.command()
def report(
    output: Path = typer.Option(Path("report.md"), "--out", "-o", help="Markdown file to write."),
    run: Optional[int] = typer.Option(None, "--run", help="Run to show in detail (default: latest index run)."),
    sample: int = typer.Option(10, "--sample", help="Random pairs to include for a quality check (0 = none)."),
    seed: int = typer.Option(42, "--seed", help="Seed of the random sample (same seed = same pairs)."),
    db_dir: Optional[Path] = typer.Option(None, "--db"),
):
    """Write a Markdown report: runs with tokens and cost, per-document quality and metadata, sample pairs."""
    from qarag.report import build_report

    store = _open_store(db_dir)
    text = build_report(store, run_id=run, sample=sample, seed=seed, db_label=str(_settings(db_dir).db_dir))
    output.write_text(text, encoding="utf-8")
    runs = store.list_runs(limit=5)
    store.close()
    typer.echo(f"Wrote {output} ({len(text.splitlines())} lines).")
    if runs:
        typer.echo("Latest runs:")
        for r in runs:
            cost = "n/a" if not r["cost_known"] and not r["cost_usd"] else llm.money(r["cost_usd"] or 0)
            typer.echo(f"  run {r['run_id']:>3}  {r['command']:<11} {(r['started_at'] or '')[:16]}  "
                       f"{r['status']:<9} {r['calls'] or 0:>5} calls  "
                       f"{(r['prompt_tokens'] or 0) + (r['completion_tokens'] or 0):>9,} tokens  {cost}")


if __name__ == "__main__":
    app()
