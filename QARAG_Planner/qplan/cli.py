"""
cli.py - the command line of qplan.

Commands (run from F:\\qplan with the venv active):

    python -m qplan info   --db F:\\qarag\\data\\wiki_test
    python -m qplan ask    "Were Scott Derrickson and Ed Wood of the same nationality?" --db ...
    python -m qplan batch  tests\\questions.txt --db ...
    python -m qplan report [--last 20] [--csv runs\\report.csv]
    python -m qplan show   last            (or a run id)
    python -m qplan export --out runs\\runs.jsonl

Every option that changes a run (models, max steps, top-k, threshold,
memory policy) can also be set in .env, so the command lines stay short.
Command-line options win over .env.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from qplan import __version__
from qplan.config import Settings, openrouter_api_key


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _add_run_options(p: argparse.ArgumentParser) -> None:
    """Options shared by `ask` and `batch`."""
    p.add_argument("--db", help="database folder, e.g. F:\\qarag\\data\\wiki_test (or QPLAN_DB in .env)")
    p.add_argument("--dry-run", action="store_true", help="fake LLM calls (offline, free)")
    p.add_argument("-q", "--quiet", action="store_true", help="do not print every step")
    p.add_argument("--max-steps", type=int, help="maximum ReAct steps (default 6)")
    p.add_argument("-k", "--top-k", type=int, help="pairs returned per search (default 5)")
    p.add_argument("--min-score", type=float, help="score needed to count as a match (default 0.35)")
    p.add_argument("--memory", choices=["on", "off", "free", "fallback"],
                   help="memory model: on = may be used when the KB lacks a fact (default), "
                        "off = knowledge base only, free = any time (fallback = on)")
    p.add_argument("--direct-first", choices=["on", "off"],
                   help="step 0: search the whole question and answer directly; ReAct only if that is "
                        "not enough (default on; off = ReAct only, as in 0.1.4)")
    p.add_argument("--direct-k", type=int, help="pairs returned by the direct-first search (default 8)")
    p.add_argument("--react-model", help="model of the ReAct agent")
    p.add_argument("--memory-model", help="model used by recall")
    p.add_argument("--synth-model", help="model of LLM-S")
    p.add_argument("--model", help="set all three model slots at once")
    p.add_argument("--runs-dir", help="where the run log and traces go (default .\\runs)")


def _settings_from_args(args) -> Settings:
    """Start from .env/defaults and apply the command-line options."""
    s = Settings()
    if getattr(args, "db", None):
        s.db = args.db
    if getattr(args, "runs_dir", None):
        s.runs_dir = args.runs_dir
    if getattr(args, "dry_run", False):
        s.dry_run = True
    if getattr(args, "quiet", False):
        s.verbose = False
    for opt, attr in (("max_steps", "max_steps"), ("top_k", "top_k"), ("min_score", "min_score"),
                      ("memory", "memory_policy"), ("direct_first", "direct_first"),
                      ("direct_k", "direct_top_k")):
        value = getattr(args, opt, None)
        if value is not None:
            setattr(s, attr, value)
    if getattr(args, "model", None):
        s.react_model = s.memory_model = s.synth_model = args.model
    for attr in ("react_model", "memory_model", "synth_model"):
        value = getattr(args, attr, None)
        if value:
            setattr(s, attr, value)
    s.validate()
    return s


def _open_db(s: Settings):
    """Open the database read-only and attach its embedding model."""
    from qplan.db import QADatabase
    if not s.db:
        sys.exit("No database given: use --db <folder> or set QPLAN_DB in .env")
    db = QADatabase(s.db)
    for w in db.warnings:
        print(f"WARNING: {w}", file=sys.stderr)       # stderr: keeps --json output clean
    # The key is optional only when the DB itself is a mock (offline) one.
    db.attach_embedder(openrouter_api_key(required=db.embed_model != "mock-embedding"))
    return db


def _engine(s: Settings):
    from qplan.fakes import fake_responder
    from qplan.react import ReActEngine
    from qplan.runlog import RunLog
    db = _open_db(s)
    log = RunLog(s.runs_dir)
    engine = ReActEngine(s, db, log, fake_responder=fake_responder if s.dry_run else None)
    return engine, db, log


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_info(args) -> None:
    s = _settings_from_args(args)
    from qplan.db import QADatabase
    if not s.db:
        sys.exit("No database given: use --db <folder> or set QPLAN_DB in .env")
    db = QADatabase(s.db)
    info = db.info()
    print(f"Database   : {info['folder']}")
    print(f"Fingerprint: {info['fingerprint']}")
    for k in ("schema_version", "embed_model", "embed_variant", "answer_style"):
        print(f"{k:<15}: {info['meta'].get(k)}")
    print(f"Documents  : {len(info['documents'])} ({info['n_documents_complete']} complete)")
    print(f"Q&A pairs  : {info['n_pairs']}   vectors in Chroma: {info['n_vectors']}")
    print()
    print(f"{'status':<14}{'pairs':>6}  title")
    for d in info["documents"]:
        print(f"{d['status'] or '-':<14}{d['n_pairs']:>6}  {d['title']}")
    for w in info["warnings"]:
        print(f"WARNING: {w}")
    db.close()


def cmd_ask(args) -> None:
    s = _settings_from_args(args)
    if args.json:
        s.verbose = False                  # stdout carries only the JSON object
    engine, db, log = _engine(s)

    # ---- optional document filter: search only these documents ------------
    titles = list(args.only_titles or [])
    if args.only_titles_file:
        titles += [line.strip() for line in Path(args.only_titles_file).read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    where, filter_info = None, None
    if titles:
        doc_ids, missing = db.doc_ids_for_titles(titles)
        # No known title -> a filter that matches nothing (never the whole database).
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else {"doc_id": "__no_document__"}
        filter_info = {"titles_requested": len(dict.fromkeys(titles)), "documents_found": len(doc_ids),
                       "titles_not_in_database": missing}
        if missing and not args.json:
            print(f"WARNING: {len(missing)} title(s) not in the database: {'; '.join(missing)}", file=sys.stderr)

    result = engine.run(args.question, where=where)
    if args.json:
        print(json.dumps(_result_json(result, s, db, filter_info), ensure_ascii=True))
    else:
        _print_result(result)
    db.close()
    log.close()


def _result_json(result, s: Settings, db, filter_info) -> dict:
    """
    Everything a calling program needs about one answer, as plain JSON.
    (No evaluation here: qplan reports what it answered and what it used.)
    """
    from qplan import __version__ as version, prompts
    f = result.final
    cited = set(f.cited) if f else set()
    evidence = []
    for e in result.evidence:
        detail = e.get("detail") or {}
        evidence.append({
            "label": e["ev_id"], "kind": e["kind"], "cited": e["ev_id"] in cited,
            "title": e.get("title"), "pair_id": e.get("pair_id"), "score": e.get("score"),
            "question": e.get("question"), "answer": e.get("answer"),
            "sentence_ids": [x.get("idx") for x in detail.get("sentences", [])] if e["kind"] == "kb" else [],
            "source": e.get("source"),
        })
    steps = result.steps
    return {
        "question": result.question, "run_id": result.run_id,
        "answer": f.answer if f else "", "short_answer": f.short_answer if f else "",
        "status": f.status if f else "error", "basis": f.basis if f else "",
        "cited": f.cited if f else [], "invalid_citations": f.invalid_citations if f else [],
        "evidence": evidence,
        "steps": len(steps),
        "searches": sum(1 for x in steps if x["outcome"] in ("match", "no_match")),
        "recalls": sum(1 for x in steps if x["outcome"].startswith("recall")),
        "stop_reason": result.stop_reason, "cost_usd": round(result.cost_usd or 0.0, 6),
        # direct-first (0.1.5): escalated = None (not used) | False (answered at step 0) | True (ReAct ran)
        "escalated": result.escalated, "direct_status": result.direct_status,
        "error": result.error, "filter": filter_info, "trace": result.trace_path,
        "settings": {"qplan": version, "database": str(db.dir), "memory": s.memory_policy,
                     "top_k": s.top_k, "min_score": s.min_score, "max_steps": s.max_steps,
                     "direct_first": s.direct_first, "direct_top_k": s.direct_top_k,
                     "react_model": s.react_model, "memory_model": s.memory_model, "synth_model": s.synth_model,
                     "reasoning": {"react": s.react_reasoning, "memory": s.memory_reasoning,
                                   "synth": s.synth_reasoning},
                     "prompts": [prompts.REACT_PROMPT_VERSION, prompts.MEMORY_PROMPT_VERSION,
                                 prompts.SYNTH_PROMPT_VERSION],
                     "embed_variant": db.meta.get("embed_variant"), "answer_style": db.meta.get("answer_style")},
    }


def _print_result(result) -> None:
    print("\n" + "=" * 70)
    if result.final:
        f = result.final
        print(f"ANSWER: {f.answer}")
        print(f"short answer: {f.short_answer}   status: {f.status}   basis: {f.basis}")
        if f.invalid_citations:
            print(f"WARNING: invalid citations (not in the evidence): {', '.join(f.invalid_citations)}")
        if not f.parse_ok:
            print("WARNING: LLM-S did not return JSON; its raw text is shown as the answer.")
        print()
        print(f.sources_text)
    if result.error:
        print(f"ERROR: {result.error}")
    print("=" * 70)
    if result.escalated is not None:
        print("direct-first: " + ("escalated to ReAct" if result.escalated else "answered at step 0, no ReAct"))
    print(f"run {result.run_id} | stop: {result.stop_reason} | steps: {len(result.steps)} | "
          f"cost: ${result.cost_usd:.5f}")
    print(f"trace: {result.trace_path}")


def _read_questions(path: Path) -> list[tuple[str, str]]:
    """
    One question per line. Optional expected answer after ' || '.
    Empty lines and lines starting with # are skipped.
    """
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q, _, expected = line.partition("||")
        items.append((q.strip(), expected.strip()))
    return items


def cmd_batch(args) -> None:
    s = _settings_from_args(args)
    questions = _read_questions(Path(args.file))
    if not questions:
        sys.exit(f"No questions in {args.file}")
    engine, db, log = _engine(s)
    out_path = Path(args.out) if args.out else Path(s.runs_dir) / f"batch-{datetime.now():%Y%m%d-%H%M%S}.csv"
    rows = []
    for i, (q, expected) in enumerate(questions, 1):
        print(f"\n##### question {i}/{len(questions)}")
        r = engine.run(q)
        if s.verbose:
            _print_result(r)
        else:
            print(f"{q}\n  -> {r.final.short_answer if r.final else 'ERROR'}  (${r.cost_usd:.5f})")
        f = r.final
        rows.append({
            "run_id": r.run_id, "question": q, "expected": expected,
            "short_answer": f.short_answer if f else "", "answer_status": f.status if f else "error",
            "basis": f.basis if f else "", "stop_reason": r.stop_reason, "n_steps": len(r.steps),
            "n_searches": sum(1 for x in r.steps if x["outcome"] in ("match", "no_match")),
            "n_recalls": sum(1 for x in r.steps if x["outcome"].startswith("recall")),
            "cited": " ".join(f.cited) if f else "", "invalid_cites": " ".join(f.invalid_citations) if f else "",
            "cost_usd": round(r.cost_usd, 6), "answer": f.answer if f else r.error,
            "trace": r.trace_path,
        })
    # utf-8-sig so that Excel opens the file with the right characters.
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["cost_usd"] for r in rows)
    print(f"\n{len(rows)} questions done, total cost ${total:.5f}\nresults: {out_path}")
    db.close()
    log.close()


def cmd_report(args) -> None:
    from qplan.runlog import RunLog
    s = _settings_from_args(args)
    log = RunLog(s.runs_dir)
    runs = log.list_runs(args.last)
    if not runs:
        print("No runs logged yet.")
        return
    print(f"{'run_id':<22}{'steps':>5}{'srch':>5}{'rcl':>4} {'status':<10}{'basis':<7}"
          f"{'tokens':>8}{'cost $':>10}  question -> short answer")
    for r in runs:
        tokens = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        cost = f"{r['cost_usd']:.5f}" if r["cost_usd"] is not None else "?"
        if r["cost_complete"] == 0:
            cost += "*"
        flag = " [dry]" if r["dry_run"] else ""
        q = (r["question"] or "")[:60]
        print(f"{r['run_id']:<22}{r['n_steps'] or 0:>5}{r['n_searches'] or 0:>5}{r['n_recalls'] or 0:>4} "
              f"{(r['answer_status'] or r['status'] or '')[:9]:<10}{(r['answer_basis'] or '')[:6]:<7}"
              f"{tokens:>8}{cost:>10}  {q}{flag} -> {r['short_answer'] or ''}")
    total = sum(r["cost_usd"] or 0 for r in runs)
    print(f"\n{len(runs)} runs, total ${total:.5f}   (* = some calls did not report a cost)")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
            w.writeheader()
            w.writerows(runs)
        print(f"written: {args.csv}")
    log.close()


def cmd_show(args) -> None:
    from qplan.react import render_trace
    from qplan.runlog import RunLog
    s = _settings_from_args(args)
    log = RunLog(s.runs_dir)
    run = log.get_run(args.run_id)
    if not run:
        sys.exit(f"run {args.run_id!r} not found")
    print(render_trace(run))
    log.close()


def cmd_export(args) -> None:
    """All runs, complete (steps, calls with prompts, evidence), as JSON Lines."""
    from qplan.runlog import RunLog
    s = _settings_from_args(args)
    log = RunLog(s.runs_dir)
    out = Path(args.out)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for run_id in log.all_run_ids():
            run = log.get_run(run_id)
            if args.no_dry and run["dry_run"]:
                continue
            fh.write(json.dumps(run, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} runs written to {out}")
    log.close()


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    # Windows consoles may not support every character: never crash on print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="qplan", description=f"qplan {__version__} - ReAct question answering over a qarag database")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="check and summarise a database")
    p.add_argument("--db")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    _add_run_options(p)
    p.add_argument("--only-titles", nargs="+", metavar="TITLE",
                   help="search only the documents with these exact titles")
    p.add_argument("--only-titles-file", help="text file: one document title per line (same as --only-titles)")
    p.add_argument("--json", action="store_true",
                   help="print the result as one JSON object (answer, status, evidence, cost) and nothing else")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("batch", help="answer every question in a text file, write a CSV")
    p.add_argument("file", help="text file: one question per line, optional ' || expected answer'")
    p.add_argument("--out", help="CSV file to write (default runs\\batch-<time>.csv)")
    _add_run_options(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("report", help="list recent runs with steps, tokens and cost")
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--csv", help="also write the list to this CSV file")
    p.add_argument("--runs-dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("show", help="print the full trace of a run")
    p.add_argument("run_id", help="a run id, or 'last'")
    p.add_argument("--runs-dir")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("export", help="write all runs as JSON Lines (for analysis/tuning)")
    p.add_argument("--out", required=True)
    p.add_argument("--no-dry", action="store_true", help="leave out dry-run runs")
    p.add_argument("--runs-dir")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        # Expected problems (missing key, wrong folder, version mismatch):
        # a clear message instead of a long traceback.
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
