"""
report.py - one readable Markdown report of a database and its runs (P4b).

    python -m qarag report                    -> report.md (latest index run in detail)
    python -m qarag report --run 3 --out run3.md --sample 20

Sections of the report
  1. Database        what is in it now, and the settings locked for it
  2. Runs            every index / consolidate / reembed / ask run: calls, tokens, cost
  3. Run details     one run (default: the latest `index` run): its settings,
                     usage per purpose (generate, coverage, merge, embeddings ...)
                     and per document, with cost per pair
  4. Documents       metadata (P4a) and quality numbers per document
  5. Questions       the latest `ask` runs and the average cost of one question
                     (the retrieval + answering baseline for the ablation study)
  6. Sample pairs    a reproducible random sample of pairs with their evidence,
                     for a quick manual quality check

All numbers come from SQLite (tables runs, run_documents, documents, qa_pairs ...),
so the report can be regenerated at any time without model calls.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Optional

from qarag.llm import Usage, money
from qarag.store import QAStore

# Order in which purposes are listed; unknown purposes are appended.
PURPOSE_ORDER = ["generate", "coverage", "embed_questions", "merge", "embed_pairs",
                 "embed_query", "answer"]
PURPOSE_HELP = {
    "generate": "Q&A generation, one call per chunk (+ JSON repair retries)",
    "coverage": "coverage pass for sentences no pair cited",
    "embed_questions": "question vectors used to find merge candidates",
    "merge": "merge decisions for groups of similar questions",
    "embed_pairs": "final pair vectors stored in Chroma",
    "embed_query": "the user's question (ask)",
    "answer": "answer synthesis (ask)",
}


def _usage(d: dict) -> Usage:
    return Usage(**{k: d.get(k, 0) for k in Usage.__dataclass_fields__})


def _cost(value: Optional[float], known: Optional[int] = 1) -> str:
    """known = 1 if every call had a cost; 0 if some had none (then the sum is partial)."""
    if value is None or (not known and not value):
        return "n/a"
    return money(value) + ("" if known else " (partial)")


def _n(value) -> str:
    return f"{int(value or 0):,}"


def _table(header: list[str], rows: list[list]) -> list[str]:
    """A Markdown table. Pipes inside cells are escaped."""
    def cell(x) -> str:
        return str("" if x is None else x).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]
    return out + [""]


def _when(text: Optional[str]) -> str:
    return (text or "")[:19].replace("T", " ")


def build_report(store: QAStore, run_id: Optional[int] = None, sample: int = 10,
                 seed: int = 42, db_label: str = "") -> str:
    lines: list[str] = [f"# qarag report{' - ' + db_label if db_label else ''}", "",
                        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.", ""]

    # ------------------------------------------------------------ 1. database
    lines += ["## 1. Database", ""]
    lines += _table(["measure", "value"], [[k, v] for k, v in store.stats().items()])
    lines += _table(["locked setting", "value"],
                    [[k, store.get_meta(k)] for k in ("answer_style", "embed_model", "embed_variant",
                                                      "schema_version")])

    # ---------------------------------------------------------------- 2. runs
    runs = store.list_runs(limit=30)
    lines += ["## 2. Runs", ""]
    if not runs:
        lines += ["No runs recorded yet (runs are logged from qarag 0.4 on).", ""]
    else:
        rows = []
        for r in runs:
            detail = json.loads(r["detail"] or "{}")
            what = (detail.get("question", "")[:50] if r["command"] == "ask"
                    else f"{len(detail.get('sources', []))} source(s), {detail.get('mode', '')}".strip(", ")
                    if r["command"] == "index" else "")
            rows.append([r["run_id"], r["command"], _when(r["started_at"]), f"{r['duration_s'] or 0:.0f}s",
                         r["status"], what, _n(r["calls"]), _n(r["prompt_tokens"]),
                         _n(r["completion_tokens"]), _cost(r["cost_usd"], r["cost_known"])])
        lines += _table(["run", "command", "started (UTC)", "time", "status", "what", "calls",
                         "tokens in", "tokens out", "cost"], rows)
        lines += ["Cost is the amount OpenRouter reported, or an estimate from the QARAG_PRICE_* "
                  "settings in .env; \"n/a\" means neither was available (tokens are always counted).", ""]

    # --------------------------------------------------------- 3. run details
    selected = store.get_run(run_id) if run_id else next((r for r in runs if r["command"] == "index"), None)
    if run_id and selected is None:
        lines += [f"## 3. Run details", "", f"Run {run_id} not found.", ""]
    elif selected is not None:
        lines += _run_details(store, selected)

    # ----------------------------------------------------------- 4. documents
    lines += ["## 4. Documents", ""]
    docs = store.document_rows()
    rows = []
    for d in docs:
        authors = json.loads(d["authors"]) if d["authors"] else []
        cov = (f"{100 * d['n_cited_sentences'] / d['n_content_sentences']:.0f}%"
               if d["n_content_sentences"] else "-")
        rows.append([d["title"], d["doc_type"] or "-", d["language"] or "-", d["published"] or "-",
                     (", ".join(authors)[:40] or "-"), _n(d["n_sentences"]), _n(d["n_chunks"]),
                     _n(d["n_candidates"]), _n(d["n_pairs"]), _n(d["n_merged"]), _n(d["n_cov_added"]),
                     _n(d["n_dropped"]), cov])
    lines += _table(["document", "type", "lang", "published", "authors", "sentences", "chunks",
                     "candidates", "pairs", "merged pairs", "added by coverage", "dropped", "coverage"], rows)
    meta_rows = []
    for d in docs:
        extra = [f"{k}: {d[k]}" for k in ("doi", "publisher", "file_name", "modified") if d[k]]
        meta_rows.append([d["title"], "; ".join(extra) or "-", (d["description"] or "-")[:120], d["source"]])
    if meta_rows:
        lines += _table(["document", "more metadata", "description", "source"], meta_rows)

    # ----------------------------------------------------------- 5. questions
    asks = store.list_runs(command="ask", limit=20)
    lines += ["## 5. Questions asked", ""]
    if not asks:
        lines += ["No `ask` runs recorded yet.", ""]
    else:
        rows, total = [], Usage()
        for r in asks:
            d = json.loads(r["detail"] or "{}")
            u = Usage(calls=r["calls"] or 0, prompt_tokens=r["prompt_tokens"] or 0,
                      completion_tokens=r["completion_tokens"] or 0, cost_usd=r["cost_usd"] or 0.0)
            total.add(u)
            rows.append([r["run_id"], d.get("question", "")[:70], d.get("k", ""),
                         f"{len(d.get('cited', []))}/{len(d.get('evidence', []))}",
                         "yes" if d.get("not_found") else "", _n(u.prompt_tokens), _n(u.completion_tokens),
                         _cost(r["cost_usd"], r["cost_known"])])
        lines += _table(["run", "question", "k", "cited/evidence", "not found", "tokens in",
                         "tokens out", "cost"], rows)
        n = len(asks)
        lines += [f"Average per question (last {n}): {total.prompt_tokens / n:,.0f} tokens in, "
                  f"{total.completion_tokens / n:,.0f} tokens out, ${total.cost_usd / n:.5f}.", ""]

    # -------------------------------------------------------- 6. sample pairs
    pairs = store.all_pairs()
    lines += ["## 6. Sample pairs", ""]
    if pairs and sample > 0:
        rng = random.Random(seed)          # fixed seed: the same sample every time
        chosen = rng.sample(pairs, min(sample, len(pairs)))
        lines += [f"{len(chosen)} of {len(pairs)} pairs, random sample (seed {seed}).", ""]
        cache: dict[str, dict] = {}
        for p in chosen:
            if p.doc_id not in cache:
                cache[p.doc_id] = {s.idx: s for s in store.load_sentences(p.doc_id)}
            lines += [f"**{p.pair_id}** - {p.citation()}" + (f" - {' | '.join(p.sections)}" if p.sections else ""),
                      "", f"- Q: {p.question}", f"- A: {p.answer}"]
            if len(p.merged_from) > 1:
                lines.append(f"- merged from {len(p.merged_from)} candidates")
            for i in p.sentence_ids:
                s = cache[p.doc_id].get(i)
                if s:
                    lines.append(f"- [S{i}] {s.text[:300]}")
            lines.append("")
    else:
        lines += ["No pairs.", ""]
    return "\n".join(lines)


def _run_details(store: QAStore, run) -> list[str]:
    lines = [f"## 3. Run {run['run_id']} details ({run['command']}, {_when(run['started_at'])})", ""]
    settings = json.loads(run["settings"] or "{}")
    lines += _table(["setting", "value"], [[k, v] for k, v in settings.items()])

    # usage per purpose (summed over documents)
    usage = json.loads(run["usage"] or "{}")
    by_purpose: dict[str, Usage] = {}
    for scope in usage.values():
        for purpose, d in scope.items():
            by_purpose.setdefault(purpose, Usage()).add(_usage(d))
    total = Usage()
    for u in by_purpose.values():
        total.add(u)
    order = [p for p in PURPOSE_ORDER if p in by_purpose] + sorted(set(by_purpose) - set(PURPOSE_ORDER))
    rows = []
    for p in order:
        u = by_purpose[p]
        share = f"{100 * u.total_tokens / total.total_tokens:.0f}%" if total.total_tokens else "-"
        rows.append([p, PURPOSE_HELP.get(p, ""), _n(u.calls), _n(u.prompt_tokens), _n(u.completion_tokens),
                     share, u.cost_text()])
    rows.append(["**total**", "", _n(total.calls), _n(total.prompt_tokens), _n(total.completion_tokens),
                 "100%", total.cost_text()])
    lines += ["Usage by purpose:", ""]
    lines += _table(["purpose", "what", "calls", "tokens in", "tokens out", "share of tokens", "cost"], rows)

    docs = store.run_documents(run["run_id"])
    if docs:
        rows = []
        for d in docs:
            per_pair = (f"${d['cost_usd'] / d['pairs']:.5f}" if d["pairs"] and d["cost_usd"] else "-")
            tokens_per_pair = (f"{(d['prompt_tokens'] + d['completion_tokens']) / d['pairs']:,.0f}"
                               if d["pairs"] and d["calls"] else "-")
            rows.append([d["title"], d["action"], _n(d["chunks"]), _n(d["candidates"]), _n(d["pairs"]),
                         d["coverage"] or "-", _n(d["calls"]), _n(d["prompt_tokens"]),
                         _n(d["completion_tokens"]), _cost(d["cost_usd"], run["cost_known"]), per_pair,
                         tokens_per_pair])
        lines += ["Per document (generation, coverage, merging; pair embeddings are in the total above):", ""]
        lines += _table(["document", "action", "chunks", "candidates", "pairs", "coverage", "calls",
                         "tokens in", "tokens out", "cost", "cost/pair", "tokens/pair"], rows)
    return lines
