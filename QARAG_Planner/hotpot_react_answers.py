"""
hotpot_react_answers.py - collect the qplan ReAct planner's answers to the
HotpotQA subset, in the CSV format that hotpot_ask_eval.py --rescore scores.

Division of work
  qplan (a separate program, its own folder and venv) answers ONE question at a
        time: `python -m qplan ask "<question>" --only-titles-file <titles>
        --json`. It knows nothing about HotpotQA, gold answers or scores.
  this script reads the subset, calls qplan once per question with the
        question's own 10 paragraph titles (the standard distractor setting),
        and saves the answers (+ supporting-fact counts) to a CSV.
  hotpot_ask_eval.py --rescore <csv> does ALL the evaluation, unchanged:
        short-answer extraction, LLM judge, EM, F1, Cover, Not found, Partial,
        SF recall, summary. So the planner is scored exactly like base100_v3.

Put this file next to hotpot_ask_eval.py (the qarag folder) and run it from
there (any Python 3.10+; it needs no packages, qplan runs in its own venv):

    python hotpot_react_answers.py --qplan F:\\QARAG_Planner\\qplan --n 5 --out react5.csv
    python hotpot_ask_eval.py --rescore react5.csv --out react5_scored.csv

    python hotpot_react_answers.py --qplan F:\\QARAG_Planner\\qplan --out react100.csv
    python hotpot_react_answers.py --qplan F:\\QARAG_Planner\\qplan --out react100.csv --resume
    python hotpot_ask_eval.py --rescore react100.csv --out react100_scored.csv

    (--score runs the second command automatically after the answers.)

    qplan 0.1.5 options passed through (left out = qplan's .env decides):
    --direct-first on|off   one search of the whole question first, ReAct only if needed
    --direct-k 8            pairs for that search
    --react-model / --synth-model   e.g. openrouter/qwen/qwen3-235b-a22b-2507
    Extra columns: escalated (0 = answered at step 0, 1 = ReAct ran), direct_status,
    kept_direct (qplan 0.1.7: 1 = the ReAct answer was weaker, the step-0 answer was kept).

Columns written for hotpot_ask_eval.py (its --rescore reads these)
  n, id, type, question, answer, gold, sf_retrieved, sf_cited, evidence
  answer is qplan's answer put in qarag's conventions, so the reference
  scoring treats both systems alike:
    * citation labels [E7] -> [7] (qarag numbers its evidence [1]..[k]);
      model-memory labels [M1] are removed (memory is off by default anyway);
    * qplan status not_found -> the answer is qarag's refusal sentence
      "The indexed documents do not contain this information." (= Not found);
    * qplan status partial   -> that sentence is appended to the answer
      (= Partial, as qarag's model writes it).
  qplan's original answer and run data are kept in extra columns (qplan_*),
  which hotpot_ask_eval.py ignores.
  sf_retrieved / sf_cited: gold supporting sentences (supporting_facts_qarag)
  inside the knowledge-base pairs qplan received over all its searches / that
  its answer cites; the same rule as hotpot_ask_eval.py's sf_recall.

Settings of the planner run come from qplan's .env, except:
  --memory  default off (HotpotQA answers are facts a model may know)
  -k, --min-score, --max-steps   passed to qplan if given
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REFUSAL = "The indexed documents do not contain this information."
REFERENCE_FIELDS = ["n", "id", "type", "question", "answer", "gold", "sf_retrieved", "sf_cited", "evidence"]
EXTRA_FIELDS = ["qplan_answer", "qplan_status", "qplan_basis", "qplan_short", "qplan_cited", "steps",
                "searches", "recalls", "stop_reason", "escalated", "direct_status", "kept_direct", "attempts", "cost_usd", "run_id",
                "titles_not_in_database", "error"]


# ---------------------------------------------------------------------------
# qplan's answer -> qarag's conventions
# ---------------------------------------------------------------------------
def to_qarag_labels(text: str) -> str:
    """[E7] -> [7], [E1, E3] -> [1][3]; [M1] (model memory) removed."""
    def group(m):
        labels = re.split(r"\s*[,;]\s*", m.group(1))
        return "".join(f"[{x[1:]}]" for x in labels if x.startswith("E"))
    text = re.sub(r"\[((?:[EM]\d+)(?:\s*[,;]\s*[EM]\d+)*)\]", group, text or "")
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", text).strip()


def answer_for_reference(result: dict) -> str:
    status, answer = result.get("status", ""), to_qarag_labels(result.get("answer", ""))
    if status == "not_found":
        return REFUSAL
    if status == "partial" and REFUSAL.lower() not in answer.lower():
        return f"{answer} {REFUSAL}".strip()
    return answer


def sf_recall(evidence: list[dict], gold_facts: list, cited_only: bool) -> tuple[int, int]:
    """Gold supporting sentences found in qplan's knowledge-base evidence (rule of hotpot_ask_eval.py)."""
    got = defaultdict(set)
    for e in evidence:
        if e.get("kind") != "kb" or (cited_only and not e.get("cited")):
            continue
        got[e.get("title") or ""] |= set(e.get("sentence_ids") or [])
    facts = [(t, ids) for t, ids in gold_facts if ids]
    return sum(any(i in got[t] for i in ids) for t, ids in facts), len(facts)


# ---------------------------------------------------------------------------
# calling qplan
# ---------------------------------------------------------------------------
def qplan_python(folder: Path, override: str | None) -> str:
    if override:
        return override
    for p in (folder / ".venv" / "Scripts" / "python.exe", folder / ".venv" / "bin" / "python"):
        if p.is_file():
            return str(p)
    return sys.executable                            # fall back to the current Python


def ask_qplan(args, python: str, question: str, titles: list[str]) -> tuple[dict | None, str]:
    """One `qplan ask ... --json` call. Returns (result or None, error text)."""
    # --setting distractor: search only the question's own 10 paragraphs (titles filter).
    # --setting full: no filter, qplan searches the WHOLE database (all questions' paragraphs).
    titles_file = None
    cmd = [python, "-m", "qplan", "ask", question, "--db", str(Path(args.db).resolve()),
           "--json", "--memory", args.memory]
    if args.setting == "distractor":
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("\n".join(titles))
            titles_file = fh.name
        cmd += ["--only-titles-file", titles_file]
    if args.top_k is not None:
        cmd += ["-k", str(args.top_k)]
    if args.min_score is not None:
        cmd += ["--min-score", str(args.min_score)]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.dry_run:
        cmd += ["--dry-run"]
    # qplan 0.1.5 options (left out = qplan's .env decides)
    for opt, value in (("--direct-first", args.direct_first), ("--direct-k", args.direct_k),
                       ("--react-model", args.react_model), ("--synth-model", args.synth_model)):
        if value is not None:
            cmd += [opt, str(value)]
    try:
        proc = subprocess.run(cmd, cwd=args.qplan, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return None, f"qplan did not answer within {args.timeout} s"
    finally:
        if titles_file:                              # no temp file in the full setting
            os.unlink(titles_file)
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or f"qplan exit code {proc.returncode}").strip()[-500:]
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1]), ""
    except (json.JSONDecodeError, IndexError):
        return None, f"qplan output was not JSON: {proc.stdout[-300:]!r} {proc.stderr[-300:]!r}"


# ---------------------------------------------------------------------------
# Re-asking after temporary API errors (rate limits)
# ---------------------------------------------------------------------------
# Run react30_V7: four questions failed with
#   429 "qwen/qwen3-30b-a3b is temporarily rate-limited upstream" (DeepInfra, overloaded).
# qplan itself is NOT changed: this script waits and asks the whole question
# again. Only temporary errors are retried; a wrong model id (400), a bad key
# (401) or no credit (402) are reported at once.
_TRANSIENT = ("429", "502", "503", "504", "rate-limited", "rate limit", "ratelimit", "overloaded",
              "temporarily", "timed out", "timeout", "did not answer within", "serviceunavailable",
              "apiconnectionerror", "connection reset", "internalservererror")
_PERMANENT = ('"code":400', '"code":401', '"code":402', "not a valid model id", "authenticationerror")


def transient(error_text: str) -> bool:
    low = (error_text or "").lower()
    return bool(low) and not any(p in low for p in _PERMANENT) and any(s in low for s in _TRANSIENT)


def ask_with_retry(args, python: str, question: str, titles: list[str]) -> tuple[dict | None, str, int]:
    """ask_qplan, re-asked after a temporary API error. Returns (result, error, attempts)."""
    waits = [int(w) for w in str(args.retry_waits).split(",") if w.strip()]
    for attempt in range(len(waits) + 1):
        result, error = ask_qplan(args, python, question, titles)
        text = (result or {}).get("error") or error
        if not transient(text) or attempt == len(waits):
            return result, error, attempt + 1
        print(f"     temporary API error ({text.splitlines()[0][:90]}...); "
              f"asking again in {waits[attempt]} s ({attempt + 1}/{len(waits)})")
        time.sleep(waits[attempt])
    return result, error, len(waits) + 1


# ---------------------------------------------------------------------------
def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REFERENCE_FIELDS + EXTRA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Collect qplan ReAct answers for hotpot_ask_eval.py --rescore.")
    ap.add_argument("--qplan", required=True, help=r"qplan folder, e.g. F:\QARAG_Planner\qplan")
    ap.add_argument("--qplan-python", help=r"Python of qplan's venv (default <qplan>\.venv\Scripts\python.exe)")
    ap.add_argument("--db", default=str(Path("data") / "hotpot"), help=r"database folder (default data\hotpot)")
    ap.add_argument("--subset", default="hotpot_subset.jsonl")
    ap.add_argument("--setting", choices=["distractor", "full"], default="distractor",
                    help="distractor = only the question's own 10 paragraphs (default); "
                         "full = search the whole database, no titles filter")
    ap.add_argument("--n", type=int, default=0, help="only the first N questions (0 = all)")
    ap.add_argument("--memory", choices=["on", "off", "free"], default="off",
                    help="qplan's memory model (default off for evaluation)")
    ap.add_argument("-k", "--top-k", type=int, help="pairs per search (default: qplan .env)")
    ap.add_argument("--min-score", type=float, help="qplan match threshold (default: qplan .env)")
    ap.add_argument("--max-steps", type=int, help="ReAct step limit (default: qplan .env)")
    ap.add_argument("--direct-first", choices=["on", "off"],
                    help="qplan 0.1.5+: answer from one search of the whole question first, ReAct only if "
                         "needed (default: qplan .env, on)")
    ap.add_argument("--direct-k", type=int, help="qplan 0.1.5+: pairs for the direct-first search (default 8)")
    ap.add_argument("--react-model", help="qplan's agent model (default: qplan .env)")
    ap.add_argument("--synth-model", help="qplan's LLM-S model (default: qplan .env)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds allowed per question")
    ap.add_argument("--retry-waits", default="30,60,120,240",
                    help="seconds to wait before re-asking a question that failed with a temporary "
                         "API error (rate limit, overload, timeout); '' = no re-asking")
    ap.add_argument("--resume", action="store_true", help="keep answered questions in --out, answer the rest")
    ap.add_argument("--dry-run", action="store_true", help="qplan with fake models (offline test)")
    ap.add_argument("--score", action="store_true",
                    help="afterwards run: hotpot_ask_eval.py --rescore <out> --out <out>_scored.csv")
    ap.add_argument("--out", default="hotpot_react_answers.csv")
    args = ap.parse_args(argv)

    args.qplan = str(Path(args.qplan).resolve())
    python = qplan_python(Path(args.qplan), args.qplan_python)
    subset = [json.loads(l) for l in Path(args.subset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        subset = subset[:args.n]

    done = {}
    if args.resume and Path(args.out).exists():
        with open(args.out, encoding="utf-8-sig", newline="") as f:
            done = {r["id"]: r for r in csv.DictReader(f) if not r.get("error")}
        print(f"Resuming: {len(done)} answer(s) kept from {args.out}")

    print(f"Asking qplan ({python}) {len(subset)} questions, {args.setting} setting, memory {args.memory} ...")
    started, rows, settings, missing_total = time.time(), [], None, 0
    for n, q in enumerate(subset, start=1):
        if q["id"] in done:
            rows.append({**done[q["id"]], "n": n})
            continue
        result, error, attempts = ask_with_retry(args, python, q["question"], q["titles"])
        row = {"n": n, "id": q["id"], "type": q["type"], "question": q["question"], "gold": q["answer"],
               "attempts": attempts}
        if result is None:
            row.update({"answer": "", "sf_retrieved": "0/0", "sf_cited": "0/0", "evidence": "", "error": error})
            print(f"  {n}/{len(subset)} ERROR: {error.splitlines()[-1] if error else ''}")
        else:
            settings = settings or result.get("settings")
            ev = result.get("evidence", [])
            r_hit, r_tot = sf_recall(ev, q["supporting_facts_qarag"], cited_only=False)
            c_hit, _ = sf_recall(ev, q["supporting_facts_qarag"], cited_only=True)
            missing = (result.get("filter") or {}).get("titles_not_in_database", [])
            missing_total += len(missing)
            row.update({
                "answer": answer_for_reference(result),
                "sf_retrieved": f"{r_hit}/{r_tot}", "sf_cited": f"{c_hit}/{r_tot}",
                "evidence": " || ".join(f"[{e['label'][1:]}] {e['title']}: {e['question']} -> {e['answer']}"
                                        for e in ev if e.get("kind") == "kb"),
                "qplan_answer": result.get("answer", ""), "qplan_status": result.get("status", ""),
                "qplan_basis": result.get("basis", ""), "qplan_short": result.get("short_answer", ""),
                "qplan_cited": " ".join(result.get("cited", [])),
                "steps": result.get("steps"), "searches": result.get("searches"), "recalls": result.get("recalls"),
                "stop_reason": result.get("stop_reason"), "cost_usd": result.get("cost_usd"),
                "escalated": "" if result.get("escalated") is None else int(bool(result.get("escalated"))),
                "direct_status": result.get("direct_status", ""),
                "kept_direct": "" if result.get("kept_direct") is None else int(bool(result.get("kept_direct"))),
                "run_id": result.get("run_id"), "titles_not_in_database": "; ".join(missing),
                "error": result.get("error") or ""})
            route = {"0": "direct", "1": "escalated"}.get(str(row["escalated"]), "react")
            if str(row.get("kept_direct")) == "1":
                route = "kept-0"
            print(f"  {n}/{len(subset)} {row['qplan_status']:<10} {route:<9} {row['steps']} steps  ${float(row['cost_usd'] or 0):.4f}")
        rows.append(row)
        write_csv(args.out, rows)                    # saved after every question (for --resume)

    # A summary next to the CSV: hotpot_ask_eval.py --rescore copies its "database ..." line
    # into the scored summary as "original run", so the scored file records the planner settings.
    cost = sum(float(r.get("cost_usd") or 0) for r in rows)
    errors = sum(1 for r in rows if r.get("error"))
    s = settings or {}
    esc = [r.get("escalated") for r in rows if r.get("escalated") not in (None, "")]
    kept = sum(1 for r in rows if str(r.get("kept_direct")) == "1")
    direct_n = (f"{sum(1 for e in esc if str(e) == '0')} answered at step 0, "
                f"{sum(1 for e in esc if str(e) == '1')} escalated, {kept} step-0 answer(s) kept") if esc else "not used"
    header = (f"database {args.db} · setting {args.setting} · system qplan ReAct {s.get('qplan', '?')} · "
              f"memory {s.get('memory', args.memory)} · top_k {s.get('top_k')} · min_score {s.get('min_score')} · "
              f"max steps {s.get('max_steps')} · direct-first {s.get('direct_first', 'off')} "
              f"(k {s.get('direct_top_k', '-')}, {direct_n}) · react {s.get('react_model')} · synth {s.get('synth_model')} · "
              f"memory model {s.get('memory_model')} · reasoning {s.get('reasoning')} · prompts {s.get('prompts')} · "
              f"answering cost ${cost:.4f} · {len(rows)} questions · {errors} errors · "
              f"titles not in the database {missing_total}")
    summary = Path(args.out).with_name(Path(args.out).stem + "_summary.md")
    summary.write_text(f"# qplan answers for HotpotQA (not scored yet)\n\n{header}\n", encoding="utf-8")
    print(f"\nWrote {args.out} ({len(rows)} answers, {errors} errors, answering cost ${cost:.4f}, "
          f"{time.time() - started:.0f} s)")
    if missing_total:
        print(f"WARNING: {missing_total} paragraph title(s) were not found in the database - check --db.")
    scored = str(Path(args.out).with_name(Path(args.out).stem + "_scored.csv"))
    if args.score:
        cmd = [sys.executable, "hotpot_ask_eval.py", "--rescore", args.out, "--out", scored]
        if args.dry_run:
            cmd.append("--dry-run")
        print("Scoring:", " ".join(cmd))
        sys.exit(subprocess.call(cmd))
    print(f"Next: python hotpot_ask_eval.py --rescore {args.out} --out {scored}")


if __name__ == "__main__":
    main()
