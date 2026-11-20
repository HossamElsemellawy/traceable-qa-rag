"""
hotpot_ask_eval.py - answer-level HotpotQA evaluation (Path 1 baseline) with
qarag's existing `ask` logic.   Evaluation version: eval-v5

Uses qarag 0.4 exactly as it is: for each HotpotQA question it calls the same
function the `python -m qarag ask` command uses (qarag.answer.answer_question:
one search with the whole question, top-k pairs, grounded answer with
citations) and compares the answer with the GOLD answer. qarag is not changed;
the database is only read (plus qarag's normal schema check on opening).

Two phases
  1. ANSWER  (needs the database): retrieval + answer for every question.
  2. SCORE   (no database): short-answer extraction, LLM judge, EM/F1 and the
             other metrics. `--rescore results.csv` runs ONLY this phase on
             answers saved earlier - no new retrieval, no new answers - so a
             change of measurement is compared on exactly the same answers.

Run it from the folder of hotpot_index.py (venv active):

    python hotpot_ask_eval.py --db data\\hotpot --temperature 0 --out base100.csv
    python hotpot_ask_eval.py --rescore base100.csv --out base100_rescored.csv

Settings
  --setting distractor (default): each question searches ONLY its own 10
            paragraphs (the standard HotpotQA distractor setting); pooled: all
  -k 8      pairs given to the answering model (as `qarag ask`)
  --temperature / --max-tokens   answering model (use temperature 0)
  --judge-model   model of the LLM judge (default qwen3-235b-a22b-2507, a
                  different and stronger model than the answering one)
  --no-judge      skip the judge (no judge accuracy)

Measurement (identical for every system that is compared)
  Short answer  extraction prompt EXTRACT_VERSION (below): the minimal span that
                answers exactly what is asked, copied from the system's answer.
                Yes/no questions are detected by code (question starts with
                is/are/was/were/do/does/did/has/have/had/can/could/will/would/
                should) and must be answered yes or no. Sees only the question
                and the system's answer - never the gold answer or the
                evidence. Temperature 0.
                extract-v5 vs v4: an EMPTY extraction reply is asked once more
                (v4 returned "" for one answer that named Steve McGarrett).
                judge-v2 vs v1: reason first, then verdict (v1 once said
                "wrong" with a reason explaining the answer was correct);
                exact matches are counted correct without a judge call.
                extract-v4 vs v3: an answer followed by a remark that other
                information is missing is still extracted (v3 returned "unknown"
                for "The population was 321,520. The documents do not give more
                recent figures."), with one retry when the reply is "unknown"
                but the answer is not a plain refusal; "Is X or Y ...?" is a
                choice question, not yes/no; markdown * removed.
                extract-v3 vs v2: yes/no detected by code (v2 answered
                "Pocahontas" to a yes/no question); comparison options copied as
                written in the answer, not as in the question (v2 shortened
                "Dino Buzzati-Traverso"); no list of words to drop (v2 cut
                "film directors" to "directors").
  EM, F1        HotpotQA official normalisation (lower case, no punctuation,
                no articles a/an/the) on the short answer.
  Judge acc     an LLM judge (JUDGE_VERSION) decides whether the system's answer
                means the same as the gold answer (extra correct detail is fine;
                several candidates, a different entity, a hedge or "not
                available" are wrong). One-line reason per question in
                judge_reason - check a sample by hand and report the agreement.
  Cover         gold answer contained in the short answer
  Empty         answering model returned no text (hidden reasoning used the
                budget); retried once with 3x budget, then counted here
  Not found     the answer is only the refusal sentence
  Partial       a real answer that ALSO says some information is missing
  SF recall     gold supporting sentences inside the retrieved / cited pairs
  Cost          per purpose: embed_query + answer (answering), extract + judge
                (measurement)

Output
  For every question (screen and summary file): question, received answer,
  short answer, gold answer, then the scores (F1 calculation written out, the
  judge's verdict and reason).
  <out>.csv          one row per question
  <out>_summary.md   averages per type and overall, settings, cost, every question
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from qarag import llm
from qarag.config import Settings

EVAL_VERSION = "eval-v5"
EXTRACT_VERSION = "extract-v5"
JUDGE_VERSION = "judge-v2"
DEFAULT_JUDGE_MODEL = "openrouter/qwen/qwen3-235b-a22b-2507"
NOT_FOUND = "do not contain this information"
YES_NO_START = ("is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
                "can", "could", "will", "would", "should")

EXTRACT_PROMPT = """You extract the final short answer from a system's long answer, in the style of HotpotQA answers.

Question: {question}
Long answer: {answer}
{yes_no_line}
Rules:
1. Give the minimal span that answers exactly what the question asks - usually 1 to 5 words.
2. Answer the thing that is asked: if the question asks for a role, title, place, date, number or
   organisation, give that - not another person or entity mentioned in the long answer.
3. Copy names, titles, dates and numbers exactly as written in the long answer. Do not add
   anything: no years or notes in brackets, no explanation.
4. Comparison questions (which / who ... first, more, older ...): give the chosen option exactly as
   it is written in the long answer.
5. If the long answer states an answer anywhere, give it - even if it ALSO says that some other
   information is missing (for example "The population was 321,520. The documents do not give
   more recent figures." -> 321,520). Reply exactly unknown only if the long answer gives no answer
   to the question at all, or says the question cannot be answered.
{retry_line}
Reply with the short answer only, on one line."""

RETRY_LINE = ("\nNote: your previous reply was \"unknown\", but the long answer is not a plain refusal. Read it "
              "again: if it commits to an answer anywhere, give that answer; otherwise reply unknown.\n")

YES_NO_LINE = ("\nThis is a yes/no question: reply only yes or no (or unknown if the long answer "
               "gives no answer).\n")

JUDGE_PROMPT = """You grade the answer of a question-answering system against the reference (gold) answer.

Question: {question}
Gold answer: {gold}
System answer: {answer}
(System's short answer: {short})

The system answer is CORRECT if its final answer to the question is the same as the gold answer:
the same entity, value, date or yes/no - even if worded differently, shorter or longer, or with
extra correct detail (for example "Los Alamos Laboratory" for "Los Alamos", "Washington State
Cougars" for "Washington State", "Ozzy Osbourne" for "John Michael Osbourne").

It is WRONG if: it gives a different entity or value; it gives several candidates when one is
asked; it answers a different thing than the question asks (for example the actor when the role
is asked); it hedges or contradicts itself about the final answer; or it says the information is
not available.

Judge only whether the system answer agrees with the gold answer. Do not use your own knowledge
to decide whether the gold answer is true.

First write your reason, then your verdict, and make the verdict follow from the reason.
Reply with JSON only: {{"reason": "<one short sentence>", "correct": true or false}}"""


# ---------------------------------------------------------------------------
# HotpotQA official answer scoring (same normalisation as the official script)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        if p != g:
            return 0.0                      # official rule: yes/no must match exactly
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def f1_detail(prediction: str, gold: str) -> str:
    """The F1 calculation written out: normalised strings, shared words, precision, recall."""
    p, g = normalize_answer(prediction), normalize_answer(gold)
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
        return f"normalised '{p}' vs '{g}': yes/no answers must match exactly -> F1 0.00"
    if not same:
        return f"normalised '{p}' vs '{g}': no shared words -> F1 0.00"
    precision, recall = same / len(pt), same / len(gt)
    return (f"normalised '{p}' vs '{g}': {same} shared word(s); precision {same}/{len(pt)} = {precision:.2f}, "
            f"recall {same}/{len(gt)} = {recall:.2f} -> F1 = 2PR/(P+R) = {2 * precision * recall / (precision + recall):.2f}")


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def cover(short_answer: str, gold: str) -> float:
    """Gold answer contained in the SHORT answer (not the full text: a comparison
    answer names both options, so the full text would always "cover")."""
    g = normalize_answer(gold)
    return float(bool(g) and f" {g} " in f" {normalize_answer(short_answer)} ")


def is_refusal(answer: str) -> bool:
    """True only if the answer is NOTHING BUT the refusal sentence."""
    rest = normalize_answer(re.sub(r"\[\d+\]", "", answer)).replace(normalize_answer(
        "The indexed documents do not contain this information."), "").strip()
    return NOT_FOUND in answer and len(rest.split()) <= 2


def is_yes_no(question: str) -> bool:
    """Questions that start with an auxiliary verb ("Was Pocahontas made before ...",
    "Are both X and Y ...") expect yes or no - unless they offer options with "or"
    ("Is Dappy or Tobias Sammet German?" asks WHICH one: a choice, not yes/no)."""
    words = question.strip().split()
    if not words or words[0].lower().strip('"\'') not in YES_NO_START:
        return False
    return " or " not in f" {question.lower()} "


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# measurement calls
# ---------------------------------------------------------------------------
def extract_short(question: str, answer: str, model: str, dry_run: bool) -> str:
    if not answer.strip():
        return ""
    if is_refusal(answer):
        return "unknown"
    if dry_run:
        return re.sub(r"\[\d+\]", "", answer).strip()
    yes_no = is_yes_no(question)

    def ask(retry_line: str) -> str:
        prompt = EXTRACT_PROMPT.format(question=question, answer=answer, retry_line=retry_line,
                                       yes_no_line=YES_NO_LINE if yes_no else "")
        reply = _clean(llm.chat(model, [{"role": "user", "content": prompt}],
                                temperature=0.0, max_tokens=2000, purpose="extract"))
        reply = re.sub(r"\[\d+\]", "", reply).replace("**", "").replace("*", "").strip().strip('"').strip()
        return reply.splitlines()[0].strip() if reply else ""

    text = ask("")
    if not normalize_answer(text):
        text = ask("")                      # empty reply (e.g. only hidden reasoning): ask once more
    if normalize_answer(text) == "unknown":
        # The answer is not a plain refusal (checked above), so look once more: a remark like
        # "the documents do not give more recent figures" must not cancel an answer given earlier.
        text = ask(RETRY_LINE)
    if yes_no:
        first = normalize_answer(text).split()[:1]
        if first and first[0] in ("yes", "no"):
            text = first[0]
    return text


def judge(question: str, gold: str, answer: str, short: str, model: str, dry_run: bool) -> tuple[str, str]:
    """('1' | '0' | '', reason). '' = the judge failed to give a valid verdict."""
    if not answer.strip():
        return "0", "no answer (empty)"
    if is_refusal(answer):
        return "0", "system said the information is not available"
    if dry_run:
        return ("1" if cover(short, gold) else "0"), "dry run: gold contained in short answer"
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=answer, short=short)
    for _ in range(2):
        reply = _clean(llm.chat(model, [{"role": "user", "content": prompt}], temperature=0.0,
                                max_tokens=2000, purpose="judge"))
        try:
            verdict = llm.parse_json_object(reply)
            correct = verdict.get("correct")
            if isinstance(correct, str):
                correct = correct.strip().lower() in ("true", "yes", "1")
            return ("1" if correct else "0"), str(verdict.get("reason", "")).strip()
        except (ValueError, TypeError):
            continue
    return "", "judge reply was not valid JSON"


# ---------------------------------------------------------------------------
# phase 1: answers (needs qarag and the database)
# ---------------------------------------------------------------------------
def sf_recall(items, gold_facts) -> tuple[int, int]:
    """Gold supporting sentences (qarag numbering) found in the given evidence items."""
    got = defaultdict(set)
    for item in items:
        got[item.record.title] |= set(item.record.sentence_ids)
    facts = [(t, ids) for t, ids in gold_facts if ids]
    return sum(any(i in got[t] for i in ids) for t, ids in facts), len(facts)


def answer_phase(args, settings: Settings) -> tuple[list[dict], dict]:
    from qarag.answer import answer_question
    from qarag.store import QAStore

    subset = [json.loads(l) for l in Path(args.subset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        subset = subset[:args.n]
    store = QAStore(settings.sqlite_path)
    if store.get_meta("embed_model") == "mock-embedding":
        settings.dry_run = True                         # offline test database
    doc_ids = {r["title"]: r["doc_id"] for r in store.conn.execute("SELECT title, doc_id FROM documents")}
    info = {"database": args.db, "setting": args.setting, "k": args.k,
            "answer model": settings.effective_answer_model(),
            "answer temperature": settings.answer_temperature, "answer max tokens": settings.answer_max_tokens,
            "pair answer style": store.get_meta("answer_style"), "embedding variant": store.get_meta("embed_variant")}
    print(f"Answering {len(subset)} questions ({args.setting}, k={args.k}, {settings.effective_answer_model()}) ...")
    rows = []
    for n, q in enumerate(subset, start=1):
        where = None
        if args.setting == "distractor":
            where = {"doc_id": {"$in": [doc_ids[t] for t in q["titles"] if t in doc_ids]}}
        result = answer_question(q["question"], store, settings, k=args.k, where=where)
        if not result.text.strip():
            # Hidden reasoning used the whole budget: retry once with three times the budget.
            budget = settings.answer_max_tokens
            settings.answer_max_tokens = budget * 3
            result = answer_question(q["question"], store, settings, k=args.k, where=where)
            settings.answer_max_tokens = budget
        cited_items = [i for i in result.evidence if i.number in result.cited]
        r_hit, r_tot = sf_recall(result.evidence, q["supporting_facts_qarag"])
        c_hit, _ = sf_recall(cited_items, q["supporting_facts_qarag"])
        rows.append({"n": n, "id": q["id"], "type": q["type"], "question": q["question"],
                     "answer": result.text, "gold": q["answer"],
                     "sf_retrieved": f"{r_hit}/{r_tot}", "sf_cited": f"{c_hit}/{r_tot}",
                     "evidence": " || ".join(f"[{i.number}] {i.record.title}: {i.record.question} -> "
                                             f"{i.record.answer}" for i in result.evidence)})
        one_line = " ".join(result.text.split())
        print(f"[{n}/{len(subset)}] {q['type']}  {q['id']}\n"
              f"  Question : {q['question']}\n"
              f"  Answer   : {one_line[:300]}{' ...' if len(one_line) > 300 else ''}\n"
              f"  Gold     : {q['answer']}\n"
              f"  Supporting facts retrieved {r_hit}/{r_tot}, cited {c_hit}/{r_tot}  (scores follow after all answers)\n")
    store.close()
    return rows, info


def load_saved(path: str) -> tuple[list[dict], dict]:
    """Answers saved by an earlier run (its CSV), for --rescore."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = ("n", "id", "type", "question", "answer", "gold", "sf_retrieved", "sf_cited", "evidence")
    rows = [{k: r.get(k, "") for k in keep} for r in rows]
    for r in rows:
        r["n"] = int(r["n"])
    summary = Path(path).with_name(Path(path).stem + "_summary.md")
    info = {"answers from": f"{path} (re-scored, not re-answered)"}
    if summary.exists():
        header = [l for l in summary.read_text(encoding="utf-8").splitlines()
                  if l.startswith("Database ") or l.startswith("database ") or l.startswith("answers from ")]
        if header:
            info["original run"] = header[0]
    return rows, info


# ---------------------------------------------------------------------------
# phase 2: scoring (identical for answers made now or loaded with --rescore)
# ---------------------------------------------------------------------------
def score(rows: list[dict], args, settings: Settings) -> None:
    measure_model = settings.effective_answer_model()
    total = len(rows)
    for row in rows:
        answer, gold = row["answer"], row["gold"]
        empty = not answer.strip()
        short = (re.sub(r"\[\d+\]", "", answer).strip() if args.no_extract
                 else extract_short(row["question"], answer, measure_model, settings.dry_run))
        r_hit, r_tot = (int(x) for x in (row["sf_retrieved"] or "0/0").split("/"))
        c_hit = int((row["sf_cited"] or "0/0").split("/")[0])
        row.update({"short_answer": short,
                    "em": exact_match(short, gold), "f1": round(f1_score(short, gold), 4),
                    "cover": cover(short, gold), "empty": float(empty),
                    "not_found": float(is_refusal(answer)),
                    "partial": float(NOT_FOUND in answer and not is_refusal(answer)),
                    "_r": (r_hit, c_hit, r_tot)})
        if args.no_judge:
            row["judge_correct"], row["judge_reason"] = "", "judge not run"
        elif row["em"]:
            # An exact match is correct by definition: no judge call, and the judge can never
            # contradict EM (judge-v1 once returned "wrong" with a reason saying "correct").
            row["judge_correct"], row["judge_reason"] = "1", "exact match with the gold answer"
        else:
            row["judge_correct"], row["judge_reason"] = judge(row["question"], gold, answer, short,
                                                              args.judge_model, settings.dry_run)
        # correct (by EM or the judge) although the CITED pairs hold no gold supporting sentence
        row["correct_without_gold_citation"] = float((row["em"] or row["judge_correct"] == "1") and c_hit == 0)
        row["verdict"] = ("correct" if row["em"] else
                          "correct (judge)" if row["judge_correct"] == "1" else
                          "partly correct" if row["f1"] > 0 else
                          "error: empty answer" if empty else "not found" if row["not_found"] else "wrong")
        row["f1_calculation"] = f1_detail(short, gold)
        judge_text = ({"1": "correct", "0": "wrong", "": "n/a"}[row["judge_correct"]]
                      + (f" - {row['judge_reason']}" if row["judge_reason"] else ""))
        row["_block"] = [f"[{row['n']}/{total}] {row['type']}  {row['id']}  -> {row['verdict'].upper()}",
                         f"  Question : {row['question']}",
                         f"  Answer   : {answer}",
                         f"  Short    : {short}",
                         f"  Gold     : {gold}",
                         f"  EM {row['em']:.0f} | F1 {row['f1']:.2f} | Cover {row['cover']:.0f} | "
                         f"supporting facts retrieved {row['sf_retrieved']}, cited {row['sf_cited']}",
                         f"  F1       : {row['f1_calculation']}",
                         f"  Judge    : {judge_text}"]
        print("\n".join(row["_block"]) + "\n")


def summarize(group: list[dict]) -> dict:
    n = len(group)
    r = [x["_r"] for x in group]
    tot = sum(t for _, _, t in r) or 1
    judged = [x for x in group if x["judge_correct"] in ("0", "1")]
    return {"questions": n,
            "EM": 100 * sum(x["em"] for x in group) / n, "F1": 100 * sum(x["f1"] for x in group) / n,
            "Judge acc": (100 * sum(x["judge_correct"] == "1" for x in judged) / len(judged)) if judged else None,
            "Cover": 100 * sum(x["cover"] for x in group) / n,
            "Empty": 100 * sum(x["empty"] for x in group) / n,
            "Not found": 100 * sum(x["not_found"] for x in group) / n,
            "Partial": 100 * sum(x["partial"] for x in group) / n,
            "SF ret": 100 * sum(h for h, _, _ in r) / tot,
            "SF cit": 100 * sum(c for _, c, _ in r) / tot,
            "NoGoldCit": 100 * sum(x["correct_without_gold_citation"] for x in group) / n}


def write_outputs(rows: list[dict], info: dict, args, started: float) -> None:
    groups = {"all": rows}
    for t in sorted({x["type"] for x in rows}):
        groups[t] = [x for x in rows if x["type"] == t]
    cols = ["EM", "F1", "Judge acc", "Cover", "Empty", "Not found", "Partial", "SF ret", "SF cit", "NoGoldCit"]
    fmt = lambda v: "n/a" if v is None else f"{v:.1f}"   # noqa: E731

    info = {**info, "evaluation": EVAL_VERSION, "short-answer extraction": EXTRACT_VERSION,
            "judge": "not run" if args.no_judge else f"{JUDGE_VERSION}, {args.judge_model}"}
    lines = ["# HotpotQA answer evaluation (qarag ask logic, Path 1)", ""]
    lines += [" · ".join(f"{k} {v}" for k, v in info.items()), ""]
    lines += ["| group | questions | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    print("\n" + f"{'group':<12}{'n':>4}" + "".join(f"{c:>10}" for c in cols))
    for name, group in groups.items():
        s = summarize(group)
        lines.append(f"| {name} | {s['questions']} | " + " | ".join(fmt(s[c]) for c in cols) + " |")
        print(f"{name:<12}{s['questions']:>4}" + "".join(f"{fmt(s[c]):>10}" for c in cols))
    if not args.no_judge:
        disagree = [x for x in rows if x["judge_correct"] in ("0", "1") and (x["judge_correct"] == "1") != bool(x["em"])]
        lines += ["", f"Judge vs EM: {len(disagree)} question(s) where the judge and EM disagree - check these by hand: "
                  + (", ".join(str(x["n"]) for x in disagree) or "none")]

    snapshot = llm.METER.snapshot().get("", {})
    usage = llm.METER.total()
    parts = ", ".join(f"{p} {u['calls']} calls / {u['prompt_tokens'] + u['completion_tokens']:,} tokens"
                      for p, u in sorted(snapshot.items()))
    cost_line = (f"{usage.calls} calls, {usage.prompt_tokens:,} tokens in, {usage.completion_tokens:,} out, "
                 f"cost {usage.cost_text()}, {time.time() - started:.0f} s ({parts})")
    lines += ["", f"Usage of this run: {cost_line}", "", "## Per question", ""]
    for x in rows:
        b = x["_block"]
        lines += [f"### {b[0]}", "", *[f"- {line.strip()}" for line in b[1:]], ""]
    print(f"\nUsage of this run: {cost_line}")

    fields = ["n", "id", "type", "question", "answer", "short_answer", "gold", "em", "f1", "judge_correct",
              "judge_reason", "cover", "empty", "not_found", "partial", "correct_without_gold_citation",
              "sf_retrieved", "sf_cited", "verdict", "f1_calculation", "evidence"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = Path(args.out).with_name(Path(args.out).stem + "_summary.md")
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} and {summary.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer-level HotpotQA evaluation with qarag's ask logic.")
    ap.add_argument("--db", help=r"database folder, e.g. data\hotpot (not needed with --rescore)")
    ap.add_argument("--rescore", help="CSV of an earlier run: re-score its answers only (no new answers)")
    ap.add_argument("--subset", default="hotpot_subset.jsonl")
    ap.add_argument("--n", type=int, default=0, help="only the first N questions (0 = all)")
    ap.add_argument("-k", type=int, default=8, help="pairs given to the answering model (default 8)")
    ap.add_argument("--setting", choices=["distractor", "pooled"], default="distractor")
    ap.add_argument("--answer-model", help="model that writes the answer (default as qarag ask)")
    ap.add_argument("--temperature", type=float, help="answering temperature for this run (qarag default 0.1)")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="answer token budget incl. hidden reasoning (qarag default 2000; default here 4000)")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="model of the LLM judge")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--no-extract", action="store_true", help="score the full answer (no extraction call)")
    ap.add_argument("--dry-run", action="store_true", help="offline test: no model calls for extraction/judge")
    ap.add_argument("--out", default="hotpot_ask_results.csv")
    args = ap.parse_args()
    if not args.db and not args.rescore:
        ap.error("give --db (answer and score) or --rescore <csv> (score saved answers)")

    settings = Settings()
    settings.dry_run = args.dry_run
    if args.answer_model:
        settings.answer_model = args.answer_model
    if args.temperature is not None:
        settings.answer_temperature = args.temperature
    settings.answer_max_tokens = args.max_tokens
    if args.db:
        settings.db_dir = Path(args.db)

    llm.METER.reset()
    started = time.time()
    rows, info = load_saved(args.rescore) if args.rescore else answer_phase(args, settings)
    if args.rescore and args.n:
        rows = rows[:args.n]
    score(rows, args, settings)
    write_outputs(rows, info, args, started)


if __name__ == "__main__":
    sys.exit(main())
