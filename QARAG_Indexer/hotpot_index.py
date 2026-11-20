"""
hotpot_index.py - HotpotQA evaluation data for qarag, WITHOUT changing qarag.

A stand-alone helper script. It only READS the HotpotQA dataset and only CALLS
the normal qarag command (`python -m qarag index ...`); no qarag file is
modified and the dataset itself is never changed.

Put this file in the folder you run `python -m qarag` from, activate the venv,
then:

  Step 1 - choose the questions and write their paragraphs as files
      python hotpot_index.py select --n 100 --seed 42

  Step 2 - index the paragraphs, question by question (append mode)
      python hotpot_index.py index --db data\\hotpot --fresh --style short --limit 20   (pilot)
      python hotpot_index.py index --db data\\hotpot --style short                     (the rest)

What `select` does
  * Loads HotpotQA validation (distractor) from Hugging Face (`pip install datasets`),
    or the official JSON file with --from-file hotpot_dev_distractor_v1.json.
  * Takes a seeded random sample of --n questions, the same number of each
    question type (bridge / comparison). The validation set has only "hard"
    questions, so the type is the only category. --proportional keeps the
    dataset mix instead (about 80% bridge).
  * Writes one Markdown file per DISTINCT paragraph into hotpot_paragraphs\\:
        # <exact HotpotQA title>
        <sentence 0>
        <sentence 1>
        ...
    One sentence per line, because qarag treats a line break as a sentence
    boundary. A paragraph shared by several questions is written once.
  * Writes hotpot_subset.jsonl (one question per line, for scoring only - the
    questions and answers are NEVER indexed) and hotpot_subset.meta.json.
  * Sentence numbers: HotpotQA's supporting facts are (title, sentence number).
    qarag numbers sentences itself (it skips empty lines and may split a line
    in two). To score retrieval exactly, this script runs qarag's OWN reading
    and sentence-splitting code on every paragraph file and records, for every
    HotpotQA sentence, the qarag sentence number(s) it became:
        supporting_facts_qarag: [[title, [qarag idx, ...]], ...]   in hotpot_subset.jsonl
        hotpot_sentence_map.json                                   all paragraphs
    It prints how many paragraphs kept HotpotQA's numbering exactly.

What `index` does
  * For each question in the subset (in file order), runs
        python -m qarag index <its 10 paragraph files> --db <db> [-s style]
    Paragraphs already indexed by an earlier question are skipped by qarag
    automatically (unchanged document = no cost). So the run can be stopped
    and started again at any time; nothing is paid twice.
  * --fresh: the FIRST call uses --mode overwrite -y (empties the database).
    Use it once, at the start. Never use it to resume.
  * --limit N: only the first N questions (pilot). --start K: begin at question K.
  * Every question is a separate qarag run, so `python -m qarag report` shows
    the cost of each one. A summary per question is appended to
    hotpot_index_log.csv.

How the planner finds one question's paragraphs (distractor setting)
  documents.title in the qarag database = the exact HotpotQA title, so the
  10 titles of a question (field "titles" in hotpot_subset.jsonl) identify its
  10 documents: SELECT doc_id FROM documents WHERE title IN (...), then filter
  the Chroma search with {"doc_id": {"$in": [...]}}.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ===========================================================================
# reading HotpotQA (two layouts -> one record format)
# ===========================================================================
def normalize(item: dict) -> dict:
    """Hugging Face layout (context = {"title": [...], "sentences": [...]}) or
    official JSON layout (context = [[title, [sentences]], ...]) -> one format."""
    ctx = item["context"]
    context = ([[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])]
               if isinstance(ctx, dict) else [[t, list(s)] for t, s in ctx])
    sf = item["supporting_facts"]
    facts = ([[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])]
             if isinstance(sf, dict) else [[t, int(i)] for t, i in sf])
    return {"id": item.get("id") or item.get("_id"), "question": item["question"],
            "answer": item["answer"], "type": item.get("type", ""), "level": item.get("level", ""),
            "supporting_facts": facts, "context": context, "titles": [t for t, _ in context]}


def load_hotpot(from_file: str | None) -> tuple[list[dict], str]:
    if from_file:
        data = json.loads(Path(from_file).read_text(encoding="utf-8"))
        return [normalize(x) for x in data], f"file {Path(from_file).name}"
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Please install the datasets package first:  pip install datasets")
    print("Loading HotpotQA from Hugging Face (downloaded once, then cached) ...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    return [normalize(x) for x in ds], "huggingface hotpotqa/hotpot_qa distractor validation"


def broken(record: dict) -> bool:
    """True if a supporting fact points to a missing title or sentence number."""
    paragraphs = {t: s for t, s in record["context"]}
    if len(paragraphs) != len(record["context"]):
        return True
    return any(t not in paragraphs or not 0 <= i < len(paragraphs[t])
               for t, i in record["supporting_facts"])


# ===========================================================================
# sampling
# ===========================================================================
def sample(records: list[dict], n: int, seed: int, proportional: bool) -> tuple[list[dict], dict]:
    strata: dict[tuple, list[dict]] = defaultdict(list)
    n_broken = 0
    for r in records:
        if broken(r):
            n_broken += 1
        else:
            strata[(r["type"], r["level"])].append(r)
    keys = sorted(strata)
    sizes = {k: len(strata[k]) for k in keys}
    if proportional:
        total = sum(sizes.values())
        take = {k: round(n * sizes[k] / total) for k in keys}
        while sum(take.values()) > n:                       # fix rounding
            take[max(keys, key=lambda k: take[k])] -= 1
        while sum(take.values()) < n:
            take[max(keys, key=lambda k: sizes[k] - take[k])] += 1
    else:                                                   # equal per stratum
        take, left, open_keys = {k: 0 for k in keys}, n, list(keys)
        while left > 0 and open_keys:
            share = max(1, left // len(open_keys))
            for k in list(open_keys):
                add = min(share, sizes[k] - take[k], left)
                take[k] += add
                left -= add
                if take[k] >= sizes[k]:
                    open_keys.remove(k)
                if left == 0:
                    break
    rng = random.Random(seed)
    chosen = []
    for k in keys:
        pool = sorted(strata[k], key=lambda r: r["id"])      # source order never matters
        chosen.extend(rng.sample(pool, min(take[k], len(pool))))
    chosen.sort(key=lambda r: r["id"])
    info = {"available": {f"{t}/{lv}": c for (t, lv), c in sizes.items()},
            "chosen": {f"{t}/{lv}": c for (t, lv), c in take.items() if c},
            "skipped_broken_annotation": n_broken}
    return chosen, info


# ===========================================================================
# paragraph files and the sentence-number map
# ===========================================================================
def file_name(title: str) -> str:
    """A Windows-safe, unique file name for a title. The short hash keeps
    titles apart that differ only in case or in characters Windows forbids."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")[:80] or "paragraph"
    return f"{safe}__{hashlib.sha1(title.encode('utf-8')).hexdigest()[:8]}.md"


def paragraph_text(title: str, sentences: list[str]) -> str:
    lines = [" ".join(s.split()) for s in sentences]          # one HotpotQA sentence per line
    return f"# {title}\n\n" + "\n".join(lines) + "\n"


def _key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def sentence_map(hotpot_sentences: list[str], qarag_sentences: list[str]) -> list[list[int]]:
    """For each HotpotQA sentence, the qarag sentence numbers it became.
    [] = qarag dropped it (empty or a stray character)."""
    result, j = [], 0
    for s in hotpot_sentences:
        target = _key(s)
        if not target:
            result.append([])
            continue
        got, joined = [], ""
        while j < len(qarag_sentences) and len(joined) < len(target):
            joined += _key(qarag_sentences[j])
            got.append(j)
            j += 1
        if joined != target:
            # Unexpected (qarag changed the text); keep what was consumed but flag it.
            result.append([-1] + got)
        else:
            result.append(got)
    return result


def qarag_sentences_of(path: Path) -> list[str]:
    """Run qarag's own file reading and sentence splitting on one file
    (exactly what `qarag index` will do), without indexing anything."""
    from qarag.chunker import split_sentences
    from qarag.sources import load_document
    document = load_document(str(path), user_agent="hotpot_index.py")
    return [s.text for s in split_sentences(document)]


# ===========================================================================
# commands
# ===========================================================================
def cmd_select(args) -> None:
    records, source = load_hotpot(args.from_file)
    chosen, info = sample(records, args.n, args.seed, args.proportional)

    folder = Path(args.paragraphs)
    folder.mkdir(parents=True, exist_ok=True)
    texts: dict[str, list[str]] = {}
    conflicts = 0
    for r in chosen:
        for title, sentences in r["context"]:
            if title in texts and texts[title] != sentences:
                conflicts += 1                       # same title, different text: keep the first
            texts.setdefault(title, sentences)

    print(f"Writing {len(texts)} paragraph files to {folder}\\ and checking sentence numbers ...")
    smap, identical, changed, problems = {}, 0, 0, 0
    for title, sentences in texts.items():
        path = folder / file_name(title)
        content = paragraph_text(title, sentences)
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        mapping = sentence_map(sentences, qarag_sentences_of(path))
        smap[title] = {"file": path.name, "hotpot_to_qarag": mapping}
        if any(m and m[0] == -1 for m in mapping):
            problems += 1
        elif all(m == [i] for i, m in enumerate(mapping)):
            identical += 1
        else:
            changed += 1

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in chosen:
            r = dict(r)
            r["paragraph_files"] = [smap[t]["file"] for t in r["titles"]]
            r["supporting_facts_qarag"] = [[t, smap[t]["hotpot_to_qarag"][i]] for t, i in r["supporting_facts"]]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.paragraphs, "hotpot_sentence_map.json").write_text(
        json.dumps(smap, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
            "questions": len(chosen), "requested": args.n, "seed": args.seed,
            "allocation": "proportional" if args.proportional else "equal", **info,
            "paragraph_files": len(texts), "paragraph_folder": str(folder),
            "same_title_different_text": conflicts,
            "paragraphs_numbering_identical": identical,
            "paragraphs_numbering_shifted": changed,
            "paragraphs_numbering_unmatched": problems,
            "sentences": sum(len(s) for s in texts.values()),
            "supporting_facts": sum(len(r["supporting_facts"]) for r in chosen)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSource: {source} ({len(records):,} questions)")
    print(f"Chosen: {len(chosen)} questions  " + ", ".join(f"{k} {v}" for k, v in info["chosen"].items()))
    print(f"Paragraph files: {len(texts)} distinct ({sum(len(r['titles']) for r in chosen)} with repeats), "
          f"{meta['sentences']:,} sentences, {meta['supporting_facts']} supporting facts")
    print(f"Sentence numbers: {identical} paragraphs identical to HotpotQA, {changed} shifted "
          f"(mapped in supporting_facts_qarag), {problems} unmatched")
    if info["skipped_broken_annotation"]:
        print(f"Skipped {info['skipped_broken_annotation']} question(s) with broken annotations.")
    if conflicts:
        print(f"Note: {conflicts} title(s) appeared with different text; the first version was kept.")
    print(f"Wrote {out} and {out.with_suffix('.meta.json').name}")


def cmd_index(args) -> None:
    subset = [json.loads(line) for line in Path(args.subset).read_text(encoding="utf-8").splitlines() if line.strip()]
    folder = Path(args.paragraphs)
    todo = subset[args.start - 1:]
    if args.limit:
        todo = todo[:args.limit]
    log_path = Path(args.log)
    new_log = not log_path.exists()
    print(f"Indexing {len(todo)} question(s) into {args.db} (questions {args.start}..{args.start + len(todo) - 1} "
          f"of {len(subset)})")
    for n, q in enumerate(todo, start=args.start):
        files = [str((folder / name).resolve()) for name in q["paragraph_files"]]
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            sys.exit(f"Paragraph file missing: {missing[0]} - run `select` again.")
        command = [sys.executable, "-m", "qarag", "index", *files, "--db", args.db]
        if args.style:
            command += ["-s", args.style]
        if args.dry_run:
            command += ["--dry-run"]
        if args.fresh and n == args.start:
            command += ["-m", "overwrite", "-y"]
        print(f"\n===== question {n}/{len(subset)}  {q['id']}  ({q['type']})  {len(files)} paragraphs")
        started = time.time()
        result = subprocess.run(command)
        seconds = round(time.time() - started, 1)
        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if new_log:
                writer.writerow(["time", "number", "question_id", "type", "paragraphs", "exit_code", "seconds"])
                new_log = False
            writer.writerow([datetime.now().isoformat(timespec="seconds"), n, q["id"], q["type"],
                             len(files), result.returncode, seconds])
        if result.returncode != 0:
            print(f"! qarag stopped with an error on question {n}. Fix it and resume with --start {n}.")
            sys.exit(result.returncode)
    print(f"\nDone. Per-question log: {log_path}. Cost per run: python -m qarag report --db {args.db}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HotpotQA evaluation data for qarag (no qarag changes).")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="choose the questions and write their paragraph files")
    s.add_argument("--n", type=int, default=100, help="number of questions (default 100)")
    s.add_argument("--seed", type=int, default=42, help="random seed (same seed = same questions)")
    s.add_argument("--proportional", action="store_true", help="keep the dataset's type mix instead of equal numbers")
    s.add_argument("--from-file", help="official hotpot_dev_distractor_v1.json instead of Hugging Face")
    s.add_argument("--out", default="hotpot_subset.jsonl", help="subset file (default hotpot_subset.jsonl)")
    s.add_argument("--paragraphs", default="hotpot_paragraphs", help="folder for the paragraph files")
    s.set_defaults(func=cmd_select)

    i = sub.add_parser("index", help="index the paragraphs question by question with qarag")
    i.add_argument("--db", required=True, help=r"qarag database folder, e.g. data\hotpot")
    i.add_argument("--subset", default="hotpot_subset.jsonl")
    i.add_argument("--paragraphs", default="hotpot_paragraphs")
    i.add_argument("--style", choices=["short", "sentence", "detailed"],
                   help="qarag answer style (must stay the same for the whole database)")
    i.add_argument("--fresh", action="store_true", help="empty the database on the first call (overwrite)")
    i.add_argument("--limit", type=int, help="only this many questions (pilot)")
    i.add_argument("--start", type=int, default=1, help="first question number (1 = first)")
    i.add_argument("--log", default="hotpot_index_log.csv")
    i.add_argument("--dry-run", action="store_true", help="pass --dry-run to qarag (offline test, no cost)")
    i.set_defaults(func=cmd_index)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
