"""
hotpot.py - HotpotQA evaluation data (Stage 0 go/no-go and planner evaluation).

Step 1 (this module, command `hotpot-subset`): choose a fixed, reproducible
subset of HotpotQA questions and write it to ONE shared file that both
programs use:

    qarag  -> indexes only the CONTEXT PARAGRAPHS of these questions
    qplan  -> asks the QUESTIONS and scores the answers

    python -m qarag hotpot-subset --n 100 --seed 42 -o hotpot_subset.jsonl

Where the data comes from
  * default: Hugging Face `hotpotqa/hotpot_qa`, config `distractor`, split
    `validation` (needs `pip install datasets`; downloaded once, then cached)
  * or `--from-file` with the official JSON file hotpot_dev_distractor_v1.json
Both layouts are converted to the same record format (see `normalize`).

How the questions are chosen (stratified random sample)
  * Strata = (type, level). type is "bridge" or "comparison"; level is easy /
    medium / hard. NOTE: the official validation (dev) set contains only
    "hard" questions (easy/medium exist only in the training set), so in
    practice the strata are the two question types.
  * --allocation equal (default): the same number from every stratum, so
    comparison questions are not swamped by bridge questions (the dev set is
    about 80% bridge). --allocation proportional keeps the dataset's mix.
    Report which one was used in the thesis.
  * Inside each stratum: a seeded random sample (same seed = same questions,
    on any computer), after sorting by id so the order of the source does not
    matter.
  * --exclude other_subset.jsonl: skip questions already used in another
    subset, e.g. to draw a separate tuning set later that never overlaps the
    test set.
  * Questions with broken annotations are skipped and counted (a supporting
    fact that points to a title missing from the context, or to a sentence
    number the paragraph does not have).

Output
  hotpot_subset.jsonl       one question per line (format below)
  hotpot_subset.meta.json   how the subset was made: source, seed, strata
                            counts, number of distinct paragraphs, date

Record format (one JSON object per line)
  {"id": "5a8b57f25542995d1e6f1371",
   "question": "...", "answer": "...", "type": "comparison", "level": "hard",
   "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
   "context": [["Scott Derrickson", ["sentence 0", "sentence 1", ...]], ... 10 paragraphs],
   "titles": ["Scott Derrickson", "Ed Wood", ...]}       # the 10 titles, = the distractor filter

The QUESTIONS and ANSWERS are in this file for scoring only: qarag never
indexes them (that would leak the test into the knowledge base).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"


# ---------------------------------------------------------------------------
# 1. reading the source, in either layout
# ---------------------------------------------------------------------------
def normalize(item: dict) -> dict:
    """One HotpotQA question in the shared record format.

    Hugging Face layout:  context = {"title": [...], "sentences": [[...], ...]}
                          supporting_facts = {"title": [...], "sent_id": [...]}
    Official JSON layout: context = [[title, [sentences]], ...]
                          supporting_facts = [[title, sent_id], ...], id in "_id"
    """
    ctx = item["context"]
    if isinstance(ctx, dict):
        context = [[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])]
    else:
        context = [[t, list(s)] for t, s in ctx]
    sf = item["supporting_facts"]
    if isinstance(sf, dict):
        facts = [[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])]
    else:
        facts = [[t, int(i)] for t, i in sf]
    return {
        "id": item.get("id") or item.get("_id"),
        "question": item["question"],
        "answer": item["answer"],
        "type": item.get("type", ""),
        "level": item.get("level", ""),
        "supporting_facts": facts,
        "context": context,
        "titles": [t for t, _ in context],
    }


def load_source(from_file: Optional[Path] = None) -> tuple[list[dict], str]:
    """All validation questions, normalized, plus a description of the source."""
    if from_file:
        data = json.loads(Path(from_file).read_text(encoding="utf-8"))
        return [normalize(x) for x in data], f"file {Path(from_file).name}"
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ValueError("The 'datasets' package is needed: pip install datasets "
                         "(or give the official JSON file with --from-file).") from error
    dataset = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    return [normalize(x) for x in dataset], f"huggingface {HF_DATASET}/{HF_CONFIG}/{HF_SPLIT}"


def problems(record: dict) -> list[str]:
    """Annotation errors that would make supporting-fact scoring impossible."""
    paragraphs = {t: s for t, s in record["context"]}
    out = []
    for title, sent_id in record["supporting_facts"]:
        if title not in paragraphs:
            out.append(f"supporting title not in context: {title!r}")
        elif not 0 <= sent_id < len(paragraphs[title]):
            out.append(f"sentence {sent_id} out of range for {title!r} ({len(paragraphs[title])} sentences)")
    if len(set(record["titles"])) != len(record["titles"]):
        out.append("duplicate titles in context")
    return out


def read_subset(path: Path) -> list[dict]:
    """Read a subset file written by `write_subset` (used by the loader too)."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 2. stratified sampling
# ---------------------------------------------------------------------------
def _allocate(sizes: dict[tuple, int], n: int, allocation: str) -> dict[tuple, int]:
    """How many questions to take from each stratum."""
    strata = sorted(sizes)
    if allocation == "proportional":
        total = sum(sizes.values())
        exact = {s: n * sizes[s] / total for s in strata}
        take = {s: min(sizes[s], int(exact[s])) for s in strata}
        # hand out the remainder to the strata with the largest fractional part
        for s in sorted(strata, key=lambda s: exact[s] - int(exact[s]), reverse=True):
            if sum(take.values()) >= n:
                break
            if take[s] < sizes[s]:
                take[s] += 1
        return take
    # equal: same share per stratum; a stratum that is too small gives its
    # unused share to the others
    take = {s: 0 for s in strata}
    remaining = n
    open_strata = [s for s in strata if sizes[s] > 0]
    while remaining > 0 and open_strata:
        share = max(1, remaining // len(open_strata))
        for s in list(open_strata):
            add = min(share, sizes[s] - take[s], remaining)
            take[s] += add
            remaining -= add
            if take[s] >= sizes[s]:
                open_strata.remove(s)
            if remaining == 0:
                break
    return take


def build_subset(records: Iterable[dict], n: int, seed: int, allocation: str = "equal",
                 exclude_ids: Optional[set[str]] = None) -> tuple[list[dict], dict]:
    """Return (chosen records, statistics)."""
    exclude_ids = exclude_ids or set()
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    skipped_broken, skipped_excluded = 0, 0
    for r in records:
        if r["id"] in exclude_ids:
            skipped_excluded += 1
            continue
        if problems(r):
            skipped_broken += 1
            continue
        by_stratum[(r["type"], r["level"])].append(r)

    sizes = {s: len(v) for s, v in by_stratum.items()}
    take = _allocate(sizes, n, allocation)
    rng = random.Random(seed)
    chosen: list[dict] = []
    for stratum in sorted(by_stratum):
        pool = sorted(by_stratum[stratum], key=lambda r: r["id"])   # source order never matters
        chosen.extend(rng.sample(pool, take[stratum]))
    chosen.sort(key=lambda r: r["id"])

    paragraphs = Counter(t for r in chosen for t in r["titles"])
    stats = {
        "questions": len(chosen),
        "requested": n,
        "seed": seed,
        "allocation": allocation,
        "available_per_stratum": {f"{t}/{lv}": c for (t, lv), c in sorted(sizes.items())},
        "chosen_per_stratum": {f"{t}/{lv}": c for (t, lv), c in sorted(take.items()) if c},
        "skipped_broken_annotation": skipped_broken,
        "skipped_excluded": skipped_excluded,
        "paragraphs_total": sum(paragraphs.values()),
        "paragraphs_distinct": len(paragraphs),
        "paragraphs_shared_by_2plus_questions": sum(1 for c in paragraphs.values() if c > 1),
        "sentences_distinct": sum(len(s) for t, s in
                                  {t: s for r in chosen for t, s in r["context"]}.items()),
        "supporting_facts": sum(len(r["supporting_facts"]) for r in chosen),
    }
    return chosen, stats


def write_subset(chosen: list[dict], stats: dict, output: Path, source: str) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta_path = output.with_suffix(".meta.json")
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source, "file": output.name, **stats}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path
