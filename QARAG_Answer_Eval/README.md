# Answer evaluation: short-answer extraction, LLM judge, EM/F1

This program scores the answers of a question-answering system on HotpotQA. It implements the **optional evaluation stage** of the paper (Section 3.3, metrics in Section 4.5, prompts in Appendix A.6 and A.7).

For each question it:

1. **Extracts a short answer.** A scoring model reads only the question and the system's long answer and returns the minimal span that answers the question. It never sees the gold answer or the evidence. Yes/no questions are detected by rule and must be answered *yes* or *no*. An answer that is only the refusal sentence becomes *unknown*.
2. **Computes Exact Match and F1** on the short answer, with the official HotpotQA normalisation (lower case, no punctuation, no articles).
3. **Asks an LLM judge** whether the long answer means the same as the gold answer. The judge writes a one-sentence reason first, then the verdict. An exact match is counted as correct without a judge call.
4. **Measures supporting-fact recall:** the share of the gold supporting sentences found in the retrieved pairs and in the cited pairs.

| Version tag | Meaning |
|---|---|
| `eval` | evaluation script version |
| `extract` | short-answer extraction prompt (paper Appendix A.6) |
| `judge` | judge prompt (paper Appendix A.7) |

The tags are written into every summary file, so a result can always be traced to the prompts that produced it.

## Two phases

| Phase | What it does | Needs |
|---|---|---|
| 1. Answer | Retrieves QA pairs and writes an answer for every question, with the indexer's single-search `ask` logic (the paper's single-retrieval baseline) | the QA knowledge base and the full `qarag` package of the [indexer](../QARAG_Indexer/) |
| 2. Score | Extraction, judge, EM/F1 and the other metrics | only this folder and an API key |

`--rescore answers.csv` runs **only phase 2** on answers saved earlier, without new retrieval or new answers. Any system's answers can therefore be scored with exactly the same measurement, including the planner's answers.

> This folder contains only the parts of the `qarag` package that phase 2 needs (`config.py` and `llm.py`). Phase 1 uses the indexer's retrieval and answering code, so it runs with the indexer's full `qarag` package: copy `hotpot_ask_eval.py` into the indexer folder and run it there.

## Installation

Python 3.10 or later.

```bash
cd QARAG_Answer_Eval
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your OpenRouter key in it:

```
OPENROUTER_API_KEY=sk-or-...
```

All model calls go through [OpenRouter](https://openrouter.ai) with LiteLLM. Never commit your `.env` file; it is excluded by `.gitignore`.

## Quick test without API calls

```bash
python hotpot_ask_eval.py --rescore examples/sample_answers.csv --dry-run --out sample_scored.csv
```

`--dry-run` makes no model calls. The "short answer" is the long answer with citation markers removed, and the "judge" checks whether the gold answer is contained in it. Use it to check the installation and the file formats, not for real scores.

## Scoring saved answers (phase 2)

```bash
python hotpot_ask_eval.py --rescore answers.csv --out answers_scored.csv \
    --answer-model openrouter/z-ai/glm-5.3-flash \
    --judge-model  openrouter/z-ai/glm-5.3-flash
```

In `--rescore` mode, `--answer-model` selects the model that **extracts the short answer**, and `--judge-model` the model that gives the verdict. Without `--answer-model`, the extraction model is `QARAG_ANSWER_MODEL` from `.env`, or the generator default in `qarag/config.py`.

### Reproducing the paper's five-judge panel

In the paper, each of the five panel judges performs both the extraction and the verdict (Section 4.5). Claude Opus 5.5 scores every answer as a reference judge outside the panel. Run the script once per judge on the same answers file:

```bash
for J in z-ai/glm-5.3-flash deepseek/deepseek-v4.1-flash qwen/qwen3-vl-235b-a22b-instruct \
         google/gemini-3.8-flash anthropic/claude-haiku-5.5 anthropic/claude-opus-5.5; do
  python hotpot_ask_eval.py --rescore answers.csv \
      --answer-model openrouter/$J --judge-model openrouter/$J \
      --out "scored_$(basename $J).csv"
done
```

On Windows `cmd`, run the six commands one by one with the model names above.

The paper's panel verdict counts an answer as correct when at least 3 of the 5 panel judges accept it. It is taken over these per-judge files: each has one row per question, with the judge's verdict in `judge_correct` and its short answer in `short_answer`.

## Answering and scoring in one run (phase 1 + 2)

Requires the indexer's full `qarag` package and a knowledge base built from HotpotQA. Run from the indexer folder:

```bash
python hotpot_ask_eval.py --db data/hotpot --subset hotpot_subset.jsonl --temperature 0 --out base.csv
```

`--setting distractor` (default) restricts each question's search to its own 10 paragraphs. `--setting pooled` searches the whole knowledge base, as in the paper's main experiments.

## Prompts

The prompts are text constants near the top of `hotpot_ask_eval.py`:

| Constant | Used for | Paper |
|---|---|---|
| `EXTRACT_PROMPT` | extracting the short answer from the long answer | Appendix A.6 |
| `YES_NO_LINE` | extra line added to the extraction prompt for yes/no questions | Appendix A.6 |
| `RETRY_LINE` | extra line for the second try when the first reply is "unknown" but the answer is not a refusal | Appendix A.6 |
| `JUDGE_PROMPT` | the judge's verdict against the gold answer | Appendix A.7 |

They are filled in and sent by `extract_short()` and `judge()` in the same file. Their wording is identical to the paper's appendix.

## Input files

**Answers CSV for `--rescore`** (UTF-8). One row per question, with these columns:

| Column | Content |
|---|---|
| `n` | running number (integer) |
| `id` | HotpotQA question id |
| `type` | `bridge` or `comparison` |
| `question` | question text |
| `answer` | the system's full answer (citation markers such as `[1]` are allowed) |
| `gold` | gold answer |
| `sf_retrieved` | gold supporting sentences found in the retrieved pairs, as `hits/total` (e.g. `2/2`; may be empty) |
| `sf_cited` | the same for the cited pairs (may be empty) |
| `evidence` | optional text of the retrieved pairs (kept in the output, not scored) |

The output CSV of any earlier run has these columns, so it can be re-scored directly.

**Question subset for phase 1** (`hotpot_subset.jsonl`). One JSON object per line with `id`, `type`, `question`, `answer`, `titles` (the question's paragraph titles) and `supporting_facts_qarag` (gold supporting sentences mapped to the knowledge base's sentence numbers). This file is written by `hotpot_index.py select` in the [indexer](../QARAG_Indexer/#building-the-hotpotqa-knowledge-bases-paper-experiments).

## Output files

`<out>.csv`, one row per question:

| Column | Content |
|---|---|
| `short_answer` | extracted short answer |
| `em`, `f1` | Exact Match and F1 on the short answer |
| `judge_correct`, `judge_reason` | verdict (`1`, `0`, or empty if the judge gave no valid JSON) and the judge's one-sentence reason |
| `cover` | gold answer contained in the short answer |
| `empty` | the system returned no text |
| `not_found` | the answer is only the refusal sentence |
| `partial` | a real answer that also says some information is missing |
| `correct_without_gold_citation` | correct, although no cited pair holds a gold supporting sentence |
| `verdict` | readable summary: correct, correct (judge), partly correct, wrong, not found or empty |
| `f1_calculation` | the F1 computation written out, for checking by hand |

`<out>_summary.md` contains:
- averages per question type and overall
- the settings and version tags
- the questions on which the judge and EM disagree (to check by hand)
- tokens and cost per purpose (extraction, judge)
- a block for every question

Costs are the amounts OpenRouter reports for each call. If OpenRouter reports no cost, they are estimated from the optional prices in `.env`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--rescore CSV` | | score saved answers only (phase 2) |
| `--db DIR` | | knowledge base folder (phase 1) |
| `--subset FILE` | `hotpot_subset.jsonl` | questions for phase 1 |
| `--n N` | `0` (all) | only the first N questions |
| `-k K` | `8` | QA pairs given to the answering model (phase 1) |
| `--setting` | `distractor` | `distractor` or `pooled` (phase 1) |
| `--answer-model` | from `.env` | answering model in phase 1; extraction model in phase 2 |
| `--temperature` | qarag default | answering temperature (phase 1) |
| `--max-tokens` | `4000` | answer budget including hidden reasoning (phase 1) |
| `--judge-model` | `openrouter/qwen/qwen3-235b-a22b-2507` | judge model |
| `--no-judge` | | skip the judge |
| `--no-extract` | | score the full answer, without extraction |
| `--dry-run` | | no model calls (offline test) |
| `--out` | `hotpot_ask_results.csv` | output CSV (the summary is written next to it) |

All extraction and judge calls use temperature 0.

## Files

```
QARAG_Answer_Eval/
├── hotpot_ask_eval.py      the evaluation script: prompts, extraction, judge, metrics, reports
├── qarag/
│   ├── __init__.py
│   ├── config.py           settings, read from .env (QARAG_* variables)
│   └── llm.py              OpenRouter chat calls through LiteLLM, retries, JSON parsing, usage and cost meter
├── examples/
│   └── sample_answers.csv  three example answers for the offline test
├── requirements.txt
└── .env.example
```

## Notes

- The HotpotQA scoring functions (`normalize_answer`, `f1_score`, `exact_match`) follow the official HotpotQA evaluation script, including its rule that yes/no answers must match exactly.
- The three example questions come from HotpotQA (Yang et al., 2018; CC BY-SA 4.0); the example answers were written for this test.
