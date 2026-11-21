# Planner: direct-first ReAct answering with citations and traces (`qplan`)

This program implements the **online stage** of the paper (Section 3.2, prompts in Appendix A.4–A.5, trace database in Appendix C). It answers a question from a QA knowledge base built by the [indexer](../QARAG_Indexer/), which it opens **read-only**.

```
question
  ─► step 0 (direct-first): one search with the whole question (k = 8)
                            → the writer answers from those pairs
        complete answer?  ── yes ──► done (no ReAct)
              │ no
              ▼
  ─► ReAct loop (at most 6 steps): Thought → Action → Observation
        search_kb("What is Scott Derrickson's nationality?")  → [E1]
        search_kb("What was Ed Wood's nationality?")          → [E2]
        finish
  ─► writer: "Yes, both were American [E1][E2]."  + Sources list
  ─► planning trace: every step, search, retrieved pair, model call, tokens and cost
```

**The step-0 gate.** An answer is complete only when all of these hold:
- its status is *answered*
- it cites at least one pair
- it contains none of the gap phrases listed in Appendix B ("not provided", "does not mention", "cannot be determined", ...)

Otherwise the ReAct loop starts, with the step-0 pairs already collected.

**Every answer is traceable.**
- Each claim cites `[E#]` labels.
- The Sources list shows, for every cited pair:
  - the document title and section
  - the exact supporting sentences
  - the source URL
  - the `pair_id`
- Citations that point to evidence the planner never received are detected and reported.

| Part | Paper | Code |
|---|---|---|
| Direct-first gate | Section 3.2, Appendix B | `qplan/react.py`, `GAP_PHRASES` in `qplan/synth.py` |
| ReAct searcher | Section 3.2, Appendix A.4 | `qplan/react.py`, `qplan/prompts.py` (`react-v4`) |
| Answer writer | Section 3.2, Appendix A.5 | `qplan/synth.py`, `qplan/prompts.py` (`synth-v5`) |
| Planning trace database | Section 3.2, Appendix C | `qplan/runlog.py` |

Prompt versions are stored with every run, so results from different prompt versions are never mixed up.

## Installation

Python 3.10 or later.

> **Use the same `chromadb` version as the indexer.** Chroma's on-disk format depends on its version, so the planner must open the knowledge base with the version that wrote it. The simplest way is one virtual environment for the indexer and the planner.

One environment for the whole repository, from the repository root:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r QARAG_Indexer/requirements.txt -r QARAG_Planner/requirements.txt
```

To use a separate environment for the planner, install `QARAG_Planner/requirements.txt`, then install the indexer's Chroma version. Check it in the indexer's environment with `pip show chromadb`, then run `pip install chromadb==<that version>`.

Copy `QARAG_Planner/.env.example` to `QARAG_Planner/.env` and put your key in it:

```
OPENROUTER_API_KEY=sk-or-...
```

The planner reads `.env` from its own folder, so you can run it from anywhere. All models are called through [OpenRouter](https://openrouter.ai). Never commit your `.env` file; it is excluded by `.gitignore`.

## Quick test without API calls

Build a small test knowledge base with the indexer's offline mode, then ask the planner with fake model calls:

```bash
cd QARAG_Indexer
python -m qarag index https://en.wikipedia.org/wiki/Scott_Derrickson --dry-run --db data/test -m overwrite -y
cd ../QARAG_Planner
python -m qplan info --db ../QARAG_Indexer/data/test
python -m qplan ask "When was Scott Derrickson born?" --db ../QARAG_Indexer/data/test --dry-run --memory off
```

With `--dry-run`, every model call is fake (the answers read "(dry-run) ..."). On a real knowledge base, `--dry-run` still makes the cheap query-embedding call, because the query vector must come from the knowledge base's own embedding model.

## Answering questions

```bash
python -m qplan info --db ../QARAG_Indexer/data/hotpot_glm                     # check and summarise a knowledge base
python -m qplan ask "Were Scott Derrickson and Ed Wood of the same nationality?" --db ../QARAG_Indexer/data/hotpot_glm
python -m qplan batch questions.txt --db ../QARAG_Indexer/data/hotpot_glm -q   # one question per line, optional " || expected answer"
```

`ask` prints every step, the answer, the Sources list, the route (answered at step 0 or escalated) and the cost. With `--json`, it prints one JSON object instead; `hotpot_react_answers.py` uses this. `--only-titles T1 T2 ...` or `--only-titles-file` restricts every search to the named documents; this is the HotpotQA distractor setting.

### Options (`ask` and `batch`)

| Option | Default | Paper (Table B1) | Meaning |
|---|---|---|---|
| `--db` | `QPLAN_DB` | | knowledge base folder (`qa.sqlite` + `chroma/`) |
| `--direct-first on\|off` | `on` | on | step-0 direct answer before ReAct (`off` = ReAct only, for ablations) |
| `--direct-k` | `8` | 8 | pairs retrieved at step 0 |
| `-k`, `--top-k` | `5` | 8 | pairs retrieved per ReAct search |
| `--min-score` | `0.35` | 0.35 | cosine score needed for a search to count as a match |
| `--max-steps` | `6` | 6 | maximum ReAct steps |
| `--memory on\|off\|free` | `on` | off | whether the agent may use `recall` (a model's own knowledge, labelled `[M#]` and marked *not verified*) |
| `--react-model` | `.env` | the planner model | the searcher (ReAct agent) |
| `--synth-model` | `.env` | the planner model | the answer writer |
| `--memory-model` | `.env` | | the model used by `recall` |
| `--model` | | | set all three model slots at once |
| `--runs-dir` | `runs/` | | where the run log and traces go |
| `--dry-run` | | | fake model calls (offline, free) |
| `-q` | | | do not print every step |

Temperatures and token budgets are set in `.env`. The defaults are the paper's values: searcher 0 / 4,096 tokens, writer 0 / 4,000 tokens. `QPLAN_REASONING` (`default | off | low | medium | high`) controls hidden reasoning for models that support it.

**Memory policy.** `on` allows `recall` only after the knowledge base has been searched. `off` uses the knowledge base only. `free` allows `recall` at any time. All experiments in the paper use `off`.

## Planning traces and reports

| What | Where / command |
|---|---|
| Run log of every run | `runs/qplan_runs.sqlite` |
| Readable trace of one run | `runs/traces/<run_id>.md`, or `python -m qplan show last` (or a run id) |
| Recent runs with steps, tokens and cost | `python -m qplan report --last 20` (add `--csv report.csv`) |
| Batch results | `runs/batch-<time>.csv` (question, expected, short answer, basis, cost, trace path) |
| All runs for analysis | `python -m qplan export --out runs.jsonl --no-dry` |

The run log is the paper's planning trace database (Appendix C):

| Table | One row per | Main fields |
|---|---|---|
| `runs` | question | run id and times; program and prompt versions; every setting; knowledge base fingerprint and embedding model; route; status, stop reason, steps, tokens and cost |
| `steps` | ReAct step | thought, action and its input, observation, outcome, best score, evidence found |
| `retrievals` | search | query, top-k, best score, every hit with its score, embedding tokens and cost |
| `evidence` | pair received | `[E#]` label, `pair_id`, score, question, answer, title, source, which search found it, whether it was cited |
| `llm_calls` | model call | role, model, full prompt and raw reply, tokens, cost and its source, duration |

Costs are the amounts OpenRouter reports. If a call reports none, the cost is estimated from an optional `prices.json`; copy `prices.example.json` and enter the prices in USD per 1M tokens. In `report`, a `*` marks runs where some calls had no cost.

## Running the HotpotQA experiments (paper)

`hotpot_react_answers.py` runs the planner on every question of the HotpotQA subset. It writes the answers in the CSV format that [`QARAG_Answer_Eval`](../QARAG_Answer_Eval/) scores with `--rescore`. It calls `python -m qplan ask ... --json` once per question, with:
- the question's 10 paragraph titles in the distractor setting
- no filter in the open-pool setting

It also converts the planner's conventions to the evaluation's:
- `[E7]` becomes `[7]`
- a *not found* status becomes the refusal sentence

Run it from `QARAG_Answer_Eval/`, so that `--score` finds `hotpot_ask_eval.py`. The paper's setting is the open pool (`--setting full`) with memory off:

```bash
cd QARAG_Answer_Eval
python ../QARAG_Planner/hotpot_react_answers.py --qplan ../QARAG_Planner \
    --db ../QARAG_Indexer/data/hotpot_glm --subset ../QARAG_Indexer/hotpot_subset.jsonl \
    --setting full --memory off --direct-first on --direct-k 8 -k 8 --max-steps 6 --min-score 0.35 \
    --react-model openrouter/z-ai/glm-5.3-flash --synth-model openrouter/z-ai/glm-5.3-flash \
    --out answers_glm.csv
```

| Option | Meaning |
|---|---|
| `--setting full` | search the whole knowledge base (the paper's open pool); `distractor` (default) restricts each question to its own 10 paragraphs |
| `--n N` | only the first N questions (pilot) |
| `--resume` | keep the questions already answered in `--out` and answer the rest |
| `--retry-waits 30,60,120,240` | wait and re-ask a question after a temporary API error (rate limit, overload, timeout) |
| `--score` | afterwards run `hotpot_ask_eval.py --rescore` on the answers |
| `--qplan-python` | Python of a separate planner environment (default: `<qplan>/.venv`, else the current Python) |
| `--dry-run` | fake model calls (offline test) |

The output CSV is saved after every question. Besides the columns the evaluation reads, it keeps the planner's own data: original answer, status, steps, searches, route (`escalated`), cost and `run_id`. The `run_id` links each answer to its full trace.

**The five planners of the paper.** Run the command once per planner, with the same model for `--react-model` and `--synth-model`:

| Planner | Model id |
|---|---|
| GLM | `openrouter/z-ai/glm-5.3-flash` |
| DeepSeek | `openrouter/deepseek/deepseek-v4.1-flash` |
| Qwen3-VL | `openrouter/qwen/qwen3-vl-235b-a22b-instruct` |
| Gemini | `openrouter/google/gemini-3.8-flash` |
| Claude Haiku | `openrouter/anthropic/claude-haiku-5.5` |

Then score each answers file with the five-judge panel; see [QARAG_Answer_Eval](../QARAG_Answer_Eval/#reproducing-the-papers-five-judge-panel).

## Files

| File | Role |
|---|---|
| `qplan/cli.py` | commands `info`, `ask`, `batch`, `report`, `show`, `export` |
| `qplan/config.py` | all settings (defaults, `.env`, command line) |
| `qplan/react.py` | direct-first gate, ReAct engine, Markdown trace |
| `qplan/synth.py` | answer writer, citation checking, gap phrases, Sources list |
| `qplan/prompts.py` | all prompts, with version labels |
| `qplan/evidence.py` | evidence collector: `[E#]`/`[M#]` labels, de-duplication by `pair_id`, which search found what |
| `qplan/db.py` | read-only access to the knowledge base, contract checks, the shared `retrieve()` |
| `qplan/embed.py` | query embedding with the knowledge base's own model |
| `qplan/llm.py` | every chat call: LiteLLM, reasoning settings, JSON parsing, tokens, cost |
| `qplan/runlog.py` | the planning trace database |
| `qplan/fakes.py` | fake model replies for `--dry-run` |
| `hotpot_react_answers.py` | runs the planner on the HotpotQA subset for scoring |
| `prices.example.json` | optional prices for cost estimates |

## Notes

- **Access control.** Every search already carries the user's roles (`roles = ["public"]`), so a role filter can be added inside `retrieve()` in one place. It is not enforced in this release.
- **Read-only.** The planner never writes to the knowledge base. It records the knowledge base's fingerprint with every run.
