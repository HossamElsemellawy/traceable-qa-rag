# Indexer: building the QA knowledge base (`qarag`)

This program implements the **offline stage** of the paper (Section 3.1, prompts in Appendix A.1–A.3). It turns documents into **atomic question–answer (QA) pairs**. Each pair cites the numbered source sentences it comes from. The pairs are stored in **SQLite**, the master record, and indexed for search in a **Chroma** vector store.

It also contains `hotpot_index.py`, which builds the HotpotQA knowledge bases used in the experiments (Sections 4.1–4.2; see [Building the HotpotQA knowledge bases](#building-the-hotpotqa-knowledge-bases-paper-experiments)).

## How a document becomes QA pairs

```
document ─► sentences, numbered, each tagged with its section ("Career > Film") and page
         ─► chunks: core sentences + read-only context BEFORE (end of the previous chunk)
                    and AFTER (start of the next chunk)
         ─► 1. GENERATION  per chunk: atomic pairs, each citing its sentence numbers
         ─► 2. COVERAGE    per chunk: content sentences that no pair cites are sent back once
         ─► 3. MERGING     per document: candidates with similar questions (embedding
                           similarity) are shown to the model, which merges true duplicates
                           into one pair citing every place the fact appears
         ─► final pairs: question, answer, ALL supporting sentences, ALL sections, pages
         ─► embeddings ─► Chroma
```

- Merging never crosses documents, so documents can later carry different access rules.
- Tables (web pages including Wikipedia infoboxes, Word, Markdown) become one sentence per row. Figure captions are kept, so facts that live only in tables are covered too.
- Every pair is stored with `allowed_roles = ["public"]`. Access control is not enforced in this release.

| Step | Paper | Prompt |
|---|---|---|
| Sentence numbering and context-aware chunking | Section 3.1 | |
| QA generation | Section 3.1, Appendix A.1 | `qarag/prompts.py`: `_GENERATION_TEMPLATE`, `STYLE_RULES`, `build_user_prompt()` |
| Coverage pass | Section 3.1, Appendix A.2 | `qarag/prompts.py`: `coverage_message()` |
| Merging within a document | Section 3.1, Appendix A.3 | `qarag/prompts.py`: `_MERGE_TEMPLATE`, `merge_user_prompt()` |

The prompt version (`qa-gen-v4`) is stored with every document, so the database records which prompt produced which pairs.

## Installation

Python 3.10 or later.

```bash
cd QARAG_Indexer
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in two values:

```
OPENROUTER_API_KEY=sk-or-...
QARAG_USER_AGENT=qarag-research/0.1 (research; mailto:you@example.com)
```

- **`OPENROUTER_API_KEY`:** all models (generation, merging, embeddings, answering) are called through [OpenRouter](https://openrouter.ai).
- **`QARAG_USER_AGENT`:** Wikipedia and other Wikimedia sites reject downloads (HTTP 403) unless the User-Agent gives a contact ([Wikimedia policy](https://meta.wikimedia.org/wiki/User-Agent_policy)). Put your own address here.

Never commit your `.env` file; it is excluded by `.gitignore`.

## Quick test without API calls

`--dry-run` uses a fake generator and fake embeddings: no API calls, no cost. Use a separate database folder:

```bash
python -m qarag index https://en.wikipedia.org/wiki/Scott_Derrickson --dry-run --db data/test -m overwrite -y
python -m qarag stats  --db data/test
python -m qarag search "When was Scott Derrickson born?" --db data/test
```

## Indexing documents

```bash
python -m qarag index https://en.wikipedia.org/wiki/Scott_Derrickson -m overwrite -y   # start a new database
python -m qarag index https://en.wikipedia.org/wiki/Ed_Wood  articles/report.pdf         # append
python -m qarag index articles/                                                          # a folder
python -m qarag index --from-list sources.txt                                            # one URL or path per line
```

Sources can be web pages, online or local PDF, Word (`.docx`), Markdown, text and HTML files.

Append mode (the default) is safe to repeat:
- unchanged sources are **skipped**, at no cost
- changed sources are **re-indexed**
- interrupted ones are **resumed**: only unfinished work is done

The database folder defaults to `data/index`; choose another with `--db`.

### Options of `index`

| Option | Default | Meaning |
|---|---|---|
| `-m append` / `-m overwrite` | `append` | add to the database / start from scratch (`-y` skips the confirmation) |
| `-s short\|sentence\|detailed` | `sentence` | answer style (one style per database) |
| `--threshold` | `0.82` | question similarity (cosine) that triggers a merge check |
| `--no-merge`, `--no-coverage` | | switch off merging / the coverage pass (for ablations) |
| `--variant qa\|q\|dual` | `qa` | what is embedded: question + answer, question only, or both separately |
| `--gen-model` | `.env` | generator model (also used for the coverage pass and, by default, merging) |
| `--embed-model` | `openai/text-embedding-3-large` | embedding model |
| `--chunk-tokens` | `1500` | size of each chunk's core |
| `--overlap-tokens` | `200` | read-only context from the previous chunk (0 = off) |
| `--lookahead-tokens` | `300` | read-only context from the next chunk (0 = off) |
| `--workers` | `4` | model calls in parallel |
| `--force` | | re-index unchanged sources |
| `--db` | `data/index` | database folder |
| `--dry-run` | | offline test with fake model calls |

Every option can also be set in `.env` (see `.env.example` and `qarag/config.py`).

### Answer styles

Each database holds one style. To compare styles, index the same sources into separate folders.

| Style | Pairs | Answer |
|---|---|---|
| `short` | many, narrow | only the exact fact: `July 16, 1966` |
| `sentence` (default, used in the paper) | one per fact | one complete sentence naming the subject |
| `detailed` | one per topic | 1–3 sentences with directly related details |

## Inspecting and using the knowledge base

| Command | Shows |
|---|---|
| `python -m qarag stats` | counts: candidates → pairs, merged pairs, multi-section pairs, coverage |
| `python -m qarag docs [--meta]` | per document: candidates → pairs, coverage %, and metadata with `--meta` |
| `python -m qarag search "..." [-k 5]` | the closest pairs, with sections, pages and supporting sentences |
| `python -m qarag gaps [--limit 20]` | content sentences that no pair cites |
| `python -m qarag export pairs.csv` | every pair with its evidence, for review in Excel (`.jsonl` for code) |
| `python -m qarag ask "..." [-k 8] [-e]` | an answer written only from the retrieved pairs, with citations `[n]` and sources |
| `python -m qarag report [-o report.md]` | Markdown report: runs, tokens and cost, document quality, sample pairs |

Add `--db <folder>` to any command, and `--help` for all options.

**`ask`** is the single-search answer of the paper's baseline. It does the following:
1. Retrieves the top-k pairs (default 8).
2. Gives them with their source sentences to the answering model (`--answer-model` or `QARAG_ANSWER_MODEL`).
3. Prints one answer with citations.

It says so when the documents do not contain the answer. `-e` also prints the evidence pairs. The planner, which adds the ReAct loop, is a separate program.

**Metadata filters.** `search` and `ask` accept the following filters:
- `--type "wikipedia article"`
- `--lang en`
- `--since 2020` and `--until 2026`

They use the document metadata read during indexing (type, language, publication year). A document without that field is excluded by the filter.

**Run log and cost.** Every `index`, `consolidate`, `reembed` and `ask` run is recorded with:
- its settings and duration
- model calls and tokens
- cost per purpose (`generate`, `coverage`, `embed_questions`, `merge`, `embed_pairs`, `answer`) and per document

The cost is the amount OpenRouter reports. If none is reported, it is estimated from the optional prices in `.env`.

## Tuning without paying for generation again

| What | Command |
|---|---|
| Merge more or less aggressively | `python -m qarag consolidate --threshold 0.78` (lower = more checks) |
| Change the embedding variant or model | `python -m qarag reembed --variant dual` / `--embed-model <id>` |

Settings that change generation (style, prompt, model, chunk size, coverage pass) need a new index run, ideally into a new `--db` folder so results stay comparable.

## Building the HotpotQA knowledge bases (paper experiments)

`hotpot_index.py` prepares HotpotQA for the experiments without changing `qarag`. It only reads the dataset and calls `python -m qarag index`. It needs the `datasets` package (included in `requirements.txt`) to download HotpotQA from Hugging Face. Alternatively, pass the official file with `--from-file hotpot_dev_distractor_v1.json`.

**Step 1: choose the questions and write their paragraphs.**

```bash
python hotpot_index.py select --n 500 --seed 42 --proportional
```

This step:
- Takes a seeded random sample of HotpotQA validation (distractor) questions. `--proportional` keeps the dataset's mix (about 80% bridge, 20% comparison), as in the paper. Without it, both types get equal numbers.
- Writes every distinct paragraph to `hotpot_paragraphs/` as one Markdown file, titled with its exact HotpotQA title, with one sentence per line.
- Writes `hotpot_subset.jsonl`, one question per line, used for scoring only. **Questions and answers are never indexed.**
- Writes `hotpot_subset.meta.json`, recording how the subset was made.
- Maps HotpotQA's gold supporting sentences to the indexer's own sentence numbers. It runs the indexer's own reading and sentence splitting on every paragraph, and stores the result as `supporting_facts_qarag` in the subset file and in `hotpot_paragraphs/hotpot_sentence_map.json`. This is what makes the paper's supporting-fact recall exact.

**Step 2: index the paragraphs.**

```bash
python hotpot_index.py index --db data/hotpot --fresh --limit 20     # pilot: the first 20 questions
python hotpot_index.py index --db data/hotpot --start 21             # the rest
```

Each question's 10 paragraphs are indexed as one `qarag index` run, in append mode:
- **Shared paragraphs:** a paragraph already indexed for an earlier question is skipped at no cost.
- **Resuming:** the run can be stopped and resumed with `--start N`.
- **`--fresh`:** empties the database on the first call. Use it only once, at the start.
- **Cost:** each question is a separate run, so `python -m qarag report` shows its cost. A per-question summary is appended to `hotpot_index_log.csv`.

All questions' paragraphs go into one database. That is the paper's **open-pool** setting: every search runs over about 5,000 paragraphs. The standard distractor setting is applied only at query time, by restricting a search to the question's 10 titles (field `titles` in the subset file).

**The two knowledge bases of the paper.** The generator model is the only difference. Set it in `.env` before step 2 and use a separate database folder for each:

| Knowledge base | `QARAG_GEN_MODEL` | Folder (example) |
|---|---|---|
| KB-GLM (first cycle) | `openrouter/z-ai/glm-5.3-flash` | `data/hotpot_glm` |
| KB-Qwen (second cycle) | `openrouter/qwen/qwen3-vl-235b-a22b-instruct` | `data/hotpot_qwen` |

The other indexing settings are the defaults, which are the values in the paper's Table B1:
- answer style `sentence`
- chunk core of 1,500 tokens, with 200 / 300 read-only tokens before and after
- merge threshold 0.82
- embedding model `openai/text-embedding-3-large`, variant `qa`
- generator temperature 0.1

## What the database contains

A knowledge base is a folder with `qa.sqlite` (the master record) and `chroma/` (the vector index).

| SQLite table | Content |
|---|---|
| `documents` | one row per source: title, type, status, counts, coverage, generator model, prompt version, metadata (authors, dates, language, DOI ...) |
| `sentences` | the exact source text: one row per numbered sentence, with section path and page |
| `qa_pairs` | the final pairs: question, answer, all supporting sentence numbers, sections, pages, merged candidates, access fields |
| `chunks`, `qa_candidates` | working tables of the indexer (chunk status, candidate pairs before merging) |
| `runs`, `run_documents` | the run log: settings, duration, calls, tokens and cost, per run and per document |
| `meta` | schema version, embedding model and variant, answer style |

The Chroma collection `qa_pairs` holds one vector per pair, or two with `--variant dual`, linked to SQLite by `pair_id`.

## Code map

| File | Role |
|---|---|
| `qarag/sources.py` | reads a URL or local file into text; headings become sections; document metadata |
| `qarag/chunker.py` | numbered sentences with section and page; chunks with read-only context |
| `qarag/prompts.py` | generation, coverage and merge prompts; the three answer styles |
| `qarag/generator.py` | per-chunk generation, validation and coverage pass |
| `qarag/consolidate.py` | document-level merging of repeated facts |
| `qarag/llm.py` | OpenRouter calls (chat through LiteLLM, embeddings through the openai client), retries, usage and cost meter, dry-run mocks |
| `qarag/store.py` | SQLite database |
| `qarag/vector_store.py` | Chroma index, embedding variants, search and metadata filters |
| `qarag/indexer.py` | the pipeline: append / overwrite / resume / change detection |
| `qarag/answer.py` | retrieval and grounded answer for `ask` |
| `qarag/report.py` | the Markdown report |
| `qarag/hotpot.py` | HotpotQA subset helper with an `--exclude` option for non-overlapping subsets (library module; the HotpotQA workflow above uses `hotpot_index.py`) |
| `qarag/cli.py` | the commands (`python -m qarag ...`) |
| `qarag/config.py` | all settings and their defaults |
| `hotpot_index.py` | HotpotQA question selection and indexing (paper experiments) |

## Scoring answers

The answer evaluation (short-answer extraction, LLM judge, EM/F1) is in [`../QARAG_Answer_Eval`](../QARAG_Answer_Eval/). Its single-search baseline run (`hotpot_ask_eval.py --db ...`) uses this package. To use it, copy `hotpot_ask_eval.py` into this folder and run it here.

## Notes

- HotpotQA (Yang et al., 2018) is distributed by its authors under CC BY-SA 4.0. The paragraphs written by `select` are taken from it unchanged.
- `--dry-run` databases are marked with the embedding model `mock-embedding`, so fake vectors are never mixed into a real index.
