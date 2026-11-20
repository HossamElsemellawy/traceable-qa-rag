# Traceable and Auditable RAG

Code for the paper **"Traceable and Auditable RAG: Generating QA Knowledge Base for Answering Multi-Hop Questions"** by Hossam Elsemellawy, Nashwa Elbendary, Walid Mohamed Rabie Abdelmoez and Mohamed Attia (submitted to *Data Mining and Knowledge Discovery*).

The framework restructures documents into atomic question–answer (QA) pairs, each linked to the numbered source sentences it comes from. A direct-first planner answers from a single search when it can, and escalates to a code-constrained ReAct loop only when another hop is needed. Every answer cites its supporting pairs and sentences, and every run writes a planning trace of searches, model calls, tokens and cost. No model training is needed.

## Programs

Each program has its own folder and README with installation, commands, inputs and outputs.

| Stage | Program | What it does | Paper |
|---|---|---|---|
| Offline | [**QARAG_Indexer**](QARAG_Indexer/) | Builds the QA knowledge base: context-aware chunking, atomic QA generation, coverage pass, merging, SQLite + Chroma | Section 3.1, Appendix A.1–A.3 |
| Offline | [**QARAG_Indexer/hotpot_index.py**](QARAG_Indexer/#building-the-hotpotqa-knowledge-bases-paper-experiments) | Samples the 500 HotpotQA questions and indexes their paragraphs (KB-GLM, KB-Qwen) | Sections 4.1–4.2 |
| Online | [**QARAG_Planner**](QARAG_Planner/) | Direct-first gate and ReAct loop, cited answers, planning trace database; runs the HotpotQA experiments | Section 3.2, Appendix A.4–A.5, C |
| Evaluation | [**QARAG_Answer_Eval**](QARAG_Answer_Eval/) | Short-answer extraction, LLM judge, EM/F1, supporting-fact recall | Section 3.3, 4.5, Appendix A.6–A.7 |
| Analysis | *Analysis scripts* (coming) | Judge panel majority, agreement, ensemble, tables and figures | Section 5 |

## Requirements

- Python 3.10 or later
- An [OpenRouter](https://openrouter.ai) API key; all models are called through OpenRouter. Put the key in a `.env` file in the program's folder (each folder has a `.env.example`). Never commit `.env`.
- The HotpotQA distractor validation set ([hotpotqa.github.io](https://hotpotqa.github.io)) for the experiments.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`; the full reference will be added on publication).

## License

Released under the MIT License (see `LICENSE`). HotpotQA is distributed by its authors under CC BY-SA 4.0.
