"""
qarag - Stage 0 of the Governed RAG framework.

This package turns articles (online URLs or local documents) into
atomic question-and-answer (Q&A) pairs with a language model (the "LLM-G" role in
the framework), stores the pairs in a SQLite database, and builds a Chroma
vector index over them so they can be searched.

Governance (review gate, role filtering) is deliberately NOT implemented yet.
The database already keeps an `allowed_roles` column defaulting to
["public"] so governance can be added later without changing the data.

Module map (in the order data flows through them):
    sources.py      read an article from a URL or a local file -> plain text
    chunker.py      numbered sentences (with section and page), grouped into chunks
    prompts.py      the instructions given to the Q&A generator model
    generator.py    call the model for each chunk, validate, coverage pass
    consolidate.py  merge facts repeated across the whole document
    llm.py          low-level calls to OpenRouter (chat + embeddings)
    store.py        SQLite database: documents, sentences, chunks, Q&A pairs
    vector_store.py Chroma vector index over the Q&A pairs
    indexer.py      orchestrates everything (append or overwrite mode)
    answer.py       retrieval + grounded answer synthesis (the `ask` command)
    report.py       Markdown report: runs, tokens and cost, documents, sample pairs
    cli.py          the command-line interface (python -m qarag ...)
"""

__version__ = "0.4.0"   # 0.4: document metadata (P4a), run log + report (P4b)
