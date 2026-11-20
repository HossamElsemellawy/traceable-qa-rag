"""
qplan - the query-phase program of the Governed RAG PhD project.

qplan reads a Q&A database built by the indexer (qarag) READ-ONLY and answers
user questions with a ReAct engine:

    question -> [ Thought -> Action -> Observation ] x N -> LLM-S final answer

Actions available in version 1:
    search_kb  : search the Q&A database (the shared retrieve() function)
    recall     : ask an LLM to answer from its own trained knowledge
                 ("parametric memory"); used when the database has no match
    finish     : stop the loop and hand the collected evidence to LLM-S

Every run is logged (steps, LLM calls, tokens, cost) in qplan's own storage.
"""

# Version of the planner program. Stored in every run log row, so results
# from different program versions can be told apart later.
__version__ = "0.1.6"
