"""
prompts.py - every prompt text of qplan, with a version label.

The version label of each prompt is stored in the run log, so when a prompt
is changed later (e.g. "react-v2"), results of old and new prompts can be
compared and never mixed up. RULE: when you change a prompt's wording,
change its version label too.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. ReAct agent (the planner that reasons and chooses actions)
# ---------------------------------------------------------------------------
REACT_PROMPT_VERSION = "react-v4"
# react-v4 (qplan 0.1.5) changes against react-v3:
#   * every pair found is shown with its SOURCE TEXT (the document sentences it
#     was extracted from); a fact stated there counts as found. HotpotQA run
#     react30: the death date of Robert Graves was in the source sentence but
#     not in the pair's answer, and the agent searched 3 times in vain;
#   * an explicit second-hop rule for bridge questions: search the entity found
#     in hop 1 before finishing. react30: 6 of 16 bridge questions stopped
#     after ONE search (e.g. found "voiced by Ashley Tisdale", never searched her);
#   * the agent may start after an automatic step 0 (direct-first): it is told
#     what step 0 found and what the first answer attempt lacked.
# react-v3 (qplan 0.1.2) changes against react-v2:
#   * finish must list the facts that are still missing ("missing": [...]).
#     In a real 0.1.1 run the agent said in its own thought that the found
#     pairs were irrelevant, was told it may use recall, and finished anyway.
#     The list lets the CODE see the declared gap and enforce the memory rule.
# react-v2 (qplan 0.1.1) changes against react-v1:
#   * the example pair is about an INVENTED entity. react-v1 used a real test
#     entity (Scott Derrickson) and the agent took the example as a
#     description of the database contents, then finished without searching;
#   * the agent is told it cannot know the database contents before searching;
#   * with memory "off" the recall action is not described at all;
#   * the method says how to use recall when memory is "on".

# The recall action as described to the agent, per memory policy.
# Memory "off" -> no recall text at all: the agent never learns it exists.
_RECALL_ACTION = {
    "on": ("- recall: input = one short question about one fact. A language model answers from its own "
           "trained knowledge; the answer is labelled [M1], [M2], ... and is NOT verified by any document. "
           "Use recall ONLY for a fact the knowledge base does not contain: search_kb for that fact first; "
           "if the results are 'no match' or none of the pairs states that fact, you may recall it.\n"),
    "free": ("- recall: input = one short question about one fact. A language model answers from its own "
             "trained knowledge; the answer is labelled [M1], [M2], ... and is NOT verified by any document. "
             "You may use it at any time, but prefer search_kb.\n"),
    "off": "",
}
_ACTION_NAMES = {"on": "search_kb | recall | finish", "free": "search_kb | recall | finish",
                 "off": "search_kb | finish"}
_RECALL_METHOD = {
    "on": ("7. If the knowledge base does not contain a fact you need (after a search and one rephrased search), use recall for "
           "that fact BEFORE you finish. A finish that lists missing facts while recall has not been tried is refused once.\n"),
    "free": "7. Use recall for facts the knowledge base does not contain.\n",
    "off": ("7. Only the knowledge base may be used. If it does not contain a fact, list that fact in \"missing\" when you finish. "
            "A finish that lists missing facts is refused once: search each missing fact by name before finishing.\n"),
}

REACT_SYSTEM = """You are a research agent. You answer a user's question by collecting evidence step by step, then you hand the evidence to a writer who produces the final answer.

THE KNOWLEDGE BASE
A database of short, atomic question-answer pairs extracted from source documents. Each pair states ONE fact. The format looks like this invented example (it is NOT from the knowledge base):
  Q: Who designed the Halvorsen Bridge?  A: The Halvorsen Bridge was designed by Ida Brenner.  text: "The Halvorsen Bridge (1932-1934) was designed by Ida Brenner."
"text" is the source text the pair was taken from. It often states more than the pair's answer (here: the years); a fact stated in the text counts as found.
You cannot know what the knowledge base contains until you search it.
Search compares your search text with the stored questions by meaning, so the best search text is a short, specific question about one fact that names the entity in full.

YOUR REPLY FORMAT
Every reply is exactly ONE JSON object and nothing else:
{{"thought": "<your reasoning: what you know so far, what is missing, what to do next>", "action": "<{action_names}>", "input": "<text for the action>", "missing": [<facts still missing; required with finish>]}}

ACTIONS
- search_kb: input = one short question about one fact. Returns matching pairs labelled [E1], [E2], ... with a similarity score, or "no match".
{recall_action}- finish: input = one sentence saying what the evidence shows. "missing" = a list of every fact the question needs that the evidence does not contain, each as a short phrase (e.g. ["the opening year of the Halvorsen Bridge"]); [] if nothing is missing. finish is only accepted after at least one search_kb. Choose it when the evidence answers every part of the question, or when more steps will not help.

METHOD
1. Split the question into single facts. Comparison questions ("same nationality", "which came first") need one search per entity. Bridge questions ("the director of the film X") need the first fact before the second: use the name you found in your next search.
   SECOND HOP: when a search reveals the entity the question is really about (a person, a band, a film, a place), your next search must name that entity and ask for the fact the question wants (e.g. found "the bridge was designed by Ida Brenner" -> search "Where was Ida Brenner born?"). Do not finish a bridge question before you have searched the entity you found.
2. Always search the knowledge base before concluding anything. Never claim a search result you have not received.
3. If a search finds no match, or the pairs found do not state the fact you need, rephrase it once (full name, other words) before treating the fact as missing.
4. Never repeat a search you already made.
5. Do not answer the question yourself in "thought"; collect evidence, then finish. Before finish, check that the evidence (pairs and their text) states the exact fact the question asks for, not only the entity.
6. You have at most {max_steps} steps in total. Finish before you run out.
{recall_method}"""

REACT_FIRST_USER = "Question: {question}\n\nThis is step 1 of at most {max_steps}. Reply with one JSON object."

# direct-first (qplan 0.1.5): the agent starts after the automatic step 0.
REACT_FIRST_USER_AFTER_DIRECT = """Question: {question}

Step 0 was done automatically: the whole question was searched in the knowledge base, and a writer tried to answer from the pairs found. The attempt was not complete (status: {status}).
Result of step 0:
{observation}

The writer's attempt: {attempt}

Collect the evidence that is still missing (usually the second hop: search the entity found above for the fact the question asks). Do not repeat the step-0 search.
This is step 1 of at most {max_steps}. Reply with one JSON object."""

REACT_OBSERVATION = "Observation (result of step {step}):\n{observation}\n\nSteps left: {steps_left}. Reply with one JSON object."


def action_names(memory_policy: str) -> str:
    """The actions the agent may name in this run, e.g. 'search_kb | finish'."""
    return _ACTION_NAMES[memory_policy]


def react_system(memory_policy: str, max_steps: int) -> str:
    """Build the agent's system prompt for this run's settings (memory on/off/free)."""
    return REACT_SYSTEM.format(action_names=_ACTION_NAMES[memory_policy],
                               recall_action=_RECALL_ACTION[memory_policy],
                               recall_method=_RECALL_METHOD[memory_policy],
                               max_steps=max_steps).rstrip()


# ---------------------------------------------------------------------------
# 2. Memory model (answers from its trained, "parametric" knowledge)
# ---------------------------------------------------------------------------
MEMORY_PROMPT_VERSION = "memory-v1"

MEMORY_SYSTEM = """Answer the question from your own general knowledge. You have no documents and no internet access.
Be brief and factual: one or two sentences.
If you are not reasonably sure, do not guess: say that you do not know.
Reply with exactly one JSON object and nothing else:
{"answer": "<your answer, or 'I do not know'>", "confidence": "<high | medium | low | unknown>"}"""

MEMORY_USER = "Question: {question}"


# ---------------------------------------------------------------------------
# 3. LLM-S synthesizer (the final, cited answer)
# ---------------------------------------------------------------------------
SYNTH_PROMPT_VERSION = "synth-v5"
# synth-v5 (qplan 0.1.6): LLM-S uses the Path 1 answer prompt of qarag 0.4
# (qarag/answer.py, ANSWER_SYSTEM_PROMPT) WORD FOR WORD, with the same
# evidence layout (numbered items [1]..[n], Q, A, "Source:" sentences) and
# the same plain-text reply with the fixed refusal sentence.
# Why: on HotpotQA questions 1-30 the Path 1 baseline (same model, 30B) was
# right on 28/30, the ReAct planner with the JSON prompts synth-v3/v4 on
# 22-23/30, mostly by refusing. With the SAME answer step in both systems,
# any difference measured is due to retrieval/planning only (apples to apples).
# Code-side differences to qarag (necessary, not in the prompt):
#   * the numbers [n] are translated back to qplan's labels [E#] after the reply;
#   * model-memory items get one extra rule, ONLY in runs that have M-items
#     (memory off -> the prompt is identical to qarag's);
#   * the agent's "missing facts" list is NOT given to LLM-S any more (it made
#     LLM-S refuse: "the facts the research agent could not find explicitly
#     note the absence of this information", run react30_1 #14).
# Older versions: synth-v4 (0.1.5, JSON, never run on real data), synth-v3
# (0.1.2-0.1.4, JSON + missing list), synth-v2, synth-v1.

# Path 1 (qarag 0.4 answer.py) ANSWER_SYSTEM_PROMPT, copied unchanged.
SYNTH_SYSTEM = """You answer questions using ONLY the numbered evidence provided.

Each evidence item is a question-and-answer pair extracted from a source document, followed by
the exact source sentences it was extracted from.

Rules:
1. Use only the evidence. Do not add facts from your own knowledge, even if you know them.
2. If the evidence does not contain the answer, reply exactly: "The indexed documents do not
   contain this information." Do not guess.
3. If the evidence is partial or conflicting, give what it supports and say what is missing or
   conflicting.
4. SYNTHESISE: read ALL evidence items and combine every item that is relevant into one
   coherent answer. Different items often hold different parts of the answer (for example one
   gives the date, another the place, a third a later detail) - bring them together, and do
   not repeat the same fact twice. Ignore items that are not relevant to the question.
5. Answer directly and concisely: first the answer itself, then any needed qualification. Keep
   names, numbers, dates and units exactly as in the evidence.
6. After each claim, cite the evidence items it comes from in square brackets, e.g. [1] or [2][3].
   Cite only items you actually used."""

# Added to SYNTH_SYSTEM only when the evidence contains model-memory items.
SYNTH_MEMORY_RULE = """
7. Items marked "MODEL MEMORY" are not from a document: they are an AI model's own recollection
   and are NOT verified. If your answer depends on such an item, say clearly that this part comes
   from model memory and is unverified."""

# Same user message as qarag: the evidence block, then the question.
SYNTH_USER = "Evidence:\n\n{evidence}\nQuestion: {question}"

# qarag's fixed refusal sentence (rule 2). Used by the code to set the status,
# with the same rule as hotpot_ask_eval.py: only this sentence -> not_found;
# the sentence plus other text -> partial.
REFUSAL = "The indexed documents do not contain this information."
REFUSAL_CORE = "do not contain this information"
