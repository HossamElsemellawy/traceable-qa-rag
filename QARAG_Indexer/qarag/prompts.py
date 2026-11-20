"""
prompts.py - every instruction given to the models, in one place.

Three prompts are used per document:

  1. GENERATION (per chunk): turn numbered sentences into ATOMIC Q&A pairs.
  2. COVERAGE (per chunk, only if needed): the program finds content sentences
     that no pair cited and asks the model, in the same conversation, to
     cover them. This is how "cover every fact" is enforced, not just hoped for.
  3. MERGE (per document, only for groups of similar candidates): the same
     fact often appears in several places (an introduction and a later
     section, a summary table and a paragraph). Candidates from ANY chunks
     whose questions are similar are shown together with their sentences, and
     the model merges true duplicates into ONE pair whose answer combines all
     places and whose citation lists every sentence and sub-section.

Why the rules exist (each prevents a failure that would hurt retrieval or
governance later):
  * Self-contained questions: a question is embedded and searched ON ITS OWN,
    so "When was he born?" is useless once separated from its paragraph.
  * Atomic pairs (one fact each): a pair is also the unit an admin will later
    grant or restrict access to; mixing a public and a sensitive fact in one
    pair would make that impossible.
  * Only from the text, with sentence ids: prevents facts remembered from
    training (hallucination) and gives provenance for citations.

ANSWER STYLES - choose with --answer-style and compare in separate databases:
  short     many narrow questions, answer = the exact fact only ("July 16, 1966")
  sentence  one fact per pair, answer = one complete self-contained sentence (default)
  detailed  one topic per pair, answer = 1-3 sentences with its directly related details

v4 (current) adopts the extraction rules of the "Comprehensive Atomic Fact
Extraction" specification in condensed form: the decomposition example,
qualifiers and conditions, exact numbers (percent vs percentage points),
negative facts, comparisons, lists, table values, cited works, and stricter
question rules (one unambiguous answer, never containing the answer). The
specification's registry, IDs, evidence and audit steps are done by code
(store.py, consolidate.py, coverage pass), not asked of the model.

Change a prompt here and bump PROMPT_VERSION, so the database records which
prompt produced which pairs.
"""
from __future__ import annotations

from qarag.models import Chunk, Sentence

PROMPT_VERSION = "qa-gen-v4"

ANSWER_STYLES = ("short", "sentence", "detailed")

# Style-specific rules inserted into both the generation and the merge prompt.
STYLE_RULES = {
    "short": (
        "ANSWER STYLE = SHORT. Write MANY narrow questions, one per fact. Each answer is ONLY the "
        "exact fact - a name, date, number, place, term or short phrase, copied as written in the "
        "document (for example: \"July 16, 1966\"), never a full sentence. Keep an essential qualifier "
        "or condition in the answer (for example: \"about 40%, not statistically significant\"). If the "
        "answer is a list, separate the items with \"; \"."
    ),
    "sentence": (
        "ANSWER STYLE = SENTENCE. One fact per pair. Each answer is one complete, self-contained "
        "sentence that states the fact and names the subject again (for example: \"Scott Derrickson "
        "was born on July 16, 1966.\"). Keep names, dates and numbers exactly as written."
    ),
    "detailed": (
        "ANSWER STYLE = DETAILED. One topic per pair: a fact together with its directly related "
        "details that the text gives (when, where, who, how much, conditions, exceptions), in 1 to 3 "
        "complete sentences that name the subject. Never combine unrelated facts in one pair. Keep "
        "names, dates and numbers exactly as written."
    ),
}

_GENERATION_TEMPLATE = """You convert documents into question-and-answer pairs for a search index.
The goal is COMPLETE factual coverage, not a summary.

INPUT
You receive part of a document as numbered sentences, grouped under their section names.
- [S<number>] lines are the sentences you must cover.
- [before:<number>] lines come from the previous part: use them only to understand references
  (who "he" or "the company" is).
- [after:<number>] lines come from the next part. Documents often state something briefly and
  explain it in the following paragraph: when an [after] sentence completes or clarifies a fact
  from an [S] sentence, include that detail in the answer and cite the [after] sentence too.
- Lines written as "<label> - <field>: <value>" are rows of a table or information box. Every
  value in them is a fact.
Never write a pair whose answer rests only on [before] or [after] sentences.

WHAT TO EXTRACT - every distinct, verifiable claim in the [S] sentences:
- entities and their attributes, roles, relationships, ownership, authorship, purpose, function,
  inputs and outputs, causes and effects;
- numbers exactly as written: dates, quantities, units, ranges, thresholds, percentages, sample
  sizes, statistical results (a percentage change is not a percentage-point change);
- conditions, assumptions, exceptions, limitations and uncertainty - keep qualifiers such as
  "approximately", "may", "under certain conditions" or "not statistically significant";
- negative facts: what is excluded, absent, failed or does not happen;
- comparisons: both things compared, the criterion, the result and the conditions;
- definitions, steps of processes, examples, and values from tables or captions;
- what THIS document says about other works it cites (not what those works themselves say).

ATOMIC PAIRS
One independently testable claim per pair. Example: "The model achieved 92% accuracy on Dataset
A, outperforming the baseline by 4 percentage points." gives separate pairs for the 92% accuracy
on Dataset A, the model outperforming the baseline, and the 4-percentage-point improvement - each
question still naming the model, the dataset and the metric. Do not split mechanically into
meaningless fragments.
Lists: one pair per meaningful item, plus one pair for the complete list when its membership
matters. Never write two pairs for the same fact with different wording.

QUESTIONS
1. Fully self-contained: name the specific person, organisation, place, work, method or thing.
   Never use "he", "it", "the company", "this approach" or "the above method" without naming
   it. Use the section name if it helps identify the subject.
2. Exactly one unambiguous expected answer; factual, never opinion.
3. Never contain the answer.
4. Keep similar entities, methods and results distinct (say which model, dataset, year, version).
5. Ask the way a person searching would, preferably starting with What, Which, Who, When, Where,
   How many, How much or Under what conditions. No broad questions such as "What are the main
   findings?" - break them into atomic questions.

ANSWERS
6. {style_rule}
7. Keep names, numbers, units, dates and technical terms exactly as written. Keep conditions,
   exceptions and qualifiers whenever leaving them out would change the meaning.
8. Use ONLY what the text states: no outside knowledge, no interpretation beyond the text, and
   keep what was observed or reported distinct from what the authors suggest or expect.

CITATIONS
9. "sentence_ids" lists EVERY sentence that supports the answer, by number (for [S12] or
   [after:12] write 12). If the same fact is stated in more than one sentence, list all of them.
   Every pair must cite at least one [S] sentence.
10. Skip text with no factual content (navigation text, greetings, empty headings).

Return ONLY a JSON object, with no explanation before or after it, in exactly this form:
{{"pairs": [{{"question": "...", "answer": "...", "sentence_ids": [12, 13]}}]}}
If the sentences contain no facts, return {{"pairs": []}}."""


def generation_system_prompt(style: str) -> str:
    return _GENERATION_TEMPLATE.format(style_rule=STYLE_RULES[style])


def _section_lines(sentences: list[Sentence], tag_fn) -> list[str]:
    """Format sentences, inserting a "(Section: ...)" line whenever the section changes."""
    lines, last = [], object()
    for s in sentences:
        if s.section != last:
            lines.append(f"(Section: {s.section})" if s.section else "(Section: none)")
            last = s.section
        lines.append(f"{tag_fn(s)} {s.text}")
    return lines


def build_user_prompt(doc_title: str, chunk: Chunk, total_chunks: int) -> str:
    """The message for one chunk: title, position, context, numbered sentences.
    The [S<idx>] numbers are document-wide, so returned ids are stored directly."""
    lines = [f"Document title: {doc_title}", f"Part {chunk.chunk_index + 1} of {total_chunks}", ""]
    if chunk.context:
        lines.append("From the previous part (background only):")
        lines.extend(_section_lines(chunk.context, lambda s: f"[before:{s.idx}]"))
        lines.append("")
    lines.append("Sentences to cover:")
    lines.extend(_section_lines(chunk.core, lambda s: f"[S{s.idx}]"))
    if chunk.lookahead:
        lines.append("")
        lines.append("From the next part (use to complete answers about the sentences above):")
        lines.extend(_section_lines(chunk.lookahead, lambda s: f"[after:{s.idx}]"))
    lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def coverage_message(uncovered: list[Sentence]) -> str:
    """Follow-up message when some content sentences were not cited by any pair."""
    listed = "\n".join(f"[S{s.idx}] {s.text}" for s in uncovered)
    return ("These sentences are not supported by any of your pairs yet:\n"
            f"{listed}\n\n"
            "Write additional pairs for every fact they contain, following exactly the same rules "
            "and answer style. Pay special attention to numbers, qualifiers, conditions, negative "
            "facts, list items and table values. Do not repeat pairs you already wrote. If one of "
            "them has no factual content, skip it. Return ONLY the JSON object with the NEW pairs: {\"pairs\": [...]}")


# Sent when the model's previous reply could not be parsed, before retrying.
JSON_REPAIR_MESSAGE = ("Your previous reply was not a valid JSON object in the required form. "
                       'Reply again with ONLY the JSON object.')


_MERGE_TEMPLATE = """You consolidate question-and-answer pairs that were generated from different parts of
the SAME document. Because documents often repeat a fact in several places (an introduction,
a later section, a summary), some candidates may ask for the same information.

You receive numbered candidate pairs and the document sentences they cite, with section names.

Group the candidates:
- Put candidates in the same group ONLY if they ask for the same information: the same fact,
  or parts of one answer (for example items of one list) stated in different places.
- Different facts about the same subject stay in SEPARATE groups (birth date and birth place
  are different facts). Claims that differ in scope, condition, time, dataset or value are
  different facts, even when the wording is similar.
- For each group, write one self-contained question and one answer that combines everything
  the group's sentences say about it, keeping exact numbers, conditions and qualifiers. Use only
  the given sentences.
- {style_rule}
- "sentence_ids": EVERY sentence number that supports the combined answer, from all sections.
- Every candidate number must appear in exactly one group. A candidate that matches no other
  candidate forms a group of its own; copy its question and answer unchanged.

Return ONLY a JSON object in exactly this form:
{{"groups": [{{"members": [1, 3], "question": "...", "answer": "...", "sentence_ids": [4, 57]}}]}}"""


def merge_system_prompt(style: str) -> str:
    return _MERGE_TEMPLATE.format(style_rule=STYLE_RULES[style])


def merge_user_prompt(doc_title: str, candidates: list[tuple[int, str, str, list[int]]],
                      sentences: list[Sentence]) -> str:
    """candidates: (number, question, answer, sentence_ids); sentences: all cited ones."""
    lines = [f"Document title: {doc_title}", "", "Candidate pairs:"]
    for number, question, answer, ids in candidates:
        lines.append(f"{number}. Q: {question}")
        lines.append(f"   A: {answer}")
        lines.append(f"   cites: {', '.join(f'S{i}' for i in ids)}")
    lines.append("")
    lines.append("Cited sentences:")
    lines.extend(_section_lines(sorted(sentences, key=lambda s: s.idx), lambda s: f"[S{s.idx}]"))
    lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)
