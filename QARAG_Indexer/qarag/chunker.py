"""
chunker.py - split a document into numbered sentences (each tagged with its
section), then group the sentences into chunks for the Q&A generator.

Why sentences first?
  Every sentence gets a permanent number (its `idx` in the document). The
  generator is shown the chunk as numbered lines like "[S12] ...", and must
  say which sentence numbers support each answer. That gives every Q&A pair
  exact PROVENANCE: we can always show the original sentences behind an
  answer, cite their section and page, and later measure coverage against
  HotpotQA's gold supporting sentences.

Why sections?
  Heading lines written by sources.py ("# Title", "## Career", "### Film")
  are not sentences. They update a heading stack, and every following
  sentence is tagged with the path, e.g. "Career > Film". When a fact is
  stated in several places, the merged pair cites ALL these sub-sections.

Why chunks with context BEFORE and AFTER?
  Each chunk has CORE sentences (the model writes Q&A pairs about these) plus
  read-only context on both sides:
    * context BEFORE = the end of the previous chunk, so the model can resolve
      words like "he" or "the company" at the start of the core;
    * context AFTER (look-ahead) = the start of the next chunk, because
      documents often state something briefly and explain it in the NEXT
      paragraph. The model may use (and cite) these sentences to complete an
      answer about a core sentence.
  Context sentences are never the subject of their own pairs, so the overlap
  does not create duplicates; when a look-ahead sentence is also questioned
  in its own chunk, the document-level merge step combines the two pairs.
"""
from __future__ import annotations

import re
from typing import Optional

from qarag.models import Chunk, RawDocument, Sentence

# A sentence ends at . ! or ? (optionally followed by a closing quote or
# bracket), followed by whitespace and then something that looks like the
# start of a new sentence (capital letter, digit, quote, bracket).
_SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=["\'(\[]?[A-Z0-9])')

# Common abbreviations that end with a period but do NOT end a sentence.
# Without this list, "Dr. Smith" or "U.S. Army" would be split in two.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "inc", "ltd", "co", "corp", "no", "vol", "fig", "al", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "mt", "ft",
}

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


def _split_block(block: str) -> list[str]:
    """Split one paragraph into sentences, re-joining splits that happened
    after an abbreviation or initials (e.g. "John F. Kennedy", "T.S. Arthur")."""
    pieces = _SENTENCE_END.split(block)
    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if sentences:
            words = sentences[-1].rstrip(".").split()
            last_word = words[-1].lower() if words else ""
            if last_word in _ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.)*[a-z]", last_word):
                sentences[-1] = f"{sentences[-1]} {piece}"
                continue
        sentences.append(piece)
    return sentences


def split_sentences(document: RawDocument) -> list[Sentence]:
    """Turn the document into a numbered list of sentences, each tagged with
    its page (PDFs) and its section path (documents with headings).

    Line breaks are hard boundaries (paragraphs, headings, list items); each
    paragraph is then split into sentences."""
    sentences: list[Sentence] = []
    heading_stack: list[str] = []          # heading_stack[0] = level-1 heading, etc.
    title = document.title.strip().lower()

    # If the document has real sub-sections, text before the first one (the
    # lead paragraph) is labelled "Introduction", so it can be cited too.
    all_headings = [h.strip() for h in re.findall(r"^#{1,6}\s+(.+)$", document.full_text, flags=re.MULTILINE)]
    has_subsections = any(h.lower() != title for h in all_headings)

    def current_section() -> Optional[str]:
        path = list(heading_stack)
        # The level-1 heading is usually the document title itself; leaving it
        # out keeps citations short ("Career > Film", not "Scott Derrickson > Career > Film").
        if path and path[0].strip().lower() == title:
            path = path[1:]
        path = [p for p in path if p]
        if path:
            return " > ".join(path)
        return "Introduction" if has_subsections else None

    for page, text in document.segments:
        for block in re.split(r"\n+", text):
            block = " ".join(block.split())
            heading = _HEADING.match(block)
            if heading:
                level = len(heading.group(1))
                # Keep headings above this level, replace this level, drop deeper ones.
                heading_stack = heading_stack[:level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append("")       # a skipped level (e.g. # then ###)
                heading_stack.append(heading.group(2).strip())
                continue
            block = re.sub(r"^[-*•]\s+", "", block)   # list bullets
            if len(block) < 3:                  # skip stray characters / page numbers
                continue
            section = current_section()
            for sentence_text in _split_block(block):
                sentences.append(Sentence(idx=len(sentences), text=sentence_text,
                                          page=page, section=section))
    return sentences


def build_chunks(sentences: list[Sentence], chunk_tokens: int, overlap_tokens: int,
                 lookahead_tokens: int = 0, chars_per_token: int = 4) -> list[Chunk]:
    """Group sentences into chunks of about `chunk_tokens` tokens.

    A chunk never cuts a sentence in half: sentences are added whole until the
    budget is reached. A single sentence longer than the budget becomes a
    chunk on its own. Context before = the last sentences of the previous
    core (up to `overlap_tokens`); context after = the first sentences of the
    next core (up to `lookahead_tokens`)."""
    max_chars = chunk_tokens * chars_per_token
    cores: list[list[Sentence]] = []
    current: list[Sentence] = []
    current_chars = 0
    for sentence in sentences:
        length = len(sentence.text) + 1
        if current and current_chars + length > max_chars:
            cores.append(current)
            current, current_chars = [], 0
        current.append(sentence)
        current_chars += length
    if current:
        cores.append(current)
    return attach_context(cores, overlap_tokens, lookahead_tokens, chars_per_token)


def _take(sentences: list[Sentence], budget_chars: int) -> list[Sentence]:
    """Whole sentences from the start of `sentences` until the character
    budget is used (always at least one sentence if the budget is > 0)."""
    taken: list[Sentence] = []
    used = 0
    for sentence in sentences:
        if taken and used + len(sentence.text) > budget_chars:
            break
        taken.append(sentence)
        used += len(sentence.text) + 1
    return taken


def attach_context(cores: list[list[Sentence]], overlap_tokens: int, lookahead_tokens: int = 0,
                   chars_per_token: int = 4) -> list[Chunk]:
    """Give every core read-only context: the tail of the previous core
    (before) and the head of the next core (after). Also used when resuming
    an interrupted document."""
    before_chars = overlap_tokens * chars_per_token
    after_chars = lookahead_tokens * chars_per_token
    chunks: list[Chunk] = []
    for i, core in enumerate(cores):
        before: list[Sentence] = []
        after: list[Sentence] = []
        if i > 0 and before_chars > 0:
            # walk backwards from the end of the previous core, then restore order
            before = list(reversed(_take(list(reversed(cores[i - 1])), before_chars)))
        if i + 1 < len(cores) and after_chars > 0:
            after = _take(cores[i + 1], after_chars)
        chunks.append(Chunk(chunk_index=i, context=before, core=core, lookahead=after))
    return chunks


def is_content_sentence(sentence: Sentence, min_words: int) -> bool:
    """Sentences long enough to be expected to carry a fact. Used by the
    coverage check: every content sentence should be cited by some pair."""
    return len(sentence.text.split()) >= min_words
