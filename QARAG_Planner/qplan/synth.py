"""
synth.py - LLM-S: the final, grounded answer with citations.

LLM-S sees ONLY the collected evidence (never the agent's reasoning), so the
final answer can be checked against exactly what it was given.

Since qplan 0.1.6 (synth-v5) the answer step is the Path 1 answer step of
qarag 0.4 (answer.py): the same prompt, the same evidence layout (numbered
items with their source sentences), the same plain-text reply with the fixed
refusal sentence. The code translates the numbers back to [E#]/[M#] and sets
the status with the rule of hotpot_ask_eval.py.

Citations are made in two steps:
  1. the model writes [E#]/[M#] markers in its answer;
  2. this code (not the model) builds the "Sources" list from the markers,
     using the stored details of each evidence item (document title, section,
     supporting sentences, URL, pair_id). The model therefore cannot invent
     a source: a marker that does not exist is reported as invalid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from qplan import prompts
from qplan.config import Settings
from qplan.evidence import EvidenceStore, KBEvidence, MemoryEvidence
from qplan.llm import LLMClient

# Matches citation markers such as [E1], [M2], and lists like [E1, E3].
_CITE_GROUP = re.compile(r"\[((?:[EM]\d+)(?:\s*[,;]\s*[EM]\d+)*)\]")
# Matches template placeholders a model may copy instead of a real label,
# e.g. [E#], [M#], [E#]/[M#], [En], [E?]. They are reported as invalid citations.
_PLACEHOLDER = re.compile(r"\[\s*[EM]\s*[#?nNxX]\s*\]")


@dataclass
class FinalAnswer:
    answer: str                         # answer text with [E#]/[M#] markers
    short_answer: str                   # synth-v5: "unknown" if not_found, else "" (the
                                        # evaluation extracts its own short answer)
    status: str                         # answered | partial | not_found
    cited: list = field(default_factory=list)          # valid ids, in order of appearance
    invalid_citations: list = field(default_factory=list)  # ids the model invented
    basis: str = "none"                 # kb | memory | mixed | none
    sources_text: str = ""              # the formatted Sources list
    parse_ok: bool = True               # (synth-v5 replies are plain text: always True)
    gap_noted: bool = False             # the answer names a missing fact (GAP_PHRASES)


def find_placeholders(text: str) -> list[str]:
    """Placeholder markers such as [E#] (copied templates, not real labels)."""
    seen: list[str] = []
    for m in _PLACEHOLDER.findall(text or ""):
        tag = m.replace(" ", "").strip("[]")          # "[E#]" -> "E#", like "E9"
        if tag not in seen:
            seen.append(tag)
    return seen


def find_citations(text: str) -> list[str]:
    """All citation ids in order of first appearance, without duplicates."""
    seen: list[str] = []
    for group in _CITE_GROUP.findall(text or ""):
        for cid in re.split(r"\s*[,;]\s*", group):
            if cid and cid not in seen:
                seen.append(cid)
    return seen


# Source text shown per pair to the ReAct AGENT (qplan 0.1.5): at most this
# many supporting sentences, each cut to this many characters (keeps the
# growing agent chat small). LLM-S gets every sentence in full (synth-v5).
MAX_TEXT_SENTENCES = 3
MAX_SENTENCE_CHARS = 400


def source_text(hit) -> str:
    """
    The pair's supporting sentences from the source document, as one string.

    Why (HotpotQA run react30): the indexer writes pairs that state ONE fact,
    but the sentence they come from often states more. "What were Robert
    Graves' full name and professions?" was taken from the sentence that also
    gives his dates; LLM-S saw only the pair and said the death date was
    missing. Showing the sentence fixes that without changing the database.
    """
    parts = []
    for s in (hit.sentences or [])[:MAX_TEXT_SENTENCES]:
        text = " ".join(str(s.get("text") or "").split())
        if len(text) > MAX_SENTENCE_CHARS:
            text = text[:MAX_SENTENCE_CHARS].rstrip() + " ..."
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts)


# Phrases that show an answer names a gap without using the refusal
# sentence (e.g. "The publication year of X is not provided"). They do NOT
# change the status (the scorer only knows the refusal sentence); they only
# make the direct-first step 0 escalate to ReAct instead of stopping.
GAP_PHRASES = ("not provided", "not stated", "not mentioned", "not specified", "not given",
               "not available", "does not specify", "does not state", "does not mention",
               "does not provide", "no information", "cannot be determined", "not possible to determine",
               "unable to determine", "is unknown")

# A group of numbered citations as qarag's prompt asks for: [1], [2, 3], [2; 3]
_NUM_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def number_evidence(store: EvidenceStore) -> list:
    """The evidence items in the order LLM-S sees them, numbered 1..n:
    knowledge-base items best score first (as in qarag: one ranked list),
    then model-memory items. Returns [(number, item), ...]."""
    items = list(store.ranked_kb()) + list(store.memory)
    return [(i, item) for i, item in enumerate(items, start=1)]


def format_evidence_for_llm(numbered: list) -> str:
    """
    The evidence block, in the layout of qarag 0.4 answer.py (_evidence_block):

        [1] Document: <title> | <sections> | pages <p>
            Q: <pair question>
            A: <pair answer>
            Source: <supporting sentence>      (every supporting sentence)

    Model-memory items (only when memory was used) are marked MODEL MEMORY.
    """
    lines = []
    for n, item in numbered:
        if isinstance(item, KBEvidence):
            h = item.hit
            where = " | ".join(h.sections) if h.sections else ""
            pages = f" | pages {', '.join(map(str, h.pages))}" if h.pages else ""
            lines.append(f"[{n}] Document: {h.title or h.doc_id}{' | ' + where if where else ''}{pages}")
            lines.append(f"    Q: {h.question}")
            lines.append(f"    A: {h.answer}")
            for s in h.sentences or []:
                lines.append(f"    Source: {s.get('text', '')}")
        else:
            lines.append(f"[{n}] MODEL MEMORY ({item.model}, NOT verified, confidence {item.confidence})")
            lines.append(f"    Q: {item.question}")
            lines.append(f"    A: {item.answer}")
        lines.append("")
    return "\n".join(lines)


def numbers_to_labels(text: str, numbered: list) -> tuple[str, list]:
    """Turn the model's [1], [2][3], [2, 3] into qplan's labels [E5], [E2][E7], ...
    Numbers that are not in the evidence stay as they are and are returned
    as invalid citations."""
    label = {n: item.ev_id for n, item in numbered}
    invalid: list[str] = []

    def one_group(m):
        parts = re.split(r"(\s*[,;]\s*)", m.group(1))
        out = []
        for part in parts:
            if part.strip().isdigit():
                n = int(part)
                if n in label:
                    out.append(label[n])
                else:
                    out.append(part)
                    if part.strip() not in invalid:
                        invalid.append(part.strip())
            else:
                out.append(part)
        return "[" + "".join(out) + "]"
    return _NUM_GROUP.sub(one_group, text or ""), invalid


def answer_status(text: str) -> str:
    """
    Status from the answer text, by the rule hotpot_ask_eval.py uses:
      only the refusal sentence (at most 2 other words)  -> not_found
      the refusal sentence plus other text               -> partial
      no refusal sentence                                -> answered
    """
    low = (text or "").lower()
    if prompts.REFUSAL_CORE not in low:
        return "answered"
    rest = re.sub(r"\[[^\]]*\]", " ", low).replace(prompts.REFUSAL.lower(), " ")
    rest = re.sub(r"[^\w\s]", " ", rest)
    return "not_found" if len(rest.split()) <= 2 else "partial"


def synthesize(question: str, store: EvidenceStore, llm: LLMClient,
               settings: Settings, role: str = "synth") -> FinalAnswer:
    """
    Call LLM-S (the Path 1 answer step, synth-v5) and return the checked,
    cited final answer.

    role: logged with the call. "synth" = the final answer after the loop;
          "synth_direct" = the direct-first attempt before the loop (0.1.5).
    """
    numbered = number_evidence(store)
    if not numbered:
        # As qarag: no evidence -> the refusal sentence, no model call.
        return FinalAnswer(answer=prompts.REFUSAL, short_answer="unknown", status="not_found",
                           sources_text=format_sources([], {}))
    system = prompts.SYNTH_SYSTEM + (prompts.SYNTH_MEMORY_RULE if store.memory else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.SYNTH_USER.format(
            evidence=format_evidence_for_llm(numbered), question=question)},
    ]
    reply = llm.chat(role, settings.synth_model, messages,
                     settings.synth_temperature, settings.synth_max_tokens,
                     settings.synth_reasoning)

    raw = (reply.text or "").strip()
    answer, invalid = numbers_to_labels(raw, numbered)
    status = answer_status(answer) if answer else "not_found"
    if not answer:
        answer = prompts.REFUSAL

    # Check every citation against the evidence that really exists.
    known = store.by_id()
    cited_all = find_citations(answer)
    cited = [c for c in cited_all if c in known]
    invalid += [c for c in cited_all if c not in known and c not in invalid]
    invalid += [p for p in find_placeholders(answer) if p not in invalid]

    has_kb = any(c.startswith("E") for c in cited)
    has_mem = any(c.startswith("M") for c in cited)
    basis = "mixed" if has_kb and has_mem else "kb" if has_kb else "memory" if has_mem else "none"
    gap = any(g in answer.lower() for g in GAP_PHRASES)

    return FinalAnswer(answer=answer, short_answer="unknown" if status == "not_found" else "",
                       status=status, cited=cited, invalid_citations=invalid, basis=basis,
                       sources_text=format_sources(cited, known), parse_ok=True, gap_noted=gap)


def format_sources(cited: list[str], known: dict) -> str:
    """The human-readable Sources list, built from stored evidence details."""
    if not cited:
        return "Sources: none cited."
    lines = ["Sources:"]
    for cid in cited:
        ev = known[cid]
        if isinstance(ev, KBEvidence):
            h = ev.hit
            section = "; ".join(h.sections) if h.sections else "-"
            lines.append(f"[{cid}] {h.title or h.doc_id} | section: {section} | score {h.score:.2f}")
            lines.append(f"     Q: {h.question}")
            lines.append(f"     A: {h.answer}")
            for s in h.sentences[:3]:   # the exact source sentences (max 3 shown)
                page = f", page {s['page']}" if s.get("page") not in (None, "") else ""
                lines.append(f"     sentence {s['idx']}{page}: \"{s['text']}\"")
            if len(h.sentences) > 3:
                lines.append(f"     (+{len(h.sentences) - 3} more supporting sentences)")
            lines.append(f"     source: {h.source or '-'}   pair_id: {h.pair_id}")
        elif isinstance(ev, MemoryEvidence):
            lines.append(f"[{cid}] MODEL MEMORY ({ev.model}, confidence {ev.confidence}) "
                         "- NOT verified against the knowledge base")
            lines.append(f"     Q: {ev.question}")
            lines.append(f"     A: {ev.answer}")
    return "\n".join(lines)
