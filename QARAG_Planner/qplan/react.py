"""
react.py - the ReAct engine (Yao et al. 2023: Reasoning + Acting).

One run:

    question
      |
      v
    +--------------------------------------------------------------+
    | step n:  ReAct agent -> {"thought", "action", "input"}        |
    |          action = search_kb -> db.retrieve()  -> [E#] items   |
    |                   recall    -> memory model   -> [M#] item    |
    |                   finish    -> leave the loop                 |
    |          the result is sent back to the agent as an           |
    |          "Observation", and the next step starts              |
    +--------------------------------------------------------------+
      |  (finish, or max_steps reached, or too many invalid replies)
      v
    LLM-S: final answer from the collected evidence, with citations

Direct-first (qplan 0.1.5, setting direct_first, default on): BEFORE the loop
there is an automatic step 0, like the Path 1 one-shot baseline:

    step 0: search_kb(<the whole question>, k = direct_top_k) -> LLM-S
            status "answered" (with an [E#] citation, no gap named) -> the answer,
                                                        no ReAct ("direct_answered")
            otherwise -> "escalated": the ReAct loop starts. The step-0 pairs
                        stay in the evidence, and the agent is told what step 0
                        found and what the first answer lacked.

The conversation with the agent is one growing chat:
    system  : rules and actions          (prompts.react_system)
    user    : the question
    assistant: step 1 JSON
    user    : Observation of step 1
    assistant: step 2 JSON
    ...
so the agent always sees everything it did and found so far (this is how
hop 2 of a bridge question can use what hop 1 found, as in IRCoT).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from qplan import __version__, prompts
from qplan.config import Settings
from qplan.db import QADatabase
from qplan.evidence import EvidenceStore
from qplan.llm import LLMClient, LLMReply, extract_json
from qplan.runlog import RunLog, new_run_id
from qplan.synth import FinalAnswer, source_text, synthesize

VALID_ACTIONS = ("search_kb", "recall", "finish")
# Stop the loop after this many invalid agent replies IN A ROW.
MAX_CONSECUTIVE_INVALID = 2


@dataclass
class RunResult:
    """What one question produced (also stored in the run log)."""
    run_id: str
    question: str
    final: FinalAnswer | None
    steps: list = field(default_factory=list)
    stop_reason: str = ""
    trace_path: str = ""
    cost_usd: float = 0.0
    error: str | None = None
    evidence: list = field(default_factory=list)   # every [E#]/[M#] item of the run
    # direct-first (0.1.5): None = not used; False = answered at step 0;
    # True = step 0 was not enough and the ReAct loop ran.
    escalated: bool | None = None
    direct_status: str = ""        # LLM-S status at step 0 (or "no_match")


def _norm(text: str) -> str:
    """Normalise a search text to detect exact repeats."""
    return " ".join((text or "").lower().split()).strip(" ?.!")


class ReActEngine:
    """Runs questions through the ReAct loop and logs everything."""

    def __init__(self, settings: Settings, db: QADatabase, runlog: RunLog,
                 fake_responder=None, printer=print):
        settings.validate()
        self.s = settings
        self.db = db
        self.log = runlog
        self.print = printer if settings.verbose else (lambda *a, **k: None)
        # Every LLM call is reported to _on_llm_call, which writes the log.
        self.llm = LLMClient(settings, on_call=self._on_llm_call, fake_responder=fake_responder)
        # Current run and step, used by the logging callback.
        self._where: dict | None = None    # document filter of the current run
        self._run_id = ""
        self._step_no = 0

    # ------------------------------------------------------------------
    def _on_llm_call(self, role: str, messages: list, reply: LLMReply) -> None:
        self.log.log_llm_call(self._run_id, self._step_no, role, messages, reply)

    # ------------------------------------------------------------------
    def run(self, question: str, where: dict | None = None) -> RunResult:
        """
        Answer one question. Never raises: errors are logged in the run.

        where: optional document filter applied INSIDE every search of this
               run (Chroma metadata filter), e.g. {"doc_id": {"$in": [...]}}:
               the agent can only find pairs of those documents.
        """
        question = question.strip()
        self._where = where
        self._run_id = run_id = new_run_id()
        self._step_no = 0
        started = time.perf_counter()
        self.log.start_run(
            run_id, question, str(self.db.dir), self.db.fingerprint(), self.db.embed_model,
            self.s.to_dict(),
            {"react": prompts.REACT_PROMPT_VERSION, "memory": prompts.MEMORY_PROMPT_VERSION,
             "synth": prompts.SYNTH_PROMPT_VERSION},
            __version__)
        self.print(f"\n=== run {run_id} ===\nQuestion: {question}")

        store = EvidenceStore()
        steps: list[dict] = []
        result = RunResult(run_id=run_id, question=question, final=None, steps=steps)
        try:
            direct = None
            if self.s.direct_first == "on":
                # ---- step 0: the whole question, one search, one answer ----
                direct = self._direct(question, store, steps)
                result.direct_status = direct["status"]
                result.escalated = not direct["accepted"]
            if direct and direct["accepted"]:
                result.final = direct["final"]
                result.stop_reason = "direct_answered"
            else:
                result.stop_reason = self._loop(question, store, steps, direct)
                # ---- LLM-S -------------------------------------------------
                self._step_no = 0          # LLM-S calls are logged as step 0
                self.print("\n--- LLM-S: writing the final answer ---")
                result.final = synthesize(question, store, self.llm, self.s)
            status = "done"
        except Exception as exc:       # keep the partial log, report the error
            result.error = f"{type(exc).__name__}: {exc}"
            result.stop_reason = result.stop_reason or "error"
            status = "error"
            self.print(f"\nERROR: {result.error}")

        # ---- store the run --------------------------------------------
        f = result.final
        fields = {
            "status": status, "stop_reason": result.stop_reason, "n_steps": len(steps),
            "n_searches": sum(1 for s in steps if s["action"] in ("search_kb", "direct")
                              and s["outcome"] in ("match", "no_match")),
            "n_recalls": sum(1 for s in steps if s["outcome"].startswith("recall")),
            "ms": int((time.perf_counter() - started) * 1000), "error": result.error,
        }
        fields["filter_json"] = _dumps(where) if where else None
        fields["escalated"] = None if result.escalated is None else int(result.escalated)
        fields["direct_status"] = result.direct_status or None
        result.evidence = store.to_list()
        if f:
            fields.update({"missing_json": _dumps(store.missing) if store.missing else None})
            fields.update({"answer": f.answer, "short_answer": f.short_answer,
                           "answer_status": f.status, "answer_basis": f.basis,
                           "cited_json": _dumps(f.cited), "invalid_cites": _dumps(f.invalid_citations),
                           "sources_text": f.sources_text})
        self.log.finish_run(run_id, fields, store.to_list(), f.cited if f else [])
        run = self.log.get_run(run_id)
        result.cost_usd = run["cost_usd"] or 0.0
        result.trace_path = str(self.log.write_trace(run_id, render_trace(run)))
        return result

    # ------------------------------------------------------------------
    def _direct(self, question: str, store: EvidenceStore, steps: list) -> dict:
        """
        Direct-first step 0 (qplan 0.1.5): search the whole question once with
        direct_top_k, and let LLM-S try to answer from those pairs.

        Accepted (no ReAct) only if the answer has status "answered" (no refusal
        sentence), cites at least one knowledge-base item, and names no gap
        (synth.GAP_PHRASES). Everything else escalates to the ReAct loop.
        The decision and the attempt are logged as step 0 (action "direct").
        """
        self._step_no = 0
        t0 = time.perf_counter()
        step = {"step_no": 0, "thought": "(automatic) direct-first: search the whole question, then try to answer",
                "action": "direct", "input": question, "observation": "", "outcome": "", "n_hits": None,
                "best_score": None, "evidence_ids": [], "missing": None}
        self.print(f"\n[step 0] direct-first: search_kb({question!r}, k={self.s.direct_top_k})")
        search_obs = self._run_search(step, question, self.s.direct_top_k, store)
        info = {"accepted": False, "status": "no_match", "final": None, "observation": search_obs, "attempt": ""}
        if step["outcome"] == "match":
            final = synthesize(question, store, self.llm, self.s, role="synth_direct")
            kb_cited = any(c.startswith("E") for c in final.cited)
            # 0.1.6: an answer that names a gap ("... is not provided") without the
            # refusal sentence also escalates (react30_1 #10: such an answer guessed).
            accepted = final.status == "answered" and kb_cited and not final.gap_noted
            info.update({"accepted": accepted, "status": final.status, "final": final, "attempt": final.answer})
            verdict = "accepted, no ReAct needed" if accepted else "not complete, escalated to ReAct"
            step["observation"] = (f"{search_obs}\n\nFirst answer attempt (status {final.status}, {verdict}):\n"
                                   f"{final.answer}")
        else:
            step["observation"] = f"{search_obs}\n\n(no pairs found: escalated to ReAct)"
        self._record_step(step, t0, steps)
        return info

    # ------------------------------------------------------------------
    def _loop(self, question: str, store: EvidenceStore, steps: list, direct: dict | None = None) -> str:
        """The Thought -> Action -> Observation cycles. Returns the stop reason.

        direct: the result of the direct-first step 0 when it ran (the agent
                then starts knowing what step 0 found and what was missing).
        """
        s = self.s
        if direct:
            first = prompts.REACT_FIRST_USER_AFTER_DIRECT.format(
                question=question, status=direct["status"], observation=direct["observation"],
                attempt=direct["attempt"] or "(none: step 0 found no pairs)", max_steps=s.max_steps)
        else:
            first = prompts.REACT_FIRST_USER.format(question=question, max_steps=s.max_steps)
        messages = [
            {"role": "system", "content": prompts.react_system(s.memory_policy, s.max_steps)},
            {"role": "user", "content": first},
        ]
        searched: dict[str, int] = {}      # normalised search text -> step where it was made
        if direct:
            searched[_norm(question)] = 0  # step 0 already searched the whole question
        invalid_in_a_row = 0
        redirected = False                 # the "missing -> recall" redirect is used once per run

        for step_no in range(1, s.max_steps + 1):
            self._step_no = step_no
            t0 = time.perf_counter()

            # ---- Thought + Action: ask the agent --------------------------
            reply = self.llm.chat("react", s.react_model, messages, s.react_temperature, s.react_max_tokens,
                                 s.react_reasoning)
            messages.append({"role": "assistant", "content": reply.text})
            step = {"step_no": step_no, "thought": None, "action": None, "input": None,
                    "observation": "", "outcome": "", "n_hits": None, "best_score": None,
                    "evidence_ids": [], "missing": None}

            try:
                data = extract_json(reply.text)
                step["thought"] = str(data.get("thought", "")).strip()
                step["action"] = str(data.get("action", "")).strip().lower()
                step["input"] = str(data.get("input", "")).strip()
                if step["action"] not in VALID_ACTIONS:
                    raise ValueError(f"unknown action {step['action']!r}; use one of {', '.join(VALID_ACTIONS)}")
                if step["action"] != "finish" and not step["input"]:
                    raise ValueError("the action input is empty")
                if step["action"] == "finish":
                    # react-v3: finish must declare what is still missing.
                    step["missing"] = _missing_list(data)
                invalid_in_a_row = 0
            except ValueError as exc:
                # ---- invalid reply: tell the agent, count it --------------
                invalid_in_a_row += 1
                step["action"] = step["action"] or "(invalid)"
                step["outcome"] = "invalid"
                step["observation"] = (f"Your reply could not be used: {exc}. Reply with exactly one JSON "
                                       'object: {"thought": "...", "action": "'
                                       + prompts.action_names(s.memory_policy) + '", "input": "..."}')
                self._record_step(step, t0, steps)
                if invalid_in_a_row >= MAX_CONSECUTIVE_INVALID:
                    return "invalid_replies"
                messages.append(self._observation_msg(step_no, step["observation"]))
                continue

            self.print(f"\n[step {step_no}] thought: {step['thought']}\n"
                       f"[step {step_no}] action : {step['action']}({step['input']!r})")

            # ---- Act ------------------------------------------------------
            if step["action"] == "finish" and not _agent_searched(steps):
                # ---- guard: no conclusion without looking -----------------
                # The prompt says "search before concluding", but a model can
                # ignore that (qplan 0.1.0 run: the agent finished at step 1
                # and invented a search result). Enforced here in code.
                # After direct-first, step 0 does not count: it was not enough.
                step["outcome"] = "finish_denied"
                if direct:
                    step["observation"] = ("finish is not accepted yet: step 0 did not answer the question, and "
                                           "you have not searched yet. Search for the missing facts first.")
                else:
                    step["observation"] = ("finish is not accepted yet: you have not searched the knowledge base. "
                                           "You cannot know what it contains before searching. Search for the "
                                           "facts the question needs first.")
                self._record_step(step, t0, steps)
                messages.append(self._observation_msg(step_no, step["observation"]))
                continue
            if (step["action"] == "finish" and s.memory_policy != "off" and step["missing"]
                    and not redirected and not _recall_tried(steps)):
                # ---- guard: memory on + declared gap -> try memory first ---
                # qplan 0.1.1 run: the agent judged the found pairs
                # irrelevant, was told recall was allowed, and finished
                # anyway. Its own "missing" list now lets the code insist,
                # ONCE per run (a second finish is always accepted).
                redirected = True
                step["outcome"] = "finish_redirected"
                listed = "; ".join(step["missing"])
                step["observation"] = ("finish is not accepted yet: memory is on and you listed missing facts "
                                       f"({listed}), but you have not tried recall. Use recall for each missing "
                                       "fact (one short question per fact), then finish again.")
                self._record_step(step, t0, steps)
                messages.append(self._observation_msg(step_no, step["observation"]))
                continue
            if (step["action"] == "finish" and s.memory_policy == "off" and step["missing"]
                    and not redirected and step_no < s.max_steps):
                # ---- guard: memory off + declared gap -> search it once ----
                # HotpotQA run react30 (0.1.4): 6 of 16 bridge questions
                # stopped after one search with the second-hop fact missing
                # (e.g. "voiced by Ashley Tisdale" found, her show never
                # searched). With memory off there was no redirect at all.
                # Now the gap is searched once; a second finish is accepted.
                redirected = True
                step["outcome"] = "finish_redirected_search"
                listed = "; ".join(step["missing"])
                step["observation"] = ("finish is not accepted yet: you listed missing facts "
                                       f"({listed}). Search each missing fact once by name: name the entity "
                                       "the evidence points to (the person, work or place you found) and ask "
                                       "for the fact itself. Then finish again.")
                self._record_step(step, t0, steps)
                messages.append(self._observation_msg(step_no, step["observation"]))
                continue
            if step["action"] == "finish":
                store.missing = list(step["missing"] or [])
                step["outcome"] = "finish"
                step["observation"] = "(loop finished; evidence handed to LLM-S)"
                self._record_step(step, t0, steps)
                return "finish"
            if step["action"] == "search_kb":
                self._do_search(step, store, searched)
            else:
                self._do_recall(step, store)

            self._record_step(step, t0, steps)
            messages.append(self._observation_msg(step_no, step["observation"]))

        return "max_steps"

    # ------------------------------------------------------------------
    def _do_search(self, step: dict, store: EvidenceStore, searched: dict) -> None:
        """search_kb: one call of the shared retriever, results as [E#]."""
        query, step_no = step["input"], step["step_no"]
        key = _norm(query)
        if key in searched:
            step["outcome"] = "repeat"
            step["observation"] = (f"You already searched exactly this at step {searched[key]}. "
                                   "Use different wording, search another fact, or finish.")
            return
        searched[key] = step_no
        step["observation"] = self._run_search(step, query, self.s.top_k, store)

    # ------------------------------------------------------------------
    def _run_search(self, step: dict, query: str, top_k: int, store: EvidenceStore) -> str:
        """
        One call of the shared retriever: log it, put the matches into the
        evidence store as [E#], fill the step's outcome / hits / score, and
        return the observation text for the agent.
        Used by search_kb steps and by the direct-first step 0.
        """
        step_no = step["step_no"]
        r = self.db.retrieve(query, top_k, roles=self.s.roles, where=self._where)
        self.log.log_retrieval(self._run_id, step_no, r)
        step["n_hits"], step["best_score"] = len(r.hits), r.best_score

        matches = [h for h in r.hits if h.score >= self.s.min_score]
        if not matches:
            store.add_no_match(step_no, query, r.best_score)
            step["outcome"] = "no_match"
            best = f"{r.best_score:.2f}" if r.best_score is not None else "none"
            obs = (f"No match in the knowledge base (best score {best}, "
                   f"threshold {self.s.min_score:.2f}).")
            if self.s.memory_policy == "off":
                obs += " Rephrase the search once, search another fact, or finish."
            else:
                obs += " Rephrase the search once, or use recall for this fact."
            return obs

        step["outcome"] = "match"
        lines = [f"{len(matches)} matching pair(s):"]
        for h in matches:
            ev, is_new = store.add_hit(h, step_no, query)
            step["evidence_ids"].append(ev.ev_id)
            seen = "" if is_new else " (already found earlier)"
            src = h.title or h.doc_id
            line = f"[{ev.ev_id}] score {h.score:.2f}{seen} | Q: {h.question} | A: {h.answer}"
            text = source_text(h)            # react-v4: the source sentences, too
            if text:
                line += f' | text: "{text}"'
            lines.append(line + f" | source: {src}")
        below = len(r.hits) - len(matches)
        if below:
            lines.append(f"({below} weaker result(s) below the threshold were left out.)")
        # A high score means similar wording, not necessarily the right fact
        # (e.g. "favourite food" matches other facts about the same person).
        if self.s.memory_policy == "off":
            lines.append("Check that these pairs (and their text) really state the fact you searched for. "
                         "If none does, rephrase once, search another fact, or finish.")
        else:
            lines.append("Check that these pairs (and their text) really state the fact you searched for. "
                         "If none does, rephrase once; if the fact is still missing, you may use recall for it.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _do_recall(self, step: dict, store: EvidenceStore) -> None:
        """recall: ask the memory model, result as [M#] (unverified)."""
        policy = self.s.memory_policy
        if policy == "off":
            # Memory switched off: knowledge base only. (The prompt does not
            # even describe recall, so this only happens if the model guesses.)
            step["outcome"] = "denied"
            step["observation"] = "recall is not available in this run. Use search_kb or finish."
            return
        # on: the knowledge base must have been searched first (enforced
        # here). Whether the results really lacked the fact is the agent's
        # judgement; it is recorded in the outcome so misuse can be measured:
        #   "recall"             - an earlier search found no match
        #   "recall_after_match" - every earlier search had matches, the agent
        #                          judged them not to contain the fact
        searched_before = bool(store.no_match or store.kb)
        if policy == "on" and not searched_before:
            step["outcome"] = "denied"
            step["observation"] = ("recall is not allowed yet: it is only for facts the knowledge base "
                                   "does not contain. Search the knowledge base for this fact first.")
            return

        messages = [{"role": "system", "content": prompts.MEMORY_SYSTEM},
                    {"role": "user", "content": prompts.MEMORY_USER.format(question=step["input"])}]
        reply = self.llm.chat("memory", self.s.memory_model, messages,
                              self.s.memory_temperature, self.s.memory_max_tokens,
                              self.s.memory_reasoning)
        try:
            data = extract_json(reply.text)
            answer = str(data.get("answer", "")).strip()
            confidence = str(data.get("confidence", "unknown")).strip().lower()
        except ValueError:
            answer, confidence = reply.text.strip(), "unknown"

        knows = answer and confidence != "unknown" and "do not know" not in answer.lower() \
            and "don't know" not in answer.lower()
        if not knows:
            step["outcome"] = "recall_unknown"
            step["observation"] = "Model memory does not know the answer to this."
            return
        ev = store.add_memory(step["input"], answer, confidence, reply.model, step["step_no"])
        step["outcome"] = "recall" if store.no_match else "recall_after_match"
        step["evidence_ids"].append(ev.ev_id)
        step["observation"] = (f"[{ev.ev_id}] model memory (UNVERIFIED, confidence {confidence}): {answer}")

    # ------------------------------------------------------------------
    def _observation_msg(self, step_no: int, observation: str) -> dict:
        return {"role": "user", "content": prompts.REACT_OBSERVATION.format(
            step=step_no, observation=observation, steps_left=self.s.max_steps - step_no)}

    def _record_step(self, step: dict, t0: float, steps: list) -> None:
        step["ms"] = int((time.perf_counter() - t0) * 1000)
        steps.append(step)
        self.log.log_step(self._run_id, step)
        if step["outcome"] != "finish":
            self.print(f"[step {step['step_no']}] observation ({step['outcome']}):\n  "
                       + step["observation"].replace("\n", "\n  "))


def _missing_list(data: dict) -> list[str]:
    """
    Read the "missing" field of a finish reply as a clean list of phrases.

    Required with finish (react-v3). Accepted: a list of strings, one
    string, or an empty list. Absent -> ValueError, reported to the agent as
    an invalid reply so it learns the format.
    """
    if "missing" not in data:
        raise ValueError('finish needs a "missing" list: the facts still missing, or [] if nothing is missing')
    value = data.get("missing")
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError('"missing" must be a list of short phrases')
    return [str(v).strip() for v in value if str(v).strip()]


def _agent_searched(steps: list) -> bool:
    """True if the AGENT made at least one real search (the direct-first
    step 0 does not count)."""
    return any(s["action"] == "search_kb" and s["outcome"] in ("match", "no_match") for s in steps)


def _recall_tried(steps: list) -> bool:
    """True if a recall was actually carried out in this run (any result)."""
    return any(s["outcome"] in ("recall", "recall_after_match", "recall_unknown") for s in steps)


def _dumps(v) -> str:
    import json
    return json.dumps(v, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown trace of one run (for reading and judging by eye)
# ---------------------------------------------------------------------------
def render_trace(run: dict) -> str:
    """Turn one logged run (RunLog.get_run) into a readable Markdown page."""
    import json

    settings = json.loads(run.get("settings_json") or "{}")
    cost = run.get("cost_usd")
    cost_txt = f"${cost:.5f}" if cost is not None else "unknown"
    if run.get("cost_complete") == 0:
        cost_txt += " (some calls did not report a cost)"
    out = [
        f"# qplan run {run['run_id']}",
        "",
        f"**Question:** {run['question']}",
        "",
        "| item | value |", "|---|---|",
        f"| started | {run['started_at']} |",
        f"| database | `{run['db_dir']}` (fingerprint {run['db_fingerprint']}, embed {run['embed_model']}) |",
        f"| models | react `{settings.get('react_model')}`, memory `{settings.get('memory_model')}`, synth `{settings.get('synth_model')}` |",
        f"| reasoning | react {settings.get('react_reasoning', 'default')}, memory {settings.get('memory_reasoning', 'default')}, "
        f"synth {settings.get('synth_reasoning', 'default')}; agent budget {settings.get('react_max_tokens')} tokens |",
        f"| settings | max_steps {settings.get('max_steps')}, top_k {settings.get('top_k')}, "
        f"min_score {settings.get('min_score')}, memory_policy {settings.get('memory_policy')}, "
        f"direct_first {settings.get('direct_first', 'off')} (k {settings.get('direct_top_k', '-')})"
        f"{', DRY-RUN' if settings.get('dry_run') else ''} |",
        f"| direct-first | {_direct_line(run)} |",
        f"| prompts | {run.get('prompt_versions')} |",
        f"| result | status {run['status']}, stop {run['stop_reason']}, {run['n_steps']} steps, "
        f"{run['n_searches']} searches, {run['n_recalls']} recalls |",
        f"| cost | {run['llm_calls']} LLM calls, {run['prompt_tokens']} prompt + {run['completion_tokens']} "
        f"completion tokens, {run['embed_tokens']} embedding tokens, {cost_txt}, {run['ms']} ms |",
        "",
        "## Final answer", "",
        run.get("answer") or "_(no answer)_", "",
        f"- short answer: **{run.get('short_answer') or '-'}**",
        f"- status: {run.get('answer_status')}, basis: {run.get('answer_basis')}",
    ]
    if run.get("invalid_cites") not in (None, "[]"):
        out.append(f"- WARNING invalid citations (not in the evidence): {run['invalid_cites']}")
    if run.get("missing_json"):
        out.append(f"- facts the agent could not find: {', '.join(json.loads(run['missing_json']))}")
    if run.get("error"):
        out.append(f"- ERROR: {run['error']}")
    out += ["", "```", run.get("sources_text") or "", "```", "", "## ReAct steps", ""]

    calls_by_step: dict = {}
    for c in run["calls"]:
        calls_by_step.setdefault(c["step_no"], []).append(c)
    for st in run["steps"]:
        out.append(f"### Step {st['step_no']} - {st['action']} ({st['outcome']})")
        out.append("")
        out.append(f"- **Thought:** {st['thought'] or '-'}")
        out.append(f"- **Action:** `{st['action']}` - {st['action_input'] or ''}")
        if st.get("missing_json"):
            out.append(f"- **Missing (declared by the agent):** {', '.join(json.loads(st['missing_json']))}")
        if st["best_score"] is not None:
            out.append(f"- hits: {st['n_hits']}, best score {st['best_score']:.3f}")
        out.append("- **Observation:**")
        out.append("")
        out.append("```")
        out.append(st["observation"] or "")
        out.append("```")
        for c in calls_by_step.get(st["step_no"], []):
            if st["step_no"] == 0 and c["role"] == "synth":
                continue               # the final LLM-S call is listed at the end
            out.append(_call_line(c))
        out.append("")

    out += ["## Evidence collected", "", "| id | kind | score | cited | question | answer | found by |",
            "|---|---|---|---|---|---|---|"]
    for e in run["evidence"]:
        found = "; ".join(f"step {f['step']}: {f['query']}" for f in json.loads(e["found_by"] or "[]"))
        score = f"{e['score']:.3f}" if e["score"] is not None else "-"
        out.append(f"| {e['ev_id']} | {e['kind']} | {score} | {'yes' if e['cited'] else ''} | "
                   f"{_cell(e['question'])} | {_cell(e['answer'])} | {_cell(found)} |")
    out.append("")
    synth_calls = [c for c in calls_by_step.get(0, []) if c["role"] == "synth"]
    for c in synth_calls:
        out.append(_call_line(c).replace("- call:", "LLM-S call:", 1))
    out.append("")
    return "\n".join(out)


def _direct_line(run: dict) -> str:
    """One trace line on the direct-first step 0."""
    esc = run.get("escalated")
    if esc is None:
        return "not used"
    if esc == 0:
        return f"answered at step 0 (status {run.get('direct_status')}), no ReAct"
    return f"step 0 status {run.get('direct_status')} -> escalated to ReAct"


def _call_line(c: dict) -> str:
    """One LLM call in the trace: role, model, tokens, cost, time, and flags."""
    cost = f"${c['cost_usd']:.5f}" if c["cost_usd"] is not None else "cost unknown"
    line = (f"- call: {c['role']} `{c['model']}` {c['prompt_tokens']}+{c['completion_tokens']} tokens "
            f"(budget {c.get('max_tokens') or '?'}, reasoning {c.get('reasoning_setting') or 'default'}), "
            f"{cost}, {c['ms']} ms")
    if c.get("finish_reason") == "length":
        line += " - **CUT OFF by the token limit**"
    if c.get("retry"):
        line += " - retry with a larger budget"
    if c.get("reasoning_text"):
        line += f" - hidden reasoning {len(c['reasoning_text'])} chars (in the run log)"
    return line


def _cell(text) -> str:
    """Make text safe inside a Markdown table cell."""
    return str(text or "").replace("|", "/").replace("\n", " ")
