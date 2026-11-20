"""
runlog.py - qplan's own storage: one SQLite file with every run.

File: <runs_dir>\\qplan_runs.sqlite   (default F:\\qplan\\runs\\qplan_runs.sqlite)
This is qplan's OWN database. The indexer's database is never written to.

Tables (one row per ...):
    runs        question: settings, final answer, totals (tokens, cost, time)
    steps       ReAct cycle: thought, action, input, observation, outcome
    llm_calls   chat-model call: role, model, full prompt, raw reply, tokens, cost
    retrievals  search_kb call: query, top_k, roles, hits (pair_id + score)
    evidence    evidence item of a run ([E#] / [M#]) and whether it was cited

Why this much detail: the plan (handoff section 3) is to collect good runs
and later fine-tune smaller planner models on them, so every run must be
complete and exportable from day one. The `export` command writes them as
JSON Lines (one complete run per line).

Also written: a Markdown trace per run in <runs_dir>\\traces\\<run_id>.md, for
reading and judging a run by eye.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    started_at        TEXT,
    finished_at       TEXT,
    program_version   TEXT,
    prompt_versions   TEXT,     -- JSON {"react": "react-v1", ...}
    question          TEXT,
    db_dir            TEXT,
    db_fingerprint    TEXT,
    embed_model       TEXT,
    settings_json     TEXT,     -- every setting of the run
    dry_run           INTEGER,
    status            TEXT,     -- running | done | error
    stop_reason       TEXT,     -- finish | max_steps | invalid_replies | error
    n_steps           INTEGER,
    n_searches        INTEGER,
    n_recalls         INTEGER,
    answer            TEXT,
    short_answer      TEXT,
    answer_status     TEXT,     -- answered | partial | not_found
    answer_basis      TEXT,     -- kb | memory | mixed | none
    missing_json      TEXT,     -- facts the agent declared missing at its final finish
    filter_json       TEXT,     -- document filter applied to every search (ask --only-titles)
    escalated         INTEGER,  -- direct-first: 0 = answered at step 0, 1 = ReAct ran, NULL = not used
    direct_status     TEXT,     -- direct-first: LLM-S status at step 0 (answered | partial | not_found | no_match)
    cited_json        TEXT,
    invalid_cites     TEXT,
    sources_text      TEXT,
    llm_calls         INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    embed_tokens      INTEGER,
    cost_usd          REAL,
    cost_complete     INTEGER,  -- 1 if every call reported its cost
    ms                INTEGER,
    error             TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    run_id        TEXT,
    step_no       INTEGER,
    thought       TEXT,
    action        TEXT,         -- direct (step 0) | search_kb | recall | finish | (invalid)
    action_input  TEXT,
    observation   TEXT,
    outcome       TEXT,         -- match | no_match | repeat | recall | recall_after_match | recall_unknown | denied | invalid | finish_denied | finish_redirected | finish_redirected_search | finish
    n_hits        INTEGER,
    best_score    REAL,
    evidence_ids  TEXT,         -- JSON list of [E#]/[M#] produced by this step
    missing_json  TEXT,         -- finish steps: the facts the agent declared missing
    ms            INTEGER,
    PRIMARY KEY (run_id, step_no)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    step_no           INTEGER,  -- 0 = after the loop (LLM-S)
    role              TEXT,     -- react | memory | synth | synth_direct (direct-first step 0)
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL,
    cost_source       TEXT,
    ms                INTEGER,
    fake              INTEGER,
    messages_json     TEXT,     -- the full prompt sent
    raw_reply         TEXT,     -- the reply exactly as received (with <think>)
    finish_reason     TEXT,     -- stop = complete; length = cut off by max_tokens
    max_tokens        INTEGER,  -- the budget of this call (hidden reasoning + answer)
    reasoning_setting TEXT,     -- default | off | low | medium | high
    retry             INTEGER,  -- 1 = repeat of a call that was cut off
    reasoning_text    TEXT      -- the model's hidden reasoning, if returned
);
CREATE TABLE IF NOT EXISTS retrievals (
    retrieval_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    step_no       INTEGER,
    query         TEXT,
    top_k         INTEGER,
    roles_json    TEXT,
    n_raw         INTEGER,
    best_score    REAL,
    hits_json     TEXT,         -- [{"pair_id":..., "score":...}, ...] all hits, best first
    embed_model   TEXT,
    embed_tokens  INTEGER,
    embed_cost    REAL,
    ms            INTEGER
);
CREATE TABLE IF NOT EXISTS evidence (
    run_id      TEXT,
    ev_id       TEXT,           -- E1, E2, ... / M1, M2, ...
    kind        TEXT,           -- kb | memory
    pair_id     TEXT,
    score       REAL,
    question    TEXT,
    answer      TEXT,
    title       TEXT,
    source      TEXT,
    detail_json TEXT,
    found_by    TEXT,           -- JSON [{"step":..,"query":..,"score":..}]
    cited       INTEGER,
    PRIMARY KEY (run_id, ev_id)
);
"""


def new_run_id() -> str:
    """Readable, sortable, unique: 20261021-173205-a1b2."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class RunLog:
    """Writes the run log. One instance can log many runs (batch mode)."""

    def __init__(self, runs_dir: str | Path):
        self.dir = Path(runs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "traces").mkdir(exist_ok=True)
        self.path = self.dir / "qplan_runs.sqlite"
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        self._add_missing_columns()
        self.con.commit()

    def _add_missing_columns(self) -> None:
        """
        Upgrade a run log made by an older qplan version.

        CREATE TABLE IF NOT EXISTS does not add new columns to an existing
        table, so every column of SCHEMA that the file lacks is added here
        (ALTER TABLE ... ADD COLUMN). Old rows keep NULL in new columns.
        """
        import re
        for table, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", SCHEMA, re.S):
            have = {r["name"] for r in self.con.execute(f"PRAGMA table_info({table})")}
            for line in body.splitlines():
                line = line.split("--")[0].strip().rstrip(",")
                parts = line.split()
                if len(parts) >= 2 and parts[0] not in have and parts[0].upper() != "PRIMARY":
                    self.con.execute(f"ALTER TABLE {table} ADD COLUMN {parts[0]} {parts[1]}")

    def close(self) -> None:
        self.con.close()

    # ---- writing ----------------------------------------------------------
    def start_run(self, run_id: str, question: str, db_dir: str, fingerprint: str,
                  embed_model: str, settings: dict, prompt_versions: dict,
                  program_version: str) -> None:
        self.con.execute(
            "INSERT INTO runs (run_id, started_at, program_version, prompt_versions, question, "
            "db_dir, db_fingerprint, embed_model, settings_json, dry_run, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?, 'running')",
            (run_id, _now(), program_version, json.dumps(prompt_versions), question, db_dir,
             fingerprint, embed_model, json.dumps(settings), int(settings.get("dry_run", False))))
        self.con.commit()

    def log_llm_call(self, run_id: str, step_no: int, role: str, messages: list, reply) -> None:
        self.con.execute(
            "INSERT INTO llm_calls (run_id, step_no, role, model, prompt_tokens, completion_tokens, "
            "cost_usd, cost_source, ms, fake, messages_json, raw_reply, finish_reason, max_tokens, "
            "reasoning_setting, retry, reasoning_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, step_no, role, reply.model, reply.prompt_tokens, reply.completion_tokens,
             reply.cost, reply.cost_source, reply.ms, int(reply.fake),
             json.dumps(messages, ensure_ascii=False), reply.raw_text, reply.finish_reason,
             reply.max_tokens, reply.reasoning_setting, int(reply.retry_of_truncated),
             reply.reasoning))
        self.con.commit()

    def log_retrieval(self, run_id: str, step_no: int, r) -> None:
        hits = [{"pair_id": h.pair_id, "score": h.score} for h in r.hits]
        self.con.execute(
            "INSERT INTO retrievals (run_id, step_no, query, top_k, roles_json, n_raw, best_score, "
            "hits_json, embed_model, embed_tokens, embed_cost, ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, step_no, r.query, r.top_k, json.dumps(r.roles), r.n_raw, r.best_score,
             json.dumps(hits), r.embed_model, r.embed_tokens, r.embed_cost, r.ms))
        self.con.commit()

    def log_step(self, run_id: str, step: dict) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO steps (run_id, step_no, thought, action, action_input, observation, "
            "outcome, n_hits, best_score, evidence_ids, missing_json, ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, step["step_no"], step.get("thought"), step.get("action"), step.get("input"),
             step.get("observation"), step.get("outcome"), step.get("n_hits"),
             step.get("best_score"), json.dumps(step.get("evidence_ids", [])),
             json.dumps(step["missing"], ensure_ascii=False) if step.get("missing") is not None else None,
             step.get("ms")))
        self.con.commit()

    def finish_run(self, run_id: str, fields: dict, evidence: list, cited: list) -> None:
        """Store the final answer, totals and the evidence list."""
        for ev in evidence:
            self.con.execute(
                "INSERT OR REPLACE INTO evidence (run_id, ev_id, kind, pair_id, score, question, answer, "
                "title, source, detail_json, found_by, cited) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ev["ev_id"], ev["kind"], ev["pair_id"], ev["score"], ev["question"],
                 ev["answer"], ev["title"], ev["source"], json.dumps(ev["detail"], ensure_ascii=False),
                 json.dumps(ev["found_by"], ensure_ascii=False), int(ev["ev_id"] in cited)))
        totals = self.con.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(prompt_tokens),0) AS pt, "
            "COALESCE(SUM(completion_tokens),0) AS ct, COALESCE(SUM(cost_usd),0) AS cost, "
            "SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END) AS unknown "
            "FROM llm_calls WHERE run_id = ?", (run_id,)).fetchone()
        emb = self.con.execute(
            "SELECT COALESCE(SUM(embed_tokens),0) AS t, COALESCE(SUM(embed_cost),0) AS c, "
            "SUM(CASE WHEN embed_cost IS NULL THEN 1 ELSE 0 END) AS unknown "
            "FROM retrievals WHERE run_id = ?", (run_id,)).fetchone()
        fields = dict(fields)
        fields.update({
            "finished_at": _now(),
            "llm_calls": totals["n"],
            "prompt_tokens": totals["pt"],
            "completion_tokens": totals["ct"],
            "embed_tokens": emb["t"],
            "cost_usd": round(totals["cost"] + emb["c"], 6),
            "cost_complete": int(not totals["unknown"] and not emb["unknown"]),
        })
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.con.execute(f"UPDATE runs SET {cols} WHERE run_id = ?", [*fields.values(), run_id])
        self.con.commit()

    def write_trace(self, run_id: str, markdown: str) -> Path:
        path = self.dir / "traces" / f"{run_id}.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    # ---- reading ----------------------------------------------------------
    def get_run(self, run_id: str) -> dict | None:
        """One run with its steps, LLM calls, retrievals and evidence."""
        if run_id == "last":
            row = self.con.execute("SELECT run_id FROM runs ORDER BY started_at DESC, rowid DESC LIMIT 1").fetchone()
            if not row:
                return None
            run_id = row["run_id"]
        run = self.con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not run:
            return None
        out = dict(run)
        # The list of LLM calls is stored under "calls" because "llm_calls"
        # is already the name of the count column of the runs table.
        for key, table, order in (("steps", "steps", "step_no"), ("calls", "llm_calls", "call_id"),
                                  ("retrievals", "retrievals", "retrieval_id"),
                                  ("evidence", "evidence", "ev_id")):
            out[key] = [dict(r) for r in self.con.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {order}", (run_id,))]
        return out

    def list_runs(self, last: int = 20) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT run_id, started_at, question, status, stop_reason, n_steps, n_searches, "
            "n_recalls, answer_status, answer_basis, short_answer, missing_json, llm_calls, prompt_tokens, "
            "completion_tokens, embed_tokens, cost_usd, cost_complete, ms, dry_run "
            "FROM runs ORDER BY started_at DESC, rowid DESC LIMIT ?", (last,))]

    def all_run_ids(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT run_id FROM runs ORDER BY started_at, rowid")]
