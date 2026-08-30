"""Shared vocabulary and bookkeeping: types, limits, the meter, the journal.

Everything here is fixed infrastructure. Nobody tunes it -- it just has to be
correct. What you tune lives in agents/<yourname>.py.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import providers

Stage = Literal["draft", "improve", "debug"]
Status = Literal["ok", "error", "timeout", "leak"]
Recovery = Literal["none", "retry", "rollback", "abandon"]
StopReason = Literal["converged", "iteration_cap", "wall_clock", "aborted"]


# --------------------------------------------------------------------------
# Hard limits (problem statement 2.3 / 2.6). Enforced by driver.py.
# --------------------------------------------------------------------------

LIMITS = {
    "max_iterations":   50,       # hard cap per run
    "wall_clock_s":     21600,    # 6 h ceiling per run
    "epsilon":          0.002,    # convergence threshold on valid primary
    "N":                3,        # ...over this many consecutive iterations
    "baseline_primary": {"valid": 0.6016, "test": 0.5946},
    "seed_std":         0.0008,   # FM over 5 seeds. epsilon is only 2.5 sigma
    "oracle_primary":   {"valid": 0.8484, "test": 0.8645},
    "solution_timeout_s": 900,
    "context_budget_tokens": 12000,
}

DEV_LIMITS = {**LIMITS, "max_iterations": 8, "wall_clock_s": 3600,
              "solution_timeout_s": 300}


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """One completed iteration."""

    id: str
    parent_id: str | None
    stage: Stage
    hypothesis: str
    code: str
    diff: str
    metrics: dict | None          # evaluate() on valid; None if it failed
    error: str | None
    tokens: tuple[int, int]       # (input, output)
    wall_clock_s: float
    status: Status = "ok"
    accepted: bool = False

    @property
    def primary(self) -> float | None:
        return None if self.metrics is None else self.metrics["primary"]


@dataclass
class RunState:
    """Everything an agent policy is allowed to know."""

    nodes: list[Node] = field(default_factory=list)
    best_id: str | None = None
    iteration: int = 1
    iters_left: int = 0
    seconds_left: float = 0.0
    tokens_used: int = 0

    def by_id(self, node_id):
        return next((n for n in self.nodes if n.id == node_id), None) if node_id else None

    @property
    def best(self) -> Node | None:
        return self.by_id(self.best_id)

    @property
    def ok_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.status == "ok" and n.metrics]


# --------------------------------------------------------------------------
# Meter -- the only place an LLM is called
# --------------------------------------------------------------------------

@dataclass
class Meter:
    """Tokens and wall-clock. Feasibility is 15% of the score and is measured
    in exactly these two numbers -- neither can be reconstructed after a run,
    so every LLM call goes through complete() and nowhere else."""

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    max_tokens: int = 8192
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    started_at: float = field(default_factory=time.time)
    _p: object = None

    def __post_init__(self):
        self._p = providers.build(self.api_key, self.base_url, self.model)
        self.model = self._p.model

    def complete(self, system: str, messages: list[dict]) -> tuple[str, int, int]:
        text, tin, tout = self._p.complete(system, messages, self.max_tokens)
        self.input_tokens += tin
        self.output_tokens += tout
        self.calls += 1
        return text, tin, tout

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def label(self) -> str:
        """Endpoint + model. 'claude-opus-5' alone is ambiguous behind a proxy."""
        return self._p.name

    def totals(self, interventions: int = 0, gpu_hours: float = 0.0) -> dict:
        return {"input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "wall_clock_s": round(self.elapsed_s, 1),
                "interventions": interventions, "gpu_hours": gpu_hours}


def rough_tokens(text: str) -> int:
    """~4 chars/token. Good enough to budget a context block."""
    return len(text) // 4


# --------------------------------------------------------------------------
# Journal -- Deliverable 3. One JSON object per line, four record types.
# --------------------------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Journal:
    def __init__(self, run_dir: str, run_id: str):
        self.run_dir, self.run_id = run_dir, run_id
        os.makedirs(os.path.join(run_dir, "nodes"), exist_ok=True)
        self.path = os.path.join(run_dir, "journal.jsonl")
        self.interventions = 0

    def _write(self, rec: dict) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run_start(self, *, member, agent, git_sha, model, limits,
                  data_dir, seeds, dev_mode) -> None:
        self._write({
            "record": "run_start", "run_id": self.run_id, "member": member,
            "agent": agent, "started_at": utcnow(), "git_sha": git_sha,
            "model": model,
            "limits": {k: limits[k] for k in
                       ("max_iterations", "wall_clock_s", "epsilon", "N")},
            "data_dir": data_dir, "seeds": seeds, "dev_mode": dev_mode})

    def iteration(self, *, node: Node, iteration: int, seeds_run: list[int],
                  recovery: dict, leak_check: str, is_best_so_far: bool,
                  started_at: str, ended_at: str, code_sha256: str) -> None:
        self._write({
            "record": "iteration", "run_id": self.run_id, "iteration": iteration,
            "node_id": node.id, "parent_id": node.parent_id, "stage": node.stage,
            "hypothesis": node.hypothesis, "diff": node.diff,
            "code_sha256": code_sha256, "status": node.status,
            "metrics": {"valid": node.metrics} if node.metrics else None,
            "seeds_run": seeds_run,
            "error": json.loads(node.error) if node.error else None,
            "recovery": recovery,
            "tokens": {"input": node.tokens[0], "output": node.tokens[1]},
            "wall_clock_s": round(node.wall_clock_s, 1),
            "leak_check": leak_check, "accepted": node.accepted,
            "is_best_so_far": is_best_so_far,
            "started_at": started_at, "ended_at": ended_at})
        with open(os.path.join(self.run_dir, "nodes", f"{node.id}.py"), "w") as fh:
            fh.write(node.code)

    def intervention(self, *, iteration: int, who: str, reason: str,
                     action: str, lines_changed: int = 0) -> None:
        """Autonomy is 20% of the score. Call this the moment you touch a live
        run -- an honest 4 beats a reconstructed 0 the journal contradicts."""
        self.interventions += 1
        self._write({"record": "intervention", "run_id": self.run_id,
                     "iteration": iteration, "at": utcnow(), "who": who,
                     "reason": reason, "action": action,
                     "lines_changed": lines_changed})

    def run_end(self, *, stop_reason: StopReason, iterations_used: int,
                final: Node | None, totals: dict, submission_path) -> None:
        delta = None
        if final and final.metrics:
            delta = {"valid_primary": round(
                final.metrics["primary"] - LIMITS["baseline_primary"]["valid"], 4)}
        self._write({
            "record": "run_end", "run_id": self.run_id, "ended_at": utcnow(),
            "stop_reason": stop_reason, "iterations_used": iterations_used,
            "final_node_id": final.id if final else None,
            "final_metrics": {"valid": final.metrics} if final and final.metrics else None,
            "delta_vs_baseline": delta, "totals": totals,
            "submission_path": submission_path})
