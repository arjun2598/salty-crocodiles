#!/usr/bin/env python3
"""The loop. This is where the agent runs.

    python3 lab/driver.py --member kk --agent kk --run-id kk-003
    python3 lab/driver.py --member kk --agent kk --run-id kk-dev-1 --dev

Once per iteration:

    1. agent.select(state)       -> parent, stage        no LLM
    2. agent.formulate(...)      -> context string       no LLM
    3. agent.build(..., context) -> system, messages     assembles the call
    4. meter.complete(...)       -> a whole solution.py  <- THE ONLY LLM CALL
    5. executor.run(code)        -> scores -> evaluate() subprocess, timeout
    6. agent.on_failure(...)     -> retry/rollback       no LLM, on failure
    7. agent.judge(...)          -> accept / reject      no LLM
    8. journal + budget + convergence

Only steps 2-3 shape what the model sees. Everything else is a decision
function over run state. The driver is fixed infrastructure -- a bug here
invalidates every member's numbers at once, so changes get reviewed.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import executor  # noqa: E402
from core import DEV_LIMITS, LIMITS, Journal, Meter, Node, RunState, utcnow  # noqa: E402

MAX_API_FAILURES = 5      # consecutive LLM-call failures before aborting

CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
HYPOTHESIS = re.compile(r"#\s*HYPOTHESIS:\s*(.+)")


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def extract(reply: str) -> tuple[str | None, str]:
    """Take the LONGEST fenced block, not the first -- smaller models often
    show a snippet before the full file, and picking the first silently feeds
    a fragment to the executor, which then fails for the wrong reason."""
    blocks = [b.strip() for b in CODE_FENCE.findall(reply)]
    if not blocks:
        return None, "(no code block in reply)"
    code = max(blocks, key=len)
    h = HYPOTHESIS.search(code)
    return code, (h.group(1).strip() if h else "(no HYPOTHESIS line)")


def unified_diff(parent: Node | None, code: str) -> str:
    if parent is None:
        return ""
    return "".join(difflib.unified_diff(
        parent.code.splitlines(keepends=True), code.splitlines(keepends=True),
        fromfile="a/solution.py", tofile="b/solution.py"))


def converged(best_history: list[float], limits: dict) -> bool:
    """valid primary has not improved by more than epsilon over the last N.

    best_history holds one entry per SUCCESSFULLY EVALUATED iteration. A
    crashed iteration produces no validation score at all, so it cannot be
    "a validation score that failed to improve" -- counting it would end runs
    early on infrastructure noise rather than on genuine convergence.
    """
    n = limits["N"]
    if len(best_history) < n + 1:
        return False
    return best_history[-1] - best_history[-1 - n] <= limits["epsilon"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--agent", default="v0", help="agents/<name>.py")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dev", action="store_true",
                    help="8 iterations, short timeouts. Not evidence.")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data", "trainval"))
    ap.add_argument("--base-url", default=None, help="default: $LLM_BASE_URL")
    ap.add_argument("--model", default=None, help="default: $LLM_MODEL")
    args = ap.parse_args()

    limits = DEV_LIMITS if args.dev else LIMITS
    agent = importlib.import_module(f"agents.{args.agent}")

    run_dir = os.path.join(HERE, "runs", args.member, args.run_id)
    if os.path.exists(os.path.join(run_dir, "journal.jsonl")):
        sys.exit(f"{args.run_id} already has a journal. Re-using a run-id "
                 f"appends to it,\nproducing a file with two run_start records "
                 f"-- which breaks it as a\ndeliverable. Pick a new --run-id, "
                 f"or delete:\n  rm -rf {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    jr = Journal(run_dir, args.run_id)
    meter = Meter(base_url=args.base_url, model=args.model)
    print(f"agent: {args.agent}   model: {meter.label}")

    jr.run_start(member=args.member, agent=args.agent, git_sha=git_sha(),
                 model=meter.label, limits=limits, data_dir=args.data_dir,
                 seeds=agent.seeds_for(RunState(), "draft"), dev_mode=args.dev)

    state = RunState(iters_left=limits["max_iterations"],
                     seconds_left=limits["wall_clock_s"])
    best_history: list[float] = []
    attempt = api_failures = 0
    stop_reason = "aborted"

    while True:
        state.iters_left = limits["max_iterations"] - len(state.nodes)
        state.seconds_left = limits["wall_clock_s"] - meter.elapsed_s
        state.tokens_used = meter.total_tokens
        state.iteration = len(state.nodes) + 1

        if state.iters_left <= 0:
            stop_reason = "iteration_cap"; break
        if state.seconds_left <= 0:
            stop_reason = "wall_clock"; break
        if converged(best_history, limits):
            stop_reason = "converged"; break

        t0, started = time.time(), utcnow()

        parent, stage = agent.select(state)                          # 1
        context = agent.formulate(state, parent,                     # 2
                                  limits["context_budget_tokens"])
        system, messages = agent.build(state, parent, stage, context)  # 3

        # 4 -- THE LLM CALL. A retry here is not an iteration, so a
        # permanently broken credential would otherwise spin forever.
        try:
            reply, tin, tout = meter.complete(system, messages)
            api_failures = 0
        except Exception as exc:
            api_failures += 1
            print(f"[iter {state.iteration}] LLM call failed "
                  f"({api_failures}/{MAX_API_FAILURES}): {exc}")
            if api_failures >= MAX_API_FAILURES:
                print("giving up -- the credential or endpoint looks broken, "
                      "not busy. check your key / credits / LLM_BASE_URL.")
                stop_reason = "aborted"; break
            time.sleep(min(10 * 2 ** (api_failures - 1), 120))
            continue

        code, hypothesis = extract(reply)
        if code is None:
            code, hypothesis = "", "(model returned no code)"
        print(f"\n[iter {state.iteration}] {stage}"
              + (f" from {parent.id}" if parent else "")
              + f"\n  hypothesis: {textwrap.shorten(hypothesis, 150)}")

        # 5 -- run it
        node_id = f"n{state.iteration}"
        node_dir = os.path.join(run_dir, "nodes", node_id)
        seeds = agent.seeds_for(state, stage)
        leak = "pass"

        if not code:
            result = {"status": "error", "metrics": None,
                      "error": {"type": "NoCode", "message": hypothesis,
                                "traceback_tail": reply[:2000]}}
        else:
            per_seed, result = [], None
            for s in seeds:
                r = executor.run(code, node_dir, s,
                                 limits["solution_timeout_s"], args.data_dir)
                if r["status"] != "ok":
                    result = r; break
                per_seed.append(r["metrics"])
            if result is None:
                result = {"status": "ok",
                          "metrics": executor.mean_metrics(per_seed),
                          "error": None}
            if result["status"] == "leak":
                leak = "fail"

        node = Node(
            id=node_id, parent_id=parent.id if parent else None, stage=stage,
            hypothesis=hypothesis, code=code, diff=unified_diff(parent, code),
            metrics=result["metrics"],
            error=json.dumps(result["error"]) if result["error"] else None,
            tokens=(tin, tout), wall_clock_s=time.time() - t0,
            status=result["status"])

        recovery = {"action": "none", "attempt": 0, "resolved": True}
        if node.status != "ok":                                      # 6
            attempt += 1
            action = agent.on_failure(node, attempt, state)
            recovery = {"action": action, "attempt": attempt, "resolved": False}
            err = json.loads(node.error) if node.error else {}
            print(f"  {node.status}: {err.get('type','?')} "
                  f"{textwrap.shorten(err.get('message',''), 90)} -> {action}")
            if action == "abandon":
                state.nodes.append(node)
                jr.iteration(node=node, iteration=state.iteration,
                             seeds_run=seeds, recovery=recovery, leak_check=leak,
                             is_best_so_far=False, started_at=started,
                             ended_at=utcnow(), code_sha256=executor.sha256(code))
                stop_reason = "aborted"; break
        else:
            attempt = 0
            accept, why = agent.judge(node, state.best, state)        # 7
            node = Node(**{**node.__dict__, "accepted": accept})
            if accept:
                state.best_id = node.id
            m = node.metrics
            print(f"  GAUC {m['GAUC']:.4f}  nDCG@5 {m['nDCG@5']:.4f}  "
                  f"primary {m['primary']:.4f}  "
                  f"(baseline {LIMITS['baseline_primary']['valid']}, "
                  f"{m['primary'] - LIMITS['baseline_primary']['valid']:+.4f})")
            print(f"  {'ACCEPT' if accept else 'reject'} -- {why}")

        state.nodes.append(node)                                      # 8
        if node.status == "ok":        # scored iterations only -- see converged()
            best_history.append(state.best.primary if state.best else 0.0)
        jr.iteration(node=node, iteration=state.iteration, seeds_run=seeds,
                     recovery=recovery, leak_check=leak,
                     is_best_so_far=(state.best_id == node.id),
                     started_at=started, ended_at=utcnow(),
                     code_sha256=executor.sha256(code))

    # the scored artifact is the validation-best node AT CONVERGENCE
    final = state.best
    jr.run_end(stop_reason=stop_reason, iterations_used=len(state.nodes),
               final=final, totals=meter.totals(interventions=jr.interventions),
               submission_path=None)
    print(f"\nstopped: {stop_reason} after {len(state.nodes)} iterations")
    if final:
        base = LIMITS["baseline_primary"]["valid"]
        print(f"final: {final.id} valid primary {final.primary:.4f} "
              f"(baseline {base}, delta {final.primary - base:+.4f})")
    print(f"tokens: {meter.total_tokens:,}   wall: {meter.elapsed_s / 60:.1f} min")


if __name__ == "__main__":
    main()
