#!/usr/bin/env python3
"""One command that answers 'is my setup correct?'. Costs zero API tokens.

    python3 lab/selfcheck.py           # everything except the FM (fast)
    python3 lab/selfcheck.py --full    # + run the starter FM (~30 s)

Run --full once after setup and again any time you touch shared
infrastructure. If the starter does not reproduce valid primary ~0.6016, the
harness is broken and every agent number after that point is meaningless --
fix this before believing anything else.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OFFICIAL_VALID_PRIMARY = 0.6016
TOLERANCE = 0.0016            # 2 sigma of the baseline's 5-seed std

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{' -- ' + detail if detail else ''}")
    if not ok:
        _fails.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the starter FM through the executor (~30 s)")
    ap.add_argument("--live", action="store_true",
                    help="also make one tiny real LLM call (a few tokens)")
    args = ap.parse_args()

    print("\n1. imports")
    try:
        import core, driver, executor, guard, providers  # noqa
        import agents.v0  # noqa
        check("all modules import", True)
    except Exception as exc:
        check("all modules import", False, repr(exc))
        sys.exit("\ncannot continue")

    print("\n2. the test wall")
    try:
        s = guard.load()
        check("train == 1,141,112 rows", len(s["train"]) == 1141112, f"got {len(s['train']):,}")
        check("valid == 124,909 rows", len(s["valid"]) == 124909, f"got {len(s['valid']):,}")
        check("no test key exists", "test" not in s)
    except FileNotFoundError as exc:
        check("data/trainval/ exists", False, f"{exc} -- run SETUP.md section 4")
    except RuntimeError as exc:
        check("no test rows leaked into trainval", False, str(exc))

    print("\n3. leak check")
    cases = [("clean solution", "import guard\nguard.load(d)", "pass"),
             ("prose about the test split", '"""never touches enc[\'test\']"""\nx=1', "pass"),
             ("opens the raw dataset", "open('../KuaiRand-Pure/data/x.csv')", "fail"),
             ("subscripts a test split", "rows = s['test']", "fail"),
             ("a test-window date", "if d >= 20220429: pass", "fail")]
    for name, code, want in cases:
        check(f"{name} -> {want}", executor.leak_check(code)[0] == want)

    print("\n4. convergence rule (eps=0.002, N=3)")
    L = core.LIMITS
    check("flat trajectory converges", driver.converged([.60, .60, .60, .60], L))
    check("rising trajectory does not", not driver.converged([.60, .61, .62, .63], L))
    check("sub-epsilon noise converges", driver.converged([.600, .6005, .601, .6015], L))
    check("too short to judge", not driver.converged([.60, .60], L))

    print("\n5. journal schema round-trip")
    d = tempfile.mkdtemp()
    jr = core.Journal(d, "selfcheck")
    jr.run_start(member="selfcheck", agent="v0", git_sha="0", model="none",
                 limits=L, data_dir="-", seeds=[0], dev_mode=True,
                 seed_baseline=True)
    n = core.Node(id="n1", parent_id=None, stage="draft", hypothesis="h",
                      code="pass", diff="", metrics={"GAUC": .1, "nDCG@5": .1,
                      "primary": .1, "users": 1, "rows": 1}, error=None,
                      tokens=(1, 1), wall_clock_s=1.0, status="ok", accepted=True)
    jr.iteration(node=n, iteration=1, seeds_run=[0],
                 recovery={"action": "none", "attempt": 0, "resolved": True},
                 leak_check="pass", is_best_so_far=True, started_at="t", ended_at="t",
                 code_sha256="x")
    jr.intervention(iteration=1, who="selfcheck", reason="r", action="a")
    jr.run_end(stop_reason="converged", iterations_used=1, final=n,
               totals={"input_tokens": 1, "output_tokens": 1, "wall_clock_s": 1.0,
                       "interventions": 1, "gpu_hours": 0.0}, submission_path=None)
    got = [json.loads(l)["record"] for l in open(os.path.join(d, "journal.jsonl"))]
    check("all four record types written",
          got == ["run_start", "iteration", "intervention", "run_end"], str(got))

    print("\n6. llm credentials")
    try:
        p = providers.build()
        check(f"llm ready -- {p.name}", True)
    except SystemExit as exc:
        check("llm reachable", False, str(exc).replace("\n", " "))
    except ImportError as exc:
        check("llm sdk installed", False, str(exc))
    except Exception as exc:
        # the Anthropic SDK also accepts an `ant auth login` profile, so a
        # missing ANTHROPIC_API_KEY is not by itself an error
        check("provider ready", False, f"{type(exc).__name__}: {exc}")

    if args.full:
        print("\n7. starter FM through the executor (~30 s)")
        res = executor.run(open(os.path.join(HERE, "starter_solution.py")).read(),
                           os.path.join(d, "starter"), seed=0,
                           timeout_s=1800, data_dir=guard.TRAINVAL_DIR)
        if res["status"] != "ok":
            check("starter runs", False, json.dumps(res["error"])[:200])
        else:
            p = res["metrics"]["primary"]
            check(f"reproduces official valid primary {OFFICIAL_VALID_PRIMARY}",
                  abs(p - OFFICIAL_VALID_PRIMARY) < TOLERANCE,
                  f"got {p:.4f} (delta {p - OFFICIAL_VALID_PRIMARY:+.4f})")
    else:
        print("\n7. starter FM -- skipped. re-run with --full")

    if args.live:
        print("\n8. one real LLM call")
        try:
            from core import Meter
            m = Meter()
            txt, tin, tout = m.complete("Reply with one word.",
                                        [{"role": "user", "content": "say ok"}])
            check(f"{m.label} answered", bool(txt.strip()),
                  f"{txt.strip()[:40]!r}  ({tin} in / {tout} out)")
        except Exception as exc:
            check("live call", False, f"{type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:160]}")
    else:
        print("\n8. live LLM call -- skipped. re-run with --live")

    print()
    if _fails:
        sys.exit(f"{len(_fails)} check(s) failed: {', '.join(_fails)}")
    print("all good. you are ready to run the driver.\n")


if __name__ == "__main__":
    main()
