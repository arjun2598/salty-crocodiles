"""Runs a generated solution.py in a subprocess and scores it.

Two jobs:
  1. the leak check -- the trainval wall is a directory convention, and nothing
     in the OS stops generated code from opening ../KuaiRand-Pure/data/. This
     makes it enforced, and the result is journalled as evidence.
  2. run with a hard timeout, load the scores, hand them to the frozen referee.
"""
from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import subprocess
import sys
import tokenize

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluate import evaluate  # noqa: E402  (the referee. never edited)

import guard  # noqa: E402

# Anything that would reach around the wall. Extend, never relax.
#
# Scanned against the source with COMMENTS AND DOCSTRINGS STRIPPED, but every
# other string literal kept -- a path is a string, so strings must be scanned,
# while prose about the test split is harmless. Scanning prose was a false
# positive generator: it rejected the starter solution over its own docstring,
# and a false rejection costs a real iteration out of 50.
LEAK_PATTERNS = [
    r"KuaiRand-Pure",
    r"data[/\\]full",
    r"log_standard_4_22_to_5_08",
    r"\b20220429\b|\b2022050[0-9]\b",
    # subscripting a split dict with 'test'. guard has no such key, so this
    # would KeyError anyway -- catching it here just gives a clearer message.
    r"\[\s*['\"]test['\"]\s*\]",
]


def scannable(code: str) -> str:
    """Strip comments and docstrings; keep every other string literal.

    Paths are strings, so string literals must be scanned. Prose is not: both
    comments and docstrings routinely discuss the test split, and rejecting a
    solution over its own documentation costs a real iteration out of 50.
    """
    try:                                    # ast.unparse drops comments for us
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        return ast.unparse(tree)
    except (SyntaxError, ValueError, RecursionError):
        pass
    try:                                    # unparseable: at least drop comments
        toks = tokenize.generate_tokens(io.StringIO(code).readline)
        return tokenize.untokenize(
            [t for t in toks if t[0] != tokenize.COMMENT])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def leak_check(code: str) -> tuple[str, str | None]:
    """Return ('pass', None) or ('fail', which_pattern)."""
    scanned = scannable(code)
    for pat in LEAK_PATTERNS:
        if re.search(pat, scanned):
            return "fail", pat
    return "pass", None


def run(code: str, node_dir: str, seed: int, timeout_s: int,
        data_dir: str | None = None) -> dict:
    """Execute a generated solution.

    The solution is called as:
        python solution.py --data_dir DIR --out PATH.npy --seed N
    and must write a float array with one score per valid row, in guard order.

    Returns {"status", "metrics"|None, "error"|None}.
    """
    node_dir = os.path.abspath(node_dir)
    # the subprocess runs with cwd=node_dir, so a relative --data_dir would
    # resolve against the wrong directory
    data_dir = os.path.abspath(data_dir or guard.TRAINVAL_DIR)
    os.makedirs(node_dir, exist_ok=True)
    sol = os.path.join(node_dir, "solution.py")
    out = os.path.join(node_dir, f"scores_seed{seed}.npy")
    with open(sol, "w") as fh:
        fh.write(code)

    verdict, pattern = leak_check(code)
    if verdict == "fail":
        return {"status": "leak", "metrics": None,
                "error": {"type": "LeakCheck",
                          "message": f"source references {pattern!r}",
                          "traceback_tail": ""}}

    try:
        proc = subprocess.run(
            [sys.executable, sol, "--data_dir", data_dir,
             "--out", out, "--seed", str(seed)],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=node_dir, env={**os.environ, "PYTHONPATH": os.path.dirname(
                os.path.abspath(__file__))},
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "metrics": None,
                "error": {"type": "Timeout",
                          "message": f"exceeded {timeout_s}s",
                          "traceback_tail": ""}}

    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-25:]
        return {"status": "error", "metrics": None,
                "error": {"type": "NonZeroExit",
                          "message": f"exit {proc.returncode}",
                          "traceback_tail": "\n".join(tail)}}

    if not os.path.exists(out):
        return {"status": "error", "metrics": None,
                "error": {"type": "NoOutput",
                          "message": f"solution did not write {out}",
                          "traceback_tail": proc.stdout.strip()[-2000:]}}

    users, labels = guard.valid_targets(data_dir)
    scores = np.load(out)
    if len(scores) != len(labels):
        return {"status": "error", "metrics": None,
                "error": {"type": "Misaligned",
                          "message": f"got {len(scores)} scores, "
                                     f"expected {len(labels)}",
                          "traceback_tail": ""}}
    if not np.all(np.isfinite(scores)):
        return {"status": "error", "metrics": None,
                "error": {"type": "NonFinite",
                          "message": "scores contain NaN or Inf",
                          "traceback_tail": ""}}

    return {"status": "ok", "metrics": evaluate(users, labels, list(scores)),
            "error": None}


def mean_metrics(runs: list[dict]) -> dict:
    """Average evaluate() outputs across seeds."""
    keys = ("GAUC", "nDCG@5", "primary")
    n = len(runs)
    out = {k: sum(r[k] for r in runs) / n for k in keys}
    out["users"] = runs[0]["users"]
    out["rows"] = runs[0]["rows"]
    return out
