"""Data access for the agent. Exposes train + valid. There is no test key.

The root data.py declares a test split; this drops it. Combined with
lab/data/trainval/ (which physically contains no test-date rows), there is
nothing for generated code to reach for by accident.

Generated solutions import THIS, never the root data.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import data as _reference  # noqa: E402  (root data.py, read-only)

TRAINVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "trainval")

LABEL = _reference.LABEL          # 'long_view'
FIELDS = list(_reference.FIELDS)  # the 5 baseline fields; solutions may extend


def load(data_dir: str | None = None) -> dict:
    """Return {'train': [...], 'valid': [...]}. No test key. Ever."""
    splits = _reference.load(data_dir or TRAINVAL_DIR)
    leaked = splits.pop("test", [])
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} test-date rows found in {data_dir or TRAINVAL_DIR}. "
            "The trainval directory was built wrong -- see SETUP.md section 4. "
            "Every validation number from this directory is meaningless."
        )
    return splits


def encode(splits: dict):
    """Reference encoder, with one fix: `users` comes back as a numpy array.

    The root data.py returns X and y as numpy but users as a Python list. Any
    solution doing batched per-user work -- which is what BPR and listwise
    losses need -- then hits `TypeError: only integer scalar arrays can be
    converted to a scalar index` on users[batch_idx]. Making the three arrays
    consistent removes a whole class of failure that has nothing to do with
    the research question.
    """
    enc, dim = _reference.encode(splits)
    return {k: (X, y, np.asarray(users)) for k, (X, y, users) in enc.items()}, dim


def valid_targets(data_dir: str | None = None) -> tuple[list, list]:
    """(user_ids, labels) for the valid split, in canonical row order.

    The driver uses this to score whatever a solution writes out. Row order is
    deterministic: log_standard_4_08 then log_standard_4_22, original file order
    within each, after date filtering.
    """
    valid = load(data_dir)["valid"]
    return [r[1] for r in valid], [r[6] for r in valid]


if __name__ == "__main__":
    s = load()
    print({k: len(v) for k, v in s.items()})
    assert len(s["train"]) == 1141112, f"train={len(s['train'])}, expected 1141112"
    assert len(s["valid"]) == 124909, f"valid={len(s['valid'])}, expected 124909"
    print("guard ok -- no test key present")
