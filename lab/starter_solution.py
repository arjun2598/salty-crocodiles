"""The starter solution: the official FM, re-plumbed onto the solution contract.

WHY THIS ISN'T JUST baseline.py
  Same model, different plumbing. The root baseline.py cannot be run by
  executor.py, because it:
    - imports the root data.load(), which returns a `test` key (no wall)
    - indexes enc['test'] -- a KeyError under guard
    - calls evaluate() on itself -- the model scoring its own work
    - prints results instead of writing an array of scores
    - has its own CLI (--model), not --out
  This file fixes all five. It is what a valid solution.py looks like.

TWO JOBS
  1. Harness self-check. Run it through executor.py: if valid primary is not
     ~0.6016 the harness is broken, and you must fix that before believing any
     agent number. Costs zero API tokens.
  2. The reference the agent has to beat, and a template for its first draft.

Contract (see lab/README.md):
    python solution.py --data_dir DIR --out PATH.npy --seed N
  Writes one float per valid row, in guard.load() order. Nothing else.

Note for lane E: this early-stops on valid primary and is then scored on valid
primary, so 0.6016 is mildly optimistic. That is the official baseline's own
protocol, kept here for exact parity with the published number -- but it means
every agent solution inherits the same optimism. Worth deciding deliberately.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import guard  # train + valid only. there is no test split to reach for
from evaluate import evaluate  # legal: the wall is around TEST, not valid.
# A solution may score itself on validation -- that is just model selection,
# and the agent is expected to develop on train + validation. What makes
# evaluate.py the referee is that the DRIVER's number is the one that counts.


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Factorization Machine, Adam, logloss. k=16 / lr=0.001 are official."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                    # (B, F, k)
        S = E.sum(1)                                     # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (
                np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9)
                              + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate(
            [self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    a = ap.parse_args()

    splits = guard.load(a.data_dir)          # {'train', 'valid'}. no 'test'
    print({k: len(v) for k, v in splits.items()}, flush=True)

    enc, dim = guard.encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    bs, patience = 8192, 4
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
                  for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
              f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
              f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s", flush=True)
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}", flush=True)
                break

    m.V, m.W, m.b = best_state
    scores = m.predict(Xva).astype(np.float64)
    assert len(scores) == len(yva), f"{len(scores)} scores vs {len(yva)} rows"
    assert np.all(np.isfinite(scores)), "scores contain NaN/Inf"
    np.save(a.out, scores)
    print(f"wrote {len(scores)} scores -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
