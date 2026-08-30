# Experiment Lab — Setup

How to stand up the agent workspace for KuaiRand-Pure.

**Purely additive.** The starter-kit files at the repo root are treated as read-only.
Nothing in them is edited or moved; `lab/` is the only thing we create.

---

## 0. Read this first — the three facts people get wrong

1. **The metric is GAUC / nDCG@5 on `long_view`.** The "Limits" table in the TikTok
   information document says `NDCG@10 / Recall@50, click = positive`. That row is stale
   and contradicts every other section of the same document. `evaluate.py` is
   authoritative. Do not build against Recall@50.
2. **The ceiling is 0.8645, not 1.0.** 27.1% of test users have no positive label, so
   their nDCG is 0 for any model. Perfect ranking scores primary 0.8645 on test
   (0.8484 on valid). The FM baseline's 0.5946 already captures ~31% of the attainable
   range. Judge progress against the oracle, not against 1.0 — and make sure the
   agent's context says so, or it will read 0.60 as "failing badly" and thrash.
3. **What we beat is the organizer's FM baseline**, not a baseline we build ourselves.

| | valid primary | test primary |
|---|---|---|
| random (harness self-check) | 0.4834 | 0.4753 |
| item popularity | 0.5807 | 0.5715 |
| **FM — the bar** | **0.6016** | **0.5946** |
| oracle ceiling | 0.8484 | 0.8645 |

FM's std over 5 seeds is 0.0008 on test primary. Remember that number; §6 depends on it.

---

## 1. Layout

The starter kit lives at the repo root. `lab/` sits beside it.

```
salty-crocodiles/                    ← repo root
├── evaluate.py                      ← READ-ONLY. the referee. never edit
├── submit.py  baseline_scores.json  ← READ-ONLY
├── data.py  baseline.py             ← READ-ONLY reference implementations
├── ablation_features.py  README.md
├── KuaiRand-Pure/data/              ← the six CSVs (gitignored, already present)
└── lab/                             ← everything we build
    ├── contract.py                  ← the ONE file requiring team consensus
    ├── driver.py                    ← the iteration loop + budget enforcement
    ├── guard.py                     ← data loading; exposes train/valid only
    ├── journal.py                   ← writes journal.jsonl (schema in §7)
    ├── accounting.py                ← token + wall-clock meter
    ├── prompts/<member>.py          ← lane A
    ├── context/<member>.py          ← lane B
    ├── search/<member>.py           ← lane C
    ├── repair/<member>.py           ← lane D
    ├── validate/<member>.py         ← lane E
    ├── data/trainval/               ← the test wall (§4). gitignored
    ├── runs/<member>/<run_id>/      ← journals. COMMITTED
    └── final/                       ← integration runs + the submission we ship
```

```bash
mkdir -p lab/{prompts,context,search,repair,validate,data,final}
mkdir -p lab/runs
```

We do **not** copy `evaluate.py` into `lab/`. Copying invites drift between two
referees. Instead the driver verifies the root files are untouched before every run:

```bash
cd "$(git rev-parse --show-toplevel)"
shasum -a 256 evaluate.py submit.py baseline_scores.json > lab/FROZEN.sha256
```

`driver.py` re-checks this manifest at startup and refuses to run on a mismatch.
If someone legitimately needs to change a frozen file, that is a team decision and a
new manifest — not a silent edit.

---

## 2. What we are actually optimizing

The agent is a loop. Five parts of it are tunable, and they are **not** all prompts.
Each is one file behind one interface, so five people can move independently.

| Lane | Question it answers | Scored under |
|---|---|---|
| **A. Prompt** | How is the task described to the model? What does it produce? | Innovation 20% |
| **B. Context formulation** | Of everything that happened, what goes into *this* call — which past nodes, which diffs, which metrics, summarized how, inside what token budget? | Innovation 20% / Feasibility 15% |
| **C. Search policy** | Which node do we expand next, and when do we stop pushing an axis? *"When should the agent stop expanding features and switch to the loss function?"* lives here. | Innovation 20% / Autonomy 20% |
| **D. Repair policy** | A step failed — code error, timeout, bad output. Retry with the traceback, roll back to the parent, or abandon the branch? | Technical Execution 35% (Robustness) |
| **E. Validation & budget** | Is this candidate genuinely better, or seed noise? How many of the 50 iterations does this direction deserve? | Technical Execution 35% (Primary) |

**Lane E is the one people underrate.** The convergence rule is ε = 0.002 and the
baseline's seed std is 0.0008. A single-seed evaluation can produce a +0.0015 swing from
noise alone. Accept that as an improvement and you promote a worse checkpoint; reject
three real gains in a row and the run declares convergence and stops early. Deciding
how many seeds a candidate gets — and that costs iterations from a budget of 50 — is a
real research question, not plumbing.

Everything else (data loading, journaling, the referee, submission formatting) is
**fixed infrastructure**. Nobody optimizes it; it just has to be correct.

---

## 3. How five people work in parallel

### The interface

`lab/contract.py` is the only file the whole team must agree on. Write it in the first
session; freeze it after. Everything downstream is then conflict-free, because no two
people ever edit the same file.

```python
# lab/contract.py  — AGREE ONCE, THEN FREEZE
from dataclasses import dataclass
from typing import Literal

Stage = Literal["draft", "improve", "debug"]

@dataclass(frozen=True)
class Node:
    """One completed iteration."""
    id: str
    parent_id: str | None
    stage: Stage
    hypothesis: str
    code: str
    diff: str
    metrics: dict | None        # evaluate() output on valid; None if it failed
    error: str | None
    tokens: tuple[int, int]     # (input, output)
    wall_clock_s: float

@dataclass(frozen=True)
class RunState:
    nodes: list[Node]
    best_id: str | None
    iteration: int              # 1-based, the one about to run
    iters_left: int
    seconds_left: float
    tokens_used: int

# --- the five pluggable surfaces -------------------------------------------
# A  build_messages(state, parent, stage)      -> list[dict]
# B  formulate(state, parent, budget_tokens)   -> str
# C  select(state)                             -> tuple[Node | None, Stage]
# D  on_failure(node, attempt)                 -> Literal["retry","rollback","abandon"]
# E  judge(candidate, incumbent, state)        -> tuple[bool, str]   # (accept?, why)
```

### Ownership

One lane each. One file each. Nobody touches anyone else's file.

| Member | Lane | Owns | Also owns |
|---|---|---|---|
| 1 | A — prompt | `prompts/<name>.py` | — |
| 2 | B — context | `context/<name>.py` | — |
| 3 | C — search | `search/<name>.py` | — |
| 4 | D — repair | `repair/<name>.py` | — |
| 5 | E — validation | `validate/<name>.py` | `accounting.py`, `journal.py` |

Whoever finishes their v0 first builds `driver.py` and `guard.py`. Those are shared
infrastructure — treat any change to them as a PR the others review, because a driver
bug invalidates everyone's numbers simultaneously.

### The discipline that makes results comparable

Lanes interact — a better search policy can rescue a bad prompt, and a fat context can
hide a weak repair loop. So:

**Every member varies only their own module, against frozen `v0` defaults for the other
four.** Ship a deliberately plain `v0` of all five on day one and commit it. Your
experiment is then an ablation with a single moving part, and the numbers mean something.

```bash
python3 lab/driver.py --member kk --lane B \
    --prompt v0 --context kk_v3 --search v0 --repair v0 --validate v0 \
    --run-id kk-b-003
```

### Dev mode — do not burn the budget while developing

Five people × full 50-iteration runs is a lot of tokens for very little signal. During
lane development use a cheap configuration and only promote to full runs when a variant
looks genuinely better:

```bash
python3 lab/driver.py ... --dev        # 8 iterations, 20% user subsample, 1 seed
```

Dev runs go to `runs/<member>/dev-*/` and are **not** evidence. Only full runs are.

### Merging at the end

The point of one-file-per-lane is that the merge is a *selection*, not a diff resolution.

1. **Bake-off.** Each member nominates their single best variant, with the full-run
   journal that supports it.
2. **Integration runs.** Combine the five winners; run it. Then run four more
   configurations, each swapping one lane back to `v0` — that tells you which lanes
   actually contributed and gives you the ablation table for the write-up.
3. **Ship** the best integration run from `lab/final/`.

Budget two full days for stage 2. Combined winners routinely underperform their parts,
and you need runway to fall back to a simpler combination.

---

## 4. Build `data/trainval/` — the test wall

The agent must never see test labels. Give it a data directory where they do not exist.

**This is not a file copy.** The log files do not line up with the splits:

| File | Dates it contains | Action |
|---|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | 0408–0421 → all train | copy as-is (1,141,112 rows) |
| `log_standard_4_22_to_5_08_pure.csv` | 0422–0508 → **valid AND test** | filter to `date <= 20220428` |

Copying the second file whole leaks the test set. Omitting it deletes your validation
set. It has to be filtered.

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p lab/data/trainval
cp KuaiRand-Pure/data/user_features_pure.csv          lab/data/trainval/
cp KuaiRand-Pure/data/video_features_basic_pure.csv   lab/data/trainval/
cp KuaiRand-Pure/data/video_features_statistic_pure.csv lab/data/trainval/
cp KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv lab/data/trainval/

python3 - <<'PY'
import csv
jobs = [('log_standard_4_22_to_5_08_pure.csv', 124909),
        ('log_random_4_22_to_5_08_pure.csv',   288338)]   # unbiased valid, see below
for name, expect in jobs:
    src = f'KuaiRand-Pure/data/{name}'
    dst = f'lab/data/trainval/{name}'
    kept = 0
    with open(src) as fi, open(dst, 'w', newline='') as fo:
        r = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if int(row['date']) <= 20220428:          # valid window ends here
                w.writerow(row); kept += 1
    status = 'OK' if kept == expect else 'MISMATCH'
    print(f'{name}: kept {kept} (expected {expect}) [{status}]')
PY
```

**Both lines must print `OK`.** If either does not, stop and fix it — every validation
number produced afterwards is meaningless otherwise.

We include the filtered **random-exposure log** because it is the one genuinely unbiased
validation signal available (README §7). It is not part of the scored split; use it to
check whether a model is only winning on biased traffic. If you decide not to use it,
delete it deliberately rather than leaving it unmentioned.

`lab/guard.py` reads only from `lab/data/trainval/` and returns a dict with keys `train`
and `valid`. **There is no `test` key to reach for by accident.** The root `data.py`
still declares one; `guard.py` drops it.

### The wall is a convention — make it an enforced one

Nothing in the OS stops generated code from opening `../KuaiRand-Pure/data/`. Cheap
enforcement, and it doubles as evidence for judges: before executing any generated
solution, `driver.py` scans its source for references to `KuaiRand-Pure`, `data/full`,
`log_standard_4_22`, or `20220429..20220508`, and records the result as
`leak_check` in the journal. A hit fails the iteration and routes to lane D as an error.

---

## 5. Environment

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -m venv lab/.venv && source lab/.venv/bin/activate
pip install anthropic numpy
pip freeze > lab/requirements.txt
```

`numpy` is all the reference pipeline needs. Add `torch` / `lightgbm` only when a lane
actually requires it, and re-freeze when you do — five people on five different
dependency sets produce five incomparable runs.

### Credentials

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

That is the whole story. The `anthropic` package (1.2.0) ships **no CLI** — there is no
`ant auth login`. The SDK reads `ANTHROPIC_API_KEY` from the environment, so no key ever
appears in code. Put it in your shell profile, not in the repo. Each member needs their
own key; it is billed per account. Never commit one.

---

## 6. Limits — enforce these from day one

All of it is from the problem statement §2.3 and §2.6. Encode it in `driver.py` now;
retrofitting the accounting after a run is impossible.

```python
# lab/contract.py
LIMITS = {
    "max_iterations":   50,       # hard cap, per benchmark run
    "wall_clock_s":     21600,    # 6 h ceiling, per run
    "epsilon":          0.002,    # convergence threshold on valid primary
    "N":                3,        # ...over this many consecutive iterations
    "baseline_primary": {"valid": 0.6016, "test": 0.5946},
    "seed_std":         0.0008,   # FM, 5 seeds — epsilon is only 2.5 sigma of this
    "oracle_primary":   {"valid": 0.8484, "test": 0.8645},
}
```

| Limit | Value | Consequence if you ignore it |
|---|---|---|
| Iteration cap | 50, hard | Run is invalid past it |
| Wall clock | 6 h per run | Same |
| Convergence | valid primary not improving by > ε=0.002 over N=3 consecutive iterations | Run **ends**; this normally fires before the cap |
| Scored artifact | the **validation-best checkpoint at the moment of convergence** — not the last iteration, not the all-time peak | You submit the wrong checkpoint |
| Token accounting | input + output, every LLM call | Feasibility 15% is unscoreable; cannot be reconstructed afterwards |
| Wall-clock accounting | total elapsed per run | Same. Replaces GPU-hours as the scored compute measure |
| Intervention count | every manual touch, logged when it happens | Autonomy is 20%; a count written from memory the night before will be wrong |

Two of these are traps:

- **Token accounting must happen at the call site.** Wrap the client once, in
  `accounting.py`, and accumulate `resp.usage.input_tokens + resp.usage.output_tokens`
  into the run record. There is no way to recover this after the fact.
- **An intervention is any time a human touches a live run** — fixing a crash by hand,
  editing generated code, restarting with a nudge. Log it the moment it happens (§7).
  Fewer is better, and fully autonomous scores highest, but an honest 4 beats a
  reconstructed 0 that the journal contradicts.

Compute is deliberately *not* the binding constraint here — 100 iterations of the FM
baseline is ~28 min on one CPU core. The binding constraints are iteration count and
your own tokens.

---

## 7. `journal.jsonl` — exact schema

One file per run at `lab/runs/<member>/<run_id>/journal.jsonl`. One JSON object per
line. Four record types, discriminated by `record`. These keys are **fixed** — five
members producing five journal shapes turns the merge into a data-cleaning job.

All timestamps are ISO-8601 UTC (`2026-08-30T14:07:00Z`). `null` where a value does not
apply; never omit a key.

### `run_start` — exactly one, first line

```json
{"record":"run_start","run_id":"kk-b-003","member":"kk","lane":"B",
 "started_at":"2026-08-30T09:00:00Z","git_sha":"2217cdc","model":"claude-opus-5",
 "config":{"prompt":"v0","context":"kk_v3","search":"v0","repair":"v0","validate":"v0"},
 "limits":{"max_iterations":50,"wall_clock_s":21600,"epsilon":0.002,"N":3},
 "data_dir":"lab/data/trainval","seeds":[0],"dev_mode":false}
```

### `iteration` — one per iteration, including failed ones

```json
{"record":"iteration","run_id":"kk-b-003","iteration":7,
 "node_id":"n7","parent_id":"n4","stage":"improve",
 "hypothesis":"Pointwise logloss is misaligned with a ranking metric; switch to BPR over within-user pairs.",
 "diff":"--- a/solution.py\n+++ b/solution.py\n@@ ...",
 "code_sha256":"9f2b...",
 "status":"ok",
 "metrics":{"valid":{"GAUC":0.6731,"nDCG@5":0.5402,"primary":0.6067,"users":23158,"rows":124909}},
 "seeds_run":[0,1,2],
 "error":null,
 "recovery":{"action":"none","attempt":0,"resolved":true},
 "tokens":{"input":18422,"output":3110},
 "wall_clock_s":214.8,
 "leak_check":"pass",
 "accepted":true,
 "is_best_so_far":true,
 "started_at":"2026-08-30T09:41:02Z","ended_at":"2026-08-30T09:44:37Z"}
```

| Key | Notes |
|---|---|
| `iteration` | 1-based; counts against the 50 cap. Failed iterations count too |
| `stage` | `draft` \| `improve` \| `debug` |
| `hypothesis` | What this iteration intended to try **and why**. Required deliverable — one or two sentences, not a restatement of the diff |
| `diff` | Unified diff against the parent node's code. `""` for a `draft` |
| `status` | `ok` \| `error` \| `timeout` \| `leak` |
| `metrics` | `evaluate()` output on **valid**. `null` when `status != "ok"`. Mean over `seeds_run` |
| `seeds_run` | Which seeds this candidate was evaluated on — lane E's decision |
| `error` | `null`, or `{"type","message","traceback_tail"}` |
| `recovery` | `action`: `none` \| `retry` \| `rollback` \| `abandon`. Lane D's decision |
| `tokens` | This iteration only. The run total is the sum |
| `leak_check` | `pass` \| `fail` — the §4 source scan |
| `accepted` | Lane E's verdict: did this become the new incumbent? |

### `intervention` — one per manual touch, written when it happens

```json
{"record":"intervention","run_id":"kk-b-003","iteration":12,
 "at":"2026-08-30T10:15:00Z","who":"kk",
 "reason":"Agent looped on the same import error for 3 attempts; repair policy had no rule for missing deps.",
 "action":"pip install lightgbm, restarted from n11",
 "lines_changed":0}
```

### `run_end` — exactly one, last line

```json
{"record":"run_end","run_id":"kk-b-003","ended_at":"2026-08-30T13:20:11Z",
 "stop_reason":"converged","iterations_used":19,
 "final_node_id":"n14",
 "final_metrics":{"valid":{"GAUC":0.6802,"nDCG@5":0.5471,"primary":0.6137}},
 "delta_vs_baseline":{"valid_primary":0.0121},
 "totals":{"input_tokens":412880,"output_tokens":66190,
           "wall_clock_s":15611.0,"interventions":1,"gpu_hours":0.0},
 "submission_path":"lab/runs/kk/kk-b-003/submission.csv"}
```

`stop_reason` is `converged` | `iteration_cap` | `wall_clock` | `aborted`.
`final_node_id` **must** be the validation-best node at the point of convergence.

This schema covers Deliverable 3 (hypothesis / diff / metrics / error-and-recovery, plus
the intervention count) and feeds Deliverable 4's resource table directly from
`run_end.totals`.

---

## 8. `.gitignore`

The current file ignores `runs/` and `*.csv` wholesale, which silently excludes the
journals — a graded deliverable. Append:

```gitignore
# ---- Lab -------------------------------------------------------------------
lab/data/
lab/.venv/
lab/runs/*/*/nodes/          # generated solution.py per node — large, not graded
lab/runs/*/dev-*/            # dev-mode runs are not evidence

# ...but journals and intervention logs ARE deliverables
!lab/runs/
!lab/runs/**/journal.jsonl
!lab/runs/**/interventions.md
```

Verify it worked — this must list your journals:

```bash
git check-ignore -v lab/runs/kk/kk-b-003/journal.jsonl || echo "tracked, good"
```

Keep the final `submission.csv` out of git (`submission*.csv` already covers it) and
attach it to the Devpost submission instead.

---

## 9. Verify

```bash
cd "$(git rev-parse --show-toplevel)"
source lab/.venv/bin/activate

python3 -c "import anthropic, numpy; print('deps ok')"
shasum -a 256 -c lab/FROZEN.sha256

python3 -c "
import sys; sys.path.insert(0, '.')
from data import load
s = load('lab/data/trainval')
print({k: len(v) for k, v in s.items()})
"
```

Expected: `{'train': 1141112, 'valid': 124909, 'test': 0}`

`test: 0` is the whole point — the rows are **gone**, not hidden.

Optional, and worth doing once: confirm the referee reproduces the published numbers.

```bash
python3 baseline.py --data_dir ./KuaiRand-Pure/data --model random
# primary ~0.4753 (+/-0.001) on test. If not, the harness is wrong -- fix that first

python3 baseline.py --data_dir ./KuaiRand-Pure/data --model fm
# test primary ~0.5946, matching baseline_scores.json
```

This is the only step that reads the full data. The agent never runs it.

---

## Done

Nothing in the starter kit was touched. Next, in this order:

1. **Together:** agree and freeze `contract.py` (§3). Nothing else can start first.
2. **Whoever is free:** `guard.py`, `journal.py`, `accounting.py`, `driver.py` — with the
   §6 limits and §7 schema wired in from the first commit.
3. **Everyone:** write a plain `v0` of your lane, commit it, and confirm a `--dev` run
   produces a well-formed journal end to end.
4. **Then** start actually optimizing.
