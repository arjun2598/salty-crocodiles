# Experiment Lab

An autonomous ML research agent for KuaiRand-Pure. Five people, five agents,
one merge at the end.

The starter-kit files at the repo root are **read-only**. Everything we build
is in `lab/`.

---

## Setup

`numpy` is the only dependency. No LLM SDK — both APIs are one HTTP POST, so
`providers.py` uses the standard library.

```bash
bash lab/setup.sh                    # checks deps, gets data, builds the test
                                     # wall, writes lab/.env. idempotent
# put your key in lab/.env
python3 lab/selfcheck.py             # every line must be [ok]
python3 lab/selfcheck.py --full      # ~30 s, no tokens. proves the harness works
```

`--full` must print `reproduces official valid primary 0.6016`. If it doesn't,
stop — the harness is broken and every number after it is meaningless.

### Any key works — `lab/.env`

`setup.sh` writes the template. The provider is inferred from the key prefix
(`sk-ant-` Anthropic, `sk-or-` OpenRouter, `sk-` OpenAI).

```ini
# lab/.env -- gitignored. NEVER commit this file.
LLM_API_KEY=sk-ant-...
```

Custom endpoint (research server, vLLM, LM Studio) — set the URL and it stops
guessing:

```ini
LLM_API_KEY=<the key>
LLM_BASE_URL=https://api.swissai.cscs.ch/v1
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

A real exported environment variable always wins over `.env`, so you can
override one value for a single run without editing the file. With no key at
all, the Anthropic SDK falls back to an `ant auth login` profile.

> **The model is a variable, not a detail.** Develop on whatever key you have,
> but every run that counts as *evidence* — and all the integration runs — must
> use one agreed model. Otherwise the model difference swamps the difference
> you are trying to measure. `run_start.model` records endpoint + model, so a
> mismatch is visible rather than silent.

---

## How it works

`driver.py` owns the loop. **One LLM call per iteration**, at step 4:

```
1. agent.select(state)        -> parent, stage        no LLM
2. agent.formulate(...)       -> context string       no LLM
3. agent.build(..., context)  -> system, messages     assembles the call
4. meter.complete(...)        -> a whole solution.py  <- THE ONLY LLM CALL
5. executor.run(code)         -> scores -> evaluate() subprocess, timeout
6. agent.on_failure(...)      -> retry/rollback       no LLM, on failure
7. agent.judge(...)           -> accept / reject      no LLM
8. journal + budget + convergence
```

The model returns a complete `solution.py` in one fenced block. The driver
writes it to disk and runs it:

```bash
python solution.py --data_dir lab/data/trainval --out scores.npy --seed 0
```

It writes one float per valid row; the driver scores that with the frozen
`evaluate.py`. **The model never scores itself.**

Note this is the *harness* loop. The five MLE stages from the problem statement
(problem understanding, EDA, feature engineering, training, validation) all
happen inside step 4, in one shot. Making them explicit separate steps is a
legitimate redesign and a stronger Innovation story — it just isn't what this
is today.

---

## Files

| | |
|---|---|
| `lab/agents/<you>.py` | **Your file.** Six functions: select, formulate, build, seeds_for, judge, on_failure |
| `lab/core.py` | Types, `LIMITS`, the token meter, the journal writer |
| `lab/driver.py` | The loop + budget enforcement |
| `lab/guard.py` | `load()` returns train + valid. **No test key** |
| `lab/executor.py` | Leak check, subprocess, timeout, scoring |
| `lab/providers.py` | Any key in, LLM out |
| `lab/starter_solution.py` | The official FM on the solution contract. Reproduces 0.6016 |
| `lab/selfcheck.py` | "Is my setup correct?" — zero tokens |
| `lab/.env` | Your key. Gitignored, never committed |

Everything except `agents/` is fixed infrastructure — nobody tunes it, it just
has to be correct. A bug in `driver.py` invalidates all five members' numbers
at once, so changes there get reviewed.

---

## Five people

```bash
cp lab/agents/v0.py lab/agents/kk.py      # your name, your file
```

One file each. Nobody edits anyone else's, so there is nothing to merge until
the end. **`v0.py` stays untouched** — it is the shared control that everyone's
experiment is measured against.

Each of the six functions in `v0.py` has a comment saying what it deliberately
does *not* do. That list is your backlog. Suggested split so five people aren't
all editing prompts:

| | Focus | The question |
|---|---|---|
| 1 | `build` + `SYSTEM` | How is the task described? |
| 2 | `formulate` | Of everything that happened, what goes into *this* call? |
| 3 | `select` | What next — and when do we stop pushing an axis? |
| 4 | `on_failure` | It crashed. Retry, roll back, or abandon? |
| 5 | `seeds_for` + `judge` | Is this delta real, or seed noise? |

Change only your own functions; leave the others at v0. One moving part, or
the comparison means nothing.

**At the end:** each person nominates their best agent, then run the combination
plus one run per function reverted to v0. That gives you the ablation table for
the write-up.

---

## Running

```bash
# cheap. 8 iterations, 1 h ceiling. NOT evidence
python3 lab/driver.py --member kk --agent kk --run-id kk-dev-1 --dev

# real. 50 iterations / 6 h
python3 lab/driver.py --member kk --agent kk --run-id kk-003
```

Watch it:

```bash
python3 -c "
import json
for l in open('lab/runs/kk/kk-003/journal.jsonl'):
    r = json.loads(l)
    if r['record'] == 'iteration':
        p = r['metrics']['valid']['primary'] if r['metrics'] else None
        print(f\"{r['iteration']:3d} {r['status']:8s} {p or '--':>8} {r['hypothesis'][:60]}\")
    elif r['record'] == 'run_end':
        print(r['stop_reason'], r['totals'])
"
```

---

## The limits (problem statement 2.3 / 2.6)

All enforced in `core.LIMITS`.

| Limit | Value |
|---|---|
| Iterations | 50, hard cap |
| Wall clock | 6 h per run |
| Convergence | valid primary not improving by > ε=0.002 over N=3 consecutive iterations |
| Scored artifact | the validation-**best** node at the moment of convergence — not the last, not the all-time peak |
| Tokens | input + output, every call. Feasibility is 15% and this can't be reconstructed later |
| Interventions | log each one **when it happens**. Autonomy is 20% |

Two things people get wrong:

1. **The ceiling is 0.8645, not 1.0.** 27.1% of test users have no positive
   label, so their nDCG is 0 for any model. FM's 0.5946 already captures ~31%
   of the attainable range.
2. **The metric is GAUC / nDCG@5 on `long_view`.** The "Limits" table in the
   TikTok doc says `NDCG@10 / Recall@50, click = positive`. That row is stale
   and contradicts the rest of the same document. `evaluate.py` is authoritative.

| | valid primary | test primary |
|---|---|---|
| random | 0.4834 | 0.4753 |
| item popularity | 0.5807 | 0.5715 |
| **FM — the bar** | **0.6016** | **0.5946** |
| oracle ceiling | 0.8484 | 0.8645 |

---

## journal.jsonl

Four record types, one JSON object per line. `run_start`, `iteration`,
`intervention`, `run_end`. The exact keys are in `core.py` — `Journal` is the
only writer, so the schema can't drift between members.

This is Deliverable 3 (hypothesis / diff / metrics / errors / interventions),
and `run_end.totals` is Deliverable 4's resource table.

Log an intervention the moment you touch a live run:

```python
jr.intervention(iteration=12, who="kk",
                reason="looped on the same import error 3x",
                action="pip install lightgbm, restarted from n11")
```

---

## The test wall

`lab/data/trainval/` physically contains no test-date rows, and `guard.load()`
returns no `test` key. `setup.sh` builds it and asserts the row counts
(124,909 and 288,338) — if those don't match, stop.

`executor.py` also scans generated source for references to the raw dataset
directory or test-window dates, with comments and docstrings stripped so prose
about the test split doesn't false-positive. The result is journalled as
`leak_check`.

---

## When it breaks

| Symptom | Cause |
|---|---|
| `guard` raises on test rows | `data/trainval/` built wrong. Re-run `lab/setup.sh` |
| `selfcheck --full` ≠ ~0.6016 | Harness is broken. Fix before believing any run |
| `llm ready` fails | Set `LLM_API_KEY` in `lab/.env` (+ `LLM_BASE_URL` for a custom endpoint) |
| `not a prefix this can route` | Unrecognised key. Set `LLM_BASE_URL` — it refuses to guess rather than mail your key somewhere |
| `CERTIFICATE_VERIFY_FAILED` | Your Python's CA bundle lacks the endpoint's CA. `pip install certifi` (used automatically). Otherwise `LLM_CA_BUNDLE=/path/ca.pem` in `lab/.env` |
| Every iteration `status: leak` | Your prompt is telling the model to open the raw dataset |
| Converged at iteration 4 | `judge` accepted noise. ε is only 2.5σ of the seed std |
| `LLM call failed (5/5)` | Dead credential or wrong base URL. The driver aborts rather than spinning |
