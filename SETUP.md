# Experiment Lab — Setup

This document explains how to set up and run the autonomous research-agent environment for the KuaiRand-Pure recommender-system challenge.

The starter-kit files at the repository root are treated as reference infrastructure. The autonomous experimentation system lives inside `lab/`.

## 1. Important Evaluation Facts

The official task uses:

- Target: `long_view`
- Metrics: `GAUC` and `nDCG@5`
- Primary score: mean of GAUC and nDCG@5

Published reference scores:

| Model | Validation Primary | Test Primary |
|---|---:|---:|
| Random | 0.4834 | 0.4753 |
| Item popularity | 0.5807 | 0.5715 |
| **FM baseline** | **0.6016** | **0.5946** |
| Oracle ceiling | 0.8484 | 0.8645 |

The official evaluator in `evaluate.py` is authoritative.

The FM baseline seed standard deviation is approximately `0.0008`.

Challenge convergence parameters:

```python
epsilon = 0.002
N = 3
```

## 2. Repository Layout

```text
salty-crocodiles/
├── evaluate.py
├── submit.py
├── baseline_scores.json
├── data.py
├── baseline.py
├── ablation_features.py
├── README.md
├── SETUP.md
├── KuaiRand-Pure/
│   └── data/
└── lab/
    ├── agents/
    │   ├── v0.py
    │   └── <member>.py
    ├── contract.py
    ├── core.py
    ├── driver.py
    ├── executor.py
    ├── guard.py
    ├── journal.py
    ├── setup.sh
    ├── selfcheck.py
    ├── data/
    │   └── trainval/
    └── runs/
        └── <member>/<run-id>/
```

Each team member can implement an autonomous research strategy in:

```text
lab/agents/<member>.py
```

The driver, evaluator, execution environment, accounting, and dataset guard remain shared infrastructure.

## 3. Download KuaiRand-Pure

Download from https://kuairand.com or directly from Zenodo:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

After extraction, confirm that this exists:

```text
KuaiRand-Pure/data/
```

## 4. Environment

Python 3.9+ is recommended.

From the repository root:

```bash
python3 -m venv lab/.venv
source lab/.venv/bin/activate
```

Install the dependencies required by the repository. At minimum, the reference recommendation pipeline uses NumPy.

If `lab/requirements.txt` is present:

```bash
pip install -r lab/requirements.txt
```

## 5. API Credentials

The autonomous agent uses an external LLM as the coding component.

Our experiments used the Swiss AI inference endpoint with:

```text
NVIDIA Nemotron-3 Super 120B
```

Create a local `.env` file containing the API credentials and endpoint configuration expected by the current repository implementation.

**Never commit API keys to Git.**

## 6. Build the Train/Validation-Only Environment

The agent must never access hidden test labels.

Run:

```bash
bash lab/setup.sh
```

This creates the restricted agent-facing dataset under:

```text
lab/data/trainval/
```

The validation window ends on `20220428`. The test period beginning on `20220429` must not be available to generated experiment code.

The execution guard performs additional checks against suspicious full-data or hidden-test references.

## 7. Verify the Setup

Run:

```bash
python3 lab/selfcheck.py --full
```

The self-check should verify:

- Required files
- Dataset availability
- Train/validation construction
- Baseline reproduction
- Evaluation harness
- Test leakage protection

The official validation FM baseline should reproduce at approximately:

```text
primary ≈ 0.6016
```

If the baseline or dataset checks fail, fix the environment before running agent experiments.

## 8. How the Autonomous Loop Works

```text
Current best recommender
        |
        v
Agent chooses research direction
        |
        v
Agent formulates hypothesis
        |
        v
Coding LLM generates solution.py
        |
        v
Executor runs candidate
        |
        v
Evaluator computes GAUC / nDCG@5
        |
        v
Agent accepts, rejects, or recovers
        |
        └────────────> next iteration
```

The agent is responsible for research decisions. The coding LLM is responsible for generating the candidate implementation.

## 9. Run an Agent

Development mode:

```bash
python3 lab/driver.py --member <member> --dev
```

Example:

```bash
python3 lab/driver.py --member shruthi --dev
```

The driver loads:

```text
lab/agents/shruthi.py
```

and creates a run directory under:

```text
lab/runs/shruthi/<run-id>/
```

## 10. Development Mode

Development mode uses a smaller iteration/time budget than a full benchmark run. Use it while testing prompts, research policies, failure recovery, generated-code execution, and candidate agent strategies.

## 11. Benchmark Limits

The full benchmark uses:

```python
LIMITS = {
    "max_iterations": 50,
    "wall_clock_s": 21600,
    "epsilon": 0.002,
    "N": 3,
}
```

Important consequences:

- Failed iterations still consume iteration budget
- Runs stop at convergence even if fewer than 50 iterations were used
- The validation-best checkpoint is the important model, not necessarily the final generated node
- Token usage and wall-clock time should be recorded
- The agent must not access test data during the loop

## 12. Journaling

Each run records its experiment history in:

```text
lab/runs/<member>/<run-id>/journal.jsonl
```

A useful iteration record includes:

- Iteration number
- Parent node
- Research stage
- Hypothesis
- Generated code diff
- Validation metrics
- Error information
- Recovery action
- Token usage
- Wall-clock time
- Acceptance/rejection decision

This creates an auditable research trace rather than only storing the final score.

## 13. Failure Recovery

Generated experiments may fail because of invalid or incomplete Python, runtime exceptions, timeouts, invalid output format, data-access violations, or unproductive changes.

The agent can retry, roll back, abandon a direction, or switch into a debug stage. Failed experiments should remain part of the research history.

## 14. Convergence

The challenge convergence rule uses:

```text
epsilon = 0.002
N = 3
```

The run stops when the best validation score has failed to improve by more than `0.002` over three consecutive iterations.

## 15. Baseline Sanity Checks

Random baseline:

```bash
python3 baseline.py --model random
```

Expected test primary: `≈ 0.4753`

FM baseline:

```bash
python3 baseline.py --model fm
```

Expected test primary: `≈ 0.5946`

These checks use the full starter-kit dataset and are for harness verification only. The autonomous agent itself should use the restricted train/validation environment.

## 16. Submission Validation

Before submitting predictions:

```bash
python3 submit.py --check --split test submission.csv
```

For local validation scoring:

```bash
python3 submit.py --score --split valid submission.csv
```

Submission format:

```csv
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
```

## 17. Reproducing Our Development Result

Prepare the environment:

```bash
bash lab/setup.sh
python3 lab/selfcheck.py --full
```

Run Shruthi's agent:

```bash
python3 lab/driver.py --member shruthi --dev
```

Inspect:

```text
lab/runs/shruthi/<run-id>/journal.jsonl
```

A development run of this agent produced a pairwise-ranking candidate with approximately:

| Metric | Score |
|---|---:|
| GAUC | 0.6702 |
| nDCG@5 | 0.5372 |
| Primary | 0.6037 |

The official validation FM baseline is `0.6016`.

Because the coding model is stochastic, a fresh run may generate a different sequence of candidate implementations.

## 18. Common Issues

### `NoCode`

The LLM response did not contain a complete parseable solution. Retry with a shorter prompt, request only executable code, or roll back after repeated failures.

### Timeout

A generated experiment exceeded the allowed runtime. Reduce training work, simplify pair generation, or abandon an expensive direction.

### Runtime exception

The generated candidate crashed. Record the traceback and provide it as context to a debug/retry step.

### Baseline mismatch

If self-check cannot reproduce the reference baseline, do not trust subsequent experiment scores until the environment is fixed.

## 19. Safety and Reproducibility Notes

- Do not commit API credentials
- Do not expose test labels to the autonomous agent
- Do not modify the official evaluator to improve scores
- Keep experiment journals
- Preserve the best validation checkpoint
- Record failures rather than silently deleting them
- Treat small score changes carefully because the FM baseline has non-zero seed variance

## Done

A clean workflow is:

```text
1. Download KuaiRand-Pure
2. Configure local API credentials
3. Run lab/setup.sh
4. Run lab/selfcheck.py --full
5. Run a development agent
6. Inspect journal.jsonl
7. Promote promising strategies to larger benchmark runs
8. Preserve the validation-best checkpoint
9. Validate the final submission format
```
