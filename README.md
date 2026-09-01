# Autonomous ML Research Agent for Recommender Systems

Built for TikTok TechJam 2026.

## Project Overview

Improving a recommender system normally involves a lot of manual experimentation: an ML engineer comes up with an idea, changes the model, trains it, evaluates the result, and decides what to try next.

Our project automates that research loop.

We built an **autonomous machine learning research agent** that can:

1. Inspect the current recommender and previous experiment results
2. Form a hypothesis about what to improve
3. Ask a coding LLM to implement the experiment
4. Run the generated recommender
5. Evaluate it using the official recommendation metrics
6. Decide whether the change is a meaningful improvement
7. Recover from failed experiments
8. Use what it learned to choose the next experiment

Instead of using an LLM as a one-shot code generator, we use it as one component inside a larger experimental system. The agent keeps state across iterations and uses previous hypotheses, code changes, metrics, and failures when deciding what to explore next.

## Problem

The challenge uses the **KuaiRand-Pure** recommendation dataset. The task is user-level ranking: for each user, the system ranks the videos that were exposed to that user.

- Target: `long_view`
- Metrics: **GAUC** and **nDCG@5**
- Primary score: mean of GAUC and nDCG@5

Official Factorization Machine baseline:

| Split | Primary |
|---|---:|
| Validation | 0.6016 |
| Test | 0.5946 |

The test-set oracle ceiling is approximately **0.8645**, rather than 1.0, because some users have no positive `long_view` labels and therefore cannot obtain a non-zero nDCG score.

## Our Approach

Our system separates the **research agent** from the **recommender being optimized**.

```text
Research Agent
      |
      v
Choose research direction
      |
      v
Form hypothesis
      |
      v
Coding LLM generates experiment
      |
      v
Run recommender
      |
      v
GAUC + nDCG@5 evaluation
      |
      v
Accept / Reject / Recover
      |
      └──────────────> Next iteration
```

The agent does not simply ask the model to “make the score better.” It reasons across research directions such as:

- Ranking-aligned loss functions
- Pairwise ranking / BPR
- User interaction history
- Multi-task learning
- Watch-time modelling
- Model architecture
- Temporal effects and distribution shift

The starter analysis showed that simply increasing embedding size or adding more static features did not meaningfully improve the model, so our agent prioritizes directions with more remaining headroom.

As the run progresses, the agent balances **exploration** and **exploitation**: early iterations can try different hypotheses, while later iterations focus more on approaches that already show useful signal.

## Failure Recovery

Generated ML code does not always work on the first attempt. Instead of treating a failed experiment as the end of the run, our agent records the failure and decides how to respond.

Possible recovery actions include:

- Retry with error context
- Debug the current research direction
- Roll back to the previous working model
- Abandon an unproductive direction

Failures remain part of the experiment history so the agent can avoid repeatedly making the same mistake.

## Validation and Convergence

The official FM baseline has a seed standard deviation of approximately `0.0008`.

The challenge convergence rule is:

```text
epsilon = 0.002
N = 3
```

A run converges when the best validation score has not improved by more than `0.002` over three consecutive iterations. The system preserves the **validation-best checkpoint**, rather than automatically using the final generated model.

| Limit | Value |
|---|---:|
| Maximum iterations | 50 |
| Maximum wall-clock time | 6 hours |
| Convergence epsilon | 0.002 |
| Convergence window | 3 iterations |

The lab records hypotheses, code diffs, validation metrics, errors, recovery actions, token usage, wall-clock time, and acceptance decisions for each experiment.

## Experimental Result

During development, our research agent found a pairwise-ranking approach that improved over the official FM validation baseline.

| Metric | Score |
|---|---:|
| GAUC | 0.6702 |
| nDCG@5 | 0.5372 |
| Primary | **0.6037** |

Official validation baseline: `0.6016`

Improvement: `+0.0021`

This result is reported on the **validation set**. The hidden test score is not visible to the agent during experimentation.

## Preventing Test Leakage

The autonomous agent is only allowed to access the training and validation portions of the data. The setup process creates:

```text
lab/data/trainval/
```

containing only permitted data. Hidden test rows are removed from the agent-facing environment, and the execution guard checks generated code for suspicious references to full-data or test-set paths.

## Project Structure

```text
salty-crocodiles/
├── baseline.py
├── baseline_scores.json
├── data.py
├── evaluate.py
├── submit.py
├── KuaiRand-Pure/
│   └── data/
├── lab/
│   ├── agents/
│   │   ├── v0.py
│   │   └── <member>.py
│   ├── core.py
│   ├── contract.py
│   ├── driver.py
│   ├── executor.py
│   ├── guard.py
│   ├── journal.py
│   ├── setup.sh
│   ├── selfcheck.py
│   ├── data/
│   │   └── trainval/
│   └── runs/
│       └── <member>/<run-id>/
│           ├── journal.jsonl
│           └── nodes/
├── SETUP.md
└── README.md
```

The root files are the original recommendation starter kit. Our autonomous experimentation framework lives primarily inside `lab/`.

# Setup and Installation

## 1. Requirements

Recommended: **Python 3.9+**

```bash
git clone https://github.com/arjun2598/salty-crocodiles.git
cd salty-crocodiles
```

## 2. Download KuaiRand-Pure

Download from https://kuairand.com or directly from Zenodo:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

After extraction, confirm that `KuaiRand-Pure/data/` exists.

## 3. Run Lab Setup

```bash
bash lab/setup.sh
```

This prepares the train/validation-only environment used by the autonomous agent.

## 4. Configure the LLM

Create a local `.env` file containing the API credentials required by the configured inference provider. **Never commit API keys to Git.**

Our experiments used the Swiss AI inference API with **NVIDIA Nemotron-3 Super 120B**. The LLM acts as the coding component; the surrounding agent controls research strategy, experiment selection, state, validation, and failure recovery.

## 5. Verify the Environment

```bash
python3 lab/selfcheck.py --full
```

The self-check verifies the dataset, train/validation construction, baseline reproduction, evaluation harness, and leakage protection. The official validation baseline should reproduce at approximately `0.6016`.

# Running the Agent

A development run can be started with:

```bash
python3 lab/driver.py --member <agent-name> --dev
```

For example:

```bash
python3 lab/driver.py --member shruthi --dev
```

The corresponding agent implementation lives in `lab/agents/shruthi_v2_2_minprompt_bprsafe.py`.

Each iteration generates a node under the run directory and records its result in:

```text
lab/runs/<member>/<run-id>/journal.jsonl
```

# Steps to Reproduce Results

```bash
bash lab/setup.sh
python3 lab/selfcheck.py --full
python3 lab/driver.py --member shruthi --dev
```

Then inspect:

```text
lab/runs/shruthi/<run-id>/journal.jsonl
```

The journal records the research hypothesis, parent experiment, generated diff, validation metrics, token usage, runtime, errors, recovery action, and acceptance/rejection decision.

Because the coding model is stochastic, individual generated experiments may differ between runs. The research policy and evaluation procedure remain fixed.

# Original Baseline

```bash
python3 baseline.py --model fm
python3 baseline.py --model pop
python3 baseline.py --model random
```

Approximate official test results:

| Model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Random | 0.4996 | 0.4511 | 0.4753 |
| Item popularity | 0.6308 | 0.5121 | 0.5715 |
| FM baseline | 0.6610 | 0.5282 | **0.5946** |

# Submission Format

```csv
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
```

Check a submission with:

```bash
python3 submit.py --check --split test submission.csv
```

Score validation predictions locally with:

```bash
python3 submit.py --score --split valid submission.csv
```

# Development Tools

We used:

- Python
- Cursor
- Git / GitHub
- NumPy
- TikTok TechJam's KuaiRand-Pure starter framework
- Swiss AI inference API
- NVIDIA Nemotron-3 Super 120B

# Dataset

We use the **KuaiRand-Pure** dataset. Relevant signals include user ID, video ID, long-view behaviour, clicks, likes, follows, comments, shares, watch time, and temporal information. Our primary target is `long_view`.

Dataset website: https://kuairand.com

# Limitations and Future Improvements

Our current system still has several limitations.

### LLM generation reliability

The coding model can occasionally produce incomplete or invalid Python code, and long responses can be truncated by output limits. We reduce the impact through shorter prompts, structured output requirements, rollback behaviour, and failure recovery.

### Limited experiment budget

The benchmark limits the system to 50 iterations and six hours, so the agent cannot exhaustively explore every possible modelling idea.

### Expensive experiments

Some approaches, especially pairwise ranking and more complex user-history models, are significantly slower than the baseline FM.

### Stochastic research process

Because an LLM generates experimental code, two runs of the same agent may not produce exactly the same experiment sequence.

### Future improvements

Given more time, we would explore better hard-negative sampling, hybrid pointwise + ranking objectives, user-history sequence models, multi-task learning, watch-time/censoring-aware objectives, more systematic architecture search, stronger duplicate-experiment detection, and better long-term research memory.


# Further Documentation

See `SETUP.md` for detailed environment setup, data-isolation rules, benchmark limits, and run structure.
