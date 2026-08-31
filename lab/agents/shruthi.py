"""The baseline agent. Copy this file to agents/<yourname>.py and change it.

ONE FILE = ONE PERSON. Nobody edits anyone else's, so there is nothing to merge
until the end. v0 stays untouched: it is the shared control that everyone's
experiment is measured against.

Six functions, in the order the driver calls them each iteration:

    select      what do we work on next?            no LLM
    formulate   what does the model get to see?     no LLM
    build       how do we ask?                      assembles the call
    ---- the driver makes THE ONE LLM CALL here ----
    seeds_for   how many seeds does it earn?        no LLM
    judge       real improvement, or seed noise?    no LLM
    on_failure  it crashed -- now what?             no LLM

v0 is deliberately plain. Each function's comment says what it does NOT do --
that list is your backlog.
"""
from __future__ import annotations

from core import LIMITS, Node, RunState, Stage, rough_tokens
import json

BASE = LIMITS["baseline_primary"]["valid"]      # 0.6016 -- the bar
CEILING = LIMITS["oracle_primary"]["valid"]     # 0.8484 -- perfect ranking

RESEARCH_AXES = [
    "loss",
    "user_history",
    "multi_task",
    "watch_time",
]


def _axis_from_hypothesis(hypothesis: str) -> str | None:
    """Infer which research axis a previous experiment investigated."""
    text = (hypothesis or "").lower()

    keywords = {
        "loss": [
            "loss", "bpr", "pairwise", "listwise", "ranking objective",
        ],
        "user_history": [
            "user history", "history", "sequence", "din", "sim",
            "behavioural", "behavioral",
        ],
        "multi_task": [
            "multi-task", "multitask", "auxiliary", "click", "like",
            "follow", "comment", "forward",
        ],
        "watch_time": [
            "watch time", "watch-time", "play time", "play_time",
            "censored", "duration",
        ],
    }

    for axis, words in keywords.items():
        if any(word in text for word in words):
            return axis

    return None


def research_axis(state: RunState, parent: Node | None = None) -> str:
    """Choose the next research direction using evidence-aware explore -> exploit."""

    # DEBUG:
    # A crash is not a new ML experiment.
    # Keep trying to repair the same research idea.
    if parent is not None and parent.status != "ok":
        previous_axis = _axis_from_hypothesis(parent.hypothesis)
        if previous_axis:
            return previous_axis

    # Collect the scores achieved by each research direction.
    results = {axis: [] for axis in RESEARCH_AXES}

    for node in state.nodes:
        if node.status != "ok" or node.primary is None:
            continue

        axis = _axis_from_hypothesis(node.hypothesis)

        if axis is not None:
            results[axis].append(node.primary)

    # EXPLOIT EARLY:
    # If an axis has produced a meaningful improvement, keep developing it.
    # But if two follow-up experiments fail to beat its best score,
    # consider that direction temporarily exhausted and explore elsewhere.
    meaningful_gain = 0.002
    max_stale_followups = 2

    promising = []

    for axis in RESEARCH_AXES:
        scores = results[axis]

        if not scores:
            continue

        best_score = max(scores)

        if best_score < BASE + meaningful_gain:
            continue

        # Find when this axis achieved its best result.
        best_index = scores.index(best_score)

        # Count scored experiments on this axis after that breakthrough.
        stale_followups = len(scores) - best_index - 1

        if stale_followups < max_stale_followups:
            promising.append((best_score, axis))

    if promising:
        promising.sort(reverse=True)
        return promising[0][1]

    # EXPLORE:
    # Nothing has clearly worked yet, so try a research direction
    # that has not received a scored experiment.
    for axis in RESEARCH_AXES:
        if not results[axis]:
            return axis

    # FALLBACK:
    # We explored everything but nothing beat the noise threshold.
    # Continue with whichever direction performed best.
    return max(
        RESEARCH_AXES,
        key=lambda axis: max(results[axis])
    )

# ==========================================================================
# 1. SEARCH -- what do we work on next?
# ==========================================================================
def select(state: RunState) -> tuple[Node | None, Stage]:
    """Expand the best node; debug in place on failure -- unless on_failure
    asked for a rollback, in which case go back to what worked.

    Does NOT: track which axis (features / loss / model / training) a node
    explored, notice that an axis has stopped paying and force a switch, back
    up to an earlier node when a branch stalls, or shift from exploring to
    exploiting as the budget runs down. "When should the agent stop expanding
    features?" is this function's question, and v0 cannot even ask it.
    """
    if not state.nodes:
        return None, "draft"

    # on_failure decided this branch is not worth more budget. Honour it --
    # otherwise "rollback" is journalled but never acted on, and the run keeps
    # debugging a node the repair policy already gave up on.
    if state.last_recovery == "rollback" and state.best is not None:
        return state.best, "improve"

    last = state.nodes[-1]
    if last.status != "ok":
        return last, "debug"
    return state.best or last, "improve"


# ==========================================================================
# 2. CONTEXT -- what does the model get to see?
# ==========================================================================
def formulate(state: RunState, parent: Node | None, budget_tokens: int) -> str:
    """Give the LLM the experimental history plus an explicit research agenda."""

    parts = [
        f"## Where we stand\n"
        f"baseline to beat: {BASE}   "
        f"oracle ceiling: {CEILING}"
    ]

    best = state.best

    if best and best.metrics:
        parts.append(
            f"best so far: {best.primary:.4f} "
            f"(node {best.id}, delta {best.primary - BASE:+.4f})"
        )

    # --------------------------------------------------------------
    # SHRUTHI V1: explicit explore -> exploit research planning
    # --------------------------------------------------------------
    axis = research_axis(state, parent)

    axis_guidance = {
        "loss": (
            "LOSS / RANKING OBJECTIVE. Investigate a ranking-aligned objective "
            "such as pairwise BPR or a listwise objective instead of merely "
            "adding model capacity."
        ),
        "user_history": (
            "USER HISTORY. Investigate behavioural history or sequence-derived "
            "signals that can change the ranking among a user's impressions."
        ),
        "multi_task": (
            "MULTI-TASK SIGNALS. Investigate auxiliary behavioural signals such "
            "as click, like, follow, comment, forward, or play time while keeping "
            "long_view as the target being evaluated."
        ),
        "watch_time": (
            "WATCH-TIME MODELLING. Investigate play-time/duration information, "
            "including the censoring/truncation issue described in the task."
        ),
    }

    parts.append(
        "\n## Research plan for this iteration\n"
        f"Selected axis: {axis}\n"
        f"{axis_guidance[axis]}\n"
        "Treat this as the research hypothesis for this iteration. "
        "Do not silently switch to another research axis."
    )

    if state.nodes:
        rows = [
            "",
            "## Experimental history",
            "| node | stage | status | primary | inferred axis | hypothesis |",
            "|---|---|---|---|---|---|",
        ]

        for n in state.nodes:
            p = f"{n.primary:.4f}" if n.primary is not None else "--"
            inferred = _axis_from_hypothesis(n.hypothesis) or "--"

            rows.append(
                f"| {n.id} | {n.stage} | {n.status} | {p} | "
                f"{inferred} | {n.hypothesis[:70]} |"
            )

        parts.append("\n".join(rows))

    if parent:
        if parent.status != "ok" and parent.error:
            parts.append(
                f"\n## The failure to fix\n"
                f"```\n{parent.error}\n```"
            )

        parts.append(
            f"\n## Current solution ({parent.id})\n"
            f"```python\n{parent.code}\n```"
        )

    text = "\n".join(parts)

    if rough_tokens(text) > budget_tokens:
        text = (
            text[: budget_tokens * 4]
            + "\n... [truncated by agents/shruthi]"
        )

    return text

# ==========================================================================
# 3. PROMPT -- how do we ask?
# ==========================================================================
SYSTEM = f"""You are an ML research engineer working on a within-user ranking task.

TASK
  Dataset : KuaiRand-Pure short-video feed logs.
  Label   : `long_view` (native 0/1 column).
  Ranking : within each user's logged impressions. NOT full-catalog retrieval.
  Metrics : GAUC and nDCG@5. primary = mean of the two.
  Beat    : the official Factorization Machine baseline, valid primary {BASE}.

READ THE SCALE CORRECTLY
  These metrics do not span [0, 1]. 27.1% of users have no positive label at
  all, so their nDCG is 0 for every possible model. A perfect ranking scores
  primary {CEILING} on validation. The baseline's {BASE} already captures about
  a third of the attainable range. Judge progress against the ceiling, not 1.0
  -- 0.61 is a good score here, not a failing one.

  The baseline's 5-seed standard deviation is {LIMITS['seed_std']}. Any gain
  below ~0.002 is indistinguishable from noise. Do not chase it.

OUTPUT CONTRACT
  Reply with exactly ONE ```python fenced block containing a complete,
  self-contained solution.py. No prose outside the block. It is run as:

      python solution.py --data_dir DIR --out PATH.npy --seed N


  DATA CONTRACT -- read this carefully, it is the most common cause of failure.

    import guard
    splits = guard.load(args.data_dir)      # {{'train': [...], 'valid': [...]}}

  Each split is a plain LIST of TUPLES. Not a DataFrame. Not dicts. No pandas.

    row = (date, user_id, video_id, author_id, tab, duration_ms, label)
           0     1        2         3          4    5            6
    date        int    e.g. 20220415
    user_id     str
    video_id    str
    author_id   str    'UNK' when unknown
    tab         str
    duration_ms float
    label       int    0 or 1 -- this is `long_view`, the target

  So: labels = [r[6] for r in splits['train']], users = [r[1] for r in ...].

  A ready-made encoder is available if you want it:

    enc, dim = guard.encode(splits)
    X, y, users = enc['train']    # X int32 (N,5), y float32 (N,), users list
                                  # dim = total embedding rows for all fields

  You are free to ignore guard.encode and build your own features from the
  tuples. Do not import pandas, sklearn, or torch.

  READING YOUR OWN RESULTS
  - GAUC below 0.5 means your ranking is INVERTED. Random scoring gets 0.5, so
    scoring worse takes real signal pointed backwards -- look for a sign error
    in a gradient or in how the score is formed, not for a better model.
  - GAUC exactly 0.5000 means every score is identical, so every pair ties.
    The model collapsed; check for saturation, a zero learning rate, or
    clipping that flattened everything.
  - A pairwise loss needs, per user, a positive and a negative. Labels are
    floats; do not use a label as an index.

  WHAT IS ALREADY KNOWN -- these were measured by the organizers. Do not
  re-run them.

    Dead ends:
    - More static features. Adding all 13 CWM feature fields (music_id,
      video_type, upload_type, plus six coarse user buckets) scored 0.5940
      against 0.5950 for the five baseline fields. No gain, slightly worse.
    - More capacity. Embedding dim 8 / 16 / 32 gives 0.5895 / 0.5902 / 0.5887.
      It barely moves. 1.14M rows will not support a bigger model.
    - Why: the user_id x video_id cross already captures most of the learnable
      signal, and coarse user buckets are redundant next to user_id.
    - Pure user-side first-order terms contribute EXACTLY ZERO. Ranking is
      within-user, so any term that is constant across a user's impressions
      cannot change their order. User features can only help through crosses
      with item-side features.

  WHERE THE HEADROOM IS -- unexplored, in the organizers' order of promise:

    1. Change the loss. Training is pointwise logloss; the metrics are ranking
       metrics. Pairwise (BPR over within-user positive/negative pairs) or
       listwise (softmax over each user's impressions) aligns the objective
       with how you are scored. Judged most likely to work.
    2. User history sequences. The current features use no behavioural
       sequence at all, yet each user has hundreds to thousands of training
       interactions. DIN / SIM style interest modelling is completely
       untouched here.
    3. Multi-task. The logs carry is_click, is_like, is_follow, is_comment,
       is_forward and play_time_ms. Use them as auxiliary tasks for the
       long_view objective.
    4. Watch-time modelling. A completed play means the true watch time was
       truncated by the video length, so a censored / one-sided loss beats
       squared error (this is the CWM paper's contribution).
    5. A different architecture (DeepFM / DCN / xDeepFM). Lower priority --
       capacity is measurably not the bottleneck.
    6. Time features and distribution drift: hourmin, date, and the shift
       between the train and evaluation windows.

  MORE DATA IS AVAILABLE than guard.load() returns. Inside args.data_dir:
    log_standard_*.csv  -- user_id, video_id, date, hourmin, time_ms,
        is_click, is_like, is_follow, is_comment, is_forward, is_hate,
        long_view, play_time_ms, duration_ms, profile_stay_time,
        comment_stay_time, is_profile_enter, is_rand, tab
    user_features_pure.csv          -- activity, follower/fan/friend counts, ...
    video_features_basic_pure.csv   -- author_id, video_type, upload_type, ...
    video_features_statistic_pure.csv -- show/play/complete counts per video
  Read them with the csv module if you need signals 2-4 above. Row order for
  the valid split is still whatever guard.load() returns -- align to that.

  It must:
    - train on train only
    - write a float array to args.out, one score per valid row, in the order
      guard returns them. Any real number; only relative order matters
    - be deterministic given --seed
    - finish inside {LIMITS['solution_timeout_s']} seconds on one CPU core
    - use only numpy and the standard library unless you truly need more

  There is no test split. guard.load() does not return one. Do not look for it
  and do not open any file outside args.data_dir -- generated code that
  references the raw dataset directory is rejected before it runs.

  Begin the block with: `# HYPOTHESIS: <what you are trying and why>`
"""


def build(state: RunState, parent: Node | None, stage: Stage,
          context: str) -> tuple[str, list[dict]]:
    """One system prompt, one user turn.

    Does NOT: vary tone or structure by stage beyond one sentence, give
    worked examples, ask for a plan before code, or say anything about the
    dataset's known dead ends (static features and model capacity are both
    measured non-starters -- see the starter kit README).
    """
    if stage == "draft":
        ask = ("Write a first solution. A factorization machine over the "
               "categorical fields is a reasonable starting point.")
    elif stage == "debug":
        ask = ("The solution above failed. Fix ONLY the error.\n"
               "Change as little as possible: keep the same model, the same "
               "loss, the same hyperparameters, the same structure. Do not "
               "take the opportunity to try a different idea -- a repair that "
               "also rewrites the approach cannot be told apart from a "
               "regression, and it wastes the iteration either way.\n"
               "State the root cause in the HYPOTHESIS line, then output the "
               "complete corrected solution.py.")
    else:
        axis = research_axis(state, parent)

        ask = (
            f"Improve on the solution above. Change ONE thing.\n"
            f"For this experiment, investigate the research direction: {axis}.\n"
            f"Keep the change focused on {axis}; do not switch to another "
            f"research direction.\n"
            "Copy everything you are not changing VERBATIM from the current "
            "solution -- the training loop, the encoder, the argument "
            "parsing and the output writing already work. Most failures in "
            "this run came from retyping code that was already correct, not "
            "from the idea being wrong.\n"
            "Say what you are changing and why in the HYPOTHESIS line, then "
            "output the complete new solution.py."
        )

    budget = (f"Iteration {state.iteration}. {state.iters_left} iterations and "
              f"{int(state.seconds_left / 60)} minutes remain.")
    return SYSTEM, [{"role": "user", "content": f"{context}\n\n{budget}\n\n{ask}"}]


# ==========================================================================
# 4. VALIDATION -- real improvement, or seed noise?
# ==========================================================================
def seeds_for(state: RunState, stage: Stage) -> list[int]:
    """One seed, always. This is the wrong answer, and it is worth knowing why:

        epsilon (convergence) = 0.002
        baseline seed std     = 0.0008

    epsilon is 2.5 sigma. A single-seed run can swing +/-0.0015 on noise alone,
    so v0 will sometimes promote a worse solution -- and three noisy
    non-improvements in a row trip the convergence rule and end the run early
    with iterations unspent.

    Does NOT: screen cheaply then confirm promising candidates on more seeds,
    or spend more as the remaining budget shrinks. Seeds cost iterations out
    of 50; that trade is this function's whole job.
    """
    return [0]


def judge(candidate: Node, incumbent: Node | None,
          state: RunState) -> tuple[bool, str]:
    """Accept anything better by any margin. See seeds_for for why that is a
    problem. Does NOT require the gain to exceed noise."""
    if candidate.metrics is None:
        return False, f"no metrics (status={candidate.status})"
    if candidate.metrics.get("degenerate"):
        return False, "constant scores -- every pair ties, GAUC is 0.5 by "\
                      "construction. The model is broken, not weak."
    if candidate.metrics.get("inverted"):
        return False, "GAUC below 0.45 -- the ranking is inverted (random is "\
                      "0.5). Sign error, not a weak model."
    if incumbent is None or incumbent.metrics is None:
        return True, "first scored solution becomes the incumbent"
    delta = candidate.primary - incumbent.primary
    if delta > 0:
        noisy = delta < 2 * LIMITS["seed_std"]
        note = " (within 2 sigma of seed noise -- v0 takes it anyway)" if noisy else ""
        return True, f"primary {delta:+.4f}{note}"
    return False, f"primary {delta:+.4f}"


# ==========================================================================
# 5. REPAIR -- it crashed. now what?
# ==========================================================================
MAX_ATTEMPTS = 3

# Failures a traceback usually lets the model fix on the next try.
FIXABLE = {
    "TypeError", "AttributeError", "NameError", "IndexError",
    "KeyError", "ValueError", "ImportError", "ModuleNotFoundError",
    "NoCode", "Misaligned", "NoOutput"
}


def _error_type(node: Node) -> str:
    if not node.error:
        return ""

    try:
        error = json.loads(node.error)
    except (ValueError, TypeError):
        return ""

    kind = error.get("type", "")

    # NonZeroExit only tells us that the generated program crashed.
    # Look inside the traceback for the actual Python error so recovery
    # can decide whether it is fixable.
    if kind == "NonZeroExit":
        traceback = error.get("traceback_tail", "")

        for fixable in FIXABLE:
            if f"{fixable}:" in traceback:
                return fixable

    return kind


def on_failure(node: Node, attempt: int, state: RunState) -> str:
    """Route recovery based on what actually failed."""

    kind = _error_type(node)

    # If the same error keeps happening, stop wasting attempts
    # and return to something that previously worked.
    prior = [_error_type(n) for n in state.nodes if n.status != "ok"]

    if kind and prior[-1:] == [kind] and attempt >= 2:
        return "rollback" if state.best_id else "retry"

    # Timeout: give the model one chance to make the experiment smaller.
    if node.status == "timeout":
        return "retry" if attempt < 2 else (
            "rollback" if state.best_id else "abandon"
        )

    # Numerical failure such as NaN/Inf.
    if kind == "NonFinite":
        return "retry" if attempt < 2 else (
            "rollback" if state.best_id else "retry"
        )

    # Data leakage should not be repaired by repeatedly retrying
    # the same experiment.
    if node.status == "leak":
        return "rollback" if state.best_id else "retry"

    # Normal coding errors are usually worth repairing.
    if kind in FIXABLE and attempt < MAX_ATTEMPTS:
        return "retry"

    if state.best_id:
        return "rollback"

    # Avoid abandoning while useful iteration budget remains.
    return "retry" if state.iters_left > 1 else "abandon"
