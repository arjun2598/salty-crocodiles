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
    "time_drift",
    "architecture",
]

# Concrete experiment ladder. The planner chooses the research area; this ladder
# stops the coding model from having to invent a completely new experiment from
# scratch every iteration.
EXPERIMENT_RECIPES = {
    "loss": [
        "Implement within-user pairwise BPR loss. Sample positive/negative impressions from the same user and keep the existing FM scoring function.",
        "Try a numerically stable within-user listwise softmax objective while keeping the existing FM architecture.",
        "Refine the best ranking loss with conservative regularization or pair sampling; do not rewrite the model.",
    ],
    "user_history": [
        "Add a leakage-safe user-history affinity signal built only from TRAIN interactions, e.g. recent positive author/video preference, and use it to rank validation impressions.",
        "Add recency weighting to the train-only user-history signal so recent positive interactions matter more than old ones.",
        "Combine the strongest train-only history signal with the incumbent model using a small deterministic score weight.",
    ],
    "multi_task": [
        "Read auxiliary TRAIN behaviour columns such as is_click/is_like/is_follow/is_comment/is_forward and create a train-only engagement signal that helps learn long_view ranking.",
        "Use a conservative weighted auxiliary engagement target while keeping long_view as the evaluated target; avoid validation-label leakage.",
        "Combine the best auxiliary behavioural signal with the incumbent score rather than replacing the whole model.",
    ],
    "watch_time": [
        "Use TRAIN play_time_ms and duration_ms to derive a bounded completion/watch-ratio auxiliary signal; keep long_view as the evaluation target.",
        "Model completed plays as censored by video duration using a simple robust one-sided or capped watch-time signal.",
        "Blend the strongest watch-time-derived signal with the incumbent ranking using a conservative weight.",
    ],
    "time_drift": [
        "Add a simple time/drift-aware item or author statistic from TRAIN only, using date/hourmin without validation labels.",
        "Use recency-weighted TRAIN statistics so behaviour closer to the validation window has more influence.",
        "Blend the best temporal signal with the incumbent score conservatively.",
    ],
    "architecture": [
        "Only if simpler directions have been explored: add a small nonlinear interaction on top of the existing FM without increasing embedding size aggressively.",
        "Try a lightweight explicit cross/interactions extension that fits CPU/time limits and preserves the working data/output pipeline.",
    ],
}


def _axis_from_hypothesis(hypothesis: str) -> str | None:
    """Infer which research axis a previous experiment investigated."""
    text = (hypothesis or "").lower()
    keywords = {
        "loss": ["loss", "bpr", "pairwise", "listwise", "ranking objective"],
        "user_history": ["user history", "history", "sequence", "recency", "affinity", "din", "sim"],
        "multi_task": ["multi-task", "multitask", "auxiliary", "engagement", "click", "like", "follow", "comment", "forward"],
        "watch_time": ["watch time", "watch-time", "play time", "play_time", "watch ratio", "completion", "censored", "duration"],
        "time_drift": ["time drift", "temporal", "hourmin", "date feature", "recency-weighted", "distribution drift"],
        "architecture": ["architecture", "deepfm", "dcn", "xdeepfm", "nonlinear", "cross network"],
    }
    for axis, words in keywords.items():
        if any(word in text for word in words):
            return axis
    return None


def _axis_scores(state: RunState) -> dict[str, list[float]]:
    results = {axis: [] for axis in RESEARCH_AXES}
    for node in state.nodes:
        if node.status == "ok" and node.primary is not None:
            axis = _axis_from_hypothesis(node.hypothesis)
            if axis:
                results[axis].append(node.primary)
    return results


def research_axis(state: RunState, parent: Node | None = None) -> str:
    """Evidence-aware explore -> exploit, with cheap/high-priority ideas first."""
    if parent is not None and parent.status != "ok":
        previous_axis = _axis_from_hypothesis(parent.hypothesis)
        if previous_axis:
            return previous_axis

    results = _axis_scores(state)
    meaningful_gain = LIMITS["epsilon"]
    max_stale_followups = 2

    promising = []
    for axis in RESEARCH_AXES:
        scores = results[axis]
        if not scores:
            continue
        best_score = max(scores)
        if best_score < BASE + meaningful_gain:
            continue
        best_index = scores.index(best_score)
        stale = len(scores) - best_index - 1
        if stale < max_stale_followups:
            promising.append((best_score, axis))

    if promising:
        return max(promising)[1]

    # Explore in deliberate order: ranking alignment first, expensive model
    # architecture last. A failed implementation does not count as evidence.
    for axis in RESEARCH_AXES:
        if not results[axis]:
            return axis

    return max(RESEARCH_AXES, key=lambda axis: max(results[axis]))


def experiment_recipe(state: RunState, axis: str) -> str:
    """Return a concrete next experiment inside an axis."""
    scored = 0
    for node in state.nodes:
        if node.status == "ok" and node.primary is not None and _axis_from_hypothesis(node.hypothesis) == axis:
            scored += 1
    recipes = EXPERIMENT_RECIPES[axis]
    return recipes[min(scored, len(recipes) - 1)]

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
    """Compact evidence + a REAL working solution.

    Important: a NoCode node has empty code.  On that failure, show the best
    runnable solution instead of asking the model to reconstruct everything
    from scratch.
    """
    parts = [f"## Status\nbaseline: {BASE}   oracle: {CEILING}"]

    best = state.best
    if best and best.metrics:
        parts.append(
            f"best: {best.primary:.4f} (node {best.id}, delta {best.primary - BASE:+.4f})"
        )

    axis = research_axis(state, parent)
    recipe = experiment_recipe(state, axis)
    parts.append(f"## Experiment\n{recipe}")

    if state.nodes:
        rows = ["## Evidence"]
        for n in state.nodes[-4:]:
            p = f"{n.primary:.4f}" if n.primary is not None else "--"
            rows.append(f"{n.id}: {n.status}, primary={p}, {n.hypothesis[:80]}")
        parts.append("\n".join(rows))

    if parent and parent.status != "ok" and parent.error:
        parts.append(f"## Failure\n{parent.error}")

    # Never feed an empty NoCode solution back to the coding model.
    working = parent if parent and parent.code else best
    if working and working.code:
        parts.append(
            f"## Working solution ({working.id}) -- preserve this code\n"
            f"```python\n{working.code}\n```"
        )

    text = "\n\n".join(parts)
    if rough_tokens(text) > budget_tokens:
        text = text[: budget_tokens * 4] + "\n... [truncated by agents/shruthi]"
    return text

# ==========================================================================
# 3. PROMPT -- how do we ask?
# ==========================================================================
SYSTEM = f"""You are modifying a working Python recommender for KuaiRand-Pure.

GOAL
- label: long_view
- rank within each user's impressions
- metrics: GAUC and nDCG@5; primary = their mean
- baseline primary: {BASE}
- meaningful gain: about 0.002

HARD OUTPUT CONTRACT
- Output EXACTLY ONE complete ```python fenced block and NOTHING else.
- Start immediately with ```python.
- End with ```.
- Complete runnable solution.py, preferably <=95 lines.
- NO explanations, analysis, docstrings, pseudocode, or long comments.
- COPY working infrastructure; edit only lines needed for the experiment.
- Do not rewrite argparse, guard loading, prediction, saving, or evaluation unless required.
- Finish the entire program before the token limit.

DATA
- guard.load(args.data_dir) returns only train/valid.
- each row is tuple:
  (date, user_id, video_id, author_id, tab, duration_ms, long_view)
- guard.encode(splits) gives:
  X, y, users = enc["train"], with X shaped (N,5)
- numpy only; no pandas/sklearn/torch; never access test.

PAIRWISE SAFETY
- Pair samples are ROW INDICES. Convert them first:
    Xp = Xtr[pos_idx]
    Xn = Xtr[neg_idx]
- Compute:
    z_pos, E_pos, S_pos = self.logits(Xp)
    z_neg, E_neg, S_neg = self.logits(Xn)
    g = ((sigmoid(z_pos - z_neg) - 1.0) / B).astype(np.float32)
- Correct FM embedding gradients MUST use FEATURE IDS and preserve field dimension:
    np.add.at(gW, Xp, g[:, None])
    np.add.at(gV, Xp, g[:, None, None] * (S_pos[:, None, :] - E_pos))
    np.add.at(gW, Xn, -g[:, None])
    np.add.at(gV, Xn, -g[:, None, None] * (S_neg[:, None, :] - E_neg))
- NEVER use pos_idx/neg_idx as indices into gV/gW.
- NEVER write (S_pos - E_pos); use S_pos[:, None, :] - E_pos.
- Divide pairwise gradient by B.
- Use bounded/vectorized sampling; avoid loops over all users/pairs.
- Keep runtime small.
"""


def build(state: RunState, parent: Node | None, stage: Stage,
          context: str) -> tuple[str, list[dict]]:
    """Research policy stays rich; coding request stays brutally compact."""
    compact = (
        "OUTPUT RULES:\n"
        "- First characters: ```python\n"
        "- Last characters: ```\n"
        "- ONE complete runnable solution.py, preferably <=95 lines.\n"
        "- ZERO prose/reasoning/docstrings/pseudocode; code only.\n"
        "- Preserve working code literally where possible; make the smallest edit.\n"
        "- Do not re-explain or re-architect working infrastructure.\n"
        "- Finish the code and closing fence before anything else.\n"
    )

    if stage == "draft":
        ask = (
            "Write the smallest complete runnable baseline solution.py.\n" + compact
        )
    elif stage == "debug":
        # NoCode means there is no broken program to repair. Re-attempt the
        # selected experiment from the working incumbent shown in context.
        no_code = parent is not None and _error_type(parent) == "NoCode"
        if no_code:
            axis = research_axis(state, parent)
            recipe = experiment_recipe(state, axis)
            ask = (
                f"Implement this experiment on the WORKING solution: {recipe}\n"
                "The previous response was truncated/missing a complete code fence. "
                "Do NOT reconstruct or expand the program unnecessarily.\n" + compact
            )
        else:
            ask = (
                "Fix ONLY the shown failure. Keep the same experiment and change "
                "the minimum number of lines.\n" + compact
            )
    else:
        axis = research_axis(state, parent)
        recipe = experiment_recipe(state, axis)
        ask = (
            f"Implement ONE experiment on the working solution: {recipe}\n"
            "Keep the existing model/training/CLI/output code unchanged unless this "
            "experiment strictly requires a small edit.\n" + compact
        )

    budget = (f"Iteration {state.iteration}; {state.iters_left} iterations; "
              f"{int(state.seconds_left / 60)} minutes left.")
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
    min_gain = 2 * LIMITS["seed_std"]
    if delta >= min_gain:
        return True, f"primary {delta:+.4f} (clears 2-sigma noise threshold {min_gain:.4f})"
    if delta > 0:
        return False, f"primary {delta:+.4f} -- positive but within seed noise; do not promote"
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
