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

BASE = LIMITS["baseline_primary"]["valid"]      # 0.6016 -- the bar
CEILING = LIMITS["oracle_primary"]["valid"]     # 0.8484 -- perfect ranking


# ==========================================================================
# 1. SEARCH -- what do we work on next?
# ==========================================================================
def select(state: RunState) -> tuple[Node | None, Stage]:
    """Greedy: always expand the best node; debug in place on failure.

    Does NOT: track which axis (features / loss / model / training) a node
    explored, notice that an axis has stopped paying and force a switch, back
    up to an earlier node when a branch stalls, or shift from exploring to
    exploiting as the budget runs down. "When should the agent stop expanding
    features?" is this function's question, and v0 cannot even ask it.
    """
    if not state.nodes:
        return None, "draft"
    last = state.nodes[-1]
    if last.status != "ok":
        return last, "debug"
    return state.best or last, "improve"


# ==========================================================================
# 2. CONTEXT -- what does the model get to see?
# ==========================================================================
def formulate(state: RunState, parent: Node | None, budget_tokens: int) -> str:
    """Everything, then truncate. Blows the budget on long runs and says
    nothing about WHY anything worked.

    Does NOT: pick which ancestors matter, summarize failed branches, send a
    diff instead of whole files, or spend its budget on the things that
    actually inform the next decision.
    """
    parts = [f"## Where we stand\nbaseline to beat: {BASE}   "
             f"oracle ceiling: {CEILING}"]

    best = state.best
    if best and best.metrics:
        parts.append(f"best so far: {best.primary:.4f} (node {best.id}, "
                     f"delta {best.primary - BASE:+.4f})")

    if state.nodes:
        rows = ["", "## History", "| node | stage | status | primary | hypothesis |",
                "|---|---|---|---|---|"]
        for n in state.nodes:
            p = f"{n.primary:.4f}" if n.primary is not None else "--"
            rows.append(f"| {n.id} | {n.stage} | {n.status} | {p} | "
                        f"{n.hypothesis[:70]} |")
        parts.append("\n".join(rows))

    if parent:
        if parent.status != "ok" and parent.error:
            parts.append(f"\n## The failure to fix\n```\n{parent.error}\n```")
        parts.append(f"\n## Current solution ({parent.id})\n"
                     f"```python\n{parent.code}\n```")

    text = "\n".join(parts)
    if rough_tokens(text) > budget_tokens:
        text = text[: budget_tokens * 4] + "\n... [truncated by agents/v0]"
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

ALREADY MEASURED -- DO NOT SPEND ITERATIONS RE-TESTING THESE
  These were run by the organizers on this exact data and metric. They are
  settled, not open questions:
    - MORE STATIC FEATURES DO NOT HELP. All 13 feature domains scored 0.5940
      against 0.5950 for just 5. user_id x video_id already carries the signal
    - MORE CAPACITY DOES NOT HELP. Embedding dim 8 / 16 / 32 scored
      0.5895 / 0.5902 / 0.5887. The data does not support a bigger model
    - PURE USER-SIDE FEATURES CONTRIBUTE EXACTLY ZERO. Ranking happens within
      one user, so any term constant across that user's rows cannot change
      their order. User features only matter crossed with an item-side term
    - A PLAIN LINEAR MODEL IS STRICTLY WORSE than the factorization machine.
      Dropping the interaction term is a regression, not a simplification

  The bottleneck is not features and not capacity. The unexplored directions,
  best first: a ranking loss that matches the metric (pairwise BPR, or a
  softmax over each user's impressions) instead of pointwise logloss; the
  user's behaviour sequence, which nothing currently uses; auxiliary targets
  (is_click, is_like, play_time_ms) alongside long_view.

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

  It must:
    - train on train only
    - write a float array to args.out, one score per valid row, in the order
      guard returns them. Any real number; only relative order matters
    - be deterministic given --seed
    - finish inside {LIMITS['solution_timeout_s']} seconds on one CPU core
    - use only numpy and the standard library unless you truly need more

  EVERY SCORE MUST BE FINITE. A single NaN or Inf in the output array makes
  the whole run unscoreable and wastes the iteration -- this is the most
  common way solutions here fail. Numerically:
    - clip logits before exp: `np.clip(z, -30, 30)`. A raw sigmoid on an
      unbounded score overflows and silently produces NaN
    - guard divisions and logs; add an epsilon rather than dividing by a
      count that can be zero
    - a diverging learning rate turns weights into NaN within a few epochs.
      If the training loss stops being finite, lower the rate rather than
      training on
    - check the scores before writing and FAIL LOUDLY if they are bad. Do
      NOT patch them up and write them anyway -- a quietly repaired array
      scores like random guessing and poisons every later iteration, which
      is far worse than an error message:

          assert np.all(np.isfinite(scores)), "non-finite scores"
          assert np.unique(scores).size > 1, "degenerate: all scores equal"

      A crash here is recoverable; a silently wrong array is not. If that
      assert fires, the fix is a lower learning rate or better clipping in
      TRAINING -- not a patch at the output.

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
        ask = ("The solution above failed. Diagnose it from the error, then "
               "output a corrected complete solution.py.")
    else:
        ask = ("Improve on the solution above. START FROM THAT EXACT CODE and "
               "change ONE thing. Do not rewrite it from scratch, do not "
               "restructure what you are not changing, and do not swap the "
               "model family unless that IS your one change -- every part you "
               "touch for no reason is a chance to lose the score you already "
               "have. Name the single change in the HYPOTHESIS line, then "
               "output the complete file with that change applied.\n\n"
               "The score to beat is the parent's, shown above. Recent "
               "attempts here have regressed by 0.02-0.08 by rewriting "
               "wholesale; a change you cannot justify is more likely to "
               "cost you than to help.")

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


def on_failure(node: Node, attempt: int, state: RunState) -> str:
    """Retry with the traceback, roll back once there is something to roll back
    to, and only abandon when the budget is genuinely spent.

    Abandoning while no solution has EVER worked kills the run at iteration 2
    with 48 iterations unused -- the early drafts are exactly where failures
    are expected. Never abandon while iterations remain.

    Still does NOT: distinguish a missing import from a logic bug from a
    timeout, notice it is retrying the same wrong fix twice, or shrink the
    problem when the failure was a timeout.
    """
    if attempt < MAX_ATTEMPTS:
        return "retry"
    if state.best_id:
        return "rollback"
    return "retry" if state.iters_left > 1 else "abandon"
