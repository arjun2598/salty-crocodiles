from agents.shruthi import research_axis, BASE


class FakeNode:
    def __init__(self, hypothesis, primary, status="ok"):
        self.hypothesis = hypothesis
        self.primary = primary
        self.status = status


class FakeState:
    def __init__(self, nodes):
        self.nodes = nodes


def test(name, nodes, expected):
    state = FakeState(nodes)
    actual = research_axis(state)

    result = "PASS" if actual == expected else "FAIL"

    print(f"{result}: {name}")
    print(f"  expected: {expected}")
    print(f"  actual:   {actual}\n")


# TEST 1:
# Nothing has been tried.
# Shruthi should start with loss.
test(
    "Start by exploring loss",
    [],
    "loss",
)


# TEST 2:
# Loss was tried but only improved by 0.0005.
# That's too small, so explore user history.
test(
    "Ignore noisy loss improvement",
    [
        FakeNode(
            "Try pairwise BPR loss for ranking",
            BASE + 0.0005,
        )
    ],
    "user_history",
)


# TEST 3:
# Loss improved by 0.003.
# That's meaningful, so exploit loss again.
test(
    "Exploit promising loss direction",
    [
        FakeNode(
            "Try pairwise BPR loss for ranking",
            BASE + 0.003,
        )
    ],
    "loss",
)


# TEST 4:
# Loss and user history didn't meaningfully improve.
# Multi-task hasn't been tried, so try it next.
test(
    "Continue exploring when nothing works",
    [
        FakeNode(
            "Try pairwise BPR loss",
            BASE + 0.0004,
        ),
        FakeNode(
            "Use behavioural user history sequence",
            BASE - 0.001,
        ),
    ],
    "multi_task",
)


# TEST 5:
# Loss produced a breakthrough and only one follow-up has failed.
# Give it one more chance.
test(
    "Keep exploiting after one stale follow-up",
    [
        FakeNode("Try pairwise BPR loss", BASE + 0.004),
        FakeNode("Improve the pairwise ranking loss", BASE + 0.003),
    ],
    "loss",
)


# TEST 6:
# Loss produced a breakthrough, but two follow-ups failed
# to improve its best score. Move on to user history.
test(
    "Stop exploiting after two stale follow-ups",
    [
        FakeNode("Try pairwise BPR loss", BASE + 0.004),
        FakeNode("Improve the pairwise ranking loss", BASE + 0.003),
        FakeNode("Refine the BPR loss objective", BASE + 0.0035),
    ],
    "user_history",
)


print("\n--- PROMPT SANITY CHECK ---")

from agents.shruthi import formulate, build


class PromptState(FakeState):
    iteration = 3
    iters_left = 5
    seconds_left = 3000

    @property
    def best(self):
        if not self.nodes:
            return None
        return max(self.nodes, key=lambda n: n.primary)

    @property
    def best_id(self):
        return self.best.id if self.best else None


class PromptNode(FakeNode):
    def __init__(self, node_id, hypothesis, primary, status="ok"):
        super().__init__(hypothesis, primary, status)
        self.id = node_id
        self.stage = "improve"
        self.metrics = {"primary": primary}
        self.code = "# fake working solution"
        self.error = None


node = PromptNode(
    "n2",
    "Try pairwise BPR loss for within-user ranking",
    BASE + 0.004,
)

state = PromptState([node])

axis = research_axis(state, node)
context = formulate(state, node, 12000)
_, messages = build(state, node, "improve", context)

prompt = messages[0]["content"]

print("Chosen axis:", axis)
print("Context contains 'Selected axis: loss':",
      "Selected axis: loss" in context)
print("Prompt tells model to investigate loss:",
      "investigate the research direction: loss" in prompt)


print("\n--- FAILURE RECOVERY TESTS ---")

from agents.shruthi import on_failure
import json


class FailureState:
    def __init__(self, nodes, best_id="n1", iters_left=5):
        self.nodes = nodes
        self.best_id = best_id
        self.iters_left = iters_left


class FailureNode:
    def __init__(self, error_type, status="error"):
        self.error = json.dumps({"type": error_type})
        self.status = status


def recovery_test(name, node, attempt, state, expected):
    actual = on_failure(node, attempt, state)
    result = "PASS" if actual == expected else "FAIL"

    print(f"{result}: {name}")
    print(f"  expected: {expected}")
    print(f"  actual:   {actual}\n")


# NoCode happened once → try repairing it
node = FailureNode("NoCode")
recovery_test(
    "Retry first NoCode",
    node,
    1,
    FailureState([]),
    "retry",
)


# Same NoCode keeps happening → stop wasting attempts
previous = FailureNode("NoCode")
current = FailureNode("NoCode")

recovery_test(
    "Rollback after repeated NoCode",
    current,
    2,
    FailureState([previous]),
    "rollback",
)


# Timeout happened once → give model one chance to shrink experiment
timeout = FailureNode("Timeout", status="timeout")

recovery_test(
    "Retry first timeout",
    timeout,
    1,
    FailureState([]),
    "retry",
)


# TEST 10:
# The generated program exited with NonZeroExit,
# but the traceback reveals a fixable TypeError.
# Shruthi should retry instead of throwing away the experiment.

type_error_node = FailureNode("NonZeroExit")
type_error_node.error = json.dumps({
    "type": "NonZeroExit",
    "message": "exit 1",
    "traceback_tail": (
        "File \"solution.py\", line 70\n"
        "m[:] = beta1 * m\n"
        "TypeError: 'float' object does not support item assignment"
    )
})

recovery_test(
    "Detect TypeError inside NonZeroExit traceback",
    type_error_node,
    1,
    FailureState([]),
    "retry",
)
