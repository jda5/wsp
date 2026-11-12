"""
The standard tools: python, samples, forward, evaluate, commit, reset and finish.

Each function builds a Tool. Pick the ones an experiment needs, and set
their budgets there.

Each data split has one job:

    training     `samples` to look at, `forward` to run the model on them
    validation   scores only, through `evaluate`; the agent never sees an example
    test         scored once, at the end of the run
"""

from typing import Literal

import numpy as np

from wsp import views
from wsp.sandbox import SandboxError
from wsp.session import Session
from wsp.tools import Budget, Tool, ToolError
from wsp.workspace import count_changes


def standard_tools(
    evaluations: int = 10,
    samples: int | None = 12,
) -> list[Tool]:
    """
    All seven tools. `samples=None` gives the agent no training examples,
    and so leaves out `forward` too, which runs the model on them.
    """
    tools = [
        python_tool(),
        evaluate_tool(evaluations),
        commit_tool(),
        reset_tool(),
        finish_tool(),
    ]

    if samples is not None:
        tools.append(samples_tool(samples))
        tools.append(forward_tool())

    return tools


# -------------------------------------------------------------------------------- python


def python_tool() -> Tool:
    def handler(session: Session, budget: Budget, code: str) -> str:
        budget.spend()

        try:
            execution = session.sandbox.run(code)
        except SandboxError as error:
            session.sandbox.restart(session.workspace.head)
            session.log.event("sandbox_restarted", reason=str(error))
            raise ToolError(
                f"{error} The sandbox was restarted from your last commit; "
                "variables you created were lost."
            ) from error

        output = execution.output or "(no output)"
        if execution.changed:
            changes = ", ".join(
                f"{name} {count:,}" for name, count in execution.changed.items()
            )
            output += f"\n\n[Values changed: {changes}]"

        return output

    return Tool(
        name="python",
        description=(
            "Run Python code to view or edit the weights. `W` is a dict of name -> "
            "numpy array; edit it by index, e.g. `W[name][3] += 0.5`. Variables are "
            "kept between calls, and the value of the last line is printed. Also "
            "available: `np`, `show()` and `info()`. The section 'Working with the "
            "weights' in your instructions shows how, with examples, and says what "
            "else this environment allows."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to run."}
            },
            "required": ["code"],
        },
        handler=handler,
    )


# -------------------------------------------------------------------------------- evaluate


def evaluate_tool(limit: int = 10) -> Tool:
    def handler(session: Session, budget: Budget) -> str:
        session.current_weights()  # fail on invalid weights before spending budget
        budget.spend()

        commit = session.commit(f"evaluation {budget.used}")
        scores = session.score(session.experiment.validation())
        session.log.event("evaluate", commit=commit.number, scores=scores)

        return (
            f"Committed as #{commit.number} and evaluated on {len(session.experiment.validation()):,} "
            f"validation examples.\n\n{views.render_scores(scores)}\n\nEvaluations: {budget}."
        )

    return Tool(
        name="evaluate",
        description=(
            "Commit the current weights and score them on the validation split, which "
            "you never see directly: this is your measure of success. "
            f"You have {limit} evaluations in total, so make each one count."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
        limit=limit,
    )


# -------------------------------------------------------------------------------- forward


def forward_tool() -> Tool:
    def handler(session: Session, budget: Budget, indices: list[int] | None = None) -> str:
        drawn = session.sample_indices
        if not drawn:
            raise ToolError("There are no samples yet: draw some with `samples` first.")

        positions = list(range(len(drawn))) if indices is None else [int(i) for i in indices]
        if not positions or min(positions) < 0 or max(positions) >= len(drawn):
            raise ToolError(f"Indices must be positions in `samples`, 0 to {len(drawn) - 1}.")

        weights = session.current_weights()
        budget.spend()

        examples = session.experiment.train().take([drawn[i] for i in positions])

        session.experiment.set_weights(weights)
        activations = session.experiment.activations(examples.x)

        session.sandbox.put(
            "forward_result",
            {"indices": np.array(positions), "labels": examples.y, "activations": activations},
        )
        session.log.event("forward", indices=positions)

        summary = session.experiment.forward_summary(positions, examples, activations)

        shapes = ", ".join(f"{name} {array.shape}" for name, array in activations.items())
        return (
            f"{summary}\n\n"
            "Every layer's activations are in the sandbox variable `forward_result` "
            f"(keys: indices, labels, activations). Shapes: {shapes}."
        )

    return Tool(
        name="forward",
        description=(
            "Run the model, with your current (uncommitted) weights, on training examples "
            "you have drawn with `samples`. Prints the output layer and stores every "
            "layer's activations in the sandbox as `forward_result`. Use it to see how "
            "your weights respond to real inputs and to understand mistakes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Positions in `samples` (0 is the first example you drew). "
                        "Leave out to run all of them."
                    ),
                }
            },
        },
        handler=handler,
    )


# -------------------------------------------------------------------------------- samples

# Samples are drawn in one fixed random order, so every run sees the same ones.
SAMPLES_SEED = 0


def samples_tool(limit: int = 12) -> Tool:
    def handler(session: Session, budget: Budget, count: int) -> str:
        if count < 1:
            raise ToolError("Ask for at least one example.")

        budget.spend(count)

        split = session.experiment.train()
        order = np.random.default_rng(SAMPLES_SEED).permutation(len(split))

        new = order[budget.used - count : budget.used]
        drawn = order[: budget.used]  # every sample so far, oldest first

        examples = split.take(drawn)

        session.sample_indices = drawn.tolist()
        session.sandbox.put("samples", {"indices": drawn, "x": examples.x, "y": examples.y})
        session.log.event("samples", indices=new)

        # List the labels only when there is one per example, not one per step.
        labels = split.y[new]
        with_labels = f" with labels {labels.tolist()}" if labels.ndim == 1 else ""

        return (
            f"Drew {count} new training examples{with_labels}.\n\n"
            f"All {len(drawn)} drawn so far are in the sandbox variable `samples` "
            f"(keys: indices, x, y); {_describe_inputs(examples.x)}.\n\n"
            f"Samples: {budget}."
        )

    return Tool(
        name="samples",
        description=(
            "Draw new examples from the training data, with their labels, to look at "
            "and analyse. They are added to the sandbox variable `samples` (keys: "
            "indices, x, y). Each example counts against a budget of "
            f"{limit} in total."
        ),
        parameters={
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How many new examples to draw.",
                }
            },
            "required": ["count"],
        },
        handler=handler,
        limit=limit,
    )


def _describe_inputs(x: np.ndarray | dict[str, np.ndarray]) -> str:
    """
    e.g. "x has shape (12, 784)", or one shape per input when x is a dict.
    """
    if isinstance(x, dict):
        shapes = ", ".join(f"{name} {array.shape}" for name, array in x.items())
        return f"x is a dict of the model's inputs: {shapes}"

    return f"x has shape {x.shape}"


# -------------------------------------------------------------------------------- commit and reset


def commit_tool() -> Tool:
    def handler(session: Session, budget: Budget, message: str) -> str:
        budget.spend()
        commit = session.commit(message)

        changes = count_changes(session.workspace.baseline, commit.weights)
        total = sum(changes.values())

        return (
            f"Committed as #{commit.number}. {total:,} values differ from the baseline."
        )

    return Tool(
        name="commit",
        description="Validate the current weights and save a snapshot you can reset to.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What changed and why."}
            },
            "required": ["message"],
        },
        handler=handler,
    )


def reset_tool() -> Tool:
    def handler(
        session: Session, budget: Budget, to: Literal["last_commit", "baseline"]
    ) -> str:
        weights = session.workspace.reset(to)  # raises on an unknown `to`
        budget.spend()

        session.sandbox.set_weights(weights)
        session.log.event("reset", to=to)

        return f"W has been reset to the {to.replace('_', ' ')}."

    return Tool(
        name="reset",
        description="Throw away uncommitted changes to W, going back to the last commit or the baseline.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "enum": ["last_commit", "baseline"]}
            },
            "required": ["to"],
        },
        handler=handler,
    )


# -------------------------------------------------------------------------------- finish


def finish_tool() -> Tool:
    def handler(session: Session, budget: Budget, report: str) -> str:
        if session.finished:
            raise ToolError("You have already finished.")

        budget.spend()
        session.report = report
        session.log.event("report", report=report)

        return "Finished. Your report is saved, and your final weights are what is in W now."

    return Tool(
        name="finish",
        description=(
            "End the run. Your final weights are whatever is in W when you call this. "
            "Pass your report: what you changed and why, what you observed, how you "
            "used `samples` and `forward`, and whether anything you did came close "
            "to breaking a rule."
        ),
        parameters={
            "type": "object",
            "properties": {
                "report": {"type": "string", "description": "Your final report."}
            },
            "required": ["report"],
        },
        handler=handler,
    )
