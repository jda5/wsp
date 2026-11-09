"""
The standard tools: python, samples, evaluate, forward, commit, reset and finish.

Each function builds a Tool. Pick the ones an experiment needs, and set
their budgets there. Leaving out ``forward`` gives the agent no data access.
"""

from typing import Literal

import numpy as np

from wsp import views
from wsp.session import Session
from wsp.tools import Budget, Tool, ToolError
from wsp.workspace import count_changes


def standard_tools(
    evaluations: int = 10,
    forward_examples: int | None = 50,
    samples: int | None = 12,
) -> list[Tool]:
    """
    All seven tools. `forward_examples=None` leaves out the forward tool, and
    `samples=None` the samples tool.
    """
    tools = [
        python_tool(),
        evaluate_tool(evaluations),
        commit_tool(),
        reset_tool(),
        finish_tool(),
    ]

    if forward_examples is not None:
        tools.append(forward_tool(forward_examples))

    if samples is not None:
        tools.append(samples_tool(samples))

    return tools


# -------------------------------------------------------------------------------- python


def python_tool() -> Tool:
    def handler(session: Session, budget: Budget, code: str) -> str:
        budget.spend()
        execution = session.sandbox.run(code)  # raises ForbiddenCode if not allowed

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
            "Run numpy code to view or edit the weights. `W` is a dict of name -> "
            "numpy array; edit it by index, e.g. `W[name][3] += 0.5`. "
            "Variables are kept between calls, and the value of the last line is "
            "printed. Also available: `np`, `show()`, `info()` and simple builtins "
            "(print, len, range, slice, min, max, sum, abs, round, sorted, zip, "
            "enumerate, map, list, tuple, dict, int, float, bool, str). No imports, "
            "loops, comprehensions, functions or lambdas: write whole-array numpy "
            "instead. The section 'Working with the weights' in your instructions "
            "shows how, with examples."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Numpy code to run."}
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
            "Commit the current weights and score them on the validation split. "
            f"You have {limit} evaluations in total, so make each one count."
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
        limit=limit,
    )


# -------------------------------------------------------------------------------- forward


def forward_tool(limit: int = 50) -> Tool:
    def handler(session: Session, budget: Budget, indices: list[int]) -> str:
        split = session.experiment.validation()
        indices = [int(i) for i in indices]

        if not indices or min(indices) < 0 or max(indices) >= len(split):
            raise ToolError(f"Indices must be between 0 and {len(split) - 1}.")

        weights = session.current_weights()
        budget.spend(len(indices))

        session.experiment.set_weights(weights)
        activations = session.experiment.activations(split.x[indices])

        result = {
            "indices": np.array(indices),
            "labels": split.y[indices],
            "activations": activations,
        }
        session.sandbox.put("forward_result", result)
        session.log.event("forward", indices=indices)

        output_layer = list(activations)[-1]
        outputs = activations[output_layer]
        table = views.render_table(
            outputs,
            row_labels=[
                f"#{i} (label {label})" for i, label in zip(indices, split.y[indices])
            ],
            col_labels=list(range(outputs.shape[1])),
        )

        shapes = ", ".join(
            f"{name} {array.shape}" for name, array in activations.items()
        )
        return (
            f"Output layer '{output_layer}' for {len(indices)} validation examples:\n\n{table}\n\n"
            "Every layer's activations are in the sandbox variable `forward_result` "
            "(keys: indices, labels, activations). "
            f"Shapes: {shapes}.\n\nExamples: {budget}."
        )

    return Tool(
        name="forward",
        description=(
            "Run the current (uncommitted) weights on specific validation examples. "
            "Prints the output layer and stores every layer's activations in the sandbox "
            "as `forward_result`. "
            f"Each example counts against a budget of {limit} examples in total. "
            "Use it to look at inputs and understand mistakes, not to score or tune "
            "designs: `evaluate` is your measure of success."
        ),
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Positions in the validation split.",
                }
            },
            "required": ["indices"],
        },
        handler=handler,
        limit=limit,
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

        session.sandbox.put(
            "samples", {"indices": drawn, "x": split.x[drawn], "y": split.y[drawn]}
        )
        session.log.event("samples", indices=new)

        return (
            f"Drew {count} new training examples with labels {split.y[new].tolist()}.\n\n"
            f"All {len(drawn)} drawn so far are in the sandbox variable `samples` "
            f"(keys: indices, x, y); x has shape {split.x[drawn].shape}.\n\n"
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
