"""
The standard tools: python, evaluate, forward, commit and reset.

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
) -> list[Tool]:
    """
    All five tools. `forward_examples=None` leaves out the forward tool.
    """
    tools = [
        python_tool(),
        evaluate_tool(evaluations),
        commit_tool(),
        reset_tool(),
    ]

    if forward_examples is not None:
        tools.append(forward_tool(forward_examples))

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
            "Run numpy code in a persistent namespace (variables carry over between "
            "calls). The weights are in `W`, a dict of name -> numpy array; edit them "
            "in place, e.g. `W['dense.bias'][3] += 0.5`.\n\n"
            "Available: `np` (the common numpy functions, plus np.linalg and "
            "np.random.default_rng); `show(name_or_array, rows=None, cols=None, "
            "precision=3)` prints a slice with its indices; `info()` prints every "
            "tensor's shape and statistics; and the builtins print, len, range, min, "
            "max, sum, abs, round, slice, sorted, zip, enumerate, list, tuple, dict, int, "
            "float, bool and str.\n\n"
            "Only assignments and expressions are allowed: no imports, loops, "
            "comprehensions, functions, lambdas or names starting with an underscore. "
            "Write vectorised numpy instead of loops, e.g. "
            "`W['dense.kernel'][:, :10] = templates.T`. "
            "The value of the last expression is printed, as in a notebook."
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
            f"Each example counts against a budget of {limit} examples in total."
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
        budget.spend()
        session.sandbox.set_weights(session.workspace.reset(to))
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
