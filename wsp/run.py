"""
`run` gives an experiment to an agent and records everything in a run folder:

    runs/<name>-<timestamp>/
        log.jsonl        every message, tool call and result
        snapshots/       baseline.npz and one .npz per commit
        summary.json     test scores before and after, budgets used
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from wsp import views
from wsp.agents import BaseAgent
from wsp.experiments import Experiment
from wsp.log import RunLog
from wsp.sandbox import Sandbox
from wsp.session import Session
from wsp.tools import Tool, Toolbox
from wsp.tools.standard_tools import standard_tools
from wsp.workspace import InvalidWeights, Weights, Workspace, count_changes

PROMPT = """\
You are a weight-space programmer. You change a neural network by editing its
weights directly with numpy code. You never train it.

Rules:
- No training of any kind: no gradient descent, no optimisers, and no loops
  that fit weights to data. Every change must come from your own reasoning
  about the model and the task.
- Before each tool call, explain what you are about to do and why.
- Your final weights are whatever is in W when you stop calling tools.
- When you are finished, reply without calling a tool and write a short
  report: what you changed, why, and what you observed.
"""


@dataclass
class RunConfig:
    """
    How a run is set up. The same experiment can run under many configs.
    """

    tools: list[Tool] = field(default_factory=standard_tools)
    max_turns: int = 50
    max_output_chars: int = 8_000
    prompt: str | None = None  # overrides experiment.prompt()


def run(
    experiment: Experiment,
    agent: BaseAgent,
    config: RunConfig | None = None,
    root: str | Path = "runs",
) -> Path:
    """
    Run `agent` on `experiment` and return the run folder.
    """
    config = config or RunConfig()

    folder = Path(root) / f"{experiment.name}-{datetime.now():%Y%m%d-%H%M%S}"
    log = RunLog(folder / "log.jsonl")

    baseline = experiment.get_weights()
    workspace = Workspace(baseline, folder / "snapshots")

    system = system_prompt(experiment)
    prompt = config.prompt or experiment.prompt()

    session = Session(experiment, Sandbox(baseline), workspace, log)
    toolbox = Toolbox(config.tools, session, config.max_output_chars)

    log.event(
        "start",
        experiment=experiment.name,
        agent=agent.name,
        system=system,
        prompt=prompt,
    )

    baseline_scores = session.score(experiment.test(), baseline)

    agent.run(system, prompt, toolbox, log, config.max_turns)

    final = _final_weights(session)
    final_scores = session.score(experiment.test(), final)

    changes = count_changes(baseline, final)

    summary = {
        "experiment": experiment.name,
        "agent": agent.name,
        "test_before": baseline_scores,
        "test_after": final_scores,
        "values_changed": sum(changes.values()),
        "values_total": sum(array.size for array in baseline.values()),
        "changes_by_tensor": changes,
        "commits": len(workspace.commits),
        "tool_usage": toolbox.usage(),
    }

    log.event("end", **summary)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_to_list)
    )

    print(f"\nTest scores before:\n{views.render_scores(baseline_scores)}")
    print(f"\nTest scores after:\n{views.render_scores(final_scores)}")
    print(f"\nRun saved to {folder}")

    return folder


def system_prompt(experiment: Experiment) -> str:
    weights = views.summarize_weights(experiment.get_weights())

    sections = [
        PROMPT,
        experiment.description(),
        f"## Weights at the start\n\n{weights}",
    ]
    return "\n\n".join(sections)


# -------------------------------------------------------------------------------- private


def _final_weights(session: Session) -> Weights:
    """
    Commit whatever is in W. If it is invalid, fall back to the last commit.
    """
    try:
        return session.commit("final").weights
    except InvalidWeights as error:
        session.log.event("final_weights_invalid", error=str(error))
        return session.workspace.head


def _to_list(value: np.ndarray):
    return value.tolist()
