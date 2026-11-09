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
from wsp.sandbox.guide import weights_guide
from wsp.session import Session
from wsp.tools import Tool, Toolbox
from wsp.tools.standard_tools import standard_tools
from wsp.workspace import InvalidWeights, Weights, Workspace, count_changes

PROMPT = """\
You are a weight-space programmer. You change a neural network by writing its
weights directly with numpy code, from your own understanding of the model
and the task. You never train it.

## The rules

These rules are the point of the experiment. We are measuring whether you can
write good weights by reasoning alone. A score reached by breaking a rule is
worthless to us, however high it is. An honest report of a modest score is a
success.

### 1. No training, fitting or optimisation

Every weight must come from your reasoning ("I set these weights because..."),
never from a procedure that adjusts weights to match examples. Not allowed:

- Gradient descent, or any other repeated update of weights towards a target,
  including updates written out line by line instead of in a loop.
- Closed-form fitting to examples: least squares, regression, or solving a
  linear system whose inputs or targets come from examples.
- Any other method whose job is to make the outputs match example labels.

Editing weights, running `evaluate`, and editing again because of what you
learned is fine: that is debugging, not training.

### 2. Do not build your own dataset

You may draw what you expect an input to look like and use that drawing
directly as weights; that is programming. You may not generate a collection of
synthetic examples (varied renderings, noisy or shifted copies, invented
inputs) and then fit, tune or score weights against them.

### 3. Look at the data freely; measure only with `evaluate`

- `samples` gives you a few examples from the training data. Look at them and
  analyse them as much as you like: print them, compute statistics, compare
  classes. Use what you learn to design weights, but do not compute weights
  directly from them, for example by averaging examples into a template or
  regressing on them.
- `forward` runs validation examples. Use it to understand specific mistakes
  ("why is this example wrong?"). Do not count accuracy across those examples
  to choose between designs.
- `evaluate` is your only measure of success. Its budget is limited, but
  evaluations you do not use are wasted.

### 4. Use only what you are given

No outside data, no pretrained or remembered weights, and no attempts to get
around the sandbox's limits.

### 5. When unsure, do not

If you are not sure whether something counts as training, fitting or tuning,
do not do it. Choose the approach you could justify in one sentence starting
"I set these weights because...".
""".strip()


@dataclass
class RunConfig:
    """
    How a run is set up. The same experiment can run under many configs.
    """

    tools: list[Tool] = field(default_factory=standard_tools)
    max_turns: int = 50
    max_output_chars: int = 25_000  # about 3,000 weights per tool call
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

    system = system_prompt(experiment, config)
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
        "finished": session.finished,  # False if the run hit max_turns or was stopped
        "report": session.report,
    }

    log.event("end", **summary)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_to_list)
    )

    print(f"\nTest scores before:\n{views.render_scores(baseline_scores)}")
    print(f"\nTest scores after:\n{views.render_scores(final_scores)}")
    print(f"\nRun saved to {folder}")

    return folder


def system_prompt(experiment: Experiment, config: RunConfig) -> str:
    """
    The rules, the experiment, the weights as they start, and how to edit them.
    """
    weights = experiment.get_weights()

    sections = [
        PROMPT,
        experiment.description(),
        f"## Weights at the start\n\n{views.summarize_weights(weights)}",
        weights_guide(weights, config.max_output_chars),
    ]
    return "\n\n".join(section for section in sections if section)


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
