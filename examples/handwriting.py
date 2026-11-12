"""
Ask your agent, or yourself, to write the weights of a handwriting synthesis
network, starting from all zeros, until evaluation stops improving.

    uv run python -m examples.handwriting
    uv run python -m examples.handwriting --data ~/handwriting-data
    uv run python -m examples.handwriting --agent human

The data is not in the repository: see wsp/experiments/handwriting.py for
what the data folder must hold.

After the run, the run folder also holds:

    final.weights.h5   the final weights, to use as the starting point for training
    generated.svg      a line of handwriting written with the final weights

To train from the agent's weights, copy final.weights.h5 into a run folder of
the notebook `04-model-improved` as model.weights.h5, and uncomment the
notebook's `model.load_weights(run_dir / "model.weights.h5")`.

Follow a run from another terminal with `uv run python -m wsp.watch --follow`.
"""

import argparse
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from examples.utils import agent_arguments, get_agent
from wsp.experiments.handwriting import DATA, HandwritingExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--evaluations", type=int, default=25)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=150)
    agent_arguments(parser, model="claude-fable-5-2")
    args = parser.parse_args()

    load_dotenv()

    experiment = HandwritingExperiment(args.data.expanduser())

    config = RunConfig(
        tools=standard_tools(
            evaluations=args.evaluations,
            samples=args.samples,
        ),
        max_turns=args.max_turns,
    )

    agent = get_agent(args.agent, {"model": args.model, "effort": args.effort})
    folder = run(experiment, agent, config)

    # The experiment holds the final weights after the run's last test score.
    experiment.model.save_weights(folder / "final.weights.h5")

    strokes = experiment.generate("Weight-space programming")
    (folder / "generated.svg").write_text(draw_svg(strokes))
    print(f"\nA line written with the final weights: {folder / 'generated.svg'}")


def draw_svg(strokes: np.ndarray, margin: float = 2.0) -> str:
    """
    Draw rows of strokes (dx, dy, pen_down, pen_up, eos) as an SVG image:
    one line for every stretch the pen spent on the paper.
    """
    points = strokes[:, :2].cumsum(axis=0)
    points = points - points.min(axis=0) + margin
    width, height = points.max(axis=0) + margin

    lifts = np.flatnonzero(
        strokes[:, 2] == 0
    )  # the rows after which the pen is in the air
    lines = [
        '<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in stroke) + '"/>'
        for stroke in np.split(points, lifts + 1)
        if len(stroke) > 1
    ]

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        'fill="none" stroke="black" stroke-width="0.1" stroke-linecap="round">\n'
        + "\n".join(lines)
        + "\n</svg>\n"
    )


if __name__ == "__main__":
    main()
