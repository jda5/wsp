"""
Ask your agent, or yourself, to write starting weights for the handwriting
synthesis network under production rules: the agent may fit to the training
lines, and is pointed at what training finds hard, reading the text.

    uv run python -m examples.handwriting_production
    uv run python -m examples.handwriting_production --data ~/handwriting-data
    uv run python -m examples.handwriting_production --agent human

The data is not in the repository: see wsp/experiments/handwriting.py for
what the data folder must hold.

After the run, the run folder also holds:

    final.weights.h5   the final weights, to use as the starting point for training
    generated.svg      a line of handwriting written with the final weights

To train from the agent's weights, call `model.load_weights` on
final.weights.h5 in the notebook `04-model-improved`, before `model.fit`.

Follow a run from another terminal with `uv run python -m wsp.watch --follow`.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from examples.handwriting import draw_svg
from examples.utils import agent_arguments, get_agent
from wsp.experiments.handwriting_production import DATA, HandwritingProductionExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--evaluations", type=int, default=40)
    parser.add_argument("--samples", type=int, help="default: every training line")
    parser.add_argument("--max-turns", type=int, default=200)
    agent_arguments(parser, model="claude-fable-5-2")
    args = parser.parse_args()

    load_dotenv()

    experiment = HandwritingProductionExperiment(args.data.expanduser())

    config = RunConfig(
        tools=standard_tools(
            evaluations=args.evaluations,
            samples=args.samples or len(experiment.train()),
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


if __name__ == "__main__":
    main()
