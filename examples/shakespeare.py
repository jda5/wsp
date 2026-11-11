"""
Ask your agent, or yourself, to improve by hand the weights of a
character-level LSTM that predicts Shakespeare, starting from a model trained
with gradient descent for only 6 epochs.

    uv run python -m examples.shakespeare
    uv run python -m examples.shakespeare --samples 12 --evaluations 10 --max-turns 50
    uv run python -m examples.shakespeare --agent human

After the run, a passage generated from the final weights is printed and
saved in the run folder as generated.txt.

Follow a run from another terminal with `uv run python -m wsp.watch --follow`.
"""

import argparse

from dotenv import load_dotenv

from examples.utils import agent_arguments, get_agent
from wsp.experiments.shakespeare import ShakespeareExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=100)
    agent_arguments(parser, model="claude-opus-5-5")
    args = parser.parse_args()

    load_dotenv()

    experiment = ShakespeareExperiment()

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
    passage = experiment.generate("ROMEO:\n", length=400)
    (folder / "generated.txt").write_text(passage)
    print(f"\n{passage}")


if __name__ == "__main__":
    main()
