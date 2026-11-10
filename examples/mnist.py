"""
Ask your agent, or yourself, to write the weights of an MNIST classifier.

    uv run python -m examples.mnist --start zeros
    uv run python -m examples.mnist --start trained --evaluations 5
    uv run python -m examples.mnist --agent human

Follow a run from another terminal with `uv run python -m wsp.watch --follow`.
"""

import argparse

from dotenv import load_dotenv

from examples.utils import agent_arguments, get_agent
from wsp.experiments.mnist import MNISTExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--max-turns", type=int, default=50)
    agent_arguments(parser)
    args = parser.parse_args()

    load_dotenv()

    experiment = MNISTExperiment()

    config = RunConfig(
        tools=standard_tools(
            evaluations=args.evaluations,
            samples=args.samples,
        ),
        max_turns=args.max_turns,
    )

    agent = get_agent(args.agent, {"model": args.model, "effort": args.effort})
    run(experiment, agent, config)


if __name__ == "__main__":
    main()
