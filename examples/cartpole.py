"""
Ask your agent, or yourself, to write the weights of a CartPole policy,
starting from all zeros.

    uv run python -m examples.cartpole
    uv run python -m examples.cartpole --evaluations 5
    uv run python -m examples.cartpole --agent human

Follow a run from another terminal with `uv run python -m wsp.watch --follow`.
"""

import argparse

from dotenv import load_dotenv

from examples.utils import get_agent
from wsp.experiments.cartpole import CartPoleExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--agent", choices=["claude", "human"], default="claude")
    parser.add_argument("--model", default="claude-opus-5-5")
    args = parser.parse_args()

    load_dotenv()

    experiment = CartPoleExperiment()

    config = RunConfig(
        # No `samples` or `forward`: CartPole has no labelled examples to draw.
        tools=standard_tools(evaluations=args.evaluations, samples=None),
        max_turns=args.max_turns,
    )

    agent = get_agent(args.agent, {"model": args.model})
    run(experiment, agent, config)


if __name__ == "__main__":
    main()
