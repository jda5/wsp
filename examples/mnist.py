"""
Ask Claude to write the weights of an MNIST classifier.

    uv run python -m examples.mnist --start zeros
    uv run python -m examples.mnist --start trained --evaluations 5

The work stays under `if __name__ == "__main__":` because the sandbox
process re-imports this file when it starts.
"""

import argparse

from dotenv import load_dotenv

from wsp.agents import ClaudeAgent
from wsp.experiments.mnist import MNISTExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=["zeros", "trained"], default="zeros")
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--forward-examples", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--model", default="claude-opus-5-5")
    args = parser.parse_args()

    load_dotenv()

    experiment = MNISTExperiment(initial_state=args.start)

    config = RunConfig(
        tools=standard_tools(
            evaluations=args.evaluations,
            forward_examples=args.forward_examples,
        ),
        max_turns=args.max_turns,
    )

    run(experiment, ClaudeAgent(model=args.model), config)


if __name__ == "__main__":
    main()
