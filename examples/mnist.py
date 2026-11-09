"""
Ask your agent, or yourself, to write the weights of an MNIST classifier.

    uv run python -m examples.mnist --start zeros
    uv run python -m examples.mnist --start trained --evaluations 5
    uv run python -m examples.mnist --agent human
"""

import argparse

from dotenv import load_dotenv

from wsp.agents import BaseAgent, ClaudeAgent, HumanAgent
from wsp.experiments.mnist import MNISTExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=["zeros", "trained"], default="zeros")
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--forward-examples", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--agent", choices=["claude", "human"], default="claude")
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

    match args.agent:
        case "human":
            agent: BaseAgent = HumanAgent()
        case "claude":
            agent: BaseAgent = ClaudeAgent(model=args.model)
        case _:
            raise ValueError(f"Unknown agent: {args.agent}")

    run(experiment, agent, config)


if __name__ == "__main__":
    main()
