"""
Ask your agent, or yourself, to write the weights of a character-level LSTM
that predicts Shakespeare, starting from all zeros, until evaluation stops
improving.

    uv run python -m examples.shakespeare_zeros
    uv run python -m examples.shakespeare_zeros --model claude-fable-5-2
    uv run python -m examples.shakespeare_zeros --agent human

After the run, the run folder also holds:

    final.weights.h5   the final weights, to use as the starting point for training
    generated.txt      a passage generated from the final weights

To train from the agent's weights, build the same architecture (one-hot, LSTM(256),
Dense(65)) and call model.load_weights("final.weights.h5") before model.fit().
Keras matches the layers by position, not by name, and the Dense layer's
weights are the same whether or not it has a softmax.

Follow a run from another terminal with `uv run python -m wsp.watch --follow`.
"""

import argparse

from dotenv import load_dotenv

from examples.utils import agent_arguments, get_agent
from wsp.experiments.shakespeare_zeros import ShakespeareZerosExperiment
from wsp.run import RunConfig, run
from wsp.tools.standard_tools import standard_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=100)
    agent_arguments(parser, model="claude-fable-5-2")
    args = parser.parse_args()

    load_dotenv()

    experiment = ShakespeareZerosExperiment()

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

    passage = experiment.generate("ROMEO:\n", length=400)
    (folder / "generated.txt").write_text(passage)
    print(f"\n{passage}")


if __name__ == "__main__":
    main()
