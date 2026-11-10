"""
CartPole with a small policy network: can an agent write the weights of a
controller that balances a pole, starting from all zeros?

There is no labelled data in reinforcement learning, so a split here is a
set of episodes: `Split.x` holds one environment seed per episode, which
fixes the episode's random starting state. Scoring a split plays every
episode with the current weights and reports the returns.
"""

import gymnasium as gym
import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment

ENVIRONMENT = "CartPole-v1"
MAX_STEPS = 500  # CartPole-v1 ends every episode after 500 steps


class CartPoleExperiment(KerasExperiment):
    def __init__(self, validation_episodes: int = 100, test_episodes: int = 100):
        self.name = "cartpole-zeros"

        # Each split gets its own block of seeds, so no episode is in two splits.
        self.splits = {
            "train": _episodes(first_seed=0, count=100),
            "validation": _episodes(first_seed=10_000, count=validation_episodes),
            "test": _episodes(first_seed=20_000, count=test_episodes),
        }

        super().__init__(_build_model())
        self.zero_weights()

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        return f"""\
## Task

CartPole: a pole is hinged on top of a cart that moves left and right along a
track. At every step the model sees the state and pushes the cart left or
right. The goal is to keep the pole upright for as long as possible.

## Observations

The model's input is the state, 4 numbers:

- input 0: cart position, in metres. 0 is the centre; positive is right.
- input 1: cart velocity, in metres per second. Positive is moving right.
- input 2: pole angle, in radians. 0 is upright; positive is leaning right.
- input 3: pole angular velocity, in radians per second. Positive is falling right.

Every episode starts with all 4 values drawn uniformly from -0.05 to 0.05.

## Actions

- output 0: push the cart left
- output 1: push the cart right

The action taken is the output with the larger value (ties go to output 0).
Each push applies 10 N to the cart for 0.02 seconds.

## Episodes

The model gets a reward of 1 for every step the episode lasts. An episode ends
when the pole leans more than 12 degrees (0.2095 radians) from upright, when
the cart moves more than 2.4 metres from the centre, or after {MAX_STEPS} steps.
A perfect episode scores {MAX_STEPS}.

## Model

{self.describe_model()}

## Metrics

`evaluate` plays one episode for each of {len(self.validation())} fixed starting
states and reports:

- mean_return: the average number of steps the episodes lasted
- min_return and max_return: the shortest and longest episodes
- share_perfect: the share of episodes that lasted all {MAX_STEPS} steps"""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model gives both actions "
            "the same probability, always pushes left, and the pole falls within about "
            "10 steps. Write weights that keep the pole balanced for as long as you can.\n\n"
            "Do not write your own simulation of the cart and pole to test or tune "
            "weights: `evaluate` is your only way to measure a policy."
        )

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.splits["train"]

    def validation(self) -> Split:
        return self.splits["validation"]

    def test(self) -> Split:
        return self.splits["test"]

    def score(self, split: Split) -> dict:
        """
        Play one episode per seed in `split` and score the returns.
        """
        returns = self.play(seeds=split.x)
        return self.metrics(returns, split.y)

    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        """
        `outputs` are episode returns. There are no labels in CartPole.
        """
        return {
            "mean_return": float(outputs.mean()),
            "min_return": float(outputs.min()),
            "max_return": float(outputs.max()),
            "share_perfect": float(np.mean(outputs == MAX_STEPS)),
        }

    # -------------------------------------------------------------------------------- playing

    def play(self, seeds: np.ndarray) -> np.ndarray:
        """
        Play one episode per seed with the current weights, all side by side,
        and return how many steps each one lasted.

        The model runs once per step on every episode's state together, which
        is much faster than one call per episode per step.
        """
        environments = [gym.make(ENVIRONMENT) for _ in seeds]
        states = np.array(
            [env.reset(seed=int(seed))[0] for env, seed in zip(environments, seeds)]
        )

        returns = np.zeros(len(seeds))
        running = np.ones(len(seeds), dtype=bool)

        while running.any():
            # Episodes that have ended still get an action; it is just ignored.
            actions = self.predict(states).argmax(axis=1)

            for i in np.flatnonzero(running):
                states[i], reward, terminated, truncated, _ = environments[i].step(
                    actions[i]
                )
                returns[i] += reward
                running[i] = not (terminated or truncated)

        return returns


# -------------------------------------------------------------------------------- building blocks


def _episodes(first_seed: int, count: int) -> Split:
    """
    `count` episodes with consecutive seeds. y is unused: there are no labels.
    """
    seeds = np.arange(first_seed, first_seed + count)
    return Split(x=seeds, y=np.zeros(count))


def _build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(16, activation="relu", name="dense"),
            keras.layers.Dense(2, activation="softmax", name="output"),
        ]
    )
