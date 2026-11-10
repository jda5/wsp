"""
Two spirals: can an agent write the weights of a network that separates two
interleaved spirals, starting from all zeros?

A classic problem that is hard for neural networks to learn (Lang and
Witbrock, 1988). There is no known recipe for its weights, so the agent has
to work out for itself how to carve up the plane with ReLU units.

Both spirals follow one formula:

    angle  a  between START_ANGLE and END_ANGLE
    radius r = a / END_ANGLE
    spiral 0: (x, y) = ( r cos a,  r sin a)
    spiral 1: (x, y) = (-r cos a, -r sin a)    (spiral 0 turned half a turn)

plus a little Gaussian noise on every point.

The formula is a secret: the agent is told only that the points have two
classes, and must find the pattern in the samples it draws.
"""

import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment
from .metrics import binary_classification_metrics

TURNS = 3  # how many times each spiral winds around the centre
START_ANGLE = np.pi / 2
END_ANGLE = START_ANGLE + TURNS * 2 * np.pi
NOISE = 0.02  # standard deviation of the noise added to x and y


class SpiralsExperiment(KerasExperiment):
    def __init__(self, hidden_units: int = 64):
        self.name = "spirals-zeros"

        # Each split is drawn with its own seed, so no point is in two splits.
        self.splits = {
            "train": _make_spirals(points_per_spiral=1_000, seed=0),
            "validation": _make_spirals(points_per_spiral=500, seed=1),
            "test": _make_spirals(points_per_spiral=500, seed=2),
        }

        super().__init__(_build_model(hidden_units))
        self.zero_weights()

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        return f"""\
## Data

Points (x, y) in the plane, each labelled 0 or 1. Half the points are in each
class. Every coordinate lies roughly between -1 and 1.

How the points were made is not given: there is a pattern, and finding it
from the samples you draw is part of the task.

## Model

{self.describe_model()}

## How they connect

Input 0 is x and input 1 is y. The output is the probability that the point
is in class 1; the prediction is class 1 when it is above 0.5.

## Metrics

Accuracy, accuracy per class (0 and 1), and a confusion matrix where
row = true class and column = predicted class."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model gives every "
            "point a probability of 0.5 and puts them all in class 0. Write "
            "weights that tell the two classes apart as accurately as you can.\n\n"
            "You are not told how the points were made: finding the pattern in "
            "the samples is part of the task. Study them as much as you like. "
            "If you describe the pattern with a formula, choose its constants by "
            "reasoning about what you see, not by fitting the formula to the "
            "samples (for example with least squares or a search over values).\n\n"
            "You may run the network yourself on any points you like, for "
            "example to draw a map of which class it predicts where. Do not "
            "generate your own labelled points to score or tune weights "
            "against: `evaluate` is your only measure."
        )

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.splits["train"]

    def validation(self) -> Split:
        return self.splits["validation"]

    def test(self) -> Split:
        return self.splits["test"]

    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        return binary_classification_metrics(outputs, labels)


# -------------------------------------------------------------------------------- building blocks


def _make_spirals(points_per_spiral: int, seed: int) -> Split:
    """
    Points on both spirals, shuffled, with labels 0 and 1.
    """
    rng = np.random.default_rng(seed)

    angle = rng.uniform(START_ANGLE, END_ANGLE, size=points_per_spiral)
    radius = angle / END_ANGLE
    spiral_0 = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    spiral_1 = -spiral_0

    x = np.concatenate([spiral_0, spiral_1])
    x += rng.normal(scale=NOISE, size=x.shape)
    y = np.repeat([0, 1], points_per_spiral)

    order = rng.permutation(len(y))
    return Split(x[order].astype(np.float32), y[order])


def _build_model(hidden_units: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(hidden_units, activation="relu", name="dense_1"),
            keras.layers.Dense(hidden_units, activation="relu", name="dense_2"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )
