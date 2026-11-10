"""
Two spirals: can an agent write the weights of a network that separates two
interleaved spirals, starting from all zeros?

A classic problem that is hard for neural networks to learn (Lang and
Witbrock, 1988). There is no known recipe for its weights, so the agent has
to work out for itself how to carve up the plane with ReLU units.

Both spirals follow one formula, which the agent is told:

    angle  a  between START_ANGLE and END_ANGLE
    radius r = a / END_ANGLE
    spiral 0: (x, y) = ( r cos a,  r sin a)
    spiral 1: (x, y) = (-r cos a, -r sin a)    (spiral 0 turned half a turn)

plus a little Gaussian noise on every point.
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

Two spirals: points in the plane, each on one of two interleaved spirals,
labelled 0 or 1 by which spiral it is on. Half the points are on each.

Every point is made by the same formula. Pick an angle a uniformly between
{START_ANGLE:.4f} (pi/2) and {END_ANGLE:.4f} (pi/2 + {TURNS} turns), and let the
radius be r = a / {END_ANGLE:.4f}, so r grows from about {START_ANGLE / END_ANGLE:.3f} to 1. Then:

- spiral 0: (x, y) = ( r cos a,  r sin a)
- spiral 1: (x, y) = (-r cos a, -r sin a), which is spiral 0 turned by half a turn

Finally, Gaussian noise with standard deviation {NOISE} is added to x and to y.

Each spiral winds {TURNS} times around the centre, anticlockwise as it moves
outwards. Along any line from the centre the two spirals alternate, about
{np.pi / END_ANGLE:.3f} apart.

## Model

{self.describe_model()}

## How they connect

Input 0 is x and input 1 is y. The output is the probability that the point
is on spiral 1; the prediction is spiral 1 when it is above 0.5.

## Metrics

Accuracy, accuracy per spiral (0 and 1), and a confusion matrix where
row = true spiral and column = predicted spiral."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model gives every "
            "point a probability of 0.5 and puts them all on spiral 0. Write "
            "weights that tell the two spirals apart as accurately as you can.\n\n"
            "You may run the network yourself on any points you like, for "
            "example to draw a map of which spiral it predicts where. Do not "
            "generate your own labelled spiral points to score or tune weights "
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
