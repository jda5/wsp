"""
MNIST with a small fully connected network: can an agent write the weights
of a digit classifier?
"""

from dataclasses import dataclass

import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment
from .metrics import classification_metrics


class MNISTExperiment(KerasExperiment):
    def __init__(self, validation_size: int = 5_000):
        self.name = "mnist-zeros"
        self.data = _load_dataset(validation_size)

        super().__init__(_build_model())

        self.zero_weights()

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        return f"""\
## Data

MNIST: 28x28 greyscale images of handwritten digits, 0 to 9.

Inputs are pixel intensities between 0 (black background) and 1 (white ink).

## Model

{self.describe_model()}

## How they connect

The Flatten layer reads the image row by row, so input i to the dense layer is
the pixel at row i // 28, column i % 28. Column j of dense.kernel is therefore
a 28x28 pixel pattern: W["dense.kernel"][:, j].reshape(28, 28).

Output j is the probability that the digit is j; the prediction is the largest.

## Metrics

Accuracy, accuracy per digit, and a confusion matrix where row = true digit
and column = predicted digit."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model predicts the same "
            "probabilities for every image. Write weights that classify handwritten "
            "digits as accurately as you can."
        )

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.data.train

    def validation(self) -> Split:
        return self.data.validation

    def test(self) -> Split:
        return self.data.test

    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        return classification_metrics(outputs, labels, num_classes=10)


# -------------------------------------------------------------------------------- building blocks


@dataclass(frozen=True)
class MNISTData:
    train: Split
    validation: Split
    test: Split


def _load_dataset(validation_size: int = 5_000) -> MNISTData:
    """
    MNIST scaled to [0, 1], with the last ``validation_size`` training
    images held out for validation.
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train, y_train = np.asarray(X_train) / 255.0, np.asarray(y_train)
    X_test, y_test = np.asarray(X_test) / 255.0, np.asarray(y_test)

    return MNISTData(
        train=Split(X_train[:-validation_size], y_train[:-validation_size]),
        validation=Split(X_train[-validation_size:], y_train[-validation_size:]),
        test=Split(X_test, y_test),
    )


def _build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense"),
            keras.layers.Dense(10, activation="softmax", name="output"),
        ]
    )
