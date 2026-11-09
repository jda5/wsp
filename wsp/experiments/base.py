"""
An experiment is the thing the agent edits: a model, the data it works on,
and what "good" means. The rest of the harness talks only to this class.

Subclasses fill in three groups of methods:

    what the agent is told    description, prompt
    the data                  validation, test, metrics
    the model                 get_weights, set_weights, predict, activations

Framework base classes such as `KerasExperiment` implement the model group,
so a concrete experiment usually writes only the first two.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

Weights = dict[str, np.ndarray]


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


class Experiment(ABC):
    name: str

    # -------------------------------------------------------------------------------- what the agent sees

    @abstractmethod
    def description(self) -> str:
        """
        The data, the model, and how they connect, e.g. which input index is
        which pixel. This becomes the body of the system prompt.
        """

    @abstractmethod
    def prompt(self) -> str:
        """
        The first message to the agent: what to do from this starting point.
        """

    # -------------------------------------------------------------------------------- the data

    @abstractmethod
    def validation(self) -> Split:
        """
        Data the agent may evaluate on, within its budget.
        """

    @abstractmethod
    def test(self) -> Split:
        """
        Held-out data. Used only to score the start and end of a run.
        """

    @abstractmethod
    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """
        Score model outputs. Values may be numbers or numpy arrays.
        """

    # -------------------------------------------------------------------------------- the model

    @abstractmethod
    def get_weights(self) -> Weights:
        """
        Every trainable tensor, keyed by a stable name such as "dense.kernel".
        """

    @abstractmethod
    def set_weights(self, weights: Weights) -> None:
        """
        Load weights with the same names and shapes as `get_weights` returns.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        The model's outputs for a batch of inputs.
        """

    @abstractmethod
    def activations(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """
        Every layer's output for a batch of inputs, keyed by layer name.
        """

    # -------------------------------------------------------------------------------- shared

    def score(self, split: Split) -> dict[str, Any]:
        """
        Run the model, as it is now, on `split` and score it.
        """
        return self.metrics(self.predict(split.x), split.y)
