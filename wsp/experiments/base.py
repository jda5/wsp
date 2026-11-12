"""
An experiment is the thing the agent edits: a model, the data it works on,
and what "good" means. The rest of the harness talks only to this class.

Subclasses fill in three groups of methods:

    what the agent is told    description, prompt (and optionally rules, sandbox_variables)
    the data                  train, validation, test, metrics
    the model                 get_weights, set_weights, predict, activations

Framework base classes such as `KerasExperiment` implement the model group,
so a concrete experiment usually writes only the first two.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from wsp import views

Weights = dict[str, np.ndarray]
Inputs = np.ndarray | dict[str, np.ndarray]


@dataclass(frozen=True)
class Split:
    """
    Examples and their labels. `x` is one array or, for a model with several
    inputs, a dict of arrays that each have one row per example.
    """

    x: Inputs
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def take(self, rows: Any) -> Split:
        """
        The examples at `rows`, as a smaller Split.
        """
        if isinstance(self.x, dict):
            x = {name: array[rows] for name, array in self.x.items()}
        else:
            x = self.x[rows]

        return Split(x, self.y[rows])


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

    def rules(self) -> str | None:
        """
        The rules the agent works under, which open the system prompt. None
        means the standard rules (PROMPT in wsp/run.py): no training, no
        fitting, and `evaluate` as the only measure.
        """
        return None

    def sandbox_variables(self) -> dict[str, Any]:
        """
        Extra variables placed in the agent's sandbox at the start, e.g. a
        vocabulary that maps input indices to words. Values are arrays,
        numbers, or dicts of them. None by default.
        """
        return {}

    # -------------------------------------------------------------------------------- the data

    @abstractmethod
    def train(self) -> Split:
        """
        The data a model would normally be trained on.
        """

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
    def predict(self, x: Inputs) -> np.ndarray:
        """
        The model's outputs for a batch of inputs.
        """

    @abstractmethod
    def activations(self, x: Inputs) -> dict[str, np.ndarray]:
        """
        Every layer's output for a batch of inputs, keyed by layer name.
        """

    # -------------------------------------------------------------------------------- shared

    def score(self, split: Split) -> dict[str, Any]:
        """
        Run the model, as it is now, on `split` and score it.
        """
        return self.metrics(self.predict(split.x), split.y)

    def forward_summary(
        self, positions: list[int], examples: Split, activations: dict[str, np.ndarray]
    ) -> str:
        """
        What the `forward` tool prints after running `examples`, the samples
        at `positions`. By default, the output layer: as a table when there is
        one row of outputs per sample, otherwise only its shape.
        """
        name = list(activations)[-1]
        outputs = activations[name]

        if outputs.ndim != 2:
            return f"Ran {len(positions)} of your samples; output layer '{name}' has shape {outputs.shape}."

        table = views.render_table(
            outputs,
            row_labels=[
                f"sample {i} (label {label})" for i, label in zip(positions, examples.y)
            ],
            col_labels=list(range(outputs.shape[1])),
        )
        return f"Output layer '{name}' for {len(positions)} of your samples:\n\n{table}"
