"""
Version control for weights: a baseline, a list of commits, and snapshots on disk.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Weights = dict[str, np.ndarray]


class InvalidWeights(ValueError):
    """
    Weights that do not match the baseline's names and shapes, or are not finite.
    """


@dataclass(frozen=True)
class Commit:
    number: int
    message: str
    weights: Weights
    path: Path


class Workspace:
    """
    Every commit is validated against the baseline and saved as an ``.npz`` file.
    """

    def __init__(self, baseline: Weights, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        self.baseline = _copy(baseline)
        self.commits: list[Commit] = []

        np.savez(self.folder / "baseline.npz", **self.baseline)

    @property
    def head(self) -> Weights:
        """
        The most recently committed weights, or the baseline.
        """
        return self.commits[-1].weights if self.commits else self.baseline

    def validate(self, weights: object) -> Weights:
        """
        Check names, shapes and that every value is finite. Returns a clean copy
        with the baseline's dtypes, or raises ``InvalidWeights`` listing every problem.
        """
        if not isinstance(weights, dict):
            raise InvalidWeights(f"W must be a dict, not {type(weights).__name__}")

        problems = []

        for name in weights.keys() - self.baseline.keys():
            problems.append(f"unknown weight '{name}'")

        for name, reference in self.baseline.items():
            if name not in weights:
                problems.append(f"'{name}' is missing")
                continue

            try:
                array = np.asarray(weights[name], dtype=reference.dtype)
            except TypeError, ValueError:
                problems.append(f"'{name}' is not numeric")
                continue

            if array.shape != reference.shape:
                problems.append(
                    f"'{name}' has shape {array.shape}, expected {reference.shape}"
                )
            elif not np.all(np.isfinite(array)):
                problems.append(f"'{name}' contains NaN or infinite values")

        if problems:
            raise InvalidWeights("Invalid weights:\n- " + "\n- ".join(problems))

        return {
            name: np.array(weights[name], dtype=reference.dtype)
            for name, reference in self.baseline.items()
        }

    def commit(self, weights: object, message: str) -> Commit:
        weights = self.validate(weights)

        number = len(self.commits) + 1
        path = self.folder / f"{number:03d}.npz"
        np.savez(path, **weights)

        commit = Commit(number, message, weights, path)
        self.commits.append(commit)
        return commit

    def reset(self, to: Literal["last_commit", "baseline"]) -> Weights:
        """
        Return a copy of the weights to go back to.
        """
        return _copy(self.head if to == "last_commit" else self.baseline)


# -------------------------------------------------------------------------------- helpers


def count_changes(before: Weights, after: Weights) -> dict[str, int]:
    """
    How many values differ in each tensor. Unchanged tensors are left out.
    """
    changes = {name: int(np.sum(before[name] != after[name])) for name in before}
    return {name: count for name, count in changes.items() if count > 0}


def _copy(weights: Weights) -> Weights:
    return {name: np.array(array, copy=True) for name, array in weights.items()}
