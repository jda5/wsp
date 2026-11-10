"""
Metrics shared between experiments.
"""

import numpy as np


def classification_metrics(
    outputs: np.ndarray, labels: np.ndarray, num_classes: int
) -> dict[str, float | np.ndarray]:
    """
    Accuracy, accuracy per class, and a confusion matrix
    (row = true class, column = predicted class).
    """
    predictions = outputs.argmax(axis=1)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)

    return {
        "accuracy": float(np.mean(predictions == labels)),
        "accuracy_per_class": confusion.diagonal() / confusion.sum(axis=1),
        "confusion_matrix": confusion,
    }


def binary_classification_metrics(
    outputs: np.ndarray, labels: np.ndarray
) -> dict[str, float | np.ndarray]:
    """
    The same metrics for a model with one sigmoid output, the probability of
    class 1. It predicts class 1 when that is above 0.5.
    """
    probability = outputs.reshape(-1)
    both_classes = np.column_stack([1 - probability, probability])

    return classification_metrics(both_classes, labels, num_classes=2)
