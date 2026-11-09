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
