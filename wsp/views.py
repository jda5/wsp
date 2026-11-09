"""
LLM-friendly text views of numpy arrays.

Everything here is plain numpy, so it can run inside the sandbox. The
functions return strings; the sandbox wraps them in helpers that print.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

MAX_ROWS = 20
MAX_COLS = 12


# -------------------------------------------------------------------------------- arrays


def render_array(
    name: str,
    array: np.ndarray,
    rows: Any = None,
    cols: Any = None,
    precision: int = 3,
) -> str:
    """
    Render a 1-D or 2-D array as a table with row and column indices.

    `rows` and `cols` accept anything numpy accepts as an index
    (an int, a slice, a list of ints). Indices in the table are always the
    original ones, so a slice of `W["dense.kernel"]` still shows which
    input and unit each number belongs to.
    """
    array = np.asarray(array)

    if array.ndim == 0:
        return f"{name} = {_format_number(array.item(), precision, signed=True)}"

    if array.ndim > 2:
        raise ValueError(
            f"{name} has shape {array.shape}; show() displays 1-D or 2-D arrays. "
            f"Index it down first, e.g. {name}[..., 0]."
        )

    is_vector = array.ndim == 1
    matrix = array.reshape(-1, 1) if is_vector else array

    row_ids = _select(matrix.shape[0], rows, MAX_ROWS)
    col_ids = [0] if is_vector else _select(matrix.shape[1], cols, MAX_COLS)

    header = f"{name}  shape {array.shape}  showing {_describe(row_ids, matrix.shape[0], 'rows')}"
    if not is_vector:
        header += f", {_describe(col_ids, matrix.shape[1], 'cols')}"

    body = render_table(
        matrix[np.ix_(row_ids, col_ids)],
        row_labels=row_ids,
        col_labels=["value"] if is_vector else col_ids,
        precision=precision,
        signed=True,
    )

    return f"{header}\n{body}"


def render_table(
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    precision: int = 3,
    signed: bool = False,
) -> str:
    """
    Render a 2-D array as right-aligned columns, pandas style.

    `signed` writes an explicit and on positive numbers, which makes
    weights easier to scan.
    """
    cells = [
        [_format_number(value, precision, signed) for value in row] for row in matrix
    ]

    labels = [str(label) for label in row_labels]
    label_width = max(len(label) for label in labels)

    widths = [
        max(len(str(col_labels[j])), *(len(row[j]) for row in cells))
        for j in range(len(col_labels))
    ]

    header = (
        " " * label_width
        + "  "
        + "  ".join(str(label).rjust(width) for label, width in zip(col_labels, widths))
    )

    lines = [header]
    for label, row in zip(labels, cells):
        values = "  ".join(cell.rjust(width) for cell, width in zip(row, widths))
        lines.append(f"{label.rjust(label_width)}  {values}")

    return "\n".join(lines)


# -------------------------------------------------------------------------------- summaries


def summarize_weights(weights: Mapping[str, np.ndarray]) -> str:
    """
    One line per tensor: name, shape and basic statistics.
    """
    names = list(weights)
    stats = np.array([_statistics(weights[name]) for name in names])
    columns = ["mean", "std", "min", "max", "zeros"]

    shapes = [str(tuple(weights[name].shape)) for name in names]
    labels = [f"{name} {shape}" for name, shape in zip(names, shapes)]

    return render_table(stats, row_labels=labels, col_labels=columns, precision=4)


def render_scores(scores: Mapping[str, Any]) -> str:
    """
    Render a dict of metrics: scalars on one line, arrays as tables.
    """
    blocks = []

    for name, value in scores.items():
        value = np.asarray(value)

        if value.ndim == 0:
            blocks.append(f"{name}: {_format_number(value.item(), 4)}")
        elif value.ndim == 1:
            table = render_table(value.reshape(1, -1), [""], list(range(len(value))), 4)
            blocks.append(f"{name}:\n{table}")
        else:
            table = render_table(
                value, list(range(value.shape[0])), list(range(value.shape[1])), 4
            )
            blocks.append(f"{name}:\n{table}")

    return "\n\n".join(blocks)


# -------------------------------------------------------------------------------- private


def _select(length: int, index: Any, limit: int) -> list[int]:
    """
    Turn an index into a list of positions, keeping at most `limit`.
    """
    positions = np.arange(length)
    if index is not None:
        positions = np.atleast_1d(positions[index])
    return positions[:limit].tolist()


def _describe(ids: list[int], total: int, unit: str) -> str:
    """
    Say which rows or columns are shown, e.g. "rows 0-19 of 784".
    """
    is_contiguous = ids == list(range(ids[0], ids[0] + len(ids)))
    if is_contiguous:
        return f"{unit} {ids[0]}-{ids[-1]} of {total}"
    return f"{len(ids)} of {total} {unit}"


def _statistics(array: np.ndarray) -> list[float]:
    """
    Mean, std, min, max and the fraction of exact zeros.
    """
    array = np.asarray(array, dtype=np.float64)
    return [
        array.mean(),
        array.std(),
        array.min(),
        array.max(),
        np.mean(array == 0),
    ]


def _format_number(value: Any, precision: int, signed: bool = False) -> str:
    """
    Integers as integers. Floats to `precision` places, never as -0.000,
    and with a + on positive numbers if `signed`.
    """
    if isinstance(value, (int, np.integer)):
        return str(value)

    if float(f"{value:.{precision}f}") == 0:
        return f"{0:.{precision}f}"

    sign = "+" if signed else ""
    return f"{value:{sign}.{precision}f}"
