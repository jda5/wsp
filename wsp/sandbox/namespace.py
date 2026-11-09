"""
A persistent namespace where cells of agent code run, notebook style. The
Docker worker (worker.py) keeps one alive inside the container.

The namespace holds:

    W       dict of weight name -> numpy array (the agent edits these)
    np      numpy
    show    show(name_or_array, rows=None, cols=None, precision=3)
    info    info() prints every tensor's shape and statistics
"""

import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from wsp import views

from .base import Execution, Weights

# Short, summarised printing: large arrays show their corners with "...".
# legacy="1.25" prints scalars as 0.5 rather than np.float32(0.5).
PRINT_OPTIONS = {
    "precision": 3,
    "suppress": True,
    "threshold": 1000,
    "edgeitems": 3,
    "linewidth": 250,
    "legacy": "1.25",
}


class Namespace:
    def __init__(self):
        self.variables = {
            "np": np,
            "show": self._show,
            "info": self._info,
            "W": {},
        }

    # -------------------------------------------------------------------------------- used by the agent

    def run(self, code: str) -> Execution:
        """
        Run a cell. Like a notebook, the value of the last expression is
        printed, and errors are printed into the output rather than raised.
        """
        tree = ast.parse(code)

        before = _copy_weights(self.weights)
        buffer = io.StringIO()

        with np.printoptions(**PRINT_OPTIONS), redirect_stdout(buffer), redirect_stderr(buffer):
            try:
                self._execute(tree)
            except (Exception, KeyboardInterrupt) as error:
                print(_format_error(error))

        return Execution(
            output=buffer.getvalue(),
            changed=_count_changes(before, self.weights),
        )

    # -------------------------------------------------------------------------------- used by the harness

    @property
    def weights(self) -> dict:
        return self.variables["W"]

    def get_weights(self) -> Weights:
        return self.weights

    def set_weights(self, weights: Weights) -> None:
        self.variables["W"] = {name: np.copy(array) for name, array in weights.items()}

    def put(self, name: str, value: object) -> None:
        """
        Place a variable in the namespace, e.g. the results of a forward pass.
        """
        self.variables[name] = value

    # -------------------------------------------------------------------------------- private

    def _execute(self, tree: ast.Module) -> None:
        """
        Run every statement; if the last one is an expression, print its value.
        """
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)

        exec(compile(tree, "<cell>", "exec"), self.variables)

        if last is not None:
            value = eval(compile(last, "<cell>", "eval"), self.variables)
            if value is not None:
                print(repr(value))

    def _show(self, x, rows=None, cols=None, precision: int = 3) -> None:
        """
        Print a 1-D or 2-D array (or a weight, by name) with its indices.

        rows and cols take any numpy index: an int, a slice, or a list of
        ints. At most 50 rows and 30 columns are shown.
        """
        name, array = (x, self.weights[x]) if isinstance(x, str) else ("array", x)
        print(views.render_array(name, array, rows, cols, precision))

    def _info(self) -> None:
        """
        Print every weight's shape and statistics.
        """
        print(views.summarize_weights(self.weights))


# -------------------------------------------------------------------------------- helpers


def _format_error(error: BaseException) -> str:
    """
    A traceback showing only the agent's own code, not the harness.
    """
    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if frame.filename == "<cell>"
    ]

    lines = ["Traceback (most recent call last):\n"] if frames else []
    lines += traceback.format_list(frames)
    lines += traceback.format_exception_only(error)

    return "".join(lines).rstrip()


def _copy_weights(weights: object) -> Weights:
    """
    A copy to compare against after a cell. Empty if the agent replaced W
    with something that is not a dict; the commit checks will catch that.
    """
    if not isinstance(weights, dict):
        return {}
    return {name: np.array(array, copy=True) for name, array in weights.items()}


def _count_changes(before: Weights, after: object) -> dict[str, int]:
    if not isinstance(after, dict):
        return {}

    changed = {}

    for name, array in after.items():
        old = before.get(name)
        new = np.asarray(array)

        if old is None or old.shape != new.shape:
            changed[name] = new.size
        elif (count := int(np.sum(old != new))) > 0:
            changed[name] = count

    return changed
