"""
The namespace where the agent reads and edits the weights.

It runs in the same Python process as the harness. What keeps the agent
contained is not a separate process but what the namespace holds and what
the checker lets code do:

    W           dict of weight name -> numpy array (the agent edits these)
    np          a curated numpy (see numpy_api.py)
    show        show(name_or_array, rows=None, cols=None, precision=3)
    info        info() prints every tensor's shape and statistics
    print, len, range, min, max, ...   a few safe builtins

The harness's own variables (the experiment, the test data, the workspace)
are never in this namespace, so agent code cannot reach them.
"""

import ast
import io
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np

from wsp import views

from .checker import check
from .numpy_api import curated_numpy

Weights = dict[str, np.ndarray]


def _import_numpy_only(name: str, *args, **kwargs):
    """
    Numpy imports its own internals lazily (e.g. to format an array as a
    string), and those imports use the agent's builtins. Allow numpy, and
    nothing else. Agent code cannot call this: names starting with _ are
    rejected by the checker.
    """
    if name.partition(".")[0] != "numpy":
        raise ImportError(f"'{name}' cannot be imported in the sandbox")
    return __import__(name, *args, **kwargs)


SAFE_BUILTINS = {
    "__import__": _import_numpy_only,
    "print": print,
    "len": len,
    "range": range,
    "slice": slice,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "zip": zip,
    "enumerate": enumerate,
    "map": map,  # safe without lambda: it can only map functions already in reach
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}

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


@dataclass(frozen=True)
class Execution:
    """
    The result of running one cell of code.

    `changed` maps each weight that changed to how many values changed.
    """

    output: str
    changed: dict[str, int]


class Sandbox:
    def __init__(self, weights: Weights):
        self.namespace = self._new_namespace()
        self.set_weights(weights)

    # -------------------------------------------------------------------------------- used by the agent

    def run(self, code: str) -> Execution:
        """
        Check ``code`` against the allow-list, then run it. Like a notebook,
        the value of the last expression is printed.

        Raises ``ForbiddenCode`` without running anything if the check fails.
        Errors while running are printed into the output, as in a notebook.
        """
        tree = check(code)

        before = {name: np.copy(array) for name, array in self.weights.items()}
        buffer = io.StringIO()

        with np.printoptions(**PRINT_OPTIONS), redirect_stdout(buffer):
            try:
                self._execute(tree)
            except Exception as error:
                print(_format_error(error))

        return Execution(
            output=buffer.getvalue(),
            changed=_count_changes(before, self.weights),
        )

    # -------------------------------------------------------------------------------- used by the harness

    @property
    def weights(self) -> dict:
        return self.namespace["W"]

    def get_weights(self) -> Weights:
        return self.weights

    def set_weights(self, weights: Weights) -> None:
        self.namespace["W"] = {name: np.copy(array) for name, array in weights.items()}

    def put(self, name: str, value: object) -> None:
        """
        Place a variable in the namespace, e.g. the results of a forward pass.
        """
        self.namespace[name] = value

    # -------------------------------------------------------------------------------- private

    def _execute(self, tree: ast.Module) -> None:
        """
        Run every statement; if the last one is an expression, print its value.
        """
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)

        exec(compile(tree, "<cell>", "exec"), self.namespace)

        if last is not None:
            value = eval(compile(last, "<cell>", "eval"), self.namespace)
            if value is not None:
                print(repr(value))

    def _new_namespace(self) -> dict:
        def show(x, rows=None, cols=None, precision: int = 3) -> None:
            """
            Print a 1-D or 2-D array (or a weight, by name) with its indices.

            rows and cols take any numpy index: an int, a slice, or a list of
            ints. At most 50 rows and 30 columns are shown.
            """
            name, array = (x, self.weights[x]) if isinstance(x, str) else ("array", x)
            print(views.render_array(name, array, rows, cols, precision))

        def info() -> None:
            """
            Print every weight's shape and statistics.
            """
            print(views.summarize_weights(self.weights))

        return {
            "__builtins__": SAFE_BUILTINS,
            "np": curated_numpy(),
            "show": show,
            "info": info,
        }


# -------------------------------------------------------------------------------- helpers


def _format_error(error: Exception) -> str:
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


def _count_changes(before: Weights, after: dict) -> dict[str, int]:
    changed = {}

    for name, array in after.items():
        old = before.get(name)
        new = np.asarray(array)

        if old is None or old.shape != new.shape:
            changed[name] = new.size
        elif (count := int(np.sum(old != new))) > 0:
            changed[name] = count

    return changed
