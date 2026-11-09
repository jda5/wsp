"""
A persistent Python namespace, running in its own process, where the agent
reads and edits the weights with numpy.

The namespace holds:

    W       dict of weight name -> numpy array (the agent edits these)
    np      numpy
    show    show(name_or_array, rows=None, cols=None, precision=3)
    info    info() prints every tensor's shape and statistics

This is a guardrail, not a security boundary. Machine learning frameworks
and networking modules cannot be imported, but determined code could still
reach the file system. The run log records every line of code, so anything
unusual is visible afterwards.
"""

import ast
import importlib.abc
import io
import multiprocessing
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Self

import numpy as np

from wsp import views

Weights = dict[str, np.ndarray]

BLOCKED_MODULES = {
    "tensorflow",
    "keras",
    "torch",
    "jax",
    "sklearn",
    "socket",
    "urllib",
    "http",
    "requests",
    "subprocess",
}


@dataclass(frozen=True)
class Execution:
    """
    The result of running one cell of code.

    ``changed`` maps each weight that changed to how many values changed.
    """

    output: str
    changed: dict[str, int]


# -------------------------------------------------------------------------------- errors


class SandboxError(Exception):
    """
    A request to the sandbox failed.
    """


class SandboxTimeout(SandboxError):
    """
    Code ran for longer than the timeout. The sandbox process was killed.
    """


# -------------------------------------------------------------------------------- sandbox


class Sandbox:
    """
    Runs agent code in a separate process that keeps its variables between calls.
    """

    def __init__(self, weights: Weights, timeout: float = 60.0):
        self.timeout = timeout
        self._start(weights)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ---- the agent's interface

    def run(self, code: str) -> Execution:
        """
        Execute ``code``. Like a notebook, the last expression is printed.
        """
        return self._request("run", code)

    # ---- the harness's interface

    def get_weights(self) -> Weights:
        return self._request("get")

    def set_weights(self, weights: Weights) -> None:
        self._request("set", weights)

    def put(self, name: str, value: object) -> None:
        """
        Place a variable in the namespace, e.g. the results of a forward pass.
        """
        self._request("put", (name, value))

    def restart(self, weights: Weights) -> None:
        """
        Start a fresh process. Every variable except ``W`` is lost.
        """
        self.close()
        self._start(weights)

    def close(self) -> None:
        self._connection.close()
        self._process.kill()
        self._process.join()

    # -------------------------------------------------------------------------------- private

    def _start(self, weights: Weights) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()

        self._process = context.Process(target=_serve, args=(child,), daemon=True)
        self._process.start()
        child.close()

        self.set_weights(weights)

    def _request(self, command: str, payload: object = None) -> object:
        self._connection.send((command, payload))

        if not self._connection.poll(self.timeout):
            self._process.kill()
            self._process.join()
            raise SandboxTimeout(f"Code ran for more than {self.timeout:.0f} seconds.")

        status, result = self._connection.recv()
        if status == "error":
            raise SandboxError(result)

        return result


# -------------------------------------------------------------------------------- worker process


def _serve(connection) -> None:
    """
    The sandbox process: answer requests until the connection closes.
    """
    _block_imports(BLOCKED_MODULES)
    namespace = _new_namespace()

    while True:
        try:
            command, payload = connection.recv()
        except EOFError:
            return

        try:
            result = _handle(command, payload, namespace)
            connection.send(("ok", result))
        except Exception as error:
            connection.send(("error", f"{type(error).__name__}: {error}"))


def _handle(command: str, payload: object, namespace: dict) -> object:
    match command:
        case "run":
            return _run(payload, namespace)

        case "get":
            return namespace["W"]

        case "set":
            namespace["W"] = {name: array.copy() for name, array in payload.items()}
            return None

        case "put":
            name, value = payload
            namespace[name] = value
            return None

    raise ValueError(f"Unknown command: {command}")


def _run(code: str, namespace: dict) -> Execution:
    """
    Execute a cell, capture what it prints, and report which weights changed.
    """
    before = {
        name: np.array(array, copy=True) for name, array in namespace["W"].items()
    }
    buffer = io.StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):
        try:
            _exec_cell(code, namespace)
        except Exception as error:
            print(_format_error(error))

    return Execution(
        output=buffer.getvalue(),
        changed=_count_changes(before, namespace["W"]),
    )


def _exec_cell(code: str, namespace: dict) -> None:
    """
    Run every statement; if the last one is an expression, print its value.
    """
    tree = ast.parse(code)

    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)

    exec(compile(tree, "<cell>", "exec"), namespace)

    if last is not None:
        value = eval(compile(last, "<cell>", "eval"), namespace)
        if value is not None:
            print(repr(value))


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


def _new_namespace() -> dict:
    # Short, summarised printing: large arrays show their corners with "...".
    # legacy="1.25" prints scalars as 0.5 rather than np.float32(0.5).
    np.set_printoptions(
        precision=3,
        suppress=True,
        threshold=200,
        edgeitems=3,
        linewidth=120,
        legacy="1.25",
    )
    namespace = {"__name__": "__sandbox__", "np": np, "W": {}}

    def show(x, rows=None, cols=None, precision: int = 3) -> None:
        """
        Print a 1-D or 2-D array (or a weight, by name) with its indices.

        rows and cols take any numpy index: an int, a slice, or a list of ints.
        At most 20 rows and 12 columns are shown.
        """
        name, array = (x, namespace["W"][x]) if isinstance(x, str) else ("array", x)
        print(views.render_array(name, array, rows, cols, precision))

    def info() -> None:
        """
        Print every weight's shape and statistics.
        """
        print(views.summarize_weights(namespace["W"]))

    namespace["show"] = show
    namespace["info"] = info
    return namespace


class _ImportBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, blocked: set[str]):
        self.blocked = blocked

    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] in self.blocked:
            raise ImportError(f"'{name}' cannot be imported in the sandbox")
        return None


def _block_imports(blocked: set[str]) -> None:
    """
    Stop blocked modules from being imported, including ones already loaded
    (the spawned process re-imports the main script, which may import keras).
    """
    for name in list(sys.modules):
        if name.partition(".")[0] in blocked:
            del sys.modules[name]

    sys.meta_path.insert(0, _ImportBlocker(blocked))
