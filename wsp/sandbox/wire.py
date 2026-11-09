"""
How the harness and the Docker worker talk.

    control   one JSON object per line, over the worker's stdin and stdout
    data      .npz files: nested dicts of arrays, flattened to "a/b/c" keys

Nothing is ever unpickled: .npz files are always loaded with
allow_pickle=False, so data written by agent code cannot run code on the host.
"""

import io
import json
import zipfile
from pathlib import Path
from typing import IO, Any

import numpy as np

SEPARATOR = "/"

# Where things live inside the container.
INBOX = "/inbox"  # read-only: data the harness sends in
WORKSPACE = "/workspace"  # read-write: the agent's working folder
WEIGHTS_FILE = ".weights.npz"  # in WORKSPACE: weights the worker sends out


# -------------------------------------------------------------------------------- control messages


def send(stream: IO[str], message: dict[str, Any]) -> None:
    stream.write(json.dumps(message) + "\n")
    stream.flush()


def receive(stream: IO[str]) -> dict[str, Any] | None:
    """
    The next message, or None once the other side has closed the stream.
    """
    line = stream.readline()
    return json.loads(line) if line else None


# -------------------------------------------------------------------------------- data


def save_tree(path: str | Path | IO[bytes], tree: dict) -> None:
    """
    Save a dict whose leaves are arrays or numbers, e.g. forward_result.

    Writes the .npz file directly rather than with np.savez(path, **arrays):
    that passes arrays as keyword arguments, so one named "file" or
    "allow_pickle" would collide with np.savez's own parameters.
    """
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key, array in _flatten(tree).items():
            with archive.open(f"{key}.npy", mode="w") as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def load_tree(source: str | Path | IO[bytes] | bytes) -> dict:
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    with np.load(source, allow_pickle=False) as data:
        return _unflatten({key: data[key] for key in data.files})


def _flatten(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    flat = {}

    for key, value in tree.items():
        key = str(key)
        if SEPARATOR in key:
            raise ValueError(f"Key '{key}' cannot contain '{SEPARATOR}'.")

        path = f"{prefix}{key}"

        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=path + SEPARATOR))
            continue

        array = np.asarray(value)
        if array.dtype == object:
            raise ValueError(f"'{path}' is not an array of numbers or strings.")
        flat[path] = array

    return flat


def _unflatten(flat: dict[str, np.ndarray]) -> dict:
    tree: dict = {}

    for path, value in flat.items():
        *parents, leaf = path.split(SEPARATOR)
        node = tree
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value

    return tree
