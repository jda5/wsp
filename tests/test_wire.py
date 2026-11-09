"""
The data format between the harness and the Docker worker. No Docker needed.

    uv run pytest tests/test_wire.py
"""

import io

import numpy as np
import pytest

from wsp.sandbox import wire


def round_trip(tree: dict) -> dict:
    buffer = io.BytesIO()
    wire.save_tree(buffer, tree)
    return wire.load_tree(buffer.getvalue())


def test_nested_trees_round_trip_exactly():
    tree = {"labels": np.array([1, 2]), "activations": {"dense": np.ones((2, 3), np.float32)}}
    back = round_trip(tree)

    np.testing.assert_array_equal(back["labels"], tree["labels"])
    assert back["activations"]["dense"].dtype == np.float32


@pytest.mark.parametrize("name", ["file", "allow_pickle", "kwds"])
def test_names_that_clash_with_np_savez_parameters_are_kept(name):
    back = round_trip({name: np.arange(3)})
    np.testing.assert_array_equal(back[name], np.arange(3))


def test_object_arrays_are_refused():
    with pytest.raises(ValueError, match="not an array of numbers"):
        wire.save_tree(io.BytesIO(), {"bad": np.array([object()])})
