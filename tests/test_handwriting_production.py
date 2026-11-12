"""
The production handwriting experiment, on a tiny model and made-up lines: no
dataset and no Docker needed.

    uv run pytest tests/test_handwriting_production.py

Only what it adds to HandwritingExperiment is tested here: its rules, and
the measure of how far the window is from the pen.
"""

import json

import numpy as np
import pytest
from test_handwriting import CONFIG, make_line

from wsp.experiments.handwriting_production import (
    FORWARD_LINES,
    RULES,
    HandwritingProductionExperiment,
    writing_progress,
)
from wsp.run import PROMPT, RunConfig, system_prompt

MEAN, STD = [0.0, 0.0], 1.0  # a normalisation that changes nothing


def line(dx: list[float], padding: int = 0) -> np.ndarray:
    """
    The targets of a line whose pen moves right by `dx` at each step, then stops.
    """
    targets = np.zeros((len(dx) + 1 + padding, 5), dtype=np.float32)
    targets[: len(dx), 0] = dx
    targets[: len(dx), 2] = 1  # pen down...
    targets[len(dx), 4] = 1  # ...then eos
    return targets


@pytest.fixture(scope="module")
def experiment(tmp_path_factory) -> HandwritingProductionExperiment:
    folder = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)

    lines = [make_line(rng, steps=rng.integers(8, 30)) for _ in range(300)]  # type: ignore
    texts = [
        "".join(rng.choice(list("abcdefgh "), size=rng.integers(1, 13)))
        for _ in range(300)
    ]

    np.savez(folder / "strokes_data.npz", *lines)
    (folder / "texts.json").write_text(json.dumps(texts))
    (folder / "sources.json").write_text(json.dumps(["real"] * 300))
    (folder / "config.json").write_text(json.dumps(CONFIG))

    return HandwritingProductionExperiment(folder)


# -------------------------------------------------------------------------------- writing progress (numpy only)


def test_writing_progress_is_the_share_of_the_way_across_the_line():
    progress = writing_progress(line([1, 1, 2]), MEAN, STD)

    np.testing.assert_allclose(progress, [0.25, 0.5, 1, 1])


def test_writing_progress_holds_when_the_pen_goes_back():
    progress = writing_progress(line([2, -1, -1, 4]), MEAN, STD)

    np.testing.assert_allclose(progress, [0.5, 0.5, 0.5, 1, 1])


def test_writing_progress_ignores_padding_and_undoes_the_normalisation():
    mean, std = [0.5, 0.0], 2.0
    normalised = line([1, 1, 2], padding=3)
    normalised[:4, 0] = (normalised[:4, 0] - mean[0]) / std

    progress = writing_progress(normalised, mean, std)

    np.testing.assert_allclose(progress[:4], [0.25, 0.5, 1, 1])
    np.testing.assert_allclose(progress[4:], 1)


def test_writing_progress_takes_a_batch_of_lines():
    batch = np.stack([line([1, 1, 2]), line([3, 1], padding=1)])

    np.testing.assert_allclose(
        writing_progress(batch, MEAN, STD), [[0.25, 0.5, 1, 1], [0.75, 1, 1, 1]]
    )


# -------------------------------------------------------------------------------- the experiment (tiny Keras model)


def test_the_production_rules_open_the_system_prompt(experiment):
    system = system_prompt(experiment, RunConfig())

    assert system.startswith(RULES)
    assert PROMPT not in system


def test_evaluate_reports_how_far_the_window_is_from_the_pen(experiment):
    scores = experiment.score(experiment.validation())

    assert 0 <= scores["window_error"] <= CONFIG["text_length"]
    assert "loss" in scores and "text_gain" in scores


def test_forward_summary_ends_with_each_lines_window_error(experiment):
    lines = experiment.train().take([0, 1])
    summary = experiment.forward_summary([4, 7], lines, experiment.activations(lines.x))

    assert "window_error, in characters:\nsample 4: " in summary
    assert "\nsample 7: " in summary


def test_the_sandbox_gets_the_normalisation(experiment):
    variables = experiment.sandbox_variables()

    assert {"vocab", "initial_step", "mean", "std"} <= variables.keys()


def test_forward_refuses_more_lines_than_fit_in_memory(experiment):
    lines = experiment.train().take(range(FORWARD_LINES + 1))

    with pytest.raises(ValueError, match="at most"):
        experiment.activations(lines.x)
