"""
The handwriting experiment, on a tiny model and made-up lines: no dataset and
no Docker needed.

    uv run pytest tests/test_handwriting.py

The numpy reference is what the agent is shown as the definition of the
model, so the first two tests check it against the two things it stands for:
the Keras model, and the notebook's loss.
"""

import json

import numpy as np
import pytest
import tensorflow as tf

from wsp.experiments import Split
from wsp.experiments import handwriting_reference as reference
from wsp.experiments.handwriting import HandwritingExperiment, _position
from wsp.experiments.handwriting_model import build_model

CONFIG = {
    "mean": [0.1, 0.0],
    "std": 0.5,
    "vocabulary": ["", "[UNK]", *" abcdefgh"],
    "text_length": 13,
    "units": 8,
    "embed_dim": 4,
    "mixtures": 3,
    "window_components": 2,
    "initial_step": 0.1,
}


def mdn_loss(y_true, y_pred, mixtures=CONFIG["mixtures"]):
    """
    The loss of the notebook `04-model-improved`, copied unchanged.
    """
    dx, dy, pen = tf.split(y_true, [1, 1, 3], axis=-1)
    logit_pi, mu_x, mu_y, raw_sigma_x, raw_sigma_y, raw_rho, pen_logits = tf.split(
        y_pred, [mixtures] * 6 + [3], axis=-1
    )
    sigma_x = tf.nn.softplus(raw_sigma_x) + 1e-2
    sigma_y = tf.nn.softplus(raw_sigma_y) + 1e-2
    rho = 0.95 * tf.tanh(raw_rho)

    zx, zy = (dx - mu_x) / sigma_x, (dy - mu_y) / sigma_y
    one_minus_rho2 = 1 - tf.square(rho)
    log_density = (
        -(tf.square(zx) + tf.square(zy) - 2 * rho * zx * zy) / (2 * one_minus_rho2)
        - tf.math.log(sigma_x) - tf.math.log(sigma_y)
        - 0.5 * tf.math.log(one_minus_rho2) - np.log(2 * np.pi)
    )
    offset_nll = -tf.reduce_logsumexp(tf.nn.log_softmax(logit_pi) + log_density, axis=-1)
    pen_nll = tf.nn.softmax_cross_entropy_with_logits(pen, pen_logits)

    valid = tf.reduce_sum(pen, axis=-1)  # 1 at real steps, 0 at padding
    drawing = 1 - pen[..., 2]            # no offset to predict at EOS
    loss = drawing * offset_nll + 3 * pen_nll
    return tf.reduce_sum(loss * valid) / tf.reduce_sum(valid)


def make_line(rng: np.random.Generator, steps: int) -> np.ndarray:
    """
    A made-up line as the dataset stores it: a start row, pen points with a
    lift every fifth point, and an eos row.
    """
    line = np.zeros((steps + 2, 5), dtype=np.float32)
    line[1:-1, :2] = rng.normal(size=(steps, 2))
    line[:-1, 2] = 1
    line[5:-1:5, 2:4] = [0, 1]
    line[-1, 4] = 1
    return line


@pytest.fixture(scope="module")
def experiment(tmp_path_factory) -> HandwritingExperiment:
    """
    An experiment on 300 made-up lines, with random weights in place of zeros.
    """
    folder = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)

    lines = [make_line(rng, steps=rng.integers(8, 30)) for _ in range(300)]
    texts = ["".join(rng.choice(list("abcdefgh "), size=rng.integers(1, 13))) for _ in range(300)]

    np.savez(folder / "strokes_data.npz", *lines)
    (folder / "texts.json").write_text(json.dumps(texts))
    (folder / "sources.json").write_text(json.dumps(["real"] * 300))
    (folder / "config.json").write_text(json.dumps(CONFIG))

    experiment = HandwritingExperiment(folder)
    experiment.set_weights(
        {name: rng.normal(0, 0.3, w.shape).astype(np.float32) for name, w in experiment.get_weights().items()}
    )
    return experiment


def test_the_reference_computes_what_the_keras_model_computes(experiment):
    lines = experiment.train().take([0, 1, 2])
    params, phi = experiment.predict(lines.x)
    trace = experiment.activations(lines.x)

    for j, steps in enumerate(trace["steps"]):
        characters = len(lines.x["text"][j]) + 1

        np.testing.assert_allclose(trace["params"][j, :steps], params[j, :steps], atol=1e-4)
        np.testing.assert_allclose(trace["phi"][j, :steps, :characters], phi[j, :steps, :characters], atol=1e-5)


def test_the_reference_loss_is_the_notebooks_loss(experiment):
    lines = experiment.train()
    params, _ = experiment.predict(lines.x)

    scores = experiment.metrics(experiment.steps(lines), lines.y)

    assert scores["loss"] == pytest.approx(float(mdn_loss(lines.y, params)), abs=1e-4)


def test_every_weight_has_its_own_name(experiment):
    weights = experiment.get_weights()

    assert len(weights) == len(experiment.model.trainable_variables) == 14
    assert weights["decoder_1.lstm.kernel"].shape == (5 + 4, 4 * 8)
    assert weights["decoder_1.window.kernel"].shape == (8, 3 * 2)


def test_saved_weights_load_into_a_freshly_built_model(experiment, tmp_path):
    experiment.model.save_weights(tmp_path / "final.weights.h5")

    model, _ = build_model(CONFIG)
    model.load_weights(tmp_path / "final.weights.h5")

    for ours, theirs in zip(experiment.model.trainable_variables, model.trainable_variables):
        np.testing.assert_array_equal(ours.numpy(), theirs.numpy())


def test_the_splits_stay_out_of_the_notebooks_validation_and_test_lines(experiment):
    for split in (experiment.train(), experiment.validation(), experiment.test()):
        assert len(split) > 0
        assert all(_position(str(text)) >= 0.10 for text in split.x["text"])


def test_text_gain_is_zero_when_the_text_is_not_read(experiment):
    weights = experiment.get_weights()
    experiment.set_weights({**weights, "embedding.embeddings": np.zeros_like(weights["embedding.embeddings"])})

    assert experiment.score(experiment.validation())["text_gain"] == pytest.approx(0, abs=1e-6)

    experiment.set_weights(weights)


def test_forward_summary_has_a_row_for_the_characters_under_the_window(experiment):
    lines = experiment.train().take([0])
    summary = experiment.forward_summary([0], lines, experiment.activations(lines.x))

    assert summary.startswith(f"sample 0: {str(lines.x['text'][0])!r}")
    assert f"0 {str(lines.x['text'][0])[0]!r}" in summary


def test_generate_writes_rows_of_strokes(experiment):
    strokes = experiment.generate("abc", max_steps=20)

    assert strokes.shape[1] == 5 and 2 <= len(strokes) <= 21
    assert np.all(strokes[:, 2:].sum(axis=1) == 1)


def test_take_keeps_every_input_of_a_split_with_several():
    split = Split(x={"a": np.arange(4), "b": np.arange(8).reshape(4, 2)}, y=np.arange(4) * 10)
    taken = split.take([3, 1])

    np.testing.assert_array_equal(taken.x["a"], [3, 1])
    np.testing.assert_array_equal(taken.x["b"], [[6, 7], [2, 3]])
    np.testing.assert_array_equal(taken.y, [30, 10])
