"""
Handwriting from zeros: can an agent write, from nothing, the weights of a
network that writes a line of text by hand?

Three LSTM layers and a Gaussian window over the text (handwriting_model.py),
trained on IAM-OnDB and DeepWriting. Every weight starts at zero, and the
agent keeps going until `evaluate` stops improving.

As in shakespeare_zeros, the point is to use the agent as an initialiser: its
final weights (saved by examples/handwriting.py as final.weights.h5) are a
starting point for the notebook's training, to compare with its usual random
initialisation.

The data is not in the repository. `data` is a folder that holds

    strokes_data.npz, texts.json, sources.json    the dataset made by `03-combine-data`
    config.json                                   saved by a run of `04-model-improved`
"""

import hashlib
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from wsp import views

from . import handwriting_reference as reference
from .base import Split
from .handwriting_model import build_model
from .keras_experiment import KerasExperiment

DATA = Path(__file__).resolve().parents[2] / "strokes_data"

# The notebook gives every line a position from 0 to 1, a hash of its text.
# Lines below 0.10 are its validation and test sets; it trains on the rest.
# The agent's three splits are bands of the lines the notebook trains on, so
# the notebook's own validation and test lines stay unseen, and the comparison
# that follows a run is fair.
BANDS = {"validation": (0.10, 0.15), "test": (0.15, 0.20), "train": (0.20, 0.25)}

# The pen states, in the order of the last three columns of a row of strokes
DOWN, UP, EOS = 0, 1, 2
START = np.array([[0, 0, 1, 0, 0]], dtype=np.float32)  # the first row of every line

# Losses for the agent to measure itself against. With zero weights, on this
# experiment's validation lines; and the notebook's val_loss after 1, 2 and 3
# epochs of training from a random initialisation.
ZERO_LOSS = 6.47
TRAINED_LOSSES = [-0.254, -0.546, -0.734]


class HandwritingExperiment(KerasExperiment):
    def __init__(self, data: Path = DATA):
        self.name = "handwriting-zeros"

        self.config = json.loads((data / "config.json").read_text())
        self.vocab = np.array(self.config["vocabulary"])
        self.splits = _load_splits(data, self.config)

        model, alignment = build_model(self.config)
        super().__init__(model, batch_size=128)
        self.zero_weights()

        # The outputs and the window weights from one pass
        self._outputs = keras.Model(model.inputs, [model.output, alignment.output])

    def name_variables(self) -> dict[str, keras.Variable]:
        """
        decoder_1 holds an LSTM and the window, each with a kernel and a bias,
        so its variables are named by the part they belong to.
        """
        cell = self.model.get_layer("decoder_1").cell

        parts = {
            "embedding": self.model.get_layer("embedding"),
            "decoder_1.lstm": cell.lstm,
            "decoder_1.window": cell.window,
            "decoder_2": self.model.get_layer("decoder_2"),
            "decoder_3": self.model.get_layer("decoder_3"),
            "mdn": self.model.get_layer("mdn"),
        }

        return {
            f"{part}.{variable.name}": variable
            for part, layer in parts.items()
            for variable in layer.trainable_weights
        }

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        H, E = self.config["units"], self.config["embed_dim"]
        K, M = self.config["window_components"], self.config["mixtures"]
        mean, std = self.config["mean"], self.config["std"]

        train = self.train()
        steps = _lengths(train.y)
        characters = np.char.str_len(train.x["text"])
        down, up, eos = train.y[..., 2:].sum(axis=(0, 1)) / steps.sum()

        return f"""\
## Task

Handwriting synthesis. The model writes a line of English text by hand, one
pen movement at a time. It is the network of Graves, "Generating Sequences
With Recurrent Neural Networks" (2013), section 5, with the changes described
below.

## Data

Lines of handwriting recorded from a pen (IAM-OnDB and DeepWriting, plus some synthetic data), 
each with the text it says. A line is a sequence of pen points, and each point is a row
of 5 numbers:

    dx, dy      how far the pen moved since the previous point. x grows to
                the right and y grows downwards.
    pen_down    1 if the pen stays on the paper after this point.
    pen_up      1 if the pen lifts after this point. The stroke ends here, and
                the next movement is through the air to where the next begins.
    eos         1 on the last row only, which ends the line.

Exactly one of pen_down, pen_up and eos is 1 in every row. Every line starts
with the row [0, 0, 1, 0, 0] and ends with the eos row.

The offsets are normalised. Their means ({mean[0]:.4f} for dx, {mean[1]:.4f} for dy) are
subtracted, and both are divided by the same standard deviation ({std:.4f}), which
keeps the handwriting's proportions. Over the whole data, normalised offsets
have mean 0, and dx and dy taken together have standard deviation 1.

About the {len(train):,} lines your samples are drawn from:

- A line has {steps.mean():.0f} pen steps on average (the longest has {steps.max():,}) and {characters.mean():.0f}
  characters (at most {self.config["text_length"] - 1}): about {np.mean(steps / characters):.1f} pen steps per character.
- The pen state to predict is pen_down at {down:.2%} of steps, pen_up at {up:.2%}
  and eos at {eos:.2%} (once per line).

The texts use {len(self.vocab) - 2} characters. Each has an ID: the sandbox variable `vocab`
holds them in order, so `vocab[i]` is the character with ID i. ID 0 is "no
character" and ID 1 is an unknown character; neither appears in a text.

    {"".join(self.vocab[2:])!r}

### Samples

`samples` gives you whole lines:

```python
samples["x"]["text"][i]       # the text of line i
samples["x"]["strokes"][i]    # the model's inputs: the line's rows, without its last (the eos row)
samples["y"][i]               # the targets: the line's rows, without its first
```

At every pen step t the model reads the row `strokes[t]` and predicts the next
one, `y[t]`. Lines are padded to the same length with rows of zeros; a padded
row has no pen state set, so `int(samples["y"][i][:, 2:].sum())` is the number
of real pen steps in line i.

## Model

The text is not encoded: each character is only looked up in `embedding.embeddings`,
which has one row of {E} numbers per character ID. Three LSTM layers of {H} units
then read the pen, one step at a time.

- decoder_1 is an LSTM that moves a window along the text. The window is a
  mixture of {K} Gaussians over the character positions, and it can only move
  forward. What the window sees is `w`: the embeddings of the characters,
  averaged with the window's weights `phi`. The text has one position past its
  last character, with ID 0; the window being there means the text has ended.
- decoder_2 and decoder_3 are plain LSTMs. Each reads the pen, the layer below
  and `w`.
- mdn is the output layer. It reads all three layers and gives {6 * M + 3} numbers:
  a mixture of {M} two-dimensional Gaussians for the next offset (dx, dy), and
  3 logits for the next pen state.

### Where everything is

In every LSTM, the last axis of kernel, recurrent_kernel and bias holds the
four gates side by side, {H} columns each: input gate {_span(0, H)}, forget gate
{_span(H, H)}, candidate {_span(2 * H, H)}, output gate {_span(3 * H, H)}.

    decoder_1.lstm.kernel      rows 0-4      the pen: dx, dy, pen_down, pen_up, eos
                               rows {_span(5, E)}     w from the step before
    decoder_1.window.kernel    rows          h1
                               columns       weight logits {_span(0, K)}, raw sigma {_span(K, K)}, raw step {_span(2 * K, K)}
    decoder_2.kernel and       rows 0-4      the pen
    decoder_3.kernel           rows {_span(5, H)}    the h of the layer below
                               rows {_span(5 + H, E)}  w
    mdn.kernel                 rows          h1 {_span(0, H)}, h2 {_span(H, H)}, h3 {_span(2 * H, H)}
                               columns       mixture logits {_span(0, M)}, mu_x {_span(M, M)}, mu_y {_span(2 * M, M)},
                                             raw sigma_x {_span(3 * M, M)}, raw sigma_y {_span(4 * M, M)}, raw rho {_span(5 * M, M)},
                                             pen logits {_span(6 * M, 3)} (pen_down, pen_up, eos)

### The model, exactly

This is the whole model in numpy. It computes the same numbers as the Keras
model that `evaluate` runs.

```python
{Path(reference.__file__).read_text().strip()}
```

You can use these functions in the sandbox:

```python
from wsp.experiments.handwriting_reference import *

line = 0
steps = int(samples["y"][line][:, 2:].sum())
ids = encode(samples["x"]["text"][line], vocab)

trace = forward(W, ids, samples["x"]["strokes"][line][:steps], initial_step)
offset_nll, pen_nll = losses(samples["y"][line][:steps], trace["params"])
```

`initial_step` is a sandbox variable, {self.config["initial_step"]:.4f}: one character for every {1 / self.config["initial_step"]:.1f} pen
steps, the average in the data. It is fixed, not a weight. When the window's
raw step is 0, every Gaussian moves forward by exactly this much.

## The forward tool

`forward` runs the reference model on your samples and prints, for each line,
one row per character: the pen steps during which that character had the most
window weight, how many times the pen lifted in them, and the mean offset_nll
and pen_nll there. Read it against what you know about letters: how many
strokes each takes, and where the pen should be in the air.

`forward_result["activations"]` holds every signal of `step` for those lines,
with a first axis for the line: `activations["phi"][j, t, u]`. Lines and texts
differ in length, so the arrays are padded with zeros; `activations["steps"][j]`
is the number of real pen steps of line j.

## Metrics

All are averages over pen steps, in nats. For the first five, lower is better.

- loss: the loss the model is trained on, the mean of offset_nll + 3 * pen_nll.
  It can go below zero, because offset_nll is a density.
- offset_nll: over every step with an offset to predict.
- offset_nll_within_stroke: over the steps where the pen was on the paper.
- offset_nll_after_lift: over the steps that follow a pen_up, where the pen
  moves through the air to the start of the next stroke.
- pen_nll: for the next pen state.
- p_down_when_down, p_up_when_up, p_eos_when_eos: the mean probability the
  model gave to the true pen state, for each state. Higher is better.
- window_at_end: how far through the text the window is when the line ends,
  averaged over lines. 1 means it has reached the position past the last
  character; lower means the pen stopped while the window was still mid-text.
- window_off_text: the share of pen steps at which the window has run so far
  past the end of the text that `phi` and `w` are nearly zero. 0 is ideal.
- text_gain: the loss when every line is given another line's text, minus the
  loss with its own. It is how many nats the model gains from reading the
  text: 0 means the text is not being used.

For scale: with every weight zero the loss is about {ZERO_LOSS}. Trained with
gradient descent from a random initialisation, this model reaches a loss of
{TRAINED_LOSSES[0]} after one epoch, {TRAINED_LOSSES[1]} after two and {TRAINED_LOSSES[2]} after three."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero. The window still moves, at "
            "the fixed pace of `initial_step`, but the embeddings are zero so it "
            "sees nothing. Every prediction is the same: one wide Gaussian "
            "centred on no movement, and three equally likely pen states. The "
            f"loss is about {ZERO_LOSS}. Write weights that bring the loss on "
            "`evaluate` as low as you can.\n\n"
            "Keep improving the weights until `evaluate` shows no further "
            "improvement.\n\n"
            "Your final weights will then be used as the starting point for "
            "training with gradient descent, in place of the usual random "
            "initialisation. `evaluate` is still how your weights are judged, "
            "but write them so the network stays trainable:\n\n"
            "- Avoid saturation where you can. A gate or unit pushed far into "
            "the flat part of its sigmoid or tanh (pre-activations like ±20) "
            "passes almost no gradient, so training will barely change it. "
            "Prefer the smallest weights that still do the job.\n"
            "- Do not leave units at exactly zero. A unit whose weights are all "
            "zero outputs zero and receives zero gradient, so it can never "
            "learn. This network has 1,200 units and you will probably give a "
            "meaning to only some of them. Give every unit you do not use "
            "small random incoming weights (for example normal with standard "
            "deviation 0.05, and a forget-gate bias of 1), and keep the weights "
            "that lead from those units into your own circuits and into the "
            "outputs at zero or very small, so that their noise does not "
            "disturb what you built. That is not fitting: it only gives "
            "training something to work with.\n"
            "- Training applies dropout of 0.1 to h1 and h2 on their way to "
            "the layer above and to the output layer, so do not rest a circuit "
            "on a single unit of those layers where a few would do.\n\n"
            "Your own knowledge is yours to use: how letters are formed, their "
            "strokes, widths, ascenders and descenders; where a pen lifts and "
            "how far it then travels; how smoothly a pen moves; how words are "
            "spaced; and how a trained network of this kind behaves. Looking "
            "at the samples to learn what the data is like is fine too.\n\n"
            "You may run the network yourself on text and strokes you write, "
            "to watch its state and predictions. Do not score or tune weights "
            "against strokes you generate: `evaluate` is your only measure."
        )

    def sandbox_variables(self) -> dict:
        return {"vocab": self.vocab, "initial_step": self.config["initial_step"]}

    def forward_summary(
        self, positions: list[int], examples: Split, activations: dict
    ) -> str:
        """
        For each line, the alignment of the window with the text, as a table.
        """
        tables = []

        for j, position in enumerate(positions):
            steps = activations["steps"][j]
            text = str(examples.x["text"][j])

            tables.append(
                _alignment_table(
                    title=f"sample {position}: {text!r}, {steps} pen steps",
                    text=text,
                    targets=examples.y[j, :steps],
                    phi=activations["phi"][j, :steps, : len(text) + 1],
                    params=activations["params"][j, :steps],
                )
            )

        return "\n\n".join(tables)

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.splits["train"]

    def validation(self) -> Split:
        return self.splits["validation"]

    def test(self) -> Split:
        return self.splits["test"]

    def score(self, split: Split) -> dict:
        """
        Score the lines of `split`, then score them again with every line
        given another line's text, to see what reading the text is worth.
        """
        other_texts = Split({**split.x, "text": np.roll(split.x["text"], 1)}, split.y)

        scores = self.metrics(self.steps(split), split.y)
        scores["text_gain"] = _loss(self.steps(other_texts)) - scores["loss"]
        return scores

    def metrics(self, outputs: dict, labels: np.ndarray) -> dict:
        """
        `outputs` is the table of pen steps that `steps` returns. It already
        holds what is needed from the labels.
        """
        offset_nll, pen_nll, pen = (
            outputs["offset_nll"],
            outputs["pen_nll"],
            outputs["pen"],
        )

        drawing = pen != EOS  # at the last step there is no offset to predict
        after_lift = outputs["after_lift"]
        p_true = np.exp(-pen_nll)

        scores = {
            "loss": _loss(outputs),
            "offset_nll": offset_nll[drawing].mean(),
            "offset_nll_within_stroke": offset_nll[drawing & ~after_lift].mean(),
            "offset_nll_after_lift": offset_nll[drawing & after_lift].mean(),
            "pen_nll": pen_nll.mean(),
            "p_down_when_down": p_true[pen == DOWN].mean(),
            "p_up_when_up": p_true[pen == UP].mean(),
            "p_eos_when_eos": p_true[pen == EOS].mean(),
            "window_at_end": outputs["window_progress"][pen == EOS].mean(),
            "window_off_text": outputs["window_off_text"].mean(),
        }
        return {name: float(value) for name, value in scores.items()}

    # -------------------------------------------------------------------------------- the model

    def predict(self, x: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        The outputs and the window weights at every pen step of the lines in `x`.
        """
        return self._outputs.predict_on_batch((tf.constant(x["text"]), x["strokes"]))

    def steps(self, split: Split) -> dict[str, np.ndarray]:
        """
        A table with one row for every real pen step of every line in `split`:
        the two parts of the loss, what the pen did, and where the window was.

        Lines are run in batches of similar length, each cut to its longest
        line, so that short lines are not run to the length of the longest.
        """
        lengths = _lengths(split.y)
        batches = np.array_split(
            np.argsort(lengths), max(1, len(split) // self.batch_size)
        )

        tables = [
            self._batch_steps(split.take(rows), lengths[rows].max()) for rows in batches
        ]
        return {
            name: np.concatenate([table[name] for table in tables])
            for name in tables[0]
        }

    def activations(self, x: dict) -> dict[str, np.ndarray]:
        """
        Every signal of the reference model at every pen step of the lines in
        `x`, with a first axis for the line. Lines and texts differ in length,
        so each array is padded with zeros; `steps` holds every line's length.
        """
        W = self.get_weights()

        traces = []
        for text, strokes in zip(x["text"], x["strokes"]):
            ids = reference.encode(str(text), self.vocab)
            real = strokes[:, 2:].sum(axis=1) > 0
            traces.append(
                reference.forward(W, ids, strokes[real], self.config["initial_step"])
            )

        activations = {
            name: _stack([trace[name] for trace in traces]) for name in traces[0]
        }
        activations["steps"] = np.array([len(trace["params"]) for trace in traces])
        return activations

    def generate(self, text: str, max_steps: int = 1500, seed: int = 0) -> np.ndarray:
        """
        Write `text` by hand with the current weights. Every row of strokes is
        drawn from the model's prediction and fed back in as the next input,
        until the model ends the line or `max_steps` is reached.

        Returns the rows with the normalisation of their offsets undone.
        """
        W = self.get_weights()
        rng = np.random.default_rng(seed)
        chars = W["embedding.embeddings"][reference.encode(text, self.vocab)]

        signals = reference.start(W)
        rows = [_normalise(START, self.config)[0]]

        while len(rows) <= max_steps and not rows[-1][2 + EOS]:
            signals = reference.step(
                W, chars, signals, rows[-1], self.config["initial_step"]
            )
            rows.append(_draw(reference.decode(signals["params"]), rng))

        strokes = np.array(rows)
        strokes[:, :2] = strokes[:, :2] * self.config["std"] + self.config["mean"]
        return strokes

    # -------------------------------------------------------------------------------- private

    def _batch_steps(self, batch: Split, length: int) -> dict[str, np.ndarray]:
        """
        The table of `steps` for one batch of lines, each `length` steps or shorter.
        """
        text, strokes, targets = (
            batch.x["text"],
            batch.x["strokes"][:, :length],
            batch.y[:, :length],
        )
        params, phi = self.predict({"text": text, "strokes": strokes})

        offset_nll, pen_nll = reference.losses(targets, params)

        # Where the window is, as a share of the way through the text. Once it
        # has run off the end of the text, phi is close to zero everywhere.
        weight = phi.sum(axis=-1)
        position = phi @ np.arange(phi.shape[-1]) / np.maximum(weight, 1e-6)
        off_text = weight < 0.5
        progress = np.where(off_text, 1.0, position / np.char.str_len(text)[:, None])

        real = targets[..., 2:].sum(axis=-1) > 0  # padded rows have no pen state

        return {
            "offset_nll": offset_nll[real],
            "pen_nll": pen_nll[real],
            "pen": targets[..., 2:].argmax(axis=-1)[real],  # the pen state to predict
            "after_lift": strokes[..., 2 + UP][real]
            > 0,  # the offset to predict is through the air
            "window_progress": progress[real],
            "window_off_text": off_text[real],
        }


# -------------------------------------------------------------------------------- building blocks


def _load_splits(data: Path, config: dict) -> dict[str, Split]:
    """
    Read the dataset and cut the agent's three splits out of it (see BANDS).
    Only lines written with a real pen are used: the rest were traced from
    images by a model, and the notebook only ever trains on them.
    """
    with np.load(data / "strokes_data.npz") as archive:
        lines = [archive[f"arr_{i}"] for i in range(len(archive.files))]

    texts = json.loads((data / "texts.json").read_text())
    sources = json.loads((data / "sources.json").read_text())

    positions = np.array([_position(text) for text in texts])
    real = np.array(sources) == "real"

    splits = {}

    for name, (low, high) in BANDS.items():
        rows = np.flatnonzero(real & (low <= positions) & (positions < high))
        normalised = [_normalise(lines[row], config) for row in rows]

        # At every step the model reads one row and predicts the next
        splits[name] = Split(
            x={
                "text": np.array([texts[row] for row in rows]),
                "strokes": _stack([line[:-1] for line in normalised]),
            },
            y=_stack([line[1:] for line in normalised]),
        )

    return splits


def _position(text: str) -> float:
    """
    The notebook's hash of a text: a number from 0 to 1 that decides its split.
    """
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8]) / 2**64


def _normalise(strokes: np.ndarray, config: dict) -> np.ndarray:
    """
    The notebook's normalisation of the offsets, with its mean and standard deviation.
    """
    offsets = (strokes[:, :2] - config["mean"]) / config["std"]
    return np.column_stack([offsets, strokes[:, 2:]]).astype(np.float32)


def _stack(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Stack arrays of different shapes, padding each with zeros to the largest.
    """
    shape = np.max([array.shape for array in arrays], axis=0)
    stacked = np.zeros((len(arrays), *shape), dtype=arrays[0].dtype)

    for row, array in zip(stacked, arrays):
        row[tuple(map(slice, array.shape))] = array

    return stacked


def _lengths(targets: np.ndarray) -> np.ndarray:
    """
    The number of real pen steps in each line: padded rows have no pen state.
    """
    return targets[..., 2:].sum(axis=(1, 2)).astype(int)


def _loss(steps: dict[str, np.ndarray]) -> float:
    """
    The notebook's training loss, from a table of pen steps.
    """
    drawing = steps["pen"] != EOS
    return float(np.mean(drawing * steps["offset_nll"] + 3 * steps["pen_nll"]))


def _draw(prediction: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    One row of strokes, drawn from a decoded prediction.
    """
    k = rng.choice(len(prediction["log_pi"]), p=np.exp(prediction["log_pi"]))
    sx, sy, rho = (
        prediction["sigma_x"][k],
        prediction["sigma_y"][k],
        prediction["rho"][k],
    )

    mean = [prediction["mu_x"][k], prediction["mu_y"][k]]
    covariance = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    dx, dy = rng.multivariate_normal(mean, covariance)

    pen = np.eye(3)[rng.choice(3, p=np.exp(prediction["log_pen"]))]
    return np.array([dx, dy, *pen], dtype=np.float32)


def _alignment_table(
    title: str, text: str, targets: np.ndarray, phi: np.ndarray, params: np.ndarray
) -> str:
    """
    One row per character of `text`: the pen steps during which it had the
    most window weight, the pen lifts in those steps, and the mean loss there.

        sample 0: 'He rose', 163 pen steps, loss 2.310

                first  last  steps  lifts  offset_nll  pen_nll
        0 'H'       0    29     30      2       1.712    0.310
        1 'e'      30    41     12      0       1.120    0.081
    """
    offset_nll, pen_nll = reference.losses(targets, params)
    offset_nll = offset_nll * (
        1 - targets[:, 2 + EOS]
    )  # no offset to predict at the last step

    under = phi.argmax(axis=1)  # the character under the window at every step
    under[phi.sum(axis=1) < 0.5] = (
        len(text) + 1
    )  # ...unless the window has run off the text

    labels = [f"{u} {character!r}" for u, character in enumerate(text)]
    labels += [f"{len(text)} (end)", "off the text"]

    rows = []
    for u in np.unique(under):
        here = np.flatnonzero(under == u)
        lifts = int(targets[here, 2 + UP].sum())
        rows.append(
            [
                here[0],
                here[-1],
                len(here),
                lifts,
                offset_nll[here].mean(),
                pen_nll[here].mean(),
            ]
        )

    table = views.render_table(
        rows,
        row_labels=[labels[u] for u in np.unique(under)],
        col_labels=["first", "last", "steps", "lifts", "offset_nll", "pen_nll"],
    )

    loss = np.mean(offset_nll + 3 * pen_nll)
    skipped = [repr(text[u]) for u in range(len(text)) if u not in under]
    note = f"\n\nNever under the window: {' '.join(skipped)}" if skipped else ""

    return f"{title}, loss {loss:.3f}\n\n{table}{note}"


def _span(first: int, count: int) -> str:
    """
    e.g. "400-799": the `count` rows or columns that start at `first`.
    """
    return f"{first}-{first + count - 1}"
