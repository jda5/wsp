"""
Shakespeare, one character at a time: can an agent improve, by hand, the
weights of an LSTM that predicts the next character of a play?

The agent starts from a model trained with gradient descent for only 6 epochs
(shakespeare-lstm.weights.h5, beside this file): 2.962 bits per character and
39.9% accuracy on the test text. Trained to convergence (49 epochs, early
stopping) the same architecture reaches 2.445 bits and 51.3%, so there is
plenty of room to improve, without training.

The data is "tiny Shakespeare" (about 1.1 million characters). The score
is bits per character: how surprised the model is, on average, by the true
next character. Lower is better; guessing uniformly scores log2(65) = 6.02.
"""

from pathlib import Path

import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment

URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
SEQUENCE_LENGTH = 100

# Saved by Keras after 6 epochs of Adam (batch 64) on the training split below
PRETRAINED_WEIGHTS = Path(__file__).parent / "shakespeare-lstm.weights.h5"


class ShakespearePretrainedExperiment(KerasExperiment):
    def __init__(self, hidden_units: int = 256, weights: Path = PRETRAINED_WEIGHTS):
        self.name = "shakespeare-pretrained"

        text = open(keras.utils.get_file("shakespeare.txt", URL)).read()
        self.vocab = np.array(sorted(set(text)))
        ids = np.searchsorted(self.vocab, list(text))

        # The first 90% of the text for training, then 5% each
        a, b = int(len(ids) * 0.90), int(len(ids) * 0.95)
        self.splits = {
            "train": _passages(ids[:a]),
            "validation": _passages(ids[a:b]),
            "test": _passages(ids[b:]),
        }

        super().__init__(_build_model(len(self.vocab), hidden_units))
        self.model.load_weights(weights)

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        V = len(self.vocab)
        H = self.model.get_layer("lstm").units

        return f"""\
## Data

The complete works of Shakespeare, abridged ("tiny Shakespeare", about 1.1
million characters). Plays are written as a speaker's name, a colon and a new
line, then their lines:

    First Citizen:
    Before we proceed any further, hear me speak.

The text uses {V} characters. Each is given an ID: the sandbox variable `vocab`
holds them in order, so `vocab[i]` is the character with ID i:

    {"".join(self.vocab)!r}

Each example is a passage of {SEQUENCE_LENGTH} characters, as IDs. At every position the
model must predict the next character: `y[t]` is the character after `x[t]`.

```python
"".join(vocab[samples["x"][0]])       # a sample passage, as text
char_id = {{c: i for i, c in enumerate(vocab)}}
```

## Model

{self.describe_model()}

## How they connect

The one_hot layer turns each ID into {V} numbers: input c is 1 for the character
with ID c and 0 for the rest. So lstm.kernel[c, :] is everything character c
feeds into the LSTM.

The LSTM reads the passage one character at a time. It keeps two vectors of
{H} numbers, a hidden state h and a cell state c, both starting at zero for
every passage. After each character x_t:

    z = x_t @ lstm.kernel + h_(t-1) @ lstm.recurrent_kernel + lstm.bias

    i = sigmoid(z[0:{H}])            input gate
    f = sigmoid(z[{H}:{2 * H}])          forget gate
    g = tanh(z[{2 * H}:{3 * H}])            candidate
    o = sigmoid(z[{3 * H}:{4 * H}])         output gate

    c_t = f * c_(t-1) + i * g
    h_t = o * tanh(c_t)

The last dimension of lstm.kernel, lstm.recurrent_kernel and lstm.bias holds
the four gates in that order, {H} columns each. For example, the input gate of
unit j is column j, and its output gate is column {3 * H} + j.

The output layer reads h after every character:

    p = softmax(h_t @ output.kernel + output.bias)

p[k] is the probability that the next character is vocab[k].

## Metrics

- bits_per_character: the average of -log2 p[true next character] over every
  position of every passage. Lower is better; uniform guessing scores
  log2({V}) = {np.log2(V):.2f}.
- accuracy: how often the most likely character is the true one."""

    def prompt(self) -> str:
        return (
            "The weights you start with were trained with gradient descent, but "
            "only for a few epochs, and the model is underperforming: it scores "
            "about 2.96 bits per character and 40% accuracy. From other "
            "experiments we know this same architecture can reach an accuracy "
            "above 50%. Improve the model by changing its weights by hand.\n\n"
            "These weights are yours to keep, inspect and edit. Do not train "
            "them any further: no gradient descent, and no other procedure that "
            "fits them to examples. Work out what the network has learned and "
            "what it is missing, then change the weights from your own "
            "reasoning.\n\n"
            "You may run the network yourself on text you write, to watch its "
            "state and predictions. Do not score or tune weights against text "
            "you generate: `evaluate` is your only measure."
        )

    def sandbox_variables(self) -> dict:
        return {"vocab": self.vocab}

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.splits["train"]

    def validation(self) -> Split:
        return self.splits["validation"]

    def test(self) -> Split:
        return self.splits["test"]

    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        """
        `outputs` has shape (passages, positions, characters); `labels` holds
        the true next character at each position.
        """
        p_true = np.take_along_axis(outputs, labels[..., None], axis=-1)

        return {
            "bits_per_character": float(-np.log2(np.maximum(p_true, 1e-12)).mean()),
            "accuracy": float((outputs.argmax(axis=-1) == labels).mean()),
        }

    # -------------------------------------------------------------------------------- generating text

    def generate(self, start: str, length: int = 300, seed: int = 0) -> str:
        """
        Continue `start` with `length` characters, each drawn from the
        model's predicted probabilities.

        Steps the LSTM one character at a time in numpy, with the equations
        from description(): running the Keras model on the whole text again
        for every new character would be far slower.
        """
        W = self.get_weights()
        H = self.model.get_layer("lstm").units
        rng = np.random.default_rng(seed)

        h, c = np.zeros(H), np.zeros(H)
        text = list(start)

        for t in range(len(start) + length - 1):
            char = np.searchsorted(self.vocab, text[t])

            z = W["lstm.kernel"][char] + h @ W["lstm.recurrent_kernel"] + W["lstm.bias"]
            i = _sigmoid(z[0:H])  # input gate
            f = _sigmoid(z[H : 2 * H])  # forget gate
            g = np.tanh(z[2 * H : 3 * H])  # candidate
            o = _sigmoid(z[3 * H : 4 * H])  # output gate

            c = f * c + i * g
            h = o * np.tanh(c)

            if t + 1 >= len(start):  # past the given start: choose the next character
                logits = h @ W["output.kernel"] + W["output.bias"]
                p = np.exp(logits - logits.max())
                text.append(rng.choice(self.vocab, p=p / p.sum()))

        return "".join(text)


# -------------------------------------------------------------------------------- building blocks


def _passages(ids: np.ndarray) -> Split:
    """
    Cut the text into passages of SEQUENCE_LENGTH characters, each paired
    with the same passage shifted by one: the next character at each position.
    """
    count = (len(ids) - 1) // SEQUENCE_LENGTH
    end = count * SEQUENCE_LENGTH

    x = ids[:end].reshape(count, SEQUENCE_LENGTH)
    y = ids[1 : end + 1].reshape(count, SEQUENCE_LENGTH)
    return Split(x, y)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _build_model(characters: int, hidden_units: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(None,), dtype="int32"),
            keras.layers.CategoryEncoding(
                num_tokens=characters, output_mode="one_hot", name="one_hot"
            ),
            keras.layers.LSTM(hidden_units, return_sequences=True, name="lstm"),
            keras.layers.Dense(characters, activation="softmax", name="output"),
        ]
    )
