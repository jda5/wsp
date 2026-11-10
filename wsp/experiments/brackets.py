"""
Closing brackets with a recurrent network, following Skachkova, Trost,
Kusmirek and Klakow (2018), "Closing Brackets with Recurrent Neural
Networks", BlackboxNLP. https://aclanthology.org/W18-5425

The task is the paper's accuracy task (section 4.2): take a balanced string
of four kinds of bracket, remove its last bracket, and ask the network which
closing bracket comes last. The data, its sizes and the network (an Elman
RNN: one sigmoid hidden layer of 64 units) are as in the paper. The paper's
Elman RNN, trained, scores 96.6% on this task (its table 2, p = 6/16, n = 4).

Strings are made one bracket at a time (the paper's section 2.1): the first
bracket opens; then, with probability P_OPEN, a new bracket of a random kind
opens; otherwise the most recently opened bracket closes, or, if none is
open, the string ends. Strings average 8 brackets, with a long tail.
"""

import warnings

import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment
from .metrics import classification_metrics

KINDS = 4  # kinds of bracket, the paper's n
P_OPEN = 6 / 16  # the paper's p
OPENING = "([{<"
CLOSING = ")]}>"
SYMBOLS = OPENING + CLOSING  # input i is SYMBOLS[i]

# The paper's dataset sizes. It has no validation set; ours is the size of its test set.
TRAIN_SIZE = 131_072
VALIDATION_SIZE = 10_000
TEST_SIZE = 10_000

# Accuracy is also reported for three groups of full string lengths. The
# paper finds trained Elman RNNs perfect up to about 8 brackets, GRUs up to 20.
LENGTH_GROUPS = [(2, 8), (10, 20), (22, None)]

# Keras warns, every time the model runs, that the rnn layer "destroys" the
# padding mask. It does use the mask to skip padding; it just doesn't pass it
# on, and the layers after it don't need it.
warnings.filterwarnings("ignore", message="Layer 'rnn' .* does not support masking")


class BracketsExperiment(KerasExperiment):
    def __init__(self, hidden_units: int = 64):
        self.name = "brackets-zeros"

        # Each split is drawn with its own seed, so the splits differ.
        self.splits = {
            "train": _make_tasks(TRAIN_SIZE, seed=0),
            "validation": _make_tasks(VALIDATION_SIZE, seed=1),
            "test": _make_tasks(TEST_SIZE, seed=2),
        }

        super().__init__(_build_model(hidden_units))
        self.zero_weights()

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        units = self.model.get_layer("rnn").units

        return f"""\
## Data

Balanced strings of four kinds of bracket: () [] {{}} <>. A string is balanced
when every bracket is closed, by a closing bracket of the same kind, after
everything opened inside it has been closed: "([]<>)" is balanced; "([)]" is not.

Every string is made one bracket at a time. The first bracket opens. Then,
with probability {P_OPEN} (= 6/16), a new bracket of a random kind opens (each
kind equally likely). Otherwise, the most recently opened bracket that is still
open closes; if no bracket is open, the string ends instead. Strings average
8 brackets, but some are much longer: about 8% are 22 or more, and a few are
over 100.

## The task

Each example is a balanced string with its last bracket removed. The network
must say which closing bracket comes last: the label is 0 for ")", 1 for "]",
2 for "}}" and 3 for ">". Each label is equally likely.

Each input is an array of shape (steps, 8), one row per bracket, one-hot over
the symbols "{SYMBOLS}": input i is 1 for the symbol "{SYMBOLS}"[i]. Inputs
shorter than the longest in a batch are padded at the start with rows of
zeros; the network skips those steps.

To read a sample as text:

```python
x = samples["x"][0]
"".join("{SYMBOLS}"[i] for i in x[x.sum(axis=1) > 0].argmax(axis=1))
```

## Model

{self.describe_model()}

## How they connect

The rnn layer reads the string one bracket at a time, keeping a state h of
{units} numbers. h starts at zero, and after each bracket x_t it becomes

    h_t = sigmoid(x_t @ rnn.kernel + h_(t-1) @ rnn.recurrent_kernel + rnn.bias)

so rnn.kernel[i, j] is what symbol "{SYMBOLS}"[i] adds to unit j, and
rnn.recurrent_kernel[i, j] is the weight from unit i at the previous step to
unit j now. Every unit of h is between 0 and 1. The rnn layer outputs h after
every step; last_state keeps only the final one, which the output layer reads:

    p = softmax(h_final @ output.kernel + output.bias)

Output k is the probability that the last bracket is "{CLOSING}"[k]; the
prediction is the largest. In `forward_result`, the rnn activations have shape
(samples, steps, units): the state after every bracket.

## Metrics

Accuracy, accuracy per closing bracket, a confusion matrix where row = true
bracket and column = predicted bracket, and accuracy for each group of full
string lengths (counting the missing last bracket): 2-8, 10-20, and 22 or more."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model gives each "
            "closing bracket a probability of 1/4 and always predicts \")\". "
            "Write weights that predict the missing last bracket as accurately "
            "as you can, for strings of any length.\n\n"
            "You may run the network yourself on strings you write, to watch "
            "how its state changes bracket by bracket. Do not generate a "
            "collection of labelled strings to score or tune weights against: "
            "`evaluate` is your only measure."
        )

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.splits["train"]

    def validation(self) -> Split:
        return self.splits["validation"]

    def test(self) -> Split:
        return self.splits["test"]

    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        return classification_metrics(outputs, labels, num_classes=KINDS)

    def score(self, split: Split) -> dict:
        """
        The usual metrics, plus accuracy for each group of full string lengths.
        """
        outputs = self.predict(split.x)
        scores = self.metrics(outputs, split.y)

        lengths = _lengths(split.x) + 1  # the full string had one more bracket
        correct = outputs.argmax(axis=1) == split.y

        for shortest, longest in LENGTH_GROUPS:
            in_group = lengths >= shortest
            if longest is not None:
                in_group &= lengths <= longest

            if in_group.any():
                name = f"{shortest}_to_{longest}" if longest else f"{shortest}_or_more"
                scores[f"accuracy_length_{name}"] = float(correct[in_group].mean())

        return scores


# -------------------------------------------------------------------------------- building blocks


def _make_tasks(count: int, seed: int) -> Split:
    """
    `count` balanced strings, each with its last bracket removed. x is
    one-hot and padded at the start; y is the kind of the removed bracket.
    """
    rng = np.random.default_rng(seed)

    inputs, labels = [], []
    for _ in range(count):
        string = _balanced_string(rng)
        inputs.append(string[:-1])
        labels.append(string[-1] - KINDS)  # closing symbols are KINDS..2*KINDS-1

    return Split(_encode(inputs), np.array(labels))


def _balanced_string(rng: np.random.Generator) -> list[int]:
    """
    One string from the paper's generator, as symbol indices into SYMBOLS.
    """
    first = int(rng.integers(KINDS))
    string, still_open = [first], [first]

    while True:
        if rng.random() < P_OPEN:
            kind = int(rng.integers(KINDS))
            string.append(kind)
            still_open.append(kind)
        elif still_open:
            string.append(KINDS + still_open.pop())
        else:
            return string


def _encode(strings: list[list[int]]) -> np.ndarray:
    """
    One-hot, shape (strings, longest, 8), padded with zero rows at the start.
    Stored as uint8 to keep the 131,072 training strings small.
    """
    longest = max(len(s) for s in strings)
    x = np.zeros((len(strings), longest, len(SYMBOLS)), dtype=np.uint8)

    for i, string in enumerate(strings):
        start = longest - len(string)
        x[i, np.arange(start, longest), string] = 1

    return x


def _lengths(x: np.ndarray) -> np.ndarray:
    """
    How many brackets each encoded input has, ignoring padding.
    """
    return (x.sum(axis=2) > 0).sum(axis=1)


def _build_model(hidden_units: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(None, len(SYMBOLS))),
            keras.layers.Masking(mask_value=0.0, name="padding"),
            keras.layers.SimpleRNN(
                hidden_units, activation="sigmoid", return_sequences=True, name="rnn"
            ),
            keras.layers.Lambda(lambda states: states[:, -1], name="last_state"),
            keras.layers.Dense(KINDS, activation="softmax", name="output"),
        ]
    )
