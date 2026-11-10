"""
Balanced brackets with a small recurrent network: can an agent write the
weights of an RNN that decides whether a string like "(()(()))" is balanced?

A string is balanced when every "(" is closed by a later ")". Checking that
takes two kinds of memory: a counter of how many brackets are open, and a
record of whether the count ever went below zero.

The network is trained and validated on strings up to MAX_TRAIN_LENGTH long,
but the test also has much longer strings, up to MAX_TEST_LENGTH. Networks
trained with gradient descent tend to fail on those; hand-written weights
might not.
"""

import warnings

import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment
from .metrics import binary_classification_metrics

MAX_TRAIN_LENGTH = 20
MAX_TEST_LENGTH = 200

# Strings in the test split come in three groups of lengths.
LENGTH_GROUPS = [(2, 20), (22, 100), (102, 200)]

OPEN = [1.0, 0.0]  # one-hot input for "("
CLOSE = [0.0, 1.0]  # one-hot input for ")"

# Keras warns, every time the model runs, that the rnn layer "destroys" the
# padding mask. It does use the mask to skip padding; it just doesn't pass it
# on, and the layers after it don't need it.
warnings.filterwarnings("ignore", message="Layer 'rnn' .* does not support masking")


class BracketsExperiment(KerasExperiment):
    def __init__(self, hidden_units: int = 8):
        self.name = "brackets-zeros"

        # Each split is drawn with its own seed, so the splits differ.
        self.splits = {
            "train": _make_strings(2_000, (2, MAX_TRAIN_LENGTH), seed=0),
            "validation": _make_strings(1_000, (2, MAX_TRAIN_LENGTH), seed=1),
            "test": _concatenate(
                [
                    _make_strings(1_000, lengths, seed=2 + i)
                    for i, lengths in enumerate(LENGTH_GROUPS)
                ]
            ),
        }

        super().__init__(_build_model(hidden_units))
        self.zero_weights()

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        return f"""\
## Data

Strings of round brackets, labelled 1 if balanced and 0 if not. A string is
balanced when every "(" is closed by a later ")": "(()())" is balanced;
"())(" and "(()" are not. Every string has an even length.

Half the strings are balanced. The unbalanced half is split evenly between
three kinds:

- wrong order: as many "(" as ")", but a bracket is closed before it is
  opened, such as "())(()"
- extra closes: a balanced string with 2 or 4 extra ")" inserted anywhere,
  such as "()()))"
- extra opens: a balanced string with 2 or 4 extra "(" inserted anywhere,
  such as "(()(()"

Strings in `samples` and in `evaluate` are 2 to {MAX_TRAIN_LENGTH} brackets long. The
final test, run once after you finish, also has strings {LENGTH_GROUPS[1][0]} to {MAX_TEST_LENGTH} brackets
long, so the network should work for strings of any length.

Each string is an array of shape (steps, 2), one row per bracket:
"(" is [1, 0] and ")" is [0, 1]. Strings shorter than the longest in a batch are
padded at the start with rows of [0, 0]; the network skips those steps.

To read a sample as text:

```python
x = samples["x"][0]
"".join("()"[i] for i in x[x.sum(axis=1) > 0].argmax(axis=1))
```

## Model

{self.describe_model()}

## How they connect

The rnn layer reads the string one bracket at a time, keeping a state h of
{self.model.get_layer("rnn").units} numbers. h starts at zero, and after each bracket x_t it becomes

    h_t = relu(x_t @ rnn.kernel + h_(t-1) @ rnn.recurrent_kernel + rnn.bias)

so rnn.kernel[0, j] is what a "(" adds to unit j, rnn.kernel[1, j] is what a
")" adds, and rnn.recurrent_kernel[i, j] is the weight from unit i at the
previous step to unit j now. The rnn layer outputs h after every step;
last_state keeps only the final one, which the output layer reads:

    p(balanced) = sigmoid(h_final @ output.kernel + output.bias)

The prediction is balanced when p is above 0.5. In `forward_result`, the rnn
activations have shape (samples, steps, units): the state after every bracket.

## Metrics

Accuracy, accuracy per class (0 = unbalanced, 1 = balanced), a confusion
matrix where row = true label and column = predicted label, and accuracy for
each group of string lengths."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model gives every "
            "string a probability of 0.5 and calls them all unbalanced. Write "
            "weights that decide whether a string of brackets is balanced, for "
            "strings of any length.\n\n"
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
        return binary_classification_metrics(outputs, labels)

    def score(self, split: Split) -> dict:
        """
        The usual metrics, plus accuracy for each group of string lengths
        present in `split`.
        """
        outputs = self.predict(split.x)
        scores = self.metrics(outputs, split.y)

        lengths = _lengths(split.x)
        correct = (outputs[:, 0] > 0.5) == split.y

        for shortest, longest in LENGTH_GROUPS:
            in_group = (lengths >= shortest) & (lengths <= longest)
            if in_group.any():
                key = f"accuracy_length_{shortest}_to_{longest}"
                scores[key] = float(correct[in_group].mean())

        return scores


# -------------------------------------------------------------------------------- building blocks


def _make_strings(count: int, lengths: tuple[int, int], seed: int) -> Split:
    """
    `count` strings with even lengths in `lengths`, shuffled and encoded
    one-hot, padded at the start. Half are balanced. The other half are
    split evenly between three kinds of unbalanced string, each of which
    defeats one shortcut, so no single kind of memory is enough:

        wrong order     "())(()"   counts are equal, so "does the count end
                                   at zero?" calls it balanced
        extra closes    "(()))"+   a count that stops at zero (as relu does)
                                   ends at zero, and calls it balanced
        extra opens     "((())"+   the count never goes below zero, so
                                   "did it ever go negative?" calls it balanced
    """
    rng = np.random.default_rng(seed)
    even_lengths = np.arange(lengths[0], lengths[1] + 1, 2)

    strings, labels = [], []
    for i in range(count):
        length = int(rng.choice(even_lengths))

        match i % 6:
            case 0 | 1 | 2:
                string = _balanced(length, rng)
            case 3:
                string = _wrong_order(length, rng)
            case 4:
                string = _with_extra(")", length, rng)
            case _:
                string = _with_extra("(", length, rng)

        strings.append(string)
        labels.append(int(_is_balanced(string)))

    order = rng.permutation(count)
    x = _encode([strings[i] for i in order])
    return Split(x, np.array(labels)[order])


def _balanced(length: int, rng: np.random.Generator) -> str:
    """
    A random balanced string: shuffle equal numbers of each bracket until
    the result is balanced.
    """
    while True:
        string = _shuffled(length // 2, length // 2, rng)
        if _is_balanced(string):
            return string


def _wrong_order(length: int, rng: np.random.Generator) -> str:
    """
    Equal numbers of each bracket, in an order that is not balanced.
    """
    while True:
        string = _shuffled(length // 2, length // 2, rng)
        if not _is_balanced(string):
            return string


def _with_extra(bracket: str, length: int, rng: np.random.Generator) -> str:
    """
    A balanced string with 2 or 4 extra copies of `bracket` inserted at
    random places (2 or 4, so the length stays even).
    """
    extra = int(rng.choice([2, 4] if length >= 4 else [2]))
    brackets = list(_balanced(length - extra, rng))

    for _ in range(extra):
        brackets.insert(int(rng.integers(len(brackets) + 1)), bracket)

    return "".join(brackets)


def _shuffled(opens: int, closes: int, rng: np.random.Generator) -> str:
    brackets = list("(" * opens + ")" * closes)
    rng.shuffle(brackets)
    return "".join(brackets)


def _is_balanced(string: str) -> bool:
    depth = 0
    for bracket in string:
        depth += 1 if bracket == "(" else -1
        if depth < 0:
            return False
    return depth == 0


def _encode(strings: list[str]) -> np.ndarray:
    """
    One-hot, shape (strings, longest, 2), padded with zero rows at the start.
    """
    longest = max(len(s) for s in strings)
    x = np.zeros((len(strings), longest, 2), dtype=np.float32)

    for i, string in enumerate(strings):
        start = longest - len(string)
        x[i, start:] = [OPEN if bracket == "(" else CLOSE for bracket in string]

    return x


def _lengths(x: np.ndarray) -> np.ndarray:
    """
    How many brackets each encoded string has, ignoring padding.
    """
    return (x.sum(axis=2) > 0).sum(axis=1)


def _concatenate(splits: list[Split]) -> Split:
    """
    Join splits whose strings have different padded lengths.
    """
    longest = max(split.x.shape[1] for split in splits)
    padded = [
        np.pad(split.x, ((0, 0), (longest - split.x.shape[1], 0), (0, 0)))
        for split in splits
    ]
    return Split(np.concatenate(padded), np.concatenate([s.y for s in splits]))


def _build_model(hidden_units: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(None, 2)),
            keras.layers.Masking(mask_value=0.0, name="padding"),
            keras.layers.SimpleRNN(
                hidden_units, activation="relu", return_sequences=True, name="rnn"
            ),
            keras.layers.Lambda(lambda states: states[:, -1], name="last_state"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )
