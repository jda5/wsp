"""
IMDB movie reviews with a bag-of-words model: can an agent write the weights
of a sentiment classifier from its own knowledge of language?

The model ignores word order. It sees only which of the 10,000 most common
words a review contains, and adds up one weight per word:

    p(positive) = sigmoid(sum of the weights of the words present + bias)

So writing its weights means writing a sentiment dictionary.
"""

from dataclasses import dataclass

import keras
import numpy as np

from .base import Split
from .keras_experiment import KerasExperiment
from .metrics import binary_classification_metrics

VOCABULARY_SIZE = 10_000

# keras.datasets.imdb encodes each review as word IDs. The first four IDs are
# markers; the word with frequency rank r (1 = most common) has ID r + 3.
MARKERS = ["<pad>", "<start>", "<unknown>", "<unused>"]
FIRST_WORD_ID = len(MARKERS)


class IMDBExperiment(KerasExperiment):
    def __init__(self, validation_size: int = 5_000):
        self.name = "imdb-zeros"
        self.data = _load_dataset(validation_size)
        self.vocabulary = _load_vocabulary()

        super().__init__(_build_model())
        self.zero_weights()

    # -------------------------------------------------------------------------------- what the agent sees

    def description(self) -> str:
        return f"""\
## Data

IMDB movie reviews, each labelled 0 (negative) or 1 (positive). Half the
reviews are positive. The text is lowercase with punctuation removed; "br"
is left over from HTML line breaks.

Each review is a sequence of word IDs, padded with zeros at the end. There
are {VOCABULARY_SIZE:,} IDs:

- 0 `<pad>`: padding after the review ends
- 1 `<start>`: the start of every review
- 2 `<unknown>`: any word outside the {VOCABULARY_SIZE:,} most common
- 3 `<unused>`: never appears
- {FIRST_WORD_ID} onwards: words, most common first ({FIRST_WORD_ID} is "the")

The sandbox variable `vocab` is an array of the {VOCABULARY_SIZE:,} words, so
`vocab[i]` is the word with ID i:

```python
word_id = {{word: i for i, word in enumerate(vocab)}}
word_id["great"]                         # the ID of a word
review = samples["x"][0]                 # a sample, once you have drawn some
" ".join(vocab[review[review != 0]])     # its text, in order
```

## Model

{self.describe_model()}

## How they connect

The bag_of_words layer turns a review into {VOCABULARY_SIZE:,} numbers: input i
is 1 if word ID i appears anywhere in the review and 0 if not. How many times
a word appears, and where, makes no difference.

So W["output.kernel"][i, 0] is the weight of the word vocab[i], and the model
computes

    p(positive) = sigmoid(sum of the weights of the words present + output.bias)

The prediction is positive when p is above 0.5, that is when the sum plus the
bias is above 0.

ID 1 (`<start>`) is in every review, so its weight acts as a second bias. ID 0
(`<pad>`) is in almost every review, because only the longest one has no
padding.

## Metrics

Accuracy, accuracy per class (0 = negative, 1 = positive), and a confusion
matrix where row = true label and column = predicted label."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model gives every "
            "review a probability of 0.5 and calls them all negative. Write "
            "weights that classify the sentiment of movie reviews as accurately "
            "as you can.\n\n"
            "Choose each word's weight from your own understanding of what the "
            "word means in a review. Do not count how often words appear in "
            "positive and negative samples and turn those counts into weights: "
            "that is fitting the weights to examples."
        )

    def sandbox_variables(self) -> dict:
        return {"vocab": self.vocabulary}

    # -------------------------------------------------------------------------------- the data

    def train(self) -> Split:
        return self.data.train

    def validation(self) -> Split:
        return self.data.validation

    def test(self) -> Split:
        return self.data.test

    def metrics(self, outputs: np.ndarray, labels: np.ndarray) -> dict:
        return binary_classification_metrics(outputs, labels)


# -------------------------------------------------------------------------------- building blocks


@dataclass(frozen=True)
class IMDBData:
    train: Split
    validation: Split
    test: Split


def _load_dataset(validation_size: int = 5_000) -> IMDBData:
    """
    The reviews as rows of word IDs, padded with zeros, with the last
    ``validation_size`` training reviews held out for validation.
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
        num_words=VOCABULARY_SIZE
    )

    # Every review has a different length, so pad them into one array.
    X_train = keras.utils.pad_sequences(X_train, padding="post")
    X_test = keras.utils.pad_sequences(X_test, padding="post")

    # np.asarray tells linters these are arrays: load_data() is untyped.
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return IMDBData(
        train=Split(X_train[:-validation_size], y_train[:-validation_size]),
        validation=Split(X_train[-validation_size:], y_train[-validation_size:]),
        test=Split(X_test, y_test),
    )


def _load_vocabulary() -> np.ndarray:
    """
    The word for each ID: an array of VOCABULARY_SIZE strings.
    """
    vocabulary = MARKERS + [""] * (VOCABULARY_SIZE - FIRST_WORD_ID)

    for word, rank in keras.datasets.imdb.get_word_index().items():
        word_id = rank + FIRST_WORD_ID - 1
        if word_id < VOCABULARY_SIZE:
            vocabulary[word_id] = word

    return np.array(vocabulary)


def _build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(None,), dtype="int32"),
            keras.layers.CategoryEncoding(
                num_tokens=VOCABULARY_SIZE, output_mode="multi_hot", name="bag_of_words"
            ),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )
