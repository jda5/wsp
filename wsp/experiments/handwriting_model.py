"""
The handwriting synthesis model of the notebook `04-model-improved`: Graves,
"Generating Sequences With Recurrent Neural Networks" (2013), with the
Gaussian window of Battenberg et al. (2020).

The three layer classes are copied from the notebook unchanged, and
`build_model` builds its layers in the notebook's order. Keras saves weights
by each layer's class and position, so weights saved from this model load
into the notebook's with `model.load_weights`.

The notebook explains the design. For the computation, step by step in plain
numpy, see handwriting_reference.py.
"""

import numpy as np
import tensorflow as tf

DROPOUT = 0.1


class GaussianWindow(tf.keras.layers.Layer):
    """
    Soft, monotonic attention over the text. Graves (2013), parameterised as
    GMMv2b from Battenberg et al. (2020).
    """

    def __init__(self, num_components: int, initial_step: float, **kwargs):
        super().__init__(**kwargs)
        self.step_offset = np.log(np.expm1(initial_step))  # softplus(step_offset) == initial_step
        self.params = tf.keras.layers.Dense(3 * num_components)

    def call(self, query, kappa, values, text_mask):
        logits, raw_sigma, raw_step = tf.split(self.params(query), 3, axis=-1)  # each (B, K)
        weight = tf.nn.softmax(logits)

        # Floor sigma - keeps the bump from vanishing between characters
        sigma = tf.nn.softplus(raw_sigma) + 0.1
        kappa = kappa + tf.nn.softplus(raw_step + self.step_offset)

        u = tf.range(tf.shape(values)[1], dtype=tf.float32)                   # (U,)
        z = (u - kappa[..., None]) / sigma[..., None]                          # (B, K, U)
        phi = tf.reduce_sum(weight[..., None] * tf.exp(-0.5 * tf.square(z)), axis=1) * text_mask
        phi = phi / (tf.reduce_sum(phi, axis=-1, keepdims=True) + 1e-6)       # (B, U), sums to 1

        context = tf.einsum("bu,bud->bd", phi, values)                        # (B, D)
        return context, kappa, phi


class WindowedLSTMCell(tf.keras.layers.Layer):
    """
    One step of decoder layer 1: an LSTM that drives the window. Outputs [h, w, phi].
    """

    def __init__(self, units, value_dim, text_length, num_components, initial_step, **kwargs):
        super().__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTMCell(units)
        self.window = GaussianWindow(num_components, initial_step)
        self.state_size = [
            units, units,                       # h, c
            num_components,                     # kappa
            value_dim,                          # previous window vector w
            tf.TensorShape([None, value_dim]),  # character embeddings (passed through)
            tf.TensorShape([None]),             # text mask (passed through)
        ]
        self.output_size = units + value_dim + text_length

    def call(self, inputs, states, training=None):
        h, c, kappa, w, values, text_mask = states
        output, (h, c) = self.lstm(tf.concat([inputs, w], axis=-1), [h, c])
        w, kappa, phi = self.window(output, kappa, values, text_mask)
        return tf.concat([output, w, phi], axis=-1), [h, c, kappa, w, values, text_mask]


class WindowedLSTM(tf.keras.layers.Layer):
    """
    Decoder layer 1 over a whole sequence. Returns the sequences h, w and phi.
    """

    def __init__(self, units, value_dim, text_length, num_components, initial_step, **kwargs):
        super().__init__(**kwargs)
        self.splits = [units, value_dim, text_length]
        self.cell = WindowedLSTMCell(units, value_dim, text_length, num_components, initial_step)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)

    def initial_state(self, values, char_ids):
        batch = tf.shape(values)[0]
        zeros = [tf.zeros([batch, size]) for size in self.cell.state_size[:4]]

        # The window sees the characters and one position past the last one, which is
        # how it can tell that the text has ended
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(char_ids, 0), tf.int32), axis=-1, keepdims=True)
        text_mask = tf.cast(tf.range(tf.shape(char_ids)[1])[None] <= lengths, tf.float32)
        return [*zeros, values, text_mask]

    def call(self, strokes, values, char_ids):
        outputs = self.rnn(strokes, initial_state=self.initial_state(values, char_ids))
        return tf.split(outputs, self.splits, axis=-1)


def build_model(config: dict) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    The notebook's two models, built from the config.json one of its runs saved.

        model       text and strokes -> the 123 output parameters at every pen step
        alignment   text and strokes -> the window weights at every pen step

    Both share every layer, so they share every weight.
    """
    units, embed_dim = config["units"], config["embed_dim"]
    text_length = config["text_length"]

    # The saved vocabulary starts with "" (padding) and "[UNK]", which the layer adds itself
    text_vectorisation = tf.keras.layers.TextVectorization(
        standardize=None,
        split="character",
        output_sequence_length=text_length,
        vocabulary=config["vocabulary"][2:],
    )

    concatenate = tf.keras.layers.concatenate

    text = tf.keras.Input(shape=[], dtype=tf.string)
    strokes = tf.keras.Input(shape=(None, 5))

    char_ids = text_vectorisation(text)
    chars = tf.keras.layers.Embedding(len(config["vocabulary"]), embed_dim, name="embedding")(char_ids)

    h1, w, phi = WindowedLSTM(
        units, embed_dim, text_length, config["window_components"], config["initial_step"], name="decoder_1"
    )(strokes, chars, char_ids)
    h1 = tf.keras.layers.Dropout(DROPOUT)(h1)
    h2 = tf.keras.layers.LSTM(units, return_sequences=True, name="decoder_2")(concatenate([strokes, h1, w]))
    h2 = tf.keras.layers.Dropout(DROPOUT)(h2)
    h3 = tf.keras.layers.LSTM(units, return_sequences=True, name="decoder_3")(concatenate([strokes, h2, w]))
    params = tf.keras.layers.Dense(6 * config["mixtures"] + 3, name="mdn")(concatenate([h1, h2, h3]))

    model = tf.keras.Model([text, strokes], params)
    alignment = tf.keras.Model([text, strokes], phi)

    return model, alignment
