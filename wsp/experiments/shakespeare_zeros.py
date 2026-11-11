"""
Shakespeare from zeros: can an agent write, from nothing, the weights of an
LSTM that predicts the next character of a play?

The same data, model, description and metrics as ShakespeareExperiment, but
every weight starts at zero instead of at the 6-epoch weights. The agent keeps
going until `evaluate` stops improving.

The point is to use the agent as an initialiser: its final weights (saved by
examples/shakespeare_zeros.py as final.weights.h5) are a starting point for
gradient descent, to compare with Keras's usual random initialisation.
"""

import numpy as np

from .shakespeare_pretrained import ShakespearePretrainedExperiment


class ShakespeareZerosExperiment(ShakespearePretrainedExperiment):
    def __init__(self, hidden_units: int = 256):
        super().__init__(hidden_units)  # loads the 6-epoch weights...
        self.zero_weights()  # ...which are then all set to zero

        self.name = "shakespeare-zeros"

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero, so the model gives every "
            f"character the same probability and scores {np.log2(len(self.vocab)):.2f} "
            "bits per character. Write weights that predict the next character "
            "as well as you can.\n\n"
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
            "learn. Give any unit you do not use small random weights (for "
            "example normal with standard deviation 0.05). That is not fitting: "
            "it only gives training something to work with.\n\n"
            "Your own knowledge of English and of Shakespeare is yours to use: "
            "how common letters and words are, what tends to follow what, how "
            "the plays are laid out. What you may not do is count characters or "
            "words in the samples and turn the counts into weights: that is "
            "fitting the weights to examples.\n\n"
            "You may run the network yourself on text you write, to watch its "
            "state and predictions. Do not score or tune weights against text "
            "you generate: `evaluate` is your only measure."
        )
