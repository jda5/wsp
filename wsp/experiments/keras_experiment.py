"""
The model half of an Experiment, implemented once for any Keras 3 model.
"""

import keras
import numpy as np

from .base import Experiment, Weights


class KerasExperiment(Experiment):
    """
    Implements get_weights, set_weights, predict and activations for
    `model`. Subclasses add the data, the description and the prompt.
    """

    def __init__(self, model: keras.Model, batch_size: int = 1024):
        self.model = model
        self.batch_size = batch_size

        self._variables = self.name_variables()

        self._layers_model = keras.Model(
            inputs=model.inputs,
            outputs={layer.name: layer.output for layer in model.layers},
        )

    # -------------------------------------------------------------------------------- model

    def name_variables(self) -> dict[str, keras.Variable]:
        """
        Every trainable variable, keyed "layer.variable", e.g. "dense.kernel".

        A layer built from several layers of its own can hold two variables
        of the same name. Override this to name such a model's variables.
        """
        variables = [
            (f"{layer.name}.{variable.name}", variable)
            for layer in self.model.layers
            for variable in layer.trainable_weights
        ]
        named = dict(variables)

        if len(named) < len(variables):
            raise ValueError(
                "Two variables share a name. Override name_variables() to tell them apart."
            )

        return named

    def get_weights(self) -> Weights:
        return {
            name: keras.ops.convert_to_numpy(variable)
            for name, variable in self._variables.items()
        }

    def set_weights(self, weights: Weights) -> None:
        for name, variable in self._variables.items():
            variable.assign(weights[name])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x, batch_size=self.batch_size, verbose=0)

    def activations(self, x: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self._layers_model.predict(x, batch_size=self.batch_size, verbose=0)
        return {name: np.asarray(value) for name, value in outputs.items()}

    # -------------------------------------------------------------------------------- helpers for subclasses

    def describe_model(self) -> str:
        """
        The architecture, layer by layer. Use it inside `description()`.
        """
        lines = [
            f"Keras {type(self.model).__name__} model.",
            f"Input shape: {tuple(self.model.inputs[0].shape)} (first axis is the batch).",
            "",
            "Layers, in order:",
        ]

        for layer in self.model.layers:
            lines.append(f"- {_describe_layer(layer)}")

        lines += [
            "",
            "Dense layers compute y = activation(x @ kernel + bias),",
            "so kernel[i, j] is the weight from input i to unit j.",
        ]

        return "\n".join(lines)

    def zero_weights(self) -> None:
        """
        Set every trainable weight to zero.
        """
        self.set_weights(
            {name: np.zeros_like(w) for name, w in self.get_weights().items()}
        )


# -------------------------------------------------------------------------------- private


def _describe_layer(layer: keras.layers.Layer) -> str:
    """
    e.g. "dense (Dense): output (None, 128), activation relu,
        weights dense.kernel (784, 128), dense.bias (128,)"
    """
    parts = [f"output {tuple(layer.output.shape)}"]  # type: ignore

    activation = getattr(layer, "activation", None)
    if activation is not None:
        parts.append(f"activation {getattr(activation, '__name__', activation)}")

    if layer.trainable_weights:
        tensors = ", ".join(
            f"{layer.name}.{variable.name} {tuple(variable.shape)}"
            for variable in layer.trainable_weights
        )
        parts.append(f"weights {tensors}")
    else:
        parts.append("no weights")

    return f"{layer.name} ({type(layer).__name__}): " + ", ".join(parts)
