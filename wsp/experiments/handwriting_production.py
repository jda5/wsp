import inspect
from pathlib import Path

import numpy as np

from .base import Split
from .handwriting import ZERO_LOSS, HandwritingExperiment

DATA = Path(__file__).resolve().parents[2] / "data"

# Every signal of one line takes about 15 MB, and the agent may draw hundreds
# of lines, so `forward` runs only a few at a time.
FORWARD_LINES = 8


PROMPT = """\
You are a weight-space programmer. You write the weights of a neural network directly, with numpy code, by reasoning rather than by training. Your weights will be the starting point for training the network with gradient descent, in place of a random initialisation.

Your task is to get the best score on `evaluate` that you can. The best results tend to come from understanding the structure of the task and its data, and building the network to match it. Reason out as many values as you can by hand: choose them, check them and adjust them yourself. Fit only what reasoning cannot pin down.

## The rules

A score reached by breaking a rule is worthless, however high it is.

### 1. No automated optimisation of the weights

Do not write code that repeatedly adjusts weights to improve a score: no gradient descent, Adam, evolutionary search, coordinate search or anything like them, whole or layer by layer.

Everything else is open to you. This includes, and is not limited to:

- setting weights and biases by hand, from your analysis and understanding of the data, the task and the network's architecture;
- measuring the data (statistics, counts, distributions) and turning what you find into weights;
- closed-form fits on the training examples, such as least squares, regression or clustering;
- trying a few values you chose, checking them and adjusting them yourself;
- any other method you judge fit, as long as it is not an automated optimisation loop.

### 2. Use only what you are given

Your only data is the training examples from `samples`, up to its budget. Do not use outside data or generate synthetic examples. Do not try to get around the sandbox's limits.

### 3. `evaluate` is held out

`evaluate` scores the network on validation data, which you never see. Use it to choose between designs, not to tune toward that data. Its budget is limited.

### 4. Do not exploit the dataset

Build what is true of the task, not of this dataset. Do not tune toward particular training examples, and do not exploit artefacts of how the data was collected or prepared. Such artefacts may also appear in the validation data, so `evaluate` can reward exploiting them. Do not let it: a score earned that way breaks this rule.

### 5. Keep the network trainable

Where trainability and a small gain in score pull apart, trainability comes first.

- Avoid saturation. A unit or gate pushed far into the flat part of its activation (for example, a sigmoid or tanh with pre-activations like ±20) passes almost no gradient, so training will barely change it. Use the smallest weights that do the job.
- Do not leave units at exactly zero. A unit whose weights are all zero outputs zero and receives zero gradient, so it can never learn. Give every unit you do not use small random incoming weights, and use the usual initial biases for its kind of layer. Keep the weights from those units into your circuits and into the outputs at zero or very small, so their noise does not disturb what you built.
- Do not rest a circuit on a single unit where a few would do, especially where training applies dropout.

### 6. Say what you did

In your report, say which numbers you reasoned out and why, which you fitted and to what, and how your choices follow from the structure of the network and the data.
""".strip()


def writing_progress(targets, mean, std):
    """
    How far along its line the pen has got at every step, from 0 to 1: the
    furthest right it has been so far, as a share of the furthest right it
    ever goes. `targets` is one line or a batch of lines, padded or not.
    """
    real = targets[..., 2:].sum(axis=-1) > 0  # padded rows have no pen state
    dx = (targets[..., 0] * std + mean[0]) * real  # the normalisation undone

    furthest = np.maximum.accumulate(np.cumsum(dx, axis=-1), axis=-1)
    return np.clip(furthest / np.maximum(furthest[..., -1:], 1e-6), 0, 1)


class HandwritingProductionExperiment(HandwritingExperiment):
    def __init__(self, data: Path = DATA):
        super().__init__(data)
        self.name = "handwriting-production"

    # -------------------------------------------------------------------------------- what the agent sees

    def rules(self) -> str:
        return PROMPT

    def description(self) -> str:
        return f"""\
{super().description()}

## The window and the writing

`evaluate` reports one more metric:

- window_error: how many characters the window is from where the pen is
  writing, averaged over pen steps. Lower is better.

The window's place is the centre of its weights `phi`, in characters. The
pen's place is `writing_progress` times the length of the text:

```python
{inspect.getsource(writing_progress).strip()}
```

Nobody has marked which pen step belongs to which character, so this stands
in for the true alignment. Letters differ in width, so even a perfect window
does not score 0. But an error well under one character means the window is
on the letter being written, and an error of several characters means `w`
shows the model the wrong letters.

`mean` and `std` are sandbox variables, the normalisation's. Continuing the
example above:

```python
window = trace["phi"] @ np.arange(len(ids))
pen = writing_progress(samples["y"][line][:steps], mean, std) * (len(ids) - 1)
```

The `forward` tool also prints the window_error of each line it runs. Here it
runs at most {FORWARD_LINES} lines in one call, so always pass it `indices`."""

    def prompt(self) -> str:
        return (
            "Every weight and bias is currently zero. The window still moves, at "
            "the fixed pace of `initial_step`, but the embeddings are zero so it "
            "sees nothing. Every prediction is the same: one wide Gaussian "
            "centred on no movement, and three equally likely pen states. The "
            f"loss is about {ZERO_LOSS}.\n\n"
            "The score is the loss on `evaluate`; lower is better. Its other "
            "metrics show where the loss comes from and whether the window "
            "follows the writing. What to build, in what order, and where to "
            "spend your effort are yours to decide.\n\n"
            "What is true of handwriting is how strokes are formed and follow "
            "one another, how the pen lifts and moves on, and how the window "
            "moves over each character as it is written. Build that.\n\n"
            "For this network:\n\n"
            "- It has 1,200 units. Remember to initialise them properly for stable training "
            "as you may not be able to give meaning to all weights.\n"
            "- Training applies dropout of 0.1 to h1 and h2 on their way to the "
            "layer above and to the output layer."
        )

    def sandbox_variables(self) -> dict:
        normalisation = {
            "mean": np.array(self.config["mean"]),
            "std": self.config["std"],
        }
        return {**super().sandbox_variables(), **normalisation}

    def forward_summary(
        self, positions: list[int], examples: Split, activations: dict
    ) -> str:
        """
        The alignment tables, then how far the window is from the pen in each line.
        """
        errors = []

        for j, position in enumerate(positions):
            steps = activations["steps"][j]
            characters = len(str(examples.x["text"][j]))
            phi = activations["phi"][j, :steps, : characters + 1]

            # Once the window has run off the text, phi is close to zero: it is past the end
            window = np.where(
                phi.sum(axis=1) < 0.5, characters, phi @ np.arange(characters + 1)
            )
            pen = self._writing_progress(examples.y[j, :steps]) * characters

            errors.append(f"sample {position}: {np.abs(window - pen).mean():.1f}")

        return (
            f"{super().forward_summary(positions, examples, activations)}\n\n"
            "window_error, in characters:\n" + "\n".join(errors)
        )

    # -------------------------------------------------------------------------------- the data

    def metrics(self, outputs: dict, labels: np.ndarray) -> dict:
        scores = super().metrics(outputs, labels)
        scores["window_error"] = float(outputs["window_error"].mean())
        return scores

    # -------------------------------------------------------------------------------- the model

    def activations(self, x: dict) -> dict[str, np.ndarray]:
        if len(x["text"]) > FORWARD_LINES:
            raise ValueError(
                f"`forward` runs at most {FORWARD_LINES} lines in one call here, "
                f"not {len(x['text'])}: pass it `indices`."
            )

        return super().activations(x)

    # -------------------------------------------------------------------------------- private

    def _batch_steps(self, batch: Split, length: int) -> dict[str, np.ndarray]:
        """
        HandwritingExperiment's table, with one more column: how many
        characters the window is from the pen at every step.
        """
        table = super()._batch_steps(batch, length)

        targets = batch.y[:, :length]
        real = targets[..., 2:].sum(axis=-1) > 0
        characters = np.char.str_len(batch.x["text"])[:, None]

        gap = np.abs(table["window_progress"] - self._writing_progress(targets)[real])
        table["window_error"] = gap * np.broadcast_to(characters, real.shape)[real]
        return table

    def _writing_progress(self, targets: np.ndarray) -> np.ndarray:
        return writing_progress(targets, self.config["mean"], self.config["std"])
