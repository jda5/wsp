"""
Handwriting for production: the agent writes the starting weights of the
handwriting network, and all that counts is the model that training makes of
them.

The same data, model, tools and metrics as HandwritingExperiment, which asks
a research question: can reasoning alone write good weights? Here the
question is practical, and three things change:

    the rules       The agent may fit to the training lines in any way that
                    helps. Only the validation lines stay out of sight.
    the aim         The weights start the notebook's training, so they must
                    hold on lines the agent has not seen. `evaluate` is the
                    measure, and the agent is asked to build what is true of
                    handwriting, not of its training lines. Direction and
                    strategy are left to the agent.
    one metric      `window_error`: how far the window is from where the pen
                    is writing. It shows whether the window follows the
                    writing, which the loss alone does not.

The splits are HandwritingExperiment's: bands of the lines the notebook
trains on. The notebook's own validation and test lines stay unseen by the
agent, and they are what makes the comparison of trained models honest.
"""

import inspect
from pathlib import Path

import numpy as np

from .base import Split
from .handwriting import ZERO_LOSS, HandwritingExperiment

DATA = Path(__file__).resolve().parents[2] / "data"

# Every signal of one line takes about 15 MB, and the agent may draw hundreds
# of lines, so `forward` runs only a few at a time.
FORWARD_LINES = 8

RULES = """\
You are a weight-space programmer. You write a neural network's weights
directly with numpy code. Your weights will be the starting point for training
the network with gradient descent.

## The rules

This is a production run. Your weights are judged on `evaluate`, and nothing is
judged on how you made them. So there are few rules.

### 1. The training data is yours to use

`samples` gives you training examples, as many as its budget allows. Measure
them, fit to them and compute your own loss on them, in any way that helps:
statistics, least squares, regression, clustering, a search over a constant.

What you bring that a script could not is the reasoning: which structure to
build, and how to wire it. Fit the numbers that structure needs.

### 2. `evaluate` is held out

`evaluate` scores the validation data, which you never see. It tells you
whether what you built and fitted on the training data holds on new data, so
choose between designs with it. Weights that capture how the task works will
hold there; weights that capture the particular examples you fitted will not.
Its budget is limited.

### 3. Leave the training to the training

Do not train the network with gradient descent in the sandbox, whole or layer
by layer. That happens next, on a GPU, far faster than numpy can do it here.
Fitting the constants of a structure you designed, as rule 1 allows, is not
training.

### 4. Use only what you are given

No outside data, and no attempts to get around the sandbox's limits.

### 5. Say what you did

In your report, say which numbers you reasoned out and which you fitted, and
to what.
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
        return RULES

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
            "Your weights will be the starting point for training this network "
            "with gradient descent, in place of a random initialisation. They "
            "are judged on `evaluate`.\n\n"
            "Your task is to produce the best evaluation you can: write "
            "weights that bring the loss on `evaluate` as low as you can. In "
            "earlier runs, the best results tended to come from understanding "
            "the structure of the data and building the network to match it. "
            "The other metrics show where the loss comes from, and whether the "
            "window follows the writing. What to build, in what order, and "
            "where to spend your effort are yours to decide.\n\n"
            "Use the training lines freely, as the rules say. But build what "
            "is true of handwriting, not of those lines: how strokes are formed "
            "and follow one another, how the pen lifts and moves on, how the "
            "window moves over each character as it is written. Weights of "
            "that kind hold on lines you have not seen, and `evaluate` is where "
            "you check that they do. A gain on your samples that does not show "
            "on `evaluate` is overfitting: take it out.\n\n"
            "Write the weights so the network stays trainable. Where that and a "
            "small gain in loss pull apart, trainability comes first:\n\n"
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
            "disturb what you built.\n"
            "- Training applies dropout of 0.1 to h1 and h2 on their way to "
            "the layer above and to the output layer, so do not rest a circuit "
            "on a single unit of those layers where a few would do.\n\n"
            "Keep improving the weights until `evaluate` shows no further "
            "improvement."
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
