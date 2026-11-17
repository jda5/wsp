# Weight-Space Programming

A harness that lets an LLM read and write the weights of a neural network directly, with no gradient descent or training loop.

Inspired by [Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming), I used it to see how well Claude can build or improve models on classic ML problems (MNIST, CartPole, IMDB, Two Spirals, Brackets and Shakespeare) solely by editing their weights.

**Read the write-up: [Experiments in Weight-Space Programming](https://jacobstrauss.bearblog.dev/experiments-in-weight-space-programming/)**

## How it works

The agent runs NumPy code inside a Docker sandbox with no internet access, and edits the weights directly. It can draw a few training examples, commit and reset snapshots, and score its weights on the validation set a limited number of times. The test set stays hidden until the end of the run, and everything is logged to `runs/`.

## Running it

You'll need Python 3.14, [Docker](https://www.docker.com/), [uv](https://docs.astral.sh/uv/) and an Anthropic API key in `.env` (`ANTHROPIC_API_KEY=...`).

```bash
uv run python -m examples.cartpole
```

Every experiment in `examples/` runs the same way. Use `--agent human` to write the weights yourself, and follow a run live from another terminal with:

```bash
uv run python -m wsp.watch --follow
```
