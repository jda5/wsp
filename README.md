# Weight-Space Programming

The Wikipedia defines [weight-space programming](https://en.wikipedia.org/wiki/Weight-space_programming) (WSP) as:

> ...an approach in machine learning in which a general-purpose large language model (LLM) reads, interprets and directly writes the weights of another artificial neural network, without using gradient descent or a training loop.

Drawing inspiration from [Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming), this project has two aims:

1. Create an agent harness that enables an LLMs to natively read and write neural network weights.
2. Run experiments on a series of classic ML problems to determine how well frontier LLMs can optimise or build models, soley by editing the model weights.

## Prerequisits

- Python
- [Docker](https://www.docker.com/) to sandbox the agent.
- (Optionally) [uv](https://docs.astral.sh/uv/) for package management.
