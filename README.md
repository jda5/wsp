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

## Experiments 1: MNIST From Scratch

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a dataset of handwritten digits that is commonly used for training various image processing systems. It is sometimes regarded as the "[Hello, world](https://en.wikipedia.org/wiki/Hello,_world)" of machine learning engineering, so I thought it might be a good place to start.

I built a simple sequential model using TensorFlow, comprising two dense layers: a hidden layer with 128 units and an output layer of 10 units. Despite how small this model is, the number of trainable parameters totals 101,770. All weights were set to 0, before running the agent.

The base LLM used in this experiment was Claude Opus 5.5 with effort to "High". The agent ran for 8 minutes, used a template matching strategy, resulting of a test score of 64.1%.