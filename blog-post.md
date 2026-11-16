# Experiments in Weight-Space Programming

Two weeks ago, MIT CSAIL published [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). In the post, the researchers showed that frontier LLMs are surprisingly good at natively reading and writing neural network weights. They ran a few experiments, and showed that that GPT-6 Astra was capable of:

- identifying a network's task just by looking at the raw weights;
- generating weights for a policy network from scratch, capable of playing simple games;
- and improving Qwen3.5-0.8B's mathematical reasoning capabilities by directly editing its weigths.

All without using gradient descent or training loops! They coined this *Weight-Space Programming* (WSP).

These results briefly took the internet by storm as neural network are considered to be a black box. They process data across a huge vector space - billions of floating point numbers connected to each other across layers artificial neurons - making them entirely incomprehensible, even to to the developer who created them. We have no idea how AI systems make their decision. Yet, if an LLM was able to reason through the tensor soup, then we might just be able to lift the lid on the black box. 

I had to try WSP for myself. My goal was to run a series of experiments on a handful of classic machine learning problems to determine how well frontier LLMs can optimise or build models, soley by editing the model weights.

This is part one of a two part series on WSP. This first part is a bit more exploratory than the second, whereas the second is a bit more formal.

## The Harness 

The first thing I did was build an agent harness that enables an LLM to natively read and write neural network weights. The code for which be found here: https://github.com/jda5/wsp. The harness broadly implements the following:

1. An agent is given access to a Python interpreter running inside a Docker container. The container has no internet access and only has Python 3.13 with Numpy installed. This was done to prevent cheating, such as peeking at the validation or test sets, using machine learning packages such as TensorFlow or Pytorch to run iterative training algorithms* or going on the internet to look up the answer.
2. For each experiment the agent is given part (or sometimes all) of the training data, the network and a description of the task.
3. The agent was given a `commit` and `revert` tool, which allowed it save a snapshot of the weigths as a kind of checkpoint that it can revert back to, so that it can explore different ideas.
4. The agent also was given an `evaluate` tool. This runs network with the current weights on the validation data and returns evaluation metrics relavant to the task, e.g. accuracy, loss, confusion matricies, etc.
5. The agent had a limited budget. It had to complete it's task in a certian number of turns (usually between 50 - 200 depending on the task), and it could only call the `evaluate` tool a limited number of times.
6. Everything is logged. Agent thinking, tool use, python exectutions, etc.
7. I also initially gave the agent access to a budgeted `forward` tool, which runs the model with the current weights on some training examples. This prints the output layer and stors the layer's activations in the sandbox, to see how weights responded to real inputs. The agent didn't use this as much as hoped, opting instead to create it's own version of the `forward` tool.

The model was prompted to maximise the evaluation score by writing the network's weights with numpy code. I gave it certain rules that it should follow, things like: not to use gradient decent, not to build it's own dataset, not to exploit the dataset, etc. I also had another LLM check over the logs of each run for evidence of rule-breaking, and no run that broke the rules I'd set.

I used Claude Opus 5.5 and Claude Fable 5.2 for all runs. The harness could support any LLM, but I have free Anthropic API credits through their start-up program which is why I went with Claude.

* The agent probably could have ran gradient descent with just Numpy and a `for` loop. In very early versions, loops were banned, but I felt that this was too restrictive, and agents were good at following the rules.

## Experiment 1: MNIST

The first experiment I tried was to see if an agent could generate the weights of a model trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). This is a dataset of handwritten digits commonly used for training various image processing systems, and is sometimes regarded as the "Hello world" of machine learning engineering, so I thought it might be a good place to start.

I built a simple sequential model using TensorFlow, comprising two dense layers: a hidden layer with 128 units and an output layer of 10 units. Despite how small this model is, the number of trainable parameters totals 101,770. All weights were set to 0, before running the agent.

The base LLM used in this experiment was Claude Opus 5.5 with effort to "High". The agent ran for 8 minutes, used a template matching strategy, resulting of a test score of 64.1%. For reference, a model trained with gradient descent can hit a test score of >85% in under 20 seconds. A 2025 paper calculates a network’s weights analytically from a small set of 30 reference digits and reports 63% on the MNIST test set with no training (https://arxiv.org/pdf/2505.23876) - which is what the model achieved.

This was a decent result, though in all honesty, given the fanfare of the MIT paper I expected better. MNIST is a well-known and is extensively written about, so I expect the agent to at least match gradient descent. That being said, the interesting thing was how the agent interpreted the data and reasoned how certain weights should be set. Taken from the logs:

> Pure template matching probably tops out around 70%, so I'm weighing whether more variants per digit, drawing on more of the 128 units, could get past that.

### Experiment 2: CartPole

For the second experiment I wanted to try a different architectures and applications of AI. I thought of the reinforcement learning tasks that the reserachers at MIT gave GPT-6 Astra, and decided to tackle [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/). CartPole is another classic benchmark in which an agent must keep a pole balanced upright on a cart by pushing the cart left or right along a frictionless track, earning a reward for every timestep the pole stays within a set angle.

I wrote a small policy model comprising one hidden layer of 16 units, and to cut a long story short the agent completely obliterated this task. It achieved a perfect score in just a few turns, using just a handful of weights. The agent thought for about 490 tokens, chose a lienar rule and decided on a weigthed sum of the for inputs. 

This is the text book answer. In truth you don't need deep learning to solve this task, and it can be accomplished with very basic rules. The agent knows the solution and didn't have to work anything out, but it still was interesting to see how it was able reason throught the networks architecture and interpret this linear rule as weigths.

*** Expand on this - add a quote.

### Experiment 3: IMDB

I wanted to try something harder. I thought a natural language experiment might be interesting, and decided on the [IMDB sentinment analysis](https://keras.io/api/datasets/imdb/) benchmark. This is a classic classification task in which a model must read a movie review and predict whether the author had positive or negative feelings about the film. 

I wrote up a bag-of-words model, and ....[describe what the agent did]..... and the results went much the same way that CartPole did. The agent scored 86.2% on test. That’s 0.8 points below the same model trained normally (87.0%), and above Géron’s GRU (about 85%). It took under 3 minutes and changed 663 of the 10,001 weights.












### Experiment 4: Two Spirals

[Two Spirals](https://developers.google.com/machine-learning/crash-course/DPE/tp-il-neural-net-intro-spiral) is another classic. This data set is a noisy spiral, which is difficult for gradient descent optimization.

This is the first experiment where the agent had to invent something rather than recall it. It's fascinating to see how the agent is able to make predictions based on it's own understanding of the problem, understand the data and then write a working network for it.

### Experiment 5: Brackets

My next experiment was an RNN. I built a task based on Skachkova et al.'s (2018) "Closing Brackets with Recurrent Neural Networks" paper. The agent reads a balanced string of four kinds of bracket (()[]{}<>) with its last bracket removed and has to predict which closing bracket completes it. The trained version of this network described Skachkova et al. (2018) in scores 96.6% but fails on strings longer than about 8 brackets.

The agent beats both the paper and our own training, especially on long strings (100% against 93.6%). It got there with a construction that’s arguably more robust than gradient descent’s. It found a flaw in a published benchmark. Spotting that the task is really a counting problem is the most interesting result of the day. It’s also worth noting for your research: the paper’s last-bracket task measures less than it appears to.

We could make this harder, but I thought that we really need to up the ante here if we are to properly stretch the model. Since I work with LSTMs in production, let's see how it handles Shakespearean passages!

### Experiment 6: Shakespeare

The architecture is a character-based LSTM; the same dataset and task that Geron and Karapathy use to showcase LSTMs. I first ran an experiment where the model weights were zeroed out, and the agent had to write the weights from scratch. It achieved around 35% accuracy by using clever language tricks (character frequencies, bigram frequencies, common word endings), and translating those to weights - which mind-mindboggling, though I observed similar strategems in the the IMDB task.

One experiment I also wanted to try not yet tried, is whether the agent make improvements to a pre-trained model? For this next experiment I will train the same character-level LSTM architecture - using gradient decent - but curtail the number of epochs. Here is the setup: