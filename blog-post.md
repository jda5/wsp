# Experiments in Weight-Space Programming

Two weeks ago, MIT CSAIL published [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). In the post, the researchers demonstrated that frontier LLMs are surprisingly good at reading and writing neural network weights directly. Across a handful of experiments, they showed that GPT-6 Astra was able to:

- identify what task a network was built for, just by looking at its raw weights;
- write the weights of a game-playing policy network from scratch;
- improve Qwen3.5-0.8B’s mathematical reasoning by editing its weights directly.

All without gradient descent or a training loop! They coined this **Weight-Space Programming** (WSP).

These results briefly took the internet by storm, as neural network are a black box. They process data across a huge vector space - billions of floating point numbers connected to each other across layers of artificial neurons - making them entirely incomprehensible, even to the developer who created them. We have very little idea how AI systems reach their decisions. Yet, if an LLM was able to reason through the tensor soup, we might be able to lift the lid on the black box. 

I had to try WSP for myself. My goal was to run a series of experiments on a handful of classic machine learning problems to see how well a frontier LLM can build or improve a model, solely by editing its weights.

This is the first of a two-part series on WSP. Part two coming soon...

## The Harness 

The first thing I did was build a harness that lets an LLM read and write neural network weights directly. The code is on [GitHub](https://github.com/jda5/wsp). Here’s how it works:

1. An agent is given access to a Python interpreter running inside a Docker container. The container has no internet access and only has Python and NumPy installed. This was done to prevent obvious ways of cheating, such as peeking at the validation or test sets, training the model with a library like TensorFlow or PyTorch*, or looking up the answer online.
2. For each experiment, the agent gets a description of the task, a description of the network and the network’s weights as NumPy arrays that it can edit however it likes.
3. The agent can draw a limited number of training examples (usually a dozen or so) with a `samples` tool, so it can see what the data looks like.
4. A `commit` tool saves a snapshot of the weights, and a `reset` tool goes back to the last one. This lets the agent explore ideas without losing its best version.
5. An `evaluate` tool runs the network on the validation set and returns scores relevant for the task (such as accuracy, loss, a confusion matrix, etc.). The test set stays hidden, and is only scored at the very start and the very end of a run.
6. Each run is limited to a set number of turns (50 to 100, depending on the task), and `evaluate` can only be called a limited number of times. This stops the agent from simply trying values until the score goes up.
7. Everything is logged. Agent’s thinking, every tool call and every piece of Python it ran.

I also gave the agent a `forward` tool, which runs the network on some examples, prints the output and stores the layer's activations in the container, so that the agent can observe how the weights respond to inputs. In practice the agent rarely used it, preferring to write its own version in NumPy.

The model was prompted to maximise the evaluation score by writing the network's weights in NumPy. I also gave it certain rules that it should follow, things like: not to use gradient descent, not to build its own dataset, not to exploit the dataset, etc. I also had another LLM check over the logs of each run for evidence of rule-breaking, and no run broke the rules I'd set.

* To be fair, the agent could have written gradient descent with NumPy and a `for` loop. Very early versions of the harness banned loops altogether, but that turned out to be too restrictive, and the agents were good at following the rules.

## Experiments

All of the models were built in [TensorFlow](https://www.tensorflow.org/). The LLM for all experiments was [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5) with effort set to “High” - though for the Shakespeare experiment I also tried [Claude Fable 5.2](https://www.anthropic.com/claude-fable-and-mythos-5-2). The harness works with any LLM, but I get free Anthropic API credits through their start-up program which is why I went with Claude.

### Experiment 1: MNIST

The first experiment I tried was to see if an agent could generate the weights of a model trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). MNIST is a dataset of 28×28 pixel images of handwritten digits. It’s sometimes regarded as the “Hello world” of machine learning, so I thought it would be a good place to start.

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b1/MNIST_dataset_example.png?utm_source=en.wikipedia.org&utm_campaign=imageinfo&utm_content=thumbnail_unscaled-url" alt="Sample images from MNIST test dataset. Courtesy of Wikipedia." height="300">

I built a simple sequential model comprising two dense layers: a hidden layer with 128 units and an output layer of 10 units. The number of trainable parameters was 101,770, and all weights were set to 0 before running the agent.

The agent ran for 8 minutes, employing a template matching strategy to complete the task. It drew each digit out of lines, arcs and ellipses. Then it made 12 versions of each by slanting it, stretching it and, for some digits, changing its shape - a 1 with and without the hook, a 7 with and without the cross-bar, etc.

> The idea is to draw stroke-based digit templates from my own knowledge...

The agent achieved a test score of 64.1%. For reference, a model trained with gradient descent can hit a test score of >85% in under 20 seconds. A [2025 paper](https://arxiv.org/pdf/2505.23876) calculated a network’s weights from 30 reference digits, with no training, and reported 63%. Which is almost exactly what the agent achieved.

This was a decent result, though in all honesty, given the fanfare of the MIT paper, I expected better. MNIST is so well known and so extensively written about that I expected the agent to at least match gradient descent. That being said, watching it draw digits in code, check them against real examples and redraw the ones that didn’t fit was fascinating. The agent also identified it was approaching a limit:

> Pure template matching probably tops out around 70%, so I'm weighing whether more variants per digit, drawing on more of the 128 units, could get past that.

### Experiment 2: CartPole

For the second experiment I wanted to try a different architecture and application of AI. I thought of the reinforcement learning tasks that the researchers at MIT gave GPT-6 Astra, and decided to tackle [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/). CartPole is another classic benchmark in which an agent must keep a pole balanced upright on a cart by pushing the cart left or right along a frictionless track, earning a reward for every timestep the pole stays within a set angle.

<img src="https://github.com/pythonlessons/CartPole_reinforcement_learning/raw/master/IMAGES/CartPole_test.gif" alt="GIF of CartPole in action. Courtesy of Rokas Liuberskis" width="300" height="300">

I wrote a small policy model comprising one hidden layer of 16 units, and to cut a long story short the agent completely obliterated this task. It achieved a perfect score in just a few turns, using just a handful of weights. The agent thought for about 490 tokens, chose a linear rule and decided on a weighted sum of the four inputs.

> I wrote a linear state-feedback controller with the bang-bang structure of LQR control into the network.

This is the textbook answer. In truth you don't need deep learning to solve this task, and it can be accomplished with very basic rules. The agent knows the solution and didn't have to work anything out, but it still was interesting to see how it was able to reason through the network's architecture and implement the linear rule as weights. 

### Experiment 3: IMDB

Next, I wanted to try something harder. I thought a natural language experiment might be interesting, and decided on the [IMDB sentiment analysis](https://keras.io/api/datasets/imdb/) benchmark. This is a classification task in which a model must read a movie review and predict whether the author had positive or negative feelings about the film. 

I wrote a bag-of-words model. Each review is turned into 10,000 inputs, one for each word in the vocabulary. These feed a single sigmoid output, so the model has one weight per word, plus a bias.

The agent began by reading through the vocabulary, then created a tiered sentiment dictionary - words like “worst” and “waste” got a negative score, words like “excellent” and “masterpiece” a positive score, etc. It also drew on what it knew about how people write reviews:

> I’m also considering a small negative weight for words like ‘horror’, ‘zombie’, ‘gore’ and ‘budget’, since low-budget horror tends to get panned.

The agent scored 86.2% on the test set, just 0.8 points below the same model trained with gradient descent (87.0%). It took under 3 minutes and changed 663 of the 10,001 weights. It also pointed out that a bag-of-words model can’t see negation (“not good”) or mixed reviews, so it had hit the limits of the model rather than its own.

The weights came from the agent’s sense of how IMDB reviewers write. I think this is exactly what makes WSP interesting. Gradient descent can learn the same pattern, but it ends up implicitly buried somewhere in the weights. Here, the agent reasons explicitly, and we can follow why each weight is there.xw

### Experiment 4: Two Spirals

[Two Spirals](https://developers.google.com/machine-learning/crash-course/DPE/tp-il-neural-net-intro-spiral) is another classic. Two spirals wind around each other, with a bit of noise added to every point. The network has to classify which spiral a point belongs to, which can actually be quite tricky for gradient descent optimisation.

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*m2oF-ZzbXhVFq6ss9R4S_Q.png" alt="Depiction of the Two Spirals problem. Courtesy of Rakesh Malviya" width="300" height="300">

I wrote a small network made up of two inputs (x and y), two hidden layers of 64 ReLU units and a single output, totalling 4,417 weights. This time I didn’t tell the agent what the data was. It was only told that there were points in a plane, each labelled 0 or 1, and that finding the pattern was part of the task. It drew 100 sample points, recognised the shape straight away, and correctly estimated how tightly the spiral winds.

The agent took roughly 6 minutes in total to solve the task. It evaluated once, scored 99.7%, and stopped, saying further tweaks would risk overfitting. On the test set it scored 99.9%.

Once again, the agent understood the data, spotted the pattern and designed a network around it. What stood out was how neatly the work was divided. The first layer drew a fan of lines spreading out from the centre. The second layer used those lines to mark the edges of the spirals, and the output layer then simply counted how many edges a point had crossed to decide which spiral it belonged to.

> I’m setting up the first layer so each unit is a half-plane test along one ray’s normal, with biases at zero.

### Experiment 5: Brackets

Next up was a recurrent neural network. I based the task on [Skachkova et al.’s (2018) “Closing Brackets with Recurrent Neural Networks”](https://aclanthology.org/W18-5425/). The network reads a string of brackets - such as `([]{<>})` - with the last bracket removed, and has to predict which closing bracket comes next. The paper’s trained network scores 96.6%, but struggles with longer strings.

The agent’s first attempt was to keep track of which bracket was opened first. However, while working through the problem it noticed a shortcut. Only the last bracket is missing, meaning that every kind of bracket is balanced except one. It realised that all it needed to do was count open minus closed brackets for each kind, and the answer would be "the one that equals 1".

It built four simple counters, one for each kind of bracket, using 16 units each. In under a minute, with a single evaluation, it scored 100% on every string length.

Much like the IMDB experiment, the agent managed to interpret the dataset and reason its way to the answer. I could make the task harder, but I thought I needed to up the ante, and since I work with LSTMs in production, I wanted to see how it handled Shakespearean passages...

### Experiment 6: Shakespeare

My final experiment was to have an agent generate weights for a character-level LSTM that ingests Shakespearean text and predicts the next character. It’s the same dataset and task that [Géron](https://github.com/ageron/handson-ml3/blob/main/16_nlp_with_rnns_and_attention.ipynb) and [Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) use to showcase LSTMs.

I first ran an experiment where the weights were zeroed out and the agent had to write them from scratch. It reached about 34% accuracy using clever language tricks (character frequencies, common letter pairs, common word endings) and translating them into weights.

One thing I hadn’t tried yet was whether the agent could improve a model that had already been trained. So, I trained the same LSTM with gradient descent, but stopped it after just a few epochs. It scored 39.9% on the test set, which is considerably less than the 51.3% it would have reached had it been allowed to train further.

The agent started by looking for something to fix, and found nothing:

> Nothing points to a defect, so the model seems simply undertrained, and any gains will have to come from hand-coded knowledge.

After 36 minutes, it nudged the test accuracy from 39.9% to 40.5%, which is roughly half an epoch of extra training. I then ran the same task with Fable 5.2. It came up with a far more sophisticated strategy, compressing a large chunk of the network by hand to make room for a tree of common word beginnings. Though this still ended up in almost exactly the same place.

This was the first task the agent couldn’t accomplish. It's clear that an undertrained model isn’t something that WSP can address. As the agent put it:

> knowledge of whole words is the main gap, spread across many of them.

## Conclusion

WSP is an interesting approach to AI development. Frontier LLMs can _explicitly_ reason through patterns in the data, and realise these patterns as weights in a structured way. Gradient descent also captures these patterns, but does so _implicitly_, making it much harder for us to interpret the network's output. For small models, a network where you can point at every unit and say what it does may be worth more than a few points of accuracy.

It was also shown in [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) that an LLM can address [shortcomings in a model post-training using linear programming](https://nathan.rs/posts/weight-space-lp/). I observed similar capabilities in the Shakespeare experiment, but ultimately it's clear that WSP isn't an effective replacement for training loops. For the moment, gradient descent is cheaper, faster and in many cases, results in more accurate models. The agent seems better at deciding what a network should look like than at choosing its exact numbers, which is exactly the part gradient descent is good at.

But perhaps replacing gradient descent is the wrong goal. In part two, I’ll explore whether an agent’s hand-written weights make a better initialiser than random ones.
