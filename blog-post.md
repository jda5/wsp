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

## Experiments

### Experiment 1: MNIST

The first experiment I tried was to see if an agent could generate the weights of a model trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). This is a dataset of handwritten digits commonly used for training various image processing systems, and is sometimes regarded as the "Hello world" of machine learning engineering, so I thought it might be a good place to start.

I built a simple sequential model using TensorFlow, comprising two dense layers: a hidden layer with 128 units and an output layer of 10 units. Despite how small this model is, the number of trainable parameters totals 101,770. All weights were set to 0, before running the agent.

The base LLM used in this experiment was Claude Opus 5.5 with effort to "High". The agent ran for 8 minutes, used a template matching strategy, resulting of a test score of 64.1%. For reference, a model trained with gradient descent can hit a test score of >85% in under 20 seconds. A 2025 paper calculates a network’s weights analytically from a small set of 30 reference digits and reports 63% on the MNIST test set with no training (https://arxiv.org/pdf/2505.23876) - which is what the model achieved.

This was a decent result, though in all honesty, given the fanfare of the MIT paper I expected better. MNIST is a well-known and is extensively written about, so I expect the agent to at least match gradient descent. That being said, the interesting thing was how the agent interpreted the data and reasoned how certain weights should be set. Taken from the logs:

> Pure template matching probably tops out around 70%, so I'm weighing whether more variants per digit, drawing on more of the 128 units, could get past that.

### Experiment 2: CartPole

For the second experiment I wanted to try a different architectures and applications of AI. I thought of the reinforcement learning tasks that the reserachers at MIT gave GPT-6 Astra, and decided to tackle [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/). CartPole is another classic benchmark in which an agent must keep a pole balanced upright on a cart by pushing the cart left or right along a frictionless track, earning a reward for every timestep the pole stays within a set angle.

I wrote a small policy model comprising one hidden layer of 16 units, and to cut a long story short the agent completely obliterated this task. It achieved a perfect score in just a few turns, using just a handful of weights. The agent thought for about 490 tokens, chose a lienar rule and decided on a weigthed sum of the for inputs.

> I wrote a linear state-feedback controller with the bang-bang structure of LQR control into the network.

> The small positive cart position and velocity terms bring the cart back toward the centre, since tilting the pole the right way first lets the cart drift back.

This is the text book answer. In truth you don't need deep learning to solve this task, and it can be accomplished with very basic rules. The agent knows the solution and didn't have to work anything out, but it still was interesting to see how it was able reason throught the networks architecture and interpret this linear rule as weigths.

*** Expand on this - add a quote.

### Experiment 3: IMDB

I wanted to try something harder. I thought a natural language experiment might be interesting, and decided on the [IMDB sentinment analysis](https://keras.io/api/datasets/imdb/) benchmark. This is a classification task in which a model must read a movie review and predict whether the author had positive or negative feelings about the film. 

I wrote a bag-of-words model, so that each review is turned into 10,000 inputs, one for each word in the vocabulary, which feed a single sigmoid output. 
The agent began by reading through the vocabulary, then created a tiered sentiment dictionary - words like “worst” and “waste” got a negative score, words like “excellent” and “masterpiece” a positive score, etc.

> I’m also considering a small negative weight for words like ‘horror’, ‘zombie’, ‘gore’ and ‘budget’, since low-budget horror tends to get panned.

The agent scored 86.2% on the test set. That’s 0.8 points below the same model trained using gradient decent (87.0%), and above Géron’s GRU (about 85%). It took under 3 minutes and changed 663 of the 10,001 weights. The agent also noted that a bag-of-words model can’t see negation (“not good”) or mixed reviews, so it had hit the model’s ceiling rather than its own. It’s worth noting that about a point of that score came from the agent’s memory of how IMDB reviewers write, with small weights on words like “zombie”, “budget” and “remake”, which is exactly the kind of intuition that makes WSP interesting, because encoding this information into a model.

### Experiment 4: Two Spirals

[Two Spirals](https://developers.google.com/machine-learning/crash-course/DPE/tp-il-neural-net-intro-spiral) is another classic. Two spirals wind around each other, with a bit of noise added to every point. The network has to classify which spiral a point belongs, which actually can be quite tricky for gradient decscent optimisation.

I wrote a small network made up of two inputs (x and y), two hidden layers of 64 ReLU units and a single output, totalling 4,417 weights, all set to zero. Though, this time I didn’t tell the agent what the data was. It was only told that there were points in a plane, each labelled 0 or 1, and that finding the pattern was part of the task. It drew 100 sample points, recognised the shape straight away, and correctly estimated how tightly the spiral winds.

The agent took roughly 6 minutes in total to solve the task. It evaluated once, scored 99.7%, and stopped, saying further tweaks would risk overfitting. On the test set it scored 99.9%.

Once again, the agent understood the data, spotted the pattern and designed a network around it. What stood out was how neatly the work was divided. The first layer drew a fan of lines spreading out from the centre. The second layer used those lines to mark the edges of the spirals, and the output layer then simply counted how many edges a point had crossed to decide which spiral it belonged to.

> I’m setting up the first layer so each unit is a half-plane test along one ray’s normal, with biases at zero.

### Experiment 5: Brackets

Next up was a recurrent neural network. I based the task on [Skachkova et al.’s (2018) “Closing Brackets with Recurrent Neural Networks”](https://aclanthology.org/W18-5425/). The network reads a string of brackets - such as `([]{<>})` - with the last bracket removed, and has to predict which closing bracket comes next. The paper’s trained network scores 96.6%, but struggles with longer strings.

The agent’s first attempt was to keep track of which bracket was opened first. However, while working through the problem it noticed a shortcut. Only the last bracket is missing, meaning that every kind of bracket is balanced except one. It reaslised that all it needed to do was count open minus closed brackets for each kind, and the answer would be "the one that equals 1".

It built four simple counters, one for each kind of bracket, using 16 units each. In under a minute, with a single evaluation, it scored 100% on every string length.

Much like the IMDB experiment, the agent managed to interpret the dataset and reason its way to the answer. I could make the task harder, but I thought I needed to up the ante, and since I work with LSTMs in production, I wanted to see how it handled Shakespearean passages...

### Experiment 6: Shakespeare

My final experiment was to have an agent generate weights for a character-level LSTM that ingestes Shakespearean text and predicts the next character. It’s the same dataset and task that [Géron](https://github.com/ageron/handson-ml3/blob/main/16_nlp_with_rnns_and_attention.ipynb) and [Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) use to showcase LSTMs.

I first ran an experiment where the weights were zeroed out and the agent had to write them from scratch. It reached about 34% accuracy using clever language tricks (character frequencies, common letter pairs, common word endings) and translating them into weights.

One thing I hadn’t tried yet was whether the agent could improve a model that had already been trained. So, I trained the same LSTM with gradient descent, but stopped it after just a few epochs. It scored 39.9% on the test set, which is considerably less than the 51.3% it would have reached had it been allowed to train further.

The agent started by looking for something to fix, and found nothing:

> Nothing points to a defect, so the model seems simply undertrained, and any gains will have to come from hand-coded knowledge.

After 36 minutes, it nudged the test accuracy from 39.9% to 40.5%, which is roughly half an epoch of extra training. I then ran the same task with Fable 5.2. It came up with a far more sophisticated strategy, compressing a large chunk of the network by hand to make room for a tree of common word beginnings. Though this still ended up in almost exactly the same place.

This was the first task the agent couldn’t crack, and I think it shows the limits of weight-space programming. What an undertrained model is missing isn’t a bug you can fix. It’s thousands of small patterns spread across all of its weights. As the agent put it, “knowledge of whole words is the main gap, spread across many of them.” Each hand-written rule was worth a tiny fraction of a percent, and there’s only room for a few dozen of them.
