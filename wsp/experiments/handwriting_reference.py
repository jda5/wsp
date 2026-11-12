"""
The handwriting model in plain numpy, one pen step at a time.

This is the same computation as the Keras model, written so that every signal
can be read. It needs nothing but numpy.

    W          the weights, by name
    ids        the text's character ids, then one 0: the position past its end
    strokes    one row per pen step: dx, dy, pen_down, pen_up, eos
"""

import numpy as np

# -------------------------------------------------------------------------------- the model


def encode(text, vocab):
    """
    The character ids of `text`, then one 0 for the position past its end.
    """
    ids = {character: i for i, character in enumerate(vocab)}
    return np.array([ids.get(character, 1) for character in text] + [0])


def forward(W, ids, strokes, initial_step):
    """
    Run the model over the pen steps in `strokes`. Returns every signal of
    `step`, each as an array with one row per pen step.
    """
    chars = W["embedding.embeddings"][ids]  # one row of numbers per character

    signals, trace = start(W), []
    for x in strokes:
        signals = step(W, chars, signals, x, initial_step)
        trace.append(signals)

    return {name: np.array([s[name] for s in trace], dtype=np.float32) for name in signals}


def start(W):
    """
    The signals before the first pen step: every state is zero.
    """
    units = W["mdn.kernel"].shape[0] // 3

    signals = {name: np.zeros(units) for name in ["h1", "c1", "h2", "c2", "h3", "c3"]}
    signals["kappa"] = np.zeros(W["decoder_1.window.bias"].size // 3)
    signals["w"] = np.zeros(W["embedding.embeddings"].shape[1])
    return signals


def step(W, chars, previous, x, initial_step):
    """
    One pen step. `x` is the pen's last move (a row of strokes), `previous`
    the signals of the step before, `chars` the text's character embeddings.
    """
    # Layer 1 reads the pen and the last window vector...
    h1, c1 = lstm(W, "decoder_1.lstm", np.concatenate([x, previous["w"]]), previous["h1"], previous["c1"])

    # ...and moves the window: K Gaussians over the character positions. Each
    # has a weight, a width (sigma) and a centre (kappa) that only moves forward.
    logits, raw_sigma, raw_step = np.split(h1 @ W["decoder_1.window.kernel"] + W["decoder_1.window.bias"], 3)
    weight = softmax(logits)
    sigma = softplus(raw_sigma) + 0.1
    kappa = previous["kappa"] + softplus(raw_step + np.log(np.expm1(initial_step)))

    # phi[u] is how much of the window is on character u. w blends the characters by phi.
    position = np.arange(len(chars))
    phi = weight @ np.exp(-0.5 * ((position - kappa[:, None]) / sigma[:, None]) ** 2)
    phi = phi / (phi.sum() + 1e-6)
    w = phi @ chars

    # Layers 2 and 3 read the pen, the layer below and the window vector.
    h2, c2 = lstm(W, "decoder_2", np.concatenate([x, h1, w]), previous["h2"], previous["c2"])
    h3, c3 = lstm(W, "decoder_3", np.concatenate([x, h2, w]), previous["h3"], previous["c3"])

    # The output layer reads all three layers.
    params = np.concatenate([h1, h2, h3]) @ W["mdn.kernel"] + W["mdn.bias"]

    return {
        "h1": h1,
        "c1": c1,
        "h2": h2,
        "c2": c2,
        "h3": h3,
        "c3": c3,
        "weight": weight,
        "sigma": sigma,
        "kappa": kappa,
        "phi": phi,
        "w": w,
        "params": params,
    }


def lstm(W, name, x, h, c):
    """
    One step of the LSTM layer `name`. Its kernels and bias hold the four
    gates side by side: input, forget, candidate, output.
    """
    z = x @ W[f"{name}.kernel"] + h @ W[f"{name}.recurrent_kernel"] + W[f"{name}.bias"]
    i, f, g, o = np.split(z, 4)

    c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
    h = sigmoid(o) * np.tanh(c)
    return h, c


# -------------------------------------------------------------------------------- the outputs and the loss


def decode(params):
    """
    The model's outputs as the prediction they describe: a mixture of 2-D
    Gaussians for the next offset (dx, dy), and the next pen state.
    """
    mixtures = (params.shape[-1] - 3) // 6
    logit_pi, mu_x, mu_y, raw_sigma_x, raw_sigma_y, raw_rho, pen_logits = np.split(
        params, np.arange(1, 7) * mixtures, axis=-1
    )

    return {
        "log_pi": log_softmax(logit_pi),  # log of each Gaussian's share of the mixture
        "mu_x": mu_x,
        "mu_y": mu_y,
        "sigma_x": softplus(raw_sigma_x) + 1e-2,
        "sigma_y": softplus(raw_sigma_y) + 1e-2,
        "rho": 0.95 * np.tanh(raw_rho),  # correlation of dx and dy
        "log_pen": log_softmax(pen_logits),  # log probability of pen_down, pen_up, eos
    }


def losses(targets, params):
    """
    The two parts of the loss at every pen step. `targets` holds the true
    next row of strokes at each step, `params` the model's outputs.

        offset_nll   -log density of the true next offset under the mixture
        pen_nll      -log probability of the true next pen state

    The training loss is the mean of offset_nll + 3 * pen_nll over the real
    pen steps, leaving offset_nll out at the last one, where the target is eos.
    """
    m = decode(params)
    dx, dy, pen = targets[..., 0:1], targets[..., 1:2], targets[..., 2:]

    zx = (dx - m["mu_x"]) / m["sigma_x"]
    zy = (dy - m["mu_y"]) / m["sigma_y"]
    q = 1 - m["rho"] ** 2

    log_density = (
        -(zx**2 + zy**2 - 2 * m["rho"] * zx * zy) / (2 * q)
        - np.log(m["sigma_x"])
        - np.log(m["sigma_y"])
        - 0.5 * np.log(q)
        - np.log(2 * np.pi)
    )

    offset_nll = -logsumexp(m["log_pi"] + log_density)
    pen_nll = -(pen * m["log_pen"]).sum(axis=-1)
    return offset_nll, pen_nll


# -------------------------------------------------------------------------------- functions


def softplus(z):
    return np.logaddexp(0, z)


def sigmoid(z):
    return np.exp(-softplus(-z))


def logsumexp(z):
    """
    log(sum(exp(z))) over the last axis, without overflow.
    """
    top = z.max(axis=-1, keepdims=True)
    return (top + np.log(np.exp(z - top).sum(axis=-1, keepdims=True)))[..., 0]


def log_softmax(z):
    return z - logsumexp(z)[..., None]


def softmax(z):
    return np.exp(log_softmax(z))
