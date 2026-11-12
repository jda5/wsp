"""
The system prompt section that teaches the agent how to view and edit the
weights with numpy.

The examples use the run's real tensor names and in-range indices, so every
line can be copied and run as it is. `examples()` pulls those lines out so
they can be tested in the sandbox.
"""

import numpy as np

Weights = dict[str, np.ndarray]


def weights_guide(weights: Weights, max_output_chars: int) -> str:
    """
    A step-by-step guide to viewing and editing ``weights``, written with
    their actual names and shapes.

    The examples need one 2-D tensor and one 1-D tensor. Without them (say,
    a model with only convolution kernels) the guide is left out: "".
    """
    picked = _pick_examples(weights)
    if picked is None:
        return ""

    m, v = picked
    rows, cols = weights[m].shape
    col = min(7, cols - 1)

    return f"""\
## Working with the weights

`W` is a Python dict. Each key is a tensor's name, each value is a numpy
array. Changing the arrays in `W` is the only way to change the model.

In the examples below, `{m}` has shape {weights[m].shape}: {rows} rows and
{cols} columns. `W['{m}'][i, j]` is the value in row i, column j.

### 1. Viewing

Start with summaries. They are much cheaper than reading raw numbers:

```python
info()                                    # every tensor: shape, mean, std, min, max, share of zeros
W['{m}'].shape                            # (rows, columns)
W['{m}'].mean(axis=0)                     # the mean of each column
np.abs(W['{m}']).max()                    # the largest magnitude
np.argsort(W['{m}'][:, {col}])[-5:]        # the 5 rows with the largest values in column {col}
```

Then look at the numbers you need. `show()` prints a table with the row and
column numbers, so you always know where each value is:

```python
show('{m}')                               # the top-left corner
show('{m}', rows=slice(0, 20), cols=[0, {col}])   # rows 0 to 19, columns 0 and {col}
show('{v}')                               # a 1-D tensor is shown as one column
```

Or index the array directly; the result is printed:

```python
W['{m}'][2, {col}]                         # one value: row 2, column {col}
W['{m}'][:, {col}]                         # all of column {col}
W['{m}'][2]                               # all of row 2
W['{m}'][0:3, 0:4]                        # rows 0-2, columns 0-3 (the end is excluded)
W['{v}'][-1]                              # negative numbers count from the end
```

Limits: `show()` displays at most 50 rows and 30 columns. Printed arrays
with more than 1,000 values are shortened with "...". Every tool result is
cut off after {max_output_chars:,} characters, so view a slice or a summary
rather than a whole large tensor.

### 2. Editing

Edit by putting an index on the left of `=`. These all change `W` directly:

```python
W['{v}'][0] = 0.5                         # set one value
W['{v}'][0] += 0.1                        # add to one value
W['{m}'][:, {col}] = 0                     # set a whole column
W['{m}'][0:3, {col}] = [0.1, 0.2, 0.3]     # set several values (the lengths must match)
W['{m}'][:, {col}] *= 2                    # scale a column
W['{m}'][W['{m}'] < -1] = -1              # set every value that meets a condition
W['{m}'] = np.clip(W['{m}'], -1, 1)       # replace a whole tensor (keep the same shape)
```

After each call, the output ends with a line such as
`[Values changed: {m} {rows}]`. Use it to check that an edit did what you meant.
If there is no such line, nothing in `W` changed.

### 3. Pitfalls

- Keep every tensor's shape. A replacement must have exactly the same
  shape; check it with `.shape` first.
- `x = W['{m}'][:, {col}]` is a view: editing `x` in place also edits `W`.
  Use `.copy()` when you want an independent array.
- Use one index on the left of `=`. `W['{m}'][[0, 1]][:, 0] = 5` changes
  nothing, because the first index makes a copy. Write
  `W['{m}'][[0, 1], 0] = 5` instead.

### 4. Your environment

This is full Python 3.14 with numpy: loops, functions, comprehensions and the
standard library all work. There is no network, and no machine learning
libraries are installed. Files you save in /workspace are kept for the rest
of the run.

Being able to write a loop does not change the rules above.

Variables you create are kept between calls, so you can build an array in
one call and assign it in the next."""


def examples(guide: str) -> list[str]:
    """
    Every line of code in the guide's ```python blocks, for testing.
    """
    lines, inside = [], False

    for line in guide.splitlines():
        if line.startswith("```"):
            inside = line == "```python"
            continue
        if inside:
            lines.append(line.split("  #")[0].strip())

    return lines


def _pick_examples(weights: Weights) -> tuple[str, str] | None:
    """
    A matrix for the 2-D examples (the one with the most rows) and a vector
    for the 1-D examples, or None if the model has no such pair.
    """
    matrices = [name for name, array in weights.items() if array.ndim == 2]
    vectors = [name for name, array in weights.items() if array.ndim == 1]

    if not matrices or not vectors:
        return None

    matrix = max(matrices, key=lambda name: weights[name].shape[0])
    return matrix, vectors[0]
