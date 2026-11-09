"""
The `np` the agent sees: only the numpy functions listed here.

The real numpy module can read files (np.load, np.fromfile, np.genfromtxt),
download them (np.DataSource) and map memory (np.memmap), so the agent gets
a curated namespace instead. To give the agent a new function, add its
name to the right list. Every addition is a deliberate decision.
"""

from types import SimpleNamespace

import numpy

CREATION = [
    "array",
    "asarray",
    "copy",
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "full_like",
    "eye",
    "identity",
    "arange",
    "linspace",
    "meshgrid",
    "mgrid",
    "ogrid",
    "diag",
]

SHAPE = [
    "reshape",
    "ravel",
    "transpose",
    "expand_dims",
    "squeeze",
    "stack",
    "vstack",
    "hstack",
    "concatenate",
    "split",
    "tile",
    "repeat",
    "flip",
    "roll",
    "rot90",
    "pad",
    "tril",
    "triu",
    "broadcast_to",
]

SELECTION = [
    "where",
    "clip",
    "take",
    "sort",
    "argsort",
    "argmax",
    "argmin",
    "argwhere",
    "nonzero",
    "unique",
    "count_nonzero",
    "isfinite",
    "isnan",
    "isclose",
    "allclose",
    "any",
    "all",
]

MATHS = [
    "abs",
    "sign",
    "sqrt",
    "square",
    "power",
    "exp",
    "log",
    "log1p",
    "tanh",
    "sin",
    "cos",
    "maximum",
    "minimum",
    "round",
    "floor",
    "ceil",
    "dot",
    "matmul",
    "outer",
    "inner",
    "einsum",
    "tensordot",
    "kron",
    "convolve",
]

STATISTICS = [
    "sum",
    "prod",
    "cumsum",
    "mean",
    "median",
    "std",
    "var",
    "min",
    "max",
    "percentile",
    "quantile",
    "histogram",
    "corrcoef",
    "cov",
]

CONSTANTS_AND_TYPES = [
    "pi",
    "e",
    "inf",
    "nan",
    "newaxis",
    "float32",
    "float64",
    "int32",
    "int64",
]

# Leaving out lstsq and pinv is a research choice: they fit weights to data
# in closed form. Add them deliberately if an experiment allows that.
LINALG = [
    "norm",
    "svd",
    "eig",
    "eigh",
    "qr",
    "inv",
    "det",
    "matrix_rank",
    "solve",
]

RANDOM = ["default_rng"]


def curated_numpy() -> SimpleNamespace:
    top_level = CREATION + SHAPE + SELECTION + MATHS + STATISTICS + CONSTANTS_AND_TYPES

    return SimpleNamespace(
        **_pick(numpy, top_level),
        linalg=SimpleNamespace(**_pick(numpy.linalg, LINALG)),
        random=SimpleNamespace(**_pick(numpy.random, RANDOM)),
    )


def _pick(module, names: list[str]) -> dict:
    return {name: getattr(module, name) for name in names}
