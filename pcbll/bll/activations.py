"""One registry of activation functions with a numpy, a torch and a CasADi form.

Every activation the trunk may use is defined in all three backends so that the
numpy artifact, the torch trainer and the CasADi export agree by construction.
GELU is the exact erf form.  ReLU is non-smooth; prefer a smooth activation
inside an NLP.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import special

__all__ = ["ACTIVATIONS", "numpy_activation", "torch_activation", "casadi_activation"]

ACTIVATIONS: tuple[str, ...] = ("tanh", "sigmoid", "relu", "gelu", "softplus", "sin")

_SQRT2 = float(np.sqrt(2.0))


def _check(name: str) -> None:
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")


def numpy_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    _check(name)
    return {
        "tanh": np.tanh,
        "sigmoid": special.expit,
        "relu": lambda x: np.maximum(x, 0.0),
        "gelu": lambda x: 0.5 * x * (1.0 + special.erf(x / _SQRT2)),
        "softplus": lambda x: np.logaddexp(0.0, x),
        "sin": np.sin,
    }[name]


def torch_activation(name: str) -> Callable:
    _check(name)
    import torch
    import torch.nn.functional as F

    return {
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "relu": F.relu,
        "gelu": F.gelu,
        "softplus": F.softplus,
        "sin": torch.sin,
    }[name]


def casadi_activation(name: str) -> Callable:
    _check(name)
    import casadi as ca

    return {
        "tanh": ca.tanh,
        "sigmoid": lambda x: 1.0 / (1.0 + ca.exp(-x)),
        "relu": lambda x: ca.fmax(x, 0.0),
        "gelu": lambda x: 0.5 * x * (1.0 + ca.erf(x / _SQRT2)),
        "softplus": lambda x: ca.log(1.0 + ca.exp(x)),
        "sin": ca.sin,
    }[name]
