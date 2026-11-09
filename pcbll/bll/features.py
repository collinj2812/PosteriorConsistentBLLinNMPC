"""The fixed nonlinear feature map phi(x): the network trunk as numpy arrays."""

from __future__ import annotations

import dataclasses

import numpy as np

from pcbll.bll.activations import numpy_activation

__all__ = ["FeatureMap"]


@dataclasses.dataclass(frozen=True)
class FeatureMap:
    """Dense trunk: scaled input (n_x,) -> hidden features (n_hidden,).

    Layer l computes ``act(weights[l] @ h + biases[l])`` with ``weights[l]`` of
    shape (n_out, n_in), the torch ``nn.Linear`` layout.  The activation is
    applied after every layer, including the last.  :meth:`features` appends
    the constant 1 for the bias of the last layer; the posterior refers to that
    augmented vector of length ``n_phi = n_hidden + 1``.
    """

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activation: str

    def __post_init__(self) -> None:
        weights = tuple(np.asarray(w, dtype=float) for w in self.weights)
        biases = tuple(np.asarray(b, dtype=float).reshape(-1) for b in self.biases)
        if len(weights) != len(biases) or not weights:
            raise ValueError("need the same, positive number of weight and bias arrays")
        n_in = weights[0].shape[1]
        for w, b in zip(weights, biases):
            if w.ndim != 2 or w.shape[1] != n_in or b.shape != (w.shape[0],):
                raise ValueError("inconsistent layer shapes")
            n_in = w.shape[0]
        numpy_activation(self.activation)  # validates the name
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "biases", biases)

    @property
    def n_x(self) -> int:
        return self.weights[0].shape[1]

    @property
    def n_hidden(self) -> int:
        return self.weights[-1].shape[0]

    @property
    def n_phi(self) -> int:
        return self.n_hidden + 1

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def hidden(self, x_scaled: np.ndarray) -> np.ndarray:
        """Trunk output without the bias feature, shape (m, n_hidden)."""
        act = numpy_activation(self.activation)
        h = np.asarray(x_scaled, dtype=float)
        if h.ndim != 2 or h.shape[1] != self.n_x:
            raise ValueError(f"x_scaled must have shape (m, {self.n_x})")
        for w, b in zip(self.weights, self.biases):
            h = act(h @ w.T + b)
        return h

    def features(self, x_scaled: np.ndarray) -> np.ndarray:
        """Augmented feature matrix Phi = [hidden, 1], shape (m, n_phi)."""
        h = self.hidden(x_scaled)
        return np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)

    def to_dict(self) -> dict:
        d: dict = {"activation": self.activation, "n_layers": self.n_layers}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            d[f"w{i}"] = w.copy()
            d[f"b{i}"] = b.copy()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureMap":
        n = int(d["n_layers"])
        return cls(tuple(d[f"w{i}"] for i in range(n)), tuple(d[f"b{i}"] for i in range(n)), str(d["activation"]))
