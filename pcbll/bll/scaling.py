"""Standard scaling of inputs and outputs, as plain arrays."""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["StandardScaler"]


@dataclasses.dataclass(frozen=True)
class StandardScaler:
    """Column-wise (x - mean) / scale.

    ``scale`` is the population standard deviation (ddof=0).  A constant column
    gets scale 1 so it passes through unchanged.
    """

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=float).reshape(-1))
        object.__setattr__(self, "scale", np.asarray(self.scale, dtype=float).reshape(-1))
        if self.mean.shape != self.scale.shape:
            raise ValueError("mean and scale must have the same length")
        if np.any(self.scale <= 0):
            raise ValueError("scale must be positive")

    @classmethod
    def fit(cls, data: np.ndarray, min_scale: float = 1e-12) -> "StandardScaler":
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be 2-D, shape (m, n)")
        if data.shape[0] < 1:
            raise ValueError("data must contain at least one row")
        mean = data.mean(axis=0)
        scale = data.std(axis=0)
        scale = np.where(scale > min_scale, scale, 1.0)
        return cls(mean, scale)

    @classmethod
    def identity(cls, n: int) -> "StandardScaler":
        return cls(np.zeros(n), np.ones(n))

    @property
    def n(self) -> int:
        return self.mean.size

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (np.asarray(data, dtype=float) - self.mean) / self.scale

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=float) * self.scale + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.copy(), "scale": self.scale.copy()}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        return cls(d["mean"], d["scale"])
