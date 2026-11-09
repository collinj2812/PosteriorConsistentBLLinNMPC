"""Shared fixtures: small random BLL models built without training."""

from __future__ import annotations

import numpy as np
import pytest

from pcbll.bll import BLLModel, FeatureMap, StandardScaler, WeightPosterior


def random_feature_map(rng: np.random.Generator, n_x: int, hidden: tuple[int, ...], activation: str) -> FeatureMap:
    weights, biases = [], []
    n_in = n_x
    for h in hidden:
        weights.append(rng.standard_normal((h, n_in)) / np.sqrt(n_in))
        biases.append(0.1 * rng.standard_normal(h))
        n_in = h
    return FeatureMap(tuple(weights), tuple(biases), activation)


def random_model(
    seed: int = 0,
    n_x: int = 2,
    n_y: int = 2,
    hidden: tuple[int, ...] = (8, 6),
    activation: str = "tanh",
    m: int = 60,
    log_alpha: float = 2.0,
    head: str = "posterior_mean",
) -> tuple[BLLModel, np.ndarray, np.ndarray]:
    """A model with random trunk and a posterior fitted to random smooth targets; returns (model, x, y)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, (m, n_x))
    y = np.stack([np.sin(2.0 * x[:, 0] + k) + 0.5 * x[:, -1] ** 2 for k in range(n_y)], axis=1)
    y += 0.05 * rng.standard_normal(y.shape)
    scaler_x = StandardScaler.fit(x)
    scaler_y = StandardScaler.fit(y)
    fm = random_feature_map(rng, n_x, hidden, activation)
    Phi = fm.features(scaler_x.transform(x))
    Y = scaler_y.transform(y)
    sigma_e2 = np.exp(2.0 * rng.uniform(-2.0, -1.0, n_y))
    mean = rng.standard_normal((fm.n_phi, n_y)) if head == "trained" else None
    post = WeightPosterior.from_features(Phi, Y, log_alpha, sigma_e2, mean=mean)
    return BLLModel(scaler_x, scaler_y, fm, post, Phi, Y, head=head), x, y


@pytest.fixture
def model_xy():
    return random_model()


@pytest.fixture
def model(model_xy):
    return model_xy[0]
