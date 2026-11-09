"""Rollouts of a NARX surrogate over an input sequence, under the different uncertainty models.

For one initial state x0 and inputs U (N, n_u):

* :func:`rollout`: the surrogate with a fixed last layer W (default: the mean).
* :func:`rollout_interval`: the output-space interval branch, x_{k+1} = mean_k + sign r sigma_k
  on every output at every step, sign held over the horizon; ``which`` selects
  the total or the epistemic-only standard deviation.
* :func:`rollout_samples`: Monte Carlo draws W_j from the posterior, each rolled
  out with its own fixed W_j (optionally with fresh noise per step).
* :func:`worst_case_rollouts`: for every (step k, output i), the fixed W on the
  radius-r ellipsoid that maximises / minimises x_{k,i} to first order (the
  weight-space branch for a constraint on that output at that step), rolled
  out exactly.
* :func:`feature_directions`: the whitened feature directions v_k along a
  rollout and their rotation.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pcbll.surrogate import NarxSurrogate

__all__ = ["rollout", "rollout_interval", "rollout_samples", "WorstCase", "worst_case_rollouts", "feature_directions"]


def _check(sur: NarxSurrogate, x0: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)
    if x0.size != sur.n_x or U.ndim != 2 or U.shape[1] != sur.n_u:
        raise ValueError("x0 must have n_x entries and U shape (N, n_u)")
    return x0, U


def rollout(sur: NarxSurrogate, x0: np.ndarray, U: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    """States (N+1, n_x) under one fixed last layer W (posterior mean if None)."""
    x0, U = _check(sur, x0, U)
    W = sur.posterior.mean if W is None else W
    X = np.empty((U.shape[0] + 1, sur.n_x))
    X[0] = x0
    for k in range(U.shape[0]):
        X[k + 1] = sur.next_with_weights(X[k], U[k], W)
    return X


def rollout_interval(sur: NarxSurrogate, x0: np.ndarray, U: np.ndarray, sign: float, r: float, which: str = "total") -> np.ndarray:
    """Output-space interval branch: mean + sign r sigma at every step, same sign on all outputs."""
    if which not in ("total", "epistemic"):
        raise ValueError("which must be 'total' or 'epistemic'")
    x0, U = _check(sur, x0, U)
    X = np.empty((U.shape[0] + 1, sur.n_x))
    X[0] = x0
    for k in range(U.shape[0]):
        p = sur.predict(X[k], U[k])
        sigma = p.std_total if which == "total" else p.std_epistemic
        X[k + 1] = p.mean + sign * r * sigma
    return X


def rollout_samples(sur: NarxSurrogate, x0: np.ndarray, U: np.ndarray, n: int, rng: np.random.Generator, with_noise: bool = False) -> np.ndarray:
    """n posterior rollouts, shape (n, N+1, n_x); with_noise adds N(0, sigma_e) per step."""
    x0, U = _check(sur, x0, U)
    Ws = sur.posterior.sample(n, rng)
    X = np.empty((n, U.shape[0] + 1, sur.n_x))
    X[:, 0] = x0
    noise_std = np.sqrt(sur.model.posterior.sigma_e2) * sur.model.scaler_y.scale
    for k in range(U.shape[0]):
        X[:, k + 1] = sur.next_batch(X[:, k], U[k], Ws)
        if with_noise:
            X[:, k + 1] += noise_std * rng.standard_normal((n, sur.n_x))
    return X


@dataclasses.dataclass(frozen=True)
class WorstCase:
    """Entry (k, i) of ``upper``/``lower`` is x_{k,i} under the ellipsoid weight that maximises/minimises it."""

    upper: np.ndarray  #: (N+1, n_x)
    lower: np.ndarray  #: (N+1, n_x)
    #: unit whitened directions D_{k,i}, shape (N+1, n_x, n_phi, n_y); row 0 is undefined (zeros)
    directions: np.ndarray


def worst_case_rollouts(sur: NarxSurrogate, x0: np.ndarray, U: np.ndarray, r: float, roll=None, jac=None) -> WorstCase:
    """First-order worst weights for every (step, output), each rolled out exactly with its fixed W."""
    x0, U = _check(sur, x0, U)
    N = U.shape[0]
    if roll is None or jac is None:
        roll, jac = sur.casadi_rollout(N)
    post = sur.posterior
    J = np.array(jac(x0, U.T, post.mean))  # ((N+1) n_x, n_phi n_y), both column-major
    upper = np.empty((N + 1, sur.n_x))
    lower = np.empty((N + 1, sur.n_x))
    upper[0] = lower[0] = x0
    directions = np.zeros((N + 1, sur.n_x, post.n_phi, post.n_y))
    for k in range(1, N + 1):
        for i in range(sur.n_x):
            row = J[k * sur.n_x + i]
            G = row.reshape(post.n_phi, post.n_y, order="F")
            if not np.any(G):
                upper[k, i] = lower[k, i] = float(np.array(roll(x0, U.T, post.mean))[i, k])
                continue
            D = post.direction_from_gradient(G)
            directions[k, i] = D
            X_up = np.array(roll(x0, U.T, post.weights_at(D, r)))
            X_lo = np.array(roll(x0, U.T, post.weights_at(-D, r)))
            upper[k, i] = X_up[i, k]
            lower[k, i] = X_lo[i, k]
    return WorstCase(upper, lower, directions)


def feature_directions(sur: NarxSurrogate, X: np.ndarray, U: np.ndarray) -> dict[str, np.ndarray]:
    """Whitened feature directions v_k = L^T phi_k / ||.|| along a trajectory and their rotation.

    Returns ``v`` (N, n_phi), ``angle_to_first`` (N,) and ``angle_consecutive`` (N-1,) in degrees,
    ``epistemic_ratio`` (N, n_x) = sigma_epistemic / sigma_aleatoric at each step, and the scalar
    ``rotation_index`` = sum_k s_k / || sum_k s_k v_k || with s_k the mean epistemic std at step k
    (1 when all v_k coincide).
    """
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    N = U.shape[0]
    L = sur.posterior.L_phi
    v = np.empty((N, sur.model.n_phi))
    ratio = np.empty((N, sur.n_x))
    for k in range(N):
        phi = sur.phi(X[k], U[k])
        w = L.T @ phi
        v[k] = w / np.linalg.norm(w)
        p = sur.predict(X[k], U[k])
        ratio[k] = p.std_epistemic / p.std_aleatoric
    cos_first = np.clip(v @ v[0], -1.0, 1.0)
    cos_next = np.clip(np.sum(v[1:] * v[:-1], axis=1), -1.0, 1.0)
    s_k = np.mean(np.sqrt(np.array([sur.predict(X[k], U[k]).var_epistemic for k in range(N)])), axis=1)
    index = float(np.sum(s_k) / max(np.linalg.norm(np.sum(s_k[:, None] * v, axis=0)), 1e-300))
    return {
        "v": v,
        "angle_to_first": np.degrees(np.arccos(cos_first)),
        "angle_consecutive": np.degrees(np.arccos(cos_next)),
        "epistemic_ratio": ratio,
        "rotation_index": index,
    }
