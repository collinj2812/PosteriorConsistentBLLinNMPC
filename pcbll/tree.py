"""Selecting the weight-space directions a scenario tree branches on.

Whitened coordinates: W = mean + L_phi Z diag(sigma_e); the posterior is a
standard normal on Z and the confidence ellipsoid is ||Z||_F = r.  For a
constraint value g(W) with gradient G = dg/dW at the mean,

    g(W) - g(mean)  ~  <G, W - mean>  =  <L_phi^T G diag(sigma_e), Z>  =:  <g_w, Z>,

so ``g_w`` (the whitened gradient) is the direction in which the constraint
is sensitive, and ``||g_w||`` is the first-order epistemic standard deviation
of the constraint value.  Its worst case on the ellipsoid is r ||g_w||,
attained at Z = r g_w / ||g_w||.

Two ways to turn the N*m constraint gradients into k directions:

* ``"maxmin"`` (Algorithm 1): greedy farthest-point selection on the captured
  fraction.  Start with the largest (weighted) row; repeatedly add the unit
  direction of the row whose weighted deficiency ``w_i (1 - captured_i)`` is
  largest, until every row is within the threshold or the budget is spent.
* ``"svd"``: stack the whitened gradients into a matrix and take the leading
  right-singular vectors (principal axes of the sensitivity).

:func:`captured_fraction` reports, per constraint, max_j |g_hat . D_j|, the
fraction of its first-order worst case a scenario set at +-r D_j reaches;
:func:`scenario_extremes` reports the same thing from exact rollouts.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from pcbll.bll.posterior import WeightPosterior
from pcbll.plant.base import Constraint, PlantSpec
from pcbll.rollout import rollout
from pcbll.surrogate import NarxSurrogate

__all__ = [
    "ConstraintRow",
    "Selection",
    "constraint_gradients",
    "select_directions",
    "scenario_weights",
    "captured_fraction",
    "scenario_extremes",
    "random_directions",
    "row_weights",
]


@dataclasses.dataclass(frozen=True)
class ConstraintRow:
    """One (constraint, stage) pair: g = sign * (x_{k,i} - bound) <= 0."""

    label: str
    stage: int
    state: int
    sign: float  #: +1 for an upper bound, -1 for a lower bound
    bound: float
    nominal_margin: float  #: -g at the nominal rollout, positive when satisfied
    sensitivity: float  #: ||g_w||, first-order epistemic std of g


def row_weights(rows: Sequence[ConstraintRow], scheme: str = "sensitivity") -> np.ndarray | None:
    """Row weights of the max-min selection: ``"sensitivity"`` = ``||g_i||``, ``"none"`` = unweighted."""
    if scheme == "sensitivity":
        return np.array([row.sensitivity for row in rows], dtype=float)
    if scheme == "none":
        return None
    raise ValueError(f"unknown weight scheme {scheme!r}")


def _whiten_gradient(post: WeightPosterior, G: np.ndarray) -> np.ndarray:
    """vec(L_phi^T G diag(sigma_e)), column-major, consistent with :meth:`WeightPosterior.weights_at`."""
    return ((post.L_phi.T @ G) * post.sigma_e[None, :]).reshape(-1, order="F")


def _unflatten(post: WeightPosterior, d: np.ndarray) -> np.ndarray:
    return d.reshape(post.n_phi, post.n_y, order="F")


def constraint_gradients(
    sur: NarxSurrogate,
    x0: np.ndarray,
    U: np.ndarray,
    constraints: Sequence[Constraint],
    spec: PlantSpec,
    roll=None,
    jac=None,
) -> tuple[np.ndarray, list[ConstraintRow], np.ndarray]:
    """Whitened gradients of every state-bound constraint at every stage along the nominal rollout.

    Returns ``G_w`` (n_rows, n_w), the rows, and the nominal states (N+1, n_x).
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)
    N = U.shape[0]
    if roll is None or jac is None:
        roll, jac = sur.casadi_rollout(N)
    post = sur.posterior
    X = np.array(roll(x0, U.T, post.mean)).T
    J = np.array(jac(x0, U.T, post.mean))
    rows_w, rows = [], []
    for c in constraints:
        i = spec.state_index(c.state)
        sign = 1.0 if c.kind == "upper" else -1.0
        for k in range(1, N + 1):
            G = sign * _unflatten(post, J[k * sur.n_x + i])
            g_w = _whiten_gradient(post, G)
            rows_w.append(g_w)
            rows.append(ConstraintRow(f"{c.state}_{c.kind}@{k}", k, i, sign, c.bound, float(c.margin(X[k, i])), float(np.linalg.norm(g_w))))
    return np.array(rows_w), rows, X


@dataclasses.dataclass(frozen=True)
class Selection:
    directions: np.ndarray  #: (n_dir, n_phi, n_y), unit Frobenius norm each
    method: str
    #: svd: singular values of the weighted gradient matrix; maxmin: number of rows assigned to each direction
    spectrum: np.ndarray
    #: svd: cumulative explained squared sensitivity; maxmin: cos threshold used
    explained: np.ndarray | float
    #: which direction each row is assigned to (maxmin) or -1
    assignment: np.ndarray

    @property
    def n_directions(self) -> int:
        return self.directions.shape[0]


def select_directions(
    post: WeightPosterior,
    G_w: np.ndarray,
    method: str = "svd",
    explained: float = 0.95,
    n_directions: int | None = None,
    cos_threshold: float = 0.9,
    weights: np.ndarray | None = None,
) -> Selection:
    """Reduce the whitened gradient rows to a few unit directions.

    ``weights`` scales the rows before every method.  For ``"svd"``,
    ``n_directions`` overrides the explained-fraction rule.  For
    ``"maxmin"``, the deficiency of a row is ``w_i (1 - captured_i) / max w``
    and the selection stops at ``max deficiency <= 1 - cos_threshold`` or at
    ``n_directions`` directions.
    """
    G_w = np.asarray(G_w, dtype=float)
    if G_w.ndim != 2 or G_w.shape[1] != post.n_w:
        raise ValueError("G_w must have shape (n_rows, n_w)")
    w = np.ones(G_w.shape[0]) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    A = G_w * w[:, None]
    if method == "svd":
        _, s, Vt = np.linalg.svd(A, full_matrices=False)
        cum = np.cumsum(s**2) / max(np.sum(s**2), 1e-300)
        k = int(np.searchsorted(cum, explained - 1e-12) + 1) if n_directions is None else int(n_directions)
        k = max(1, min(k, Vt.shape[0]))
        D = np.stack([_unflatten(post, Vt[j]) for j in range(k)])
        return Selection(D, "svd", s, cum, np.full(G_w.shape[0], -1))
    if method == "maxmin":
        norms = np.linalg.norm(A, axis=1)
        ok = norms > 0
        unit = np.zeros_like(A)
        unit[ok] = A[ok] / norms[ok, None]
        rel = w / max(w.max(), 1e-300)
        reps = [unit[int(np.argmax(norms))]]
        assignment = np.full(G_w.shape[0], -1)
        while True:
            cos = np.abs(unit @ np.array(reps).T)
            captured = np.max(cos, axis=1)
            assignment[:] = np.argmax(cos, axis=1)
            deficiency = rel * (1.0 - captured)
            worst = int(np.argmax(deficiency))
            if deficiency[worst] <= 1.0 - cos_threshold + 1e-12:
                break
            if n_directions is not None and len(reps) >= n_directions:
                break
            if not ok[worst]:
                break
            reps.append(unit[worst])
        D = np.stack([_unflatten(post, rep) for rep in reps])
        sizes = np.bincount(assignment[ok], minlength=len(reps)).astype(float)
        return Selection(D, "maxmin", sizes, float(cos_threshold), assignment)
    raise ValueError("method must be 'svd' or 'maxmin'")


def scenario_weights(post: WeightPosterior, D: np.ndarray, r: float, include_nominal: bool = True) -> tuple[np.ndarray, list[str]]:
    """+-r along every direction (plus the mean): weight matrices (n_s, n_phi, n_y) and labels."""
    D = np.asarray(D, dtype=float)
    Ws, labels = [], []
    if include_nominal:
        Ws.append(post.mean.copy())
        labels.append("nominal")
    for j in range(D.shape[0]):
        Ws.append(post.weights_at(D[j], r))
        labels.append(f"+{j}")
        Ws.append(post.weights_at(-D[j], r))
        labels.append(f"-{j}")
    return np.stack(Ws), labels


def captured_fraction(G_w: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Per constraint row, max_j |g_hat . D_j|: the fraction of its first-order worst case reached at +-r D_j."""
    G_w = np.asarray(G_w, dtype=float)
    Dm = np.asarray(D, dtype=float).reshape(D.shape[0], -1, order="F") if np.asarray(D).ndim == 3 else np.asarray(D, dtype=float)
    norms = np.linalg.norm(G_w, axis=1)
    out = np.zeros(G_w.shape[0])
    ok = norms > 0
    out[ok] = np.max(np.abs((G_w[ok] / norms[ok, None]) @ Dm.T), axis=1)
    return out


def scenario_extremes(sur: NarxSurrogate, x0: np.ndarray, U: np.ndarray, Ws: np.ndarray, rows: Sequence[ConstraintRow]) -> dict[str, np.ndarray]:
    """Exact rollouts of every scenario: per constraint row, the worst g over scenarios, relative to the nominal.

    Returns ``worst`` (n_rows,) = max_s g_s - g_nominal (>= 0 means the tree
    pushes the constraint the bad way, as intended), ``trajectories``
    (n_s, N+1, n_x).
    """
    X = np.stack([rollout(sur, x0, U, W) for W in Ws])
    nominal = rollout(sur, x0, U)
    worst = np.empty(len(rows))
    for n, row in enumerate(rows):
        g = row.sign * (X[:, row.stage, row.state] - row.bound)
        g0 = row.sign * (nominal[row.stage, row.state] - row.bound)
        worst[n] = np.max(g) - g0
    return {"worst": worst, "trajectories": X}


def random_directions(post: WeightPosterior, n: int, rng: np.random.Generator) -> np.ndarray:
    """n directions uniform on the unit sphere of whitened weight space, shape (n, n_phi, n_y)."""
    Z = rng.standard_normal((int(n), post.n_phi, post.n_y))
    return Z / np.linalg.norm(Z.reshape(n, -1), axis=1)[:, None, None]
