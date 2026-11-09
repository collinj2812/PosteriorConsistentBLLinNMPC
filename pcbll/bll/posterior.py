"""The matrix-normal posterior over the last layer, and the exact log marginal likelihood.

Model (scaled units), for output i with feature matrix Phi (m, n_phi):

    y_i = Phi w_i + e_i,   e_i ~ N(0, sigma_e2[i] I),   w_i ~ N(0, alpha * sigma_e2[i] I)

alpha is the signal-to-noise ratio shared by all outputs.  Then

    Lambda    = alpha^-1 I + Phi^T Phi           (scaled posterior precision)
    Sigma_phi = Lambda^-1
    w_i | data ~ N( Lambda^-1 Phi^T y_i,  sigma_e2[i] * Sigma_phi )

so W = [w_1 ... w_ny] is matrix-normal MN(mean, Sigma_phi, diag(sigma_e2)).
With L_phi the lower Cholesky factor of Sigma_phi, the whitened coordinates

    W = mean + L_phi Z diag(sigma_e),     Z ~ N(0, I) entrywise,

turn the posterior into a standard normal on Z, and ||Z||_F = r is the
confidence ellipsoid of radius r.  Scenario construction uses exactly this
parametrisation (:meth:`WeightPosterior.weights_at`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import linalg

__all__ = ["WeightPosterior", "exact_negative_lml"]


def _cholesky_lower(a: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor with one jitter retry for nearly singular SPD input."""
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * np.trace(a) / a.shape[0]
        try:
            return np.linalg.cholesky(a + jitter * np.eye(a.shape[0]))
        except np.linalg.LinAlgError as err:
            raise np.linalg.LinAlgError("posterior covariance is not positive definite") from err


@dataclasses.dataclass(frozen=True)
class WeightPosterior:
    mean: np.ndarray  #: (n_phi, n_y)
    Sigma_phi: np.ndarray  #: (n_phi, n_phi) shared feature-space covariance
    L_phi: np.ndarray  #: lower Cholesky factor, Sigma_phi = L_phi L_phi^T
    sigma_e2: np.ndarray  #: (n_y,) noise variances, scaled units
    log_alpha: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        sigma = np.asarray(self.Sigma_phi, dtype=float)
        L = np.asarray(self.L_phi, dtype=float)
        sigma_e2 = np.asarray(self.sigma_e2, dtype=float).reshape(-1)
        if mean.ndim != 2:
            raise ValueError("mean must have shape (n_phi, n_y)")
        n_phi, n_y = mean.shape
        if sigma.shape != (n_phi, n_phi) or L.shape != (n_phi, n_phi):
            raise ValueError("Sigma_phi and L_phi must have shape (n_phi, n_phi)")
        if sigma_e2.shape != (n_y,) or np.any(sigma_e2 <= 0):
            raise ValueError("sigma_e2 must be a positive vector of length n_y")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "Sigma_phi", sigma)
        object.__setattr__(self, "L_phi", L)
        object.__setattr__(self, "sigma_e2", sigma_e2)
        object.__setattr__(self, "log_alpha", float(self.log_alpha))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_features(
        cls,
        Phi: np.ndarray,
        Y: np.ndarray,
        log_alpha: float,
        sigma_e2: np.ndarray,
        mean: np.ndarray | None = None,
    ) -> "WeightPosterior":
        """Posterior from the training features Phi (m, n_phi) and scaled targets Y (m, n_y).

        ``mean`` overrides the closed-form posterior mean (to keep a
        gradient-trained last layer); the covariance never depends on it.
        """
        Phi = np.asarray(Phi, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Phi.ndim != 2 or Y.ndim != 2 or Phi.shape[0] != Y.shape[0]:
            raise ValueError("Phi and Y must be 2-D with the same number of rows")
        n_phi = Phi.shape[1]
        Lambda = np.exp(-float(log_alpha)) * np.eye(n_phi) + Phi.T @ Phi
        cho = linalg.cho_factor(Lambda, lower=True)
        if mean is None:
            mean = linalg.cho_solve(cho, Phi.T @ Y)
        Sigma = linalg.cho_solve(cho, np.eye(n_phi))
        Sigma = 0.5 * (Sigma + Sigma.T)
        return cls(mean, Sigma, _cholesky_lower(Sigma), sigma_e2, log_alpha)

    # ------------------------------------------------------------- properties
    @property
    def n_phi(self) -> int:
        return self.mean.shape[0]

    @property
    def n_y(self) -> int:
        return self.mean.shape[1]

    @property
    def n_w(self) -> int:
        """Number of uncertain weights, n_phi * n_y."""
        return self.mean.size

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha))

    @property
    def sigma_e(self) -> np.ndarray:
        return np.sqrt(self.sigma_e2)

    # ------------------------------------------------------------ predictive
    def epistemic_variance(self, Phi: np.ndarray) -> np.ndarray:
        """sigma_e2[i] * phi^T Sigma_phi phi for every row of Phi; shape (m, n_y)."""
        q = np.sum((np.asarray(Phi, dtype=float) @ self.L_phi) ** 2, axis=1)
        return q[:, None] * self.sigma_e2[None, :]

    def predict(self, Phi: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
        """Scaled outputs Phi @ W, with W defaulting to the posterior mean."""
        W = self.mean if W is None else np.asarray(W, dtype=float)
        if W.shape != self.mean.shape:
            raise ValueError(f"W must have shape {self.mean.shape}")
        return np.asarray(Phi, dtype=float) @ W

    # ---------------------------------------------------------- weight space
    def unwhiten(self, Z: np.ndarray) -> np.ndarray:
        """W = mean + L_phi Z diag(sigma_e); Z of shape (n_phi, n_y) or (n, n_phi, n_y)."""
        Z = np.asarray(Z, dtype=float)
        return self.mean + (self.L_phi @ Z) * self.sigma_e[None, :]

    def whiten(self, W: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`unwhiten` for one weight matrix."""
        W = np.asarray(W, dtype=float)
        if W.shape != self.mean.shape:
            raise ValueError(f"W must have shape {self.mean.shape}")
        return linalg.solve_triangular(self.L_phi, W - self.mean, lower=True) / self.sigma_e[None, :]

    def mahalanobis(self, W: np.ndarray) -> float:
        """Radius ||Z||_F of the ellipsoid on which W lies."""
        return float(np.linalg.norm(self.whiten(W)))

    def direction_from_gradient(self, G: np.ndarray) -> np.ndarray:
        """Unit whitened direction that maximises <G, W - mean> on the ellipsoid.

        For a quantity g(W) with gradient G = dg/dW (n_phi, n_y) at the mean,
        the linearised extreme of g over ||Z||_F <= r is attained at
        Z = r * D with D = L_phi^T G diag(sigma_e) / ||.||_F.  At one step and
        one output this reproduces mean +- r * sigma_epistemic exactly.
        """
        G = np.asarray(G, dtype=float)
        if G.shape != self.mean.shape:
            raise ValueError(f"G must have shape {self.mean.shape}")
        D = (self.L_phi.T @ G) * self.sigma_e[None, :]
        norm = np.linalg.norm(D)
        if norm == 0.0:
            raise ValueError("zero gradient has no direction")
        return D / norm

    def weights_at(self, D: np.ndarray, r: float) -> np.ndarray:
        """Weight matrix on the ellipsoid of radius r in unit direction D."""
        D = np.asarray(D, dtype=float)
        if not np.isclose(np.linalg.norm(D), 1.0, rtol=1e-6, atol=1e-9):
            raise ValueError("D must have unit Frobenius norm")
        return self.unwhiten(float(r) * D)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """n posterior draws of W, shape (n, n_phi, n_y)."""
        Z = rng.standard_normal((int(n), self.n_phi, self.n_y))
        return self.unwhiten(Z)

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "Sigma_phi": self.Sigma_phi.copy(),
            "L_phi": self.L_phi.copy(),
            "sigma_e2": self.sigma_e2.copy(),
            "log_alpha": self.log_alpha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightPosterior":
        return cls(d["mean"], d["Sigma_phi"], d["L_phi"], d["sigma_e2"], float(d["log_alpha"]))


def exact_negative_lml(Phi: np.ndarray, Y: np.ndarray, log_alpha: float, sigma_e2: np.ndarray) -> float:
    """Negative log marginal likelihood per data point, summed over outputs.

    For output i, -log p(y_i | Phi, alpha, sigma_e2[i]) with the last layer
    integrated out, in the n_phi-dimensional form

        m log sigma_e + (n_phi log alpha + log det Lambda) / 2
        + ( ||y - Phi w*||^2 / sigma_e2 + ||w*||^2 / (alpha sigma_e2) ) / 2,

    w* the posterior mean.  Divided by m and summed over i, this is the loss
    the torch trainer minimises (evaluated there at the current last layer
    instead of w*, an upper bound that is tight at w*).
    """
    Phi = np.asarray(Phi, dtype=float)
    Y = np.asarray(Y, dtype=float)
    sigma_e2 = np.asarray(sigma_e2, dtype=float).reshape(-1)
    m, n_phi = Phi.shape
    n_y = Y.shape[1]
    alpha = float(np.exp(log_alpha))
    Lambda = np.eye(n_phi) / alpha + Phi.T @ Phi
    cho = linalg.cho_factor(Lambda, lower=True)
    logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
    W = linalg.cho_solve(cho, Phi.T @ Y)
    resid = Y - Phi @ W
    J = 0.5 * n_y * np.log(2.0 * np.pi) + n_y * (logdet + n_phi * np.log(alpha)) / (2.0 * m)
    J += 0.5 * np.sum(np.log(sigma_e2))
    J += np.sum(np.sum(resid**2, axis=0) / sigma_e2) / (2.0 * m)
    J += np.sum(np.sum(W**2, axis=0) / (alpha * sigma_e2)) / (2.0 * m)
    return float(J)
