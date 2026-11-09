"""CasADi expressions and functions for a trained :class:`BLLModel`.

Everything is built from the numpy artifact.  Inputs to the exported functions
are unscaled (plant units); the weight matrix W is in posterior units, exactly
as :meth:`BLLModel.predict_with_weights` takes it, so a scenario weight from
:meth:`WeightPosterior.weights_at` can be passed straight into an OCP.

:func:`feature_expr` / :func:`predict_expr` / :func:`variance_expr` embed the
surrogate in a larger symbolic graph (a do-mpc model); :func:`export` returns
ready-made ``ca.Function`` objects.
"""

from __future__ import annotations

import dataclasses

import casadi as ca
import numpy as np

from pcbll.bll.activations import casadi_activation
from pcbll.bll.model import BLLModel

__all__ = ["CasadiBLL", "feature_expr", "predict_expr", "variance_expr", "export"]


def feature_expr(model: BLLModel, x):
    """Augmented features phi (n_phi, 1) of an unscaled symbolic input x (n_x, 1)."""
    act = casadi_activation(model.features.activation)
    h = (x - ca.DM(model.scaler_x.mean)) / ca.DM(model.scaler_x.scale)
    for w, b in zip(model.features.weights, model.features.biases):
        h = act(ca.mtimes(ca.DM(w), h) + ca.DM(b))
    return ca.vertcat(h, 1.0)


def predict_expr(model: BLLModel, phi, W=None):
    """Unscaled output (n_y, 1) from features phi and a weight matrix W (n_phi, n_y), default the mean."""
    if W is None:
        W = ca.DM(model.posterior.mean)
    y_scaled = ca.mtimes(W.T, phi)
    return y_scaled * ca.DM(model.scaler_y.scale) + ca.DM(model.scaler_y.mean)


def variance_expr(model: BLLModel, phi):
    """(epistemic, aleatoric) variance vectors (n_y, 1) in unscaled units."""
    v = ca.mtimes(ca.DM(model.posterior.L_phi.T), phi)
    q = ca.dot(v, v)
    s2 = ca.DM(model.scaler_y.scale**2)
    sigma_e2 = ca.DM(model.posterior.sigma_e2)
    return q * sigma_e2 * s2, sigma_e2 * s2


@dataclasses.dataclass(frozen=True)
class CasadiBLL:
    features: ca.Function  #: x (n_x,1) -> phi (n_phi,1)
    predict: ca.Function  #: x (n_x,1), W (n_phi,n_y) -> y (n_y,1)
    predict_mean: ca.Function  #: x -> y at the posterior mean
    variances: ca.Function  #: x -> (var_epistemic (n_y,1), var_aleatoric (n_y,1))

    def weights_as_dm(self, W: np.ndarray) -> ca.DM:
        return ca.DM(np.asarray(W, dtype=float))


def export(model: BLLModel) -> CasadiBLL:
    x = ca.SX.sym("x", model.n_x)
    W = ca.SX.sym("W", model.n_phi, model.n_y)
    phi = feature_expr(model, x)
    var_epi, var_ale = variance_expr(model, phi)
    return CasadiBLL(
        features=ca.Function("features", [x], [phi], ["x"], ["phi"]),
        predict=ca.Function("predict", [x, W], [predict_expr(model, phi, W)], ["x", "W"], ["y"]),
        predict_mean=ca.Function("predict_mean", [x], [predict_expr(model, phi)], ["x"], ["y"]),
        variances=ca.Function("variances", [x], [var_epi, var_ale], ["x"], ["var_epistemic", "var_aleatoric"]),
    )
