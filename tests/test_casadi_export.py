"""The CasADi functions reproduce the numpy artifact, for the mean and for arbitrary weights."""

import casadi as ca
import numpy as np
import pytest

from pcbll.bll import ACTIVATIONS
from pcbll.bll.casadi_export import export, feature_expr, predict_expr
from conftest import random_model


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_functions_match_numpy(activation):
    model, x, _ = random_model(activation=activation, hidden=(7, 4))
    cas = export(model)
    rng = np.random.default_rng(0)
    W = model.posterior.sample(1, rng)[0]
    pred = model.predict(x)
    y_w = model.predict_with_weights(x, W)
    for k in range(6):
        xk = x[k]
        assert np.allclose(np.array(cas.features(xk)).ravel(), model.phi(x[[k]])[0], atol=1e-12)
        assert np.allclose(np.array(cas.predict_mean(xk)).ravel(), pred.mean[k], atol=1e-10)
        assert np.allclose(np.array(cas.predict(xk, cas.weights_as_dm(W))).ravel(), y_w[k], atol=1e-10)
        var_epi, var_ale = cas.variances(xk)
        assert np.allclose(np.array(var_epi).ravel(), pred.var_epistemic[k], rtol=1e-10)
        assert np.allclose(np.array(var_ale).ravel(), pred.var_aleatoric[k], rtol=1e-10)


def test_scenario_weight_moves_output_by_r_sigma():
    """A weight from the ellipsoid, passed as a parameter, shifts the CasADi output by r * sigma_epistemic."""
    model, x, _ = random_model()
    cas = export(model)
    xk = x[3]
    phi = model.phi(x[[3]])
    r = 2.0
    for i in range(model.n_y):
        G = np.zeros_like(model.posterior.mean)
        G[:, i] = phi[0]
        W = model.posterior.weights_at(model.posterior.direction_from_gradient(G), r)
        shift = float(cas.predict(xk, ca.DM(W))[i] - cas.predict_mean(xk)[i])
        assert np.isclose(shift, r * model.predict(x[[3]]).std_epistemic[0, i])


def test_expressions_embed_in_a_larger_graph():
    model, x, _ = random_model()
    xs = ca.SX.sym("x", model.n_x)
    W = ca.SX.sym("W", model.n_phi, model.n_y)
    y = predict_expr(model, feature_expr(model, xs), W)
    g = ca.Function("g", [xs, W], [ca.sum1(y)])
    assert np.isclose(float(g(x[0], model.posterior.mean)), model.predict(x[[0]]).mean.sum())
    # differentiable with respect to the weights: the rollout sensitivities the selection rule needs
    J = ca.Function("J", [xs, W], [ca.jacobian(ca.sum1(y), W)])
    jac = np.array(J(x[0], model.posterior.mean)).reshape(model.n_phi, model.n_y, order="F")
    phi = model.phi(x[[0]])[0]
    assert np.allclose(jac, phi[:, None] * model.scaler_y.scale[None, :])
