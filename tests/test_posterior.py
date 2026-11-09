"""The weight posterior equals textbook Bayesian linear regression, and its weight-space geometry is right."""

import numpy as np
import pytest

from pcbll.bll import WeightPosterior


def make(seed=0, m=50, n_phi=5, n_y=2, log_alpha=1.0):
    rng = np.random.default_rng(seed)
    Phi = np.column_stack([rng.standard_normal((m, n_phi - 1)), np.ones(m)])
    Y = rng.standard_normal((m, n_y))
    sigma_e2 = np.array([0.1, 0.3])[:n_y]
    return Phi, Y, sigma_e2, WeightPosterior.from_features(Phi, Y, log_alpha, sigma_e2), log_alpha


def test_matches_bayesian_linear_regression_per_output():
    Phi, Y, sigma_e2, post, log_alpha = make()
    alpha = np.exp(log_alpha)
    for i in range(Y.shape[1]):
        sigma_w2 = alpha * sigma_e2[i]
        precision = Phi.T @ Phi / sigma_e2[i] + np.eye(Phi.shape[1]) / sigma_w2
        cov = np.linalg.inv(precision)
        mean = cov @ Phi.T @ Y[:, i] / sigma_e2[i]
        assert np.allclose(post.mean[:, i], mean, atol=1e-10)
        assert np.allclose(sigma_e2[i] * post.Sigma_phi, cov, atol=1e-10)
    assert np.allclose(post.L_phi @ post.L_phi.T, post.Sigma_phi, atol=1e-12)
    assert np.allclose(np.tril(post.L_phi), post.L_phi)
    assert post.n_phi == 5 and post.n_y == 2 and post.n_w == 10


def test_epistemic_variance_is_quadratic_form():
    Phi, Y, sigma_e2, post, _ = make()
    q = np.einsum("mi,ij,mj->m", Phi, post.Sigma_phi, Phi)
    assert np.allclose(post.epistemic_variance(Phi), q[:, None] * sigma_e2[None, :])


def test_whiten_round_trip_and_mahalanobis():
    _, _, _, post, _ = make()
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((post.n_phi, post.n_y))
    W = post.unwhiten(Z)
    assert np.allclose(post.whiten(W), Z, atol=1e-10)
    assert np.isclose(post.mahalanobis(W), np.linalg.norm(Z))
    assert np.isclose(post.mahalanobis(post.mean), 0.0)


def test_samples_have_the_posterior_moments():
    _, _, sigma_e2, post, _ = make(n_phi=3, n_y=2)
    rng = np.random.default_rng(2)
    Ws = post.sample(200_000, rng)
    assert Ws.shape == (200_000, 3, 2)
    assert np.allclose(Ws.mean(axis=0), post.mean, atol=2e-2)
    for i in range(2):
        cov = np.cov(Ws[:, :, i].T)
        assert np.allclose(cov, sigma_e2[i] * post.Sigma_phi, atol=2e-2 * np.max(np.abs(cov)) + 2e-3)
    # the two output columns are independent
    cross = np.cov(Ws[:, 0, 0], Ws[:, 0, 1])[0, 1]
    assert abs(cross) < 2e-3


def test_direction_from_gradient_reproduces_one_step_interval():
    """The linearised worst weight along phi gives exactly mean +- r * sigma_epistemic."""
    Phi, _, _, post, _ = make()
    phi = Phi[[7]]
    r = 2.5
    for i in range(post.n_y):
        G = np.zeros_like(post.mean)
        G[:, i] = phi[0]  # gradient of y_i = phi^T w_i with respect to W
        D = post.direction_from_gradient(G)
        assert np.isclose(np.linalg.norm(D), 1.0)
        assert np.allclose(D[:, [j for j in range(post.n_y) if j != i]], 0.0)
        y_mean = post.predict(phi)[0, i]
        up = post.predict(phi, post.weights_at(D, r))[0, i]
        down = post.predict(phi, post.weights_at(-D, r))[0, i]
        sigma_epi = np.sqrt(post.epistemic_variance(phi)[0, i])
        assert np.isclose(up - y_mean, r * sigma_epi)
        assert np.isclose(down - y_mean, -r * sigma_epi)
        assert np.isclose(post.mahalanobis(post.weights_at(D, r)), r)


def test_worst_direction_is_optimal_on_the_ellipsoid():
    """No other point of the radius-r ellipsoid moves the linear functional further."""
    Phi, _, _, post, _ = make()
    rng = np.random.default_rng(4)
    G = rng.standard_normal(post.mean.shape)
    r = 1.7
    best = np.sum(G * (post.weights_at(post.direction_from_gradient(G), r) - post.mean))
    for _ in range(50):
        Z = rng.standard_normal(post.mean.shape)
        Z *= r / np.linalg.norm(Z)
        assert np.sum(G * (post.unwhiten(Z) - post.mean)) <= best + 1e-12


def test_validation_and_dict_round_trip():
    Phi, Y, sigma_e2, post, _ = make()
    post2 = WeightPosterior.from_dict(post.to_dict())
    assert np.allclose(post2.mean, post.mean) and post2.log_alpha == post.log_alpha
    with pytest.raises(ValueError):
        post.weights_at(np.ones_like(post.mean), 1.0)
    with pytest.raises(ValueError):
        post.direction_from_gradient(np.zeros_like(post.mean))
    with pytest.raises(ValueError):
        post.predict(Phi, np.ones((2, 2)))
    with pytest.raises(ValueError):
        WeightPosterior.from_features(Phi, Y, 0.0, np.array([0.1, -0.3]))


def test_explicit_mean_keeps_covariance():
    Phi, Y, sigma_e2, post, log_alpha = make()
    W0 = np.ones_like(post.mean)
    kept = WeightPosterior.from_features(Phi, Y, log_alpha, sigma_e2, mean=W0)
    assert np.allclose(kept.mean, W0)
    assert np.allclose(kept.Sigma_phi, post.Sigma_phi)
