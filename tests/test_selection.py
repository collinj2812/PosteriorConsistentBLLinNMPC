"""Direction selection on the toy plant: gradients, max-min / SVD selection, coverage metrics, random baseline."""

import numpy as np
import pytest

from pcbll.bll.training import TrainingConfig
from pcbll.data import excitation_episodes, narx_dataset
from pcbll.plant.base import Constraint
from pcbll.rollout import worst_case_rollouts
from pcbll.surrogate import train_surrogate
from pcbll.tree import (
    captured_fraction,
    constraint_gradients,
    random_directions,
    scenario_extremes,
    scenario_weights,
    select_directions,
)
from toy_plant import ToyPlant


@pytest.fixture(scope="module")
def setup():
    plant = ToyPlant()
    eps = excitation_episodes(plant, 6, np.random.default_rng(0))
    data = narx_dataset(eps, np.random.default_rng(1), noise_level=0.02, target="delta")
    sur, _ = train_surrogate(data, TrainingConfig(hidden=(12,), learning_rate=1e-2, max_epochs=300, patience=None, seed=0))
    constraints = (Constraint("x1", "upper", 1.5), Constraint("x2", "lower", -1.0))
    x0, U = eps[0].x[0], eps[0].u[:6]
    G_w, rows, X = constraint_gradients(sur, x0, U, constraints, plant.spec)
    return plant, sur, constraints, x0, U, G_w, rows, X


def test_gradient_rows_and_margins(setup):
    plant, sur, constraints, x0, U, G_w, rows, X = setup
    assert G_w.shape == (12, sur.posterior.n_w) and len(rows) == 12
    assert [r.stage for r in rows[:6]] == list(range(1, 7))
    assert rows[0].sign == 1.0 and rows[6].sign == -1.0
    assert np.isclose(rows[0].nominal_margin, 1.5 - X[1, 0]) and np.isclose(rows[6].nominal_margin, X[1, 1] + 1.0)
    assert np.allclose([r.sensitivity for r in rows], np.linalg.norm(G_w, axis=1))
    # the sensitivity is the first-order epistemic std of the constraint value: at step 1 it equals the model's
    assert np.isclose(rows[0].sensitivity, sur.predict(x0, U[0]).std_epistemic[0], rtol=1e-6)


def test_single_row_selection_is_the_worst_direction(setup):
    plant, sur, constraints, x0, U, G_w, rows, X = setup
    post = sur.posterior
    sel = select_directions(post, G_w[:1], method="svd", n_directions=1)
    G = rows[0].sign * np.array(sur.casadi_rollout(1)[1](x0, U[:1].T, post.mean))[1 * sur.n_x + 0].reshape(post.n_phi, post.n_y, order="F")
    D_ref = post.direction_from_gradient(G)
    assert np.isclose(abs(np.sum(sel.directions[0] * D_ref)), 1.0, atol=1e-8)
    assert np.isclose(captured_fraction(G_w[:1], sel.directions)[0], 1.0)
    assert np.isclose(sel.explained[0], 1.0)


def test_svd_explained_fraction_and_full_rank_span(setup):
    plant, sur, constraints, x0, U, G_w, rows, X = setup
    post = sur.posterior
    sel = select_directions(post, G_w, method="svd", explained=0.99)
    assert 1 <= sel.n_directions <= 12 and sel.explained[sel.n_directions - 1] >= 0.99
    assert np.allclose(np.linalg.norm(sel.directions.reshape(sel.n_directions, -1), axis=1), 1.0)
    full = select_directions(post, G_w, method="svd", n_directions=12)
    frac = captured_fraction(G_w, full.directions)
    assert np.all(frac <= 1.0 + 1e-12) and np.all(frac > 0.0)
    # more directions never capture less
    few = captured_fraction(G_w, select_directions(post, G_w, method="svd", n_directions=2).directions)
    assert np.all(frac >= few - 1e-12)


def test_maxmin_guarantees_threshold(setup):
    plant, sur, constraints, x0, U, G_w, rows, X = setup
    post = sur.posterior
    for thr in (0.8, 0.95, 0.999):
        sel = select_directions(post, G_w, method="maxmin", cos_threshold=thr)
        frac = captured_fraction(G_w, sel.directions)
        assert np.all(frac >= thr - 1e-9)
        assert sel.spectrum.sum() == 12 and sel.assignment.min() >= 0
    strict = select_directions(post, G_w, method="maxmin", cos_threshold=0.999)
    loose = select_directions(post, G_w, method="maxmin", cos_threshold=0.8)
    assert strict.n_directions >= loose.n_directions
    budget = select_directions(post, G_w, method="maxmin", cos_threshold=0.999, n_directions=2)
    assert budget.n_directions == 2
    # the first direction is the largest (weighted) row; every added one is the worst-captured row
    first = G_w[np.argmax(np.linalg.norm(G_w, axis=1))]
    assert np.isclose(abs(np.sum(budget.directions[0].reshape(-1, order="F") * first)) / np.linalg.norm(first), 1.0)
    # zero weight on a row means it may be ignored
    w = np.ones(12)
    w[5] = 0.0
    sel = select_directions(post, G_w, method="maxmin", cos_threshold=0.999, weights=w)
    assert np.all(np.delete(captured_fraction(G_w, sel.directions), 5) >= 0.999 - 1e-9)


def test_scenario_weights_and_extremes_match_worst_case(setup):
    plant, sur, constraints, x0, U, G_w, rows, X = setup
    post = sur.posterior
    r = 2.0
    # one direction per row: the tree then reaches every constraint's own first-order worst case
    D = np.stack([post.direction_from_gradient(g.reshape(post.n_phi, post.n_y, order="F")) for g in _unwhiten_rows(post, G_w)])
    Ws, labels = scenario_weights(post, D, r)
    assert Ws.shape == (25, post.n_phi, post.n_y) and labels[0] == "nominal" and labels[1] == "+0"
    assert np.allclose(Ws[0], post.mean) and np.isclose(post.mahalanobis(Ws[1]), r)
    ext = scenario_extremes(sur, x0, U, Ws, rows)
    wc = worst_case_rollouts(sur, x0, U, r)
    for n, row in enumerate(rows):
        exact = row.sign * ((wc.upper if row.sign > 0 else wc.lower)[row.stage, row.state] - X[row.stage, row.state])
        assert np.isclose(ext["worst"][n], exact, rtol=1e-8, atol=1e-10)


def _unwhiten_rows(post, G_w):
    """Recover the raw gradient matrices G from whitened rows (inverse of _whiten_gradient), for the test above."""
    from scipy import linalg

    out = []
    for g in G_w:
        Z = g.reshape(post.n_phi, post.n_y, order="F") / post.sigma_e[None, :]
        out.append(linalg.solve_triangular(post.L_phi.T, Z, lower=False).reshape(-1, order="F"))
    return out


def test_random_directions_and_unknown_method(setup):
    plant, sur, constraints, x0, U, G_w, rows, X = setup
    post = sur.posterior
    R = random_directions(post, 5, np.random.default_rng(3))
    assert R.shape == (5, post.n_phi, post.n_y) and np.allclose(np.linalg.norm(R.reshape(5, -1), axis=1), 1.0)
    with pytest.raises(ValueError):
        select_directions(post, G_w, method="cluster")
