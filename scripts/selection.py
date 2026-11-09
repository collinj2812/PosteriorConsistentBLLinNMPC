"""Direction selection on a cached surrogate: captured fraction against k, and open-loop coverage.

    python scripts/selection.py --plant cstr_econ_pn0.02 --n-episodes 5 --t0 0 15 30
    python scripts/selection.py --plant synthetic_a1.4 --n-episodes 10

For each start step along the reference run: the whitened gradients of every
state-bound constraint at every stage over the horizon, then

1. the captured fraction kappa (sensitivity-weighted mean and minimum over the
   relevant rows, those with at least a tenth of the largest sensitivity) of
   the max-min, svd and random direction sets against the number of
   directions k, with the exact-rollout check of the weighted mean;
2. for the max-min tree with k directions, the empirical coverage of every
   relevant row by the tree's worst scenario over posterior Monte Carlo
   rollouts, at several radii r, against the design level Phi(r) and the
   first-order prediction Phi(r kappa).

Prints both tables and writes figures/<plant>/selection_n<n>_seed<seed>.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

from pcbll.cache import FIGURES, reference_run, surrogate
from pcbll.plant import get_plant
from pcbll.rollout import rollout, rollout_samples
from pcbll.tree import captured_fraction, constraint_gradients, random_directions, row_weights, scenario_extremes, scenario_weights, select_directions


def row_values(X: np.ndarray, rows) -> np.ndarray:
    """g for every row, X of shape (..., N+1, n_x) -> (..., n_rows)."""
    return np.stack([row.sign * (X[..., row.stage, row.state] - row.bound) for row in rows], axis=-1)


def exact_fraction(sur, x0, U, Ws, rows, r):
    """Exact-rollout analogue of captured_fraction: worst g over scenarios / (r * first-order sensitivity)."""
    ext = scenario_extremes(sur, x0, U, Ws, rows)
    sens = np.array([row.sensitivity for row in rows])
    return ext["worst"] / np.maximum(r * sens, 1e-300)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", required=True, help="plant name, e.g. cstr_econ_pn0.02 or synthetic_a1.4")
    ap.add_argument("--n-episodes", type=int, default=5, help="training-set size of the cached surrogate")
    ap.add_argument("--seed", type=int, default=0, help="data seed of the surrogate; also seeds the random directions and the Monte Carlo draws")
    ap.add_argument("--t0", type=int, nargs="+", default=[0], help="start steps along the reference run")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--r", type=float, default=2.0, help="radius of the exact-rollout check")
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16], help="numbers of directions k for the captured fraction")
    ap.add_argument("--k", type=int, default=4, help="directions of the max-min tree in the coverage check")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--samples", type=int, default=2000, help="posterior Monte Carlo rollouts")
    args = ap.parse_args()

    plant = get_plant(args.plant)
    spec = plant.spec
    fig_dir = FIGURES / spec.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    ref = reference_run(plant)
    sur, _ = surrogate(plant, args.n_episodes, seed=args.seed, verbose=False)
    post = sur.posterior
    roll, jac = sur.casadi_rollout(args.horizon)
    captured, coverage = [], []
    for t0 in args.t0:
        x0, U = ref.x[t0], ref.u[t0 : t0 + args.horizon]
        G_w, rows, X_nom = constraint_gradients(sur, x0, U, spec.constraints, spec, roll, jac)
        sens = row_weights(rows, "sensitivity")
        relevant = sens >= 0.1 * sens.max()
        print(f"\n{spec.name} n={args.n_episodes} seed={args.seed} t0={t0}: {len(rows)} constraint rows, {int(relevant.sum())} relevant, n_w={post.n_w}")

        # 1. captured fraction against k
        print("| method | k | scenarios | weighted mean kappa | min kappa (relevant) | exact weighted mean |")
        print("|---|---|---|---|---|---|")
        for k in args.counts:
            sets = {
                "maxmin": select_directions(post, G_w, method="maxmin", cos_threshold=1.0, n_directions=k, weights=sens).directions,
                "svd": select_directions(post, G_w, method="svd", n_directions=k).directions,
                "random": random_directions(post, k, np.random.default_rng([args.seed, t0, k])),
            }
            for name, D in sets.items():
                fo = captured_fraction(G_w, D)
                Ws, _ = scenario_weights(post, D, args.r)
                ex = exact_fraction(sur, x0, U, Ws, rows, args.r)
                res = dict(t0=t0, method=name, k=int(D.shape[0]), weighted_mean=float(np.sum(fo * sens) / sens.sum()), min_relevant=float(fo[relevant].min()),
                           min=float(fo.min()), exact_weighted_mean=float(np.sum(ex * sens) / sens.sum()))
                captured.append(res)
                print(f"| {name} | {res['k']} | {1 + 2 * res['k']} | {res['weighted_mean']:.3f} | {res['min_relevant']:.3f} | {res['exact_weighted_mean']:.3f} |")

        # 2. coverage of the max-min tree over posterior rollouts
        D = select_directions(post, G_w, method="maxmin", cos_threshold=1.0, n_directions=args.k, weights=sens).directions
        cap = captured_fraction(G_w, D)
        samples = rollout_samples(sur, x0, U, args.samples, np.random.default_rng([args.seed, t0]))
        g_samples = row_values(samples, rows)  # (S, n_rows)
        X_plant = np.empty_like(X_nom)
        X_plant[0] = x0
        for i in range(U.shape[0]):
            X_plant[i + 1] = plant.step(X_plant[i], U[i])
        g_plant = row_values(X_plant, rows)
        print(f"\nmax-min tree with k={args.k}: min kappa over relevant rows {cap[relevant].min():.3f}, {args.samples} posterior rollouts")
        print("| r | design Phi(r) | coverage min (relevant) | Phi(r kappa_min) | coverage mean (relevant) | Phi(r kappa) mean | plant inside |")
        print("|---|---|---|---|---|---|---|")
        for r in args.radii:
            Ws, _ = scenario_weights(post, D, r)
            g_tree = row_values(np.stack([rollout(sur, x0, U, W) for W in Ws]), rows).max(axis=0)
            cov = np.mean(g_samples <= g_tree[None, :], axis=0)
            pred = stats.norm.cdf(r * cap)
            res = dict(t0=t0, k=args.k, r=r, design=float(stats.norm.cdf(r)), cov_min=float(cov[relevant].min()), pred_min=float(stats.norm.cdf(r * cap[relevant].min())),
                       cov_mean=float(cov[relevant].mean()), pred_mean=float(pred[relevant].mean()), plant_inside=float(np.mean(g_plant[relevant] <= g_tree[relevant])))
            coverage.append(res)
            print(f"| {r:g} | {res['design']:.3f} | {res['cov_min']:.3f} | {res['pred_min']:.3f} | {res['cov_mean']:.3f} | {res['pred_mean']:.3f} | {res['plant_inside']:.2f} |")

    out = fig_dir / f"selection_n{args.n_episodes}_seed{args.seed}.json"
    out.write_text(json.dumps(dict(plant=spec.name, n=args.n_episodes, seed=args.seed, horizon=args.horizon, captured=captured, coverage=coverage), indent=1))
    print(out)


if __name__ == "__main__":
    main()
