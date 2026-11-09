"""Closed-loop comparison of the scenario trees on a plant.

    python scripts/closed_loop.py --plant cstr_econ_pn0.02 --n-episodes 5 --seeds 0 1 2
    python scripts/closed_loop.py --plant synthetic_a1.4 --n-episodes 10 --seeds 0

Every controller gets the same objective, constraints, horizon and tuning;
the plant runs one episode from its x0.  One closed loop per (training-set
size, training draw, tree kind); logs go to runs/<plant>/ and are reused
unless ``--rerun``.  The aleatoric tightening applies to the
output_space_epistemic and weight_space trees (``--tightening-mode``); the
nominal controller and the lumped output_space interval take none.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from pcbll.cache import RUNS, surrogate
from pcbll.control import ClosedLoopLog, SurrogateMPC, TreeConfig, run_closed_loop, violations
from pcbll.control.surrogate_mpc import enable_hsl
from pcbll.plant import get_plant

TIGHTENED = ("output_space_epistemic", "weight_space")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", required=True, help="plant name, e.g. cstr_econ_pn0.02 or synthetic_a1.4")
    ap.add_argument("--n-episodes", type=int, nargs="+", default=[5], help="training-set sizes of the cached surrogates")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="data seeds of the surrogates (training draws)")
    ap.add_argument("--kinds", nargs="+", default=["nominal", "output_space", "output_space_epistemic", "weight_space"])
    ap.add_argument("--k", type=int, default=4, help="directions of the weight-space tree; scenarios = 1 + 2k")
    ap.add_argument("--r", type=float, default=2.0)
    ap.add_argument("--tightening", type=float, default=2.0, help="aleatoric back-off in units of sigma_n")
    ap.add_argument("--tightening-mode", choices=["constant", "tube"], default="tube")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--steps", type=int, default=None, help="closed-loop length (default: the plant's episode length)")
    ap.add_argument("--linearize", choices=["previous", "nominal_ocp"], default="previous")
    ap.add_argument("--measurement-noise", type=float, default=0.0, help="factor on the learned noise std added to the state the controller sees (0 = exact state feedback)")
    ap.add_argument("--input-move-weight", type=float, default=1.0, help="multiplier on the plant's input-change penalty")
    ap.add_argument("--linear-solver", default="ma57", help="IPOPT linear solver; falls back to mumps when the HSL library is not found")
    ap.add_argument("--noise-seed", type=int, default=0, help="seed of the process and measurement noise")
    ap.add_argument("--store-predictions", action="store_true", help="keep the branch predictions of every solve")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    solver = args.linear_solver
    if solver != "mumps" and not enable_hsl():
        print(f"HSL library not found: using mumps instead of {solver}")
        solver = "mumps"

    plant = get_plant(args.plant)
    spec = plant.spec
    run_dir = RUNS / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    n_steps = spec.episode_steps if args.steps is None else args.steps
    imw = "" if args.input_move_weight == 1.0 else f"_imw{args.input_move_weight:g}"
    rows = []
    for seed in args.seeds:
        for n in args.n_episodes:
            sur, _ = surrogate(plant, n, seed=seed, verbose=False)
            noise_std = np.sqrt(sur.posterior.sigma_e2) * sur.model.scaler_y.scale * args.measurement_noise
            for kind in args.kinds:
                mode = args.tightening_mode if kind in TIGHTENED else "constant"
                cfg = TreeConfig(kind=kind, horizon=args.horizon, r=args.r, n_directions=args.k, tightening=args.tightening, linearize=args.linearize,
                                 store_predictions=args.store_predictions, tightening_mode=mode,
                                 input_move_weight=args.input_move_weight, linear_solver=solver)
                tube = "_tube" if cfg.uses_tube else ""
                stem = f"cl_n{n}_draw{seed}_{kind}{tube}_k{args.k}_r{args.r:g}_c{args.tightening:g}_N{args.horizon}_T{n_steps}_mn{args.measurement_noise:g}{imw}_ns{args.noise_seed}"
                path = run_dir / f"{stem}.npz"
                if path.exists() and not args.rerun:
                    log = ClosedLoopLog.load(path)
                else:
                    t = time.time()
                    ctrl = SurrogateMPC(sur, plant, cfg)
                    if hasattr(plant, "reseed"):
                        plant.reseed(1000 + args.noise_seed)
                    log = run_closed_loop(plant, ctrl, spec.x0, n_steps, noise_std=noise_std, rng=np.random.default_rng(args.noise_seed))
                    log.save(path)
                    print(f"n={n} draw={seed} {kind}: {n_steps} steps in {time.time() - t:.0f}s", flush=True)
                v = violations(spec, log.x)
                ts = log.solve_times()
                rows.append({
                    "n": n, "draw": seed, "kind": kind, "scenarios": cfg.n_scenarios, "tightening_mode": mode, "cost": float(plant.cost(log.x, log.u)),
                    "viol_freq": {c: d["frequency"] for c, d in v.items()}, "viol_max": {c: d["max"] for c, d in v.items()},
                    "t_solve_mean": float(np.nanmean(ts)), "t_solve_max": float(np.nanmax(ts)),
                    "failures": int(sum(1 for d in log.info if not d.get("success", True))),
                })
    summary = run_dir / f"closed_loop_k{args.k}_r{args.r:g}_c{args.tightening:g}_N{args.horizon}_mn{args.measurement_noise:g}{imw}_ns{args.noise_seed}.json"
    summary.write_text(json.dumps(rows, indent=1))

    names = [f"{c.state}_{c.kind}" for c in spec.constraints]
    print(f"\n{spec.name}: {n_steps} steps, horizon {args.horizon}, r={args.r:g}, tightening {args.tightening:g} sigma_n ({args.tightening_mode}), k={args.k}, linear solver {solver}")
    print("| runs | draw | tree | scenarios | cost | " + " | ".join(f"viol {nm} (freq / max)" for nm in names) + " | solve mean / max [s] | fails |")
    print("|" + "---|" * (7 + len(names)))
    for r in rows:
        viol = " | ".join(f"{r['viol_freq'][nm]:.2f} / {r['viol_max'][nm]:.3g}" for nm in names)
        print(f"| {r['n']} | {r['draw']} | {r['kind']} | {r['scenarios']} | {r['cost']:.4g} | {viol} | {r['t_solve_mean']:.2f} / {r['t_solve_max']:.2f} | {r['failures']} |")
    if len(args.seeds) > 1:
        print("\nmean over draws")
        print("| runs | tree | cost | " + " | ".join(f"viol {nm} (freq / max)" for nm in names) + " | solve mean / max [s] |")
        print("|" + "---|" * (4 + len(names)))
        for n in args.n_episodes:
            for kind in args.kinds:
                sel = [r for r in rows if r["n"] == n and r["kind"] == kind]
                viol = " | ".join(f"{np.mean([r['viol_freq'][nm] for r in sel]):.2f} / {np.max([r['viol_max'][nm] for r in sel]):.3g}" for nm in names)
                print(f"| {n} | {kind} | {np.mean([r['cost'] for r in sel]):.4g} | {viol} | {np.mean([r['t_solve_mean'] for r in sel]):.2f} / {np.max([r['t_solve_max'] for r in sel]):.2f} |")
    print(summary)


if __name__ == "__main__":
    main()
