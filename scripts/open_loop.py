"""Open-loop comparison of the weight-space tree, the output-space interval and posterior Monte Carlo.

    python scripts/open_loop.py --plant cstr_econ_pn0.02 --n-episodes 5 --t0 0 15 30
    python scripts/open_loop.py --plant synthetic_a1.4 --n-episodes 10

On the cached surrogate, along the reference run from each start step: the
rotation of the worst weight over the horizon, the final width of the
weight-space set and of the output-space interval relative to the Monte
Carlo band, and whether the plant lies inside.  Figure and json go to
figures/<plant>/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from pcbll.cache import FIGURES, reference_run, surrogate
from pcbll.open_loop import DiagnosticConfig, plot_diagnostic, run_diagnostic, summarize
from pcbll.plant import get_plant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", required=True, help="plant name, e.g. cstr_econ_pn0.02 or synthetic_a1.4")
    ap.add_argument("--n-episodes", type=int, nargs="+", default=[5], help="training-set sizes of the cached surrogates")
    ap.add_argument("--t0", type=int, nargs="+", default=[0], help="start steps along the reference run")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--r", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="data seed of the surrogate; also seeds the Monte Carlo draws")
    ap.add_argument("--states", nargs="+", default=None, help="states to report (default: the constrained ones)")
    args = ap.parse_args()

    plant = get_plant(args.plant)
    spec = plant.spec
    states = args.states or sorted({c.state for c in spec.constraints}, key=spec.state_index)
    fig_dir = FIGURES / spec.name
    ref = reference_run(plant)
    dcfg = DiagnosticConfig(horizon=args.horizon, r=args.r, seed=args.seed)
    rows = []
    for n in args.n_episodes:
        sur, _ = surrogate(plant, n, seed=args.seed, verbose=False)
        for t0 in args.t0:
            if t0 + args.horizon > ref.n_steps:
                print(f"skip t0={t0}: horizon exceeds the reference run")
                continue
            res = run_diagnostic(sur, ref.x[t0], ref.u[t0 : t0 + args.horizon], dcfg, plant=plant)
            s = summarize(res, spec, states)
            s.update(n_episodes=n, t0=t0, seed=args.seed)
            rows.append(s)
            stem = f"open_loop_n{n}_seed{args.seed}_t{t0}"
            plot_diagnostic(res, spec, states, title=f"{spec.name}: n={n} runs, t0={t0}, r={args.r:g}", path=fig_dir / f"{stem}.png")
            (fig_dir / f"{stem}.json").write_text(json.dumps(s, indent=2))

    print(f"\n{spec.name}: horizon {args.horizon}, r={args.r:g}, states {states}")
    head = "| runs | t0 | max angle [deg] | mean step angle [deg] | epi/ale | output space / MC | weight space / MC | plant in output space | plant in MC |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for s in rows:
        ratio = np.mean([s["epistemic_ratio_mean"][x] for x in states])
        os_ = ", ".join(f"{s['width_ratio_final']['output_space_epistemic/mc_epistemic'][x]:.2f}" for x in states)
        ws = ", ".join(f"{s['width_ratio_final']['weight_space/mc_epistemic'][x]:.2f}" for x in states)
        pin = ", ".join(f"{s['plant_inside_output_space_epistemic'][x]:.2f}" for x in states)
        pmc = ", ".join(f"{s['plant_inside_mc_epistemic'][x]:.2f}" for x in states)
        print(f"| {s['n_episodes']} | {s['t0']} | {s['max_angle_to_first_deg']:.1f} | {s['mean_consecutive_angle_deg']:.1f} | {ratio:.2f} | {os_} | {ws} | {pin} | {pmc} |")


if __name__ == "__main__":
    main()
