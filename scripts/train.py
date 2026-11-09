"""Generate the reference run and the dithered training runs of a plant and train the BLL surrogates.

    python scripts/train.py --plant cstr_econ_pn0.02 --n-episodes 5 --seeds 0 1 2
    python scripts/train.py --plant synthetic_a1.4 --n-episodes 10

Everything is cached under data/<plant>/ (see pcbll.cache); one surrogate
per (training-set size, data seed).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from pcbll.cache import reference_run, surrogate, surrogate_path
from pcbll.plant import get_plant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", required=True, help="plant name, e.g. cstr_econ_pn0.02 or synthetic_a1.4")
    ap.add_argument("--n-episodes", type=int, nargs="+", default=[5], help="training-set sizes in data runs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="data seeds (independent training draws)")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[40], help="hidden layer widths (tanh)")
    ap.add_argument("--retrain", action="store_true", help="retrain cached surrogates")
    args = ap.parse_args()

    plant = get_plant(args.plant)
    reference_run(plant)
    for seed in args.seeds:
        for n in args.n_episodes:
            cached = surrogate_path(plant, n, hidden=tuple(args.hidden), seed=seed).exists() and not args.retrain
            sur, val = surrogate(plant, n, hidden=tuple(args.hidden), epochs=args.epochs, seed=seed, retrain=args.retrain, verbose=not cached)
            if cached:
                p = sur.model.predict(val.X)
                rmse = np.sqrt(np.mean((p.mean - val.Y_clean) ** 2, axis=0)) / val.Y_clean.std(axis=0)
                print(f"{plant.spec.name}: n={n} seed={seed}: cached, val nRMSE {np.round(rmse, 3)}, log_alpha {sur.posterior.log_alpha:.2f}")


if __name__ == "__main__":
    main()
