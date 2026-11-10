# Posterior-consistent scenario trees for BLL surrogates in multi-stage NMPC

Scenario trees for multi-stage NMPC in which every branch is one fixed
last-layer weight matrix of a Bayesian last layer (BLL) surrogate, drawn from
the posterior ellipsoid along constraint-sensitive directions (sensitivity-
weighted max-min selection, Algorithm 1 of the paper), with the aleatoric
part as a constraint tightening. The code compares this weight-space tree
against the output-space interval branch (mean +- r sigma, sign held) in open
loop and in closed loop on a CSTR with a production-rate objective and on a
synthetic second-order plant.

## Install

```
pip install -e .
```

IPOPT runs with HSL MA57 when the HSL library is found (`HSL_DIR` or
`~/hsl/bin`), otherwise with mumps. Python >= 3.11.

## Reproducing the paper

All scripts take `--plant` and default to the paper's settings (r = 2, k = 4,
horizon 20, tightening 2 sigma_n, exact state feedback, one hidden layer of
40 tanh units, target "delta", 3000 epochs). Caches go to `data/`, logs to
`runs/`, figures and json to `figures/`; all three are git-ignored and
regenerable.

CSTR with 2 % process noise, 5 dithered closed-loop training runs, training
draws 0, 1, 2:

```
python scripts/train.py       --plant cstr_econ_pn0.02 --n-episodes 5 --seeds 0 1 2
python scripts/open_loop.py   --plant cstr_econ_pn0.02 --n-episodes 5 --t0 0 15 30
python scripts/selection.py   --plant cstr_econ_pn0.02 --n-episodes 5 --t0 0 15 30
python scripts/closed_loop.py --plant cstr_econ_pn0.02 --n-episodes 5 --seeds 0 1 2
```

Synthetic plant, 10 training runs:

```
python scripts/train.py       --plant synthetic_a1.4 --n-episodes 10
python scripts/open_loop.py   --plant synthetic_a1.4 --n-episodes 10
python scripts/selection.py   --plant synthetic_a1.4 --n-episodes 10
python scripts/closed_loop.py --plant synthetic_a1.4 --n-episodes 10 --seeds 0
```

| script | produces |
|---|---|
| `scripts/open_loop.py` | Fig. 3, Table 1 (rotation, set widths, plant inside) |
| `scripts/selection.py` | Fig. 4 and the coverage numbers (captured fraction against k, coverage against Phi(r)) |
| `scripts/closed_loop.py` | Tables 2 and 3 (cost, violations, solve times per tree) |

`scripts/closed_loop.py` reuses the logs under `runs/<plant>/` unless
`--rerun` is given; `--kinds` selects the controllers of Table 3: `nominal`,
`nominal_tube` (nominal with the aleatoric tube), `output_space` (total
sigma), `output_space_epistemic` (epistemic sigma plus tube), `random`
(random directions plus tube) and `weight_space` (Algorithm 1 plus tube).
Table 2 is the same script with `--k 2`, `4`, `8`, `16`.

## Layout

```
pcbll/
  bll/           Bayesian last layer: scaling, features, activations, posterior, training (torch), CasADi export
  surrogate.py   NarxSurrogate: the one-step map [x_k, u_k] -> x_{k+1} - x_k as a BLL model
  rollout.py     rollouts of the surrogate with fixed weights, posterior samples, and the output-space worst case
  data.py        reference run, dithered closed-loop excitation, NARX datasets
  plant/         Plant protocol, the CSTR and the synthetic plant, process-noise wrapper, get_plant
  tree.py        constraint gradients in whitened weight space, max-min / svd selection, scenario weights, captured fraction
  control/       do-mpc multi-stage controller with one weight matrix per branch, tightening, closed-loop harness
  open_loop.py   open-loop comparison of the tree, the interval and posterior Monte Carlo
  cache.py       cached reference runs, data runs and surrogates under data/
scripts/         train.py, open_loop.py, selection.py, closed_loop.py
tests/           pytest, no data files needed
```

## Tests

```
pytest
```
