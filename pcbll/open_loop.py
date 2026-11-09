"""Open-loop comparison of the predicted sets under one fixed input sequence.

Along one nominal rollout of the surrogate from x0 under the inputs U:

1. the rotation of the whitened feature direction v_k, the direction of the
   worst weight the output-space interval implies at step k;
2. the epistemic-to-aleatoric ratio per output;
3. the width of the predicted set per step under
   - the output-space interval (total sigma, and epistemic sigma only),
   - the weight-space branches (per-(step, output) worst weights at radius r),
   - Monte Carlo rollouts of posterior weight draws (epistemic only, and with
     noise), summarised as mean +- r times the sample standard deviation and
     as the two-sided quantiles Phi(-r), Phi(r).

The output-space interval and the weight-space branches coincide at step 1
by construction and separate as v_k rotates.  Widths are per output in plant
units; ratios to the Monte Carlo width are the scale-free numbers.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
from scipy import stats

from pcbll.plant.base import PlantSpec
from pcbll.rollout import feature_directions, rollout, rollout_interval, rollout_samples, worst_case_rollouts
from pcbll.surrogate import NarxSurrogate

__all__ = ["DiagnosticConfig", "DiagnosticResult", "run_diagnostic", "summarize", "plot_diagnostic"]


@dataclasses.dataclass(frozen=True)
class DiagnosticConfig:
    horizon: int = 20
    r: float = 2.0
    n_samples: int = 500
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class DiagnosticResult:
    x0: np.ndarray
    U: np.ndarray  #: (N, n_u)
    nominal: np.ndarray  #: (N+1, n_x) surrogate mean rollout
    plant: np.ndarray | None  #: (N+1, n_x) true plant rollout, if given
    angle_to_first: np.ndarray  #: (N,) degrees
    angle_consecutive: np.ndarray  #: (N-1,) degrees
    epistemic_ratio: np.ndarray  #: (N, n_x)
    #: sum of per-step epistemic widths / width one fixed weight reaches at once (1 = no rotation)
    rotation_index: float
    output_space_total: tuple[np.ndarray, np.ndarray]  #: (upper, lower), each (N+1, n_x)
    output_space_epistemic: tuple[np.ndarray, np.ndarray]
    weight_space: tuple[np.ndarray, np.ndarray]
    mc_epistemic: tuple[np.ndarray, np.ndarray]  #: mean +- r std over posterior rollouts
    mc_total: tuple[np.ndarray, np.ndarray]  #: same with noise per step
    mc_epistemic_quantile: tuple[np.ndarray, np.ndarray]  #: quantiles Phi(r), Phi(-r) over posterior rollouts
    mc_total_quantile: tuple[np.ndarray, np.ndarray]
    r: float

    def width(self, name: str) -> np.ndarray:
        up, lo = getattr(self, name)
        return up - lo


def run_diagnostic(sur: NarxSurrogate, x0: np.ndarray, U: np.ndarray, config: DiagnosticConfig = DiagnosticConfig(), plant=None) -> DiagnosticResult:
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)[: config.horizon]
    rng = np.random.default_rng(config.seed)
    r = config.r

    nominal = rollout(sur, x0, U)
    rot = feature_directions(sur, nominal, U)
    os_t = (rollout_interval(sur, x0, U, +1.0, r, "total"), rollout_interval(sur, x0, U, -1.0, r, "total"))
    os_e = (rollout_interval(sur, x0, U, +1.0, r, "epistemic"), rollout_interval(sur, x0, U, -1.0, r, "epistemic"))
    wc = worst_case_rollouts(sur, x0, U, r)
    mc_e = rollout_samples(sur, x0, U, config.n_samples, rng, with_noise=False)
    mc_t = rollout_samples(sur, x0, U, config.n_samples, rng, with_noise=True)

    def band(samples):
        m, s = samples.mean(axis=0), samples.std(axis=0)
        return m + r * s, m - r * s

    p_hi = float(stats.norm.cdf(r))

    def quantile_band(samples):
        return np.quantile(samples, p_hi, axis=0), np.quantile(samples, 1.0 - p_hi, axis=0)

    plant_X = None
    if plant is not None:
        plant_X = np.empty_like(nominal)
        plant_X[0] = x0
        for k in range(U.shape[0]):
            plant_X[k + 1] = plant.step(plant_X[k], U[k])

    return DiagnosticResult(
        x0=x0, U=U, nominal=nominal, plant=plant_X,
        angle_to_first=rot["angle_to_first"], angle_consecutive=rot["angle_consecutive"], epistemic_ratio=rot["epistemic_ratio"], rotation_index=rot["rotation_index"],
        output_space_total=os_t, output_space_epistemic=os_e, weight_space=(wc.upper, wc.lower),
        mc_epistemic=band(mc_e), mc_total=band(mc_t),
        mc_epistemic_quantile=quantile_band(mc_e), mc_total_quantile=quantile_band(mc_t), r=r,
    )


def summarize(res: DiagnosticResult, spec: PlantSpec | None = None, states: list[str] | None = None) -> dict:
    """Scale-free numbers: rotation, epistemic share, and width ratios to the Monte Carlo set at the last step."""
    N = res.U.shape[0]
    idx = list(range(res.nominal.shape[1]))
    names = [f"x{i}" for i in idx]
    if spec is not None:
        names = list(spec.state_names)
        if states is not None:
            idx = [spec.state_index(s) for s in states]
            names = list(states)
    eps = 1e-300

    def ratio(name, ref, k=N):
        return {n: float(res.width(name)[k, i] / max(res.width(ref)[k, i], eps)) for n, i in zip(names, idx)}

    def mean_ratio_over_horizon(name, ref):
        w, wr = res.width(name)[1:], np.maximum(res.width(ref)[1:], eps)
        return {n: float(np.mean(w[:, i] / wr[:, i])) for n, i in zip(names, idx)}

    out = {
        "horizon": int(N),
        "r": float(res.r),
        "max_angle_to_first_deg": float(np.max(res.angle_to_first)),
        "final_angle_to_first_deg": float(res.angle_to_first[-1]),
        "mean_consecutive_angle_deg": float(np.mean(res.angle_consecutive)) if res.angle_consecutive.size else 0.0,
        "rotation_index": float(res.rotation_index),
        "epistemic_ratio_mean": {n: float(np.mean(res.epistemic_ratio[:, i])) for n, i in zip(names, idx)},
        "width_ratio_final": {
            "output_space_epistemic/mc_epistemic": ratio("output_space_epistemic", "mc_epistemic"),
            "weight_space/mc_epistemic": ratio("weight_space", "mc_epistemic"),
            "output_space_total/mc_total": ratio("output_space_total", "mc_total"),
            "output_space_epistemic/mc_epistemic_quantile": ratio("output_space_epistemic", "mc_epistemic_quantile"),
            "weight_space/mc_epistemic_quantile": ratio("weight_space", "mc_epistemic_quantile"),
        },
        "width_ratio_mean_over_horizon": {
            "output_space_epistemic/mc_epistemic": mean_ratio_over_horizon("output_space_epistemic", "mc_epistemic"),
            "weight_space/mc_epistemic": mean_ratio_over_horizon("weight_space", "mc_epistemic"),
        },
    }
    if res.plant is not None:
        err = np.abs(res.nominal - res.plant)[1:]
        out["surrogate_abs_error_final"] = {n: float(err[-1, i]) for n, i in zip(names, idx)}
        # fraction of steps at which the plant lies inside each band (1 = always)
        up, lo = res.mc_epistemic
        inside = (res.plant[1:] <= up[1:]) & (res.plant[1:] >= lo[1:])
        out["plant_inside_mc_epistemic"] = {n: float(np.mean(inside[:, i])) for n, i in zip(names, idx)}
        up, lo = res.output_space_epistemic
        inside = (res.plant[1:] <= up[1:]) & (res.plant[1:] >= lo[1:])
        out["plant_inside_output_space_epistemic"] = {n: float(np.mean(inside[:, i])) for n, i in zip(names, idx)}
        up, lo = res.mc_total
        inside = (res.plant[1:] <= up[1:]) & (res.plant[1:] >= lo[1:])
        out["plant_inside_mc_total"] = {n: float(np.mean(inside[:, i])) for n, i in zip(names, idx)}
    return out


def plot_diagnostic(res: DiagnosticResult, spec: PlantSpec, states: list[str], title: str = "", path: str | pathlib.Path | None = None):
    """Angle panel plus one panel per state with the nominal, plant, and the three sets."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = res.U.shape[0]
    k = np.arange(N + 1)
    fig, axes = plt.subplots(1 + len(states), 1, figsize=(7.5, 2.2 * (1 + len(states))), sharex=True)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    ax.plot(np.arange(N), res.angle_to_first, "k.-", label=f"angle(v_k, v_0); rotation index {res.rotation_index:.2f}")
    ax.plot(np.arange(1, N), res.angle_consecutive, "k.--", alpha=0.5, label="angle(v_k, v_{k-1})")
    ax.set_ylabel("rotation [deg]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=10)
    for ax, s in zip(axes[1:], states):
        i = spec.state_index(s)
        up, lo = res.mc_epistemic
        ax.fill_between(k, lo[:, i], up[:, i], color="C2", alpha=0.25, label=f"posterior MC ±{res.r:g}σ")
        up, lo = res.output_space_epistemic
        ax.plot(k, up[:, i], "C3-", label="output-space interval (epistemic)")
        ax.plot(k, lo[:, i], "C3-")
        up, lo = res.output_space_total
        ax.plot(k, up[:, i], "C3:", alpha=0.7, label="output-space interval (total)")
        ax.plot(k, lo[:, i], "C3:", alpha=0.7)
        up, lo = res.weight_space
        ax.plot(k, up[:, i], "C0-", label="weight-space worst case")
        ax.plot(k, lo[:, i], "C0-")
        ax.plot(k, res.nominal[:, i], "k-", lw=1, label="surrogate mean")
        if res.plant is not None:
            ax.plot(k, res.plant[:, i], "k--", lw=1, label="plant")
        for c in spec.constraints:
            if c.state == s:
                ax.axhline(c.bound, color="gray", ls="-.", lw=0.8)
        ax.set_ylabel(s)
    axes[1].legend(loc="best", fontsize=7, ncol=2)
    axes[-1].set_xlabel("prediction step k")
    fig.tight_layout()
    if path is not None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def save_summary(summary: dict, path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
