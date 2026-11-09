"""Cached reference runs, data runs and surrogates under data/<plant>/.

    reference.npz / reference_<steps>.npz
    episodes_<tag>_seed<seed>.npz
    bll_<tag>_n<n>_noise<noise>_<target>_h<hidden>_<act>_seed<seed>.npz

with ``tag = mpc_p<perturbation>`` (dithered closed-loop excitation).
Training sets are nested: the first n data runs of one cached set.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from pcbll.data import Episode, closed_loop_episode, excitation_episodes, narx_dataset
from pcbll.plant.base import Plant

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"

__all__ = ["ROOT", "DATA", "RUNS", "FIGURES", "tag_for", "reference_run", "episodes", "load_episodes", "surrogate_path", "surrogate"]


def tag_for(perturbation: float = 0.15) -> str:
    return f"mpc_p{perturbation:g}"


def _save_episodes(path: pathlib.Path, eps: list[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"x{i}": e.x for i, e in enumerate(eps)}, **{f"u{i}": e.u for i, e in enumerate(eps)}, n=len(eps))


def load_episodes(path: pathlib.Path) -> list[Episode]:
    with np.load(path) as f:
        return [Episode(f[f"x{i}"], f[f"u{i}"]) for i in range(int(f["n"]))]


def reference_run(plant: Plant, n_steps: int | None = None, verbose: bool = True) -> Episode:
    """The nominal controller in closed loop from the plant's x0 (cached)."""
    spec = plant.spec
    n_steps = spec.episode_steps if n_steps is None else int(n_steps)
    path = DATA / spec.name / ("reference.npz" if n_steps == spec.episode_steps else f"reference_{n_steps}.npz")
    if path.exists():
        return load_episodes(path)[0]
    t = time.time()
    ref = closed_loop_episode(plant, plant.make_controller(), spec.x0, n_steps)
    _save_episodes(path, [ref])
    if verbose:
        print(f"{spec.name}: reference run, {n_steps} steps in {time.time() - t:.1f}s")
    return ref


def episodes(plant: Plant, n: int, perturbation: float = 0.15, seed: int = 0, verbose: bool = True) -> list[Episode]:
    """At least n dithered closed-loop data runs (cached, extended in place when more are needed)."""
    spec = plant.spec
    path = DATA / spec.name / f"episodes_{tag_for(perturbation)}_seed{seed}.npz"
    eps = load_episodes(path) if path.exists() else []
    if len(eps) < n:
        t = time.time()
        more = excitation_episodes(plant, n - len(eps), np.random.default_rng([seed, len(eps)]), perturbation=perturbation)
        eps = eps + more
        _save_episodes(path, eps)
        if verbose:
            print(f"{spec.name}: {len(more)} new data runs in {time.time() - t:.1f}s")
    return eps


def surrogate_path(plant: Plant, n: int, noise: float = 0.02, target: str = "delta", hidden=(40,), activation: str = "tanh", seed: int = 0, perturbation: float = 0.15) -> pathlib.Path:
    tag = tag_for(perturbation)
    return DATA / plant.spec.name / f"bll_{tag}_n{n}_noise{noise:g}_{target}_h{'x'.join(map(str, hidden))}_{activation}_seed{seed}.npz"


def surrogate(
    plant: Plant,
    n: int,
    noise: float | None = None,
    target: str = "delta",
    hidden=(40,),
    activation: str = "tanh",
    epochs: int = 3000,
    lr: float = 5e-3,
    patience: int | None = 500,
    seed: int = 0,
    perturbation: float = 0.15,
    val_episodes: int = 2,
    retrain: bool = False,
    verbose: bool = True,
):
    """A trained NARX BLL surrogate on the first n data runs (cached).

    Returns ``(surrogate, val_dataset)``; the validation set is the last
    ``val_episodes`` runs of a cache holding at least ``n + val_episodes``.
    ``noise`` defaults to 0 on process-noise plants (their data carries the
    noise) and to 0.02 otherwise.
    """
    from pcbll.bll.training import TrainingConfig
    from pcbll.surrogate import NarxSurrogate, train_surrogate

    if noise is None:
        noise = 0.0 if hasattr(plant, "noise_std") else 0.02
    path = surrogate_path(plant, n, noise, target, hidden, activation, seed, perturbation)
    eps = episodes(plant, n + val_episodes, perturbation, seed, verbose)
    fb = plant.spec.feedback_index
    val = narx_dataset(eps[-val_episodes:], np.random.default_rng([seed, 999]), noise, target, feedback=fb)
    if path.exists() and not retrain:
        return NarxSurrogate.load(path), val
    data = narx_dataset(eps[:n], np.random.default_rng([seed, n]), noise, target, feedback=fb)
    cfg = TrainingConfig(hidden=tuple(hidden), activation=activation, learning_rate=lr, max_epochs=epochs, patience=patience, seed=seed)
    t = time.time()
    sur, hist = train_surrogate(data, cfg, val=val)
    sur.save(path)
    if verbose:
        p = sur.model.predict(val.X)
        rmse = np.sqrt(np.mean((p.mean - val.Y_clean) ** 2, axis=0)) / val.Y_clean.std(axis=0)
        print(f"{plant.spec.name}: n={n} seed={seed}: {hist['epochs']} epochs, {time.time() - t:.0f}s, val nRMSE {np.round(rmse, 3)}, log_alpha {sur.posterior.log_alpha:.2f}")
    return sur, val
