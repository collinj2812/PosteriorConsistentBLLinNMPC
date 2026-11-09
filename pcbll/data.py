"""Closed-loop episodes on a plant and the one-step NARX dataset built from them.

Excitation is the plant's nominal controller in closed loop, with a
piecewise-constant random perturbation added to the applied input and a
jittered initial state.  The dataset maps ``[x_k[feedback], u_k]`` to
``x_{k+1}`` (``target="next"``) or to ``x_{k+1} - x_k`` (``target="delta"``).
Measurement noise is added to the targets only, with a standard deviation of
``noise_level`` times the per-state standard deviation of the targets.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from pcbll.plant.base import Controller, Plant

__all__ = ["Episode", "NarxDataset", "closed_loop_episode", "excitation_episodes", "narx_dataset"]


@dataclasses.dataclass(frozen=True)
class Episode:
    x: np.ndarray  #: (T+1, n_x)
    u: np.ndarray  #: (T, n_u)

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        u = np.asarray(self.u, dtype=float)
        if x.ndim != 2 or u.ndim != 2 or x.shape[0] != u.shape[0] + 1:
            raise ValueError("x must have one more row than u")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "u", u)

    @property
    def n_steps(self) -> int:
        return self.u.shape[0]


def closed_loop_episode(
    plant: Plant,
    controller: Controller,
    x0: np.ndarray,
    n_steps: int,
    perturbation: np.ndarray | None = None,
) -> Episode:
    """Closed loop with the controller; ``perturbation`` (n_steps, n_u) is added to the applied input and clipped."""
    spec = plant.spec
    x = np.empty((n_steps + 1, spec.n_x))
    u = np.empty((n_steps, spec.n_u))
    x[0] = np.asarray(x0, dtype=float).reshape(-1)
    controller.reset(x[0])
    for k in range(n_steps):
        uk = controller.step(x[k])
        if perturbation is not None:
            uk = uk + perturbation[k]
        u[k] = spec.clip_input(uk)
        x[k + 1] = plant.step(x[k], u[k])
    return Episode(x, u)


def _piecewise_constant(rng: np.random.Generator, n_steps: int, n_u: int, hold: tuple[int, int]) -> np.ndarray:
    """Uniform(-1, 1) values per input, held for random durations in ``hold`` (inclusive)."""
    out = np.empty((n_steps, n_u))
    k = 0
    while k < n_steps:
        h = int(rng.integers(hold[0], hold[1] + 1))
        out[k : k + h] = rng.uniform(-1.0, 1.0, n_u)
        k += h
    return out


def excitation_episodes(
    plant: Plant,
    n_episodes: int,
    rng: np.random.Generator,
    mode: str = "mpc",
    n_steps: int | None = None,
    perturbation: float = 0.15,
    hold: tuple[int, int] = (3, 10),
    controller: Controller | None = None,
) -> list[Episode]:
    """Data runs on the plant: nominal closed loop plus a piecewise-constant input perturbation.

    ``perturbation`` is the amplitude relative to the input range, held for a
    random number of steps in ``hold``; x0 is drawn by ``plant.sample_x0``.
    """
    if mode != "mpc":
        raise ValueError("mode must be 'mpc'")
    spec = plant.spec
    n_steps = spec.episode_steps if n_steps is None else int(n_steps)
    controller = plant.make_controller() if controller is None else controller
    span = spec.u_upper - spec.u_lower
    episodes = []
    for _ in range(n_episodes):
        pert = perturbation * span * _piecewise_constant(rng, n_steps, spec.n_u, hold)
        episodes.append(closed_loop_episode(plant, controller, plant.sample_x0(rng), n_steps, pert))
    return episodes


@dataclasses.dataclass(frozen=True)
class NarxDataset:
    X: np.ndarray  #: (m, n_fb + n_u) = [x_k[feedback], u_k]
    Y: np.ndarray  #: (m, n_x) noisy targets
    Y_clean: np.ndarray  #: (m, n_x) targets before noise
    noise_std: np.ndarray  #: (n_x,) absolute noise standard deviation per state
    target: str
    n_x: int
    n_u: int
    #: indices of the states used as inputs
    feedback: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.feedback:
            object.__setattr__(self, "feedback", tuple(range(self.n_x)))

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=self.X, Y=self.Y, Y_clean=self.Y_clean, noise_std=self.noise_std, target=self.target, n_x=self.n_x, n_u=self.n_u, feedback=np.array(self.feedback))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "NarxDataset":
        with np.load(pathlib.Path(path).with_suffix(".npz")) as f:
            fb = tuple(int(i) for i in f["feedback"]) if "feedback" in f else ()
            return cls(f["X"], f["Y"], f["Y_clean"], f["noise_std"], str(f["target"]), int(f["n_x"]), int(f["n_u"]), fb)


def narx_dataset(episodes: list[Episode], rng: np.random.Generator, noise_level: float = 0.0, target: str = "next", feedback=None) -> NarxDataset:
    """``feedback``: indices of the states used as inputs (default all)."""
    if target not in ("next", "delta"):
        raise ValueError("target must be 'next' or 'delta'")
    if not episodes:
        raise ValueError("no episodes")
    n_x = episodes[0].x.shape[1]
    n_u = episodes[0].u.shape[1]
    fb = tuple(range(n_x)) if feedback is None else tuple(int(i) for i in feedback)
    X = np.concatenate([np.concatenate([e.x[:-1][:, fb], e.u], axis=1) for e in episodes], axis=0)
    Y = np.concatenate([e.x[1:] - e.x[:-1] if target == "delta" else e.x[1:] for e in episodes], axis=0)
    noise_std = float(noise_level) * Y.std(axis=0)
    Y_noisy = Y + noise_std * rng.standard_normal(Y.shape)
    return NarxDataset(X, Y_noisy, Y, noise_std, target, n_x, n_u, fb)
