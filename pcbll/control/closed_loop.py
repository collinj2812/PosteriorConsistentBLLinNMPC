"""Closed-loop simulation of a controller on the true plant, with measurement noise, and the metrics."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from pcbll.plant.base import Controller, Plant, PlantSpec

__all__ = ["ClosedLoopLog", "run_closed_loop", "violations"]


@dataclasses.dataclass(frozen=True)
class ClosedLoopLog:
    x: np.ndarray  #: (T+1, n_x) true states
    x_meas: np.ndarray  #: (T+1, n_x) what the controller saw
    u: np.ndarray  #: (T, n_u)
    info: list[dict]  #: per-step controller info (solve time, scenarios, selection metrics)
    #: branch predictions per step, (T, n_scenarios, N+1, n_x), if the controller stored them
    predictions: np.ndarray | None = None

    @property
    def n_steps(self) -> int:
        return self.u.shape[0]

    def solve_times(self) -> np.ndarray:
        return np.array([d.get("t_solve", np.nan) for d in self.info])

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        import json

        path = pathlib.Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.predictions is None else {"predictions": self.predictions}
        np.savez(path, x=self.x, x_meas=self.x_meas, u=self.u, info=json.dumps(self.info), **extra)
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ClosedLoopLog":
        import json

        with np.load(pathlib.Path(path).with_suffix(".npz")) as f:
            return cls(f["x"], f["x_meas"], f["u"], json.loads(str(f["info"])), f["predictions"] if "predictions" in f else None)


def run_closed_loop(
    plant: Plant,
    controller: Controller,
    x0: np.ndarray,
    n_steps: int,
    noise_std: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> ClosedLoopLog:
    """Run the controller on the plant; ``noise_std`` (n_x,) is added to the state the controller sees."""
    spec = plant.spec
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.empty((n_steps + 1, spec.n_x))
    x_meas = np.empty_like(x)
    u = np.empty((n_steps, spec.n_u))
    x[0] = np.asarray(x0, dtype=float).reshape(-1)
    x_meas[0] = x[0] if noise_std is None else x[0] + np.asarray(noise_std) * rng.standard_normal(spec.n_x)
    controller.reset(x_meas[0])
    preds = []
    for k in range(n_steps):
        u[k] = spec.clip_input(controller.step(x_meas[k]))
        pred = getattr(controller, "last_prediction", None)
        if pred is not None:
            preds.append(np.asarray(pred))
        x[k + 1] = plant.step(x[k], u[k])
        x_meas[k + 1] = x[k + 1] if noise_std is None else x[k + 1] + np.asarray(noise_std) * rng.standard_normal(spec.n_x)
    info = list(getattr(controller, "log", []))
    return ClosedLoopLog(x, x_meas, u, info, np.stack(preds) if len(preds) == n_steps else None)


def violations(spec: PlantSpec, X: np.ndarray) -> dict[str, dict[str, float]]:
    """Per constraint: fraction of steps violated and the largest violation, on the true states."""
    out = {}
    for c in spec.constraints:
        m = c.margin(np.asarray(X)[1:, spec.state_index(c.state)])
        out[f"{c.state}_{c.kind}"] = {"frequency": float(np.mean(m < 0)), "max": float(max(0.0, -m.min()))}
    return out
