"""A two-state nonlinear oscillator as a numpy plant for tests (no do-mpc, no IPOPT for the plant itself)."""

from __future__ import annotations

import numpy as np

from pcbll.plant.base import Constraint, PlantSpec


class ToyPlant:
    """x1' = x1 + dt x2;  x2' = x2 + dt (-x1 - 0.3 x2 - 0.2 x1^3 + u).

    The MPC task drives x1 to 1.4 under the bound x1 <= 1.5, so the
    constraint is nearly active and the tree choice matters.
    """

    dt = 0.1
    setpoint = 1.4

    def __init__(self):
        self.spec = PlantSpec(
            name="toy",
            state_names=("x1", "x2"),
            input_names=("u",),
            x0=np.array([1.0, 0.0]),
            u_lower=np.array([-2.0]),
            u_upper=np.array([2.0]),
            t_step=self.dt,
            episode_steps=40,
            constraints=(Constraint("x1", "upper", 1.5),),
            x0_jitter=np.array([0.3, 0.3]),
        )

    def step(self, x, u):
        x = np.asarray(x, dtype=float)
        u = float(np.asarray(u).reshape(-1)[0])
        return np.array([x[0] + self.dt * x[1], x[1] + self.dt * (-x[0] - 0.3 * x[1] - 0.2 * x[0] ** 3 + u)])

    def sample_x0(self, rng):
        return self.spec.x0 + self.spec.x0_jitter * rng.standard_normal(2)

    def make_controller(self):
        return PController()

    def configure_mpc(self, mpc, model, back_off=None, scenario_weight: float = 1.0) -> None:
        b = 0.0 if not back_off else float(back_off.get("x1", 0.0))
        mpc.set_objective(mterm=(model.x["x1"] - self.setpoint) ** 2, lterm=(model.x["x1"] - self.setpoint) ** 2)
        mpc.set_rterm(u=1e-3)
        mpc.bounds["upper", "_x", "x1"] = 1.5 - b
        mpc.bounds["lower", "_u", "u"] = -2.0
        mpc.bounds["upper", "_u", "u"] = 2.0

    def cost(self, X, U) -> float:
        return float(np.sum((np.asarray(X)[1:, 0] - self.setpoint) ** 2))


class PController:
    def reset(self, x0):
        pass

    def step(self, x):
        return np.array([-0.8 * x[0] - 0.5 * x[1]])
