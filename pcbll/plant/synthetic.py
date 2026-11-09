"""A synthetic second-order plant with a saturating restoring force.

    x1+ = x1 + dt x2
    x2+ = x2 + dt ( -x1 - damping x2 - gain tanh(beta x1) / beta + u )
    s   = x1 + kappa x2,   d = x1 - kappa x2

``beta`` sets the nonlinearity (linear for beta -> 0, saturating at
|x1| > 1/beta); ``setpoint`` sets the size of the transient and therefore how
far the features travel through the saturation.  The constraints scale with
the setpoint (x1 <= 1.07 sp, s <= 1.21 sp, d >= -0.29 sp) so that all three
are active during the transient at every amplitude.  ``s`` and ``d`` are
carried as states so that constraints on combinations of outputs are
ordinary state bounds; only ``x1`` and ``x2`` are fed back to the surrogate.
The nominal controller is a do-mpc MPC on the exact discrete model.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import do_mpc
from casadi import tanh as ca_tanh

from pcbll.plant.base import Constraint, PlantSpec
from pcbll.plant.cstr import DoMpcController, _backoff

__all__ = ["SyntheticParams", "SyntheticPlant"]


@dataclasses.dataclass(frozen=True)
class SyntheticParams:
    beta: float = 3.0
    dt: float = 0.1
    damping: float = 0.8
    gain: float = 1.0
    kappa: float = 0.5
    setpoint: float = 1.4
    #: large enough that the transient is limited by the constraints, not by input saturation
    u_max: float = 12.0
    episode_steps: int = 40
    horizon: int = 20

    @property
    def x1_max(self) -> float:
        return 1.5 / 1.4 * self.setpoint

    @property
    def s_max(self) -> float:
        return 1.7 / 1.4 * self.setpoint

    @property
    def d_min(self) -> float:
        return -0.4 / 1.4 * self.setpoint


class SyntheticPlant:
    def __init__(self, setpoint: float = 1.4, beta: float = 3.0, **kwargs):
        self.params = SyntheticParams(beta=float(beta), setpoint=float(setpoint), **kwargs)
        p = self.params
        self.spec = PlantSpec(
            name=f"synthetic_a{p.setpoint:g}" + ("" if p.beta == 3.0 else f"_b{p.beta:g}") + ("" if p.u_max == 12.0 else f"_u{p.u_max:g}"),
            state_names=("x1", "x2", "s", "d"),
            input_names=("u",),
            x0=np.zeros(4),
            u_lower=np.array([-p.u_max]),
            u_upper=np.array([p.u_max]),
            t_step=p.dt,
            episode_steps=p.episode_steps,
            constraints=(Constraint("x1", "upper", p.x1_max), Constraint("s", "upper", p.s_max), Constraint("d", "lower", p.d_min)),
            x0_jitter=np.array([0.15, 0.15, 0.0, 0.0]) * p.setpoint,
            feedback=("x1", "x2"),
        )

    # ----------------------------------------------------------- dynamics
    def _next(self, x1, x2, u, tanh):
        p = self.params
        restoring = p.gain * tanh(p.beta * x1) / p.beta
        x1n = x1 + p.dt * x2
        x2n = x2 + p.dt * (-x1 - p.damping * x2 - restoring + u)
        return x1n, x2n

    def derived(self, x1, x2):
        return x1 + self.params.kappa * x2, x1 - self.params.kappa * x2

    def step(self, x, u):
        x = np.asarray(x, dtype=float)
        u = float(np.asarray(u).reshape(-1)[0])
        x1n, x2n = self._next(x[0], x[1], u, np.tanh)
        s, d = self.derived(x1n, x2n)
        return np.array([x1n, x2n, s, d])

    def sample_x0(self, rng):
        x = self.spec.x0[:2] + self.spec.x0_jitter[:2] * rng.standard_normal(2)
        return np.array([x[0], x[1], *self.derived(x[0], x[1])])

    def consistent(self, x):
        x = np.asarray(x, dtype=float).copy()
        x[2], x[3] = self.derived(x[0], x[1])
        return x

    # --------------------------------------------------------------- do-mpc
    def make_model(self):
        model = do_mpc.model.Model("discrete", "SX")
        x1 = model.set_variable("_x", "x1")
        x2 = model.set_variable("_x", "x2")
        model.set_variable("_x", "s")
        model.set_variable("_x", "d")
        u = model.set_variable("_u", "u")
        x1n, x2n = self._next(x1, x2, u, ca_tanh)
        s, d = self.derived(x1n, x2n)
        model.set_rhs("x1", x1n)
        model.set_rhs("x2", x2n)
        model.set_rhs("s", s)
        model.set_rhs("d", d)
        model.setup()
        return model

    def configure_mpc(self, mpc, model, back_off=None, scenario_weight: float = 1.0) -> None:
        p = self.params
        b = _backoff(back_off)
        mpc.set_objective(mterm=(model.x["x1"] - p.setpoint) ** 2, lterm=(model.x["x1"] - p.setpoint) ** 2)
        mpc.set_rterm(u=1e-2)
        mpc.bounds["upper", "_x", "x1"] = p.x1_max - b("x1")
        mpc.bounds["upper", "_x", "s"] = p.s_max - b("s")
        mpc.bounds["lower", "_x", "d"] = p.d_min + b("d")
        mpc.bounds["lower", "_u", "u"] = -p.u_max
        mpc.bounds["upper", "_u", "u"] = p.u_max

    def make_controller(self) -> DoMpcController:
        model = self.make_model()
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = self.params.horizon
        mpc.settings.n_robust = 0
        mpc.settings.t_step = self.params.dt
        mpc.settings.state_discretization = "discrete"
        mpc.settings.store_full_solution = False
        mpc.settings.supress_ipopt_output()
        self.configure_mpc(mpc, model)
        mpc.setup()
        return DoMpcController(mpc)

    def cost(self, X, U) -> float:
        return float(np.sum((np.asarray(X)[1:, 0] - self.params.setpoint) ** 2))
