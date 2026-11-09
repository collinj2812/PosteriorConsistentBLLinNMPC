"""The CSTR case study as a do-mpc plant with its nominal controller.

Physical parameters of Klatt and Engell (1998); bounds, scaling, input-change
penalties and the 18 s sampling time of the do-mpc CSTR example.  The
objective is the production rate of B.  ``step(x, u)`` integrates the ODE
over one sampling interval with CVODES; the controller is the nominal MPC
(``n_robust = 0``).
"""

from __future__ import annotations

import numpy as np

import do_mpc
from casadi import exp

from pcbll.plant.base import Constraint, PlantSpec

__all__ = ["DoMpcPlant", "DoMpcController", "cstr_econ", "PLANTS"]


class DoMpcController:
    """A configured do-mpc MPC behind the Controller protocol."""

    def __init__(self, mpc: do_mpc.controller.MPC):
        self.mpc = mpc

    def reset(self, x0: np.ndarray) -> None:
        self.mpc.reset_history()
        self.mpc.x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.mpc.set_initial_guess()

    def step(self, x: np.ndarray) -> np.ndarray:
        u = self.mpc.make_step(np.asarray(x, dtype=float).reshape(-1, 1))
        return np.asarray(u, dtype=float).reshape(-1)


def _backoff(back_off: dict | None):
    back_off = {} if back_off is None else dict(back_off)

    def b(name: str) -> float:
        return float(back_off.get(name, 0.0))

    return b


class DoMpcPlant:
    """A do-mpc plant.

    ``configure_mpc(mpc, model, back_off, scenario_weight)`` applies the case
    study's objective, penalties, bounds and scaling to any do-mpc controller
    whose model carries the same state and input names, so the surrogate
    controllers share the tuning of the nominal one.  ``cost(X, U)`` is the
    closed-loop performance measure.
    """

    def __init__(self, spec: PlantSpec, model, p_nominal: dict, make_mpc, configure=None, cost=None, x0_fix=None, abstol=1e-8, reltol=1e-8):
        self.spec = spec
        self.model = model
        self.p_nominal = dict(p_nominal)
        self._make_mpc = make_mpc
        self.configure_mpc = configure
        self._cost = cost
        self._x0_fix = x0_fix
        sim = do_mpc.simulator.Simulator(model)
        sim.set_param(integration_tool="cvodes", abstol=abstol, reltol=reltol, t_step=spec.t_step)
        p_num = sim.get_p_template()
        for k, v in self.p_nominal.items():
            p_num[k] = v
        sim.set_p_fun(lambda t_now: p_num)
        sim.setup()
        self.simulator = sim

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        self.simulator.x0 = np.asarray(x, dtype=float).reshape(-1, 1)
        x_next = self.simulator.make_step(np.asarray(u, dtype=float).reshape(-1, 1))
        # the simulator's history is never read; keep it from growing
        self.simulator.reset_history()
        return np.asarray(x_next, dtype=float).reshape(-1)

    def sample_x0(self, rng: np.random.Generator) -> np.ndarray:
        x0 = self.spec.x0.copy()
        if self.spec.x0_jitter is not None:
            x0 = x0 + self.spec.x0_jitter * rng.standard_normal(x0.shape)
        if self._x0_fix is not None:
            x0 = self._x0_fix(x0)
        return x0

    def make_controller(self) -> DoMpcController:
        return DoMpcController(self._make_mpc(self.model))

    def consistent(self, x: np.ndarray) -> np.ndarray:
        """Recompute derived states from the independent ones (and keep states physical)."""
        return x if self._x0_fix is None else self._x0_fix(np.asarray(x, dtype=float).copy())

    def cost(self, X: np.ndarray, U: np.ndarray) -> float:
        if self._cost is None:
            raise NotImplementedError("no closed-loop cost defined for this plant")
        return self._cost(np.asarray(X, dtype=float), np.asarray(U, dtype=float))


# ----------------------------------------------------------------------------
# CSTR: states C_a, C_b [mol/l], T_R, T_K [degC]; inputs F [1/h], Q_dot [kW]
# ----------------------------------------------------------------------------


def _cstr_model():
    K0_ab, K0_bc, K0_ad = 1.287e12, 1.287e12, 9.043e9
    E_A_ab, E_A_bc, E_A_ad = 9758.3, 9758.3, 8560.0
    H_R_ab, H_R_bc, H_R_ad = 4.2, -11.0, -41.85
    Rou, Cp, Cp_k, A_R, V_R, m_k, T_in, K_w = 0.9342, 3.01, 2.0, 0.215, 10.01, 5.0, 130.0, 4032.0
    C_A0 = (5.7 + 4.5) / 2.0
    model = do_mpc.model.Model("continuous", "SX")
    C_a = model.set_variable("_x", "C_a")
    C_b = model.set_variable("_x", "C_b")
    T_R = model.set_variable("_x", "T_R")
    T_K = model.set_variable("_x", "T_K")
    F = model.set_variable("_u", "F")
    Q_dot = model.set_variable("_u", "Q_dot")
    alpha = model.set_variable("_p", "alpha")
    beta = model.set_variable("_p", "beta")
    T_dif = T_R - T_K
    K_1 = beta * K0_ab * exp((-E_A_ab) / ((T_R + 273.15)))
    K_2 = K0_bc * exp((-E_A_bc) / ((T_R + 273.15)))
    K_3 = K0_ad * exp((-alpha * E_A_ad) / ((T_R + 273.15)))
    model.set_rhs("C_a", F * (C_A0 - C_a) - K_1 * C_a - K_3 * (C_a**2))
    model.set_rhs("C_b", -F * C_b + K_1 * C_a - K_2 * C_b)
    model.set_rhs("T_R", ((K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a**2) * H_R_ad) / (-Rou * Cp)) + F * (T_in - T_R) + (((K_w * A_R) * (-T_dif)) / (Rou * Cp * V_R)))
    model.set_rhs("T_K", (Q_dot + K_w * A_R * (T_dif)) / (m_k * Cp_k))
    model.setup()
    return model


def _cstr_configure(mpc, model, back_off: dict | None = None, scenario_weight: float = 1.0) -> None:
    """Scaling, bounds and the soft bound on T_R of the do-mpc CSTR example, bounds moved inward by ``back_off``."""
    b = _backoff(back_off)
    mpc.scaling["_x", "T_R"] = 100
    mpc.scaling["_x", "T_K"] = 100
    mpc.scaling["_u", "Q_dot"] = 2000
    mpc.scaling["_u", "F"] = 100
    mpc.set_objective(mterm=(model.x["C_b"] - 0.6) ** 2, lterm=(model.x["C_b"] - 0.6) ** 2)
    mpc.set_rterm(F=0.1, Q_dot=1e-3)
    mpc.bounds["lower", "_x", "C_a"] = 0.1 + b("C_a")
    mpc.bounds["lower", "_x", "C_b"] = 0.1 + b("C_b")
    mpc.bounds["lower", "_x", "T_R"] = 50 + b("T_R")
    mpc.bounds["lower", "_x", "T_K"] = 50 + b("T_K")
    mpc.bounds["upper", "_x", "C_a"] = 2 - b("C_a")
    mpc.bounds["upper", "_x", "C_b"] = 2 - b("C_b")
    mpc.bounds["upper", "_x", "T_K"] = 140 - b("T_K")
    mpc.bounds["lower", "_u", "F"] = 5
    mpc.bounds["lower", "_u", "Q_dot"] = -8500
    mpc.bounds["upper", "_u", "F"] = 100
    mpc.bounds["upper", "_u", "Q_dot"] = 0.0
    mpc.set_nl_cons("T_R", model.x["T_R"], ub=140 - b("T_R"), soft_constraint=True, penalty_term_cons=1e2 * scenario_weight)


def _cstr_econ_configure(mpc, model, back_off: dict | None = None, scenario_weight: float = 1.0, q_min: float = -8500.0) -> None:
    """Economic objective: maximise the production rate of B, F C_b (scaled by 1/100); ``q_min`` bounds the cooling duty."""
    _cstr_configure(mpc, model, back_off, scenario_weight)
    prod = -model.u["F"] * model.x["C_b"] / 100.0
    mpc.set_objective(mterm=0 * model.x["C_b"], lterm=prod)
    mpc.set_rterm(F=0.1, Q_dot=1e-3)
    mpc.bounds["lower", "_u", "Q_dot"] = q_min


def _cstr_econ_mpc(model, q_min: float = -8500.0):
    mpc = do_mpc.controller.MPC(model)
    mpc.settings.n_horizon = 20
    mpc.settings.n_robust = 0
    mpc.settings.t_step = 0.005
    mpc.settings.state_discretization = "collocation"
    mpc.settings.collocation_type = "radau"
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 1
    mpc.settings.store_full_solution = False
    mpc.settings.supress_ipopt_output()
    _cstr_econ_configure(mpc, model, q_min=q_min)
    mpc.set_uncertainty_values(alpha=np.array([1.0]), beta=np.array([1.0]))
    mpc.setup()
    return mpc


def _cstr_econ_cost(X: np.ndarray, U: np.ndarray) -> float:
    """Minus the amount of B produced over the run, sum of F C_b t_step."""
    return float(-np.sum(U[:, 0] * X[1:, 1]) * 0.005)


def cstr_econ(q_min: float = -8500.0) -> DoMpcPlant:
    """The CSTR with the production-rate objective; soft bound on T_R and hard bound on T_K at 140 degC."""
    name = "cstr_econ" if q_min == -8500.0 else f"cstr_econ_q{abs(q_min):g}"
    spec = PlantSpec(
        name=name,
        state_names=("C_a", "C_b", "T_R", "T_K"),
        input_names=("F", "Q_dot"),
        x0=np.array([0.8, 0.5, 134.14, 130.0]),
        u_lower=np.array([5.0, q_min]),
        u_upper=np.array([100.0, 0.0]),
        t_step=0.005,
        episode_steps=50,
        constraints=(Constraint("T_R", "upper", 140.0), Constraint("T_K", "upper", 140.0)),
        x0_jitter=np.array([0.1, 0.1, 3.0, 3.0]),
    )
    return DoMpcPlant(spec, _cstr_model(), {"alpha": 1.0, "beta": 1.0}, lambda model: _cstr_econ_mpc(model, q_min),
                      lambda mpc, model, back_off=None, scenario_weight=1.0: _cstr_econ_configure(mpc, model, back_off, scenario_weight, q_min),
                      _cstr_econ_cost, x0_fix=lambda x: np.maximum(x, [0.05, 0.05, 50.0, 50.0]))


PLANTS = {"cstr_econ": cstr_econ}
