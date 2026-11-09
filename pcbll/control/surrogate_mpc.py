"""Multi-stage NMPC on a BLL surrogate with do-mpc, one fixed last layer per branch.

Tree kinds (``TreeConfig.kind``):

* ``"nominal"``: one branch, the posterior mean.
* ``"output_space"``: three branches, ``mean + s r sigma`` on every output at
  every step with s in {0, +1, -1} held over the horizon, sigma the total
  (epistemic + aleatoric) one-step std, no tightening.
* ``"output_space_epistemic"``: the same with the epistemic std only, plus
  the aleatoric tightening.
* ``"weight_space"``: 1 + 2k branches, the posterior mean and ``mean +- r``
  along k whitened directions selected at every MPC step by the
  sensitivity-weighted max-min rule on the constraint gradients of the
  linearised prediction; each branch carries its fixed weight matrix W over
  the whole horizon.

The uncertain parameter of the do-mpc model is ``vec(W)`` (or the sign s);
do-mpc calls ``p_fun`` at every step, so the per-branch values are updated in
place without rebuilding the NLP.  Robust horizon 1: branches split at the
root and keep their parameter afterwards.  The objective is the average over
branches, constraints hold on every branch.

The aleatoric tightening moves the bounds of the constrained states inward by
``tightening * sigma_n`` (sigma_n the learned noise std of that state, in
plant units), either constant over the horizon or as a tube propagated along
the linearised surrogate.

Objective weighting: do-mpc averages the stage and terminal costs over the
branches but sums the input-change penalty and the slack penalties.  Here the
input-change penalty is written into the stage cost through augmented
previous-input states (so it is averaged like the rest, in do-mpc's scaled
input units), and the plant's soft-constraint penalties are scaled by 1 / n.
With identical branches every tree reproduces the nominal controller exactly.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Literal

import casadi as ca
import numpy as np

import do_mpc

from pcbll.bll.casadi_export import feature_expr, predict_expr, variance_expr
from pcbll.plant.base import PlantSpec
from pcbll.surrogate import NarxSurrogate
from pcbll.tree import captured_fraction, constraint_gradients, row_weights, scenario_weights, select_directions

__all__ = ["TreeConfig", "SurrogateMPC"]

Kind = Literal["nominal", "output_space", "output_space_epistemic", "weight_space"]

_KINDS = ("nominal", "output_space", "output_space_epistemic", "weight_space")
_OUTPUT_SPACE = ("output_space", "output_space_epistemic")


_HSL_ENABLED = False


def enable_hsl(directory: str | None = None) -> bool:
    """Put the HSL library on the DLL search path so that IPOPT finds ma27/ma57/ma97.

    Looks at ``directory``, then the environment variable ``HSL_DIR``, then
    ``~/hsl/bin``.  Returns whether a library was found.  Idempotent.
    ``OMP_NUM_THREADS`` is set to 1 before the library is loaded unless the
    environment already sets it or ``PCBLL_KEEP_OMP`` is set: the OpenMP
    thread team costs seconds per factorisation on these small NLPs.
    """
    global _HSL_ENABLED
    if _HSL_ENABLED:
        return True
    candidates = [directory, os.environ.get("HSL_DIR"), os.path.join(os.path.expanduser("~"), "hsl", "bin")]
    for d in candidates:
        if d and os.path.isdir(d) and any(f.lower().startswith(("libhsl", "libcoinhsl")) for f in os.listdir(d)):
            if not os.environ.get("PCBLL_KEEP_OMP"):
                os.environ.setdefault("OMP_NUM_THREADS", "1")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            _HSL_ENABLED = True
            return True
    return False


@dataclasses.dataclass(frozen=True)
class TreeConfig:
    kind: Kind = "weight_space"
    horizon: int = 20
    r: float = 2.0
    #: number of selected directions k (weight_space); scenarios = 1 + 2k
    n_directions: int = 4
    #: aleatoric back-off in units of sigma_n; ignored for 'nominal' (unless tube) and 'output_space'
    tightening: float = 2.0
    #: row weights of the max-min selection: 'sensitivity' = ||g_w|| (rows with a wide posterior
    #: spread first), 'none' = unweighted
    selection_weights: Literal["sensitivity", "none"] = "sensitivity"
    #: trajectory to linearise the constraints around: the previous solution's nominal branch, or a fresh nominal OCP
    linearize: Literal["previous", "nominal_ocp"] = "previous"
    #: keep the branch predictions of every solve
    store_predictions: bool = False
    #: aleatoric tightening per stage: 'constant' = tightening * sigma_n on every stage (through the plant's
    #: bounds); 'tube' = tightening * sqrt(diag Sigma_k) with Sigma_{k+1} = A_k Sigma_k A_k^T + diag(sigma_n^2)
    #: propagated along the linearised surrogate (soft constraints with time-varying parameters)
    tightening_mode: Literal["constant", "tube"] = "constant"
    tube_penalty: float = 1e4
    #: multiplier on the case study's input-change penalty for the surrogate controllers
    input_move_weight: float = 1.0
    #: IPOPT linear solver: 'mumps' (always available) or an HSL solver ('ma27', 'ma57', 'ma97') if the HSL
    #: library is found (see ``enable_hsl``)
    linear_solver: str = "mumps"
    #: further nlpsol options, e.g. {'ipopt.warm_start_init_point': 'yes', 'ipopt.tol': 1e-6}
    nlpsol_opts: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown tree kind {self.kind!r}")
        if self.n_directions < 1 and self.kind == "weight_space":
            raise ValueError("n_directions must be >= 1")

    @property
    def n_scenarios(self) -> int:
        if self.kind == "nominal":
            return 1
        if self.kind in _OUTPUT_SPACE:
            return 3
        return 1 + 2 * self.n_directions

    @property
    def uses_tightening(self) -> bool:
        """The total-sigma output-space tree never tightens; the nominal controller only with the tube."""
        if self.kind == "output_space":
            return False
        if self.kind == "nominal":
            return self.tightening_mode == "tube"
        return True

    @property
    def uses_tube(self) -> bool:
        return self.uses_tightening and self.tightening_mode == "tube" and self.tightening > 0


def _surrogate_model(sur: NarxSurrogate, spec: PlantSpec, kind: str, r: float, tube_states=()):
    """Discrete do-mpc model of the surrogate with the branch parameter as ``_p``.

    Augmented states ``uprev_<input>`` carry the previous input so the
    input-change penalty can live in the (branch-averaged) stage cost.
    ``tube_states`` get a time-varying parameter ``tube_<state>`` for the
    stage-dependent aleatoric back-off.
    """
    model = do_mpc.model.Model("discrete", "SX")
    xs = [model.set_variable("_x", name) for name in spec.state_names]
    uprev = [model.set_variable("_x", f"uprev_{name}") for name in spec.input_names]
    us = [model.set_variable("_u", name) for name in spec.input_names]
    for name in tube_states:
        model.set_variable("_tvp", f"tube_{name}")
    x = ca.vertcat(*xs)
    u = ca.vertcat(*us)
    phi = feature_expr(sur.model, ca.vertcat(x[sur.fb], u))
    if kind in _OUTPUT_SPACE:
        s = model.set_variable("_p", "s")
        y = predict_expr(sur.model, phi)
        var_epi, var_ale = variance_expr(sur.model, phi)
        var = var_epi + var_ale if kind == "output_space" else var_epi
        y = y + s * r * ca.sqrt(var)
    else:
        w = model.set_variable("_p", "W", shape=(sur.model.n_phi * sur.model.n_y, 1))
        y = predict_expr(sur.model, phi, ca.reshape(w, sur.model.n_phi, sur.model.n_y))
    x_next = x + y if sur.target == "delta" else y
    for i, name in enumerate(spec.state_names):
        model.set_rhs(name, x_next[i])
    for name, u_i in zip(spec.input_names, us):
        model.set_rhs(f"uprev_{name}", u_i)
    model.setup()
    return model


class SurrogateMPC:
    """Controller protocol implementation; see the module docstring."""

    def __init__(self, sur: NarxSurrogate, plant, config: TreeConfig = TreeConfig()):
        self.sur = sur
        self.plant = plant
        self.spec: PlantSpec = plant.spec
        self.config = config
        self.post = sur.posterior
        self.noise_std = np.sqrt(self.post.sigma_e2) * sur.model.scaler_y.scale  # plant units, per state

        back_off = None
        if config.uses_tightening and config.tightening > 0 and not config.uses_tube:
            back_off = {c.state: config.tightening * float(self.noise_std[self.spec.state_index(c.state)]) for c in self.spec.constraints}
        self.back_off = back_off or {}
        self.tube_states = tuple(dict.fromkeys(c.state for c in self.spec.constraints)) if config.uses_tube else ()

        self.model = _surrogate_model(sur, self.spec, config.kind, config.r, self.tube_states)
        self.mpc = self._build_mpc(self.model, config.n_scenarios, back_off)
        self._p = self.mpc.get_p_template(config.n_scenarios)
        self.mpc.set_p_fun(lambda t_now: self._p)
        if self.tube_states:
            self._tvp = self.mpc.get_tvp_template()
            self.mpc.set_tvp_fun(lambda t_now: self._tvp)
            self._step_jac = sur.casadi_step_jacobian()
        self._fill_fixed_scenarios()
        self.mpc.setup()

        # for the selection: the rollout Jacobian and (lazily) a nominal OCP to linearise around
        self._nominal: SurrogateMPC | None = None
        self._roll = self._jac = None
        if config.kind == "weight_space":
            self._roll, self._jac = sur.casadi_rollout(config.horizon)
        self._U_prev: np.ndarray | None = None
        self._u_last = np.zeros(self.spec.n_u)
        self.log: list[dict] = []

    def _get_nominal(self) -> "SurrogateMPC":
        if self._nominal is None:
            self._nominal = SurrogateMPC(self.sur, self.plant, dataclasses.replace(self.config, kind="nominal", tightening_mode="constant"))
        return self._nominal

    # ---------------------------------------------------------------- build
    def _build_mpc(self, model, n_scenarios: int, back_off):
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = self.config.horizon
        mpc.settings.n_robust = 1 if n_scenarios > 1 else 0
        mpc.settings.t_step = self.spec.t_step
        mpc.settings.state_discretization = "discrete"
        mpc.settings.store_full_solution = True
        mpc.settings.supress_ipopt_output()
        opts = dict(mpc.settings.nlpsol_opts)
        if self.config.linear_solver != "mumps":
            enable_hsl()
            opts["ipopt.linear_solver"] = self.config.linear_solver
        opts.update(self.config.nlpsol_opts)
        mpc.settings.nlpsol_opts = opts
        self.plant.configure_mpc(mpc, model, back_off, 1.0 / n_scenarios)
        # move the input-change penalty into the stage cost (see the module docstring)
        penalty = 0
        for name in self.spec.input_names:
            factor = float(mpc.rterm_factor[name]) * self.config.input_move_weight
            scale = float(mpc.scaling["_u", name])
            penalty = penalty + factor * ((model.u[name] - model.x[f"uprev_{name}"]) / scale) ** 2
        mpc.set_objective(mterm=mpc.mterm, lterm=mpc.lterm + penalty)
        mpc.set_rterm(**{name: 0.0 for name in self.spec.input_names})
        # stage-dependent aleatoric tube: soft constraints  +-x + tube_x <= +-bound  with the back-off as a tvp
        for c in self.spec.constraints:
            if c.state in self.tube_states:
                sign = 1.0 if c.kind == "upper" else -1.0
                mpc.set_nl_cons(f"tube_{c.state}_{c.kind}", sign * model.x[c.state] + model.tvp[f"tube_{c.state}"], ub=sign * c.bound,
                                soft_constraint=True, penalty_term_cons=self.config.tube_penalty / n_scenarios)
        return mpc

    def _update_tube(self, x: np.ndarray, U_lin: np.ndarray) -> np.ndarray:
        """Propagate the noise covariance along the linearised surrogate and write the back-offs into the tvp."""
        N = self.config.horizon
        n_x = self.spec.n_x
        Sigma = np.zeros((n_x, n_x))
        Q = np.diag(self.noise_std**2)
        backoff = np.zeros((N + 1, n_x))
        xk = np.asarray(x, dtype=float).reshape(-1)
        W = self.post.mean
        for k in range(N):
            A = np.array(self._step_jac(xk, U_lin[k], W))
            Sigma = A @ Sigma @ A.T + Q
            backoff[k + 1] = self.config.tightening * np.sqrt(np.maximum(np.diag(Sigma), 0.0))
            xk = self.sur.next_with_weights(xk, U_lin[k], W)
        for name in self.tube_states:
            i = self.spec.state_index(name)
            for k in range(N + 1):
                self._tvp["_tvp", k, f"tube_{name}"] = backoff[k, i]
        return backoff

    def _augment(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(x, dtype=float).reshape(-1), self._u_last]).reshape(-1, 1)

    def _fill_fixed_scenarios(self) -> None:
        kind = self.config.kind
        if kind == "nominal":
            self._set_weights(self.post.mean[None])
        elif kind in _OUTPUT_SPACE:
            for i, s in enumerate((0.0, 1.0, -1.0)):
                self._p["_p", i, "s"] = s
        elif kind == "weight_space":  # replaced at every step by the selection
            self._set_weights(np.repeat(self.post.mean[None], self.config.n_scenarios, axis=0))

    def _set_weights(self, Ws: np.ndarray) -> None:
        if Ws.shape[0] != self.config.n_scenarios:
            raise ValueError("wrong number of scenarios")
        for i in range(Ws.shape[0]):
            self._p["_p", i, "W"] = Ws[i].reshape(-1, order="F")

    # ------------------------------------------------------------ interface
    def reset(self, x0: np.ndarray) -> None:
        self.mpc.reset_history()
        self._u_last = np.zeros(self.spec.n_u)  # do-mpc's own convention for the first input change
        self.mpc.x0 = self._augment(x0)
        self.mpc.set_initial_guess()
        if self.config.kind == "weight_space" or self.tube_states:
            self._get_nominal().reset(x0)
        self._U_prev = None
        self.log = []

    def predicted_inputs(self, scenario: int = 0) -> np.ndarray:
        """Input sequence (N, n_u) of one branch of the last solution, in plant units."""
        N = self.config.horizon
        U = np.empty((N, self.spec.n_u))
        for k in range(N):
            s = scenario if k > 0 else 0
            U[k] = np.array(self.mpc.opt_x_num["_u", k, s] * self.mpc._u_scaling).reshape(-1)
        return U

    def _linearisation_inputs(self, x: np.ndarray) -> np.ndarray:
        if self.config.linearize == "nominal_ocp" or self._U_prev is None:
            nominal = self._get_nominal()
            nominal.step(x)
            return nominal.predicted_inputs()
        return np.vstack([self._U_prev[1:], self._U_prev[-1:]])

    def _select(self, x: np.ndarray, U_lin: np.ndarray | None = None) -> dict:
        if U_lin is None:
            U_lin = self._linearisation_inputs(x)
        G_w, rows, _ = constraint_gradients(self.sur, x, U_lin, self.spec.constraints, self.spec, self._roll, self._jac)
        sens = np.array([row.sensitivity for row in rows])
        weights = row_weights(rows, self.config.selection_weights)
        sel = select_directions(self.post, G_w, method="maxmin", cos_threshold=1.0, n_directions=self.config.n_directions, weights=weights)
        D = sel.directions
        if D.shape[0] < self.config.n_directions:  # fewer distinct rows than the budget: pad with copies
            D = np.concatenate([D, np.repeat(D[-1:], self.config.n_directions - D.shape[0], axis=0)])
        frac = captured_fraction(G_w, D)
        relevant = sens >= 0.1 * sens.max() if sens.max() > 0 else np.ones_like(sens, dtype=bool)
        self._set_weights(scenario_weights(self.post, D, self.config.r)[0])
        return {"captured_min": float(frac.min()), "captured_min_relevant": float(frac[relevant].min()),
                "captured_weighted_mean": float(np.sum(frac * sens) / max(sens.sum(), 1e-300)), "max_sensitivity": float(sens.max())}

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        info: dict = {"kind": self.config.kind, "n_scenarios": self.config.n_scenarios}
        t = time.perf_counter()
        U_lin = None
        if self.config.kind == "weight_space" or self.tube_states:
            U_lin = self._linearisation_inputs(x)
        if self.config.kind == "weight_space":
            info.update(self._select(x, U_lin))
        if self.tube_states:
            backoff = self._update_tube(x, U_lin)
            info["tube_backoff_final"] = {name: float(backoff[-1, self.spec.state_index(name)]) for name in self.tube_states}
        info["t_select"] = time.perf_counter() - t
        t = time.perf_counter()
        u = np.asarray(self.mpc.make_step(self._augment(x)), dtype=float).reshape(-1)
        info["t_solve"] = time.perf_counter() - t
        self._u_last = u.copy()
        info["success"] = bool(self.mpc.solver_stats.get("success", True)) if hasattr(self.mpc, "solver_stats") else True
        info["iterations"] = int(self.mpc.solver_stats.get("iter_count", -1)) if hasattr(self.mpc, "solver_stats") else -1
        self._U_prev = self.predicted_inputs()
        self.last_prediction = self.predicted_states() if self.config.store_predictions else None
        self.log.append(info)
        return u

    def predicted_states(self) -> np.ndarray:
        """Branch predictions of the last solve, shape (n_scenarios, N+1, n_x), plant units."""
        N, n_s = self.config.horizon, self.config.n_scenarios
        X = np.empty((n_s, N + 1, self.spec.n_x))
        for i, name in enumerate(self.spec.state_names):
            pred = np.array(self.mpc.data.prediction(("_x", name)))  # (1, N+1, n_scenarios)
            X[:, :, i] = pred[0].T
        return X
