"""A BLL model of the one-step map x_{k+1} = f(x_k, u_k), with a symbolic rollout."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from pcbll.bll.model import BLLModel, Prediction
from pcbll.data import NarxDataset

__all__ = ["NarxSurrogate", "train_surrogate"]


@dataclasses.dataclass(frozen=True)
class NarxSurrogate:
    """``model`` maps [x[feedback], u] to x_next (``target='next'``) or to x_next - x (``target='delta'``).

    All methods take and return plant units.  Weight matrices W are in
    posterior units (see :mod:`pcbll.bll`).
    """

    model: BLLModel
    n_x: int
    n_u: int
    target: str = "next"
    #: indices of the states fed back as inputs (default all)
    feedback: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.target not in ("next", "delta"):
            raise ValueError("target must be 'next' or 'delta'")
        if not self.feedback:
            object.__setattr__(self, "feedback", tuple(range(self.n_x)))
        object.__setattr__(self, "feedback", tuple(int(i) for i in self.feedback))
        if self.model.n_x != len(self.feedback) + self.n_u or self.model.n_y != self.n_x:
            raise ValueError("model dimensions do not match n_x, n_u, feedback")

    @classmethod
    def from_dataset(cls, model: BLLModel, data: NarxDataset) -> "NarxSurrogate":
        return cls(model, data.n_x, data.n_u, data.target, data.feedback)

    @property
    def fb(self) -> list[int]:
        return list(self.feedback)

    @property
    def posterior(self):
        return self.model.posterior

    # ------------------------------------------------------------- one step
    def inputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        if x.ndim == 1:
            return np.concatenate([x[self.fb], u])[None, :]
        return np.concatenate([x[:, self.fb], np.broadcast_to(u, (x.shape[0], u.size)) if u.ndim == 1 else u], axis=1)

    def _to_state(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y + np.asarray(x, dtype=float) if self.target == "delta" else y

    def predict(self, x: np.ndarray, u: np.ndarray) -> Prediction:
        """Predictive distribution of x_next for one (x, u); mean in state units, variances unchanged by the target choice."""
        p = self.model.predict(self.inputs(x, u))
        return Prediction(self._to_state(x, p.mean[0]), p.var_epistemic[0], p.var_aleatoric[0])

    def next_mean(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.predict(x, u).mean

    def next_with_weights(self, x: np.ndarray, u: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self._to_state(x, self.model.predict_with_weights(self.inputs(x, u), W)[0])

    def next_batch(self, X: np.ndarray, u: np.ndarray, Ws: np.ndarray) -> np.ndarray:
        """One step for n states X (n, n_x) under n weight matrices Ws (n, n_phi, n_y), same input u."""
        Phi = self.model.phi(self.inputs(X, u))
        y = self.model.scaler_y.inverse(np.einsum("np,npk->nk", Phi, Ws))
        return self._to_state(X, y)

    def phi(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.model.phi(self.inputs(x, u))[0]

    # ------------------------------------------------------------ symbolic
    def casadi_rollout(self, n_steps: int):
        """``roll(x0, U, W) -> X`` with U (n_u, N) and X (n_x, N+1) columns x_0 ... x_N, plus ``jac(x0, U, W)``
        = d vec(X) / d vec(W) at that W, shape ((N+1) n_x, n_phi n_y), column-major in both."""
        import casadi as ca

        from pcbll.bll.casadi_export import feature_expr, predict_expr

        x0 = ca.SX.sym("x0", self.n_x)
        U = ca.SX.sym("U", self.n_u, n_steps)
        W = ca.SX.sym("W", self.model.n_phi, self.model.n_y)
        cols = [x0]
        x = x0
        for k in range(n_steps):
            y = predict_expr(self.model, feature_expr(self.model, ca.vertcat(x[self.fb], U[:, k])), W)
            x = x + y if self.target == "delta" else y
            cols.append(x)
        X = ca.horzcat(*cols)
        roll = ca.Function("roll", [x0, U, W], [X], ["x0", "U", "W"], ["X"])
        jac = ca.Function("rollout_jacobian", [x0, U, W], [ca.jacobian(ca.vec(X), ca.vec(W))], ["x0", "U", "W"], ["J"])
        return roll, jac

    def casadi_step_jacobian(self):
        """``A(x, u, W) -> d x_next / d x`` (n_x, n_x) of one surrogate step (zero columns for non-feedback states)."""
        import casadi as ca

        from pcbll.bll.casadi_export import feature_expr, predict_expr

        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)
        W = ca.SX.sym("W", self.model.n_phi, self.model.n_y)
        y = predict_expr(self.model, feature_expr(self.model, ca.vertcat(x[self.fb], u)), W)
        x_next = x + y if self.target == "delta" else y
        return ca.Function("step_jacobian", [x, u, W], [ca.jacobian(x_next, x)], ["x", "u", "W"], ["A"])

    # ------------------------------------------------------------------- io
    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        path = self.model.save(path)
        meta = pathlib.Path(path).with_suffix(".narx.npz")
        np.savez(meta, n_x=self.n_x, n_u=self.n_u, target=self.target, feedback=np.array(self.feedback))
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "NarxSurrogate":
        path = pathlib.Path(path).with_suffix(".npz")
        with np.load(path.with_suffix(".narx.npz")) as f:
            fb = tuple(int(i) for i in f["feedback"]) if "feedback" in f else ()
            return cls(BLLModel.load(path), int(f["n_x"]), int(f["n_u"]), str(f["target"]), fb)


def train_surrogate(data: NarxDataset, config=None, val: NarxDataset | None = None, verbose: bool = False):
    """Train a BLL on a NARX dataset; returns (surrogate, history).  Imports torch here only."""
    from pcbll.bll.training import TrainingConfig, fit

    config = TrainingConfig() if config is None else config
    model, history = fit(data.X, data.Y, config, val=None if val is None else (val.X, val.Y), verbose=verbose)
    return NarxSurrogate.from_dataset(model, data), history
