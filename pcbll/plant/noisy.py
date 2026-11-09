"""A plant with additive Gaussian process noise on the independent states.

    x_{k+1} = f(x_k, u_k) + w_k,   w_k ~ N(0, diag(sigma_w^2))

``sigma_w`` is ``level`` times the per-state standard deviation of the one-step
change along the base plant's noise-free reference run (nominal controller
from ``spec.x0``).  Noise enters the states in ``spec.feedback`` only; the
base plant's ``consistent`` hook recomputes derived states.  Registry name:
``<base>_pn<level>``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pcbll.plant.base import Plant

__all__ = ["ProcessNoisePlant"]


def _reference_states(base: Plant) -> np.ndarray:
    """States (T+1, n_x) of the nominal controller's closed loop from ``spec.x0``."""
    from pcbll.cache import reference_run

    return reference_run(base, verbose=False).x


class ProcessNoisePlant:
    def __init__(self, base: Plant, level: float, seed: int = 12345):
        self.base = base
        self.level = float(level)
        dx_std = np.std(np.diff(_reference_states(base), axis=0), axis=0)
        mask = np.zeros(base.spec.n_x)
        mask[list(base.spec.feedback_index)] = 1.0
        self.noise_std = self.level * dx_std * mask
        self.rng = np.random.default_rng(seed)
        self.spec = dataclasses.replace(base.spec, name=f"{base.spec.name}_pn{self.level:g}")

    def consistent(self, x):
        fix = getattr(self.base, "consistent", None)
        return x if fix is None else fix(x)

    def step(self, x, u):
        x_next = self.base.step(x, u) + self.noise_std * self.rng.standard_normal(self.spec.n_x)
        return self.consistent(x_next)

    def sample_x0(self, rng):
        return self.base.sample_x0(rng)

    def make_controller(self):
        return self.base.make_controller()

    def configure_mpc(self, mpc, model, back_off=None, scenario_weight: float = 1.0):
        return self.base.configure_mpc(mpc, model, back_off, scenario_weight)

    def cost(self, X, U):
        return self.base.cost(X, U)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
