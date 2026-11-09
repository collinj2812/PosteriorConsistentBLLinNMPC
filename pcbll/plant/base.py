"""Plant and controller protocols, independent of do-mpc.

A plant is a discrete-time map ``x_next = step(x, u)`` with a spec describing
names, bounds and the reference initial state.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Constraint", "Controller", "Plant", "PlantSpec"]


@dataclasses.dataclass(frozen=True)
class Constraint:
    """A bound on one state that the controller of the case study enforces."""

    state: str
    kind: str  #: 'upper' or 'lower'
    bound: float

    def __post_init__(self) -> None:
        if self.kind not in ("upper", "lower"):
            raise ValueError("kind must be 'upper' or 'lower'")

    def margin(self, value: np.ndarray) -> np.ndarray:
        """Signed distance to the bound, positive when satisfied."""
        return self.bound - value if self.kind == "upper" else value - self.bound


@dataclasses.dataclass(frozen=True)
class PlantSpec:
    name: str
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    x0: np.ndarray
    u_lower: np.ndarray
    u_upper: np.ndarray
    t_step: float
    #: length of one reference run in steps
    episode_steps: int
    constraints: tuple[Constraint, ...] = ()
    #: absolute standard deviation per state used to jitter x0 between data runs
    x0_jitter: np.ndarray | None = None
    #: states the surrogate takes as inputs (None = all).  States that are functions of other
    #: states are predicted but not fed back, so a perturbed last layer cannot push the
    #: fed-back state off the manifold.
    feedback: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x0", np.asarray(self.x0, dtype=float).reshape(-1))
        object.__setattr__(self, "u_lower", np.asarray(self.u_lower, dtype=float).reshape(-1))
        object.__setattr__(self, "u_upper", np.asarray(self.u_upper, dtype=float).reshape(-1))
        if self.x0.size != len(self.state_names):
            raise ValueError("x0 and state_names disagree")
        if self.u_lower.shape != (len(self.input_names),) or self.u_upper.shape != self.u_lower.shape:
            raise ValueError("input bounds and input_names disagree")
        if np.any(self.u_upper <= self.u_lower):
            raise ValueError("u_upper must exceed u_lower")
        jitter = self.x0_jitter
        if jitter is not None:
            jitter = np.asarray(jitter, dtype=float).reshape(-1)
            if jitter.shape != self.x0.shape:
                raise ValueError("x0_jitter must have one entry per state")
            object.__setattr__(self, "x0_jitter", jitter)
        for c in self.constraints:
            if c.state not in self.state_names:
                raise ValueError(f"constraint on unknown state {c.state!r}")
        if self.feedback is not None:
            for name in self.feedback:
                if name not in self.state_names:
                    raise ValueError(f"feedback state {name!r} is not a state")

    @property
    def n_x(self) -> int:
        return len(self.state_names)

    @property
    def n_u(self) -> int:
        return len(self.input_names)

    def state_index(self, name: str) -> int:
        return self.state_names.index(name)

    @property
    def feedback_index(self) -> tuple[int, ...]:
        """Indices of the states the surrogate takes as inputs."""
        names = self.state_names if self.feedback is None else self.feedback
        return tuple(self.state_names.index(n) for n in names)

    def clip_input(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u, dtype=float).reshape(-1), self.u_lower, self.u_upper)


@runtime_checkable
class Controller(Protocol):
    def reset(self, x0: np.ndarray) -> None: ...

    def step(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Plant(Protocol):
    spec: PlantSpec

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...

    def sample_x0(self, rng: np.random.Generator) -> np.ndarray: ...

    def make_controller(self) -> Controller: ...
