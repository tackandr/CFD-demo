"""Boundary conditions applied through the numerical flux (ADR 0007).

A boundary condition only supplies the exterior state of the selected
interface flux; it never modifies interior values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLIP_WALL = 1
FIXED_STATE = 2


@dataclass(frozen=True)
class SlipWall:
    """Inviscid wall: the exterior state mirrors the interior trace, with the
    same density, total energy, and tangential momentum and reversed normal
    momentum."""

    kind: int = SLIP_WALL


@dataclass(frozen=True)
class FixedState:
    """Prescribed constant exterior state in conserved variables
    ``(rho, rho*u, rho*v, E)``."""

    state: tuple[float, float, float, float]
    kind: int = FIXED_STATE

    def __post_init__(self) -> None:
        state = tuple(float(value) for value in self.state)
        if len(state) != 4 or not np.all(np.isfinite(state)):
            raise ValueError("FixedState needs four finite conserved values.")
        object.__setattr__(self, "state", state)

    @classmethod
    def from_primitive(cls, rho: float, u: float, v: float, p: float, gamma: float) -> FixedState:
        energy = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
        return cls((rho, rho * u, rho * v, energy))


BoundaryCondition = SlipWall | FixedState
