"""Verification case setups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import FixedState, SlipWall
from .mesh import rectangle_mesh
from .solver import EulerSolver, SolverConfig
from .time_integration import DEFAULT_CFL

# Two-dimensional Sod shock tube (ADR 0005).
SOD_GAMMA = 1.4
SOD_LEFT = (1.0, 0.0, 0.0, 1.0)  # (rho, u, v, p) for x < 0.5
SOD_RIGHT = (0.125, 0.0, 0.0, 0.1)  # (rho, u, v, p) for x > 0.5
SOD_DISCONTINUITY = 0.5
SOD_X_RANGE = (0.0, 1.0)
SOD_Y_RANGE = (0.0, 0.1)
SOD_FINAL_TIME = 0.2


@dataclass(frozen=True, eq=False)
class SodShockTube:
    solver: EulerSolver
    initial_state: np.ndarray
    final_time: float


def sod_shock_tube(nx: int = 200, ny: int = 20, *, flux: str, limiter: str, cfl: float = DEFAULT_CFL) -> SodShockTube:
    """The ADR 0005 Sod shock tube on ``nx * ny`` rectangles split into
    triangles, with slip walls at the top and bottom and the initial left and
    right states as fixed exterior states at the ends."""
    mesh = rectangle_mesh(nx, ny, SOD_X_RANGE, SOD_Y_RANGE)
    left = FixedState.from_primitive(*SOD_LEFT, gamma=SOD_GAMMA)
    right = FixedState.from_primitive(*SOD_RIGHT, gamma=SOD_GAMMA)
    boundary_conditions = {"left": left, "right": right, "bottom": SlipWall(), "top": SlipWall()}
    config = SolverConfig(flux=flux, limiter=limiter, gamma=SOD_GAMMA, cfl=cfl)
    solver = EulerSolver(mesh, boundary_conditions, config)
    on_left = solver.geometry.centroid[:, 0] < SOD_DISCONTINUITY
    element_state = np.where(on_left[:, None], np.array(left.state), np.array(right.state))
    initial_state = np.broadcast_to(element_state[None], solver.state_shape).copy()
    return SodShockTube(solver=solver, initial_state=initial_state, final_time=SOD_FINAL_TIME)
