"""Two-dimensional compressible Euler solver using nodal DG methods in JAX.

Importing this package enables JAX 64-bit mode (ADR 0006).
"""

import jax

jax.config.update("jax_enable_x64", True)

from .boundary import FixedState, SlipWall  # noqa: E402
from .cases import SodShockTube, sod_shock_tube  # noqa: E402
from .fluxes import INTERFACE_FLUXES, get_interface_flux  # noqa: E402
from .limiters import LIMITER_FUNCTIONS, get_limiter_function  # noqa: E402
from .mesh import TriangleMesh, rectangle_mesh  # noqa: E402
from .solver import EulerSolver, RunResult, SolverConfig, SolverError  # noqa: E402

__all__ = [
    "INTERFACE_FLUXES",
    "LIMITER_FUNCTIONS",
    "EulerSolver",
    "FixedState",
    "RunResult",
    "SlipWall",
    "SodShockTube",
    "SolverConfig",
    "SolverError",
    "TriangleMesh",
    "get_interface_flux",
    "get_limiter_function",
    "rectangle_mesh",
    "sod_shock_tube",
]
