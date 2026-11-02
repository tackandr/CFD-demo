"""Euler solver assembling the accepted solver decisions (ADRs 0006-0010)."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from .boundary import BoundaryCondition
from .dg import build_dg_operators, cell_averages, dg_rhs, domain_integral, resolve_boundary_conditions
from .euler import NUM_VARIABLES, pressure
from .fluxes import get_interface_flux
from .limiters import get_limiter_function
from .limiting import apply_limiters, build_limiter_operators
from .mesh import TriangleMesh, build_connectivity, element_geometry, node_coordinates
from .reference_element import DEGREE, build_reference_triangle
from .time_integration import (
    DEFAULT_CFL,
    MAX_REJECTIONS,
    StepReport,
    ssp_rk3_step,
    stable_time_step,
    time_step_geometry,
)


class SolverError(RuntimeError):
    """Raised when the solver cannot advance an admissible solution."""


@dataclass(frozen=True)
class SolverConfig:
    """Solver configuration.

    ``flux`` and ``limiter`` are stable ADR 0004 names and have no defaults
    (ADR 0010). ``gamma`` is the ratio of specific heats (ADR 0008).
    """

    flux: str
    limiter: str
    gamma: float
    cfl: float = DEFAULT_CFL
    max_rejections: int = MAX_REJECTIONS

    def __post_init__(self) -> None:
        get_interface_flux(self.flux)
        get_limiter_function(self.limiter)
        if not self.gamma > 1.0:
            raise ValueError("gamma must be greater than one.")
        if not self.cfl > 0.0:
            raise ValueError("cfl must be positive.")
        if self.max_rejections < 0:
            raise ValueError("max_rejections must be nonnegative.")


@dataclass(frozen=True, eq=False)
class RunResult:
    """Outcome of :meth:`EulerSolver.run`.

    Attributes:
        state: Final nodal solution, shape (Np, K, 4).
        time: Final time reached.
        steps: Accepted time steps.
        rejected_steps: Rejected step attempts (ADR 0009).
        boundary_flux_integral: Time-integrated net outward boundary flux per
            conserved variable, so that integral(q(T)) - integral(q(0)) +
            boundary_flux_integral = 0 up to round-off.
        min_density, min_pressure: Minimum nodal values over the initial state
            and every limited Runge-Kutta stage of accepted steps.
        max_troubled_cells: Largest troubled-cell count in any stage.
        wall_time: Elapsed seconds, including compilation.
    """

    state: np.ndarray
    time: float
    steps: int
    rejected_steps: int
    boundary_flux_integral: np.ndarray
    min_density: float
    min_pressure: float
    max_troubled_cells: int
    wall_time: float


class EulerSolver:
    """Degree-four nodal DG solver for the two-dimensional Euler equations."""

    def __init__(
        self,
        mesh: TriangleMesh,
        boundary_conditions: Mapping[str, BoundaryCondition],
        config: SolverConfig,
    ) -> None:
        self.mesh = mesh
        self.config = config
        self.boundary_conditions = MappingProxyType(dict(boundary_conditions))
        self.reference = build_reference_triangle(DEGREE)
        self.connectivity = build_connectivity(mesh)
        self.geometry = element_geometry(mesh)
        kind, state = resolve_boundary_conditions(self.connectivity, self.boundary_conditions)
        self.operators = build_dg_operators(
            mesh, self.connectivity, self.geometry, self.reference, kind, state
        )
        self.limiter_operators = build_limiter_operators(
            self.connectivity, self.geometry, self.reference, kind, state
        )
        self._dt_geometry = jnp.asarray(time_step_geometry(self.geometry, self.reference.degree))

        flux = get_interface_flux(config.flux)
        phi = get_limiter_function(config.limiter)
        limit_slopes = config.limiter != "unlimited"
        self._rhs = jax.jit(partial(dg_rhs, flux=flux, gamma=config.gamma))
        self._limit = jax.jit(
            partial(apply_limiters, phi=phi, gamma=config.gamma, limit_slopes=limit_slopes)
        )
        self._time_step = jax.jit(partial(stable_time_step, gamma=config.gamma, cfl=config.cfl))
        self._step = jax.jit(
            partial(ssp_rk3_step, flux=flux, phi=phi, gamma=config.gamma, limit_slopes=limit_slopes)
        )

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (self.reference.num_nodes, self.mesh.num_elements, NUM_VARIABLES)

    def node_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical node coordinates (x, y), each of shape (Np, K)."""
        return node_coordinates(self.mesh, self.reference.r, self.reference.s)

    def _as_state(self, q) -> jax.Array:
        q = jnp.asarray(q, dtype=jnp.float64)
        if q.shape != self.state_shape:
            raise ValueError(f"Expected a state of shape {self.state_shape}, got {q.shape}.")
        return q

    def rhs(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Semi-discrete right-hand side and net outward boundary flux."""
        rhs, boundary_flux = self._rhs(self._as_state(q), self.operators)
        return np.asarray(rhs), np.asarray(boundary_flux)

    def limit(self, q) -> np.ndarray:
        """Apply the ADR 0010 limiters once."""
        limited, _, _ = self._limit(self._as_state(q), self.limiter_operators)
        return np.asarray(limited)

    def time_step(self, q) -> float:
        """CFL time step for the state ``q`` (ADR 0009)."""
        return float(self._time_step(self._as_state(q), self._dt_geometry))

    def step(self, q, dt: float) -> tuple[np.ndarray, StepReport]:
        """One SSP-RK3 step without rejection handling."""
        q_new, report = self._step(self._as_state(q), jnp.float64(dt), self.operators, self.limiter_operators)
        return np.asarray(q_new), jax.tree.map(np.asarray, report)

    def cell_averages(self, q) -> np.ndarray:
        return np.asarray(cell_averages(self._as_state(q), self.operators))

    def integrate(self, q) -> np.ndarray:
        """Domain integral of each conserved variable."""
        return np.asarray(domain_integral(self._as_state(q), self.operators))

    def run(self, initial_state, final_time: float) -> RunResult:
        """Advance ``initial_state`` from t = 0 to ``final_time``."""
        start = time.perf_counter()
        q = self._as_state(initial_state)
        density = q[..., 0]
        initial_pressure = pressure(q, self.config.gamma)
        if not (bool(jnp.all(jnp.isfinite(q))) and bool(jnp.all(density > 0.0)) and bool(jnp.all(initial_pressure > 0.0))):
            raise SolverError("The initial state must be finite with positive nodal density and pressure.")
        min_density = float(jnp.min(density))
        min_pressure = float(jnp.min(initial_pressure))
        flux_integral = np.zeros(NUM_VARIABLES)
        current_time, steps, rejected, max_troubled = 0.0, 0, 0, 0

        while current_time < final_time:
            dt = float(self._time_step(q, self._dt_geometry))
            if not np.isfinite(dt) or dt <= 0.0:
                raise SolverError(f"Invalid time step {dt} at t = {current_time}.")
            remaining = final_time - current_time
            reaches_end = dt >= remaining
            dt = remaining if reaches_end else dt
            for _ in range(self.config.max_rejections + 1):
                q_new, report = self._step(q, jnp.float64(dt), self.operators, self.limiter_operators)
                if bool(report.admissible):
                    break
                rejected += 1
                reaches_end = False
                dt *= 0.5
            else:
                raise SolverError(
                    f"Step at t = {current_time} was rejected {self.config.max_rejections + 1} times."
                )
            q = q_new
            current_time = final_time if reaches_end else current_time + dt
            steps += 1
            flux_integral += np.asarray(report.boundary_flux)
            min_density = min(min_density, float(report.min_density))
            min_pressure = min(min_pressure, float(report.min_pressure))
            max_troubled = max(max_troubled, int(report.troubled_cells))

        return RunResult(
            state=np.asarray(q),
            time=current_time,
            steps=steps,
            rejected_steps=rejected,
            boundary_flux_integral=flux_integral,
            min_density=min_density,
            min_pressure=min_pressure,
            max_troubled_cells=max_troubled,
            wall_time=time.perf_counter() - start,
        )
