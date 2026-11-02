"""SSP-RK3 time integration with CFL-based step control (ADR 0009)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .dg import DGOperators, dg_rhs
from .euler import conserved_to_primitive, pressure, sound_speed
from .fluxes import InterfaceFlux
from .limiters import LimiterFunction
from .limiting import LimiterOperators, apply_limiters
from .mesh import ElementGeometry

DEFAULT_CFL = 0.5
MAX_REJECTIONS = 10


class StepReport(NamedTuple):
    """Diagnostics of one SSP-RK3 step (a JAX pytree)."""

    boundary_flux: jax.Array  # (4,): dt (B0/6 + B1/6 + 2 B2/3)
    admissible: jax.Array  # finite stages with positive average density and pressure
    min_density: jax.Array  # minimum nodal density over the limited stages
    min_pressure: jax.Array  # minimum nodal pressure over the limited stages
    troubled_cells: jax.Array  # maximum troubled-cell count over the stages


def time_step_geometry(geometry: ElementGeometry, degree: int) -> np.ndarray:
    """Per-element factor min_e 2|K| / ((N + 1)^2 |e|), shape (K,)."""
    return np.min(2.0 * geometry.area[:, None] / ((degree + 1) ** 2 * geometry.edge_length), axis=1)


def stable_time_step(q, dt_geometry, *, gamma: float, cfl: float):
    """CFL time step with lambda_K = max over element nodes of |u| + c."""
    rho, u, v, p = conserved_to_primitive(q, gamma)
    wave_speed = jnp.sqrt(u * u + v * v) + sound_speed(rho, p, gamma)
    return cfl * jnp.min(dt_geometry / jnp.max(wave_speed, axis=0))


def ssp_rk3_step(
    q,
    dt,
    operators: DGOperators,
    limiter_operators: LimiterOperators,
    *,
    flux: InterfaceFlux,
    phi: LimiterFunction,
    gamma: float,
    limit_slopes: bool,
):
    """One Shu-Osher SSP-RK3 step with limiting after every stage."""

    def operator(u):
        return dg_rhs(u, operators, flux, gamma)

    def limit(u):
        u, averages, troubled = apply_limiters(u, limiter_operators, phi, gamma, limit_slopes)
        admissible = (
            jnp.all(jnp.isfinite(u))
            & jnp.all(averages[:, 0] > 0.0)
            & jnp.all(pressure(averages, gamma) > 0.0)
        )
        return u, admissible, troubled

    rhs0, flux0 = operator(q)
    q1, ok1, troubled1 = limit(q + dt * rhs0)
    rhs1, flux1 = operator(q1)
    q2, ok2, troubled2 = limit(0.75 * q + 0.25 * (q1 + dt * rhs1))
    rhs2, flux2 = operator(q2)
    q3, ok3, troubled3 = limit(q / 3.0 + 2.0 / 3.0 * (q2 + dt * rhs2))

    stages = (q1, q2, q3)
    report = StepReport(
        boundary_flux=dt * (flux0 / 6.0 + flux1 / 6.0 + 2.0 * flux2 / 3.0),
        admissible=ok1 & ok2 & ok3,
        min_density=jnp.min(jnp.stack([jnp.min(s[..., 0]) for s in stages])),
        min_pressure=jnp.min(jnp.stack([jnp.min(pressure(s, gamma)) for s in stages])),
        troubled_cells=jnp.max(jnp.stack([troubled1, troubled2, troubled3])),
    )
    return q3, report
