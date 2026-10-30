"""Calorically perfect gas relations for the two-dimensional Euler equations.

Conserved variables are ordered ``(rho, rho*u, rho*v, E)`` along the last array
axis (ADR 0008). All functions are pure JAX functions.
"""

from __future__ import annotations

import jax.numpy as jnp

NUM_VARIABLES = 4


def primitive_to_conserved(rho, u, v, p, gamma: float):
    """Convert primitive ``(rho, u, v, p)`` to conserved variables."""
    rho, u, v, p = (jnp.asarray(value) for value in (rho, u, v, p))
    energy = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return jnp.stack(jnp.broadcast_arrays(rho, rho * u, rho * v, energy), axis=-1)


def conserved_to_primitive(q, gamma: float):
    """Return ``(rho, u, v, p)`` for conserved states ``q`` of shape (..., 4)."""
    rho = q[..., 0]
    u = q[..., 1] / rho
    v = q[..., 2] / rho
    p = (gamma - 1.0) * (q[..., 3] - 0.5 * (q[..., 1] * u + q[..., 2] * v))
    return rho, u, v, p


def pressure(q, gamma: float):
    """Pressure of conserved states ``q``."""
    return conserved_to_primitive(q, gamma)[3]


def sound_speed(rho, p, gamma: float):
    """Speed of sound of an ideal gas."""
    return jnp.sqrt(gamma * p / rho)


def cartesian_fluxes(q, gamma: float):
    """Physical fluxes ``(F, G)`` in x and y, each with the shape of ``q``."""
    rho, u, v, p = conserved_to_primitive(q, gamma)
    mx, my, energy = q[..., 1], q[..., 2], q[..., 3]
    f = jnp.stack([mx, mx * u + p, my * u, (energy + p) * u], axis=-1)
    g = jnp.stack([my, mx * v, my * v + p, (energy + p) * v], axis=-1)
    return f, g


def normal_flux(q, nx, ny, gamma: float):
    """Physical flux ``F(q) . n`` for unit normals ``(nx, ny)``."""
    rho, u, v, p = conserved_to_primitive(q, gamma)
    un = u * nx + v * ny
    return jnp.stack(
        [
            rho * un,
            q[..., 1] * un + p * nx,
            q[..., 2] * un + p * ny,
            (q[..., 3] + p) * un,
        ],
        axis=-1,
    )
