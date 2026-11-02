"""Selectable numerical interface fluxes (ADRs 0004 and 0008).

Each flux has the signature ``flux(q_minus, q_plus, nx, ny, gamma)`` and
returns the numerical approximation of ``F . n`` on the face, where ``n`` is
the unit normal pointing from the ``q_minus`` side to the ``q_plus`` side.
States have shape (..., 4); normals broadcast against the leading axes. The
functions are pure JAX and compatible with ``jit``, ``vmap``, and ``grad``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from .euler import conserved_to_primitive, normal_flux, sound_speed

InterfaceFlux = Callable[..., jnp.ndarray]

# Harten entropy-fix parameter for the Roe flux (ADR 0008).
ROE_ENTROPY_FIX = 0.1


def _safe_denominator(value):
    return jnp.where(value == 0.0, jnp.finfo(jnp.result_type(value)).tiny, value)


def rusanov(q_minus, q_plus, nx, ny, gamma: float):
    """Rusanov (local Lax-Friedrichs) flux."""
    rho_m, u_m, v_m, p_m = conserved_to_primitive(q_minus, gamma)
    rho_p, u_p, v_p, p_p = conserved_to_primitive(q_plus, gamma)
    speed_m = jnp.abs(u_m * nx + v_m * ny) + sound_speed(rho_m, p_m, gamma)
    speed_p = jnp.abs(u_p * nx + v_p * ny) + sound_speed(rho_p, p_p, gamma)
    alpha = jnp.maximum(speed_m, speed_p)[..., None]
    f_m = normal_flux(q_minus, nx, ny, gamma)
    f_p = normal_flux(q_plus, nx, ny, gamma)
    return 0.5 * (f_m + f_p) - 0.5 * alpha * (q_plus - q_minus)


def _roe_average(q_minus, q_plus, gamma: float):
    rho_m, u_m, v_m, p_m = conserved_to_primitive(q_minus, gamma)
    rho_p, u_p, v_p, p_p = conserved_to_primitive(q_plus, gamma)
    h_m = (q_minus[..., 3] + p_m) / rho_m
    h_p = (q_plus[..., 3] + p_p) / rho_p
    w_m = jnp.sqrt(rho_m)
    w_p = jnp.sqrt(rho_p)
    u = (w_m * u_m + w_p * u_p) / (w_m + w_p)
    v = (w_m * v_m + w_p * v_p) / (w_m + w_p)
    h = (w_m * h_m + w_p * h_p) / (w_m + w_p)
    c = jnp.sqrt((gamma - 1.0) * (h - 0.5 * (u * u + v * v)))
    return w_m * w_p, u, v, h, c


def roe(q_minus, q_plus, nx, ny, gamma: float):
    """Roe flux with Harten's entropy fix on the acoustic waves."""
    rho_m, u_m, v_m, p_m = conserved_to_primitive(q_minus, gamma)
    rho_p, u_p, v_p, p_p = conserved_to_primitive(q_plus, gamma)
    rho, u, v, h, c = _roe_average(q_minus, q_plus, gamma)
    un = u * nx + v * ny
    ut = -u * ny + v * nx

    d_rho = rho_p - rho_m
    d_p = p_p - p_m
    d_un = (u_p * nx + v_p * ny) - (u_m * nx + v_m * ny)
    d_ut = (-u_p * ny + v_p * nx) - (-u_m * ny + v_m * nx)

    c2 = c * c
    alpha_minus = (d_p - rho * c * d_un) / (2.0 * c2)
    alpha_entropy = d_rho - d_p / c2
    alpha_shear = rho * d_ut
    alpha_plus = (d_p + rho * c * d_un) / (2.0 * c2)

    delta = ROE_ENTROPY_FIX * (jnp.abs(un) + c)

    def entropy_fixed(eigenvalue):
        magnitude = jnp.abs(eigenvalue)
        return jnp.where(magnitude < delta, (eigenvalue**2 + delta**2) / (2.0 * delta), magnitude)

    lam_minus = entropy_fixed(un - c)
    lam_plus = entropy_fixed(un + c)
    lam_mid = jnp.abs(un)

    ones = jnp.ones_like(un)
    zeros = jnp.zeros_like(un)
    r_minus = jnp.stack([ones, u - c * nx, v - c * ny, h - c * un], axis=-1)
    r_entropy = jnp.stack([ones, u, v, 0.5 * (u * u + v * v)], axis=-1)
    r_shear = jnp.stack([zeros, -ny * ones, nx * ones, ut], axis=-1)
    r_plus = jnp.stack([ones, u + c * nx, v + c * ny, h + c * un], axis=-1)

    dissipation = (
        (lam_minus * alpha_minus)[..., None] * r_minus
        + (lam_mid * alpha_entropy)[..., None] * r_entropy
        + (lam_mid * alpha_shear)[..., None] * r_shear
        + (lam_plus * alpha_plus)[..., None] * r_plus
    )
    f_m = normal_flux(q_minus, nx, ny, gamma)
    f_p = normal_flux(q_plus, nx, ny, gamma)
    return 0.5 * (f_m + f_p) - 0.5 * dissipation


def hllc(q_minus, q_plus, nx, ny, gamma: float):
    """Toro's HLLC flux with Einfeldt (Roe-average) wave-speed estimates."""
    rho_m, u_m, v_m, p_m = conserved_to_primitive(q_minus, gamma)
    rho_p, u_p, v_p, p_p = conserved_to_primitive(q_plus, gamma)
    un_m = u_m * nx + v_m * ny
    un_p = u_p * nx + v_p * ny
    c_m = sound_speed(rho_m, p_m, gamma)
    c_p = sound_speed(rho_p, p_p, gamma)
    _, u, v, _, c = _roe_average(q_minus, q_plus, gamma)
    un = u * nx + v * ny

    s_m = jnp.minimum(un_m - c_m, un - c)
    s_p = jnp.maximum(un_p + c_p, un + c)
    mass_m = rho_m * (s_m - un_m)
    mass_p = rho_p * (s_p - un_p)
    s_star = (p_p - p_m + mass_m * un_m - mass_p * un_p) / _safe_denominator(mass_m - mass_p)

    def star_state(q, rho, u_k, v_k, p, un_k, s_k, mass_k):
        factor = mass_k / _safe_denominator(s_k - s_star)
        shift = s_star - un_k
        return factor[..., None] * jnp.stack(
            [
                jnp.ones_like(rho),
                u_k + shift * nx,
                v_k + shift * ny,
                q[..., 3] / rho + shift * (s_star + p / _safe_denominator(mass_k)),
            ],
            axis=-1,
        )

    f_m = normal_flux(q_minus, nx, ny, gamma)
    f_p = normal_flux(q_plus, nx, ny, gamma)
    star_m = star_state(q_minus, rho_m, u_m, v_m, p_m, un_m, s_m, mass_m)
    star_p = star_state(q_plus, rho_p, u_p, v_p, p_p, un_p, s_p, mass_p)
    f_star_m = f_m + s_m[..., None] * (star_m - q_minus)
    f_star_p = f_p + s_p[..., None] * (star_p - q_plus)

    s_m, s_p, s_star = s_m[..., None], s_p[..., None], s_star[..., None]
    return jnp.where(
        s_m >= 0.0,
        f_m,
        jnp.where(s_star >= 0.0, f_star_m, jnp.where(s_p > 0.0, f_star_p, f_p)),
    )


INTERFACE_FLUXES: dict[str, InterfaceFlux] = {
    "rusanov": rusanov,
    "roe": roe,
    "hllc": hllc,
}


def get_interface_flux(name: str) -> InterfaceFlux:
    """Return the interface flux with the given stable name."""
    try:
        return INTERFACE_FLUXES[name]
    except KeyError:
        options = ", ".join(sorted(INTERFACE_FLUXES))
        raise ValueError(f"Unknown interface flux {name!r}; choose one of: {options}.") from None
