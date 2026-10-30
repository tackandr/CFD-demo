"""Unit tests for the Euler relations, interface fluxes and limiter functions
(ADR 0004)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfd_demo.euler import (
    NUM_VARIABLES,
    cartesian_fluxes,
    conserved_to_primitive,
    normal_flux,
    pressure,
    primitive_to_conserved,
    sound_speed,
)
from cfd_demo.fluxes import INTERFACE_FLUXES, get_interface_flux, hllc, roe, rusanov
from cfd_demo.limiters import LIMITER_FUNCTIONS, get_limiter_function, mc, minmod, superbee, van_leer

GAMMA = 1.4

STATES = (
    (1.0, 0.0, 0.0, 1.0),
    (0.125, 0.0, 0.0, 0.1),
    (1.3, 0.7, -0.4, 2.1),
    (0.4, -1.2, 0.9, 0.3),
)


def _conserved(primitive):
    rho, u, v, p = primitive
    return jnp.asarray(primitive_to_conserved(rho, u, v, p, GAMMA))


def test_primitive_conversion_round_trips():
    for primitive in STATES:
        q = _conserved(primitive)
        assert q.shape == (NUM_VARIABLES,)
        assert np.allclose(conserved_to_primitive(q, GAMMA), primitive, atol=1e-14)
        assert pressure(q, GAMMA) == pytest.approx(primitive[3], abs=1e-14)


def test_sound_speed_matches_definition():
    for rho, _, _, p in STATES:
        assert sound_speed(rho, p, GAMMA) == pytest.approx(np.sqrt(GAMMA * p / rho), abs=1e-14)


def test_normal_flux_is_the_projected_cartesian_flux():
    angles = np.linspace(0.0, 2.0 * np.pi, 7)
    for primitive in STATES:
        q = _conserved(primitive)
        fx, fy = cartesian_fluxes(q, GAMMA)
        for angle in angles:
            nx, ny = np.cos(angle), np.sin(angle)
            assert np.allclose(normal_flux(q, nx, ny, GAMMA), fx * nx + fy * ny, atol=1e-13)


def test_cartesian_flux_matches_the_closed_form():
    rho, u, v, p = 1.3, 0.7, -0.4, 2.1
    q = _conserved((rho, u, v, p))
    energy = q[3]
    fx, fy = cartesian_fluxes(q, GAMMA)
    assert np.allclose(fx, [rho * u, rho * u * u + p, rho * u * v, (energy + p) * u], atol=1e-14)
    assert np.allclose(fy, [rho * v, rho * u * v, rho * v * v + p, (energy + p) * v], atol=1e-14)


@pytest.mark.parametrize("name", sorted(INTERFACE_FLUXES))
def test_interface_flux_is_consistent(name):
    """A flux with identical left and right states reduces to the physical
    normal flux (ADR 0004)."""
    flux = get_interface_flux(name)
    for primitive in STATES:
        q = _conserved(primitive)
        for nx, ny in ((1.0, 0.0), (0.0, 1.0), (0.6, -0.8)):
            assert np.allclose(flux(q, q, nx, ny, GAMMA), normal_flux(q, nx, ny, GAMMA), atol=1e-12)


@pytest.mark.parametrize("name", sorted(INTERFACE_FLUXES))
def test_interface_flux_is_conservative(name):
    """Swapping the states and flipping the normal negates the flux, so the
    two sides of a face see equal and opposite contributions."""
    flux = get_interface_flux(name)
    for left in STATES:
        for right in STATES:
            q_m, q_p = _conserved(left), _conserved(right)
            for nx, ny in ((1.0, 0.0), (0.6, -0.8)):
                forward = flux(q_m, q_p, nx, ny, GAMMA)
                backward = flux(q_p, q_m, -nx, -ny, GAMMA)
                assert np.allclose(forward, -backward, atol=1e-12)


@pytest.mark.parametrize("name", sorted(INTERFACE_FLUXES))
def test_interface_flux_is_rotationally_invariant(name):
    """The flux of the rotated state equals the rotated flux."""
    flux = get_interface_flux(name)
    angle = 0.7
    cos, sin = np.cos(angle), np.sin(angle)

    def rotate(q):
        return jnp.asarray([q[0], cos * q[1] - sin * q[2], sin * q[1] + cos * q[2], q[3]])

    q_m, q_p = _conserved(STATES[2]), _conserved(STATES[3])
    nx, ny = 0.6, -0.8
    rotated_normal = (cos * nx - sin * ny, sin * nx + cos * ny)
    direct = rotate(flux(q_m, q_p, nx, ny, GAMMA))
    rotated = flux(rotate(q_m), rotate(q_p), *rotated_normal, GAMMA)
    assert np.allclose(direct, rotated, atol=1e-12)


def test_upwind_fluxes_resolve_a_stationary_contact_exactly():
    """Roe and HLLC carry no dissipation across a contact discontinuity, while
    Rusanov smears it; this is the ADR 0004 reason for offering all three."""
    q_m = _conserved((1.0, 0.0, 0.0, 1.0))
    q_p = _conserved((0.125, 0.0, 0.0, 1.0))
    exact = normal_flux(q_m, 1.0, 0.0, GAMMA)
    assert np.allclose(roe(q_m, q_p, 1.0, 0.0, GAMMA), exact, atol=1e-12)
    assert np.allclose(hllc(q_m, q_p, 1.0, 0.0, GAMMA), exact, atol=1e-12)
    assert not np.allclose(rusanov(q_m, q_p, 1.0, 0.0, GAMMA), exact, atol=1e-6)


def test_upwind_fluxes_pick_the_supersonic_side():
    """In supersonic flow the whole flux comes from the upwind state."""
    right_going = _conserved((1.0, 5.0, 0.0, 1.0))
    left_going = _conserved((0.8, 5.0, 0.0, 1.2))
    for flux in (roe, hllc):
        assert np.allclose(
            flux(right_going, left_going, 1.0, 0.0, GAMMA),
            normal_flux(right_going, 1.0, 0.0, GAMMA),
            atol=1e-12,
        )
        assert np.allclose(
            flux(left_going, right_going, -1.0, 0.0, GAMMA),
            normal_flux(right_going, -1.0, 0.0, GAMMA),
            atol=1e-12,
        )


def test_rusanov_matches_its_definition():
    q_m, q_p = _conserved(STATES[2]), _conserved(STATES[3])
    nx, ny = 0.6, -0.8
    speeds = []
    for q in (q_m, q_p):
        rho, u, v, p = conserved_to_primitive(q, GAMMA)
        speeds.append(abs(u * nx + v * ny) + sound_speed(rho, p, GAMMA))
    expected = 0.5 * (
        normal_flux(q_m, nx, ny, GAMMA) + normal_flux(q_p, nx, ny, GAMMA) - max(speeds) * (q_p - q_m)
    )
    assert np.allclose(rusanov(q_m, q_p, nx, ny, GAMMA), expected, atol=1e-14)


def test_hllc_preserves_positivity_for_a_strong_expansion():
    """A double rarefaction keeps the HLLC flux finite and the scheme stable."""
    q_m = _conserved((1.0, -2.0, 0.0, 0.4))
    q_p = _conserved((1.0, 2.0, 0.0, 0.4))
    result = np.asarray(hllc(q_m, q_p, 1.0, 0.0, GAMMA))
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", sorted(INTERFACE_FLUXES))
def test_interface_flux_is_jit_vmap_and_grad_compatible(name):
    """ADR 0004 requires pure JAX transformable implementations."""
    flux = get_interface_flux(name)
    q_m = jnp.stack([_conserved(state) for state in STATES])
    q_p = jnp.stack([_conserved(state) for state in reversed(STATES)])
    normals = jnp.asarray([[1.0, 0.0], [0.6, -0.8], [0.0, 1.0], [-0.6, 0.8]])

    batched = jax.jit(jax.vmap(flux, in_axes=(0, 0, 0, 0, None)))
    values = batched(q_m, q_p, normals[:, 0], normals[:, 1], GAMMA)
    assert values.shape == (len(STATES), NUM_VARIABLES)
    for index in range(len(STATES)):
        assert np.allclose(values[index], flux(q_m[index], q_p[index], *normals[index], GAMMA))

    def mass_flux(q):
        return flux(q, q_p[0], 0.6, -0.8, GAMMA)[0]

    gradient = jax.grad(mass_flux)(q_m[0])
    assert gradient.shape == (NUM_VARIABLES,)
    assert np.all(np.isfinite(gradient))


def test_unknown_interface_flux_is_rejected():
    with pytest.raises(ValueError, match="Unknown interface flux 'godunov'"):
        get_interface_flux("godunov")


@pytest.mark.parametrize(
    ("name", "function"),
    [("minmod", minmod), ("mc", mc), ("van_leer", van_leer), ("superbee", superbee)],
)
def test_limiter_is_second_order_tvd(name, function):
    """Every limiter vanishes at extrema, is bounded by the Sweby region, and
    passes through phi(1) = 1."""
    r = jnp.asarray(np.linspace(-5.0, 5.0, 401))
    phi = np.asarray(function(r))
    r = np.asarray(r)
    assert np.all(phi[r <= 0.0] == 0.0)
    assert function(1.0) == pytest.approx(1.0, abs=1e-14)
    positive = r > 0.0
    assert np.all(phi[positive] <= np.minimum(2.0 * r[positive], 2.0) + 1e-12)
    assert np.all(phi[positive] >= 0.0)
    assert get_limiter_function(name) is function


def test_limiter_closed_forms():
    r = jnp.asarray([-1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    assert np.allclose(minmod(r), [0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 1.0])
    assert np.allclose(mc(r), [0.0, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0])
    assert np.allclose(van_leer(r), [0.0, 0.0, 0.4, 2 / 3, 1.0, 4 / 3, 1.6])
    assert np.allclose(superbee(r), [0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0])


def test_unlimited_returns_one_everywhere():
    unlimited = get_limiter_function("unlimited")
    r = jnp.asarray([-3.0, 0.0, 0.5, 7.0])
    assert np.allclose(unlimited(r), 1.0)
    assert unlimited(r).shape == r.shape


@pytest.mark.parametrize("name", sorted(LIMITER_FUNCTIONS))
def test_limiter_is_jit_vmap_and_grad_compatible(name):
    function = get_limiter_function(name)
    r = jnp.asarray([0.3, 1.0, 2.5])
    assert np.allclose(jax.jit(jax.vmap(function))(r), function(r))
    gradient = jax.grad(lambda value: function(value).sum())(r)
    assert np.all(np.isfinite(gradient))


def test_unknown_limiter_is_rejected():
    with pytest.raises(ValueError, match="Unknown limiter 'ospre'"):
        get_limiter_function("ospre")
