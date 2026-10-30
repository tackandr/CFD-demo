"""Unit tests for the DG operator and its conservation properties
(ADRs 0008 and 0007)."""

from __future__ import annotations

import numpy as np
import pytest

from cfd_demo.boundary import FixedState, SlipWall
from cfd_demo.euler import conserved_to_primitive, primitive_to_conserved
from cfd_demo.fluxes import INTERFACE_FLUXES
from cfd_demo.mesh import TriangleMesh, rectangle_mesh
from cfd_demo.solver import EulerSolver, SolverConfig

GAMMA = 1.4
FREE_STREAM = (1.2, 0.4, -0.3, 0.9)


def perturbed_mesh(nx: int, ny: int, amplitude: float = 0.25, seed: int = 0) -> TriangleMesh:
    """Rectangle mesh with randomly displaced interior vertices."""
    mesh = rectangle_mesh(nx, ny)
    vertices = np.array(mesh.vertices)
    interior = np.all((vertices > 1e-12) & (vertices < 1.0 - 1e-12), axis=1)
    rng = np.random.default_rng(seed)
    displacement = rng.uniform(-1.0, 1.0, size=vertices.shape)
    vertices[interior] += amplitude * displacement[interior] / np.array([nx, ny])
    return TriangleMesh(vertices=vertices, elements=mesh.elements, boundary_edges=mesh.boundary_edges)


def make_solver(mesh: TriangleMesh, conditions, flux: str = "rusanov", limiter: str = "minmod"):
    config = SolverConfig(flux=flux, limiter=limiter, gamma=GAMMA)
    return EulerSolver(mesh, conditions, config)


def uniform_state(solver: EulerSolver, primitive) -> np.ndarray:
    conserved = np.asarray(primitive_to_conserved(*primitive, GAMMA))
    return np.broadcast_to(conserved, solver.state_shape).copy()


def fixed_state_conditions(primitive):
    condition = FixedState.from_primitive(*primitive, GAMMA)
    return {tag: condition for tag in ("left", "right", "bottom", "top")}


@pytest.mark.parametrize("flux", sorted(INTERFACE_FLUXES))
def test_free_stream_is_preserved_on_a_distorted_mesh(flux):
    """A uniform state is an exact steady solution: the volume and surface
    terms cancel to round-off on arbitrary straight-sided triangles."""
    solver = make_solver(perturbed_mesh(6, 5), fixed_state_conditions(FREE_STREAM), flux=flux)
    q = uniform_state(solver, FREE_STREAM)
    rhs, _ = solver.rhs(q)
    assert np.abs(rhs).max() < 1e-11


def test_free_stream_survives_limiting_and_a_time_step():
    solver = make_solver(perturbed_mesh(6, 5), fixed_state_conditions(FREE_STREAM))
    q = uniform_state(solver, FREE_STREAM)
    assert np.abs(solver.limit(q) - q).max() < 1e-13
    advanced, report = solver.step(q, solver.time_step(q))
    assert np.abs(advanced - q).max() < 1e-11
    assert bool(report.admissible)
    assert int(report.troubled_cells) == 0


def test_slip_walls_keep_a_closed_box_at_rest():
    """With reflecting walls and a quiescent state the residual vanishes and no
    momentum crosses the boundary."""
    walls = {tag: SlipWall() for tag in ("left", "right", "bottom", "top")}
    solver = make_solver(perturbed_mesh(5, 4), walls)
    q = uniform_state(solver, (1.0, 0.0, 0.0, 1.0))
    rhs, boundary_flux = solver.rhs(q)
    assert np.abs(rhs).max() < 1e-12
    # The wall pressure integrates to zero over a closed boundary.
    assert np.abs(boundary_flux).max() < 1e-12


def test_slip_wall_reflects_the_normal_momentum():
    """A uniform flow in a closed box exchanges no mass or energy with the
    walls; the mirrored state leaves only a streamwise force."""
    walls = {tag: SlipWall() for tag in ("left", "right", "bottom", "top")}
    solver = make_solver(rectangle_mesh(4, 4), walls, flux="rusanov")
    rho, u, p = 1.0, 0.5, 1.0
    q = uniform_state(solver, (rho, u, 0.0, p))
    _, boundary_flux = solver.rhs(q)

    # Mass, transverse momentum and energy cannot cross a mirrored boundary.
    assert np.abs(boundary_flux[[0, 2, 3]]).max() < 1e-12
    # The pressure terms of the two streamwise walls cancel, leaving only the
    # Rusanov dissipation against the mirrored normal velocity.
    wave_speed = abs(u) + np.sqrt(GAMMA * p / rho)
    assert boundary_flux[1] == pytest.approx(2.0 * wave_speed * rho * u, abs=1e-12)


@pytest.mark.parametrize("flux", sorted(INTERFACE_FLUXES))
def test_discrete_conservation_identity(flux):
    """Summing the residual over the mesh leaves only the boundary flux: the
    interior face contributions cancel in pairs (ADR 0008)."""
    solver = make_solver(perturbed_mesh(5, 4, seed=3), fixed_state_conditions(FREE_STREAM), flux=flux)
    rng = np.random.default_rng(11)
    primitives = rng.uniform(
        [0.5, -0.4, -0.4, 0.5], [1.5, 0.4, 0.4, 1.5], size=solver.state_shape[:2] + (4,)
    )
    q = np.asarray(primitive_to_conserved(*(primitives[..., k] for k in range(4)), GAMMA))
    rhs, boundary_flux = solver.rhs(q)
    assert np.abs(solver.integrate(rhs) + boundary_flux).max() < 1e-12
    assert np.abs(boundary_flux).max() > 1e-3


def test_integrate_and_cell_averages_agree():
    solver = make_solver(perturbed_mesh(4, 3), fixed_state_conditions(FREE_STREAM))
    q = uniform_state(solver, FREE_STREAM)
    conserved = np.asarray(primitive_to_conserved(*FREE_STREAM, GAMMA))
    assert np.allclose(solver.cell_averages(q), conserved)
    # The unit square: the domain integral is the conserved state itself.
    assert np.allclose(solver.integrate(q), conserved)

    areas = solver.geometry.area
    weighted = (solver.cell_averages(q) * areas[:, None]).sum(axis=0)
    assert np.allclose(weighted, solver.integrate(q))


def _smooth_primitives(x, y):
    two_pi = 2.0 * np.pi
    return (
        1.0 + 0.2 * np.sin(two_pi * x) * np.sin(two_pi * y),
        0.3 + 0.1 * np.cos(two_pi * x) * np.sin(two_pi * y),
        -0.2 + 0.1 * np.sin(two_pi * x) * np.cos(two_pi * y),
        1.0 + 0.15 * np.sin(two_pi * (x + y)),
    )


def _smooth_state(x, y):
    return np.asarray(primitive_to_conserved(*_smooth_primitives(x, y), GAMMA))


def _analytic_rhs(x, y, h: float = 1e-4):
    """-div F of the smooth state, by fourth-order central differences."""

    def flux(x, y, axis):
        rho, u, v, p = _smooth_primitives(x, y)
        energy = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
        velocity = u if axis == 0 else v
        return np.stack(
            [
                rho * velocity,
                rho * u * velocity + (p if axis == 0 else 0.0),
                rho * v * velocity + (p if axis == 1 else 0.0),
                (energy + p) * velocity,
            ],
            axis=-1,
        )

    def derivative(axis):
        shifts = [(-2, 1 / 12), (-1, -2 / 3), (1, 2 / 3), (2, -1 / 12)]
        total = 0.0
        for steps, weight in shifts:
            dx = steps * h if axis == 0 else 0.0
            dy = steps * h if axis == 1 else 0.0
            total = total + weight * flux(x + dx, y + dy, axis)
        return total / h

    return -(derivative(0) + derivative(1))


def _consistency_error(nx: int, ny: int) -> float:
    solver = make_solver(rectangle_mesh(nx, ny), fixed_state_conditions(FREE_STREAM))
    x, y = solver.node_coordinates()
    rhs, _ = solver.rhs(_smooth_state(x, y))
    exact = _analytic_rhs(x, y)
    # Boundary conditions do not match the smooth field, so only elements
    # entirely surrounded by neighbors measure the interior operator.
    interior = np.all(solver.connectivity.neighbor_element >= 0, axis=1)
    assert interior.sum() > 0.4 * solver.mesh.num_elements
    return float(np.abs(rhs[:, interior] - exact[:, interior]).max())


def test_interior_operator_is_fourth_order_accurate():
    """Halving the mesh size cuts the residual error by about 2^4, which is the
    design order of the degree-four operator (ADR 0008)."""
    errors = [_consistency_error(size, size) for size in (8, 16, 32)]
    assert errors[0] < 5e-2
    orders = [np.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:], strict=False)]
    assert all(order > 3.5 for order in orders), f"errors {errors}, orders {orders}"


def test_rhs_rejects_a_badly_shaped_state():
    solver = make_solver(rectangle_mesh(2, 2), fixed_state_conditions(FREE_STREAM))
    with pytest.raises(ValueError, match="Expected a state of shape"):
        solver.rhs(np.zeros((3, 4)))


def test_solver_config_validates_its_options():
    with pytest.raises(ValueError, match="Unknown interface flux"):
        SolverConfig(flux="godunov", limiter="minmod", gamma=GAMMA)
    with pytest.raises(ValueError, match="Unknown limiter"):
        SolverConfig(flux="rusanov", limiter="ospre", gamma=GAMMA)
    with pytest.raises(ValueError, match="gamma"):
        SolverConfig(flux="rusanov", limiter="minmod", gamma=1.0)
    with pytest.raises(ValueError, match="cfl"):
        SolverConfig(flux="rusanov", limiter="minmod", gamma=GAMMA, cfl=0.0)


def test_state_round_trips_through_primitive_variables():
    solver = make_solver(rectangle_mesh(3, 3), fixed_state_conditions(FREE_STREAM))
    x, y = solver.node_coordinates()
    q = _smooth_state(x, y)
    assert q.shape == solver.state_shape
    recovered = conserved_to_primitive(q, GAMMA)
    for computed, expected in zip(recovered, _smooth_primitives(x, y), strict=True):
        assert np.allclose(computed, expected, atol=1e-13)
