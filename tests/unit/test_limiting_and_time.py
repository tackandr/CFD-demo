"""Unit tests for the limiters and the time integrator (ADRs 0009 and 0010)."""

from __future__ import annotations

import numpy as np
import pytest

from cfd_demo.boundary import FixedState, SlipWall
from cfd_demo.euler import conserved_to_primitive, pressure, primitive_to_conserved, sound_speed
from cfd_demo.limiting import POSITIVITY_FLOOR, apply_limiters
from cfd_demo.mesh import element_geometry, rectangle_mesh
from cfd_demo.solver import DEFAULT_CFL, EulerSolver, SolverConfig, SolverError
from cfd_demo.time_integration import time_step_geometry

from .test_dg_operator import (
    FREE_STREAM,
    GAMMA,
    fixed_state_conditions,
    make_solver,
    perturbed_mesh,
    uniform_state,
)


def smooth_linear_state(solver: EulerSolver) -> np.ndarray:
    """A state whose conserved variables are linear in x and y, which both
    ADR 0010 limiters must leave unchanged away from the boundary."""
    x, y = solver.node_coordinates()
    base = np.asarray(primitive_to_conserved(*FREE_STREAM, GAMMA))
    gradient = np.array([0.05, 0.03, -0.02, 0.07])
    return base + gradient * (0.3 * x + 0.2 * y)[..., None]


def test_limiting_preserves_cell_averages():
    """Both limiters only redistribute within an element (ADR 0010)."""
    solver = make_solver(perturbed_mesh(6, 5, seed=5), fixed_state_conditions(FREE_STREAM))
    rng = np.random.default_rng(1)
    q = smooth_linear_state(solver)
    q = q * (1.0 + 0.3 * rng.standard_normal(q.shape))
    before = solver.cell_averages(q)
    after = solver.cell_averages(solver.limit(q))
    assert np.abs(after - before).max() < 1e-12


def test_limiter_leaves_a_linear_solution_alone():
    """A globally linear field has no spurious extrema, so the troubled-cell
    detector must not fire in the interior."""
    solver = make_solver(rectangle_mesh(8, 8), fixed_state_conditions(FREE_STREAM))
    q = smooth_linear_state(solver)
    limited = solver.limit(q)
    interior = np.all(solver.connectivity.neighbor_element >= 0, axis=1)
    assert np.abs(limited[:, interior] - q[:, interior]).max() < 1e-12


def test_troubled_cell_limiter_flattens_an_oscillation():
    """An element carrying an oscillation beyond its neighbor differences is
    replaced by a limited linear reconstruction."""
    solver = make_solver(rectangle_mesh(8, 8), fixed_state_conditions(FREE_STREAM))
    q = uniform_state(solver, FREE_STREAM)
    target = 2 * (4 * 8 + 4)
    reference = solver.reference
    oscillation = 0.2 * np.cos(4.0 * np.pi * reference.r) * np.cos(3.0 * np.pi * reference.s)
    oscillation -= reference.cell_weights @ oscillation / reference.cell_weights.sum()
    q[:, target, 0] += oscillation
    limited = solver.limit(q)

    assert np.abs(limited[:, target] - q[:, target]).max() > 1e-6
    others = np.arange(solver.mesh.num_elements) != target
    assert np.abs(limited[:, others] - q[:, others]).max() < 1e-12
    # The replacement is linear: it is reproduced by its own P1 projection.
    projected = reference.p1_midpoint @ limited[:, target, 0]
    basis = reference.linear_midpoint_basis
    assert np.allclose(basis @ projected, limited[:, target, 0], atol=1e-12)
    assert np.abs(solver.cell_averages(limited) - solver.cell_averages(q)).max() < 1e-12


def test_unlimited_skips_slope_limiting_but_keeps_positivity():
    """ADR 0010 makes ``unlimited`` an explicit choice, not a default."""
    solver = make_solver(rectangle_mesh(6, 6), fixed_state_conditions(FREE_STREAM), limiter="unlimited")
    q = uniform_state(solver, FREE_STREAM)
    target = 2 * (3 * 6 + 3)
    q[:, target, 0] += 0.2 * np.cos(4.0 * np.pi * solver.reference.r)
    assert np.abs(solver.limit(q) - q).max() < 1e-13

    _, _, troubled = apply_limiters(
        q, solver.limiter_operators, lambda r: r, GAMMA, limit_slopes=False
    )
    assert int(troubled) == 0


def test_positivity_limiter_repairs_negative_nodes():
    """Zhang-Shu scaling toward the average restores positive density and
    pressure at every node while keeping the average (ADR 0010)."""
    solver = make_solver(rectangle_mesh(4, 4), fixed_state_conditions(FREE_STREAM), limiter="unlimited")
    q = uniform_state(solver, (1.0, 0.0, 0.0, 1.0))
    target = 5
    deviation = np.linspace(-1.5, 1.5, solver.reference.num_nodes)
    deviation -= solver.reference.cell_weights @ deviation / solver.reference.cell_weights.sum()
    q[:, target, 0] += deviation
    q[:, target, 3] += 1.2 * deviation
    assert q[:, target, 0].min() < 0.0

    limited = solver.limit(q)
    rho, _, _, p = conserved_to_primitive(limited, GAMMA)
    assert float(rho.min()) >= POSITIVITY_FLOOR * (1.0 - 1e-9)
    assert float(p.min()) >= POSITIVITY_FLOOR * (1.0 - 1e-9)
    assert np.abs(solver.cell_averages(limited) - solver.cell_averages(q)).max() < 1e-12
    untouched = np.arange(solver.mesh.num_elements) != target
    assert np.abs(limited[:, untouched] - q[:, untouched]).max() < 1e-13


def test_time_step_matches_the_cfl_formula():
    """ADR 0009: dt = CFL min_K (min_e 2|K| / ((N+1)^2 |e|)) / lambda_K."""
    solver = make_solver(rectangle_mesh(5, 4), fixed_state_conditions(FREE_STREAM))
    q = uniform_state(solver, FREE_STREAM)
    rho, u, v, p = FREE_STREAM
    wave_speed = np.hypot(u, v) + sound_speed(rho, p, GAMMA)
    geometry = time_step_geometry(solver.geometry, solver.reference.degree)
    expected = DEFAULT_CFL * geometry.min() / wave_speed
    assert solver.time_step(q) == pytest.approx(float(expected), rel=1e-12)

    # Halving the mesh size halves the step.
    finer = make_solver(rectangle_mesh(10, 8), fixed_state_conditions(FREE_STREAM))
    assert finer.time_step(uniform_state(finer, FREE_STREAM)) == pytest.approx(
        0.5 * solver.time_step(q), rel=1e-12
    )


def test_time_step_geometry_is_the_inscribed_length():
    """For a right triangle the factor reduces to 2|K| / ((N+1)^2 |e|_max)."""
    geometry = time_step_geometry(element_geometry(rectangle_mesh(1, 1)), 4)
    longest = np.sqrt(2.0)
    assert np.allclose(geometry, 2.0 * 0.5 / (25.0 * longest))


def test_ssp_rk3_reaches_the_requested_time_exactly():
    solver = sod_like_solver()
    q = initial_sod_state(solver)
    final_time = 0.01
    result = solver.run(q, final_time)
    assert result.time == pytest.approx(final_time, abs=0.0)
    assert result.steps > 1
    assert result.rejected_steps == 0
    assert result.wall_time > 0.0
    assert result.state.shape == solver.state_shape


def test_ssp_rk3_is_conservative_over_a_run():
    """ADR 0009 accumulates the stage boundary fluxes with the SSP-RK3 weights
    so that the discrete conservation statement closes to round-off."""
    solver = sod_like_solver()
    q = initial_sod_state(solver)
    result = solver.run(q, 0.01)
    defect = solver.integrate(result.state) - solver.integrate(q) + result.boundary_flux_integral
    assert np.abs(defect).max() < 1e-13


def test_third_order_convergence_of_the_time_integrator():
    """Halving the step size cuts the error of a smooth integration by about
    eight (ADR 0009)."""
    solver = make_solver(
        perturbed_mesh(5, 4, seed=7), fixed_state_conditions(FREE_STREAM), limiter="unlimited"
    )
    x, y = solver.node_coordinates()
    base = np.asarray(primitive_to_conserved(*FREE_STREAM, GAMMA))
    q0 = base * (1.0 + 0.02 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y))[..., None]

    final_time = 0.02
    reference = march(solver, q0, final_time, 64)
    errors = [np.abs(march(solver, q0, final_time, steps) - reference).max() for steps in (4, 8)]
    order = np.log2(errors[0] / errors[1])
    assert order > 2.5, f"errors {errors}, order {order}"


def march(solver: EulerSolver, q, final_time: float, steps: int):
    dt = final_time / steps
    for _ in range(steps):
        q, report = solver.step(q, dt)
        assert bool(report.admissible)
    return q


def test_run_rejects_an_inadmissible_initial_state():
    solver = sod_like_solver()
    q = initial_sod_state(solver)
    q[0, 0, 0] = -1.0
    with pytest.raises(SolverError, match="positive nodal density and pressure"):
        solver.run(q, 0.01)


def test_run_halves_the_step_and_gives_up_after_max_rejections():
    """An absurd CFL makes every attempt fail; ADR 0009 bounds the retries."""
    config = SolverConfig(flux="rusanov", limiter="minmod", gamma=GAMMA, cfl=1e6, max_rejections=2)
    solver = EulerSolver(rectangle_mesh(4, 2), sod_boundaries(), config)
    with pytest.raises(SolverError, match="was rejected 3 times"):
        solver.run(initial_sod_state(solver), 0.5)


def test_step_reports_stage_minima_and_troubled_cells():
    solver = sod_like_solver()
    q = initial_sod_state(solver)
    _, report = solver.step(q, solver.time_step(q))
    assert float(report.min_density) > 0.0
    assert float(report.min_pressure) > 0.0
    assert int(report.troubled_cells) > 0
    assert bool(report.admissible)
    assert report.boundary_flux.shape == (4,)


def sod_boundaries():
    left = FixedState.from_primitive(1.0, 0.0, 0.0, 1.0, GAMMA)
    right = FixedState.from_primitive(0.125, 0.0, 0.0, 0.1, GAMMA)
    return {"left": left, "right": right, "bottom": SlipWall(), "top": SlipWall()}


def sod_like_solver(nx: int = 20, ny: int = 2) -> EulerSolver:
    config = SolverConfig(flux="rusanov", limiter="minmod", gamma=GAMMA)
    return EulerSolver(rectangle_mesh(nx, ny), sod_boundaries(), config)


def initial_sod_state(solver: EulerSolver) -> np.ndarray:
    left = np.asarray(primitive_to_conserved(1.0, 0.0, 0.0, 1.0, GAMMA))
    right = np.asarray(primitive_to_conserved(0.125, 0.0, 0.0, 0.1, GAMMA))
    on_left = solver.geometry.centroid[:, 0] < 0.5
    state = np.empty(solver.state_shape)
    state[:, on_left] = left
    state[:, ~on_left] = right
    return state


def test_pressure_helper_matches_the_primitive_conversion():
    solver = sod_like_solver()
    q = initial_sod_state(solver)
    assert np.allclose(pressure(q, GAMMA), conserved_to_primitive(q, GAMMA)[3])
