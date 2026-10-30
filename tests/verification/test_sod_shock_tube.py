"""ADR 0005 verification: the two-dimensional Sod shock tube.

Runs the baseline case (`rusanov` with `minmod`, ADR 0010) at both required
resolutions and applies the four acceptance checks of ADR 0005. The exact
reference and all diagnostics are independent NumPy test code (ADR 0006).

The run is shared by every test in this module through a session fixture; it
dominates the suite runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cfd_demo import sod_shock_tube
from cfd_demo.cases import (
    SOD_DISCONTINUITY,
    SOD_FINAL_TIME,
    SOD_GAMMA,
    SOD_LEFT,
    SOD_RIGHT,
    SOD_X_RANGE,
    SOD_Y_RANGE,
)

from . import diagnostics as dg
from .sod_reference import sod_solution

pytestmark = pytest.mark.verification

RESOLUTIONS = {"coarse": (100, 10), "fine": (200, 20)}
BASELINE = "fine"
FLUX = "rusanov"
LIMITER = "minmod"

# ADR 0005 acceptance thresholds.
MAX_DENSITY_L1 = 0.03
MAX_SHOCK_ERROR = 0.02
MAX_CONSERVATION_DEFECT = 1e-8
SHOCK_WINDOW = (0.8, 0.9)


@dataclass(frozen=True, eq=False)
class SodRun:
    """A completed Sod run with its diagnostics."""

    nx: int
    ny: int
    case: object
    result: object
    initial_integral: np.ndarray
    final_integral: np.ndarray
    l1_errors: dict[str, float]
    shock_x: float
    column_averages: np.ndarray

    @property
    def conservation_defect(self) -> np.ndarray:
        return self.final_integral - self.initial_integral + self.result.boundary_flux_integral


def _analyse(nx: int, ny: int) -> SodRun:
    case = sod_shock_tube(nx, ny, flux=FLUX, limiter=LIMITER)
    solver = case.solver
    initial_integral = solver.integrate(case.initial_state)
    result = solver.run(case.initial_state, case.final_time)

    exact = sod_solution(SOD_DISCONTINUITY, SOD_GAMMA)
    waves = exact.wave_positions(result.time)
    reference = solver.reference
    coefficients = dg.nodal_coefficients(result.state, reference.r, reference.s, reference.degree)

    split = dg.build_quadrature(solver.mesh, list(waves.values()))
    numerical = dg.evaluate_state(coefficients, split, reference.degree)
    rho, u, v, p = dg.to_primitive(numerical, SOD_GAMMA)
    exact_rho, exact_u, exact_p = exact.sample(split.x, result.time)
    errors = dg.l1_errors(
        {"density": rho, "pressure": p, "x velocity": u, "y velocity": v},
        {
            "density": exact_rho,
            "pressure": exact_p,
            "x velocity": exact_u,
            "y velocity": np.zeros_like(exact_u),
        },
        split,
    )

    unsplit = dg.build_quadrature(solver.mesh)
    density = dg.evaluate_state(coefficients, unsplit, reference.degree)[..., 0]
    averages, face_x = dg.column_density_averages(density, unsplit, solver.mesh, nx)
    return SodRun(
        nx=nx,
        ny=ny,
        case=case,
        result=result,
        initial_integral=initial_integral,
        final_integral=solver.integrate(result.state),
        l1_errors=errors,
        shock_x=dg.shock_location(averages, face_x, SHOCK_WINDOW),
        column_averages=averages,
    )


@pytest.fixture(scope="session")
def sod_runs(report) -> dict[str, SodRun]:
    """Run the baseline case once per resolution and report the diagnostics."""
    runs = {name: _analyse(*size) for name, size in RESOLUTIONS.items()}
    exact = sod_solution(SOD_DISCONTINUITY, SOD_GAMMA)
    waves = exact.wave_positions(SOD_FINAL_TIME)
    report(f"ADR 0005 Sod shock tube, flux {FLUX!r}, limiter {LIMITER!r}, t = {SOD_FINAL_TIME}")
    report("  exact waves: " + ", ".join(f"{k} {v:.5f}" for k, v in waves.items()))
    for name, run in runs.items():
        errors = ", ".join(f"{field} {value:.3e}" for field, value in run.l1_errors.items())
        report(f"  {name} {run.nx}x{run.ny} ({run.case.solver.mesh.num_elements} elements)")
        report(f"    L1 errors: {errors}")
        report(
            f"    shock {run.shock_x:.5f} (exact {waves['shock']:.5f},"
            f" error {abs(run.shock_x - waves['shock']):.5f})"
        )
        report(
            f"    steps {run.result.steps}, rejected {run.result.rejected_steps},"
            f" max troubled cells {run.result.max_troubled_cells},"
            f" wall time {run.result.wall_time:.1f} s"
        )
        report(
            f"    min density {run.result.min_density:.3e},"
            f" min pressure {run.result.min_pressure:.3e},"
            f" max conservation defect {np.abs(run.conservation_defect).max():.3e}"
        )
    return runs


@pytest.fixture(scope="session")
def baseline(sod_runs) -> SodRun:
    return sod_runs[BASELINE]


def test_case_matches_adr_0005_setup(baseline):
    """The baseline discretization is the one ADR 0005 prescribes."""
    solver = baseline.case.solver
    mesh = solver.mesh
    assert (baseline.nx, baseline.ny) == (200, 20)
    assert mesh.num_elements == 2 * baseline.nx * baseline.ny
    assert solver.reference.degree == 4
    assert solver.config.flux == FLUX and solver.config.limiter == LIMITER
    assert solver.config.gamma == SOD_GAMMA
    assert baseline.case.final_time == SOD_FINAL_TIME
    assert baseline.result.time == pytest.approx(SOD_FINAL_TIME, abs=1e-14)
    assert baseline.result.state.dtype == np.float64

    corners = mesh.vertices[mesh.elements]
    assert corners[..., 0].min() == SOD_X_RANGE[0] and corners[..., 0].max() == SOD_X_RANGE[1]
    assert corners[..., 1].min() == SOD_Y_RANGE[0] and corners[..., 1].max() == SOD_Y_RANGE[1]

    # Every rectangle is split along its lower-left to upper-right diagonal.
    cells = corners.reshape(-1, 2, 3, 2)
    lower_left = cells.reshape(-1, 6, 2).min(axis=1)
    upper_right = cells.reshape(-1, 6, 2).max(axis=1)
    for triangle in (cells[:, 0], cells[:, 1]):
        has_lower_left = np.isclose(triangle, lower_left[:, None, :]).all(axis=-1).any(axis=1)
        has_upper_right = np.isclose(triangle, upper_right[:, None, :]).all(axis=-1).any(axis=1)
        assert has_lower_left.all() and has_upper_right.all()

    # The initial discontinuity is aligned with element faces at x = 0.5.
    left_of = corners[..., 0] <= SOD_DISCONTINUITY
    right_of = corners[..., 0] >= SOD_DISCONTINUITY
    assert np.all(left_of.all(axis=1) | right_of.all(axis=1))


def test_initial_condition_is_set_per_element(baseline):
    """Each element is initialized from its centroid, with independent traces
    on either side of the discontinuity (ADR 0005)."""
    solver = baseline.case.solver
    state = baseline.case.initial_state
    on_left = solver.geometry.centroid[:, 0] < SOD_DISCONTINUITY
    assert on_left.sum() == solver.mesh.num_elements // 2

    gamma = SOD_GAMMA
    expected_left = np.array(
        [SOD_LEFT[0], SOD_LEFT[0] * SOD_LEFT[1], SOD_LEFT[0] * SOD_LEFT[2], SOD_LEFT[3] / (gamma - 1.0)]
    )
    expected_right = np.array(
        [SOD_RIGHT[0], SOD_RIGHT[0] * SOD_RIGHT[1], SOD_RIGHT[0] * SOD_RIGHT[2], SOD_RIGHT[3] / (gamma - 1.0)]
    )
    assert np.allclose(state[:, on_left, :], expected_left)
    assert np.allclose(state[:, ~on_left, :], expected_right)

    # Coincident nodes on the discontinuity carry different traces.
    x, _ = solver.node_coordinates()
    on_face = np.isclose(x, SOD_DISCONTINUITY)
    assert on_face.any()
    densities = np.unique(np.round(state[..., 0][on_face], 12))
    assert sorted(densities) == pytest.approx([SOD_RIGHT[0], SOD_LEFT[0]])


@pytest.mark.parametrize("resolution", sorted(RESOLUTIONS))
def test_density_and_pressure_stay_positive(sod_runs, resolution):
    """ADR 0005 check 1, at the solution nodes, which are also the quadrature
    points of the collocated operator (ADR 0008) and the points at which
    ADR 0010 enforces positivity."""
    run = sod_runs[resolution]
    state = run.result.state
    assert np.isfinite(state).all()
    rho, _, _, p = dg.to_primitive(state, SOD_GAMMA)
    assert rho.min() > 0.0
    assert p.min() > 0.0
    # The solver's minima cover every limited Runge-Kutta stage of every step.
    assert run.result.min_density > 0.0
    assert run.result.min_pressure > 0.0


@pytest.mark.parametrize("resolution", sorted(RESOLUTIONS))
def test_conservation_against_boundary_fluxes(sod_runs, resolution):
    """ADR 0005 check 3: mass, momentum, and energy against the
    time-integrated numerical boundary fluxes, including wall pressure."""
    run = sod_runs[resolution]
    defect = run.conservation_defect
    assert np.abs(defect).max() <= MAX_CONSERVATION_DEFECT, f"conservation defect {defect}"

    # The walls exert a net streamwise force, so the check is not trivial.
    assert abs(run.result.boundary_flux_integral[1]) > 1e-3
    assert run.result.rejected_steps == 0


@pytest.mark.parametrize("resolution", sorted(RESOLUTIONS))
def test_density_error_and_shock_location(sod_runs, resolution):
    """ADR 0005 check 2."""
    run = sod_runs[resolution]
    exact = sod_solution(SOD_DISCONTINUITY, SOD_GAMMA)
    shock = exact.wave_positions(run.result.time)["shock"]
    assert run.l1_errors["density"] <= MAX_DENSITY_L1
    assert abs(run.shock_x - shock) <= MAX_SHOCK_ERROR


def test_errors_decrease_under_refinement(sod_runs):
    """ADR 0005 check 4: report all errors and compare the two resolutions."""
    coarse = sod_runs["coarse"].l1_errors["density"]
    fine = sod_runs["fine"].l1_errors["density"]
    assert fine < coarse, f"fine density L1 {fine:.3e} is not below coarse {coarse:.3e}"
    for run in sod_runs.values():
        assert set(run.l1_errors) == {"density", "pressure", "x velocity", "y velocity"}
        assert all(np.isfinite(value) for value in run.l1_errors.values())


def test_transverse_velocity_stays_small(baseline):
    """The case is uniform in y, so the transverse velocity measures only
    mesh-induced asymmetry (ADR 0007)."""
    assert baseline.l1_errors["y velocity"] < baseline.l1_errors["x velocity"]
