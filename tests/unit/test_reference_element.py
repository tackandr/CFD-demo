"""Unit tests for the degree-four reference triangle (ADRs 0003 and 0008)."""

from __future__ import annotations

import numpy as np
import pytest

from cfd_demo.reference_element import (
    DEGREE,
    build_reference_triangle,
    equispaced_nodes,
    gauss_lobatto_nodes,
    vandermonde_2d,
    warp_blend_nodes,
)


@pytest.fixture(scope="module")
def reference():
    return build_reference_triangle(DEGREE)


def test_node_count_and_degree(reference):
    assert reference.degree == 4
    assert reference.num_nodes == (DEGREE + 1) * (DEGREE + 2) // 2 == 15
    assert reference.num_face_nodes == DEGREE + 1 == 5


def test_nodes_lie_in_the_reference_triangle(reference):
    r, s = reference.r, reference.s
    assert np.all(r >= -1 - 1e-12) and np.all(s >= -1 - 1e-12)
    assert np.all(r + s <= 1e-12)
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    for corner in corners:
        assert np.isclose(np.hypot(r - corner[0], s - corner[1]), 0.0, atol=1e-12).sum() == 1


def test_face_nodes_are_gauss_lobatto_points(reference):
    """ADR 0003 requires the GLL edge nodes that make neighboring face nodes
    coincide."""
    expected = gauss_lobatto_nodes(DEGREE)
    assert np.allclose(expected, [-1.0, -np.sqrt(3 / 7), 0.0, np.sqrt(3 / 7), 1.0])
    r, s = reference.r, reference.s
    face_coordinates = (
        r[reference.face_nodes[0]],
        r[reference.face_nodes[1]],
        s[reference.face_nodes[2]],
    )
    for coordinate in face_coordinates:
        assert np.allclose(np.sort(coordinate), expected, atol=1e-14)
    assert np.allclose(s[reference.face_nodes[0]], -1.0)
    assert np.allclose((r + s)[reference.face_nodes[1]], 0.0, atol=1e-14)
    assert np.allclose(r[reference.face_nodes[2]], -1.0)


def test_face_midpoint_nodes(reference):
    r, s = reference.r, reference.s
    midpoints = np.stack([r[reference.face_midpoint_nodes], s[reference.face_midpoint_nodes]], axis=-1)
    assert np.allclose(midpoints, [[0.0, -1.0], [0.0, 0.0], [-1.0, 0.0]], atol=1e-14)


def test_differentiation_is_exact_for_degree_four(reference):
    r, s = reference.r, reference.s
    worst = 0.0
    for i in range(DEGREE + 1):
        for j in range(DEGREE + 1 - i):
            f = r**i * s**j
            dr = i * r ** max(i - 1, 0) * s**j if i else np.zeros_like(r)
            ds = j * r**i * s ** max(j - 1, 0) if j else np.zeros_like(r)
            worst = max(worst, np.abs(reference.dr @ f - dr).max(), np.abs(reference.ds @ f - ds).max())
    assert worst < 1e-12


def test_mass_matrix_integrates_exactly(reference):
    """Cell weights integrate degree-four polynomials over the reference
    triangle exactly (ADR 0008)."""
    r, s = reference.r, reference.s
    assert reference.cell_weights.sum() == pytest.approx(2.0, abs=1e-13)
    for i in range(DEGREE + 1):
        for j in range(DEGREE + 1 - i):
            # Exact integral of r^i s^j over the reference triangle.
            exact = _monomial_integral(i, j)
            assert reference.cell_weights @ (r**i * s**j) == pytest.approx(exact, abs=1e-12)


def _monomial_integral(i: int, j: int) -> float:
    """Integral of r^i s^j over the triangle (-1,-1), (1,-1), (-1,1)."""
    from math import comb, factorial

    # Substitute r = -1 + 2a, s = -1 + 2b with a, b >= 0 and a + b <= 1.
    total = 0.0
    for m in range(i + 1):
        for n in range(j + 1):
            coefficient = comb(i, m) * comb(j, n) * (-1.0) ** (i + j - m - n) * 2.0 ** (m + n)
            # Integral of a^m b^n over the unit simplex, times the Jacobian 4.
            beta = factorial(m) * factorial(n) / factorial(m + n + 2)
            total += coefficient * beta * 4.0
    return total


def test_lift_matrix_reproduces_edge_weights(reference):
    """Integrating the lifted face data over the element reproduces the edge
    quadrature weights, so face integrals use the GLL nodes (ADR 0008)."""
    lifted = reference.cell_weights @ reference.lift
    assert np.allclose(lifted, reference.face_weights.reshape(-1), atol=1e-13)
    assert np.allclose(reference.face_weights.sum(axis=1), 2.0)


def test_warp_and_blend_improves_conditioning():
    """ADR 0003 selects warp-and-blend nodes over an equispaced lattice."""
    warp = np.linalg.cond(vandermonde_2d(DEGREE, *warp_blend_nodes(DEGREE)))
    equispaced = np.linalg.cond(vandermonde_2d(DEGREE, *equispaced_nodes(DEGREE)))
    assert warp < equispaced


def test_linear_midpoint_basis_is_a_partition_of_unity(reference):
    """The basis used by the ADR 0010 reconstruction is linear, equal to one
    at its own face midpoint and zero at the others."""
    basis = reference.linear_midpoint_basis
    assert np.allclose(basis.sum(axis=1), 1.0)
    assert np.allclose(basis[reference.face_midpoint_nodes], np.eye(3), atol=1e-14)


def test_p1_projection_preserves_linear_functions(reference):
    """The L2 projection onto linear polynomials is a projection."""
    r, s = reference.r, reference.s
    for field in (np.ones_like(r), r, s, 2.0 - 0.5 * r + 3.0 * s):
        expected = field[reference.face_midpoint_nodes]
        assert np.allclose(reference.p1_midpoint @ field, expected, atol=1e-12)
    # A quadratic is changed by the projection.
    assert not np.allclose(reference.p1_midpoint @ (r * r), (r * r)[reference.face_midpoint_nodes])
