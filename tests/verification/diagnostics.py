"""Verification diagnostics for the ADR 0005 acceptance checks.

Test-side NumPy code (ADR 0006): it reads the solver's mesh, nodes, and
solution, but implements its own polynomial evaluation and quadrature instead
of reusing solver kernels.

The domain-averaged L1 error is the integral over the rectangle of the
absolute pointwise difference, divided by the domain area. Integration splits
every element at the exact wave positions, so the discontinuities of the
reference solution always fall on sub-element boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Points per direction of the collapsed Gauss rule on each sub-triangle.
QUADRATURE_POINTS = 10
# Relative tolerance for treating a breakpoint as interior to an element.
SPLIT_TOLERANCE = 1e-12


def monomial_powers(degree: int) -> np.ndarray:
    """Exponent pairs ``(i, j)`` with ``i + j <= degree``, shape (Np, 2)."""
    return np.array([(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)])


def monomial_matrix(r: np.ndarray, s: np.ndarray, degree: int) -> np.ndarray:
    """Monomial Vandermonde matrix ``r**i * s**j`` of shape (len(r), Np)."""
    powers = monomial_powers(degree)
    return r[:, None] ** powers[None, :, 0] * s[:, None] ** powers[None, :, 1]


def nodal_coefficients(state: np.ndarray, r: np.ndarray, s: np.ndarray, degree: int) -> np.ndarray:
    """Monomial coefficients of the nodal solution, shape (Np, K, nvars)."""
    vandermonde = monomial_matrix(r, s, degree)
    return np.linalg.solve(vandermonde, state.reshape(state.shape[0], -1)).reshape(state.shape)


def collapsed_gauss_rule(points: int = QUADRATURE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature on the reference triangle (-1,-1), (1,-1), (-1,1).

    Uses the Duffy map of a tensor Gauss-Legendre rule on the square; the
    weights sum to the reference area 2. Returns barycentric coordinates of
    shape (n, 3) and weights of shape (n,) normalized to a unit-area triangle.
    """
    nodes, weights = np.polynomial.legendre.leggauss(points)
    a, b = np.meshgrid(nodes, nodes, indexing="ij")
    wa, wb = np.meshgrid(weights, weights, indexing="ij")
    r = 0.5 * (1.0 + a) * (1.0 - b) - 1.0
    s = b
    jacobian = 0.25 * (1.0 - b)
    area_weights = (wa * wb * jacobian).ravel() / 2.0
    r, s = r.ravel(), s.ravel()
    barycentric = np.stack([-(r + s) / 2.0, (1.0 + r) / 2.0, (1.0 + s) / 2.0], axis=-1)
    return barycentric, area_weights


def clip_polygon(polygon: np.ndarray, bound: float, keep_below: bool) -> np.ndarray:
    """Clip a convex polygon by a vertical half-plane (Sutherland-Hodgman)."""
    sign = 1.0 if keep_below else -1.0
    distance = sign * (bound - polygon[:, 0])
    inside = distance >= 0.0
    if inside.all():
        return polygon
    if not inside.any():
        return np.empty((0, 2))
    result = []
    for index, point in enumerate(polygon):
        following = (index + 1) % len(polygon)
        if inside[index]:
            result.append(point)
        if inside[index] != inside[following]:
            fraction = distance[index] / (distance[index] - distance[following])
            result.append(point + fraction * (polygon[following] - point))
    return np.array(result)


def split_elements(
    vertices: np.ndarray, breakpoints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split triangles at vertical lines and fan-triangulate the pieces.

    Args:
        vertices: Element corner coordinates, shape (K, 3, 2).
        breakpoints: Sorted x coordinates of the reference discontinuities.

    Returns:
        ``(sub_triangles, parent)``: corner coordinates of shape (M, 3, 2) and
        the parent element index of each sub-triangle, shape (M,).
    """
    x_min = vertices[..., 0].min(axis=1)
    x_max = vertices[..., 0].max(axis=1)
    scale = np.maximum(x_max - x_min, SPLIT_TOLERANCE)
    interior = (breakpoints[None, :] > x_min[:, None] + SPLIT_TOLERANCE * scale[:, None]) & (
        breakpoints[None, :] < x_max[:, None] - SPLIT_TOLERANCE * scale[:, None]
    )
    needs_split = np.flatnonzero(interior.any(axis=1))

    sub_triangles = [vertices[np.setdiff1d(np.arange(len(vertices)), needs_split)]]
    parent = [np.setdiff1d(np.arange(len(vertices)), needs_split)]
    for element in needs_split:
        cuts = breakpoints[interior[element]]
        edges = np.concatenate(([-np.inf], cuts, [np.inf]))
        for low, high in zip(edges[:-1], edges[1:]):
            piece = vertices[element]
            if np.isfinite(low):
                piece = clip_polygon(piece, low, keep_below=False)
            if len(piece) and np.isfinite(high):
                piece = clip_polygon(piece, high, keep_below=True)
            if len(piece) < 3:
                continue
            fan = np.stack(
                [np.broadcast_to(piece[0], (len(piece) - 2, 2)), piece[1:-1], piece[2:]], axis=1
            )
            sub_triangles.append(fan)
            parent.append(np.full(len(fan), element))
    return np.concatenate(sub_triangles), np.concatenate(parent)


@dataclass(frozen=True, eq=False)
class QuadraturePoints:
    """Quadrature points covering the domain, grouped by parent element.

    Attributes:
        x, y: Physical coordinates, shape (M, n).
        weight: Integration weights, shape (M, n), summing to the domain area.
        parent: Parent element of each sub-triangle, shape (M,).
        reference: Reference coordinates (r, s) in the parent, shape (M, n, 2).
    """

    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    parent: np.ndarray
    reference: np.ndarray

    @property
    def area(self) -> float:
        return float(self.weight.sum())


def build_quadrature(
    mesh, breakpoints=(), points: int = QUADRATURE_POINTS
) -> QuadraturePoints:
    """Quadrature over the whole mesh, split at the given x coordinates."""
    corners = mesh.vertices[mesh.elements]
    breakpoints = np.sort(np.asarray(breakpoints, dtype=np.float64).reshape(-1))
    sub_triangles, parent = split_elements(corners, breakpoints)

    barycentric, weights = collapsed_gauss_rule(points)
    physical = np.einsum("qv,mvd->mqd", barycentric, sub_triangles)
    edge_a = sub_triangles[:, 1] - sub_triangles[:, 0]
    edge_b = sub_triangles[:, 2] - sub_triangles[:, 0]
    double_area = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    weight = np.abs(double_area)[:, None] * weights[None, :] / 2.0

    # Invert the affine map of the parent element to get reference coordinates.
    parent_corners = corners[parent]
    origin = parent_corners[:, 0]
    basis = np.stack([parent_corners[:, 1] - origin, parent_corners[:, 2] - origin], axis=-1)
    offsets = physical - origin[:, None, :]
    barycentric_parent = np.linalg.solve(basis[:, None, :, :], offsets[..., None])[..., 0]
    reference = 2.0 * barycentric_parent - 1.0
    return QuadraturePoints(
        x=physical[..., 0], y=physical[..., 1], weight=weight, parent=parent, reference=reference
    )


def evaluate_state(
    coefficients: np.ndarray, quadrature: QuadraturePoints, degree: int
) -> np.ndarray:
    """Evaluate the nodal polynomial at the quadrature points, shape (M, n, nvars)."""
    powers = monomial_powers(degree)
    r = quadrature.reference[..., 0]
    s = quadrature.reference[..., 1]
    basis = r[..., None] ** powers[None, None, :, 0] * s[..., None] ** powers[None, None, :, 1]
    return np.einsum("mqb,bmc->mqc", basis, coefficients[:, quadrature.parent, :])


def to_primitive(q: np.ndarray, gamma: float):
    """Primitive variables ``(rho, u, v, p)`` from conserved states."""
    rho = q[..., 0]
    u = q[..., 1] / rho
    v = q[..., 2] / rho
    p = (gamma - 1.0) * (q[..., 3] - 0.5 * (q[..., 1] * u + q[..., 2] * v))
    return rho, u, v, p


def l1_errors(
    values: dict[str, np.ndarray], reference: dict[str, np.ndarray], quadrature: QuadraturePoints
) -> dict[str, float]:
    """Domain-averaged L1 errors of each named field."""
    area = quadrature.area
    return {
        name: float(np.sum(quadrature.weight * np.abs(values[name] - reference[name])) / area)
        for name in values
    }


def element_integrals(
    values: np.ndarray, quadrature: QuadraturePoints, num_elements: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element integrals of ``values`` (M, n) and element areas."""
    integrals = np.bincount(
        quadrature.parent, weights=np.sum(quadrature.weight * values, axis=1), minlength=num_elements
    )
    areas = np.bincount(
        quadrature.parent, weights=quadrature.weight.sum(axis=1), minlength=num_elements
    )
    return integrals, areas


def column_density_averages(
    density: np.ndarray, quadrature: QuadraturePoints, mesh, nx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh-column density averages and the x coordinates of the faces between
    adjacent columns.

    Columns are the ``nx`` strips of the structured rectangle mesh; elements
    ``2 * (j * nx + i)`` and ``2 * (j * nx + i) + 1`` belong to column ``i``.
    """
    num_elements = len(mesh.elements)
    integrals, areas = element_integrals(density, quadrature, num_elements)
    column = (np.arange(num_elements) // 2) % nx
    column_integral = np.bincount(column, weights=integrals, minlength=nx)
    column_area = np.bincount(column, weights=areas, minlength=nx)
    x_low, x_high = mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()
    face_x = x_low + (x_high - x_low) * (np.arange(1, nx) / nx)
    return column_integral / column_area, face_x


def shock_location(
    column_averages: np.ndarray, face_x: np.ndarray, window: tuple[float, float]
) -> float:
    """Face with the greatest decrease in column-averaged density inside the
    window, as required by ADR 0005 check 2."""
    inside = (face_x >= window[0]) & (face_x <= window[1])
    if not inside.any():
        raise ValueError(f"No mesh column face lies in {window}.")
    drop = column_averages[:-1] - column_averages[1:]
    candidates = np.where(inside, drop, -np.inf)
    return float(face_x[int(np.argmax(candidates))])
