"""Degree-four nodal reference triangle (ADRs 0003 and 0008).

Host-side NumPy construction of warp-and-blend nodes and the collocated nodal
DG operators of Hesthaven and Warburton, *Nodal Discontinuous Galerkin
Methods* (2008), sections 3.1-3.2, 6.1-6.2.

The reference triangle has vertices (-1, -1), (1, -1), and (-1, 1). Local face
``f`` joins local vertices ``f`` and ``f + 1 (mod 3)`` (ADR 0007).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma as gamma_function

import numpy as np

DEGREE = 4
NODE_TOLERANCE = 1e-10

# Optimized warp-and-blend parameters (Hesthaven and Warburton, table 6.1),
# indexed by polynomial degree.
_WARP_BLEND_ALPHA = (
    0.0, 0.0, 0.0, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
    1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258,
)


def jacobi_p(x, alpha: float, beta: float, n: int) -> np.ndarray:
    """Evaluate the orthonormal Jacobi polynomial P_n^(alpha, beta) at x."""
    x = np.asarray(x, dtype=np.float64)
    gamma0 = (
        2.0 ** (alpha + beta + 1) / (alpha + beta + 1)
        * gamma_function(alpha + 1) * gamma_function(beta + 1)
        / gamma_function(alpha + beta + 1)
    )
    p_previous = np.full_like(x, 1.0 / np.sqrt(gamma0))
    if n == 0:
        return p_previous
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    p = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)
    a_old = 2 / (2 + alpha + beta) * np.sqrt(
        (alpha + 1) * (beta + 1) / (alpha + beta + 3)
    )
    for i in range(1, n):
        h1 = 2 * i + alpha + beta
        a_new = 2 / (h1 + 2) * np.sqrt(
            (i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta)
            / (h1 + 1) / (h1 + 3)
        )
        b_new = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        p_previous, p = p, (-a_old * p_previous + (x - b_new) * p) / a_new
        a_old = a_new
    return p


def grad_jacobi_p(x, alpha: float, beta: float, n: int) -> np.ndarray:
    """Derivative of the orthonormal Jacobi polynomial P_n^(alpha, beta)."""
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.zeros_like(x)
    return np.sqrt(n * (n + alpha + beta + 1)) * jacobi_p(x, alpha + 1, beta + 1, n - 1)


def jacobi_gauss(alpha: float, beta: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Jacobi nodes and weights with n + 1 points (Golub-Welsch)."""
    if n == 0:
        return np.array([(alpha - beta) / (alpha + beta + 2)]), np.array([2.0])
    h1 = 2.0 * np.arange(n + 1) + alpha + beta
    k = np.arange(1, n + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        diagonal = -0.5 * (alpha**2 - beta**2) / (h1 + 2) / h1
    if alpha + beta < 10 * np.finfo(np.float64).eps:
        diagonal[0] = 0.0
    off_diagonal = 2 / (h1[:n] + 2) * np.sqrt(
        k * (k + alpha + beta) * (k + alpha) * (k + beta) / (h1[:n] + 1) / (h1[:n] + 3)
    )
    matrix = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes, vectors = np.linalg.eigh(matrix)
    weights = (
        vectors[0, :] ** 2 * 2 ** (alpha + beta + 1) / (alpha + beta + 1)
        * gamma_function(alpha + 1) * gamma_function(beta + 1)
        / gamma_function(alpha + beta + 1)
    )
    return nodes, weights


def gauss_lobatto_nodes(n: int) -> np.ndarray:
    """The n + 1 Gauss-Lobatto-Legendre nodes on [-1, 1], ascending."""
    if n == 1:
        return np.array([-1.0, 1.0])
    interior, _ = jacobi_gauss(1.0, 1.0, n - 2)
    return np.concatenate(([-1.0], interior, [1.0]))


def vandermonde_1d(n: int, r) -> np.ndarray:
    """One-dimensional Vandermonde matrix of the orthonormal Legendre basis."""
    r = np.asarray(r, dtype=np.float64)
    return np.stack([jacobi_p(r, 0.0, 0.0, j) for j in range(n + 1)], axis=-1)


def _warp_factor(n: int, r_out: np.ndarray) -> np.ndarray:
    """One-dimensional edge warp from equispaced to GLL nodes, divided by the
    blend factor (Hesthaven and Warburton, ``Warpfactor``)."""
    lgl = gauss_lobatto_nodes(n)
    r_equispaced = np.linspace(-1.0, 1.0, n + 1)
    v_equispaced = vandermonde_1d(n, r_equispaced)
    p_matrix = np.stack([jacobi_p(r_out, 0.0, 0.0, i) for i in range(n + 1)])
    lagrange = np.linalg.solve(v_equispaced.T, p_matrix)
    warp = lagrange.T @ (lgl - r_equispaced)
    interior = (np.abs(r_out) < 1.0 - 1.0e-10).astype(np.float64)
    scale = 1.0 - (interior * r_out) ** 2
    return warp / scale + warp * (interior - 1.0)


def _equilateral_to_reference(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map the equilateral triangle used by ``Nodes2D`` to (r, s) (``xytors``)."""
    l1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    l2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    l3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    return -l2 + l3 - l1, -l2 - l3 + l1


def _barycentric_lattice(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l1, l3 = [], []
    for i in range(n + 1):
        for m in range(n + 1 - i):
            l1.append(i / n)
            l3.append(m / n)
    l1 = np.array(l1)
    l3 = np.array(l3)
    return l1, 1.0 - l1 - l3, l3


def warp_blend_nodes(n: int = DEGREE) -> tuple[np.ndarray, np.ndarray]:
    """Warp-and-blend nodes (r, s) on the reference triangle (``Nodes2D``).

    Nodes are ordered row by row in s and by increasing r within each row.
    """
    alpha = _WARP_BLEND_ALPHA[n] if n < len(_WARP_BLEND_ALPHA) else 5.0 / 3.0
    l1, l2, l3 = _barycentric_lattice(n)
    x = -l2 + l3
    y = (-l2 - l3 + 2 * l1) / np.sqrt(3.0)
    warp1 = 4 * l2 * l3 * _warp_factor(n, l3 - l2) * (1 + (alpha * l1) ** 2)
    warp2 = 4 * l1 * l3 * _warp_factor(n, l1 - l3) * (1 + (alpha * l2) ** 2)
    warp3 = 4 * l1 * l2 * _warp_factor(n, l2 - l1) * (1 + (alpha * l3) ** 2)
    x = x + warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    y = y + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3
    return _equilateral_to_reference(x, y)


def equispaced_nodes(n: int = DEGREE) -> tuple[np.ndarray, np.ndarray]:
    """Equispaced lattice nodes (r, s) with the same ordering as the
    warp-and-blend nodes."""
    l1, _, l3 = _barycentric_lattice(n)
    return 2 * l3 - 1, 2 * l1 - 1


def _rs_to_ab(r: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapsed coordinates (a, b) of the reference point (r, s)."""
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    top = np.abs(1.0 - s) < 1e-12
    denominator = np.where(top, 1.0, 1.0 - s)
    a = np.where(top, -1.0, 2.0 * (1.0 + r) / denominator - 1.0)
    return a, s


def mode_indices(n: int = DEGREE) -> list[tuple[int, int]]:
    """Orthonormal basis modes (i, j), in Vandermonde column order."""
    return [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]


def _simplex_2d_p(a, b, i: int, j: int) -> np.ndarray:
    return np.sqrt(2.0) * jacobi_p(a, 0.0, 0.0, i) * jacobi_p(b, 2 * i + 1.0, 0.0, j) * (1 - b) ** i


def _grad_simplex_2d_p(a, b, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    fa = jacobi_p(a, 0.0, 0.0, i)
    dfa = grad_jacobi_p(a, 0.0, 0.0, i)
    gb = jacobi_p(b, 2 * i + 1.0, 0.0, j)
    dgb = grad_jacobi_p(b, 2 * i + 1.0, 0.0, j)
    half_one_minus_b = 0.5 * (1 - b)
    d_dr = dfa * gb
    d_ds = dfa * (gb * (0.5 * (1 + a)))
    if i > 0:
        d_dr = d_dr * half_one_minus_b ** (i - 1)
        d_ds = d_ds * half_one_minus_b ** (i - 1)
    tmp = dgb * half_one_minus_b**i
    if i > 0:
        tmp = tmp - 0.5 * i * gb * half_one_minus_b ** (i - 1)
    d_ds = d_ds + fa * tmp
    scale = 2.0 ** (i + 0.5)
    return scale * d_dr, scale * d_ds


def vandermonde_2d(n: int, r, s) -> np.ndarray:
    """Vandermonde matrix of the orthonormal Koornwinder-Dubiner basis."""
    a, b = _rs_to_ab(r, s)
    return np.stack([_simplex_2d_p(a, b, i, j) for i, j in mode_indices(n)], axis=-1)


def grad_vandermonde_2d(n: int, r, s) -> tuple[np.ndarray, np.ndarray]:
    """Gradient Vandermonde matrices (d/dr, d/ds) of the orthonormal basis."""
    a, b = _rs_to_ab(r, s)
    gradients = [_grad_simplex_2d_p(a, b, i, j) for i, j in mode_indices(n)]
    return (
        np.stack([g[0] for g in gradients], axis=-1),
        np.stack([g[1] for g in gradients], axis=-1),
    )


@dataclass(frozen=True, eq=False)
class ReferenceTriangle:
    """Nodal reference element and its collocated DG operators.

    Attributes:
        degree: Polynomial degree N.
        r, s: Reference node coordinates, shape (Np,).
        vandermonde: Orthonormal-basis Vandermonde matrix V, shape (Np, Np).
        mass: Mass matrix (V V^T)^-1.
        dr, ds: Nodal differentiation matrices.
        face_nodes: Node indices on each face, shape (3, Nfp); face 0 joins
            vertices 0-1 (s = -1), face 1 joins 1-2 (r + s = 0), face 2 joins
            2-0 (r = -1).
        lift: Lift matrix M^-1 E, shape (Np, 3 Nfp), for face data ordered as
            ``face_nodes.ravel()`` and edge integrals parametrized on [-1, 1].
        cell_weights: Integration weights 1^T M, shape (Np,), summing to the
            reference area 2.
        face_weights: Edge integration weights (GLL weights) for the face
            nodes, shape (3, Nfp).
        face_midpoint_nodes: Index of the node at each face midpoint, (3,).
        p1_midpoint: Values at the face midpoints of the L2 projection onto
            linear polynomials, shape (3, Np).
        linear_midpoint_basis: Values at the nodes of the linear functions
            equal to one at one face midpoint and zero at the others,
            shape (Np, 3).
    """

    degree: int
    r: np.ndarray
    s: np.ndarray
    vandermonde: np.ndarray
    mass: np.ndarray
    dr: np.ndarray
    ds: np.ndarray
    face_nodes: np.ndarray
    lift: np.ndarray
    cell_weights: np.ndarray
    face_weights: np.ndarray
    face_midpoint_nodes: np.ndarray
    p1_midpoint: np.ndarray
    linear_midpoint_basis: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.r.size

    @property
    def num_face_nodes(self) -> int:
        return self.face_nodes.shape[1]


def build_reference_triangle(n: int = DEGREE) -> ReferenceTriangle:
    """Construct the nodal reference triangle of degree n."""
    r, s = warp_blend_nodes(n)
    v = vandermonde_2d(n, r, s)
    v_inverse = np.linalg.inv(v)
    mass = v_inverse.T @ v_inverse
    vr, vs = grad_vandermonde_2d(n, r, s)
    dr = vr @ v_inverse
    ds = vs @ v_inverse

    face_masks = (
        np.abs(s + 1) < NODE_TOLERANCE,
        np.abs(r + s) < NODE_TOLERANCE,
        np.abs(r + 1) < NODE_TOLERANCE,
    )
    face_nodes = np.stack([np.flatnonzero(mask) for mask in face_masks])
    if face_nodes.shape != (3, n + 1):
        raise RuntimeError("Reference nodes do not have n + 1 nodes on every face.")

    face_coordinates = (r[face_nodes[0]], r[face_nodes[1]], s[face_nodes[2]])
    num_face_nodes = n + 1
    edge_matrix = np.zeros((r.size, 3 * num_face_nodes))
    face_weights = np.zeros((3, num_face_nodes))
    for face, coordinate in enumerate(face_coordinates):
        v1d = vandermonde_1d(n, coordinate)
        edge_mass = np.linalg.inv(v1d @ v1d.T)
        columns = slice(face * num_face_nodes, (face + 1) * num_face_nodes)
        edge_matrix[face_nodes[face], columns] = edge_mass
        face_weights[face] = edge_mass.sum(axis=1)
    lift = v @ (v.T @ edge_matrix)

    face_midpoint_nodes = face_nodes[:, n // 2]
    linear_modes = [k for k, (i, j) in enumerate(mode_indices(n)) if i + j <= 1]
    p1_projection = v[:, linear_modes] @ v_inverse[linear_modes, :]
    p1_midpoint = p1_projection[face_midpoint_nodes, :]
    linear_midpoint_basis = np.stack([-s, 1.0 + r + s, -r], axis=-1)

    return ReferenceTriangle(
        degree=n,
        r=r,
        s=s,
        vandermonde=v,
        mass=mass,
        dr=dr,
        ds=ds,
        face_nodes=face_nodes,
        lift=lift,
        cell_weights=mass.sum(axis=0),
        face_weights=face_weights,
        face_midpoint_nodes=face_midpoint_nodes,
        p1_midpoint=p1_midpoint,
        linear_midpoint_basis=linear_midpoint_basis,
    )
