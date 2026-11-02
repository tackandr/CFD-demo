"""Troubled-cell slope limiting and nodal positivity limiting (ADR 0010).

Applied after every Runge-Kutta stage. Both steps preserve cell averages.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .boundary import FIXED_STATE, SLIP_WALL
from .dg import mirror_state
from .euler import pressure
from .limiters import LimiterFunction
from .mesh import ElementGeometry, MeshConnectivity
from .reference_element import ReferenceTriangle

# Cockburn-Shu neighbor-difference factor nu.
NEIGHBOR_FACTOR = 1.5
# Relative tolerance for troubled-cell detection.
DETECTION_TOLERANCE = 1e-10
# Conserved components used for detection: density and total energy.
DETECTION_COMPONENTS = (0, 3)
# Upper bound of the Zhang-Shu positivity floor epsilon.
POSITIVITY_FLOOR = 1e-13
BISECTION_STEPS = 60
_COEFFICIENT_TOLERANCE = 1e-12


class LimiterOperators(NamedTuple):
    """Device arrays for limiting (a JAX pytree); face arrays are (3, K)."""

    average_weights: jax.Array  # (Np,): cell-average weights
    midpoint_nodes: jax.Array  # (3,): face-midpoint node indices
    p1_midpoint: jax.Array  # (3, Np): P1-projection values at face midpoints
    linear_basis: jax.Array  # (Np, 3): linear midpoint basis at the nodes
    neighbor: jax.Array  # (3, K): neighbor element, or the element itself on boundaries
    neighbor_kind: jax.Array  # (3, K): 0 interior, SLIP_WALL, or FIXED_STATE
    normal_x: jax.Array  # (3, K)
    normal_y: jax.Array  # (3, K)
    fixed_state: jax.Array  # (3, K, 4)
    alpha: jax.Array  # (3, K)
    beta: jax.Array  # (3, K)
    partner_face: jax.Array  # (3, K): face j paired with face i


def neighbor_coefficients(
    connectivity: MeshConnectivity, geometry: ElementGeometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonnegative Cockburn-Shu coefficients for every face.

    For face i of element K0 with barycenter b0 and face midpoint m_i, find a
    face j != i and alpha, beta >= 0 with m_i - b0 = alpha (b_i - b0) +
    beta (b_j - b0). Boundary neighbors use the barycenter of b0 reflected
    across the face. Returns ``(alpha, beta, partner_face)``, each (K, 3).
    """
    centroid = geometry.centroid
    midpoint = geometry.face_midpoint  # (K, 3, 2)
    normal = geometry.normal
    neighbor = connectivity.neighbor_element
    interior = centroid[np.maximum(neighbor, 0)]
    offset = np.sum((midpoint - centroid[:, None, :]) * normal, axis=-1, keepdims=True)
    reflected = centroid[:, None, :] + 2.0 * offset * normal
    neighbor_centroid = np.where((neighbor >= 0)[..., None], interior, reflected)
    d = neighbor_centroid - centroid[:, None, :]
    target = midpoint - centroid[:, None, :]

    num_elements = centroid.shape[0]
    alpha = np.zeros((num_elements, 3))
    beta = np.zeros((num_elements, 3))
    partner = -np.ones((num_elements, 3), dtype=np.int64)
    for face in range(3):
        for shift in (1, 2):
            other = (face + shift) % 3
            di, dj, t = d[:, face], d[:, other], target[:, face]
            determinant = di[:, 0] * dj[:, 1] - di[:, 1] * dj[:, 0]
            safe = np.where(determinant == 0.0, 1.0, determinant)
            a = (t[:, 0] * dj[:, 1] - t[:, 1] * dj[:, 0]) / safe
            b = (di[:, 0] * t[:, 1] - di[:, 1] * t[:, 0]) / safe
            valid = (determinant != 0.0) & (a >= -_COEFFICIENT_TOLERANCE) & (b >= -_COEFFICIENT_TOLERANCE)
            choose = valid & (partner[:, face] < 0)
            alpha[choose, face] = np.maximum(a[choose], 0.0)
            beta[choose, face] = np.maximum(b[choose], 0.0)
            partner[choose, face] = other
    if np.any(partner < 0):
        element, face = np.argwhere(partner < 0)[0]
        raise ValueError(
            f"No nonnegative Cockburn-Shu neighbor coefficients for face {face} of element {element}."
        )
    return alpha, beta, partner


def build_limiter_operators(
    connectivity: MeshConnectivity,
    geometry: ElementGeometry,
    reference: ReferenceTriangle,
    boundary_kind: np.ndarray,
    boundary_state: np.ndarray,
) -> LimiterOperators:
    alpha, beta, partner = neighbor_coefficients(connectivity, geometry)
    own = np.arange(connectivity.neighbor_element.shape[0])[:, None]
    neighbor = np.where(connectivity.neighbor_element >= 0, connectivity.neighbor_element, own)
    weights = reference.cell_weights / reference.cell_weights.sum()
    return LimiterOperators(
        average_weights=jnp.asarray(weights),
        midpoint_nodes=jnp.asarray(reference.face_midpoint_nodes),
        p1_midpoint=jnp.asarray(reference.p1_midpoint),
        linear_basis=jnp.asarray(reference.linear_midpoint_basis),
        neighbor=jnp.asarray(neighbor.T),
        neighbor_kind=jnp.asarray(boundary_kind.T),
        normal_x=jnp.asarray(geometry.normal[..., 0].T),
        normal_y=jnp.asarray(geometry.normal[..., 1].T),
        fixed_state=jnp.asarray(np.moveaxis(boundary_state, 1, 0)),
        alpha=jnp.asarray(alpha.T),
        beta=jnp.asarray(beta.T),
        partner_face=jnp.asarray(partner.T),
    )


def _minmod(a, b):
    return jnp.where(a * b > 0.0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def neighbor_averages(averages, operators: LimiterOperators):
    """Averages of face neighbors or boundary ghost cells, shape (3, K, 4)."""
    interior = averages[operators.neighbor]
    wall = mirror_state(averages[None], operators.normal_x, operators.normal_y)
    kind = operators.neighbor_kind[..., None]
    return jnp.where(kind == SLIP_WALL, wall, jnp.where(kind == FIXED_STATE, operators.fixed_state, interior))


def troubled_cell_limit(q, averages, operators: LimiterOperators, phi: LimiterFunction):
    """Cockburn-Shu troubled-cell limiter, component-wise in conserved variables.

    Returns the limited solution and a boolean troubled-cell indicator (K,).
    """
    differences = neighbor_averages(averages, operators) - averages[None]
    paired = jnp.take_along_axis(differences, operators.partner_face[..., None], axis=0)
    scaled_neighbor = NEIGHBOR_FACTOR * (
        operators.alpha[..., None] * differences + operators.beta[..., None] * paired
    )

    trace = q[operators.midpoint_nodes] - averages[None]
    components = jnp.asarray(DETECTION_COMPONENTS)
    excess = jnp.abs(_minmod(trace, scaled_neighbor) - trace)[..., components]
    threshold = DETECTION_TOLERANCE * jnp.abs(averages[:, components])
    troubled = jnp.any(excess > threshold[None], axis=(0, 2))

    projected = jnp.einsum("fi,ikc->fkc", operators.p1_midpoint, q) - averages[None]
    nonzero = projected != 0.0
    ratio = scaled_neighbor / jnp.where(nonzero, projected, 1.0)
    deltas = jnp.where(nonzero, projected * jnp.minimum(1.0, phi(ratio)), 0.0)
    positive = jnp.sum(jnp.maximum(deltas, 0.0), axis=0)
    negative = jnp.sum(jnp.maximum(-deltas, 0.0), axis=0)
    theta_positive = jnp.where(positive > 0.0, jnp.minimum(1.0, negative / jnp.where(positive > 0.0, positive, 1.0)), 0.0)
    theta_negative = jnp.where(negative > 0.0, jnp.minimum(1.0, positive / jnp.where(negative > 0.0, negative, 1.0)), 0.0)
    deltas = theta_positive[None] * jnp.maximum(deltas, 0.0) - theta_negative[None] * jnp.maximum(-deltas, 0.0)
    linear = averages[None] + jnp.einsum("if,fkc->ikc", operators.linear_basis, deltas)
    return jnp.where(troubled[None, :, None], linear, q), troubled


def positivity_limit(q, averages, gamma: float):
    """Zhang-Shu scaling toward the cell average so that density and then
    pressure are at least epsilon at every solution node.

    Elements must have admissible averages; otherwise the step is rejected by
    the time integrator (ADR 0009).
    """
    density_average = averages[:, 0]
    pressure_average = pressure(averages, gamma)
    epsilon = jnp.minimum(POSITIVITY_FLOOR, jnp.minimum(density_average, pressure_average))
    deviation = q - averages[None]

    density_min = jnp.min(q[..., 0], axis=0)
    needs_density = density_min < epsilon
    span = jnp.where(needs_density, density_average - density_min, 1.0)
    theta = jnp.clip(jnp.where(needs_density, (density_average - epsilon) / span, 1.0), 0.0, 1.0)
    q = jnp.where(needs_density[None, :, None], averages[None] + theta[None, :, None] * deviation, q)

    needs_pressure = pressure(q, gamma) < epsilon[None]

    def scale_pressure(q):
        deviation = q - averages[None]

        def bisect(_, bounds):
            low, high = bounds
            middle = 0.5 * (low + high)
            admissible = pressure(averages[None] + middle[..., None] * deviation, gamma) >= epsilon[None]
            return jnp.where(admissible, middle, low), jnp.where(admissible, high, middle)

        low, _ = lax.fori_loop(0, BISECTION_STEPS, bisect, (jnp.zeros(q.shape[:2]), jnp.ones(q.shape[:2])))
        theta = jnp.min(jnp.where(needs_pressure, low, 1.0), axis=0)
        limited = averages[None] + theta[None, :, None] * deviation
        return jnp.where((theta < 1.0)[None, :, None], limited, q)

    return lax.cond(jnp.any(needs_pressure), scale_pressure, lambda q: q, q)


def apply_limiters(q, operators: LimiterOperators, phi: LimiterFunction, gamma: float, limit_slopes: bool):
    """Apply ADR 0010 limiting to a stage solution.

    Returns ``(q, averages, troubled_count)``; ``limit_slopes`` is False only
    for the ``unlimited`` limiter, which skips troubled-cell limiting.
    """
    averages = jnp.einsum("i,ikc->kc", operators.average_weights, q)
    if limit_slopes:
        q, troubled = troubled_cell_limit(q, averages, operators, phi)
        troubled_count = jnp.sum(troubled)
    else:
        troubled_count = jnp.zeros((), dtype=jnp.int32)
    q = positivity_limit(q, averages, gamma)
    return q, averages, troubled_count
