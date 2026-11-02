"""Collocated strong-form nodal DG operator for the Euler equations (ADR 0008).

Solutions are stored as nodal values with shape (Np, K, 4): node, element,
conserved variable. On each element the semi-discrete operator is

    du/dt = -(Dr F~ + Ds G~) + LIFT (Fscale (F(u-) . n - F^)),

with contravariant fluxes F~ = rx F + ry G and G~ = sx F + sy G (Hesthaven and
Warburton, chapter 6). Boundary conditions only supply the exterior state of
the interface flux (ADR 0007).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .boundary import FIXED_STATE, SLIP_WALL, BoundaryCondition
from .euler import cartesian_fluxes
from .fluxes import InterfaceFlux
from .mesh import MeshConnectivity, ElementGeometry, TriangleMesh, node_coordinates, pair_face_nodes
from .reference_element import ReferenceTriangle


class DGOperators(NamedTuple):
    """Device arrays of the DG operator (a JAX pytree).

    Face-node arrays have shape (3 Nfp, K) with faces ordered as
    ``ReferenceTriangle.face_nodes.ravel()``.
    """

    differentiation: jax.Array  # (Np, 2 Np): [Dr | Ds]
    lift: jax.Array  # (Np, 3 Nfp)
    face_nodes: jax.Array  # (3 Nfp,)
    inverse_metric: jax.Array  # (4, K): rx, ry, sx, sy
    normal_x: jax.Array  # (3 Nfp, K)
    normal_y: jax.Array  # (3 Nfp, K)
    face_scale: jax.Array  # (3 Nfp, K): sJ / J
    exterior_index: jax.Array  # (3 Nfp, K): flat node-major index node * K + element
    boundary_kind: jax.Array  # (3 Nfp, K): 0 interior, SLIP_WALL, or FIXED_STATE
    boundary_state: jax.Array  # (3 Nfp, K, 4): prescribed exterior states
    boundary_weight: jax.Array  # (3 Nfp, K): edge quadrature weight * sJ on boundary faces
    cell_weights: jax.Array  # (Np,): reference integration weights
    jacobian: jax.Array  # (K,)


def resolve_boundary_conditions(
    connectivity: MeshConnectivity, boundary_conditions: Mapping[str, BoundaryCondition]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face boundary kinds (K, 3) and fixed exterior states (K, 3, 4)."""
    tags = set(connectivity.boundary_tags)
    missing = sorted(tags - set(boundary_conditions))
    if missing:
        raise ValueError(f"Boundary tags without a boundary condition: {missing}.")
    unknown = sorted(set(boundary_conditions) - tags)
    if unknown:
        raise ValueError(f"Boundary conditions for tags that are not in the mesh: {unknown}.")
    kind = np.zeros(connectivity.face_tag.shape, dtype=np.int32)
    state = np.zeros(connectivity.face_tag.shape + (4,))
    for tag_index, tag in enumerate(connectivity.boundary_tags):
        condition = boundary_conditions[tag]
        mask = connectivity.face_tag == tag_index
        kind[mask] = condition.kind
        if condition.kind == FIXED_STATE:
            state[mask] = condition.state
        elif condition.kind != SLIP_WALL:
            raise ValueError(f"Unsupported boundary condition for tag {tag!r}: {condition!r}.")
    return kind, state


def build_dg_operators(
    mesh: TriangleMesh,
    connectivity: MeshConnectivity,
    geometry: ElementGeometry,
    reference: ReferenceTriangle,
    boundary_kind: np.ndarray,
    boundary_state: np.ndarray,
) -> DGOperators:
    """Assemble the DG operator arrays on the host and move them to JAX."""
    num_elements = mesh.num_elements
    num_face_nodes = reference.num_face_nodes
    x, y = node_coordinates(mesh, reference.r, reference.s)
    partner = pair_face_nodes(mesh, connectivity, x, y, reference.face_nodes)  # (3, Nfp, K)
    own = np.arange(num_elements)[:, None]
    neighbor = np.where(connectivity.neighbor_element >= 0, connectivity.neighbor_element, own)  # (K, 3)
    exterior_index = partner * num_elements + neighbor.T[:, None, :]

    def per_face_node(values: np.ndarray) -> np.ndarray:
        """Broadcast (K, 3, ...) face data to (3 Nfp, K, ...)."""
        values = np.moveaxis(values, 1, 0)[:, None]
        shape = (3, num_face_nodes) + values.shape[2:]
        return np.broadcast_to(values, shape).reshape((3 * num_face_nodes,) + values.shape[2:])

    face_scale = 0.5 * geometry.edge_length / geometry.jacobian[:, None]
    is_boundary = per_face_node(boundary_kind) > 0
    edge_weight = reference.face_weights.reshape(-1)[:, None] * per_face_node(0.5 * geometry.edge_length)
    return DGOperators(
        differentiation=jnp.asarray(np.concatenate([reference.dr, reference.ds], axis=1)),
        lift=jnp.asarray(reference.lift),
        face_nodes=jnp.asarray(reference.face_nodes.reshape(-1)),
        inverse_metric=jnp.asarray(np.stack([geometry.rx, geometry.ry, geometry.sx, geometry.sy])),
        normal_x=jnp.asarray(per_face_node(geometry.normal[..., 0])),
        normal_y=jnp.asarray(per_face_node(geometry.normal[..., 1])),
        face_scale=jnp.asarray(per_face_node(face_scale)),
        exterior_index=jnp.asarray(exterior_index.reshape(3 * num_face_nodes, num_elements)),
        boundary_kind=jnp.asarray(per_face_node(boundary_kind)),
        boundary_state=jnp.asarray(per_face_node(boundary_state)),
        boundary_weight=jnp.asarray(np.where(is_boundary, edge_weight, 0.0)),
        cell_weights=jnp.asarray(reference.cell_weights),
        jacobian=jnp.asarray(geometry.jacobian),
    )


def mirror_state(q, nx, ny):
    """Slip-wall mirror state: reverse the normal momentum."""
    normal_momentum = q[..., 1] * nx + q[..., 2] * ny
    zeros = jnp.zeros_like(normal_momentum)
    correction = jnp.stack(
        [zeros, 2.0 * normal_momentum * nx, 2.0 * normal_momentum * ny, zeros], axis=-1
    )
    return q - correction


def exterior_states(q_minus, q_neighbor, operators: DGOperators):
    """Exterior face states: neighbor traces, or boundary states."""
    kind = operators.boundary_kind[..., None]
    wall = mirror_state(q_minus, operators.normal_x, operators.normal_y)
    return jnp.where(
        kind == SLIP_WALL,
        wall,
        jnp.where(kind == FIXED_STATE, operators.boundary_state, q_neighbor),
    )


def dg_rhs(q, operators: DGOperators, flux: InterfaceFlux, gamma: float):
    """Semi-discrete right-hand side and net outward boundary flux.

    Returns ``(rhs, boundary_flux)`` where ``rhs`` has the shape of ``q`` and
    ``boundary_flux`` (4,) is the outward boundary integral of the numerical
    flux, computed with the same face-node fluxes and weights as the update.
    """
    num_variables = q.shape[-1]
    f, g = cartesian_fluxes(q, gamma)
    rx, ry, sx, sy = (row[None, :, None] for row in operators.inverse_metric)
    contravariant = jnp.concatenate([rx * f + ry * g, sx * f + sy * g], axis=0)
    divergence = jnp.einsum("ij,jkc->ikc", operators.differentiation, contravariant)

    nx = operators.normal_x
    ny = operators.normal_y
    q_minus = q[operators.face_nodes]
    q_neighbor = q.reshape(-1, num_variables)[operators.exterior_index]
    q_plus = exterior_states(q_minus, q_neighbor, operators)
    # The barrier stops XLA from recomputing the interface flux inside each of
    # its fused consumers, which costs about 30% of the operator.
    f_hat = lax.optimization_barrier(flux(q_minus, q_plus, nx, ny, gamma))
    f_minus = f[operators.face_nodes] * nx[..., None] + g[operators.face_nodes] * ny[..., None]
    jump = operators.face_scale[..., None] * (f_minus - f_hat)
    surface = jnp.einsum("ij,jkc->ikc", operators.lift, jump)
    boundary_flux = jnp.einsum("jk,jkc->c", operators.boundary_weight, f_hat)
    return surface - divergence, boundary_flux


def cell_averages(q, operators: DGOperators):
    """Exact cell averages, shape (K, 4)."""
    weights = operators.cell_weights / jnp.sum(operators.cell_weights)
    return jnp.einsum("i,ikc->kc", weights, q)


def domain_integral(q, operators: DGOperators):
    """Exact integral of each conserved variable over the mesh, shape (4,)."""
    return jnp.einsum("k,i,ikc->c", operators.jacobian, operators.cell_weights, q)
