"""Conforming triangle meshes with tagged boundaries (ADR 0007).

All mesh processing happens on the host in NumPy float64. Elements list their
vertices counterclockwise; local face ``f`` joins local vertices ``f`` and
``f + 1 (mod 3)``. Each element is the affine image of the reference triangle
with vertices (-1, -1), (1, -1), and (-1, 1), with local vertex ``i`` mapped to
reference vertex ``i``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

NUM_FACES = 3


def _read_only(array: np.ndarray) -> np.ndarray:
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class TriangleMesh:
    """Immutable conforming mesh of straight-sided triangles.

    Attributes:
        vertices: Vertex coordinates, shape (Nv, 2), float64.
        elements: Counterclockwise vertex indices per element, shape (K, 3).
        boundary_edges: Mapping from boundary tag to the vertex index pairs of
            the boundary edges carrying that tag, each of shape (n, 2).
    """

    vertices: np.ndarray
    elements: np.ndarray
    boundary_edges: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        elements = np.asarray(self.elements)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (Nv, 2).")
        if elements.ndim != 2 or elements.shape[1] != 3 or not np.issubdtype(elements.dtype, np.integer):
            raise ValueError("elements must be an integer array of shape (K, 3).")
        if elements.size and (elements.min() < 0 or elements.max() >= len(vertices)):
            raise ValueError("elements reference vertices that do not exist.")
        edges = {}
        for tag, pairs in self.boundary_edges.items():
            pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            edges[str(tag)] = _read_only(pairs)
        object.__setattr__(self, "vertices", _read_only(vertices))
        object.__setattr__(self, "elements", _read_only(elements.astype(np.int64)))
        object.__setattr__(self, "boundary_edges", MappingProxyType(edges))
        if np.any(signed_areas(self) <= 0.0):
            raise ValueError("Every element must have counterclockwise vertices and positive area.")

    @property
    def num_elements(self) -> int:
        return self.elements.shape[0]


def signed_areas(mesh: TriangleMesh) -> np.ndarray:
    """Signed element areas, positive for counterclockwise vertices."""
    p0, p1, p2 = (mesh.vertices[mesh.elements[:, i]] for i in range(3))
    return 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def rectangle_mesh(
    nx: int,
    ny: int,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
) -> TriangleMesh:
    """Structured mesh of ``nx * ny`` rectangles, each split along the diagonal
    from its lower-left to its upper-right corner.

    Boundary tags are ``left``, ``right``, ``bottom``, and ``top``. Elements
    of cell ``(i, j)`` (column ``i``, row ``j``) have indices
    ``2 * (j * nx + i)`` (lower-right triangle) and ``2 * (j * nx + i) + 1``
    (upper-left triangle).
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive.")
    (x0, x1), (y0, y1) = x_range, y_range
    xs = x0 + (x1 - x0) * (np.arange(nx + 1) / nx)
    ys = y0 + (y1 - y0) * (np.arange(ny + 1) / ny)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    def index(i, j):
        return j * (nx + 1) + i

    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = i.ravel(), j.ravel()
    v00, v10, v11, v01 = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
    lower = np.stack([v00, v10, v11], axis=-1)
    upper = np.stack([v00, v11, v01], axis=-1)
    elements = np.stack([lower, upper], axis=1).reshape(-1, 3)

    columns = np.arange(nx)
    rows = np.arange(ny)
    boundary_edges = {
        "bottom": np.stack([index(columns, 0), index(columns + 1, 0)], axis=-1),
        "top": np.stack([index(columns, ny), index(columns + 1, ny)], axis=-1),
        "left": np.stack([index(0, rows), index(0, rows + 1)], axis=-1),
        "right": np.stack([index(nx, rows), index(nx, rows + 1)], axis=-1),
    }
    return TriangleMesh(vertices=vertices, elements=elements, boundary_edges=boundary_edges)


@dataclass(frozen=True, eq=False)
class MeshConnectivity:
    """Face connectivity of a conforming mesh.

    Attributes:
        neighbor_element: Neighbor across each face, shape (K, 3); -1 on the
            boundary.
        neighbor_face: Local face index in the neighbor, shape (K, 3); -1 on
            the boundary.
        boundary_tags: Sorted boundary tag names.
        face_tag: Index into ``boundary_tags`` for boundary faces, shape
            (K, 3); -1 for interior faces.
    """

    neighbor_element: np.ndarray
    neighbor_face: np.ndarray
    boundary_tags: tuple[str, ...]
    face_tag: np.ndarray


def build_connectivity(mesh: TriangleMesh) -> MeshConnectivity:
    """Match faces by their sorted vertex pairs; reject nonconforming meshes.

    A face shared by more than two elements, a boundary face without a tag,
    or a tagged edge that is not a boundary face is an error. Hanging nodes
    produce untagged boundary faces and are therefore rejected.
    """
    num_elements = mesh.num_elements
    first = mesh.elements
    second = np.roll(mesh.elements, -1, axis=1)
    keys = np.sort(np.stack([first, second], axis=-1).reshape(-1, 2), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if np.any(counts > 2):
        raise ValueError("Nonconforming mesh: a face is shared by more than two elements.")

    order = np.argsort(inverse, kind="stable")
    sorted_ids = inverse[order]
    neighbor_element = -np.ones(num_elements * NUM_FACES, dtype=np.int64)
    neighbor_face = -np.ones(num_elements * NUM_FACES, dtype=np.int64)
    paired = np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])
    left, right = order[paired], order[paired + 1]
    neighbor_element[left], neighbor_face[left] = right // NUM_FACES, right % NUM_FACES
    neighbor_element[right], neighbor_face[right] = left // NUM_FACES, left % NUM_FACES

    tags = tuple(sorted(mesh.boundary_edges))
    lookup = {}
    for tag_index, tag in enumerate(tags):
        for a, b in mesh.boundary_edges[tag]:
            key = (min(a, b), max(a, b))
            if key in lookup:
                raise ValueError(f"Boundary edge {key} has more than one tag.")
            lookup[key] = tag_index

    face_tag = -np.ones(num_elements * NUM_FACES, dtype=np.int64)
    for face in np.flatnonzero(neighbor_element < 0):
        key = (int(keys[face, 0]), int(keys[face, 1]))
        if key not in lookup:
            element, local = divmod(int(face), NUM_FACES)
            raise ValueError(f"Boundary face {local} of element {element} (vertices {key}) has no tag.")
        face_tag[face] = lookup.pop(key)
    if lookup:
        raise ValueError(f"Tagged edges are not boundary faces: {sorted(lookup)[:5]}.")

    shape = (num_elements, NUM_FACES)
    return MeshConnectivity(
        neighbor_element=_read_only(neighbor_element.reshape(shape)),
        neighbor_face=_read_only(neighbor_face.reshape(shape)),
        boundary_tags=tags,
        face_tag=_read_only(face_tag.reshape(shape)),
    )


@dataclass(frozen=True, eq=False)
class ElementGeometry:
    """Affine element geometry in float64.

    Attributes:
        jacobian: Determinant of the reference-to-physical map, |K| / 2, (K,).
        rx, ry, sx, sy: Constant inverse-map derivatives, each (K,).
        centroid: Element barycenters, (K, 2).
        normal: Outward unit normals per face, (K, 3, 2).
        edge_length: Face lengths, (K, 3).
        face_midpoint: Face midpoints, (K, 3, 2).
    """

    jacobian: np.ndarray
    rx: np.ndarray
    ry: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    edge_length: np.ndarray
    face_midpoint: np.ndarray

    @property
    def area(self) -> np.ndarray:
        return 2.0 * self.jacobian


def element_geometry(mesh: TriangleMesh) -> ElementGeometry:
    """Geometric factors computed directly from vertex coordinates, so that
    both sides of an interior face have exactly opposite normals and equal
    lengths."""
    corners = mesh.vertices[mesh.elements]  # (K, 3, 2)
    x1, x2, x3 = corners[:, 0, 0], corners[:, 1, 0], corners[:, 2, 0]
    y1, y2, y3 = corners[:, 0, 1], corners[:, 1, 1], corners[:, 2, 1]
    xr, xs = 0.5 * (x2 - x1), 0.5 * (x3 - x1)
    yr, ys = 0.5 * (y2 - y1), 0.5 * (y3 - y1)
    jacobian = xr * ys - xs * yr

    start = corners
    end = np.roll(corners, -1, axis=1)
    dx = end[..., 0] - start[..., 0]
    dy = end[..., 1] - start[..., 1]
    edge_length = np.sqrt(dx * dx + dy * dy)
    normal = np.stack([dy / edge_length, -dx / edge_length], axis=-1)
    return ElementGeometry(
        jacobian=_read_only(jacobian),
        rx=_read_only(ys / jacobian),
        ry=_read_only(-xs / jacobian),
        sx=_read_only(-yr / jacobian),
        sy=_read_only(xr / jacobian),
        centroid=_read_only(corners.mean(axis=1)),
        normal=_read_only(normal),
        edge_length=_read_only(edge_length),
        face_midpoint=_read_only(0.5 * (start + end)),
    )


def node_coordinates(mesh: TriangleMesh, r: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates of reference nodes in every element, each (Np, K)."""
    corners = mesh.vertices[mesh.elements]  # (K, 3, 2)
    weights = np.stack([-(r + s) / 2, (1 + r) / 2, (1 + s) / 2], axis=-1)  # (Np, 3)
    points = np.einsum("nv,kvd->nkd", weights, corners)
    return points[..., 0], points[..., 1]


def pair_face_nodes(
    mesh: TriangleMesh,
    connectivity: MeshConnectivity,
    x: np.ndarray,
    y: np.ndarray,
    face_nodes: np.ndarray,
    relative_tolerance: float = 1e-9,
) -> np.ndarray:
    """Match face nodes of neighboring elements by physical coordinates.

    Returns ``partner`` of shape (3, Nfp, K): the local node index in the
    neighbor element coinciding with each face node, or the node itself on
    boundary faces.
    """
    num_faces, num_face_nodes = face_nodes.shape
    num_elements = mesh.num_elements
    partner = np.broadcast_to(face_nodes[:, :, None], (num_faces, num_face_nodes, num_elements)).copy()
    geometry_scale = element_geometry(mesh).edge_length
    for face in range(num_faces):
        elements = np.flatnonzero(connectivity.neighbor_element[:, face] >= 0)
        if elements.size == 0:
            continue
        neighbors = connectivity.neighbor_element[elements, face]
        neighbor_faces = connectivity.neighbor_face[elements, face]
        own_nodes = face_nodes[face]  # (Nfp,)
        candidate_nodes = face_nodes[neighbor_faces]  # (n, Nfp)
        own = np.stack([x[own_nodes][:, elements], y[own_nodes][:, elements]], axis=-1)  # (Nfp, n, 2)
        other = np.stack(
            [x[candidate_nodes.T, neighbors[None, :]], y[candidate_nodes.T, neighbors[None, :]]], axis=-1
        )  # (Nfp, n, 2)
        distance = np.linalg.norm(own[:, None, :, :] - other[None, :, :, :], axis=-1)  # (Nfp, Nfp, n)
        best = distance.argmin(axis=1)  # (Nfp, n)
        best_distance = np.take_along_axis(distance, best[:, None, :], axis=1)[:, 0, :]
        tolerance = relative_tolerance * geometry_scale[elements, face]
        if np.any(best_distance > tolerance[None, :]):
            raise ValueError("Neighboring face nodes do not coincide; the mesh is not conforming.")
        partner[face][:, elements] = np.take_along_axis(candidate_nodes.T, best, axis=0)
    return partner
