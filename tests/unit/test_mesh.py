"""Unit tests for the mesh, connectivity and geometry (ADR 0007)."""

from __future__ import annotations

import numpy as np
import pytest

from cfd_demo.boundary import FIXED_STATE, SLIP_WALL, FixedState, SlipWall
from cfd_demo.dg import resolve_boundary_conditions
from cfd_demo.mesh import (
    TriangleMesh,
    build_connectivity,
    element_geometry,
    node_coordinates,
    pair_face_nodes,
    rectangle_mesh,
    signed_areas,
)
from cfd_demo.reference_element import build_reference_triangle

FACE_VERTICES = ((0, 1), (1, 2), (2, 0))


@pytest.fixture(scope="module")
def reference():
    return build_reference_triangle()


def test_rectangle_mesh_counts_and_orientation():
    nx, ny = 4, 3
    mesh = rectangle_mesh(nx, ny, (0.0, 2.0), (0.0, 0.5))
    assert mesh.vertices.shape == ((nx + 1) * (ny + 1), 2)
    assert mesh.elements.shape == (2 * nx * ny, 3)
    areas = signed_areas(mesh)
    assert np.all(areas > 0.0)
    assert areas.sum() == pytest.approx(1.0)
    assert np.allclose(areas, 2.0 * 0.5 / (2 * nx * ny))
    assert sorted(mesh.boundary_edges) == ["bottom", "left", "right", "top"]
    assert len(mesh.boundary_edges["bottom"]) == nx
    assert len(mesh.boundary_edges["left"]) == ny


def test_rectangle_mesh_element_ordering_is_documented():
    """The documented column/row element indexing is what the Sod initial
    condition and the column diagnostics rely on."""
    nx, ny = 3, 2
    mesh = rectangle_mesh(nx, ny)
    geometry = element_geometry(mesh)
    for j in range(ny):
        for i in range(nx):
            lower, upper = 2 * (j * nx + i), 2 * (j * nx + i) + 1
            for element in (lower, upper):
                centroid = geometry.centroid[element]
                assert i / nx < centroid[0] < (i + 1) / nx
                assert j / ny < centroid[1] < (j + 1) / ny
            # The lower-right triangle sits to the right of the upper-left one.
            assert geometry.centroid[lower, 0] > geometry.centroid[upper, 0]


def test_mesh_rejects_invalid_input():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    clockwise = np.array([[0, 2, 1]])
    with pytest.raises(ValueError, match="counterclockwise"):
        TriangleMesh(vertices=vertices, elements=clockwise, boundary_edges={})
    with pytest.raises(ValueError, match="vertices that do not exist"):
        TriangleMesh(vertices=vertices, elements=np.array([[0, 1, 7]]), boundary_edges={})
    with pytest.raises(ValueError, match=r"shape \(Nv, 2\)"):
        TriangleMesh(vertices=np.zeros((3, 3)), elements=np.array([[0, 1, 2]]), boundary_edges={})
    with pytest.raises(ValueError, match=r"shape \(K, 3\)"):
        TriangleMesh(vertices=vertices, elements=np.zeros((1, 3)), boundary_edges={})
    with pytest.raises(ValueError, match="nx and ny must be positive"):
        rectangle_mesh(0, 2)


def test_mesh_arrays_are_read_only():
    mesh = rectangle_mesh(2, 2)
    with pytest.raises(ValueError):
        mesh.vertices[0, 0] = 1.0
    with pytest.raises(ValueError):
        mesh.elements[0, 0] = 1


def test_connectivity_is_symmetric_and_tagged():
    mesh = rectangle_mesh(3, 2)
    connectivity = build_connectivity(mesh)
    neighbor, face = connectivity.neighbor_element, connectivity.neighbor_face
    assert connectivity.boundary_tags == ("bottom", "left", "right", "top")

    interior = neighbor >= 0
    elements, faces = np.nonzero(interior)
    for element, local in zip(elements, faces, strict=True):
        other, other_face = neighbor[element, local], face[element, local]
        assert neighbor[other, other_face] == element
        assert face[other, other_face] == local
        own = {mesh.elements[element, v] for v in FACE_VERTICES[local]}
        assert own == {mesh.elements[other, v] for v in FACE_VERTICES[other_face]}

    boundary = ~interior
    assert np.all(face[boundary] == -1)
    assert np.all(connectivity.face_tag[boundary] >= 0)
    assert np.all(connectivity.face_tag[interior] == -1)
    # Four boundary faces per side of the 3 x 2 unit square.
    counts = np.bincount(connectivity.face_tag[boundary], minlength=4)
    assert counts.tolist() == [3, 2, 2, 3]


def test_connectivity_rejects_nonconforming_and_mistagged_meshes():
    mesh = rectangle_mesh(2, 1)
    partial = TriangleMesh(
        vertices=mesh.vertices,
        elements=mesh.elements,
        boundary_edges={k: v for k, v in mesh.boundary_edges.items() if k != "top"},
    )
    with pytest.raises(ValueError, match="has no tag"):
        build_connectivity(partial)

    interior_edge = np.array([[1, 4]])
    mistagged = TriangleMesh(
        vertices=mesh.vertices,
        elements=mesh.elements,
        boundary_edges={**mesh.boundary_edges, "bogus": interior_edge},
    )
    with pytest.raises(ValueError, match="not boundary faces"):
        build_connectivity(mistagged)

    doubled = TriangleMesh(
        vertices=mesh.vertices,
        elements=mesh.elements,
        boundary_edges={**mesh.boundary_edges, "extra": mesh.boundary_edges["top"]},
    )
    with pytest.raises(ValueError, match="more than one tag"):
        build_connectivity(doubled)

    # Three triangles meeting along one edge is not a conforming mesh.
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0], [0.5, 2.0]])
    elements = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    with pytest.raises(ValueError, match="more than two elements"):
        build_connectivity(TriangleMesh(vertices=vertices, elements=elements, boundary_edges={}))


def test_geometry_is_affine_and_consistent():
    mesh = rectangle_mesh(3, 2, (0.0, 1.5), (-1.0, 1.0))
    geometry = element_geometry(mesh)
    areas = signed_areas(mesh)
    assert np.allclose(geometry.jacobian, areas / 2.0)

    # The metric terms invert the affine map exactly.
    corners = mesh.vertices[mesh.elements]
    dxdr = 0.5 * (corners[:, 1] - corners[:, 0])
    dxds = 0.5 * (corners[:, 2] - corners[:, 0])
    for element in range(mesh.num_elements):
        jacobian_matrix = np.stack([dxdr[element], dxds[element]], axis=-1)
        inverse = np.array(
            [
                [geometry.rx[element], geometry.ry[element]],
                [geometry.sx[element], geometry.sy[element]],
            ]
        )
        assert np.allclose(inverse @ jacobian_matrix, np.eye(2), atol=1e-13)
    assert np.allclose(geometry.centroid, corners.mean(axis=1))


def test_face_normals_are_outward_unit_vectors():
    mesh = rectangle_mesh(3, 2)
    geometry = element_geometry(mesh)
    corners = mesh.vertices[mesh.elements]
    assert np.allclose(np.linalg.norm(geometry.normal, axis=-1), 1.0)
    for local, (a, b) in enumerate(FACE_VERTICES):
        edge = corners[:, b] - corners[:, a]
        assert np.allclose(geometry.edge_length[:, local], np.linalg.norm(edge, axis=-1))
        assert np.allclose(np.einsum("kd,kd->k", geometry.normal[:, local], edge), 0.0, atol=1e-13)
        outward = geometry.face_midpoint[:, local] - geometry.centroid
        assert np.all(np.einsum("kd,kd->k", geometry.normal[:, local], outward) > 0.0)
    # The divergence theorem on a constant field: the weighted normals cancel.
    weighted = (geometry.normal * geometry.edge_length[..., None]).sum(axis=1)
    assert np.allclose(weighted, 0.0, atol=1e-13)


def test_neighboring_faces_have_opposite_normals():
    mesh = rectangle_mesh(3, 2)
    connectivity = build_connectivity(mesh)
    geometry = element_geometry(mesh)
    elements, faces = np.nonzero(connectivity.neighbor_element >= 0)
    others = connectivity.neighbor_element[elements, faces]
    other_faces = connectivity.neighbor_face[elements, faces]
    assert np.allclose(geometry.normal[elements, faces], -geometry.normal[others, other_faces])
    assert np.allclose(geometry.edge_length[elements, faces], geometry.edge_length[others, other_faces])


def test_paired_face_nodes_coincide(reference):
    """GLL face nodes of neighboring elements must sit at the same physical
    points, which is what makes the collocated trace exchange exact."""
    mesh = rectangle_mesh(3, 2, (0.0, 2.0), (0.0, 1.0))
    connectivity = build_connectivity(mesh)
    x, y = node_coordinates(mesh, reference.r, reference.s)
    assert x.shape == (reference.num_nodes, mesh.num_elements)

    partner = pair_face_nodes(mesh, connectivity, x, y, reference.face_nodes)
    assert partner.shape == (3, reference.num_face_nodes, mesh.num_elements)
    for element in range(mesh.num_elements):
        for local in range(3):
            other = connectivity.neighbor_element[element, local]
            if other < 0:
                assert np.array_equal(partner[local, :, element], reference.face_nodes[local])
                continue
            own_nodes = reference.face_nodes[local]
            neighbor_nodes = partner[local, :, element]
            assert np.allclose(x[own_nodes, element], x[neighbor_nodes, other], atol=1e-12)
            assert np.allclose(y[own_nodes, element], y[neighbor_nodes, other], atol=1e-12)


def test_pair_face_nodes_rejects_a_distorted_mesh(reference):
    mesh = rectangle_mesh(2, 1)
    connectivity = build_connectivity(mesh)
    x, y = node_coordinates(mesh, reference.r, reference.s)
    moved = x.copy()
    moved[reference.face_nodes[1][1], 0] += 0.1
    with pytest.raises(ValueError, match="not conforming"):
        pair_face_nodes(mesh, connectivity, moved, y, reference.face_nodes)


def test_resolve_boundary_conditions():
    mesh = rectangle_mesh(2, 2)
    connectivity = build_connectivity(mesh)
    state = (1.0, 0.0, 0.0, 2.5)
    conditions = {
        "left": FixedState(state),
        "right": FixedState(state),
        "bottom": SlipWall(),
        "top": SlipWall(),
    }
    kind, values = resolve_boundary_conditions(connectivity, conditions)
    tags = connectivity.boundary_tags
    assert kind.shape == connectivity.face_tag.shape
    assert values.shape == connectivity.face_tag.shape + (4,)
    assert np.all(kind[connectivity.face_tag < 0] == 0)
    left = connectivity.face_tag == tags.index("left")
    bottom = connectivity.face_tag == tags.index("bottom")
    assert np.all(kind[left] == FIXED_STATE)
    assert np.all(kind[bottom] == SLIP_WALL)
    assert np.allclose(values[left], state)
    assert np.allclose(values[bottom], 0.0)

    with pytest.raises(ValueError, match="top"):
        resolve_boundary_conditions(connectivity, {k: v for k, v in conditions.items() if k != "top"})
    with pytest.raises(ValueError, match="outlet"):
        resolve_boundary_conditions(connectivity, {**conditions, "outlet": SlipWall()})
    with pytest.raises(ValueError, match="four finite conserved values"):
        FixedState((1.0, 0.0, 0.0))
