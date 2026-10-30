# ADR 0007: Represent conforming triangle meshes with tagged boundaries

- Status: Accepted
- Date: 2026-10-30
- Approval: User, 2026-10-30

## Context

ADR 0003 requires degree-four nodal elements on triangles, and ADR 0005
requires rectangle meshes of straight-sided triangles with slip walls at the
top and bottom and fixed exterior states at the ends. No accepted ADR defines
mesh data structures, element connectivity, or how boundary conditions attach
to faces. ADR 0005 requires boundary states to act only through the selected
numerical flux.

## Decision

- **Mesh data:** A mesh is an immutable host-side object with vertex
  coordinates (`(Nv, 2)`, float64) and element-to-vertex indices (`(K, 3)`),
  listing each triangle's vertices counterclockwise. Local face `f` joins local
  vertices `f` and `f + 1 (mod 3)`. Elements are straight-sided and mapped
  affinely from the reference triangle with vertices \((-1,-1)\), \((1,-1)\),
  and \((-1,1)\); local vertex `i` maps to reference vertex `i`.
- **Connectivity:** Build element-to-element and element-to-face connectivity
  on the host by matching the sorted vertex pairs of faces. Support only
  conforming meshes: each face belongs to at most two elements, and hanging
  nodes are rejected. Pair the face nodes of neighboring elements by matching
  physical coordinates; the GLL edge nodes of ADR 0003 make them coincide.
- **Geometry:** Compute Jacobians, outward unit normals, and edge lengths on
  the host in float64 from vertex coordinates, so both sides of an interior
  face use exactly opposite normals and identical lengths.
- **Boundary representation:** Every boundary face carries a string tag. The
  solver configuration maps each tag to a boundary condition; an untagged
  boundary face or a tag without a condition is an error. Initially provide:
  - `SlipWall`: the exterior state mirrors the interior trace, with the same
    density, total energy, and tangential momentum and reversed normal
    momentum.
  - `FixedState`: the exterior state is a prescribed constant conserved state.

  Both conditions only supply the exterior state of the selected numerical
  flux (ADR 0004); they never modify interior values.
- **Mesh generation:** Provide only a structured rectangle generator: `nx × ny`
  rectangles, each split into two triangles along the diagonal from its
  lower-left to its upper-right corner, with boundary tags `left`, `right`,
  `bottom`, and `top`. Importing general unstructured meshes is out of scope.

## Alternatives considered

- **Mesh files or external generators (for example, Gmsh with meshio):**
  Needed for general geometries later, but add dependencies the current cases
  do not need.
- **Ghost elements for boundaries:** Unify interior and boundary handling, but
  complicate connectivity and conservation bookkeeping.
- **Imposing the wall flux directly (pressure only):** Exact for an ideal wall,
  but bypasses the selected numerical flux that ADR 0005 asks boundaries to
  use.
- **Alternating or symmetric diagonals:** Reduce mesh-induced asymmetry, but
  ADR 0005 requires a consistent diagonal.

## Consequences

- The ADR 0005 Sod case can be meshed and tagged directly, and its walls and
  reservoirs use the selected interface flux.
- Coincident neighbor face nodes support exact discrete conservation across
  interior faces.
- The consistent diagonal makes the mesh asymmetric in \(y\), so small
  transverse velocities are expected and are reported.
- Curved elements, periodic boundaries, mesh import, and nonconforming meshes
  need future decisions.
- Validation: element and boundary-face counts, counterclockwise orientation,
  symmetric connectivity, and coincident paired face nodes.
