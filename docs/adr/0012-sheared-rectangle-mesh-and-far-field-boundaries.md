# ADR 0012: Sheared rectangle meshes and exact far-field boundaries

- Status: Proposed
- Date: 2026-11-02
- Approval: Exact far-field states approved by the user, 2026-11-02. Sheared mesh and ADR 0007 amendment pending.

## Context

ADR 0011 needs a domain with an inclined lower wall and two boundary types
not covered by ADR 0007: a far field where a shock crosses the boundary, and
a supersonic outflow. ADR 0007 has two limits that block this:

- Mesh generation is limited to "only a structured rectangle generator."
- Boundary conditions are the `SlipWall` and `FixedState` types, and the
  ADR 0010 ghost-cell rule covers only those two.

Any inclined wall is also a polygon edge that is not axis-aligned, and the
rectangle generator produces only axis-aligned tags.

## Decision

- **Sheared rectangle mesh:** Add a generator that applies the affine map
  \((x,y)\mapsto(x,\,y+x\tan\theta)\) to `rectangle_mesh(nx, ny, x_range,
  y_range)`. The map is applied to vertices only, so the element connectivity,
  the consistent lower-left to upper-right diagonal, and the boundary tags
  `left`, `right`, `bottom`, and `top` are unchanged. Elements remain
  straight-sided affine triangles (ADR 0007), and the host-side normals and
  edge lengths are computed from the transformed vertices. The tag `bottom` is
  the inclined wall; `top` is the sheared upper boundary.
- **Amends ADR 0007:** On approval, the "Mesh generation" bullet is extended
  to these two generators, and the "only" restriction is removed for this
  affine case. General unstructured meshes remain out of scope.
- **Exact far-field boundary:** Add `ExactState`, a boundary condition whose
  exterior state at each boundary node is the exact piecewise-constant solution
  supplied by a function of physical coordinates. The function is evaluated at
  each face node once, when the solver is built, giving per-node exterior
  states; the current per-face `(K, 3, 4)` state array is generalized to
  `(K, 3, Nfp, 4)`. For the oblique case, nodes above the shock line use the
  post-shock state, and nodes below use the upstream state. It supplies only the exterior state of the
  numerical flux (ADR 0004); it never modifies interior values, matching ADR
  0007's rule.
- **Ghost cells (amends ADR 0010):** For an `ExactState` face, the ghost-cell
  average is the exact cell average of the state function, evaluated over the
  reflected element. Slip-wall and fixed-state ghost rules are unchanged.
- **Scope:** The `ExactState` type is for verification with a known analytic
  solution. It is not a general outflow condition.

## Alternatives considered

- **Supersonic extrapolation (transmissive) outflow:** Requires supersonic
  outflow, which is not guaranteed at the outlet for an attached shock. Its
  failure would show up only in the verification run. Rejected as the
  default, and left available as a future decision.
- **Constant `FixedState` on the far field:** Wrong where the shock crosses
  the boundary (see ADR 0011).
- **Gmsh or meshio-based mesh import:** Handles arbitrary geometry, but adds
  dependencies not needed for the sheared rectangle. ADR 0007 excludes them.
- **Trapezoidal domain with a horizontal top:** Requires a new generator,
  rather than a single affine map of the existing rectangle generator.

## Consequences

- The sheared mesh reuses the tested rectangle generator and the diagonal
  convention, with a single affine transformation.
- `ExactState` depends on an analytic reference, so it is restricted to
  verification cases. Its use in production cases would need its own decision.
- The shock is captured across the boundary on the sheared top, and the far-field
  states are exact, so reflections from the far boundary are excluded by
  construction. The test checks that they do not affect the interior.
- Validation: element and face counts, counterclockwise orientation of the
  sheared triangles, coincident paired face nodes, and that a zero-slope shear
  reproduces `rectangle_mesh` exactly.

## References

- ADR 0007, ADR 0010, ADR 0011.
