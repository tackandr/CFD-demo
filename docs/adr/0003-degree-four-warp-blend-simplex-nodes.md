# ADR 0003: Use degree-four warp-and-blend simplex nodes

- Status: Proposed
- Date: 2026-10-26
- Approval: Pending

## Context

The solver uses a Discontinuous Galerkin discretization on an unstructured
two-dimensional simplex mesh. Its nodal approximation requires a choice of
polynomial degree and nodal distribution. The project needs a distribution
that works on simplices and provides a prescribed one-dimensional node
distribution along element edges. Nodal placement also affects the numerical
properties of the interpolation and differentiation operators used to
construct the DG discretization.

## Decision

Use degree-four polynomial approximations (polynomial degree \(p=4\)) on
two-dimensional simplex elements (triangles). Construct the nodal points using
the warp-and-blend method. Along each edge, the points must follow the
one-dimensional degree-four Gauss-Lobatto-Legendre (GLL) distribution, with
five edge nodes including the vertices.

This decision specifies the polynomial degree and nodal construction. It does
not prescribe a particular basis normalization, quadrature rule, or treatment
of any future three-dimensional simplex elements.

## Alternatives considered

- **Equispaced simplex nodes:** A straightforward nodal distribution, but it
  does not provide the requested GLL distribution along edges. Hesthaven and
  Warburton discuss the importance of nodal placement to the conditioning of
  nodal DG operators; the equispaced distribution has less favorable
  differentiation-operator matrix properties than the warp-and-blend
  distribution for this use.
- **Other simplex nodal constructions:** Other distributions could be used,
  but warp-and-blend is selected to satisfy the requested GLL edge placement.

## Consequences

- Element approximations use degree-four polynomials and five nodes per edge,
  with shared edge positions following the one-dimensional degree-four GLL
  distribution.
- Implementations must map and order the warp-and-blend nodes consistently
  across neighboring elements so that their edge-node positions agree.
- Validation should check the degree-four node count, edge positions against
  the one-dimensional GLL nodes, and consistency under element vertex
  permutations.
- Compare the conditioning of the reference-element differentiation operators
  for the selected degree-four warp-and-blend nodes and degree-four equispaced
  nodes, using the same basis and matrix norm. Treat this as a numerical
  validation rather than assuming a conditioning value independent of basis
  and scaling.
- Basis normalization, quadrature, and three-dimensional extensions remain
  open for future decisions.

## References

- Jan S. Hesthaven and Tim Warburton, *Nodal Discontinuous Galerkin Methods:
  Algorithms, Analysis, and Applications*, Texts in Applied Mathematics 54,
  Springer, 2008.
