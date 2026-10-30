# ADR 0003: Use degree-four warp-and-blend simplex nodes

- Status: Accepted
- Date: 2026-10-30
- Approval: Andreas

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

## Consequences

- Element approximations use degree-four polynomials and five nodes per edge,
  with shared edge positions following the one-dimensional degree-four GLL
  distribution.
- Implementations must map and order the warp-and-blend nodes consistently
  across neighboring elements so that their edge-node positions agree.
- Validation should check the degree-four node count, edge positions against
  the one-dimensional GLL nodes, and consistency under element vertex
  permutations.
- Compare the conditioning of the reference-element Vandermonde matrices
  for the selected degree-four warp-and-blend nodes and degree-four equispaced
  nodes, using the same polynomial basis, normalization, and matrix norm.
  The full differentiation matrices are singular because constants have zero
  derivative, so their ordinary condition numbers cannot compare node sets.
- Basis normalization, quadrature, and three-dimensional extensions remain
  open for future decisions.

## References

- Jan S. Hesthaven and Tim Warburton, *Nodal Discontinuous Galerkin Methods:
  Algorithms, Analysis, and Applications*, Texts in Applied Mathematics 54,
  Springer, 2008.
