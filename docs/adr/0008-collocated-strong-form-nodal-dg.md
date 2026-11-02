# ADR 0008: Discretize space with collocated strong-form nodal DG

- Status: Accepted
- Date: 2026-10-30
- Approval: User, 2026-10-30

## Context

ADR 0003 fixes degree-four warp-and-blend nodes on triangles but leaves basis
normalization and quadrature open. ADR 0004 names the Rusanov, Roe, and HLLC
interface fluxes without defining their wave-speed estimates. ADR 0005
requires the spatial operators and quadrature to be decided before
implementation, and `CONTEXT.md` directs the numerical methods to follow
Hesthaven and Warburton's *Nodal Discontinuous Galerkin Methods*.

## Decision

- **Equations:** Solve the two-dimensional compressible Euler equations in
  conserved variables \((\rho,\rho u,\rho v,E)\) for a calorically perfect gas,
  \(p=(\gamma-1)(E-\rho(u^2+v^2)/2)\), with \(\gamma\) as a configuration
  parameter (1.4 for ADR 0005). There are no source terms yet.
- **Formulation:** Use the semi-discrete strong-form nodal DG method of
  Hesthaven and Warburton (chapter 6). On each element,
  \(\partial_t u_h=-\nabla\cdot I_N F(u_h)+M^{-1}\oint \ell\,(F(u_h^-)\cdot n-\hat F)\,ds\),
  where \(\hat F\) is the selected ADR 0004 interface flux and \(n\) is the
  outward unit normal.
- **Nodes, basis, and operators:** Generate the ADR 0003 nodes with Hesthaven
  and Warburton's `Nodes2D` construction and its optimized blending parameter
  for \(N=4\), \(\alpha=0.1001\). Build Vandermonde, mass, differentiation, and
  lift matrices with the orthonormal Koornwinder–Dubiner basis
  (`Simplex2DP`). Store the solution as nodal values.
- **Quadrature:** Use collocation without over-integration. Evaluate the
  physical flux at the solution nodes and represent it by its degree-four
  interpolant. Mass and face-mass matrices are exact for nodal polynomials,
  so face integrals use the numerical flux at the five GLL nodes of each edge.
  Cell averages and domain integrals use the exact mass matrix. Do not filter.
- **Geometry:** Use straight-sided affine elements (ADR 0007), with constant
  geometric factors per element.
- **Interface fluxes,** selected by the stable configuration names
  `rusanov`, `roe`, and `hllc` (ADR 0004):
  - `rusanov`: \(\tfrac12(F_L+F_R)\cdot n-\tfrac12\alpha(u_R-u_L)\), with
    \(\alpha\) the larger of \(|u\cdot n|+c\) over the two states.
  - `roe`: the Roe-averaged linearization in the normal direction, with
    Harten's entropy fix on the acoustic waves,
    \(\delta=0.1(|\tilde u\cdot n|+\tilde c)\).
  - `hllc`: Toro's HLLC flux with Einfeldt wave-speed estimates
    \(S_L=\min(u_L\cdot n-c_L,\tilde u\cdot n-\tilde c)\) and
    \(S_R=\max(u_R\cdot n+c_R,\tilde u\cdot n+\tilde c)\), where tildes denote
    Roe averages.
- **Conservation bookkeeping:** The operator also returns the net outward
  boundary integral of the numerical flux for each conserved variable. It uses
  the same face-node fluxes and face-mass weights as the update and therefore
  includes wall pressure forces.

## Alternatives considered

- **Over-integration (volume cubature with Gauss face quadrature, Hesthaven
  and Warburton section 6.6):** Reduces aliasing errors from the nonlinear
  flux and matches the Zhang–Xia–Shu positivity theory, but costs roughly three
  to four times more per stage and needs interpolated face values. A later ADR
  may adopt it if aliasing instabilities appear.
- **Weak form:** Equivalent with exact integration; the strong form with
  nodal collocation is the reference implementation in Hesthaven and
  Warburton.
- **Modal storage:** Convenient for limiting, but the project follows the
  nodal approach of ADR 0003 and `CONTEXT.md`.
- **Spectral filtering:** Stabilizes aliasing, but adds tunable dissipation in
  smooth regions; ADR 0010 handles shocks instead.

## Consequences

- The operator is conservative: interior face fluxes cancel between neighbors
  up to round-off, so the change in each domain integral equals the
  time-integrated boundary flux (ADR 0005 check 3).
- Collocation can cause aliasing errors for the nonlinear Euler flux. Limiting
  (ADR 0010) mitigates them near shocks; monitor smooth-flow robustness.
- The operator evaluates fluxes only at solution nodes, so the volume and face
  quadrature points of ADR 0005 check 1 are the solution nodes; face nodes are
  a subset.
- Validation: exactness of the reference operators for degree-four
  polynomials, free-stream preservation, discrete conservation, consistency and
  conservation of the interface fluxes, and high-order consistency for smooth
  fields.

## References

- Jan S. Hesthaven and Tim Warburton, *Nodal Discontinuous Galerkin Methods*,
  Springer, 2008.
- Eleuterio F. Toro, *Riemann Solvers and Numerical Methods for Fluid
  Dynamics*, 3rd ed., Springer, 2009.
