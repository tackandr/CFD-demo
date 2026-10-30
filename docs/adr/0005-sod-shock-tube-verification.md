# ADR 0005: Verify the solver with a two-dimensional Sod shock tube

- Status: Accepted
- Date: 2026-10-30
- Approval: User, 2026-10-30

## Context

The project needs a shock-containing verification case for its intended
two-dimensional compressible Euler solver. There is currently no solver, mesh
generator, or test infrastructure in this repository. This decision describes a
benchmark to implement after the necessary solver decisions are approved; it
does not authorize implementation of those missing components.

The requested tube has an inlet on the left and an outlet on the right. In the
standard Sod problem, however, both initial states are at rest. Before the
waves reach the ends of the tube, the exact flow at both ends remains at rest:
the case does not exercise sustained inlet-to-outlet flow or independently
verify an operating inflow/outflow boundary condition. The proposed treatment
uses fixed exterior reservoir states at the left and right ends and a numerical
flux, rather than claiming that flow is entering or leaving there.

This case uses the Python/JAX framework of ADR 0002 and the degree-four
warp-and-blend triangular elements of ADR 0003. ADR 0004 provides selectable
scalar limiter functions and interface fluxes, but explicitly does not decide
how to limit a multidimensional DG solution. Its scalar TVD functions alone do
not make a degree-four triangular DG scheme TVD, shock stable, or
positivity-preserving. No accepted ADR specifies the complete spatial and time
discretization, equation of state, boundary implementation, mesh
representation, or DG limiting and positivity treatment. These gaps are not
contradictions between the accepted ADRs.

## Decision

Use the following dimensionless verification case:

- Solve the two-dimensional compressible Euler equations for a calorically
  perfect gas with ratio of specific heats \(\gamma=1.4\), without a heat
  source. Set \((\rho,u,v,p)=(1,0,0,1)\) for \(x<0.5\) and
  \((0.125,0,0,0.1)\) for \(x>0.5\). Convert primitive to conserved variables
  using \(E=p/(\gamma-1)+\rho(u^2+v^2)/2\).
- Use the rectangle \([0,1]\times[0,0.1]\). Impose inviscid slip walls at the
  top and bottom. At the left and right boundaries, evaluate the selected
  numerical flux with the initial left and right states, respectively, as
  fixed exterior states. In particular, do not impose an extra velocity or
  pressure condition on the interior state. This is a fixed-reservoir
  boundary treatment, not a demonstration of sustained throughflow.
- Divide the rectangle into \(200\times20\) rectangles, each split into two
  straight-sided triangles using a consistent diagonal. Align the initial
  discontinuity with element faces at \(x=0.5\). Use degree-four
  warp-and-blend interpolation nodes, including the GLL edge nodes, as
  required by ADR 0003. Initialize the conserved solution separately on each
  element, including independent traces at the discontinuity.
- Use the Rusanov interface flux from ADR 0004 for the baseline case. The
  eventual DG shock-limiting and positivity treatment must be separately
  decided; selecting an ADR 0004 scalar function is not itself such a method.
  Other ADR 0004 fluxes and limiter choices remain available for separately
  configured comparisons, not implied baseline defaults.
- Run to \(t=0.2\), before the exact waves reach either end. Compare the
  density, pressure, and velocities with an independent one-dimensional exact
  Sod Riemann solution extended uniformly in \(y\). At this time the exact
  rarefaction head, contact, and shock are approximately at \(x=0.26336\),
  \(0.68549\), and \(0.85043\), respectively.

The baseline acceptance checks are:

1. Density and pressure remain finite and strictly positive at all solution
   nodes, volume and face quadrature points, and any additional points used by
   the chosen method to enforce positivity. These samples alone do not prove
   positivity everywhere inside an element.
2. The domain-averaged \(L^1\) density error relative to the exact solution
   is at most \(0.03\), and the shock location is within \(0.02\) of the exact
   position. Average density over each mesh column, find the greatest decrease
   between adjacent column averages whose shared face lies in
   \(0.8\le x\le0.9\), and use that face as the shock location.
3. Check conservation of mass, both momentum components, and total energy
   against the time-integrated numerical boundary fluxes, including wall
   pressure forces. Require absolute conservation defects no greater than
   \(10^{-8}\) per conserved component for this dimensionless domain, using
   double precision.
4. Report errors in density, pressure, and both velocity components, and run a
   refinement comparison with \(100\times10\) and \(200\times20\) rectangles.
   The fine-grid density \(L^1\) error must be lower than the coarse-grid error.

Define domain-averaged \(L^1\) error as the integral over the full rectangle of
the absolute pointwise difference divided by the domain area. Evaluate the
numerical polynomial and exact solution with sufficiently accurate integration,
splitting integration at exact discontinuities where needed; comparing only
nodal samples is not equivalent. These numerical tolerances are targets, not
results from an implemented solver. If an approved method cannot meet them,
review the measured refinement and error studies with the user and follow the
ADR process for any decision changes.

## Alternatives considered

- **Transmissive end boundaries:** Extrapolating the interior state is also
  suitable for this finite-time benchmark when waves do not reach the ends.
  Fixed exterior states more directly express the requested left and right
  reservoirs, although neither choice tests sustained inflow and outflow.
- **True throughflow:** Prescribe nonzero inlet flow and a compatible outlet
  condition in a separate test. This would test boundary operation, but is not
  the standard stationary Sod problem and needs its own reference solution.
- **One-dimensional mesh or quadrilateral elements:** Simpler to construct,
  but would not exercise the two-dimensional triangular elements required by
  ADR 0003.
- **Treat an ADR 0004 TVD function as a complete DG limiter:** This would
  avoid a further decision, but does not define the reconstruction, its
  multidimensional application, or positivity preservation.

## Consequences

- The case exercises shock, contact, and rarefaction propagation on a
  two-dimensional triangular mesh against an independent exact reference.
  Its \(y\)-uniform initial condition does not verify general multidimensional
  wave behavior.
- The fixed exterior states and slip walls must be implemented consistently
  with the chosen DG numerical flux. A separate throughflow test is needed to
  verify physical inflow/outflow operation.
- Before implementation, propose and obtain approval for the missing solver
  decisions, including spatial operators and quadrature, time integration and
  stability control, mesh/connectivity and boundary representation, and a
  multidimensional DG shock-limiting and positivity strategy. The latter must
  explain any use of ADR 0004 limiter functions and interface fluxes.
- The acceptance targets and diagnostic integration require validation against
  the implemented method and reference solution. Refinement is a check, not a
  claim of high-order convergence at discontinuities.
- This ADR does not supersede ADR 0002, 0003, or 0004. If a necessary solver
  decision conflicts with an accepted ADR, propose a superseding ADR and
  obtain approval before depending on it.
