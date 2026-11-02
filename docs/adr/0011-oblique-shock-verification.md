# ADR 0011: Verify the solver with a steady oblique shock on an inclined wall

- Status: Proposed
- Date: 2026-11-02
- Approval: Pending

## Context

ADR 0005 provides a stationary Sod problem with reservoir ends, and states that
a separate throughflow test is needed to verify physical inflow and outflow
operation. A supersonic inflow over a compression corner is such a test: the
flow enters supersonically, is turned by an inclined slip wall, and a steady
attached oblique shock forms with an analytic post-shock state. The case
must be parameterized by inflow Mach number \(M_1\) and wall slope angle
\(\theta\).

The solver already applies slip-wall mirror states using the face normal
(`cfd_demo/dg.py`, `mirror_state`), so an inclined wall needs no change to the
DG operator. The other requirements depend on ADR 0007 and ADR 0012 (mesh
geometry and far-field boundaries), which this ADR does not decide.

## Decision

Use the following dimensionless, steady verification case.

- **Equations and gas:** Same as ADR 0005: calorically perfect gas with
  \(\gamma=1.4\), no source terms.
- **Parameters:** Inflow Mach number \(M_1>1\) and wall slope angle
  \(\theta\). The configuration accepts only the attached weak-shock regime,
  \(0<\theta<\theta_{\max}(M_1)\), where \(\theta_{\max}\) is the maximum
  deflection for an attached shock. Other inputs raise `ValueError`.
- **Upstream state:** \((\rho,u,v,p)=(1,\,M_1,\,0,\,1/\gamma)\), so the sound
  speed is 1 and \(u=M_1\). Inflow is parallel to the \(x\)-axis.
- **Exact reference:** The standard \(\theta\)-\(\beta\)-\(M\) relation with
  \(\beta\) the weak-shock angle. The post-shock state is uniform, moves
  parallel to the wall, and is obtained from the normal-shock jump relations
  applied to the normal Mach number \(M_1\sin\beta\). The shock is the line
  \(y=x\tan\beta\) from the corner at the origin. The exact solution is
  piecewise constant on either side of this line.
- **Domain:** Region \(0\le x\le L\), \(x\tan\theta\le y\le x\tan\theta+H\),
  with the wall on the lower boundary \(y=x\tan\theta\) and the corner at the
  origin. Geometry and tags are defined by ADR 0012.
- **Boundary conditions:**
  - Left (\(x=0\), supersonic inflow): `FixedState` with the upstream state.
  - Bottom (wall): `SlipWall`, using the inclined face normal.
  - Top and right (far field): exact-solution exterior states from ADR 0012.
- **Initial condition:** Upstream state everywhere. The solution is run to a
  final time \(T\) at which the interior is steady to the acceptance
  tolerance. Choose \(T\) and the domain size \(L,H\) from a pilot run and
  record them in the test.
- **Baseline numerics:** Rusanov interface flux with `minmod`, as in ADR 0005.
  Other fluxes and limiters are comparison options only.
- **Acceptance checks**, to be confirmed after a pilot run:
  1. Density and pressure remain finite and strictly positive at all solution
     nodes.
  2. Domain-averaged \(L^1\) error of density relative to the exact solution
     is at most 0.03.
  3. Shock angle, from the density jump along a column cut at \(x\) in
     the mid-domain, differs from \(\beta\) by at most 1°.
  4. Post-shock density, pressure, and velocity components agree with the
     exact state within the same \(L^1\) and pointwise targets as check 2,
     measured over the post-shock region only.
  5. Conservation of mass, both momenta, and energy, measured as the net
     change in the domain integral plus the time-integrated boundary fluxes
     including wall pressure, at most \(10^{-8}\) per component.
  6. Refinement: \(100\times10\) and \(200\times20\) (per ADR 0012's mesh
     generator) give a lower \(L^1\) density error on the fine grid.
- **Test plan:** Add a `verification`-marked test with the baseline
  \((M_1,\theta)=(2.0,\,10^\circ)\) and at least one second \((M_1,\theta)\)
  pair. Add unit tests for the exact relation (round-trip \(\theta\to\beta\to
  \theta\), limits \(\theta\to0\) and \(\theta\to\theta_{\max}\)) and for input
  validation.

## Alternatives considered

- **Supersonic extrapolation at the far field:** Simple, and the local
  velocity is supersonic in the upstream region. Not used as the default
  because the post-shock normal Mach number can be subsonic, which makes
  extrapolation ill-posed and introduces reflections at the far boundary.
- **Constant `FixedState` at the far field:** Only valid where the state is
  uniform. The shock crosses the top and right boundaries, so a single state
  is wrong.
- **Shock reflection (symmetric top wall):** Avoids far-field conditions, but
  it is a different problem with an interior reflected shock and a different
  reference solution.
- **Steady-state solver without time stepping:** Not part of the accepted
  architecture (ADR 0009 uses SSP-RK3).

## Consequences

- The oblique case is the first verification with nonzero inflow and wall
  slope. It checks the boundary treatment, the inclined slip wall, and the
  shock-capturing machinery on a non-axis-aligned shock.
- The exact solution depends on \((M_1,\theta)\), so the reference must be
  implemented and tested independently of the solver.
- Targets are proposed, not validated. If the approved method misses them,
  the review follows ADR 0001's process before the targets are changed.
- Dependencies: ADR 0012 must be approved before implementation.

## References

- Anderson, J. D., *Modern Compressible Flow*, theta-beta-M relation and
  oblique shock jump conditions.
