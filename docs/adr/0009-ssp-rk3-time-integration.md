# ADR 0009: Integrate in time with SSP-RK3 and CFL-based step control

- Status: Accepted
- Date: 2026-10-30
- Approval: User, 2026-10-30

## Context

ADR 0005 requires a decision on time integration and stability control. The
DG operator of ADR 0008 is explicit, and the limiters of ADR 0010 must be
applied consistently during time stepping. ADR 0005's conservation check
needs boundary fluxes integrated consistently with the time integrator.

## Decision

- **Integrator:** Use the explicit three-stage, third-order
  strong-stability-preserving Runge–Kutta method (SSP-RK3) in Shu–Osher form:
  \(u^{(1)}=\Lambda(u^n+\Delta t\,L(u^n))\),
  \(u^{(2)}=\Lambda(\tfrac34u^n+\tfrac14(u^{(1)}+\Delta t\,L(u^{(1)})))\), and
  \(u^{n+1}=\Lambda(\tfrac13u^n+\tfrac23(u^{(2)}+\Delta t\,L(u^{(2)})))\),
  where \(\Lambda\) applies the ADR 0010 limiters.
- **Time step:** At the start of each step, set
  \(\Delta t=\mathrm{CFL}\cdot\min_{K,\,e\subset\partial K}2|K|/((N+1)^2|e|\lambda_K)\),
  Hesthaven and Warburton's estimate, where \(\lambda_K\) is the maximum of
  \(|u|+c\) over the nodes of element \(K\) and \(N=4\). The default CFL is
  0.5 and is configurable. Shorten the last step to end exactly at the
  requested final time.
- **Step rejection:** If any stage produces a non-finite value or a cell
  average with nonpositive density or pressure, discard the step and retry it
  from its initial state with \(\Delta t\) halved, at most 10 times, and then
  stop with an error. Report the number of rejected steps.
- **Boundary-flux accounting:** For each accepted step, accumulate
  \(\Delta t\,(B(u^n)/6+B(u^{(1)})/6+2B(u^{(2)})/3)\), where \(B\) is the
  boundary-flux integral of ADR 0008. Because the limiters preserve cell
  averages, this equals the change in the domain integral produced by the
  step.
- **JAX execution:** Compile each step with `jax.jit`; run the outer loop in
  Python to handle rejection, the final step, and reporting.

## Alternatives considered

- **Low-storage RK4 (Hesthaven and Warburton's default):** Has a larger
  stability region, but is not SSP, so limiting between stages loses the
  convex-combination property.
- **Higher-order SSP methods, such as SSP-RK(5,4):** Improve temporal accuracy
  with more stages per step; shocks dominate the error here.
- **Fixed time step:** Simpler, but not robust as wave speeds change.
- **Implicit integration:** Unnecessary for this explicit shock problem and
  much more complex.

## Consequences

- Each SSP-RK3 stage is a convex combination of forward-Euler steps, so
  limiting after every stage is well defined, and conservation can be checked
  exactly against the accumulated boundary fluxes.
- A one-dimensional linear analysis of upwind degree-four DG with SSP-RK3 is
  stable up to \(a\Delta t/h\approx0.0897\), a CFL of about 2.24 in this
  normalization. The default of 0.5 leaves margin for triangles,
  nonlinearity, and limiting. Positivity of cell averages is not proven; step
  rejection is the safeguard.
- Cost per unit simulated time scales with \((N+1)^2/h\); the ADR 0005 Sod
  run takes thousands of steps.
- Validation: the ADR 0005 benchmark (stability, positivity, and
  conservation), with step and rejection counts reported.
