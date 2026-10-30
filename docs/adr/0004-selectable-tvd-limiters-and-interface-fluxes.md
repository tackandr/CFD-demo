# ADR 0004: Provide selectable TVD limiters and interface fluxes

- Status: Accepted
- Date: 2026-10-30
- Approval: Andreas

## Context

High-resolution schemes use TVD limiter functions to reduce spurious oscillations, while numerical
interface fluxes provide alternative ways to compute fluxes between elements.
Limiter behavior varies: more compressive limiters can preserve sharp
features, while more restrictive limiters can be more diffusive. Users need
to select both limiter functions and interface fluxes, and to be able to
compose those components in future flux-limiting strategies.

Here, “TVD limiter” means a scalar limiter function \(\phi(r)\) used by a
high-resolution reconstruction. A numerical interface flux (such as
Rusanov or HLLC) is a separate component that may be used together with a
limiter. Neither term specifies a discontinuous Galerkin
solution/slope-limiting algorithm.

## Decision

Provide named, selectable, pure JAX-compatible TVD limiter functions for:

- **Minmod:** \(\phi(r)=\max(0,\min(1,r))\)
- **Monotonized central (MC):**
  \(\phi(r)=\max(0,\min(2r,(1+r)/2,2))\)
- **Van Leer:** \(\phi(r)=(r+|r|)/(1+|r|)\)
- **Superbee:**
  \(\phi(r)=\max(0,\min(2r,1),\min(r,2))\)
- **Unlimited:** \(\phi(r)=1\), as an explicit no-limiting option

Provide selectable numerical interface fluxes for the compressible-flow
solver, initially including:

- **Rusanov (local Lax-Friedrichs)**
- **Roe**
- **HLLC**

Expose both choices by stable names in solver configuration, so callers can
choose each component without changing numerical-kernel code. Limiter
functions and interface fluxes must operate on JAX arrays and be composable
with JAX transformations. Keep the two component choices independent so that
later flux-limiting strategies can combine them.

This decision defines the available component choices and their selection
mechanism, but does not prescribe how a TVD limiter modifies or combines with
a numerical interface flux. It also does not select a reconstruction method
or where and how these components are applied in the DG discretization. In
particular, the multidimensional and unstructured-triangle treatment, and
whether limiting is component-wise or characteristic-wise, require a separate
decision before integration into the solver.


## Consequences

- Users can select common limiter functions and interface fluxes using
  consistent named configuration options.
- Limiter kernels and interface fluxes can be tested independently for their
  numerical behavior, vectorized JAX behavior, and compatibility with JAX
  transformations.
- Keeping the components independent enables later combined flux-limiting
  strategies without committing to a particular coupling in this ADR.
- The unlimited option provides a baseline for comparison; it does not imply
  that an unlimited scheme is stable or appropriate for every problem.
- A follow-up decision is required to specify limiter/flux composition, the
  reconstruction, and multidimensional DG integration before these components
  affect solver solutions.
- Default limiter and interface-flux choices, if any, should be selected with
  that integration decision rather than assumed by this ADR.
