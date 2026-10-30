# ADR 0010: Limit troubled cells and enforce nodal positivity

- Status: Accepted
- Date: 2026-10-30
- Approval: User, 2026-10-30

## Context

ADR 0004 provides scalar TVD limiter functions and interface fluxes, but
leaves their composition, the reconstruction, the multidimensional DG
integration, and the choice between component-wise and characteristic-wise
limiting to a further decision. ADR 0005 requires a multidimensional
shock-limiting and positivity strategy that explains any use of ADR 0004
components. A degree-four DG solution should stay high order in smooth
regions, while shocks require nonlinear limiting.

## Decision

After every Runge–Kutta stage (ADR 0009), apply the following two steps.

1. **Troubled-cell slope limiting (after Cockburn and Shu, 1998),
   component-wise in conserved variables:**
   - For element \(K_0\) with barycenter \(b_0\) and cell average
     \(\bar u_0\), let \(m_i\) be the midpoint of face \(i\), which is the
     middle GLL node. Use the averages \(\bar u_i\) and barycenters \(b_i\) of
     the three face neighbors. At a boundary face, use a ghost cell whose
     barycenter reflects \(b_0\) across the face and whose average is the
     mirror state of \(\bar u_0\) (slip wall) or the prescribed state (fixed
     state).
   - The neighbor deviation at \(m_i\) is
     \(\Delta\bar u_i=\alpha_i(\bar u_i-\bar u_0)+\beta_i(\bar u_j-\bar u_0)\),
     with \(\alpha_i,\beta_i\ge0\) satisfying
     \(m_i-b_0=\alpha_i(b_i-b_0)+\beta_i(b_j-b_0)\) for one of the other faces
     \(j\).
   - **Detection:** \(K_0\) is troubled if, for density or total energy at
     any face midpoint, \(\operatorname{minmod}(\tilde u_i,\nu\Delta\bar u_i)\)
     differs from \(\tilde u_i\) by more than \(10^{-10}|\bar u_0|\), where
     \(\tilde u_i=u_h(m_i)-\bar u_0\) uses the degree-four trace and
     \(\nu=1.5\). Detection does not depend on the selected limiter function.
   - **Reconstruction:** In a troubled cell, replace every conserved
     component by a linear function with the same average. From the midpoint
     deviations \(\tilde u^1_i\) of the component's \(L^2\) projection onto
     linear polynomials, set \(\Delta_i=\tilde u^1_i\min(1,\phi(r_i))\) with
     \(r_i=\nu\Delta\bar u_i/\tilde u^1_i\) (\(\Delta_i=0\) if
     \(\tilde u^1_i=0\)), where \(\phi\) is the selected ADR 0004 limiter
     function. Rescale the positive and negative \(\Delta_i\) so that
     \(\sum_i\Delta_i=0\) (Cockburn and Shu), and build the linear function
     from \(\bar u_0\) and these midpoint deviations.
   - **Limiter choices:** Select \(\phi\) by the stable configuration names
     `minmod`, `mc`, `van_leer`, `superbee`, and `unlimited` (ADR 0004).
     `minmod` reproduces the Cockburn–Shu limiter. `mc`, `van_leer`, and
     `superbee` keep progressively more of the slope in troubled cells; the
     cap \(\min(1,\phi)\) prevents steepening beyond the projected DG slope.
     `unlimited` skips this step.
2. **Positivity (Zhang–Shu scaling limiter):** For each element with an
   admissible average (\(\bar\rho>0\), \(p(\bar u)>0\)), let
   \(\varepsilon=\min(10^{-13},\bar\rho,p(\bar u))\). Scale the deviation from
   the average, \(u\leftarrow\bar u+\theta(u-\bar u)\), first so that density
   is at least \(\varepsilon\) at all solution nodes and then so that pressure
   is at least \(\varepsilon\) at all solution nodes, finding \(\theta\) along
   each straight segment by bisection. Cell averages are unchanged. An
   inadmissible average triggers step rejection (ADR 0009).

Interface fluxes are selected independently (ADRs 0004 and 0008) and enter
only through the DG operator; limiting and fluxes are not otherwise coupled.
The solver has no implicit defaults for either choice: callers name both
explicitly. The ADR 0005 baseline uses `rusanov` with `minmod`. The positivity
step always runs, including with `unlimited`.

## Alternatives considered

- **Artificial viscosity (for example, Persson and Peraire):** Smooth and
  high order, but needs a viscous operator, a sensor, and tuning, and tightens
  the time step.
- **Subcell finite-volume or a posteriori (MOOD) limiting:** Robust, but
  considerably more complex.
- **WENO limiters on unstructured meshes (Zhu, Qiu, and Shu):** More accurate
  in troubled cells, with more complex stencils.
- **Vertex-based Barth–Jespersen limiting:** Simpler, but more diffusive, and
  most ADR 0004 functions do not fit its bounds.
- **Characteristic-wise limiting:** Fewer oscillations at strong shocks, but
  needs the eigensystem in each face direction; it can be adopted later.
- **Limiting every cell or using TVB thresholds:** Limiting everywhere
  over-damps degree-four solutions; TVB constants need problem-dependent
  tuning.
- **Provably positivity-preserving over-integrated schemes (Zhang, Xia, and
  Shu, 2012):** Stronger guarantees, but they require the over-integrated
  operator not selected in ADR 0008.

## Consequences

- Shocks and contacts are captured with locally linear solutions in a few
  cells, while smooth regions keep degree four. Smooth extrema that are
  detected are clipped (TVD behavior).
- All limiting preserves cell averages, so conservation is unaffected.
- Positivity is enforced at the solution nodes, which are all the points where
  the collocated operator evaluates fluxes. As ADR 0005 notes, this does not
  prove positivity everywhere inside an element; positivity of cell averages
  relies on the CFL limit and step rejection.
- Component-wise limiting can leave small oscillations near strong
  discontinuities.
- ADR 0005 validates only the baseline (`rusanov` with `minmod`); other
  combinations are comparison options.
- Validation: limiting preserves cell averages, linear data are not limited,
  positivity limiting removes negative nodal states, and the ADR 0005
  acceptance checks pass.

## References

- Bernardo Cockburn and Chi-Wang Shu, "The Runge–Kutta discontinuous Galerkin
  method for conservation laws V: multidimensional systems," *Journal of
  Computational Physics* 141 (1998).
- Xiangxiong Zhang and Chi-Wang Shu, "On positivity-preserving high order
  discontinuous Galerkin schemes for compressible Euler equations on
  rectangular meshes," *Journal of Computational Physics* 229 (2010).
