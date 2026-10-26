# ADR 0002: Use Python with JAX as the programming framework

- Status: Proposed
- Date: 2026-10-26
- Approval: Pending

## Context

The project needs a programming framework for developing its two-dimensional
compressible-flow solver. The solver is intended to use Discontinuous Galerkin
methods and is expected to benefit from array-oriented numerical computation
and compiled execution. The project context already identifies Python and JAX
as the implementation direction; this ADR proposes that choice for approval.

## Decision

Use Python as the programming language and JAX as the numerical computing
framework for the solver.

JAX provides a Python interface for array-based numerical computation and
supports transformations such as just-in-time compilation and automatic
differentiation. These capabilities are a suitable foundation for implementing
and composing the solver's numerical kernels.

This decision does not specify Python or JAX versions, dependency management,
supported hardware, or execution backends. Those details remain open for later
decisions.

## Alternatives considered

- **PyTorch:** A Python tensor framework with automatic differentiation and
  accelerator support. It is a viable alternative, but JAX is preferred for its
  composable transformations and the proposed fit with compiling numerical
  kernels.
- **CuPy:** A Python GPU-array library with a NumPy-like interface. It is a
  viable alternative for GPU array computation, but JAX is preferred as the
  framework for this solver because its transformations provide a broader
  foundation for composing and compiling numerical kernels.

## Consequences

- Solver numerical kernels will be implemented using JAX's array and
  transformation APIs, subject to later decisions about implementation details.
- The project can use JAX's compilation and differentiation capabilities where
  appropriate; this ADR does not require automatic differentiation in any
  particular algorithm.
- The project will depend on Python and JAX. Version constraints, installation
  instructions, and hardware support must be specified separately.
- Validation should confirm that representative solver kernels produce the
  expected numerical results under the chosen JAX execution configuration.
