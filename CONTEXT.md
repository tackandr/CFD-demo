# Project context

## Purpose

This project is a demonstration of agentic development for a two-dimensional
solver of compressible Euler equations. The intended application is hypersonic
airflow through a ramjet engine, including combustion.

## Numerical approach

- Use an unstructured mesh of simplex elements.
- Use the Discontinuous Galerkin (DG) discretization, with flux limiters to
  capture shocks.
- Base the numerical methods on *Nodal Discontinuous Galerkin Methods*.
- Couple the energy equation to heat-source terms to represent combustion.

## Implementation direction

Implement the solver in Python using JAX.

## Guidance for agents

Treat the details above as project goals and direction, not as evidence that
features have already been implemented. Specific choices and requirements will
be established through Architecture Decision Records (ADRs) in `docs/adr/`.
Follow accepted ADRs; do not invent or silently settle details that have not
been decided. The current ADR process is described in
`docs/adr/0001-adr-driven-development.md`.
