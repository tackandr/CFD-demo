# CFD-demo

This is a demo implementation to study agentic coding to build a CFD solver.

It solves the two-dimensional compressible Euler equations with a degree-four
nodal discontinuous Galerkin method on triangles, written in JAX. Every design
decision is recorded as an architecture decision record in [docs/adr](docs/adr/);
read those before changing the code, and see [AGENTS.md](AGENTS.md).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) (ADR 0006) and needs Python
3.12 or newer:

```sh
uv sync
```

## Tests

```sh
uv run pytest                      # unit tests and the ADR 0005 verification case
uv run pytest -m "not verification"  # unit tests only, about 30 seconds
```

The full run takes roughly seven minutes because it solves the Sod shock tube
at both resolutions required by ADR 0005. It prints a report of the measured
errors, wave positions, and conservation defects in the terminal summary.

## Example

```python
from cfd_demo import sod_shock_tube

case = sod_shock_tube(100, 10, flux="rusanov", limiter="minmod")
result = case.solver.run(case.initial_state, case.final_time)
print(result.steps, result.min_density, result.min_pressure)
```

`flux` is one of `rusanov`, `roe`, or `hllc`, and `limiter` is one of `minmod`,
`mc`, `van_leer`, `superbee`, or `unlimited` (ADR 0004). Neither has a default:
ADR 0010 requires the choice to be explicit.

## Layout

| Path | Contents |
| --- | --- |
| `src/cfd_demo/reference_element.py` | Warp-and-blend nodes and the reference operators (ADR 0003) |
| `src/cfd_demo/euler.py`, `fluxes.py`, `limiters.py` | Euler relations, interface fluxes, and limiter functions (ADR 0004) |
| `src/cfd_demo/mesh.py`, `boundary.py` | Conforming triangle mesh, connectivity, and tagged boundaries (ADR 0007) |
| `src/cfd_demo/dg.py` | Collocated strong-form DG operator (ADR 0008) |
| `src/cfd_demo/limiting.py` | Troubled-cell and positivity limiting (ADR 0010) |
| `src/cfd_demo/time_integration.py`, `solver.py` | SSP-RK3 integration and the solver driver (ADR 0009) |
| `src/cfd_demo/cases.py` | The Sod shock tube setup (ADR 0005) |
| `tests/unit/` | Unit tests of the components above |
| `tests/verification/` | The ADR 0005 verification case with an independent exact Riemann solver |
