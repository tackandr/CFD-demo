# ADR 0006: Use a uv-managed Python package and pytest verification

- Status: Accepted
- Date: 2026-10-30
- Approval: User, 2026-10-30

## Context

ADR 0002 selects Python and JAX but leaves versions, dependency management,
and execution backends open. ADR 0001 does not select testing tools, and ADR
0005 notes that the repository has no test infrastructure. The user has asked
for the ADR 0005 Sod shock tube case to run as an automatic pytest test that
agents run when validating code. ADR 0005 also requires double precision for
its conservation check.

## Decision

- **Layout:** Put the solver in an importable package `cfd_demo` under `src/`,
  with tests under `tests/`. Declare metadata and dependencies in
  `pyproject.toml` (PEP 621), built with `hatchling`.
- **Versions and dependencies:** Require Python 3.12 or newer, as required by
  current JAX releases. Runtime dependencies are `jax>=0.12` and `numpy>=2.2`.
  The test dependency `pytest>=8` is in the PEP 735 `dev` dependency group.
  Commit `uv.lock` so verification runs use reproducible versions.
- **Tooling:** Use `uv`: `uv sync` creates the environment and `uv run pytest`
  runs the tests.
- **Execution configuration:** CPU is the supported and validated backend.
  Importing `cfd_demo` enables JAX 64-bit mode (`jax_enable_x64`), and solver
  arrays use float64.
- **Tests:** Configure pytest in `pyproject.toml` with `testpaths = ["tests"]`,
  strict markers, and reporting of skip reasons. Mark verification benchmarks,
  such as ADR 0005, with `verification`; they run by default. Agents must run
  the full suite with `uv run pytest` before completing code changes and must
  report failures.
- **Independent references:** Exact reference solutions and verification
  diagnostics are test code implemented with NumPy in float64. They may read
  solver outputs (mesh, nodes, solution, and boundary-flux integrals) but do
  not reuse solver numerical kernels.

## Alternatives considered

- **`requirements.txt` with pip:** Familiar and simple, but has no lock file,
  so numerical results are less reproducible across environments.
- **Poetry or PDM:** Capable, but add another tool; `uv` is already available
  and fast.
- **unittest:** Part of the standard library, but pytest fixtures, markers,
  and reporting suit expensive shared verification runs better.
- **Opt-in verification tests:** Faster routine runs, but agents could miss the
  regressions the benchmark is meant to catch.
- **Enabling float64 only in tests:** Avoids a global side effect, but solver
  users could silently run in single precision.

## Consequences

- Agents have one documented validation command, `uv run pytest`. Because it
  includes the Sod benchmark, the suite takes minutes on CPU; report measured
  times and propose a faster default if this becomes a burden.
- Importing `cfd_demo` changes JAX's process-wide precision setting.
- Dependency changes must update `pyproject.toml` and `uv.lock` together.
- GPU/TPU execution and continuous integration need separate decisions.
