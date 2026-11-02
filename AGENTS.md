# Agent instructions

## ADR-driven development

- Before planning or producing code, read the architecture decision records (ADRs)
  in [docs/adr](docs/adr/) and identify the accepted decisions relevant to the task.
- Plans, code, tests, and documentation must adhere to accepted ADRs. Reference
  the relevant ADRs in the plan.
- If a decision is needed that is not covered by an accepted ADR, inform the user
  during planning. Explain the decision, alternatives, and recommendation, and
  create a new ADR with status `Proposed` before implementing affected code.
- Use [ADR 0001](docs/adr/0001-adr-driven-development.md) as the structure for new
  ADRs. Follow its naming and lifecycle conventions.
- Ask the user to approve proposed decisions. Do not treat a proposed ADR as
  accepted or implement code that depends on it until the user approves it.
- If a task conflicts with an accepted ADR, explain the conflict to the user and
  propose a superseding ADR during planning rather than silently deviating.
- When implementing an approved plan, validate the result against the relevant
  ADRs and run existing applicable checks. Report any unresolved decisions.

## Validating changes

- The project uses `uv` (ADR 0006). Install the environment with `uv sync`.
- Run `uv run pytest` before reporting a change as complete. This runs the unit
  tests and the ADR 0005 Sod shock tube verification case, and takes roughly
  seven minutes.
- While iterating, `uv run pytest -m "not verification"` runs the unit tests
  alone in about half a minute.
- The verification run prints a report of the measured errors, wave positions
  and conservation defects in the terminal summary. Check it against the
  acceptance criteria of ADR 0005 when changing the discretization.
- Add tests for new behavior and reference the ADR the behavior comes from.
