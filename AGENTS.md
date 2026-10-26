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

## Initial scope

This initial setup is documentation only. Do not add source code, dependencies,
or build/test infrastructure as part of this setup; implementation belongs to
later tasks guided by accepted ADRs.
