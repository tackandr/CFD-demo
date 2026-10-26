# ADR 0001: Use ADR-driven development

- Status: Accepted
- Date: 2026-10-26
- Approval: Established by the user's request for ADR-driven development.

## Context

This project explores agent-driven development of a CFD solver. Agents need an
explicit record of architectural decisions so implementation follows agreed
direction rather than making undocumented choices.

## Decision

Record architectural decisions in Markdown ADRs in this directory. Agents must
follow accepted ADRs when planning and producing code.

When a necessary decision is not covered by an accepted ADR, the agent must
inform the user and draft a new ADR during planning. Implementation that depends
on the decision must wait for user approval.

### Creating subsequent ADRs

Use this document as the initial example and reusable template:

1. Create `NNNN-short-decision-title.md`, using the next unused four-digit number
   in sequence. Do not renumber existing ADRs.
2. Copy the title, metadata, and the Context, Decision, Alternatives considered,
   and Consequences sections. Replace this ADR's content with the new decision;
   the creation instructions themselves need not be copied.
3. Set the title to `ADR NNNN: <decision title>`, status to `Proposed`, and date to
   the creation date (`YYYY-MM-DD`). Set approval to `Pending` rather than copying
   this ADR's approval.
4. In Context, describe the problem, constraints, and decision drivers. In
   Decision, state the proposed choice and rationale. In Alternatives considered,
   record viable alternatives and why they were not selected. In Consequences,
   record benefits, costs, risks, and validation expectations.
5. Request user review during planning. On approval, change status to `Accepted`
   and record the approval; otherwise keep it `Proposed` or mark it `Rejected`
   if rejected.
6. Preserve accepted decisions as history. To change one, create a new ADR and,
   once approved, mark the old one `Superseded by ADR NNNN`. Link the new ADR back
   to the decision it supersedes.

## Alternatives considered

- Make decisions implicitly in code or conversations: simpler initially, but
  difficult for later agents to discover and apply consistently.
- Document decisions only after implementation: preserves some history, but
  prevents user review before code depends on a decision.

## Consequences

- Agents and reviewers have a shared, version-controlled source of decisions.
- Planning may pause for user approval when an architectural decision is missing
  or must change.
- Each implementation plan must reference applicable accepted ADRs, and validation
  must check that the implementation follows them.
- This ADR establishes the process only; it does not select a language, solver
  architecture, numerical method, dependencies, or testing tools.
