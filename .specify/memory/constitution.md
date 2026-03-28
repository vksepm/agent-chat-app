<!--
SYNC IMPACT REPORT
==================
Version change: (unversioned template) → 1.0.0
Modified principles: N/A — initial population from template
Added sections:
  - Core Principles (4 principles: PoC First, MVP Delivery, Speed to Market, Decent UX)
  - Development Constraints
  - Governance
Removed sections: N/A
Templates reviewed:
  ✅ .specify/templates/plan-template.md   — Constitution Check gate aligns with principles
  ✅ .specify/templates/spec-template.md   — MVP/P1 story pattern consistent with Principle II
  ✅ .specify/templates/tasks-template.md  — MVP First strategy consistent with Principle II & III
Deferred TODOs: None
-->

# SpecKit Chat App Constitution

## Core Principles

### I. Proof of Concept First

Before committing to a full implementation, every new feature or technical approach MUST be
validated as a working proof of concept (PoC).

- A PoC MUST answer one specific question: "Does this approach work at all?"
- PoC code MUST be throwaway by default; do not design it for production re-use unless
  explicitly promoted after review.
- A PoC is considered complete when it demonstrates the core mechanic end-to-end, even with
  hard-coded values or missing edge-case handling.
- PoC findings (what worked, what did not) MUST be documented before proceeding to MVP build.

**Rationale**: Discovering a fundamental blocker after weeks of production-quality work wastes
time and money. Cheap validation upfront de-risks every subsequent investment.

### II. MVP Delivery

Every deliverable MUST ship as the smallest slice that provides real, demonstrable user value.

- Features MUST be broken into independently deliverable user stories (P1, P2, P3...).
- P1 stories alone MUST constitute a viable, shippable product — not merely a foundation.
- Scope creep (adding non-P1 work before P1 ships) is a violation of this principle.
- "Done" means deployed and usable by a real user, not just passing tests locally.
- Gold-plating (adding polish, options, or abstractions not required by a current story)
  is explicitly forbidden until MVP is live.

**Rationale**: An MVP in production generates real feedback. Unreleased features generate none.

### III. Speed to Market

Time-to-market MUST be treated as a first-class constraint alongside correctness.

- The team MUST bias toward action: a working solution shipped today beats a perfect solution
  shipped next month.
- Decisions that block progress for more than one day MUST be escalated and resolved, not
  deferred indefinitely.
- Technical debt incurred in service of speed MUST be recorded as a tracked backlog item at
  the time it is introduced — not left undocumented.
- Complexity MUST be justified. YAGNI (You Aren't Gonna Need It) applies by default; any
  abstraction, pattern, or tooling added "for the future" requires explicit written justification.
- Feature flags and incremental rollout are preferred over big-bang releases.

**Rationale**: Market windows close. Validated learning from real users accelerates better
decisions than extended pre-release design cycles.

### IV. Decent User Experience

Every user-facing surface MUST meet a baseline of usability, even in early iterations.

- "Decent UX" means: the primary user journey works without confusion, errors surface with
  actionable messages, and the interface does not feel broken or unfinished.
- Visual polish is NOT required at MVP; functional clarity IS required at MVP.
- The team MUST test the primary user journey manually before each release — no exceptions.
- Loading states, error states, and empty states MUST be handled for every P1 user story.
- Accessibility baseline (keyboard navigability, sufficient color contrast, readable font sizes)
  MUST be met before any story is marked done.

**Rationale**: Users abandon products that feel broken. A minimum viable experience retains
users long enough to generate learning; a broken experience does not.

## Development Constraints

- **No over-engineering**: Solutions MUST be as simple as possible. Complexity requires
  justification recorded in the plan's Complexity Tracking table.
- **Iterative deployment**: Code MUST be deployable at every phase checkpoint. No multi-week
  branches that are not deployable.
- **Feedback loops**: User or stakeholder feedback MUST be solicited after each MVP increment.
  Development priorities for the next increment MUST reflect that feedback.
- **Dependency hygiene**: Third-party dependencies MUST be evaluated for maintenance status
  and license compatibility before adoption. Prefer well-maintained, widely-used libraries.

## Governance

This constitution supersedes all other project practices and informal conventions.

- All feature specs MUST reference the active constitution version in their preamble.
- Amendments require: (1) a written proposal, (2) team review, (3) version bump per semver
  rules below, and (4) propagation to dependent templates.
- **Versioning policy**:
  - MAJOR bump: removal or redefinition of a core principle.
  - MINOR bump: new principle or section added.
  - PATCH bump: clarifications, wording fixes, non-semantic refinements.
- All pull requests MUST include a Constitution Check confirming no principle is violated,
  or a documented justification if a temporary exception is granted.
- Constitution compliance is reviewed at each sprint/milestone retrospective.

**Version**: 1.0.0 | **Ratified**: 2026-03-26 | **Last Amended**: 2026-03-26
