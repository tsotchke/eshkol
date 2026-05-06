# Implementation Workflow

## Purpose

This document defines the working procedure for implementing the freestanding / platform program at production quality while the main roadmap continues in parallel.

The goal is to keep platform work:

- isolated from release-critical churn
- continuously integrable
- fully documented
- testable at each slice boundary

## Working Topology

Use a dedicated worktree layout.

- `~/Desktop/eshkol`
  - primary roadmap worktree
  - branch: `feature/v1.2-scale`
- `~/Desktop/eshkol-platform`
  - platform integration worktree
  - branch: `feature/platform-freestanding`
- optional topic worktrees
  - created as needed for short-lived topic branches
  - removed after merge

This layout keeps the release branch, the platform integration branch, and any active topic slice physically separate.

## Branch Roles

- `feature/v1.2-scale`
  - release integration branch
  - source of truth for current roadmap work
- `feature/platform-freestanding`
  - platform integration branch
  - accumulates accepted platform slices
- `topic/platform-*`
  - short-lived implementation branches
  - each owns one bounded slice

Do not implement freestanding work directly on `feature/platform-freestanding` unless the change is trivial and administrative. Use a topic branch by default.

## Topic Slice Lifecycle

Every platform slice follows the same lifecycle.

1. Confirm the slice boundary.
   The work must map cleanly to one subsystem or one bounded cross-cutting change.
2. Confirm the architectural contract.
   If the change affects profiles, syntax, runtime family boundaries, BSP contracts, or repo boundaries, record the decision first in `DECISIONS.md`.
3. Cut a topic branch from `feature/platform-freestanding`.
   Use `topic/platform-...` naming.
4. Implement the slice.
   Keep changes focused. Avoid mixing opportunistic refactors with platform semantics.
5. Update docs in the same branch.
   Any change to public or cross-subsystem behavior updates the relevant docs in `docs/platform/`.
6. Add or update tests.
   Every accepted slice adds verification for the new behavior or constraint.
7. Validate locally.
   Run the narrowest useful build and tests first, then the broader checks required by the slice.
8. Commit the slice cleanly.
   Use a message that describes the subsystem and behavior change.
9. Merge into `feature/platform-freestanding` with an explicit merge commit.
   The merge commit is the integration marker for the slice.
10. Decide whether the slice is safe to merge back to the roadmap branch now or later.

## Merge Discipline

### Roadmap syncs into platform

Merge `feature/v1.2-scale` into `feature/platform-freestanding` regularly. Use explicit merge commits so platform sync points are visible in history.

Use [SYNC_POLICY.md](SYNC_POLICY.md) for the exact cadence, hotspot rules, and merge-back criteria.

Recommended cadence:

- daily while shared compiler/runtime files are churning heavily
- every 2-3 days during lower churn
- immediately after a major roadmap change in overlapping files

### Topic branches into platform

Merge accepted topic branches into `feature/platform-freestanding` with `--no-ff`.

Reasons:

- preserves slice boundaries
- keeps integration history legible
- makes rollback and bisect easier

### Platform back into roadmap

Only merge platform work back into `feature/v1.2-scale` when all are true:

- hosted behavior is preserved
- the slice has targeted tests
- the slice fits the current release window
- the slice does not depend on downstream kernel artifacts
- the docs reflect the new state

## Validation Expectations

Validation is slice-specific, but these rules are global.

- every slice must have at least one direct verification path
- hosted regressions are treated as release blockers
- freestanding work may begin with smoke coverage, but no slice merges without at least one concrete test or executable verification step
- CI expectations must be documented before a slice becomes required for merge-back

For examples:

- profile changes require parser/resolution tests
- LLVM path changes require codegen or link verification
- runtime boundary changes require symbol hygiene checks
- BSP changes require QEMU smoke tests once the BSP framework exists

## Commit Standards

Commit messages should be subsystem-scoped and descriptive.

Good examples:

- `toolchain: add execution profile model`
- `runtime: split hosted and freestanding bootstrap surface`
- `docs: add platform program documentation`

Avoid vague commits such as:

- `wip`
- `more platform work`
- `fix stuff`

## Documentation Standards

A slice is incomplete if it changes any of the following without updating docs:

- execution profile semantics
- runtime family boundaries
- validation requirements
- merge policy
- target support expectations
- kernel handoff criteria

At minimum, update:

- the relevant detailed doc
- `README.md` in `docs/platform/` if the program status changes materially
- `DECISIONS.md` if the change establishes or revises a boundary decision

## Escalation Rules

If a task starts expanding beyond its slice boundary:

- stop broadening the branch
- record the issue
- create a follow-up topic branch instead of silently absorbing unrelated work

This is especially important in:

- `llvm_codegen.cpp`
- runtime core files
- VM core
- build system integration

## Definition of Done for a Slice

A platform slice is done only when all are true:

- implementation is complete for the scoped behavior
- docs are updated
- tests or executable validation exist
- the branch is clean
- the slice is merged into `feature/platform-freestanding`
- any deferred follow-up is recorded explicitly
