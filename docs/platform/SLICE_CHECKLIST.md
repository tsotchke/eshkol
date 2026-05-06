# Slice Checklist

Use this checklist for every platform topic branch before merge.

## 1. Scope

- the branch owns one bounded subsystem change
- the branch name follows `topic/platform-*`
- unrelated roadmap work is not mixed into the branch
- unrelated cleanup is deferred unless required by the slice

## 2. Architecture

- the slice matches an approved workstream
- any boundary decision is recorded in `DECISIONS.md`
- profile semantics are explicit, not implied by existing hosted behavior
- hosted and freestanding expectations are both considered

## 3. Implementation

- files changed are consistent with the slice boundary
- dangerous behavior is gated by profile or explicit configuration
- hosted defaults are preserved unless the slice intentionally changes them
- error paths and invalid combinations are handled explicitly

## 4. Documentation

- the relevant platform doc is updated
- `docs/platform/README.md` is updated if program status changed
- command-line or user-facing behavior changes are documented where users will find them
- deferred follow-up work is recorded rather than left implicit

## 5. Verification

- the narrowest relevant build target was run
- direct tests for the new behavior were run
- failures from invalid combinations are covered when applicable
- manual verification steps are written down if automation does not exist yet

## 6. Integration

- branch is clean before merge
- commits are coherent and message quality is acceptable
- merge into `feature/platform-freestanding` uses an explicit merge commit
- merge-back to `feature/v1.2-scale` is evaluated explicitly, not assumed

## 7. Release Safety

- hosted behavior has not regressed
- the slice does not silently expand platform support claims beyond documented support tiers
- experimental behavior is labeled as experimental
- kernel-only artifacts are not used to justify compiler/runtime merges

## 8. After Merge

- integration branch status is clean
- any follow-up topic branches are named and scoped
- milestone progress is updated if the slice changes program status materially
- remaining risks are recorded if the slice leaves known limitations in place
