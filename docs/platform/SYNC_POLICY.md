# Sync Policy

## Purpose

This document defines how `feature/v1.2-scale` and `feature/platform-freestanding` stay convergent while both streams are active.

The policy is intentionally asymmetric:

- `feature/v1.2-scale` is the roadmap source branch
- `feature/platform-freestanding` regularly absorbs roadmap progress
- platform work merges back to the roadmap branch only when a slice is stable and release-safe

## Core Rules

- never sync from a dirty worktree
- sync only from committed, reviewable changes
- prefer explicit merge commits for branch-to-branch sync points
- use topic branches only for bounded platform slices, not for ongoing roadmap sync
- treat sync as operational work, not as an afterthought at release time

## Default Direction

The default direction is:

- `feature/v1.2-scale` -> `feature/platform-freestanding`

Do this regularly so the platform branch reflects the current compiler, runtime, build, and release work.

The reverse direction is selective:

- `feature/platform-freestanding` -> `feature/v1.2-scale`

Do this only when a platform slice:

- preserves hosted behavior
- has been validated
- fits the current release scope
- does not require downstream kernel artifacts to justify its merge

## Sync Cadence

Use this cadence unless there is a reason to tighten it further.

- daily while shared files are changing heavily
- every 2-3 days during lower churn
- immediately before or after work in shared hotspot files
- immediately after a major roadmap slice becomes stable and committed

Do not let the branches drift for more than a week if both streams are active.

## Hotspot Files

Sync before touching or merging around these areas when possible:

- `CMakeLists.txt`
- `lib/core/runtime*`
- `lib/core/arena_memory*`
- `lib/backend/llvm_codegen.cpp`
- `exe/eshkol-run.cpp`
- `docs/platform/*`

These files produce disproportionate merge friction and should be kept as current as possible on the platform branch.

## Sync Modes

### Full roadmap sync

Use a full sync when the roadmap branch contains coherent, stable work that the platform branch should absorb as-is.

Example:

```bash
git -C ~/Desktop/eshkol-platform checkout feature/platform-freestanding
git -C ~/Desktop/eshkol-platform merge --no-ff feature/v1.2-scale -m "merge: sync v1.2-scale into platform"
```

Use this as the default mode once the roadmap slice is ready.

### Selective roadmap sync

Use a selective sync when `feature/v1.2-scale` contains unfinished, experimental, or destabilizing work that the platform branch should not absorb yet.

Options:

- cherry-pick a finished roadmap commit
- merge a narrower roadmap topic branch
- delay the sync until the roadmap work stabilizes

Selective sync is the exception, not the default.

## When To Sync

Sync `feature/v1.2-scale` into `feature/platform-freestanding` when any of the following is true:

- a meaningful roadmap slice is complete and committed
- the roadmap build becomes stable after churn in shared files
- platform work is about to enter a shared hotspot file
- the branches have been diverging for several days

For the current program, the next likely sync point is when the active roadmap build is stable enough to merge without dragging obvious breakage into the platform branch.

## Merge-Back Criteria

A platform slice is eligible to merge back into `feature/v1.2-scale` only when:

- the slice has already landed on `feature/platform-freestanding`
- relevant validation passed on the integration branch
- hosted defaults are unchanged or intentionally and explicitly revised
- platform docs are current
- the slice is useful to the roadmap now rather than merely future-facing

If those conditions are not met, keep the slice on the platform branch and continue forward there.

## Topic Branch Relationship

Platform topic branches should be cut from the current `feature/platform-freestanding` head, not from `feature/v1.2-scale`.

Sequence:

1. sync roadmap into platform if needed
2. cut `topic/platform-*` from updated `feature/platform-freestanding`
3. implement and validate the slice
4. merge the topic branch back into `feature/platform-freestanding`
5. decide explicitly whether the result should merge back to `feature/v1.2-scale`

This keeps platform slice history independent from roadmap sync history.

## History Policy

- use merge commits for `v1.2-scale` -> `platform-freestanding`
- use merge commits for `topic/platform-*` -> `platform-freestanding`
- rebase only unpublished personal topic branches if needed
- do not rewrite shared branch history as part of ordinary sync work

This keeps integration history auditable and reduces confusion when multiple streams are active.

## Failure Handling

If a sync produces conflicts:

- resolve them on the receiving branch
- keep the source branch untouched
- validate immediately after the merge
- record any new boundary issue in `DECISIONS.md` if the conflict exposed a real architectural ambiguity

If the roadmap branch is temporarily unstable:

- do not force the platform branch to absorb it immediately
- wait for the roadmap slice to stabilize or sync selectively

## Operational Summary

Use this as the default operating rule:

- roadmap work lands on `feature/v1.2-scale`
- stable roadmap work is merged into `feature/platform-freestanding` frequently
- platform slices land through `topic/platform-*`
- only stable, validated, release-safe platform slices merge back to `feature/v1.2-scale`

This is how the two streams stay convergent without forcing the platform program to move at the same granularity as day-to-day roadmap churn.
