# Eshkol Public Release Readiness Report

**Branch:** `master`
**Release line:** `v1.3.x-evolve`
**Current release-candidate head:** the commit containing this report
**Published base tag:** `v1.3.0-evolve` (`e619da2f`)
**Date:** 2026-07-08

This report is the public-facing release gate for the v1.3 evolve line. It
replaces the older v1.2 historical readiness report; older audit material now
lives under `docs/reports/`.

## Verdict

Eshkol is locally release-ready at the commit containing this report, pending
only the final GitHub CI matrix result for that exact head before publishing a
follow-up public tag.

The already-published `v1.3.0-evolve` tag is green. The current head adds
release-gate portability and hardening after that tag, so it should become the
public patch release once CI completes green.

## Public Release Gates

| Gate | Result |
|---|---|
| Full local suite | `39` suites, `594` tests passed, `0` failed |
| SICP full-book gate | `88/88` probes, JIT and AOT |
| VM parity gate | `56/56` supported probes |
| AD validated bounds gate | JIT `22/22`, AOT `22/22` |
| ICC readiness | `ready`, score `100/100`, `0` contract gaps |
| Agent FFI smoke | `complete 7/7`, HTTPS GET and POST probes passed |
| Secret scan | No obvious key material; one source-code comment false positive |
| Tracked junk scan | No tracked `.DS_Store`, logs, temp files, caches, or build dirs |
| Git state | Clean `master`, pushed to `origin/master` |

## Current Public Head

Post-tag hardening before this cleanup (`44a61e28`) included:

- Locale-portable SICP release gate.
- Locale-portable VM parity and AD validated-bounds gates.
- VM support for `string-byte-length`.
- Explicit VM parity manifest entries for native-only or justified operations.
- ICC smoke checks that depend on build success rather than generator-specific
  output text.

These changes do not alter the language surface; they make the public release
proof reproducible on more hosts.

## Boundary Policy

The release line should not depend on arbitrary hardcoded boundary values. Fixed
limits are acceptable only when they are:

- Derived from the runtime, platform, data shape, or declared ABI.
- Part of a named language/runtime contract.
- A defensive cap with an explicit error path and test coverage.
- A test budget whose purpose is documented in the harness.

Magic constants that silently truncate, clamp, or skip behavior are not release
quality. Recent hardening replaced several such assumptions with checked bounds,
derived dimensions, overflow guards, or explicit parity waivers.

## Tsotchke-chan Discourse

tsotchke-chan is part of the release discourse as a guard and coordination
signal, not as a replacement for Eshkol's own proof gates.

Current local guard status:

- Last validation: `ok: true`.
- Critical failures: none.
- Checkpoint count: `30`.
- Latest validation timestamp: `2026-07-08T06:48:05Z`.

For this release pass, tsotchke-chan is used for:

- Discourse around public readiness and operational posture.
- Guard/checkpoint evidence that the assistant substrate is coherent.
- Text/code review support when grounded in repo-local gates.

It is not being used to dispatch remote Enki jobs for release-critical work in
this pass. Public readiness is decided by the local suite, ICC, VM parity, AD
oracles, SICP gates, and GitHub CI for the exact commit being tagged.

## Public Repository Hygiene

The repository root is now reserved for durable public documents:

- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `ROADMAP.md`
- Release notes and this readiness report
- Language/design references that are intended as first-click public material

Internal campaign reports, depth reports, sanitizer reports, and Noesis/Eshkol
handoff notes are retained under `docs/reports/` instead of cluttering the root.

## Release Rule

Do not publish another public tag from this branch until the GitHub Actions
matrix for the commit containing this report completes successfully. If it does,
publish a new patch tag instead of moving `v1.3.0-evolve`.
