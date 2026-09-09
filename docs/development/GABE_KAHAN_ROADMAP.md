# Gabriel “Gabe” Kahen — Eshkol Subsystem Ownership Roadmap

- **Owner:** [Gabriel-Kahen](https://github.com/Gabriel-Kahen)
- **Maintainer sponsor:** Eshkol maintainers
- **Time budget:** about 10 hours per week
- **Cadence:** one milestone pull request every two to four weeks
- **Last reviewed:** 2026-08-31
- **Review again:** after 12 weeks, or whenever the maintainer changes the active work packet

This is a rolling contributor plan, not a release plan. The canonical project
direction remains [`ROADMAP.md`](../../ROADMAP.md), and the canonical contribution
rules remain [`CONTRIBUTING.md`](../../CONTRIBUTING.md). This file answers a
narrower question: what substantial subsystem can Gabe own in durable,
reviewable increments within a 10-hour week?

## Demonstrated level

This plan is based on completed Eshkol work, not academic seniority:

- [PR #19](https://github.com/tsotchke/eshkol/pull/19) added the ESKM model and
  tensor checkpoint format, CRC32 and version validation, arena-backed loading,
  LLVM/JIT and bytecode-VM integration, type-checker surface, documentation,
  and 293 lines of round-trip/corruption tests. The merged change was 1,559
  additions across 16 files.
- [PR #28](https://github.com/tsotchke/eshkol/pull/28) hardened
  `eshkol-server` binding, authentication, CORS, and compile-request exposure,
  with a dedicated security surface test.
- [PR #18](https://github.com/tsotchke/eshkol/pull/18) fixed REPL/module-loader
  behavior and added a 243-line end-to-end regression.
- Issues [#549](https://github.com/tsotchke/eshkol/issues/549) through
  [#553](https://github.com/tsotchke/eshkol/issues/553) identify concrete model
  payload, tensor shape, tensor AD, cross-entropy, and RNG parity failures.

That evidence supports medium-to-deep systems work spanning file formats,
runtime safety, VM/native parity, and security boundaries. Tests and docs remain
part of the work, but they are evidence for subsystem changes rather than the
ceiling of the assignment.

## Goal

Give Gabe durable ownership of Eshkol's model/checkpoint serialization boundary:
format evolution, bounded and adversarial loading, compatibility, native/VM
parity, atomic persistence, and the public contract around those behaviors.

After the first 12 weeks, Gabe should be able to:

- write and defend a format/compatibility decision before changing bytes on disk;
- harden native and VM deserializers against truncation, corruption, overflow,
  resource exhaustion, and partial writes;
- evolve ESKM without breaking v1.2 checkpoints or silently accepting unknown
  data;
- keep model/tensor behavior consistent across native JIT, native AOT, VM
  source, and VM bytecode;
- ship a substantial subsystem change as several independently reviewable PRs;
  and
- leave executable compatibility evidence and a clear next-maintainer handoff.

This roadmap does **not** make Gabe responsible for release management,
production infrastructure, GPU/XLA backends, or the full AD compiler. It does
authorize runtime/compiler-adjacent changes required by an approved
serialization milestone.

## Working agreement

Use the following weekly budget as a default, not a quota:

| Activity | Hours |
|---|---:|
| Read/design the current milestone | 2 |
| Reproduce, inspect fixtures, or measure compatibility | 1 |
| Implement one bounded subsystem slice | 4 |
| Run focused and cross-engine tests; document evidence | 2 |
| Respond to review or update this roadmap | 1 |

Keep no more than one milestone implementation pull request open at a time. A
small prerequisite/test PR is fine when it clearly reduces the next milestone's
risk.

If a task cannot be explained as one observable behavior and one acceptance
gate, split it before coding.

## Scope boundaries

### Green zone — owned subsystem

- `inc/eshkol/model_io.h`, `lib/core/model_io.cpp`, and
  `tests/core/model_io_test.cpp`;
- the VM model/tensor serialization path and its native-call bindings;
- model/tensor checkpoint format documentation and compatibility fixtures;
- bounded parsing, checksums, versioning, shape/dtype validation, integer
  overflow checks, atomic-save behavior, and corruption refusal;
- focused LLVM/JIT, type-checker, REPL, and CMake surface changes required to
  expose an approved serialization capability;
- cross-engine model/tensor I/O parity tests and diagnostic corpus cases; and
- closely related server or module-loader hardening when explicitly selected
  as the active track.

### Design review before changing

- any on-disk format byte, version, checksum rule, or compatibility promise;
- public API signatures or native IDs;
- general tensor allocation/shape rules outside the I/O boundary;
- reusable parser/arena abstractions shared by other subsystems;
- checked-in generated artifacts or `.icc/` policy evidence; and
- a milestone expected to exceed roughly 800 non-generated lines or four weeks.

### Outside this ownership track

- automatic differentiation lowering, memory ownership/regions,
  continuations, or the object ABI except for a reviewed serialization adapter;
- XLA, CUDA, Metal, WebGPU, platform release matrices, or artifact publishing;
- GitHub Actions billing, runner administration, branch protection, secrets,
  or release cuts; and
- a currently failing release-blocking pull request.

These boundaries prevent unrelated critical-path work from being folded into a
serialization milestone. They are not a judgment about Gabe's ability.

## Generated-file rule

Never repair a generated file by hand. Change its source and run its generator,
or ask a maintainer when the source is unclear. Important examples include:

- `docs/api/` — generated by `python3 scripts/gen_api_docs.py`;
- `.icc/` ledgers, baselines, and readiness artifacts — maintainer-owned; use
  the repository's current generator when explicitly assigned;
- `lib/backend/vm_prelude_cache.h` — generated by
  `scripts/regenerate_vm_prelude_cache.sh`; and
- checked-in site/WASM artifacts.

A surprising generated diff is a reason to stop and ask, not a reason to
commit it blindly.

## Twelve-week path

| Weeks | Theme | Deliverable | Acceptance |
|---|---|---|---|
| 1–2 | Recover the ESKM contract | Write the exact v1.2 byte-layout/limits/compatibility table and check in representative golden checkpoints | Current native and VM loaders agree on every fixture; each fixture records expected outcome and hash |
| 3–4 | Adversarial admission | Add a table-driven corruption/truncation/overflow corpus, then close one uncovered loader failure after #555 lands | Every malformed case fails before allocation/use, with no partial model returned and no crash |
| 5–6 | Cross-engine compatibility | Add native-write/VM-read and VM-write/native-read coverage for tensors and multi-tensor models | Round trips preserve names, rank, dimensions, dtype, element count, and payload bytes |
| 7–8 | Crash-consistent saving | Design and implement atomic checkpoint replacement without exposing a partial destination | Injected failure before commit preserves the old checkpoint; success leaves one valid new checkpoint and no orphan temp file |
| 9–10 | Format evolution | Propose and implement one backward-compatible ESKM extension or explicit v2 decision | Old v1.2 fixtures still load; unknown mandatory features or versions refuse loudly; the format document is normative |
| 11 | Resource and fuzz evidence | Add bounded randomized/adversarial cases around counts, dimensions, lengths, and checksums | Sanitizer/focused fuzz run has a stated seed/budget and zero crashes, hangs, or unbounded allocations |
| 12 | Consolidate ownership | Publish a subsystem status/handoff and prioritized next six-month backlog | Record merged PRs, compatibility matrix, open risks, performance numbers, and the next milestone |

The sequence can pause for exams or review latency. Resume at the same phase;
do not compensate by taking a larger task.

## Work packets

The serialization track is primary. The other tracks are real alternatives if
Gabe and the maintainer deliberately change ownership; they are not side quests
to mix into a serialization PR.

### GK-SER-01 — ESKM compatibility corpus and normative format

Recover the exact format implemented by PR #19 from `model_io` sources and
existing tests. Check in small v1.2 golden tensors/models covering scalar,
empty, large, multi-tensor, unusual rank, and corrupt inputs. Document byte
order, integer widths, checksum coverage, count/size limits, dtype rules, and
version behavior. The fixtures—not prose alone—become the compatibility oracle.

### GK-SER-02 — Bounded, fail-closed loader

Build on issue #549 and its active implementation rather than duplicating it.
After that PR settles, audit both native and VM readers for integer overflow,
shape-product overflow, allocation-before-validation, duplicate names, unknown
dtype/version, truncated metadata/payload, trailing bytes, checksum mismatch,
and declared sizes larger than the file. Add missing refusals and exact tests.

### GK-SER-03 — Native/VM differential round trips

Create a matrix that writes with each available engine and reads with every
other engine. Compare semantic metadata and payload bytes, not printed summaries.
Include a negative matrix in which every engine rejects the same malformed
checkpoint class.

### GK-SER-04 — Atomic and durable checkpoint saves

Specify the destination replacement contract, including same-directory temp
files, write/flush/close errors, rename behavior, permissions, cleanup, and what
the caller sees after interruption. Implement the smallest portable contract
and add injected-failure tests. Do not claim power-loss durability unless the
implementation and platform tests actually provide it.

### GK-SER-05 — ESKM evolution

Prepare a short design decision for the next format revision: optional metadata,
endianness, additional dtypes, compression/chunking, or streaming. It must state
how old readers behave, how new readers recognize v1.2, which fields are
mandatory, and how unknown mandatory features fail. Implementation follows
review of the byte-level decision.

### GK-SER-06 — Fuzz and resource-bound campaign

Add a deterministic generator/mutator for headers, counts, dimensions, names,
payload lengths, and CRCs. Gate on a clear time/input budget and record seeds.
The useful output is a minimized fixture plus a normal regression test, not a
large permanent corpus of random files.

### GK-SRV-01 — Compile-service security follow-up

An alternate ownership track building on PR #28: request-size/resource limits,
authentication failure behavior, origin parsing, compile isolation, response
content types, and adversarial HTTP tests. Any public-listen or execution-policy
change requires a threat-model review first.

### GK-REPL-01 — Module-loading correctness

An alternate ownership track building on PR #18: precompiled/source fallback,
module identity, load cycles, cache invalidation, path normalization, and
sourceful diagnostics across `-e`, file, and REPL execution. VM/compiler changes
are acceptable when the reproducer and ownership boundary are agreed first.

## Active task

Update this table in the same pull request that starts or finishes a packet.
Keep completed rows so the file serves as a durable handoff log.

| Status | Packet | Concrete scope | Acceptance command | PR |
|---|---|---|---|---|
| next | GK-SER-01 | Recover the v1.2 ESKM contract and propose the golden compatibility corpus; coordinate with active issue #549 / PR #555 before touching overlapping loader code | `model_io_test` plus native/VM fixture matrix recorded in the task PR | — |

Allowed status values are `next`, `in progress`, `review`, `done`, and
`blocked`. A `blocked` row includes one sentence naming the decision or resource
needed.

## Starting a task

1. Pull current `master` and confirm the task is not already covered by an open
   pull request.
2. Write the observable before/after behavior in the issue or PR description.
3. For a byte-format change, write the compatibility decision and example bytes
   before editing code.
4. Identify the narrowest existing test command before editing code.
5. Create a branch named `gabe/<short-task-name>`.
6. Keep the first commit to the reproducer or measurement when the task involves
   a bug.
7. Implement only the approved packet.
8. Run focused tests first, then the required repository checks below.

## Local verification

Use the repository instructions for platform-specific LLVM setup. A normal
native build is:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

During development, run the smallest relevant selection:

```bash
ctest --test-dir build -N
ctest --test-dir build --output-on-failure -R '<focused-test-name>'
```

The serialization track normally builds and runs at least:

```bash
cmake --build build --target model_io_test eshkol-run eshkol-vm-standalone-test --parallel
ctest --test-dir build --output-on-failure -R 'model_io|vm_canonical_stdlib|vm_prelude_cache'
```

For documentation or test-only changes, also run the checks that match the
files touched:

```bash
python3 scripts/gen_api_docs.py --check
python3 scripts/check_test_coverage.py
git diff --check
```

Do not claim a platform result that was not run. In the PR, distinguish `PASS`,
`FAIL`, and `NOT RUN` explicitly.

## Pull-request contract

Every PR should contain:

- a one-paragraph statement of the observed problem;
- the exact files intentionally in scope;
- the acceptance test and its result;
- a negative control for new or changed test logic;
- the ESKM versions and producer/consumer engines exercised;
- the compatibility and resource-limit effect of any format/loader change;
- a note naming platforms or engines not run locally; and
- no unrelated cleanup.

Suggested PR description:

```markdown
## Problem

## Scope

## Evidence before

## Change

## Verification
- PASS:
- NOT RUN:

## Out of scope
```

Prefer a small PR that teaches one thing over a large PR that needs several
reviewers to reconstruct its intent.

## Stop and ask

Stop and post a short note when any of these occurs:

- the task requires a file in the maintainer-owned zone;
- the same unexplained failure survives two focused approaches or two hours;
- a generated diff is much larger than the source change;
- a serialized-format change lacks an approved compatibility decision;
- the fix changes the public language rule, object ABI, or release claim beyond
  the active serialization milestone;
- local and GitHub results disagree and a single rerun does not explain it;
- a test passes only after weakening an expected value, timeout, or coverage
  floor; or
- the PR grows beyond the approved scope.

The handoff note should include the command, the shortest relevant output, what
was already tried, and the current commit SHA. Do not spend the remaining weekly
budget guessing inside an unfamiliar subsystem.

## Weekly log

Append one row per active week. Keep entries factual and short.

| Week ending | Hours | Packet | Result | Next question |
|---|---:|---|---|---|
| — | — | GK-SER-01 | Roadmap recalibrated from PRs #18, #19, and #28 | Confirm the compatibility-corpus slice against #549/#555 |

## Maintainer review checklist

Before assigning a packet, the maintainer confirms:

- the task is not duplicated by an open issue or PR;
- the milestone fits two to four 10-hour weeks and can be split into reviewable
  commits or PRs;
- the acceptance command is available without private infrastructure;
- the task does not sit on the active release critical path; and
- a maintainer can review it within a few days.

At the 12-week review, decide whether Gabe continues ESKM/model-I/O ownership or
takes the server-security or REPL/module-loading track. Expansion should follow
the architecture and evidence in completed PRs, not academic seniority.
