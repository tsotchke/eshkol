---
kind: project
status: current
owner-area: release
since: v1.3.5
sources:
  - .github/workflows/release.yml
  - scripts/release_readiness_guard.py
  - scripts/release_autopilot.py
  - scripts/run_v1_3_readiness.sh
  - scripts/run_v1_3_release_producers.sh
  - scripts/run_release_invariant_probes.sh
  - scripts/lib/release_invariant_probes.sh
  - scripts/lib/evidence_paths.sh
  - scripts/check_surface_counts.py
  - scripts/check_changelog_completeness.py
  - scripts/check_package_manifest.py
  - scripts/verify_release_package.py
  - scripts/verify_site_release.py
  - tests/coverage/release_record.json
  - tests/toolchain/test_v1_3_release_evidence_recipe.py
  - .icc/completion-oracles.yaml
  - .icc/package-manifest.yaml
  - docs/design/adr/0010-closed-loop-assurance.md
  - docs/design/adr/0014-release-invariant-contracts.md
---
# Release Process

What stands behind an Eshkol tag: the workflow that builds and publishes a
release, the evidence it regenerates, the gate that binds that evidence to the
tagged commit, and the short list of things a maintainer does by hand.

One rule organises everything on this page. **A release is certified by
evidence produced from the commit the tag names, and by nothing else.**
Evidence from another commit cannot certify a cut, and evidence that is still
pending blocks publication.

The design rationale is in
[ADR-0010 (closed-loop assurance)](../design/adr/0010-closed-loop-assurance.md)
and [ADR-0014 (release invariant contracts)](../design/adr/0014-release-invariant-contracts.md).
The measured figures for the current release are in the "Final verification"
section of [RELEASE_NOTES.md](../../RELEASE_NOTES.md#final-verification).

## The Release workflow

`.github/workflows/release.yml` has two triggers:

| Trigger | Publishes | Readiness gate |
|---------|-----------|----------------|
| Push of a tag matching `v*` | Yes | Blocking |
| `workflow_dispatch` with `candidate_tag` (and optionally `strict_readiness`) | Never | Advisory, or blocking with `strict_readiness=true` |

A manual dispatch is always non-publishing. Its `candidate_tag` input is only a
label, so the dry-run archives carry the exact names the tag run will use.
Runs for one ref are serialised and never cancel each other.

The jobs, in dependency order:

1. **`unix-release-matrix`** builds ten packages: Linux x86-64 and ARM64 in the
   lite, XLA and CUDA configurations, and macOS ARM64 and x86-64 in the lite and
   XLA configurations. Each job validates the tag label, configures and builds,
   verifies that `stdlib.bc` targets a generic CPU
   (`scripts/verify_portable_stdlib.py`), checks the WebAssembly import glue on
   the lite configuration (`scripts/check_wasm_imports.py`), runs the
   configuration's test command plus the agent HTTP, SSE and capability tests,
   and then packages.
2. **`prefetch-windows-llvm-archives`** downloads the pinned LLVM SDK archives
   once, on Linux, into a cross-OS cache that the Windows jobs restore.
3. **`windows-release-matrix`** builds five packages: Windows x86-64 and ARM64
   in lite and XLA, and Windows x86-64 CUDA. In addition to the steps above it
   verifies the exports the cache-disabled JIT path needs
   (`scripts/verify_windows_runtime_exports.py`).
4. **`release-readiness-gate`** runs on a self-hosted Linux x86-64 runner that
   has the ICC oracle provisioned. It is described in the next section.
5. **`publish-release`** needs both asset matrices and the readiness gate. It downloads the packaged
   assets, validates that exactly the fifteen expected archives are present and
   non-empty, writes `SHA256SUMS.txt`, extracts the current release's section
   of `RELEASE_NOTES.md`, and on a tag push creates the GitHub release. On a
   dispatch it uploads the validated set as a `release-dry-run-<sha>` artifact
   instead.
6. **`bump-homebrew-tap`** runs only on a tag push, after publication.

### Packaging, per platform job

Packaging is where a release can lose a file without any test noticing, so
each asset job checks its own staged directory before archiving it:

- `scripts/stage_linux_runtime_dependencies.py` (Linux) and
  `scripts/stage_third_party_licenses.py` stage the runtime dependencies and
  the licences of the bundled libraries.
- `scripts/check_package_manifest.py --package-dir <dir> --platform <linux|macos|windows>`
  compares the staged directory with the single declared package surface,
  `.icc/package-manifest.yaml`: binaries, stdlib artifacts, static libraries
  with the platform-correct prefix and extension, the full `lib/**/*.esk`
  module tree as a path-set mirror, documents, and the CMake integration
  modules. A missing file is reported as a manifest violation naming the file.
- `scripts/verify_release_package.py` runs a smoke program and an agent smoke
  program from the staged package through both a cold and a warm
  `eshkol-run -r` cache path, with no system-wide installation to fall back
  on, rejects the in-process JIT fallback, and checks the reported version
  against the tag.

Both archive-producing paths (the Unix `tar.gz` path and the Windows `zip`
path) run the manifest check and the package verifier.

### Publication

`publish-release` refuses to overwrite: if a release for the tag already
exists the job fails rather than appending assets. The release is created as a
draft, assets are uploaded, and then it is published, so an interrupted upload
never leaves a public partial release. A tag containing `alpha`, `beta` or `rc`
is published as a prerelease.

The release notes are produced by
`scripts/release_readiness_guard.py notes`. It takes everything above the first
horizontal rule in `RELEASE_NOTES.md`, requires the heading to be
`# Eshkol <tag> — Release Notes`, and fails when the text still contains the
marker `RELEASE_EVIDENCE_PENDING`. Only a dispatch passes `--allow-pending`, so
a dry run can be rehearsed before the evidence is final while a tag push
cannot publish notes that still promise evidence.

### Homebrew tap bump

`bump-homebrew-tap` fetches the source archive GitHub generates for the tag,
hashes it, and rewrites the top-level `url` and `sha256` lines of the formula
in the tap repository. The `resource` blocks that pin the bundled agent-FFI
dependencies keep their own checksums. The job needs a tap token secret with
write access to the tap; without one it skips with a notice. Prerelease tags
do not bump the tap, and a formula already at the tag is left untouched.

## The readiness gate

`release-readiness-gate` turns "the oracle said ready" from a statement about a
workstation into a property of the released commit. `publish-release` depends
on it, so no asset is published unless it passes.

### Bound to the exact commit

`scripts/release_readiness_guard.py` enforces identity in two places.

- `bind`, before any evidence is produced, requires that the checkout `HEAD`
  equals the full 40-character release SHA, that the checkout has no tracked
  source change, and that the ICC repository resolves to that checkout. When
  the configured repository name points elsewhere it registers a dedicated
  alias derived from the SHA and the workspace path; it never rebinds a
  maintainer's configured repository.
- `check`, before and after the verdict, repeats the identity test and then
  requires the verdict to be `status == "ready"` with a numeric score of
  exactly 100. A missing or non-numeric score is not a pass.

On a tag push, an unavailable oracle is a blocked release. The gate never
falls open to a green publish. On a default dispatch the same condition is a
warning; with `strict_readiness=true` it blocks exactly as a tag push does.

When every check passes, the job writes a receipt,
`release-readiness-receipt-<sha>`, with the schema
`eshkol.release-readiness.v1`: the commit SHA, the workflow run id and run
attempt, the target `v1.3.5-evolve`, `status: ready` and `score: 100`. A
receipt is only ever written for the commit and the run attempt that produced
it.

`tests/toolchain/test_release_readiness_guard.py` runs first in the job as the
negative controls for this logic: a mismatched SHA, a dirty checkout, a score
that is missing, non-numeric or below 100, and release notes with a planted
pending-evidence marker must each be refused, and a repository name that
resolves to a different checkout must get a separate verified alias rather
than a rebind.

### What the gate builds

The gate configures with the pinned LLVM major and builds, with at most four
parallel jobs:

- `build`: `RelWithDebInfo`, tests on, agent FFI on, and the Python bindings on
  against an isolated virtual environment that the job creates with pybind11,
  NumPy and PyYAML;
- `build-fuzz`: only the bounded ESKM model-loader fuzz probe;
- `build-quantum`: the quantum-enabled tree that complete language coverage
  requires;
- `build-asan-ubsan`: built later by the sanitizer producer, in its own tree,
  so the main build and its fingerprint stay untouched.

The toolchain is provisioned on the runner ahead of time. The workflow
installs no system packages there; its preflight step only resolves what is
present and fails with a specific message when something is missing. See
[SELF_HOSTED_RUNNERS.md](SELF_HOSTED_RUNNERS.md).

## The evidence recipe

`scripts/run_v1_3_readiness.sh` is the one recipe, and the workflow delegates
to it in four phases, each a separate step with its own time budget:

```
scripts/run_v1_3_readiness.sh --phase baseline
scripts/run_v1_3_readiness.sh --phase smoke
scripts/run_v1_3_readiness.sh --phase final-evidence
scripts/run_v1_3_readiness.sh --phase readiness
```

Run without `--phase`, it executes all four in order.

| Phase | What it produces |
|-------|------------------|
| `baseline` | Archives any earlier trace cohort outside the active trace root, captures the build cohort manifest, then runs `scripts/run_language_coverage.sh` (which runs the complete suite once and records exactly one `core_suite` PASS) and `scripts/run_vm_parity.sh`. |
| `smoke` | The Taylor monomorphization equivalence gate, the ESKM model-loader fuzz smoke, and the runtime smoke battery `scripts/run_icc_smoke.sh`, which includes the release invariant probes. |
| `final-evidence` | Refreshes the ICC index, runs `scripts/run_v1_3_release_producers.sh`, verifies the build cohort is unchanged, checks the evidence set with `scripts/verify_v1_3_release_evidence.py`, and only then asks ICC for the architecture grade (`icc architecture-verify` against `.icc/architecture-model.yaml`). |
| `readiness` | Re-verifies the evidence set and asks ICC for trace-aware readiness of the `v1.3.5-evolve` target. |

Three mechanisms keep the phases honest:

- **One cohort.** `scripts/check_release_build_cohort.py` fingerprints the
  compiler and runtime artifacts at the start and checks them before each later
  phase. Evidence from a rebuilt or mutated tree is refused.
- **Ordered phases.** `scripts/release_phase_state.py` records each completed
  phase against a phase id derived from the workflow run id and attempt. A
  phase requires its predecessor under the same id, so a failed or missing
  earlier phase cannot be resumed as a completed one.
- **Phase receipts.** `scripts/check_release_phase_receipts.py` confirms that
  the receipts a phase owes exist before the next phase starts.

`scripts/run_v1_3_release_producers.sh` owns the criteria the general smoke
battery does not: the node-identity gate, the nested-expression compile-time
budget, the dense tensor AD gate, the self-verdict scanner self-test, the test
coverage inventory, the recipe's own self-test, five named CTests recorded
through JUnit (`qubit_linearity_engine_parity_gate`,
`closure_upvalue_capacity_overflow_gate`, `abi_layout_pin_test`,
`v1_3_quoted_datum_kinds_runtime_smoke`, `python_bindings_capsule_lifetime`),
an ASan+UBSan build and a bounded sanitizer corpus run with leak detection,
the pinned Rosette Wire oracle, every tutorial example under both the JIT and
AOT, and the build-free checks (ledger integrity, oracle schema, false-green
audit, PowerShell encoding, public API documentation, generated API docs,
disclosure, required-context consistency, documentation-claims residual). The
evidence staleness gate runs last, over the fresh cohort.

`tests/toolchain/test_v1_3_release_evidence_recipe.py` tests the recipe itself:
producer ordering before grading, exact CTest cardinality, archive isolation
of a prior cohort, phase state bound to head, run and order, and fingerprint
fault injection.

### Evidence paths are absolute

`scripts/lib/evidence_paths.sh` fixes the meaning of a relative evidence path
in one place: **a relative `TRACE_DIR` or `ICC_TRACE_DIR` is relative to the
repository root**, and it is made absolute before any producer uses it.

The reason is that producers change directory as a matter of course.
`ctest --test-dir build --output-junit P` resolves a relative `P` inside
`build/`, and a harness that enters a scratch tree resolves it there, so a
relative path would name one file for the tool that writes the evidence and
another for the recorder that reads it. Every script that accepts an evidence
location from its environment sources the helper and calls
`eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT"` before first use; the Release
workflow passes an absolute `TRACE_DIR`. The path does not have to exist yet,
and symlinks are not resolved. The recipe test asserts all three properties:
every environment-reading script normalises before use, the helper's contract,
and a relative `TRACE_DIR` reaching `ctest` as an absolute JUnit path.

`BUILD_DIR` selects the build tree the harnesses use (default `build`), and
`QUANTUM_BUILD_DIR` the quantum tree (default `build-quantum`). Both accept a
relative path, read against the repository root. The full list is in
[environment-variables.md](../reference/runtime/environment-variables.md#test-gate-and-release-harness-variables).

### Receipts before the architecture grade

An invariant grade is release evidence only after its concrete probe has run
on the release build. Four typed `test_result` receipts are therefore emitted
**before** ICC grades the architecture model
(`scripts/lib/release_invariant_probes.sh`, shared by the smoke battery and by
`scripts/run_release_invariant_probes.sh`):

| Receipt | What is measured |
|---------|------------------|
| `abi_layout_pin` | The object header layout and its guard symbol, by running the built `abi_layout_pin_test`. |
| `abi_object_header_ratchet` | No new dependence on the object-header layout: `scripts/abi_header_inventory.py check` against the recorded baseline. |
| `closed_enum_dispatch_exhaustive` | Closed-enum dispatch is exhaustive and the gate is armed: `scripts/gate_exhaustive_dispatch.py`. |
| `ad_exactness_gate` | The live AD counter. A **positive case**: exact gradients report zero finite-difference evaluations. A **negative control**: a real finite-difference backward reports exactly its perturbations and turns the shipped assertion false, on both engines. Matmul AD tape node counts stay within their ratchet with exact gradients. |

VM parity supplies its own aggregate receipt, `vm_parity_gate`, from
`scripts/run_vm_parity.sh` in the baseline phase.

A probe distinguishes a measured outcome from an infrastructure failure. Only
a measured `PASS` or `FAIL` emits a typed receipt; a harness that could not
obtain a result emits none, so a tooling failure is never promoted to a product
verdict in either direction. The standalone probe script clears its trace file
first and requires exactly four probes with no failure and no infrastructure
failure, so a missing build cannot leave an earlier success standing.
`tests/toolchain/test_release_invariant_contracts.py` and
`tests/toolchain/test_release_probe_receipts.py` are registered in CTest and
assert these contracts, including the order of the recipe.

Compiler-assurance executions (the closed-enum mutation corpus and the
compiler-capability runs) are described in
[COMPILER_ASSURANCE.md](COMPILER_ASSURANCE.md).

### The readiness target

The `v1.3.5-evolve` target in `.icc/completion-oracles.yaml` lists 35 required
criteria: 20 runtime events, 14 test-evidence criteria and one
no-stubbed-paths criterion. Each is bound to a committed gate, script or CTest
name and states the command that produces its evidence. Examples: the
node-identity substrate and its span-coverage floor, the VM region evacuator's
flat-RSS and subtype-coverage gates, linear `Qubit` cloning as a compile-time
error, the nested-expression compile-time budget, the AD exactness gate, the
AD carrier manifest, the ABI layout pin, the sanitizer failure path, ESKM
model-loader fuzzing and engine parity, the ledger, oracle-schema, false-green
and staleness audits, public API and generated API documentation, the tutorial
example gate, the disclosure scan, required-context consistency, engine
semantic parity thresholds, and the Rosette Wire oracle.

`scripts/verify_v1_3_release_evidence.py` maps every test-evidence criterion to
its receipt name and requires exactly one passing `test_result` for each:
missing, duplicated, failed and unmapped evidence are all refused. A `FAIL`
trace is zero evidence by design.

## The release record and the consistency gate

`tests/coverage/release_record.json` is the single source for the release
facts that many documents restate and no generated manifest carries:

| Key | Meaning |
|-----|---------|
| `tag` | The release tag. |
| `previous_tag` | The preceding release tag; the base of the changelog range. |
| `release_date` | The release date, ISO format. |
| `status` | The status label, for example `SHIPPED`. |
| `ctest_total` | The `N` of CTest's own "100% tests passed ... out of N" line from the full run at the release commit. `null` until recorded. |
| `vm_parity_total` | The total of the `scripts/run_vm_parity.sh` summary line at the release commit. |

`scripts/check_surface_counts.py` grades the registered documents, the
release-facing documents and the generated site pages against the record and
against the surface manifests under `tests/coverage/`:

- every statement of the release date is extracted and compared, and a stated
  weekday is checked against the calendar;
- the roadmap heading and ladder rows for the tag carry the record's status
  and date;
- pre-release wording is refused once the status is `SHIPPED`;
- anchor documents must state the date at all, so a claim cannot be deleted to
  pass;
- `site/static/content/*.html` is graded as text, so a mirror that lags its
  Markdown source fails;
- a record-owned span must equal the record's rendering.

A record-owned span looks like this in Markdown:

```
<!-- release-record:vm-parity -->VM parity differential **340/340**<!-- /release-record -->
```

The keys are `ctest`, `ctest-cell`, `vm-parity` and `vm-parity-figure`. While a
total is `null` the span renders the claim without a number ("the full CTest
suite", "every registered test"), which reads correctly as it stands.

| Command | Effect |
|---------|--------|
| `python3 scripts/check_surface_counts.py --no-trace` | Grade only; write no trace. |
| `python3 scripts/check_surface_counts.py --sync` | Rewrite every graded release-record claim (date, status label, record-owned spans, CTest and VM-parity totals) from the record, then grade. |
| `python3 scripts/check_surface_counts.py --require-complete` | Also fail while the record still has a `null` total. |
| `python3 scripts/check_surface_counts.py --self-test` | Run the built-in red and green fixtures. |

`--ctest-log` and `--parity-log` grade the documented totals against the
captured output of a real run.

## Standing gates

The gates a release-facing change meets, all runnable locally from the
repository root:

| Gate | One-sentence contract |
|------|-----------------------|
| `scripts/check_surface_counts.py` | Every registered document's surface counts and every release-record claim agree with the manifests and the release record. |
| `scripts/check_changelog_completeness.py` | Every pull request merged in the range `previous_tag..HEAD` is referenced by the changelog release section or listed in `tests/coverage/changelog_no_user_facing_change.json`. |
| `scripts/check_package_manifest.py` | A staged package directory contains everything `.icc/package-manifest.yaml` declares. |
| `scripts/verify_release_package.py` | A staged package is self-contained and runs through the cold and warm `-r` cache paths. |
| `scripts/verify_site_release.py` | The website matches the release matrix. |
| `scripts/check_disclosure.py` | No commit message, pull-request text or added line discloses a private infrastructure identifier. |
| `scripts/check_test_coverage.py` | The documented test inventory matches the complete-suite runner. |
| `scripts/check_public_api_docs.py`, `scripts/gen_api_docs.py --check` | Public API documentation is complete and the generated pages are current. |
| `scripts/check_doc_claims_residual.py` | Every wrong typed documentation claim is allowlisted with a reason or tracked as open documentation debt. |
| `scripts/check_evidence_staleness.py --require-trace-dir` | No high-severity criterion's newest evidence is older than the configured window, and an empty trace root is not graded. |
| `scripts/check_wasm_imports.py` | Both browser glue files provide every `env` import the WebAssembly build asks for, and the generated flat-AD block is current. |

The changelog gate derives the pull-request numbers from commit subjects, with
no network access, and fails closed: it needs the full history and the previous
release tag, so a shallow checkout, an unresolvable tag or a range that yields
no pull request is reported as `NO_DATA` rather than passing over an empty
list. Its ledger accepts only a closed set of classes (`build-internal`, `ci`,
`docs-only`, `merge-integration`, `release-machinery`,
`reverted-or-superseded`, `test-only`), each with a specific reason.

### Site checks

`scripts/verify_site_release.py` reads the expected asset array straight from
the Release workflow and requires the site source to list every package in it,
to mark the one unsupported matrix cell as unsupported, and to describe the
payload as fifteen packages plus the checksum file. It also checks the
generated announcement page against its Markdown source and inspects the
committed `site/static/eshkol-site.wasm`: the module must export a
zero-argument `scheme_main`, the size statistics stated in `site/src/main.esk`
must match the committed WebAssembly artifacts, and the release strings must be
present in the compiled module.

The Pages deployment publishes the committed site artifacts; it does not
rebuild them. Regenerate `site/static/content/*.html` with
`scripts/build-site-content.sh` whenever a mirrored Markdown source changes,
and rebuild the site module with `scripts/build-site.sh` only when
`site/src/main.esk` changes.

## The strict dry run

Before tagging, dispatch the Release workflow on the candidate commit with
`candidate_tag` set to the intended tag and `strict_readiness=true`.

A strict dry run exercises what pull-request CI does not:

- **Packaging.** Every staging step, on every platform, including the
  dependency and licence staging that exists only in the asset jobs.
- **Manifests.** The package manifest check and the package verifier against
  the real staged directory, and the asset-set validation with checksums.
- **Platform links.** The ahead-of-time links and cold-cache `-r` runs inside
  each package on its own platform, which is where a missing platform symbol
  or archive ordering problem appears.
- **The full evidence recipe** at the exact commit, ending in a bound
  `ready`/100 receipt.

It publishes nothing: `publish-release` uploads the validated set as an
artifact and stops, and the tap is not touched.

## What the maintainer does by hand

The tag. Everything else is either a gate or a generated artifact. The
checklist, in order:

1. **Record the totals.** Put the full-suite CTest total and the VM parity
   total from the run at the release commit into
   `tests/coverage/release_record.json`, and confirm `tag`, `previous_tag`,
   `release_date` and `status`.
2. **Sync the documents.**
   `python3 scripts/check_surface_counts.py --sync`
3. **Regenerate the site content.** `scripts/build-site-content.sh`
4. **Run the gates.**

   ```
   python3 scripts/check_surface_counts.py --no-trace --require-complete
   python3 scripts/check_changelog_completeness.py --no-trace
   python3 scripts/verify_site_release.py
   python3 scripts/check_test_coverage.py
   python3 scripts/gen_api_docs.py --check
   python3 scripts/check_disclosure.py --base origin/master --head HEAD
   python3 scripts/release_readiness_guard.py notes --notes RELEASE_NOTES.md \
       --tag <tag> --output <scratch>/release-notes.md
   ```

   The last command succeeds without `--allow-pending` only when the release
   notes carry no pending-evidence marker. Then dispatch the strict dry run on
   the merged commit and wait for the bound receipt.
5. **Tag.** Create the annotated tag on the commit the receipt names and push
   it. The tag push runs the same workflow with the gate blocking, publishes
   the release, and bumps the tap.

If the workflow fails after the tag is pushed, fix forward. Tags are never
moved or replaced, and an existing release is never overwritten.

### The release controller

`scripts/release_autopilot.py` is an optional, restartable controller for the
same sequence. Each invocation performs one bounded step and then waits: it
previews by default and acts only with `--execute`, inside an explicitly
configured publication window. It requires every required branch check to be
green, refuses a receipt that does not match the exact commit, run id and run
attempt, refuses an incomplete asset set, verifies the tap formula against the
published source archive checksum, and honours a `PAUSE` file in its state
directory. It performs no administrator merges and no force pushes, weakens no
check, and never moves or overwrites a tag.

## See also

- [CONTRIBUTING.md](../../CONTRIBUTING.md), "Release-blocking readiness"
- [CI_LANES.md](CI_LANES.md) and [SELF_HOSTED_RUNNERS.md](SELF_HOSTED_RUNNERS.md)
- [COMPILER_ASSURANCE.md](COMPILER_ASSURANCE.md)
- [TESTING.md](../TESTING.md) and [TEST_COVERAGE.md](../TEST_COVERAGE.md)
- [TROUBLESHOOTING.md](../TROUBLESHOOTING.md), "Release and gate failures met locally"
