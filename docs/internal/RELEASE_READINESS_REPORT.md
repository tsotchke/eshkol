# Eshkol v1.3.3-evolve Release-Candidate Readiness Report

**Candidate branch:** `feat/complete-remaining-hardening`
**Release line:** `v1.3.x-evolve`
**Candidate head:** the commit containing this report
**Published base tag:** `v1.3.2-evolve` (`8443ddae`)
**Candidate date:** 2026-07-13
**Tag status:** **untagged** — publishing any tag remains a maintainer action.

This report is the release contract for the untagged v1.3.3-evolve candidate.
It supersedes the July 8 readiness snapshot and records gates executed from an
isolated Release build (`build-hardening`, tests enabled, XLA/GPU disabled with
their CPU fallbacks still exercised). A public tag must not be created until
this candidate is merged and the GitHub Actions matrix for the exact master
head is green.

## Verdict

The candidate is locally release-ready. All deterministic code, differential,
coverage, architecture, freestanding, WebAssembly, and full-book gates are
green. The remaining publication condition is the required GitHub Actions
matrix on the final merged master head. No release tag is authorized or
created by this campaign.

## Release Gates

| Gate | Result |
|---|---|
| Aggregate test harness | **44/44 suites, 714/714 tests**, zero failed/skipped suites |
| CTest | **69/69** |
| SICP full-book | **88/88** JIT+AOT probes, zero xfail/XPASS |
| Chibi R7RS reference differential | **34/34 AGREE**, zero divergence/error |
| Generative multi-oracle differential | **31 programs**, Chibi/JIT/AOT-O0/AOT-O2/VM, zero divergence |
| Native/VM parity | **68/68** source+ESKB probes; unsupported paths fail explicitly |
| VM extended hosted surface | **52/52** |
| Executable language surface | **1056/1056 (100%)**, zero uncovered/high-risk rows |
| Taylor monomorphization equivalence | **441/441 JIT + 441/441 AOT**, bit-exact through order 8 |
| Poincare manifold invariant | **17/17** analytic checks |
| Automatic differentiation | **54/54** aggregate AD suite plus adversarial finite-difference gates |
| Memory/ownership | **16/16** aggregate memory suite plus C ABI barrier checks |
| WebAssembly import contract | **100/100 unique imports** provided in both JS runtimes |
| ICC architecture model | **8/8 PASS**, zero FAIL/UNCHECKABLE |
| ICC release readiness | **100/100**, oracle complete, zero contract/fallback gaps |
| Working-tree hygiene | Only generated, gitignored ICC runtime evidence is untracked |

## What This Candidate Delivers

### Complete quantum trajectory

S1-S5 is complete: Moonlab-backed gates and Bell-verified QRNG; VQE with
machine-precision H2 energy; composable differentiation through circuits via
`AD_NODE_CUSTOM`; FIPS 203 ML-KEM 512/768/1024; and a permanent hosted quantum
CI/CHSH/coverage/architecture/oracle gate. Quantum remains opt-in and the
default build remains dependency-light.

### Complete executable surface evidence

The language manifest is no longer aspirational. Every declared row must be
executed by a deterministic test and runtime trace. Token sightings, parser
recognition, unreachable branches, and dead code earn no credit. The resulting
ratchet is 1056/1056 and cannot silently regress below 100%.

### Honest backend parity

The VM does not claim parity by returning placeholder values. Supported paths
are cross-executed against native JIT and ESKB behavior; unsupported paths must
emit a clean explicit error. The campaign completed multiple values, vector
constructors, closure mutation, parameter objects, includes, hosted system and
image operations, polling, and process environment propagation.

### Correctness and memory safety

The campaign corrected the Poincare tangent metric convention; exact 64-bit
time values; proper-list directory walking; hosted port lifecycle/rebinding;
image-buffer ownership; rational/bignum region evacuation; parameter write
barriers; and O2 library-export retention. The large-list sort was replaced by
a stable memory-bounded vector merge sort, reducing the two-million-element
stress case from roughly 32 GB to about 362 MiB peak RSS.

## Architecture Evidence

`.icc/architecture-model.yaml` is verified against both indexed source and
oracle-view runtime traces. The model checks eight invariants, including:

- Poincare exp/log and Riemannian tangent-length consistency.
- Native/VM dispatch parity backed by executable corpus traces.
- Complete executable language-surface coverage.
- Quantum randomness provenance through Moonlab rather than a bare PRNG when
  quantum support is enabled.
- WebAssembly runtime import/glue completeness.

The ICC index and long-memory artifact are rebuilt from the final candidate
before the pull request is opened.

## Boundary and Evidence Policy

Fixed limits are release-quality only when derived from a runtime/platform/data
contract, part of a named ABI, or a documented defensive cap with an explicit
error and test. A test budget may be fixed only when its purpose is explicit.
Magic truncation, fabricated VM values, skipped execution, and declaration-only
coverage are release blockers.

## Repository Hygiene

- Public release material remains in durable root/docs locations.
- Generated build products, ICC runtime traces, differential artifacts, and
  caches remain untracked.
- Existing user-owned untracked notes in the primary checkout are preserved.
- This campaign creates no release tag and does not move an existing tag.

## Publication Rule

1. Merge the candidate only after every required pull-request check is green.
2. Verify the GitHub Actions matrix for the exact merged master head is green.
3. Re-run the release smoke against that head if packaging inputs change.
4. Only the maintainer may authorize and create `v1.3.3-evolve` (or another
   version) after reviewing these notes and artifacts.
