# Eshkol v1.3.3-evolve Release-Candidate Readiness Report

**Candidate branch:** `master` plus the release-finalization change set
**Release line:** `v1.3.x-evolve`
**Verified hardening baseline:** `502b5a17c180a5e0281be2b339049a783b3e4066`
**Baseline GitHub Actions:** run `29305913676`, **15/15 jobs green**
**Candidate head:** the final merged master commit containing this report
**Published base tag:** `v1.3.2-evolve` (`8443ddae`)
**Candidate date:** 2026-07-14
**Tag status:** **untagged** — publishing any tag remains a maintainer action.

This report is the release contract for the untagged v1.3.3-evolve candidate.
It supersedes the July 8 readiness snapshot and records gates executed from an
isolated Release build (`build-hardening`, tests enabled, XLA/GPU disabled with
their CPU fallbacks still exercised), the final hosted matrix, and dedicated
mesh evidence. A public tag must not be created until this finalization change
is merged, the exact master matrix is green, and the non-publishing release
workflow dry run succeeds.

## Verdict

The candidate is release-ready. All deterministic code, differential,
coverage, architecture, freestanding, WebAssembly, full-book, hosted-matrix,
and cross-platform mesh gates are green. The verified hardening baseline's
required GitHub Actions matrix passed 15/15; the finalization commit must repeat
that exact-master check and complete the manual non-publishing release matrix.
No release tag is authorized or created by this campaign.

## Release Gates

| Gate | Result |
|---|---|
| Aggregate test harness | **44/44 suites, 716/716 tests**, zero failed/skipped suites |
| CTest | **71/71** |
| SICP full-book | **88/88** JIT+AOT probes, zero xfail/XPASS |
| Chibi R7RS reference differential | **34/34 AGREE**, zero divergence/error |
| Generative multi-oracle differential | **127 programs**, Chibi/JIT/AOT-O0/AOT-O2/VM, zero divergence |
| Native/VM parity | **68/68** source+ESKB probes; unsupported paths fail explicitly |
| VM extended hosted surface | **53/53** |
| Executable language surface | **1056/1056 (100%)**, zero uncovered/high-risk rows |
| Taylor monomorphization equivalence | **441/441 JIT + 441/441 AOT**, bit-exact through order 8 |
| Poincare manifold invariant | **17/17** analytic checks |
| Automatic differentiation | **54/54** aggregate AD suite plus adversarial finite-difference gates |
| Memory/ownership | **16/16** aggregate memory suite plus C ABI barrier checks |
| WebAssembly import contract | **101/101 unique imports** provided in both JS runtimes |
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

The campaign locked the Poincare coordinate-tangent metric convention; exact
64-bit time values; proper-list directory walking; hosted port lifecycle/rebinding;
image-buffer ownership; rational/bignum region evacuation; parameter write
barriers; and O2 library-export retention. The large-list sort was replaced by
a stable memory-bounded vector merge sort, reducing the two-million-element
stress case from roughly 32 GB to about 362 MiB peak RSS.

Promise forcing now also has a unified native/VM nonlocal-control contract.
Exceptions and continuation invocations roll back an intrusive O(1)-auxiliary
evaluation chain before control transfer, so a thunk that does not return
normally remains retryable while successful `delay-force` chains retain
iterative path compression. The release corpus covers both handled raises and
continuation escape/retry on the hosted LLVM and serialized VM paths.

### Independent manifold audit reconciliation

[PR #275](https://github.com/tsotchke/eshkol/pull/275) contributed a
good-faith independent Poincare/AD audit and reproducible C cross-validation.
Its final exp-map patch removes the base conformal factor because it interprets
the input as a normalized tangent whose geodesic length is `||v||`. Eshkol's
public API instead uses coordinate tangent vectors under
`g_x = lambda_x^2 I`; therefore the invariant is
`distance(x, exp_x(v)) = lambda_x ||v||`, and the conformal factor belongs in
the standard coordinate-tangent exponential map. The matching log map and the
17 analytic tests lock that convention, including off-origin round trips and
lengths. The audit is retained as provenance and helped make this convention
explicit; its unrelated header/backend decompositions require focused PRs and
the full LLVM/VM/Windows matrix rather than merging the conflicting omnibus
branch.

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
3. Run the complete release workflow manually against that head; it must build,
   test, package, validate, and checksum all 16 assets while publishing nothing.
4. Only the maintainer may authorize and create `v1.3.3-evolve` (or another
   version) after reviewing these notes and artifacts.
5. After the tag workflow succeeds, smoke representative downloaded assets on
   macOS ARM64, Linux x64/ARM64, and Windows x64 and confirm the Homebrew tap
   advanced to the exact tag and source-archive checksum.
