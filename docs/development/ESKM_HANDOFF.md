# ESKM subsystem status and week 12 handoff

Snapshot: 2026-09-08 UTC. Owner: [Gabriel-Kahen](https://github.com/Gabriel-Kahen).
This consolidates the personal serialization assignment from
[roadmap PR #587](https://github.com/tsotchke/eshkol/pull/587), whose source is
[`GABE_KAHAN_ROADMAP.md` at `10301403f6ae434321108751bc46a0c3bad7ca62`](https://github.com/tsotchke/eshkol/blob/10301403f6ae434321108751bc46a0c3bad7ca62/docs/development/GABE_KAHAN_ROADMAP.md).
It does not replace that roadmap, the project roadmap, or
[CONTRIBUTING.md](../../CONTRIBUTING.md).

**The assignment is not accepted or complete.** The six packets below are
pending review or implementation. Open PRs are labelled **review**, never
done. Week 12 is the roadmap milestone being documented, not a claim that
twelve weeks of work or the planned hours have elapsed. The format-v2 decision
is **Proposed**; implementation awaits review of the byte-level decision.

## Packet and dependency register

PR states and evidence are point-in-time observations. Recheck the remote head
and review disposition before using any result for acceptance. PR-reported
results below are attributed reports, not reruns by this handoff task.

| Packet | Status | PR / scope | Remaining acceptance |
|---|---|---|---|
| GK-SER-01 | review | [#596](https://github.com/tsotchke/eshkol/pull/596): v1 specification and golden corpus | Review; all-fixture engine agreement remains limited by scalar/empty VM materialization |
| GK-SER-02 | review | [#602](https://github.com/tsotchke/eshkol/pull/602): VM preflight and checked cursor bounds | Depends on [#555](https://github.com/tsotchke/eshkol/pull/555); inherited prelude-cache failure and integrated verification remain |
| GK-SER-03 | review | [#597](https://github.com/tsotchke/eshkol/pull/597): four-engine model-I/O matrix | Depends on #596; scalar/empty remains open; public tensor matrix is separate review PR #615 |
| GK-SER-04 | review | [#600](https://github.com/tsotchke/eshkol/pull/600): atomic checkpoint replacement | Required CI unverified (GitHub merge state BLOCKED); native macOS/Windows evidence and durability decision remain |
| GK-SER-05 | review | [#613](https://github.com/tsotchke/eshkol/pull/613): Proposed v2 decision | Byte-level design/cap review, then implementation and executable compatibility evidence |
| GK-SER-06 | review | [#601](https://github.com/tsotchke/eshkol/pull/601): deterministic bounded native campaign | Depends on #555; archived sanitizer evidence has a different address-space policy; remaining checks and integrated acceptance incomplete |

The shared prerequisite #555 is OPEN at
`e737c10a28c34099d502b0b788dba09c89d06875`; neither #602 nor #601 is an
independent replacement for it. #596 and #597 are an ordered documentation/test
stack. #600 and #602 touch the same native/VM I/O files, so passing separately
does not establish their combined behavior. This handoff is documentation and
a valid-input measurement harness; it requires no sibling implementation to
merge, but its status links depend on those PRs.

The roadmap normally allows one milestone implementation PR at a time. The
current user-directed parallel PR arrangement exceeds that default. Consolidate
review and dependency order before starting another implementation milestone.

A separate [scalar/empty VM I/O adapter proposal, #614](https://github.com/tsotchke/eshkol/pull/614),
at `fc2ae7be9f9ca376fde6e402e4adc22d9c916f60` is **review / Proposed**. It
addresses the existing v1 representation gap, independently of the v2 decision.
Implementation requires maintainer disposition and coordinated preflight and
materialization changes after #555/#602; the target is all six valid fixtures
and all 16 producer/consumer combinations per fixture, plus VM single-tensor
routes and bounded positive lifetime checks. It supplies no runtime acceptance.

The independent [ESKT tensor-file matrix, #615](https://github.com/tsotchke/eshkol/pull/615),
at `44b366442f1d60d39dfdb63b2033e7ebbbb0672d` is **review**. It adds bounded
positive public tensor-I/O coverage without depending on #596/#597/#600. Its
local 16-pair result closes the missing positive cross-reader test coverage,
not packet acceptance. Required CI, unsupported shapes and platform coverage
remain separate. ESKT uses host-endian bytes and implicit binary64 payloads,
with no stored dtype or element-count field; it is not the ESKM wire contract.

### Merge compatibility checkpoint

The handoff navigation now has its own section in the development index,
preserving #600's separate atomic-save link. Textual merge checks pass against
master `5ce74beeac25aca56c0c8129083e6db77f96f7dc` and the inspected #596,
#597, #600, #613, #614 and #615 heads below. #597 and #615 relocated their
CMake registration blocks independently; each reports the same focused CTests
passing after reconfiguration, reusing its unchanged runtime build.

#555, #601 and #602 still conflict in generated `docs/api/INDEX.md` and
`docs/api/README.md`. Those conflicts are also present when merging their heads
with the master baseline; this handoff changes neither file. They are inherited
prerequisite integration blockers, not solved merges. `git merge-tree
--write-tree <handoff-head> <target-head>` checks textual compatibility without
changing any target branch; a clean result does not establish combined runtime,
CI or maintainer acceptance.

### Merged foundation versus pending work

The roadmap records [#19](https://github.com/tsotchke/eshkol/pull/19) as the
merged model/tensor serialization foundation, and
[#18](https://github.com/tsotchke/eshkol/pull/18) and
[#28](https://github.com/tsotchke/eshkol/pull/28) as merged REPL and server work.
Those are historical foundations, not acceptance of GK-SER-01 through -06.
No serialization packet PR listed above is recorded as merged in this snapshot.

## Compatibility matrix

ESKM's format version `1` is distinct from Eshkol's `v1.2.4` release tag.
The #596 fixture provenance is peeled release commit
`b98dc8b32399de739a037e9fa0a470bf0426eca9` (tag object
`4e07f166a7a0da28d24c78fb1c1af4258c4c1845`). At #596 head
`5ad058b2bc416c97a76c21f246d6b3bb4044b48f`, six accepted and six rejected
files total 9,160 bytes; the largest is the 8,237-byte `large-32x32.eskm`
(1,024 binary64 values). All eleven earlier fixture files are unchanged. These
limits describe the corpus, not the production format or a production resource
ceiling.

| Input / operation | Native JIT | Native AOT | VM source | VM bytecode | Evidence and boundary |
|---|---|---|---|---|---|
| Ordinary, rank-8, named multi-tensor and 32×32 ESKM v1 model load/rewrite | Reported PASS | Reported PASS | Reported PASS | Reported PASS | #597: exact bytes, names/order, dimensions and payload |
| Four producers × four consumers, common two-tensor model | Reported PASS | Reported PASS | Reported PASS | Reported PASS | #597: all 16 cells, four producer golden comparisons |
| Six malformed golden files | Reported refusal | Reported refusal | Reported refusal | Reported refusal | #597; bounded corpus only, not universal malformed-input proof |
| Scalar / zero-extent ESKM | Native core supported; JIT not in matrix | Native core supported; AOT not in matrix | Materialization limitation | Materialization limitation | #596 specification; #597 explicitly excludes these shapes |
| Public ESKT tensor-save/load, three valid shapes | Reported PASS | Reported PASS | Reported PASS | Reported PASS | #615: 16 pairs × 3 tensors, 12 producer files, 48 exact rewrites; host-endian f64 contract. Separate from #597 ESKM models and ESKM single-record helpers |
| ESKM v2 | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | Proposed decision only; no new-reader compatibility result |

The exact v1 layout and limits live in the
[#596 specification](https://github.com/tsotchke/eshkol/blob/5ad058b2bc416c97a76c21f246d6b3bb4044b48f/docs/reference/tensors/eskm-v1.md),
not this summary. V1 has little-endian fields, binary64 payload bits, and CRC32
over all bytes before the footer. Preserve duplicate names and record order;
do not silently reinterpret them as a map. Writers emit zero reserved flags;
historical readers ignored nonzero flags. Consequently, assigning a mandatory
feature to those ignored bits cannot ensure old-reader refusal. Unknown
versions must refuse; proposed v2 behavior is not a current runtime guarantee.

The [v2 proposal at `2d85dc648d3088ebb75dcfd35f076c9efd7daedc`](https://github.com/tsotchke/eshkol/blob/2d85dc648d3088ebb75dcfd35f076c9efd7daedc/docs/design/ESKM_V2_FORMAT_DECISION.md)
uses a 24-byte header, bounded optional TLVs, advisory annotations and a
separate required-feature bitmap. Its proposed caps and backend budget tables
need review. It deliberately pins the earlier #596 contract/corpus head
`edb614e57904229fc626b23801c9bffbbb248b25`; its five-positive-fixture check is
not evidence for the later sixth fixture. Implementation slices 05a–05e cover
parser/reference/budgets, reader integration, separately reviewed opt-in writer,
four-engine compatibility, and final evidence/status closure. Default writers
remain v1. No v2 runtime behavior is provided by #613.

## Evidence ledger

### PR-reported evidence at inspected heads

The source of each row is the linked PR description at the stated head, as
read on the snapshot date. This table does not assert that the current GitHub
CI checks are green or that maintainers accepted the results.

| PR and inspected head SHA | Reported PASS | Reported FAIL / NOT RUN / qualification |
|---|---|---|
| #596 `5ad058b2bc416c97a76c21f246d6b3bb4044b48f` | Checker: 6 accepted / 6 rejected and all 12 hashes; checksum negative control; historical writer: 6/6 byte-identical; exact-head native test/codec recompile using matching cached runtime support | Three stale generated API pages reproduced on base `ea81a854`; eight inherited surface-count mismatches; full build/battery, four-engine matrix, macOS/Windows/big-endian NOT RUN here |
| #602 `e588e1f10c2c4b02d85efd26f943b83e4c05f861` | Focused CTest 2/2 normal and ASan+UBSan (leak detection); duplicate-name controls; JIT/AOT/hosted-VM-bytecode payload parity; scalar/empty preflight negative controls fail on prior head and pass after fix | Incremental private builds; inherited `vm_prelude_cache_is_current` FAIL reproduced with #555 base; no matching `vm_canonical_stdlib` test; full suite and platform/sanitized engine matrix NOT RUN |
| #597 `fc22ceb200de5919c0476a76d577878a47ad74c6` | Initial fresh private LLVM 21.1.8 runtime build; latest registration-only follow-up reconfigured and CTest 5/5; 16/16 model pairs, 4/4 producer golden bytes, 12/12 negative controls; all 1,024 large-fixture values; wrong-callback regression control catches both native engines; oracle/schema checks | Inherited API-doc and surface-count FAIL on dependency stack; external ICC, full smoke/release pipeline, full suite, sanitizers, macOS/Windows NOT RUN; scalar/empty excluded |
| #600 `83467dde5fc6ac427436d457d821af094238a93b` | Fresh private Clang 22 / LLVM 21.1.8 build; focused CTest 7/7 and four-engine atomic matrix; capability policy 50/50; helper ASan+UBSan/fallback; Emscripten helper and checked-in bundle Node checks | No source change; required GitHub CI unverified and merge state BLOCKED; full battery, native macOS/Windows and fresh full WASM regeneration NOT RUN |
| #613 `2d85dc648d3088ebb75dcfd35f076c9efd7daedc` | API/coverage checks; actual 28-byte example CRC/header; pinned earlier #596 corpus 5 accepted / 6 rejected and negative control; original user files preserved | Documentation only: v2 runtime, sanitizer, all-engine gates NOT RUN; design and admission caps Proposed |
| #614 `fc2ae7be9f9ca376fde6e402e4adc22d9c916f60` | Source/fixture inspection, API check, 46-suite inventory, links and independent read-only design review | Proposed only; runtime, future all-fixture matrix, lifetime controls and platform acceptance NOT RUN |
| #615 `44b366442f1d60d39dfdb63b2033e7ebbbb0672d` | Initial fresh private Clang 22.1.6 / LLVM 21.1.8 runtime build; latest registration-only follow-up reconfigured and CTest 4/4; ESKT 16/16 pairs, 12 exact producers, 48 exact rewrites and 48/48 wrong expected-byte controls; API/coverage checks | Full suite, macOS/Windows, cross-endian interoperability and required GitHub CI NOT RUN; native Windows registration excluded because its current ESKT I/O is disabled |
| #601 `c148e32243803a231ac2fb22700e19b3311516a9` | Earlier PR report: CTest 2/2, canonical 70-case smoke and oracle/schema checks. Current archived diagnostic runs are detailed below | Historical generated/oracle checks were not all rerun; external `icc`, full battery, macOS/Windows, TSan/MSan/standalone LSan NOT RUN; remaining verification stopped by environment restriction |

For #601 the documented ordinary probe limits are a two-second per-input
timeout, 256 MB POSIX address-space ceiling and 8 MB retained-artifact ceiling.
These are harness limits, not a promise that every production loader allocation
is bounded identically. Native-only campaign evidence does not establish VM
resource behavior. Use the existing bounded regression selection; this handoff
adds no malformed inputs or expanded campaign.

### Archived GK-SER-06 local evidence

The coordinator supplied completed logs for unchanged #601 head
`c148e32243803a231ac2fb22700e19b3311516a9`. This handoff inspected those logs
read-only; it did not execute or resume that campaign. Further execution was
stopped by the task environment's cybersecurity restriction, so no later
verification is inferred. These are archived local results, **not** an accepted
canonical release event or a maintainer verdict.

| Archived run | Seed | Inputs | Failures | Elapsed | Log SHA-256 |
|---|---:|---:|---:|---:|---|
| Native smoke | 1592614637 | 70 | 0 | 0.39 s | `85bbd82792ca9e46af85fde94f5739db2a722e5fa3d93c917cd8c3fc163472d8` |
| Native full | 1592614637 | 700 | 0 | 3.84 s | `ad80d60859e192cc30a81df22aa7ecaa4cc5b19991d5ac8f7dad2c5663494582` |
| ASan smoke | 1592614637 | 70 | 0 | 1.41 s | `5563096e4bd2584263a7ee0a561c37cdded33931974eab08d6443ed310521eb7` |
| ASan full | 1592614637 | 700 | 0 | 14.56 s | `f189a02109c72bcebaf05b33a4465f053469a98fa0130f627d0b25cf5179781c` |

Files are `native-smoke-fresh.log`, `native-full-fresh.log`,
`asan-smoke-fresh.log`, and `asan-full-fresh.log` in the coordinator's local
`/tmp/eskm-601-verification-b29a/` archive, which is not a durable public artifact.
The accompanying native trace identifies the source SHA above and probe SHA-256
`4282bcf1d74f3676e759cb9a2b39e586421bd9de36522561ac80b65a177c37b2`.
Each log records 13 passing harness controls and zero retained artifact bytes.
Native and ASan `model_io_test` logs end in PASS; archived API and coverage
checks also pass. The coordinator reports a fresh source build for these runs.
Native runs used the ordinary address-space cap; ASan removed that cap because
of shadow mapping, retaining the reported 32 MiB arena and timeout policy.
Do not treat that ASan configuration as an identically capped run. Remaining
oracle/generated checks and canonical acceptance are not established here.

### Runs performed by this handoff task

Source baseline: `5ce74beeac25aca56c0c8129083e6db77f96f7dc`, upstream master
observed on 2026-09-08. The PR adds documentation/measurement files on that
base and changes no reader, writer, on-disk byte, public API or resource policy.

- **PASS:** `python3 scripts/gen_api_docs.py --check` (up to date).
- **PASS:** `python3 scripts/check_test_coverage.py` (46 suites).
- **PASS:** fresh compilation and two runs of the valid-input codec benchmark
  below; all 108 write/parse pairs preserved metadata and every payload bit
  (90 measured pairs plus 18 warmups).
- **PASS:** local relative links and raw CSV sample counts/sizes; benchmark
  refuses a missing directory argument or existing output (exit 2, existing
  bytes preserved). A private copy with one deliberately wrong expected payload
  bit exits 1 before emitting any measured row; production inputs remain valid.
- **NOT RUN:** sibling runtime matrices, `model_io_test`, CTest VM suites,
  sanitizer campaigns, external ICC acceptance and the full repository battery
  in this documentation task. Historical results remain separate above.

## Measured native checkpoint baseline

The checked-in [harness](evidence/eskm_checkpoint_baseline.cpp) directly includes
the unchanged native codec translation unit. Section garbage collection omits
unused public runtime entry points, avoiding a dependency on uncertain cached
builds. This is a fresh native codec measurement, **not** an LLVM/JIT/AOT/VM
end-to-end or public API benchmark.

Environment: Linux `7.1.2-3-cachyos`, x86-64 AMD Ryzen 7 3700X, GCC
`16.1.1 20260625`, `-O2 -DNDEBUG`, btrfs on `/dev/nvme0n1p6` mounted at `/home`.
The machine was shared with other tasks; CPU frequency and cache state were
not pinned. No shared build was modified. LLVM 21 caches were found but were
not needed for this isolated codec measurement.

Native `lib/core/model_io.cpp` SHA-256:
`c6e7463e0394969c24b5eddedd5043b050ee5b0fa4e1bf5cc50768777a3ea335`.
Measured executable SHA-256:
`4633d7ba7035fda54405890e971bfaeb197ed6473d268ff4e938cc2997628948`.

Each size uses one rank-1 binary64 tensor named `weights`. Each independent run
discards three warmups then records 15 consecutive samples, alternating write
and immediate parse of the same path. Write time includes encoding, CRC, stdio
write and close; parse time includes file read, CRC, metadata decoding and
payload-vector allocation. Validation, input construction and parsed-vector
destruction are outside the timed intervals. The kernel page cache is warm;
there is no `fsync`, cache eviction, arena materialization or model training.

| Payload | File bytes | Run 1 write median, µs (min–max) | Run 1 parse median, µs (min–max) | Run 2 write median, µs | Run 2 parse median, µs |
|---|---:|---:|---:|---:|---:|
| 4 KiB | 4,140 | 87.657 (86.565–132.842) | 37.651 (37.562–38.002) | 94.800 | 37.892 |
| 256 KiB | 262,188 | 2,588.310 (2,554.800–2,661.760) | 2,264.200 (2,220.490–2,522.410) | 2,632.890 | 2,277.640 |
| 1 MiB | 1,048,620 | 10,077.100 (9,942.180–10,185.700) | 9,171.970 (9,040.510–9,372.070) | 10,035.900 | 9,164.060 |

Raw samples: [run 1](evidence/eskm-baseline-run1.csv),
[run 2](evidence/eskm-baseline-run2.csv). These are local baseline numbers, not
an optimization speedup or a performance gate. No before/after overhead for
#600 or #602, RSS peak, cold-disk throughput, tail-latency confidence interval
or cross-platform performance is claimed.

Reproduce from the repository root using GCC and GNU-compatible section GC:

```bash
set -euo pipefail
# Check the source SHA above before comparing results.
sha256sum lib/core/model_io.cpp
bench_dir=$(mktemp -d "$PWD/.eskm-baseline-XXXXXX")
trap 'rm -rf "$bench_dir"' EXIT
g++ -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -I inc \
  docs/development/evidence/eskm_checkpoint_baseline.cpp \
  -Wl,--gc-sections -o "$bench_dir/benchmark"
mkdir "$bench_dir/data"
"$bench_dir/benchmark" "$bench_dir/data" > "$bench_dir/run1.csv"
"$bench_dir/benchmark" "$bench_dir/data" > "$bench_dir/run2.csv"
cat "$bench_dir/run1.csv" "$bench_dir/run2.csv"
```

The benchmark refuses an existing `baseline.eskm` at startup and uses only
valid, modest inputs. Its exact-source inclusion is deliberate: after codec
changes, recompile and record the new SHA; do not relabel these results.

## Open risks and unmet roadmap acceptance

1. **Review/integration:** #555 and every packet PR remain open, with inherited
   generated-document conflicts on the #555/#601/#602 stack. Independent
   passing reports do not show that a combined master build passes. Record
   integrated SHAs and rerun relevant existing gates after dependencies land.
2. **Compatibility coverage:** common-model 16-cell evidence leaves the
   roadmap's all-fixture scalar/empty requirement unmet. The added 32×32 fixture
   closes the missing larger-file representation, not this shape gap. Public
   tensor ESKT and ESKM helper routes must stay distinct. #615 supplies the
   separate bounded positive ESKT matrix, but it is open, host-endian-only and
   not native-Windows acceptance. Platform and resource limits remain
   explicit exceptions rather than silent reinterpretations.
3. **Admission guarantees:** #602 addresses allocation before VM validation;
   it is pending on #555. Its scalar/empty change moves existing VM refusal
   before allocation and does not repair native/VM acceptance parity. Its
   inherited prelude-cache check fails identically on the prerequisite base.
   Neither this handoff nor a finite corpus establishes
   universal safety or combined native/VM resource bounds.
4. **Persistence:** #600 offers process-level atomic replacement, not power-loss
   durability. Abrupt termination can leave temporary files; ownership/ACLs and
   extended attributes are not all preserved. Native Windows/macOS execution
   remains unmeasured in the inspected report, and required GitHub CI acceptance
   is unverified despite the fresh local checks.
5. **Format evolution:** a Proposed v2 document does not satisfy weeks 9–10's
   implementation gate. Review the byte decision before changing bytes; retain
   old fixtures and add unknown-mandatory-feature refusal evidence afterward.
6. **Resource/sanitizer evidence:** retain seed, input/time/memory budget, exact
   executable/source hashes and observed outcomes for the accepted campaign.
   A sanitizer run with a different memory policy must be labelled separately.
7. **Week 12 acceptance:** status, risks, baseline numbers and backlog are now
   recorded, but maintainer acceptance, merged packet results, integrated matrix
   evidence and the next approved implementation milestone remain outstanding.

## Prioritized next six months

Sequence around the roadmap's approximately 10-hour week and review latency;
these are proposed priorities, not delivery promises or release assignments.

| Priority / window | Bounded milestone | Dependency | Exit evidence |
|---|---|---|---|
| P0 / month 1 | Resolve review/dependency order and land one serialization implementation slice at a time | Maintainer disposition of #555 and six packet PRs | Merged SHA register plus focused gates on integrated head; no status inferred from filing |
| P1 / month 2 | Review #614 and integrate #615; close scalar/empty compatibility gaps with the approved backend contract | #596/#597, #614/#615 and loader review | Each supported shape/API has an exact producer/consumer result; unsupported combinations have explicit refusal/limit evidence |
| P1 / month 3 | Implement the smallest approved v2 decision | GK-SER-05 byte-level approval | Existing v1 fixtures unchanged; reviewed v2 fixtures; old-reader and unknown-mandatory-feature refusal matrix |
| P2 / month 4 | Validate atomic replacement on native macOS/Windows; decide whether stronger durability is needed | #600 review and platform availability | Platform test results and a separate reviewed durability contract before any stronger guarantee |
| P2 / month 5 | Refresh bounded sanitizer/resource evidence across supported loaders | Integrated loader implementation | Reproducible seed/budget/SHA logs, ordinary regression for each finding, stated peak memory and explicit engine gaps |
| P2 / month 6 | Establish public API and larger representative checkpoint performance baselines | Stable accepted implementation | Warm/cold methodology, codec vs materialization cost, memory peaks and reproducible before/after numbers before optimizing |

The next action is the P0 review/integration checkpoint. Continue serialization
ownership or change tracks only through the roadmap's maintainer review; do not
expand into release operations, infrastructure, GPU backends or unrelated AD
compiler work as part of this handoff.
