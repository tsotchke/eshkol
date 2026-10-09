---
kind: project
status: current
owner-area: release
since: v1.3.6-evolve
sources:
  - ROADMAP.md
  - docs/COMPILER_ROADMAP.md
  - .icc/completion-oracles.yaml
---

# Release ladder: v1.3.6 through v1.5.0

This is the single reconciled near-term release ladder for the roadmap and
compiler roadmap. Statuses below are checked against the repository at
`34fb71417df7273aba11889091a2c2542c8eedff` (master, 2026-09-28). A planned
producer or oracle name is not evidence that its script, target, or receipt
exists. The v1.3.5 source was released internally on 2026-09-22 and published
on GitHub on 2026-09-27; the frozen release record keeps its internal release
date. Owners are left unassigned where this snapshot records no named lane.

> **Status at v1.3.6-evolve (2026-10-09).** The v1.3.6 row's scope was admitted
> and the release is PREPARED FOR PUBLICATION per the release record; its
> contents, each with its pull request, are in the
> [ROADMAP v1.3.6 section](../ROADMAP.md#v136-evolve--runtime-fixes---prepared-for-publication).
> The v1.4.0 through v1.5.0 rows still describe the 2026-09-28 snapshot.

| Release | Scope and dependency | Owner lane | Producer / trace | Acceptance and blocker | Snapshot status |
|---|---|---|---|---|---|
| **v1.3.6-evolve** (planned as conditional) | Depends on the frozen v1.3.5-evolve tag `74dc40d68`. Narrow post-tag scope only. #727 is a draft for interval edge cases. The cold compiler-embedding consumer shows that the package does not ship the embedding header and static archive. State the published package documentation/contract here; do not claim the package already supports embedding or alias the compiler archive as the runtime. The current [FindEshkol package contract](../cmake/FindEshkol.cmake#L45) and [FFI API guide](api/eshkol_ffi.md#L3) are the source docs to update. #721 allocation work is admitted only after independent correctness and performance gates; #722 remains experimental. A complete embedding SDK is a separate v1.4.1 deliverable. | Unassigned | #727 draft; #721 follow-up after independent gates; #722 experimental. Cold compiler-embedding consumer probe; durable ICC receipt path pending. | Accept #727 only with a reproduced interval edge case and regression evidence; #721 must pass independently defined correctness and measured performance acceptance. v1.3.6 package work is the precise documentation/contract update the cold consumer probe calls for; full SDK acceptance requires the headers, LLVM/platform link contract, static archive and cold consumer tests. | **PREPARED FOR PUBLICATION** (status at 2026-10-09, release record `tests/coverage/release_record.json`): #727 (98f7db6db), #721 and #722 (experimental, doubly gated) were admitted, and the package contract is stated in `cmake/FindEshkol.cmake` and [the FFI guide](api/eshkol_ffi.md) (#729); the embedding SDK remains v1.4.1. Planning snapshot below. Conditional; source archive SHA-256 `0862dafe7dffba97202519a9b20fdc294cf97babccc54b018d58f5e3ae187968`, built from tag commit `74dc40d68`. Generated-program integration and 31-entry manifest pass; minimal embedding consumer requires `eshkol/eshkol_ffi.h` and `libeshkol-static.a`, which are not packaged. Planning snapshot `34fb71417df7273aba11889091a2c2542c8eedff`. |
| **v1.4.0-connection** | Depends on shipped v1.3.5; conditional v1.3.6 fixes are not prerequisites. Scope: networking, concurrency, resource-sound profile, linear types, ADR-0017 SBLC first slice, and limited W6 PJRT/XLA interface spike. Native collectives over sockets remain in v1.4 scope pending owner R7 ruling. The PJRT spike proves one CPU-plugin round trip; it does not deliver accelerator execution. | Unassigned | `scripts/run_v14_connection_gate.sh` emits probe events but intentionally exits 0 for a running count; ICC readiness is the release gate. `scripts/run_xla_gate.sh` and `scripts/run_sblc_gate.sh` are absent. The literal v1.5 oracle action is `replace_with_v1_5_intelligence_oracle`. | Release acceptance requires ICC readiness, connection/resource deliverables, and SBLC first-slice criteria with bound producers/traces. A shell exit code alone does not certify v1.4. SBLC criteria are specified in ADR-0017 but its gate script/targets are missing. Native collectives scope awaits owner ruling. | Planned; exact source snapshot `34fb71417df7273aba11889091a2c2542c8eedff`. |
| **v1.4.1-ABI** | Depends on v1.4.0 and follows the frozen ABI v2 header/inventory substrate. Publishes after v1.4.0 and before v1.4.5. Remaining ADR-0012 migration Stages 3–6 and full embedding SDK belong here. | Unassigned | ADR-0012 migration producers and cold SDK consumer gate are unassigned. The package contract sources are [FindEshkol](../cmake/FindEshkol.cmake#L45) and [FFI guide](api/eshkol_ffi.md#L3). | Acceptance requires conversion of the remaining inventory sites, persisted-format ABI declaration, final layout flip, public header set, LLVM/platform link contract, packaged static archive, and passing cold embedding consumers. | Planned; prior target 2026-11-06 requires joint schedule review. Stages 0–2 are already complete. Exact source snapshot `34fb71417df7273aba11889091a2c2542c8eedff`. |
| **v1.4.5-accelerate** | Publication must follow v1.4.1 and precede v1.5.0; engineering may run in parallel. Depends on v1.4.0 PJRT interface contract and v1.4.1 ABI contract work. Scope: device runtime on accelerator silicon: PJRT, StableHLO device execution, device gradients/geometric ops, region formation, multi-device sharding, bounded bf16, and end-to-end model training/inference. | Unassigned | The named `scripts/run_xla_gate.sh` and `xla-tpu-ready` oracle stages in `ROADMAP.md` are not present at this snapshot. XLA stage scripts and corresponding build targets must be added before they can be treated as producers. | Each device stage needs an executable producer, trace, and acceptance result; final acceptance needs an end-to-end training and inference run on accelerator hardware. Publication is conditional on all device gates. If they are not ready, v1.5.0 slides; v1.4.5 does not move after it. Current source has XLA/StableHLO build scaffolding and an optional PJRT smoke target, but no release gate implementation or device receipts. | Planned, not shipped. The existing Q1 2027 target conflicts with the prior Dec 5, 2026 v1.5.0 target; both require joint rebaseline. Exact source snapshot `34fb71417df7273aba11889091a2c2542c8eedff`. |
| **v1.5.0-intelligence** | Depends on v1.4.5 publication and separately evidenced native mesh/data-parallel work. Scope: `core.dbsp` GA, native PGO, Noesis M2, neuro-symbolic features, and W6 native mesh bit-identity. This is distinct from the v1.4.5 accelerator runtime. | Unassigned | ICC `sicp-completeness` exists for SICP smoke coverage but is not a v1.5 intelligence oracle. The literal v1.5 oracle action is `replace_with_v1_5_intelligence_oracle`; it must be replaced and given bound producers before release. | Each claimed feature needs an executable gate. Release cannot proceed until v1.4.5 has published and the v1.5 oracle is implemented with criteria/traces. SBLC first-slice acceptance is v1.4.0; do not claim its producer/gate exists today. | Planned; release oracle incomplete. Current Dec 5, 2026 target is subject to joint rebaseline after v1.4.5 device gates. Exact source snapshot `34fb71417df7273aba11889091a2c2542c8eedff`. |

## Developed branch candidates

A read-only portfolio audit covered 1,714 refs, 1,083 unique heads, and 169
worktrees against this master snapshot. Squash ancestry is not enough to prove
content is new: candidate paths were compared with master and findings below
are admission leads, not accepted patches. Do not merge whole branches. The
indexed [v1.4 branch portfolio](V14_BRANCH_PORTFOLIO.md) records all 19
focused candidates, exact heads, release lanes, classifications, evidence
caveats, ICC alias status, and next gates. The [machine-readable portfolio](../.icc/v14-branch-portfolio.json) is the audit source. ICC aliases or tasks are missing or stale for several candidates; register and bind each chosen candidate to its exact head before giving it release credit.

| Candidate and exact head | Current evidence/status | Next gate and ICC admission gap |
|---|---|---|
| GUW analytic oracle — `feat/v14-guw-analytic-oracle` @ `29065314225aeb26505838cc62e96a301b30324a` | 4 changed paths, 3 commits over master. Local JIT/AOT 2/2 receipt lacks head/source fingerprint, so it is not release evidence yet. | Bind the analytic `D^(3,3,2)` witness and permutation check to this exact source head, rerun the focused gate, then register/update `eshkol_v14_guw_oracle`. |
| Connection runtime — `feat/astra-v14-connection` @ `aaab5989dd6a1a3e1862b70587f9457f4695e843` | 27 paths, 25 differing from master. Bounded numeric IPv4/IPv6 TCP/UDP, synchronization, UTF-8/MessagePack fixes and loopback tests exist. No DNS/TLS/HTTP/WebSocket or compiler-enforced resource ownership; numeric handles need lifetime/cancellation/capability review. | Rebase/port only selected code to current master; resolve CMake, VM and LLVM conflicts; add evidence for resource lifetime/cancellation/capability behavior and missing promised protocols. No matching isolated ICC alias/task. |
| WebGPU residuals — `feat/astra-v14-webgpu` @ `2d0dc90774fbde968bbf5b33982c09566a63fbd9`; alternate VM lane `fix/webgpu-vm-master` @ `5292924738a88dc5bf4ee81ab37bb7db3f09be3c` | First branch is stacked on connection work and has 49 differing paths; VM lane has 23. A broad older train is substantially equal to master. No live browser compute receipt is established. | Compare residuals against current browser/WASM artifacts, select one focused lane, and capture live browser compute evidence on the integrated head. No dedicated candidate ICC alias/task. |
| XLA production — `feat/astra-xla-production` @ `2d8c73a63e52f6ed80f73b1ceeeed62b65bf445f` | 126 paths, 120 differing. PJRT/device lowering and S1–S9 harnesses exist on this head. S9 deliberately FAILS: device training restore, multi-device checkpoint reassembly, and atomic pair publication are incomplete; 2+ addressable PJRT devices are required for the multi-device gate. | Admit stages against the chosen exact current-master head, preserve FAIL on missing devices, complete S9 and resolve model-I/O/allocation dependencies before any v1.4.5 release claim. No current ICC candidate alias/task. |
| Math example stream — `math/g1-ipm-residual` @ `1bc5692996b3e7379f6d186564e1ada4ad68a4de` | 74 paths, 53 differing and 21 equal. Existing exact finite algebra/topology examples overlap shipped code; local IPM JIT/AOT 24/24 evidence lacks fingerprint; new Aoki examples lack equivalent traces. Finite computations are not general theorem proofs. Its ICC alias `eshkol-math-stream` is stale. | Reindex exact current source; classify examples individually, bind focused gates and traces, and keep runtime AD residuals separate. Do not copy its older `run_math_example_gate.sh` wholesale. |

Already shipped/superseded material receives no additional release credit:
ADR-0012 Stages 0–2, the `core.dbsp` library, Noesis packaging/benchmark work,
and the nested-carrier implementation are represented in master. The old
full-carrier head `efed708b5bd5e570adbb132b75f31409ebe022a2` is substantially
superseded. ICC task inventory for this master snapshot returned zero tasks;
candidate registration and source-bound evidence remain an explicit blocker.

## Cross-document status corrections

ADR-0000's existing attainment table is the detailed architectural stage
record. Its evidence supports **2 of 14 stages satisfied, 1 partial (Stage 5),
and 11 not started**. The `0/14, 3 partial` and `0/14, 2 partial` summaries
in `ROADMAP.md` and `docs/COMPILER_ROADMAP.md` were stale. This reconciliation
records the current source snapshot; it does not claim a fresh execution of the
ADR gate scripts. The stage evidence remains in
[`design/adr/0000-unified-trajectory.md`](design/adr/0000-unified-trajectory.md).

ADR-0012 records ABI migration Stages 0, 1, and 2 complete and Stages 3–6
proposed. Consequently v1.4.1 carries the remaining Stages 3–6; describing
Stages 1–6 as future work duplicates completed work. See
[`design/adr/0012-object-abi-staged-migration.md`](design/adr/0012-object-abi-staged-migration.md).

The v1.3.5 release record remains bound to its frozen tag. This document
describes post-tag planning at the stated master snapshot and does not revise
the release record.
