# ESKM v1 scalar and empty-tensor VM materialization

- **Status:** Proposed; requires maintainer review before implementation.
- **Decision owners:** ESKM maintainers and VM memory/runtime maintainers.
- **Scope:** A serialization-local adapter supporting the remaining scalar/empty
  compatibility gate in GK-SER-01 and GK-SER-03.
- **Implementation and acceptance:** Not started; this document supplies no
  runtime acceptance evidence and does not complete either work packet.

## Problem and intended behavior

ESKM wire version 1, used by the v1.2 checkpoints, can represent a scalar with
rank 0 and one binary64 element, and an empty tensor with shape `[0, 3]` and no
elements. The historical valid fixtures in
[PR #596](https://github.com/tsotchke/eshkol/pull/596) include both. Native model
loading can materialize them; current VM model loading cannot.
[PR #597](https://github.com/tsotchke/eshkol/pull/597) explicitly excludes this
gap from its cross-engine gate.

After a reviewed implementation, loading either fixture through ESKM model I/O
would produce a tensor with its original rank, dimensions, dtype, and payload.
Native JIT, native AOT, VM source, and VM bytecode would inspect and rewrite that
tensor without changing the checkpoint bytes. Rank 0 must remain rank 0; an
empty tensor must remain empty. This is an I/O compatibility promise, not a new
promise about general tensor arithmetic or construction.

## Source evidence and dependencies

This diagnosis is a source read, not a newly executed runtime reproduction.
The source baseline is `upstream/master` at
`5ce74beeac25aca56c0c8129083e6db77f96f7dc`.

| Source | Relevant behavior |
|---|---|
| [`lib/core/model_io.cpp`](../../lib/core/model_io.cpp), `tensor_from_record` | Materializes dimensions only when rank is positive and elements only when count is positive; the empty shape product for rank 0 is 1. |
| [`lib/backend/vm_model_io.c`](../../lib/backend/vm_model_io.c), `vm_model_make_tensor_value` | Decodes payload into temporary storage and calls `vm_tensor_from_data`; boxes the result as `HEAP_TENSOR` / `VAL_TENSOR`. |
| [`lib/backend/vm_tensor.c`](../../lib/backend/vm_tensor.c) | `vm_tensor_from_data` rejects null data, `vm_tensor_new` rejects rank 0 and nonpositive totals, and `vm_tensor_bind_dims` rejects rank 0. |
| [`lib/backend/vm_native.c`](../../lib/backend/vm_native.c), calls 413/414 | Language `tensor-shape` iterates rank; `tensor-data` returns a vector and iterates element count. This supports the proposed observation surface by inspection, pending tests. |
| [`lib/backend/vm_arena.h`](../../lib/backend/vm_arena.h) | Existing `vm_alloc_object` and `vm_alloc` allocate in the active region or global arena. |
| [`lib/backend/vm_region_evac.c`](../../lib/backend/vm_region_evac.c) | Tensor payloads are handled by the existing retention/scan path, including pointers within `VmTensor`; this must be verified for the adapter's lifetime, not replaced. |

[PR #602](https://github.com/tsotchke/eshkol/pull/602), inspected at
`e588e1f10c2c4b02d85efd26f943b83e4c05f861`, deliberately refuses rank 0 and
zero-element records in preflight before persistent VM allocation. It preserves
the existing VM restriction while waiting for a reviewed adapter. Its dependency,
[PR #555](https://github.com/tsotchke/eshkol/pull/555), was inspected at
`e737c10a28c34099d502b0b788dba09c89d06875`. Both were open when inspected.

Implementation must follow review and coordinate with the landed #555/#602
behavior. Adding materialization alone would leave #602's refusal in place;
removing that refusal alone would admit objects that still cannot be built.
Update both together in a subsequent focused implementation PR, keeping whole-file
preflight ahead of persistent object creation. Preserve the remaining validation
and failure contracts from those PRs.

The ownership basis is the
[Gabe roadmap](https://github.com/tsotchke/eshkol/blob/10301403f6ae434321108751bc46a0c3bad7ca62/docs/development/GABE_KAHAN_ROADMAP.md),
read from `upstream/docs/gabe-kahan-roadmap`, and
[`CONTRIBUTING.md`](../../CONTRIBUTING.md). The roadmap allows VM serialization
work but requires a reviewed serialization adapter at the object/memory boundary.
This proposal does not start or finish a milestone implementation, import the
roadmap branch, or change its active-task table.

## Proposed adapter contract

Keep the adapter private to `lib/backend/vm_model_io.c`, used by both VM ESKM
load entry points. Ordinary positive-rank, positive-count tensors retain their
existing construction path. Only scalar and zero-element materialization use
the reviewed special path. Do not change `VmTensor` layout, object headers,
heap tags, native IDs, public signatures, or the general tensor constructors.

The adapter consumes metadata and payload ranges already admitted by the I/O
preflight. Rank, each dimension's VM representation, count, allocation sizes,
and any stored strides must be representable under the settled I/O limits.
Zero total is not permission to skip checks on the remaining dimensions or
strides. This proposal adds no rank-limit increase or global file-size policy.

### Decisions requiring explicit maintainer disposition

| Decision | Proposed choice | Review consequence |
|---|---|---|
| D1: Rank-0 representation | `n_dims = 0`, `total = 1`, `dtype = VM_TENSOR_DTYPE_F64`; bind `shape` and `strides` to the object's inline arrays, initialized deterministically, with zero logical entries. Allocate one double for the scalar's original bits. | Approve an explicit I/O-local exception to the current `vm_tensor_bind_dims` construction rule; keep that helper's behavior unchanged. Never substitute shape `[1]`. |
| D2: Empty representation | Preserve rank and every dimension, set `total = 0`, keep f64 dtype, and allocate one initialized, non-null double-sized arena storage slot with `owns_data = 1`. That slot is storage only, never an element. | Approve the non-null storage convention; no payload read/write or element iteration may include the slot. Do not use a global singleton or a pointer into the file buffer. |
| D3: Empty strides | Use the existing dimension binding for positive rank and I/O-local checked row-major stride calculation. Shape `[0, 3]` has strides `[3, 1]`; reject unrepresentable metadata before materialization under the settled I/O rules. | Approve this convention rather than inferring strides from zero total or changing general tensor stride rules. |
| D4: Ownership and lifetime | Allocate the tensor through `vm_alloc_object` with `VM_SUBTYPE_TENSOR`, keep all owned storage in the same existing region system, then use the existing `HEAP_TENSOR` / `VAL_TENSOR` box. | Require evidence that returned tensors survive the loader's temporary-buffer release and ordinary function/region exit using existing retention machinery. No new ownership or evacuation scheme is authorized. |
| D5: Supported observation surface | ESKM load/save plus existing `tensor-shape`, `tensor-data`, and metadata checks needed for exact round trips. | Accept this narrow compatibility scope. General construction, indexing, reshape, broadcasting, arithmetic, AD, and device behavior receive no new guarantee. |

For D1, the rank-0 exception must be documented at the existing construction
invariant if approved; that comment clarification would not relax the general
helper. For D2, non-null storage is proposed to avoid making null data a new
validity convention across unrelated consumers. For D4, all fields must be
initialized before the tensor is published, and no stack, temporary decoding,
or file-buffer pointer may escape. Allocation failure retains the existing
load failure behavior; transactional arena rollback is not promised.

These objects remain ordinary tensor values and can reach other operations.
Maintainers must review that consequence rather than treat the narrow promise
as an enforced type restriction. If safe integration requires general tensor
rules, layout, or region-machinery changes, revise the design and obtain the
corresponding review before expanding implementation scope.

## Compatibility and alternatives

Preserve the default ESKM v1 writer: magic, version, flags, names/order, rank and
dimension encoding, dtype byte 0, binary64 payload bits, and CRC coverage remain
unchanged. The empty storage slot is never serialized. Existing accepted
ordinary tensors must produce the same bytes. No new extension or feature bit
is needed because these shapes already exist in the historical format.

This is independent of
[PR #613](https://github.com/tsotchke/eshkol/pull/613)'s proposed v2 wire-format
decision. Accepting this adapter would not accept v2, and v2 review cannot close
the existing v1 materialization gap.

Changing the general constructors would reach tensor operations beyond ESKM and
needs a different scope. Encoding a scalar as `[1]`, dropping empty dimensions,
or replacing a tensor with a special wrapper would break metadata parity or
require an API/type decision. Continuing the current refusal is the fallback
until review and implementation; it leaves the SER-01/03 gate open.

## Follow-on acceptance gate (not executed here)

Use the unchanged valid fixtures and manifest from #596, inspected at
`5ad058b2bc416c97a76c21f246d6b3bb4044b48f`:

| Fixture | Required metadata/payload |
|---|---|
| `scalar.eskm` | One unnamed record; rank 0; dimensions `[]`; dtype 0; count 1; binary64 bits `4045400000000000`. |
| `empty-0x3.eskm` | One unnamed record; rank 2; dimensions `[0, 3]`; dtype 0; count 0; zero payload bytes. |
| `ordinary-2x3.eskm`, `rank8.eskm`, `multi-tensor.eskm`, `large-32x32.eskm` | All manifest names/order, ranks, dimensions, dtype, element counts, and exact payload bits remain unchanged. |

Extend #597's focused compatibility runner after its dependencies settle:

1. Each of native JIT, native AOT, VM source, and VM bytecode loads all six valid
   fixtures using `model-load`, checks semantic metadata against the manifest,
   and rewrites through `model-save`. Compare complete output bytes to the
   corresponding golden file, including signed zero and NaN payload bits.
   Printed numeric equality is insufficient.
2. For each fixture, feed every engine's rewrite to every consumer engine:
   16 producer/consumer combinations per fixture. Verify metadata and exact
   golden bytes again. The scalar/empty producer step deliberately loads the
   historical fixture, so it does not require a new language constructor rule.
3. Also verify the public ESKM `tensor-load` / `tensor-save` route on the five
   single-record fixtures across all four engines. Since #555 merged on
   2026-09-08, native and VM public tensor I/O both use ESKM; the previous
   separate native ESKT route is no longer the baseline. This proposal remains
   Proposed and does not authorize additional dispatch changes.
4. Add bounded positive cases returning a loaded scalar/empty model from an
   ordinary function/region scope, then inspect and rewrite after return.
   Prove the payload, inline shape pointers, and empty storage remain valid
   after temporary input storage and the loader scope end. Use existing runtime
   test infrastructure without changing evacuation machinery.
5. Establish a regression control: the new positive VM scalar/empty assertions
   fail against the settled pre-adapter #602 behavior and pass with the adapter.
   A test-only change to an expected rank/dimension or scalar payload must make
   the matrix fail; do not generate a new malformed-file campaign for this slice.

The follow-on PR must retain #602's other admission guarantees while replacing
its unsupported-shape expectations with the reviewed acceptance behavior.
Do not claim completion from the already passing ordinary-model matrix alone.
Record exact revisions, commands, available engines, and PASS/FAIL/NOT RUN for
the complete gate; unavailable engines do not count as passes. This proposal
does not authorize changes to `.icc/` policy or release evidence.

## Delivery boundary and verification of this proposal

This PR intentionally changes only this design document. Implementation, fixture
changes, malformed-input/resource campaigns, general tensor API/shape rules,
object ABI changes, generated files, `.icc/`, and release work are excluded.
The follow-on implementation should be one separately reviewable I/O adapter
and compatibility-test slice after decisions D1-D5 are accepted and #555/#602
are coordinated. Revise this document if review chooses different conventions.

For this documentation-only PR, source references and fixture metadata are
checked by reading the pinned sources. Run `git diff --check`,
`python3 scripts/gen_api_docs.py --check`, and
`python3 scripts/check_test_coverage.py`; report their actual results in the PR.
Runtime builds, engine executions, platform tests, and the future acceptance
gate are **NOT RUN** here. No runtime pass is inferred from source inspection.
