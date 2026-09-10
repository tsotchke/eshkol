# ESKM v2 format decision: an extensible metadata envelope

- **Status:** Proposed; no v2 writer or reader is authorized by this document
- **Date:** 2026-09-06
- **Technical review updated:** 2026-09-07
- **Author:** Gabriel “Gabe” Kahen
- **Decision owners:** Eshkol model/checkpoint maintainers
- **Packet:** GK-SER-05
- **Depends on:** [PR #596](https://github.com/tsotchke/eshkol/pull/596), specifically its [ESKM v1 contract](https://github.com/Gabriel-Kahen/eshkol/blob/edb614e57904229fc626b23801c9bffbbb248b25/docs/reference/tensors/eskm-v1.md) and [compatibility corpus](https://github.com/Gabriel-Kahen/eshkol/tree/edb614e57904229fc626b23801c9bffbbb248b25/tests/core/fixtures/eskm-v1)
- **Implementation prerequisites:** GK-SER-02 fail-closed all-record preflight and GK-SER-04 atomic checkpoint publication
- **Supersedes:** none; ESKM v1 remains the default and compatibility baseline

The v1 links pin the reviewed dependency while #596 is unmerged. This proposal
does not incorporate its files or claim its runtime acceptance gates passed.

## Context

ESKM v1 has one reserved header word but no extension framing. Deployed readers
have ignored that word, so assigning meaning to a nonzero v1 value would make
old readers silently accept bytes whose semantics they do not understand. The
v1 dtype and record layout are similarly not safe places for an in-place
extension.

There is not yet a product requirement for compression, chunking, streaming,
or a new tensor dtype. Choosing one speculatively would spend compatibility
budget and couple the checkpoint format to an unproven API. The concrete need
is smaller: establish a byte-level v2 envelope that can carry advisory metadata
and can distinguish skippable extensions from features that change tensor
meaning.

This is a design decision, not an implementation change. Existing public save
APIs continue to emit v1. Implementation begins only after this byte layout is
reviewed.

## Decision

ESKM v2 keeps the v1 tensor record, little-endian encoding, and trailing CRC-32
unchanged. It inserts a bounded extension area between a new fixed header and
the records. The extension area contains length-prefixed TLVs that cannot alter
tensor interpretation. A separate required-feature bitmap covers changes that
do alter interpretation.

The first defined extension is optional checkpoint annotations. No required
feature bits are assigned by this decision.

### Fixed header

All integers are unsigned and little-endian. Offsets are from the start of the
file.

| Offset | Size | Field | Required value |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `ESKM` |
| 4 | 4 | format version (`u32`) | `2` |
| 8 | 4 | tensor record count (`u32`) | number of records |
| 12 | 4 | required features (`u32`) | `0` until bits are assigned |
| 16 | 8 | extension byte count (`u64`) | exact size of the TLV area |
| 24 | variable | extension TLVs | exactly `extension byte count` bytes |
| variable | variable | tensor records | exactly `record count` v1 records |
| end - 4 | 4 | CRC-32 (`u32`) | checksum of every preceding byte |

The smallest v2 file is 28 bytes: a 24-byte header, no extensions or records,
and the four-byte checksum. The checksum remains reflected CRC-32/ISO-HDLC with
polynomial `0xedb88320`, initial value `0xffffffff`, and final XOR
`0xffffffff`.

The fixed header, exact extension span, exact record count, complete record
payloads, and checksum are mandatory. Padding and trailing bytes are forbidden.

### Extension TLVs

Each extension is encoded without padding:

| Order | Size | Field |
|---:|---:|---|
| 1 | 4 | extension type (`u32`) |
| 2 | 4 | payload byte count (`u32`) |
| 3 | variable | payload bytes |

Type `0` is invalid. Unknown nonzero types are optional and skippable. A reader
must bounds-check the complete extension area and every TLV before reading or
allocating from an advertised length. The TLVs must exactly consume the
declared extension span.

There is no mandatory flag hidden in the type: high-bit type values are also
optional. Unknown types may repeat, appear in any order, and have zero-length
payloads. Their payloads are opaque, but their framing and all envelope resource
caps still apply. Known types obey their own payload and multiplicity rules.

An optional TLV must not change tensor names, order, shapes, dtypes, element
bits, or the interpretation of any core field. A feature that changes any of
those properties is mandatory and needs an assigned required-feature bit or a
new format version.

A later required feature must not redefine an existing optional type or make
annotations necessary for correct tensor use. Any new semantic encoding needs
its own reviewed representation as well as its required-feature bit.

#### Type 1: checkpoint annotations

Type `1` carries advisory bytewise key/value annotations:

| Order | Size | Field |
|---:|---:|---|
| 1 | 4 | entry count (`u32`) |
| 2 | repeated | entries |

Each entry is `key byte count (u32)`, `value byte count (u32)`, key bytes, then
value bytes. Keys must be nonempty and unique by their bytes. Keys and values
are otherwise opaque. The payload must contain exactly the declared entries
with no trailing bytes, and type `1` may occur at most once.

An absent type-1 TLV and a type-1 payload containing a zero entry count are both
valid. A present type-1 payload is at least four bytes. Empty values are valid;
empty keys are not. Opaque bytes may include NUL and need not be UTF-8.

Writers produce entries in ascending unsigned-octet lexicographic order for
deterministic output; a key that is a prefix sorts first. Readers may accept any
unique order. An implementation that does not expose annotations must still
parse and validate type `1`, but it may then discard the values. Annotations
must never be required for correct tensor use. The current load-then-save APIs
are permitted to discard annotations until an API explicitly preserves them.

### Tensor records

After the extension area, records use the ESKM v1 record encoding byte for byte:

1. name byte count (`u32`) and name bytes;
2. rank (`u32`) and `rank` dimensions (`u64` each);
3. dtype (`u8`), currently `0` for binary64; and
4. `8 * product(dimensions)` raw element bytes.

The v1 rules for names, record order, raw binary64 bit preservation, and the
ordered shape product also apply: rank zero has one element; the first zero
dimension ends the product calculation, while overflow before that zero is
invalid. Tensor names need not be unique on the wire. An implementation's
documented name/materialization restrictions remain separate from wire rules;
the unique-key requirement above applies only to annotations.

V2 does not add dtypes, compression, record framing, alternate endianness, or
streaming. Those features change core interpretation or admission behavior and
require a separately reviewed mandatory feature or later version.

## Feature and version rules

Required-feature bits are semantic promises, not capability hints. A reader
must reject a v2 file if any set bit is outside its supported mask. It must do
so after validating enough fixed bytes to identify the condition but before
parsing extensions, performing extension-derived allocation, materializing
tensors, or publishing a partial model. Reading the bounded file buffer is not
an extension-derived allocation.

No required-feature bit is assigned here, so the supported and valid mask for
the first v2 implementation is zero. A future decision assigning a bit must
specify its byte effects, resource limits, interaction with every earlier bit,
and a positive and negative compatibility fixture.

The v1 reserved word remains reserved and canonical writers continue to write
zero. It must never become a feature bitmap: existing v1 readers have ignored
it. Hardened readers must reject nonzero v1 reserved values, but that refusal
does not make reinterpretation safe.

## Reader and writer behavior

A version-aware reader dispatches before record parsing:

- wire version `1` uses the exact v1 decoder and retains all v1 limits;
- wire version `2` uses the v2 decoder defined here; and
- every other version is rejected with no partial result.

"V1.2 checkpoint" refers to the Eshkol release that produced it; its wire
version is still `1`. A v2-capable reader must continue to accept the immutable
v1 corpus's accepted fixtures without changing tensor metadata or payload bits,
and reject its malformed fixtures.

Existing v1 readers reject v2 because the version word is not `1`. Existing
public save APIs continue to emit canonical v1 so their output remains readable
by those readers. Emitting v2 requires a separately reviewed opt-in API or an
explicitly approved default-version change.

Here, public save/load means the ESKM model path and ESKM single-tensor entry
points. Update after #555 merged on 2026-09-08: native and VM language
`tensor-save`/`tensor-load` now both dispatch to ESKM v1. The ESKT boundary
references in the original review gates below describe the pre-#555 baseline;
they do not require restoring legacy ESKT dispatch. This proposal makes no
further dispatch change and remains Proposed. Test both public tensor paths
explicitly, preserving their current ESKM v1 output unless a separate API
decision approves a change.

Public load boundaries must report unsupported versions, unknown mandatory
features, and malformed extension lengths through a deterministic, nonempty
checkpoint-error diagnostic. Treating a rejected input as a successful empty
model or publishing a partially materialized model is forbidden.

Entry-point constraints remain separate from wire validity. A model loader may
accept a zero-record container; a single-tensor loader still requires exactly
one record after the container has passed structural validation.

## Compatibility matrix

| Producer | Input | V1 reader | V2-capable reader |
|---|---|---|---|
| Existing writer | valid v1 | accept within documented limits | accept through v1 decoder within the same limits |
| V2 writer | valid v2, no extensions | reject version | accept within documented limits |
| V2 writer | v2 with type-1 annotations | reject version | validate, then accept within documented limits; values may be discarded |
| Future writer | v2 with unknown optional TLV | reject version | skip TLV and accept core tensors within documented limits |
| Future writer | v2 with unknown required bit | reject version | reject before materialization |
| Future writer | unknown format version | reject version | reject version |

## Canonical empty example

This is the complete 28-byte encoding of a v2 container with zero records,
zero required features, and no extensions:

```text
45 53 4b 4d  02 00 00 00  00 00 00 00  00 00 00 00
00 00 00 00  00 00 00 00  05 70 e8 c3
```

The final four bytes are the little-endian CRC-32 `c3e87005` of the preceding
24 bytes. This example reserves no permission to emit v2 before implementation
review and compatibility tests land.

## Resource and failure rules

A v2 reader must:

- enforce a finite file-size limit before buffering the input;
- verify the checksum and fixed header before materialization;
- reject an extension span that exceeds the checksummed payload;
- use subtraction-based bounds checks for every length;
- cap extension bytes, TLV count, annotation count, key bytes, and value bytes
  before allocation;
- reject empty or duplicate annotation keys, duplicate type-1 TLVs, arithmetic
  overflow, incomplete TLVs, and non-exact span consumption;
- preflight every tensor record before publishing any model object; and
- release temporary parser allocations on rejection and publish no partial
  model; failed materialization must reclaim allocations made for that load
  without disturbing pre-existing caller objects.

### Proposed initial admission profile

These inclusive caps are part of the decision awaiting review. They bound the
first v2 implementation; they are admission policy, not new integer widths or
v1 limits. There is no requirement to allocate the maximum-sized buffer.

| Resource | Maximum | Accounting |
|---|---:|---|
| Complete v2 file | 256 MiB (268,435,456 bytes) | Header, extensions, records, and CRC together; check before buffering |
| Extension area | 1 MiB (1,048,576 bytes) | All TLV headers and payloads together |
| TLVs | 1,024 | All known and unknown TLVs, including empty ones |
| Annotation entries | 1,024 | Across the sole permitted type-1 TLV |
| Annotation key | 4 KiB (4,096 bytes) | Per entry; minimum one byte |
| Annotation value | 64 KiB (65,536 bytes) | Per entry; minimum zero bytes |

The extension cap also bounds aggregate annotation bytes, including entry
framing. Per-entry maxima cannot all be reached simultaneously. These modest
metadata limits target advisory labels, not embedded datasets; the initial file
cap is a conservative bounded-buffer policy and needs maintainer confirmation
against intended checkpoint sizes. Changing a cap later requires a documented
admission-policy review and updated compatibility evidence.

Enforce limits even when metadata is discarded. Skip unknown payloads by a
validated span rather than copying their bytes. Validate each nested length
against its enclosing span, excluding the CRC; a valid file checksum does not
waive structural validation. Once total length is at least 28, the extension
count must be at most `file_size - 28` before conversion to a host-sized offset.
Zero extensions and zero records are allowed independently.

Core tensor limits and aggregate memory accounting remain GK-SER-02
prerequisites. Before a v2 parser ships, the normative reference must tabulate
each backend's numeric record/name/rank/dimension/element and peak-memory limits,
including the input buffer, validation scratch, and materialized objects, and
the parser must enforce them before the corresponding allocation. The file cap
alone is not a peak-memory bound. This proposal does not invent or expand those
backend capabilities. V1 retains its existing admission policy.

A file may be wire-valid but rejected as unsupported-resource when it exceeds
these caps or a backend's documented materialization limits. Cross-engine
acceptance is required within the common supported profile; resource refusals
outside it must be explicit. Numeric cap tables and boundary evidence must ship
with the parser, not be deferred until the final implementation PR.

CRC-32 detects accidental corruption only. It is not authentication and must
not be presented as protection against a malicious writer.

## Decision lifecycle

Maintainer byte-level review advances this document from **Proposed** to
**Accepted** and authorizes implementation. Acceptance of the decision does not
claim that v2 exists. The status advances to **Implemented** only after the
gates below pass and the normative `docs/reference/tensors/eskm-v2.md` ships.

The decision review must explicitly settle the header offsets and CRC example,
required-feature/optional-TLV distinction, type-1 grammar and metadata-loss
policy, proposed admission caps, v1/ESKT compatibility boundary, and the gates
below. Record the accepting maintainer's review link and the accepted document
commit when changing status. Filing or merging a Proposed document alone is
not byte-level acceptance. Public opt-in API signatures/native IDs remain a
separate review prerequisite for the writer slice.

Until then, GK-SER-05 has completed its design slice but not its implementation
or roadmap acceptance.

## Required implementation gates

The implementation must prove:

1. every accepted and rejected v1 golden fixture behaves unchanged;
2. all available engines accept a canonical one-record v2 fixture through the
   ESKM model entry point, with the native tagged C and VM ESKM single-tensor
   entry points also checked; native language ESKT dispatch is unchanged. The
   empty example above is checked only through an internal status-bearing
   container parser because NULL/NIL cannot distinguish successful emptiness
   from rejection;
3. known annotations and unknown optional TLVs leave tensor bytes unchanged;
4. unknown required bits and unknown versions reject before extension-derived
   allocation or tensor materialization and emit a deterministic nonempty
   diagnostic on every public engine axis;
5. truncated, overflowing, duplicate, trailing, and checksum-invalid extension
   cases reject without a partial result;
6. native JIT, native AOT, VM source, and VM bytecode agree on names, record
   order, rank, dimensions, dtype, element count, and raw payload bits within
   their common supported profile; any capability restriction is documented;
7. default ESKM writers still emit byte-identical v1 unless the caller opts into
   v2, and native language ESKT output remains unchanged;
8. the normative reference and public API documentation publish the approved
   caps and error contract with the parser; bounded tests check limits just
   below, at, and above each cap, with allocation accounting proving refusals
   precede the forbidden allocation (large-cap checks use admission-unit tests
   or a reduced test budget, not large files); and
9. opt-in v2 saves use GK-SER-04 publication, including its injected pre-commit
   failure check that the old destination remains intact.

The v1 matrix must include a CRC-correct nonzero-reserved-word fixture and a
CRC-correct unknown-version fixture. The reserved-word refusal is an explicit
GK-SER-02 prerequisite, not a behavior claimed by #596's historical corpus;
add that case separately without rewriting the immutable v1 fixtures. The v2
matrix must cover unknown required bits with no extension-derived allocation,
bounded unknown optional TLVs,
duplicate type-1 extensions and keys, count and length overflow, incomplete
TLVs, type zero, and exact extension-span consumption. Positive cases include
absent annotations, zero entries, empty values, opaque NUL/non-UTF-8 bytes,
unsorted unique keys, and repeated/empty/high-bit unknown types. Record fixture
hashes, expected outcomes, producer/consumer entry points (including the concrete
native/VM bindings tested), and test commands; each new rejection assertion
needs a bounded valid control.

### Follow-on PR slices (only after decision acceptance)

| Slice | Focus and prerequisite | Acceptance evidence |
|---|---|---|
| GK-SER-05a | Internal v2 preflight parser, known type-1 validation, bounded fixtures, and normative `eskm-v2.md`; requires GK-SER-02 and approved cap tables | Gates 1, 3–5, 8 at the parser boundary; exact empty and one-record bytes/CRC; no public writer |
| GK-SER-05b | ESKM reader integration on native and VM, annotation validation/discard behavior, and public checkpoint-error diagnostics | Gates 2–6 on the available engine axes; allocation accounting and no partial result; no ESKT dispatch changes |
| GK-SER-05c | Explicitly reviewed opt-in v2 writer/API using GK-SER-04 publication; default ESKM writer unchanged | Gates 3, 7, 9; canonical sorted annotations, byte-identical repeated saves and v1 output, reader consumption of emitted v2 |
| GK-SER-05d | End-to-end producer/consumer compatibility matrix | All four engines write/read each other's ESKM output; compare tensor names, order, shapes, dtype, and raw payload bits; annotations follow the approved API contract; agree on bounded rejection classes and diagnostics |
| GK-SER-05e | Final normative reference, evidence links, and ownership status | All gates recorded PASS for supported targets on identified commits; Accepted → Implemented only with complete evidence |

Each slice must record exact base/head SHAs, dependencies, commands, and
PASS/FAIL/NOT RUN results. An unavailable engine is NOT RUN and leaves that
acceptance gate open; it cannot be counted as agreement. GK-SER-05e updates the
reference that shipped with 05a rather than documenting an already-public
format for the first time. The writer must reuse GK-SER-04's publication
contract unchanged rather than reimplement atomic saving in the format parser.

## Alternatives rejected

### Reuse the v1 reserved flags

Rejected because deployed readers have ignored them. A new meaning could be
silently misread by an old implementation.

### Add compression, chunking, streaming, or new dtypes now

Rejected without a concrete workload and API. Each changes resource behavior
or tensor interpretation and deserves an evidence-backed required feature or a
later version.

### Encode metadata as JSON

Rejected to avoid adding parser, canonicalization, number, and Unicode
semantics to the checkpoint compatibility boundary. Length-prefixed bytes are
smaller and independently bounded.

### Switch checksum or byte order

Rejected because neither change addresses the current extension problem.
Changing them would enlarge the migration while providing no compatibility
benefit.

## Consequences

V2 gains a bounded place for advisory metadata and a fail-closed route for
future semantic features. V1 remains frozen and remains the default interchange
format. The cost is a second parser path and explicit metadata-loss behavior in
the current API; both costs are visible before implementation begins.
