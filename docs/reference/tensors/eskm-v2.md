---
kind: reference
status: current
owner-area: tensors
since: v1.3.5
sources:
  - lib/core/eskm_v2_preflight.c
  - lib/core/eskm_v2_preflight.h
  - docs/design/ESKM_V2_FORMAT_DECISION.md
---
# ESKM v2: experimental internal preflight contract

**Status: experimental, test-only; the format decision remains Proposed.**
This page describes the private preflight validator staged in
`lib/core/eskm_v2_preflight.c`. It is not a supported public checkpoint format
or an installed API. Public model and tensor loading/saving retain their
existing ESKM v1 behavior. The prototype may change following maintainer
review of the [format decision](../../design/ESKM_V2_FORMAT_DECISION.md).

The validator checks a complete immutable byte buffer without constructing
tensors, doing I/O, allocating heap memory, or invoking callbacks. Success
means the whole buffer satisfies the grammar and configured parser limits.
It does not mean that a native or VM backend can materialize its tensors.

## Container bytes

Integers are unsigned little-endian. Offsets are relative to the start of the
supplied buffer. There is no padding or alignment requirement on wire bytes.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | Magic: ASCII `ESKM` |
| 4 | 4 | Version (`u32`): `2` |
| 8 | 4 | Tensor record count (`u32`) |
| 12 | 4 | Required-feature mask (`u32`): currently `0` |
| 16 | 8 | Extension area byte count (`u64`) |
| 24 | variable | Extension area, exactly the declared byte count |
| after extensions | variable | Exactly the declared tensor records |
| end − 4 | 4 | CRC-32 (`u32`) of all preceding bytes |

The checksum is reflected CRC-32/ISO-HDLC: polynomial `0xedb88320`, initial
value `0xffffffff`, final XOR `0xffffffff`. It detects accidental corruption;
it is not authentication. The minimum complete container is 28 bytes. The
canonical empty container is:

```text
45 53 4b 4d 02 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 05 70 e8 c3
```

Its footer encodes CRC `c3e87005`. Zero records and zero extensions are allowed
independently. Every byte before the footer must belong to the declared header,
extensions, or records; trailing bytes are invalid. Required features are
checked against a supported mask of zero before interpreting any extension.

## Extensions and annotations

Each TLV is a type (`u32`), payload byte count (`u32`), and that many payload
bytes. TLVs must exactly consume the extension area. Type zero is invalid.
Unknown nonzero types are optional and skipped after validating their span;
they may repeat, have empty payloads, and use the high bit of the type. They
cannot change tensor interpretation.

Type `1` contains annotations and may appear at most once. Its payload starts
with an entry count (`u32`). Each entry is a key byte count (`u32`), a value
byte count (`u32`), then the key bytes and value bytes. Entries must exactly
consume the payload. An empty annotation table and empty values are valid;
keys must be nonempty and unique by raw bytes. Keys and values are opaque and
may contain NUL or non-UTF-8 bytes. The validator accepts any unique key order.
Canonical writer sorting and annotation exposure/preservation are future work.

## Tensor records

Records retain the [ESKM v1 wire encoding](eskm-v1.md#tensor-record):

1. Name byte count (`u32`), then uninterpreted name bytes.
2. Rank (`u32`), then that many dimensions (`u64` each).
3. Dtype (`u8`), currently only `0` for IEEE-754 binary64.
4. Eight bytes per element, preserving raw binary64 bits in record order.

Names may be empty or repeated. Rank zero has one element. Dimensions are
multiplied in wire order using unsigned 64-bit arithmetic: overflow before the
first zero is invalid; the first zero ends product accumulation. All remaining
dimensions are still consumed. A shape beginning with zero can therefore have
later dimensions larger than a backend's signed range and remain structurally
valid. The final element byte count must also fit unsigned 64-bit arithmetic
and the actual bounded record area. No record enumeration or materialization
is performed by this interface.

## Provisional parser limits and memory

These inclusive ceilings exercise the proposed admission policy. Callers pass
explicit limits and may lower any ceiling, but cannot raise one. The
`ESKM_V2_LIMITS_DEFAULT` initializer supplies all compiled ceilings; zero means
a limit of zero, not an implicit default.

| Resource | Compiled ceiling |
|---|---:|
| Complete buffer, including header and CRC | 268,435,456 bytes (256 MiB) |
| Extension area, including TLV framing | 1,048,576 bytes (1 MiB) |
| TLVs, including empty and unknown types | 1,024 |
| Annotation entries | 1,024 |
| Key bytes, per entry | 4,096 (4 KiB) |
| Value bytes, per entry | 65,536 (64 KiB) |

The extension ceiling also bounds aggregate annotation bytes. Record counts,
name lengths, dimension arrays, and payloads must fit the actual bounded buffer;
they cause no proportional allocation. Wire lengths are validated against the
enclosing span before conversion to host offsets or pointer use.

The caller provides a fixed workspace containing 1,024 key spans of two
`uint64_t` fields each: **16,384 bytes (16 KiB)**, enforced by a size assertion.
Unused entries do not require initialization. Duplicate detection uses an
iterative in-place heapsort, with no recursion or additional per-entry arrays.
Worst-case key-comparison work is `O(A log A * K)`, where `A` is the annotation
count (at most 1,024) and `K` is key length (at most 4,096). Ancillary state is
fixed-size and independent of record/annotation counts; its stack placement
and padding depend on the compiler and target. The validator allocates no heap
memory and does not copy unknown payloads.

The private C17/C++ header declares `eskm_v2_preflight(input, size, limits,
workspace, result)`. It returns `eskm_v2_error` by value and writes the
`eskm_v2_result` only after complete validation. All pointers must be nonnull,
even when `size` is zero. The input, `eskm_v2_limits`, `eskm_v2_workspace`, and
result storage must be disjoint and correctly aligned for their types; the
byte buffer itself needs no additional alignment. These are caller obligations.
The caller owns all storage. Keep the input alive and immutable while using
returned ranges. Workspace contents are scratch and have no output meaning;
separate concurrent calls need separate workspace and output storage. Success
returns `extension_offset`/`extension_bytes`,
`records_offset`/`records_bytes` (excluding the CRC), `record_count`, `tlv_count`,
`annotation_count`, and `has_annotations`. An absent annotation TLV and a present
empty table are distinguished by `has_annotations`. Failure clears the result
entirely and returns only a status and byte offset, so no partial records become
observable.

The buffer limit is an admission check on already supplied bytes. It cannot
prevent a caller from allocating those bytes beforehand, and is not a total
process-memory limit. Before public reader integration, the project still needs
approved numeric backend record/name/rank/dimension/element limits, file
admission before buffering, aggregate memory accounting including materialized
tensors, and transactional cleanup/publication guarantees. Those gates remain
open under GK-SER-05a/05b; this prototype neither changes v1 admission nor
imposes backend signed-dimension limits as new wire rules.

## Failure contract

The return value contains a status and an offset relative to the buffer.
`ESKM_V2_OK` has offset zero. On failure, every byte of a nonnull result is
zeroed, even when another pointer is null. No error string allocation or public
checkpoint diagnostic is performed.

| Status | Meaning / offset |
|---|---|
| `ESKM_V2_INVALID_ARGUMENT` | Null required pointer; offset zero |
| `ESKM_V2_INVALID_LIMITS` | A configured limit exceeds its compiled ceiling; offset zero |
| `ESKM_V2_BAD_MAGIC` | Wrong magic; offset zero |
| `ESKM_V2_UNSUPPORTED_VERSION` | Version is not two; offset 4 |
| `ESKM_V2_UNSUPPORTED_FEATURES` | Required-feature mask is nonzero; offset 12 |
| `ESKM_V2_CHECKSUM_MISMATCH` | Incorrect CRC; offset at the footer start |
| `ESKM_V2_MALFORMED` | Invalid or incomplete wire structure; field offsets below |
| `ESKM_V2_RESOURCE_LIMIT` | Configured admission limit exceeded; corresponding field offset (zero for buffer size) |

A missing field reports its expected start. A bad length reports the length
field, including offset 16 for an invalid extension span. A truncated dimension
array reports the rank field. Shape-product overflow reports the offending
dimension; payload-size overflow or truncation reports the dtype field. A
container shorter than 28 bytes reports offset zero. Extra bytes report the
first extra byte. A duplicate type-1 TLV reports its type field. Duplicate keys
report the later occurrence's key-length field; among multiple duplicates the
earliest such later occurrence is chosen. Duplicate checking runs after all
annotation entries have passed structural and limit validation, so a later
invalid entry takes precedence over an earlier duplicate.

Checks run in this order:

1. Arguments and configured ceilings, complete-buffer cap, and minimum size.
2. Magic, version, and required-feature mask.
3. CRC-32.
4. Extension framing/content and then records, in wire order.

Thus an unsupported required bit wins over a checksum mismatch or malformed
extension. A checksum mismatch wins over structural errors after the fixed
header. Lowered resource limits can change which refusal is reached first.
Tests pin status/offset behavior so public integrations can later map it to
their own diagnostics without relying on a partial result.

## Scope and validation

The parser is compiled into private tests only and is not linked into public
model I/O, the VM, or its browser bundle. The standalone runner builds C and
C++ consumers without LLVM:

```sh
scripts/run_eskm_v2_preflight_tests.sh
# Optional focused AddressSanitizer + UndefinedBehaviorSanitizer build:
CC=clang CXX=clang++ ESKM_V2_SANITIZE=1 scripts/run_eskm_v2_preflight_tests.sh
```

The Bash runner requires C/C++ compilers, Python 3, and `nm`. Its object-import
audit permits only byte operations and stack-protector helpers. This audit is
validated on x86-64 Linux with GCC and Clang; another ABI may need reviewed
nonallocating compiler helpers added to the allowlist. CMake also registers the
C/C++ tests and, when Python is available, the fixture and negative-control
checks. The normal repository CMake configuration still requires LLVM.

The fixture directory `tests/core/fixtures/eskm-v2/` records exact golden bytes,
metadata, hashes, provenance, and expected outcomes. Focused tests cover valid
optional extensions and annotations, scalar/empty records, deterministic
rejection classes and precedence, reduced limit boundaries, and negative
controls that deliberately change expected metadata/status/offset. Existing v1
fixtures remain unchanged.

This is preparatory evidence, not completion of the accepted GK-SER-05a gate.
There are no public v2 producers or consumers, so the native JIT/AOT and VM
source/bytecode v2 interoperability matrix remains future integration work.
Maintainer acceptance of the wire layout, resource policy, and public contract
is still required before public integration. An opt-in writer/API follows
reader integration; default-format changes, compression, new dtypes, and
checkpoint-publication changes are outside this prototype.
