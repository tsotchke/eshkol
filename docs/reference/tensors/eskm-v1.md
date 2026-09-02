# ESKM v1 checkpoint wire format

This page specifies the byte-level ESKM version 1 format used by Eshkol's
model-I/O implementation and by the bytecode VM's tensor persistence path. It
is a compatibility contract for checkpoint readers and writers, not a
description of a particular in-memory tensor implementation. The native
compiler's `tensor-save` and `tensor-load` dispatch uses the separate ESKT
single-tensor format; that format is not specified here.

The normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their
usual RFC 2119 meanings.

## Byte order and primitive fields

All integers are unsigned and little-endian. Offsets below are byte offsets
from the start of the file.

| Offset | Size | Field | Required value |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `ESKM` (`45 53 4b 4d`) |
| 4 | 4 | format version (`u32`) | `1` |
| 8 | 4 | tensor record count (`u32`) | number of records that follow |
| 12 | 4 | reserved flags (`u32`) | `0` |
| 16 | variable | tensor records | exactly `record count` records |
| end - 4 | 4 | CRC-32 (`u32`) | checksum of every preceding byte |

The smallest valid container is 20 bytes: the fixed 16-byte header, zero
records, and the four-byte footer.

The footer is the standard reflected CRC-32/ISO-HDLC value (polynomial
`0xedb88320`, initial value `0xffffffff`, final XOR `0xffffffff`), equivalent
to `zlib.crc32(file_without_footer)`. The four footer bytes are not included in
the checksum.

Writers **MUST** write zero reserved flags. Conforming readers **MUST** reject
an unsupported version, a checksum mismatch, a truncated field, or bytes left
over after the declared records.

The v1.2 readers and the pre-hardening implementation inspected for this
document read and ignored nonzero flags. ESKM v1 assigns those values no
meaning, so producers still have no valid nonzero value to write. The
compatibility corpus contains only the canonical zero value and makes no claim
about how a loader handles a nonzero reserved field.

## Tensor record

Records are concatenated without padding or alignment bytes. There is no
per-record length or checksum.

| Order | Size | Field |
|---:|---:|---|
| 1 | 4 | name byte length `N` (`u32`) |
| 2 | `N` | name bytes |
| 3 | 4 | rank `R` (`u32`) |
| 4 | `8 * R` | `R` dimensions (`u64` each) |
| 5 | 1 | dtype code (`u8`) |
| 6 | `8 * product(dimensions)` | element bit patterns (`u64` each) |

The only v1 dtype code is `0`, IEEE-754 binary64. Each element is stored as its
raw 64-bit representation, so signed zero, infinities, subnormals, and NaN
payloads are preserved. Elements appear in the tensor's flat contiguous order;
Eshkol tensors use row-major indexing. Readers **MUST NOT** numerically convert
or canonicalize these bits while decoding v1.

Names are uninterpreted bytes on the wire. An empty name is valid and is what
the ESKM single-tensor writer used by the bytecode VM writes for its record.
Eshkol model writers conventionally use UTF-8 names, but v1 neither requires
UTF-8 nor requires names to be unique. Record order is significant and
**MUST** be preserved.

A zero record count is a valid empty model container. The ESKM single-tensor
loader used by the bytecode VM requires exactly one record but does not require
its name to be empty; the model loader accepts any record count. These are
entry-point constraints, not different wire formats.

The element count accumulator starts at one and visits dimensions in stored
order. The empty product for rank zero is therefore one. A zero dimension sets
the count to zero and stops the product calculation, so later dimensions do
not affect the element count; an overflow encountered before the first zero is
still invalid. This order-sensitive rule is the one implemented by the v1.2
writer's paired reader. A reader **MUST** reject such an overflow or an element
payload that cannot fit in the remaining file bytes before allocating for it.

## Decoder validity and implementation limits

Wire validity is independent of a backend's in-memory limits. A decoder may
reject a valid file when it cannot materialize its rank, dimensions, element
count, or allocation, but it must do so as an unsupported-resource condition,
not reinterpret the bytes as a different tensor.

The v1.2 C++ model-I/O implementation supports the rank-zero and zero-extent
encodings in this specification. The v1.2 bytecode VM had an eight-dimension
materialization limit and could not materialize rank-zero or zero-extent
tensors; those were backend restrictions, not alternate wire rules. The rank-8
fixture deliberately marks the historical VM boundary.

Before allocating from advertised lengths, readers should establish that the
record count, name length, rank, dimension array, and element payload can fit
in the checksummed bytes. A file is valid only when parsing the declared record
count consumes the payload exactly.

## Compatibility fixtures

The immutable corpus is in
[`tests/core/fixtures/eskm-v1/`](../../../tests/core/fixtures/eskm-v1/). Its
[`manifest.json`](../../../tests/core/fixtures/eskm-v1/manifest.json) records
the exact producer tag and commit, file SHA-256 values, expected metadata,
payload bit patterns, and acceptance or rejection for every case.

The five accepted fixtures were written by the `write_checkpoint` implementation
from annotated tag `v1.2.4` (tag object
`4e07f166a7a0da28d24c78fb1c1af4258c4c1845`, peeled commit
`b98dc8b32399de739a037e9fa0a470bf0426eca9`). The malformed fixtures are
deterministic mutations of `ordinary-2x3.eskm`; the manifest identifies each
mutation as a structured operation rather than claiming the historical writer
emitted invalid data. The checker reconstructs and compares every mutation.

Verify the corpus without configuring or building Eshkol:

```bash
python3 scripts/check_eskm_v1_fixtures.py
python3 scripts/check_eskm_v1_fixtures.py --self-test
```

The checker's 64 KiB per-file ceiling is an immutable-corpus policy, not an
ESKM format size limit. The checked-in corpus is 923 bytes total and its
largest file is 101 bytes.

The self-test copies the corpus to a temporary directory, changes a scalar
payload bit, recomputes its CRC-32 and manifest SHA-256, and requires the normal
checker to reject the changed bit pattern. It never modifies the committed
fixtures.
