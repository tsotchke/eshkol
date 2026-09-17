# Experimental ESKM v2 golden fixtures

These three small containers pin the proposed format for the private preflight
validator. They are independently constructed specification examples, not
outputs of a public Eshkol v2 writer. Public model I/O remains v1.

`manifest.json` records SHA-256, size, CRC-32, extension order, opaque annotation
bytes, record names, dimensions, dtype, and exact binary64 element bits. Each
`elements_hex` entry is a numeric 64-bit bit pattern; the file stores it in
little-endian order. Opaque name/key/value fields are byte-order hex strings.

- `empty.eskm`: canonical 28-byte empty container, CRC `c3e87005`.
- `raw-bits.eskm`: one named 2-by-3 record preserving positive/negative zero,
  one, positive/negative infinity, and a NaN payload.
- `annotations-scalar.eskm`: a scalar with empty name, three out-of-order opaque
  keys (including a prefix pair), empty and opaque values, and repeated unknown
  optional TLVs with the high type bit set.

Python standard-library `struct.pack` and `zlib.crc32` produced these bytes
without calling the Eshkol serializer or preflight validator. The checker
reconstructs the whole stream from the manifest and compares every byte in
addition to sizes, hashes, and CRCs:

```sh
python3 scripts/check_eskm_v2_fixtures.py --self-test
```

The C test separately checks acceptance, result ranges/counts, representative
metadata/raw bits, and the literal canonical empty vector. Rejection tests are
small in-memory constructions rather than a second on-disk corpus. Keep these
goldens stable; a future proposed-format change needs explicit review of both
metadata and bytes. They make no promise of a supported public v2 format.
