# ESKT tensor-file engine parity

`scripts/run_eskt_engine_parity.py` tests the public `tensor-save` / `tensor-load`
path across native JIT, native AOT, VM source, and VM bytecode. Each engine
produces three files; each of the four consumers loads and rewrites all twelve
producer files. This is a literal 4 × 4 matrix with 48 consumer rewrites.

The fixture uses small f64 tensors of shapes `(6)`, `(2 3)`, and
`(1 1 1 1 1 1 1 2)`. Every producer and consumer checks tensor identity, rank,
dimensions, element count via `vector-length` of `tensor-data`, and f64 dtype.
Native dtype is a symbol and VM dtype is a string; the fixture accepts those
representations of the same f64 type. Python independently encodes expected
files and compares every producer output and consumer rewrite byte for byte,
including signed zero and fractional payload values. Runtime success requires
exactly one success marker and no failure marker, as well as a zero exit code.

## Format scope

This tests **ESKT v1 tensor files**, distinct from **ESKM model checkpoints**.
The current public VM dispatch uses IDs 1820/1821. The older model-I/O helper
IDs 802/803 do not identify this public path.

The implemented ESKT layout is host-endian: uint32 magic `0x45534B54`, uint32
version `1`, uint32 rank, one int64 per dimension, then one binary64 per
element. On little-endian hosts the magic bytes spell `TKSE`. There is no
dtype field, stored element-count field, or checksum. The oracle uses fixed
integer widths without alignment padding and the host's byte order, matching
the existing implementation. It does not assert cross-endian portability or
preservation of other dtypes.

Only valid modest inputs are read. Scalar/empty tensors, malformed-file
rejection, fuzzing, resource limits, atomic replacement, and format or API
changes are outside this test. It supplies the ESKT positive cross-reader
portion of GK-SER-03, not the entire roadmap packet's negative matrix.

## Run

```sh
cmake --build build --target eshkol-run eshkol-vm-standalone-test --parallel 2
python3 scripts/run_eskt_engine_parity.py \
  build/eshkol-run build/eshkol-vm-standalone-test --self-test
ctest --test-dir build --output-on-failure -R '^eskt_tensor_engine_parity$'
```

`--self-test` additionally requires 48 refusals by the byte oracle, one for
each rewrite, with deliberately incorrect expected payload bytes. It changes
only Python expectations in memory, never the files consumed by Eshkol.
CTest includes these controls. Compilation happens once for AOT and bytecode;
all execution and file outputs use a fresh isolated directory. The runner
accepts explicit executable paths, `--timeout` per invocation, `--keep`, and
`ESHKOL_TEST_TMPDIR` / `ESHKOL_TEST_TMP_ROOT` for the temporary parent.
Failures retain logs and files; successful runs clean up unless retention is
requested with `--keep` or `ESHKOL_TEST_KEEP_TMPDIR`.

Exit 0 means PASS, 1 means FAIL, and 125 means INFRA (missing executables,
timeout, interrupted process, or filesystem failure). INFRA does not count as
a CTest pass or skip. Native Windows is excluded from CTest registration
because the current native ESKT implementation is disabled there; execution
on macOS and other platforms must be verified separately.

This test is independent of the ESKM corpus/matrix (#596/#597) and atomic-save
implementation (#600); it needs none of their fixtures, scripts, or runtime
changes. It does not emit release or ICC evidence.
