# Tensor-file engine parity

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

This tests the public **ESKM v1 tensor-file path** across all four engines. A
single tensor is an unnamed record in the checkpoint format used by
`model-save`/`model-load`; the public dispatch IDs are 802/803.

The implemented ESKM layout is little-endian: `ESKM` magic, version, record
count, flags, each record's name, rank, uint64 dimensions, f64 dtype byte, and
binary64 element bits, followed by a CRC-32 footer. The oracle emits the same
fixed-width bytes and checksum as both implementations.

Only valid modest inputs are read. Scalar/empty tensors, malformed-file
rejection, fuzzing, resource limits, atomic replacement, and format or API
changes are outside this test. It supplies the positive cross-reader
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
because the current native tensor-file implementation is disabled there; execution
on macOS and other platforms must be verified separately.

This test is independent of the ESKM corpus/matrix (#596/#597) and atomic-save
implementation (#600); it needs none of their fixtures, scripts, or runtime
changes. It does not emit release or ICC evidence.
