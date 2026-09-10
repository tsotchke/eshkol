# ESKM public tensor-file engine parity

This page keeps its historical filename for existing links. Since #555 merged,
the public tensor APIs use ESKM, superseding the ESKT baseline tested by #615.

`scripts/run_eskm_tensor_engine_parity.py` tests the public `tensor-save` / `tensor-load`
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

This tests **single-record ESKM v1 tensor files**, using the same container
format as ESKM model checkpoints. The public VM dispatch now uses IDs 802/803;
the legacy ESKT IDs 1820/1821 were removed by #555.

The independent oracle implements the [normative ESKM v1 layout](eskm-v1.md):
ASCII `ESKM`, little-endian uint32 version `1`, record count `1`, zero flags,
zero name length, uint32 rank, uint64 dimensions, f64 dtype byte `0`, binary64
payload bits, and a little-endian CRC-32 footer covering all preceding bytes.
It compares whole files, not only payloads. It neither derives expected bytes
from engine output nor accepts either format opportunistically. The offline
oracle tests match four immutable v1.2.4 single-record fixtures, including
signed zero and a specific NaN payload. This does not claim other dtype support,
execution on big-endian hardware, or legacy ESKT reader compatibility.

Only valid modest inputs are read. Scalar/empty tensors, malformed-file
rejection, fuzzing, resource limits, atomic replacement, and format or API
changes are outside this test. It supplies the public tensor positive cross-reader
portion of GK-SER-03, not the entire roadmap packet's negative matrix.

## Run

```sh
cmake --build build --target eshkol-run eshkol-vm-standalone-test --parallel 2
python3 scripts/run_eskm_tensor_engine_parity.py \
  build/eshkol-run build/eshkol-vm-standalone-test --self-test
python3 scripts/test_eskm_tensor_engine_parity.py
ctest --test-dir build --output-on-failure -R '^eskm_tensor_(engine_parity|oracle_self_test)$'
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
a CTest pass or skip. The existing non-Windows CTest registration boundary is
retained; this test-contract update does not establish Windows harness support.
Execution on macOS and other platforms must be verified separately.

The runtime matrix is independent of the model matrix (#597) and atomic-save
implementation (#600). Its offline oracle test uses the already-merged #596
historical fixtures. It does not emit release or ICC evidence.
