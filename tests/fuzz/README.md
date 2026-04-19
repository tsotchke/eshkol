# Eshkol Fuzzing Harnesses

libFuzzer drivers for the high-value ingest surfaces. Each harness is
a standalone libFuzzer binary that exercises one parser / loader /
validator in a tight loop under ASan+UBSan instrumentation.

## Build

```
cmake -S . -B build-fuzz -DESHKOL_ENABLE_FUZZ=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-fuzz --target fuzz_parser
```

Only clang supports `-fsanitize=fuzzer`; gcc skips the target with a
warning.

## Run

```
mkdir -p build-fuzz/tests/fuzz/corpus-parser
./build-fuzz/tests/fuzz/fuzz_parser build-fuzz/tests/fuzz/corpus-parser
```

libFuzzer runs until it finds a crash or is killed. Seed the corpus
with known-good `.esk` snippets for faster coverage:

```
cp lib/core/list/*.esk build-fuzz/tests/fuzz/corpus-parser/
```

## Adding a new harness

1. Write `fuzz_XXX.cpp` in this directory with an
   `LLVMFuzzerTestOneInput` entry point.
2. Add an `add_executable(fuzz_XXX fuzz_XXX.cpp)` block in
   `CMakeLists.txt` mirroring `fuzz_parser`.
3. Exercise exactly one ingest point. Keep iteration count per input
   low enough that a 1-MB blob finishes in <100 ms.

Current harnesses:

| Harness | Surface | Status |
|---|---|---|
| `fuzz_parser` | `eshkol_parse_next_ast_from_stream` | landed |
| `fuzz_bitcode` | stdlib.bc loader | TODO |
| `fuzz_eskb` | `.eshkol-model` loader | TODO |
| `fuzz_json` | core.json reader | TODO |
