# Eshkol Fuzzing Harnesses

Two kinds of harness live here:

- **`reader_fuzz`** — a plain executable with a seeded PRNG that
  *generates* specific adversarial S-expression categories (long flat
  lists, over-cap nesting, dotted-pair edge cases, arena exhaustion,
  ...) and feeds them to the hosted reader (`eshkol_read_sexpr`) under
  fork+watchdog isolation. Deterministic and replayable by design — no
  libFuzzer, no coverage-guided mutation, any compiler. See
  `reader_fuzz_driver.cpp`'s header comment and
  `scripts/run_reader_fuzz.sh`.
- **libFuzzer drivers** (`fuzz_parser`, and TODOs below) — standalone
  libFuzzer binaries that exercise one parser/loader/validator in a
  tight loop under ASan+UBSan instrumentation, coverage-guided. Clang
  only.

## Build

```
cmake -S . -B build-fuzz -DESHKOL_ENABLE_FUZZ=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-fuzz --target reader_fuzz
```

`reader_fuzz` builds with any compiler (including AppleClang) and links
only `eshkol-runtime` (no LLVM-backed compiler objects), so this is
fast. The libFuzzer targets below need
`-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++` (Homebrew LLVM
on macOS — Xcode's AppleClang doesn't ship `libclang_rt.fuzzer`) and
link `eshkol-static`.

```
cmake --build build-fuzz --target fuzz_parser
```

Only clang supports `-fsanitize=fuzzer`; gcc/AppleClang skip that
target with a warning (`reader_fuzz` is unaffected).

For an ASan+UBSan pass over the reader itself (not just the driver),
configure with `-DESHKOL_FUZZ_ASAN=ON`:

```
cmake -S . -B build-fuzz-asan -DCMAKE_BUILD_TYPE=Debug \
    -DESHKOL_ENABLE_FUZZ=ON -DESHKOL_FUZZ_ASAN=ON \
    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build-fuzz-asan --target reader_fuzz
ASAN_OPTIONS=detect_leaks=0 ./build-fuzz-asan/tests/fuzz/reader_fuzz --full
```

## Run

```
scripts/run_reader_fuzz.sh              # smoke (~10s) — same probe CI runs
scripts/run_reader_fuzz.sh --full       # full seeded sweep (~10-30s):
                                         # 10^5-10^6-element flat lists,
                                         # over-cap nesting, dotted pairs,
                                         # embedded NULs, invalid UTF-8,
                                         # mixed vector/list nesting, quote
                                         # pathologies, arena exhaustion, ...
scripts/run_reader_fuzz.sh --regression # just the fixed depth-guard /
                                         # bounded-stack assertions
```

Or invoke the binary directly once built:

```
./build-fuzz/tests/fuzz/reader_fuzz --full --artifact-dir /tmp/reader-fuzz-artifacts
```

Every case runs in a forked child with `RLIMIT_CORE=0`, a bounded
`RLIMIT_AS`, and a `SIGALRM` watchdog, so one crashing/hanging case
doesn't take the rest of the sweep down with it — a single run reports
every distinct finding. Findings are deterministic and replayable: the
same fixed seed set always produces the same case sequence, and the
exact triggering input is saved as an artifact (see disk budget below).

Exit code is 0 iff no crash/hang was observed and the regression
assertions held.

### libFuzzer harnesses (fuzz_parser, ...)

```
mkdir -p build-fuzz/tests/fuzz/corpus-parser
./build-fuzz/tests/fuzz/fuzz_parser build-fuzz/tests/fuzz/corpus-parser
```

libFuzzer runs until it finds a crash or is killed. Seed the corpus
with known-good `.esk` snippets for faster coverage:

```
cp lib/core/list/*.esk build-fuzz/tests/fuzz/corpus-parser/
```

## Disk budget

Hard rule (repeat fuzz/sanitizer-harness disk incidents in this repo):
no unbounded corpus/artifact growth, ever.

- `reader_fuzz` hard-caps total artifact bytes at 64 MB
  (`kArtifactBudgetBytes` in `reader_fuzz_driver.cpp`, well under the
  project's 200 MB ceiling) and only ever writes the specific input
  that triggered a crash/hang — nothing is retained on a passing run.
- `scripts/run_reader_fuzz.sh` runs each pass in a fresh `mktemp -d`
  artifact directory outside the repo and `trap`s on `EXIT`: a clean
  run deletes the directory entirely; a run with findings keeps the
  (capped) directory and prints its path.
- `fuzz-artifacts/`, `tests/fuzz/fuzz-artifacts/`, and
  `tests/fuzz/corpus-*/` are all `.gitignore`d — never commit a corpus.
- libFuzzer corpus directories (`corpus-parser/` etc.) are unbounded by
  libFuzzer itself; prune them manually if a long local session grows
  one past a few tens of MB.

## Adding a new libFuzzer harness

1. Write `fuzz_XXX.cpp` in this directory with an
   `LLVMFuzzerTestOneInput` entry point.
2. Add an `add_executable(fuzz_XXX fuzz_XXX.cpp)` block in
   `CMakeLists.txt` mirroring `fuzz_parser`.
3. Exercise exactly one ingest point. Keep iteration count per input
   low enough that a 1-MB blob finishes in <100 ms.

Current harnesses:

| Harness | Surface | Status |
|---|---|---|
| `reader_fuzz` | `eshkol_read_sexpr` (hosted S-expression reader / `(read)`) | landed |
| `fuzz_parser` | `eshkol_parse_next_ast_from_stream` (frontend source parser) | landed |
| `fuzz_bitcode` | stdlib.bc loader | TODO |
| `fuzz_eskb` | `.eshkol-model` loader | TODO |
| `fuzz_json` | core.json reader | TODO |
