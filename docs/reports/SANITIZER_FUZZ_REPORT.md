# Sanitizer Fuzz Report

- Started: `2026-07-08T01:04:05Z`
- Finished: `2026-07-08T01:15:52Z`
- Build: `build-asan-ubsan`
- Corpus files: `150`
- Run records: `750`
- Axes: `-r`, `AOT -O0 compile/run`, `AOT -O2 compile/run`
- Generated differential fuzz: seed `42`, count `200`
- ASAN_OPTIONS: `detect_leaks=0:halt_on_error=1:abort_on_error=1:disable_coredump=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1`
- UBSAN_OPTIONS: `print_stacktrace=1:halt_on_error=1:abort_on_error=1:disable_coredump=1`
- Retained sanitizer artifacts: `artifacts/sanitizer-fuzz`
- ICC trace: `scripts/icc_traces/sanitizer_fuzz.jsonl`

## Teeth Check

- Status: `PASS`
- Summary: Synthetic unterminated numeric buffer repro emitted a sanitizer report rc=134

## Coverage

- Status: `PASS`
- Summary: -r runs=150, AOT -O0 runs=150, AOT -O2 runs=150
- Records by axis/phase/rc:
  - `aot-o0/compile rc=0`: `150`
  - `aot-o0/run rc=0`: `150`
  - `aot-o2/compile rc=0`: `150`
  - `aot-o2/run rc=0`: `150`
  - `r/run rc=0`: `150`

## Findings

No ASan/UBSan reports were observed in corpus logs.

