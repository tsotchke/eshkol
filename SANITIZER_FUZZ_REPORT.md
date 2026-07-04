# Sanitizer Fuzz Report

- Started: `2026-07-04T02:51:12Z`
- Finished: `2026-07-04T04:44:04Z`
- Build: `build-asan-ubsan`
- Corpus files: `1363`
- Run records: `6572`
- Axes: `-r`, `AOT -O0 compile/run`, `AOT -O2 compile/run`
- Generated differential fuzz: seed `42`, count `200`
- ASAN_OPTIONS: `detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1`
- UBSAN_OPTIONS: `print_stacktrace=1:halt_on_error=1:abort_on_error=1`
- Raw logs: `artifacts/sanitizer-fuzz`
- ICC trace: `scripts/icc_traces/sanitizer_fuzz.jsonl`

## Teeth Check

- Status: `PASS`
- Summary: Synthetic unterminated numeric buffer repro emitted a sanitizer report rc=134
- Synthetic repro: `artifacts/sanitizer-fuzz/self-test/esh0116_unterminated_numeric_buffer.c`
- Synthetic stderr: `artifacts/sanitizer-fuzz/self-test/run.err`

## Coverage

- Status: `PASS`
- Summary: -r runs=1363, AOT -O0 runs=1243, AOT -O2 runs=1240
- Records by axis/phase/rc:
  - `aot-o0/compile rc=0`: `1243`
  - `aot-o0/compile rc=1`: `32`
  - `aot-o0/compile rc=124`: `1`
  - `aot-o0/compile rc=134`: `87`
  - `aot-o0/run rc=0`: `1217`
  - `aot-o0/run rc=1`: `13`
  - `aot-o0/run rc=124`: `1`
  - `aot-o0/run rc=132`: `5`
  - `aot-o0/run rc=134`: `5`
  - `aot-o0/run rc=138`: `1`
  - `aot-o0/run rc=139`: `1`
  - `aot-o2/compile rc=0`: `1240`
  - `aot-o2/compile rc=1`: `31`
  - `aot-o2/compile rc=124`: `5`
  - `aot-o2/compile rc=134`: `87`
  - `aot-o2/run rc=0`: `1217`
  - `aot-o2/run rc=1`: `13`
  - `aot-o2/run rc=124`: `1`
  - `aot-o2/run rc=132`: `2`
  - `aot-o2/run rc=134`: `6`
  - `aot-o2/run rc=139`: `1`
  - `r/run rc=0`: `1224`
  - `r/run rc=1`: `37`
  - `r/run rc=124`: `2`
  - `r/run rc=132`: `5`
  - `r/run rc=134`: `93`
  - `r/run rc=138`: `1`
  - `r/run rc=139`: `1`

## Findings

| Task | Kind | Type | Hits | Top frame | First repro |
|---|---|---|---:|---|---|
| `ESH-0160` | `AddressSanitizer` | `container-overflow` | 339 | `#0 0xADDR in ControlFlowCallbacks::registerFuncBindingWrapper(char const*, void*, void*) llvm_codegen.cpp:36354` | `tests/ad/sweep_c_regressions_test.esk` |
| `ESH-0161` | `UndefinedBehaviorSanitizer` | `negation of -9223372036854775808 cannot be represented in type 'int64_t' (aka 'long long'); cast to an unsigned type to negate this value to itself` | 6 | `#0 0xADDR in eshkol_bignum_fits_int64 bignum.cpp:641` | `tests/bignum/bignum_edge_cases_test.esk` |
| `ESH-0162` | `AddressSanitizer` | `heap-use-after-free` | 3 | `#0 0xADDR in eshkol_deep_equal runtime_deep_equal.cpp:68` | `tests/edge_matrix/generated/pair243_quote__regions.esk` |
| `ESH-0163` | `AddressSanitizer` | `heap-buffer-overflow` | 6 | `#0 0xADDR in display_tensor_recursive(__sFILE*, eshkol_tensor const*, unsigned long long, unsigned long long) runtime_display_hosted.cpp:672` | `tests/stdlib/v12_consciousness_test.esk` |
| `ESH-0164` | `AddressSanitizer` | `heap-buffer-overflow` | 8 | `#0 0xADDR in memcpy+0xADDR (libclang_rt.asan_osx_dynamic.dylib:arm64e+0xADDR)` | `tests/stress/found/string_nul_long_literal.esk` |
| `ESH-0165` | `AddressSanitizer` | `stack-buffer-overflow` | 1 | `#0 0xADDR in eshkol_deep_equal runtime_deep_equal.cpp:27` | `tests/v1_2_edge_cases/dotted_pair_reader_test.esk` |
| `ESH-0166` | `AddressSanitizer` | `stack-buffer-overflow` | 2 | `#0 0xADDR in eshkol_type_error_with_operand runtime_errors_hosted.cpp:201` | `tests/v1_2_edge_cases/type_safety_test.esk` |
| `ESH-0167` | `AddressSanitizer` | `heap-buffer-overflow` | 1 | `#0 0xADDR in eshkol_display_value_opts runtime_display_hosted.cpp:219` | `tests/vm/vm_system_test.esk` |

### ESH-0160

- Kind: `AddressSanitizer`
- Type: `container-overflow`
- Hits: `339`
- Top frame: `#0 0xADDR in ControlFlowCallbacks::registerFuncBindingWrapper(char const*, void*, void*) llvm_codegen.cpp:36354`
- Summary: `SUMMARY: AddressSanitizer: container-overflow llvm_codegen.cpp:36354 in ControlFlowCallbacks::registerFuncBindingWrapper(char const*, void*, void*)`
- First repro: `tests/ad/sweep_c_regressions_test.esk` via `r/run` rc `134`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -r tests/ad/sweep_c_regressions_test.esk`
- First log: `artifacts/sanitizer-fuzz/logs/206_r_211912727.err`
- Additional samples:
  - `tests/ad/sweep_c_regressions_test.esk` via `r/run`, log `artifacts/sanitizer-fuzz/logs/206_r_211912727.err`
  - `tests/ad/sweep_c_regressions_test.esk` via `aot-o0/compile`, log `artifacts/sanitizer-fuzz/logs/206_aot-o0_211912727.compile.err`
  - `tests/ad/sweep_c_regressions_test.esk` via `aot-o2/compile`, log `artifacts/sanitizer-fuzz/logs/206_aot-o2_211912727.compile.err`
  - `tests/ad_oracle/generated/ad_oracle_deriv_01.esk` via `r/run`, log `artifacts/sanitizer-fuzz/logs/215_r_579100824.err`

```text
==25845==ERROR: AddressSanitizer: container-overflow on address 0x62100015f500 at pc 0x00010043007c bp 0x00016fc79c30 sp 0x00016fc79c28
READ of size 8 at 0x62100015f500 thread T0
    #0 0x100430078 in ControlFlowCallbacks::registerFuncBindingWrapper(char const*, void*, void*) llvm_codegen.cpp:36354
    #1 0x100306628 in eshkol::BindingCodegen::define(eshkol_operation const*) binding_codegen.cpp:373
    #2 0x1004df4a0 in EshkolLLVMCodeGen::codegenDefine(eshkol_ast const*) llvm_codegen.cpp:10419
    #3 0x100417f38 in EshkolLLVMCodeGen::codegenAST(eshkol_ast const*) llvm_codegen.cpp:8893
    #4 0x1003c2918 in EshkolLLVMCodeGen::generateIR(eshkol_ast const*, unsigned long) llvm_codegen.cpp:2758
    #5 0x1003bbd20 in eshkol_generate_llvm_ir llvm_codegen.cpp:36747
    #6 0x100e04734 in main eshkol-run.cpp:4244
    #7 0x193b88270  (<unknown module>)

0x62100015f500 is located 0 bytes inside of 4048-byte region [0x62100015f500,0x6210001604d0)
allocated by thread T0 here:
    #0 0x103e5fe94 in _Znwm+0x74 (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x63e94)
    #1 0x10043914c in std::__1::deque<TypedValue, std::__1::allocator<TypedValue>>::__add_back_capacity() deque:2163
    #2 0x10043327c in std::__1::deque<TypedValue, std::__1::allocator<TypedValue>>::push_back(TypedValue&&) deque:1567
    #3 0x10042c464 in ControlFlowCallbacks::codegenTypedASTWrapper(void const*, void*) llvm_codegen.cpp:36301
    #4 0x10072aadc in eshkol::StringIOCodegen::display(eshkol_operation const*) string_io_codegen.cpp:1859
    #5 0x10052c910 in EshkolLLVMCodeGen::codegenCall(eshkol_operation const*) llvm_codegen.cpp:13951
    #6 0x1004e05b8 in EshkolLLVMCodeGen::codegenOperation(eshkol_operation const*) llvm_codegen.cpp:10098
    #7 0x1004181d0 in EshkolLLVMCodeGen::codegenAST(eshkol_ast const*) llvm_codegen.cpp:8895
    #8 0x100570124 in EshkolLLVMCodeGen::codegenSequence(eshkol_operation const*) llvm_codegen.cpp:23084
    #9 0x1004e0574 in EshkolLLVMCodeGen::codegenOperation(eshkol_operation const*) llvm_codegen.cpp:10101
    #10 0x1004181d0 in EshkolLLVMCodeGen::codegenAST(eshkol_ast const*) llvm_codegen.cpp:8895
    #11 0x1004f0044 in EshkolLLVMCodeGen::codegenFunctionDefinition(eshkol_ast const*) llvm_codegen.cpp:10612
    #12 0x1004df150 in EshkolLLVMCodeGen::codegenDefine(eshkol_ast const*) llvm_codegen.cpp:10392
    #13 0x100417f38 in EshkolLLVMCodeGen::codegenAST(eshkol_ast const*) llvm_codegen.cpp:8893
    #14 0x1003c1e94 in EshkolLLVMCodeGen::generateIR(eshkol_ast const*, unsigned long) llvm_codegen.cpp:2702
    #15 0x1003bbd20 in eshkol_generate_llvm_ir llvm_codegen.cpp:36747
    #16 0x100e04734 in main eshkol-run.cpp:4244
    #17 0x193b88270  (<unknown module>)

HINT: if you don't care about these errors you may set ASAN_OPTIONS=detect_container_overflow=0.
If you suspect a false positive see also: https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow.
SUMMARY: AddressSanitizer: container-overflow llvm_codegen.cpp:36354 in ControlFlowCallbacks::registerFuncBindingWrapper(char const*, void*, void*)
Shadow bytes around the buggy address:
```

### ESH-0161

- Kind: `UndefinedBehaviorSanitizer`
- Type: `negation of -9223372036854775808 cannot be represented in type 'int64_t' (aka 'long long'); cast to an unsigned type to negate this value to itself`
- Hits: `6`
- Top frame: `#0 0xADDR in eshkol_bignum_fits_int64 bignum.cpp:641`
- Summary: `SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior ./lib/core/bignum.cpp:641:29 in`
- First repro: `tests/bignum/bignum_edge_cases_test.esk` via `r/run` rc `134`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -r tests/bignum/bignum_edge_cases_test.esk`
- First log: `artifacts/sanitizer-fuzz/logs/298_r_2816962780.err`
- Additional samples:
  - `tests/bignum/bignum_edge_cases_test.esk` via `aot-o0/run`, log `artifacts/sanitizer-fuzz/logs/298_aot-o0_2816962780.run.err`
  - `tests/bignum/bignum_edge_cases_test.esk` via `aot-o2/run`, log `artifacts/sanitizer-fuzz/logs/298_aot-o2_2816962780.run.err`
  - `tests/numeric/critical_regression_test.esk` via `r/run`, log `artifacts/sanitizer-fuzz/logs/930_r_2652083108.err`
  - `tests/numeric/critical_regression_test.esk` via `aot-o0/run`, log `artifacts/sanitizer-fuzz/logs/930_aot-o0_2652083108.run.err`

```text
./lib/core/bignum.cpp:641:29: runtime error: negation of -9223372036854775808 cannot be represented in type 'int64_t' (aka 'long long'); cast to an unsigned type to negate this value to itself
    #0 0x103e3e0b4 in eshkol_bignum_fits_int64 bignum.cpp:641
    #1 0x103e3f204 in eshkol_bignum_binary_tagged bignum.cpp:803
    #2 0x103e282f8 in main+0xac8 (run-42814376ec591ac31d83e7998a18df607eb164384982eb2ae6aafc5bb9f859a7:arm64+0x10159c2f8)
    #3 0x193b88270  (<unknown module>)

SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior ./lib/core/bignum.cpp:641:29 in
```

### ESH-0162

- Kind: `AddressSanitizer`
- Type: `heap-use-after-free`
- Hits: `3`
- Top frame: `#0 0xADDR in eshkol_deep_equal runtime_deep_equal.cpp:68`
- Summary: `SUMMARY: AddressSanitizer: heap-use-after-free runtime_deep_equal.cpp:68 in eshkol_deep_equal`
- First repro: `tests/edge_matrix/generated/pair243_quote__regions.esk` via `r/run` rc `134`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -r tests/edge_matrix/generated/pair243_quote__regions.esk`
- First log: `artifacts/sanitizer-fuzz/logs/614_r_2370875467.err`
- Additional samples:
  - `tests/edge_matrix/generated/pair243_quote__regions.esk` via `aot-o0/run`, log `artifacts/sanitizer-fuzz/logs/614_aot-o0_2370875467.run.err`
  - `tests/edge_matrix/generated/pair243_quote__regions.esk` via `aot-o2/run`, log `artifacts/sanitizer-fuzz/logs/614_aot-o2_2370875467.run.err`

```text
==8477==ERROR: AddressSanitizer: heap-use-after-free on address 0x625000002908 at pc 0x000102359efc bp 0x00016f056190 sp 0x00016f056188
READ of size 1 at 0x625000002908 thread T0
    #0 0x102359ef8 in eshkol_deep_equal runtime_deep_equal.cpp:68
    #1 0x10234383c in edge-chk+0x9c (run-c3957f9868169cf49823d9acbb1d214fa26c7763af4e1a5247e3695739d0331c:arm64+0x10159b83c)
    #2 0x102343bf4 in edge-check-g2134+0x150 (run-c3957f9868169cf49823d9acbb1d214fa26c7763af4e1a5247e3695739d0331c:arm64+0x10159bbf4)

0x625000002908 is located 8 bytes inside of 8192-byte region [0x625000002900,0x625000004900)
freed by thread T0 here:
    #0 0x102ec8d40 in free+0x98 (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x54d40)
    #1 0x102354400 in arena_destroy runtime_arena_core.cpp:174
    #2 0x102361988 in region_destroy runtime_regions.cpp:200
    #3 0x102343b74 in edge-check-g2134+0xd0 (run-c3957f9868169cf49823d9acbb1d214fa26c7763af4e1a5247e3695739d0331c:arm64+0x10159bb74)

previously allocated by thread T0 here:
    #0 0x102ec8c04 in malloc+0x94 (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x54c04)
    #1 0x102354020 in create_arena_block(unsigned long) runtime_arena_core.cpp:60
    #2 0x102353df0 in arena_create runtime_arena_core.cpp:94
    #3 0x102361610 in region_create runtime_regions.cpp:155
    #4 0x102343ac0 in edge-check-g2134+0x1c (run-c3957f9868169cf49823d9acbb1d214fa26c7763af4e1a5247e3695739d0331c:arm64+0x10159bac0)

SUMMARY: AddressSanitizer: heap-use-after-free runtime_deep_equal.cpp:68 in eshkol_deep_equal
Shadow bytes around the buggy address:
  0x625000002680: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x625000002700: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x625000002780: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x625000002800: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x625000002880: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
=>0x625000002900: fd[fd]fd fd fd fd fd fd fd fd fd fd fd fd fd fd
  0x625000002980: fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd
  0x625000002a00: fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd
  0x625000002a80: fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd
  0x625000002b00: fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd
  0x625000002b80: fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd fd
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
```

### ESH-0163

- Kind: `AddressSanitizer`
- Type: `heap-buffer-overflow`
- Hits: `6`
- Top frame: `#0 0xADDR in display_tensor_recursive(__sFILE*, eshkol_tensor const*, unsigned long long, unsigned long long) runtime_display_hosted.cpp:672`
- Summary: `SUMMARY: AddressSanitizer: heap-buffer-overflow runtime_display_hosted.cpp:672 in display_tensor_recursive(__sFILE*, eshkol_tensor const*, unsigned long long, unsigned long long)`
- First repro: `tests/stdlib/v12_consciousness_test.esk` via `r/run` rc `134`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -r tests/stdlib/v12_consciousness_test.esk`
- First log: `artifacts/sanitizer-fuzz/logs/1057_r_1980305526.err`
- Additional samples:
  - `tests/stdlib/v12_consciousness_test.esk` via `aot-o0/run`, log `artifacts/sanitizer-fuzz/logs/1057_aot-o0_1980305526.run.err`
  - `tests/stdlib/v12_consciousness_test.esk` via `aot-o2/run`, log `artifacts/sanitizer-fuzz/logs/1057_aot-o2_1980305526.run.err`
  - `tests/vm/vm_kb_tensor_test.esk` via `r/run`, log `artifacts/sanitizer-fuzz/logs/1288_r_1592037548.err`
  - `tests/vm/vm_kb_tensor_test.esk` via `aot-o0/run`, log `artifacts/sanitizer-fuzz/logs/1288_aot-o0_1592037548.run.err`

```text
==67725==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x631000114800 at pc 0x0001062b64b8 bp 0x00016b131fd0 sp 0x00016b131fc8
READ of size 8 at 0x631000114800 thread T0
    #0 0x1062b64b4 in display_tensor_recursive(__sFILE*, eshkol_tensor const*, unsigned long long, unsigned long long) runtime_display_hosted.cpp:672
    #1 0x1062b4008 in eshkol_display_value_opts runtime_display_hosted.cpp
    #2 0x1062b37fc in eshkol_display_value runtime_display_hosted.cpp:161
    #3 0x106265520 in main+0xbd8 (run-518e0514618815b085ab030890c94b4f3e9eaeadc35281a8489752284b77fe4c:arm64+0x101599520)
    #4 0x193b88270  (<unknown module>)

0x631000114800 is located 0 bytes after 65536-byte region [0x631000104800,0x631000114800)
allocated by thread T0 here:
    #0 0x107124c04 in malloc+0x94 (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x54c04)
    #1 0x106275e0c in create_arena_block(unsigned long) runtime_arena_core.cpp:60
    #2 0x10627670c in arena_allocate_aligned runtime_arena_core.cpp:235
    #3 0x106265194 in main+0x84c (run-518e0514618815b085ab030890c94b4f3e9eaeadc35281a8489752284b77fe4c:arm64+0x101599194)
    #4 0x193b88270  (<unknown module>)

SUMMARY: AddressSanitizer: heap-buffer-overflow runtime_display_hosted.cpp:672 in display_tensor_recursive(__sFILE*, eshkol_tensor const*, unsigned long long, unsigned long long)
Shadow bytes around the buggy address:
  0x631000114580: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000114600: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000114680: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000114700: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000114780: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x631000114800:[fa]fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000114880: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000114900: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000114980: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000114a00: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000114a80: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
```

### ESH-0164

- Kind: `AddressSanitizer`
- Type: `heap-buffer-overflow`
- Hits: `8`
- Top frame: `#0 0xADDR in memcpy+0xADDR (libclang_rt.asan_osx_dynamic.dylib:arm64e+0xADDR)`
- Summary: `SUMMARY: AddressSanitizer: heap-buffer-overflow (libclang_rt.asan_osx_dynamic.dylib:arm64e+0xADDR) in memcpy+0xADDR`
- First repro: `tests/stress/found/string_nul_long_literal.esk` via `r/run` rc `134`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -r tests/stress/found/string_nul_long_literal.esk`
- First log: `artifacts/sanitizer-fuzz/logs/1080_r_3733549987.err`
- Additional samples:
  - `tests/stress/found/string_nul_long_literal.esk` via `r/run`, log `artifacts/sanitizer-fuzz/logs/1080_r_3733549987.err`
  - `tests/stress/found/string_nul_long_literal.esk` via `aot-o0/compile`, log `artifacts/sanitizer-fuzz/logs/1080_aot-o0_3733549987.compile.err`
  - `tests/stress/found/string_nul_long_literal.esk` via `aot-o2/compile`, log `artifacts/sanitizer-fuzz/logs/1080_aot-o2_3733549987.compile.err`
  - `tests/stress/parser_string_5k_escapes.esk` via `r/run`, log `artifacts/sanitizer-fuzz/logs/1091_r_2046740991.err`

```text
==70165==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000e88d1 at pc 0x000103eeeda8 bp 0x00016fbd8b80 sp 0x00016fbd8330
READ of size 300 at 0x6020000e88d1 thread T0
    #0 0x103eeeda4 in memcpy+0x3fc (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x52da4)
    #1 0x193e39404 in std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init(char const*, unsigned long)+0x7c (libc++.1.dylib:arm64e+0x19404)
    #2 0x1004b7e70 in EshkolLLVMCodeGen::codegenAST(eshkol_ast const*) llvm_codegen.cpp:8883
    #3 0x1004d6ee8 in EshkolLLVMCodeGen::codegenTypedAST(eshkol_ast const*) llvm_codegen.cpp:6028
    #4 0x1004cc458 in ControlFlowCallbacks::codegenTypedASTWrapper(void const*, void*) llvm_codegen.cpp:36301
    #5 0x1003a6468 in eshkol::BindingCodegen::define(eshkol_operation const*) binding_codegen.cpp:336
    #6 0x10057f4a0 in EshkolLLVMCodeGen::codegenDefine(eshkol_ast const*) llvm_codegen.cpp:10419
    #7 0x1004b7f38 in EshkolLLVMCodeGen::codegenAST(eshkol_ast const*) llvm_codegen.cpp:8893
    #8 0x100462918 in EshkolLLVMCodeGen::generateIR(eshkol_ast const*, unsigned long) llvm_codegen.cpp:2758
    #9 0x10045bd20 in eshkol_generate_llvm_ir llvm_codegen.cpp:36747
    #10 0x100ea4734 in main eshkol-run.cpp:4244
    #11 0x193b88270  (<unknown module>)

0x6020000e88d1 is located 0 bytes after 1-byte region [0x6020000e88d0,0x6020000e88d1)
allocated by thread T0 here:
    #0 0x103ee9cb8 in strdup+0x11c (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x4dcb8)
    #1 0x100ad3ea4 in eshkol::MacroExpander::copyAst(eshkol_ast const&) macro_expander.cpp:1194
    #2 0x100acffac in eshkol::MacroExpander::expandNode(eshkol_ast const&) macro_expander.cpp:147
    #3 0x100ad0f1c in eshkol::MacroExpander::expandNode(eshkol_ast const&) macro_expander.cpp:183
    #4 0x100acf5dc in eshkol::MacroExpander::expandAll(std::__1::vector<eshkol_ast, std::__1::allocator<eshkol_ast>> const&) macro_expander.cpp:74
    #5 0x10045c9a0 in EshkolLLVMCodeGen::generateIR(eshkol_ast const*, unsigned long) llvm_codegen.cpp:2130
    #6 0x10045bd20 in eshkol_generate_llvm_ir llvm_codegen.cpp:36747
    #7 0x100ea4734 in main eshkol-run.cpp:4244
    #8 0x193b88270  (<unknown module>)

SUMMARY: AddressSanitizer: heap-buffer-overflow (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x52da4) in memcpy+0x3fc
Shadow bytes around the buggy address:
  0x6020000e8600: fa fa 00 00 fa fa 00 00 fa fa 06 fa fa fa 02 fa
  0x6020000e8680: fa fa 00 00 fa fa 06 fa fa fa 02 fa fa fa 00 fa
  0x6020000e8700: fa fa 06 fa fa fa 00 00 fa fa 02 fa fa fa 02 fa
  0x6020000e8780: fa fa 00 03 fa fa 06 fa fa fa 05 fa fa fa 02 fa
  0x6020000e8800: fa fa 02 fa fa fa 02 fa fa fa 02 fa fa fa 00 00
=>0x6020000e8880: fa fa 06 fa fa fa 00 03 fa fa[01]fa fa fa 00 fa
  0x6020000e8900: fa fa 04 fa fa fa 00 fa fa fa 00 06 fa fa 05 fa
```

### ESH-0165

- Kind: `AddressSanitizer`
- Type: `stack-buffer-overflow`
- Hits: `1`
- Top frame: `#0 0xADDR in eshkol_deep_equal runtime_deep_equal.cpp:27`
- Summary: `SUMMARY: AddressSanitizer: stack-buffer-overflow runtime_deep_equal.cpp:27 in eshkol_deep_equal`
- First repro: `tests/v1_2_edge_cases/dotted_pair_reader_test.esk` via `aot-o2/run` rc `134`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -O2 tests/v1_2_edge_cases/dotted_pair_reader_test.esk -o /tmp/eshkol-sanitize-repro.bin`
- First log: `artifacts/sanitizer-fuzz/logs/1171_aot-o2_3462549093.run.err`
- Additional samples:

```text
==91318==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x00016ceea6e8 at pc 0x0001044cd6c4 bp 0x00016ceea530 sp 0x00016ceea528
READ of size 1 at 0x00016ceea6e8 thread T0
    #0 0x1044cd6c0 in eshkol_deep_equal runtime_deep_equal.cpp:27
    #1 0x1044aea74 in check+0xb0 (1171_aot-o2_3462549093.bin:arm64+0x10159aa74)
    #2 0x1044b7f2c in main+0x630c (1171_aot-o2_3462549093.bin:arm64+0x1015a3f2c)
    #3 0x193b88270  (<unknown module>)

Address 0x00016ceea6e8 is located in stack of thread T0 at offset 424 in frame
    #0 0x1044cc514 in eshkol_deep_equal runtime_deep_equal.cpp:18

  This frame has 4 object(s):
    [32, 48) 'car1' (line 76)
    [64, 80) 'car2' (line 77)
    [96, 112) 'cdr1' (line 80)
    [128, 144) 'cdr2' (line 81) <== Memory access at offset 424 overflows this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-buffer-overflow runtime_deep_equal.cpp:27 in eshkol_deep_equal
Shadow bytes around the buggy address:
  0x00016ceea400: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x00016ceea480: 00 00 00 00 f1 f1 f1 f1 00 00 00 00 00 00 00 00
  0x00016ceea500: 00 00 00 00 00 00 00 00 f1 f1 f1 f1 f8 f8 f2 f2
  0x00016ceea580: f8 f8 f2 f2 f8 f8 f2 f2 f8 f8 f3 f3 00 00 00 00
  0x00016ceea600: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x00016ceea680: 00 00 00 00 00 00 00 00 f2 f2 f2 f2 f2[f2]f2 f2
  0x00016ceea700: 00 f3 f3 f3 00 00 00 00 00 00 00 00 00 00 00 00
  0x00016ceea780: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x00016ceea800: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x00016ceea880: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x00016ceea900: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
```

### ESH-0166

- Kind: `AddressSanitizer`
- Type: `stack-buffer-overflow`
- Hits: `2`
- Top frame: `#0 0xADDR in eshkol_type_error_with_operand runtime_errors_hosted.cpp:201`
- First repro: `tests/v1_2_edge_cases/type_safety_test.esk` via `r/run` rc `138`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -r tests/v1_2_edge_cases/type_safety_test.esk`
- First log: `artifacts/sanitizer-fuzz/logs/1227_r_2911697867.err`
- Additional samples:
  - `tests/v1_2_edge_cases/type_safety_test.esk` via `aot-o0/run`, log `artifacts/sanitizer-fuzz/logs/1227_aot-o0_2911697867.run.err`

```text
==98752==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x00016b7f1530 at pc 0x000105bfcb68 bp 0x00016b7f14f0 sp 0x00016b7f14e8
READ of size 8 at 0x00016b7f1530 thread T0
    #0 0x105bfcb64 in eshkol_type_error_with_operand runtime_errors_hosted.cpp:201
    #1 0x105baaa80 in main+0x1d9c (run-110d8c53cf0ac460d0647a2053c1e60d968faf93581af056599981dd1f9acc66:arm64+0x10159ea80)
    #2 0x193b88270  (<unknown module>)

Address 0x00016b7f1530 is located in stack of thread T0 at offset 400 in frame
    #0 0x100000000d  (<unknown module>)


[Eshkol] fatal signal: SIGBUS (bus error) — terminating; output above is what made it to stdout before the crash
```

### ESH-0167

- Kind: `AddressSanitizer`
- Type: `heap-buffer-overflow`
- Hits: `1`
- Top frame: `#0 0xADDR in eshkol_display_value_opts runtime_display_hosted.cpp:219`
- Summary: `SUMMARY: AddressSanitizer: heap-buffer-overflow runtime_display_hosted.cpp:219 in eshkol_display_value_opts`
- First repro: `tests/vm/vm_system_test.esk` via `r/run` rc `134`
- Repro command: `ASAN_OPTIONS='detect_leaks=0:halt_on_error=1:abort_on_error=1:print_stacktrace=1:strict_string_checks=1:detect_stack_use_after_return=0:symbolize=1' UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:abort_on_error=1' build-asan-ubsan/eshkol-run -r tests/vm/vm_system_test.esk`
- First log: `artifacts/sanitizer-fuzz/logs/1289_r_50244341.err`
- Additional samples:

```text
==12238==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x631000ce47f8 at pc 0x000104a2da4c bp 0x00016bdf52d0 sp 0x00016bdf52c8
READ of size 1 at 0x631000ce47f8 thread T0
    #0 0x104a2da48 in eshkol_display_value_opts runtime_display_hosted.cpp:219
    #1 0x104a2cc24 in eshkol_display_value runtime_display_hosted.cpp:161
    #2 0x1079f43e4  (<unknown module>)
    #3 0x104c31ccc in eshkol::ReplJITContext::executeBatch(std::__1::vector<eshkol_ast, std::__1::allocator<eshkol_ast>>&, bool) repl_jit.cpp:2977
    #4 0x104c8abc0 in main eshkol-run.cpp:3825
    #5 0x193b88270  (<unknown module>)

0x631000ce47f8 is located 8 bytes before 65536-byte region [0x631000ce4800,0x631000cf4800)
allocated by thread T0 here:
    #0 0x107cd4c04 in malloc+0x94 (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x54c04)
    #1 0x1049da21c in create_arena_block(unsigned long) runtime_arena_core.cpp:60
    #2 0x1049dab1c in arena_allocate_aligned runtime_arena_core.cpp:235
    #3 0x1079f4330  (<unknown module>)
    #4 0x104c31ccc in eshkol::ReplJITContext::executeBatch(std::__1::vector<eshkol_ast, std::__1::allocator<eshkol_ast>>&, bool) repl_jit.cpp:2977
    #5 0x104c8abc0 in main eshkol-run.cpp:3825
    #6 0x193b88270  (<unknown module>)

SUMMARY: AddressSanitizer: heap-buffer-overflow runtime_display_hosted.cpp:219 in eshkol_display_value_opts
Shadow bytes around the buggy address:
  0x631000ce4500: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000ce4580: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000ce4600: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000ce4680: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x631000ce4700: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
=>0x631000ce4780: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa[fa]
  0x631000ce4800: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000ce4880: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000ce4900: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000ce4980: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x631000ce4a00: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
```
