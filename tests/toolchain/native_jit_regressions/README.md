# Native in-process JIT regression probes

These four small programs distinguish in-process JIT behavior from native AOT
behavior. `ESHKOL_JIT_CACHE=0` is essential: ordinary `-r` may execute a cached
AOT binary. Every probe also uses `--no-stdlib`, so neither engine depends on an
installed or previously generated stdlib object cache.

Run from the repository root with a working compiler and its normal native-link
toolchain and DLL/shared-library dependencies on the environment search paths:

```console
python scripts/run_native_jit_regressions.py --runner path/to/eshkol-run --work-dir build/jit-probe-evidence
```

`--runner` defaults to `ESHKOL_RUN_OVERRIDE`, then `eshkol-run` on PATH. Repeat
`--case` to select cases; `--timeout` defaults to 30 seconds per subprocess.
The harness creates a new evidence directory per invocation and never cleans
an existing directory. It retains JSON containing the compiler SHA256,
`--version`/`--features`, architecture, fixture hashes, commands, timings, exact
stdout/stderr, and return codes. Known source, runner and evidence paths are
replaced with placeholders; review dependency diagnostics before publishing
because an external linker can mention additional machine paths.

Each case compiles AOT, runs the resulting executable, and then invokes the
in-process JIT. A regression requires a correct AOT control and an incorrect,
crashing, timed-out, or error-reporting JIT result. Compilation/launch/control
failures are recorded as **inconclusive**, not as proof of a JIT defect.
Output matching is exact apart from universal newline handling. A process
that prints `()` or an error and exits 0 cannot satisfy the expected transcript.

Exit codes are 0 for all passing cases, 1 for demonstrated JIT/AOT discrepancies,
and 2 for an incomplete control or launch. The runner continues after individual
failures so the final JSON contains every selected case. Dependency problems,
filesystem restrictions, or an incomplete build can make controls inconclusive;
resolve those before attributing a failure to the compiler.

The harness's own negative controls use real subprocesses:

```console
python tests/toolchain/test_native_jit_regressions.py
```

They reject wrong or extra stdout, error diagnostics with exit 0, a nonzero exit
after correct stdout, timeouts, and failed AOT controls. Launch failures remain
inconclusive. A separate test checks path sanitization.

## Measured baseline

Measured on 2026-10-25 against the published `v1.3.6-evolve` Windows x64 CUDA
package, corresponding to source tag `341ac812ed600fd9d27c28e9509d66ab71363947`.
The environment had LLVM 21 and CUDA 12.9 runtime/link dependencies available.
These measurements ran outside the review sandbox: sandbox-only early AOT
filesystem exceptions were separately ruled out and are not language findings.

| Case | Expected stdout | Native AOT | Forced in-process JIT |
|---|---|---|---|
| `forward_ad` | `12` | passes | passes |
| `callcc_escape` | `42` | passes | exits `0xC00000FF` |
| `guard_raise` | `42` | passes | exits `0xC0000005` |
| `parallel_map` | `(1 4 9 16 25)` | passes | prints `()`, reports missing workers, exits 0 |

For parallel-map, source inspection identifies a likely visibility mismatch:
`workerInitLinkage()` in `lib/backend/parallel_llvm_codegen.cpp` returns internal
linkage on Windows, while `ReplJITContext::addModule()` attempts an external
lookup of `__eshkol_init_parallel_workers`. The runtime's lazy `dlsym` fallback
is unavailable on Windows. A separate fresh-stdlib-cache reproduction observed
the same behavior. This is a diagnostic lead, not a validated compiler fix.
The root causes of the two control-flow crashes are not established here.

## Opt-in Windows CTest

Configure an existing supported Windows build with
`-DESHKOL_TEST_WINDOWS_JIT_REGRESSIONS=ON`, build `eshkol-run`, then run:

```console
ctest --test-dir build -C Release -R "^windows_jit_regressions" --output-on-failure
```

The option defaults to OFF because this is a regression-reproduction change,
and the recorded release baseline fails. Enabling it adds the probe and its
harness self-test to CTest; failures remain failures (`WILL_FAIL` is not used).
This change does not fix the compiler and does not claim that all Windows,
other architectures, or other Eshkol releases reproduce the findings. Once
the affected paths are repaired and verified, the gate can become mandatory.
