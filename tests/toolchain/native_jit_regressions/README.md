# Native in-process JIT regression probes

These seven programs compare in-process JIT behavior with native AOT behavior.
They cover forward AD, immediate continuation escape, guard/raise, parallel-map,
parallel continuation escape, repeated reentry after a capturing function has
returned, and restoration of captured guard handlers. Existing continuation
fixtures are reused from `tests/continuations` and `tests/vm_parity/corpus`.
`ESHKOL_JIT_CACHE=0` is essential: ordinary `-r` may execute a cached
AOT binary. Every probe also requests `--no-stdlib`. The fixtures use compiler
primitives and can run without installed stdlib artifacts; the retained stderr
records any artifact discovery or loading reported by the runner.

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

## Release baseline

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

## Windows x64 repairs

Each JIT module exposes its own uniquely named parallel-worker initializer.
The initializer retains internal linkage for Windows AOT, where combining
several object files must not create duplicate symbols. A fresh stdlib JIT
object exposes its initializer too, and the cache format is bumped to v5 so
previous objects with private initializers are rebuilt.

The Windows x64 JIT memory manager registers each generated `.pdata` unwind
table with the operating system and unregisters it before freeing the module.
Native `longjmp` can then walk JIT frames when handling an exception or a
nonlocal exit. The LLVM unit test checks a missing-registration negative
control, a real jump back into JIT code, and removal of the table on unload.

Resuming a saved continuation needs a separate operation: after copying the
saved stack image over the original frames, those overwritten frames cannot
be unwound again. A hosted Windows x64 hook uses the documented
[`RtlRestoreContext` `STATUS_LONGJUMP` path](https://learn.microsoft.com/en-us/windows/win32/api/winnt/nf-winnt-rtlrestorecontext)
at that point. Ordinary exception and escape-only `longjmp` paths remain in
place. The runtime core calls a platform hook and has no Windows dependency.
The direct helper unit checks native `setjmp` return values 37, 0 (normalized
to 1), and -9; language fixtures check restored frames, dynamic state, and
parallel escapes on both execution routes.

These changes also retain the Windows runtime export table in builds without
optional agent modules, and avoid registering VM-only tests when the VM target
is unavailable. The Windows CPU build exercises those configuration paths.

The targeted validation covers Windows x64 with LLVM 21. It does not establish
Windows ARM64 or other LLVM-version behavior. Hardware-enforced shadow-stack
and dynamic EH continuation-target policies were not specifically configured
or validated;
using the operating system's context API does not by itself establish CET
support for every JIT or continuation scenario. ICC was not available in the
review environment.

The repaired CPU build passed all seven JIT cases and their AOT controls on
2026-10-25. Separate follow-ups passed fresh and warm stdlib object loading,
worker redefinition in a persistent REPL with and without stdlib, and eleven
existing continuation, dynamic-wind, guard, and parallel fixtures on both
execution routes. The two fixtures that use `for-each` and `reverse` need the
stdlib module source tree available to AOT as well as the compiled stdlib
artifacts; an incomplete isolated library directory is not a valid control.

## Windows CTest

Configure an existing supported Windows build with
`-DESHKOL_TEST_WINDOWS_JIT_REGRESSIONS=ON`, build `eshkol-run`, then run:

```console
ctest --test-dir build -C Release -R "^windows_jit_regressions" --output-on-failure
```

The option defaults to ON for Windows x64 and OFF for other Windows
architectures. An explicit cached option continues to take precedence.
The probe and its harness self-test preserve failures (`WILL_FAIL` is not
used). A control failure still fails the gate even though it is classified
as inconclusive evidence about a JIT-specific defect. The LLVM initializer,
Windows unwind-table, and Windows context-restoration unit tests supplement
the language-level probe.

The separate `windows-x64-native-jit` completion oracle requires the two probe
CTest events and all three supporting unit-test events. It is intentionally
separate from platform-independent compiler readiness: Linux and Windows ARM64
builds do not register the Windows x64 unit targets. A full Windows CTest
evidence run through `scripts/run_ctest_gate.sh` produces those events.
