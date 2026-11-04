# Subprocess streaming local checkpoint — 2026-11-04

Builds on #845. Base: `51a93586951b1c75b413b14531d0465c660ae579`,
the assigned `codex/v14-integration-wave9-20261021` integration branch.
Local implementation branch: `codex/subprocess-streaming-20261104`.

## Implemented scope

The existing argv launcher now supports scoped, output-only POSIX streaming via
`call-with-subprocess-streams`. One common-registry PROCESS lease covers both
outputs and retains the existing event loop until detachment. Bytevector reads
distinguish data, would-block, EOF, and error, preserving embedded NULs and
applying pipe backpressure instead of an output capture limit.

Polling uses monotonic deadlines, including a nonblocking zero-timeout pass.
Child status is independent of pipe EOF. Direct-child cancellation escalates
from TERM to KILL. Bounded teardown retains a discoverable, budgeted cleanup
record until confirmed reap, even when the original loop closes or capabilities
are revoked. Clock-read errors preserve closed ownership and cancellation intent.

The Scheme scope uses one captured state vector and `dynamic-wind`; normal
return, exceptions, resumable exceptions, continuation escape, and closed-scope
re-entry have separate public checks. Legacy spawn-options rejects event-loop
mode and directs callers to the scoped helper.

## Validation

Linux epoll; Clang 22.1.8 compiling against LLVM 21.1.8; object ABI v1.

| Check | Result |
|---|---|
| Native lifecycle/stream suite | 10 scenarios pass |
| Native ASan + UBSan | Same scenarios pass; no sanitizer findings |
| Native capability checker absent | Acquisition denied |
| Injected monotonic clock failure | Closed owner remains discoverable; later drain/reap/consume succeeds |
| Public Scheme API | 23 checks in each of JIT and AOT |
| Slow native consumer | 17,825,829 bytes per stream, every byte checked; maximum stream skew 4,096 bytes |
| Public large-output fixture | 17,825,829 bytes per stream in each of JIT and AOT; exit status 19 |
| Continuation re-entry | Intended exception and exit status 1 in JIT and AOT |
| Standalone VM stream bridge | Explicit unsupported-extern rejection; no streaming execution claim |
| Existing event-loop suite | 27 checks in each of JIT, AOT, and standalone VM |
| Existing subprocess shell/argv suite | JIT and AOT pass |
| Common handle lifecycle and C API | Both pass |

Native commands, without the compiler dependency aggregate:

```sh
tests/agent/run_subprocess_stream_test.sh
CC=clang CXX=clang++ ESHKOL_STREAM_SANITIZE=1 tests/agent/run_subprocess_stream_test.sh
```

The runner includes missing-checker and Linux linker-wrapped clock controls.
For a build with the production agent FFI enabled:

```sh
cmake --build ../build --target eshkol-run eshkol-subprocess-stream-test
python3 scripts/test_subprocess_stream_public.py \
  --runner ../build/eshkol-run --build-dir ../build \
  --child ../build/eshkol-subprocess-stream-test
```

The local engine checks used an **agent-FFI-off build with a test-only subprocess
archive and support DSO**, containing the actual launcher and bytevector shim.
They resolve the real Eshkol registry, loop, and capability checker; they do not
substitute fake runtime behavior. This validates the public streaming path but
does not qualify the complete packaged agent dependency bundle. The local run
adds `--support-library ../checks/libstream_test_support.so` to the command above.
The [validation record](evidence/subprocess-streaming-2026-11-04.json) contains
source/artifact hashes, refresh commands, and measured results. Its paths are
relative to the source root; `../build` denotes a separate test build configured
with `ESHKOL_BUILD_AGENT_FFI=OFF` and LLVM 21.1.8. Run `mkdir -p ../checks` before
the support compilation commands. The minimal archive is test-only and belongs
in that separate build directory. Running the public test command with
`--evidence-dir ../checks/public-final` retains its execution logs.

## Remaining integration

General resource-event polling, native strict-resource checker contracts for
these scopes, nonblocking stdin, process groups, Windows streaming, and a VM
adapter remain outside this first slice. Streaming loops currently admit managed
subprocess watches only. The runtime is confined to one owner thread and its
explicit shutdown is terminal. LLVM 22 compatibility was not changed.

## Darwin feature-profile follow-up — 2026-11-04

Review identified that `_POSIX_C_SOURCE=200809L` hides Darwin's `ru_maxrss`
field in the new native test. The test now enables `_DARWIN_C_SOURCE` before
headers. The standalone runner also enables Darwin extensions for all of its C
translation units because its `_XOPEN_SOURCE=700` profile otherwise hides native
resource-limit fields as well.

The focused `Subprocess streams native` workflow runs Linux epoll, Linux
ASan/UBSan, and macOS kqueue checks, including the reviewer's exact macOS compile
command. It targets the assigned integration branch as well as master. Hosted
results are recorded by that workflow; the Linux evidence above does not imply
a macOS runtime pass or full packaged-agent validation.

Public API details are in [the subprocess reference](../reference/agent/subprocess.md).
