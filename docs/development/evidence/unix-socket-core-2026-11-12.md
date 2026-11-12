# Unix socket core: Linux execution record

The implementation is reviewable on `gabe/unix-socket-core`. It follows the explicit-loop, sticky-cancellation and shared deferred-close contract in the [design addendum](../../../notes/SOCKET_DESIGN_ADDENDUM-2026-11-10.md), with a concrete cleanup-ticket interface proposed for review on [#917](https://github.com/tsotchke/eshkol/pull/917#issuecomment-6829879987). The [#916 findings](../../../notes/SOCKET_DESIGN_REVIEW-2026-11-10.md) explain the earlier scaffold gaps.

The [machine-readable manifest](unix-socket-core-2026-11-12.json) binds results to source commit `b4c26cde661f6995f9e055c04f084d476d065a1d`, source/artifact hashes and retained raw logs. Later evidence-only commits do not change the tested code.

## Qualification

Linux x86_64 epoll, Clang 22.1.8, LLVM 21.1.8. The scoped `eshkol-run`/`stdlib` build and qualification targets completed. The selected CTest gate passed **65/65 tests**, including:

- All 16 native socket cases, all 15 public scenarios in both JIT and AOT, public lifecycle tests in both modes, explicit VM refusal, wrapper/runner controls, and native cleanup/contract tests.
- Shared explicit-loop resources, actual PROCESS/socket readiness and identity integration, PROCESS failed-watch-removal recovery, public subprocess JIT/AOT, handle lifecycle/C API, and selected legacy net/TLS tests.
- Separate full-instrumentation native socket, common-resource and PROCESS runs passed ASan/UBSan with leak detection enabled.
- Three actual JIT processes started simultaneously against a private cold stdlib cache. All executed the success marker, the published object passed `readelf -h`, and no temporary files remained.
- Generated API docs and public API coverage checks passed: 142/142 header symbols and 227/227 Eshkol exports.

The socket round trip checks request and reply bytes independently, including NULs. Large-payload acceptance transfers 16,777,253 bytes; the slow-peer fixture observes actual partial writes and would-block; the descriptor fixture measures 1,000 cycles. Linux backlog exhaustion and pending-connect deadlines/cancellation use real kernel behavior. Cleanup tests exercise an outstanding borrow, worker-thread close, pathname-removal failure/repair and descriptor reuse without repeated close.

Public execution uses the production compiler, runtime and agent archive. A process-visible test DSO supplies clock/fd/error observations and controlled test conditions; it does not replace the socket core. Darwin-specific pending-connect controls are not Linux/macOS execution evidence. The public byte oracle compares every byte because generic `equal?` currently rejects separately allocated equal bytevectors.

## Reproduction and artifacts

The manifest records the CTest command. Build options are Release, agent FFI/tests on, examples/GPU/BLAS/Python off, Clang and LLVM 21 selected explicitly. The extracted LLVM toolchain requires its `usr/lib` directory in `LD_LIBRARY_PATH` when building stdlib. Retained logs and sanitizer commands are under [the artifact directory](unix-socket-core-2026-11-12/); public receipts include exact runner commands.

The amended scaffold packet at `32274b1ec` was verified against all 14 source hashes, patch hash and its prerequisite tree before integration. Current integration base is `9c4a0c774` (`integ/v14-final-20261109`). [Implementation-only comparison](https://github.com/Gabriel-Kahen/eshkol/compare/13334a927...b4c26cde661f6995f9e055c04f084d476d065a1d) excludes the merged prerequisite stack.

## Limits and other observations

This establishes Linux qualification for this slice, not full release acceptance. macOS kqueue, Windows socket/readiness execution, portable VM execution, the full repository test suite and release/package acceptance were **NOT RUN**. The PR remains a draft pending maintainer review, including the cleanup-ticket API proposal.

The broader parallel all-target build failed while generating shared compiler-bootstrap outputs; its failure log is retained separately from the successful scoped build. The prose surface-count checker reports six mismatches in five unchanged documents (1,067 vs canonical 1,069); generated API checks pass. No claim is made that either broader check passed.

Small integration build repairs are separate commits: Taylor epoch pointer width, refinement `<algorithm>`, and resident runtime memory-context access. The public matrix also exposed concurrent writers sharing one stdlib cache temporary file; the fix publishes completed unique files atomically. PROCESS fault controls verify that failed watch removal retains descriptors, loop ownership and lifetime leases until recovery.
