# Shared loop lease followup — 2026-11-13

Tested shared source `b682c40c4fbb4bb0944226ff2d49b0928c1fa785`, based on integration `53220efc1638c77d108ab009d4a5e6939c488c7b`.

Ten fresh scoped Release CTests passed, covering handle, net/TLS and explicit-loop resource behavior. Assertions remained enabled. The standalone shared contract also passed normally and with ASan, UBSan and leak detection.

The controls verify that an expired explicit wait adds no watch and performs no poll, preserves an existing foreign watch, and gives sticky cancellation precedence. Empty unbounded polling returns once the last managed owner finishes even while a terminal ticket remains held; new ownership on that loop remains wakeable. The current base's legacy readiness/deadline policy passes unchanged.

The shared contract retains pending/retryable owners until owner-thread cleanup completes. Terminal tickets retain result data without pinning the loop. Cleanup detachment is public, and cleanup stages use `ESHKOL_CLEANUP_STAGE_*` with compatibility aliases.

The [manifest](shared-loop-leases-2026-11-13.json) records tested source hashes and original-byte evidence hashes. Separate [hosted native qualification](https://github.com/Gabriel-Kahen/eshkol/actions/runs/43493801710) passed all three jobs at combined source `8d99676f6467e362da34122bbe7f36d7e6e63907`: Linux epoll, Linux ASan/UBSan and macOS arm64 kqueue. That combined-source evidence does not establish own-branch macOS net/TLS, public compiler/JIT/AOT, package or release acceptance.
