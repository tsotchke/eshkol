# Shared explicit-loop lease foundation

This branch adds loop-aware adoption, sticky cancellation, accepted deferred close and separately reserved cleanup results while preserving legacy net/TLS behavior. EVENT_LOOP, SOCKET, PROCESS and CLEANUP kinds use the common exact registry; platform kind values remain stable. Resource adapters are separate from this foundation.

Legacy I/O reads immutable registry metadata through its existing lookup and acquires no managed-list lock. Managed owners use a full-ID hash index, with a pending-only owner queue and conditional pumping on release. A test keeps 128 failed results on one loop, reuses retired slots for new owners, and recovers all original results independently. Serialized callbacks therefore do not eliminate concurrent outstanding failure ownership.

Watch removal retains its slot on a genuine backend error; callers repair/retry without closing the borrowed descriptor. Both backend source paths normalize already-gone registrations. A private legacy wait context retires its own backend on a removal error before the borrowed descriptor may be released. Linux wake pipes are created atomically close-on-exec. Owner-thread and BUSY/refusal behavior is documented and tested.

The [manifest](shared-loop-leases-2026-11-12.json) binds the tested source revision and original local-log hashes. Nine fresh scoped CTests passed. Five standalone shared/handle/net/TLS suites passed in both normal and ASan/UBSan/leak modes. Relevant generated core API pages match their edited headers. This does not establish macOS, compiler/JIT/AOT or package/release acceptance.
