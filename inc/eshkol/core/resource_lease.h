/**
 * @file resource_lease.h
 * @brief Shared lease and byte-counted outcome layer for hosted I/O resources.
 *
 * Every owned I/O resource kind (TCP/UDP sockets, TLS sessions, and the local
 * IPC kinds that follow the same contract) is adopted into one process-wide
 * registry from eshkol/core/handle.h. This header is the small common layer on
 * top of it:
 *
 *   - one acquire site per kind: eshkol_resource_adopt();
 *   - one close site per kind:   eshkol_resource_close();
 *   - exclusive borrow/release around each operation;
 *   - one monotonic deadline per operation, shared across retries and phases;
 *   - readiness waits through the portable event loop (kqueue/epoll/IOCP),
 *     interruptible by eshkol_resource_cancel() from any thread;
 *   - one table of byte-counted outcome codes, identical for every kind.
 *
 * Identities are exact positive integers no larger than 2^53 - 1 and are never
 * reused. Native bookkeeping lives outside movable arena regions.
 */
#ifndef ESHKOL_CORE_RESOURCE_LEASE_H
#define ESHKOL_CORE_RESOURCE_LEASE_H
#include <stdint.h>
#include "eshkol/core/handle.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name Outcome codes
 * A nonnegative result is an exact byte count (or a new identity). A stream
 * read that returns zero for a positive request means end of stream.
 * @{
 */
#define ESHKOL_IO_SYSTEM       (-1)  /**< Operating-system failure.                    */
#define ESHKOL_IO_TIMEOUT      (-2)  /**< Deadline reached; zero deadline = would-block. */
#define ESHKOL_IO_UNSUPPORTED  (-3)  /**< Not available on this build or platform.     */
#define ESHKOL_IO_INVALID      (-4)  /**< Invalid argument, retired or other-kind handle. */
#define ESHKOL_IO_BUSY         (-5)  /**< Another operation holds the lease.            */
#define ESHKOL_IO_CANCELLED    (-6)  /**< Cancelled, or closed during the operation.    */
#define ESHKOL_IO_RESET        (-7)  /**< Peer reset the connection or closed its reader. */
#define ESHKOL_IO_TRUNCATED    (-8)  /**< A datagram exceeded the receive capacity.     */
#define ESHKOL_IO_REFUSED      (-9)  /**< Connection refused by the peer.               */
#define ESHKOL_IO_UNTRUSTED    (-10) /**< TLS peer certificate failed verification.     */
#define ESHKOL_IO_IDENTITY     (-11) /**< TLS peer certificate names another host.      */
/** @} */

/** @name Readiness interest, identical to ESHKOL_EVENT_READ / ESHKOL_EVENT_WRITE. @{ */
#define ESHKOL_RESOURCE_READ  0x1
#define ESHKOL_RESOURCE_WRITE 0x2
/** @} */

/** An active exclusive borrow. Valid until eshkol_resource_release(). */
typedef struct {
    int64_t id;
    uint64_t lease;
    uintptr_t native;
    uint32_t tag;
} eshkol_resource_lease_t;

/** @brief The process-wide exact registry, or NULL when it cannot be created. */
eshkol_handle_registry_t *eshkol_resource_registry(void);

/** @brief Monotonic milliseconds, or -1 when the clock fails. */
int64_t eshkol_resource_now_ms(void);
/** @brief now + timeout_ms; -1 on clock failure or a negative timeout. */
int64_t eshkol_resource_deadline(int32_t timeout_ms);
/** @brief Remaining milliseconds (0 once reached), or -1 on clock failure. */
int32_t eshkol_resource_remaining(int64_t deadline);

/**
 * @brief The acquire site: adopt sole ownership of @p native under @p kind/@p tag.
 * @return A positive identity; on failure ownership stays with the caller and
 * ESHKOL_IO_SYSTEM is returned.
 */
int64_t eshkol_resource_adopt(eshkol_handle_kind_t kind, uint32_t tag, uintptr_t native,
                              eshkol_handle_close_fn close_fn, void *user_data);
/**
 * @brief Exclusively borrow @p id when its kind matches and its tag is in @p tag_mask
 * (bit 1u << tag); a zero mask accepts any tag.
 * @return 0, ESHKOL_IO_INVALID (unknown, retired, closed, other kind) or ESHKOL_IO_BUSY.
 */
int eshkol_resource_borrow(int64_t id, eshkol_handle_kind_t kind, uint32_t tag_mask,
                           eshkol_resource_lease_t *out);
/** @brief End a borrow; a close requested meanwhile runs now, exactly once. */
void eshkol_resource_release(eshkol_resource_lease_t *lease);
/**
 * @brief The close site. Open identities close now; a borrowed identity
 * becomes unusable at once, its borrower is woken (its operation returns
 * ESHKOL_IO_CANCELLED) and the close runs when that borrow ends.
 * @return 0, or ESHKOL_IO_INVALID for unknown, retired, double-closed or
 * other-kind identities.
 */
int eshkol_resource_close(int64_t id, eshkol_handle_kind_t kind, uint32_t tag_mask);
/**
 * @brief Cancel the operation waiting on @p id, or the next one to wait.
 * Safe from any thread. @return 0 or ESHKOL_IO_INVALID.
 */
int eshkol_resource_cancel(int64_t id, eshkol_handle_kind_t kind, uint32_t tag_mask);
/**
 * @brief Wait until @p fd is ready for @p events, the deadline passes, or the
 * borrow is cancelled. Uses the calling thread's event loop.
 * @return 0 ready, ESHKOL_IO_TIMEOUT, ESHKOL_IO_CANCELLED, or ESHKOL_IO_SYSTEM.
 */
int eshkol_resource_wait(const eshkol_resource_lease_t *lease, int fd, int events,
                         int64_t deadline);
/* Explicit-loop owners use the same registry and deadline/outcome machinery.
 * The loop is retained until owner-thread cleanup completes. */
#define ESHKOL_CLEANUP_PENDING 0
#define ESHKOL_CLEANUP_COMPLETE 1
#define ESHKOL_CLEANUP_FAILED 2
#define ESHKOL_CLEANUP_WATCH 1
#define ESHKOL_CLEANUP_DESCRIPTOR 2
#define ESHKOL_CLEANUP_PATH 3
#define ESHKOL_CLEANUP_LOOP 4

typedef struct {
    int32_t state;
    int32_t stage;
    int32_t os_error;
    int32_t retryable;
} eshkol_resource_cleanup_result_t;
/* Return 0 after disposing native/data. On failure fill state/stage/os_error
 * and retryable; retain native/data when retryable, dispose them otherwise.
 * The callback is called only on the adopting thread, outside shared locks. */
typedef int (*eshkol_resource_cleanup_fn)(uintptr_t native, void *data,
                                         eshkol_resource_cleanup_result_t *result);
int64_t eshkol_resource_adopt_on_loop(eshkol_handle_kind_t kind, uint32_t tag,
    uintptr_t native, eshkol_resource_cleanup_fn cleanup_fn, void *data,
    int64_t loop, int sticky_cancel);
/* Check immediately before a native I/O attempt. */
int eshkol_resource_check(const eshkol_resource_lease_t *lease);
/* Positive requested readiness mask, or a shared error. */
int eshkol_resource_wait_on_loop(const eshkol_resource_lease_t *lease,
    int fd, int events, int64_t deadline);
/* Accept close and retire its identity; return a separately owned result ticket.
 * The ticket is reserved at adoption, so accepting close never allocates.
 * Result inspection may drive queued cleanup on the adopting thread. */
int64_t eshkol_resource_close_request(int64_t id, eshkol_handle_kind_t kind,
                                    uint32_t tag_mask);
int eshkol_resource_cleanup_result(int64_t ticket,
                                  eshkol_resource_cleanup_result_t *out);
int eshkol_resource_cleanup_retry(int64_t ticket);
int eshkol_resource_cleanup_release(int64_t ticket);
/* Transfer ownership of a retained FAILED ticket from detached convenience
 * close. Pending/successful tickets cannot be recovered or race auto-release. */
int64_t eshkol_resource_cleanup_ticket(int64_t retired_id);
/* Drive pending cleanup belonging to this thread/loop, without reading a clock. */
int eshkol_resource_pump(int64_t loop);
/* Count and validate common wait registrations for mixed PROCESS/SOCKET loops. */
int eshkol_resource_managed_registration_count(int64_t loop);

/**
 * @brief Validate [start, start + count) inside an Eshkol bytevector object
 * (an exact length word followed by its bytes) and return its first byte.
 * @return 0 or ESHKOL_IO_INVALID.
 */
int eshkol_resource_bytes(const void *bytevector, int64_t start, int64_t count,
                          uint8_t **out_bytes);

#ifdef __cplusplus
}
#endif
#endif
