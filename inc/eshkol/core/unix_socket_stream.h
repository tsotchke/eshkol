/* SPDX-License-Identifier: MIT */
#ifndef ESHKOL_CORE_UNIX_SOCKET_STREAM_H
#define ESHKOL_CORE_UNIX_SOCKET_STREAM_H
#include <stdint.h>
#include "eshkol/core/resource_lease.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Unix stream sockets use the shared resource registry, wait facility and
 * outcome table from resource_lease.h. */
enum {
    ESHKOL_UNIX_STREAM_SOCKET_EINVAL = ESHKOL_IO_INVALID,
    ESHKOL_UNIX_STREAM_SOCKET_WOULD_BLOCK = ESHKOL_IO_TIMEOUT,
    ESHKOL_UNIX_STREAM_SOCKET_ETIMEDOUT = ESHKOL_IO_TIMEOUT,
    ESHKOL_UNIX_STREAM_SOCKET_ENOTSUP = ESHKOL_IO_UNSUPPORTED,
    ESHKOL_UNIX_STREAM_SOCKET_ESYSTEM = ESHKOL_IO_SYSTEM,
    ESHKOL_UNIX_STREAM_SOCKET_EBUSY = ESHKOL_IO_BUSY,
    ESHKOL_UNIX_STREAM_SOCKET_ECANCELED = ESHKOL_IO_CANCELLED,
    ESHKOL_UNIX_STREAM_SOCKET_ERESET = ESHKOL_IO_RESET,
    ESHKOL_UNIX_STREAM_SOCKET_EREFUSED = ESHKOL_IO_REFUSED,
    ESHKOL_UNIX_STREAM_SOCKET_PLANNED = -100,
    ESHKOL_UNIX_STREAM_SOCKET_READABLE = 1,
    ESHKOL_UNIX_STREAM_SOCKET_WRITABLE = 2
};

/* loop is an exact positive EVENT_LOOP handle; zero has no implicit meaning.
 * Acquisition retains it; accepted peers inherit/retain it. Close releases
 * the retain only after watch removal and descriptor teardown. Loop close
 * refuses while retained. Use the shared lease adoption/borrow/release/close
 * machinery and a common wait extension for explicit loops; never a private
 * socket registry or polling implementation.
 *
 * Paths must be absolute/NUL-free, fit sun_path in encoded bytes, and live
 * in a verified owned private directory with mode 0700. Refuse every existing entry.
 * Bind, set mode 0600, then listen/publish; undo unpublished setup on failure.
 * Only remove the listener's recorded entry, never a replacement. If unpublished
 * rollback cannot remove that entry, last_path_error reports its OS error
 * separately from the acquisition error; the supplied path needs explicit repair. */
int64_t eshkol_unix_stream_socket_listen(int64_t loop, const char *path);
int64_t eshkol_unix_stream_socket_accept(int64_t listener, int32_t deadline_ms);

/* begin captures one monotonic deadline and publishes a connecting owner
 * before any readiness wait. Immediate errors roll back unpublished state.
 * Zero budget never waits: return a connected owner or WOULD_BLOCK with
 * complete rollback. finish returns 0 when connected or a shared error,
 * waiting only within begin's remaining budget. A failed/cancelled published
 * attempt remains owned until close. Read/write require connected state. */
int64_t eshkol_unix_stream_socket_connect_begin(int64_t loop, const char *path,
                                               int32_t deadline_ms);
int32_t eshkol_unix_stream_socket_connect_finish(int64_t id);

/* Nonblocking exact byte counts; positive request sizes only. Read0 is EOF;
 * short counts are progress; WOULD_BLOCK is not EOF/error. SIGPIPE is
 * suppressed per socket/send, without changing process signal disposition. */
int64_t eshkol_unix_stream_socket_read(int64_t id, void *buffer, int64_t capacity);
int64_t eshkol_unix_stream_socket_write(int64_t id, const void *buffer, int64_t count);

/* Return a positive requested readiness mask, TIMEOUT, CANCELLED or another
 * shared error. Zero timeout makes one nonblocking pass. While connecting,
 * the stored acquisition deadline caps this wait. Connected I/O has its own
 * poll budget. Preserve other PROCESS/SOCKET owners' readiness cookies. */
int32_t eshkol_unix_stream_socket_poll(int64_t id, int32_t interests, int32_t timeout_ms);

/* Sticky, idempotent cancellation: a lifetime-safe generation-checked request
 * and shared wakeup may arrive from another thread; further I/O is cancelled.
 * I/O and watch mutation stay owner-thread confined. Ownership remains. */
int32_t eshkol_unix_stream_socket_cancel(int64_t id);
/* Public close follows the shared lease's deferred-close contract: once
 * accepted, invalidate the identity and wake its borrower; descriptor close
 * runs once after release. Remove watches before committing teardown; if
 * removal fails, retain ownership for retry. A socket-path failure permits one
 * explicit repair/retry; a second failure is terminal and surrenders the residual
 * pathname, recording its PATH error while releasing the native record/loop retain.
 * The callback owns terminality; the shared layer never times out a retry.
 * Never retry uncertain fd close.
 * Cancel/close require neither renewed network capability nor a clock;
 * runtime/kind/tag/generation checks remain mandatory. Acquisition and I/O
 * require native and Scheme network admission. */
int32_t eshkol_unix_stream_socket_close(int64_t id);
/* Accept close and return a separately owned cleanup ticket. The socket
 * identity is retired immediately. Inspect and release the ticket even when
 * cleanup completes immediately; failed retryable cleanup retains ownership. */
int64_t eshkol_unix_stream_socket_close_request(int64_t id);
/* Scheme boundary: one coherent snapshot encoded as state bits0..1, stage
 * bits2..4, retryable bit5 and nonnegative OS error bits6..36. Negative values
 * are shared errors. The encoding stays exact in the generic numeric FFI. */
int64_t eshkol_unix_stream_socket_cleanup_result_packed(int64_t ticket);
int32_t eshkol_unix_stream_socket_cleanup_retry(int64_t ticket);
int32_t eshkol_unix_stream_socket_cleanup_release(int64_t ticket);
/* Recover a retained cleanup failure after convenience close; INVALID means
 * no retained result is available for that retired socket identity. */
int64_t eshkol_unix_stream_socket_cleanup_ticket(int64_t retired_id);
/* Calling-thread OS detail for the last operation; read immediately after a
 * shared negative outcome. Zero means no OS detail. */
int32_t eshkol_unix_stream_socket_last_os_error(void);
/* Calling-thread OS error for an unpublished listener rollback whose pathname
 * could not be removed/verified. Read immediately after a failed acquisition.
 * Nonzero means the supplied path needs explicit repair; it grants no authority
 * to unlink a replacement entry. The next native operation clears this detail. */
int32_t eshkol_unix_stream_socket_last_path_error(void);
#ifdef __cplusplus
}
#endif
#endif
