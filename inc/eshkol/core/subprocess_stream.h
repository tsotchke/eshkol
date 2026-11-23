/* SPDX-License-Identifier: MIT */
#ifndef ESHKOL_CORE_SUBPROCESS_STREAM_H
#define ESHKOL_CORE_SUBPROCESS_STREAM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum {
    ESHKOL_PROCESS_STREAM_STDOUT = 1,
    ESHKOL_PROCESS_STREAM_STDERR = 2,
    ESHKOL_PROCESS_STREAM_EXITED = 4,
    ESHKOL_PROCESS_STREAM_PENDING = 1,
    ESHKOL_PROCESS_STREAM_EINVAL = -1,
    ESHKOL_PROCESS_STREAM_WOULD_BLOCK = -2,
    ESHKOL_PROCESS_STREAM_ENOTSUP = -3,
    ESHKOL_PROCESS_STREAM_ESYSTEM = -4,
    ESHKOL_PROCESS_STREAM_EBUSY = -5,
    ESHKOL_PROCESS_STREAM_ELIMIT = -6,
    ESHKOL_PROCESS_STREAM_ETHREAD = -7,
    ESHKOL_PROCESS_STREAM_EDENIED = -8
};

/* Single owner thread, POSIX first slice. argv/env use the existing tab-packed
 * wire format. stdin is null. The existing loop must have room for both pipes.
 * A positive generation-tagged ID (at most 2^53-1) owns one exclusive lease
 * covering both outputs, preserving identity through existing FFI numeric boxing.
 * Unsupported platforms return ENOTSUP without spawning. */
int64_t eshkol_process_stream_open(int64_t loop, const char *argv,
    const char *cwd, const char *env);
/* Positive byte count, zero EOF, WOULD_BLOCK, or a negative error. No capture
 * buffer or competing reader; capacity is 1..INT32_MAX. Embedded NULs survive. */
int64_t eshkol_process_stream_read(int64_t id, int32_t which,
    void *buffer, int64_t capacity);
/* Readiness mask STDOUT|STDERR|EXITED, zero timeout, or a negative error.
 * Zero timeout never sleeps. Positive timeouts share one monotonic deadline;
 * child status and explicit cancellation progress independently of pipe EOF.
 * EXITED is emitted once; exit_code retains the observed status. */
int32_t eshkol_process_stream_poll(int64_t id, int32_t timeout_ms);
/* Nonblocking: child exit code, WOULD_BLOCK while running, or a negative error. */
int32_t eshkol_process_stream_exit_code(int64_t id);
/* Request direct-child TERM then KILL after grace; never wait or reset a request. */
int32_t eshkol_process_stream_cancel(int64_t id, int32_t grace_ms);
/* End stream access, unregister/close pipes, and release the loop/owner lease.
 * Returns 0 when reaped, PENDING when budget expires, or a negative error.
 * Repeated close observes the same cleanup record. The runtime retains pending
 * ownership; no close callback runs before reap is confirmed. */
int32_t eshkol_process_stream_close(int64_t id, int32_t timeout_ms,
    int32_t grace_ms);
/* Closed records only, ordered by ID; start enumeration with after=0. */
int64_t eshkol_process_cleanup_next(int64_t after);
int32_t eshkol_process_cleanup_status(int64_t id);
/* Advance all pending records on an existing loop, returning pending count.
 * Loop lifetime is independent of the original session's loop. */
int32_t eshkol_process_cleanup_drain(int64_t loop, int32_t timeout_ms);
/* Retire a terminal record; pending records return EBUSY. IDs never revive. */
int32_t eshkol_process_cleanup_consume(int64_t id);
/* EBUSY while any record remains; otherwise release the empty registry.
 * Successful shutdown is terminal; subsequent opens return EBUSY. */
int32_t eshkol_process_stream_runtime_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif
