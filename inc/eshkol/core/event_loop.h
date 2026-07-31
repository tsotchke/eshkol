/**
 * @file event_loop.h
 * @brief Portable readiness-style event loop (kqueue / epoll / IOCP).
 *
 * ESH-0011. One internal abstraction over the host's I/O multiplexer:
 *
 *   - macOS / *BSD : kqueue + kevent            (lib/core/event_loop_kqueue.c)
 *   - Linux        : epoll_create1 + epoll_wait (lib/core/event_loop_epoll.c)
 *   - Windows      : IOCP + WSAPoll/PeekNamedPipe
 *                                               (lib/core/event_loop_iocp.c)
 *   - other/wasm   : fail-closed stub           (lib/core/event_loop_stub.c)
 *
 * Exactly one backend file is compiled, selected in CMakeLists.txt the same
 * way the GPU backend picks gpu_memory.mm / gpu_memory_cuda.cpp /
 * gpu_memory_stub.cpp. The portable half (handle registry, argument
 * validation, registration bookkeeping) lives in lib/core/event_loop.c and is
 * shared by every backend.
 *
 * @section surface Readiness, not completion
 *
 * The surface is *readiness*-style, matching kqueue and epoll: you register
 * interest in a descriptor and poll asks "which of these can I act on now?".
 * That is the model v1.4's sockets, fibers and timers are specified against.
 *
 * Windows IOCP is a *completion* model and cannot answer that question
 * directly. See the honest adaptation notes at the top of
 * lib/core/event_loop_iocp.c: the loop owns a real completion port (created,
 * closed, and used as the interruptible wait primitive), while readiness
 * itself is derived from WSAPoll for sockets and the documented
 * PeekNamedPipe/GetFileType inspection for anonymous pipes, consoles and
 * files — because anonymous Win32 pipe handles cannot be associated with a
 * completion port at all.
 *
 * @section memory Memory model — deliberately outside the arena
 *
 * An `eshkol_event_loop_t` is `malloc`'d, never arena-allocated, and the
 * Scheme-visible value is a plain integer handle (see
 * eshkol_event_loop_open_handle). This is the convention every other
 * kernel-resource holder in the tree already uses (the sqlite handle tables
 * in lib/agent/c/agent_sqlite.c, the line-reader table in
 * lib/agent/c/agent_poll.c, the watcher table in lib/agent/c/agent_watch.c).
 *
 * It is also the only *safe* choice: lib/core/runtime_regions.cpp evacuates a
 * region by `memcpy`-ing objects to a new address, so an arena-resident struct
 * owning a kernel handle would be byte-duplicated on escape — two structs, one
 * descriptor, two closes. Because a malloc'd pointer is never inside any
 * region arena, `region_index_owning()` returns -1 and the evacuator leaves it
 * alone unconditionally; because the Scheme value is an integer, it is not
 * even a heap value to the evacuator. No `evac_kind_for` entry is required and
 * none is added.
 *
 * @section handles Relationship to ESH-0010
 *
 * ESH-0010 (`ESHKOL_VALUE_HANDLE` lifecycle states) had not landed when this
 * was written. The generation-tagged integer handle below already gives the
 * three properties that ticket wants — a stable identity, a closed state, and
 * a use-after-close that fails closed. When ESH-0010 lands, the integer
 * becomes the `native_handle` payload of a `HANDLE_KIND_EVENT_LOOP` handle and
 * nothing in this file has to change.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#ifndef ESHKOL_CORE_EVENT_LOOP_H
#define ESHKOL_CORE_EVENT_LOOP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name Interest and readiness bits
 *
 * Identical encoding to `eshkol_poll()` in lib/agent/c/agent_poll.c so the two
 * multiplexing layers agree on the wire. ESHKOL_EVENT_ERROR is output-only:
 * passing it as interest is ignored, but it is always reported when the
 * descriptor errors or hangs up, exactly like `POLLERR|POLLHUP` — you never
 * have to ask for error notifications.
 * @{
 */
#define ESHKOL_EVENT_READ   0x1  /**< Readable / EVFILT_READ / EPOLLIN.  */
#define ESHKOL_EVENT_WRITE  0x2  /**< Writable / EVFILT_WRITE / EPOLLOUT. */
#define ESHKOL_EVENT_ERROR  0x4  /**< Error or hangup. Reported, never requested. */
#define ESHKOL_EVENT_ALL    0x3  /**< Mask of the requestable bits.      */
/** @} */

/**
 * @name Result codes
 *
 * Every entry point returns `>= 0` on success and one of these on failure.
 * They are stable across backends so a caller can branch on them portably.
 * @{
 */
#define ESHKOL_EVENT_LOOP_OK        0   /**< Success.                            */
#define ESHKOL_EVENT_LOOP_EINVAL   (-1) /**< Bad argument (null loop, bad fd, …). */
#define ESHKOL_EVENT_LOOP_ENOMEM   (-2) /**< Allocation failed.                   */
#define ESHKOL_EVENT_LOOP_EOSERR   (-3) /**< Kernel call failed; see …_last_os_error. */
#define ESHKOL_EVENT_LOOP_ENOTSUP  (-4) /**< Platform cannot watch this object.   */
#define ESHKOL_EVENT_LOOP_ENOENT   (-5) /**< Descriptor is not registered.        */
#define ESHKOL_EVENT_LOOP_EFULL    (-6) /**< Registration table is at max_events. */
/** @} */

/** Upper bound accepted by eshkol_event_loop_create() for @p max_events. */
#define ESHKOL_EVENT_LOOP_MAX_EVENTS 65536

/**
 * @brief One readiness notification produced by eshkol_event_loop_poll().
 *
 * Backends that report per-filter (kqueue emits separate EVFILT_READ and
 * EVFILT_WRITE records for the same descriptor) coalesce into exactly one
 * entry per descriptor, so the shape is identical on every platform.
 */
typedef struct {
    int32_t  fd;         /**< The descriptor that became ready.            */
    int32_t  events;     /**< ESHKOL_EVENT_* bits that actually fired.     */
    uint64_t user_data;  /**< Opaque cookie supplied at registration time. */
} eshkol_event_t;

/** Opaque loop object. Allocated with malloc(); never arena-allocated. */
typedef struct eshkol_event_loop eshkol_event_loop_t;

/* ─────────────────────────────────────────────────────────────────────
 * Pointer API — what v1.4 sockets, fibers and timers link against.
 * ───────────────────────────────────────────────────────────────────── */

/**
 * @brief Create a loop able to hold and report up to @p max_events descriptors.
 *
 * @p max_events bounds both the registration table and the number of
 * notifications a single poll can return, so one poll can always drain every
 * registered descriptor and starvation is impossible by construction.
 *
 * @param max_events Capacity, in (0, ESHKOL_EVENT_LOOP_MAX_EVENTS].
 * @return A new loop, or NULL on a bad argument, allocation failure, or a
 *         kernel that refused to create the multiplexer.
 */
eshkol_event_loop_t* eshkol_event_loop_create(int max_events);

/**
 * @brief Register or re-register @p fd, replacing any previous interest.
 *
 * Calling add twice for the same descriptor is a modify, not an error — the
 * new @p events and @p user_data win. This mirrors kqueue's EV_ADD and hides
 * the EPOLL_CTL_ADD/EPOLL_CTL_MOD distinction from callers.
 *
 * @param loop      Loop from eshkol_event_loop_create().
 * @param fd        Descriptor to watch (a CRT file descriptor on Windows).
 * @param events    ESHKOL_EVENT_READ and/or ESHKOL_EVENT_WRITE; must be nonzero.
 * @param user_data Cookie echoed back in eshkol_event_t::user_data.
 * @return ESHKOL_EVENT_LOOP_OK, or a negative result code. In particular
 *         ESHKOL_EVENT_LOOP_ENOTSUP when the platform cannot watch this kind
 *         of object (Linux epoll refuses regular files with EPERM).
 */
int eshkol_event_loop_add(eshkol_event_loop_t* loop, int fd, int events,
                          uint64_t user_data);

/**
 * @brief Stop watching @p fd.
 * @return ESHKOL_EVENT_LOOP_OK, or ESHKOL_EVENT_LOOP_ENOENT if @p fd was never
 *         registered.
 */
int eshkol_event_loop_remove(eshkol_event_loop_t* loop, int fd);

/**
 * @brief Wait for readiness on any registered descriptor.
 *
 * On return @p out_events points into storage owned by @p loop that stays
 * valid until the next poll or close on the same loop — no caller buffer, no
 * ownership transfer. A timeout is a success with `*n_events == 0`, never an
 * error.
 *
 * A signal that interrupts the wait (EINTR) is reported as a timeout rather
 * than an error, matching `eshkol_poll()`. That is what keeps an unbounded
 * wait breakable: Ctrl-C interrupts the syscall and the call returns instead
 * of wedging the REPL.
 *
 * @param loop       Loop from eshkol_event_loop_create().
 * @param timeout_ms Milliseconds to wait; 0 polls without blocking, negative
 *                   blocks until an event arrives or a signal interrupts.
 * @param out_events Receives a pointer to @p loop's event array.
 * @param n_events   Receives the number of valid entries (0 on timeout).
 * @return ESHKOL_EVENT_LOOP_OK, or a negative result code.
 */
int eshkol_event_loop_poll(eshkol_event_loop_t* loop, int timeout_ms,
                           const eshkol_event_t** out_events, int* n_events);

/**
 * @brief Release the underlying kqueue/epoll/completion-port object and free
 *        the loop.
 *
 * Registered descriptors are *not* closed — the loop never owned them. After
 * this returns, @p loop is freed and must not be touched again.
 *
 * @return ESHKOL_EVENT_LOOP_OK, or ESHKOL_EVENT_LOOP_EINVAL for a NULL loop.
 */
int eshkol_event_loop_close(eshkol_event_loop_t* loop);

/** @return Number of descriptors currently registered, or -1 for a NULL loop. */
int eshkol_event_loop_count(const eshkol_event_loop_t* loop);

/**
 * @brief Host error detail (errno, or GetLastError/WSAGetLastError) for the
 *        most recent ESHKOL_EVENT_LOOP_EOSERR on @p loop.
 */
int eshkol_event_loop_last_os_error(const eshkol_event_loop_t* loop);

/**
 * @brief Name of the compiled-in backend: "kqueue", "epoll", "iocp" or "none".
 *
 * Reported by `(event-loop-backend)` so a test can say which implementation it
 * actually exercised instead of assuming one.
 */
const char* eshkol_event_loop_backend_name(void);

/* ─────────────────────────────────────────────────────────────────────
 * Handle API — the Scheme-visible surface.
 *
 * A handle is a positive int64 carrying a generation counter in its high bits,
 * so a stale handle whose slot has been recycled is rejected instead of
 * silently addressing a different loop. Every lookup failure is reported to
 * the caller, which turns it into a catchable Scheme condition.
 * ───────────────────────────────────────────────────────────────────── */

/**
 * @brief Create a loop and return a generation-tagged handle for it.
 * @return A handle >= 1, or -1 on failure.
 */
int64_t eshkol_event_loop_open_handle(int max_events);

/**
 * @brief Resolve a handle to its loop.
 * @return The loop, or NULL if the handle is malformed, already closed, or
 *         refers to a recycled slot.
 */
eshkol_event_loop_t* eshkol_event_loop_from_handle(int64_t handle);

/**
 * @brief Close the loop a handle names and retire the handle.
 *
 * Idempotent-observable: closing an already-closed or unknown handle returns
 * ESHKOL_EVENT_LOOP_EINVAL rather than raising, so double-close is reportable
 * without being fatal — which is the contract ESH-0010 specifies for handle
 * lifecycles. Using a closed handle for anything else fails closed, because
 * eshkol_event_loop_from_handle() returns NULL for it.
 *
 * @return ESHKOL_EVENT_LOOP_OK, or ESHKOL_EVENT_LOOP_EINVAL.
 */
int eshkol_event_loop_close_handle(int64_t handle);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_EVENT_LOOP_H */
