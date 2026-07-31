/**
 * @file event_loop_internal.h
 * @brief Private contract between lib/core/event_loop.c and the one
 *        per-platform backend file CMake selects.
 *
 * Not installed and not part of the public API — include
 * <eshkol/core/event_loop.h> instead.
 *
 * The split is deliberate. `event_loop.c` owns everything that is identical
 * on every platform (handle registry, argument validation, the registration
 * table, coalescing rules, the shape of the result array), so a backend file
 * contains nothing but the five kernel calls it exists to make. That keeps
 * the code that cannot be tested on the developer's machine — the IOCP path —
 * as small as it can possibly be.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#ifndef ESHKOL_CORE_EVENT_LOOP_INTERNAL_H
#define ESHKOL_CORE_EVENT_LOOP_INTERNAL_H

#include "eshkol/core/event_loop.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** One registered descriptor and the interest currently held for it. */
typedef struct {
    int32_t  fd;         /**< Descriptor; -1 marks a free slot.        */
    int32_t  events;     /**< Interest bits currently armed.           */
    uint64_t user_data;  /**< Cookie echoed back on notification.      */
} eshkol_event_reg_t;

/**
 * @brief The loop. Transparent to the backend, opaque to everyone else.
 *
 * Mirrors the GPU backend's `EshkolGPUBuffer` convention: a shared struct
 * carrying a `backend_data` escape hatch rather than a vtable, because exactly
 * one backend is ever compiled in.
 */
struct eshkol_event_loop {
    int  max_events;          /**< Capacity for both registrations and results. */
    int  n_regs;              /**< Number of live entries in `regs`.            */
    int  last_os_error;       /**< errno / GetLastError from the last failure.  */

    eshkol_event_reg_t* regs; /**< malloc'd, `max_events` entries.              */
    eshkol_event_t*     out;  /**< malloc'd, `max_events` entries; poll output. */

    intptr_t backend_fd;      /**< kqueue fd, epoll fd, or IOCP HANDLE. -1 = none. */
    void*    backend_data;    /**< Backend scratch (kevent/epoll_event array).  */
};

/* ─────────────────────────────────────────────────────────────────────
 * Backend contract. Exactly one translation unit defines these.
 *
 * Every entry point returns an ESHKOL_EVENT_LOOP_* code and records host
 * error detail in `loop->last_os_error`. The core has already validated
 * arguments and updated `loop->regs` before calling in, so a backend never
 * has to re-check them.
 * ───────────────────────────────────────────────────────────────────── */

/**
 * @brief Create the kernel multiplexer and store it in `loop->backend_fd`.
 *        Allocate any per-poll scratch into `loop->backend_data`.
 */
int eshkol_event_backend_open(eshkol_event_loop_t* loop);

/**
 * @brief Arm interest in @p fd.
 * @param prev_events Interest previously armed for @p fd, or 0 if this is the
 *        first registration. Backends that distinguish add from modify (epoll)
 *        use this to choose EPOLL_CTL_ADD vs EPOLL_CTL_MOD without a syscall.
 */
int eshkol_event_backend_add(eshkol_event_loop_t* loop, int fd, int events,
                             int prev_events);

/**
 * @brief Disarm all interest in @p fd.
 * @param prev_events Interest currently armed, so kqueue knows which filters
 *        to EV_DELETE.
 */
int eshkol_event_backend_remove(eshkol_event_loop_t* loop, int fd,
                                int prev_events);

/**
 * @brief Wait and fill `loop->out`.
 *
 * The backend writes at most `loop->max_events` entries, already coalesced to
 * one per descriptor, and stores the count in @p n_out. A timeout is
 * ESHKOL_EVENT_LOOP_OK with `*n_out == 0`; so is a signal interruption.
 */
int eshkol_event_backend_poll(eshkol_event_loop_t* loop, int timeout_ms,
                              int* n_out);

/** @brief Release the kernel multiplexer and any `backend_data`. */
void eshkol_event_backend_close(eshkol_event_loop_t* loop);

/** @brief Backend identity: "kqueue", "epoll", "iocp" or "none". */
const char* eshkol_event_backend_name(void);

/* ─────────────────────────────────────────────────────────────────────
 * Helpers the backends share.
 * ───────────────────────────────────────────────────────────────────── */

/**
 * @brief Look up the cookie registered for @p fd, or 0 if it is not registered.
 *
 * Backends that cannot round-trip a 64-bit cookie through the kernel (epoll's
 * `epoll_data` is a union that must hold the fd for us to identify the
 * descriptor; IOCP has no per-readiness cookie at all) recover it from the
 * registration table instead.
 */
uint64_t eshkol_event_lookup_user_data(const eshkol_event_loop_t* loop, int fd);

/**
 * @brief Merge a readiness notification into `loop->out`, coalescing on fd.
 *
 * kqueue reports EVFILT_READ and EVFILT_WRITE as separate records for the same
 * descriptor; every backend funnels through here so the public array holds
 * exactly one entry per descriptor with the bits OR'd together, on every
 * platform.
 *
 * @param n_out In/out count of entries currently in `loop->out`.
 * @return 1 if the notification was recorded, 0 if the array was already full.
 */
int eshkol_event_emit(eshkol_event_loop_t* loop, int* n_out, int fd, int events);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_EVENT_LOOP_INTERNAL_H */
