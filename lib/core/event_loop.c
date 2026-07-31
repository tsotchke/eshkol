/**
 * @file event_loop.c
 * @brief Portable half of the ESH-0011 event loop: handle registry, argument
 *        validation, registration bookkeeping, and result coalescing.
 *
 * Everything here is platform-independent. The five kernel calls live in the
 * one backend file CMake selects (event_loop_kqueue.c / event_loop_epoll.c /
 * event_loop_iocp.c / event_loop_stub.c) behind the contract in
 * event_loop_internal.h.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#include "event_loop_internal.h"

#include <stdlib.h>
#include <string.h>

/* ─────────────────────────────────────────────────────────────────────
 * Pointer API
 * ───────────────────────────────────────────────────────────────────── */

eshkol_event_loop_t* eshkol_event_loop_create(int max_events) {
    if (max_events <= 0 || max_events > ESHKOL_EVENT_LOOP_MAX_EVENTS) return NULL;

    /* calloc, never the arena: lib/core/runtime_regions.cpp evacuates a region
     * by memcpy-ing objects to a new address, which would duplicate a struct
     * that owns a kernel descriptor. A malloc'd pointer is outside every
     * region arena, so region_index_owning() returns -1 and the evacuator
     * leaves it alone. See the memory-model note in event_loop.h. */
    eshkol_event_loop_t* loop =
        (eshkol_event_loop_t*)calloc(1, sizeof(eshkol_event_loop_t));
    if (!loop) return NULL;

    loop->max_events = max_events;
    loop->backend_fd = -1;
    loop->regs = (eshkol_event_reg_t*)calloc((size_t)max_events,
                                             sizeof(eshkol_event_reg_t));
    loop->out = (eshkol_event_t*)calloc((size_t)max_events,
                                        sizeof(eshkol_event_t));
    if (!loop->regs || !loop->out) {
        free(loop->regs);
        free(loop->out);
        free(loop);
        return NULL;
    }
    for (int i = 0; i < max_events; ++i) loop->regs[i].fd = -1;

    if (eshkol_event_backend_open(loop) != ESHKOL_EVENT_LOOP_OK) {
        free(loop->regs);
        free(loop->out);
        free(loop);
        return NULL;
    }
    return loop;
}

/** Index of @p fd in the registration table, or -1. */
static int reg_find(const eshkol_event_loop_t* loop, int fd) {
    for (int i = 0; i < loop->max_events; ++i)
        if (loop->regs[i].fd == fd) return i;
    return -1;
}

/** Index of a free registration slot, or -1 when the table is full. */
static int reg_free_slot(const eshkol_event_loop_t* loop) {
    for (int i = 0; i < loop->max_events; ++i)
        if (loop->regs[i].fd < 0) return i;
    return -1;
}

uint64_t eshkol_event_lookup_user_data(const eshkol_event_loop_t* loop, int fd) {
    if (!loop) return 0;
    const int idx = reg_find(loop, fd);
    return idx < 0 ? 0u : loop->regs[idx].user_data;
}

int eshkol_event_emit(eshkol_event_loop_t* loop, int* n_out, int fd, int events) {
    if (!loop || !n_out) return 0;

    /* Coalesce: kqueue delivers one record per filter, epoll and the Windows
     * path one per descriptor. Callers of the public API see the epoll shape
     * everywhere. */
    for (int i = 0; i < *n_out; ++i) {
        if (loop->out[i].fd == fd) {
            loop->out[i].events |= events;
            return 1;
        }
    }
    if (*n_out >= loop->max_events) return 0;

    loop->out[*n_out].fd = fd;
    loop->out[*n_out].events = events;
    loop->out[*n_out].user_data = eshkol_event_lookup_user_data(loop, fd);
    ++(*n_out);
    return 1;
}

int eshkol_event_loop_add(eshkol_event_loop_t* loop, int fd, int events,
                          uint64_t user_data) {
    if (!loop || fd < 0) return ESHKOL_EVENT_LOOP_EINVAL;

    /* ESHKOL_EVENT_ERROR is reported, never requested — asking for it alone is
     * a caller mistake, not a silent no-op registration. */
    events &= ESHKOL_EVENT_ALL;
    if (events == 0) return ESHKOL_EVENT_LOOP_EINVAL;

    int idx = reg_find(loop, fd);
    const int prev_events = (idx >= 0) ? loop->regs[idx].events : 0;
    if (idx < 0) {
        idx = reg_free_slot(loop);
        if (idx < 0) return ESHKOL_EVENT_LOOP_EFULL;
    }

    const int rc = eshkol_event_backend_add(loop, fd, events, prev_events);
    if (rc != ESHKOL_EVENT_LOOP_OK) return rc;

    if (prev_events == 0) ++loop->n_regs;
    loop->regs[idx].fd = fd;
    loop->regs[idx].events = events;
    loop->regs[idx].user_data = user_data;
    return ESHKOL_EVENT_LOOP_OK;
}

int eshkol_event_loop_remove(eshkol_event_loop_t* loop, int fd) {
    if (!loop || fd < 0) return ESHKOL_EVENT_LOOP_EINVAL;

    const int idx = reg_find(loop, fd);
    if (idx < 0) return ESHKOL_EVENT_LOOP_ENOENT;

    const int rc = eshkol_event_backend_remove(loop, fd, loop->regs[idx].events);

    /* Drop the registration even when the kernel complained. A descriptor the
     * kernel no longer recognises (already closed behind our back) must not
     * stay pinned in the table forever, or the loop leaks capacity and every
     * later poll re-reports it. */
    loop->regs[idx].fd = -1;
    loop->regs[idx].events = 0;
    loop->regs[idx].user_data = 0;
    if (loop->n_regs > 0) --loop->n_regs;
    return rc;
}

int eshkol_event_loop_poll(eshkol_event_loop_t* loop, int timeout_ms,
                           const eshkol_event_t** out_events, int* n_events) {
    if (!loop || !out_events || !n_events) return ESHKOL_EVENT_LOOP_EINVAL;

    *out_events = loop->out;
    *n_events = 0;

    /* Nothing registered: honour the timeout contract without a syscall. A
     * kqueue with no filters and an infinite timeout would block forever with
     * no possible waker, which is a hang, not a wait. */
    if (loop->n_regs == 0) return ESHKOL_EVENT_LOOP_OK;

    int n = 0;
    const int rc = eshkol_event_backend_poll(loop, timeout_ms, &n);
    if (rc != ESHKOL_EVENT_LOOP_OK) return rc;

    if (n < 0) n = 0;
    if (n > loop->max_events) n = loop->max_events;
    *n_events = n;
    return ESHKOL_EVENT_LOOP_OK;
}

int eshkol_event_loop_close(eshkol_event_loop_t* loop) {
    if (!loop) return ESHKOL_EVENT_LOOP_EINVAL;
    eshkol_event_backend_close(loop);
    free(loop->regs);
    free(loop->out);
    free(loop);
    return ESHKOL_EVENT_LOOP_OK;
}

int eshkol_event_loop_count(const eshkol_event_loop_t* loop) {
    return loop ? loop->n_regs : -1;
}

int eshkol_event_loop_last_os_error(const eshkol_event_loop_t* loop) {
    return loop ? loop->last_os_error : 0;
}

const char* eshkol_event_loop_backend_name(void) {
    return eshkol_event_backend_name();
}

/* ─────────────────────────────────────────────────────────────────────
 * Handle registry — the Scheme-visible surface.
 *
 * Same shape as the sqlite handle table in lib/agent/c/agent_sqlite.c and the
 * line-reader table in lib/agent/c/agent_poll.c: a fixed static table, slot 0
 * reserved so a handle is never 0, and a plain integer handed to Scheme so the
 * region evacuator never sees a heap value at all.
 *
 * The one addition is a generation counter. Without it, closing loop A and
 * opening loop B would hand B the same slot, and a stale reference to A would
 * silently address B. With it, the stale handle fails closed.
 * ───────────────────────────────────────────────────────────────────── */

#define ESHKOL_EVENT_LOOP_MAX_HANDLES 256
#define ESHKOL_EVENT_LOOP_SLOT_BITS   16
#define ESHKOL_EVENT_LOOP_SLOT_MASK   ((int64_t)0xFFFF)

typedef struct {
    eshkol_event_loop_t* loop;       /**< NULL when the slot is free.       */
    int64_t              generation; /**< Bumped on every (re)use of a slot. */
} eshkol_event_loop_slot_t;

static eshkol_event_loop_slot_t g_loop_slots[ESHKOL_EVENT_LOOP_MAX_HANDLES];

/*
 * Thread-safety, stated plainly: this table is not synchronised, exactly like
 * every other handle table in the tree (agent_sqlite.c, agent_poll.c,
 * agent_watch.c). A loop handle is meant to be owned by the thread that opened
 * it. v1.4's fiber scheduler is single-threaded per loop by design, so this is
 * sufficient for the ticket's payoff; sharing one loop across OS threads needs
 * a lock that lands with that work, not before it.
 */

int64_t eshkol_event_loop_open_handle(int max_events) {
    eshkol_event_loop_t* loop = eshkol_event_loop_create(max_events);
    if (!loop) return -1;

    for (int slot = 1; slot < ESHKOL_EVENT_LOOP_MAX_HANDLES; ++slot) {
        if (g_loop_slots[slot].loop) continue;
        g_loop_slots[slot].loop = loop;
        ++g_loop_slots[slot].generation;
        return ((int64_t)g_loop_slots[slot].generation << ESHKOL_EVENT_LOOP_SLOT_BITS)
             | (int64_t)slot;
    }

    /* Table exhausted — do not leak the kernel object we just created. */
    eshkol_event_loop_close(loop);
    return -1;
}

eshkol_event_loop_t* eshkol_event_loop_from_handle(int64_t handle) {
    if (handle <= 0) return NULL;
    const int64_t slot = handle & ESHKOL_EVENT_LOOP_SLOT_MASK;
    const int64_t generation = handle >> ESHKOL_EVENT_LOOP_SLOT_BITS;
    if (slot <= 0 || slot >= ESHKOL_EVENT_LOOP_MAX_HANDLES) return NULL;
    if (!g_loop_slots[slot].loop) return NULL;
    if (g_loop_slots[slot].generation != generation) return NULL;
    return g_loop_slots[slot].loop;
}

int eshkol_event_loop_close_handle(int64_t handle) {
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(handle);
    if (!loop) return ESHKOL_EVENT_LOOP_EINVAL;

    const int64_t slot = handle & ESHKOL_EVENT_LOOP_SLOT_MASK;
    g_loop_slots[slot].loop = NULL;
    /* generation is deliberately left at its current value and bumped on the
     * next allocation, so this handle can never validate again. */
    return eshkol_event_loop_close(loop);
}
