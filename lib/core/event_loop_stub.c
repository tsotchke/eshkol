/**
 * @file event_loop_stub.c
 * @brief Fail-closed event-loop backend for targets with no I/O multiplexer.
 *
 * Selected when the host is neither Apple/BSD (kqueue), Linux (epoll), nor
 * Windows (IOCP) — in practice WebAssembly and any future bare-metal target.
 * Mirrors lib/backend/gpu/gpu_memory_stub.cpp: the symbols exist so the tree
 * links everywhere, and every entry point refuses honestly instead of
 * pretending to work.
 *
 * `eshkol_event_backend_open` failing makes `eshkol_event_loop_create` return
 * NULL, which makes `(make-event-loop n)` return #f. A caller therefore
 * discovers at the first call that this platform has no event loop, rather than
 * receiving a handle that silently never reports anything — and #f is the same
 * degradation `make-pipe` and `fd-write` already use where descriptors do not
 * exist.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#include "event_loop_internal.h"

const char* eshkol_event_backend_name(void) { return "none"; }

int eshkol_event_backend_open(eshkol_event_loop_t* loop) {
    (void)loop;
    return ESHKOL_EVENT_LOOP_ENOTSUP;
}

int eshkol_event_backend_add(eshkol_event_loop_t* loop, int fd, int events,
                             int prev_events) {
    (void)loop; (void)fd; (void)events; (void)prev_events;
    return ESHKOL_EVENT_LOOP_ENOTSUP;
}

int eshkol_event_backend_remove(eshkol_event_loop_t* loop, int fd,
                                int prev_events) {
    (void)loop; (void)fd; (void)prev_events;
    return ESHKOL_EVENT_LOOP_ENOTSUP;
}

int eshkol_event_backend_poll(eshkol_event_loop_t* loop, int timeout_ms,
                              int* n_out) {
    (void)loop; (void)timeout_ms;
    if (n_out) *n_out = 0;
    return ESHKOL_EVENT_LOOP_ENOTSUP;
}

void eshkol_event_backend_close(eshkol_event_loop_t* loop) { (void)loop; }
