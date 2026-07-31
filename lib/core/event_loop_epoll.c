/**
 * @file event_loop_epoll.c
 * @brief Linux event-loop backend built on epoll_create1(2) / epoll_ctl(2) /
 *        epoll_wait(2).
 *
 * epoll is the model the public API is shaped after — one record per
 * descriptor, bits OR'd together — so this backend is the thinnest of the
 * three. Three behaviours are worth naming:
 *
 *  1. epoll refuses regular files. `epoll_ctl` returns EPERM for anything that
 *     is always ready (regular files, and on some kernels /dev/null), because
 *     epoll has nothing to report. kqueue happily accepts them. Rather than let
 *     that difference leak out as a mysterious OS error, EPERM is reported as
 *     ESHKOL_EVENT_LOOP_ENOTSUP and the Scheme layer turns it into a message
 *     that says so. The portable intersection of the three backends is
 *     therefore pipes, sockets and terminals — which is exactly what v1.4's
 *     sockets, subprocess plumbing and timers need.
 *
 *  2. `epoll_data` is a union. It cannot carry both the descriptor and a 64-bit
 *     user cookie, and we need the descriptor to identify what fired, so
 *     `data.fd` holds the descriptor and the cookie is recovered from the
 *     registration table via eshkol_event_lookup_user_data(). kqueue's separate
 *     `udata` field is unused for the same reason — one code path, not two.
 *
 *  3. ADD and MOD are separate operations, and the core already knows which one
 *     applies from `prev_events`. The EEXIST/ENOENT fallbacks below exist for
 *     the case where the kernel's idea of the registration and ours have
 *     diverged (a descriptor closed behind our back and its number reused), so
 *     a re-registration self-heals instead of failing permanently.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#include "event_loop_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <unistd.h>

const char* eshkol_event_backend_name(void) { return "epoll"; }

int eshkol_event_backend_open(eshkol_event_loop_t* loop) {
    const int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd < 0) {
        loop->last_os_error = errno;
        return ESHKOL_EVENT_LOOP_EOSERR;
    }

    struct epoll_event* scratch =
        (struct epoll_event*)calloc((size_t)loop->max_events,
                                    sizeof(struct epoll_event));
    if (!scratch) {
        close(epfd);
        return ESHKOL_EVENT_LOOP_ENOMEM;
    }

    loop->backend_fd = (intptr_t)epfd;
    loop->backend_data = scratch;
    return ESHKOL_EVENT_LOOP_OK;
}

/** Translate an epoll_ctl errno into the portable result code. */
static int ep_map_errno(eshkol_event_loop_t* loop, int err) {
    loop->last_os_error = err;
    /* EPERM: the object is always ready and epoll will not watch it. */
    if (err == EPERM) return ESHKOL_EVENT_LOOP_ENOTSUP;
    if (err == EBADF || err == EINVAL) return ESHKOL_EVENT_LOOP_EINVAL;
    if (err == ENOMEM || err == ENOSPC) return ESHKOL_EVENT_LOOP_ENOMEM;
    return ESHKOL_EVENT_LOOP_EOSERR;
}

int eshkol_event_backend_add(eshkol_event_loop_t* loop, int fd, int events,
                             int prev_events) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    if (events & ESHKOL_EVENT_READ)  ev.events |= (uint32_t)EPOLLIN;
    if (events & ESHKOL_EVENT_WRITE) ev.events |= (uint32_t)EPOLLOUT;
    ev.data.fd = fd;

    const int op = (prev_events == 0) ? EPOLL_CTL_ADD : EPOLL_CTL_MOD;
    if (epoll_ctl((int)loop->backend_fd, op, fd, &ev) == 0)
        return ESHKOL_EVENT_LOOP_OK;

    /* Self-heal a disagreement between the kernel's registration set and ours;
     * see note 3 in the file header. */
    const int first_errno = errno;
    if (op == EPOLL_CTL_ADD && first_errno == EEXIST) {
        if (epoll_ctl((int)loop->backend_fd, EPOLL_CTL_MOD, fd, &ev) == 0)
            return ESHKOL_EVENT_LOOP_OK;
        return ep_map_errno(loop, errno);
    }
    if (op == EPOLL_CTL_MOD && first_errno == ENOENT) {
        if (epoll_ctl((int)loop->backend_fd, EPOLL_CTL_ADD, fd, &ev) == 0)
            return ESHKOL_EVENT_LOOP_OK;
        return ep_map_errno(loop, errno);
    }
    return ep_map_errno(loop, first_errno);
}

int eshkol_event_backend_remove(eshkol_event_loop_t* loop, int fd,
                                int prev_events) {
    (void)prev_events; /* epoll deletes the descriptor, not per-filter state. */

    if (epoll_ctl((int)loop->backend_fd, EPOLL_CTL_DEL, fd, NULL) == 0)
        return ESHKOL_EVENT_LOOP_OK;

    /* Already gone is the state we wanted. */
    if (errno == ENOENT || errno == EBADF) return ESHKOL_EVENT_LOOP_OK;
    return ep_map_errno(loop, errno);
}

int eshkol_event_backend_poll(eshkol_event_loop_t* loop, int timeout_ms,
                              int* n_out) {
    struct epoll_event* scratch = (struct epoll_event*)loop->backend_data;

    /* epoll_wait already uses a negative timeout for "block indefinitely",
     * which is the same convention the public API exposes. */
    const int n = epoll_wait((int)loop->backend_fd, scratch, loop->max_events,
                             timeout_ms < 0 ? -1 : timeout_ms);
    if (n < 0) {
        /* Interrupted by a signal is a timeout, not a failure — same
         * convention as the kqueue backend and eshkol_poll(). */
        if (errno == EINTR) return ESHKOL_EVENT_LOOP_OK;
        loop->last_os_error = errno;
        return ESHKOL_EVENT_LOOP_EOSERR;
    }

    for (int i = 0; i < n; ++i) {
        const uint32_t got = scratch[i].events;
        int bits = 0;

        if (got & (uint32_t)(EPOLLIN | EPOLLPRI)) bits |= ESHKOL_EVENT_READ;
        if (got & (uint32_t)EPOLLOUT)             bits |= ESHKOL_EVENT_WRITE;

        /* Hangup rides alongside any readiness bit rather than replacing it, so
         * a reader still drains buffered bytes before seeing EOF. This matches
         * the kqueue backend's EV_EOF handling exactly. */
        if (got & (uint32_t)(EPOLLERR | EPOLLHUP)) bits |= ESHKOL_EVENT_ERROR;
#ifdef EPOLLRDHUP
        if (got & (uint32_t)EPOLLRDHUP) bits |= ESHKOL_EVENT_ERROR;
#endif

        if (bits) (void)eshkol_event_emit(loop, n_out, scratch[i].data.fd, bits);
    }

    return ESHKOL_EVENT_LOOP_OK;
}

void eshkol_event_backend_close(eshkol_event_loop_t* loop) {
    if (loop->backend_fd >= 0) {
        close((int)loop->backend_fd);
        loop->backend_fd = -1;
    }
    free(loop->backend_data);
    loop->backend_data = NULL;
}
