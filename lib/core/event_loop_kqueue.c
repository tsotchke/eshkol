/**
 * @file event_loop_kqueue.c
 * @brief macOS / *BSD event-loop backend built on kqueue(2) and kevent(2).
 *
 * kqueue is a native readiness multiplexer, so this backend is a direct
 * mapping with two portability details worth naming:
 *
 *  1. kqueue is *per filter*. Watching a descriptor for read and write is two
 *     registrations (EVFILT_READ, EVFILT_WRITE) and produces two independent
 *     records when both fire. epoll produces one record with both bits. The
 *     public API is the epoll shape, so every record is funnelled through
 *     eshkol_event_emit(), which coalesces on descriptor. This is why the
 *     receive array is sized 2 * max_events while the public array is
 *     max_events: the kernel may hand back one record per armed filter, and we
 *     must be able to drain them all in a single call or a descriptor could
 *     starve behind a busier one.
 *
 *  2. EV_EOF is not an error. On a pipe whose write end has closed, kqueue
 *     sets EV_EOF on the read filter *and* leaves the filter readable so the
 *     reader can still drain buffered bytes and then see the 0-byte EOF. We
 *     therefore report ESHKOL_EVENT_ERROR (the hangup bit, matching poll(2)'s
 *     POLLHUP) *in addition to* ESHKOL_EVENT_READ, never instead of it.
 *
 * The tree already uses kqueue in lib/agent/c/agent_subprocess.c (EVFILT_PROC
 * on child exit) and lib/agent/c/agent_watch.c (EVFILT_VNODE); the EINTR
 * convention here matches theirs.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#include "event_loop_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

const char* eshkol_event_backend_name(void) { return "kqueue"; }

int eshkol_event_backend_open(eshkol_event_loop_t* loop) {
    const int kq = kqueue();
    if (kq < 0) {
        loop->last_os_error = errno;
        return ESHKOL_EVENT_LOOP_EOSERR;
    }
    /* A kqueue is not inherited across fork(2) and BSD closes it on exec, but
     * set the flag explicitly so the intent survives any future porting. */
    (void)fcntl(kq, F_SETFD, FD_CLOEXEC);

    /* Room for one record per armed filter, i.e. up to two per descriptor. */
    struct kevent* scratch =
        (struct kevent*)calloc((size_t)loop->max_events * 2u, sizeof(struct kevent));
    if (!scratch) {
        close(kq);
        return ESHKOL_EVENT_LOOP_ENOMEM;
    }

    loop->backend_fd = (intptr_t)kq;
    loop->backend_data = scratch;
    return ESHKOL_EVENT_LOOP_OK;
}

/**
 * @brief Apply one EV_ADD / EV_DELETE for a single filter.
 *
 * Changes are applied one at a time rather than as a batch: a batched kevent()
 * stops at the first failing change and reports a single errno, leaving the
 * caller unable to tell which change failed or how many were applied. One call
 * per change costs a syscall and buys an unambiguous result.
 *
 * @param tolerate_enoent Treat ENOENT as success — deleting a filter that was
 *        never armed, or whose descriptor is already gone, is the state we
 *        wanted.
 */
static int kq_change(eshkol_event_loop_t* loop, int fd, int16_t filter,
                     uint16_t flags, int tolerate_enoent) {
    struct kevent change;
    EV_SET(&change, (uintptr_t)fd, filter, flags, 0, 0, NULL);

    for (;;) {
        const int rc = kevent((int)loop->backend_fd, &change, 1, NULL, 0, NULL);
        if (rc >= 0) return ESHKOL_EVENT_LOOP_OK;
        if (errno == EINTR) continue;

        loop->last_os_error = errno;
        if (tolerate_enoent && (errno == ENOENT || errno == EBADF))
            return ESHKOL_EVENT_LOOP_OK;
        /* EINVAL/ENODEV here means "this kind of object is not pollable" —
         * report it as unsupported rather than a generic OS failure so the
         * Scheme layer can say something true about why. */
        if (errno == EINVAL || errno == ENODEV)
            return ESHKOL_EVENT_LOOP_ENOTSUP;
        return ESHKOL_EVENT_LOOP_EOSERR;
    }
}

int eshkol_event_backend_add(eshkol_event_loop_t* loop, int fd, int events,
                             int prev_events) {
    const int want_read  = (events & ESHKOL_EVENT_READ)  != 0;
    const int want_write = (events & ESHKOL_EVENT_WRITE) != 0;
    const int had_read   = (prev_events & ESHKOL_EVENT_READ)  != 0;
    const int had_write  = (prev_events & ESHKOL_EVENT_WRITE) != 0;

    if (want_read) {
        const int rc = kq_change(loop, fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0);
        if (rc != ESHKOL_EVENT_LOOP_OK) return rc;
    } else if (had_read) {
        const int rc = kq_change(loop, fd, EVFILT_READ, EV_DELETE, 1);
        if (rc != ESHKOL_EVENT_LOOP_OK) return rc;
    }

    if (want_write) {
        const int rc = kq_change(loop, fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0);
        if (rc != ESHKOL_EVENT_LOOP_OK) {
            /* Roll the read filter back so a failed re-arm cannot leave the
             * loop half-registered relative to what the core recorded. */
            if (want_read && !had_read)
                (void)kq_change(loop, fd, EVFILT_READ, EV_DELETE, 1);
            return rc;
        }
    } else if (had_write) {
        const int rc = kq_change(loop, fd, EVFILT_WRITE, EV_DELETE, 1);
        if (rc != ESHKOL_EVENT_LOOP_OK) return rc;
    }

    return ESHKOL_EVENT_LOOP_OK;
}

int eshkol_event_backend_remove(eshkol_event_loop_t* loop, int fd,
                                int prev_events) {
    int result = ESHKOL_EVENT_LOOP_OK;

    if (prev_events & ESHKOL_EVENT_READ) {
        const int rc = kq_change(loop, fd, EVFILT_READ, EV_DELETE, 1);
        if (rc != ESHKOL_EVENT_LOOP_OK) result = rc;
    }
    if (prev_events & ESHKOL_EVENT_WRITE) {
        const int rc = kq_change(loop, fd, EVFILT_WRITE, EV_DELETE, 1);
        if (rc != ESHKOL_EVENT_LOOP_OK) result = rc;
    }
    return result;
}

int eshkol_event_backend_poll(eshkol_event_loop_t* loop, int timeout_ms,
                              int* n_out) {
    struct kevent* scratch = (struct kevent*)loop->backend_data;
    const int capacity = loop->max_events * 2;

    struct timespec ts;
    struct timespec* ts_ptr = NULL;
    if (timeout_ms >= 0) {
        ts.tv_sec = (time_t)(timeout_ms / 1000);
        ts.tv_nsec = (long)(timeout_ms % 1000) * 1000000L;
        ts_ptr = &ts;
    }

    const int n = kevent((int)loop->backend_fd, NULL, 0, scratch, capacity, ts_ptr);
    if (n < 0) {
        /* A signal that interrupts the wait is a timeout, not a failure —
         * matching eshkol_poll() in lib/agent/c/agent_poll.c. This is what
         * keeps an unbounded wait breakable with Ctrl-C instead of wedging. */
        if (errno == EINTR) return ESHKOL_EVENT_LOOP_OK;
        loop->last_os_error = errno;
        return ESHKOL_EVENT_LOOP_EOSERR;
    }

    for (int i = 0; i < n; ++i) {
        const int fd = (int)scratch[i].ident;
        int bits = 0;

        if (scratch[i].flags & EV_ERROR) {
            /* kqueue reports per-change failures inline, with errno in data.
             * data == 0 means "change accepted", which is not an event. */
            if (scratch[i].data == 0) continue;
            bits |= ESHKOL_EVENT_ERROR;
        } else if (scratch[i].filter == EVFILT_READ) {
            bits |= ESHKOL_EVENT_READ;
        } else if (scratch[i].filter == EVFILT_WRITE) {
            bits |= ESHKOL_EVENT_WRITE;
        }

        /* Hangup rides alongside the readiness bit; see the header comment. */
        if (scratch[i].flags & EV_EOF) bits |= ESHKOL_EVENT_ERROR;

        if (bits) (void)eshkol_event_emit(loop, n_out, fd, bits);
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
