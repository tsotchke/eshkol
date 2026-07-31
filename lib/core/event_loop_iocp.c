/**
 * @file event_loop_iocp.c
 * @brief Windows event-loop backend: a real I/O completion port, plus an
 *        honestly-labelled readiness layer over it.
 *
 * ══════════════════════════════════════════════════════════════════════
 *  READ THIS BEFORE CHANGING ANYTHING HERE
 * ══════════════════════════════════════════════════════════════════════
 *
 * The public API (inc/eshkol/core/event_loop.h) is *readiness*-style, because
 * that is what kqueue and epoll provide and what v1.4's sockets, fibers and
 * timers are specified against: "which of these descriptors can I act on now?"
 *
 * IOCP does not answer that question. It is a *completion* model: you associate
 * a handle opened with FILE_FLAG_OVERLAPPED, you *start* an operation, and the
 * port tells you when that operation finished. There is no "is it readable?"
 * query. Adapting one to the other is a real design decision with real limits,
 * so here is exactly what this file does and does not do.
 *
 * ── What the completion port is actually used for ─────────────────────
 *
 * The loop owns a genuine port from CreateIoCompletionPort(). It is not
 * decoration:
 *
 *   • It is the *wait* primitive. Every blocking wait in this backend is a
 *     GetQueuedCompletionStatus() on that port, never a bare Sleep(). That
 *     makes the wait interruptible — a PostQueuedCompletionStatus() from
 *     another thread breaks it immediately, which is what a future
 *     `event-loop-wake!` and the v1.4 fiber scheduler need.
 *   • It is the association point v1.4's overlapped sockets will use. When
 *     real completion-mode I/O lands, those handles associate with *this*
 *     port and their completions are dequeued by the same wait, alongside the
 *     readiness scan. The foundation is here; the operations are not.
 *   • `event-loop-close` closes it, so the acceptance criterion "releases the
 *     underlying kqueue/epoll/iocp resource" is true on Windows in the literal
 *     sense: a kernel object is created and destroyed.
 *
 * ── Where readiness actually comes from ───────────────────────────────
 *
 * Readiness for the descriptor kinds that exist today is NOT derived from
 * GetQueuedCompletionStatus. It comes from two documented per-object queries,
 * chosen by classifying each descriptor once at registration time:
 *
 *   • SOCKETS → WSAPoll(). This is genuine kernel readiness with a genuine
 *     timeout. A loop watching only sockets does exactly one WSAPoll per poll
 *     call and does not spin at all.
 *   • ANONYMOUS PIPES, CONSOLES, FILES → PeekNamedPipe() / GetNamedPipeInfo()
 *     / GetNumberOfConsoleInputEvents(), which are non-consuming readiness
 *     queries, driven by a bounded exponential backoff (1→8 ms) whose sleep is
 *     the completion-port wait described above.
 *
 * The second bullet is the honest limit: for non-socket descriptors this is
 * *emulated* level-triggered readiness with latency bounded by the 8 ms backoff
 * cap, not a zero-latency kernel notification. It is not a busy-wait (the
 * thread is blocked in the kernel between probes) and it is not a lie about
 * being IOCP-driven — it is a polling loop, and this comment says so.
 *
 * ── Why not the two standard IOCP-to-readiness tricks ─────────────────
 *
 * The parent brief named both. Neither is the right call here:
 *
 *   • Zero-byte WSARecv. Works *only for sockets* — and sockets are precisely
 *     the case WSAPoll already handles correctly, simply, and with a real
 *     timeout. It buys nothing for anonymous pipes, which are the only
 *     descriptor source Eshkol has today (`make-pipe`). Against that it costs a
 *     per-descriptor pending OVERLAPPED whose cancellation on remove/close is
 *     the classic source of use-after-free in this style of code. More
 *     unverifiable kernel state for zero benefit.
 *
 *   • AFD polling (\Device\Afd + IOCTL_AFD_POLL, what libuv and Rust's mio do).
 *     This is the only way to get true IOCP-delivered readiness, and it is
 *     *undocumented* — it depends on internal NT interfaces with no stability
 *     contract. Not an acceptable dependency for a foundation primitive whose
 *     whole job is to be portable, and still sockets-only.
 *
 *   • Named pipes with FILE_FLAG_OVERLAPPED (libuv's actual `uv_pipe`) would
 *     make pipes completion-capable, but only by replacing `make-pipe`'s
 *     anonymous CreatePipe/_pipe descriptors wholesale — a change to the
 *     process-plumbing surface that belongs to the socket/async ticket, not to
 *     the ticket that introduces the loop.
 *
 * ── What CI can and cannot prove ──────────────────────────────────────
 *
 * The Windows lanes (windows-arm64-lite, windows-arm64-xla, windows-x64-cuda)
 * compile this file and run the .esk suite, so the pipe round-trip, the
 * timeout case and the close-then-use case are genuinely exercised on Windows
 * through the PeekNamedPipe path. The WSAPoll path has no coverage until v1.4
 * brings sockets, and the completion-dequeue path has no coverage until v1.4
 * issues overlapped operations. Both are stated as such in the PR rather than
 * claimed as verified.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 /* Windows 7: WSAPoll, InitOnceExecuteOnce. */
#endif
#define WIN32_LEAN_AND_MEAN

#include "event_loop_internal.h"

#include <winsock2.h>  /* must precede windows.h */
#include <ws2tcpip.h>
#include <windows.h>

#include <io.h>
#include <stdlib.h>
#include <string.h>

const char* eshkol_event_backend_name(void) { return "iocp"; }

/* ─────────────────────────────────────────────────────────────────────
 * Winsock initialisation
 *
 * WSAPoll needs Winsock started. Winsock is reference-counted, so starting it
 * once and never calling WSACleanup is correct for a process-lifetime facility
 * (the same choice lib/agent/c/agent_http.c makes); the alternative — pairing
 * cleanup with loop teardown — would tear Winsock down under any other part of
 * the runtime still holding a socket.
 * ───────────────────────────────────────────────────────────────────── */

static INIT_ONCE g_winsock_once = INIT_ONCE_STATIC_INIT;
static int       g_winsock_ok = 0;

static BOOL CALLBACK winsock_init_once(PINIT_ONCE once, PVOID param, PVOID* ctx) {
    WSADATA data;
    (void)once; (void)param; (void)ctx;
    g_winsock_ok = (WSAStartup(MAKEWORD(2, 2), &data) == 0);
    return TRUE;
}

static int winsock_ready(void) {
    InitOnceExecuteOnce(&g_winsock_once, winsock_init_once, NULL, NULL);
    return g_winsock_ok;
}

/* ─────────────────────────────────────────────────────────────────────
 * Per-descriptor classification
 *
 * Windows has no unified descriptor space: a CRT file descriptor and a SOCKET
 * are different namespaces that both arrive here as an int. Classification is
 * therefore explicit, done once at registration, and resolved in an
 * unambiguous order:
 *
 *   1. _get_osfhandle(fd) — authoritative for CRT descriptors, which is what
 *      `make-pipe` produces via _pipe(). If it yields a handle, that handle is
 *      the object.
 *   2. If that handle is itself a socket (someone _open_osfhandle'd one),
 *      getsockopt(SO_TYPE) succeeds and we treat it as a socket.
 *   3. Only if the CRT lookup fails do we consider the int a raw SOCKET.
 *
 * Probing getsockopt first would be wrong: SOCKET values are kernel handle
 * values and can collide numerically with small CRT descriptors, so a pipe
 * could be misclassified as a socket. The CRT table is checked first precisely
 * to remove that ambiguity.
 * ───────────────────────────────────────────────────────────────────── */

#define ESH_IOCP_KIND_FREE   0
#define ESH_IOCP_KIND_HANDLE 1  /* pipe / console / file — PeekNamedPipe path. */
#define ESH_IOCP_KIND_SOCKET 2  /* WSAPoll path.                              */

typedef struct {
    int32_t   fd;
    int32_t   kind;
    HANDLE    handle;  /* valid when kind == HANDLE */
    SOCKET    socket;  /* valid when kind == SOCKET */
} esh_iocp_slot_t;

static int socket_probe(SOCKET s) {
    int type = 0;
    int len = (int)sizeof(type);
    if (!winsock_ready()) return 0;
    return getsockopt(s, SOL_SOCKET, SO_TYPE, (char*)&type, &len) == 0;
}

/** Classify @p fd, filling @p slot. Returns an ESHKOL_EVENT_LOOP_* code. */
static int classify(eshkol_event_loop_t* loop, int fd, esh_iocp_slot_t* slot) {
    const intptr_t raw = _get_osfhandle(fd);

    if (raw != -1 && raw != (intptr_t)INVALID_HANDLE_VALUE) {
        if (socket_probe((SOCKET)raw)) {
            slot->kind = ESH_IOCP_KIND_SOCKET;
            slot->socket = (SOCKET)raw;
            slot->handle = INVALID_HANDLE_VALUE;
        } else {
            slot->kind = ESH_IOCP_KIND_HANDLE;
            slot->handle = (HANDLE)raw;
            slot->socket = INVALID_SOCKET;
        }
        slot->fd = fd;
        return ESHKOL_EVENT_LOOP_OK;
    }

    if (socket_probe((SOCKET)(intptr_t)fd)) {
        slot->kind = ESH_IOCP_KIND_SOCKET;
        slot->socket = (SOCKET)(intptr_t)fd;
        slot->handle = INVALID_HANDLE_VALUE;
        slot->fd = fd;
        return ESHKOL_EVENT_LOOP_OK;
    }

    loop->last_os_error = (int)GetLastError();
    return ESHKOL_EVENT_LOOP_EINVAL;
}

static esh_iocp_slot_t* slot_find(eshkol_event_loop_t* loop, int fd) {
    esh_iocp_slot_t* slots = (esh_iocp_slot_t*)loop->backend_data;
    for (int i = 0; i < loop->max_events; ++i)
        if (slots[i].kind != ESH_IOCP_KIND_FREE && slots[i].fd == fd)
            return &slots[i];
    return NULL;
}

/* ─────────────────────────────────────────────────────────────────────
 * Backend contract
 * ───────────────────────────────────────────────────────────────────── */

int eshkol_event_backend_open(eshkol_event_loop_t* loop) {
    const HANDLE port =
        CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    if (port == NULL) {
        loop->last_os_error = (int)GetLastError();
        return ESHKOL_EVENT_LOOP_EOSERR;
    }

    esh_iocp_slot_t* slots =
        (esh_iocp_slot_t*)calloc((size_t)loop->max_events, sizeof(esh_iocp_slot_t));
    if (!slots) {
        CloseHandle(port);
        return ESHKOL_EVENT_LOOP_ENOMEM;
    }

    loop->backend_fd = (intptr_t)port;
    loop->backend_data = slots;
    return ESHKOL_EVENT_LOOP_OK;
}

int eshkol_event_backend_add(eshkol_event_loop_t* loop, int fd, int events,
                             int prev_events) {
    (void)events;      /* Interest lives in the core's registration table. */
    (void)prev_events; /* Re-registration just re-classifies.             */

    esh_iocp_slot_t* slot = slot_find(loop, fd);
    if (!slot) {
        esh_iocp_slot_t* slots = (esh_iocp_slot_t*)loop->backend_data;
        for (int i = 0; i < loop->max_events; ++i) {
            if (slots[i].kind == ESH_IOCP_KIND_FREE) { slot = &slots[i]; break; }
        }
        if (!slot) return ESHKOL_EVENT_LOOP_EFULL;
    }
    return classify(loop, fd, slot);
}

int eshkol_event_backend_remove(eshkol_event_loop_t* loop, int fd,
                                int prev_events) {
    (void)prev_events;
    esh_iocp_slot_t* slot = slot_find(loop, fd);
    if (slot) memset(slot, 0, sizeof(*slot)); /* kind = FREE */
    return ESHKOL_EVENT_LOOP_OK;
}

/**
 * @brief Non-consuming readiness inspection for a non-socket handle.
 *
 * Mirrors inspect_fd() in lib/agent/c/agent_poll.c, which is the tree's
 * existing answer to the same problem — anonymous Win32 pipe handles are not
 * waitable objects, and PeekNamedPipe is the documented way to ask whether one
 * has bytes without consuming them. Keep the two in agreement.
 */
static int inspect_handle(HANDLE handle, int interest) {
    int bits = 0;
    if (handle == INVALID_HANDLE_VALUE) return ESHKOL_EVENT_ERROR;

    const DWORD type = GetFileType(handle);
    if (type == FILE_TYPE_UNKNOWN && GetLastError() != NO_ERROR)
        return ESHKOL_EVENT_ERROR;

    if (interest & ESHKOL_EVENT_READ) {
        if (type == FILE_TYPE_PIPE) {
            DWORD available = 0;
            if (PeekNamedPipe(handle, NULL, 0, NULL, &available, NULL)) {
                if (available > 0) bits |= ESHKOL_EVENT_READ;
            } else {
                /* A broken pipe is a hangup, and a hangup must also report
                 * readable so the reader observes the 0-byte EOF — the same
                 * rule the kqueue and epoll backends follow for EV_EOF and
                 * EPOLLHUP. */
                bits |= ESHKOL_EVENT_ERROR | ESHKOL_EVENT_READ;
            }
        } else if (type == FILE_TYPE_CHAR) {
            DWORD pending = 0;
            if (GetNumberOfConsoleInputEvents(handle, &pending) && pending > 0)
                bits |= ESHKOL_EVENT_READ;
        } else {
            /* Regular files are always ready for a synchronous read. Note that
             * Linux epoll refuses to watch them at all (EPERM → ENOTSUP), so
             * portable code must not depend on this. */
            bits |= ESHKOL_EVENT_READ;
        }
    }

    if (interest & ESHKOL_EVENT_WRITE) {
        if (type == FILE_TYPE_PIPE) {
            DWORD flags = 0, outbound = 0;
            if (GetNamedPipeInfo(handle, &flags, &outbound, NULL, NULL)) {
                bits |= ESHKOL_EVENT_WRITE;
            } else {
                const DWORD err = GetLastError();
                if (err == ERROR_BROKEN_PIPE || err == ERROR_PIPE_NOT_CONNECTED)
                    bits |= ESHKOL_EVENT_ERROR;
                else
                    bits |= ESHKOL_EVENT_WRITE;
            }
        } else {
            bits |= ESHKOL_EVENT_WRITE;
        }
    }

    return bits;
}

/** @return Interest bits the core has armed for @p fd, or 0 if unregistered. */
static int registered_interest(const eshkol_event_loop_t* loop, int fd) {
    for (int i = 0; i < loop->max_events; ++i)
        if (loop->regs[i].fd == fd) return loop->regs[i].events;
    return 0;
}

/**
 * @brief One non-blocking readiness sweep over every registered descriptor.
 * @param socket_timeout_ms Timeout handed to WSAPoll. Pass 0 when non-socket
 *        descriptors are also registered (they need the backoff loop, so the
 *        socket query must not consume the whole budget); pass the caller's
 *        full timeout when the set is sockets-only, which is what lets a
 *        sockets-only loop block properly in the kernel with no spinning.
 * @return Number of ready descriptors recorded, or a negative result code.
 */
static int sweep(eshkol_event_loop_t* loop, int* n_out, int socket_timeout_ms) {
    esh_iocp_slot_t* slots = (esh_iocp_slot_t*)loop->backend_data;
    WSAPOLLFD stack_pfds[64];
    WSAPOLLFD* pfds = stack_pfds;
    int pfd_fds[64];
    int* fd_map = pfd_fds;
    int n_sockets = 0;
    int recorded = 0;
    int rc = ESHKOL_EVENT_LOOP_OK;

    if (loop->max_events > 64) {
        pfds = (WSAPOLLFD*)calloc((size_t)loop->max_events, sizeof(WSAPOLLFD));
        fd_map = (int*)calloc((size_t)loop->max_events, sizeof(int));
        if (!pfds || !fd_map) {
            free(pfds); free(fd_map);
            return ESHKOL_EVENT_LOOP_ENOMEM;
        }
    }

    for (int i = 0; i < loop->max_events; ++i) {
        if (slots[i].kind == ESH_IOCP_KIND_FREE) continue;

        /* Interest is owned by the core's registration table, which is shared
         * with backends through event_loop_internal.h. A slot with no matching
         * registration means the core removed it; skip it. */
        const int want = registered_interest(loop, slots[i].fd);
        if (want == 0) continue;

        if (slots[i].kind == ESH_IOCP_KIND_SOCKET) {
            pfds[n_sockets].fd = slots[i].socket;
            pfds[n_sockets].events = 0;
            if (want & ESHKOL_EVENT_READ)  pfds[n_sockets].events |= POLLRDNORM;
            if (want & ESHKOL_EVENT_WRITE) pfds[n_sockets].events |= POLLWRNORM;
            pfds[n_sockets].revents = 0;
            fd_map[n_sockets] = slots[i].fd;
            ++n_sockets;
        } else {
            const int bits = inspect_handle(slots[i].handle, want);
            if (bits) {
                if (eshkol_event_emit(loop, n_out, slots[i].fd, bits)) ++recorded;
            }
        }
    }

    if (n_sockets > 0) {
        if (!winsock_ready()) {
            rc = ESHKOL_EVENT_LOOP_ENOTSUP;
        } else {
            const int n = WSAPoll(pfds, (ULONG)n_sockets, socket_timeout_ms);
            if (n < 0) {
                loop->last_os_error = WSAGetLastError();
                rc = ESHKOL_EVENT_LOOP_EOSERR;
            } else {
                for (int i = 0; i < n_sockets; ++i) {
                    int bits = 0;
                    const SHORT got = pfds[i].revents;
                    if (got & (POLLRDNORM | POLLRDBAND)) bits |= ESHKOL_EVENT_READ;
                    if (got & POLLWRNORM)                bits |= ESHKOL_EVENT_WRITE;
                    if (got & (POLLERR | POLLHUP | POLLNVAL)) bits |= ESHKOL_EVENT_ERROR;
                    if (bits) {
                        if (eshkol_event_emit(loop, n_out, fd_map[i], bits)) ++recorded;
                    }
                }
            }
        }
    }

    if (pfds != stack_pfds) { free(pfds); free(fd_map); }
    return rc == ESHKOL_EVENT_LOOP_OK ? recorded : rc;
}

/** @return 1 if any registered descriptor needs the polling path. */
static int has_non_socket(const eshkol_event_loop_t* loop) {
    const esh_iocp_slot_t* slots = (const esh_iocp_slot_t*)loop->backend_data;
    for (int i = 0; i < loop->max_events; ++i)
        if (slots[i].kind == ESH_IOCP_KIND_HANDLE) return 1;
    return 0;
}

int eshkol_event_backend_poll(eshkol_event_loop_t* loop, int timeout_ms,
                              int* n_out) {
    const HANDLE port = (HANDLE)loop->backend_fd;

    /* Sockets-only: one WSAPoll with the caller's real timeout. True kernel
     * readiness, no backoff, no spinning. */
    if (!has_non_socket(loop)) {
        const int rc = sweep(loop, n_out, timeout_ms < 0 ? -1 : timeout_ms);
        return rc < 0 ? rc : ESHKOL_EVENT_LOOP_OK;
    }

    /* Mixed or pipe-only: emulate level-triggered readiness with a bounded
     * backoff. See the file header for why this is the honest ceiling on
     * Windows for anonymous pipes. */
    const ULONGLONG start = GetTickCount64();
    DWORD backoff_ms = 0;

    for (;;) {
        const int rc = sweep(loop, n_out, 0);
        if (rc < 0) return rc;
        if (*n_out > 0) return ESHKOL_EVENT_LOOP_OK;
        if (timeout_ms == 0) return ESHKOL_EVENT_LOOP_OK;

        DWORD wait_ms;
        if (timeout_ms > 0) {
            const ULONGLONG elapsed = GetTickCount64() - start;
            if (elapsed >= (ULONGLONG)timeout_ms) return ESHKOL_EVENT_LOOP_OK;
            const DWORD remaining = (DWORD)((ULONGLONG)timeout_ms - elapsed);
            backoff_ms = backoff_ms == 0 ? 1 : (backoff_ms < 8 ? backoff_ms * 2 : 8);
            wait_ms = backoff_ms > remaining ? remaining : backoff_ms;
        } else {
            backoff_ms = backoff_ms == 0 ? 1 : (backoff_ms < 8 ? backoff_ms * 2 : 8);
            wait_ms = backoff_ms;
        }

        /* Block on the completion port rather than Sleep(): this is what makes
         * the wait interruptible by PostQueuedCompletionStatus and what will
         * dequeue v1.4's overlapped completions once they exist. A dequeued
         * packet simply means "re-scan now". */
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        OVERLAPPED* ov = NULL;
        if (!GetQueuedCompletionStatus(port, &bytes, &key, &ov, wait_ms)) {
            const DWORD err = GetLastError();
            if (err != WAIT_TIMEOUT && ov == NULL) {
                loop->last_os_error = (int)err;
                return ESHKOL_EVENT_LOOP_EOSERR;
            }
        }
    }
}

void eshkol_event_backend_close(eshkol_event_loop_t* loop) {
    if (loop->backend_fd != -1 && (HANDLE)loop->backend_fd != NULL) {
        CloseHandle((HANDLE)loop->backend_fd);
        loop->backend_fd = -1;
    }
    free(loop->backend_data);
    loop->backend_data = NULL;
}
