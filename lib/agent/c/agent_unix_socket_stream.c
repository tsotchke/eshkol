/* SPDX-License-Identifier: MIT */
#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#include "eshkol/core/unix_socket_stream.h"
#include "eshkol/core/event_loop.h"
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
extern int eshkol_capability_runtime_allows(const char *) __attribute__((weak_import));
#else
extern int eshkol_capability_runtime_allows(const char *) __attribute__((weak));
#endif
static _Thread_local int socket_os_error, socket_path_error;
int32_t eshkol_unix_stream_socket_last_os_error(void) { return socket_os_error; }
int32_t eshkol_unix_stream_socket_last_path_error(void) { return socket_path_error; }
void eshkol_unix_stream_socket_clear_os_error(void) { socket_os_error = socket_path_error = 0; }
static int outcome(int error) {
    socket_os_error = error;
    switch (error) {
    case EINTR: case EAGAIN: return ESHKOL_IO_TIMEOUT;
#if EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK: return ESHKOL_IO_TIMEOUT;
#endif
    case EPIPE: case ECONNRESET: case ECONNABORTED: return ESHKOL_IO_RESET;
    case ECONNREFUSED: return ESHKOL_IO_REFUSED;
    case ETIMEDOUT: return ESHKOL_IO_TIMEOUT;
    default: return ESHKOL_IO_SYSTEM;
    }
}
static int admitted(void) {
    eshkol_unix_stream_socket_clear_os_error();
    if (!eshkol_capability_runtime_allows || !eshkol_capability_runtime_allows("network")) {
        socket_os_error = EACCES; return ESHKOL_IO_SYSTEM;
    }
    return 0;
}
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__) && !defined(ESHKOL_VM_WASM)
#include <fcntl.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

enum { UNIX_LISTENER = 8, UNIX_STREAM = 9 };
#define UNIX_MASK ((UINT32_C(1) << UNIX_LISTENER) | (UINT32_C(1) << UNIX_STREAM))
#define MASK(tag) (UINT32_C(1) << (tag))
typedef struct {
    int fd, directory_fd, listener, connected, retry_connect, close_error, path_failures, path_error;
    int64_t loop, deadline;
    struct sockaddr_un address;
    dev_t device;
    ino_t inode;
    char basename[sizeof(((struct sockaddr_un *)0)->sun_path)];
} UnixSocket;

static int valid_path(const char *path, struct sockaddr_un *address) {
    if (!path || path[0] != '/' || strlen(path) >= sizeof(address->sun_path)) return 0;
    memset(address, 0, sizeof(*address)); address->sun_family = AF_UNIX;
    strcpy(address->sun_path, path); return 1;
}
static int private_parent(UnixSocket *s, const char *path) {
    char parent[sizeof(s->address.sun_path)]; strcpy(parent, path);
    char *slash = strrchr(parent, '/');
    if (!slash || !slash[1]) return ESHKOL_IO_INVALID;
    strcpy(s->basename, slash + 1);
    if (slash == parent) slash[1] = 0; else *slash = 0;
    s->directory_fd = open(parent, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (s->directory_fd < 0) return outcome(errno);
    struct stat directory;
    if (fstat(s->directory_fd, &directory)) return outcome(errno);
    if (directory.st_uid != geteuid() || (directory.st_mode & 0777) != 0700)
        return outcome(EACCES);
    return 0;
}
static int prepare(int fd) {
#if !defined(__linux__) || !defined(SOCK_CLOEXEC) || !defined(SOCK_NONBLOCK)
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0 ||
        fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) return -1;
#endif
#ifdef SO_NOSIGPIPE
    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one))) return -1;
#endif
    (void)fd;
    return 0;
}
static int new_socket(void) {
#if defined(__linux__) && defined(SOCK_CLOEXEC) && defined(SOCK_NONBLOCK)
    return socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
#else
    return socket(AF_UNIX, SOCK_STREAM, 0);
#endif
}
static int accept_socket(int listener) {
#if defined(__linux__) && defined(SOCK_CLOEXEC) && defined(SOCK_NONBLOCK)
    return accept4(listener, NULL, NULL, SOCK_CLOEXEC | SOCK_NONBLOCK);
#else
    return accept(listener, NULL, NULL);
#endif
}
static int cleanup(uintptr_t native, void *data, eshkol_resource_cleanup_result_t *result) {
    (void)data; UnixSocket *s = (UnixSocket *)native;
    /* Common waits remove their own watches before releasing the borrow. */
    int error = s->close_error, stage = error ? ESHKOL_CLEANUP_DESCRIPTOR : 0;
    if (s->fd >= 0) {
        int fd = s->fd; s->fd = -1;
        if (close(fd)) { error = s->close_error = errno; stage = ESHKOL_CLEANUP_DESCRIPTOR; }
    }
    if (s->listener && s->directory_fd >= 0) {
        struct stat st;
        int path_error = 0;
        if (fstatat(s->directory_fd, s->basename, &st, AT_SYMLINK_NOFOLLOW) == 0) {
            if (!S_ISSOCK(st.st_mode) || st.st_dev != s->device || st.st_ino != s->inode)
                path_error = ESTALE;
            else if (unlinkat(s->directory_fd, s->basename, 0))
                path_error = errno;
        } else if (errno != ENOENT) {
            path_error = errno;
        }
        s->path_error = path_error;
        /* Keep the earlier descriptor error as the primary outcome. Retryable
         * below describes remaining pathname work, never another close of fd:
         * its uncertain close has already retired it to -1. */
        if (path_error && !error) { error = path_error; stage = ESHKOL_CLEANUP_PATH; }
        /* The callback owns terminality: allow one explicit repair/retry, then
         * surrender the remaining pathname with a terminal result. The shared
         * layer neither times out nor silently retries this decision. */
        if (path_error && path_error != ESTALE && ++s->path_failures < 2) {
            result->state = ESHKOL_CLEANUP_FAILED; result->stage = stage;
            result->os_error = error; result->retryable = 1; return ESHKOL_IO_SYSTEM;
        }
    }
    if (s->directory_fd >= 0) {
        int fd = s->directory_fd; s->directory_fd = -1;
        if (close(fd) && !error) { error = errno; stage = ESHKOL_CLEANUP_DESCRIPTOR; }
    }
    free(s);
    if (error) {
        result->state = ESHKOL_CLEANUP_FAILED; result->stage = stage;
        result->os_error = error; result->retryable = 0; return ESHKOL_IO_SYSTEM;
    }
    return 0;
}
static void rollback(UnixSocket *s) {
    eshkol_resource_cleanup_result_t ignored = {0};
    /* Unpublished setup has no watches or outstanding borrower. */
    if (cleanup((uintptr_t)s, NULL, &ignored) && ignored.retryable) {
        /* The caller still has its supplied path. Report failed unpublished
         * removal separately from the primary acquisition error; it may repair
         * that path explicitly, without gaining authority over a replacement. */
        socket_path_error = s->path_error;
        if (s->directory_fd >= 0) close(s->directory_fd);
        free(s);
    }
}
static int64_t adopt(UnixSocket *s, int tag) {
    int64_t id = eshkol_resource_adopt_on_loop(ESHKOL_HANDLE_KIND_SOCKET, tag,
        (uintptr_t)s, cleanup, NULL, s->loop, 1);
    if (id < 0) rollback(s);
    return id;
}
static int borrow(int64_t id, uint32_t mask, eshkol_resource_lease_t *lease, UnixSocket **s) {
    int rc = admitted(); if (rc) return rc;
    rc = eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, mask, lease);
    if (rc) return rc;
    *s = (UnixSocket *)lease->native;
    rc = eshkol_resource_check(lease);
    if (rc) eshkol_resource_release(lease);
    return rc;
}
static int64_t done(eshkol_resource_lease_t *lease, int64_t result) {
    eshkol_resource_release(lease); return result;
}
int64_t eshkol_unix_stream_socket_listen(int64_t loop, const char *path) {
    int rc = admitted(); if (rc) return rc;
    UnixSocket *s = calloc(1, sizeof(*s)); if (!s) return outcome(ENOMEM);
    s->fd = s->directory_fd = -1; s->loop = loop;
    if (!valid_path(path, &s->address) || !eshkol_event_loop_from_handle(loop)) { free(s); return ESHKOL_IO_INVALID; }
    rc = private_parent(s, path);
    if (rc) { int e = socket_os_error; rollback(s); socket_os_error = e; return rc; }
    struct stat entry;
    if (fstatat(s->directory_fd, s->basename, &entry, AT_SYMLINK_NOFOLLOW) == 0) { rollback(s); return outcome(EEXIST); }
    if (errno != ENOENT) { int e = errno; rollback(s); return outcome(e); }
    s->fd = new_socket();
    if (s->fd < 0) { int e = errno; rollback(s); return outcome(e); }
    if (prepare(s->fd) || bind(s->fd, (struct sockaddr *)&s->address, sizeof(s->address))) {
        int e = errno; rollback(s); return outcome(e == EADDRINUSE ? EEXIST : e);
    }
    if (fstatat(s->directory_fd, s->basename, &entry, AT_SYMLINK_NOFOLLOW)) {
        int e = errno; if (e != ENOENT) socket_path_error = e;
        rollback(s); return outcome(e);
    }
    s->listener = 1; s->device = entry.st_dev; s->inode = entry.st_ino;
    /* The verified 0700 parent protects the bind-to-chmod interval; no process umask change. */
    if (fchmodat(s->directory_fd, s->basename, 0600, 0) || listen(s->fd, 64)) {
        int e = errno; rollback(s); return outcome(e);
    }
    return adopt(s, UNIX_LISTENER);
}
int64_t eshkol_unix_stream_socket_connect_begin(int64_t loop, const char *path, int32_t timeout) {
    int rc = admitted(); if (rc) return rc;
    if (timeout < 0 || !eshkol_event_loop_from_handle(loop)) return ESHKOL_IO_INVALID;
    UnixSocket *s = calloc(1, sizeof(*s)); if (!s) return outcome(ENOMEM);
    s->fd = s->directory_fd = -1; s->loop = loop;
    if (!valid_path(path, &s->address)) { free(s); return ESHKOL_IO_INVALID; }
    rc = private_parent(s, path);
    if (rc) { int e = socket_os_error; rollback(s); socket_os_error = e; return rc; }
    s->deadline = eshkol_resource_deadline(timeout);
    if (s->deadline < 0) { rollback(s); return outcome(EIO); }
    s->fd = new_socket();
    if (s->fd < 0) { int e = errno; rollback(s); return outcome(e); }
    if (prepare(s->fd)) { int e = errno; rollback(s); return outcome(e); }
    rc = connect(s->fd, (struct sockaddr *)&s->address, sizeof(s->address));
    if (rc == 0) s->connected = 1;
    else {
        int e = errno;
        if (e != EINPROGRESS && e != EAGAIN && e != EWOULDBLOCK && e != EINTR) { rollback(s); return outcome(e); }
        if (timeout == 0) { rollback(s); return ESHKOL_IO_TIMEOUT; }
        s->retry_connect = (e == EAGAIN || e == EWOULDBLOCK || e == EINTR);
    }
    return adopt(s, UNIX_STREAM);
}
static int connect_backoff(eshkol_resource_lease_t *lease, int64_t deadline, int *delay) {
    int rc = eshkol_resource_check(lease); if (rc) return rc;
    int32_t left = eshkol_resource_remaining(deadline);
    if (left <= 0) return left < 0 ? outcome(EIO) : ESHKOL_IO_TIMEOUT;
    int wait = *delay < left ? *delay : left;
    struct timespec pause = {0, wait * 1000000L};
    /* An unconnected AF_UNIX socket can be writable while its target backlog
     * remains full. Sleep instead of polling that readiness in a busy loop.
     * Eight milliseconds bounds cancellation/accepted-close response latency. */
    while (nanosleep(&pause, &pause)) {
        if (errno != EINTR) return outcome(errno);
        rc = eshkol_resource_check(lease); if (rc) return rc;
    }
    if (*delay < 8) *delay *= 2;
    return eshkol_resource_check(lease);
}
int32_t eshkol_unix_stream_socket_connect_finish(int64_t id) {
    eshkol_resource_lease_t lease; UnixSocket *s;
    int rc = borrow(id, MASK(UNIX_STREAM), &lease, &s); if (rc) return rc;
    int delay = 1;
    while (!s->connected) {
        rc = eshkol_resource_check(&lease); if (rc) break;
        int32_t left = eshkol_resource_remaining(s->deadline);
        if (left <= 0) { rc = left < 0 ? outcome(EIO) : ESHKOL_IO_TIMEOUT; break; }
        if (s->retry_connect) {
            if (connect(s->fd, (struct sockaddr *)&s->address, sizeof(s->address)) == 0 || errno == EISCONN) { s->connected = 1; break; }
            int e = errno;
            if (e != EAGAIN && e != EWOULDBLOCK && e != EINPROGRESS && e != EINTR && e != EALREADY) { rc = outcome(e); break; }
            s->retry_connect = (e == EAGAIN || e == EWOULDBLOCK || e == EINTR);
            if (s->retry_connect) {
                rc = connect_backoff(&lease, s->deadline, &delay);
                if (rc) break;
                continue;
            }
        }
        rc = eshkol_resource_wait_on_loop(&lease, s->fd, ESHKOL_EVENT_WRITE, s->deadline);
        if (rc < 0) break;
        if (!s->retry_connect) {
            int e = 0; socklen_t n = sizeof(e);
            if (getsockopt(s->fd, SOL_SOCKET, SO_ERROR, &e, &n)) { rc = outcome(errno); break; }
            if (e) { rc = outcome(e); break; }
            struct sockaddr_un peer; n = sizeof(peer);
            if (getpeername(s->fd, (struct sockaddr *)&peer, &n) == 0) s->connected = 1;
        }
        if (!s->connected) {
            rc = connect_backoff(&lease, s->deadline, &delay);
            if (rc) break;
        }
    }
    return (int32_t)done(&lease, s->connected ? 0 : rc);
}
int64_t eshkol_unix_stream_socket_accept(int64_t id, int32_t timeout) {
    if (timeout < 0) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; UnixSocket *listener;
    int rc = borrow(id, MASK(UNIX_LISTENER), &lease, &listener); if (rc) return rc;
    int64_t deadline = eshkol_resource_deadline(timeout);
    if (deadline < 0) return done(&lease, outcome(EIO));
    for (;;) {
        rc = eshkol_resource_check(&lease); if (rc) return done(&lease, rc);
        int fd = accept_socket(listener->fd);
        if (fd >= 0) {
            if (prepare(fd)) { int e = errno; close(fd); return done(&lease, outcome(e)); }
            UnixSocket *s = calloc(1, sizeof(*s));
            if (!s) { close(fd); return done(&lease, outcome(ENOMEM)); }
            s->fd = fd; s->directory_fd = -1; s->loop = listener->loop; s->connected = 1;
            int64_t peer = adopt(s, UNIX_STREAM); return done(&lease, peer);
        }
        int e = errno;
        if (e != EAGAIN && e != EWOULDBLOCK && e != EINTR) return done(&lease, outcome(e));
        rc = eshkol_resource_wait_on_loop(&lease, listener->fd, ESHKOL_EVENT_READ, deadline);
        if (rc < 0) return done(&lease, rc);
    }
}
int64_t eshkol_unix_stream_socket_read(int64_t id, void *buffer, int64_t count) {
    if (!buffer || count <= 0 || (uint64_t)count > (uint64_t)SSIZE_MAX) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; UnixSocket *s;
    int rc = borrow(id, MASK(UNIX_STREAM), &lease, &s); if (rc) return rc;
    if (!s->connected) return done(&lease, ESHKOL_IO_INVALID);
    ssize_t n = recv(s->fd, buffer, (size_t)count, 0); int e = errno;
    return done(&lease, n < 0 ? outcome(e) : n);
}
int64_t eshkol_unix_stream_socket_write(int64_t id, const void *buffer, int64_t count) {
    if (!buffer || count <= 0 || (uint64_t)count > (uint64_t)SSIZE_MAX) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; UnixSocket *s;
    int rc = borrow(id, MASK(UNIX_STREAM), &lease, &s); if (rc) return rc;
    if (!s->connected) return done(&lease, ESHKOL_IO_INVALID);
    int flags = 0;
#ifdef MSG_NOSIGNAL
    flags = MSG_NOSIGNAL;
#endif
    ssize_t n = send(s->fd, buffer, (size_t)count, flags); int e = errno;
    return done(&lease, n < 0 ? outcome(e) : n);
}
int32_t eshkol_unix_stream_socket_poll(int64_t id, int32_t interests, int32_t timeout) {
    if (interests <= 0 || (interests & ~3) || timeout < 0) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; UnixSocket *s;
    int rc = borrow(id, UNIX_MASK, &lease, &s); if (rc) return rc;
    int64_t deadline = eshkol_resource_deadline(timeout);
    if (deadline < 0) return (int32_t)done(&lease, outcome(EIO));
    if (!s->listener && !s->connected) {
        int32_t left = eshkol_resource_remaining(s->deadline);
        if (left <= 0) return (int32_t)done(&lease, left < 0 ? outcome(EIO) : ESHKOL_IO_TIMEOUT);
        if (s->deadline < deadline) deadline = s->deadline;
    }
    rc = eshkol_resource_wait_on_loop(&lease, s->fd, interests, deadline);
    return (int32_t)done(&lease, rc);
}
#else
int64_t eshkol_unix_stream_socket_listen(int64_t loop,const char *path) { (void)loop;(void)path;return ESHKOL_IO_UNSUPPORTED; }
int64_t eshkol_unix_stream_socket_accept(int64_t id,int32_t timeout) { (void)id;(void)timeout;return ESHKOL_IO_UNSUPPORTED; }
int64_t eshkol_unix_stream_socket_connect_begin(int64_t loop,const char *path,int32_t timeout) { (void)loop;(void)path;(void)timeout;return ESHKOL_IO_UNSUPPORTED; }
int32_t eshkol_unix_stream_socket_connect_finish(int64_t id) { (void)id;return ESHKOL_IO_UNSUPPORTED; }
int64_t eshkol_unix_stream_socket_read(int64_t id,void *b,int64_t n) { (void)id;(void)b;(void)n;return ESHKOL_IO_UNSUPPORTED; }
int64_t eshkol_unix_stream_socket_write(int64_t id,const void *b,int64_t n) { (void)id;(void)b;(void)n;return ESHKOL_IO_UNSUPPORTED; }
int32_t eshkol_unix_stream_socket_poll(int64_t id,int32_t events,int32_t timeout) { (void)id;(void)events;(void)timeout;return ESHKOL_IO_UNSUPPORTED; }
#define UNIX_MASK ((UINT32_C(1)<<8)|(UINT32_C(1)<<9))
#endif
int32_t eshkol_unix_stream_socket_cancel(int64_t id) {
    eshkol_unix_stream_socket_clear_os_error(); return eshkol_resource_cancel(id, ESHKOL_HANDLE_KIND_SOCKET, UNIX_MASK);
}
int32_t eshkol_unix_stream_socket_close(int64_t id) {
    eshkol_unix_stream_socket_clear_os_error(); return eshkol_resource_close(id, ESHKOL_HANDLE_KIND_SOCKET, UNIX_MASK);
}
int64_t eshkol_unix_stream_socket_close_request(int64_t id) {
    eshkol_unix_stream_socket_clear_os_error(); return eshkol_resource_close_request(id, ESHKOL_HANDLE_KIND_SOCKET, UNIX_MASK);
}
