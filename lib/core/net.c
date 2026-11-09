/* Numeric-address TCP/UDP sockets on the shared resource lease layer.
 *
 * Every socket is adopted into the process resource registry as a SOCKET-kind
 * handle tagged listener, stream or datagram (see resource_lease.h). Each
 * operation borrows the handle exclusively, performs nonblocking system calls,
 * and waits for readiness through the calling thread's event loop under one
 * monotonic deadline. Close from another thread wakes the borrower, whose
 * operation returns ESHKOL_IO_CANCELLED; the descriptor is closed exactly once
 * when that borrow ends. Identities are exact, positive and never reused, so a
 * stale handle cannot reach a newer socket that reused its descriptor. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "eshkol/core/net.h"
#include "eshkol/core/resource_lease.h"
#include <limits.h>
#include <stddef.h>
#include <string.h>
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM) && !defined(__EMSCRIPTEN__)
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#define NET_MAX_BYTES (1024 * 1024)
#define NET_MAX_DATAGRAM 65507
enum { NET_LISTENER = 1, NET_STREAM = 2, NET_DATAGRAM = 3 };
#define NET_MASK(tag) (UINT32_C(1) << (tag))
#define NET_ANY 0u
#define NET_TCP (NET_MASK(NET_LISTENER) | NET_MASK(NET_STREAM))

typedef struct { int fd; int nonblocking; } NetSocket;

static int net_close_native(eshkol_handle_kind_t kind, uintptr_t native, void *user_data) {
    (void)kind; (void)user_data;
    NetSocket *socket_record = (NetSocket *)native;
    int result = close(socket_record->fd);
    free(socket_record);
    return result == 0 || errno == EINTR ? 0 : -1;
}
static int address(const char *host, int port, struct sockaddr_storage *out, socklen_t *len) {
    if (!host || port < 0 || port > 65535) return ESHKOL_IO_INVALID;
    memset(out, 0, sizeof(*out));
    struct sockaddr_in *a4 = (struct sockaddr_in *)out;
    struct sockaddr_in6 *a6 = (struct sockaddr_in6 *)out;
    if (inet_pton(AF_INET, host, &a4->sin_addr) == 1) {
        a4->sin_family = AF_INET; a4->sin_port = htons((uint16_t)port);
        *len = sizeof(*a4); return AF_INET;
    }
    memset(out, 0, sizeof(*out));
    if (inet_pton(AF_INET6, host, &a6->sin6_addr) == 1) {
        a6->sin6_family = AF_INET6; a6->sin6_port = htons((uint16_t)port);
        *len = sizeof(*a6); return AF_INET6;
    }
    return ESHKOL_IO_INVALID;
}
static int prepare(int fd) {
    int f = fcntl(fd, F_GETFL);
    if (f < 0 || fcntl(fd, F_SETFL, f | O_NONBLOCK) < 0 ||
        fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) return -1;
#ifdef SO_NOSIGPIPE
    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one))) return -1;
#endif
    return 0;
}
/* The single acquire site for sockets: ownership of fd moves to the registry
 * on success and is closed here on failure. */
static int64_t adopt_fd(int fd, int tag) {
    NetSocket *record = (NetSocket *)calloc(1, sizeof(*record));
    if (!record) { close(fd); return ESHKOL_IO_SYSTEM; }
    record->fd = fd;
    int64_t id = eshkol_resource_adopt(ESHKOL_HANDLE_KIND_SOCKET, (uint32_t)tag,
                                       (uintptr_t)record, net_close_native, NULL);
    if (id < 0) { close(fd); free(record); }
    return id;
}
static int errno_outcome(int error) {
    switch (error) {
        case ECONNRESET: case EPIPE: case ECONNABORTED: return ESHKOL_IO_RESET;
        case ECONNREFUSED: return ESHKOL_IO_REFUSED;
        case ETIMEDOUT: return ESHKOL_IO_TIMEOUT;
        default: return ESHKOL_IO_SYSTEM;
    }
}
static int again(void) { return errno == EAGAIN || errno == EWOULDBLOCK; }
/* Borrow, then compute the operation deadline: user nonblocking mode makes
 * every wait a single readiness pass. */
static int borrow(int64_t h, uint32_t mask, int32_t timeout, eshkol_resource_lease_t *lease,
                  NetSocket **out, int64_t *deadline) {
    if (h <= 0 || timeout < 0) return ESHKOL_IO_INVALID;
    int status = eshkol_resource_borrow(h, ESHKOL_HANDLE_KIND_SOCKET, mask, lease);
    if (status) return status;
    *out = (NetSocket *)lease->native;
    if (deadline) {
        *deadline = eshkol_resource_deadline((*out)->nonblocking ? 0 : timeout);
        if (*deadline < 0) { eshkol_resource_release(lease); return ESHKOL_IO_SYSTEM; }
    }
    return 0;
}

int64_t eshkol_net_open(const char *host, int32_t port, int32_t kind, int32_t timeout) {
    struct sockaddr_storage a; socklen_t len;
    if (kind < NET_LISTENER || kind > NET_DATAGRAM || timeout < 0) return ESHKOL_IO_INVALID;
    int family = address(host, port, &a, &len);
    if (family < 0) return family;
    int64_t deadline = eshkol_resource_deadline(timeout);
    if (deadline < 0) return ESHKOL_IO_SYSTEM;
    int fd = socket(family, kind == NET_DATAGRAM ? SOCK_DGRAM : SOCK_STREAM, 0);
    if (fd < 0) return ESHKOL_IO_SYSTEM;
    if (prepare(fd)) { close(fd); return ESHKOL_IO_SYSTEM; }
    if (kind != NET_STREAM) {
        if (bind(fd, (struct sockaddr *)&a, len) || (kind == NET_LISTENER && listen(fd, 64))) {
            close(fd); return ESHKOL_IO_SYSTEM;
        }
        return adopt_fd(fd, kind);
    }
    int64_t id = adopt_fd(fd, NET_STREAM);
    if (id < 0) return id;
    eshkol_resource_lease_t lease; NetSocket *s;
    int result = borrow(id, NET_MASK(NET_STREAM), timeout, &lease, &s, NULL);
    if (result) { (void)eshkol_resource_close(id, ESHKOL_HANDLE_KIND_SOCKET, NET_ANY); return result; }
    if (connect(s->fd, (struct sockaddr *)&a, len) < 0) {
        if (errno != EINPROGRESS && errno != EINTR) result = errno_outcome(errno);
        else if (!(result = eshkol_resource_wait(&lease, s->fd, ESHKOL_RESOURCE_WRITE, deadline))) {
            int error = 0; socklen_t size = sizeof(error);
            if (getsockopt(s->fd, SOL_SOCKET, SO_ERROR, &error, &size)) result = ESHKOL_IO_SYSTEM;
            else if (error) result = errno_outcome(error);
        }
    }
    eshkol_resource_release(&lease);
    if (result) { (void)eshkol_resource_close(id, ESHKOL_HANDLE_KIND_SOCKET, NET_ANY); return result; }
    return id;
}
int64_t eshkol_net_accept(int64_t h, int32_t timeout) {
    eshkol_resource_lease_t lease; NetSocket *s; int64_t deadline;
    int result = borrow(h, NET_MASK(NET_LISTENER), timeout, &lease, &s, &deadline);
    if (result) return result;
    int fd = -1;
    for (;;) {
        fd = accept(s->fd, NULL, NULL);
        if (fd >= 0) break;
        if (errno == EINTR || errno == ECONNABORTED) continue;
        if (!again()) { result = ESHKOL_IO_SYSTEM; break; }
        if ((result = eshkol_resource_wait(&lease, s->fd, ESHKOL_RESOURCE_READ, deadline))) break;
    }
    eshkol_resource_release(&lease);
    if (result) return result;
    if (prepare(fd)) { close(fd); return ESHKOL_IO_SYSTEM; }
    return adopt_fd(fd, NET_STREAM);
}
/* One stream transfer under an active borrow: returns as soon as any bytes move. */
static int32_t stream_io(const eshkol_resource_lease_t *lease, NetSocket *s, uint8_t *buf,
                         int32_t length, int64_t deadline, int writing) {
    int32_t result;
    for (;;) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        if (writing) flags = MSG_NOSIGNAL;
#endif
        ssize_t count = writing ? send(s->fd, buf, (size_t)length, flags)
                                : recv(s->fd, buf, (size_t)length, 0);
        if (count >= 0) { result = (int32_t)count; break; }
        if (errno == EINTR) continue;
        if (!again()) { result = errno_outcome(errno); break; }
        if ((result = eshkol_resource_wait(lease, s->fd,
                writing ? ESHKOL_RESOURCE_WRITE : ESHKOL_RESOURCE_READ, deadline))) break;
    }
    return result;
}
static int32_t stream_transfer(int64_t h, uint8_t *buf, int32_t length, int32_t timeout, int writing) {
    eshkol_resource_lease_t lease; NetSocket *s; int64_t deadline;
    int result = borrow(h, NET_MASK(NET_STREAM), timeout, &lease, &s, &deadline);
    if (result) return result;
    result = stream_io(&lease, s, buf, length, deadline, writing);
    eshkol_resource_release(&lease);
    return result;
}
int32_t eshkol_net_leased_io(const eshkol_resource_lease_t *lease, void *buffer, int32_t length,
                             int64_t deadline, int32_t writing) {
    if (!lease || !lease->lease || lease->tag != NET_STREAM || !buffer || length <= 0 ||
        deadline < 0)
        return ESHKOL_IO_INVALID;
    return stream_io(lease, (NetSocket *)lease->native, (uint8_t *)buffer, length, deadline,
                     writing != 0);
}
int32_t eshkol_net_borrow_stream(int64_t handle, eshkol_resource_lease_t *lease) {
    return eshkol_resource_borrow(handle, ESHKOL_HANDLE_KIND_SOCKET, NET_MASK(NET_STREAM), lease);
}
int32_t eshkol_net_read(int64_t h, void *buf, int32_t cap, int32_t timeout) {
    if (!buf || cap <= 0 || cap > NET_MAX_BYTES) return ESHKOL_IO_INVALID;
    return stream_transfer(h, (uint8_t *)buf, cap, timeout, 0);
}
int32_t eshkol_net_write(int64_t h, const void *buf, int32_t len, int32_t timeout) {
    if (!buf || len < 0 || len > NET_MAX_BYTES) return ESHKOL_IO_INVALID;
    return stream_transfer(h, (uint8_t *)buf, len, timeout, 1);
}
int32_t eshkol_net_read_bytes(int64_t h, void *bytevector, int64_t start, int64_t count,
                              int32_t timeout) {
    uint8_t *bytes;
    if (count <= 0 || count > INT32_MAX ||
        eshkol_resource_bytes(bytevector, start, count, &bytes)) return ESHKOL_IO_INVALID;
    return stream_transfer(h, bytes, (int32_t)count, timeout, 0);
}
int32_t eshkol_net_write_bytes(int64_t h, const void *bytevector, int64_t start, int64_t count,
                               int32_t timeout) {
    uint8_t *bytes;
    if (count < 0 || count > INT32_MAX ||
        eshkol_resource_bytes(bytevector, start, count, &bytes)) return ESHKOL_IO_INVALID;
    return stream_transfer(h, bytes, (int32_t)count, timeout, 1);
}
int64_t eshkol_net_write_all_bytes(int64_t h, const void *bytevector, int64_t start,
                                   int64_t count, int32_t timeout) {
    uint8_t *bytes;
    if (count < 0 || eshkol_resource_bytes(bytevector, start, count, &bytes))
        return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; NetSocket *s; int64_t deadline;
    int result = borrow(h, NET_MASK(NET_STREAM), timeout, &lease, &s, &deadline);
    if (result) return result;
    int64_t done = 0;
    while (done < count) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        size_t chunk = (size_t)(count - done > INT32_MAX ? INT32_MAX : count - done);
        ssize_t sent = send(s->fd, bytes + done, chunk, flags);
        if (sent >= 0) { done += sent; continue; }
        if (errno == EINTR) continue;
        if (!again()) { result = errno_outcome(errno); break; }
        if ((result = eshkol_resource_wait(&lease, s->fd, ESHKOL_RESOURCE_WRITE, deadline))) break;
    }
    eshkol_resource_release(&lease);
    return result ? result : done;
}
static int32_t datagram_send(int64_t h, const char *host, int32_t port, const uint8_t *buf,
                             int32_t len, int32_t timeout) {
    struct sockaddr_storage a; socklen_t size;
    if (!buf || len < 0 || len > NET_MAX_DATAGRAM) return ESHKOL_IO_INVALID;
    if (address(host, port, &a, &size) < 0) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; NetSocket *s; int64_t deadline;
    int result = borrow(h, NET_MASK(NET_DATAGRAM), timeout, &lease, &s, &deadline);
    if (result) return result;
    for (;;) {
        ssize_t count = sendto(s->fd, buf, (size_t)len, 0, (struct sockaddr *)&a, size);
        if (count >= 0) { result = (int32_t)count; break; }
        if (errno == EINTR) continue;
        if (!again()) { result = errno_outcome(errno); break; }
        if ((result = eshkol_resource_wait(&lease, s->fd, ESHKOL_RESOURCE_WRITE, deadline))) break;
    }
    eshkol_resource_release(&lease);
    return result;
}
int32_t eshkol_net_sendto(int64_t h, const char *host, int32_t port, const void *buf, int32_t len,
                          int32_t timeout) {
    return datagram_send(h, host, port, (const uint8_t *)buf, len, timeout);
}
int32_t eshkol_net_sendto_bytes(int64_t h, const char *host, int32_t port, const void *bytevector,
                                int64_t start, int64_t count, int32_t timeout) {
    uint8_t *bytes;
    if (count < 0 || count > NET_MAX_DATAGRAM ||
        eshkol_resource_bytes(bytevector, start, count, &bytes)) return ESHKOL_IO_INVALID;
    return datagram_send(h, host, port, bytes, (int32_t)count, timeout);
}
static int render(const struct sockaddr_storage *a, char *buf, int cap, int with_port) {
    char host[INET6_ADDRSTRLEN]; const void *src; int port;
    if (a->ss_family == AF_INET) {
        const struct sockaddr_in *v = (const struct sockaddr_in *)a;
        src = &v->sin_addr; port = ntohs(v->sin_port);
    } else if (a->ss_family == AF_INET6) {
        const struct sockaddr_in6 *v = (const struct sockaddr_in6 *)a;
        src = &v->sin6_addr; port = ntohs(v->sin6_port);
    } else return ESHKOL_IO_SYSTEM;
    if (!inet_ntop(a->ss_family, src, host, sizeof(host))) return ESHKOL_IO_SYSTEM;
    int length = with_port ? snprintf(buf, (size_t)cap, a->ss_family == AF_INET6 ? "[%s]:%d" : "%s:%d", host, port)
                           : snprintf(buf, (size_t)cap, "%s", host);
    return length >= cap ? ESHKOL_IO_INVALID : length;
}
/* Receive one datagram. A datagram larger than the capacity is consumed; its
 * prefix and sender are stored and ESHKOL_IO_TRUNCATED is returned. */
static int32_t datagram_receive(int64_t h, uint8_t *buf, int32_t cap, char *peer, int32_t peer_cap,
                                int32_t timeout) {
    if (!buf || cap <= 0 || cap > 65535 || !peer || peer_cap < 64) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; NetSocket *s; int64_t deadline;
    int result = borrow(h, NET_MASK(NET_DATAGRAM), timeout, &lease, &s, &deadline);
    if (result) return result;
    for (;;) {
        struct sockaddr_storage a;
        struct iovec iov = {buf, (size_t)cap};
        struct msghdr msg; memset(&msg, 0, sizeof(msg));
        msg.msg_name = &a; msg.msg_namelen = sizeof(a); msg.msg_iov = &iov; msg.msg_iovlen = 1;
        ssize_t count = recvmsg(s->fd, &msg, 0);
        if (count >= 0) {
            result = (msg.msg_flags & MSG_TRUNC) ? ESHKOL_IO_TRUNCATED : (int32_t)count;
            int rendered = render(&a, peer, peer_cap, 1);
            if (rendered < 0) result = rendered;
            break;
        }
        if (errno == EINTR) continue;
        if (!again()) { result = errno_outcome(errno); break; }
        if ((result = eshkol_resource_wait(&lease, s->fd, ESHKOL_RESOURCE_READ, deadline))) break;
    }
    eshkol_resource_release(&lease);
    return result;
}
int32_t eshkol_net_recvfrom(int64_t h, void *buf, int32_t cap, char *peer, int32_t peer_cap,
                            int32_t timeout) {
    return datagram_receive(h, (uint8_t *)buf, cap, peer, peer_cap, timeout);
}
int32_t eshkol_net_recvfrom_bytes(int64_t h, void *bytevector, int64_t start, int64_t count,
                                  char *peer, int32_t peer_cap, int32_t timeout) {
    uint8_t *bytes;
    if (count <= 0 || count > 65535 || eshkol_resource_bytes(bytevector, start, count, &bytes))
        return ESHKOL_IO_INVALID;
    return datagram_receive(h, bytes, (int32_t)count, peer, peer_cap, timeout);
}
static int endpoint(NetSocket *s, int peer, struct sockaddr_storage *a) {
    socklen_t size = sizeof(*a);
    return peer ? getpeername(s->fd, (struct sockaddr *)a, &size)
                : getsockname(s->fd, (struct sockaddr *)a, &size);
}
int32_t eshkol_net_address(int64_t h, int32_t peer, char *buf, int32_t cap) {
    if (!buf || cap < INET6_ADDRSTRLEN || (peer != 0 && peer != 1)) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; NetSocket *s;
    int result = borrow(h, NET_ANY, 0, &lease, &s, NULL);
    if (result) return result;
    struct sockaddr_storage a;
    result = endpoint(s, peer, &a) ? errno_outcome(errno) : render(&a, buf, cap, 0);
    eshkol_resource_release(&lease);
    return result;
}
int32_t eshkol_net_port(int64_t h, int32_t peer) {
    if (peer != 0 && peer != 1) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; NetSocket *s;
    int result = borrow(h, NET_ANY, 0, &lease, &s, NULL);
    if (result) return result;
    struct sockaddr_storage a;
    if (endpoint(s, peer, &a)) result = errno_outcome(errno);
    else result = a.ss_family == AF_INET ? ntohs(((struct sockaddr_in *)&a)->sin_port)
                                         : ntohs(((struct sockaddr_in6 *)&a)->sin6_port);
    eshkol_resource_release(&lease);
    return result;
}
int32_t eshkol_net_nonblocking(int64_t h, int32_t enabled) {
    if (enabled != 0 && enabled != 1) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; NetSocket *s;
    int result = borrow(h, NET_ANY, 0, &lease, &s, NULL);
    if (result) return result;
    s->nonblocking = enabled;
    eshkol_resource_release(&lease);
    return 0;
}
int32_t eshkol_net_shutdown(int64_t h, int32_t how) {
    if (how < 0 || how > 2) return ESHKOL_IO_INVALID;
    eshkol_resource_lease_t lease; NetSocket *s;
    int result = borrow(h, NET_MASK(NET_STREAM), 0, &lease, &s, NULL);
    if (result) return result;
    static const int modes[3] = {SHUT_RD, SHUT_WR, SHUT_RDWR};
    if (shutdown(s->fd, modes[how]) != 0) result = errno_outcome(errno);
    eshkol_resource_release(&lease);
    return result;
}
int32_t eshkol_net_cancel(int64_t h) {
    return eshkol_resource_cancel(h, ESHKOL_HANDLE_KIND_SOCKET, NET_ANY);
}
int32_t eshkol_net_close_kind(int64_t h, int32_t family) {
    uint32_t mask = family == 1 ? NET_TCP : family == 2 ? NET_MASK(NET_DATAGRAM) : NET_ANY;
    if (family < 0 || family > 2) return ESHKOL_IO_INVALID;
    return eshkol_resource_close(h, ESHKOL_HANDLE_KIND_SOCKET, mask);
}
int32_t eshkol_net_close(int64_t h) { return eshkol_net_close_kind(h, 0); }
#else
/* Unsupported platforms refuse; they never act as successful no-op sockets. */
#define UNUSED(x) (void)(x)
int64_t eshkol_net_open(const char*a,int32_t b,int32_t c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return ESHKOL_IO_UNSUPPORTED;}
int64_t eshkol_net_accept(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_read(int64_t a,void*b,int32_t c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_write(int64_t a,const void*b,int32_t c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_read_bytes(int64_t a,void*b,int64_t c,int64_t d,int32_t e){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_write_bytes(int64_t a,const void*b,int64_t c,int64_t d,int32_t e){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);return ESHKOL_IO_UNSUPPORTED;}
int64_t eshkol_net_write_all_bytes(int64_t a,const void*b,int64_t c,int64_t d,int32_t e){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_sendto(int64_t a,const char*b,int32_t c,const void*d,int32_t e,int32_t f){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);UNUSED(f);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_sendto_bytes(int64_t a,const char*b,int32_t c,const void*d,int64_t e,int64_t f,int32_t g){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);UNUSED(f);UNUSED(g);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_recvfrom(int64_t a,void*b,int32_t c,char*d,int32_t e,int32_t f){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);UNUSED(f);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_recvfrom_bytes(int64_t a,void*b,int64_t c,int64_t d,char*e,int32_t f,int32_t g){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);UNUSED(f);UNUSED(g);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_address(int64_t a,int32_t b,char*c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_port(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_nonblocking(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_shutdown(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_cancel(int64_t a){UNUSED(a);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_close_kind(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_close(int64_t a){UNUSED(a);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_leased_io(const eshkol_resource_lease_t*a,void*b,int32_t c,int64_t d,int32_t e){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);return ESHKOL_IO_UNSUPPORTED;}
int32_t eshkol_net_borrow_stream(int64_t a,eshkol_resource_lease_t*b){UNUSED(a);UNUSED(b);return ESHKOL_IO_UNSUPPORTED;}
#endif
