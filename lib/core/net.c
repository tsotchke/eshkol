/* Numeric-address networking. All kernel sockets are nonblocking; poll owns
 * each deadline. Opaque monotonically increasing handles prevent a stale
 * handle from closing a newly reused OS descriptor. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "eshkol/core/net.h"
#include <limits.h>
#include <stddef.h>
#include <string.h>
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM) && !defined(__EMSCRIPTEN__)
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define NET_SLOTS 256
#define NET_MAX_BYTES (1024 * 1024)
/* Handles remain exact even across numeric FFI paths using binary64. */
#define NET_LAST_ID INT64_C(9007199254740991)
typedef struct { int fd, kind, refs, closed, nonblocking; int64_t id; } Net;
static Net slots[NET_SLOTS];
static int64_t next_id = 1;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int64_t now_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) return -1;
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
static int wait_fd(int fd, short events, int64_t deadline) {
    for (;;) {
        int64_t current = now_ms();
        if (current < 0) return -1;
        int64_t remaining = deadline - current;
        struct pollfd p = {fd, events, 0};
        int r = poll(&p, 1, remaining > 0 ? (int)remaining : 0);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return -1;
        if (!r) return -2;
        if (p.revents & POLLNVAL) return -1;
        /* HUP/ERR must reach recv/send/getsockopt for EOF or exact failure. */
        return 0;
    }
}
static int address(const char *host, int port, struct sockaddr_storage *out, socklen_t *len) {
    if (!host || port < 0 || port > 65535) return -4;
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
    return -4;
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
static int64_t register_fd(int fd, int kind) {
    int64_t result = -1;
    pthread_mutex_lock(&lock);
    if (next_id <= NET_LAST_ID) for (int i = 0; i < NET_SLOTS; i++) {
        if (!slots[i].id) {
            slots[i] = (Net){fd, kind, 0, 0, 0, next_id++};
            result = slots[i].id; break;
        }
    }
    pthread_mutex_unlock(&lock);
    if (result < 0) close(fd);
    return result;
}
static Net *acquire(int64_t id, int kind, int timeout, int64_t *deadline) {
    Net *n = NULL;
    if (deadline) *deadline = 0;
    if (id <= 0 || timeout < 0) return NULL;
    pthread_mutex_lock(&lock);
    for (int i = 0; i < NET_SLOTS; i++) if (slots[i].id == id && !slots[i].closed) {
        if (!kind || slots[i].kind == kind) {
            int64_t current = deadline ? now_ms() : 0;
            if (current < 0) { *deadline = -1; break; }
            n = &slots[i]; n->refs++;
            if (deadline) *deadline = current + (n->nonblocking ? 0 : timeout);
        }
        break;
    }
    pthread_mutex_unlock(&lock);
    return n;
}
static void release(Net *n) {
    pthread_mutex_lock(&lock);
    n->refs--;
    if (n->closed && !n->refs) { close(n->fd); memset(n, 0, sizeof(*n)); }
    pthread_mutex_unlock(&lock);
}
static int again(void) { return errno == EAGAIN || errno == EWOULDBLOCK; }
int64_t eshkol_net_open(const char *host, int32_t port, int32_t kind, int32_t timeout) {
    struct sockaddr_storage a; socklen_t len;
    if (kind < 1 || kind > 3 || timeout < 0) return -4;
    int family = address(host, port, &a, &len);
    if (family < 0) return family;
    int fd = socket(family, kind == 3 ? SOCK_DGRAM : SOCK_STREAM, 0);
    if (fd < 0) return -1;
    if (prepare(fd)) { close(fd); return -1; }
    int result = 0;
    if (kind == 2) {
        if (connect(fd, (struct sockaddr *)&a, len) < 0) {
            if (errno != EINPROGRESS) result = -1;
            else {
                int64_t current = now_ms();
                result = current < 0 ? -1 : wait_fd(fd, POLLOUT, current + timeout);
                if (!result) {
                    int error = 0; socklen_t size = sizeof(error);
                    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) || error) result = -1;
                }
            }
        }
    } else {
        if (bind(fd, (struct sockaddr *)&a, len) || (kind == 1 && listen(fd, 64))) result = -1;
    }
    if (result) { close(fd); return result; }
    return register_fd(fd, kind);
}
int64_t eshkol_net_accept(int64_t h, int32_t timeout) {
    int64_t deadline; Net *n = acquire(h, 1, timeout, &deadline);
    if (!n) return deadline < 0 ? -1 : -4;
    int fd = -1, result = 0;
    for (;;) {
        fd = accept(n->fd, NULL, NULL);
        if (fd >= 0) break;
        if (errno == EINTR) {
            int64_t current = now_ms();
            if (current < 0) { result = -1; break; }
            if (current >= deadline) { result = -2; break; }
            continue;
        }
        if (!again()) { result = -1; break; }
        if ((result = wait_fd(n->fd, POLLIN, deadline))) break;
    }
    release(n);
    if (result) return result;
    if (prepare(fd)) { close(fd); return -1; }
    return register_fd(fd, 2);
}
static int transfer(int64_t h, void *buf, int32_t length, int32_t timeout, int writing) {
    if (!buf || length < 0 || length > NET_MAX_BYTES || (!writing && !length)) return -4;
    int64_t deadline; Net *n = acquire(h, 2, timeout, &deadline);
    if (!n) return deadline < 0 ? -1 : -4;
    int result;
    for (;;) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        if (writing) flags = MSG_NOSIGNAL;
#endif
        ssize_t count = writing ? send(n->fd, buf, (size_t)length, flags) : recv(n->fd, buf, (size_t)length, 0);
        if (count >= 0) { result = (int)count; break; }
        if (errno == EINTR) {
            int64_t current = now_ms();
            if (current < 0) { result = -1; break; }
            if (current >= deadline) { result = -2; break; }
            continue;
        }
        if (!again()) { result = -1; break; }
        if ((result = wait_fd(n->fd, writing ? POLLOUT : POLLIN, deadline))) break;
    }
    release(n); return result;
}
int32_t eshkol_net_read(int64_t h, void *buf, int32_t cap, int32_t timeout) {
    return transfer(h, buf, cap, timeout, 0);
}
int32_t eshkol_net_write(int64_t h, const void *buf, int32_t len, int32_t timeout) {
    return transfer(h, (void *)buf, len, timeout, 1);
}
int32_t eshkol_net_sendto(int64_t h, const char *host, int32_t port, const void *buf, int32_t len, int32_t timeout) {
    if (!buf || len < 0 || len > 65507) return -4;
    struct sockaddr_storage a; socklen_t size;
    if (address(host, port, &a, &size) < 0) return -4;
    int64_t deadline; Net *n = acquire(h, 3, timeout, &deadline);
    if (!n) return deadline < 0 ? -1 : -4;
    int result;
    for (;;) {
        ssize_t count = sendto(n->fd, buf, (size_t)len, 0, (struct sockaddr *)&a, size);
        if (count >= 0) { result = (int)count; break; }
        if (errno == EINTR) {
            int64_t current = now_ms();
            if (current < 0) { result = -1; break; }
            if (current >= deadline) { result = -2; break; }
            continue;
        }
        if (!again()) { result = -1; break; }
        if ((result = wait_fd(n->fd, POLLOUT, deadline))) break;
    }
    release(n); return result;
}
static int render(const struct sockaddr_storage *a, char *buf, int cap, int with_port) {
    char host[INET6_ADDRSTRLEN]; const void *src; int port;
    if (a->ss_family == AF_INET) {
        const struct sockaddr_in *v = (const struct sockaddr_in *)a;
        src = &v->sin_addr; port = ntohs(v->sin_port);
    } else if (a->ss_family == AF_INET6) {
        const struct sockaddr_in6 *v = (const struct sockaddr_in6 *)a;
        src = &v->sin6_addr; port = ntohs(v->sin6_port);
    } else return -1;
    if (!inet_ntop(a->ss_family, src, host, sizeof(host))) return -1;
    int length = with_port ? snprintf(buf, (size_t)cap, a->ss_family == AF_INET6 ? "[%s]:%d" : "%s:%d", host, port)
                           : snprintf(buf, (size_t)cap, "%s", host);
    return length >= cap ? -4 : length;
}
int32_t eshkol_net_recvfrom(int64_t h, void *buf, int32_t cap, char *peer, int32_t peer_cap, int32_t timeout) {
    if (!buf || cap <= 0 || cap > 65535 || !peer || peer_cap < 64) return -4;
    int64_t deadline; Net *n = acquire(h, 3, timeout, &deadline);
    if (!n) return deadline < 0 ? -1 : -4;
    int result;
    for (;;) {
        struct sockaddr_storage a;
        struct iovec iov = {buf, (size_t)cap};
        struct msghdr msg; memset(&msg, 0, sizeof(msg));
        msg.msg_name = &a; msg.msg_namelen = sizeof(a); msg.msg_iov = &iov; msg.msg_iovlen = 1;
        ssize_t count = recvmsg(n->fd, &msg, 0);
        if (count >= 0) {
            result = (msg.msg_flags & MSG_TRUNC) ? -4 : (int)count;
            if (result >= 0 && render(&a, peer, peer_cap, 1) < 0) result = -1;
            break;
        }
        if (errno == EINTR) {
            int64_t current = now_ms();
            if (current < 0) { result = -1; break; }
            if (current >= deadline) { result = -2; break; }
            continue;
        }
        if (!again()) { result = -1; break; }
        if ((result = wait_fd(n->fd, POLLIN, deadline))) break;
    }
    release(n); return result;
}
static int endpoint(Net *n, int peer, struct sockaddr_storage *a) {
    socklen_t size = sizeof(*a);
    return peer ? getpeername(n->fd, (struct sockaddr *)a, &size) : getsockname(n->fd, (struct sockaddr *)a, &size);
}
int32_t eshkol_net_address(int64_t h, int32_t peer, char *buf, int32_t cap) {
    if (!buf || cap < INET6_ADDRSTRLEN || (peer != 0 && peer != 1)) return -4;
    Net *n = acquire(h, 0, 0, NULL); if (!n) return -4;
    struct sockaddr_storage a;
    int result = endpoint(n, peer, &a) ? -1 : render(&a, buf, cap, 0);
    release(n); return result;
}
int32_t eshkol_net_port(int64_t h, int32_t peer) {
    if (peer != 0 && peer != 1) return -4;
    Net *n = acquire(h, 0, 0, NULL); if (!n) return -4;
    struct sockaddr_storage a; int result = -1;
    if (!endpoint(n, peer, &a)) result = a.ss_family == AF_INET ? ntohs(((struct sockaddr_in *)&a)->sin_port)
                                                          : ntohs(((struct sockaddr_in6 *)&a)->sin6_port);
    release(n); return result;
}
int32_t eshkol_net_nonblocking(int64_t h, int32_t enabled) {
    if (enabled != 0 && enabled != 1) return -4;
    int result = -4;
    pthread_mutex_lock(&lock);
    for (int i = 0; i < NET_SLOTS; i++) if (slots[i].id == h && !slots[i].closed && h > 0) {
        slots[i].nonblocking = enabled; result = 0; break;
    }
    pthread_mutex_unlock(&lock); return result;
}
int32_t eshkol_net_close(int64_t h) {
    int result = -4;
    pthread_mutex_lock(&lock);
    for (int i = 0; i < NET_SLOTS; i++) if (slots[i].id == h && !slots[i].closed && h > 0) {
        Net *n = &slots[i]; n->closed = 1;
        shutdown(n->fd, SHUT_RDWR);
        if (!n->refs) { close(n->fd); memset(n, 0, sizeof(*n)); }
        result = 0; break;
    }
    pthread_mutex_unlock(&lock); return result;
}
#else
/* Unsupported platforms must refuse, never act as successful no-op sockets. */
#define UNUSED(x) (void)(x)
int64_t eshkol_net_open(const char*a,int32_t b,int32_t c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return -3;}
int64_t eshkol_net_accept(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return -3;}
int32_t eshkol_net_read(int64_t a,void*b,int32_t c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return -3;}
int32_t eshkol_net_write(int64_t a,const void*b,int32_t c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return -3;}
int32_t eshkol_net_sendto(int64_t a,const char*b,int32_t c,const void*d,int32_t e,int32_t f){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);UNUSED(f);return -3;}
int32_t eshkol_net_recvfrom(int64_t a,void*b,int32_t c,char*d,int32_t e,int32_t f){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);UNUSED(e);UNUSED(f);return -3;}
int32_t eshkol_net_address(int64_t a,int32_t b,char*c,int32_t d){UNUSED(a);UNUSED(b);UNUSED(c);UNUSED(d);return -3;}
int32_t eshkol_net_port(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return -3;}
int32_t eshkol_net_nonblocking(int64_t a,int32_t b){UNUSED(a);UNUSED(b);return -3;}
int32_t eshkol_net_close(int64_t a){UNUSED(a);return -3;}
#endif
