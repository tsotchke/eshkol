/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

/* Test-only observations and a private-path syscall-error control; no socket
 * ownership implementation or replacement agent archive. */
int32_t eshkol_unix_socket_test_fd_count(void) {
#if defined(__APPLE__)
    DIR *d = opendir("/dev/fd");
#else
    DIR *d = opendir("/proc/self/fd");
#endif
    if (!d) return -1;
    int own = dirfd(d), count = 0; struct dirent *e;
    while ((e = readdir(d))) {
        char *end; long fd = strtol(e->d_name, &end, 10);
        if (e->d_name[0] && !*end && fd != own) ++count;
    }
    if (closedir(d)) return -1;
    return count;
}
int64_t eshkol_unix_socket_test_monotonic_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return -1;
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
#include <errno.h>
int32_t eshkol_unix_socket_test_sleep_ms(int32_t ms) {
    if (ms < 0) return -1;
    struct timespec t = {ms / 1000, (ms % 1000) * 1000000L};
    while (nanosleep(&t, &t)) if (errno != EINTR) return -1;
    return 0;
}
int32_t eshkol_unix_socket_test_errno(int32_t which) {
    switch (which) {
        case 1: return ENOENT;
        case 2: return ECONNREFUSED;
        case 3: return EEXIST;
        case 4: return EACCES;
        case 5: return EPIPE;
        case 6: return ECONNRESET;
        default: return 0;
    }
}
#include <stddef.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#if !defined(__APPLE__)
#include <dlfcn.h>
#endif

/* Darwin reports a full Unix listen queue as immediate ECONNREFUSED. For
 * pending-connect controls only, inject EAGAIN at this one private path.
 * All other calls reach the real syscall. This is an explicit test seam,
 * not a socket implementation or evidence of an OS-pending connection. */
int32_t eshkol_unix_socket_test_pending_path(const char *path) {
#if defined(__APPLE__)
    if (!path || strncmp(path, "/tmp/eshkol-unix-", 17) != 0) return -1;
    return setenv("ESHKOL_UNIX_SOCKET_TEST_PENDING_PATH", path, 1);
#else
    (void)path;
    return 0;
#endif
}
static int fixture_connect(int fd, const struct sockaddr *address, socklen_t length) {
    const char *target = getenv("ESHKOL_UNIX_SOCKET_TEST_PENDING_PATH");
    size_t offset = offsetof(struct sockaddr_un, sun_path);
    if (target && address && length > offset && address->sa_family == AF_UNIX) {
        const struct sockaddr_un *un = (const struct sockaddr_un *)address;
        size_t n = strlen(target), available = (size_t)length - offset;
        if (n < available && n < sizeof(un->sun_path) && un->sun_path[n] == 0 &&
            memcmp(un->sun_path, target, n) == 0) {
            errno = EAGAIN;
            return -1;
        }
    }
#if defined(__APPLE__)
    /* dyld does not interpose the original reference in this image itself. */
    return connect(fd, address, length);
#else
    int (*real_connect)(int, const struct sockaddr *, socklen_t) =
        (int (*)(int, const struct sockaddr *, socklen_t))dlsym(RTLD_NEXT, "connect");
    if (!real_connect) { errno = ENOSYS; return -1; }
    return real_connect(fd, address, length);
#endif
}
#if defined(__APPLE__)
__attribute__((used, section("__DATA,__interpose")))
static const struct { const void *replacement; const void *original; } connect_interpose = {
    (const void *)fixture_connect, (const void *)connect
};
#else
int connect(int fd, const struct sockaddr *address, socklen_t length) {
    return fixture_connect(fd, address, length);
}
#endif
