/* SPDX-License-Identifier: MIT */
/* Darwin-only private-path pending-connect control. Loaded solely by the
 * three pending-control cases; the observation DSO never interposes connect. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#if defined(__APPLE__)
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
    /* dyld does not interpose the original reference in this image itself. */
    return connect(fd, address, length);
}
__attribute__((used, section("__DATA,__interpose")))
static const struct { const void *replacement; const void *original; } connect_interpose = {
    (const void *)fixture_connect, (const void *)connect
};
#endif
