/* SPDX-License-Identifier: MIT */
#ifndef ESHKOL_UNIX_SOCKET_TEST_DIRECTORY_H
#define ESHKOL_UNIX_SOCKET_TEST_DIRECTORY_H
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Keep the generated private directory and fixture leaf within Darwin's
 * shorter sun_path as well as Linux's. Honour a short writable absolute TMPDIR;
 * an absent, relative, inaccessible or overlong value uses the short /tmp
 * fallback. mkdtemp creates the caller-owned directory with mode 0700. */
static inline char *eshkol_test_socket_directory(char *buffer, size_t capacity,
                                                const char *prefix) {
    const char *parent = getenv("TMPDIR");
    if (!parent || parent[0] != '/' || strlen(parent) > 40 ||
        access(parent, W_OK | X_OK) != 0) parent = "/tmp";
    int length = snprintf(buffer, capacity, "%s/%sXXXXXX", parent, prefix);
    if (length < 0 || (size_t)length >= capacity) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    return mkdtemp(buffer);
}
#endif
