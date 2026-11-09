/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include <eshkol/core/unix_socket_stream.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Default-suite check for the Unix-socket scaffold. Two things must stay
 * true while the ipc-unix-pending acceptance suite is excluded from the
 * default CTest run:
 *
 *  1. every scaffold entry point still reports the planned result for an
 *     otherwise well-formed call (so this check notices if a stub starts
 *     doing real I/O, or silently succeeding, before the core lands);
 *  2. the pending acceptance cases are actually registered with CTest under
 *     the "ipc-unix-pending" label, not merely written to disk.
 */

static unsigned int checks;

static void check(int ok, const char *expr, int line) {
    ++checks;
    if (!ok) {
        fprintf(stderr, "FAIL: unix_socket_stream_stub_contract_test:%d: %s\n", line, expr);
        exit(1);
    }
}
#define CHECK(expr) check(!!(expr), #expr, __LINE__)

static void check_stub_contract(void) {
    char buf[4] = {0};
    CHECK(eshkol_unix_stream_socket_listen(0, "/eshkol-scaffold/does-not-exist.sock")
          == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_accept(1, 0) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_connect(0, "/eshkol-scaffold/does-not-exist.sock", 0)
          == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_read(1, buf, sizeof(buf)) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_write(1, buf, sizeof(buf)) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_cancel(1) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_close(1) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
}

static int count_pending_tests(const char *ctest_bin, const char *build_dir) {
    char command[1024];
    int n = snprintf(command, sizeof(command),
        "%s --test-dir %s -L ipc-unix-pending -N 2>/dev/null", ctest_bin, build_dir);
    if (n < 0 || (size_t)n >= sizeof(command)) return -1;
    FILE *pipe = popen(command, "r");
    if (!pipe) return -1;
    int count = 0;
    char line[512];
    while (fgets(line, sizeof(line), pipe)) {
        if (strstr(line, "Test #")) ++count;
    }
    pclose(pipe);
    return count;
}

int main(int argc, char **argv) {
    check_stub_contract();
    if (argc >= 4) {
        int expected = atoi(argv[3]);
        int found = count_pending_tests(argv[1], argv[2]);
        CHECK(found == expected);
        fprintf(stderr,
            "ipc-unix-pending: %d acceptance cases registered (pending Gabe Kahen's core)\n",
            found);
    } else {
        fprintf(stderr,
            "ipc-unix-pending: registration count skipped (no ctest/build-dir/expected args)\n");
    }
    fprintf(stderr, "PASS: unix_socket_stream_stub_contract_test (%u checks)\n", checks);
    return 0;
}
