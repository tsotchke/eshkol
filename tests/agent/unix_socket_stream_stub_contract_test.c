/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include <eshkol/core/unix_socket_stream.h>

#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
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
    const char *path = "/tmp/eshkol-scaffold.sock";
    CHECK(ESHKOL_UNIX_STREAM_SOCKET_EINVAL == ESHKOL_IO_INVALID);
    CHECK(ESHKOL_UNIX_STREAM_SOCKET_ECANCELED == ESHKOL_IO_CANCELLED);
    CHECK(eshkol_unix_stream_socket_listen(1, path) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_listen(0, path) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_listen(1, "relative") == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_listen(1, NULL) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_accept(1, 0) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_accept(1, -1) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_connect_begin(1, path, 0) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_connect_begin(0, path, 0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_connect_finish(1) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_connect_finish(0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_read(1, buf, sizeof(buf)) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_write(1, buf, sizeof(buf)) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_read(1, buf, 0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_write(1, buf, 0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_poll(1, 3, 0) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_poll(1, 0, 0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_poll(1, 4, 0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_poll(1, 1, -1) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_cancel(1) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_close(1) == ESHKOL_UNIX_STREAM_SOCKET_PLANNED);
    CHECK(eshkol_unix_stream_socket_cancel(0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_close(0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_last_os_error() == 0);
}

extern char **environ;
static int count_pending_tests(const char *ctest_bin, const char *build_dir) {
    int fds[2];
    if (pipe(fds) != 0) return -1;
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) { close(fds[0]); close(fds[1]); return -1; }
    int error = posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
    if (!error) error = posix_spawn_file_actions_addclose(&actions, fds[0]);
    if (!error) error = posix_spawn_file_actions_addclose(&actions, fds[1]);
    char *const args[] = {(char *)ctest_bin, "--test-dir", (char *)build_dir,
                         "-L", "ipc-unix-pending", "-N", NULL};
    pid_t pid = -1;
    if (!error) error = posix_spawn(&pid, ctest_bin, &actions, NULL, args, environ);
    posix_spawn_file_actions_destroy(&actions); close(fds[1]);
    if (error) { close(fds[0]); return -1; }
    FILE *stream = fdopen(fds[0], "r");
    int count = 0;
    if (stream) {
        char line[1024];
        while (fgets(line, sizeof(line), stream)) {
            char *p = strstr(line, "Test");
            if (!p) continue;
            p += 4;
            while (isspace((unsigned char)*p)) ++p;
            if (*p == '#' && isdigit((unsigned char)p[1])) ++count;
        }
        fclose(stream);
    } else close(fds[0]);
    int status;
    while (waitpid(pid, &status, 0) < 0) if (errno != EINTR) return -1;
    if (!stream || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return -1;
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
