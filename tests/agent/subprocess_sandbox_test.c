/* SPDX-License-Identifier: MIT
 *
 * Kernel-enforced OS subprocess sandbox acceptance test.
 * Verifies Darwin Seatbelt / Linux seccomp+Landlock enforcement bound to
 * Eshkol's active capability policy (network denial, file-write denial,
 * scoped write-directory exception, and positive control when permitted).
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct eshkol_subprocess eshkol_subprocess_t;
extern eshkol_subprocess_t* qllm_process_spawn_argv(const char* tab_packed_argv,
                                                    const char* cwd_arg);
extern int qllm_process_wait(eshkol_subprocess_t* proc, int64_t timeout_ms);
extern int32_t qllm_process_exit_code(eshkol_subprocess_t* proc);
extern void qllm_process_destroy(eshkol_subprocess_t* proc);

extern int eshkol_capability_runtime_install(const char* names);
extern void eshkol_capability_runtime_host_reset(void);

static void check_cond(int cond, const char* expr, int line) {
    if (!cond) {
        fprintf(stderr, "FAIL: subprocess_sandbox_test:%d: %s\n", line, expr);
        exit(1);
    }
}
#define CHECK(expr) check_cond(!!(expr), #expr, __LINE__)

static int run_child_probe(const char* mode, const char* arg) {
    if (strcmp(mode, "echo") == 0) {
        return 0;
    }
    if (strcmp(mode, "net-probe") == 0) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return (errno == EPERM || errno == EACCES) ? 42 : 43;
        }
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(1);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        int rc = connect(fd, (struct sockaddr*)&addr, sizeof(addr));
        int err = errno;
        close(fd);
        if (rc < 0 && (err == EPERM || err == EACCES)) return 42;
        return 0;
    }
    if (strcmp(mode, "write-probe") == 0 && arg) {
        int fd = open(arg, O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (fd < 0) {
            return (errno == EPERM || errno == EACCES || errno == EROFS) ? 44 : 45;
        }
        ssize_t n = write(fd, "ok", 2);
        close(fd);
        return (n == 2) ? 0 : 46;
    }
    return 99;
}

static int spawn_self(const char* self, const char* mode, const char* arg, const char* cwd) {
    char packed[2048];
    if (arg) {
        snprintf(packed, sizeof(packed), "%s\t--child\t%s\t%s", self, mode, arg);
    } else {
        snprintf(packed, sizeof(packed), "%s\t--child\t%s", self, mode);
    }
    eshkol_subprocess_t* p = qllm_process_spawn_argv(packed, cwd);
    CHECK(p != NULL);
    int wrc = qllm_process_wait(p, 10000);
    int rc = qllm_process_exit_code(p);
    qllm_process_destroy(p);
    if (wrc < 0) return -1;
    return rc;
}

int main(int argc, char** argv) {
    if (argc >= 3 && strcmp(argv[1], "--child") == 0) {
        return run_child_probe(argv[2], argc >= 4 ? argv[3] : NULL);
    }

    const char* self = argv[0];
    char tmp_dir[] = "/tmp/eshkol-sandbox-XXXXXX";
    CHECK(mkdtemp(tmp_dir) != NULL);

    char allowed_dir[512];
    snprintf(allowed_dir, sizeof(allowed_dir), "%s/allowed", tmp_dir);
    CHECK(mkdir(allowed_dir, 0700) == 0);

    char blocked_file[512];
    char allowed_file[512];
    snprintf(blocked_file, sizeof(blocked_file), "%s/blocked.txt", tmp_dir);
    snprintf(allowed_file, sizeof(allowed_file), "%s/ok.txt", allowed_dir);

    unsetenv("ESHKOL_SUBPROC_OS_SANDBOX");
    unsetenv("ESHKOL_SUBPROC_SANDBOX_WRITE_DIR");

    /* 1. Policy grants subprocess + file-read only (denies network & file-write). */
    eshkol_capability_runtime_host_reset();
    CHECK(eshkol_capability_runtime_install("subprocess file-read") == 1);

    /* Benign child succeeds under sandbox. */
    CHECK(spawn_self(self, "echo", NULL, ".") == 0);

    /* Kernel denies TCP socket / connect with EPERM or EACCES (exit 42). */
    CHECK(spawn_self(self, "net-probe", NULL, ".") == 42);

    /* Kernel denies file creation outside allowed dirs (exit 44). */
    CHECK(spawn_self(self, "write-probe", blocked_file, ".") == 44);
    CHECK(access(blocked_file, F_OK) != 0);

    /* Same enforcement on fork+exec chdir path (cwd != "."). */
    CHECK(spawn_self(self, "net-probe", NULL, tmp_dir) == 42);
    CHECK(spawn_self(self, "write-probe", blocked_file, tmp_dir) == 44);

    /* 2. Scoped write directory allows writes inside allowed_dir, still blocks outside. */
    setenv("ESHKOL_SUBPROC_SANDBOX_WRITE_DIR", allowed_dir, 1);
    CHECK(spawn_self(self, "write-probe", allowed_file, ".") == 0);
    CHECK(access(allowed_file, F_OK) == 0);
    CHECK(spawn_self(self, "write-probe", blocked_file, ".") == 44);
    unsetenv("ESHKOL_SUBPROC_SANDBOX_WRITE_DIR");

    /* 3. Positive control: full policy permits writing blocked_file. */
    eshkol_capability_runtime_host_reset();
    CHECK(eshkol_capability_runtime_install("subprocess file-read file-write network") == 1);
    CHECK(spawn_self(self, "write-probe", blocked_file, ".") == 0);
    CHECK(access(blocked_file, F_OK) == 0);

    eshkol_capability_runtime_host_reset();
    unlink(allowed_file);
    unlink(blocked_file);
    rmdir(allowed_dir);
    rmdir(tmp_dir);

    printf("PASS: subprocess_sandbox_test (kernel network + file-write isolation verified)\n");
    return 0;
}
