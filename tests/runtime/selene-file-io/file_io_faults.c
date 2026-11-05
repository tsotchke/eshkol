/* Test-only native fixtures; no runtime interposition or mocked stdio. */
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

static struct rlimit saved_limit;
static struct sigaction saved_action;
static int armed;

/* Each generated test executable runs as a child of the CMake driver. Save
 * both soft/hard limits and the complete signal action; never lower the hard
 * limit. No assertions/output happen until restoration has completed. */
int32_t selene_file_io_arm(void) {
    struct sigaction ignore;
    struct rlimit limited;
    if (armed || getrlimit(RLIMIT_FSIZE, &saved_limit) != 0 ||
        sigaction(SIGXFSZ, NULL, &saved_action) != 0) return -1;
    memset(&ignore, 0, sizeof(ignore));
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    if (sigaction(SIGXFSZ, &ignore, NULL) != 0) return -2;
    limited = saved_limit;
    limited.rlim_cur = 0;
    if (setrlimit(RLIMIT_FSIZE, &limited) != 0) {
        if (sigaction(SIGXFSZ, &saved_action, NULL) != 0) _exit(91);
        return -3;
    }
    armed = 1;
    return 0;
}

int32_t selene_file_io_restore(void) {
    struct rlimit restored;
    if (!armed) return -1;
    /* Attempt both restorations even if the first fails. On failure terminate
     * without stdio, so no assertion writes under the temporary restriction. */
    const int limit_rc = setrlimit(RLIMIT_FSIZE, &saved_limit);
    const int action_rc = sigaction(SIGXFSZ, &saved_action, NULL);
    if (limit_rc != 0 || action_rc != 0) _exit(92);
    armed = 0;
    if (getrlimit(RLIMIT_FSIZE, &restored) != 0 ||
        restored.rlim_cur != saved_limit.rlim_cur ||
        restored.rlim_max != saved_limit.rlim_max) _exit(93);
    return 0;
}

/* Establish that this host actually produces the late-flush error tested by
 * the one-byte primitive calls. Fail the test if the fixture cannot do so. */
int32_t selene_file_io_late_control(const char* path) {
    if (!armed) return 0;
    FILE* file = fopen(path, "wb");
    if (!file) return 0;
    char buffer[BUFSIZ];
    if (setvbuf(file, buffer, _IOFBF, sizeof(buffer)) != 0) {
        fclose(file);
        return 0;
    }
    const size_t written = fwrite("x", 1, 1, file);
    errno = 0;
    const int closed = fclose(file);
    return written == 1 && closed != 0 && errno == EFBIG;
}

int32_t selene_file_io_short_control(const char* path) {
    if (!armed) return 0;
    FILE* file = fopen(path, "wb");
    if (!file) return 0;
    char buffer[BUFSIZ];
    char payload[BUFSIZ * 2] = {0};
    if (setvbuf(file, buffer, _IOFBF, sizeof(buffer)) != 0) {
        fclose(file);
        return 0;
    }
    errno = 0;
    const size_t written = fwrite(payload, 1, sizeof(payload), file);
    const int rejected = written < sizeof(payload) && ferror(file) && errno == EFBIG;
    fclose(file); /* Close even on a short write; its result is irrelevant here. */
    return rejected;
}

/* Independent native byte oracle avoids relying on read-file's codegen. */
int32_t selene_file_io_bytes(const char* path, const char* expected, int64_t length) {
    if (length < 0) return 0;
    FILE* file = fopen(path, "rb");
    if (!file) return 0;
    int ok = 1;
    for (int64_t i = 0; i < length; ++i) {
        const int byte = fgetc(file);
        if (byte == EOF || byte != (unsigned char)expected[i]) { ok = 0; break; }
    }
    if (ok && (fgetc(file) != EOF || ferror(file))) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok;
}
