#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#if defined(ESHKOL_HAVE_MKOSTEMP) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "model_io_atomic.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

#ifdef ESHKOL_MODEL_IO_TEST_HOOKS
static const char* atomic_checkpoint_failpoint(void) {
    return getenv("ESHKOL_TEST_MODEL_IO_FAIL");
}

static int atomic_checkpoint_should_fail(const char* point) {
    const char* configured = atomic_checkpoint_failpoint();
    return configured && strcmp(configured, point) == 0;
}

static int atomic_checkpoint_should_fail_write(unsigned long call) {
    const char* configured = atomic_checkpoint_failpoint();
    char* end = NULL;
    unsigned long requested;
    if (!configured) return 0;
    if (strcmp(configured, "write") == 0) return call == 1;
    if (strncmp(configured, "write:", 6) != 0) return 0;
    errno = 0;
    requested = strtoul(configured + 6, &end, 10);
    return errno == 0 && end && *end == '\0' && requested > 0 && call == requested;
}
#define ATOMIC_CHECKPOINT_SHOULD_FAIL(point) atomic_checkpoint_should_fail(point)
#define ATOMIC_CHECKPOINT_SHOULD_FAIL_WRITE(call) atomic_checkpoint_should_fail_write(call)
#else
#define ATOMIC_CHECKPOINT_SHOULD_FAIL(point) 0
#define ATOMIC_CHECKPOINT_SHOULD_FAIL_WRITE(call) 0
#endif

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
static int atomic_checkpoint_block_signals(eshkol_atomic_checkpoint_file_t* file) {
    sigset_t blocked;
    if (sigemptyset(&blocked) != 0 ||
        sigaddset(&blocked, SIGHUP) != 0 ||
        sigaddset(&blocked, SIGINT) != 0 ||
        sigaddset(&blocked, SIGQUIT) != 0 ||
        sigaddset(&blocked, SIGTERM) != 0 ||
        pthread_sigmask(SIG_BLOCK, &blocked, &file->previous_signal_mask) != 0) {
        return 0;
    }
    file->signals_blocked = 1;
    return 1;
}

static void atomic_checkpoint_restore_signals(eshkol_atomic_checkpoint_file_t* file) {
    if (file->signals_blocked) {
        (void)pthread_sigmask(SIG_SETMASK, &file->previous_signal_mask, NULL);
        file->signals_blocked = 0;
    }
}
#else
static int atomic_checkpoint_block_signals(eshkol_atomic_checkpoint_file_t* file) {
    (void)file;
    return 1;
}

static void atomic_checkpoint_restore_signals(eshkol_atomic_checkpoint_file_t* file) {
    (void)file;
}
#endif

static char* atomic_checkpoint_strdup(const char* value) {
    size_t size;
    char* copy;
    if (!value) return NULL;
    size = strlen(value) + 1;
    copy = (char*)malloc(size);
    if (copy) memcpy(copy, value, size);
    return copy;
}

static char* atomic_checkpoint_template(const char* destination) {
    const char* slash;
    size_t directory_len;
    size_t total;
    char* result;
    static const char basename_template[] = ".eshkol.XXXXXX";

    if (!destination || !*destination) return NULL;
    slash = strrchr(destination, '/');
#ifdef _WIN32
    {
        const char* backslash = strrchr(destination, '\\');
        if (backslash && (!slash || backslash > slash)) slash = backslash;
    }
#endif
    if (slash && !slash[1]) return NULL;
    directory_len = slash ? (size_t)(slash - destination + 1) : 0;
#ifdef _WIN32
    if (!slash && destination[0] && destination[1] == ':') directory_len = 2;
#endif
    if (directory_len > SIZE_MAX - sizeof(basename_template)) return NULL;
    total = directory_len + sizeof(basename_template);
    result = (char*)malloc(total);
    if (!result) return NULL;
    memcpy(result, destination, directory_len);
    memcpy(result + directory_len, basename_template, sizeof(basename_template));
    return result;
}

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__) && !defined(ESHKOL_HAVE_MKOSTEMP)
static int atomic_checkpoint_set_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    return flags >= 0 && fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}
#endif

static int atomic_checkpoint_create(char* path_template) {
#ifdef _WIN32
    static volatile LONG counter = 0;
    size_t path_length = strlen(path_template);
    char* marker = path_length >= 6 ? path_template + path_length - 6 : NULL;
    unsigned long attempt;
    if (!marker || strcmp(marker, "XXXXXX") != 0) return -1;
    for (attempt = 0; attempt < 256; ++attempt) {
        unsigned long value = (unsigned long)InterlockedIncrement(&counter) ^
                              (unsigned long)_getpid();
        (void)snprintf(marker, 7, "%06lx", value & 0xFFFFFFul);
        {
            int fd = _open(path_template,
                           _O_BINARY | _O_CREAT | _O_EXCL | _O_NOINHERIT | _O_RDWR,
                           _S_IREAD | _S_IWRITE);
            if (fd >= 0 || errno != EEXIST) return fd;
        }
    }
    errno = EEXIST;
    return -1;
#else
#if defined(__EMSCRIPTEN__)
    return mkstemp(path_template);
#elif defined(ESHKOL_HAVE_MKOSTEMP)
    return mkostemp(path_template, O_CLOEXEC);
#else
    int fd = mkstemp(path_template);
    if (fd >= 0 && !atomic_checkpoint_set_cloexec(fd)) {
        int saved_errno = errno;
        (void)close(fd);
        (void)unlink(path_template);
        errno = saved_errno;
        return -1;
    }
    return fd;
#endif
#endif
}

static int atomic_checkpoint_replace(const char* temporary_path,
                                     const char* destination_path) {
#ifdef _WIN32
    return MoveFileExA(temporary_path, destination_path,
                       MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return rename(temporary_path, destination_path) == 0;
#endif
}

static void atomic_checkpoint_unlink(const char* path) {
    if (!path) return;
#ifdef _WIN32
    (void)_unlink(path);
#else
    (void)unlink(path);
#endif
}

void eshkol_atomic_checkpoint_abort(eshkol_atomic_checkpoint_file_t* file) {
    if (!file) return;
    if (file->stream) {
        (void)fclose(file->stream);
        file->stream = NULL;
    }
    atomic_checkpoint_unlink(file->temporary_path);
    free(file->temporary_path);
    free(file->destination_path);
    file->temporary_path = NULL;
    file->destination_path = NULL;
    atomic_checkpoint_restore_signals(file);
}

int eshkol_atomic_checkpoint_begin(eshkol_atomic_checkpoint_file_t* file,
                                   const char* destination_path) {
    char* path_template;
    int fd;
#ifndef _WIN32
    struct stat destination_status;
    int preserve_mode = 0;
    mode_t destination_mode = 0600;
#endif

    if (!file || !destination_path || !*destination_path) return 0;
    memset(file, 0, sizeof(*file));
    if (!atomic_checkpoint_block_signals(file)) return 0;
    if (ATOMIC_CHECKPOINT_SHOULD_FAIL("open")) {
        atomic_checkpoint_restore_signals(file);
        return 0;
    }

#ifndef _WIN32
    if (lstat(destination_path, &destination_status) == 0 &&
        S_ISREG(destination_status.st_mode)) {
        preserve_mode = 1;
        destination_mode = destination_status.st_mode & 0777;
    }
#endif

    file->destination_path = atomic_checkpoint_strdup(destination_path);
    path_template = atomic_checkpoint_template(destination_path);
    if (!file->destination_path || !path_template) {
        free(path_template);
        eshkol_atomic_checkpoint_abort(file);
        return 0;
    }

    fd = atomic_checkpoint_create(path_template);
    if (fd < 0) {
        free(path_template);
        eshkol_atomic_checkpoint_abort(file);
        return 0;
    }
    file->temporary_path = path_template;

#ifndef _WIN32
    if (fchmod(fd, preserve_mode ? destination_mode : 0600) != 0) {
        (void)close(fd);
        eshkol_atomic_checkpoint_abort(file);
        return 0;
    }
    file->stream = fdopen(fd, "wb");
#else
    file->stream = _fdopen(fd, "wb");
#endif
    if (!file->stream) {
#ifdef _WIN32
        (void)_close(fd);
#else
        (void)close(fd);
#endif
        eshkol_atomic_checkpoint_abort(file);
        return 0;
    }
    return 1;
}

size_t eshkol_atomic_checkpoint_write(eshkol_atomic_checkpoint_file_t* file,
                                      const void* data,
                                      size_t size) {
    if (!file || !file->stream || (!data && size != 0)) return 0;
    file->write_calls++;
    if (ATOMIC_CHECKPOINT_SHOULD_FAIL_WRITE(file->write_calls)) return 0;
    if (size == 0) return 0;
    return fwrite(data, 1, size, file->stream);
}

int eshkol_atomic_checkpoint_commit(eshkol_atomic_checkpoint_file_t* file) {
    int flush_ok;
    int close_ok;
    int replace_ok;
    if (!file || !file->stream || !file->temporary_path || !file->destination_path) {
        return 0;
    }

    flush_ok = !ATOMIC_CHECKPOINT_SHOULD_FAIL("flush") && fflush(file->stream) == 0;
    close_ok = fclose(file->stream) == 0;
    file->stream = NULL;
    if (ATOMIC_CHECKPOINT_SHOULD_FAIL("close")) close_ok = 0;
    if (!flush_ok || !close_ok || ATOMIC_CHECKPOINT_SHOULD_FAIL("interrupt")) {
        eshkol_atomic_checkpoint_abort(file);
        return 0;
    }
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#ifdef ESHKOL_MODEL_IO_TEST_HOOKS
    if (ATOMIC_CHECKPOINT_SHOULD_FAIL("signal")) {
        (void)raise(SIGTERM);
        eshkol_atomic_checkpoint_abort(file);
        return 0;
    }
#endif
#endif

    replace_ok = !ATOMIC_CHECKPOINT_SHOULD_FAIL("rename") &&
                 atomic_checkpoint_replace(file->temporary_path, file->destination_path);
    if (!replace_ok) {
        eshkol_atomic_checkpoint_abort(file);
        return 0;
    }

    free(file->temporary_path);
    free(file->destination_path);
    file->temporary_path = NULL;
    file->destination_path = NULL;
    atomic_checkpoint_restore_signals(file);
    return 1;
}
