/*
 * system_builtins.c — C runtime functions for system/path/process builtins
 *
 * These are called from LLVM-compiled code via CALL_BUILTIN.
 * Each function takes tagged value arguments and returns a tagged value.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <dirent.h>
#include <libgen.h>
#include <limits.h>
#include <errno.h>
#include <signal.h>
#include <poll.h>
#include <time.h>
#include <pwd.h>
#include <fcntl.h>
#include <glob.h>
#include <fnmatch.h>
#ifdef __APPLE__
#include <util.h>
#else
#include <pty.h>
#endif
#else
#include <windows.h>
#include <direct.h>
#include <process.h>
#endif

/* Forward declarations from arena_memory.h */
extern void* get_global_arena(void);
extern void* arena_allocate(void* arena, size_t size);
extern char* arena_allocate_string_with_header(void* arena, size_t length);

/* Tagged value layout MUST match LLVM IR: {i8, i8, i16, i32, i64} = 16 bytes.
 * The i32 is explicit padding — this matches the LLVM struct exactly so that
 * struct return calling conventions are identical between C and LLVM-compiled code. */
typedef struct {
    uint8_t type;       /* Field 0: type tag */
    uint8_t flags;      /* Field 1: flags */
    uint16_t reserved;  /* Field 2: reserved */
    uint32_t padding;   /* Field 3: explicit padding (matches LLVM i32) */
    uint64_t data;      /* Field 4: data (matches LLVM i64) */
} eshkol_sysbuiltin_value_t;

#define SYS_TYPE_NULL    0
#define SYS_TYPE_INT64   1
#define SYS_TYPE_DOUBLE  2
#define SYS_TYPE_BOOL    3
#define SYS_TYPE_HEAP_PTR 8

static eshkol_sysbuiltin_value_t sys_make_null(void) {
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    return v;
}

static eshkol_sysbuiltin_value_t sys_make_bool(int val) {
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_BOOL;
    v.data = val ? 1 : 0;
    return v;
}

static eshkol_sysbuiltin_value_t sys_make_int64(int64_t val) {
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_INT64;
    memcpy(&v.data, &val, sizeof(int64_t));
    return v;
}

static eshkol_sysbuiltin_value_t sys_make_double(double val) {
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_DOUBLE;
    memcpy(&v.data, &val, sizeof(double));
    return v;
}

static eshkol_sysbuiltin_value_t sys_make_string(const char* s) {
    if (!s) return sys_make_null();
    void* arena = get_global_arena();
    if (!arena) return sys_make_null();
    size_t len = strlen(s);
    /* Use arena_allocate_string_with_header so the string has a proper
     * object header — required by eshkol_display_value and the runtime. */
    char* copy = arena_allocate_string_with_header(arena, len);
    if (!copy) return sys_make_null();
    memcpy(copy, s, len + 1);
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.flags = 0x01; /* string subtype */
    v.data = (uint64_t)copy;
    return v;
}

static const char* sys_extract_string(eshkol_sysbuiltin_value_t v) {
    /* Strings from LLVM codegen have type=HEAP_PTR, flags=0 (subtype is in
     * the object header at ptr-8, not in the tagged value flags byte).
     * Strings from FFI/system_builtins have type=HEAP_PTR, flags=0x01.
     * Accept both patterns. */
    if (v.type == SYS_TYPE_HEAP_PTR && v.data != 0) {
        return (const char*)(uintptr_t)v.data;
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════
 * System Information
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_os_type_v(void) {
#ifdef __APPLE__
    return sys_make_string("darwin");
#elif defined(_WIN32)
    return sys_make_string("windows");
#elif defined(__linux__)
    return sys_make_string("linux");
#elif defined(__FreeBSD__)
    return sys_make_string("freebsd");
#else
    return sys_make_string("unknown");
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_os_arch_v(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return sys_make_string("arm64");
#elif defined(__x86_64__) || defined(_M_X64)
    return sys_make_string("x86_64");
#elif defined(__i386__) || defined(_M_IX86)
    return sys_make_string("x86");
#else
    return sys_make_string("unknown");
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_hostname_v(void) {
#ifndef _WIN32
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) {
        buf[sizeof(buf) - 1] = '\0';
        return sys_make_string(buf);
    }
#else
    char buf[256];
    DWORD sz = sizeof(buf);
    if (GetComputerNameA(buf, &sz)) {
        return sys_make_string(buf);
    }
#endif
    return sys_make_string("unknown");
}

static eshkol_sysbuiltin_value_t eshkol_builtin_username_v(void) {
#ifndef _WIN32
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_name) return sys_make_string(pw->pw_name);
    const char* user = getenv("USER");
    if (user) return sys_make_string(user);
#else
    char buf[256];
    DWORD sz = sizeof(buf);
    if (GetUserNameA(buf, &sz)) return sys_make_string(buf);
#endif
    return sys_make_string("unknown");
}

static eshkol_sysbuiltin_value_t eshkol_builtin_cpu_count_v(void) {
#ifndef _WIN32
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return sys_make_int64(n > 0 ? n : 1);
#else
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return sys_make_int64((int64_t)si.dwNumberOfProcessors);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_getpid_v(void) {
#ifndef _WIN32
    return sys_make_int64((int64_t)getpid());
#else
    return sys_make_int64((int64_t)_getpid());
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_home_directory_v(void) {
#ifndef _WIN32
    const char* home = getenv("HOME");
    if (home) return sys_make_string(home);
#else
    const char* home = getenv("USERPROFILE");
    if (home) return sys_make_string(home);
#endif
    return sys_make_string("");
}

static eshkol_sysbuiltin_value_t eshkol_builtin_sleep_ms_v(eshkol_sysbuiltin_value_t ms_val) {
    int64_t ms = (int64_t)ms_val.data;
    if (ms <= 0) return sys_make_null();
#ifndef _WIN32
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
#else
    Sleep((DWORD)ms);
#endif
    return sys_make_null();
}

static eshkol_sysbuiltin_value_t eshkol_builtin_executable_exists_v(eshkol_sysbuiltin_value_t name_val) {
    const char* name = sys_extract_string(name_val);
    if (!name) return sys_make_bool(0);
#ifndef _WIN32
    /* Search PATH */
    const char* path_env = getenv("PATH");
    if (!path_env) return sys_make_bool(0);
    char* path_copy = strdup(path_env);
    if (!path_copy) return sys_make_bool(0);
    char* dir = strtok(path_copy, ":");
    while (dir) {
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", dir, name);
        if (access(full, X_OK) == 0) {
            free(path_copy);
            return sys_make_bool(1);
        }
        dir = strtok(NULL, ":");
    }
    free(path_copy);
    return sys_make_bool(0);
#else
    /* Windows: use SearchPathA */
    char result[MAX_PATH];
    if (SearchPathA(NULL, name, ".exe", MAX_PATH, result, NULL) > 0)
        return sys_make_bool(1);
    return sys_make_bool(0);
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Path Manipulation
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_path_join_v(eshkol_sysbuiltin_value_t a, eshkol_sysbuiltin_value_t b) {
    const char* sa = sys_extract_string(a);
    const char* sb = sys_extract_string(b);
    if (!sa || !sb) return sys_make_null();
    char buf[PATH_MAX];
    size_t la = strlen(sa);
    if (la > 0 && sa[la - 1] == '/') {
        snprintf(buf, sizeof(buf), "%s%s", sa, sb);
    } else {
        snprintf(buf, sizeof(buf), "%s/%s", sa, sb);
    }
    return sys_make_string(buf);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_dirname_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
    char* copy = strdup(path);
    if (!copy) return sys_make_null();
    char* dir = dirname(copy);
    eshkol_sysbuiltin_value_t result = sys_make_string(dir);
    free(copy);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_basename_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
    char* copy = strdup(path);
    if (!copy) return sys_make_null();
    char* base = basename(copy);
    eshkol_sysbuiltin_value_t result = sys_make_string(base);
    free(copy);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_extname_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
    const char* dot = strrchr(path, '.');
    const char* sep = strrchr(path, '/');
    if (dot && (!sep || dot > sep)) {
        return sys_make_string(dot);
    }
    return sys_make_string("");
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_is_absolute_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_bool(0);
#ifndef _WIN32
    return sys_make_bool(path[0] == '/');
#else
    return sys_make_bool((path[0] == '\\') || (path[0] == '/') ||
                         (path[1] == ':' && (path[2] == '\\' || path[2] == '/')));
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_normalize_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
    /* Simple normalization: resolve . and .. */
    char buf[PATH_MAX];
    const char* parts[256];
    int nparts = 0;
    char* copy = strdup(path);
    if (!copy) return sys_make_null();
    int is_abs = (copy[0] == '/');
    char* tok = strtok(copy, "/");
    while (tok) {
        if (strcmp(tok, ".") == 0) {
            /* skip */
        } else if (strcmp(tok, "..") == 0) {
            if (nparts > 0) nparts--;
        } else {
            if (nparts < 256) parts[nparts++] = tok;
        }
        tok = strtok(NULL, "/");
    }
    buf[0] = '\0';
    if (is_abs) strcat(buf, "/");
    for (int i = 0; i < nparts; i++) {
        if (i > 0) strcat(buf, "/");
        strcat(buf, parts[i]);
    }
    if (buf[0] == '\0') strcpy(buf, ".");
    free(copy);
    return sys_make_string(buf);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_realpath_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    char resolved[PATH_MAX];
    if (realpath(path, resolved)) {
        return sys_make_string(resolved);
    }
#else
    char resolved[MAX_PATH];
    if (GetFullPathNameA(path, MAX_PATH, resolved, NULL)) {
        return sys_make_string(resolved);
    }
#endif
    return sys_make_null();
}

/* ═══════════════════════════════════════════════════════════════════
 * Filesystem Operations
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_file_stat_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    struct stat st;
    if (stat(path, &st) != 0) return sys_make_null();
    /* Return size as a simple int64 — could return an alist later */
    return sys_make_int64((int64_t)st.st_size);
#else
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &fad)) return sys_make_null();
    LARGE_INTEGER sz;
    sz.HighPart = fad.nFileSizeHigh;
    sz.LowPart = fad.nFileSizeLow;
    return sys_make_int64((int64_t)sz.QuadPart);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_file_copy_v(eshkol_sysbuiltin_value_t src_val, eshkol_sysbuiltin_value_t dst_val) {
    const char* src = sys_extract_string(src_val);
    const char* dst = sys_extract_string(dst_val);
    if (!src || !dst) return sys_make_bool(0);
#ifndef _WIN32
    FILE* fin = fopen(src, "rb");
    if (!fin) return sys_make_bool(0);
    FILE* fout = fopen(dst, "wb");
    if (!fout) { fclose(fin); return sys_make_bool(0); }
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fin)) > 0) {
        fwrite(buf, 1, n, fout);
    }
    fclose(fin);
    fclose(fout);
    return sys_make_bool(1);
#else
    return sys_make_bool(CopyFileA(src, dst, FALSE) != 0);
#endif
}

static int mkdir_recursive_impl(const char* path) {
#ifndef _WIN32
    struct stat st;
    if (stat(path, &st) == 0) return 0; /* exists */
    char tmp[PATH_MAX];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return mkdir(tmp, 0755);
#else
    /* Windows: use SHCreateDirectoryExA or manual loop */
    char tmp[MAX_PATH];
    strncpy(tmp, path, sizeof(tmp) - 1);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '\\' || *p == '/') {
            char saved = *p;
            *p = '\0';
            _mkdir(tmp);
            *p = saved;
        }
    }
    return _mkdir(tmp);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_mkdir_recursive_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_bool(0);
    mkdir_recursive_impl(path);
    /* Check if it exists now */
#ifndef _WIN32
    struct stat st;
    return sys_make_bool(stat(path, &st) == 0 && S_ISDIR(st.st_mode));
#else
    DWORD attr = GetFileAttributesA(path);
    return sys_make_bool(attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_mkdtemp_v(eshkol_sysbuiltin_value_t template_val) {
    const char* tmpl = sys_extract_string(template_val);
    if (!tmpl) return sys_make_null();
#ifndef _WIN32
    char* copy = strdup(tmpl);
    if (!copy) return sys_make_null();
    char* result = mkdtemp(copy);
    if (result) {
        eshkol_sysbuiltin_value_t r = sys_make_string(result);
        free(copy);
        return r;
    }
    free(copy);
    return sys_make_null();
#else
    char buf[MAX_PATH];
    if (GetTempPathA(MAX_PATH, buf)) {
        char name[MAX_PATH];
        if (GetTempFileNameA(buf, "esk", 0, name)) {
            DeleteFileA(name);
            if (_mkdir(name) == 0) return sys_make_string(name);
        }
    }
    return sys_make_null();
#endif
}

static int rmdir_recursive_impl(const char* path) {
#ifndef _WIN32
    DIR* d = opendir(path);
    if (!d) return -1;
    struct dirent* ent;
    while ((ent = readdir(d))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        char child[PATH_MAX];
        snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
        struct stat st;
        if (stat(child, &st) == 0 && S_ISDIR(st.st_mode)) {
            rmdir_recursive_impl(child);
        } else {
            unlink(child);
        }
    }
    closedir(d);
    return rmdir(path);
#else
    return -1; /* TODO: Windows recursive delete */
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_directory_delete_recursive_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_bool(0);
    return sys_make_bool(rmdir_recursive_impl(path) == 0);
}

/* ═══════════════════════════════════════════════════════════════════
 * Shell Utilities
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_shell_quote_v(eshkol_sysbuiltin_value_t str_val) {
    const char* s = sys_extract_string(str_val);
    if (!s) return sys_make_null();
    /* Single-quote wrapping with internal ' escaped as '\'' */
    size_t len = strlen(s);
    size_t out_len = 2 + len * 4; /* worst case */
    void* arena = get_global_arena();
    if (!arena) return sys_make_null();
    char* buf = arena_allocate_string_with_header(arena, out_len);
    if (!buf) return sys_make_null();
    char* p = buf;
    *p++ = '\'';
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '\'') {
            *p++ = '\''; *p++ = '\\'; *p++ = '\''; *p++ = '\'';
        } else {
            *p++ = s[i];
        }
    }
    *p++ = '\'';
    *p = '\0';
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)buf;
    return v;
}

/* ═══════════════════════════════════════════════════════════════════
 * Process Management
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_process_spawn_v(eshkol_sysbuiltin_value_t cmd_val,
                                                        eshkol_sysbuiltin_value_t args_val) {
    const char* cmd = sys_extract_string(cmd_val);
    if (!cmd) return sys_make_int64(-1);
#ifndef _WIN32
    pid_t pid = fork();
    if (pid == 0) {
        /* Child: build argv from the args list */
        /* For now, simple exec without args list parsing */
        execlp(cmd, cmd, (char*)NULL);
        _exit(127);
    }
    if (pid < 0) return sys_make_int64(-1);
    return sys_make_int64((int64_t)pid);
#else
    return sys_make_int64(-1); /* TODO: Windows CreateProcess */
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_process_wait_v(eshkol_sysbuiltin_value_t pid_val) {
    int64_t pid = (int64_t)pid_val.data;
    if (pid <= 0) return sys_make_int64(-1);
#ifndef _WIN32
    int status = 0;
    if (waitpid((pid_t)pid, &status, 0) < 0) return sys_make_int64(-1);
    if (WIFEXITED(status)) return sys_make_int64(WEXITSTATUS(status));
    if (WIFSIGNALED(status)) return sys_make_int64(128 + WTERMSIG(status));
    return sys_make_int64(-1);
#else
    return sys_make_int64(-1); /* TODO: Windows WaitForSingleObject */
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * IO Multiplexing
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_poll_fd_v(eshkol_sysbuiltin_value_t fd_val,
                                                   eshkol_sysbuiltin_value_t timeout_val) {
    int64_t fd = (int64_t)fd_val.data;
    int64_t timeout_ms = (int64_t)timeout_val.data;
#ifndef _WIN32
    struct pollfd pfd;
    pfd.fd = (int)fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int ret = poll(&pfd, 1, (int)timeout_ms);
    if (ret > 0 && (pfd.revents & POLLIN)) return sys_make_bool(1);
    return sys_make_bool(0);
#else
    (void)fd; (void)timeout_ms;
    return sys_make_bool(0);
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Additional Filesystem Operations (VM parity)
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_file_chmod_v(eshkol_sysbuiltin_value_t path_val,
                                                              eshkol_sysbuiltin_value_t mode_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_bool(0);
#ifndef _WIN32
    int mode = (int)(int64_t)mode_val.data;
    return sys_make_bool(chmod(path, (mode_t)mode) == 0);
#else
    (void)mode_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_symlink_create_v(eshkol_sysbuiltin_value_t target_val,
                                                                  eshkol_sysbuiltin_value_t link_val) {
    const char* target = sys_extract_string(target_val);
    const char* link = sys_extract_string(link_val);
    if (!target || !link) return sys_make_bool(0);
#ifndef _WIN32
    return sys_make_bool(symlink(target, link) == 0);
#else
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_symlink_read_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    char buf[PATH_MAX];
    ssize_t len = readlink(path, buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        return sys_make_string(buf);
    }
#endif
    return sys_make_null();
}

static eshkol_sysbuiltin_value_t eshkol_builtin_directory_walk_v(eshkol_sysbuiltin_value_t path_val) {
    /* Returns a flat list of all file paths under path (recursive BFS) */
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    /* BFS with stack of directories */
    char dirs[256][PATH_MAX];
    int dir_count = 0, dir_idx = 0;
    strncpy(dirs[0], path, PATH_MAX - 1); dirs[0][PATH_MAX - 1] = '\0';
    dir_count = 1;

    /* Build result as a linked list of strings (reversed, then return) */
    /* For simplicity, collect into a flat array first */
    char* results[4096];
    int result_count = 0;

    while (dir_idx < dir_count && dir_count < 256) {
        DIR* d = opendir(dirs[dir_idx]);
        dir_idx++;
        if (!d) continue;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL && result_count < 4096) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
            char full[PATH_MAX];
            snprintf(full, sizeof(full), "%s/%s", dirs[dir_idx - 1], ent->d_name);
            struct stat st;
            if (stat(full, &st) == 0 && S_ISDIR(st.st_mode) && dir_count < 256) {
                strncpy(dirs[dir_count], full, PATH_MAX - 1);
                dirs[dir_count][PATH_MAX - 1] = '\0';
                dir_count++;
            }
            results[result_count] = strdup(full);
            result_count++;
        }
        closedir(d);
    }

    /* Build tagged value list (reversed — cons prepends) */
    /* For LLVM path, we'd need to build proper cons cells.
     * For now, return a simple string with newline-separated paths
     * that can be split in Eshkol. */
    if (result_count == 0) {
        return sys_make_null();
    }

    /* Calculate total length */
    size_t total_len = 0;
    for (int i = 0; i < result_count; i++) {
        total_len += strlen(results[i]) + 1; /* +1 for newline */
    }

    void* arena = get_global_arena();
    char* buf = arena_allocate_string_with_header(arena, total_len);
    if (!buf) {
        for (int i = 0; i < result_count; i++) free(results[i]);
        return sys_make_null();
    }
    char* p = buf;
    for (int i = 0; i < result_count; i++) {
        size_t len = strlen(results[i]);
        memcpy(p, results[i], len);
        p += len;
        *p++ = '\n';
        free(results[i]);
    }
    if (p > buf) p[-1] = '\0'; /* replace last newline with null */
    else *p = '\0';

    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)buf;
    return v;
#else
    return sys_make_null();
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_mkstemp_v(eshkol_sysbuiltin_value_t tmpl_val) {
    const char* tmpl = sys_extract_string(tmpl_val);
    if (!tmpl) return sys_make_null();
#ifndef _WIN32
    char* copy = strdup(tmpl);
    if (!copy) return sys_make_null();
    int fd = mkstemp(copy);
    if (fd >= 0) {
        /* Return the path string (caller gets the fd from the path via open) */
        eshkol_sysbuiltin_value_t result = sys_make_string(copy);
        free(copy);
        close(fd);
        return result;
    }
    free(copy);
#endif
    return sys_make_null();
}

static eshkol_sysbuiltin_value_t eshkol_builtin_process_kill_v(eshkol_sysbuiltin_value_t pid_val,
                                                                eshkol_sysbuiltin_value_t sig_val) {
    int64_t pid = (int64_t)pid_val.data;
    int64_t sig = (int64_t)sig_val.data;
#ifndef _WIN32
    return sys_make_bool(kill((pid_t)pid, (int)sig) == 0);
#else
    (void)pid; (void)sig;
    return sys_make_bool(0);
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * New Builtins — not in VM yet either
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_file_mtime_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    struct stat st;
    if (stat(path, &st) == 0) {
        return sys_make_int64((int64_t)st.st_mtime);
    }
#endif
    return sys_make_null();
}

static eshkol_sysbuiltin_value_t eshkol_builtin_file_atime_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    struct stat st;
    if (stat(path, &st) == 0) {
        return sys_make_int64((int64_t)st.st_atime);
    }
#endif
    return sys_make_null();
}

static eshkol_sysbuiltin_value_t eshkol_builtin_file_lock_v(eshkol_sysbuiltin_value_t fd_val) {
    int64_t fd = (int64_t)fd_val.data;
#ifndef _WIN32
    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    return sys_make_bool(fcntl((int)fd, F_SETLK, &fl) != -1);
#else
    (void)fd;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_file_unlock_v(eshkol_sysbuiltin_value_t fd_val) {
    int64_t fd = (int64_t)fd_val.data;
#ifndef _WIN32
    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_type = F_UNLCK;
    fl.l_whence = SEEK_SET;
    return sys_make_bool(fcntl((int)fd, F_SETLK, &fl) != -1);
#else
    (void)fd;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_relative_v(eshkol_sysbuiltin_value_t from_val,
                                                                 eshkol_sysbuiltin_value_t to_val) {
    const char* from = sys_extract_string(from_val);
    const char* to = sys_extract_string(to_val);
    if (!from || !to) return sys_make_null();

    /* Simple relative path: strip common prefix */
    size_t common = 0;
    size_t last_sep = 0;
    while (from[common] && to[common] && from[common] == to[common]) {
        if (from[common] == '/') last_sep = common + 1;
        common++;
    }
    /* If we stopped at a separator boundary, use common as-is */
    if (from[common] == '/' || from[common] == '\0') last_sep = common + (from[common] == '/');
    if (to[common] == '/' || to[common] == '\0') {
        if (common > last_sep) last_sep = common + (to[common] == '/');
    }

    /* Count remaining directories in 'from' after common prefix */
    int up_count = 0;
    for (const char* p = from + last_sep; *p; p++) {
        if (*p == '/') up_count++;
    }
    if (from[last_sep]) up_count++; /* count the last segment */

    char buf[PATH_MAX];
    buf[0] = '\0';
    for (int i = 0; i < up_count; i++) {
        strcat(buf, i > 0 ? "/.." : "..");
    }
    if (to[last_sep]) {
        if (buf[0]) strcat(buf, "/");
        strcat(buf, to + last_sep);
    }
    if (buf[0] == '\0') strcpy(buf, ".");
    return sys_make_string(buf);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_resolve_v(eshkol_sysbuiltin_value_t base_val,
                                                                eshkol_sysbuiltin_value_t rel_val) {
    const char* base = sys_extract_string(base_val);
    const char* rel = sys_extract_string(rel_val);
    if (!rel) return sys_make_null();
    /* If rel is absolute, return it directly */
    if (rel[0] == '/') return sys_make_string(rel);
    if (!base) return sys_make_null();
    /* Join base dir + rel, then normalize */
    char buf[PATH_MAX];
    /* Get directory of base (strip filename if base doesn't end with /) */
    char base_copy[PATH_MAX];
    strncpy(base_copy, base, PATH_MAX - 1); base_copy[PATH_MAX - 1] = '\0';
    snprintf(buf, sizeof(buf), "%s/%s", base_copy, rel);

    /* Normalize */
    eshkol_sysbuiltin_value_t norm_input = sys_make_string(buf);
    return eshkol_builtin_path_normalize_v(norm_input);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_glob_expand_v(eshkol_sysbuiltin_value_t pattern_val) {
    const char* pattern = sys_extract_string(pattern_val);
    if (!pattern) return sys_make_null();
#ifndef _WIN32
    glob_t g;
    memset(&g, 0, sizeof(g));
    int ret = glob(pattern, GLOB_NOSORT | GLOB_TILDE, NULL, &g);
    if (ret != 0) {
        globfree(&g);
        return sys_make_null();
    }
    /* Build newline-separated string of results */
    size_t total_len = 0;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        total_len += strlen(g.gl_pathv[i]) + 1;
    }
    if (total_len == 0) { globfree(&g); return sys_make_null(); }
    void* arena = get_global_arena();
    char* buf = arena_allocate_string_with_header(arena, total_len);
    if (!buf) { globfree(&g); return sys_make_null(); }
    char* p = buf;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        size_t len = strlen(g.gl_pathv[i]);
        memcpy(p, g.gl_pathv[i], len);
        p += len;
        *p++ = '\n';
    }
    if (p > buf) p[-1] = '\0';
    else *p = '\0';
    globfree(&g);
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)buf;
    return v;
#else
    return sys_make_null();
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_glob_match_v(eshkol_sysbuiltin_value_t pattern_val,
                                                              eshkol_sysbuiltin_value_t str_val) {
    const char* pattern = sys_extract_string(pattern_val);
    const char* str = sys_extract_string(str_val);
    if (!pattern || !str) return sys_make_bool(0);
#ifndef _WIN32
    return sys_make_bool(fnmatch(pattern, str, 0) == 0);
#else
    return sys_make_bool(0);
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Advanced Process Management
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_process_setpgid_v(eshkol_sysbuiltin_value_t pid_val,
                                                                    eshkol_sysbuiltin_value_t pgid_val) {
    int64_t pid = (int64_t)pid_val.data;
    int64_t pgid = (int64_t)pgid_val.data;
#ifndef _WIN32
    return sys_make_bool(setpgid((pid_t)pid, (pid_t)pgid) == 0);
#else
    (void)pid; (void)pgid;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_process_kill_tree_v(eshkol_sysbuiltin_value_t pid_val,
                                                                     eshkol_sysbuiltin_value_t sig_val) {
    int64_t pid = (int64_t)pid_val.data;
    int64_t sig = (int64_t)sig_val.data;
#ifndef _WIN32
    /* Kill the entire process group: -pid sends signal to all processes in group */
    if (pid > 0) {
        /* First try killing the process group */
        int rc = kill(-(pid_t)pid, (int)sig);
        if (rc != 0) {
            /* Fall back to killing just the process */
            rc = kill((pid_t)pid, (int)sig);
        }
        return sys_make_bool(rc == 0);
    }
    return sys_make_bool(0);
#else
    (void)pid; (void)sig;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_process_spawn_pty_v(eshkol_sysbuiltin_value_t cmd_val) {
    /* PTY spawn — allocate a pseudo-terminal for interactive processes */
    const char* cmd = sys_extract_string(cmd_val);
    if (!cmd) return sys_make_int64(-1);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int master_fd;
    pid_t pid = forkpty(&master_fd, NULL, NULL, NULL);
    if (pid < 0) return sys_make_int64(-1);
    if (pid == 0) {
        /* Child: exec the command via shell */
        execlp("/bin/sh", "sh", "-c", cmd, (char*)NULL);
        _exit(127);
    }
    /* Parent: return master fd (can read/write to it) */
    /* Pack as cons pair: (pid . master-fd) */
    /* For simplicity, return just the pid — the master_fd is tracked internally */
    return sys_make_int64((int64_t)pid);
#else
    return sys_make_int64(-1);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_process_read_nonblocking_v(eshkol_sysbuiltin_value_t fd_val,
                                                                            eshkol_sysbuiltin_value_t max_val) {
    int64_t fd = (int64_t)fd_val.data;
    int64_t max_bytes = (int64_t)max_val.data;
    if (fd < 0 || max_bytes <= 0) return sys_make_null();
#ifndef _WIN32
    /* Set non-blocking */
    int flags = fcntl((int)fd, F_GETFL);
    fcntl((int)fd, F_SETFL, flags | O_NONBLOCK);

    void* arena = get_global_arena();
    if (!arena) return sys_make_null();
    char* buf = arena_allocate_string_with_header(arena, (size_t)max_bytes);
    if (!buf) return sys_make_null();

    ssize_t n = read((int)fd, buf, (size_t)max_bytes);

    /* Restore blocking */
    fcntl((int)fd, F_SETFL, flags);

    if (n <= 0) return sys_make_null(); /* EAGAIN or EOF */
    buf[n] = '\0';

    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)buf;
    return v;
#else
    (void)fd; (void)max_bytes;
    return sys_make_null();
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Tensor Persistence
 * ═══════════════════════════════════════════════════════════════════ */

/* Forward declaration for tensor type — MUST match arena_memory.h exactly:
 * struct eshkol_tensor {
 *     uint64_t* dimensions;     // idx 0
 *     uint64_t  num_dimensions; // idx 1
 *     int64_t*  elements;       // idx 2 (doubles as int64 bit patterns)
 *     uint64_t  total_elements; // idx 3
 * }; // 32 bytes */
typedef struct {
    uint64_t* dimensions;
    uint64_t  num_dimensions;
    int64_t*  elements;
    uint64_t  total_elements;
} eshkol_tensor_t_ffi;

extern void* arena_allocate_tensor_full(void* arena, uint64_t ndims, uint64_t total);

#define TENSOR_FILE_MAGIC 0x45534B54 /* "ESKT" */

static eshkol_sysbuiltin_value_t eshkol_builtin_tensor_save_v(eshkol_sysbuiltin_value_t path_val,
                                                               eshkol_sysbuiltin_value_t tensor_val) {
    const char* path = sys_extract_string(path_val);
    if (!path || tensor_val.type != SYS_TYPE_HEAP_PTR) return sys_make_bool(0);
    eshkol_tensor_t_ffi* t = (eshkol_tensor_t_ffi*)(uintptr_t)tensor_val.data;
    if (!t || !t->elements) return sys_make_bool(0);
#ifndef _WIN32
    FILE* f = fopen(path, "wb");
    if (!f) return sys_make_bool(0);
    uint32_t magic = TENSOR_FILE_MAGIC;
    uint32_t version = 1;
    uint32_t ndims = (uint32_t)t->num_dimensions;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&ndims, 4, 1, f);
    for (uint32_t i = 0; i < ndims; i++) {
        fwrite(&t->dimensions[i], 8, 1, f);
    }
    /* Elements are int64 bit patterns of doubles, 8 bytes each */
    fwrite(t->elements, 8, (size_t)t->total_elements, f);
    fclose(f);
    return sys_make_bool(1);
#else
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_tensor_load_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    FILE* f = fopen(path, "rb");
    if (!f) return sys_make_null();
    uint32_t magic, version, ndims;
    if (fread(&magic, 4, 1, f) != 1 || magic != TENSOR_FILE_MAGIC ||
        fread(&version, 4, 1, f) != 1 || version != 1 ||
        fread(&ndims, 4, 1, f) != 1 || ndims == 0 || ndims > 8) {
        fclose(f);
        return sys_make_null();
    }
    int64_t shape[8];
    for (uint32_t i = 0; i < ndims; i++) {
        if (fread(&shape[i], 8, 1, f) != 1) { fclose(f); return sys_make_null(); }
    }
    int64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) total *= shape[i];

    void* arena = get_global_arena();
    if (!arena) { fclose(f); return sys_make_null(); }
    eshkol_tensor_t_ffi* t = (eshkol_tensor_t_ffi*)arena_allocate_tensor_full(arena, ndims, (uint64_t)total);
    if (!t || !t->elements) { fclose(f); return sys_make_null(); }
    if ((int64_t)fread(t->elements, 8, (size_t)total, f) != total) {
        fclose(f);
        return sys_make_null();
    }
    fclose(f);
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)t;
    return v;
#else
    return sys_make_null();
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * SRET Wrappers — called from LLVM-compiled code
 *
 * LLVM IR calls void func(tagged_value_t* out, ...) to avoid struct
 * return ABI mismatches between LLVM IR and C on ARM64/etc.
 * The original functions are renamed to _inner, and these wrappers
 * write the result through a pointer.
 * ═══════════════════════════════════════════════════════════════════ */

typedef eshkol_sysbuiltin_value_t sv_t;

/* All-pointer interface: result AND args passed via pointer.
 * This eliminates ALL struct ABI mismatches between LLVM IR and C. */
void eshkol_builtin_os_type(sv_t* out) { *out = eshkol_builtin_os_type_v(); }
void eshkol_builtin_os_arch(sv_t* out) { *out = eshkol_builtin_os_arch_v(); }
void eshkol_builtin_hostname(sv_t* out) { *out = eshkol_builtin_hostname_v(); }
void eshkol_builtin_username(sv_t* out) { *out = eshkol_builtin_username_v(); }
void eshkol_builtin_cpu_count(sv_t* out) { *out = eshkol_builtin_cpu_count_v(); }
void eshkol_builtin_getpid(sv_t* out) { *out = eshkol_builtin_getpid_v(); }
void eshkol_builtin_home_directory(sv_t* out) { *out = eshkol_builtin_home_directory_v(); }
void eshkol_builtin_sleep_ms(sv_t* out, const sv_t* a) { *out = eshkol_builtin_sleep_ms_v(*a); }
void eshkol_builtin_executable_exists(sv_t* out, const sv_t* a) { *out = eshkol_builtin_executable_exists_v(*a); }
void eshkol_builtin_path_join(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_path_join_v(*a, *b); }
void eshkol_builtin_path_dirname(sv_t* out, const sv_t* a) { *out = eshkol_builtin_path_dirname_v(*a); }
void eshkol_builtin_path_basename(sv_t* out, const sv_t* a) { *out = eshkol_builtin_path_basename_v(*a); }
void eshkol_builtin_path_extname(sv_t* out, const sv_t* a) { *out = eshkol_builtin_path_extname_v(*a); }
void eshkol_builtin_path_is_absolute(sv_t* out, const sv_t* a) { *out = eshkol_builtin_path_is_absolute_v(*a); }
void eshkol_builtin_path_normalize(sv_t* out, const sv_t* a) { *out = eshkol_builtin_path_normalize_v(*a); }
void eshkol_builtin_realpath(sv_t* out, const sv_t* a) { *out = eshkol_builtin_realpath_v(*a); }
void eshkol_builtin_file_stat(sv_t* out, const sv_t* a) { *out = eshkol_builtin_file_stat_v(*a); }
void eshkol_builtin_file_copy(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_file_copy_v(*a, *b); }
void eshkol_builtin_mkdir_recursive(sv_t* out, const sv_t* a) { *out = eshkol_builtin_mkdir_recursive_v(*a); }
void eshkol_builtin_mkdtemp(sv_t* out, const sv_t* a) { *out = eshkol_builtin_mkdtemp_v(*a); }
void eshkol_builtin_directory_delete_recursive(sv_t* out, const sv_t* a) { *out = eshkol_builtin_directory_delete_recursive_v(*a); }
void eshkol_builtin_shell_quote(sv_t* out, const sv_t* a) { *out = eshkol_builtin_shell_quote_v(*a); }
void eshkol_builtin_process_spawn(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_spawn_v(*a, *b); }
void eshkol_builtin_process_wait(sv_t* out, const sv_t* a) { *out = eshkol_builtin_process_wait_v(*a); }
void eshkol_builtin_poll_fd(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_poll_fd_v(*a, *b); }
void eshkol_builtin_tensor_save(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_tensor_save_v(*a, *b); }
void eshkol_builtin_tensor_load(sv_t* out, const sv_t* a) { *out = eshkol_builtin_tensor_load_v(*a); }
/* v1.2 batch 2: VM-parity + new builtins */
void eshkol_builtin_file_chmod(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_file_chmod_v(*a, *b); }
void eshkol_builtin_symlink_create(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_symlink_create_v(*a, *b); }
void eshkol_builtin_symlink_read(sv_t* out, const sv_t* a) { *out = eshkol_builtin_symlink_read_v(*a); }
void eshkol_builtin_directory_walk(sv_t* out, const sv_t* a) { *out = eshkol_builtin_directory_walk_v(*a); }
void eshkol_builtin_mkstemp(sv_t* out, const sv_t* a) { *out = eshkol_builtin_mkstemp_v(*a); }
void eshkol_builtin_process_kill(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_kill_v(*a, *b); }
void eshkol_builtin_file_mtime(sv_t* out, const sv_t* a) { *out = eshkol_builtin_file_mtime_v(*a); }
void eshkol_builtin_file_atime(sv_t* out, const sv_t* a) { *out = eshkol_builtin_file_atime_v(*a); }
void eshkol_builtin_file_lock(sv_t* out, const sv_t* a) { *out = eshkol_builtin_file_lock_v(*a); }
void eshkol_builtin_file_unlock(sv_t* out, const sv_t* a) { *out = eshkol_builtin_file_unlock_v(*a); }
void eshkol_builtin_path_relative(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_path_relative_v(*a, *b); }
void eshkol_builtin_path_resolve(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_path_resolve_v(*a, *b); }
void eshkol_builtin_glob_expand(sv_t* out, const sv_t* a) { *out = eshkol_builtin_glob_expand_v(*a); }
void eshkol_builtin_glob_match(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_glob_match_v(*a, *b); }
/* v1.2 batch 3: advanced process management */
void eshkol_builtin_process_setpgid(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_setpgid_v(*a, *b); }
void eshkol_builtin_process_kill_tree(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_kill_tree_v(*a, *b); }
void eshkol_builtin_process_spawn_pty(sv_t* out, const sv_t* a) { *out = eshkol_builtin_process_spawn_pty_v(*a); }
void eshkol_builtin_process_read_nonblocking(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_read_nonblocking_v(*a, *b); }
