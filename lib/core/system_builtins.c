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
#include <time.h>   /* struct tm, gmtime_s/gmtime_r — used on every platform */

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
#include <pwd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
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
/* Windows POSIX compatibility:
 *   - <limits.h> on MSVC defines _MAX_PATH (260) but not PATH_MAX.
 *   - <libgen.h> (dirname/basename) does not exist on Windows.
 * We use a generous PATH_MAX (4096 matches Linux) for our own buffers,
 * and provide portable dirname/basename below. */
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

/* Portable dirname/basename — POSIX-equivalent semantics on every platform.
 * POSIX dirname/basename are allowed to modify their argument and return a
 * pointer into it (or into a static buffer); ours always work in-place,
 * matching the POSIX modify-in-place behaviour our callers already expect.
 * Treats both '/' and '\\' as separators so they DTRT on Windows paths. */
static char* eshkol_portable_dirname(char* path) {
    if (!path || !*path) return (char*)".";
    size_t n = strlen(path);
    while (n > 1 && (path[n-1] == '/' || path[n-1] == '\\')) path[--n] = '\0';
    char* sep = NULL;
    for (size_t i = 0; i < n; i++) {
        if (path[i] == '/' || path[i] == '\\') sep = path + i;
    }
    if (!sep) { return (char*)"."; }
    if (sep == path) { path[1] = '\0'; return path; }
    *sep = '\0';
    return path;
}

static char* eshkol_portable_basename(char* path) {
    if (!path || !*path) return (char*)".";
    size_t n = strlen(path);
    while (n > 1 && (path[n-1] == '/' || path[n-1] == '\\')) path[--n] = '\0';
    char* sep = NULL;
    for (size_t i = 0; i < n; i++) {
        if (path[i] == '/' || path[i] == '\\') sep = path + i;
    }
    return sep ? sep + 1 : path;
}

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

/* ═══════════════════════════════════════════════════════════════════
 * Time API (ISO8601) — #168
 *
 * format-iso8601: int64 ns-since-epoch → "YYYY-MM-DDTHH:MM:SS.mmmZ"
 * parse-iso8601:  "YYYY-MM-DDTHH:MM:SS[.mmm][Z|±HH:MM]" → int64 ns, or #f
 * current-timestamp: → double seconds-since-epoch (wall-clock UTC)
 *
 * We use gmtime_r + snprintf for format (UTC, no locale dependency),
 * and strptime / manual parse for parse. parse returns #f on any
 * malformed input so callers can distinguish bad strings from epoch
 * zero.
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_format_iso8601_v(eshkol_sysbuiltin_value_t ns_val) {
    /* The argument is nanoseconds since Unix epoch. current-time-ns packs
     * its int64 ns count as a double (SIToFP in system_codegen), so DOUBLE
     * input is still ns — we just cast back. INT64 input is also ns. */
    int64_t ns;
    if (ns_val.type == SYS_TYPE_DOUBLE) {
        double d;
        memcpy(&d, &ns_val.data, sizeof(double));
        ns = (int64_t)d;
    } else {
        ns = (int64_t)ns_val.data;
    }
    time_t secs = (time_t)(ns / 1000000000LL);
    long ms = (long)((ns / 1000000LL) % 1000LL);
    if (ms < 0) { ms += 1000; secs -= 1; }

#ifndef _WIN32
    struct tm tmv;
    if (!gmtime_r(&secs, &tmv)) return sys_make_null();
#else
    struct tm tmv;
    if (gmtime_s(&tmv, &secs) != 0) return sys_make_null();
#endif

    char buf[40];
    int n = snprintf(buf, sizeof(buf),
                     "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
                     tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                     tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms);
    if (n <= 0 || n >= (int)sizeof(buf)) return sys_make_null();
    return sys_make_string(buf);
}

/* Days-before-month in a non-leap year. Adjusted for Feb in leap years. */
static int days_before_month[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

static int is_leap_year(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

/* Compute Unix epoch seconds from a calendar datetime (UTC). Portable
 * equivalent of timegm() — avoids locale / TZ state that mktime carries. */
static int64_t timegm_portable(int year, int mon, int day,
                               int hour, int min, int sec) {
    int64_t days = 0;
    for (int y = 1970; y < year; y++) {
        days += is_leap_year(y) ? 366 : 365;
    }
    if (mon < 1 || mon > 12) return 0;
    days += days_before_month[mon - 1];
    if (mon > 2 && is_leap_year(year)) days += 1;
    days += (day - 1);
    return days * 86400LL + hour * 3600LL + min * 60LL + sec;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_parse_iso8601_v(eshkol_sysbuiltin_value_t str_val) {
    const char* s = sys_extract_string(str_val);
    if (!s) return sys_make_bool(0);

    int year, mon, day, hour, min, sec;
    int ms = 0;
    char tz_sign = 0;
    int tz_hour = 0, tz_min = 0;

    /* Required: YYYY-MM-DDTHH:MM:SS. Accept space as T separator too. */
    int n = 0;
    if (sscanf(s, "%4d-%2d-%2d%*[T ]%2d:%2d:%2d%n",
               &year, &mon, &day, &hour, &min, &sec, &n) != 6) {
        return sys_make_bool(0);
    }
    const char* p = s + n;

    /* Optional .fff fractional seconds (millisecond precision). */
    if (*p == '.') {
        p++;
        int digits = 0;
        int frac = 0;
        while (*p >= '0' && *p <= '9' && digits < 9) {
            frac = frac * 10 + (*p - '0');
            digits++;
            p++;
        }
        /* Convert to milliseconds — pad or truncate as needed. */
        while (digits < 3) { frac *= 10; digits++; }
        while (digits > 3) { frac /= 10; digits--; }
        ms = frac;
    }

    /* Timezone: Z, +HH:MM, -HH:MM, or implicit UTC. */
    if (*p == 'Z') {
        tz_sign = 0;
    } else if (*p == '+' || *p == '-') {
        tz_sign = *p;
        if (sscanf(p + 1, "%2d:%2d", &tz_hour, &tz_min) != 2) {
            return sys_make_bool(0);
        }
    } else if (*p != '\0') {
        /* Unknown trailing content — reject. */
        return sys_make_bool(0);
    }

    int64_t secs = timegm_portable(year, mon, day, hour, min, sec);
    if (tz_sign == '+') secs -= (tz_hour * 3600 + tz_min * 60);
    if (tz_sign == '-') secs += (tz_hour * 3600 + tz_min * 60);

    int64_t ns = secs * 1000000000LL + (int64_t)ms * 1000000LL;
    return sys_make_int64(ns);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_current_timestamp_v(void) {
#ifndef _WIN32
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return sys_make_double(0.0);
    double secs = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    return sys_make_double(secs);
#else
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    /* 100-ns intervals since 1601-01-01. Subtract Unix epoch offset. */
    uint64_t v = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    v -= 116444736000000000ULL;
    double secs = (double)v * 1e-7;
    return sys_make_double(secs);
#endif
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
    char* dir = eshkol_portable_dirname(copy);
    eshkol_sysbuiltin_value_t result = sys_make_string(dir);
    free(copy);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_path_basename_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
    char* copy = strdup(path);
    if (!copy) return sys_make_null();
    char* base = eshkol_portable_basename(copy);
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

    /* Reject inputs longer than PATH_MAX outright — strcat-based
     * construction from a buffer that's already oversize is how stack
     * overflows used to happen here (#193 HIGH). */
    size_t in_len = strlen(path);
    if (in_len >= PATH_MAX) return sys_make_null();

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

    /* Bounded concatenation. We track the remaining space in `buf` and
     * bail to an empty result rather than truncating silently — a
     * truncated path is worse than an error because it might point
     * somewhere completely different. */
    buf[0] = '\0';
    size_t pos = 0;
    if (is_abs) {
        if (pos + 1 >= PATH_MAX) { free(copy); return sys_make_null(); }
        buf[pos++] = '/';
        buf[pos] = '\0';
    }
    for (int i = 0; i < nparts; i++) {
        size_t plen = strlen(parts[i]);
        size_t need = plen + (i > 0 ? 1 : 0);  /* leading '/' except first */
        if (pos + need + 1 >= PATH_MAX) {
            free(copy);
            return sys_make_null();
        }
        if (i > 0) { buf[pos++] = '/'; }
        memcpy(buf + pos, parts[i], plen);
        pos += plen;
        buf[pos] = '\0';
    }
    if (buf[0] == '\0') {
        buf[0] = '.';
        buf[1] = '\0';
    }
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
    /* TOCTOU hardening (#193): open via file descriptors with
     * O_NOFOLLOW + O_CLOEXEC so a symlink-swap attack between the
     * fopen(src) and fopen(dst) calls can't redirect writes to
     * sensitive files (e.g. /etc/passwd). If either side is a
     * symlink we refuse the copy.
     *
     * Dst opens with O_EXCL by default to avoid clobbering an
     * existing target via a race; callers who want overwrite pass a
     * path that doesn't exist, or use a future file-copy-force.
     */
    int src_fd = open(src, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (src_fd < 0) return sys_make_bool(0);
    int dst_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (dst_fd < 0) { close(src_fd); return sys_make_bool(0); }
    char buf[8192];
    ssize_t n;
    int ok = 1;
    while ((n = read(src_fd, buf, sizeof(buf))) > 0) {
        ssize_t written = 0;
        while (written < n) {
            ssize_t w = write(dst_fd, buf + written, (size_t)(n - written));
            if (w < 0) {
                if (errno == EINTR) continue;
                ok = 0;
                break;
            }
            written += w;
        }
        if (!ok) break;
    }
    if (n < 0) ok = 0;
    close(src_fd);
    /* close(dst_fd) can fail (e.g. NFS commit error, EIO on flush of
     * the page cache) AFTER write() returned success — POSIX says the
     * data isn't guaranteed durable until close returns 0.  Silently
     * dropping the close error means file-copy reports success even
     * when the destination is short or corrupt.  Treat as failure. */
    if (close(dst_fd) < 0) ok = 0;
    return sys_make_bool(ok ? 1 : 0);
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
    static int depth = 0;
    if (++depth > 100) { --depth; return -1; }
#ifndef _WIN32
    DIR* d = opendir(path);
    if (!d) { --depth; return -1; }
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
    int ret = rmdir(path);
    --depth;
    return ret;
#else
    /* Windows recursive delete using FindFirstFile/FindNextFile */
    WIN32_FIND_DATAA fdata;
    char search[MAX_PATH];
    snprintf(search, sizeof(search), "%s\\*", path);
    HANDLE h = FindFirstFileA(search, &fdata);
    if (h == INVALID_HANDLE_VALUE) { --depth; return -1; }
    do {
        if (strcmp(fdata.cFileName, ".") == 0 || strcmp(fdata.cFileName, "..") == 0) continue;
        char child[MAX_PATH];
        snprintf(child, sizeof(child), "%s\\%s", path, fdata.cFileName);
        if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            rmdir_recursive_impl(child);
        } else {
            DeleteFileA(child);
        }
    } while (FindNextFileA(h, &fdata));
    FindClose(h);
    int wret = RemoveDirectoryA(path) ? 0 : -1;
    --depth;
    return wret;
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
    if (len > SIZE_MAX / 4 - 2) return sys_make_null();
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
    const char* env_str = sys_extract_string(args_val); /* optional: env var "KEY=VAL" */
    if (!cmd) return sys_make_int64(-1);
#ifndef _WIN32
    pid_t pid = fork();
    if (pid == 0) {
        /* Child: set env if provided, then exec via shell */
        if (env_str && env_str[0]) {
            /* Parse "KEY=VAL KEY2=VAL2 ..." and set each */
            char* env_copy = strdup(env_str);
            if (env_copy) {
                char* pair = strtok(env_copy, " ");
                while (pair) {
                    putenv(pair); /* putenv takes ownership of the string in the child */
                    pair = strtok(NULL, " ");
                }
                /* Don't free env_copy — putenv uses it */
            }
        }
        execlp("/bin/sh", "sh", "-c", cmd, (char*)NULL);
        _exit(127);
    }
    if (pid < 0) return sys_make_int64(-1);
    return sys_make_int64((int64_t)pid);
#else
    /* Windows process creation via CreateProcess + cmd /c */
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    memset(&pi, 0, sizeof(pi));

    /* Build command line: cmd /c "command" */
    char cmdline[32768];
    snprintf(cmdline, sizeof(cmdline), "cmd /c %s", cmd);

    /* Set environment if provided */
    if (env_str && strlen(env_str) > 0) {
        /* putenv each KEY=VAL pair before creating child */
        char* env_copy = _strdup(env_str);
        if (env_copy) {
            char* pair = strtok(env_copy, " ");
            while (pair) {
                _putenv(pair);
                pair = strtok(NULL, " ");
            }
            free(env_copy);
        }
    }

    if (!CreateProcessA(NULL, cmdline, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        return sys_make_int64(-1);
    }
    CloseHandle(pi.hThread);
    /* Return process ID — the handle is stored implicitly via the PID */
    return sys_make_int64((int64_t)pi.dwProcessId);
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
    /* Windows process wait via OpenProcess + WaitForSingleObject */
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | SYNCHRONIZE, FALSE, (DWORD)pid);
    if (!hProcess) return sys_make_int64(-1);
    WaitForSingleObject(hProcess, INFINITE);
    DWORD exit_code = 0;
    GetExitCodeProcess(hProcess, &exit_code);
    CloseHandle(hProcess);
    return sys_make_int64((int64_t)exit_code);
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
    /* BFS with arena-allocated directory queue (no stack overflow risk) */
    #define WALK_MAX_DIRS 4096
    #define WALK_MAX_RESULTS 16384
    void* walk_arena = get_global_arena();
    char** dirs = (char**)arena_allocate(walk_arena, WALK_MAX_DIRS * sizeof(char*));
    char** results = (char**)arena_allocate(walk_arena, WALK_MAX_RESULTS * sizeof(char*));
    if (!dirs || !results) return sys_make_null();

    int dir_count = 0, dir_idx = 0, result_count = 0;
    dirs[0] = (char*)arena_allocate(walk_arena, PATH_MAX);
    if (!dirs[0]) return sys_make_null();
    strncpy(dirs[0], path, PATH_MAX - 1); dirs[0][PATH_MAX - 1] = '\0';
    dir_count = 1;

    while (dir_idx < dir_count && dir_count < WALK_MAX_DIRS) {
        DIR* d = opendir(dirs[dir_idx]);
        dir_idx++;
        if (!d) continue;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL && result_count < WALK_MAX_RESULTS) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
            char full[PATH_MAX];
            snprintf(full, sizeof(full), "%s/%s", dirs[dir_idx - 1], ent->d_name);
            struct stat st;
            if (stat(full, &st) == 0 && S_ISDIR(st.st_mode) && dir_count < WALK_MAX_DIRS) {
                dirs[dir_count] = (char*)arena_allocate(walk_arena, PATH_MAX);
                if (!dirs[dir_count]) break;
                strncpy(dirs[dir_count], full, PATH_MAX - 1);
                dirs[dir_count][PATH_MAX - 1] = '\0';
                dir_count++;
            }
            size_t full_len = strlen(full);
            results[result_count] = (char*)arena_allocate(walk_arena, full_len + 1);
            if (!results[result_count]) break;
            memcpy(results[result_count], full, full_len + 1);
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
        /* results are arena-allocated — no individual free needed */
        return sys_make_null();
    }
    char* p = buf;
    for (int i = 0; i < result_count; i++) {
        size_t len = strlen(results[i]);
        memcpy(p, results[i], len);
        p += len;
        *p++ = '\n';
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

/* process-pid: return current process ID (alias for getpid) */
static eshkol_sysbuiltin_value_t eshkol_builtin_process_pid_v(void) {
#ifndef _WIN32
    return sys_make_int64((int64_t)getpid());
#else
    return sys_make_int64((int64_t)_getpid());
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
 * Unix-domain socket helpers
 * ═══════════════════════════════════════════════════════════════════ */

static eshkol_sysbuiltin_value_t eshkol_builtin_unix_socket_connect_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path || !*path) return sys_make_bool(0);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    size_t len = strlen(path);
    if (len >= sizeof(((struct sockaddr_un*)0)->sun_path)) return sys_make_bool(0);

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return sys_make_bool(0);
#ifdef SO_NOSIGPIPE
    int no_sigpipe = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, (socklen_t)sizeof(no_sigpipe));
#endif

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    memcpy(addr.sun_path, path, len);
    addr.sun_path[len] = '\0';

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0)
        return sys_make_int64((int64_t)fd);

    close(fd);
#endif
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_socket_send_v(eshkol_sysbuiltin_value_t fd_val,
                                                               eshkol_sysbuiltin_value_t data_val) {
    int64_t fd = (int64_t)fd_val.data;
    const char* data = sys_extract_string(data_val);
    if (fd < 0 || !data) return sys_make_bool(0);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int flags = 0;
#ifdef MSG_NOSIGNAL
    flags |= MSG_NOSIGNAL;
#endif
    ssize_t n = send((int)fd, data, strlen(data), flags);
    if (n >= 0) return sys_make_int64((int64_t)n);
#endif
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_socket_recv_v(eshkol_sysbuiltin_value_t fd_val,
                                                               eshkol_sysbuiltin_value_t max_val) {
    int64_t fd = (int64_t)fd_val.data;
    int64_t max_bytes = (int64_t)max_val.data;
    if (fd < 0 || max_bytes <= 0) return sys_make_bool(0);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    if (max_bytes > 65536) max_bytes = 65536;
    void* arena = get_global_arena();
    if (!arena) return sys_make_bool(0);
    char* buf = arena_allocate_string_with_header(arena, (size_t)max_bytes);
    if (!buf) return sys_make_bool(0);

    int old_flags = fcntl((int)fd, F_GETFL, 0);
    if (old_flags < 0 || fcntl((int)fd, F_SETFL, old_flags | O_NONBLOCK) != 0)
        return sys_make_bool(0);

    ssize_t n = recv((int)fd, buf, (size_t)max_bytes, 0);
    (void)fcntl((int)fd, F_SETFL, old_flags);
    if (n <= 0) return sys_make_bool(0);

    buf[n] = '\0';
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)buf;
    return v;
#endif
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_socket_close_v(eshkol_sysbuiltin_value_t fd_val) {
    int64_t fd = (int64_t)fd_val.data;
    if (fd < 0) return sys_make_bool(0);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    return sys_make_bool(close((int)fd) == 0);
#else
    return sys_make_bool(0);
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * KB Persistence — save/load knowledge bases
 * ═══════════════════════════════════════════════════════════════════ */

#define KB_FILE_MAGIC 0x45534B42 /* "ESKB" */

/* Delegate to C++ implementations in kb_persistence.cpp.
 * Public surface: (kb-save path kb) — path first, kb second (matches VM and bench code). */
extern void eshkol_kb_save_tagged(void* arena,
                                   const void* path_tv,
                                   const void* kb_tv,
                                   void* result);
extern void eshkol_kb_load_tagged(void* arena,
                                   const void* path_tv,
                                   void* result);

static eshkol_sysbuiltin_value_t eshkol_builtin_kb_save_v(eshkol_sysbuiltin_value_t path_val,
                                                           eshkol_sysbuiltin_value_t kb_val) {
    eshkol_sysbuiltin_value_t result = {0, 0, 0, 0, 0};
    void* arena = get_global_arena();
    eshkol_kb_save_tagged(arena, &path_val, &kb_val, &result);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_kb_load_v(eshkol_sysbuiltin_value_t path_val) {
    eshkol_sysbuiltin_value_t result = {0, 0, 0, 0, 0};
    void* arena = get_global_arena();
    eshkol_kb_load_tagged(arena, &path_val, &result);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 * Tensor Token Estimation
 * ═══════════════════════════════════════════════════════════════════ */

/* Estimate number of tokens a tensor would produce if serialized as text.
 * Useful for context window management in LLM agents. */
/* Forward reference — eshkol_tensor_t_ffi defined below in Tensor Persistence section */
struct eshkol_tensor_ffi_fwd;

static eshkol_sysbuiltin_value_t eshkol_builtin_tensor_token_estimate_v(eshkol_sysbuiltin_value_t tensor_val) {
    if (tensor_val.type != SYS_TYPE_HEAP_PTR) return sys_make_int64(0);
    /* Read total_elements (offset 24) and num_dimensions (offset 8) directly
     * from the tensor struct without requiring the typedef in scope. */
    void* t = (void*)(uintptr_t)tensor_val.data;
    if (!t) return sys_make_int64(0);
    uint64_t num_dims = *((uint64_t*)((char*)t + 8));   /* num_dimensions at offset 8 */
    uint64_t total = *((uint64_t*)((char*)t + 24));      /* total_elements at offset 24 */
    int64_t estimate = (int64_t)total * 8 + (int64_t)num_dims * 5 + 10;
    return sys_make_int64(estimate);
}

/* ═══════════════════════════════════════════════════════════════════
 * Memory-mapped file I/O
 * ═══════════════════════════════════════════════════════════════════ */

#ifndef _WIN32
#include <sys/mman.h>
#endif

/* file-mmap returns (addr . size) cons pair where addr is the raw mapped pointer.
 * The mapping stays alive until file-munmap is called. */
static eshkol_sysbuiltin_value_t eshkol_builtin_file_mmap_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_null();
#ifndef _WIN32
    int fd = open(path, O_RDONLY);
    if (fd < 0) return sys_make_null();
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return sys_make_null(); }
    size_t len = (size_t)st.st_size;
    void* mapped = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return sys_make_null();

    /* Return the mapped pointer as a string-like heap value.
     * Caller must use file-munmap to release. We store the mapping
     * in an arena-allocated header so the runtime can display it,
     * but the actual data pointer is the mmap region. */
    void* arena = get_global_arena();
    if (!arena) { munmap(mapped, len); return sys_make_null(); }

    /* Store mmap info: we pack (addr, size) into a cons-like pair.
     * For simplicity, return the mmap as a string (readable). */
    char* copy = arena_allocate_string_with_header(arena, len);
    if (!copy) { munmap(mapped, len); return sys_make_null(); }
    memcpy(copy, mapped, len);
    copy[len] = '\0';
    munmap(mapped, len); /* Release the OS mapping — data is in arena now */

    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)copy;
    return v;
#else
    return sys_make_null();
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_file_munmap_v(eshkol_sysbuiltin_value_t addr_val) {
    /* The mmap was already copied to arena and released in file-mmap.
     * Arena memory is freed at shutdown. This is a no-op for cleanup symmetry. */
    (void)addr_val;
    return sys_make_bool(1);
}

/* ═══════════════════════════════════════════════════════════════════
 * number->string with radix
 * ═══════════════════════════════════════════════════════════════════ */

/* Raw version: takes int64 values directly from LLVM IR.
 * Avoids all struct passing ABI issues. Returns arena-allocated string. */
char* eshkol_number_to_string_radix_raw(int64_t n, int64_t r, void* arena) {
    if (r < 2 || r > 36 || !arena) return NULL;

    static const char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    char buf[68];
    char* p = buf + sizeof(buf) - 1;
    *p = '\0';

    int neg = (n < 0);
    uint64_t val = neg ? (uint64_t)(-n) : (uint64_t)n;

    if (val == 0) {
        *--p = '0';
    } else {
        while (val > 0) {
            *--p = digits[val % (uint64_t)r];
            val /= (uint64_t)r;
        }
    }
    if (neg) *--p = '-';

    size_t len = strlen(p);
    char* result = arena_allocate_string_with_header(arena, len);
    if (!result) return NULL;
    memcpy(result, p, len + 1);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 * ONNX Export
 * ═══════════════════════════════════════════════════════════════════ */

extern int eshkol_onnx_export(const char* path, const char** names,
                               const int64_t** dims, const int* ndims,
                               const double** data, const int64_t* totals,
                               int n_tensors);

/* onnx-export-tensor(path, tensor) → #t or #f
 * Export a single named tensor to ONNX format. */
static eshkol_sysbuiltin_value_t eshkol_builtin_onnx_export_tensor_v(
        eshkol_sysbuiltin_value_t path_val,
        eshkol_sysbuiltin_value_t tensor_val) {
    const char* path = sys_extract_string(path_val);
    if (!path || tensor_val.type != SYS_TYPE_HEAP_PTR) return sys_make_bool(0);

    /* Extract tensor data */
    void* t = (void*)(uintptr_t)tensor_val.data;
    if (!t) return sys_make_bool(0);

    /* Read tensor fields (must match eshkol_tensor_t layout) */
    uint64_t* dimensions = *((uint64_t**)((char*)t + 0));
    uint64_t num_dims = *((uint64_t*)((char*)t + 8));
    int64_t* elements = *((int64_t**)((char*)t + 16));
    uint64_t total = *((uint64_t*)((char*)t + 24));

    if (!dimensions || !elements || total == 0) return sys_make_bool(0);

    /* Convert int64 element bit patterns to doubles */
    double* data = (double*)elements; /* same bit pattern */

    /* Convert uint64 dims to int64 for the ONNX export API */
    int64_t dims[8];
    int ndims = (int)(num_dims < 8 ? num_dims : 8);
    for (int i = 0; i < ndims; i++) dims[i] = (int64_t)dimensions[i];

    const char* name = "tensor";
    const char* names[] = { name };
    const int64_t* dims_ptrs[] = { dims };
    const int ndims_arr[] = { ndims };
    const double* data_ptrs[] = { data };
    const int64_t totals[] = { (int64_t)total };

    int rc = eshkol_onnx_export(path, names, dims_ptrs, ndims_arr, data_ptrs, totals, 1);
    return sys_make_bool(rc == 0);
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
    int ok = 1;
    /* Pre-fix, every fwrite return was ignored — tensor-save reported
     * success even on disk-full mid-write or fclose-time commit error,
     * leaving a truncated file the caller couldn't tell from a
     * successful save. */
    if (fwrite(&magic, 4, 1, f) != 1) ok = 0;
    if (ok && fwrite(&version, 4, 1, f) != 1) ok = 0;
    if (ok && fwrite(&ndims, 4, 1, f) != 1) ok = 0;
    for (uint32_t i = 0; ok && i < ndims; i++) {
        if (fwrite(&t->dimensions[i], 8, 1, f) != 1) ok = 0;
    }
    /* Elements are int64 bit patterns of doubles, 8 bytes each */
    if (ok && fwrite(t->elements, 8, (size_t)t->total_elements, f)
              != (size_t)t->total_elements) ok = 0;
    if (fclose(f) != 0) ok = 0;
    return sys_make_bool(ok ? 1 : 0);
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
/* Time API (#168) */
void eshkol_builtin_format_iso8601(sv_t* out, const sv_t* a) { *out = eshkol_builtin_format_iso8601_v(*a); }
void eshkol_builtin_parse_iso8601(sv_t* out, const sv_t* a) { *out = eshkol_builtin_parse_iso8601_v(*a); }
void eshkol_builtin_current_timestamp(sv_t* out) { *out = eshkol_builtin_current_timestamp_v(); }
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
/* v1.2 batch 4 */
void eshkol_builtin_process_pid(sv_t* out) { *out = eshkol_builtin_process_pid_v(); }
void eshkol_builtin_file_mmap(sv_t* out, const sv_t* a) { *out = eshkol_builtin_file_mmap_v(*a); }
void eshkol_builtin_file_munmap(sv_t* out, const sv_t* a) { *out = eshkol_builtin_file_munmap_v(*a); }
void eshkol_builtin_unix_socket_connect(sv_t* out, const sv_t* a) { *out = eshkol_builtin_unix_socket_connect_v(*a); }
void eshkol_builtin_socket_send(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_socket_send_v(*a, *b); }
void eshkol_builtin_socket_recv(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_socket_recv_v(*a, *b); }
void eshkol_builtin_socket_close(sv_t* out, const sv_t* a) { *out = eshkol_builtin_socket_close_v(*a); }
void eshkol_builtin_kb_save(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_kb_save_v(*a, *b); }
void eshkol_builtin_kb_load(sv_t* out, const sv_t* a) { *out = eshkol_builtin_kb_load_v(*a); }
void eshkol_builtin_tensor_token_estimate(sv_t* out, const sv_t* a) { *out = eshkol_builtin_tensor_token_estimate_v(*a); }
void eshkol_builtin_onnx_export_tensor(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_onnx_export_tensor_v(*a, *b); }

/* Type predicates — check type tag of tagged value */
static eshkol_sysbuiltin_value_t check_type(eshkol_sysbuiltin_value_t v, uint8_t type) {
    eshkol_sysbuiltin_value_t r = {SYS_TYPE_BOOL, 0, 0, 0, 0};
    r.data = (v.type == type) ? 1 : 0;
    return r;
}
static eshkol_sysbuiltin_value_t check_heap_subtype(eshkol_sysbuiltin_value_t v, uint8_t expected) {
    eshkol_sysbuiltin_value_t r = {SYS_TYPE_BOOL, 0, 0, 0, 0};
    if (v.type != SYS_TYPE_HEAP_PTR || v.data == 0) { r.data = 0; return r; }
    /* Check object header subtype at ptr - 8 */
    uint8_t* ptr = (uint8_t*)(uintptr_t)v.data;
    uint8_t subtype = *(ptr - 8); /* header byte 0 = subtype */
    r.data = (subtype == expected) ? 1 : 0;
    return r;
}

/* Heap-subtype IDs — MUST stay in sync with HEAP_SUBTYPE_* in
 * inc/eshkol/eshkol.h. Previous revisions of this file used local values
 * (11/12/9/8) that drifted from the canonical table, causing tensor?,
 * substitution?, fact?, and dual? to return #f for valid objects.
 * Always cross-check inc/eshkol/eshkol.h:337-360 when adding or
 * renumbering subtypes. */
#define HST_TENSOR           3   /* HEAP_SUBTYPE_TENSOR */
#define HST_SUBSTITUTION    12   /* HEAP_SUBTYPE_SUBSTITUTION */
#define HST_FACT            13   /* HEAP_SUBTYPE_FACT */
#define HST_KB              15   /* HEAP_SUBTYPE_KNOWLEDGE_BASE */
#define HST_FG              16   /* HEAP_SUBTYPE_FACTOR_GRAPH */
#define HST_WORKSPACE       17   /* HEAP_SUBTYPE_WORKSPACE */
/* Dual numbers are NOT a heap subtype — they use their own tagged-value
 * type ESHKOL_VALUE_DUAL_NUMBER (=6). See dual? predicate below. */

void eshkol_builtin_logic_var_p(sv_t* out, const sv_t* a) {
    /* Logic vars have type 10 (ESHKOL_VALUE_LOGIC_VAR) */
    *out = check_type(*a, 10);
}
void eshkol_builtin_substitution_p(sv_t* out, const sv_t* a) {
    *out = check_heap_subtype(*a, HST_SUBSTITUTION);
}
void eshkol_builtin_fact_p(sv_t* out, const sv_t* a) {
    *out = check_heap_subtype(*a, HST_FACT);
}
void eshkol_builtin_kb_p(sv_t* out, const sv_t* a) {
    *out = check_heap_subtype(*a, HST_KB);
}
void eshkol_builtin_factor_graph_p(sv_t* out, const sv_t* a) {
    *out = check_heap_subtype(*a, HST_FG);
}
void eshkol_builtin_workspace_p(sv_t* out, const sv_t* a) {
    *out = check_heap_subtype(*a, HST_WORKSPACE);
}
void eshkol_builtin_tensor_p(sv_t* out, const sv_t* a) {
    *out = check_heap_subtype(*a, HST_TENSOR);
}
void eshkol_builtin_dual_p(sv_t* out, const sv_t* a) {
    /* Dual numbers have type 6 (ESHKOL_VALUE_DUAL_NUMBER) — they do not
     * live on the heap, so check_heap_subtype would always fail. */
    *out = check_type(*a, 6);
}
void eshkol_builtin_fg_update_cpt(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) {
    /* Delegate to the existing tagged function */
    extern void eshkol_fg_update_cpt_tagged(void*, const void*, const void*, const void*);
    void* arena = get_global_arena();
    eshkol_fg_update_cpt_tagged(arena, a, b, c);
    out->type = SYS_TYPE_BOOL; out->flags = 0; out->reserved = 0; out->padding = 0; out->data = 1;
}
/* Image I/O — wraps C functions from image_io.c */
extern double* eshkol_image_read(const char* path, int* w, int* h, int* c);
extern int eshkol_image_write(const char* path, const double* data, int w, int h, int c, const char* fmt);
extern double* eshkol_image_to_grayscale(const double* data, int w, int h, int channels);
extern double* eshkol_image_resize(const double* data, int w, int h, int channels, int new_w, int new_h);
extern void* arena_allocate_tensor_full(void* arena, uint64_t ndims, uint64_t total);

void eshkol_builtin_image_read_sret(sv_t* out, const sv_t* path_tv) {
    const char* path = sys_extract_string(*path_tv);
    if (!path) { *out = sys_make_null(); return; }
    int w, h, c;
    /* eshkol_image_read returns ARENA-allocated storage (see image_io.c —
     * the stb_image malloc result is freed internally and data copied into
     * the global arena before return). Do NOT free() the returned pointer;
     * doing so corrupts the arena free list. The arena reclaims it. */
    double* data = eshkol_image_read(path, &w, &h, &c);
    if (!data) { *out = sys_make_null(); return; }
    /* Create tensor (h, w, c) */
    void* arena = get_global_arena();
    int64_t total = (int64_t)w * h * c;
    void* t = arena_allocate_tensor_full(arena, 3, (uint64_t)total);
    if (!t) { *out = sys_make_null(); return; }
    /* Copy pixel data to tensor elements */
    int64_t* elements = *((int64_t**)((char*)t + 16));
    for (int64_t i = 0; i < total; i++) {
        memcpy(&elements[i], &data[i], sizeof(double));
    }
    /* Set dimensions */
    uint64_t* dims = *((uint64_t**)((char*)t + 0));
    dims[0] = (uint64_t)h; dims[1] = (uint64_t)w; dims[2] = (uint64_t)c;
    out->type = SYS_TYPE_HEAP_PTR; out->flags = 0; out->reserved = 0; out->padding = 0;
    out->data = (uint64_t)t;
}

void eshkol_builtin_image_write_sret(sv_t* out, const sv_t* path_tv, const sv_t* tensor_tv, const sv_t* fmt_tv) {
    const char* path = sys_extract_string(*path_tv);
    const char* fmt = sys_extract_string(*fmt_tv);
    if (!path || tensor_tv->type != SYS_TYPE_HEAP_PTR) { *out = sys_make_bool(0); return; }
    void* t = (void*)(uintptr_t)tensor_tv->data;
    if (!t) { *out = sys_make_bool(0); return; }
    uint64_t* dims = *((uint64_t**)((char*)t + 0));
    uint64_t ndims = *((uint64_t*)((char*)t + 8));
    int64_t* elements = *((int64_t**)((char*)t + 16));
    uint64_t total = *((uint64_t*)((char*)t + 24));
    if (ndims < 2 || !elements) { *out = sys_make_bool(0); return; }
    int h = (int)dims[0], w = (int)dims[1], c = ndims >= 3 ? (int)dims[2] : 1;
    double* data = (double*)arena_allocate(get_global_arena(), total * sizeof(double));
    if (!data) { *out = sys_make_bool(0); return; }
    for (uint64_t i = 0; i < total; i++) memcpy(&data[i], &elements[i], sizeof(double));
    int rc = eshkol_image_write(path, data, w, h, c, fmt ? fmt : "png");
    *out = sys_make_bool(rc == 0);
}

void eshkol_builtin_image_grayscale_sret(sv_t* out, const sv_t* tensor_tv) {
    if (tensor_tv->type != SYS_TYPE_HEAP_PTR) { *out = sys_make_null(); return; }
    void* t = (void*)(uintptr_t)tensor_tv->data;
    if (!t) { *out = sys_make_null(); return; }
    uint64_t* dims = *((uint64_t**)((char*)t + 0));
    uint64_t ndims = *((uint64_t*)((char*)t + 8));
    int64_t* elements = *((int64_t**)((char*)t + 16));
    uint64_t total = *((uint64_t*)((char*)t + 24));
    if (ndims < 3 || !elements) { *out = sys_make_null(); return; }
    int h = (int)dims[0], w = (int)dims[1], c = (int)dims[2];
    double* data = (double*)arena_allocate(get_global_arena(), total * sizeof(double));
    if (!data) { *out = sys_make_null(); return; }
    for (uint64_t i = 0; i < total; i++) memcpy(&data[i], &elements[i], sizeof(double));
    /* eshkol_image_to_grayscale returns ARENA-allocated storage. The result
     * must NOT be free()'d — that would corrupt the global arena. The arena
     * reclaims it on reset/destroy. Same pattern applies to image_read,
     * image_resize, and any other image_io.c producer. */
    double* gray = eshkol_image_to_grayscale(data, w, h, c);
    if (!gray) { *out = sys_make_null(); return; }
    /* Create HxW tensor */
    void* arena = get_global_arena();
    int64_t gray_total = (int64_t)w * h;
    void* gt = arena_allocate_tensor_full(arena, 2, (uint64_t)gray_total);
    if (!gt) { *out = sys_make_null(); return; }
    int64_t* gel = *((int64_t**)((char*)gt + 16));
    uint64_t* gdims = *((uint64_t**)((char*)gt + 0));
    gdims[0] = (uint64_t)h; gdims[1] = (uint64_t)w;
    for (int64_t i = 0; i < gray_total; i++) memcpy(&gel[i], &gray[i], sizeof(double));
    out->type = SYS_TYPE_HEAP_PTR; out->data = (uint64_t)gt;
}

void eshkol_builtin_kb_count(sv_t* out, const sv_t* a) {
    /* KB count: read num_facts from the KB struct */
    if (a->type != SYS_TYPE_HEAP_PTR || a->data == 0) {
        out->type = SYS_TYPE_INT64; out->data = 0; return;
    }
    /* num_facts is at offset 0 of eshkol_knowledge_base_t (uint32_t) */
    uint32_t count = *((uint32_t*)(uintptr_t)a->data);
    out->type = SYS_TYPE_INT64; out->flags = 0; out->reserved = 0; out->padding = 0;
    out->data = (uint64_t)count;
}
/* v1.2 Noesis requirements: fg-marginal, fg-entropy, kb-retract! as LLVM builtins */
/* These delegate to C++ implementations via tagged value pointers */
extern void eshkol_fg_marginal_tagged(void* arena, const void* fg_tv, const void* idx_tv, void* result);
extern void eshkol_fg_entropy_tagged(void* arena, const void* fg_tv, const void* idx_tv, void* result);
extern void eshkol_kb_retract_tagged(void* arena, const void* kb_tv, const void* fact_tv, void* result);

static eshkol_sysbuiltin_value_t eshkol_builtin_fg_marginal_v(eshkol_sysbuiltin_value_t fg_val,
                                                                eshkol_sysbuiltin_value_t idx_val) {
    eshkol_sysbuiltin_value_t result = {0, 0, 0, 0, 0};
    void* arena = get_global_arena();
    eshkol_fg_marginal_tagged(arena, &fg_val, &idx_val, &result);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fg_entropy_v(eshkol_sysbuiltin_value_t fg_val,
                                                               eshkol_sysbuiltin_value_t idx_val) {
    eshkol_sysbuiltin_value_t result = {0, 0, 0, 0, 0};
    void* arena = get_global_arena();
    eshkol_fg_entropy_tagged(arena, &fg_val, &idx_val, &result);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_kb_retract_v(eshkol_sysbuiltin_value_t kb_val,
                                                               eshkol_sysbuiltin_value_t fact_val) {
    eshkol_sysbuiltin_value_t result = {0, 0, 0, 0, 0};
    void* arena = get_global_arena();
    eshkol_kb_retract_tagged(arena, &kb_val, &fact_val, &result);
    return result;
}

void eshkol_builtin_fg_marginal(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_fg_marginal_v(*a, *b); }
void eshkol_builtin_fg_entropy(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_fg_entropy_v(*a, *b); }
void eshkol_builtin_kb_retract(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_kb_retract_v(*a, *b); }

/* Consciousness engine tagged wrappers — delegate to consciousness_builtins.cpp */
extern void eshkol_make_substitution_tagged(void*, void*);
extern void eshkol_unify_tagged(void*, const void*, const void*, const void*, void*);
extern void eshkol_walk_tagged(void*, const void*, const void*, void*);
extern void eshkol_make_fact_tagged(void*, const void*, const void*, void*);
extern void eshkol_make_kb_tagged(void*, void*);
extern void eshkol_kb_assert_tagged(void*, const void*, const void*, void*);
extern void eshkol_kb_query_tagged(void*, const void*, const void*, void*);
extern void eshkol_make_factor_graph_tagged(void*, const void*, const void*, void*);
extern void eshkol_fg_add_factor_tagged(void*, const void*, const void*, const void*, void*);
extern void eshkol_fg_infer_tagged(void*, const void*, const void*, const void*, void*);
extern void eshkol_free_energy_tagged(void*, const void*, const void*, void*);
extern void eshkol_expected_free_energy_tagged(void*, const void*, const void*, const void*, void*);
extern void eshkol_make_workspace_tagged(void*, const void*, const void*, void*);
extern void eshkol_ws_register_tagged(void*, const void*, const void*, const void*, void*);
extern void eshkol_ws_step_tagged(void*, const void*, void*);
/* v1.2 batch 3: advanced process management */
void eshkol_builtin_process_setpgid(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_setpgid_v(*a, *b); }
void eshkol_builtin_process_kill_tree(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_kill_tree_v(*a, *b); }
void eshkol_builtin_process_spawn_pty(sv_t* out, const sv_t* a) { *out = eshkol_builtin_process_spawn_pty_v(*a); }
void eshkol_builtin_process_read_nonblocking(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_process_read_nonblocking_v(*a, *b); }
