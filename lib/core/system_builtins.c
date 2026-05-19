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
#include <ctype.h>
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
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#ifndef ESHKOL_VM_WASM
#include <netdb.h>
#endif
#include <glob.h>
#include <fnmatch.h>
#ifndef ESHKOL_VM_WASM
#include <regex.h>
#endif
#ifdef __APPLE__
#include <util.h>
#else
#include <pty.h>
#endif
#else
#include <windows.h>
#include <direct.h>
#include <process.h>
#include <sys/stat.h>
#include <io.h>
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
extern void* arena_allocate_cons_with_header(void* arena);

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

static eshkol_sysbuiltin_value_t sys_make_string_len(const char* s, size_t len) {
    if (!s) return sys_make_null();
    void* arena = get_global_arena();
    if (!arena) return sys_make_null();
    /* Use arena_allocate_string_with_header so the string has a proper
     * object header — required by eshkol_display_value and the runtime. */
    char* copy = arena_allocate_string_with_header(arena, len);
    if (!copy) return sys_make_null();
    memcpy(copy, s, len);
    copy[len] = '\0';
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.flags = 0x01; /* string subtype */
    v.data = (uint64_t)copy;
    return v;
}

static eshkol_sysbuiltin_value_t sys_make_string(const char* s) {
    return s ? sys_make_string_len(s, strlen(s)) : sys_make_null();
}

static eshkol_sysbuiltin_value_t sys_make_pair(eshkol_sysbuiltin_value_t car,
                                                eshkol_sysbuiltin_value_t cdr) {
    void* arena = get_global_arena();
    if (!arena) return sys_make_null();
    void* cell = arena_allocate_cons_with_header(arena);
    if (!cell) return sys_make_null();
    memcpy(cell, &car, sizeof(car));
    memcpy((char*)cell + sizeof(car), &cdr, sizeof(cdr));
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.flags = 0x00; /* cons subtype */
    v.data = (uint64_t)cell;
    return v;
}

static eshkol_sysbuiltin_value_t sys_alist_entry(const char* key,
                                                  eshkol_sysbuiltin_value_t value) {
    return sys_make_pair(sys_make_string(key), value);
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

static int64_t sys_extract_int64(eshkol_sysbuiltin_value_t v) {
    if (v.type == SYS_TYPE_DOUBLE) {
        double d = 0.0;
        memcpy(&d, &v.data, sizeof(double));
        return (int64_t)d;
    }
    int64_t i = 0;
    memcpy(&i, &v.data, sizeof(int64_t));
    return i;
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

static eshkol_sysbuiltin_value_t eshkol_builtin_format_relative_v(eshkol_sysbuiltin_value_t seconds_val) {
    int64_t seconds_ago = sys_extract_int64(seconds_val);
    if (seconds_ago < 0) seconds_ago = 0;
    char buf[32];
    if (seconds_ago < 60)
        snprintf(buf, sizeof(buf), "%llds ago", (long long)seconds_ago);
    else if (seconds_ago < 3600)
        snprintf(buf, sizeof(buf), "%lldm ago", (long long)(seconds_ago / 60));
    else if (seconds_ago < 86400)
        snprintf(buf, sizeof(buf), "%lldh ago", (long long)(seconds_ago / 3600));
    else
        snprintf(buf, sizeof(buf), "%lldd ago", (long long)(seconds_ago / 86400));
    return sys_make_string(buf);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_local_timezone_offset_v(void) {
#if defined(ESHKOL_VM_WASM)
    return sys_make_int64(0);
#else
    time_t now = time(NULL);
    struct tm local_tm;
    struct tm utc_tm;
#if defined(_WIN32)
    if (localtime_s(&local_tm, &now) != 0 || gmtime_s(&utc_tm, &now) != 0)
        return sys_make_int64(0);
#else
    if (!localtime_r(&now, &local_tm) || !gmtime_r(&now, &utc_tm))
        return sys_make_int64(0);
#endif
    return sys_make_int64((int64_t)difftime(mktime(&local_tm), mktime(&utc_tm)));
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

static eshkol_sysbuiltin_value_t eshkol_builtin_executable_path_v(eshkol_sysbuiltin_value_t name_val) {
    const char* name = sys_extract_string(name_val);
    if (!name || !*name) return sys_make_bool(0);
#ifndef _WIN32
    if (strchr(name, '/')) {
        if (access(name, X_OK) == 0) return sys_make_string(name);
        return sys_make_bool(0);
    }
    const char* path_env = getenv("PATH");
    if (!path_env || !*path_env) return sys_make_bool(0);
    char* path_copy = strdup(path_env);
    if (!path_copy) return sys_make_bool(0);
    char* save = NULL;
    for (char* dir = strtok_r(path_copy, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
        if (!*dir) dir = ".";
        char full[PATH_MAX];
        int n = snprintf(full, sizeof(full), "%s/%s", dir, name);
        if (n > 0 && n < (int)sizeof(full) && access(full, X_OK) == 0) {
            eshkol_sysbuiltin_value_t result = sys_make_string(full);
            free(path_copy);
            return result;
        }
    }
    free(path_copy);
    return sys_make_bool(0);
#else
    char result[MAX_PATH];
    DWORD n = SearchPathA(NULL, name, ".exe", MAX_PATH, result, NULL);
    if (n > 0 && n < MAX_PATH) return sys_make_string(result);
    n = SearchPathA(NULL, name, NULL, MAX_PATH, result, NULL);
    if (n > 0 && n < MAX_PATH) return sys_make_string(result);
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_monotonic_time_ms_v(void) {
#ifndef _WIN32
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return sys_make_int64((int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000);
#endif
    return sys_make_int64((int64_t)time(NULL) * 1000);
#else
    return sys_make_int64((int64_t)GetTickCount64());
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_temp_directory_v(void) {
#ifndef _WIN32
    const char* tmp = getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = getenv("TMP");
    if (!tmp || !*tmp) tmp = getenv("TEMP");
    if (!tmp || !*tmp) tmp = "/tmp";
    return sys_make_string(tmp);
#else
    char buf[MAX_PATH];
    DWORD n = GetTempPathA(MAX_PATH, buf);
    if (n > 0 && n < MAX_PATH) {
        while (n > 1 && (buf[n - 1] == '\\' || buf[n - 1] == '/')) buf[--n] = '\0';
        return sys_make_string(buf);
    }
    const char* tmp = getenv("TEMP");
    return sys_make_string((tmp && *tmp) ? tmp : ".");
#endif
}

typedef struct {
    int active;
    int64_t handle;
} eshkol_sys_sleep_inhibitor_t;

static eshkol_sys_sleep_inhibitor_t g_sys_sleep_inhibitors[16];
static int64_t g_sys_next_sleep_inhibitor = 1;

static eshkol_sysbuiltin_value_t eshkol_builtin_prevent_sleep_v(eshkol_sysbuiltin_value_t reason_val) {
    (void)reason_val;
    int slot = -1;
    for (int i = 1; i < (int)(sizeof(g_sys_sleep_inhibitors) / sizeof(g_sys_sleep_inhibitors[0])); i++) {
        if (!g_sys_sleep_inhibitors[i].active) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return sys_make_bool(0);
    int64_t handle = g_sys_next_sleep_inhibitor++;
    if (g_sys_next_sleep_inhibitor <= 0) g_sys_next_sleep_inhibitor = 1;
    g_sys_sleep_inhibitors[slot].active = 1;
    g_sys_sleep_inhibitors[slot].handle = handle;
#ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
#endif
    return sys_make_int64(handle);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_allow_sleep_v(eshkol_sysbuiltin_value_t handle_val) {
    int64_t handle = (int64_t)handle_val.data;
    if (handle <= 0) return sys_make_bool(0);
    for (int i = 1; i < (int)(sizeof(g_sys_sleep_inhibitors) / sizeof(g_sys_sleep_inhibitors[0])); i++) {
        if (g_sys_sleep_inhibitors[i].active && g_sys_sleep_inhibitors[i].handle == handle) {
            memset(&g_sys_sleep_inhibitors[i], 0, sizeof(g_sys_sleep_inhibitors[i]));
#ifdef _WIN32
            int still_active = 0;
            for (int j = 1; j < (int)(sizeof(g_sys_sleep_inhibitors) / sizeof(g_sys_sleep_inhibitors[0])); j++)
                if (g_sys_sleep_inhibitors[j].active) still_active = 1;
            if (!still_active) SetThreadExecutionState(ES_CONTINUOUS);
#endif
            return sys_make_bool(1);
        }
    }
    return sys_make_bool(0);
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

static const char* sys_temp_dir_or_default(const char* dir) {
    if (dir && *dir) return dir;
#ifndef _WIN32
    const char* tmp = getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = getenv("TMP");
    if (!tmp || !*tmp) tmp = getenv("TEMP");
    return (tmp && *tmp) ? tmp : "/tmp";
#else
    return ".";
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_make_temp_file_v(
    eshkol_sysbuiltin_value_t prefix_val,
    eshkol_sysbuiltin_value_t suffix_val,
    eshkol_sysbuiltin_value_t dir_val) {
    const char* prefix = sys_extract_string(prefix_val);
    const char* suffix = sys_extract_string(suffix_val);
    const char* dir = sys_temp_dir_or_default(sys_extract_string(dir_val));
    if (!prefix || !suffix || !dir || !*dir) return sys_make_bool(0);
#if !defined(_WIN32)
    const char* sep = (dir[strlen(dir) - 1] == '/') ? "" : "/";
    struct timeval tv;
    gettimeofday(&tv, NULL);
    for (int i = 0; i < 128; i++) {
        char path[4096];
        uint64_t nonce = ((uint64_t)(uint32_t)getpid() << 32) ^
                         (uint64_t)tv.tv_usec ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
        int n = snprintf(path, sizeof(path), "%s%s%s%016llx%s",
                         dir, sep, prefix, (unsigned long long)nonce, suffix);
        if (n <= 0 || n >= (int)sizeof(path)) break;
        int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd >= 0) {
            close(fd);
            return sys_make_string(path);
        }
        if (errno != EEXIST) break;
    }
#endif
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_make_temp_dir_v(
    eshkol_sysbuiltin_value_t prefix_val,
    eshkol_sysbuiltin_value_t dir_val) {
    const char* prefix = sys_extract_string(prefix_val);
    const char* dir = sys_temp_dir_or_default(sys_extract_string(dir_val));
    if (!prefix || !dir || !*dir) return sys_make_bool(0);
#if !defined(_WIN32)
    const char* sep = (dir[strlen(dir) - 1] == '/') ? "" : "/";
    struct timeval tv;
    gettimeofday(&tv, NULL);
    for (int i = 0; i < 128; i++) {
        char path[4096];
        uint64_t nonce = ((uint64_t)(uint32_t)getpid() << 32) ^
                         (uint64_t)tv.tv_usec ^ ((uint64_t)i * 0x9e3779b97f4a7c15ULL);
        int n = snprintf(path, sizeof(path), "%s%s%s%016llx",
                         dir, sep, prefix, (unsigned long long)nonce);
        if (n <= 0 || n >= (int)sizeof(path)) break;
        if (mkdir(path, 0700) == 0) return sys_make_string(path);
        if (errno != EEXIST) break;
    }
#endif
    return sys_make_bool(0);
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

static eshkol_sysbuiltin_value_t eshkol_builtin_fork_v(void) {
#if !defined(_WIN32)
    pid_t pid = fork();
    if (pid >= 0) return sys_make_int64((int64_t)pid);
#endif
    return sys_make_bool(0);
}

static int sys_execv_argv_from_list(eshkol_sysbuiltin_value_t list,
                                    char** argv,
                                    int max_args) {
    int argc = 0;
    eshkol_sysbuiltin_value_t cur = list;
    while (cur.type == SYS_TYPE_HEAP_PTR && cur.flags == 0x00 && cur.data != 0) {
        if (argc >= max_args - 1) return -1;
        eshkol_sysbuiltin_value_t car;
        eshkol_sysbuiltin_value_t cdr;
        memcpy(&car, (void*)(uintptr_t)cur.data, sizeof(car));
        memcpy(&cdr, (char*)(uintptr_t)cur.data + sizeof(car), sizeof(cdr));
        const char* arg = sys_extract_string(car);
        if (!arg) return -1;
        argv[argc++] = (char*)arg;
        cur = cdr;
    }
    if (cur.type != SYS_TYPE_NULL) return -1;
    argv[argc] = NULL;
    return argc;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_execv_v(eshkol_sysbuiltin_value_t path_val,
                                                         eshkol_sysbuiltin_value_t argv_val) {
    const char* path = sys_extract_string(path_val);
    if (!path || !*path) return sys_make_bool(0);
#if !defined(_WIN32)
    char* argv[256];
    int argc = sys_execv_argv_from_list(argv_val, argv, (int)(sizeof(argv) / sizeof(argv[0])));
    if (argc >= 0) {
        if (argc == 0) {
            argv[argc++] = (char*)path;
            argv[argc] = NULL;
        }
        execv(path, argv);
    }
#else
    (void)argv_val;
#endif
    return sys_make_bool(0);
}

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

/* Terminal helpers.  These mirror the standalone VM surface with conservative
 * compiled-runtime behavior: write escape sequences only for real TTY output
 * and return #f for interactive reads that need a terminal response. */
static int eshkol_sys_stdout_is_tty(void) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    return isatty(STDOUT_FILENO);
#else
    return 0;
#endif
}

static void eshkol_sys_term_write_tty(const char* s) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    if (s && eshkol_sys_stdout_is_tty()) {
        fputs(s, stdout);
        fflush(stdout);
    }
#else
    (void)s;
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_set_scroll_region_v(
    eshkol_sysbuiltin_value_t top_val,
    eshkol_sysbuiltin_value_t bottom_val) {
    int64_t top = (int64_t)top_val.data;
    int64_t bottom = (int64_t)bottom_val.data;
    if (top <= 0 || bottom < top) return sys_make_bool(0);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    if (eshkol_sys_stdout_is_tty()) {
        printf("\033[%lld;%lldr", (long long)top, (long long)bottom);
        fflush(stdout);
    }
#endif
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_reset_scroll_region_v(void) {
    eshkol_sys_term_write_tty("\033[r");
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_enable_mouse_v(void) {
    eshkol_sys_term_write_tty("\033[?1000h\033[?1006h");
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_disable_mouse_v(void) {
    eshkol_sys_term_write_tty("\033[?1006l\033[?1000l");
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_read_mouse_event_v(
    eshkol_sysbuiltin_value_t timeout_val) {
    (void)timeout_val;
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_enable_alternate_screen_v(void) {
    eshkol_sys_term_write_tty("\033[?1049h");
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_disable_alternate_screen_v(void) {
    eshkol_sys_term_write_tty("\033[?1049l");
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_clipboard_write_v(
    eshkol_sysbuiltin_value_t text_val) {
    const char* text = sys_extract_string(text_val);
    if (!text) return sys_make_bool(0);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    if (eshkol_sys_stdout_is_tty() && strlen(text) <= 4096) {
        static const char table[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        size_t len = strlen(text);
        fputs("\033]52;c;", stdout);
        for (size_t i = 0; i < len; i += 3) {
            unsigned b0 = (unsigned char)text[i];
            unsigned b1 = (i + 1 < len) ? (unsigned char)text[i + 1] : 0;
            unsigned b2 = (i + 2 < len) ? (unsigned char)text[i + 2] : 0;
            fputc(table[(b0 >> 2) & 0x3F], stdout);
            fputc(table[((b0 << 4) | (b1 >> 4)) & 0x3F], stdout);
            fputc(i + 1 < len ? table[((b1 << 2) | (b2 >> 6)) & 0x3F] : '=', stdout);
            fputc(i + 2 < len ? table[b2 & 0x3F] : '=', stdout);
        }
        fputc('\a', stdout);
        fflush(stdout);
    }
#endif
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_clipboard_read_v(void) {
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_hyperlink_v(
    eshkol_sysbuiltin_value_t url_val,
    eshkol_sysbuiltin_value_t text_val) {
    const char* url = sys_extract_string(url_val);
    const char* text = sys_extract_string(text_val);
    if (!url || !text) return sys_make_bool(0);
    size_t len = strlen(url) + strlen(text) + 16;
    void* arena = get_global_arena();
    if (!arena) return sys_make_bool(0);
    char* out = arena_allocate_string_with_header(arena, len);
    if (!out) return sys_make_bool(0);
    int n = snprintf(out, len + 1, "\033]8;;%s\033\\%s\033]8;;\033\\", url, text);
    if (n <= 0 || (size_t)n > len) return sys_make_bool(0);
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.flags = 0x01;
    v.data = (uint64_t)out;
    return v;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_detect_capabilities_v(void) {
    const char* term = getenv("TERM");
    const char* colorterm = getenv("COLORTERM");
    const char* lang = getenv("LANG");
    int color_depth = 8;
    if (colorterm && (strstr(colorterm, "truecolor") || strstr(colorterm, "24bit")))
        color_depth = 24;
    else if (term && strstr(term, "256color"))
        color_depth = 8;
    int unicode = (lang && (strstr(lang, "UTF-8") || strstr(lang, "utf8"))) ||
                  (term && strstr(term, "utf"));
    char buf[128];
    snprintf(buf, sizeof(buf), "color-depth=%d unicode=%d tty=%d",
             color_depth, unicode ? 1 : 0, eshkol_sys_stdout_is_tty());
    return sys_make_string(buf);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_term_bell_v(void) {
    eshkol_sys_term_write_tty("\a");
    return sys_make_bool(1);
}

typedef struct {
    int active;
    int recursive;
    int exists;
    int64_t mtime_ns;
    int64_t size;
    char path[1024];
} eshkol_sys_file_watcher_t;

static eshkol_sys_file_watcher_t g_sys_file_watchers[32];

static void eshkol_sys_file_watch_signature(const char* path,
                                            int* exists,
                                            int64_t* mtime_ns,
                                            int64_t* size) {
    if (exists) *exists = 0;
    if (mtime_ns) *mtime_ns = 0;
    if (size) *size = 0;
    if (!path || !*path) return;
#ifdef _WIN32
    struct _stat64 st;
    if (_stat64(path, &st) == 0) {
        if (exists) *exists = 1;
        if (mtime_ns) *mtime_ns = (int64_t)st.st_mtime * 1000000000LL;
        if (size) *size = (int64_t)st.st_size;
    }
#else
    struct stat st;
    if (stat(path, &st) == 0) {
        if (exists) *exists = 1;
#ifdef __APPLE__
        if (mtime_ns) *mtime_ns = (int64_t)st.st_mtimespec.tv_sec * 1000000000LL +
                                  (int64_t)st.st_mtimespec.tv_nsec;
#else
        if (mtime_ns) *mtime_ns = (int64_t)st.st_mtim.tv_sec * 1000000000LL +
                                  (int64_t)st.st_mtim.tv_nsec;
#endif
        if (size) *size = (int64_t)st.st_size;
    }
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fs_watch_start_v(eshkol_sysbuiltin_value_t path_val,
                                                                  int recursive) {
    const char* path = sys_extract_string(path_val);
    if (!path || !*path || strlen(path) >= sizeof(g_sys_file_watchers[0].path))
        return sys_make_bool(0);

    int slot = -1;
    for (int i = 1; i < (int)(sizeof(g_sys_file_watchers) / sizeof(g_sys_file_watchers[0])); ++i) {
        if (!g_sys_file_watchers[i].active) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return sys_make_bool(0);

    int exists = 0;
    int64_t mtime_ns = 0;
    int64_t size = 0;
    eshkol_sys_file_watch_signature(path, &exists, &mtime_ns, &size);
    g_sys_file_watchers[slot].active = 1;
    g_sys_file_watchers[slot].recursive = recursive ? 1 : 0;
    g_sys_file_watchers[slot].exists = exists;
    g_sys_file_watchers[slot].mtime_ns = mtime_ns;
    g_sys_file_watchers[slot].size = size;
    snprintf(g_sys_file_watchers[slot].path, sizeof(g_sys_file_watchers[slot].path), "%s", path);
    return sys_make_int64(slot);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fs_watch_native_v(eshkol_sysbuiltin_value_t path_val,
                                                                   eshkol_sysbuiltin_value_t callback_val) {
    (void)callback_val;
    return eshkol_builtin_fs_watch_start_v(path_val, 0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fs_watch_recursive_v(eshkol_sysbuiltin_value_t path_val,
                                                                      eshkol_sysbuiltin_value_t callback_val) {
    (void)callback_val;
    return eshkol_builtin_fs_watch_start_v(path_val, 1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fs_watch_poll_v(eshkol_sysbuiltin_value_t handle_val) {
    int handle = (int)((int64_t)handle_val.data);
    if (handle <= 0 || handle >= (int)(sizeof(g_sys_file_watchers) / sizeof(g_sys_file_watchers[0])) ||
        !g_sys_file_watchers[handle].active)
        return sys_make_bool(0);

    int exists = 0;
    int64_t mtime_ns = 0;
    int64_t size = 0;
    eshkol_sys_file_watch_signature(g_sys_file_watchers[handle].path, &exists, &mtime_ns, &size);

    const char* event = NULL;
    if (g_sys_file_watchers[handle].exists && !exists)
        event = "delete";
    else if (!g_sys_file_watchers[handle].exists && exists)
        event = "create";
    else if (exists && (mtime_ns != g_sys_file_watchers[handle].mtime_ns ||
                        size != g_sys_file_watchers[handle].size))
        event = "change";

    g_sys_file_watchers[handle].exists = exists;
    g_sys_file_watchers[handle].mtime_ns = mtime_ns;
    g_sys_file_watchers[handle].size = size;
    if (!event) return sys_make_bool(0);

    void* arena = get_global_arena();
    if (!arena) return sys_make_bool(0);
    char tmp[1200];
    int n = snprintf(tmp, sizeof(tmp), "%s\t%s", event, g_sys_file_watchers[handle].path);
    if (n <= 0 || n >= (int)sizeof(tmp)) return sys_make_bool(0);
    char* out = arena_allocate_string_with_header(arena, (size_t)n);
    if (!out) return sys_make_bool(0);
    memcpy(out, tmp, (size_t)n + 1);
    eshkol_sysbuiltin_value_t v = {0, 0, 0, 0, 0};
    v.type = SYS_TYPE_HEAP_PTR;
    v.data = (uint64_t)out;
    return v;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fs_unwatch_v(eshkol_sysbuiltin_value_t handle_val) {
    int handle = (int)((int64_t)handle_val.data);
    if (handle > 0 && handle < (int)(sizeof(g_sys_file_watchers) / sizeof(g_sys_file_watchers[0])) &&
        g_sys_file_watchers[handle].active) {
        memset(&g_sys_file_watchers[handle], 0, sizeof(g_sys_file_watchers[handle]));
        return sys_make_bool(1);
    }
    return sys_make_bool(0);
}

static int sys_ansi_escape_len(const char* s, int len, int pos) {
    if (!s || pos < 0 || pos >= len) return 0;
    unsigned char c = (unsigned char)s[pos];

    if (c == 0x9B) {
        int i = pos + 1;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i++];
            if (ch >= 0x40 && ch <= 0x7E) return i - pos;
        }
        return len - pos;
    }
    if (c == 0x9D) {
        int i = pos + 1;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i];
            if (ch == 0x07) return i + 1 - pos;
            if (ch == 0x1B && i + 1 < len && s[i + 1] == '\\') return i + 2 - pos;
            i++;
        }
        return len - pos;
    }
    if (c != 0x1B) return 0;
    if (pos + 1 >= len) return 1;

    unsigned char next = (unsigned char)s[pos + 1];
    if (next == '[') {
        int i = pos + 2;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i++];
            if (ch >= 0x40 && ch <= 0x7E) return i - pos;
        }
        return len - pos;
    }
    if (next == ']' || next == 'P' || next == '^' || next == '_' || next == 'X') {
        int i = pos + 2;
        while (i < len) {
            unsigned char ch = (unsigned char)s[i];
            if (ch == 0x07) return i + 1 - pos;
            if (ch == 0x1B && i + 1 < len && s[i + 1] == '\\') return i + 2 - pos;
            i++;
        }
        return len - pos;
    }
    if (strchr("()*+-./", next)) return (pos + 2 < len) ? 3 : 2;
    return 2;
}

static int sys_display_is_wide_char(uint32_t cp) {
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1;
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 1;
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1;
    if (cp >= 0xFF01 && cp <= 0xFF60) return 1;
    if (cp >= 0xFFE0 && cp <= 0xFFE6) return 1;
    if (cp >= 0x2E80 && cp <= 0x303E) return 1;
    if (cp >= 0x3040 && cp <= 0x30FF) return 1;
    if (cp >= 0x31F0 && cp <= 0x31FF) return 1;
    if (cp >= 0x1F300 && cp <= 0x1F9FF) return 1;
    if (cp >= 0x1FA00 && cp <= 0x1FAFF) return 1;
    if (cp >= 0x2600 && cp <= 0x27BF) return 1;
    return 0;
}

static int sys_display_is_zero_width(uint32_t cp) {
    if (cp >= 0x0300 && cp <= 0x036F) return 1;
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return 1;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return 1;
    if (cp >= 0x20D0 && cp <= 0x20FF) return 1;
    if (cp >= 0xFE20 && cp <= 0xFE2F) return 1;
    if (cp == 0x200B || cp == 0x200C || cp == 0x200D ||
        cp == 0x200E || cp == 0x200F || cp == 0xFEFF) return 1;
    if (cp >= 0xFE00 && cp <= 0xFE0F) return 1;
    if (cp >= 0xE0100 && cp <= 0xE01EF) return 1;
    return 0;
}

static uint32_t sys_decode_utf8_display(const char* str, int len, int* pos) {
    unsigned char c = (unsigned char)str[*pos];
    uint32_t cp = 0xFFFD;
    int bytes = 1;
    if (c < 0x80) { cp = c; bytes = 1; }
    else if (c >= 0xC2 && c < 0xE0) { cp = c & 0x1F; bytes = 2; }
    else if (c >= 0xE0 && c < 0xF0) { cp = c & 0x0F; bytes = 3; }
    else if (c >= 0xF0 && c < 0xF5) { cp = c & 0x07; bytes = 4; }
    if (*pos + bytes > len) {
        (*pos)++;
        return 0xFFFD;
    }
    for (int i = 1; i < bytes; i++) {
        unsigned char cc = (unsigned char)str[*pos + i];
        if ((cc & 0xC0) != 0x80) {
            (*pos)++;
            return 0xFFFD;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    *pos += bytes;
    return cp;
}

static int64_t sys_string_display_width_bytes(const char* data, int len) {
    if (!data || len <= 0) return 0;
    int64_t width = 0;
    int pos = 0;
    while (pos < len) {
        int skip = sys_ansi_escape_len(data, len, pos);
        if (skip > 0) {
            pos += skip;
            continue;
        }
        uint32_t cp = sys_decode_utf8_display(data, len, &pos);
        if (sys_display_is_zero_width(cp)) continue;
        width += sys_display_is_wide_char(cp) ? 2 : 1;
    }
    return width;
}

static int sys_display_prefix_byte_len(const char* data, int len, int64_t max_cols) {
    if (!data || len <= 0 || max_cols < 0) return 0;
    int64_t width = 0;
    int pos = 0;
    int end = 0;
    while (pos < len) {
        int skip = sys_ansi_escape_len(data, len, pos);
        if (skip > 0) {
            pos += skip;
            end = pos;
            continue;
        }
        int before = pos;
        uint32_t cp = sys_decode_utf8_display(data, len, &pos);
        if (pos <= before) pos = before + 1;
        int char_width = 0;
        if (!sys_display_is_zero_width(cp))
            char_width = sys_display_is_wide_char(cp) ? 2 : 1;
        if (width + char_width > max_cols) break;
        width += char_width;
        end = pos;
    }
    return end;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ansi_strip_v(eshkol_sysbuiltin_value_t str_val) {
    const char* input = sys_extract_string(str_val);
    if (!input) return sys_make_bool(0);
    int len = (int)strlen(input);
    char* out = (char*)malloc((size_t)len + 1);
    if (!out) return sys_make_bool(0);

    int i = 0;
    int j = 0;
    while (i < len) {
        int skip = sys_ansi_escape_len(input, len, i);
        if (skip > 0) {
            i += skip;
            continue;
        }
        out[j++] = input[i++];
    }
    out[j] = '\0';
    eshkol_sysbuiltin_value_t result = sys_make_string(out);
    free(out);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_string_display_width_v(eshkol_sysbuiltin_value_t str_val) {
    const char* input = sys_extract_string(str_val);
    if (!input) return sys_make_int64(0);
    return sys_make_int64(sys_string_display_width_bytes(input, (int)strlen(input)));
}

static eshkol_sysbuiltin_value_t eshkol_builtin_string_truncate_display_v(
    eshkol_sysbuiltin_value_t str_val,
    eshkol_sysbuiltin_value_t max_val,
    eshkol_sysbuiltin_value_t suffix_val) {
    const char* input = sys_extract_string(str_val);
    if (!input) return sys_make_string("");

    int64_t max_cols = (int64_t)max_val.data;
    if (max_cols < 0) max_cols = 0;
    int input_len = (int)strlen(input);
    int64_t full_width = sys_string_display_width_bytes(input, input_len);
    if (full_width <= max_cols) return sys_make_string(input);

    const char* suffix = sys_extract_string(suffix_val);
    if (!suffix) suffix = "";
    int suffix_len = (int)strlen(suffix);
    int64_t suffix_width = sys_string_display_width_bytes(suffix, suffix_len);

    int prefix_len = 0;
    int append_suffix_len = suffix_len;
    if (suffix_width <= max_cols) {
        prefix_len = sys_display_prefix_byte_len(input, input_len, max_cols - suffix_width);
    } else {
        append_suffix_len = sys_display_prefix_byte_len(suffix, suffix_len, max_cols);
    }

    size_t out_len = (size_t)prefix_len + (size_t)append_suffix_len;
    char* out = (char*)malloc(out_len + 1);
    if (!out) return sys_make_bool(0);
    if (prefix_len > 0) memcpy(out, input, (size_t)prefix_len);
    if (append_suffix_len > 0)
        memcpy(out + prefix_len, suffix, (size_t)append_suffix_len);
    out[out_len] = '\0';
    eshkol_sysbuiltin_value_t result = sys_make_string(out);
    free(out);
    return result;
}

static int sys_url_is_unreserved(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_' ||
           c == '.' || c == '~';
}

static int sys_url_hex_value(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_url_encode_v(eshkol_sysbuiltin_value_t str_val) {
    const char* input = sys_extract_string(str_val);
    if (!input) return sys_make_bool(0);
    static const char hex[] = "0123456789ABCDEF";
    size_t len = strlen(input);
    char* out = (char*)malloc(len * 3 + 1);
    if (!out) return sys_make_bool(0);
    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)input[i];
        if (sys_url_is_unreserved(c)) {
            out[pos++] = (char)c;
        } else {
            out[pos++] = '%';
            out[pos++] = hex[(c >> 4) & 0x0F];
            out[pos++] = hex[c & 0x0F];
        }
    }
    out[pos] = '\0';
    eshkol_sysbuiltin_value_t result = sys_make_string(out);
    free(out);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_url_decode_v(eshkol_sysbuiltin_value_t str_val) {
    const char* input = sys_extract_string(str_val);
    if (!input) return sys_make_bool(0);
    size_t len = strlen(input);
    char* out = (char*)malloc(len + 1);
    if (!out) return sys_make_bool(0);
    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)input[i];
        if (c == '%' && i + 2 < len) {
            int hi = sys_url_hex_value((unsigned char)input[i + 1]);
            int lo = sys_url_hex_value((unsigned char)input[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out[pos++] = (char)((hi << 4) | lo);
                i += 2;
                continue;
            }
        }
        out[pos++] = (c == '+') ? ' ' : (char)c;
    }
    out[pos] = '\0';
    eshkol_sysbuiltin_value_t result = sys_make_string(out);
    free(out);
    return result;
}

static const char* sys_find_url_sep(const char* p) {
    while (*p) {
        if (*p == '/' || *p == '?' || *p == '#') return p;
        p++;
    }
    return p;
}

static int sys_span_eq_cstr(const char* p, size_t len, const char* s) {
    return p && s && len == strlen(s) && memcmp(p, s, len) == 0;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_url_parse_v(eshkol_sysbuiltin_value_t str_val) {
    const char* input = sys_extract_string(str_val);
    if (!input || !*input) return sys_make_bool(0);
    const char* scheme_end = strstr(input, "://");
    if (!scheme_end || scheme_end == input) return sys_make_bool(0);

    const char* authority = scheme_end + 3;
    const char* authority_end = sys_find_url_sep(authority);
    if (authority == authority_end) return sys_make_bool(0);

    const char* host_start = authority;
    const char* host_end = authority_end;
    int64_t port = 0;
    const char* colon = NULL;
    for (const char* p = authority_end; p > authority; p--) {
        if (*(p - 1) == ':') {
            colon = p - 1;
            break;
        }
    }
    if (colon && colon + 1 < authority_end) {
        int all_digits = 1;
        int64_t parsed = 0;
        for (const char* p = colon + 1; p < authority_end; p++) {
            if (*p < '0' || *p > '9') { all_digits = 0; break; }
            parsed = parsed * 10 + (*p - '0');
            if (parsed > 65535) { all_digits = 0; break; }
        }
        if (all_digits) {
            host_end = colon;
            port = parsed;
        }
    }
    if (host_start == host_end) return sys_make_bool(0);

    size_t scheme_len = (size_t)(scheme_end - input);
    if (port == 0) {
        if (sys_span_eq_cstr(input, scheme_len, "https")) port = 443;
        else if (sys_span_eq_cstr(input, scheme_len, "http")) port = 80;
    }

    const char* path_start = authority_end;
    const char* path_end = authority_end;
    if (*path_start == '/') {
        path_end = path_start;
        while (*path_end && *path_end != '?' && *path_end != '#') path_end++;
    }

    const char* query_start = NULL;
    const char* query_end = NULL;
    const char* fragment_start = NULL;
    for (const char* p = authority_end; *p; p++) {
        if (*p == '?' && !query_start && !fragment_start) {
            query_start = p + 1;
            query_end = input + strlen(input);
        } else if (*p == '#') {
            fragment_start = p + 1;
            if (query_start) query_end = p;
            break;
        }
    }

    eshkol_sysbuiltin_value_t result = sys_make_null();
    if (fragment_start) {
        result = sys_make_pair(sys_alist_entry("fragment", sys_make_string(fragment_start)), result);
    }
    if (query_start) {
        result = sys_make_pair(sys_alist_entry("query",
                              sys_make_string_len(query_start, (size_t)(query_end - query_start))), result);
    }
    if (*path_start == '/') {
        result = sys_make_pair(sys_alist_entry("path",
                              sys_make_string_len(path_start, (size_t)(path_end - path_start))), result);
    } else {
        result = sys_make_pair(sys_alist_entry("path", sys_make_string("/")), result);
    }
    if (port > 0) {
        result = sys_make_pair(sys_alist_entry("port", sys_make_int64(port)), result);
    }
    result = sys_make_pair(sys_alist_entry("host",
                          sys_make_string_len(host_start, (size_t)(host_end - host_start))), result);
    result = sys_make_pair(sys_alist_entry("scheme",
                          sys_make_string_len(input, scheme_len)), result);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_base64url_encode_v(eshkol_sysbuiltin_value_t data_val) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    const char* data = sys_extract_string(data_val);
    if (!data) return sys_make_bool(0);
    size_t len = strlen(data);
    size_t out_len = (len / 3) * 4;
    if (len % 3 == 1) out_len += 2;
    else if (len % 3 == 2) out_len += 3;
    char* out = (char*)malloc(out_len + 1);
    if (!out) return sys_make_bool(0);
    size_t pos = 0;
    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        unsigned int n = ((unsigned int)(unsigned char)data[i] << 16) |
                         ((unsigned int)(unsigned char)data[i + 1] << 8) |
                         (unsigned int)(unsigned char)data[i + 2];
        out[pos++] = table[(n >> 18) & 63];
        out[pos++] = table[(n >> 12) & 63];
        out[pos++] = table[(n >> 6) & 63];
        out[pos++] = table[n & 63];
    }
    if (i < len) {
        unsigned int n = (unsigned int)(unsigned char)data[i] << 16;
        if (i + 1 < len) n |= (unsigned int)(unsigned char)data[i + 1] << 8;
        out[pos++] = table[(n >> 18) & 63];
        out[pos++] = table[(n >> 12) & 63];
        if (i + 1 < len) out[pos++] = table[(n >> 6) & 63];
    }
    out[pos] = '\0';
    eshkol_sysbuiltin_value_t result = sys_make_string_len(out, pos);
    free(out);
    return result;
}

static int sys_base64url_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '-') return 62;
    if (c == '_') return 63;
    return -1;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_base64url_decode_v(eshkol_sysbuiltin_value_t data_val) {
    const char* data = sys_extract_string(data_val);
    if (!data) return sys_make_bool(0);
    size_t len = strlen(data);
    while (len > 0 && data[len - 1] == '=') len--;
    if ((len % 4) == 1) return sys_make_bool(0);
    size_t out_cap = (len * 6) / 8 + 1;
    char* out = (char*)malloc(out_cap);
    if (!out) return sys_make_bool(0);
    int acc = 0;
    int bits = 0;
    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        int v = sys_base64url_value((unsigned char)data[i]);
        if (v < 0) {
            free(out);
            return sys_make_bool(0);
        }
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[pos++] = (char)((acc >> bits) & 0xFF);
        }
    }
    out[pos] = '\0';
    eshkol_sysbuiltin_value_t result = sys_make_string_len(out, pos);
    free(out);
    return result;
}

static int sys_random_bytes(unsigned char* out, size_t len) {
    if (!out) return 0;
#ifndef _WIN32
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t pos = 0;
        while (pos < len) {
            ssize_t n = read(fd, out + pos, len - pos);
            if (n <= 0) break;
            pos += (size_t)n;
        }
        close(fd);
        if (pos == len) return 1;
    }
#endif
    srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)out);
    for (size_t i = 0; i < len; i++) out[i] = (unsigned char)(rand() & 0xFF);
    return 1;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_uuid_v4_v(void) {
    unsigned char bytes[16];
    if (!sys_random_bytes(bytes, sizeof(bytes))) return sys_make_bool(0);
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    char buf[37];
    snprintf(buf, sizeof(buf),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0], bytes[1], bytes[2], bytes[3],
             bytes[4], bytes[5], bytes[6], bytes[7],
             bytes[8], bytes[9], bytes[10], bytes[11],
             bytes[12], bytes[13], bytes[14], bytes[15]);
    return sys_make_string(buf);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_constant_time_equal_v(
    eshkol_sysbuiltin_value_t a_val,
    eshkol_sysbuiltin_value_t b_val) {
    const char* a = sys_extract_string(a_val);
    const char* b = sys_extract_string(b_val);
    if (!a || !b) return sys_make_bool(0);
    size_t a_len = strlen(a);
    size_t b_len = strlen(b);
    size_t max_len = a_len > b_len ? a_len : b_len;
    volatile unsigned char diff = (unsigned char)(a_len ^ b_len);
    for (size_t i = 0; i < max_len; i++) {
        unsigned char ac = i < a_len ? (unsigned char)a[i] : 0;
        unsigned char bc = i < b_len ? (unsigned char)b[i] : 0;
        diff |= (unsigned char)(ac ^ bc);
    }
    return sys_make_bool(diff == 0);
}

typedef struct {
    uint32_t h[8];
    uint64_t bit_len;
    unsigned char buf[64];
    size_t buf_len;
} SysSha256;

static uint32_t sys_sha256_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sys_sha256_transform(SysSha256* ctx, const unsigned char block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = sys_sha256_rotr(w[i - 15], 7) ^ sys_sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = sys_sha256_rotr(w[i - 2], 17) ^ sys_sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = ctx->h[0], b = ctx->h[1], c = ctx->h[2], d = ctx->h[3];
    uint32_t e = ctx->h[4], f = ctx->h[5], g = ctx->h[6], h = ctx->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = sys_sha256_rotr(e, 6) ^ sys_sha256_rotr(e, 11) ^ sys_sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = sys_sha256_rotr(a, 2) ^ sys_sha256_rotr(a, 13) ^ sys_sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
    ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
}

static void sys_sha256_init(SysSha256* ctx) {
    static const uint32_t init[8] = {
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U
    };
    memcpy(ctx->h, init, sizeof(init));
    ctx->bit_len = 0;
    ctx->buf_len = 0;
}

static void sys_sha256_update(SysSha256* ctx, const unsigned char* data, size_t len) {
    ctx->bit_len += (uint64_t)len * 8U;
    while (len > 0) {
        size_t n = 64 - ctx->buf_len;
        if (n > len) n = len;
        memcpy(ctx->buf + ctx->buf_len, data, n);
        ctx->buf_len += n;
        data += n;
        len -= n;
        if (ctx->buf_len == 64) {
            sys_sha256_transform(ctx, ctx->buf);
            ctx->buf_len = 0;
        }
    }
}

static void sys_sha256_final(SysSha256* ctx, unsigned char out[32]) {
    ctx->buf[ctx->buf_len++] = 0x80;
    if (ctx->buf_len > 56) {
        while (ctx->buf_len < 64) ctx->buf[ctx->buf_len++] = 0;
        sys_sha256_transform(ctx, ctx->buf);
        ctx->buf_len = 0;
    }
    while (ctx->buf_len < 56) ctx->buf[ctx->buf_len++] = 0;
    for (int i = 7; i >= 0; i--)
        ctx->buf[ctx->buf_len++] = (unsigned char)((ctx->bit_len >> (i * 8)) & 0xFF);
    sys_sha256_transform(ctx, ctx->buf);
    for (int i = 0; i < 8; i++) {
        out[i * 4] = (unsigned char)(ctx->h[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(ctx->h[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(ctx->h[i] >> 8);
        out[i * 4 + 3] = (unsigned char)ctx->h[i];
    }
}

static eshkol_sysbuiltin_value_t eshkol_builtin_sha256_file_v(eshkol_sysbuiltin_value_t path_val) {
    const char* path = sys_extract_string(path_val);
    if (!path) return sys_make_bool(0);
    FILE* f = fopen(path, "rb");
    if (!f) return sys_make_bool(0);
    SysSha256 ctx;
    sys_sha256_init(&ctx);
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        sys_sha256_update(&ctx, buf, n);
    int ok = !ferror(f);
    fclose(f);
    if (!ok) return sys_make_bool(0);
    unsigned char digest[32];
    sys_sha256_final(&ctx, digest);
    static const char hex[] = "0123456789abcdef";
    char out[65];
    for (int i = 0; i < 32; i++) {
        out[i * 2] = hex[(digest[i] >> 4) & 0x0F];
        out[i * 2 + 1] = hex[digest[i] & 0x0F];
    }
    out[64] = '\0';
    return sys_make_string(out);
}

typedef struct {
    int active;
    void* regex;
} eshkol_sys_regex_handle_t;

static eshkol_sys_regex_handle_t g_sys_regex_handles[32];

static eshkol_sysbuiltin_value_t eshkol_builtin_regex_compile_v(eshkol_sysbuiltin_value_t pattern_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    const char* pattern = sys_extract_string(pattern_val);
    if (!pattern) return sys_make_bool(0);
    int slot = -1;
    for (int i = 1; i < (int)(sizeof(g_sys_regex_handles) / sizeof(g_sys_regex_handles[0])); ++i) {
        if (!g_sys_regex_handles[i].active) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return sys_make_bool(0);
    regex_t* re = (regex_t*)calloc(1, sizeof(regex_t));
    if (!re) return sys_make_bool(0);
    if (regcomp(re, pattern, REG_EXTENDED) != 0) {
        free(re);
        return sys_make_bool(0);
    }
    g_sys_regex_handles[slot].active = 1;
    g_sys_regex_handles[slot].regex = re;
    return sys_make_int64((int64_t)slot);
#else
    (void)pattern_val;
    return sys_make_bool(0);
#endif
}

static void* sys_regex_get(eshkol_sysbuiltin_value_t handle_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int handle = (int)sys_extract_int64(handle_val);
    if (handle <= 0 || handle >= (int)(sizeof(g_sys_regex_handles) / sizeof(g_sys_regex_handles[0])) ||
        !g_sys_regex_handles[handle].active)
        return NULL;
    return g_sys_regex_handles[handle].regex;
#else
    (void)handle_val;
    return NULL;
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_regex_free_v(eshkol_sysbuiltin_value_t handle_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int handle = (int)sys_extract_int64(handle_val);
    if (handle <= 0 || handle >= (int)(sizeof(g_sys_regex_handles) / sizeof(g_sys_regex_handles[0])) ||
        !g_sys_regex_handles[handle].active)
        return sys_make_bool(0);
    regex_t* re = (regex_t*)g_sys_regex_handles[handle].regex;
    if (re) {
        regfree(re);
        free(re);
    }
    memset(&g_sys_regex_handles[handle], 0, sizeof(g_sys_regex_handles[handle]));
    return sys_make_bool(1);
#else
    (void)handle_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_regex_match_v(eshkol_sysbuiltin_value_t handle_val,
                                                               eshkol_sysbuiltin_value_t subject_val,
                                                               int boolean_only) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    regex_t* re = (regex_t*)sys_regex_get(handle_val);
    const char* subject = sys_extract_string(subject_val);
    if (!re || !subject) return sys_make_bool(0);
    regmatch_t m[1];
    int rc = regexec(re, subject, 1, m, 0);
    if (rc != 0 || m[0].rm_so < 0) return sys_make_bool(0);
    if (boolean_only) return sys_make_bool(1);
    return sys_make_string_len(subject + m[0].rm_so, (size_t)(m[0].rm_eo - m[0].rm_so));
#else
    (void)handle_val; (void)subject_val; (void)boolean_only;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_regex_match_groups_v(eshkol_sysbuiltin_value_t handle_val,
                                                                      eshkol_sysbuiltin_value_t subject_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    regex_t* re = (regex_t*)sys_regex_get(handle_val);
    const char* subject = sys_extract_string(subject_val);
    if (!re || !subject) return sys_make_bool(0);
    size_t nmatch = re->re_nsub + 1;
    if (nmatch > 32) nmatch = 32;
    regmatch_t matches[32];
    int rc = regexec(re, subject, nmatch, matches, 0);
    if (rc != 0 || matches[0].rm_so < 0) return sys_make_bool(0);

    eshkol_sysbuiltin_value_t groups = sys_make_null();
    for (size_t i = nmatch; i > 1; --i) {
        regmatch_t m = matches[i - 1];
        eshkol_sysbuiltin_value_t item = (m.rm_so >= 0)
            ? sys_make_string_len(subject + m.rm_so, (size_t)(m.rm_eo - m.rm_so))
            : sys_make_string("");
        groups = sys_make_pair(item, groups);
    }

    eshkol_sysbuiltin_value_t result = sys_make_null();
    result = sys_make_pair(sys_alist_entry("end", sys_make_int64((int64_t)matches[0].rm_eo)), result);
    result = sys_make_pair(sys_alist_entry("start", sys_make_int64((int64_t)matches[0].rm_so)), result);
    result = sys_make_pair(sys_alist_entry("groups", groups), result);
    result = sys_make_pair(sys_alist_entry("full",
                          sys_make_string_len(subject + matches[0].rm_so,
                                              (size_t)(matches[0].rm_eo - matches[0].rm_so))), result);
    return result;
#else
    (void)handle_val; (void)subject_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_regex_split_v(eshkol_sysbuiltin_value_t handle_val,
                                                               eshkol_sysbuiltin_value_t subject_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    regex_t* re = (regex_t*)sys_regex_get(handle_val);
    const char* subject = sys_extract_string(subject_val);
    if (!re || !subject) return sys_make_bool(0);
    int subject_len = (int)strlen(subject);
    int offset = 0;
    int parts = 0;
    eshkol_sysbuiltin_value_t rev = sys_make_null();
    while (offset <= subject_len && parts < 1024) {
        regmatch_t m[1];
        int rc = regexec(re, subject + offset, 1, m, 0);
        if (rc != 0 || m[0].rm_so < 0) break;
        int start = offset + (int)m[0].rm_so;
        int end = offset + (int)m[0].rm_eo;
        rev = sys_make_pair(sys_make_string_len(subject + offset, (size_t)(start - offset)), rev);
        parts++;
        if (end <= offset) break;
        offset = end;
    }
    rev = sys_make_pair(sys_make_string_len(subject + offset, (size_t)(subject_len - offset)), rev);
    eshkol_sysbuiltin_value_t out = sys_make_null();
    while (rev.type == SYS_TYPE_HEAP_PTR && rev.flags == 0x00 && rev.data != 0) {
        eshkol_sysbuiltin_value_t car;
        eshkol_sysbuiltin_value_t cdr;
        memcpy(&car, (void*)(uintptr_t)rev.data, sizeof(car));
        memcpy(&cdr, (char*)(uintptr_t)rev.data + sizeof(car), sizeof(cdr));
        out = sys_make_pair(car, out);
        rev = cdr;
    }
    return out;
#else
    (void)handle_val; (void)subject_val;
    return sys_make_bool(0);
#endif
}

typedef struct {
    const char* data;
    int len;
} SysLineSlice;

static int sys_split_lines(const char* s, SysLineSlice* out, int cap) {
    if (!s || !out || cap <= 0) return -1;
    int len = (int)strlen(s);
    int count = 0;
    int start = 0;
    for (int i = 0; i <= len; i++) {
        if (i == len || s[i] == '\n') {
            if (count >= cap) return -1;
            out[count].data = s + start;
            out[count].len = i - start;
            count++;
            start = i + 1;
        }
    }
    return count;
}

static int sys_line_equal(SysLineSlice a, SysLineSlice b) {
    return a.len == b.len && (a.len == 0 || memcmp(a.data, b.data, (size_t)a.len) == 0);
}

static eshkol_sysbuiltin_value_t sys_prefixed_line_value(char prefix, SysLineSlice line) {
    char* buf = (char*)malloc((size_t)line.len + 2);
    if (!buf) return sys_make_bool(0);
    buf[0] = prefix;
    if (line.len > 0) memcpy(buf + 1, line.data, (size_t)line.len);
    buf[line.len + 1] = '\0';
    eshkol_sysbuiltin_value_t out = sys_make_string_len(buf, (size_t)line.len + 1);
    free(buf);
    return out;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_diff_lines_v(eshkol_sysbuiltin_value_t old_val,
                                                              eshkol_sysbuiltin_value_t new_val) {
    enum { MAX_LINES = 256 };
    const char* old_s = sys_extract_string(old_val);
    const char* new_s = sys_extract_string(new_val);
    if (!old_s || !new_s) return sys_make_bool(0);
    SysLineSlice old_lines[MAX_LINES], new_lines[MAX_LINES];
    int n_old = sys_split_lines(old_s, old_lines, MAX_LINES);
    int n_new = sys_split_lines(new_s, new_lines, MAX_LINES);
    if (n_old < 0 || n_new < 0) return sys_make_bool(0);
    int cols = n_new + 1;
    int* dp = (int*)calloc((size_t)(n_old + 1) * (size_t)(n_new + 1), sizeof(int));
    if (!dp) return sys_make_bool(0);
    for (int i = 1; i <= n_old; i++) {
        for (int j = 1; j <= n_new; j++) {
            if (sys_line_equal(old_lines[i - 1], new_lines[j - 1]))
                dp[i * cols + j] = dp[(i - 1) * cols + (j - 1)] + 1;
            else {
                int a = dp[(i - 1) * cols + j];
                int b = dp[i * cols + (j - 1)];
                dp[i * cols + j] = a > b ? a : b;
            }
        }
    }
    int i = n_old, j = n_new;
    eshkol_sysbuiltin_value_t out = sys_make_null();
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && sys_line_equal(old_lines[i - 1], new_lines[j - 1])) {
            out = sys_make_pair(sys_prefixed_line_value('=', old_lines[i - 1]), out);
            i--; j--;
        } else if (j > 0 && (i == 0 || dp[i * cols + (j - 1)] >= dp[(i - 1) * cols + j])) {
            out = sys_make_pair(sys_prefixed_line_value('+', new_lines[j - 1]), out);
            j--;
        } else {
            out = sys_make_pair(sys_prefixed_line_value('-', old_lines[i - 1]), out);
            i--;
        }
    }
    free(dp);
    return out;
}

static int sys_fuzzy_score(const char* pattern, const char* candidate) {
    if (!pattern || !candidate) return -1;
    if (!*pattern) return 0;
    int score = 0, pi = 0, last = -2;
    for (int ci = 0; candidate[ci] && pattern[pi]; ci++) {
        if (tolower((unsigned char)pattern[pi]) != tolower((unsigned char)candidate[ci]))
            continue;
        score += 10;
        if (ci == last + 1) score += 5;
        if (ci == 0 || candidate[ci - 1] == '-' || candidate[ci - 1] == '_' ||
            candidate[ci - 1] == ' ' || candidate[ci - 1] == '/' ||
            (islower((unsigned char)candidate[ci - 1]) && isupper((unsigned char)candidate[ci])))
            score += 3;
        score -= ci - last - 1;
        last = ci;
        pi++;
    }
    return pattern[pi] ? -1 : score;
}

typedef struct {
    int score;
    eshkol_sysbuiltin_value_t candidate;
} SysFuzzyResult;

static int sys_fuzzy_result_cmp(const void* a, const void* b) {
    const SysFuzzyResult* ra = (const SysFuzzyResult*)a;
    const SysFuzzyResult* rb = (const SysFuzzyResult*)b;
    return rb->score - ra->score;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fuzzy_match_v(eshkol_sysbuiltin_value_t pattern_val,
                                                               eshkol_sysbuiltin_value_t candidates_val,
                                                               eshkol_sysbuiltin_value_t key_fn_val,
                                                               eshkol_sysbuiltin_value_t max_val) {
    (void)key_fn_val;
    const char* pattern = sys_extract_string(pattern_val);
    int64_t max_results = sys_extract_int64(max_val);
    if (!pattern || max_results <= 0) return sys_make_null();
    SysFuzzyResult results[1024];
    int n = 0;
    eshkol_sysbuiltin_value_t cur = candidates_val;
    while (cur.type == SYS_TYPE_HEAP_PTR && cur.flags == 0x00 && cur.data != 0 &&
           n < (int)(sizeof(results) / sizeof(results[0]))) {
        eshkol_sysbuiltin_value_t car, cdr;
        memcpy(&car, (void*)(uintptr_t)cur.data, sizeof(car));
        memcpy(&cdr, (char*)(uintptr_t)cur.data + sizeof(car), sizeof(cdr));
        const char* candidate = sys_extract_string(car);
        if (candidate) {
            int score = sys_fuzzy_score(pattern, candidate);
            if (score >= 0) {
                results[n].score = score;
                results[n].candidate = car;
                n++;
            }
        }
        cur = cdr;
    }
    qsort(results, (size_t)n, sizeof(results[0]), sys_fuzzy_result_cmp);
    if (max_results > n) max_results = n;
    eshkol_sysbuiltin_value_t out = sys_make_null();
    for (int i = (int)max_results - 1; i >= 0; i--) {
        eshkol_sysbuiltin_value_t entry = sys_make_pair(sys_make_int64((int64_t)results[i].score),
                                                        results[i].candidate);
        out = sys_make_pair(entry, out);
    }
    return out;
}

typedef struct {
    int major;
    int minor;
    int patch;
    char prerelease[128];
    char build[128];
} SysSemver;

static int sys_parse_uint_component(const char** p, int* out) {
    if (!p || !*p || !isdigit((unsigned char)**p)) return 0;
    int value = 0;
    while (isdigit((unsigned char)**p)) {
        value = value * 10 + (**p - '0');
        (*p)++;
    }
    *out = value;
    return 1;
}

static int sys_semver_parse_cstr(const char* s, SysSemver* out) {
    if (!s || !out) return 0;
    memset(out, 0, sizeof(*out));
    const char* p = s;
    if (*p == 'v' || *p == 'V') p++;
    if (!sys_parse_uint_component(&p, &out->major) || *p++ != '.') return 0;
    if (!sys_parse_uint_component(&p, &out->minor) || *p++ != '.') return 0;
    if (!sys_parse_uint_component(&p, &out->patch)) return 0;
    if (*p == '-') {
        p++;
        int n = 0;
        while (*p && *p != '+' && n < (int)sizeof(out->prerelease) - 1)
            out->prerelease[n++] = *p++;
        out->prerelease[n] = '\0';
        if (n == 0) return 0;
    }
    if (*p == '+') {
        p++;
        int n = 0;
        while (*p && n < (int)sizeof(out->build) - 1)
            out->build[n++] = *p++;
        out->build[n] = '\0';
        if (n == 0) return 0;
    }
    return *p == '\0';
}

static int sys_semver_identifier_numeric(const char* s, int len) {
    if (len <= 0) return 0;
    for (int i = 0; i < len; i++)
        if (!isdigit((unsigned char)s[i])) return 0;
    return 1;
}

static int sys_semver_compare_prerelease(const char* a, const char* b) {
    if (!a[0] && !b[0]) return 0;
    if (!a[0]) return 1;
    if (!b[0]) return -1;
    const char* pa = a;
    const char* pb = b;
    while (*pa || *pb) {
        const char* ea = strchr(pa, '.');
        const char* eb = strchr(pb, '.');
        int la = ea ? (int)(ea - pa) : (int)strlen(pa);
        int lb = eb ? (int)(eb - pb) : (int)strlen(pb);
        int na = sys_semver_identifier_numeric(pa, la);
        int nb = sys_semver_identifier_numeric(pb, lb);
        if (na && nb) {
            long va = 0, vb = 0;
            for (int i = 0; i < la; i++) va = va * 10 + (pa[i] - '0');
            for (int i = 0; i < lb; i++) vb = vb * 10 + (pb[i] - '0');
            if (va != vb) return va < vb ? -1 : 1;
        } else if (na != nb) {
            return na ? -1 : 1;
        } else {
            int cmp = strncmp(pa, pb, (size_t)(la < lb ? la : lb));
            if (cmp != 0) return cmp < 0 ? -1 : 1;
            if (la != lb) return la < lb ? -1 : 1;
        }
        pa = ea ? ea + 1 : pa + la;
        pb = eb ? eb + 1 : pb + lb;
    }
    return 0;
}

static int sys_semver_compare_structs(const SysSemver* a, const SysSemver* b) {
    if (a->major != b->major) return a->major < b->major ? -1 : 1;
    if (a->minor != b->minor) return a->minor < b->minor ? -1 : 1;
    if (a->patch != b->patch) return a->patch < b->patch ? -1 : 1;
    return sys_semver_compare_prerelease(a->prerelease, b->prerelease);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_semver_parse_v(eshkol_sysbuiltin_value_t str_val) {
    SysSemver v;
    const char* s = sys_extract_string(str_val);
    if (!s || !sys_semver_parse_cstr(s, &v)) return sys_make_bool(0);
    eshkol_sysbuiltin_value_t result = sys_make_null();
    result = sys_make_pair(sys_alist_entry("build", sys_make_string(v.build)), result);
    result = sys_make_pair(sys_alist_entry("prerelease", sys_make_string(v.prerelease)), result);
    result = sys_make_pair(sys_alist_entry("patch", sys_make_int64((int64_t)v.patch)), result);
    result = sys_make_pair(sys_alist_entry("minor", sys_make_int64((int64_t)v.minor)), result);
    result = sys_make_pair(sys_alist_entry("major", sys_make_int64((int64_t)v.major)), result);
    return result;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_semver_compare_v(eshkol_sysbuiltin_value_t a_val,
                                                                  eshkol_sysbuiltin_value_t b_val) {
    SysSemver a, b;
    const char* as = sys_extract_string(a_val);
    const char* bs = sys_extract_string(b_val);
    if (!as || !bs || !sys_semver_parse_cstr(as, &a) || !sys_semver_parse_cstr(bs, &b))
        return sys_make_bool(0);
    int cmp = sys_semver_compare_structs(&a, &b);
    return sys_make_int64((int64_t)(cmp < 0 ? -1 : cmp > 0 ? 1 : 0));
}

static int sys_semver_satisfies_range(const SysSemver* version, const char* range) {
    while (*range && isspace((unsigned char)*range)) range++;
    char op[3] = "";
    if ((range[0] == '>' || range[0] == '<' || range[0] == '=') && range[1] == '=') {
        op[0] = range[0]; op[1] = '='; range += 2;
    } else if (*range == '>' || *range == '<' || *range == '=') {
        op[0] = *range++;
    } else if (*range == '^' || *range == '~') {
        op[0] = *range++;
    }
    while (*range && isspace((unsigned char)*range)) range++;
    SysSemver target;
    if (!sys_semver_parse_cstr(range, &target)) return 0;
    int cmp = sys_semver_compare_structs(version, &target);
    if (!op[0] || op[0] == '=') return cmp == 0;
    if (strcmp(op, ">=") == 0) return cmp >= 0;
    if (strcmp(op, "<=") == 0) return cmp <= 0;
    if (strcmp(op, ">") == 0) return cmp > 0;
    if (strcmp(op, "<") == 0) return cmp < 0;
    if (op[0] == '^') {
        SysSemver upper = target;
        upper.major++; upper.minor = 0; upper.patch = 0; upper.prerelease[0] = '\0';
        return cmp >= 0 && sys_semver_compare_structs(version, &upper) < 0;
    }
    if (op[0] == '~') {
        SysSemver upper = target;
        upper.minor++; upper.patch = 0; upper.prerelease[0] = '\0';
        return cmp >= 0 && sys_semver_compare_structs(version, &upper) < 0;
    }
    return 0;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_semver_satisfies_v(eshkol_sysbuiltin_value_t version_val,
                                                                    eshkol_sysbuiltin_value_t range_val) {
    SysSemver version;
    const char* vs = sys_extract_string(version_val);
    const char* range = sys_extract_string(range_val);
    if (!vs || !range || !sys_semver_parse_cstr(vs, &version)) return sys_make_bool(0);
    return sys_make_bool(sys_semver_satisfies_range(&version, range));
}

typedef struct {
    int active;
    int fd;
    int len;
    char buffer[4096];
} SysLineReader;

static SysLineReader g_sys_line_readers[32];

static eshkol_sysbuiltin_value_t eshkol_builtin_make_pipe_v(void) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int fds[2];
    if (pipe(fds) != 0) return sys_make_bool(0);
    return sys_make_pair(sys_make_int64((int64_t)fds[0]), sys_make_int64((int64_t)fds[1]));
#else
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fd_write_v(eshkol_sysbuiltin_value_t fd_val,
                                                            eshkol_sysbuiltin_value_t data_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int fd = (int)sys_extract_int64(fd_val);
    const char* data = sys_extract_string(data_val);
    if (fd < 0 || !data) return sys_make_bool(0);
    ssize_t n = write(fd, data, strlen(data));
    return n >= 0 ? sys_make_int64((int64_t)n) : sys_make_bool(0);
#else
    (void)fd_val; (void)data_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_make_line_reader_v(eshkol_sysbuiltin_value_t fd_val,
                                                                    eshkol_sysbuiltin_value_t callback_val) {
    (void)callback_val;
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int fd = (int)sys_extract_int64(fd_val);
    if (fd < 0) return sys_make_bool(0);
    for (int i = 1; i < (int)(sizeof(g_sys_line_readers) / sizeof(g_sys_line_readers[0])); ++i) {
        if (g_sys_line_readers[i].active) continue;
        g_sys_line_readers[i].active = 1;
        g_sys_line_readers[i].fd = fd;
        g_sys_line_readers[i].len = 0;
        return sys_make_int64((int64_t)i);
    }
    return sys_make_bool(0);
#else
    (void)fd_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_line_reader_poll_v(eshkol_sysbuiltin_value_t handle_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int handle = (int)sys_extract_int64(handle_val);
    if (handle <= 0 || handle >= (int)(sizeof(g_sys_line_readers) / sizeof(g_sys_line_readers[0])) ||
        !g_sys_line_readers[handle].active)
        return sys_make_bool(0);
    int len = g_sys_line_readers[handle].len;
    for (int i = 0; i < len; ++i) {
        if (g_sys_line_readers[handle].buffer[i] == '\n') {
            eshkol_sysbuiltin_value_t line = sys_make_string_len(g_sys_line_readers[handle].buffer, (size_t)i);
            int remaining = len - i - 1;
            if (remaining > 0)
                memmove(g_sys_line_readers[handle].buffer,
                        g_sys_line_readers[handle].buffer + i + 1,
                        (size_t)remaining);
            g_sys_line_readers[handle].len = remaining;
            return line;
        }
    }

    int fd = g_sys_line_readers[handle].fd;
    int old_flags = fcntl(fd, F_GETFL, 0);
    if (old_flags < 0) return sys_make_bool(0);
    if (fcntl(fd, F_SETFL, old_flags | O_NONBLOCK) != 0) return sys_make_bool(0);
    char chunk[512];
    ssize_t n = read(fd, chunk, sizeof(chunk));
    (void)fcntl(fd, F_SETFL, old_flags);
    if (n <= 0) return sys_make_bool(0);
    int space = (int)sizeof(g_sys_line_readers[handle].buffer) - len;
    if (n > space) n = space;
    if (n > 0) {
        memcpy(g_sys_line_readers[handle].buffer + len, chunk, (size_t)n);
        len += (int)n;
        g_sys_line_readers[handle].len = len;
    }
    for (int i = 0; i < len; ++i) {
        if (g_sys_line_readers[handle].buffer[i] == '\n') {
            eshkol_sysbuiltin_value_t line = sys_make_string_len(g_sys_line_readers[handle].buffer, (size_t)i);
            int remaining = len - i - 1;
            if (remaining > 0)
                memmove(g_sys_line_readers[handle].buffer,
                        g_sys_line_readers[handle].buffer + i + 1,
                        (size_t)remaining);
            g_sys_line_readers[handle].len = remaining;
            return line;
        }
    }
    return sys_make_bool(0);
#else
    (void)handle_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_line_reader_close_v(eshkol_sysbuiltin_value_t handle_val) {
    int handle = (int)sys_extract_int64(handle_val);
    if (handle > 0 && handle < (int)(sizeof(g_sys_line_readers) / sizeof(g_sys_line_readers[0])) &&
        g_sys_line_readers[handle].active) {
        memset(&g_sys_line_readers[handle], 0, sizeof(g_sys_line_readers[handle]));
        return sys_make_bool(1);
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_fd_close_v(eshkol_sysbuiltin_value_t fd_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int fd = (int)sys_extract_int64(fd_val);
    return (fd >= 0 && close(fd) == 0) ? sys_make_bool(1) : sys_make_bool(0);
#else
    (void)fd_val;
    return sys_make_bool(0);
#endif
}

typedef struct {
    int active;
    int64_t tick;
    eshkol_sysbuiltin_value_t key;
    eshkol_sysbuiltin_value_t value;
} SysLruEntry;

typedef struct {
    int active;
    int max_size;
    int size;
    int64_t tick;
    SysLruEntry entries[32];
} SysLruCache;

static SysLruCache g_sys_lru_caches[16];

static int sys_values_equal_simple(eshkol_sysbuiltin_value_t a, eshkol_sysbuiltin_value_t b) {
    const char* as = sys_extract_string(a);
    const char* bs = sys_extract_string(b);
    if (as || bs) return as && bs && strcmp(as, bs) == 0;
    return a.type == b.type && a.flags == b.flags && a.data == b.data;
}

static int sys_lru_valid(int handle) {
    return handle > 0 && handle < (int)(sizeof(g_sys_lru_caches) / sizeof(g_sys_lru_caches[0])) &&
           g_sys_lru_caches[handle].active;
}

static int sys_lru_find(int handle, eshkol_sysbuiltin_value_t key) {
    if (!sys_lru_valid(handle)) return -1;
    for (int i = 0; i < g_sys_lru_caches[handle].max_size; ++i) {
        if (g_sys_lru_caches[handle].entries[i].active &&
            sys_values_equal_simple(g_sys_lru_caches[handle].entries[i].key, key))
            return i;
    }
    return -1;
}

static int sys_lru_alloc_entry(int handle) {
    if (!sys_lru_valid(handle)) return -1;
    for (int i = 0; i < g_sys_lru_caches[handle].max_size; ++i)
        if (!g_sys_lru_caches[handle].entries[i].active)
            return i;
    int evict = 0;
    int64_t oldest = g_sys_lru_caches[handle].entries[0].tick;
    for (int i = 1; i < g_sys_lru_caches[handle].max_size; ++i) {
        if (g_sys_lru_caches[handle].entries[i].tick < oldest) {
            oldest = g_sys_lru_caches[handle].entries[i].tick;
            evict = i;
        }
    }
    return evict;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_make_lru_cache_v(eshkol_sysbuiltin_value_t max_val) {
    int max_size = (int)sys_extract_int64(max_val);
    if (max_size <= 0) return sys_make_bool(0);
    if (max_size > 32) max_size = 32;
    for (int i = 1; i < (int)(sizeof(g_sys_lru_caches) / sizeof(g_sys_lru_caches[0])); ++i) {
        if (g_sys_lru_caches[i].active) continue;
        memset(&g_sys_lru_caches[i], 0, sizeof(g_sys_lru_caches[i]));
        g_sys_lru_caches[i].active = 1;
        g_sys_lru_caches[i].max_size = max_size;
        g_sys_lru_caches[i].tick = 1;
        return sys_make_int64((int64_t)i);
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_lru_get_v(eshkol_sysbuiltin_value_t cache_val,
                                                           eshkol_sysbuiltin_value_t key_val) {
    int handle = (int)sys_extract_int64(cache_val);
    int idx = sys_lru_find(handle, key_val);
    if (idx < 0) return sys_make_bool(0);
    g_sys_lru_caches[handle].entries[idx].tick = ++g_sys_lru_caches[handle].tick;
    return g_sys_lru_caches[handle].entries[idx].value;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_lru_set_v(eshkol_sysbuiltin_value_t cache_val,
                                                           eshkol_sysbuiltin_value_t key_val,
                                                           eshkol_sysbuiltin_value_t value_val) {
    int handle = (int)sys_extract_int64(cache_val);
    if (!sys_lru_valid(handle)) return sys_make_bool(0);
    int idx = sys_lru_find(handle, key_val);
    if (idx < 0) idx = sys_lru_alloc_entry(handle);
    if (idx < 0) return sys_make_bool(0);
    if (!g_sys_lru_caches[handle].entries[idx].active)
        g_sys_lru_caches[handle].size++;
    g_sys_lru_caches[handle].entries[idx].active = 1;
    g_sys_lru_caches[handle].entries[idx].tick = ++g_sys_lru_caches[handle].tick;
    g_sys_lru_caches[handle].entries[idx].key = key_val;
    g_sys_lru_caches[handle].entries[idx].value = value_val;
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_lru_has_v(eshkol_sysbuiltin_value_t cache_val,
                                                           eshkol_sysbuiltin_value_t key_val) {
    return sys_make_bool(sys_lru_find((int)sys_extract_int64(cache_val), key_val) >= 0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_lru_delete_v(eshkol_sysbuiltin_value_t cache_val,
                                                              eshkol_sysbuiltin_value_t key_val) {
    int handle = (int)sys_extract_int64(cache_val);
    int idx = sys_lru_find(handle, key_val);
    if (idx < 0) return sys_make_bool(0);
    memset(&g_sys_lru_caches[handle].entries[idx], 0, sizeof(g_sys_lru_caches[handle].entries[idx]));
    if (g_sys_lru_caches[handle].size > 0) g_sys_lru_caches[handle].size--;
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_lru_clear_v(eshkol_sysbuiltin_value_t cache_val) {
    int handle = (int)sys_extract_int64(cache_val);
    if (!sys_lru_valid(handle)) return sys_make_bool(0);
    memset(g_sys_lru_caches[handle].entries, 0, sizeof(g_sys_lru_caches[handle].entries));
    g_sys_lru_caches[handle].size = 0;
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_lru_size_v(eshkol_sysbuiltin_value_t cache_val) {
    int handle = (int)sys_extract_int64(cache_val);
    return sys_lru_valid(handle) ? sys_make_int64((int64_t)g_sys_lru_caches[handle].size)
                                 : sys_make_int64(0);
}

static int sys_format_append(char* out, size_t cap, size_t* pos, const char* s, size_t len) {
    if (!out || !pos || !s) return 0;
    if (*pos + len >= cap) return 0;
    memcpy(out + *pos, s, len);
    *pos += len;
    out[*pos] = '\0';
    return 1;
}

static int sys_format_append_cstr(char* out, size_t cap, size_t* pos, const char* s) {
    return sys_format_append(out, cap, pos, s ? s : "", s ? strlen(s) : 0);
}

static double sys_extract_double_display(eshkol_sysbuiltin_value_t v) {
    if (v.type == SYS_TYPE_DOUBLE) {
        double d = 0.0;
        memcpy(&d, &v.data, sizeof(double));
        return d;
    }
    return (double)sys_extract_int64(v);
}

static int sys_format_append_value(char* out,
                                   size_t cap,
                                   size_t* pos,
                                   eshkol_sysbuiltin_value_t value,
                                   char directive) {
    char buf[128];
    switch (directive) {
    case 'd':
        snprintf(buf, sizeof(buf), "%lld", (long long)sys_extract_int64(value));
        return sys_format_append_cstr(out, cap, pos, buf);
    case 'x':
        snprintf(buf, sizeof(buf), "%llx", (unsigned long long)sys_extract_int64(value));
        return sys_format_append_cstr(out, cap, pos, buf);
    case 'f':
        snprintf(buf, sizeof(buf), "%.6g", sys_extract_double_display(value));
        return sys_format_append_cstr(out, cap, pos, buf);
    case 's': {
        const char* s = sys_extract_string(value);
        if (s) {
            return sys_format_append_cstr(out, cap, pos, "\"") &&
                   sys_format_append_cstr(out, cap, pos, s) &&
                   sys_format_append_cstr(out, cap, pos, "\"");
        }
        /* fall through for non-strings */
    }
    case 'a':
    default: {
        const char* s = sys_extract_string(value);
        if (s) return sys_format_append_cstr(out, cap, pos, s);
        if (value.type == SYS_TYPE_INT64) {
            snprintf(buf, sizeof(buf), "%lld", (long long)sys_extract_int64(value));
            return sys_format_append_cstr(out, cap, pos, buf);
        }
        if (value.type == SYS_TYPE_DOUBLE) {
            snprintf(buf, sizeof(buf), "%.6g", sys_extract_double_display(value));
            return sys_format_append_cstr(out, cap, pos, buf);
        }
        if (value.type == SYS_TYPE_BOOL)
            return sys_format_append_cstr(out, cap, pos, value.data ? "#t" : "#f");
        if (value.type == SYS_TYPE_NULL)
            return sys_format_append_cstr(out, cap, pos, "()");
        snprintf(buf, sizeof(buf), "#<value:%u>", (unsigned)value.type);
        return sys_format_append_cstr(out, cap, pos, buf);
    }
    }
}

static int sys_format_next_arg(eshkol_sysbuiltin_value_t* args,
                               eshkol_sysbuiltin_value_t* out) {
    if (!args || !out || args->type != SYS_TYPE_HEAP_PTR || args->flags != 0x00 ||
        args->data == 0)
        return 0;
    memcpy(out, (void*)(uintptr_t)args->data, sizeof(*out));
    memcpy(args, (char*)(uintptr_t)args->data + sizeof(*out), sizeof(*args));
    return 1;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_format_list_v(
    eshkol_sysbuiltin_value_t fmt_val,
    eshkol_sysbuiltin_value_t args_val) {
    const char* fmt = sys_extract_string(fmt_val);
    if (!fmt) return sys_make_bool(0);
    size_t fmt_len = strlen(fmt);
    char out[16384];
    size_t pos = 0;
    out[0] = '\0';

    for (size_t i = 0; i < fmt_len; i++) {
        char ch = fmt[i];
        if (ch != '~') {
            if (!sys_format_append(out, sizeof(out), &pos, &ch, 1)) return sys_make_bool(0);
            continue;
        }
        if (++i >= fmt_len) {
            if (!sys_format_append_cstr(out, sizeof(out), &pos, "~")) return sys_make_bool(0);
            break;
        }
        char directive = fmt[i];
        if (directive == '~') {
            if (!sys_format_append_cstr(out, sizeof(out), &pos, "~")) return sys_make_bool(0);
        } else if (directive == '%') {
            if (!sys_format_append_cstr(out, sizeof(out), &pos, "\n")) return sys_make_bool(0);
        } else if (directive == 'a' || directive == 's' || directive == 'd' ||
                   directive == 'x' || directive == 'f') {
            eshkol_sysbuiltin_value_t arg;
            if (!sys_format_next_arg(&args_val, &arg)) return sys_make_bool(0);
            if (!sys_format_append_value(out, sizeof(out), &pos, arg, directive))
                return sys_make_bool(0);
        } else {
            if (!sys_format_append_cstr(out, sizeof(out), &pos, "~") ||
                !sys_format_append(out, sizeof(out), &pos, &directive, 1))
                return sys_make_bool(0);
        }
    }

    return sys_make_string_len(out, pos);
}

typedef struct {
    int active;
    int listen_fd;
    int client_fd;
    int port;
} SysHttpServer;

static SysHttpServer g_sys_http_servers[8];

static int sys_http_server_valid(int handle) {
    return handle > 0 &&
           handle < (int)(sizeof(g_sys_http_servers) / sizeof(g_sys_http_servers[0])) &&
           g_sys_http_servers[handle].active;
}

static int sys_http_server_store(int listen_fd, int port) {
    for (int i = 1; i < (int)(sizeof(g_sys_http_servers) / sizeof(g_sys_http_servers[0])); i++) {
        if (!g_sys_http_servers[i].active) {
            g_sys_http_servers[i].active = 1;
            g_sys_http_servers[i].listen_fd = listen_fd;
            g_sys_http_servers[i].client_fd = -1;
            g_sys_http_servers[i].port = port;
            return i;
        }
    }
    return -1;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_http_server_create_v(eshkol_sysbuiltin_value_t port_val) {
#if !defined(_WIN32)
    int requested_port = (int)sys_extract_int64(port_val);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) {
        int opt = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, (socklen_t)sizeof(opt));
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons((uint16_t)requested_port);
        if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0 &&
            listen(fd, 8) == 0) {
            socklen_t addr_len = (socklen_t)sizeof(addr);
            if (getsockname(fd, (struct sockaddr*)&addr, &addr_len) == 0) {
                int handle = sys_http_server_store(fd, (int)ntohs(addr.sin_port));
                if (handle > 0) return sys_make_int64((int64_t)handle);
            }
        }
        close(fd);
    }
#else
    (void)port_val;
#endif
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_http_server_port_v(eshkol_sysbuiltin_value_t server_val) {
    int handle = (int)sys_extract_int64(server_val);
    return sys_http_server_valid(handle) ? sys_make_int64((int64_t)g_sys_http_servers[handle].port)
                                         : sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_http_server_accept_v(
    eshkol_sysbuiltin_value_t server_val,
    eshkol_sysbuiltin_value_t buffer_val,
    eshkol_sysbuiltin_value_t timeout_val) {
#if !defined(_WIN32)
    int handle = (int)sys_extract_int64(server_val);
    int buffer_size = (int)sys_extract_int64(buffer_val);
    int timeout_ms = (int)sys_extract_int64(timeout_val);
    if (buffer_size < 128) buffer_size = 128;
    if (buffer_size > 65536) buffer_size = 65536;
    if (timeout_ms < 0) timeout_ms = 0;
    if (!sys_http_server_valid(handle)) return sys_make_bool(0);

    struct pollfd pfd;
    pfd.fd = g_sys_http_servers[handle].listen_fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    if (poll(&pfd, 1, timeout_ms) <= 0 || !(pfd.revents & POLLIN))
        return sys_make_bool(0);

    int client_fd = accept(g_sys_http_servers[handle].listen_fd, NULL, NULL);
    if (client_fd < 0) return sys_make_bool(0);
    if (g_sys_http_servers[handle].client_fd >= 0)
        close(g_sys_http_servers[handle].client_fd);
    g_sys_http_servers[handle].client_fd = client_fd;

    char* buf = (char*)malloc((size_t)buffer_size + 1);
    if (!buf) return sys_make_bool(0);
    int total = 0;
    while (total < buffer_size) {
        ssize_t n = recv(client_fd, buf + total, (size_t)(buffer_size - total), 0);
        if (n <= 0) break;
        total += (int)n;
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n") || strstr(buf, "\n\n")) break;
    }
    eshkol_sysbuiltin_value_t result = total > 0 ? sys_make_string_len(buf, (size_t)total)
                                                  : sys_make_bool(0);
    free(buf);
    return result;
#else
    (void)server_val; (void)buffer_val; (void)timeout_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_http_server_respond_v(
    eshkol_sysbuiltin_value_t server_val,
    eshkol_sysbuiltin_value_t status_val,
    eshkol_sysbuiltin_value_t type_val,
    eshkol_sysbuiltin_value_t body_val) {
#if !defined(_WIN32)
    int handle = (int)sys_extract_int64(server_val);
    int status = (int)sys_extract_int64(status_val);
    const char* content_type = sys_extract_string(type_val);
    const char* body = sys_extract_string(body_val);
    if (!sys_http_server_valid(handle) || g_sys_http_servers[handle].client_fd < 0 || !body)
        return sys_make_bool(0);
    if (status <= 0) status = 200;
    const char* reason = status == 404 ? "Not Found" :
                         status == 500 ? "Internal Server Error" :
                         status == 400 ? "Bad Request" : "OK";
    const char* ctype = (content_type && *content_type) ? content_type : "text/plain";
    size_t body_len = strlen(body);
    char header[512];
    int hlen = snprintf(header, sizeof(header),
                        "HTTP/1.1 %d %s\r\n"
                        "Content-Type: %s\r\n"
                        "Content-Length: %lld\r\n"
                        "Connection: close\r\n\r\n",
                        status, reason, ctype, (long long)body_len);
    int ok = hlen > 0 && hlen < (int)sizeof(header) &&
             send(g_sys_http_servers[handle].client_fd, header, (size_t)hlen, 0) == hlen &&
             send(g_sys_http_servers[handle].client_fd, body, body_len, 0) == (ssize_t)body_len;
    close(g_sys_http_servers[handle].client_fd);
    g_sys_http_servers[handle].client_fd = -1;
    return sys_make_bool(ok);
#else
    (void)server_val; (void)status_val; (void)type_val; (void)body_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_http_server_close_v(eshkol_sysbuiltin_value_t server_val) {
    int handle = (int)sys_extract_int64(server_val);
    if (!sys_http_server_valid(handle)) return sys_make_bool(0);
#if !defined(_WIN32)
    if (g_sys_http_servers[handle].client_fd >= 0) close(g_sys_http_servers[handle].client_fd);
    if (g_sys_http_servers[handle].listen_fd >= 0) close(g_sys_http_servers[handle].listen_fd);
#endif
    memset(&g_sys_http_servers[handle], 0, sizeof(g_sys_http_servers[handle]));
    return sys_make_bool(1);
}

static int sys_http_request_parse_url(const char* url, char* host, size_t host_cap,
                                      int* port, char* path, size_t path_cap) {
    if (!url || !host || !port || !path || host_cap == 0 || path_cap == 0)
        return 0;
    const char* prefix = "http://";
    size_t prefix_len = strlen(prefix);
    if (strncmp(url, prefix, prefix_len) != 0) return 0;
    const char* host_start = url + prefix_len;
    const char* path_start = host_start;
    while (*path_start && *path_start != '/') path_start++;
    const char* colon = NULL;
    for (const char* p = host_start; p < path_start; p++) {
        if (*p == ':') colon = p;
    }
    size_t host_len = (size_t)((colon ? colon : path_start) - host_start);
    if (host_len == 0 || host_len >= host_cap) return 0;
    memcpy(host, host_start, host_len);
    host[host_len] = '\0';
    *port = 80;
    if (colon) {
        int parsed_port = atoi(colon + 1);
        if (parsed_port <= 0 || parsed_port > 65535) return 0;
        *port = parsed_port;
    }
    if (*path_start) {
        size_t path_len = strlen(path_start);
        if (path_len >= path_cap) return 0;
        memcpy(path, path_start, path_len + 1);
    } else {
        snprintf(path, path_cap, "/");
    }
    return 1;
}

static int sys_http_request_connect(const char* host, int port, int timeout_ms) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    char port_buf[16];
    snprintf(port_buf, sizeof(port_buf), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = NULL;
    if (getaddrinfo(host, port_buf, &hints, &res) != 0) return -1;
    int fd = -1;
    for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv;
        tv.tv_sec = timeout_ms > 0 ? timeout_ms / 1000 : 30;
        tv.tv_usec = timeout_ms > 0 ? (timeout_ms % 1000) * 1000 : 0;
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, (socklen_t)sizeof(tv));
        (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, (socklen_t)sizeof(tv));
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
#else
    (void)host; (void)port; (void)timeout_ms;
    return -1;
#endif
}

static eshkol_sysbuiltin_value_t sys_http_response_value(int status,
                                                         const char* headers,
                                                         size_t headers_len,
                                                         const char* body,
                                                         size_t body_len) {
    eshkol_sysbuiltin_value_t body_value = sys_make_string_len(body ? body : "", body_len);
    eshkol_sysbuiltin_value_t headers_value = sys_make_string_len(headers ? headers : "", headers_len);
    if (body_value.type != SYS_TYPE_HEAP_PTR || headers_value.type != SYS_TYPE_HEAP_PTR)
        return sys_make_bool(0);
    return sys_make_pair(sys_make_int64((int64_t)status),
                         sys_make_pair(headers_value,
                                       sys_make_pair(body_value, sys_make_null())));
}

static eshkol_sysbuiltin_value_t eshkol_builtin_http_request_v(
    eshkol_sysbuiltin_value_t method_val,
    eshkol_sysbuiltin_value_t url_val,
    eshkol_sysbuiltin_value_t headers_val,
    eshkol_sysbuiltin_value_t body_val,
    eshkol_sysbuiltin_value_t timeout_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    const char* method = sys_extract_string(method_val);
    const char* url = sys_extract_string(url_val);
    const char* headers = sys_extract_string(headers_val);
    const char* body = sys_extract_string(body_val);
    char host[256];
    char path[2048];
    int port = 80;
    int timeout_ms = (int)sys_extract_int64(timeout_val);
    if (timeout_ms <= 0) timeout_ms = 30000;
    if (!method || !*method ||
        !sys_http_request_parse_url(url, host, sizeof(host), &port, path, sizeof(path)))
        return sys_make_bool(0);

    int fd = sys_http_request_connect(host, port, timeout_ms);
    if (fd < 0) return sys_make_bool(0);

    const char* body_data = body ? body : "";
    size_t body_len = body ? strlen(body) : 0;
    const char* extra_headers = headers ? headers : "";
    char header[8192];
    int hlen = snprintf(header, sizeof(header),
                        "%s %s HTTP/1.1\r\n"
                        "Host: %s\r\n"
                        "Connection: close\r\n"
                        "%s%s"
                        "Content-Length: %lld\r\n\r\n",
                        method, path, host,
                        extra_headers,
                        extra_headers[0] ? (strstr(extra_headers, "\n") ? "" : "\r\n") : "",
                        (long long)body_len);
    int ok = hlen > 0 && hlen < (int)sizeof(header);
    if (ok) ok = send(fd, header, (size_t)hlen, 0) == hlen;
    if (ok && body_len > 0)
        ok = send(fd, body_data, body_len, 0) == (ssize_t)body_len;

    if (ok) {
        int cap = 65536;
        char* response = (char*)malloc((size_t)cap + 1);
        if (response) {
            int total = 0;
            while (total < cap) {
                ssize_t n = recv(fd, response + total, (size_t)(cap - total), 0);
                if (n <= 0) break;
                total += (int)n;
            }
            response[total] = '\0';
            close(fd);
            int status = 0;
            (void)sscanf(response, "HTTP/%*s %d", &status);
            char* body_start = strstr(response, "\r\n\r\n");
            int header_len = total;
            int body_offset = total;
            if (body_start) {
                body_start += 4;
                body_offset = (int)(body_start - response);
                header_len = body_offset - 4;
            } else {
                body_start = response + total;
            }
            eshkol_sysbuiltin_value_t result =
                sys_http_response_value(status, response, (size_t)header_len,
                                        body_start, (size_t)(total - body_offset));
            free(response);
            return result;
        }
    }
    close(fd);
#else
    (void)method_val; (void)url_val; (void)headers_val; (void)body_val; (void)timeout_val;
#endif
    return sys_make_bool(0);
}

typedef struct {
    int active;
    int fd;
    int closed;
} SysWebSocketClient;

static SysWebSocketClient g_sys_websocket_clients[16];

static int sys_ws_parse_url(const char* url, char* host, size_t host_cap,
                            int* port, char* path, size_t path_cap) {
    if (!url) return 0;
    const char* prefix = "ws://";
    size_t prefix_len = strlen(prefix);
    if (strncmp(url, prefix, prefix_len) != 0) return 0;
    char http_url[4096];
    int n = snprintf(http_url, sizeof(http_url), "http://%s", url + prefix_len);
    if (n <= 0 || n >= (int)sizeof(http_url)) return 0;
    return sys_http_request_parse_url(http_url, host, host_cap, port, path, path_cap);
}

static int sys_ws_valid(int handle) {
    return handle > 0 &&
           handle < (int)(sizeof(g_sys_websocket_clients) / sizeof(g_sys_websocket_clients[0])) &&
           g_sys_websocket_clients[handle].active &&
           !g_sys_websocket_clients[handle].closed;
}

static int sys_ws_store(int fd) {
    for (int i = 1; i < (int)(sizeof(g_sys_websocket_clients) / sizeof(g_sys_websocket_clients[0])); i++) {
        if (!g_sys_websocket_clients[i].active) {
            g_sys_websocket_clients[i].active = 1;
            g_sys_websocket_clients[i].fd = fd;
            g_sys_websocket_clients[i].closed = 0;
            return i;
        }
    }
    return -1;
}

static int sys_ws_send_all(int fd, const void* data, size_t len) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    const char* p = (const char*)data;
    size_t sent_total = 0;
    while (sent_total < len) {
        ssize_t n = send(fd, p + sent_total, len - sent_total, 0);
        if (n <= 0) return 0;
        sent_total += (size_t)n;
    }
    return 1;
#else
    (void)fd; (void)data; (void)len;
    return 0;
#endif
}

static int sys_ws_send_frame(int fd, int opcode, const void* data, size_t len) {
    unsigned char header[14];
    int hlen = 0;
    header[hlen++] = (unsigned char)(0x80 | (opcode & 0x0f));
    if (len < 126) {
        header[hlen++] = (unsigned char)(0x80 | len);
    } else if (len <= 65535) {
        header[hlen++] = 0x80 | 126;
        header[hlen++] = (unsigned char)((len >> 8) & 0xff);
        header[hlen++] = (unsigned char)(len & 0xff);
    } else {
        header[hlen++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--)
            header[hlen++] = (unsigned char)((len >> (i * 8)) & 0xff);
    }
    unsigned char mask[4] = {0, 0, 0, 0};
    memcpy(header + hlen, mask, sizeof(mask));
    hlen += 4;
    return sys_ws_send_all(fd, header, (size_t)hlen) &&
           (len == 0 || sys_ws_send_all(fd, data, len));
}

static eshkol_sysbuiltin_value_t eshkol_builtin_websocket_connect_v(
    eshkol_sysbuiltin_value_t url_val,
    eshkol_sysbuiltin_value_t headers_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    const char* url = sys_extract_string(url_val);
    const char* headers = sys_extract_string(headers_val);
    char host[256];
    char path[2048];
    int port = 80;
    if (!sys_ws_parse_url(url, host, sizeof(host), &port, path, sizeof(path)))
        return sys_make_bool(0);
    int fd = sys_http_request_connect(host, port, 5000);
    if (fd < 0) return sys_make_bool(0);

    const char* extra_headers = headers ? headers : "";
    char request[4096];
    int n = snprintf(request, sizeof(request),
                     "GET %s HTTP/1.1\r\n"
                     "Host: %s\r\n"
                     "Upgrade: websocket\r\n"
                     "Connection: Upgrade\r\n"
                     "Sec-WebSocket-Version: 13\r\n"
                     "Sec-WebSocket-Key: ZXNoa29sLXZtLXdlYnNvY2tldA==\r\n"
                     "%s%s"
                     "\r\n",
                     path, host, extra_headers,
                     extra_headers[0] ? (strstr(extra_headers, "\n") ? "" : "\r\n") : "");
    if (n > 0 && n < (int)sizeof(request) && sys_ws_send_all(fd, request, (size_t)n)) {
        char response[4096];
        ssize_t r = recv(fd, response, sizeof(response) - 1, 0);
        if (r > 0) {
            response[r] = '\0';
            if (strstr(response, " 101 ") || strstr(response, " 101\r") ||
                strstr(response, " 101\n")) {
                int handle = sys_ws_store(fd);
                if (handle > 0) return sys_make_int64((int64_t)handle);
            }
        }
    }
    close(fd);
#else
    (void)url_val; (void)headers_val;
#endif
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_websocket_send_v(
    eshkol_sysbuiltin_value_t ws_val,
    eshkol_sysbuiltin_value_t data_val) {
    int handle = (int)sys_extract_int64(ws_val);
    const char* data = sys_extract_string(data_val);
    if (!sys_ws_valid(handle) || !data) return sys_make_bool(0);
    return sys_make_bool(sys_ws_send_frame(g_sys_websocket_clients[handle].fd, 1,
                                           data, strlen(data)));
}

static eshkol_sysbuiltin_value_t eshkol_builtin_websocket_send_binary_v(
    eshkol_sysbuiltin_value_t ws_val,
    eshkol_sysbuiltin_value_t data_val) {
    int handle = (int)sys_extract_int64(ws_val);
    const char* data = sys_extract_string(data_val);
    if (!sys_ws_valid(handle) || !data) return sys_make_bool(0);
    return sys_make_bool(sys_ws_send_frame(g_sys_websocket_clients[handle].fd, 2,
                                           data, strlen(data)));
}

static eshkol_sysbuiltin_value_t eshkol_builtin_websocket_receive_v(
    eshkol_sysbuiltin_value_t ws_val,
    eshkol_sysbuiltin_value_t timeout_val) {
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    int handle = (int)sys_extract_int64(ws_val);
    int timeout_ms = (int)sys_extract_int64(timeout_val);
    if (timeout_ms < 0) timeout_ms = 0;
    if (!sys_ws_valid(handle)) return sys_make_bool(0);
    int fd = g_sys_websocket_clients[handle].fd;
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    if (poll(&pfd, 1, timeout_ms) <= 0 || !(pfd.revents & POLLIN))
        return sys_make_bool(0);

    unsigned char hdr[2];
    if (recv(fd, hdr, 2, MSG_WAITALL) != 2) return sys_make_bool(0);
    int opcode = hdr[0] & 0x0f;
    int masked = hdr[1] & 0x80;
    uint64_t len = hdr[1] & 0x7f;
    if (len == 126) {
        unsigned char ext[2];
        if (recv(fd, ext, 2, MSG_WAITALL) != 2) return sys_make_bool(0);
        len = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (len == 127) {
        unsigned char ext[8];
        if (recv(fd, ext, 8, MSG_WAITALL) != 8) return sys_make_bool(0);
        len = 0;
        for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
    }
    unsigned char mask[4] = {0, 0, 0, 0};
    if (masked && recv(fd, mask, 4, MSG_WAITALL) != 4) return sys_make_bool(0);
    if (len > 65536) return sys_make_bool(0);

    char* payload = (char*)malloc((size_t)len + 1);
    if (!payload) return sys_make_bool(0);
    if (recv(fd, payload, (size_t)len, MSG_WAITALL) != (ssize_t)len) {
        free(payload);
        return sys_make_bool(0);
    }
    if (masked) {
        for (uint64_t i = 0; i < len; i++) payload[i] ^= (char)mask[i % 4];
    }
    payload[len] = '\0';
    const char* type = opcode == 1 ? "text" :
                       opcode == 2 ? "binary" :
                       opcode == 9 ? "ping" :
                       opcode == 8 ? "close" : "unknown";
    if (opcode == 8) g_sys_websocket_clients[handle].closed = 1;
    eshkol_sysbuiltin_value_t result =
        sys_make_pair(sys_make_string(type), sys_make_string_len(payload, (size_t)len));
    free(payload);
    return result;
#else
    (void)ws_val; (void)timeout_val;
    return sys_make_bool(0);
#endif
}

static eshkol_sysbuiltin_value_t eshkol_builtin_websocket_close_v(eshkol_sysbuiltin_value_t ws_val) {
    int handle = (int)sys_extract_int64(ws_val);
    if (!sys_ws_valid(handle)) return sys_make_bool(0);
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM)
    (void)sys_ws_send_frame(g_sys_websocket_clients[handle].fd, 8, "", 0);
    close(g_sys_websocket_clients[handle].fd);
#endif
    memset(&g_sys_websocket_clients[handle], 0, sizeof(g_sys_websocket_clients[handle]));
    return sys_make_bool(1);
}

static char g_sys_http_proxy_url[512];
static char g_sys_http_tls_cert[512];
static char g_sys_http_tls_key[512];
static char g_sys_http_tls_ca[512];

static eshkol_sysbuiltin_value_t eshkol_builtin_http_set_proxy_v(eshkol_sysbuiltin_value_t proxy_val) {
    const char* proxy = sys_extract_string(proxy_val);
    if (!proxy || strlen(proxy) >= sizeof(g_sys_http_proxy_url))
        return sys_make_bool(0);
    snprintf(g_sys_http_proxy_url, sizeof(g_sys_http_proxy_url), "%s", proxy);
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_http_set_tls_client_cert_v(
    eshkol_sysbuiltin_value_t cert_val,
    eshkol_sysbuiltin_value_t key_val,
    eshkol_sysbuiltin_value_t ca_val) {
    const char* cert = sys_extract_string(cert_val);
    const char* key = sys_extract_string(key_val);
    const char* ca = sys_extract_string(ca_val);
    if (!cert || !key || !ca ||
        strlen(cert) >= sizeof(g_sys_http_tls_cert) ||
        strlen(key) >= sizeof(g_sys_http_tls_key) ||
        strlen(ca) >= sizeof(g_sys_http_tls_ca))
        return sys_make_bool(0);
    snprintf(g_sys_http_tls_cert, sizeof(g_sys_http_tls_cert), "%s", cert);
    snprintf(g_sys_http_tls_key, sizeof(g_sys_http_tls_key), "%s", key);
    snprintf(g_sys_http_tls_ca, sizeof(g_sys_http_tls_ca), "%s", ca);
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_display_error_v(eshkol_sysbuiltin_value_t str_val) {
    const char* str = sys_extract_string(str_val);
    if (!str) return sys_make_bool(0);
    fputs(str, stderr);
    fflush(stderr);
    return sys_make_bool(1);
}

typedef struct {
    int active;
    char language[32];
} SysTsParser;

typedef struct {
    int active;
    int parser;
    int root_node;
    const char* source;
    int64_t source_len;
    char language[32];
} SysTsTree;

typedef struct {
    int active;
    int tree;
    int parent;
    int64_t start;
    int64_t end;
    char type[48];
} SysTsNode;

typedef struct {
    int active;
    char language[32];
    char pattern[256];
    char capture[64];
} SysTsQuery;

static SysTsParser g_sys_ts_parsers[16];
static SysTsTree g_sys_ts_trees[16];
static SysTsNode g_sys_ts_nodes[128];
static SysTsQuery g_sys_ts_queries[32];

static int sys_ts_language_supported(const char* language) {
    static const char* names[] = {
        "javascript", "js", "typescript", "ts", "python", "py", "rust", "rs",
        "go", "c", "cpp", "c++", "java", "ruby", "rb", "bash", "sh",
        "scheme", "scm", "lisp", NULL
    };
    if (!language || !*language) return 0;
    for (int i = 0; names[i]; i++) {
        if (strcmp(language, names[i]) == 0) return 1;
    }
    return 0;
}

static const char* sys_ts_root_type(const char* language) {
    if (!language) return "source_file";
    if (strcmp(language, "python") == 0 || strcmp(language, "py") == 0)
        return "module";
    if (strcmp(language, "javascript") == 0 || strcmp(language, "js") == 0 ||
        strcmp(language, "typescript") == 0 || strcmp(language, "ts") == 0)
        return "program";
    if (strcmp(language, "c") == 0 || strcmp(language, "cpp") == 0 ||
        strcmp(language, "c++") == 0)
        return "translation_unit";
    return "source_file";
}

static int sys_ts_starts_with(const char* s, int64_t len, const char* prefix) {
    size_t n = prefix ? strlen(prefix) : 0;
    return s && prefix && len >= (int64_t)n && memcmp(s, prefix, n) == 0;
}

static int sys_ts_contains(const char* s, int64_t len, const char* needle) {
    size_t n = needle ? strlen(needle) : 0;
    if (!s || !needle || n == 0 || len < (int64_t)n) return 0;
    for (int64_t i = 0; i <= len - (int64_t)n; i++) {
        if (memcmp(s + i, needle, n) == 0) return 1;
    }
    return 0;
}

static const char* sys_ts_classify_line(const char* language, const char* line, int64_t len) {
    while (len > 0 && isspace((unsigned char)*line)) {
        line++;
        len--;
    }
    if (len <= 0) return NULL;
    if (language && (strcmp(language, "python") == 0 || strcmp(language, "py") == 0)) {
        if (sys_ts_starts_with(line, len, "def ") || sys_ts_starts_with(line, len, "async def "))
            return "function_definition";
        if (sys_ts_starts_with(line, len, "class "))
            return "class_definition";
        if (sys_ts_starts_with(line, len, "import ") || sys_ts_starts_with(line, len, "from "))
            return "import_statement";
    } else if (language && (strcmp(language, "javascript") == 0 || strcmp(language, "js") == 0 ||
                            strcmp(language, "typescript") == 0 || strcmp(language, "ts") == 0)) {
        if (sys_ts_starts_with(line, len, "function ") || sys_ts_contains(line, len, "=>"))
            return "function_declaration";
        if (sys_ts_starts_with(line, len, "class "))
            return "class_declaration";
        if (sys_ts_starts_with(line, len, "import "))
            return "import_statement";
    } else if (language && (strcmp(language, "c") == 0 || strcmp(language, "cpp") == 0 ||
                            strcmp(language, "c++") == 0)) {
        if (sys_ts_starts_with(line, len, "#include"))
            return "preproc_include";
        if (sys_ts_contains(line, len, "(") && sys_ts_contains(line, len, ")") &&
            sys_ts_contains(line, len, "{"))
            return "function_definition";
    } else if (language && (strcmp(language, "scheme") == 0 || strcmp(language, "scm") == 0 ||
                            strcmp(language, "lisp") == 0)) {
        if (sys_ts_starts_with(line, len, "(define "))
            return "definition";
        if (sys_ts_starts_with(line, len, "(lambda"))
            return "lambda_expression";
    }
    return "line";
}

static int sys_ts_alloc_node(int tree, int parent, int64_t start, int64_t end, const char* type) {
    if (!type) return -1;
    for (int i = 16; i < (int)(sizeof(g_sys_ts_nodes) / sizeof(g_sys_ts_nodes[0])); i++) {
        if (!g_sys_ts_nodes[i].active) {
            g_sys_ts_nodes[i].active = 1;
            g_sys_ts_nodes[i].tree = tree;
            g_sys_ts_nodes[i].parent = parent;
            g_sys_ts_nodes[i].start = start;
            g_sys_ts_nodes[i].end = end;
            snprintf(g_sys_ts_nodes[i].type, sizeof(g_sys_ts_nodes[i].type), "%s", type);
            return i;
        }
    }
    return -1;
}

static eshkol_sysbuiltin_value_t sys_reverse_list(eshkol_sysbuiltin_value_t list) {
    eshkol_sysbuiltin_value_t out = sys_make_null();
    while (list.type == SYS_TYPE_HEAP_PTR && list.flags == 0x00 && list.data != 0) {
        eshkol_sysbuiltin_value_t car;
        eshkol_sysbuiltin_value_t cdr;
        memcpy(&car, (void*)(uintptr_t)list.data, sizeof(car));
        memcpy(&cdr, (char*)(uintptr_t)list.data + sizeof(car), sizeof(cdr));
        out = sys_make_pair(car, out);
        list = cdr;
    }
    return out;
}

static void sys_ts_capture_name(const char* pattern, char* out, size_t out_cap) {
    if (!out || out_cap == 0) return;
    snprintf(out, out_cap, "match");
    const char* at = pattern ? strchr(pattern, '@') : NULL;
    if (!at) return;
    at++;
    size_t n = 0;
    while (at[n] && (isalnum((unsigned char)at[n]) || at[n] == '_' || at[n] == '-' || at[n] == '.'))
        n++;
    if (n == 0) return;
    if (n >= out_cap) n = out_cap - 1;
    memcpy(out, at, n);
    out[n] = '\0';
}

static int sys_ts_query_matches_text(const char* pattern, const char* type,
                                     const char* text, int64_t text_len) {
    if (!pattern || !*pattern) return 1;
    if (type && strstr(pattern, type)) return 1;
    int wants_structural = strstr(pattern, "function") || strstr(pattern, "definition") ||
                           strstr(pattern, "class") || strstr(pattern, "import");
    if (strstr(pattern, "function")) return type && strstr(type, "function");
    if (strstr(pattern, "definition")) return type && strstr(type, "definition");
    if (strstr(pattern, "class")) return type && strstr(type, "class");
    if (strstr(pattern, "import")) return type && strstr(type, "import");
    if (wants_structural) return 0;
    if (strstr(pattern, "identifier")) {
        for (int64_t i = 0; i < text_len; i++) {
            if (isalpha((unsigned char)text[i]) || text[i] == '_') return 1;
        }
    }
    return sys_ts_contains(text, text_len, pattern);
}

static eshkol_sysbuiltin_value_t sys_ts_match_value(const char* capture, const char* type,
                                                     int64_t start, int64_t end,
                                                     const char* text, int64_t text_len) {
    eshkol_sysbuiltin_value_t match = sys_make_null();
    match = sys_make_pair(sys_alist_entry("text", sys_make_string_len(text, (size_t)text_len)), match);
    match = sys_make_pair(sys_alist_entry("end", sys_make_int64(end)), match);
    match = sys_make_pair(sys_alist_entry("start", sys_make_int64(start)), match);
    match = sys_make_pair(sys_alist_entry("type", sys_make_string(type)), match);
    match = sys_make_pair(sys_alist_entry("capture", sys_make_string(capture)), match);
    return match;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_parser_new_v(eshkol_sysbuiltin_value_t language_val) {
    const char* language = sys_extract_string(language_val);
    if (!sys_ts_language_supported(language)) return sys_make_bool(0);
    for (int i = 1; i < (int)(sizeof(g_sys_ts_parsers) / sizeof(g_sys_ts_parsers[0])); i++) {
        if (!g_sys_ts_parsers[i].active) {
            g_sys_ts_parsers[i].active = 1;
            snprintf(g_sys_ts_parsers[i].language, sizeof(g_sys_ts_parsers[i].language), "%s", language);
            return sys_make_int64((int64_t)i);
        }
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_parser_free_v(eshkol_sysbuiltin_value_t parser_val) {
    int handle = (int)sys_extract_int64(parser_val);
    if (handle > 0 && handle < (int)(sizeof(g_sys_ts_parsers) / sizeof(g_sys_ts_parsers[0])) &&
        g_sys_ts_parsers[handle].active) {
        memset(&g_sys_ts_parsers[handle], 0, sizeof(g_sys_ts_parsers[handle]));
        return sys_make_bool(1);
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_parse_v(eshkol_sysbuiltin_value_t parser_val,
                                                           eshkol_sysbuiltin_value_t source_val) {
    int parser = (int)sys_extract_int64(parser_val);
    const char* source = sys_extract_string(source_val);
    if (parser <= 0 || parser >= (int)(sizeof(g_sys_ts_parsers) / sizeof(g_sys_ts_parsers[0])) ||
        !g_sys_ts_parsers[parser].active || !source)
        return sys_make_bool(0);
    for (int i = 1; i < (int)(sizeof(g_sys_ts_trees) / sizeof(g_sys_ts_trees[0])); i++) {
        if (!g_sys_ts_trees[i].active) {
            g_sys_ts_trees[i].active = 1;
            g_sys_ts_trees[i].parser = parser;
            g_sys_ts_trees[i].root_node = i;
            g_sys_ts_trees[i].source = source;
            g_sys_ts_trees[i].source_len = (int64_t)strlen(source);
            snprintf(g_sys_ts_trees[i].language, sizeof(g_sys_ts_trees[i].language),
                     "%s", g_sys_ts_parsers[parser].language);
            memset(&g_sys_ts_nodes[i], 0, sizeof(g_sys_ts_nodes[i]));
            g_sys_ts_nodes[i].active = 1;
            g_sys_ts_nodes[i].tree = i;
            g_sys_ts_nodes[i].parent = 0;
            g_sys_ts_nodes[i].start = 0;
            g_sys_ts_nodes[i].end = g_sys_ts_trees[i].source_len;
            snprintf(g_sys_ts_nodes[i].type, sizeof(g_sys_ts_nodes[i].type),
                     "%s", sys_ts_root_type(g_sys_ts_trees[i].language));
            return sys_make_int64((int64_t)i);
        }
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_tree_free_v(eshkol_sysbuiltin_value_t tree_val) {
    int handle = (int)sys_extract_int64(tree_val);
    if (handle > 0 && handle < (int)(sizeof(g_sys_ts_trees) / sizeof(g_sys_ts_trees[0])) &&
        g_sys_ts_trees[handle].active) {
        for (int i = 0; i < (int)(sizeof(g_sys_ts_nodes) / sizeof(g_sys_ts_nodes[0])); i++) {
            if (g_sys_ts_nodes[i].active && g_sys_ts_nodes[i].tree == handle)
                memset(&g_sys_ts_nodes[i], 0, sizeof(g_sys_ts_nodes[i]));
        }
        memset(&g_sys_ts_trees[handle], 0, sizeof(g_sys_ts_trees[handle]));
        return sys_make_bool(1);
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_node_type_v(eshkol_sysbuiltin_value_t node_val) {
    int node = (int)sys_extract_int64(node_val);
    if (node > 0 && node < (int)(sizeof(g_sys_ts_nodes) / sizeof(g_sys_ts_nodes[0])) &&
        g_sys_ts_nodes[node].active)
        return sys_make_string(g_sys_ts_nodes[node].type);
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_node_text_v(eshkol_sysbuiltin_value_t node_val,
                                                               eshkol_sysbuiltin_value_t source_val) {
    int node = (int)sys_extract_int64(node_val);
    const char* fallback = sys_extract_string(source_val);
    if (node <= 0 || node >= (int)(sizeof(g_sys_ts_nodes) / sizeof(g_sys_ts_nodes[0])) ||
        !g_sys_ts_nodes[node].active)
        return sys_make_bool(0);
    int tree = g_sys_ts_nodes[node].tree;
    const char* source = fallback;
    int64_t source_len = fallback ? (int64_t)strlen(fallback) : 0;
    if (tree > 0 && tree < (int)(sizeof(g_sys_ts_trees) / sizeof(g_sys_ts_trees[0])) &&
        g_sys_ts_trees[tree].active && g_sys_ts_trees[tree].source) {
        source = g_sys_ts_trees[tree].source;
        source_len = g_sys_ts_trees[tree].source_len;
    }
    if (!source || g_sys_ts_nodes[node].start < 0 || g_sys_ts_nodes[node].end > source_len ||
        g_sys_ts_nodes[node].start > g_sys_ts_nodes[node].end)
        return sys_make_bool(0);
    return sys_make_string_len(source + g_sys_ts_nodes[node].start,
                               (size_t)(g_sys_ts_nodes[node].end - g_sys_ts_nodes[node].start));
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_node_children_v(eshkol_sysbuiltin_value_t node_val) {
    int node = (int)sys_extract_int64(node_val);
    if (node <= 0 || node >= (int)(sizeof(g_sys_ts_nodes) / sizeof(g_sys_ts_nodes[0])) ||
        !g_sys_ts_nodes[node].active)
        return sys_make_bool(0);
    int tree = g_sys_ts_nodes[node].tree;
    if (tree <= 0 || tree >= (int)(sizeof(g_sys_ts_trees) / sizeof(g_sys_ts_trees[0])) ||
        !g_sys_ts_trees[tree].active || !g_sys_ts_trees[tree].source)
        return sys_make_bool(0);
    if (g_sys_ts_nodes[node].parent != 0)
        return sys_make_null();
    const char* src = g_sys_ts_trees[tree].source;
    int64_t len = g_sys_ts_trees[tree].source_len;
    eshkol_sysbuiltin_value_t out = sys_make_null();
    int64_t line_start = 0;
    for (int64_t i = 0; i <= len; i++) {
        if (i == len || src[i] == '\n') {
            int64_t line_end = i;
            const char* type = sys_ts_classify_line(g_sys_ts_trees[tree].language,
                                                    src + line_start, line_end - line_start);
            if (type) {
                int child = sys_ts_alloc_node(tree, node, line_start, line_end, type);
                if (child > 0) out = sys_make_pair(sys_make_int64((int64_t)child), out);
            }
            line_start = i + 1;
        }
    }
    return sys_reverse_list(out);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_query_new_v(eshkol_sysbuiltin_value_t language_val,
                                                               eshkol_sysbuiltin_value_t pattern_val) {
    const char* language = sys_extract_string(language_val);
    const char* pattern = sys_extract_string(pattern_val);
    if (!sys_ts_language_supported(language) || !pattern) return sys_make_bool(0);
    for (int i = 1; i < (int)(sizeof(g_sys_ts_queries) / sizeof(g_sys_ts_queries[0])); i++) {
        if (!g_sys_ts_queries[i].active) {
            g_sys_ts_queries[i].active = 1;
            snprintf(g_sys_ts_queries[i].language, sizeof(g_sys_ts_queries[i].language), "%s", language);
            snprintf(g_sys_ts_queries[i].pattern, sizeof(g_sys_ts_queries[i].pattern), "%s", pattern);
            sys_ts_capture_name(g_sys_ts_queries[i].pattern, g_sys_ts_queries[i].capture,
                                sizeof(g_sys_ts_queries[i].capture));
            return sys_make_int64((int64_t)i);
        }
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_query_matches_v(eshkol_sysbuiltin_value_t query_val,
                                                                   eshkol_sysbuiltin_value_t tree_val,
                                                                   eshkol_sysbuiltin_value_t source_val) {
    (void)source_val;
    int query = (int)sys_extract_int64(query_val);
    int tree = (int)sys_extract_int64(tree_val);
    if (query <= 0 || query >= (int)(sizeof(g_sys_ts_queries) / sizeof(g_sys_ts_queries[0])) ||
        tree <= 0 || tree >= (int)(sizeof(g_sys_ts_trees) / sizeof(g_sys_ts_trees[0])) ||
        !g_sys_ts_queries[query].active || !g_sys_ts_trees[tree].active ||
        !g_sys_ts_trees[tree].source)
        return sys_make_bool(0);
    const char* src = g_sys_ts_trees[tree].source;
    int64_t len = g_sys_ts_trees[tree].source_len;
    eshkol_sysbuiltin_value_t out = sys_make_null();
    int64_t line_start = 0;
    for (int64_t i = 0; i <= len; i++) {
        if (i == len || src[i] == '\n') {
            int64_t line_end = i;
            const char* type = sys_ts_classify_line(g_sys_ts_trees[tree].language,
                                                    src + line_start, line_end - line_start);
            if (type && sys_ts_query_matches_text(g_sys_ts_queries[query].pattern, type,
                                                  src + line_start, line_end - line_start)) {
                out = sys_make_pair(sys_ts_match_value(g_sys_ts_queries[query].capture, type,
                                                       line_start, line_end,
                                                       src + line_start, line_end - line_start),
                                    out);
            }
            line_start = i + 1;
        }
    }
    return sys_reverse_list(out);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_query_free_v(eshkol_sysbuiltin_value_t query_val) {
    int handle = (int)sys_extract_int64(query_val);
    if (handle > 0 && handle < (int)(sizeof(g_sys_ts_queries) / sizeof(g_sys_ts_queries[0])) &&
        g_sys_ts_queries[handle].active) {
        memset(&g_sys_ts_queries[handle], 0, sizeof(g_sys_ts_queries[handle]));
        return sys_make_bool(1);
    }
    return sys_make_bool(0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_available_v(void) {
    return sys_make_bool(1);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_ts_tree_root_v(eshkol_sysbuiltin_value_t tree_val) {
    int tree = (int)sys_extract_int64(tree_val);
    if (tree > 0 && tree < (int)(sizeof(g_sys_ts_trees) / sizeof(g_sys_ts_trees[0])) &&
        g_sys_ts_trees[tree].active)
        return sys_make_int64((int64_t)g_sys_ts_trees[tree].root_node);
    return sys_make_bool(0);
}

static int sys_utf8_encode_codepoint(int cp, char* out) {
    if (!out) return 0;
    if (cp < 0) cp = ' ';
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    out[0] = ' ';
    return 1;
}

static eshkol_sysbuiltin_value_t eshkol_builtin_string_ends_with_v(
    eshkol_sysbuiltin_value_t str_val,
    eshkol_sysbuiltin_value_t suffix_val) {
    const char* s = sys_extract_string(str_val);
    const char* suffix = sys_extract_string(suffix_val);
    if (!s || !suffix) return sys_make_bool(0);
    size_t s_len = strlen(s);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > s_len) return sys_make_bool(0);
    return sys_make_bool(memcmp(s + s_len - suffix_len, suffix, suffix_len) == 0);
}

static eshkol_sysbuiltin_value_t eshkol_builtin_string_index_of_v(
    eshkol_sysbuiltin_value_t str_val,
    eshkol_sysbuiltin_value_t sub_val,
    eshkol_sysbuiltin_value_t start_val) {
    const char* s = sys_extract_string(str_val);
    const char* sub = sys_extract_string(sub_val);
    if (!s || !sub) return sys_make_bool(0);
    int64_t start = (int64_t)start_val.data;
    size_t s_len = strlen(s);
    if (start < 0) start = 0;
    if ((size_t)start > s_len) return sys_make_bool(0);
    if (*sub == '\0') return sys_make_int64(start);
    const char* found = strstr(s + start, sub);
    if (!found) return sys_make_bool(0);
    return sys_make_int64((int64_t)(found - s));
}

static eshkol_sysbuiltin_value_t eshkol_builtin_string_pad_v(
    eshkol_sysbuiltin_value_t str_val,
    eshkol_sysbuiltin_value_t width_val,
    eshkol_sysbuiltin_value_t ch_val,
    int left) {
    const char* s = sys_extract_string(str_val);
    if (!s) return sys_make_bool(0);
    int64_t width = (int64_t)width_val.data;
    size_t s_len = strlen(s);
    if (width <= (int64_t)s_len) return sys_make_string(s);
    if (width > 1000000) width = 1000000;
    char enc[4];
    int enc_len = sys_utf8_encode_codepoint((int)((int64_t)ch_val.data), enc);
    int64_t pad_count = width - (int64_t)s_len;
    size_t out_len = s_len + (size_t)pad_count * (size_t)enc_len;
    char* out = (char*)malloc(out_len + 1);
    if (!out) return sys_make_bool(0);
    char* p = out;
    if (!left) {
        memcpy(p, s, s_len);
        p += s_len;
    }
    for (int64_t i = 0; i < pad_count; i++) {
        memcpy(p, enc, (size_t)enc_len);
        p += enc_len;
    }
    if (left) {
        memcpy(p, s, s_len);
        p += s_len;
    }
    *p = '\0';
    eshkol_sysbuiltin_value_t result = sys_make_string(out);
    free(out);
    return result;
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
void eshkol_builtin_format_relative(sv_t* out, const sv_t* a) { *out = eshkol_builtin_format_relative_v(*a); }
void eshkol_builtin_local_timezone_offset(sv_t* out) { *out = eshkol_builtin_local_timezone_offset_v(); }
void eshkol_builtin_executable_exists(sv_t* out, const sv_t* a) { *out = eshkol_builtin_executable_exists_v(*a); }
void eshkol_builtin_executable_path(sv_t* out, const sv_t* a) { *out = eshkol_builtin_executable_path_v(*a); }
void eshkol_builtin_monotonic_time_ms(sv_t* out) { *out = eshkol_builtin_monotonic_time_ms_v(); }
void eshkol_builtin_temp_directory(sv_t* out) { *out = eshkol_builtin_temp_directory_v(); }
void eshkol_builtin_prevent_sleep(sv_t* out, const sv_t* a) { *out = eshkol_builtin_prevent_sleep_v(*a); }
void eshkol_builtin_allow_sleep(sv_t* out, const sv_t* a) { *out = eshkol_builtin_allow_sleep_v(*a); }
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
void eshkol_builtin_make_temp_file(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_make_temp_file_v(*a, *b, *c); }
void eshkol_builtin_make_temp_dir(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_make_temp_dir_v(*a, *b); }
void eshkol_builtin_directory_delete_recursive(sv_t* out, const sv_t* a) { *out = eshkol_builtin_directory_delete_recursive_v(*a); }
void eshkol_builtin_shell_quote(sv_t* out, const sv_t* a) { *out = eshkol_builtin_shell_quote_v(*a); }
void eshkol_builtin_fork(sv_t* out) { *out = eshkol_builtin_fork_v(); }
void eshkol_builtin_execv(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_execv_v(*a, *b); }
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
void eshkol_builtin_term_set_scroll_region(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_term_set_scroll_region_v(*a, *b); }
void eshkol_builtin_term_reset_scroll_region(sv_t* out) { *out = eshkol_builtin_term_reset_scroll_region_v(); }
void eshkol_builtin_term_enable_mouse(sv_t* out) { *out = eshkol_builtin_term_enable_mouse_v(); }
void eshkol_builtin_term_disable_mouse(sv_t* out) { *out = eshkol_builtin_term_disable_mouse_v(); }
void eshkol_builtin_term_read_mouse_event(sv_t* out, const sv_t* a) { *out = eshkol_builtin_term_read_mouse_event_v(*a); }
void eshkol_builtin_term_enable_alternate_screen(sv_t* out) { *out = eshkol_builtin_term_enable_alternate_screen_v(); }
void eshkol_builtin_term_disable_alternate_screen(sv_t* out) { *out = eshkol_builtin_term_disable_alternate_screen_v(); }
void eshkol_builtin_term_clipboard_write(sv_t* out, const sv_t* a) { *out = eshkol_builtin_term_clipboard_write_v(*a); }
void eshkol_builtin_term_clipboard_read(sv_t* out) { *out = eshkol_builtin_term_clipboard_read_v(); }
void eshkol_builtin_term_hyperlink(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_term_hyperlink_v(*a, *b); }
void eshkol_builtin_term_detect_capabilities(sv_t* out) { *out = eshkol_builtin_term_detect_capabilities_v(); }
void eshkol_builtin_term_bell(sv_t* out) { *out = eshkol_builtin_term_bell_v(); }
void eshkol_builtin_fs_watch_native(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_fs_watch_native_v(*a, *b); }
void eshkol_builtin_fs_watch_recursive(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_fs_watch_recursive_v(*a, *b); }
void eshkol_builtin_fs_watch_poll(sv_t* out, const sv_t* a) { *out = eshkol_builtin_fs_watch_poll_v(*a); }
void eshkol_builtin_fs_unwatch(sv_t* out, const sv_t* a) { *out = eshkol_builtin_fs_unwatch_v(*a); }
void eshkol_builtin_ansi_strip(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ansi_strip_v(*a); }
void eshkol_builtin_string_display_width(sv_t* out, const sv_t* a) { *out = eshkol_builtin_string_display_width_v(*a); }
void eshkol_builtin_string_truncate_display(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_string_truncate_display_v(*a, *b, *c); }
void eshkol_builtin_url_encode(sv_t* out, const sv_t* a) { *out = eshkol_builtin_url_encode_v(*a); }
void eshkol_builtin_url_decode(sv_t* out, const sv_t* a) { *out = eshkol_builtin_url_decode_v(*a); }
void eshkol_builtin_url_parse(sv_t* out, const sv_t* a) { *out = eshkol_builtin_url_parse_v(*a); }
void eshkol_builtin_base64url_encode(sv_t* out, const sv_t* a) { *out = eshkol_builtin_base64url_encode_v(*a); }
void eshkol_builtin_base64url_decode(sv_t* out, const sv_t* a) { *out = eshkol_builtin_base64url_decode_v(*a); }
void eshkol_builtin_uuid_v4(sv_t* out) { *out = eshkol_builtin_uuid_v4_v(); }
void eshkol_builtin_constant_time_equal(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_constant_time_equal_v(*a, *b); }
void eshkol_builtin_sha256_file(sv_t* out, const sv_t* a) { *out = eshkol_builtin_sha256_file_v(*a); }
void eshkol_builtin_regex_compile(sv_t* out, const sv_t* a) { *out = eshkol_builtin_regex_compile_v(*a); }
void eshkol_builtin_regex_free(sv_t* out, const sv_t* a) { *out = eshkol_builtin_regex_free_v(*a); }
void eshkol_builtin_regex_match(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_regex_match_v(*a, *b, 0); }
void eshkol_builtin_regex_match_p(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_regex_match_v(*a, *b, 1); }
void eshkol_builtin_regex_match_groups(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_regex_match_groups_v(*a, *b); }
void eshkol_builtin_regex_split(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_regex_split_v(*a, *b); }
void eshkol_builtin_diff_lines(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_diff_lines_v(*a, *b); }
void eshkol_builtin_fuzzy_match(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c, const sv_t* d) { *out = eshkol_builtin_fuzzy_match_v(*a, *b, *c, *d); }
void eshkol_builtin_semver_parse(sv_t* out, const sv_t* a) { *out = eshkol_builtin_semver_parse_v(*a); }
void eshkol_builtin_semver_compare(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_semver_compare_v(*a, *b); }
void eshkol_builtin_semver_satisfies(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_semver_satisfies_v(*a, *b); }
void eshkol_builtin_make_pipe(sv_t* out) { *out = eshkol_builtin_make_pipe_v(); }
void eshkol_builtin_fd_write(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_fd_write_v(*a, *b); }
void eshkol_builtin_make_line_reader(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_make_line_reader_v(*a, *b); }
void eshkol_builtin_line_reader_poll(sv_t* out, const sv_t* a) { *out = eshkol_builtin_line_reader_poll_v(*a); }
void eshkol_builtin_line_reader_close(sv_t* out, const sv_t* a) { *out = eshkol_builtin_line_reader_close_v(*a); }
void eshkol_builtin_fd_close(sv_t* out, const sv_t* a) { *out = eshkol_builtin_fd_close_v(*a); }
void eshkol_builtin_make_lru_cache(sv_t* out, const sv_t* a) { *out = eshkol_builtin_make_lru_cache_v(*a); }
void eshkol_builtin_lru_get(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_lru_get_v(*a, *b); }
void eshkol_builtin_lru_set(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_lru_set_v(*a, *b, *c); }
void eshkol_builtin_lru_has(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_lru_has_v(*a, *b); }
void eshkol_builtin_lru_delete(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_lru_delete_v(*a, *b); }
void eshkol_builtin_lru_clear(sv_t* out, const sv_t* a) { *out = eshkol_builtin_lru_clear_v(*a); }
void eshkol_builtin_lru_size(sv_t* out, const sv_t* a) { *out = eshkol_builtin_lru_size_v(*a); }
void eshkol_builtin_format_list(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_format_list_v(*a, *b); }
void eshkol_builtin_http_server_create(sv_t* out, const sv_t* a) { *out = eshkol_builtin_http_server_create_v(*a); }
void eshkol_builtin_http_server_port(sv_t* out, const sv_t* a) { *out = eshkol_builtin_http_server_port_v(*a); }
void eshkol_builtin_http_server_accept(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_http_server_accept_v(*a, *b, *c); }
void eshkol_builtin_http_server_respond(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c, const sv_t* d) { *out = eshkol_builtin_http_server_respond_v(*a, *b, *c, *d); }
void eshkol_builtin_http_server_close(sv_t* out, const sv_t* a) { *out = eshkol_builtin_http_server_close_v(*a); }
void eshkol_builtin_http_request(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c, const sv_t* d, const sv_t* e) { *out = eshkol_builtin_http_request_v(*a, *b, *c, *d, *e); }
void eshkol_builtin_websocket_connect(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_websocket_connect_v(*a, *b); }
void eshkol_builtin_websocket_send(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_websocket_send_v(*a, *b); }
void eshkol_builtin_websocket_send_binary(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_websocket_send_binary_v(*a, *b); }
void eshkol_builtin_websocket_receive(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_websocket_receive_v(*a, *b); }
void eshkol_builtin_websocket_close(sv_t* out, const sv_t* a) { *out = eshkol_builtin_websocket_close_v(*a); }
void eshkol_builtin_ts_parser_new(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ts_parser_new_v(*a); }
void eshkol_builtin_ts_parser_free(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ts_parser_free_v(*a); }
void eshkol_builtin_ts_parse(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_ts_parse_v(*a, *b); }
void eshkol_builtin_ts_tree_free(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ts_tree_free_v(*a); }
void eshkol_builtin_ts_node_type(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ts_node_type_v(*a); }
void eshkol_builtin_ts_node_text(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_ts_node_text_v(*a, *b); }
void eshkol_builtin_ts_node_children(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ts_node_children_v(*a); }
void eshkol_builtin_ts_query_new(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_ts_query_new_v(*a, *b); }
void eshkol_builtin_ts_query_matches(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_ts_query_matches_v(*a, *b, *c); }
void eshkol_builtin_ts_query_free(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ts_query_free_v(*a); }
void eshkol_builtin_ts_available(sv_t* out) { *out = eshkol_builtin_ts_available_v(); }
void eshkol_builtin_ts_tree_root(sv_t* out, const sv_t* a) { *out = eshkol_builtin_ts_tree_root_v(*a); }
void eshkol_builtin_http_set_proxy(sv_t* out, const sv_t* a) { *out = eshkol_builtin_http_set_proxy_v(*a); }
void eshkol_builtin_http_set_tls_client_cert(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_http_set_tls_client_cert_v(*a, *b, *c); }
void eshkol_builtin_display_error(sv_t* out, const sv_t* a) { *out = eshkol_builtin_display_error_v(*a); }
void eshkol_builtin_string_ends_with(sv_t* out, const sv_t* a, const sv_t* b) { *out = eshkol_builtin_string_ends_with_v(*a, *b); }
void eshkol_builtin_string_index_of(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_string_index_of_v(*a, *b, *c); }
void eshkol_builtin_string_pad_left(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_string_pad_v(*a, *b, *c, 1); }
void eshkol_builtin_string_pad_right(sv_t* out, const sv_t* a, const sv_t* b, const sv_t* c) { *out = eshkol_builtin_string_pad_v(*a, *b, *c, 0); }
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
