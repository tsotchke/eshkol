/*******************************************************************************
 * System Platform Primitives for Eshkol Agent
 *
 * Provides: OS info, home directory, shell quoting, temp files, sleep,
 *           process identity, stderr output, recursive mkdir/rmdir,
 *           file stat, UUID, timestamps, executable search.
 *
 * All functions use the eshkol agent FFI conventions:
 *   - Strings: null-terminated, output via pre-allocated (buf, buf_size)
 *   - Returns: 0 success / -1 error, or strlen for string outputs
 *   - Handles: int64_t indices into static tables
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE    /* flock, mkdtemp, timegm on macOS */
#elif !defined(_WIN32)
#define _GNU_SOURCE         /* nftw, strptime, timegm, mkdtemp on Linux */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <wctype.h>
#include <limits.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winioctl.h>
#include <shlobj.h>
#include <lmcons.h>
#include <bcrypt.h>
#include <io.h>
#include <direct.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifndef PATH_MAX
#define PATH_MAX 32768
#endif
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <pwd.h>
#include <fnmatch.h>
#endif

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <copyfile.h>
#endif

#ifdef _WIN32
static wchar_t* platform_utf8_to_wide(const char* value) {
    if (!value) return NULL;
    int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                     value, -1, NULL, 0);
    if (needed <= 0) return NULL;
    wchar_t* wide = (wchar_t*)calloc((size_t)needed, sizeof(wchar_t));
    if (!wide) return NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                            value, -1, wide, needed) <= 0) {
        free(wide);
        return NULL;
    }
    return wide;
}

static int32_t platform_wide_to_utf8(const wchar_t* value,
                                     char* buf, int32_t buf_size) {
    if (!value || !buf || buf_size <= 0) return -1;
    int needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                     value, -1, NULL, 0, NULL, NULL);
    if (needed <= 0 || needed > buf_size) return -1;
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                            value, -1, buf, buf_size, NULL, NULL) <= 0) return -1;
    return needed - 1;
}

static int windows_random_bytes(unsigned char* bytes, size_t count) {
    if (!bytes || count > ULONG_MAX) return 0;
    return BCryptGenRandom(NULL, bytes, (ULONG)count,
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
}

typedef struct {
    DWORD ReparseTag;
    WORD ReparseDataLength;
    WORD Reserved;
    union {
        struct {
            WORD SubstituteNameOffset;
            WORD SubstituteNameLength;
            WORD PrintNameOffset;
            WORD PrintNameLength;
            DWORD Flags;
            WCHAR PathBuffer[1];
        } SymbolicLinkReparseBuffer;
        struct {
            WORD SubstituteNameOffset;
            WORD SubstituteNameLength;
            WORD PrintNameOffset;
            WORD PrintNameLength;
            WCHAR PathBuffer[1];
        } MountPointReparseBuffer;
    };
} EshkolReparseDataBuffer;

static void normalize_windows_path(char* path) {
    if (!path) return;
    for (char* p = path; *p; ++p) if (*p == '\\') *p = '/';
}

static char g_windows_temp_path[PATH_MAX];
static INIT_ONCE g_windows_temp_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK initialize_windows_temp_path(PINIT_ONCE once, PVOID parameter,
                                                   PVOID* context) {
    (void)once; (void)parameter; (void)context;
    wchar_t wide[32768];
    DWORD got = GetTempPathW((DWORD)(sizeof(wide) / sizeof(wide[0])), wide);
    if (got > 0 && got < sizeof(wide) / sizeof(wide[0]) &&
        platform_wide_to_utf8(wide, g_windows_temp_path,
                              (int32_t)sizeof(g_windows_temp_path)) >= 0) {
        normalize_windows_path(g_windows_temp_path);
        size_t n = strlen(g_windows_temp_path);
        while (n > 1 && g_windows_temp_path[n - 1] == '/') {
            g_windows_temp_path[--n] = '\0';
        }
        return TRUE;
    }
    strcpy(g_windows_temp_path, ".");
    return TRUE;
}

static void windows_slashes_to_backslashes(wchar_t* path) {
    if (!path) return;
    for (wchar_t* p = path; *p; ++p) if (*p == L'/') *p = L'\\';
}

static int windows_path_is_root(const wchar_t* path) {
    if (!path) return 1;
    size_t n = wcslen(path);
    if (n == 3 && path[1] == L':' && path[2] == L'\\') return 1;
    if (n >= 2 && path[0] == L'\\' && path[1] == L'\\') {
        const wchar_t* server_end = wcschr(path + 2, L'\\');
        if (!server_end) return 1;
        const wchar_t* share_end = wcschr(server_end + 1, L'\\');
        return share_end == NULL || share_end[1] == L'\0';
    }
    return 0;
}

static int windows_path_is_protected(const wchar_t* path) {
    if (windows_path_is_root(path)) return 1;
    wchar_t protected_path[32768];
    UINT n = GetWindowsDirectoryW(protected_path, 32768);
    if (n > 0 && n < 32768 && _wcsicmp(path, protected_path) == 0) return 1;
    n = GetSystemDirectoryW(protected_path, 32768);
    if (n > 0 && n < 32768 && _wcsicmp(path, protected_path) == 0) return 1;
    static const wchar_t* variables[] = {
        L"ProgramFiles", L"ProgramFiles(x86)", L"ProgramData", L"SystemDrive", NULL
    };
    for (const wchar_t** variable = variables; *variable; ++variable) {
        DWORD got = GetEnvironmentVariableW(*variable, protected_path, 32768);
        if (got > 0 && got < 32768) {
            windows_slashes_to_backslashes(protected_path);
            size_t len = wcslen(protected_path);
            while (len > 3 && protected_path[len - 1] == L'\\') protected_path[--len] = L'\0';
            if (_wcsicmp(path, protected_path) == 0) return 1;
        }
    }
    return 0;
}

static int windows_remove_tree(const wchar_t* path) {
    DWORD attrs = GetFileAttributesW(path);
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        return GetLastError() == ERROR_FILE_NOT_FOUND ||
               GetLastError() == ERROR_PATH_NOT_FOUND ? 0 : -1;
    }
    if (!(attrs & FILE_ATTRIBUTE_DIRECTORY) || (attrs & FILE_ATTRIBUTE_REPARSE_POINT)) {
        SetFileAttributesW(path, attrs & ~FILE_ATTRIBUTE_READONLY);
        return DeleteFileW(path) ? 0 : -1;
    }

    size_t n = wcslen(path);
    wchar_t* pattern = (wchar_t*)malloc((n + 3) * sizeof(wchar_t));
    if (!pattern) return -1;
    wcscpy(pattern, path);
    if (n && path[n - 1] != L'\\') pattern[n++] = L'\\';
    pattern[n++] = L'*';
    pattern[n] = L'\0';
    WIN32_FIND_DATAW data;
    HANDLE find = FindFirstFileW(pattern, &data);
    free(pattern);
    if (find != INVALID_HANDLE_VALUE) {
        int result = 0;
        do {
            if (wcscmp(data.cFileName, L".") == 0 || wcscmp(data.cFileName, L"..") == 0) continue;
            size_t child_len = wcslen(path) + wcslen(data.cFileName) + 2;
            wchar_t* child = (wchar_t*)malloc(child_len * sizeof(wchar_t));
            if (!child) { result = -1; break; }
            swprintf(child, child_len, L"%ls\\%ls", path, data.cFileName);
            if (windows_remove_tree(child) != 0) result = -1;
            free(child);
            if (result != 0) break;
        } while (FindNextFileW(find, &data));
        FindClose(find);
        if (result != 0) return -1;
    } else if (GetLastError() != ERROR_FILE_NOT_FOUND) {
        return -1;
    }
    SetFileAttributesW(path, attrs & ~FILE_ATTRIBUTE_READONLY);
    return RemoveDirectoryW(path) ? 0 : -1;
}

static int windows_glob_match_impl(const wchar_t* pattern, const wchar_t* text) {
    while (*pattern) {
        if (*pattern == L'*') {
            while (*pattern == L'*') ++pattern;
            if (!*pattern) return wcschr(text, L'/') == NULL && wcschr(text, L'\\') == NULL;
            for (const wchar_t* cursor = text;; ++cursor) {
                if (windows_glob_match_impl(pattern, cursor)) return 1;
                if (!*cursor || *cursor == L'/' || *cursor == L'\\') break;
            }
            return 0;
        }
        if (!*text) return 0;
        if (*pattern == L'?') {
            if (*text == L'/' || *text == L'\\') return 0;
            ++pattern; ++text;
            continue;
        }
        if (*pattern == L'[') {
            ++pattern;
            int negate = (*pattern == L'!' || *pattern == L'^');
            if (negate) ++pattern;
            int matched = 0;
            wchar_t value = (wchar_t)towlower(*text);
            while (*pattern && *pattern != L']') {
                wchar_t first = (wchar_t)towlower(*pattern++);
                if (*pattern == L'-' && pattern[1] && pattern[1] != L']') {
                    ++pattern;
                    wchar_t last = (wchar_t)towlower(*pattern++);
                    if (value >= first && value <= last) matched = 1;
                } else if (value == first) {
                    matched = 1;
                }
            }
            if (*pattern == L']') ++pattern;
            if (matched == negate) return 0;
            ++text;
            continue;
        }
        wchar_t p = *pattern == L'/' ? L'\\' : (wchar_t)towlower(*pattern);
        wchar_t t = *text == L'/' ? L'\\' : (wchar_t)towlower(*text);
        if (p != t) return 0;
        ++pattern; ++text;
    }
    return *text == L'\0';
}

int32_t eshkol_executable_path(const char* name, char* buf, int32_t buf_size);
#endif

/*******************************************************************************
 * D.1 / B.7: OS Information (compile-time constants)
 ******************************************************************************/

/**
 * @brief Returns a short lowercase identifier for the host operating system.
 *
 * @return A static string literal: "darwin", "linux", "windows", or
 *         "unknown", selected at compile time from platform macros.
 */
const char* eshkol_os_type(void) {
#ifdef __APPLE__
    return "darwin";
#elif defined(__linux__)
    return "linux";
#elif defined(_WIN32)
    return "windows";
#else
    return "unknown";
#endif
}

/**
 * @brief Returns a short lowercase identifier for the host CPU architecture.
 *
 * @return A static string literal: "aarch64", "x86_64", "x86", "arm", or
 *         "unknown", selected at compile time from platform macros.
 */
const char* eshkol_os_arch(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__)
    return "arm";
#else
    return "unknown";
#endif
}

/*******************************************************************************
 * B.7: Home Directory
 ******************************************************************************/

/**
 * @brief Writes the current user's home directory path into @p buf.
 *
 * Looks up the home directory via getpwuid() first, falling back to the
 * $HOME environment variable if the passwd entry is unavailable or empty.
 *
 * @param buf Destination buffer for the null-terminated path.
 * @param buf_size Capacity of @p buf.
 * @return Length of the path written (excluding the null terminator), or -1
 *         if no home directory could be determined or @p buf is too small.
 */
int32_t eshkol_home_directory(char* buf, int32_t buf_size) {
#ifdef _WIN32
    if (!buf || buf_size <= 0) return -1;
    PWSTR profile = NULL;
    if (SUCCEEDED(SHGetKnownFolderPath(&FOLDERID_Profile, 0, NULL, &profile))) {
        int32_t result = platform_wide_to_utf8(profile, buf, buf_size);
        CoTaskMemFree(profile);
        if (result >= 0) { normalize_windows_path(buf); return result; }
    }
    const char* home = getenv("USERPROFILE");
    if (!home || !*home) return -1;
    size_t len = strlen(home);
    if (len >= (size_t)buf_size) return -1;
    memcpy(buf, home, len + 1);
    normalize_windows_path(buf);
    return (int32_t)len;
#else
    /* Try getpwuid first (most reliable) */
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir && pw->pw_dir[0] != '\0') {
        int len = (int32_t)strlen(pw->pw_dir);
        if (len >= buf_size) return -1;
        memcpy(buf, pw->pw_dir, (size_t)len + 1);
        return len;
    }
    /* Fallback to $HOME */
    const char* home = getenv("HOME");
    if (home && home[0] != '\0') {
        int len = (int32_t)strlen(home);
        if (len >= buf_size) return -1;
        memcpy(buf, home, (size_t)len + 1);
        return len;
    }
    return -1;
#endif
}

/*******************************************************************************
 * B.7: Hostname, Username
 ******************************************************************************/

/**
 * @brief Writes the local machine's hostname into @p buf.
 *
 * @param buf Destination buffer; if the result would not fit, the last byte
 *        of @p buf is forced to '\0' to guarantee termination.
 * @param buf_size Capacity of @p buf.
 * @return Length of the hostname string, or -1 if gethostname() fails.
 */
int32_t eshkol_hostname(char* buf, int32_t buf_size) {
#ifdef _WIN32
    if (!buf || buf_size <= 0) return -1;
    wchar_t name[MAX_COMPUTERNAME_LENGTH + 1];
    DWORD count = MAX_COMPUTERNAME_LENGTH + 1;
    if (!GetComputerNameW(name, &count)) return -1;
    return platform_wide_to_utf8(name, buf, buf_size);
#else
    if (!buf || buf_size <= 0) return -1;
    if (gethostname(buf, (size_t)buf_size) != 0) return -1;
    buf[buf_size - 1] = '\0';
    return (int32_t)strlen(buf);
#endif
}

/**
 * @brief Writes the current user's login name into @p buf.
 *
 * @param buf Destination buffer for the null-terminated username.
 * @param buf_size Capacity of @p buf.
 * @return Length of the username, or -1 if the passwd entry is unavailable
 *         or @p buf is too small.
 */
int32_t eshkol_username(char* buf, int32_t buf_size) {
#ifdef _WIN32
    if (!buf || buf_size <= 0) return -1;
    wchar_t name[UNLEN + 1];
    DWORD count = UNLEN + 1;
    if (!GetUserNameW(name, &count)) return -1;
    return platform_wide_to_utf8(name, buf, buf_size);
#else
    struct passwd* pw = getpwuid(getuid());
    if (!pw || !pw->pw_name) return -1;
    int len = (int32_t)strlen(pw->pw_name);
    if (len >= buf_size) return -1;
    memcpy(buf, pw->pw_name, (size_t)len + 1);
    return len;
#endif
}

/*******************************************************************************
 * B.7: Executable Search (PATH lookup)
 ******************************************************************************/

/**
 * @brief Checks whether an executable named @p name can be found and run.
 *
 * If @p name is an absolute path, checks it directly with access(X_OK).
 * Otherwise searches each colon-separated directory in the $PATH
 * environment variable for an executable file with that name.
 *
 * @return 1 if a matching executable is found, 0 otherwise.
 */
int32_t eshkol_executable_exists(const char* name) {
#ifdef _WIN32
    char path[PATH_MAX];
    return eshkol_executable_path(name, path, (int32_t)sizeof(path)) >= 0 ? 1 : 0;
#else
    if (!name || name[0] == '\0') return 0;
    /* If absolute path, check directly */
    if (name[0] == '/') return access(name, X_OK) == 0 ? 1 : 0;

    const char* path_env = getenv("PATH");
    if (!path_env) return 0;

    char full[PATH_MAX];
    const char* p = path_env;
    while (*p) {
        const char* colon = strchr(p, ':');
        size_t dir_len = colon ? (size_t)(colon - p) : strlen(p);
        if (dir_len > 0 && dir_len + strlen(name) + 2 < PATH_MAX) {
            memcpy(full, p, dir_len);
            full[dir_len] = '/';
            strcpy(full + dir_len + 1, name);
            if (access(full, X_OK) == 0) return 1;
        }
        if (!colon) break;
        p = colon + 1;
    }
    return 0;
#endif
}

/**
 * @brief Resolves @p name to a full executable path, searching $PATH if needed.
 *
 * If @p name is an absolute path, verifies it is executable and copies it
 * to @p buf as-is. Otherwise searches each colon-separated directory in
 * $PATH for an executable file named @p name.
 *
 * @param buf Destination buffer for the resolved path.
 * @param buf_size Capacity of @p buf.
 * @return Length of the resolved path written to @p buf, or -1 if no
 *         matching executable is found or @p buf is too small.
 */
int32_t eshkol_executable_path(const char* name, char* buf, int32_t buf_size) {
    if (!name || !buf || buf_size < 2) return -1;
#ifdef _WIN32
    wchar_t* wide_name = platform_utf8_to_wide(name);
    if (!wide_name) return -1;
    DWORD capacity = 32768;
    wchar_t* resolved = (wchar_t*)calloc(capacity, sizeof(wchar_t));
    if (!resolved) { free(wide_name); return -1; }
    DWORD got = SearchPathW(NULL, wide_name, L".exe", capacity, resolved, NULL);
    if (got == 0 || got >= capacity) {
        got = SearchPathW(NULL, wide_name, NULL, capacity, resolved, NULL);
    }
    free(wide_name);
    if (got == 0 || got >= capacity) { free(resolved); return -1; }
    int32_t result = platform_wide_to_utf8(resolved, buf, buf_size);
    free(resolved);
    if (result >= 0) normalize_windows_path(buf);
    return result;
#else
    if (name[0] == '/') {
        if (access(name, X_OK) == 0) {
            int len = (int32_t)strlen(name);
            if (len >= buf_size) return -1;
            memcpy(buf, name, (size_t)len + 1);
            return len;
        }
        return -1;
    }
    const char* path_env = getenv("PATH");
    if (!path_env) return -1;

    char full[PATH_MAX];
    const char* p = path_env;
    while (*p) {
        const char* colon = strchr(p, ':');
        size_t dir_len = colon ? (size_t)(colon - p) : strlen(p);
        if (dir_len > 0 && dir_len + strlen(name) + 2 < PATH_MAX) {
            memcpy(full, p, dir_len);
            full[dir_len] = '/';
            strcpy(full + dir_len + 1, name);
            if (access(full, X_OK) == 0) {
                int len = (int32_t)strlen(full);
                if (len >= buf_size) return -1;
                memcpy(buf, full, (size_t)len + 1);
                return len;
            }
        }
        if (!colon) break;
        p = colon + 1;
    }
    return -1;
#endif
}

/*******************************************************************************
 * B.7: Time (millisecond precision)
 ******************************************************************************/

/**
 * @brief Returns the current wall-clock time as milliseconds since the Unix epoch.
 */
int64_t eshkol_current_time_ms(void) {
#ifdef _WIN32
    FILETIME ft;
    ULARGE_INTEGER ticks;
    GetSystemTimePreciseAsFileTime(&ft);
    ticks.LowPart = ft.dwLowDateTime;
    ticks.HighPart = ft.dwHighDateTime;
    /* FILETIME epoch is 1601-01-01 in 100 ns intervals. */
    return (int64_t)((ticks.QuadPart - 116444736000000000ULL) / 10000ULL);
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
#endif
}

/**
 * @brief Returns a monotonic clock reading in milliseconds, suitable for measuring elapsed time.
 *
 * Unlike eshkol_current_time_ms(), this is unaffected by system clock
 * adjustments and has no defined relation to wall-clock time.
 */
int64_t eshkol_monotonic_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    if (!QueryPerformanceCounter(&counter) || !QueryPerformanceFrequency(&frequency)) {
        return (int64_t)GetTickCount64();
    }
    return (int64_t)((counter.QuadPart * 1000LL) / frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
#endif
}

/**
 * @brief Returns the directory to use for temporary files.
 *
 * @return The value of $TMPDIR if set and non-empty, otherwise "/tmp".
 */
const char* eshkol_temp_directory(void) {
#ifdef _WIN32
    InitOnceExecuteOnce(&g_windows_temp_once, initialize_windows_temp_path,
                        NULL, NULL);
    return g_windows_temp_path;
#else
    const char* tmp = getenv("TMPDIR");
    if (tmp && tmp[0] != '\0') return tmp;
    return "/tmp";
#endif
}

/*******************************************************************************
 * E.6: getpid
 ******************************************************************************/

/**
 * @brief Returns the current process ID.
 */
int64_t eshkol_getpid_val(void) {
#ifdef _WIN32
    return (int64_t)GetCurrentProcessId();
#else
    return (int64_t)getpid();
#endif
}

/*******************************************************************************
 * E.5: stderr output
 ******************************************************************************/

/**
 * @brief Writes @p str to stderr and flushes immediately.
 *
 * No-op if @p str is NULL.
 */
void eshkol_eprint(const char* str) {
    if (str) {
        fputs(str, stderr);
        fflush(stderr);
    }
}

/*******************************************************************************
 * E.7: Precise millisecond sleep
 ******************************************************************************/

/**
 * @brief Sleeps the calling thread for approximately @p ms milliseconds.
 *
 * Uses nanosleep() internally; values of @p ms <= 0 return immediately
 * without sleeping.
 */
void eshkol_sleep_ms(int64_t ms) {
    if (ms <= 0) return;
#ifdef _WIN32
    while (ms > 0) {
        DWORD slice = ms > (int64_t)MAXDWORD ? MAXDWORD : (DWORD)ms;
        Sleep(slice);
        ms -= slice;
    }
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

/*******************************************************************************
 * B.22: Shell Quoting (POSIX single-quote escaping)
 ******************************************************************************/

/**
 * @brief Quotes @p str for safe inclusion as a single argument in a POSIX shell command line.
 *
 * If none of the shell-special characters (whitespace, quotes, `$`, `` ` ``,
 * globbing/redirection characters, etc.) are present, @p str is copied to
 * @p buf unchanged. Otherwise the result is wrapped in single quotes, with
 * any embedded single quote escaped as `'\''`.
 *
 * @param buf Destination buffer for the quoted string.
 * @param buf_size Capacity of @p buf; must be at least 3.
 * @return Length of the quoted string written to @p buf, or -1 if
 *         arguments are invalid or @p buf is too small.
 */
int32_t eshkol_shell_quote(const char* str, char* buf, int32_t buf_size) {
    if (!str || !buf || buf_size < 3) return -1;

#ifdef _WIN32
    int needs_quoting = *str == '\0';
    for (const char* p = str; *p; ++p) {
        if (*p == ' ' || *p == '\t' || *p == '"' || *p == '%' || *p == '!' ||
            *p == '&' || *p == '|' || *p == '<' || *p == '>' || *p == '^' ||
            *p == '(' || *p == ')') {
            needs_quoting = 1;
            break;
        }
    }
    if (!needs_quoting) {
        size_t len = strlen(str);
        if (len >= (size_t)buf_size) return -1;
        memcpy(buf, str, len + 1);
        return (int32_t)len;
    }
    int32_t out = 0;
    buf[out++] = '"';
    size_t slashes = 0;
    for (const char* p = str;; ++p) {
        char ch = *p;
        if (ch == '\\') { ++slashes; continue; }
        if (ch == '"' || ch == '\0') {
            size_t copies = slashes * (ch == '"' ? 2 : 2);
            while (copies--) { if (out >= buf_size - 1) return -1; buf[out++] = '\\'; }
            slashes = 0;
            if (ch == '"') {
                if (out >= buf_size - 2) return -1;
                buf[out++] = '\\'; buf[out++] = '"';
                continue;
            }
            break;
        }
        while (slashes--) { if (out >= buf_size - 1) return -1; buf[out++] = '\\'; }
        slashes = 0;
        if (ch == '%') {
            if (out >= buf_size - 2) return -1;
            buf[out++] = '%'; buf[out++] = '%';
        } else if (ch == '!') {
            if (out >= buf_size - 2) return -1;
            buf[out++] = '^'; buf[out++] = '!';
        } else {
            if (out >= buf_size - 1) return -1;
            buf[out++] = ch;
        }
    }
    if (out >= buf_size - 1) return -1;
    buf[out++] = '"'; buf[out] = '\0';
    return out;
#else
    /* Check if quoting is needed */
    int needs_quoting = 0;
    for (const char* p = str; *p; p++) {
        if (*p == ' ' || *p == '\'' || *p == '"' || *p == '\\' ||
            *p == '$' || *p == '`' || *p == '!' || *p == '&' ||
            *p == '|' || *p == ';' || *p == '(' || *p == ')' ||
            *p == '<' || *p == '>' || *p == '*' || *p == '?' ||
            *p == '[' || *p == ']' || *p == '{' || *p == '}' ||
            *p == '#' || *p == '~' || *p == '\n' || *p == '\t') {
            needs_quoting = 1;
            break;
        }
    }
    if (!needs_quoting) {
        int len = (int32_t)strlen(str);
        if (len >= buf_size) return -1;
        memcpy(buf, str, (size_t)len + 1);
        return len;
    }

    /* Single-quote escaping: wrap in '' and escape internal ' as '\'' */
    int32_t out = 0;
    buf[out++] = '\'';
    for (const char* p = str; *p; p++) {
        if (*p == '\'') {
            /* Need 4 chars: '\'' */
            if (out + 4 >= buf_size) return -1;
            buf[out++] = '\'';
            buf[out++] = '\\';
            buf[out++] = '\'';
            buf[out++] = '\'';
        } else {
            if (out + 1 >= buf_size) return -1;
            buf[out++] = *p;
        }
    }
    if (out + 1 >= buf_size) return -1;
    buf[out++] = '\'';
    buf[out] = '\0';
    return out;
#endif
}

/*******************************************************************************
 * E.8: Temp File/Dir Creation (race-free)
 ******************************************************************************/

/**
 * @brief Creates a uniquely-named temporary file (race-free) and returns its path.
 *
 * Builds a template "<dir-or-tmpdir>/<prefix>XXXXXX", creates it via
 * mkstemp(), closes the resulting file descriptor, and — if @p suffix is
 * non-empty — renames the file to append @p suffix.
 *
 * @param prefix Filename prefix; must be non-NULL.
 * @param suffix Optional suffix appended after the random characters (may
 *        be NULL/empty for none).
 * @param dir Optional directory to create the file in; falls back to
 *        eshkol_temp_directory() if NULL/empty.
 * @param path_buf Destination buffer for the created file's path.
 * @param buf_size Capacity of @p path_buf; must be at least 32.
 * @return Length of the path written to @p path_buf, or -1 on failure (the
 *         partially-created file is unlinked before returning).
 */
int32_t eshkol_mkstemp_path(const char* prefix, const char* suffix,
                             const char* dir, char* path_buf, int32_t buf_size) {
    if (!prefix || !path_buf || buf_size < 32) return -1;
#ifdef _WIN32
    const char* parent = dir && *dir ? dir : eshkol_temp_directory();
    for (int attempt = 0; attempt < 128; ++attempt) {
        unsigned char random[16];
        if (!windows_random_bytes(random, sizeof(random))) return -1;
        char name[128];
        int name_len = snprintf(name, sizeof(name),
            "%s%02x%02x%02x%02x%02x%02x%02x%02x%s",
            prefix, random[0], random[1], random[2], random[3],
            random[4], random[5], random[6], random[7], suffix ? suffix : "");
        if (name_len <= 0 || name_len >= (int)sizeof(name)) return -1;
        char candidate[PATH_MAX];
        int n = snprintf(candidate, sizeof(candidate), "%s/%s", parent, name);
        if (n <= 0 || n >= (int)sizeof(candidate)) return -1;
        wchar_t* wide = platform_utf8_to_wide(candidate);
        if (!wide) return -1;
        HANDLE file = CreateFileW(wide, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                                  CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, NULL);
        DWORD error = GetLastError();
        free(wide);
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
            if (n >= buf_size) {
                DeleteFileA(candidate);
                return -1;
            }
            memcpy(path_buf, candidate, (size_t)n + 1);
            normalize_windows_path(path_buf);
            return n;
        }
        if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS) return -1;
    }
    return -1;
#else
    const char* tmpdir = dir && dir[0] != '\0' ? dir : eshkol_temp_directory();
    char tmpl[PATH_MAX];
    int n = snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", tmpdir, prefix);
    if (n < 0 || n >= (int)sizeof(tmpl)) return -1;

    int fd = mkstemp(tmpl);
    if (fd < 0) return -1;
    close(fd);

    /* If suffix requested, rename */
    if (suffix && suffix[0] != '\0') {
        char final_path[PATH_MAX];
        snprintf(final_path, sizeof(final_path), "%s%s", tmpl, suffix);
        rename(tmpl, final_path);
        int len = (int32_t)strlen(final_path);
        if (len >= buf_size) { unlink(final_path); return -1; }
        memcpy(path_buf, final_path, (size_t)len + 1);
        return len;
    }

    int len = (int32_t)strlen(tmpl);
    if (len >= buf_size) { unlink(tmpl); return -1; }
    memcpy(path_buf, tmpl, (size_t)len + 1);
    return len;
#endif
}

/**
 * @brief Creates a uniquely-named temporary directory (race-free) and returns its path.
 *
 * Builds a template "<dir-or-tmpdir>/<prefix>XXXXXX" and creates it via
 * mkdtemp().
 *
 * @param prefix Directory name prefix; must be non-NULL.
 * @param dir Optional parent directory; falls back to
 *        eshkol_temp_directory() if NULL/empty.
 * @param path_buf Destination buffer for the created directory's path.
 * @param buf_size Capacity of @p path_buf; must be at least 32.
 * @return Length of the path written to @p path_buf, or -1 on failure.
 */
int32_t eshkol_mkdtemp_path(const char* prefix, const char* dir,
                              char* path_buf, int32_t buf_size) {
    if (!prefix || !path_buf || buf_size < 32) return -1;
#ifdef _WIN32
    const char* parent = dir && *dir ? dir : eshkol_temp_directory();
    for (int attempt = 0; attempt < 128; ++attempt) {
        unsigned char random[16];
        if (!windows_random_bytes(random, sizeof(random))) return -1;
        char candidate[PATH_MAX];
        int n = snprintf(candidate, sizeof(candidate),
            "%s/%s%02x%02x%02x%02x%02x%02x%02x%02x", parent, prefix,
            random[0], random[1], random[2], random[3],
            random[4], random[5], random[6], random[7]);
        if (n <= 0 || n >= (int)sizeof(candidate)) return -1;
        wchar_t* wide = platform_utf8_to_wide(candidate);
        if (!wide) return -1;
        BOOL made = CreateDirectoryW(wide, NULL);
        DWORD error = GetLastError();
        free(wide);
        if (made) {
            if (n >= buf_size) { RemoveDirectoryA(candidate); return -1; }
            memcpy(path_buf, candidate, (size_t)n + 1);
            normalize_windows_path(path_buf);
            return n;
        }
        if (error != ERROR_ALREADY_EXISTS) return -1;
    }
    return -1;
#else
    const char* tmpdir = dir && dir[0] != '\0' ? dir : eshkol_temp_directory();
    char tmpl[PATH_MAX];
    int n = snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", tmpdir, prefix);
    if (n < 0 || n >= (int)sizeof(tmpl)) return -1;

    if (!mkdtemp(tmpl)) return -1;

    int len = (int32_t)strlen(tmpl);
    if (len >= buf_size) return -1;
    memcpy(path_buf, tmpl, (size_t)len + 1);
    return len;
#endif
}

/*******************************************************************************
 * B.1: Recursive mkdir (create all parents)
 ******************************************************************************/

/**
 * @brief Creates @p path and all missing parent directories, like `mkdir -p`.
 *
 * Walks each '/'-separated path component, creating it with mode @p mode if
 * it does not already exist. Pre-existing components (EEXIST) are not
 * treated as errors.
 *
 * @return 0 on success, -1 if @p path is invalid, too long, or a component
 *         could not be created.
 */
int32_t eshkol_mkdir_recursive(const char* path, int32_t mode) {
    if (!path || path[0] == '\0') return -1;
#ifdef _WIN32
    (void)mode;
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    windows_slashes_to_backslashes(wide);
    size_t len = wcslen(wide);
    while (len > 3 && wide[len - 1] == L'\\') wide[--len] = L'\0';
    size_t start = 0;
    if (len >= 3 && wide[1] == L':' && wide[2] == L'\\') start = 3;
    else if (len >= 2 && wide[0] == L'\\' && wide[1] == L'\\') {
        wchar_t* server = wcschr(wide + 2, L'\\');
        wchar_t* share = server ? wcschr(server + 1, L'\\') : NULL;
        start = share ? (size_t)(share - wide + 1) : len;
    }
    int result = 0;
    for (size_t i = start; i <= len; ++i) {
        if (wide[i] != L'\\' && wide[i] != L'\0') continue;
        wchar_t saved = wide[i];
        wide[i] = L'\0';
        if (*wide && !CreateDirectoryW(wide, NULL)) {
            DWORD error = GetLastError();
            if (error != ERROR_ALREADY_EXISTS ||
                !(GetFileAttributesW(wide) & FILE_ATTRIBUTE_DIRECTORY)) {
                result = -1;
                wide[i] = saved;
                break;
            }
        }
        wide[i] = saved;
    }
    free(wide);
    return result;
#else
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len >= PATH_MAX) return -1;
    memcpy(tmp, path, len + 1);

    /* Walk each component and mkdir */
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, (mode_t)mode) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, (mode_t)mode) != 0 && errno != EEXIST) return -1;
    return 0;
#endif
}

/*******************************************************************************
 * B.1: Recursive rmdir (with safety checks)
 ******************************************************************************/

#ifndef _WIN32
static const char* g_dangerous_paths[] = {
    "/", "/usr", "/bin", "/sbin", "/etc", "/var", "/tmp",
    "/home", "/Users", "/System", "/Library", "/opt",
    NULL
};

/**
 * @brief nftw() visitor callback that deletes each visited file or directory.
 *
 * Used by eshkol_rmdir_recursive() with FTW_DEPTH so directories are
 * removed only after their contents; directories (FTW_D/FTW_DP) are
 * rmdir()'d, everything else is unlink()'d.
 *
 * @return The result of the underlying rmdir()/unlink() call.
 */
static int rmdir_recursive_cb(const char* fpath, const struct stat* sb,
                               int typeflag, struct FTW* ftwbuf) {
    (void)sb; (void)ftwbuf;
    if (typeflag == FTW_D || typeflag == FTW_DP) {
        return rmdir(fpath);
    }
    return unlink(fpath);
}
#endif

/**
 * @brief Recursively deletes the directory tree at @p path.
 *
 * Resolves @p path with realpath() and refuses to proceed if it matches one
 * of a hardcoded list of dangerous system directories (@ref
 * g_dangerous_paths), such as "/", "/usr", or "/Users". Deletion itself is
 * performed via nftw() with rmdir_recursive_cb().
 *
 * @return 0 on success or if @p path does not exist; -1 if @p path is
 *         invalid, resolves to a protected directory (errno set to EPERM),
 *         or deletion fails.
 */
int32_t eshkol_rmdir_recursive(const char* path) {
    if (!path || path[0] == '\0') return -1;

#ifdef _WIN32
    wchar_t* input = platform_utf8_to_wide(path);
    if (!input) return -1;
    wchar_t* resolved = (wchar_t*)calloc(32768, sizeof(wchar_t));
    if (!resolved) { free(input); return -1; }
    DWORD got = GetFullPathNameW(input, 32768, resolved, NULL);
    free(input);
    if (got == 0 || got >= 32768) { free(resolved); return -1; }
    windows_slashes_to_backslashes(resolved);
    size_t len = wcslen(resolved);
    while (len > 3 && resolved[len - 1] == L'\\') resolved[--len] = L'\0';
    if (windows_path_is_protected(resolved)) {
        free(resolved);
        errno = EPERM;
        return -1;
    }
    int result = windows_remove_tree(resolved);
    free(resolved);
    return result;
#else
    /* Safety: refuse dangerous paths */
    char resolved[PATH_MAX];
    if (!realpath(path, resolved)) {
        /* Path doesn't exist, nothing to delete */
        return 0;
    }
    for (const char** p = g_dangerous_paths; *p; p++) {
        if (strcmp(resolved, *p) == 0) {
            errno = EPERM;
            return -1;
        }
    }

    return nftw(resolved, rmdir_recursive_cb, 64, FTW_DEPTH | FTW_PHYS);
#endif
}

/*******************************************************************************
 * B.1: File Stat (structured)
 ******************************************************************************/

/**
 * @brief Retrieves lstat() metadata for @p path into individually-typed output parameters.
 *
 * Uses lstat() (not stat()), so symlinks are reported as themselves rather
 * than being followed.
 *
 * @param out_size Set to the file size in bytes, if non-NULL.
 * @param out_mtime Set to the modification time (epoch seconds), if non-NULL.
 * @param out_ctime Set to the status-change time (epoch seconds), if non-NULL.
 * @param out_mode Set to the raw st_mode bits, if non-NULL.
 * @param out_type Set to a simplified file-type code, if non-NULL:
 *        0 = regular file, 1 = directory, 2 = symlink, 3 = other.
 * @return 0 on success, -1 if @p path is NULL or lstat() fails.
 */
int32_t eshkol_file_stat_fields(const char* path,
                                  int64_t* out_size, int64_t* out_mtime,
                                  int64_t* out_ctime, int32_t* out_mode,
                                  int32_t* out_type) {
    if (!path) return -1;
#ifdef _WIN32
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    WIN32_FILE_ATTRIBUTE_DATA data;
    BOOL ok = GetFileAttributesExW(wide, GetFileExInfoStandard, &data);
    free(wide);
    if (!ok) return -1;
    ULARGE_INTEGER size, modified, created;
    size.LowPart = data.nFileSizeLow; size.HighPart = data.nFileSizeHigh;
    modified.LowPart = data.ftLastWriteTime.dwLowDateTime;
    modified.HighPart = data.ftLastWriteTime.dwHighDateTime;
    created.LowPart = data.ftCreationTime.dwLowDateTime;
    created.HighPart = data.ftCreationTime.dwHighDateTime;
    if (out_size) *out_size = (int64_t)size.QuadPart;
    if (out_mtime) *out_mtime = (int64_t)((modified.QuadPart - 116444736000000000ULL) / 10000000ULL);
    if (out_ctime) *out_ctime = (int64_t)((created.QuadPart - 116444736000000000ULL) / 10000000ULL);
    if (out_mode) {
        int32_t mode = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? _S_IFDIR : _S_IFREG;
        mode |= _S_IREAD;
        if (!(data.dwFileAttributes & FILE_ATTRIBUTE_READONLY)) mode |= _S_IWRITE;
        *out_mode = mode;
    }
    if (out_type) {
        if (data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) *out_type = 2;
        else if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) *out_type = 1;
        else *out_type = 0;
    }
    return 0;
#else
    struct stat st;
    if (lstat(path, &st) != 0) return -1;

    if (out_size)  *out_size  = (int64_t)st.st_size;
    if (out_mtime) *out_mtime = (int64_t)st.st_mtime;
    if (out_ctime) *out_ctime = (int64_t)st.st_ctime;
    if (out_mode)  *out_mode  = (int32_t)st.st_mode;

    if (out_type) {
        if (S_ISREG(st.st_mode))       *out_type = 0;  /* file */
        else if (S_ISDIR(st.st_mode))  *out_type = 1;  /* directory */
        else if (S_ISLNK(st.st_mode))  *out_type = 2;  /* symlink */
        else                           *out_type = 3;  /* other */
    }
    return 0;
#endif
}

/*******************************************************************************
 * B.1: File Copy
 ******************************************************************************/

/**
 * @brief Copies the file at @p src to @p dst.
 *
 * On macOS, first attempts a copy-on-write clone via copyfile()
 * (COPYFILE_ALL | COPYFILE_CLONE) for an instant, no-I/O copy. If that is
 * unavailable or fails (or on non-Apple platforms), falls back to a
 * chunked read/write loop, creating/truncating @p dst with mode 0644.
 *
 * @return 0 on success, -1 if either file cannot be opened or an I/O error occurs.
 */
int32_t eshkol_file_copy(const char* src, const char* dst) {
    if (!src || !dst) return -1;

#ifdef _WIN32
    wchar_t* source = platform_utf8_to_wide(src);
    wchar_t* destination = platform_utf8_to_wide(dst);
    if (!source || !destination) { free(source); free(destination); return -1; }
    BOOL ok = CopyFileW(source, destination, FALSE);
    free(source); free(destination);
    return ok ? 0 : -1;
#else
#ifdef __APPLE__
    /* Try CoW clone first (instant, no I/O) */
    if (copyfile(src, dst, NULL, COPYFILE_ALL | COPYFILE_CLONE) == 0) return 0;
#endif

    /* Fallback: read/write in chunks */
    int in_fd = open(src, O_RDONLY);
    if (in_fd < 0) return -1;

    int out_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) { close(in_fd); return -1; }

    char chunk[65536];
    ssize_t n;
    int result = 0;
    while ((n = read(in_fd, chunk, sizeof(chunk))) > 0) {
        ssize_t written = 0;
        while (written < n) {
            ssize_t w = write(out_fd, chunk + written, (size_t)(n - written));
            if (w < 0) { result = -1; goto done; }
            written += w;
        }
    }
    if (n < 0) result = -1;

done:
    close(in_fd);
    close(out_fd);
    return result;
#endif
}

/*******************************************************************************
 * B.1: File chmod, symlink, realpath, glob-match, file-lock
 ******************************************************************************/

/**
 * @brief Changes the permission bits of @p path to @p mode.
 *
 * @return 0 on success, -1 on failure.
 */
int32_t eshkol_file_chmod(const char* path, int32_t mode) {
#ifdef _WIN32
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    int result = _wchmod(wide, mode);
    free(wide);
    return result == 0 ? 0 : -1;
#else
    return chmod(path, (mode_t)mode) == 0 ? 0 : -1;
#endif
}

/**
 * @brief Creates a symbolic link at @p link_path pointing to @p target.
 *
 * @return 0 on success, -1 on failure.
 */
int32_t eshkol_symlink_create(const char* target, const char* link_path) {
#ifdef _WIN32
    wchar_t* wide_target = platform_utf8_to_wide(target);
    wchar_t* wide_link = platform_utf8_to_wide(link_path);
    if (!wide_target || !wide_link) { free(wide_target); free(wide_link); return -1; }
    DWORD attrs = GetFileAttributesW(wide_target);
    DWORD flags = (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY))
        ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0;
#ifdef SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE
    flags |= SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
#endif
    BOOL ok = CreateSymbolicLinkW(wide_link, wide_target, flags);
    free(wide_target); free(wide_link);
    return ok ? 0 : -1;
#else
    return symlink(target, link_path) == 0 ? 0 : -1;
#endif
}

/**
 * @brief Reads the target of the symbolic link at @p path into @p buf.
 *
 * @param buf Destination buffer for the target path; this function
 *        null-terminates it (readlink() itself does not).
 * @param buf_size Capacity of @p buf.
 * @return Length of the link target written to @p buf, or -1 on failure.
 */
int32_t eshkol_symlink_read(const char* path, char* buf, int32_t buf_size) {
#ifdef _WIN32
    if (!path || !buf || buf_size <= 0) return -1;
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    HANDLE file = CreateFileW(wide, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              NULL, OPEN_EXISTING,
                              FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                              NULL);
    free(wide);
    if (file == INVALID_HANDLE_VALUE) return -1;
    unsigned char raw[MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
    DWORD returned = 0;
    BOOL ok = DeviceIoControl(file, FSCTL_GET_REPARSE_POINT, NULL, 0,
                              raw, sizeof(raw), &returned, NULL);
    CloseHandle(file);
    if (!ok) return -1;
    EshkolReparseDataBuffer* reparse = (EshkolReparseDataBuffer*)raw;
    const wchar_t* target = NULL;
    USHORT bytes = 0;
    if (reparse->ReparseTag == IO_REPARSE_TAG_SYMLINK) {
        target = (const wchar_t*)((const unsigned char*)reparse->SymbolicLinkReparseBuffer.PathBuffer +
            reparse->SymbolicLinkReparseBuffer.PrintNameOffset);
        bytes = reparse->SymbolicLinkReparseBuffer.PrintNameLength;
        if (bytes == 0) {
            target = (const wchar_t*)((const unsigned char*)reparse->SymbolicLinkReparseBuffer.PathBuffer +
                reparse->SymbolicLinkReparseBuffer.SubstituteNameOffset);
            bytes = reparse->SymbolicLinkReparseBuffer.SubstituteNameLength;
        }
    } else if (reparse->ReparseTag == IO_REPARSE_TAG_MOUNT_POINT) {
        target = (const wchar_t*)((const unsigned char*)reparse->MountPointReparseBuffer.PathBuffer +
            reparse->MountPointReparseBuffer.PrintNameOffset);
        bytes = reparse->MountPointReparseBuffer.PrintNameLength;
    }
    if (!target || bytes == 0) return -1;
    size_t chars = bytes / sizeof(wchar_t);
    wchar_t* copy = (wchar_t*)malloc((chars + 1) * sizeof(wchar_t));
    if (!copy) return -1;
    memcpy(copy, target, bytes); copy[chars] = L'\0';
    int32_t result = platform_wide_to_utf8(copy, buf, buf_size);
    free(copy);
    if (result >= 0) normalize_windows_path(buf);
    return result;
#else
    ssize_t n = readlink(path, buf, (size_t)(buf_size - 1));
    if (n < 0) return -1;
    buf[n] = '\0';
    return (int32_t)n;
#endif
}

/**
 * @brief Resolves @p path to an absolute, symlink-free canonical path via realpath().
 *
 * @param buf Destination buffer for the resolved path.
 * @param buf_size Capacity of @p buf.
 * @return Length of the resolved path, or -1 if realpath() fails or @p buf
 *         is too small.
 */
int32_t eshkol_realpath_resolve(const char* path, char* buf, int32_t buf_size) {
#ifdef _WIN32
    if (!path || !buf || buf_size <= 0) return -1;
    wchar_t* input = platform_utf8_to_wide(path);
    if (!input) return -1;
    HANDLE file = CreateFileW(input, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    free(input);
    if (file == INVALID_HANDLE_VALUE) return -1;
    wchar_t* resolved = (wchar_t*)calloc(32768, sizeof(wchar_t));
    if (!resolved) { CloseHandle(file); return -1; }
    DWORD got = GetFinalPathNameByHandleW(file, resolved, 32768, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    CloseHandle(file);
    if (got == 0 || got >= 32768) { free(resolved); return -1; }
    wchar_t* visible = resolved;
    if (wcsncmp(visible, L"\\\\?\\UNC\\", 8) == 0) {
        visible += 6; visible[0] = L'\\';
    } else if (wcsncmp(visible, L"\\\\?\\", 4) == 0) {
        visible += 4;
    }
    int32_t result = platform_wide_to_utf8(visible, buf, buf_size);
    free(resolved);
    if (result >= 0) normalize_windows_path(buf);
    return result;
#else
    char resolved[PATH_MAX];
    if (!realpath(path, resolved)) return -1;
    int len = (int32_t)strlen(resolved);
    if (len >= buf_size) return -1;
    memcpy(buf, resolved, (size_t)len + 1);
    return len;
#endif
}

/**
 * @brief Tests whether @p path matches the shell glob @p pattern.
 *
 * Wraps fnmatch() with FNM_PATHNAME, so wildcards do not match '/'.
 *
 * @return 1 if @p path matches, 0 otherwise.
 */
int32_t eshkol_glob_match(const char* pattern, const char* path) {
#ifdef _WIN32
    wchar_t* wide_pattern = platform_utf8_to_wide(pattern);
    wchar_t* wide_path = platform_utf8_to_wide(path);
    if (!wide_pattern || !wide_path) { free(wide_pattern); free(wide_path); return 0; }
    int result = windows_glob_match_impl(wide_pattern, wide_path);
    free(wide_pattern); free(wide_path);
    return result;
#else
    return fnmatch(pattern, path, FNM_PATHNAME) == 0 ? 1 : 0;
#endif
}

/**
 * @brief Opens (creating if needed) and acquires an exclusive, non-blocking advisory lock on @p path.
 *
 * @return A file descriptor handle to be passed to eshkol_file_unlock(), or
 *         -1 if the file could not be opened or is already locked by
 *         another holder.
 */
int64_t eshkol_file_lock(const char* path) {
#ifdef _WIN32
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    HANDLE file = CreateFileW(wide, GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE,
                              NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    free(wide);
    if (file == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED overlap; memset(&overlap, 0, sizeof(overlap));
    if (!LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                    0, MAXDWORD, MAXDWORD, &overlap)) {
        CloseHandle(file); return -1;
    }
    return (int64_t)(intptr_t)file;
#else
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return -1;
    }
    return (int64_t)fd;
#endif
}

/**
 * @brief Releases a lock previously acquired by eshkol_file_lock() and closes its file descriptor.
 *
 * @param fd The handle returned by eshkol_file_lock().
 * @return 0 on success, -1 if @p fd is negative.
 */
int32_t eshkol_file_unlock(int64_t fd) {
    if (fd < 0) return -1;
#ifdef _WIN32
    HANDLE file = (HANDLE)(intptr_t)fd;
    OVERLAPPED overlap; memset(&overlap, 0, sizeof(overlap));
    BOOL ok = UnlockFileEx(file, 0, MAXDWORD, MAXDWORD, &overlap);
    CloseHandle(file);
    return ok ? 0 : -1;
#else
    flock((int)fd, LOCK_UN);
    close((int)fd);
    return 0;
#endif
}

/*******************************************************************************
 * B.6: UUID v4
 ******************************************************************************/

/**
 * @brief Generates a random RFC 4122 version-4 UUID and writes its canonical string form to @p buf.
 *
 * Sources 16 random bytes from /dev/urandom, falling back to a
 * time/pid-seeded rand() if /dev/urandom cannot be opened (not
 * cryptographically secure in that case). Sets the version and variant
 * bits per RFC 4122 before formatting.
 *
 * @param buf Destination buffer; must be at least 37 bytes
 *        ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" plus null terminator).
 */
void eshkol_uuid_v4(char* buf) {
    unsigned char bytes[16];
#ifdef _WIN32
    if (!buf || !windows_random_bytes(bytes, sizeof(bytes))) {
        if (buf) buf[0] = '\0';
        return;
    }
#else
    /* Read from /dev/urandom */
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        read(fd, bytes, 16);
        close(fd);
    } else {
        /* Fallback: not cryptographically secure but functional */
        srand((unsigned)time(NULL) ^ (unsigned)getpid());
        for (int i = 0; i < 16; i++) bytes[i] = (unsigned char)(rand() & 0xFF);
    }
#endif
    /* Set version (4) and variant (RFC 4122) */
    bytes[6] = (bytes[6] & 0x0F) | 0x40;  /* version 4 */
    bytes[8] = (bytes[8] & 0x3F) | 0x80;  /* variant 10xx */

    snprintf(buf, 37, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0], bytes[1], bytes[2], bytes[3],
             bytes[4], bytes[5], bytes[6], bytes[7],
             bytes[8], bytes[9], bytes[10], bytes[11],
             bytes[12], bytes[13], bytes[14], bytes[15]);
}

/*******************************************************************************
 * B.11: Date/Time Formatting
 ******************************************************************************/

/**
 * @brief Formats @p epoch (Unix seconds, UTC) as an ISO 8601 string ("YYYY-MM-DDTHH:MM:SSZ").
 *
 * @param buf Destination buffer.
 * @param buf_size Capacity of @p buf; must be at least 21.
 * @return Number of characters written (excluding null terminator), or -1
 *         on failure.
 */
int32_t eshkol_format_iso8601(int64_t epoch, char* buf, int32_t buf_size) {
    if (!buf || buf_size < 21) return -1;
    time_t t = (time_t)epoch;
    struct tm tm;
#ifdef _WIN32
    if (gmtime_s(&tm, &t) != 0) return -1;
#else
    gmtime_r(&t, &tm);
#endif
    int n = (int32_t)strftime(buf, (size_t)buf_size, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return n > 0 ? n : -1;
}

/**
 * @brief Parses an ISO 8601 timestamp ("YYYY-MM-DDTHH:MM:SS") into Unix epoch seconds (UTC).
 *
 * @return Epoch seconds, or -1 if @p str is NULL or does not match the
 *         expected format.
 */
int64_t eshkol_parse_iso8601(const char* str) {
    if (!str) return -1;
#ifdef _WIN32
    SYSTEMTIME system_time;
    memset(&system_time, 0, sizeof(system_time));
    char trailing = '\0';
    int year, month, day, hour, minute, second;
    int parsed = sscanf(str, "%d-%d-%dT%d:%d:%d%c",
                        &year, &month, &day, &hour, &minute, &second, &trailing);
    if (parsed < 6 || (parsed == 7 && trailing != 'Z')) return -1;
    if (year < 1601 || year > 30827 || month < 1 || month > 12 ||
        day < 1 || day > 31 || hour < 0 || hour > 23 ||
        minute < 0 || minute > 59 || second < 0 || second > 60) return -1;
    system_time.wYear = (WORD)year; system_time.wMonth = (WORD)month;
    system_time.wDay = (WORD)day; system_time.wHour = (WORD)hour;
    system_time.wMinute = (WORD)minute; system_time.wSecond = (WORD)second;
    FILETIME ft;
    if (!SystemTimeToFileTime(&system_time, &ft)) return -1;
    ULARGE_INTEGER ticks;
    ticks.LowPart = ft.dwLowDateTime; ticks.HighPart = ft.dwHighDateTime;
    if (ticks.QuadPart < 116444736000000000ULL) return -1;
    return (int64_t)((ticks.QuadPart - 116444736000000000ULL) / 10000000ULL);
#else
    struct tm tm;
    memset(&tm, 0, sizeof(tm));
    if (!strptime(str, "%Y-%m-%dT%H:%M:%S", &tm)) return -1;
    return (int64_t)timegm(&tm);
#endif
}

/**
 * @brief Formats @p seconds_ago as a short human-readable relative-time string.
 *
 * Chooses the coarsest unit that keeps the value under 3 digits: seconds
 * ("Ns ago") below 60s, minutes ("Nm ago") below an hour, hours ("Nh ago")
 * below a day, and days ("Nd ago") otherwise.
 *
 * @param buf Destination buffer.
 * @param buf_size Capacity of @p buf; must be at least 16.
 * @return Number of characters written, or -1 on failure.
 */
int32_t eshkol_format_relative(int64_t seconds_ago, char* buf, int32_t buf_size) {
    if (!buf || buf_size < 16) return -1;
    int n;
    if (seconds_ago < 60)
        n = snprintf(buf, (size_t)buf_size, "%llds ago", (long long)seconds_ago);
    else if (seconds_ago < 3600)
        n = snprintf(buf, (size_t)buf_size, "%lldm ago", (long long)(seconds_ago / 60));
    else if (seconds_ago < 86400)
        n = snprintf(buf, (size_t)buf_size, "%lldh ago", (long long)(seconds_ago / 3600));
    else
        n = snprintf(buf, (size_t)buf_size, "%lldd ago", (long long)(seconds_ago / 86400));
    return n > 0 ? n : -1;
}

/**
 * @brief Returns the local timezone's current UTC offset, in seconds.
 *
 * Computed as the difference between the local and UTC broken-down
 * representations of the current time, so it reflects DST if currently
 * in effect.
 */
int64_t eshkol_local_timezone_offset(void) {
#ifdef _WIN32
    TIME_ZONE_INFORMATION zone;
    DWORD state = GetTimeZoneInformation(&zone);
    LONG bias = zone.Bias;
    if (state == TIME_ZONE_ID_STANDARD) bias += zone.StandardBias;
    else if (state == TIME_ZONE_ID_DAYLIGHT) bias += zone.DaylightBias;
    return -(int64_t)bias * 60;
#else
    time_t t = time(NULL);
    struct tm local, utc;
    localtime_r(&t, &local);
    gmtime_r(&t, &utc);
    return (int64_t)(mktime(&local) - mktime(&utc));
#endif
}

/*******************************************************************************
 * B.1: file-mmap / file-munmap
 ******************************************************************************/

#ifndef _WIN32
#include <sys/mman.h>
#endif

/* mmap handle table */
#define MAX_MMAPS 16
static struct {
    void* ptr;
    size_t len;
#ifdef _WIN32
    void* base;
    size_t map_len;
    HANDLE mapping;
#endif
} g_mmaps[MAX_MMAPS] = {{0}};

/**
 * @brief Memory-maps a read-only region of the file at @p path and registers it in the internal mmap table.
 *
 * If @p length <= 0, maps from @p offset to the end of the file. The
 * mapping is tracked in a small fixed-size table (@ref g_mmaps) so it can
 * be read via eshkol_mmap_read() and released via eshkol_file_munmap()
 * using an integer handle instead of a raw pointer.
 *
 * @return A handle (index into @ref g_mmaps) for use with
 *         eshkol_mmap_read()/eshkol_mmap_length()/eshkol_file_munmap(), or
 *         -1 on failure (including if the mmap table is full, in which case
 *         the mapping is unmapped before returning).
 */
int64_t eshkol_file_mmap(const char* path, int64_t offset, int64_t length) {
    if (!path) return -1;
#ifdef _WIN32
    if (offset < 0) return -1;
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    HANDLE file = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(wide);
    if (file == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || offset >= file_size.QuadPart) {
        CloseHandle(file); return -1;
    }
    if (length <= 0 || offset + length > file_size.QuadPart) {
        length = file_size.QuadPart - offset;
    }
    SYSTEM_INFO info; GetSystemInfo(&info);
    uint64_t granularity = info.dwAllocationGranularity;
    uint64_t aligned = (uint64_t)offset - ((uint64_t)offset % granularity);
    size_t delta = (size_t)((uint64_t)offset - aligned);
    if ((uint64_t)length + delta > SIZE_MAX) { CloseHandle(file); return -1; }
    size_t map_length = (size_t)length + delta;
    HANDLE mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(file);
    if (!mapping) return -1;
    void* base = MapViewOfFile(mapping, FILE_MAP_READ,
                               (DWORD)(aligned >> 32), (DWORD)aligned, map_length);
    if (!base) { CloseHandle(mapping); return -1; }
    for (int i = 0; i < MAX_MMAPS; ++i) {
        if (!g_mmaps[i].ptr) {
            g_mmaps[i].base = base;
            g_mmaps[i].ptr = (unsigned char*)base + delta;
            g_mmaps[i].len = (size_t)length;
            g_mmaps[i].map_len = map_length;
            g_mmaps[i].mapping = mapping;
            return i;
        }
    }
    UnmapViewOfFile(base); CloseHandle(mapping);
    return -1;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    /* If length is 0, map the whole file */
    if (length <= 0) {
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); return -1; }
        length = st.st_size - offset;
        if (length <= 0) { close(fd); return -1; }
    }

    void* ptr = mmap(NULL, (size_t)length, PROT_READ, MAP_PRIVATE, fd, (off_t)offset);
    close(fd);
    if (ptr == MAP_FAILED) return -1;

    for (int i = 0; i < MAX_MMAPS; i++) {
        if (!g_mmaps[i].ptr) {
            g_mmaps[i].ptr = ptr;
            g_mmaps[i].len = (size_t)length;
            return (int64_t)i;
        }
    }
    munmap(ptr, (size_t)length);
    return -1;
#endif
}

/**
 * @brief Unmaps and releases the mmap handle previously returned by eshkol_file_mmap().
 *
 * @return 0 on success, -1 if @p handle is out of range or not currently mapped.
 */
int32_t eshkol_file_munmap(int64_t handle) {
    if (handle < 0 || handle >= MAX_MMAPS || !g_mmaps[handle].ptr) return -1;
#ifdef _WIN32
    BOOL ok = UnmapViewOfFile(g_mmaps[handle].base);
    CloseHandle(g_mmaps[handle].mapping);
    g_mmaps[handle].base = NULL;
    g_mmaps[handle].map_len = 0;
    g_mmaps[handle].mapping = NULL;
#else
    munmap(g_mmaps[handle].ptr, g_mmaps[handle].len);
#endif
    g_mmaps[handle].ptr = NULL;
    g_mmaps[handle].len = 0;
    return
#ifdef _WIN32
        ok ? 0 : -1;
#else
        0;
#endif
}

/**
 * @brief Copies up to @p buf_size bytes starting at @p offset from a memory-mapped region into @p buf.
 *
 * @param handle A handle returned by eshkol_file_mmap().
 * @param offset Byte offset into the mapped region to start reading from.
 * @param buf Destination buffer (not null-terminated).
 * @param buf_size Capacity of @p buf.
 * @return Number of bytes copied (may be less than @p buf_size if fewer
 *         bytes remain in the mapping), or -1 if @p handle is invalid or
 *         @p offset is out of range.
 */
/* Read bytes from mmap'd region */
int32_t eshkol_mmap_read(int64_t handle, int64_t offset, char* buf, int32_t buf_size) {
    if (handle < 0 || handle >= MAX_MMAPS || !g_mmaps[handle].ptr) return -1;
    if (!buf || buf_size <= 0) return -1;
    size_t avail = g_mmaps[handle].len - (size_t)offset;
    if (offset < 0 || (size_t)offset >= g_mmaps[handle].len) return -1;
    int32_t to_read = (int32_t)(avail < (size_t)buf_size ? avail : (size_t)buf_size);
    memcpy(buf, (char*)g_mmaps[handle].ptr + offset, (size_t)to_read);
    return to_read;
}

/**
 * @brief Returns the length in bytes of the memory-mapped region for @p handle.
 *
 * @return The mapping length, or -1 if @p handle is invalid or not currently mapped.
 */
int64_t eshkol_mmap_length(int64_t handle) {
    if (handle < 0 || handle >= MAX_MMAPS || !g_mmaps[handle].ptr) return -1;
    return (int64_t)g_mmaps[handle].len;
}

/*******************************************************************************
 * B.1: directory-walk — recursive directory traversal
 ******************************************************************************/

/**
 * @brief Recursive worker for eshkol_directory_walk() that lists @p base and its subdirectories.
 *
 * Emits one "type:path" NUL-terminated entry per directory item into @p buf
 * (type is 'f' file / 'd' directory / 'l' symlink), incrementing *@p count
 * for each, and recurses into subdirectories while @p depth stays within
 * @p max_depth (unlimited if @p max_depth < 0).
 *
 * @param written In/out cursor tracking bytes written so far into @p buf.
 * @param count In/out running count of entries written.
 * @return 0 on success (including when @p base cannot be opened, which is
 *         silently skipped), or -1 if @p buf runs out of space.
 */
#ifndef _WIN32
static int32_t directory_walk_impl(const char* base, int depth, int max_depth,
                                     char* buf, int32_t buf_size,
                                     int32_t* written, int32_t* count) {
    if (depth > max_depth && max_depth >= 0) return 0;

    DIR* dir = opendir(base);
    if (!dir) return 0;

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;

        char full_path[PATH_MAX];
        int plen = snprintf(full_path, sizeof(full_path), "%s/%s", base, ent->d_name);
        if (plen < 0 || plen >= (int)sizeof(full_path)) continue;

        struct stat st;
        if (lstat(full_path, &st) != 0) continue;

        char type_ch = 'f';
        if (S_ISDIR(st.st_mode)) type_ch = 'd';
        else if (S_ISLNK(st.st_mode)) type_ch = 'l';

        /* Write "type:path\0" to buffer */
        if (*count > 0 && *written < buf_size - 1) {
            buf[*written] = '\0';
            (*written)++;
        }
        int n = snprintf(buf + *written, (size_t)(buf_size - *written),
                         "%c:%s", type_ch, full_path);
        if (n < 0 || *written + n >= buf_size) { closedir(dir); return -1; }
        *written += n;
        (*count)++;

        /* Recurse into directories */
        if (type_ch == 'd') {
            directory_walk_impl(full_path, depth + 1, max_depth,
                               buf, buf_size, written, count);
        }
    }
    closedir(dir);
    return 0;
}
#else
static int32_t directory_walk_impl_windows(const wchar_t* base, int depth,
                                           int max_depth, char* buf,
                                           int32_t buf_size, int32_t* written,
                                           int32_t* count) {
    if (max_depth >= 0 && depth > max_depth) return 0;
    size_t base_len = wcslen(base);
    wchar_t* pattern = (wchar_t*)malloc((base_len + 3) * sizeof(wchar_t));
    if (!pattern) return -1;
    swprintf(pattern, base_len + 3, L"%ls\\*", base);
    WIN32_FIND_DATAW data;
    HANDLE find = FindFirstFileW(pattern, &data);
    free(pattern);
    if (find == INVALID_HANDLE_VALUE) return 0;
    int32_t result = 0;
    do {
        if (wcscmp(data.cFileName, L".") == 0 || wcscmp(data.cFileName, L"..") == 0) continue;
        size_t child_len = base_len + wcslen(data.cFileName) + 2;
        wchar_t* child = (wchar_t*)malloc(child_len * sizeof(wchar_t));
        if (!child) { result = -1; break; }
        swprintf(child, child_len, L"%ls\\%ls", base, data.cFileName);
        int utf8_len = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                           child, -1, NULL, 0, NULL, NULL);
        char* utf8 = utf8_len > 0 ? (char*)malloc((size_t)utf8_len) : NULL;
        if (!utf8 || WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                         child, -1, utf8, utf8_len, NULL, NULL) <= 0) {
            free(utf8); free(child); result = -1; break;
        }
        normalize_windows_path(utf8);
        char type = (data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ? 'l' :
                    ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 'd' : 'f');
        if (*count > 0) {
            if (*written >= buf_size - 1) { free(utf8); free(child); result = -1; break; }
            buf[(*written)++] = '\0';
        }
        int n = snprintf(buf + *written, (size_t)(buf_size - *written),
                         "%c:%s", type, utf8);
        free(utf8);
        if (n < 0 || *written + n >= buf_size) { free(child); result = -1; break; }
        *written += n;
        ++*count;
        if (type == 'd' && directory_walk_impl_windows(
                child, depth + 1, max_depth, buf, buf_size, written, count) != 0) {
            free(child); result = -1; break;
        }
        free(child);
    } while (FindNextFileW(find, &data));
    FindClose(find);
    return result;
}
#endif

/**
 * @brief Recursively lists the contents of @p path up to @p max_depth levels deep.
 *
 * Writes NUL-separated "type:path" entries (see directory_walk_impl()) into
 * @p buf and reports the number of entries in @p count.
 *
 * @param max_depth Maximum recursion depth, or a negative value for
 *        unlimited depth.
 * @param buf Destination buffer for the NUL-separated entry list.
 * @param buf_size Capacity of @p buf.
 * @param count Set to the number of entries written.
 * @return Number of bytes written to @p buf, or -1 on invalid arguments.
 */
int32_t eshkol_directory_walk(const char* path, int32_t max_depth,
                                char* buf, int32_t buf_size, int32_t* count) {
    if (!path || !buf || buf_size <= 0 || !count) return -1;
    int32_t written = 0;
    *count = 0;
#ifdef _WIN32
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    windows_slashes_to_backslashes(wide);
    int32_t result = directory_walk_impl_windows(wide, 0, max_depth,
                                                 buf, buf_size, &written, count);
    free(wide);
    if (result != 0) return -1;
#else
    directory_walk_impl(path, 0, max_depth, buf, buf_size, &written, count);
#endif
    if (written < buf_size) buf[written] = '\0';
    return written;
}

/*******************************************************************************
 * B.1: glob-expand — walk + fnmatch
 ******************************************************************************/

/**
 * @brief Recursive worker for eshkol_glob_expand() that matches @p pattern against entries under @p dir_path.
 *
 * Compares each directory entry's name (not full path) against @p pattern
 * with fnmatch(), appending full matching paths as NUL-separated entries
 * into @p buf, and recurses into every subdirectory regardless of whether
 * it matched.
 *
 * @param written In/out cursor tracking bytes written so far into @p buf.
 * @param count In/out running count of matches written.
 */
#ifndef _WIN32
static void glob_expand_impl(const char* dir_path, const char* pattern,
                               char* buf, int32_t buf_size,
                               int32_t* written, int32_t* count) {
    DIR* dir = opendir(dir_path);
    if (!dir) return;

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;

        char full_path[PATH_MAX];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, ent->d_name);

        struct stat st;
        if (lstat(full_path, &st) != 0) continue;

        /* Check if name matches pattern */
        if (fnmatch(pattern, ent->d_name, 0) == 0) {
            if (*count > 0 && *written < buf_size - 1) {
                buf[*written] = '\0';
                (*written)++;
            }
            int n = snprintf(buf + *written, (size_t)(buf_size - *written), "%s", full_path);
            if (n >= 0 && *written + n < buf_size) {
                *written += n;
                (*count)++;
            }
        }

        /* Recurse into subdirectories */
        if (S_ISDIR(st.st_mode)) {
            glob_expand_impl(full_path, pattern, buf, buf_size, written, count);
        }
    }
    closedir(dir);
}
#else
static int glob_expand_impl_windows(const wchar_t* directory,
                                    const wchar_t* pattern, char* buf,
                                    int32_t buf_size, int32_t* written,
                                    int32_t* count) {
    size_t dir_len = wcslen(directory);
    wchar_t* query = (wchar_t*)malloc((dir_len + 3) * sizeof(wchar_t));
    if (!query) return -1;
    swprintf(query, dir_len + 3, L"%ls\\*", directory);
    WIN32_FIND_DATAW data;
    HANDLE find = FindFirstFileW(query, &data);
    free(query);
    if (find == INVALID_HANDLE_VALUE) return 0;
    int result = 0;
    do {
        if (wcscmp(data.cFileName, L".") == 0 || wcscmp(data.cFileName, L"..") == 0) continue;
        size_t child_len = dir_len + wcslen(data.cFileName) + 2;
        wchar_t* child = (wchar_t*)malloc(child_len * sizeof(wchar_t));
        if (!child) { result = -1; break; }
        swprintf(child, child_len, L"%ls\\%ls", directory, data.cFileName);
        if (windows_glob_match_impl(pattern, data.cFileName)) {
            int utf8_len = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                               child, -1, NULL, 0, NULL, NULL);
            char* utf8 = utf8_len > 0 ? (char*)malloc((size_t)utf8_len) : NULL;
            if (!utf8 || WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                             child, -1, utf8, utf8_len, NULL, NULL) <= 0) {
                free(utf8); free(child); result = -1; break;
            }
            normalize_windows_path(utf8);
            if (*count > 0) {
                if (*written >= buf_size - 1) { free(utf8); free(child); result = -1; break; }
                buf[(*written)++] = '\0';
            }
            int n = snprintf(buf + *written, (size_t)(buf_size - *written), "%s", utf8);
            free(utf8);
            if (n < 0 || *written + n >= buf_size) { free(child); result = -1; break; }
            *written += n;
            ++*count;
        }
        if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
            !(data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) &&
            glob_expand_impl_windows(child, pattern, buf, buf_size, written, count) != 0) {
            free(child); result = -1; break;
        }
        free(child);
    } while (FindNextFileW(find, &data));
    FindClose(find);
    return result;
}
#endif

/**
 * @brief Recursively searches under @p root for entries whose name matches @p pattern.
 *
 * Writes NUL-separated matching full paths (see glob_expand_impl()) into
 * @p buf and reports the number of matches in @p count.
 *
 * @param buf Destination buffer for the NUL-separated match list.
 * @param buf_size Capacity of @p buf.
 * @param count Set to the number of matches written.
 * @return Number of bytes written to @p buf, or -1 on invalid arguments.
 */
int32_t eshkol_glob_expand(const char* pattern, const char* root,
                             char* buf, int32_t buf_size, int32_t* count) {
    if (!pattern || !root || !buf || buf_size <= 0 || !count) return -1;
    int32_t written = 0;
    *count = 0;
#ifdef _WIN32
    wchar_t* wide_pattern = platform_utf8_to_wide(pattern);
    wchar_t* wide_root = platform_utf8_to_wide(root);
    if (!wide_pattern || !wide_root) { free(wide_pattern); free(wide_root); return -1; }
    windows_slashes_to_backslashes(wide_root);
    int result = glob_expand_impl_windows(wide_root, wide_pattern, buf,
                                          buf_size, &written, count);
    free(wide_pattern); free(wide_root);
    if (result != 0) return -1;
#else
    glob_expand_impl(root, pattern, buf, buf_size, &written, count);
#endif
    if (written < buf_size) buf[written] = '\0';
    return written;
}

/*******************************************************************************
 * B.22: shell-split — parse shell command into argv
 ******************************************************************************/

/**
 * @brief Splits @p cmd into shell-style arguments, honoring single/double quoting and backslash escapes.
 *
 * Writes each argument as a NUL-terminated string, one after another, into
 * @p buf, and reports the number of arguments in @p argc. Within double
 * quotes, backslash escapes the next character; outside quotes, backslash
 * escapes the next character and single/double quote characters start a
 * quoted region. Unquoted whitespace (space/tab) separates arguments.
 *
 * @param buf Destination buffer for the concatenated NUL-separated arguments.
 * @param buf_size Capacity of @p buf.
 * @param argc Set to the number of arguments parsed.
 * @return Number of bytes written to @p buf, or -1 on invalid arguments.
 */
int32_t eshkol_shell_split(const char* cmd, char* buf, int32_t buf_size, int32_t* argc) {
    if (!cmd || !buf || buf_size <= 0 || !argc) return -1;
    *argc = 0;
    int32_t out = 0;
    const char* p = cmd;

    while (*p) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;

        /* Parse one argument */
        char quote = 0;
        if (*p == '\'' || *p == '"') { quote = *p; p++; }

        if (*argc > 0 && out < buf_size - 1) buf[out++] = '\0';

        while (*p) {
            if (quote) {
                if (*p == quote) { p++; quote = 0; break; }
                if (*p == '\\' && quote == '"' && *(p+1)) {
                    p++;
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                } else {
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                }
            } else {
                if (*p == ' ' || *p == '\t') break;
                if (*p == '\\' && *(p+1)) {
                    p++;
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                } else if (*p == '\'' || *p == '"') {
                    quote = *p; p++;
                } else {
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                }
            }
        }
        (*argc)++;
    }

    if (out < buf_size) buf[out] = '\0';
    return out;
}

/*******************************************************************************
 * B.8: string-display-width — Unicode-aware terminal column width
 *
 * Uses East Asian Width (UAX #11) classification:
 * - CJK ideographs (U+4E00-U+9FFF, U+3400-U+4DBF, etc.): 2 columns
 * - Fullwidth forms (U+FF01-U+FF60, U+FFE0-U+FFE6): 2 columns
 * - Emoji (U+1F300-U+1F9FF, etc.): 2 columns
 * - Zero-width marks (U+0300-U+036F, U+200B-U+200F, etc.): 0 columns
 * - ANSI escape sequences: 0 columns
 * - Everything else: 1 column
 ******************************************************************************/

/**
 * @brief Determines whether Unicode codepoint @p cp occupies two terminal columns.
 *
 * Classifies @p cp as double-width per East Asian Width conventions: CJK
 * ideographs, Hangul syllables, fullwidth forms, Hiragana/Katakana, and
 * emoji ranges.
 *
 * @return 1 if @p cp is double-width, 0 otherwise.
 */
static int is_wide_char(uint32_t cp) {
    /* CJK Unified Ideographs */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1;
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 1;
    /* CJK Compatibility Ideographs */
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1;
    /* Hangul Syllables */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1;
    /* Fullwidth Forms */
    if (cp >= 0xFF01 && cp <= 0xFF60) return 1;
    if (cp >= 0xFFE0 && cp <= 0xFFE6) return 1;
    /* CJK Symbols */
    if (cp >= 0x2E80 && cp <= 0x303E) return 1;
    /* Katakana/Hiragana */
    if (cp >= 0x3040 && cp <= 0x30FF) return 1;
    if (cp >= 0x31F0 && cp <= 0x31FF) return 1;
    /* Emoji */
    if (cp >= 0x1F300 && cp <= 0x1F9FF) return 1;
    if (cp >= 0x1FA00 && cp <= 0x1FA6F) return 1;
    if (cp >= 0x1FA70 && cp <= 0x1FAFF) return 1;
    if (cp >= 0x2600 && cp <= 0x27BF) return 1;
    return 0;
}

/**
 * @brief Determines whether Unicode codepoint @p cp occupies zero terminal columns.
 *
 * Classifies @p cp as zero-width if it is a combining mark, zero-width
 * space/joiner/non-joiner, BOM, or variation selector.
 *
 * @return 1 if @p cp is zero-width, 0 otherwise.
 */
static int is_zero_width(uint32_t cp) {
    /* Combining marks */
    if (cp >= 0x0300 && cp <= 0x036F) return 1;
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return 1;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return 1;
    if (cp >= 0x20D0 && cp <= 0x20FF) return 1;
    if (cp >= 0xFE20 && cp <= 0xFE2F) return 1;
    /* Zero-width characters */
    if (cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0x200E || cp == 0x200F) return 1;
    if (cp == 0xFEFF) return 1;  /* BOM / ZWNBSP */
    /* Variation selectors */
    if (cp >= 0xFE00 && cp <= 0xFE0F) return 1;
    if (cp >= 0xE0100 && cp <= 0xE01EF) return 1;
    return 0;
}

/**
 * @brief Decodes one UTF-8 codepoint from @p str starting at *@p pos, advancing *@p pos past it.
 *
 * Determines the sequence length from the leading byte and folds in
 * continuation bytes; invalid leading bytes (0x80-0xBF) decode to the
 * replacement character U+FFFD as a single byte.
 *
 * @param len Total length of @p str, used to avoid reading continuation
 *        bytes past the end of the string.
 * @param pos In/out byte offset; advanced by the number of bytes consumed.
 * @return The decoded codepoint.
 */
/* Decode one UTF-8 codepoint, advance *pos */
static uint32_t decode_utf8(const char* str, int len, int* pos) {
    unsigned char c = (unsigned char)str[*pos];
    uint32_t cp;
    int bytes;
    if (c < 0x80) { cp = c; bytes = 1; }
    else if (c < 0xC0) { cp = 0xFFFD; bytes = 1; }
    else if (c < 0xE0) { cp = c & 0x1F; bytes = 2; }
    else if (c < 0xF0) { cp = c & 0x0F; bytes = 3; }
    else { cp = c & 0x07; bytes = 4; }
    for (int i = 1; i < bytes && (*pos + i) < len; i++) {
        cp = (cp << 6) | ((unsigned char)str[*pos + i] & 0x3F);
    }
    *pos += bytes;
    return cp;
}

/**
 * @brief Computes the terminal column width of @p str, accounting for wide/zero-width Unicode and ANSI escapes.
 *
 * Skips over ANSI CSI ("ESC[...") and OSC ("ESC]...BEL"/"ESC]...ST")
 * escape sequences (contributing 0 columns), decodes the remaining text as
 * UTF-8, and sums 0/1/2 columns per codepoint via is_zero_width() /
 * is_wide_char().
 *
 * @return The total display width in columns, or 0 if @p str is NULL.
 */
int32_t eshkol_string_display_width(const char* str) {
    if (!str) return 0;
    int len = (int)strlen(str);
    int width = 0;
    int pos = 0;
    int in_escape = 0;
    int escape_type = 0;

    while (pos < len) {
        unsigned char c = (unsigned char)str[pos];

        /* ANSI escape handling */
        if (!in_escape && c == 0x1B) {
            if (pos + 1 < len) {
                unsigned char next = (unsigned char)str[pos + 1];
                if (next == '[') { in_escape = 1; escape_type = 1; pos += 2; continue; }
                if (next == ']') { in_escape = 1; escape_type = 2; pos += 2; continue; }
                if (next >= 0x40 && next <= 0x7E) { pos += 2; continue; }
            }
            pos++;
            continue;
        }
        if (in_escape) {
            if (escape_type == 1) {
                /* CSI: consume until 0x40-0x7E */
                if (c >= 0x40 && c <= 0x7E) in_escape = 0;
                pos++;
            } else {
                /* OSC: consume until BEL or ST */
                if (c == 0x07) { in_escape = 0; pos++; }
                else if (c == 0x1B && pos + 1 < len && str[pos+1] == '\\') {
                    in_escape = 0; pos += 2;
                } else { pos++; }
            }
            continue;
        }

        /* Decode UTF-8 codepoint */
        uint32_t cp = decode_utf8(str, len, &pos);
        if (is_zero_width(cp)) continue;
        width += is_wide_char(cp) ? 2 : 1;
    }
    return width;
}

/**
 * @brief Truncates @p str to at most @p max_width display columns, appending @p suffix if truncation occurs.
 *
 * Walks @p str codepoint-by-codepoint (skipping ANSI escapes, as in
 * eshkol_string_display_width()) accumulating display width until adding
 * the next codepoint would exceed the truncation target (@p max_width minus
 * the display width of @p suffix). If the full string already fits within
 * @p max_width, it is copied unchanged; otherwise the safely-truncated
 * prefix is copied followed by @p suffix.
 *
 * @param suffix Optional string appended when truncation occurs (may be
 *        NULL for none).
 * @param buf Destination buffer.
 * @param buf_size Capacity of @p buf.
 * @return Number of bytes written to @p buf, or -1 on invalid arguments.
 */
int32_t eshkol_string_truncate_display(const char* str, int32_t max_width,
                                         const char* suffix,
                                         char* buf, int32_t buf_size) {
    if (!str || !buf || buf_size <= 0 || max_width <= 0) return -1;
    int suffix_width = suffix ? eshkol_string_display_width(suffix) : 0;
    int target = max_width - suffix_width;
    if (target <= 0) target = max_width;

    int len = (int)strlen(str);
    int width = 0;
    int pos = 0;
    int last_safe = 0;
    int in_escape = 0, escape_type = 0;

    while (pos < len && width < target) {
        unsigned char c = (unsigned char)str[pos];
        if (!in_escape && c == 0x1B) {
            if (pos + 1 < len && str[pos+1] == '[') { in_escape = 1; escape_type = 1; pos += 2; continue; }
            if (pos + 1 < len && str[pos+1] == ']') { in_escape = 1; escape_type = 2; pos += 2; continue; }
            pos++;
            continue;
        }
        if (in_escape) {
            if (escape_type == 1 && c >= 0x40 && c <= 0x7E) in_escape = 0;
            else if (escape_type == 2 && (c == 0x07 || (c == 0x1B && pos+1 < len && str[pos+1] == '\\'))) {
                in_escape = 0;
                if (c == 0x1B) pos++;
            }
            pos++;
            continue;
        }
        int old_pos = pos;
        uint32_t cp = decode_utf8(str, len, &pos);
        int cw = is_zero_width(cp) ? 0 : (is_wide_char(cp) ? 2 : 1);
        if (width + cw > target) { pos = old_pos; break; }
        width += cw;
        last_safe = pos;
    }

    int full_width = eshkol_string_display_width(str);
    if (full_width <= max_width) {
        /* No truncation needed */
        int slen = (int)strlen(str);
        if (slen >= buf_size) slen = buf_size - 1;
        memcpy(buf, str, (size_t)slen);
        buf[slen] = '\0';
        return slen;
    }

    /* Truncate + add suffix */
    int out = 0;
    if (last_safe > 0 && last_safe < buf_size - 1) {
        memcpy(buf, str, (size_t)last_safe);
        out = last_safe;
    }
    if (suffix && suffix_width > 0) {
        int slen = (int)strlen(suffix);
        if (out + slen < buf_size) {
            memcpy(buf + out, suffix, (size_t)slen);
            out += slen;
        }
    }
    buf[out] = '\0';
    return out;
}

/*******************************************************************************
 * B.6: base64url-encode / decode (URL-safe, no padding)
 ******************************************************************************/

static const char b64url_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/**
 * @brief Encodes @p data as URL-safe base64 (RFC 4648 §5), with no padding.
 *
 * @param data Input bytes (need not be null-terminated).
 * @param data_len Number of input bytes.
 * @param buf Destination buffer for the encoded, null-terminated string.
 * @param buf_size Capacity of @p buf.
 * @return Number of characters written to @p buf (excluding null
 *         terminator), or -1 on invalid arguments. Encoding stops early
 *         (producing a truncated result) if @p buf_size is insufficient
 *         for the full output.
 */
int32_t eshkol_base64url_encode(const char* data, int32_t data_len,
                                  char* buf, int32_t buf_size) {
    if (!data || !buf || data_len < 0 || buf_size <= 0) return -1;
    int j = 0;
    int i;
    for (i = 0; i + 2 < data_len && j + 4 < buf_size; i += 3) {
        unsigned int n = ((unsigned char)data[i] << 16) |
                         ((unsigned char)data[i+1] << 8) |
                          (unsigned char)data[i+2];
        buf[j++] = b64url_table[(n >> 18) & 63];
        buf[j++] = b64url_table[(n >> 12) & 63];
        buf[j++] = b64url_table[(n >> 6) & 63];
        buf[j++] = b64url_table[n & 63];
    }
    if (i < data_len && j + 3 < buf_size) {
        unsigned int n = (unsigned char)data[i] << 16;
        if (i + 1 < data_len) n |= (unsigned char)data[i+1] << 8;
        buf[j++] = b64url_table[(n >> 18) & 63];
        buf[j++] = b64url_table[(n >> 12) & 63];
        if (i + 1 < data_len) buf[j++] = b64url_table[(n >> 6) & 63];
        /* No padding in URL-safe base64 */
    }
    buf[j] = '\0';
    return j;
}

/**
 * @brief Maps a single URL-safe base64 character to its 6-bit value.
 *
 * @return The 6-bit value (0-63) for a valid base64url character, or -1 if
 *         @p c is not part of the alphabet.
 */
static int b64url_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '-') return 62;
    if (c == '_') return 63;
    return -1;
}

/**
 * @brief Decodes a URL-safe base64 (RFC 4648 §5) string back into raw bytes.
 *
 * Characters outside the base64url alphabet (as determined by
 * b64url_val()) are silently skipped rather than treated as errors.
 *
 * @param data Input base64url characters (need not be null-terminated).
 * @param data_len Number of input characters to consider.
 * @param buf Destination buffer for the decoded bytes (null-terminated for
 *        convenience, though the data may contain embedded NULs).
 * @param buf_size Capacity of @p buf.
 * @return Number of bytes written to @p buf, or -1 on invalid arguments.
 */
int32_t eshkol_base64url_decode(const char* data, int32_t data_len,
                                  char* buf, int32_t buf_size) {
    if (!data || !buf || data_len < 0 || buf_size <= 0) return -1;
    int out = 0, val = 0, bits = 0;
    for (int i = 0; i < data_len && out < buf_size - 1; i++) {
        int v = b64url_val(data[i]);
        if (v < 0) continue;
        val = (val << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf[out++] = (char)((val >> bits) & 0xFF);
        }
    }
    buf[out] = '\0';
    return out;
}

/*******************************************************************************
 * B.6: constant-time-equal
 ******************************************************************************/

/**
 * @brief Compares two byte buffers for equality in constant time, to avoid timing side-channels.
 *
 * Intended for comparing secrets (e.g. tokens, MACs) where early-exit
 * comparison could leak information via timing. Always scans the full
 * length when sizes match; a length mismatch is detected (and returns
 * false) before any byte comparison.
 *
 * @return 1 if @p a and @p b are equal in length and content, 0 otherwise.
 */
int32_t eshkol_constant_time_equal(const char* a, int32_t a_len,
                                     const char* b, int32_t b_len) {
    if (a_len != b_len) return 0;
    if (!a || !b) return 0;
    volatile unsigned char diff = 0;
    for (int i = 0; i < a_len; i++) {
        diff |= (unsigned char)a[i] ^ (unsigned char)b[i];
    }
    return diff == 0 ? 1 : 0;
}

/*******************************************************************************
 * B.6: sha256-file (streaming, no full-file load)
 ******************************************************************************/

#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>

/**
 * @brief Computes the SHA-256 digest of the file at @p path, streaming its contents in chunks.
 *
 * Uses CommonCrypto (CC_SHA256_*) on Apple platforms. Reads the file in
 * 64KB chunks rather than loading it fully into memory, then writes the
 * digest as a 64-character lowercase hex string to @p hex_buf.
 *
 * @param hex_buf Destination buffer for the null-terminated hex digest.
 * @param buf_size Capacity of @p hex_buf; must be at least 65.
 * @return 0 on success, -1 if the file cannot be opened, cannot be fully
 *         read, or @p buf_size is too small.
 */
int32_t eshkol_sha256_file(const char* path, char* hex_buf, int32_t buf_size) {
    if (!path || !hex_buf || buf_size < 65) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    char chunk[65536];
    ssize_t n;
    while ((n = read(fd, chunk, sizeof(chunk))) > 0) {
        CC_SHA256_Update(&ctx, chunk, (CC_LONG)n);
    }
    close(fd);
    if (n < 0) return -1;

    unsigned char hash[32];
    CC_SHA256_Final(hash, &ctx);

    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex_buf[i*2]   = hex[(hash[i] >> 4) & 0xf];
        hex_buf[i*2+1] = hex[hash[i] & 0xf];
    }
    hex_buf[64] = '\0';
    return 0;
}
#elif defined(_WIN32)
int32_t eshkol_sha256_file(const char* path, char* hex_buf, int32_t buf_size) {
    if (!path || !hex_buf || buf_size < 65) return -1;
    wchar_t* wide = platform_utf8_to_wide(path);
    if (!wide) return -1;
    FILE* file = _wfopen(wide, L"rb");
    free(wide);
    if (!file) return -1;
    BCRYPT_ALG_HANDLE algorithm = NULL;
    BCRYPT_HASH_HANDLE hash = NULL;
    unsigned char* object = NULL;
    unsigned char digest[32];
    DWORD object_size = 0, hash_size = 0, returned = 0;
    NTSTATUS status = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                                  NULL, 0);
    if (status == 0) status = BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                                                (PUCHAR)&object_size, sizeof(object_size),
                                                &returned, 0);
    if (status == 0) status = BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                                                (PUCHAR)&hash_size, sizeof(hash_size),
                                                &returned, 0);
    if (status == 0 && hash_size != sizeof(digest)) status = (NTSTATUS)-1;
    if (status == 0) {
        object = (unsigned char*)malloc(object_size);
        if (!object) status = (NTSTATUS)-1;
    }
    if (status == 0) status = BCryptCreateHash(algorithm, &hash, object,
                                               object_size, NULL, 0, 0);
    unsigned char chunk[65536];
    while (status == 0) {
        size_t got = fread(chunk, 1, sizeof(chunk), file);
        if (got && BCryptHashData(hash, chunk, (ULONG)got, 0) != 0) status = (NTSTATUS)-1;
        if (got < sizeof(chunk)) {
            if (ferror(file)) status = (NTSTATUS)-1;
            break;
        }
    }
    fclose(file);
    if (status == 0) status = BCryptFinishHash(hash, digest, sizeof(digest), 0);
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    free(object);
    if (status != 0) return -1;
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        hex_buf[i * 2] = hex[digest[i] >> 4];
        hex_buf[i * 2 + 1] = hex[digest[i] & 15];
    }
    hex_buf[64] = '\0';
    return 0;
}
#else
/* Linux: use OpenSSL */
#include <openssl/sha.h>

/**
 * @brief Computes the SHA-256 digest of the file at @p path, streaming its contents in chunks.
 *
 * Uses OpenSSL (SHA256_*) on non-Apple platforms. Reads the file in 64KB
 * chunks rather than loading it fully into memory, then writes the digest
 * as a 64-character lowercase hex string to @p hex_buf.
 *
 * @param hex_buf Destination buffer for the null-terminated hex digest.
 * @param buf_size Capacity of @p hex_buf; must be at least 65.
 * @return 0 on success, -1 if the file cannot be opened, cannot be fully
 *         read, or @p buf_size is too small.
 */
int32_t eshkol_sha256_file(const char* path, char* hex_buf, int32_t buf_size) {
    if (!path || !hex_buf || buf_size < 65) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    char chunk[65536];
    ssize_t n;
    while ((n = read(fd, chunk, sizeof(chunk))) > 0) {
        SHA256_Update(&ctx, chunk, (size_t)n);
    }
    close(fd);
    if (n < 0) return -1;

    unsigned char hash[32];
    SHA256_Final(hash, &ctx);

    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex_buf[i*2]   = hex[(hash[i] >> 4) & 0xf];
        hex_buf[i*2+1] = hex[hash[i] & 0xf];
    }
    hex_buf[64] = '\0';
    return 0;
}
#endif
