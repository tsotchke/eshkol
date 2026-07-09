/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted runtime-export wrappers for generated code and JIT consumers.
 */

#include <eshkol/platform_runtime.h>
#include <eshkol/runtime_exports.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <setjmp.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr std::uint64_t kDrand48Multiplier = 0x5DEECE66DULL;
constexpr std::uint64_t kDrand48Addend = 0xBULL;
constexpr std::uint64_t kDrand48Mask = (1ULL << 48) - 1;

std::mutex g_drand48_mutex;
std::uint64_t g_drand48_state = 0x1234ABCD330EULL;

std::mutex g_capability_mutex;
bool g_capability_policy_active = false;
std::unordered_set<std::string> g_capability_allow_list;
std::unordered_set<std::string> g_capability_denials_reported;

#ifdef _WIN32
struct windows_dirent_shim {
    unsigned char reserved[21];
    char d_name[MAX_PATH];
};

static_assert(offsetof(windows_dirent_shim, d_name) == 21);

struct windows_dir_handle {
    HANDLE handle = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATAA find_data{};
    bool first_entry = true;
    windows_dirent_shim entry{};
};
#endif

/**
 * @brief Normalize a Unix-style runtime path for the host filesystem.
 *
 * On non-Windows hosts this is just a pass-through conversion to
 * std::filesystem::path. On Windows it additionally rewrites Unix-style
 * absolute paths so generated/REPL code that hardcodes POSIX-style paths
 * still resolves correctly: a leading "/X/..." drive-letter form becomes
 * "X:/...", and "/tmp" or "/tmp/..." is redirected to the platform's real
 * temp directory (falling back to the current directory if that lookup
 * fails).
 *
 * @param raw_path  Path as supplied by generated code (POSIX-style).
 * @return           Filesystem path usable with the host's native APIs.
 */
std::filesystem::path normalize_runtime_path(std::string_view raw_path) {
    if (raw_path.empty()) {
        return {};
    }

#ifdef _WIN32
    const auto runtime_temp_directory = []() {
        std::error_code ec;
        auto temp_dir = std::filesystem::temp_directory_path(ec);
        if (ec || temp_dir.empty()) {
            temp_dir = eshkol::platform::current_directory();
        }
        return temp_dir;
    };

    if (raw_path.size() >= 3 &&
        raw_path[0] == '/' &&
        std::isalpha(static_cast<unsigned char>(raw_path[1])) != 0 &&
        raw_path[2] == '/') {
        const char drive = static_cast<char>(std::toupper(static_cast<unsigned char>(raw_path[1])));
        auto normalized = std::filesystem::path(std::string(1, drive) + ":/");
        if (raw_path.size() > 3) {
            normalized /= std::filesystem::path(std::string(raw_path.substr(3)));
        }
        return normalized;
    }

    if (raw_path == "/tmp") {
        return runtime_temp_directory();
    }

    if (raw_path.rfind("/tmp/", 0) == 0) {
        auto normalized = runtime_temp_directory();
        if (raw_path.size() > 5) {
            normalized /= std::filesystem::path(std::string(raw_path.substr(5)));
        }
        return normalized;
    }
#endif

    return std::filesystem::path(std::string(raw_path));
}

/**
 * @brief Check whether the named capability is permitted under the current
 * capability policy.
 *
 * If no capability policy has been installed (via
 * eshkol_capability_runtime_begin_install()), every capability is allowed.
 * Once a policy is active, only capability names previously added with
 * eshkol_capability_runtime_allow() are permitted. Thread-safe (guarded by
 * g_capability_mutex).
 *
 * @param capability  Capability name, e.g. "file-read", "file-write",
 *                     "env-read", "env-write". NULL or empty is always denied.
 * @return             true if the capability is allowed, false otherwise.
 */
bool runtime_capability_allows(const char* capability) {
    if (capability == nullptr || capability[0] == '\0') {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_capability_mutex);
    if (!g_capability_policy_active) {
        return true;
    }
    return g_capability_allow_list.find(capability) != g_capability_allow_list.end();
}

/**
 * @brief Determine which capability (if any) an fopen()-style mode string
 * requires but is not currently granted.
 *
 * Inspects the first mode character ('r' implies read; 'w'/'a' imply write)
 * and any '+' flag (implies both read and write), then checks each implied
 * requirement against the active capability policy.
 *
 * @param mode  fopen()-style mode string (e.g. "r", "w+", "a"). NULL/empty
 *              is treated as requiring "file-read".
 * @return      "file-read" or "file-write" naming the first missing
 *              capability, or nullptr if the mode is fully permitted.
 */
const char* runtime_file_mode_missing_capability(const char* mode) {
    if (mode == nullptr || mode[0] == '\0') {
        return "file-read";
    }

    bool needs_read = mode[0] == 'r';
    bool needs_write = mode[0] == 'w' || mode[0] == 'a';
    for (const char* p = mode; *p; ++p) {
        if (*p == '+') {
            needs_read = true;
            needs_write = true;
        }
    }

    if (needs_read && !runtime_capability_allows("file-read")) {
        return "file-read";
    }
    if (needs_write && !runtime_capability_allows("file-write")) {
        return "file-write";
    }
    return nullptr;
}

/** @brief Whether an fopen()-style mode string is fully permitted by the current capability policy. */
bool runtime_file_mode_allows(const char* mode) {
    return runtime_file_mode_missing_capability(mode) == nullptr;
}

/**
 * @brief Set errno to EACCES and log a one-time "capability denied" message
 * to stderr for the given capability.
 *
 * Deduplicates via g_capability_denials_reported so repeated denials of the
 * same capability (e.g. many file-read attempts in a loop) only produce a
 * single stderr line. Thread-safe.
 *
 * @param capability  Capability name that was denied; "unknown" is used if
 *                     NULL or empty.
 */
void report_capability_denied(const char* capability) {
    errno = EACCES;
    const char* name = (capability != nullptr && capability[0] != '\0')
        ? capability
        : "unknown";

    bool should_report = false;
    {
        std::lock_guard<std::mutex> lock(g_capability_mutex);
        should_report = g_capability_denials_reported.insert(name).second;
    }
    if (should_report) {
        std::fprintf(stderr, "capability denied: %s\n", name);
    }
}

/** @brief Thin alias for report_capability_denied(), used at call sites that deny a capability. */
void deny_capability(const char* capability) {
    report_capability_denied(capability);
}

} // namespace

/** @brief Return the host's stdout FILE* so generated code can call libc stdio without linking against the CRT's own global directly. */
extern "C" FILE* eshkol_stdout_stream() {
    return stdout;
}

/** @brief Return the host's stdin FILE*, mirroring eshkol_stdout_stream(). */
extern "C" FILE* eshkol_stdin_stream() {
    return stdin;
}

/** @brief Return the host's stderr FILE*, mirroring eshkol_stdout_stream(). */
extern "C" FILE* eshkol_stderr_stream() {
    return stderr;
}

/** @brief Return sizeof(jmp_buf) on this platform, so generated code can allocate a correctly sized buffer for setjmp/longjmp without depending on the host's jmp_buf layout at compile time. */
extern "C" std::uint64_t eshkol_jmp_buf_size() {
    return sizeof(jmp_buf);
}

#ifdef _WIN32
/** @brief Windows shim for the POSIX drand48() libc function; forwards to eshkol_drand48() so the same LCG state backs both entry points. */
extern "C" double drand48() {
    return eshkol_drand48();
}

/**
 * @brief Windows shim for the POSIX lrand48() libc function.
 *
 * Advances the shared drand48 LCG state (same generator/state as
 * eshkol_drand48()/eshkol_srand48()) and returns the top 31 bits as a
 * non-negative long, matching glibc's lrand48() range. Thread-safe (guarded
 * by g_drand48_mutex).
 *
 * @return  A pseudo-random value in [0, 2^31).
 */
extern "C" long lrand48() {
    std::lock_guard<std::mutex> lock(g_drand48_mutex);
    g_drand48_state = (g_drand48_state * kDrand48Multiplier + kDrand48Addend) & kDrand48Mask;
    return static_cast<long>((g_drand48_state >> 17) & 0x7fffffffULL);
}

/** @brief Windows shim for the POSIX srand48() libc function; forwards to eshkol_srand48(). */
extern "C" void srand48(std::int64_t seed) {
    eshkol_srand48(seed);
}

/** @brief Windows shim for the POSIX setenv() libc function; forwards to eshkol_setenv() (which applies the env-write capability check). */
extern "C" int setenv(const char* name, const char* value, int overwrite) {
    return eshkol_setenv(name, value, overwrite);
}

/** @brief Windows shim for the POSIX unsetenv() libc function; forwards to eshkol_unsetenv() (which applies the env-write capability check). */
extern "C" int unsetenv(const char* name) {
    return eshkol_unsetenv(name);
}

#ifndef __MINGW32__
/**
 * @brief Windows shim implementing the POSIX clock_gettime() API in terms of
 * FILETIME.
 *
 * Ignores clock_id (always uses the system's current time via
 * GetSystemTimePreciseAsFileTime when available, falling back to
 * GetSystemTimeAsFileTime) and converts the Windows FILETIME epoch
 * (1601-01-01, 100ns ticks) to a Unix-epoch struct-timespec-shaped result
 * written through ts_raw.
 *
 * @param clock_id  Unused; accepted for POSIX signature compatibility.
 * @param ts_raw    Out-pointer to a struct compatible with `struct timespec`
 *                  (tv_sec, tv_nsec as int64_t); must not be NULL.
 * @return          0 on success, -1 with errno set to EINVAL if ts_raw is NULL.
 */
extern "C" int clock_gettime(int clock_id, void* ts_raw) {
    (void)clock_id;
    if (ts_raw == nullptr) {
        errno = EINVAL;
        return -1;
    }

    struct timespec_shim {
        std::int64_t tv_sec;
        std::int64_t tv_nsec;
    };

    FILETIME file_time{};
    if (&GetSystemTimePreciseAsFileTime != nullptr) {
        GetSystemTimePreciseAsFileTime(&file_time);
    } else {
        GetSystemTimeAsFileTime(&file_time);
    }

    ULARGE_INTEGER value{};
    value.LowPart = file_time.dwLowDateTime;
    value.HighPart = file_time.dwHighDateTime;

    constexpr std::uint64_t kUnixEpoch100ns = 116444736000000000ULL;
    const std::uint64_t unix_100ns = value.QuadPart - kUnixEpoch100ns;

    auto* ts = static_cast<timespec_shim*>(ts_raw);
    ts->tv_sec = static_cast<std::int64_t>(unix_100ns / 10000000ULL);
    ts->tv_nsec = static_cast<std::int64_t>((unix_100ns % 10000000ULL) * 100ULL);
    return 0;
}

/**
 * @brief Windows shim implementing the POSIX gettimeofday() API in terms of
 * FILETIME.
 *
 * Ignores tz_raw (timezone argument, unused by modern POSIX callers) and
 * converts the current system time (via GetSystemTimePreciseAsFileTime when
 * available, else GetSystemTimeAsFileTime) from the Windows FILETIME epoch
 * to a Unix-epoch struct-timeval-shaped result written through tv_raw.
 *
 * @param tv_raw  Out-pointer to a struct compatible with `struct timeval`
 *                (tv_sec, tv_usec as int64_t); must not be NULL.
 * @param tz_raw  Unused; accepted for POSIX signature compatibility.
 * @return        0 on success, -1 with errno set to EINVAL if tv_raw is NULL.
 */
extern "C" int gettimeofday(void* tv_raw, void* tz_raw) {
    (void)tz_raw;
    if (tv_raw == nullptr) {
        errno = EINVAL;
        return -1;
    }

    struct timeval_shim {
        std::int64_t tv_sec;
        std::int64_t tv_usec;
    };

    FILETIME file_time{};
    if (&GetSystemTimePreciseAsFileTime != nullptr) {
        GetSystemTimePreciseAsFileTime(&file_time);
    } else {
        GetSystemTimeAsFileTime(&file_time);
    }

    ULARGE_INTEGER value{};
    value.LowPart = file_time.dwLowDateTime;
    value.HighPart = file_time.dwHighDateTime;

    constexpr std::uint64_t kUnixEpoch100ns = 116444736000000000ULL;
    const std::uint64_t unix_100ns = value.QuadPart - kUnixEpoch100ns;

    auto* tv = static_cast<timeval_shim*>(tv_raw);
    tv->tv_sec = static_cast<std::int64_t>(unix_100ns / 10000000ULL);
    tv->tv_usec = static_cast<std::int64_t>((unix_100ns % 10000000ULL) / 10ULL);
    return 0;
}
#endif
#endif

/**
 * @brief Seed the runtime's cross-platform drand48-compatible generator.
 *
 * Reproduces glibc's drand48 seeding rule: the 48-bit state is formed from
 * the low 32 bits of `seed` shifted into the high bits with the fixed
 * constant 0x330E in the low 16 bits. Thread-safe (guarded by
 * g_drand48_mutex) so this backs eshkol_drand48()/lrand48()/drand48()
 * consistently regardless of platform.
 *
 * @param seed  Seed value (only the low 32 bits are significant).
 */
extern "C" void eshkol_srand48(std::int64_t seed) {
    std::lock_guard<std::mutex> lock(g_drand48_mutex);
    g_drand48_state = ((static_cast<std::uint64_t>(seed) << 16) | 0x330EULL) & kDrand48Mask;
}

/**
 * @brief Cross-platform reimplementation of POSIX drand48().
 *
 * Advances the shared 48-bit linear congruential generator state (same
 * multiplier/addend/mask as glibc's drand48 family) and returns it scaled
 * to [0.0, 1.0). Thread-safe (guarded by g_drand48_mutex); this is the
 * generator exported to generated code for portable random-number support
 * on platforms (e.g. Windows) that lack a native drand48().
 *
 * @return  A pseudo-random double in [0.0, 1.0).
 */
extern "C" double eshkol_drand48() {
    std::lock_guard<std::mutex> lock(g_drand48_mutex);
    g_drand48_state = (g_drand48_state * kDrand48Multiplier + kDrand48Addend) & kDrand48Mask;
    return static_cast<double>(g_drand48_state) / static_cast<double>(kDrand48Mask + 1ULL);
}

/**
 * @brief Read an environment variable, subject to the "env-read" capability.
 *
 * @param name  Variable name; NULL or empty returns nullptr.
 * @return      The value from std::getenv(), or nullptr if name is invalid
 *              or the "env-read" capability is denied (errno is set to
 *              EACCES in the denied case via deny_capability()).
 */
extern "C" char* eshkol_getenv(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }
    if (!runtime_capability_allows("env-read")) {
        deny_capability("env-read");
        return nullptr;
    }

    return std::getenv(name);
}

/**
 * @brief Set an environment variable, subject to the "env-write" capability.
 *
 * Delegates to `::_putenv_s`/`::setenv` per platform after validating
 * arguments and the capability check.
 *
 * @param name       Variable name; must be non-NULL/non-empty.
 * @param value      Value to set; must be non-NULL.
 * @param overwrite  Non-zero to replace an existing value (POSIX semantics);
 *                   on Windows, if zero and the variable already exists,
 *                   this is a no-op that returns success.
 * @return           0 on success, -1 with errno set (EINVAL for bad
 *                    arguments, EACCES if "env-write" is denied).
 */
extern "C" int eshkol_setenv(const char* name, const char* value, int overwrite) {
    if (name == nullptr || name[0] == '\0' || value == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("env-write")) {
        deny_capability("env-write");
        return -1;
    }

#ifdef _WIN32
    if (!overwrite && std::getenv(name) != nullptr) {
        return 0;
    }
    return ::_putenv_s(name, value);
#else
    return ::setenv(name, value, overwrite);
#endif
}

/**
 * @brief Remove an environment variable, subject to the "env-write" capability.
 *
 * @param name  Variable name; must be non-NULL/non-empty.
 * @return      0 on success, -1 with errno set (EINVAL for a bad name,
 *              EACCES if "env-write" is denied).
 */
extern "C" int eshkol_unsetenv(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("env-write")) {
        deny_capability("env-write");
        return -1;
    }

#ifdef _WIN32
    return ::_putenv_s(name, "");
#else
    return ::unsetenv(name);
#endif
}

/**
 * @brief Suspend the calling thread for the given number of microseconds.
 *
 * Portable replacement for POSIX usleep(), implemented with
 * std::this_thread::sleep_for so it also works on Windows.
 *
 * @param usec  Sleep duration in microseconds.
 * @return      Always 0.
 */
extern "C" int eshkol_usleep(std::uint32_t usec) {
    std::this_thread::sleep_for(std::chrono::microseconds(usec));
    return 0;
}

/** @brief Deactivate the capability policy and clear the allow-list, reverting to "everything permitted". Thread-safe. */
extern "C" void eshkol_capability_runtime_clear() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    g_capability_policy_active = false;
    g_capability_allow_list.clear();
}

/**
 * @brief Activate the capability policy with an empty allow-list.
 *
 * After this call, every capability is denied until explicitly permitted via
 * eshkol_capability_runtime_allow(). Thread-safe.
 */
extern "C" void eshkol_capability_runtime_begin_install() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    g_capability_policy_active = true;
    g_capability_allow_list.clear();
}

/**
 * @brief Add a capability name to the active policy's allow-list.
 *
 * No-op if no policy is currently active (i.e. between
 * eshkol_capability_runtime_clear() and the next begin_install()) or if
 * `capability` is NULL/empty. Thread-safe.
 *
 * @param capability  Capability name to permit, e.g. "file-read".
 */
extern "C" void eshkol_capability_runtime_allow(const char* capability) {
    if (capability == nullptr || capability[0] == '\0') {
        return;
    }

    std::lock_guard<std::mutex> lock(g_capability_mutex);
    if (g_capability_policy_active) {
        g_capability_allow_list.insert(capability);
    }
}

/** @brief Whether a capability policy is currently installed (1) or all capabilities are unrestricted (0). Thread-safe. */
extern "C" int eshkol_capability_runtime_is_active() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    return g_capability_policy_active ? 1 : 0;
}

/** @brief C-ABI (int-returning) wrapper around runtime_capability_allows() for generated code to query a capability. */
extern "C" int eshkol_capability_runtime_allows(const char* capability) {
    return runtime_capability_allows(capability) ? 1 : 0;
}

/** @brief C-ABI (int-returning) wrapper around runtime_file_mode_allows() for generated code to check an fopen() mode string against the capability policy. */
extern "C" int eshkol_capability_runtime_allows_file_mode(const char* mode) {
    return runtime_file_mode_allows(mode) ? 1 : 0;
}

/** @brief Report `capability` as denied (sets errno=EACCES, logs once to stderr). Exposed so generated code can surface a capability failure through the same reporting path as the internal file/env helpers. */
extern "C" void eshkol_capability_runtime_deny(const char* capability) {
    deny_capability(capability);
}

/**
 * @brief Capability-checked, path-normalizing wrapper around std::fopen().
 *
 * Validates arguments, denies the call if the requested mode needs a
 * capability ("file-read" and/or "file-write") that isn't granted, then
 * normalizes `path` (see normalize_runtime_path()) before opening it.
 *
 * @param path  File path (POSIX-style; normalized for the host).
 * @param mode  fopen()-style mode string.
 * @return      Open FILE* on success; nullptr on invalid arguments (errno
 *              EINVAL), capability denial (errno EACCES), or an underlying
 *              fopen() failure.
 */
extern "C" FILE* eshkol_fopen(const char* path, const char* mode) {
    if (path == nullptr || mode == nullptr) {
        errno = EINVAL;
        return nullptr;
    }
    if (const char* missing = runtime_file_mode_missing_capability(mode)) {
        deny_capability(missing);
        return nullptr;
    }

    const auto normalized = normalize_runtime_path(path);
    return std::fopen(normalized.string().c_str(), mode);
}

/**
 * @brief Thin, NULL-checked wrapper around std::fputs() exported to generated code.
 *
 * @param str     Null-terminated string to write; NULL is rejected.
 * @param stream  Destination stream; NULL is rejected.
 * @return        Non-negative on success; EOF with errno EINVAL if either
 *                argument is NULL, or EOF as propagated from std::fputs().
 */
extern "C" int eshkol_fputs(const char* str, FILE* stream) {
    if (str == nullptr || stream == nullptr) {
        errno = EINVAL;
        return EOF;
    }

    return std::fputs(str, stream);
}

/**
 * @brief Capability-checked, path-normalizing wrapper around access()/`_access()`.
 *
 * `mode` follows the standard access() bitmask (bit 2 = write, bit 4 = read
 * or, for the special mode 0, existence-only which is treated as a read
 * check). Checks the corresponding "file-read"/"file-write" capabilities
 * before normalizing the path and delegating to the platform's access().
 *
 * @param path  File path to check (normalized for the host).
 * @param mode  access()-style mode bitmask.
 * @return      0 if accessible and permitted; -1 with errno set (EINVAL for
 *              a NULL path, EACCES for a denied capability, or as set by
 *              the underlying access() call).
 */
extern "C" int eshkol_access(const char* path, int mode) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    const bool needs_write = (mode & 2) != 0;
    const bool needs_read = !needs_write || ((mode & 4) != 0);
    if (needs_read && !runtime_capability_allows("file-read")) {
        deny_capability("file-read");
        return -1;
    }
    if (needs_write && !runtime_capability_allows("file-write")) {
        deny_capability("file-write");
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    return ::_access(normalized.string().c_str(), mode);
#else
    return ::access(normalized.string().c_str(), mode);
#endif
}

/**
 * @brief Capability-checked, path-normalizing wrapper around std::remove().
 *
 * @param path  Path to delete; requires the "file-write" capability.
 * @return      0 on success; -1 with errno set (EINVAL for a NULL path,
 *              EACCES if "file-write" is denied, or as set by std::remove()).
 */
extern "C" int eshkol_remove(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_capability("file-write");
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
    return std::remove(normalized.string().c_str());
}

/**
 * @brief Capability-checked, path-normalizing wrapper around std::rename().
 *
 * @param old_path  Existing path; requires the "file-write" capability.
 * @param new_path  Destination path.
 * @return          0 on success; -1 with errno set (EINVAL for a NULL
 *                  argument, EACCES if "file-write" is denied, or as set by
 *                  std::rename()).
 */
extern "C" int eshkol_rename(const char* old_path, const char* new_path) {
    if (old_path == nullptr || new_path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_capability("file-write");
        return -1;
    }

    const auto normalized_old = normalize_runtime_path(old_path);
    const auto normalized_new = normalize_runtime_path(new_path);
    return std::rename(normalized_old.string().c_str(), normalized_new.string().c_str());
}

/**
 * @brief Capability-checked, path-normalizing wrapper around mkdir()/`_mkdir()`.
 *
 * @param path  Directory to create; requires the "file-write" capability.
 * @param mode  POSIX permission bits; ignored on Windows (`_mkdir()` takes
 *              no mode).
 * @return      0 on success; -1 with errno set (EINVAL for a NULL path,
 *              EACCES if "file-write" is denied, or as set by the
 *              underlying mkdir() call).
 */
extern "C" int eshkol_mkdir(const char* path, int mode) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_capability("file-write");
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    (void)mode;
    return ::_mkdir(normalized.string().c_str());
#else
    return ::mkdir(normalized.string().c_str(), static_cast<mode_t>(mode));
#endif
}

/**
 * @brief Capability-checked, path-normalizing wrapper around rmdir()/`_rmdir()`.
 *
 * @param path  Directory to remove; requires the "file-write" capability.
 * @return      0 on success; -1 with errno set (EINVAL for a NULL path,
 *              EACCES if "file-write" is denied, or as set by the
 *              underlying rmdir() call).
 */
extern "C" int eshkol_rmdir(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_capability("file-write");
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    return ::_rmdir(normalized.string().c_str());
#else
    return ::rmdir(normalized.string().c_str());
#endif
}

/**
 * @brief Capability-checked, path-normalizing wrapper around chdir()/`_chdir()`.
 *
 * Note this is gated on the "file-read" capability rather than
 * "file-write", since changing the working directory doesn't itself modify
 * any file.
 *
 * @param path  Directory to change into.
 * @return      0 on success; -1 with errno set (EINVAL for a NULL path,
 *              EACCES if "file-read" is denied, or as set by the underlying
 *              chdir() call).
 */
extern "C" int eshkol_chdir(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-read")) {
        deny_capability("file-read");
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    return ::_chdir(normalized.string().c_str());
#else
    return ::chdir(normalized.string().c_str());
#endif
}

/**
 * @brief Capability-checked, path-normalizing wrapper around ::stat().
 *
 * @param path  Path to stat; requires the "file-read" capability.
 * @param buf   Out-pointer, cast to `struct stat*`, filled in on success.
 * @return      0 on success; -1 with errno set (EINVAL for a NULL argument,
 *              EACCES if "file-read" is denied, or as set by ::stat()).
 */
extern "C" int eshkol_stat(const char* path, void* buf) {
    if (path == nullptr || buf == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-read")) {
        deny_capability("file-read");
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
    return ::stat(normalized.string().c_str(), static_cast<struct stat*>(buf));
}

/**
 * @brief Capability-checked, path-normalizing wrapper around opendir().
 *
 * On POSIX hosts this normalizes the path and delegates straight to
 * ::opendir(). On Windows there is no native opendir()/readdir()/closedir()
 * family, so this allocates a windows_dir_handle that wraps FindFirstFileA
 * over the directory's wildcard search pattern and drives the
 * readdir()/closedir() shims defined further
 * below; the caller owns the returned handle and must release it via
 * closedir().
 *
 * @param path  Directory to open; requires the "file-read" capability.
 * @return      An opaque directory handle (DIR* on POSIX, a
 *              windows_dir_handle* on Windows) on success; nullptr on
 *              invalid arguments (errno EINVAL), capability denial (errno
 *              EACCES), or if the directory could not be opened.
 */
extern "C" void* eshkol_opendir(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return nullptr;
    }
    if (!runtime_capability_allows("file-read")) {
        deny_capability("file-read");
        return nullptr;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    auto search_path = normalized;
    search_path /= "*";

    auto* dir = new windows_dir_handle();
    dir->handle = ::FindFirstFileA(search_path.string().c_str(), &dir->find_data);
    if (dir->handle == INVALID_HANDLE_VALUE) {
        delete dir;
        errno = ENOENT;
        return nullptr;
    }
    dir->first_entry = true;
    return dir;
#else
    return ::opendir(normalized.string().c_str());
#endif
}

#ifdef _WIN32
/** @brief Windows shim for POSIX opendir(); forwards to eshkol_opendir() (which applies the "file-read" capability check and path normalization). */
extern "C" void* opendir(const char* path) {
    return eshkol_opendir(path);
}

/**
 * @brief Windows shim for POSIX readdir(), backed by the FindFirstFileA/FindNextFileA state in a windows_dir_handle.
 *
 * The first call returns the entry already captured by FindFirstFileA (see
 * eshkol_opendir()); subsequent calls advance via FindNextFileA. The
 * returned dirent-shaped struct is reused/overwritten on every call (it
 * lives inside `dir`), matching the usual readdir() contract that the
 * result is only valid until the next call on the same handle.
 *
 * @param dir_raw  Handle previously returned by opendir()/eshkol_opendir().
 * @return         Pointer to a windows_dirent_shim with `d_name` populated,
 *                 or nullptr at end-of-directory or on a NULL/invalid handle
 *                 (errno EINVAL in the latter case).
 */
extern "C" void* readdir(void* dir_raw) {
    if (dir_raw == nullptr) {
        errno = EINVAL;
        return nullptr;
    }

    auto* dir = static_cast<windows_dir_handle*>(dir_raw);
    const WIN32_FIND_DATAA* data = nullptr;

    if (dir->first_entry) {
        dir->first_entry = false;
        data = &dir->find_data;
    } else {
        if (!::FindNextFileA(dir->handle, &dir->find_data)) {
            return nullptr;
        }
        data = &dir->find_data;
    }

    dir->entry = {};
    std::snprintf(dir->entry.d_name, sizeof(dir->entry.d_name), "%s", data->cFileName);
    return &dir->entry;
}

/**
 * @brief Windows shim for POSIX closedir(); closes the FindFirstFileA/FindNextFileA search handle and frees the windows_dir_handle.
 *
 * @param dir_raw  Handle previously returned by opendir()/eshkol_opendir().
 * @return         0 on success; -1 with errno EINVAL if dir_raw is NULL.
 */
extern "C" int closedir(void* dir_raw) {
    if (dir_raw == nullptr) {
        errno = EINVAL;
        return -1;
    }

    auto* dir = static_cast<windows_dir_handle*>(dir_raw);
    if (dir->handle != INVALID_HANDLE_VALUE) {
        ::FindClose(dir->handle);
    }
    delete dir;
    return 0;
}
#endif
