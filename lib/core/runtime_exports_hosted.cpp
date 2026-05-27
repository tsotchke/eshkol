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

bool runtime_file_mode_allows(const char* mode) {
    if (mode == nullptr || mode[0] == '\0') {
        return false;
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
        return false;
    }
    if (needs_write && !runtime_capability_allows("file-write")) {
        return false;
    }
    return true;
}

void deny_file_capability() {
    errno = EACCES;
}

void deny_capability() {
    errno = EACCES;
}

} // namespace

extern "C" FILE* eshkol_stdout_stream() {
    return stdout;
}

extern "C" FILE* eshkol_stdin_stream() {
    return stdin;
}

extern "C" FILE* eshkol_stderr_stream() {
    return stderr;
}

extern "C" std::uint64_t eshkol_jmp_buf_size() {
    return sizeof(jmp_buf);
}

#ifdef _WIN32
extern "C" double drand48() {
    return eshkol_drand48();
}

extern "C" long lrand48() {
    std::lock_guard<std::mutex> lock(g_drand48_mutex);
    g_drand48_state = (g_drand48_state * kDrand48Multiplier + kDrand48Addend) & kDrand48Mask;
    return static_cast<long>((g_drand48_state >> 17) & 0x7fffffffULL);
}

extern "C" void srand48(std::int64_t seed) {
    eshkol_srand48(seed);
}

extern "C" int setenv(const char* name, const char* value, int overwrite) {
    return eshkol_setenv(name, value, overwrite);
}

extern "C" int unsetenv(const char* name) {
    return eshkol_unsetenv(name);
}

#ifndef __MINGW32__
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

extern "C" void eshkol_srand48(std::int64_t seed) {
    std::lock_guard<std::mutex> lock(g_drand48_mutex);
    g_drand48_state = ((static_cast<std::uint64_t>(seed) << 16) | 0x330EULL) & kDrand48Mask;
}

extern "C" double eshkol_drand48() {
    std::lock_guard<std::mutex> lock(g_drand48_mutex);
    g_drand48_state = (g_drand48_state * kDrand48Multiplier + kDrand48Addend) & kDrand48Mask;
    return static_cast<double>(g_drand48_state) / static_cast<double>(kDrand48Mask + 1ULL);
}

extern "C" char* eshkol_getenv(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }
    if (!runtime_capability_allows("env-read")) {
        deny_capability();
        return nullptr;
    }

    return std::getenv(name);
}

extern "C" int eshkol_setenv(const char* name, const char* value, int overwrite) {
    if (name == nullptr || name[0] == '\0' || value == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("env-write")) {
        deny_capability();
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

extern "C" int eshkol_unsetenv(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("env-write")) {
        deny_capability();
        return -1;
    }

#ifdef _WIN32
    return ::_putenv_s(name, "");
#else
    return ::unsetenv(name);
#endif
}

extern "C" int eshkol_usleep(std::uint32_t usec) {
    std::this_thread::sleep_for(std::chrono::microseconds(usec));
    return 0;
}

extern "C" void eshkol_capability_runtime_clear() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    g_capability_policy_active = false;
    g_capability_allow_list.clear();
}

extern "C" void eshkol_capability_runtime_begin_install() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    g_capability_policy_active = true;
    g_capability_allow_list.clear();
}

extern "C" void eshkol_capability_runtime_allow(const char* capability) {
    if (capability == nullptr || capability[0] == '\0') {
        return;
    }

    std::lock_guard<std::mutex> lock(g_capability_mutex);
    if (g_capability_policy_active) {
        g_capability_allow_list.insert(capability);
    }
}

extern "C" int eshkol_capability_runtime_is_active() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    return g_capability_policy_active ? 1 : 0;
}

extern "C" int eshkol_capability_runtime_allows(const char* capability) {
    return runtime_capability_allows(capability) ? 1 : 0;
}

extern "C" int eshkol_capability_runtime_allows_file_mode(const char* mode) {
    return runtime_file_mode_allows(mode) ? 1 : 0;
}

extern "C" FILE* eshkol_fopen(const char* path, const char* mode) {
    if (path == nullptr || mode == nullptr) {
        errno = EINVAL;
        return nullptr;
    }
    if (!runtime_file_mode_allows(mode)) {
        deny_file_capability();
        return nullptr;
    }

    const auto normalized = normalize_runtime_path(path);
    return std::fopen(normalized.string().c_str(), mode);
}

extern "C" int eshkol_fputs(const char* str, FILE* stream) {
    if (str == nullptr || stream == nullptr) {
        errno = EINVAL;
        return EOF;
    }

    return std::fputs(str, stream);
}

extern "C" int eshkol_access(const char* path, int mode) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    const bool needs_write = (mode & 2) != 0;
    const bool needs_read = !needs_write || ((mode & 4) != 0);
    if ((needs_read && !runtime_capability_allows("file-read")) ||
        (needs_write && !runtime_capability_allows("file-write"))) {
        deny_file_capability();
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    return ::_access(normalized.string().c_str(), mode);
#else
    return ::access(normalized.string().c_str(), mode);
#endif
}

extern "C" int eshkol_remove(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_file_capability();
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
    return std::remove(normalized.string().c_str());
}

extern "C" int eshkol_rename(const char* old_path, const char* new_path) {
    if (old_path == nullptr || new_path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_file_capability();
        return -1;
    }

    const auto normalized_old = normalize_runtime_path(old_path);
    const auto normalized_new = normalize_runtime_path(new_path);
    return std::rename(normalized_old.string().c_str(), normalized_new.string().c_str());
}

extern "C" int eshkol_mkdir(const char* path, int mode) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_file_capability();
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

extern "C" int eshkol_rmdir(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-write")) {
        deny_file_capability();
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    return ::_rmdir(normalized.string().c_str());
#else
    return ::rmdir(normalized.string().c_str());
#endif
}

extern "C" int eshkol_chdir(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-read")) {
        deny_file_capability();
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    return ::_chdir(normalized.string().c_str());
#else
    return ::chdir(normalized.string().c_str());
#endif
}

extern "C" int eshkol_stat(const char* path, void* buf) {
    if (path == nullptr || buf == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (!runtime_capability_allows("file-read")) {
        deny_file_capability();
        return -1;
    }

    const auto normalized = normalize_runtime_path(path);
    return ::stat(normalized.string().c_str(), static_cast<struct stat*>(buf));
}

extern "C" void* eshkol_opendir(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return nullptr;
    }
    if (!runtime_capability_allows("file-read")) {
        deny_file_capability();
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
extern "C" void* opendir(const char* path) {
    return eshkol_opendir(path);
}

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
