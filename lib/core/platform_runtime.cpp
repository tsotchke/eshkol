/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <eshkol/platform_runtime.h>
#include <eshkol/build_config.h>
#include <eshkol/runtime_exports.h>

#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <setjmp.h>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <system_error>

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

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace eshkol::platform {

namespace {

constexpr std::uint64_t kDrand48Multiplier = 0x5DEECE66DULL;
constexpr std::uint64_t kDrand48Addend = 0xBULL;
constexpr std::uint64_t kDrand48Mask = (1ULL << 48) - 1;

std::mutex g_drand48_mutex;
std::uint64_t g_drand48_state = 0x1234ABCD330EULL;

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

std::filesystem::path canonical_if_exists(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return {};
    }

    auto canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        return std::filesystem::absolute(path, ec);
    }
    return canonical;
}

#ifdef _WIN32
std::wstring widen_utf8(std::string_view value) {
    if (value.empty()) {
        return {};
    }

    const auto* bytes = value.data();
    const int size = static_cast<int>(value.size());

    int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, bytes, size, nullptr, 0);
    if (required <= 0) {
        required = MultiByteToWideChar(CP_ACP, 0, bytes, size, nullptr, 0);
        if (required <= 0) {
            return {};
        }

        std::wstring wide(required, L'\0');
        if (MultiByteToWideChar(CP_ACP, 0, bytes, size, wide.data(), required) <= 0) {
            return {};
        }
        return wide;
    }

    std::wstring wide(required, L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, bytes, size, wide.data(), required) <= 0) {
        return {};
    }
    return wide;
}

std::wstring quote_windows_argument(std::string_view argument) {
    const auto wide = widen_utf8(argument);
    if (wide.empty()) {
        return L"\"\"";
    }

    if (wide.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return wide;
    }

    std::wstring quoted;
    quoted.reserve(wide.size() + 2);
    quoted.push_back(L'"');

    std::size_t backslash_count = 0;
    for (wchar_t ch : wide) {
        if (ch == L'\\') {
            ++backslash_count;
            continue;
        }

        if (ch == L'"') {
            quoted.append(backslash_count * 2 + 1, L'\\');
            quoted.push_back(L'"');
            backslash_count = 0;
            continue;
        }

        quoted.append(backslash_count, L'\\');
        backslash_count = 0;
        quoted.push_back(ch);
    }

    quoted.append(backslash_count * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}

std::wstring build_windows_command_line(const std::vector<std::string>& arguments) {
    std::wstring command_line;
    bool first = true;
    for (const auto& argument : arguments) {
        if (!first) {
            command_line.push_back(L' ');
        }
        first = false;
        command_line += quote_windows_argument(argument);
    }
    return command_line;
}
#endif

} // namespace

std::filesystem::path executable_path() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        DWORD len = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (len == 0) {
            return {};
        }
        if (len < buffer.size()) {
            buffer.resize(len);
            return std::filesystem::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    return std::filesystem::path(buffer.c_str());
#elif defined(__linux__)
    std::array<char, 4096> buffer{};
    ssize_t len = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0) {
        return {};
    }
    buffer[static_cast<std::size_t>(len)] = '\0';
    return std::filesystem::path(buffer.data());
#else
    return {};
#endif
}

std::filesystem::path executable_directory() {
    auto path = executable_path();
    if (path.empty()) {
        return {};
    }
    return path.parent_path();
}

std::filesystem::path current_directory() {
    std::error_code ec;
    auto cwd = std::filesystem::current_path(ec);
    return ec ? std::filesystem::path{} : cwd;
}

std::string find_first_existing(const std::vector<std::filesystem::path>& candidates) {
    for (const auto& candidate : candidates) {
        auto resolved = canonical_if_exists(candidate);
        if (!resolved.empty()) {
            return resolved.string();
        }
    }
    return {};
}

std::string home_directory() {
    const char* home = std::getenv("HOME");
    if (home && home[0] != '\0') {
        return home;
    }

#ifdef _WIN32
    const char* user_profile = std::getenv("USERPROFILE");
    if (user_profile && user_profile[0] != '\0') {
        return user_profile;
    }

    const char* app_data = std::getenv("APPDATA");
    if (app_data && app_data[0] != '\0') {
        return app_data;
    }
#endif

    auto cwd = current_directory();
    return cwd.empty() ? std::string{} : cwd.string();
}

bool stdin_isatty() {
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

bool stdout_isatty() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

bool initialize_interactive_console() {
#ifdef _WIN32
    if (!stdout_isatty()) {
        return false;
    }

    SetConsoleOutputCP(CP_UTF8);
    if (stdin_isatty()) {
        SetConsoleCP(CP_UTF8);
    }

    HANDLE stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (stdout_handle != nullptr && stdout_handle != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(stdout_handle, &mode)) {
#ifdef ENABLE_VIRTUAL_TERMINAL_PROCESSING
            SetConsoleMode(stdout_handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
        }
    }

    return stdout_supports_utf8();
#else
    return false;
#endif
}

bool stdout_supports_utf8() {
#ifdef _WIN32
    return stdout_isatty() && GetConsoleOutputCP() == CP_UTF8;
#else
    return true;
#endif
}

std::filesystem::path make_temp_path(std::string_view stem, std::string_view extension) {
    std::error_code ec;
    auto temp_dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        temp_dir = current_directory();
    }

    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<std::mt19937_64::result_type>(ticks));

    for (int attempt = 0; attempt < 128; ++attempt) {
        auto candidate = temp_dir /
            (std::string(stem) + "_" + std::to_string(rng()) + "_" + std::to_string(attempt) + std::string(extension));

        if (!std::filesystem::exists(candidate, ec)) {
            return candidate;
        }
    }

    return temp_dir / (std::string(stem) + std::string(extension));
}

std::string cxx_compiler() {
#ifdef _WIN32
    std::string compiler = ESHKOL_HOST_CXX_COMPILER;
    if (compiler.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(compiler[0])) &&
        compiler[1] == ':' &&
        compiler[2] == '/') {
        std::replace(compiler.begin(), compiler.end(), '/', '\\');
    }
    return compiler;
#else
    return ESHKOL_HOST_CXX_COMPILER;
#endif
}

std::string llc_executable() {
    return ESHKOL_HOST_LLC_EXECUTABLE;
}

std::string executable_suffix() {
    return ESHKOL_HOST_EXECUTABLE_SUFFIX;
}

std::string static_library_name(std::string_view stem) {
    return std::string(ESHKOL_HOST_STATIC_LIBRARY_PREFIX) +
           std::string(stem) +
           std::string(ESHKOL_HOST_STATIC_LIBRARY_SUFFIX);
}

std::vector<std::string> host_runtime_link_args() {
    std::vector<std::string> args;
    std::stringstream stream(ESHKOL_HOST_RUNTIME_LINK_ARGS);
    std::string item;

    while (std::getline(stream, item, ';')) {
        if (!item.empty()) {
#ifdef _WIN32
            if (item.size() >= 3 &&
                std::isalpha(static_cast<unsigned char>(item[0])) &&
                item[1] == ':' &&
                item[2] == '/') {
                std::replace(item.begin(), item.end(), '/', '\\');
            }
#endif
            args.push_back(item);
        }
    }

    return args;
}

std::filesystem::path with_executable_suffix(const std::filesystem::path& path) {
    if (path.empty()) {
        return {};
    }

    const auto suffix = executable_suffix();
    if (suffix.empty() || path.extension() == suffix) {
        return path;
    }

    auto normalized = path;
    normalized += suffix;
    return normalized;
}

std::string shell_quote(std::string_view argument) {
#ifdef _WIN32
    std::string escaped = "\"";
    for (char ch : argument) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
#else
    if (argument.find_first_of(" '\"\\$`") == std::string_view::npos) {
        return std::string(argument);
    }

    std::string escaped = "'";
    for (char ch : argument) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
#endif
}

int run_command(const std::vector<std::string>& arguments) {
    if (arguments.empty()) {
        return -1;
    }

#ifdef _WIN32
    const auto application = widen_utf8(arguments.front());
    if (application.empty()) {
        return ERROR_INVALID_PARAMETER;
    }

    auto command_line = build_windows_command_line(arguments);
    if (command_line.empty()) {
        return ERROR_INVALID_PARAMETER;
    }

    std::vector<wchar_t> mutable_command_line(command_line.begin(), command_line.end());
    mutable_command_line.push_back(L'\0');

    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);

    PROCESS_INFORMATION process_info{};
    if (!CreateProcessW(application.c_str(),
                        mutable_command_line.data(),
                        nullptr,
                        nullptr,
                        FALSE,
                        0,
                        nullptr,
                        nullptr,
                        &startup_info,
                        &process_info)) {
        return static_cast<int>(GetLastError());
    }

    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(process_info.hProcess, &exit_code)) {
        exit_code = GetLastError();
    }

    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return static_cast<int>(exit_code);
#else
    std::string command;
    for (const auto& argument : arguments) {
        if (!command.empty()) {
            command.push_back(' ');
        }
        command += shell_quote(argument);
    }
    return std::system(command.c_str());
#endif
}

std::filesystem::path resolve_executable_output(const std::filesystem::path& base_path) {
    if (base_path.empty()) {
        return {};
    }

    auto preferred = with_executable_suffix(base_path);
    auto resolved = canonical_if_exists(preferred);
    if (!resolved.empty()) {
        return resolved;
    }

    resolved = canonical_if_exists(base_path);
    if (!resolved.empty()) {
        return resolved;
    }

    return preferred.empty() ? base_path : preferred;
}

} // namespace eshkol::platform

namespace {

std::filesystem::path runtime_temp_directory() {
    std::error_code ec;
    auto temp_dir = std::filesystem::temp_directory_path(ec);
    if (ec || temp_dir.empty()) {
        temp_dir = eshkol::platform::current_directory();
    }
    return temp_dir;
}

std::filesystem::path normalize_runtime_path(std::string_view raw_path) {
    if (raw_path.empty()) {
        return {};
    }

#ifdef _WIN32
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
    std::lock_guard<std::mutex> lock(eshkol::platform::g_drand48_mutex);
    eshkol::platform::g_drand48_state =
        (eshkol::platform::g_drand48_state * eshkol::platform::kDrand48Multiplier +
         eshkol::platform::kDrand48Addend) &
        eshkol::platform::kDrand48Mask;
    return static_cast<long>((eshkol::platform::g_drand48_state >> 17) & 0x7fffffffULL);
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

extern "C" void eshkol_srand48(std::int64_t seed) {
    std::lock_guard<std::mutex> lock(eshkol::platform::g_drand48_mutex);
    eshkol::platform::g_drand48_state =
        ((static_cast<std::uint64_t>(seed) << 16) | 0x330EULL) & eshkol::platform::kDrand48Mask;
}

extern "C" double eshkol_drand48() {
    std::lock_guard<std::mutex> lock(eshkol::platform::g_drand48_mutex);
    eshkol::platform::g_drand48_state =
        (eshkol::platform::g_drand48_state * eshkol::platform::kDrand48Multiplier +
         eshkol::platform::kDrand48Addend) &
        eshkol::platform::kDrand48Mask;
    return static_cast<double>(eshkol::platform::g_drand48_state) /
           static_cast<double>(eshkol::platform::kDrand48Mask + 1ULL);
}

extern "C" char* eshkol_getenv(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }

    return std::getenv(name);
}

extern "C" int eshkol_setenv(const char* name, const char* value, int overwrite) {
    if (name == nullptr || name[0] == '\0' || value == nullptr) {
        errno = EINVAL;
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

extern "C" FILE* eshkol_fopen(const char* path, const char* mode) {
    if (path == nullptr || mode == nullptr) {
        errno = EINVAL;
        return nullptr;
    }

    const auto normalized = normalize_runtime_path(path);
    return std::fopen(normalized.string().c_str(), mode);
}

extern "C" int eshkol_access(const char* path, int mode) {
    if (path == nullptr) {
        errno = EINVAL;
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

    const auto normalized = normalize_runtime_path(path);
    return std::remove(normalized.string().c_str());
}

extern "C" int eshkol_rename(const char* old_path, const char* new_path) {
    if (old_path == nullptr || new_path == nullptr) {
        errno = EINVAL;
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

    const auto normalized = normalize_runtime_path(path);
    return ::stat(normalized.string().c_str(), static_cast<struct stat*>(buf));
}

extern "C" void* eshkol_opendir(const char* path) {
    if (path == nullptr) {
        errno = EINVAL;
        return nullptr;
    }

    const auto normalized = normalize_runtime_path(path);
#ifdef _WIN32
    auto search_path = normalized;
    search_path /= "*";

    auto* dir = new eshkol::platform::windows_dir_handle();
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

    auto* dir = static_cast<eshkol::platform::windows_dir_handle*>(dir_raw);
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

    auto* dir = static_cast<eshkol::platform::windows_dir_handle*>(dir_raw);
    if (dir->handle != INVALID_HANDLE_VALUE) {
        ::FindClose(dir->handle);
    }
    delete dir;
    return 0;
}
#endif
