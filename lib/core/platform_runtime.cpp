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
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <thread>
#include <system_error>

#ifdef _WIN32
#include <Windows.h>
#include <io.h>
#include <process.h>
#else
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
    return ESHKOL_HOST_CXX_COMPILER;
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
    std::vector<const char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const auto& argument : arguments) {
        argv.push_back(argument.c_str());
    }
    argv.push_back(nullptr);

    int result = _spawnv(_P_WAIT, arguments.front().c_str(), argv.data());
    if (result == -1) {
        return errno == 0 ? -1 : errno;
    }
    return result;
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

extern "C" FILE* eshkol_stdout_stream() {
    return stdout;
}

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
