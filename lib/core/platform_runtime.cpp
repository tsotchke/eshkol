/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <eshkol/platform_runtime.h>
#include <eshkol/build_config.h>

#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
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

std::string macos_sdk_lib_dir() {
#ifdef __APPLE__
    static const std::string cached = [] {
        // Trusted fixed command, no user input — popen is safe.
        FILE* pipe = popen("xcrun --show-sdk-path 2>/dev/null", "r");
        if (!pipe) {
            return std::string{};
        }
        std::string out;
        char buffer[512];
        while (std::fgets(buffer, sizeof(buffer), pipe)) {
            out += buffer;
        }
        int status = pclose(pipe);
        while (!out.empty() &&
               (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
            out.pop_back();
        }
        if (status != 0 || out.empty()) {
            return std::string{};
        }
        std::filesystem::path lib_dir = std::filesystem::path(out) / "usr" / "lib";
        std::error_code ec;
        if (!std::filesystem::is_directory(lib_dir, ec)) {
            return std::string{};
        }
        return lib_dir.string();
    }();
    return cached;
#else
    return {};
#endif
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
