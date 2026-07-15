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

/** @brief Resolve @p path to a canonical (symlink-free) absolute path if
 *  it exists on disk.
 *  @return Weakly-canonical form of @p path, falling back to
 *          std::filesystem::absolute() if canonicalization errors; an
 *          empty path if @p path does not exist. */
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
/** @brief (Windows) Convert a UTF-8 string to UTF-16 for Win32 wide APIs.
 *
 * Tries MultiByteToWideChar with CP_UTF8 first; if that fails (e.g. the
 * bytes are not valid UTF-8), falls back to the active code page (CP_ACP).
 * @return Wide-string conversion of @p value, or an empty wstring on
 *         failure (or if @p value is empty). */
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

/** @brief (Windows) Quote and backslash-escape a single argument per the
 *  Win32 CreateProcess command-line argument convention.
 *
 * Leaves the argument unquoted if it contains no whitespace or quote
 * characters; otherwise wraps it in double quotes, doubling backslashes
 * that immediately precede a quote (or the closing quote) as required by
 * the MSVC/Win32 argv parsing rules.
 * @param argument UTF-8 argument to quote.
 * @return Wide-string, quoted-as-needed form of @p argument (`""` if
 *         @p argument widens to empty). */
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

/** @brief (Windows) Join and quote a list of arguments into a single
 *  space-separated command line suitable for CreateProcessW.
 *  @param arguments Argument list (first entry is conventionally argv[0]). */
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

/** @brief Return the absolute path of the currently running executable.
 *
 * Platform-specific: uses GetModuleFileNameW on Windows,
 * _NSGetExecutablePath on macOS, and /proc/self/exe on Linux.
 * @return The executable's path, or an empty path if it cannot be
 *         determined (including on unsupported platforms). */
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

/** Return the directory containing the current executable (the parent of
 *  executable_path()), or empty if that path cannot be determined. */
std::filesystem::path executable_directory() {
    auto path = executable_path();
    if (path.empty()) {
        return {};
    }
    return path.parent_path();
}

/** Return the process's current working directory, or an empty path if
 *  it cannot be queried. */
std::filesystem::path current_directory() {
    std::error_code ec;
    auto cwd = std::filesystem::current_path(ec);
    return ec ? std::filesystem::path{} : cwd;
}

/** @brief Return the first of @p candidates that exists on disk, canonicalized.
 *  @return Canonical path string of the first existing candidate, or an
 *          empty string if none exist. */
std::string find_first_existing(const std::vector<std::filesystem::path>& candidates) {
    for (const auto& candidate : candidates) {
        auto resolved = canonical_if_exists(candidate);
        if (!resolved.empty()) {
            return resolved.string();
        }
    }
    return {};
}

/** @brief Determine the current user's home directory.
 *
 * Prefers $HOME; on Windows also tries %USERPROFILE% and %APPDATA%;
 * falls back to the current working directory if none are set.
 * @return Home directory path, or empty string if none could be found. */
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

/** Return whether standard input is connected to an interactive terminal. */
bool stdin_isatty() {
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

/** Return whether standard output is connected to an interactive terminal. */
bool stdout_isatty() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

/** @brief (Windows) Configure the console for interactive UTF-8 output.
 *
 * Sets the console output (and, if stdin is a TTY, input) code page to
 * UTF-8, and enables ANSI virtual terminal processing on stdout when
 * available. On non-Windows platforms this is a no-op that returns false.
 * @return false if stdout is not a TTY (or on non-Windows builds);
 *         otherwise the result of stdout_supports_utf8(). */
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

/** @brief Report whether stdout can be expected to render UTF-8 correctly.
 *  @return On Windows, true only if stdout is a TTY with its output code
 *          page set to CP_UTF8; on other platforms, always true. */
bool stdout_supports_utf8() {
#ifdef _WIN32
    return stdout_isatty() && GetConsoleOutputCP() == CP_UTF8;
#else
    return true;
#endif
}

/** @brief Generate a unique path in the system temp directory (or the
 *  current directory if the temp directory can't be determined).
 *
 * Combines @p stem with a random 64-bit value and an attempt counter,
 * retrying up to 128 times to avoid colliding with an existing file.
 * @param stem Base filename component.
 * @param extension Filename suffix (e.g. ".ll"), appended verbatim.
 * @return A path that did not exist at the time of the check (best
 *         effort; not reserved/created). */
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

/** @brief Return the configured host C++ compiler path (ESHKOL_HOST_CXX_COMPILER).
 *
 * On Windows, converts a leading drive-letter forward-slash path
 * (e.g. "C:/...") to backslashes for native tool compatibility. */
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

std::string cxx_driver_link_arg(std::string argument) {
#ifdef _WIN32
    if (argument.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(argument[0])) &&
        argument[1] == ':' &&
        argument[2] == '/') {
        std::replace(argument.begin(), argument.end(), '/', '\\');
    }

    const bool is_path =
        argument.find('/') != std::string::npos ||
        argument.find('\\') != std::string::npos;
    if (!is_path && argument.size() > 4) {
        const auto suffix_offset = argument.size() - 4;
        const bool is_bare_msvc_library =
            argument[suffix_offset] == '.' &&
            std::tolower(static_cast<unsigned char>(argument[suffix_offset + 1])) == 'l' &&
            std::tolower(static_cast<unsigned char>(argument[suffix_offset + 2])) == 'i' &&
            std::tolower(static_cast<unsigned char>(argument[suffix_offset + 3])) == 'b';
        if (is_bare_msvc_library) {
            argument.resize(suffix_offset);
            return "-l" + argument;
        }
    }
#endif
    return argument;
}

/** Return the configured host `llc` executable path (ESHKOL_HOST_LLC_EXECUTABLE). */
std::string llc_executable() {
    return ESHKOL_HOST_LLC_EXECUTABLE;
}

/** Return the platform's native executable file suffix (e.g. ".exe" on
 *  Windows, empty elsewhere), from ESHKOL_HOST_EXECUTABLE_SUFFIX. */
std::string executable_suffix() {
    return ESHKOL_HOST_EXECUTABLE_SUFFIX;
}

/** @brief Build a platform-native static library filename from @p stem
 *  (e.g. "libfoo.a" on POSIX, "foo.lib" on Windows), using the configured
 *  ESHKOL_HOST_STATIC_LIBRARY_PREFIX/SUFFIX. */
std::string static_library_name(std::string_view stem) {
    return std::string(ESHKOL_HOST_STATIC_LIBRARY_PREFIX) +
           std::string(stem) +
           std::string(ESHKOL_HOST_STATIC_LIBRARY_SUFFIX);
}

/** @brief Parse the configured ESHKOL_HOST_RUNTIME_LINK_ARGS (a
 *  semicolon-separated list) into individual linker argument strings.
 *
 * Skips empty items. On Windows, converts leading drive-letter
 * forward-slash paths (e.g. "C:/...") to backslashes.
 * @return The parsed list of linker arguments. */
std::vector<std::string> host_runtime_link_args() {
    std::vector<std::string> args;
    std::stringstream stream(ESHKOL_HOST_RUNTIME_LINK_ARGS);
    std::string item;

    while (std::getline(stream, item, ';')) {
        if (!item.empty()) {
            args.push_back(cxx_driver_link_arg(std::move(item)));
        }
    }

    return args;
}

/** @brief (macOS) Locate the active Xcode/Command Line Tools SDK's `usr/lib`
 *  directory by shelling out to `xcrun --show-sdk-path`.
 *
 * The result is computed once and cached in a function-local static.
 * @return The SDK lib directory path, or an empty string on non-Apple
 *         platforms or if `xcrun` fails / the directory doesn't exist. */
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

/** @brief Append the platform executable suffix to @p path if it doesn't
 *  already have it.
 *  @return @p path unchanged if empty, if there's no platform suffix, or
 *          if the extension already matches; otherwise @p path with the
 *          suffix appended. */
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

/** @brief Quote @p argument for safe inclusion in a shell command line.
 *
 * On Windows, wraps in double quotes and escapes embedded backslashes/
 * quotes. On POSIX, leaves the argument bare if it contains no characters
 * special to the shell, otherwise wraps it in single quotes (escaping any
 * embedded single quote as `'\''`). */
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

/** @brief Run an external process to completion and return its exit code.
 *
 * On Windows, launches @p arguments.front() via CreateProcessW with a
 * quoted command line built from the remaining arguments, waits for it
 * to exit, and returns its exit code (or the Win32 error code on launch
 * failure). On POSIX, shell-quotes and joins @p arguments and runs them
 * through std::system().
 * @param arguments Argument vector; arguments[0] is the program to run.
 * @return -1 if @p arguments is empty; otherwise the subprocess's exit
 *         status (platform-specific encoding). */
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

/** @brief Resolve the actual on-disk path of a build output that may or
 *  may not carry the platform executable suffix.
 *
 * Tries @p base_path with the executable suffix appended first (via
 * with_executable_suffix()), falling back to @p base_path as-is; each
 * candidate is canonicalized only if it exists.
 * @param base_path Requested output path, without or with the suffix.
 * @return Canonical resolved path if either candidate exists on disk;
 *         otherwise the suffixed (or original, if empty) candidate path
 *         unresolved. Empty if @p base_path is empty. */
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
