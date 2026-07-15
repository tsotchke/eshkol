/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef ESHKOL_PLATFORM_RUNTIME_H
#define ESHKOL_PLATFORM_RUNTIME_H

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace eshkol::platform {

/**
 * @brief Get the absolute path to the currently running executable.
 * @return Absolute path resolved via the platform-native API
 *         (GetModuleFileNameW on Windows, _NSGetExecutablePath on macOS,
 *         /proc/self/exe on Linux), or an empty path if it cannot be
 *         determined (including on unsupported platforms).
 */
std::filesystem::path executable_path();

/**
 * @brief Get the directory containing the currently running executable.
 * @return Parent directory of executable_path(), or an empty path if
 *         executable_path() could not be determined.
 */
std::filesystem::path executable_directory();

/**
 * @brief Get the process's current working directory.
 * @return Absolute current working directory, or an empty path on error.
 */
std::filesystem::path current_directory();

/**
 * @brief Find the first candidate path that exists on disk.
 * @param candidates Ordered list of paths to probe.
 * @return The canonicalized (or, if canonicalization fails, absolute) string
 *         form of the first candidate that exists, or an empty string if
 *         none of the candidates exist.
 */
std::string find_first_existing(const std::vector<std::filesystem::path>& candidates);

/**
 * @brief Get the current user's home directory.
 * @return Value of the `HOME` environment variable; on Windows, falls back
 *         to `USERPROFILE` then `APPDATA` if `HOME` is unset; if none of
 *         those are set, falls back to current_directory(); returns an
 *         empty string if nothing can be determined.
 */
std::string home_directory();

/**
 * @brief Check whether standard input is connected to an interactive
 *        terminal.
 * @return true if stdin is a TTY.
 */
bool stdin_isatty();

/**
 * @brief Check whether standard output is connected to an interactive
 *        terminal.
 * @return true if stdout is a TTY.
 */
bool stdout_isatty();

/**
 * @brief Configure the console for interactive UTF-8 input/output.
 *
 * On Windows, sets the console output (and, if stdin is a TTY, input) code
 * page to UTF-8 and enables ANSI virtual terminal processing on stdout when
 * available. No-op on other platforms.
 *
 * @return true if stdout ends up supporting UTF-8 after configuration
 *         (see stdout_supports_utf8()); false if stdout is not a TTY, or
 *         unconditionally false on non-Windows platforms.
 */
bool initialize_interactive_console();

/**
 * @brief Check whether stdout can currently render UTF-8 text.
 * @return On Windows, true only if stdout is a TTY and the console output
 *         code page is UTF-8; always true on other platforms.
 */
bool stdout_supports_utf8();

/**
 * @brief Generate a path to a not-currently-existing temporary file.
 *
 * Combines @p stem, a randomized numeric suffix, and @p extension inside the
 * system temporary directory (falling back to current_directory() if the
 * temp directory cannot be determined). No file is created by this call, so
 * the returned path is not guaranteed to remain unused afterward.
 *
 * @param stem      Filename prefix.
 * @param extension Filename suffix, including the leading dot (default
 *                  ".tmp").
 * @return Candidate temporary file path.
 */
std::filesystem::path make_temp_path(std::string_view stem, std::string_view extension = ".tmp");

/**
 * @brief Get the path to the C++ compiler used to build the host toolchain.
 * @return Absolute path to the compiler executable baked in at build time
 *         (ESHKOL_HOST_CXX_COMPILER), with forward slashes normalized to
 *         backslashes on Windows when the path is drive-letter rooted.
 */
std::string cxx_compiler();

/**
 * @brief Normalize one library/linker argument for the configured host C++
 *        compiler driver.
 *
 * On Windows Eshkol invokes the GNU-compatible `clang++` driver for generated
 * programs. A bare MSVC library token such as `winhttp.lib` is interpreted by
 * that driver as an input file in the current directory; `-lwinhttp` is the
 * portable driver form that searches the Visual Studio/Windows SDK paths
 * discovered by Clang. Absolute and relative paths are retained. On other
 * platforms the argument is returned unchanged.
 *
 * @param argument Raw configured link argument.
 * @return Driver-ready argument.
 */
std::string cxx_driver_link_arg(std::string argument);

/**
 * @brief Get the path to the `llc` executable bundled with the LLVM
 *        toolchain used at build time.
 * @return Absolute path baked in at build time (ESHKOL_HOST_LLC_EXECUTABLE).
 */
std::string llc_executable();

/**
 * @brief Get the platform's native executable filename suffix.
 * @return Suffix such as ".exe" on Windows, or an empty string on platforms
 *         with no executable extension.
 */
std::string executable_suffix();

/**
 * @brief Build the platform's native static library filename for a given
 *        library stem.
 * @param stem Base component of the library name, without prefix or suffix
 *             (e.g. "eshkol").
 * @return Platform-specific filename, e.g. "libeshkol.a" on Unix-like
 *         systems or "eshkol.lib" on Windows.
 */
std::string static_library_name(std::string_view stem);

/**
 * @brief Get the linker arguments required to link against the host
 *        runtime libraries.
 * @return Arguments parsed from the build-time
 *         ESHKOL_HOST_RUNTIME_LINK_ARGS list (semicolon-separated), with
 *         forward-slash paths normalized to backslashes on Windows.
 */
std::vector<std::string> host_runtime_link_args();
// On macOS, the resolved SDK library directory (<sdk>/usr/lib) discovered at
// runtime via `xcrun --show-sdk-path`. Adding it to the linker search path lets
// a bare `-lobjc` resolve on any mac, not just the builder. Empty on non-macOS
// or when xcrun is unavailable.
std::string macos_sdk_lib_dir();

/**
 * @brief Ensure a path carries the platform's native executable suffix.
 * @param path Base path, which may or may not already have the suffix.
 * @return An empty path if @p path is empty; @p path unchanged if the
 *         platform's executable suffix is empty or @p path's extension
 *         already matches it; otherwise @p path with the suffix appended.
 */
std::filesystem::path with_executable_suffix(const std::filesystem::path& path);

/**
 * @brief Quote a single argument for safe inclusion in a platform shell
 *        command line.
 * @param argument Raw, unescaped argument text.
 * @return The argument quoted/escaped for the platform's shell (cmd.exe
 *         double-quote escaping on Windows, POSIX single-quote escaping
 *         elsewhere); returned unmodified if it needs no escaping.
 */
std::string shell_quote(std::string_view argument);

/**
 * @brief Run an external command synchronously and wait for it to finish.
 * @param arguments Argv-style argument vector; `arguments[0]` is the
 *                  program path or name to launch.
 * @return The child process's exit code on success; -1 if @p arguments is
 *         empty; on Windows, a `GetLastError()`-derived code (e.g.
 *         `ERROR_INVALID_PARAMETER`) if the process could not be launched.
 * @note On non-Windows platforms this shells out via `std::system()` with
 *       each argument passed through shell_quote().
 */
int run_command(const std::vector<std::string>& arguments);

/**
 * @brief Resolve a base output path to the executable file that actually
 *        exists on disk.
 * @param base_path Desired output path, with or without the platform
 *                  executable suffix.
 * @return An empty path if @p base_path is empty; otherwise the canonical
 *         path of whichever of {@p base_path + suffix, @p base_path} exists
 *         first, or @p base_path + suffix if neither exists yet.
 */
std::filesystem::path resolve_executable_output(const std::filesystem::path& base_path);

} // namespace eshkol::platform

#endif // ESHKOL_PLATFORM_RUNTIME_H
