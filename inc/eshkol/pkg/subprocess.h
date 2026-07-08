/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * subprocess.h — header-only subprocess launcher used by eshkol-pkg and its
 * regression tests.
 *
 * Why this exists
 * ---------------
 * Earlier versions of eshkol-pkg shelled out via std::system(), which let
 * dependency-manifest data inject arbitrary shell commands. This header
 * replaces that path with a structured argv-based launcher:
 *
 *   • POSIX:   fork() + execvp() — every argv element is passed to the new
 *              process literally, with no shell interpretation.
 *   • Windows: CreateProcessW with lpApplicationName == nullptr so PATH
 *              search works for tools like git.exe / cl.exe; each argv
 *              element is pre-quoted into the command line. There is still
 *              no shell involvement.
 *
 * The launcher returns the exit code on success, or a small platform code:
 *   125 — chdir(cwd) failed (POSIX only, when cwd is given)
 *   126 — execvp returned with EACCES / similar
 *   127 — execvp returned with ENOENT (program not found)
 *   128+sig — child died via signal sig (POSIX only)
 *   124 — child exceeded the caller-supplied timeout and was killed
 *         (matches GNU coreutils `timeout(1)`)
 *
 * Both the production code and tests share this single implementation so
 * security-relevant fixes only have to be made in one place.
 *
 * Timeouts
 * --------
 * `run_subprocess` accepts an optional `timeout_seconds`. The default (0)
 * preserves the historical unbounded wait. A positive value bounds the wait:
 * on expiry the child is killed and SUBPROCESS_TIMEOUT (124) is returned, so a
 * stuck downstream tool (e.g. a hung `ld` on a misconfigured host) can never
 * wedge the parent — the caller fails fast instead of blocking forever.
 */

#ifndef ESHKOL_PKG_SUBPROCESS_H
#define ESHKOL_PKG_SUBPROCESS_H

#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <cerrno>
#  include <csignal>
#  include <ctime>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace eshkol::pkg {

/**
 * @brief Exit code returned by run_subprocess() when a bounded wait expires.
 *
 * Mirrors the convention used by GNU coreutils `timeout(1)`: if
 * `timeout_seconds` elapses before the child exits, the child is killed and
 * this value is returned instead of the child's real exit status.
 */
constexpr int SUBPROCESS_TIMEOUT = 124;

#ifdef _WIN32

/**
 * @brief Convert a UTF-8 encoded string to a UTF-16 wide string (Windows only).
 *
 * Uses MultiByteToWideChar with CP_UTF8. Needed because CreateProcessW and
 * related Win32 APIs take wide-character strings.
 *
 * @param text UTF-8 encoded input string
 * @return Equivalent UTF-16 string, or an empty string if `text` is empty or
 *         the conversion fails
 */
inline std::wstring widen_utf8(const std::string& text) {
    if (text.empty()) return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (size <= 0) return {};
    std::wstring wide(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    return wide;
}

/**
 * @brief Append a single argv element to a Windows command line, quoted per
 * the documented msvcrt parsing rules (Windows only).
 *
 * Quoting follows the rules CommandLineToArgvW uses to reverse the process,
 * so `arg` round-trips exactly. The interesting case is a run of N
 * backslashes followed by a quote: those backslashes get doubled (2N) and an
 * extra backslash is inserted before the quote; otherwise a run of
 * backslashes is left alone. See:
 *   https://learn.microsoft.com/en-us/cpp/cpp/main-function-command-line-args
 *
 * @param out Wide-character command line to append to
 * @param arg Single argv element (already UTF-16) to quote and append
 */
inline void append_quoted_windows_arg(std::wstring& out, const std::wstring& arg) {
    out.push_back(L'"');
    size_t i = 0;
    while (i < arg.size()) {
        size_t backslashes = 0;
        while (i < arg.size() && arg[i] == L'\\') {
            ++backslashes;
            ++i;
        }
        if (i == arg.size()) {
            out.append(backslashes * 2, L'\\');
            break;
        } else if (arg[i] == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(L'"');
            ++i;
        } else {
            out.append(backslashes, L'\\');
            out.push_back(arg[i]);
            ++i;
        }
    }
    out.push_back(L'"');
}

/**
 * @brief Build a full Windows command line string from an argv-style vector.
 *
 * Widens each UTF-8 argument to UTF-16 and quotes it with
 * append_quoted_windows_arg(), joining the results with single spaces, ready
 * to pass as `lpCommandLine` to CreateProcessW.
 *
 * @param args Argv-style argument list (args[0] is the program)
 * @return Windows command line string
 */
inline std::wstring build_windows_command_line(const std::vector<std::string>& args) {
    std::wstring command_line;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) command_line.push_back(L' ');
        append_quoted_windows_arg(command_line, widen_utf8(args[i]));
    }
    return command_line;
}

#endif // _WIN32

/**
 * @brief Launch a child process with an explicit argv, no shell involved.
 *
 * `args[0]` is the program to run; the remaining elements are passed as its
 * argv, exactly as given — on POSIX via fork()+execvp(), on Windows via
 * CreateProcessW with a hand-quoted command line. Because no shell parses
 * the arguments on either platform, data from a dependency manifest can
 * never inject additional commands.
 *
 * @param args           Argv-style argument list; args[0] is the program.
 *                       Must be non-empty.
 * @param cwd            If non-null, the child's working directory
 * @param timeout_seconds If non-zero, bounds the wait for the child; on
 *                       expiry the child is killed and SUBPROCESS_TIMEOUT is
 *                       returned. If zero (the default), waits unboundedly.
 * @return The child's exit code on normal completion, a platform-specific
 *         failure code (see the file-level comment for the POSIX codes), or
 *         SUBPROCESS_TIMEOUT if the timeout elapsed.
 */
inline int run_subprocess(const std::vector<std::string>& args,
                          const std::filesystem::path* cwd = nullptr,
                          unsigned int timeout_seconds = 0) {
    if (args.empty()) {
        return 1;
    }

#ifdef _WIN32
    std::wstring command_line = build_windows_command_line(args);
    std::vector<wchar_t> mutable_command_line(command_line.begin(), command_line.end());
    mutable_command_line.push_back(L'\0');

    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info{};

    std::wstring working_dir;
    LPCWSTR working_dir_ptr = nullptr;
    if (cwd) {
        working_dir = widen_utf8(cwd->string());
        working_dir_ptr = working_dir.c_str();
    }

    // lpApplicationName == nullptr so CreateProcessW parses the executable
    // from the front of lpCommandLine AND searches %PATH% (an explicit
    // application name disables PATH search per MSDN — that would break
    // `git`, `cl`, `clang`, and any other tool the user has on PATH).
    if (!CreateProcessW(nullptr,
                        mutable_command_line.data(),
                        nullptr,
                        nullptr,
                        FALSE,
                        0,
                        nullptr,
                        working_dir_ptr,
                        &startup_info,
                        &process_info)) {
        return static_cast<int>(GetLastError());
    }

    const DWORD wait_ms =
        timeout_seconds == 0 ? INFINITE : timeout_seconds * 1000u;
    DWORD wait_result = WaitForSingleObject(process_info.hProcess, wait_ms);
    if (wait_result == WAIT_TIMEOUT) {
        // Stuck child: terminate it and report a bounded-wait timeout so the
        // caller can fail fast rather than block forever.
        TerminateProcess(process_info.hProcess, 1);
        WaitForSingleObject(process_info.hProcess, INFINITE);
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
        return SUBPROCESS_TIMEOUT;
    }
    DWORD exit_code = 1;
    GetExitCodeProcess(process_info.hProcess, &exit_code);
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return static_cast<int>(exit_code);
#else
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        if (cwd && chdir(cwd->c_str()) != 0) {
            _exit(125);
        }
        execvp(argv[0], argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }
    if (pid < 0) {
        return errno;
    }

    int status = 0;
    if (timeout_seconds == 0) {
        // Historical unbounded wait.
        while (waitpid(pid, &status, 0) < 0) {
            if (errno != EINTR) {
                return errno;
            }
        }
    } else {
        // Bounded wait: poll with WNOHANG until the child exits or the deadline
        // passes, then SIGKILL a stuck child and reap it. Polling (rather than
        // SIGCHLD/alarm) keeps this header free of process-wide signal state,
        // which matters because run_subprocess is called from a long-lived host
        // process that installs its own handlers.
        const struct timespec poll_interval = {0, 10 * 1000 * 1000}; // 10ms
        const auto deadline =
            std::time(nullptr) + static_cast<std::time_t>(timeout_seconds);
        bool timed_out = false;
        for (;;) {
            pid_t r = waitpid(pid, &status, WNOHANG);
            if (r == pid) {
                break; // child exited
            }
            if (r < 0) {
                if (errno == EINTR) continue;
                return errno;
            }
            // r == 0: still running.
            if (std::time(nullptr) >= deadline) {
                timed_out = true;
                break;
            }
            nanosleep(&poll_interval, nullptr);
        }
        if (timed_out) {
            kill(pid, SIGKILL);
            // Reap so we don't leak a zombie.
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
            return SUBPROCESS_TIMEOUT;
        }
    }

    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
#endif
}

} // namespace eshkol::pkg

#endif // ESHKOL_PKG_SUBPROCESS_H
