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
 *
 * Both the production code and tests share this single implementation so
 * security-relevant fixes only have to be made in one place.
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
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace eshkol::pkg {

#ifdef _WIN32

inline std::wstring widen_utf8(const std::string& text) {
    if (text.empty()) return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (size <= 0) return {};
    std::wstring wide(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    return wide;
}

// Quote a single argv element using the documented msvcrt parsing rules so
// CommandLineToArgvW reverses it correctly. The interesting case is a run
// of N backslashes followed by a quote: those backslashes get doubled
// (2N) and an extra backslash is inserted before the quote, otherwise a
// run of backslashes is left alone. See:
//   https://learn.microsoft.com/en-us/cpp/cpp/main-function-command-line-args
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

inline std::wstring build_windows_command_line(const std::vector<std::string>& args) {
    std::wstring command_line;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) command_line.push_back(L' ');
        append_quoted_windows_arg(command_line, widen_utf8(args[i]));
    }
    return command_line;
}

#endif // _WIN32

// Launch `args[0]` with the rest of `args` as its argv. If `cwd` is non-null,
// the child runs with that working directory. No shell is involved on either
// platform — all elements are passed verbatim, so manifest data can never
// inject commands.
inline int run_subprocess(const std::vector<std::string>& args,
                          const std::filesystem::path* cwd = nullptr) {
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

    WaitForSingleObject(process_info.hProcess, INFINITE);
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
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return errno;
        }
    }

    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
#endif
}

} // namespace eshkol::pkg

#endif // ESHKOL_PKG_SUBPROCESS_H
