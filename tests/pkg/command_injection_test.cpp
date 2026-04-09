#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

#ifdef _WIN32
std::wstring widen_utf8(const std::string& text) {
    if (text.empty()) return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (size <= 0) return {};
    std::wstring wide(static_cast<size_t>(size - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    return wide;
}

std::wstring build_command_line(const std::vector<std::string>& args) {
    std::wstring command_line;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) command_line.push_back(L' ');
        command_line.push_back(L'"');
        for (wchar_t ch : widen_utf8(args[i])) {
            if (ch == L'\\' || ch == L'"') {
                command_line.push_back(L'\\');
            }
            command_line.push_back(ch);
        }
        command_line.push_back(L'"');
    }
    return command_line;
}
#endif

int run_command(const std::vector<std::string>& args, const fs::path& cwd) {
    if (args.empty()) return 1;

#ifdef _WIN32
    std::wstring application = widen_utf8(args.front());
    std::wstring command_line = build_command_line(args);
    std::vector<wchar_t> mutable_command_line(command_line.begin(), command_line.end());
    mutable_command_line.push_back(L'\0');

    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info{};
    std::wstring working_dir = widen_utf8(cwd.string());

    if (!CreateProcessW(application.c_str(), mutable_command_line.data(), nullptr, nullptr,
                        FALSE, 0, nullptr, working_dir.c_str(), &startup_info, &process_info)) {
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
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        if (chdir(cwd.c_str()) != 0) _exit(125);
        execvp(argv[0], argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }
    if (pid < 0) return errno;

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return errno;
    }

    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 1;
#endif
}

std::string fake_compiler_path(const fs::path& dir) {
#ifdef _WIN32
    fs::path compiler = dir / "fake-compiler.bat";
    std::ofstream out(compiler);
    out << "@echo off\r\n";
    out << "exit /b 0\r\n";
    return compiler.string();
#else
    fs::path compiler = dir / "fake-compiler.sh";
    std::ofstream out(compiler);
    out << "#!/bin/sh\n";
    out << "exit 0\n";
    out.close();
    fs::permissions(compiler,
                    fs::perms::owner_exec | fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::replace);
    return compiler.string();
#endif
}

void set_compiler_env(const std::string& compiler) {
#ifdef _WIN32
    _putenv_s("ESHKOL_COMPILER", compiler.c_str());
#else
    setenv("ESHKOL_COMPILER", compiler.c_str(), 1);
#endif
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return fail("expected path to eshkol-pkg binary");

    const fs::path pkg_binary = fs::absolute(argv[1]);
    const fs::path temp_root = fs::temp_directory_path() / "eshkol-pkg-command-injection-test";
    const fs::path project_dir = temp_root / "project";
    const fs::path marker = temp_root / "marker";

    std::error_code ec;
    fs::remove_all(temp_root, ec);
    fs::create_directories(project_dir / "src");

    std::ofstream manifest(project_dir / "eshkol.toml");
    manifest << "[package]\n";
#ifdef _WIN32
    manifest << "name = \"%TEMP%\"\n";
#else
    manifest << "name = \"$(touch " << marker.string() << ")\"\n";
#endif
    manifest << "version = \"0.1.0\"\n";
    manifest << "entry = \"src/main.esk\"\n";
    manifest.close();

    std::ofstream source(project_dir / "src" / "main.esk");
    source << ";; command injection regression\n";
    source.close();

    const std::string compiler = fake_compiler_path(temp_root);
    set_compiler_env(compiler);

    const int exit_code = run_command({pkg_binary.string(), "build"}, project_dir);
    if (exit_code != 0) {
        return fail("eshkol-pkg build failed with exit code " + std::to_string(exit_code));
    }

#ifndef _WIN32
    if (fs::exists(marker)) {
        return fail("shell metacharacters in manifest data were executed");
    }
#endif

    std::cout << "PASS" << std::endl;
    fs::remove_all(temp_root, ec);
    return 0;
}
