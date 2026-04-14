#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <cstdio>
#include <cstdlib>
#else
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

struct ProcessResult {
    int exit_code = 1;
    std::string output;
};

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

#ifdef _WIN32
std::string quote_arg(const std::string& arg) {
    std::string quoted = "\"";
    for (char ch : arg) {
        if (ch == '\\' || ch == '"') {
            quoted.push_back('\\');
        }
        quoted.push_back(ch);
    }
    quoted.push_back('"');
    return quoted;
}

ProcessResult run_process_capture(const std::vector<std::string>& args,
                                  const fs::path& cwd,
                                  const fs::path& lib_dir,
                                  bool set_eshkol_path) {
    if (args.empty()) return {};

    if (set_eshkol_path) {
        _putenv_s("ESHKOL_PATH", lib_dir.string().c_str());
    } else {
        _putenv_s("ESHKOL_PATH", "");
    }

    const fs::path previous_cwd = fs::current_path();
    fs::current_path(cwd);

    std::string command;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) command.push_back(' ');
        command += quote_arg(args[i]);
    }
    command += " 2>&1";

    FILE* pipe = _popen(command.c_str(), "r");
    ProcessResult result;
    if (!pipe) {
        fs::current_path(previous_cwd);
        return result;
    }

    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result.output += buffer;
    }

    result.exit_code = _pclose(pipe);
    fs::current_path(previous_cwd);
    return result;
}
#else
ProcessResult run_process_capture(const std::vector<std::string>& args,
                                  const fs::path& cwd,
                                  const fs::path& lib_dir,
                                  bool set_eshkol_path) {
    if (args.empty()) return {};

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return {};
    }

    pid_t pid = fork();
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        if (chdir(cwd.c_str()) != 0) {
            _exit(125);
        }

        if (set_eshkol_path) {
            setenv("ESHKOL_PATH", lib_dir.string().c_str(), 1);
        } else {
            unsetenv("ESHKOL_PATH");
        }

        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        execvp(argv[0], argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }

    close(pipefd[1]);
    ProcessResult result;
    char buffer[4096];
    ssize_t count = 0;
    while ((count = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
        result.output.append(buffer, static_cast<size_t>(count));
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            result.exit_code = errno;
            return result;
        }
    }

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }

    return result;
}
#endif

bool output_has_assoc_success(const std::string& output) {
    return output.find("(W1 . 1)") != std::string::npos;
}

bool output_has_forward_ref_failure(const std::string& output) {
    return output.find("called a forward-referenced function that was never defined") !=
           std::string::npos;
}

int assert_assoc_eval(const std::string& label,
                      const ProcessResult& result,
                      bool expect_success) {
    if (expect_success) {
        if (result.exit_code != 0) {
            return fail(label + " exited with code " + std::to_string(result.exit_code) +
                        "\n" + result.output);
        }
        if (!output_has_assoc_success(result.output)) {
            return fail(label + " did not print assoc result\n" + result.output);
        }
        if (output_has_forward_ref_failure(result.output)) {
            return fail(label + " still hit forward-reference failure\n" + result.output);
        }
    } else {
        if (!output_has_forward_ref_failure(result.output)) {
            return fail(label + " was expected to miss stdlib and fail\n" + result.output);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        return fail("expected: <eshkol-run> <source-root> <stdlib.o>");
    }

    const fs::path run_binary = fs::absolute(argv[1]);
    const fs::path source_root = fs::absolute(argv[2]);
    const fs::path stdlib_object = fs::absolute(argv[3]);
    const fs::path lib_dir = source_root / "lib";

    if (!fs::exists(run_binary)) {
        return fail("eshkol-run binary not found: " + run_binary.string());
    }
    if (!fs::exists(lib_dir)) {
        return fail("lib directory not found: " + lib_dir.string());
    }
    if (!fs::exists(stdlib_object)) {
        return fail("stdlib.o not found: " + stdlib_object.string());
    }

    const std::vector<std::string> assoc_eval = {
        run_binary.string(),
        "-e",
        "(display (assoc \"W1\" (list (cons \"W1\" 1))))"
    };
    const fs::path build_dir = run_binary.parent_path();
    ProcessResult precompiled = run_process_capture(assoc_eval, build_dir, lib_dir, true);
    if (int rc = assert_assoc_eval("precompiled-stdlib eval", precompiled, true)) {
        return rc;
    }

    std::error_code ec;
    const fs::path temp_root = fs::temp_directory_path() / "eshkol-assoc-eval-test";
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root / "bin", ec);
    if (ec) {
        return fail("failed to create temp directory: " + ec.message());
    }

    const fs::path isolated_binary = temp_root / "bin" / run_binary.filename();
    fs::copy_file(run_binary, isolated_binary, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        return fail("failed to copy eshkol-run for source fallback test: " + ec.message());
    }

    ProcessResult source_fallback = run_process_capture(
        {isolated_binary.string(), "-e", assoc_eval[2]}, temp_root / "bin", lib_dir, true);
    if (int rc = assert_assoc_eval("source-fallback eval", source_fallback, true)) {
        return rc;
    }

    ProcessResult no_stdlib = run_process_capture(
        {isolated_binary.string(), "-n", "-e", assoc_eval[2]}, temp_root / "bin", lib_dir, false);
    if (int rc = assert_assoc_eval("no-stdlib eval", no_stdlib, false)) {
        return rc;
    }

    fs::remove_all(temp_root, ec);
    std::cout << "PASS" << std::endl;
    return 0;
}
