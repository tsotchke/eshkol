#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cerrno>
#include <cstdlib>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::string native_target_triple() {
    if (const char* override = std::getenv("ESHKOL_TEST_TARGET_TRIPLE")) {
        if (*override) return override;
    }

#if defined(__APPLE__) && defined(__aarch64__)
    return "arm64-apple-darwin";
#elif defined(__APPLE__) && defined(__x86_64__)
    return "x86_64-apple-darwin";
#elif defined(__linux__) && defined(__aarch64__)
    return "aarch64-unknown-linux-gnu";
#elif defined(__linux__) && defined(__x86_64__)
    return "x86_64-unknown-linux-gnu";
#else
    return "";
#endif
}

ProcessResult run_process_capture(const std::vector<std::string>& args,
                                  const fs::path& cwd) {
    ProcessResult result;
    if (args.empty()) return result;

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        result.output = "pipe failed";
        return result;
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

int expect_success(const std::string& label, const ProcessResult& result) {
    if (result.exit_code != 0) {
        return fail(label + " exited with code " + std::to_string(result.exit_code) +
                    "\n" + result.output);
    }
    return 0;
}

const std::vector<std::string>& forbidden_symbols() {
    static const std::vector<std::string> symbols = {
        "__eshkol_lib_init__",
        "__eshkol_register_parallel_workers",
        "__eshkol_call_",
        "__global_arena",
        "__parallel_",
        "eshkol_lambda_registry",
        "eshkol_runtime_init",
        "scheme_main",
        " printf",
        " _main",
        " main",
    };
    return symbols;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        return fail("expected: <eshkol-run> [llvm-dis]");
    }

    const fs::path run_binary = fs::absolute(argv[1]);
    if (!fs::exists(run_binary)) {
        return fail("eshkol-run binary not found: " + run_binary.string());
    }

    const std::string target = native_target_triple();
    if (target.empty()) {
        return fail("no native target triple for this test platform");
    }

    std::error_code ec;
    const fs::path temp_root = fs::temp_directory_path() / "eshkol-freestanding-lowlevel-test";
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root, ec);
    if (ec) {
        return fail("failed to create temp directory: " + ec.message());
    }

    const fs::path source_path = temp_root / "lowlevel.esk";
    {
        std::ofstream source(source_path);
        source
            << "(define (kernel-next) : ptr\n"
            << "  (ptr-add (usize->ptr 4096) 16)\n"
            << "  :export-symbol kernel_next)\n"
            << "(define (kernel-lowlevel) : null\n"
            << "  (begin\n"
            << "    (compiler-fence seq-cst)\n"
            << "    (memory-fence acquire)\n"
            << "    (volatile-store! u8 (usize->ptr 4096)\n"
            << "                     (volatile-load u8 (usize->ptr 4096)))\n"
            << "    (atomic-store! u16 (ptr-add (usize->ptr 8192) 2)\n"
            << "                   (atomic-load u16 (ptr-add (usize->ptr 8192) 2) acquire)\n"
            << "                   release)\n"
            << "    (compiler-fence release))\n"
            << "  :export-symbol kernel_lowlevel)\n";
    }

    const fs::path object_path = temp_root / "lowlevel.o";
    ProcessResult compile = run_process_capture(
        {run_binary.string(), "--profile", "freestanding-kernel-native",
         "--target", target, "-o", object_path.string(), source_path.string()},
        temp_root);
    if (int rc = expect_success("freestanding low-level object compile", compile)) {
        return rc;
    }
    if (!fs::exists(object_path)) {
        return fail("freestanding low-level object was not created");
    }

    ProcessResult undefined_symbols =
        run_process_capture({"nm", "-u", object_path.string()}, temp_root);
    if (int rc = expect_success("nm -u", undefined_symbols)) {
        return rc;
    }
    if (!undefined_symbols.output.empty()) {
        return fail("freestanding low-level object has undefined symbols\n" +
                    undefined_symbols.output);
    }

    ProcessResult global_symbols =
        run_process_capture({"nm", "-g", object_path.string()}, temp_root);
    if (int rc = expect_success("nm -g", global_symbols)) {
        return rc;
    }
    if (!contains(global_symbols.output, "kernel_next")) {
        return fail("kernel_next export missing\n" + global_symbols.output);
    }
    if (!contains(global_symbols.output, "kernel_lowlevel")) {
        return fail("kernel_lowlevel export missing\n" + global_symbols.output);
    }
    for (const auto& symbol : forbidden_symbols()) {
        if (contains(global_symbols.output, symbol)) {
            return fail("freestanding low-level object leaked hosted symbol '" + symbol +
                        "'\n" + global_symbols.output);
        }
    }

    if (argc == 3 && argv[2] && argv[2][0]) {
        const fs::path bitcode_path = object_path.string() + ".bc";
        if (!fs::exists(bitcode_path)) {
            return fail("freestanding low-level bitcode sidecar was not created");
        }

        ProcessResult ir = run_process_capture(
            {argv[2], bitcode_path.string(), "-o", "-"}, temp_root);
        if (int rc = expect_success("llvm-dis", ir)) {
            return rc;
        }
        if (contains(ir.output, "declare ")) {
            return fail("freestanding low-level bitcode has unused external declarations\n" +
                        ir.output);
        }
        const std::vector<std::string> required_ir = {
            "kernel_next",
            "kernel_lowlevel",
            "load volatile i8",
            "store volatile i8",
            "load atomic i16",
            "store atomic i16",
            "fence",
        };
        for (const auto& needle : required_ir) {
            if (!contains(ir.output, needle)) {
                return fail("freestanding low-level IR missing '" + needle +
                            "'\n" + ir.output);
            }
        }
        for (const auto& symbol : forbidden_symbols()) {
            if (contains(ir.output, symbol)) {
                return fail("freestanding low-level bitcode leaked hosted symbol '" +
                            symbol + "'\n" + ir.output);
            }
        }
    }

    fs::remove_all(temp_root, ec);
    std::cout << "PASS" << std::endl;
    return 0;
}
