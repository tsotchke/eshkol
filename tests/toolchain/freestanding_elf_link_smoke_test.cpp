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

int skip(const std::string& message) {
    std::cout << "SKIP: " << message << std::endl;
    return 0;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

bool executable(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && !fs::is_directory(path, ec) &&
           access(path.c_str(), X_OK) == 0;
}

std::vector<fs::path> search_dirs() {
    std::vector<fs::path> dirs;
    if (const char* path = std::getenv("PATH")) {
        std::string value(path);
        size_t start = 0;
        while (start <= value.size()) {
            const size_t end = value.find(':', start);
            std::string dir = value.substr(start, end == std::string::npos
                                                      ? std::string::npos
                                                      : end - start);
            if (!dir.empty()) dirs.emplace_back(dir);
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }
    dirs.emplace_back("/opt/homebrew/opt/llvm/bin");
    dirs.emplace_back("/usr/local/opt/llvm/bin");
    dirs.emplace_back("/usr/bin");
    dirs.emplace_back("/bin");
    return dirs;
}

std::string find_tool(const char* env_name,
                      const std::vector<std::string>& names) {
    if (const char* override = std::getenv(env_name)) {
        if (*override) {
            fs::path path(override);
            if (executable(path)) return fs::absolute(path).string();
        }
    }

    for (const auto& dir : search_dirs()) {
        for (const auto& name : names) {
            fs::path path = dir / name;
            if (executable(path)) return path.string();
        }
    }
    return "";
}

std::string elf_target_triple(bool has_lld) {
    if (const char* override = std::getenv("ESHKOL_TEST_ELF_TARGET_TRIPLE")) {
        if (*override) return override;
    }

#if defined(__linux__) && defined(__aarch64__)
    return "aarch64-unknown-linux-gnu";
#elif defined(__linux__) && defined(__x86_64__)
    return "x86_64-unknown-linux-gnu";
#elif defined(__APPLE__) && defined(__aarch64__)
    return has_lld ? "aarch64-unknown-elf" : "";
#elif defined(__APPLE__) && defined(__x86_64__)
    return has_lld ? "x86_64-unknown-elf" : "";
#else
    (void)has_lld;
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
    if (argc != 2) {
        return fail("expected: <eshkol-run>");
    }

    const fs::path run_binary = fs::absolute(argv[1]);
    if (!fs::exists(run_binary)) {
        return fail("eshkol-run binary not found: " + run_binary.string());
    }

    std::string linker = find_tool("ESHKOL_TEST_ELF_LINKER", {"ld.lld"});
#if defined(__linux__)
    if (linker.empty()) {
        linker = find_tool("ESHKOL_TEST_ELF_LINKER", {"ld"});
    }
#endif
    if (linker.empty()) {
        return skip("no ELF linker found");
    }

    const bool has_lld = contains(fs::path(linker).filename().string(), "lld");
    const std::string target = elf_target_triple(has_lld);
    if (target.empty()) {
        return skip("no ELF target triple for this platform/linker");
    }

    const std::string readelf =
        find_tool("ESHKOL_TEST_READELF", {"llvm-readelf", "readelf"});
    if (readelf.empty()) {
        return skip("no readelf-compatible tool found");
    }

    const std::string nm = find_tool("ESHKOL_TEST_NM", {"llvm-nm", "nm"});
    if (nm.empty()) {
        return skip("no nm-compatible tool found");
    }

    std::error_code ec;
    const fs::path temp_root = fs::temp_directory_path() / "eshkol-freestanding-elf-link-test";
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root, ec);
    if (ec) {
        return fail("failed to create temp directory: " + ec.message());
    }

    const fs::path source_path = temp_root / "kernel.esk";
    {
        std::ofstream source(source_path);
        source << "(define (kernel-main) : null\n"
               << "  (compiler-fence seq-cst)\n"
               << "  :export-symbol kernel_entry)\n"
               << "(define (kernel-ready) : u32\n"
               << "  1\n"
               << "  :export-symbol kernel_ready)\n";
    }

    const fs::path script_path = temp_root / "kernel.ld";
    {
        std::ofstream script(script_path);
        script << "ENTRY(kernel_entry)\n"
               << "SECTIONS\n"
               << "{\n"
               << "  . = 0x100000;\n"
               << "  .text : { *(.text .text.*) }\n"
               << "  .rodata : { *(.rodata .rodata.*) }\n"
               << "  .data : { *(.data .data.*) }\n"
               << "  .bss : { *(.bss .bss.* COMMON) }\n"
               << "  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }\n"
               << "}\n";
    }

    const fs::path object_path = temp_root / "kernel.o";
    ProcessResult compile = run_process_capture(
        {run_binary.string(), "--profile", "freestanding-kernel-native",
         "--target", target, "-o", object_path.string(), source_path.string()},
        temp_root);
    if (int rc = expect_success("freestanding ELF object compile", compile)) {
        return rc;
    }
    if (!fs::exists(object_path)) {
        return fail("freestanding object was not created");
    }

    ProcessResult object_undefined =
        run_process_capture({nm, "-u", object_path.string()}, temp_root);
    if (int rc = expect_success("nm -u object", object_undefined)) {
        return rc;
    }
    if (!object_undefined.output.empty()) {
        return fail("freestanding object has undefined symbols\n" +
                    object_undefined.output);
    }

    const fs::path elf_path = temp_root / "kernel.elf";
    ProcessResult link = run_process_capture(
        {linker, "-T", script_path.string(), "-o", elf_path.string(),
         object_path.string()},
        temp_root);
    if (int rc = expect_success("freestanding ELF link", link)) {
        return rc;
    }
    if (!fs::exists(elf_path)) {
        return fail("freestanding ELF was not created");
    }

    ProcessResult header = run_process_capture({readelf, "-h", elf_path.string()}, temp_root);
    if (int rc = expect_success("readelf -h", header)) {
        return rc;
    }
    if (!contains(header.output, "ELF Header:")) {
        return fail("linked artifact is not an ELF file\n" + header.output);
    }
    if (!contains(header.output, "Entry point address:")) {
        return fail("linked ELF has no entry point in header\n" + header.output);
    }

    ProcessResult linked_undefined =
        run_process_capture({nm, "-u", elf_path.string()}, temp_root);
    if (int rc = expect_success("nm -u ELF", linked_undefined)) {
        return rc;
    }
    if (!linked_undefined.output.empty()) {
        return fail("linked ELF has undefined symbols\n" + linked_undefined.output);
    }

    ProcessResult symbols = run_process_capture({readelf, "-s", elf_path.string()}, temp_root);
    if (int rc = expect_success("readelf -s", symbols)) {
        return rc;
    }
    if (!contains(symbols.output, "kernel_entry")) {
        return fail("kernel_entry export missing from linked ELF\n" + symbols.output);
    }
    if (!contains(symbols.output, "kernel_ready")) {
        return fail("kernel_ready export missing from linked ELF\n" + symbols.output);
    }
    for (const auto& symbol : forbidden_symbols()) {
        if (contains(symbols.output, symbol)) {
            return fail("linked ELF leaked hosted symbol '" + symbol + "'\n" +
                        symbols.output);
        }
    }

    fs::remove_all(temp_root, ec);
    std::cout << "PASS" << std::endl;
    return 0;
}
