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

std::string native_target_triple() {
    if (const char* override = std::getenv("ESHKOL_TEST_TARGET_TRIPLE")) {
        if (*override) return override;
    }

#if defined(_WIN32) && defined(_M_ARM64)
    return "aarch64-pc-windows-msvc";
#elif defined(_WIN32) && defined(_M_X64)
    return "x86_64-pc-windows-msvc";
#elif defined(__APPLE__) && defined(__aarch64__)
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

#ifdef _WIN32
std::string quote_arg(const std::string& arg) {
    std::string quoted = "\"";
    for (char ch : arg) {
        if (ch == '\\' || ch == '"') quoted.push_back('\\');
        quoted.push_back(ch);
    }
    quoted.push_back('"');
    return quoted;
}

ProcessResult run_process_capture(const std::vector<std::string>& args,
                                  const fs::path& cwd) {
    ProcessResult result;
    if (args.empty()) return result;

    const fs::path previous_cwd = fs::current_path();
    fs::current_path(cwd);

    std::string command;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) command.push_back(' ');
        command += quote_arg(args[i]);
    }
    command += " 2>&1";

    FILE* pipe = _popen(command.c_str(), "r");
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
                                  const fs::path& cwd) {
    ProcessResult result;
    if (args.empty()) return result;

    int pipefd[2];
    if (pipe(pipefd) != 0) {
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
#endif

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

int expect_success(const std::string& label, const ProcessResult& result) {
    if (result.exit_code != 0) {
        return fail(label + " exited with code " + std::to_string(result.exit_code) +
                    "\n" + result.output);
    }
    return 0;
}

int expect_failure_containing(const std::string& label,
                              const ProcessResult& result,
                              const std::string& expected) {
    if (result.exit_code == 0) {
        return fail(label + " unexpectedly succeeded\n" + result.output);
    }
    if (!contains(result.output, expected)) {
        return fail(label + " did not contain expected diagnostic: " + expected +
                    "\n" + result.output);
    }
    return 0;
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

    const std::string target = native_target_triple();
    if (target.empty()) {
        return fail("no native target triple for this test platform");
    }

    std::error_code ec;
    const fs::path temp_root = fs::temp_directory_path() / "eshkol-profile-cli-test";
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root, ec);
    if (ec) {
        return fail("failed to create temp directory: " + ec.message());
    }

    const fs::path source_path = temp_root / "min.esk";
    {
        std::ofstream source(source_path);
        source << "(define (entry) 0)\n";
    }

    ProcessResult invalid_profile = run_process_capture(
        {run_binary.string(), "--profile", "not-a-profile", source_path.string()},
        temp_root);
    if (int rc = expect_failure_containing("invalid profile", invalid_profile,
                                           "Unknown execution profile")) {
        return rc;
    }
    if (!contains(invalid_profile.output, "freestanding-kernel-native")) {
        return fail("invalid profile diagnostic did not list supported profiles\n" +
                    invalid_profile.output);
    }
    if (!contains(invalid_profile.output, "embedded-vm")) {
        return fail("invalid profile diagnostic did not list embedded-vm\n" +
                    invalid_profile.output);
    }

    ProcessResult hosted_native_wasm = run_process_capture(
        {run_binary.string(), "--profile", "hosted-native", "--wasm", source_path.string()},
        temp_root);
    if (int rc = expect_failure_containing("hosted-native plus wasm",
                                           hosted_native_wasm,
                                           "--profile hosted-native cannot be combined with --wasm")) {
        return rc;
    }

    ProcessResult hosted_wasm_eval = run_process_capture(
        {run_binary.string(), "--profile", "hosted-wasm", "-e", "(display 1)"},
        temp_root);
    if (int rc = expect_failure_containing("hosted-wasm eval",
                                           hosted_wasm_eval,
                                           "--profile hosted-wasm does not support JIT eval/run")) {
        return rc;
    }

    ProcessResult freestanding_missing_target = run_process_capture(
        {run_binary.string(), "--profile", "freestanding-kernel-native",
         "-o", (temp_root / "missing-target.o").string(), source_path.string()},
        temp_root);
    if (int rc = expect_failure_containing("freestanding missing target",
                                           freestanding_missing_target,
                                           "requires --target <triple>")) {
        return rc;
    }

    const fs::path object_path = temp_root / "profile-object.o";
    ProcessResult freestanding_object = run_process_capture(
        {run_binary.string(), "--profile", "freestanding-kernel-native",
         "--target", target, "-o", object_path.string(), source_path.string()},
        temp_root);
    if (int rc = expect_success("freestanding object", freestanding_object)) {
        return rc;
    }
    if (!fs::exists(object_path)) {
        return fail("freestanding object was not created");
    }
    if (fs::exists(temp_root / "profile-object.o.o")) {
        return fail("freestanding object path appended .o unexpectedly");
    }

    ProcessResult embedded_missing_eskb = run_process_capture(
        {run_binary.string(), "--profile", "embedded-vm", source_path.string()},
        temp_root);
    if (int rc = expect_failure_containing("embedded VM missing ESKB",
                                           embedded_missing_eskb,
                                           "--profile embedded-vm requires --emit-eskb <path>")) {
        return rc;
    }

    const fs::path embedded_eskb_path = temp_root / "embedded.eskb";
    ProcessResult embedded_eskb = run_process_capture(
        {run_binary.string(), "--profile", "embedded-vm",
         "--emit-eskb", embedded_eskb_path.string(), source_path.string()},
        temp_root);
    if (int rc = expect_success("embedded VM ESKB emission", embedded_eskb)) {
        return rc;
    }
    if (!fs::exists(embedded_eskb_path)) {
        return fail("embedded VM ESKB file was not created");
    }
    if (fs::exists(temp_root / "a.out") || fs::exists(temp_root / "a.exe")) {
        return fail("embedded VM profile unexpectedly produced a native executable");
    }

    const fs::path required_eskb_path = temp_root / "embedded-required.eskb";
    ProcessResult embedded_required = run_process_capture(
        {run_binary.string(), "--profile", "embedded-vm",
         "--emit-eskb", required_eskb_path.string(),
         "--require-vm-entry", "entry", source_path.string()},
        temp_root);
    if (int rc = expect_success("embedded VM required entry admission",
                                embedded_required)) {
        return rc;
    }
    if (!fs::exists(required_eskb_path)) {
        return fail("embedded VM required-entry ESKB file was not created");
    }

    const fs::path zero_arg_required_path = temp_root / "embedded-zero-arg-required.eskb";
    ProcessResult embedded_zero_arg_required = run_process_capture(
        {run_binary.string(), "--profile", "embedded-vm",
         "--emit-eskb", zero_arg_required_path.string(),
         "--require-vm-entry-zero-arg", "entry", source_path.string()},
        temp_root);
    if (int rc = expect_success("embedded VM zero-arg required entry admission",
                                embedded_zero_arg_required)) {
        return rc;
    }
    if (!fs::exists(zero_arg_required_path)) {
        return fail("embedded VM zero-arg required-entry ESKB file was not created");
    }

    const fs::path missing_required_path = temp_root / "embedded-missing-required.eskb";
    ProcessResult embedded_missing_required = run_process_capture(
        {run_binary.string(), "--profile", "embedded-vm",
         "--emit-eskb", missing_required_path.string(),
         "--require-vm-entry", "tick", source_path.string()},
        temp_root);
    if (int rc = expect_failure_containing("embedded VM missing required entry",
                                           embedded_missing_required,
                                           "missing required VM entry 'tick'")) {
        return rc;
    }
    if (fs::exists(missing_required_path)) {
        return fail("embedded VM kept ESKB after missing required entry");
    }

    const fs::path argful_entry_source = temp_root / "argful-entry.esk";
    {
        std::ofstream source(argful_entry_source);
        source << "(define (tick dt) dt)\n";
    }

    const fs::path argful_required_path = temp_root / "embedded-argful-required.eskb";
    ProcessResult embedded_argful_required = run_process_capture(
        {run_binary.string(), "--profile", "embedded-vm",
         "--emit-eskb", argful_required_path.string(),
         "--require-vm-entry-zero-arg", "tick", argful_entry_source.string()},
        temp_root);
    if (int rc = expect_failure_containing("embedded VM rejects argumentful required entry",
                                           embedded_argful_required,
                                           "ESKB admission failed for profile embedded-vm")) {
        return rc;
    }
    if (fs::exists(argful_required_path)) {
        return fail("embedded VM kept ESKB after argumentful required entry");
    }

    const fs::path captured_entry_source = temp_root / "captured-entry.esk";
    {
        std::ofstream source(captured_entry_source);
        source << "(define scale 2)\n";
        source << "(define (tick) scale)\n";
    }

    const fs::path captured_required_path = temp_root / "hosted-captured-required.eskb";
    ProcessResult hosted_captured_required = run_process_capture(
        {run_binary.string(), "--profile", "hosted-vm",
         "--emit-eskb", captured_required_path.string(),
         "--require-vm-entry-zero-arg", "tick", captured_entry_source.string()},
        temp_root);
    if (int rc = expect_failure_containing("hosted VM rejects captured required entry",
                                           hosted_captured_required,
                                           "ESKB admission failed for profile hosted-vm")) {
        return rc;
    }
    if (fs::exists(captured_required_path)) {
        return fail("hosted VM kept ESKB after captured required entry");
    }

    const fs::path desktop_native_source = temp_root / "desktop-native.esk";
    {
        std::ofstream source(desktop_native_source);
        source << "(make-vector 1 0)\n";
    }

    ProcessResult embedded_desktop_native = run_process_capture(
        {run_binary.string(), "--profile", "embedded-vm",
         "--emit-eskb", (temp_root / "desktop-native.eskb").string(),
         desktop_native_source.string()},
        temp_root);
    if (int rc = expect_failure_containing("embedded VM desktop native rejection",
                                           embedded_desktop_native,
                                           "rejected desktop native call")) {
        return rc;
    }
    if (fs::exists(temp_root / "desktop-native.eskb")) {
        return fail("embedded VM wrote ESKB after rejecting desktop native bytecode");
    }

    fs::remove_all(temp_root, ec);
    std::cout << "PASS" << std::endl;
    return 0;
}
