/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U4 Gate (`os_u4_userspace_elf`):
 *   - `--profile eshkol-os-user` cross-compiles a userspace service (`init` +
 *     IPC client/server using `core.freestanding_pure` + `core.os_cap`) to
 *     native ELF objects for x86_64 (`x86_64-unknown-none`), AArch64
 *     (`aarch64-unknown-none`), and RV64 (`riscv64gc-unknown-none-elf`).
 *   - LLVM IR (`llvm-dis`) and object symbol table (`llvm-nm -u`) inspection:
 *     assert ONLY `eshkol_os_syscall6` appears as an external symbol, with
 *     ZERO POSIX/glibc/hosted symbols (`open`, `read`, `write`, `mmap`,
 *     `ioctl`, `pthread_*`, `printf`, `eshkol_capability_enforce`) and NO
 *     forced kernel `noredzone`/`noimplicitfloat` attributes.
 *   - C11 ABI layout cross-check of `inc/eshkol/core/os_syscall_abi.h`
 *     (`sizeof(eshkol_ipc_sqe_t) == 64`, `sizeof(eshkol_ipc_cqe_t) == 32`,
 *     `sizeof(eshkol_os_syscall_result_t) == 16`) under
 *     `clang -std=c11 -Wall -Werror -Wextra -Wpedantic`, plus live host-ABI
 *     execution oracle verifying every `core.os_cap` wrapper dispatches the
 *     exact syscall numbers and arguments.
 *   - Fail-closed negative controls:
 *       1. `--profile eshkol-os-user` without `--target` -> rejected
 *       2. `--profile eshkol-os-user` importing Tier 3 `core.io` (`hosted-os`) -> rejected
 *       3. `--profile freestanding-kernel-native` importing `core.os_cap` (`os-userspace-oalr`) -> rejected
 *       4. `--profile eshkol-os-user` with `--wasm`, `--shared-lib`, `--lib`, or `-e` -> rejected
 *       5. `--profile eshkol-os-user` calling hosted builtin `parallel-execute` -> rejected
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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

int g_checks = 0;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::string host_target_triple() {
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

ProcessResult run_process(const std::vector<std::string>& args, const fs::path& cwd) {
    ProcessResult result;
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
        if (chdir(cwd.c_str()) != 0) _exit(125);
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }

    close(pipefd[1]);
    char buf[4096];
    ssize_t n = 0;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        result.output.append(buf, static_cast<size_t>(n));
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            result.exit_code = errno;
            return result;
        }
    }
    if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) result.exit_code = 128 + WTERMSIG(status);
    return result;
}

bool write_text(const fs::path& path, const std::string& content) {
    std::ofstream out(path);
    if (!out) return false;
    out << content;
    return out.good();
}

std::string find_tool(const std::string& llvm_dis, const std::string& name) {
    if (!llvm_dis.empty()) {
        fs::path sibling = fs::path(llvm_dis).parent_path() / name;
        if (fs::exists(sibling)) return sibling.string();
    }
    return name;
}

bool is_allowed_userspace_symbol(const std::string& sym) {
    return sym == "eshkol_os_syscall6";
}

bool is_forbidden_posix_symbol(const std::string& sym) {
    if (sym == "open" || sym == "read" || sym == "write" ||
        sym == "mmap" || sym == "munmap" || sym == "ioctl" ||
        sym == "printf" || sym == "fprintf" || sym == "puts" ||
        sym == "malloc" || sym == "free" || sym == "calloc" || sym == "realloc" ||
        sym == "eshkol_capability_enforce") {
        return true;
    }
    if (sym.rfind("pthread_", 0) == 0) return true;
    return false;
}

std::string extract_declared_symbol(const std::string& line) {
    const std::size_t decl = line.find("declare ");
    if (decl == std::string::npos) return {};
    const std::size_t at = line.find('@', decl);
    if (at == std::string::npos) return {};
    std::size_t start = at + 1;
    if (start < line.size() && line[start] == '"') {
        ++start;
        const std::size_t end = line.find('"', start);
        if (end == std::string::npos) return {};
        return line.substr(start, end - start);
    }
    std::size_t end = start;
    while (end < line.size() &&
           (std::isalnum(static_cast<unsigned char>(line[end])) ||
            line[end] == '_' || line[end] == '.' || line[end] == '$')) {
        ++end;
    }
    return line.substr(start, end - start);
}

std::string parse_nm_undefined_symbol(const std::string& line) {
    std::istringstream iss(line);
    std::string tok1, tok2;
    if (!(iss >> tok1)) return {};
    if (tok1 == "U" || tok1 == "u" || tok1 == "w") {
        if (!(iss >> tok2)) return {};
        return tok2;
    }
    if (iss >> tok2) {
        if (tok2 == "U" || tok2 == "u" || tok2 == "w") {
            std::string tok3;
            if (iss >> tok3) return tok3;
        }
    }
    return {};
}

int check_userspace_target(
    const std::string& compiler,
    const std::string& llvm_dis,
    const std::string& llvm_nm,
    const fs::path& dir,
    const std::string& label,
    const std::string& target,
    const fs::path& src) {
    const fs::path obj = dir / ("user_" + label + ".o");
    const fs::path bc = obj.string() + ".bc";
    auto res = run_process(
        {compiler, "--profile", "eshkol-os-user", "--target", target,
         "--emit-object", "-o", obj.string(), src.string()},
        dir);
    if (res.exit_code != 0) {
        return fail("eshkol-os-user compile failed for " + label + " (" + target + "):\n" + res.output);
    }
    if (!fs::exists(obj) || fs::file_size(obj) == 0) {
        return fail("missing userspace object output for " + label);
    }
    ++g_checks;

    if (!llvm_dis.empty() && fs::exists(bc)) {
        auto dis = run_process({llvm_dis, "-o", "-", bc.string()}, dir);
        if (dis.exit_code != 0) {
            return fail("llvm-dis failed for " + label + ":\n" + dis.output);
        }

        bool saw_syscall6 = false;
        std::istringstream ss(dis.output);
        std::string line;
        while (std::getline(ss, line)) {
            const std::string sym = extract_declared_symbol(line);
            if (sym.empty()) continue;
            if (sym.rfind("llvm.", 0) == 0) continue;
            if (is_forbidden_posix_symbol(sym)) {
                return fail("forbidden POSIX/hosted symbol declared in " + label + " IR: " + sym);
            }
            if (!is_allowed_userspace_symbol(sym)) {
                return fail("unexpected external declaration in " + label + " IR: " + line);
            }
            if (sym == "eshkol_os_syscall6") saw_syscall6 = true;
        }
        if (!saw_syscall6) {
            return fail("expected @eshkol_os_syscall6 declaration missing from " + label + " IR");
        }
        if (contains(dis.output, "eshkol_capability_enforce")) {
            return fail("hosted FFI capability gate leaked into " + label + " IR");
        }
        if (contains(dis.output, "noredzone") || contains(dis.output, "noimplicitfloat")) {
            return fail("kernel noredzone/noimplicitfloat attribute unexpectedly forced on " + label + " userspace IR");
        }
        if (!contains(dis.output, "nounwind")) {
            return fail("expected freestanding nounwind attribute missing from " + label + " IR");
        }
        ++g_checks;
    }

    auto nm = run_process({llvm_nm, "-u", obj.string()}, dir);
    if (nm.exit_code != 0) {
        return fail("llvm-nm -u failed for " + label + ":\n" + nm.output);
    }
    bool nm_saw_syscall6 = false;
    std::istringstream nm_ss(nm.output);
    std::string nm_line;
    while (std::getline(nm_ss, nm_line)) {
        std::string sym = parse_nm_undefined_symbol(nm_line);
        if (sym.empty()) continue;
        if (sym.size() > 1 && sym[0] == '_') sym.erase(0, 1);
        if (is_forbidden_posix_symbol(sym)) {
            return fail("forbidden POSIX/hosted undefined symbol in " + label + " object: " + sym);
        }
        if (!is_allowed_userspace_symbol(sym)) {
            return fail("unexpected undefined symbol in " + label + " object: " + sym + " (raw: " + nm_line + ")");
        }
        if (sym == "eshkol_os_syscall6") nm_saw_syscall6 = true;
    }
    if (!nm_saw_syscall6) {
        return fail("expected undefined symbol eshkol_os_syscall6 missing from " + label + " object\n" + nm.output);
    }
    ++g_checks;

    std::cout << "PASS userspace target: " << label << " (" << target << ")" << std::endl;
    return 0;
}

int check_negative_args(
    const fs::path& dir,
    const std::string& label,
    const std::vector<std::string>& args,
    const fs::path& forbidden_artifact,
    const std::string& expected_fragment) {
    std::error_code ec;
    if (!forbidden_artifact.empty()) {
        fs::remove(forbidden_artifact, ec);
    }
    auto res = run_process(args, dir);
    if (res.exit_code == 0) {
        return fail("negative control '" + label + "' unexpectedly succeeded\n" + res.output);
    }
    if (!expected_fragment.empty() && !contains(res.output, expected_fragment)) {
        return fail("negative control '" + label + "' rejected without expected diagnostic '" +
                    expected_fragment + "':\n" + res.output);
    }
    if (!forbidden_artifact.empty() && fs::exists(forbidden_artifact)) {
        return fail("negative control '" + label + "' left an artifact behind: " + forbidden_artifact.string());
    }
    ++g_checks;
    std::cout << "PASS negative: " << label << std::endl;
    return 0;
}

int check_negative_source(
    const std::string& compiler,
    const fs::path& dir,
    const std::string& label,
    const std::string& profile,
    const std::string& target,
    const std::string& source,
    const std::string& expected_fragment) {
    const fs::path src = dir / ("neg_" + label + ".esk");
    const fs::path obj = dir / ("neg_" + label + ".o");
    if (!write_text(src, source)) {
        return fail("unable to write negative test " + label);
    }
    return check_negative_args(
        dir, label,
        {compiler, "--profile", profile, "--target", target,
         "--emit-object", "-o", obj.string(), src.string()},
        obj, expected_fragment);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return fail("usage: os_u4_userspace_elf_test <eshkol-run> [llvm-dis] [inc-dir]");
    }
    const std::string compiler = argv[1];
    const std::string llvm_dis = (argc >= 3 && argv[2] && argv[2][0]) ? argv[2] : "";
    const std::string inc_dir  = (argc >= 4 && argv[3] && argv[3][0]) ? argv[3] : "";
    const std::string llvm_nm  = find_tool(llvm_dis, "llvm-nm");

    const fs::path work_dir =
        fs::temp_directory_path() / ("eshkol_os_u4_user_" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    // Userspace init + IPC client/server service using Tier 1 (core.freestanding_pure)
    // and Tier 2 (core.os_cap) with zero POSIX/glibc dependencies.
    const std::string userspace_service_src = R"ESK(
(require core.freestanding_pure)
(require core.os_cap)

(define (os-user-init (space-cap : u64) (ring-cap : u64) (console-cap : u64) (vaddr : u64)) : i64
  (let* ((aligned (pure-align-up-u64 vaddr 4096))
         (mapped  (os-page-map space-cap aligned 2 11))
         (rights  (bitwise-or (os-cap-right-send) (os-cap-right-recv)))
         (derived (os-cap-derive ring-cap rights 77))
         (logged  (os-console-write console-cap aligned 16)))
    (+ mapped (+ derived logged)))
  :export-symbol os_user_init)

(define (os-user-ipc-client-send (ring-cap : u64) (sqe-addr : u64) (count : u64)) : i64
  (let ((submitted (os-ipc-submit ring-cap sqe-addr count 0)))
    (if (< submitted 0)
        submitted
        (os-ipc-wait ring-cap 1 1000000)))
  :export-symbol os_user_ipc_client_send)

(define (os-user-ipc-server-poll (ring-cap : u64) (cqe-addr : u64) (max-count : u64)) : i64
  (let ((n (os-ipc-complete ring-cap cqe-addr max-count 0)))
    (if (= n 0)
        (os-task-yield)
        n))
  :export-symbol os_user_ipc_server_poll)

(define (os-user-cleanup (space-cap : u64) (vaddr : u64) (num-pages : u64)
                         (cap : u64) (exit-code : u64)) : i64
  (let* ((rev  (os-cap-revoke cap))
         (drp  (os-cap-drop cap))
         (unm  (os-page-unmap space-cap (pure-align-down-u64 vaddr 4096) num-pages))
         (ex   (os-task-exit exit-code)))
    (+ (+ rev drp) (+ unm ex)))
  :export-symbol os_user_cleanup)
)ESK";

    const fs::path svc_src_path = work_dir / "userspace_service.esk";
    if (!write_text(svc_src_path, userspace_service_src)) {
        return fail("failed to write userspace_service.esk");
    }

    // 1. Cross-compile userspace ELF objects across x86_64, AArch64, and RV64
    if (check_userspace_target(compiler, llvm_dis, llvm_nm, work_dir,
            "x86_64", "x86_64-unknown-none", svc_src_path) != 0)
        return 1;
    if (check_userspace_target(compiler, llvm_dis, llvm_nm, work_dir,
            "aarch64", "aarch64-unknown-none", svc_src_path) != 0)
        return 1;
    if (check_userspace_target(compiler, llvm_dis, llvm_nm, work_dir,
            "rv64", "riscv64gc-unknown-none-elf", svc_src_path) != 0)
        return 1;

    // 2. C11 ABI layout check of inc/eshkol/core/os_syscall_abi.h + live host-ABI
    //    execution oracle linking Eshkol userspace service .o with a mock kernel.
    const std::string host_triple = host_target_triple();
    if (!host_triple.empty() && !inc_dir.empty()) {
        const fs::path host_obj = work_dir / "user_host.o";
        auto host_compile = run_process(
            {compiler, "--profile", "eshkol-os-user", "--target", host_triple,
             "--emit-object", "-o", host_obj.string(), svc_src_path.string()},
            work_dir);
        if (host_compile.exit_code != 0) {
            return fail("host compile of userspace_service.esk failed:\n" + host_compile.output);
        }
        ++g_checks;

        const fs::path c_oracle = work_dir / "syscall_abi_oracle.c";
        const std::string c_src = R"C(
#include <eshkol/core/os_syscall_abi.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(ESHKOL_SYS_CAP_DERIVE == 1, "CAP_DERIVE sysno");
_Static_assert(ESHKOL_SYS_CAP_REVOKE == 2, "CAP_REVOKE sysno");
_Static_assert(ESHKOL_SYS_CAP_DROP == 3, "CAP_DROP sysno");
_Static_assert(ESHKOL_SYS_IPC_SUBMIT == 10, "IPC_SUBMIT sysno");
_Static_assert(ESHKOL_SYS_IPC_COMPLETE == 11, "IPC_COMPLETE sysno");
_Static_assert(ESHKOL_SYS_IPC_WAIT == 12, "IPC_WAIT sysno");
_Static_assert(ESHKOL_SYS_PAGE_MAP == 20, "PAGE_MAP sysno");
_Static_assert(ESHKOL_SYS_PAGE_UNMAP == 21, "PAGE_UNMAP sysno");
_Static_assert(ESHKOL_SYS_TASK_YIELD == 30, "TASK_YIELD sysno");
_Static_assert(ESHKOL_SYS_TASK_EXIT == 31, "TASK_EXIT sysno");
_Static_assert(ESHKOL_SYS_CONSOLE_WRITE == 40, "CONSOLE_WRITE sysno");

_Static_assert(sizeof(eshkol_ipc_sqe_t) == 64, "SQE must be 64 bytes");
_Static_assert(_Alignof(eshkol_ipc_sqe_t) == 64, "SQE must be 64-byte aligned");
_Static_assert(sizeof(eshkol_ipc_cqe_t) == 32, "CQE must be 32 bytes");
_Static_assert(_Alignof(eshkol_ipc_cqe_t) == 32, "CQE must be 32-byte aligned");
_Static_assert(sizeof(eshkol_os_syscall_result_t) == 16, "syscall result must be 16 bytes");

extern int64_t os_user_init(uint64_t space_cap, uint64_t ring_cap,
                            uint64_t console_cap, uint64_t vaddr);
extern int64_t os_user_ipc_client_send(uint64_t ring_cap, uint64_t sqe_addr, uint64_t count);
extern int64_t os_user_ipc_server_poll(uint64_t ring_cap, uint64_t cqe_addr, uint64_t max_count);
extern int64_t os_user_cleanup(uint64_t space_cap, uint64_t vaddr, uint64_t num_pages,
                               uint64_t cap, uint64_t exit_code);

typedef struct {
    uint64_t sysno;
    uint64_t a0, a1, a2, a3, a4, a5;
} syscall_log_t;

static syscall_log_t g_log[32];
static size_t g_log_count = 0;
static int g_complete_returns_zero = 1;

int64_t eshkol_os_syscall6(uint64_t sysno,
                           uint64_t arg0, uint64_t arg1, uint64_t arg2,
                           uint64_t arg3, uint64_t arg4, uint64_t arg5) {
    if (g_log_count < 32) {
        g_log[g_log_count].sysno = sysno;
        g_log[g_log_count].a0 = arg0;
        g_log[g_log_count].a1 = arg1;
        g_log[g_log_count].a2 = arg2;
        g_log[g_log_count].a3 = arg3;
        g_log[g_log_count].a4 = arg4;
        g_log[g_log_count].a5 = arg5;
        g_log_count++;
    }
    switch (sysno) {
        case ESHKOL_SYS_CAP_DERIVE:    return 101;
        case ESHKOL_SYS_CAP_REVOKE:    return 0;
        case ESHKOL_SYS_CAP_DROP:      return 0;
        case ESHKOL_SYS_IPC_SUBMIT:    return (int64_t)arg2;
        case ESHKOL_SYS_IPC_COMPLETE:  return g_complete_returns_zero ? 0 : 3;
        case ESHKOL_SYS_IPC_WAIT:      return 1;
        case ESHKOL_SYS_PAGE_MAP:      return 0;
        case ESHKOL_SYS_PAGE_UNMAP:    return 0;
        case ESHKOL_SYS_TASK_YIELD:    return 0;
        case ESHKOL_SYS_TASK_EXIT:     return (int64_t)arg0;
        case ESHKOL_SYS_CONSOLE_WRITE: return (int64_t)arg2;
        default:                       return ESHKOL_OS_ERR_INVALID_SYSCALL;
    }
}

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "ORACLE FAIL: %s\n", #expr); return 1; } } while (0)

int main(void) {
    eshkol_ipc_sqe_t sqe;
    eshkol_ipc_cqe_t cqe;
    memset(&sqe, 0, sizeof(sqe));
    memset(&cqe, 0, sizeof(cqe));

    /* 1. os_user_init: PAGE_MAP (aligned 4097->8192), CAP_DERIVE, CONSOLE_WRITE */
    int64_t r_init = os_user_init(10, 20, 30, 4097);
    CHECK(r_init == (0 + 101 + 16));
    CHECK(g_log_count == 3);
    CHECK(g_log[0].sysno == ESHKOL_SYS_PAGE_MAP && g_log[0].a0 == 10 && g_log[0].a1 == 8192 && g_log[0].a2 == 2 && g_log[0].a3 == 11);
    CHECK(g_log[1].sysno == ESHKOL_SYS_CAP_DERIVE && g_log[1].a0 == 20 &&
          g_log[1].a1 == (ESHKOL_CAP_RIGHT_SEND | ESHKOL_CAP_RIGHT_RECV) && g_log[1].a2 == 77);
    CHECK(g_log[2].sysno == ESHKOL_SYS_CONSOLE_WRITE && g_log[2].a0 == 30 && g_log[2].a1 == 8192 && g_log[2].a2 == 16);

    /* 2. os_user_ipc_client_send: IPC_SUBMIT + IPC_WAIT */
    int64_t r_send = os_user_ipc_client_send(20, (uint64_t)(uintptr_t)&sqe, 1);
    CHECK(r_send == 1);
    CHECK(g_log_count == 5);
    CHECK(g_log[3].sysno == ESHKOL_SYS_IPC_SUBMIT && g_log[3].a0 == 20 && g_log[3].a2 == 1);
    CHECK(g_log[4].sysno == ESHKOL_SYS_IPC_WAIT && g_log[4].a0 == 20 && g_log[4].a1 == 1 && g_log[4].a2 == 1000000);

    /* 3. os_user_ipc_server_poll (idle -> IPC_COMPLETE + TASK_YIELD, then busy -> IPC_COMPLETE) */
    g_complete_returns_zero = 1;
    CHECK(os_user_ipc_server_poll(20, (uint64_t)(uintptr_t)&cqe, 4) == 0);
    CHECK(g_log_count == 7);
    CHECK(g_log[5].sysno == ESHKOL_SYS_IPC_COMPLETE && g_log[5].a0 == 20 && g_log[5].a2 == 4);
    CHECK(g_log[6].sysno == ESHKOL_SYS_TASK_YIELD);

    g_complete_returns_zero = 0;
    CHECK(os_user_ipc_server_poll(20, (uint64_t)(uintptr_t)&cqe, 4) == 3);
    CHECK(g_log_count == 8);
    CHECK(g_log[7].sysno == ESHKOL_SYS_IPC_COMPLETE);

    /* 4. os_user_cleanup: CAP_REVOKE, CAP_DROP, PAGE_UNMAP (aligned down 8193->8192), TASK_EXIT */
    CHECK(os_user_cleanup(10, 8193, 2, 101, 42) == 42);
    CHECK(g_log_count == 12);
    CHECK(g_log[8].sysno == ESHKOL_SYS_CAP_REVOKE && g_log[8].a0 == 101);
    CHECK(g_log[9].sysno == ESHKOL_SYS_CAP_DROP && g_log[9].a0 == 101);
    CHECK(g_log[10].sysno == ESHKOL_SYS_PAGE_UNMAP && g_log[10].a0 == 10 && g_log[10].a1 == 8192 && g_log[10].a2 == 2);
    CHECK(g_log[11].sysno == ESHKOL_SYS_TASK_EXIT && g_log[11].a0 == 42);

    printf("C11 SYSCALL ABI & LIVE USERSPACE ORACLE: PASS (12 syscalls verified)\n");
    return 0;
}
)C";
        if (!write_text(c_oracle, c_src)) {
            return fail("failed to write syscall_abi_oracle.c");
        }
        const fs::path oracle_bin = work_dir / "syscall_abi_oracle";
        auto link = run_process(
            {"clang", "-std=c11", "-Wall", "-Werror", "-Wextra", "-Wpedantic",
             "-I" + inc_dir, c_oracle.string(), host_obj.string(),
             "-o", oracle_bin.string()},
            work_dir);
        if (link.exit_code != 0) {
            return fail("clang C11 ABI cross-check & link failed:\n" + link.output);
        }
        ++g_checks;

        auto run = run_process({oracle_bin.string()}, work_dir);
        if (run.exit_code != 0 || !contains(run.output, "PASS")) {
            return fail("live userspace syscall oracle failed:\n" + run.output);
        }
        ++g_checks;
        std::cout << run.output;
    }

    // 3. Negative Control 1: --profile eshkol-os-user without --target -> rejected
    if (check_negative_args(
            work_dir, "missing_target",
            {compiler, "--profile", "eshkol-os-user",
             "--emit-object", "-o", (work_dir / "neg_no_target.o").string(), svc_src_path.string()},
            work_dir / "neg_no_target.o",
            "requires --target <triple>") != 0)
        return 1;

    // 4. Negative Control 2: --profile eshkol-os-user importing Tier 3 core.io (hosted-os) -> rejected fail-closed
    if (check_negative_source(
            compiler, work_dir, "tier3_core_io_in_user",
            "eshkol-os-user", "x86_64-unknown-none",
            "(require core.io)\n(define (u (x : u64)) : u64 x :export-symbol u_entry)\n",
            "hosted-os") != 0)
        return 1;

    // 5. Negative Control 3: --profile freestanding-kernel-native importing core.os_cap (os-userspace-oalr) -> rejected fail-closed
    if (check_negative_source(
            compiler, work_dir, "core_os_cap_in_kernel",
            "freestanding-kernel-native", "x86_64-unknown-none",
            "(require core.os_cap)\n(define (k (x : u64)) : u64 x :no-alloc :export-symbol k_entry)\n",
            "os-userspace-oalr") != 0)
        return 1;

    // 6. Negative Control 4a: --profile eshkol-os-user with --wasm -> rejected
    if (check_negative_args(
            work_dir, "user_with_wasm",
            {compiler, "--profile", "eshkol-os-user", "--target", "x86_64-unknown-none",
             "--wasm", "-o", (work_dir / "neg_wasm.o").string(), svc_src_path.string()},
            work_dir / "neg_wasm.o",
            "cannot be combined with --wasm") != 0)
        return 1;

    // 7. Negative Control 4b: --profile eshkol-os-user with --shared-lib -> rejected
    if (check_negative_args(
            work_dir, "user_with_shared_lib",
            {compiler, "--profile", "eshkol-os-user", "--target", "x86_64-unknown-none",
             "--shared-lib", "-o", (work_dir / "neg_shared.o").string(), svc_src_path.string()},
            work_dir / "neg_shared.o",
            "cannot be combined with --shared-lib") != 0)
        return 1;

    // 8. Negative Control 4c: --profile eshkol-os-user with --lib -> rejected
    if (check_negative_args(
            work_dir, "user_with_lib",
            {compiler, "--profile", "eshkol-os-user", "--target", "x86_64-unknown-none",
             "--lib", "c", "-o", (work_dir / "neg_lib.o").string(), svc_src_path.string()},
            work_dir / "neg_lib.o",
            "cannot be combined with --lib") != 0)
        return 1;

    // 9. Negative Control 4d: --profile eshkol-os-user with --eval -> rejected
    if (check_negative_args(
            work_dir, "user_with_eval",
            {compiler, "--profile", "eshkol-os-user", "--target", "x86_64-unknown-none",
             "-e", "(+ 1 2)"},
            fs::path(),
            "does not support JIT eval/run") != 0)
        return 1;

    // 10. Negative Control 5: --profile eshkol-os-user calling hosted parallel builtin -> rejected fail-closed
    if (check_negative_source(
            compiler, work_dir, "hosted_builtin_in_user",
            "eshkol-os-user", "x86_64-unknown-none",
            "(define (bad-user) : null (begin (parallel-execute (lambda () 1)) (compiler-fence seq-cst)) :export-symbol bad_user)\n",
            "not available in freestanding codegen") != 0)
        return 1;

    fs::remove_all(work_dir, ec);
    std::cout << "PASS: os_u4_userspace_elf_test (" << g_checks << " checks)" << std::endl;
    return 0;
}
