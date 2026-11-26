/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U5 Gate (`os_u5_in_os_vm_test`):
 *   - Compiles real `.esk` expressions and modules to `.eskb` bytecode via
 *     `eshkol-run --profile embedded-vm --emit-eskb` and executes them inside
 *     the zero-LLVM, zero-POSIX, zero-malloc in-OS VM shell (`os_vm_shell.c`).
 *   - Exercises capability-gated host calls backed exclusively by
 *     `eshkol_os_syscall6` (`os_syscall_abi.h`): admitted capability calls
 *     succeed; ungranted, insufficient-rights, revoked, and desktop-native
 *     calls fail closed with explicit error codes.
 *   - Enforces deterministic instruction fuel limits (`ESHKOL_VM_FUEL_EXHAUSTED`
 *     on compiled infinite loops) and bounded OALR heap limits
 *     (`ESHKOL_VM_ERR_OOM` on allocation loops) with zero leaked pages and zero
 *     arena corruption on subsequent evaluations.
 *   - Audits `libeshkol-os-vm-shell.a` AND cross-compiled bare-metal ELF
 *     objects (`x86_64-unknown-none`, `aarch64-unknown-none`,
 *     `riscv64-unknown-none-elf`) via `llvm-nm -u` to prove ONLY
 *     `eshkol_os_syscall6` is referenced (ZERO LLVM/POSIX/glibc/malloc/kernel
 *     allocator symbols).
 */

#include <eshkol/core/os_syscall_abi.h>
#include <eshkol/core/os_vm_shell.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

int g_checks = 0;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

#define CHECK(cond, msg) do { \
    if (!(cond)) return fail(msg); \
    ++g_checks; \
} while (0)

/* ── Mock capability microkernel backing eshkol_os_syscall6 ── */

struct MockKernelState {
    uint64_t syscall_count = 0;
    uint64_t live_mapped_pages = 0;
    uint64_t total_pages_mapped = 0;
    uint64_t total_pages_unmapped = 0;
    uint64_t yields = 0;
    uint64_t last_exit_code = 0;
    eshkol_cap_id_t next_cap_id = 100;
    eshkol_cap_id_t kernel_revoked_cap = ESHKOL_CAP_ID_INVALID;
    std::string console_output;
    std::vector<uint64_t> seen_sysnos;
};

MockKernelState g_kernel;

}  // namespace

extern "C" int64_t eshkol_os_syscall6(uint64_t sysno,
                                      uint64_t arg0,
                                      uint64_t arg1,
                                      uint64_t arg2,
                                      uint64_t arg3,
                                      uint64_t arg4,
                                      uint64_t arg5) {
    (void)arg3;
    (void)arg4;
    (void)arg5;
    g_kernel.syscall_count++;
    g_kernel.seen_sysnos.push_back(sysno);

    if (arg0 != ESHKOL_CAP_ID_INVALID && arg0 == g_kernel.kernel_revoked_cap) {
        return ESHKOL_OS_ERR_REVOKED;
    }

    switch (sysno) {
    case ESHKOL_SYS_CAP_DERIVE:
        return static_cast<int64_t>(++g_kernel.next_cap_id);
    case ESHKOL_SYS_CAP_REVOKE:
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_CAP_DROP:
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_IPC_SUBMIT:
        return static_cast<int64_t>(arg2);
    case ESHKOL_SYS_IPC_COMPLETE:
        return static_cast<int64_t>(arg2 > 0 ? 1 : 0);
    case ESHKOL_SYS_IPC_WAIT:
        return static_cast<int64_t>(arg1);
    case ESHKOL_SYS_PAGE_MAP:
        g_kernel.live_mapped_pages += arg2;
        g_kernel.total_pages_mapped += arg2;
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_PAGE_UNMAP:
        if (arg2 > g_kernel.live_mapped_pages) return ESHKOL_OS_ERR_INVALID_ARG;
        g_kernel.live_mapped_pages -= arg2;
        g_kernel.total_pages_unmapped += arg2;
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_TASK_YIELD:
        g_kernel.yields++;
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_TASK_EXIT:
        g_kernel.last_exit_code = arg0;
        return static_cast<int64_t>(arg0);
    case ESHKOL_SYS_CONSOLE_WRITE:
        if (arg1 != 0 && arg2 > 0 && arg2 <= 4096) {
            const char* ptr = reinterpret_cast<const char*>(static_cast<uintptr_t>(arg1));
            g_kernel.console_output.append(ptr, static_cast<size_t>(arg2));
        }
        return static_cast<int64_t>(arg2);
    default:
        return ESHKOL_OS_ERR_INVALID_SYSCALL;
    }
}

namespace {

struct ProcessResult {
    int exit_code = 1;
    std::string output;
};

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

std::vector<uint8_t> read_binary(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>());
}

std::string find_tool(const std::string& llvm_dis, const std::string& name) {
    if (!llvm_dis.empty()) {
        fs::path sibling = fs::path(llvm_dis).parent_path() / name;
        if (fs::exists(sibling)) return sibling.string();
    }
    return name;
}

/* ── Minimal ESKB v2 assembler for targeted host-call / negative bytecode tests ── */

void append_leb128(std::vector<uint8_t>& out, uint64_t val) {
    do {
        uint8_t byte = static_cast<uint8_t>(val & 0x7Fu);
        val >>= 7;
        if (val) byte |= 0x80u;
        out.push_back(byte);
    } while (val);
}

void append_u32_le(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

void append_i64_le(std::vector<uint8_t>& out, int64_t v) {
    uint64_t u = static_cast<uint64_t>(v);
    for (int i = 0; i < 8; i++) out.push_back(static_cast<uint8_t>((u >> (8 * i)) & 0xFFu));
}

uint32_t calc_crc32(const std::vector<uint8_t>& data) {
    uint32_t crc = 0xFFFFFFFFu;
    for (uint8_t b : data) {
        crc ^= b;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
        }
    }
    return ~crc;
}

struct AsmInstr {
    uint8_t op;
    int32_t operand;
};

std::vector<uint8_t> assemble_eskb(const std::vector<int64_t>& int_consts,
                                   const std::vector<AsmInstr>& instrs) {
    std::vector<uint8_t> const_sec;
    append_leb128(const_sec, int_consts.size());
    for (int64_t v : int_consts) {
        const_sec.push_back(1u); /* ESKB_CONST_INT64 */
        append_i64_le(const_sec, v);
    }

    std::vector<uint8_t> code_sec;
    append_leb128(code_sec, 1u); /* 1 function: "main" */
    append_leb128(code_sec, 4u);
    code_sec.insert(code_sec.end(), {'m', 'a', 'i', 'n'});
    code_sec.push_back(0u);      /* n_params */
    append_leb128(code_sec, 0u); /* n_locals */
    append_leb128(code_sec, 0u); /* n_upvalues */
    append_leb128(code_sec, instrs.size());
    for (const auto& ins : instrs) {
        code_sec.push_back(ins.op);
        uint64_t zz = (static_cast<uint64_t>(static_cast<int64_t>(ins.operand) << 1)) ^
                      (static_cast<uint64_t>(static_cast<int64_t>(ins.operand) >> 63));
        append_leb128(code_sec, zz);
    }

    std::vector<uint8_t> payload;
    append_leb128(payload, 2u); /* 2 sections */
    payload.push_back(0u);      /* ESKB_SECTION_CONST */
    append_leb128(payload, const_sec.size());
    payload.push_back(1u);      /* ESKB_SECTION_CODE */
    append_leb128(payload, code_sec.size());
    payload.insert(payload.end(), const_sec.begin(), const_sec.end());
    payload.insert(payload.end(), code_sec.begin(), code_sec.end());

    std::vector<uint8_t> image;
    append_u32_le(image, 0x45534B42u); /* ESKB */
    append_u32_le(image, 2u);          /* version 2 */
    append_u32_le(image, 1u);          /* little-endian */
    append_u32_le(image, calc_crc32(payload));
    image.insert(image.end(), payload.begin(), payload.end());
    return image;
}

std::string parse_nm_undef_sym(const std::string& line) {
    std::istringstream iss(line);
    std::string t1, t2;
    if (!(iss >> t1)) return {};
    if (!t1.empty() && t1.back() == ':') return {};
    if (t1 == "U" || t1 == "u" || t1 == "w") {
        if (iss >> t2) return t2;
        return {};
    }
    if (iss >> t2) {
        if (t2 == "U" || t2 == "u" || t2 == "w") {
            std::string t3;
            if (iss >> t3) return t3;
        }
        return {};
    }
    /* Single-token undefined symbol line (e.g. llvm-nm -u on Mach-O/ELF archives) */
    return t1;
}

int audit_zero_llvm_posix_object(const std::string& llvm_nm,
                                 const fs::path& cwd,
                                 const std::string& label,
                                 const fs::path& obj_or_archive) {
    auto nm = run_process({llvm_nm, "-u", obj_or_archive.string()}, cwd);
    if (nm.exit_code != 0) {
        return fail("llvm-nm -u failed for " + label + ":\n" + nm.output);
    }
    bool saw_syscall6 = false;
    std::istringstream ss(nm.output);
    std::string line;
    while (std::getline(ss, line)) {
        std::string sym = parse_nm_undef_sym(line);
        if (sym.empty()) continue;
        if (sym.size() > 1 && sym[0] == '_') sym.erase(0, 1);
        if (sym != "eshkol_os_syscall6") {
            return fail("forbidden external symbol in " + label + ": '" + sym +
                        "' (raw: " + line + ")");
        }
        saw_syscall6 = true;
    }
    if (!saw_syscall6) {
        return fail("expected eshkol_os_syscall6 undefined symbol missing in " + label +
                    "\n" + nm.output);
    }
    ++g_checks;
    std::cout << "PASS symbol audit (" << label << "): ONLY eshkol_os_syscall6" << std::endl;
    return 0;
}

int compile_esk_to_eskb(const std::string& compiler,
                        const fs::path& dir,
                        const std::string& name,
                        const std::string& source,
                        std::vector<uint8_t>& out_bytes) {
    const fs::path src = dir / (name + ".esk");
    const fs::path eskb = dir / (name + ".eskb");
    if (!write_text(src, source)) return fail("failed to write " + src.string());
    auto res = run_process(
        {compiler, "--profile", "embedded-vm", "--emit-eskb", eskb.string(), src.string()},
        dir);
    if (res.exit_code != 0) {
        return fail("embedded-vm --emit-eskb failed for " + name + ":\n" + res.output);
    }
    out_bytes = read_binary(eskb);
    if (out_bytes.size() < 16u) {
        return fail("empty or truncated ESKB emitted for " + name);
    }
    ++g_checks;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        return fail("usage: os_u5_in_os_vm_test <eshkol-run> <llvm-dis> <repo-root> <os-vm-shell-lib>");
    }
    const std::string compiler = argv[1];
    const std::string llvm_dis = argv[2] ? argv[2] : "";
    const fs::path repo_root   = argv[3];
    const fs::path vm_lib_path = argv[4];
    const std::string llvm_nm  = find_tool(llvm_dis, "llvm-nm");

    const fs::path work_dir =
        fs::temp_directory_path() / ("eshkol_os_u5_vm_" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    /* ── 1. Initialize in-OS VM shell backed by mock PAGE_MAP / PAGE_UNMAP ── */
    alignas(4096) static uint8_t workspace[65536];
    eshkol_os_vm_shell_t shell;
    eshkol_os_vm_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.workspace = workspace;
    cfg.workspace_bytes = sizeof(workspace);
    cfg.space_cap = 10;
    cfg.page_size = 4096;
    cfg.console_cap = 30;
    cfg.console_rights = ESHKOL_CAP_RIGHT_WRITE;
    cfg.fuel_limit = 50000;
    cfg.max_heap_bytes = sizeof(workspace);

    CHECK(eshkol_os_vm_init(&shell, &cfg) == ESHKOL_VM_OK, "shell init failed");
    CHECK(g_kernel.live_mapped_pages == 16, "expected 16 workspace pages mapped (64KiB / 4KiB)");
    CHECK(eshkol_os_vm_has_cap(&shell, 30, ESHKOL_CAP_RIGHT_WRITE) == 1, "console cap should be active");

    /* Grant address-space, IPC ring, and root caps */
    CHECK(eshkol_os_vm_grant_cap(&shell, 10, ESHKOL_CAP_RIGHT_READ | ESHKOL_CAP_RIGHT_WRITE) == ESHKOL_VM_OK,
          "grant space cap");
    CHECK(eshkol_os_vm_grant_cap(&shell, 20, ESHKOL_CAP_RIGHT_SEND | ESHKOL_CAP_RIGHT_RECV |
                                              ESHKOL_CAP_RIGHT_GRANT | ESHKOL_CAP_RIGHT_REVOKE) == ESHKOL_VM_OK,
          "grant ring cap");
    CHECK(eshkol_os_vm_grant_cap(&shell, 21, ESHKOL_CAP_RIGHT_READ) == ESHKOL_VM_OK,
          "grant read-only cap");

    /* ── 2. Compile real .esk modules via eshkol-run --profile embedded-vm --emit-eskb ── */
    {
        std::vector<uint8_t> fact_eskb;
        if (compile_esk_to_eskb(
                compiler, work_dir, "fact",
                "(define (shell-entry)\n"
                "  (let fact ((n 10))\n"
                "    (if (<= n 1) 1 (* n (fact (- n 1))))))\n",
                fact_eskb) != 0) {
            return 1;
        }
        eshkol_os_vm_result_t res;
        CHECK(eshkol_os_vm_eval_eskb(&shell, fact_eskb.data(), fact_eskb.size(), "shell-entry", &res) == ESHKOL_VM_OK,
              "fact(10) eval failed");
        CHECK(res.value_kind == ESHKOL_OS_VM_VAL_INT && res.int_value == 3628800,
              "fact(10) expected 3628800");
        CHECK(res.heap_bytes_live == 0, "ephemeral OALR arena must rewind to 0 live bytes");
        CHECK(res.mapped_pages_live == 16, "workspace pages invariant");
        std::cout << "PASS real ESKB eval: fact(10) = " << res.int_value
                  << " (" << res.instructions_executed << " insns)" << std::endl;
    }

    {
        std::vector<uint8_t> tail_eskb;
        if (compile_esk_to_eskb(
                compiler, work_dir, "tail_sum",
                "(define (run-sum)\n"
                "  (let sum-loop ((n 100) (acc 0))\n"
                "    (if (<= n 0) acc (sum-loop (- n 1) (+ acc n)))))\n",
                tail_eskb) != 0) {
            return 1;
        }
        eshkol_os_vm_result_t res;
        CHECK(eshkol_os_vm_eval_eskb(&shell, tail_eskb.data(), tail_eskb.size(), "run-sum", &res) == ESHKOL_VM_OK,
              "tail sum(100) eval failed");
        CHECK(res.int_value == 5050, "tail sum(100) expected 5050");
        CHECK(res.heap_bytes_live == 0, "arena rewind after tail loop");
    }

    {
        std::vector<uint8_t> pairs_eskb;
        if (compile_esk_to_eskb(
                compiler, work_dir, "pairs",
                "(define (run-pairs) (let ((p (cons 19 23))) (+ (car p) (cdr p))))\n",
                pairs_eskb) != 0) {
            return 1;
        }
        eshkol_os_vm_result_t res;
        CHECK(eshkol_os_vm_eval_eskb(&shell, pairs_eskb.data(), pairs_eskb.size(), "run-pairs", &res) == ESHKOL_VM_OK,
              "pairs eval failed");
        CHECK(res.int_value == 42, "cons/car/cdr expected 42");
        CHECK(res.heap_bytes_peak > 0 && res.heap_bytes_live == 0,
              "cons allocated in bounded OALR arena and rewound cleanly");
    }

    /* ── 3. Capability-gated host calls via eshkol_os_syscall6 ── */
    {
        /* Admitted CAP_DERIVE(20, SEND|RECV, 77) -> IPC_SUBMIT(derived, 0x1000, 2, 0)
         * -> PAGE_MAP(10, 0x4000, 2, RW) -> PAGE_UNMAP(10, 0x4000, 2) -> TASK_YIELD -> PRINT */
        const uint64_t send_recv = ESHKOL_CAP_RIGHT_SEND | ESHKOL_CAP_RIGHT_RECV;
        auto cap_ok_eskb = assemble_eskb(
            {20, static_cast<int64_t>(send_recv), 77,
             0x1000, 2, 0,
             10, 0x4000, 2, static_cast<int64_t>(ESHKOL_PAGE_FLAG_READ | ESHKOL_PAGE_FLAG_WRITE),
             1337},
            {
                /* CAP_DERIVE(20, send_recv, 77) -> pushes derived cap (101) */
                {1, 0}, {1, 1}, {1, 2}, {37, ESHKOL_OS_VM_CALL_CAP_DERIVE},
                /* DUP derived cap, IPC_SUBMIT(derived, 0x1000, 2, 0) */
                {6, 0}, {1, 3}, {1, 4}, {1, 5}, {37, ESHKOL_OS_VM_CALL_IPC_SUBMIT},
                /* POP submit result, keep derived cap -> IPC_WAIT(derived, 1, 1000) */
                {5, 0}, {1, 4}, {1, 3}, {37, ESHKOL_OS_VM_CALL_IPC_WAIT},
                {5, 0},
                /* PAGE_MAP(10, 0x4000, 2, RW) + PAGE_UNMAP(10, 0x4000, 2) */
                {1, 6}, {1, 7}, {1, 8}, {1, 9}, {37, ESHKOL_OS_VM_CALL_PAGE_MAP}, {5, 0},
                {1, 6}, {1, 7}, {1, 8}, {37, ESHKOL_OS_VM_CALL_PAGE_UNMAP}, {5, 0},
                /* TASK_YIELD */
                {37, ESHKOL_OS_VM_CALL_TASK_YIELD}, {5, 0},
                /* OP_PRINT 1337 (routed through CONSOLE_WRITE cap 30) */
                {1, 10}, {6, 0}, {35, 0},
                {36, 0} /* HALT */
            });
        eshkol_os_vm_result_t res;
        g_kernel.console_output.clear();
        CHECK(eshkol_os_vm_eval_eskb(&shell, cap_ok_eskb.data(), cap_ok_eskb.size(), nullptr, &res) == ESHKOL_VM_OK,
              "admitted capability host calls failed");
        CHECK(res.int_value == 1337, "expected final stack value 1337");
        CHECK(res.syscalls_dispatched == 7, "expected 7 syscalls dispatched");
        CHECK(res.cap_denials == 0, "expected 0 cap denials");
        CHECK(g_kernel.console_output == "1337", "OP_PRINT must route through CONSOLE_WRITE");
        CHECK(g_kernel.live_mapped_pages == 16, "user PAGE_MAP/UNMAP must leave 0 leaked pages");
        CHECK(eshkol_os_vm_has_cap(&shell, 101, send_recv) == 1, "derived cap 101 registered with monotonic rights");
        CHECK(eshkol_os_vm_has_cap(&shell, 101, ESHKOL_CAP_RIGHT_GRANT) == 0, "derived cap must NOT retain unrequested GRANT right");
        std::cout << "PASS capability host calls: 7 syscalls dispatched, monotonic rights verified" << std::endl;
    }

    /* ── 4. Fail-closed capability & security negative controls ── */
    {
        /* 4a: Ungranted capability handle 999 -> ESHKOL_VM_ERR_CAP_DENIED, 0 syscalls */
        auto ungranted = assemble_eskb(
            {999, 0x1000, 1, 0},
            {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {37, ESHKOL_OS_VM_CALL_IPC_SUBMIT}, {36, 0}});
        eshkol_os_vm_result_t res;
        const uint64_t before_sc = g_kernel.syscall_count;
        CHECK(eshkol_os_vm_eval_eskb(&shell, ungranted.data(), ungranted.size(), nullptr, &res) == ESHKOL_VM_ERR_CAP_DENIED,
              "ungranted cap must fail closed with CAP_DENIED");
        CHECK(g_kernel.syscall_count == before_sc, "ungranted cap must never reach kernel syscall");
        CHECK(res.cap_denials == 1, "expected cap_denials == 1");

        /* 4b: Insufficient rights (cap 21 has READ only, lacks SEND) -> CAP_DENIED */
        auto low_rights = assemble_eskb(
            {21, 0x1000, 1, 0},
            {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {37, ESHKOL_OS_VM_CALL_IPC_SUBMIT}, {36, 0}});
        CHECK(eshkol_os_vm_eval_eskb(&shell, low_rights.data(), low_rights.size(), nullptr, &res) == ESHKOL_VM_ERR_CAP_DENIED,
              "insufficient rights cap must fail closed with CAP_DENIED");
        CHECK(g_kernel.syscall_count == before_sc, "insufficient rights must never reach kernel syscall");

        /* 4c: Revoked capability (shell-side CAP_REVOKE) -> subsequent call fails closed */
        CHECK(eshkol_os_vm_revoke_cap(&shell, 101) == ESHKOL_VM_OK, "revoke cap 101");
        auto revoked_call = assemble_eskb(
            {101, 0x1000, 1, 0},
            {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {37, ESHKOL_OS_VM_CALL_IPC_SUBMIT}, {36, 0}});
        CHECK(eshkol_os_vm_eval_eskb(&shell, revoked_call.data(), revoked_call.size(), nullptr, &res) == ESHKOL_VM_ERR_CAP_DENIED,
              "revoked cap must fail closed with CAP_DENIED");

        /* 4d: Kernel-side revocation feedback -> shell marks cap inactive and fails closed */
        CHECK(eshkol_os_vm_grant_cap(&shell, 202, ESHKOL_CAP_RIGHT_SEND) == ESHKOL_VM_OK, "grant cap 202");
        g_kernel.kernel_revoked_cap = 202;
        auto kern_rev = assemble_eskb(
            {202, 0x1000, 1, 0},
            {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {37, ESHKOL_OS_VM_CALL_IPC_SUBMIT}, {36, 0}});
        CHECK(eshkol_os_vm_eval_eskb(&shell, kern_rev.data(), kern_rev.size(), nullptr, &res) == ESHKOL_VM_ERR_CAP_DENIED,
              "kernel-revoked cap must return CAP_DENIED");
        CHECK(eshkol_os_vm_has_cap(&shell, 202, ESHKOL_CAP_RIGHT_SEND) == 0,
              "shell must deactivate cap after kernel ESHKOL_OS_ERR_REVOKED");
        g_kernel.kernel_revoked_cap = ESHKOL_CAP_ID_INVALID;

        /* 4e: Desktop-native OP_NATIVE_CALL (fid 580 = open-input-file) -> FORBIDDEN_NATIVE */
        auto desktop_native = assemble_eskb({0}, {{1, 0}, {37, 580}, {36, 0}});
        CHECK(eshkol_os_vm_eval_eskb(&shell, desktop_native.data(), desktop_native.size(), nullptr, &res) == ESHKOL_VM_ERR_FORBIDDEN_NATIVE,
              "desktop-native fid must fail closed with FORBIDDEN_NATIVE");

        /* 4f: Corrupt ESKB CRC32 -> CRC_MISMATCH */
        auto bad_crc = desktop_native;
        bad_crc.back() ^= 0x5Au;
        CHECK(eshkol_os_vm_eval_eskb(&shell, bad_crc.data(), bad_crc.size(), nullptr, &res) == ESHKOL_VM_ERR_CRC_MISMATCH,
              "corrupted ESKB must fail closed with CRC_MISMATCH");

        /* 4g: OP_PRINT after revoking console_cap -> CAP_DENIED */
        CHECK(eshkol_os_vm_revoke_cap(&shell, 30) == ESHKOL_VM_OK, "revoke console cap");
        auto print_eskb = assemble_eskb({7}, {{1, 0}, {35, 0}, {36, 0}});
        CHECK(eshkol_os_vm_eval_eskb(&shell, print_eskb.data(), print_eskb.size(), nullptr, &res) == ESHKOL_VM_ERR_CAP_DENIED,
              "OP_PRINT without console cap must fail closed");

        std::cout << "PASS fail-closed capability & ESKB security negative controls (7/7)" << std::endl;
    }

    /* ── 5. Deterministic instruction fuel limit & bounded OALR heap OOM ── */
    {
        /* 5a: Compiled infinite loop from real .esk -> ESHKOL_VM_FUEL_EXHAUSTED */
        std::vector<uint8_t> inf_eskb;
        if (compile_esk_to_eskb(
                compiler, work_dir, "infinite_loop",
                "(define (run-spin)\n"
                "  (let spin ((n 0)) (spin (+ n 1))))\n",
                inf_eskb) != 0) {
            return 1;
        }
        CHECK(eshkol_os_vm_set_fuel_limit(&shell, 4096) == ESHKOL_VM_OK, "set fuel limit 4096");
        eshkol_os_vm_result_t res;
        CHECK(eshkol_os_vm_eval_eskb(&shell, inf_eskb.data(), inf_eskb.size(), "run-spin", &res) == ESHKOL_VM_FUEL_EXHAUSTED,
              "infinite loop must terminate with ESHKOL_VM_FUEL_EXHAUSTED");
        CHECK(res.instructions_executed == 4096, "exact fuel exhaustion instruction count");
        CHECK(res.heap_bytes_live == 0, "zero live heap bytes after fuel exhaustion");
        CHECK(g_kernel.live_mapped_pages == 16, "zero leaked pages after fuel exhaustion");

        /* Shell must remain completely uncorrupted after fuel exhaustion */
        std::vector<uint8_t> healthy_eskb;
        if (compile_esk_to_eskb(
                compiler, work_dir, "post_fuel",
                "(define (recover) (+ 100 23))\n",
                healthy_eskb) != 0) {
            return 1;
        }
        CHECK(eshkol_os_vm_eval_eskb(&shell, healthy_eskb.data(), healthy_eskb.size(), "recover", &res) == ESHKOL_VM_OK,
              "shell must recover cleanly after fuel exhaustion");
        CHECK(res.int_value == 123, "post-fuel eval expected 123");

        /* 5b: Allocation bomb under tight bounded OALR heap -> ESHKOL_VM_ERR_OOM */
        std::vector<uint8_t> oom_eskb;
        if (compile_esk_to_eskb(
                compiler, work_dir, "alloc_bomb",
                "(define (run-grow)\n"
                "  (let grow ((xs '())) (grow (cons 1 xs))))\n",
                oom_eskb) != 0) {
            return 1;
        }
        alignas(4096) static uint8_t tiny_ws[8192];
        eshkol_os_vm_shell_t tiny_shell;
        eshkol_os_vm_config_t tiny_cfg;
        std::memset(&tiny_cfg, 0, sizeof(tiny_cfg));
        tiny_cfg.workspace = tiny_ws;
        tiny_cfg.workspace_bytes = sizeof(tiny_ws);
        tiny_cfg.space_cap = 10;
        tiny_cfg.page_size = 4096;
        tiny_cfg.fuel_limit = 50000;
        tiny_cfg.max_heap_bytes = 4096;
        CHECK(eshkol_os_vm_init(&tiny_shell, &tiny_cfg) == ESHKOL_VM_OK, "tiny shell init");
        CHECK(eshkol_os_vm_eval_eskb(&tiny_shell, oom_eskb.data(), oom_eskb.size(), "run-grow", &res) == ESHKOL_VM_ERR_OOM,
              "allocation bomb must fail cleanly with ESHKOL_VM_ERR_OOM");
        CHECK(res.heap_bytes_live == 0, "tiny arena rewound to 0 live bytes after OOM");
        {
            bool all_zero = true;
            for (size_t i = 0; i < sizeof(tiny_ws); ++i) {
                if (tiny_ws[i] != 0u) { all_zero = false; break; }
            }
            CHECK(all_zero, "every byte of bounded OALR workspace must be zeroed on OOM rewind");
        }
        CHECK(eshkol_os_vm_eval_eskb(&tiny_shell, healthy_eskb.data(), healthy_eskb.size(), "recover", &res) == ESHKOL_VM_OK &&
              res.int_value == 123, "tiny shell recovers cleanly after OOM");
        CHECK(eshkol_os_vm_destroy(&tiny_shell) == ESHKOL_VM_OK, "tiny shell destroy");
        CHECK(g_kernel.live_mapped_pages == 16, "tiny shell unmapped all its pages cleanly");

        std::cout << "PASS fuel exhaustion (4096 insns) & bounded OALR OOM with 0 leaked pages" << std::endl;
    }

    /* ── 6. Tear down primary shell and prove 0 leaked pages across entire lifecycle ── */
    CHECK(eshkol_os_vm_destroy(&shell) == ESHKOL_VM_OK, "primary shell destroy");
    CHECK(g_kernel.live_mapped_pages == 0, "all mapped workspace pages must be unmapped on destroy");
    CHECK(g_kernel.total_pages_mapped == g_kernel.total_pages_unmapped,
          "exact 1:1 page map/unmap ledger balance");

    /* ── 7. Zero-LLVM / Zero-POSIX / Zero-glibc / Zero-malloc symbol audit ── */
    if (audit_zero_llvm_posix_object(llvm_nm, work_dir, "host libeshkol-os-vm-shell.a", vm_lib_path) != 0) {
        return 1;
    }

    const fs::path src_c = repo_root / "lib" / "core" / "os_vm_shell.c";
    const fs::path inc_dir = repo_root / "inc";
    struct CrossTarget {
        std::string label;
        std::string triple;
        std::vector<std::string> extra_flags;
    };
    const std::vector<CrossTarget> targets = {
        {"x86_64", "x86_64-unknown-none", {}},
        {"aarch64", "aarch64-unknown-none", {}},
        {"rv64gc", "riscv64-unknown-none-elf", {"-march=rv64gc", "-mabi=lp64d"}},
    };
    for (const auto& t : targets) {
        const fs::path obj = work_dir / ("os_vm_shell_" + t.label + ".o");
        std::vector<std::string> cmd = {
            "clang", "--target=" + t.triple, "-std=c11", "-ffreestanding", "-fno-builtin",
            "-fno-stack-protector", "-O2", "-Wall", "-Wextra", "-Werror",
            "-I" + inc_dir.string()
        };
        cmd.insert(cmd.end(), t.extra_flags.begin(), t.extra_flags.end());
        cmd.push_back("-c");
        cmd.push_back(src_c.string());
        cmd.push_back("-o");
        cmd.push_back(obj.string());
        auto cc = run_process(cmd, work_dir);
        if (cc.exit_code != 0) {
            return fail("freestanding cross-compile of os_vm_shell.c failed for " + t.label + ":\n" + cc.output);
        }
        ++g_checks;
        if (audit_zero_llvm_posix_object(llvm_nm, work_dir, "bare-metal " + t.label + " ELF", obj) != 0) {
            return 1;
        }
    }

    fs::remove_all(work_dir, ec);
    std::cout << "PASS: os_u5_in_os_vm_test (" << g_checks << " checks)" << std::endl;
    return 0;
}
