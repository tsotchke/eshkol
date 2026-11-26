/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U6 Gate (`os_u6_native_self_host_test`):
 *   - Compiles Eshkol source inside the freestanding zero-LLVM, zero-malloc
 *     in-OS compiler (`os_self_host.c`) directly to CRC32-verified ESKB v2
 *     bytecode and executes it inside the Phase U5 in-OS VM shell
 *     (`os_vm_shell.c`).
 *   - Enforces fuel exhaustion negative control on compiled infinite loops at
 *     exact instruction ceilings (`500` and `1234`), asserting
 *     `ESHKOL_VM_FUEL_EXHAUSTED`, exact `instructions_executed == fuel_limit`,
 *     and `heap_bytes_live == 0`.
 *   - Enforces capability escalation negative controls on compiled
 *     `(os-syscall ...)` programs (ungranted capability, insufficient rights,
 *     and post-revocation denial), asserting `ESHKOL_VM_ERR_CAP_DENIED`,
 *     `cap_denials >= 1`, and zero dispatched syscalls.
 *   - Enforces byte-identical determinism across ESKB v2 and all three 64-bit
 *     ELF64 target ISAs (`x86_64`, `aarch64`, `riscv64`) under differing
 *     workspace poison fills (`0xAA` vs `0x55`), shifted workspace offsets,
 *     and interleaved unrelated compilations and VM evaluations.
 *   - Verifies emitted standalone static W^X ELF64 executables across x86_64,
 *     AArch64, and RISC-V 64 via binary header/phdr inspection, machine-code
 *     verification, `llvm-readelf`, and `llvm-objdump`.
 *   - Audits host `libeshkol-os-self-host.a` and bare-metal cross-compiled
 *     objects (`x86_64-unknown-none`, `aarch64-unknown-none`,
 *     `riscv64-unknown-none-elf`) via `llvm-nm -u` to prove ZERO undefined
 *     external symbols (zero allocator, POSIX, libc, or LLVM symbols).
 */

#include <eshkol/core/os_self_host.h>
#include <eshkol/core/os_syscall_abi.h>
#include <eshkol/core/os_vm_shell.h>

#include <cerrno>
#include <cmath>
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
    eshkol_cap_id_t next_cap_id = 300;
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
    (void)arg1;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    g_kernel.syscall_count++;

    switch (sysno) {
    case ESHKOL_SYS_CAP_DERIVE:
        return static_cast<int64_t>(++g_kernel.next_cap_id);
    case ESHKOL_SYS_CAP_REVOKE:
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
        return static_cast<int64_t>(arg0);
    case ESHKOL_SYS_CONSOLE_WRITE:
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

std::string find_tool(const std::string& llvm_dis, const std::string& name) {
    if (!llvm_dis.empty()) {
        fs::path sibling = fs::path(llvm_dis).parent_path() / name;
        if (fs::exists(sibling)) return sibling.string();
    }
    return name;
}

bool write_binary(const fs::path& path, const uint8_t* data, size_t len) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
    return out.good();
}

uint16_t rd_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(p[1] << 8);
}

uint32_t rd_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t rd_u64(const uint8_t* p) {
    return static_cast<uint64_t>(rd_u32(p)) |
           (static_cast<uint64_t>(rd_u32(p + 4)) << 32);
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
    return t1;
}

int audit_zero_external_symbols(const std::string& llvm_nm,
                                const fs::path& cwd,
                                const std::string& label,
                                const fs::path& obj_or_archive) {
    auto nm = run_process({llvm_nm, "-u", obj_or_archive.string()}, cwd);
    if (nm.exit_code != 0) {
        return fail("llvm-nm -u failed for " + label + ":\n" + nm.output);
    }
    std::istringstream ss(nm.output);
    std::string line;
    while (std::getline(ss, line)) {
        std::string sym = parse_nm_undef_sym(line);
        if (!sym.empty()) {
            return fail("unexpected external symbol in " + label + ": '" + sym +
                        "' (raw: " + line + ")");
        }
    }
    ++g_checks;
    std::cout << "PASS symbol audit (" << label << "): 0 undefined external symbols" << std::endl;
    return 0;
}

int verify_elf64_image(const uint8_t* buf,
                       size_t len,
                       eshkol_os_target_isa_t target,
                       const eshkol_os_compile_result_t& cres,
                       const std::string& llvm_readelf,
                       const std::string& llvm_objdump,
                       const fs::path& work_dir) {
    const std::string isa = eshkol_os_target_isa_name(target);
    CHECK(len >= 4096u + 64u, isa + ": ELF64 image truncated");
    CHECK(buf[0] == 0x7Fu && buf[1] == 'E' && buf[2] == 'L' && buf[3] == 'F',
          isa + ": invalid ELF magic");
    CHECK(buf[4] == 2u, isa + ": expected ELFCLASS64");
    CHECK(buf[5] == 1u, isa + ": expected ELFDATA2LSB");
    CHECK(buf[6] == 1u, isa + ": expected EV_CURRENT");

    const uint16_t e_type = rd_u16(buf + 16);
    const uint16_t e_machine = rd_u16(buf + 18);
    const uint64_t e_entry = rd_u64(buf + 24);
    const uint64_t e_phoff = rd_u64(buf + 32);
    const uint64_t e_shoff = rd_u64(buf + 40);
    const uint16_t e_phentsize = rd_u16(buf + 54);
    const uint16_t e_phnum = rd_u16(buf + 56);
    const uint16_t e_shentsize = rd_u16(buf + 58);
    const uint16_t e_shnum = rd_u16(buf + 60);

    const uint16_t expected_machine =
        (target == ESHKOL_OS_TARGET_X86_64) ? 62u :
        (target == ESHKOL_OS_TARGET_AARCH64) ? 183u : 243u;

    CHECK(e_type == 2u, isa + ": expected ET_EXEC");
    CHECK(e_machine == expected_machine, isa + ": unexpected e_machine");
    CHECK(e_entry == ESHKOL_OS_SH_ELF_TEXT_VADDR && e_entry == cres.entry_vaddr,
          isa + ": entry_vaddr mismatch");
    CHECK(e_phoff == 64u && e_phentsize == 56u && e_phnum == 2u,
          isa + ": unexpected phdr layout");
    CHECK(e_shoff > 0u && e_shentsize == 64u && e_shnum == 4u &&
              e_shoff + static_cast<uint64_t>(e_shnum) * 64u == len,
          isa + ": unexpected shdr layout");

    /* Strict W^X audit across every program header */
    bool saw_rx_text = false;
    bool saw_rw_data = false;
    uint64_t text_off = 0, text_sz = 0, text_va = 0, data_va = 0;
    for (uint16_t i = 0; i < e_phnum; i++) {
        const uint8_t* ph = buf + e_phoff + i * e_phentsize;
        const uint32_t p_type = rd_u32(ph + 0);
        const uint32_t p_flags = rd_u32(ph + 4);
        const uint64_t p_offset = rd_u64(ph + 8);
        const uint64_t p_vaddr = rd_u64(ph + 16);
        const uint64_t p_filesz = rd_u64(ph + 32);
        const uint64_t p_memsz = rd_u64(ph + 40);
        const uint64_t p_align = rd_u64(ph + 48);

        /* Zero PT_INTERP (3), zero PT_DYNAMIC (2), only PT_LOAD (1) */
        CHECK(p_type == 1u, isa + ": non-PT_LOAD program header present");
        /* Strict W^X: NEVER both PF_W (2) and PF_X (1) */
        CHECK((p_flags & 3u) != 3u, isa + ": W^X violation (PF_W | PF_X)");
        CHECK(p_align == ESHKOL_OS_SH_ELF_PAGE_SIZE, isa + ": expected 4KiB p_align");
        CHECK((p_offset % ESHKOL_OS_SH_ELF_PAGE_SIZE) == 0u &&
                  (p_vaddr % ESHKOL_OS_SH_ELF_PAGE_SIZE) == 0u,
              isa + ": segment not page-aligned");
        CHECK(p_offset + p_filesz <= len && p_filesz <= p_memsz,
              isa + ": segment bounds invalid");

        if (p_flags == 5u) { /* PF_R | PF_X */
            saw_rx_text = true;
            text_off = p_offset;
            text_sz = p_filesz;
            text_va = p_vaddr;
        } else if (p_flags == 6u) { /* PF_R | PF_W */
            saw_rw_data = true;
            data_va = p_vaddr;
        } else {
            return fail(isa + ": unexpected segment flags");
        }
    }
    CHECK(saw_rx_text && saw_rw_data, isa + ": expected separated RX .text and RW .data segments");
    CHECK(data_va >= text_va + text_sz &&
              (data_va - text_va) >= ESHKOL_OS_SH_ELF_PAGE_SIZE,
          isa + ": .text and .data must occupy disjoint pages");

    /* Direct machine-code encoding checks at _start entry stub */
    const uint8_t* code = buf + text_off;
    CHECK(text_sz >= 16u && text_sz == cres.text_bytes, isa + ": text_bytes mismatch");
    if (target == ESHKOL_OS_TARGET_X86_64) {
        /* _start: call rel32 (E8 ..), mov rdi, rax (48 89 C7), movabs rax, 31 (48 B8 1F 00..), syscall (0F 05) */
        CHECK(code[0] == 0xE8u, "x86_64: _start must open with call main");
        CHECK(code[5] == 0x48u && code[6] == 0x89u && code[7] == 0xC7u,
              "x86_64: expected mov rdi, rax");
        CHECK(code[8] == 0x48u && code[9] == 0xB8u &&
                  rd_u64(code + 10) == static_cast<uint64_t>(ESHKOL_SYS_TASK_EXIT),
              "x86_64: expected movabs rax, ESHKOL_SYS_TASK_EXIT (31)");
        CHECK(code[18] == 0x0Fu && code[19] == 0x05u,
              "x86_64: expected native syscall (0F 05)");
    } else if (target == ESHKOL_OS_TARGET_AARCH64) {
        /* _start: bl main, movz x8, #31 (0xD28003E8), svc #0 (0xD4000001), brk #0 */
        CHECK((rd_u32(code + 0) & 0xFC000000u) == 0x94000000u,
              "aarch64: _start must open with bl main");
        CHECK(rd_u32(code + 4) == 0xD28003E8u,
              "aarch64: expected movz x8, #31 (ESHKOL_SYS_TASK_EXIT)");
        CHECK(rd_u32(code + 8) == 0xD4000001u,
              "aarch64: expected native svc #0");
    } else {
        /* _start: jal ra, main, addi a7, zero, 31 (0x01F00893), ecall (0x00000073), ebreak */
        CHECK((rd_u32(code + 0) & 0xFFFu) == 0x0EFu,
              "riscv64: _start must open with jal ra, main");
        CHECK(rd_u32(code + 4) == 0x01F00893u,
              "riscv64: expected addi a7, zero, 31 (ESHKOL_SYS_TASK_EXIT)");
        CHECK(rd_u32(code + 8) == 0x00000073u,
              "riscv64: expected native ecall");
    }

    /* External LLVM tool validation (llvm-readelf + llvm-objdump) */
    const fs::path elf_path = work_dir / ("self_host_" + isa + ".elf");
    CHECK(write_binary(elf_path, buf, len), isa + ": failed writing ELF64 test file");

    auto re = run_process({llvm_readelf, "-h", "-l", "-S", elf_path.string()}, work_dir);
    if (re.exit_code != 0) {
        return fail(isa + ": llvm-readelf failed:\n" + re.output);
    }
    CHECK(re.output.find("EXEC") != std::string::npos, isa + ": readelf missing ET_EXEC");
    CHECK(re.output.find("INTERP") == std::string::npos, isa + ": readelf found forbidden INTERP");
    CHECK(re.output.find("DYNAMIC") == std::string::npos, isa + ": readelf found forbidden DYNAMIC");
    CHECK(re.output.find("RWE") == std::string::npos, isa + ": readelf found forbidden RWE segment");

    std::vector<std::string> od_cmd = {llvm_objdump, "-d"};
    if (target == ESHKOL_OS_TARGET_RISCV64) {
        od_cmd.push_back("--mattr=+m");
    }
    od_cmd.push_back(elf_path.string());
    auto od = run_process(od_cmd, work_dir);
    if (od.exit_code != 0) {
        return fail(isa + ": llvm-objdump -d failed:\n" + od.output);
    }
    CHECK(od.output.find("<unknown>") == std::string::npos,
          isa + ": llvm-objdump reported unknown instruction:\n" + od.output);
    if (target == ESHKOL_OS_TARGET_X86_64) {
        CHECK(od.output.find("syscall") != std::string::npos,
              "x86_64: objdump missing syscall:\n" + od.output);
    } else if (target == ESHKOL_OS_TARGET_AARCH64) {
        CHECK(od.output.find("svc") != std::string::npos,
              "aarch64: objdump missing svc:\n" + od.output);
    } else {
        CHECK(od.output.find("ecall") != std::string::npos,
              "riscv64: objdump missing ecall:\n" + od.output);
    }

    std::cout << "PASS W^X ELF64 (" << isa << "): " << len << " bytes, .text=" << text_sz
              << " bytes @ 0x" << std::hex << e_entry << std::dec << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        return fail("usage: os_u6_native_self_host_test <llvm-dis> <repo-root> <os-self-host-lib>");
    }
    const std::string llvm_dis     = argv[1] ? argv[1] : "";
    const fs::path repo_root       = argv[2];
    const fs::path self_host_lib   = argv[3];
    const std::string llvm_nm      = find_tool(llvm_dis, "llvm-nm");
    const std::string llvm_readelf = find_tool(llvm_dis, "llvm-readelf");
    const std::string llvm_objdump = find_tool(llvm_dis, "llvm-objdump");

    const fs::path work_dir =
        fs::temp_directory_path() / ("eshkol_os_u6_sh_" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    /* ── Initialize Phase U5 in-OS VM shell for live self-host execution ── */
    alignas(4096) static uint8_t vm_workspace[65536];
    eshkol_os_vm_shell_t shell;
    eshkol_os_vm_config_t vm_cfg;
    std::memset(&vm_cfg, 0, sizeof(vm_cfg));
    vm_cfg.workspace = vm_workspace;
    vm_cfg.workspace_bytes = sizeof(vm_workspace);
    vm_cfg.space_cap = 10;
    vm_cfg.page_size = 4096;
    vm_cfg.console_cap = 30;
    vm_cfg.console_rights = ESHKOL_CAP_RIGHT_WRITE;
    vm_cfg.fuel_limit = 50000;
    vm_cfg.max_heap_bytes = sizeof(vm_workspace);

    CHECK(eshkol_os_vm_init(&shell, &vm_cfg) == ESHKOL_VM_OK, "U5 VM shell init failed");
    CHECK(g_kernel.live_mapped_pages == 16, "expected 16 mapped VM workspace pages");
    CHECK(eshkol_os_vm_grant_cap(&shell, 10, ESHKOL_CAP_RIGHT_READ | ESHKOL_CAP_RIGHT_WRITE) == ESHKOL_VM_OK,
          "grant space cap 10");
    CHECK(eshkol_os_vm_grant_cap(&shell, 20, ESHKOL_CAP_RIGHT_SEND | ESHKOL_CAP_RIGHT_RECV |
                                              ESHKOL_CAP_RIGHT_GRANT | ESHKOL_CAP_RIGHT_REVOKE) == ESHKOL_VM_OK,
          "grant IPC cap 20");
    CHECK(eshkol_os_vm_grant_cap(&shell, 21, ESHKOL_CAP_RIGHT_READ) == ESHKOL_VM_OK,
          "grant read-only cap 21");

    alignas(16) static uint8_t compiler_ws[131072];
    static uint8_t out_eskb[32768];
    static uint8_t out_elf[32768];

    /* ── 1. End-to-end self-hosted ESKB v2 compilation & live U5 VM execution ── */
    {
        const std::string fib_src =
            "(define (fib n)\n"
            "  (if (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))\n"
            "(define (fact n)\n"
            "  (if (<= n 1) 1 (* n (fact (- n 1)))))\n"
            "(+ (fib 10) (fact 7))\n";
        eshkol_os_compile_result_t cres;
        CHECK(eshkol_os_compile_to_eskb(
                  fib_src.data(), fib_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile fib+fact to ESKB failed");
        CHECK(cres.bytes_written > 16u && cres.function_count == 3u,
              "expected 3 functions (main, fib, fact)");

        /* Confirm workspace was cleanly zeroed after compilation */
        bool ws_clean = true;
        for (size_t i = 0; i < cres.workspace_peak_bytes; ++i) {
            if (compiler_ws[i] != 0u) { ws_clean = false; break; }
        }
        CHECK(ws_clean, "compiler workspace must rewind to zero after compile");

        eshkol_os_vm_result_t vres;
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_OK,
              "VM eval of self-hosted fib(10)+fact(7) failed");
        CHECK(vres.value_kind == ESHKOL_OS_VM_VAL_INT && vres.int_value == (55 + 5040),
              "expected fib(10) + fact(7) == 5095");
        CHECK(vres.heap_bytes_live == 0, "expected 0 live heap bytes after fib+fact");
        std::cout << "PASS self-host ESKB eval: fib(10)+fact(7) = " << vres.int_value
                  << " (" << vres.instructions_executed << " VM insns)" << std::endl;
    }

    {
        /* let, let*, named let, begin, comparisons, arithmetic, pairs, float literals */
        const std::string rich_src =
            "(define base 10)\n"
            "(define (sum-to n)\n"
            "  (let loop ((i n) (acc 0))\n"
            "    (if (<= i 0) acc (loop (- i 1) (+ acc i)))))\n"
            "(let* ((a (sum-to 10))\n"
            "       (b (let ((p (cons 19 23))) (+ (car p) (cdr p))))\n"
            "       (c (begin (+ 1 2) (modulo (+ a b base) 100))))\n"
            "  (if (and (> c 0) (not (= c 999))) (+ c (abs (- 3))) 0))\n";
        eshkol_os_compile_result_t cres;
        CHECK(eshkol_os_compile_to_eskb(
                  rich_src.data(), rich_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile rich Eshkol program failed");
        eshkol_os_vm_result_t vres;
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_OK,
              "VM eval of rich Eshkol program failed");
        /* a=55, b=42, base=10 -> (55+42+10)%100 = 7 -> 7 + 3 = 10 */
        CHECK(vres.int_value == 10 && vres.heap_bytes_live == 0,
              "rich Eshkol program expected 10, got " + std::to_string(vres.int_value) +
                  " kind=" + std::to_string(vres.value_kind));

        const std::string float_src = "(+ 1.5 (* 2.0 1.125))\n";
        CHECK(eshkol_os_compile_to_eskb(
                  float_src.data(), float_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile float literal program failed");
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_OK,
              "VM eval of float literal program failed");
        CHECK(vres.value_kind == ESHKOL_OS_VM_VAL_FLOAT &&
                  std::fabs(vres.float_value - 3.75) < 1e-12,
              "float eval expected 3.75");
    }

    {
        /* Admitted capability syscalls compiled from Eshkol (os-syscall ...) */
        const std::string cap_ok_src =
            "(let* ((derived (os-syscall 1 20 24 77))\n"
            "       (sub     (os-syscall 10 derived 4096 2 0))\n"
            "       (m       (os-syscall 20 10 16384 2 3))\n"
            "       (u       (os-syscall 21 10 16384 2))\n"
            "       (y       (os-syscall 30)))\n"
            "  (+ derived sub m u y))\n";
        eshkol_os_compile_result_t cres;
        CHECK(eshkol_os_compile_to_eskb(
                  cap_ok_src.data(), cap_ok_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile admitted os-syscall program failed");
        eshkol_os_vm_result_t vres;
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_OK,
              "VM eval of admitted os-syscall program failed");
        CHECK(vres.int_value == 303 && vres.syscalls_dispatched == 5 && vres.cap_denials == 0,
              "expected derived(301) + sub(2) = 303 across 5 syscalls");
        CHECK(g_kernel.live_mapped_pages == 16, "user PAGE_MAP/UNMAP must leave 0 leaked pages");
        std::cout << "PASS self-host capability syscalls: 5 syscalls dispatched, 0 denials" << std::endl;
    }

    /* ── 2. Rigor Guard #1: Fuel Exhaustion Negative Control (exact ceilings 500 & 1234) ── */
    {
        const std::string spin_src =
            "(define (spin x) (spin (+ x 1)))\n"
            "(spin 0)\n";
        eshkol_os_compile_result_t cres;
        CHECK(eshkol_os_compile_to_eskb(
                  spin_src.data(), spin_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile infinite-loop spin failed");

        const uint64_t ceilings[2] = {500ULL, 1234ULL};
        for (uint64_t limit : ceilings) {
            CHECK(eshkol_os_vm_set_fuel_limit(&shell, limit) == ESHKOL_VM_OK,
                  "set fuel limit failed");
            eshkol_os_vm_result_t vres;
            eshkol_os_vm_status_t vst = eshkol_os_vm_eval_eskb(
                &shell, out_eskb, cres.bytes_written, nullptr, &vres);
            CHECK(vst == ESHKOL_VM_FUEL_EXHAUSTED,
                  "infinite loop must halt with ESHKOL_VM_FUEL_EXHAUSTED");
            CHECK(vres.instructions_executed == limit,
                  "instructions_executed must equal exact fuel_limit");
            CHECK(vres.heap_bytes_live == 0,
                  "heap_bytes_live must be 0 after fuel exhaustion");
            CHECK(g_kernel.live_mapped_pages == 16,
                  "mapped workspace pages must remain balanced after fuel exhaustion");
        }

        /* Restore fuel limit and verify healthy post-exhaustion execution */
        CHECK(eshkol_os_vm_set_fuel_limit(&shell, 50000ULL) == ESHKOL_VM_OK, "restore fuel");
        const std::string healthy = "(+ 40 2)\n";
        CHECK(eshkol_os_compile_to_eskb(
                  healthy.data(), healthy.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile healthy recovery program");
        eshkol_os_vm_result_t vres;
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_OK &&
                  vres.int_value == 42,
              "VM shell must recover cleanly after fuel exhaustion");
        std::cout << "PASS Rigor Guard #1 (Fuel Exhaustion): exact halts at 500/500 and 1234/1234 insns, heap_bytes_live=0" << std::endl;
    }

    /* ── 3. Rigor Guard #2: Capability Escalation Negative Control ── */
    {
        /* 3a: Ungranted capability 999 -> ESHKOL_VM_ERR_CAP_DENIED, 0 syscalls */
        const std::string ungranted_src = "(os-syscall 10 999 4096 1 0)\n";
        eshkol_os_compile_result_t cres;
        CHECK(eshkol_os_compile_to_eskb(
                  ungranted_src.data(), ungranted_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile ungranted os-syscall");
        const uint64_t sc_before_ungranted = g_kernel.syscall_count;
        eshkol_os_vm_result_t vres;
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_ERR_CAP_DENIED,
              "ungranted capability must fail closed with ESHKOL_VM_ERR_CAP_DENIED");
        CHECK(vres.cap_denials >= 1 && vres.syscalls_dispatched == 0 &&
                  g_kernel.syscall_count == sc_before_ungranted,
              "ungranted capability must dispatch 0 syscalls");

        /* 3b: Insufficient rights (cap 21 has READ only, lacks SEND) -> CAP_DENIED */
        const std::string low_rights_src = "(os-syscall 10 21 4096 1 0)\n";
        CHECK(eshkol_os_compile_to_eskb(
                  low_rights_src.data(), low_rights_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile insufficient-rights os-syscall");
        const uint64_t sc_before_low = g_kernel.syscall_count;
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_ERR_CAP_DENIED,
              "insufficient-rights capability must fail closed");
        CHECK(vres.cap_denials >= 1 && vres.syscalls_dispatched == 0 &&
                  g_kernel.syscall_count == sc_before_low,
              "insufficient-rights capability must dispatch 0 syscalls");

        /* 3c: Granted cap 20 succeeds before revocation, then fails closed after revocation */
        const std::string revoke_src = "(os-syscall 10 20 4096 1 0)\n";
        CHECK(eshkol_os_compile_to_eskb(
                  revoke_src.data(), revoke_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_OK,
              "compile cap-20 os-syscall");
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_OK &&
                  vres.syscalls_dispatched == 1 && vres.cap_denials == 0,
              "cap 20 should succeed before revocation");

        CHECK(eshkol_os_vm_revoke_cap(&shell, 20) == ESHKOL_VM_OK, "revoke cap 20");
        const uint64_t sc_before_revoked = g_kernel.syscall_count;
        CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, cres.bytes_written, nullptr, &vres) == ESHKOL_VM_ERR_CAP_DENIED,
              "revoked cap 20 must fail closed with ESHKOL_VM_ERR_CAP_DENIED");
        CHECK(vres.cap_denials >= 1 && vres.syscalls_dispatched == 0 &&
                  g_kernel.syscall_count == sc_before_revoked,
              "revoked cap 20 must dispatch 0 syscalls");

        /* 3d: Invalid syscall number rejected fail-closed at compile time */
        const std::string bad_sys_src = "(os-syscall 9999 10 0)\n";
        CHECK(eshkol_os_compile_to_eskb(
                  bad_sys_src.data(), bad_sys_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_ERR_INVALID_SYSCALL,
              "invalid syscall number must fail closed at compile time");

        std::cout << "PASS Rigor Guard #2 (Capability Escalation): ungranted/low-rights/post-revoke all ESHKOL_VM_ERR_CAP_DENIED, syscalls_dispatched=0" << std::endl;
    }

    /* ── 4. Rigor Guard #3: Byte-Identical Determinism (0xAA vs 0x55 poison + shifted offset + interleaved noise) ── */
    {
        const std::string det_src =
            "(define (fib n)\n"
            "  (if (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))\n"
            "(define (fact n)\n"
            "  (if (<= n 1) 1 (* n (fact (- n 1)))))\n"
            "(define (mix a b)\n"
            "  (let ((s (+ a b))\n"
            "        (d (- a b)))\n"
            "    (begin (os-syscall 30) (+ (* s 2) (abs d)))))\n"
            "(mix (fib 8) (fact 5))\n";

        alignas(16) static uint8_t ws_a[131072 + 64];
        alignas(16) static uint8_t ws_b[131072 + 64];
        std::vector<uint8_t> eskb_a(32768), eskb_b(32768);
        std::vector<uint8_t> elf_a[3], elf_b[3];
        eshkol_os_compile_result_t eskb_res_a, eskb_res_b;
        eshkol_os_compile_result_t elf_res_a[3], elf_res_b[3];

        const eshkol_os_target_isa_t isas[3] = {
            ESHKOL_OS_TARGET_X86_64,
            ESHKOL_OS_TARGET_AARCH64,
            ESHKOL_OS_TARGET_RISCV64
        };

        /* Pass A: workspace poisoned with 0xAA, offset +0, output poisoned with 0xEE */
        std::memset(ws_a, 0xAA, sizeof(ws_a));
        std::memset(eskb_a.data(), 0xEE, eskb_a.size());
        CHECK(eshkol_os_compile_to_eskb(
                  det_src.data(), det_src.size(),
                  ws_a, 131072u,
                  eskb_a.data(), eskb_a.size(), &eskb_res_a) == ESHKOL_SH_OK,
              "determinism pass A ESKB failed");
        for (int t = 0; t < 3; ++t) {
            elf_a[t].assign(32768u, 0xEEu);
            std::memset(ws_a, 0xAA, sizeof(ws_a));
            CHECK(eshkol_os_compile_to_elf64(
                      det_src.data(), det_src.size(), isas[t],
                      ws_a, 131072u,
                      elf_a[t].data(), elf_a[t].size(), &elf_res_a[t]) == ESHKOL_SH_OK,
                  "determinism pass A ELF64 failed");
        }

        /* Interleaved unrelated compilations + live VM evaluation + failing parse */
        {
            const std::string noise_src =
                "(define (sq x) (* x x))\n(let ((v (sq 17))) (- v 100))\n";
            eshkol_os_compile_result_t noise_res;
            CHECK(eshkol_os_compile_to_eskb(
                      noise_src.data(), noise_src.size(),
                      compiler_ws, sizeof(compiler_ws),
                      out_eskb, sizeof(out_eskb), &noise_res) == ESHKOL_SH_OK,
                  "interleaved noise ESKB compile");
            eshkol_os_vm_result_t vres;
            CHECK(eshkol_os_vm_eval_eskb(&shell, out_eskb, noise_res.bytes_written, nullptr, &vres) == ESHKOL_VM_OK &&
                      vres.int_value == 189,
                  "interleaved noise VM eval");
            const std::string bad_src = "(+ 1 unbound_xyz)\n";
            CHECK(eshkol_os_compile_to_eskb(
                      bad_src.data(), bad_src.size(),
                      compiler_ws, sizeof(compiler_ws),
                      out_eskb, sizeof(out_eskb), &noise_res) == ESHKOL_SH_ERR_UNBOUND_SYMBOL,
                  "interleaved negative compile");
        }

        /* Pass B: workspace poisoned with 0x55, shifted by +13 bytes, output poisoned with 0x33 */
        std::memset(ws_b, 0x55, sizeof(ws_b));
        std::memset(eskb_b.data(), 0x33, eskb_b.size());
        CHECK(eshkol_os_compile_to_eskb(
                  det_src.data(), det_src.size(),
                  ws_b + 13, 131072u,
                  eskb_b.data(), eskb_b.size(), &eskb_res_b) == ESHKOL_SH_OK,
              "determinism pass B ESKB failed");
        CHECK(eskb_res_a.bytes_written == eskb_res_b.bytes_written &&
                  eskb_res_a.crc32 == eskb_res_b.crc32 &&
                  std::memcmp(eskb_a.data(), eskb_b.data(), eskb_res_a.bytes_written) == 0,
              "ESKB v2 output is NOT byte-identical across 0xAA/0x55 poison and shifted workspace");

        for (int t = 0; t < 3; ++t) {
            elf_b[t].assign(32768u, 0x33u);
            std::memset(ws_b, 0x55, sizeof(ws_b));
            CHECK(eshkol_os_compile_to_elf64(
                      det_src.data(), det_src.size(), isas[t],
                      ws_b + 13 + t * 5, 131072u,
                      elf_b[t].data(), elf_b[t].size(), &elf_res_b[t]) == ESHKOL_SH_OK,
                  "determinism pass B ELF64 failed");
            CHECK(elf_res_a[t].bytes_written == elf_res_b[t].bytes_written &&
                      elf_res_a[t].crc32 == elf_res_b[t].crc32 &&
                      std::memcmp(elf_a[t].data(), elf_b[t].data(), elf_res_a[t].bytes_written) == 0,
                  std::string("ELF64 output NOT byte-identical for ") + eshkol_os_target_isa_name(isas[t]));

            /* Verify W^X ELF64 structure, machine encodings, llvm-readelf & llvm-objdump */
            if (verify_elf64_image(elf_a[t].data(), elf_res_a[t].bytes_written,
                                   isas[t], elf_res_a[t],
                                   llvm_readelf, llvm_objdump, work_dir) != 0) {
                return 1;
            }
        }

        /* Also verify the compiled ESKB executes cleanly in U5 VM shell */
        eshkol_os_vm_result_t vres;
        CHECK(eshkol_os_vm_eval_eskb(&shell, eskb_a.data(), eskb_res_a.bytes_written, nullptr, &vres) == ESHKOL_VM_OK,
              "determinism ESKB eval failed");
        /* fib(8)=21, fact(5)=120 -> s=141, d=-99 -> 282 + 99 = 381 */
        CHECK(vres.int_value == 381 && vres.syscalls_dispatched == 1 && vres.heap_bytes_live == 0,
              "determinism ESKB expected 381 with 1 TASK_YIELD syscall");

        std::cout << "PASS Rigor Guard #3 (Byte-Identical Determinism): ESKB=" << eskb_res_a.bytes_written
                  << "B, x86_64=" << elf_res_a[0].bytes_written
                  << "B, aarch64=" << elf_res_a[1].bytes_written
                  << "B, riscv64=" << elf_res_a[2].bytes_written
                  << "B (0xAA vs 0x55 + shifted workspace + interleaved noise: memcmp == 0)" << std::endl;
    }

    /* ── 5. Bounded parser & compiler negative controls ── */
    {
        std::string deep;
        for (uint32_t i = 0; i < ESHKOL_OS_SH_MAX_PARSE_DEPTH + 4u; ++i) deep += "(+ 1 ";
        deep += "0";
        for (uint32_t i = 0; i < ESHKOL_OS_SH_MAX_PARSE_DEPTH + 4u; ++i) deep += ")";
        eshkol_os_compile_result_t cres;
        CHECK(eshkol_os_compile_to_eskb(
                  deep.data(), deep.size(),
                  compiler_ws, sizeof(compiler_ws),
                  out_eskb, sizeof(out_eskb), &cres) == ESHKOL_SH_ERR_DEPTH_EXCEEDED,
              "excessive nesting depth must fail closed with DEPTH_EXCEEDED");

        const std::string tiny_src = "(+ 1 2)\n";
        uint8_t tiny_out[32];
        CHECK(eshkol_os_compile_to_eskb(
                  tiny_src.data(), tiny_src.size(),
                  compiler_ws, sizeof(compiler_ws),
                  tiny_out, sizeof(tiny_out), &cres) == ESHKOL_SH_ERR_OUTPUT_TOO_SMALL,
              "undersized ESKB output buffer must fail closed");

        CHECK(eshkol_os_compile_to_elf64(
                  tiny_src.data(), tiny_src.size(),
                  static_cast<eshkol_os_target_isa_t>(99),
                  compiler_ws, sizeof(compiler_ws),
                  out_elf, sizeof(out_elf), &cres) == ESHKOL_SH_ERR_INVALID_TARGET,
              "unknown target ISA must fail closed with INVALID_TARGET");

        std::cout << "PASS parser & compiler fail-closed negative controls" << std::endl;
    }

    /* ── 6. Tear down VM shell and verify 0 leaked pages ── */
    CHECK(eshkol_os_vm_destroy(&shell) == ESHKOL_VM_OK, "VM shell destroy failed");
    CHECK(g_kernel.live_mapped_pages == 0 &&
              g_kernel.total_pages_mapped == g_kernel.total_pages_unmapped,
          "VM page map/unmap ledger imbalance");

    /* ── 7. Zero-LLVM / Zero-POSIX / Zero-malloc symbol audit via llvm-nm -u ── */
    if (audit_zero_external_symbols(llvm_nm, work_dir, "host libeshkol-os-self-host.a", self_host_lib) != 0) {
        return 1;
    }

    const fs::path src_c = repo_root / "lib" / "core" / "os_self_host.c";
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
        const fs::path obj = work_dir / ("os_self_host_" + t.label + ".o");
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
            return fail("freestanding cross-compile of os_self_host.c failed for " + t.label + ":\n" + cc.output);
        }
        ++g_checks;
        if (audit_zero_external_symbols(llvm_nm, work_dir, "bare-metal " + t.label + " ELF", obj) != 0) {
            return 1;
        }
    }

    fs::remove_all(work_dir, ec);
    std::cout << "PASS: os_u6_native_self_host_test (" << g_checks << " checks)" << std::endl;
    return 0;
}
