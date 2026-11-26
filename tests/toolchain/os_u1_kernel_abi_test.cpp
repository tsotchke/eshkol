/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U1 Gate (`os_u1_interrupt_safety`):
 *   - Kernel-safe codegen on `freestanding-kernel-native` (`noredzone`,
 *     `noimplicitfloat`, zero floating-point/SSE/x87 ops by default)
 *   - `:naked`, `:interrupt`, `:arch-hal`, `:no-alloc`, `:fpu-safe` modifiers
 *   - Typed privileged `target-intrinsic` (`eshkol.cpu.*`) across all 5 ISAs:
 *     x86_64, aarch64, riscv64, riscv32, thumbv7em (Cortex-M4)
 *   - Negative controls (fail closed at compile time, no .o emitted):
 *       1. `unsafe-asm` outside `:arch-hal`
 *       2. Allocation / unverified external call inside kernel / `:interrupt`
 *       3. Floating-point instruction inside kernel / `:interrupt` without `:fpu-safe`
 *       4. `:interrupt` calling `:fpu-safe` callee without `:fpu-safe`
 *       5. Architecture-mismatched privileged `target-intrinsic`
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

int check_positive(const fs::path& run_bin,
                   const std::string& llvm_dis,
                   const fs::path& work,
                   const std::string& name,
                   const std::string& profile,
                   const std::string& target,
                   const std::string& source,
                   const std::vector<std::string>& ir_needles,
                   const std::vector<std::string>& ir_forbidden = {}) {
    const fs::path dir = work / name;
    fs::create_directories(dir);
    const fs::path src = dir / (name + ".esk");
    const fs::path obj = dir / (name + ".o");
    if (!write_text(src, source)) return fail(name + ": could not write source");

    ProcessResult res = run_process(
        {run_bin.string(), "--profile", profile, "--target", target,
         "-o", obj.string(), src.string()},
        dir);
    g_checks++;
    if (res.exit_code != 0) {
        return fail(name + ": expected compile success on " + target +
                    ", got exit " + std::to_string(res.exit_code) + "\n" + res.output);
    }
    if (!fs::exists(obj)) return fail(name + ": object file missing");

    if (!llvm_dis.empty()) {
        const fs::path bc = obj.string() + ".bc";
        if (!fs::exists(bc)) return fail(name + ": bitcode sidecar missing");
        ProcessResult ir = run_process({llvm_dis, bc.string(), "-o", "-"}, dir);
        if (ir.exit_code != 0) return fail(name + ": llvm-dis failed\n" + ir.output);
        for (const auto& needle : ir_needles) {
            g_checks++;
            if (!contains(ir.output, needle)) {
                return fail(name + ": IR missing expected needle '" + needle + "'\n" + ir.output);
            }
        }
        for (const auto& bad : ir_forbidden) {
            g_checks++;
            if (contains(ir.output, bad)) {
                return fail(name + ": IR unexpectedly contains '" + bad + "'\n" + ir.output);
            }
        }
    }

    std::cout << "PASS positive: " << name << " (" << target << ")" << std::endl;
    return 0;
}

int check_negative(const fs::path& run_bin,
                   const fs::path& work,
                   const std::string& name,
                   const std::string& profile,
                   const std::string& target,
                   const std::string& source,
                   const std::string& expected_diag) {
    const fs::path dir = work / name;
    fs::create_directories(dir);
    const fs::path src = dir / (name + ".esk");
    const fs::path obj = dir / (name + ".o");
    std::error_code ec;
    fs::remove(obj, ec);
    if (!write_text(src, source)) return fail(name + ": could not write source");

    ProcessResult res = run_process(
        {run_bin.string(), "--profile", profile, "--target", target,
         "-o", obj.string(), src.string()},
        dir);
    g_checks++;
    if (res.exit_code == 0) {
        return fail(name + ": negative control unexpectedly succeeded\n" + res.output);
    }
    g_checks++;
    if (fs::exists(obj)) {
        return fail(name + ": negative control emitted object file despite compile error");
    }
    g_checks++;
    if (!expected_diag.empty() && !contains(res.output, expected_diag)) {
        return fail(name + ": diagnostic missing '" + expected_diag + "'\n" + res.output);
    }
    std::cout << "PASS negative: " << name << " (rejected cleanly)" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        return fail("usage: os_u1_kernel_abi_test <eshkol-run> [llvm-dis]");
    }
    const fs::path run_bin = fs::absolute(argv[1]);
    const std::string llvm_dis = (argc == 3 && argv[2] && argv[2][0]) ? argv[2] : "";

    std::error_code ec;
    const fs::path work = fs::temp_directory_path() / "eshkol-os-u1-kernel-abi-test";
    fs::remove_all(work, ec);
    fs::create_directories(work, ec);

    // 1. x86_64 kernel ABI: x86_interruptcc, noredzone, noimplicitfloat, :naked :arch-hal,
    //    privileged CPU intrinsics (port I/O, CR0/CR3, invlpg, cli/sti/hlt/pause).
    const std::string x86_src = R"ESK(
(define (x86-tick-helper (port : u16) (val : u8)) : u8
  (begin
    (target-intrinsic null "eshkol.cpu.outb" u16 port u8 val)
    (target-intrinsic u8 "eshkol.cpu.inb" u16 port))
  :no-alloc)

(define (x86-timer-isr (frame : ptr)) : null
  (begin
    (target-intrinsic null "eshkol.cpu.cli")
    (x86-tick-helper 32 32)
    (target-intrinsic null "eshkol.cpu.invlpg" ptr frame)
    (target-intrinsic null "eshkol.cpu.sti"))
  :export-symbol x86_timer_isr :interrupt)

(define (x86-read-regs) : u64
  (let ((cr0 (target-intrinsic u64 "eshkol.cpu.read-cr0"))
        (cr3 (target-intrinsic u64 "eshkol.cpu.read-cr3")))
    (begin
      (target-intrinsic null "eshkol.cpu.write-cr3" u64 cr3)
      (target-intrinsic null "eshkol.cpu.pause")
      (bitwise-xor cr0 cr3)))
  :export-symbol x86_read_regs)

(define (x86-hal-halt) : null
  (unsafe-asm null "cli; hlt" "~~{memory}")
  :export-symbol x86_hal_halt :arch-hal :naked :no-return)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_x86_64_kernel",
            "freestanding-kernel-native", "x86_64-unknown-none", x86_src,
            {"x86_intrcc", "byval(", "noredzone", "noimplicitfloat",
             "naked", "noreturn", "inb $1, $0", "outb $1, $0",
             "movq %cr0, $0", "movq %cr3, $0", "invlpg ($0)", "cli; hlt"})) {
        return rc;
    }

    // 2. RISC-V 64-bit machine-mode ISR + CSR intrinsics
    const std::string rv64_src = R"ESK(
(define (rv64-timer-isr) : null
  (let ((cause (target-intrinsic u64 "eshkol.cpu.csrr-mcause"))
        (st    (target-intrinsic u64 "eshkol.cpu.csrr-mstatus")))
    (begin
      (target-intrinsic null "eshkol.cpu.csrw-mstatus" u64 (bitwise-or st cause))
      (target-intrinsic null "eshkol.cpu.sfence-vma")))
  :export-symbol rv64_timer_isr :interrupt)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_rv64_kernel",
            "freestanding-kernel-native", "riscv64gc-unknown-none-elf", rv64_src,
            {"\"interrupt\"=\"machine\"", "noredzone", "noimplicitfloat",
             "csrr $0, mcause", "csrw mstatus, $0", "sfence.vma zero, zero"})) {
        return rc;
    }

    // 3. RISC-V 32-bit machine-mode ISR
    const std::string rv32_src = R"ESK(
(define (rv32-timer-isr) : null
  (begin
    (target-intrinsic null "eshkol.cpu.irq-disable")
    (target-intrinsic null "eshkol.cpu.wfi")
    (target-intrinsic null "eshkol.cpu.irq-enable"))
  :export-symbol rv32_timer_isr :interrupt)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_rv32_mcu",
            "freestanding-mcu-native", "riscv32imac-unknown-none-elf", rv32_src,
            {"\"interrupt\"=\"machine\"", "noredzone", "noimplicitfloat",
             "csrci mstatus, 8", "wfi", "csrsi mstatus, 8"})) {
        return rc;
    }

    // 4. AArch64 kernel ABI + system register / TLB / barrier intrinsics
    const std::string arm64_src = R"ESK(
(define (arm64-timer-isr) : null
  (let ((el (target-intrinsic u64 "eshkol.cpu.mrs-currentel"))
        (cnt (target-intrinsic u64 "eshkol.cpu.mrs-cntpct")))
    (begin
      (target-intrinsic null "eshkol.cpu.tlbi-vmalle1")
      (target-intrinsic null "eshkol.cpu.dsb-sy")
      (target-intrinsic null "eshkol.cpu.isb")
      (volatile-store! u64 (usize->ptr 4096) (bitwise-xor el cnt))))
  :export-symbol arm64_timer_isr :interrupt)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_aarch64_kernel",
            "freestanding-kernel-native", "aarch64-unknown-none", arm64_src,
            {"noredzone", "noimplicitfloat", "mrs $0, CurrentEL",
             "mrs $0, cntpct_el0", "tlbi vmalle1", "dsb sy", "isb"})) {
        return rc;
    }

    // 5. Cortex-M4 MCU interrupt safety
    const std::string cm4_src = R"ESK(
(define (cm4-systick-isr) : null
  (begin
    (target-intrinsic null "eshkol.cpu.irq-disable")
    (target-intrinsic null "eshkol.cpu.dsb-sy")
    (target-intrinsic null "eshkol.cpu.isb")
    (target-intrinsic null "eshkol.cpu.wfi")
    (target-intrinsic null "eshkol.cpu.irq-enable"))
  :export-symbol cm4_systick_isr :interrupt)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_cortex_m4_mcu",
            "freestanding-mcu-native", "thumbv7em-none-eabi", cm4_src,
            {"noredzone", "noimplicitfloat", "cpsid i", "dsb sy", "isb", "wfi", "cpsie i"})) {
        return rc;
    }

    // NEGATIVE CONTROL 1: unsafe-asm outside :arch-hal MUST fail at compile time
    const std::string neg_asm_src = R"ESK(
(define (sneaky-asm) : null
  (unsafe-asm null "nop" "~~{memory}")
  :export-symbol sneaky_asm)
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_unsafe_asm_outside_arch_hal",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_asm_src, ":arch-hal")) {
        return rc;
    }

    // NEGATIVE CONTROL 2: unverified external/allocating call in kernel / NoAlloc
    const std::string neg_alloc_src = R"ESK(
(extern ptr malloc usize)
(define (bad-kernel-alloc) : ptr
  (malloc 64)
  :export-symbol bad_kernel_alloc)
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_alloc_in_kernel",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_alloc_src, "forbidden allocating or unverified external symbol")) {
        return rc;
    }

    // POSITIVE CONTROL 6: extern explicitly annotated :no-alloc IS allowed in kernel
    const std::string pos_noalloc_extern_src = R"ESK(
(extern ptr hal-frame-pool usize :no-alloc)
(define (ok-kernel-pool) : ptr
  (hal-frame-pool 4096)
  :export-symbol ok_kernel_pool)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_noalloc_extern",
            "freestanding-kernel-native", "x86_64-unknown-none",
            pos_noalloc_extern_src, {"\"eshkol.no-alloc\"=\"true\"", "noredzone", "noimplicitfloat"})) {
        return rc;
    }

    // NEGATIVE CONTROL 3a: floating-point arithmetic in freestanding rejected at front-door
    const std::string neg_real_arith_src = R"ESK(
(define (bad-isr-real (frame : ptr)) : null
  (let ((x (+ 1.5 2.25)))
    (if (> x 0.0) '() '()))
  :export-symbol bad_isr_real :interrupt)
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_real_arith_in_interrupt",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_real_arith_src, "machine-integer operands only")) {
        return rc;
    }

    // NEGATIVE CONTROL 3b: FPU register/value usage via :no-alloc double extern in :interrupt without :fpu-safe
    const std::string neg_fpu_src = R"ESK(
(extern double fpu-read-sensor :no-alloc)
(define (bad-isr-fp (frame : ptr)) : null
  (begin
    (fpu-read-sensor)
    '())
  :export-symbol bad_isr_fp :interrupt)
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_fpu_in_interrupt",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_fpu_src, ":fpu-safe")) {
        return rc;
    }

    // NEGATIVE CONTROL 4: :interrupt calling :fpu-safe function without :fpu-safe
    const std::string neg_isr_calls_fpu_src = R"ESK(
(extern double fpu-read-sensor :no-alloc)
(define (fp-helper) : null
  (begin
    (fpu-read-sensor)
    '())
  :fpu-safe)

(define (bad-isr-calls-fp (frame : ptr)) : null
  (begin
    (fp-helper)
    '())
  :export-symbol bad_isr_calls_fp :interrupt)
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_isr_calls_fpu_safe",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_isr_calls_fpu_src, "without :fpu-safe")) {
        return rc;
    }

    // NEGATIVE CONTROL 5: wrong-architecture privileged CPU intrinsic
    const std::string neg_arch_src = R"ESK(
(define (bad-x86-op-on-arm) : u8
  (target-intrinsic u8 "eshkol.cpu.inb" u16 64)
  :export-symbol bad_x86_op_on_arm)
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_wrong_arch_intrinsic",
            "freestanding-kernel-native", "aarch64-unknown-none",
            neg_arch_src, "only valid on x86_64")) {
        return rc;
    }

    std::cout << "OS U1 KERNEL ABI & INTERRUPT SAFETY TEST: PASS ("
              << g_checks << " checks)" << std::endl;
    return 0;
}
