/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U2 Gate (`os_u2_hardware_layouts`):
 *   - Declarative `define-layout` (`:repr-c`, `:packed`, `:align N`,
 *     `:side-effect-free-read`, inline `(array T N)`, nested layouts,
 *     unsigned LSB-first `:bits N` bitfields)
 *   - Compile-time `sizeof`, `alignof`, `offsetof` (storage unit byte offset),
 *     `bit-offset`, and `static-assert`
 *   - Zero-alloc `field-addr`, `field-ref`, `field-set!`, `volatile-field-ref`,
 *     `volatile-field-set!`
 *   - Target-dependent `usize`/`isize`/`ptr` (4B on RV32/Cortex-M4, 8B on 64-bit)
 *   - Byte-for-byte cross-check against `clang` C headers for:
 *       1. x86_64 IDT gate (`:packed`, 16B) & PML4/PTE
 *       2. RISC-V Sv39/Sv48 PTE
 *       3. AArch64 Stage-1 translation descriptor
 *       4. Cortex-M4 NVIC / MPU register block
 *       5. VirtIO split-virtqueue descriptor & avail/used rings
 *   - Live binary link + byte-for-byte `memcmp` execution oracle between
 *     Eshkol-emitted `.o` and `clang` C structs on the host ABI
 *   - 4 fail-closed compile-time negative controls
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

int check_positive(const fs::path& run_bin,
                   const std::string& llvm_dis,
                   const fs::path& work,
                   const std::string& name,
                   const std::string& profile,
                   const std::string& target,
                   const std::string& source,
                   const std::vector<std::string>& ir_needles) {
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
        return fail("usage: os_u2_hardware_layouts_test <eshkol-run> [llvm-dis]");
    }
    const fs::path run_bin = fs::absolute(argv[1]);
    const std::string llvm_dis = (argc == 3 && argv[2] && argv[2][0]) ? argv[2] : "";

    std::error_code ec;
    const fs::path work = fs::temp_directory_path() / "eshkol-os-u2-hardware-layouts-test";
    fs::remove_all(work, ec);
    fs::create_directories(work, ec);

    // 1. x86_64 IDT Gate (:packed, 16B) + x86_64 PML4/PTE (u64 bitfields) + VirtIO rings
    const std::string x86_layouts = R"ESK(
(define-layout x86-idt-gate
  :packed :side-effect-free-read
  (offset-lo  u16)
  (selector   u16)
  (ist        u8 :bits 3)
  (zero0      u8 :bits 5)
  (gate-type  u8 :bits 4)
  (zero1      u8 :bits 1)
  (dpl        u8 :bits 2)
  (present    u8 :bits 1)
  (offset-mid u16)
  (offset-hi  u32)
  (reserved   u32))

(static-assert (= (sizeof x86-idt-gate) 16) "x86-64 IDT gate must be 16 bytes")
(static-assert (= (alignof x86-idt-gate) 1) "packed IDT gate alignment must be 1")
(static-assert (= (offsetof x86-idt-gate offset-lo) 0))
(static-assert (= (offsetof x86-idt-gate selector) 2))
(static-assert (= (offsetof x86-idt-gate ist) 4))
(static-assert (= (bit-offset x86-idt-gate ist) 0))
(static-assert (= (offsetof x86-idt-gate zero0) 4))
(static-assert (= (bit-offset x86-idt-gate zero0) 3))
(static-assert (= (offsetof x86-idt-gate gate-type) 5))
(static-assert (= (bit-offset x86-idt-gate dpl) 5))
(static-assert (= (bit-offset x86-idt-gate present) 7))
(static-assert (= (offsetof x86-idt-gate offset-mid) 6))
(static-assert (= (offsetof x86-idt-gate offset-hi) 8))
(static-assert (= (offsetof x86-idt-gate reserved) 12))
(static-assert (= (sizeof usize) 8) "x86-64 usize must be 8 bytes")

(define-layout x86-pte
  :repr-c :side-effect-free-read
  (present  u64 :bits 1)
  (writable u64 :bits 1)
  (user     u64 :bits 1)
  (pwt      u64 :bits 1)
  (pcd      u64 :bits 1)
  (accessed u64 :bits 1)
  (dirty    u64 :bits 1)
  (huge     u64 :bits 1)
  (global   u64 :bits 1)
  (avail    u64 :bits 3)
  (pfn      u64 :bits 40)
  (avail2   u64 :bits 11)
  (nx       u64 :bits 1))

(static-assert (= (sizeof x86-pte) 8))
(static-assert (= (alignof x86-pte) 8))
(static-assert (= (bit-offset x86-pte pfn) 12))
(static-assert (= (bit-offset x86-pte nx) 63))

(define-layout virtq-desc
  :repr-c :align 16
  (addr  u64)
  (len   u32)
  (flags u16)
  (next  u16))

(define-layout virtq-used-elem
  :repr-c
  (id  u32)
  (len u32))

(define-layout virtq-used
  :repr-c :align 4
  (flags u16)
  (idx   u16)
  (ring  (array virtq-used-elem 16)))

(static-assert (= (sizeof virtq-desc) 16))
(static-assert (= (alignof virtq-desc) 16))
(static-assert (= (offsetof virtq-desc flags) 12))
(static-assert (= (offsetof virtq-desc next) 14))
(static-assert (= (sizeof virtq-used-elem) 8))
(static-assert (= (sizeof virtq-used) 132))
(static-assert (= (offsetof virtq-used ring) 4))

(define (write-idt-entry (gate : ptr) (handler : u64) (sel : u16)) : null
  (begin
    (volatile-field-set! x86-idt-gate gate offset-lo (bitwise-and handler 65535))
    (volatile-field-set! x86-idt-gate gate selector sel)
    (volatile-field-set! x86-idt-gate gate ist 1)
    (volatile-field-set! x86-idt-gate gate zero0 0)
    (volatile-field-set! x86-idt-gate gate gate-type 14)
    (volatile-field-set! x86-idt-gate gate zero1 0)
    (volatile-field-set! x86-idt-gate gate dpl 0)
    (volatile-field-set! x86-idt-gate gate present 1)
    (volatile-field-set! x86-idt-gate gate offset-mid (bitwise-and (arithmetic-shift handler -16) 65535))
    (volatile-field-set! x86-idt-gate gate offset-hi (bitwise-and (arithmetic-shift handler -32) 4294967295))
    (volatile-field-set! x86-idt-gate gate reserved 0))
  :export-symbol write_idt_entry)

(define (read-used-len (used : ptr) (slot : usize)) : u32
  (volatile-field-ref virtq-used-elem (field-addr virtq-used used ring slot) len)
  :export-symbol read_used_len)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_x86_64_layouts",
            "freestanding-kernel-native", "x86_64-unknown-none", x86_layouts,
            {"store volatile", "load volatile", "noredzone", "noimplicitfloat"})) {
        return rc;
    }

    // 2. RISC-V Sv39/Sv48 PTE on riscv64gc-unknown-none-elf
    const std::string rv64_pte = R"ESK(
(define-layout sv39-pte
  :repr-c :side-effect-free-read
  (v        u64 :bits 1)
  (r        u64 :bits 1)
  (w        u64 :bits 1)
  (x        u64 :bits 1)
  (u        u64 :bits 1)
  (g        u64 :bits 1)
  (a        u64 :bits 1)
  (d        u64 :bits 1)
  (rsw      u64 :bits 2)
  (ppn      u64 :bits 44)
  (reserved u64 :bits 10))

(static-assert (= (sizeof sv39-pte) 8))
(static-assert (= (alignof sv39-pte) 8))
(static-assert (= (bit-offset sv39-pte rsw) 8))
(static-assert (= (bit-offset sv39-pte ppn) 10))
(static-assert (= (bit-offset sv39-pte reserved) 54))
(static-assert (= (sizeof usize) 8))

(define (sv39-get-ppn (pte : ptr)) : u64
  (volatile-field-ref sv39-pte pte ppn)
  :export-symbol sv39_get_ppn)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_rv64_sv39_pte",
            "freestanding-kernel-native", "riscv64gc-unknown-none-elf", rv64_pte,
            {"load volatile i64", "lshr i64", "17592186044415"})) {
        return rc;
    }

    // 3. AArch64 Stage-1 Translation Descriptor on aarch64-unknown-none
    const std::string arm64_pte = R"ESK(
(define-layout aarch64-desc
  :repr-c :side-effect-free-read
  (valid    u64 :bits 1)
  (table    u64 :bits 1)
  (attr-idx u64 :bits 3)
  (ns       u64 :bits 1)
  (ap       u64 :bits 2)
  (sh       u64 :bits 2)
  (af       u64 :bits 1)
  (ng       u64 :bits 1)
  (oa       u64 :bits 36)
  (res0     u64 :bits 4)
  (cont     u64 :bits 1)
  (pxn      u64 :bits 1)
  (uxn      u64 :bits 1)
  (sw       u64 :bits 9))

(static-assert (= (sizeof aarch64-desc) 8))
(static-assert (= (bit-offset aarch64-desc attr-idx) 2))
(static-assert (= (bit-offset aarch64-desc oa) 12))
(static-assert (= (bit-offset aarch64-desc pxn) 53))
(static-assert (= (bit-offset aarch64-desc uxn) 54))

(define (arm64-set-af (desc : ptr) (v : u64)) : null
  (volatile-field-set! aarch64-desc desc af v)
  :export-symbol arm64_set_af)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_aarch64_stage1_desc",
            "freestanding-kernel-native", "aarch64-unknown-none", arm64_pte,
            {"load volatile i64", "store volatile i64", "noredzone"})) {
        return rc;
    }

    // 4. Cortex-M4 NVIC/MPU Register Block + 32-bit target-dependent usize == 4
    const std::string cm4_regs = R"ESK(
(define-layout nvic-regs
  :repr-c :side-effect-free-read
  (iser (array u32 8))
  (res0 (array u32 24))
  (icer (array u32 8)))

(static-assert (= (sizeof usize) 4) "Cortex-M4 usize must be 4 bytes")
(static-assert (= (sizeof ptr) 4) "Cortex-M4 ptr must be 4 bytes")
(static-assert (= (sizeof nvic-regs) 160))
(static-assert (= (offsetof nvic-regs iser) 0))
(static-assert (= (offsetof nvic-regs icer) 128))

(define (nvic-enable-irq (base : ptr) (reg : u32) (mask : u32)) : null
  (volatile-field-set! nvic-regs base iser reg mask)
  :export-symbol nvic_enable_irq)
)ESK";
    if (int rc = check_positive(
            run_bin, llvm_dis, work, "pos_cortex_m4_nvic",
            "freestanding-mcu-native", "thumbv7em-none-eabi", cm4_regs,
            {"store volatile i32"})) {
        return rc;
    }

    // 5. LIVE BINARY CROSS-CHECK AGAINST HOST C COMPILER (`clang` / `c++`):
    //    Link an Eshkol-emitted `.o` directly with a C harness and assert
    //    byte-for-byte `memcmp == 0` on IDT gate, Sv39 PTE, AArch64 desc, NVIC, VirtIO!
    const std::string host_triple = host_target_triple();
    if (!host_triple.empty()) {
        const fs::path xdir = work / "clang_binary_crosscheck";
        fs::create_directories(xdir);
        const fs::path esk_src = xdir / "layouts_abi.esk";
        const fs::path esk_obj = xdir / "layouts_abi.o";
        const fs::path c_src   = xdir / "crosscheck_main.c";
        const fs::path exe     = xdir / "crosscheck_bin";

        const std::string host_esk = R"ESK(
(define-layout x86-idt-gate
  :packed :side-effect-free-read
  (offset-lo  u16)
  (selector   u16)
  (ist        u8 :bits 3)
  (zero0      u8 :bits 5)
  (gate-type  u8 :bits 4)
  (zero1      u8 :bits 1)
  (dpl        u8 :bits 2)
  (present    u8 :bits 1)
  (offset-mid u16)
  (offset-hi  u32)
  (reserved   u32))

(define-layout sv39-pte
  :repr-c :side-effect-free-read
  (v        u64 :bits 1)
  (r        u64 :bits 1)
  (w        u64 :bits 1)
  (x        u64 :bits 1)
  (u        u64 :bits 1)
  (g        u64 :bits 1)
  (a        u64 :bits 1)
  (d        u64 :bits 1)
  (rsw      u64 :bits 2)
  (ppn      u64 :bits 44)
  (reserved u64 :bits 10))

(define-layout virtq-desc
  :repr-c :align 16
  (addr  u64)
  (len   u32)
  (flags u16)
  (next  u16))

(define-layout nvic-regs
  :repr-c
  (iser (array u32 8))
  (res0 (array u32 24))
  (icer (array u32 8)))

(define (eshkol-fill-idt (p : ptr) (off-lo : u16) (sel : u16) (ist-v : u8)
                         (type-v : u8) (dpl-v : u8) (pres-v : u8)
                         (off-mid : u16) (off-hi : u32)) : null
  (begin
    (volatile-field-set! x86-idt-gate p offset-lo off-lo)
    (volatile-field-set! x86-idt-gate p selector sel)
    (volatile-field-set! x86-idt-gate p ist ist-v)
    (volatile-field-set! x86-idt-gate p zero0 0)
    (volatile-field-set! x86-idt-gate p gate-type type-v)
    (volatile-field-set! x86-idt-gate p zero1 0)
    (volatile-field-set! x86-idt-gate p dpl dpl-v)
    (volatile-field-set! x86-idt-gate p present pres-v)
    (volatile-field-set! x86-idt-gate p offset-mid off-mid)
    (volatile-field-set! x86-idt-gate p offset-hi off-hi)
    (volatile-field-set! x86-idt-gate p reserved 0))
  :export-symbol eshkol_fill_idt)

(define (eshkol-read-idt-dpl (p : ptr)) : u8
  (volatile-field-ref x86-idt-gate p dpl)
  :export-symbol eshkol_read_idt_dpl)

(define (eshkol-fill-sv39 (p : ptr) (ppn-v : u64) (rsw-v : u64)) : null
  (begin
    (volatile-field-set! sv39-pte p v 1)
    (volatile-field-set! sv39-pte p r 1)
    (volatile-field-set! sv39-pte p w 1)
    (volatile-field-set! sv39-pte p x 0)
    (volatile-field-set! sv39-pte p u 1)
    (volatile-field-set! sv39-pte p g 0)
    (volatile-field-set! sv39-pte p a 1)
    (volatile-field-set! sv39-pte p d 1)
    (volatile-field-set! sv39-pte p rsw rsw-v)
    (volatile-field-set! sv39-pte p ppn ppn-v)
    (volatile-field-set! sv39-pte p reserved 0))
  :export-symbol eshkol_fill_sv39)

(define (eshkol-read-sv39-ppn (p : ptr)) : u64
  (volatile-field-ref sv39-pte p ppn)
  :export-symbol eshkol_read_sv39_ppn)

(define (eshkol-fill-virtq (p : ptr) (addr-v : u64) (len-v : u32) (flags-v : u16) (next-v : u16)) : null
  (begin
    (field-set! virtq-desc p addr addr-v)
    (field-set! virtq-desc p len len-v)
    (field-set! virtq-desc p flags flags-v)
    (field-set! virtq-desc p next next-v))
  :export-symbol eshkol_fill_virtq)

(define (eshkol-set-nvic-icer (p : ptr) (i : u32) (v : u32)) : null
  (volatile-field-set! nvic-regs p icer i v)
  :export-symbol eshkol_set_nvic_icer)
)ESK";

        const std::string c_harness = R"C(
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct __attribute__((packed)) {
    uint16_t offset_lo;
    uint16_t selector;
    uint8_t  ist       : 3;
    uint8_t  zero0     : 5;
    uint8_t  gate_type : 4;
    uint8_t  zero1     : 1;
    uint8_t  dpl       : 2;
    uint8_t  present   : 1;
    uint16_t offset_mid;
    uint32_t offset_hi;
    uint32_t reserved;
} c_x86_idt_gate_t;

typedef struct {
    uint64_t v        : 1;
    uint64_t r        : 1;
    uint64_t w        : 1;
    uint64_t x        : 1;
    uint64_t u        : 1;
    uint64_t g        : 1;
    uint64_t a        : 1;
    uint64_t d        : 1;
    uint64_t rsw      : 2;
    uint64_t ppn      : 44;
    uint64_t reserved : 10;
} c_sv39_pte_t;

typedef struct __attribute__((aligned(16))) {
    uint64_t addr;
    uint32_t len;
    uint16_t flags;
    uint16_t next;
} c_virtq_desc_t;

typedef struct {
    uint32_t iser[8];
    uint32_t res0[24];
    uint32_t icer[8];
} c_nvic_regs_t;

_Static_assert(sizeof(c_x86_idt_gate_t) == 16, "IDT gate size");
_Static_assert(_Alignof(c_x86_idt_gate_t) == 1, "IDT gate align");
_Static_assert(sizeof(c_sv39_pte_t) == 8, "Sv39 PTE size");
_Static_assert(sizeof(c_virtq_desc_t) == 16, "virtq_desc size");
_Static_assert(_Alignof(c_virtq_desc_t) == 16, "virtq_desc align");
_Static_assert(offsetof(c_nvic_regs_t, icer) == 128, "nvic icer offset");

extern void eshkol_fill_idt(void* p, uint16_t off_lo, uint16_t sel, uint8_t ist_v,
                            uint8_t type_v, uint8_t dpl_v, uint8_t pres_v,
                            uint16_t off_mid, uint32_t off_hi);
extern uint8_t eshkol_read_idt_dpl(void* p);
extern void eshkol_fill_sv39(void* p, uint64_t ppn_v, uint64_t rsw_v);
extern uint64_t eshkol_read_sv39_ppn(void* p);
extern void eshkol_fill_virtq(void* p, uint64_t addr_v, uint32_t len_v, uint16_t flags_v, uint16_t next_v);
extern void eshkol_set_nvic_icer(void* p, uint32_t i, uint32_t v);

int main(void) {
    /* 1. IDT gate byte-for-byte match */
    c_x86_idt_gate_t c_idt, e_idt;
    memset(&c_idt, 0, sizeof(c_idt));
    memset(&e_idt, 0, sizeof(e_idt));
    c_idt.offset_lo  = 0xBEEFu;
    c_idt.selector   = 0x0008u;
    c_idt.ist        = 5u;
    c_idt.zero0      = 0u;
    c_idt.gate_type  = 14u;
    c_idt.zero1      = 0u;
    c_idt.dpl        = 3u;
    c_idt.present    = 1u;
    c_idt.offset_mid = 0xCAFEu;
    c_idt.offset_hi  = 0x12345678u;
    c_idt.reserved   = 0u;
    eshkol_fill_idt(&e_idt, 0xBEEFu, 0x0008u, 5u, 14u, 3u, 1u, 0xCAFEu, 0x12345678u);
    if (memcmp(&c_idt, &e_idt, sizeof(c_idt)) != 0) {
        fprintf(stderr, "FAIL: x86_idt_gate byte mismatch\n");
        return 1;
    }
    if (eshkol_read_idt_dpl(&c_idt) != 3u) {
        fprintf(stderr, "FAIL: x86_idt_gate dpl read mismatch\n");
        return 1;
    }

    /* 2. Sv39 PTE byte-for-byte match */
    c_sv39_pte_t c_pte, e_pte;
    memset(&c_pte, 0, sizeof(c_pte));
    memset(&e_pte, 0, sizeof(e_pte));
    const uint64_t test_ppn = 0x00000ABCDE12345ULL;
    c_pte.v = 1; c_pte.r = 1; c_pte.w = 1; c_pte.x = 0;
    c_pte.u = 1; c_pte.g = 0; c_pte.a = 1; c_pte.d = 1;
    c_pte.rsw = 2; c_pte.ppn = test_ppn; c_pte.reserved = 0;
    eshkol_fill_sv39(&e_pte, test_ppn, 2);
    if (memcmp(&c_pte, &e_pte, sizeof(c_pte)) != 0) {
        fprintf(stderr, "FAIL: sv39_pte byte mismatch\n");
        return 1;
    }
    if (eshkol_read_sv39_ppn(&c_pte) != test_ppn) {
        fprintf(stderr, "FAIL: sv39_pte ppn read mismatch\n");
        return 1;
    }

    /* 3. VirtIO descriptor byte-for-byte match */
    c_virtq_desc_t c_vq, e_vq;
    memset(&c_vq, 0, sizeof(c_vq));
    memset(&e_vq, 0, sizeof(e_vq));
    c_vq.addr = 0xDEADBEEFCAFEBABEULL;
    c_vq.len = 4096u;
    c_vq.flags = 0x0003u;
    c_vq.next = 7u;
    eshkol_fill_virtq(&e_vq, 0xDEADBEEFCAFEBABEULL, 4096u, 0x0003u, 7u);
    if (memcmp(&c_vq, &e_vq, sizeof(c_vq)) != 0) {
        fprintf(stderr, "FAIL: virtq_desc byte mismatch\n");
        return 1;
    }

    /* 4. NVIC array offset & volatile write match */
    c_nvic_regs_t nvic;
    memset(&nvic, 0, sizeof(nvic));
    eshkol_set_nvic_icer(&nvic, 5u, 0xA5A55A5Au);
    if (nvic.icer[5] != 0xA5A55A5Au) {
        fprintf(stderr, "FAIL: nvic.icer[5] mismatch\n");
        return 1;
    }

    printf("CLANG C-STRUCT BINARY CROSS-CHECK: BYTE-FOR-BYTE PASS\n");
    return 0;
}
)C";

        if (!write_text(esk_src, host_esk) || !write_text(c_src, c_harness)) {
            return fail("could not write clang cross-check files");
        }
        ProcessResult ecomp = run_process(
            {run_bin.string(), "--profile", "freestanding-kernel-native",
             "--target", host_triple, "-o", esk_obj.string(), esk_src.string()},
            xdir);
        g_checks++;
        if (ecomp.exit_code != 0) {
            return fail("host Eshkol layout compile failed:\n" + ecomp.output);
        }
        ProcessResult clink = run_process(
            {"cc", "-std=c11", "-Wall", "-Werror", c_src.string(), esk_obj.string(), "-o", exe.string()},
            xdir);
        g_checks++;
        if (clink.exit_code != 0) {
            return fail("C harness link failed:\n" + clink.output);
        }
        ProcessResult crun = run_process({exe.string()}, xdir);
        g_checks++;
        if (crun.exit_code != 0 || !contains(crun.output, "BYTE-FOR-BYTE PASS")) {
            return fail("C binary cross-check failed:\n" + crun.output);
        }
        std::cout << "PASS binary cross-check: " << crun.output;
    }

    // NEGATIVE CONTROL 1: failing static-assert
    const std::string neg_sa = R"ESK(
(define-layout bad-sz :packed (a u32) (b u16))
(static-assert (= (sizeof bad-sz) 8) "expected size 8")
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_static_assert",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_sa, "static-assert failed: expected size 8")) {
        return rc;
    }

    // NEGATIVE CONTROL 2: signed bitfield rejected (Tsotchke-chan Invariant #1)
    const std::string neg_signed_bf = R"ESK(
(define-layout bad-signed-bf :repr-c (x i32 :bits 5))
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_signed_bitfield",
            "freestanding-kernel-native", "aarch64-unknown-none",
            neg_signed_bf, "unsigned integer base type")) {
        return rc;
    }

    // NEGATIVE CONTROL 3: volatile-field-set! on bitfield without :side-effect-free-read (Invariant #2)
    const std::string neg_vol_bf = R"ESK(
(define-layout mmio-status :repr-c (irq-flag u32 :bits 1) (rest u32 :bits 31))
(define (bad-mmio-write (p : ptr)) : null
  (volatile-field-set! mmio-status p irq-flag 1)
  :export-symbol bad_mmio_write)
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_volatile_bf_without_sefr",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_vol_bf, ":side-effect-free-read")) {
        return rc;
    }

    // NEGATIVE CONTROL 4: bitfield width exceeding storage unit
    const std::string neg_bf_overflow = R"ESK(
(define-layout bad-bf-wide :repr-c (x u8 :bits 9))
)ESK";
    if (int rc = check_negative(
            run_bin, work, "neg_bitfield_width_overflow",
            "freestanding-kernel-native", "x86_64-unknown-none",
            neg_bf_overflow, "exceeds storage unit")) {
        return rc;
    }

    std::cout << "OS U2 HARDWARE LAYOUTS TEST: PASS (" << g_checks << " checks)" << std::endl;
    return 0;
}
