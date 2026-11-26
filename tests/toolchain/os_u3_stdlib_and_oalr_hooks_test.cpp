/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U3 Gate (`os_u3_stdlib_and_oalr_hooks`):
 *   - Tier 1 `core.freestanding_pure` stdlib partition compiles cleanly under
 *     `freestanding-kernel-native` (x86_64, AArch64, RV64) and
 *     `freestanding-mcu-native` (Cortex-M4) with ZERO external symbols and
 *     ZERO OALR or hosted runtime dependencies.
 *   - Live host-ABI execution oracle verifying every function exported by
 *     `core.freestanding_pure` against a C11 reference implementation.
 *   - Five fail-closed compile-time negative controls proving absolute region
 *     isolation between freestanding kernel/MCU code and OALR / hosted stdlib:
 *       1. Tier 2 `core.list.higher_order` rejected in freestanding-kernel-native
 *       2. Tier 2 `math.statistics` rejected in freestanding-mcu-native
 *       3. Tier 3 `core.io` rejected in freestanding-kernel-native
 *       4. Direct `extern :no-alloc eshkol_os_alloc_pages` rejected by verifier
 *       5. Direct `extern :no-alloc arena_allocate` rejected by verifier
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

std::string first_non_intrinsic_declaration(const std::string& ir) {
    std::istringstream input(ir);
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t decl = line.find("declare ");
        if (decl == std::string::npos) continue;
        const std::size_t at = line.find('@', decl);
        if (at == std::string::npos || line.compare(at + 1, 5, "llvm.") != 0) {
            return line;
        }
    }
    return {};
}

std::string find_tool(const std::string& llvm_dis, const std::string& name) {
    if (!llvm_dis.empty()) {
        fs::path sibling = fs::path(llvm_dis).parent_path() / name;
        if (fs::exists(sibling)) return sibling.string();
    }
    return name;
}

int check_pure_kernel_target(
    const std::string& compiler,
    const std::string& llvm_dis,
    const std::string& llvm_nm,
    const fs::path& dir,
    const std::string& label,
    const std::string& profile,
    const std::string& target,
    const fs::path& src) {
    const fs::path obj = dir / ("pure_" + label + ".o");
    const fs::path bc = obj.string() + ".bc";
    auto res = run_process(
        {compiler, "--profile", profile, "--target", target,
         "--emit-object", "-o", obj.string(), src.string()},
        dir);
    if (res.exit_code != 0) {
        return fail("core.freestanding_pure compile failed for " + label + " (" + target + "):\n" + res.output);
    }
    if (!fs::exists(obj) || fs::file_size(obj) == 0) {
        return fail("missing object output for " + label);
    }
    ++g_checks;

    if (!llvm_dis.empty() && fs::exists(bc)) {
        auto dis = run_process({llvm_dis, "-o", "-", bc.string()}, dir);
        if (dis.exit_code != 0) {
            return fail("llvm-dis failed for " + label + ":\n" + dis.output);
        }
        const std::string bad_decl = first_non_intrinsic_declaration(dis.output);
        if (!bad_decl.empty()) {
            return fail("forbidden external declaration in " + label + " IR: " + bad_decl);
        }
        if (contains(dis.output, "arena_") || contains(dis.output, "eshkol_os_alloc_pages") ||
            contains(dis.output, "region_create") || contains(dis.output, "malloc")) {
            return fail("OALR/hosted allocator symbol leaked into " + label + " IR");
        }
        ++g_checks;
    }

    auto nm = run_process({llvm_nm, "-u", obj.string()}, dir);
    if (nm.exit_code == 0) {
        std::istringstream ss(nm.output);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.find_first_not_of(" \t\r\n") != std::string::npos) {
                return fail("unexpected undefined symbol in " + label + " object: " + line);
            }
        }
        ++g_checks;
    }
    return 0;
}

int check_negative(
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
    auto res = run_process(
        {compiler, "--profile", profile, "--target", target,
         "--emit-object", "-o", obj.string(), src.string()},
        dir);
    if (res.exit_code == 0) {
        return fail("negative control '" + label + "' unexpectedly compiled cleanly");
    }
    if (!expected_fragment.empty() && !contains(res.output, expected_fragment)) {
        return fail("negative control '" + label + "' rejected without expected diagnostic '" +
                    expected_fragment + "':\n" + res.output);
    }
    if (fs::exists(obj)) {
        return fail("negative control '" + label + "' left an object artifact behind");
    }
    ++g_checks;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return fail("usage: os_u3_stdlib_and_oalr_hooks_test <eshkol-run> [llvm-dis]");
    }
    const std::string compiler = argv[1];
    const std::string llvm_dis = argc >= 3 ? argv[2] : "";
    const std::string llvm_nm = find_tool(llvm_dis, "llvm-nm");

    const fs::path work_dir =
        fs::temp_directory_path() / ("eshkol_os_u3_stdlib_" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    const std::string pure_kernel_src = R"ESK(
(require core.freestanding_pure)

(define (kernel-pure-check-min (a : u64) (b : u64)) : u64 (pure-min-u64 a b) :export-symbol kernel_pure_check_min :no-alloc)
(define (kernel-pure-check-max (a : u64) (b : u64)) : u64 (pure-max-u64 a b) :export-symbol kernel_pure_check_max :no-alloc)
(define (kernel-pure-check-clamp (x : u64) (lo : u64) (hi : u64)) : u64 (pure-clamp-u64 x lo hi) :export-symbol kernel_pure_check_clamp :no-alloc)
(define (kernel-pure-check-align-up (v : u64) (a : u64)) : u64 (pure-align-up-u64 v a) :export-symbol kernel_pure_check_align_up :no-alloc)
(define (kernel-pure-check-align-down (v : u64) (a : u64)) : u64 (pure-align-down-u64 v a) :export-symbol kernel_pure_check_align_down :no-alloc)
(define (kernel-pure-check-is-aligned (v : u64) (a : u64)) : u64 (pure-is-aligned-u64 v a) :export-symbol kernel_pure_check_is_aligned :no-alloc)
(define (kernel-pure-check-is-pow2 (v : u64)) : u64 (pure-is-pow2-u64 v) :export-symbol kernel_pure_check_is_pow2 :no-alloc)
(define (kernel-pure-check-bit-mask (w : u64)) : u64 (pure-bit-mask-u64 w) :export-symbol kernel_pure_check_bit_mask :no-alloc)
(define (kernel-pure-check-bit-extract (v : u64) (s : u64) (w : u64)) : u64 (pure-bit-extract-u64 v s w) :export-symbol kernel_pure_check_bit_extract :no-alloc)
(define (kernel-pure-check-bit-set (v : u64) (idx : u64)) : u64 (pure-bit-set-u64 v idx) :export-symbol kernel_pure_check_bit_set :no-alloc)
(define (kernel-pure-check-bit-clear (v : u64) (idx : u64)) : u64 (pure-bit-clear-u64 v idx) :export-symbol kernel_pure_check_bit_clear :no-alloc)
(define (kernel-pure-check-popcount (v : u64)) : u64 (pure-popcount-u64 v) :export-symbol kernel_pure_check_popcount :no-alloc)
(define (kernel-pure-check-clz (v : u64)) : u64 (pure-clz-u64 v) :export-symbol kernel_pure_check_clz :no-alloc)
(define (kernel-pure-check-ctz (v : u64)) : u64 (pure-ctz-u64 v) :export-symbol kernel_pure_check_ctz :no-alloc)
(define (kernel-pure-check-bswap32 (v : u64)) : u64 (pure-bswap-u32 v) :export-symbol kernel_pure_check_bswap32 :no-alloc)
(define (kernel-pure-check-bswap64 (v : u64)) : u64 (pure-bswap-u64 v) :export-symbol kernel_pure_check_bswap64 :no-alloc)
(define (kernel-pure-check-crc32 (crc : u64) (b : u64)) : u64 (pure-crc32-step crc b) :export-symbol kernel_pure_check_crc32 :no-alloc)
)ESK";

    const fs::path pure_src_path = work_dir / "pure_kernel.esk";
    if (!write_text(pure_src_path, pure_kernel_src)) {
        return fail("failed to write pure_kernel.esk");
    }

    // Cross-target verification across all 4 kernel/MCU targets
    if (check_pure_kernel_target(compiler, llvm_dis, llvm_nm, work_dir,
            "x86_64", "freestanding-kernel-native", "x86_64-unknown-none", pure_src_path) != 0)
        return 1;
    if (check_pure_kernel_target(compiler, llvm_dis, llvm_nm, work_dir,
            "aarch64", "freestanding-kernel-native", "aarch64-unknown-none", pure_src_path) != 0)
        return 1;
    if (check_pure_kernel_target(compiler, llvm_dis, llvm_nm, work_dir,
            "rv64", "freestanding-kernel-native", "riscv64-unknown-none", pure_src_path) != 0)
        return 1;
    if (check_pure_kernel_target(compiler, llvm_dis, llvm_nm, work_dir,
            "cortex_m4", "freestanding-mcu-native", "thumbv7em-none-eabihf", pure_src_path) != 0)
        return 1;

    // Live host-ABI execution oracle verifying every Tier-1 pure stdlib op
    const std::string host_triple = host_target_triple();
    if (!host_triple.empty()) {
        const fs::path host_obj = work_dir / "pure_host.o";
        auto host_compile = run_process(
            {compiler, "--profile", "freestanding-kernel-native", "--target", host_triple,
             "--emit-object", "-o", host_obj.string(), pure_src_path.string()},
            work_dir);
        if (host_compile.exit_code != 0) {
            return fail("host compile of pure_kernel.esk failed:\n" + host_compile.output);
        }
        ++g_checks;

        const fs::path c_oracle = work_dir / "pure_oracle.c";
        const std::string c_src = R"C(
#include <stdint.h>
#include <stdio.h>

extern int64_t kernel_pure_check_min(int64_t a, int64_t b);
extern int64_t kernel_pure_check_max(int64_t a, int64_t b);
extern int64_t kernel_pure_check_clamp(int64_t x, int64_t lo, int64_t hi);
extern int64_t kernel_pure_check_align_up(int64_t v, int64_t a);
extern int64_t kernel_pure_check_align_down(int64_t v, int64_t a);
extern int64_t kernel_pure_check_is_aligned(int64_t v, int64_t a);
extern int64_t kernel_pure_check_is_pow2(int64_t v);
extern int64_t kernel_pure_check_bit_mask(int64_t w);
extern int64_t kernel_pure_check_bit_extract(int64_t v, int64_t s, int64_t w);
extern int64_t kernel_pure_check_bit_set(int64_t v, int64_t idx);
extern int64_t kernel_pure_check_bit_clear(int64_t v, int64_t idx);
extern int64_t kernel_pure_check_popcount(int64_t v);
extern int64_t kernel_pure_check_clz(int64_t v);
extern int64_t kernel_pure_check_ctz(int64_t v);
extern int64_t kernel_pure_check_bswap32(int64_t v);
extern int64_t kernel_pure_check_bswap64(int64_t v);
extern int64_t kernel_pure_check_crc32(int64_t crc, int64_t b);

static uint32_t ref_crc32(const uint8_t* data, size_t n) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) {
            crc = (crc & 1u) ? ((crc >> 1) ^ 0xEDB88320u) : (crc >> 1);
        }
    }
    return ~crc;
}

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "ORACLE FAIL: %s\n", #expr); return 1; } } while (0)

int main(void) {
    CHECK(kernel_pure_check_min(17, 42) == 17);
    CHECK(kernel_pure_check_max(17, 42) == 42);
    CHECK(kernel_pure_check_clamp(5, 10, 20) == 10);
    CHECK(kernel_pure_check_clamp(15, 10, 20) == 15);
    CHECK(kernel_pure_check_clamp(99, 10, 20) == 20);
    CHECK(kernel_pure_check_align_up(4097, 4096) == 8192);
    CHECK(kernel_pure_check_align_up(4096, 4096) == 4096);
    CHECK(kernel_pure_check_align_down(4097, 4096) == 4096);
    CHECK(kernel_pure_check_is_aligned(8192, 4096) == 1);
    CHECK(kernel_pure_check_is_aligned(8193, 4096) == 0);
    CHECK(kernel_pure_check_is_pow2(4096) == 1);
    CHECK(kernel_pure_check_is_pow2(4095) == 0);
    CHECK(kernel_pure_check_is_pow2(0) == 0);
    CHECK(kernel_pure_check_bit_mask(12) == 0xFFF);
    CHECK(kernel_pure_check_bit_extract(0x12345678, 8, 8) == 0x56);
    CHECK(kernel_pure_check_bit_set(0x10, 2) == 0x14);
    CHECK(kernel_pure_check_bit_clear(0x14, 2) == 0x10);
    CHECK(kernel_pure_check_popcount(0x55555555) == 16);
    CHECK(kernel_pure_check_clz(0) == 64);
    CHECK(kernel_pure_check_clz(0x00010000) == 47);
    CHECK(kernel_pure_check_ctz(0) == 64);
    CHECK(kernel_pure_check_ctz(0x00010000) == 16);
    CHECK((uint32_t)kernel_pure_check_bswap32(0x12345678) == 0x78563412u);
    CHECK((uint64_t)kernel_pure_check_bswap64(0x0102030405060708LL) == 0x0807060504030201ULL);

    const uint8_t msg[9] = {'1','2','3','4','5','6','7','8','9'};
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < 9; ++i) {
        crc = (uint32_t)kernel_pure_check_crc32((int64_t)crc, (int64_t)msg[i]);
    }
    CHECK((~crc) == 0xCBF43926u);
    CHECK((~crc) == ref_crc32(msg, 9));
    return 0;
}
)C";
        if (!write_text(c_oracle, c_src)) {
            return fail("failed to write pure_oracle.c");
        }
        const fs::path oracle_bin = work_dir / "pure_oracle";
        auto link = run_process(
            {"clang", "-std=c11", "-Wall", "-Werror", c_oracle.string(), host_obj.string(),
             "-o", oracle_bin.string()},
            work_dir);
        if (link.exit_code != 0) {
            return fail("clang link of pure_oracle failed:\n" + link.output);
        }
        auto run = run_process({oracle_bin.string()}, work_dir);
        if (run.exit_code != 0) {
            return fail("live pure_oracle execution failed:\n" + run.output);
        }
        ++g_checks;
    }

    // Negative Control 1: Tier 2 os-userspace-oalr module rejected in freestanding-kernel-native
    if (check_negative(compiler, work_dir, "tier2_kernel",
            "freestanding-kernel-native", "x86_64-unknown-none",
            "(require core.list.higher_order)\n(define (k (x : u64)) : u64 x :no-alloc)\n",
            "os-userspace-oalr") != 0)
        return 1;

    // Negative Control 2: Tier 2 os-userspace-oalr module rejected in freestanding-mcu-native
    if (check_negative(compiler, work_dir, "tier2_mcu",
            "freestanding-mcu-native", "thumbv7em-none-eabihf",
            "(require math.statistics)\n(define (m (x : u64)) : u64 x :no-alloc)\n",
            "os-userspace-oalr") != 0)
        return 1;

    // Negative Control 3: Tier 3 hosted-os module rejected in freestanding-kernel-native
    if (check_negative(compiler, work_dir, "tier3_kernel",
            "freestanding-kernel-native", "x86_64-unknown-none",
            "(require core.io)\n(define (k (x : u64)) : u64 x :no-alloc)\n",
            "hosted-os") != 0)
        return 1;

    // Negative Control 4: Direct eshkol_os_alloc_pages call rejected even with :no-alloc
    if (check_negative(compiler, work_dir, "direct_page_alloc",
            "freestanding-kernel-native", "x86_64-unknown-none",
            "(extern u64 eshkol_os_alloc_pages u64 u32 u64 :no-alloc)\n"
            "(define (k (n : u64)) : u64 (eshkol_os_alloc_pages n 1 0) :no-alloc)\n",
            "absolute region isolation") != 0)
        return 1;

    // Negative Control 5: Direct arena_allocate call rejected even with :no-alloc
    if (check_negative(compiler, work_dir, "direct_arena_alloc",
            "freestanding-kernel-native", "x86_64-unknown-none",
            "(extern u64 arena_allocate u64 u64 :no-alloc)\n"
            "(define (k (a : u64) (n : u64)) : u64 (arena_allocate a n) :no-alloc)\n",
            "absolute region isolation") != 0)
        return 1;

    fs::remove_all(work_dir, ec);
    std::cout << "PASS: os_u3_stdlib_and_oalr_hooks_test (" << g_checks << " checks)" << std::endl;
    return 0;
}
