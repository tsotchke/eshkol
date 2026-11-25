/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K4 Gate (`os_k4_drivers_vfs_test`):
 *   - Guard #1: Linear single-ownership handles (`DeviceRegion`, `IrqLease`,
 *     `DmaBuffer`) — monotonic generation tokens, ownership transfer invalidating
 *     prior handle, double-free rejection (`ESHKOL_DRV_ERR_DOUBLE_FREE`), and
 *     post-revocation rejection (`ESHKOL_DRV_ERR_REVOKED`).
 *   - Guard #2: DMA lease bounds overflow negative controls — 1-byte overshoot,
 *     straddling write, `SIZE_MAX` wraparound, and VirtIO descriptor segment
 *     overshoot (`ESHKOL_DRV_ERR_DMA_BOUNDS`), verified against sentinel guard
 *     bytes.
 *   - Guard #3: VirtIO split-virtqueue negative & positive controls — duplicate
 *     or already-in-flight descriptor chains (`ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC`),
 *     out-of-order and unsubmitted used-ring completions
 *     (`ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER`), plus full VirtIO block & net
 *     round-trips matching Phase U2 hardware struct layouts.
 *   - Guard #4: Atomic SHA-256 content-addressed VFS power-loss simulation —
 *     deterministic power cuts at mid-payload write, mid-index write,
 *     pre-superblock flip, and mid-superblock torn write (192/512 bytes),
 *     verifying zero partial/uncommitted entries, bit-for-bit SHA-256 integrity
 *     of all prior committed files, clean post-recovery commits, and bit-flip
 *     detection (`ESHKOL_DRV_ERR_VFS_CORRUPT`).
 *   - Guard #5: `llvm-nm -u` freestanding symbol audit on host
 *     `libeshkol-os-drivers-vfs.a` and bare-metal `x86_64-unknown-none`,
 *     `aarch64-unknown-none`, and `riscv64-unknown-none-elf` objects proving
 *     ZERO undefined external symbols.
 */

#include <eshkol/core/os_drivers_vfs.h>
#include <eshkol/core/os_syscall_abi.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
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

struct MockKernel {
    uint64_t syscall_count = 0;
    uint64_t live_mapped_pages = 0;
    uint64_t total_mapped = 0;
    uint64_t total_unmapped = 0;
    uint64_t irq_acks = 0;
    uint64_t notifies = 0;
    uint64_t revokes = 0;
};

MockKernel g_kernel;

}  // namespace

extern "C" int64_t eshkol_os_syscall6(uint64_t sysno,
                                      uint64_t arg0,
                                      uint64_t arg1,
                                      uint64_t arg2,
                                      uint64_t arg3,
                                      uint64_t arg4,
                                      uint64_t arg5) {
    (void)arg0;
    (void)arg1;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    g_kernel.syscall_count++;

    switch (sysno) {
    case ESHKOL_SYS_PAGE_MAP:
        g_kernel.live_mapped_pages += arg2;
        g_kernel.total_mapped += arg2;
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_PAGE_UNMAP:
        if (arg2 > g_kernel.live_mapped_pages) return ESHKOL_OS_ERR_INVALID_ARG;
        g_kernel.live_mapped_pages -= arg2;
        g_kernel.total_unmapped += arg2;
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_IPC_SUBMIT:
        g_kernel.notifies++;
        return static_cast<int64_t>(arg2);
    case ESHKOL_SYS_IPC_COMPLETE:
        g_kernel.irq_acks++;
        return 1;
    case ESHKOL_SYS_CAP_REVOKE:
        g_kernel.revokes++;
        return ESHKOL_OS_OK;
    default:
        return ESHKOL_OS_ERR_INVALID_SYSCALL;
    }
}

namespace {

/* ── Simulated VirtIO Block Device with Deterministic Power-Loss Injection ── */

enum class PowerCutMode {
    None,
    FailAfterNWrites,
    TearSuperblockWrite
};

struct SimVirtioBlkDevice {
    static constexpr uint32_t kTotalSectors = 256u;
    std::vector<uint8_t> disk;
    uint64_t write_ops = 0;
    uint64_t read_ops = 0;
    PowerCutMode cut_mode = PowerCutMode::None;
    uint64_t writes_remaining_before_cut = 0;
    size_t torn_bytes = 192u;
    bool power_cut_triggered = false;

    SimVirtioBlkDevice() : disk(kTotalSectors * ESHKOL_OS_VFS_SECTOR_SIZE, 0u) {}
};

int sim_virtio_blk_kick(eshkol_os_virtq_t* vq,
                        uint8_t* dma_base,
                        size_t dma_bytes,
                        void* dev_ctx) {
    auto* dev = static_cast<SimVirtioBlkDevice*>(dev_ctx);
    const uintptr_t base_va = reinterpret_cast<uintptr_t>(dma_base);

    const uint16_t head = vq->avail.ring[(vq->avail.idx - 1u) % ESHKOL_OS_VIRTQ_SIZE];
    const eshkol_virtq_desc_t& d0 = vq->desc[head];
    const eshkol_virtq_desc_t& d1 = vq->desc[d0.next];
    const eshkol_virtq_desc_t& d2 = vq->desc[d1.next];

    if (d0.addr < base_va || d1.addr < base_va || d2.addr < base_va) return -1;
    const size_t off0 = static_cast<size_t>(d0.addr - base_va);
    const size_t off1 = static_cast<size_t>(d1.addr - base_va);
    const size_t off2 = static_cast<size_t>(d2.addr - base_va);
    if (off0 + d0.len > dma_bytes || off1 + d1.len > dma_bytes || off2 + d2.len > dma_bytes) {
        return -1;
    }

    eshkol_virtio_blk_req_hdr_t hdr;
    std::memcpy(&hdr, dma_base + off0, sizeof(hdr));
    const size_t disk_off = static_cast<size_t>(hdr.sector) * ESHKOL_OS_VFS_SECTOR_SIZE;
    if (disk_off + d1.len > dev->disk.size()) return -1;

    if (hdr.type == ESHKOL_VIRTIO_BLK_T_OUT) {
        if (dev->cut_mode == PowerCutMode::FailAfterNWrites) {
            if (dev->writes_remaining_before_cut == 0u) {
                dev->power_cut_triggered = true;
                return -2;
            }
            dev->writes_remaining_before_cut--;
        } else if (dev->cut_mode == PowerCutMode::TearSuperblockWrite &&
                   (hdr.sector == 0u || hdr.sector == 1u)) {
            /* Persist only the leading torn_bytes of the superblock, simulating mid-sector power cut */
            const size_t partial = (dev->torn_bytes < d1.len) ? dev->torn_bytes : d1.len;
            std::memcpy(dev->disk.data() + disk_off, dma_base + off1, partial);
            dev->power_cut_triggered = true;
            return -3;
        }
        std::memcpy(dev->disk.data() + disk_off, dma_base + off1, d1.len);
        dev->write_ops++;
    } else if (hdr.type == ESHKOL_VIRTIO_BLK_T_IN) {
        std::memcpy(dma_base + off1, dev->disk.data() + disk_off, d1.len);
        dev->read_ops++;
    } else {
        return -1;
    }

    dma_base[off2] = ESHKOL_VIRTIO_BLK_S_OK;
    const uint16_t u_slot = vq->used.idx % ESHKOL_OS_VIRTQ_SIZE;
    vq->used.ring[u_slot].id = head;
    vq->used.ring[u_slot].len = (hdr.type == ESHKOL_VIRTIO_BLK_T_IN) ? (d1.len + 1u) : 1u;
    vq->used.idx = static_cast<uint16_t>(vq->used.idx + 1u);
    return 0;
}

/* ── Simulated VirtIO Net Loopback Device ── */

struct SimVirtioNetLoopback {
    std::vector<uint8_t> queued_frame;
    uint64_t tx_frames = 0;
    uint64_t rx_frames = 0;
};

int sim_virtio_net_kick(eshkol_os_virtq_t* vq,
                        uint8_t* dma_base,
                        size_t dma_bytes,
                        void* dev_ctx) {
    auto* net = static_cast<SimVirtioNetLoopback*>(dev_ctx);
    const uintptr_t base_va = reinterpret_cast<uintptr_t>(dma_base);
    const uint16_t head = vq->avail.ring[(vq->avail.idx - 1u) % ESHKOL_OS_VIRTQ_SIZE];
    const eshkol_virtq_desc_t& d0 = vq->desc[head];
    const eshkol_virtq_desc_t& d1 = vq->desc[d0.next];
    if (d0.addr < base_va || d1.addr < base_va) return -1;
    const size_t off0 = static_cast<size_t>(d0.addr - base_va);
    const size_t off1 = static_cast<size_t>(d1.addr - base_va);
    if (off0 + d0.len > dma_bytes || off1 + d1.len > dma_bytes) return -1;

    uint32_t used_len = 0u;
    if ((d1.flags & ESHKOL_VIRTQ_DESC_F_WRITE) == 0u) {
        /* TX: capture frame */
        net->queued_frame.assign(dma_base + off1, dma_base + off1 + d1.len);
        net->tx_frames++;
        used_len = d0.len + d1.len;
    } else {
        /* RX: deliver loopback frame */
        const size_t copy_len = (net->queued_frame.size() < d1.len) ? net->queued_frame.size() : d1.len;
        std::memset(dma_base + off0, 0, d0.len);
        if (copy_len > 0u) {
            std::memcpy(dma_base + off1, net->queued_frame.data(), copy_len);
        }
        net->rx_frames++;
        used_len = static_cast<uint32_t>(d0.len + copy_len);
    }

    const uint16_t u_slot = vq->used.idx % ESHKOL_OS_VIRTQ_SIZE;
    vq->used.ring[u_slot].id = head;
    vq->used.ring[u_slot].len = used_len;
    vq->used.idx = static_cast<uint16_t>(vq->used.idx + 1u);
    return 0;
}

/* ── Subprocess & llvm-nm Audit Helpers ── */

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

std::string hex_digest(const uint8_t d[ESHKOL_OS_SHA256_BYTES]) {
    static const char* lut = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (size_t i = 0; i < ESHKOL_OS_SHA256_BYTES; ++i) {
        out.push_back(lut[(d[i] >> 4) & 0xFu]);
        out.push_back(lut[d[i] & 0xFu]);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        return fail("usage: os_k4_drivers_vfs_test <llvm-dis> <repo-root> <os-drivers-vfs-lib>");
    }
    const std::string llvm_dis     = argv[1] ? argv[1] : "";
    const fs::path repo_root       = argv[2];
    const fs::path drivers_vfs_lib = argv[3];
    const std::string llvm_nm      = find_tool(llvm_dis, "llvm-nm");

    const fs::path work_dir =
        fs::temp_directory_path() / ("eshkol_os_k4_drv_" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    /* ── Initialize userspace OALR driver runtime strictly over eshkol_os_syscall6 ── */
    eshkol_os_drv_runtime_t rt;
    eshkol_os_drv_runtime_config_t bad_cfg;
    std::memset(&bad_cfg, 0, sizeof(bad_cfg));
    CHECK(eshkol_os_drv_runtime_init(&rt, &bad_cfg) == ESHKOL_DRV_ERR_INVALID_ARG,
          "runtime init without syscall_fn must fail closed");

    eshkol_os_drv_runtime_config_t cfg;
    cfg.syscall_fn = eshkol_os_syscall6;
    cfg.page_size = 4096u;
    CHECK(eshkol_os_drv_runtime_init(&rt, &cfg) == ESHKOL_DRV_OK,
          "driver runtime init failed");

    /* ── GUARD #1: Linear Single-Ownership, Double-Free & Use-After-Revoke ── */
    {
        alignas(16) uint8_t mmio_regs[64];
        std::memset(mmio_regs, 0, sizeof(mmio_regs));
        alignas(4096) uint8_t dma_mem[4096];

        /* 1a. DeviceRegion acquire, transfer, stale-handle rejection, double-free */
        eshkol_os_device_region_t reg1, reg2;
        CHECK(eshkol_os_device_region_acquire(&rt, 100, mmio_regs, sizeof(mmio_regs), &reg1) == ESHKOL_DRV_OK,
              "acquire DeviceRegion");
        const eshkol_os_device_region_t reg1_copy = reg1;
        CHECK(eshkol_os_device_region_write32(&rt, &reg1, 0u, 0xCAFEBABEu) == ESHKOL_DRV_OK,
              "valid MMIO write32");
        uint32_t v = 0u;
        CHECK(eshkol_os_device_region_read32(&rt, &reg1, 0u, &v) == ESHKOL_DRV_OK && v == 0xCAFEBABEu,
              "valid MMIO read32");
        CHECK(eshkol_os_device_region_write32(&rt, &reg1, 64u, 1u) == ESHKOL_DRV_ERR_MMIO_BOUNDS,
              "MMIO bounds overshoot rejected");

        /* Linear ownership transfer invalidates reg1 and pre-transfer copy */
        CHECK(eshkol_os_device_region_transfer(&rt, &reg1, &reg2) == ESHKOL_DRV_OK,
              "linear DeviceRegion transfer");
        CHECK(eshkol_os_device_region_read32(&rt, &reg1, 0u, &v) == ESHKOL_DRV_ERR_STALE_HANDLE,
              "transferred source DeviceRegion rejected");
        CHECK(eshkol_os_device_region_read32(&rt, &reg1_copy, 0u, &v) == ESHKOL_DRV_ERR_STALE_HANDLE,
              "pre-transfer copied DeviceRegion rejected");
        CHECK(eshkol_os_device_region_write32(&rt, &reg2, 4u, 0x12345678u) == ESHKOL_DRV_OK,
              "transferred destination DeviceRegion active");

        /* Single-ownership release & double-free (both same variable & saved copy) */
        eshkol_os_device_region_t reg2_copy = reg2;
        CHECK(eshkol_os_device_region_release(&rt, &reg2) == ESHKOL_DRV_OK,
              "release DeviceRegion");
        CHECK(eshkol_os_device_region_release(&rt, &reg2) == ESHKOL_DRV_ERR_DOUBLE_FREE,
              "same-var DeviceRegion double-free rejected");
        CHECK(eshkol_os_device_region_release(&rt, &reg2_copy) == ESHKOL_DRV_ERR_DOUBLE_FREE,
              "copied-handle DeviceRegion double-free rejected");
        CHECK(eshkol_os_device_region_read32(&rt, &reg2_copy, 0u, &v) == ESHKOL_DRV_ERR_STALE_HANDLE,
              "use-after-release DeviceRegion rejected");

        /* 1b. IrqLease acquire, ack, transfer, release, double-free */
        eshkol_os_irq_lease_t irq1, irq2;
        CHECK(eshkol_os_irq_lease_acquire(&rt, 101, 42u, &irq1) == ESHKOL_DRV_OK, "acquire IrqLease");
        const eshkol_os_irq_lease_t irq1_copy = irq1;
        CHECK(eshkol_os_irq_lease_ack(&rt, &irq1) == ESHKOL_DRV_OK, "valid IrqLease ack");
        CHECK(eshkol_os_irq_lease_transfer(&rt, &irq1, &irq2) == ESHKOL_DRV_OK, "transfer IrqLease");
        CHECK(eshkol_os_irq_lease_ack(&rt, &irq1_copy) == ESHKOL_DRV_ERR_STALE_HANDLE,
              "stale IrqLease copy rejected");
        CHECK(eshkol_os_irq_lease_ack(&rt, &irq2) == ESHKOL_DRV_OK, "new IrqLease owner ack");
        eshkol_os_irq_lease_t irq2_copy = irq2;
        CHECK(eshkol_os_irq_lease_release(&rt, &irq2) == ESHKOL_DRV_OK, "release IrqLease");
        CHECK(eshkol_os_irq_lease_release(&rt, &irq2) == ESHKOL_DRV_ERR_DOUBLE_FREE,
              "same-var IrqLease double-free rejected");
        CHECK(eshkol_os_irq_lease_release(&rt, &irq2_copy) == ESHKOL_DRV_ERR_DOUBLE_FREE,
              "copied-handle IrqLease double-free rejected");

        /* 1c. DmaBuffer acquire, transfer, release, double-free & page ledger */
        eshkol_os_dma_buffer_t dma1, dma2;
        CHECK(eshkol_os_dma_buffer_acquire(&rt, 10, 102, dma_mem, sizeof(dma_mem), &dma1) == ESHKOL_DRV_OK,
              "acquire DmaBuffer");
        CHECK(g_kernel.live_mapped_pages == 1u && rt.live_dma_pages == 1u,
              "DMA acquire must map 1 page via PAGE_MAP");
        const eshkol_os_dma_buffer_t dma1_copy = dma1;
        CHECK(eshkol_os_dma_buffer_transfer(&rt, &dma1, &dma2) == ESHKOL_DRV_OK, "transfer DmaBuffer");
        const uint8_t pat[4] = {1, 2, 3, 4};
        CHECK(eshkol_os_dma_write(&rt, &dma1_copy, 0u, pat, 4u) == ESHKOL_DRV_ERR_STALE_HANDLE,
              "stale DmaBuffer copy rejected");
        CHECK(eshkol_os_dma_write(&rt, &dma2, 0u, pat, 4u) == ESHKOL_DRV_OK, "valid DmaBuffer write");
        eshkol_os_dma_buffer_t dma2_copy = dma2;
        CHECK(eshkol_os_dma_buffer_release(&rt, &dma2) == ESHKOL_DRV_OK, "release DmaBuffer");
        CHECK(g_kernel.live_mapped_pages == 0u && rt.live_dma_pages == 0u,
              "DMA release must unmap page via PAGE_UNMAP");
        CHECK(eshkol_os_dma_buffer_release(&rt, &dma2) == ESHKOL_DRV_ERR_DOUBLE_FREE,
              "same-var DmaBuffer double-free rejected");
        CHECK(eshkol_os_dma_buffer_release(&rt, &dma2_copy) == ESHKOL_DRV_ERR_DOUBLE_FREE,
              "copied-handle DmaBuffer double-free rejected");

        /* 1d. Capability revocation invalidates all 3 linear handle classes */
        eshkol_os_device_region_t rev_reg;
        eshkol_os_irq_lease_t rev_irq;
        eshkol_os_dma_buffer_t rev_dma;
        CHECK(eshkol_os_device_region_acquire(&rt, 200, mmio_regs, sizeof(mmio_regs), &rev_reg) == ESHKOL_DRV_OK,
              "acquire revocable DeviceRegion");
        CHECK(eshkol_os_irq_lease_acquire(&rt, 200, 43u, &rev_irq) == ESHKOL_DRV_OK,
              "acquire revocable IrqLease");
        CHECK(eshkol_os_dma_buffer_acquire(&rt, 10, 200, dma_mem, sizeof(dma_mem), &rev_dma) == ESHKOL_DRV_OK,
              "acquire revocable DmaBuffer");
        CHECK(g_kernel.live_mapped_pages == 1u, "expected 1 live DMA page before revoke");

        CHECK(eshkol_os_drv_revoke_cap(&rt, 200) == ESHKOL_DRV_OK, "revoke cap 200");
        CHECK(g_kernel.live_mapped_pages == 0u && rt.live_dma_pages == 0u,
              "revoke must unmap DMA pages immediately");

        CHECK(eshkol_os_device_region_read32(&rt, &rev_reg, 0u, &v) == ESHKOL_DRV_ERR_REVOKED,
              "DeviceRegion read after revoke must return REVOKED");
        CHECK(eshkol_os_device_region_write32(&rt, &rev_reg, 0u, 99u) == ESHKOL_DRV_ERR_REVOKED,
              "DeviceRegion write after revoke must return REVOKED");
        CHECK(eshkol_os_irq_lease_ack(&rt, &rev_irq) == ESHKOL_DRV_ERR_REVOKED,
              "IrqLease ack after revoke must return REVOKED");
        uint8_t tmp[4] = {0};
        CHECK(eshkol_os_dma_write(&rt, &rev_dma, 0u, pat, 4u) == ESHKOL_DRV_ERR_REVOKED,
              "DmaBuffer write after revoke must return REVOKED");
        CHECK(eshkol_os_dma_read(&rt, &rev_dma, 0u, tmp, 4u) == ESHKOL_DRV_ERR_REVOKED,
              "DmaBuffer read after revoke must return REVOKED");

        std::cout << "PASS Guard #1 (Linear Handles): transfer/double-free/use-after-revoke all fail closed, 0 leaked DMA pages" << std::endl;
    }

    /* ── GUARD #2: DMA Lease Bounds Overflow Negative Control (+ Guard Bytes) ── */
    {
        static constexpr size_t kLeaseBytes = 4096u;
        static constexpr size_t kGuardBytes = 64u;
        alignas(4096) uint8_t dma_guarded[kLeaseBytes + kGuardBytes];
        std::memset(dma_guarded + kLeaseBytes, 0xA5, kGuardBytes);

        alignas(16) uint8_t mmio_regs[64];
        std::memset(mmio_regs, 0, sizeof(mmio_regs));

        eshkol_os_device_region_t reg;
        eshkol_os_irq_lease_t irq;
        eshkol_os_dma_buffer_t dma;
        CHECK(eshkol_os_device_region_acquire(&rt, 300, mmio_regs, sizeof(mmio_regs), &reg) == ESHKOL_DRV_OK, "acquire reg");
        CHECK(eshkol_os_irq_lease_acquire(&rt, 301, 50u, &irq) == ESHKOL_DRV_OK, "acquire irq");
        CHECK(eshkol_os_dma_buffer_acquire(&rt, 10, 302, dma_guarded, kLeaseBytes, &dma) == ESHKOL_DRV_OK,
              "acquire guarded DMA buffer");

        uint8_t payload[16];
        std::memset(payload, 0x5A, sizeof(payload));
        uint8_t out_buf[16];

        /* 1-byte past end */
        CHECK(eshkol_os_dma_write(&rt, &dma, kLeaseBytes, payload, 1u) == ESHKOL_DRV_ERR_DMA_BOUNDS,
              "1-byte past end write must fail with DMA_BOUNDS");
        /* Straddling end by 1 byte */
        CHECK(eshkol_os_dma_write(&rt, &dma, kLeaseBytes - 6u, payload, 7u) == ESHKOL_DRV_ERR_DMA_BOUNDS,
              "straddling overshoot write must fail with DMA_BOUNDS");
        /* Integer overflow wraparound */
        const size_t wrap_off = std::numeric_limits<size_t>::max() - 3u;
        CHECK(eshkol_os_dma_write(&rt, &dma, wrap_off, payload, 16u) == ESHKOL_DRV_ERR_DMA_BOUNDS,
              "integer-overflow wraparound write must fail with DMA_BOUNDS");
        CHECK(eshkol_os_dma_read(&rt, &dma, kLeaseBytes, out_buf, 1u) == ESHKOL_DRV_ERR_DMA_BOUNDS,
              "1-byte past end read must fail with DMA_BOUNDS");
        CHECK(eshkol_os_dma_read(&rt, &dma, wrap_off, out_buf, 16u) == ESHKOL_DRV_ERR_DMA_BOUNDS,
              "integer-overflow wraparound read must fail with DMA_BOUNDS");

        /* VirtIO descriptor extending 1 byte past lease_bytes */
        eshkol_os_virtq_t vq;
        CHECK(eshkol_os_virtq_init(&vq, &rt, reg, irq, dma, 303, 0u) == ESHKOL_DRV_OK, "vq init");
        eshkol_os_virtq_seg_t bad_seg;
        bad_seg.desc_idx = 0u;
        bad_seg.flags = 0u;
        bad_seg.dma_offset = kLeaseBytes - 6u;
        bad_seg.len = 7u;
        CHECK(eshkol_os_virtq_submit_chain(&vq, &bad_seg, 1u) == ESHKOL_DRV_ERR_DMA_BOUNDS,
              "VirtIO descriptor extending 1B past lease must fail with DMA_BOUNDS");
        CHECK(vq.avail.idx == 0u && vq.in_flight_mask == 0u,
              "rejected VirtIO DMA overshoot must leave ring untouched");

        /* Verify all 64 trailing sentinel guard bytes are completely intact */
        bool guard_intact = true;
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (dma_guarded[kLeaseBytes + i] != 0xA5u) { guard_intact = false; break; }
        }
        CHECK(guard_intact, "trailing DMA guard bytes modified");

        CHECK(eshkol_os_dma_buffer_release(&rt, &dma) == ESHKOL_DRV_OK, "release guarded DMA");
        CHECK(eshkol_os_irq_lease_release(&rt, &irq) == ESHKOL_DRV_OK, "release irq");
        CHECK(eshkol_os_device_region_release(&rt, &reg) == ESHKOL_DRV_OK, "release reg");
        std::cout << "PASS Guard #2 (DMA Lease Bounds): 1-byte overshoot, wraparound & VirtIO segment overflow all DMA_BOUNDS, 64/64 guard bytes intact" << std::endl;
    }

    /* ── GUARD #3: VirtIO Split-Virtqueue Duplicate/Out-of-Order Controls + Block/Net Round-Trips ── */
    {
        alignas(16) uint8_t mmio_regs[64];
        std::memset(mmio_regs, 0, sizeof(mmio_regs));
        alignas(4096) uint8_t dma_mem[8192];

        eshkol_os_device_region_t reg;
        eshkol_os_irq_lease_t irq;
        eshkol_os_dma_buffer_t dma;
        CHECK(eshkol_os_device_region_acquire(&rt, 400, mmio_regs, sizeof(mmio_regs), &reg) == ESHKOL_DRV_OK, "reg");
        CHECK(eshkol_os_irq_lease_acquire(&rt, 401, 51u, &irq) == ESHKOL_DRV_OK, "irq");
        CHECK(eshkol_os_dma_buffer_acquire(&rt, 10, 402, dma_mem, sizeof(dma_mem), &dma) == ESHKOL_DRV_OK, "dma");

        eshkol_os_virtq_t vq;
        CHECK(eshkol_os_virtq_init(&vq, &rt, reg, irq, dma, 403, 0u) == ESHKOL_DRV_OK, "vq init");

        /* 3a: Duplicate descriptor inside single chain (cyclic self-chain {3, 3}) */
        eshkol_os_virtq_seg_t dup_self[2] = {
            {3u, 0u, 64u, 0u},
            {3u, ESHKOL_VIRTQ_DESC_F_WRITE, 64u, 64u}
        };
        CHECK(eshkol_os_virtq_submit_chain(&vq, dup_self, 2u) == ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC,
              "duplicate descriptor within chain must fail with VQ_DUPLICATE_DESC");
        CHECK(vq.avail.idx == 0u && vq.in_flight_mask == 0u, "duplicate chain must not mutate ring");

        /* 3b: Submit valid chain {0, 1}, then attempt to reuse in-flight desc 1 */
        eshkol_os_virtq_seg_t ch0[2] = {
            {0u, 0u, 64u, 0u},
            {1u, ESHKOL_VIRTQ_DESC_F_WRITE, 64u, 64u}
        };
        CHECK(eshkol_os_virtq_submit_chain(&vq, ch0, 2u) == ESHKOL_DRV_OK, "submit chain {0,1}");
        eshkol_os_virtq_seg_t reuse_inflight[2] = {
            {1u, 0u, 64u, 128u},
            {2u, ESHKOL_VIRTQ_DESC_F_WRITE, 64u, 192u}
        };
        CHECK(eshkol_os_virtq_submit_chain(&vq, reuse_inflight, 2u) == ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC,
              "reusing in-flight descriptor must fail with VQ_DUPLICATE_DESC");

        /* 3c: Submit second valid chain {2, 3}, then inject out-of-order & spurious completions */
        eshkol_os_virtq_seg_t ch1[2] = {
            {2u, 0u, 64u, 128u},
            {3u, ESHKOL_VIRTQ_DESC_F_WRITE, 64u, 192u}
        };
        CHECK(eshkol_os_virtq_submit_chain(&vq, ch1, 2u) == ESHKOL_DRV_OK, "submit chain {2,3}");

        /* Inject out-of-order head id=2 when head id=0 is expected next */
        vq.used.ring[0].id = 2u;
        vq.used.ring[0].len = 64u;
        vq.used.idx = 1u;
        eshkol_os_virtq_completion_t comp;
        CHECK(eshkol_os_virtq_complete_one(&vq, &comp) == ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER,
              "out-of-order used completion must fail with VQ_OUT_OF_ORDER");

        /* Inject unsubmitted/spurious head id=9 */
        vq.used.ring[0].id = 9u;
        CHECK(eshkol_os_virtq_complete_one(&vq, &comp) == ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER,
              "spurious unsubmitted descriptor completion must fail with VQ_OUT_OF_ORDER");

        /* Deliver valid in-order completions (0 then 2) */
        vq.used.ring[0].id = 0u;
        vq.used.ring[0].len = 64u;
        vq.used.ring[1].id = 2u;
        vq.used.ring[1].len = 64u;
        vq.used.idx = 2u;
        CHECK(eshkol_os_virtq_complete_one(&vq, &comp) == ESHKOL_DRV_OK && comp.head_id == 0u,
              "in-order completion 0");
        CHECK(eshkol_os_virtq_complete_one(&vq, &comp) == ESHKOL_DRV_OK && comp.head_id == 2u,
              "in-order completion 2");
        CHECK(vq.in_flight_mask == 0u && vq.fifo_count == 0u,
              "all descriptors released after completions");

        /* 3d: Valid VirtIO Block round-trip */
        SimVirtioBlkDevice blk_sim;
        eshkol_os_blk_dev_t blk_dev;
        blk_dev.vq = &vq;
        blk_dev.kick_fn = sim_virtio_blk_kick;
        blk_dev.dev_ctx = &blk_sim;
        blk_dev.total_sectors = SimVirtioBlkDevice::kTotalSectors;
        blk_dev.dma_scratch_offset = 0u;

        uint8_t sec_wr[ESHKOL_OS_VFS_SECTOR_SIZE];
        uint8_t sec_rd[ESHKOL_OS_VFS_SECTOR_SIZE];
        for (size_t i = 0; i < sizeof(sec_wr); ++i) sec_wr[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xFFu);
        std::memset(sec_rd, 0, sizeof(sec_rd));
        CHECK(eshkol_os_virtio_blk_io(&blk_dev, 1, 5u, sec_wr, 1u) == ESHKOL_DRV_OK, "virtio-blk write");
        CHECK(eshkol_os_virtio_blk_io(&blk_dev, 0, 5u, sec_rd, 1u) == ESHKOL_DRV_OK, "virtio-blk read");
        CHECK(std::memcmp(sec_wr, sec_rd, sizeof(sec_wr)) == 0, "virtio-blk payload integrity");

        /* 3e: Valid VirtIO Net TX + RX loopback round-trip */
        SimVirtioNetLoopback net_sim;
        const std::string tx_frame = "ESHKOL-OS-K4-VIRTIO-NET-LOOPBACK-FRAME-PAYLOAD-1234567890";
        std::vector<uint8_t> tx_buf(tx_frame.begin(), tx_frame.end());
        std::vector<uint8_t> rx_buf(tx_frame.size(), 0u);
        size_t tx_done = 0u, rx_done = 0u;
        CHECK(eshkol_os_virtio_net_xfer(&vq, sim_virtio_net_kick, &net_sim, 1, 1024u,
                                        tx_buf.data(), tx_buf.size(), &tx_done) == ESHKOL_DRV_OK,
              "virtio-net TX");
        CHECK(eshkol_os_virtio_net_xfer(&vq, sim_virtio_net_kick, &net_sim, 0, 2048u,
                                        rx_buf.data(), rx_buf.size(), &rx_done) == ESHKOL_DRV_OK,
              "virtio-net RX");
        CHECK(rx_done == tx_buf.size() && std::memcmp(tx_buf.data(), rx_buf.data(), tx_buf.size()) == 0,
              "virtio-net loopback frame integrity");

        CHECK(eshkol_os_dma_buffer_release(&rt, &dma) == ESHKOL_DRV_OK, "release vq dma");
        CHECK(eshkol_os_irq_lease_release(&rt, &irq) == ESHKOL_DRV_OK, "release vq irq");
        CHECK(eshkol_os_device_region_release(&rt, &reg) == ESHKOL_DRV_OK, "release vq reg");
        std::cout << "PASS Guard #3 (VirtIO Split-Virtqueue): duplicate/in-flight/out-of-order/spurious rejected; blk & net round-trips verified" << std::endl;
    }

    /* ── GUARD #4: Atomic Content-Addressed VFS & Mid-Write Power-Loss Simulation ── */
    {
        /* 4a: Verify FIPS 180-4 SHA-256 test vectors */
        uint8_t d_empty[ESHKOL_OS_SHA256_BYTES];
        uint8_t d_abc[ESHKOL_OS_SHA256_BYTES];
        eshkol_os_sha256("", 0u, d_empty);
        eshkol_os_sha256("abc", 3u, d_abc);
        CHECK(hex_digest(d_empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "SHA-256 empty vector mismatch");
        CHECK(hex_digest(d_abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              "SHA-256 'abc' vector mismatch");

        /* Helper to stand up a clean VirtIO block stack over persistent disk storage */
        SimVirtioBlkDevice disk_dev;
        alignas(16) uint8_t mmio_regs[64];
        std::memset(mmio_regs, 0, sizeof(mmio_regs));
        alignas(4096) uint8_t dma_mem[8192];

        eshkol_os_device_region_t reg;
        eshkol_os_irq_lease_t irq;
        eshkol_os_dma_buffer_t dma;
        CHECK(eshkol_os_device_region_acquire(&rt, 500, mmio_regs, sizeof(mmio_regs), &reg) == ESHKOL_DRV_OK, "vfs reg");
        CHECK(eshkol_os_irq_lease_acquire(&rt, 501, 52u, &irq) == ESHKOL_DRV_OK, "vfs irq");
        CHECK(eshkol_os_dma_buffer_acquire(&rt, 10, 502, dma_mem, sizeof(dma_mem), &dma) == ESHKOL_DRV_OK, "vfs dma");

        auto make_blk = [&](eshkol_os_virtq_t& vq_out, eshkol_os_blk_dev_t& blk_out) {
            CHECK(eshkol_os_virtq_init(&vq_out, &rt, reg, irq, dma, 503, 0u) == ESHKOL_DRV_OK,
                  "reinit vq");
            blk_out.vq = &vq_out;
            blk_out.kick_fn = sim_virtio_blk_kick;
            blk_out.dev_ctx = &disk_dev;
            blk_out.total_sectors = SimVirtioBlkDevice::kTotalSectors;
            blk_out.dma_scratch_offset = 0u;
            return 0;
        };

        eshkol_os_virtq_t vq;
        eshkol_os_blk_dev_t blk;
        if (make_blk(vq, blk) != 0) return 1;

        eshkol_os_cas_vfs_t vfs;
        CHECK(eshkol_os_cas_vfs_format(&vfs, &blk) == ESHKOL_DRV_OK, "format CAS VFS");

        /* Commit baseline directory + 2 multi-sector files */
        uint32_t etc_dir_id = 0u;
        CHECK(eshkol_os_cas_vfs_mkdir(&vfs, "etc", 0u, &etc_dir_id) == ESHKOL_DRV_OK && etc_dir_id == 1u,
              "mkdir /etc");

        std::vector<uint8_t> boot_data(1400u);
        for (size_t i = 0; i < boot_data.size(); ++i) boot_data[i] = static_cast<uint8_t>((i * 29u + 7u) & 0xFFu);
        std::vector<uint8_t> cfg_data(700u);
        for (size_t i = 0; i < cfg_data.size(); ++i) cfg_data[i] = static_cast<uint8_t>((i * 13u + 3u) & 0xFFu);

        uint8_t boot_sha[ESHKOL_OS_SHA256_BYTES];
        uint8_t cfg_sha[ESHKOL_OS_SHA256_BYTES];
        CHECK(eshkol_os_cas_vfs_put_file(&vfs, "boot.esk", 0u, boot_data.data(), boot_data.size(), boot_sha) == ESHKOL_DRV_OK,
              "commit boot.esk");
        CHECK(eshkol_os_cas_vfs_put_file(&vfs, "config.cfg", etc_dir_id, cfg_data.data(), cfg_data.size(), cfg_sha) == ESHKOL_DRV_OK,
              "commit etc/config.cfg");

        const uint64_t baseline_gen = vfs.generation;
        const uint32_t baseline_entries = vfs.entry_count;

        /* Verify content-addressed read by hash */
        std::vector<uint8_t> rd_buf(2048u);
        size_t rd_len = 0u;
        CHECK(eshkol_os_cas_vfs_read_by_hash(&vfs, boot_sha, rd_buf.data(), rd_buf.size(), &rd_len) == ESHKOL_DRV_OK &&
                  rd_len == boot_data.size() &&
                  std::memcmp(rd_buf.data(), boot_data.data(), boot_data.size()) == 0,
              "CAS read_by_hash boot.esk");

        /* Helper to assert remounted state matches baseline bit-for-bit with zero uncommitted entries */
        auto verify_remount_baseline = [&](const std::string& stage_label) {
            disk_dev.cut_mode = PowerCutMode::None;
            eshkol_os_virtq_t fresh_vq;
            eshkol_os_blk_dev_t fresh_blk;
            if (make_blk(fresh_vq, fresh_blk) != 0) return 1;
            eshkol_os_cas_vfs_t mounted;
            CHECK(eshkol_os_cas_vfs_mount(&mounted, &fresh_blk) == ESHKOL_DRV_OK,
                  stage_label + ": remount after power loss failed");
            CHECK(mounted.generation == baseline_gen && mounted.entry_count == baseline_entries,
                  stage_label + ": unexpected generation or uncommitted entry leaked into mounted VFS");

            /* Uncommitted file MUST NOT exist */
            size_t tmp_len = 0u;
            CHECK(eshkol_os_cas_vfs_read_file(&mounted, "staged.bin", 0u, rd_buf.data(), rd_buf.size(), &tmp_len, nullptr) ==
                      ESHKOL_DRV_ERR_VFS_NOT_FOUND,
                  stage_label + ": uncommitted staged.bin must not appear after power loss");

            /* Prior committed files MUST read back bit-for-bit with exact SHA-256 */
            uint8_t chk_sha[ESHKOL_OS_SHA256_BYTES];
            CHECK(eshkol_os_cas_vfs_read_file(&mounted, "boot.esk", 0u, rd_buf.data(), rd_buf.size(), &tmp_len, chk_sha) ==
                      ESHKOL_DRV_OK &&
                      tmp_len == boot_data.size() &&
                      std::memcmp(rd_buf.data(), boot_data.data(), boot_data.size()) == 0 &&
                      std::memcmp(chk_sha, boot_sha, ESHKOL_OS_SHA256_BYTES) == 0,
                  stage_label + ": boot.esk corrupted or lost after power cut");

            CHECK(eshkol_os_cas_vfs_read_file(&mounted, "config.cfg", etc_dir_id, rd_buf.data(), rd_buf.size(), &tmp_len, chk_sha) ==
                      ESHKOL_DRV_OK &&
                      tmp_len == cfg_data.size() &&
                      std::memcmp(rd_buf.data(), cfg_data.data(), cfg_data.size()) == 0 &&
                      std::memcmp(chk_sha, cfg_sha, ESHKOL_OS_SHA256_BYTES) == 0,
                  stage_label + ": config.cfg corrupted or lost after power cut");
            return 0;
        };

        std::vector<uint8_t> staged_data(2048u, 0xCCu); /* 4 payload sectors */
        uint8_t staged_sha[ESHKOL_OS_SHA256_BYTES];

        /* Power-loss Cut A: mid-payload write (cut after 2 of 4 payload sectors) */
        {
            eshkol_os_virtq_t vq_a;
            eshkol_os_blk_dev_t blk_a;
            if (make_blk(vq_a, blk_a) != 0) return 1;
            eshkol_os_cas_vfs_t vfs_a;
            CHECK(eshkol_os_cas_vfs_mount(&vfs_a, &blk_a) == ESHKOL_DRV_OK, "mount before Cut A");
            disk_dev.cut_mode = PowerCutMode::FailAfterNWrites;
            disk_dev.writes_remaining_before_cut = 2u;
            disk_dev.power_cut_triggered = false;
            CHECK(eshkol_os_cas_vfs_put_file(&vfs_a, "staged.bin", 0u, staged_data.data(), staged_data.size(), staged_sha) ==
                      ESHKOL_DRV_ERR_DEVICE_IO,
                  "Cut A must abort cleanly with DEVICE_IO");
            CHECK(disk_dev.power_cut_triggered, "Cut A fault must fire");
            if (verify_remount_baseline("Cut A (mid-payload)") != 0) return 1;
        }

        /* Power-loss Cut B: mid-index write (4 payload sectors + 3 of 6 index sectors = 7 writes) */
        {
            eshkol_os_virtq_t vq_b;
            eshkol_os_blk_dev_t blk_b;
            if (make_blk(vq_b, blk_b) != 0) return 1;
            eshkol_os_cas_vfs_t vfs_b;
            CHECK(eshkol_os_cas_vfs_mount(&vfs_b, &blk_b) == ESHKOL_DRV_OK, "mount before Cut B");
            disk_dev.cut_mode = PowerCutMode::FailAfterNWrites;
            disk_dev.writes_remaining_before_cut = 7u;
            disk_dev.power_cut_triggered = false;
            CHECK(eshkol_os_cas_vfs_put_file(&vfs_b, "staged.bin", 0u, staged_data.data(), staged_data.size(), staged_sha) ==
                      ESHKOL_DRV_ERR_DEVICE_IO,
                  "Cut B must abort cleanly with DEVICE_IO");
            CHECK(disk_dev.power_cut_triggered, "Cut B fault must fire");
            if (verify_remount_baseline("Cut B (mid-index)") != 0) return 1;
        }

        /* Power-loss Cut C: right before superblock flip (4 payload + 6 index = 10 writes) */
        {
            eshkol_os_virtq_t vq_c;
            eshkol_os_blk_dev_t blk_c;
            if (make_blk(vq_c, blk_c) != 0) return 1;
            eshkol_os_cas_vfs_t vfs_c;
            CHECK(eshkol_os_cas_vfs_mount(&vfs_c, &blk_c) == ESHKOL_DRV_OK, "mount before Cut C");
            disk_dev.cut_mode = PowerCutMode::FailAfterNWrites;
            disk_dev.writes_remaining_before_cut = 10u;
            disk_dev.power_cut_triggered = false;
            CHECK(eshkol_os_cas_vfs_put_file(&vfs_c, "staged.bin", 0u, staged_data.data(), staged_data.size(), staged_sha) ==
                      ESHKOL_DRV_ERR_DEVICE_IO,
                  "Cut C must abort cleanly with DEVICE_IO");
            CHECK(disk_dev.power_cut_triggered, "Cut C fault must fire");
            if (verify_remount_baseline("Cut C (pre-superblock)") != 0) return 1;
        }

        /* Power-loss Cut D: mid-superblock torn write (192/512 bytes persisted) */
        {
            eshkol_os_virtq_t vq_d;
            eshkol_os_blk_dev_t blk_d;
            if (make_blk(vq_d, blk_d) != 0) return 1;
            eshkol_os_cas_vfs_t vfs_d;
            CHECK(eshkol_os_cas_vfs_mount(&vfs_d, &blk_d) == ESHKOL_DRV_OK, "mount before Cut D");
            disk_dev.cut_mode = PowerCutMode::TearSuperblockWrite;
            disk_dev.torn_bytes = 192u;
            disk_dev.power_cut_triggered = false;
            CHECK(eshkol_os_cas_vfs_put_file(&vfs_d, "boot.esk", 0u, staged_data.data(), staged_data.size(), staged_sha) ==
                      ESHKOL_DRV_ERR_DEVICE_IO,
                  "Cut D torn superblock write must fail cleanly");
            CHECK(disk_dev.power_cut_triggered, "Cut D torn superblock fault must fire");
            if (verify_remount_baseline("Cut D (torn superblock 192/512B)") != 0) return 1;
        }

        /* Post-recovery commit succeeds cleanly and persists across remount */
        {
            eshkol_os_virtq_t vq_post;
            eshkol_os_blk_dev_t blk_post;
            if (make_blk(vq_post, blk_post) != 0) return 1;
            eshkol_os_cas_vfs_t vfs_post;
            CHECK(eshkol_os_cas_vfs_mount(&vfs_post, &blk_post) == ESHKOL_DRV_OK, "mount post-recovery");
            const std::string rec_msg = "post-recovery-atomic-commit-verified";
            uint8_t rec_sha[ESHKOL_OS_SHA256_BYTES];
            CHECK(eshkol_os_cas_vfs_put_file(&vfs_post, "recovered.esk", 0u,
                                             rec_msg.data(), rec_msg.size(), rec_sha) == ESHKOL_DRV_OK,
                  "post-recovery commit failed");

            eshkol_os_cas_vfs_t vfs_remount;
            CHECK(eshkol_os_cas_vfs_mount(&vfs_remount, &blk_post) == ESHKOL_DRV_OK,
                  "remount after post-recovery commit");
            size_t rlen = 0u;
            uint8_t rsha[ESHKOL_OS_SHA256_BYTES];
            CHECK(eshkol_os_cas_vfs_read_file(&vfs_remount, "recovered.esk", 0u,
                                              rd_buf.data(), rd_buf.size(), &rlen, rsha) == ESHKOL_DRV_OK &&
                      rlen == rec_msg.size() &&
                      std::memcmp(rd_buf.data(), rec_msg.data(), rec_msg.size()) == 0 &&
                      std::memcmp(rsha, rec_sha, ESHKOL_OS_SHA256_BYTES) == 0,
                  "post-recovery file readback mismatch");

            /* Verify silent storage bit-flip detection via SHA-256 verification */
            uint32_t cfg_sec = 0u;
            for (uint32_t i = 0u; i < ESHKOL_OS_VFS_MAX_ENTRIES; ++i) {
                if (vfs_remount.entries[i].active &&
                    std::strcmp(vfs_remount.entries[i].name, "config.cfg") == 0) {
                    cfg_sec = vfs_remount.entries[i].first_sector;
                    break;
                }
            }
            CHECK(cfg_sec > 0u, "locate config.cfg data sector");
            disk_dev.disk[static_cast<size_t>(cfg_sec) * ESHKOL_OS_VFS_SECTOR_SIZE + 17u] ^= 0x01u;
            CHECK(eshkol_os_cas_vfs_read_file(&vfs_remount, "config.cfg", etc_dir_id,
                                              rd_buf.data(), rd_buf.size(), &rlen, nullptr) ==
                      ESHKOL_DRV_ERR_VFS_CORRUPT,
                  "on-disk bit-flip must fail closed with VFS_CORRUPT");
        }

        CHECK(eshkol_os_dma_buffer_release(&rt, &dma) == ESHKOL_DRV_OK, "release vfs dma");
        CHECK(eshkol_os_irq_lease_release(&rt, &irq) == ESHKOL_DRV_OK, "release vfs irq");
        CHECK(eshkol_os_device_region_release(&rt, &reg) == ESHKOL_DRV_OK, "release vfs reg");
        CHECK(g_kernel.live_mapped_pages == 0u && rt.live_dma_pages == 0u &&
                  g_kernel.total_mapped == g_kernel.total_unmapped,
              "exact 1:1 DMA page map/unmap ledger balance across entire K4 test");
        std::cout << "PASS Guard #4 (Atomic CAS VFS Power-Loss): 4/4 cut points (mid-payload, mid-index, pre-SB, torn-SB 192/512B) recovered with 0 partial blocks + SHA-256 bit-flip detection" << std::endl;
    }

    /* ── GUARD #5: Zero-Symbol Freestanding Audit via llvm-nm -u ── */
    if (audit_zero_external_symbols(llvm_nm, work_dir, "host libeshkol-os-drivers-vfs.a", drivers_vfs_lib) != 0) {
        return 1;
    }

    const fs::path src_c = repo_root / "lib" / "core" / "os_drivers_vfs.c";
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
        const fs::path obj = work_dir / ("os_drivers_vfs_" + t.label + ".o");
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
            return fail("freestanding cross-compile of os_drivers_vfs.c failed for " + t.label + ":\n" + cc.output);
        }
        ++g_checks;
        if (audit_zero_external_symbols(llvm_nm, work_dir, "bare-metal " + t.label + " ELF", obj) != 0) {
            return 1;
        }
    }

    fs::remove_all(work_dir, ec);
    std::cout << "PASS: os_k4_drivers_vfs_test (" << g_checks << " checks)" << std::endl;
    return 0;
}
