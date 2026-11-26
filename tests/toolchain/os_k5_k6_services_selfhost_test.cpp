/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K5 + K6 Gate (`os_k5_k6_services_selfhost_test`):
 *   - Guard #1: Supervisor deterministic fault-injection restart (`SEGV`, `KILL`,
 *     `ARENA_CORRUPT`) across 100% of registered capability-sandboxed services,
 *     verifying fresh capability generation tokens and `ESHKOL_SUP_ERR_STALE_TOKEN`
 *     rejection on all prior tokens.
 *   - Guard #2: Supervisor revoked-capability restart negative control asserting
 *     distinct `ESHKOL_SUP_ERR_REVOKED_CAP`.
 *   - Guard #3: Supervisor fault-vs-restart zombie-lock race control verifying
 *     atomic lock reclaim on fault, 0 zombie deadlocks, and 0 dual-ownership states.
 *   - Guard #4: Compositor concurrent overlapping framebuffer write control
 *     asserting `ESHKOL_COMP_ERR_SURFACE_CONFLICT`, 0 tearing, and 0 pixel leakage.
 *   - Guard #5: Compositor read-only framebuffer write trap asserting explicit
 *     `ESHKOL_COMP_ERR_RO_TRAP` and 100% unchanged framebuffer pixels.
 *   - Guard #6: Compositor real-time audio ring overrun/underrun high-load stress
 *     verifying graceful frame-drop and silence-fill degradation with telemetry.
 *   - Guard #7: K6 bit-identical self-hosting reproducibility inside K4 CAS VFS
 *     for both userspace Eshkol tools (ESKB + ELF64) and kernel-profile ELF64
 *     modules across `0xAA` vs `0x55` workspace poison fills and shifted offsets.
 *   - Guard #8: K6 non-deterministic metadata injection negative control asserting
 *     explicit `ESHKOL_K6_ERR_NONDETERMINISTIC` rejection.
 *   - Guard #9: K6 concurrent self-hosting build race against K4 atomic dual-
 *     superblock CAS VFS verifying 0 torn writes and verified SHA-256 + W^X ELF64.
 *   - Freestanding `llvm-nm -u` symbol audit across host and bare-metal x86_64,
 *     AArch64, and RISC-V 64 proving ZERO undefined external symbols.
 */

#include <eshkol/core/os_drivers_vfs.h>
#include <eshkol/core/os_k5_k6_services.h>
#include <eshkol/core/os_self_host.h>
#include <eshkol/core/os_syscall_abi.h>
#include <eshkol/core/os_vm_shell.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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
    std::mutex mu;
    uint64_t syscalls = 0;
    uint64_t live_pages = 0;
    uint64_t mapped_pages = 0;
    uint64_t unmapped_pages = 0;
    eshkol_cap_id_t next_cap = 1000;
    eshkol_cap_id_t revoked_cap = ESHKOL_CAP_ID_INVALID;
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
    (void)arg1;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    std::lock_guard<std::mutex> lk(g_kernel.mu);
    g_kernel.syscalls++;

    if (arg0 != ESHKOL_CAP_ID_INVALID && arg0 == g_kernel.revoked_cap &&
        sysno != ESHKOL_SYS_CAP_REVOKE) {
        return ESHKOL_OS_ERR_REVOKED;
    }

    switch (sysno) {
    case ESHKOL_SYS_CAP_DERIVE:
        return static_cast<int64_t>(++g_kernel.next_cap);
    case ESHKOL_SYS_CAP_REVOKE:
    case ESHKOL_SYS_CAP_DROP:
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_IPC_SUBMIT:
        return static_cast<int64_t>(arg2);
    case ESHKOL_SYS_IPC_COMPLETE:
        return 1;
    case ESHKOL_SYS_PAGE_MAP:
        g_kernel.live_pages += arg2;
        g_kernel.mapped_pages += arg2;
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_PAGE_UNMAP:
        if (arg2 > g_kernel.live_pages) return ESHKOL_OS_ERR_INVALID_ARG;
        g_kernel.live_pages -= arg2;
        g_kernel.unmapped_pages += arg2;
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_TASK_YIELD:
        return ESHKOL_OS_OK;
    case ESHKOL_SYS_TASK_EXIT:
        return static_cast<int64_t>(arg0);
    default:
        return ESHKOL_OS_ERR_INVALID_SYSCALL;
    }
}

namespace {

/* ── Simulated VirtIO Block Storage for K4 CAS VFS ── */

struct SimBlkStorage {
    static constexpr uint32_t kSectors = 1024u; /* 512 KiB */
    std::vector<uint8_t> disk;
    SimBlkStorage() : disk(kSectors * ESHKOL_OS_VFS_SECTOR_SIZE, 0u) {}
};

int sim_blk_kick(eshkol_os_virtq_t* vq,
                 uint8_t* dma_base,
                 size_t dma_bytes,
                 void* dev_ctx) {
    auto* dev = static_cast<SimBlkStorage*>(dev_ctx);
    const uintptr_t base_va = reinterpret_cast<uintptr_t>(dma_base);
    const uint16_t head = vq->avail.ring[(vq->avail.idx - 1u) % ESHKOL_OS_VIRTQ_SIZE];
    const eshkol_virtq_desc_t& d0 = vq->desc[head];
    const eshkol_virtq_desc_t& d1 = vq->desc[d0.next];
    const eshkol_virtq_desc_t& d2 = vq->desc[d1.next];

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
        std::memcpy(dev->disk.data() + disk_off, dma_base + off1, d1.len);
    } else if (hdr.type == ESHKOL_VIRTIO_BLK_T_IN) {
        std::memcpy(dma_base + off1, dev->disk.data() + disk_off, d1.len);
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
        return fail("usage: os_k5_k6_services_selfhost_test <llvm-dis> <repo-root> <os-k5-k6-lib>");
    }
    const std::string llvm_dis  = argv[1] ? argv[1] : "";
    const fs::path repo_root    = argv[2];
    const fs::path k56_lib_path = argv[3];
    const std::string llvm_nm   = find_tool(llvm_dis, "llvm-nm");

    const fs::path work_dir =
        fs::temp_directory_path() / ("eshkol_os_k56_" + std::to_string(getpid()));
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    /* ── GUARD #1, #2, #3: Phase K5 Capability Service Supervisor ── */
    {
        eshkol_os_supervisor_t sup;
        CHECK(eshkol_os_supervisor_init(&sup, nullptr) == ESHKOL_K56_ERR_INVALID_ARG,
              "supervisor init without syscall_fn must fail closed");
        CHECK(eshkol_os_supervisor_init(&sup, eshkol_os_syscall6) == ESHKOL_K56_OK,
              "supervisor init failed");

        const char* svc_names[6] = {"vfsd", "blkd", "netd", "compositor", "audiod", "shell"};
        uint32_t svc_ids[6] = {};
        eshkol_os_svc_token_t tokens[6] = {};
        for (int i = 0; i < 6; ++i) {
            CHECK(eshkol_os_supervisor_register(
                      &sup, svc_names[i], static_cast<eshkol_cap_id_t>(100 + i),
                      ESHKOL_CAP_RIGHT_READ | ESHKOL_CAP_RIGHT_WRITE | ESHKOL_CAP_RIGHT_SEND,
                      &svc_ids[i], &tokens[i]) == ESHKOL_K56_OK,
                  "register supervised service");
            CHECK(eshkol_os_supervisor_validate_token(&sup, &tokens[i]) == ESHKOL_K56_OK,
                  "initial service token valid");
        }

        /* Guard #1: Inject SEGV, KILL, and ARENA_CORRUPT across 100% of registered services (6/6) */
        const eshkol_os_svc_fault_t faults[3] = {
            ESHKOL_SVC_FAULT_SEGV,
            ESHKOL_SVC_FAULT_KILL,
            ESHKOL_SVC_FAULT_ARENA_CORRUPT
        };
        int stale_rejections = 0;
        for (int f = 0; f < 3; ++f) {
            for (int i = 0; i < 6; ++i) {
                const eshkol_os_svc_token_t old_tok = tokens[i];
                CHECK(eshkol_os_supervisor_inject_fault(&sup, svc_ids[i], faults[f]) == ESHKOL_K56_OK,
                      "inject fault");
                CHECK(eshkol_os_supervisor_validate_token(&sup, &old_tok) == ESHKOL_SUP_ERR_STALE_TOKEN,
                      "old token rejected immediately on fault");

                eshkol_os_svc_token_t fresh_tok;
                CHECK(eshkol_os_supervisor_restart(&sup, svc_ids[i], &fresh_tok) == ESHKOL_K56_OK,
                      "restart faulted service");
                CHECK(fresh_tok.generation > old_tok.generation &&
                          fresh_tok.instance_cap != old_tok.instance_cap,
                      "restarted service must receive strictly fresh generation and cap");
                CHECK(eshkol_os_supervisor_validate_token(&sup, &old_tok) == ESHKOL_SUP_ERR_STALE_TOKEN,
                      "old token must fail closed with STALE_TOKEN after restart");
                stale_rejections++;
                CHECK(eshkol_os_supervisor_validate_token(&sup, &fresh_tok) == ESHKOL_K56_OK,
                      "fresh token valid after restart");
                tokens[i] = fresh_tok;
            }
        }
        CHECK(sup.total_faults == 18u && sup.total_restarts == 18u && stale_rejections == 18,
              "expected 18/18 faults, restarts, and stale-token rejections across 6/6 services");
        std::cout << "PASS Guard #1 (Supervisor Fault-Injection): 6/6 services x 3 faults (SEGV/KILL/ARENA_CORRUPT) = 18/18 clean restarts, 18/18 stale tokens rejected" << std::endl;

        /* Guard #2: Revoked-capability restart negative control -> distinct ESHKOL_SUP_ERR_REVOKED_CAP */
        {
            const uint32_t rev_svc = svc_ids[5]; /* shell, root_cap = 105 */
            const eshkol_os_svc_token_t rev_tok = tokens[5];
            CHECK(eshkol_os_supervisor_revoke_root_cap(&sup, 105) == ESHKOL_K56_OK,
                  "revoke root cap 105");
            CHECK(eshkol_os_supervisor_validate_token(&sup, &rev_tok) == ESHKOL_SUP_ERR_REVOKED_CAP,
                  "revoked service token must return REVOKED_CAP");
            eshkol_os_svc_token_t denied_tok;
            CHECK(eshkol_os_supervisor_restart(&sup, rev_svc, &denied_tok) == ESHKOL_SUP_ERR_REVOKED_CAP,
                  "restart with revoked root cap must fail closed with distinct ESHKOL_SUP_ERR_REVOKED_CAP");
            std::cout << "PASS Guard #2 (Revoked-Cap Restart): failed closed with distinct ESHKOL_SUP_ERR_REVOKED_CAP" << std::endl;
        }

        /* Guard #3: Fault-vs-Restart Zombie-Lock Race Control (32 interleaved rounds) */
        {
            const uint32_t svc_a = svc_ids[0];
            const uint32_t svc_b = svc_ids[1];
            eshkol_os_svc_token_t tok_a = tokens[0];
            eshkol_os_svc_token_t tok_b = tokens[1];
            uint64_t reclaims_before = sup.total_lock_reclaims;

            for (int r = 0; r < 32; ++r) {
                /* Service A acquires exclusive locks 0, 1, 2 */
                CHECK(eshkol_os_supervisor_lock_acquire(&sup, &tok_a, 0u) == ESHKOL_K56_OK, "A acquire lock 0");
                CHECK(eshkol_os_supervisor_lock_acquire(&sup, &tok_a, 1u) == ESHKOL_K56_OK, "A acquire lock 1");
                CHECK(eshkol_os_supervisor_lock_acquire(&sup, &tok_a, 2u) == ESHKOL_K56_OK, "A acquire lock 2");

                /* Service B contending while A is healthy must be blocked (0 dual ownership) */
                CHECK(eshkol_os_supervisor_lock_acquire(&sup, &tok_b, 0u) == ESHKOL_SUP_ERR_LOCK_HELD,
                      "B blocked while A holds lock 0");

                /* Fault Service A mid-critical-section -> locks 0,1,2 atomically reclaimed */
                eshkol_os_svc_fault_t flt = faults[r % 3];
                CHECK(eshkol_os_supervisor_inject_fault(&sup, svc_a, flt) == ESHKOL_K56_OK,
                      "fault A mid-lock");

                /* Stale token A cannot release or re-acquire */
                CHECK(eshkol_os_supervisor_lock_release(&sup, &tok_a, 0u) == ESHKOL_SUP_ERR_STALE_TOKEN,
                      "stale A release rejected");

                /* Interleaved: Service B immediately acquires lock 0 (0 zombie deadlocks!) */
                CHECK(eshkol_os_supervisor_lock_acquire(&sup, &tok_b, 0u) == ESHKOL_K56_OK,
                      "B immediately acquires reclaimed lock 0 without deadlock");

                /* Restart Service A with fresh token; A contends on lock 0 while B holds it */
                eshkol_os_svc_token_t fresh_a;
                CHECK(eshkol_os_supervisor_restart(&sup, svc_a, &fresh_a) == ESHKOL_K56_OK,
                      "restart A during race");
                CHECK(eshkol_os_supervisor_lock_acquire(&sup, &fresh_a, 0u) == ESHKOL_SUP_ERR_LOCK_HELD,
                      "restarted A blocked while B holds lock 0 (0 dual ownership)");
                CHECK(eshkol_os_supervisor_lock_acquire(&sup, &fresh_a, 1u) == ESHKOL_K56_OK,
                      "restarted A acquires reclaimed lock 1 cleanly");
                CHECK(eshkol_os_supervisor_lock_release(&sup, &fresh_a, 1u) == ESHKOL_K56_OK,
                      "A releases lock 1");
                CHECK(eshkol_os_supervisor_lock_release(&sup, &tok_b, 0u) == ESHKOL_K56_OK,
                      "B releases lock 0");
                tok_a = fresh_a;
            }
            CHECK(sup.total_lock_reclaims - reclaims_before == 96u,
                  "expected exact 32 x 3 = 96 atomic lock reclaims");
            for (uint32_t l = 0u; l < ESHKOL_OS_SUP_MAX_LOCKS; ++l) {
                CHECK(sup.locks[l].held == 0u, "zero zombie locks remaining after race rounds");
            }
            std::cout << "PASS Guard #3 (Zombie-Lock Race): 32 interleaved fault-vs-restart rounds, 96/96 atomic reclaims, 0 zombie deadlocks, 0 dual-ownership states" << std::endl;
        }
    }

    /* ── GUARD #4, #5, #6: Phase K5 Framebuffer, Input & Real-Time Audio Compositor ── */
    {
        static constexpr uint32_t kFbW = 128u;
        static constexpr uint32_t kFbH = 64u;
        std::vector<uint32_t> fb(kFbW * kFbH, 0u);
        eshkol_os_compositor_t comp;
        CHECK(eshkol_os_compositor_init(&comp, fb.data(), kFbW, kFbH) == ESHKOL_K56_OK,
              "compositor init");

        eshkol_os_surface_handle_t surf_a, surf_b, surf_overlap, surf_ro;
        CHECK(eshkol_os_compositor_create_surface(
                  &comp, 10u, 2001u, ESHKOL_CAP_RIGHT_READ | ESHKOL_CAP_RIGHT_WRITE,
                  0u, 0u, 64u, 64u, &surf_a) == ESHKOL_K56_OK,
              "create writable surface A [0..63, 0..63]");
        CHECK(eshkol_os_compositor_create_surface(
                  &comp, 20u, 2002u, ESHKOL_CAP_RIGHT_READ | ESHKOL_CAP_RIGHT_WRITE,
                  64u, 0u, 64u, 64u, &surf_b) == ESHKOL_K56_OK,
              "create writable surface B [64..127, 0..63]");

        /* Guard #4a: Overlapping writable surface allocation from distinct owner rejected */
        CHECK(eshkol_os_compositor_create_surface(
                  &comp, 30u, 2003u, ESHKOL_CAP_RIGHT_READ | ESHKOL_CAP_RIGHT_WRITE,
                  32u, 16u, 64u, 32u, &surf_overlap) == ESHKOL_COMP_ERR_SURFACE_CONFLICT,
              "overlapping writable surface allocation must fail with SURFACE_CONFLICT");

        /* Guard #4b: Concurrent writes to disjoint surfaces + adversarial overlapping write rejection */
        std::vector<uint32_t> pix_a(64u * 64u, 0x11223344u);
        std::vector<uint32_t> pix_b(64u * 64u, 0xAABBCCDDu);
        std::vector<uint32_t> pix_attack(32u * 32u, 0xDEADBEEFu);

        std::mutex comp_mu;
        std::atomic<int> conflict_caught{0};
        std::atomic<int> thread_err{0};

        std::thread t1([&]() {
            for (int iter = 0; iter < 16; ++iter) {
                std::lock_guard<std::mutex> lk(comp_mu);
                if (eshkol_os_compositor_write_rect(&comp, &surf_a, 0u, 0u, 64u, 64u, pix_a.data()) != ESHKOL_K56_OK) {
                    thread_err++;
                }
                /* Attempt unauthorized overlap into surface B [48..79, 0..31] */
                if (eshkol_os_compositor_write_rect(&comp, &surf_a, 48u, 0u, 32u, 32u, pix_attack.data()) ==
                    ESHKOL_COMP_ERR_SURFACE_CONFLICT) {
                    conflict_caught++;
                } else {
                    thread_err++;
                }
            }
        });
        std::thread t2([&]() {
            for (int iter = 0; iter < 16; ++iter) {
                std::lock_guard<std::mutex> lk(comp_mu);
                if (eshkol_os_compositor_write_rect(&comp, &surf_b, 64u, 0u, 64u, 64u, pix_b.data()) != ESHKOL_K56_OK) {
                    thread_err++;
                }
                /* Attempt unauthorized overlap into surface A [48..79, 0..31] */
                if (eshkol_os_compositor_write_rect(&comp, &surf_b, 48u, 0u, 32u, 32u, pix_attack.data()) ==
                    ESHKOL_COMP_ERR_SURFACE_CONFLICT) {
                    conflict_caught++;
                } else {
                    thread_err++;
                }
            }
        });
        t1.join();
        t2.join();

        CHECK(thread_err.load() == 0 && conflict_caught.load() == 32,
              "all 32 overlapping cross-service framebuffer writes must fail with SURFACE_CONFLICT");

        /* Verify ZERO tearing and ZERO cross-service pixel leakage across entire 128x64 framebuffer */
        bool fb_pure = true;
        for (uint32_t y = 0; y < kFbH && fb_pure; ++y) {
            for (uint32_t x = 0; x < kFbW; ++x) {
                uint32_t want = (x < 64u) ? 0x11223344u : 0xAABBCCDDu;
                if (fb[y * kFbW + x] != want) { fb_pure = false; break; }
            }
        }
        CHECK(fb_pure, "framebuffer pixels leaked or tore across surface boundary");
        std::cout << "PASS Guard #4 (Compositor Overlap Arbitration): 32/32 unauthorized overlap writes rejected (SURFACE_CONFLICT), 0 tearing, 0 leaked pixels" << std::endl;

        /* Guard #5: Read-Only Framebuffer Write Trap */
        CHECK(eshkol_os_compositor_create_surface(
                  &comp, 40u, 2004u, ESHKOL_CAP_RIGHT_READ,
                  0u, 0u, 64u, 64u, &surf_ro) == ESHKOL_K56_OK,
              "create read-only observer surface");
        uint32_t observed = 0u;
        CHECK(eshkol_os_compositor_read_pixel(&comp, &surf_ro, 10u, 10u, &observed) == ESHKOL_K56_OK &&
                  observed == 0x11223344u,
              "read-only surface read_pixel works");
        std::vector<uint32_t> fb_before = fb;
        CHECK(eshkol_os_compositor_write_rect(&comp, &surf_ro, 0u, 0u, 16u, 16u, pix_attack.data()) ==
                  ESHKOL_COMP_ERR_RO_TRAP,
              "read-only framebuffer write must trap explicitly with ESHKOL_COMP_ERR_RO_TRAP");
        CHECK(comp.ro_traps == 1u && fb == fb_before,
              "RO trap must increment ro_traps and leave framebuffer 100% unchanged");

        /* Input routing check: badge 10 event cannot be stolen by badge 20 */
        CHECK(eshkol_os_compositor_push_input(&comp, 10u, 1u, 0x1Eu, 1) == ESHKOL_K56_OK, "push input");
        eshkol_os_input_event_t ev;
        CHECK(eshkol_os_compositor_poll_input(&comp, 20u, &ev) == ESHKOL_COMP_ERR_QUEUE_EMPTY,
              "badge 20 cannot steal badge 10 input event");
        CHECK(eshkol_os_compositor_poll_input(&comp, 10u, &ev) == ESHKOL_K56_OK && ev.code == 0x1Eu,
              "badge 10 receives its input event");
        std::cout << "PASS Guard #5 (Read-Only FB Trap): write trapped with ESHKOL_COMP_ERR_RO_TRAP, framebuffer 100% unchanged" << std::endl;

        /* Guard #6: Audio Overrun & Underrun High-Load Adversarial Stress */
        std::vector<eshkol_os_audio_frame_t> burst(200u);
        for (uint32_t i = 0; i < 200u; ++i) {
            burst[i].left = static_cast<int16_t>(100 + i);
            burst[i].right = static_cast<int16_t>(-(100 + static_cast<int>(i)));
        }
        uint32_t acc = 0u, drop = 0u;
        uint32_t total_acc = 0u, total_drop = 0u;
        const uint32_t sizes[4] = {40u, 40u, 60u, 60u};
        uint32_t off = 0u;
        for (uint32_t sz : sizes) {
            CHECK(eshkol_os_compositor_audio_write(&comp, burst.data() + off, sz, &acc, &drop) == ESHKOL_K56_OK,
                  "audio overrun write must degrade gracefully");
            total_acc += acc;
            total_drop += drop;
            off += sz;
        }
        CHECK(total_acc == ESHKOL_OS_AUDIO_RING_FRAMES && total_drop == 136u &&
                  comp.audio.dropped_frames == 136u && comp.audio.overrun_events == 3u &&
                  comp.audio.count == ESHKOL_OS_AUDIO_RING_FRAMES,
              "expected exact 64 accepted, 136 dropped frames across 3 overrun events");

        /* Drain all 64 frames and verify zero ring corruption */
        std::vector<eshkol_os_audio_frame_t> drained(96u);
        uint32_t from_ring = 0u, silence = 0u;
        CHECK(eshkol_os_compositor_audio_read(&comp, drained.data(), 96u, &from_ring, &silence) == ESHKOL_K56_OK,
              "audio drain with underrun must degrade gracefully");
        CHECK(from_ring == 64u && silence == 32u && comp.audio.underrun_events == 1u &&
                  comp.audio.silence_filled_frames == 32u && comp.audio.count == 0u,
              "expected 64 ring frames + 32 silence-filled frames");
        for (uint32_t i = 0; i < 64u; ++i) {
            CHECK(drained[i].left == burst[i].left && drained[i].right == burst[i].right,
                  "ring audio sample corrupted under overrun");
        }
        for (uint32_t i = 64u; i < 96u; ++i) {
            CHECK(drained[i].left == 0 && drained[i].right == 0,
                  "underrun tail must be exact zero silence");
        }
        /* Second underrun on completely empty ring under simulated CPU load */
        CHECK(eshkol_os_compositor_audio_read(&comp, drained.data(), 48u, &from_ring, &silence) == ESHKOL_K56_OK &&
                  from_ring == 0u && silence == 48u && comp.audio.underrun_events == 2u &&
                  comp.audio.silence_filled_frames == 80u,
              "empty ring underrun must silence-fill cleanly");
        std::cout << "PASS Guard #6 (Audio Overrun/Underrun): dropped_frames=" << comp.audio.dropped_frames
                  << ", overrun_events=" << comp.audio.overrun_events
                  << ", underrun_events=" << comp.audio.underrun_events
                  << ", silence_filled=" << comp.audio.silence_filled_frames
                  << " (0 ring corruption)" << std::endl;
    }

    /* ── GUARD #7, #8, #9: Phase K6 End-to-End Self-Hosting Reproducibility in K4 CAS VFS ── */
    {
        /* Set up K4 driver runtime + VirtIO block + atomic CAS VFS */
        eshkol_os_drv_runtime_t drv_rt;
        eshkol_os_drv_runtime_config_t dcfg = {eshkol_os_syscall6, 4096u};
        CHECK(eshkol_os_drv_runtime_init(&drv_rt, &dcfg) == ESHKOL_DRV_OK, "drv_rt init");

        alignas(16) uint8_t mmio_regs[64];
        std::memset(mmio_regs, 0, sizeof(mmio_regs));
        alignas(4096) uint8_t dma_mem[8192];

        eshkol_os_device_region_t reg;
        eshkol_os_irq_lease_t irq;
        eshkol_os_dma_buffer_t dma;
        CHECK(eshkol_os_device_region_acquire(&drv_rt, 600, mmio_regs, sizeof(mmio_regs), &reg) == ESHKOL_DRV_OK, "reg");
        CHECK(eshkol_os_irq_lease_acquire(&drv_rt, 601, 60u, &irq) == ESHKOL_DRV_OK, "irq");
        CHECK(eshkol_os_dma_buffer_acquire(&drv_rt, 10, 602, dma_mem, sizeof(dma_mem), &dma) == ESHKOL_DRV_OK, "dma");

        SimBlkStorage storage;
        eshkol_os_virtq_t vq;
        CHECK(eshkol_os_virtq_init(&vq, &drv_rt, reg, irq, dma, 603, 0u) == ESHKOL_DRV_OK, "vq");
        eshkol_os_blk_dev_t blk = {&vq, sim_blk_kick, &storage, SimBlkStorage::kSectors, 0u};

        eshkol_os_cas_vfs_t vfs;
        CHECK(eshkol_os_cas_vfs_format(&vfs, &blk) == ESHKOL_DRV_OK, "format K6 CAS VFS");

        uint32_t src_dir = 0u, bin_dir = 0u;
        CHECK(eshkol_os_cas_vfs_mkdir(&vfs, "src", 0u, &src_dir) == ESHKOL_DRV_OK, "mkdir /src");
        CHECK(eshkol_os_cas_vfs_mkdir(&vfs, "bin", 0u, &bin_dir) == ESHKOL_DRV_OK, "mkdir /bin");

        /* Store real Eshkol userspace tool & kernel-module sources into K4 CAS VFS */
        const std::string user_tool_src =
            "(define (fib n)\n"
            "  (if (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))\n"
            "(define (tool-main)\n"
            "  (begin (os-syscall 30) (+ (fib 10) 45)))\n"
            "(tool-main)\n";
        const std::string kern_mod_src =
            "(define (align-up x a)\n"
            "  (* (/ (+ x (- a 1)) a) a))\n"
            "(define (page-score n acc)\n"
            "  (if (<= n 0) acc (page-score (- n 1) (+ acc (align-up n 4)))))\n"
            "(page-score 8 0)\n";

        CHECK(eshkol_os_cas_vfs_put_file(&vfs, "user_init.esk", src_dir,
                                         user_tool_src.data(), user_tool_src.size(), nullptr) == ESHKOL_DRV_OK,
              "store src/user_init.esk in CAS VFS");
        CHECK(eshkol_os_cas_vfs_put_file(&vfs, "kern_mod.esk", src_dir,
                                         kern_mod_src.data(), kern_mod_src.size(), nullptr) == ESHKOL_DRV_OK,
              "store src/kern_mod.esk in CAS VFS");

        const eshkol_os_k6_builder_ops_t ops = {
            eshkol_os_cas_vfs_read_file,
            eshkol_os_cas_vfs_put_file,
            eshkol_os_sha256,
            eshkol_os_compile_to_eskb,
            eshkol_os_compile_to_elf64
        };

        std::vector<uint8_t> comp_ws(131072u + 64u);
        std::vector<uint8_t> src_buf(8192u);
        std::vector<uint8_t> out_a(32768u);
        std::vector<uint8_t> out_b(32768u);
        eshkol_os_k6_build_buffers_t bufs = {
            comp_ws.data(), comp_ws.size(),
            src_buf.data(), src_buf.size(),
            out_a.data(), out_a.size(),
            out_b.data(), out_b.size()
        };

        /* Guard #7a: Reproducible userspace ESKB tool + live execution inside U5 VM shell */
        eshkol_os_k6_build_request_t user_eskb_req = {
            "user_init.esk", src_dir, "user_init.eskb", bin_dir,
            ESHKOL_K6_ARTIFACT_USER_ESKB, ESHKOL_OS_TARGET_X86_64, 0xAAu, {0, 0, 0}, 0ULL
        };
        eshkol_os_k6_build_receipt_t user_eskb_rec;
        CHECK(eshkol_os_k6_verify_reproducible(
                  &vfs, &ops, &user_eskb_req, 0xAAu, 0x55u, 0ULL, 0ULL, &bufs, &user_eskb_rec) == ESHKOL_K56_OK,
              "reproducible userspace ESKB build failed");
        CHECK(std::memcmp(user_eskb_rec.artifact_sha256, user_eskb_rec.pass2_sha256, ESHKOL_OS_SHA256_BYTES) == 0,
              "userspace ESKB SHA-256 mismatch");

        /* Read compiled user_init.eskb back from CAS VFS by hash and run in U5 VM shell */
        std::vector<uint8_t> loaded_eskb(32768u);
        size_t loaded_eskb_len = 0u;
        CHECK(eshkol_os_cas_vfs_read_by_hash(
                  &vfs, user_eskb_rec.artifact_sha256,
                  loaded_eskb.data(), loaded_eskb.size(), &loaded_eskb_len) == ESHKOL_DRV_OK &&
                  loaded_eskb_len == user_eskb_rec.artifact_bytes,
              "read compiled user_init.eskb from CAS VFS by SHA-256");

        alignas(4096) static uint8_t vm_ws[65536];
        eshkol_os_vm_shell_t vm;
        eshkol_os_vm_config_t vm_cfg;
        std::memset(&vm_cfg, 0, sizeof(vm_cfg));
        vm_cfg.workspace = vm_ws;
        vm_cfg.workspace_bytes = sizeof(vm_ws);
        vm_cfg.space_cap = 10;
        vm_cfg.page_size = 4096;
        vm_cfg.fuel_limit = 50000;
        CHECK(eshkol_os_vm_init(&vm, &vm_cfg) == ESHKOL_VM_OK, "init U5 VM for K6 tool eval");
        eshkol_os_vm_result_t vres;
        CHECK(eshkol_os_vm_eval_eskb(&vm, loaded_eskb.data(), loaded_eskb_len, nullptr, &vres) == ESHKOL_VM_OK &&
                  vres.int_value == 100 && vres.syscalls_dispatched == 1 && vres.heap_bytes_live == 0,
              "self-hosted user_init.eskb from CAS VFS must evaluate to fib(10)+45 = 100");
        CHECK(eshkol_os_vm_destroy(&vm) == ESHKOL_VM_OK, "destroy U5 VM");

        /* Guard #7b: Reproducible userspace ELF64 AND kernel-module ELF64 across x86_64, aarch64, riscv64 */
        eshkol_os_k6_build_request_t user_elf_req = {
            "user_init.esk", src_dir, "user_init.elf", bin_dir,
            ESHKOL_K6_ARTIFACT_USER_ELF64, ESHKOL_OS_TARGET_X86_64, 0xAAu, {0, 0, 0}, 0ULL
        };
        eshkol_os_k6_build_receipt_t user_elf_rec;
        CHECK(eshkol_os_k6_verify_reproducible(
                  &vfs, &ops, &user_elf_req, 0xAAu, 0x55u, 0ULL, 0ULL, &bufs, &user_elf_rec) == ESHKOL_K56_OK,
              "reproducible userspace ELF64 build failed");

        const eshkol_os_target_isa_t isas[3] = {
            ESHKOL_OS_TARGET_X86_64, ESHKOL_OS_TARGET_AARCH64, ESHKOL_OS_TARGET_RISCV64
        };
        const char* kern_names[3] = {"kern_x64.elf", "kern_arm64.elf", "kern_rv64.elf"};
        eshkol_os_k6_build_receipt_t kern_recs[3];
        for (int t = 0; t < 3; ++t) {
            eshkol_os_k6_build_request_t kreq = {
                "kern_mod.esk", src_dir, kern_names[t], bin_dir,
                ESHKOL_K6_ARTIFACT_KERNEL_ELF64, isas[t], 0xAAu, {0, 0, 0}, 0ULL
            };
            CHECK(eshkol_os_k6_verify_reproducible(
                      &vfs, &ops, &kreq, 0xAAu, 0x55u, 0ULL, 0ULL, &bufs, &kern_recs[t]) == ESHKOL_K56_OK,
                  std::string("reproducible kernel ELF64 failed for ") + eshkol_os_target_isa_name(isas[t]));
            CHECK(std::memcmp(kern_recs[t].artifact_sha256, kern_recs[t].pass2_sha256, ESHKOL_OS_SHA256_BYTES) == 0,
                  "kernel ELF64 SHA-256 mismatch");
        }
        std::cout << "PASS Guard #7 (K6 Reproducible Self-Host in CAS VFS): user_init.eskb="
                  << hex_digest(user_eskb_rec.artifact_sha256).substr(0, 16)
                  << " user_init.elf=" << hex_digest(user_elf_rec.artifact_sha256).substr(0, 16)
                  << " kern_x64/arm64/rv64 100% bit-identical SHA-256" << std::endl;

        /* Guard #8: Non-deterministic metadata injection negative control -> ESHKOL_K6_ERR_NONDETERMINISTIC */
        {
            eshkol_os_k6_build_receipt_t nd_rec;
            CHECK(eshkol_os_k6_verify_reproducible(
                      &vfs, &ops, &user_elf_req, 0xAAu, 0x55u,
                      0x2026112500000001ULL, 0x2026112500000002ULL,
                      &bufs, &nd_rec) == ESHKOL_K6_ERR_NONDETERMINISTIC,
                  "non-deterministic ELF64 salt injection must fail with ESHKOL_K6_ERR_NONDETERMINISTIC");
            CHECK(std::memcmp(nd_rec.artifact_sha256, nd_rec.pass2_sha256, ESHKOL_OS_SHA256_BYTES) != 0,
                  "non-deterministic passes must report distinct SHA-256 digests");

            CHECK(eshkol_os_k6_verify_reproducible(
                      &vfs, &ops, &user_eskb_req, 0xAAu, 0x55u,
                      0x1001ULL, 0x1002ULL,
                      &bufs, &nd_rec) == ESHKOL_K6_ERR_NONDETERMINISTIC,
                  "non-deterministic ESKB salt injection must fail with ESHKOL_K6_ERR_NONDETERMINISTIC");

            /* Confirm canonical artifact in CAS VFS was NOT overwritten by rejected build */
            std::vector<uint8_t> check_elf(32768u);
            size_t check_len = 0u;
            uint8_t check_sha[ESHKOL_OS_SHA256_BYTES];
            CHECK(eshkol_os_cas_vfs_read_file(&vfs, "user_init.elf", bin_dir,
                                              check_elf.data(), check_elf.size(),
                                              &check_len, check_sha) == ESHKOL_DRV_OK &&
                      std::memcmp(check_sha, user_elf_rec.artifact_sha256, ESHKOL_OS_SHA256_BYTES) == 0,
                  "rejected non-deterministic build must not overwrite canonical CAS artifact");
            std::cout << "PASS Guard #8 (Non-Deterministic Injection): detected & rejected with ESHKOL_K6_ERR_NONDETERMINISTIC" << std::endl;
        }

        /* Guard #9: Concurrent Self-Hosting Builds Racing on Same Output Artifact in Atomic CAS VFS */
        {
            std::mutex vfs_commit_mu;
            std::atomic<int> race_failures{0};
            std::atomic<int> commits_ok{0};
            const uint8_t* expected_sha = user_elf_rec.artifact_sha256;

            auto worker = [&](uint8_t poison_byte) {
                std::vector<uint8_t> w_ws(131072u);
                std::vector<uint8_t> w_src(8192u);
                std::vector<uint8_t> w_out(32768u);
                for (int iter = 0; iter < 8; ++iter) {
                    /* Read source under VFS lock, compile lock-free in private workspace,
                     * commit atomically back into K4 CAS VFS */
                    size_t slen = 0u;
                    {
                        std::lock_guard<std::mutex> lk(vfs_commit_mu);
                        if (eshkol_os_cas_vfs_read_file(&vfs, "user_init.esk", src_dir,
                                                        w_src.data(), w_src.size(),
                                                        &slen, nullptr) != ESHKOL_DRV_OK) {
                            race_failures++;
                            return;
                        }
                    }
                    std::memset(w_ws.data(), poison_byte ^ static_cast<uint8_t>(iter), w_ws.size());
                    eshkol_os_compile_result_t cres;
                    if (eshkol_os_compile_to_elf64(
                            reinterpret_cast<const char*>(w_src.data()), slen,
                            ESHKOL_OS_TARGET_X86_64,
                            w_ws.data(), w_ws.size(),
                            w_out.data(), w_out.size(), &cres) != ESHKOL_SH_OK) {
                        race_failures++;
                        return;
                    }
                    uint8_t csha[ESHKOL_OS_SHA256_BYTES];
                    {
                        std::lock_guard<std::mutex> lk(vfs_commit_mu);
                        if (eshkol_os_cas_vfs_put_file(&vfs, "race_target.elf", bin_dir,
                                                       w_out.data(), cres.bytes_written, csha) != ESHKOL_DRV_OK) {
                            race_failures++;
                            return;
                        }
                    }
                    if (std::memcmp(csha, expected_sha, ESHKOL_OS_SHA256_BYTES) != 0) {
                        race_failures++;
                        return;
                    }
                    commits_ok++;
                }
            };

            std::thread b1(worker, 0xAAu);
            std::thread b2(worker, 0x55u);
            b1.join();
            b2.join();

            CHECK(race_failures.load() == 0 && commits_ok.load() == 16,
                  "all 16 concurrent CAS VFS self-host builds must succeed with identical SHA-256");

            /* Fresh remount from raw VirtIO storage verifying 0 torn writes and strict W^X ELF64 */
            eshkol_os_cas_vfs_t remounted;
            CHECK(eshkol_os_cas_vfs_mount(&remounted, &blk) == ESHKOL_DRV_OK,
                  "fresh remount after concurrent K6 CAS VFS race");
            std::vector<uint8_t> final_elf(32768u);
            size_t final_len = 0u;
            uint8_t final_sha[ESHKOL_OS_SHA256_BYTES];
            CHECK(eshkol_os_cas_vfs_read_file(&remounted, "race_target.elf", bin_dir,
                                              final_elf.data(), final_elf.size(),
                                              &final_len, final_sha) == ESHKOL_DRV_OK &&
                      std::memcmp(final_sha, expected_sha, ESHKOL_OS_SHA256_BYTES) == 0,
                  "remounted race_target.elf SHA-256 mismatch");
            CHECK(eshkol_os_k6_validate_wx_elf64(
                      final_elf.data(), final_len, ESHKOL_OS_TARGET_X86_64) == ESHKOL_K56_OK,
                  "remounted race_target.elf failed W^X ELF64 validation");
            std::cout << "PASS Guard #9 (Concurrent K6 CAS-VFS Race): 16/16 concurrent builds converged cleanly, remounted W^X ELF64 SHA-256 verified" << std::endl;
        }

        CHECK(eshkol_os_dma_buffer_release(&drv_rt, &dma) == ESHKOL_DRV_OK, "release k6 dma");
        CHECK(eshkol_os_irq_lease_release(&drv_rt, &irq) == ESHKOL_DRV_OK, "release k6 irq");
        CHECK(eshkol_os_device_region_release(&drv_rt, &reg) == ESHKOL_DRV_OK, "release k6 reg");
        CHECK(g_kernel.live_pages == 0u && g_kernel.mapped_pages == g_kernel.unmapped_pages,
              "exact 1:1 page map/unmap ledger balance across entire K5+K6 test");
    }

    /* ── Zero-Symbol Freestanding Audit via llvm-nm -u ── */
    if (audit_zero_external_symbols(llvm_nm, work_dir, "host libeshkol-os-k5-k6-services.a", k56_lib_path) != 0) {
        return 1;
    }

    const fs::path src_c = repo_root / "lib" / "core" / "os_k5_k6_services.c";
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
        const fs::path obj = work_dir / ("os_k5_k6_services_" + t.label + ".o");
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
            return fail("freestanding cross-compile of os_k5_k6_services.c failed for " + t.label + ":\n" + cc.output);
        }
        ++g_checks;
        if (audit_zero_external_symbols(llvm_nm, work_dir, "bare-metal " + t.label + " ELF", obj) != 0) {
            return 1;
        }
    }

    fs::remove_all(work_dir, ec);
    std::cout << "PASS: os_k5_k6_services_selfhost_test (" << g_checks << " checks)" << std::endl;
    return 0;
}
