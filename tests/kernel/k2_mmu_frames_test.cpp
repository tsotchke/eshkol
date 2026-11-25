/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K2 Verification & Adversarial Rigor Suite:
 *   1. W^X & Privilege-Bit Bypass Negative Controls + TLB Shootdown
 *      across all 3 MMU ISAs (RV64 Sv39/Sv48, AArch64 4K, x86_64 4-Level).
 *   2. Kernel Aliasing & Conflicting-Permission Alias Negative Controls.
 *   3. 100,000-Op Frame Conservation + Independent Shadow Bitmap Double-Free
 *      & Stale-Handle Guard + 10,000 Map/Unmap Cycle Gate across all 3 ISAs.
 *   4. MCU MPU Overlap & Reserved Vector-Table / Stack-Guard Exclusion Controls.
 *   5. Runtime Zero-Alloc Verification via global operator new instrumentation.
 */

#include <eshkol/kernel/k2_mmu_frames.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

std::atomic<uint64_t> g_heap_alloc_count{0};
std::atomic<bool> g_track_allocs{false};

int g_checks_passed = 0;
int g_checks_failed = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (cond) {
        g_checks_passed += 1;
    } else {
        g_checks_failed += 1;
        std::fprintf(stderr, "FAIL [line %d]: %s\n", line, expr);
    }
}

#define CHECK(expr) check_impl((expr), #expr, __LINE__)

/* Deterministic xorshift64* PRNG for repeatable 100,000-op stress tests. */
uint64_t next_rand(uint64_t& state) {
    uint64_t x = state;
    x ^= x >> 12u;
    x ^= x << 25u;
    x ^= x >> 27u;
    state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Raw hardware W=1,X=1 PTE builder per ISA (for adversarial raw-injection negative controls). */
uint64_t make_raw_wx_pte(k2_mmu_mode_t mode, uint64_t phys_addr) {
    switch (mode) {
        case K2_MMU_RV64_SV39:
        case K2_MMU_RV64_SV48:
            return ((phys_addr >> 12u) << 10u) | K2_RV64_PTE_V | K2_RV64_PTE_R |
                   K2_RV64_PTE_W | K2_RV64_PTE_X | K2_RV64_PTE_U | K2_RV64_PTE_A | K2_RV64_PTE_D;
        case K2_MMU_AARCH64_4K:
            /* AP[2]=0 (RW), AP[1]=1 (User), UXN=0 (User Exec), PXN=1 */
            return (phys_addr & K2_AARCH64_PTE_ADDR_MASK) | K2_AARCH64_PTE_VALID |
                   K2_AARCH64_PTE_AP_USER | K2_AARCH64_PTE_AF | K2_AARCH64_PTE_NG |
                   K2_AARCH64_PTE_PXN;
        case K2_MMU_X86_64_4L:
            /* Present + RW + US + NX=0 (Executable) */
            return (phys_addr & K2_X86_64_PTE_ADDR_MASK) | K2_X86_64_PTE_P |
                   K2_X86_64_PTE_RW | K2_X86_64_PTE_US | K2_X86_64_PTE_A | K2_X86_64_PTE_D;
        default:
            return 0ULL;
    }
}

/* Raw hardware privilege-violation PTE per ISA (kernel-only or PXN=0 user page in user space). */
uint64_t make_raw_priv_violation_pte(k2_mmu_mode_t mode, uint64_t phys_addr) {
    switch (mode) {
        case K2_MMU_RV64_SV39:
        case K2_MMU_RV64_SV48:
            /* Supervisor-only (U=0) PTE injected into user space */
            return ((phys_addr >> 12u) << 10u) | K2_RV64_PTE_V | K2_RV64_PTE_R | K2_RV64_PTE_A;
        case K2_MMU_AARCH64_4K:
            /* User page with PXN=0 (privileged-executable user page) */
            return (phys_addr & K2_AARCH64_PTE_ADDR_MASK) | K2_AARCH64_PTE_VALID |
                   K2_AARCH64_PTE_AP_USER | K2_AARCH64_PTE_AP_RO | K2_AARCH64_PTE_AF |
                   K2_AARCH64_PTE_UXN;
        case K2_MMU_X86_64_4L:
            /* Supervisor-only (US=0) PTE injected into user space */
            return (phys_addr & K2_X86_64_PTE_ADDR_MASK) | K2_X86_64_PTE_P |
                   K2_X86_64_PTE_NX | K2_X86_64_PTE_A;
        default:
            return 0ULL;
    }
}

/* Guard 1: W^X & Privilege-Bit Bypass Negative Controls + TLB Shootdown across all 3 ISAs. */
void test_wx_priv_and_tlb_shootdown(uint64_t& total_shootdowns_out, uint64_t& shootdown_ns_out) {
    static k2_frame_pool_t pool;
    static k2_aspace_t aspace;

    static const k2_mmu_mode_t k_modes[4] = {
        K2_MMU_RV64_SV39,
        K2_MMU_RV64_SV48,
        K2_MMU_AARCH64_4K,
        K2_MMU_X86_64_4L
    };

    total_shootdowns_out = 0ULL;
    auto t0 = std::chrono::steady_clock::now();

    for (size_t m = 0u; m < 4u; ++m) {
        const k2_mmu_mode_t mode = k_modes[m];
        CHECK(k2_frame_pool_init(&pool, 0x80000000ULL, 64u, 8u) == K2_OK);
        CHECK(k2_aspace_init(&aspace, mode, static_cast<uint32_t>(m + 1u), 1u) == K2_OK);

        k2_page_frame_t frame{};
        CHECK(k2_frame_alloc_user(&pool, &frame) == K2_OK);

        const uint64_t va = 0x0000100000ULL;

        /* Negative 1A: Map with W=1, X=1 simultaneously -> K2_ERR_WX_VIOLATION */
        CHECK(k2_aspace_map(&aspace, &pool, va, &frame,
                            K2_PERM_READ | K2_PERM_WRITE | K2_PERM_EXEC | K2_PERM_USER) ==
              K2_ERR_WX_VIOLATION);

        /* Negative 1B: Map with U=1, K=1 simultaneously -> K2_ERR_PRIV_VIOLATION */
        CHECK(k2_aspace_map(&aspace, &pool, va, &frame,
                            K2_PERM_READ | K2_PERM_USER | K2_PERM_KERNEL_ONLY) ==
              K2_ERR_PRIV_VIOLATION);

        /* Negative 1C: Map kernel-only in user address space -> K2_ERR_PRIV_VIOLATION */
        CHECK(k2_aspace_map(&aspace, &pool, va, &frame,
                            K2_PERM_READ | K2_PERM_KERNEL_ONLY) ==
              K2_ERR_PRIV_VIOLATION);

        /* Negative 1D: Raw PTE hardware injection with W=1,X=1 -> rejected fail-closed */
        const uint64_t bad_wx_pte = make_raw_wx_pte(mode, frame.phys_addr);
        CHECK(k2_validate_raw_pte(mode, bad_wx_pte, 1u) == K2_ERR_WX_VIOLATION);
        CHECK(k2_aspace_insert_raw_pte(&aspace, va, bad_wx_pte) == K2_ERR_WX_VIOLATION);

        /* Negative 1E: Raw PTE hardware injection with privilege violation -> rejected fail-closed */
        const uint64_t bad_priv_pte = make_raw_priv_violation_pte(mode, frame.phys_addr);
        CHECK(k2_validate_raw_pte(mode, bad_priv_pte, 1u) == K2_ERR_PRIV_VIOLATION);
        CHECK(k2_aspace_insert_raw_pte(&aspace, va, bad_priv_pte) == K2_ERR_PRIV_VIOLATION);

        /* Positive map RW -> warm TLB -> verify hit -> flip RW->RO -> verify immediate TLB shootdown */
        CHECK(k2_aspace_map(&aspace, &pool, va, &frame, K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) == K2_OK);

        uint64_t out_phys = 0ULL;
        CHECK(k2_aspace_translate(&aspace, va + 0x40ULL, K2_ACCESS_WRITE | K2_ACCESS_USER, &out_phys) == K2_OK);
        CHECK(out_phys == frame.phys_addr + 0x40ULL);
        CHECK(aspace.tlb_misses == 1ULL);

        /* Second translation hits the warmed TLB entry */
        CHECK(k2_aspace_translate(&aspace, va + 0x80ULL, K2_ACCESS_WRITE | K2_ACCESS_USER, &out_phys) == K2_OK);
        CHECK(aspace.tlb_hits == 1ULL);

        /* Flip permissions RW -> RO: TLB shootdown must invalidate stale RW entry immediately */
        const uint64_t prev_shootdowns = aspace.tlb_shootdowns;
        const uint64_t prev_invalidated = aspace.tlb_invalidated_entries;
        CHECK(k2_aspace_remap_perms(&aspace, &pool, va, K2_PERM_READ | K2_PERM_USER) == K2_OK);
        CHECK(aspace.tlb_shootdowns == prev_shootdowns + 1ULL);
        CHECK(aspace.tlb_invalidated_entries == prev_invalidated + 1ULL);

        /* Stale write MUST be denied after flip; read still succeeds */
        CHECK(k2_aspace_translate(&aspace, va, K2_ACCESS_WRITE | K2_ACCESS_USER, &out_phys) == K2_ERR_ACCESS_DENIED);
        CHECK(k2_aspace_translate(&aspace, va, K2_ACCESS_READ | K2_ACCESS_USER, &out_phys) == K2_OK);

        /* Flip RO -> RX (legal), warm TLB, then unmap -> exec MUST fault immediately */
        CHECK(k2_aspace_remap_perms(&aspace, &pool, va, K2_PERM_READ | K2_PERM_EXEC | K2_PERM_USER) == K2_OK);
        CHECK(k2_aspace_translate(&aspace, va, K2_ACCESS_EXEC | K2_ACCESS_USER, &out_phys) == K2_OK);
        CHECK(k2_aspace_unmap(&aspace, &pool, va) == K2_OK);
        CHECK(k2_aspace_translate(&aspace, va, K2_ACCESS_EXEC | K2_ACCESS_USER, &out_phys) == K2_ERR_VA_NOT_MAPPED);

        /* Adversarial simulated memory-corrupt PTE bitflip (W=1,X=1 directly in PT slot):
         * translate MUST still reject fail-closed on walk! */
        CHECK(k2_aspace_map(&aspace, &pool, va, &frame, K2_PERM_READ | K2_PERM_USER) == K2_OK);
        aspace.entries[0].raw_pte = bad_wx_pte;
        k2_aspace_tlb_shootdown_va(&aspace, va);
        CHECK(k2_aspace_translate(&aspace, va, K2_ACCESS_READ | K2_ACCESS_USER, &out_phys) == K2_ERR_WX_VIOLATION);

        CHECK(k2_aspace_unmap(&aspace, &pool, va) == K2_OK);
        CHECK(k2_frame_free_user(&pool, &frame) == K2_OK);
        CHECK(k2_frame_pool_invariants_hold(&pool) == 1);
        total_shootdowns_out += aspace.tlb_shootdowns;
    }

    auto t1 = std::chrono::steady_clock::now();
    shootdown_ns_out = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

/* Guard 2: Kernel Aliasing & Conflicting-Permission Alias Negative Controls. */
void test_kernel_and_permission_aliasing() {
    static k2_frame_pool_t pool;
    static k2_aspace_t user_as1;
    static k2_aspace_t user_as2;
    static k2_aspace_t kernel_as;

    CHECK(k2_frame_pool_init(&pool, 0x40000000ULL, 128u, 16u) == K2_OK);
    CHECK(k2_aspace_init(&user_as1, K2_MMU_RV64_SV48, 10u, 1u) == K2_OK);
    CHECK(k2_aspace_init(&user_as2, K2_MMU_RV64_SV48, 11u, 1u) == K2_OK);
    CHECK(k2_aspace_init(&kernel_as, K2_MMU_RV64_SV48, 0u, 0u) == K2_OK);

    /* Allocate an additional runtime kernel stack frame (tagged K2_OWNER_KERNEL) */
    k2_page_frame_t kstack_frame{};
    CHECK(k2_frame_alloc_kernel(&pool, &kstack_frame) == K2_OK);
    CHECK(kstack_frame.owner == K2_OWNER_KERNEL);

    /* Negative 2A: Attempt to map EVERY boot kernel-reserved frame (0..15) and kernel stack frame into user space */
    for (uint32_t i = 0u; i < 16u; ++i) {
        k2_page_frame_t kframe{};
        CHECK(k2_frame_get_kernel_handle(&pool, i, &kframe) == K2_OK);
        CHECK(k2_aspace_map(&user_as1, &pool, 0x10000ULL + (static_cast<uint64_t>(i) << 12u),
                            &kframe, K2_PERM_READ | K2_PERM_USER) == K2_ERR_KERNEL_ALIAS);
        CHECK(k2_frame_free_user(&pool, &kframe) == K2_ERR_KERNEL_FRAME_LOCKED);
    }
    CHECK(k2_aspace_map(&user_as1, &pool, 0x50000ULL, &kstack_frame,
                        K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) == K2_ERR_KERNEL_ALIAS);
    CHECK(k2_frame_free_user(&pool, &kstack_frame) == K2_ERR_KERNEL_FRAME_LOCKED);
    CHECK(user_as1.mapped_count == 0u);

    /* Kernel address space MAY map kernel frame with KERNEL_ONLY */
    CHECK(k2_aspace_map(&kernel_as, &pool, 0xFFFF800000010000ULL, &kstack_frame,
                        K2_PERM_READ | K2_PERM_WRITE | K2_PERM_KERNEL_ONLY) == K2_OK);

    /* Negative 2B: Same physical user frame mapped RW at va1 and RX at va2 -> K2_ERR_PERM_ALIAS */
    k2_page_frame_t uframe{};
    CHECK(k2_frame_alloc_user(&pool, &uframe) == K2_OK);
    CHECK(k2_aspace_map(&user_as1, &pool, 0x200000ULL, &uframe,
                        K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) == K2_OK);
    CHECK(k2_aspace_map(&user_as1, &pool, 0x201000ULL, &uframe,
                        K2_PERM_READ | K2_PERM_EXEC | K2_PERM_USER) == K2_ERR_PERM_ALIAS);

    /* Negative 2C: Distinct non-shared user address space attempting to alias same frame -> K2_ERR_PERM_ALIAS */
    CHECK(k2_aspace_map(&user_as2, &pool, 0x200000ULL, &uframe,
                        K2_PERM_READ | K2_PERM_USER) == K2_ERR_PERM_ALIAS);

    /* Negative 2D: Freeing a still-mapped user frame -> K2_ERR_FRAME_STILL_MAPPED */
    CHECK(k2_frame_free_user(&pool, &uframe) == K2_ERR_FRAME_STILL_MAPPED);

    /* Reverse order: map RX first, then attempt RW alias -> K2_ERR_PERM_ALIAS */
    CHECK(k2_aspace_unmap(&user_as1, &pool, 0x200000ULL) == K2_OK);
    CHECK(k2_aspace_map(&user_as1, &pool, 0x300000ULL, &uframe,
                        K2_PERM_READ | K2_PERM_EXEC | K2_PERM_USER) == K2_OK);
    CHECK(k2_aspace_map(&user_as1, &pool, 0x301000ULL, &uframe,
                        K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) == K2_ERR_PERM_ALIAS);

    CHECK(k2_aspace_unmap(&user_as1, &pool, 0x300000ULL) == K2_OK);
    CHECK(k2_frame_free_user(&pool, &uframe) == K2_OK);
    CHECK(k2_aspace_unmap(&kernel_as, &pool, 0xFFFF800000010000ULL) == K2_OK);
    CHECK(k2_frame_pool_invariants_hold(&pool) == 1);
}

struct ConservationStats {
    uint64_t conservation_checks;
    uint64_t double_free_rejects;
    uint64_t stale_handle_rejects;
    uint64_t map_unmap_cycles;
};

/* Guard 3: 100,000-Op Frame Conservation + Independent Shadow Bitmap + 10,000 Map/Unmap Gate. */
void test_100k_conservation_and_10k_map_unmap(ConservationStats& stats_out) {
    static k2_frame_pool_t pool;
    static k2_aspace_t as_rv64;
    static k2_aspace_t as_arm64;
    static k2_aspace_t as_x86;

    const uint32_t kTotalFrames = 512u;
    const uint32_t kKernelReserved = 32u;
    CHECK(k2_frame_pool_init(&pool, 0x100000000ULL, kTotalFrames, kKernelReserved) == K2_OK);
    CHECK(k2_aspace_init(&as_rv64, K2_MMU_RV64_SV39, 1u, 1u) == K2_OK);
    CHECK(k2_aspace_init(&as_arm64, K2_MMU_AARCH64_4K, 2u, 1u) == K2_OK);
    CHECK(k2_aspace_init(&as_x86, K2_MMU_X86_64_4L, 3u, 1u) == K2_OK);

    k2_aspace_t* aspaces[3] = {&as_rv64, &as_arm64, &as_x86};

    /* Independent test-side shadow tracking arrays (zero heap allocation). */
    struct ShadowSlot {
        bool allocated;
        bool mapped;
        uint32_t aspace_idx;
        uint64_t va;
        k2_page_frame_t handle;
        k2_page_frame_t stale_handle;
        bool has_stale;
    };
    static ShadowSlot shadow[K2_MAX_FRAMES];
    std::memset(shadow, 0, sizeof(shadow));

    uint32_t shadow_user_alloc = 0u;
    uint32_t shadow_live_indices[K2_MAX_FRAMES];
    uint32_t shadow_live_count = 0u;
    uint64_t rng = 0xCAFEBABE12345678ULL;
    uint64_t double_free_rejects = 0ULL;
    uint64_t stale_handle_rejects = 0ULL;
    uint64_t conservation_checks = 0ULL;

    for (uint32_t step = 0u; step < 100000u; ++step) {
        uint64_t r = next_rand(rng);
        uint32_t action = static_cast<uint32_t>(r % 10u);

        if (action <= 3u && shadow_live_count < (kTotalFrames - kKernelReserved)) {
            /* Allocate a user frame and optionally map it into one of the 3 ISA address spaces */
            k2_page_frame_t f{};
            k2_status_t st = k2_frame_alloc_user(&pool, &f);
            if (st == K2_OK) {
                if (shadow[f.frame_idx].allocated) {
                    CHECK(false);
                }
                shadow[f.frame_idx].allocated = true;
                shadow[f.frame_idx].mapped = false;
                shadow[f.frame_idx].handle = f;
                shadow_live_indices[shadow_live_count++] = f.frame_idx;
                shadow_user_alloc += 1u;

                /* Map into chosen ISA address space if slot available */
                uint32_t aidx = static_cast<uint32_t>((r >> 8u) % 3u);
                uint64_t va = 0x00400000ULL + (static_cast<uint64_t>(f.frame_idx) << K2_PAGE_SHIFT);
                uint32_t perms = ((r & 16u) != 0u)
                    ? (K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER)
                    : (K2_PERM_READ | K2_PERM_EXEC | K2_PERM_USER);
                if (k2_aspace_map(aspaces[aidx], &pool, va, &f, perms) == K2_OK) {
                    shadow[f.frame_idx].mapped = true;
                    shadow[f.frame_idx].aspace_idx = aidx;
                    shadow[f.frame_idx].va = va;
                }
            }
        } else if (action <= 7u && shadow_live_count > 0u) {
            /* Unmap (if mapped) and free a randomly selected live user frame */
            uint32_t pick = static_cast<uint32_t>((r >> 16u) % shadow_live_count);
            uint32_t fidx = shadow_live_indices[pick];
            ShadowSlot& s = shadow[fidx];
            if (s.mapped) {
                if (k2_aspace_unmap(aspaces[s.aspace_idx], &pool, s.va) != K2_OK) {
                    CHECK(false);
                }
                s.mapped = false;
            }
            if (k2_frame_free_user(&pool, &s.handle) != K2_OK) {
                CHECK(false);
            }
            /* Immediate deliberate double-free negative control */
            if (k2_frame_free_user(&pool, &s.handle) == K2_ERR_DOUBLE_FREE) {
                double_free_rejects += 1ULL;
            } else {
                CHECK(false);
            }
            s.stale_handle = s.handle;
            s.has_stale = true;
            s.allocated = false;
            shadow_live_indices[pick] = shadow_live_indices[--shadow_live_count];
            shadow_user_alloc -= 1u;
        } else if (action == 8u && shadow_live_count > 0u) {
            /* Stale-generation handle negative control on a currently re-allocated frame */
            uint32_t pick = static_cast<uint32_t>((r >> 24u) % shadow_live_count);
            uint32_t fidx = shadow_live_indices[pick];
            ShadowSlot& s = shadow[fidx];
            if (s.has_stale && s.stale_handle.generation != s.handle.generation) {
                k2_status_t st = k2_frame_free_user(&pool, &s.stale_handle);
                if (st == K2_ERR_STALE_HANDLE) {
                    stale_handle_rejects += 1ULL;
                } else {
                    CHECK(false);
                }
            }
        } else if (shadow_live_count > 0u) {
            /* Permission flip + TLB shootdown on a mapped live frame */
            uint32_t pick = static_cast<uint32_t>((r >> 32u) % shadow_live_count);
            uint32_t fidx = shadow_live_indices[pick];
            ShadowSlot& s = shadow[fidx];
            if (s.mapped) {
                uint32_t new_p = ((step & 1u) != 0u)
                    ? (K2_PERM_READ | K2_PERM_USER)
                    : (K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER);
                if (k2_aspace_remap_perms(aspaces[s.aspace_idx], &pool, s.va, new_p) != K2_OK) {
                    CHECK(false);
                }
            }
        }

        /* Assert conservation at EVERY SINGLE ONE of the 100,000 steps! */
        if ((pool.free_frames + pool.allocated_frames) != pool.total_frames ||
            pool.user_allocated_frames != shadow_user_alloc ||
            pool.kernel_reserved_frames != kKernelReserved) {
            CHECK(false);
        }
        conservation_checks += 1ULL;
    }

    CHECK(conservation_checks == 100000ULL);
    CHECK(double_free_rejects > 10000ULL);
    CHECK(stale_handle_rejects > 0ULL);

    /* Unmap and free all remaining live frames -> assert 0 leaked frames */
    while (shadow_live_count > 0u) {
        uint32_t fidx = shadow_live_indices[--shadow_live_count];
        ShadowSlot& s = shadow[fidx];
        if (s.mapped) {
            CHECK(k2_aspace_unmap(aspaces[s.aspace_idx], &pool, s.va) == K2_OK);
            s.mapped = false;
        }
        CHECK(k2_frame_free_user(&pool, &s.handle) == K2_OK);
        s.allocated = false;
        shadow_user_alloc -= 1u;
    }

    CHECK(shadow_user_alloc == 0u);
    CHECK(pool.user_allocated_frames == 0u);
    CHECK(pool.free_frames == (kTotalFrames - kKernelReserved));
    CHECK(k2_frame_pool_invariants_hold(&pool) == 1);

    /* OS_ENABLEMENT_PLAN K2 Gate: 10,000 map/unmap cycles across RV64/AArch64/x86_64 with 0 leaks */
    for (uint32_t cycle = 0u; cycle < 10000u; ++cycle) {
        k2_aspace_t* as = aspaces[cycle % 3u];
        k2_page_frame_t f{};
        if (k2_frame_alloc_user(&pool, &f) != K2_OK) {
            CHECK(false);
            break;
        }
        uint64_t va = 0x00800000ULL + (static_cast<uint64_t>(cycle & 0x3Fu) << K2_PAGE_SHIFT);
        uint64_t phys = 0ULL;
        if (k2_aspace_map(as, &pool, va, &f, K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) != K2_OK ||
            k2_aspace_translate(as, va, K2_ACCESS_WRITE | K2_ACCESS_USER, &phys) != K2_OK ||
            phys != f.phys_addr ||
            k2_aspace_unmap(as, &pool, va) != K2_OK ||
            k2_frame_free_user(&pool, &f) != K2_OK) {
            CHECK(false);
            break;
        }
    }
    CHECK(pool.user_allocated_frames == 0u);
    CHECK(pool.free_frames == (kTotalFrames - kKernelReserved));
    CHECK(k2_frame_pool_invariants_hold(&pool) == 1);

    stats_out.conservation_checks = conservation_checks;
    stats_out.double_free_rejects = double_free_rejects;
    stats_out.stale_handle_rejects = stale_handle_rejects;
    stats_out.map_unmap_cycles = 10000ULL;
}

/* Guard 4: MCU MPU Overlap & Guard-Zone Negative Controls. */
void test_mcu_mpu_region_manager() {
    k2_mpu_table_t mpu{};
    /* Cortex-M4 style memory map:
     * Vector table: 0x00000000..0x00000400 (1 KiB)
     * Kernel stack guard: 0x20000000..0x20000100 (256 B) */
    CHECK(k2_mpu_table_init(&mpu, 0x00000000ULL, 0x400ULL, 0x20000000ULL, 0x100ULL) == K2_OK);

    /* Negative 4A: User region overlapping vector table -> K2_ERR_MPU_RESERVED_GUARD */
    CHECK(k2_mpu_add_region(&mpu, 0u, 0x00000000ULL, 0x1000ULL,
                            K2_PERM_READ | K2_PERM_EXEC | K2_PERM_USER) ==
          K2_ERR_MPU_RESERVED_GUARD);

    /* Negative 4B: User region overlapping kernel stack guard -> K2_ERR_MPU_RESERVED_GUARD */
    CHECK(k2_mpu_add_region(&mpu, 0u, 0x20000000ULL, 0x1000ULL,
                            K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) ==
          K2_ERR_MPU_RESERVED_GUARD);

    /* Negative 4C: Non-power-of-two size or misaligned base -> K2_ERR_MPU_ALIGNMENT */
    CHECK(k2_mpu_add_region(&mpu, 0u, 0x20004000ULL, 0x300ULL,
                            K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) ==
          K2_ERR_MPU_ALIGNMENT);
    CHECK(k2_mpu_add_region(&mpu, 0u, 0x20004100ULL, 0x1000ULL,
                            K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) ==
          K2_ERR_MPU_ALIGNMENT);

    /* Negative 4D: W^X violation on MPU region -> K2_ERR_WX_VIOLATION */
    CHECK(k2_mpu_add_region(&mpu, 0u, 0x20004000ULL, 0x1000ULL,
                            K2_PERM_READ | K2_PERM_WRITE | K2_PERM_EXEC | K2_PERM_USER) ==
          K2_ERR_WX_VIOLATION);

    /* Positive: Install valid disjoint user code (RX) and user SRAM (RW) regions */
    CHECK(k2_mpu_add_region(&mpu, 0u, 0x00010000ULL, 0x4000ULL,
                            K2_PERM_READ | K2_PERM_EXEC | K2_PERM_USER) == K2_OK);
    CHECK(k2_mpu_add_region(&mpu, 1u, 0x20004000ULL, 0x2000ULL,
                            K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) == K2_OK);
    CHECK(mpu.region_count == 2u);

    /* Negative 4E: Overlapping MPU region -> K2_ERR_MPU_OVERLAP */
    CHECK(k2_mpu_add_region(&mpu, 2u, 0x20005000ULL, 0x1000ULL,
                            K2_PERM_READ | K2_PERM_USER) == K2_ERR_MPU_OVERLAP);
    CHECK(k2_mpu_add_region(&mpu, 2u, 0x00012000ULL, 0x2000ULL,
                            K2_PERM_READ | K2_PERM_USER) == K2_ERR_MPU_OVERLAP);

    /* Access authorization positive & negative controls */
    CHECK(k2_mpu_check_access(&mpu, 0x00010100ULL, 4ULL, K2_ACCESS_EXEC | K2_ACCESS_USER) == K2_OK);
    CHECK(k2_mpu_check_access(&mpu, 0x00010100ULL, 4ULL, K2_ACCESS_WRITE | K2_ACCESS_USER) == K2_ERR_ACCESS_DENIED);
    CHECK(k2_mpu_check_access(&mpu, 0x20004200ULL, 8ULL, K2_ACCESS_WRITE | K2_ACCESS_USER) == K2_OK);
    CHECK(k2_mpu_check_access(&mpu, 0x20004200ULL, 8ULL, K2_ACCESS_EXEC | K2_ACCESS_USER) == K2_ERR_ACCESS_DENIED);
    CHECK(k2_mpu_check_access(&mpu, 0x00000010ULL, 4ULL, K2_ACCESS_READ | K2_ACCESS_USER) == K2_ERR_MPU_RESERVED_GUARD);
    CHECK(k2_mpu_check_access(&mpu, 0x20000040ULL, 4ULL, K2_ACCESS_WRITE | K2_ACCESS_USER) == K2_ERR_MPU_RESERVED_GUARD);
}

} /* namespace */

/* Runtime zero-allocation guard: intercept all C++ global operator new calls. */
void* operator new(std::size_t sz) {
    if (g_track_allocs.load(std::memory_order_relaxed)) {
        g_heap_alloc_count.fetch_add(1ULL, std::memory_order_relaxed);
    }
    void* p = std::malloc(sz == 0u ? 1u : sz);
    if (!p) {
        throw std::bad_alloc();
    }
    return p;
}

void* operator new[](std::size_t sz) {
    return ::operator new(sz);
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete[](void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}

int main() {
    g_heap_alloc_count.store(0ULL, std::memory_order_relaxed);
    g_track_allocs.store(true, std::memory_order_release);

    uint64_t shootdowns = 0ULL;
    uint64_t shootdown_ns = 0ULL;
    ConservationStats stats{};

    test_wx_priv_and_tlb_shootdown(shootdowns, shootdown_ns);
    test_kernel_and_permission_aliasing();
    test_100k_conservation_and_10k_map_unmap(stats);
    test_mcu_mpu_region_manager();

    g_track_allocs.store(false, std::memory_order_release);
    uint64_t allocs = g_heap_alloc_count.load(std::memory_order_acquire);
    CHECK(allocs == 0ULL);

    std::printf("K2 MMU & Frames Test Suite: %d passed, %d failed "
                "(heap_allocs=%llu, conservation_checks=%llu/100000, "
                "double_free_rejects=%llu, stale_handle_rejects=%llu, "
                "map_unmap_cycles=%llu, tlb_shootdowns=%llu, tlb_bench_ns=%llu)\n",
                g_checks_passed,
                g_checks_failed,
                static_cast<unsigned long long>(allocs),
                static_cast<unsigned long long>(stats.conservation_checks),
                static_cast<unsigned long long>(stats.double_free_rejects),
                static_cast<unsigned long long>(stats.stale_handle_rejects),
                static_cast<unsigned long long>(stats.map_unmap_cycles),
                static_cast<unsigned long long>(shootdowns),
                static_cast<unsigned long long>(shootdown_ns));
    return (g_checks_failed == 0 && allocs == 0ULL) ? 0 : 1;
}
