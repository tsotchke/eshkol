/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K5/K6 Formal Policy Verification & Unified K1-K3
 * Microkernel Convergence Verification Suite:
 *   1. Exhaustive Bounded State-Space Model Checker (BFS) exploring
 *      thousands of interleaved reachable states across supervisor faults,
 *      restarts, capability revocations, lock contention, compositor
 *      surface arbitration, and audio overrun/underrun degradation.
 *   2. End-to-End Unified K1 + K2 + K3 + Policy Microkernel Integration
 *      Test executing with strictly zero runtime heap allocations
 *      (`heap_allocs == 0`).
 */

#include <eshkol/kernel/k_kernel_policy.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <unordered_set>
#include <vector>

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

void write_be32(uint8_t* dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>((v >> 24u) & 0xFFu);
    dst[1] = static_cast<uint8_t>((v >> 16u) & 0xFFu);
    dst[2] = static_cast<uint8_t>((v >> 8u) & 0xFFu);
    dst[3] = static_cast<uint8_t>(v & 0xFFu);
}

void build_valid_dtb(uint8_t buf[64]) {
    std::memset(buf, 0, 64);
    write_be32(buf + 0, K1_DTB_MAGIC);
    write_be32(buf + 4, 64u);
    write_be32(buf + 8, 48u);
    write_be32(buf + 12, 56u);
    write_be32(buf + 16, 40u);
    write_be32(buf + 20, 17u);
    write_be32(buf + 24, 16u);
    write_be32(buf + 28, 0u);
    write_be32(buf + 32, 8u);
    write_be32(buf + 36, 8u);
}

/* Pack canonical bounded model-checker state key (captures service states, cap liveness,
 * lock ownership, monotonic restart epoch mod bound, and audio ring fill level). */
uint64_t pack_policy_state_key(const k_policy_state_t& pol) {
    uint64_t k = 0ULL;
    for (uint32_t s = 0u; s < 2u; ++s) {
        const k_policy_service_t& svc = pol.services[s];
        uint64_t gen_clamped = (svc.token_gen > 3u) ? 3u : svc.token_gen;
        uint64_t fg_clamped = (svc.last_faulted_gen > 3u) ? 3u : svc.last_faulted_gen;
        uint64_t bits = (static_cast<uint64_t>(svc.state) & 0x3ULL) |
                        ((static_cast<uint64_t>(svc.root_cap_live) & 0x1ULL) << 2u) |
                        ((gen_clamped & 0x3ULL) << 3u) |
                        ((fg_clamped & 0x3ULL) << 5u);
        k |= (bits << (s * 8u));
    }
    for (uint32_t l = 0u; l < 2u; ++l) {
        uint64_t ow = (pol.lock_owner[l] == K_POL_NO_OWNER) ? 3ULL : (pol.lock_owner[l] & 0x3ULL);
        k |= (ow << (16u + l * 2u));
    }
    k |= (static_cast<uint64_t>(pol.audio.count & 0x7u) << 20u);
    k |= (static_cast<uint64_t>(pol.framebuffer[0] & 0x3u) << 24u);
    return k;
}

/* Part 1: Exhaustive Bounded State-Space Model Checker (BFS). */
void run_bounded_state_space_model_checker(uint64_t& states_explored_out,
                                           uint64_t& transitions_checked_out) {
    k_policy_state_t init{};
    k_policy_init(&init, 3u); /* 3-slot audio ring for exhaustive overrun/underrun exploration */

    uint32_t tok0 = 0u, tok1 = 0u;
    CHECK(k_policy_service_register(&init, 0u, 100u, &tok0) == K_POL_OK);
    CHECK(k_policy_service_register(&init, 1u, 200u, &tok1) == K_POL_OK);
    CHECK(k_policy_surface_register(&init, 0u, 100u, 0u, 16u, 1u) == K_POL_OK);  /* RW badge 100 */
    CHECK(k_policy_surface_register(&init, 1u, 200u, 16u, 16u, 0u) == K_POL_OK); /* RO badge 200 */
    CHECK(k_policy_invariants_hold(&init) == 1);

    std::unordered_set<uint64_t> visited;
    std::vector<k_policy_state_t> frontier;
    visited.reserve(16384);
    frontier.reserve(16384);

    visited.insert(pack_policy_state_key(init));
    frontier.push_back(init);

    uint64_t transitions = 0ULL;
    size_t head = 0u;

    while (head < frontier.size() && visited.size() < 8000u) {
        const k_policy_state_t cur = frontier[head++];
        CHECK(k_policy_invariants_hold(&cur) == 1);

        /* Always verify adversarial negative controls at EVERY reachable state:
         * (A) Read-only surface write MUST trap with K_POL_ERR_RO_TRAP & zero FB mutation.
         * (B) Cross-badge surface write MUST fail closed with K_POL_ERR_SURFACE_CONFLICT & zero FB mutation.
         * (C) Stale faulted token (if any) MUST fail closed with K_POL_ERR_STALE_TOKEN or K_POL_ERR_REVOKED_CAP. */
        {
            k_policy_state_t probe = cur;
            uint32_t fb_snap[K_POL_FB_CELLS];
            std::memcpy(fb_snap, probe.framebuffer, sizeof(fb_snap));
            if (k_policy_surface_write(&probe, 1u, 200u, 0u, 4u, 0xCAFEu) != K_POL_ERR_RO_TRAP ||
                std::memcmp(probe.framebuffer, fb_snap, sizeof(fb_snap)) != 0) {
                CHECK(false);
            }
            if (k_policy_surface_write(&probe, 0u, 200u, 0u, 4u, 0xBEEFu) != K_POL_ERR_SURFACE_CONFLICT ||
                std::memcmp(probe.framebuffer, fb_snap, sizeof(fb_snap)) != 0) {
                CHECK(false);
            }
            for (uint32_t s = 0u; s < 2u; ++s) {
                uint32_t fgen = cur.services[s].last_faulted_gen;
                if (fgen > 0u) {
                    k_pol_status_t vst = k_policy_validate_token(&cur, s, fgen);
                    if (vst != K_POL_ERR_STALE_TOKEN && vst != K_POL_ERR_REVOKED_CAP) {
                        CHECK(false);
                    }
                }
            }
        }

        /* Explore all 12 enabled transitions from `cur`. */
        for (uint32_t act = 0u; act < 12u; ++act) {
            k_policy_state_t next = cur;
            transitions += 1ULL;

            switch (act) {
                case 0u: { /* Fault service 0 */
                    uint32_t rel = 0u;
                    if (k_policy_service_fault(&next, 0u, &rel) != K_POL_OK) CHECK(false);
                    /* Zero zombie locks check immediately after fault */
                    for (uint32_t l = 0u; l < K_POL_MAX_LOCKS; ++l) {
                        if (next.lock_owner[l] == 0u) CHECK(false);
                    }
                    break;
                }
                case 1u: { /* Fault service 1 */
                    uint32_t rel = 0u;
                    if (k_policy_service_fault(&next, 1u, &rel) != K_POL_OK) CHECK(false);
                    for (uint32_t l = 0u; l < K_POL_MAX_LOCKS; ++l) {
                        if (next.lock_owner[l] == 1u) CHECK(false);
                    }
                    break;
                }
                case 2u: { /* Restart service 0 (cap bound at gen <= 3 to keep state space finite) */
                    if (next.services[0u].token_gen >= 3u) continue;
                    uint32_t prev_fg = next.services[0u].last_faulted_gen;
                    uint32_t was_revoked = (next.services[0u].root_cap_live == 0u);
                    uint32_t new_gen = 0u;
                    k_pol_status_t st = k_policy_service_restart(&next, 0u, &new_gen);
                    if (was_revoked) {
                        if (st != K_POL_ERR_REVOKED_CAP || next.services[0u].state != K_POL_SVC_STOPPED) CHECK(false);
                    } else if (st == K_POL_OK) {
                        if (new_gen <= prev_fg) CHECK(false);
                    }
                    break;
                }
                case 3u: { /* Restart service 1 */
                    if (next.services[1u].token_gen >= 3u) continue;
                    uint32_t prev_fg = next.services[1u].last_faulted_gen;
                    uint32_t was_revoked = (next.services[1u].root_cap_live == 0u);
                    uint32_t new_gen = 0u;
                    k_pol_status_t st = k_policy_service_restart(&next, 1u, &new_gen);
                    if (was_revoked) {
                        if (st != K_POL_ERR_REVOKED_CAP || next.services[1u].state != K_POL_SVC_STOPPED) CHECK(false);
                    } else if (st == K_POL_OK) {
                        if (new_gen <= prev_fg) CHECK(false);
                    }
                    break;
                }
                case 4u: { /* Revoke service 0 root capability */
                    (void)k_policy_revoke_service_cap(&next, 0u, nullptr);
                    break;
                }
                case 5u: { /* Revoke service 1 root capability */
                    (void)k_policy_revoke_service_cap(&next, 1u, nullptr);
                    break;
                }
                case 6u: { /* Service 0 acquires Lock 0 */
                    (void)k_policy_lock_acquire(&next, 0u, next.services[0u].token_gen, 0u);
                    break;
                }
                case 7u: { /* Service 1 acquires Lock 0 */
                    (void)k_policy_lock_acquire(&next, 1u, next.services[1u].token_gen, 0u);
                    break;
                }
                case 8u: { /* Service 0 acquires Lock 1 */
                    (void)k_policy_lock_acquire(&next, 0u, next.services[0u].token_gen, 1u);
                    break;
                }
                case 9u: { /* Compositor valid RW surface write by badge 100 */
                    uint32_t pix = (next.framebuffer[0] + 1u) & 0x3u;
                    if (k_policy_surface_write(&next, 0u, 100u, 0u, 4u, pix) != K_POL_OK) CHECK(false);
                    break;
                }
                case 10u: { /* Audio produce (including overrun) */
                    uint32_t deg = 0u;
                    if (k_policy_audio_step(&next, 1u, 0x1234u, nullptr, &deg) != K_POL_OK) CHECK(false);
                    if (next.audio.count > next.audio.capacity || next.audio.supervisor_halted != 0u) CHECK(false);
                    break;
                }
                case 11u: { /* Audio consume (including underrun) */
                    uint32_t sample = 0u, deg = 0u;
                    if (k_policy_audio_step(&next, 0u, 0u, &sample, &deg) != K_POL_OK) CHECK(false);
                    if (next.audio.count > next.audio.capacity || next.audio.supervisor_halted != 0u) CHECK(false);
                    break;
                }
            }

            if (k_policy_invariants_hold(&next) != 1) {
                CHECK(false);
            }

            uint64_t key = pack_policy_state_key(next);
            if (visited.find(key) == visited.end()) {
                visited.insert(key);
                frontier.push_back(next);
            }
        }
    }

    states_explored_out = static_cast<uint64_t>(visited.size());
    transitions_checked_out = transitions;
    CHECK(states_explored_out >= 1000ULL);
    CHECK(transitions_checked_out >= 10000ULL);
}

/* Part 2: End-to-End Unified K1 + K2 + K3 + Policy Microkernel Convergence (Zero Heap Allocs!). */
void test_unified_microkernel_convergence_zero_alloc() {
    static k_microkernel_t mk;
    uint8_t dtb[64];
    build_valid_dtb(dtb);

    k1_boot_info_t boot{};
    boot.isa = K1_ISA_RISCV64;
    boot.boot_kind = K1_BOOT_DTB;
    boot.cpu_count = 4u;
    boot.boot_cpu_id = 0u;
    boot.ram_base = 0x80000000ULL;
    boot.ram_size = 0x10000000ULL;
    boot.console_mmio_base = 0x10000000ULL;
    boot.handoff_payload = dtb;
    boot.handoff_size = sizeof(dtb);

    k1_kernel_context_t baseline_ctx{};
    k1_kernel_context_init(&baseline_ctx, 0xFFFF800000080000ULL, 0xFFFF800000002000ULL, 0x200ULL);
    for (uint32_t i = 0u; i < K1_KERNEL_GPRS; ++i) {
        baseline_ctx.gpr[i] = 0x55AA000000000000ULL | (static_cast<uint64_t>(i + 1u) * 0x1111111111111111ULL);
    }

    CHECK(k_microkernel_converge_boot(&mk, &boot, &baseline_ctx, K2_MMU_RV64_SV39, 256u, 16u) == K_POL_OK);
    CHECK(k2_frame_pool_invariants_hold(&mk.k2_frames) == 1);
    CHECK(k_policy_invariants_hold(&mk.policy) == 1);

    /* Register 3 isolated services:
     *   Service 0: compositor (badge 10, cpu 0)
     *   Service 1: client app (badge 20, cpu 1)
     *   Service 2: audio daemon (badge 30, cpu 2) */
    uint32_t tok0 = 0u, tok1 = 0u, tok2 = 0u;
    CHECK(k_policy_service_register(&mk.policy, 0u, 10u, &tok0) == K_POL_OK);
    CHECK(k_policy_service_register(&mk.policy, 1u, 20u, &tok1) == K_POL_OK);
    CHECK(k_policy_service_register(&mk.policy, 2u, 30u, &tok2) == K_POL_OK);

    /* Register K1 user tasks + K3 SMP tasks + K2 W^X user frames */
    k2_page_frame_t svc_frames[3]{};
    for (uint32_t s = 0u; s < 3u; ++s) {
        CHECK(k1_user_task_register(&mk.k1, s, 0x400000ULL + s * 0x10000ULL, 0x7FFF0000ULL) == K1_OK);
        CHECK(k3_smp_task_spawn(&mk.k3_sched, s, s, s, 5u) == K3_OK);
        CHECK(k2_frame_alloc_user(&mk.k2_frames, &svc_frames[s]) == K2_OK);
        CHECK(k2_aspace_map(&mk.k2_aspaces[s], &mk.k2_frames, 0x400000ULL,
                            &svc_frames[s], K2_PERM_READ | K2_PERM_WRITE | K2_PERM_USER) == K2_OK);
    }

    /* Negative: W^X attempt on Service 1 rejected fail-closed */
    CHECK(k2_aspace_remap_perms(&mk.k2_aspaces[1u], &mk.k2_frames, 0x400000ULL,
                                K2_PERM_READ | K2_PERM_WRITE | K2_PERM_EXEC | K2_PERM_USER) ==
          K2_ERR_WX_VIOLATION);

    /* Mint CDT capabilities for IPC channel 2 (Service 1 client -> Service 0 compositor server) */
    k3_cap_handle_t root_send{}, child_send{}, srv_recv{};
    CHECK(k3_cdt_mint_root(&mk.k3_cdt, 1u, 2u,
                           K3_RIGHT_SEND | K3_RIGHT_DERIVE | K3_RIGHT_REVOKE, &root_send) == K3_OK);
    CHECK(k3_cdt_derive(&mk.k3_cdt, &root_send, 1u, K3_RIGHT_SEND, &child_send) == K3_OK);
    CHECK(k3_cdt_mint_root(&mk.k3_cdt, 0u, 2u, K3_RIGHT_RECV | K3_RIGHT_SEND, &srv_recv) == K3_OK);

    /* Register non-overlapping compositor surfaces (Surface 0 RW badge 10, Surface 1 RO badge 20) */
    CHECK(k_policy_surface_register(&mk.policy, 0u, 10u, 0u, 32u, 1u) == K_POL_OK);
    CHECK(k_policy_surface_register(&mk.policy, 1u, 20u, 32u, 32u, 0u) == K_POL_OK);
    /* Negative: overlapping surface registration rejected fail-closed */
    CHECK(k_policy_surface_register(&mk.policy, 2u, 30u, 16u, 32u, 1u) == K_POL_ERR_SURFACE_CONFLICT);

    /* Fire 1,000 K1 timer IRQs across 4 CPUs (250 each) + drain cleanly */
    static k1_irq_event_t irq_drain[K1_IRQ_RING_CAPACITY];
    for (uint32_t i = 0u; i < 1000u; ++i) {
        CHECK(k1_dispatch_timer_irq(&mk.k1, i & 3u, i, 7u) == K1_OK);
    }
    for (uint32_t c = 0u; c < 4u; ++c) {
        CHECK(k1_drain_cpu_irqs(&mk.k1, c, irq_drain, K1_IRQ_RING_CAPACITY) == 250u);
    }

    /* Service 1 acquires Lock 0 and submits an in-flight IPC request */
    CHECK(k_policy_lock_acquire(&mk.policy, 1u, tok1, 0u) == K_POL_OK);
    CHECK(mk.policy.lock_owner[0u] == 1u);

    const uint8_t req_msg[8] = {0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4};
    CHECK(k3_ipc_submit(&mk.k3_channels[1u], &mk.k3_cdt, &child_send, 1u,
                        K3_IPC_MSG_REQUEST, req_msg, sizeof(req_msg), 0x101ULL, nullptr) == K3_OK);

    /* Inject real RV64 illegal instruction trap (0x00000000 -> cause 2) into Service 1! */
    k1_trap_frame_t trap{};
    trap.isa = K1_ISA_RISCV64;
    trap.origin_mode = K1_MODE_USER;
    trap.cpu_id = 1u;
    trap.task_id = 1u;
    trap.raw_cause = K1_RISCV_CAUSE_ILLEGAL_INSN;
    trap.fault_pc = 0x410020ULL;
    trap.insn_len = 4u; /* zeroed insn_bytes = {0,0,0,0} */

    k1_trap_outcome_t tout = k1_dispatch_user_trap(&mk.k1, &trap);
    CHECK(tout.status == K1_OK);
    CHECK(tout.trap_code == K1_TRAP_ILLEGAL_INSN);
    CHECK(tout.native_cause == K1_RISCV_CAUSE_ILLEGAL_INSN);
    CHECK(tout.task_terminated == 1u);
    CHECK(tout.kernel_intact == 1u);
    CHECK(std::memcmp(&mk.k1.kernel_ctx, &baseline_ctx, sizeof(baseline_ctx)) == 0);

    /* Converged fault containment across K3 + Policy: purge SMP task, revoke CDT subtree, release locks! */
    uint32_t rev_cnt = 0u, rel_locks = 0u;
    CHECK(k3_smp_fault_task(&mk.k3_sched, 1u, tout.native_cause) == K3_OK);
    CHECK(k3_cdt_revoke_subtree(&mk.k3_cdt, &root_send, &rev_cnt) == K3_OK);
    CHECK(rev_cnt == 2u);
    CHECK(k_policy_service_fault(&mk.policy, 1u, &rel_locks) == K_POL_OK);
    CHECK(rel_locks == 1u);
    CHECK(mk.policy.lock_owner[0u] == K_POL_NO_OWNER);

    /* Zero zombie lock proof: Service 0 acquires Lock 0 immediately after Service 1 fault! */
    CHECK(k_policy_lock_acquire(&mk.policy, 0u, tok0, 0u) == K_POL_OK);
    CHECK(k_policy_lock_release(&mk.policy, 0u, tok0, 0u) == K_POL_OK);

    /* Mid-flight revoked IPC completion fails closed with zero side effects */
    uint8_t rx_buf[16];
    uint8_t rx_canary[16];
    std::memset(rx_buf, 0x77, sizeof(rx_buf));
    std::memset(rx_canary, 0x77, sizeof(rx_canary));
    CHECK(k3_ipc_complete(&mk.k3_channels[1u], &mk.k3_cdt, &srv_recv, 0u,
                          rx_buf, sizeof(rx_buf), nullptr, nullptr, nullptr) == K3_ERR_CAP_REVOKED);
    CHECK(std::memcmp(rx_buf, rx_canary, sizeof(rx_buf)) == 0);

    /* Restart Service 1: fresh monotonic token > old token; prior faulted token permanently rejected! */
    uint32_t tok1_restarted = 0u;
    CHECK(k_policy_service_restart(&mk.policy, 1u, &tok1_restarted) == K_POL_OK);
    CHECK(tok1_restarted > tok1);
    CHECK(k_policy_validate_token(&mk.policy, 1u, tok1) == K_POL_ERR_STALE_TOKEN);
    CHECK(k_policy_lock_acquire(&mk.policy, 1u, tok1, 0u) == K_POL_ERR_STALE_TOKEN);
    CHECK(k_policy_lock_acquire(&mk.policy, 1u, tok1_restarted, 0u) == K_POL_OK);
    CHECK(k_policy_lock_release(&mk.policy, 1u, tok1_restarted, 0u) == K_POL_OK);

    /* Revoke Service 2 root capability -> restart MUST fail closed with K_POL_ERR_REVOKED_CAP */
    CHECK(k_policy_revoke_service_cap(&mk.policy, 2u, nullptr) == K_POL_OK);
    CHECK(k_policy_service_restart(&mk.policy, 2u, nullptr) == K_POL_ERR_REVOKED_CAP);
    CHECK(mk.policy.services[2u].state == K_POL_SVC_STOPPED);

    /* Compositor RO trap + cross-badge conflict + valid RW write */
    CHECK(k_policy_surface_write(&mk.policy, 0u, 10u, 0u, 8u, 0x11223344u) == K_POL_OK);
    CHECK(k_policy_surface_write(&mk.policy, 1u, 20u, 0u, 4u, 0x99999999u) == K_POL_ERR_RO_TRAP);
    CHECK(mk.policy.framebuffer[32u] == 0u);
    CHECK(k_policy_surface_write(&mk.policy, 0u, 20u, 0u, 4u, 0x88888888u) == K_POL_ERR_SURFACE_CONFLICT);
    CHECK(mk.policy.framebuffer[0u] == 0x11223344u);

    /* Audio overrun + underrun bounded degradation without halting */
    uint32_t deg = 0u, sample = 0u;
    for (uint32_t i = 0u; i < K_POL_AUDIO_RING_CAP + 10u; ++i) {
        CHECK(k_policy_audio_step(&mk.policy, 1u, i + 1u, nullptr, &deg) == K_POL_OK);
    }
    CHECK(mk.policy.audio.count == K_POL_AUDIO_RING_CAP);
    CHECK(mk.policy.audio.overrun_drops == 10u);
    CHECK(mk.policy.audio.supervisor_halted == 0u);

    for (uint32_t i = 0u; i < K_POL_AUDIO_RING_CAP + 5u; ++i) {
        CHECK(k_policy_audio_step(&mk.policy, 0u, 0u, &sample, &deg) == K_POL_OK);
    }
    CHECK(mk.policy.audio.count == 0u);
    CHECK(mk.policy.audio.underrun_silences == 5u);
    CHECK(mk.policy.audio.supervisor_halted == 0u);

    /* Clean teardown: unmap and free all K2 frames -> 0 leaked frames */
    for (uint32_t s = 0u; s < 3u; ++s) {
        CHECK(k2_aspace_unmap(&mk.k2_aspaces[s], &mk.k2_frames, 0x400000ULL) == K2_OK);
        CHECK(k2_frame_free_user(&mk.k2_frames, &svc_frames[s]) == K2_OK);
    }
    CHECK(mk.k2_frames.user_allocated_frames == 0u);
    CHECK(k2_frame_pool_invariants_hold(&mk.k2_frames) == 1);
    CHECK(k_policy_invariants_hold(&mk.policy) == 1);
    CHECK(std::memcmp(&mk.k1.kernel_ctx, &baseline_ctx, sizeof(baseline_ctx)) == 0);
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
    /* Part 1: Exhaustive bounded state-space model checker (BFS uses host std::vector/set). */
    uint64_t states_explored = 0ULL;
    uint64_t transitions_checked = 0ULL;
    run_bounded_state_space_model_checker(states_explored, transitions_checked);

    /* Part 2: Unified K1+K2+K3+Policy convergence test under strict zero-alloc guard. */
    g_heap_alloc_count.store(0ULL, std::memory_order_relaxed);
    g_track_allocs.store(true, std::memory_order_release);
    test_unified_microkernel_convergence_zero_alloc();
    g_track_allocs.store(false, std::memory_order_release);

    uint64_t allocs = g_heap_alloc_count.load(std::memory_order_acquire);
    CHECK(allocs == 0ULL);

    std::printf("K5/K6 Convergence & Policy Model-Checker Suite: %d passed, %d failed "
                "(heap_allocs=%llu, bfs_states=%llu, bfs_transitions=%llu)\n",
                g_checks_passed,
                g_checks_failed,
                static_cast<unsigned long long>(allocs),
                static_cast<unsigned long long>(states_explored),
                static_cast<unsigned long long>(transitions_checked));
    return (g_checks_failed == 0 && allocs == 0ULL) ? 0 : 1;
}
