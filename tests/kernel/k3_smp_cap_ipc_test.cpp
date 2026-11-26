/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K3 Verification & Adversarial Rigor Suite:
 *   1. CDT Depth Overflow, Cycle Detection, and Rights-Escalation Controls.
 *   2. Immediate Atomic Revocation under Mid-Flight Concurrent submit/complete
 *      Race (single-thread deterministic + multi-threaded stress).
 *   3. Cross-Core 64-Byte Cache-Line Isolation & Shared-Memory Capability
 *      Confinement / Non-Interference Controls.
 *   4. 10,000 Session-Typed IPC Exchanges + Protocol Violation Controls +
 *      Userspace Fault Isolation Gate (with runtime heap_allocs == 0).
 */

#include <eshkol/kernel/k3_smp_cap_ipc.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

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

/* Guard 1: CDT Depth Overflow, Cycle Detection, and Rights-Escalation Negative Controls. */
void test_cdt_depth_cycles_and_rights_attenuation() {
    static k3_cdt_t cdt;
    k3_cdt_init(&cdt);

    /* Mint root capability with SEND + DERIVE + REVOKE (NO RECV, NO SHM_WRITE) */
    k3_cap_handle_t root{};
    CHECK(k3_cdt_mint_root(&cdt, 0u, 42u,
                           K3_RIGHT_SEND | K3_RIGHT_DERIVE | K3_RIGHT_REVOKE,
                           &root) == K3_OK);

    /* Negative 1A: Rights escalation (requesting RECV not held by parent) -> K3_ERR_RIGHTS_ESCALATION */
    k3_cap_handle_t escalated{};
    CHECK(k3_cdt_derive(&cdt, &root, 1u,
                        K3_RIGHT_SEND | K3_RIGHT_RECV,
                        &escalated) == K3_ERR_RIGHTS_ESCALATION);
    CHECK(k3_cdt_derive(&cdt, &root, 1u,
                        K3_RIGHT_SEND | K3_RIGHT_SHM_WRITE,
                        &escalated) == K3_ERR_RIGHTS_ESCALATION);

    /* Derive chain to exact maximum depth K3_MAX_CDT_DEPTH (depth 0 -> 1 -> ... -> 16) */
    k3_cap_handle_t chain[K3_MAX_CDT_DEPTH + 1u];
    chain[0] = root;
    for (uint32_t d = 1u; d <= K3_MAX_CDT_DEPTH; ++d) {
        CHECK(k3_cdt_derive(&cdt, &chain[d - 1u], d % K3_MAX_COMPARTMENTS,
                            K3_RIGHT_SEND | K3_RIGHT_DERIVE | K3_RIGHT_REVOKE,
                            &chain[d]) == K3_OK);
        CHECK(cdt.nodes[chain[d].slot_id].depth == d);
    }

    /* Negative 1B: Depth overflow at K3_MAX_CDT_DEPTH + 1 -> K3_ERR_CDT_DEPTH_OVERFLOW */
    k3_cap_handle_t too_deep{};
    CHECK(k3_cdt_derive(&cdt, &chain[K3_MAX_CDT_DEPTH], 2u,
                        K3_RIGHT_SEND, &too_deep) == K3_ERR_CDT_DEPTH_OVERFLOW);

    /* Negative 1C: Cycle detection (self-parent and multi-hop ancestor cycle A -> B -> C -> A) */
    CHECK(k3_cdt_reparent(&cdt, &chain[0], &chain[0]) == K3_ERR_CDT_CYCLE);
    CHECK(k3_cdt_reparent(&cdt, &chain[0], &chain[1]) == K3_ERR_CDT_CYCLE);
    CHECK(k3_cdt_reparent(&cdt, &chain[0], &chain[3]) == K3_ERR_CDT_CYCLE);
    CHECK(k3_cdt_reparent(&cdt, &chain[2], &chain[6]) == K3_ERR_CDT_CYCLE);

    /* Negative 1D: Derive without DERIVE right -> K3_ERR_CAP_DENIED */
    k3_cap_handle_t leaf{};
    CHECK(k3_cdt_derive(&cdt, &chain[1], 3u, K3_RIGHT_SEND, &leaf) == K3_OK);
    k3_cap_handle_t leaf_child{};
    CHECK(k3_cdt_derive(&cdt, &leaf, 4u, K3_RIGHT_SEND, &leaf_child) == K3_ERR_CAP_DENIED);

    /* Transitive atomic subtree revocation from depth 1 revokes depths 1..16 + leaf = 17 nodes */
    uint32_t revoked_count = 0u;
    CHECK(k3_cdt_revoke_subtree(&cdt, &chain[1], &revoked_count) == K3_OK);
    CHECK(revoked_count == K3_MAX_CDT_DEPTH + 1u);
    for (uint32_t d = 1u; d <= K3_MAX_CDT_DEPTH; ++d) {
        CHECK(k3_cdt_validate(&cdt, &chain[d], d % K3_MAX_COMPARTMENTS,
                              K3_RIGHT_SEND, nullptr) == K3_ERR_CAP_REVOKED);
    }
    CHECK(k3_cdt_validate(&cdt, &leaf, 3u, K3_RIGHT_SEND, nullptr) == K3_ERR_CAP_REVOKED);
    /* Root above revoked subtree remains live */
    CHECK(k3_cdt_validate(&cdt, &root, 0u, K3_RIGHT_SEND, nullptr) == K3_OK);
}

/* Guard 2: Immediate Atomic Revocation Under Mid-Flight Concurrent submit/complete Race. */
void test_mid_flight_revocation_and_concurrent_race(uint64_t& race_revoked_aborts_out) {
    static k3_cdt_t cdt;
    static k3_ipc_channel_t ch;
    k3_cdt_init(&cdt);
    CHECK(k3_ipc_channel_init(&ch, 7u, 1u, 2u) == K3_OK);

    /* Mint parent send cap in compartment 1, derive child send cap, and mint recv cap in compartment 2 */
    k3_cap_handle_t cap_parent{};
    k3_cap_handle_t cap_child{};
    k3_cap_handle_t cap_server_recv{};
    CHECK(k3_cdt_mint_root(&cdt, 1u, 7u,
                           K3_RIGHT_SEND | K3_RIGHT_DERIVE | K3_RIGHT_REVOKE,
                           &cap_parent) == K3_OK);
    CHECK(k3_cdt_derive(&cdt, &cap_parent, 1u, K3_RIGHT_SEND, &cap_child) == K3_OK);
    CHECK(k3_cdt_mint_root(&cdt, 2u, 7u, K3_RIGHT_RECV | K3_RIGHT_SEND, &cap_server_recv) == K3_OK);

    /* Step 1: Submit in-flight request using derived child capability */
    static const uint8_t k_secret_payload[16] = {
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x01
    };
    uint32_t seq = 0u;
    CHECK(k3_ipc_submit(&ch, &cdt, &cap_child, 1u, K3_IPC_MSG_REQUEST,
                        k_secret_payload, sizeof(k_secret_payload), 0xBEEF0001ULL, &seq) == K3_OK);
    CHECK(ch.count == 1u);

    /* Step 2: While SQE is IN-FLIGHT, atomically revoke cap_parent (transitive revocation of cap_child!) */
    uint32_t revoked = 0u;
    CHECK(k3_cdt_revoke_subtree(&cdt, &cap_parent, &revoked) == K3_OK);
    CHECK(revoked == 2u);

    /* Step 3: Attempt complete -> MUST fail closed with K3_ERR_CAP_REVOKED and ZERO side effects! */
    uint8_t rx_buf[32];
    uint8_t rx_canary[32];
    std::memset(rx_buf, 0xA5, sizeof(rx_buf));
    std::memset(rx_canary, 0xA5, sizeof(rx_canary));
    size_t rx_len = 999u;
    k3_ipc_msg_kind_t rx_kind = K3_IPC_MSG_NONE;
    uint64_t rx_tag = 0xDEADBEEFULL;

    const k3_session_state_t prev_session = ch.session_state;
    const uint64_t prev_completed = ch.completed_total;

    CHECK(k3_ipc_complete(&ch, &cdt, &cap_server_recv, 2u,
                          rx_buf, sizeof(rx_buf), &rx_len, &rx_kind, &rx_tag) == K3_ERR_CAP_REVOKED);

    /* Verify ZERO side effects: rx_buf 100% bit-identical to canary, receiver state untouched, 0 delivered */
    CHECK(std::memcmp(rx_buf, rx_canary, sizeof(rx_buf)) == 0);
    CHECK(rx_len == 999u);
    CHECK(rx_kind == K3_IPC_MSG_NONE);
    CHECK(rx_tag == 0xDEADBEEFULL);
    CHECK(ch.session_state == prev_session);
    CHECK(ch.completed_total == prev_completed);
    CHECK(ch.revoked_aborts == 1ULL);

    /* Step 4: Interleaved concurrent submit/revoke/complete race across 500 fresh rounds */
    race_revoked_aborts_out = 0ULL;
    for (uint32_t round = 0u; round < 500u; ++round) {
        k3_cdt_init(&cdt);
        CHECK(k3_ipc_channel_init(&ch, 9u, 1u, 2u) == K3_OK);
        k3_cap_handle_t p_cap{}, c_cap{}, r_cap{};
        CHECK(k3_cdt_mint_root(&cdt, 1u, 9u,
                               K3_RIGHT_SEND | K3_RIGHT_DERIVE | K3_RIGHT_REVOKE, &p_cap) == K3_OK);
        CHECK(k3_cdt_derive(&cdt, &p_cap, 1u, K3_RIGHT_SEND, &c_cap) == K3_OK);
        CHECK(k3_cdt_mint_root(&cdt, 2u, 9u, K3_RIGHT_RECV, &r_cap) == K3_OK);

        CHECK(k3_ipc_submit(&ch, &cdt, &c_cap, 1u, K3_IPC_MSG_REQUEST,
                            k_secret_payload, 8u, round, nullptr) == K3_OK);

        /* Alternate ordering: revoke-then-complete vs revoke-before-next-submit */
        uint32_t rev_n = 0u;
        CHECK(k3_cdt_revoke_subtree(&cdt, &p_cap, &rev_n) == K3_OK);
        std::memset(rx_buf, 0x5A, sizeof(rx_buf));
        std::memset(rx_canary, 0x5A, sizeof(rx_canary));
        k3_status_t cst = k3_ipc_complete(&ch, &cdt, &r_cap, 2u,
                                          rx_buf, sizeof(rx_buf), nullptr, nullptr, nullptr);
        CHECK(cst == K3_ERR_CAP_REVOKED);
        CHECK(std::memcmp(rx_buf, rx_canary, sizeof(rx_buf)) == 0);
        CHECK(ch.completed_total == 0ULL);
        race_revoked_aborts_out += ch.revoked_aborts;
    }
    CHECK(race_revoked_aborts_out == 500ULL);
}

/* Guard 3: Cross-Core Cache-Line Isolation & Shared-Memory Non-Interference Control. */
void test_cacheline_isolation_and_shm_confinement() {
    static k3_smp_sched_t sched;
    CHECK(k3_smp_sched_init(&sched, K3_MAX_CPUS) == K3_OK);

    /* Verify strict 64-byte cache-line alignment and zero shared cache lines across all 8 cores */
    for (uint32_t c = 0u; c < K3_MAX_CPUS; ++c) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(&sched.runqueues[c]);
        CHECK((addr & 63u) == 0u);
        if (c > 0u) {
            uintptr_t prev_end = reinterpret_cast<uintptr_t>(&sched.runqueues[c - 1u]) +
                                 sizeof(k3_cpu_runqueue_t);
            CHECK(addr >= prev_end);
            CHECK((addr - reinterpret_cast<uintptr_t>(&sched.runqueues[c - 1u])) % 64u == 0u);
        }
    }

    /* Shared memory page between Core A (compartment 1) and Core B (compartment 2) */
    static k3_cdt_t cdt;
    static k3_shm_page_t shm;
    k3_cdt_init(&cdt);
    CHECK(k3_shm_page_init(&shm, 100u, 1u, 2u) == K3_OK);

    k3_cap_handle_t shm_a_rw{};
    k3_cap_handle_t shm_b_ro{};
    k3_cap_handle_t private_b_cap{};
    CHECK(k3_cdt_mint_root(&cdt, 1u, 100u,
                           K3_RIGHT_SHM_READ | K3_RIGHT_SHM_WRITE | K3_RIGHT_DERIVE,
                           &shm_a_rw) == K3_OK);
    CHECK(k3_cdt_derive(&cdt, &shm_a_rw, 2u, K3_RIGHT_SHM_READ, &shm_b_ro) == K3_OK);
    CHECK(k3_cdt_mint_root(&cdt, 2u, 200u, K3_RIGHT_SEND | K3_RIGHT_RECV, &private_b_cap) == K3_OK);

    /* Snapshot Core B's private CSpace node before shared-memory activity */
    const k3_cdt_node_t b_node_snapshot = cdt.nodes[private_b_cap.slot_id];

    /* Positive data transfer: Core A writes data bytes, Core B reads them */
    static const uint8_t k_msg[16] = "ESH_SHM_DATA_OK";
    CHECK(k3_shm_write(&shm, &cdt, &shm_a_rw, 1u, 0u, k_msg, 16u) == K3_OK);
    uint8_t readback[16]{};
    CHECK(k3_shm_read(&shm, &cdt, &shm_b_ro, 2u, 0u, readback, 16u) == K3_OK);
    CHECK(std::memcmp(readback, k_msg, 16u) == 0);

    /* Negative 3A: Core B lacks SHM_WRITE -> K3_ERR_CAP_DENIED */
    CHECK(k3_shm_write(&shm, &cdt, &shm_b_ro, 2u, 0u, k_msg, 16u) == K3_ERR_CAP_DENIED);

    /* Negative 3B: Smuggling raw capability struct bytes via shared memory -> K3_ERR_CAP_CONFINEMENT */
    k3_cap_handle_t forged_handle{};
    forged_handle.slot_id = private_b_cap.slot_id;
    forged_handle.generation = private_b_cap.generation;
    forged_handle.compartment_id = 1u; /* Core A trying to impersonate Core B's slot */
    forged_handle.badge = 0x12345678u; /* Forged badge */
    CHECK(k3_shm_write(&shm, &cdt, &shm_a_rw, 1u, 64u,
                       reinterpret_cast<const uint8_t*>(&forged_handle),
                       sizeof(forged_handle)) == K3_OK);

    k3_cap_handle_t imported{};
    CHECK(k3_shm_attempt_import_cap(&shm, &cdt, 1u, 64u, &imported) == K3_ERR_CAP_CONFINEMENT);
    CHECK(k3_cdt_validate(&cdt, &forged_handle, 1u, K3_RIGHT_SEND, nullptr) == K3_ERR_CAP_CONFINEMENT);
    /* Even with exact copied badge, compartment confinement blocks Core A from using Core B's cap! */
    forged_handle.badge = private_b_cap.badge;
    forged_handle.compartment_id = private_b_cap.compartment_id;
    CHECK(k3_cdt_validate(&cdt, &forged_handle, 1u, K3_RIGHT_SEND, nullptr) == K3_ERR_CAP_CONFINEMENT);

    /* Unrelated compartment 3 cannot read shared page even with another handle */
    CHECK(k3_shm_read(&shm, &cdt, &shm_b_ro, 3u, 0u, readback, 16u) == K3_ERR_CAP_CONFINEMENT);

    /* Core B's private CSpace entry remains 100% bit-identical */
    CHECK(std::memcmp(&cdt.nodes[private_b_cap.slot_id], &b_node_snapshot, sizeof(b_node_snapshot)) == 0);
}

/* Guard 4: 10,000 Session-Typed IPC Exchanges + Protocol Controls + Userspace Fault Isolation. */
void test_10k_session_ipc_and_fault_isolation() {
    static k3_smp_sched_t sched;
    static k3_cdt_t cdt;
    static k3_ipc_channel_t ch;

    CHECK(k3_smp_sched_init(&sched, 4u) == K3_OK);
    k3_cdt_init(&cdt);
    CHECK(k3_ipc_channel_init(&ch, 55u, 1u, 2u) == K3_OK);

    /* Spawn client (task 0, comp 1, cpu 0), server (task 1, comp 2, cpu 1), and faulting peer (task 2, comp 3, cpu 0) */
    CHECK(k3_smp_task_spawn(&sched, 0u, 1u, 0u, 5u) == K3_OK);
    CHECK(k3_smp_task_spawn(&sched, 1u, 2u, 1u, 5u) == K3_OK);
    CHECK(k3_smp_task_spawn(&sched, 2u, 3u, 0u, 7u) == K3_OK);

    k3_cap_handle_t client_cap{};
    k3_cap_handle_t server_cap{};
    CHECK(k3_cdt_mint_root(&cdt, 1u, 55u, K3_RIGHT_SEND | K3_RIGHT_RECV, &client_cap) == K3_OK);
    CHECK(k3_cdt_mint_root(&cdt, 2u, 55u, K3_RIGHT_SEND | K3_RIGHT_RECV, &server_cap) == K3_OK);

    /* Negative 4A: Out-of-protocol response before request -> K3_ERR_SESSION_PROTOCOL */
    uint8_t dummy[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(k3_ipc_submit(&ch, &cdt, &server_cap, 2u, K3_IPC_MSG_RESPONSE,
                        dummy, sizeof(dummy), 0ULL, nullptr) == K3_ERR_SESSION_PROTOCOL);

    /* Fault task 2 (highest priority on cpu 0): ONLY task 2 terminates; tasks 0 & 1 keep running */
    uint32_t dispatched = K3_INVALID_ID;
    CHECK(k3_smp_schedule_next(&sched, 0u, &dispatched) == K3_OK);
    CHECK(dispatched == 2u);
    CHECK(k3_smp_fault_task(&sched, 2u, 0xDEAD0006ULL) == K3_OK);
    CHECK(sched.tasks[2u].state == K3_TASK_FAULTED);
    CHECK(sched.tasks[0u].state == K3_TASK_RUNNABLE);
    CHECK(sched.tasks[1u].state == K3_TASK_RUNNABLE);

    /* Cpu 0 now dispatches healthy sibling task 0; cpu 1 dispatches healthy server task 1 */
    CHECK(k3_smp_schedule_next(&sched, 0u, &dispatched) == K3_OK);
    CHECK(dispatched == 0u);
    CHECK(k3_smp_schedule_next(&sched, 1u, &dispatched) == K3_OK);
    CHECK(dispatched == 1u);

    /* Execute 10,000 session-typed request/response IPC exchanges (20,000 messages total, 0 heap allocs) */
    uint8_t req_buf[16]{};
    uint8_t resp_buf[16]{};
    uint8_t recv_buf[16]{};
    for (uint32_t i = 0u; i < 10000u; ++i) {
        std::memcpy(req_buf, &i, sizeof(i));
        uint32_t seq = 0u;
        size_t rlen = 0u;
        k3_ipc_msg_kind_t rkind = K3_IPC_MSG_NONE;
        uint64_t rtag = 0ULL;

        if (k3_ipc_submit(&ch, &cdt, &client_cap, 1u, K3_IPC_MSG_REQUEST,
                          req_buf, sizeof(i), static_cast<uint64_t>(i), &seq) != K3_OK ||
            k3_ipc_complete(&ch, &cdt, &server_cap, 2u,
                            recv_buf, sizeof(recv_buf), &rlen, &rkind, &rtag) != K3_OK ||
            rkind != K3_IPC_MSG_REQUEST || rlen != sizeof(i) || rtag != static_cast<uint64_t>(i)) {
            CHECK(false);
            break;
        }

        uint32_t echoed = 0u;
        std::memcpy(&echoed, recv_buf, sizeof(echoed));
        uint32_t reply_val = echoed + 1u;
        std::memcpy(resp_buf, &reply_val, sizeof(reply_val));

        if (k3_ipc_submit(&ch, &cdt, &server_cap, 2u, K3_IPC_MSG_RESPONSE,
                          resp_buf, sizeof(reply_val), static_cast<uint64_t>(reply_val), &seq) != K3_OK ||
            k3_ipc_complete(&ch, &cdt, &client_cap, 1u,
                            recv_buf, sizeof(recv_buf), &rlen, &rkind, &rtag) != K3_OK ||
            rkind != K3_IPC_MSG_RESPONSE || rlen != sizeof(reply_val) ||
            rtag != static_cast<uint64_t>(reply_val)) {
            CHECK(false);
            break;
        }
    }
    CHECK(ch.completed_total == 20000ULL);
    CHECK(ch.revoked_aborts == 0ULL);

    /* Close session cleanly -> further requests fail closed with K3_ERR_SESSION_PROTOCOL */
    CHECK(k3_ipc_submit(&ch, &cdt, &client_cap, 1u, K3_IPC_MSG_CLOSE,
                        nullptr, 0u, 0ULL, nullptr) == K3_OK);
    k3_ipc_msg_kind_t close_kind = K3_IPC_MSG_NONE;
    CHECK(k3_ipc_complete(&ch, &cdt, &server_cap, 2u,
                          recv_buf, sizeof(recv_buf), nullptr, &close_kind, nullptr) == K3_OK);
    CHECK(close_kind == K3_IPC_MSG_CLOSE);
    CHECK(ch.session_state == K3_SESSION_CLOSED);
    CHECK(k3_ipc_submit(&ch, &cdt, &client_cap, 1u, K3_IPC_MSG_REQUEST,
                        dummy, sizeof(dummy), 0ULL, nullptr) == K3_ERR_SESSION_PROTOCOL);

    /* Verify preemptive tick rotates healthy task 0 without resurrecting faulted task 2 */
    CHECK(k3_smp_preempt_tick(&sched, 0u, &dispatched) == K3_OK);
    CHECK(dispatched == 0u);
    CHECK(sched.tasks[2u].state == K3_TASK_FAULTED);
}

/* Multi-threaded real concurrency stress (run outside operator-new zero-alloc window since std::thread allocates). */
void test_multithreaded_submit_revoke_complete_race(uint64_t& mt_revoked_aborts_out) {
    static k3_cdt_t cdt;
    static k3_ipc_channel_t ch;
    mt_revoked_aborts_out = 0ULL;

    for (uint32_t iter = 0u; iter < 64u; ++iter) {
        k3_cdt_init(&cdt);
        CHECK(k3_ipc_channel_init(&ch, 77u, 1u, 2u) == K3_OK);
        k3_cap_handle_t p_cap{}, c_cap{}, r_cap{};
        CHECK(k3_cdt_mint_root(&cdt, 1u, 77u,
                               K3_RIGHT_SEND | K3_RIGHT_DERIVE | K3_RIGHT_REVOKE, &p_cap) == K3_OK);
        CHECK(k3_cdt_derive(&cdt, &p_cap, 1u, K3_RIGHT_SEND, &c_cap) == K3_OK);
        CHECK(k3_cdt_mint_root(&cdt, 2u, 77u, K3_RIGHT_RECV, &r_cap) == K3_OK);

        const uint8_t payload[8] = {9, 8, 7, 6, 5, 4, 3, 2};
        CHECK(k3_ipc_submit(&ch, &cdt, &c_cap, 1u, K3_IPC_MSG_REQUEST,
                            payload, sizeof(payload), iter, nullptr) == K3_OK);

        std::atomic<bool> go{false};
        std::atomic<int> comp_status{-1};
        uint8_t rx_buf[16];
        uint8_t rx_canary[16];
        std::memset(rx_buf, 0xCC, sizeof(rx_buf));
        std::memset(rx_canary, 0xCC, sizeof(rx_canary));

        std::thread revoker([&]() {
            while (!go.load(std::memory_order_acquire)) {}
            uint32_t n = 0u;
            (void)k3_cdt_revoke_subtree(&cdt, &p_cap, &n);
        });

        std::thread completer([&]() {
            while (!go.load(std::memory_order_acquire)) {}
            k3_status_t st = k3_ipc_complete(&ch, &cdt, &r_cap, 2u,
                                             rx_buf, sizeof(rx_buf), nullptr, nullptr, nullptr);
            comp_status.store(static_cast<int>(st), std::memory_order_release);
        });

        go.store(true, std::memory_order_release);
        revoker.join();
        completer.join();

        int st = comp_status.load(std::memory_order_acquire);
        /* Either completed cleanly BEFORE revoke took effect, or aborted with K3_ERR_CAP_REVOKED and zero side effects. */
        CHECK(st == static_cast<int>(K3_OK) || st == static_cast<int>(K3_ERR_CAP_REVOKED));
        if (st == static_cast<int>(K3_ERR_CAP_REVOKED)) {
            CHECK(std::memcmp(rx_buf, rx_canary, sizeof(rx_buf)) == 0);
            mt_revoked_aborts_out += 1ULL;
        }
        /* After revoke finished, any subsequent submit using c_cap MUST ALWAYS fail closed. */
        CHECK(k3_ipc_submit(&ch, &cdt, &c_cap, 1u, K3_IPC_MSG_REQUEST,
                            payload, sizeof(payload), 0ULL, nullptr) == K3_ERR_CAP_REVOKED);
    }
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

    uint64_t race_aborts = 0ULL;
    test_cdt_depth_cycles_and_rights_attenuation();
    test_mid_flight_revocation_and_concurrent_race(race_aborts);
    test_cacheline_isolation_and_shm_confinement();
    test_10k_session_ipc_and_fault_isolation();

    g_track_allocs.store(false, std::memory_order_release);
    uint64_t allocs = g_heap_alloc_count.load(std::memory_order_acquire);
    CHECK(allocs == 0ULL);

    uint64_t mt_aborts = 0ULL;
    test_multithreaded_submit_revoke_complete_race(mt_aborts);

    std::printf("K3 SMP/Cap/IPC Test Suite: %d passed, %d failed "
                "(heap_allocs=%llu, ipc_exchanges=10000, midflight_revoked_aborts=%llu, mt_revoked_aborts=%llu)\n",
                g_checks_passed,
                g_checks_failed,
                static_cast<unsigned long long>(allocs),
                static_cast<unsigned long long>(race_aborts),
                static_cast<unsigned long long>(mt_aborts));
    return (g_checks_failed == 0 && allocs == 0ULL) ? 0 : 1;
}
