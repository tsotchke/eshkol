/*
 * CE-10 Native Verification Suite:
 *   1. Z-Set Complementary Learning Systems (CLS) Catastrophic Forgetting Guard
 *   2. Differentiable STL Control Loop: NaN/Timeout Clamp & Silent-Drift Guard
 *   3. Linearizable Quorum-Fenced Cognitive Checkpoints Split-Brain Guard
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/cls_zset_memory.h"
#include "eshkol/core/stl_control_loop.h"
#include "eshkol/core/fenced_cognitive_ckpt.h"
#include "eshkol/core/cert_chain_v2.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (cond) {                                                           \
            ++g_pass;                                                         \
        } else {                                                              \
            ++g_fail;                                                         \
            std::printf("  [FAIL] %s (line %d)\n", (msg), __LINE__);          \
        }                                                                     \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double next_unit() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
    double next_range(double lo, double hi) {
        return lo + (hi - lo) * next_unit();
    }
};

/* ============================================================================
 * Guard 1: Z-Set CLS Catastrophic Forgetting Guard + Unguarded Negative Control
 * ============================================================================ */

void test_cls_catastrophic_forgetting_guard() {
    std::printf("\n=== CE-10 Part 1: Z-Set CLS Memory & Catastrophic Forgetting Guard ===\n");

    const uint32_t IN = 4, HID = 16, OUT = 2;
    const size_t N_SEM = 8;

    /* Structured semantic associations. */
    const double sem_xs[N_SEM * IN] = {
         1.0,  0.0,  0.5, -0.5,
         0.0,  1.0, -0.5,  0.5,
        -1.0,  0.0,  0.5,  0.5,
         0.0, -1.0, -0.5, -0.5,
         0.7,  0.7,  0.2, -0.2,
        -0.7,  0.7, -0.2,  0.2,
        -0.7, -0.7,  0.2,  0.2,
         0.7, -0.7, -0.2, -0.2
    };
    const double sem_ys[N_SEM * OUT] = {
         0.9, -0.3,
        -0.4,  0.8,
        -0.8, -0.5,
         0.5,  0.6,
         0.3,  0.9,
        -0.9,  0.2,
        -0.2, -0.9,
         0.8, -0.7
    };

    eshkol_cls_memory_t* cls_guarded = eshkol_cls_memory_create(IN, HID, OUT, 0xCE100001ULL);
    eshkol_cls_memory_t* cls_unguarded = eshkol_cls_memory_create(IN, HID, OUT, 0xCE100001ULL);
    CHECK(cls_guarded != nullptr && cls_unguarded != nullptr, "CLS instances created");

    CHECK(eshkol_cls_semantic_train(cls_guarded, sem_xs, sem_ys, N_SEM, 600, 0.03),
          "guarded semantic network trained");
    CHECK(eshkol_cls_semantic_train(cls_unguarded, sem_xs, sem_ys, N_SEM, 600, 0.03),
          "unguarded baseline semantic network trained");

    double base_loss = 999.0, base_grad_norm = 999.0;
    uint64_t fde = 999;
    CHECK(eshkol_cls_semantic_eval(cls_guarded, sem_xs, sem_ys, N_SEM,
                                   &base_loss, &base_grad_norm, &fde),
          "baseline semantic evaluation succeeded");
    CHECK(fde == 0, "semantic eval uses 0 FD evals");
    CHECK(base_loss < 1e-3, "baseline semantic loss < 1e-3");

    const uint64_t base_checksum = eshkol_cls_semantic_weights_checksum(cls_guarded);

    /* Generate 10,000 random adversarial episodic updates. */
    SplitMix64 rng(0xAD7E5A10ULL);
    const size_t N_ADV = 10000;
    std::vector<double> adv_xs(N_ADV * IN);
    std::vector<double> adv_ys(N_ADV * OUT);
    for (size_t i = 0; i < N_ADV * IN; ++i) adv_xs[i] = rng.next_range(-2.0, 2.0);
    for (size_t i = 0; i < N_ADV * OUT; ++i) adv_ys[i] = rng.next_range(-4.0, 4.0);

    /* Ingest all 10,000 adversarial updates into isolated hippocampal Z-set buffer. */
    for (size_t i = 0; i < N_ADV; ++i) {
        eshkol_cls_episodic_ingest(cls_guarded, 100000ULL + i,
                                   adv_xs.data() + i * IN,
                                   adv_ys.data() + i * OUT, 1);
    }
    CHECK(eshkol_cls_episodic_active_count(cls_guarded) == N_ADV,
          "all 10,000 adversarial episodes stored in hippocampal Z-set");

    double post_ingest_loss = 999.0, post_ingest_grad_norm = 999.0;
    eshkol_cls_semantic_eval(cls_guarded, sem_xs, sem_ys, N_SEM,
                             &post_ingest_loss, &post_ingest_grad_norm, &fde);
    const double ingest_loss_delta = std::fabs(post_ingest_loss - base_loss);
    const double ingest_grad_drift = std::fabs(post_ingest_grad_norm - base_grad_norm);
    const uint64_t post_ingest_checksum = eshkol_cls_semantic_weights_checksum(cls_guarded);

    /* Run guarded Fisher-protected consolidation over all 10,000 adversarial episodes. */
    size_t accepted_batches = 0, rejected_batches = 0;
    CHECK(eshkol_cls_consolidate_guarded(cls_guarded, sem_xs, sem_ys, N_SEM,
                                         0.02, 50.0, 1e-4,
                                         &accepted_batches, &rejected_batches),
          "guarded consolidation completed");
    double post_consol_loss = 999.0, post_consol_grad = 999.0;
    eshkol_cls_semantic_eval(cls_guarded, sem_xs, sem_ys, N_SEM,
                             &post_consol_loss, &post_consol_grad, &fde);

    /* Negative Control: apply the exact same 10,000 adversarial updates unguarded. */
    for (size_t i = 0; i < N_ADV; ++i) {
        eshkol_cls_unguarded_direct_step(cls_unguarded,
                                         adv_xs.data() + i * IN,
                                         adv_ys.data() + i * OUT, 0.02);
    }
    double unguarded_loss = 0.0, unguarded_grad = 0.0;
    eshkol_cls_semantic_eval(cls_unguarded, sem_xs, sem_ys, N_SEM,
                             &unguarded_loss, &unguarded_grad, &fde);

    std::printf("  [Guard 1] base_loss=%.3e | post-10k-ingest delta=%.3e, grad_drift=%.3e | "
                "post-guarded-consol=%.3e (rejected=%zu) | unguarded_neg_ctrl=%.3f (%.0fx explosion)\n",
                base_loss, ingest_loss_delta, ingest_grad_drift,
                post_consol_loss, rejected_batches,
                unguarded_loss, unguarded_loss / std::max(1e-12, post_consol_loss));

    CHECK(post_ingest_checksum == base_checksum,
          "Guard 1: semantic weights bit-identical across 10,000 episodic ingestions");
    CHECK(ingest_loss_delta == 0.0,
          "Guard 1: semantic_loss_delta == 0.0 during 10,000 episodic ingestions");
    CHECK(ingest_grad_drift < 1e-12,
          "Guard 1: semantic_grad_drift < 1e-12 during 10,000 episodic ingestions");
    CHECK(post_consol_loss < 1e-3 && rejected_batches > 0,
          "Guard 1: guarded consolidation preserves semantic loss < 1e-3 and rejects adversarial drift");
    CHECK(unguarded_loss > 0.1 && unguarded_loss > 1000.0 * post_consol_loss,
          "Guard 1 Negative Control: unguarded direct updates trigger catastrophic forgetting");

    eshkol_cls_memory_destroy(cls_guarded);
    eshkol_cls_memory_destroy(cls_unguarded);
}

/* ============================================================================
 * Guard 2: STL Control Loop, NaN/Timeout Clamp & Silent-Drift Guard
 * ============================================================================ */

void test_stl_control_loop_guards() {
    std::printf("\n=== CE-10 Part 2: Differentiable STL Control Loop & Safety Watchdogs ===\n");

    const uint32_t D = 2, M = 2;
    const double A[4] = { 0.95, 0.10, -0.05, 0.92 };
    const double B[4] = { 0.20, 0.00,  0.00, 0.20 };
    const double x_ref[2] = { 0.0, 0.0 };
    const double tube_eps = 0.50;
    const double u_max = 2.0;
    const double beta = 8.0;
    const double max_delay = 0.05;

    eshkol_stl_controller_t* ctrl = eshkol_stl_controller_create(
        D, M, A, B, x_ref, tube_eps, u_max, beta, max_delay);
    CHECK(ctrl != nullptr, "STL controller created");

    /* 2a. Verify exact analytical STL gradient against external central FD (< 1e-7). */
    const uint32_t H = 5;
    const double x0_grad[2] = { 0.32, -0.21 };
    std::vector<double> u_seq(H * M, 0.0);
    for (size_t i = 0; i < u_seq.size(); ++i) u_seq[i] = 0.15 * static_cast<double>(i % 3) - 0.10;
    std::vector<double> grad_u(H * M, 0.0);
    double rho_g = 0.0, rho_f = 0.0;
    uint64_t fde = 999;
    CHECK(eshkol_stl_eval_robustness_grad(ctrl, x0_grad, u_seq.data(), H,
                                          &rho_g, &rho_f, grad_u.data(), &fde),
          "STL robustness + analytical grad succeeded");
    CHECK(fde == 0, "STL robustness grad uses 0 internal FD evals");

    double max_stl_grad_err = 0.0;
    const double eps = 1e-6;
    for (size_t k = 0; k < u_seq.size(); ++k) {
        double orig = u_seq[k];
        double rp = 0.0, rm = 0.0;
        u_seq[k] = orig + eps;
        eshkol_stl_eval_robustness_grad(ctrl, x0_grad, u_seq.data(), H, &rp, nullptr, nullptr, nullptr);
        u_seq[k] = orig - eps;
        eshkol_stl_eval_robustness_grad(ctrl, x0_grad, u_seq.data(), H, &rm, nullptr, nullptr, nullptr);
        u_seq[k] = orig;
        double fd = (rp - rm) / (2.0 * eps);
        double err = std::fabs(grad_u[k] - fd);
        if (err > max_stl_grad_err) max_stl_grad_err = err;
    }

    /* 2b. Nominal closed-loop control keeps plant inside tube and drives toward x_ref. */
    double x_cur[2] = { 0.42, -0.25 }; /* ||x|| = 0.489 < 0.50 */
    double min_margin = 1e9;
    uint64_t total_fd = 0;
    for (int step = 0; step < 25; ++step) {
        double u_cmd[2] = { 0.0, 0.0 };
        uint64_t step_fd = 999;
        eshkol_stl_status_t st = eshkol_stl_step(
            ctrl, x_cur, 0.01, false, 0.0, 5, 15, 0.35, u_cmd, &step_fd);
        CHECK(st == ESHKOL_STL_OK, "nominal STL step returned ESHKOL_STL_OK");
        total_fd += step_fd;
        CHECK(std::isfinite(u_cmd[0]) && std::isfinite(u_cmd[1]) &&
              std::fabs(u_cmd[0]) <= u_max && std::fabs(u_cmd[1]) <= u_max,
              "nominal actuator outputs finite and within clamp");
        eshkol_stl_plant_step(ctrl, x_cur, u_cmd, x_cur);
        double margin = eshkol_stl_physical_margin(ctrl, x_cur);
        if (margin < min_margin) min_margin = margin;
    }
    double final_norm = std::sqrt(x_cur[0] * x_cur[0] + x_cur[1] * x_cur[1]);
    double final_margin = eshkol_stl_physical_margin(ctrl, x_cur);

    std::printf("  [Guard 2 Nominal] stl_grad_err=%.3e (< 1e-7), fd_evals=%llu, "
                "min_margin=%.4f (> 0), final_norm=%.4f (< 0.10)\n",
                max_stl_grad_err, (unsigned long long)total_fd, min_margin, final_norm);
    CHECK(max_stl_grad_err < 1e-7, "Guard 2: STL analytical gradient matches FD < 1e-7");
    CHECK(total_fd == 0, "Guard 2: 0 internal FD evals across closed-loop rollout");
    CHECK(min_margin > 0.0 && final_norm < 0.10 && final_margin > 0.40,
          "Guard 2: nominal STL controller maintains tube invariance and converges");

    /* 2c. Negative Control A: NaN, +Inf, -Inf, infinite delay, and stale delay. */
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double pinf = std::numeric_limits<double>::infinity();
    const double ninf = -std::numeric_limits<double>::infinity();

    struct FaultCase { double s0, s1, dt; const char* name; };
    const FaultCase faults[5] = {
        { qnan,  0.10, 0.01, "NaN sensor" },
        { 0.10,  pinf, 0.01, "+Inf sensor" },
        { ninf,  0.00, 0.01, "-Inf sensor" },
        { 0.10,  0.10, pinf, "+Inf sensor delay" },
        { 0.10,  0.10, 0.50, "stale sensor delay > max_delay" }
    };
    int non_finite_actuator_leaks = 0;
    for (const auto& fc : faults) {
        double s_in[2] = { fc.s0, fc.s1 };
        double u_out[2] = { 99.0, -99.0 };
        eshkol_stl_status_t st = eshkol_stl_step(
            ctrl, s_in, fc.dt, false, 0.0, 5, 10, 0.3, u_out, nullptr);
        CHECK(st == ESHKOL_STL_ERR_SENSOR_FAULT, fc.name);
        if (!std::isfinite(u_out[0]) || !std::isfinite(u_out[1])) ++non_finite_actuator_leaks;
        CHECK(u_out[0] == 0.0 && u_out[1] == 0.0, "fault clamped actuator output to safe 0.0");
    }
    std::printf("  [Guard 2 Neg-A] NaN/Inf/timeout faults=%llu/5, actuator_non_finite_leaks=%d\n",
                (unsigned long long)eshkol_stl_sensor_fault_count(ctrl),
                non_finite_actuator_leaks);
    CHECK(non_finite_actuator_leaks == 0 && eshkol_stl_sensor_fault_count(ctrl) == 5,
          "Guard 2 Neg-A: zero NaN/Inf values ever reach actuators across all 5 fault modes");

    /* 2d. Negative Control B: Silent Drift (surrogate strictly decreases while physical state breaches tube). */
    const double drift_states[4][2] = {
        { 0.20, 0.00 }, /* margin = +0.30, surrogate = 1.00 */
        { 0.38, 0.00 }, /* margin = +0.12, surrogate = 0.70 */
        { 0.56, 0.00 }, /* margin = -0.06 (breach!), surrogate = 0.40 -> HALT! */
        { 0.75, 0.00 }  /* margin = -0.25 (breach!), surrogate = 0.15 -> HALT! */
    };
    const double surrogate_losses[4] = { 1.00, 0.70, 0.40, 0.15 };
    eshkol_stl_status_t drift_statuses[4];
    double drift_u_at_halt[2] = { 99.0, 99.0 };
    for (int i = 0; i < 4; ++i) {
        double u_cmd[2] = { 1.0, 1.0 };
        drift_statuses[i] = eshkol_stl_step(
            ctrl, drift_states[i], 0.01, true, surrogate_losses[i], 5, 5, 0.2, u_cmd, nullptr);
        if (i == 2) {
            drift_u_at_halt[0] = u_cmd[0];
            drift_u_at_halt[1] = u_cmd[1];
        }
    }
    std::printf("  [Guard 2 Neg-B] Silent-drift statuses=[%d, %d, %d, %d] (expected 3=SILENT_DRIFT at breach), "
                "halted_u=[%.1f, %.1f], drift_halts=%llu\n",
                (int)drift_statuses[0], (int)drift_statuses[1],
                (int)drift_statuses[2], (int)drift_statuses[3],
                drift_u_at_halt[0], drift_u_at_halt[1],
                (unsigned long long)eshkol_stl_silent_drift_count(ctrl));
    CHECK(drift_statuses[0] == ESHKOL_STL_OK && drift_statuses[1] == ESHKOL_STL_OK,
          "Guard 2 Neg-B: in-tube steps proceed normally");
    CHECK(drift_statuses[2] == ESHKOL_STL_ERR_SILENT_DRIFT &&
          drift_statuses[3] == ESHKOL_STL_ERR_SILENT_DRIFT,
          "Guard 2 Neg-B: silent drift caught with ESHKOL_STL_ERR_SILENT_DRIFT");
    CHECK(drift_u_at_halt[0] == 0.0 && drift_u_at_halt[1] == 0.0,
          "Guard 2 Neg-B: silent-drift halt clamps actuators to safe 0.0");

    eshkol_stl_controller_destroy(ctrl);
}

/* ============================================================================
 * Guard 3: Linearizable Quorum-Fenced Cognitive Checkpoints Split-Brain Guard
 * ============================================================================ */

void test_fenced_checkpoint_linearizability() {
    std::printf("\n=== CE-10 Part 3: Quorum-Fenced Cognitive Checkpoints (Userspace Transport) ===\n");

    auto propose_str = [](eshkol_fenced_ckpt_cluster_t* c, uint32_t leader,
                          uint64_t epoch, const char* text) {
        return eshkol_ckpt_cluster_propose(
            c, leader, epoch,
            reinterpret_cast<const uint8_t*>(text), std::strlen(text));
    };

    /* --- Scenario A: Asymmetric Partition 3 vs 2 on 5-node cluster --- */
    eshkol_fenced_ckpt_cluster_t* c5 = eshkol_ckpt_cluster_create(5);
    CHECK(c5 != nullptr && eshkol_ckpt_cluster_quorum_size(c5) == 3,
          "5-node cluster created with strict majority quorum == 3");

    CHECK(propose_str(c5, 0, 1, "ckpt-v1-baseline") == ESHKOL_CKPT_OK,
          "Scenario A: baseline epoch 1 committed on all 5 nodes");
    CHECK(eshkol_ckpt_cluster_all_converged(c5), "Scenario A: all 5 nodes converged at epoch 1");

    /* Partition 3 vs 2: majority {0, 1, 2} vs minority {3, 4}. */
    CHECK(eshkol_ckpt_cluster_partition(c5, 3), "Scenario A: partitioned 3 vs 2");

    eshkol_ckpt_status_t st_min_e2 = propose_str(c5, 3, 2, "minority-rogue-state");
    eshkol_ckpt_status_t st_maj_e2 = propose_str(c5, 0, 2, "majority-truth-state");
    eshkol_ckpt_status_t st_min_e3 = propose_str(c5, 4, 3, "minority-higher-epoch");

    CHECK(st_min_e2 == ESHKOL_CKPT_ERR_NO_QUORUM,
          "Scenario A: minority (2/5) rejected fail-closed with ESHKOL_CKPT_ERR_NO_QUORUM");
    CHECK(st_maj_e2 == ESHKOL_CKPT_OK,
          "Scenario A: majority (3/5) committed epoch 2 cleanly");
    CHECK(st_min_e3 == ESHKOL_CKPT_ERR_NO_QUORUM,
          "Scenario A: minority (2/5) at higher epoch 3 still rejected fail-closed");

    /* Heal partition -> all 5 nodes MUST converge to majority-truth-state with 0 forks. */
    CHECK(eshkol_ckpt_cluster_heal(c5), "Scenario A: partition healed");
    CHECK(eshkol_ckpt_cluster_all_converged(c5),
          "Scenario A: all 5 nodes bit-identically converged after heal");

    uint8_t expected_maj_sha[32];
    const char maj_txt[] = "majority-truth-state";
    eshkol_cert_sha256(reinterpret_cast<const uint8_t*>(maj_txt), std::strlen(maj_txt), expected_maj_sha);

    bool all5_match_maj = true;
    for (uint32_t i = 0; i < 5; ++i) {
        uint64_t ce = 0, pe = 0;
        uint8_t sha[32];
        size_t plen = 0;
        eshkol_ckpt_node_state(c5, i, &ce, &pe, sha, &plen);
        if (ce != 2 || std::memcmp(sha, expected_maj_sha, 32) != 0) all5_match_maj = false;
    }
    CHECK(all5_match_maj, "Scenario A: all 5 nodes hold exact majority epoch 2 SHA-256");
    CHECK(eshkol_ckpt_cluster_fork_count(c5) == 0, "Scenario A: 0 split-brain forks");

    /* --- Negative Control C: Stale Leader Fencing --- */
    CHECK(propose_str(c5, 1, 3, "ckpt-v3-new-leader") == ESHKOL_CKPT_OK,
          "Neg-C: new leader committed epoch 3");
    eshkol_ckpt_status_t st_stale_e2 = propose_str(c5, 0, 2, "stale-leader-overwrite");
    eshkol_ckpt_status_t st_stale_e3 = propose_str(c5, 0, 3, "stale-leader-same-epoch");

    uint8_t expected_v3_sha[32];
    const char v3_txt[] = "ckpt-v3-new-leader";
    eshkol_cert_sha256(reinterpret_cast<const uint8_t*>(v3_txt), std::strlen(v3_txt), expected_v3_sha);
    uint64_t c5_e = 0; uint8_t c5_sha[32];
    eshkol_ckpt_node_state(c5, 0, &c5_e, nullptr, c5_sha, nullptr);

    std::printf("  [Guard 3 Scen-A & Neg-C] 3v2: min=%d, maj=%d, all5_match=%d | "
                "stale_leader(e2=%d, e3=%d, expected 3=FENCED), forks=%llu\n",
                (int)st_min_e2, (int)st_maj_e2, (int)all5_match_maj,
                (int)st_stale_e2, (int)st_stale_e3,
                (unsigned long long)eshkol_ckpt_cluster_fork_count(c5));

    CHECK(st_stale_e2 == ESHKOL_CKPT_ERR_FENCED_SPLIT_BRAIN &&
          st_stale_e3 == ESHKOL_CKPT_ERR_FENCED_SPLIT_BRAIN,
          "Neg-C: stale leader rejected with ESHKOL_CKPT_ERR_FENCED_SPLIT_BRAIN");
    CHECK(c5_e == 3 && std::memcmp(c5_sha, expected_v3_sha, 32) == 0 &&
          eshkol_ckpt_cluster_all_converged(c5) && eshkol_ckpt_cluster_fork_count(c5) == 0,
          "Neg-C: committed epoch 3 state remains bit-identical with 0 forks");

    eshkol_ckpt_cluster_destroy(c5);

    /* --- Scenario B: Symmetric 2 vs 2 Partition on 4-node cluster (NO Quorum) --- */
    eshkol_fenced_ckpt_cluster_t* c4 = eshkol_ckpt_cluster_create(4);
    CHECK(c4 != nullptr && eshkol_ckpt_cluster_quorum_size(c4) == 3,
          "4-node cluster created with strict majority quorum == 3");

    const char c4_base_txt[] = "ckpt-4node-v1";
    uint8_t expected_c4_sha[32];
    eshkol_cert_sha256(reinterpret_cast<const uint8_t*>(c4_base_txt),
                       std::strlen(c4_base_txt), expected_c4_sha);

    CHECK(propose_str(c4, 0, 1, c4_base_txt) == ESHKOL_CKPT_OK,
          "Scenario B: baseline epoch 1 committed on all 4 nodes");

    /* Symmetric 2 vs 2 partition: {0, 1} vs {2, 3}. Neither side has quorum 3! */
    CHECK(eshkol_ckpt_cluster_partition(c4, 2), "Scenario B: partitioned symmetrically 2 vs 2");
    eshkol_ckpt_status_t st_left = propose_str(c4, 0, 2, "left-half-proposal");
    eshkol_ckpt_status_t st_right = propose_str(c4, 2, 2, "right-half-proposal");

    CHECK(st_left == ESHKOL_CKPT_ERR_NO_QUORUM,
          "Scenario B: left 2/4 half rejected fail-closed with ESHKOL_CKPT_ERR_NO_QUORUM");
    CHECK(st_right == ESHKOL_CKPT_ERR_NO_QUORUM,
          "Scenario B: right 2/4 half rejected fail-closed with ESHKOL_CKPT_ERR_NO_QUORUM");

    CHECK(eshkol_ckpt_cluster_heal(c4), "Scenario B: 2 vs 2 partition healed");

    bool all4_preserved = true;
    for (uint32_t i = 0; i < 4; ++i) {
        uint64_t ce = 0; uint8_t sha[32];
        eshkol_ckpt_node_state(c4, i, &ce, nullptr, sha, nullptr);
        if (ce != 1 || std::memcmp(sha, expected_c4_sha, 32) != 0) all4_preserved = false;
    }

    std::printf("  [Guard 3 Scen-B] 2v2 symmetric partition: left=%d, right=%d (both expected 2=NO_QUORUM), "
                "all4_preserved_v1=%d, converged=%d, forks=%llu\n",
                (int)st_left, (int)st_right, (int)all4_preserved,
                (int)eshkol_ckpt_cluster_all_converged(c4),
                (unsigned long long)eshkol_ckpt_cluster_fork_count(c4));

    CHECK(all4_preserved && eshkol_ckpt_cluster_all_converged(c4),
          "Scenario B: all 4 nodes preserve exact prior epoch 1 checkpoint");
    CHECK(eshkol_ckpt_cluster_fork_count(c4) == 0,
          "Scenario B: 0 split-brain forks under symmetric 2 vs 2 partition");

    eshkol_ckpt_cluster_destroy(c4);
}

} /* namespace */

int main() {
    test_cls_catastrophic_forgetting_guard();
    test_stl_control_loop_guards();
    test_fenced_checkpoint_linearizability();

    std::printf("\nCE-10 Summary: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
