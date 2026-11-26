/*
 * CE-7 Verification Suite:
 *   1. 100,000-tick Resident Flat-RSS Continuous-Cognition Soak (waking-to-sleep
 *      arena evacuation & poison rotation): exact steady_state_growth == 0 and
 *      post_warmup_total_growth == 0 from warmup tick 100 through tick 100,000
 *   2. Holomorphic Equilibrium Propagation (hEP): 4-point Cauchy-Fourier circle
 *      matches exact analytical backpropagation < 1e-7 (fd_evals == 0), plus
 *      negative control proving real-only large-beta nudge has O(beta) bias
 *      while holomorphic K=4 cancels O(beta), O(beta^2), O(beta^3) bias
 *   3. Wasserstein-Fisher-Rao (WFR) Particle Flow: recovers exact mode mass
 *      balance (< 0.01 error) across a 32-nat bimodal energy barrier where pure
 *      Wasserstein transport remains permanently trapped (error > 0.50)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/holomorphic_wfr.h>
#include <eshkol/core/resident_cognition.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_checks = 0;

static void check(bool cond, const char* msg) {
    g_checks++;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void test_resident_100k_flat_rss_soak() {
    eshkol_resident_cog_config_t cfg = eshkol_resident_cog_default_config();
    cfg.evac_interval_ticks = 16;
    cfg.poison_rotate_evacs = 256;
    cfg.warmup_ticks = 160; /* exact multiple of evac_interval_ticks=16 */
    cfg.h1_inject_period = 1000;
    cfg.fhrr_dim = 64;

    eshkol_resident_cog_session_t* sess = eshkol_resident_cog_create(&cfg);
    check(sess != nullptr, "resident cognition session created");

    const uint64_t N_TICKS = 100000ULL;
    auto t0 = std::chrono::steady_clock::now();

    /* Run warmup phase (160 ticks) to establish steady-state OALR allocator baseline */
    eshkol_resident_cog_stats_t warm{};
    check(eshkol_resident_cog_run(sess, cfg.warmup_ticks, &warm), "warmup ticks succeeded");
    check(warm.warmup_peak_bytes > 0 && warm.live_bytes > 0, "warmup arena telemetry active");
    const size_t base_live = warm.live_bytes;
    const size_t base_peak = warm.warmup_peak_bytes;
    const size_t base_total = warm.warmup_total_arena_bytes;

    /* Rigor Guard 1: Checkpointed OALR allocator telemetry every 10k ticks (10k..100k) */
    eshkol_resident_cog_stats_t st{};
    uint64_t done_ticks = cfg.warmup_ticks;
    for (uint64_t target_tick = 10000ULL; target_tick <= N_TICKS; target_tick += 10000ULL) {
        uint64_t delta = target_tick - done_ticks;
        check(eshkol_resident_cog_run(sess, delta, &st), "10k-tick checkpoint segment succeeded");
        done_ticks = target_tick;

        std::printf("  [OALR ckpt tick=%6llu] live=%zu B  peak=%zu B  total=%zu B  evacs=%llu  rot=%llu\n",
                    static_cast<unsigned long long>(st.ticks),
                    st.live_bytes, st.peak_bytes, st.total_arena_bytes,
                    static_cast<unsigned long long>(st.evacuations),
                    static_cast<unsigned long long>(st.scratch_rotations));

        check(st.ticks == target_tick, "checkpoint tick count exact");
        check(st.live_bytes == base_live,
              "OALR checkpoint live_bytes == warmup baseline (zero drift/oscillating leak)");
        check(st.peak_bytes == base_peak,
              "OALR checkpoint peak_bytes == warmup high-water mark (exact flat RSS)");
        check(st.total_arena_bytes == base_total,
              "OALR checkpoint total_arena_bytes == warmup total (zero arena block creep)");
        check(st.steady_state_growth == 0 && st.post_warmup_total_growth == 0,
              "OALR checkpoint zero post-warmup growth");
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    check(st.ticks == N_TICKS, "exact 100,000 ticks executed");
    check(st.evacuations == N_TICKS / cfg.evac_interval_ticks,
          "all waking-to-sleep evacuations executed");
    check(st.scratch_rotations >= 20, "periodic scratch-arena rotations executed (poison-safe)");
    check(st.h1_obstructions == N_TICKS / cfg.h1_inject_period,
          "every injected H^1 episode detected cleanly");
    check(st.cleanup_hits == N_TICKS, "FHRR cleanup recovered exact atom on all 100,000 ticks");
    check(std::isfinite(st.last_bethe_free_energy), "Bethe free energy finite after 100k ticks");
    check(st.last_nars_confidence > 0.0 && st.last_nars_confidence < 1.0,
          "NARS consolidated confidence remains strictly in (0, 1) after 100k ticks");

    eshkol_resident_cog_destroy(sess);
    std::printf("PASS: 100k-tick resident flat-RSS soak (%.1f ms, 10x10k ckpts flat, evacs=%llu, h1=%llu, live=%zu B, peak=%zu B, growth=0 B)\n",
                ms,
                static_cast<unsigned long long>(st.evacuations),
                static_cast<unsigned long long>(st.h1_obstructions),
                st.live_bytes,
                st.peak_bytes);
}

static void test_holomorphic_equilibrium_propagation() {
    const uint32_t dims[4] = {3, 5, 4, 2};
    eshkol_hep_net_t* net = eshkol_hep_create(dims, 4, 20261125ULL);
    check(net != nullptr, "4-layer holomorphic EP network created");

    const uint32_t P = eshkol_hep_total_params(net);
    const double input[3] = {0.65, -0.45, 0.30};
    const double target[2] = {0.90, -0.55};

    std::vector<double> bp_grads(P, 0.0);
    double loss = 0.0;
    check(eshkol_hep_exact_backprop_grad(net, input, target, bp_grads.data(), P, &loss),
          "exact analytical backprop computed");
    check(loss > 0.0, "hEP test loss strictly positive");

    /* 1. High-accuracy 4-point holomorphic circle at radius = 0.02 (< 1e-7 vs exact BP!) */
    std::vector<double> hep_small(P, 0.0);
    uint64_t fd_evals = 99;
    check(eshkol_hep_fourier_grad(net, input, target, 0.02, 4, 600, 0.15,
                                  hep_small.data(), P, &fd_evals),
          "hEP K=4 r=0.02 computed");
    check(fd_evals == 0, "hEP uses 0 finite-difference evaluations");

    double max_err_small = 0.0;
    for (uint32_t i = 0; i < P; i++) {
        max_err_small = std::max(max_err_small, std::fabs(hep_small[i] - bp_grads[i]));
        check(std::fabs(hep_small[i] - bp_grads[i]) < 1e-7,
              "hEP K=4 r=0.02 matches exact analytical backprop < 1e-7");
    }

    /* 2. Negative control at large radius r = 0.25:
     *    Compare K=1 (real-only positive nudge, O(r) bias) vs
     *    K=2 (real symmetric, O(r^2) bias) vs
     *    K=4 (holomorphic 4-point Fourier, O(r^4) bias)! */
    const double large_r = 0.25;
    std::vector<double> g_k1(P, 0.0), g_k2(P, 0.0), g_k4(P, 0.0);
    check(eshkol_hep_fourier_grad(net, input, target, large_r, 1, 800, 0.12, g_k1.data(), P, nullptr),
          "real-only K=1 large-beta nudge");
    check(eshkol_hep_fourier_grad(net, input, target, large_r, 2, 800, 0.12, g_k2.data(), P, nullptr),
          "symmetric real K=2 large-beta nudge");
    check(eshkol_hep_fourier_grad(net, input, target, large_r, 4, 800, 0.12, g_k4.data(), P, nullptr),
          "holomorphic K=4 large-beta nudge");

    double err_k1 = 0.0, err_k2 = 0.0, err_k4 = 0.0;
    for (uint32_t i = 0; i < P; i++) {
        err_k1 = std::max(err_k1, std::fabs(g_k1[i] - bp_grads[i]));
        err_k2 = std::max(err_k2, std::fabs(g_k2[i] - bp_grads[i]));
        err_k4 = std::max(err_k4, std::fabs(g_k4[i] - bp_grads[i]));
    }
    check(err_k1 > 1e-3, "negative control: real-only K=1 large-beta nudge exhibits O(beta) bias");
    check(err_k2 < err_k1 * 0.2, "symmetric K=2 cancels O(beta) bias");
    check(err_k4 < err_k2 * 0.05 && err_k4 < 1e-5,
          "holomorphic K=4 cancels O(beta), O(beta^2), O(beta^3) bias even at r=0.25");

    eshkol_hep_destroy(net);
    std::printf("PASS: Holomorphic EP (r=0.02 K=4 err=%.2e; r=0.25 bias: K1=%.2e > K2=%.2e > K4=%.2e, fd=0)\n",
                max_err_small, err_k1, err_k2, err_k4);

    /* 3. Rigor Guard 2: Non-Hermitian Complex Weight Matrix Wirtinger Derivative Check!
     *    Explicitly non-Hermitian (W_rec != W_rec^H) AND non-symmetric (W_rec != W_rec^T). */
    const uint32_t d = 3, m = 2;
    double w_rec_re[9] = {
         0.12, -0.18,  0.07,
         0.15, -0.09, -0.14,
        -0.11,  0.16,  0.08
    };
    double w_rec_im[9] = {
         0.09,  0.14, -0.06,
         0.05, -0.13,  0.11,
         0.12, -0.08,  0.04
    };
    /* Explicitly assert non-Hermitian (W[0,1] != conj(W[1,0])) and non-symmetric */
    check(std::fabs(w_rec_re[1] - w_rec_re[3]) > 0.1 &&
          std::fabs(w_rec_im[1] + w_rec_im[3]) > 0.1,
          "W_rec is strictly non-Hermitian and non-symmetric");

    const double w_out_re[6] = { 0.40, -0.25,  0.30, -0.20,  0.35, -0.15 };
    const double w_out_im[6] = { 0.15,  0.22, -0.18,  0.28, -0.12,  0.20 };
    const double u_re[3] = { 0.35, -0.25, 0.20 };
    const double u_im[3] = { -0.15, 0.20, -0.10 };
    const double y_re[2] = { 0.70, -0.40 };
    const double y_im[2] = { -0.30, 0.50 };

    double g_re[9] = {0}, g_im[9] = {0};
    uint64_t nh_fd = 99;
    check(eshkol_hep_nonhermitian_wirtinger_grad(
              d, m, w_rec_re, w_rec_im, w_out_re, w_out_im,
              u_re, u_im, y_re, y_im,
              0.02, 4, 600, 0.20, g_re, g_im, &nh_fd),
          "non-Hermitian complex hEP Wirtinger gradient succeeded");
    check(nh_fd == 0, "non-Hermitian hEP uses 0 finite-difference evaluations");

    const double h_fd = 1e-6;
    double max_abs_re = 0.0, max_abs_im = 0.0, max_rel_err = 0.0;
    for (uint32_t idx = 0; idx < 9; idx++) {
        double orig_r = w_rec_re[idx], lp = 0.0, lm = 0.0;
        w_rec_re[idx] = orig_r + h_fd;
        check(eshkol_hep_nonhermitian_eval_loss(d, m, w_rec_re, w_rec_im, w_out_re, w_out_im,
                                                u_re, u_im, y_re, y_im, 600, &lp), "FD Re+");
        w_rec_re[idx] = orig_r - h_fd;
        check(eshkol_hep_nonhermitian_eval_loss(d, m, w_rec_re, w_rec_im, w_out_re, w_out_im,
                                                u_re, u_im, y_re, y_im, 600, &lm), "FD Re-");
        w_rec_re[idx] = orig_r;
        double fd_re = (lp - lm) / (2.0 * h_fd);
        double er_re = std::fabs(g_re[idx] - fd_re);
        double rel_re = er_re / std::max(1.0, std::fabs(fd_re));
        max_abs_re = std::max(max_abs_re, er_re);
        max_rel_err = std::max(max_rel_err, rel_re);
        check(rel_re < 1e-6, "non-Hermitian Wirtinger dL/dRe(W) matches central FD < 1e-6");

        double orig_i = w_rec_im[idx];
        w_rec_im[idx] = orig_i + h_fd;
        check(eshkol_hep_nonhermitian_eval_loss(d, m, w_rec_re, w_rec_im, w_out_re, w_out_im,
                                                u_re, u_im, y_re, y_im, 600, &lp), "FD Im+");
        w_rec_im[idx] = orig_i - h_fd;
        check(eshkol_hep_nonhermitian_eval_loss(d, m, w_rec_re, w_rec_im, w_out_re, w_out_im,
                                                u_re, u_im, y_re, y_im, 600, &lm), "FD Im-");
        w_rec_im[idx] = orig_i;
        double fd_im = (lp - lm) / (2.0 * h_fd);
        double er_im = std::fabs(g_im[idx] - fd_im);
        double rel_im = er_im / std::max(1.0, std::fabs(fd_im));
        max_abs_im = std::max(max_abs_im, er_im);
        max_rel_err = std::max(max_rel_err, rel_im);
        check(rel_im < 1e-6, "non-Hermitian Wirtinger dL/dIm(W) matches central FD < 1e-6");
    }

    std::printf("PASS: Non-Hermitian complex hEP Wirtinger partials (max_re_err=%.2e, max_im_err=%.2e, max_rel=%.2e, fd=0)\n",
                max_abs_re, max_abs_im, max_rel_err);
}

static void test_wfr_particle_flow_bimodal() {
    /* Well-separated bimodal Gaussian mixture (modes at -4 and +4, sigma=0.5 => 32-nat barrier!)
     * Target weights: mode 1 = 0.75, mode 2 = 0.25.
     * Initial weights: severely imbalanced mode 1 = 0.20, mode 2 = 0.80. */
    eshkol_wfr_bimodal_target_t tgt{};
    tgt.mean1 = -4.0;
    tgt.mean2 =  4.0;
    tgt.sigma1 = 0.5;
    tgt.sigma2 = 0.5;
    tgt.weight1 = 0.75;

    /* 1. Pure Wasserstein baseline (eta_fisher_rao = 0.0): trapped by 32-nat energy barrier! */
    eshkol_wfr_config_t w_only{};
    w_only.num_particles = 120;
    w_only.init_mode1_fraction = 0.20;
    w_only.num_steps = 250;
    w_only.eta_wasserstein = 0.02;
    w_only.eta_fisher_rao = 0.0;
    w_only.kde_bandwidth = 0.15;
    w_only.seed = 7771ULL;

    eshkol_wfr_result_t r_wonly{};
    check(eshkol_wfr_run_bimodal(&tgt, &w_only, &r_wonly) && r_wonly.valid,
          "pure Wasserstein run completed");
    check(r_wonly.mode1_mass_error > 0.50,
          "negative control: pure Wasserstein transport stays trapped by 32-nat barrier (err > 0.50)");

    /* 2. Full Wasserstein-Fisher-Rao (WFR) particle flow: teleports mass through barrier! */
    eshkol_wfr_config_t wfr = w_only;
    wfr.eta_fisher_rao = 0.08;

    eshkol_wfr_result_t r_wfr{};
    check(eshkol_wfr_run_bimodal(&tgt, &wfr, &r_wfr) && r_wfr.valid,
          "WFR particle flow completed");
    check(r_wfr.mode1_mass_error < 0.01,
          "WFR particle flow recovers exact mode mass balance (< 0.01 error)");
    check(std::fabs(r_wfr.mode1_mean - tgt.mean1) < 0.15 &&
          std::fabs(r_wfr.mode2_mean - tgt.mean2) < 0.15,
          "WFR preserves accurate mode centers");

    std::printf("PASS: WFR particle flow (target=0.750: WFR=%.4f err=%.4f vs pure-W=%.4f trapped err=%.4f)\n",
                r_wfr.mode1_mass, r_wfr.mode1_mass_error,
                r_wonly.mode1_mass, r_wonly.mode1_mass_error);
}

int main() {
    test_resident_100k_flat_rss_soak();
    test_holomorphic_equilibrium_propagation();
    test_wfr_particle_flow_bimodal();
    std::printf("CE-7 RESIDENT SOAK, HOLOMORPHIC EP & WFR TEST: PASS (%d checks)\n", g_checks);
    return 0;
}
