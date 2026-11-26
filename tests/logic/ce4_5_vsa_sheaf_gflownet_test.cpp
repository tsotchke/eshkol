/*
 * CE-4 & CE-5/6 Verification Suite:
 *   1. Native VSA/HDC: FHRR (phasor binding/unbinding/bundling), HRR (circular
 *      convolution/correlation), Fractional Power Encoding (FPE) shift-invariant
 *      RBF kernel reproduction, and associative cleanup multi-binding recovery
 *   2. Sheaf-Cohomological & GNWT Global Workspace:
 *      - Unbounded dynamic specialist registration (128 specialists)
 *      - H^0 gluing when [c] = 0
 *      - Twisted 3-chart Mobius/antipodal belief loop (H^1 != 0): naive averaging
 *        collapses to 0 while sheaf workspace detects H^1 obstruction, splits
 *        charts preserving both perspectives, and boosts Dialectic specialist
 *      - Single-overlap Cech cocycle failure negative control (H^1 != 0 detected,
 *        never silently returning 0)
 *      - GNWT refractory gating (winner cannot re-ignite immediately) + AST
 *        metacognitive surprise
 *   3. Trajectory-Balance GFlowNet:
 *      - Exact analytical TB gradients (fd_evals == 0, < 1e-7 vs central FD)
 *      - Training on multi-path diamond DAG to TV < 0.02 + per-state local TV
 *        calibration + per-terminal relative error + untrained negative control
 *   4. Predictive Coding Network (PCN):
 *      - Local symmetric-nudged Hebbian equilibrium gradient matches exact
 *        analytical backpropagation < 1e-6 (fd_evals == 0)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/gflownet_ebm.h>
#include <eshkol/core/sheaf_workspace.h>
#include <eshkol/core/vsa_hdc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_checks = 0;

static void check(bool cond, const char* msg) {
    g_checks++;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static bool near(double a, double b, double tol = 1e-7) {
    return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol;
}

static void test_vsa_fhrr_hrr_fpe_and_cleanup() {
    /* 1. FHRR exact binding inverse + multi-binding superposition cleanup recovery */
    const uint32_t D = 2048;
    std::vector<double> r1(D), r2(D), r3(D), v1(D), v2(D), v3(D), v4(D);
    check(eshkol_fhrr_random(r1.data(), D, 101), "fhrr r1");
    check(eshkol_fhrr_random(r2.data(), D, 102), "fhrr r2");
    check(eshkol_fhrr_random(r3.data(), D, 103), "fhrr r3");
    check(eshkol_fhrr_random(v1.data(), D, 201), "fhrr v1");
    check(eshkol_fhrr_random(v2.data(), D, 202), "fhrr v2");
    check(eshkol_fhrr_random(v3.data(), D, 203), "fhrr v3");
    check(eshkol_fhrr_random(v4.data(), D, 204), "fhrr v4");

    /* Exact bind/unbind identity */
    std::vector<double> b1(D), b2(D), b3(D), rec(D);
    check(eshkol_fhrr_bind(r1.data(), v1.data(), b1.data(), D), "fhrr bind r1*v1");
    check(eshkol_fhrr_unbind(b1.data(), r1.data(), rec.data(), D), "fhrr unbind");
    check(near(eshkol_fhrr_similarity(rec.data(), v1.data(), D), 1.0, 1e-12),
          "FHRR unbind(bind(r1, v1), r1) == v1 exact");
    check(std::fabs(eshkol_fhrr_similarity(r1.data(), v1.data(), D)) < 0.08,
          "random FHRR atoms are quasi-orthogonal");

    /* Superposition S = bundle(r1*v1, r2*v2, r3*v3) -> cleanup recovery for all 3 */
    check(eshkol_fhrr_bind(r2.data(), v2.data(), b2.data(), D), "fhrr bind r2*v2");
    check(eshkol_fhrr_bind(r3.data(), v3.data(), b3.data(), D), "fhrr bind r3*v3");
    const double* items[3] = {b1.data(), b2.data(), b3.data()};
    std::vector<double> superpos(D);
    check(eshkol_fhrr_bundle(items, nullptr, 3, superpos.data(), D), "fhrr bundle");

    eshkol_vsa_cleanup_t* mem = eshkol_vsa_cleanup_create(ESHKOL_VSA_FHRR, D);
    check(eshkol_vsa_cleanup_add(mem, "alpha", v1.data(), D) == 0, "cleanup add alpha");
    check(eshkol_vsa_cleanup_add(mem, "beta",  v2.data(), D) == 1, "cleanup add beta");
    check(eshkol_vsa_cleanup_add(mem, "gamma", v3.data(), D) == 2, "cleanup add gamma");
    check(eshkol_vsa_cleanup_add(mem, "delta", v4.data(), D) == 3, "cleanup add delta");

    const std::vector<double>* roles[3] = {&r1, &r2, &r3};
    const char* expected_names[3] = {"alpha", "beta", "gamma"};
    for (int i = 0; i < 3; i++) {
        check(eshkol_fhrr_unbind(superpos.data(), roles[i]->data(), rec.data(), D), "unbind role");
        eshkol_vsa_match_t m{};
        check(eshkol_vsa_cleanup_query(mem, rec.data(), D, &m) && m.found, "cleanup matched");
        check(m.atom_id == static_cast<uint32_t>(i) && std::strcmp(m.name, expected_names[i]) == 0,
              "cleanup recovers exact bound value from 3-way superposition");
        check(m.similarity > 0.35 && m.margin > 0.25, "cleanup recovery has strong margin");
    }
    eshkol_vsa_cleanup_destroy(mem);

    /* 2. Fractional Power Encoding (FPE): shift-invariant RBF kernel reproduction */
    const uint32_t D_fpe = 4096;
    const double sigma = 1.2;
    std::vector<double> ax0(D_fpe), ax1(D_fpe);
    check(eshkol_fhrr_gaussian_axis(ax0.data(), D_fpe, sigma, 9001), "gaussian axis 0");
    check(eshkol_fhrr_gaussian_axis(ax1.data(), D_fpe, sigma, 9002), "gaussian axis 1");
    const double* axes[2] = {ax0.data(), ax1.data()};

    double p_a[2] = {0.4, -0.2};
    double p_b[2] = {0.9,  0.3};
    double shift[2] = {2.3, -1.7};
    double p_as[2] = {p_a[0] + shift[0], p_a[1] + shift[1]};
    double p_bs[2] = {p_b[0] + shift[0], p_b[1] + shift[1]};
    double p_sum[2] = {p_a[0] + p_b[0], p_a[1] + p_b[1]};

    std::vector<double> fa(D_fpe), fb(D_fpe), fas(D_fpe), fbs(D_fpe), fsum(D_fpe), fbind(D_fpe);
    check(eshkol_fhrr_fpe_encode(axes, p_a, 2, fa.data(), D_fpe), "fpe(a)");
    check(eshkol_fhrr_fpe_encode(axes, p_b, 2, fb.data(), D_fpe), "fpe(b)");
    check(eshkol_fhrr_fpe_encode(axes, p_as, 2, fas.data(), D_fpe), "fpe(a+delta)");
    check(eshkol_fhrr_fpe_encode(axes, p_bs, 2, fbs.data(), D_fpe), "fpe(b+delta)");
    check(eshkol_fhrr_fpe_encode(axes, p_sum, 2, fsum.data(), D_fpe), "fpe(a+b)");
    check(eshkol_fhrr_bind(fa.data(), fb.data(), fbind.data(), D_fpe), "bind(fpe(a), fpe(b))");

    double sim_ab = eshkol_fhrr_similarity(fa.data(), fb.data(), D_fpe);
    double sim_shifted = eshkol_fhrr_similarity(fas.data(), fbs.data(), D_fpe);
    double dx = p_a[0] - p_b[0], dy = p_a[1] - p_b[1];
    double exact_rbf = std::exp(-0.5 * sigma * sigma * (dx * dx + dy * dy));
    check(near(sim_ab, sim_shifted, 1e-12), "FPE similarity is strictly shift-invariant");
    check(std::fabs(sim_ab - exact_rbf) < 0.03,
          "Gaussian FPE inner product reproduces continuous RBF kernel");
    check(near(eshkol_fhrr_similarity(fbind.data(), fsum.data(), D_fpe), 1.0, 1e-12),
          "FPE homomorphism: bind(phi(a), phi(b)) == phi(a + b)");

    /* 3. HRR real circular convolution / correlation unbinding + cleanup */
    const uint32_t Dh = 512;
    std::vector<double> hr1(Dh), hr2(Dh), hv1(Dh), hv2(Dh), hb1(Dh), hb2(Dh), hsup(Dh), hrec(Dh);
    check(eshkol_hrr_random(hr1.data(), Dh, 301), "hrr r1");
    check(eshkol_hrr_random(hr2.data(), Dh, 302), "hrr r2");
    check(eshkol_hrr_random(hv1.data(), Dh, 401), "hrr v1");
    check(eshkol_hrr_random(hv2.data(), Dh, 402), "hrr v2");
    check(eshkol_hrr_bind(hr1.data(), hv1.data(), hb1.data(), Dh), "hrr bind 1");
    check(eshkol_hrr_bind(hr2.data(), hv2.data(), hb2.data(), Dh), "hrr bind 2");
    const double* hitems[2] = {hb1.data(), hb2.data()};
    check(eshkol_hrr_bundle(hitems, nullptr, 2, true, hsup.data(), Dh), "hrr bundle");
    check(eshkol_hrr_unbind(hsup.data(), hr2.data(), hrec.data(), Dh), "hrr unbind r2");

    eshkol_vsa_cleanup_t* hmem = eshkol_vsa_cleanup_create(ESHKOL_VSA_HRR, Dh);
    eshkol_vsa_cleanup_add(hmem, "hv1", hv1.data(), Dh);
    eshkol_vsa_cleanup_add(hmem, "hv2", hv2.data(), Dh);
    eshkol_vsa_match_t hm{};
    check(eshkol_vsa_cleanup_query(hmem, hrec.data(), Dh, &hm) && hm.found && hm.atom_id == 1,
          "HRR circular correlation unbinding recovers hv2 via cleanup");
    check(hm.margin > 0.25, "HRR cleanup recovery has strong margin");
    eshkol_vsa_cleanup_destroy(hmem);

    std::printf("PASS: VSA/HDC FHRR + HRR + FPE RBF kernel (sim=%.4f vs rbf=%.4f) + cleanup\n",
                sim_ab, exact_rbf);
}

static void test_sheaf_workspace_cohomology_gnwt_ast() {
    const uint32_t d = 3;
    const double I3[9] = {1, 0, 0,  0, 1, 0,  0, 0, 1};
    const double NegI3[9] = {-1, 0, 0,  0, -1, 0,  0, 0, -1};

    /* 1. Unbounded specialist registration (128 specialists) + H^0 gluing when [c] = 0 */
    {
        eshkol_sheaf_ws_config_t cfg = eshkol_sheaf_ws_default_config();
        cfg.refractory_steps = 2;
        eshkol_sheaf_workspace_t* ws = eshkol_sheaf_ws_create(d, &cfg);
        for (uint32_t i = 0; i < 128; i++) {
            std::string nm = "spec_" + std::to_string(i);
            check(eshkol_sheaf_ws_add_specialist(ws, nm.c_str(), i == 127) == i,
                  "unbounded specialist registered");
        }
        check(eshkol_sheaf_ws_num_specialists(ws) == 128, "128 specialists active (no cap)");

        double s_compat[3] = {1.0, 2.0, -1.0};
        check(eshkol_sheaf_ws_propose(ws, 0, s_compat, d, 0.9), "propose chart 0");
        check(eshkol_sheaf_ws_propose(ws, 1, s_compat, d, 0.7), "propose chart 1");
        check(eshkol_sheaf_ws_propose(ws, 2, s_compat, d, 0.5), "propose chart 2");
        check(eshkol_sheaf_ws_add_transition(ws, 0, 1, I3, d), "rho_01 = I");
        check(eshkol_sheaf_ws_add_transition(ws, 1, 2, I3, d), "rho_12 = I");
        check(eshkol_sheaf_ws_add_transition(ws, 2, 0, I3, d), "rho_20 = I");

        eshkol_sheaf_ws_step_result_t r1{};
        check(eshkol_sheaf_ws_step(ws, &r1), "H^0 step 1 succeeded");
        check(r1.status == ESHKOL_WS_GLUED_H0, "compatible loop glues into H^0");
        check(r1.cech_holonomy_norm < 1e-12 && r1.cech_coboundary_residual < 1e-12,
              "compatible loop has zero Cech obstruction");
        check(r1.winner_id == 0, "highest-salience specialist 0 ignites first");
        check(r1.ast_surprise > 0.0, "AST metacognitive surprise > 0");
        check(eshkol_sheaf_ws_refractory_remaining(ws, 0) == 2,
              "winner 0 enters refractory period (2 steps)");

        double glued[3] = {0};
        check(eshkol_sheaf_ws_get_broadcast(ws, 0, glued, d), "glued H^0 section retrieved");
        check(near(glued[0], 1.0, 1e-12) && near(glued[1], 2.0, 1e-12) && near(glued[2], -1.0, 1e-12),
              "glued H^0 section preserves exact proposal");

        /* Next step: specialist 0 is refractory -> CANNOT re-ignite immediately; specialist 1 wins! */
        eshkol_sheaf_ws_step_result_t r2{};
        check(eshkol_sheaf_ws_step(ws, &r2), "H^0 step 2 succeeded");
        check(r2.winner_id == 1 && r2.winner_id != 0,
              "GNWT refractory gating prevents immediate re-ignition of winner 0");
        eshkol_sheaf_ws_destroy(ws);
    }

    /* 2. Twisted 3-chart Mobius/antipodal belief loop (H^1 != 0):
     *    rho_01 = +I, rho_12 = +I, rho_20 = -I => holonomy H = -I! */
    {
        eshkol_sheaf_ws_config_t cfg = eshkol_sheaf_ws_default_config();
        cfg.dialectic_boost = 2.0;
        eshkol_sheaf_workspace_t* ws = eshkol_sheaf_ws_create(d, &cfg);
        uint32_t c0 = eshkol_sheaf_ws_add_specialist(ws, "chart_0", false);
        uint32_t c1 = eshkol_sheaf_ws_add_specialist(ws, "chart_1", false);
        uint32_t c2 = eshkol_sheaf_ws_add_specialist(ws, "chart_2", false);
        uint32_t c_dial = eshkol_sheaf_ws_add_specialist(ws, "dialectic", true);

        double v[3] = {1.0, 0.5, -0.5};
        double v_norm = std::sqrt(1.0 + 0.25 + 0.25);
        check(eshkol_sheaf_ws_propose(ws, c0, v, d, 0.9), "mobius c0");
        check(eshkol_sheaf_ws_propose(ws, c1, v, d, 0.8), "mobius c1");
        check(eshkol_sheaf_ws_propose(ws, c2, v, d, 0.7), "mobius c2");
        check(eshkol_sheaf_ws_propose(ws, c_dial, v, d, 0.2), "dialectic low base salience");

        check(eshkol_sheaf_ws_add_transition(ws, c0, c1, I3, d), "rho_01 = +I");
        check(eshkol_sheaf_ws_add_transition(ws, c1, c2, I3, d), "rho_12 = +I");
        check(eshkol_sheaf_ws_add_transition(ws, c2, c0, NegI3, d), "rho_20 = -I (Mobius twist)");

        eshkol_sheaf_ws_step_result_t res{};
        check(eshkol_sheaf_ws_step(ws, &res), "Mobius loop step executed");
        check(res.status == ESHKOL_WS_H1_OBSTRUCTION,
              "Mobius 3-chart loop emits ESHKOL_WS_H1_OBSTRUCTION");
        check(near(res.cech_holonomy_norm, 2.0 * v_norm, 1e-10),
              "exact Cech holonomy norm == 2*||v|| on antipodal loop");
        check(res.naive_average_norm < 1e-12,
              "naive single-chart averaging collapses to zero on Mobius loop");
        check(res.num_split_charts == 2 &&
              near(res.split_branch_norms[0], v_norm, 1e-10) &&
              near(res.split_branch_norms[1], v_norm, 1e-10),
              "chart splitting preserves BOTH antipodal perspectives with full norm");
        check(res.winner_id == c_dial && res.winner_effective_salience > 2.0,
              "H^1 obstruction boosts Dialectic specialist to win GNWT ignition");

        double b0[3], b1[3];
        check(eshkol_sheaf_ws_get_broadcast(ws, 0, b0, d) &&
              eshkol_sheaf_ws_get_broadcast(ws, 1, b1, d),
              "both split branches retrievable");
        check(near(b0[0], -b1[0], 1e-12) && near(b0[1], -b1[1], 1e-12) && near(b0[2], -b1[2], 1e-12),
              "split charts preserve exact opposite +v and -v perspectives");
        eshkol_sheaf_ws_destroy(ws);
    }

    /* 3. Tsotchke-chan Rigor Guard: Single-overlap Cech cocycle failure negative control!
     *    Only ONE overlap edge (c0 -> c1, no cycle) where rho_01(s0) = -s1 => H^1 != 0! */
    {
        eshkol_sheaf_ws_config_t cfg = eshkol_sheaf_ws_default_config();
        cfg.dialectic_boost = 2.0;
        eshkol_sheaf_workspace_t* ws = eshkol_sheaf_ws_create(d, &cfg);
        uint32_t c0 = eshkol_sheaf_ws_add_specialist(ws, "left_chart", false);
        uint32_t c1 = eshkol_sheaf_ws_add_specialist(ws, "right_chart", false);
        uint32_t c_dial = eshkol_sheaf_ws_add_specialist(ws, "dialectic", true);

        double s_pos[3] = {1.0, 0.0, 0.0};
        double s_neg[3] = {-1.0, 0.0, 0.0};
        check(eshkol_sheaf_ws_propose(ws, c0, s_pos, d, 0.85), "single-overlap left +e1");
        check(eshkol_sheaf_ws_propose(ws, c1, s_neg, d, 0.75), "single-overlap right -e1");
        check(eshkol_sheaf_ws_propose(ws, c_dial, s_pos, d, 0.15), "dialectic base 0.15");
        check(eshkol_sheaf_ws_add_transition(ws, c0, c1, I3, d), "single overlap rho_01 = I");

        eshkol_sheaf_ws_step_result_t res{};
        check(eshkol_sheaf_ws_step(ws, &res), "single-overlap step executed");
        check(res.status == ESHKOL_WS_H1_OBSTRUCTION,
              "single-overlap cocycle failure emits ESHKOL_WS_H1_OBSTRUCTION (never silent 0)");
        check(res.cech_coboundary_residual > 1.0,
              "single-overlap coboundary residual strictly nonzero");
        check(res.naive_average_norm < 1e-12,
              "single-overlap naive average collapses to 0");
        check(res.num_split_charts == 2 &&
              near(res.split_branch_norms[0], 1.0, 1e-10) &&
              near(res.split_branch_norms[1], 1.0, 1e-10),
              "single-overlap chart splitting preserves both +e1 and -e1");
        check(res.winner_id == c_dial,
              "single-overlap obstruction boosts Dialectic specialist");
        eshkol_sheaf_ws_destroy(ws);
    }

    std::printf("PASS: Sheaf workspace (128 specialists, H^0 glue, Mobius H^1 split, single-overlap guard, GNWT refractory + AST)\n");
}

static eshkol_gfn_dag_t* build_diamond_gfn_dag() {
    /* 6-state multi-path diamond DAG:
     *   0 -> 1, 0 -> 2
     *   1 -> 3 (terminal R=4.0), 1 -> 4 (terminal R=2.0)
     *   2 -> 4 (terminal R=2.0 — shared via TWO parents 1 & 2!), 2 -> 5 (terminal R=6.0)
     * Exact Z = 4 + 2 + 6 = 12, exact targets: p(3)=4/12, p(4)=2/12, p(5)=6/12. */
    eshkol_gfn_dag_t* gfn = eshkol_gfn_create(6);
    eshkol_gfn_add_edge(gfn, 0, 1, 0.2);   /* e0 */
    eshkol_gfn_add_edge(gfn, 0, 2, -0.3);  /* e1 */
    eshkol_gfn_add_edge(gfn, 1, 3, 0.1);   /* e2 */
    eshkol_gfn_add_edge(gfn, 1, 4, -0.1);  /* e3 */
    eshkol_gfn_add_edge(gfn, 2, 4, 0.4);   /* e4 */
    eshkol_gfn_add_edge(gfn, 2, 5, -0.2);  /* e5 */
    eshkol_gfn_set_terminal_reward(gfn, 3, 4.0);
    eshkol_gfn_set_terminal_reward(gfn, 4, 2.0);
    eshkol_gfn_set_terminal_reward(gfn, 5, 6.0);
    eshkol_gfn_set_log_z(gfn, 1.5);
    return gfn;
}

static void test_gflownet_tb_and_per_state_tv() {
    eshkol_gfn_dag_t* gfn = build_diamond_gfn_dag();
    const uint32_t E = eshkol_gfn_num_edges(gfn);
    check(E == 6, "diamond DAG has 6 edges");

    /* 1. Exact analytical TB gradients vs independent central FD < 1e-7 */
    uint32_t traj[2] = {0, 3}; /* 0 -> 1 -> 4 (shared terminal with |in(4)| = 2) */
    double loss = 0.0, g_z = 0.0;
    std::vector<double> g_l(E, 0.0);
    uint64_t fd_evals = 99;
    check(eshkol_gfn_tb_loss_and_grad(gfn, traj, 2, &loss, &g_z, g_l.data(), E, &fd_evals),
          "TB loss & grad computed");
    check(fd_evals == 0, "TB gradient uses 0 finite-difference evals");

    const double h = 1e-6;
    double orig_z = eshkol_gfn_get_log_z(gfn);
    double lp = 0.0, lm = 0.0;
    eshkol_gfn_set_log_z(gfn, orig_z + h);
    eshkol_gfn_tb_loss_and_grad(gfn, traj, 2, &lp, nullptr, nullptr, 0, nullptr);
    eshkol_gfn_set_log_z(gfn, orig_z - h);
    eshkol_gfn_tb_loss_and_grad(gfn, traj, 2, &lm, nullptr, nullptr, 0, nullptr);
    eshkol_gfn_set_log_z(gfn, orig_z);
    double max_err = std::fabs(g_z - (lp - lm) / (2.0 * h));
    check(near(g_z, (lp - lm) / (2.0 * h), 1e-7), "TB grad_log_z matches central FD < 1e-7");

    for (uint32_t e = 0; e < E; e++) {
        double orig = 0.0;
        eshkol_gfn_get_edge_logit(gfn, e, &orig);
        eshkol_gfn_set_edge_logit(gfn, e, orig + h);
        eshkol_gfn_tb_loss_and_grad(gfn, traj, 2, &lp, nullptr, nullptr, 0, nullptr);
        eshkol_gfn_set_edge_logit(gfn, e, orig - h);
        eshkol_gfn_tb_loss_and_grad(gfn, traj, 2, &lm, nullptr, nullptr, 0, nullptr);
        eshkol_gfn_set_edge_logit(gfn, e, orig);
        double fd = (lp - lm) / (2.0 * h);
        max_err = std::max(max_err, std::fabs(g_l[e] - fd));
        check(near(g_l[e], fd, 1e-7), "TB grad_logit matches central FD < 1e-7");
    }

    /* 2. Negative control BEFORE training: untrained GFlowNet fails per-state TV gate! */
    double pre_state_tv[6] = {0};
    double pre_max_stv = 0.0, pre_max_rel = 0.0;
    check(eshkol_gfn_eval_per_state_tv(gfn, pre_state_tv, 6, &pre_max_stv, &pre_max_rel),
          "untrained per-state TV computed");
    check(pre_max_stv > 0.10 && pre_max_rel > 0.15,
          "negative control: untrained/miscalibrated DAG fails per-state TV calibration gate");

    /* 3. Train TB GFlowNet to exact Boltzmann target R(x)/Z */
    check(eshkol_gfn_train(gfn, 1200, 0.05, 0.05, 20261125ULL), "TB GFlowNet training succeeded");

    double probs[6] = {0}, tv = 1.0, exact_lz = 0.0;
    check(eshkol_gfn_eval_distribution(gfn, probs, 6, &tv, &exact_lz),
          "trained distribution evaluated");
    check(tv < 0.02, "trained GFlowNet global TV(p_theta, R/Z) < 0.02");
    check(std::fabs(eshkol_gfn_get_log_z(gfn) - exact_lz) < 0.05,
          "trained log_Z matches exact log(12)");

    /* Tsotchke-chan Rigor Guard: per-state local TV & per-terminal relative error */
    double post_state_tv[6] = {0};
    double post_max_stv = 1.0, post_max_rel = 1.0;
    check(eshkol_gfn_eval_per_state_tv(gfn, post_state_tv, 6, &post_max_stv, &post_max_rel),
          "trained per-state TV computed");
    check(post_max_stv < 0.02, "trained GFlowNet max per-state local TV < 0.02");
    check(post_max_rel < 0.03, "trained GFlowNet max per-terminal relative error < 0.03");

    /* Negative control AFTER training: perturb a single state's edge logit -> fails per-state gate */
    double saved_e4 = 0.0;
    eshkol_gfn_get_edge_logit(gfn, 4, &saved_e4);
    eshkol_gfn_set_edge_logit(gfn, 4, saved_e4 + 2.5);
    double bad_stv[6] = {0}, bad_max_stv = 0.0, bad_max_rel = 0.0;
    check(eshkol_gfn_eval_per_state_tv(gfn, bad_stv, 6, &bad_max_stv, &bad_max_rel),
          "miscalibrated state per-state TV computed");
    check(bad_stv[2] > 0.15 && bad_max_stv > 0.15,
          "negative control: perturbing state 2 fails per-state local TV gate at state 2");

    eshkol_gfn_destroy(gfn);
    std::printf("PASS: TB GFlowNet (max_grad_err=%.2e, tv=%.4f, max_state_tv=%.4f, max_term_rel=%.4f)\n",
                max_err, tv, post_max_stv, post_max_rel);
}

static void test_pcn_local_hebbian_vs_backprop() {
    const uint32_t dims[4] = {3, 5, 4, 2};
    eshkol_pcn_t* pcn = eshkol_pcn_create(dims, 4, 424242ULL);
    check(pcn != nullptr, "4-layer PCN created");

    const uint32_t P = eshkol_pcn_total_params(pcn);
    const double input[3] = {0.7, -0.4, 0.25};
    const double target[2] = {0.85, -0.35};

    std::vector<double> bp_grads(P, 0.0);
    double loss = 0.0;
    check(eshkol_pcn_exact_backprop_grad(pcn, input, target, bp_grads.data(), P, &loss),
          "PCN exact backprop computed");
    check(loss > 0.0, "PCN initial loss positive");

    std::vector<double> pcn_grads(P, 0.0);
    uint64_t fd_evals = 99;
    check(eshkol_pcn_local_hebbian_grad(pcn, input, target, 1e-4, 600, 0.15,
                                        pcn_grads.data(), P, &fd_evals),
          "PCN local Hebbian equilibrium grad computed");
    check(fd_evals == 0, "PCN local Hebbian grad uses 0 parameter FD evals");

    double max_err = 0.0;
    for (uint32_t i = 0; i < P; i++) {
        max_err = std::max(max_err, std::fabs(pcn_grads[i] - bp_grads[i]));
        check(near(pcn_grads[i], bp_grads[i], 1e-6),
              "PCN local Hebbian equilibrium gradient matches exact backprop < 1e-6");
    }

    eshkol_pcn_destroy(pcn);
    std::printf("PASS: PCN local Hebbian equilibrium == exact backprop (max_err=%.2e, fd_evals=0)\n",
                max_err);
}

int main() {
    test_vsa_fhrr_hrr_fpe_and_cleanup();
    test_sheaf_workspace_cohomology_gnwt_ast();
    test_gflownet_tb_and_per_state_tv();
    test_pcn_local_hebbian_vs_backprop();
    std::printf("CE-4/5 VSA, SHEAF WORKSPACE & GFLOWNET TEST: PASS (%d checks)\n", g_checks);
    return 0;
}
