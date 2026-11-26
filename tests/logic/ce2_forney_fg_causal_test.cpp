/*
 * CE-2 Differentiable Forney Factor Graphs, Certified Free-Energy Enclosures,
 * Product-Manifold Soft-Unify, 12-Semiring Provenance & Pearl Level-3 SCM Test
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/forney_fg.h>
#include <eshkol/core/soft_unify_semiring.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/* Build a 4-variable tree factor graph: star + branch (0--1, 1--2, 1--3) + unary fields. */
struct GraphSpec {
    std::vector<uint32_t> dims;
    std::vector<std::vector<uint32_t>> fvars;
    std::vector<std::vector<double>> fpots;
};

static eshkol_forney_fg_t* instantiate_graph(const GraphSpec& spec) {
    eshkol_forney_fg_t* fg = eshkol_forney_fg_create(
        static_cast<uint32_t>(spec.dims.size()), spec.dims.data());
    for (size_t i = 0; i < spec.fvars.size(); i++) {
        uint32_t fid = eshkol_forney_fg_add_factor(
            fg, spec.fvars[i].data(), static_cast<uint32_t>(spec.fvars[i].size()),
            spec.fpots[i].data(), static_cast<uint32_t>(spec.fpots[i].size()));
        check(fid == static_cast<uint32_t>(i), "factor added cleanly");
    }
    return fg;
}

static GraphSpec make_tree4_spec() {
    GraphSpec g;
    g.dims = {2, 2, 2, 2};
    g.fvars = {{0}, {1}, {2}, {3}, {0, 1}, {1, 2}, {1, 3}};
    g.fpots = {
        {0.4, -0.3},
        {-0.2, 0.5},
        {0.1, 0.35},
        {-0.15, 0.25},
        {0.7, -0.4, -0.2, 0.6},
        {0.5, -0.1, -0.3, 0.8},
        {-0.25, 0.45, 0.35, -0.15}
    };
    return g;
}

static GraphSpec make_ising_2x2_spec(bool frustrated) {
    GraphSpec g;
    g.dims = {2, 2, 2, 2};
    /* 0 -- 1
     * |    |
     * 3 -- 2 */
    g.fvars = {{0}, {1}, {2}, {3}, {0, 1}, {1, 2}, {2, 3}, {3, 0}};
    double j_last = frustrated ? -0.45 : 0.45;
    g.fpots = {
        {0.3, -0.2},
        {-0.15, 0.25},
        {0.2, 0.1},
        {-0.25, 0.35},
        {0.4, -0.4, -0.4, 0.4},
        {0.35, -0.35, -0.35, 0.35},
        {0.3, -0.3, -0.3, 0.3},
        {j_last, -j_last, -j_last, j_last}
    };
    return g;
}

static GraphSpec make_ising_3x3_spec() {
    GraphSpec g;
    g.dims.assign(9, 2u);
    /* Unary fields */
    for (uint32_t v = 0; v < 9; v++) {
        double h = 0.12 * (static_cast<double>(v % 3) - 1.0) + 0.08 * (static_cast<double>(v / 3) - 1.0);
        g.fvars.push_back({v});
        g.fpots.push_back({h, -h});
    }
    /* Grid edges (12 edges, mixed ferromagnetic & frustrated) */
    const uint32_t edges[12][2] = {
        {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, {7, 8},
        {0, 3}, {3, 6}, {1, 4}, {4, 7}, {2, 5}, {5, 8}
    };
    for (int e = 0; e < 12; e++) {
        double J = (e % 4 == 3) ? -0.30 : 0.35;
        g.fvars.push_back({edges[e][0], edges[e][1]});
        g.fpots.push_back({J, -J, -J, J});
    }
    return g;
}

static void test_forney_fg_tree_bp_and_observation() {
    GraphSpec spec = make_tree4_spec();
    eshkol_forney_fg_t* fg = instantiate_graph(spec);

    double exact_nlz = 0.0;
    check(eshkol_forney_fg_exact_neg_log_z(fg, &exact_nlz), "exact -log Z computed on tree4");

    /* Synchronous BP */
    eshkol_bp_config_t sync_cfg = eshkol_bp_default_config();
    sync_cfg.schedule = ESHKOL_BP_SYNC;
    sync_cfg.tolerance = 1e-12;
    sync_cfg.damping = 1.0;
    eshkol_bp_stats_t sync_st{};
    check(eshkol_forney_fg_infer(fg, &sync_cfg, &sync_st), "sync BP converged on tree4");
    check(near(sync_st.bethe_free_energy, exact_nlz, 1e-10),
          "sync BP F_Bethe == exact -log Z on tree4");

    std::vector<double> sync_m0(2);
    check(eshkol_forney_fg_var_marginal(fg, 0, sync_m0.data(), 2), "sync var0 marginal read");

    /* Residual-priority asynchronous BP */
    eshkol_bp_config_t async_cfg = eshkol_bp_default_config();
    async_cfg.schedule = ESHKOL_BP_RESIDUAL_ASYNC;
    async_cfg.tolerance = 1e-12;
    async_cfg.damping = 0.8;
    eshkol_bp_stats_t async_st{};
    check(eshkol_forney_fg_infer(fg, &async_cfg, &async_st), "residual async BP converged on tree4");
    check(near(async_st.bethe_free_energy, exact_nlz, 1e-10),
          "residual async BP F_Bethe == exact -log Z on tree4");

    std::vector<double> async_m0(2);
    check(eshkol_forney_fg_var_marginal(fg, 0, async_m0.data(), 2), "async var0 marginal read");
    check(near(sync_m0[0], async_m0[0], 1e-10) && near(sync_m0[1], async_m0[1], 1e-10),
          "sync and residual async BP marginals agree to 1e-10");

    /* Observation clamping matches exact conditioned -log Z */
    check(eshkol_forney_fg_observe(fg, 2, 1), "clamped var2 = 1");
    double exact_cond_nlz = 0.0;
    check(eshkol_forney_fg_exact_neg_log_z(fg, &exact_cond_nlz), "exact conditioned -log Z");
    check(eshkol_forney_fg_infer(fg, &sync_cfg, &sync_st), "BP converged with observation");
    check(near(sync_st.bethe_free_energy, exact_cond_nlz, 1e-10),
          "clamped F_Bethe == exact conditioned -log Z");

    eshkol_forney_fg_destroy(fg);
    std::printf("PASS: CSR Forney FG sync + residual-async BP exactness (tree4)\n");
}

static void verify_implicit_grads_against_fd(const char* label, const GraphSpec& spec) {
    eshkol_forney_fg_t* fg = instantiate_graph(spec);
    eshkol_bp_config_t cfg = eshkol_bp_default_config();
    cfg.schedule = ESHKOL_BP_SYNC;
    cfg.max_iterations = 500;
    cfg.tolerance = 1e-13;
    cfg.damping = 0.7;

    eshkol_bp_stats_t st{};
    check(eshkol_forney_fg_infer(fg, &cfg, &st), "BP converged for implicit grad oracle");

    const uint32_t total_p = eshkol_forney_fg_total_params(fg);
    std::vector<double> analytic_fe_grad(total_p, 0.0);
    uint64_t fd_evals = 999;
    check(eshkol_fg_free_energy_grad(fg, analytic_fe_grad.data(), total_p, &fd_evals),
          "analytic Bethe FE gradient succeeded");
    check(fd_evals == 0, "analytic Bethe FE gradient uses 0 finite-difference evals");

    /* Also test arbitrary downstream loss L(b) = sum_v sum_s target_w[v,s] * b_v(s)^2 */
    uint32_t total_vs = 0;
    for (uint32_t d : spec.dims) total_vs += d;
    std::vector<double> target_w(total_vs);
    for (uint32_t i = 0; i < total_vs; i++) {
        target_w[i] = 0.3 + 0.17 * static_cast<double>(i);
    }

    auto eval_loss = [&](const eshkol_forney_fg_t* g) -> double {
        double loss = 0.0;
        uint32_t off = 0;
        for (uint32_t v = 0; v < spec.dims.size(); v++) {
            std::vector<double> m(spec.dims[v]);
            eshkol_forney_fg_var_marginal(g, v, m.data(), spec.dims[v]);
            for (uint32_t s = 0; s < spec.dims[v]; s++) {
                loss += target_w[off + s] * m[s] * m[s];
            }
            off += spec.dims[v];
        }
        return loss;
    };

    std::vector<double> dL_db(total_vs, 0.0);
    {
        uint32_t off = 0;
        for (uint32_t v = 0; v < spec.dims.size(); v++) {
            std::vector<double> m(spec.dims[v]);
            eshkol_forney_fg_var_marginal(fg, v, m.data(), spec.dims[v]);
            for (uint32_t s = 0; s < spec.dims[v]; s++) {
                dL_db[off + s] = 2.0 * target_w[off + s] * m[s];
            }
            off += spec.dims[v];
        }
    }

    std::vector<double> analytic_vjp_grad(total_p, 0.0);
    uint64_t vjp_fd_evals = 999;
    check(eshkol_fg_marginal_vjp(fg, dL_db.data(), total_vs,
                                 analytic_vjp_grad.data(), total_p,
                                 500, 1e-13, &vjp_fd_evals),
          "implicit marginal VJP succeeded");
    check(vjp_fd_evals == 0, "implicit marginal VJP uses 0 finite-difference evals");

    /* Independent test-only central finite-difference oracle */
    const double h = 1e-6;
    double max_fe_err = 0.0;
    double max_vjp_err = 0.0;
    uint32_t p_cursor = 0;
    for (size_t fi = 0; fi < spec.fpots.size(); fi++) {
        std::vector<double> pot = spec.fpots[fi];
        for (size_t idx = 0; idx < pot.size(); idx++) {
            double orig = pot[idx];

            pot[idx] = orig + h;
            eshkol_forney_fg_set_factor_potentials(fg, static_cast<uint32_t>(fi), pot.data(),
                                                   static_cast<uint32_t>(pot.size()));
            eshkol_forney_fg_infer(fg, &cfg, nullptr);
            double fe_plus = eshkol_forney_fg_bethe_free_energy(fg);
            double loss_plus = eval_loss(fg);

            pot[idx] = orig - h;
            eshkol_forney_fg_set_factor_potentials(fg, static_cast<uint32_t>(fi), pot.data(),
                                                   static_cast<uint32_t>(pot.size()));
            eshkol_forney_fg_infer(fg, &cfg, nullptr);
            double fe_minus = eshkol_forney_fg_bethe_free_energy(fg);
            double loss_minus = eval_loss(fg);

            pot[idx] = orig;
            eshkol_forney_fg_set_factor_potentials(fg, static_cast<uint32_t>(fi), pot.data(),
                                                   static_cast<uint32_t>(pot.size()));

            double fd_fe = (fe_plus - fe_minus) / (2.0 * h);
            double fd_loss = (loss_plus - loss_minus) / (2.0 * h);
            max_fe_err = std::max(max_fe_err, std::fabs(analytic_fe_grad[p_cursor] - fd_fe));
            max_vjp_err = std::max(max_vjp_err, std::fabs(analytic_vjp_grad[p_cursor] - fd_loss));
            check(near(analytic_fe_grad[p_cursor], fd_fe, 1e-7),
                  "Bethe FE analytic grad matches central FD < 1e-7");
            check(near(analytic_vjp_grad[p_cursor], fd_loss, 1e-7),
                  "implicit marginal VJP grad matches central FD < 1e-7");
            p_cursor++;
        }
    }

    eshkol_forney_fg_destroy(fg);
    std::printf("PASS: exact implicit differentiation (%s: max_fe_err=%.2e, max_vjp_err=%.2e, fd_evals=0)\n",
                label, max_fe_err, max_vjp_err);
}

static void test_certified_free_energy_enclosures() {
    auto run_enclosure = [](const char* label, const GraphSpec& spec, double max_expected_width) {
        eshkol_forney_fg_t* fg = instantiate_graph(spec);
        double exact_nlz = 0.0;
        check(eshkol_forney_fg_exact_neg_log_z(fg, &exact_nlz), "exact -log Z for enclosure");

        eshkol_fe_interval_t iv{};
        check(eshkol_fg_certify_free_energy(fg, nullptr, &iv), "certified enclosure succeeded");
        check(iv.valid, "certified enclosure valid");
        check(eshkol_fg_interval_contains(&iv, exact_nlz),
              "certified [F_L, F_U] provably encloses exact -log Z");
        double width = iv.upper - iv.lower;
        check(width >= 0.0 && width <= max_expected_width, "certified enclosure width within bound");

        /* Negative controls: tightened/corrupted enclosures must fail validation */
        eshkol_fe_interval_t bad_lo = iv;
        bad_lo.lower = exact_nlz + 1e-4;
        check(!eshkol_fg_interval_contains(&bad_lo, exact_nlz),
              "negative control: raised lower bound fails containment");

        eshkol_fe_interval_t bad_hi = iv;
        bad_hi.upper = exact_nlz - 1e-4;
        check(!eshkol_fg_interval_contains(&bad_hi, exact_nlz),
              "negative control: lowered upper bound fails containment");

        eshkol_fe_interval_t bad_inv = iv;
        bad_inv.valid = false;
        check(!eshkol_fg_interval_contains(&bad_inv, exact_nlz),
              "negative control: invalid interval fails containment");

        eshkol_forney_fg_destroy(fg);
        std::printf("PASS: certified FE enclosure (%s: [%.6f, %.6f] encloses exact %.6f, width=%.2e, trees=%u)\n",
                    label, iv.lower, iv.upper, exact_nlz, width, iv.num_trees);
    };

    run_enclosure("tree4", make_tree4_spec(), 1e-11);
    run_enclosure("ising-2x2-ferro", make_ising_2x2_spec(false), 1.5);
    run_enclosure("ising-2x2-frustrated", make_ising_2x2_spec(true), 1.5);
    run_enclosure("ising-3x3-mixed", make_ising_3x3_spec(), 4.0);
}

static void test_product_manifold_soft_unify_and_proof_dag() {
    eshkol_soft_unify_config_t cfg{};
    cfg.curvature = 1.5;
    cfg.weight_eucl = 0.8;
    cfg.weight_hyp = 1.2;
    cfg.temperature = 1.4;

    double e1[3] = {0.3, -0.5, 0.7};
    double e2[3] = {-0.2, 0.4, 0.1};
    double h1[3] = {0.25, -0.18, 0.12};
    double h2[3] = {-0.15, 0.22, -0.10};

    eshkol_soft_unify_result_t res{};
    double ge1[3], ge2[3], gh1_e[3], gh2_e[3], gh1_r[3], gh2_r[3];
    check(eshkol_soft_unify_eval(e1, e2, 3, h1, h2, 3, &cfg, &res,
                                 ge1, ge2, gh1_e, gh2_e, gh1_r, gh2_r),
          "product-manifold soft-unify evaluated");
    check(res.valid && res.fd_evals == 0, "soft-unify uses 0 finite-difference evals");
    check(res.score > 0.0 && res.score < 1.0 && res.hyp_dist > 0.0,
          "soft-unify score strictly in (0, 1)");

    /* Verify exact Riemannian conformal factor ((1 - c*||h||^2)^2 / 4) */
    double h1_sq = h1[0] * h1[0] + h1[1] * h1[1] + h1[2] * h1[2];
    double h2_sq = h2[0] * h2[0] + h2[1] * h2[1] + h2[2] * h2[2];
    double f1 = (1.0 - cfg.curvature * h1_sq);
    double f2 = (1.0 - cfg.curvature * h2_sq);
    for (int i = 0; i < 3; i++) {
        check(near(gh1_r[i], 0.25 * f1 * f1 * gh1_e[i], 1e-12), "Riemannian metric factor h1 exact");
        check(near(gh2_r[i], 0.25 * f2 * f2 * gh2_e[i], 1e-12), "Riemannian metric factor h2 exact");
    }

    /* Cross-check all 12 coordinate gradients against independent central FD < 1e-7 */
    const double eps = 1e-6;
    auto eval_score_only = [&](const double* ae1, const double* ae2,
                               const double* ah1, const double* ah2) -> double {
        eshkol_soft_unify_result_t r{};
        check(eshkol_soft_unify_eval(ae1, ae2, 3, ah1, ah2, 3, &cfg, &r,
                                     nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
              "FD score probe valid");
        return r.score;
    };
    double max_err = 0.0;
    for (int i = 0; i < 3; i++) {
        double orig = e1[i];
        e1[i] = orig + eps; double sp = eval_score_only(e1, e2, h1, h2);
        e1[i] = orig - eps; double sm = eval_score_only(e1, e2, h1, h2);
        e1[i] = orig;
        double fd = (sp - sm) / (2.0 * eps);
        max_err = std::max(max_err, std::fabs(ge1[i] - fd));
        check(near(ge1[i], fd, 1e-7), "grad_e1 matches central FD < 1e-7");
        check(near(ge2[i], -fd, 1e-7), "grad_e2 antisymmetric exact");

        orig = h1[i];
        h1[i] = orig + eps; sp = eval_score_only(e1, e2, h1, h2);
        h1[i] = orig - eps; sm = eval_score_only(e1, e2, h1, h2);
        h1[i] = orig;
        fd = (sp - sm) / (2.0 * eps);
        max_err = std::max(max_err, std::fabs(gh1_e[i] - fd));
        check(near(gh1_e[i], fd, 1e-7), "grad_h1_eucl matches central FD < 1e-7");

        orig = h2[i];
        h2[i] = orig + eps; sp = eval_score_only(e1, e2, h1, h2);
        h2[i] = orig - eps; sm = eval_score_only(e1, e2, h1, h2);
        h2[i] = orig;
        fd = (sp - sm) / (2.0 * eps);
        max_err = std::max(max_err, std::fabs(gh2_e[i] - fd));
        check(near(gh2_e[i], fd, 1e-7), "grad_h2_eucl matches central FD < 1e-7");
    }

    /* Coincident point smoothness (score == 1, gradients == 0) */
    eshkol_soft_unify_result_t eq_res{};
    check(eshkol_soft_unify_eval(e1, e1, 3, h1, h1, 3, &cfg, &eq_res,
                                 ge1, ge2, gh1_e, gh2_e, gh1_r, gh2_r),
          "coincident soft-unify valid");
    check(near(eq_res.score, 1.0, 1e-12), "coincident soft-unify score == 1.0");
    for (int i = 0; i < 3; i++) {
        check(near(ge1[i], 0.0, 1e-12) && near(gh1_e[i], 0.0, 1e-12),
              "coincident gradients vanish smoothly");
    }

    /* Negative control: Poincare point outside ball boundary rejected */
    double h_bad[3] = {0.9, 0.5, 0.0}; /* c * ||h||^2 = 1.5 * 1.06 > 1 */
    eshkol_soft_unify_result_t bad_res{};
    check(!eshkol_soft_unify_eval(e1, e2, 3, h_bad, h2, 3, &cfg, &bad_res,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
          "negative control: point outside Poincare ball rejected");

    /* Replayable Merkle-hashed PROOF_DAG */
    eshkol_proof_dag_t* dag = eshkol_proof_dag_create();
    uint32_t n0 = eshkol_proof_dag_add_node(
        dag, ESHKOL_PROOF_AXIOM, "fact:edge(a,b)", nullptr, 0,
        eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.9));
    uint32_t n1 = eshkol_proof_dag_add_node(
        dag, ESHKOL_PROOF_SOFT_UNIFY, "soft_unify(b,b')", nullptr, 0,
        eshkol_sr_scalar(ESHKOL_SR_VITERBI, res.score));
    eshkol_semiring_val_t v_mul{}, v_add{};
    eshkol_sr_mul(eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.9),
                  eshkol_sr_scalar(ESHKOL_SR_VITERBI, res.score), &v_mul);
    uint32_t p_mul[2] = {n0, n1};
    uint32_t n2 = eshkol_proof_dag_add_node(
        dag, ESHKOL_PROOF_SEMIRING_MUL, "chain_step", p_mul, 2, v_mul);
    uint32_t n3 = eshkol_proof_dag_add_node(
        dag, ESHKOL_PROOF_AXIOM, "alt_path", nullptr, 0,
        eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.65));
    eshkol_sr_add(v_mul, eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.65), &v_add);
    uint32_t p_add[2] = {n2, n3};
    uint32_t n4 = eshkol_proof_dag_add_node(
        dag, ESHKOL_PROOF_SEMIRING_ADD, "join_paths", p_add, 2, v_add);
    check(n4 == 4 && eshkol_proof_dag_node_hash(dag, n4) != 0ULL, "PROOF_DAG built");
    check(eshkol_proof_dag_verify_replay(dag), "PROOF_DAG deterministic Merkle replay passes");

    /* Negative control: tampering with intermediate value breaks replay verification */
    check(eshkol_proof_dag_corrupt_value_for_test(dag, n1, 0.12345), "corrupted node n1");
    check(!eshkol_proof_dag_verify_replay(dag),
          "negative control: tampered PROOF_DAG rejected by replay verifier");
    eshkol_proof_dag_destroy(dag);

    std::printf("PASS: product-manifold soft-unify (max_grad_err=%.2e) + Merkle PROOF_DAG\n", max_err);
}

static void test_12_semirings_and_diff_topk() {
    struct SampleTriplet {
        eshkol_semiring_kind_t kind;
        eshkol_semiring_val_t x, y, z;
        bool check_distributive;
        bool check_strict_one;
    };

    const SampleTriplet cases[12] = {
        {ESHKOL_SR_BOOLEAN, eshkol_sr_scalar(ESHKOL_SR_BOOLEAN, 1.0),
         eshkol_sr_scalar(ESHKOL_SR_BOOLEAN, 0.0), eshkol_sr_scalar(ESHKOL_SR_BOOLEAN, 1.0), true, true},
        {ESHKOL_SR_VITERBI, eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.7),
         eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.4), eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.8), true, true},
        {ESHKOL_SR_SUM_PRODUCT, eshkol_sr_scalar(ESHKOL_SR_SUM_PRODUCT, 1.5),
         eshkol_sr_scalar(ESHKOL_SR_SUM_PRODUCT, 2.2), eshkol_sr_scalar(ESHKOL_SR_SUM_PRODUCT, 0.7), true, true},
        {ESHKOL_SR_TROPICAL, eshkol_sr_scalar(ESHKOL_SR_TROPICAL, 3.0),
         eshkol_sr_scalar(ESHKOL_SR_TROPICAL, 1.5), eshkol_sr_scalar(ESHKOL_SR_TROPICAL, 4.2), true, true},
        {ESHKOL_SR_MAX_PLUS, eshkol_sr_scalar(ESHKOL_SR_MAX_PLUS, -2.0),
         eshkol_sr_scalar(ESHKOL_SR_MAX_PLUS, 1.5), eshkol_sr_scalar(ESHKOL_SR_MAX_PLUS, 0.4), true, true},
        {ESHKOL_SR_LOG_SPACE, eshkol_sr_scalar(ESHKOL_SR_LOG_SPACE, -1.2),
         eshkol_sr_scalar(ESHKOL_SR_LOG_SPACE, -0.5), eshkol_sr_scalar(ESHKOL_SR_LOG_SPACE, -2.3), true, true},
        {ESHKOL_SR_LUKASIEWICZ, eshkol_sr_scalar(ESHKOL_SR_LUKASIEWICZ, 0.8),
         eshkol_sr_scalar(ESHKOL_SR_LUKASIEWICZ, 0.6), eshkol_sr_scalar(ESHKOL_SR_LUKASIEWICZ, 0.9), true, true},
        {ESHKOL_SR_GODEL, eshkol_sr_scalar(ESHKOL_SR_GODEL, 0.75),
         eshkol_sr_scalar(ESHKOL_SR_GODEL, 0.35), eshkol_sr_scalar(ESHKOL_SR_GODEL, 0.60), true, true},
        {ESHKOL_SR_PRODUCT_FUZZY, eshkol_sr_scalar(ESHKOL_SR_PRODUCT_FUZZY, 0.6),
         eshkol_sr_scalar(ESHKOL_SR_PRODUCT_FUZZY, 0.5), eshkol_sr_scalar(ESHKOL_SR_PRODUCT_FUZZY, 0.3), false, true},
        {ESHKOL_SR_BELNAP4, eshkol_sr_belnap(ESHKOL_BELNAP_NEITHER),
         eshkol_sr_belnap(ESHKOL_BELNAP_BOTH), eshkol_sr_belnap(ESHKOL_BELNAP_TRUE), true, true},
        {ESHKOL_SR_NARS, eshkol_sr_nars(0.8, 0.7),
         eshkol_sr_nars(0.5, 0.6), eshkol_sr_nars(0.9, 0.4), false, false},
        {ESHKOL_SR_CERTIFIED_INTERVAL, eshkol_sr_interval(0.4, 0.5),
         eshkol_sr_interval(0.2, 0.3), eshkol_sr_interval(0.6, 0.8), false, true}
    };

    for (const auto& tc : cases) {
        eshkol_semiring_val_t z0 = eshkol_sr_zero(tc.kind);
        eshkol_semiring_val_t u1 = eshkol_sr_one(tc.kind);
        eshkol_semiring_val_t lhs{}, rhs{}, t1{}, t2{};
        const double tol = (tc.kind == ESHKOL_SR_CERTIFIED_INTERVAL) ? 1e-11 : 1e-11;

        /* Additive identity & commutativity & associativity */
        check(eshkol_sr_add(tc.x, z0, &lhs) && eshkol_sr_equal(lhs, tc.x, tol),
              "semiring additive identity x + 0 == x");
        check(eshkol_sr_add(tc.x, tc.y, &lhs) && eshkol_sr_add(tc.y, tc.x, &rhs) &&
              eshkol_sr_equal(lhs, rhs, tol),
              "semiring additive commutativity");
        check(eshkol_sr_add(tc.x, tc.y, &t1) && eshkol_sr_add(t1, tc.z, &lhs) &&
              eshkol_sr_add(tc.y, tc.z, &t2) && eshkol_sr_add(tc.x, t2, &rhs) &&
              eshkol_sr_equal(lhs, rhs, tol),
              "semiring additive associativity");

        /* Multiplicative associativity */
        check(eshkol_sr_mul(tc.x, tc.y, &t1) && eshkol_sr_mul(t1, tc.z, &lhs) &&
              eshkol_sr_mul(tc.y, tc.z, &t2) && eshkol_sr_mul(tc.x, t2, &rhs) &&
              eshkol_sr_equal(lhs, rhs, tol),
              "semiring multiplicative associativity");

        if (tc.check_strict_one) {
            check(eshkol_sr_mul(tc.x, u1, &lhs) && eshkol_sr_equal(lhs, tc.x, tol),
                  "semiring multiplicative identity x * 1 == x");
        }
        if (tc.check_distributive) {
            check(eshkol_sr_add(tc.y, tc.z, &t1) && eshkol_sr_mul(tc.x, t1, &lhs) &&
                  eshkol_sr_mul(tc.x, tc.y, &t1) && eshkol_sr_mul(tc.x, tc.z, &t2) &&
                  eshkol_sr_add(t1, t2, &rhs) && eshkol_sr_equal(lhs, rhs, tol),
                  "semiring left distributivity x*(y+z) == x*y + x*z");
        }
        if (tc.kind == ESHKOL_SR_CERTIFIED_INTERVAL) {
            /* Outward-rounded enclosure sub-distributivity: x*(y+z) subset of x*y + x*z */
            check(eshkol_sr_add(tc.y, tc.z, &t1) && eshkol_sr_mul(tc.x, t1, &lhs) &&
                  eshkol_sr_mul(tc.x, tc.y, &t1) && eshkol_sr_mul(tc.x, tc.z, &t2) &&
                  eshkol_sr_add(t1, t2, &rhs),
                  "interval semiring ops valid");
            check(rhs.a <= lhs.a + 1e-13 && rhs.b >= lhs.b - 1e-13,
                  "interval semiring sub-distributivity holds");
        }
    }

    /* Differentiable diff-top-k forward + exact analytical implicit backward */
    double scores[5] = {1.8, -0.3, 2.5, 0.9, 0.1};
    double probs[5] = {0};
    double lam = 0.0;
    const double k = 2.0;
    const double tau = 0.5;
    check(eshkol_diff_topk_forward(scores, 5, k, tau, probs, &lam), "diff-top-k forward succeeded");
    double p_sum = 0.0;
    for (int i = 0; i < 5; i++) {
        check(probs[i] > 0.0 && probs[i] < 1.0, "diff-top-k prob strictly in (0, 1)");
        p_sum += probs[i];
    }
    check(near(p_sum, k, 1e-12), "diff-top-k probabilities sum to k");
    check(probs[2] > probs[0] && probs[0] > probs[3] && probs[3] > probs[4] && probs[4] > probs[1],
          "diff-top-k preserves exact rank order");

    const double w[5] = {0.4, -0.7, 1.3, 0.2, -0.5};
    double grad_s[5] = {0};
    uint64_t fd_evals = 77;
    check(eshkol_diff_topk_backward(probs, w, 5, tau, grad_s, &fd_evals),
          "diff-top-k implicit backward succeeded");
    check(fd_evals == 0, "diff-top-k backward uses 0 finite-difference evals");

    const double h = 1e-6;
    double max_err = 0.0;
    for (int i = 0; i < 5; i++) {
        double orig = scores[i];
        double pp[5], pm[5];
        scores[i] = orig + h;
        eshkol_diff_topk_forward(scores, 5, k, tau, pp, nullptr);
        scores[i] = orig - h;
        eshkol_diff_topk_forward(scores, 5, k, tau, pm, nullptr);
        scores[i] = orig;

        double lp = 0.0, lm = 0.0;
        for (int j = 0; j < 5; j++) { lp += w[j] * pp[j]; lm += w[j] * pm[j]; }
        double fd = (lp - lm) / (2.0 * h);
        max_err = std::max(max_err, std::fabs(grad_s[i] - fd));
        check(near(grad_s[i], fd, 1e-7), "diff-top-k implicit grad matches central FD < 1e-7");
    }

    std::printf("PASS: 12-semiring provenance axioms + diff-top-k pullback (max_grad_err=%.2e)\n", max_err);
}

/* Structural equations for confounded SCM:
 *   X = U XOR E_X
 *   Y = (X AND U) OR E_Y */
static uint32_t scm_eq_x(const uint32_t*, uint32_t, const uint32_t* u, uint32_t, void*) {
    return u[0] ^ u[1];
}

static uint32_t scm_eq_y(const uint32_t* p, uint32_t, const uint32_t* u, uint32_t, void*) {
    return (p[0] & u[0]) | u[1];
}

static void test_pearl_level3_causal_counterfactuals() {
    eshkol_scm_t* scm = eshkol_scm_create();
    check(scm != nullptr, "SCM created");

    const double p_u[2]  = {0.5, 0.5};
    const double p_ex[2] = {0.8, 0.2};
    const double p_ey[2] = {0.9, 0.1};
    uint32_t u_conf = eshkol_scm_add_exogenous(scm, "U", p_u, 2);
    uint32_t u_ex   = eshkol_scm_add_exogenous(scm, "E_X", p_ex, 2);
    uint32_t u_ey   = eshkol_scm_add_exogenous(scm, "E_Y", p_ey, 2);
    check(u_conf == 0 && u_ex == 1 && u_ey == 2, "exogenous variables added");

    uint32_t x_exog[2] = {u_conf, u_ex};
    uint32_t v_x = eshkol_scm_add_endogenous(scm, "X", 2, nullptr, 0, x_exog, 2, scm_eq_x, nullptr);
    uint32_t y_parents[1] = {v_x};
    uint32_t y_exog[2] = {u_conf, u_ey};
    uint32_t v_y = eshkol_scm_add_endogenous(scm, "Y", 2, y_parents, 1, y_exog, 2, scm_eq_y, nullptr);
    check(v_x == 0 && v_y == 1, "endogenous variables X, Y added");

    /* Level 1 (Observational): P(Y=1 | X=1) == 0.82 */
    double p_obs = 0.0;
    uint32_t c_var[1] = {v_x}, c_val[1] = {1};
    check(eshkol_scm_observational_query(scm, v_y, 1, c_var, c_val, 1, &p_obs),
          "Level-1 observational query succeeded");
    check(near(p_obs, 0.82, 1e-12), "Level-1 exact P(Y=1 | X=1) == 0.82");

    /* Level 2 (Interventional): P(Y=1 | do(X=1)) == 0.55 */
    double p_do = 0.0;
    uint32_t d_var[1] = {v_x}, d_val[1] = {1};
    check(eshkol_scm_do_query(scm, v_y, 1, d_var, d_val, 1, &p_do),
          "Level-2 interventional do-query succeeded");
    check(near(p_do, 0.55, 1e-12), "Level-2 exact P(Y=1 | do(X=1)) == 0.55");

    /* Level 3 (Counterfactual): P(Y_{do(X=1)}=1 | X=0, Y=0) == 0.20 */
    double p_cf = 0.0;
    uint32_t o_vars[2] = {v_x, v_y};
    uint32_t o_vals[2] = {0, 0};
    check(eshkol_scm_counterfactual(scm, v_y, 1, d_var, d_val, 1, o_vars, o_vals, 2, &p_cf),
          "Level-3 counterfactual query succeeded");
    check(near(p_cf, 0.20, 1e-12), "Level-3 exact P(Y_{do(X=1)}=1 | X=0, Y=0) == 0.20");

    /* Strict Pearl 3-level separation: Observational != Interventional != Counterfactual */
    check(std::fabs(p_obs - p_do) > 0.20 &&
          std::fabs(p_do - p_cf) > 0.30 &&
          std::fabs(p_obs - p_cf) > 0.60,
          "Pearl Ladder separation: P(Y|X) != P(Y|do(X)) != P(Y_x|X',Y')");

    /* Consistency axiom: P(Y_{do(X=0)}=0 | X=0, Y=0) == 1.0 */
    double p_cons = 0.0;
    uint32_t d0[1] = {0};
    check(eshkol_scm_counterfactual(scm, v_y, 0, d_var, d0, 1, o_vars, o_vals, 2, &p_cons),
          "counterfactual consistency query succeeded");
    check(near(p_cons, 1.0, 1e-12), "Pearl counterfactual consistency axiom == 1.0");

    /* Negative control: invalid state value rejected */
    double dummy = 0.0;
    check(!eshkol_scm_do_query(scm, v_y, 5, d_var, d_val, 1, &dummy),
          "negative control: out-of-domain target value rejected");

    eshkol_scm_destroy(scm);
    std::printf("PASS: Pearl Level-3 SCM (obs=%.2f != do=%.2f != cf=%.2f, consistency=1.00)\n",
                p_obs, p_do, p_cf);
}

int main() {
    test_forney_fg_tree_bp_and_observation();
    verify_implicit_grads_against_fd("tree4", make_tree4_spec());
    verify_implicit_grads_against_fd("ising-2x2-loopy", make_ising_2x2_spec(false));
    test_certified_free_energy_enclosures();
    test_product_manifold_soft_unify_and_proof_dag();
    test_12_semirings_and_diff_topk();
    test_pearl_level3_causal_counterfactuals();
    std::printf("CE-2 FORNEY FG & CAUSAL TEST: PASS (%d checks)\n", g_checks);
    return 0;
}
