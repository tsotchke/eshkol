/*
 * Multi-device parity for the mixed-curvature training step (S8): the SAME
 * step that tests/xla/training_step_parity_test.cpp graded on one device,
 * sharded across N addressable devices through Eshkol's PJRT client, graded
 * against the single-device step on the identical full batch.
 *
 * WHAT IS COMPARED.
 *
 * The reference is runTrainingStep: one device, the whole batch, the module
 * S6 proved against the host. The subject is runTrainingStepSharded: the
 * batch's leading axis split into N equal shards, one replica per device,
 * parameters and moments replicated, gradients summed across replicas with
 * stablehlo.all_reduce inside the compiled program, the optimizer run on the
 * reduced gradient on every replica (the reduction is spelled out on
 * TrainingStepSharding in training_step_lowering.h). Both sides run on the
 * device in the device's dtype, so what this harness isolates is the sharding
 * and the reduction, not the lowering: any difference is the reassociation of
 * a sum over n rows into N partial sums plus a cross-replica sum.
 *
 * The two S6 families are kept. STEP rows re-seed the sharded model from the
 * reference's pre-step state, so one application of the sharded operator is
 * compared with one application of the single-device operator on identical
 * inputs. TRAJECTORY rows let the sharded model run free for K steps, so the
 * drift the moments carry from step to step is compounded rather than reset.
 *
 * TOLERANCE CLASSES. Euclidean tensors (W, P_euc and their moments) are
 * reached from the reduced gradient by exact arithmetic — Adam's moments and
 * a Euclidean retraction — so they are graded at the ARITHMETIC bound: the
 * only thing that can separate two f32 evaluations of the same graph is the
 * order of a sum. Manifold tensors (P_hyp, P_sph and their moments) pass
 * through the Poincare and sphere retractions, which are exp maps, and are
 * graded at the TRANSCENDENTAL bound; the loss is a log-sum-exp over acosh
 * and atan2 scores and takes the same. Trajectory rows for manifold tensors
 * take the S6 optimizer bound 2 * k * lr * transcendental, since k updates
 * have compounded. Nothing here is a new tolerance: the constants come from
 * tests/xla/parity_compare.h and the optimizer bound from S6.
 *
 * REPLICA AGREEMENT. Every replica's thirteen results are read back and
 * compared bit for bit with replica 0's. The all_reduce gives every replica
 * the same reduced gradient only up to the reduction's own arithmetic, and
 * the optimizer that follows is deterministic, so the expectation is
 * bit-identity; if it does not hold the divergence is printed and bounded at
 * the arithmetic tolerance rather than waved through.
 *
 * THE CONTROLS.
 *
 *  1. The comparator control from parity_compare.h, before any device.
 *  2. A perturbed-expectation control on real step-1 output.
 *  3. THE NEGATIVE CONTROL: one sharded step compiled WITHOUT the all_reduce
 *     (TrainingStepSharding::omit_all_reduce), so every replica updates from
 *     its own shard's partial gradient. That step MUST fail parity against
 *     the single-device reference. A harness that could not tell a broken
 *     reduction from a working one would be grading nothing about sharding.
 *  4. The cache control: one executable per (shape, N), none per curvature or
 *     per step, checked on the executor's counters.
 *
 * ESHKOL_XLA_MULTIDEVICE_FORCE_FAIL=no_all_reduce runs every graded sharded
 * step without the reduction, so the gate can be shown to fail on demand;
 * any other value names a tensor whose reference is perturbed by 1e-3 at
 * every step, as the S6 harness does.
 *
 * N in {2, 4, 8}. An N above the addressable device count is reported as
 * SKIPPED, by name, in the summary; it is never counted as a pass.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/training_step_lowering.h"
#include "eshkol/ml/mixed_curvature_step.h"

#include "parity_compare.h"

using eshkol::xla::DeviceExecutor;
using eshkol::xla::DeviceStats;
using eshkol::xla::ShardedStepReport;
using eshkol::xla::TrainingStepSharding;
using eshkol::xla::registerStableHLODeviceExecutor;
using eshkol::xla::runTrainingStep;
using eshkol::xla::runTrainingStepSharded;
using eshkol_parity::Comparison;
using eshkol_parity::ToleranceClass;
using eshkol_parity::compareArrays;
using eshkol_parity::setTolerancesForDtype;
using eshkol_parity::toleranceFor;
using eshkol_parity::toleranceClassName;

namespace {

int g_rows_passed = 0;
int g_rows_failed = 0;
int g_constraints_passed = 0;
int g_constraints_failed = 0;
int g_replica_checks_passed = 0;
int g_replica_checks_failed = 0;
int g_replica_bit_identical = 0;
double g_replica_worst_diff = 0.0;
int g_controls_failed = 0;
int g_negative_controls_passed = 0;
int g_configs_tested = 0;
std::string g_dtype = "f32";
const char* g_force_fail = nullptr;
bool g_force_no_all_reduce = false;

/** @brief Steps per configuration. Five so that drift compounds. */
constexpr int kSteps = 5;

struct Model {
    EshkolMixedCurvatureShape shape{};
    std::vector<double> w, p_hyp, p_sph, p_euc;
    std::vector<double> m_w, v_w, m_hyp, v_hyp, m_sph, v_sph, m_euc, v_euc;
    EshkolMixedCurvatureParams params{};
    EshkolMixedCurvatureMoments moments{};

    void allocate(EshkolMixedCurvatureShape s) {
        shape = s;
        const size_t we = static_cast<size_t>(eshkol_mixed_curvature_w_elements(s));
        const size_t pe = static_cast<size_t>(eshkol_mixed_curvature_p_elements(s));
        w.assign(we, 0.0); m_w.assign(we, 0.0); v_w.assign(we, 0.0);
        p_hyp.assign(pe, 0.0); p_sph.assign(pe, 0.0); p_euc.assign(pe, 0.0);
        m_hyp.assign(pe, 0.0); v_hyp.assign(pe, 0.0);
        m_sph.assign(pe, 0.0); v_sph.assign(pe, 0.0);
        m_euc.assign(pe, 0.0); v_euc.assign(pe, 0.0);
        rebind();
    }

    void rebind() {
        params.w = w.data(); params.p_hyp = p_hyp.data();
        params.p_sph = p_sph.data(); params.p_euc = p_euc.data();
        moments.m_w = m_w.data(); moments.v_w = v_w.data();
        moments.m_hyp = m_hyp.data(); moments.v_hyp = v_hyp.data();
        moments.m_sph = m_sph.data(); moments.v_sph = v_sph.data();
        moments.m_euc = m_euc.data(); moments.v_euc = v_euc.data();
    }

    void copyFrom(const Model& o) {
        w = o.w; p_hyp = o.p_hyp; p_sph = o.p_sph; p_euc = o.p_euc;
        m_w = o.m_w; v_w = o.v_w; m_hyp = o.m_hyp; v_hyp = o.v_hyp;
        m_sph = o.m_sph; v_sph = o.v_sph; m_euc = o.m_euc; v_euc = o.v_euc;
        moments.step = o.moments.step;
        rebind();
    }
};

struct Row {
    const char* name;
    const std::vector<double>* subject;
    const std::vector<double>* reference;
    ToleranceClass cls;   ///< Arithmetic for Euclidean tensors, Transcendental for manifold ones.
    bool manifold;
};

double optimizerTolerance(int k, double lr) {
    return 2.0 * static_cast<double>(k) * lr * toleranceFor(ToleranceClass::Transcendental);
}

std::vector<Row> gradedRows(const Model& subject, const Model& ref) {
    return {
        {"W",     &subject.w,     &ref.w,     ToleranceClass::Arithmetic,     false},
        {"P_hyp", &subject.p_hyp, &ref.p_hyp, ToleranceClass::Transcendental, true},
        {"P_sph", &subject.p_sph, &ref.p_sph, ToleranceClass::Transcendental, true},
        {"P_euc", &subject.p_euc, &ref.p_euc, ToleranceClass::Arithmetic,     false},
        {"m_W",   &subject.m_w,   &ref.m_w,   ToleranceClass::Arithmetic,     false},
        {"v_W",   &subject.v_w,   &ref.v_w,   ToleranceClass::Arithmetic,     false},
        {"m_hyp", &subject.m_hyp, &ref.m_hyp, ToleranceClass::Transcendental, true},
        {"v_hyp", &subject.v_hyp, &ref.v_hyp, ToleranceClass::Transcendental, true},
        {"m_sph", &subject.m_sph, &ref.m_sph, ToleranceClass::Transcendental, true},
        {"v_sph", &subject.v_sph, &ref.v_sph, ToleranceClass::Transcendental, true},
        {"m_euc", &subject.m_euc, &ref.m_euc, ToleranceClass::Arithmetic,     false},
        {"v_euc", &subject.v_euc, &ref.v_euc, ToleranceClass::Arithmetic,     false},
    };
}

void printHeader() {
    std::printf("\n%-20s %-3s %-4s %-5s %-7s %-14s %10s %10s %10s  %s\n",
                "config", "N", "step", "fam", "tensor", "class", "tol",
                "max_abs", "max_rel", "verdict");
    std::printf("%s\n", std::string(106, '-').c_str());
}

/**
 * @brief Grade one tensor. @p count false grades without touching the
 *        pass/fail counters (the negative control uses this).
 */
bool gradeRow(const std::string& config, int N, int step, const Row& row,
              const char* family, double tol, bool count = true) {
    std::vector<double> ref = *row.reference;
    if (count && g_force_fail && std::strcmp(g_force_fail, row.name) == 0 && !ref.empty()) {
        ref[0] += 1e-3;
    }
    const Comparison c = compareArrays(*row.subject, ref, tol);
    const bool ok = c.agreed;
    if (count) { if (ok) g_rows_passed++; else g_rows_failed++; }
    std::printf("%-20s %-3d %-4d %-5s %-7s %-14s %10.3e %10.3e %10.3e  %s\n",
                config.c_str(), N, step, family, row.name, toleranceClassName(row.cls),
                tol, c.max_abs, c.max_rel, ok ? "PASS" : (count ? "FAIL" : "fail(expected)"));
    if (!ok && count) {
        std::printf("    first disagreement at index %d: sharded=%.17g single=%.17g tol=%g\n",
                    c.worst_index,
                    c.worst_index >= 0 ? (*row.subject)[static_cast<size_t>(c.worst_index)] : 0.0,
                    c.worst_index >= 0 ? ref[static_cast<size_t>(c.worst_index)] : 0.0, tol);
    }
    return ok;
}

bool gradeConstraints(const std::string& config, int N, int step, const Model& dev,
                      const EshkolMixedCurvatureHyper& h) {
    double worst_hyp = 0.0, worst_sph = 0.0;
    const bool ok = eshkol_mixed_curvature_constraints_hold(
        dev.shape, h.curvature, dev.p_hyp.data(), dev.p_sph.data(),
        h.guard_eps, 1e-6, &worst_hyp, &worst_sph);
    if (ok) g_constraints_passed++; else g_constraints_failed++;
    std::printf("%-20s %-3d %-4d %-5s %-7s %-14s %10s %10.3e %10.3e  %s\n",
                config.c_str(), N, step, "dev", "manifld", "constraint", "-",
                worst_hyp, worst_sph, ok ? "PASS" : "FAIL");
    return ok;
}

/** @brief Replica agreement: bit-identical, or bounded at the arithmetic tolerance. */
bool gradeReplicas(const std::string& config, int N, int step, const ShardedStepReport& r) {
    const double tol = toleranceFor(ToleranceClass::Arithmetic);
    const bool ok = r.replicas_identical || r.max_replica_abs_diff <= tol;
    if (ok) g_replica_checks_passed++; else g_replica_checks_failed++;
    if (r.replicas_identical) g_replica_bit_identical++;
    if (r.max_replica_abs_diff > g_replica_worst_diff) g_replica_worst_diff = r.max_replica_abs_diff;
    std::printf("%-20s %-3d %-4d %-5s %-7s %-14s %10.3e %10.3e %10s  %s%s\n",
                config.c_str(), N, step, "rep", "all", "replicas", tol,
                r.max_replica_abs_diff, "-", ok ? "PASS" : "FAIL",
                r.replicas_identical ? " (bit-identical)" : "");
    if (!r.replicas_identical) {
        std::printf("    replica %d output %d differs from replica 0 by up to %.17g\n",
                    r.worst_replica, r.worst_output, r.max_replica_abs_diff);
    }
    return ok;
}

bool controlPerturbedExpectationFails(const Model& subject, const Model& ref) {
    std::cout << "Control: a perturbed reference for W is rejected... ";
    if (ref.w.empty()) { std::cout << "FAIL (no W)" << std::endl; return false; }
    std::vector<double> perturbed = ref.w;
    perturbed[0] += 1e-3;
    const double tol = toleranceFor(ToleranceClass::Arithmetic);
    const Comparison good = compareArrays(subject.w, ref.w, tol);
    const Comparison bad = compareArrays(subject.w, perturbed, tol);
    if (!good.agreed) { std::cout << "INCONCLUSIVE (the unperturbed row already disagreed)" << std::endl; return false; }
    if (bad.agreed) { std::cout << "FAIL (a 1e-3 perturbation was accepted)" << std::endl; return false; }
    std::cout << "PASS (exact accepted, 1e-3 rejected at index " << bad.worst_index << ")" << std::endl;
    return true;
}

/**
 * @brief The negative control: one sharded step WITHOUT the all_reduce, from
 *        the reference's pre-step state, must disagree with the reference.
 *
 * Graded with the same rows and tolerances as the real comparison, so what
 * is proven is that THIS grading detects a missing reduction. Passes when at
 * least one graded row fails; every row agreeing would mean the harness
 * cannot see the reduction at all.
 */
bool controlMissingAllReduceIsDetected(DeviceExecutor* exec, const std::string& config,
                                       EshkolMixedCurvatureShape s,
                                       const EshkolMixedCurvatureHyper& hyper, int N,
                                       const Model& pre, const Model& post_ref,
                                       double ref_loss, const std::vector<double>& batch,
                                       const std::vector<double>& targets) {
    std::cout << "Control: a sharded step WITHOUT the all_reduce is detected at N=" << N << "... "
              << std::endl;
    Model broken;
    broken.allocate(s);
    broken.copyFrom(pre);
    TrainingStepSharding control;
    control.omit_all_reduce = true;
    double loss = 0.0;
    std::string err;
    if (!runTrainingStepSharded(exec, s, hyper, N, &broken.params, &broken.moments,
                                batch.data(), targets.data(), &loss, nullptr, &control, &err)) {
        std::cout << "  INCONCLUSIVE (the control step refused to run: " << err << ")" << std::endl;
        return false;
    }
    int failed = 0;
    std::vector<double> sl{loss}, rl{ref_loss};
    Row loss_row{"loss", &sl, &rl, ToleranceClass::Transcendental, true};
    if (!gradeRow(config, N, 1, loss_row, "ctrl", toleranceFor(ToleranceClass::Transcendental), false)) failed++;
    for (const Row& r : gradedRows(broken, post_ref)) {
        if (!gradeRow(config, N, 1, r, "ctrl", toleranceFor(r.cls), false)) failed++;
    }
    std::cout << "  " << failed << " of 13 rows disagreed without the reduction: "
              << (failed > 0 ? "PASS (detected)" : "FAIL (a missing all_reduce was not detected)")
              << std::endl;
    return failed > 0;
}

struct ConfigResult {
    bool ok = true;
    int N = 0;
    uint64_t compiled = 0;
    uint64_t cache_hits = 0;
    double device_seconds_total = 0.0;       ///< all K sharded steps, incl. step 1's compile
    double device_seconds_after_first = 0.0; ///< steps 2..K
    std::vector<double> loss;
};

struct ReferenceRun {
    std::vector<Model> pre;    ///< state before each step, size K
    std::vector<Model> post;   ///< state after each step, size K
    std::vector<double> loss;
    double device_seconds_total = 0.0;
    double device_seconds_after_first = 0.0;
    uint64_t compiled = 0;
    bool ok = true;
};

/** @brief K single-device steps on the full batch, every state kept. */
ReferenceRun runReference(DeviceExecutor* exec, const std::string& config,
                          EshkolMixedCurvatureShape s, const EshkolMixedCurvatureHyper& hyper,
                          const Model& init, const std::vector<double>& batch,
                          const std::vector<double>& targets) {
    ReferenceRun out;
    Model m;
    m.allocate(s);
    m.copyFrom(init);
    const DeviceStats before = exec->stats();
    for (int step = 1; step <= kSteps; ++step) {
        Model pre; pre.allocate(s); pre.copyFrom(m);
        out.pre.push_back(pre);
        double loss = 0.0;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = runTrainingStep(exec, s, hyper, &m.params, &m.moments,
                                        batch.data(), targets.data(), &loss, &err);
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        out.device_seconds_total += secs;
        if (step > 1) out.device_seconds_after_first += secs;
        if (!ok) {
            std::printf("%-20s %-3d %-4d  the single-device reference refused to run: %s\n",
                        config.c_str(), 1, step, err.c_str());
            out.ok = false;
            return out;
        }
        Model post; post.allocate(s); post.copyFrom(m);
        out.post.push_back(post);
        out.loss.push_back(loss);
    }
    out.compiled = exec->stats().compiled - before.compiled;
    return out;
}

/** @brief K sharded steps at N, graded against @p ref after every one. */
ConfigResult runSharded(DeviceExecutor* exec, const std::string& config,
                        EshkolMixedCurvatureShape s, const EshkolMixedCurvatureHyper& hyper,
                        int N, const Model& init, const ReferenceRun& ref,
                        const std::vector<double>& batch, const std::vector<double>& targets,
                        bool run_controls) {
    ConfigResult out;
    out.N = N;
    Model free_model, forced;
    free_model.allocate(s);
    forced.allocate(s);
    free_model.copyFrom(init);

    TrainingStepSharding force;
    force.omit_all_reduce = true;
    const TrainingStepSharding* control = g_force_no_all_reduce ? &force : nullptr;

    const DeviceStats before = exec->stats();
    for (int step = 1; step <= kSteps; ++step) {
        const size_t k = static_cast<size_t>(step - 1);

        // ---- step family: from the reference's pre-step state ----
        forced.copyFrom(ref.pre[k]);
        double forced_loss = 0.0;
        ShardedStepReport forced_report;
        std::string err;
        if (!runTrainingStepSharded(exec, s, hyper, N, &forced.params, &forced.moments,
                                    batch.data(), targets.data(), &forced_loss,
                                    &forced_report, control, &err)) {
            std::printf("%-20s %-3d %-4d  the sharded step refused to run: %s\n",
                        config.c_str(), N, step, err.c_str());
            g_rows_failed++;
            out.ok = false;
            return out;
        }
        std::vector<double> fl{forced_loss}, rl{ref.loss[k]};
        Row loss_row{"loss", &fl, &rl, ToleranceClass::Transcendental, true};
        if (!gradeRow(config, N, step, loss_row, "step", toleranceFor(ToleranceClass::Transcendental))) out.ok = false;
        for (const Row& r : gradedRows(forced, ref.post[k])) {
            if (!gradeRow(config, N, step, r, "step", toleranceFor(r.cls))) out.ok = false;
        }
        if (!gradeReplicas(config, N, step, forced_report)) out.ok = false;

        // ---- trajectory family: free-running ----
        double loss = 0.0;
        ShardedStepReport report;
        const auto t0 = std::chrono::steady_clock::now();
        if (!runTrainingStepSharded(exec, s, hyper, N, &free_model.params, &free_model.moments,
                                    batch.data(), targets.data(), &loss, &report, control, &err)) {
            std::printf("%-20s %-3d %-4d  the sharded trajectory step refused to run: %s\n",
                        config.c_str(), N, step, err.c_str());
            g_rows_failed++;
            out.ok = false;
            return out;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        out.device_seconds_total += secs;
        if (step > 1) out.device_seconds_after_first += secs;
        out.loss.push_back(loss);

        std::vector<double> tl{loss};
        Row traj_loss{"loss", &tl, &rl, ToleranceClass::Transcendental, true};
        if (!gradeRow(config, N, step, traj_loss, "traj", toleranceFor(ToleranceClass::Transcendental))) out.ok = false;
        const double mtol = optimizerTolerance(step, hyper.lr);
        for (const Row& r : gradedRows(free_model, ref.post[k])) {
            const double tol = r.manifold ? mtol : toleranceFor(r.cls);
            if (!gradeRow(config, N, step, r, "traj", tol)) out.ok = false;
        }
        if (!gradeReplicas(config, N, step, report)) out.ok = false;
        if (!gradeConstraints(config, N, step, free_model, hyper)) out.ok = false;

        if (step == 1 && run_controls) {
            if (!controlPerturbedExpectationFails(forced, ref.post[0])) g_controls_failed++;
            if (controlMissingAllReduceIsDetected(exec, config, s, hyper, N, ref.pre[0],
                                                  ref.post[0], ref.loss[0], batch, targets)) {
                g_negative_controls_passed++;
            } else {
                g_controls_failed++;
            }
        }
    }
    const DeviceStats after = exec->stats();
    out.compiled = after.compiled - before.compiled;
    out.cache_hits = after.cache_hits - before.cache_hits;
    return out;
}

struct ShapeSpec {
    const char* name;
    int64_t batch, seq, d, c;
};

}  // namespace

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "  XLA Multi-Device Training Step Parity (sharded vs single, S8)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    ::setenv("ESHKOL_XLA_PJRT", "1", 1);
    g_force_fail = std::getenv("ESHKOL_XLA_MULTIDEVICE_FORCE_FAIL");
    if (g_force_fail && !*g_force_fail) g_force_fail = nullptr;
    if (g_force_fail && std::strcmp(g_force_fail, "no_all_reduce") == 0) {
        g_force_no_all_reduce = true;
        g_force_fail = nullptr;
        std::cout << "FORCE FAIL requested: every graded sharded step runs WITHOUT the all_reduce."
                  << std::endl;
    } else if (g_force_fail) {
        std::cout << "FORCE FAIL requested for tensor '" << g_force_fail
                  << "': its reference is perturbed by 1e-3." << std::endl;
    }

    if (!eshkol_parity::test_comparator_rejects_a_perturbed_result()) {
        std::cerr << "The comparator control failed; no row below would mean anything." << std::endl;
        return 1;
    }

    DeviceExecutor* exec = registerStableHLODeviceExecutor();
    std::string why;
    if (!exec || !exec->available(&why)) {
        std::cout << "\nNo PJRT device is reachable: " << why << std::endl;
        std::cout << "A multi-device parity claim needs devices, so this is not a pass." << std::endl;
        return 77;
    }
    const int devices = exec->addressableDeviceCount();
    std::cout << "\nDevice: " << exec->description() << std::endl;
    std::cout << "Addressable devices: " << devices << std::endl;
    if (devices < 2) {
        std::cout << "Fewer than two addressable devices: nothing can be sharded, so this is not a pass."
                  << std::endl;
        return 77;
    }
    g_dtype = exec->dtypeName();
    setTolerancesForDtype(g_dtype);
    std::printf("Tolerance (%s): arithmetic=%g transcendental=%g; optimizer bound 2*k*lr*transcendental\n",
                g_dtype.c_str(), toleranceFor(ToleranceClass::Arithmetic),
                toleranceFor(ToleranceClass::Transcendental));
    std::printf("Steps per configuration: K=%d\n", kSteps);

    const std::vector<ShapeSpec> shapes = {
        {"b8xs16xd32xc8", 8, 16, 32, 8},
        {"b32xs64xd64xc8", 32, 64, 64, 8},
    };
    const std::vector<double> curvatures = {1.0, 0.5};
    const std::vector<int> replica_counts = {2, 4, 8};

    std::vector<int> tested_n, skipped_n;
    for (int N : replica_counts) {
        if (N <= devices) tested_n.push_back(N); else skipped_n.push_back(N);
    }
    for (int N : skipped_n) {
        std::printf("SKIPPED N=%d: only %d addressable device(s) on this node\n", N, devices);
    }

    printHeader();

    struct Report {
        std::string config;
        int64_t n;
        ReferenceRun ref;
        std::vector<ConfigResult> sharded;
        uint64_t compiled_total;
    };
    std::vector<Report> reports;

    bool first = true;
    for (const ShapeSpec& sp : shapes) {
        EshkolMixedCurvatureShape s{sp.batch * sp.seq, sp.d, sp.c};
        uint64_t seed = 0x5EEDu;
        for (double curv : curvatures) {
            EshkolMixedCurvatureHyper hyper{};
            eshkol_mixed_curvature_default_hyper(&hyper);
            hyper.curvature = curv;
            std::string config = std::string(sp.name) + "|c=" + std::to_string(curv).substr(0, 3);

            Model init;
            init.allocate(s);
            std::vector<double> batch(static_cast<size_t>(eshkol_mixed_curvature_x_elements(s)));
            std::vector<double> targets(static_cast<size_t>(eshkol_mixed_curvature_t_elements(s)));
            eshkol_mixed_curvature_init(s, &hyper, seed, &init.params, &init.moments,
                                        batch.data(), targets.data());

            const DeviceStats before = exec->stats();
            Report rep{config, s.n, {}, {}, 0};
            rep.ref = runReference(exec, config, s, hyper, init, batch, targets);
            if (!rep.ref.ok) {
                g_rows_failed++;
                reports.push_back(rep);
                seed++;
                continue;
            }
            for (int N : tested_n) {
                rep.sharded.push_back(runSharded(exec, config, s, hyper, N, init, rep.ref,
                                                 batch, targets, first && N == tested_n.front()));
                g_configs_tested++;
            }
            rep.compiled_total = exec->stats().compiled - before.compiled;
            first = false;
            reports.push_back(rep);
            seed++;
        }
    }

    // ---- timing ----
    // Wall time of the whole device call from the host's side, including the
    // f64->f32 staging, the transfers and the read-back, for both the single
    // device reference and the sharded step; step 1 carries the compile and
    // is reported separately from steps 2..K. The N=1 column is re-measured
    // here in the same process on the same batch, not reused from S6.
    std::cout << "\nTiming (seconds per step, wall time of the device call from the host):" << std::endl;
    std::printf("%-20s %-3s %14s %14s %8s %10s %10s\n", "config", "N",
                "step1(compile)", "steps2..K/step", "vs N=1", "compiled", "cache_hits");
    for (const Report& rep : reports) {
        if (!rep.ref.ok) continue;
        const double ref_first = rep.ref.device_seconds_total - rep.ref.device_seconds_after_first;
        const double ref_per = rep.ref.device_seconds_after_first / (kSteps - 1);
        std::printf("%-20s %-3d %14.6f %14.6f %8s %10s %10s\n", rep.config.c_str(), 1,
                    ref_first, ref_per, "1.00", "-", "-");
        for (const ConfigResult& r : rep.sharded) {
            const double first_s = r.device_seconds_total - r.device_seconds_after_first;
            const double per = r.device_seconds_after_first / (kSteps - 1);
            std::printf("%-20s %-3d %14.6f %14.6f %8.2f %10llu %10llu\n", rep.config.c_str(), r.N,
                        first_s, per, ref_per > 0.0 ? per / ref_per : 0.0,
                        static_cast<unsigned long long>(r.compiled),
                        static_cast<unsigned long long>(r.cache_hits));
        }
    }

    // ---- cache control ----
    // The first curvature at a shape compiles one single-device executable
    // plus one per N; the second curvature at that shape compiles nothing.
    // The negative control at the first configuration compiles one more
    // (its module is a different program and must not share a key).
    std::cout << "\nControl: one executable per (shape, N), none per curvature or step... ";
    bool cache_ok = true;
    for (size_t i = 0; i < reports.size(); ++i) {
        const Report& rep = reports[i];
        if (!rep.ref.ok) continue;
        const bool first_at_shape = (i % curvatures.size() == 0);
        uint64_t expected = first_at_shape ? 1u + static_cast<uint64_t>(tested_n.size()) : 0u;
        if (i == 0) expected += 1;   // the negative control's module
        if (rep.compiled_total != expected) {
            std::cout << "\n  " << rep.config << " compiled " << rep.compiled_total
                      << " executables, expected " << expected;
            cache_ok = false;
        }
        for (const ConfigResult& r : rep.sharded) {
            // 2 sharded calls per step (step family + trajectory family);
            // all but the compile must be hits.
            const uint64_t calls = 2u * static_cast<uint64_t>(kSteps);
            if (r.cache_hits + r.compiled < calls) {
                std::cout << "\n  " << rep.config << " N=" << r.N << " served only "
                          << r.cache_hits << " of " << calls << " sharded calls from the cache";
                cache_ok = false;
            }
        }
    }
    if (!cache_ok) g_controls_failed++;
    std::cout << (cache_ok ? "PASS" : " -> FAIL") << std::endl;

    std::string tested, skipped;
    for (int N : tested_n) tested += (tested.empty() ? "" : ",") + std::to_string(N);
    for (int N : skipped_n) skipped += (skipped.empty() ? "" : ",") + std::to_string(N);

    const DeviceStats stats = exec->stats();
    std::printf("\nSUMMARY: rows_passed=%d rows_failed=%d constraints_passed=%d constraints_failed=%d "
                "replica_checks_passed=%d replica_checks_failed=%d replica_bit_identical=%d "
                "replica_worst_diff=%.3e negative_controls_passed=%d controls_failed=%d "
                "configs_tested=%d replicas_tested=%s replicas_skipped=%s devices=%d steps=%d "
                "dtype=%s executed=%llu compiled=%llu cache_hits=%llu\n",
                g_rows_passed, g_rows_failed, g_constraints_passed, g_constraints_failed,
                g_replica_checks_passed, g_replica_checks_failed, g_replica_bit_identical,
                g_replica_worst_diff, g_negative_controls_passed, g_controls_failed,
                g_configs_tested, tested.empty() ? "none" : tested.c_str(),
                skipped.empty() ? "none" : skipped.c_str(), devices, kSteps, g_dtype.c_str(),
                static_cast<unsigned long long>(stats.executed),
                static_cast<unsigned long long>(stats.compiled),
                static_cast<unsigned long long>(stats.cache_hits));

    const bool all_ok = (g_rows_failed == 0) && (g_constraints_failed == 0) &&
                        (g_replica_checks_failed == 0) && (g_controls_failed == 0) &&
                        (g_negative_controls_passed > 0) && (g_rows_passed > 0) &&
                        (g_constraints_passed > 0) && (g_configs_tested > 0);
    std::cout << (all_ok ? "MULTIDEVICE STEP PARITY: PASS" : "MULTIDEVICE STEP PARITY: FAIL")
              << std::endl;
    return all_ok ? 0 : 1;
}
