/*
 * Device/host parity for a FULL TRAINING STEP of the mixed-curvature model:
 * forward, backward and the Riemannian Adam update, lowered as ONE StableHLO
 * program (lib/backend/xla/training_step_lowering.cpp) and executed through
 * Eshkol's PJRT client, against the host's own training step
 * (lib/ml/mixed_curvature_step.cpp) called through its public entry point.
 *
 * WHAT THIS PROVES, AND IN TWO DIFFERENT SENSES.
 *
 * Both sides start from parameters produced by ONE generator,
 * eshkol_mixed_curvature_init, and consume one batch. From there the harness
 * grades TWO families of rows at every step, because they answer two different
 * questions and one bound cannot serve both:
 *
 *   STEP rows. The host is stepped from the DEVICE's own current state, so the
 *   two sides consume identical inputs and what is compared is one application
 *   of the step operator. These carry the per-op tolerance classes of
 *   docs/design/ESHKOL_S_FRAGMENT.md unchanged, and they are graded at five
 *   DIFFERENT points along a real trajectory, with real moments that have been
 *   accumulating — not five times at the initial point.
 *
 *   TRAJECTORY rows. A second host model runs free, never re-seeded, so the
 *   two trajectories evolve independently for K steps exactly as the stage
 *   brief asks and an error at step 1 compounds through the moments rather
 *   than being re-seeded away.
 *
 * The trajectory rows CANNOT be graded at the per-op bound, and the reason is
 * a property of Adam rather than of this lowering. Adam's delta is
 * -lr m_hat/(sqrt(v_hat) + eps): a NORMALISED step whose magnitude is about lr
 * whatever the gradient's magnitude. A relative error e in the gradient
 * therefore survives into the delta essentially undamped, and at most doubled
 * (numerator and denominator each carry it), so after k steps the two
 * parameter sets can differ by
 *
 *      2 * k * lr * (relative accuracy of the gradient).
 *
 * Every gradient in this model flows through acosh, atan2 and tanh, so that
 * accuracy is the TRANSCENDENTAL class, whichever parameter the gradient lands
 * on. That expression, with nothing fitted to a measurement, is the bound the
 * trajectory rows use, and it is printed with each row. Measured against it on
 * the TPU at f32 the worst trajectory row sits at roughly half the bound.
 *
 * Grading the compounding rows at the per-op bound instead would not be
 * stricter, it would be wrong: it would demand that five f32 optimizer steps
 * land where five f64 optimizer steps did, which no correct implementation of
 * this step can do.
 *
 * This harness RE-IMPLEMENTS NOTHING. Every number on the host side comes from
 * the public entry points in inc/eshkol/ml/mixed_curvature_step.h; the file
 * below allocates, calls, and compares.
 *
 * WHY THE TWO BACKWARD PASSES ARE INDEPENDENT.
 *
 * The device's backward is StableHLOEmitter::emitVJP walking the forward
 * graph. The host's is derived by hand, stage by stage, with the chain rule
 * written above each function. Neither consults the other, so an agreement
 * over five compounding steps is evidence about both, and a disagreement
 * localises to whichever tensor first moves.
 *
 * THE CONTROLS.
 *
 *  1. The comparator control from tests/xla/parity_compare.h, run before any
 *     device is required, so a comparator that had regressed into `return
 *     true` is caught rather than passing every row beneath it.
 *  2. A perturbed-expectation control: after step 1, one element of the
 *     expected W is moved by 1e-3 and the same comparison MUST report FAIL. A
 *     harness whose rows cannot fail grades nothing.
 *  3. A cache control: the second step at a shape must be served by an
 *     already-compiled executable, and two curvatures at one shape must not
 *     produce two compiles. A curvature or a step index baked into the module
 *     would give correct numbers for the first call and silently wrong ones
 *     after, so this is checked on the compile counter, not on values.
 *
 * ESHKOL_XLA_TRAINING_FORCE_FAIL=<tensor> perturbs the HOST expectation for
 * that tensor by 1e-3 at every step, so the gate can be shown to fail on
 * demand.
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
using eshkol::xla::registerStableHLODeviceExecutor;
using eshkol::xla::runTrainingStep;
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
int g_controls_failed = 0;
int g_trajectories_passed = 0;
int g_trajectories_failed = 0;
std::string g_dtype = "f32";
const char* g_force_fail = nullptr;

/** @brief Steps per configuration. Five so that drift compounds. */
constexpr int kSteps = 5;

/**
 * @brief One model's storage: the four parameters, the eight moments, and the
 *        scratch the host step needs.
 *
 * A struct because the host step takes its state through two structs of
 * pointers, and building those by hand at each of the twenty call sites is
 * where the device's moments would eventually get passed to the host.
 */
struct Model {
    EshkolMixedCurvatureShape shape{};
    std::vector<double> w, p_hyp, p_sph, p_euc;
    std::vector<double> m_w, v_w, m_hyp, v_hyp, m_sph, v_sph, m_euc, v_euc;
    std::vector<double> scratch;
    EshkolMixedCurvatureParams params{};
    EshkolMixedCurvatureMoments moments{};

    void allocate(EshkolMixedCurvatureShape s) {
        shape = s;
        const size_t we = static_cast<size_t>(eshkol_mixed_curvature_w_elements(s));
        const size_t pe = static_cast<size_t>(eshkol_mixed_curvature_p_elements(s));
        w.assign(we, 0.0);
        m_w.assign(we, 0.0);
        v_w.assign(we, 0.0);
        p_hyp.assign(pe, 0.0);
        p_sph.assign(pe, 0.0);
        p_euc.assign(pe, 0.0);
        m_hyp.assign(pe, 0.0);
        v_hyp.assign(pe, 0.0);
        m_sph.assign(pe, 0.0);
        v_sph.assign(pe, 0.0);
        m_euc.assign(pe, 0.0);
        v_euc.assign(pe, 0.0);
        scratch.assign(static_cast<size_t>(eshkol_mixed_curvature_scratch_elements(s)), 0.0);
        rebind();
    }

    void rebind() {
        params.w = w.data();
        params.p_hyp = p_hyp.data();
        params.p_sph = p_sph.data();
        params.p_euc = p_euc.data();
        moments.m_w = m_w.data();
        moments.v_w = v_w.data();
        moments.m_hyp = m_hyp.data();
        moments.v_hyp = v_hyp.data();
        moments.m_sph = m_sph.data();
        moments.v_sph = v_sph.data();
        moments.m_euc = m_euc.data();
        moments.v_euc = v_euc.data();
    }

    /** @brief Copy every tensor and the step count from @p other. */
    void copyFrom(const Model& other) {
        w = other.w; p_hyp = other.p_hyp; p_sph = other.p_sph; p_euc = other.p_euc;
        m_w = other.m_w; v_w = other.v_w;
        m_hyp = other.m_hyp; v_hyp = other.v_hyp;
        m_sph = other.m_sph; v_sph = other.v_sph;
        m_euc = other.m_euc; v_euc = other.v_euc;
        moments.step = other.moments.step;
        rebind();
    }
};

/** @brief One graded tensor: its name, the two sides, and its tolerance class. */
struct Row {
    const char* name;
    const std::vector<double>* device;
    const std::vector<double>* host;
    ToleranceClass cls;
};

/**
 * @brief The trajectory bound at step @p k, derived in the file comment.
 *
 * 2 * k * lr * (transcendental tolerance). Nothing here is fitted to a
 * measurement: the 2 is the numerator-and-denominator doubling in
 * m_hat/sqrt(v_hat), the lr is the magnitude of a normalised Adam step, and
 * the transcendental tolerance is the gradient's relative accuracy, which is
 * the class of every gradient in this model because every one of them flows
 * through acosh, atan2 and tanh.
 */
double trajectoryTolerance(int k, double lr) {
    return 2.0 * static_cast<double>(k) * lr * toleranceFor(ToleranceClass::Transcendental);
}

/**
 * @brief The thirteen graded tensors of a step.
 *
 * W and P_euc are reached by exact arithmetic and keep the arithmetic bound;
 * P_hyp and P_sph pass through exp maps and take the transcendental one, and
 * each moment takes its parameter's class because it is a running average of
 * that parameter's gradient. The loss is transcendental: it is a log-sum-exp
 * over scores that contain acosh and atan2.
 */
std::vector<Row> gradedRows(const Model& dev, const Model& host) {
    return {
        {"W",     &dev.w,     &host.w,     ToleranceClass::Arithmetic},
        {"P_hyp", &dev.p_hyp, &host.p_hyp, ToleranceClass::Transcendental},
        {"P_sph", &dev.p_sph, &host.p_sph, ToleranceClass::Transcendental},
        {"P_euc", &dev.p_euc, &host.p_euc, ToleranceClass::Arithmetic},
        {"m_W",   &dev.m_w,   &host.m_w,   ToleranceClass::Arithmetic},
        {"v_W",   &dev.v_w,   &host.v_w,   ToleranceClass::Arithmetic},
        {"m_hyp", &dev.m_hyp, &host.m_hyp, ToleranceClass::Transcendental},
        {"v_hyp", &dev.v_hyp, &host.v_hyp, ToleranceClass::Transcendental},
        {"m_sph", &dev.m_sph, &host.m_sph, ToleranceClass::Transcendental},
        {"v_sph", &dev.v_sph, &host.v_sph, ToleranceClass::Transcendental},
        {"m_euc", &dev.m_euc, &host.m_euc, ToleranceClass::Arithmetic},
        {"v_euc", &dev.v_euc, &host.v_euc, ToleranceClass::Arithmetic},
    };
}

void printHeader() {
    std::printf("\n%-22s %-6s %-5s %-7s %-14s %10s %10s %10s  %s\n",
                "config", "step", "fam", "tensor", "class", "tol",
                "max_abs", "max_rel", "verdict");
    std::printf("%s\n", std::string(104, '-').c_str());
}

/**
 * @brief Grade one tensor and report it.
 *
 * @param family "step" for the re-seeded one-step comparison, "traj" for the
 *               free-running one. Printed, so no row's verdict can be read
 *               without knowing which bound produced it.
 * @param tol    The bound. Passed rather than derived from the class, because
 *               the trajectory family's bound depends on the step index.
 */
bool gradeRow(const std::string& config, int step, const Row& row,
              const char* family, double tol) {
    std::vector<double> host = *row.host;
    if (g_force_fail && std::strcmp(g_force_fail, row.name) == 0 && !host.empty()) {
        host[0] += 1e-3;
    }
    const Comparison c = compareArrays(*row.device, host, tol);
    const bool ok = c.agreed;
    if (ok) g_rows_passed++; else g_rows_failed++;
    std::printf("%-22s %-6d %-5s %-7s %-14s %10.3e %10.3e %10.3e  %s\n",
                config.c_str(), step, family, row.name,
                toleranceClassName(row.cls), tol, c.max_abs, c.max_rel,
                ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("    first disagreement at index %d: device=%.17g host=%.17g tol=%g\n",
                    c.worst_index,
                    c.worst_index >= 0 ? (*row.device)[static_cast<size_t>(c.worst_index)] : 0.0,
                    c.worst_index >= 0 ? host[static_cast<size_t>(c.worst_index)] : 0.0,
                    tol);
    }
    return ok;
}

/**
 * @brief Control: the same comparison, with one expected element moved, must FAIL.
 *
 * Run on real step-1 output rather than on synthetic numbers, so what is
 * proven is that THIS harness's grading of THIS tensor can return FAIL — not
 * merely that compareArrays can.
 */
bool controlPerturbedExpectationFails(const Model& dev, const Model& host) {
    std::cout << "Control: a perturbed expectation for W is rejected... ";
    if (host.w.empty()) { std::cout << "FAIL (no W to perturb)" << std::endl; return false; }
    std::vector<double> perturbed = host.w;
    perturbed[0] += 1e-3;   // far outside the arithmetic bound at either dtype
    const double tol = toleranceFor(ToleranceClass::Arithmetic);
    const Comparison good = compareArrays(dev.w, host.w, tol);
    const Comparison bad = compareArrays(dev.w, perturbed, tol);
    if (!good.agreed) {
        std::cout << "INCONCLUSIVE (the unperturbed row already disagreed)" << std::endl;
        return false;
    }
    if (bad.agreed) {
        std::cout << "FAIL (a 1e-3 perturbation was accepted)" << std::endl;
        return false;
    }
    std::cout << "PASS (exact accepted, 1e-3 rejected at index " << bad.worst_index << ")"
              << std::endl;
    return true;
}

/** @brief The manifold constraints, on the DEVICE's own outputs. */
bool gradeConstraints(const std::string& config, int step, const Model& dev,
                      const EshkolMixedCurvatureHyper& h) {
    double worst_hyp = 0.0, worst_sph = 0.0;
    // The margin is the guard the retraction itself enforces; asking for more
    // would be asking the clip to overshoot, and asking for less would accept
    // a point on the boundary, where the conformal factor is singular.
    const bool ok = eshkol_mixed_curvature_constraints_hold(
        dev.shape, h.curvature, dev.p_hyp.data(), dev.p_sph.data(),
        h.guard_eps, 1e-6, &worst_hyp, &worst_sph);
    if (ok) g_constraints_passed++; else g_constraints_failed++;
    std::printf("%-22s %-6d %-5s %-7s %-14s %10s %10.3e %10.3e  %s\n",
                config.c_str(), step, "dev", "manifld", "constraint", "-",
                worst_hyp, worst_sph, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("    c|P_hyp|^2 max %.17g must be <= %.17g; "
                    "| |P_sph| - 1 | max %.17g must be <= 1e-6\n",
                    worst_hyp, 1.0 - h.guard_eps, worst_sph);
    }
    return ok;
}

struct ConfigResult {
    bool ok = true;
    uint64_t compiled = 0;
    uint64_t cache_hits = 0;
    double device_seconds = 0.0;
    double host_seconds = 0.0;
    std::vector<double> device_loss;
    std::vector<double> host_loss;
};

/**
 * @brief K steps on each side from one initial state, graded after every step.
 */
ConfigResult runConfig(DeviceExecutor* exec, const std::string& config,
                       EshkolMixedCurvatureShape s,
                       const EshkolMixedCurvatureHyper& hyper,
                       uint64_t seed, bool run_control) {
    ConfigResult out;

    // `free` runs the whole trajectory without ever being re-seeded; `forced`
    // is re-seeded from the device's own state before every step so that the
    // step operator is graded on identical inputs. Two models rather than one
    // because the two questions cannot share a state.
    Model free_host, forced, dev;
    free_host.allocate(s);
    forced.allocate(s);
    dev.allocate(s);

    std::vector<double> batch(static_cast<size_t>(eshkol_mixed_curvature_x_elements(s)));
    std::vector<double> targets(static_cast<size_t>(eshkol_mixed_curvature_t_elements(s)));
    eshkol_mixed_curvature_init(s, &hyper, seed, &free_host.params, &free_host.moments,
                                batch.data(), targets.data());
    dev.copyFrom(free_host);

    exec->resetStats();
    const DeviceStats before = exec->stats();

    for (int step = 1; step <= kSteps; ++step) {
        double host_loss = 0.0, dev_loss = 0.0, forced_loss = 0.0;

        // Re-seed the forced model from the DEVICE's pre-step state, so this
        // step's comparison is of one operator on one input.
        forced.copyFrom(dev);
        bool forced_ok = eshkol_mixed_curvature_train_step(
            s, &hyper, &forced.params, &forced.moments, batch.data(), targets.data(),
            forced.scratch.data(), &forced_loss, nullptr);

        const auto h0 = std::chrono::steady_clock::now();
        const bool host_ok = eshkol_mixed_curvature_train_step(
            s, &hyper, &free_host.params, &free_host.moments, batch.data(), targets.data(),
            free_host.scratch.data(), &host_loss, nullptr);
        const auto h1 = std::chrono::steady_clock::now();
        out.host_seconds += std::chrono::duration<double>(h1 - h0).count();

        if (!host_ok || !forced_ok) {
            std::printf("%-22s %-6d %-7s  a HOST step refused to run\n",
                        config.c_str(), step, "-");
            g_rows_failed++;
            out.ok = false;
            return out;
        }

        std::string err;
        const auto d0 = std::chrono::steady_clock::now();
        const bool dev_ok = runTrainingStep(exec, s, hyper, &dev.params, &dev.moments,
                                            batch.data(), targets.data(), &dev_loss, &err);
        const auto d1 = std::chrono::steady_clock::now();
        out.device_seconds += std::chrono::duration<double>(d1 - d0).count();

        if (!dev_ok) {
            std::printf("%-22s %-6d %-7s  the DEVICE step refused to run: %s\n",
                        config.c_str(), step, "-", err.c_str());
            g_rows_failed++;
            out.ok = false;
            return out;
        }

        out.host_loss.push_back(host_loss);
        out.device_loss.push_back(dev_loss);

        // ---- step family: identical inputs, per-op tolerance classes ----
        // The loss is a row like any other, graded through the same comparator.
        std::vector<double> dl{dev_loss}, fl{forced_loss};
        Row loss_row{"loss", &dl, &fl, ToleranceClass::Transcendental};
        if (!gradeRow(config, step, loss_row, "step",
                      toleranceFor(ToleranceClass::Transcendental))) out.ok = false;
        for (const Row& r : gradedRows(dev, forced)) {
            if (!gradeRow(config, step, r, "step", toleranceFor(r.cls))) out.ok = false;
        }

        // ---- trajectory family: independent evolution, derived bound ----
        const double ttol = trajectoryTolerance(step, hyper.lr);
        std::vector<double> hl{host_loss};
        Row traj_loss{"loss", &dl, &hl, ToleranceClass::Transcendental};
        if (!gradeRow(config, step, traj_loss, "traj", ttol)) out.ok = false;
        for (const Row& r : gradedRows(dev, free_host)) {
            if (!gradeRow(config, step, r, "traj", ttol)) out.ok = false;
        }

        if (!gradeConstraints(config, step, dev, hyper)) out.ok = false;

        if (step == 1 && run_control) {
            if (!controlPerturbedExpectationFails(dev, forced)) g_controls_failed++;
        }
    }

    const DeviceStats after = exec->stats();
    out.compiled = after.compiled - before.compiled;
    out.cache_hits = after.cache_hits - before.cache_hits;
    return out;
}

/**
 * @brief The loss trajectories must move the same way, step for step.
 *
 * Not "both decreased": a device that decreased for a different reason than
 * the host, or by a different amount, is a different optimizer. The step-wise
 * SIGN of the change is required to agree, and the fact that the losses
 * themselves already agreed to tolerance is what makes the sign check
 * meaningful rather than redundant — a trajectory can agree pointwise at a
 * coarse tolerance while moving the wrong way between two close points.
 */
bool gradeTrajectory(const std::string& config, const ConfigResult& r) {
    if (r.device_loss.size() < 2 || r.device_loss.size() != r.host_loss.size()) {
        std::printf("%-22s %-6s %-5s  no trajectory to grade\n", config.c_str(), "-", "traj");
        g_trajectories_failed++;
        return false;
    }
    bool ok = true;
    int device_decreases = 0, host_decreases = 0;
    for (size_t i = 1; i < r.device_loss.size(); ++i) {
        const double dd = r.device_loss[i] - r.device_loss[i - 1];
        const double hd = r.host_loss[i] - r.host_loss[i - 1];
        if (dd < 0.0) device_decreases++;
        if (hd < 0.0) host_decreases++;
        if ((dd < 0.0) != (hd < 0.0)) ok = false;
    }
    const int steps = static_cast<int>(r.device_loss.size()) - 1;

    // The two sides must agree on the DIRECTION at every step (checked above),
    // must have taken the same NUMBER of downhill steps, and the loss must be
    // lower at K than at 1 on both.
    //
    // What is deliberately NOT required is a decrease at every single step.
    // Adam's delta is normalised: its magnitude is about lr regardless of the
    // gradient, so a coordinate near its minimum is stepped past by a fixed
    // distance and a single step can raise the loss. That is a property of the
    // optimizer being mirrored, not of the mirroring, and demanding otherwise
    // would be demanding that the device implement a different optimizer than
    // the host. The per-step counts are printed so that the two sides
    // over-stepping in the same places stays visible rather than being
    // summarised away.
    const bool same_count = (device_decreases == host_decreases);
    const bool net_down = (r.device_loss.back() < r.device_loss.front()) &&
                          (r.host_loss.back() < r.host_loss.front());
    ok = ok && same_count && net_down;
    if (ok) g_trajectories_passed++; else g_trajectories_failed++;
    std::printf("%-22s %-6s %-5s %-7s %-14s device %d/%d host %d/%d  "
                "first %.10g -> last %.10g  %s\n",
                config.c_str(), "1..K", "traj", "loss", "direction",
                device_decreases, steps, host_decreases, steps,
                r.device_loss.front(), r.device_loss.back(), ok ? "PASS" : "FAIL");
    return ok;
}

struct ShapeSpec {
    const char* name;
    int64_t batch;
    int64_t seq;
    int64_t d;
    int64_t c;
};

}  // namespace

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  XLA Training Step Parity (device vs host, S6)" << std::endl;
    std::cout << "=================================================" << std::endl;

    ::setenv("ESHKOL_XLA_PJRT", "1", 1);
    g_force_fail = std::getenv("ESHKOL_XLA_TRAINING_FORCE_FAIL");
    if (g_force_fail && !*g_force_fail) g_force_fail = nullptr;
    if (g_force_fail) {
        std::cout << "FORCE FAIL requested for tensor '" << g_force_fail
                  << "': its host expectation is perturbed by 1e-3." << std::endl;
    }

    if (!eshkol_parity::test_comparator_rejects_a_perturbed_result()) {
        std::cerr << "The comparator control failed; no row below would mean anything."
                  << std::endl;
        return 1;
    }

    DeviceExecutor* exec = registerStableHLODeviceExecutor();
    std::string why;
    if (!exec || !exec->available(&why)) {
        std::cout << "\nNo PJRT device is reachable: " << why << std::endl;
        std::cout << "A training-step parity claim needs a device, so this is not a pass."
                  << std::endl;
        return 77;
    }
    std::cout << "\nDevice: " << exec->description() << std::endl;
    g_dtype = exec->dtypeName();
    setTolerancesForDtype(g_dtype);
    std::printf("Tolerance (%s, absolute or relative, whichever is looser; "
                "docs/design/ESHKOL_S_FRAGMENT.md): arithmetic=%g transcendental=%g\n",
                g_dtype.c_str(), toleranceFor(ToleranceClass::Arithmetic),
                toleranceFor(ToleranceClass::Transcendental));
    std::printf("Steps per configuration: K=%d (drift compounds through the moments)\n", kSteps);

    const std::vector<ShapeSpec> shapes = {
        {"b8xs16xd32xc8", 8, 16, 32, 8},
        {"b32xs64xd64xc8", 32, 64, 64, 8},
    };
    const std::vector<double> curvatures = {1.0, 0.5};

    printHeader();

    struct Report {
        std::string config;
        ConfigResult result;
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
            ConfigResult r = runConfig(exec, config, s, hyper, seed, first);
            first = false;
            gradeTrajectory(config, r);
            reports.push_back({config, r});
            seed++;
        }
    }

    // ---- cache control ----
    // Two curvatures at one shape must share one executable, and steps 2..K
    // must all be cache hits. Both are checked on the counters rather than on
    // values: a module with a curvature or a step index baked in would return
    // correct numbers the first time and silently wrong ones after.
    std::cout << "\nCache and timing (per configuration):" << std::endl;
    std::printf("%-22s %8s %8s %10s %14s %14s %8s\n",
                "config", "steps", "compiled", "cache_hits",
                "device_s/step", "host_s/step", "ratio");
    for (const Report& rep : reports) {
        const double ds = rep.result.device_seconds / kSteps;
        const double hs = rep.result.host_seconds / kSteps;
        std::printf("%-22s %8d %8llu %10llu %14.6f %14.6f %8.2f\n",
                    rep.config.c_str(), kSteps,
                    static_cast<unsigned long long>(rep.result.compiled),
                    static_cast<unsigned long long>(rep.result.cache_hits),
                    ds, hs, hs > 0.0 ? ds / hs : 0.0);
    }

    std::cout << "\nControl: the executable cache is keyed by shape alone... ";
    bool cache_ok = true;
    for (size_t i = 0; i < reports.size(); ++i) {
        const ConfigResult& r = reports[i].result;
        // The FIRST configuration at a shape compiles once; the second, which
        // differs only in curvature, must compile nothing at all.
        const uint64_t expected_compiles = (i % curvatures.size() == 0) ? 1u : 0u;
        if (r.compiled != expected_compiles) {
            std::cout << "\n  " << reports[i].config << " compiled " << r.compiled
                      << " executables, expected " << expected_compiles;
            cache_ok = false;
        }
        if (r.cache_hits < static_cast<uint64_t>(kSteps) - expected_compiles) {
            std::cout << "\n  " << reports[i].config << " served only " << r.cache_hits
                      << " of " << (kSteps - expected_compiles) << " steps from the cache";
            cache_ok = false;
        }
    }
    if (!cache_ok) g_controls_failed++;
    std::cout << (cache_ok ? "PASS" : " -> FAIL") << std::endl;

    const DeviceStats stats = exec->stats();
    std::printf("\nSUMMARY: rows_passed=%d rows_failed=%d constraints_passed=%d "
                "constraints_failed=%d trajectories_passed=%d trajectories_failed=%d "
                "controls_failed=%d steps=%d dtype=%s executed=%llu compiled=%llu "
                "cache_hits=%llu\n",
                g_rows_passed, g_rows_failed, g_constraints_passed, g_constraints_failed,
                g_trajectories_passed, g_trajectories_failed, g_controls_failed,
                kSteps, g_dtype.c_str(),
                static_cast<unsigned long long>(stats.executed),
                static_cast<unsigned long long>(stats.compiled),
                static_cast<unsigned long long>(stats.cache_hits));

    const bool all_ok = (g_rows_failed == 0) && (g_constraints_failed == 0) &&
                        (g_controls_failed == 0) && (g_trajectories_failed == 0) &&
                        (g_rows_passed > 0) && (g_constraints_passed > 0) &&
                        (g_trajectories_passed > 0);
    std::cout << (all_ok ? "TRAINING STEP PARITY: PASS" : "TRAINING STEP PARITY: FAIL")
              << std::endl;
    return all_ok ? 0 : 1;
}
