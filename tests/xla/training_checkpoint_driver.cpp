/**
 * @file training_checkpoint_driver.cpp
 * @brief Standalone S9 preemption-survival driver.
 *
 * A real, separately-exec'd process (not a thread, not a fork of the test
 * harness) so that the SIGKILL test in training_checkpoint_test.cpp exercises
 * an actual OS process kill and restart -- SIGKILL cannot be caught, which is
 * the entire point of the test: the driver gets no chance to react, and
 * correctness has to come from what was already durably on disk.
 *
 * Usage: training_checkpoint_driver <ckpt_dir> <total_steps> <checkpoint_every> <seed>
 *
 * On start, looks for the newest checkpoint under <ckpt_dir> (files named
 * ckpt.<step>.eskm), validates it (manifest + payload CRC agree, shapes
 * match), and falls back to the next-newest on a refusal, repeating until one
 * validates or none remain -- then trains from step 0. Checkpoints every
 * <checkpoint_every> steps. Installs a SIGTERM handler that finishes the
 * in-flight step, checkpoints, and exits 0 (the "PJRT device loss" path: a
 * graceful preemption notice, distinct from the ungraceful SIGKILL path this
 * binary cannot do anything about by construction).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/ml/training_checkpoint.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

namespace {

volatile sig_atomic_t g_sigterm_received = 0;

void handle_sigterm(int) { g_sigterm_received = 1; }

struct Model {
    EshkolMixedCurvatureShape shape{};
    std::vector<double> w, p_hyp, p_sph, p_euc;
    std::vector<double> m_w, v_w, m_hyp, v_hyp, m_sph, v_sph, m_euc, v_euc;
    std::vector<double> scratch;
    EshkolMixedCurvatureParams params{};
    EshkolMixedCurvatureMoments moments{};

    void allocate(EshkolMixedCurvatureShape s) {
        shape = s;
        size_t we = static_cast<size_t>(eshkol_mixed_curvature_w_elements(s));
        size_t pe = static_cast<size_t>(eshkol_mixed_curvature_p_elements(s));
        w.assign(we, 0.0); m_w.assign(we, 0.0); v_w.assign(we, 0.0);
        p_hyp.assign(pe, 0.0); p_sph.assign(pe, 0.0); p_euc.assign(pe, 0.0);
        m_hyp.assign(pe, 0.0); v_hyp.assign(pe, 0.0);
        m_sph.assign(pe, 0.0); v_sph.assign(pe, 0.0);
        m_euc.assign(pe, 0.0); v_euc.assign(pe, 0.0);
        scratch.assign(static_cast<size_t>(eshkol_mixed_curvature_scratch_elements(s)), 0.0);
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
};

/** Candidate checkpoint file names under dir, newest step first. */
std::vector<std::string> list_checkpoints_newest_first(const std::string& dir) {
    std::vector<std::pair<int64_t, std::string>> found;
    DIR* d = opendir(dir.c_str());
    if (!d) return {};
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name.rfind("ckpt.", 0) == 0 && name.size() > 5 &&
            name.size() >= 6 && name.compare(name.size() - 5, 5, ".eskm") == 0) {
            std::string step_part = name.substr(5, name.size() - 5 - 5);
            char* end = nullptr;
            long long step = std::strtoll(step_part.c_str(), &end, 10);
            if (end && *end == '\0') found.push_back({step, dir + "/" + name});
        }
    }
    closedir(d);
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::string> paths;
    for (auto& p : found) paths.push_back(p.second);
    return paths;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <ckpt_dir> <total_steps> <checkpoint_every> <seed>\n", argv[0]);
        return 2;
    }
    std::string ckpt_dir = argv[1];
    int64_t total_steps = std::atoll(argv[2]);
    int64_t checkpoint_every = std::atoll(argv[3]);
    uint64_t seed = static_cast<uint64_t>(std::strtoull(argv[4], nullptr, 10));

    std::signal(SIGTERM, handle_sigterm);

    EshkolMixedCurvatureShape shape{8, 4, 3};
    EshkolMixedCurvatureHyper hyper;
    eshkol_mixed_curvature_default_hyper(&hyper);

    Model model;
    model.allocate(shape);
    std::vector<double> batch(static_cast<size_t>(eshkol_mixed_curvature_x_elements(shape)));
    std::vector<double> targets(static_cast<size_t>(eshkol_mixed_curvature_t_elements(shape)));

    int64_t resume_step = 0;
    bool resumed = false;
    for (const std::string& candidate : list_checkpoints_newest_first(ckpt_dir)) {
        EshkolTrainingCheckpointMeta meta{};
        if (eshkol_training_checkpoint_load(candidate.c_str(), &shape, &meta, &model.params, &model.moments)) {
            hyper.curvature = meta.hyper.curvature;
            seed = meta.seed;
            resume_step = model.moments.step;
            resumed = true;
            std::fprintf(stderr, "RESUMED from %s at step %" PRId64 "\n", candidate.c_str(), resume_step);
            break;
        }
        std::fprintf(stderr, "REFUSED corrupt/invalid checkpoint %s, trying previous\n", candidate.c_str());
    }
    /* The batch/targets are regenerated from the seed on every launch --
     * init() is the ONLY thing that consumes the seed, and it draws params
     * and batch from ONE running stream in that order, so the batch this
     * call produces only matches the original run's batch if params are
     * drawn too (passing null for params SKIPS those draws and desyncs the
     * stream, silently producing a different batch). So init() always runs
     * against a throwaway model; on a resume its params/moments are
     * discarded in favor of the loaded checkpoint, and only batch/targets
     * are kept. */
    Model init_scratch;
    init_scratch.allocate(shape);
    eshkol_mixed_curvature_init(shape, &hyper, seed, &init_scratch.params, &init_scratch.moments,
                               batch.data(), targets.data());
    if (!resumed) {
        model.w = init_scratch.w; model.p_hyp = init_scratch.p_hyp;
        model.p_sph = init_scratch.p_sph; model.p_euc = init_scratch.p_euc;
        model.m_w = init_scratch.m_w; model.v_w = init_scratch.v_w;
        model.m_hyp = init_scratch.m_hyp; model.v_hyp = init_scratch.v_hyp;
        model.m_sph = init_scratch.m_sph; model.v_sph = init_scratch.v_sph;
        model.m_euc = init_scratch.m_euc; model.v_euc = init_scratch.v_euc;
        model.moments.step = init_scratch.moments.step;
        model.rebind();
    }

    for (int64_t step = resume_step; step < total_steps; ++step) {
        double loss = 0.0;
        bool ok = eshkol_mixed_curvature_train_step(shape, &hyper, &model.params, &model.moments,
                                                     batch.data(), targets.data(), model.scratch.data(),
                                                     &loss, nullptr);
        if (!ok) {
            std::fprintf(stderr, "train_step refused at step %" PRId64 "\n", step);
            return 1;
        }
        std::printf("%" PRId64 " %.17g\n", model.moments.step, loss);
        std::fflush(stdout);

        bool should_checkpoint = ((step + 1) % checkpoint_every == 0) || g_sigterm_received;
        if (should_checkpoint) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/ckpt.%" PRId64 ".eskm", ckpt_dir.c_str(), model.moments.step);
            EshkolTrainingCheckpointMeta meta{shape, hyper, seed, ESHKOL_TRAINING_DTYPE_F32};
            if (!eshkol_training_checkpoint_save(path, &meta, &model.params, &model.moments)) {
                std::fprintf(stderr, "checkpoint save FAILED at step %" PRId64 "\n", model.moments.step);
                return 1;
            }
            std::fprintf(stderr, "checkpointed step %" PRId64 " -> %s\n", model.moments.step, path);
        }
        if (g_sigterm_received) {
            std::fprintf(stderr, "SIGTERM: checkpointed and exiting cleanly\n");
            return 0;
        }
    }
    return 0;
}
