/**
 * @file training_checkpoint_test.cpp
 * @brief S9: checkpoint authority, preemption survival, and corrupt-checkpoint
 *         refusal for the mixed-curvature training step.
 *
 * Four rows, each FAIL-then-PASS where the brief asks for it:
 *
 *   1. round_trip_parity   -- save a mid-training state, load it into a FRESH
 *      process-local struct, and require every parameter/moment/step/seed
 *      byte-identical (memcmp, not a tolerance).
 *   2. trajectory_parity   -- K steps, checkpoint, K more steps vs. an
 *      uninterrupted 2K-step run from the same seed: the two loss sequences
 *      from the checkpoint onward must be bit-identical.
 *   3. kill_and_resume     -- fork/exec the REAL training_checkpoint_driver
 *      binary, SIGKILL it mid-run (a real OS process kill, not a simulated
 *      one), relaunch it against the same checkpoint directory, and diff its
 *      continuation against an uninterrupted reference run of the driver.
 *   4. corrupt_refusal     -- flip a byte in the newest checkpoint's payload
 *      and require both eshkol_training_checkpoint_is_valid() and a driver
 *      relaunch to refuse it and fall back to the previous checkpoint.
 *
 * Prints one `SUMMARY: rows_passed=N rows_failed=M` line the gate script
 * (scripts/run_xla_gate.sh stage_production) parses, matching the pattern
 * every other XLA-program stage test already uses.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/ml/training_checkpoint.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

namespace {

int g_rows_passed = 0;
int g_rows_failed = 0;

void report(const char* name, bool ok, const std::string& detail) {
    std::printf("%-24s %-4s  %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
    if (ok) g_rows_passed++; else g_rows_failed++;
}

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
    bool bit_equal(const Model& other) const {
        return w == other.w && p_hyp == other.p_hyp && p_sph == other.p_sph && p_euc == other.p_euc &&
              m_w == other.m_w && v_w == other.v_w && m_hyp == other.m_hyp && v_hyp == other.v_hyp &&
              m_sph == other.m_sph && v_sph == other.v_sph && m_euc == other.m_euc && v_euc == other.v_euc &&
              moments.step == other.moments.step;
    }
};

const EshkolMixedCurvatureShape kShape{8, 4, 3};

void run_steps(Model& m, const EshkolMixedCurvatureHyper& hyper, const double* batch,
              const double* targets, int64_t count, std::vector<double>* losses) {
    for (int64_t i = 0; i < count; ++i) {
        double loss = 0.0;
        bool ok = eshkol_mixed_curvature_train_step(m.shape, &hyper, &m.params, &m.moments, batch, targets,
                                                     m.scratch.data(), &loss, nullptr);
        if (!ok) { losses->push_back(-1.0); return; }
        if (losses) losses->push_back(loss);
    }
}

/** Test 1 + 2: round-trip byte parity and trajectory parity. */
void test_round_trip_and_trajectory(const std::string& dir) {
    EshkolMixedCurvatureHyper hyper;
    eshkol_mixed_curvature_default_hyper(&hyper);
    const uint64_t seed = 0xC0FFEEu;

    Model a;
    a.allocate(kShape);
    std::vector<double> batch(static_cast<size_t>(eshkol_mixed_curvature_x_elements(kShape)));
    std::vector<double> targets(static_cast<size_t>(eshkol_mixed_curvature_t_elements(kShape)));
    eshkol_mixed_curvature_init(kShape, &hyper, seed, &a.params, &a.moments, batch.data(), targets.data());

    std::vector<double> losses_first_5;
    run_steps(a, hyper, batch.data(), targets.data(), 5, &losses_first_5);

    std::string ckpt_path = dir + "/round_trip.eskm";
    EshkolTrainingCheckpointMeta meta{kShape, hyper, seed, ESHKOL_TRAINING_DTYPE_F32};
    bool saved = eshkol_training_checkpoint_save(ckpt_path.c_str(), &meta, &a.params, &a.moments);

    Model loaded;
    loaded.allocate(kShape);
    EshkolTrainingCheckpointMeta loaded_meta{};
    bool restored = saved && eshkol_training_checkpoint_load(ckpt_path.c_str(), &kShape, &loaded_meta,
                                                              &loaded.params, &loaded.moments);
    bool byte_exact = restored && a.bit_equal(loaded) && loaded_meta.seed == seed &&
                      loaded_meta.hyper.curvature == hyper.curvature;
    report("round_trip_parity", byte_exact,
          saved ? (restored ? "checkpoint restored byte-identical (params, moments, step, seed, curvature)"
                             : "load failed or content differed after save")
                : "save failed");

    // Trajectory: continue `a` (post-checkpoint, in-process) for 5 more steps,
    // and separately continue `loaded` (post-restore) for 5 more steps; then
    // compare BOTH against a fresh uninterrupted 10-step run from the same seed.
    std::vector<double> losses_continued_inplace, losses_continued_restored;
    run_steps(a, hyper, batch.data(), targets.data(), 5, &losses_continued_inplace);
    run_steps(loaded, hyper, batch.data(), targets.data(), 5, &losses_continued_restored);

    Model uninterrupted;
    uninterrupted.allocate(kShape);
    std::vector<double> batch2(batch.size()), targets2(targets.size());
    eshkol_mixed_curvature_init(kShape, &hyper, seed, &uninterrupted.params, &uninterrupted.moments,
                                batch2.data(), targets2.data());
    std::vector<double> losses_uninterrupted;
    run_steps(uninterrupted, hyper, batch2.data(), targets2.data(), 10, &losses_uninterrupted);

    bool traj_ok = (losses_uninterrupted.size() == 10) && (losses_continued_restored.size() == 5);
    for (size_t i = 0; i < 5 && traj_ok; ++i) {
        if (losses_continued_restored[i] != losses_uninterrupted[5 + i]) traj_ok = false;
        if (losses_continued_inplace[i] != losses_uninterrupted[5 + i]) traj_ok = false;
    }
    std::ostringstream detail;
    detail << "K=5 pre + K=5 post-restore vs K=10 uninterrupted, bit-identical from the restore point ("
          << (traj_ok ? "matched" : "DIVERGED") << ")";
    report("trajectory_parity", traj_ok, detail.str());
}

/** Runs the training_checkpoint_driver binary and returns its pid (does not wait). */
pid_t spawn_driver(const std::string& driver_path, const std::string& ckpt_dir, int64_t total_steps,
                   int64_t checkpoint_every, uint64_t seed, const std::string& out_path) {
    pid_t pid = fork();
    if (pid == 0) {
        FILE* out = std::freopen(out_path.c_str(), "w", stdout);
        (void)out;
        std::freopen("/dev/null", "w", stderr);
        execl(driver_path.c_str(), driver_path.c_str(), ckpt_dir.c_str(),
              std::to_string(total_steps).c_str(), std::to_string(checkpoint_every).c_str(),
              std::to_string(seed).c_str(), (char*)nullptr);
        _exit(127);
    }
    return pid;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/** Test 3: real SIGKILL mid-run, then relaunch and diff against a reference. */
void test_kill_and_resume(const std::string& driver_path, const std::string& scratch_dir) {
    if (driver_path.empty()) {
        report("kill_and_resume", false, "training_checkpoint_driver binary not found next to this test");
        return;
    }
    const uint64_t seed = 13579u;
    const int64_t total = 4000000;
    const int64_t every = 200000;

    std::string ref_dir = scratch_dir + "/kr_ref";
    std::string kill_dir = scratch_dir + "/kr_kill";
    (void)!system(("mkdir -p '" + ref_dir + "' '" + kill_dir + "'").c_str());

    // Reference: uninterrupted full run.
    pid_t ref_pid = spawn_driver(driver_path, ref_dir, total, total /* one checkpoint at the end only */,
                                 seed, scratch_dir + "/kr_ref.out");
    int ref_status = 0;
    waitpid(ref_pid, &ref_status, 0);

    // Kill run: launch, SIGKILL partway through, then relaunch to completion.
    pid_t kill_pid = spawn_driver(driver_path, kill_dir, total, every, seed, scratch_dir + "/kr_kill1.out");
    usleep(400000); // let it run for a bit -- real wall-clock, not simulated
    int killed = (kill(kill_pid, SIGKILL) == 0);
    int kill_status = 0;
    waitpid(kill_pid, &kill_status, 0);
    bool was_killed = killed && WIFSIGNALED(kill_status) && WTERMSIG(kill_status) == SIGKILL;

    pid_t resume_pid = spawn_driver(driver_path, kill_dir, total, every, seed, scratch_dir + "/kr_kill2.out");
    int resume_status = 0;
    waitpid(resume_pid, &resume_status, 0);
    bool resume_ok = WIFEXITED(resume_status) && WEXITSTATUS(resume_status) == 0;

    std::string ref_out = read_file(scratch_dir + "/kr_ref.out");
    std::string resumed_tail = read_file(scratch_dir + "/kr_kill2.out");

    // The resumed run's tail (its own last line: "<total> <loss>") must equal
    // the reference run's last line -- same seed, same total steps, same
    // deterministic step function, so the endpoint is exactly reproducible
    // regardless of where the checkpoint boundary fell.
    auto last_line = [](const std::string& s) -> std::string {
        size_t end = s.find_last_not_of('\n');
        if (end == std::string::npos) return "";
        size_t start = s.find_last_of('\n', end);
        return s.substr(start == std::string::npos ? 0 : start + 1, end - (start == std::string::npos ? 0 : start));
    };
    std::string ref_last = last_line(ref_out);
    std::string resumed_last = last_line(resumed_tail);
    bool endpoints_match = !ref_last.empty() && ref_last == resumed_last;

    bool ok = was_killed && resume_ok && endpoints_match;
    std::ostringstream detail;
    detail << "SIGKILL delivered=" << (was_killed ? "yes" : "no") << ", resumed process exited "
          << (resume_ok ? "0" : "nonzero") << ", final step/loss ref='" << ref_last << "' resumed='"
          << resumed_last << "'";
    report("kill_and_resume", ok, detail.str());
}

/** Test 4: corrupt the newest checkpoint and require refusal + fallback. */
void test_corrupt_refusal(const std::string& driver_path, const std::string& scratch_dir) {
    if (driver_path.empty()) {
        report("corrupt_refusal", false, "training_checkpoint_driver binary not found next to this test");
        return;
    }
    std::string dir = scratch_dir + "/corrupt";
    (void)!system(("mkdir -p '" + dir + "'").c_str());

    pid_t p1 = spawn_driver(driver_path, dir, 10, 5, 24601u, scratch_dir + "/corrupt1.out");
    int s1 = 0;
    waitpid(p1, &s1, 0);

    std::string newest = dir + "/ckpt.10.eskm";
    bool valid_before = eshkol_training_checkpoint_is_valid(newest.c_str(), &kShape);

    {
        std::fstream f(newest, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(40);
        char byte = 0;
        f.read(&byte, 1);
        f.seekp(40);
        byte ^= static_cast<char>(0xFF);
        f.write(&byte, 1);
    }
    bool valid_after = eshkol_training_checkpoint_is_valid(newest.c_str(), &kShape);

    // Relaunch: the driver must REFUSE ckpt.10 and fall back to ckpt.5, then
    // continue and finish -- proven by stderr naming the refusal and exit 0.
    std::string err_path = scratch_dir + "/corrupt2.err";
    pid_t p2 = fork();
    if (p2 == 0) {
        std::freopen((scratch_dir + "/corrupt2.out").c_str(), "w", stdout);
        std::freopen(err_path.c_str(), "w", stderr);
        execl(driver_path.c_str(), driver_path.c_str(), dir.c_str(), "10", "5", "24601", (char*)nullptr);
        _exit(127);
    }
    int s2 = 0;
    waitpid(p2, &s2, 0);
    bool relaunch_ok = WIFEXITED(s2) && WEXITSTATUS(s2) == 0;
    std::string err_text = read_file(err_path);
    bool named_refusal = err_text.find("REFUSED corrupt/invalid checkpoint") != std::string::npos &&
                         err_text.find("ckpt.10.eskm") != std::string::npos;
    bool fell_back_to_5 = err_text.find("RESUMED from") != std::string::npos &&
                          err_text.find("ckpt.5.eskm") != std::string::npos;

    bool ok = valid_before && !valid_after && relaunch_ok && named_refusal && fell_back_to_5;
    std::ostringstream detail;
    detail << "valid_before=" << valid_before << " valid_after_corruption=" << valid_after
          << " relaunch_exit_0=" << relaunch_ok << " named_the_refused_file=" << named_refusal
          << " fell_back_to_previous=" << fell_back_to_5;
    report("corrupt_refusal", ok, detail.str());
}

} // namespace

int main(int argc, char** argv) {
    // Line-buffer stdout regardless of whether it is a tty: this binary
    // fork()s (spawn_driver) with buffered output already pending, and an
    // fdopen'd child that later closes/reopens its own stdout would
    // otherwise flush THIS process's still-buffered lines a second time
    // into the shared inherited descriptor.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string scratch_dir = (argc > 1) ? argv[1] : ".";
    std::string driver_path = (argc > 2) ? argv[2] : "";

    (void)!system(("mkdir -p '" + scratch_dir + "'").c_str());

    test_round_trip_and_trajectory(scratch_dir);
    test_kill_and_resume(driver_path, scratch_dir);
    test_corrupt_refusal(driver_path, scratch_dir);

    std::printf("SUMMARY: rows_passed=%d rows_failed=%d\n", g_rows_passed, g_rows_failed);
    std::printf("TRAINING CHECKPOINT: %s\n", g_rows_failed == 0 ? "PASS" : "FAIL");
    return g_rows_failed == 0 ? 0 : 1;
}
