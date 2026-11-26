/*
 * CE-12 (v2.0-starlight) Native Verification Suite:
 *   1. Unified `differentiate` Operator & Mode Disagreement Control (Guard 1)
 *   2. Exact ZX-Calculus Quantum Interference & Singularity Control (Guard 2)
 *   3. Real Lean 4.12.0 Kernel Re-Verification & 1-Bit Drift Control (Guard 3)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/starlight_unified_diff_zx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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

constexpr double kPi = 3.141592653589793238462643383279502884;

std::string resolve_lean4_binary() {
    if (const char* env_bin = std::getenv("ESHKOL_LEAN4_BIN")) {
        if (env_bin[0] != '\0') return std::string(env_bin);
    }
    if (const char* home = std::getenv("HOME")) {
        std::string elan_lean = std::string(home) + "/.elan/bin/lean";
        std::ifstream f(elan_lean);
        if (f.good()) return elan_lean;
    }
    return "lean";
}

std::string temp_dir_prefix() {
    if (const char* tmp = std::getenv("TMPDIR")) {
        if (tmp[0] != '\0') {
            std::string s(tmp);
            if (s.back() != '/') s.push_back('/');
            return s;
        }
    }
    return "/tmp/";
}

/* ============================================================================
 * Guard 1: Unified `differentiate` Cross-Mode Agreement & Kink Disagreement
 * ============================================================================ */

eshkol_udiff_result_t test_unified_differentiate() {
    std::printf("\n=== CE-12 Part 1: Unified `differentiate` Operator (3 Modes) ===\n");

    /* 1a. Linear DBSP belief operator: all 3 modes match to machine epsilon. */
    eshkol_udiff_spec_t lin_spec{ ESHKOL_UDIFF_FN_LINEAR_DBSP, 2.75, -0.40, 0.0 };
    eshkol_udiff_result_t lin_res{};
    CHECK(eshkol_udiff_eval_unified(&lin_spec, 1.3, 1e-10, &lin_res) == ESHKOL_CE12_OK,
          "linear DBSP unified differentiate returns ESHKOL_CE12_OK");
    CHECK(lin_res.fd_evals == 0 && lin_res.max_mode_disagreement < 1e-14,
          "linear DBSP all 3 modes agree < 1e-14 with 0 FD evals");

    /* 1b. Smooth nonlinear contraction equilibrium: z* = 0.5*tanh(z*) + 1.2*x + 0.1 */
    eshkol_udiff_spec_t eq_spec{ ESHKOL_UDIFF_FN_SMOOTH_EQUILIBRIUM, 1.2, 0.1, 0.5 };
    eshkol_udiff_result_t eq_res{};
    CHECK(eshkol_udiff_eval_unified(&eq_spec, 0.45, 1e-10, &eq_res) == ESHKOL_CE12_OK,
          "smooth equilibrium unified differentiate returns ESHKOL_CE12_OK");
    CHECK(eq_res.fd_evals == 0 && eq_res.max_mode_disagreement < 1e-10,
          "smooth equilibrium: Numeric AD, Implicit IFT, and DBSP Delta agree < 1e-10");

    /* 1c. Smooth side of abs(x - 0.5) away from kink: all 3 modes agree. */
    eshkol_udiff_spec_t kink_spec{ ESHKOL_UDIFF_FN_ABS_KINK, 1.5, 0.2, 0.5 };
    eshkol_udiff_result_t off_kink{};
    CHECK(eshkol_udiff_eval_unified(&kink_spec, 0.9, 1e-10, &off_kink) == ESHKOL_CE12_OK,
          "away from kink x=0.9: all 3 modes agree cleanly");

    /* --- GUARD 1: Mode Disagreement Control at non-smooth kink x = c = 0.5 --- */
    eshkol_udiff_result_t at_kink{};
    eshkol_ce12_status_t st_kink = eshkol_udiff_eval_unified(&kink_spec, 0.5, 1e-10, &at_kink);

    std::printf("  [Guard 1] Smooth equilibrium 3-mode disagreement=%.3e (< 1e-10, fd=0) | "
                "Kink x=0.5 status=%d (expected %d=MODE_DISAGREEMENT), "
                "num=%.2f, imp=%.2f, dbsp=%.2f\n",
                eq_res.max_mode_disagreement,
                (int)st_kink, (int)ESHKOL_UDIFF_ERR_MODE_DISAGREEMENT,
                at_kink.deriv_numeric, at_kink.deriv_implicit_fp, at_kink.deriv_dbsp_delta);

    CHECK(st_kink == ESHKOL_UDIFF_ERR_MODE_DISAGREEMENT,
          "Guard 1: kink at x=0.5 rejected with ESHKOL_UDIFF_ERR_MODE_DISAGREEMENT");
    CHECK(at_kink.max_mode_disagreement > 1.0 && at_kink.fd_evals == 0,
          "Guard 1: kink disagreement explicitly reported (> 1.0) with 0 FD evals");

    return eq_res;
}

/* ============================================================================
 * Guard 2: ZX-Calculus Quantum Cognitive Interference & Singularity Control
 * ============================================================================ */

eshkol_zx_eval_result_t test_zx_calculus_interference() {
    std::printf("\n=== CE-12 Part 2: Exact ZX-Calculus Quantum Cognitive Interference ===\n");

    /* Smooth non-singular ZX interference circuit. */
    eshkol_zx_circuit_t circuit{ 0.7071067811865476, 0.7071067811865476, 0.40 };
    eshkol_zx_eval_result_t smooth_res{};
    CHECK(eshkol_zx_eval_interference(&circuit, 0.85, 1e-7, &smooth_res) == ESHKOL_CE12_OK,
          "smooth ZX interference evaluates cleanly");

    const double shift_err = std::fabs(smooth_res.d_prob_param_shift - smooth_res.d_prob_analytical);
    CHECK(smooth_res.fd_evals == 0 && shift_err < 1e-10,
          "ZX parameter-shift gradient matches exact analytical derivative < 1e-10 (fd=0)");
    CHECK(std::isfinite(smooth_res.d_phase_analytical) && smooth_res.amplitude_norm > 0.1,
          "smooth ZX circuit has finite phase derivative and positive amplitude norm");

    /* --- GUARD 2: ZX-Calculus Destructive Interference Singularity Control --- */
    /* Construct exact destructive phase interference: 2*theta - phi = pi => theta = (phi + pi)/2. */
    const double theta_sing = 0.5 * (circuit.phi + kPi);
    eshkol_zx_eval_result_t sing_res{};
    eshkol_ce12_status_t st_sing = eshkol_zx_eval_interference(&circuit, theta_sing, 1e-7, &sing_res);

    std::printf("  [Guard 2] Smooth ZX param_shift_err=%.3e (< 1e-10, fd=0), norm=%.4f | "
                "Destructive singularity theta=%.4f status=%d (expected %d=INTERFERENCE_SINGULARITY), "
                "|psi|=%.3e\n",
                shift_err, smooth_res.amplitude_norm, theta_sing,
                (int)st_sing, (int)ESHKOL_ZX_ERR_INTERFERENCE_SINGULARITY,
                sing_res.amplitude_norm);

    CHECK(st_sing == ESHKOL_ZX_ERR_INTERFERENCE_SINGULARITY,
          "Guard 2: destructive ZX interference singularity rejected with ESHKOL_ZX_ERR_INTERFERENCE_SINGULARITY");
    CHECK(sing_res.amplitude_norm < 1e-12 && std::isnan(sing_res.d_phase_analytical),
          "Guard 2: amplitude collapses to ~0 and phase singularity is flagged");

    return smooth_res;
}

/* ============================================================================
 * Guard 3: Lean 4.12.0 Kernel Re-Verification & 1-Bit Flip Drift Control
 * ============================================================================ */

void test_starlight_lean4_reverification(const eshkol_udiff_result_t& udiff_res,
                                         const eshkol_zx_eval_result_t& zx_res,
                                         const char* export_lean_path) {
    std::printf("\n=== CE-12 Part 3: Starlight Lean 4.12.0 Kernel Re-Verification ===\n");

    eshkol_starlight_summary_t summary{};
    summary.lob_rank_before = 10U;
    summary.lob_rank_after = 8U;
    summary.linear_resource_uses = 1U;
    summary.erased_runtime_uses = 0U;
    summary.cubical_path_acyclic = true;
    summary.udiff_numeric = udiff_res.deriv_numeric;
    summary.udiff_implicit_fp = udiff_res.deriv_implicit_fp;
    summary.udiff_dbsp_delta = udiff_res.deriv_dbsp_delta;
    summary.zx_prob = zx_res.prob;
    summary.zx_d_prob_analytical = zx_res.d_prob_analytical;
    summary.zx_d_prob_param_shift = zx_res.d_prob_param_shift;
    summary.zx_amplitude_norm = zx_res.amplitude_norm;

    std::vector<char> lean_buf(16384, '\0');
    size_t lean_len = 0;
    uint8_t artifact_sha[32]{};
    CHECK(eshkol_starlight_export_lean4(&summary, "EshkolCE12Starlight",
                                        lean_buf.data(), lean_buf.size(),
                                        &lean_len, artifact_sha),
          "Starlight Lean 4.12.0 proof exported");

    std::string lean_src(lean_buf.data(), lean_len);
    CHECK(lean_src.find("sorry") == std::string::npos, "Guard 3: exported Lean 4 has ZERO sorry");
    CHECK(lean_src.find("axiom ") == std::string::npos, "Guard 3: exported Lean 4 has ZERO custom axioms");

    if (export_lean_path && export_lean_path[0] != '\0') {
        std::ofstream out(export_lean_path, std::ios::binary | std::ios::trunc);
        out.write(lean_src.data(), static_cast<std::streamsize>(lean_src.size()));
        out.close();
    }

    const std::string lean_bin = resolve_lean4_binary();
    const std::string valid_path = temp_dir_prefix() + "eshkol_ce12_starlight_valid.lean";
    const std::string drift_path = temp_dir_prefix() + "eshkol_ce12_starlight_1bit_drift.lean";

    {
        std::ofstream out(valid_path, std::ios::binary | std::ios::trunc);
        out.write(lean_src.data(), static_cast<std::streamsize>(lean_src.size()));
    }

    /* Valid Lean 4.12.0 kernel re-verification MUST succeed (exit 0 + bit-exact SHA-256). */
    int valid_rc = -1;
    eshkol_ce12_status_t st_valid = eshkol_starlight_verify_lean4_artifact(
        valid_path.c_str(), artifact_sha, lean_bin.c_str(), &valid_rc);
    CHECK(st_valid == ESHKOL_CE12_OK && valid_rc == 0,
          "Guard 3: valid Starlight Lean 4.12.0 proof passes kernel re-verification (rc=0)");

    /* --- GUARD 3: 1-Bit Flip Kernel Re-Verification Drift Control --- */
    /* Flip 1 single bit in `lobRankAfter := 8` (ASCII '8'=0x38 -> XOR 0x01 = '9'=0x39, or
     * better yet flip 1 bit in `erasedRuntimeUses := 0` ('0'=0x30 -> XOR 0x01 = '1'=0x31)
     * so BOTH the bit-exact SHA-256 check AND the Lean 4 `by decide` kernel reject!). */
    std::string drifted = lean_src;
    size_t pos = drifted.find("erasedRuntimeUses := 0");
    CHECK(pos != std::string::npos, "Guard 3: located erasedRuntimeUses field for 1-bit flip");
    if (pos != std::string::npos) {
        size_t digit_idx = pos + std::strlen("erasedRuntimeUses := ");
        drifted[digit_idx] = static_cast<char>(drifted[digit_idx] ^ 0x01); /* Exact 1-bit flip: '0' -> '1' */
    }
    {
        std::ofstream out(drift_path, std::ios::binary | std::ios::trunc);
        out.write(drifted.data(), static_cast<std::streamsize>(drifted.size()));
    }

    int drift_rc = 0;
    eshkol_ce12_status_t st_drift = eshkol_starlight_verify_lean4_artifact(
        drift_path.c_str(), artifact_sha, lean_bin.c_str(), &drift_rc);

    std::remove(valid_path.c_str());
    std::remove(drift_path.c_str());

    std::printf("  [Guard 3] Lean 4.12.0 kernel re-verify: valid_status=%d (rc=%d) | "
                "1-bit-flip status=%d (expected %d=KERNEL_DRIFT, lean_rc=%d != 0), bytes=%zu\n",
                (int)st_valid, valid_rc,
                (int)st_drift, (int)ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT,
                drift_rc, lean_len);

    CHECK(st_drift == ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT,
          "Guard 3: 1-bit flip rejected fail-closed with ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT");
    CHECK(drift_rc != 0,
          "Guard 3: Lean 4.12.0 kernel independently rejected 1-bit-flipped theorem (rc != 0)");
}

} /* namespace */

int main(int argc, char** argv) {
    const char* export_lean_path = nullptr;
    if (argc >= 3 && std::strcmp(argv[1], "--export-lean") == 0) {
        export_lean_path = argv[2];
    }

    eshkol_udiff_result_t udiff_res = test_unified_differentiate();
    eshkol_zx_eval_result_t zx_res = test_zx_calculus_interference();
    test_starlight_lean4_reverification(udiff_res, zx_res, export_lean_path);

    std::printf("\nCE-12 Summary: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
