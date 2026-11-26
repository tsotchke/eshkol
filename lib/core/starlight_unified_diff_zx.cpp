/*
 * Starlight (v2.0) Unified Differentiate Operator, Exact ZX-Calculus Quantum
 * Cognitive Interference Engine, and Lean 4.12.0 Kernel Re-Verification Pipeline
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/starlight_unified_diff_zx.h"
#include "eshkol/core/cert_chain_v2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::string hex_encode(const uint8_t* data, size_t len) {
    static const char kHex[] = "0123456789abcdef";
    std::string s;
    s.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        s.push_back(kHex[(data[i] >> 4) & 0x0Fu]);
        s.push_back(kHex[data[i] & 0x0Fu]);
    }
    return s;
}

std::string format_lean_int(int64_t v) {
    char tmp[64];
    if (v < 0) {
        std::snprintf(tmp, sizeof(tmp), "(%lld : Int)", (long long)v);
    } else {
        std::snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
    }
    return std::string(tmp);
}

} /* namespace */

extern "C" {

eshkol_ce12_status_t eshkol_udiff_eval_unified(
    const eshkol_udiff_spec_t* spec,
    double x,
    double consistency_tol,
    eshkol_udiff_result_t* out_res) {
    if (!spec || !out_res || !std::isfinite(x) ||
        !std::isfinite(consistency_tol) || consistency_tol <= 0.0) {
        return ESHKOL_CE12_ERR_INVALID_ARG;
    }

    std::memset(out_res, 0, sizeof(*out_res));
    out_res->fd_evals = 0;

    switch (spec->kind) {
        case ESHKOL_UDIFF_FN_LINEAR_DBSP: {
            const double val = spec->a * x + spec->b;
            /* Numeric AD: exact dual slope a. */
            const double d_num = spec->a;
            /* Implicit equilibrium: F(z, x) = z - (a*x + b) = 0 -> -(dF/dz)^-1 (dF/dx) = a. */
            const double d_imp = spec->a;
            /* DBSP linear belief delta operator: Delta(a*x + b) / Delta(x) = a. */
            const double d_dbsp = spec->a;

            out_res->value = val;
            out_res->deriv_numeric = d_num;
            out_res->deriv_implicit_fp = d_imp;
            out_res->deriv_dbsp_delta = d_dbsp;
            break;
        }

        case ESHKOL_UDIFF_FN_SMOOTH_EQUILIBRIUM: {
            /* Contraction equilibrium z* = c * tanh(z*) + a * x + b, requiring |c| < 1. */
            if (!std::isfinite(spec->c) || std::fabs(spec->c) >= 0.999) {
                return ESHKOL_CE12_ERR_INVALID_ARG;
            }
            /* Mode 1 (Numeric AD): unrolled dual-number Picard iteration (z, dz/dx). */
            double z_num = spec->a * x + spec->b;
            double dz_num = spec->a;
            for (int it = 0; it < 120; ++it) {
                double th = std::tanh(z_num);
                double sech2 = 1.0 - th * th;
                double next_z = spec->c * th + spec->a * x + spec->b;
                double next_dz = spec->c * sech2 * dz_num + spec->a;
                double step = std::fabs(next_z - z_num) + std::fabs(next_dz - dz_num);
                z_num = next_z;
                dz_num = next_dz;
                if (step < 1e-15) break;
            }

            /* Mode 2 (Implicit Equilibrium IFT): Newton solve F(z, x) = 0, then exact IFT. */
            double z_imp = spec->a * x + spec->b;
            for (int it = 0; it < 40; ++it) {
                double th = std::tanh(z_imp);
                double sech2 = 1.0 - th * th;
                double F = z_imp - spec->c * th - spec->a * x - spec->b;
                double dF_dz = 1.0 - spec->c * sech2;
                double delta = F / dF_dz;
                z_imp -= delta;
                if (std::fabs(delta) < 1e-15) break;
            }
            double th_imp = std::tanh(z_imp);
            double dF_dz = 1.0 - spec->c * (1.0 - th_imp * th_imp);
            double d_imp = spec->a / dF_dz;

            /* Mode 3 (DBSP Incremental Belief Delta): solve linearized tangent circuit
             * Delta z = c * sech^2(z*) * Delta z + a * Delta x via Neumann series. */
            double lam = spec->c * (1.0 - th_imp * th_imp);
            double d_dbsp = 0.0;
            double term = spec->a;
            for (int k = 0; k < 120; ++k) {
                d_dbsp += term;
                term *= lam;
                if (std::fabs(term) < 1e-16) break;
            }

            out_res->value = z_imp;
            out_res->deriv_numeric = dz_num;
            out_res->deriv_implicit_fp = d_imp;
            out_res->deriv_dbsp_delta = d_dbsp;
            break;
        }

        case ESHKOL_UDIFF_FN_ABS_KINK: {
            /* f(x) = a * |x - c| + b: non-smooth kink at x = c. */
            const double diff = x - spec->c;
            out_res->value = spec->a * std::fabs(diff) + spec->b;

            if (std::fabs(diff) <= 1e-9) {
                /*
                 * At the kink x = c, right-sided Numeric AD subgradient (+a),
                 * left-sided Implicit proximal subgradient (-a), and symmetric
                 * DBSP Clarke midpoint (0.0) disagree cleanly!
                 */
                out_res->deriv_numeric = spec->a;
                out_res->deriv_implicit_fp = -spec->a;
                out_res->deriv_dbsp_delta = 0.0;
            } else {
                const double sgn = (diff > 0.0) ? 1.0 : -1.0;
                out_res->deriv_numeric = spec->a * sgn;
                out_res->deriv_implicit_fp = spec->a * sgn;
                out_res->deriv_dbsp_delta = spec->a * sgn;
            }
            break;
        }

        default:
            return ESHKOL_CE12_ERR_INVALID_ARG;
    }

    const double d01 = std::fabs(out_res->deriv_numeric - out_res->deriv_implicit_fp);
    const double d02 = std::fabs(out_res->deriv_numeric - out_res->deriv_dbsp_delta);
    const double d12 = std::fabs(out_res->deriv_implicit_fp - out_res->deriv_dbsp_delta);
    out_res->max_mode_disagreement = std::max(d01, std::max(d02, d12));

    if (out_res->max_mode_disagreement > consistency_tol) {
        return ESHKOL_UDIFF_ERR_MODE_DISAGREEMENT;
    }
    return ESHKOL_CE12_OK;
}

eshkol_ce12_status_t eshkol_zx_eval_interference(
    const eshkol_zx_circuit_t* circuit,
    double theta,
    double singularity_eps,
    eshkol_zx_eval_result_t* out_res) {
    if (!circuit || !out_res || !std::isfinite(theta) ||
        !std::isfinite(circuit->w1) || !std::isfinite(circuit->w2) ||
        !std::isfinite(circuit->phi) || !std::isfinite(singularity_eps) ||
        singularity_eps <= 0.0) {
        return ESHKOL_CE12_ERR_INVALID_ARG;
    }

    std::memset(out_res, 0, sizeof(*out_res));
    out_res->fd_evals = 0;

    auto eval_prob = [circuit](double t) {
        double a1 = t;
        double a2 = circuit->phi - t;
        double re = circuit->w1 * std::cos(a1) + circuit->w2 * std::cos(a2);
        double im = circuit->w1 * std::sin(a1) + circuit->w2 * std::sin(a2);
        return re * re + im * im;
    };

    const double a1 = theta;
    const double a2 = circuit->phi - theta;
    const double re = circuit->w1 * std::cos(a1) + circuit->w2 * std::cos(a2);
    const double im = circuit->w1 * std::sin(a1) + circuit->w2 * std::sin(a2);
    const double prob = re * re + im * im;
    const double norm = std::sqrt(prob);

    /* Exact analytical derivative of psi(theta) = w1*e^{i*theta} + w2*e^{i*(phi-theta)}. */
    const double d_re = -circuit->w1 * std::sin(a1) + circuit->w2 * std::sin(a2);
    const double d_im =  circuit->w1 * std::cos(a1) - circuit->w2 * std::cos(a2);

    const double d_prob_analytic = 2.0 * (re * d_re + im * d_im);
    /* Exact ZX parameter-shift rule for frequency-2 interference term: P(t + pi/4) - P(t - pi/4). */
    const double d_prob_shift = eval_prob(theta + kPi * 0.25) - eval_prob(theta - kPi * 0.25);

    out_res->psi_re = re;
    out_res->psi_im = im;
    out_res->prob = prob;
    out_res->amplitude_norm = norm;
    out_res->d_prob_analytical = d_prob_analytic;
    out_res->d_prob_param_shift = d_prob_shift;

    /* Detect destructive ZX-calculus phase interference singularity (|psi| -> 0). */
    if (norm < singularity_eps) {
        out_res->d_phase_analytical = std::numeric_limits<double>::quiet_NaN();
        return ESHKOL_ZX_ERR_INTERFERENCE_SINGULARITY;
    }

    out_res->d_phase_analytical = (re * d_im - im * d_re) / prob;
    return ESHKOL_CE12_OK;
}

bool eshkol_starlight_export_lean4(
    const eshkol_starlight_summary_t* summary,
    const char* module_namespace,
    char* out_buf,
    size_t out_cap,
    size_t* out_len,
    uint8_t out_artifact_sha256[32]) {
    if (!summary || !module_namespace || !out_buf || out_cap == 0 || !out_artifact_sha256) {
        return false;
    }

    const double kScale = 1e9;
    const int64_t u_num = static_cast<int64_t>(std::llround(summary->udiff_numeric * kScale));
    const int64_t u_imp = static_cast<int64_t>(std::llround(summary->udiff_implicit_fp * kScale));
    const int64_t u_dbsp = static_cast<int64_t>(std::llround(summary->udiff_dbsp_delta * kScale));
    const int64_t zx_prob_i = static_cast<int64_t>(std::llround(summary->zx_prob * kScale));
    const int64_t zx_an_i = static_cast<int64_t>(std::llround(summary->zx_d_prob_analytical * kScale));
    const int64_t zx_sh_i = static_cast<int64_t>(std::llround(summary->zx_d_prob_param_shift * kScale));
    const int64_t zx_norm_i = static_cast<int64_t>(std::llround(summary->zx_amplitude_norm * kScale));

    /* Compute a deterministic summary digest embedded in the Lean theorem record. */
    char digest_seed[256];
    std::snprintf(digest_seed, sizeof(digest_seed),
                  "STARLIGHT_CE12|%u|%u|%u|%u|%d|%lld|%lld|%lld|%lld|%lld|%lld|%lld",
                  summary->lob_rank_before, summary->lob_rank_after,
                  summary->linear_resource_uses, summary->erased_runtime_uses,
                  (int)summary->cubical_path_acyclic,
                  (long long)u_num, (long long)u_imp, (long long)u_dbsp,
                  (long long)zx_prob_i, (long long)zx_an_i, (long long)zx_sh_i, (long long)zx_norm_i);
    uint8_t summary_digest[32];
    eshkol_cert_sha256(reinterpret_cast<const uint8_t*>(digest_seed),
                       std::strlen(digest_seed), summary_digest);

    std::string lean;
    lean.reserve(4096);
    lean += "/-\n";
    lean += "  Eshkol v2.0-starlight (CE-11 + CE-12) Kernel Verification Proof\n";
    lean += "  Self-contained Lean 4.12.0 decidable reflection certificate.\n";
    lean += "-/\n\n";
    lean += "namespace ";
    lean += module_namespace;
    lean += "\n\n";

    lean += "structure StarlightCert where\n";
    lean += "  lobRankBefore      : Nat\n";
    lean += "  lobRankAfter       : Nat\n";
    lean += "  linearResourceUses : Nat\n";
    lean += "  erasedRuntimeUses  : Nat\n";
    lean += "  cubicalPathAcyclic : Bool\n";
    lean += "  udiffNumeric       : Int\n";
    lean += "  udiffImplicitFp    : Int\n";
    lean += "  udiffDbspDelta     : Int\n";
    lean += "  zxProbScaled       : Int\n";
    lean += "  zxDerivAnalytical  : Int\n";
    lean += "  zxDerivParamShift  : Int\n";
    lean += "  zxAmpNormScaled    : Int\n";
    lean += "  summaryDigestHex   : String\n";
    lean += "  deriving DecidableEq, Repr\n\n";

    lean += "def absDiff (a b : Int) : Int :=\n";
    lean += "  if a ≥ b then a - b else b - a\n\n";

    lean += "def verifyStarlight (c : StarlightCert) : Bool :=\n";
    lean += "  decide (c.lobRankAfter < c.lobRankBefore) &&\n";
    lean += "  decide (c.linearResourceUses ≤ 1) &&\n";
    lean += "  decide (c.erasedRuntimeUses = 0) &&\n";
    lean += "  c.cubicalPathAcyclic &&\n";
    lean += "  decide (absDiff c.udiffNumeric c.udiffImplicitFp ≤ 10) &&\n";
    lean += "  decide (absDiff c.udiffNumeric c.udiffDbspDelta ≤ 10) &&\n";
    lean += "  decide (absDiff c.udiffImplicitFp c.udiffDbspDelta ≤ 10) &&\n";
    lean += "  decide (c.zxAmpNormScaled > 0) &&\n";
    lean += "  decide (c.zxProbScaled > 0) &&\n";
    lean += "  decide (absDiff c.zxDerivAnalytical c.zxDerivParamShift ≤ 10)\n\n";

    char num_buf[64];
    lean += "def starlightCert : StarlightCert := {\n";
    std::snprintf(num_buf, sizeof(num_buf), "%u", summary->lob_rank_before);
    lean += "  lobRankBefore := " + std::string(num_buf) + ",\n";
    std::snprintf(num_buf, sizeof(num_buf), "%u", summary->lob_rank_after);
    lean += "  lobRankAfter := " + std::string(num_buf) + ",\n";
    std::snprintf(num_buf, sizeof(num_buf), "%u", summary->linear_resource_uses);
    lean += "  linearResourceUses := " + std::string(num_buf) + ",\n";
    std::snprintf(num_buf, sizeof(num_buf), "%u", summary->erased_runtime_uses);
    lean += "  erasedRuntimeUses := " + std::string(num_buf) + ",\n";
    lean += std::string("  cubicalPathAcyclic := ") + (summary->cubical_path_acyclic ? "true" : "false") + ",\n";
    lean += "  udiffNumeric := " + format_lean_int(u_num) + ",\n";
    lean += "  udiffImplicitFp := " + format_lean_int(u_imp) + ",\n";
    lean += "  udiffDbspDelta := " + format_lean_int(u_dbsp) + ",\n";
    lean += "  zxProbScaled := " + format_lean_int(zx_prob_i) + ",\n";
    lean += "  zxDerivAnalytical := " + format_lean_int(zx_an_i) + ",\n";
    lean += "  zxDerivParamShift := " + format_lean_int(zx_sh_i) + ",\n";
    lean += "  zxAmpNormScaled := " + format_lean_int(zx_norm_i) + ",\n";
    lean += "  summaryDigestHex := \"" + hex_encode(summary_digest, 32) + "\"\n";
    lean += "}\n\n";

    lean += "theorem starlight_ce12_verified :\n";
    lean += "    verifyStarlight starlightCert = true := by\n";
    lean += "  decide\n\n";

    lean += "theorem starlight_lob_and_qtt_sound :\n";
    lean += "    starlightCert.lobRankAfter < starlightCert.lobRankBefore ∧\n";
    lean += "    starlightCert.linearResourceUses ≤ 1 ∧\n";
    lean += "    starlightCert.cubicalPathAcyclic = true := by\n";
    lean += "  decide\n\n";

    lean += "theorem starlight_udiff_and_zx_consistent :\n";
    lean += "    absDiff starlightCert.udiffNumeric starlightCert.udiffImplicitFp ≤ 10 ∧\n";
    lean += "    absDiff starlightCert.udiffNumeric starlightCert.udiffDbspDelta ≤ 10 ∧\n";
    lean += "    absDiff starlightCert.zxDerivAnalytical starlightCert.zxDerivParamShift ≤ 10 := by\n";
    lean += "  decide\n\n";

    lean += "end ";
    lean += module_namespace;
    lean += "\n";

    if (out_len) *out_len = lean.size();
    if (lean.size() + 1 > out_cap) return false;
    std::memcpy(out_buf, lean.data(), lean.size() + 1);
    eshkol_cert_sha256(reinterpret_cast<const uint8_t*>(lean.data()),
                       lean.size(), out_artifact_sha256);
    return true;
}

eshkol_ce12_status_t eshkol_starlight_verify_lean4_artifact(
    const char* lean_file_path,
    const uint8_t expected_artifact_sha256[32],
    const char* lean4_bin_path,
    int* out_lean_exit_code) {
    if (!lean_file_path || !expected_artifact_sha256 || !lean4_bin_path) {
        return ESHKOL_CE12_ERR_INVALID_ARG;
    }

    std::ifstream in(lean_file_path, std::ios::binary);
    if (!in.good()) {
        return ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT;
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    in.close();

    /* 1. Bit-exact SHA-256 check catches ANY 1-bit flip immediately. */
    uint8_t actual_sha[32];
    eshkol_cert_sha256(bytes.data(), bytes.size(), actual_sha);
    uint8_t diff = 0;
    for (int i = 0; i < 32; ++i) diff |= static_cast<uint8_t>(actual_sha[i] ^ expected_artifact_sha256[i]);

    /* 2. Run the real Lean 4.12.0 kernel binary on the file as well. */
    std::string cmd = "\"" + std::string(lean4_bin_path) + "\" \"" +
                      std::string(lean_file_path) + "\" > /dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    if (out_lean_exit_code) *out_lean_exit_code = rc;

    if (diff != 0 || rc != 0) {
        return ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT;
    }
    return ESHKOL_CE12_OK;
}

} /* extern "C" */
