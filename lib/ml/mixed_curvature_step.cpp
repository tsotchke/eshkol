/**
 * @file mixed_curvature_step.cpp
 * @brief The host's mixed-curvature training step: forward, hand-derived
 *        backward, Riemannian Adam update.
 *
 * The written definition is docs/design/XLA_TRAINING_STEP.md, and section 1 of
 * it is the table saying which S4 device primitive each stage below mirrors.
 * Read that first; this file is the transcription, not the design.
 *
 * WHY THE BACKWARD IS WRITTEN OUT BY HAND.
 *
 * The device program's backward comes from StableHLOEmitter::emitVJP walking
 * the forward graph. If this file also obtained its gradients from an
 * automatic reverse mode, the two would share a derivation and the parity test
 * would only be checking that two evaluators of one formula agree. Each stage
 * here therefore carries its VJP derived from the chain rule, written above
 * the function that implements it, and the harness compares the two.
 *
 * WHY EVERY GUARD IS A COMPARISON AND NOT A max/min.
 *
 * Same reason geometric_lowering.cpp gives: `clamp_min(a, eps)` is
 * `a < eps ? eps : a`, whose gradient at the tie keeps `a`, while
 * `maximum(a, eps)` gives the tie to `eps` under Eshkol's convention and would
 * route a zero to the input. The device module emits selects on these same
 * comparisons, so the two sides tie the same way by construction.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/ml/mixed_curvature_step.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

/** @brief The reference's "the vector is zero" threshold (TINY in tests/qllm_oracle). */
constexpr double kTiny = 1e-10;

/**
 * @brief The acosh floor, and the one place this step departs from
 *        GeometricPrimitive::HyperbolicDistance's floor of exactly 1.
 *
 * d^2's derivative carries acosh(A)/sqrt(A^2-1). Its limit at A = 1 is 1, but
 * numerator and denominator both vanish there, so evaluating it at A = 1 is
 * 0/0 and gives NaN. Flooring A at 1 + 1e-12 bounds sqrt(A^2-1) below by about
 * 1.4e-6. The device module floors at the identical constant, so this is part
 * of the operator on both sides rather than a host-only repair.
 */
constexpr double kAcoshFloor = 1.0 + 1e-12;

/** @brief The oracle's cosine clamp (clampUnit in geometric_lowering.cpp). */
constexpr double kCosClamp = 1.0 - 1e-7;

inline double dotN(const double* a, const double* b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline double clampMin(double a, double lo) { return a < lo ? lo : a; }

/** @brief splitmix64, so "deterministic" does not depend on any libc. */
inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/** @brief Uniform in (-1, 1). */
inline double uniform(uint64_t* state) {
    const uint64_t r = splitmix64(state) >> 11;             // 53 bits
    const double u = static_cast<double>(r) * (1.0 / 9007199254740992.0);
    return 2.0 * u - 1.0;
}

// ---------------------------------------------------------------------------
// The manifold operations the OPTIMIZER uses, transcribed from the S4
// compositions in lib/backend/xla/geometric_lowering.cpp. They are not
// differentiated (the optimizer update is not on the backward path), so they
// are plain forward code.
// ---------------------------------------------------------------------------

/** @brief x (+)_c y, GeometricPrimitive::MobiusAdd. */
void mobiusAdd(const double* x, const double* y, double c, int64_t d, double* out) {
    const double xy = dotN(x, y, d);
    const double x2 = dotN(x, x, d);
    const double y2 = dotN(y, y, d);
    const double two_c_xy = 2.0 * c * xy;
    const double nx = 1.0 + two_c_xy + c * y2;
    const double ny = 1.0 - c * x2;
    const double den = 1.0 + two_c_xy + c * c * x2 * y2;
    for (int64_t i = 0; i < d; ++i) out[i] = (nx * x[i] + ny * y[i]) / den;
}

/** @brief exp^c_x(v), the Ganea formula. GeometricPrimitive::PoincareExpMap. */
void poincareExpMap(const double* x, const double* v, double c, int64_t d, double* out) {
    const double nv = std::sqrt(dotN(v, v, d));
    if (nv < kTiny) {
        for (int64_t i = 0; i < d; ++i) out[i] = x[i];
        return;
    }
    const double sc = std::sqrt(c);
    const double lam = 2.0 / (1.0 - c * dotN(x, x, d));
    const double t = sc * lam * nv / 2.0;
    const double scale = std::tanh(t) / (sc * nv);
    double second[64];
    double* sec = (d <= 64) ? second : nullptr;
    if (sec) {
        for (int64_t i = 0; i < d; ++i) sec[i] = v[i] * scale;
        mobiusAdd(x, sec, c, d, out);
        return;
    }
    // d > 64: reuse `out` as the scratch for the second operand. mobiusAdd
    // reads y only through reductions computed before it writes, so aliasing
    // out and y is safe — but it is spelled out here rather than assumed,
    // because a future rearrangement of mobiusAdd would break it silently.
    for (int64_t i = 0; i < d; ++i) out[i] = v[i] * scale;
    double xy = dotN(x, out, d), x2 = dotN(x, x, d), y2 = dotN(out, out, d);
    const double two_c_xy = 2.0 * c * xy;
    const double nx = 1.0 + two_c_xy + c * y2;
    const double ny = 1.0 - c * x2;
    const double den = 1.0 + two_c_xy + c * c * x2 * y2;
    for (int64_t i = 0; i < d; ++i) out[i] = (nx * x[i] + ny * out[i]) / den;
}

/** @brief Radial clip back inside the ball. GeometricPrimitive::PoincareRetract, zero step. */
void poincareClip(double* z, double c, double eps, int64_t d) {
    const double n2 = dotN(z, z, d);
    const double maxn2 = (1.0 - eps) / c;
    if (n2 > maxn2) {
        const double scale = std::sqrt(maxn2 / clampMin(n2, eps));
        for (int64_t i = 0; i < d; ++i) z[i] *= scale;
    }
}

/** @brief exp_x(v) on the unit sphere. GeometricPrimitive::SphereExpMap. */
void sphereExpMap(const double* x, const double* v, int64_t d, double* out) {
    const double n = std::sqrt(dotN(v, v, d));
    if (n < kTiny) {
        for (int64_t i = 0; i < d; ++i) out[i] = x[i];
        return;
    }
    const double cn = std::cos(n);
    const double sn = std::sin(n) / n;
    for (int64_t i = 0; i < d; ++i) out[i] = cn * x[i] + sn * v[i];
}

/** @brief Renormalise. GeometricPrimitive::SphereRetract, zero step. */
void sphereNormalise(double* z, double eps, int64_t d) {
    const double n = std::sqrt(dotN(z, z, d));
    if (n > eps) {
        const double inv = 1.0 / clampMin(n, eps);
        for (int64_t i = 0; i < d; ++i) z[i] *= inv;
    }
}

/**
 * @brief Adam's moments and delta, elementwise.
 *
 * Transcribed from vm_riemannian_adam_delta (lib/backend/vm_geometric.c:267):
 * epsilon OUTSIDE the square root, bias corrections 1 - beta^step, and the
 * minus sign carried in the delta rather than applied by the retraction.
 *
 * @param step The count AFTER incrementing, which is what that function uses.
 */
void adamDelta(const double* rg, double* m, double* v, int64_t nelem,
               const EshkolMixedCurvatureHyper& h, int64_t step, double* delta) {
    const double b1c = 1.0 - std::pow(h.beta1, static_cast<double>(step));
    const double b2c = 1.0 - std::pow(h.beta2, static_cast<double>(step));
    for (int64_t e = 0; e < nelem; ++e) {
        const double g = rg[e];
        m[e] = h.beta1 * m[e] + (1.0 - h.beta1) * g;
        v[e] = h.beta2 * v[e] + (1.0 - h.beta2) * g * g;
        const double mh = m[e] / b1c;
        const double vh = v[e] / b2c;
        delta[e] = -h.lr * mh / (std::sqrt(vh) + h.adam_eps);
    }
}

}  // namespace

extern "C" {

void eshkol_mixed_curvature_default_hyper(EshkolMixedCurvatureHyper* out) {
    if (!out) return;
    out->curvature = 1.0;
    out->alpha = 0.5;
    out->lr = 0.05;
    out->beta1 = 0.9;
    out->beta2 = 0.999;
    out->adam_eps = 1e-8;
    out->guard_eps = 1e-5;
    out->w_hyp = 1.0;
    out->w_sph = 1.0;
    out->w_euc = 1.0;
}

int64_t eshkol_mixed_curvature_w_elements(EshkolMixedCurvatureShape s) { return s.d * s.d; }
int64_t eshkol_mixed_curvature_p_elements(EshkolMixedCurvatureShape s) { return s.c * s.d; }
int64_t eshkol_mixed_curvature_x_elements(EshkolMixedCurvatureShape s) { return s.n * s.d; }
int64_t eshkol_mixed_curvature_t_elements(EshkolMixedCurvatureShape s) { return s.n * s.c; }

int64_t eshkol_mixed_curvature_scratch_elements(EshkolMixedCurvatureShape s) {
    // U, Z, Y, gU, gZ, gY : 6 [n,d]
    // Dh, Ds, De, L, gL   : 5 [n,c]
    // rg, delta           : 2 max(d*d, c*d)
    const int64_t nd = s.n * s.d;
    const int64_t nc = s.n * s.c;
    const int64_t big = (s.d * s.d > s.c * s.d) ? s.d * s.d : s.c * s.d;
    return 6 * nd + 5 * nc + 2 * big + 16;
}

void eshkol_mixed_curvature_init(EshkolMixedCurvatureShape s,
                                 const EshkolMixedCurvatureHyper* hyper,
                                 uint64_t seed,
                                 EshkolMixedCurvatureParams* params,
                                 EshkolMixedCurvatureMoments* moments,
                                 double* batch,
                                 double* targets) {
    if (s.n <= 0 || s.d <= 0 || s.c <= 0) return;
    const double c = (hyper && hyper->curvature > 0.0) ? hyper->curvature : 1.0;
    uint64_t st = seed * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL;

    if (params) {
        if (params->w) {
            // Small enough that alpha*U stays well inside the ball's tangent
            // range on the first step, so the exp map is not asked for a point
            // it would have to clip on iteration one.
            for (int64_t e = 0; e < s.d * s.d; ++e) params->w[e] = 0.15 * uniform(&st);
        }
        if (params->p_hyp) {
            // Radius 0.3/sqrt(c): strictly inside the ball, and far enough from
            // the boundary that the first steps are not measuring the clip.
            const double radius = 0.3 / std::sqrt(c);
            for (int64_t k = 0; k < s.c; ++k) {
                double* row = params->p_hyp + k * s.d;
                for (int64_t i = 0; i < s.d; ++i) row[i] = uniform(&st);
                const double n = std::sqrt(dotN(row, row, s.d));
                const double scale = (n > kTiny) ? radius / n : 0.0;
                for (int64_t i = 0; i < s.d; ++i) row[i] *= scale;
            }
        }
        if (params->p_sph) {
            for (int64_t k = 0; k < s.c; ++k) {
                double* row = params->p_sph + k * s.d;
                for (int64_t i = 0; i < s.d; ++i) row[i] = uniform(&st);
                const double n = std::sqrt(dotN(row, row, s.d));
                const double scale = (n > kTiny) ? 1.0 / n : 0.0;
                for (int64_t i = 0; i < s.d; ++i) row[i] *= scale;
                if (scale == 0.0) row[0] = 1.0;
            }
        }
        if (params->p_euc) {
            for (int64_t e = 0; e < s.c * s.d; ++e) params->p_euc[e] = 0.5 * uniform(&st);
        }
    }

    if (moments) {
        const int64_t we = s.d * s.d, pe = s.c * s.d;
        if (moments->m_w) std::memset(moments->m_w, 0, sizeof(double) * static_cast<size_t>(we));
        if (moments->v_w) std::memset(moments->v_w, 0, sizeof(double) * static_cast<size_t>(we));
        if (moments->m_hyp) std::memset(moments->m_hyp, 0, sizeof(double) * static_cast<size_t>(pe));
        if (moments->v_hyp) std::memset(moments->v_hyp, 0, sizeof(double) * static_cast<size_t>(pe));
        if (moments->m_sph) std::memset(moments->m_sph, 0, sizeof(double) * static_cast<size_t>(pe));
        if (moments->v_sph) std::memset(moments->v_sph, 0, sizeof(double) * static_cast<size_t>(pe));
        if (moments->m_euc) std::memset(moments->m_euc, 0, sizeof(double) * static_cast<size_t>(pe));
        if (moments->v_euc) std::memset(moments->v_euc, 0, sizeof(double) * static_cast<size_t>(pe));
        moments->step = 0;
    }

    if (batch) {
        for (int64_t e = 0; e < s.n * s.d; ++e) batch[e] = uniform(&st);
    }
    if (targets) {
        // Softmax of uniform draws: strictly positive, rows sum to one, and
        // never one-hot — a one-hot target makes softmax(L) - T vanish in all
        // but one column and hides a wrong gradient in the other C-1.
        for (int64_t n = 0; n < s.n; ++n) {
            double* row = targets + n * s.c;
            double mx = -1e300;
            for (int64_t k = 0; k < s.c; ++k) { row[k] = uniform(&st); if (row[k] > mx) mx = row[k]; }
            double sum = 0.0;
            for (int64_t k = 0; k < s.c; ++k) { row[k] = std::exp(row[k] - mx); sum += row[k]; }
            for (int64_t k = 0; k < s.c; ++k) row[k] /= sum;
        }
    }
}

}  // extern "C"

namespace {

/**
 * @brief Everything the forward computes, laid over the caller's scratch.
 *
 * A struct rather than a dozen pointer arguments because the backward needs
 * exactly the same buffers and an argument list that long is where the wrong
 * one gets passed.
 */
struct Work {
    double* u;    // [n,d]
    double* z;    // [n,d]
    double* y;    // [n,d]
    double* gu;   // [n,d]
    double* gz;   // [n,d]
    double* gy;   // [n,d]
    double* dh;   // [n,c]
    double* ds;   // [n,c]
    double* de;   // [n,c]
    double* logit;// [n,c]
    double* gl;   // [n,c]
    double* rg;   // max(d*d, c*d)
    double* delta;// max(d*d, c*d)
};

Work layout(double* scratch, EshkolMixedCurvatureShape s) {
    const int64_t nd = s.n * s.d, nc = s.n * s.c;
    const int64_t big = (s.d * s.d > s.c * s.d) ? s.d * s.d : s.c * s.d;
    Work w{};
    double* p = scratch;
    w.u = p; p += nd;
    w.z = p; p += nd;
    w.y = p; p += nd;
    w.gu = p; p += nd;
    w.gz = p; p += nd;
    w.gy = p; p += nd;
    w.dh = p; p += nc;
    w.ds = p; p += nc;
    w.de = p; p += nc;
    w.logit = p; p += nc;
    w.gl = p; p += nc;
    w.rg = p; p += big;
    w.delta = p; p += big;
    return w;
}

/** @brief U = X W. Mirrors stablehlo.dot_general at HIGHEST precision. */
void forwardProjection(const double* x, const double* wmat, double* u,
                       int64_t n, int64_t d) {
    for (int64_t r = 0; r < n; ++r) {
        double* out = u + r * d;
        for (int64_t i = 0; i < d; ++i) out[i] = 0.0;
        const double* xr = x + r * d;
        for (int64_t j = 0; j < d; ++j) {
            const double xv = xr[j];
            const double* wr = wmat + j * d;
            for (int64_t i = 0; i < d; ++i) out[i] += xv * wr[i];
        }
    }
}

/** @brief Z = exp_0^c(alpha U), GeometricPrimitive::PoincareExpMapOrigin per row. */
void forwardExpMapOrigin(const double* u, double* z, int64_t n, int64_t d,
                         double c, double alpha) {
    const double sc = std::sqrt(c);
    for (int64_t r = 0; r < n; ++r) {
        const double* ur = u + r * d;
        double* zr = z + r * d;
        double nv2 = 0.0;
        for (int64_t i = 0; i < d; ++i) { const double v = alpha * ur[i]; nv2 += v * v; }
        const double nv = std::sqrt(nv2);
        if (nv < kTiny) {
            for (int64_t i = 0; i < d; ++i) zr[i] = alpha * ur[i];
            continue;
        }
        const double t = sc * nv;
        const double k = std::tanh(t) / t;
        for (int64_t i = 0; i < d; ++i) zr[i] = alpha * ur[i] * k;
    }
}

/** @brief Y = U / clamp_min(|U|, eps). */
void forwardSphereNormalise(const double* u, double* y, int64_t n, int64_t d, double eps) {
    for (int64_t r = 0; r < n; ++r) {
        const double* ur = u + r * d;
        double* yr = y + r * d;
        const double rn = std::sqrt(dotN(ur, ur, d));
        const double inv = 1.0 / clampMin(rn, eps);
        for (int64_t i = 0; i < d; ++i) yr[i] = ur[i] * inv;
    }
}

/** @brief Dh[n,k] = d_c(Z[n], Ph[k])^2, GeometricPrimitive::HyperbolicDistance squared. */
void forwardHyperbolicSq(const double* z, const double* ph, double* dh,
                         int64_t n, int64_t d, int64_t nc, double c) {
    const double sc = std::sqrt(c);
    for (int64_t r = 0; r < n; ++r) {
        const double* zr = z + r * d;
        const double ax = 1.0 - c * dotN(zr, zr, d);
        for (int64_t k = 0; k < nc; ++k) {
            const double* pk = ph + k * d;
            const double bx = 1.0 - c * dotN(pk, pk, d);
            double q = 0.0;
            for (int64_t i = 0; i < d; ++i) { const double e = zr[i] - pk[i]; q += e * e; }
            const double raw = 1.0 + 2.0 * c * q / (ax * bx);
            const double a = (raw < kAcoshFloor) ? kAcoshFloor : raw;
            const double dist = std::log(a + std::sqrt(a * a - 1.0)) / sc;
            dh[r * nc + k] = dist * dist;
        }
    }
}

/** @brief Ds[n,k] = theta(Y[n], Ps[k])^2, GeometricPrimitive::SphericalDistance squared. */
void forwardSphericalSq(const double* y, const double* ps, double* ds,
                        int64_t n, int64_t d, int64_t nc) {
    for (int64_t r = 0; r < n; ++r) {
        const double* yr = y + r * d;
        for (int64_t k = 0; k < nc; ++k) {
            const double s = dotN(yr, ps + k * d, d);
            const double hi = (s > kCosClamp) ? kCosClamp : s;
            const double sc = (hi < -kCosClamp) ? -kCosClamp : hi;
            const double th = std::atan2(std::sqrt(1.0 - sc * sc), sc);
            ds[r * nc + k] = th * th;
        }
    }
}

/** @brief De[n,k] = |U[n] - Pe[k]|^2. */
void forwardEuclideanSq(const double* u, const double* pe, double* de,
                        int64_t n, int64_t d, int64_t nc) {
    for (int64_t r = 0; r < n; ++r) {
        const double* ur = u + r * d;
        for (int64_t k = 0; k < nc; ++k) {
            const double* pk = pe + k * d;
            double q = 0.0;
            for (int64_t i = 0; i < d; ++i) { const double e = ur[i] - pk[i]; q += e * e; }
            de[r * nc + k] = q;
        }
    }
}

/**
 * @brief Logits, mean soft-target cross entropy, and dL/dlogits.
 *
 * loss  = (1/N) sum_n ( logsumexp_k L - sum_k T L ),  rows of T summing to 1.
 * dloss/dL[n,k] = (softmax(L)[n,k] - T[n,k]) / N, which is
 * tensor_cross_entropy_backward's rule with the 1/N of section 3.4.
 */
double forwardLogitsAndLoss(const Work& w, const double* targets,
                            int64_t n, int64_t nc,
                            const EshkolMixedCurvatureHyper& h, bool want_grad) {
    double acc = 0.0;
    const double inv_n = 1.0 / static_cast<double>(n);
    for (int64_t r = 0; r < n; ++r) {
        double* lr = w.logit + r * nc;
        double mx = -1e300;
        for (int64_t k = 0; k < nc; ++k) {
            lr[k] = -(h.w_hyp * w.dh[r * nc + k] + h.w_sph * w.ds[r * nc + k] +
                      h.w_euc * w.de[r * nc + k]);
            if (lr[k] > mx) mx = lr[k];
        }
        double sum = 0.0;
        for (int64_t k = 0; k < nc; ++k) sum += std::exp(lr[k] - mx);
        const double lse = mx + std::log(sum);
        double dotT = 0.0;
        for (int64_t k = 0; k < nc; ++k) dotT += targets[r * nc + k] * lr[k];
        acc += lse - dotT;
        if (want_grad) {
            for (int64_t k = 0; k < nc; ++k) {
                const double sm = std::exp(lr[k] - mx) / sum;
                w.gl[r * nc + k] = (sm - targets[r * nc + k]) * inv_n;
            }
        }
    }
    return acc * inv_n;
}

// -------------------------------------------------------------------------
// Backward. Each function states its derivation.
// -------------------------------------------------------------------------

/**
 * De = sum_i (U_i - Pe_i)^2, so dDe/dU_i = 2(U_i - Pe_i) and
 * dDe/dPe_i = -2(U_i - Pe_i). Accumulates into gU and gPe.
 */
void backwardEuclideanSq(const Work& w, const double* pe, double* gpe,
                         int64_t n, int64_t d, int64_t nc, double weight) {
    for (int64_t r = 0; r < n; ++r) {
        const double* ur = w.u + r * d;
        double* gur = w.gu + r * d;
        for (int64_t k = 0; k < nc; ++k) {
            const double gd = -weight * w.gl[r * nc + k];
            if (gd == 0.0) continue;
            const double* pk = pe + k * d;
            double* gpk = gpe + k * d;
            for (int64_t i = 0; i < d; ++i) {
                const double t = 2.0 * gd * (ur[i] - pk[i]);
                gur[i] += t;
                gpk[i] -= t;
            }
        }
    }
}

/**
 * Ds = theta^2 with theta = acos(sc), sc the clamped <Y, Ps>. So
 *   dDs/dsc = 2 theta * (-1 / sqrt(1 - sc^2)),
 * zero where the clamp bound: above 1 - 1e-7 the operator is constant in sc,
 * exactly as the oracle's clampUnit makes it. sqrt(1 - sc^2) is bounded below
 * by 4.5e-4 by that same clamp, so the 0/0 at coincident points cannot occur.
 * Then dsc/dY = Ps and dsc/dPs = Y.
 */
void backwardSphericalSq(const Work& w, const double* ps, double* gps,
                         int64_t n, int64_t d, int64_t nc, double weight) {
    for (int64_t r = 0; r < n; ++r) {
        const double* yr = w.y + r * d;
        double* gyr = w.gy + r * d;
        for (int64_t k = 0; k < nc; ++k) {
            const double gd = -weight * w.gl[r * nc + k];
            if (gd == 0.0) continue;
            const double* pk = ps + k * d;
            const double s = dotN(yr, pk, d);
            if (s > kCosClamp || s < -kCosClamp) continue;  // clamped: exact zero
            const double th = std::atan2(std::sqrt(1.0 - s * s), s);
            const double gt = gd * 2.0 * th * (-1.0 / std::sqrt(1.0 - s * s));
            double* gpk = gps + k * d;
            for (int64_t i = 0; i < d; ++i) {
                gyr[i] += gt * pk[i];
                gpk[i] += gt * yr[i];
            }
        }
    }
}

/**
 * Y = U / r with r = |U| where r > eps, so
 *   dY_i/dU_j = delta_ij/r - U_i U_j / r^3,
 * giving gU = (gY - <gY, Y> Y) / r. Where r <= eps the divisor is the constant
 * eps and gU = gY / eps.
 */
void backwardSphereNormalise(const Work& w, int64_t n, int64_t d, double eps) {
    for (int64_t r = 0; r < n; ++r) {
        const double* ur = w.u + r * d;
        const double* yr = w.y + r * d;
        const double* gyr = w.gy + r * d;
        double* gur = w.gu + r * d;
        const double rn = std::sqrt(dotN(ur, ur, d));
        if (rn > eps) {
            const double dot = dotN(gyr, yr, d);
            const double inv = 1.0 / rn;
            for (int64_t i = 0; i < d; ++i) gur[i] += (gyr[i] - dot * yr[i]) * inv;
        } else {
            const double inv = 1.0 / eps;
            for (int64_t i = 0; i < d; ++i) gur[i] += gyr[i] * inv;
        }
    }
}

/**
 * Dh = (acosh(A)/sqrt(c))^2 with A = max_floor(1 + 2 c q / (ax bx)).
 *   dDh/dA = 2 (acosh(A)/sqrt(c)) * (1/sqrt(c)) / sqrt(A^2 - 1)
 * and, below the floor, zero.
 *   dA/dq = 2c/(ax bx),  dA/dax = -2 c q/(ax^2 bx),  dA/dbx = -2 c q/(ax bx^2)
 *   dq/dZ = 2(Z - Ph),   dq/dPh = -2(Z - Ph)
 *   dax/dZ = -2 c Z,     dbx/dPh = -2 c Ph
 * ax and bx are NOT floored, matching GeometricPrimitive::HyperbolicDistance,
 * which divides by them unguarded; both stay positive while the points are
 * interior, which the retraction guarantees.
 */
void backwardHyperbolicSq(const Work& w, const double* ph, double* gph,
                          int64_t n, int64_t d, int64_t nc, double c, double weight) {
    const double sc = std::sqrt(c);
    for (int64_t r = 0; r < n; ++r) {
        const double* zr = w.z + r * d;
        double* gzr = w.gz + r * d;
        const double ax = 1.0 - c * dotN(zr, zr, d);
        for (int64_t k = 0; k < nc; ++k) {
            const double gd = -weight * w.gl[r * nc + k];
            if (gd == 0.0) continue;
            const double* pk = ph + k * d;
            const double bx = 1.0 - c * dotN(pk, pk, d);
            double q = 0.0;
            for (int64_t i = 0; i < d; ++i) { const double e = zr[i] - pk[i]; q += e * e; }
            const double raw = 1.0 + 2.0 * c * q / (ax * bx);
            if (raw < kAcoshFloor) continue;  // the floor is a select: exact zero below it
            const double a = raw;
            const double dist = std::log(a + std::sqrt(a * a - 1.0)) / sc;
            const double ga = gd * 2.0 * dist / (sc * std::sqrt(a * a - 1.0));
            const double gq = ga * 2.0 * c / (ax * bx);
            const double gax = -ga * 2.0 * c * q / (ax * ax * bx);
            const double gbx = -ga * 2.0 * c * q / (ax * bx * bx);
            double* gpk = gph + k * d;
            for (int64_t i = 0; i < d; ++i) {
                const double e = zr[i] - pk[i];
                gzr[i] += 2.0 * gq * e - 2.0 * c * gax * zr[i];
                gpk[i] += -2.0 * gq * e - 2.0 * c * gbx * pk[i];
            }
        }
    }
}

/**
 * Z = v k(t), v = alpha U, t = sqrt(c)|v|, k(t) = tanh(t)/t. Then
 *   dZ_i/dv_j = k delta_ij + v_i k'(t) sqrt(c) v_j/|v|,
 *   k'(t) = (1 - tanh^2 t)/t - tanh(t)/t^2,
 * so gv = k gZ + <gZ, v> k'(t) sqrt(c) v/|v| and gU = alpha gv. Below the TINY
 * short circuit Z = v exactly and gv = gZ.
 */
void backwardExpMapOrigin(const Work& w, int64_t n, int64_t d, double c, double alpha) {
    const double sc = std::sqrt(c);
    for (int64_t r = 0; r < n; ++r) {
        const double* ur = w.u + r * d;
        const double* gzr = w.gz + r * d;
        double* gur = w.gu + r * d;
        double nv2 = 0.0;
        for (int64_t i = 0; i < d; ++i) { const double v = alpha * ur[i]; nv2 += v * v; }
        const double nv = std::sqrt(nv2);
        if (nv < kTiny) {
            for (int64_t i = 0; i < d; ++i) gur[i] += alpha * gzr[i];
            continue;
        }
        const double t = sc * nv;
        const double th = std::tanh(t);
        const double k = th / t;
        const double kp = (1.0 - th * th) / t - th / (t * t);
        double dot = 0.0;
        for (int64_t i = 0; i < d; ++i) dot += gzr[i] * (alpha * ur[i]);
        const double radial = dot * kp * sc / nv;
        for (int64_t i = 0; i < d; ++i) {
            const double v = alpha * ur[i];
            gur[i] += alpha * (k * gzr[i] + radial * v);
        }
    }
}

/** U = X W, so gW = X^T gU. X is data and receives nothing. */
void backwardProjection(const double* x, const Work& w, double* gw,
                        int64_t n, int64_t d) {
    for (int64_t e = 0; e < d * d; ++e) gw[e] = 0.0;
    for (int64_t r = 0; r < n; ++r) {
        const double* xr = x + r * d;
        const double* gur = w.gu + r * d;
        for (int64_t j = 0; j < d; ++j) {
            const double xv = xr[j];
            if (xv == 0.0) continue;
            double* gwr = gw + j * d;
            for (int64_t i = 0; i < d; ++i) gwr[i] += xv * gur[i];
        }
    }
}

}  // namespace

extern "C" {

bool eshkol_mixed_curvature_loss(EshkolMixedCurvatureShape s,
                                 const EshkolMixedCurvatureHyper* hyper,
                                 const EshkolMixedCurvatureParams* params,
                                 const double* batch,
                                 const double* targets,
                                 double* scratch,
                                 double* out_loss) {
    if (!hyper || !params || !batch || !targets || !scratch || !out_loss) return false;
    if (s.n <= 0 || s.d <= 0 || s.c <= 0 || hyper->curvature <= 0.0) return false;
    if (!params->w || !params->p_hyp || !params->p_sph || !params->p_euc) return false;

    Work w = layout(scratch, s);
    forwardProjection(batch, params->w, w.u, s.n, s.d);
    forwardExpMapOrigin(w.u, w.z, s.n, s.d, hyper->curvature, hyper->alpha);
    forwardSphereNormalise(w.u, w.y, s.n, s.d, hyper->guard_eps);
    forwardHyperbolicSq(w.z, params->p_hyp, w.dh, s.n, s.d, s.c, hyper->curvature);
    forwardSphericalSq(w.y, params->p_sph, w.ds, s.n, s.d, s.c);
    forwardEuclideanSq(w.u, params->p_euc, w.de, s.n, s.d, s.c);
    *out_loss = forwardLogitsAndLoss(w, targets, s.n, s.c, *hyper, false);
    return true;
}

bool eshkol_mixed_curvature_train_step(EshkolMixedCurvatureShape s,
                                       const EshkolMixedCurvatureHyper* hyper,
                                       EshkolMixedCurvatureParams* params,
                                       EshkolMixedCurvatureMoments* moments,
                                       const double* batch,
                                       const double* targets,
                                       double* scratch,
                                       double* out_loss,
                                       EshkolMixedCurvatureParams* out_grads) {
    if (!hyper || !params || !moments || !batch || !targets || !scratch) return false;
    if (s.n <= 0 || s.d <= 0 || s.c <= 0 || hyper->curvature <= 0.0) return false;
    if (!params->w || !params->p_hyp || !params->p_sph || !params->p_euc) return false;
    if (!moments->m_w || !moments->v_w || !moments->m_hyp || !moments->v_hyp ||
        !moments->m_sph || !moments->v_sph || !moments->m_euc || !moments->v_euc) return false;

    const double c = hyper->curvature;
    const int64_t d = s.d, nc = s.c, n = s.n;
    const int64_t we = d * d, pe = nc * d;
    Work w = layout(scratch, s);

    // ---- forward ----
    forwardProjection(batch, params->w, w.u, n, d);
    forwardExpMapOrigin(w.u, w.z, n, d, c, hyper->alpha);
    forwardSphereNormalise(w.u, w.y, n, d, hyper->guard_eps);
    forwardHyperbolicSq(w.z, params->p_hyp, w.dh, n, d, nc, c);
    forwardSphericalSq(w.y, params->p_sph, w.ds, n, d, nc);
    forwardEuclideanSq(w.u, params->p_euc, w.de, n, d, nc);
    const double loss = forwardLogitsAndLoss(w, targets, n, nc, *hyper, true);
    if (out_loss) *out_loss = loss;

    // ---- backward ----
    // The gradient destinations are the caller's when it asked for them, so
    // that reporting them costs no copy; otherwise they borrow the scratch's
    // moment-sized blocks, which the optimizer below has not touched yet.
    double* gw = out_grads && out_grads->w ? out_grads->w : nullptr;
    double* gph = out_grads && out_grads->p_hyp ? out_grads->p_hyp : nullptr;
    double* gps = out_grads && out_grads->p_sph ? out_grads->p_sph : nullptr;
    double* gpe = out_grads && out_grads->p_euc ? out_grads->p_euc : nullptr;

    // Local storage when the caller wants no gradients back. Sized from the
    // shape, never from a fixed constant.
    static thread_local std::vector<double> local;
    if (!gw || !gph || !gps || !gpe) {
        local.assign(static_cast<size_t>(we + 3 * pe), 0.0);
        double* p = local.data();
        if (!gw) { gw = p; } p += we;
        if (!gph) { gph = p; } p += pe;
        if (!gps) { gps = p; } p += pe;
        if (!gpe) { gpe = p; }
    }

    for (int64_t e = 0; e < n * d; ++e) { w.gu[e] = 0.0; w.gz[e] = 0.0; w.gy[e] = 0.0; }
    for (int64_t e = 0; e < pe; ++e) { gph[e] = 0.0; gps[e] = 0.0; gpe[e] = 0.0; }

    backwardEuclideanSq(w, params->p_euc, gpe, n, d, nc, hyper->w_euc);
    backwardSphericalSq(w, params->p_sph, gps, n, d, nc, hyper->w_sph);
    backwardSphereNormalise(w, n, d, hyper->guard_eps);
    backwardHyperbolicSq(w, params->p_hyp, gph, n, d, nc, c, hyper->w_hyp);
    backwardExpMapOrigin(w, n, d, c, hyper->alpha);
    backwardProjection(batch, w, gw, n, d);

    // ---- Riemannian Adam ----
    const int64_t step = moments->step + 1;

    // W: Euclidean, retraction x + delta.
    adamDelta(gw, moments->m_w, moments->v_w, we, *hyper, step, w.delta);
    for (int64_t e = 0; e < we; ++e) params->w[e] += w.delta[e];

    // P_euc: Euclidean.
    adamDelta(gpe, moments->m_euc, moments->v_euc, pe, *hyper, step, w.delta);
    for (int64_t e = 0; e < pe; ++e) params->p_euc[e] += w.delta[e];

    // P_hyp: Poincare gradient rescaling, then exp map at the point, then the
    // radial clip that keeps it inside the ball.
    for (int64_t k = 0; k < nc; ++k) {
        const double* xk = params->p_hyp + k * d;
        double conf = 1.0 - c * dotN(xk, xk, d);
        conf = clampMin(conf, hyper->guard_eps);
        const double scale = 0.25 * conf * conf;
        for (int64_t i = 0; i < d; ++i) w.rg[k * d + i] = scale * gph[k * d + i];
    }
    adamDelta(w.rg, moments->m_hyp, moments->v_hyp, pe, *hyper, step, w.delta);
    for (int64_t k = 0; k < nc; ++k) {
        double* xk = params->p_hyp + k * d;
        double* moved = w.rg + k * d;  // rg is spent; reuse as the destination
        poincareExpMap(xk, w.delta + k * d, c, d, moved);
        poincareClip(moved, c, hyper->guard_eps, d);
        for (int64_t i = 0; i < d; ++i) xk[i] = moved[i];
    }

    // P_sph: tangent projection, then the sphere exp map, then renormalise.
    for (int64_t k = 0; k < nc; ++k) {
        const double* xk = params->p_sph + k * d;
        const double* gk = gps + k * d;
        const double dot = dotN(gk, xk, d);
        for (int64_t i = 0; i < d; ++i) w.rg[k * d + i] = gk[i] - dot * xk[i];
    }
    adamDelta(w.rg, moments->m_sph, moments->v_sph, pe, *hyper, step, w.delta);
    for (int64_t k = 0; k < nc; ++k) {
        double* xk = params->p_sph + k * d;
        double* moved = w.rg + k * d;
        sphereExpMap(xk, w.delta + k * d, d, moved);
        sphereNormalise(moved, hyper->guard_eps, d);
        for (int64_t i = 0; i < d; ++i) xk[i] = moved[i];
    }

    moments->step = step;
    return true;
}

bool eshkol_mixed_curvature_constraints_hold(EshkolMixedCurvatureShape s,
                                             double curvature,
                                             const double* p_hyp,
                                             const double* p_sph,
                                             double margin,
                                             double sph_tol,
                                             double* worst_hyp,
                                             double* worst_sph) {
    if (s.c <= 0 || s.d <= 0 || curvature <= 0.0) return false;
    bool ok = true;
    double wh = 0.0, ws = 0.0;
    if (p_hyp) {
        for (int64_t k = 0; k < s.c; ++k) {
            const double* row = p_hyp + k * s.d;
            const double v = curvature * dotN(row, row, s.d);
            if (v > wh) wh = v;
            if (!(v <= 1.0 - margin)) ok = false;
        }
    }
    if (p_sph) {
        for (int64_t k = 0; k < s.c; ++k) {
            const double* row = p_sph + k * s.d;
            const double dev = std::fabs(std::sqrt(dotN(row, row, s.d)) - 1.0);
            if (dev > ws) ws = dev;
            if (!(dev <= sph_tol)) ok = false;
        }
    }
    if (worst_hyp) *worst_hyp = wh;
    if (worst_sph) *worst_sph = ws;
    return ok;
}

}  // extern "C"
