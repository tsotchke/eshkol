/**
 * @file training_step_lowering.cpp
 * @brief The device composition for one mixed-curvature training step.
 *
 * Every forward formula here is the batched form of the corresponding
 * per-vector composition in geometric_lowering.cpp, and the host's
 * lib/ml/mixed_curvature_step.cpp is the scalar form of the same. The three
 * are transcriptions of one written definition,
 * docs/design/XLA_TRAINING_STEP.md; where this file departs from an S4
 * primitive (the acosh floor at 1 + 1e-12) the host departs identically and
 * the document says why.
 *
 * WHY THE PAIRWISE DISTANCES ARE BUILT AT [n,c,d] RATHER THAN BY THE
 * |a|^2 + |b|^2 - 2<a,b> IDENTITY.
 *
 * That identity is one matmul instead of a broadcast and a reduce, and it is
 * catastrophically cancelling exactly where these distances matter: when a row
 * approaches a prototype, |a|^2 + |b|^2 and 2<a,b> approach each other and the
 * difference loses most of its significant digits. The squared distance then
 * comes back small and wrong, or negative, and the hyperbolic branch's
 * `sqrt(A^2 - 1)` turns that into a NaN. So the difference is formed
 * explicitly and reduced, which is what the host does too and therefore what
 * the two can be compared over.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/training_step_lowering.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "eshkol/backend/xla/stablehlo_emitter.h"

namespace eshkol {
namespace xla {

namespace {

/** @brief The reference's zero-vector threshold, as in geometric_lowering.cpp. */
constexpr double kTiny = 1e-10;
/** @brief The squared-distance acosh floor. See the host header for the derivation. */
constexpr double kAcoshFloor = 1.0 + 1e-12;
/** @brief The oracle's cosine clamp. */
constexpr double kCosClamp = 1.0 - 1e-7;

/**
 * @brief The same carry-the-first-failure wrapper geometric_lowering.cpp uses.
 *
 * A step this long has on the order of two hundred emitter calls; an
 * `if (!v) return false;` after each is where the missing one hides.
 */
struct Ts {
    StableHLOEmitter& e;
    bool ok = true;
    std::string err;

    explicit Ts(StableHLOEmitter& emitter) : e(emitter) {}

    void* fail(const char* what) {
        if (ok) { ok = false; err = what; }
        return nullptr;
    }

    void* add(void* a, void* b) { void* r = (a && b) ? e.emitAdd(a, b) : nullptr; return r ? r : fail("add"); }
    void* sub(void* a, void* b) { void* r = (a && b) ? e.emitSubtract(a, b) : nullptr; return r ? r : fail("subtract"); }
    void* mul(void* a, void* b) { void* r = (a && b) ? e.emitMultiply(a, b) : nullptr; return r ? r : fail("multiply"); }
    void* div(void* a, void* b) { void* r = (a && b) ? e.emitDivide(a, b) : nullptr; return r ? r : fail("divide"); }
    void* neg(void* a)   { void* r = a ? e.emitNegate(a) : nullptr; return r ? r : fail("negate"); }
    void* sqrt_(void* a) { void* r = a ? e.emitSqrt(a) : nullptr;   return r ? r : fail("sqrt"); }
    void* tanh_(void* a) { void* r = a ? e.emitTanh(a) : nullptr;   return r ? r : fail("tanh"); }
    void* log_(void* a)  { void* r = a ? e.emitLog(a) : nullptr;    return r ? r : fail("log"); }
    void* exp_(void* a)  { void* r = a ? e.emitExp(a) : nullptr;    return r ? r : fail("exp"); }
    void* sin_(void* a)  { void* r = a ? e.emitSin(a) : nullptr;    return r ? r : fail("sin"); }
    void* cos_(void* a)  { void* r = a ? e.emitCos(a) : nullptr;    return r ? r : fail("cos"); }
    void* pow_(void* a, void* b) { void* r = (a && b) ? e.emitPow(a, b) : nullptr; return r ? r : fail("pow"); }
    void* atan2_(void* y, void* x) {
        void* r = (y && x) ? e.emitAtan2(y, x) : nullptr;
        return r ? r : fail("atan2");
    }

    void* k(void* like, double value) {
        void* r = like ? e.emitConstantLike(like, value) : nullptr;
        return r ? r : fail("constant");
    }

    void* cmp(void* a, void* b, ComparisonDirection dir) {
        void* r = (a && b) ? e.emitCompare(a, b, dir) : nullptr;
        return r ? r : fail("compare");
    }

    void* sel(void* pred, void* t, void* f) {
        void* r = (pred && t && f) ? e.emitSelect(pred, t, f) : nullptr;
        return r ? r : fail("select");
    }

    /** @brief broadcast_in_dim to @p shape, mapping operand dims by @p dims. */
    void* bcast(void* v, const std::vector<int64_t>& shape, const std::vector<int64_t>& dims) {
        void* r = v ? e.emitBroadcastInDim(v, shape, dims) : nullptr;
        return r ? r : fail("broadcast_in_dim");
    }

    /** @brief Sum-reduce over one axis. */
    void* rsum(void* v, int64_t axis) {
        void* r = v ? e.emitReduce(v, {axis}, StableHLOOp::REDUCE_SUM) : nullptr;
        return r ? r : fail("reduce_sum");
    }

    /** @brief Max-reduce over one axis. */
    void* rmax(void* v, int64_t axis) {
        void* r = v ? e.emitReduce(v, {axis}, StableHLOOp::REDUCE_MAX) : nullptr;
        return r ? r : fail("reduce_max");
    }

    /** @brief Row-wise squared norm of an [r,d]: reduce(v*v) over the last axis. */
    void* rownorm2(void* v) { return rsum(mul(v, v), 1); }

    /**
     * @brief clamp_min as the reference spells it: `a < lo ? lo : a`.
     *
     * NOT maximum(a, lo): under Eshkol's tie convention max gives the tie to
     * its right operand, so at a == lo the input would receive zero where the
     * reference keeps it. Same rule as geometric_lowering.cpp's clampMin.
     */
    void* clampMin(void* a, void* lo) { return sel(cmp(a, lo, ComparisonDirection::LT), lo, a); }
};

/** @brief A vector holding one repeated extent, for the shapes below. */
std::vector<int64_t> sh2(int64_t a, int64_t b) { return {a, b}; }
std::vector<int64_t> sh3(int64_t a, int64_t b, int64_t c) { return {a, b, c}; }

/**
 * @brief Adam's moment update and delta, emitted for one parameter.
 *
 * Transcribed from adamDelta in lib/ml/mixed_curvature_step.cpp, itself from
 * vm_riemannian_adam_delta: eps outside the sqrt, corrections 1 - beta^step,
 * the minus sign in the delta.
 *
 * @param rg    Riemannian gradient, shaped like the parameter.
 * @param m,v   Incoming moments.
 * @param out_m,out_v,out_delta Set on success.
 */
void emitAdam(Ts& t, void* rg, void* m, void* v,
              void* b1, void* b2, void* lr, void* adam_eps, void* bc1, void* bc2,
              const std::vector<int64_t>& shape,
              void** out_m, void** out_v, void** out_delta) {
    void* b1b = t.bcast(b1, shape, {});
    void* b2b = t.bcast(b2, shape, {});
    void* lrb = t.bcast(lr, shape, {});
    void* epsb = t.bcast(adam_eps, shape, {});
    void* bc1b = t.bcast(bc1, shape, {});
    void* bc2b = t.bcast(bc2, shape, {});
    void* one = t.k(b1b, 1.0);

    void* nm = t.add(t.mul(b1b, m), t.mul(t.sub(one, b1b), rg));
    void* nv = t.add(t.mul(b2b, v), t.mul(t.sub(one, b2b), t.mul(rg, rg)));
    void* mh = t.div(nm, bc1b);
    void* vh = t.div(nv, bc2b);
    void* delta = t.neg(t.div(t.mul(lrb, mh), t.add(t.sqrt_(vh), epsb)));

    *out_m = nm;
    *out_v = nv;
    *out_delta = delta;
}

/**
 * @brief exp^c_x(delta) then the radial clip, row-wise over an [rows,d].
 *
 * The batched form of GeometricPrimitive::PoincareExpMap followed by
 * GeometricPrimitive::PoincareRetract with a zero step.
 */
void* emitHyperbolicRetraction(Ts& t, void* x, void* delta, void* c, void* eps,
                               int64_t rows, int64_t d) {
    const std::vector<int64_t> row = {rows};
    const std::vector<int64_t> mat = sh2(rows, d);

    void* cs = t.bcast(c, row, {});
    void* sc = t.sqrt_(cs);
    void* one_r = t.k(cs, 1.0);
    void* two_r = t.k(cs, 2.0);

    void* nv = t.sqrt_(t.rownorm2(delta));
    void* x2 = t.rownorm2(x);
    void* lam = t.div(two_r, t.sub(one_r, t.mul(cs, x2)));
    void* tt = t.div(t.mul(t.mul(sc, lam), nv), two_r);
    void* scale = t.div(t.tanh_(tt), t.mul(sc, nv));
    void* second = t.mul(delta, t.bcast(scale, mat, {0}));

    // Mobius addition, batched.
    void* xy = t.rsum(t.mul(x, second), 1);
    void* y2 = t.rownorm2(second);
    void* two_c_xy = t.mul(t.mul(two_r, cs), xy);
    void* nxc = t.add(t.add(one_r, two_c_xy), t.mul(cs, y2));
    void* nyc = t.sub(one_r, t.mul(cs, x2));
    void* den = t.add(t.add(one_r, two_c_xy), t.mul(t.mul(cs, cs), t.mul(x2, y2)));
    void* num = t.add(t.mul(t.bcast(nxc, mat, {0}), x), t.mul(t.bcast(nyc, mat, {0}), second));
    void* moved = t.div(num, t.bcast(den, mat, {0}));

    void* tiny = t.k(nv, kTiny);
    void* degenerate = t.bcast(t.cmp(nv, tiny, ComparisonDirection::LT), mat, {0});
    moved = t.sel(degenerate, x, moved);

    // Radial clip: PoincareRetract with a zero step.
    void* n2 = t.rownorm2(moved);
    void* epsr = t.bcast(eps, row, {});
    void* maxn2 = t.div(t.sub(one_r, epsr), cs);
    void* clipped = t.sqrt_(t.div(maxn2, t.clampMin(n2, epsr)));
    void* cscale = t.sel(t.cmp(n2, maxn2, ComparisonDirection::GT), clipped, one_r);
    return t.mul(moved, t.bcast(cscale, mat, {0}));
}

/**
 * @brief exp_x(delta) on the sphere then renormalise, row-wise over an [rows,d].
 *
 * GeometricPrimitive::SphereExpMap then SphereRetract with a zero step.
 */
void* emitSphericalRetraction(Ts& t, void* x, void* delta, void* eps,
                              int64_t rows, int64_t d) {
    const std::vector<int64_t> row = {rows};
    const std::vector<int64_t> mat = sh2(rows, d);

    void* nd = t.sqrt_(t.rownorm2(delta));
    void* cn = t.cos_(nd);
    void* sn = t.div(t.sin_(nd), nd);
    void* moved = t.add(t.mul(t.bcast(cn, mat, {0}), x),
                        t.mul(t.bcast(sn, mat, {0}), delta));
    void* tiny = t.k(nd, kTiny);
    moved = t.sel(t.bcast(t.cmp(nd, tiny, ComparisonDirection::LT), mat, {0}), x, moved);

    void* nrm = t.sqrt_(t.rownorm2(moved));
    void* epsr = t.bcast(eps, row, {});
    void* one_r = t.k(nrm, 1.0);
    void* scaled = t.mul(moved, t.bcast(t.div(one_r, t.clampMin(nrm, epsr)), mat, {0}));
    return t.sel(t.bcast(t.cmp(nrm, epsr, ComparisonDirection::GT), mat, {0}), scaled, moved);
}

}  // namespace

std::vector<std::vector<int64_t>> trainingStepInputShapes(EshkolMixedCurvatureShape s) {
    const std::vector<int64_t> w = sh2(s.d, s.d);
    const std::vector<int64_t> p = sh2(s.c, s.d);
    const std::vector<int64_t> scalar = {};
    std::vector<std::vector<int64_t>> shapes(kTsInputCount);
    shapes[kTsW] = w;
    shapes[kTsPHyp] = p;
    shapes[kTsPSph] = p;
    shapes[kTsPEuc] = p;
    shapes[kTsMW] = w;
    shapes[kTsVW] = w;
    shapes[kTsMHyp] = p;
    shapes[kTsVHyp] = p;
    shapes[kTsMSph] = p;
    shapes[kTsVSph] = p;
    shapes[kTsMEuc] = p;
    shapes[kTsVEuc] = p;
    shapes[kTsBatch] = sh2(s.n, s.d);
    shapes[kTsTargets] = sh2(s.n, s.c);
    for (int i = kTsCurvature; i < kTsInputCount; ++i) shapes[i] = scalar;
    return shapes;
}

std::vector<std::vector<int64_t>> trainingStepOutputShapes(EshkolMixedCurvatureShape s) {
    const std::vector<int64_t> w = sh2(s.d, s.d);
    const std::vector<int64_t> p = sh2(s.c, s.d);
    std::vector<std::vector<int64_t>> shapes(kTsOutputCount);
    shapes[kTsOutW] = w;
    shapes[kTsOutPHyp] = p;
    shapes[kTsOutPSph] = p;
    shapes[kTsOutPEuc] = p;
    shapes[kTsOutMW] = w;
    shapes[kTsOutVW] = w;
    shapes[kTsOutMHyp] = p;
    shapes[kTsOutVHyp] = p;
    shapes[kTsOutMSph] = p;
    shapes[kTsOutVSph] = p;
    shapes[kTsOutMEuc] = p;
    shapes[kTsOutVEuc] = p;
    shapes[kTsOutLoss] = {};
    return shapes;
}

std::string trainingStepCacheKey(EshkolMixedCurvatureShape s, ElementType elem) {
    std::ostringstream os;
    os << "train|n" << s.n << "|d" << s.d << "|c" << s.c
       << "|" << (elem == ElementType::F64 ? "f64" : "f32");
    return os.str();
}

std::string shardedTrainingStepCacheKey(EshkolMixedCurvatureShape shard, ElementType elem,
                                        const TrainingStepSharding& sharding) {
    std::ostringstream os;
    os << trainingStepCacheKey(shard, elem)
       << "|rep" << sharding.num_replicas
       << "|rows" << (sharding.loss_rows > 0 ? sharding.loss_rows : shard.n)
       << (sharding.omit_all_reduce ? "|NO_ALL_REDUCE" : "");
    return os.str();
}

bool buildTrainingStepModule(EshkolMixedCurvatureShape s, ElementType elem,
                             std::string* module_text, std::string* error) {
    return buildShardedTrainingStepModule(s, elem, TrainingStepSharding{}, module_text, error);
}

bool buildShardedTrainingStepModule(EshkolMixedCurvatureShape s, ElementType elem,
                                    const TrainingStepSharding& sharding,
                                    std::string* module_text, std::string* error) {
    if (!module_text || !error) return false;
    if (s.n <= 0 || s.d <= 0 || s.c <= 0) {
        *error = "the training step needs positive n, d and c";
        return false;
    }
    if (sharding.num_replicas < 1) {
        *error = "the training step needs num_replicas >= 1";
        return false;
    }
    // The mean's denominator: the full batch, of which this module sees a
    // shard. See TrainingStepSharding for why this is n and not n / N.
    const int64_t loss_rows = sharding.loss_rows > 0 ? sharding.loss_rows : s.n;
    const bool reduce = sharding.num_replicas > 1 && !sharding.omit_all_reduce;

    StableHLOEmitter emitter;
    if (!emitter.isAvailable()) {
        *error = "this build has no StableHLO emitter";
        return false;
    }

    const int64_t n = s.n, d = s.d, nc = s.c;
    const std::vector<int64_t> shp_w = sh2(d, d);
    const std::vector<int64_t> shp_p = sh2(nc, d);
    const std::vector<int64_t> shp_nd = sh2(n, d);
    const std::vector<int64_t> shp_ncls = sh2(n, nc);
    const std::vector<int64_t> shp_ncd = sh3(n, nc, d);
    const std::vector<int64_t> shp_row_n = {n};
    const std::vector<int64_t> shp_row_c = {nc};

    std::vector<StableHLOEmitter::ParamSpec> params;
    for (const std::vector<int64_t>& sh : trainingStepInputShapes(s)) {
        params.push_back({sh, elem});
    }
    std::vector<void*> a = emitter.beginFunction("main", params);
    if (a.size() != params.size()) {
        *error = "beginFunction did not return one argument per parameter";
        return false;
    }

    Ts t(emitter);

    void* pw = a[kTsW];
    void* phyp = a[kTsPHyp];
    void* psph = a[kTsPSph];
    void* peuc = a[kTsPEuc];
    void* xb = a[kTsBatch];
    void* tg = a[kTsTargets];
    void* cs0 = a[kTsCurvature];
    void* alpha0 = a[kTsAlpha];
    void* guard0 = a[kTsGuardEps];

    // ---------------- forward ----------------

    // U = X W. dot_general contracting X's dim 1 against W's dim 0.
    DotDimensionNumbers dot{};
    dot.lhs_contracting_dims = {1};
    dot.rhs_contracting_dims = {0};
    void* u = emitter.emitMatmul(xb, pw, dot);
    if (!u) { *error = "could not emit the projection matmul"; return false; }

    // Z = exp_0^c(alpha U), GeometricPrimitive::PoincareExpMapOrigin per row.
    void* csn = t.bcast(cs0, shp_row_n, {});
    void* scn = t.sqrt_(csn);
    void* v = t.mul(u, t.bcast(alpha0, shp_nd, {}));
    void* nv = t.sqrt_(t.rownorm2(v));
    void* tt = t.mul(scn, nv);
    void* kcoef = t.div(t.tanh_(tt), tt);
    void* zscaled = t.mul(v, t.bcast(kcoef, shp_nd, {0}));
    void* zdegen = t.bcast(t.cmp(nv, t.k(nv, kTiny), ComparisonDirection::LT), shp_nd, {0});
    void* z = t.sel(zdegen, v, zscaled);

    // Dh = d_c(Z, P_hyp)^2, batched HyperbolicDistance, squared.
    void* zb = t.bcast(z, shp_ncd, {0, 2});
    void* phb = t.bcast(phyp, shp_ncd, {1, 2});
    void* hdiff = t.sub(zb, phb);
    void* q = t.rsum(t.mul(hdiff, hdiff), 2);                    // [n,c]
    void* one_nc = t.k(q, 1.0);
    void* two_nc = t.k(q, 2.0);
    void* csnc = t.bcast(cs0, shp_ncls, {});
    void* zn2 = t.rownorm2(z);                                   // [n]
    void* pn2 = t.rownorm2(phyp);                                // [c]
    void* ax = t.sub(one_nc, t.mul(csnc, t.bcast(zn2, shp_ncls, {0})));
    void* bx = t.sub(one_nc, t.mul(csnc, t.bcast(pn2, shp_ncls, {1})));
    void* raw = t.add(one_nc, t.div(t.mul(t.mul(two_nc, csnc), q), t.mul(ax, bx)));
    void* floor_nc = t.k(raw, kAcoshFloor);
    void* aarg = t.sel(t.cmp(raw, floor_nc, ComparisonDirection::LT), floor_nc, raw);
    void* acosh = t.log_(t.add(aarg, t.sqrt_(t.sub(t.mul(aarg, aarg), one_nc))));
    void* dist = t.div(acosh, t.sqrt_(csnc));
    void* dh = t.mul(dist, dist);

    // Y = U / clamp_min(|U|, eps); Ds = theta(Y, P_sph)^2.
    void* guard_n = t.bcast(guard0, shp_row_n, {});
    void* rn = t.sqrt_(t.rownorm2(u));
    void* y = t.div(u, t.bcast(t.clampMin(rn, guard_n), shp_nd, {0}));
    void* yb = t.bcast(y, shp_ncd, {0, 2});
    void* psb = t.bcast(psph, shp_ncd, {1, 2});
    void* ip = t.rsum(t.mul(yb, psb), 2);                        // [n,c]
    void* cc = t.k(ip, kCosClamp);
    void* mcc = t.k(ip, -kCosClamp);
    void* hi = t.sel(t.cmp(ip, cc, ComparisonDirection::GT), cc, ip);
    void* sclamped = t.sel(t.cmp(hi, mcc, ComparisonDirection::LT), mcc, hi);
    void* one_ip = t.k(sclamped, 1.0);
    void* th = t.atan2_(t.sqrt_(t.sub(one_ip, t.mul(sclamped, sclamped))), sclamped);
    void* ds = t.mul(th, th);

    // De = |U - P_euc|^2.
    void* ub = t.bcast(u, shp_ncd, {0, 2});
    void* peb = t.bcast(peuc, shp_ncd, {1, 2});
    void* ediff = t.sub(ub, peb);
    void* de = t.rsum(t.mul(ediff, ediff), 2);

    // Logits and the mean soft-target cross entropy.
    void* wh = t.bcast(a[kTsWHyp], shp_ncls, {});
    void* ws = t.bcast(a[kTsWSph], shp_ncls, {});
    void* we = t.bcast(a[kTsWEuc], shp_ncls, {});
    void* logits = t.neg(t.add(t.add(t.mul(wh, dh), t.mul(ws, ds)), t.mul(we, de)));

    void* mx = t.rmax(logits, 1);                                // [n]
    void* shifted = t.sub(logits, t.bcast(mx, shp_ncls, {0}));
    void* sumexp = t.rsum(t.exp_(shifted), 1);                   // [n]
    void* lse = t.add(mx, t.log_(sumexp));
    void* dott = t.rsum(t.mul(tg, logits), 1);                   // [n]
    void* per_row = t.sub(lse, dott);
    void* total = t.rsum(per_row, 0);                            // rank-0
    void* nrows = t.k(total, static_cast<double>(loss_rows));
    void* loss = t.div(total, nrows);

    if (!t.ok) {
        *error = std::string("could not emit the training step forward: ") + t.err;
        return false;
    }

    // ---------------- backward ----------------
    // Automatic. No VJP rule for this step is written anywhere; the host's
    // backward is derived by hand instead, and the harness compares them.
    VJPResult vjp = emitter.emitVJP(loss, {pw, phyp, psph, peuc}, nullptr);
    if (!vjp.complete) {
        *error = std::string("no device gradient for the training step: ") +
                 (vjp.diagnostic.empty() ? "emitVJP reported no diagnostic" : vjp.diagnostic);
        return false;
    }
    if (vjp.gradients.size() != 4) {
        *error = "emitVJP returned " + std::to_string(vjp.gradients.size()) +
                 " gradients for the training step's four parameters";
        return false;
    }
    void* gw = vjp.gradients[0];
    void* gph = vjp.gradients[1];
    void* gps = vjp.gradients[2];
    void* gpe = vjp.gradients[3];

    // ---------------- cross-replica reduction (S8) ----------------
    // Each replica's gradients are of shard_total / n; their sum over the
    // replica group is the full-batch mean gradient. The partial losses sum
    // the same way. The optimizer below then runs on the reduced gradient,
    // identically on every replica. With one replica there is nothing to
    // reduce and no op is emitted, so the N = 1 module is unchanged.
    if (reduce) {
        const int64_t R = sharding.num_replicas;
        gw = emitter.emitAllReduceSum(gw, R);
        gph = emitter.emitAllReduceSum(gph, R);
        gps = emitter.emitAllReduceSum(gps, R);
        gpe = emitter.emitAllReduceSum(gpe, R);
        loss = emitter.emitAllReduceSum(loss, R);
        if (!gw || !gph || !gps || !gpe || !loss) {
            *error = "could not emit the cross-replica all_reduce of the gradients";
            return false;
        }
    }

    // ---------------- Riemannian Adam ----------------
    void* lr = a[kTsLr];
    void* b1 = a[kTsBeta1];
    void* b2 = a[kTsBeta2];
    void* aeps = a[kTsAdamEps];
    void* step = a[kTsStep];
    void* one0 = t.k(step, 1.0);
    void* bc1 = t.sub(one0, t.pow_(b1, step));
    void* bc2 = t.sub(one0, t.pow_(b2, step));

    // W: Euclidean gradient, Euclidean retraction.
    void *mw = nullptr, *vw = nullptr, *dw = nullptr;
    emitAdam(t, gw, a[kTsMW], a[kTsVW], b1, b2, lr, aeps, bc1, bc2, shp_w, &mw, &vw, &dw);
    void* new_w = t.add(pw, dw);

    // P_euc: Euclidean.
    void *me = nullptr, *ve = nullptr, *dpe = nullptr;
    emitAdam(t, gpe, a[kTsMEuc], a[kTsVEuc], b1, b2, lr, aeps, bc1, bc2, shp_p, &me, &ve, &dpe);
    void* new_pe = t.add(peuc, dpe);

    // P_hyp: PoincareProject rescaling, then the hyperbolic retraction.
    void* guard_c = t.bcast(guard0, shp_row_c, {});
    void* csc = t.bcast(cs0, shp_row_c, {});
    void* one_c = t.k(csc, 1.0);
    void* hx2 = t.rownorm2(phyp);
    void* conf = t.clampMin(t.sub(one_c, t.mul(csc, hx2)), guard_c);
    void* quarter = t.k(csc, 0.25);
    void* hscale = t.mul(quarter, t.mul(conf, conf));
    void* rg_h = t.mul(gph, t.bcast(hscale, shp_p, {0}));
    void *mh_o = nullptr, *vh_o = nullptr, *dph = nullptr;
    emitAdam(t, rg_h, a[kTsMHyp], a[kTsVHyp], b1, b2, lr, aeps, bc1, bc2, shp_p,
             &mh_o, &vh_o, &dph);
    void* new_ph = emitHyperbolicRetraction(t, phyp, dph, cs0, guard0, nc, d);

    // P_sph: SphereProject rescaling, then the spherical retraction.
    void* sdot = t.rsum(t.mul(gps, psph), 1);
    void* rg_s = t.sub(gps, t.mul(psph, t.bcast(sdot, shp_p, {0})));
    void *ms_o = nullptr, *vs_o = nullptr, *dps = nullptr;
    emitAdam(t, rg_s, a[kTsMSph], a[kTsVSph], b1, b2, lr, aeps, bc1, bc2, shp_p,
             &ms_o, &vs_o, &dps);
    void* new_ps = emitSphericalRetraction(t, psph, dps, guard0, nc, d);

    if (!t.ok) {
        *error = std::string("could not emit the training step optimizer: ") + t.err;
        return false;
    }

    std::vector<void*> results(kTsOutputCount, nullptr);
    results[kTsOutW] = new_w;
    results[kTsOutPHyp] = new_ph;
    results[kTsOutPSph] = new_ps;
    results[kTsOutPEuc] = new_pe;
    results[kTsOutMW] = mw;
    results[kTsOutVW] = vw;
    results[kTsOutMHyp] = mh_o;
    results[kTsOutVHyp] = vh_o;
    results[kTsOutMSph] = ms_o;
    results[kTsOutVSph] = vs_o;
    results[kTsOutMEuc] = me;
    results[kTsOutVEuc] = ve;
    results[kTsOutLoss] = loss;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i]) {
            *error = "training step result " + std::to_string(i) + " was never produced";
            return false;
        }
    }

    if (!emitter.endFunction(results)) {
        *error = "endFunction failed for the training step";
        return false;
    }

    *module_text = emitter.serializeToString();
    if (module_text->empty()) {
        *error = "serializeToString produced an empty training step module";
        return false;
    }
    return true;
}

namespace {

ElementType executorElementType(DeviceExecutor* executor) {
    return executor->dtypeName() == "f64" ? ElementType::F64 : ElementType::F32;
}

}  // namespace

bool runTrainingStep(DeviceExecutor* executor,
                     EshkolMixedCurvatureShape s,
                     const EshkolMixedCurvatureHyper& hyper,
                     EshkolMixedCurvatureParams* params,
                     EshkolMixedCurvatureMoments* moments,
                     const double* batch,
                     const double* targets,
                     double* out_loss,
                     std::string* error) {
    std::string local;
    std::string* err = error ? error : &local;
    if (!executor) { *err = "no device executor"; return false; }
    if (!params || !moments || !batch || !targets) { *err = "null training step argument"; return false; }
    if (hyper.curvature <= 0.0) { *err = "the training step needs a positive curvature"; return false; }

    const ElementType elem = executorElementType(executor);
    std::string module;
    if (!buildTrainingStepModule(s, elem, &module, err)) return false;

    const int64_t step = moments->step + 1;
    const double scalars[] = {
        hyper.curvature, hyper.alpha, hyper.lr, hyper.beta1, hyper.beta2,
        hyper.adam_eps, hyper.guard_eps, static_cast<double>(step),
        hyper.w_hyp, hyper.w_sph, hyper.w_euc
    };

    std::vector<const double*> operands(kTsInputCount, nullptr);
    operands[kTsW] = params->w;
    operands[kTsPHyp] = params->p_hyp;
    operands[kTsPSph] = params->p_sph;
    operands[kTsPEuc] = params->p_euc;
    operands[kTsMW] = moments->m_w;
    operands[kTsVW] = moments->v_w;
    operands[kTsMHyp] = moments->m_hyp;
    operands[kTsVHyp] = moments->v_hyp;
    operands[kTsMSph] = moments->m_sph;
    operands[kTsVSph] = moments->v_sph;
    operands[kTsMEuc] = moments->m_euc;
    operands[kTsVEuc] = moments->v_euc;
    operands[kTsBatch] = batch;
    operands[kTsTargets] = targets;
    for (int i = kTsCurvature; i < kTsInputCount; ++i) {
        operands[i] = &scalars[i - kTsCurvature];
    }
    for (size_t i = 0; i < operands.size(); ++i) {
        if (!operands[i]) {
            *err = "training step operand " + std::to_string(i) + " is null";
            return false;
        }
    }

    // Staging buffers. The device writes here, and the caller's parameters and
    // moments are overwritten only once every one of the thirteen results has
    // come back — a step that updated three of four parameters is a model that
    // still runs and is wrong.
    const int64_t we = s.d * s.d, pe = s.c * s.d;
    std::vector<double> out_w(static_cast<size_t>(we));
    std::vector<double> out_ph(static_cast<size_t>(pe));
    std::vector<double> out_ps(static_cast<size_t>(pe));
    std::vector<double> out_pe(static_cast<size_t>(pe));
    std::vector<double> o_mw(static_cast<size_t>(we)), o_vw(static_cast<size_t>(we));
    std::vector<double> o_mh(static_cast<size_t>(pe)), o_vh(static_cast<size_t>(pe));
    std::vector<double> o_ms(static_cast<size_t>(pe)), o_vs(static_cast<size_t>(pe));
    std::vector<double> o_me(static_cast<size_t>(pe)), o_ve(static_cast<size_t>(pe));
    double loss = 0.0;

    std::vector<double*> results(kTsOutputCount, nullptr);
    results[kTsOutW] = out_w.data();
    results[kTsOutPHyp] = out_ph.data();
    results[kTsOutPSph] = out_ps.data();
    results[kTsOutPEuc] = out_pe.data();
    results[kTsOutMW] = o_mw.data();
    results[kTsOutVW] = o_vw.data();
    results[kTsOutMHyp] = o_mh.data();
    results[kTsOutVHyp] = o_vh.data();
    results[kTsOutMSph] = o_ms.data();
    results[kTsOutVSph] = o_vs.data();
    results[kTsOutMEuc] = o_me.data();
    results[kTsOutVEuc] = o_ve.data();
    results[kTsOutLoss] = &loss;

    if (!executor->runModule(module, trainingStepCacheKey(s, elem),
                             trainingStepInputShapes(s), operands,
                             trainingStepOutputShapes(s), results, err)) {
        return false;
    }

    std::memcpy(params->w, out_w.data(), sizeof(double) * static_cast<size_t>(we));
    std::memcpy(params->p_hyp, out_ph.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(params->p_sph, out_ps.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(params->p_euc, out_pe.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(moments->m_w, o_mw.data(), sizeof(double) * static_cast<size_t>(we));
    std::memcpy(moments->v_w, o_vw.data(), sizeof(double) * static_cast<size_t>(we));
    std::memcpy(moments->m_hyp, o_mh.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(moments->v_hyp, o_vh.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(moments->m_sph, o_ms.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(moments->v_sph, o_vs.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(moments->m_euc, o_me.data(), sizeof(double) * static_cast<size_t>(pe));
    std::memcpy(moments->v_euc, o_ve.data(), sizeof(double) * static_cast<size_t>(pe));
    moments->step = step;
    if (out_loss) *out_loss = loss;
    return true;
}

bool runTrainingStepSharded(DeviceExecutor* executor,
                            EshkolMixedCurvatureShape s,
                            const EshkolMixedCurvatureHyper& hyper,
                            int num_replicas,
                            EshkolMixedCurvatureParams* params,
                            EshkolMixedCurvatureMoments* moments,
                            const double* batch,
                            const double* targets,
                            double* out_loss,
                            ShardedStepReport* report,
                            const TrainingStepSharding* control,
                            std::string* error) {
    std::string local;
    std::string* err = error ? error : &local;
    if (!executor) { *err = "no device executor"; return false; }
    if (!params || !moments || !batch || !targets) { *err = "null training step argument"; return false; }
    if (hyper.curvature <= 0.0) { *err = "the training step needs a positive curvature"; return false; }
    if (num_replicas < 1) { *err = "num_replicas must be >= 1"; return false; }
    if (s.n % num_replicas != 0) {
        *err = "batch rows " + std::to_string(s.n) + " are not divisible by " +
               std::to_string(num_replicas) + " replicas; a ragged shard would change the "
               "mean's denominator on one replica";
        return false;
    }
    const int available = executor->addressableDeviceCount();
    if (num_replicas > available) {
        *err = "requested " + std::to_string(num_replicas) + " replicas but the executor has " +
               std::to_string(available) + " addressable device(s)";
        return false;
    }

    const ElementType elem = executorElementType(executor);
    const size_t R = static_cast<size_t>(num_replicas);
    EshkolMixedCurvatureShape shard{s.n / num_replicas, s.d, s.c};

    TrainingStepSharding sharding;
    sharding.num_replicas = num_replicas;
    sharding.loss_rows = s.n;
    sharding.omit_all_reduce = control ? control->omit_all_reduce : false;

    std::string module;
    if (!buildShardedTrainingStepModule(shard, elem, sharding, &module, err)) return false;

    const int64_t step = moments->step + 1;
    const double scalars[] = {
        hyper.curvature, hyper.alpha, hyper.lr, hyper.beta1, hyper.beta2,
        hyper.adam_eps, hyper.guard_eps, static_cast<double>(step),
        hyper.w_hyp, hyper.w_sph, hyper.w_euc
    };

    // Replicated operands are the same host pointers on every row; the batch
    // and targets are the only per-replica rows, offset into the full arrays.
    const int64_t shard_x = shard.n * s.d;
    const int64_t shard_t = shard.n * s.c;
    std::vector<std::vector<const double*>> operands(R, std::vector<const double*>(kTsInputCount, nullptr));
    for (size_t r = 0; r < R; ++r) {
        std::vector<const double*>& o = operands[r];
        o[kTsW] = params->w;
        o[kTsPHyp] = params->p_hyp;
        o[kTsPSph] = params->p_sph;
        o[kTsPEuc] = params->p_euc;
        o[kTsMW] = moments->m_w;
        o[kTsVW] = moments->v_w;
        o[kTsMHyp] = moments->m_hyp;
        o[kTsVHyp] = moments->v_hyp;
        o[kTsMSph] = moments->m_sph;
        o[kTsVSph] = moments->v_sph;
        o[kTsMEuc] = moments->m_euc;
        o[kTsVEuc] = moments->v_euc;
        o[kTsBatch] = batch + static_cast<int64_t>(r) * shard_x;
        o[kTsTargets] = targets + static_cast<int64_t>(r) * shard_t;
        for (int i = kTsCurvature; i < kTsInputCount; ++i) o[i] = &scalars[i - kTsCurvature];
        for (size_t i = 0; i < o.size(); ++i) {
            if (!o[i]) { *err = "training step operand " + std::to_string(i) + " is null"; return false; }
        }
    }

    // One full set of result staging per replica: every replica's outputs
    // come back, so that agreement between them can be measured rather than
    // presumed from the fact that they ran the same program.
    const std::vector<std::vector<int64_t>> out_shapes = trainingStepOutputShapes(shard);
    std::vector<std::vector<std::vector<double>>> staging(R);
    std::vector<std::vector<double*>> results(R);
    for (size_t r = 0; r < R; ++r) {
        staging[r].resize(kTsOutputCount);
        results[r].resize(kTsOutputCount);
        for (int i = 0; i < kTsOutputCount; ++i) {
            int64_t n = 1;
            for (int64_t d : out_shapes[static_cast<size_t>(i)]) n *= d;
            staging[r][static_cast<size_t>(i)].assign(static_cast<size_t>(n), 0.0);
            results[r][static_cast<size_t>(i)] = staging[r][static_cast<size_t>(i)].data();
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    if (!executor->runModuleReplicated(module, shardedTrainingStepCacheKey(shard, elem, sharding),
                                       num_replicas, trainingStepInputShapes(shard), operands,
                                       out_shapes, results, err)) {
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();

    if (report) {
        *report = ShardedStepReport{};
        report->num_replicas = num_replicas;
        report->replicas_identical = true;
        report->device_seconds = std::chrono::duration<double>(t1 - t0).count();
        for (size_t r = 1; r < R; ++r) {
            for (int i = 0; i < kTsOutputCount; ++i) {
                const std::vector<double>& a = staging[0][static_cast<size_t>(i)];
                const std::vector<double>& b = staging[r][static_cast<size_t>(i)];
                for (size_t j = 0; j < a.size(); ++j) {
                    if (std::memcmp(&a[j], &b[j], sizeof(double)) != 0) {
                        report->replicas_identical = false;
                        const double diff = std::fabs(a[j] - b[j]);
                        if (!(diff <= report->max_replica_abs_diff)) {
                            report->max_replica_abs_diff = diff;
                            report->worst_output = i;
                            report->worst_replica = static_cast<int>(r);
                        }
                    }
                }
            }
        }
    }

    // Replica 0's results are the model's new state. Written only now, once
    // every replica has returned every result — the same no-partial-success
    // rule runTrainingStep keeps.
    const size_t we = static_cast<size_t>(s.d * s.d), pe = static_cast<size_t>(s.c * s.d);
    const std::vector<std::vector<double>>& o = staging[0];
    std::memcpy(params->w, o[kTsOutW].data(), sizeof(double) * we);
    std::memcpy(params->p_hyp, o[kTsOutPHyp].data(), sizeof(double) * pe);
    std::memcpy(params->p_sph, o[kTsOutPSph].data(), sizeof(double) * pe);
    std::memcpy(params->p_euc, o[kTsOutPEuc].data(), sizeof(double) * pe);
    std::memcpy(moments->m_w, o[kTsOutMW].data(), sizeof(double) * we);
    std::memcpy(moments->v_w, o[kTsOutVW].data(), sizeof(double) * we);
    std::memcpy(moments->m_hyp, o[kTsOutMHyp].data(), sizeof(double) * pe);
    std::memcpy(moments->v_hyp, o[kTsOutVHyp].data(), sizeof(double) * pe);
    std::memcpy(moments->m_sph, o[kTsOutMSph].data(), sizeof(double) * pe);
    std::memcpy(moments->v_sph, o[kTsOutVSph].data(), sizeof(double) * pe);
    std::memcpy(moments->m_euc, o[kTsOutMEuc].data(), sizeof(double) * pe);
    std::memcpy(moments->v_euc, o[kTsOutVEuc].data(), sizeof(double) * pe);
    moments->step = step;
    if (out_loss) *out_loss = o[kTsOutLoss][0];
    return true;
}

}  // namespace xla
}  // namespace eshkol
