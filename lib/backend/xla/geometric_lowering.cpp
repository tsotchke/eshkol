/**
 * @file geometric_lowering.cpp
 * @brief The device compositions for the mixed-curvature geometric primitives.
 *
 * Every formula here is transcribed from the exporters in tests/qllm_oracle/,
 * which generated tests/qllm_oracle/golden/*.json. The transcription is
 * line-by-line on purpose: a formula rearranged into a "nicer" algebraic form
 * would be a different sequence of roundings, and then a disagreement with the
 * golden Jacobian could be either a lowering defect or the rearrangement, with
 * no way to tell which. Where a reference does something numerically odd (an
 * unclamped denominator next to a clamped numerator, an artanh clamp that
 * makes the operator locally constant) that oddity is reproduced and the
 * comment says so.
 *
 * WHY THE GUARDS ARE SELECTS AND NOT max/min.
 *
 * `clamp_min(a, eps)` in the reference is `(if (< a eps) eps a)`. Its VALUE is
 * maximum(a, eps), but its GRADIENT at a == eps is not: Eshkol's elementwise
 * max gives a tie to its right-hand operand (eps, a constant, so the input
 * would receive zero), while clamp_min's own comparison is strict and keeps
 * `a`. Emitting a maximum here would be right everywhere except exactly on the
 * clamp, which is the one place a clamp exists to be — so each guard is a
 * select on the same comparison the reference performs.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/geometric_lowering.h"

#include <sstream>
#include <string>
#include <vector>

#include "eshkol/backend/xla/stablehlo_emitter.h"

namespace eshkol {
namespace xla {

namespace {

/**
 * @brief The scalar short-circuit threshold the reference uses for "the
 *        tangent vector is zero" (TINY in tests/qllm_oracle/poincare_maps.esk).
 */
constexpr double kTiny = 1e-10;

/**
 * @brief qLLM's artanh clamp. Above it the operator is constant in its
 *        argument and its exact derivative is identically zero — which is a
 *        property of the shipped kernel, not an approximation of one, and is
 *        why the golden corpus contains a case that sits above it.
 */
constexpr double kArtanhClamp = 1.0 - 1e-7;

/**
 * @brief A tiny wrapper that carries the first failure instead of returning
 *        null through twenty call sites.
 *
 * Every helper returns null once `ok` is false, so a composition can be
 * written as the formula reads and checked once at the end. The alternative —
 * an `if (!v) return false;` after each of forty emitter calls — is where a
 * missing check hides.
 */
struct Geo {
    StableHLOEmitter& e;
    int64_t d;
    bool ok = true;
    std::string err;

    Geo(StableHLOEmitter& emitter, int64_t dim) : e(emitter), d(dim) {}

    void* fail(const char* what) {
        if (ok) { ok = false; err = what; }
        return nullptr;
    }

    void* add(void* a, void* b) { void* r = (a && b) ? e.emitAdd(a, b) : nullptr; return r ? r : fail("add"); }
    void* sub(void* a, void* b) { void* r = (a && b) ? e.emitSubtract(a, b) : nullptr; return r ? r : fail("subtract"); }
    void* mul(void* a, void* b) { void* r = (a && b) ? e.emitMultiply(a, b) : nullptr; return r ? r : fail("multiply"); }
    void* div(void* a, void* b) { void* r = (a && b) ? e.emitDivide(a, b) : nullptr; return r ? r : fail("divide"); }
    void* neg(void* a)          { void* r = a ? e.emitNegate(a) : nullptr;           return r ? r : fail("negate"); }
    void* sqrt_(void* a)        { void* r = a ? e.emitSqrt(a) : nullptr;             return r ? r : fail("sqrt"); }
    void* tanh_(void* a)        { void* r = a ? e.emitTanh(a) : nullptr;             return r ? r : fail("tanh"); }
    void* log_(void* a)         { void* r = a ? e.emitLog(a) : nullptr;              return r ? r : fail("log"); }
    void* atanh_(void* a)       { void* r = a ? e.emitAtanh(a) : nullptr;            return r ? r : fail("atanh"); }
    void* sin_(void* a)         { void* r = a ? e.emitSin(a) : nullptr;              return r ? r : fail("sin"); }
    void* cos_(void* a)         { void* r = a ? e.emitCos(a) : nullptr;              return r ? r : fail("cos"); }
    void* atan2_(void* y, void* x) {
        void* r = (y && x) ? e.emitAtan2(y, x) : nullptr;
        return r ? r : fail("atan2");
    }

    /** @brief A constant splat shaped like @p like. */
    void* k(void* like, double value) {
        void* r = like ? e.emitConstantLike(like, value) : nullptr;
        return r ? r : fail("constant");
    }

    /** @brief Broadcast a rank-0 value to [d]. */
    void* splat(void* scalar) {
        void* r = scalar ? e.emitBroadcastInDim(scalar, {d}, {}) : nullptr;
        return r ? r : fail("scalar broadcast to [d]");
    }

    /** @brief <a, b> as a rank-0 value. */
    void* dot(void* a, void* b) {
        void* p = mul(a, b);
        void* r = p ? e.emitReduce(p, {0}, StableHLOOp::REDUCE_SUM) : nullptr;
        return r ? r : fail("dot product reduce");
    }

    /** @brief |a| as a rank-0 value. */
    void* norm(void* a) { return sqrt_(dot(a, a)); }

    /** @brief Rank-0 comparison; the result is an i1 rank-0 tensor. */
    void* cmp(void* a, void* b, ComparisonDirection dir) {
        void* r = (a && b) ? e.emitCompare(a, b, dir) : nullptr;
        return r ? r : fail("compare");
    }

    /** @brief Rank-0 select. */
    void* sel(void* pred, void* t, void* f) {
        void* r = (pred && t && f) ? e.emitSelect(pred, t, f) : nullptr;
        return r ? r : fail("scalar select");
    }

    /**
     * @brief Vector select on a RANK-0 predicate.
     *
     * The predicate is broadcast to [d] rather than relied on to splat: a
     * scalar-predicate select is legal StableHLO but its behaviour under the
     * VJP walk would then depend on the predicate's rank, and one shape of
     * select in the graph is one rule to be sure of.
     */
    void* selv(void* pred, void* t, void* f) {
        void* pv = pred ? e.emitBroadcastInDim(pred, {d}, {}) : nullptr;
        if (!pv) return fail("predicate broadcast to [d]");
        void* r = (t && f) ? e.emitSelect(pv, t, f) : nullptr;
        return r ? r : fail("vector select");
    }

    /**
     * @brief clamp_min(a, lo) as the reference spells it: a < lo ? lo : a.
     *        See the file comment for why this is not maximum(a, lo).
     */
    void* clampMin(void* a, void* lo) {
        return sel(cmp(a, lo, ComparisonDirection::LT), lo, a);
    }

    /**
     * @brief qLLM's artanh: clamp the argument at 1 - 1e-7, then artanh.
     *
     * The clamp is INSIDE the operator being differentiated, so it is emitted
     * before the artanh and not applied to the result: above the clamp the
     * value is constant and the derivative is zero, which is the behaviour the
     * golden corpus records and the behaviour a finite difference cannot see.
     */
    void* qllmArtanh(void* arg) {
        void* clamp = k(arg, kArtanhClamp);
        void* a = sel(cmp(arg, clamp, ComparisonDirection::GE), clamp, arg);
        return atanh_(a);
    }

    /**
     * @brief Clamp a rank-0 cosine into [-1, 1] the way the reference does:
     *        `(if (> cs 1) 1 cs)` then `(if (< cs -1) -1 cs)`, i.e. two
     *        comparisons, not a maximum/minimum pair. Same tie reason as
     *        clampMin() above.
     */
    void* clampUnit(void* t) {
        void* one = k(t, 1.0);
        void* mone = k(t, -1.0);
        void* hi = sel(cmp(t, one, ComparisonDirection::GT), one, t);
        return sel(cmp(hi, mone, ComparisonDirection::LT), mone, hi);
    }

    /**
     * @brief acos(t) as atan2(sqrt(1 - t^2), t).
     *
     * StableHLO has no acos. This identity is exact for t in [-1, 1] and is
     * the standard way to get one from an atan2: it is well conditioned near
     * t = +/-1, where acos has infinite slope but both atan2 arguments stay
     * finite and correctly signed — which an `atan(sqrt(1-t^2)/t)` spelling
     * does not (it loses the quadrant at t < 0 and divides by zero at t = 0).
     */
    void* acos_(void* t) {
        void* one = k(t, 1.0);
        void* s = sqrt_(sub(one, mul(t, t)));
        return atan2_(s, t);
    }

    /**
     * @brief Mobius addition, transcribed from `mobius-add` in
     *        tests/qllm_oracle/poincare_maps.esk.
     *
     *   x (+)_c y = ((1 + 2c<x,y> + c|y|^2) x + (1 - c|x|^2) y)
     *               / (1 + 2c<x,y> + c^2|x|^2|y|^2)
     *
     * The reference divides by the denominator with NO floor. The host bridge
     * (lib/bridge/qllm_bridge.cpp) floors it at 1e-15; the two agree
     * everywhere the denominator is not within 1e-15 of zero, which on the
     * Poincare ball means everywhere both points are interior. This follows
     * the reference, because the golden Jacobians are the reference's.
     *
     * @param cs rank-0 curvature
     */
    void* mobiusAdd(void* x, void* y, void* cs) {
        void* xy = dot(x, y);
        void* x2 = dot(x, x);
        void* y2 = dot(y, y);
        void* one = k(xy, 1.0);
        void* two = k(xy, 2.0);
        void* two_c_xy = mul(mul(two, cs), xy);
        void* nx = add(add(one, two_c_xy), mul(cs, y2));
        void* ny = sub(one, mul(cs, x2));
        void* den = add(add(one, two_c_xy), mul(mul(cs, cs), mul(x2, y2)));
        void* num = add(mul(splat(nx), x), mul(splat(ny), y));
        return div(num, splat(den));
    }

    /** @brief lambda_x = 2/(1 - c|x|^2), rank-0. */
    void* conformalLambda(void* x, void* cs) {
        void* x2 = dot(x, x);
        void* one = k(x2, 1.0);
        void* two = k(x2, 2.0);
        return div(two, sub(one, mul(cs, x2)));
    }
};

/** @brief Emit the forward body; returns the single result value. */
void* emitGeometricBody(Geo& g, GeometricPrimitive p, const std::vector<void*>& args) {
    // args: vector operands first, then scalars, exactly as the header states.
    switch (p) {
        case GeometricPrimitive::MobiusAdd:
            return g.mobiusAdd(args[0], args[1], args[2]);

        case GeometricPrimitive::PoincareExpMapOrigin: {
            // (if (< n TINY) v (vscale v (/ (tanh t) t))), t = sqrt(c)|v|
            void* v = args[0];
            void* cs = args[1];
            void* n = g.norm(v);
            void* t = g.mul(g.sqrt_(cs), n);
            void* scale = g.div(g.tanh_(t), t);
            void* scaled = g.mul(v, g.splat(scale));
            return g.selv(g.cmp(n, g.k(n, kTiny), ComparisonDirection::LT), v, scaled);
        }

        case GeometricPrimitive::PoincareLogMapOrigin: {
            // (if (< n TINY) y (vscale y (/ (qllm-artanh t) t))), t = sqrt(c)|y|
            // Note the denominator is the UNCLAMPED t, exactly as
            // fast_log_map_origin has it; only the artanh argument is clamped.
            void* y = args[0];
            void* cs = args[1];
            void* n = g.norm(y);
            void* t = g.mul(g.sqrt_(cs), n);
            void* scale = g.div(g.qllmArtanh(t), t);
            void* scaled = g.mul(y, g.splat(scale));
            return g.selv(g.cmp(n, g.k(n, kTiny), ComparisonDirection::LT), y, scaled);
        }

        case GeometricPrimitive::PoincareExpMap: {
            // lam = 2/(1-c|x|^2); t = sc*lam*|v|/2;
            // second = v * tanh(t)/(sc|v|); out = x (+)_c second
            void* x = args[0];
            void* v = args[1];
            void* cs = args[2];
            void* sc = g.sqrt_(cs);
            void* nv = g.norm(v);
            void* lam = g.conformalLambda(x, cs);
            void* two = g.k(nv, 2.0);
            void* t = g.div(g.mul(g.mul(sc, lam), nv), two);
            void* scale = g.div(g.tanh_(t), g.mul(sc, nv));
            void* second = g.mul(v, g.splat(scale));
            void* moved = g.mobiusAdd(x, second, cs);
            return g.selv(g.cmp(nv, g.k(nv, kTiny), ComparisonDirection::LT), x, moved);
        }

        case GeometricPrimitive::PoincareLogMap: {
            // u = (-x) (+)_c y; out = u * (2/(sc*lam)) * artanh(sc|u|) / |u|
            void* x = args[0];
            void* y = args[1];
            void* cs = args[2];
            void* sc = g.sqrt_(cs);
            void* u = g.mobiusAdd(g.neg(x), y, cs);
            void* nu = g.norm(u);
            void* lam = g.conformalLambda(x, cs);
            void* two = g.k(nu, 2.0);
            void* front = g.div(two, g.mul(sc, lam));
            void* scale = g.div(g.mul(front, g.qllmArtanh(g.mul(sc, nu))), nu);
            void* scaled = g.mul(u, g.splat(scale));
            void* zeros = g.e.emitZerosLike(u);
            if (!zeros) return g.fail("zeros for the degenerate log map");
            return g.selv(g.cmp(nu, g.k(nu, kTiny), ComparisonDirection::LT), zeros, scaled);
        }

        case GeometricPrimitive::HyperbolicDistance: {
            // The host bridge's formula (ad_hyperbolic_distance):
            //   arg = 1 + 2c|x-y|^2 / ((1-c|x|^2)(1-c|y|^2)), floored at 1
            //   d   = acosh(arg)/sqrt(c)
            // acosh is not a StableHLO op; acosh(a) = log(a + sqrt(a^2 - 1)),
            // whose derivative 1/sqrt(a^2-1) is infinite at a = 1, i.e. at
            // coincident points — which is the derivative the distance
            // actually has there, so it is not guarded away.
            void* x = args[0];
            void* y = args[1];
            void* cs = args[2];
            void* diff = g.sub(x, y);
            void* diff2 = g.dot(diff, diff);
            void* one = g.k(diff2, 1.0);
            void* two = g.k(diff2, 2.0);
            void* dx = g.sub(one, g.mul(cs, g.dot(x, x)));
            void* dy = g.sub(one, g.mul(cs, g.dot(y, y)));
            void* raw = g.add(one, g.div(g.mul(g.mul(two, cs), diff2), g.mul(dx, dy)));
            // (if (< arg 1) 1 arg) — the reference's own floor, spelled as its
            // comparison rather than as a maximum, for the tie reason above.
            void* arg = g.sel(g.cmp(raw, one, ComparisonDirection::LT), one, raw);
            void* acosh = g.log_(g.add(arg, g.sqrt_(g.sub(g.mul(arg, arg), one))));
            return g.div(acosh, g.sqrt_(cs));
        }

        case GeometricPrimitive::PoincareProject: {
            // conf = clamp_min(1 - c|x|^2, eps); out = 0.25 conf^2 g
            void* x = args[0];
            void* grad = args[1];
            void* cs = args[2];
            void* eps = args[3];
            void* x2 = g.dot(x, x);
            void* one = g.k(x2, 1.0);
            void* conf = g.clampMin(g.sub(one, g.mul(cs, x2)), eps);
            void* quarter = g.k(x2, 0.25);
            void* scale = g.mul(quarter, g.mul(conf, conf));
            return g.mul(grad, g.splat(scale));
        }

        case GeometricPrimitive::PoincareRetract: {
            // z = x + step; max_n2 = (1-eps)/c;
            // scale = (if (> |z|^2 max_n2) sqrt(max_n2/clamp_min(|z|^2, eps)) 1)
            void* x = args[0];
            void* step = args[1];
            void* cs = args[2];
            void* eps = args[3];
            void* z = g.add(x, step);
            void* n2 = g.dot(z, z);
            void* one = g.k(n2, 1.0);
            void* maxn2 = g.div(g.sub(one, eps), cs);
            void* clipped = g.sqrt_(g.div(maxn2, g.clampMin(n2, eps)));
            void* scale = g.sel(g.cmp(n2, maxn2, ComparisonDirection::GT), clipped, one);
            return g.mul(z, g.splat(scale));
        }

        case GeometricPrimitive::SphereProject: {
            // out = g - <g, x> x
            void* x = args[0];
            void* grad = args[1];
            return g.sub(grad, g.mul(x, g.splat(g.dot(grad, x))));
        }

        case GeometricPrimitive::SphereRetract: {
            // z = x + step; (if (> |z| eps) (vscale z (/ 1 (clamp_min |z| eps))) x)
            void* x = args[0];
            void* step = args[1];
            void* eps = args[2];
            void* z = g.add(x, step);
            void* n = g.norm(z);
            void* one = g.k(n, 1.0);
            void* scaled = g.mul(z, g.splat(g.div(one, g.clampMin(n, eps))));
            return g.selv(g.cmp(n, eps, ComparisonDirection::GT), scaled, x);
        }

        case GeometricPrimitive::SphereExpMap: {
            // exp_x(v) = cos|v| x + (sin|v|/|v|) v, the geodesic on the unit
            // sphere. Degenerate at v = 0, where the limit is x.
            void* x = args[0];
            void* v = args[1];
            void* n = g.norm(v);
            void* moved = g.add(g.mul(x, g.splat(g.cos_(n))),
                                g.mul(v, g.splat(g.div(g.sin_(n), n))));
            return g.selv(g.cmp(n, g.k(n, kTiny), ComparisonDirection::LT), x, moved);
        }

        case GeometricPrimitive::SphereLogMap: {
            // u = y - <x,y> x is the component of y tangent at x; the geodesic
            // from x toward y has length theta = acos<x,y> and direction u/|u|.
            void* x = args[0];
            void* y = args[1];
            void* ip = g.clampUnit(g.dot(x, y));
            void* u = g.sub(y, g.mul(x, g.splat(ip)));
            void* nu = g.norm(u);
            void* scaled = g.mul(u, g.splat(g.div(g.acos_(ip), nu)));
            void* zeros = g.e.emitZerosLike(u);
            if (!zeros) return g.fail("zeros for the degenerate sphere log map");
            return g.selv(g.cmp(nu, g.k(nu, kTiny), ComparisonDirection::LT), zeros, scaled);
        }

        case GeometricPrimitive::SphericalDistance: {
            // The great-circle distance as the VM's geometric fallback
            // computes it (native id 819): acos of the normalised inner
            // product, with the cosine clamped into [-1,1] first.
            void* x = args[0];
            void* y = args[1];
            void* nx = g.norm(x);
            void* ny = g.norm(y);
            void* cs = g.clampUnit(g.div(g.dot(x, y), g.mul(nx, ny)));
            return g.acos_(cs);
        }

        case GeometricPrimitive::EuclideanExpMap:
            return g.add(args[0], args[1]);

        case GeometricPrimitive::EuclideanLogMap:
            return g.sub(args[1], args[0]);

        case GeometricPrimitive::EuclideanDistance: {
            void* diff = g.sub(args[0], args[1]);
            return g.sqrt_(g.dot(diff, diff));
        }
    }
    return g.fail("unhandled geometric primitive");
}

}  // namespace

const char* geometricPrimitiveName(GeometricPrimitive p) {
    switch (p) {
        case GeometricPrimitive::MobiusAdd:            return "mobius_add";
        case GeometricPrimitive::PoincareExpMapOrigin: return "poincare_exp_map_origin";
        case GeometricPrimitive::PoincareLogMapOrigin: return "poincare_log_map_origin";
        case GeometricPrimitive::PoincareExpMap:       return "poincare_exp_map";
        case GeometricPrimitive::PoincareLogMap:       return "poincare_log_map";
        case GeometricPrimitive::HyperbolicDistance:   return "hyperbolic_distance";
        case GeometricPrimitive::PoincareProject:      return "poincare_project";
        case GeometricPrimitive::PoincareRetract:      return "poincare_retract";
        case GeometricPrimitive::SphereProject:        return "sphere_project";
        case GeometricPrimitive::SphereRetract:        return "sphere_retract";
        case GeometricPrimitive::SphereExpMap:         return "sphere_exp_map";
        case GeometricPrimitive::SphereLogMap:         return "sphere_log_map";
        case GeometricPrimitive::SphericalDistance:    return "spherical_distance";
        case GeometricPrimitive::EuclideanExpMap:      return "euclidean_exp_map";
        case GeometricPrimitive::EuclideanLogMap:      return "euclidean_log_map";
        case GeometricPrimitive::EuclideanDistance:    return "euclidean_distance";
    }
    return "unknown";
}

int geometricVectorOperands(GeometricPrimitive p) {
    switch (p) {
        case GeometricPrimitive::PoincareExpMapOrigin:
        case GeometricPrimitive::PoincareLogMapOrigin:
            return 1;
        default:
            return 2;
    }
}

int geometricScalarOperands(GeometricPrimitive p) {
    switch (p) {
        case GeometricPrimitive::MobiusAdd:
        case GeometricPrimitive::PoincareExpMapOrigin:
        case GeometricPrimitive::PoincareLogMapOrigin:
        case GeometricPrimitive::PoincareExpMap:
        case GeometricPrimitive::PoincareLogMap:
        case GeometricPrimitive::HyperbolicDistance:
            return 1;   // c
        case GeometricPrimitive::PoincareProject:
        case GeometricPrimitive::PoincareRetract:
            return 2;   // c, eps
        case GeometricPrimitive::SphereRetract:
            return 1;   // eps
        case GeometricPrimitive::SphereProject:
        case GeometricPrimitive::SphereExpMap:
        case GeometricPrimitive::SphereLogMap:
        case GeometricPrimitive::SphericalDistance:
        case GeometricPrimitive::EuclideanExpMap:
        case GeometricPrimitive::EuclideanLogMap:
        case GeometricPrimitive::EuclideanDistance:
            return 0;
    }
    return 0;
}

bool geometricResultIsScalar(GeometricPrimitive p) {
    return p == GeometricPrimitive::HyperbolicDistance ||
           p == GeometricPrimitive::SphericalDistance ||
           p == GeometricPrimitive::EuclideanDistance;
}

std::string geometricCacheKey(GeometricPrimitive p, int64_t dim, ElementType elem,
                              bool with_gradient) {
    std::ostringstream os;
    os << "geo|" << geometricPrimitiveName(p) << "|d" << dim
       << "|" << (elem == ElementType::F64 ? "f64" : "f32")
       << "|" << (with_gradient ? "grad" : "fwd");
    return os.str();
}

bool buildGeometricModule(GeometricPrimitive p, int64_t dim, ElementType elem,
                          bool with_gradient, std::string* module_text,
                          std::string* error) {
    if (!module_text || !error) return false;
    if (dim <= 0) { *error = "geometric lowering needs a positive dimension"; return false; }

    StableHLOEmitter emitter;
    if (!emitter.isAvailable()) {
        *error = "this build has no StableHLO emitter";
        return false;
    }

    const int n_vec = geometricVectorOperands(p);
    const int n_scal = geometricScalarOperands(p);
    const std::vector<int64_t> vshape = {dim};
    const std::vector<int64_t> sshape = {};
    const std::vector<int64_t> rshape = geometricResultIsScalar(p) ? sshape : vshape;

    std::vector<StableHLOEmitter::ParamSpec> params;
    for (int i = 0; i < n_vec; ++i) params.push_back({vshape, elem});
    for (int i = 0; i < n_scal; ++i) params.push_back({sshape, elem});
    if (with_gradient) params.push_back({rshape, elem});

    std::vector<void*> args = emitter.beginFunction("main", params);
    if (args.size() != params.size()) {
        *error = "beginFunction did not return one argument per parameter";
        return false;
    }

    Geo g(emitter, dim);
    std::vector<void*> operand_args(args.begin(), args.begin() + (n_vec + n_scal));
    void* out = emitGeometricBody(g, p, operand_args);
    if (!g.ok || !out) {
        *error = std::string("could not emit ") + geometricPrimitiveName(p) + ": " +
                 (g.err.empty() ? "unknown emitter failure" : g.err);
        return false;
    }

    if (!with_gradient) {
        if (!emitter.endFunction({out})) {
            *error = std::string("endFunction failed for ") + geometricPrimitiveName(p);
            return false;
        }
    } else {
        // Differentiate with respect to the VECTOR operands only; see the
        // header for why curvature and the guard epsilons are excluded.
        std::vector<void*> wrt(args.begin(), args.begin() + n_vec);
        void* seed = args[static_cast<size_t>(n_vec + n_scal)];
        VJPResult vjp = emitter.emitVJP(out, wrt, seed);
        if (!vjp.complete) {
            *error = std::string("no device gradient for ") + geometricPrimitiveName(p) + ": " +
                     (vjp.diagnostic.empty() ? "emitVJP reported no diagnostic" : vjp.diagnostic);
            return false;
        }
        if (vjp.gradients.size() != static_cast<size_t>(n_vec)) {
            *error = std::string("emitVJP returned ") + std::to_string(vjp.gradients.size()) +
                     " gradients for " + std::to_string(n_vec) + " vector operands of " +
                     geometricPrimitiveName(p);
            return false;
        }
        if (!emitter.endFunction(vjp.gradients)) {
            *error = std::string("endFunction failed for the gradient of ") +
                     geometricPrimitiveName(p);
            return false;
        }
    }

    *module_text = emitter.serializeToString();
    if (module_text->empty()) {
        *error = std::string("serializeToString produced an empty module for ") +
                 geometricPrimitiveName(p);
        return false;
    }
    return true;
}

namespace {

/** @brief The operand shapes a primitive's module declares, in order. */
std::vector<std::vector<int64_t>> operandShapes(GeometricPrimitive p, int64_t dim) {
    std::vector<std::vector<int64_t>> shapes;
    for (int i = 0; i < geometricVectorOperands(p); ++i) shapes.push_back({dim});
    for (int i = 0; i < geometricScalarOperands(p); ++i) shapes.push_back({});
    return shapes;
}

/** @brief The device element type the executor reports, as an ElementType. */
ElementType executorElementType(DeviceExecutor* executor) {
    return executor->dtypeName() == "f64" ? ElementType::F64 : ElementType::F32;
}

}  // namespace

bool runGeometric(DeviceExecutor* executor, GeometricPrimitive p, int64_t dim,
                  const std::vector<const double*>& operands,
                  double* result, std::string* error) {
    std::string local;
    std::string* err = error ? error : &local;
    if (!executor) { *err = "no device executor"; return false; }
    if (!result) { *err = "null result pointer"; return false; }

    std::vector<std::vector<int64_t>> shapes = operandShapes(p, dim);
    if (operands.size() != shapes.size()) {
        *err = std::string("operand count for ") + geometricPrimitiveName(p) + ": expected " +
               std::to_string(shapes.size()) + ", got " + std::to_string(operands.size());
        return false;
    }

    const ElementType elem = executorElementType(executor);
    std::string module;
    if (!buildGeometricModule(p, dim, elem, false, &module, err)) return false;

    const std::vector<int64_t> rshape =
        geometricResultIsScalar(p) ? std::vector<int64_t>{} : std::vector<int64_t>{dim};
    return executor->runModule(module, geometricCacheKey(p, dim, elem, false),
                               shapes, operands, {rshape}, {result}, err);
}

bool runGeometricGradient(DeviceExecutor* executor, GeometricPrimitive p, int64_t dim,
                          const std::vector<const double*>& operands,
                          const double* cotangent,
                          const std::vector<double*>& gradients,
                          std::string* error) {
    std::string local;
    std::string* err = error ? error : &local;
    if (!executor) { *err = "no device executor"; return false; }

    std::vector<std::vector<int64_t>> shapes = operandShapes(p, dim);
    if (operands.size() != shapes.size()) {
        *err = std::string("operand count for ") + geometricPrimitiveName(p) + ": expected " +
               std::to_string(shapes.size()) + ", got " + std::to_string(operands.size());
        return false;
    }
    const int n_vec = geometricVectorOperands(p);
    if (gradients.size() != static_cast<size_t>(n_vec)) {
        *err = std::string("gradient destination count for ") + geometricPrimitiveName(p) +
               ": expected " + std::to_string(n_vec) + ", got " +
               std::to_string(gradients.size());
        return false;
    }

    const ElementType elem = executorElementType(executor);
    std::string module;
    if (!buildGeometricModule(p, dim, elem, true, &module, err)) return false;

    const bool scalar_result = geometricResultIsScalar(p);
    const std::vector<int64_t> rshape =
        scalar_result ? std::vector<int64_t>{} : std::vector<int64_t>{dim};

    // A null cotangent means a ones seed, materialised here rather than baked
    // into the module: the module is cached by shape, so a constant seed
    // inside it would answer a later request that supplied a real cotangent
    // with the constant. Same rule as runGradient() in device_lowering.cpp.
    std::vector<double> ones;
    if (!cotangent) {
        ones.assign(scalar_result ? 1u : static_cast<size_t>(dim), 1.0);
        cotangent = ones.data();
    }

    std::vector<std::vector<int64_t>> in_shapes = shapes;
    in_shapes.push_back(rshape);
    std::vector<const double*> ins = operands;
    ins.push_back(cotangent);

    std::vector<std::vector<int64_t>> out_shapes(static_cast<size_t>(n_vec),
                                                 std::vector<int64_t>{dim});
    return executor->runModule(module, geometricCacheKey(p, dim, elem, true),
                               in_shapes, ins, out_shapes, gradients, err);
}

}  // namespace xla
}  // namespace eshkol
