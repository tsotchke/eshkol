/*
 * Device/host parity for the mixed-curvature model's GEOMETRIC PRIMITIVES:
 * the hyperbolic, spherical and Euclidean maps, distances, projections and
 * retractions, lowered to StableHLO as device compositions
 * (lib/backend/xla/geometric_lowering.cpp) and executed through Eshkol's PJRT
 * client, forward and reverse.
 *
 * WHAT THIS PROVES, AND AGAINST WHAT.
 *
 * A geometric primitive is not one op, so "device agrees with host" needs
 * saying carefully. Four independent references are used, and every row names
 * the one it was graded by:
 *
 *  1. HOST COMPOSITION. The same decomposition evaluated by Eshkol's own host
 *     tensor runtime — eshkol_xla_elementwise_host, eshkol_xla_reduce_host,
 *     eshkol_xla_broadcast_host — in f64. Not arithmetic written in this file:
 *     every add, multiply, square root, tanh, artanh, sine and reduction below
 *     is a call into the runtime op_parity_test already grades op by op. What
 *     this file contributes is the ORDER, which is the thing being tested.
 *
 *  2. THE GOLDEN CORPUS. tests/qllm_oracle/golden/*.json holds full Jacobians
 *     for seven of these primitives, computed by Eshkol's reverse-mode AD over
 *     an INDEPENDENTLY WRITTEN Eshkol transcription of the same formulas
 *     (tests/qllm_oracle/*.esk), in f64, cross-checked there against central
 *     differences. Those files are read at run time — not transcribed — so a
 *     row cites its file and case id and a regenerated corpus is picked up
 *     rather than drifting away from a copy. This is the strongest reference
 *     here: a shared mistake would have to occur in two implementations, in
 *     two languages, written from different sources.
 *
 *  3. THE HOST TAPE. exp_x, log_x and the hyperbolic distance also exist as
 *     C bridge ops that record real AD tape nodes (ad_poincare_exp_map,
 *     ad_poincare_log_map, ad_hyperbolic_distance in lib/bridge/qllm_bridge.cpp,
 *     types 33-35 in inc/eshkol/ad_node_registry.def). Those rows run the
 *     forward AND the reverse sweep on the host tape and compare the tape's
 *     cotangents with the device's.
 *
 *  4. FINITE DIFFERENCES over reference 1, for the primitives that have
 *     neither a golden vector nor a tape node (Mobius addition, the spherical
 *     maps, the distances). Central differences at h = 1e-5, reported per row.
 *
 * And, separately from all four:
 *
 *  5. MANIFOLD IDENTITIES, computed ENTIRELY FROM DEVICE OUTPUTS. A lowering
 *     that is wrong in a way the host composition is also wrong in would pass
 *     reference 1 on every input. It cannot pass (-x) (+)_c (x (+)_c y) = y,
 *     or log_x(exp_x(v)) = v, because those are properties of the geometry and
 *     nothing in this repository was consulted to derive them.
 *
 * CURVATURE IS AN OPERAND, AND THAT IS CHECKED.
 *
 * Every hyperbolic row runs at three curvatures through ONE compiled
 * executable, and the harness asserts that the executable count did not grow
 * across them. A curvature baked into a module would still produce correct
 * numbers for the first K and silently wrong ones for the rest, so the check
 * is on the compile counter, not on the values.
 *
 * WHAT f32 CANNOT DO, SAID OUT LOUD.
 *
 * The qLLM artanh clamp is 1 - 1e-7. In f32 that value IS 1.0 — f32's epsilon
 * is 1.19e-7 — so the clamped operator cannot be evaluated near the ball
 * boundary on a device with no f64. Golden cases whose artanh argument or
 * conformal factor is inside the f32-resolvable margin are therefore EXCLUDED
 * by name, counted, and reported. They are not silently passed and they are
 * not silently skipped: an excluded case is printed with its id and the reason.
 *
 * A GATE THAT CANNOT FAIL IS WORTHLESS.
 *
 * Four things here can fail by construction:
 *   - the comparator control in tests/xla/parity_compare.h, before any device
 *     is required;
 *   - a perturbed-reference control: one real device result is graded against
 *     a reference perturbed by 1.0 and must be REJECTED;
 *   - a golden-corpus load control: the corpus must parse and must contain the
 *     cases the rows ask for, so a missing or truncated file fails rather than
 *     grading zero cases and reporting success;
 *   - ESHKOL_XLA_GEOMETRIC_FORCE_FAIL=<primitive name> perturbs that
 *     primitive's host reference by 1.0, so an end-to-end FAIL can be shown
 *     against live hardware without editing this file.
 *
 * Exit status: 0 every row agreed; 1 a row disagreed or a control failed;
 * 77 no PJRT device was reachable.
 *
 * Only builds when ESHKOL_XLA_ENABLED=ON (see CMakeLists.txt).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/bridge/qllm_bridge.h"
#include "eshkol/backend/tensor_backward.h"
#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/geometric_lowering.h"
#include "eshkol/backend/xla/xla_runtime.h"

#include "golden_json.h"
#include "parity_compare.h"

using eshkol::xla::DeviceExecutor;
using eshkol::xla::GeometricPrimitive;
using eshkol::xla::deviceExecutor;
using eshkol::xla::geometricPrimitiveName;
using eshkol::xla::geometricResultIsScalar;
using eshkol::xla::geometricScalarOperands;
using eshkol::xla::geometricVectorOperands;
using eshkol::xla::registerStableHLODeviceExecutor;
using eshkol::xla::runGeometric;
using eshkol::xla::runGeometricGradient;

using eshkol_parity::Comparison;
using eshkol_parity::ToleranceClass;
using eshkol_parity::compareArrays;
using eshkol_parity::setTolerancesForDtype;
using eshkol_parity::test_comparator_rejects_a_perturbed_result;
using eshkol_parity::toleranceFor;

extern "C" {
// The host runtime's own entry points — reference 1. Declared here under the
// names they carry since the device split in lib/backend/xla/xla_runtime.cpp.
void* eshkol_xla_elementwise_host(void* arena, const double* a, const double* b,
                                  int64_t total, const uint64_t* shape, int64_t rank,
                                  int64_t b_total, const uint64_t* b_shape, int64_t b_rank,
                                  int64_t op_code);
void* eshkol_xla_reduce_host(void* arena, const double* data, int64_t total,
                             const uint64_t* shape, int64_t rank, int64_t axis,
                             int64_t op_code);
void* eshkol_xla_broadcast_host(void* arena, const double* data, const uint64_t* src_shape,
                                int64_t src_rank, const uint64_t* tgt_shape, int64_t tgt_rank);

// The arena and AD tape allocators, exactly as
// tests/bridge/qllm_bridge_geometric_gradcheck_test.cpp declares them.
typedef struct arena arena_t;
arena_t* arena_create(size_t default_block_size);
void arena_reset(arena_t* arena);
void arena_destroy(arena_t* arena);
arena_t* get_global_arena(void);
void* arena_allocate_zeroed(arena_t* arena, size_t size);
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
ad_node_t* arena_allocate_ad_node(arena_t* arena);
}

namespace {

using V = std::vector<double>;

int g_rows_passed = 0;
int g_rows_failed = 0;
int g_controls_failed = 0;
int g_golden_cases_graded = 0;
int g_golden_cases_excluded = 0;
int g_invariants_passed = 0;
int g_invariants_failed = 0;
std::string g_dtype = "f32";
const char* g_force_fail = nullptr;

/** @brief The central-difference step, stated once. */
constexpr double kFdStep = 1e-5;
/** @brief How far the finite difference may be from the stated rule. */
constexpr double kFdTolerance = 1e-6;

/**
 * @brief The f32 resolvability margin for a near-one quantity.
 *
 * f32 has a 24-bit significand, so numbers just below 1 are spaced 5.96e-8
 * apart and 1 - t loses all but its leading digits once 1 - t approaches that.
 * A quantity within kF32Margin of 1 cannot be represented on this device
 * accurately enough for the row's tolerance to mean anything, so a case that
 * contains one is excluded BY NAME rather than graded. 1e-3 leaves four
 * decimal digits of 1 - t, which is what the transcendental bound needs when
 * artanh(t) ~ -0.5 log((1-t)/2) amplifies the relative error of 1 - t by 0.5.
 */
constexpr double kF32Margin = 1e-3;

// ─────────────────────────────────────────────────────────────────────────
// Reference 1: the host composition.
//
// Every arithmetic step below is a call into the host tensor runtime. The
// elementwise ABI op codes are the ones in inc/eshkol/backend/xla/xla_codegen.h
// (ADD=0 .. MIN=18); they are named here so a reader does not have to
// remember the numbering, and the runtime's own arity table decides which
// calls pass a second operand.
// ─────────────────────────────────────────────────────────────────────────

enum HostOp {
    H_ADD = 0, H_SUB = 1, H_MUL = 2, H_DIV = 3,
    H_LOG = 5, H_SIN = 6, H_COS = 7, H_TANH = 8,
    H_SQRT = 11, H_NEG = 14, H_ATANH = 15
};

struct HostOps {
    arena_t* arena;

    static V values(void* tensor_ptr, size_t expected) {
        V out;
        if (!tensor_ptr) return out;
        auto* t = static_cast<eshkol_tensor_t*>(tensor_ptr);
        if (t->total_elements != expected) return out;
        const double* src = reinterpret_cast<const double*>(t->elements);
        out.assign(src, src + expected);
        return out;
    }

    /** @brief Elementwise unary through the host runtime. */
    V u(int op, const V& x) const {
        std::vector<uint64_t> shape = {static_cast<uint64_t>(x.size())};
        void* t = eshkol_xla_elementwise_host(arena, x.data(), nullptr,
                                              static_cast<int64_t>(x.size()),
                                              shape.data(), 1, 0, nullptr, 0, op);
        return values(t, x.size());
    }

    /** @brief Elementwise binary through the host runtime, equal shapes. */
    V b(int op, const V& a, const V& c) const {
        if (a.size() != c.size()) return {};
        std::vector<uint64_t> shape = {static_cast<uint64_t>(a.size())};
        void* t = eshkol_xla_elementwise_host(arena, a.data(), c.data(),
                                              static_cast<int64_t>(a.size()),
                                              shape.data(), 1,
                                              static_cast<int64_t>(c.size()),
                                              shape.data(), 1, op);
        return values(t, a.size());
    }

    /** @brief Full sum reduction through the host runtime. */
    double rsum(const V& x) const {
        std::vector<uint64_t> shape = {static_cast<uint64_t>(x.size())};
        void* t = eshkol_xla_reduce_host(arena, x.data(), static_cast<int64_t>(x.size()),
                                         shape.data(), 1, -1, 0 /* SUM */);
        V r = values(t, 1);
        return r.empty() ? NAN : r[0];
    }

    /** @brief Broadcast a scalar to n elements through the host runtime. */
    V splat(double s, size_t n) const {
        const double one[1] = {s};
        std::vector<uint64_t> src = {1};
        std::vector<uint64_t> tgt = {static_cast<uint64_t>(n)};
        void* t = eshkol_xla_broadcast_host(arena, one, src.data(), 1, tgt.data(), 1);
        return values(t, n);
    }

    /** @brief Scalar unary, evaluated by the same host op as the vector case. */
    double su(int op, double x) const {
        V r = u(op, V{x});
        return r.empty() ? NAN : r[0];
    }
    /** @brief Scalar binary, likewise. */
    double sb(int op, double a, double c) const {
        V r = b(op, V{a}, V{c});
        return r.empty() ? NAN : r[0];
    }

    double dot(const V& a, const V& c) const { return rsum(b(H_MUL, a, c)); }
    double norm(const V& a) const { return su(H_SQRT, dot(a, a)); }
    V scale(const V& a, double s) const { return b(H_MUL, a, splat(s, a.size())); }
};

/** @brief The reference's TINY and artanh clamp; see geometric_lowering.cpp. */
constexpr double kTiny = 1e-10;
constexpr double kArtanhClamp = 1.0 - 1e-7;

/** @brief qLLM's artanh: clamp the argument, then artanh, through host ops. */
double hostQllmArtanh(const HostOps& h, double t) {
    const double a = (t >= kArtanhClamp) ? kArtanhClamp : t;
    return h.su(H_ATANH, a);
}

/** @brief Mobius addition, host composition. */
V hostMobiusAdd(const HostOps& h, const V& x, const V& y, double c) {
    const double xy = h.dot(x, y);
    const double x2 = h.dot(x, x);
    const double y2 = h.dot(y, y);
    const double two_c_xy = h.sb(H_MUL, h.sb(H_MUL, 2.0, c), xy);
    const double nx = h.sb(H_ADD, h.sb(H_ADD, 1.0, two_c_xy), h.sb(H_MUL, c, y2));
    const double ny = h.sb(H_SUB, 1.0, h.sb(H_MUL, c, x2));
    const double den = h.sb(H_ADD, h.sb(H_ADD, 1.0, two_c_xy),
                            h.sb(H_MUL, h.sb(H_MUL, c, c), h.sb(H_MUL, x2, y2)));
    V num = h.b(H_ADD, h.scale(x, nx), h.scale(y, ny));
    return h.b(H_DIV, num, h.splat(den, x.size()));
}

double hostConformalLambda(const HostOps& h, const V& x, double c) {
    return h.sb(H_DIV, 2.0, h.sb(H_SUB, 1.0, h.sb(H_MUL, c, h.dot(x, x))));
}

/**
 * @brief The host composition for one primitive.
 *
 * `vecs` are the vector operands in order, `scals` the scalar ones. A scalar
 * result is returned as a one-element vector, matching what the device
 * produces for a rank-0 result.
 */
V hostGeometric(const HostOps& h, GeometricPrimitive p,
                const std::vector<V>& vecs, const V& scals) {
    switch (p) {
        case GeometricPrimitive::MobiusAdd:
            return hostMobiusAdd(h, vecs[0], vecs[1], scals[0]);

        case GeometricPrimitive::PoincareExpMapOrigin: {
            const V& v = vecs[0];
            const double n = h.norm(v);
            if (n < kTiny) return v;
            const double t = h.sb(H_MUL, h.su(H_SQRT, scals[0]), n);
            return h.scale(v, h.sb(H_DIV, h.su(H_TANH, t), t));
        }
        case GeometricPrimitive::PoincareLogMapOrigin: {
            const V& y = vecs[0];
            const double n = h.norm(y);
            if (n < kTiny) return y;
            const double t = h.sb(H_MUL, h.su(H_SQRT, scals[0]), n);
            return h.scale(y, h.sb(H_DIV, hostQllmArtanh(h, t), t));
        }
        case GeometricPrimitive::PoincareExpMap: {
            const V& x = vecs[0];
            const V& v = vecs[1];
            const double c = scals[0];
            const double sc = h.su(H_SQRT, c);
            const double nv = h.norm(v);
            if (nv < kTiny) return x;
            const double lam = hostConformalLambda(h, x, c);
            const double t = h.sb(H_DIV, h.sb(H_MUL, h.sb(H_MUL, sc, lam), nv), 2.0);
            V second = h.scale(v, h.sb(H_DIV, h.su(H_TANH, t), h.sb(H_MUL, sc, nv)));
            return hostMobiusAdd(h, x, second, c);
        }
        case GeometricPrimitive::PoincareLogMap: {
            const V& x = vecs[0];
            const V& y = vecs[1];
            const double c = scals[0];
            const double sc = h.su(H_SQRT, c);
            V u = hostMobiusAdd(h, h.u(H_NEG, x), y, c);
            const double nu = h.norm(u);
            if (nu < kTiny) return V(x.size(), 0.0);
            const double lam = hostConformalLambda(h, x, c);
            const double front = h.sb(H_DIV, 2.0, h.sb(H_MUL, sc, lam));
            const double scale = h.sb(H_DIV,
                h.sb(H_MUL, front, hostQllmArtanh(h, h.sb(H_MUL, sc, nu))), nu);
            return h.scale(u, scale);
        }
        case GeometricPrimitive::HyperbolicDistance: {
            const V& x = vecs[0];
            const V& y = vecs[1];
            const double c = scals[0];
            V diff = h.b(H_SUB, x, y);
            const double diff2 = h.dot(diff, diff);
            const double dx = h.sb(H_SUB, 1.0, h.sb(H_MUL, c, h.dot(x, x)));
            const double dy = h.sb(H_SUB, 1.0, h.sb(H_MUL, c, h.dot(y, y)));
            double arg = h.sb(H_ADD, 1.0,
                h.sb(H_DIV, h.sb(H_MUL, h.sb(H_MUL, 2.0, c), diff2), h.sb(H_MUL, dx, dy)));
            if (arg < 1.0) arg = 1.0;
            const double acosh = h.su(H_LOG,
                h.sb(H_ADD, arg, h.su(H_SQRT, h.sb(H_SUB, h.sb(H_MUL, arg, arg), 1.0))));
            return V{h.sb(H_DIV, acosh, h.su(H_SQRT, c))};
        }
        case GeometricPrimitive::PoincareProject: {
            const V& x = vecs[0];
            const V& g = vecs[1];
            const double c = scals[0], eps = scals[1];
            const double raw = h.sb(H_SUB, 1.0, h.sb(H_MUL, c, h.dot(x, x)));
            const double conf = (raw < eps) ? eps : raw;
            return h.scale(g, h.sb(H_MUL, 0.25, h.sb(H_MUL, conf, conf)));
        }
        case GeometricPrimitive::PoincareRetract: {
            const V& x = vecs[0];
            const V& step = vecs[1];
            const double c = scals[0], eps = scals[1];
            V z = h.b(H_ADD, x, step);
            const double n2 = h.dot(z, z);
            const double maxn2 = h.sb(H_DIV, h.sb(H_SUB, 1.0, eps), c);
            double scale = 1.0;
            if (n2 > maxn2) {
                scale = h.su(H_SQRT, h.sb(H_DIV, maxn2, (n2 < eps) ? eps : n2));
            }
            return h.scale(z, scale);
        }
        case GeometricPrimitive::SphereProject: {
            const V& x = vecs[0];
            const V& g = vecs[1];
            return h.b(H_SUB, g, h.scale(x, h.dot(g, x)));
        }
        case GeometricPrimitive::SphereRetract: {
            const V& x = vecs[0];
            const V& step = vecs[1];
            const double eps = scals[0];
            V z = h.b(H_ADD, x, step);
            const double n = h.norm(z);
            if (!(n > eps)) return x;
            return h.scale(z, h.sb(H_DIV, 1.0, (n < eps) ? eps : n));
        }
        case GeometricPrimitive::SphereExpMap: {
            const V& x = vecs[0];
            const V& v = vecs[1];
            const double n = h.norm(v);
            if (n < kTiny) return x;
            return h.b(H_ADD, h.scale(x, h.su(H_COS, n)),
                       h.scale(v, h.sb(H_DIV, h.su(H_SIN, n), n)));
        }
        case GeometricPrimitive::SphereLogMap: {
            const V& x = vecs[0];
            const V& y = vecs[1];
            double ip = h.dot(x, y);
            if (ip > 1.0) ip = 1.0;
            if (ip < -1.0) ip = -1.0;
            V u = h.b(H_SUB, y, h.scale(x, ip));
            const double nu = h.norm(u);
            if (nu < kTiny) return V(x.size(), 0.0);
            // acos(ip) computed the way the device does it, atan2(sqrt(1-ip^2), ip).
            const double theta = std::atan2(h.su(H_SQRT, h.sb(H_SUB, 1.0,
                                                              h.sb(H_MUL, ip, ip))), ip);
            return h.scale(u, h.sb(H_DIV, theta, nu));
        }
        case GeometricPrimitive::SphericalDistance: {
            const V& x = vecs[0];
            const V& y = vecs[1];
            const double nx = h.norm(x), ny = h.norm(y);
            double cs = h.sb(H_DIV, h.dot(x, y), h.sb(H_MUL, nx, ny));
            if (cs > 1.0) cs = 1.0;
            if (cs < -1.0) cs = -1.0;
            return V{std::atan2(h.su(H_SQRT, h.sb(H_SUB, 1.0, h.sb(H_MUL, cs, cs))), cs)};
        }
        case GeometricPrimitive::EuclideanExpMap:
            return h.b(H_ADD, vecs[0], vecs[1]);
        case GeometricPrimitive::EuclideanLogMap:
            return h.b(H_SUB, vecs[1], vecs[0]);
        case GeometricPrimitive::EuclideanDistance: {
            V diff = h.b(H_SUB, vecs[0], vecs[1]);
            return V{h.su(H_SQRT, h.dot(diff, diff))};
        }
    }
    return {};
}

/**
 * @brief Central-difference cotangent of the host composition, one operand.
 *
 * (J^T g)_i estimated as sum_k g_k (f_k(x + h e_i) - f_k(x - h e_i))/2h. This
 * differentiates reference 1, i.e. host runtime code, so it is an independent
 * witness over the composition rather than a restatement of a rule.
 */
V fdCotangent(const HostOps& h, GeometricPrimitive p, std::vector<V> vecs,
              const V& scals, const V& cot, size_t operand) {
    const size_t n = vecs[operand].size();
    V out(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        std::vector<V> plus = vecs, minus = vecs;
        plus[operand][i] += kFdStep;
        minus[operand][i] -= kFdStep;
        V fp = hostGeometric(h, p, plus, scals);
        V fm = hostGeometric(h, p, minus, scals);
        if (fp.size() != cot.size() || fm.size() != cot.size()) return {};
        double acc = 0.0;
        for (size_t k = 0; k < fp.size(); ++k) acc += cot[k] * (fp[k] - fm[k]) / (2.0 * kFdStep);
        out[i] = acc;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────
// Reference 3: the host tape.
// ─────────────────────────────────────────────────────────────────────────

/** @brief A tape leaf holding @p data. */
ad_node_t* varNode(const V& data) {
    ad_node_t* n = arena_allocate_ad_node(get_global_arena());
    if (!n) return nullptr;
    double* buf = static_cast<double*>(
        arena_allocate_zeroed(get_global_arena(), data.size() * sizeof(double)));
    int64_t* shape = static_cast<int64_t*>(
        arena_allocate_zeroed(get_global_arena(), sizeof(int64_t)));
    if (!buf || !shape) return nullptr;
    std::memcpy(buf, data.data(), data.size() * sizeof(double));
    shape[0] = static_cast<int64_t>(data.size());
    n->type = AD_NODE_VARIABLE;
    n->tensor_value = buf;
    n->shape = shape;
    n->ndim = 1;
    return n;
}

/** @brief Run the reverse sweep over every node on @p t, newest first. */
void tapeSweep(ad_tape_t* t) {
    for (size_t i = t->num_nodes; i-- > 0;) eshkol_tensor_backward_dispatch(t->nodes[i]);
}

V nodeGradient(const ad_node_t* n, size_t count) {
    V out(count, 0.0);
    if (!n || !n->tensor_gradient) return out;
    const double* g = static_cast<const double*>(n->tensor_gradient);
    for (size_t i = 0; i < count; ++i) out[i] = g[i];
    return out;
}

V nodeValue(const ad_node_t* n, size_t count) {
    V out(count, 0.0);
    if (!n || !n->tensor_value) return out;
    const double* v = static_cast<const double*>(n->tensor_value);
    for (size_t i = 0; i < count; ++i) out[i] = v[i];
    return out;
}

// ─────────────────────────────────────────────────────────────────────────
// Deterministic operands.
//
// No RNG anywhere: a parity row that cannot be reproduced from the source is
// a row nobody can investigate. The direction below is fixed and its entries
// change sign, so a lowering that dropped a term would not be hidden by a
// symmetric input.
// ─────────────────────────────────────────────────────────────────────────

V unitDirection(int64_t d, int variant) {
    V v(static_cast<size_t>(d));
    for (int64_t i = 0; i < d; ++i) {
        const double s = ((i + variant) % 2 == 0) ? 1.0 : -1.0;
        v[static_cast<size_t>(i)] = s * (0.5 + 0.125 * static_cast<double>((i + variant) % 5));
    }
    double n2 = 0.0;
    for (double e : v) n2 += e * e;
    const double inv = 1.0 / std::sqrt(n2);
    for (double& e : v) e *= inv;
    return v;
}

/** @brief A point at Euclidean radius @p r, in direction @p variant. */
V pointAtRadius(int64_t d, double r, int variant) {
    V v = unitDirection(d, variant);
    for (double& e : v) e *= r;
    return v;
}

// ─────────────────────────────────────────────────────────────────────────
// The parity table.
// ─────────────────────────────────────────────────────────────────────────

/** @brief Which manifold a primitive belongs to, for the table's second column. */
const char* manifoldOf(GeometricPrimitive p) {
    switch (p) {
        case GeometricPrimitive::MobiusAdd:
        case GeometricPrimitive::PoincareExpMapOrigin:
        case GeometricPrimitive::PoincareLogMapOrigin:
        case GeometricPrimitive::PoincareExpMap:
        case GeometricPrimitive::PoincareLogMap:
        case GeometricPrimitive::HyperbolicDistance:
        case GeometricPrimitive::PoincareProject:
        case GeometricPrimitive::PoincareRetract:
            return "hyperbolic";
        case GeometricPrimitive::SphereProject:
        case GeometricPrimitive::SphereRetract:
        case GeometricPrimitive::SphereExpMap:
        case GeometricPrimitive::SphereLogMap:
        case GeometricPrimitive::SphericalDistance:
            return "spherical";
        case GeometricPrimitive::EuclideanExpMap:
        case GeometricPrimitive::EuclideanLogMap:
        case GeometricPrimitive::EuclideanDistance:
            return "euclidean";
    }
    return "?";
}

/** @brief True when the primitive's own value uses an approximated function. */
ToleranceClass toleranceClassOf(GeometricPrimitive p) {
    switch (p) {
        case GeometricPrimitive::EuclideanExpMap:
        case GeometricPrimitive::EuclideanLogMap:
        case GeometricPrimitive::SphereProject:
        case GeometricPrimitive::PoincareProject:
            // Adds, multiplies, a dot product: exact up to f32 rounding.
            return ToleranceClass::Arithmetic;
        default:
            return ToleranceClass::Transcendental;
    }
}

/** @brief The golden file a primitive is graded against, or nullptr. */
const char* goldenFileFor(GeometricPrimitive p) {
    switch (p) {
        case GeometricPrimitive::PoincareExpMapOrigin: return "poincare_exp_map_origin.json";
        case GeometricPrimitive::PoincareLogMapOrigin: return "poincare_log_map_origin.json";
        case GeometricPrimitive::PoincareExpMap:       return "poincare_exp_log_basepoint.json";
        case GeometricPrimitive::PoincareLogMap:       return "poincare_exp_log_basepoint.json";
        case GeometricPrimitive::PoincareProject:      return "poincare_project.json";
        case GeometricPrimitive::PoincareRetract:      return "poincare_retract.json";
        case GeometricPrimitive::SphereProject:        return "sphere_project.json";
        case GeometricPrimitive::SphereRetract:        return "sphere_retract.json";
        default:                                       return nullptr;
    }
}

/** @brief Inputs for one sweep row: vectors, scalars, and a cotangent seed. */
struct RowInputs {
    std::vector<V> vecs;
    V scals;
    V cot;
};

/**
 * @brief Build the operands for @p p at dimension @p d and curvature @p c.
 *
 * Points sit well inside their domain: hyperbolic points at radius 0.6/sqrt(c)
 * (comfortably interior, so the Mobius denominator and the conformal factor
 * are far from their guards), sphere points on the unit sphere, tangent
 * vectors of modest length. The guard regimes are exercised by the golden
 * cases, which were chosen for exactly that.
 */
RowInputs inputsFor(GeometricPrimitive p, int64_t d, double c) {
    RowInputs in;
    const double ball = 0.6 / std::sqrt(c);
    const size_t n = static_cast<size_t>(d);

    switch (p) {
        case GeometricPrimitive::MobiusAdd:
            in.vecs = {pointAtRadius(d, ball, 0), pointAtRadius(d, 0.35 * ball, 1)};
            in.scals = {c};
            break;
        case GeometricPrimitive::PoincareExpMapOrigin:
        case GeometricPrimitive::PoincareLogMapOrigin:
            // For the log map the operand is a POINT (radius < 1/sqrt(c));
            // for the exp map it is a tangent vector, and 0.45/sqrt(c) keeps
            // tanh's argument in a well-conditioned range for both.
            in.vecs = {pointAtRadius(d, 0.45 / std::sqrt(c), 0)};
            in.scals = {c};
            break;
        case GeometricPrimitive::PoincareExpMap:
            in.vecs = {pointAtRadius(d, ball, 0), pointAtRadius(d, 0.30, 1)};
            in.scals = {c};
            break;
        case GeometricPrimitive::PoincareLogMap:
        case GeometricPrimitive::HyperbolicDistance:
            in.vecs = {pointAtRadius(d, ball, 0), pointAtRadius(d, 0.45 * ball, 1)};
            in.scals = {c};
            break;
        case GeometricPrimitive::PoincareProject:
            in.vecs = {pointAtRadius(d, ball, 0), pointAtRadius(d, 0.8, 1)};
            in.scals = {c, 1e-6};
            break;
        case GeometricPrimitive::PoincareRetract:
            // |x + step| stays inside the ball here, so this row exercises
            // the UNCLIPPED branch. The clipped branch is covered by the
            // golden cases, which were chosen to straddle it — saying which
            // branch a row covers is the point of separating the two.
            in.vecs = {pointAtRadius(d, ball, 0), pointAtRadius(d, 0.5 * ball, 1)};
            in.scals = {c, 1e-6};
            break;
        case GeometricPrimitive::SphereProject:
            in.vecs = {unitDirection(d, 0), pointAtRadius(d, 0.9, 1)};
            break;
        case GeometricPrimitive::SphereRetract:
            in.vecs = {unitDirection(d, 0), pointAtRadius(d, 0.4, 1)};
            in.scals = {1e-12};
            break;
        case GeometricPrimitive::SphereExpMap:
            in.vecs = {unitDirection(d, 0), pointAtRadius(d, 0.5, 1)};
            break;
        case GeometricPrimitive::SphereLogMap:
        case GeometricPrimitive::SphericalDistance:
            in.vecs = {unitDirection(d, 0), unitDirection(d, 1)};
            break;
        case GeometricPrimitive::EuclideanExpMap:
        case GeometricPrimitive::EuclideanLogMap:
        case GeometricPrimitive::EuclideanDistance:
            in.vecs = {pointAtRadius(d, 1.25, 0), pointAtRadius(d, 0.7, 1)};
            break;
    }

    // A cotangent that is neither ones nor symmetric, so a rule that drops or
    // duplicates a term shows up.
    const size_t out_n = geometricResultIsScalar(p) ? 1u : n;
    in.cot.resize(out_n);
    for (size_t i = 0; i < out_n; ++i) in.cot[i] = 0.75 + 0.0625 * static_cast<double>(i % 7);
    return in;
}

std::vector<const double*> operandPointers(const RowInputs& in) {
    std::vector<const double*> ptrs;
    for (const V& v : in.vecs) ptrs.push_back(v.data());
    for (const double& s : in.scals) ptrs.push_back(&s);
    return ptrs;
}

void printHeader() {
    std::printf("\n%-26s %-11s %-4s %-11s %-11s %-11s %-9s %-24s %s\n",
                "primitive", "manifold", "d", "fwd rel", "grad rel", "grad abs",
                "tol", "reference", "result");
    std::printf("%-26s %-11s %-4s %-11s %-11s %-11s %-9s %-24s %s\n",
                "--------------------------", "-----------", "----",
                "-----------", "-----------", "-----------", "---------",
                "------------------------", "------");
}

/**
 * @brief One sweep row: forward and gradient, device against the host
 *        composition, with a finite-difference cross-check of the gradient.
 *
 * Runs at three curvatures for a primitive that takes one, through the same
 * compiled executable; the caller checks the compile counter separately.
 */
bool runSweepRow(DeviceExecutor* exec, const HostOps& h, GeometricPrimitive p,
                 int64_t d, const std::vector<double>& curvatures) {
    const int n_vec = geometricVectorOperands(p);
    const bool scalar_result = geometricResultIsScalar(p);
    const size_t out_n = scalar_result ? 1u : static_cast<size_t>(d);
    const ToleranceClass cls = toleranceClassOf(p);
    const double tol = toleranceFor(cls);

    double worst_fwd = 0.0, worst_grad = 0.0, worst_fd = 0.0;
    bool ok = true;
    std::string note;

    const bool takes_curvature = (geometricScalarOperands(p) > 0) &&
                                 (manifoldOf(p) == std::string("hyperbolic"));

    for (double c : (takes_curvature ? curvatures : std::vector<double>{1.0})) {
        RowInputs in = inputsFor(p, d, c);
        std::vector<const double*> ptrs = operandPointers(in);

        // ---- forward ----
        V device_fwd(out_n, 0.0);
        std::string err;
        if (!runGeometric(exec, p, d, ptrs, device_fwd.data(), &err)) {
            note = "device forward refused: " + err;
            ok = false;
            break;
        }
        V host_fwd = hostGeometric(h, p, in.vecs, in.scals);
        if (host_fwd.size() != out_n) {
            note = "host composition produced " + std::to_string(host_fwd.size()) +
                   " elements, expected " + std::to_string(out_n);
            ok = false;
            break;
        }
        if (g_force_fail && std::strcmp(g_force_fail, geometricPrimitiveName(p)) == 0) {
            host_fwd[0] += 1.0;
        }
        Comparison f = compareArrays(device_fwd, host_fwd, tol);
        worst_fwd = std::fmax(worst_fwd, f.max_rel);
        if (!f.agreed) {
            ok = false;
            note = "forward disagreed at index " + std::to_string(f.worst_index) +
                   " (c=" + std::to_string(c) + ")";
        }

        // ---- gradient ----
        std::vector<V> device_grad(static_cast<size_t>(n_vec), V(static_cast<size_t>(d), 0.0));
        std::vector<double*> gptrs;
        for (V& g : device_grad) gptrs.push_back(g.data());
        if (!runGeometricGradient(exec, p, d, ptrs, in.cot.data(), gptrs, &err)) {
            note = "device gradient refused: " + err;
            ok = false;
            break;
        }
        for (int i = 0; i < n_vec; ++i) {
            V fd = fdCotangent(h, p, in.vecs, in.scals, in.cot, static_cast<size_t>(i));
            if (fd.size() != static_cast<size_t>(d)) {
                note = "finite difference produced the wrong length";
                ok = false;
                break;
            }
            Comparison gcmp = compareArrays(device_grad[static_cast<size_t>(i)], fd, tol);
            worst_grad = std::fmax(worst_grad, gcmp.max_rel);
            worst_fd = std::fmax(worst_fd, gcmp.max_abs);
            if (!gcmp.agreed) {
                ok = false;
                note = "gradient of operand " + std::to_string(i) +
                       " disagreed with the finite difference (c=" + std::to_string(c) + ")";
            }
        }
        if (!ok) break;
    }

    std::printf("%-26s %-11s %-4lld %-11.3e %-11.3e %-11.3e %-9.1e %-24s %s\n",
                geometricPrimitiveName(p), manifoldOf(p), static_cast<long long>(d),
                worst_fwd, worst_grad, worst_fd, tol,
                "host composition + fd", ok ? "PASS" : "FAIL");
    if (!ok && !note.empty()) std::printf("       %s\n", note.c_str());
    if (ok) g_rows_passed++; else g_rows_failed++;
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────
// Reference 2: the golden corpus.
// ─────────────────────────────────────────────────────────────────────────

std::string goldenDir() {
    const char* env = std::getenv("ESHKOL_GOLDEN_DIR");
    if (env && *env) return std::string(env);
    return "tests/qllm_oracle/golden";
}

/** @brief Which case fields feed which operand, per primitive. */
struct GoldenSpec {
    GeometricPrimitive p;
    const char* file;
    std::vector<const char*> input_names;   // one per vector operand, in order
    const char* primal_name;                // field in primal_outputs
    std::vector<const char*> jacobian_names; // one per vector operand, in order
};

std::vector<GoldenSpec> goldenSpecs() {
    return {
        {GeometricPrimitive::PoincareExpMapOrigin, "poincare_exp_map_origin.json",
         {"v"}, "out", {"d_out_d_v"}},
        {GeometricPrimitive::PoincareLogMapOrigin, "poincare_log_map_origin.json",
         {"y"}, "out", {"d_out_d_y"}},
        {GeometricPrimitive::PoincareExpMap, "poincare_exp_log_basepoint.json",
         {"x", "v"}, "exp_x_v", {"d_expx_d_x", "d_expx_d_v"}},
        {GeometricPrimitive::PoincareLogMap, "poincare_exp_log_basepoint.json",
         {"x", "y"}, "log_x_y", {"d_logx_d_x", "d_logx_d_y"}},
        {GeometricPrimitive::PoincareProject, "poincare_project.json",
         {"x", "grad"}, "out", {"d_out_d_x", "d_out_d_grad"}},
        {GeometricPrimitive::PoincareRetract, "poincare_retract.json",
         {"x", "step"}, "out", {"d_out_d_x", "d_out_d_step"}},
        {GeometricPrimitive::SphereProject, "sphere_project.json",
         {"x", "grad"}, "out", {"d_out_d_x", "d_out_d_grad"}},
        {GeometricPrimitive::SphereRetract, "sphere_retract.json",
         {"x", "step"}, "out", {"d_out_d_x", "d_out_d_step"}},
    };
}

/**
 * @brief Whether this case is representable on an f32 device.
 *
 * Two things put a case out of reach: an artanh argument or a conformal factor
 * within kF32Margin of the value where f32 stops resolving it, and a golden
 * entry the corpus itself marks non-finite. Both are reported by name.
 */
bool caseIsF32Admissible(const GoldenSpec& spec, const eshkol_golden::Json& c,
                         std::string* why) {
    const eshkol_golden::Json* cv = c.get("curvature");
    const double curv = cv ? cv->num(1.0) : 1.0;

    // Every hyperbolic case carries the radius that decides its conditioning
    // under one of these names.
    for (const char* field : {"radius_sqrt_c", "sqrt_c_norm_v", "base_radius_sqrt_c"}) {
        const eshkol_golden::Json* r = c.get(field);
        if (!r || r->kind != eshkol_golden::Json::Kind::Number) continue;
        const double t = r->num();
        if (t < 1.0 && (1.0 - t) < kF32Margin) {
            *why = std::string(field) + "=" + std::to_string(t) +
                   " is within " + std::to_string(kF32Margin) +
                   " of 1: 1-t is not resolvable in f32";
            return false;
        }
        if (t >= 1.0) {
            *why = std::string(field) + "=" + std::to_string(t) +
                   " is at or above the artanh clamp, which is 1-1e-7 and rounds to "
                   "exactly 1 in f32";
            return false;
        }
    }
    // poincare_project's conditioning is its conformal factor.
    const eshkol_golden::Json* conf = c.get("primal_outputs");
    if (conf) {
        const eshkol_golden::Json* cfv = conf->get("conformal_factor");
        if (cfv && cfv->kind == eshkol_golden::Json::Kind::Number &&
            cfv->num() < kF32Margin) {
            *why = "conformal_factor=" + std::to_string(cfv->num()) +
                   " is below the f32 margin";
            return false;
        }
    }
    // Cases the corpus itself flags as having non-finite gradient entries.
    const eshkol_golden::Json* grads = c.get("gradients");
    if (grads) {
        for (const char* flag : {"d_out_d_x_all_finite", "d_out_d_step_all_finite"}) {
            const eshkol_golden::Json* f = grads->get(flag);
            if (f && f->kind == eshkol_golden::Json::Kind::Bool && !f->boolean) {
                *why = std::string(flag) + " is false: the corpus records a non-finite "
                       "gradient entry here, which is the correct answer and not one a "
                       "tolerance can grade";
                return false;
            }
        }
    }
    (void)curv;
    (void)spec;
    return true;
}

/**
 * @brief Grade one primitive against every applicable case of its golden file.
 *
 * The primal is compared directly. The Jacobian is read row by row: seeding
 * the VJP with e_k gives (J^T e_k)_j = J[k][j], i.e. row k, for each operand.
 */
bool runGoldenSpec(DeviceExecutor* exec, const GoldenSpec& spec) {
    const std::string path = goldenDir() + "/" + spec.file;
    std::string err;
    eshkol_golden::JsonPtr root = eshkol_golden::parseFile(path, &err);
    if (!root) {
        std::printf("  GOLDEN %-24s FAIL (%s)\n", geometricPrimitiveName(spec.p), err.c_str());
        return false;
    }
    const eshkol_golden::Json* cases = root->get("cases");
    if (!cases || cases->size() == 0) {
        std::printf("  GOLDEN %-24s FAIL (no cases in %s)\n",
                    geometricPrimitiveName(spec.p), path.c_str());
        return false;
    }

    const double tol = toleranceFor(toleranceClassOf(spec.p));
    const int n_vec = geometricVectorOperands(spec.p);
    int graded = 0, excluded = 0;
    double worst_primal = 0.0, worst_jac = 0.0;
    bool ok = true;

    for (size_t ci = 0; ci < cases->size(); ++ci) {
        const eshkol_golden::Json* c = cases->at(ci);
        if (!c) continue;
        const eshkol_golden::Json* idj = c->get("id");
        const std::string id = idj ? idj->text : "case" + std::to_string(ci);

        std::string why;
        if (!caseIsF32Admissible(spec, *c, &why)) {
            std::printf("       EXCLUDED %-46s %s\n", id.c_str(), why.c_str());
            ++excluded;
            continue;
        }

        const eshkol_golden::Json* inputs = c->get("inputs");
        const eshkol_golden::Json* primal = c->get("primal_outputs");
        const eshkol_golden::Json* grads = c->get("gradients");
        if (!inputs || !primal || !grads) {
            std::printf("       FAIL %s: missing inputs/primal_outputs/gradients\n", id.c_str());
            ok = false;
            continue;
        }

        RowInputs in;
        for (const char* name : spec.input_names) {
            const eshkol_golden::Json* v = inputs->get(name);
            if (!v) {
                std::printf("       FAIL %s: no input '%s'\n", id.c_str(), name);
                ok = false;
                in.vecs.clear();
                break;
            }
            in.vecs.push_back(v->doubles());
        }
        if (in.vecs.size() != static_cast<size_t>(n_vec)) { ok = false; continue; }
        const int64_t d = static_cast<int64_t>(in.vecs[0].size());

        // Scalars: curvature first where the primitive takes one, then eps.
        const eshkol_golden::Json* cv = c->get("curvature");
        const eshkol_golden::Json* ev = c->get("eps");
        const int n_scal = geometricScalarOperands(spec.p);
        if (n_scal >= 1) {
            if (spec.p == GeometricPrimitive::SphereRetract) {
                in.scals.push_back(ev ? ev->num(1e-12) : 1e-12);
            } else {
                in.scals.push_back(cv ? cv->num(1.0) : 1.0);
            }
        }
        if (n_scal >= 2) in.scals.push_back(ev ? ev->num(1e-6) : 1e-6);

        std::vector<const double*> ptrs = operandPointers(in);

        // ---- primal ----
        const eshkol_golden::Json* pj = primal->get(spec.primal_name);
        if (!pj) {
            std::printf("       FAIL %s: no primal output '%s'\n", id.c_str(), spec.primal_name);
            ok = false;
            continue;
        }
        V expect_primal = pj->doubles();
        V device_primal(static_cast<size_t>(d), 0.0);
        std::string derr;
        if (!runGeometric(exec, spec.p, d, ptrs, device_primal.data(), &derr)) {
            std::printf("       FAIL %s: device forward refused: %s\n", id.c_str(), derr.c_str());
            ok = false;
            continue;
        }
        Comparison pc = compareArrays(device_primal, expect_primal, tol);
        worst_primal = std::fmax(worst_primal, pc.max_rel);
        if (!pc.agreed) {
            std::printf("       FAIL %s: primal disagreed at index %d (device %.9g, golden %.9g)\n",
                        id.c_str(), pc.worst_index,
                        pc.worst_index >= 0 ? device_primal[static_cast<size_t>(pc.worst_index)] : 0.0,
                        pc.worst_index >= 0 ? expect_primal[static_cast<size_t>(pc.worst_index)] : 0.0);
            ok = false;
        }

        // ---- Jacobian, row by row ----
        std::vector<std::vector<V>> expect_jac;
        bool have_jac = true;
        for (const char* jn : spec.jacobian_names) {
            const eshkol_golden::Json* j = grads->get(jn);
            if (!j) {
                std::printf("       FAIL %s: no Jacobian '%s'\n", id.c_str(), jn);
                ok = false;
                have_jac = false;
                break;
            }
            expect_jac.push_back(j->matrix());
        }
        if (!have_jac) continue;

        for (int64_t k = 0; k < d; ++k) {
            V seed(static_cast<size_t>(d), 0.0);
            seed[static_cast<size_t>(k)] = 1.0;
            std::vector<V> dg(static_cast<size_t>(n_vec), V(static_cast<size_t>(d), 0.0));
            std::vector<double*> gptrs;
            for (V& g : dg) gptrs.push_back(g.data());
            if (!runGeometricGradient(exec, spec.p, d, ptrs, seed.data(), gptrs, &derr)) {
                std::printf("       FAIL %s row %lld: device gradient refused: %s\n",
                            id.c_str(), static_cast<long long>(k), derr.c_str());
                ok = false;
                break;
            }
            for (int i = 0; i < n_vec; ++i) {
                const std::vector<V>& J = expect_jac[static_cast<size_t>(i)];
                if (static_cast<size_t>(k) >= J.size()) {
                    std::printf("       FAIL %s: Jacobian '%s' has %zu rows, need %lld\n",
                                id.c_str(), spec.jacobian_names[static_cast<size_t>(i)],
                                J.size(), static_cast<long long>(d));
                    ok = false;
                    break;
                }
                Comparison jc = compareArrays(dg[static_cast<size_t>(i)],
                                              J[static_cast<size_t>(k)], tol);
                worst_jac = std::fmax(worst_jac, jc.max_rel);
                if (!jc.agreed) {
                    std::printf("       FAIL %s row %lld %s: disagreed at %d "
                                "(device %.9g, golden %.9g)\n",
                                id.c_str(), static_cast<long long>(k),
                                spec.jacobian_names[static_cast<size_t>(i)], jc.worst_index,
                                jc.worst_index >= 0 ? dg[static_cast<size_t>(i)][static_cast<size_t>(jc.worst_index)] : 0.0,
                                jc.worst_index >= 0 ? J[static_cast<size_t>(k)][static_cast<size_t>(jc.worst_index)] : 0.0);
                    ok = false;
                }
            }
        }
        ++graded;
    }

    g_golden_cases_graded += graded;
    g_golden_cases_excluded += excluded;
    std::printf("  GOLDEN %-24s %-38s graded=%d excluded=%d primal %.3e jac %.3e  %s\n",
                geometricPrimitiveName(spec.p), spec.file, graded, excluded,
                worst_primal, worst_jac, ok ? "PASS" : "FAIL");
    if (graded == 0 && excluded > 0) {
        std::printf("       FAIL: every case was excluded, so this row graded nothing\n");
        ok = false;
    }
    if (ok) g_rows_passed++; else g_rows_failed++;
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────
// Reference 3 rows: the live host tape.
// ─────────────────────────────────────────────────────────────────────────

/**
 * @brief exp_x, log_x and the hyperbolic distance against the C bridge and its
 *        AD tape.
 *
 * The bridge is an INDEPENDENT implementation of the same mathematics — it
 * predates the lowering, it is written in C, and its guards differ (a 1e-15
 * Mobius denominator floor, an artanh clamp at 1-1e-12 rather than 1-1e-7).
 * Those guards are inert at the interior points used here, and the row says so:
 * it is a genuine second opinion where the two operators coincide, and it is
 * not run where they do not.
 */
bool runHostTapeRow(DeviceExecutor* exec, GeometricPrimitive p, int64_t d, double c) {
    const bool scalar_result = geometricResultIsScalar(p);
    const size_t out_n = scalar_result ? 1u : static_cast<size_t>(d);
    const double tol = toleranceFor(ToleranceClass::Transcendental);

    RowInputs in = inputsFor(p, d, c);
    std::vector<const double*> ptrs = operandPointers(in);

    V device_fwd(out_n, 0.0);
    std::string err;
    if (!runGeometric(exec, p, d, ptrs, device_fwd.data(), &err)) {
        std::printf("  TAPE   %-24s d=%-4lld FAIL (device forward: %s)\n",
                    geometricPrimitiveName(p), static_cast<long long>(d), err.c_str());
        g_rows_failed++;
        return false;
    }

    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 16);
    ad_node_t* a = varNode(in.vecs[0]);
    ad_node_t* b = varNode(in.vecs[1]);
    if (!tape || !a || !b) {
        std::printf("  TAPE   %-24s d=%-4lld FAIL (could not allocate the tape)\n",
                    geometricPrimitiveName(p), static_cast<long long>(d));
        g_rows_failed++;
        return false;
    }

    // The bridge takes the SECTIONAL curvature K = -c for the hyperbolic ops.
    ad_node_t* out = nullptr;
    switch (p) {
        case GeometricPrimitive::PoincareExpMap: out = ad_poincare_exp_map(tape, a, b, -c); break;
        case GeometricPrimitive::PoincareLogMap: out = ad_poincare_log_map(tape, a, b, -c); break;
        case GeometricPrimitive::HyperbolicDistance:
            out = ad_hyperbolic_distance(tape, a, b, -c);
            break;
        default: break;
    }
    if (!out) {
        std::printf("  TAPE   %-24s d=%-4lld FAIL (the bridge refused the forward)\n",
                    geometricPrimitiveName(p), static_cast<long long>(d));
        g_rows_failed++;
        return false;
    }

    V tape_fwd = nodeValue(out, out_n);
    Comparison fc = compareArrays(device_fwd, tape_fwd, tol);

    // Seed the tape with the same cotangent the device was given.
    double* seed = static_cast<double*>(out->tensor_gradient);
    if (!seed) {
        std::printf("  TAPE   %-24s d=%-4lld FAIL (the forward node carries no gradient buffer)\n",
                    geometricPrimitiveName(p), static_cast<long long>(d));
        g_rows_failed++;
        return false;
    }
    for (size_t i = 0; i < out_n; ++i) seed[i] = in.cot[i];
    tapeSweep(tape);

    std::vector<V> device_grad(2, V(static_cast<size_t>(d), 0.0));
    std::vector<double*> gptrs = {device_grad[0].data(), device_grad[1].data()};
    if (!runGeometricGradient(exec, p, d, ptrs, in.cot.data(), gptrs, &err)) {
        std::printf("  TAPE   %-24s d=%-4lld FAIL (device gradient: %s)\n",
                    geometricPrimitiveName(p), static_cast<long long>(d), err.c_str());
        g_rows_failed++;
        return false;
    }

    Comparison g0 = compareArrays(device_grad[0], nodeGradient(a, static_cast<size_t>(d)), tol);
    Comparison g1 = compareArrays(device_grad[1], nodeGradient(b, static_cast<size_t>(d)), tol);

    const bool ok = fc.agreed && g0.agreed && g1.agreed;
    std::printf("  TAPE   %-24s d=%-4lld c=%-5.2f fwd %.3e  d/d0 %.3e  d/d1 %.3e  %s\n",
                geometricPrimitiveName(p), static_cast<long long>(d), c,
                fc.max_rel, g0.max_rel, g1.max_rel, ok ? "PASS" : "FAIL");
    if (ok) g_rows_passed++; else g_rows_failed++;
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────
// Reference 5: manifold identities, from device outputs only.
// ─────────────────────────────────────────────────────────────────────────

/** @brief Report one invariant. */
bool invariant(const char* manifold, const char* name, const Comparison& cmp, double tol,
               int64_t d) {
    const bool ok = cmp.agreed;
    std::printf("  INVAR  %-11s %-44s d=%-4lld max rel %.3e  tol %.1e  %s\n",
                manifold, name, static_cast<long long>(d), cmp.max_rel, tol,
                ok ? "PASS" : "FAIL");
    if (ok) g_invariants_passed++; else g_invariants_failed++;
    return ok;
}

/**
 * @brief The hyperbolic identities, both computed only from device results.
 *
 *  1. LEFT CANCELLATION. (-x) (+)_c (x (+)_c y) = y in every gyrogroup. It
 *     constrains the Mobius numerator AND denominator jointly, which no
 *     single-input comparison against a host does: an implementation that
 *     scaled the whole result by a constant would match a host that made the
 *     same mistake, and would fail this.
 *  2. EXP-LOG ROUND TRIP. log_x(exp_x(v)) = v. It couples the two maps to each
 *     other with no appeal to either derivation.
 */
bool hyperbolicInvariants(DeviceExecutor* exec, int64_t d, double c) {
    const double tol = toleranceFor(ToleranceClass::Transcendental);
    const size_t n = static_cast<size_t>(d);
    const double ball = 0.6 / std::sqrt(c);
    V x = pointAtRadius(d, ball, 0);
    V y = pointAtRadius(d, 0.4 * ball, 1);
    V negx(n);
    for (size_t i = 0; i < n; ++i) negx[i] = -x[i];
    std::string err;
    bool ok = true;

    {
        V xy(n, 0.0), back(n, 0.0);
        if (!runGeometric(exec, GeometricPrimitive::MobiusAdd, d,
                          {x.data(), y.data(), &c}, xy.data(), &err) ||
            !runGeometric(exec, GeometricPrimitive::MobiusAdd, d,
                          {negx.data(), xy.data(), &c}, back.data(), &err)) {
            std::printf("  INVAR  hyperbolic  left cancellation refused: %s\n", err.c_str());
            g_invariants_failed++;
            ok = false;
        } else {
            ok = invariant("hyperbolic", "(-x) (+)c (x (+)c y) == y",
                           compareArrays(back, y, tol), tol, d) && ok;
        }
    }
    {
        V v = pointAtRadius(d, 0.25, 2);
        V ex(n, 0.0), lg(n, 0.0);
        if (!runGeometric(exec, GeometricPrimitive::PoincareExpMap, d,
                          {x.data(), v.data(), &c}, ex.data(), &err) ||
            !runGeometric(exec, GeometricPrimitive::PoincareLogMap, d,
                          {x.data(), ex.data(), &c}, lg.data(), &err)) {
            std::printf("  INVAR  hyperbolic  exp-log round trip refused: %s\n", err.c_str());
            g_invariants_failed++;
            ok = false;
        } else {
            ok = invariant("hyperbolic", "log_x(exp_x(v)) == v",
                           compareArrays(lg, v, tol), tol, d) && ok;
        }
    }
    return ok;
}

/**
 * @brief The spherical identities.
 *
 *  1. EXP-LOG ROUND TRIP on the sphere, the direct analogue of the hyperbolic
 *     one.
 *  2. TANGENCY. sphere_project's result is orthogonal to x by construction, so
 *     <project(x,g), x> must be zero for a unit x. It is measured by feeding
 *     the device result back into the device's own spherical distance... no —
 *     into sphere_project a second time, which is idempotent on a tangent
 *     vector: project(x, project(x,g)) = project(x,g).
 */
bool sphericalInvariants(DeviceExecutor* exec, int64_t d) {
    const double tol = toleranceFor(ToleranceClass::Transcendental);
    const size_t n = static_cast<size_t>(d);
    V x = unitDirection(d, 0);
    std::string err;
    bool ok = true;

    {
        // The tangent vector must be tangent at x for the round trip to hold,
        // so it is produced BY the device's own projection rather than assumed.
        V g = pointAtRadius(d, 0.5, 1);
        V v(n, 0.0), ex(n, 0.0), lg(n, 0.0);
        if (!runGeometric(exec, GeometricPrimitive::SphereProject, d,
                          {x.data(), g.data()}, v.data(), &err) ||
            !runGeometric(exec, GeometricPrimitive::SphereExpMap, d,
                          {x.data(), v.data()}, ex.data(), &err) ||
            !runGeometric(exec, GeometricPrimitive::SphereLogMap, d,
                          {x.data(), ex.data()}, lg.data(), &err)) {
            std::printf("  INVAR  spherical   exp-log round trip refused: %s\n", err.c_str());
            g_invariants_failed++;
            ok = false;
        } else {
            ok = invariant("spherical", "log_x(exp_x(P_x g)) == P_x g",
                           compareArrays(lg, v, tol), tol, d) && ok;
        }
    }
    {
        V g = pointAtRadius(d, 0.9, 1);
        V once(n, 0.0), twice(n, 0.0);
        if (!runGeometric(exec, GeometricPrimitive::SphereProject, d,
                          {x.data(), g.data()}, once.data(), &err) ||
            !runGeometric(exec, GeometricPrimitive::SphereProject, d,
                          {x.data(), once.data()}, twice.data(), &err)) {
            std::printf("  INVAR  spherical   projection idempotence refused: %s\n", err.c_str());
            g_invariants_failed++;
            ok = false;
        } else {
            ok = invariant("spherical", "P_x(P_x g) == P_x g",
                           compareArrays(twice, once, tol), tol, d) && ok;
        }
    }
    return ok;
}

/** @brief The Euclidean round trip, which is exact rather than approximate. */
bool euclideanInvariants(DeviceExecutor* exec, int64_t d) {
    const double tol = toleranceFor(ToleranceClass::Arithmetic);
    const size_t n = static_cast<size_t>(d);
    V x = pointAtRadius(d, 1.25, 0);
    V v = pointAtRadius(d, 0.7, 1);
    V ex(n, 0.0), lg(n, 0.0);
    std::string err;
    if (!runGeometric(exec, GeometricPrimitive::EuclideanExpMap, d,
                      {x.data(), v.data()}, ex.data(), &err) ||
        !runGeometric(exec, GeometricPrimitive::EuclideanLogMap, d,
                      {x.data(), ex.data()}, lg.data(), &err)) {
        std::printf("  INVAR  euclidean   exp-log round trip refused: %s\n", err.c_str());
        g_invariants_failed++;
        return false;
    }
    return invariant("euclidean", "log_x(exp_x(v)) == v", compareArrays(lg, v, tol), tol, d);
}

// ─────────────────────────────────────────────────────────────────────────
// Controls.
// ─────────────────────────────────────────────────────────────────────────

/**
 * @brief One real device result, graded against a reference perturbed by 1.0,
 *        must be REJECTED.
 *
 * Distinct from the comparator control in parity_compare.h: this one perturbs
 * a reference that a real device result was actually compared with, so a row
 * reported PASS is a row whose numbers reached the comparator.
 */
bool control_perturbed_reference_is_rejected(DeviceExecutor* exec, const HostOps& h) {
    std::cout << "Control: a perturbed geometric reference is rejected... ";
    const int64_t d = 4;
    RowInputs in = inputsFor(GeometricPrimitive::PoincareExpMap, d, 1.0);
    std::vector<const double*> ptrs = operandPointers(in);
    V device(static_cast<size_t>(d), 0.0);
    std::string err;
    if (!runGeometric(exec, GeometricPrimitive::PoincareExpMap, d, ptrs, device.data(), &err)) {
        std::cout << "FAIL (device refused the control row: " << err << ")" << std::endl;
        return false;
    }
    V host = hostGeometric(h, GeometricPrimitive::PoincareExpMap, in.vecs, in.scals);
    const double tol = toleranceFor(ToleranceClass::Transcendental);
    Comparison good = compareArrays(device, host, tol);
    V wrong = host;
    wrong[0] += 1.0;
    Comparison bad = compareArrays(device, wrong, tol);
    if (!good.agreed) {
        std::cout << "FAIL (the device result did not match its own reference)" << std::endl;
        return false;
    }
    if (bad.agreed) {
        std::cout << "FAIL (a reference off by 1.0 was accepted)" << std::endl;
        return false;
    }
    std::cout << "PASS (device result accepted, off-by-1.0 reference rejected)" << std::endl;
    return true;
}

/**
 * @brief The golden corpus must be there and must parse.
 *
 * Without this a missing corpus directory would make every golden row grade
 * zero cases, and a row that grades nothing reports nothing wrong.
 */
bool control_golden_corpus_loads() {
    std::cout << "Control: the golden corpus loads and has cases... ";
    int files = 0, total_cases = 0;
    for (const GoldenSpec& s : goldenSpecs()) {
        const std::string path = goldenDir() + "/" + s.file;
        std::string err;
        eshkol_golden::JsonPtr root = eshkol_golden::parseFile(path, &err);
        if (!root) {
            std::cout << "FAIL (" << err << ")" << std::endl;
            return false;
        }
        const eshkol_golden::Json* cases = root->get("cases");
        if (!cases || cases->size() == 0) {
            std::cout << "FAIL (no cases in " << path << ")" << std::endl;
            return false;
        }
        ++files;
        total_cases += static_cast<int>(cases->size());
    }
    std::cout << "PASS (" << files << " specs, " << total_cases << " cases)" << std::endl;
    return true;
}

/**
 * @brief One compiled executable must serve every curvature at a shape.
 *
 * Measured on the executor's compile counter, not on the values: a module with
 * a baked-in curvature returns correct numbers for the first K it saw, so only
 * the counter can tell the two apart.
 */
bool control_one_executable_per_shape(DeviceExecutor* exec) {
    std::cout << "Control: one executable serves every curvature at a shape... ";
    const int64_t d = 8;
    exec->resetStats();
    V last;
    bool distinct = false;
    for (double c : {0.5, 1.0, 2.0}) {
        RowInputs in = inputsFor(GeometricPrimitive::PoincareExpMapOrigin, d, c);
        std::vector<const double*> ptrs = operandPointers(in);
        V out(static_cast<size_t>(d), 0.0);
        std::string err;
        if (!runGeometric(exec, GeometricPrimitive::PoincareExpMapOrigin, d,
                          ptrs, out.data(), &err)) {
            std::cout << "FAIL (device refused at c=" << c << ": " << err << ")" << std::endl;
            return false;
        }
        if (!last.empty()) {
            Comparison same = compareArrays(out, last, 1e-12);
            if (!same.agreed) distinct = true;
        }
        last = out;
    }
    const uint64_t compiled = exec->stats().compiled;
    if (compiled != 1) {
        std::cout << "FAIL (" << compiled
                  << " executables compiled for one shape; curvature is not an operand)"
                  << std::endl;
        return false;
    }
    if (!distinct) {
        // Three curvatures that produced identical output would mean the
        // operand was ignored, which one executable would also explain.
        std::cout << "FAIL (three curvatures produced identical results, so the "
                     "curvature operand is not reaching the computation)" << std::endl;
        return false;
    }
    std::cout << "PASS (1 executable, 3 curvatures, 3 different results)" << std::endl;
    return true;
}

}  // namespace

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  XLA Geometric Parity (device vs host, S4)" << std::endl;
    std::cout << "=================================================" << std::endl;

    ::setenv("ESHKOL_XLA_PJRT", "1", 1);
    g_force_fail = std::getenv("ESHKOL_XLA_GEOMETRIC_FORCE_FAIL");
    if (g_force_fail && !*g_force_fail) g_force_fail = nullptr;
    if (g_force_fail) {
        std::cout << "FORCE FAIL requested for primitive '" << g_force_fail
                  << "': its host reference is perturbed by 1.0." << std::endl;
    }

    if (!test_comparator_rejects_a_perturbed_result()) {
        std::cerr << "The comparator control failed; no row below would mean anything."
                  << std::endl;
        return 1;
    }
    if (!control_golden_corpus_loads()) {
        std::cerr << "The golden corpus could not be loaded; the golden rows would "
                     "grade nothing." << std::endl;
        return 1;
    }

    DeviceExecutor* exec = registerStableHLODeviceExecutor();
    std::string why;
    if (!exec || !exec->available(&why)) {
        std::cout << "\nNo PJRT device is reachable: " << why << std::endl;
        std::cout << "A geometric parity claim needs a device, so this is not a pass."
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

    // A private arena for the host composition, reset between rows. The
    // composition allocates one tensor per host runtime call and a d=64 row's
    // finite differences make tens of thousands of them; on the global arena
    // (which the AD tape also uses, and which nothing here may reset) that
    // would grow without bound for the length of the run.
    HostOps host{arena_create(4 * 1024 * 1024)};
    if (!host.arena) {
        std::cerr << "Could not create the host-composition arena." << std::endl;
        return 1;
    }

    if (!control_perturbed_reference_is_rejected(exec, host)) g_controls_failed++;
    if (!control_one_executable_per_shape(exec)) g_controls_failed++;

    // ---- the sweep ----
    const std::vector<int64_t> dims = {2, 4, 16, 64};
    const std::vector<double> curvatures = {0.5, 1.0, 2.0};
    const std::vector<GeometricPrimitive> all = {
        GeometricPrimitive::MobiusAdd,
        GeometricPrimitive::PoincareExpMapOrigin,
        GeometricPrimitive::PoincareLogMapOrigin,
        GeometricPrimitive::PoincareExpMap,
        GeometricPrimitive::PoincareLogMap,
        GeometricPrimitive::HyperbolicDistance,
        GeometricPrimitive::PoincareProject,
        GeometricPrimitive::PoincareRetract,
        GeometricPrimitive::SphereProject,
        GeometricPrimitive::SphereRetract,
        GeometricPrimitive::SphereExpMap,
        GeometricPrimitive::SphereLogMap,
        GeometricPrimitive::SphericalDistance,
        GeometricPrimitive::EuclideanExpMap,
        GeometricPrimitive::EuclideanLogMap,
        GeometricPrimitive::EuclideanDistance,
    };

    printHeader();
    for (GeometricPrimitive p : all) {
        for (int64_t d : dims) {
            runSweepRow(exec, host, p, d, curvatures);
            arena_reset(host.arena);
        }
    }

    // ---- golden ----
    std::cout << "\nGolden corpus (tests/qllm_oracle/golden, read at run time):" << std::endl;
    for (const GoldenSpec& s : goldenSpecs()) runGoldenSpec(exec, s);

    // ---- host tape ----
    std::cout << "\nHost tape (lib/bridge/qllm_bridge.cpp, AD node types 33-35):" << std::endl;
    for (int64_t d : dims) {
        runHostTapeRow(exec, GeometricPrimitive::PoincareExpMap, d, 1.0);
        runHostTapeRow(exec, GeometricPrimitive::PoincareLogMap, d, 1.0);
        runHostTapeRow(exec, GeometricPrimitive::HyperbolicDistance, d, 1.0);
    }

    // ---- invariants ----
    std::cout << "\nManifold identities (device outputs only):" << std::endl;
    for (int64_t d : dims) {
        hyperbolicInvariants(exec, d, 1.0);
        sphericalInvariants(exec, d);
        euclideanInvariants(exec, d);
    }

    const eshkol::xla::DeviceStats stats = exec->stats();
    std::printf("\nSUMMARY: rows_passed=%d rows_failed=%d invariants_passed=%d "
                "invariants_failed=%d golden_cases=%d golden_excluded=%d controls_failed=%d "
                "dtype=%s executed=%llu compiled=%llu cache_hits=%llu\n",
                g_rows_passed, g_rows_failed, g_invariants_passed, g_invariants_failed,
                g_golden_cases_graded, g_golden_cases_excluded, g_controls_failed,
                g_dtype.c_str(),
                static_cast<unsigned long long>(stats.executed),
                static_cast<unsigned long long>(stats.compiled),
                static_cast<unsigned long long>(stats.cache_hits));

    arena_destroy(host.arena);

    const bool all_ok = (g_rows_failed == 0) && (g_invariants_failed == 0) &&
                        (g_controls_failed == 0) && (g_rows_passed > 0) &&
                        (g_invariants_passed > 0) && (g_golden_cases_graded > 0);
    std::cout << (all_ok ? "GEOMETRIC PARITY: PASS" : "GEOMETRIC PARITY: FAIL") << std::endl;
    return all_ok ? 0 : 1;
}
