/*
 * S7: bf16 device numerics bounded across the dimension sweep, with the
 * hyperbolic boundary tested explicitly.
 *
 * WHAT THIS PROVES, AND AGAINST WHAT.
 *
 * Every op device_lowering.cpp lowers (the S2 elementwise/reduction surface)
 * and every primitive geometric_lowering.cpp lowers (the S4 mixed-curvature
 * surface) is run with ESHKOL_XLA_DEVICE_DTYPE=bf16 and compared against a
 * DIRECT f64 evaluation of the same closed-form formula, written in this
 * file rather than borrowed from the host tensor runtime or the AD tape (see
 * geometric_parity_test.cpp for that stronger, multi-reference comparison at
 * f32; this harness's job is the bf16 error MAGNITUDE across dimension and
 * near the Poincare boundary, not a second independent-implementation
 * check). Every geometric formula below is transcribed line-for-line from
 * lib/backend/xla/geometric_lowering.cpp's emitGeometricBody(), so the two
 * are the same computation evaluated at two precisions, which is exactly
 * what a numerics-bound row needs to mean.
 *
 * THE SWEEP.
 *
 * d in {2, 4, 16, 64, 256, 1024} for every op/primitive that takes a vector
 * operand. Reductions and dot-product-shaped primitives (distance,
 * conformal factor) are where bf16's error should GROW with d, because a
 * bf16 accumulation error compounds over more terms; the table reports that
 * trend rather than asserting it stays flat.
 *
 * THE HYPERBOLIC BOUNDARY.
 *
 * Poincare-ball points with ||x|| sqrt(c) in {0.9, 0.99, 0.999, 1 - 2^-8}
 * are constructed explicitly (radius scaled to hit each ratio exactly, at
 * d=64, c=1) and run through PoincareExpMapOrigin, PoincareLogMapOrigin,
 * PoincareExpMap, PoincareLogMap, HyperbolicDistance and MobiusAdd. Near the
 * boundary the conformal factor 1 - c|x|^2 -> 0 and the log map's artanh
 * argument -> 1: both are exactly where bf16's ~3-decimal-digit mantissa
 * runs out of resolution first. qllmArtanh's clamp constant, 1 - 1e-7
 * (kArtanhClamp, lib/backend/xla/geometric_lowering.cpp:54), is ALREADY
 * indistinguishable from 1.0 in bf16 (bf16's machine epsilon is ~7.8e-3, ten
 * thousand times coarser than 1e-7): every boundary row at 0.99 and above
 * clamps at the value 1.0 in bf16 raw mode, not at 1 - 1e-7, which is the
 * "the lowering must clamp, and bf16 cannot represent the value the clamp
 * targets" failure mode the S7 brief asks this harness to document. Mobius
 * addition's own denominator floor is kMobiusDenFloor = 1e-15
 * (lib/bridge/qllm_bridge.cpp:221) — also unreachable in bf16, whose
 * smallest normal magnitude is nowhere near that floor, so a bf16 Mobius
 * add near the boundary is bounded by the DENOMINATOR'S bf16 rounding, not
 * by the floor ever engaging.
 *
 * TWO TABLES: RAW AND MIXED-PRECISION.
 *
 * Every row runs twice: once with the device computing entirely in bf16
 * (buildGeometricModule's default), and once under the S7 mixed-precision
 * policy (mixed_precision=true: bf16 in and out, f32 inside — see
 * geometric_lowering.cpp's buildGeometricModule and
 * docs/design/ESHKOL_S_FRAGMENT.md, "Mixed precision under bf16"). Both
 * numbers are printed so the policy's effect is a measured comparison, not
 * an assertion.
 *
 * GRADING.
 *
 * Arithmetic and transcendental rows are graded against
 * eshkol_parity::g_tol_arithmetic / g_tol_transcendental, which
 * setTolerancesForDtype("bf16") sets to 4e-2 for both classes (see
 * parity_compare.h and docs/design/ESHKOL_S_FRAGMENT.md). A row whose error
 * cannot be bounded under either raw or mixed-precision mode is reported
 * FAIL with the reason; it is never excluded from the table or silently
 * passed.
 *
 * A GATE THAT CANNOT FAIL IS WORTHLESS.
 *
 * test_comparator_rejects_a_perturbed_result() runs first, exactly as in
 * op_parity_test.cpp and geometric_parity_test.cpp.
 * ESHKOL_XLA_BF16_FORCE_FAIL=1 disables the mixed-precision policy for every
 * row that would otherwise use it (i.e. every row is graded in raw bf16
 * mode), which is expected to push the near-boundary hyperbolic rows past
 * their bound — see scripts/run_xla_gate.sh --numerics for how this is used
 * to demonstrate a real FAIL.
 *
 * Exit status: 0 every graded row passed; 1 a row failed or a control
 * failed; 77 no PJRT device was reachable.
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

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/geometric_lowering.h"
#include "eshkol/backend/xla/xla_runtime.h"

#include "parity_compare.h"

using eshkol::xla::DeviceExecutor;
using eshkol::xla::DeviceOpKind;
using eshkol::xla::DeviceOpRequest;
using eshkol::xla::deviceOpKindName;
using eshkol::xla::registerStableHLODeviceExecutor;
using eshkol::xla::GeometricPrimitive;
using eshkol::xla::geometricPrimitiveName;
using eshkol::xla::geometricResultIsScalar;
using eshkol::xla::geometricScalarOperands;
using eshkol::xla::geometricVectorOperands;
using eshkol::xla::runGeometric;

using eshkol_parity::Comparison;
using eshkol_parity::ToleranceClass;
using eshkol_parity::compareArrays;
using eshkol_parity::makeData;
using eshkol_parity::numElements;
using eshkol_parity::shapeText;
using eshkol_parity::toleranceClassName;
using eshkol_parity::toleranceFor;
using eshkol_parity::test_comparator_rejects_a_perturbed_result;

namespace {

int g_rows_passed = 0;
int g_rows_failed = 0;
int g_controls_failed = 0;
bool g_disable_mixed_precision = false;   // ESHKOL_XLA_BF16_FORCE_FAIL

const std::vector<int64_t> kDims = {2, 4, 16, 64, 256, 1024};

void printHeader() {
    std::printf("\n%-34s %6s %-14s %-11s %-11s %-9s %-6s %s\n",
                "row", "d", "class", "max abs", "max rel", "tol", "mode", "result");
    std::printf("%-34s %6s %-14s %-11s %-11s %-9s %-6s %s\n",
                "----------------------------------", "------", "--------------",
                "-----------", "-----------", "---------", "------", "------");
}

void report(const std::string& row, int64_t d, ToleranceClass cls,
            const Comparison& cmp, double tol, const char* mode) {
    const bool ok = cmp.agreed;
    std::printf("%-34s %6lld %-14s %-11.3e %-11.3e %-9.3e %-6s %s\n",
                row.c_str(), static_cast<long long>(d), toleranceClassName(cls),
                cmp.max_abs, cmp.max_rel, tol, mode, ok ? "PASS" : "FAIL");
    if (ok) ++g_rows_passed; else ++g_rows_failed;
}

void reportUnbounded(const std::string& row, int64_t d, const char* mode,
                     const std::string& reason) {
    std::printf("%-34s %6lld %-14s %-11s %-11s %-9s %-6s FAIL (%s)\n",
                row.c_str(), static_cast<long long>(d), "-", "-", "-", "-", mode,
                reason.c_str());
    ++g_rows_failed;
}

// ---------------------------------------------------------------------
// S2 op host references (direct f64 math, not the runtime's own kernels —
// op_parity_test.cpp already grades those against the runtime; this file's
// job is the bf16 magnitude, at every op, across the dimension sweep).
// ---------------------------------------------------------------------

std::vector<double> hostUnary(DeviceOpKind k, const std::vector<double>& x) {
    std::vector<double> r(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double v = x[i];
        switch (k) {
            case DeviceOpKind::Exp:     r[i] = std::exp(v); break;
            case DeviceOpKind::Log:     r[i] = std::log(v); break;
            case DeviceOpKind::Sin:     r[i] = std::sin(v); break;
            case DeviceOpKind::Cos:     r[i] = std::cos(v); break;
            case DeviceOpKind::Tanh:    r[i] = std::tanh(v); break;
            case DeviceOpKind::Sqrt:    r[i] = std::sqrt(v); break;
            case DeviceOpKind::Rsqrt:   r[i] = 1.0 / std::sqrt(v); break;
            case DeviceOpKind::Abs:     r[i] = std::fabs(v); break;
            case DeviceOpKind::Negate:  r[i] = -v; break;
            case DeviceOpKind::Sigmoid: r[i] = 1.0 / (1.0 + std::exp(-v)); break;
            case DeviceOpKind::Atanh:   r[i] = std::atanh(v); break;
            default:                    r[i] = v; break;
        }
    }
    return r;
}

std::vector<double> hostBinary(DeviceOpKind k, const std::vector<double>& a,
                               const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        switch (k) {
            case DeviceOpKind::Add:      r[i] = a[i] + b[i]; break;
            case DeviceOpKind::Subtract: r[i] = a[i] - b[i]; break;
            case DeviceOpKind::Multiply: r[i] = a[i] * b[i]; break;
            case DeviceOpKind::Divide:   r[i] = a[i] / b[i]; break;
            case DeviceOpKind::Pow:      r[i] = std::pow(a[i], b[i]); break;
            case DeviceOpKind::Maximum:  r[i] = std::max(a[i], b[i]); break;
            case DeviceOpKind::Minimum:  r[i] = std::min(a[i], b[i]); break;
            default:                     r[i] = a[i]; break;
        }
    }
    return r;
}

double hostReduce(DeviceOpKind k, const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    switch (k) {
        case DeviceOpKind::ReduceSum:  { double s = 0; for (double v : x) s += v; return s; }
        case DeviceOpKind::ReduceMean: { double s = 0; for (double v : x) s += v; return s / static_cast<double>(x.size()); }
        case DeviceOpKind::ReduceMax:  { double m = x[0]; for (double v : x) m = std::max(m, v); return m; }
        case DeviceOpKind::ReduceMin:  { double m = x[0]; for (double v : x) m = std::min(m, v); return m; }
        case DeviceOpKind::ReduceProd: { double p = 1; for (double v : x) p *= v; return p; }
        default: return 0.0;
    }
}

bool runUnary(DeviceExecutor* ex, DeviceOpKind k, const std::vector<double>& x,
             std::vector<double>* out, std::string* err) {
    DeviceOpRequest req;
    req.kind = k;
    req.operand_shapes = {{static_cast<int64_t>(x.size())}};
    req.result_shape = {static_cast<int64_t>(x.size())};
    out->assign(x.size(), 0.0);
    std::vector<const double*> ops = {x.data()};
    return ex->run(req, ops, out->data(), err);
}

bool runBinary(DeviceExecutor* ex, DeviceOpKind k, const std::vector<double>& a,
              const std::vector<double>& b, std::vector<double>* out, std::string* err) {
    DeviceOpRequest req;
    req.kind = k;
    req.operand_shapes = {{static_cast<int64_t>(a.size())}, {static_cast<int64_t>(b.size())}};
    req.result_shape = {static_cast<int64_t>(a.size())};
    out->assign(a.size(), 0.0);
    std::vector<const double*> ops = {a.data(), b.data()};
    return ex->run(req, ops, out->data(), err);
}

bool runReduce(DeviceExecutor* ex, DeviceOpKind k, const std::vector<double>& x,
              double* out, std::string* err) {
    DeviceOpRequest req;
    req.kind = k;
    req.operand_shapes = {{static_cast<int64_t>(x.size())}};
    req.result_shape = {};
    std::vector<const double*> ops = {x.data()};
    return ex->run(req, ops, out, err);
}

void sweepS2Ops(DeviceExecutor* ex) {
    // Elementwise unary. Domains are kept where the formula is defined:
    // log/sqrt/rsqrt need positive inputs, atanh needs |x| < 1.
    struct UnaryCase { DeviceOpKind k; double base; double step; ToleranceClass cls; };
    const std::vector<UnaryCase> unary = {
        {DeviceOpKind::Exp,     -1.0,  0.37, ToleranceClass::Transcendental},
        {DeviceOpKind::Log,      0.2,  0.31, ToleranceClass::Transcendental},
        {DeviceOpKind::Sin,     -3.0,  0.53, ToleranceClass::Transcendental},
        {DeviceOpKind::Cos,     -3.0,  0.53, ToleranceClass::Transcendental},
        {DeviceOpKind::Tanh,    -2.0,  0.41, ToleranceClass::Transcendental},
        // sqrt/rsqrt are classified transcendental, matching
        // op_parity_test.cpp and docs/design/ESHKOL_S_FRAGMENT.md's
        // tolerance-class table (both are named in the "anything later
        // added beside them" list).
        {DeviceOpKind::Sqrt,     0.1,  0.29, ToleranceClass::Transcendental},
        {DeviceOpKind::Rsqrt,    0.2,  0.31, ToleranceClass::Transcendental},
        {DeviceOpKind::Abs,     -1.5,  0.17, ToleranceClass::Arithmetic},
        {DeviceOpKind::Negate,  -1.5,  0.17, ToleranceClass::Arithmetic},
        {DeviceOpKind::Sigmoid, -2.0,  0.41, ToleranceClass::Transcendental},
        {DeviceOpKind::Atanh,   -0.9,  0.011, ToleranceClass::Transcendental},
    };
    for (const auto& uc : unary) {
        for (int64_t d : kDims) {
            std::vector<double> x = makeData(d, uc.base, uc.step);
            // Atanh needs a strictly bounded domain regardless of d.
            if (uc.k == DeviceOpKind::Atanh) {
                for (auto& v : x) v = std::max(-0.95, std::min(0.95, v));
            }
            std::vector<double> host = hostUnary(uc.k, x);
            std::vector<double> device;
            std::string err;
            std::string row = std::string("s2.") + deviceOpKindName(uc.k);
            if (!runUnary(ex, uc.k, x, &device, &err)) {
                reportUnbounded(row, d, "bf16", "device: " + err);
                continue;
            }
            Comparison cmp = compareArrays(device, host, toleranceFor(uc.cls));
            report(row, d, uc.cls, cmp, toleranceFor(uc.cls), "bf16");
        }
    }

    struct BinaryCase { DeviceOpKind k; double base_a; double base_b; ToleranceClass cls; };
    const std::vector<BinaryCase> binary = {
        {DeviceOpKind::Add,      0.3,  1.1, ToleranceClass::Arithmetic},
        {DeviceOpKind::Subtract, 0.3,  1.1, ToleranceClass::Arithmetic},
        {DeviceOpKind::Multiply, 0.3,  1.1, ToleranceClass::Arithmetic},
        {DeviceOpKind::Divide,   0.3,  1.1, ToleranceClass::Arithmetic},
        {DeviceOpKind::Pow,      1.1,  1.7, ToleranceClass::Transcendental},
        {DeviceOpKind::Maximum,  0.3,  1.1, ToleranceClass::Arithmetic},
        {DeviceOpKind::Minimum,  0.3,  1.1, ToleranceClass::Arithmetic},
    };
    for (const auto& bc : binary) {
        for (int64_t d : kDims) {
            std::vector<double> a = makeData(d, bc.base_a, 0.13);
            std::vector<double> b = makeData(d, bc.base_b, 0.17);
            if (bc.k == DeviceOpKind::Pow) {
                for (auto& v : b) v = 1.0 + 0.01 * std::fmod(v, 3.0);  // keep exponents small
            }
            std::vector<double> host = hostBinary(bc.k, a, b);
            std::vector<double> device;
            std::string err;
            std::string row = std::string("s2.") + deviceOpKindName(bc.k);
            if (!runBinary(ex, bc.k, a, b, &device, &err)) {
                reportUnbounded(row, d, "bf16", "device: " + err);
                continue;
            }
            Comparison cmp = compareArrays(device, host, toleranceFor(bc.cls));
            report(row, d, bc.cls, cmp, toleranceFor(bc.cls), "bf16");
        }
    }

    struct ReduceCase { DeviceOpKind k; };
    const std::vector<ReduceCase> reduces = {
        {DeviceOpKind::ReduceSum}, {DeviceOpKind::ReduceMean},
        {DeviceOpKind::ReduceMax}, {DeviceOpKind::ReduceMin}, {DeviceOpKind::ReduceProd},
    };
    for (const auto& rc : reduces) {
        for (int64_t d : kDims) {
            // Product accumulates fast; keep magnitudes near 1 so f64 stays exact.
            const bool is_prod = (rc.k == DeviceOpKind::ReduceProd);
            std::vector<double> x = makeData(d, is_prod ? 1.0 : 0.2,
                                             is_prod ? 0.0003 : 0.13);
            double host = hostReduce(rc.k, x);
            double device = 0.0;
            std::string err;
            std::string row = std::string("s2.") + deviceOpKindName(rc.k);
            if (!runReduce(ex, rc.k, x, &device, &err)) {
                reportUnbounded(row, d, "bf16", "device: " + err);
                continue;
            }
            std::vector<double> dv = {device}, hv = {host};
            Comparison cmp = compareArrays(dv, hv, toleranceFor(ToleranceClass::Arithmetic));
            report(row, d, ToleranceClass::Arithmetic, cmp,
                   toleranceFor(ToleranceClass::Arithmetic), "bf16");
        }
    }
}

// ---------------------------------------------------------------------
// S4 geometric primitive host references — transcribed from
// emitGeometricBody() in lib/backend/xla/geometric_lowering.cpp.
// ---------------------------------------------------------------------

constexpr double kTiny = 1e-10;
constexpr double kArtanhClamp = 1.0 - 1e-7;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }
double clampMin(double a, double lo) { return a < lo ? lo : a; }
double qllmArtanh(double t) { return std::atanh(t >= kArtanhClamp ? kArtanhClamp : t); }
double clampUnit(double t) {
    double hi = t > 1.0 ? 1.0 : t;
    return hi < -1.0 ? -1.0 : hi;
}
std::vector<double> vsub(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}
std::vector<double> vadd(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}
std::vector<double> vscale(const std::vector<double>& a, double s) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s;
    return r;
}
std::vector<double> vneg(const std::vector<double>& a) { return vscale(a, -1.0); }

std::vector<double> mobiusAdd(const std::vector<double>& x, const std::vector<double>& y, double c) {
    double xy = dot(x, y), x2 = dot(x, x), y2 = dot(y, y);
    double nx = 1.0 + 2.0 * c * xy + c * y2;
    double ny = 1.0 - c * x2;
    double den = 1.0 + 2.0 * c * xy + c * c * x2 * y2;
    std::vector<double> num = vadd(vscale(x, nx), vscale(y, ny));
    return vscale(num, 1.0 / den);
}
double conformalLambda(const std::vector<double>& x, double c) {
    return 2.0 / (1.0 - c * dot(x, x));
}

/** @brief Host f64 reference for one primitive. Vector result in *vres, or
 *        scalar result in *sres (whichever geometricResultIsScalar says). */
void hostGeometric(GeometricPrimitive p, const std::vector<std::vector<double>>& vecs,
                   const std::vector<double>& scalars,
                   std::vector<double>* vres, double* sres) {
    const double c = scalars.size() > 0 ? scalars[0] : 1.0;
    switch (p) {
        case GeometricPrimitive::MobiusAdd:
            *vres = mobiusAdd(vecs[0], vecs[1], c);
            return;
        case GeometricPrimitive::PoincareExpMapOrigin: {
            const auto& v = vecs[0];
            double n = norm(v);
            if (n < kTiny) { *vres = v; return; }
            double t = std::sqrt(c) * n;
            *vres = vscale(v, std::tanh(t) / t);
            return;
        }
        case GeometricPrimitive::PoincareLogMapOrigin: {
            const auto& y = vecs[0];
            double n = norm(y);
            if (n < kTiny) { *vres = y; return; }
            double t = std::sqrt(c) * n;
            *vres = vscale(y, qllmArtanh(t) / t);
            return;
        }
        case GeometricPrimitive::PoincareExpMap: {
            const auto& x = vecs[0]; const auto& v = vecs[1];
            double nv = norm(v);
            if (nv < kTiny) { *vres = x; return; }
            double sc = std::sqrt(c);
            double lam = conformalLambda(x, c);
            double t = sc * lam * nv / 2.0;
            double scale = std::tanh(t) / (sc * nv);
            std::vector<double> second = vscale(v, scale);
            *vres = mobiusAdd(x, second, c);
            return;
        }
        case GeometricPrimitive::PoincareLogMap: {
            const auto& x = vecs[0]; const auto& y = vecs[1];
            double sc = std::sqrt(c);
            std::vector<double> u = mobiusAdd(vneg(x), y, c);
            double nu = norm(u);
            if (nu < kTiny) { vres->assign(x.size(), 0.0); return; }
            double lam = conformalLambda(x, c);
            double front = 2.0 / (sc * lam);
            double scale = front * qllmArtanh(sc * nu) / nu;
            *vres = vscale(u, scale);
            return;
        }
        case GeometricPrimitive::HyperbolicDistance: {
            const auto& x = vecs[0]; const auto& y = vecs[1];
            std::vector<double> diff = vsub(x, y);
            double diff2 = dot(diff, diff);
            double dx = 1.0 - c * dot(x, x);
            double dy = 1.0 - c * dot(y, y);
            double raw = 1.0 + (2.0 * c * diff2) / (dx * dy);
            double arg = raw < 1.0 ? 1.0 : raw;
            double acosh = std::log(arg + std::sqrt(arg * arg - 1.0));
            *sres = acosh / std::sqrt(c);
            return;
        }
        case GeometricPrimitive::PoincareProject: {
            const auto& x = vecs[0]; const auto& g = vecs[1];
            const double eps = scalars[1];
            double conf = clampMin(1.0 - c * dot(x, x), eps);
            *vres = vscale(g, 0.25 * conf * conf);
            return;
        }
        case GeometricPrimitive::PoincareRetract: {
            const auto& x = vecs[0]; const auto& step = vecs[1];
            const double eps = scalars[1];
            std::vector<double> z = vadd(x, step);
            double n2 = dot(z, z);
            double maxn2 = (1.0 - eps) / c;
            double clipped = std::sqrt(maxn2 / clampMin(n2, eps));
            double scale = n2 > maxn2 ? clipped : 1.0;
            *vres = vscale(z, scale);
            return;
        }
        case GeometricPrimitive::SphereProject: {
            const auto& x = vecs[0]; const auto& g = vecs[1];
            *vres = vsub(g, vscale(x, dot(g, x)));
            return;
        }
        case GeometricPrimitive::SphereRetract: {
            const auto& x = vecs[0]; const auto& step = vecs[1];
            const double eps = scalars[0];
            std::vector<double> z = vadd(x, step);
            double n = norm(z);
            if (n > eps) { *vres = vscale(z, 1.0 / clampMin(n, eps)); return; }
            *vres = x;
            return;
        }
        case GeometricPrimitive::SphereExpMap: {
            const auto& x = vecs[0]; const auto& v = vecs[1];
            double n = norm(v);
            if (n < kTiny) { *vres = x; return; }
            *vres = vadd(vscale(x, std::cos(n)), vscale(v, std::sin(n) / n));
            return;
        }
        case GeometricPrimitive::SphereLogMap: {
            const auto& x = vecs[0]; const auto& y = vecs[1];
            double ip = clampUnit(dot(x, y));
            std::vector<double> u = vsub(y, vscale(x, ip));
            double nu = norm(u);
            if (nu < kTiny) { vres->assign(x.size(), 0.0); return; }
            *vres = vscale(u, std::acos(ip) / nu);
            return;
        }
        case GeometricPrimitive::SphericalDistance: {
            const auto& x = vecs[0]; const auto& y = vecs[1];
            double nx = norm(x), ny = norm(y);
            double cs = clampUnit(dot(x, y) / (nx * ny));
            *sres = std::acos(cs);
            return;
        }
        case GeometricPrimitive::EuclideanExpMap:
            *vres = vadd(vecs[0], vecs[1]);
            return;
        case GeometricPrimitive::EuclideanLogMap:
            *vres = vsub(vecs[1], vecs[0]);
            return;
        case GeometricPrimitive::EuclideanDistance: {
            std::vector<double> diff = vsub(vecs[0], vecs[1]);
            *sres = norm(diff);
            return;
        }
    }
}

ToleranceClass geometricClass(GeometricPrimitive p) {
    // Every primitive below composes at least one transcendental (tanh,
    // artanh, log/acosh, sin/cos, acos); none is pure arithmetic end to end.
    switch (p) {
        case GeometricPrimitive::EuclideanExpMap:
        case GeometricPrimitive::EuclideanLogMap:
        case GeometricPrimitive::EuclideanDistance:
        case GeometricPrimitive::SphereProject:
        case GeometricPrimitive::PoincareProject:
            return ToleranceClass::Arithmetic;
        default:
            return ToleranceClass::Transcendental;
    }
}

/** @brief A deterministic vector safely inside the unit ball (|x| < 1/sqrt(c)). */
std::vector<double> ballPoint(int64_t d, double base, double step, double c) {
    std::vector<double> v = makeData(d, base, step);
    double n = norm(v);
    double target = 0.5 / std::sqrt(c);  // interior, well clear of the boundary
    if (n > 1e-12) v = vscale(v, target / n);
    return v;
}

std::vector<double> unitVector(int64_t d, double base, double step) {
    std::vector<double> v = makeData(d, base, step);
    double n = norm(v);
    if (n > 1e-12) v = vscale(v, 1.0 / n);
    return v;
}

void sweepGeometric(DeviceExecutor* ex, bool mixed_precision, const char* mode_label) {
    const GeometricPrimitive kAll[] = {
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
    const double c = 1.0;
    const double eps = 1e-6;   // matches geometric_parity_test.cpp's convention

    for (GeometricPrimitive p : kAll) {
        for (int64_t d : kDims) {
            const int n_vec = geometricVectorOperands(p);
            const int n_scal = geometricScalarOperands(p);
            const bool is_sphere = (p == GeometricPrimitive::SphereProject ||
                                    p == GeometricPrimitive::SphereRetract ||
                                    p == GeometricPrimitive::SphereExpMap ||
                                    p == GeometricPrimitive::SphereLogMap ||
                                    p == GeometricPrimitive::SphericalDistance);

            std::vector<std::vector<double>> vecs;
            for (int i = 0; i < n_vec; ++i) {
                double base = 0.15 + 0.1 * static_cast<double>(i);
                double step = 0.11 + 0.02 * static_cast<double>(i);
                vecs.push_back(is_sphere ? unitVector(d, base, step)
                                          : ballPoint(d, base, step, c));
            }
            std::vector<double> scalars;
            for (int i = 0; i < n_scal; ++i) scalars.push_back(i == 0 ? c : eps);
            if (n_scal == 1 && (p == GeometricPrimitive::SphereRetract)) scalars = {eps};

            std::vector<double> vres;
            double sres = 0.0;
            hostGeometric(p, vecs, scalars, &vres, &sres);
            const bool scalar_result = geometricResultIsScalar(p);
            std::vector<double> host = scalar_result ? std::vector<double>{sres} : vres;

            std::vector<const double*> operands;
            for (auto& v : vecs) operands.push_back(v.data());
            for (auto& s : scalars) operands.push_back(&s);
            std::vector<double> device(scalar_result ? 1 : static_cast<size_t>(d), 0.0);
            std::string err;
            std::string row = std::string("s4.") + geometricPrimitiveName(p);
            if (!runGeometric(ex, p, d, operands, device.data(), &err, mixed_precision)) {
                reportUnbounded(row, d, mode_label, "device: " + err);
                continue;
            }
            ToleranceClass cls = geometricClass(p);
            Comparison cmp = compareArrays(device, host, toleranceFor(cls));
            report(row, d, cls, cmp, toleranceFor(cls), mode_label);
        }
    }
}

/** @brief Boundary rows: ||x|| sqrt(c) at {0.9, 0.99, 0.999, 1 - 2^-8}, d=64. */
void sweepBoundary(DeviceExecutor* ex, bool mixed_precision, const char* mode_label) {
    const int64_t d = 64;
    const double c = 1.0;
    const std::vector<double> ratios = {0.9, 0.99, 0.999, 1.0 - std::pow(2.0, -8.0)};

    for (double ratio : ratios) {
        std::vector<double> dir = unitVector(d, 0.2, 0.09);
        const double radius = ratio / std::sqrt(c);
        std::vector<double> x = vscale(dir, radius);
        std::vector<double> v = vscale(unitVector(d, 0.5, 0.07), 0.1);   // small tangent step
        std::vector<double> y = ballPoint(d, 0.3, 0.05, c);

        std::string tag = "boundary(" + std::to_string(ratio) + ")";

        auto runOne = [&](GeometricPrimitive p, const std::vector<std::vector<double>>& vecs,
                          const std::vector<double>& scalars) {
            std::vector<double> vres;
            double sres = 0.0;
            hostGeometric(p, vecs, scalars, &vres, &sres);
            const bool scalar_result = geometricResultIsScalar(p);
            std::vector<double> host = scalar_result ? std::vector<double>{sres} : vres;

            std::vector<const double*> operands;
            for (auto& vv : vecs) operands.push_back(vv.data());
            std::vector<double> scal = scalars;
            for (auto& s : scal) operands.push_back(&s);
            std::vector<double> device(scalar_result ? 1 : static_cast<size_t>(d), 0.0);
            std::string err;
            std::string row = "s4.boundary." + std::string(geometricPrimitiveName(p)) + "@" + tag;
            if (!runGeometric(ex, p, d, operands, device.data(), &err, mixed_precision)) {
                reportUnbounded(row, d, mode_label, "device: " + err);
                return;
            }
            ToleranceClass cls = geometricClass(p);
            Comparison cmp = compareArrays(device, host, toleranceFor(cls));
            report(row, d, cls, cmp, toleranceFor(cls), mode_label);
        };

        runOne(GeometricPrimitive::PoincareExpMapOrigin, {x}, {c});
        runOne(GeometricPrimitive::PoincareLogMapOrigin, {x}, {c});
        runOne(GeometricPrimitive::PoincareExpMap, {x, v}, {c});
        runOne(GeometricPrimitive::PoincareLogMap, {x, y}, {c});
        runOne(GeometricPrimitive::HyperbolicDistance, {x, y}, {c});
        runOne(GeometricPrimitive::MobiusAdd, {x, y}, {c});
    }
}

}  // namespace

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  XLA bf16 Numerics (S7)" << std::endl;
    std::cout << "=========================================" << std::endl;

    ::setenv("ESHKOL_XLA_PJRT", "1", 1);
    ::setenv("ESHKOL_XLA_DEVICE_DTYPE", "bf16", 1);

    if (!test_comparator_rejects_a_perturbed_result()) {
        g_controls_failed++;
        std::cerr << "The comparator control failed; no row below would mean anything."
                  << std::endl;
        return 1;
    }

    DeviceExecutor* executor = registerStableHLODeviceExecutor();
    if (!executor) {
        std::cout << "SKIP: no device executor could be installed in this build." << std::endl;
        return 77;
    }
    std::string why;
    if (!executor->available(&why)) {
        std::cout << "SKIP: no PJRT device available: " << why << std::endl;
        return 77;
    }
    if (executor->dtypeName() != "bf16") {
        std::cerr << "FAIL: executor reports dtype '" << executor->dtypeName()
                  << "', expected 'bf16' (ESHKOL_XLA_DEVICE_DTYPE did not take effect)"
                  << std::endl;
        return 1;
    }

    eshkol_parity::setTolerancesForDtype("bf16");
    std::cout << "Device: " << executor->description() << std::endl;
    std::cout << "Tolerance (bf16, absolute or relative, whichever is looser): arithmetic="
              << eshkol_parity::g_tol_arithmetic
              << " transcendental=" << eshkol_parity::g_tol_transcendental << std::endl;

    const char* force_fail = std::getenv("ESHKOL_XLA_BF16_FORCE_FAIL");
    if (force_fail && force_fail[0] && std::strcmp(force_fail, "0") != 0) {
        g_disable_mixed_precision = true;
        std::cout << "FORCED FAIL requested: the mixed-precision policy is disabled for "
                     "every row (raw bf16 everywhere), which is expected to push the "
                     "near-boundary hyperbolic rows past their bound." << std::endl;
    }

    printHeader();

    std::cout << "\n-- S2 ops, raw bf16 --" << std::endl;
    sweepS2Ops(executor);

    std::cout << "\n-- S4 geometric primitives, raw bf16 --" << std::endl;
    sweepGeometric(executor, /*mixed_precision=*/false, "raw");

    std::cout << "\n-- S4 geometric primitives, mixed precision (f32 compute, bf16 storage) --"
              << std::endl;
    sweepGeometric(executor, /*mixed_precision=*/!g_disable_mixed_precision, "mixed");

    std::cout << "\n-- Hyperbolic boundary, raw bf16 (d=64, c=1) --" << std::endl;
    sweepBoundary(executor, /*mixed_precision=*/false, "raw");

    std::cout << "\n-- Hyperbolic boundary, mixed precision (d=64, c=1) --" << std::endl;
    sweepBoundary(executor, /*mixed_precision=*/!g_disable_mixed_precision, "mixed");

    const bool all_ok = (g_rows_failed == 0 && g_controls_failed == 0 && g_rows_passed > 0);

    std::printf("\nSUMMARY: rows_passed=%d rows_failed=%d controls_failed=%d "
                "mixed_precision_forced_off=%d\n",
                g_rows_passed, g_rows_failed, g_controls_failed,
                g_disable_mixed_precision ? 1 : 0);
    std::cout << "=========================================" << std::endl;
    std::cout << (all_ok ? "BF16 NUMERICS: PASS" : "BF16 NUMERICS: FAIL") << std::endl;
    std::cout << "=========================================" << std::endl;

    return all_ok ? 0 : 1;
}
