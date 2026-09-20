/**
 * @file vm_complex.c
 * @brief Complex number arithmetic for the Eshkol bytecode VM.
 *
 * Implements R7RS complex number operations with Smith's formula
 * for overflow-safe division and magnitude computation.
 *
 * Native call IDs: 300-319
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <eshkol/core/complex_math.h>
#include <math.h>
#include <stdio.h>

/* ── Allocation ── */

/** @brief Allocate a complex number with the given rectangular components. */
static VmComplex* vm_complex_new(VmRegionStack* rs, double real, double imag) {
    VmComplex* z = (VmComplex*)vm_alloc(rs, sizeof(VmComplex));
    if (!z) return NULL;
    z->real = real;
    z->imag = imag;
    z->dreal = 0.0;
    z->dimag = 0.0;
    z->creal = NULL;
    z->cimag = NULL;
    return z;
}

/* ── Forward-mode tangents (ADR-0025) ──
 *
 * A complex value may carry a tangent (dreal, dimag). Every operation below
 * propagates it by its own rule: a holomorphic function f by f'(z) dz, and
 * conjugate, magnitude and angle by theirs. Where no derivative exists (the
 * origin for magnitude and angle, a branch point for sqrt and log) and a
 * tangent is present, the operation sets vm_complex_d_error and returns NULL;
 * the native dispatcher raises it. A derivative is never answered as 0. */
static const char* vm_complex_d_error = NULL;

static int vm_complex_has_tangent(const VmComplex* z) {
    return z && (z->dreal != 0.0 || z->dimag != 0.0 ||
                 (z->creal && (z->creal->tangent != 0.0 || z->creal->kind == VM_DUAL_KIND_TAYLOR)) ||
                 (z->cimag && (z->cimag->tangent != 0.0 || z->cimag->kind == VM_DUAL_KIND_TAYLOR)));
}

static eshkol_cpx vm_cpx_tangent(const VmComplex* z) {
    return eshkol_cpx_make(z->dreal, z->dimag);
}

/** Allocate a complex with an explicit tangent. */
static VmComplex* vm_complex_new_d(VmRegionStack* rs, double real, double imag,
                                   double dreal, double dimag) {
    VmComplex* z = vm_complex_new(rs, real, imag);
    if (z) {
        z->dreal = dreal; z->dimag = dimag;
        if (dreal != 0.0 || dimag != 0.0) {
            z->creal = vm_dual_make(rs, real, dreal);
            z->cimag = vm_dual_make(rs, imag, dimag);
        }
    }
    return z;
}

static int vm_complex_has_carrier(const VmComplex* z) {
    return z && (z->creal || z->cimag);
}

static VmDual* vm_cpx_component(VmRegionStack* rs, const VmComplex* z, int imag) {
    VmDual* d = imag ? z->cimag : z->creal;
    return d ? d : vm_dual_make(rs, imag ? z->imag : z->real,
                                imag ? z->dimag : z->dreal);
}

#define VM_CPX_DUAL_BINARY(name) \
static VmDual* vm_cpx_d##name(VmRegionStack* rs, const VmDual* a, const VmDual* b) { \
    return a && b ? vm_dual_##name(rs, a, b) : NULL; \
}
VM_CPX_DUAL_BINARY(add)
VM_CPX_DUAL_BINARY(sub)
VM_CPX_DUAL_BINARY(mul)
VM_CPX_DUAL_BINARY(div)
#undef VM_CPX_DUAL_BINARY
static VmDual* vm_cpx_dscale(VmRegionStack* rs, double scale, const VmDual* a) {
    return a ? vm_dual_scale(rs, scale, a) : NULL;
}

static int vm_cpx_set_carriers(VmRegionStack* rs, VmComplex* z,
                                VmDual* re, VmDual* im) {
    (void)rs;
    if (!z || !re || !im) {
        vm_complex_d_error = "complex derivative carrier allocation failed";
        return 0;
    }
    z->creal = re; z->cimag = im;
    z->real = re->kind == VM_DUAL_KIND_TAYLOR ? re->coeff[0] : re->primal;
    z->imag = im->kind == VM_DUAL_KIND_TAYLOR ? im->coeff[0] : im->primal;
    z->dreal = re ? re->tangent : 0.0;
    z->dimag = im ? im->tangent : 0.0;
    return 1;
}

enum VmComplexUnary { VM_CPX_EXP, VM_CPX_LOG, VM_CPX_SQRT, VM_CPX_SIN,
    VM_CPX_COS, VM_CPX_TAN, VM_CPX_ASIN, VM_CPX_ACOS, VM_CPX_ATAN,
    VM_CPX_SINH, VM_CPX_COSH, VM_CPX_TANH };
static VmComplex* vm_cpx_lift_unary(VmRegionStack*, const VmComplex*,
                                   eshkol_cpx, eshkol_cpx, enum VmComplexUnary);

/** Result v of a holomorphic function at z, with derivative fp = f'(z):
 *  tangent = fp * dz. */
static VmComplex* vm_cpx_out_holo(VmRegionStack* rs, eshkol_cpx v, eshkol_cpx fp,
                                  const VmComplex* z, enum VmComplexUnary kind) {
    if (vm_complex_has_carrier(z)) return vm_cpx_lift_unary(rs, z, v, fp, kind);
    eshkol_cpx d = eshkol_cpx_mul(fp, vm_cpx_tangent(z));
    return vm_complex_new_d(rs, v.re, v.im, d.re, d.im);
}

/* ── Core Operations ── */

/** @brief Native call 300: `(make-rectangular real imag)`. */
static VmComplex* vm_make_rectangular(VmRegionStack* rs, double real, double imag) {
    return vm_complex_new(rs, real, imag);
}

/** @brief Native call 301: `(make-polar mag angle)`. */
static VmComplex* vm_make_polar(VmRegionStack* rs, double mag, double angle) {
    return vm_complex_new(rs, mag * cos(angle), mag * sin(angle));
}

/** @brief Native call 304: `(magnitude z)`, via Smith's formula (scaling
 *         by the larger component) to avoid intermediate overflow. */
static double vm_complex_magnitude(const VmComplex* z) {
    double a = fabs(z->real), b = fabs(z->imag);
    if (a == 0 && b == 0) return 0.0;
    if (a >= b) {
        double r = b / a;
        return a * sqrt(1.0 + r * r);
    } else {
        double r = a / b;
        return b * sqrt(1.0 + r * r);
    }
}

/** Tangent of |z|: Re(conj(z) dz) / |z|. Sets vm_complex_d_error and returns
 *  0 when z is the origin and a tangent is present (no derivative there). */
static double vm_complex_magnitude_tangent(const VmComplex* z) {
    if (!vm_complex_has_tangent(z)) return 0.0;
    double m = vm_complex_magnitude(z);
    if (m == 0.0) {
        vm_complex_d_error = "magnitude: not differentiable at 0+0i (the magnitude of a complex number has no derivative at the origin)";
        return 0.0;
    }
    return (z->real * z->dreal + z->imag * z->dimag) / m;
}

/** Tangent of angle(z): Im(dz / z) = (re dim - im dre) / |z|^2. */
static double vm_complex_angle_tangent(const VmComplex* z) {
    if (!vm_complex_has_tangent(z)) return 0.0;
    double m2 = z->real * z->real + z->imag * z->imag;
    if (m2 == 0.0) {
        vm_complex_d_error = "angle: not differentiable at 0+0i (the angle of a complex number has no derivative at the origin)";
        return 0.0;
    }
    return (z->real * z->dimag - z->imag * z->dreal) / m2;
}

/** @brief Native call 305: `(angle z)`. */
static double vm_complex_angle(const VmComplex* z) {
    return atan2(z->imag, z->real);
}

/** @brief Native call 306: `(conjugate z)`. */
static VmComplex* vm_complex_conjugate(VmRegionStack* rs, const VmComplex* z) {
    VmComplex* out = vm_complex_new_d(rs, z->real, -z->imag, z->dreal, -z->dimag);
    if (out && vm_complex_has_carrier(z)) if (!vm_cpx_set_carriers(rs, out,
        vm_cpx_component(rs,z,0), vm_cpx_dscale(rs,-1.0,vm_cpx_component(rs,z,1)))) return NULL;
    return out;
}

/** @brief Native call 307: complex addition. */
static VmComplex* vm_complex_add(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    VmComplex* out = vm_complex_new_d(rs, a->real+b->real, a->imag+b->imag,
                                      a->dreal+b->dreal, a->dimag+b->dimag);
    if (out && (vm_complex_has_carrier(a)||vm_complex_has_carrier(b)))
        if (!vm_cpx_set_carriers(rs,out,vm_cpx_dadd(rs,vm_cpx_component(rs,a,0),vm_cpx_component(rs,b,0)),
                                 vm_cpx_dadd(rs,vm_cpx_component(rs,a,1),vm_cpx_component(rs,b,1)))) return NULL;
    return out;
}

/** @brief Native call 308: complex subtraction. */
static VmComplex* vm_complex_sub(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    VmComplex* out = vm_complex_new_d(rs,a->real-b->real,a->imag-b->imag,a->dreal-b->dreal,a->dimag-b->dimag);
    if (out && (vm_complex_has_carrier(a)||vm_complex_has_carrier(b)))
        if (!vm_cpx_set_carriers(rs,out,vm_cpx_dsub(rs,vm_cpx_component(rs,a,0),vm_cpx_component(rs,b,0)),
                                 vm_cpx_dsub(rs,vm_cpx_component(rs,a,1),vm_cpx_component(rs,b,1)))) return NULL;
    return out;
}

/** @brief Native call 309: complex multiplication. */
static VmComplex* vm_complex_mul(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    /* d(ab) = da b + a db */
    eshkol_cpx d = eshkol_cpx_add(
        eshkol_cpx_mul(vm_cpx_tangent(a), eshkol_cpx_make(b->real, b->imag)),
        eshkol_cpx_mul(eshkol_cpx_make(a->real, a->imag), vm_cpx_tangent(b)));
    VmComplex* out = vm_complex_new_d(rs,
        a->real * b->real - a->imag * b->imag,
        a->real * b->imag + a->imag * b->real, d.re, d.im);
    if (out && (vm_complex_has_carrier(a)||vm_complex_has_carrier(b))) {
        VmDual *ar=vm_cpx_component(rs,a,0),*ai=vm_cpx_component(rs,a,1),*br=vm_cpx_component(rs,b,0),*bi=vm_cpx_component(rs,b,1);
        if (!vm_cpx_set_carriers(rs,out,vm_cpx_dsub(rs,vm_cpx_dmul(rs,ar,br),vm_cpx_dmul(rs,ai,bi)),
                                 vm_cpx_dadd(rs,vm_cpx_dmul(rs,ar,bi),vm_cpx_dmul(rs,ai,br)))) return NULL;
    }
    return out;
}

/**
 * @brief Native call 310: complex division, via Smith's formula for
 *        overflow safety.
 *
 * If |c| >= |d|: r = d/c, denom = c + d*r
 *   real = (a + b*r) / denom
 *   imag = (b - a*r) / denom
 * Else: r = c/d, denom = d + c*r
 *   real = (b + a*r) / denom
 *   imag = (-a + b*r) / denom  (note sign!)
 */
static VmComplex* vm_complex_div_value(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    double c = b->real, d = b->imag;
    if (fabs(c) >= fabs(d)) {
        double r = d / c;
        double denom = c + d * r;
        return vm_complex_new(rs,
            (a->real + a->imag * r) / denom,
            (a->imag - a->real * r) / denom);
    } else {
        double r = c / d;
        double denom = d + c * r;
        return vm_complex_new(rs,
            (a->imag + a->real * r) / denom,
            (-a->real + a->imag * r) / denom);
    }
}

/** Complex division with the quotient rule: d(a/b) = (da - (a/b) db) / b. */
static VmComplex* vm_complex_div(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    VmComplex* q = vm_complex_div_value(rs, a, b);
    if (!q || (!vm_complex_has_tangent(a) && !vm_complex_has_tangent(b))) return q;
    eshkol_cpx qv = eshkol_cpx_make(q->real, q->imag);
    eshkol_cpx d = eshkol_cpx_div(
        eshkol_cpx_sub(vm_cpx_tangent(a), eshkol_cpx_mul(qv, vm_cpx_tangent(b))),
        eshkol_cpx_make(b->real, b->imag));
    q->dreal = d.re;
    q->dimag = d.im;
    if (vm_complex_has_carrier(a) || vm_complex_has_carrier(b)) {
        VmDual *ar=vm_cpx_component(rs,a,0),*ai=vm_cpx_component(rs,a,1);
        VmDual *br=vm_cpx_component(rs,b,0),*bi=vm_cpx_component(rs,b,1);
        VmDual *den=vm_cpx_dadd(rs,vm_cpx_dmul(rs,br,br),vm_cpx_dmul(rs,bi,bi));
        VmDual *nr=vm_cpx_dadd(rs,vm_cpx_dmul(rs,ar,br),vm_cpx_dmul(rs,ai,bi));
        VmDual *ni=vm_cpx_dsub(rs,vm_cpx_dmul(rs,ai,br),vm_cpx_dmul(rs,ar,bi));
        if (!vm_cpx_set_carriers(rs,q,vm_cpx_ddiv(rs,nr,den),vm_cpx_ddiv(rs,ni,den))) return NULL;
    }
    return q;
}

/*
 * ── Transcendentals: the shared core ──────────────────────────────────────
 *
 * Task #113. These used to be hand-written here and hand-written again in the
 * LLVM back end, which is how the two engines came to disagree: the native
 * polar-form sqrt returned 6.1e-17+1i for `(sqrt (make-rectangular -1.0 0.0))`
 * where this file's half-angle form returned the documented 0.0+1.0i. Both now
 * evaluate the SAME expressions from <eshkol/core/complex_math.h>, so the
 * engines agree bit-for-bit by construction rather than by review.
 */

/** @brief Convert a VM complex to the shared pair type. */
static eshkol_cpx vm_cpx_in(const VmComplex* z) {
    return eshkol_cpx_make(z->real, z->imag);
}

/** @brief Allocate a VM complex from a shared-core result. */
static VmComplex* vm_cpx_out(VmRegionStack* rs, eshkol_cpx v) {
    return vm_complex_new(rs, v.re, v.im);
}

/* Evaluate analytic functions in the existing component-carrier algebra.
 * A constant-slope linearization loses every mixed derivative. Common entire
 * functions use component formulas; the remaining functions use their exact
 * finite Taylor expansion in the nilpotent increment. */
static VmComplex* vm_cpx_lift_unary(VmRegionStack* rs, const VmComplex* z,
                                   eshkol_cpx value, eshkol_cpx first,
                                   enum VmComplexUnary kind) {
    VmDual* a = vm_cpx_component(rs, z, 0);
    VmDual* b = vm_cpx_component(rs, z, 1);
    if (!a || !b) return NULL;
    VmDual *re = NULL, *im = NULL;
    if (kind == VM_CPX_EXP) {
        VmDual* e = vm_dual_exp(rs, a);
        re = vm_cpx_dmul(rs, e, vm_dual_cos(rs, b));
        im = vm_cpx_dmul(rs, e, vm_dual_sin(rs, b));
    } else if (kind == VM_CPX_SIN || kind == VM_CPX_COS) {
        VmDual *sa = vm_dual_sin(rs, a), *ca = vm_dual_cos(rs, a);
        VmDual *sh = vm_dual_sinh(rs, b), *ch = vm_dual_cosh(rs, b);
        re = vm_cpx_dmul(rs, kind == VM_CPX_SIN ? sa : ca, ch);
        im = vm_cpx_dmul(rs, kind == VM_CPX_SIN ? ca : sa, sh);
        if (kind == VM_CPX_COS) im = vm_cpx_dscale(rs, -1.0, im);
    } else if (kind == VM_CPX_SINH || kind == VM_CPX_COSH) {
        VmDual *sh = vm_dual_sinh(rs, a), *ch = vm_dual_cosh(rs, a);
        VmDual *sb = vm_dual_sin(rs, b), *cb = vm_dual_cos(rs, b);
        re = vm_cpx_dmul(rs, kind == VM_CPX_SINH ? sh : ch, cb);
        im = vm_cpx_dmul(rs, kind == VM_CPX_SINH ? ch : sh, sb);
    }
    if (kind == VM_CPX_EXP || kind == VM_CPX_SIN || kind == VM_CPX_COS ||
        kind == VM_CPX_SINH || kind == VM_CPX_COSH) {
        VmComplex* out = vm_cpx_out(rs, value);
        return vm_cpx_set_carriers(rs, out, re, im) ? out : NULL;
    }

    uint32_t degree = 1;
    const VmDual* components[2] = {a, b};
    for (int j = 0; j < 2; ++j) if (components[j]->kind == VM_DUAL_KIND_TAYLOR) {
        uint32_t n = components[j]->order + (components[j]->tangent_coeff ? 1u : 0u)
                     + (components[j]->tangent2_coeff ? 1u : 0u);
        if (n > degree) degree = n;
    }
    eshkol_cpx* c = (eshkol_cpx*)vm_alloc(rs, ((size_t)degree + 1) * sizeof(*c));
    if (!c) return NULL;
    eshkol_cpx p = vm_cpx_in(z), one = eshkol_cpx_make(1.0, 0.0);
    c[0] = value;
    c[1] = first;
    for (uint32_t n = 2; n <= degree; ++n) {
        if (kind == VM_CPX_LOG || kind == VM_CPX_SQRT) {
            double factor = kind == VM_CPX_LOG ? -(double)(n - 1) / n : (1.5 - n) / n;
            c[n] = eshkol_cpx_scale(eshkol_cpx_div(c[n - 1], p), factor);
        } else if (kind == VM_CPX_TAN || kind == VM_CPX_TANH) {
            eshkol_cpx sum = eshkol_cpx_make(0.0, 0.0);
            for (uint32_t j = 0; j < n; ++j)
                sum = eshkol_cpx_add(sum, eshkol_cpx_mul(c[j], c[n - 1 - j]));
            c[n] = eshkol_cpx_scale(sum, (kind == VM_CPX_TAN ? 1.0 : -1.0) / n);
        } else {
            const int inverse_sine = kind == VM_CPX_ASIN || kind == VM_CPX_ACOS;
            eshkol_cpx p2 = eshkol_cpx_mul(p, p);
            eshkol_cpx q0 = inverse_sine ? eshkol_cpx_sub(one, p2) : eshkol_cpx_add(one, p2);
            eshkol_cpx q1 = eshkol_cpx_scale(p, inverse_sine ? -2.0 : 2.0);
            double f1 = inverse_sine ? (n - 1.5) * (n - 1.0) : n - 1.0;
            double f2 = inverse_sine ? -(n - 2.0) * (n - 2.0) : n - 2.0;
            eshkol_cpx sum = eshkol_cpx_add(
                eshkol_cpx_scale(eshkol_cpx_mul(q1, c[n - 1]), f1),
                eshkol_cpx_scale(c[n - 2], f2));
            double denom = inverse_sine ? (double)n * (n - 1.0) : n;
            c[n] = eshkol_cpx_scale(eshkol_cpx_div(sum, q0), -1.0 / denom);
        }
    }
    VmComplex* constant = vm_cpx_out(rs, p);
    if (!constant) return NULL;
    VmComplex* delta = vm_complex_sub(rs, z, constant);
    VmComplex* out = vm_cpx_out(rs, c[degree]);
    if (!delta || !out) return NULL;
    for (uint32_t n = degree; n > 0; --n) {
        VmComplex* term = vm_complex_mul(rs, out, delta);
        VmComplex* coefficient = vm_cpx_out(rs, c[n - 1]);
        if (!term || !coefficient) return NULL;
        out = vm_complex_add(rs, term, coefficient);
        if (!out) return NULL;
    }
    out->real = value.re; out->imag = value.im;
    if (out->creal) { out->creal->primal = value.re;
        if (out->creal->kind == VM_DUAL_KIND_TAYLOR) out->creal->coeff[0] = value.re; }
    if (out->cimag) { out->cimag->primal = value.im;
        if (out->cimag->kind == VM_DUAL_KIND_TAYLOR) out->cimag->coeff[0] = value.im; }
    return out;
}

/** @brief Native call 311: principal complex square root. */
static VmComplex* vm_complex_sqrt(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_sqrt(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    if (v.re == 0.0 && v.im == 0.0) {
        vm_complex_d_error = "sqrt: not differentiable at 0 (the square root has no derivative at the origin)";
        return NULL;
    }
    return vm_cpx_out_holo(rs, v, eshkol_cpx_div(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_scale(v, 2.0)), z, VM_CPX_SQRT);
}

/** @brief Native call 312: complex exponential. */
static VmComplex* vm_complex_exp(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_exp(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, v, z, VM_CPX_EXP);
}

/** @brief Native call 313: principal complex natural logarithm. */
static VmComplex* vm_complex_log(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_log(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    if (zi.re == 0.0 && zi.im == 0.0) {
        vm_complex_d_error = "log: not differentiable at 0 (the logarithm has no derivative at the origin)";
        return NULL;
    }
    return vm_cpx_out_holo(rs, v, eshkol_cpx_div(eshkol_cpx_make(1.0, 0.0), zi), z, VM_CPX_LOG);
}

/** @brief Native call 314: complex sine. */
static VmComplex* vm_complex_sin(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_sin(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_cos(zi), z, VM_CPX_SIN);
}

/** @brief Native call 315: complex cosine. */
static VmComplex* vm_complex_cos(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_cos(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_scale(eshkol_cpx_sin(zi), -1.0), z, VM_CPX_COS);
}

/** @brief Native call 316: complex tangent. */
static VmComplex* vm_complex_tan(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_tan(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_add(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_mul(v, v)), z, VM_CPX_TAN);
}

/** @brief Complex arcsine (principal branch). */
static VmComplex* vm_complex_asin(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_asin(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_div(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_sqrt(eshkol_cpx_sub(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_mul(zi, zi)))), z, VM_CPX_ASIN);
}

/** @brief Complex arccosine (principal branch). */
static VmComplex* vm_complex_acos(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_acos(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_div(eshkol_cpx_make(-1.0, 0.0), eshkol_cpx_sqrt(eshkol_cpx_sub(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_mul(zi, zi)))), z, VM_CPX_ACOS);
}

/** @brief Complex arctangent (principal branch). */
static VmComplex* vm_complex_atan(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_atan(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_div(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_add(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_mul(zi, zi))), z, VM_CPX_ATAN);
}

/** @brief Complex hyperbolic sine. */
static VmComplex* vm_complex_sinh(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_sinh(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_cosh(zi), z, VM_CPX_SINH);
}

/** @brief Complex hyperbolic cosine. */
static VmComplex* vm_complex_cosh(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_cosh(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_sinh(zi), z, VM_CPX_COSH);
}

/** @brief Complex hyperbolic tangent. */
static VmComplex* vm_complex_tanh(VmRegionStack* rs, const VmComplex* z) {
    eshkol_cpx zi = vm_cpx_in(z);
    eshkol_cpx v = eshkol_cpx_tanh(zi);
    if (!vm_complex_has_tangent(z)) return vm_cpx_out(rs, v);
    return vm_cpx_out_holo(rs, v, eshkol_cpx_sub(eshkol_cpx_make(1.0, 0.0), eshkol_cpx_mul(v, v)), z, VM_CPX_TANH);
}

/** @brief Native call 318: complex exponentiation, a^b = exp(b log a). */
static VmComplex* vm_complex_expt(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    eshkol_cpx av = vm_cpx_in(a), bv = vm_cpx_in(b);
    eshkol_cpx v = eshkol_cpx_pow(av, bv);
    if (!vm_complex_has_tangent(a) && !vm_complex_has_tangent(b)) return vm_cpx_out(rs, v);
    if (av.re == 0.0 && av.im == 0.0) {
        vm_complex_d_error = "expt: not differentiable at a zero base (a^b = exp(b log a) has no derivative there)";
        return NULL;
    }
    /* d(a^b) = a^b (db log a + b da / a) */
    eshkol_cpx d = eshkol_cpx_mul(v, eshkol_cpx_add(
        eshkol_cpx_mul(vm_cpx_tangent(b), eshkol_cpx_log(av)),
        eshkol_cpx_div(eshkol_cpx_mul(bv, vm_cpx_tangent(a)), av)));
    return vm_complex_new_d(rs, v.re, v.im, d.re, d.im);
}

/* ── Self-Test ── */

#ifdef VM_COMPLEX_TEST
#include <assert.h>

/** @brief Standalone self-test (built when VM_COMPLEX_TEST is defined):
 *         exercises rectangular/polar construction, magnitude/angle,
 *         arithmetic, sqrt, exp/log (including Euler's identity), and
 *         trig functions against known values. */
int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* make-rectangular */
    VmComplex* z1 = vm_make_rectangular(&rs, 3.0, 4.0);
    assert(z1 && z1->real == 3.0 && z1->imag == 4.0);

    /* magnitude (Smith's formula) */
    double mag = vm_complex_magnitude(z1);
    assert(fabs(mag - 5.0) < 1e-10);

    /* angle */
    double ang = vm_complex_angle(z1);
    assert(fabs(ang - atan2(4.0, 3.0)) < 1e-10);

    /* make-polar round-trip */
    VmComplex* z2 = vm_make_polar(&rs, 5.0, ang);
    assert(fabs(z2->real - 3.0) < 1e-10 && fabs(z2->imag - 4.0) < 1e-10);

    /* conjugate */
    VmComplex* zc = vm_complex_conjugate(&rs, z1);
    assert(zc->real == 3.0 && zc->imag == -4.0);

    /* add */
    VmComplex* z3 = vm_complex_new(&rs, 1.0, 2.0);
    VmComplex* sum = vm_complex_add(&rs, z1, z3);
    assert(sum->real == 4.0 && sum->imag == 6.0);

    /* sub */
    VmComplex* diff = vm_complex_sub(&rs, z1, z3);
    assert(diff->real == 2.0 && diff->imag == 2.0);

    /* mul: (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i */
    VmComplex* prod = vm_complex_mul(&rs, z1, z3);
    assert(fabs(prod->real - (-5.0)) < 1e-10 && fabs(prod->imag - 10.0) < 1e-10);

    /* div: (3+4i)/(1+2i) = (3+4i)(1-2i)/((1+2i)(1-2i)) = (3+8+4i-6i)/(1+4) = (11-2i)/5 */
    VmComplex* quot = vm_complex_div(&rs, z1, z3);
    assert(fabs(quot->real - 2.2) < 1e-10 && fabs(quot->imag - (-0.4)) < 1e-10);

    /* sqrt of -1 = i */
    VmComplex* neg1 = vm_complex_new(&rs, -1.0, 0.0);
    VmComplex* sq = vm_complex_sqrt(&rs, neg1);
    assert(fabs(sq->real) < 1e-10 && fabs(sq->imag - 1.0) < 1e-10);

    /* exp(iπ) + 1 = 0 (Euler's identity) */
    VmComplex* ipi = vm_complex_new(&rs, 0.0, M_PI);
    VmComplex* eipi = vm_complex_exp(&rs, ipi);
    assert(fabs(eipi->real + 1.0) < 1e-10 && fabs(eipi->imag) < 1e-10);

    /* log(e^z) = z */
    VmComplex* lz = vm_complex_log(&rs, eipi);
    assert(fabs(lz->real) < 1e-10); /* log(-1) = iπ */

    /* sin(0) = 0 */
    VmComplex* zero = vm_complex_new(&rs, 0.0, 0.0);
    VmComplex* s0 = vm_complex_sin(&rs, zero);
    assert(fabs(s0->real) < 1e-10 && fabs(s0->imag) < 1e-10);

    /* cos(0) = 1 */
    VmComplex* c0 = vm_complex_cos(&rs, zero);
    assert(fabs(c0->real - 1.0) < 1e-10 && fabs(c0->imag) < 1e-10);

    vm_region_stack_destroy(&rs);
    printf("vm_complex: ALL TESTS PASSED\n");
    return 0;
}
#endif
