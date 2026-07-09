/**
 * @file vm_hyperdual.c
 * @brief Hyper-dual numbers for exact second derivatives (Hessians).
 *
 * A hyper-dual number has four components:
 *   h = f + f₁·ε₁ + f₂·ε₂ + f₁₂·ε₁ε₂
 * where ε₁² = ε₂² = 0, ε₁ε₂ ≠ 0.
 *
 * Evaluating f(x + ε₁ + ε₂) gives f(x), f'(x), f'(x), f''(x) exactly.
 * For mixed partials: seed xᵢ with ε₁, xⱼ with ε₂ → f₁₂ = ∂²f/∂xᵢ∂xⱼ.
 *
 * Reference: Fike & Alonso, "The Development of Hyper-Dual Numbers
 * for Exact Second-Derivative Calculations" (AIAA 2011).
 *
 * Native call IDs: 410-439
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <math.h>

/* ── Allocation ── */

/** @brief Allocate a hyper-dual number with the given (f, f1, f2, f12)
 *         components. */
static VmHyperDual* hd_new(VmRegionStack* rs, double f, double f1, double f2, double f12) {
    VmHyperDual* h = (VmHyperDual*)vm_alloc_object(rs, VM_SUBTYPE_HYPER_DUAL, sizeof(VmHyperDual));
    if (!h) return NULL;
    h->f = f; h->f1 = f1; h->f2 = f2; h->f12 = f12;
    return h;
}

/* ── Core Operations (IDs 410-429) ── */

/** @brief Native call 410: `(make-hyper-dual f f1 f2 f12)`. */
VmHyperDual* vm_hd_make(VmRegionStack* rs, double f, double f1, double f2, double f12) {
    return hd_new(rs, f, f1, f2, f12);
}

/* 411: hyper-dual-f */
/* 412: hyper-dual-f1 */
/* 413: hyper-dual-f2 */
/* 414: hyper-dual-f12 */

/** @brief Native call 415: hyper-dual addition (component-wise). */
VmHyperDual* vm_hd_add(VmRegionStack* rs, const VmHyperDual* a, const VmHyperDual* b) {
    return hd_new(rs, a->f + b->f, a->f1 + b->f1, a->f2 + b->f2, a->f12 + b->f12);
}

/** @brief Native call 416: hyper-dual subtraction (component-wise). */
VmHyperDual* vm_hd_sub(VmRegionStack* rs, const VmHyperDual* a, const VmHyperDual* b) {
    return hd_new(rs, a->f - b->f, a->f1 - b->f1, a->f2 - b->f2, a->f12 - b->f12);
}

/** @brief Native call 417: hyper-dual multiplication, applying the product
 *         rule to first derivatives and the second-order product rule to
 *         f12. */
VmHyperDual* vm_hd_mul(VmRegionStack* rs, const VmHyperDual* a, const VmHyperDual* b) {
    return hd_new(rs,
        a->f * b->f,
        a->f1 * b->f + a->f * b->f1,
        a->f2 * b->f + a->f * b->f2,
        a->f12 * b->f + a->f1 * b->f2 + a->f2 * b->f1 + a->f * b->f12);
}

/** @brief Native call 418: hyper-dual division, via the quotient rule
 *         extended to second-order mixed partials. */
VmHyperDual* vm_hd_div(VmRegionStack* rs, const VmHyperDual* a, const VmHyperDual* b) {
    double bf2 = b->f * b->f;
    double bf3 = bf2 * b->f;
    return hd_new(rs,
        a->f / b->f,
        (a->f1 * b->f - a->f * b->f1) / bf2,
        (a->f2 * b->f - a->f * b->f2) / bf2,
        (a->f12 * b->f - a->f * b->f12 - a->f1 * b->f2 - a->f2 * b->f1) / bf2
            + 2.0 * a->f * b->f1 * b->f2 / bf3);
}

/** @brief Native call 419: hyper-dual negation (component-wise). */
VmHyperDual* vm_hd_neg(VmRegionStack* rs, const VmHyperDual* a) {
    return hd_new(rs, -a->f, -a->f1, -a->f2, -a->f12);
}

/* ── Unary Functions (IDs 420-429) ──
 * General formula for g(a + a₁ε₁ + a₂ε₂ + a₁₂ε₁ε₂):
 *   f   = g(a)
 *   f₁  = g'(a) · a₁
 *   f₂  = g'(a) · a₂
 *   f₁₂ = g''(a) · a₁ · a₂ + g'(a) · a₁₂ */

/** @brief Native call 420: hyper-dual sin, per the unary chain-rule formula
 *         above (g=sin, g'=cos). */
VmHyperDual* vm_hd_sin(VmRegionStack* rs, const VmHyperDual* a) {
    double sa = sin(a->f), ca = cos(a->f);
    return hd_new(rs, sa, ca * a->f1, ca * a->f2,
                  -sa * a->f1 * a->f2 + ca * a->f12);
}

/** @brief Native call 421: hyper-dual cos, per the unary chain-rule formula
 *         above (g=cos, g'=-sin). */
VmHyperDual* vm_hd_cos(VmRegionStack* rs, const VmHyperDual* a) {
    double sa = sin(a->f), ca = cos(a->f);
    return hd_new(rs, ca, -sa * a->f1, -sa * a->f2,
                  -ca * a->f1 * a->f2 - sa * a->f12);
}

/** @brief Native call 422: hyper-dual exp, per the unary chain-rule formula
 *         above (g=g'=g''=exp). */
VmHyperDual* vm_hd_exp(VmRegionStack* rs, const VmHyperDual* a) {
    double ea = exp(a->f);
    return hd_new(rs, ea, ea * a->f1, ea * a->f2,
                  ea * a->f1 * a->f2 + ea * a->f12);
}

/** @brief Native call 423: hyper-dual natural log, per the unary
 *         chain-rule formula above (g=log, g'=1/a, g''=-1/a^2). */
VmHyperDual* vm_hd_log(VmRegionStack* rs, const VmHyperDual* a) {
    double inv = 1.0 / a->f;
    double inv2 = -inv * inv;
    return hd_new(rs, log(a->f), inv * a->f1, inv * a->f2,
                  inv2 * a->f1 * a->f2 + inv * a->f12);
}

/** @brief Native call 424: hyper-dual sqrt, per the unary chain-rule
 *         formula above (g=sqrt, g'=1/(2 sqrt(a)), g''=-1/(4 a sqrt(a))). */
VmHyperDual* vm_hd_sqrt(VmRegionStack* rs, const VmHyperDual* a) {
    double sa = sqrt(a->f);
    double gp = 0.5 / sa;
    double gpp = -0.25 / (a->f * sa);
    return hd_new(rs, sa, gp * a->f1, gp * a->f2,
                  gpp * a->f1 * a->f2 + gp * a->f12);
}

/** @brief Native call 425: hyper-dual power with constant exponent @p n
 *         (g=a^n), per the unary chain-rule formula above. */
VmHyperDual* vm_hd_pow(VmRegionStack* rs, const VmHyperDual* a, double n) {
    double p = pow(a->f, n);
    double gp = n * pow(a->f, n - 1.0);
    double gpp = n * (n - 1.0) * pow(a->f, n - 2.0);
    return hd_new(rs, p, gp * a->f1, gp * a->f2,
                  gpp * a->f1 * a->f2 + gp * a->f12);
}

/** @brief Native call 426: hyper-dual absolute value, scaling all
 *         derivative components by sign(a) (0 at a==0). */
VmHyperDual* vm_hd_abs(VmRegionStack* rs, const VmHyperDual* a) {
    double sign = (a->f > 0) ? 1.0 : (a->f < 0) ? -1.0 : 0.0;
    return hd_new(rs, fabs(a->f), sign * a->f1, sign * a->f2, sign * a->f12);
}

/** @brief Native call 427: hyper-dual ReLU — passes @p a through unchanged
 *         when f > 0, else all components are zero. */
VmHyperDual* vm_hd_relu(VmRegionStack* rs, const VmHyperDual* a) {
    if (a->f > 0.0) return hd_new(rs, a->f, a->f1, a->f2, a->f12);
    return hd_new(rs, 0.0, 0.0, 0.0, 0.0);
}

/** @brief Native call 428: hyper-dual sigmoid, per the unary chain-rule
 *         formula above (g'=s(1-s), g''=g' * (1-2s)). */
VmHyperDual* vm_hd_sigmoid(VmRegionStack* rs, const VmHyperDual* a) {
    double s = 1.0 / (1.0 + exp(-a->f));
    double gp = s * (1.0 - s);
    double gpp = gp * (1.0 - 2.0 * s);
    return hd_new(rs, s, gp * a->f1, gp * a->f2,
                  gpp * a->f1 * a->f2 + gp * a->f12);
}

/** @brief Native call 429: hyper-dual tanh, per the unary chain-rule
 *         formula above (g'=1-t^2, g''=-2t*g'). */
VmHyperDual* vm_hd_tanh(VmRegionStack* rs, const VmHyperDual* a) {
    double t = tanh(a->f);
    double gp = 1.0 - t * t;
    double gpp = -2.0 * t * gp;
    return hd_new(rs, t, gp * a->f1, gp * a->f2,
                  gpp * a->f1 * a->f2 + gp * a->f12);
}

/** @brief Native call 430: promote a plain scalar @p x to a hyper-dual
 *         constant (all derivative components zero). */
VmHyperDual* vm_hd_from_double(VmRegionStack* rs, double x) {
    return hd_new(rs, x, 0.0, 0.0, 0.0);
}

/** @brief Native call 431: scale hyper-dual @p a by scalar @p c
 *         (component-wise). */
VmHyperDual* vm_hd_scale(VmRegionStack* rs, double c, const VmHyperDual* a) {
    return hd_new(rs, c * a->f, c * a->f1, c * a->f2, c * a->f12);
}
