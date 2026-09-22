/**
 * @file vm_dual.c
 * @brief Forward-mode automatic differentiation via dual numbers.
 *
 * Dual numbers: a + a'*epsilon, where epsilon^2 = 0.
 * Propagates derivatives through arithmetic and transcendental
 * functions using the chain rule.
 *
 * Native call IDs: 370-389
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <math.h>
#include <stdio.h>

static uint32_t vm_taylor_epoch_counter;

uint32_t vm_dual_next_taylor_epoch(void) {
    vm_taylor_epoch_counter++;
    if (vm_taylor_epoch_counter == 0) vm_taylor_epoch_counter = 1;
    return vm_taylor_epoch_counter;
}

/* ── Allocation ── */

/** @brief Allocate a dual number with the given primal and tangent
 *         components. */
static VmDual* vm_dual_new(VmRegionStack* rs, double primal, double tangent) {
    VmDual* d = (VmDual*)vm_alloc_object(rs, VM_SUBTYPE_DUAL, sizeof(VmDual));
    if (!d) return NULL;
    d->primal = primal;
    d->tangent = tangent;
    /* Inexact by construction. Every transcendental below reaches the carrier
     * through here, which is what makes R7RS exactness contagion the DEFAULT
     * rather than something each operator has to remember to do. */
    d->eprimal = NULL;
    d->etangent = NULL;
    d->kind = VM_DUAL_KIND_SCALAR;
    d->order = 0;
    d->epoch = 0;
    d->primal_sign = 0;
    d->coeff = NULL;
    d->exact_coeff = NULL;
    d->lcoeff = NULL;
    return d;
}

/* ── Exact halves (SW-85) ─────────────────────────────────────────────────
 *
 * The VM used to answer `(derivative (lambda (x) (* x x)) 1/3)` as
 * 0.6666666666666666 where native answers 2/3, because a rational point has
 * nowhere to live in a `double` and the exactness was lost at the SEED. The
 * carrier now has two optional exact halves; these helpers are the only place
 * that decides whether a result keeps them.
 *
 * The rule is the one native's tower uses: an operation is exact only if BOTH
 * operands are exact AND the operation itself preserves exactness. + - * / and
 * integer expt qualify; every transcendental does not, and demotes by simply
 * not setting the halves. */

/** @brief True when both halves of @p d are exact. A dual with only one exact
 *         half cannot be used exactly — the missing half is a double whose
 *         value we would have to invent a rational for — so this is
 *         deliberately an AND, not an OR. */
static int dual_is_exact(const VmDual* d) {
    return d && d->eprimal && d->etangent;
}

/** @brief Allocate a dual from exact halves, deriving the doubles so that
 *         every existing `d->primal` / `d->tangent` reader keeps seeing a
 *         correct (correctly-rounded) value. Falls back to @p fp / @p ft and
 *         an inexact carrier if either exact half is missing. */
static VmDual* vm_dual_new_exact(VmRegionStack* rs,
                                 VmRational* ep, VmRational* et,
                                 double fp, double ft) {
    if (!ep || !et) return vm_dual_new(rs, fp, ft);
    VmDual* d = (VmDual*)vm_alloc_object(rs, VM_SUBTYPE_DUAL, sizeof(VmDual));
    if (!d) return NULL;
    d->primal   = vm_rational_to_double(ep);
    d->tangent  = vm_rational_to_double(et);
    d->eprimal  = ep;
    d->etangent = et;
    d->kind = VM_DUAL_KIND_SCALAR;
    d->order = 0;
    d->epoch = 0;
    d->primal_sign = 0;
    d->coeff = NULL;
    d->exact_coeff = NULL;
    d->lcoeff = NULL;
    return d;
}

/** @brief Exact a `op` b, or NULL if either side is inexact — the caller then
 *         falls back to its double arm. */
static VmRational* rex(VmRegionStack* rs, const VmRational* a,
                       const VmRational* b, char op) {
    if (!a || !b) return NULL;
    return vm_rational_op_exact(rs, a, b, op);
}

/** @brief Seed a dual at an exact point with an exact unit tangent — the
 *         `x + 1·ε` of forward mode, with both halves exact. */
VmDual* vm_dual_make_exact_seed(VmRegionStack* rs, VmRational* point) {
    if (!point) return NULL;
    VmRational* one = vm_rational_from_int(vm_active_arena(rs), 1);
    if (!one) return NULL;
    return vm_dual_new_exact(rs, point, one,
                             vm_rational_to_double(point), 1.0);
}


/** @brief The dual's exact tangent, or NULL when it is inexact. This is the
 *         EXTRACTION side: the AD entry points push an exact Value when this
 *         answers non-NULL, and a FLOAT_VAL otherwise. */
VmRational* vm_dual_exact_tangent(const VmDual* d) {
    return dual_is_exact(d) ? d->etangent : NULL;
}

/** @brief The dual's exact primal, or NULL when it is inexact. */
VmRational* vm_dual_exact_primal(const VmDual* d) {
    return dual_is_exact(d) ? d->eprimal : NULL;
}

/* ── Arbitrary-order Taylor carrier ──────────────────────────────────────
 *
 * The bytecode VM uses the existing HEAP_DUAL envelope for both a first-order
 * dual and a univariate Taylor tower.  This keeps ordinary VM arithmetic and
 * the native-call bridge on one registered carrier while adding the K+1
 * coefficient storage that derivative-n/taylor require. */

static int dual_exact_operand(const VmDual* d) {
    if (!d) return 0;
    if (d->kind == VM_DUAL_KIND_TAYLOR) return d->exact_coeff != NULL;
    return d->eprimal != NULL;
}

/* The classic tower: the un-nested derivative-n/taylor pass (ADR-0027
 * section 3), a truncated series with double coefficients and an optional
 * exact sidecar. It meets only constants and towers of its own epoch; any
 * other carrier sends the operation down the level path. */
static VmDual* taylor_alloc(VmRegionStack* rs, uint32_t order, int exact) {
    VmDual* d = (VmDual*)vm_alloc_object(rs, VM_SUBTYPE_DUAL, sizeof(VmDual));
    if (!d) return NULL;
    memset(d, 0, sizeof *d);
    d->kind = VM_DUAL_KIND_TAYLOR;
    d->order = order;
    d->coeff = (double*)vm_alloc(rs, (size_t)(order + 1) * sizeof(double));
    d->exact_coeff = exact
        ? (VmRational**)vm_alloc(rs, (size_t)(order + 1) * sizeof(VmRational*))
        : NULL;
    if (!d->coeff || (exact && !d->exact_coeff)) return NULL;
    memset(d->coeff, 0, (size_t)(order + 1) * sizeof(double));
    if (d->exact_coeff)
        memset(d->exact_coeff, 0, (size_t)(order + 1) * sizeof(VmRational*));
    return d;
}

VmDual* vm_dual_make_taylor_seed(VmRegionStack* rs, VmRational* point,
                                 double point_value, uint32_t order, int exact,
                                 uint32_t epoch) {
    if (order > 4096u) return NULL;
    VmDual* d = taylor_alloc(rs, order, exact && point != NULL);
    if (!d) return NULL;
    d->coeff[0] = point_value;
    if (order >= 1) d->coeff[1] = 1.0;
    d->primal = point_value;
    d->tangent = order >= 1 ? 1.0 : 0.0;
    d->epoch = epoch;
    if (d->exact_coeff) {
        d->exact_coeff[0] = point;
        for (uint32_t i = 1; i <= order; i++) {
            d->exact_coeff[i] = vm_rational_from_int(vm_active_arena(rs), i == 1 ? 1 : 0);
            if (!d->exact_coeff[i]) return NULL;
        }
    }
    return d;
}

int vm_dual_is_taylor(const VmDual* d) {
    return d && d->kind == VM_DUAL_KIND_TAYLOR;
}

int vm_dual_taylor_is_exact(const VmDual* d) {
    return vm_dual_is_taylor(d) && d->exact_coeff != NULL;
}

double vm_dual_taylor_coeff(const VmDual* d, uint32_t n) {
    return (vm_dual_is_taylor(d) && n <= d->order) ? d->coeff[n] : 0.0;
}

VmRational* vm_dual_taylor_exact_coeff(const VmDual* d, uint32_t n) {
    return (vm_dual_taylor_is_exact(d) && n <= d->order) ? d->exact_coeff[n] : NULL;
}

/* Coefficient n of a tower operand; a scalar operand is a constant. */
static double taylor_coeff_as_double(const VmDual* d, uint32_t n) {
    if (!d) return 0.0;
    if (d->kind == VM_DUAL_KIND_TAYLOR)
        return n <= d->order ? d->coeff[n] : 0.0;
    return n == 0 ? d->primal : 0.0;
}

static VmRational* taylor_coeff_as_exact(const VmDual* d, uint32_t n) {
    if (!d) return NULL;
    if (d->kind == VM_DUAL_KIND_TAYLOR)
        return n <= d->order && d->exact_coeff ? d->exact_coeff[n] : NULL;
    return n == 0 ? d->eprimal : NULL;
}

static VmRational* taylor_coeff_as_exact_or_zero(VmRegionStack* rs,
                                                  const VmDual* d, uint32_t n) {
    VmRational* r = taylor_coeff_as_exact(d, n);
    if (r) return r;
    if (dual_exact_operand(d)) return vm_rational_from_int(vm_active_arena(rs), 0);
    return NULL;
}

static int taylor_exact_primal_sign(VmRegionStack* rs, const VmDual* a,
                                    const VmDual* b, char op) {
    VmRational* ar = taylor_coeff_as_exact(a, 0);
    VmRational* br = taylor_coeff_as_exact(b, 0);
    if (!ar) ar = vm_rational_from_double_exact(rs, taylor_coeff_as_double(a, 0));
    if (!br) br = vm_rational_from_double_exact(rs, taylor_coeff_as_double(b, 0));
    if (!ar || !br) return 0;
    VmRational* value = vm_rational_op_exact(rs, ar, br, op);
    return value ? vm_rational_sign(value) : 0;
}

static void taylor_exp_coeffs(double* out, const VmDual* a, uint32_t n) {
    out[0] = exp(taylor_coeff_as_double(a, 0));
    for (uint32_t k = 1; k < n; k++) {
        double sum = 0.0;
        for (uint32_t i = 1; i <= k; i++)
            sum += (double)i * taylor_coeff_as_double(a, i) * out[k - i];
        out[k] = sum / (double)k;
    }
}

static void taylor_div_coeffs(double* out, const double* numerator,
                              const double* denominator, uint32_t n) {
    for (uint32_t k = 0; k < n; k++) {
        double sum = numerator[k];
        for (uint32_t i = 1; i <= k; i++)
            sum -= denominator[i] * out[k - i];
        out[k] = sum / denominator[0];
    }
}

static void taylor_sigmoid_coeffs(VmRegionStack* rs, double* out,
                                  const VmDual* a, uint32_t n) {
    double* e = (double*)vm_alloc(rs, (size_t)n * sizeof(double));
    double* den = (double*)vm_alloc(rs, (size_t)n * sizeof(double));
    double* one = (double*)vm_alloc(rs, (size_t)n * sizeof(double));
    VmDual neg = {0};
    if (!e || !den || !one) return;
    neg.kind = VM_DUAL_KIND_TAYLOR;
    neg.order = n - 1;
    neg.coeff = (double*)vm_alloc(rs, (size_t)n * sizeof(double));
    if (!neg.coeff) return;
    for (uint32_t i = 0; i < n; i++) {
        neg.coeff[i] = -taylor_coeff_as_double(a, i);
        one[i] = 0.0;
    }
    one[0] = 1.0;
    if (taylor_coeff_as_double(a, 0) >= 0.0) {
        taylor_exp_coeffs(e, &neg, n);
        for (uint32_t i = 0; i < n; i++) den[i] = e[i];
        den[0] += 1.0;
        taylor_div_coeffs(out, one, den, n);
    } else {
        taylor_exp_coeffs(e, a, n);
        for (uint32_t i = 0; i < n; i++) den[i] = e[i];
        den[0] += 1.0;
        taylor_div_coeffs(out, e, den, n);
    }
}

static VmDual* taylor_binary(VmRegionStack* rs, const VmDual* a,
                             const VmDual* b, char op) {
    uint32_t n = a->kind == VM_DUAL_KIND_TAYLOR ? a->order : UINT32_MAX;
    if (b->kind == VM_DUAL_KIND_TAYLOR && b->order < n) n = b->order;
    uint32_t epoch = a->kind == VM_DUAL_KIND_TAYLOR ? a->epoch : b->epoch;
    int exact = dual_exact_operand(a) && dual_exact_operand(b);
    /* Test exact denominators as exact values: very small nonzero rationals
     * round to 0.0, so a double comparison would silently drop exactness. */
    if (op == '/' && exact) {
        VmRational* denominator = taylor_coeff_as_exact(b, 0);
        if (!denominator || vm_rational_is_zero(denominator)) exact = 0;
    }
    VmDual* r = taylor_alloc(rs, n, exact);
    if (!r) return NULL;
    r->epoch = epoch;
    r->primal_sign = taylor_exact_primal_sign(rs, a, b, op);
    for (uint32_t k = 0; k <= n; k++) {
        if (op == '+') r->coeff[k] = taylor_coeff_as_double(a, k) + taylor_coeff_as_double(b, k);
        else if (op == '-') r->coeff[k] = taylor_coeff_as_double(a, k) - taylor_coeff_as_double(b, k);
        else if (op == '*') {
            double sum = 0.0;
            for (uint32_t i = 0; i <= k; i++)
                sum += taylor_coeff_as_double(a, i) * taylor_coeff_as_double(b, k - i);
            r->coeff[k] = sum;
        } else {
            double sum = taylor_coeff_as_double(a, k);
            for (uint32_t i = 1; i <= k; i++)
                sum -= taylor_coeff_as_double(b, i) * r->coeff[k - i];
            r->coeff[k] = sum / taylor_coeff_as_double(b, 0);
        }
        if (r->exact_coeff) {
            VmRational* out = NULL;
            if (op == '+' || op == '-')
                out = vm_rational_op_exact(rs, taylor_coeff_as_exact_or_zero(rs, a, k),
                                           taylor_coeff_as_exact_or_zero(rs, b, k), op);
            else if (op == '*') {
                out = vm_rational_from_int(vm_active_arena(rs), 0);
                for (uint32_t i = 0; out && i <= k; i++)
                    out = vm_rational_op_exact(rs, out, vm_rational_op_exact(
                        rs, taylor_coeff_as_exact_or_zero(rs, a, i),
                        taylor_coeff_as_exact_or_zero(rs, b, k - i), '*'), '+');
            } else {
                out = taylor_coeff_as_exact_or_zero(rs, a, k);
                for (uint32_t i = 1; out && i <= k; i++)
                    out = vm_rational_op_exact(rs, out, vm_rational_op_exact(
                        rs, taylor_coeff_as_exact_or_zero(rs, b, i),
                        r->exact_coeff[k - i], '*'), '-');
                if (out) out = vm_rational_op_exact(rs, out,
                    taylor_coeff_as_exact_or_zero(rs, b, 0), '/');
            }
            if (!out) r->exact_coeff = NULL;
            else r->exact_coeff[k] = out;
        }
    }
    r->primal = r->coeff[0];
    r->tangent = n >= 1 ? r->coeff[1] : 0.0;
    return r;
}

static VmDual* taylor_unary(VmRegionStack* rs, const VmDual* a, int op) {
    uint32_t n = a->order;
    int exact = dual_exact_operand(a) && (op == 0 || op == 1 || op == 2);
    VmDual* r = taylor_alloc(rs, n, exact);
    if (!r) return NULL;
    r->epoch = a->epoch;
    double u0 = taylor_coeff_as_double(a, 0);
    if (op == 0) { /* neg */
        for (uint32_t k=0;k<=n;k++) r->coeff[k] = -taylor_coeff_as_double(a,k);
    } else if (op == 1 || op == 2) { /* abs / relu */
        /* abs has one sign for the complete series, not one abs() per
         * coefficient.  Both abs and ReLU use the zero subgradient at 0. */
        VmRational* exact0 = taylor_coeff_as_exact(a, 0);
        int sign = exact0 ? vm_rational_sign(exact0)
                          : (u0 > 0.0 ? 1 : (u0 < 0.0 ? -1 : a->primal_sign));
        double s = op == 2 ? (sign > 0 ? 1.0 : 0.0)
                           : (sign > 0 ? 1.0 : (sign < 0 ? -1.0 : 0.0));
        for (uint32_t k=0;k<=n;k++)
            r->coeff[k] = (k == 0 && op == 1) ? fabs(u0)
                                              : s * taylor_coeff_as_double(a,k);
    } else if (op == 3) { /* exp */
        taylor_exp_coeffs(r->coeff, a, n + 1);
    } else if (op == 4 || op == 5) { /* sin / cos, coupled recurrence */
        r->coeff[0] = op == 4 ? sin(u0) : cos(u0);
        double* other = (double*)vm_alloc(rs, (size_t)(n+1)*sizeof(double));
        if (!other) return NULL;
        other[0] = op == 4 ? cos(u0) : sin(u0);
        for (uint32_t k=1;k<=n;k++) {
            double sum=0.0, osum=0.0;
            for (uint32_t i=1;i<=k;i++) {
                sum += i*taylor_coeff_as_double(a,i)*other[k-i];
                osum += i*taylor_coeff_as_double(a,i)*r->coeff[k-i];
            }
            /* For sin, r'=other and other'=-r.  For cos the roles are
             * reversed: r'=-other and other'=r. */
            r->coeff[k] = (op == 4 ? sum : -sum) / k;
            other[k] = (op == 4 ? -osum : osum) / k;
        }
    } else if (op == 6) { /* log */
        r->coeff[0] = log(u0);
        double* q = (double*)vm_alloc(rs, (size_t)(n+1)*sizeof(double));
        if (!q) return NULL;
        q[0] = 0.0;
        for (uint32_t k=1;k<=n;k++) {
            double num = k*taylor_coeff_as_double(a,k);
            for (uint32_t i=1;i<k;i++) num -= taylor_coeff_as_double(a,i)*q[k-i];
            q[k] = num/taylor_coeff_as_double(a,0);
            r->coeff[k] = q[k]/k;
        }
    } else if (op == 8) { /* sigmoid, stable at both tails */
        taylor_sigmoid_coeffs(rs, r->coeff, a, n + 1);
    } else if (op == 9) { /* tanh = 2*sigmoid(2u)-1 */
        VmDual scaled = {0};
        double* sig = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
        scaled.kind = VM_DUAL_KIND_TAYLOR;
        scaled.order = n;
        scaled.coeff = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
        if (!sig || !scaled.coeff) return NULL;
        for (uint32_t k = 0; k <= n; k++)
            scaled.coeff[k] = 2.0 * taylor_coeff_as_double(a, k);
        taylor_sigmoid_coeffs(rs, sig, &scaled, n + 1);
        r->coeff[0] = 2.0 * sig[0] - 1.0;
        for (uint32_t k = 1; k <= n; k++) r->coeff[k] = 2.0 * sig[k];
    } else if (op == 10 || op == 11) { /* cosh / sinh */
        double *other=(double*)vm_alloc(rs,(size_t)(n+1)*sizeof(double));
        if (!other) return NULL;
        r->coeff[0]=op==10?cosh(u0):sinh(u0);
        other[0]=op==10?sinh(u0):cosh(u0);
        for (uint32_t k=1;k<=n;k++) {
            double sum=0.0, osum=0.0;
            for (uint32_t i=1;i<=k;i++) { sum+=i*taylor_coeff_as_double(a,i)*other[k-i]; osum+=i*taylor_coeff_as_double(a,i)*r->coeff[k-i]; }
            r->coeff[k]=sum/k; other[k]=osum/k;
        }
    } else { /* sqrt */
        r->coeff[0] = sqrt(u0);
        for (uint32_t k=1;k<=n;k++) {
            double sum=taylor_coeff_as_double(a,k);
            for (uint32_t i=1;i<k;i++) sum -= r->coeff[i]*r->coeff[k-i];
            r->coeff[k] = sum/(2.0*r->coeff[0]);
        }
    }
    if (r->exact_coeff) {
        VmRational* exact0 = taylor_coeff_as_exact(a, 0);
        int sign = exact0 ? vm_rational_sign(exact0)
                          : (u0 > 0.0 ? 1 : (u0 < 0.0 ? -1 : a->primal_sign));
        for (uint32_t k=0;k<=n;k++) {
            VmRational* in = taylor_coeff_as_exact(a,k);
            VmRational* out = NULL;
            if (op == 0) out = vm_rational_negate_exact(rs, in);
            else if (op == 2 && sign <= 0)
                out = vm_rational_from_int(vm_active_arena(rs), 0);
            else if (op == 2) out = in;
            else out = sign < 0 ? vm_rational_negate_exact(rs, in)
                     : (sign > 0 ? in : vm_rational_from_int(vm_active_arena(rs), 0));
            if (!out) { r->exact_coeff = NULL; break; }
            r->exact_coeff[k] = out;
        }
    }
    r->primal=r->coeff[0]; r->tangent=n>=1?r->coeff[1]:0.0;
    return r;
}

static VmDual* taylor_pow_integer(VmRegionStack* rs, const VmDual* a, int64_t exponent) {
    VmDual one = {0};
    one.primal = 1.0;
    one.eprimal = vm_rational_from_int(vm_active_arena(rs), 1);
    one.etangent = vm_rational_from_int(vm_active_arena(rs), 0);
    if (!one.eprimal || !one.etangent) return NULL;
    VmDual* out = taylor_alloc(rs, a->order, dual_exact_operand(a));
    if (!out) return NULL;
    out->epoch = a->epoch;
    out->coeff[0]=1.0;
    if (out->exact_coeff) {
        out->exact_coeff[0]=one.eprimal;
        for (uint32_t i = 1; i <= out->order; i++)
            out->exact_coeff[i] = one.etangent;
    }
    uint64_t k = exponent < 0 ? (uint64_t)(-(exponent + 1)) + 1u
                              : (uint64_t)exponent;
    VmDual base = *a;
    while (k > 0) {
        if (k & 1u) {
            VmDual* next = taylor_binary(rs, out, &base, '*');
            if (!next) return NULL;
            out = next;
        }
        k >>= 1;
        if (k > 0) {
            VmDual* next = taylor_binary(rs, &base, &base, '*');
            if (!next) return NULL;
            base = *next;
        }
    }
    if (exponent < 0) {
        VmDual* reciprocal = taylor_binary(rs, &one, out, '/');
        if (!reciprocal) return NULL;
        out = reciprocal;
    }
    out->primal=out->coeff[0]; out->tangent=out->order>=1?out->coeff[1]:0.0;
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Recursive level carrier (ADR-0027)
 *
 * A pass nested inside another live pass runs as a LEVEL: the truncated series
 * c[0] + c[1] t_E + ... + c[K] t_E^K in its own perturbation t_E, whose
 * coefficients are VmDuals of any kind. Two perturbations combine only when
 * their epochs are equal; every other carrier met by a level operation is a
 * constant coefficient of it, kept WHOLE, so the enclosing levels keep their
 * perturbations and no nesting shape needs a lane of its own.
 *
 * Every coefficient operation goes back through the public vm_dual_* entries,
 * so the recursion bottoms out in the scalar dual, the classic tower or the
 * exact rational arithmetic, and exactness follows the numeric tower.
 * ══════════════════════════════════════════════════════════════════════════ */

int vm_dual_is_level(const VmDual* d) {
    return d && d->kind == VM_DUAL_KIND_LEVEL;
}

uint32_t vm_dual_carrier_epoch(const VmDual* d) {
    return d && d->kind != VM_DUAL_KIND_SCALAR ? d->epoch : 0u;
}

/** @brief Primal of any carrier, recursively (a level caches its own). */
static double lv_primal(const VmDual* d) {
    if (!d) return 0.0;
    if (d->kind == VM_DUAL_KIND_TAYLOR) return d->coeff ? d->coeff[0] : d->primal;
    return d->primal;
}

/** @brief Exact primal of any carrier, recursively, or NULL when inexact. */
VmRational* vm_dual_exact_value(const VmDual* d) {
    if (!d) return NULL;
    if (d->kind == VM_DUAL_KIND_TAYLOR)
        return d->exact_coeff ? d->exact_coeff[0] : NULL;
    return d->eprimal;
}

int vm_dual_is_constant(const VmDual* d) {
    return d && d->kind == VM_DUAL_KIND_SCALAR && d->tangent == 0.0 &&
           (!d->etangent || vm_rational_is_zero(d->etangent));
}

VmDual* vm_dual_constant(VmRegionStack* rs, double value, VmRational* exact) {
    if (exact) {
        VmRational* zero = vm_rational_from_int(vm_active_arena(rs), 0);
        if (zero) return vm_dual_new_exact(rs, exact, zero,
                                           vm_rational_to_double(exact), 0.0);
    }
    return vm_dual_new(rs, value, 0.0);
}

static VmDual* lv_int(VmRegionStack* rs, int64_t n, int exact) {
    if (!exact) return vm_dual_new(rs, (double)n, 0.0);
    return vm_dual_constant(rs, (double)n,
                            vm_rational_from_int(vm_active_arena(rs), n));
}

/* An exact rational num/den constant, used for the recurrences' i/k factors. */
static VmDual* lv_ratio(VmRegionStack* rs, int64_t num, int64_t den) {
    VmRational* q = vm_rational_make(vm_active_arena(rs), num, den);
    return q ? vm_dual_constant(rs, (double)num / (double)den, q) : NULL;
}

/* Operands may live on the C stack (vm_dual_operand copies by value); a
 * coefficient that is kept must be arena-owned. */
static VmDual* lv_own(VmRegionStack* rs, const VmDual* d) {
    VmDual* copy = (VmDual*)vm_alloc_object(rs, VM_SUBTYPE_DUAL, sizeof(VmDual));
    if (copy) *copy = *d;
    return copy;
}

static uint32_t lv_order_at(const VmDual* d, uint32_t epoch) {
    if (d && d->kind != VM_DUAL_KIND_SCALAR && d->epoch == epoch) return d->order;
    return UINT32_MAX;          /* a constant of this level: every order */
}

/* Coefficient k of @p d read as a series in @p epoch. NULL is a structural
 * zero: beyond the series' order, or above c[0] of a constant. */
static VmDual* lv_coeff(VmRegionStack* rs, const VmDual* d, uint32_t epoch,
                        uint32_t k) {
    if (d->kind == VM_DUAL_KIND_LEVEL && d->epoch == epoch)
        return k <= d->order ? d->lcoeff[k] : NULL;
    if (d->kind == VM_DUAL_KIND_TAYLOR && d->epoch == epoch) {
        if (k > d->order) return NULL;
        return vm_dual_constant(rs, d->coeff[k],
                                d->exact_coeff ? d->exact_coeff[k] : NULL);
    }
    return k == 0 ? (VmDual*)d : NULL;
}

static VmDual* lv_alloc(VmRegionStack* rs, uint32_t epoch, uint32_t order) {
    if (order > 4096u) return NULL;
    VmDual* d = vm_dual_new(rs, 0.0, 0.0);
    if (!d) return NULL;
    d->kind = VM_DUAL_KIND_LEVEL;
    d->epoch = epoch;
    d->order = order;
    d->lcoeff = (VmDual**)vm_alloc(rs, ((size_t)order + 1) * sizeof(VmDual*));
    if (!d->lcoeff) return NULL;
    memset(d->lcoeff, 0, ((size_t)order + 1) * sizeof(VmDual*));
    return d;
}

/* Replace structural zeros by explicit zeros and cache the primal. A zero is
 * exact when the series' primal is exact (or, for a zero primal, when every
 * operand's primal was), so exactness follows the point as it does un-nested. */
static VmDual* lv_finish(VmRegionStack* rs, VmDual* d, int exact_hint) {
    if (!d) return NULL;
    int exact = d->lcoeff[0] ? vm_dual_exact_value(d->lcoeff[0]) != NULL
                             : exact_hint;
    for (uint32_t k = 0; k <= d->order; ++k)
        if (!d->lcoeff[k] && !(d->lcoeff[k] = lv_int(rs, 0, exact))) return NULL;
    d->primal = lv_primal(d->lcoeff[0]);
    d->eprimal = vm_dual_exact_value(d->lcoeff[0]);
    d->tangent = 0.0;
    d->etangent = NULL;
    return d;
}

static VmDual* lv_add(VmRegionStack* rs, VmDual* x, VmDual* y) {
    if (!x) return y;
    if (!y) return x;
    return vm_dual_add(rs, x, y);
}
static VmDual* lv_sub(VmRegionStack* rs, VmDual* x, VmDual* y) {
    if (!y) return x;
    if (!x) return vm_dual_neg(rs, y);
    return vm_dual_sub(rs, x, y);
}
static VmDual* lv_mul(VmRegionStack* rs, VmDual* x, VmDual* y) {
    return x && y ? vm_dual_mul(rs, x, y) : NULL;
}
static VmDual* lv_scale(VmRegionStack* rs, VmDual* x, int64_t num, int64_t den) {
    if (!x) return NULL;
    VmDual* q = lv_ratio(rs, num, den);
    return q ? vm_dual_mul(rs, q, x) : NULL;
}

static int lv_scalar_has_tangent(const VmDual* d) {
    return d && d->kind == VM_DUAL_KIND_SCALAR &&
           (d->tangent != 0.0 || (d->etangent && !vm_rational_is_zero(d->etangent)));
}

/* ADR-0027 section 2: the level path is taken when an operand is a level, or
 * when two carriers of different perturbations meet. */
static int lv_route(const VmDual* a, const VmDual* b) {
    if (vm_dual_is_level(a) || vm_dual_is_level(b)) return 1;
    if (!a || !b) return 0;
    int ta = a->kind == VM_DUAL_KIND_TAYLOR, tb = b->kind == VM_DUAL_KIND_TAYLOR;
    if (ta && tb) return a->epoch != b->epoch;
    if (ta) return lv_scalar_has_tangent(b);
    if (tb) return lv_scalar_has_tangent(a);
    return 0;
}

static VmDual* lv_binary(VmRegionStack* rs, const VmDual* a0, const VmDual* b0,
                         char op) {
    VmDual* a = lv_own(rs, a0);
    VmDual* b = lv_own(rs, b0);
    if (!a || !b) return NULL;
    uint32_t ea = vm_dual_carrier_epoch(a), eb = vm_dual_carrier_epoch(b);
    uint32_t epoch = ea > eb ? ea : eb;
    uint32_t na = lv_order_at(a, epoch), nb = lv_order_at(b, epoch);
    uint32_t n = na < nb ? na : nb;
    VmDual* r = lv_alloc(rs, epoch, n);
    if (!r) return NULL;
    int hint = vm_dual_exact_value(a) && vm_dual_exact_value(b);
    VmDual* b_0 = lv_coeff(rs, b, epoch, 0);
    for (uint32_t k = 0; k <= n; ++k) {
        VmDual* out = NULL;
        if (op == '+')
            out = lv_add(rs, lv_coeff(rs, a, epoch, k), lv_coeff(rs, b, epoch, k));
        else if (op == '-')
            out = lv_sub(rs, lv_coeff(rs, a, epoch, k), lv_coeff(rs, b, epoch, k));
        else if (op == '*') {
            for (uint32_t i = 0; i <= k; ++i)
                out = lv_add(rs, out, lv_mul(rs, lv_coeff(rs, a, epoch, i),
                                             lv_coeff(rs, b, epoch, k - i)));
        } else {
            out = lv_coeff(rs, a, epoch, k);
            for (uint32_t i = 1; i <= k; ++i)
                out = lv_sub(rs, out, lv_mul(rs, lv_coeff(rs, b, epoch, i),
                                             r->lcoeff[k - i]));
            if (!out) out = lv_int(rs, 0, hint);
            out = vm_dual_div(rs, out, b_0 ? b_0 : lv_int(rs, 0, hint));
            if (!out) return NULL;
        }
        r->lcoeff[k] = out;
    }
    return lv_finish(rs, r, hint);
}

/* The unary operation codes shared with taylor_unary. */
enum { LV_NEG = 0, LV_ABS = 1, LV_RELU = 2, LV_EXP = 3, LV_SIN = 4,
       LV_COS = 5, LV_LOG = 6, LV_SQRT = 7, LV_SIGMOID = 8, LV_TANH = 9,
       LV_COSH = 10, LV_SINH = 11 };

static VmDual* lv_unary_base(VmRegionStack* rs, VmDual* u0, int op) {
    switch (op) {
    case LV_EXP: return vm_dual_exp(rs, u0);
    case LV_SIN: return vm_dual_sin(rs, u0);
    case LV_COS: return vm_dual_cos(rs, u0);
    case LV_LOG: return vm_dual_log(rs, u0);
    case LV_SQRT: return vm_dual_sqrt(rs, u0);
    case LV_SIGMOID: return vm_dual_sigmoid(rs, u0);
    case LV_TANH: return vm_dual_tanh(rs, u0);
    case LV_COSH: return vm_dual_cosh(rs, u0);
    case LV_SINH: return vm_dual_sinh(rs, u0);
    default: return NULL;
    }
}

/* sum_{i=1..k} i u_i v_{k-i} -- the derivative-convolution every
 * first-order ODE recurrence below is built from. */
static VmDual* lv_dconv(VmRegionStack* rs, VmDual** u, VmDual** v, uint32_t k) {
    VmDual* s = NULL;
    for (uint32_t i = 1; i <= k; ++i) {
        VmDual* t = lv_mul(rs, u[i], v[k - i]);
        if (t && i > 1) t = vm_dual_mul(rs, lv_int(rs, (int64_t)i, 1), t);
        s = lv_add(rs, s, t);
    }
    return s;
}

static VmDual* lv_unary(VmRegionStack* rs, const VmDual* a0, int op) {
    VmDual* a = lv_own(rs, a0);
    if (!a) return NULL;
    uint32_t epoch = a->epoch, n = a->order;
    VmDual* r = lv_alloc(rs, epoch, n);
    if (!r) return NULL;
    VmDual** u = a->lcoeff;
    VmDual** y = r->lcoeff;
    int hint = vm_dual_exact_value(a) != NULL;
    if (op == LV_NEG) {
        for (uint32_t k = 0; k <= n; ++k) y[k] = vm_dual_neg(rs, u[k]);
        return lv_finish(rs, r, hint);
    }
    if (op == LV_ABS || op == LV_RELU) {
        /* Branches read the primal, recursively down to a plain number; the
         * zero subgradient is used at 0 for both. */
        VmRational* e = vm_dual_exact_value(a);
        double p = lv_primal(a);
        int sign = e ? vm_rational_sign(e) : (p > 0.0 ? 1 : (p < 0.0 ? -1 : 0));
        for (uint32_t k = 0; k <= n; ++k) {
            if (sign > 0) y[k] = u[k];
            else if (sign < 0 && op == LV_ABS) y[k] = vm_dual_neg(rs, u[k]);
            else y[k] = NULL;
        }
        return lv_finish(rs, r, hint);
    }
    y[0] = lv_unary_base(rs, u[0], op);
    if (!y[0]) return NULL;
    if (op == LV_EXP) {
        for (uint32_t k = 1; k <= n; ++k) y[k] = lv_scale(rs, lv_dconv(rs, u, y, k), 1, k);
    } else if (op == LV_SIN || op == LV_COS || op == LV_SINH || op == LV_COSH) {
        /* Coupled pair: y' = s1 * w u', w' = s2 * y u'. */
        VmDual** w = (VmDual**)vm_alloc(rs, ((size_t)n + 1) * sizeof(VmDual*));
        if (!w) return NULL;
        w[0] = lv_unary_base(rs, u[0], op == LV_SIN ? LV_COS : op == LV_COS ? LV_SIN
                                       : op == LV_SINH ? LV_COSH : LV_SINH);
        int64_t sy = op == LV_COS ? -1 : 1;
        int64_t sw = op == LV_SIN ? -1 : 1;
        for (uint32_t k = 1; k <= n; ++k) {
            y[k] = lv_scale(rs, lv_dconv(rs, u, w, k), sy, k);
            w[k] = lv_scale(rs, lv_dconv(rs, u, y, k), sw, k);
        }
    } else if (op == LV_LOG) {
        /* u y' = u'  =>  k u0 y_k = k u_k - sum_{i=1}^{k-1} i y_i u_{k-i} */
        for (uint32_t k = 1; k <= n; ++k) {
            VmDual* s = NULL;
            for (uint32_t i = 1; i < k; ++i) {
                VmDual* t = lv_mul(rs, y[i], u[k - i]);
                if (t && i > 1) t = vm_dual_mul(rs, lv_int(rs, (int64_t)i, 1), t);
                s = lv_add(rs, s, t);
            }
            VmDual* num = lv_sub(rs, u[k], lv_scale(rs, s, 1, k));
            y[k] = num ? vm_dual_div(rs, num, u[0]) : NULL;
        }
    } else if (op == LV_SQRT) {
        VmDual* two_y0 = vm_dual_mul(rs, lv_int(rs, 2, 1), y[0]);
        for (uint32_t k = 1; k <= n; ++k) {
            VmDual* s = NULL;
            for (uint32_t i = 1; i < k; ++i) s = lv_add(rs, s, lv_mul(rs, y[i], y[k - i]));
            VmDual* num = lv_sub(rs, u[k], s);
            y[k] = num ? vm_dual_div(rs, num, two_y0) : NULL;
        }
    } else if (op == LV_SIGMOID || op == LV_TANH) {
        /* y' = w u' with w = y(1-y) for sigmoid and 1-y^2 for tanh. */
        VmDual** w = (VmDual**)vm_alloc(rs, ((size_t)n + 1) * sizeof(VmDual*));
        if (!w) return NULL;
        for (uint32_t k = 0; k <= n; ++k) {
            if (k > 0) y[k] = lv_scale(rs, lv_dconv(rs, u, w, k), 1, k);
            VmDual* sq = NULL;
            for (uint32_t i = 0; i <= k; ++i) sq = lv_add(rs, sq, lv_mul(rs, y[i], y[k - i]));
            if (op == LV_SIGMOID) w[k] = lv_sub(rs, y[k], sq);
            else w[k] = k == 0 ? lv_sub(rs, lv_int(rs, 1, 0), sq) : lv_sub(rs, NULL, sq);
        }
    } else {
        return NULL;
    }
    return lv_finish(rs, r, hint);
}

/* u^p for a constant real p: u y' = p y u', so
 * k u0 y_k = sum_{i=1..k} (p i - (k - i)) u_i y_{k-i}. */
static VmDual* lv_pow_real(VmRegionStack* rs, const VmDual* a0, double p) {
    VmDual* a = lv_own(rs, a0);
    if (!a) return NULL;
    uint32_t n = a->order;
    VmDual* r = lv_alloc(rs, a->epoch, n);
    if (!r) return NULL;
    VmDual** u = a->lcoeff;
    VmDual** y = r->lcoeff;
    y[0] = vm_dual_pow(rs, u[0], p);
    if (!y[0]) return NULL;
    for (uint32_t k = 1; k <= n; ++k) {
        VmDual* s = NULL;
        for (uint32_t i = 1; i <= k; ++i) {
            VmDual* t = lv_mul(rs, u[i], y[k - i]);
            if (t) t = vm_dual_mul(rs, vm_dual_new(rs, p * (double)i - (double)(k - i), 0.0), t);
            s = lv_add(rs, s, t);
        }
        VmDual* den = vm_dual_mul(rs, lv_int(rs, (int64_t)k, 1), u[0]);
        y[k] = s ? vm_dual_div(rs, s, den) : NULL;
    }
    return lv_finish(rs, r, 0);
}

/* Integer powers by repeated multiplication: exact, and defined at 0. */
static VmDual* lv_pow_int(VmRegionStack* rs, const VmDual* a, int64_t e) {
    int exact = vm_dual_exact_value(a) != NULL;
    uint64_t m = e < 0 ? (uint64_t)(-(e + 1)) + 1u : (uint64_t)e;
    VmDual* acc = NULL;
    VmDual* base = lv_own(rs, a);
    while (base && m) {
        if (m & 1u) acc = acc ? vm_dual_mul(rs, acc, base) : base;
        m >>= 1;
        if (m) base = vm_dual_mul(rs, base, base);
    }
    if (!acc) {                       /* a^0 = 1, a constant of this level */
        VmDual* one = lv_alloc(rs, a->epoch, a->order);
        if (!one) return NULL;
        one->lcoeff[0] = lv_int(rs, 1, exact);
        return lv_finish(rs, one, exact);
    }
    if (e < 0) acc = vm_dual_div(rs, lv_int(rs, 1, exact), acc);
    return acc;
}

/* Does @p d vary in any perturbation? A level varies when a coefficient above
 * c[0] is nonzero or c[0] itself varies. */
static int lv_varies(const VmDual* d);
static int lv_coeff_nonzero(const VmDual* d) {
    if (!d) return 0;
    if (d->kind != VM_DUAL_KIND_SCALAR) return 1;
    return d->primal != 0.0 || lv_scalar_has_tangent(d) ||
           (d->eprimal && !vm_rational_is_zero(d->eprimal));
}
static int lv_varies(const VmDual* d) {
    if (!vm_dual_is_level(d)) return -1;
    for (uint32_t k = 1; k <= d->order; ++k) if (lv_coeff_nonzero(d->lcoeff[k])) return 1;
    const VmDual* c0 = d->lcoeff[0];
    if (vm_dual_is_level(c0)) return lv_varies(c0);
    return c0->kind != VM_DUAL_KIND_SCALAR || lv_scalar_has_tangent(c0);
}

/* The degree at which the perturbation part of @p d vanishes: a scalar dual's
 * is 1, a tower's its order, and a level's its own order plus the largest of
 * its coefficients', since the perturbations of nested levels multiply. */
uint32_t vm_dual_nilpotent_degree(const VmDual* d) {
    if (!d) return 0;
    if (d->kind == VM_DUAL_KIND_SCALAR) return lv_scalar_has_tangent(d) ? 1u : 0u;
    if (d->kind == VM_DUAL_KIND_TAYLOR) return d->order;
    uint32_t inner = 0;
    for (uint32_t k = 0; k <= d->order; ++k) {
        uint32_t n = vm_dual_nilpotent_degree(d->lcoeff[k]);
        if (n > inner) inner = n;
    }
    return d->order + inner;
}

VmDual* vm_dual_level_seed(VmRegionStack* rs, const VmDual* point,
                           uint32_t order, uint32_t epoch) {
    if (!rs || !point) return NULL;
    VmDual* d = lv_alloc(rs, epoch, order);
    if (!d) return NULL;
    int exact = vm_dual_exact_value(point) != NULL;
    d->lcoeff[0] = lv_own(rs, point);
    if (order >= 1) d->lcoeff[1] = lv_int(rs, 1, exact);
    return lv_finish(rs, d, exact);
}

VmDual* vm_dual_level_coefficient(VmRegionStack* rs, const VmDual* r,
                                  uint32_t epoch, uint32_t k) {
    if (!r || r->kind == VM_DUAL_KIND_SCALAR || r->epoch != epoch) return NULL;
    VmDual* c = lv_coeff(rs, r, epoch, k);
    return c ? c : lv_int(rs, 0, vm_dual_exact_value(r) != NULL);
}

VmDual* vm_dual_level_derivative(VmRegionStack* rs, const VmDual* r,
                                 uint32_t epoch, uint32_t k) {
    VmDual* c = vm_dual_level_coefficient(rs, r, epoch, k);
    if (!c || k < 2) return c;
    VmRational* fact = vm_rational_from_int(vm_active_arena(rs), 1);
    for (uint32_t i = 2; fact && i <= k; ++i)
        fact = vm_rational_op_exact(rs, fact,
                   vm_rational_from_int(vm_active_arena(rs), (int64_t)i), '*');
    if (!fact) return NULL;
    return vm_dual_mul(rs, vm_dual_constant(rs, vm_rational_to_double(fact), fact), c);
}

/* ── Functions defined through their derivative (ADR-0027 section 2) ──
 *
 * asin, acos, atan and atan2 have no closed coefficient recurrence of their
 * own, but each is the integral of a function the carrier arithmetic already
 * computes: y' = g(u) u' with g = 1/sqrt(1-u^2), -1/sqrt(1-u^2), 1/(1+u^2).
 * On a series u that gives k y_k = sum_{i=1..k} i u_i g_{k-i}, with y_0 the
 * same function applied to the coefficient u_0 -- recursively, so every
 * enclosing level is carried. A classic tower is read as a level of its own
 * epoch for this. */

static VmDual* lv_as_level(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_own(rs, a);
    if (!a || a->kind != VM_DUAL_KIND_TAYLOR) return NULL;
    VmDual* d = lv_alloc(rs, a->epoch, a->order);
    if (!d) return NULL;
    for (uint32_t k = 0; k <= a->order; ++k) d->lcoeff[k] = lv_coeff(rs, a, a->epoch, k);
    return lv_finish(rs, d, a->exact_coeff != NULL);
}

static VmDual* lv_integrate(VmRegionStack* rs, VmDual* u, const VmDual* g0,
                            VmDual* y0) {
    VmDual* g = lv_own(rs, g0);
    VmDual* r = u && g && y0 ? lv_alloc(rs, u->epoch, u->order) : NULL;
    if (!r) return NULL;
    VmDual** gk = (VmDual**)vm_alloc(rs, ((size_t)u->order + 1) * sizeof(VmDual*));
    if (!gk) return NULL;
    for (uint32_t k = 0; k <= u->order; ++k) gk[k] = lv_coeff(rs, g, u->epoch, k);
    r->lcoeff[0] = y0;
    for (uint32_t k = 1; k <= u->order; ++k)
        r->lcoeff[k] = lv_scale(rs, lv_dconv(rs, u->lcoeff, gk, k), 1, k);
    return lv_finish(rs, r, 0);
}

/* which: 0 asin, 1 acos, 2 atan */
VmDual* vm_dual_inverse_trig(VmRegionStack* rs, const VmDual* a, int which) {
    if (!a) return NULL;
    if (a->kind == VM_DUAL_KIND_SCALAR) {
        double p = a->primal, t = a->tangent;
        if (which == 2) return vm_dual_new(rs, atan(p), t / (1.0 + p * p));
        double s = t == 0.0 ? 0.0 : t / sqrt(1.0 - p * p);
        return vm_dual_new(rs, which == 0 ? asin(p) : acos(p), which == 0 ? s : -s);
    }
    VmDual* u = lv_as_level(rs, a);
    if (!u) return NULL;
    VmDual* one = vm_dual_new(rs, 1.0, 0.0);
    VmDual* sq = vm_dual_mul(rs, u, u);
    VmDual* g = which == 2
        ? vm_dual_div(rs, one, vm_dual_add(rs, one, sq))
        : vm_dual_div(rs, one, vm_dual_sqrt(rs, vm_dual_sub(rs, one, sq)));
    if (g && which == 1) g = vm_dual_neg(rs, g);
    return lv_integrate(rs, u, g, vm_dual_inverse_trig(rs, u->lcoeff[0], which));
}

/* atan2(y, x): the angle of (x, y); its derivative is that of atan(y/x). */
VmDual* vm_dual_atan2(VmRegionStack* rs, const VmDual* y, const VmDual* x) {
    if (!y || !x) return NULL;
    if (y->kind == VM_DUAL_KIND_SCALAR && x->kind == VM_DUAL_KIND_SCALAR) {
        double r2 = x->primal * x->primal + y->primal * y->primal;
        double t = x->tangent == 0.0 && y->tangent == 0.0 ? 0.0
                 : (x->primal * y->tangent - y->primal * x->tangent) / r2;
        return vm_dual_new(rs, atan2(y->primal, x->primal), t);
    }
    VmDual* w = vm_dual_div(rs, y, x);
    VmDual* u = w && w->kind != VM_DUAL_KIND_SCALAR ? lv_as_level(rs, w) : NULL;
    if (!u) return NULL;
    VmDual* one = vm_dual_new(rs, 1.0, 0.0);
    VmDual* g = vm_dual_div(rs, one, vm_dual_add(rs, one, vm_dual_mul(rs, u, u)));
    VmDual* ya = lv_own(rs, y);
    VmDual* xa = lv_own(rs, x);
    VmDual* y0 = ya ? lv_coeff(rs, ya, u->epoch, 0) : NULL;
    VmDual* x0 = xa ? lv_coeff(rs, xa, u->epoch, 0) : NULL;
    return lv_integrate(rs, u, g, y0 && x0 ? vm_dual_atan2(rs, y0, x0) : NULL);
}

/* ── Core Operations ── */

/** @brief Native call 370: `(make-dual primal tangent)`. */
VmDual* vm_dual_make(VmRegionStack* rs, double primal, double tangent) {
    return vm_dual_new(rs, primal, tangent);
}

/** @brief Native call 373: dual addition, (a+a'e)+(b+b'e) = (a+b)+(a'+b')e.
 *         Exactness-preserving: exact when both operands are. */
VmDual* vm_dual_add(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    if (lv_route(a, b)) return lv_binary(rs, a, b, '+');
    if (vm_dual_is_taylor(a) || vm_dual_is_taylor(b)) return taylor_binary(rs, a, b, '+');
    if (dual_is_exact(a) && dual_is_exact(b)) {
        VmRational* p = rex(rs, a->eprimal,  b->eprimal,  '+');
        VmRational* t = rex(rs, a->etangent, b->etangent, '+');
        if (p && t) return vm_dual_new_exact(rs, p, t,
                                             a->primal + b->primal,
                                             a->tangent + b->tangent);
    }
    return vm_dual_new(rs, a->primal + b->primal, a->tangent + b->tangent);
}

/** @brief Native call 374: dual subtraction, (a+a'e)-(b+b'e) = (a-b)+(a'-b')e.
 *         Exactness-preserving: exact when both operands are. */
VmDual* vm_dual_sub(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    if (lv_route(a, b)) return lv_binary(rs, a, b, '-');
    if (vm_dual_is_taylor(a) || vm_dual_is_taylor(b)) return taylor_binary(rs, a, b, '-');
    if (dual_is_exact(a) && dual_is_exact(b)) {
        VmRational* p = rex(rs, a->eprimal,  b->eprimal,  '-');
        VmRational* t = rex(rs, a->etangent, b->etangent, '-');
        if (p && t) return vm_dual_new_exact(rs, p, t,
                                             a->primal - b->primal,
                                             a->tangent - b->tangent);
    }
    return vm_dual_new(rs, a->primal - b->primal, a->tangent - b->tangent);
}

/** @brief Native call 375: dual multiplication (product rule), (a+a'e)(b+b'e)
 *         = ab + (a'b+ab')e. */
VmDual* vm_dual_mul(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    if (lv_route(a, b)) return lv_binary(rs, a, b, '*');
    if (vm_dual_is_taylor(a) || vm_dual_is_taylor(b)) return taylor_binary(rs, a, b, '*');
    if (dual_is_exact(a) && dual_is_exact(b)) {
        /* product rule, entirely in the exact domain */
        VmRational* p  = rex(rs, a->eprimal,  b->eprimal,  '*');
        VmRational* l  = rex(rs, a->etangent, b->eprimal,  '*');
        VmRational* r  = rex(rs, a->eprimal,  b->etangent, '*');
        VmRational* t  = rex(rs, l, r, '+');
        if (p && t) return vm_dual_new_exact(rs, p, t,
                        a->primal * b->primal,
                        a->tangent * b->primal + a->primal * b->tangent);
    }
    return vm_dual_new(rs,
        a->primal * b->primal,
        a->tangent * b->primal + a->primal * b->tangent);
}

/** @brief Native call 376: dual division (quotient rule), (a+a'e)/(b+b'e) =
 *         a/b + (a'b-ab')/b^2 e. */
VmDual* vm_dual_div(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    if (lv_route(a, b)) return lv_binary(rs, a, b, '/');
    if (vm_dual_is_taylor(a) || vm_dual_is_taylor(b)) return taylor_binary(rs, a, b, '/');
    double b2 = b->primal * b->primal;
    if (dual_is_exact(a) && dual_is_exact(b)) {
        /* quotient rule. rex() answers NULL on exact division by exact zero,
         * so a 1/0 falls through to the double arm and its infinity rather
         * than fabricating an exact value for it. */
        VmRational* p  = rex(rs, a->eprimal, b->eprimal, '/');
        VmRational* l  = rex(rs, a->etangent, b->eprimal,  '*');
        VmRational* r  = rex(rs, a->eprimal,  b->etangent, '*');
        VmRational* n  = rex(rs, l, r, '-');
        VmRational* d2 = rex(rs, b->eprimal, b->eprimal, '*');
        VmRational* t  = rex(rs, n, d2, '/');
        if (p && t) return vm_dual_new_exact(rs, p, t,
                        a->primal / b->primal,
                        (a->tangent * b->primal - a->primal * b->tangent) / b2);
    }
    return vm_dual_new(rs,
        a->primal / b->primal,
        (a->tangent * b->primal - a->primal * b->tangent) / b2);
}

/** @brief Native call 377: dual sin, sin(a+a'e) = sin(a) + a'*cos(a)*e. */
VmDual* vm_dual_sin(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_SIN);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 4);
    double s = sin(a->primal);
    double c = cos(a->primal);
    return vm_dual_new(rs, s, a->tangent * c);
}

/** @brief Native call 378: dual cos, cos(a+a'e) = cos(a) - a'*sin(a)*e. */
VmDual* vm_dual_cos(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_COS);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 5);
    double c = cos(a->primal);
    double s = sin(a->primal);
    return vm_dual_new(rs, c, -a->tangent * s);
}

/** @brief Native call 379: dual exp, exp(a+a'e) = exp(a) + a'*exp(a)*e. */
VmDual* vm_dual_exp(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_EXP);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 3);
    double ea = exp(a->primal);
    return vm_dual_new(rs, ea, a->tangent * ea);
}

/** @brief Native call 380: dual log, log(a+a'e) = log(a) + (a'/a)*e. */
VmDual* vm_dual_log(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_LOG);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 6);
    return vm_dual_new(rs, log(a->primal), a->tangent / a->primal);
}

/** @brief Native call 381: dual sqrt, sqrt(a+a'e) = sqrt(a) +
 *         a'/(2*sqrt(a))*e. */
VmDual* vm_dual_sqrt(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_SQRT);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 7);
    double sa = sqrt(a->primal);
    return vm_dual_new(rs, sa, a->tangent / (2.0 * sa));
}

VmDual* vm_dual_sinh(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_SINH);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 11);
    return vm_dual_new(rs, sinh(a->primal), a->tangent * cosh(a->primal));
}

VmDual* vm_dual_cosh(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_COSH);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 10);
    return vm_dual_new(rs, cosh(a->primal), a->tangent * sinh(a->primal));
}

/**
 * @brief Native call 382: dual power with constant exponent @p n,
 *        (a+a'e)^n = a^n + n*a^(n-1)*a'*e. @p n must be a plain constant,
 *        not itself a dual number; for a dual exponent use exp(n*log(a))
 *        instead.
 */
VmDual* vm_dual_pow(VmRegionStack* rs, const VmDual* a, double n) {
    if (vm_dual_is_level(a))
        return isfinite(n) && n == floor(n) && fabs(n) < 9.0e18
            ? lv_pow_int(rs, a, (int64_t)n) : lv_pow_real(rs, a, n);
    if (vm_dual_is_taylor(a)) {
        if (isfinite(n) && n == floor(n))
            return taylor_pow_integer(rs, a, (int64_t)n);
        VmDual* ln = taylor_unary(rs, a, 6);
        VmDual scale = {0}; scale.primal = n;
        VmDual* product = ln ? taylor_binary(rs, &scale, ln, '*') : NULL;
        return product ? taylor_unary(rs, product, 3) : NULL;
    }
    double p = pow(a->primal, n);
    double dp = n == 0.0 ? 0.0 : n * pow(a->primal, n - 1.0) * a->tangent;
    /* SW-85: every integer exponent is exactness-preserving at a nonzero exact
     * point. Negative powers stay in the rational domain through reciprocal
     * arithmetic; fractional powers still use the inexact libm path. */
    if (dual_is_exact(a) && isfinite(n) && n == floor(n)) {
        int64_t k = (int64_t)n;
        VmRational* acc = vm_rational_from_int(vm_active_arena(rs), 1);   /* a^k   */
        VmRational* acck1 = NULL;                                        /* a^(k-1) */
        int ok = (acc != NULL);
        uint64_t magnitude = k < 0 ? (uint64_t)(-(k + 1)) + 1u : (uint64_t)k;
        for (uint64_t i = 0; ok && i < magnitude; i++) {
            acck1 = acc;
            acc = rex(rs, acc, a->eprimal, '*');
            if (!acc) ok = 0;
        }
        if (ok && k < 0) {
            VmRational* one = vm_rational_from_int(vm_active_arena(rs), 1);
            VmRational* value = one ? rex(rs, one, acc, '/') : NULL;
            VmRational* magnitude_r = vm_rational_from_int(
                vm_active_arena(rs), (int64_t)magnitude);
            VmRational* derivative = magnitude_r ? rex(rs, magnitude_r, value, '*') : NULL;
            derivative = derivative ? rex(rs, derivative, a->eprimal, '/') : NULL;
            VmRational* neg = derivative ? vm_rational_negate_exact(rs, derivative) : NULL;
            derivative = neg ? rex(rs, neg, a->etangent, '*') : NULL;
            if (value && derivative)
                return vm_dual_new_exact(rs, value, derivative, p, dp);
        } else if (ok && k == 0) {
            /* d/dx of a constant 1 is 0 */
            VmRational* zero = vm_rational_from_int(vm_active_arena(rs), 0);
            if (zero) return vm_dual_new_exact(rs, acc, zero, p, dp);
        } else if (ok && acck1) {
            VmRational* kr = vm_rational_from_int(vm_active_arena(rs), k);
            VmRational* t  = rex(rs, kr, acck1, '*');
            t = rex(rs, t, a->etangent, '*');
            if (t) return vm_dual_new_exact(rs, acc, t, p, dp);
        }
    }
    return vm_dual_new(rs, p, dp);
}

static int vm_dual_has_variation(const VmDual* d) {
    if (vm_dual_is_level(d)) return lv_varies(d);
    if (d->tangent != 0.0 || (d->etangent && !vm_rational_is_zero(d->etangent)))
        return 1;
    if (!vm_dual_is_taylor(d)) return 0;
    for (uint32_t k = 1; k <= d->order; ++k)
        if (d->coeff[k] != 0.0 ||
            (d->exact_coeff && d->exact_coeff[k] && !vm_rational_is_zero(d->exact_coeff[k])))
            return 1;
    return 0;
}

/* The exponent is an operand too: coercing an active exponent to its primal
 * silently differentiates a different function. Keep constant exponents on
 * the integer-power route (including negative bases and exact coefficients). */
static VmDual* vm_dual_pow_active(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    if (!vm_dual_has_variation(b)) return vm_dual_pow(rs, a, b->primal);
    VmDual* logarithm = vm_dual_log(rs, a);
    VmDual* product = logarithm ? vm_dual_mul(rs, b, logarithm) : NULL;
    VmDual* result = product ? vm_dual_exp(rs, product) : NULL;
    if (result && !vm_dual_is_level(result)) {
        result->primal = pow(a->primal, b->primal);
        if (vm_dual_is_taylor(result) && result->coeff)
            result->coeff[0] = result->primal;
    }
    return result;
}

/** @brief Native call 383: dual absolute value, |a+a'e| = |a| +
 *         a'*sign(a)*e. */
VmDual* vm_dual_abs(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_ABS);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 1);
    if (dual_is_exact(a)) {
        int sign = vm_rational_sign(a->eprimal);
        VmRational* p = vm_rational_absolute_exact(rs, a->eprimal);
        VmRational* t = sign == 0 ? vm_rational_from_int(vm_active_arena(rs), 0)
                                  : (sign < 0 ? vm_rational_negate_exact(rs, a->etangent)
                                              : a->etangent);
        if (p && t)
            return vm_dual_new_exact(rs, p, t, fabs(a->primal),
                                     a->tangent * (double)sign);
    }
    double sign;
    if (a->primal > 0.0) sign = 1.0;
    else if (a->primal < 0.0) sign = -1.0;
    else sign = 0.0;
    return vm_dual_new(rs, fabs(a->primal), a->tangent * sign);
}

/** @brief Native call 384: dual negation, -(a+a'e) = -a + (-a')e. */
VmDual* vm_dual_neg(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_NEG);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 0);
    if (dual_is_exact(a)) {
        VmRational* p = vm_rational_negate_exact(rs, a->eprimal);
        VmRational* t = vm_rational_negate_exact(rs, a->etangent);
        if (p && t) return vm_dual_new_exact(rs, p, t, -a->primal, -a->tangent);
    }
    return vm_dual_new(rs, -a->primal, -a->tangent);
}

/** @brief Native call 385: dual ReLU, relu(a+a'e) = max(0,a) + (a>0 ? a' :
 *         0)*e. */
VmDual* vm_dual_relu(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_RELU);
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 2);
    if (dual_is_exact(a)) {
        int sign = vm_rational_sign(a->eprimal);
        VmRational* zero = vm_rational_from_int(vm_active_arena(rs), 0);
        VmRational* p = sign > 0 ? a->eprimal : zero;
        VmRational* t = sign > 0 ? a->etangent : zero;
        if (p && t)
            return vm_dual_new_exact(rs, p, t,
                                     sign > 0 ? a->primal : 0.0,
                                     sign > 0 ? a->tangent : 0.0);
    }
    if (a->primal > 0.0)
        return vm_dual_new(rs, a->primal, a->tangent);
    else
        return vm_dual_new(rs, 0.0, 0.0);
}

/** @brief Native call 386: dual sigmoid, sigma(a+a'e) = sigma(a) +
 *         a'*sigma(a)*(1-sigma(a))*e. */
VmDual* vm_dual_sigmoid(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_SIGMOID);
    if (vm_dual_is_taylor(a)) {
        return taylor_unary(rs, a, 8);
    }
    double sig = 1.0 / (1.0 + exp(-a->primal));
    return vm_dual_new(rs, sig, a->tangent * sig * (1.0 - sig));
}

/** @brief Native call 387: dual tanh, tanh(a+a'e) = tanh(a) + a'*(1 -
 *         tanh(a)^2)*e. */
VmDual* vm_dual_tanh(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_unary(rs, a, LV_TANH);
    if (vm_dual_is_taylor(a)) {
        return taylor_unary(rs, a, 9);
    }
    double th = tanh(a->primal);
    return vm_dual_new(rs, th, a->tangent * (1.0 - th * th));
}

/** @brief Native call 388: promote a plain scalar @p x to a dual constant
 *         (zero tangent). */
VmDual* vm_dual_from_double(VmRegionStack* rs, double x) {
    return vm_dual_new(rs, x, 0.0);
}

/** @brief Native call 389: scale dual @p a by scalar @p c, c*(a+a'e) =
 *         c*a + c*a'*e. */
VmDual* vm_dual_scale(VmRegionStack* rs, double c, const VmDual* a) {
    if (vm_dual_is_level(a)) return lv_binary(rs, vm_dual_new(rs, c, 0.0), a, '*');
    if (vm_dual_is_taylor(a)) {
        VmDual scalar = {0}; scalar.primal = c;
        return taylor_binary(rs, &scalar, a, '*');
    }
    return vm_dual_new(rs, c * a->primal, c * a->tangent);
}

/*******************************************************************************
 * Dispatch — called from bytecode VM's NATIVE_CALL instruction
 ******************************************************************************/

typedef struct { double d; void* p; } VmDualResult;

/**
 * vm_dual_dispatch — route a native call ID in [370,389] to the
 * correct dual-number operation.
 *
 * @param rs   Active region stack (for allocation)
 * @param id   Native call ID (370-389)
 * @param args Pointer to argument array (doubles and VmDual*)
 * @param nargs Number of arguments
 * @return Pointer to result VmDual, or NULL on error
 */
void* vm_dual_dispatch(VmRegionStack* rs, int id, void** args, int nargs) {
    switch (id) {
    case 370: /* make-dual(primal, tangent) */
        if (nargs < 2) return NULL;
        return vm_dual_make(rs, *(double*)args[0], *(double*)args[1]);

    case 371: /* dual-primal(d) — returns double, caller must unpack */
        return args[0] ? (void*)&((VmDual*)args[0])->primal : NULL;

    case 372: /* dual-tangent(d) — returns double, caller must unpack */
        return args[0] ? (void*)&((VmDual*)args[0])->tangent : NULL;

    case 373: return vm_dual_add(rs, (VmDual*)args[0], (VmDual*)args[1]);
    case 374: return vm_dual_sub(rs, (VmDual*)args[0], (VmDual*)args[1]);
    case 375: return vm_dual_mul(rs, (VmDual*)args[0], (VmDual*)args[1]);
    case 376: return vm_dual_div(rs, (VmDual*)args[0], (VmDual*)args[1]);
    case 377: return vm_dual_sin(rs, (VmDual*)args[0]);
    case 378: return vm_dual_cos(rs, (VmDual*)args[0]);
    case 379: return vm_dual_exp(rs, (VmDual*)args[0]);
    case 380: return vm_dual_log(rs, (VmDual*)args[0]);
    case 381: return vm_dual_sqrt(rs, (VmDual*)args[0]);
    case 382: return vm_dual_pow(rs, (VmDual*)args[0], *(double*)args[1]);
    case 383: return vm_dual_abs(rs, (VmDual*)args[0]);
    case 384: return vm_dual_neg(rs, (VmDual*)args[0]);
    case 385: return vm_dual_relu(rs, (VmDual*)args[0]);
    case 386: return vm_dual_sigmoid(rs, (VmDual*)args[0]);
    case 387: return vm_dual_tanh(rs, (VmDual*)args[0]);
    case 388: return vm_dual_from_double(rs, *(double*)args[0]);
    case 389: return vm_dual_scale(rs, *(double*)args[0], (VmDual*)args[1]);

    default:
        fprintf(stderr, "ERROR: unknown dual native ID %d\n", id);
        return NULL;
    }
}

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_DUAL_TEST

#include <assert.h>

#define DUAL_EPS 1e-12

/** @brief Approximate equality check used by the self-test assertions
 *         below. */
static int dual_near(double a, double b) {
    return fabs(a - b) < DUAL_EPS;
}

/** @brief Standalone self-test (built when VM_DUAL_TEST is defined):
 *         verifies forward-mode derivatives of each dual operation
 *         (including chain rule and quotient rule compositions) against
 *         known analytic values. */
int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s\n", name); } \
} while(0)

    printf("=== vm_dual self-test ===\n\n");

    /* --- derivative of sin at 0: d/dx sin(x)|_{x=0} = cos(0) = 1 --- */
    {
        VmDual* x = vm_dual_make(&rs, 0.0, 1.0); /* x=0, dx=1 */
        VmDual* y = vm_dual_sin(&rs, x);
        CHECK("sin(0) primal = 0", dual_near(y->primal, 0.0));
        CHECK("d/dx sin(x)|_{x=0} = 1", dual_near(y->tangent, 1.0));
    }

    /* --- derivative of x^2 at x=3: d/dx x^2 = 2x = 6 --- */
    {
        VmDual* x = vm_dual_make(&rs, 3.0, 1.0);
        VmDual* y = vm_dual_mul(&rs, x, x); /* x * x = x^2 */
        CHECK("x^2 at x=3: primal = 9", dual_near(y->primal, 9.0));
        CHECK("d/dx x^2 at x=3 = 6", dual_near(y->tangent, 6.0));
    }

    /* --- derivative of exp at 0: d/dx exp(x)|_{x=0} = exp(0) = 1 --- */
    {
        VmDual* x = vm_dual_make(&rs, 0.0, 1.0);
        VmDual* y = vm_dual_exp(&rs, x);
        CHECK("exp(0) primal = 1", dual_near(y->primal, 1.0));
        CHECK("d/dx exp(x)|_{x=0} = 1", dual_near(y->tangent, 1.0));
    }

    /* --- derivative of cos at 0: d/dx cos(x)|_{x=0} = -sin(0) = 0 --- */
    {
        VmDual* x = vm_dual_make(&rs, 0.0, 1.0);
        VmDual* y = vm_dual_cos(&rs, x);
        CHECK("cos(0) primal = 1", dual_near(y->primal, 1.0));
        CHECK("d/dx cos(x)|_{x=0} = 0", dual_near(y->tangent, 0.0));
    }

    /* --- derivative of log at 1: d/dx log(x)|_{x=1} = 1/1 = 1 --- */
    {
        VmDual* x = vm_dual_make(&rs, 1.0, 1.0);
        VmDual* y = vm_dual_log(&rs, x);
        CHECK("log(1) primal = 0", dual_near(y->primal, 0.0));
        CHECK("d/dx log(x)|_{x=1} = 1", dual_near(y->tangent, 1.0));
    }

    /* --- derivative of sqrt at 4: d/dx sqrt(x)|_{x=4} = 1/(2*2) = 0.25 --- */
    {
        VmDual* x = vm_dual_make(&rs, 4.0, 1.0);
        VmDual* y = vm_dual_sqrt(&rs, x);
        CHECK("sqrt(4) primal = 2", dual_near(y->primal, 2.0));
        CHECK("d/dx sqrt(x)|_{x=4} = 0.25", dual_near(y->tangent, 0.25));
    }

    /* --- derivative of x^3 at x=2: d/dx x^3 = 3x^2 = 12 via pow --- */
    {
        VmDual* x = vm_dual_make(&rs, 2.0, 1.0);
        VmDual* y = vm_dual_pow(&rs, x, 3.0);
        CHECK("pow(2,3) primal = 8", dual_near(y->primal, 8.0));
        CHECK("d/dx x^3 at x=2 = 12", dual_near(y->tangent, 12.0));
    }

    /* --- derivative of abs at -3: d/dx |x| = sign(x) = -1 --- */
    {
        VmDual* x = vm_dual_make(&rs, -3.0, 1.0);
        VmDual* y = vm_dual_abs(&rs, x);
        CHECK("abs(-3) primal = 3", dual_near(y->primal, 3.0));
        CHECK("d/dx |x| at x=-3 = -1", dual_near(y->tangent, -1.0));
    }

    /* --- derivative of relu at 3: 1; at -1: 0 --- */
    {
        VmDual* x1 = vm_dual_make(&rs, 3.0, 1.0);
        VmDual* y1 = vm_dual_relu(&rs, x1);
        CHECK("relu(3) primal = 3", dual_near(y1->primal, 3.0));
        CHECK("d/dx relu(x) at x=3 = 1", dual_near(y1->tangent, 1.0));

        VmDual* x2 = vm_dual_make(&rs, -1.0, 1.0);
        VmDual* y2 = vm_dual_relu(&rs, x2);
        CHECK("relu(-1) primal = 0", dual_near(y2->primal, 0.0));
        CHECK("d/dx relu(x) at x=-1 = 0", dual_near(y2->tangent, 0.0));
    }

    /* --- derivative of sigmoid at 0: sigma(0)=0.5, sigma'(0)=0.25 --- */
    {
        VmDual* x = vm_dual_make(&rs, 0.0, 1.0);
        VmDual* y = vm_dual_sigmoid(&rs, x);
        CHECK("sigmoid(0) primal = 0.5", dual_near(y->primal, 0.5));
        CHECK("d/dx sigmoid(x)|_{x=0} = 0.25", dual_near(y->tangent, 0.25));
    }

    /* --- derivative of tanh at 0: tanh(0)=0, tanh'(0)=1 --- */
    {
        VmDual* x = vm_dual_make(&rs, 0.0, 1.0);
        VmDual* y = vm_dual_tanh(&rs, x);
        CHECK("tanh(0) primal = 0", dual_near(y->primal, 0.0));
        CHECK("d/dx tanh(x)|_{x=0} = 1", dual_near(y->tangent, 1.0));
    }

    /* --- chain rule: d/dx sin(x^2) at x=2: cos(4)*4 --- */
    {
        VmDual* x = vm_dual_make(&rs, 2.0, 1.0);
        VmDual* x2 = vm_dual_mul(&rs, x, x);
        VmDual* y = vm_dual_sin(&rs, x2);
        double expected_primal = sin(4.0);
        double expected_tangent = cos(4.0) * 4.0; /* 2x * cos(x^2) at x=2 */
        CHECK("sin(x^2) at x=2 primal", dual_near(y->primal, expected_primal));
        CHECK("d/dx sin(x^2) at x=2 chain rule", dual_near(y->tangent, expected_tangent));
    }

    /* --- quotient rule: d/dx (x/(1+x^2)) at x=1 --- */
    {
        VmDual* x = vm_dual_make(&rs, 1.0, 1.0);
        VmDual* one = vm_dual_make(&rs, 1.0, 0.0);
        VmDual* x2 = vm_dual_mul(&rs, x, x);
        VmDual* denom = vm_dual_add(&rs, one, x2);
        VmDual* y = vm_dual_div(&rs, x, denom);
        /* f = x/(1+x^2), f' = (1+x^2 - x*2x)/(1+x^2)^2 = (1-x^2)/(1+x^2)^2
         * At x=1: (1-1)/(1+1)^2 = 0/4 = 0 */
        CHECK("x/(1+x^2) at x=1 primal = 0.5", dual_near(y->primal, 0.5));
        CHECK("d/dx x/(1+x^2) at x=1 = 0", dual_near(y->tangent, 0.0));
    }

    /* --- neg: d/dx (-x) = -1 --- */
    {
        VmDual* x = vm_dual_make(&rs, 5.0, 1.0);
        VmDual* y = vm_dual_neg(&rs, x);
        CHECK("neg(5) primal = -5", dual_near(y->primal, -5.0));
        CHECK("d/dx (-x) = -1", dual_near(y->tangent, -1.0));
    }

    /* --- scale: d/dx (3*x) = 3 --- */
    {
        VmDual* x = vm_dual_make(&rs, 2.0, 1.0);
        VmDual* y = vm_dual_scale(&rs, 3.0, x);
        CHECK("3*2 primal = 6", dual_near(y->primal, 6.0));
        CHECK("d/dx (3*x) = 3", dual_near(y->tangent, 3.0));
    }

    /* ADR-0027 level carrier: two nested levels over x*y at (2, 3).
     * The outer level (epoch 1) seeds x, the inner (epoch 2) seeds y; the
     * product's t2-coefficient is x, whose t1-coefficient is 1. */
    {
        VmDual* x = vm_dual_level_seed(&rs, vm_dual_constant(&rs, 2.0, NULL), 1, 1);
        VmDual* y = vm_dual_level_seed(&rs, vm_dual_constant(&rs, 3.0, NULL), 2, 2);
        VmDual* p = vm_dual_mul(&rs, x, y);
        CHECK("level product primal", p && dual_near(p->primal, 6.0));
        VmDual* c1 = vm_dual_level_coefficient(&rs, p, 2, 1);
        CHECK("inner coefficient is the outer carrier", c1 && vm_dual_is_level(c1) && dual_near(c1->primal, 2.0));
        VmDual* d = c1 ? vm_dual_level_coefficient(&rs, c1, 1, 1) : NULL;
        CHECK("d/dx d/dy xy = 1", d && dual_near(d->primal, 1.0));
        VmDual* e = vm_dual_exp(&rs, y);
        VmDual* e2 = vm_dual_level_derivative(&rs, e, 2, 2);
        CHECK("level exp second derivative", e2 && dual_near(e2->primal, exp(3.0)));
        VmDual* s = vm_dual_inverse_trig(&rs, y, 2);
        VmDual* s2 = vm_dual_level_derivative(&rs, s, 2, 2);
        CHECK("level atan second derivative", s2 && dual_near(s2->primal, -6.0 / 100.0));
        VmDual* ptower = vm_dual_make_taylor_seed(&rs, NULL, 2.0, 2, 0, 3);
        VmDual* mixed = vm_dual_mul(&rs, ptower, vm_dual_make(&rs, 5.0, 1.0));
        CHECK("tower times a live scalar dual is a level", mixed && vm_dual_is_level(mixed));
    }

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_DUAL_TEST */
