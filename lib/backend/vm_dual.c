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
    d->tangent_coeff = NULL;
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
    d->tangent_coeff = NULL;
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

static VmDual* taylor_alloc(VmRegionStack* rs, uint32_t order, int exact,
                            int with_tangent) {
    VmDual* d = (VmDual*)vm_alloc_object(rs, VM_SUBTYPE_DUAL, sizeof(VmDual));
    if (!d) return NULL;
    d->primal = d->tangent = 0.0;
    d->eprimal = d->etangent = NULL;
    d->kind = VM_DUAL_KIND_TAYLOR;
    d->order = order;
    d->epoch = 0;
    d->primal_sign = 0;
    d->coeff = (double*)vm_alloc(rs, (size_t)(order + 1) * sizeof(double));
    d->exact_coeff = exact
        ? (VmRational**)vm_alloc(rs, (size_t)(order + 1) * sizeof(VmRational*))
        : NULL;
    d->tangent_coeff = with_tangent
        ? (double*)vm_alloc(rs, (size_t)(order + 1) * sizeof(double)) : NULL;
    if (!d->coeff || (exact && !d->exact_coeff) ||
        (with_tangent && !d->tangent_coeff)) return NULL;
    memset(d->coeff, 0, (size_t)(order + 1) * sizeof(double));
    if (d->exact_coeff)
        memset(d->exact_coeff, 0, (size_t)(order + 1) * sizeof(VmRational*));
    if (d->tangent_coeff)
        memset(d->tangent_coeff, 0, (size_t)(order + 1) * sizeof(double));
    return d;
}

VmDual* vm_dual_make_taylor_seed(VmRegionStack* rs, VmRational* point,
                                 double point_value, uint32_t order, int exact,
                                 uint32_t epoch) {
    if (order > 4096u) return NULL;
    VmDual* d = taylor_alloc(rs, order, exact && point != NULL, 0);
    if (!d) return NULL;
    d->coeff[0] = point_value;
    if (order >= 1) d->coeff[1] = 1.0;
    d->primal = point_value;
    d->tangent = order >= 1 ? 1.0 : 0.0;
    d->epoch = epoch;
    if (d->exact_coeff) {
        d->exact_coeff[0] = point;
        for (uint32_t i = 2; i <= order; i++)
            d->exact_coeff[i] = vm_rational_from_int(vm_active_arena(rs), 0);
        for (uint32_t i = 2; i <= order; i++)
            if (!d->exact_coeff[i]) return NULL;
        if (order >= 1) {
            d->exact_coeff[1] = vm_rational_from_int(vm_active_arena(rs), 1);
            if (!d->exact_coeff[1]) return NULL;
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

static double taylor_coeff_as_double(const VmDual* d, uint32_t n) {
    if (!d) return 0.0;
    if (d->kind == VM_DUAL_KIND_TAYLOR)
        return n <= d->order ? d->coeff[n] : 0.0;
    return n == 0 ? d->primal : (n == 1 ? d->tangent : 0.0);
}

static uint32_t taylor_epoch(const VmDual* d) {
    return d && d->kind == VM_DUAL_KIND_TAYLOR ? d->epoch : 0;
}

/* Read a value coefficient in the active perturbation context.  A Taylor
 * carrier from another epoch is a constant at this level, so only c[0]
 * participates and all higher coefficients are zero. */
static double taylor_coeff_as_double_at(const VmDual* d, uint32_t n,
                                        uint32_t active_epoch) {
    if (!d) return 0.0;
    if (d->kind != VM_DUAL_KIND_TAYLOR)
        return n == 0 ? d->primal : 0.0;
    if (d->epoch != active_epoch)
        return n == 0 ? d->coeff[0] : 0.0;
    return n <= d->order ? d->coeff[n] : 0.0;
}

static VmRational* taylor_coeff_as_exact(const VmDual* d, uint32_t n) {
    if (!d) return NULL;
    if (d->kind == VM_DUAL_KIND_TAYLOR)
        return n <= d->order && d->exact_coeff ? d->exact_coeff[n] : NULL;
    if (n == 0) return d->eprimal;
    if (n == 1) return d->etangent;
    return NULL;
}

static VmRational* taylor_coeff_as_exact_at(const VmDual* d, uint32_t n,
                                            uint32_t active_epoch) {
    if (!d) return NULL;
    if (d->kind != VM_DUAL_KIND_TAYLOR)
        return n == 0 ? d->eprimal : NULL;
    if (d->epoch != active_epoch)
        return n == 0 && d->exact_coeff ? d->exact_coeff[0] : NULL;
    return n <= d->order && d->exact_coeff ? d->exact_coeff[n] : NULL;
}

static VmRational* taylor_coeff_as_exact_or_zero_at(VmRegionStack* rs,
                                                     const VmDual* d,
                                                     uint32_t n,
                                                     uint32_t active_epoch) {
    VmRational* r = taylor_coeff_as_exact_at(d, n, active_epoch);
    if (r) return r;
    if (d && (d->kind != VM_DUAL_KIND_TAYLOR ||
              (d->kind == VM_DUAL_KIND_TAYLOR && d->exact_coeff != NULL)))
        return vm_rational_from_int(vm_active_arena(rs), 0);
    return NULL;
}

static int dual_has_tangent(const VmDual* d) {
    return d && (d->tangent_coeff != NULL ||
                 (d->kind == VM_DUAL_KIND_SCALAR && d->tangent != 0.0));
}

static int dual_has_foreign_tangent(const VmDual* d, uint32_t active_epoch) {
    return d && d->kind == VM_DUAL_KIND_TAYLOR &&
           d->epoch != 0 && active_epoch != 0 &&
           d->epoch != active_epoch;
}

/* Tangent-side view shared by all Taylor recurrences.  A same-epoch Taylor
 * has no orthogonal tangent unless its companion is present.  A foreign epoch
 * is opaque coefficient data and therefore rides the companion unchanged. */
static double taylor_tangent_as_double_at(const VmDual* d, uint32_t n,
                                          uint32_t active_epoch) {
    if (!d) return 0.0;
    if (d->kind != VM_DUAL_KIND_TAYLOR)
        return n == 0 ? d->tangent : 0.0;
    if (d->tangent_coeff)
        return n <= d->order ? d->tangent_coeff[n] : 0.0;
    if (d->epoch != active_epoch)
        return n <= d->order ? d->coeff[n] : 0.0;
    return 0.0;
}

static int taylor_exact_primal_sign(VmRegionStack* rs,
                                    const VmDual* a, const VmDual* b,
                                    char op, uint32_t epoch) {
    VmRational* ar = taylor_coeff_as_exact_at(a, 0, epoch);
    VmRational* br = taylor_coeff_as_exact_at(b, 0, epoch);
    if (!ar) ar = vm_rational_from_double_exact(rs,
        taylor_coeff_as_double_at(a, 0, epoch));
    if (!br) br = vm_rational_from_double_exact(rs,
        taylor_coeff_as_double_at(b, 0, epoch));
    if (!ar || !br) return 0;
    VmRational* value = vm_rational_op_exact(rs, ar, br, op);
    return value ? vm_rational_sign(value) : 0;
}

static void taylor_exp_coeffs(VmRegionStack* rs, double* out,
                              const VmDual* a, uint32_t n) {
    out[0] = exp(taylor_coeff_as_double(a, 0));
    for (uint32_t k = 1; k < n; k++) {
        double sum = 0.0;
        for (uint32_t i = 1; i <= k; i++)
            sum += (double)i * taylor_coeff_as_double(a, i) * out[k - i];
        out[k] = sum / (double)k;
    }
    (void)rs;
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

static void taylor_mul_coeffs(double* out, const double* a, const double* b,
                              uint32_t n) {
    for (uint32_t k = 0; k < n; k++) {
        double sum = 0.0;
        for (uint32_t i = 0; i <= k; i++) sum += a[i] * b[k - i];
        out[k] = sum;
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
        taylor_exp_coeffs(rs, e, &neg, n);
        for (uint32_t i = 0; i < n; i++) den[i] = e[i];
        den[0] += 1.0;
        taylor_div_coeffs(out, one, den, n);
    } else {
        taylor_exp_coeffs(rs, e, a, n);
        for (uint32_t i = 0; i < n; i++) den[i] = e[i];
        den[0] += 1.0;
        taylor_div_coeffs(out, e, den, n);
    }
}

static VmDual* taylor_binary(VmRegionStack* rs, const VmDual* a,
                             const VmDual* b, char op) {
    uint32_t n = a->kind == VM_DUAL_KIND_TAYLOR ? a->order : 1;
    if (b->kind == VM_DUAL_KIND_TAYLOR && b->order > n) n = b->order;
    uint32_t active_epoch = taylor_epoch(a);
    if (taylor_epoch(b) > active_epoch) active_epoch = taylor_epoch(b);
    int exact = dual_exact_operand(a) && dual_exact_operand(b);
    /* Test exact denominators as exact values.  Very small nonzero rationals
     * round to 0.0, so a double comparison would silently discard exactness. */
    if (op == '/' && exact) {
        VmRational* denominator = taylor_coeff_as_exact_at(b, 0, active_epoch);
        if (!denominator || vm_rational_is_zero(denominator)) exact = 0;
    }
    VmDual* r = taylor_alloc(rs, n, exact,
                             dual_has_tangent(a) || dual_has_tangent(b) ||
                             dual_has_foreign_tangent(a, active_epoch) ||
                             dual_has_foreign_tangent(b, active_epoch));
    if (!r) return NULL;
    r->epoch = active_epoch;
    r->primal_sign = taylor_exact_primal_sign(rs, a, b, op, active_epoch);
    for (uint32_t k = 0; k <= n; k++) {
        if (op == '+') r->coeff[k] = taylor_coeff_as_double_at(a,k,active_epoch) + taylor_coeff_as_double_at(b,k,active_epoch);
        else if (op == '-') r->coeff[k] = taylor_coeff_as_double_at(a,k,active_epoch) - taylor_coeff_as_double_at(b,k,active_epoch);
        else if (op == '*') {
            double sum = 0.0;
            for (uint32_t i = 0; i <= k; i++)
                sum += taylor_coeff_as_double_at(a,i,active_epoch) * taylor_coeff_as_double_at(b,k-i,active_epoch);
            r->coeff[k] = sum;
        } else {
            double sum = taylor_coeff_as_double_at(a,k,active_epoch);
            for (uint32_t i = 1; i <= k; i++)
                sum -= taylor_coeff_as_double_at(b,i,active_epoch) * r->coeff[k-i];
            r->coeff[k] = sum / taylor_coeff_as_double_at(b,0,active_epoch);
        }
        if (r->exact_coeff) {
            VmRational* ea = taylor_coeff_as_exact_or_zero_at(rs, a,k,active_epoch);
            VmRational* eb = taylor_coeff_as_exact_or_zero_at(rs, b,k,active_epoch);
            VmRational* out = NULL;
            if (op == '+' || op == '-') out = vm_rational_op_exact(rs, ea, eb, op);
            else if (op == '*') {
                out = vm_rational_from_int(vm_active_arena(rs), 0);
                for (uint32_t i = 0; i <= k; i++) {
                    VmRational* term = vm_rational_op_exact(
                        rs, taylor_coeff_as_exact_or_zero_at(rs, a,i,active_epoch),
                        taylor_coeff_as_exact_or_zero_at(rs, b,k-i,active_epoch), '*');
                    out = vm_rational_op_exact(rs, out, term, '+');
                    if (!out) break;
                }
            } else {
                VmRational* sum = taylor_coeff_as_exact_or_zero_at(rs, a,k,active_epoch);
                for (uint32_t i = 1; sum && i <= k; i++) {
                    VmRational* term = vm_rational_op_exact(
                        rs, taylor_coeff_as_exact_or_zero_at(rs, b,i,active_epoch), r->exact_coeff[k-i], '*');
                    sum = vm_rational_op_exact(rs, sum, term, '-');
                }
                out = vm_rational_op_exact(rs, sum, taylor_coeff_as_exact_or_zero_at(rs, b,0,active_epoch), '/');
            }
            if (!out) { r->exact_coeff = NULL; exact = 0; }
            else r->exact_coeff[k] = out;
        }
    }
    if (r->tangent_coeff) {
        for (uint32_t k = 0; k <= n; k++) {
            double at = taylor_tangent_as_double_at(a, k, active_epoch);
            double bt = taylor_tangent_as_double_at(b, k, active_epoch);
            if (op == '+') r->tangent_coeff[k] = at + bt;
            else if (op == '-') r->tangent_coeff[k] = at - bt;
            else if (op == '*') {
                double sum = 0.0;
                for (uint32_t i = 0; i <= k; i++) {
                    double ati = taylor_tangent_as_double_at(a, i, active_epoch);
                    double bti = taylor_tangent_as_double_at(b, i, active_epoch);
                    uint32_t j = k - i;
                    double aj = (a->kind == VM_DUAL_KIND_TAYLOR)
                        ? taylor_coeff_as_double_at(a, j, active_epoch)
                        : (j == 0 ? a->primal : 0.0);
                    double bj = (b->kind == VM_DUAL_KIND_TAYLOR)
                        ? taylor_coeff_as_double_at(b, j, active_epoch)
                        : (j == 0 ? b->primal : 0.0);
                    sum += ati * bj + aj * bti;
                }
                r->tangent_coeff[k] = sum;
            } else {
                double sum = at;
                for (uint32_t i = 1; i <= k; i++) {
                    double bvi = taylor_coeff_as_double_at(b, i, active_epoch);
                    double bti = taylor_tangent_as_double_at(b, i, active_epoch);
                    sum -= bti * r->coeff[k-i] + bvi * r->tangent_coeff[k-i];
                }
                sum -= r->coeff[k] * bt;
                r->tangent_coeff[k] = sum / taylor_coeff_as_double_at(b, 0, active_epoch);
            }
        }
    }
    r->primal = r->coeff[0];
    r->tangent = n >= 1 ? r->coeff[1] : 0.0;
    return r;
}

static VmDual* taylor_unary(VmRegionStack* rs, const VmDual* a, int op) {
    uint32_t n = a->kind == VM_DUAL_KIND_TAYLOR ? a->order : 1;
    int exact = dual_exact_operand(a) && (op == 0 || op == 1 || op == 2);
    VmDual* r = taylor_alloc(rs, n, exact, a->tangent_coeff != NULL);
    if (!r) return NULL;
    r->epoch = taylor_epoch(a);
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
        r->coeff[0] = exp(u0);
        for (uint32_t k=1;k<=n;k++) {
            double sum=0.0;
            for (uint32_t i=1;i<=k;i++) sum += i*taylor_coeff_as_double(a,i)*r->coeff[k-i];
            r->coeff[k] = sum/k;
        }
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
    } else { /* sqrt */
        r->coeff[0] = sqrt(u0);
        for (uint32_t k=1;k<=n;k++) {
            double sum=taylor_coeff_as_double(a,k);
            for (uint32_t i=1;i<k;i++) sum -= r->coeff[i]*r->coeff[k-i];
            r->coeff[k] = sum/(2.0*r->coeff[0]);
        }
    }
    if (r->tangent_coeff) {
        double* input_tangent = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
        if (!input_tangent) return NULL;
        for (uint32_t k = 0; k <= n; k++)
            input_tangent[k] = a->tangent_coeff && k <= a->order
                ? a->tangent_coeff[k]
                : (a->kind == VM_DUAL_KIND_SCALAR && k == 0 ? a->tangent : 0.0);
        if (op == 0) {
            for (uint32_t k = 0; k <= n; k++) r->tangent_coeff[k] = -input_tangent[k];
        } else if (op == 1 || op == 2) {
            int sign = u0 > 0.0 ? 1 : (u0 < 0.0 ? -1 : a->primal_sign);
            r->tangent_coeff[0] = (op == 1 ? sign : (sign > 0 ? 1 : 0)) * input_tangent[0];
            for (uint32_t k = 1; k <= n; k++)
                r->tangent_coeff[k] = (op == 1 ? sign : (sign > 0 ? 1 : 0)) * input_tangent[k];
        } else if (op == 3) {
            taylor_mul_coeffs(r->tangent_coeff, r->coeff, input_tangent, n + 1);
        } else if (op == 4 || op == 5) {
            double* other = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
            if (!other) return NULL;
            other[0] = op == 4 ? cos(u0) : sin(u0);
            for (uint32_t k = 1; k <= n; k++) {
                double sum = 0.0;
                for (uint32_t i = 1; i <= k; i++)
                    sum += i * taylor_coeff_as_double(a, i) *
                           (op == 4 ? other[k - i] : r->coeff[k - i]);
                other[k] = (op == 4 ? -sum : sum) / k;
            }
            taylor_mul_coeffs(r->tangent_coeff, other, input_tangent, n + 1);
            if (op == 5)
                for (uint32_t k = 0; k <= n; k++) r->tangent_coeff[k] = -r->tangent_coeff[k];
        } else if (op == 6) {
            taylor_div_coeffs(r->tangent_coeff, input_tangent, a->coeff, n + 1);
        } else if (op == 7) {
            double* q = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
            if (!q) return NULL;
            taylor_div_coeffs(q, input_tangent, r->coeff, n + 1);
            for (uint32_t k = 0; k <= n; k++) r->tangent_coeff[k] = 0.5 * q[k];
        } else if (op == 8 || op == 9) {
            double* factor = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
            double* one_minus = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
            if (!factor || !one_minus) return NULL;
            for (uint32_t k = 0; k <= n; k++) one_minus[k] = (k == 0 ? 1.0 : 0.0) - r->coeff[k];
            taylor_mul_coeffs(factor, r->coeff, one_minus, n + 1);
            if (op == 9) {
                double* square = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
                if (!square) return NULL;
                taylor_mul_coeffs(square, r->coeff, r->coeff, n + 1);
                for (uint32_t k = 0; k <= n; k++) factor[k] = (k == 0 ? 1.0 : 0.0) - square[k];
            }
            taylor_mul_coeffs(r->tangent_coeff, factor, input_tangent, n + 1);
        } else if (op == 10 || op == 11 || op == 12) {
            double* ep = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
            double* em = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
            double* neg = (double*)vm_alloc(rs, (size_t)(n + 1) * sizeof(double));
            if (!ep || !em || !neg) return NULL;
            for (uint32_t k = 0; k <= n; k++) neg[k] = -taylor_coeff_as_double(a, k);
            taylor_exp_coeffs(rs, ep, a, n + 1);
            taylor_exp_coeffs(rs, em, (VmDual*)&(VmDual){
                .kind = VM_DUAL_KIND_TAYLOR, .order = n, .coeff = neg}, n + 1);
            for (uint32_t k = 0; k <= n; k++) {
                double other = op == 10 ? 0.5 * (ep[k] + em[k])
                              : op == 11 ? 0.5 * (ep[k] - em[k])
                                          : 0.0;
                r->tangent_coeff[k] = other * input_tangent[k];
            }
        }
    }
    if (r->exact_coeff) {
        for (uint32_t k=0;k<=n;k++) {
            VmRational* in = taylor_coeff_as_exact(a,k);
            VmRational* out = NULL;
            if (op == 0) out = vm_rational_negate_exact(rs, in);
            else if (op == 1 || op == 2) {
        VmRational* exact0 = taylor_coeff_as_exact(a, 0);
        int sign = exact0 ? vm_rational_sign(exact0)
                          : (u0 > 0.0 ? 1 : (u0 < 0.0 ? -1 : a->primal_sign));
                if (op == 2 && sign <= 0)
                    out = vm_rational_from_int(vm_active_arena(rs), 0);
                else if (op == 1) {
                    out = sign < 0 ? vm_rational_negate_exact(rs, in) :
                          (sign > 0 ? in : vm_rational_from_int(vm_active_arena(rs), 0));
                } else out = op == 2 ? in : vm_rational_absolute_exact(rs, in);
            }
            if (!out) { r->exact_coeff = NULL; break; }
            r->exact_coeff[k] = out;
        }
    }
    r->primal=r->coeff[0]; r->tangent=n>=1?r->coeff[1]:0.0;
    return r;
}

VmDual* vm_dual_make_taylor_ride_seed(VmRegionStack* rs,
                                      const VmDual* outer) {
    if (!rs || !outer || !vm_dual_is_taylor(outer)) return NULL;
    VmDual* d = taylor_alloc(rs, outer->order, 0, 1);
    if (!d) return NULL;
    d->epoch = outer->epoch;
    for (uint32_t k = 0; k <= outer->order; k++)
        d->coeff[k] = taylor_coeff_as_double(outer, k);
    d->tangent_coeff[0] = 1.0;
    return d;
}

VmDual* vm_dual_make_taylor_carry_seed(VmRegionStack* rs,
                                       const VmDual* outer,
                                       uint32_t order) {
    if (!rs || !outer || !vm_dual_is_taylor(outer) || outer->order != 1)
        return NULL;
    VmDual* d = taylor_alloc(rs, order, 0, 1);
    if (!d) return NULL;
    d->epoch = vm_dual_next_taylor_epoch();
    d->coeff[0] = taylor_coeff_as_double(outer, 0);
    if (order >= 1) d->coeff[1] = 1.0;
    d->tangent_coeff[0] = taylor_coeff_as_double(outer, 1);
    return d;
}

VmDual* vm_dual_taylor_promote_tangent(VmRegionStack* rs,
                                       const VmDual* result) {
    if (!rs || !result || !vm_dual_is_taylor(result) || !result->tangent_coeff)
        return NULL;
    VmDual* d = taylor_alloc(rs, result->order, 0, 0);
    if (!d) return NULL;
    d->epoch = result->epoch;
    for (uint32_t k = 0; k <= result->order; k++)
        d->coeff[k] = result->tangent_coeff[k];
    d->primal = d->coeff[0];
    d->tangent = d->order >= 1 ? d->coeff[1] : 0.0;
    return d;
}

VmDual* vm_dual_taylor_carry_result(VmRegionStack* rs,
                                    const VmDual* result,
                                    uint32_t order,
                                    uint32_t outer_epoch) {
    if (!rs || !result || !vm_dual_is_taylor(result) ||
        order > result->order || !result->tangent_coeff)
        return NULL;
    VmDual* d = taylor_alloc(rs, 1, 0, 0);
    if (!d) return NULL;
    double factorial = 1.0;
    for (uint32_t i = 2; i <= order; i++) factorial *= (double)i;
    d->epoch = outer_epoch;
    d->coeff[0] = factorial * result->coeff[order];
    d->coeff[1] = factorial * result->tangent_coeff[order];
    d->primal = d->coeff[0];
    d->tangent = d->coeff[1];
    return d;
}

static VmDual* taylor_pow_integer(VmRegionStack* rs, const VmDual* a, int64_t exponent) {
    VmDual one = {0};
    one.primal = 1.0;
    one.eprimal = vm_rational_from_int(vm_active_arena(rs), 1);
    one.etangent = vm_rational_from_int(vm_active_arena(rs), 0);
    if (!one.eprimal || !one.etangent) return NULL;
    VmDual* out = taylor_alloc(rs, a->kind == VM_DUAL_KIND_TAYLOR ? a->order : 1,
                               dual_exact_operand(a), 0);
    if (!out) return NULL;
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
        VmDual one = {0};
        one.primal = 1.0;
        one.eprimal = vm_rational_from_int(vm_active_arena(rs), 1);
        one.etangent = vm_rational_from_int(vm_active_arena(rs), 0);
        if (!one.eprimal || !one.etangent) return NULL;
        VmDual* reciprocal = taylor_binary(rs, &one, out, '/');
        if (!reciprocal) return NULL;
        out = reciprocal;
    }
    out->primal=out->coeff[0]; out->tangent=out->order>=1?out->coeff[1]:0.0;
    return out;
}

VmRational* vm_dual_taylor_exact_derivative(VmRegionStack* rs,
                                             const VmDual* d, uint32_t n) {
    VmRational* c = vm_dual_taylor_exact_coeff(d, n);
    if (!c) return NULL;
    VmRational* fact = vm_rational_from_int(vm_active_arena(rs), 1);
    if (!fact) return NULL;
    for (uint32_t i=2; i<=n; i++) {
        VmRational* q = vm_rational_from_int(vm_active_arena(rs), i);
        fact = vm_rational_op_exact(rs, fact, q, '*');
        if (!fact) return NULL;
    }
    return vm_rational_op_exact(rs, fact, c, '*');
}

/* ── Core Operations ── */

/** @brief Native call 370: `(make-dual primal tangent)`. */
VmDual* vm_dual_make(VmRegionStack* rs, double primal, double tangent) {
    return vm_dual_new(rs, primal, tangent);
}

/** @brief Native call 373: dual addition, (a+a'e)+(b+b'e) = (a+b)+(a'+b')e.
 *         Exactness-preserving: exact when both operands are. */
VmDual* vm_dual_add(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
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
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 4);
    double s = sin(a->primal);
    double c = cos(a->primal);
    return vm_dual_new(rs, s, a->tangent * c);
}

/** @brief Native call 378: dual cos, cos(a+a'e) = cos(a) - a'*sin(a)*e. */
VmDual* vm_dual_cos(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 5);
    double c = cos(a->primal);
    double s = sin(a->primal);
    return vm_dual_new(rs, c, -a->tangent * s);
}

/** @brief Native call 379: dual exp, exp(a+a'e) = exp(a) + a'*exp(a)*e. */
VmDual* vm_dual_exp(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 3);
    double ea = exp(a->primal);
    return vm_dual_new(rs, ea, a->tangent * ea);
}

/** @brief Native call 380: dual log, log(a+a'e) = log(a) + (a'/a)*e. */
VmDual* vm_dual_log(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 6);
    return vm_dual_new(rs, log(a->primal), a->tangent / a->primal);
}

/** @brief Native call 381: dual sqrt, sqrt(a+a'e) = sqrt(a) +
 *         a'/(2*sqrt(a))*e. */
VmDual* vm_dual_sqrt(VmRegionStack* rs, const VmDual* a) {
    if (vm_dual_is_taylor(a)) return taylor_unary(rs, a, 7);
    double sa = sqrt(a->primal);
    return vm_dual_new(rs, sa, a->tangent / (2.0 * sa));
}

/**
 * @brief Native call 382: dual power with constant exponent @p n,
 *        (a+a'e)^n = a^n + n*a^(n-1)*a'*e. @p n must be a plain constant,
 *        not itself a dual number; for a dual exponent use exp(n*log(a))
 *        instead.
 */
VmDual* vm_dual_pow(VmRegionStack* rs, const VmDual* a, double n) {
    if (vm_dual_is_taylor(a)) {
        if (isfinite(n) && n == floor(n))
            return taylor_pow_integer(rs, a, (int64_t)n);
        VmDual* ln = taylor_unary(rs, a, 6);
        VmDual scale = {0}; scale.primal = n;
        VmDual* product = ln ? taylor_binary(rs, &scale, ln, '*') : NULL;
        return product ? taylor_unary(rs, product, 3) : NULL;
    }
    double p = pow(a->primal, n);
    double dp = n * pow(a->primal, n - 1.0) * a->tangent;
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

/** @brief Native call 383: dual absolute value, |a+a'e| = |a| +
 *         a'*sign(a)*e. */
VmDual* vm_dual_abs(VmRegionStack* rs, const VmDual* a) {
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
    if (vm_dual_is_taylor(a)) {
        return taylor_unary(rs, a, 8);
    }
    double sig = 1.0 / (1.0 + exp(-a->primal));
    return vm_dual_new(rs, sig, a->tangent * sig * (1.0 - sig));
}

/** @brief Native call 387: dual tanh, tanh(a+a'e) = tanh(a) + a'*(1 -
 *         tanh(a)^2)*e. */
VmDual* vm_dual_tanh(VmRegionStack* rs, const VmDual* a) {
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

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_DUAL_TEST */
