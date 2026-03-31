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

/* ── Allocation ── */

static VmDual* vm_dual_new(VmRegionStack* rs, double primal, double tangent) {
    VmDual* d = (VmDual*)vm_alloc_object(rs, VM_SUBTYPE_DUAL, sizeof(VmDual));
    if (!d) return NULL;
    d->primal = primal;
    d->tangent = tangent;
    return d;
}

/* ── Core Operations ── */

/* 370: make-dual */
VmDual* vm_dual_make(VmRegionStack* rs, double primal, double tangent) {
    return vm_dual_new(rs, primal, tangent);
}

/* 371: dual-primal — extract primal component */
double vm_dual_primal(const VmDual* d) {
    return d->primal;
}

/* 372: dual-tangent — extract tangent (derivative) component */
double vm_dual_tangent(const VmDual* d) {
    return d->tangent;
}

/* 373: dual-add — (a + a'e) + (b + b'e) = (a+b) + (a'+b')e */
VmDual* vm_dual_add(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    return vm_dual_new(rs, a->primal + b->primal, a->tangent + b->tangent);
}

/* 374: dual-sub — (a + a'e) - (b + b'e) = (a-b) + (a'-b')e */
VmDual* vm_dual_sub(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    return vm_dual_new(rs, a->primal - b->primal, a->tangent - b->tangent);
}

/* 375: dual-mul — (a + a'e)(b + b'e) = ab + (a'b + ab')e */
VmDual* vm_dual_mul(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    return vm_dual_new(rs,
        a->primal * b->primal,
        a->tangent * b->primal + a->primal * b->tangent);
}

/* 376: dual-div — (a + a'e)/(b + b'e) = a/b + (a'b - ab')/(b^2) e */
VmDual* vm_dual_div(VmRegionStack* rs, const VmDual* a, const VmDual* b) {
    double b2 = b->primal * b->primal;
    return vm_dual_new(rs,
        a->primal / b->primal,
        (a->tangent * b->primal - a->primal * b->tangent) / b2);
}

/* 377: dual-sin — sin(a + a'e) = sin(a) + a'*cos(a)*e */
VmDual* vm_dual_sin(VmRegionStack* rs, const VmDual* a) {
    double s = sin(a->primal);
    double c = cos(a->primal);
    return vm_dual_new(rs, s, a->tangent * c);
}

/* 378: dual-cos — cos(a + a'e) = cos(a) - a'*sin(a)*e */
VmDual* vm_dual_cos(VmRegionStack* rs, const VmDual* a) {
    double c = cos(a->primal);
    double s = sin(a->primal);
    return vm_dual_new(rs, c, -a->tangent * s);
}

/* 379: dual-exp — exp(a + a'e) = exp(a) + a'*exp(a)*e */
VmDual* vm_dual_exp(VmRegionStack* rs, const VmDual* a) {
    double ea = exp(a->primal);
    return vm_dual_new(rs, ea, a->tangent * ea);
}

/* 380: dual-log — log(a + a'e) = log(a) + (a'/a)*e */
VmDual* vm_dual_log(VmRegionStack* rs, const VmDual* a) {
    return vm_dual_new(rs, log(a->primal), a->tangent / a->primal);
}

/* 381: dual-sqrt — sqrt(a + a'e) = sqrt(a) + a'/(2*sqrt(a))*e */
VmDual* vm_dual_sqrt(VmRegionStack* rs, const VmDual* a) {
    double sa = sqrt(a->primal);
    return vm_dual_new(rs, sa, a->tangent / (2.0 * sa));
}

/* 382: dual-pow — (a + a'e)^n = a^n + n*a^(n-1)*a'*e
 * n is a constant (not a dual). For dual exponent, use exp(n*log(a)). */
VmDual* vm_dual_pow(VmRegionStack* rs, const VmDual* a, double n) {
    double p = pow(a->primal, n);
    double dp = n * pow(a->primal, n - 1.0) * a->tangent;
    return vm_dual_new(rs, p, dp);
}

/* 383: dual-abs — |a + a'e| = |a| + a'*sign(a)*e */
VmDual* vm_dual_abs(VmRegionStack* rs, const VmDual* a) {
    double sign;
    if (a->primal > 0.0) sign = 1.0;
    else if (a->primal < 0.0) sign = -1.0;
    else sign = 0.0;
    return vm_dual_new(rs, fabs(a->primal), a->tangent * sign);
}

/* 384: dual-neg — -(a + a'e) = -a + (-a')e */
VmDual* vm_dual_neg(VmRegionStack* rs, const VmDual* a) {
    return vm_dual_new(rs, -a->primal, -a->tangent);
}

/* 385: dual-relu — relu(a + a'e) = max(0,a) + (a>0 ? a' : 0)*e */
VmDual* vm_dual_relu(VmRegionStack* rs, const VmDual* a) {
    if (a->primal > 0.0)
        return vm_dual_new(rs, a->primal, a->tangent);
    else
        return vm_dual_new(rs, 0.0, 0.0);
}

/* 386: dual-sigmoid — sigma(a + a'e) = sigma(a) + a'*sigma(a)*(1-sigma(a))*e */
VmDual* vm_dual_sigmoid(VmRegionStack* rs, const VmDual* a) {
    double sig = 1.0 / (1.0 + exp(-a->primal));
    return vm_dual_new(rs, sig, a->tangent * sig * (1.0 - sig));
}

/* 387: dual-tanh — tanh(a + a'e) = tanh(a) + a'*(1 - tanh(a)^2)*e */
VmDual* vm_dual_tanh(VmRegionStack* rs, const VmDual* a) {
    double th = tanh(a->primal);
    return vm_dual_new(rs, th, a->tangent * (1.0 - th * th));
}

/* 388: dual-from-double — promote a scalar to a dual with zero tangent */
VmDual* vm_dual_from_double(VmRegionStack* rs, double x) {
    return vm_dual_new(rs, x, 0.0);
}

/* 389: dual-scale — scalar * dual: c*(a + a'e) = c*a + c*a'*e */
VmDual* vm_dual_scale(VmRegionStack* rs, double c, const VmDual* a) {
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

static int dual_near(double a, double b) {
    return fabs(a - b) < DUAL_EPS;
}

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
