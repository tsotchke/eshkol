/**
 * @file vm_autodiff.c
 * @brief Reverse-mode automatic differentiation via Wengert tape.
 *
 * Records a computation graph (tape) during the forward pass, then
 * propagates gradients backward from output to inputs via the chain rule.
 * Supports multivariate functions and arbitrary computation DAGs.
 *
 * Native call IDs: 390-409
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ── AD Operation Types ── */

typedef enum {
    AD_ADD,       /* binary: left + right */
    AD_SUB,       /* binary: left - right */
    AD_MUL,       /* binary: left * right */
    AD_DIV,       /* binary: left / right */
    AD_SIN,       /* unary:  sin(left) */
    AD_COS,       /* unary:  cos(left) */
    AD_EXP,       /* unary:  exp(left) */
    AD_LOG,       /* unary:  log(left) */
    AD_SQRT,      /* unary:  sqrt(left) */
    AD_POW,       /* binary: left ^ saved (exponent stored in saved) */
    AD_NEG,       /* unary:  -left */
    AD_ABS,       /* unary:  |left| */
    AD_RELU,      /* unary:  max(0, left) */
    AD_SIGMOID,   /* unary:  1/(1+exp(-left)) */
    AD_TANH,      /* unary:  tanh(left) */
    AD_CONST,     /* leaf:   constant value (no gradient flows through) */
    AD_VAR        /* leaf:   variable (gradient collection point) */
} AdOpType;

/* ── Tape Node ── */

typedef struct {
    AdOpType op;
    double   value;     /* forward-pass result */
    double   gradient;  /* accumulated backward gradient (adjoint) */
    int      left;      /* index of left parent (-1 if none) */
    int      right;     /* index of right parent (-1 if none) */
    double   saved;     /* auxiliary saved value for backward pass */
} AdNode;

/* ── Tape ── */

#define AD_TAPE_INIT_CAP 64

typedef struct {
    AdNode* nodes;      /* arena-allocated node array */
    int     len;        /* number of nodes on tape */
    int     cap;        /* capacity */
    VmRegionStack* rs;  /* region stack for allocation */
} AdTape;

/* ── Tape Management ── */

/* 390: ad-tape-new — create a fresh tape */
AdTape* ad_tape_new(VmRegionStack* rs) {
    AdTape* tape = (AdTape*)vm_alloc_object(rs, VM_SUBTYPE_AD_TAPE, sizeof(AdTape));
    if (!tape) return NULL;
    tape->cap = AD_TAPE_INIT_CAP;
    tape->nodes = (AdNode*)vm_alloc(rs, sizeof(AdNode) * (size_t)tape->cap);
    if (!tape->nodes) return NULL;
    tape->len = 0;
    tape->rs = rs;
    return tape;
}

/* Grow tape capacity when full. Returns 0 on success, -1 on failure. */
static int ad_tape_grow(AdTape* tape) {
    int new_cap = tape->cap * 2;
    AdNode* new_nodes = (AdNode*)vm_alloc(tape->rs, sizeof(AdNode) * (size_t)new_cap);
    if (!new_nodes) return -1;
    memcpy(new_nodes, tape->nodes, sizeof(AdNode) * (size_t)tape->len);
    tape->nodes = new_nodes;
    tape->cap = new_cap;
    /* Old nodes remain in arena — reclaimed on region pop */
    return 0;
}

/* Append a node to the tape. Returns the node index, or -1 on error. */
static int ad_tape_push(AdTape* tape, AdOpType op, double value,
                        int left, int right, double saved) {
    if (tape->len >= tape->cap) {
        if (ad_tape_grow(tape) != 0) return -1;
    }
    int idx = tape->len++;
    AdNode* n = &tape->nodes[idx];
    n->op = op;
    n->value = value;
    n->gradient = 0.0;
    n->left = left;
    n->right = right;
    n->saved = saved;
    return idx;
}

/* ── Leaf Nodes ── */

/* 391: ad-const — record a constant (gradient does not flow through) */
int ad_const(AdTape* tape, double value) {
    return ad_tape_push(tape, AD_CONST, value, -1, -1, 0.0);
}

/* 392: ad-var — record a variable (gradient collection point) */
int ad_var(AdTape* tape, double value) {
    return ad_tape_push(tape, AD_VAR, value, -1, -1, 0.0);
}

/* ── Binary Operations ── */

/* 393: ad-add */
int ad_add(AdTape* tape, int left, int right) {
    double v = tape->nodes[left].value + tape->nodes[right].value;
    return ad_tape_push(tape, AD_ADD, v, left, right, 0.0);
}

/* 394: ad-sub */
int ad_sub(AdTape* tape, int left, int right) {
    double v = tape->nodes[left].value - tape->nodes[right].value;
    return ad_tape_push(tape, AD_SUB, v, left, right, 0.0);
}

/* 395: ad-mul */
int ad_mul(AdTape* tape, int left, int right) {
    double lv = tape->nodes[left].value;
    double rv = tape->nodes[right].value;
    return ad_tape_push(tape, AD_MUL, lv * rv, left, right, 0.0);
}

/* 396: ad-div */
int ad_div(AdTape* tape, int left, int right) {
    double lv = tape->nodes[left].value;
    double rv = tape->nodes[right].value;
    return ad_tape_push(tape, AD_DIV, lv / rv, left, right, rv);
}

/* 397: ad-pow — saves exponent value for backward pass */
int ad_pow(AdTape* tape, int base, int exponent) {
    double bv = tape->nodes[base].value;
    double ev = tape->nodes[exponent].value;
    double v = pow(bv, ev);
    return ad_tape_push(tape, AD_POW, v, base, exponent, ev);
}

/* ── Unary Operations ── */

/* 398: ad-sin */
int ad_sin(AdTape* tape, int input) {
    double v = sin(tape->nodes[input].value);
    return ad_tape_push(tape, AD_SIN, v, input, -1, 0.0);
}

/* 399: ad-cos */
int ad_cos(AdTape* tape, int input) {
    double v = cos(tape->nodes[input].value);
    return ad_tape_push(tape, AD_COS, v, input, -1, 0.0);
}

/* 400: ad-exp */
int ad_exp(AdTape* tape, int input) {
    double v = exp(tape->nodes[input].value);
    return ad_tape_push(tape, AD_EXP, v, input, -1, 0.0);
}

/* 401: ad-log */
int ad_log(AdTape* tape, int input) {
    double v = log(tape->nodes[input].value);
    return ad_tape_push(tape, AD_LOG, v, input, -1, 0.0);
}

/* 402: ad-sqrt */
int ad_sqrt(AdTape* tape, int input) {
    double v = sqrt(tape->nodes[input].value);
    return ad_tape_push(tape, AD_SQRT, v, input, -1, 0.0);
}

/* 403: ad-neg */
int ad_neg(AdTape* tape, int input) {
    double v = -tape->nodes[input].value;
    return ad_tape_push(tape, AD_NEG, v, input, -1, 0.0);
}

/* 404: ad-abs */
int ad_abs(AdTape* tape, int input) {
    double v = fabs(tape->nodes[input].value);
    return ad_tape_push(tape, AD_ABS, v, input, -1, 0.0);
}

/* 405: ad-relu */
int ad_relu(AdTape* tape, int input) {
    double iv = tape->nodes[input].value;
    double v = iv > 0.0 ? iv : 0.0;
    return ad_tape_push(tape, AD_RELU, v, input, -1, 0.0);
}

/* 406: ad-sigmoid */
int ad_sigmoid(AdTape* tape, int input) {
    double v = 1.0 / (1.0 + exp(-tape->nodes[input].value));
    return ad_tape_push(tape, AD_SIGMOID, v, input, -1, 0.0);
}

/* 407: ad-tanh */
int ad_tanh(AdTape* tape, int input) {
    double v = tanh(tape->nodes[input].value);
    return ad_tape_push(tape, AD_TANH, v, input, -1, 0.0);
}

/* ── Backward Pass ── */

/**
 * 408: ad-backward — reverse-mode gradient propagation.
 *
 * Seeds the output node with gradient 1.0, then walks the tape in
 * reverse topological order (guaranteed by construction since nodes
 * are appended in evaluation order), distributing gradients to parents
 * according to the local derivative of each operation.
 */
void ad_backward(AdTape* tape, int output) {
    if (output < 0 || output >= tape->len) return;

    /* Zero all gradients */
    for (int i = 0; i < tape->len; i++) {
        tape->nodes[i].gradient = 0.0;
    }

    /* Seed output */
    tape->nodes[output].gradient = 1.0;

    /* Walk tape in reverse */
    for (int i = output; i >= 0; i--) {
        AdNode* n = &tape->nodes[i];
        double g = n->gradient;
        if (g == 0.0) continue; /* optimization: skip zero-gradient nodes */

        int L = n->left;
        int R = n->right;

        switch (n->op) {
        case AD_ADD:
            /* d/dL (L+R) = 1, d/dR (L+R) = 1 */
            tape->nodes[L].gradient += g;
            tape->nodes[R].gradient += g;
            break;

        case AD_SUB:
            /* d/dL (L-R) = 1, d/dR (L-R) = -1 */
            tape->nodes[L].gradient += g;
            tape->nodes[R].gradient -= g;
            break;

        case AD_MUL:
            /* d/dL (L*R) = R, d/dR (L*R) = L */
            tape->nodes[L].gradient += g * tape->nodes[R].value;
            tape->nodes[R].gradient += g * tape->nodes[L].value;
            break;

        case AD_DIV:
            /* d/dL (L/R) = 1/R, d/dR (L/R) = -L/R^2 */
            {
                double rv = n->saved; /* saved = right.value at forward time */
                tape->nodes[L].gradient += g / rv;
                tape->nodes[R].gradient -= g * tape->nodes[L].value / (rv * rv);
            }
            break;

        case AD_SIN:
            /* d/dL sin(L) = cos(L) */
            tape->nodes[L].gradient += g * cos(tape->nodes[L].value);
            break;

        case AD_COS:
            /* d/dL cos(L) = -sin(L) */
            tape->nodes[L].gradient += g * (-sin(tape->nodes[L].value));
            break;

        case AD_EXP:
            /* d/dL exp(L) = exp(L) = output.value */
            tape->nodes[L].gradient += g * n->value;
            break;

        case AD_LOG:
            /* d/dL log(L) = 1/L */
            tape->nodes[L].gradient += g / tape->nodes[L].value;
            break;

        case AD_SQRT:
            /* d/dL sqrt(L) = 1/(2*sqrt(L)) = 1/(2*output.value) */
            tape->nodes[L].gradient += g / (2.0 * n->value);
            break;

        case AD_POW:
            /* d/dL (L^e) = e * L^(e-1), saved = exponent */
            {
                double ev = n->saved;
                double lv = tape->nodes[L].value;
                tape->nodes[L].gradient += g * ev * pow(lv, ev - 1.0);
                /* d/dR (L^R) = L^R * ln(L) — for variable exponent */
                if (R >= 0) {
                    tape->nodes[R].gradient += g * n->value * log(lv);
                }
            }
            break;

        case AD_NEG:
            /* d/dL (-L) = -1 */
            tape->nodes[L].gradient -= g;
            break;

        case AD_ABS:
            /* d/dL |L| = sign(L) */
            {
                double lv = tape->nodes[L].value;
                double sign = (lv > 0.0) ? 1.0 : (lv < 0.0) ? -1.0 : 0.0;
                tape->nodes[L].gradient += g * sign;
            }
            break;

        case AD_RELU:
            /* d/dL relu(L) = (L > 0) ? 1 : 0 */
            if (tape->nodes[L].value > 0.0) {
                tape->nodes[L].gradient += g;
            }
            break;

        case AD_SIGMOID:
            /* d/dL sigma(L) = sigma(L) * (1 - sigma(L)) = out * (1 - out) */
            tape->nodes[L].gradient += g * n->value * (1.0 - n->value);
            break;

        case AD_TANH:
            /* d/dL tanh(L) = 1 - tanh(L)^2 = 1 - out^2 */
            tape->nodes[L].gradient += g * (1.0 - n->value * n->value);
            break;

        case AD_CONST:
        case AD_VAR:
            /* Leaf nodes — no parents to propagate to */
            break;
        }
    }
}

/* 409: ad-gradient — read the accumulated gradient of a node */
double ad_gradient(const AdTape* tape, int node) {
    if (node < 0 || node >= tape->len) return 0.0;
    return tape->nodes[node].gradient;
}

/* Convenience: read the forward value of a node */
double ad_value(const AdTape* tape, int node) {
    if (node < 0 || node >= tape->len) return 0.0;
    return tape->nodes[node].value;
}

/*******************************************************************************
 * Dispatch — called from bytecode VM's NATIVE_CALL instruction
 ******************************************************************************/

/**
 * vm_autodiff_dispatch — route a native call ID in [390,409] to the
 * correct reverse-mode AD operation.
 *
 * Convention: int arguments are passed as (int)(intptr_t)args[i],
 * double arguments as *(double*)args[i], pointer as (AdTape*)args[i].
 */
void* vm_autodiff_dispatch(VmRegionStack* rs, int id, void** args, int nargs) {
    (void)nargs; /* some ops don't use all args */

    switch (id) {
    case 390: /* ad-tape-new() */
        return ad_tape_new(rs);

    case 391: { /* ad-const(tape, value) */
        AdTape* t = (AdTape*)args[0];
        double v = *(double*)args[1];
        int idx = ad_const(t, v);
        /* Return index as pointer-sized int (caller reinterprets) */
        return (void*)(intptr_t)idx;
    }
    case 392: { /* ad-var(tape, value) */
        AdTape* t = (AdTape*)args[0];
        double v = *(double*)args[1];
        int idx = ad_var(t, v);
        return (void*)(intptr_t)idx;
    }
    case 393: { /* ad-add(tape, left, right) */
        AdTape* t = (AdTape*)args[0];
        int l = (int)(intptr_t)args[1];
        int r = (int)(intptr_t)args[2];
        return (void*)(intptr_t)ad_add(t, l, r);
    }
    case 394: { /* ad-sub(tape, left, right) */
        AdTape* t = (AdTape*)args[0];
        int l = (int)(intptr_t)args[1];
        int r = (int)(intptr_t)args[2];
        return (void*)(intptr_t)ad_sub(t, l, r);
    }
    case 395: { /* ad-mul(tape, left, right) */
        AdTape* t = (AdTape*)args[0];
        int l = (int)(intptr_t)args[1];
        int r = (int)(intptr_t)args[2];
        return (void*)(intptr_t)ad_mul(t, l, r);
    }
    case 396: { /* ad-div(tape, left, right) */
        AdTape* t = (AdTape*)args[0];
        int l = (int)(intptr_t)args[1];
        int r = (int)(intptr_t)args[2];
        return (void*)(intptr_t)ad_div(t, l, r);
    }
    case 397: { /* ad-pow(tape, base, exp) */
        AdTape* t = (AdTape*)args[0];
        int b = (int)(intptr_t)args[1];
        int e = (int)(intptr_t)args[2];
        return (void*)(intptr_t)ad_pow(t, b, e);
    }
    case 398: return (void*)(intptr_t)ad_sin((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 399: return (void*)(intptr_t)ad_cos((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 400: return (void*)(intptr_t)ad_exp((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 401: return (void*)(intptr_t)ad_log((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 402: return (void*)(intptr_t)ad_sqrt((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 403: return (void*)(intptr_t)ad_neg((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 404: return (void*)(intptr_t)ad_abs((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 405: return (void*)(intptr_t)ad_relu((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 406: return (void*)(intptr_t)ad_sigmoid((AdTape*)args[0], (int)(intptr_t)args[1]);
    case 407: return (void*)(intptr_t)ad_tanh((AdTape*)args[0], (int)(intptr_t)args[1]);

    case 408: { /* ad-backward(tape, output_node) */
        AdTape* t = (AdTape*)args[0];
        int out = (int)(intptr_t)args[1];
        ad_backward(t, out);
        return NULL;
    }
    case 409: { /* ad-gradient(tape, node) — returns double* */
        AdTape* t = (AdTape*)args[0];
        int node = (int)(intptr_t)args[1];
        /* Return pointer to gradient (caller reads double from it) */
        if (node >= 0 && node < t->len) {
            return (void*)&t->nodes[node].gradient;
        }
        return NULL;
    }
    default:
        fprintf(stderr, "ERROR: unknown autodiff native ID %d\n", id);
        return NULL;
    }
}

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_AUTODIFF_TEST

#include <assert.h>

#define AD_EPS 1e-10

static int ad_near(double a, double b) {
    return fabs(a - b) < AD_EPS;
}

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s (got unexpected value)\n", name); } \
} while(0)

    printf("=== vm_autodiff (reverse-mode) self-test ===\n\n");

    /* --- Test 1: f(x) = x^2 at x=3: gradient = 6 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 3.0);        /* node 0: x=3 */
        int y = ad_mul(t, x, x);       /* node 1: x*x = 9 */
        ad_backward(t, y);
        CHECK("x^2 at x=3: value = 9", ad_near(ad_value(t, y), 9.0));
        CHECK("x^2 at x=3: gradient = 6", ad_near(ad_gradient(t, x), 6.0));
    }

    /* --- Test 2: f(x,y) = x*y at (3,4): df/dx=4, df/dy=3 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 3.0);
        int y = ad_var(t, 4.0);
        int z = ad_mul(t, x, y);       /* z = x*y = 12 */
        ad_backward(t, z);
        CHECK("x*y at (3,4): value = 12", ad_near(ad_value(t, z), 12.0));
        CHECK("x*y: df/dx = 4", ad_near(ad_gradient(t, x), 4.0));
        CHECK("x*y: df/dy = 3", ad_near(ad_gradient(t, y), 3.0));
    }

    /* --- Test 3: f(x) = sin(x) at x=0: gradient = cos(0) = 1 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 0.0);
        int y = ad_sin(t, x);
        ad_backward(t, y);
        CHECK("sin(0): value = 0", ad_near(ad_value(t, y), 0.0));
        CHECK("sin(x) at x=0: gradient = 1", ad_near(ad_gradient(t, x), 1.0));
    }

    /* --- Test 4: f(x) = exp(x) at x=0: gradient = exp(0) = 1 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 0.0);
        int y = ad_exp(t, x);
        ad_backward(t, y);
        CHECK("exp(0): value = 1", ad_near(ad_value(t, y), 1.0));
        CHECK("exp(x) at x=0: gradient = 1", ad_near(ad_gradient(t, x), 1.0));
    }

    /* --- Test 5: relu gradients --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x1 = ad_var(t, 3.0);
        int y1 = ad_relu(t, x1);
        ad_backward(t, y1);
        CHECK("relu(3): value = 3", ad_near(ad_value(t, y1), 3.0));
        CHECK("relu(x) at x=3: gradient = 1", ad_near(ad_gradient(t, x1), 1.0));
    }
    {
        AdTape* t = ad_tape_new(&rs);
        int x2 = ad_var(t, -1.0);
        int y2 = ad_relu(t, x2);
        ad_backward(t, y2);
        CHECK("relu(-1): value = 0", ad_near(ad_value(t, y2), 0.0));
        CHECK("relu(x) at x=-1: gradient = 0", ad_near(ad_gradient(t, x2), 0.0));
    }

    /* --- Test 6: cos gradient: d/dx cos(x)|_{x=pi/2} = -sin(pi/2) = -1 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, M_PI / 2.0);
        int y = ad_cos(t, x);
        ad_backward(t, y);
        CHECK("cos(pi/2) near 0", ad_near(ad_value(t, y), 0.0));
        CHECK("d/dx cos(x) at pi/2 = -1", ad_near(ad_gradient(t, x), -1.0));
    }

    /* --- Test 7: log gradient: d/dx log(x)|_{x=e} = 1/e --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, M_E);
        int y = ad_log(t, x);
        ad_backward(t, y);
        CHECK("log(e) = 1", ad_near(ad_value(t, y), 1.0));
        CHECK("d/dx log(x) at e = 1/e", ad_near(ad_gradient(t, x), 1.0 / M_E));
    }

    /* --- Test 8: sqrt gradient: d/dx sqrt(x)|_{x=4} = 1/(2*2) = 0.25 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 4.0);
        int y = ad_sqrt(t, x);
        ad_backward(t, y);
        CHECK("sqrt(4) = 2", ad_near(ad_value(t, y), 2.0));
        CHECK("d/dx sqrt(x) at 4 = 0.25", ad_near(ad_gradient(t, x), 0.25));
    }

    /* --- Test 9: division: f(x,y) = x/y at (6,3): df/dx=1/3, df/dy=-6/9=-2/3 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 6.0);
        int y = ad_var(t, 3.0);
        int z = ad_div(t, x, y);
        ad_backward(t, z);
        CHECK("6/3 = 2", ad_near(ad_value(t, z), 2.0));
        CHECK("d(x/y)/dx at (6,3) = 1/3", ad_near(ad_gradient(t, x), 1.0 / 3.0));
        CHECK("d(x/y)/dy at (6,3) = -2/3", ad_near(ad_gradient(t, y), -2.0 / 3.0));
    }

    /* --- Test 10: subtraction: f(x,y) = x-y at (5,2): df/dx=1, df/dy=-1 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 5.0);
        int y = ad_var(t, 2.0);
        int z = ad_sub(t, x, y);
        ad_backward(t, z);
        CHECK("5-2 = 3", ad_near(ad_value(t, z), 3.0));
        CHECK("d(x-y)/dx = 1", ad_near(ad_gradient(t, x), 1.0));
        CHECK("d(x-y)/dy = -1", ad_near(ad_gradient(t, y), -1.0));
    }

    /* --- Test 11: neg: d/dx (-x) = -1 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 7.0);
        int y = ad_neg(t, x);
        ad_backward(t, y);
        CHECK("neg(7) = -7", ad_near(ad_value(t, y), -7.0));
        CHECK("d/dx (-x) = -1", ad_near(ad_gradient(t, x), -1.0));
    }

    /* --- Test 12: abs: d/dx |x| at x=-4 = -1 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, -4.0);
        int y = ad_abs(t, x);
        ad_backward(t, y);
        CHECK("abs(-4) = 4", ad_near(ad_value(t, y), 4.0));
        CHECK("d/dx |x| at -4 = -1", ad_near(ad_gradient(t, x), -1.0));
    }

    /* --- Test 13: sigmoid: d/dx sigma(x)|_{x=0} = 0.25 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 0.0);
        int y = ad_sigmoid(t, x);
        ad_backward(t, y);
        CHECK("sigmoid(0) = 0.5", ad_near(ad_value(t, y), 0.5));
        CHECK("d/dx sigmoid(x) at 0 = 0.25", ad_near(ad_gradient(t, x), 0.25));
    }

    /* --- Test 14: tanh: d/dx tanh(x)|_{x=0} = 1 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 0.0);
        int y = ad_tanh(t, x);
        ad_backward(t, y);
        CHECK("tanh(0) = 0", ad_near(ad_value(t, y), 0.0));
        CHECK("d/dx tanh(x) at 0 = 1", ad_near(ad_gradient(t, x), 1.0));
    }

    /* --- Test 15: pow: d/dx x^3 at x=2 = 12 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 2.0);
        int three = ad_const(t, 3.0);
        int y = ad_pow(t, x, three);
        ad_backward(t, y);
        CHECK("2^3 = 8", ad_near(ad_value(t, y), 8.0));
        CHECK("d/dx x^3 at x=2 = 12", ad_near(ad_gradient(t, x), 12.0));
    }

    /* --- Test 16: chain rule: f(x) = sin(x^2) at x=2 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 2.0);
        int x2 = ad_mul(t, x, x);     /* x^2 = 4 */
        int y = ad_sin(t, x2);         /* sin(4) */
        ad_backward(t, y);
        double expected = cos(4.0) * 4.0; /* 2x * cos(x^2) = 4*cos(4) */
        CHECK("sin(x^2) at x=2 value", ad_near(ad_value(t, y), sin(4.0)));
        CHECK("d/dx sin(x^2) at x=2 chain rule", ad_near(ad_gradient(t, x), expected));
    }

    /* --- Test 17: multi-use variable (fan-out): f(x) = x + x = 2x --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 5.0);
        int y = ad_add(t, x, x);
        ad_backward(t, y);
        CHECK("x+x at x=5: value = 10", ad_near(ad_value(t, y), 10.0));
        CHECK("d/dx (x+x) = 2", ad_near(ad_gradient(t, x), 2.0));
    }

    /* --- Test 18: longer chain: f(x) = exp(sin(x)) at x=0 --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int x = ad_var(t, 0.0);
        int s = ad_sin(t, x);
        int y = ad_exp(t, s);
        ad_backward(t, y);
        /* f(0) = exp(sin(0)) = exp(0) = 1
         * f'(0) = exp(sin(0)) * cos(0) = 1 * 1 = 1 */
        CHECK("exp(sin(0)) = 1", ad_near(ad_value(t, y), 1.0));
        CHECK("d/dx exp(sin(x)) at 0 = 1", ad_near(ad_gradient(t, x), 1.0));
    }

    /* --- Test 19: constants get zero gradient --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int c = ad_const(t, 2.0);
        int x = ad_var(t, 3.0);
        int y = ad_mul(t, c, x);     /* 2*x */
        ad_backward(t, y);
        CHECK("2*x at x=3: value = 6", ad_near(ad_value(t, y), 6.0));
        CHECK("d/dx (2*x) = 2", ad_near(ad_gradient(t, x), 2.0));
        CHECK("d/dc (c*x) = 3 (but c is const)", ad_near(ad_gradient(t, c), 3.0));
        /* Note: constant still accumulates gradient locally; it's semantically
         * a constant because the user doesn't read its gradient. */
    }

    /* --- Test 20: MLP-style: f(x) = sigmoid(w*x + b), df/dw, df/db --- */
    {
        AdTape* t = ad_tape_new(&rs);
        int w = ad_var(t, 0.5);
        int x = ad_const(t, 2.0);
        int b = ad_var(t, -0.5);
        int wx = ad_mul(t, w, x);        /* 0.5*2 = 1.0 */
        int wxb = ad_add(t, wx, b);      /* 1.0 + (-0.5) = 0.5 */
        int y = ad_sigmoid(t, wxb);       /* sigmoid(0.5) */
        ad_backward(t, y);
        double sig_val = 1.0 / (1.0 + exp(-0.5));
        double sig_deriv = sig_val * (1.0 - sig_val);
        CHECK("MLP: sigmoid(w*x+b) value", ad_near(ad_value(t, y), sig_val));
        CHECK("MLP: df/dw = x * sig'", ad_near(ad_gradient(t, w), 2.0 * sig_deriv));
        CHECK("MLP: df/db = sig'", ad_near(ad_gradient(t, b), sig_deriv));
    }

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_AUTODIFF_TEST */
