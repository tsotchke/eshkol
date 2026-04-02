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
#include <math.h>
#include <stdio.h>

/* ── Allocation ── */

static VmComplex* vm_complex_new(VmRegionStack* rs, double real, double imag) {
    VmComplex* z = (VmComplex*)vm_alloc(rs, sizeof(VmComplex));
    if (!z) return NULL;
    z->real = real;
    z->imag = imag;
    return z;
}

/* ── Core Operations ── */

/* 300: make-rectangular */
static VmComplex* vm_make_rectangular(VmRegionStack* rs, double real, double imag) {
    return vm_complex_new(rs, real, imag);
}

/* 301: make-polar */
static VmComplex* vm_make_polar(VmRegionStack* rs, double mag, double angle) {
    return vm_complex_new(rs, mag * cos(angle), mag * sin(angle));
}

/* 304: magnitude — Smith's formula for overflow safety */
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

/* 305: angle */
static double vm_complex_angle(const VmComplex* z) {
    return atan2(z->imag, z->real);
}

/* 306: conjugate */
static VmComplex* vm_complex_conjugate(VmRegionStack* rs, const VmComplex* z) {
    return vm_complex_new(rs, z->real, -z->imag);
}

/* 307: add */
static VmComplex* vm_complex_add(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    return vm_complex_new(rs, a->real + b->real, a->imag + b->imag);
}

/* 308: sub */
static VmComplex* vm_complex_sub(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    return vm_complex_new(rs, a->real - b->real, a->imag - b->imag);
}

/* 309: mul */
static VmComplex* vm_complex_mul(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    return vm_complex_new(rs,
        a->real * b->real - a->imag * b->imag,
        a->real * b->imag + a->imag * b->real);
}

/* 310: div — Smith's formula for overflow safety
 * If |c| >= |d|: r = d/c, denom = c + d*r
 *   real = (a + b*r) / denom
 *   imag = (b - a*r) / denom
 * Else: r = c/d, denom = d + c*r
 *   real = (b + a*r) / denom
 *   imag = (-a + b*r) / denom  (note sign!) */
static VmComplex* vm_complex_div(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
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

/* 311: sqrt */
static VmComplex* vm_complex_sqrt(VmRegionStack* rs, const VmComplex* z) {
    double r = vm_complex_magnitude(z);
    double real = sqrt((r + z->real) / 2.0);
    double imag = sqrt((r - z->real) / 2.0);
    if (z->imag < 0) imag = -imag;
    return vm_complex_new(rs, real, imag);
}

/* 312: exp — e^(a+bi) = e^a * (cos(b) + i*sin(b)) */
static VmComplex* vm_complex_exp(VmRegionStack* rs, const VmComplex* z) {
    double ea = exp(z->real);
    return vm_complex_new(rs, ea * cos(z->imag), ea * sin(z->imag));
}

/* 313: log — log(z) = (log|z|, angle(z)) */
static VmComplex* vm_complex_log(VmRegionStack* rs, const VmComplex* z) {
    return vm_complex_new(rs, log(vm_complex_magnitude(z)), vm_complex_angle(z));
}

/* 314: sin — sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b) */
static VmComplex* vm_complex_sin(VmRegionStack* rs, const VmComplex* z) {
    return vm_complex_new(rs,
        sin(z->real) * cosh(z->imag),
        cos(z->real) * sinh(z->imag));
}

/* 315: cos — cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b) */
static VmComplex* vm_complex_cos(VmRegionStack* rs, const VmComplex* z) {
    return vm_complex_new(rs,
        cos(z->real) * cosh(z->imag),
        -sin(z->real) * sinh(z->imag));
}

/* 316: tan — sin(z)/cos(z) */
static VmComplex* vm_complex_tan(VmRegionStack* rs, const VmComplex* z) {
    VmComplex* s = vm_complex_sin(rs, z);
    VmComplex* c = vm_complex_cos(rs, z);
    if (!s || !c) return NULL;
    return vm_complex_div(rs, s, c);
}

/* 318: expt — a^b = exp(b * log(a)) */
static VmComplex* vm_complex_expt(VmRegionStack* rs, const VmComplex* a, const VmComplex* b) {
    VmComplex* log_a = vm_complex_log(rs, a);
    if (!log_a) return NULL;
    VmComplex* b_log_a = vm_complex_mul(rs, b, log_a);
    if (!b_log_a) return NULL;
    return vm_complex_exp(rs, b_log_a);
}

/* ── Self-Test ── */

#ifdef VM_COMPLEX_TEST
#include <assert.h>

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
