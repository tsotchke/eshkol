#include "ad.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int AD_ORDER = 1;

Value *ad_promote(Value *v) {
    if (v->type == V_TOWER) {
        if (v->as.tower.order == AD_ORDER) { v_incref(v); return v; }
        Value *t = v_tower(AD_ORDER, NULL);
        int n = v->as.tower.order < AD_ORDER ? v->as.tower.order : AD_ORDER;
        for (int i = 0; i <= n; i++) t->as.tower.coeff[i] = v->as.tower.coeff[i];
        return t;
    }
    /* number -> constant tower */
    Value *t = v_tower(AD_ORDER, NULL);
    t->as.tower.coeff[0] = v->as.num;
    return t;
}

int ad_is_tower(Value *v) { return v->type == V_TOWER; }

static double *dup_coeff(Value *t) {
    double *c = malloc(sizeof(double) * (AD_ORDER + 1));
    if (t->type == V_TOWER) {
        int n = t->as.tower.order < AD_ORDER ? t->as.tower.order : AD_ORDER;
        for (int i = 0; i <= AD_ORDER; i++) c[i] = (i <= n) ? t->as.tower.coeff[i] : 0.0;
    } else {
        c[0] = t->as.num;
        for (int i = 1; i <= AD_ORDER; i++) c[i] = 0.0;
    }
    return c;
}

Value *ad_add(Value *a, Value *b) {
    Value *ta = ad_promote(a), *tb = ad_promote(b);
    Value *r = v_tower(AD_ORDER, NULL);
    for (int i = 0; i <= AD_ORDER; i++)
        r->as.tower.coeff[i] = ta->as.tower.coeff[i] + tb->as.tower.coeff[i];
    v_decref(ta); v_decref(tb);
    return r;
}
Value *ad_sub(Value *a, Value *b) {
    Value *ta = ad_promote(a), *tb = ad_promote(b);
    Value *r = v_tower(AD_ORDER, NULL);
    for (int i = 0; i <= AD_ORDER; i++)
        r->as.tower.coeff[i] = ta->as.tower.coeff[i] - tb->as.tower.coeff[i];
    v_decref(ta); v_decref(tb);
    return r;
}
Value *ad_neg(Value *a) {
    Value *ta = ad_promote(a);
    Value *r = v_tower(AD_ORDER, NULL);
    for (int i = 0; i <= AD_ORDER; i++) r->as.tower.coeff[i] = -ta->as.tower.coeff[i];
    v_decref(ta);
    return r;
}
Value *ad_mul(Value *a, Value *b) {
    Value *ta = ad_promote(a), *tb = ad_promote(b);
    Value *r = v_tower(AD_ORDER, NULL);
    for (int n = 0; n <= AD_ORDER; n++) {
        double s = 0.0;
        for (int i = 0; i <= n; i++)
            s += ta->as.tower.coeff[i] * tb->as.tower.coeff[n - i];
        r->as.tower.coeff[n] = s;
    }
    v_decref(ta); v_decref(tb);
    return r;
}
Value *ad_div(Value *a, Value *b) {
    Value *ta = ad_promote(a), *tb = ad_promote(b);
    Value *r = v_tower(AD_ORDER, NULL);
    double b0 = tb->as.tower.coeff[0];
    for (int n = 0; n <= AD_ORDER; n++) {
        double s = ta->as.tower.coeff[n];
        for (int i = 1; i <= n; i++)
            s -= r->as.tower.coeff[n - i] * tb->as.tower.coeff[i];
        r->as.tower.coeff[n] = s / b0;
    }
    v_decref(ta); v_decref(tb);
    return r;
}

/* exp: e[n] = (1/n) sum_{i=1..n} i*a[i]*e[n-i], e[0]=exp(a0) */
Value *ad_exp(Value *a) {
    Value *ta = ad_promote(a);
    Value *r = v_tower(AD_ORDER, NULL);
    r->as.tower.coeff[0] = exp(ta->as.tower.coeff[0]);
    for (int n = 1; n <= AD_ORDER; n++) {
        double s = 0.0;
        for (int i = 1; i <= n; i++)
            s += i * ta->as.tower.coeff[i] * r->as.tower.coeff[n - i];
        r->as.tower.coeff[n] = s / n;
    }
    v_decref(ta);
    return r;
}

/* sin & cos share the recurrence:
 *   s[0]=sin(a0), c[0]=cos(a0)
 *   s[n]=(1/n) sum_{i=1..n} i*a[i]*c[n-i]
 *   c[n]=(1/n) sum_{i=1..n} i*a[i]*(-s[n-i]) */
static void sincos_tower(Value *a, Value **out_s, Value **out_c) {
    Value *ta = ad_promote(a);
    Value *s = v_tower(AD_ORDER, NULL);
    Value *c = v_tower(AD_ORDER, NULL);
    double a0 = ta->as.tower.coeff[0];
    s->as.tower.coeff[0] = sin(a0);
    c->as.tower.coeff[0] = cos(a0);
    for (int n = 1; n <= AD_ORDER; n++) {
        double ss = 0.0, sc = 0.0;
        for (int i = 1; i <= n; i++) {
            ss += i * ta->as.tower.coeff[i] * c->as.tower.coeff[n - i];
            sc += i * ta->as.tower.coeff[i] * (-s->as.tower.coeff[n - i]);
        }
        s->as.tower.coeff[n] = ss / n;
        c->as.tower.coeff[n] = sc / n;
    }
    v_decref(ta);
    *out_s = s; *out_c = c;
}
Value *ad_sin(Value *a) { Value *s, *c; sincos_tower(a, &s, &c); v_decref(c); return s; }
Value *ad_cos(Value *a) { Value *s, *c; sincos_tower(a, &s, &c); v_decref(s); return c; }

/* log: l[0]=log(a0), l[n]=(1/n) sum_{i=1..n} i*a[i]*l[n-i] */
Value *ad_log(Value *a) {
    Value *ta = ad_promote(a);
    Value *r = v_tower(AD_ORDER, NULL);
    r->as.tower.coeff[0] = log(ta->as.tower.coeff[0]);
    for (int n = 1; n <= AD_ORDER; n++) {
        double s = 0.0;
        for (int i = 1; i <= n; i++)
            s += i * ta->as.tower.coeff[i] * r->as.tower.coeff[n - i];
        r->as.tower.coeff[n] = s / n;
    }
    v_decref(ta);
    return r;
}

/* sqrt: s^2=a => 2 s s' = a'. s[0]=sqrt(a0);
 *   for n>=1: 2 n s[n] = a[n] - sum_{i=1}^{n-1} 2 i s[i] s[n-i] */
Value *ad_sqrt(Value *a) {
    Value *ta = ad_promote(a);
    Value *r = v_tower(AD_ORDER, NULL);
    r->as.tower.coeff[0] = sqrt(ta->as.tower.coeff[0]);
    for (int n = 1; n <= AD_ORDER; n++) {
        double s = ta->as.tower.coeff[n];
        for (int i = 1; i < n; i++)
            s -= 2.0 * i * r->as.tower.coeff[i] * r->as.tower.coeff[n - i];
        r->as.tower.coeff[n] = s / (2.0 * n);
    }
    v_decref(ta);
    return r;
}

/* a^c with constant exponent c (generalized binomial) */
Value *ad_pow_const(Value *a, double c) {
    Value *ta = ad_promote(a);
    double a0 = ta->as.tower.coeff[0];
    /* Integer exponent: build by repeated tower multiplication (exact). */
    if (fabs(c - round(c)) < 1e-12 && c >= 0.0) {
        long k = (long)round(c);
        Value *r = v_tower(AD_ORDER, NULL);
        r->as.tower.coeff[0] = 1.0;
        Value *base = ad_promote(a);
        for (long i = 0; i < k; i++) {
            Value *nr = ad_mul(r, base);
            v_decref(r); r = nr;
        }
        v_decref(base); v_decref(ta);
        return r;
    }
    /* Non-integer exponent: f(x)=x^c, f'(x)=c*x^{c-1}. Use the chain rule
     * via the identity (x^c)' = c * x^{c-1} * x', computed tower-wise. */
    Value *r = v_tower(AD_ORDER, NULL);
    r->as.tower.coeff[0] = pow(a0, c);
    for (int n = 1; n <= AD_ORDER; n++) {
        double s = 0.0;
        for (int i = 1; i <= n; i++) {
            double binom = c - (i - 1);   /* falling factor */
            s += i * binom * ta->as.tower.coeff[i] * r->as.tower.coeff[n - i];
        }
        r->as.tower.coeff[n] = s / (n * a0);
    }
    v_decref(ta);
    return r;
}

double ad_coeff(Value *v, int k) {
    if (v->type == V_TOWER) {
        if (k <= v->as.tower.order && k >= 0) return v->as.tower.coeff[k];
        return 0.0;
    }
    return (k == 0) ? v->as.num : 0.0;
}

Value *ad_seed(double x0, int seed) {
    Value *t = v_tower(AD_ORDER, NULL);
    t->as.tower.coeff[0] = x0;
    if (seed >= 0 && seed <= AD_ORDER) t->as.tower.coeff[seed] = 1.0;
    return t;
}
