/*
 * taylor_poc.c  --  Phase-0 proof-of-concept for Eshkol arbitrary-order AD.
 *
 * Self-contained C (no repo dependencies). Implements univariate truncated-
 * Taylor arithmetic (a "Taylor tower") via the closed recurrences documented
 * in docs/design/AD_TAYLOR_TOWER.md section 5, and verifies that the k-th
 * derivative recovered as f^(k)(x0) = k! * c[k] matches the closed-form
 * analytic derivative for k = 0..8, to relative error < 1e-12.
 *
 * A Taylor tower is the coefficient array c[0..K] of the truncated series
 *
 *     f(x0 + t) = sum_{k=0..K} c[k] * t^k      =>   f^(k)(x0) = k! * c[k].
 *
 * The differentiation variable x is seeded as the series {x0, 1, 0, ..., 0}
 * (value x0, unit first-order perturbation, nothing higher); constants seed
 * {value, 0, 0, ...}.
 *
 * Build & run:  cc -O2 -std=c11 -Wall -Wextra taylor_poc.c -lm -o taylor_poc && ./taylor_poc
 * or:           ./run.sh
 *
 * Exit status 0  iff  every check passes.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Truncation order K: series carries K+1 coefficients (indices 0..K). */
#define K 8
#define N (K + 1)

typedef double tower[N];   /* c[0..K] */

/* ---- seeding ----------------------------------------------------------- */

static void t_const(tower s, double v) {
    memset(s, 0, sizeof(tower));
    s[0] = v;
}

/* the differentiation variable: value x0, unit first-order slope */
static void t_var(tower s, double x0) {
    memset(s, 0, sizeof(tower));
    s[0] = x0;
    s[1] = 1.0;
}

/* ---- elementwise linear ops -------------------------------------------- */

static void t_add(tower s, const tower u, const tower w) {
    for (int k = 0; k < N; k++) s[k] = u[k] + w[k];
}

static void t_sub(tower s, const tower u, const tower w) {
    for (int k = 0; k < N; k++) s[k] = u[k] - w[k];
}

/* ---- multiplicative / transcendental recurrences ----------------------- */

/* s = u * w :  s_k = sum_{j=0..k} u_j * w_{k-j}   (Cauchy convolution) */
static void t_mul(tower s, const tower u, const tower w) {
    tower out;
    for (int k = 0; k < N; k++) {
        double acc = 0.0;
        for (int j = 0; j <= k; j++) acc += u[j] * w[k - j];
        out[k] = acc;
    }
    memcpy(s, out, sizeof(tower));
}

/* s = u / w :  s_k = ( u_k - sum_{j=1..k} w_j * s_{k-j} ) / w_0 */
static void t_div(tower s, const tower u, const tower w) {
    tower out;
    for (int k = 0; k < N; k++) {
        double acc = u[k];
        for (int j = 1; j <= k; j++) acc -= w[j] * out[k - j];
        out[k] = acc / w[0];
    }
    memcpy(s, out, sizeof(tower));
}

/* s = exp(u) :  s_0 = exp(u_0);  s_k = (1/k) sum_{j=1..k} j * u_j * s_{k-j} */
static void t_exp(tower s, const tower u) {
    tower out;
    out[0] = exp(u[0]);
    for (int k = 1; k < N; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++) acc += (double)j * u[j] * out[k - j];
        out[k] = acc / (double)k;
    }
    memcpy(s, out, sizeof(tower));
}

/* s = log(u) :  s_0 = log(u_0);
 * s_k = ( u_k - (1/k) sum_{j=1..k-1} j * s_j * u_{k-j} ) / u_0 */
static void t_log(tower s, const tower u) {
    tower out;
    out[0] = log(u[0]);
    for (int k = 1; k < N; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k - 1; j++) acc += (double)j * out[j] * u[k - j];
        out[k] = (u[k] - acc / (double)k) / u[0];
    }
    memcpy(s, out, sizeof(tower));
}

/* coupled:  s = sin(u), c = cos(u)
 * s_0 = sin(u_0), c_0 = cos(u_0);
 * s_k =  (1/k) sum_{j=1..k} j * u_j * c_{k-j}
 * c_k = -(1/k) sum_{j=1..k} j * u_j * s_{k-j} */
static void t_sincos(tower s, tower c, const tower u) {
    tower so, co;
    so[0] = sin(u[0]);
    co[0] = cos(u[0]);
    for (int k = 1; k < N; k++) {
        double as = 0.0, ac = 0.0;
        for (int j = 1; j <= k; j++) {
            double ju = (double)j * u[j];
            as += ju * co[k - j];
            ac += ju * so[k - j];
        }
        so[k] =  as / (double)k;
        co[k] = -ac / (double)k;
    }
    memcpy(s, so, sizeof(tower));
    memcpy(c, co, sizeof(tower));
}

/* s = u^r :  s_0 = u_0^r;
 * s_k = (1/(k*u_0)) sum_{j=1..k} (j*r - (k-j)) * u_j * s_{k-j} */
static void t_pow(tower s, const tower u, double r) {
    tower out;
    out[0] = pow(u[0], r);
    for (int k = 1; k < N; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++)
            acc += ((double)j * r - (double)(k - j)) * u[j] * out[k - j];
        out[k] = acc / ((double)k * u[0]);
    }
    memcpy(s, out, sizeof(tower));
}

/* ---- verification harness ---------------------------------------------- */

static long g_fail = 0;
static long g_pass = 0;

/* factorial 0..8 */
static double fct(int k) {
    double f = 1.0;
    for (int i = 2; i <= k; i++) f *= (double)i;
    return f;
}

/* Compare recovered derivatives f^(k) = k!*c[k] against analytic[]. */
static void check(const char *name, const tower c, const double *analytic) {
    printf("\n  %s\n", name);
    printf("    %2s  %-22s  %-22s  %-12s  %s\n",
           "k", "c[k]", "f^(k)=k!*c[k]", "analytic", "rel-err");
    for (int k = 0; k < N; k++) {
        double got = fct(k) * c[k];
        double ref = analytic[k];
        double denom = fabs(ref);
        double err = (denom > 1e-30) ? fabs(got - ref) / denom : fabs(got - ref);
        /* pass if rel-err < 1e-12, or (for ref ~ 0) abs-err < 1e-9 */
        int ok = (denom > 1e-30) ? (err < 1e-12) : (fabs(got - ref) < 1e-9);
        if (ok) g_pass++; else g_fail++;
        printf("    %2d  % -22.15g  % -22.15g  % -12.6g  %-10.3g %s\n",
               k, c[k], got, ref, err, ok ? "ok" : "FAIL");
    }
}

int main(void) {
    printf("Eshkol Taylor-tower Phase-0 POC  (K=%d, orders k=0..%d)\n", K, K);
    printf("f^(k)(x0) = k! * c[k];  pass threshold rel-err < 1e-12\n");

    /* --- f(x) = x^5 at x0 = 2.0 ------------------------------------------ */
    {
        const double x0 = 2.0, r = 5.0;
        tower x, f, f_mul;
        t_var(x, x0);
        t_pow(f, x, r);

        /* cross-check: x^5 via repeated Cauchy convolution */
        t_const(f_mul, 1.0);
        for (int i = 0; i < 5; i++) t_mul(f_mul, f_mul, x);

        double an[N];
        for (int k = 0; k < N; k++) {
            if (k > 5) { an[k] = 0.0; continue; }
            /* d^k/dx^k x^5 = 5!/(5-k)! * x0^(5-k) */
            double coef = 1.0;
            for (int i = 0; i < k; i++) coef *= (double)(5 - i);
            an[k] = coef * pow(x0, (double)(5 - k));
        }
        check("f(x) = x^5   at x0 = 2.0   (via t_pow)", f, an);
        check("f(x) = x^5   at x0 = 2.0   (via repeated t_mul, cross-check)", f_mul, an);
    }

    /* --- f(x) = sin(x) at x0 = 0.0 -------------------------------------- */
    {
        const double x0 = 0.0;
        tower x, s, c;
        t_var(x, x0);
        t_sincos(s, c, x);

        double an_sin[N], an_cos[N];
        /* At x0 = 0 the derivatives cycle exactly with period 4; use the exact
         * integer cycle (not sin(k*pi/2), which floats to ~1e-16 for pi). */
        static const double sin0[4] = { 0.0,  1.0,  0.0, -1.0 };
        static const double cos0[4] = { 1.0,  0.0, -1.0,  0.0 };
        for (int k = 0; k < N; k++) {
            an_sin[k] = sin0[k % 4];
            an_cos[k] = cos0[k % 4];
        }
        check("f(x) = sin(x) at x0 = 0.0", s, an_sin);
        check("f(x) = cos(x) at x0 = 0.0   (coupled companion)", c, an_cos);
    }

    /* --- f(x) = exp(x) at x0 = 0.5 -------------------------------------- */
    {
        const double x0 = 0.5;
        tower x, f;
        t_var(x, x0);
        t_exp(f, x);
        double an[N];
        for (int k = 0; k < N; k++) an[k] = exp(x0);  /* exp^(k) = exp */
        check("f(x) = exp(x) at x0 = 0.5", f, an);
    }

    /* --- f(x) = 1/(1-x) at x0 = 0.5 ------------------------------------- */
    {
        const double x0 = 0.5;
        tower x, one, w, f;
        t_var(x, x0);
        t_const(one, 1.0);
        t_sub(w, one, x);          /* w = 1 - x */
        t_div(f, one, w);          /* f = 1 / (1 - x) */
        double an[N];
        /* d^k/dx^k 1/(1-x) = k! / (1-x)^(k+1) */
        for (int k = 0; k < N; k++)
            an[k] = fct(k) / pow(1.0 - x0, (double)(k + 1));
        check("f(x) = 1/(1-x) at x0 = 0.5", f, an);
    }

    /* --- f(x) = log(1+x) at x0 = 0.0 ------------------------------------ */
    {
        const double x0 = 0.0;
        tower x, one, u, f;
        t_var(x, x0);
        t_const(one, 1.0);
        t_add(u, one, x);          /* u = 1 + x */
        t_log(f, u);               /* f = log(1 + x) */
        double an[N];
        an[0] = log(1.0 + x0);
        for (int k = 1; k < N; k++) {
            /* d^k/dx^k log(1+x) = (-1)^(k-1) (k-1)! / (1+x)^k */
            double sgn = (k % 2 == 1) ? 1.0 : -1.0;
            an[k] = sgn * fct(k - 1) / pow(1.0 + x0, (double)k);
        }
        check("f(x) = log(1+x) at x0 = 0.0", f, an);
    }

    printf("\n================================================================\n");
    printf("checks passed: %ld   failed: %ld\n", g_pass, g_fail);
    if (g_fail == 0) {
        printf("RESULT: PASS  (all derivatives k=0..%d exact to rel-err < 1e-12)\n", K);
        return 0;
    }
    printf("RESULT: FAIL\n");
    return 1;
}
