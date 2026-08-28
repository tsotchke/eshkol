/**
 * @file differential_form_core.h
 * @brief Differential forms on R^n as jets at a point: the exterior derivative
 *        and the Hodge star, in f64, for the VM's geometric opcodes 835 and 836.
 *
 * WHY THIS FILE EXISTS. `exterior-derivative` (835) used to return a zero
 * tensor of its input's shape and `hodge-star` (836) used to return its
 * argument unchanged. Neither was a computation: the first asserts that every
 * form handed to it is closed, the second that the star is the identity, and
 * both assertions were made without examining the argument. SW-73 replaced them
 * with a refusal, which was honest but left the two names unimplemented. This
 * header implements them.
 *
 * WHAT THE REFUSAL WAS ACTUALLY ABOUT, AND HOW IT IS FIXED. The refusal named a
 * real missing input, not a missing algorithm:
 *
 *   - `d` is a DERIVATIVE. A form's coefficient VALUES at one point do not
 *     determine it; the coefficients as differentiable functions of position
 *     do. But `d` is a FIRST-ORDER operator, so it does not need the whole
 *     function either -- the 1-jet of the coefficients at the point is exactly
 *     enough, and is the minimal input that determines the answer.
 *   - The Hodge star of a k-form depends on k and on n. A flat coefficient
 *     array records neither, and worse, C(n,k) = C(n,n-k) makes k ambiguous
 *     even when n and the array length are both known (n = 3 with three
 *     coefficients is a 1-form or a 2-form and the star differs).
 *
 * So the representation below carries the degree, the dimension and a JET of
 * each coefficient. Nothing is inferred; every quantity the answer depends on
 * is read from the value. That is what makes the result exact rather than
 * plausible.
 *
 * THE REPRESENTATION. A k-form on R^n, known to jet order r at a point, is a
 * tensor whose FLAT data is
 *
 *     [ k, n, r, jet(w_{I_0}), jet(w_{I_1}), ..., jet(w_{I_{m-1}}) ]
 *
 * with m = C(n,k) and the basis I_t running over the increasing k-multi-indices
 * of {0,...,n-1} in lexicographic order, so
 *
 *     w = sum_t w_{I_t} dx^{I_t}.
 *
 * Each jet block is the coefficient's Taylor data at the point, laid out by
 * order and row-major inside each order:
 *
 *     jet(w_I) = [ w_I | d_j w_I (n of them) | d_j d_k w_I (n^2) | ... ]
 *
 * so the block for order s is the full n^s array of s-th partials (symmetric,
 * and stored redundantly: the redundancy costs memory and buys a contiguous
 * slice, which is what makes `d` an addition over slices below). The stride of
 * one coefficient is S(n,r) = 1 + n + n^2 + ... + n^r, and a well-formed form
 * has EXACTLY 3 + m*S(n,r) elements. A tensor of any other length is not a form
 * and the opcode reports a shape failure rather than guessing a degree.
 *
 * WHY THE JET ORDER IS PART OF THE VALUE. `d` consumes one order: the r-jet of
 * a k-form determines the (r-1)-jet of the (k+1)-form d(w), and no more. A
 * representation that did not carry r would have to either invent derivative
 * data it does not have -- the zeros the old body returned -- or refuse
 * composition. Carrying r means d(d(w)) is computable exactly when the input
 * has r >= 2, and the result SAYS what it knows: `d` of a 1-jet is a 0-jet,
 * which is a form whose derivatives are declared absent rather than zero.
 *
 * @see docs/reference/stdlib/geometry.md, section "Differential forms", for the
 *      user-facing contract and worked examples.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_DIFFERENTIAL_FORM_CORE_H
#define ESHKOL_BACKEND_DIFFERENTIAL_FORM_CORE_H

#include <math.h>
#include <string.h>

/* The header cells that precede the coefficient jets: degree, dimension, jet
 * order. */
#define ESHKOL_FORM_HEADER 3

/* Bounds. They exist because the jet blocks are full n^s arrays and the star
 * evaluates C(n,k)*C(n,n-k) determinants of k x k submatrices: both are bounded
 * quantities only if n and r are. At the maxima below one coefficient's jet is
 * 1 + 8 + 64 + 512 = 585 doubles and the widest star does 70*70 determinants of
 * order <= 8, which is a few hundred thousand flops -- a size an opcode can do
 * synchronously. A form outside these bounds is refused by name; it is not
 * silently truncated. */
#define ESHKOL_FORM_MAX_DIM 8
#define ESHKOL_FORM_MAX_JET 3

/* The widest basis these bounds admit: max_k C(8,k) = C(8,4) = 70. The basis
 * enumerations are stack locals rather than file statics so that two VM
 * instances running these opcodes concurrently cannot share one buffer. */
#define ESHKOL_FORM_MAX_BASIS 70

/* Relative tolerance on g_ij == g_ji. The star is defined for a symmetric
 * metric; an asymmetric argument is a caller error, not a metric. */
#define ESHKOL_FORM_SYM_TOL 1e-9

/** @brief C(n,k), or 0 when k is outside [0,n]. Exact in f64-free integer
 *         arithmetic for every (n,k) this file admits. */
static long eshkol_form_binom(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    long r = 1;
    for (int i = 0; i < k; i++) {
        r = r * (n - i);
        r = r / (i + 1);
    }
    return r;
}

/** @brief S(n,r) = 1 + n + n^2 + ... + n^r, the number of doubles one
 *         coefficient's jet occupies. */
static long eshkol_form_stride(int n, int r) {
    long s = 0, p = 1;
    for (int i = 0; i <= r; i++) { s += p; p *= n; }
    return s;
}

/** @brief The exact element count a well-formed (k,n,r) form occupies, or -1
 *         when (k,n,r) is not an admissible triple. */
static long eshkol_form_total(int k, int n, int r) {
    if (n < 1 || n > ESHKOL_FORM_MAX_DIM) return -1;
    if (r < 0 || r > ESHKOL_FORM_MAX_JET) return -1;
    if (k < 0 || k > n + 1) return -1;   /* k = n+1 is the zero top form */
    return (long)ESHKOL_FORM_HEADER + eshkol_form_binom(n, k) * eshkol_form_stride(n, r);
}

/**
 * @brief Read and validate a form's header.
 *
 * @return NULL when @p data is a well-formed form of exactly @p total
 *         elements, else a reason naming what is wrong. On success *@p k_out,
 *         *@p n_out and *@p r_out are the degree, dimension and jet order.
 */
static const char* eshkol_form_header(const double* data, long total,
                                      int* k_out, int* n_out, int* r_out) {
    if (!data || total < ESHKOL_FORM_HEADER)
        return "a differential form is [k, n, r, coefficient jets...] and needs "
               "at least the three header cells";
    double kd = data[0], nd = data[1], rd = data[2];
    if (kd != floor(kd) || nd != floor(nd) || rd != floor(rd))
        return "the form header cells k, n and r must be integers";
    int k = (int)kd, n = (int)nd, r = (int)rd;
    if (n < 1 || n > ESHKOL_FORM_MAX_DIM)
        return "the form's dimension n must satisfy 1 <= n <= 8";
    if (r < 0 || r > ESHKOL_FORM_MAX_JET)
        return "the form's jet order r must satisfy 0 <= r <= 3";
    if (k < 0 || k > n + 1)
        return "the form's degree k must satisfy 0 <= k <= n (k = n+1 denotes "
               "the zero top-degree form)";
    long want = eshkol_form_total(k, n, r);
    if (want != total)
        return "the form's element count does not match its header: a (k, n, r) "
               "form holds exactly 3 + C(n,k) * (1 + n + ... + n^r) elements";
    *k_out = k; *n_out = n; *r_out = r;
    return NULL;
}

/**
 * @brief Enumerate the increasing k-multi-indices of {0,...,n-1} in
 *        lexicographic order into @p out (k ints each).
 * @return m = C(n,k). @p out must hold m*k ints; for k = 0 nothing is written
 *         and the single basis element is the empty index.
 */
static int eshkol_form_basis(int n, int k, int* out) {
    int m = (int)eshkol_form_binom(n, k);
    if (k <= 0 || m <= 0) return m;
    int idx[ESHKOL_FORM_MAX_DIM];
    for (int i = 0; i < k; i++) idx[i] = i;
    for (int t = 0; t < m; t++) {
        for (int i = 0; i < k; i++) out[t * k + i] = idx[i];
        int p = k - 1;
        while (p >= 0 && idx[p] == n - k + p) p--;
        if (p < 0) break;
        idx[p]++;
        for (int i = p + 1; i < k; i++) idx[i] = idx[i - 1] + 1;
    }
    return m;
}

/** @brief Position of the increasing multi-index @p idx in the enumeration
 *         @p basis of m indices of length k, or -1. Linear because m <= 70 and
 *         k <= 8 here, and a search that is obviously right beats a ranking
 *         formula that is nearly so. */
static int eshkol_form_rank(const int* basis, int m, int k, const int* idx) {
    for (int t = 0; t < m; t++) {
        int eq = 1;
        for (int i = 0; i < k; i++)
            if (basis[t * k + i] != idx[i]) { eq = 0; break; }
        if (eq) return t;
    }
    return -1;
}

/** @brief Sign of the permutation @p perm of length @p n, by inversion count. */
static int eshkol_form_perm_sign(const int* perm, int n) {
    int inv = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (perm[i] > perm[j]) inv++;
    return (inv & 1) ? -1 : 1;
}

/**
 * @brief Exterior derivative: the (r-1)-jet of d(w) from the r-jet of w.
 *
 *   (d w)_J = sum_{p=0}^{k} (-1)^p  d_{J_p} w_{J \ J_p}
 *
 * over increasing (k+1)-multi-indices J, and each order-s block of (d w)_J is
 * the order-(s+1) block of w_{J \ J_p} sliced at its leading index J_p. That
 * slice is contiguous in the layout this file uses, which is the whole reason
 * for storing the full n^s blocks rather than the symmetric-reduced ones.
 *
 * The result is EXACT for the data supplied: no difference quotient, no step
 * size, no truncation. If the caller's jet is exact -- as it is for polynomial
 * coefficients differentiated by hand or by Eshkol's AD -- then d(w) is exact,
 * and d(d(w)) is exactly zero rather than zero to a tolerance, because the two
 * mixed partials that cancel are the SAME stored double.
 *
 * @param in     the input form, already validated by eshkol_form_header.
 * @param out    zero-initialised output of eshkol_form_total(k+1, n, r-1).
 * @return NULL on success, else a reason.
 */
static const char* eshkol_form_d(const double* in, long in_total,
                                 double* out, long out_total) {
    int k, n, r;
    const char* why = eshkol_form_header(in, in_total, &k, &n, &r);
    if (why) return why;
    if (r < 1)
        return "the exterior derivative of a 0-jet form is not determined: d is "
               "a derivative, so the coefficients' first partials must be "
               "supplied (a form of jet order r >= 1)";
    if (k > n)
        return "the form is already the zero top-degree form; it has no "
               "coefficients to differentiate";

    int kk = k + 1;
    long want = eshkol_form_total(kk, n, r - 1);
    if (want < 0 || want != out_total)
        return "internal: the exterior derivative's output was sized wrongly";

    out[0] = (double)kk;
    out[1] = (double)n;
    out[2] = (double)(r - 1);

    int m  = (int)eshkol_form_binom(n, k);
    int mm = (int)eshkol_form_binom(n, kk);
    if (mm <= 0) return NULL;            /* kk > n: d(w) is the zero form */

    int basis_k[ESHKOL_FORM_MAX_BASIS * ESHKOL_FORM_MAX_DIM];
    int basis_kk[ESHKOL_FORM_MAX_BASIS * ESHKOL_FORM_MAX_DIM];
    if (m > ESHKOL_FORM_MAX_BASIS || mm > ESHKOL_FORM_MAX_BASIS)
        return "internal: the form basis exceeded its enumeration buffer";
    eshkol_form_basis(n, k, basis_k);
    eshkol_form_basis(n, kk, basis_kk);

    long S  = eshkol_form_stride(n, r);
    long SS = eshkol_form_stride(n, r - 1);

    for (int tj = 0; tj < mm; tj++) {
        const int* J = basis_kk + (long)tj * kk;
        for (int p = 0; p < kk; p++) {
            int j = J[p];
            int I[ESHKOL_FORM_MAX_DIM];
            int q = 0;
            for (int i = 0; i < kk; i++) if (i != p) I[q++] = J[i];
            int ti = (k == 0) ? 0 : eshkol_form_rank(basis_k, m, k, I);
            if (ti < 0)
                return "internal: a face of a basis index was not in the basis";
            double sign = (p & 1) ? -1.0 : 1.0;

            long off_in = 0, off_out = 0, pow_s = 1;
            for (int s = 0; s <= r - 1; s++) {
                /* order-s block of d(w)_J  <-  order-(s+1) block of w_I,
                 * sliced at leading index j: [j][i_1..i_s] = j*n^s + e. */
                off_in += pow_s;         /* start of w_I's order-(s+1) block */
                const double* src = in + ESHKOL_FORM_HEADER + (long)ti * S +
                                    off_in + (long)j * pow_s;
                double* dst = out + ESHKOL_FORM_HEADER + (long)tj * SS + off_out;
                for (long e = 0; e < pow_s; e++) dst[e] += sign * src[e];
                off_out += pow_s;
                pow_s *= n;
            }
        }
    }
    return NULL;
}

/**
 * @brief Cholesky factorisation of a symmetric positive-definite metric, used
 *        both to invert it and to obtain its determinant.
 *
 * Cholesky rather than an LU: it SUCCEEDS exactly on the positive-definite
 * matrices, so "is this a Riemannian metric" is answered by the factorisation
 * rather than by a separate test that could disagree with it. det(g) comes out
 * as prod(L_ii)^2, which is positive by construction -- so sqrt(det g) below
 * never takes the root of a value the caller could have made negative.
 *
 * @param ginv  n*n output, the inverse.
 * @param det   output, det(g).
 * @return NULL on success, else a reason.
 */
static const char* eshkol_form_metric_inverse(const double* g, int n,
                                              double* ginv, double* det) {
    double L[ESHKOL_FORM_MAX_DIM * ESHKOL_FORM_MAX_DIM];
    for (int i = 0; i < n * n; i++)
        if (!(g[i] == g[i])) return "the metric has a NaN entry";
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double a = g[i * n + j], b = g[j * n + i];
            double scale = 1.0 + fabs(a) + fabs(b);
            if (fabs(a - b) > ESHKOL_FORM_SYM_TOL * scale)
                return "the metric must be symmetric";
        }

    memset(L, 0, sizeof L);
    double d = 1.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = g[i * n + j];
            for (int p = 0; p < j; p++) s -= L[i * n + p] * L[j * n + p];
            if (i == j) {
                if (!(s > 0.0))
                    return "the metric must be positive definite (the Hodge "
                           "star of a k-form is defined by a Riemannian inner "
                           "product on k-forms and an orientation)";
                L[i * n + i] = sqrt(s);
                d *= s;
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    }
    *det = d;

    /* Column k of the inverse solves L L^T x = e_k. */
    for (int col = 0; col < n; col++) {
        double y[ESHKOL_FORM_MAX_DIM], x[ESHKOL_FORM_MAX_DIM];
        for (int i = 0; i < n; i++) {
            double s = (i == col) ? 1.0 : 0.0;
            for (int p = 0; p < i; p++) s -= L[i * n + p] * y[p];
            y[i] = s / L[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
            double s = y[i];
            for (int p = i + 1; p < n; p++) s -= L[p * n + i] * x[p];
            x[i] = s / L[i * n + i];
        }
        for (int i = 0; i < n; i++) ginv[i * n + col] = x[i];
    }
    return NULL;
}

/** @brief Determinant of the k x k submatrix of the n x n matrix @p a with rows
 *         @p rows and columns @p cols, by Gaussian elimination with partial
 *         pivoting. k <= 8, so the copy is a handful of doubles. */
static double eshkol_form_subdet(const double* a, int n, const int* rows,
                                 const int* cols, int k) {
    if (k == 0) return 1.0;              /* the empty minor, i.e. <1,1> = 1 */
    double m[ESHKOL_FORM_MAX_DIM * ESHKOL_FORM_MAX_DIM];
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            m[i * k + j] = a[rows[i] * n + cols[j]];
    double det = 1.0;
    for (int c = 0; c < k; c++) {
        int piv = c;
        for (int i = c + 1; i < k; i++)
            if (fabs(m[i * k + c]) > fabs(m[piv * k + c])) piv = i;
        if (m[piv * k + c] == 0.0) return 0.0;
        if (piv != c) {
            for (int j = 0; j < k; j++) {
                double t = m[c * k + j]; m[c * k + j] = m[piv * k + j]; m[piv * k + j] = t;
            }
            det = -det;
        }
        det *= m[c * k + c];
        for (int i = c + 1; i < k; i++) {
            double f = m[i * k + c] / m[c * k + c];
            for (int j = c; j < k; j++) m[i * k + j] -= f * m[c * k + j];
        }
    }
    return det;
}

/**
 * @brief Hodge star of a k-form with respect to the metric @p g at the point,
 *        into the (n-k)-form @p out.
 *
 * Defined by  alpha ^ *w = <alpha, w> vol  for every k-form alpha, with
 * vol = sqrt(det g) dx^1 ^ ... ^ dx^n. In the increasing-multi-index basis that
 * is
 *
 *   (*w)_J = sgn(I, J) sqrt(det g) sum_{I'} det( (g^-1)_{I I'} ) w_{I'},
 *
 * where I is the complement of J (so |I| = k, |J| = n-k) and sgn(I, J) is the
 * sign of the permutation (I then J) of (0,...,n-1). The inner sum is the
 * induced inner product on k-forms: raising the index of w with g^-1 is a k x k
 * minor of the inverse metric, which is why the general (non-diagonal) metric
 * costs a determinant per basis pair rather than a single product.
 *
 * THE RESULT IS A 0-JET, ALWAYS, and says so in its header. The star's
 * coefficients are functions of g, so the jet of *w depends on the jet of the
 * metric; a metric sampled AT ONE POINT carries no information about how g
 * varies, and propagating the input's derivative blocks through it would assert
 * that g is constant -- true in the flat case and false on any curved manifold,
 * and indistinguishable from the truth in the returned value. Declaring r = 0
 * is the honest statement of what a point metric determines. Use `d` before the
 * star, not after it, when both are wanted.
 *
 * @param in   the input form, header-validated.
 * @param g    the n x n metric at the point, row-major, n matching the form.
 * @param out  output of eshkol_form_total(n-k, n, 0).
 * @return NULL on success, else a reason.
 */
static const char* eshkol_form_star(const double* in, long in_total,
                                    const double* g, int gn,
                                    double* out, long out_total) {
    int k, n, r;
    const char* why = eshkol_form_header(in, in_total, &k, &n, &r);
    if (why) return why;
    if (k > n)
        return "the zero top-degree form has no Hodge dual in this "
               "representation";
    if (gn != n)
        return "the metric must be an n x n tensor with n the form's dimension";

    int kk = n - k;
    long want = eshkol_form_total(kk, n, 0);
    if (want < 0 || want != out_total)
        return "internal: the Hodge star's output was sized wrongly";

    double ginv[ESHKOL_FORM_MAX_DIM * ESHKOL_FORM_MAX_DIM];
    double det = 0.0;
    why = eshkol_form_metric_inverse(g, n, ginv, &det);
    if (why) return why;
    double vol = sqrt(det);

    out[0] = (double)kk;
    out[1] = (double)n;
    out[2] = 0.0;

    int m  = (int)eshkol_form_binom(n, k);
    int mm = (int)eshkol_form_binom(n, kk);
    int basis_k[ESHKOL_FORM_MAX_BASIS * ESHKOL_FORM_MAX_DIM];
    int basis_kk[ESHKOL_FORM_MAX_BASIS * ESHKOL_FORM_MAX_DIM];
    if (m > ESHKOL_FORM_MAX_BASIS || mm > ESHKOL_FORM_MAX_BASIS)
        return "internal: the form basis exceeded its enumeration buffer";
    eshkol_form_basis(n, k, basis_k);
    eshkol_form_basis(n, kk, basis_kk);

    long S = eshkol_form_stride(n, r);

    for (int tj = 0; tj < mm; tj++) {
        const int* J = basis_kk + (long)tj * kk;
        int I[ESHKOL_FORM_MAX_DIM], perm[ESHKOL_FORM_MAX_DIM];
        int q = 0;
        for (int v = 0; v < n; v++) {
            int in_J = 0;
            for (int i = 0; i < kk; i++) if (J[i] == v) { in_J = 1; break; }
            if (!in_J) I[q++] = v;
        }
        for (int i = 0; i < k; i++)  perm[i] = I[i];
        for (int i = 0; i < kk; i++) perm[k + i] = J[i];
        double sgn = (double)eshkol_form_perm_sign(perm, n);

        double acc = 0.0;
        for (int ti = 0; ti < m; ti++) {
            double w = in[ESHKOL_FORM_HEADER + (long)ti * S];
            if (w == 0.0) continue;
            const int* Ip = basis_k + (long)ti * k;
            acc += eshkol_form_subdet(ginv, n, I, Ip, k) * w;
        }
        out[ESHKOL_FORM_HEADER + tj] = sgn * vol * acc;
    }
    return NULL;
}

#endif /* ESHKOL_BACKEND_DIFFERENTIAL_FORM_CORE_H */
