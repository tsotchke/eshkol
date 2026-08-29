#ifndef ESHKOL_BACKEND_RIEMANNIAN_CORE_H
#define ESHKOL_BACKEND_RIEMANNIAN_CORE_H

#include <math.h>

/* The gyration of the Poincare-ball gyrovector space is linear in its
 * tangent-vector argument.  Keep this closed form in a header because the VM
 * geometric dispatcher is included into more than one translation unit. */
static double eshkol_rm_dot(const double* a, const double* b, int n)
{
    double result = 0.0;
    for (int i = 0; i < n; ++i) result += a[i] * b[i];
    return result;
}

static double eshkol_rm_mobius_den(const double* a, const double* b,
                                   double curvature_parameter, int n)
{
    const double ab = eshkol_rm_dot(a, b, n);
    const double a2 = eshkol_rm_dot(a, a, n);
    const double b2 = eshkol_rm_dot(b, b, n);
    const double q = 1.0 + curvature_parameter * ab;
    return q * q + curvature_parameter * curvature_parameter *
                         (a2 * b2 - ab * ab);
}

/*
 *   D = 1 + 2B<a,b> + B^2|a|^2|b|^2
 *   A = -B^2<a,w>|b|^2 + B<b,w> + 2B^2<a,b><b,w>
 *   C = -B^2<b,w>|a|^2 - B<a,w>
 *   gyr[a,b]w = w + 2(Aa + Cb)/D
 *
 * Unlike expanding gyration through Mobius point addition, this accepts an
 * arbitrary tangent vector.  The tangent vector need not be inside the ball,
 * so the only denominator here must be the denominator of the two validated
 * points a and b.
 */
static void eshkol_rm_gyration(const double* a, const double* b, const double* w,
                               double curvature_parameter, int n, double* out)
{
    const double ab = eshkol_rm_dot(a, b, n);
    const double a2 = eshkol_rm_dot(a, a, n);
    const double b2 = eshkol_rm_dot(b, b, n);
    const double aw = eshkol_rm_dot(a, w, n);
    const double bw = eshkol_rm_dot(b, w, n);
    const double denominator = eshkol_rm_mobius_den(
        a, b, curvature_parameter, n);
    const double A = -curvature_parameter * curvature_parameter * aw * b2 +
                     curvature_parameter * bw +
                     2.0 * curvature_parameter * curvature_parameter * ab * bw;
    const double C = -curvature_parameter * curvature_parameter * bw * a2 -
                     curvature_parameter * aw;
    for (int i = 0; i < n; ++i) {
        out[i] = w[i] + 2.0 * (A * a[i] + C * b[i]) / denominator;
    }
}

#endif
