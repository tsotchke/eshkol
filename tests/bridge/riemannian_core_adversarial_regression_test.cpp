#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "eshkol/backend/riemannian_core.h"

static int passed = 0;
static int failed = 0;

static void check(const char* name, bool ok) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) ++passed; else ++failed;
}

static void check_norm_and_map_sweep() {
    bool norm_ok = true, map_ok = true;
    for (int n = 1; n <= 64; ++n) {
        std::vector<double> x((size_t)n), v((size_t)n), map_x((size_t)n), y((size_t)n), back((size_t)n);
        std::vector<double> scratch((size_t)n * 2);
        for (int i = 0; i < n; ++i) {
            double t = n == 1 ? 0.5 : (double)i / (double)(n - 1);
            double mag = std::pow(10.0, -300.0 + 600.0 * t);
            x[(size_t)i] = (i & 1) ? -mag : mag;
            v[(size_t)i] = ((i % 5) - 2) * 0.03 / (double)n;
            map_x[(size_t)i] = ((i % 3) - 1) * 0.1 / (double)n;
        }
        double norm = eshkol_rm_norm(x.data(), n);
        long double ref2 = 0.0L;
        for (double a : x) ref2 += (long double)a * (long double)a;
        long double ref = std::sqrt(ref2);
        if (!(std::fabs((long double)norm - ref) / ref < 1e-15L)) norm_ok = false;
        const char* why = eshkol_rm_exp_map(map_x.data(), v.data(), -1.0, n,
                                            y.data(), scratch.data());
        if (why || eshkol_rm_log_map(map_x.data(), y.data(), -1.0, n,
                                     back.data(), scratch.data())) {
            map_ok = false;
        } else {
            for (int i = 0; i < n; ++i)
                if (std::fabs(back[(size_t)i] - v[(size_t)i]) > 1e-12)
                    map_ok = false;
        }
    }
    check("norm_exponent_dimension_sweep_n1_to_64", norm_ok);
    check("exp_log_dimension_sweep_n1_to_64", map_ok);
}

static void check_adversarial_roots() {
    const double factor = 1e-300;
    const double a[2] = {1e-24, 0.0};
    const double b[2] = {1e150, 0.0};
    const double dot = eshkol_rm_scaled_dot_factor(a, b, factor, 2);
    check("scaled_dot_preserves_representable_product", dot != 0.0 &&
          std::fabs(dot - 1e-174) / 1e-174 < 1e-15);

    const double boundary[2] = {0.70183641524540785, 0.71233815441507509};
    const double denom = eshkol_rm_one_minus_bnorm2(boundary, 1.0, 2);
    double distance = 0.0;
    const double origin[2] = {0.0, 0.0};
    const char* distance_why = eshkol_rm_distance(boundary, origin, -1.0, 2,
                                                  &distance);
    check("ball_denominator_keeps_valid_boundary_point", denom > 0.0 &&
          distance_why == nullptr && std::isfinite(distance));

    const double sphere_k = 1e308;
    const double sphere_x[1] = {1.0 / std::sqrt(sphere_k)};
    const double radial_v[1] = {1e-170};
    check("sphere_tangent_rejects_scaled_radial_vector",
          eshkol_rm_check_tangent(sphere_x, radial_v, sphere_k, 1) != nullptr);

    const double huge_y[2] = {0.0, 9.99999999999e149};
    double d = 0.0, d1 = 0.0, d2 = 0.0;
    const char* why = eshkol_rm_distance_dK(origin, huge_y, -1e-300, 2,
                                            &d, &d1, &d2);
    check("curvature_derivative_never_returns_nan_success",
          why != nullptr || (std::isfinite(d) && std::isfinite(d1) &&
                             std::isfinite(d2)));
}

int main() {
    check_norm_and_map_sweep();
    check_adversarial_roots();
    std::printf("Results: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
