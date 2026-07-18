/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Verify hosted linear-solve environment control and explicit core options.
 */

#include <eshkol/core/linear_solve.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << "linear_solve_hosted_control_test: " << message << '\n';
    return 1;
}

void set_env(const char* key, const char* value) {
#ifdef _WIN32
    if (_putenv_s(key, value) != 0) {
        std::cerr << "linear_solve_hosted_control_test: _putenv_s failed" << '\n';
        std::exit(1);
    }
#else
    if (setenv(key, value, 1) != 0) {
        std::cerr << "linear_solve_hosted_control_test: setenv failed" << '\n';
        std::exit(1);
    }
#endif
}

void clear_env(const char* key) {
#ifdef _WIN32
    if (_putenv_s(key, "") != 0) {
        std::cerr << "linear_solve_hosted_control_test: _putenv_s clear failed" << '\n';
        std::exit(1);
    }
#else
    if (unsetenv(key) != 0) {
        std::cerr << "linear_solve_hosted_control_test: unsetenv failed" << '\n';
        std::exit(1);
    }
#endif
}

void apply_scenario_env(const char* scenario) {
    if (std::strcmp(scenario, "unset") == 0) {
        clear_env("ESHKOL_LINSOLVE_FORCE_DGESV");
    } else if (std::strcmp(scenario, "zero") == 0) {
        set_env("ESHKOL_LINSOLVE_FORCE_DGESV", "0");
    } else if (std::strcmp(scenario, "one") == 0) {
        set_env("ESHKOL_LINSOLVE_FORCE_DGESV", "1");
    } else {
        std::cerr << "linear_solve_hosted_control_test: unknown scenario: " << scenario << '\n';
        std::exit(1);
    }
}

double relative_residual(const double* A, const double* b, const double* x, int n) {
    double rnorm = 0.0;
    double bnorm = 0.0;
    for (int i = 0; i < n; i++) {
        double row = 0.0;
        for (int j = 0; j < n; j++) {
            row += A[i * n + j] * x[j];
        }
        const double resid = row - b[i];
        rnorm += resid * resid;
        bnorm += b[i] * b[i];
    }
    if (bnorm == 0.0) return rnorm == 0.0 ? 0.0 : 1e300;
    return std::sqrt(rnorm / bnorm);
}

int run_case(const char* label, const char* scenario, uint32_t expected_query,
             uint32_t core_options) {
    apply_scenario_env(scenario);
    const uint32_t observed = eshkol_linear_solve_query_options();
    if (observed != expected_query) {
        std::cerr << "linear_solve_hosted_control_test: " << label
                  << " query mismatch (observed=" << observed
                  << ", expected=" << expected_query << ')' << '\n';
        return fail("query mismatch");
    }

    const int64_t a_dims[2] = {3, 3};
    const int64_t b_dims[1] = {3};
    const double A[9] = {
        4.0, 1.0, 0.0,
        1.0, 4.0, 1.0,
        0.0, 1.0, 4.0
    };
    const double b[3] = {6.0, 12.0, 14.0};
    double x_core[3] = {0.0, 0.0, 0.0};
    double x_hosted[3] = {0.0, 0.0, 0.0};

    const int64_t status_core = eshkol_linear_solve_with_options(
        2, a_dims, 1, b_dims, A, b, x_core, core_options);
    if (status_core != ESHKOL_LINSOLVE_OK) {
        return fail("core solve failed");
    }

    if (relative_residual(A, b, x_core, 3) > 1e-12) {
        return fail("core residual too large");
    }

    const int64_t status_hosted = eshkol_linear_solve(
        2, a_dims, 1, b_dims, A, b, x_hosted);
    if (status_hosted != ESHKOL_LINSOLVE_OK) {
        return fail("hosted solve failed");
    }

    if (relative_residual(A, b, x_hosted, 3) > 1e-12) {
        return fail("hosted residual too large");
    }

    (void)label;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: linear_solve_hosted_control_test <unset|zero|one>");
    }

    if (std::strcmp(argv[1], "unset") == 0) {
        return run_case("unset", "unset", 0u, 0u);
    }
    if (std::strcmp(argv[1], "zero") == 0) {
        return run_case("zero", "zero", 0u, 0u);
    }
    if (std::strcmp(argv[1], "one") == 0) {
        return run_case("one", "one", ESHKOL_LINSOLVE_FORCE_DGESV, ESHKOL_LINSOLVE_FORCE_DGESV);
    }
    return fail("unknown scenario");
}
