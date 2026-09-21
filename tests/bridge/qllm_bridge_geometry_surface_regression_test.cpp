#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/bridge/qllm_bridge.h"
#include "eshkol/backend/riemannian_core.h"

extern "C" {
typedef struct arena arena_t;
arena_t* get_global_arena(void);
void* arena_allocate_zeroed(arena_t*, size_t);
ad_node_t* arena_allocate_ad_node(arena_t*);
}

static int passed = 0;
static int failed = 0;
static void check(const char* name, bool ok) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) ++passed; else ++failed;
}

static ad_node_t* point(const double* values, int n) {
    ad_node_t* node = arena_allocate_ad_node(get_global_arena());
    double* data = (double*)arena_allocate_zeroed(get_global_arena(),
                                                   (size_t)n * sizeof(double));
    int64_t* shape = (int64_t*)arena_allocate_zeroed(get_global_arena(),
                                                     sizeof(int64_t));
    std::memcpy(data, values, (size_t)n * sizeof(double));
    shape[0] = n;
    node->type = AD_NODE_VARIABLE;
    node->tensor_value = data;
    node->shape = shape;
    node->ndim = 1;
    return node;
}

static bool same_bits(double a, double b) {
    uint64_t ua = 0, ub = 0;
    std::memcpy(&ua, &a, sizeof ua);
    std::memcpy(&ub, &b, sizeof ub);
    return ua == ub;
}

static void compare_distance_exp_log() {
    bool distance_ok = true, exp_ok = true, log_ok = true;
    const double values[] = {0.5, 1e-9, 1e-16, 1e-300};
    for (double y : values) {
        double x0[1] = {0.0}, yv[1] = {y}, vm = 0.0, scratch[2] = {}, out[1] = {};
        const char* why = eshkol_rm_distance(x0, yv, -1.0, 1, &vm);
        ad_node_t* native = ad_hyperbolic_distance(nullptr, point(x0, 1),
                                                   point(yv, 1), -1.0);
        distance_ok = distance_ok && !why && native &&
                      same_bits(vm, ((double*)native->tensor_value)[0]);
        why = eshkol_rm_exp_map(x0, yv, -1.0, 1, out, scratch);
        native = ad_poincare_exp_map(nullptr, point(x0, 1), point(yv, 1), -1.0);
        exp_ok = exp_ok && !why && native &&
                 same_bits(out[0], ((double*)native->tensor_value)[0]);
        why = eshkol_rm_log_map(x0, yv, -1.0, 1, out, scratch);
        native = ad_poincare_log_map(nullptr, point(x0, 1), point(yv, 1), -1.0);
        log_ok = log_ok && !why && native &&
                 same_bits(out[0], ((double*)native->tensor_value)[0]);
    }
    check("bridge_distance_matches_core_bitwise_at_exponents", distance_ok);
    check("bridge_exp_matches_core_bitwise_at_exponents", exp_ok);
    check("bridge_log_matches_core_bitwise_at_exponents", log_ok);

    bool sweep_ok = true;
    for (int n = 1; n <= 64; ++n) {
        std::vector<double> x((size_t)n), y((size_t)n), out((size_t)n), scratch((size_t)n);
        for (int i = 0; i < n; ++i) {
            x[(size_t)i] = ((i % 3) - 1) * 0.1 / n;
            y[(size_t)i] = ((i % 5) - 2) * 0.08 / n;
        }
        double vm = 0.0;
        const char* why = eshkol_rm_distance(x.data(), y.data(), -1.0, n, &vm);
        ad_node_t* native = ad_hyperbolic_distance(nullptr, point(x.data(), n),
                                                   point(y.data(), n), -1.0);
        sweep_ok = sweep_ok && !why && native &&
                   same_bits(vm, ((double*)native->tensor_value)[0]);
        why = eshkol_rm_exp_map(x.data(), y.data(), -1.0, n, out.data(), scratch.data());
        native = ad_poincare_exp_map(nullptr, point(x.data(), n), point(y.data(), n), -1.0);
        if (!why && native) for (int i = 0; i < n; ++i)
            sweep_ok = sweep_ok && same_bits(out[(size_t)i],
                                             ((double*)native->tensor_value)[i]);
    }
    check("bridge_core_bit_identity_dimension_sweep_n1_to_64", sweep_ok);
}

int main() {
    compare_distance_exp_log();
    std::printf("Results: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
