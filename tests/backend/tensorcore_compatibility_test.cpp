/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Compatibility-window gate: compare Eshkol's canonical adapter with the
 * historical TensorCore-owned tc_eshkol_* shim before that shim is demoted.
 */

#include <eshkol/tensorcore_adapter.h>
#include <tensorcore/eshkol_bridge.h>

#include <cmath>
#include <cstring>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool close(float left, float right) {
    return std::fabs(left - right) < 1.0e-5f;
}

}  // namespace

int main() {
    void* ctx = eshkol_tc_init();
    if (!ctx) return fail("canonical initialization failed");

    constexpr int64_t bytes = 4 * static_cast<int64_t>(sizeof(float));
    void* a = eshkol_tc_buffer_alloc(ctx, bytes);
    void* b = eshkol_tc_buffer_alloc(ctx, bytes);
    void* canonical_c = eshkol_tc_buffer_alloc(ctx, bytes);
    void* compatibility_c = eshkol_tc_buffer_alloc(ctx, bytes);
    if (!a || !b || !canonical_c || !compatibility_c) {
        return fail("buffer allocation failed");
    }

    auto* a_data = static_cast<float*>(eshkol_tc_buffer_map(a));
    auto* b_data = static_cast<float*>(eshkol_tc_buffer_map(b));
    auto* canonical_data = static_cast<float*>(eshkol_tc_buffer_map(canonical_c));
    auto* compatibility_data = static_cast<float*>(eshkol_tc_buffer_map(compatibility_c));
    if (!a_data || !b_data || !canonical_data || !compatibility_data) {
        return fail("buffer mapping failed");
    }
    a_data[0] = 1.0f; a_data[1] = 2.0f;
    a_data[2] = 3.0f; a_data[3] = 4.0f;
    b_data[0] = 5.0f; b_data[1] = 6.0f;
    b_data[2] = 7.0f; b_data[3] = 8.0f;
    std::memset(canonical_data, 0, static_cast<std::size_t>(bytes));
    std::memset(compatibility_data, 0, static_cast<std::size_t>(bytes));

    const int32_t canonical_status = eshkol_tc_gemm(
        ctx, eshkol_tc_dtype_f32(), a, b, canonical_c,
        2, 2, 2, 1.0, 0.0, 0, 0);
    const int32_t compatibility_status = tc_eshkol_gemm(
        ctx, eshkol_tc_dtype_f32(), a, b, compatibility_c,
        2, 2, 2, 1.0, 0.0, 0, 0);
    if (canonical_status != compatibility_status || canonical_status != 0) {
        return fail("canonical and compatibility status diverged");
    }
    for (int i = 0; i < 4; ++i) {
        if (!close(canonical_data[i], compatibility_data[i])) {
            return fail("canonical and compatibility GEMM results diverged");
        }
    }
    if (std::strcmp(eshkol_tc_version(), tc_eshkol_version()) != 0 ||
        std::strcmp(eshkol_tc_status_string(0), tc_eshkol_status_string(0)) != 0) {
        return fail("canonical and compatibility diagnostics diverged");
    }

    if (eshkol_tc_buffer_free(ctx, compatibility_c) != 0 ||
        eshkol_tc_buffer_free(ctx, canonical_c) != 0 ||
        eshkol_tc_buffer_free(ctx, b) != 0 ||
        eshkol_tc_buffer_free(ctx, a) != 0 ||
        eshkol_tc_shutdown(ctx) != 0) {
        return fail("cleanup failed");
    }

    std::cout << "PASS: TensorCore compatibility-window parity\n";
    return 0;
}
