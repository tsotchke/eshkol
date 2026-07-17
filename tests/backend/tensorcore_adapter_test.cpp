/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/tensorcore_adapter.h>

#include <cmath>
#include <cstdint>
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
    if (std::strlen(eshkol_tc_status_string(ESHKOL_TC_ERR_ABI_MISMATCH)) == 0) {
        return fail("adapter mismatch diagnostic is empty");
    }
    if (std::strlen(eshkol_tc_status_string(
            ESHKOL_TC_ERR_CAPABILITY_MISMATCH)) == 0) {
        return fail("capability mismatch diagnostic is empty");
    }

    if (!eshkol_tc_adapter_available()) {
        if (eshkol_tc_adapter_status() != ESHKOL_TC_ERR_UNAVAILABLE) {
            return fail("disabled adapter did not report explicit unavailable");
        }
        if (eshkol_tc_check_abi_version(0, 1, 22) != ESHKOL_TC_ERR_UNAVAILABLE) {
            return fail("disabled adapter accepted an ABI version");
        }
        if (eshkol_tc_init() != nullptr ||
            eshkol_tc_last_status() != ESHKOL_TC_ERR_UNAVAILABLE) {
            return fail("disabled init did not preserve unavailable status");
        }
        if (eshkol_tc_runtime_capabilities_abi_version() != 0 ||
            eshkol_tc_runtime_capabilities_status(nullptr, 1) !=
                ESHKOL_TC_ERR_UNAVAILABLE ||
            eshkol_tc_known_capability_mask(nullptr) != 0) {
            return fail("disabled capability discovery was not fail-closed");
        }
        std::cout << "PASS: TensorCore adapter explicit unavailable\n";
        return 0;
    }

    if (eshkol_tc_adapter_status() != ESHKOL_TC_OK) {
        return fail("enabled adapter failed its runtime ABI check");
    }
    if (eshkol_tc_check_abi_version(0, 1, 22) != ESHKOL_TC_OK ||
        eshkol_tc_check_abi_version(0, 1, 21) != ESHKOL_TC_ERR_ABI_MISMATCH ||
        eshkol_tc_check_abi_version(0, 2, 0) != ESHKOL_TC_ERR_ABI_MISMATCH) {
        return fail("ABI compatibility window is not deterministic");
    }
    void* ctx = eshkol_tc_init();
    if (!ctx) return fail("TensorCore initialization failed");

    const int32_t capability_abi =
        eshkol_tc_runtime_capabilities_abi_version();
    if (capability_abi <= 0 ||
        eshkol_tc_runtime_capabilities_status(ctx, capability_abi) !=
            ESHKOL_TC_OK ||
        eshkol_tc_runtime_capabilities_status(ctx, capability_abi + 1) ==
            ESHKOL_TC_OK) {
        return fail("runtime capability ABI version did not fail closed");
    }
    const uint64_t known_capabilities =
        eshkol_tc_known_capability_mask(ctx);
    const uint64_t available_capabilities =
        eshkol_tc_available_capability_mask(ctx);
    const uint64_t compiled_backends =
        eshkol_tc_compiled_backend_mask(ctx);
    const uint64_t available_backends =
        eshkol_tc_available_backend_mask(ctx);
    if (known_capabilities == 0 || compiled_backends == 0 ||
        eshkol_tc_validate_runtime_capabilities(
            capability_abi,
            known_capabilities,
            available_capabilities,
            compiled_backends,
            available_backends) != ESHKOL_TC_OK) {
        return fail("runtime capability masks were not accepted");
    }
    constexpr uint64_t unknown_bit = UINT64_C(1) << 63;
    if (eshkol_tc_validate_runtime_capabilities(
            capability_abi,
            known_capabilities | unknown_bit,
            available_capabilities,
            compiled_backends,
            available_backends) != ESHKOL_TC_ERR_CAPABILITY_MISMATCH ||
        eshkol_tc_validate_runtime_capabilities(
            capability_abi,
            known_capabilities,
            available_capabilities | unknown_bit,
            compiled_backends,
            available_backends) != ESHKOL_TC_ERR_CAPABILITY_MISMATCH ||
        eshkol_tc_validate_runtime_capabilities(
            capability_abi,
            known_capabilities,
            available_capabilities,
            compiled_backends | unknown_bit,
            available_backends) != ESHKOL_TC_ERR_CAPABILITY_MISMATCH ||
        eshkol_tc_validate_runtime_capabilities(
            capability_abi,
            known_capabilities,
            available_capabilities,
            compiled_backends,
            available_backends | unknown_bit) !=
                ESHKOL_TC_ERR_CAPABILITY_MISMATCH) {
        return fail("unknown capability/backend bits did not fail closed");
    }

    constexpr int64_t bytes = 4 * static_cast<int64_t>(sizeof(float));
    void* a = eshkol_tc_buffer_alloc(ctx, bytes);
    void* b = eshkol_tc_buffer_alloc(ctx, bytes);
    void* c = eshkol_tc_buffer_alloc(ctx, bytes);
    if (!a || !b || !c) return fail("portable buffers could not be allocated");
    if (eshkol_tc_buffer_size(a) < bytes) return fail("buffer size was truncated");

    auto* a_data = static_cast<float*>(eshkol_tc_buffer_map(a));
    auto* b_data = static_cast<float*>(eshkol_tc_buffer_map(b));
    auto* c_data = static_cast<float*>(eshkol_tc_buffer_map(c));
    if (!a_data || !b_data || !c_data) return fail("portable buffers could not be mapped");

    a_data[0] = 1.0f; a_data[1] = 2.0f;
    a_data[2] = 3.0f; a_data[3] = 4.0f;
    b_data[0] = 5.0f; b_data[1] = 6.0f;
    b_data[2] = 7.0f; b_data[3] = 8.0f;
    std::memset(c_data, 0, static_cast<std::size_t>(bytes));

    const int32_t status = eshkol_tc_gemm(
        ctx, eshkol_tc_dtype_f32(), a, b, c, 2, 2, 2, 1.0, 0.0, 0, 0);
    if (status != ESHKOL_TC_OK) {
        std::cerr << "TensorCore status: " << eshkol_tc_status_string(status) << '\n';
        return fail("portable FP32 GEMM failed");
    }
    if (!close(c_data[0], 19.0f) || !close(c_data[1], 22.0f) ||
        !close(c_data[2], 43.0f) || !close(c_data[3], 50.0f)) {
        return fail("portable FP32 GEMM result was incorrect");
    }
    const char* backend_name = eshkol_tc_last_backend_name();
    const char* device_name = eshkol_tc_device_name(ctx);
    if (!backend_name || std::strlen(backend_name) == 0) {
        return fail("TensorCore did not publish the serving backend");
    }

    constexpr int32_t head_dim = 64;
    constexpr int64_t attention_bytes =
        head_dim * static_cast<int64_t>(sizeof(uint16_t));
    void* q = eshkol_tc_buffer_alloc(ctx, attention_bytes);
    void* key = eshkol_tc_buffer_alloc(ctx, attention_bytes);
    void* v = eshkol_tc_buffer_alloc(ctx, attention_bytes);
    void* output = eshkol_tc_buffer_alloc(ctx, attention_bytes);
    if (!q || !key || !v || !output) {
        return fail("attention buffers could not be allocated");
    }
    void* q_data = eshkol_tc_buffer_map(q);
    void* key_data = eshkol_tc_buffer_map(key);
    void* v_data = eshkol_tc_buffer_map(v);
    void* output_data = eshkol_tc_buffer_map(output);
    if (!q_data || !key_data || !v_data || !output_data) {
        return fail("attention buffers could not be mapped");
    }
    std::memset(q_data, 0, static_cast<std::size_t>(attention_bytes));
    std::memset(key_data, 0, static_cast<std::size_t>(attention_bytes));
    std::memset(v_data, 0, static_cast<std::size_t>(attention_bytes));
    std::memset(output_data, 0xff,
                static_cast<std::size_t>(attention_bytes));
    if (eshkol_tc_attention_forward(
            ctx, q, key, v, output,
            1, 1, 1, 1, head_dim, 0.125, 0) != ESHKOL_TC_OK) {
        return fail("attention descriptor/runtime path failed");
    }
    if (eshkol_tc_buffer_free(ctx, output) != 0 ||
        eshkol_tc_buffer_free(ctx, v) != 0 ||
        eshkol_tc_buffer_free(ctx, key) != 0 ||
        eshkol_tc_buffer_free(ctx, q) != 0) {
        return fail("attention buffer cleanup failed");
    }
    if (eshkol_tc_gemm(ctx, -77, a, b, c, 2, 2, 2, 1.0, 0.0, 0, 0) == 0) {
        return fail("invalid dtype was accepted");
    }

    if (eshkol_tc_buffer_free(ctx, c) != 0 ||
        eshkol_tc_buffer_free(ctx, b) != 0 ||
        eshkol_tc_buffer_free(ctx, a) != 0) {
        return fail("buffer cleanup failed");
    }
    if (eshkol_tc_shutdown(ctx) != 0) return fail("TensorCore shutdown failed");

    std::cout << "PASS: TensorCore adapter runtime backend=" << backend_name
              << " device=" << (device_name ? device_name : "unknown") << '\n';
    return 0;
}
