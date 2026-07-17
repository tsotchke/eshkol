/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Canonical Eshkol/TensorCore FFI adapter. TensorCore owns the public ABI and
 * execution implementation; Eshkol owns only this language-facing flattening.
 */

#include <eshkol/tensorcore_adapter.h>

#include <cctype>
#include <cstdio>
#include <cstring>

#ifdef ESHKOL_TENSORCORE_ENABLED
#include <tensorcore/tensorcore.h>
#endif

namespace {

thread_local int32_t last_status = ESHKOL_TC_OK;

int32_t record_status(int32_t status) {
    last_status = status;
    return status;
}

#ifdef ESHKOL_TENSORCORE_ENABLED

constexpr int32_t required_major = 0;
constexpr int32_t required_minor = 1;
constexpr int32_t required_patch = 22;

int32_t runtime_abi_status() {
    const char* version = tc_version();
    int major = -1;
    int minor = -1;
    int patch = -1;
    const char* numeric = version;
    while (numeric && *numeric &&
           !std::isdigit(static_cast<unsigned char>(*numeric))) {
        ++numeric;
    }
    if (!numeric || !*numeric ||
        std::sscanf(numeric, "%d.%d.%d", &major, &minor, &patch) != 3) {
        return ESHKOL_TC_ERR_ABI_MISMATCH;
    }
    return eshkol_tc_check_abi_version(major, minor, patch);
}

constexpr uint64_t known_backend_mask =
    TC_BACKEND_MASK_SIMDGROUP_MATRIX |
    TC_BACKEND_MASK_TENSOROPS_M5 |
    TC_BACKEND_MASK_MPS |
    TC_BACKEND_MASK_ACCELERATE_CPU |
    TC_BACKEND_MASK_PORTABLE_CPU |
    TC_BACKEND_MASK_METAL_COMPUTE |
    TC_BACKEND_MASK_CUDA |
    TC_BACKEND_MASK_HIP;

int32_t validate_capability_masks(int32_t abi_version,
                                  uint64_t known_capabilities,
                                  uint64_t available_capabilities,
                                  uint64_t compiled_backends,
                                  uint64_t available_backends) {
    if (abi_version !=
        static_cast<int32_t>(TC_RUNTIME_CAPABILITIES_ABI_VERSION_1)) {
        return ESHKOL_TC_ERR_ABI_MISMATCH;
    }
    if (known_capabilities != TC_CAPABILITY_V1_KNOWN_MASK ||
        (available_capabilities & ~known_capabilities) != 0 ||
        (compiled_backends & ~known_backend_mask) != 0 ||
        (available_backends & ~compiled_backends) != 0) {
        return ESHKOL_TC_ERR_CAPABILITY_MISMATCH;
    }
    return ESHKOL_TC_OK;
}

int32_t query_runtime_capabilities(void* ctx,
                                   int32_t requested_abi_version,
                                   tc_runtime_capabilities& capabilities) {
    if (!ctx || requested_abi_version <= 0) {
        return static_cast<int32_t>(TC_ERR_INVALID_ARG);
    }
    std::memset(&capabilities, 0, sizeof(capabilities));
    const int32_t status = static_cast<int32_t>(tc_runtime_capabilities_get(
        static_cast<tc_context*>(ctx),
        static_cast<uint32_t>(requested_abi_version),
        &capabilities,
        sizeof(capabilities)));
    if (status != static_cast<int32_t>(TC_OK)) return status;
    if (capabilities.struct_size < TC_RUNTIME_CAPABILITIES_V1_MIN_SIZE ||
        capabilities.abi_version !=
            static_cast<uint32_t>(requested_abi_version) ||
        capabilities.reserved0 != 0) {
        return ESHKOL_TC_ERR_ABI_MISMATCH;
    }
    for (uint64_t value : capabilities.reserved) {
        if (value != 0) return ESHKOL_TC_ERR_ABI_MISMATCH;
    }
    return validate_capability_masks(
        static_cast<int32_t>(capabilities.abi_version),
        capabilities.known_capability_mask,
        capabilities.available_capability_mask,
        capabilities.compiled_backend_mask,
        capabilities.available_backend_mask);
}

bool load_device_info(void* ctx, tc_device_info& info) {
    if (!ctx) {
        record_status(static_cast<int32_t>(TC_ERR_INVALID_ARG));
        return false;
    }
    std::memset(&info, 0, sizeof(info));
    const int32_t status = static_cast<int32_t>(
        tc_device_info_get(static_cast<tc_context*>(ctx), &info));
    record_status(status);
    return status == static_cast<int32_t>(TC_OK);
}

bool valid_dtype(int32_t dtype) {
    switch (static_cast<tc_dtype_t>(dtype)) {
        case TC_DTYPE_F16:
        case TC_DTYPE_BF16:
        case TC_DTYPE_F32:
        case TC_DTYPE_I8:
        case TC_DTYPE_I32:
            return true;
        default:
            return false;
    }
}

#endif

}  // namespace

extern "C" int32_t eshkol_tc_check_abi_version(int32_t major,
                                                 int32_t minor,
                                                 int32_t patch) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    if (major != required_major || minor != required_minor || patch < required_patch) {
        return ESHKOL_TC_ERR_ABI_MISMATCH;
    }
    return ESHKOL_TC_OK;
#else
    (void)major;
    (void)minor;
    (void)patch;
    return ESHKOL_TC_ERR_UNAVAILABLE;
#endif
}

extern "C" int32_t eshkol_tc_adapter_status(void) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    return runtime_abi_status();
#else
    return ESHKOL_TC_ERR_UNAVAILABLE;
#endif
}

extern "C" int32_t eshkol_tc_adapter_available(void) {
    return eshkol_tc_adapter_status() == ESHKOL_TC_OK ? 1 : 0;
}

extern "C" int32_t eshkol_tc_last_status(void) {
    return last_status;
}

extern "C" int32_t eshkol_tc_runtime_capabilities_abi_version(void) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    return static_cast<int32_t>(TC_RUNTIME_CAPABILITIES_ABI_VERSION_1);
#else
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return 0;
#endif
}

extern "C" int32_t eshkol_tc_validate_runtime_capabilities(
    int32_t abi_version,
    uint64_t known_capability_mask,
    uint64_t available_capability_mask,
    uint64_t compiled_backend_mask,
    uint64_t available_backend_mask) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    return record_status(validate_capability_masks(
        abi_version,
        known_capability_mask,
        available_capability_mask,
        compiled_backend_mask,
        available_backend_mask));
#else
    (void)abi_version;
    (void)known_capability_mask;
    (void)available_capability_mask;
    (void)compiled_backend_mask;
    (void)available_backend_mask;
    return record_status(ESHKOL_TC_ERR_UNAVAILABLE);
#endif
}

extern "C" int32_t eshkol_tc_runtime_capabilities_status(
    void* ctx,
    int32_t requested_abi_version) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    tc_runtime_capabilities capabilities{};
    return record_status(query_runtime_capabilities(
        ctx, requested_abi_version, capabilities));
#else
    (void)ctx;
    (void)requested_abi_version;
    return record_status(ESHKOL_TC_ERR_UNAVAILABLE);
#endif
}

#ifdef ESHKOL_TENSORCORE_ENABLED
#define ESHKOL_TC_CAPABILITY_MASK_FUNCTION(name, field)                    \
    extern "C" uint64_t name(void* ctx) {                                 \
        tc_runtime_capabilities capabilities{};                            \
        const int32_t status = query_runtime_capabilities(                 \
            ctx, static_cast<int32_t>(                                     \
                     TC_RUNTIME_CAPABILITIES_ABI_VERSION_1),               \
            capabilities);                                                 \
        record_status(status);                                             \
        return status == ESHKOL_TC_OK ? capabilities.field : 0;            \
    }
#else
#define ESHKOL_TC_CAPABILITY_MASK_FUNCTION(name, field)                    \
    extern "C" uint64_t name(void* ctx) {                                 \
        (void)ctx;                                                         \
        (void)sizeof(#field);                                              \
        record_status(ESHKOL_TC_ERR_UNAVAILABLE);                          \
        return 0;                                                          \
    }
#endif

ESHKOL_TC_CAPABILITY_MASK_FUNCTION(
    eshkol_tc_known_capability_mask, known_capability_mask)
ESHKOL_TC_CAPABILITY_MASK_FUNCTION(
    eshkol_tc_available_capability_mask, available_capability_mask)
ESHKOL_TC_CAPABILITY_MASK_FUNCTION(
    eshkol_tc_compiled_backend_mask, compiled_backend_mask)
ESHKOL_TC_CAPABILITY_MASK_FUNCTION(
    eshkol_tc_available_backend_mask, available_backend_mask)

#undef ESHKOL_TC_CAPABILITY_MASK_FUNCTION

extern "C" void* eshkol_tc_init(void) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    const int32_t abi_status = runtime_abi_status();
    if (abi_status != ESHKOL_TC_OK) {
        record_status(abi_status);
        return nullptr;
    }
    tc_context* ctx = nullptr;
    const int32_t status = static_cast<int32_t>(tc_init(&ctx));
    if (status != static_cast<int32_t>(TC_OK)) {
        record_status(status);
        return nullptr;
    }
    tc_runtime_capabilities capabilities{};
    const int32_t capability_status = query_runtime_capabilities(
        ctx,
        static_cast<int32_t>(TC_RUNTIME_CAPABILITIES_ABI_VERSION_1),
        capabilities);
    if (capability_status != ESHKOL_TC_OK) {
        (void)tc_shutdown(ctx);
        record_status(capability_status);
        return nullptr;
    }
    record_status(ESHKOL_TC_OK);
    return ctx;
#else
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return nullptr;
#endif
}

extern "C" int32_t eshkol_tc_shutdown(void* ctx) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    if (!ctx) return record_status(static_cast<int32_t>(TC_ERR_INVALID_ARG));
    return record_status(static_cast<int32_t>(tc_shutdown(static_cast<tc_context*>(ctx))));
#else
    (void)ctx;
    return record_status(ESHKOL_TC_ERR_UNAVAILABLE);
#endif
}

extern "C" const char* eshkol_tc_device_name(void* ctx) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    thread_local char name[sizeof(((tc_device_info*)nullptr)->name)] = {};
    tc_device_info info;
    if (!load_device_info(ctx, info)) return "unknown";
    std::memcpy(name, info.name, sizeof(name));
    name[sizeof(name) - 1] = '\0';
    return name;
#else
    (void)ctx;
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return "unavailable";
#endif
}

#ifdef ESHKOL_TENSORCORE_ENABLED
#define ESHKOL_TC_DEVICE_FLAG_FUNCTION(name, field)                         \
    extern "C" int32_t name(void* ctx) {                                  \
        tc_device_info info;                                                \
        if (!load_device_info(ctx, info)) return 0;                         \
        return info.field ? 1 : 0;                                         \
    }
#else
#define ESHKOL_TC_DEVICE_FLAG_FUNCTION(name, field)                         \
    extern "C" int32_t name(void* ctx) {                                  \
        (void)ctx;                                                          \
        (void)sizeof(#field);                                               \
        record_status(ESHKOL_TC_ERR_UNAVAILABLE);                           \
        return 0;                                                           \
    }
#endif

extern "C" int32_t eshkol_tc_device_family(void* ctx) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    tc_device_info info;
    return load_device_info(ctx, info) ? static_cast<int32_t>(info.family) : 0;
#else
    (void)ctx;
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return 0;
#endif
}

ESHKOL_TC_DEVICE_FLAG_FUNCTION(eshkol_tc_device_unified_memory, unified_memory)
ESHKOL_TC_DEVICE_FLAG_FUNCTION(eshkol_tc_device_supports_bf16, supports_bf16_simdgroup)
ESHKOL_TC_DEVICE_FLAG_FUNCTION(eshkol_tc_device_supports_i8, supports_i8_simdgroup)
ESHKOL_TC_DEVICE_FLAG_FUNCTION(eshkol_tc_device_supports_tensorops_m5, supports_tensorops_m5)

#undef ESHKOL_TC_DEVICE_FLAG_FUNCTION

extern "C" void* eshkol_tc_buffer_alloc(void* ctx, int64_t bytes) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    if (!ctx || bytes <= 0) {
        record_status(static_cast<int32_t>(TC_ERR_INVALID_ARG));
        return nullptr;
    }
    tc_buffer* buffer = nullptr;
    const int32_t status = static_cast<int32_t>(
        tc_buffer_alloc(static_cast<tc_context*>(ctx), static_cast<size_t>(bytes), &buffer));
    record_status(status);
    return status == static_cast<int32_t>(TC_OK) ? buffer : nullptr;
#else
    (void)ctx;
    (void)bytes;
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return nullptr;
#endif
}

extern "C" int32_t eshkol_tc_buffer_free(void* ctx, void* buffer) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    if (!ctx || !buffer) return record_status(static_cast<int32_t>(TC_ERR_INVALID_ARG));
    return record_status(static_cast<int32_t>(
        tc_buffer_free(static_cast<tc_context*>(ctx), static_cast<tc_buffer*>(buffer))));
#else
    (void)ctx;
    (void)buffer;
    return record_status(ESHKOL_TC_ERR_UNAVAILABLE);
#endif
}

extern "C" void* eshkol_tc_buffer_map(void* buffer) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    if (!buffer) {
        record_status(static_cast<int32_t>(TC_ERR_INVALID_ARG));
        return nullptr;
    }
    void* mapped = nullptr;
    const int32_t status = static_cast<int32_t>(
        tc_buffer_map(static_cast<tc_buffer*>(buffer), &mapped));
    record_status(status);
    return status == static_cast<int32_t>(TC_OK) ? mapped : nullptr;
#else
    (void)buffer;
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return nullptr;
#endif
}

extern "C" int64_t eshkol_tc_buffer_size(void* buffer) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    if (!buffer) {
        record_status(static_cast<int32_t>(TC_ERR_INVALID_ARG));
        return -1;
    }
    record_status(static_cast<int32_t>(TC_OK));
    return static_cast<int64_t>(tc_buffer_size(static_cast<tc_buffer*>(buffer)));
#else
    (void)buffer;
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return -1;
#endif
}

#ifdef ESHKOL_TENSORCORE_ENABLED
#define ESHKOL_TC_DTYPE_FUNCTION(name, value) \
    extern "C" int32_t name(void) { return static_cast<int32_t>(value); }
#else
#define ESHKOL_TC_DTYPE_FUNCTION(name, value) \
    extern "C" int32_t name(void) { (void)sizeof(#value); return -1; }
#endif

ESHKOL_TC_DTYPE_FUNCTION(eshkol_tc_dtype_f16, TC_DTYPE_F16)
ESHKOL_TC_DTYPE_FUNCTION(eshkol_tc_dtype_bf16, TC_DTYPE_BF16)
ESHKOL_TC_DTYPE_FUNCTION(eshkol_tc_dtype_f32, TC_DTYPE_F32)
ESHKOL_TC_DTYPE_FUNCTION(eshkol_tc_dtype_i8, TC_DTYPE_I8)
ESHKOL_TC_DTYPE_FUNCTION(eshkol_tc_dtype_i32, TC_DTYPE_I32)

#undef ESHKOL_TC_DTYPE_FUNCTION

extern "C" int32_t eshkol_tc_gemm(void* ctx,
                                    int32_t dtype,
                                    void* a,
                                    void* b,
                                    void* c,
                                    int32_t m,
                                    int32_t n,
                                    int32_t k,
                                    double alpha,
                                    double beta,
                                    int32_t transpose_a,
                                    int32_t transpose_b) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    if (!valid_dtype(dtype)) {
        return record_status(static_cast<int32_t>(TC_ERR_UNSUPPORTED_DTYPE));
    }
    tc_gemm_desc desc{};
    desc.M = m;
    desc.N = n;
    desc.K = k;
    desc.a_dtype = static_cast<tc_dtype_t>(dtype);
    desc.b_dtype = static_cast<tc_dtype_t>(dtype);
    desc.c_dtype = desc.a_dtype == TC_DTYPE_I8 ? TC_DTYPE_I32 : desc.a_dtype;
    desc.accum_dtype = desc.a_dtype == TC_DTYPE_I8 ? TC_DTYPE_I32 : TC_DTYPE_F32;
    desc.transpose_a = transpose_a != 0;
    desc.transpose_b = transpose_b != 0;
    desc.alpha = static_cast<float>(alpha);
    desc.beta = static_cast<float>(beta);
    return record_status(static_cast<int32_t>(tc_gemm(
        static_cast<tc_context*>(ctx), &desc,
        static_cast<const tc_buffer*>(a), static_cast<const tc_buffer*>(b),
        static_cast<tc_buffer*>(c))));
#else
    (void)ctx; (void)dtype; (void)a; (void)b; (void)c;
    (void)m; (void)n; (void)k; (void)alpha; (void)beta;
    (void)transpose_a; (void)transpose_b;
    return record_status(ESHKOL_TC_ERR_UNAVAILABLE);
#endif
}

extern "C" int32_t eshkol_tc_attention_forward(void* ctx,
                                                 void* q,
                                                 void* k,
                                                 void* v,
                                                 void* output,
                                                 int32_t batch,
                                                 int32_t heads,
                                                 int32_t seq_q,
                                                 int32_t seq_kv,
                                                 int32_t head_dim,
                                                 double softmax_scale,
                                                 int32_t causal) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    tc_attention_desc desc{};
    desc.batch = batch;
    desc.heads = heads;
    desc.seq_q = seq_q;
    desc.seq_kv = seq_kv;
    desc.head_dim = head_dim;
    desc.io_dtype = TC_DTYPE_F16;
    desc.accum_dtype = TC_DTYPE_F32;
    desc.softmax_scale = static_cast<float>(softmax_scale);
    desc.causal = causal != 0;
    desc.return_lse = false;
    desc.kv_heads = heads;
    return record_status(static_cast<int32_t>(tc_attention_forward(
        static_cast<tc_context*>(ctx), &desc,
        static_cast<const tc_buffer*>(q), static_cast<const tc_buffer*>(k),
        static_cast<const tc_buffer*>(v), static_cast<tc_buffer*>(output), nullptr)));
#else
    (void)ctx; (void)q; (void)k; (void)v; (void)output;
    (void)batch; (void)heads; (void)seq_q; (void)seq_kv; (void)head_dim;
    (void)softmax_scale; (void)causal;
    return record_status(ESHKOL_TC_ERR_UNAVAILABLE);
#endif
}

extern "C" int32_t eshkol_tc_last_backend_code(void) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    record_status(static_cast<int32_t>(TC_OK));
    return static_cast<int32_t>(tc_last_backend());
#else
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return 0;
#endif
}

extern "C" const char* eshkol_tc_last_backend_name(void) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    record_status(static_cast<int32_t>(TC_OK));
    return tc_backend_name(tc_last_backend());
#else
    record_status(ESHKOL_TC_ERR_UNAVAILABLE);
    return "unavailable";
#endif
}

extern "C" const char* eshkol_tc_version(void) {
#ifdef ESHKOL_TENSORCORE_ENABLED
    return tc_version();
#else
    return "unavailable";
#endif
}

extern "C" const char* eshkol_tc_status_string(int32_t status) {
    switch (status) {
        case ESHKOL_TC_ERR_UNAVAILABLE:
            return "Eshkol TensorCore adapter unavailable";
        case ESHKOL_TC_ERR_ABI_MISMATCH:
            return "Eshkol TensorCore adapter ABI mismatch";
        case ESHKOL_TC_ERR_CAPABILITY_MISMATCH:
            return "Eshkol TensorCore runtime capability mismatch";
        default:
#ifdef ESHKOL_TENSORCORE_ENABLED
            return tc_status_string(static_cast<tc_status_t>(status));
#else
            return status == ESHKOL_TC_OK ? "ok" : "TensorCore status unavailable";
#endif
    }
}
