/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol-owned flat FFI adapter for TensorCore's public C ABI.
 */
#ifndef ESHKOL_TENSORCORE_ADAPTER_H
#define ESHKOL_TENSORCORE_ADAPTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Adapter-specific failures deliberately live outside TensorCore's status
 * range. TensorCore status values are otherwise returned unchanged. */
enum {
    ESHKOL_TC_OK = 0,
    ESHKOL_TC_ERR_UNAVAILABLE = -1000,
    ESHKOL_TC_ERR_ABI_MISMATCH = -1001,
    ESHKOL_TC_ERR_CAPABILITY_MISMATCH = -1002
};

int32_t  eshkol_tc_adapter_available(void);
int32_t  eshkol_tc_adapter_status(void);
int32_t  eshkol_tc_check_abi_version(int32_t major, int32_t minor, int32_t patch);
int32_t  eshkol_tc_last_status(void);

/* Capability values are queried from TensorCore's size/versioned public ABI.
 * Eshkol deliberately does not publish a second set of capability or backend
 * bits. Unknown ABI versions, feature bits, and backend bits fail closed. */
int32_t  eshkol_tc_runtime_capabilities_abi_version(void);
int32_t  eshkol_tc_runtime_capabilities_status(void* ctx,
                                               int32_t requested_abi_version);
int32_t  eshkol_tc_validate_runtime_capabilities(int32_t abi_version,
                                                  uint64_t known_capability_mask,
                                                  uint64_t available_capability_mask,
                                                  uint64_t compiled_backend_mask,
                                                  uint64_t available_backend_mask);
uint64_t eshkol_tc_known_capability_mask(void* ctx);
uint64_t eshkol_tc_available_capability_mask(void* ctx);
uint64_t eshkol_tc_compiled_backend_mask(void* ctx);
uint64_t eshkol_tc_available_backend_mask(void* ctx);

void*   eshkol_tc_init(void);
int32_t eshkol_tc_shutdown(void* ctx);

const char* eshkol_tc_device_name(void* ctx);
int32_t     eshkol_tc_device_family(void* ctx);
int32_t     eshkol_tc_device_unified_memory(void* ctx);
int32_t     eshkol_tc_device_supports_bf16(void* ctx);
int32_t     eshkol_tc_device_supports_i8(void* ctx);
int32_t     eshkol_tc_device_supports_tensorops_m5(void* ctx);

void*   eshkol_tc_buffer_alloc(void* ctx, int64_t bytes);
int32_t eshkol_tc_buffer_free(void* ctx, void* buffer);
void*   eshkol_tc_buffer_map(void* buffer);
int64_t eshkol_tc_buffer_size(void* buffer);

/* Dtype values are obtained from TensorCore's installed public headers by the
 * adapter implementation. Eshkol code must call these accessors instead of
 * copying TensorCore enum ordinals into language modules. */
int32_t eshkol_tc_dtype_f16(void);
int32_t eshkol_tc_dtype_bf16(void);
int32_t eshkol_tc_dtype_f32(void);
int32_t eshkol_tc_dtype_i8(void);
int32_t eshkol_tc_dtype_i32(void);

int32_t eshkol_tc_gemm(void* ctx,
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
                       int32_t transpose_b);

int32_t eshkol_tc_attention_forward(void* ctx,
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
                                    int32_t causal);

int32_t     eshkol_tc_last_backend_code(void);
const char* eshkol_tc_last_backend_name(void);
const char* eshkol_tc_version(void);
const char* eshkol_tc_status_string(int32_t status);

#ifdef __cplusplus
}
#endif

#endif
