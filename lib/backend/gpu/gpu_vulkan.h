/*
 * gpu_vulkan.h — Vulkan compute backend behind the ordinary GPU dispatch.
 *
 * The public GPU API (inc/eshkol/backend/gpu/gpu_memory.h) is implemented by
 * one host backend file per build (gpu_memory_cuda.cpp, gpu_memory_stub.cpp,
 * ...). On Linux those files also carry ESHKOL_GPU_VULKAN: when it is the
 * active backend they route Vulkan buffers to the eshkol_vk_* functions below.
 *
 * Contract:
 *   - The Vulkan loader is opened at run time; nothing links against it.
 *   - A device is usable only if it has a compute queue and shaderFloat64.
 *   - Every kernel is admitted per device by a bitwise probe against the host
 *     reference at initialisation. An operation that is not admitted (or not
 *     implemented, e.g. f64 transcendentals, which Vulkan's GLSL.std.450 set
 *     only defines for 16/32-bit floats) returns ESHKOL_VK_UNSUPPORTED and the
 *     caller computes it on the CPU.
 *   - Wrapped host buffers keep host_ptr equal to the caller's pointer; every
 *     operation reads its inputs from host_ptr and writes its outputs back to
 *     host_ptr before returning.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_GPU_VULKAN_H
#define ESHKOL_GPU_VULKAN_H

#include <eshkol/backend/gpu/gpu_memory.h>
#include "gpu_backend_request.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_VK_OK            0
#define ESHKOL_VK_INVALID      (-1)
#define ESHKOL_VK_UNSUPPORTED  (-2)
#define ESHKOL_VK_DEVICE_ERROR (-3)

/* eshkol_vk_init flag: accept a CPU-type device (e.g. Mesa lavapipe). Auto
 * selection leaves it out so a software rasteriser never shadows the CPU path. */
#define ESHKOL_VK_ALLOW_CPU_DEVICE 0x1u

/* Chunk size of the deterministic reduction tree (see reduce_f64.comp). */
#define ESHKOL_VK_REDUCE_CHUNK 256u

/* Pick and open a device. ESHKOL_VULKAN_DEVICE=<index> selects one
 * explicitly (any type). Returns 1 when a device is ready, 0 otherwise. */
int eshkol_vk_init(unsigned flags);
/* Open device `index` from eshkol_vk_device_count(); any type. 1 = ready. */
int eshkol_vk_init_device(int index);
void eshkol_vk_shutdown(void);
int eshkol_vk_available(void);
/* Human-readable reason for the last init result (never NULL). */
const char* eshkol_vk_status(void);
const char* eshkol_vk_device_name(void);
int eshkol_vk_device_type(void);   /* VkPhysicalDeviceType, -1 if none */
int eshkol_vk_device_index(void);  /* -1 if none */
int eshkol_vk_supports_f64(void);

/* Enumeration (opens the loader and an instance, not a device). */
int eshkol_vk_device_count(void);
int eshkol_vk_device_describe(int index, char* name, size_t name_cap,
                              int* device_type, int* has_f64);

/* Per-device admission of each kernel (1 = runs on the device). */
int eshkol_vk_elementwise_admitted(EshkolElementwiseOp op);
int eshkol_vk_reduce_admitted(EshkolReduceOp op);
int eshkol_vk_matmul_admitted(int single_precision);

/* Buffers: backend == ESHKOL_GPU_VULKAN, backend_data owns the Vulkan memory. */
int eshkol_vk_alloc(size_t size_bytes, EshkolGPUBuffer* out_buffer);
int eshkol_vk_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out_buffer);
void eshkol_vk_free(EshkolGPUBuffer* buffer);
int eshkol_vk_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction);

/* Kernels: ESHKOL_VK_OK, or a negative status (caller falls back to CPU). */
int eshkol_vk_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                              EshkolGPUBuffer* out, uint64_t n,
                              EshkolElementwiseOp op);
int eshkol_vk_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                         uint64_t n, EshkolReduceOp op);
int eshkol_vk_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                         uint64_t M, uint64_t K, uint64_t N);
int eshkol_vk_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                         uint64_t M, uint64_t K, uint64_t N);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_GPU_VULKAN_H */
