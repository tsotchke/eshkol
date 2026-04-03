/**
 * @file vm_gpu_dispatch.h
 * @brief GPU dispatch layer for VM tensor operations.
 *
 * Sits between the VM's native call dispatch and the CPU tensor
 * implementations. Each vm_gpu_try_* function attempts GPU execution
 * and returns the result tensor on success, or NULL to fall through
 * to the CPU path.
 *
 * On platforms without GPU (WASM, Linux without NVIDIA), all try_*
 * functions return NULL — zero overhead, pure CPU fallback.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */
#ifndef VM_GPU_DISPATCH_H
#define VM_GPU_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

/* GPU backend declarations (C-linkage, no gpu_memory.h dependency) */
extern int  eshkol_gpu_init(void);
extern int  eshkol_gpu_should_use(size_t num_elements);

/* Lazy initialization */
typedef struct {
    int initialized;
    int gpu_available;
    size_t threshold;
} VmGpuState;

static VmGpuState g_vm_gpu = { 0, 0, 100000 };

static inline void vm_gpu_ensure_init(void) {
    if (!g_vm_gpu.initialized) {
        g_vm_gpu.initialized = 1;
#if defined(ESHKOL_GPU_ENABLED)
        int devs = eshkol_gpu_init();
        g_vm_gpu.gpu_available = (devs > 0);
#else
        g_vm_gpu.gpu_available = 0;
#endif
    }
}

static inline int vm_gpu_should_dispatch(int64_t total_elements) {
    vm_gpu_ensure_init();
    if (!g_vm_gpu.gpu_available) return 0;
    return (size_t)total_elements >= g_vm_gpu.threshold;
}

/* GPU-accelerated matmul — returns result tensor or NULL (fall to CPU) */
#if defined(ESHKOL_GPU_ENABLED)
extern int eshkol_gpu_matmul_f64(void* A, void* B, void* C,
                                   uint64_t M, uint64_t K, uint64_t N);
extern int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes, void* out_buffer);
extern void eshkol_gpu_free(void* buffer);

/* GPU buffer (matches EshkolGPUBuffer layout) */
typedef struct { void* host; void* dev; size_t size; int mem; int be; uint32_t fl; void* bd; } VmGpuBuf;

static inline VmTensor* vm_gpu_try_matmul(VmRegionStack* rs,
                                            const VmTensor* a, const VmTensor* b) {
    if (!vm_gpu_should_dispatch(a->total * b->total)) return NULL;
    if (a->n_dims != 2 || b->n_dims != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    int64_t out_shape[2] = { M, N };
    VmTensor* out = vm_tensor_zeros(rs, out_shape, 2);
    if (!out) return NULL;

    VmGpuBuf ba, bb, bc;
    if (eshkol_gpu_wrap_host(a->data, M*K*sizeof(double), &ba) != 0 ||
        eshkol_gpu_wrap_host(b->data, K*N*sizeof(double), &bb) != 0 ||
        eshkol_gpu_wrap_host(out->data, M*N*sizeof(double), &bc) != 0)
        return NULL;

    int rc = eshkol_gpu_matmul_f64(&ba, &bb, &bc, M, K, N);
    if (rc == 0 && bc.host != (void*)out->data)
        memcpy(out->data, bc.host, M*N*sizeof(double));
    eshkol_gpu_free(&ba);
    eshkol_gpu_free(&bb);
    eshkol_gpu_free(&bc);
    return (rc == 0) ? out : NULL;
}
#else
static inline VmTensor* vm_gpu_try_matmul(VmRegionStack* rs,
                                            const VmTensor* a, const VmTensor* b) {
    (void)rs; (void)a; (void)b;
    return NULL; /* No GPU — fall to CPU */
}
#endif

#endif /* VM_GPU_DISPATCH_H */
