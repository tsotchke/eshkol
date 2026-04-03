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
/* GPU elementwise binary: add(0), sub(1), mul(2), div(3) */
extern int eshkol_gpu_elementwise_f64(void* a, void* b, void* out, uint64_t n, int op);

static inline VmTensor* vm_gpu_try_binary(VmRegionStack* rs,
                                            const VmTensor* a, const VmTensor* b, int gpu_op) {
    if (!vm_gpu_should_dispatch(a->total)) return NULL;
    if (a->total != b->total) return NULL; /* No broadcast on GPU */
    if (gpu_op < 0) return NULL;

    VmTensor* out = vm_tensor_zeros(rs, a->shape, a->n_dims);
    if (!out) return NULL;

    VmGpuBuf ba, bb, bc;
    if (eshkol_gpu_wrap_host(a->data, a->total*8, &ba) != 0 ||
        eshkol_gpu_wrap_host(b->data, b->total*8, &bb) != 0 ||
        eshkol_gpu_wrap_host(out->data, out->total*8, &bc) != 0) return NULL;

    int rc = eshkol_gpu_elementwise_f64(&ba, &bb, &bc, a->total, gpu_op);
    if (rc == 0 && bc.host != (void*)out->data) memcpy(out->data, bc.host, out->total*8);
    eshkol_gpu_free(&ba); eshkol_gpu_free(&bb); eshkol_gpu_free(&bc);
    return (rc == 0) ? out : NULL;
}

/* GPU reduce: sum(0), min(2), max(3), mean(4) */
extern int eshkol_gpu_reduce_f64(void* in, void* out, uint64_t n, int op);

static inline double vm_gpu_try_reduce(const VmTensor* t, int gpu_op) {
    if (!vm_gpu_should_dispatch(t->total)) return NAN;

    double result = 0;
    VmGpuBuf bi, bo;
    if (eshkol_gpu_wrap_host(t->data, t->total*8, &bi) != 0 ||
        eshkol_gpu_wrap_host(&result, 8, &bo) != 0) return NAN;

    int rc = eshkol_gpu_reduce_f64(&bi, &bo, t->total, gpu_op);
    if (rc == 0 && bo.host != (void*)&result) memcpy(&result, bo.host, 8);
    eshkol_gpu_free(&bi); eshkol_gpu_free(&bo);
    return (rc == 0) ? result : NAN;
}

/* GPU softmax */
extern int eshkol_gpu_softmax_f64(void* in, void* out, uint64_t num_slices, uint64_t slice_len);

static inline VmTensor* vm_gpu_try_softmax(VmRegionStack* rs, const VmTensor* t) {
    if (!vm_gpu_should_dispatch(t->total)) return NULL;
    int64_t slice_len = t->shape[t->n_dims - 1];
    int64_t num_slices = t->total / slice_len;

    VmTensor* out = vm_tensor_zeros(rs, t->shape, t->n_dims);
    if (!out) return NULL;

    VmGpuBuf bi, bo;
    if (eshkol_gpu_wrap_host(t->data, t->total*8, &bi) != 0 ||
        eshkol_gpu_wrap_host(out->data, out->total*8, &bo) != 0) return NULL;

    int rc = eshkol_gpu_softmax_f64(&bi, &bo, num_slices, slice_len);
    if (rc == 0 && bo.host != (void*)out->data) memcpy(out->data, bo.host, out->total*8);
    eshkol_gpu_free(&bi); eshkol_gpu_free(&bo);
    return (rc == 0) ? out : NULL;
}

#else
/* No GPU — all try functions return NULL, fall to CPU */
static inline VmTensor* vm_gpu_try_matmul(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    (void)rs; (void)a; (void)b; return NULL;
}
static inline VmTensor* vm_gpu_try_binary(VmRegionStack* rs, const VmTensor* a, const VmTensor* b, int op) {
    (void)rs; (void)a; (void)b; (void)op; return NULL;
}
static inline double vm_gpu_try_reduce(const VmTensor* t, int op) {
    (void)t; (void)op; return NAN;
}
static inline VmTensor* vm_gpu_try_softmax(VmRegionStack* rs, const VmTensor* t) {
    (void)rs; (void)t; return NULL;
}
#endif

#endif /* VM_GPU_DISPATCH_H */
