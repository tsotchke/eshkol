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

#endif /* VM_GPU_DISPATCH_H */
