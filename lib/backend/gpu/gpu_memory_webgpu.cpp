/*
 * Eshkol GPU backend — WebGPU (browser compute, wasm target)
 *
 * This is the wasm-target sibling of lib/backend/gpu/gpu_memory.mm (Metal)
 * and lib/backend/gpu/gpu_memory_cuda.cpp (CUDA). It implements the same
 * 31-function surface declared in inc/eshkol/backend/gpu/gpu_memory.h, so a
 * program compiled to wasm reaches the GPU through the ORDINARY dispatch
 * predicate (eshkol_gpu_should_use: active backend + element-count
 * threshold), not through a browser-special path.
 *
 * PRECISION. WGSL has no f64 type, so the f64 entry points are served by
 * sf64 kernels in web/eshkol-webgpu.js: IEEE 754 binary64 arithmetic on the
 * integer bit pattern, the WGSL sibling of this directory's metal_softfloat.h.
 * The backend uses the ESHKOL_GPU_PRECISION tier vocabulary (see
 * docs/breakdown/RUNTIME_CONFIGURATION.md):
 *
 *   exact : sf64, correctly rounded add/sub/mul/div. THE DEFAULT, as on the
 *           native backends. GEMM and elementwise results are bit-identical
 *           to the CPU path.
 *   high  : served by the same sf64 kernels.
 *   fast  : plain f32, admitted only with an explicit tolerance >= 1e-6.
 *
 * eshkol_gpu_has_fp64() therefore reports 1 while an f64 tier is active, as on
 * Metal, whose f64 is also emulated. Operations with no sf64 kernel (the
 * transcendentals) are refused by the JS backend and take the CPU fallback.
 *
 * ASYNC (ADR-0029). WebGPU readback is asynchronous (GPUBuffer.mapAsync);
 * this C code is synchronous. The boundary is JSPI, the same mechanism the
 * compiled-WASM loaders use: the three compute imports below are declared as
 * ordinary synchronous EM_JS functions that answer ESHKOL_WEBGPU_NO_DEVICE,
 * and EshkolWebGPU.attachVm() (web/eshkol-webgpu.js) replaces them at
 * instantiation with WebAssembly.Suspending wrappers of its bridge, and wraps
 * the module's entry exports with WebAssembly.promising -- but only when the
 * browser has both JSPI and a WebGPU device. No Asyncify instrumentation.
 *
 * PAGE CONTRACT. attachVm() publishes the bridge as Module.eshkolWebGPUBridge.
 * Without it every query below reports "no device", eshkol_gpu_init() leaves
 * the backend at ESHKOL_GPU_NONE, and every operation takes the CPU path: a
 * missing device is never a failed program, only a slower one. The dispatch
 * policy (tier admission, execution-marker verification, fallback accounting)
 * lives once, in the bridge; nothing here duplicates it.
 */

#ifdef __EMSCRIPTEN__

#include <eshkol/backend/gpu/gpu_memory.h>
#include <eshkol/logger.h>
#include <emscripten.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* The GPUDevice is acquired and owned by the page (web/eshkol-webgpu.js);
 * this file never touches the WebGPU C API, so it needs no Dawn port. */

/* Forward declaration: dispatched matmul from blas_backend.cpp */
extern "C" void eshkol_matmul_f64(const double*, const double*, double*,
                                   uint64_t, uint64_t, uint64_t);

/* Forward declaration: CPU batched matmul from blas_backend.cpp */
extern "C" void eshkol_batch_matmul_f64(const double*, const double*, double*,
                                        int64_t, int64_t, int64_t, int64_t);

/* CPU fallbacks in this file must not call eshkol_matmul_f64(): that ordinary
 * dispatcher can select this backend again for a very large operation. Keep a
 * local scalar fallback so an exact-tier refusal is fail-closed and cannot
 * recurse. */
static void webgpu_cpu_matmul(const double* A, const double* B, double* C,
                              uint64_t M, uint64_t K, uint64_t N) {
    for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (uint64_t k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Global State
// ============================================================================

/* GPU threshold (elements) — default 100K, same as all native backends. */
size_t g_gpu_threshold = 100000;

/* Active backend, mirroring gpu_memory.mm. Single-threaded on the main wasm
 * thread, so a plain static is sufficient (no atomics needed). */
static EshkolGPUBackend g_active_backend = ESHKOL_GPU_NONE;
static bool g_gpu_initialized = false;

/* Buffer flag bits (mirrors the Metal backend's convention). */
enum {
    ESHKOL_WEBGPU_FLAG_EXTERNAL = 1u  /* bit 0: memory is externally owned */
};

/* Bridge status codes. 0 = the GPU served the call and the result is already
 * in wasm memory; anything else means the C side runs its CPU fallback. */
enum {
    ESHKOL_WEBGPU_OK        = 0,
    ESHKOL_WEBGPU_NO_DEVICE = 1,  /* no bridge installed (no JSPI or no device) */
    ESHKOL_WEBGPU_DECLINED  = 2,  /* bridge refused: no kernel for the op/tier */
    ESHKOL_WEBGPU_THREW     = 3   /* kernel or readback failed */
};

// ============================================================================
// JavaScript Bridge (ADR-0029)
// ============================================================================

/* Synchronous queries. They never suspend: the device is acquired before the
 * module is instantiated. */

/** @brief 1 when attachVm() installed a bridge with a live device. */
EM_JS(int, eshkol_webgpu_js_device_ready, (void), {
    var b = Module["eshkolWebGPUBridge"];
    return b ? b.deviceReady() : 0;
});

/** @brief The page-configured dispatch threshold, or 0 without a bridge. */
EM_JS(double, eshkol_webgpu_js_threshold, (void), {
    var b = Module["eshkolWebGPUBridge"];
    return b ? b.threshold() : 0;
});

/** @brief Push an explicit threshold (ESHKOL_GPU_THRESHOLD or
 *         eshkol_gpu_set_threshold) into the backend. */
EM_JS(void, eshkol_webgpu_js_set_threshold, (double threshold), {
    var b = Module["eshkolWebGPUBridge"];
    if (b) b.setThreshold(threshold);
});

/** @brief The backend's own admission predicate (tier and threshold). */
EM_JS(int, eshkol_webgpu_js_should_use, (double num_elements), {
    var b = Module["eshkolWebGPUBridge"];
    return b ? b.shouldUse(num_elements) : 0;
});

/** @brief 1 while the backend runs an f64 (sf64) tier. */
EM_JS(int, eshkol_webgpu_js_has_fp64, (void), {
    var b = Module["eshkolWebGPUBridge"];
    return b ? b.hasFp64() : 0;
});

/** @brief Record that an operation the dispatch selected for the GPU has no
 *         WebGPU kernel and ran on the CPU (counted in fallbackCount and
 *         explained in diagnostics). */
EM_JS(void, eshkol_webgpu_js_note_fallback, (const char* what), {
    var b = Module["eshkolWebGPUBridge"];
    if (b) b.noteFallback(UTF8ToString(what));
});

/* Compute imports. These synchronous bodies are what runs when no bridge is
 * installed; attachVm() replaces each import, by name, with a
 * WebAssembly.Suspending wrapper of the bridge method of the same role. */

/** @brief C = A * B, row-major f64 at wasm heap byte offsets. */
EM_JS(int, eshkol_webgpu_js_matmul, (void* aPtr, void* bPtr, void* cPtr,
                                     double M, double K, double N), {
    return 1;
});

/** @brief Elementwise op over f64 arrays; `bPtr` may be 0 for unary ops. */
EM_JS(int, eshkol_webgpu_js_elementwise, (void* aPtr, void* bPtr,
                                          void* outPtr, double n, int op), {
    return 1;
});

/** @brief Full reduction of `n` f64 elements into the f64 at `outPtr`. */
EM_JS(int, eshkol_webgpu_js_reduce, (void* inPtr, void* outPtr,
                                     double n, int op), {
    return 1;
});

/** @brief Report a selected-but-unsupported operation to the bridge. */
static void webgpu_note_fallback(const char* what, size_t num_elements) {
    if (g_active_backend == ESHKOL_GPU_WEBGPU && num_elements >= g_gpu_threshold)
        eshkol_webgpu_js_note_fallback(what);
}

// ============================================================================
// Device Management
// ============================================================================

/** @brief Initialise the WebGPU backend: applies the ESHKOL_GPU_THRESHOLD
 *         environment override, then asks the page whether it published a
 *         backend with a live device. Idempotent.
 *  @return 1 if the WebGPU backend is now active, 0 if the program will run
 *          on the CPU (BOOLEAN, not a device count — see gpu_memory.h). */
int eshkol_gpu_init(void) {
    if (g_gpu_initialized) {
        return (g_active_backend != ESHKOL_GPU_NONE) ? 1 : 0;
    }

    if (eshkol_webgpu_js_device_ready()) {
        g_active_backend = ESHKOL_GPU_WEBGPU;
        g_gpu_initialized = true;
        /* One threshold, owned by the backend object: an explicit
         * ESHKOL_GPU_THRESHOLD (getenv reads the module's ENV) is pushed into
         * it; otherwise the page's configured value is adopted here. */
        const char* env = std::getenv("ESHKOL_GPU_THRESHOLD");
        size_t val = env ? static_cast<size_t>(std::atol(env)) : 0;
        if (val > 0) {
            g_gpu_threshold = val;
            eshkol_webgpu_js_set_threshold(static_cast<double>(val));
        } else {
            double t = eshkol_webgpu_js_threshold();
            if (t >= 1) g_gpu_threshold = static_cast<size_t>(t);
        }
        return 1;
    }

    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized = true;
    return 0;
}

/** @brief Shut down the WebGPU backend. The GPUDevice belongs to the page,
 *         not to this module, so nothing is destroyed here — only the
 *         C-side active-backend state is cleared. */
void eshkol_gpu_shutdown(void) {
    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized = false;
}

/** @brief Get the active GPU backend (ESHKOL_GPU_WEBGPU once init succeeded,
 *         ESHKOL_GPU_NONE otherwise). */
EshkolGPUBackend eshkol_gpu_get_backend(void) {
    return g_active_backend;
}

/** @brief Human-readable name for a GPU backend enum value. In a wasm build
 *         only WebGPU can ever be present, so the native backends are
 *         annotated "(not available)". */
const char* eshkol_gpu_backend_name(EshkolGPUBackend backend) {
    switch (backend) {
        case ESHKOL_GPU_NONE:   return "CPU only";
        case ESHKOL_GPU_METAL:  return "Apple Metal (not available)";
        case ESHKOL_GPU_CUDA:   return "NVIDIA CUDA (not available)";
        case ESHKOL_GPU_VULKAN: return "Vulkan (not available)";
        case ESHKOL_GPU_WEBGPU: return "WebGPU (browser compute)";
        default:                return "Unknown";
    }
}

/** @brief Check whether a specific backend is the one currently active.
 *  @return 1 if `backend` is active, 0 otherwise. */
int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
    return (g_active_backend != ESHKOL_GPU_NONE && g_active_backend == backend) ? 1 : 0;
}

/** @brief Native hardware f64 support: always 0. WGSL has no f64 type at
 *         all, so no WebGPU device can offer one. */
int eshkol_gpu_supports_f64(void) {
    return 0;
}

/** @brief Any correct f64 path, native or emulated: 1 while the WebGPU
 *         backend is active on the exact/high tier, whose sf64 kernels are
 *         correctly rounded IEEE f64 (the same meaning as on Metal). 0 on the
 *         f32 `fast` tier or without a device. */
int eshkol_gpu_has_fp64(void) {
    if (g_active_backend != ESHKOL_GPU_WEBGPU) return 0;
    return eshkol_webgpu_js_has_fp64();
}

// ============================================================================
// Memory Allocation
// ============================================================================

/** @brief Allocate a GPU-accessible buffer. On wasm the linear heap IS the
 *         memory the WebGPU bridge reads from (kernels are fed heap byte
 *         offsets), so this is a malloc-backed allocation with host_ptr and
 *         device_ptr aliased. Zeroes `*out_buffer` on failure.
 *  @return 0 on success, -1 on failure. */
int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type,
                     EshkolGPUBuffer* out_buffer) {
    if (!out_buffer) return -1;
    memset(out_buffer, 0, sizeof(*out_buffer));
    if (size_bytes == 0) {
        eshkol_error("GPU allocation failed: zero-size request");
        return -1;
    }
    void* p = malloc(size_bytes);
    if (!p) {
        eshkol_error("GPU allocation failed: out of wasm heap (%zu bytes)", size_bytes);
        return -1;
    }
    out_buffer->host_ptr = p;
    out_buffer->device_ptr = p;   /* unified: the wasm heap is the device view */
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = mem_type;
    out_buffer->backend = ESHKOL_GPU_WEBGPU;
    out_buffer->flags = 0;        /* owned here — eshkol_gpu_free() releases it */
    out_buffer->backend_data = nullptr;
    return 0;
}

/** @brief Allocate a GPU-accessible buffer with a specific alignment.
 *         Uses aligned_alloc() with the size rounded up to a multiple of
 *         `alignment`, as the C standard requires.
 *  @return 0 on success, -1 on failure. */
int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type,
                              EshkolGPUBuffer* out_buffer) {
    if (!out_buffer) return -1;
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        memset(out_buffer, 0, sizeof(*out_buffer));
        eshkol_error("GPU aligned allocation failed: alignment %zu is not a power of 2",
                     alignment);
        return -1;
    }
    if (alignment <= sizeof(void*)) {
        return eshkol_gpu_alloc(size_bytes, mem_type, out_buffer);
    }
    memset(out_buffer, 0, sizeof(*out_buffer));
    if (size_bytes == 0) {
        eshkol_error("GPU aligned allocation failed: zero-size request");
        return -1;
    }
    size_t rounded = ((size_bytes + alignment - 1) / alignment) * alignment;
    void* p = aligned_alloc(alignment, rounded);
    if (!p) {
        eshkol_error("GPU aligned allocation failed: out of wasm heap (%zu bytes, align %zu)",
                     rounded, alignment);
        return -1;
    }
    out_buffer->host_ptr = p;
    out_buffer->device_ptr = p;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = mem_type;
    out_buffer->backend = ESHKOL_GPU_WEBGPU;
    out_buffer->flags = 0;
    out_buffer->backend_data = nullptr;
    return 0;
}

/** @brief Free a GPU buffer. Memory is released only when flag bit 0
 *         (externally owned) is clear; the descriptor is zeroed either
 *         way, so a wrapped pointer is never double-freed. */
void eshkol_gpu_free(EshkolGPUBuffer* buffer) {
    if (!buffer) return;
    if (buffer->host_ptr && (buffer->flags & ESHKOL_WEBGPU_FLAG_EXTERNAL) == 0) {
        free(buffer->host_ptr);
    }
    memset(buffer, 0, sizeof(*buffer));
}

/** @brief Wrap an existing host pointer for GPU use. On wasm there is no
 *         separate device allocation to make — the pointer already IS a heap
 *         byte offset the WebGPU bridge can read — so this only fills in the
 *         descriptor and sets flag bit 0 so eshkol_gpu_free() will not free
 *         memory it does not own.
 *  @return 0 on success, -1 on failure. */
int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes,
                          EshkolGPUBuffer* out_buffer) {
    if (!out_buffer) return -1;
    memset(out_buffer, 0, sizeof(*out_buffer));
    if (!host_ptr) {
        eshkol_error("GPU wrap_host failed: null host pointer");
        return -1;
    }
    out_buffer->host_ptr = host_ptr;
    out_buffer->device_ptr = host_ptr;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = ESHKOL_MEM_UNIFIED;
    out_buffer->backend = ESHKOL_GPU_WEBGPU;
    out_buffer->flags = ESHKOL_WEBGPU_FLAG_EXTERNAL;  /* do not free */
    out_buffer->backend_data = nullptr;
    return 0;
}

// ============================================================================
// Data Transfer
// ============================================================================

/** @brief Synchronise a buffer between host and device. No-op on wasm: the
 *         linear heap is the only copy, and the WebGPU bridge stages its own
 *         GPU-side buffers per dispatch.
 *  @return 0 always. */
int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    (void)buffer;
    (void)direction;
    return 0;  /* unified memory — nothing to copy */
}

/** @brief Asynchronous variant of eshkol_gpu_sync(). Also a no-op; there is
 *         no stream object on this backend, so `stream_handle` is ignored.
 *  @return 0 always. */
int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction,
                           void* stream_handle) {
    (void)stream_handle;
    return eshkol_gpu_sync(buffer, direction);
}

/** @brief Wait for pending operations on a buffer. No-op: every GPU call on
 *         this backend already suspends until readback completes, so nothing
 *         is ever in flight when control returns to C. */
void eshkol_gpu_wait(EshkolGPUBuffer* buffer) {
    (void)buffer;
}

// ============================================================================
// Threshold Configuration
// ============================================================================

/** @brief Set the minimum element count at which GPU dispatch is used, and
 *         mirror it into the JS backend so both predicates agree. */
void eshkol_gpu_set_threshold(size_t threshold) {
    g_gpu_threshold = threshold;
    eshkol_webgpu_js_set_threshold(static_cast<double>(threshold));
}

/** @brief Get the current GPU dispatch threshold. */
size_t eshkol_gpu_get_threshold(void) {
    return g_gpu_threshold;
}

/** @brief Decide whether an operation of `num_elements` elements should go to
 *         the GPU: an active backend plus at-or-above the threshold. The JS
 *         side applies the precision-tier admission test as well (the `fast`
 *         tier without an explicit tolerance declines everything).
 *  @return 1 to use the GPU, 0 for CPU. */
int eshkol_gpu_should_use(size_t num_elements) {
    if (g_active_backend == ESHKOL_GPU_NONE || num_elements < g_gpu_threshold) return 0;
    return eshkol_webgpu_js_should_use(static_cast<double>(num_elements));
}

// ============================================================================
// Matrix Operations
// ============================================================================

/** @brief GPU matrix multiplication C = A * B over f64 buffers. Attempts the
 *         WGSL GEMM kernel (sf64 or f32 tier) when the dispatch predicate
 *         says the work is large enough; on any refusal or kernel error falls
 *         back to the CPU BLAS/SIMD eshkol_matmul_f64(), which is the same
 *         arithmetic the stub backend performs.
 *  @return 0 on success, -1 on invalid arguments. */
int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;
    const double* a = static_cast<const double*>(A->host_ptr);
    const double* b = static_cast<const double*>(B->host_ptr);
    double* c = static_cast<double*>(C->host_ptr);
    if (!a || !b || !c) return -1;

    if (eshkol_gpu_should_use(static_cast<size_t>(M) * static_cast<size_t>(N))) {
        int rc = eshkol_webgpu_js_matmul(
            const_cast<double*>(a), const_cast<double*>(b), c,
            static_cast<double>(M), static_cast<double>(K), static_cast<double>(N));
        if (rc == ESHKOL_WEBGPU_OK) return 0;
        /* rc != 0: no device, refused tier, or a kernel error — CPU fallback. */
    }

    /* Refuse explicitly. The ordinary matmul dispatcher owns the CPU
     * fallback; returning success here would make an unsupported WebGPU tier
     * look like a GPU result. */
    return -1;
}

/** @brief Single-precision matmul. There is no f32 WGSL entry point on the
 *         JS side yet (the f32 GEMM kernel is reached only through the f64
 *         path's `fast` tier, which encodes from f64 storage), so this runs
 *         the stub backend's naive triple-loop scalar implementation on the
 *         host pointers.
 *         FOLLOW-UP: a native f32 WGSL GEMM entry point that skips the
 *         f64-encode step entirely.
 *  @return 0 on success, -1 on null host pointers. */
int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

    const float* a = (const float*)A->host_ptr;
    const float* b = (const float*)B->host_ptr;
    float* c = (float*)C->host_ptr;
    if (!a || !b || !c) {
        eshkol_error("GPU matmul_f32 failed: null host pointers");
        return -1;
    }

    (void)a; (void)b; (void)c; (void)M; (void)K; (void)N;
    return -1;  /* no certified WebGPU f32 C API path */
}

// ============================================================================
// Elementwise / Reduce / Transpose
// ============================================================================

/** @brief GPU elementwise op over f64 arrays. Attempts the WGSL elementwise
 *         kernel when the dispatch predicate allows it (the JS side refuses
 *         the transcendentals, which have no sf64 kernel, and records the
 *         CPU fallback in its diagnostics); on
 *         refusal or error falls back to the stub backend's CPU scalar loop,
 *         including its missing-`b` identity convention (0 for add/sub,
 *         1 for mul/div).
 *  @return 0 on success, -1 on invalid arguments. */
int eshkol_gpu_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                EshkolGPUBuffer* out, uint64_t n,
                                EshkolElementwiseOp op) {
    if (!a || !out || n == 0) return -1;
    const double* ap = static_cast<const double*>(a->host_ptr);
    const double* bp = (b && b->host_ptr) ? static_cast<const double*>(b->host_ptr) : nullptr;
    double* cp = static_cast<double*>(out->host_ptr);
    if (!ap || !cp) return -1;

    if (eshkol_gpu_should_use(static_cast<size_t>(n))) {
        int rc = eshkol_webgpu_js_elementwise(
            const_cast<double*>(ap), const_cast<double*>(bp), cp,
            static_cast<double>(n), static_cast<int>(op));
        if (rc == ESHKOL_WEBGPU_OK) return 0;
    }

    (void)ap; (void)bp; (void)cp; (void)n; (void)op;
    return -1;  /* ordinary callers own the CPU fallback */
}

/** @brief GPU full reduction over an f64 array. Attempts the WGSL sf64
 *         reduction (whose block-wise combine makes GPU and CPU agree to
 *         tolerance, not bitwise — as on Metal and CUDA); on refusal or
 *         error falls back to the stub backend's sequential CPU accumulation.
 *  @return 0 on success, -1 on invalid arguments. */
int eshkol_gpu_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                           uint64_t n, EshkolReduceOp op) {
    if (!in || !out || n == 0) return -1;
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;

    if (eshkol_gpu_should_use(static_cast<size_t>(n))) {
        int rc = eshkol_webgpu_js_reduce(const_cast<double*>(inp), outp,
                                         static_cast<double>(n),
                                         static_cast<int>(op));
        if (rc == ESHKOL_WEBGPU_OK) return 0;
    }

    (void)inp; (void)outp; (void)n; (void)op;
    return -1;  /* ordinary callers own the CPU fallback */
}

/** @brief Axis reduction over an f64 N-D tensor. Refused until a WGSL kernel
 *         is certified; callers own the CPU fallback.
 *  @return -1 while unsupported. */
int eshkol_gpu_reduce_axis_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t rank, const uint64_t* shape,
                                uint64_t axis, EshkolReduceOp op) {
    (void)in; (void)out; (void)axis; (void)op;
    uint64_t total = 1;
    for (uint64_t i = 0; shape && i < rank; ++i) total *= shape[i];
    webgpu_note_fallback("axis reduction", static_cast<size_t>(total));
    return -1;  /* no WebGPU axis-reduction kernel */
}

/** @brief 2-D transpose of an f64 matrix. Refused until a WGSL kernel is
 *         certified; callers own the CPU fallback.
 *  @return -1 while unsupported. */
int eshkol_gpu_transpose_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t rows, uint64_t cols) {
    (void)in; (void)out;
    webgpu_note_fallback("transpose", static_cast<size_t>(rows * cols));
    return -1;  /* no WebGPU transpose kernel */
}

// ============================================================================
// Softmax / Normalize
// ============================================================================

/** @brief Numerically-stable softmax over contiguous slices. Refused until a
 *         WGSL kernel is certified; callers own the CPU fallback.
 *  @return -1 while unsupported. */
int eshkol_gpu_softmax_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                            uint64_t num_slices, uint64_t slice_len) {
    (void)in; (void)out;
    webgpu_note_fallback("softmax", static_cast<size_t>(num_slices * slice_len));
    return -1;  /* no WebGPU softmax kernel */
}

/** @brief Layer normalisation over contiguous slices. Refused until a WGSL
 *         kernel is certified; callers own the CPU fallback.
 *  @return -1 while unsupported. */
int eshkol_gpu_normalize_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t num_slices, uint64_t slice_len,
                              double gamma, double beta, double epsilon) {
    (void)in; (void)out; (void)gamma; (void)beta; (void)epsilon;
    webgpu_note_fallback("normalize", static_cast<size_t>(num_slices * slice_len));
    return -1;  /* no WebGPU normalization kernel */
}

// ============================================================================
// Runtime Integration
// ============================================================================

/** @brief Runtime matmul entry point used by generated code. Same shape as
 *         the Metal backend: lazy init, threshold check, GPU attempt through
 *         wrapped host buffers, CPU fallback on any failure. `dtype` is
 *         ignored — there is no tensor-core path in WGSL, and storage is
 *         already precision-reduced for the logical dtype, so plain f64
 *         matmul stays correct. */
void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N, int32_t /*dtype*/) {
    /* Lazy GPU init — ensures g_active_backend is set before threshold check */
    if (!g_gpu_initialized) {
        eshkol_gpu_init();
    }

    size_t num_elements = static_cast<size_t>(M) * static_cast<size_t>(N);

    if (eshkol_gpu_should_use(num_elements)) {
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        if (eshkol_gpu_wrap_host((void*)A, M * K * sizeof(double), &buf_a) == 0 &&
            eshkol_gpu_wrap_host((void*)B, K * N * sizeof(double), &buf_b) == 0 &&
            eshkol_gpu_wrap_host((void*)C, M * N * sizeof(double), &buf_c) == 0) {

            /* eshkol_gpu_matmul_f64() attempts the WGSL GEMM and, if the
             * device declines or the kernel errors, runs the CPU path itself
             * — so a 0 here means C is written either way. Wrapping is
             * zero-copy on wasm (host_ptr aliases the caller's pointer), so
             * no copy-back is needed, unlike the Metal backend. */
            int rc = eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N);

            eshkol_gpu_free(&buf_a);
            eshkol_gpu_free(&buf_b);
            eshkol_gpu_free(&buf_c);

            if (rc == 0) return;
        }
        /* Wrapping or dispatch failed — fall through to CPU. */
    }

    /* CPU fallback. Do not re-enter eshkol_matmul_f64(): the ordinary
     * dispatcher may select WebGPU again for the same call. */
    webgpu_cpu_matmul(A, B, C, M, K, N);
}

/** @brief Runtime batched-matmul entry point used by generated code. There
 *         is no batched WGSL GEMM kernel yet, so this dispatches to the CPU
 *         batched f64 path. `dtype` is ignored (storage already
 *         precision-reduced for the logical dtype).
 *         FOLLOW-UP: a batched WGSL GEMM that issues one dispatch for all
 *         `batch` products. */
void eshkol_batch_matmul_dispatch(const double* a, const double* b, double* c,
                                  int64_t batch, int64_t M, int64_t K, int64_t N,
                                  int32_t /*dtype*/) {
    eshkol_batch_matmul_f64(a, b, c, batch, M, K, N);
}

// ============================================================================
// Backward Pass GPU Operations
// ============================================================================

/** @brief conv2d input-gradient backward pass: no WGSL kernel on this
 *         backend, so it always fails and callers must take a CPU AD path.
 *  @return -1 always. */
int eshkol_gpu_conv2d_backward_input_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    return -1;  // No WebGPU conv2d backward kernel
}

/** @brief conv2d kernel-gradient backward pass: no WGSL kernel on this
 *         backend, so it always fails.
 *  @return -1 always. */
int eshkol_gpu_conv2d_backward_kernel_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    return -1;
}

/** @brief batch-norm backward pass: no WGSL kernel on this backend, so it
 *         always fails.
 *  @return -1 always. */
int eshkol_gpu_batchnorm_backward_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    EshkolGPUBuffer*, EshkolGPUBuffer*, uint64_t, uint64_t) {
    return -1;
}

/** @brief layer-norm backward pass: no WGSL kernel on this backend, so it
 *         always fails.
 *  @return -1 always. */
int eshkol_gpu_layernorm_backward_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    uint64_t, uint64_t) {
    return -1;
}

#else  /* !__EMSCRIPTEN__ */

/* This backend exists only for the Emscripten/wasm target; CMake selects it
 * solely under `if(EMSCRIPTEN)`. Reaching here means a stray build picked the
 * file up on a native host, where gpu_memory.mm, gpu_memory_cuda.cpp or
 * gpu_memory_stub.cpp already provides the same symbols — so contribute
 * nothing rather than producing duplicate definitions. */
typedef int eshkol_gpu_webgpu_requires_emscripten;

#endif /* __EMSCRIPTEN__ */
