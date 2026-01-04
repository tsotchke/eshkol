/*
 * Cross-Platform GPU Memory Implementation for Eshkol
 *
 * Provides unified GPU acceleration across:
 * - Metal (macOS/iOS) with unified memory
 * - CUDA (Linux/Windows) with discrete GPU memory
 * - Fallback to CPU when no GPU available
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/gpu/gpu_memory.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

// ============================================================================
// Platform Detection
// ============================================================================

// Metal (macOS/iOS with Apple GPU)
#if defined(__APPLE__) && defined(ESHKOL_GPU_METAL_ENABLED)
#define ESHKOL_GPU_METAL_AVAILABLE 1
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

// CUDA (NVIDIA GPU)
#if defined(ESHKOL_GPU_CUDA_ENABLED)
#define ESHKOL_GPU_CUDA_AVAILABLE 1
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// ============================================================================
// Global State
// ============================================================================

// GPU threshold (elements) - default 100K
size_t g_gpu_threshold = 100000;

// Active backend
static EshkolGPUBackend g_active_backend = ESHKOL_GPU_NONE;
static bool g_gpu_initialized = false;

// ============================================================================
// Metal Backend State
// ============================================================================

#if ESHKOL_GPU_METAL_AVAILABLE

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_metal_queue = nil;
static id<MTLComputePipelineState> g_matmul_f64_pipeline = nil;
static id<MTLLibrary> g_metal_library = nil;
static bool g_metal_unified_memory = false;

// Metal shader for f64 matrix multiplication (MPS only supports f32)
static NSString* g_matmul_f64_source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f64(
    device const double* A [[buffer(0)]],
    device const double* B [[buffer(1)]],
    device double* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    double sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
)";

static int metal_init(void) {
    @autoreleasepool {
        g_metal_device = MTLCreateSystemDefaultDevice();
        if (!g_metal_device) return -1;

        g_metal_queue = [g_metal_device newCommandQueue];
        if (!g_metal_queue) {
            g_metal_device = nil;
            return -1;
        }

        // Check for unified memory (Apple Silicon)
        if (@available(macOS 10.15, *)) {
            g_metal_unified_memory = [g_metal_device supportsFamily:MTLGPUFamilyApple1];
        }

        // Compile f64 matmul shader with Metal 2.3+ for 64-bit support
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 11.0, *)) {
            options.languageVersion = MTLLanguageVersion2_3;  // Required for 64-bit types
        }
        // Try to compile f64 shader - this will fail on most Metal devices since
        // Metal doesn't support double precision. This is expected behavior.
        // When it fails, we silently fall back to BLAS/SIMD implementations.
        g_metal_library = [g_metal_device newLibraryWithSource:g_matmul_f64_source
                                                       options:options
                                                         error:&error];
        if (g_metal_library) {
            id<MTLFunction> func = [g_metal_library newFunctionWithName:@"matmul_f64"];
            if (func) {
                g_matmul_f64_pipeline = [g_metal_device newComputePipelineStateWithFunction:func
                                                                                      error:&error];
                // Silent failure - f64 not supported on this GPU, will use BLAS fallback
            }
            // Silent failure - f64 not supported, will use BLAS fallback
        }
        // Silent failure - Metal f64 not supported (expected), will use BLAS fallback

        return 0;
    }
}

static void metal_shutdown(void) {
    @autoreleasepool {
        g_matmul_f64_pipeline = nil;
        g_metal_library = nil;
        g_metal_queue = nil;
        g_metal_device = nil;
    }
}

static int metal_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out) {
    @autoreleasepool {
        MTLResourceOptions options;
        switch (mem_type) {
            case ESHKOL_MEM_DEVICE:
                options = MTLResourceStorageModePrivate;
                break;
            case ESHKOL_MEM_HOST_PINNED:
                options = g_metal_unified_memory ? MTLResourceStorageModeShared
                                                  : MTLResourceStorageModeManaged;
                break;
            default:
                options = MTLResourceStorageModeShared;
                break;
        }

        id<MTLBuffer> buffer = [g_metal_device newBufferWithLength:size_bytes options:options];
        if (!buffer) return -1;

        out->size_bytes = size_bytes;
        out->mem_type = mem_type;
        out->backend = ESHKOL_GPU_METAL;
        out->backend_data = (__bridge_retained void*)buffer;

        if (options != MTLResourceStorageModePrivate) {
            out->host_ptr = [buffer contents];
            out->device_ptr = out->host_ptr;
        } else {
            out->host_ptr = nullptr;
            out->device_ptr = (__bridge void*)buffer;
        }

        return 0;
    }
}

static void metal_free(EshkolGPUBuffer* buffer) {
    @autoreleasepool {
        if (buffer->backend_data) {
            id<MTLBuffer> mtl_buf = (__bridge_transfer id<MTLBuffer>)buffer->backend_data;
            mtl_buf = nil;
        }
    }
}

static int metal_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    @autoreleasepool {
        // Unified memory: no sync needed
        if (g_metal_unified_memory && buffer->mem_type != ESHKOL_MEM_DEVICE) {
            return 0;
        }

        id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buffer->backend_data;

        if (direction == ESHKOL_SYNC_HOST_TO_DEVICE || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            if (@available(macOS 10.11, *)) {
                [mtl_buf didModifyRange:NSMakeRange(0, buffer->size_bytes)];
            }
        }

        if (direction == ESHKOL_SYNC_DEVICE_TO_HOST || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            if (@available(macOS 10.11, *)) {
                [blit synchronizeResource:mtl_buf];
            }
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        return 0;
    }
}

static int metal_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        if (!g_matmul_f64_pipeline) return -1;

        id<MTLBuffer> buf_a = (__bridge id<MTLBuffer>)A->backend_data;
        id<MTLBuffer> buf_b = (__bridge id<MTLBuffer>)B->backend_data;
        id<MTLBuffer> buf_c = (__bridge id<MTLBuffer>)C->backend_data;

        // Create dimension buffers
        uint32_t dims[3] = {(uint32_t)M, (uint32_t)K, (uint32_t)N};
        id<MTLBuffer> dim_m = [g_metal_device newBufferWithBytes:&dims[0] length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> dim_k = [g_metal_device newBufferWithBytes:&dims[1] length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> dim_n = [g_metal_device newBufferWithBytes:&dims[2] length:4 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_matmul_f64_pipeline];
        [encoder setBuffer:buf_a offset:0 atIndex:0];
        [encoder setBuffer:buf_b offset:0 atIndex:1];
        [encoder setBuffer:buf_c offset:0 atIndex:2];
        [encoder setBuffer:dim_m offset:0 atIndex:3];
        [encoder setBuffer:dim_k offset:0 atIndex:4];
        [encoder setBuffer:dim_n offset:0 atIndex:5];

        // Dispatch threads
        MTLSize grid = MTLSizeMake(N, M, 1);
        NSUInteger w = g_matmul_f64_pipeline.threadExecutionWidth;
        NSUInteger h = g_matmul_f64_pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroup = MTLSizeMake(w, h, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

static int metal_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        id<MTLBuffer> buf_a = (__bridge id<MTLBuffer>)A->backend_data;
        id<MTLBuffer> buf_b = (__bridge id<MTLBuffer>)B->backend_data;
        id<MTLBuffer> buf_c = (__bridge id<MTLBuffer>)C->backend_data;

        // Use MPS for f32 (highly optimized)
        MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                           rowBytes:K * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc_c];

        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_metal_device
            transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K
            alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

static int metal_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [g_metal_device newBufferWithBytesNoCopy:host_ptr
                                                                 length:size_bytes
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!buffer) {
            // Fallback: allocate and copy
            int result = metal_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
            if (result != 0) return result;
            memcpy(out->host_ptr, host_ptr, size_bytes);
            return 0;
        }

        out->host_ptr = host_ptr;
        out->device_ptr = host_ptr;
        out->size_bytes = size_bytes;
        out->mem_type = ESHKOL_MEM_UNIFIED;
        out->backend = ESHKOL_GPU_METAL;
        out->flags = 1;  // Wrapped, don't free underlying memory
        out->backend_data = (__bridge_retained void*)buffer;

        return 0;
    }
}

#endif // ESHKOL_GPU_METAL_AVAILABLE

// ============================================================================
// CUDA Backend State
// ============================================================================

#if ESHKOL_GPU_CUDA_AVAILABLE

static cudaStream_t g_cuda_stream = nullptr;
static cublasHandle_t g_cublas_handle = nullptr;

static int cuda_init(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return -1;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) return -1;

    err = cudaStreamCreate(&g_cuda_stream);
    if (err != cudaSuccess) return -1;

    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(g_cuda_stream);
        return -1;
    }

    cublasSetStream(g_cublas_handle, g_cuda_stream);

    return 0;
}

static void cuda_shutdown(void) {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
    if (g_cuda_stream) {
        cudaStreamDestroy(g_cuda_stream);
        g_cuda_stream = nullptr;
    }
}

static int cuda_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out) {
    cudaError_t err;
    void* ptr = nullptr;

    switch (mem_type) {
        case ESHKOL_MEM_UNIFIED:
            err = cudaMallocManaged(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = ptr;
            break;

        case ESHKOL_MEM_DEVICE:
            err = cudaMalloc(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = nullptr;
            out->device_ptr = ptr;
            break;

        case ESHKOL_MEM_HOST_PINNED:
            err = cudaMallocHost(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = nullptr;  // Need separate device alloc for pinned
            break;

        default:
            err = cudaMallocManaged(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = ptr;
            break;
    }

    out->size_bytes = size_bytes;
    out->mem_type = mem_type;
    out->backend = ESHKOL_GPU_CUDA;
    out->backend_data = nullptr;
    out->flags = 0;

    return 0;
}

static void cuda_free(EshkolGPUBuffer* buffer) {
    switch (buffer->mem_type) {
        case ESHKOL_MEM_HOST_PINNED:
            if (buffer->host_ptr) cudaFreeHost(buffer->host_ptr);
            if (buffer->device_ptr && buffer->device_ptr != buffer->host_ptr) {
                cudaFree(buffer->device_ptr);
            }
            break;
        default:
            if (buffer->device_ptr) cudaFree(buffer->device_ptr);
            break;
    }
}

static int cuda_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (buffer->mem_type == ESHKOL_MEM_UNIFIED) {
        cudaStreamSynchronize(g_cuda_stream);
        return 0;
    }

    if (buffer->mem_type == ESHKOL_MEM_HOST_PINNED && buffer->device_ptr) {
        cudaError_t err;
        if (direction == ESHKOL_SYNC_HOST_TO_DEVICE || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            err = cudaMemcpyAsync(buffer->device_ptr, buffer->host_ptr, buffer->size_bytes,
                                   cudaMemcpyHostToDevice, g_cuda_stream);
            if (err != cudaSuccess) return -1;
        }
        if (direction == ESHKOL_SYNC_DEVICE_TO_HOST || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            err = cudaMemcpyAsync(buffer->host_ptr, buffer->device_ptr, buffer->size_bytes,
                                   cudaMemcpyDeviceToHost, g_cuda_stream);
            if (err != cudaSuccess) return -1;
        }
        cudaStreamSynchronize(g_cuda_stream);
    }

    return 0;
}

static int cuda_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N) {
    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS uses column-major, so we compute C = B^T * A^T = (A * B)^T
    // But since we want row-major C, we do: C^T = B * A (in cuBLAS terms)
    // which gives us row-major C = A * B
    cublasStatus_t status = cublasDgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        (const double*)B->device_ptr, (int)N,
        (const double*)A->device_ptr, (int)K,
        &beta,
        (double*)C->device_ptr, (int)N);

    cudaStreamSynchronize(g_cuda_stream);
    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

static int cuda_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        (const float*)B->device_ptr, (int)N,
        (const float*)A->device_ptr, (int)K,
        &beta,
        (float*)C->device_ptr, (int)N);

    cudaStreamSynchronize(g_cuda_stream);
    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

static int cuda_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out) {
    // Register host memory with CUDA
    cudaError_t err = cudaHostRegister(host_ptr, size_bytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        // Fallback: allocate and copy
        int result = cuda_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
        if (result != 0) return result;
        memcpy(out->host_ptr, host_ptr, size_bytes);
        return 0;
    }

    out->host_ptr = host_ptr;
    out->device_ptr = host_ptr;  // Managed access
    out->size_bytes = size_bytes;
    out->mem_type = ESHKOL_MEM_HOST_PINNED;
    out->backend = ESHKOL_GPU_CUDA;
    out->flags = 1;  // Wrapped
    out->backend_data = nullptr;

    return 0;
}

#endif // ESHKOL_GPU_CUDA_AVAILABLE

// ============================================================================
// Public API Implementation
// ============================================================================

extern "C" {

int eshkol_gpu_init(void) {
    if (g_gpu_initialized) {
        return (g_active_backend != ESHKOL_GPU_NONE) ? 1 : 0;
    }

    g_gpu_initialized = true;

    // Try Metal first (macOS)
#if ESHKOL_GPU_METAL_AVAILABLE
    if (metal_init() == 0) {
        g_active_backend = ESHKOL_GPU_METAL;
        return 1;
    }
#endif

    // Try CUDA
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (cuda_init() == 0) {
        g_active_backend = ESHKOL_GPU_CUDA;
        return 1;
    }
#endif

    // No GPU available
    g_active_backend = ESHKOL_GPU_NONE;
    return 0;
}

void eshkol_gpu_shutdown(void) {
#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        metal_shutdown();
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        cuda_shutdown();
    }
#endif

    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized = false;
}

EshkolGPUBackend eshkol_gpu_get_backend(void) {
    return g_active_backend;
}

const char* eshkol_gpu_backend_name(EshkolGPUBackend backend) {
    switch (backend) {
        case ESHKOL_GPU_NONE: return "CPU (no GPU)";
        case ESHKOL_GPU_METAL: return "Apple Metal";
        case ESHKOL_GPU_CUDA: return "NVIDIA CUDA";
        case ESHKOL_GPU_VULKAN: return "Vulkan";
    }
    return "Unknown";
}

int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
    return (g_active_backend == backend) ? 1 : 0;
}

int eshkol_gpu_supports_f64(void) {
    // Metal does NOT support f64 (double precision) in compute shaders
    // CUDA supports f64 on all modern GPUs
    switch (g_active_backend) {
        case ESHKOL_GPU_CUDA:
            return 1;  // CUDA supports f64
        case ESHKOL_GPU_METAL:
            return 0;  // Metal does NOT support f64
        default:
            return 0;
    }
}

int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    if (!out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_alloc(size_bytes, mem_type, out_buffer);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_alloc(size_bytes, mem_type, out_buffer);
    }
#endif

    // CPU fallback
    out_buffer->host_ptr = malloc(size_bytes);
    if (!out_buffer->host_ptr) return -1;
    out_buffer->device_ptr = out_buffer->host_ptr;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = ESHKOL_MEM_HOST;
    out_buffer->backend = ESHKOL_GPU_NONE;
    return 0;
}

int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    size_t aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1);
    return eshkol_gpu_alloc(aligned_size, mem_type, out_buffer);
}

void eshkol_gpu_free(EshkolGPUBuffer* buffer) {
    if (!buffer) return;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_METAL) {
        metal_free(buffer);
        memset(buffer, 0, sizeof(EshkolGPUBuffer));
        return;
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        cuda_free(buffer);
        memset(buffer, 0, sizeof(EshkolGPUBuffer));
        return;
    }
#endif

    // CPU fallback
    if (buffer->host_ptr && !(buffer->flags & 1)) {
        free(buffer->host_ptr);
    }
    memset(buffer, 0, sizeof(EshkolGPUBuffer));
}

int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out_buffer) {
    if (!host_ptr || !out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_wrap_host(host_ptr, size_bytes, out_buffer);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_wrap_host(host_ptr, size_bytes, out_buffer);
    }
#endif

    // CPU: just use the pointer directly
    out_buffer->host_ptr = host_ptr;
    out_buffer->device_ptr = host_ptr;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = ESHKOL_MEM_HOST;
    out_buffer->backend = ESHKOL_GPU_NONE;
    out_buffer->flags = 1;  // Wrapped
    return 0;
}

int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (!buffer) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_METAL) {
        return metal_sync(buffer, direction);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        return cuda_sync(buffer, direction);
    }
#endif

    return 0;  // CPU: no sync needed
}

int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction, void* stream) {
    (void)stream;
    return eshkol_gpu_sync(buffer, direction);
}

void eshkol_gpu_wait(EshkolGPUBuffer* buffer) {
    (void)buffer;
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && g_cuda_stream) {
        cudaStreamSynchronize(g_cuda_stream);
    }
#endif
}

int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_matmul_f64(A, B, C, M, K, N);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_matmul_f64(A, B, C, M, K, N);
    }
#endif

    // CPU fallback
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64((const double*)A->host_ptr, (const double*)B->host_ptr,
                      (double*)C->host_ptr, M, K, N);
    return 0;
}

int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_matmul_f32(A, B, C, M, K, N);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_matmul_f32(A, B, C, M, K, N);
    }
#endif

    // CPU fallback: convert and use f64
    // TODO: Add f32 SIMD path
    return -1;
}

void eshkol_gpu_set_threshold(size_t threshold) {
    g_gpu_threshold = threshold;
}

size_t eshkol_gpu_get_threshold(void) {
    return g_gpu_threshold;
}

int eshkol_gpu_should_use(size_t num_elements) {
    return (g_active_backend != ESHKOL_GPU_NONE && num_elements >= g_gpu_threshold) ? 1 : 0;
}

void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    size_t num_elements = M * N;

    // GPU path if available and large enough
    if (eshkol_gpu_should_use(num_elements)) {
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        if (eshkol_gpu_wrap_host((void*)A, M * K * sizeof(double), &buf_a) == 0 &&
            eshkol_gpu_wrap_host((void*)B, K * N * sizeof(double), &buf_b) == 0 &&
            eshkol_gpu_wrap_host((void*)C, M * N * sizeof(double), &buf_c) == 0) {

            if (eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0) {
                eshkol_gpu_free(&buf_a);
                eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
                return;
            }
        }
        // GPU failed, fall through to CPU
    }

    // CPU fallback (BLAS/SIMD/scalar)
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64(A, B, C, M, K, N);
}

} // extern "C"
