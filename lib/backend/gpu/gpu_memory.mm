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
#include <eshkol/logger.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <mutex>

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
static std::atomic<bool> g_gpu_initialized{false};
static std::mutex g_gpu_init_mutex;

// ============================================================================
// Metal Backend State
// ============================================================================

#if ESHKOL_GPU_METAL_AVAILABLE

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_metal_queue = nil;
static id<MTLComputePipelineState> g_matmul_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_elementwise_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_reduce_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_transpose_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_reduce_axis_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_softmax_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_normalize_sf64_pipeline = nil;
static id<MTLLibrary> g_metal_library = nil;
static bool g_metal_unified_memory = false;

// ============================================================================
// Metal Buffer Pool — reuses MTLBuffer objects to reduce allocation overhead
// ============================================================================
// Size-binned pool: rounds requested size up to next power-of-2, reuses buffers
// of that bucket size. Significant win for batched operations (ML training loops).

#include <unordered_map>
#include <vector>

static std::unordered_map<size_t, std::vector<id<MTLBuffer>>> g_buffer_pool;

static size_t pool_bucket(size_t bytes) {
    if (bytes == 0) return 1;
    size_t v = bytes - 1;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return v + 1;
}

static id<MTLBuffer> pool_alloc(size_t bytes) {
    size_t bucket = pool_bucket(bytes);
    auto it = g_buffer_pool.find(bucket);
    if (it != g_buffer_pool.end() && !it->second.empty()) {
        id<MTLBuffer> buf = it->second.back();
        it->second.pop_back();
        return buf;
    }
    return [g_metal_device newBufferWithLength:bucket
                                       options:MTLResourceStorageModeShared];
}

static void pool_release(id<MTLBuffer> buf) {
    if (!buf) return;
    size_t bucket = pool_bucket(buf.length);
    g_buffer_pool[bucket].push_back(buf);
}

static void pool_drain() {
    g_buffer_pool.clear();
}

// ============================================================================
// SoftFloat IEEE 754 f64 Emulation for Metal
// ============================================================================
// Apple Silicon GPUs lack native f64 hardware. We emulate f64 using uint2
// (two 32-bit integers) with full IEEE 754 compliance including:
// - 52-bit mantissa precision (bit-exact with CPU)
// - Proper rounding (round-to-nearest-even)
// - Special value handling (zero, infinity, NaN)
// Based on Berkeley SoftFloat library algorithms.
//
// SINGLE SOURCE OF TRUTH: lib/backend/gpu/metal_softfloat.h
// The shader source below is auto-generated from metal_softfloat.h at build time
// via CMake custom command. Do NOT edit inline — modify metal_softfloat.h instead.
#include "metal_sf64_embedded.inc"


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

        // Compile sf64 (SoftFloat) matmul shader
        // This uses uint2 to emulate f64 with full IEEE 754 compliance
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        // Disable fast-math to ensure exact IEEE 754 rounding behavior
        options.fastMathEnabled = NO;

        g_metal_library = [g_metal_device newLibraryWithSource:g_matmul_sf64_source
                                                       options:options
                                                         error:&error];
        if (!g_metal_library) {
            eshkol_error("Metal: failed to compile sf64 shader: %s",
                    [[error localizedDescription] UTF8String]);
            g_metal_queue = nil;
            g_metal_device = nil;
            return -1;  // Shader compilation failed — GPU unusable
        }

        id<MTLFunction> func = [g_metal_library newFunctionWithName:@"matmul_sf64"];
        if (!func) {
            eshkol_error("Metal: failed to find matmul_sf64 kernel in compiled library");
            g_metal_library = nil;
            g_metal_queue = nil;
            g_metal_device = nil;
            return -1;
        }

        g_matmul_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:func
                                                                               error:&error];
        if (!g_matmul_sf64_pipeline) {
            eshkol_error("Metal: failed to create sf64 compute pipeline: %s",
                    [[error localizedDescription] UTF8String]);
            g_metal_library = nil;
            g_metal_queue = nil;
            g_metal_device = nil;
            return -1;
        }

        // Create pipeline states for elementwise, reduce, transpose kernels
        id<MTLFunction> elem_func = [g_metal_library newFunctionWithName:@"elementwise_sf64"];
        if (elem_func) {
            g_elementwise_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:elem_func error:&error];
        }
        id<MTLFunction> reduce_func = [g_metal_library newFunctionWithName:@"reduce_sf64"];
        if (reduce_func) {
            g_reduce_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:reduce_func error:&error];
        }
        id<MTLFunction> transpose_func = [g_metal_library newFunctionWithName:@"transpose_sf64"];
        if (transpose_func) {
            g_transpose_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:transpose_func error:&error];
        }
        id<MTLFunction> reduce_axis_func = [g_metal_library newFunctionWithName:@"reduce_sf64_axis"];
        if (reduce_axis_func) {
            g_reduce_axis_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:reduce_axis_func error:&error];
        }
        id<MTLFunction> softmax_func = [g_metal_library newFunctionWithName:@"softmax_sf64"];
        if (softmax_func) {
            g_softmax_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:softmax_func error:&error];
        }
        id<MTLFunction> normalize_func = [g_metal_library newFunctionWithName:@"normalize_sf64"];
        if (normalize_func) {
            g_normalize_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:normalize_func error:&error];
        }

        return 0;
    }
}

static void metal_shutdown(void) {
    @autoreleasepool {
        pool_drain();
        g_matmul_sf64_pipeline = nil;
        g_elementwise_sf64_pipeline = nil;
        g_reduce_sf64_pipeline = nil;
        g_reduce_axis_sf64_pipeline = nil;
        g_transpose_sf64_pipeline = nil;
        g_softmax_sf64_pipeline = nil;
        g_normalize_sf64_pipeline = nil;
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

// ============================================================================
// f64 <-> sf64 Conversion (bit-exact reinterpretation)
// ============================================================================
// IEEE 754 f64 bits are split into two uint32: high word (.x) and low word (.y)

static void convert_f64_to_sf64(const double* src, uint32_t* dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint64_t bits;
        memcpy(&bits, &src[i], sizeof(double));
        dst[i * 2] = static_cast<uint32_t>(bits >> 32);      // High word
        dst[i * 2 + 1] = static_cast<uint32_t>(bits);        // Low word
    }
}

static void convert_sf64_to_f64(const uint32_t* src, double* dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint64_t bits = (static_cast<uint64_t>(src[i * 2]) << 32) | src[i * 2 + 1];
        memcpy(&dst[i], &bits, sizeof(double));
    }
}

static int metal_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        if (!g_matmul_sf64_pipeline) return -1;

        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        // Allocate GPU buffers for sf64 format (8 bytes per element, same as f64)
        // Using uint2 representation: each f64 becomes (high32, low32)
        id<MTLBuffer> buf_a = pool_alloc(elementsA * 8);
        id<MTLBuffer> buf_b = pool_alloc(elementsB * 8);
        id<MTLBuffer> buf_c = pool_alloc(elementsC * 8);

        if (!buf_a || !buf_b || !buf_c) {
            if (buf_a) pool_release(buf_a);
            if (buf_b) pool_release(buf_b);
            return -1;
        }

        // Convert f64 inputs to sf64 format (bit reinterpretation)
        convert_f64_to_sf64(static_cast<const double*>(A->host_ptr),
                            static_cast<uint32_t*>([buf_a contents]), elementsA);
        convert_f64_to_sf64(static_cast<const double*>(B->host_ptr),
                            static_cast<uint32_t*>([buf_b contents]), elementsB);

        // Create command buffer
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_matmul_sf64_pipeline];
        [encoder setBuffer:buf_a offset:0 atIndex:0];
        [encoder setBuffer:buf_b offset:0 atIndex:1];
        [encoder setBuffer:buf_c offset:0 atIndex:2];

        // Pass dimensions as bytes
        uint32_t dims[3] = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
        [encoder setBytes:&dims[0] length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&dims[1] length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&dims[2] length:sizeof(uint32_t) atIndex:5];

        // Dispatch with shared-memory tiled kernel:
        // Each threadgroup (8×8 = 64 threads) computes a 32×32 output block.
        // Each thread computes a 4×4 sub-tile from shared memory.
        const uint32_t BLK_SIZE = 32;
        NSUInteger groupsX = (N + BLK_SIZE - 1) / BLK_SIZE;
        NSUInteger groupsY = (M + BLK_SIZE - 1) / BLK_SIZE;
        MTLSize threadgroupCount = MTLSizeMake(groupsX, groupsY, 1);
        MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Convert sf64 result back to f64
        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_c contents]),
                            static_cast<double*>(C->host_ptr), elementsC);

        pool_release(buf_a);
        pool_release(buf_b);
        pool_release(buf_c);
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

static int metal_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                  EshkolGPUBuffer* out, uint64_t n,
                                  int op) {
    @autoreleasepool {
        if (!g_elementwise_sf64_pipeline) return -1;

        id<MTLBuffer> buf_a = pool_alloc(n * 8);
        id<MTLBuffer> buf_b = pool_alloc(n * 8);
        id<MTLBuffer> buf_c = pool_alloc(n * 8);
        if (!buf_a || !buf_b || !buf_c) {
            if (buf_a) pool_release(buf_a);
            if (buf_b) pool_release(buf_b);
            return -1;
        }

        convert_f64_to_sf64(static_cast<const double*>(a->host_ptr),
                            static_cast<uint32_t*>([buf_a contents]), n);
        if (b && b->host_ptr && op <= 3) {
            convert_f64_to_sf64(static_cast<const double*>(b->host_ptr),
                                static_cast<uint32_t*>([buf_b contents]), n);
        }

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:g_elementwise_sf64_pipeline];
        [encoder setBuffer:buf_a offset:0 atIndex:0];
        [encoder setBuffer:buf_b offset:0 atIndex:1];
        [encoder setBuffer:buf_c offset:0 atIndex:2];
        uint32_t n32 = static_cast<uint32_t>(n);
        uint32_t op32 = static_cast<uint32_t>(op);
        [encoder setBytes:&n32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&op32 length:sizeof(uint32_t) atIndex:4];

        NSUInteger groups = (n + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_c contents]),
                            static_cast<double*>(out->host_ptr), n);
        pool_release(buf_a);
        pool_release(buf_b);
        pool_release(buf_c);
        return 0;
    }
}

static int metal_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                             uint64_t n, int op) {
    @autoreleasepool {
        if (!g_reduce_sf64_pipeline) return -1;

        // Two-pass reduction: first pass reduces to partial results per threadgroup
        uint64_t groups = (n + 255) / 256;

        id<MTLBuffer> buf_in = pool_alloc(n * 8);
        id<MTLBuffer> buf_partial = pool_alloc(groups * 8);
        if (!buf_in || !buf_partial) return -1;

        convert_f64_to_sf64(static_cast<const double*>(in->host_ptr),
                            static_cast<uint32_t*>([buf_in contents]), n);

        // Pass 1
        uint32_t n32 = static_cast<uint32_t>(n);
        uint32_t op32 = static_cast<uint32_t>(op);
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:g_reduce_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_partial offset:0 atIndex:1];
        [encoder setBytes:&n32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&op32 length:sizeof(uint32_t) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Pass 2: reduce partial results on CPU (small number of groups)
        double* partials = new double[groups];
        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_partial contents]),
                            partials, groups);

        double result;
        switch (op) {
            case 0: case 4: result = 0.0; break;
            case 1: result = 1.0; break;
            case 2: result = INFINITY; break;
            case 3: result = -INFINITY; break;
            default: result = 0.0; break;
        }
        for (uint64_t i = 0; i < groups; i++) {
            switch (op) {
                case 0: case 4: result += partials[i]; break;
                case 1: result *= partials[i]; break;
                case 2: result = (partials[i] < result) ? partials[i] : result; break;
                case 3: result = (partials[i] > result) ? partials[i] : result; break;
                default: break;
            }
        }
        if (op == 4) result /= (double)n; // Mean = sum / n
        delete[] partials;

        static_cast<double*>(out->host_ptr)[0] = result;
        pool_release(buf_in);
        pool_release(buf_partial);
        return 0;
    }
}

static int metal_reduce_axis_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                  uint64_t rank, const uint64_t* shape,
                                  uint64_t axis, int op) {
    @autoreleasepool {
        if (!g_reduce_axis_sf64_pipeline) return -1;

        // Compute total input elements and output elements
        uint64_t total_in = 1;
        for (uint64_t i = 0; i < rank; i++) total_in *= shape[i];
        uint64_t out_total = total_in / shape[axis];
        if (out_total == 0) return -1;

        // Allocate Metal buffers from pool
        id<MTLBuffer> buf_in = pool_alloc(total_in * 8);
        id<MTLBuffer> buf_out = pool_alloc(out_total * 8);
        // Dims buffer (uint32 for Metal)
        id<MTLBuffer> buf_dims = pool_alloc(rank * sizeof(uint32_t));
        if (!buf_in || !buf_out || !buf_dims) return -1;

        // Convert input f64 → sf64
        convert_f64_to_sf64(static_cast<const double*>(in->host_ptr),
                            static_cast<uint32_t*>([buf_in contents]), total_in);

        // Copy dims as uint32
        uint32_t* dims32 = static_cast<uint32_t*>([buf_dims contents]);
        for (uint64_t i = 0; i < rank; i++) dims32[i] = static_cast<uint32_t>(shape[i]);

        uint32_t rank32 = static_cast<uint32_t>(rank);
        uint32_t axis32 = static_cast<uint32_t>(axis);
        uint32_t op32 = static_cast<uint32_t>(op);
        uint32_t out_size32 = static_cast<uint32_t>(out_total);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:g_reduce_axis_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&rank32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBuffer:buf_dims offset:0 atIndex:3];
        [encoder setBytes:&axis32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&op32 length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&out_size32 length:sizeof(uint32_t) atIndex:6];

        NSUInteger groups = (out_total + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Convert output sf64 → f64
        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_out contents]),
                            static_cast<double*>(out->host_ptr), out_total);
        pool_release(buf_in);
        pool_release(buf_out);
        pool_release(buf_dims);
        return 0;
    }
}

static int metal_transpose_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t rows, uint64_t cols) {
    @autoreleasepool {
        if (!g_transpose_sf64_pipeline) return -1;
        uint64_t n = rows * cols;

        id<MTLBuffer> buf_in = pool_alloc(n * 8);
        id<MTLBuffer> buf_out = pool_alloc(n * 8);
        if (!buf_in || !buf_out) return -1;

        convert_f64_to_sf64(static_cast<const double*>(in->host_ptr),
                            static_cast<uint32_t*>([buf_in contents]), n);

        uint32_t rows32 = static_cast<uint32_t>(rows);
        uint32_t cols32 = static_cast<uint32_t>(cols);
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:g_transpose_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&rows32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&cols32 length:sizeof(uint32_t) atIndex:3];

        NSUInteger groups_x = (cols + 15) / 16;
        NSUInteger groups_y = (rows + 15) / 16;
        [encoder dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_out contents]),
                            static_cast<double*>(out->host_ptr), n);
        pool_release(buf_in);
        pool_release(buf_out);
        return 0;
    }
}

static int metal_softmax_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t num_slices, uint64_t slice_len) {
    @autoreleasepool {
        if (!g_softmax_sf64_pipeline) return -1;
        uint64_t total = num_slices * slice_len;

        id<MTLBuffer> buf_in = pool_alloc(total * 8);
        id<MTLBuffer> buf_out = pool_alloc(total * 8);
        if (!buf_in || !buf_out) return -1;

        convert_f64_to_sf64(static_cast<const double*>(in->host_ptr),
                            static_cast<uint32_t*>([buf_in contents]), total);

        uint32_t slice_len32 = static_cast<uint32_t>(slice_len);
        uint32_t num_slices32 = static_cast<uint32_t>(num_slices);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:g_softmax_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&slice_len32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&num_slices32 length:sizeof(uint32_t) atIndex:3];

        NSUInteger groups = (num_slices + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_out contents]),
                            static_cast<double*>(out->host_ptr), total);
        pool_release(buf_in);
        pool_release(buf_out);
        return 0;
    }
}

static int metal_normalize_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t num_slices, uint64_t slice_len,
                                double gamma, double beta, double epsilon) {
    @autoreleasepool {
        if (!g_normalize_sf64_pipeline) return -1;
        uint64_t total = num_slices * slice_len;

        id<MTLBuffer> buf_in = pool_alloc(total * 8);
        id<MTLBuffer> buf_out = pool_alloc(total * 8);
        if (!buf_in || !buf_out) return -1;

        convert_f64_to_sf64(static_cast<const double*>(in->host_ptr),
                            static_cast<uint32_t*>([buf_in contents]), total);

        uint32_t slice_len32 = static_cast<uint32_t>(slice_len);
        uint32_t num_slices32 = static_cast<uint32_t>(num_slices);

        // Convert gamma, beta, epsilon to sf64 (uint2 = 8 bytes)
        uint32_t gamma_sf[2], beta_sf[2], epsilon_sf[2];
        convert_f64_to_sf64(&gamma, gamma_sf, 1);
        convert_f64_to_sf64(&beta, beta_sf, 1);
        convert_f64_to_sf64(&epsilon, epsilon_sf, 1);

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:g_normalize_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&slice_len32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&num_slices32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:gamma_sf length:8 atIndex:4];
        [encoder setBytes:beta_sf length:8 atIndex:5];
        [encoder setBytes:epsilon_sf length:8 atIndex:6];

        NSUInteger groups = (num_slices + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_out contents]),
                            static_cast<double*>(out->host_ptr), total);
        pool_release(buf_in);
        pool_release(buf_out);
        return 0;
    }
}

#endif // ESHKOL_GPU_METAL_AVAILABLE

// ============================================================================
// CUDA Backend State
// ============================================================================

#if ESHKOL_GPU_CUDA_AVAILABLE

// Forward declarations for CUDA kernel launchers (in gpu_cuda_kernels.cu)
extern "C" int cuda_launch_elementwise_f64(const double* a, const double* b, double* out,
                                            int64_t n, int op, void* stream);
extern "C" int cuda_launch_reduce_f64(const double* in, double* out, int64_t n, int op,
                                       void* stream);
extern "C" int cuda_launch_reduce_axis_f64(const double* in, double* out,
                                            uint64_t rank, const uint64_t* dims,
                                            uint64_t axis, int op, uint64_t out_size,
                                            void* stream);
extern "C" int cuda_launch_transpose_f64(const double* in, double* out,
                                          uint64_t rows, uint64_t cols, void* stream);
extern "C" int cuda_launch_softmax_f64(const double* in, double* out,
                                        uint64_t num_slices, uint64_t slice_len,
                                        void* stream);
extern "C" int cuda_launch_normalize_f64(const double* in, double* out,
                                          uint64_t num_slices, uint64_t slice_len,
                                          double gamma, double beta, double epsilon,
                                          void* stream);

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
    if (g_gpu_initialized.load(std::memory_order_acquire)) {
        return (g_active_backend != ESHKOL_GPU_NONE) ? 1 : 0;
    }

    std::lock_guard<std::mutex> lock(g_gpu_init_mutex);
    // Double-check after acquiring the lock
    if (g_gpu_initialized.load(std::memory_order_relaxed)) {
        return (g_active_backend != ESHKOL_GPU_NONE) ? 1 : 0;
    }

    // Allow override of GPU dispatch threshold via environment variable
    if (const char* env = std::getenv("ESHKOL_GPU_THRESHOLD")) {
        size_t val = static_cast<size_t>(std::atol(env));
        if (val > 0) g_gpu_threshold = val;
    }

    // Try Metal first (macOS)
#if ESHKOL_GPU_METAL_AVAILABLE
    if (metal_init() == 0) {
        g_active_backend = ESHKOL_GPU_METAL;
        g_gpu_initialized.store(true, std::memory_order_release);
        return 1;
    }
#endif

    // Try CUDA
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (cuda_init() == 0) {
        g_active_backend = ESHKOL_GPU_CUDA;
        g_gpu_initialized.store(true, std::memory_order_release);
        return 1;
    }
#endif

    // No GPU available
    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized.store(true, std::memory_order_release);
    return 0;
}

void eshkol_gpu_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_gpu_init_mutex);

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
    g_gpu_initialized.store(false, std::memory_order_release);
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

    // CPU fallback: f32 scalar matmul
    {
        const float* a = (const float*)A->host_ptr;
        const float* b = (const float*)B->host_ptr;
        float* c = (float*)C->host_ptr;
        if (!a || !b || !c) return -1;

        for (uint64_t i = 0; i < M; i++) {
            for (uint64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (uint64_t k = 0; k < K; k++) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
        return 0;
    }
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

int eshkol_gpu_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                EshkolGPUBuffer* out, uint64_t n,
                                EshkolElementwiseOp op) {
    if (!a || !out || n == 0) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_elementwise_f64(a, b, out, n, (int)op);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && a->device_ptr && out->device_ptr) {
        const double* dp_b = (b && b->device_ptr) ? static_cast<const double*>(b->device_ptr) : nullptr;
        int result = cuda_launch_elementwise_f64(
            static_cast<const double*>(a->device_ptr), dp_b,
            static_cast<double*>(out->device_ptr), static_cast<int64_t>(n),
            static_cast<int>(op), static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback: scalar loop
    const double* ap = static_cast<const double*>(a->host_ptr);
    const double* bp = (b && b->host_ptr) ? static_cast<const double*>(b->host_ptr) : nullptr;
    double* cp = static_cast<double*>(out->host_ptr);
    if (!ap || !cp) return -1;
    for (uint64_t i = 0; i < n; i++) {
        switch (op) {
            case ESHKOL_ELEMWISE_ADD: cp[i] = ap[i] + (bp ? bp[i] : 0); break;
            case ESHKOL_ELEMWISE_SUB: cp[i] = ap[i] - (bp ? bp[i] : 0); break;
            case ESHKOL_ELEMWISE_MUL: cp[i] = ap[i] * (bp ? bp[i] : 1); break;
            case ESHKOL_ELEMWISE_DIV: cp[i] = ap[i] / (bp ? bp[i] : 1); break;
            case ESHKOL_ELEMWISE_NEG: cp[i] = -ap[i]; break;
            case ESHKOL_ELEMWISE_ABS: cp[i] = ap[i] < 0 ? -ap[i] : ap[i]; break;
            case ESHKOL_ELEMWISE_EXP: cp[i] = exp(ap[i]); break;
            case ESHKOL_ELEMWISE_LOG: cp[i] = log(ap[i]); break;
            case ESHKOL_ELEMWISE_SIN: cp[i] = sin(ap[i]); break;
            case ESHKOL_ELEMWISE_COS: cp[i] = cos(ap[i]); break;
            case ESHKOL_ELEMWISE_TANH: cp[i] = tanh(ap[i]); break;
            case ESHKOL_ELEMWISE_RELU: cp[i] = ap[i] > 0 ? ap[i] : 0; break;
            case ESHKOL_ELEMWISE_SIGMOID: cp[i] = 1.0 / (1.0 + exp(-ap[i])); break;
            case ESHKOL_ELEMWISE_SQRT: cp[i] = sqrt(ap[i]); break;
            case ESHKOL_ELEMWISE_RECIPROCAL: cp[i] = 1.0 / ap[i]; break;
        }
    }
    return 0;
}

int eshkol_gpu_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                           uint64_t n, EshkolReduceOp op) {
    if (!in || !out || n == 0) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_reduce_f64(in, out, n, (int)op);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_reduce_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            static_cast<int64_t>(n), static_cast<int>(op),
            static_cast<void*>(g_cuda_stream));
        if (result == 0) {
            if (op == ESHKOL_REDUCE_MEAN) {
                cudaStreamSynchronize(g_cuda_stream);
                double sum_val;
                cudaMemcpy(&sum_val, out->device_ptr, sizeof(double), cudaMemcpyDeviceToHost);
                sum_val /= (double)n;
                cudaMemcpy(out->device_ptr, &sum_val, sizeof(double), cudaMemcpyHostToDevice);
            }
            return 0;
        }
    }
#endif

    // CPU fallback
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    double result;
    switch (op) {
        case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: result = 0.0; break;
        case ESHKOL_REDUCE_PROD: result = 1.0; break;
        case ESHKOL_REDUCE_MIN: result = INFINITY; break;
        case ESHKOL_REDUCE_MAX: result = -INFINITY; break;
    }
    for (uint64_t i = 0; i < n; i++) {
        switch (op) {
            case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: result += inp[i]; break;
            case ESHKOL_REDUCE_PROD: result *= inp[i]; break;
            case ESHKOL_REDUCE_MIN: result = (inp[i] < result) ? inp[i] : result; break;
            case ESHKOL_REDUCE_MAX: result = (inp[i] > result) ? inp[i] : result; break;
        }
    }
    if (op == ESHKOL_REDUCE_MEAN) result /= (double)n;
    outp[0] = result;
    return 0;
}

int eshkol_gpu_reduce_axis_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t rank, const uint64_t* shape,
                                uint64_t axis, EshkolReduceOp op) {
    if (!in || !out || !shape || rank == 0 || axis >= rank) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_reduce_axis_f64(in, out, rank, shape, axis, (int)op);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        uint64_t total_in = 1;
        for (uint64_t i = 0; i < rank; i++) total_in *= shape[i];
        uint64_t out_total = total_in / shape[axis];
        int result = cuda_launch_reduce_axis_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            rank, shape, axis, static_cast<int>(op), out_total,
            static_cast<void*>(g_cuda_stream));
        if (result == 0) {
            if (op == ESHKOL_REDUCE_MEAN) {
                // MEAN needs post-divide
                double* outp = static_cast<double*>(out->host_ptr);
                cudaStreamSynchronize(g_cuda_stream);
                for (uint64_t i = 0; i < out_total; i++) outp[i] /= (double)shape[axis];
            }
            return 0;
        }
    }
#endif

    // CPU fallback: N-D axis reduction with stride computation
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;

    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (uint64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];
    uint64_t outer_stride = axis_len * inner_stride;

    uint64_t total_in = 1;
    for (uint64_t i = 0; i < rank; i++) total_in *= shape[i];
    uint64_t out_total = total_in / axis_len;

    for (uint64_t out_idx = 0; out_idx < out_total; out_idx++) {
        uint64_t outer = out_idx / inner_stride;
        uint64_t inner = out_idx % inner_stride;

        double acc;
        switch (op) {
            case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: acc = 0.0; break;
            case ESHKOL_REDUCE_PROD: acc = 1.0; break;
            case ESHKOL_REDUCE_MIN: acc = INFINITY; break;
            case ESHKOL_REDUCE_MAX: acc = -INFINITY; break;
        }
        for (uint64_t k = 0; k < axis_len; k++) {
            uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
            double val = inp[src_idx];
            switch (op) {
                case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: acc += val; break;
                case ESHKOL_REDUCE_PROD: acc *= val; break;
                case ESHKOL_REDUCE_MIN: acc = (val < acc) ? val : acc; break;
                case ESHKOL_REDUCE_MAX: acc = (val > acc) ? val : acc; break;
            }
        }
        if (op == ESHKOL_REDUCE_MEAN) acc /= (double)axis_len;
        outp[out_idx] = acc;
    }
    return 0;
}

int eshkol_gpu_transpose_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t rows, uint64_t cols) {
    if (!in || !out || rows == 0 || cols == 0) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_transpose_f64(in, out, rows, cols);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_transpose_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            rows, cols, static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    for (uint64_t i = 0; i < rows; i++) {
        for (uint64_t j = 0; j < cols; j++) {
            outp[j * rows + i] = inp[i * cols + j];
        }
    }
    return 0;
}

int eshkol_gpu_softmax_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                            uint64_t num_slices, uint64_t slice_len) {
    if (!in || !out || num_slices == 0 || slice_len == 0) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_softmax_f64(in, out, num_slices, slice_len);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_softmax_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            num_slices, slice_len, static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback: numerically stable softmax
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    for (uint64_t s = 0; s < num_slices; s++) {
        uint64_t base = s * slice_len;
        double max_val = inp[base];
        for (uint64_t k = 1; k < slice_len; k++)
            if (inp[base + k] > max_val) max_val = inp[base + k];
        double sum_exp = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) {
            outp[base + k] = std::exp(inp[base + k] - max_val);
            sum_exp += outp[base + k];
        }
        if (sum_exp == 0.0) sum_exp = 1.0;
        for (uint64_t k = 0; k < slice_len; k++)
            outp[base + k] /= sum_exp;
    }
    return 0;
}

int eshkol_gpu_normalize_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t num_slices, uint64_t slice_len,
                              double gamma, double beta, double epsilon) {
    if (!in || !out || num_slices == 0 || slice_len == 0) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_normalize_f64(in, out, num_slices, slice_len,
                                   gamma, beta, epsilon);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_normalize_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            num_slices, slice_len, gamma, beta, epsilon,
            static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback: layer normalization
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    for (uint64_t s = 0; s < num_slices; s++) {
        uint64_t base = s * slice_len;
        double sum = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) sum += inp[base + k];
        double mean = sum / static_cast<double>(slice_len);
        double var_sum = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) {
            double diff = inp[base + k] - mean;
            var_sum += diff * diff;
        }
        double inv_std = 1.0 / std::sqrt(var_sum / static_cast<double>(slice_len) + epsilon);
        for (uint64_t k = 0; k < slice_len; k++)
            outp[base + k] = gamma * (inp[base + k] - mean) * inv_std + beta;
    }
    return 0;
}

} // extern "C"
