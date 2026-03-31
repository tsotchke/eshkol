/*
 * XLA Runtime Implementation for Eshkol
 *
 * Provides runtime support for XLA-compiled tensor operations.
 * Currently delegates to BLAS/SIMD while XLA JIT compilation is developed.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_runtime.h"
#include "eshkol/backend/xla/xla_codegen.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <future>
#include <unordered_map>
#include <chrono>
#include <mutex>

// Use the existing BLAS matmul for now
extern "C" {
    void eshkol_matmul_f64(const double* A, const double* B, double* C,
                           uint64_t M, uint64_t K, uint64_t N);
}

// GPU acceleration
#include "eshkol/backend/gpu/gpu_memory.h"

// Lazy GPU initialization for XLA runtime functions
// Uses std::call_once for thread-safe one-time initialization
static std::once_flag g_xla_gpu_init_flag;

static void ensure_gpu_initialized() {
    std::call_once(g_xla_gpu_init_flag, []() {
        eshkol_gpu_init();
    });
}

// Arena allocation functions
extern "C" void* arena_allocate_aligned(void* arena, size_t size, size_t alignment);

// Forward-declare the canonical tensor type and allocator from arena_memory.h
// struct eshkol_tensor { uint64_t* dimensions, uint64_t num_dimensions,
//                        int64_t* elements, uint64_t total_elements }
// arena_allocate_tensor_full allocates tensor struct WITH object header
// (HEAP_SUBTYPE_TENSOR at ptr-8) + dimensions array + elements array
typedef struct eshkol_tensor {
    uint64_t* dimensions;     // idx 0: dimension sizes array
    uint64_t  num_dimensions; // idx 1: rank
    int64_t*  elements;       // idx 2: doubles as int64 bit patterns
    uint64_t  total_elements; // idx 3: product of all dimensions
} eshkol_tensor_t;

extern "C" eshkol_tensor_t* arena_allocate_tensor_full(
    void* arena, uint64_t num_dims, uint64_t total_elements);

// XLA matmul runtime function - called by generated LLVM IR
// Performs matrix multiplication using the existing BLAS/SIMD backend
// Parameters:
//   arena - arena allocator for result tensor
//   a_data, b_data - input tensor data pointers
//   a_shape, b_shape - shape arrays (dimensions)
//   a_rank, b_rank - number of dimensions
// Returns: pointer to result tensor struct
extern "C" void* eshkol_xla_matmul(
    void* arena,
    const double* a_data,
    const double* b_data,
    const int64_t* a_shape,
    const int64_t* b_shape,
    int64_t a_rank,
    int64_t b_rank) {

    // Currently only support 2D matmul
    if (a_rank != 2 || b_rank != 2) {
        return nullptr;
    }

    // Extract dimensions: A is MxK, B is KxN, result is MxN
    uint64_t M = static_cast<uint64_t>(a_shape[0]);
    uint64_t K = static_cast<uint64_t>(a_shape[1]);
    uint64_t K2 = static_cast<uint64_t>(b_shape[0]);
    uint64_t N = static_cast<uint64_t>(b_shape[1]);

    // Verify inner dimensions match
    if (K != K2) {
        return nullptr;
    }

    // Allocate result tensor with object header (HEAP_SUBTYPE_TENSOR)
    // arena_allocate_tensor_full gives us: header + tensor struct + dims + elements
    eshkol_tensor_t* result = arena_allocate_tensor_full(arena, 2, M * N);
    if (!result) return nullptr;

    // Set dimension sizes
    result->dimensions[0] = M;
    result->dimensions[1] = N;

    // Perform matrix multiplication using GPU/BLAS/SIMD cascade
    // elements is int64_t* but stores doubles as bit patterns — safe to cast
    // since sizeof(double) == sizeof(int64_t) == 8
    double* out = reinterpret_cast<double*>(result->elements);

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    // Try GPU dispatch for large matrices
    int gpu_should = eshkol_gpu_should_use(M * N);
    if (gpu_should) {
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        if (eshkol_gpu_wrap_host((void*)a_data, M * K * sizeof(double), &buf_a) == 0 &&
            eshkol_gpu_wrap_host((void*)b_data, K * N * sizeof(double), &buf_b) == 0 &&
            eshkol_gpu_wrap_host((void*)out, M * N * sizeof(double), &buf_c) == 0) {
            if (eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0) {
                eshkol_gpu_free(&buf_a);
                eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
                return result;
            }
        }
    }
    // CPU fallback (BLAS/SIMD)
    eshkol_matmul_f64(a_data, b_data, out, M, K, N);

    return result;
}

// ===== XLA Elementwise Runtime =====
// Applies binary or unary operations element-wise across tensors.
// Op codes match ElementwiseOp enum: ADD=0,SUB=1,MUL=2,DIV=3,
//   EXP=4,LOG=5,SIN=6,COS=7,TANH=8,RELU=9,SIGMOID=10
extern "C" void* eshkol_xla_elementwise(
    void* arena,
    const double* a_data,
    const double* b_data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t op_code) {

    if (total_elements <= 0 || !a_data) return nullptr;

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        arena, static_cast<uint64_t>(rank), static_cast<uint64_t>(total_elements));
    if (!result) return nullptr;

    // Copy shape
    for (int64_t i = 0; i < rank; i++) {
        result->dimensions[i] = shape[i];
    }

    double* out = reinterpret_cast<double*>(result->elements);
    uint64_t n = static_cast<uint64_t>(total_elements);

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    // Try GPU dispatch for large tensors
    // XLA ElementwiseOp: ADD=0,SUB=1,MUL=2,DIV=3,EXP=4,LOG=5,SIN=6,COS=7,TANH=8,RELU=9,SIGMOID=10
    // GPU EshkolElementwiseOp: ADD=0,SUB=1,MUL=2,DIV=3,NEG=4,ABS=5,EXP=6,LOG=7,SIN=8,COS=9,TANH=10,RELU=11,SIGMOID=12
    // GPU enum has NEG(4) and ABS(5) inserted, so XLA unary ops 4-10 map to GPU 6-12
    static const int xla_to_gpu_elemwise[] = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12};
    static const int xla_elemwise_count = sizeof(xla_to_gpu_elemwise) / sizeof(xla_to_gpu_elemwise[0]);

    if (eshkol_gpu_should_use(n) && op_code >= 0 && op_code < xla_elemwise_count) {
        int gpu_op = xla_to_gpu_elemwise[op_code];
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        bool gpu_ok = eshkol_gpu_wrap_host((void*)a_data, n * sizeof(double), &buf_a) == 0;
        if (gpu_ok && b_data && op_code <= 3) {
            gpu_ok = eshkol_gpu_wrap_host((void*)b_data, n * sizeof(double), &buf_b) == 0;
        }
        if (gpu_ok) {
            gpu_ok = eshkol_gpu_wrap_host((void*)out, n * sizeof(double), &buf_c) == 0;
        }
        if (gpu_ok) {
            EshkolGPUBuffer* bp = (b_data && op_code <= 3) ? &buf_b : nullptr;
            if (eshkol_gpu_elementwise_f64(&buf_a, bp, &buf_c, n,
                    static_cast<EshkolElementwiseOp>(gpu_op)) == 0) {
                eshkol_gpu_free(&buf_a);
                if (b_data && op_code <= 3) eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
                return result;
            }
        }
    }

    // CPU fallback
    switch (op_code) {
        case 0: // ADD
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] + b_data[i];
            break;
        case 1: // SUB
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] - b_data[i];
            break;
        case 2: // MUL
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] * b_data[i];
            break;
        case 3: // DIV
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] / b_data[i];
            break;
        case 4: // EXP
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::exp(a_data[i]);
            break;
        case 5: // LOG
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::log(a_data[i]);
            break;
        case 6: // SIN
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::sin(a_data[i]);
            break;
        case 7: // COS
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::cos(a_data[i]);
            break;
        case 8: // TANH
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::tanh(a_data[i]);
            break;
        case 9: // RELU
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] > 0.0 ? a_data[i] : 0.0;
            break;
        case 10: // SIGMOID
            for (int64_t i = 0; i < total_elements; i++) out[i] = 1.0 / (1.0 + std::exp(-a_data[i]));
            break;
        default:
            return nullptr;
    }

    return result;
}

// ===== XLA Reduce Runtime =====
// Reduces a tensor along an axis (or all axes if axis == -1).
// Op codes match ReduceOp enum: SUM=0,MEAN=1,MAX=2,MIN=3,PROD=4
extern "C" void* eshkol_xla_reduce(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    int64_t op_code) {

    if (total_elements <= 0 || !data) return nullptr;

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    if (axis == -1) {
        // Reduce all — result is a scalar tensor (rank 1, 1 element)
        eshkol_tensor_t* result = arena_allocate_tensor_full(arena, 1, 1);
        if (!result) return nullptr;
        result->dimensions[0] = 1;

        double* out = reinterpret_cast<double*>(result->elements);
        uint64_t n = static_cast<uint64_t>(total_elements);

        // Try GPU dispatch for large reductions
        // XLA op codes: SUM=0,MEAN=1,MAX=2,MIN=3,PROD=4
        // GPU op codes: SUM=0,PROD=1,MIN=2,MAX=3,MEAN=4
        if (eshkol_gpu_should_use(n) && op_code >= 0 && op_code <= 4) {
            static const int xla_to_gpu_reduce[] = {0, 4, 3, 2, 1};
            EshkolGPUBuffer buf_in, buf_out;
            if (eshkol_gpu_wrap_host((void*)data, n * sizeof(double), &buf_in) == 0 &&
                eshkol_gpu_wrap_host((void*)out, sizeof(double), &buf_out) == 0) {
                if (eshkol_gpu_reduce_f64(&buf_in, &buf_out, n,
                        static_cast<EshkolReduceOp>(xla_to_gpu_reduce[op_code])) == 0) {
                    eshkol_gpu_free(&buf_in);
                    eshkol_gpu_free(&buf_out);
                    return result;
                }
            }
        }

        // CPU fallback
        double acc;
        switch (op_code) {
            case 0: // SUM
                acc = 0.0;
                for (int64_t i = 0; i < total_elements; i++) acc += data[i];
                break;
            case 1: // MEAN
                acc = 0.0;
                for (int64_t i = 0; i < total_elements; i++) acc += data[i];
                acc /= static_cast<double>(total_elements);
                break;
            case 2: // MAX
                acc = data[0];
                for (int64_t i = 1; i < total_elements; i++) acc = std::fmax(acc, data[i]);
                break;
            case 3: // MIN
                acc = data[0];
                for (int64_t i = 1; i < total_elements; i++) acc = std::fmin(acc, data[i]);
                break;
            case 4: // PROD
                acc = 1.0;
                for (int64_t i = 0; i < total_elements; i++) acc *= data[i];
                break;
            default:
                return nullptr;
        }

        out[0] = acc;
        return result;
    }

    // Reduce along specific axis
    if (axis < 0 || axis >= rank) return nullptr;

    // Compute output shape (remove the reduced axis)
    if (rank > 16) return nullptr; // max 16D tensors supported
    uint64_t out_rank = static_cast<uint64_t>(rank - 1);
    if (out_rank == 0) out_rank = 1; // scalar result

    // Compute strides and output dimensions
    uint64_t out_total = 1;
    uint64_t out_dims[16];
    uint64_t j = 0;
    for (int64_t i = 0; i < rank; i++) {
        if (i != axis) {
            out_dims[j++] = shape[i];
            out_total *= shape[i];
        }
    }
    if (j == 0) { out_dims[0] = 1; out_total = 1; }

    eshkol_tensor_t* result = arena_allocate_tensor_full(arena, out_rank, out_total);
    if (!result) return nullptr;
    for (uint64_t i = 0; i < out_rank; i++) result->dimensions[i] = out_dims[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Try GPU dispatch for axis-reduce (large tensors)
    if (eshkol_gpu_should_use(static_cast<size_t>(total_elements)) && op_code >= 0 && op_code <= 4) {
        // XLA op codes: SUM=0, MEAN=1, MAX=2, MIN=3, PROD=4
        // GPU op codes: SUM=0, PROD=1, MIN=2, MAX=3, MEAN=4
        static const int xla_to_gpu_reduce[] = {0, 4, 3, 2, 1};
        EshkolGPUBuffer buf_in, buf_out;
        if (eshkol_gpu_wrap_host((void*)data, static_cast<size_t>(total_elements) * sizeof(double), &buf_in) == 0 &&
            eshkol_gpu_wrap_host((void*)out, out_total * sizeof(double), &buf_out) == 0) {
            if (eshkol_gpu_reduce_axis_f64(&buf_in, &buf_out, static_cast<uint64_t>(rank),
                    shape, static_cast<uint64_t>(axis),
                    static_cast<EshkolReduceOp>(xla_to_gpu_reduce[op_code])) == 0) {
                eshkol_gpu_free(&buf_in);
                eshkol_gpu_free(&buf_out);
                return result;
            }
            eshkol_gpu_free(&buf_in);
            eshkol_gpu_free(&buf_out);
        }
    }

    // CPU fallback: compute stride for the reduced axis
    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];
    uint64_t outer_stride = axis_len * inner_stride;

    // Reduce
    for (uint64_t outer = 0; outer < out_total / (inner_stride > 0 ? inner_stride : 1); outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            uint64_t out_idx = outer * inner_stride + inner;
            double acc;
            switch (op_code) {
                case 0: case 1: acc = 0.0; break;
                case 2: acc = -INFINITY; break;
                case 3: acc = INFINITY; break;
                case 4: acc = 1.0; break;
                default: acc = 0.0; break;
            }
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
                double val = data[src_idx];
                switch (op_code) {
                    case 0: case 1: acc += val; break;
                    case 2: acc = std::fmax(acc, val); break;
                    case 3: acc = std::fmin(acc, val); break;
                    case 4: acc *= val; break;
                }
            }
            if (op_code == 1) acc /= static_cast<double>(axis_len); // MEAN
            out[out_idx] = acc;
        }
    }

    return result;
}

// ===== XLA Scale In-Place Runtime =====
// Multiplies every element of a tensor data buffer by a scalar.
// Used by MEAN gradient to divide broadcasted gradient by n.
extern "C" void* eshkol_xla_scale_inplace(
    double* data,
    int64_t total_elements,
    double scale) {
    for (int64_t i = 0; i < total_elements; i++) {
        data[i] *= scale;
    }
    return data;
}

// ===== XLA Softmax Runtime =====
// Numerically stable softmax along a specified axis.
// axis == -1 means softmax over all elements (global softmax).
extern "C" void* eshkol_xla_softmax(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis) {

    if (total_elements <= 0 || !data) return nullptr;

    // Handle negative axis
    if (axis < -1) axis = axis + rank;

    // Allocate result tensor with same shape
    uint64_t out_rank = static_cast<uint64_t>(rank);
    eshkol_tensor_t* result = arena_allocate_tensor_full(arena, out_rank, static_cast<uint64_t>(total_elements));
    if (!result) return nullptr;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // GPU dispatch for contiguous softmax (global or last-axis)
    uint64_t num_slices = 0, slice_len = 0;
    bool gpu_eligible = false;
    if (axis == -1) {
        num_slices = 1;
        slice_len = static_cast<uint64_t>(total_elements);
        gpu_eligible = true;
    } else if (axis >= 0 && axis < rank) {
        // Check if axis is contiguous (inner_stride == 1, i.e. last axis)
        uint64_t inner_stride = 1;
        for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];
        if (inner_stride == 1) {
            slice_len = shape[axis];
            num_slices = static_cast<uint64_t>(total_elements) / slice_len;
            gpu_eligible = true;
        }
    }

    if (gpu_eligible && eshkol_gpu_should_use(static_cast<size_t>(total_elements))) {
        EshkolGPUBuffer in_buf = {const_cast<double*>(data), nullptr,
                                   static_cast<size_t>(total_elements) * sizeof(double),
                                   ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        EshkolGPUBuffer out_buf = {out, nullptr,
                                    static_cast<size_t>(total_elements) * sizeof(double),
                                    ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        if (eshkol_gpu_softmax_f64(&in_buf, &out_buf, num_slices, slice_len) == 0)
            return result;
    }

    if (axis == -1) {
        // Global softmax: max, exp, sum, normalize over all elements
        double max_val = data[0];
        for (int64_t i = 1; i < total_elements; i++)
            if (data[i] > max_val) max_val = data[i];

        double sum_exp = 0.0;
        for (int64_t i = 0; i < total_elements; i++) {
            out[i] = std::exp(data[i] - max_val);
            sum_exp += out[i];
        }
        if (sum_exp == 0.0) sum_exp = 1e-10;
        for (int64_t i = 0; i < total_elements; i++)
            out[i] /= sum_exp;
        return result;
    }

    // Axis-specific softmax
    if (axis < 0 || axis >= rank) return nullptr;

    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];

    uint64_t outer_count = static_cast<uint64_t>(total_elements) / (axis_len * inner_stride);

    // For each "slice" perpendicular to the axis:
    // 1) find max, 2) compute exp(x - max), 3) sum, 4) normalize
    for (uint64_t outer = 0; outer < outer_count; outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            // Step 1: max
            double max_val = -INFINITY;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                if (data[idx] > max_val) max_val = data[idx];
            }
            // Step 2-3: exp and sum
            double sum_exp = 0.0;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                double e = std::exp(data[idx] - max_val);
                out[idx] = e;
                sum_exp += e;
            }
            // Step 4: normalize
            if (sum_exp == 0.0) sum_exp = 1e-10;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                out[idx] /= sum_exp;
            }
        }
    }
    return result;
}

// ===== XLA Normalize Runtime =====
// Axis-aware normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Computes mean and variance along the specified axis.
// gamma and beta are scalar (applied uniformly).
extern "C" void* eshkol_xla_normalize(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    double gamma,
    double beta,
    double epsilon) {

    if (total_elements <= 0 || !data) return nullptr;

    // Handle negative axis
    if (axis < 0) axis = axis + rank;
    if (axis < 0 || axis >= rank) return nullptr;

    // Allocate result tensor with same shape
    eshkol_tensor_t* result = arena_allocate_tensor_full(arena,
        static_cast<uint64_t>(rank), static_cast<uint64_t>(total_elements));
    if (!result) return nullptr;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];

    // GPU dispatch for contiguous normalize (last-axis, inner_stride == 1)
    if (inner_stride == 1 && eshkol_gpu_should_use(static_cast<size_t>(total_elements))) {
        uint64_t num_slices = static_cast<uint64_t>(total_elements) / axis_len;
        EshkolGPUBuffer in_buf = {const_cast<double*>(data), nullptr,
                                   static_cast<size_t>(total_elements) * sizeof(double),
                                   ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        EshkolGPUBuffer out_buf = {out, nullptr,
                                    static_cast<size_t>(total_elements) * sizeof(double),
                                    ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        if (eshkol_gpu_normalize_f64(&in_buf, &out_buf, num_slices, axis_len,
                                      gamma, beta, epsilon) == 0)
            return result;
    }

    uint64_t outer_count = static_cast<uint64_t>(total_elements) / (axis_len * inner_stride);

    // For each slice perpendicular to the axis:
    // 1) compute mean, 2) compute variance, 3) normalize
    for (uint64_t outer = 0; outer < outer_count; outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            // Mean
            double sum = 0.0;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                sum += data[idx];
            }
            double mean = sum / static_cast<double>(axis_len);

            // Variance
            double var_sum = 0.0;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                double diff = data[idx] - mean;
                var_sum += diff * diff;
            }
            double var = var_sum / static_cast<double>(axis_len);

            // Normalize: y = gamma * (x - mean) / sqrt(var + eps) + beta
            double inv_std = 1.0 / std::sqrt(var + epsilon);
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                out[idx] = gamma * (data[idx] - mean) * inv_std + beta;
            }
        }
    }
    return result;
}

// ===== XLA Argreduce Runtime =====
// Returns tensor of indices (as doubles) for argmax/argmin along an axis.
// axis == -1 means argreduce over all elements (returns scalar index as double).
extern "C" void* eshkol_xla_argreduce(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    int64_t is_max) {  // 1=argmax, 0=argmin

    if (total_elements <= 0 || !data) return nullptr;

    if (axis == -1) {
        // Argreduce all — return scalar tensor with flattened index as double
        eshkol_tensor_t* result = arena_allocate_tensor_full(arena, 1, 1);
        if (!result) return nullptr;
        result->dimensions[0] = 1;

        int64_t best_idx = 0;
        double best_val = data[0];
        for (int64_t i = 1; i < total_elements; i++) {
            bool better = is_max ? (data[i] > best_val) : (data[i] < best_val);
            if (better) { best_val = data[i]; best_idx = i; }
        }
        double idx_as_double = static_cast<double>(best_idx);
        reinterpret_cast<double*>(result->elements)[0] = idx_as_double;
        return result;
    }

    // Argreduce along specific axis
    if (axis < 0 || axis >= rank) return nullptr;
    if (rank > 16) return nullptr; // max 16D tensors supported

    uint64_t out_rank = static_cast<uint64_t>(rank - 1);
    if (out_rank == 0) out_rank = 1;

    uint64_t out_total = 1;
    uint64_t out_dims[16];
    uint64_t j = 0;
    for (int64_t i = 0; i < rank; i++) {
        if (i != axis) {
            out_dims[j++] = shape[i];
            out_total *= shape[i];
        }
    }
    if (j == 0) { out_dims[0] = 1; out_total = 1; }

    eshkol_tensor_t* result = arena_allocate_tensor_full(arena, out_rank, out_total);
    if (!result) return nullptr;
    for (uint64_t i = 0; i < out_rank; i++) result->dimensions[i] = out_dims[i];

    double* out = reinterpret_cast<double*>(result->elements);
    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];

    for (uint64_t outer = 0; outer < out_total / (inner_stride > 0 ? inner_stride : 1); outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            uint64_t out_idx = outer * inner_stride + inner;
            double best_val = data[outer * axis_len * inner_stride + inner];
            int64_t best_k = 0;
            for (uint64_t k = 1; k < axis_len; k++) {
                uint64_t src_idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                bool better = is_max ? (data[src_idx] > best_val) : (data[src_idx] < best_val);
                if (better) { best_val = data[src_idx]; best_k = static_cast<int64_t>(k); }
            }
            out[out_idx] = static_cast<double>(best_k);
        }
    }
    return result;
}

// ===== XLA Reduce Gradient Runtime =====
// Computes the gradient of a reduce operation w.r.t. the input.
// Supports MAX, MIN, PROD gradients (SUM/MEAN handled in codegen).
// Parameters:
//   arena       - arena allocator
//   grad_data   - upstream gradient elements (reduced shape)
//   input_data  - original input elements (full shape)
//   input_shape - shape of the original input
//   input_rank  - rank of the original input
//   total_input - total elements in input
//   axis        - axis that was reduced (-1 for reduce-all)
//   op_code     - 2=MAX, 3=MIN, 4=PROD
// Returns: tensor* with same shape as input, containing gradient
extern "C" void* eshkol_xla_reduce_gradient(
    void* arena,
    const double* grad_data,
    const double* input_data,
    const uint64_t* input_shape,
    int64_t input_rank,
    int64_t total_input,
    int64_t axis,
    int64_t op_code) {

    if (!grad_data || !input_data || total_input <= 0) return nullptr;

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        arena, static_cast<uint64_t>(input_rank), static_cast<uint64_t>(total_input));
    if (!result) return nullptr;
    for (int64_t i = 0; i < input_rank; i++) {
        result->dimensions[i] = input_shape[i];
    }
    double* out = reinterpret_cast<double*>(result->elements);

    if (axis == -1) {
        // Reduce-all gradient
        if (op_code == 2 || op_code == 3) {
            // MAX/MIN: gradient is upstream_grad / count where input == extremum
            double extremum = input_data[0];
            for (int64_t i = 1; i < total_input; i++) {
                if (op_code == 2) extremum = std::fmax(extremum, input_data[i]);
                else extremum = std::fmin(extremum, input_data[i]);
            }
            int64_t count = 0;
            for (int64_t i = 0; i < total_input; i++) {
                if (input_data[i] == extremum) count++;
            }
            double grad_val = (count > 0) ? grad_data[0] / static_cast<double>(count) : 0.0;
            for (int64_t i = 0; i < total_input; i++) {
                out[i] = (input_data[i] == extremum) ? grad_val : 0.0;
            }
        } else if (op_code == 4) {
            // PROD: gradient[i] = prod(x_j for j!=i) * upstream_grad
            // = total_product / x[i] * upstream_grad
            // Handle zeros: count zeros, if >1 all grads are 0
            int64_t zero_count = 0;
            int64_t zero_idx = -1;
            double total_product = 1.0;
            for (int64_t i = 0; i < total_input; i++) {
                if (input_data[i] == 0.0) {
                    zero_count++;
                    zero_idx = i;
                } else {
                    total_product *= input_data[i];
                }
            }
            if (zero_count > 1) {
                // More than one zero: all gradients are 0
                for (int64_t i = 0; i < total_input; i++) out[i] = 0.0;
            } else if (zero_count == 1) {
                // Exactly one zero: only the zero element gets a gradient
                for (int64_t i = 0; i < total_input; i++) out[i] = 0.0;
                out[zero_idx] = total_product * grad_data[0];
            } else {
                // No zeros: grad[i] = total_product / x[i] * upstream
                double upstream = grad_data[0];
                for (int64_t i = 0; i < total_input; i++) {
                    out[i] = (total_product / input_data[i]) * upstream;
                }
            }
        }
    } else {
        // Axis-specific gradient (same logic but per-slice along axis)
        if (axis < 0 || axis >= input_rank) return nullptr;

        uint64_t axis_len = input_shape[axis];
        uint64_t inner_stride = 1;
        for (int64_t i = axis + 1; i < input_rank; i++) inner_stride *= input_shape[i];
        uint64_t outer_stride = axis_len * inner_stride;
        uint64_t out_total = static_cast<uint64_t>(total_input) / axis_len;

        for (uint64_t outer = 0; outer < out_total / (inner_stride > 0 ? inner_stride : 1); outer++) {
            for (uint64_t inner = 0; inner < inner_stride; inner++) {
                uint64_t grad_idx = outer * inner_stride + inner;

                if (op_code == 2 || op_code == 3) {
                    // Find extremum along this slice
                    double extremum = input_data[outer * outer_stride + inner];
                    for (uint64_t k = 1; k < axis_len; k++) {
                        double val = input_data[outer * outer_stride + k * inner_stride + inner];
                        if (op_code == 2) extremum = std::fmax(extremum, val);
                        else extremum = std::fmin(extremum, val);
                    }
                    int64_t count = 0;
                    for (uint64_t k = 0; k < axis_len; k++) {
                        if (input_data[outer * outer_stride + k * inner_stride + inner] == extremum) count++;
                    }
                    double gv = (count > 0) ? grad_data[grad_idx] / static_cast<double>(count) : 0.0;
                    for (uint64_t k = 0; k < axis_len; k++) {
                        uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
                        out[src_idx] = (input_data[src_idx] == extremum) ? gv : 0.0;
                    }
                } else if (op_code == 4) {
                    // PROD gradient along axis
                    int64_t zc = 0;
                    int64_t zi = -1;
                    double tp = 1.0;
                    for (uint64_t k = 0; k < axis_len; k++) {
                        double val = input_data[outer * outer_stride + k * inner_stride + inner];
                        if (val == 0.0) { zc++; zi = static_cast<int64_t>(k); }
                        else tp *= val;
                    }
                    double upstream = grad_data[grad_idx];
                    for (uint64_t k = 0; k < axis_len; k++) {
                        uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
                        if (zc > 1) {
                            out[src_idx] = 0.0;
                        } else if (zc == 1) {
                            out[src_idx] = (static_cast<int64_t>(k) == zi) ? tp * upstream : 0.0;
                        } else {
                            out[src_idx] = (tp / input_data[src_idx]) * upstream;
                        }
                    }
                }
            }
        }
    }

    return result;
}

// ===== XLA Transpose Runtime =====
// Transposes a 2D tensor (matrix transpose).
// For higher-rank tensors, perm specifies the permutation of axes.
extern "C" void* eshkol_xla_transpose(
    void* arena,
    const double* data,
    const uint64_t* shape,
    int64_t rank,
    const int64_t* perm) {

    if (!data || rank <= 0) return nullptr;

    // Compute transposed shape and total elements
    uint64_t total = 1;
    uint64_t out_shape[16];
    for (int64_t i = 0; i < rank; i++) {
        out_shape[i] = shape[perm[i]];
        total *= out_shape[i];
    }

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        arena, static_cast<uint64_t>(rank), total);
    if (!result) return nullptr;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = out_shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    // Try GPU dispatch for 2D transpose of large matrices
    if (rank == 2 && eshkol_gpu_should_use(total)) {
        uint64_t rows = shape[0];
        uint64_t cols = shape[1];
        EshkolGPUBuffer buf_in, buf_out;
        if (eshkol_gpu_wrap_host((void*)data, total * sizeof(double), &buf_in) == 0 &&
            eshkol_gpu_wrap_host((void*)out, total * sizeof(double), &buf_out) == 0) {
            if (eshkol_gpu_transpose_f64(&buf_in, &buf_out, rows, cols) == 0) {
                eshkol_gpu_free(&buf_in);
                eshkol_gpu_free(&buf_out);
                return result;
            }
        }
    }

    // CPU fallback — general N-dimensional transpose
    // Compute source strides
    uint64_t src_strides[16];
    src_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    // Compute destination strides
    uint64_t dst_strides[16];
    dst_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        dst_strides[i] = dst_strides[i + 1] * out_shape[i + 1];
    }

    // Transpose via index mapping
    for (uint64_t flat = 0; flat < total; flat++) {
        // Convert flat index to multi-dimensional indices in output space
        uint64_t remaining = flat;
        uint64_t src_flat = 0;
        for (int64_t d = 0; d < rank; d++) {
            uint64_t idx = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            // This output dimension d corresponds to source dimension perm[d]
            src_flat += idx * src_strides[perm[d]];
        }
        out[flat] = data[src_flat];
    }

    return result;
}

// ===== XLA Broadcast Runtime =====
// Broadcasts a tensor from src_shape to tgt_shape.
extern "C" void* eshkol_xla_broadcast(
    void* arena,
    const double* data,
    const uint64_t* src_shape,
    int64_t src_rank,
    const uint64_t* tgt_shape,
    int64_t tgt_rank) {

    if (!data || tgt_rank <= 0) return nullptr;

    uint64_t total = 1;
    for (int64_t i = 0; i < tgt_rank; i++) total *= tgt_shape[i];

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        arena, static_cast<uint64_t>(tgt_rank), total);
    if (!result) return nullptr;
    for (int64_t i = 0; i < tgt_rank; i++) result->dimensions[i] = tgt_shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Compute source strides (right-aligned with target)
    int64_t offset = tgt_rank - src_rank;

    // Compute target strides
    uint64_t tgt_strides[16];
    tgt_strides[tgt_rank - 1] = 1;
    for (int64_t i = tgt_rank - 2; i >= 0; i--) {
        tgt_strides[i] = tgt_strides[i + 1] * tgt_shape[i + 1];
    }

    // Compute source strides
    uint64_t src_strides[16];
    if (src_rank > 0) {
        src_strides[src_rank - 1] = 1;
        for (int64_t i = src_rank - 2; i >= 0; i--) {
            src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
        }
    }

    for (uint64_t flat = 0; flat < total; flat++) {
        uint64_t remaining = flat;
        uint64_t src_flat = 0;
        for (int64_t d = 0; d < tgt_rank; d++) {
            uint64_t idx = remaining / tgt_strides[d];
            remaining %= tgt_strides[d];
            int64_t src_d = d - offset;
            if (src_d >= 0 && src_d < src_rank && src_shape[src_d] > 1) {
                src_flat += idx * src_strides[src_d];
            }
        }
        out[flat] = data[src_flat];
    }

    return result;
}

// ===== XLA Slice Runtime =====
// Slices a tensor with starts, limits, and strides per dimension.
extern "C" void* eshkol_xla_slice(
    void* arena,
    const double* data,
    const uint64_t* shape,
    int64_t rank,
    const int64_t* starts,
    const int64_t* limits,
    const int64_t* strides) {

    if (!data || rank <= 0) return nullptr;

    // Compute output shape
    uint64_t out_shape[16];
    uint64_t total = 1;
    for (int64_t i = 0; i < rank; i++) {
        int64_t stride = strides ? strides[i] : 1;
        out_shape[i] = static_cast<uint64_t>((limits[i] - starts[i] + stride - 1) / stride);
        total *= out_shape[i];
    }

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        arena, static_cast<uint64_t>(rank), total);
    if (!result) return nullptr;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = out_shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Compute source strides
    uint64_t src_strides[16];
    src_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    // Compute output strides
    uint64_t dst_strides[16];
    dst_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        dst_strides[i] = dst_strides[i + 1] * out_shape[i + 1];
    }

    for (uint64_t flat = 0; flat < total; flat++) {
        uint64_t remaining = flat;
        uint64_t src_flat = 0;
        for (int64_t d = 0; d < rank; d++) {
            uint64_t idx = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            int64_t stride = strides ? strides[d] : 1;
            uint64_t src_idx = static_cast<uint64_t>(starts[d]) + idx * static_cast<uint64_t>(stride);
            src_flat += src_idx * src_strides[d];
        }
        out[flat] = data[src_flat];
    }

    return result;
}

namespace eshkol {
namespace xla {

// ===== XLARuntime Implementation =====

class XLARuntime::Impl {
public:
    bool initialized_ = false;
    Target target_ = Target::CPU;
    size_t allocated_bytes_ = 0;
    size_t peak_bytes_ = 0;
    std::unordered_map<void*, std::future<ExecutionResult>> async_handles_;
    size_t next_handle_id_ = 1;
};

XLARuntime::XLARuntime()
    : impl_(std::make_unique<Impl>()) {}

XLARuntime::~XLARuntime() = default;

// ===== Initialization =====

bool XLARuntime::initialize(Target target) {
    impl_->target_ = target;

    if (target == Target::CPU) {
        impl_->initialized_ = true;
    } else {
        // Initialize GPU backend if not already done
        int gpu_devices = eshkol_gpu_init();
        if (gpu_devices > 0) {
            // Verify the requested GPU backend is available
            EshkolGPUBackend backend = eshkol_gpu_get_backend();
            bool match = false;
            switch (target) {
                case Target::Metal:  match = (backend == ESHKOL_GPU_METAL);  break;
                case Target::CUDA:   match = (backend == ESHKOL_GPU_CUDA);   break;
                case Target::Vulkan: match = (backend == ESHKOL_GPU_VULKAN); break;
                default: break;
            }
            impl_->initialized_ = match;
        } else {
            // No GPU devices — fall back, but still mark as initialized
            // since XLA runtime functions have CPU fallbacks
            impl_->initialized_ = true;
        }
    }

    return impl_->initialized_;
}

bool XLARuntime::isInitialized() const {
    return impl_->initialized_;
}

Target XLARuntime::getTarget() const {
    return impl_->target_;
}

// ===== Execution =====
// Infrastructure for StableHLO JIT — execute/executeAsync provide the execution
// interface for compiled XLA programs. Currently unused; actual XLA ops dispatch
// through 11 C runtime functions called directly from codegen.

ExecutionResult XLARuntime::execute(void* executable,
                                     const std::vector<BufferDescriptor>& inputs,
                                     std::vector<BufferDescriptor>& outputs) {
    if (!impl_->initialized_ || !executable) {
        return ExecutionResult{
            .success = false,
            .error_message = "Runtime not initialized or null executable",
            .execution_time_ns = 0
        };
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Cast executable to function pointer and call directly
    // In LLVM direct mode, the executable is a function pointer to the compiled IR
    using ExecFn = void(*)(const void* const*, void* const*);
    auto fn = reinterpret_cast<ExecFn>(executable);

    // Build input/output pointer arrays
    std::vector<const void*> input_ptrs;
    std::vector<void*> output_ptrs;
    input_ptrs.reserve(inputs.size());
    output_ptrs.reserve(outputs.size());
    for (const auto& in : inputs) input_ptrs.push_back(in.data);
    for (auto& out : outputs) output_ptrs.push_back(out.data);

    fn(input_ptrs.data(), output_ptrs.data());

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return ExecutionResult{
        .success = true,
        .error_message = "",
        .execution_time_ns = ns
    };
}

void* XLARuntime::executeAsync(void* executable,
                                const std::vector<BufferDescriptor>& inputs,
                                std::vector<BufferDescriptor>& outputs) {
    if (!impl_->initialized_ || !executable) return nullptr;

    // Capture copies for the async task
    auto inputs_copy = inputs;
    auto outputs_copy = outputs;

    auto future = std::async(std::launch::async, [this, executable, inputs_copy, outputs_copy]() mutable {
        return this->execute(executable, inputs_copy, outputs_copy);
    });

    // Store the future and return a handle
    void* handle = reinterpret_cast<void*>(impl_->next_handle_id_++);
    impl_->async_handles_[handle] = std::move(future);
    return handle;
}

ExecutionResult XLARuntime::wait(void* handle) {
    auto it = impl_->async_handles_.find(handle);
    if (it == impl_->async_handles_.end()) {
        return ExecutionResult{
            .success = false,
            .error_message = "Invalid async execution handle",
            .execution_time_ns = 0
        };
    }

    auto result = it->second.get();
    impl_->async_handles_.erase(it);
    return result;
}

// ===== Buffer Management =====

BufferDescriptor XLARuntime::allocateDevice(const std::vector<int64_t>& shape,
                                             size_t element_size) {
    // CPU: allocate on host. Compute total size from shape.
    size_t total = 1;
    for (auto dim : shape) total *= static_cast<size_t>(dim);
    size_t size_bytes = total * element_size;

    void* data = std::calloc(total, element_size);
    if (data) {
        impl_->allocated_bytes_ += size_bytes;
        if (impl_->allocated_bytes_ > impl_->peak_bytes_) {
            impl_->peak_bytes_ = impl_->allocated_bytes_;
        }
    }

    return BufferDescriptor{
        .data = data,
        .shape = shape,
        .element_size = element_size,
        .on_device = false
    };
}

BufferDescriptor XLARuntime::toDevice(void* host_data,
                                       const std::vector<int64_t>& shape,
                                       size_t element_size) {
    // CPU: data is already on host — zero-copy wrap
    return BufferDescriptor{
        .data = host_data,
        .shape = shape,
        .element_size = element_size,
        .on_device = false
    };
}

void XLARuntime::toHost(const BufferDescriptor& device_buffer, void* host_data) {
    // CPU: data is already on host. If pointers differ, memcpy.
    if (device_buffer.data && host_data && device_buffer.data != host_data) {
        size_t total = 1;
        for (auto dim : device_buffer.shape) total *= static_cast<size_t>(dim);
        std::memcpy(host_data, device_buffer.data, total * device_buffer.element_size);
    }
}

void XLARuntime::freeBuffer(BufferDescriptor& buffer) {
    // Only free buffers we allocated (on_device=false for CPU allocs from allocateDevice)
    // Arena-managed buffers should NOT be freed here
    if (buffer.data) {
        size_t total = 1;
        for (auto dim : buffer.shape) total *= static_cast<size_t>(dim);
        size_t size_bytes = total * buffer.element_size;
        if (impl_->allocated_bytes_ >= size_bytes) {
            impl_->allocated_bytes_ -= size_bytes;
        }
    }
    buffer.data = nullptr;
}

// ===== Synchronization =====

void XLARuntime::synchronize() {
    // CPU: all operations are synchronous. Wait for any pending async executions.
    for (auto& [handle, future] : impl_->async_handles_) {
        if (future.valid()) future.wait();
    }
    impl_->async_handles_.clear();
}

// ===== Diagnostics =====

void XLARuntime::getMemoryStats(size_t& allocated_bytes, size_t& peak_bytes) {
    allocated_bytes = impl_->allocated_bytes_;
    peak_bytes = impl_->peak_bytes_;
}

std::string XLARuntime::getDescription() const {
    std::string desc = "XLA Runtime (";
    if (impl_->initialized_) {
        switch (impl_->target_) {
            case Target::CPU: desc += "CPU, LLVM direct dispatch"; break;
            case Target::CUDA: desc += "CUDA"; break;
            case Target::Metal: desc += "Metal"; break;
            case Target::Vulkan: desc += "Vulkan"; break;
        }
        desc += ")";
    } else {
        desc += "not initialized)";
    }
    return desc;
}

// ===== Global Runtime =====

XLARuntime& getDefaultRuntime() {
    static XLARuntime runtime;
    static std::once_flag init_flag;
    std::call_once(init_flag, [&]() {
        // Detect best available GPU target
        eshkol_gpu_init();
        EshkolGPUBackend backend = eshkol_gpu_get_backend();
        Target target;
        switch (backend) {
            case ESHKOL_GPU_METAL:  target = Target::Metal;  break;
            case ESHKOL_GPU_CUDA:   target = Target::CUDA;   break;
            case ESHKOL_GPU_VULKAN: target = Target::Vulkan;  break;
            default:                target = Target::CPU;      break;
        }
        runtime.initialize(target);
    });
    return runtime;
}

} // namespace xla
} // namespace eshkol
