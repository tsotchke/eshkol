/*
 * CUDA Compute Kernels for Eshkol GPU Backend
 *
 * Real CUDA kernels for elementwise, reduce, and transpose operations.
 * Compiled by nvcc when CUDA toolkit is available.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

// ============================================================================
// Elementwise Kernel
// ============================================================================

// Op codes must match EshkolElementwiseOp enum
__global__ void elementwise_f64_kernel(const double* a, const double* b,
                                        double* out, int64_t n, int op) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        switch (op) {
            case 0:  out[i] = a[i] + (b ? b[i] : 0.0); break; // ADD
            case 1:  out[i] = a[i] - (b ? b[i] : 0.0); break; // SUB
            case 2:  out[i] = a[i] * (b ? b[i] : 1.0); break; // MUL
            case 3:  out[i] = a[i] / (b ? b[i] : 1.0); break; // DIV
            case 4:  out[i] = -a[i]; break;                     // NEG
            case 5:  out[i] = fabs(a[i]); break;                // ABS
            case 6:  out[i] = exp(a[i]); break;                 // EXP
            case 7:  out[i] = log(a[i]); break;                 // LOG
            case 8:  out[i] = sin(a[i]); break;                 // SIN
            case 9:  out[i] = cos(a[i]); break;                 // COS
            case 10: out[i] = tanh(a[i]); break;                // TANH
            case 11: out[i] = a[i] > 0.0 ? a[i] : 0.0; break;  // RELU
            case 12: out[i] = 1.0 / (1.0 + exp(-a[i])); break;  // SIGMOID
            case 13: out[i] = sqrt(a[i]); break;                 // SQRT
            case 14: out[i] = 1.0 / a[i]; break;                 // RECIPROCAL
        }
    }
}

// ============================================================================
// Full-Tensor Reduce Kernel (two-pass: block reduce → CPU finalize)
// ============================================================================

// Warp-level shuffle reduction
__device__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ double warp_reduce_max(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ double warp_reduce_min(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmin(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ double warp_reduce_prod(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val *= __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block-level reduction using shared memory + warp shuffle
// Each block reduces its chunk to a single value
__global__ void reduce_f64_kernel(const double* in, double* block_results,
                                   int64_t n, int op) {
    __shared__ double shared[32]; // One slot per warp (max 1024 threads = 32 warps)

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    // Initialize accumulator based on op
    double acc;
    switch (op) {
        case 0: case 4: acc = 0.0; break;  // SUM, MEAN
        case 1: acc = 1.0; break;           // PROD
        case 2: acc = INFINITY; break;      // MIN
        case 3: acc = -INFINITY; break;     // MAX
    }

    // Grid-stride loop: each thread accumulates multiple elements
    for (int64_t i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        double val = in[i];
        switch (op) {
            case 0: case 4: acc += val; break;
            case 1: acc *= val; break;
            case 2: acc = fmin(acc, val); break;
            case 3: acc = fmax(acc, val); break;
        }
    }

    // Warp-level reduction
    double warp_result;
    switch (op) {
        case 0: case 4: warp_result = warp_reduce_sum(acc); break;
        case 1: warp_result = warp_reduce_prod(acc); break;
        case 2: warp_result = warp_reduce_min(acc); break;
        case 3: warp_result = warp_reduce_max(acc); break;
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) shared[warp_id] = warp_result;
    __syncthreads();

    // First warp reduces the per-warp results
    if (warp_id == 0) {
        double val;
        if (lane_id < num_warps) {
            val = shared[lane_id];
        } else {
            switch (op) {
                case 0: case 4: val = 0.0; break;
                case 1: val = 1.0; break;
                case 2: val = INFINITY; break;
                case 3: val = -INFINITY; break;
            }
        }

        switch (op) {
            case 0: case 4: val = warp_reduce_sum(val); break;
            case 1: val = warp_reduce_prod(val); break;
            case 2: val = warp_reduce_min(val); break;
            case 3: val = warp_reduce_max(val); break;
        }

        if (lane_id == 0) block_results[blockIdx.x] = val;
    }
}

// ============================================================================
// Axis-Reduce Kernel
// ============================================================================

__global__ void reduce_f64_axis_kernel(const double* in, double* out,
                                        uint64_t rank, const uint64_t* dims,
                                        uint64_t axis, int op, uint64_t out_size) {
    for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < out_size; idx += blockDim.x * gridDim.x) {

        uint64_t axis_len = dims[axis];
        uint64_t inner_stride = 1;
        for (uint64_t d = axis + 1; d < rank; d++) inner_stride *= dims[d];
        uint64_t outer = idx / inner_stride;
        uint64_t inner = idx % inner_stride;

        double acc;
        switch (op) {
            case 0: case 4: acc = 0.0; break;  // SUM, MEAN
            case 1: acc = 1.0; break;           // PROD
            case 2: acc = INFINITY; break;      // MIN
            case 3: acc = -INFINITY; break;     // MAX
        }

        for (uint64_t k = 0; k < axis_len; k++) {
            uint64_t src_idx = outer * axis_len * inner_stride + k * inner_stride + inner;
            double val = in[src_idx];
            switch (op) {
                case 0: case 4: acc += val; break;
                case 1: acc *= val; break;
                case 2: acc = fmin(acc, val); break;
                case 3: acc = fmax(acc, val); break;
            }
        }
        if (op == 4) acc /= (double)axis_len; // MEAN
        out[idx] = acc;
    }
}

// ============================================================================
// Transpose Kernel (tiled with shared memory)
// ============================================================================

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_f64_kernel(const double* in, double* out,
                                      uint64_t rows, uint64_t cols) {
    // Shared memory with +1 padding to avoid bank conflicts
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input (coalesced reads)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < (int)rows && x < (int)cols) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
        }
    }
    __syncthreads();

    // Write tile to output (coalesced writes) with transposed indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < (int)cols && x < (int)rows) {
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Host-Callable Launcher Functions (extern "C")
// ============================================================================

extern "C" {

int cuda_launch_elementwise_f64(const double* a, const double* b, double* out,
                                 int64_t n, int op, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block_size = 256;
    int grid_size = (int)((n + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    elementwise_f64_kernel<<<grid_size, block_size, 0, s>>>(a, b, out, n, op);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

int cuda_launch_reduce_f64(const double* in, double* out, int64_t n, int op,
                            void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block_size = 256;
    int grid_size = (int)((n + block_size - 1) / block_size);
    if (grid_size > 1024) grid_size = 1024; // Limit blocks for reduce

    // Allocate temporary block results
    double* block_results;
    cudaError_t err = cudaMalloc(&block_results, grid_size * sizeof(double));
    if (err != cudaSuccess) return -1;

    // Launch kernel
    reduce_f64_kernel<<<grid_size, block_size, 0, s>>>(in, block_results, n, op);
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(block_results); return -1; }

    if (grid_size == 1) {
        // Single block — result is directly in block_results[0]
        cudaMemcpyAsync(out, block_results, sizeof(double),
                         cudaMemcpyDeviceToDevice, s);
    } else {
        // Second pass: reduce block results (small enough for single block)
        reduce_f64_kernel<<<1, 256, 0, s>>>(block_results, out, grid_size, op);
        err = cudaGetLastError();
        if (err != cudaSuccess) { cudaFree(block_results); return -1; }
    }

    // For MEAN, divide by n
    if (op == 4) {
        // Simple kernel to divide result by n
        double result;
        cudaStreamSynchronize(s);
        cudaMemcpy(&result, out, sizeof(double), cudaMemcpyDeviceToHost);
        result /= (double)n;
        cudaMemcpy(out, &result, sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaStreamSynchronize(s);
    cudaFree(block_results);
    return 0;
}

int cuda_launch_reduce_axis_f64(const double* in, double* out,
                                 uint64_t rank, const uint64_t* dims,
                                 uint64_t axis, int op, uint64_t out_size,
                                 void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // Copy dims to device
    uint64_t* d_dims;
    cudaError_t err = cudaMalloc(&d_dims, rank * sizeof(uint64_t));
    if (err != cudaSuccess) return -1;
    cudaMemcpyAsync(d_dims, dims, rank * sizeof(uint64_t), cudaMemcpyHostToDevice, s);

    int block_size = 256;
    int grid_size = (int)((out_size + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    reduce_f64_axis_kernel<<<grid_size, block_size, 0, s>>>(in, out, rank, d_dims, axis, op, out_size);
    err = cudaGetLastError();
    cudaStreamSynchronize(s);
    cudaFree(d_dims);
    return (err == cudaSuccess) ? 0 : -1;
}

int cuda_launch_transpose_f64(const double* in, double* out,
                               uint64_t rows, uint64_t cols, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((unsigned int)((cols + TILE_DIM - 1) / TILE_DIM),
              (unsigned int)((rows + TILE_DIM - 1) / TILE_DIM));

    transpose_f64_kernel<<<grid, block, 0, s>>>(in, out, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

// ============================================================================
// Softmax Kernel — Numerically Stable (one thread per slice)
// ============================================================================

__global__ void softmax_f64_kernel(const double* in, double* out,
                                    uint64_t num_slices, uint64_t slice_len) {
    for (uint64_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_slices; s += blockDim.x * gridDim.x) {
        uint64_t base = s * slice_len;

        // Pass 1: max
        double max_val = in[base];
        for (uint64_t k = 1; k < slice_len; k++)
            max_val = fmax(max_val, in[base + k]);

        // Pass 2: exp(x - max) + sum
        double sum_exp = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) {
            double e = exp(in[base + k] - max_val);
            out[base + k] = e;
            sum_exp += e;
        }

        // Pass 3: normalize
        if (sum_exp == 0.0) sum_exp = 1.0;
        for (uint64_t k = 0; k < slice_len; k++)
            out[base + k] /= sum_exp;
    }
}

int cuda_launch_softmax_f64(const double* in, double* out,
                              uint64_t num_slices, uint64_t slice_len,
                              void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block_size = 256;
    int grid_size = (int)((num_slices + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    softmax_f64_kernel<<<grid_size, block_size, 0, s>>>(in, out, num_slices, slice_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

// ============================================================================
// Normalize Kernel — Layer Normalization (one thread per slice)
// ============================================================================

__global__ void normalize_f64_kernel(const double* in, double* out,
                                      uint64_t num_slices, uint64_t slice_len,
                                      double gamma, double beta, double epsilon) {
    for (uint64_t s = blockIdx.x * blockDim.x + threadIdx.x;
         s < num_slices; s += blockDim.x * gridDim.x) {
        uint64_t base = s * slice_len;
        double n = (double)slice_len;

        // Mean
        double sum = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) sum += in[base + k];
        double mean = sum / n;

        // Variance
        double var_sum = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) {
            double diff = in[base + k] - mean;
            var_sum += diff * diff;
        }
        double inv_std = rsqrt(var_sum / n + epsilon);

        // Normalize
        for (uint64_t k = 0; k < slice_len; k++)
            out[base + k] = gamma * (in[base + k] - mean) * inv_std + beta;
    }
}

int cuda_launch_normalize_f64(const double* in, double* out,
                                uint64_t num_slices, uint64_t slice_len,
                                double gamma, double beta, double epsilon,
                                void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block_size = 256;
    int grid_size = (int)((num_slices + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    normalize_f64_kernel<<<grid_size, block_size, 0, s>>>(in, out, num_slices, slice_len,
                                                           gamma, beta, epsilon);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

} // extern "C"
