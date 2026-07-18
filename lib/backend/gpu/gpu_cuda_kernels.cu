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

// ============================================================================
// Backward Pass Kernels
// ============================================================================

// Conv2d backward — grad_input: one thread per input element
__global__ void conv2d_backward_input_f64_kernel(
    const double* grad_out, const double* kernel_weights, double* grad_input,
    int in_h, int in_w, int k_h, int k_w,
    int out_h, int out_w, int stride_h, int stride_w,
    int channels_in, int channels_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockIdx.z; // batch * channels_in encoded in grid z
    int ci = total % channels_in;
    int b = total / channels_in;
    int ih = (idx / in_w);
    int iw = idx % in_w;
    if (ih >= in_h || iw >= in_w) return;

    int in_spatial = in_h * in_w;
    int out_spatial = out_h * out_w;
    int kernel_spatial = k_h * k_w;

    double sum = 0.0;
    for (int co = 0; co < channels_out; co++) {
        for (int kh = 0; kh < k_h; kh++) {
            for (int kw = 0; kw < k_w; kw++) {
                int oh_cand = ih - kh;
                int ow_cand = iw - kw;
                if (oh_cand < 0 || ow_cand < 0) continue;
                if (oh_cand % stride_h != 0 || ow_cand % stride_w != 0) continue;
                int oh = oh_cand / stride_h;
                int ow = ow_cand / stride_w;
                if (oh >= out_h || ow >= out_w) continue;

                double g = grad_out[(b * channels_out + co) * out_spatial + oh * out_w + ow];
                double k = kernel_weights[(co * channels_in + ci) * kernel_spatial + kh * k_w + kw];
                sum += g * k;
            }
        }
    }
    grad_input[(b * channels_in + ci) * in_spatial + ih * in_w + iw] = sum;
}

// Conv2d backward — grad_kernel: one thread per kernel element
__global__ void conv2d_backward_kernel_f64_kernel(
    const double* grad_out, const double* saved_input, double* grad_kernel,
    int in_h, int in_w, int k_h, int k_w,
    int out_h, int out_w, int stride_h, int stride_w,
    int channels_in, int channels_out, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int co_ci = blockIdx.z;
    int ci = co_ci % channels_in;
    int co = co_ci / channels_in;
    if (co >= channels_out) return;

    int kh_idx = idx / k_w;
    int kw_idx = idx % k_w;
    if (kh_idx >= k_h || kw_idx >= k_w) return;

    int in_spatial = in_h * in_w;
    int out_spatial = out_h * out_w;
    int kernel_spatial = k_h * k_w;

    double sum = 0.0;
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int ih = oh * stride_h + kh_idx;
                int iw = ow * stride_w + kw_idx;
                double g = grad_out[(b * channels_out + co) * out_spatial + oh * out_w + ow];
                double inp = saved_input[(b * channels_in + ci) * in_spatial + ih * in_w + iw];
                sum += g * inp;
            }
        }
    }
    grad_kernel[(co * channels_in + ci) * kernel_spatial + kh_idx * k_w + kw_idx] = sum;
}

// BatchNorm backward — one thread per feature
__global__ void batchnorm_backward_f64_kernel(
    const double* grad_out, const double* saved_input,
    const double* saved_mean, const double* saved_inv_std,
    const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int batch_size, int feature_size)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= feature_size) return;

    double mean_f = saved_mean[f];
    double inv_std_f = saved_inv_std[f];
    double gamma_f = saved_gamma[f];
    double N = (double)batch_size;

    double sum_g = 0.0, sum_gn = 0.0;
    for (int b = 0; b < batch_size; b++) {
        int idx = b * feature_size + f;
        double g = grad_out[idx];
        double normalized = (saved_input[idx] - mean_f) * inv_std_f;
        sum_g += g;
        sum_gn += g * normalized;
    }
    grad_beta[f] = sum_g;
    grad_gamma[f] = sum_gn;

    double coeff = gamma_f * inv_std_f / N;
    for (int b = 0; b < batch_size; b++) {
        int idx = b * feature_size + f;
        double g = grad_out[idx];
        double normalized = (saved_input[idx] - mean_f) * inv_std_f;
        grad_input[idx] = coeff * (N * g - sum_g - normalized * sum_gn);
    }
}

// LayerNorm backward — one thread per sample
__global__ void layernorm_backward_f64_kernel(
    const double* grad_out, const double* saved_input,
    const double* saved_mean, const double* saved_inv_std,
    const double* saved_gamma,
    double* grad_input,
    int num_samples, int feature_size)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_samples) return;

    double mean_s = saved_mean[s];
    double inv_std_s = saved_inv_std[s];
    double D = (double)feature_size;

    double sum_grad = 0.0, sum_grad_norm = 0.0;
    for (int f = 0; f < feature_size; f++) {
        int idx = s * feature_size + f;
        double g = grad_out[idx] * saved_gamma[f];
        double normalized = (saved_input[idx] - mean_s) * inv_std_s;
        sum_grad += g;
        sum_grad_norm += g * normalized;
    }

    double inv_D = inv_std_s / D;
    for (int f = 0; f < feature_size; f++) {
        int idx = s * feature_size + f;
        double g = grad_out[idx] * saved_gamma[f];
        double normalized = (saved_input[idx] - mean_s) * inv_std_s;
        grad_input[idx] = inv_D * (D * g - sum_grad - normalized * sum_grad_norm);
    }
}

// ============================================================================
// Backward Pass Launch Wrappers
// ============================================================================

extern "C" int cuda_conv2d_backward_input_f64(
    const double* grad_out, const double* kernel_weights, double* grad_input,
    int in_h, int in_w, int k_h, int k_w,
    int out_h, int out_w, int stride_h, int stride_w,
    int channels_in, int channels_out, int batch_size,
    cudaStream_t s)
{
    dim3 block(256);
    dim3 grid((in_h * in_w + 255) / 256, 1, batch_size * channels_in);
    conv2d_backward_input_f64_kernel<<<grid, block, 0, s>>>(
        grad_out, kernel_weights, grad_input,
        in_h, in_w, k_h, k_w, out_h, out_w,
        stride_h, stride_w, channels_in, channels_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

extern "C" int cuda_conv2d_backward_kernel_f64(
    const double* grad_out, const double* saved_input, double* grad_kernel,
    int in_h, int in_w, int k_h, int k_w,
    int out_h, int out_w, int stride_h, int stride_w,
    int channels_in, int channels_out, int batch_size,
    cudaStream_t s)
{
    dim3 block(256);
    dim3 grid((k_h * k_w + 255) / 256, 1, channels_out * channels_in);
    conv2d_backward_kernel_f64_kernel<<<grid, block, 0, s>>>(
        grad_out, saved_input, grad_kernel,
        in_h, in_w, k_h, k_w, out_h, out_w,
        stride_h, stride_w, channels_in, channels_out, batch_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

extern "C" int cuda_batchnorm_backward_f64(
    const double* grad_out, const double* saved_input,
    const double* saved_mean, const double* saved_inv_std,
    const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int batch_size, int feature_size,
    cudaStream_t s)
{
    int block_size = 256;
    int grid_size = (feature_size + block_size - 1) / block_size;
    batchnorm_backward_f64_kernel<<<grid_size, block_size, 0, s>>>(
        grad_out, saved_input, saved_mean, saved_inv_std, saved_gamma,
        grad_input, grad_gamma, grad_beta, batch_size, feature_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

extern "C" int cuda_layernorm_backward_f64(
    const double* grad_out, const double* saved_input,
    const double* saved_mean, const double* saved_inv_std,
    const double* saved_gamma,
    double* grad_input,
    int num_samples, int feature_size,
    cudaStream_t s)
{
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;
    layernorm_backward_f64_kernel<<<grid_size, block_size, 0, s>>>(
        grad_out, saved_input, saved_mean, saved_inv_std, saved_gamma,
        grad_input, num_samples, feature_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaStreamSynchronize(s);
    return 0;
}

} // extern "C"

// ============================================================================
// INT8 Tensor-Core Ozaki f64 GEMM kernels (opt-in high-throughput DGEMM)
// ============================================================================
// Recover FP64-accurate C = A*B from the INT8 (IMMA) tensor cores, which run
// ~500x faster than the crippled native FP64 pipeline on consumer/prosumer
// NVIDIA GPUs (GeForce Ampere f64 = 1/64 FP32). Ootomo-Ozaki-Yokota per-row/col
// scaled 7-bit integer slicing. The cuBLAS orchestration (INT8 GEMMs +
// diagonal-fused reconstruction) lives in gpu_memory_cuda.cpp; these are the
// scale/slice/reconstruct device kernels. See docs/breakdown/GPU_ACCELERATION.md.
//
// Row-major (Eshkol) convention: A is MxK, B is KxN, C is MxN.
//   r_i = max_k|A[i,k]| (per row of A),  c_j = max_k|B[k,j]| (per col of B)
//   Abar = A/r in [-1,1] sliced into s int8 slices A_p (base 128, |A_p|<=127)
//   Bbar = B/c in [-1,1] sliced into s int8 slices B_q (base 128, |B_q|<=127)
//   G_{p,q}[i,j] = sum_k A_p[i,k]*B_q[k,j]   (one INT8->INT32 tensor-core GEMM)
//   C[i,j] = r_i c_j * sum_{p+q<=T} 128^{-(p+q+2)} G_{p,q}[i,j]
//
// A_p slices are stored in natural row-major order (which is exactly col-major
// KxM A_p^T); B_q slices are stored TRANSPOSED to col-major KxN. With those
// stores the per-pair GEMM issued as cublasGemmEx(OP_T, OP_N, m=N,n=M,k=K,
// B_q, A_p) computes col-major NxM G^T (== row-major MxN G) and lands on the
// fast IMMA "TN" path — mandatory on sm_86 (3.7x cliff otherwise; Blackwell is
// layout-indifferent so TN is safe everywhere).

// per-row max of |A|, A row-major MxK -> rmax[M]
__global__ void ozaki_rowmax_f64_kernel(const double* A, uint64_t M, uint64_t K,
                                        double* rmax) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < M; i += (uint64_t)blockDim.x * gridDim.x) {
        double m = 0.0;
        const double* row = A + i * K;
        for (uint64_t k = 0; k < K; k++) { double v = fabs(row[k]); if (v > m) m = v; }
        rmax[i] = (m > 0.0) ? m : 1.0;
    }
}

// per-col max of |B|, B row-major KxN -> cmax[N]
__global__ void ozaki_colmax_f64_kernel(const double* B, uint64_t K, uint64_t N,
                                        double* cmax) {
    for (uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         j < N; j += (uint64_t)blockDim.x * gridDim.x) {
        double m = 0.0;
        for (uint64_t k = 0; k < K; k++) { double v = fabs(B[k * N + j]); if (v > m) m = v; }
        cmax[j] = (m > 0.0) ? m : 1.0;
    }
}

// Split A (row-major MxK) into s int8 slices, scaled per-row by rmax.
// Slice p stored at slices[p*M*K + (i*K+k)] (natural == col-major KxM A_p^T).
__global__ void ozaki_split_a_f64_kernel(const double* A, uint64_t M, uint64_t K,
                                         const double* rmax, int s, int8_t* slices) {
    uint64_t tot = M * K;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < tot; idx += (uint64_t)blockDim.x * gridDim.x) {
        uint64_t i = idx / K;
        double res = A[idx] / rmax[i];        // in [-1,1]
        for (int p = 0; p < s; p++) {
            res *= 128.0;
            double a = rint(res);
            if (a > 127.0) a = 127.0; else if (a < -127.0) a = -127.0;
            slices[(uint64_t)p * tot + idx] = (int8_t)a;
            res -= a;
        }
    }
}

// Split B (row-major KxN) into s int8 slices, scaled per-col by cmax.
// Slice q stored TRANSPOSED at slices[q*K*N + (k + j*K)] (col-major KxN).
__global__ void ozaki_split_b_f64_kernel(const double* B, uint64_t K, uint64_t N,
                                         const double* cmax, int s, int8_t* slices) {
    uint64_t tot = K * N;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < tot; idx += (uint64_t)blockDim.x * gridDim.x) {
        uint64_t k = idx / N, j = idx % N;
        uint64_t tidx = k + j * K;            // transposed store -> col-major KxN
        double res = B[idx] / cmax[j];
        for (int q = 0; q < s; q++) {
            res *= 128.0;
            double b = rint(res);
            if (b > 127.0) b = 127.0; else if (b < -127.0) b = -127.0;
            slices[(uint64_t)q * tot + tidx] = (int8_t)b;
            res -= b;
        }
    }
}

// Reconstruct C (row-major MxN) from nbuf int32 diagonal buffers (each MxN, in
// col-major NxM G^T layout == row-major MxN), applying the per-buffer weight,
// summing in FP64, and scaling by r_i*c_j.
__global__ void ozaki_reconstruct_f64_kernel(double* C, const int32_t* Gall,
                                             const double* weights, int nbuf,
                                             const double* rmax, const double* cmax,
                                             uint64_t M, uint64_t N) {
    uint64_t tot = M * N;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < tot; idx += (uint64_t)blockDim.x * gridDim.x) {
        uint64_t i = idx / N, j = idx % N;
        double acc = 0.0;
        for (int b = 0; b < nbuf; b++)
            acc += weights[b] * (double)Gall[(uint64_t)b * tot + idx];
        C[idx] = acc * rmax[i] * cmax[j];
    }
}

extern "C" {

// Compute per-row max of A then split A into `s` int8 slices (natural layout).
int cuda_ozaki_split_a_f64(const double* A, uint64_t M, uint64_t K, int s,
                           double* rmax, int8_t* slices, void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    int bs = 256;
    int g1 = (int)((M + bs - 1) / bs); if (g1 > 65535) g1 = 65535; if (g1 < 1) g1 = 1;
    ozaki_rowmax_f64_kernel<<<g1, bs, 0, st>>>(A, M, K, rmax);
    if (cudaGetLastError() != cudaSuccess) return -1;
    uint64_t tot = M * K;
    int g2 = (int)((tot + bs - 1) / bs); if (g2 > 65535) g2 = 65535; if (g2 < 1) g2 = 1;
    ozaki_split_a_f64_kernel<<<g2, bs, 0, st>>>(A, M, K, rmax, s, slices);
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaStreamSynchronize(st);
    return 0;
}

// Compute per-col max of B then split B into `s` int8 slices (transposed layout).
int cuda_ozaki_split_b_f64(const double* B, uint64_t K, uint64_t N, int s,
                           double* cmax, int8_t* slices, void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    int bs = 256;
    int g1 = (int)((N + bs - 1) / bs); if (g1 > 65535) g1 = 65535; if (g1 < 1) g1 = 1;
    ozaki_colmax_f64_kernel<<<g1, bs, 0, st>>>(B, K, N, cmax);
    if (cudaGetLastError() != cudaSuccess) return -1;
    uint64_t tot = K * N;
    int g2 = (int)((tot + bs - 1) / bs); if (g2 > 65535) g2 = 65535; if (g2 < 1) g2 = 1;
    ozaki_split_b_f64_kernel<<<g2, bs, 0, st>>>(B, K, N, cmax, s, slices);
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaStreamSynchronize(st);
    return 0;
}

// Reconstruct C from the diagonal-fused int32 buffers. `h_weights` is a host
// array of nbuf per-buffer weights; it is staged to the device internally.
int cuda_ozaki_reconstruct_f64(double* C, const int32_t* Gall,
                               const double* h_weights, int nbuf,
                               const double* rmax, const double* cmax,
                               uint64_t M, uint64_t N, void* stream) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    double* d_w = nullptr;
    if (cudaMalloc(&d_w, (size_t)nbuf * sizeof(double)) != cudaSuccess) return -1;
    if (cudaMemcpyAsync(d_w, h_weights, (size_t)nbuf * sizeof(double),
                        cudaMemcpyHostToDevice, st) != cudaSuccess) {
        cudaFree(d_w); return -1;
    }
    uint64_t tot = M * N;
    int bs = 256;
    int g = (int)((tot + bs - 1) / bs); if (g > 65535) g = 65535; if (g < 1) g = 1;
    ozaki_reconstruct_f64_kernel<<<g, bs, 0, st>>>(C, Gall, d_w, nbuf, rmax, cmax, M, N);
    cudaError_t err = cudaGetLastError();
    cudaStreamSynchronize(st);
    cudaFree(d_w);
    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"
