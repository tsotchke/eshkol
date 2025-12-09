# Eshkol Required Improvements: Complete Implementation Guide

## Document Overview

This document specifies **every improvement** required to make Eshkol the complete solution for systems-level advanced intelligent computing. Each section provides:

- Current state analysis
- Detailed technical requirements
- Implementation specifications with code
- File modifications required
- Effort estimates
- Dependencies and ordering

---

## Table of Contents

1. [SIMD/Vectorization](#1-simdvectorization)
2. [GPU/CUDA Support](#2-gpucuda-support)
3. [XLA Integration](#3-xla-integration)
4. [Parallel Execution](#4-parallel-execution)
5. [Macro System](#5-macro-system)
6. [Neural Network Primitives](#6-neural-network-primitives)
7. [Distributed Computing](#7-distributed-computing)
8. [Serialization & Checkpointing](#8-serialization--checkpointing)
9. [Profiling & Debugging](#9-profiling--debugging)
10. [Extended Tensor Operations](#10-extended-tensor-operations)
11. [Optimizer Implementations](#11-optimizer-implementations)
12. [Memory Optimization](#12-memory-optimization)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. SIMD/Vectorization

### 1.1 Current State

**Status:** ❌ NOT IMPLEMENTED

No SIMD support exists. All tensor operations are scalar loops:

```cpp
// Current: Scalar loop in tensor_codegen.cpp
for (size_t i = 0; i < total_elements; i++) {
    result[i] = a[i] + b[i];
}
```

### 1.2 Required Additions

#### 1.2.1 LLVM Vector Types

**File:** `lib/backend/tensor_codegen.cpp`

```cpp
// Add SIMD vector types
llvm::Type* getSimdType(CodegenContext& ctx, size_t width) {
    switch (width) {
        case 2: return llvm::FixedVectorType::get(ctx.doubleType(), 2);   // SSE
        case 4: return llvm::FixedVectorType::get(ctx.doubleType(), 4);   // AVX
        case 8: return llvm::FixedVectorType::get(ctx.doubleType(), 8);   // AVX-512
        default: return ctx.doubleType();
    }
}

// Detect best SIMD width at compile time
size_t getBestSimdWidth() {
    llvm::StringMap<bool> features;
    llvm::sys::getHostCPUFeatures(features);

    if (features["avx512f"]) return 8;
    if (features["avx"]) return 4;
    if (features["sse2"]) return 2;
    return 1;  // Scalar fallback
}
```

#### 1.2.2 Vectorized Tensor Operations

**File:** `lib/backend/tensor_codegen.cpp`

```cpp
llvm::Value* TensorCodegen::tensorAddSimd(llvm::Value* a, llvm::Value* b,
                                           size_t total_elements) {
    size_t simd_width = getBestSimdWidth();
    llvm::Type* vec_type = getSimdType(ctx_, simd_width);

    // Allocate result tensor
    llvm::Value* result = allocateTensor(total_elements);

    // Get element pointers
    llvm::Value* a_ptr = getTensorElements(a);
    llvm::Value* b_ptr = getTensorElements(b);
    llvm::Value* r_ptr = getTensorElements(result);

    // Vectorized loop
    size_t vec_iterations = total_elements / simd_width;
    for (size_t i = 0; i < vec_iterations; i++) {
        size_t offset = i * simd_width;

        // Load vectors
        llvm::Value* a_vec = ctx_.builder().CreateAlignedLoad(
            vec_type,
            ctx_.builder().CreateGEP(ctx_.doubleType(), a_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), offset)),
            llvm::Align(simd_width * 8));

        llvm::Value* b_vec = ctx_.builder().CreateAlignedLoad(
            vec_type,
            ctx_.builder().CreateGEP(ctx_.doubleType(), b_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), offset)),
            llvm::Align(simd_width * 8));

        // SIMD add
        llvm::Value* r_vec = ctx_.builder().CreateFAdd(a_vec, b_vec);

        // Store result
        ctx_.builder().CreateAlignedStore(
            r_vec,
            ctx_.builder().CreateGEP(ctx_.doubleType(), r_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), offset)),
            llvm::Align(simd_width * 8));
    }

    // Scalar cleanup for remainder
    for (size_t i = vec_iterations * simd_width; i < total_elements; i++) {
        // ... scalar operations for remaining elements
    }

    return result;
}
```

#### 1.2.3 Memory Alignment for SIMD

**File:** `lib/core/arena_memory.cpp`

```cpp
// Update allocation to ensure SIMD alignment
void* arena_allocate_simd_aligned(arena_t* arena, size_t size) {
    size_t alignment = 64;  // AVX-512 alignment
    return arena_allocate_aligned(arena, size, alignment);
}

// Add tensor allocation with SIMD alignment
eshkol_tensor_t* arena_allocate_tensor_simd(arena_t* arena,
                                             size_t* dims,
                                             size_t num_dims) {
    size_t total = 1;
    for (size_t i = 0; i < num_dims; i++) total *= dims[i];

    eshkol_tensor_t* tensor = arena_allocate_simd_aligned(arena,
        sizeof(eshkol_tensor_t));
    tensor->elements = arena_allocate_simd_aligned(arena,
        total * sizeof(double));
    // ... rest of initialization
    return tensor;
}
```

#### 1.2.4 FMA (Fused Multiply-Add) Support

**File:** `lib/backend/tensor_codegen.cpp`

```cpp
// Use FMA intrinsics for matmul inner loop
llvm::Value* TensorCodegen::fmaVector(llvm::Value* a, llvm::Value* b,
                                       llvm::Value* c) {
    // a * b + c with single rounding
    llvm::Function* fma_intrinsic = llvm::Intrinsic::getDeclaration(
        &ctx_.module(),
        llvm::Intrinsic::fma,
        {getSimdType(ctx_, getBestSimdWidth())}
    );
    return ctx_.builder().CreateCall(fma_intrinsic, {a, b, c});
}
```

### 1.3 Files to Modify

| File | Changes |
|------|---------|
| `lib/backend/tensor_codegen.cpp` | Add SIMD operations |
| `inc/eshkol/backend/tensor_codegen.h` | SIMD function declarations |
| `lib/core/arena_memory.cpp` | SIMD-aligned allocation |
| `lib/core/arena_memory.h` | Alignment constants |
| `CMakeLists.txt` | AVX/AVX-512 compiler flags |

### 1.4 Compiler Flags

**File:** `CMakeLists.txt`

```cmake
# Detect and enable SIMD
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq")
    add_definitions(-DESHKOL_AVX512_ENABLED)
elseif(COMPILER_SUPPORTS_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mfma")
    add_definitions(-DESHKOL_AVX_ENABLED)
endif()
```

### 1.5 Effort Estimate

| Task | Time |
|------|------|
| SIMD type infrastructure | 2 days |
| Vectorized element-wise ops | 3 days |
| Vectorized matmul | 3 days |
| Memory alignment | 1 day |
| Testing & validation | 3 days |
| **Total** | **12 days** |

---

## 2. GPU/CUDA Support

### 2.1 Current State

**Status:** ❌ NOT IMPLEMENTED

No GPU support. All computation runs on CPU.

### 2.2 Required Additions

#### 2.2.1 CUDA Runtime Integration

**New File:** `lib/backend/cuda_runtime.cpp`

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

namespace eshkol {
namespace cuda {

// Global CUDA state
struct CudaContext {
    int device_id;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    size_t memory_pool_size;
    void* memory_pool;
};

static CudaContext* g_cuda_ctx = nullptr;

// Initialize CUDA
bool cuda_init(int device_id = 0) {
    if (g_cuda_ctx) return true;

    g_cuda_ctx = new CudaContext();
    g_cuda_ctx->device_id = device_id;

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&g_cuda_ctx->stream));
    CUBLAS_CHECK(cublasCreate(&g_cuda_ctx->cublas_handle));
    CUBLAS_CHECK(cublasSetStream(g_cuda_ctx->cublas_handle, g_cuda_ctx->stream));
    CUDNN_CHECK(cudnnCreate(&g_cuda_ctx->cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(g_cuda_ctx->cudnn_handle, g_cuda_ctx->stream));

    return true;
}

// GPU memory allocation
void* cuda_allocate(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

// Host-device transfers
void cuda_copy_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                g_cuda_ctx->stream));
}

void cuda_copy_to_host(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                g_cuda_ctx->stream));
}

void cuda_synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(g_cuda_ctx->stream));
}

} // namespace cuda
} // namespace eshkol
```

#### 2.2.2 GPU Tensor Type

**New File:** `inc/eshkol/cuda/gpu_tensor.h`

```cpp
namespace eshkol {
namespace cuda {

enum class TensorLocation {
    CPU,
    GPU,
    UNIFIED  // Managed memory
};

struct GpuTensor {
    double* data;           // Device pointer
    size_t* dims;           // Host-side dimensions
    size_t num_dims;
    size_t total_elements;
    TensorLocation location;
    bool owns_data;         // For views

    // Lazy transfer tracking
    bool cpu_valid;
    bool gpu_valid;
    double* cpu_data;       // Host mirror
};

// Create GPU tensor
GpuTensor* gpu_tensor_create(size_t* dims, size_t num_dims);

// Transfer operations
void gpu_tensor_to_device(GpuTensor* tensor);
void gpu_tensor_to_host(GpuTensor* tensor);

// Ensure data is on correct device
void gpu_tensor_ensure_device(GpuTensor* tensor);
void gpu_tensor_ensure_host(GpuTensor* tensor);

} // namespace cuda
} // namespace eshkol
```

#### 2.2.3 cuBLAS Integration for Matrix Operations

**New File:** `lib/backend/cuda_tensor_ops.cpp`

```cpp
namespace eshkol {
namespace cuda {

// GPU matrix multiplication via cuBLAS
GpuTensor* gpu_matmul(GpuTensor* A, GpuTensor* B) {
    // Validate dimensions
    assert(A->num_dims == 2 && B->num_dims == 2);
    assert(A->dims[1] == B->dims[0]);  // A: [M,K], B: [K,N]

    size_t M = A->dims[0];
    size_t K = A->dims[1];
    size_t N = B->dims[1];

    // Allocate result
    size_t result_dims[] = {M, N};
    GpuTensor* C = gpu_tensor_create(result_dims, 2);

    // Ensure inputs are on GPU
    gpu_tensor_ensure_device(A);
    gpu_tensor_ensure_device(B);

    // cuBLAS GEMM (note: column-major, so we compute B^T * A^T = (A*B)^T)
    const double alpha = 1.0;
    const double beta = 0.0;

    CUBLAS_CHECK(cublasDgemm(
        g_cuda_ctx->cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B->data, N,
        A->data, K,
        &beta,
        C->data, N
    ));

    C->gpu_valid = true;
    C->cpu_valid = false;

    return C;
}

// Element-wise operations via custom kernels
__global__ void tensor_add_kernel(double* a, double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

GpuTensor* gpu_tensor_add(GpuTensor* A, GpuTensor* B) {
    assert(A->total_elements == B->total_elements);

    GpuTensor* C = gpu_tensor_create(A->dims, A->num_dims);

    gpu_tensor_ensure_device(A);
    gpu_tensor_ensure_device(B);

    size_t block_size = 256;
    size_t num_blocks = (A->total_elements + block_size - 1) / block_size;

    tensor_add_kernel<<<num_blocks, block_size, 0, g_cuda_ctx->stream>>>(
        A->data, B->data, C->data, A->total_elements
    );

    C->gpu_valid = true;
    C->cpu_valid = false;

    return C;
}

} // namespace cuda
} // namespace eshkol
```

#### 2.2.4 Eshkol Language Integration

**File:** `lib/backend/llvm_codegen.cpp`

Add new operations:

```cpp
// New operation types in eshkol.h
ESHKOL_GPU_TENSOR_OP,
ESHKOL_GPU_MATMUL_OP,
ESHKOL_GPU_TRANSFER_OP,
ESHKOL_WITH_DEVICE_OP,

// Codegen for GPU operations
llvm::Value* LLVMCodegen::codegenWithDevice(const eshkol_operations_t* op) {
    // Get device specification
    const char* device = op->with_device_op.device;  // "cuda:0", "cpu"

    // Generate device initialization
    if (strncmp(device, "cuda:", 5) == 0) {
        int device_id = atoi(device + 5);
        llvm::Function* cuda_init = getCudaInitFunction();
        ctx_.builder().CreateCall(cuda_init, {
            llvm::ConstantInt::get(ctx_.int32Type(), device_id)
        });
    }

    // Generate body with GPU context active
    llvm::Value* result = nullptr;
    for (size_t i = 0; i < op->with_device_op.num_expressions; i++) {
        result = codegenAST(&op->with_device_op.expressions[i]);
    }

    // Synchronize and cleanup
    llvm::Function* cuda_sync = getCudaSyncFunction();
    ctx_.builder().CreateCall(cuda_sync, {});

    return result;
}
```

#### 2.2.5 Eshkol Syntax

```scheme
;; GPU tensor creation
(define A (gpu-tensor 1024 1024))     ; Allocate on GPU
(define B (gpu-zeros 1024 1024))      ; Zeros on GPU
(define C (gpu-ones 1024 1024))       ; Ones on GPU

;; Transfer operations
(gpu-to-device! tensor)               ; CPU → GPU
(gpu-to-host! tensor)                 ; GPU → CPU

;; GPU computation block
(with-device 'cuda:0
  (let ((A (gpu-tensor 1024 1024))
        (B (gpu-tensor 1024 1024)))
    (gpu-matmul A B)))                ; Returns GPU tensor

;; Automatic transfer on access
(tensor-get (gpu-tensor ...) 0 0)     ; Auto-transfers to host
```

### 2.3 cuDNN Integration for Neural Network Operations

**New File:** `lib/backend/cuda_nn_ops.cpp`

```cpp
namespace eshkol {
namespace cuda {

// Softmax via cuDNN
GpuTensor* gpu_softmax(GpuTensor* input, int axis) {
    cudnnTensorDescriptor_t input_desc, output_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    // Set tensor descriptors based on dimensions
    // ... (dimension-specific setup)

    GpuTensor* output = gpu_tensor_create(input->dims, input->num_dims);

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(
        g_cuda_ctx->cudnn_handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        input_desc, input->data,
        &beta,
        output_desc, output->data
    ));

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);

    return output;
}

// Batch normalization
GpuTensor* gpu_batch_norm(GpuTensor* input, GpuTensor* gamma,
                          GpuTensor* beta, double epsilon) {
    // cuDNN batch normalization implementation
    // ...
}

// Convolution 2D
GpuTensor* gpu_conv2d(GpuTensor* input, GpuTensor* kernel,
                      int stride_h, int stride_w,
                      int pad_h, int pad_w) {
    // cuDNN convolution implementation
    // ...
}

} // namespace cuda
} // namespace eshkol
```

### 2.4 Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lib/backend/cuda_runtime.cpp` | CREATE | CUDA initialization & memory |
| `lib/backend/cuda_tensor_ops.cpp` | CREATE | GPU tensor operations |
| `lib/backend/cuda_tensor_ops.cu` | CREATE | CUDA kernels |
| `lib/backend/cuda_nn_ops.cpp` | CREATE | cuDNN operations |
| `inc/eshkol/cuda/gpu_tensor.h` | CREATE | GPU tensor type |
| `inc/eshkol/cuda/cuda_runtime.h` | CREATE | CUDA API |
| `lib/backend/llvm_codegen.cpp` | MODIFY | GPU operation codegen |
| `lib/frontend/parser.cpp` | MODIFY | GPU syntax parsing |
| `inc/eshkol/eshkol.h` | MODIFY | GPU operation types |
| `CMakeLists.txt` | MODIFY | CUDA compilation |

### 2.5 CMake CUDA Configuration

**File:** `CMakeLists.txt`

```cmake
# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add CUDA source files
set(CUDA_SOURCES
    lib/backend/cuda_tensor_ops.cu
)

# Create CUDA library
add_library(eshkol_cuda STATIC ${CUDA_SOURCES})
target_link_libraries(eshkol_cuda
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
)

# Link to main library
target_link_libraries(eshkol_lib eshkol_cuda)
```

### 2.6 Effort Estimate

| Task | Time |
|------|------|
| CUDA runtime integration | 3 days |
| GPU tensor type | 2 days |
| cuBLAS matmul | 2 days |
| Element-wise kernels | 3 days |
| cuDNN integration | 5 days |
| Memory management | 3 days |
| Language integration | 3 days |
| Testing | 4 days |
| **Total** | **25 days** |

---

## 3. XLA Integration

### 3.1 Current State

**Status:** ❌ NOT IMPLEMENTED

No XLA support. Cannot target TPUs or leverage XLA optimizations.

### 3.2 Required Additions

#### 3.2.1 XLA Client Integration

**New File:** `lib/backend/xla_backend.cpp`

```cpp
#include "xla/client/xla_builder.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"

namespace eshkol {
namespace xla {

class XlaCompiler {
public:
    XlaCompiler() {
        // Get local client (CPU or GPU)
        client_ = ::xla::ClientLibrary::GetOrCreateLocalClient().value();
    }

    // Compile Eshkol AST to XLA computation
    std::unique_ptr<::xla::LocalExecutable> compile(const eshkol_ast_t* ast) {
        ::xla::XlaBuilder builder("eshkol_computation");

        // Lower AST to XLA operations
        ::xla::XlaOp result = lowerAST(ast, &builder);

        // Build computation
        auto computation = builder.Build().value();

        // Compile for local execution
        ::xla::ExecutableBuildOptions options;
        return client_->Compile(computation, {}, options).value()[0];
    }

private:
    ::xla::LocalClient* client_;

    ::xla::XlaOp lowerAST(const eshkol_ast_t* ast, ::xla::XlaBuilder* builder) {
        switch (ast->type) {
            case ESHKOL_TENSOR:
                return lowerTensor(ast, builder);
            case ESHKOL_OP:
                return lowerOperation(&ast->operation, builder);
            // ... other cases
        }
    }

    ::xla::XlaOp lowerOperation(const eshkol_operations_t* op,
                                 ::xla::XlaBuilder* builder) {
        switch (op->op) {
            case ESHKOL_ADD_OP:
                return ::xla::Add(
                    lowerAST(op->binary.left, builder),
                    lowerAST(op->binary.right, builder)
                );
            case ESHKOL_MATMUL_OP:
                return ::xla::Dot(
                    lowerAST(op->binary.left, builder),
                    lowerAST(op->binary.right, builder)
                );
            case ESHKOL_GRADIENT_OP:
                return lowerGradient(op, builder);
            // ... other operations
        }
    }

    ::xla::XlaOp lowerGradient(const eshkol_operations_t* op,
                                ::xla::XlaBuilder* builder) {
        // XLA has built-in autodiff support
        auto computation = lowerAST(op->gradient_op.function, builder);
        auto input = lowerAST(op->gradient_op.point, builder);

        // Use XLA's gradient computation
        return ::xla::Grad(computation, {input})[0];
    }
};

} // namespace xla
} // namespace eshkol
```

#### 3.2.2 XLA JIT Mode

**File:** `lib/backend/xla_backend.cpp`

```cpp
class XlaJitContext {
public:
    // JIT compile and execute
    ::xla::Literal execute(const eshkol_ast_t* ast,
                           const std::vector<::xla::Literal>& inputs) {
        // Check cache
        auto key = computeASTHash(ast);
        auto it = cache_.find(key);

        if (it == cache_.end()) {
            // Compile
            auto executable = compiler_.compile(ast);
            cache_[key] = std::move(executable);
        }

        // Execute
        auto& executable = cache_[key];
        auto result = executable->Run(inputs, {}).value();
        return result.Decompose().value()[0];
    }

private:
    XlaCompiler compiler_;
    std::unordered_map<size_t, std::unique_ptr<::xla::LocalExecutable>> cache_;
};
```

#### 3.2.3 Eshkol Integration

```scheme
;; XLA JIT compilation
(with-xla-jit
  (define (model x)
    (let* ((h1 (relu (matmul x W1)))
           (h2 (relu (matmul h1 W2)))
           (out (softmax (matmul h2 W3))))
      out))

  ;; XLA compiles and optimizes entire computation graph
  (gradient model params))

;; TPU execution
(with-device 'tpu:0
  (train model dataset))
```

### 3.3 Files to Create

| File | Purpose |
|------|---------|
| `lib/backend/xla_backend.cpp` | XLA compilation |
| `lib/backend/xla_jit.cpp` | XLA JIT context |
| `inc/eshkol/xla/xla_backend.h` | XLA API |

### 3.4 Effort Estimate

| Task | Time |
|------|------|
| XLA client setup | 2 days |
| AST → HLO lowering | 5 days |
| Autodiff integration | 3 days |
| TPU support | 3 days |
| Testing | 2 days |
| **Total** | **15 days** |

---

## 4. Parallel Execution

### 4.1 Current State

**Status:** ❌ NOT IMPLEMENTED (only REPL mutex)

No parallelism beyond thread-safe REPL symbol access.

### 4.2 Required Additions

#### 4.2.1 Thread Pool

**New File:** `lib/core/thread_pool.cpp`

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>

namespace eshkol {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }

        for (size_t i = 0; i < num_threads; i++) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();

        return future;
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// Global thread pool
static ThreadPool* g_thread_pool = nullptr;

void init_thread_pool(size_t num_threads) {
    if (!g_thread_pool) {
        g_thread_pool = new ThreadPool(num_threads);
    }
}

ThreadPool* get_thread_pool() {
    if (!g_thread_pool) {
        init_thread_pool(0);
    }
    return g_thread_pool;
}

} // namespace eshkol
```

#### 4.2.2 Parallel Map Implementation

**New File:** `lib/core/parallel_ops.cpp`

```cpp
namespace eshkol {

// Parallel map over list
eshkol_tagged_value_t parallel_map(
    eshkol_closure_t* func,
    eshkol_tagged_value_t* list,
    size_t chunk_size = 0
) {
    // Convert list to vector for random access
    std::vector<eshkol_tagged_value_t> elements;
    eshkol_tagged_value_t current = *list;
    while (current.type == ESHKOL_VALUE_CONS_PTR) {
        auto* cons = (arena_tagged_cons_cell_t*)current.data.ptr_val;
        elements.push_back(cons->car);
        current = cons->cdr;
    }

    size_t n = elements.size();
    if (chunk_size == 0) {
        chunk_size = std::max(1UL, n / (4 * std::thread::hardware_concurrency()));
    }

    std::vector<eshkol_tagged_value_t> results(n);
    std::vector<std::future<void>> futures;

    auto* pool = get_thread_pool();

    for (size_t i = 0; i < n; i += chunk_size) {
        size_t end = std::min(i + chunk_size, n);
        futures.push_back(pool->submit([&, i, end] {
            for (size_t j = i; j < end; j++) {
                results[j] = call_closure(func, &elements[j], 1);
            }
        }));
    }

    // Wait for all tasks
    for (auto& f : futures) {
        f.get();
    }

    // Convert back to list
    return vector_to_list(results);
}

// Parallel reduce
eshkol_tagged_value_t parallel_reduce(
    eshkol_closure_t* func,
    eshkol_tagged_value_t init,
    eshkol_tagged_value_t* list
) {
    // Convert to vector
    std::vector<eshkol_tagged_value_t> elements = list_to_vector(list);
    size_t n = elements.size();

    if (n == 0) return init;

    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<eshkol_tagged_value_t> partial_results(num_threads);
    std::vector<std::future<void>> futures;

    auto* pool = get_thread_pool();

    // Parallel partial reductions
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);

        if (start >= n) break;

        futures.push_back(pool->submit([&, t, start, end] {
            eshkol_tagged_value_t acc = (start == 0) ? init : elements[start];
            size_t i = (start == 0) ? 0 : start + 1;

            for (; i < end; i++) {
                eshkol_tagged_value_t args[2] = {acc, elements[i]};
                acc = call_closure(func, args, 2);
            }
            partial_results[t] = acc;
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    // Sequential final reduction
    eshkol_tagged_value_t result = partial_results[0];
    for (size_t t = 1; t < futures.size(); t++) {
        eshkol_tagged_value_t args[2] = {result, partial_results[t]};
        result = call_closure(func, args, 2);
    }

    return result;
}

} // namespace eshkol
```

#### 4.2.3 Spawn/Await Primitives

**File:** `lib/core/parallel_ops.cpp`

```cpp
namespace eshkol {

struct TaskHandle {
    std::future<eshkol_tagged_value_t> future;
    uint64_t id;
};

static std::atomic<uint64_t> g_task_id{0};
static std::unordered_map<uint64_t, TaskHandle> g_tasks;
static std::mutex g_tasks_mutex;

// Spawn async task
uint64_t spawn_task(eshkol_closure_t* func) {
    uint64_t id = g_task_id++;

    auto future = get_thread_pool()->submit([func] {
        return call_closure(func, nullptr, 0);
    });

    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        g_tasks[id] = {std::move(future), id};
    }

    return id;
}

// Await task completion
eshkol_tagged_value_t await_task(uint64_t task_id) {
    std::future<eshkol_tagged_value_t> future;
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        auto it = g_tasks.find(task_id);
        if (it == g_tasks.end()) {
            // Task not found - return error
            return make_null();
        }
        future = std::move(it->second.future);
        g_tasks.erase(it);
    }

    return future.get();
}

// Check if task is complete
bool task_ready(uint64_t task_id) {
    std::lock_guard<std::mutex> lock(g_tasks_mutex);
    auto it = g_tasks.find(task_id);
    if (it == g_tasks.end()) return true;

    return it->second.future.wait_for(std::chrono::seconds(0))
           == std::future_status::ready;
}

} // namespace eshkol
```

#### 4.2.4 Eshkol Syntax

```scheme
;; Parallel map
(parallel-map process-item items)

;; Parallel reduce
(parallel-reduce + 0 numbers)

;; Spawn async task
(define task (spawn (lambda () (expensive-computation))))

;; Await result
(define result (await task))

;; Check if ready
(if (task-ready? task)
    (await task)
    (display "Still computing..."))

;; Parallel let - evaluate bindings in parallel
(parallel-let ((a (compute-a))
               (b (compute-b))
               (c (compute-c)))
  (combine a b c))
```

### 4.3 Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lib/core/thread_pool.cpp` | CREATE | Thread pool |
| `lib/core/parallel_ops.cpp` | CREATE | Parallel operations |
| `inc/eshkol/core/thread_pool.h` | CREATE | Thread pool API |
| `inc/eshkol/core/parallel_ops.h` | CREATE | Parallel ops API |
| `lib/backend/llvm_codegen.cpp` | MODIFY | Parallel op codegen |
| `lib/frontend/parser.cpp` | MODIFY | Parallel syntax |

### 4.4 Effort Estimate

| Task | Time |
|------|------|
| Thread pool | 2 days |
| Parallel map/reduce | 3 days |
| Spawn/await | 2 days |
| Language integration | 2 days |
| Testing | 2 days |
| **Total** | **11 days** |

---

## 5. Macro System

### 5.1 Current State

**Status:** ❌ NOT IMPLEMENTED

Only `quote` is supported. No quasiquote, no macros.

### 5.2 Required Additions

#### 5.2.1 Quasiquotation

**File:** `lib/frontend/parser.cpp`

```cpp
// Add new token types
TOKEN_QUASIQUOTE,      // `
TOKEN_UNQUOTE,         // ,
TOKEN_UNQUOTE_SPLICING // ,@

// Tokenizer additions
case '`':
    return Token{TOKEN_QUASIQUOTE, "`"};
case ',':
    if (peek() == '@') {
        advance();
        return Token{TOKEN_UNQUOTE_SPLICING, ",@"};
    }
    return Token{TOKEN_UNQUOTE, ","};

// Parser for quasiquote
eshkol_ast_t* Parser::parse_quasiquote(int depth) {
    if (current_token_.type == TOKEN_UNQUOTE && depth == 1) {
        advance();
        return parse_expression();  // Evaluate this part
    }

    if (current_token_.type == TOKEN_UNQUOTE_SPLICING && depth == 1) {
        advance();
        auto* expr = parse_expression();
        // Mark for splicing
        expr->flags |= AST_FLAG_SPLICE;
        return expr;
    }

    if (current_token_.type == TOKEN_QUASIQUOTE) {
        advance();
        return parse_quasiquote(depth + 1);
    }

    if (current_token_.type == TOKEN_LPAREN) {
        advance();
        std::vector<eshkol_ast_t*> elements;
        while (current_token_.type != TOKEN_RPAREN) {
            elements.push_back(parse_quasiquote(depth));
        }
        advance();  // consume RPAREN
        return make_quoted_list(elements);
    }

    // Literal - quote it
    return parse_quoted_datum();
}
```

#### 5.2.2 Syntax-Rules Macros

**New File:** `lib/macro/syntax_rules.cpp`

```cpp
namespace eshkol {
namespace macro {

struct Pattern {
    enum Type { LITERAL, VARIABLE, LIST, ELLIPSIS };
    Type type;
    std::string name;
    std::vector<Pattern> children;
    bool has_ellipsis;
};

struct SyntaxRule {
    Pattern pattern;
    eshkol_ast_t* template_ast;
};

struct SyntaxRulesMacro {
    std::string name;
    std::vector<std::string> literals;  // Keywords that must match exactly
    std::vector<SyntaxRule> rules;
};

class MacroExpander {
public:
    // Match pattern against input
    bool match(const Pattern& pattern, const eshkol_ast_t* input,
               std::unordered_map<std::string, std::vector<eshkol_ast_t*>>& bindings) {
        switch (pattern.type) {
            case Pattern::LITERAL:
                return input->type == ESHKOL_VAR &&
                       input->variable.name == pattern.name;

            case Pattern::VARIABLE:
                bindings[pattern.name].push_back(const_cast<eshkol_ast_t*>(input));
                return true;

            case Pattern::LIST:
                if (input->type != ESHKOL_OP ||
                    input->operation.op != ESHKOL_CALL_OP) {
                    return false;
                }
                return matchList(pattern, input, bindings);

            case Pattern::ELLIPSIS:
                return matchEllipsis(pattern, input, bindings);
        }
        return false;
    }

    // Expand template with bindings
    eshkol_ast_t* expand(const eshkol_ast_t* template_ast,
                         const std::unordered_map<std::string,
                             std::vector<eshkol_ast_t*>>& bindings) {
        // Deep copy template, substituting bound variables
        return expandRecursive(template_ast, bindings, 0);
    }

    // Expand macro invocation
    eshkol_ast_t* expandMacro(const SyntaxRulesMacro& macro,
                              const eshkol_ast_t* invocation) {
        for (const auto& rule : macro.rules) {
            std::unordered_map<std::string, std::vector<eshkol_ast_t*>> bindings;
            if (match(rule.pattern, invocation, bindings)) {
                return expand(rule.template_ast, bindings);
            }
        }
        // No rule matched
        return nullptr;
    }

private:
    // ... helper methods
};

// Global macro registry
static std::unordered_map<std::string, SyntaxRulesMacro> g_macros;

void register_macro(const SyntaxRulesMacro& macro) {
    g_macros[macro.name] = macro;
}

bool is_macro(const std::string& name) {
    return g_macros.find(name) != g_macros.end();
}

eshkol_ast_t* expand_if_macro(const std::string& name,
                               const eshkol_ast_t* invocation) {
    auto it = g_macros.find(name);
    if (it == g_macros.end()) return nullptr;

    static MacroExpander expander;
    return expander.expandMacro(it->second, invocation);
}

} // namespace macro
} // namespace eshkol
```

#### 5.2.3 Eshkol Syntax

```scheme
;; Define a macro with syntax-rules
(define-syntax when
  (syntax-rules ()
    ((when test expr ...)
     (if test (begin expr ...) #f))))

(define-syntax unless
  (syntax-rules ()
    ((unless test expr ...)
     (if test #f (begin expr ...)))))

;; Macro with literals
(define-syntax cond
  (syntax-rules (else =>)
    ((cond (else result ...))
     (begin result ...))
    ((cond (test => func) clause ...)
     (let ((temp test))
       (if temp (func temp) (cond clause ...))))
    ((cond (test result ...) clause ...)
     (if test (begin result ...) (cond clause ...)))))

;; Usage
(when (> x 0)
  (display "positive")
  (newline))

;; Procedural macro (defmacro)
(defmacro define-layer (name inputs outputs)
  `(begin
     (define ,(symbol-append name '-weights)
       (random-tensor ,inputs ,outputs))
     (define ,(symbol-append name '-bias)
       (zeros ,outputs))
     (define (,name x)
       (+ (matmul x ,(symbol-append name '-weights))
          ,(symbol-append name '-bias)))))

;; Usage
(define-layer hidden 784 256)
(hidden input-tensor)
```

### 5.3 Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `lib/macro/syntax_rules.cpp` | CREATE | Syntax-rules implementation |
| `lib/macro/defmacro.cpp` | CREATE | Procedural macros |
| `inc/eshkol/macro/macro.h` | CREATE | Macro API |
| `lib/frontend/parser.cpp` | MODIFY | Quasiquote, macro expansion |
| `inc/eshkol/eshkol.h` | MODIFY | AST flags for splice |

### 5.4 Effort Estimate

| Task | Time |
|------|------|
| Quasiquotation | 3 days |
| Pattern matching | 3 days |
| Template expansion | 3 days |
| Syntax-rules | 3 days |
| Defmacro | 2 days |
| Testing | 2 days |
| **Total** | **16 days** |

---

## 6. Neural Network Primitives

### 6.1 Current State

**Status:** Partial - basic tensor ops exist

Has: matmul, tensor-add/sub/mul/div, reshape, transpose
Missing: softmax, layer-norm, relu, gelu, attention, conv2d

### 6.2 Required Additions

#### 6.2.1 Activation Functions

**File:** `lib/nn/activations.esk`

```scheme
;;; Neural Network Activation Functions
;;; All are differentiable via autodiff

;; ReLU: max(0, x)
(define (relu x)
  (if (tensor? x)
      (tensor-apply (lambda (v) (max 0.0 v)) x)
      (max 0.0 x)))

;; Leaky ReLU
(define (leaky-relu x (alpha 0.01))
  (if (tensor? x)
      (tensor-apply (lambda (v) (if (> v 0) v (* alpha v))) x)
      (if (> x 0) x (* alpha x))))

;; GELU: x * Φ(x) where Φ is standard normal CDF
(define (gelu x)
  (let ((sqrt-2-over-pi 0.7978845608))
    (tensor-apply
      (lambda (v)
        (* 0.5 v (+ 1.0 (tanh (* sqrt-2-over-pi
                                  (+ v (* 0.044715 (* v v v))))))))
      x)))

;; Sigmoid: 1 / (1 + exp(-x))
(define (sigmoid x)
  (if (tensor? x)
      (tensor-apply (lambda (v) (/ 1.0 (+ 1.0 (exp (- 0.0 v))))) x)
      (/ 1.0 (+ 1.0 (exp (- 0.0 x))))))

;; Softmax: exp(x) / sum(exp(x))
;; Numerically stable version
(define (softmax x (axis -1))
  (let* ((max-val (tensor-max x axis))
         (shifted (tensor-sub x (tensor-broadcast max-val (tensor-shape x))))
         (exp-vals (tensor-apply exp shifted))
         (sum-exp (tensor-sum exp-vals axis)))
    (tensor-div exp-vals (tensor-broadcast sum-exp (tensor-shape x)))))

;; Swish: x * sigmoid(x)
(define (swish x)
  (tensor-mul x (sigmoid x)))

;; Tanh (already exists as builtin, but for completeness)
(define (tanh-activation x)
  (if (tensor? x)
      (tensor-apply tanh x)
      (tanh x)))
```

#### 6.2.2 Normalization Layers

**File:** `lib/nn/normalization.esk`

```scheme
;;; Normalization Layers

;; Layer Normalization
;; Normalizes across the last dimension
(define (layer-norm x gamma beta (epsilon 1e-5))
  (let* ((shape (tensor-shape x))
         (last-dim (- (length shape) 1))
         (mean (tensor-mean x last-dim))
         (variance (tensor-variance x last-dim))
         (normalized (tensor-div
                       (tensor-sub x (tensor-broadcast mean shape))
                       (tensor-apply sqrt
                         (tensor-add variance
                           (tensor-fill epsilon shape))))))
    (tensor-add (tensor-mul normalized gamma) beta)))

;; Batch Normalization
;; Normalizes across batch dimension (axis 0)
(define (batch-norm x gamma beta running-mean running-var
                    (epsilon 1e-5) (momentum 0.1) (training #t))
  (if training
      (let* ((batch-mean (tensor-mean x 0))
             (batch-var (tensor-variance x 0))
             ;; Update running statistics
             (_ (tensor-set! running-mean
                  (tensor-add
                    (tensor-mul running-mean (- 1.0 momentum))
                    (tensor-mul batch-mean momentum))))
             (_ (tensor-set! running-var
                  (tensor-add
                    (tensor-mul running-var (- 1.0 momentum))
                    (tensor-mul batch-var momentum))))
             (normalized (tensor-div
                           (tensor-sub x (tensor-broadcast batch-mean (tensor-shape x)))
                           (tensor-apply sqrt
                             (tensor-add batch-var (tensor-fill epsilon (tensor-shape batch-var)))))))
        (tensor-add (tensor-mul normalized gamma) beta))
      ;; Inference mode - use running statistics
      (let ((normalized (tensor-div
                          (tensor-sub x (tensor-broadcast running-mean (tensor-shape x)))
                          (tensor-apply sqrt
                            (tensor-add running-var (tensor-fill epsilon (tensor-shape running-var)))))))
        (tensor-add (tensor-mul normalized gamma) beta))))

;; RMS Normalization (used in LLaMA)
(define (rms-norm x gamma (epsilon 1e-5))
  (let* ((rms (tensor-apply sqrt
                (tensor-add
                  (tensor-mean (tensor-mul x x) -1)
                  (tensor-fill epsilon '(1)))))
         (normalized (tensor-div x (tensor-broadcast rms (tensor-shape x)))))
    (tensor-mul normalized gamma)))
```

#### 6.2.3 Attention Mechanisms

**File:** `lib/nn/attention.esk`

```scheme
;;; Attention Mechanisms

;; Scaled Dot-Product Attention
;; Q: [batch, seq_q, d_k]
;; K: [batch, seq_k, d_k]
;; V: [batch, seq_k, d_v]
;; Returns: [batch, seq_q, d_v]
(define (scaled-dot-product-attention Q K V (mask #f))
  (let* ((d-k (tensor-get (tensor-shape Q) 2))
         (scale (/ 1.0 (sqrt d-k)))
         ;; Q @ K^T
         (scores (tensor-mul (matmul Q (transpose K 1 2)) scale)))
    ;; Apply mask if provided
    (when mask
      (set! scores (tensor-add scores
                     (tensor-mul (tensor-sub 1.0 mask) -1e9))))
    ;; Softmax over last dimension
    (let ((attn-weights (softmax scores -1)))
      ;; Attention @ V
      (matmul attn-weights V))))

;; Multi-Head Attention
(define (multi-head-attention x num-heads d-model W-q W-k W-v W-o (mask #f))
  (let* ((batch-size (tensor-get (tensor-shape x) 0))
         (seq-len (tensor-get (tensor-shape x) 1))
         (d-k (/ d-model num-heads))

         ;; Linear projections
         (Q (matmul x W-q))  ; [batch, seq, d_model]
         (K (matmul x W-k))
         (V (matmul x W-v))

         ;; Reshape for multi-head: [batch, seq, num_heads, d_k]
         (Q (reshape Q (list batch-size seq-len num-heads d-k)))
         (K (reshape K (list batch-size seq-len num-heads d-k)))
         (V (reshape V (list batch-size seq-len num-heads d-k)))

         ;; Transpose to [batch, num_heads, seq, d_k]
         (Q (transpose Q 1 2))
         (K (transpose K 1 2))
         (V (transpose V 1 2))

         ;; Apply attention per head
         (attn-output (scaled-dot-product-attention Q K V mask))

         ;; Transpose back: [batch, seq, num_heads, d_k]
         (attn-output (transpose attn-output 1 2))

         ;; Concatenate heads: [batch, seq, d_model]
         (concat-output (reshape attn-output (list batch-size seq-len d-model))))

    ;; Final linear projection
    (matmul concat-output W-o)))

;; Self-Attention (Q=K=V=x)
(define (self-attention x num-heads d-model W-q W-k W-v W-o (mask #f))
  (multi-head-attention x num-heads d-model W-q W-k W-v W-o mask))

;; Causal mask for autoregressive models
(define (causal-mask seq-len)
  (let ((mask (ones seq-len seq-len)))
    ;; Lower triangular
    (tensor-apply-indexed
      (lambda (i j v) (if (<= i j) 1.0 0.0))
      mask)))
```

#### 6.2.4 Transformer Components

**File:** `lib/nn/transformer.esk`

```scheme
;;; Transformer Building Blocks

;; Feed-Forward Network
(define (feed-forward x d-model d-ff W1 b1 W2 b2)
  (let* ((hidden (gelu (tensor-add (matmul x W1) b1)))
         (output (tensor-add (matmul hidden W2) b2)))
    output))

;; Transformer Encoder Layer
(define (transformer-encoder-layer x
                                    num-heads d-model d-ff
                                    W-q W-k W-v W-o
                                    W1 b1 W2 b2
                                    gamma1 beta1 gamma2 beta2)
  ;; Self-attention with residual and layer norm
  (let* ((attn-out (self-attention x num-heads d-model W-q W-k W-v W-o))
         (x (layer-norm (tensor-add x attn-out) gamma1 beta1))
         ;; Feed-forward with residual and layer norm
         (ff-out (feed-forward x d-model d-ff W1 b1 W2 b2))
         (x (layer-norm (tensor-add x ff-out) gamma2 beta2)))
    x))

;; Transformer Decoder Layer (with cross-attention)
(define (transformer-decoder-layer x encoder-output
                                    num-heads d-model d-ff
                                    W-q-self W-k-self W-v-self W-o-self
                                    W-q-cross W-k-cross W-v-cross W-o-cross
                                    W1 b1 W2 b2
                                    gamma1 beta1 gamma2 beta2 gamma3 beta3
                                    (causal-mask #f))
  ;; Masked self-attention
  (let* ((self-attn-out (self-attention x num-heads d-model
                          W-q-self W-k-self W-v-self W-o-self causal-mask))
         (x (layer-norm (tensor-add x self-attn-out) gamma1 beta1))
         ;; Cross-attention with encoder output
         (cross-attn-out (multi-head-attention x num-heads d-model
                           W-q-cross W-k-cross W-v-cross W-o-cross))
         (x (layer-norm (tensor-add x cross-attn-out) gamma2 beta2))
         ;; Feed-forward
         (ff-out (feed-forward x d-model d-ff W1 b1 W2 b2))
         (x (layer-norm (tensor-add x ff-out) gamma3 beta3)))
    x))

;; Positional Encoding (sinusoidal)
(define (positional-encoding seq-len d-model)
  (let ((pe (zeros seq-len d-model)))
    (do ((pos 0 (+ pos 1)))
        ((= pos seq-len) pe)
      (do ((i 0 (+ i 2)))
          ((>= i d-model))
        (let ((angle (/ pos (expt 10000.0 (/ i d-model)))))
          (tensor-set! pe pos i (sin angle))
          (when (< (+ i 1) d-model)
            (tensor-set! pe pos (+ i 1) (cos angle))))))))
```

### 6.3 Files to Create

| File | Purpose |
|------|---------|
| `lib/nn/activations.esk` | Activation functions |
| `lib/nn/normalization.esk` | Normalization layers |
| `lib/nn/attention.esk` | Attention mechanisms |
| `lib/nn/transformer.esk` | Transformer blocks |
| `lib/nn/conv.esk` | Convolution operations |
| `lib/nn/loss.esk` | Loss functions |
| `lib/nn.esk` | Main NN module re-export |

### 6.4 Effort Estimate

| Task | Time |
|------|------|
| Activation functions | 2 days |
| Normalization layers | 2 days |
| Attention mechanisms | 3 days |
| Transformer blocks | 2 days |
| Convolution (conv2d) | 3 days |
| Loss functions | 1 day |
| Testing | 2 days |
| **Total** | **15 days** |

---

## 7. Distributed Computing

### 7.1 Current State

**Status:** ❌ NOT IMPLEMENTED

No distributed computing support.

### 7.2 Required Additions

#### 7.2.1 MPI Integration

**New File:** `lib/distributed/mpi_backend.cpp`

```cpp
#include <mpi.h>

namespace eshkol {
namespace distributed {

struct MpiContext {
    int rank;
    int world_size;
    bool initialized;
};

static MpiContext g_mpi_ctx = {0, 1, false};

void mpi_init() {
    if (g_mpi_ctx.initialized) return;

    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_ctx.world_size);
    g_mpi_ctx.initialized = true;
}

void mpi_finalize() {
    if (g_mpi_ctx.initialized) {
        MPI_Finalize();
        g_mpi_ctx.initialized = false;
    }
}

int get_rank() { return g_mpi_ctx.rank; }
int get_world_size() { return g_mpi_ctx.world_size; }

// All-reduce for gradient synchronization
void all_reduce_sum(double* data, size_t count) {
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
}

// Broadcast parameters from rank 0
void broadcast(double* data, size_t count, int root = 0) {
    MPI_Bcast(data, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

// Scatter data across ranks
void scatter(double* send_data, double* recv_data,
             size_t count_per_rank, int root = 0) {
    MPI_Scatter(send_data, count_per_rank, MPI_DOUBLE,
                recv_data, count_per_rank, MPI_DOUBLE,
                root, MPI_COMM_WORLD);
}

// Gather data to root
void gather(double* send_data, double* recv_data,
            size_t count_per_rank, int root = 0) {
    MPI_Gather(send_data, count_per_rank, MPI_DOUBLE,
               recv_data, count_per_rank, MPI_DOUBLE,
               root, MPI_COMM_WORLD);
}

} // namespace distributed
} // namespace eshkol
```

#### 7.2.2 Distributed Data Parallel

**New File:** `lib/distributed/data_parallel.cpp`

```cpp
namespace eshkol {
namespace distributed {

class DataParallelTrainer {
public:
    DataParallelTrainer(size_t batch_size)
        : local_batch_size_(batch_size / get_world_size()) {
    }

    // Synchronized gradient step
    void step(eshkol_tensor_t* gradients, size_t num_params) {
        // Average gradients across all ranks
        all_reduce_sum(gradients->elements, num_params);

        // Divide by world size to get average
        double scale = 1.0 / get_world_size();
        for (size_t i = 0; i < num_params; i++) {
            gradients->elements[i] *= scale;
        }
    }

    // Scatter batch to ranks
    eshkol_tensor_t* scatter_batch(eshkol_tensor_t* full_batch) {
        size_t local_size = full_batch->total_elements / get_world_size();
        eshkol_tensor_t* local_batch = allocate_tensor(local_size);

        scatter(full_batch->elements, local_batch->elements,
                local_size, 0);

        return local_batch;
    }

private:
    size_t local_batch_size_;
};

} // namespace distributed
} // namespace eshkol
```

#### 7.2.3 Eshkol Syntax

```scheme
;; Initialize distributed environment
(distributed-init)

;; Get rank and world size
(define rank (distributed-rank))
(define world-size (distributed-world-size))

;; Data parallel training
(with-data-parallel
  (train model dataset))

;; Manual gradient synchronization
(define grads (gradient loss params))
(all-reduce-sum! grads)
(tensor-scale! grads (/ 1.0 world-size))

;; Broadcast parameters
(when (= rank 0)
  (broadcast! params))

;; Scatter data
(define local-batch (scatter batch 0))

;; Barrier synchronization
(distributed-barrier)

;; Finalize
(distributed-finalize)
```

### 7.3 Effort Estimate

| Task | Time |
|------|------|
| MPI integration | 3 days |
| All-reduce/broadcast | 2 days |
| Data parallel trainer | 3 days |
| Language integration | 2 days |
| Testing | 2 days |
| **Total** | **12 days** |

---

## 8. Serialization & Checkpointing

### 8.1 Current State

**Status:** ❌ NOT IMPLEMENTED

No serialization support.

### 8.2 Required Additions

#### 8.2.1 Binary Tensor Format

**New File:** `lib/io/tensor_io.cpp`

```cpp
namespace eshkol {
namespace io {

// Header for binary tensor format
struct TensorFileHeader {
    char magic[4] = {'E', 'S', 'H', 'K'};
    uint32_t version = 1;
    uint32_t num_tensors;
    uint32_t flags;
};

struct TensorHeader {
    uint32_t name_length;
    uint32_t num_dims;
    uint32_t dtype;  // 0 = float64, 1 = float32, 2 = int64
    uint64_t total_elements;
    // Followed by: name (name_length bytes), dims (num_dims * 8 bytes), data
};

// Save tensor to file
void save_tensor(const char* path, const eshkol_tensor_t* tensor,
                 const char* name = nullptr) {
    FILE* f = fopen(path, "wb");
    if (!f) return;

    TensorFileHeader file_header;
    file_header.num_tensors = 1;
    fwrite(&file_header, sizeof(file_header), 1, f);

    TensorHeader header;
    header.name_length = name ? strlen(name) : 0;
    header.num_dims = tensor->num_dims;
    header.dtype = 0;  // float64
    header.total_elements = tensor->total_elements;
    fwrite(&header, sizeof(header), 1, f);

    if (name) fwrite(name, 1, header.name_length, f);
    fwrite(tensor->dims, sizeof(size_t), tensor->num_dims, f);
    fwrite(tensor->elements, sizeof(double), tensor->total_elements, f);

    fclose(f);
}

// Load tensor from file
eshkol_tensor_t* load_tensor(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;

    TensorFileHeader file_header;
    fread(&file_header, sizeof(file_header), 1, f);

    // Validate magic
    if (memcmp(file_header.magic, "ESHK", 4) != 0) {
        fclose(f);
        return nullptr;
    }

    TensorHeader header;
    fread(&header, sizeof(header), 1, f);

    // Skip name
    fseek(f, header.name_length, SEEK_CUR);

    // Read dimensions
    size_t* dims = new size_t[header.num_dims];
    fread(dims, sizeof(size_t), header.num_dims, f);

    // Allocate and read data
    eshkol_tensor_t* tensor = arena_allocate_tensor(
        get_global_arena(), dims, header.num_dims);
    fread(tensor->elements, sizeof(double), header.total_elements, f);

    delete[] dims;
    fclose(f);
    return tensor;
}

} // namespace io
} // namespace eshkol
```

#### 8.2.2 Model Checkpointing

**New File:** `lib/io/checkpoint.cpp`

```cpp
namespace eshkol {
namespace io {

// Save model state (all named tensors)
void save_checkpoint(const char* path,
                     const std::unordered_map<std::string,
                         eshkol_tensor_t*>& tensors) {
    FILE* f = fopen(path, "wb");
    if (!f) return;

    TensorFileHeader file_header;
    file_header.num_tensors = tensors.size();
    fwrite(&file_header, sizeof(file_header), 1, f);

    for (const auto& [name, tensor] : tensors) {
        TensorHeader header;
        header.name_length = name.length();
        header.num_dims = tensor->num_dims;
        header.dtype = 0;
        header.total_elements = tensor->total_elements;
        fwrite(&header, sizeof(header), 1, f);

        fwrite(name.c_str(), 1, name.length(), f);
        fwrite(tensor->dims, sizeof(size_t), tensor->num_dims, f);
        fwrite(tensor->elements, sizeof(double), tensor->total_elements, f);
    }

    fclose(f);
}

// Load model state
std::unordered_map<std::string, eshkol_tensor_t*>
load_checkpoint(const char* path) {
    std::unordered_map<std::string, eshkol_tensor_t*> result;

    FILE* f = fopen(path, "rb");
    if (!f) return result;

    TensorFileHeader file_header;
    fread(&file_header, sizeof(file_header), 1, f);

    for (uint32_t i = 0; i < file_header.num_tensors; i++) {
        TensorHeader header;
        fread(&header, sizeof(header), 1, f);

        std::string name(header.name_length, '\0');
        fread(&name[0], 1, header.name_length, f);

        size_t* dims = new size_t[header.num_dims];
        fread(dims, sizeof(size_t), header.num_dims, f);

        eshkol_tensor_t* tensor = arena_allocate_tensor(
            get_global_arena(), dims, header.num_dims);
        fread(tensor->elements, sizeof(double), header.total_elements, f);

        result[name] = tensor;
        delete[] dims;
    }

    fclose(f);
    return result;
}

} // namespace io
} // namespace eshkol
```

#### 8.2.3 Eshkol Syntax

```scheme
;; Save single tensor
(tensor-save weights "weights.bin")

;; Load single tensor
(define weights (tensor-load "weights.bin"))

;; Save model checkpoint
(save-checkpoint "model.ckpt"
  (list (cons "layer1.weight" layer1-weight)
        (cons "layer1.bias" layer1-bias)
        (cons "layer2.weight" layer2-weight)
        (cons "layer2.bias" layer2-bias)))

;; Load model checkpoint
(define checkpoint (load-checkpoint "model.ckpt"))
(define layer1-weight (assoc-ref checkpoint "layer1.weight"))
(define layer1-bias (assoc-ref checkpoint "layer1.bias"))

;; Auto-save during training
(define (train-with-checkpoint model data epochs save-every)
  (do ((epoch 0 (+ epoch 1)))
      ((= epoch epochs))
    (train-epoch model data)
    (when (= (modulo epoch save-every) 0)
      (save-checkpoint (format "model_epoch_~a.ckpt" epoch)
                       (model-parameters model)))))
```

### 8.3 Effort Estimate

| Task | Time |
|------|------|
| Binary format | 2 days |
| Tensor I/O | 2 days |
| Checkpointing | 2 days |
| Language integration | 1 day |
| Testing | 1 day |
| **Total** | **8 days** |

---

## 9. Profiling & Debugging

### 9.1 Required Additions

#### 9.1.1 Execution Profiler

**New File:** `lib/debug/profiler.cpp`

```cpp
namespace eshkol {
namespace debug {

struct ProfileEntry {
    std::string name;
    uint64_t calls;
    double total_time_ms;
    double min_time_ms;
    double max_time_ms;
};

class Profiler {
public:
    void start(const std::string& name) {
        starts_[name] = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto start = starts_[name];
        double elapsed = std::chrono::duration<double, std::milli>(
            end - start).count();

        auto& entry = entries_[name];
        entry.name = name;
        entry.calls++;
        entry.total_time_ms += elapsed;
        entry.min_time_ms = std::min(entry.min_time_ms, elapsed);
        entry.max_time_ms = std::max(entry.max_time_ms, elapsed);
    }

    void report() {
        std::cout << "=== Profile Report ===" << std::endl;
        for (const auto& [name, entry] : entries_) {
            std::cout << name << ": "
                      << entry.calls << " calls, "
                      << entry.total_time_ms << "ms total, "
                      << (entry.total_time_ms / entry.calls) << "ms avg"
                      << std::endl;
        }
    }

private:
    std::unordered_map<std::string,
        std::chrono::high_resolution_clock::time_point> starts_;
    std::unordered_map<std::string, ProfileEntry> entries_;
};

} // namespace debug
} // namespace eshkol
```

#### 9.1.2 Eshkol Syntax

```scheme
;; Profile a computation
(with-profile 'matmul
  (matmul A B))

;; Get profile report
(profile-report)

;; Memory profiling
(memory-stats)  ; Returns current memory usage

;; Tensor shape debugging
(tensor-debug tensor)  ; Prints shape, dtype, sample values
```

### 9.2 Effort Estimate

| Task | Time |
|------|------|
| Profiler | 2 days |
| Memory stats | 1 day |
| Debug utilities | 1 day |
| **Total** | **4 days** |

---

## 10. Extended Tensor Operations

### 10.1 Required Additions

**File:** `lib/backend/tensor_codegen.cpp` (additions)

```scheme
;; Broadcasting
(tensor-broadcast tensor target-shape)

;; Concatenation
(tensor-concat tensors axis)

;; Splitting
(tensor-split tensor num-splits axis)

;; Stacking
(tensor-stack tensors axis)

;; Einsum
(einsum "ij,jk->ik" A B)  ; matmul
(einsum "bi,bj->bij" a b)  ; outer product

;; Advanced indexing
(tensor-slice tensor '(0:10 : 0:5))  ; Slicing syntax
(tensor-gather tensor indices axis)
(tensor-scatter! tensor indices values axis)

;; Comparisons
(tensor-eq tensor1 tensor2)
(tensor-lt tensor1 tensor2)
(tensor-where condition true-tensor false-tensor)

;; Statistical
(tensor-variance tensor axis)
(tensor-std tensor axis)
(tensor-argmax tensor axis)
(tensor-argmin tensor axis)
(tensor-topk tensor k axis)

;; Linear algebra
(tensor-svd tensor)        ; Returns (U, S, V)
(tensor-eig tensor)        ; Returns (eigenvalues, eigenvectors)
(tensor-qr tensor)         ; Returns (Q, R)
(tensor-cholesky tensor)   ; Lower triangular
(tensor-lstsq A b)         ; Least squares solution
```

### 10.2 Effort Estimate

| Task | Time |
|------|------|
| Broadcasting | 2 days |
| Concat/split/stack | 2 days |
| Einsum | 3 days |
| Advanced indexing | 2 days |
| Statistical ops | 2 days |
| Linear algebra | 3 days |
| **Total** | **14 days** |

---

## 11. Optimizer Implementations

### 11.1 Required Additions

**File:** `lib/nn/optimizers.esk`

```scheme
;;; Optimizers

;; SGD with momentum
(define (make-sgd params learning-rate (momentum 0.0) (weight-decay 0.0))
  (let ((velocities (map (lambda (p) (zeros-like p)) params)))
    (lambda (grads)
      (for-each
        (lambda (param grad velocity)
          ;; v = momentum * v - lr * (grad + weight_decay * param)
          (tensor-scale! velocity momentum)
          (tensor-axpy! velocity (- 0 learning-rate)
            (tensor-add grad (tensor-scale param weight-decay)))
          ;; param += v
          (tensor-add! param velocity))
        params grads velocities))))

;; Adam optimizer
(define (make-adam params learning-rate (beta1 0.9) (beta2 0.999) (epsilon 1e-8))
  (let ((m (map (lambda (p) (zeros-like p)) params))  ; First moment
        (v (map (lambda (p) (zeros-like p)) params))  ; Second moment
        (t 0))
    (lambda (grads)
      (set! t (+ t 1))
      (for-each
        (lambda (param grad m-i v-i)
          ;; m = beta1 * m + (1 - beta1) * grad
          (tensor-scale! m-i beta1)
          (tensor-axpy! m-i (- 1 beta1) grad)
          ;; v = beta2 * v + (1 - beta2) * grad^2
          (tensor-scale! v-i beta2)
          (tensor-axpy! v-i (- 1 beta2) (tensor-mul grad grad))
          ;; Bias correction
          (let ((m-hat (tensor-scale m-i (/ 1.0 (- 1 (expt beta1 t)))))
                (v-hat (tensor-scale v-i (/ 1.0 (- 1 (expt beta2 t))))))
            ;; param -= lr * m-hat / (sqrt(v-hat) + epsilon)
            (tensor-axpy! param (- 0 learning-rate)
              (tensor-div m-hat
                (tensor-add (tensor-sqrt v-hat)
                  (tensor-fill epsilon (tensor-shape v-hat)))))))
        params grads m v))))

;; AdamW (Adam with decoupled weight decay)
(define (make-adamw params learning-rate (beta1 0.9) (beta2 0.999)
                    (epsilon 1e-8) (weight-decay 0.01))
  (let ((adam (make-adam params learning-rate beta1 beta2 epsilon)))
    (lambda (grads)
      ;; Apply weight decay directly to params
      (for-each
        (lambda (param)
          (tensor-axpy! param (- 0 (* learning-rate weight-decay)) param))
        params)
      ;; Then apply Adam update
      (adam grads))))

;; Learning rate scheduler
(define (make-cosine-scheduler initial-lr total-steps (min-lr 0.0))
  (lambda (step)
    (+ min-lr
       (* 0.5 (- initial-lr min-lr)
          (+ 1 (cos (* pi (/ step total-steps))))))))
```

### 11.2 Effort Estimate

| Task | Time |
|------|------|
| SGD | 1 day |
| Adam/AdamW | 2 days |
| Schedulers | 1 day |
| Testing | 1 day |
| **Total** | **5 days** |

---

## 12. Memory Optimization

### 12.1 Required Additions

#### 12.1.1 Gradient Checkpointing

**File:** `lib/nn/checkpointing.esk`

```scheme
;; Gradient checkpointing for memory efficiency
;; Recompute activations during backward pass instead of storing

(define (checkpoint-sequential layers)
  (lambda (x)
    (fold
      (lambda (layer input)
        (checkpoint-segment (lambda () (layer input))))
      x
      layers)))

;; Low-level checkpoint primitive (implemented in C++)
;; Saves computation graph node, discards activations,
;; recomputes during backward pass
```

#### 12.1.2 Memory Pool

**File:** `lib/core/memory_pool.cpp`

```cpp
class TensorMemoryPool {
public:
    void* allocate(size_t size) {
        // Check free list for existing block
        auto it = free_blocks_.find(size);
        if (it != free_blocks_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
        // Allocate new block
        return arena_allocate_simd_aligned(arena_, size);
    }

    void deallocate(void* ptr, size_t size) {
        // Return to free list instead of freeing
        free_blocks_[size].push_back(ptr);
    }

private:
    arena_t* arena_;
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
};
```

### 12.2 Effort Estimate

| Task | Time |
|------|------|
| Gradient checkpointing | 3 days |
| Memory pool | 2 days |
| Testing | 1 day |
| **Total** | **6 days** |

---

## 13. Implementation Roadmap

### Phase 1: Performance Foundation (Weeks 1-3)
| Week | Tasks | Effort |
|------|-------|--------|
| 1 | SIMD vectorization | 12 days |
| 2-3 | Parallel execution | 11 days |
| **Subtotal** | | **23 days** |

### Phase 2: Hardware Acceleration (Weeks 4-8)
| Week | Tasks | Effort |
|------|-------|--------|
| 4-6 | GPU/CUDA support | 25 days |
| 7-8 | XLA integration | 15 days |
| **Subtotal** | | **40 days** |

### Phase 3: Neural Networks (Weeks 9-12)
| Week | Tasks | Effort |
|------|-------|--------|
| 9-10 | NN primitives | 15 days |
| 11-12 | Macro system | 16 days |
| **Subtotal** | | **31 days** |

### Phase 4: Production Features (Weeks 13-17)
| Week | Tasks | Effort |
|------|-------|--------|
| 13-14 | Distributed computing | 12 days |
| 15 | Serialization | 8 days |
| 16 | Extended tensor ops | 14 days |
| 17 | Optimizers + memory | 11 days |
| **Subtotal** | | **45 days** |

### Phase 5: Polish (Weeks 18-20)
| Week | Tasks | Effort |
|------|-------|--------|
| 18-19 | Profiling & debugging | 4 days |
| 19-20 | Integration testing | 10 days |
| 20 | Documentation | 5 days |
| **Subtotal** | | **19 days** |

---

## Total Implementation Summary

| Category | Effort (days) |
|----------|---------------|
| SIMD/Vectorization | 12 |
| GPU/CUDA | 25 |
| XLA Integration | 15 |
| Parallel Execution | 11 |
| Macro System | 16 |
| Neural Network Primitives | 15 |
| Distributed Computing | 12 |
| Serialization | 8 |
| Profiling & Debugging | 4 |
| Extended Tensor Ops | 14 |
| Optimizers | 5 |
| Memory Optimization | 6 |
| Integration Testing | 10 |
| Documentation | 5 |
| **GRAND TOTAL** | **158 days (~32 weeks)** |

---

## Conclusion

This document provides a complete implementation guide for transforming Eshkol into the definitive language for intelligent computing. The work is substantial but tractable:

- **Core performance** (SIMD + parallelism): 23 days
- **Hardware acceleration** (CUDA + XLA): 40 days
- **Neural network ecosystem** (NN + macros): 31 days
- **Production features** (distributed + IO): 45 days

With focused engineering effort, Eshkol can achieve feature parity with JAX/PyTorch while maintaining its unique advantages: native compilation, HoTT types, homoiconicity, and zero-GC memory management.
