/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen - Tensor operations code generation
 *
 * This module handles:
 * - Tensor creation and access
 * - Tensor arithmetic (add, sub, mul, div, dot)
 * - Tensor transformations (apply, reduce, transpose)
 * - Shape operations
 */
#ifndef ESHKOL_BACKEND_TENSOR_CODEGEN_H
#define ESHKOL_BACKEND_TENSOR_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <string>
#include <memory>

namespace eshkol {

// Forward declaration for XLA backend
namespace xla {
    class XLACodegen;
}

/**
 * TensorCodegen handles tensor operations.
 *
 * Tensors in Eshkol are n-dimensional arrays with:
 * - Arbitrary dimensions (1D vectors, 2D matrices, etc.)
 * - Element-wise arithmetic with broadcasting
 * - Support for autodiff operations
 */
class TensorCodegen {
public:
    /**
     * Construct TensorCodegen with context and helpers.
     */
    TensorCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem);

    /**
     * Destructor - defined in .cpp where XLACodegen is complete.
     */
    ~TensorCodegen();

    // === Tensor Creation ===

    /**
     * Create a tensor from a literal: #[1 2 3] or #[[1 2] [3 4]]
     * @param ast The tensor AST node
     * @return Tagged tensor pointer
     */
    llvm::Value* createTensor(const eshkol_ast_t* ast);

    /**
     * Create a tensor via operation: (tensor ...)
     * @param op The tensor operation AST node
     * @return Tagged tensor pointer
     */
    llvm::Value* tensorOperation(const eshkol_operations_t* op);

    // === Tensor Access ===

    /**
     * Get element at indices: (tensor-get tensor idx1 idx2 ...)
     * @param op The operation AST node
     * @return Tagged value at the specified indices
     */
    llvm::Value* tensorGet(const eshkol_operations_t* op);

    /**
     * Vector reference (1D): (vref tensor idx)
     * @param op The operation AST node
     * @return Tagged value at index
     */
    llvm::Value* vectorRef(const eshkol_operations_t* op);

    /**
     * Set element at indices: (tensor-set tensor val idx1 idx2 ...)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* tensorSet(const eshkol_operations_t* op);

    // === Tensor Arithmetic ===

    /**
     * Element-wise arithmetic: tensor-add, tensor-sub, tensor-mul, tensor-div
     * @param op The operation AST node
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result tensor
     */
    llvm::Value* tensorArithmetic(const eshkol_operations_t* op, const std::string& operation);

    /**
     * Internal arithmetic implementation for two tagged values.
     * @param left Left operand (tagged)
     * @param right Right operand (tagged)
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result tensor
     */
    llvm::Value* tensorArithmeticInternal(llvm::Value* left, llvm::Value* right, const std::string& operation);

    /**
     * Dot product / matrix multiplication: (tensor-dot A B)
     * @param op The operation AST node
     * @return Result tensor or scalar
     */
    llvm::Value* tensorDot(const eshkol_operations_t* op);

    /**
     * SIMD-accelerated matrix multiplication: (matmul A B)
     * Uses vectorized operations to process 4 columns at a time.
     * @param ptr_a Pointer to tensor A [M x K]
     * @param ptr_b Pointer to tensor B [K x N]
     * @param M Rows of A
     * @param K Columns of A / Rows of B
     * @param N Columns of B
     * @return Result tensor [M x N]
     */
    llvm::Value* matmulSIMD(llvm::Value* ptr_a, llvm::Value* ptr_b,
                            llvm::Value* M, llvm::Value* K, llvm::Value* N);

    // === Tensor Transformations ===

    /**
     * Apply function element-wise: (tensor-apply tensor func)
     * @param op The operation AST node
     * @return Result tensor
     */
    llvm::Value* tensorApply(const eshkol_operations_t* op);

    /**
     * Reduce all elements: (tensor-reduce-all tensor func init)
     * @param op The operation AST node
     * @return Scalar result
     */
    llvm::Value* tensorReduceAll(const eshkol_operations_t* op);

    /**
     * Reduce along dimension: (tensor-reduce tensor func init dim)
     * @param op The operation AST node
     * @return Result tensor
     */
    llvm::Value* tensorReduceWithDim(const eshkol_operations_t* op);

    /**
     * Sum all elements: (tensor-sum tensor)
     * @param op The operation AST node
     * @return Scalar sum
     */
    llvm::Value* tensorSum(const eshkol_operations_t* op);

    /**
     * Mean of all elements: (tensor-mean tensor)
     * @param op The operation AST node
     * @return Scalar mean
     */
    llvm::Value* tensorMean(const eshkol_operations_t* op);

    // === Type Conversion ===

    /**
     * Convert Scheme vector to tensor: (vector->tensor vec)
     * @param op The operation AST node
     * @return 1D tensor with elements copied from vector
     */
    llvm::Value* vectorToTensor(const eshkol_operations_t* op);

    /**
     * Convert tensor to Scheme vector: (tensor->vector tensor)
     * @param op The operation AST node
     * @return Scheme vector with elements from flattened tensor
     */
    llvm::Value* tensorToVector(const eshkol_operations_t* op);

    // === Activation Functions (SIMD-Accelerated) ===

    /**
     * ReLU activation: max(0, x)
     * @param op The operation AST node
     * @return Tensor with ReLU applied element-wise
     */
    llvm::Value* tensorRelu(const eshkol_operations_t* op);

    /**
     * Sigmoid activation: 1 / (1 + exp(-x))
     * @param op The operation AST node
     * @return Tensor with sigmoid applied element-wise
     */
    llvm::Value* tensorSigmoid(const eshkol_operations_t* op);

    /**
     * Softmax activation: exp(x_i) / sum(exp(x))
     * Numerically stable version using max subtraction
     * @param op The operation AST node
     * @return Tensor with softmax applied
     */
    llvm::Value* tensorSoftmax(const eshkol_operations_t* op);

    /**
     * GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     * Uses fast sigmoid approximation: x * σ(1.702 * x)
     * @param op The operation AST node
     * @return Tensor with GELU applied element-wise
     */
    llvm::Value* tensorGelu(const eshkol_operations_t* op);

    /**
     * Leaky ReLU activation: x if x > 0, else alpha * x
     * @param op The operation AST node (tensor, optional alpha=0.01)
     * @return Tensor with leaky ReLU applied element-wise
     */
    llvm::Value* tensorLeakyRelu(const eshkol_operations_t* op);

    /**
     * SiLU/Swish activation: x * sigmoid(x)
     * @param op The operation AST node
     * @return Tensor with SiLU applied element-wise
     */
    llvm::Value* tensorSilu(const eshkol_operations_t* op);

    // === Activation Backward Functions (for autodiff) ===

    /**
     * Softmax backward pass: computes gradient through softmax
     * Formula: dL/dx_i = s_i * (g_i - sum_j(g_j * s_j))
     * where s = softmax output, g = upstream gradient
     *
     * @param softmax_output The softmax forward pass output tensor
     * @param upstream_grad The gradient from the next layer
     * @return Tensor containing input gradients
     */
    llvm::Value* tensorSoftmaxBackward(llvm::Value* softmax_output, llvm::Value* upstream_grad);

    /**
     * ReLU backward pass: dL/dx = dL/dy * (x > 0 ? 1 : 0)
     * @param input The original input tensor (for sign check)
     * @param upstream_grad The gradient from the next layer
     * @return Tensor containing input gradients
     */
    llvm::Value* tensorReluBackward(llvm::Value* input, llvm::Value* upstream_grad);

    /**
     * Sigmoid backward pass: dL/dx = dL/dy * σ(x) * (1 - σ(x))
     * @param sigmoid_output The sigmoid forward pass output
     * @param upstream_grad The gradient from the next layer
     * @return Tensor containing input gradients
     */
    llvm::Value* tensorSigmoidBackward(llvm::Value* sigmoid_output, llvm::Value* upstream_grad);

    /**
     * GELU backward pass
     * @param input The original input tensor
     * @param upstream_grad The gradient from the next layer
     * @return Tensor containing input gradients
     */
    llvm::Value* tensorGeluBackward(llvm::Value* input, llvm::Value* upstream_grad);

    /**
     * Leaky ReLU backward pass: dL/dx = dL/dy * (x > 0 ? 1 : alpha)
     * @param input The original input tensor
     * @param upstream_grad The gradient from the next layer
     * @param alpha The negative slope (default 0.01)
     * @return Tensor containing input gradients
     */
    llvm::Value* tensorLeakyReluBackward(llvm::Value* input, llvm::Value* upstream_grad, double alpha = 0.01);

    /**
     * SiLU backward pass: dL/dx = dL/dy * (σ(x) + x * σ(x) * (1 - σ(x)))
     * @param input The original input tensor
     * @param upstream_grad The gradient from the next layer
     * @return Tensor containing input gradients
     */
    llvm::Value* tensorSiluBackward(llvm::Value* input, llvm::Value* upstream_grad);

    // === Statistics Operations ===

    /**
     * Compute tensor variance: (tensor-var tensor)
     * @param op The operation AST node
     * @return Scalar variance value
     */
    llvm::Value* tensorVar(const eshkol_operations_t* op);

    /**
     * Compute tensor standard deviation: (tensor-std tensor)
     * @param op The operation AST node
     * @return Scalar standard deviation value
     */
    llvm::Value* tensorStd(const eshkol_operations_t* op);

    /**
     * Compute tensor minimum: (tensor-min tensor)
     * @param op The operation AST node
     * @return Scalar minimum value
     */
    llvm::Value* tensorMin(const eshkol_operations_t* op);

    /**
     * Compute tensor maximum: (tensor-max tensor)
     * @param op The operation AST node
     * @return Scalar maximum value
     */
    llvm::Value* tensorMax(const eshkol_operations_t* op);

    /**
     * Compute index of minimum: (tensor-argmin tensor)
     * @param op The operation AST node
     * @return Integer index of minimum value
     */
    llvm::Value* tensorArgmin(const eshkol_operations_t* op);

    /**
     * Compute index of maximum: (tensor-argmax tensor)
     * @param op The operation AST node
     * @return Integer index of maximum value
     */
    llvm::Value* tensorArgmax(const eshkol_operations_t* op);

    /**
     * Compute covariance matrix: (tensor-cov x y)
     * @param op The operation AST node
     * @return Covariance value or matrix
     */
    llvm::Value* tensorCov(const eshkol_operations_t* op);

    /**
     * Compute correlation coefficient: (tensor-corrcoef x y)
     * @param op The operation AST node
     * @return Correlation coefficient
     */
    llvm::Value* tensorCorrcoef(const eshkol_operations_t* op);

    // === Random Tensor Generation ===

    /**
     * Create tensor with uniform random values [0, 1): (rand dim1 dim2 ...)
     * @param op The operation AST node with dimensions
     * @return Tensor filled with uniform random values
     */
    llvm::Value* tensorRand(const eshkol_operations_t* op);

    /**
     * Create tensor with normal random values: (randn dim1 dim2 ...)
     * @param op The operation AST node with dimensions
     * @return Tensor filled with standard normal values (mean=0, std=1)
     */
    llvm::Value* tensorRandn(const eshkol_operations_t* op);

    /**
     * Create tensor with random integers: (randint low high dim1 dim2 ...)
     * @param op The operation AST node with low, high, and dimensions
     * @return Tensor filled with random integers in [low, high)
     */
    llvm::Value* tensorRandint(const eshkol_operations_t* op);

    // === Shape Operations ===

    /**
     * Get tensor shape: (tensor-shape tensor)
     * @param op The operation AST node
     * @return Vector of dimensions
     */
    llvm::Value* tensorShape(const eshkol_operations_t* op);

    /**
     * Transpose tensor: (transpose tensor)
     * @param op The operation AST node
     * @return Transposed tensor
     */
    llvm::Value* transpose(const eshkol_operations_t* op);

    /**
     * Reshape tensor: (reshape tensor new-dims...)
     * @param op The operation AST node
     * @return Reshaped tensor (shares data with original)
     */
    llvm::Value* reshape(const eshkol_operations_t* op);

    /**
     * Squeeze tensor: remove dimensions of size 1
     * (squeeze tensor) - remove all size-1 dims
     * (squeeze tensor dim) - remove specific dim if size is 1
     * @param op The operation AST node
     * @return Tensor with same data, fewer dimensions
     */
    llvm::Value* squeeze(const eshkol_operations_t* op);

    /**
     * Unsqueeze tensor: add a dimension of size 1 at position
     * (unsqueeze tensor dim) - add size-1 dim at position
     * @param op The operation AST node
     * @return Tensor with same data, one more dimension
     */
    llvm::Value* unsqueeze(const eshkol_operations_t* op);

    /**
     * Concatenate tensors along an axis
     * (concatenate axis tensor1 tensor2 ...)
     * @param op The operation AST node
     * @return Concatenated tensor
     */
    llvm::Value* concatenate(const eshkol_operations_t* op);

    /**
     * Stack tensors on a new axis
     * (stack axis tensor1 tensor2 ...)
     * @param op The operation AST node
     * @return Stacked tensor with new dimension
     */
    llvm::Value* stack(const eshkol_operations_t* op);

    /**
     * Split tensor into chunks along an axis
     * (split tensor num-chunks axis) - split into equal chunks
     * @param op The operation AST node
     * @return List of tensor views
     */
    llvm::Value* split(const eshkol_operations_t* op);

    /**
     * Slice tensor: extract a subtensor
     * (slice tensor start end) - 1D slice
     * (slice tensor starts ends) - multi-dim slice with lists
     * @param op The operation AST node
     * @return Sliced tensor (view or copy based on contiguity)
     */
    llvm::Value* slice(const eshkol_operations_t* op);

    /**
     * Flatten tensor to 1D
     * (flatten tensor) - flatten all dimensions
     * @param op The operation AST node
     * @return 1D tensor with all elements
     */
    llvm::Value* flatten(const eshkol_operations_t* op);

    /**
     * Tile tensor: repeat tensor along dimensions
     * (tile tensor reps) - repeat according to reps list
     * @param op The operation AST node
     * @return Tiled tensor
     */
    llvm::Value* tile(const eshkol_operations_t* op);

    /**
     * Pad tensor: add padding with a value
     * (pad tensor pad-width value) - pad tensor on each side
     * @param op The operation AST node
     * @return Padded tensor
     */
    llvm::Value* pad(const eshkol_operations_t* op);

    // === Convolution & Pooling Operations ===

    /**
     * 2D max pooling: (max-pool2d input kernel-size stride padding)
     * @param op The operation AST node
     * @return Pooled tensor
     */
    llvm::Value* maxPool2d(const eshkol_operations_t* op);

    /**
     * 2D average pooling: (avg-pool2d input kernel-size stride padding)
     * @param op The operation AST node
     * @return Pooled tensor
     */
    llvm::Value* avgPool2d(const eshkol_operations_t* op);

    /**
     * 1D convolution: (conv1d input kernel stride padding)
     * @param op The operation AST node
     * @return Convolved tensor
     */
    llvm::Value* conv1d(const eshkol_operations_t* op);

    /**
     * 2D convolution: (conv2d input kernel stride padding)
     * Uses im2col + GEMM for efficient computation
     * @param op The operation AST node
     * @return Convolved tensor
     */
    llvm::Value* conv2d(const eshkol_operations_t* op);

    /**
     * 3D convolution: (conv3d input kernel stride padding)
     * Uses im2col + GEMM for efficient computation
     * @param op The operation AST node
     * @return Convolved tensor
     */
    llvm::Value* conv3d(const eshkol_operations_t* op);

    /**
     * Batch normalization: (batch-norm input gamma beta epsilon)
     * @param op The operation AST node
     * @return Normalized tensor
     */
    llvm::Value* batchNorm(const eshkol_operations_t* op);

    /**
     * Layer normalization: (layer-norm input gamma beta epsilon)
     * @param op The operation AST node
     * @return Normalized tensor
     */
    llvm::Value* layerNorm(const eshkol_operations_t* op);

    // === Tensor Creation Functions ===

    /**
     * Create tensor from elements: (tensor e1 e2 e3 ...)
     * @param op The operation AST node
     * @return 1D tensor with the provided elements
     */
    llvm::Value* tensor(const eshkol_operations_t* op);

    /**
     * Create zero-filled tensor: (zeros dim1 dim2 ...)
     * @param op The operation AST node
     * @return Tensor filled with zeros
     */
    llvm::Value* zeros(const eshkol_operations_t* op);

    /**
     * Create one-filled tensor: (ones dim1 dim2 ...)
     * @param op The operation AST node
     * @return Tensor filled with ones
     */
    llvm::Value* ones(const eshkol_operations_t* op);

    /**
     * Create identity matrix: (eye n) or (eye rows cols)
     * @param op The operation AST node
     * @return Identity matrix
     */
    llvm::Value* eye(const eshkol_operations_t* op);

    /**
     * Create range tensor: (arange stop) or (arange start stop) or (arange start stop step)
     * @param op The operation AST node
     * @return Range tensor
     */
    llvm::Value* arange(const eshkol_operations_t* op);

    /**
     * Create linspace tensor: (linspace start stop num)
     * @param op The operation AST node
     * @return Evenly spaced tensor
     */
    llvm::Value* linspace(const eshkol_operations_t* op);

    // === Tensor Utility (Public for use by other codegen modules) ===

    /**
     * Create a tensor with given dimensions.
     * @param dims Vector of dimension values
     * @param fill_value Optional fill value (as i64 bit pattern)
     * @param use_memset_zero If true, use memset for efficient zero-fill
     * @return Pointer to tensor struct
     */
    llvm::Value* createTensorWithDims(const std::vector<llvm::Value*>& dims,
                                       llvm::Value* fill_value = nullptr,
                                       bool use_memset_zero = false);

    // === Optimizers (Track 10.1) ===

    /**
     * SGD optimizer step with momentum: (sgd-step! params grads lr [momentum velocity])
     *
     * Updates parameters in-place using SGD with optional momentum.
     * Without momentum: param = param - lr * grad
     * With momentum: v = momentum * v + grad; param = param - lr * v
     *
     * @param op The operation AST node containing:
     *   - params: tensor of parameters to update (modified in-place)
     *   - grads: tensor of gradients (same shape as params)
     *   - lr: learning rate (scalar)
     *   - momentum: optional momentum coefficient (default 0.0, no momentum)
     *   - velocity: optional velocity tensor (required if momentum > 0, modified in-place)
     * @return The updated params tensor
     */
    llvm::Value* sgdStep(const eshkol_operations_t* op);

    /**
     * Adam optimizer step: (adam-step! params grads lr m v t [beta1 beta2 eps])
     *
     * Implements the Adam optimizer algorithm (Kingma & Ba, 2014):
     * m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
     * v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
     * m̂_t = m_t / (1 - β₁^t)
     * v̂_t = v_t / (1 - β₂^t)
     * θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
     *
     * @param op The operation AST node containing:
     *   - params: tensor of parameters to update (modified in-place)
     *   - grads: tensor of gradients (same shape as params)
     *   - lr: learning rate (scalar, typically 0.001)
     *   - m: first moment estimate tensor (modified in-place)
     *   - v: second moment estimate tensor (modified in-place)
     *   - t: timestep (integer, starts at 1)
     *   - beta1: optional first moment decay (default 0.9)
     *   - beta2: optional second moment decay (default 0.999)
     *   - eps: optional numerical stability term (default 1e-8)
     * @return The updated params tensor
     */
    llvm::Value* adamStep(const eshkol_operations_t* op);

    /**
     * Zero gradients: (zero-grad! grad-tensor)
     *
     * Zeros out all elements of a gradient tensor in-place.
     * Should be called before each forward pass in training loop.
     *
     * @param op The operation AST node containing gradient tensor
     * @return The zeroed tensor
     */
    llvm::Value* zeroGrad(const eshkol_operations_t* op);

    /**
     * Clip gradient norm: (clip-grad-norm! grads max-norm)
     *
     * Clips gradients by global norm. If ||grads|| > max_norm,
     * scales all gradients by max_norm / ||grads||.
     * Helps prevent gradient explosion in training.
     *
     * @param op The operation AST node containing:
     *   - grads: tensor of gradients (modified in-place)
     *   - max_norm: maximum allowed norm (scalar)
     * @return Total norm of gradients before clipping
     */
    llvm::Value* clipGradNorm(const eshkol_operations_t* op);

    /**
     * RMSprop optimizer step: (rmsprop-step! params grads lr v [alpha eps])
     *
     * Implements RMSprop optimizer:
     * v_t = α * v_{t-1} + (1 - α) * g_t²
     * θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
     *
     * @param op The operation AST node
     * @return The updated params tensor
     */
    llvm::Value* rmspropStep(const eshkol_operations_t* op);

    // === Loss Functions (Track 6.3) ===

    /**
     * Mean Squared Error loss: (mse-loss predictions targets)
     *
     * MSE = (1/n) * Σ(pred_i - target_i)²
     *
     * @param op The operation AST node
     * @return Scalar MSE loss value
     */
    llvm::Value* mseLoss(const eshkol_operations_t* op);

    /**
     * Cross-entropy loss: (cross-entropy-loss logits targets)
     *
     * For classification: -Σ target_i * log(softmax(logits)_i)
     *
     * @param op The operation AST node containing:
     *   - logits: raw unnormalized predictions
     *   - targets: one-hot encoded or class indices
     * @return Scalar cross-entropy loss value
     */
    llvm::Value* crossEntropyLoss(const eshkol_operations_t* op);

    /**
     * Binary cross-entropy loss: (bce-loss predictions targets)
     *
     * BCE = -Σ(target_i * log(pred_i) + (1 - target_i) * log(1 - pred_i))
     *
     * @param op The operation AST node
     * @return Scalar BCE loss value
     */
    llvm::Value* bceLoss(const eshkol_operations_t* op);

    /**
     * Huber loss (smooth L1): (huber-loss predictions targets [delta])
     *
     * L_δ(a) = 0.5 * a² if |a| ≤ δ, else δ * (|a| - 0.5 * δ)
     * Less sensitive to outliers than MSE.
     *
     * @param op The operation AST node
     * @return Scalar Huber loss value
     */
    llvm::Value* huberLoss(const eshkol_operations_t* op);

    /**
     * Mean Absolute Error loss: (mae-loss predictions targets)
     *
     * Computes MAE = (1/n) * sum(|pred - target|)
     * Also known as L1 loss. More robust to outliers than MSE.
     *
     * @param op The operation AST node
     * @return Scalar MAE loss value
     */
    llvm::Value* maeLoss(const eshkol_operations_t* op);

    /**
     * Binary Cross-Entropy loss: (binary-cross-entropy-loss predictions targets)
     *
     * For binary classification where predictions are probabilities [0, 1].
     * BCE = -(1/n) * sum(target * log(pred) + (1-target) * log(1-pred))
     *
     * @param op The operation AST node containing:
     *   - predictions: tensor of predicted probabilities (after sigmoid)
     *   - targets: tensor of binary labels (0 or 1)
     * @return Scalar binary cross-entropy loss value
     */
    llvm::Value* binaryCrossEntropyLoss(const eshkol_operations_t* op);

    // === Data Loading Infrastructure ===

    /**
     * Create a dataloader: (make-dataloader data-tensor batch-size [shuffle])
     *
     * Creates a dataloader that iterates over a dataset tensor in batches.
     * The dataloader maintains internal state for current position.
     *
     * @param op The operation AST node containing:
     *   - data: tensor of shape [num_samples, ...] containing all data
     *   - batch_size: number of samples per batch
     *   - shuffle: optional boolean to shuffle indices (default: false)
     * @return Dataloader object (opaque pointer)
     */
    llvm::Value* makeDataloader(const eshkol_operations_t* op);

    /**
     * Get next batch from dataloader: (dataloader-next loader)
     *
     * Returns the next batch of data, or null if end of epoch.
     *
     * @param op The operation AST node containing the dataloader
     * @return Tensor containing the next batch, or null if exhausted
     */
    llvm::Value* dataloaderNext(const eshkol_operations_t* op);

    /**
     * Reset dataloader to beginning: (dataloader-reset! loader)
     *
     * Resets the dataloader to the first batch, optionally re-shuffling.
     *
     * @param op The operation AST node containing the dataloader
     * @return The dataloader object (for chaining)
     */
    llvm::Value* dataloaderReset(const eshkol_operations_t* op);

    /**
     * Get number of batches in dataloader: (dataloader-length loader)
     *
     * @param op The operation AST node containing the dataloader
     * @return Integer number of batches
     */
    llvm::Value* dataloaderLength(const eshkol_operations_t* op);

    /**
     * Check if dataloader has more batches: (dataloader-has-next? loader)
     *
     * @param op The operation AST node containing the dataloader
     * @return Boolean indicating if more batches available
     */
    llvm::Value* dataloaderHasNext(const eshkol_operations_t* op);

    /**
     * Create train/test split: (train-test-split data labels ratio [shuffle])
     *
     * Splits dataset into training and test sets.
     *
     * @param op The operation AST node containing:
     *   - data: tensor of samples
     *   - labels: tensor of labels (optional)
     *   - ratio: fraction for training (e.g., 0.8)
     *   - shuffle: whether to shuffle before splitting
     * @return Vector of (train-data train-labels test-data test-labels)
     */
    llvm::Value* trainTestSplit(const eshkol_operations_t* op);

    // ════════════════════════════════════════════════════════════════════════════
    // TRANSFORMER ARCHITECTURE (Track 8)
    // ════════════════════════════════════════════════════════════════════════════

    // === Track 8.1: Scaled Dot-Product Attention ===

    /**
     * Scaled dot-product attention: (scaled-dot-attention Q K V [mask])
     *
     * Implements the attention mechanism from "Attention Is All You Need":
     *   scores = Q @ K^T / sqrt(d_k)
     *   attention = softmax(scores + mask)  ; mask is optional
     *   output = attention @ V
     *
     * @param op The operation AST node containing:
     *   - Q: Query tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
     *   - K: Key tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
     *   - V: Value tensor of shape (batch, seq_len, d_v) or (seq_len, d_v)
     *   - mask: Optional attention mask (0 for attend, -inf for mask)
     * @return Output tensor of shape (batch, seq_len, d_v) or (seq_len, d_v)
     */
    llvm::Value* scaledDotProductAttention(const eshkol_operations_t* op);

    // === Track 8.2: Multi-Head Attention ===

    /**
     * Multi-head attention: (multi-head-attention Q K V num-heads W_Q W_K W_V W_O [mask])
     *
     * Projects Q, K, V through learned weight matrices, splits into num_heads,
     * applies scaled dot-product attention to each head, concatenates, and
     * projects through output weights.
     *
     * @param op The operation AST node containing:
     *   - Q: Query tensor (batch, seq_len, d_model) or (seq_len, d_model)
     *   - K: Key tensor (batch, seq_len, d_model) or (seq_len, d_model)
     *   - V: Value tensor (batch, seq_len, d_model) or (seq_len, d_model)
     *   - num_heads: Number of attention heads (must divide d_model evenly)
     *   - W_Q: Query projection weights (d_model, d_model)
     *   - W_K: Key projection weights (d_model, d_model)
     *   - W_V: Value projection weights (d_model, d_model)
     *   - W_O: Output projection weights (d_model, d_model)
     *   - mask: Optional attention mask
     * @return Output tensor of shape (batch, seq_len, d_model)
     */
    llvm::Value* multiHeadAttention(const eshkol_operations_t* op);

    // === Track 8.3: Positional Encoding ===

    /**
     * Sinusoidal positional encoding: (positional-encoding max-len d-model)
     *
     * Generates positional embeddings using sine and cosine functions:
     *   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
     *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
     *
     * @param op The operation AST node containing:
     *   - max_len: Maximum sequence length
     *   - d_model: Model dimension (embedding size)
     * @return Positional encoding tensor of shape (max_len, d_model)
     */
    llvm::Value* positionalEncoding(const eshkol_operations_t* op);

    /**
     * Rotary positional encoding (RoPE): (rotary-embedding x seq-positions dim)
     *
     * Implements RoPE from "RoFormer: Enhanced Transformer with Rotary
     * Position Embedding". Applies rotation matrices to queries and keys
     * based on position.
     *
     * @param op The operation AST node containing:
     *   - x: Input tensor to embed (batch, seq_len, d_model)
     *   - seq_positions: Position indices (seq_len,)
     *   - dim: Embedding dimension (typically d_model / num_heads)
     * @return Tensor with rotary embeddings applied
     */
    llvm::Value* rotaryEmbedding(const eshkol_operations_t* op);

    // === Track 8.4-8.6: Transformer Layers (Building Blocks) ===

    /**
     * Causal attention mask: (causal-mask seq-len)
     *
     * Creates a lower-triangular mask for autoregressive (decoder) attention.
     * Positions that should not attend get -infinity, others get 0.
     *
     * @param op The operation AST node containing:
     *   - seq_len: Sequence length
     * @return Mask tensor of shape (seq_len, seq_len)
     */
    llvm::Value* causalMask(const eshkol_operations_t* op);

    /**
     * Padding mask: (padding-mask lengths max-len)
     *
     * Creates mask to ignore padding tokens. Returns 0 for valid positions,
     * -infinity for padding positions.
     *
     * @param op The operation AST node containing:
     *   - lengths: Actual sequence lengths (batch,)
     *   - max_len: Maximum sequence length
     * @return Mask tensor of shape (batch, max_len)
     */
    llvm::Value* paddingMask(const eshkol_operations_t* op);

    /**
     * Feed-forward network: (feed-forward x W1 b1 W2 b2)
     *
     * Two-layer MLP with ReLU/GELU activation:
     *   FFN(x) = W2 * ReLU(W1 * x + b1) + b2
     *
     * @param op The operation AST node containing:
     *   - x: Input tensor (batch, seq_len, d_model)
     *   - W1: First layer weights (d_model, d_ff)
     *   - b1: First layer bias (d_ff,)
     *   - W2: Second layer weights (d_ff, d_model)
     *   - b2: Second layer bias (d_model,)
     * @return Output tensor (batch, seq_len, d_model)
     */
    llvm::Value* feedForward(const eshkol_operations_t* op);

    /**
     * Dropout layer: (dropout x rate training)
     *
     * Randomly zeros elements during training, scales by 1/(1-rate).
     *
     * @param op The operation AST node containing:
     *   - x: Input tensor
     *   - rate: Dropout probability (0.0 to 1.0)
     *   - training: Boolean - apply dropout only if true
     * @return Tensor with dropout applied (or unchanged if not training)
     */
    llvm::Value* dropout(const eshkol_operations_t* op);

    /**
     * Embedding lookup: (embedding indices weights)
     *
     * Looks up embeddings from a weight matrix by indices.
     *
     * @param op The operation AST node containing:
     *   - indices: Integer tensor of token indices (batch, seq_len)
     *   - weights: Embedding matrix (vocab_size, d_model)
     * @return Embedded tensor (batch, seq_len, d_model)
     */
    llvm::Value* embedding(const eshkol_operations_t* op);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

#ifdef ESHKOL_XLA_ENABLED
    // XLA backend for accelerated tensor operations on large tensors
    std::unique_ptr<xla::XLACodegen> xla_;

    /**
     * Check if XLA should be used for an operation.
     * @param num_elements Total elements in the operation
     * @return true if XLA should be used (size > threshold and XLA available)
     */
    bool shouldUseXLA(size_t num_elements) const;
#endif

    // Callback for AST code generation (matches other codegen modules)
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);
    using TypedToTaggedFunc = llvm::Value* (*)(void* typed_value, void* context);

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    TypedToTaggedFunc typed_to_tagged_callback_ = nullptr;
    void* callback_context_ = nullptr;

    // === Internal Helpers ===

    /**
     * Call codegenAST via callback.
     */
    llvm::Value* codegenAST(const eshkol_ast_t* ast) {
        if (codegen_ast_callback_ && ast) {
            return codegen_ast_callback_(ast, callback_context_);
        }
        return nullptr;
    }

    /**
     * Scheme vector arithmetic (VECTOR_PTR type).
     * Vectors use tagged_value elements after an 8-byte length field.
     * @param vec1 First vector (tagged)
     * @param vec2 Second vector (tagged)
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result vector (tagged)
     */
    llvm::Value* schemeVectorArithmetic(llvm::Value* vec1, llvm::Value* vec2, const std::string& operation);

    /**
     * Tensor arithmetic for TENSOR_PTR type.
     * Tensors use double elements in a contiguous array.
     * @param tensor1 First tensor (tagged)
     * @param tensor2 Second tensor (tagged)
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result tensor (tagged)
     */
    llvm::Value* rawTensorArithmetic(llvm::Value* tensor1, llvm::Value* tensor2, const std::string& operation);

    /**
     * SIMD-accelerated tensor arithmetic for TENSOR_PTR type.
     * Automatically selects optimal vector width based on CPU capabilities:
     * - ARM NEON: 2 doubles (128-bit)
     * - x86 SSE2: 2 doubles (128-bit)
     * - x86 AVX/AVX2: 4 doubles (256-bit)
     * - x86 AVX-512: 8 doubles (512-bit)
     * Falls back to scalar for tail elements (count % SIMD_WIDTH != 0).
     * @param tensor1 First tensor (tagged)
     * @param tensor2 Second tensor (tagged)
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result tensor (tagged)
     */
    llvm::Value* rawTensorArithmeticSIMD(llvm::Value* tensor1, llvm::Value* tensor2, const std::string& operation);

    /**
     * Get the optimal SIMD vector width based on detected CPU features.
     * @return Vector width in number of doubles (1, 2, 4, or 8)
     */
    unsigned getSIMDWidth() const;

    /**
     * Get the LLVM vector type for the current SIMD width.
     * @return <2 x double>, <4 x double>, or <8 x double> based on CPU
     */
    llvm::VectorType* getSIMDVectorType() const;

    /**
     * Extract a tagged value as double, handling both int64 and double types.
     * @param tagged_val The tagged value
     * @return The extracted double value
     */
    llvm::Value* extractAsDouble(llvm::Value* tagged_val);

public:
    /**
     * Set callbacks for AST code generation.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        TypedToTaggedFunc typed_to_tagged,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        typed_to_tagged_callback_ = typed_to_tagged;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_TENSOR_CODEGEN_H
