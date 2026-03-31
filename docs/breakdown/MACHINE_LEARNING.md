# Eshkol Machine Learning Framework

## Technical Reference

**Source:** `lib/backend/tensor_codegen.cpp` (19,187 lines)
**Implementation level:** Compiler-level builtins via LLVM IR codegen
**Tensor representation:** Homogeneous `double` arrays stored as `int64` bitpatterns (8 bytes per element)
**Acceleration:** SIMD vectorization (ARM NEON / x86 SSE/AVX), Apple Accelerate (cBLAS/AMX), Metal GPU

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Activation Functions](#2-activation-functions)
3. [Loss Functions](#3-loss-functions)
4. [Optimizers](#4-optimizers)
5. [Weight Initializers](#5-weight-initializers)
6. [Learning Rate Schedulers](#6-learning-rate-schedulers)
7. [CNN Operations](#7-cnn-operations)
8. [Transformer Operations](#8-transformer-operations)
9. [Data Loading](#9-data-loading)
10. [Linear Algebra](#10-linear-algebra)
11. [Einstein Summation](#11-einstein-summation)
12. [Training Loop Example](#12-training-loop-example)

---

## 1. Introduction

Eshkol provides a comprehensive machine learning framework consisting of 75+ compiler-level
builtins spanning neural network layers, optimizers, loss functions, linear algebra
decompositions, and data loading infrastructure. Unlike library-based ML frameworks where
operations are interpreted or dispatched through a runtime, every ML primitive in Eshkol is
lowered directly to LLVM IR during compilation. This eliminates interpreter overhead, enables
aggressive LLVM optimization passes (constant folding, loop unrolling, vectorization), and
produces native machine code that operates on the same tensor representation used by the rest
of the language.

### Codegen Architecture

All ML builtins are implemented in `TensorCodegen`. Each operation follows a consistent pattern:

1. **Argument unpacking** -- Tagged values are unwrapped to raw tensor pointers.
2. **Shape extraction** -- Dimensions, rank, element pointer, and total element count are
   loaded from the tensor struct `{dims*, ndim, elems*, total}`.
3. **Result allocation** -- A new tensor is allocated via the arena allocator, with
   dimensions copied from the input.
4. **SIMD loop** -- The main computation processes `SIMD_WIDTH` elements per iteration
   using LLVM vector types.
5. **Scalar tail loop** -- Remaining elements (count mod `SIMD_WIDTH`) are processed individually.
6. **Result packing** -- The result tensor pointer is packed into a tagged value and returned.

Shape-preserving operations (activations, elementwise losses) produce new tensors. In-place
mutation is used only for optimizers (suffixed with `!`), weight initializers, and gradient
utilities. Backward passes for key activations (GELU, SiLU) are implemented as dedicated
codegen methods (`tensorGeluBackward`, `tensorSiluBackward`) for AD integration.

---

## 2. Activation Functions

Eshkol provides 14 activation functions, each compiled to SIMD-accelerated native code.
All accept a single tensor argument (some accept optional parameters) and return a new
tensor of identical shape.

### Key Implementations

**ReLU** (`tensor_codegen.cpp:3064-3193`): `f(x) = max(0, x)`. The SIMD path uses
`fcmp ogt` followed by `select` on vector types for branchless evaluation.

```scheme
(relu #(1.0 -2.0 3.0 -0.5))  ; => #(1.0 0.0 3.0 0.0)
```

**Sigmoid** (`tensor_codegen.cpp:3206-3370`): Uses a numerically stable formulation
that computes `exp(-|x|)` (argument always non-positive, preventing overflow), then
selects between two equivalent forms:

    x >= 0:  sigma(x) = 1 / (1 + exp(-|x|))
    x <  0:  sigma(x) = exp(-|x|) / (1 + exp(-|x|))

Both SIMD and scalar paths employ `llvm.exp` and `llvm.fabs` vector intrinsics.

**Softmax** (`tensor_codegen.cpp:3371-3483+`): Three-pass numerically stable algorithm:
(1) find global max, (2) compute `exp(x_i - max)` and accumulate sum, (3) divide.
The two-argument form `(softmax tensor axis)` delegates to `eshkol_xla_softmax` for
axis-aware computation in transformer batch dimensions.

**GELU** (`tensor_codegen.cpp:3631+`): PyTorch-standard tanh approximation:
`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. A dedicated backward
pass (`tensorGeluBackward`) computes exact analytical gradients.

**Leaky ReLU** (`tensor_codegen.cpp:3824-3932`): `x > 0 ? x : alpha*x` with
default alpha=0.01. SIMD path uses `fmul` for scaling and `select` for branching.

### Complete Activation Reference

| Function | Formula | Parameters | Use Case |
|---|---|---|---|
| `relu` | max(0, x) | -- | Hidden layers, CNNs |
| `sigmoid` | 1/(1+exp(-x)) | -- | Binary classification output |
| `softmax` | exp(x_i)/sum(exp(x_j)) | optional axis | Multi-class output |
| `gelu` | 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))) | -- | Transformer FFN |
| `leaky-relu` | x if x>0, else alpha*x | alpha (default 0.01) | Avoiding dead neurons |
| `silu` | x * sigmoid(x) | -- | Modern architectures (SiLU/Swish) |
| `elu` | x if x>0, else alpha*(exp(x)-1) | alpha (default 1.0) | Smooth negative region |
| `selu` | lambda*(x if x>0, else alpha*(exp(x)-1)) | lambda=1.0507, alpha=1.6733 | Self-normalizing networks |
| `mish` | x * tanh(softplus(x)) | -- | Object detection (YOLOv4) |
| `hard-swish` | x * min(max(x+3,0),6)/6 | -- | MobileNetV3 (efficient) |
| `hard-sigmoid` | min(max(x+3,0),6)/6 | -- | Piecewise linear approx |
| `softplus` | ln(1 + exp(x)) | -- | Smooth ReLU approximation |
| `celu` | max(0,x)+min(0,alpha*(exp(x/alpha)-1)) | alpha (default 1.0) | Continuously differentiable ELU |
| `dropout` | x * mask / (1-p) | p (drop probability) | Regularization (training) |

All activations use `alloca`-based loop counters (not PHI nodes) to avoid conflicts
with basic block creation in closure calls.

---

## 3. Loss Functions

14 loss functions covering regression, classification, metric learning, and specialized
applications. Each returns a scalar tensor (1-element) representing the aggregated loss.

### Key Implementations

**MSE Loss** (`tensor_codegen.cpp:14330+`): `L = (1/N) * sum((y_hat - y)^2)`.
Gradient: `2(y_hat - y)/N`. Sensitive to outliers, strong gradients far from optimum.

**Cross-Entropy Loss** (`tensor_codegen.cpp:14424+`): `L = -(1/N) * sum(y * log(y_hat + eps))`.
Expects post-softmax probabilities and one-hot targets. Epsilon `1e-12` prevents log(0).

**BCE Loss** (`tensor_codegen.cpp:14590+`):
`L = -(1/N) * sum(y*log(p+eps) + (1-y)*log(1-p+eps))`. For binary classification with
sigmoid output. Aliased as `binary-cross-entropy-loss`.

**Huber Loss** (`tensor_codegen.cpp:14705+`): Quadratic for |error| <= delta, linear
otherwise. Default delta=1.0. Combines MSE precision with MAE outlier robustness.

```scheme
(mse-loss #(2.5 0.0 2.1) #(3.0 -0.5 2.0))
(cross-entropy-loss (softmax logits) one-hot-labels)
(huber-loss predictions targets 1.5)  ; custom delta
```

### Complete Loss Function Reference

| Function | Formula | Arguments | Application |
|---|---|---|---|
| `mse-loss` | mean((y_hat-y)^2) | predictions, targets | Regression |
| `mae-loss` | mean(\|y_hat-y\|) | predictions, targets | Robust regression |
| `cross-entropy-loss` | -mean(y*log(p)) | predictions, targets | Multi-class classification |
| `bce-loss` | -mean(y*log(p)+(1-y)*log(1-p)) | predictions, targets | Binary classification |
| `huber-loss` | quadratic/linear hybrid | preds, targets, [delta=1.0] | Outlier-robust regression |
| `kl-div-loss` | sum(P*log(P/Q)) | P (true), Q (predicted) | Distribution matching |
| `hinge-loss` | mean(max(0, 1-y*y_hat)) | preds, targets (+/-1) | SVM-style classification |
| `smooth-l1-loss` | smooth absolute | preds, targets, [beta=1.0] | Object detection |
| `focal-loss` | -(1-p)^gamma * log(p) | preds, targets, [gamma=2.0] | Class-imbalanced detection |
| `triplet-loss` | max(0, d(a,p)-d(a,n)+m) | anchor, pos, neg, [margin=1.0] | Metric learning |
| `contrastive-loss` | y*d^2+(1-y)*max(0,m-d)^2 | t1, t2, labels, [margin=1.0] | Siamese networks |
| `label-smoothing-loss` | CE with smoothed targets | logits, targets, n_cls, [eps=0.1] | Regularized classification |
| `cosine-embedding-loss` | 1-cos or max(0,cos-m) | t1, t2, label, [margin=0.0] | Embedding similarity |

---

## 4. Optimizers

Five gradient-based update rules plus three gradient management utilities. Optimizers
mutate parameters in-place (indicated by the `!` suffix) and operate directly on tensor
element arrays without allocation.

### 4.1 SGD with Momentum

**Signature:** `(sgd-step! params grads lr [momentum velocity])`
**Implementation:** `tensor_codegen.cpp:11576+`

    Without momentum:  theta <- theta - lr * grad
    With momentum:     v <- momentum * v + grad;  theta <- theta - lr * v

```scheme
(sgd-step! weights gradients 0.01)                    ; basic
(sgd-step! weights gradients 0.01 0.9 velocity)       ; with momentum
```

### 4.2 Adam

**Signature:** `(adam-step! params grads lr m v t [beta1 beta2 eps])`
**Implementation:** `tensor_codegen.cpp:11718+`

    m <- beta1*m + (1-beta1)*grad                 (first moment)
    v <- beta2*v + (1-beta2)*grad^2               (second moment)
    m_hat <- m / (1 - beta1^t)                    (bias correction)
    v_hat <- v / (1 - beta2^t)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

Defaults: beta1=0.9, beta2=0.999, eps=1e-8. Bias correction uses C `pow`.

### 4.3 AdamW

**Signature:** `(adamw-step! params grads lr m v t [beta1 beta2 eps weight_decay])`
**Implementation:** `tensor_codegen.cpp:12201+`

Decoupled weight decay (Loshchilov & Hutter, 2017). Weight decay is applied directly
to parameters, not through the gradient:

    theta <- theta * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

Default weight_decay=0.01. Recommended for transformer training.

### 4.4 RMSprop

**Signature:** `(rmsprop-step! params grads lr v [alpha eps])`
**Implementation:** `tensor_codegen.cpp:12075+`

    v <- alpha*v + (1-alpha)*grad^2;  theta <- theta - lr*grad/(sqrt(v)+eps)

Default: alpha=0.99, eps=1e-8. Effective for RNNs and non-stationary objectives.

### 4.5 Adagrad

**Signature:** `(adagrad-step! params grads lr accum [eps])`
**Implementation:** `tensor_codegen.cpp:12350+`

    accum <- accum + grad^2;  theta <- theta - lr*grad/(sqrt(accum)+eps)

Well-suited for sparse features but monotonically increasing accumulator can
cause premature learning rate decay.

### 4.6 Gradient Utilities

| Function | Signature | Description |
|---|---|---|
| `zero-grad!` | `(zero-grad! tensor)` | Sets all elements to 0.0 before backward pass |
| `clip-grad-norm!` | `(clip-grad-norm! grads max-norm)` | Rescales gradients to bound L2 norm |
| `check-grad-health` | `(check-grad-health tensor)` | Returns `#t` if all finite, `#f` if NaN/Inf found |

---

## 5. Weight Initializers

Five initialization strategies based on fan-in/fan-out analysis. All mutate in-place.

### Formulas

**Xavier Uniform** (`tensor_codegen.cpp:12509-12577`):
`U(-limit, limit)` where `limit = sqrt(6 / (fan_in + fan_out))`.
Uses `drand48()` scaled to range. Glorot & Bengio (2010).

**Xavier Normal** (`tensor_codegen.cpp:12579+`):
`N(0, std^2)` where `std = sqrt(2 / (fan_in + fan_out))`.

**Kaiming Uniform** (`tensor_codegen.cpp:12673+`):
`U(-limit, limit)` where `limit = sqrt(6 / fan_in)`. He et al. (2015), for ReLU.

**Kaiming Normal** (`tensor_codegen.cpp:12740+`):
`N(0, std^2)` where `std = sqrt(2 / fan_in)`. Standard for ReLU/Leaky ReLU.

**LeCun Normal** (`tensor_codegen.cpp:12829+`):
`N(0, std^2)` where `std = sqrt(1 / fan_in)`. LeCun et al. (1998), for SELU.

```scheme
(define W (make-tensor (list 256 128) 0.0))
(xavier-uniform! W 256 128)       ; sigmoid/tanh layers
(kaiming-normal! W 256)            ; ReLU layers
```

| Activation | Recommended Initializer |
|---|---|
| ReLU, Leaky ReLU | `kaiming-normal!` or `kaiming-uniform!` |
| Sigmoid, Tanh | `xavier-normal!` or `xavier-uniform!` |
| SELU | `lecun-normal!` |
| GELU, SiLU | `kaiming-normal!` |

---

## 6. Learning Rate Schedulers

Pure functions computing the learning rate for the current step. No mutable state.

**Linear Warmup** (`tensor_codegen.cpp:13007+`):
`lr = base_lr * min(1.0, step / warmup_steps)`. Essential for transformer training.

**Step Decay** (`tensor_codegen.cpp:12965+`):
`lr = base_lr * gamma^(floor(epoch / step_size))`. Classic CNN schedule.

**Exponential Decay** (`tensor_codegen.cpp:13037+`):
`lr = base_lr * gamma^epoch`. Smooth per-epoch decay.

**Cosine Annealing** (`tensor_codegen.cpp:12922+`):
`lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi * step / total_steps))`.
Smooth warmdown, commonly combined with linear warmup.

```scheme
(define lr (if (< step 4000)
               (linear-warmup-lr 0.001 step 4000)
               (cosine-annealing-lr 0.001 1e-6 step 100000)))
(step-decay-lr 0.1 0.1 epoch 30)            ; reduce 10x every 30 epochs
(exponential-decay-lr 0.01 0.95 epoch)       ; 5% decay per epoch
```

---

## 7. CNN Operations

Three convolution operations, two pooling operations, and two normalization layers.

### Convolutions

**Conv1d** (`tensor_codegen.cpp:9370+`): `(conv1d input kernel stride)`.
Output: `L_out = floor((L_in - K) / stride) + 1`. Direct convolution.

**Conv2d** (`tensor_codegen.cpp:9588+`): `(conv2d input kernel stride)`.
Kernel must be >= 2D. `H_out = floor((H_in - K_h) / stride) + 1`. Runtime guard on rank.

**Conv3d** (`tensor_codegen.cpp:11236+`): `(conv3d input kernel [stride] [padding])`.
Both tensors must be >= 3D. Six nested loops (3 output + 3 kernel dimensions).

```scheme
(conv1d signal #(0.25 0.5 0.25) 1)              ; 1D smoothing
(conv2d image (make-tensor (list 3 3) 0.0) 1)   ; 2D feature extraction
```

### Pooling

**Max Pool 2D** (`tensor_codegen.cpp:8794+`): `(max-pool2d input kernel-size stride)`.
Selects maximum within each window. Preserves strongest activations.

**Avg Pool 2D** (`tensor_codegen.cpp:9090+`): `(avg-pool2d input kernel-size stride)`.
Mean within each window. Smoother downsampling.

### Normalization

**Batch Norm** (`tensor_codegen.cpp:9855+`): `(batch-norm input gamma beta epsilon [axis])`.
`y = gamma * (x - mean) / sqrt(var + eps) + beta`. Normalizes across batch dimension.
5-argument form delegates to runtime for axis-aware normalization.

**Layer Norm** (`tensor_codegen.cpp:10101+`): `(layer-norm input gamma beta epsilon [axis])`.
Same formula but normalizes across the feature dimension (per-sample). Standard for
transformers. Independent of batch size.

---

## 8. Transformer Operations

Complete building blocks for encoder-decoder, decoder-only, and encoder-only architectures.

### Scaled Dot-Product Attention

**Signature:** `(scaled-dot-attention Q K V [mask])`
**Implementation:** `tensor_codegen.cpp:16647+`

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

`d_k` inferred from Q's last dimension. Optional mask uses additive convention
(masked positions set to `-1e9`). Supports both causal and padding masking.

### Multi-Head Attention

**Signature:** `(multi-head-attention Q K V num-heads W_Q W_K W_V W_O [mask])`
**Implementation:** `tensor_codegen.cpp:17186+`

Projects Q/K/V through learned weight matrices, splits into `num-heads` parallel
heads, applies scaled dot-product attention independently, concatenates, and
projects through W_O.

```scheme
(define W-Q (make-tensor (list 512 512) 0.0))
(xavier-uniform! W-Q 512 512)
;; ... similarly for W-K, W-V, W-O
(multi-head-attention Q K V 8 W-Q W-K W-V W-O mask)
```

### Positional Encoding

**Signature:** `(positional-encoding max-len d-model)` -- `tensor_codegen.cpp:17990+`

Sinusoidal encodings from Vaswani et al. (2017):
`PE(pos,2i) = sin(pos/10000^(2i/d))`, `PE(pos,2i+1) = cos(pos/10000^(2i/d))`.
Returns `[max-len, d-model]` tensor added to input embeddings.

### Rotary Position Embedding (RoPE)

**Signature:** `(rotary-embedding x seq-positions dim)` -- `tensor_codegen.cpp:18128+`

Su et al. (2021). Encodes position through rotation of feature pairs. Standard in
modern LLMs (LLaMA, Mistral) for superior length generalization.

### Masking

**`(causal-mask seq-len)`** (`tensor_codegen.cpp:18329+`): Lower-triangular additive
mask, upper triangle set to `-1e9`. For autoregressive decoders.

**`(padding-mask lengths max-len)`** (`tensor_codegen.cpp:18421+`): Creates mask from
sequence lengths. Positions beyond actual length are masked.

### Feed-Forward Network

**Signature:** `(feed-forward x W1 b1 W2 b2)` -- `tensor_codegen.cpp:18556+`

Position-wise FFN: `W2 * relu(W1 * x + b1) + b2`. Fused two-layer linear transform.

### Embedding

**Signature:** `(embedding indices weights)` -- `tensor_codegen.cpp:19031+`

Lookup table: maps integer indices to dense vectors. Weight matrix must be 2D
`[vocab-size, embed-dim]`. Runtime guard validates rank.

```scheme
(define embed-weights (make-tensor (list 10000 256) 0.0))
(xavier-uniform! embed-weights 10000 256)
(embedding #(42 17 256) embed-weights)  ; => 3 x 256 tensor
```

---

## 9. Data Loading

Batched iteration with optional shuffling for SGD-based training.

**`(make-dataloader data batch-size [shuffle])`** (`tensor_codegen.cpp:15995+`):
Creates dataloader (64-byte struct: position, batch size, length, shuffle index array).

| Function | Signature | Description |
|---|---|---|
| `dataloader-next` | `(dataloader-next loader)` | Returns next batch, advances cursor |
| `dataloader-has-next?` | `(dataloader-has-next? loader)` | `#t` if batches remain |
| `dataloader-reset!` | `(dataloader-reset! loader)` | Resets cursor; re-shuffles if enabled |
| `dataloader-length` | `(dataloader-length loader)` | Total number of batches |

**`(train-test-split data ratio [shuffle])`** (`tensor_codegen.cpp:16496+`):
Returns pair of tensors. Ratio specifies training fraction (e.g., 0.8 for 80/20 split).

```scheme
(define loader (make-dataloader training-data 32 #t))
(let loop ()
  (when (dataloader-has-next? loader)
    (let ((batch (dataloader-next loader)))
      ;; process batch
      (loop))))
```

---

## 10. Linear Algebra

Seven matrix decomposition and solver operations for numerical linear algebra. These
compile to LLVM IR loops over tensor element arrays — no LAPACK dependency required
(though cBLAS dispatch is used for matmul via the cost model).

### Decompositions

**LU Decomposition** (`tensor_codegen.cpp:13102+`): `(tensor-lu A)`.
Factorizes `A = P * L * U` via partial pivoting. Returns a list `(L U P)` where P is
a permutation vector. O(n^3) for n x n matrix. Foundation for determinant and solve.

**Cholesky Decomposition** (`tensor_codegen.cpp:13588+`): `(tensor-cholesky A)`.
Factorizes symmetric positive-definite `A = L * L^T`. Returns lower-triangular L.
Half the cost of LU. Raises error if A is not positive-definite (diagonal element
becomes non-positive during factorization).

**QR Decomposition** (`tensor_codegen.cpp:13684+`): `(tensor-qr A)`.
Factorizes `A = Q * R` via modified Gram-Schmidt. Returns list `(Q R)`.
Used for least-squares and eigenvalue algorithms.

**SVD** (`tensor_codegen.cpp:13869+`): `(tensor-svd A)`.
Singular Value Decomposition `A = U * S * V^T`. Returns list `(U S V)` where S is
a vector of singular values. Iterative algorithm with convergence threshold.

### Solvers

**Linear Solve** (`tensor_codegen.cpp:13452+`): `(tensor-solve A b)`.
Solves `Ax = b` via LU decomposition with forward/backward substitution. Returns x.

**Determinant** (`tensor_codegen.cpp:13272+`): `(tensor-det A)`.
Computes det(A) via LU decomposition as product of diagonal elements of U,
adjusted for permutation sign.

**Matrix Inverse** (`tensor_codegen.cpp:13348+`): `(tensor-inverse A)`.
Computes A^(-1) by solving `A * X = I` column by column via LU.

```scheme
(define A #(#(4.0 3.0) #(6.0 3.0)))
(define b #(10.0 12.0))
(tensor-solve A b)                ; => #(1.0 2.0)
(tensor-det A)                    ; => -6.0
(tensor-inverse A)                ; => #(#(-0.5 0.5) #(1.0 -0.667))

(define L (tensor-cholesky (tensor-dot (tensor-transpose A) A)))
(define (svd-result) (tensor-svd A))   ; => (U S V)
```

---

## 11. Einstein Summation

**Signature:** `(einsum spec A B)` -- `tensor_codegen.cpp:14092+`

General tensor contraction using Einstein summation convention. The `spec` string
encodes index patterns following NumPy/PyTorch notation:

```scheme
(einsum "ij,jk->ik" A B)         ; matrix multiply
(einsum "ij->ji" A)               ; transpose
(einsum "ii->" A)                 ; trace
(einsum "ij,ij->" A B)            ; Frobenius inner product
(einsum "ijk,ikl->ijl" A B)      ; batched matmul
```

The codegen parses the spec string at compile time, extracts index dimensions from
operand shapes, and generates nested loops with the correct contraction pattern.
Supports up to rank-16 operands (matching the XLA backend limit).

---

## 12. Training Loop Example

End-to-end two-layer network: initialization, forward pass, loss, gradient clipping,
Adam update, and cosine-annealed learning rate with warmup.

```scheme
;; Two-layer classifier: 784 -> 256 (GELU) -> 10 (softmax)

;; Weights
(define W1 (make-tensor (list 784 256) 0.0))
(define b1 (make-tensor (list 256) 0.0))
(define W2 (make-tensor (list 256 10) 0.0))
(define b2 (make-tensor (list 10) 0.0))
(kaiming-normal! W1 784)
(xavier-uniform! W2 256 10)

;; Adam state
(define m-W1 (make-tensor (list 784 256) 0.0))
(define v-W1 (make-tensor (list 784 256) 0.0))
(define m-W2 (make-tensor (list 256 10) 0.0))
(define v-W2 (make-tensor (list 256 10) 0.0))

;; Data
(define loader (make-dataloader training-data 32 #t))

(define (train epochs)
  (let epoch-loop ((epoch 0))
    (when (< epoch epochs)
      (define total-steps (* epochs (dataloader-length loader)))
      (dataloader-reset! loader)
      (let batch-loop ((t (+ (* epoch (dataloader-length loader)) 1)))
        (when (dataloader-has-next? loader)
          (define batch (dataloader-next loader))
          (define x (car batch))
          (define y (cdr batch))

          ;; LR: warmup then cosine decay
          (define lr (if (< t 1000)
                         (linear-warmup-lr 0.001 t 1000)
                         (cosine-annealing-lr 0.001 1e-6 t total-steps)))

          ;; Forward
          (define probs (softmax (tensor+ (matmul (gelu (tensor+ (matmul x W1) b1)) W2) b2)))
          (define loss (cross-entropy-loss probs y))

          ;; Backward (AD)
          (define grads (gradient loss (list W1 b1 W2 b2)))

          ;; Clip and update
          (when (check-grad-health (list-ref grads 0))
            (clip-grad-norm! (list-ref grads 0) 1.0)
            (clip-grad-norm! (list-ref grads 2) 1.0)
            (adam-step! W1 (list-ref grads 0) lr m-W1 v-W1 t)
            (adam-step! W2 (list-ref grads 2) lr m-W2 v-W2 t))

          (batch-loop (+ t 1))))
      (epoch-loop (+ epoch 1)))))

(train 100)
```

### Design Principles

1. **No framework boilerplate.** Tensors, activations, losses, and optimizers are all
   first-class compiler builtins -- no imports, no session management.
2. **Arena allocation.** Intermediate tensors are arena-allocated and automatically
   reclaimed. No GC overhead in the training loop.
3. **Compiled SIMD.** Every activation compiles to dual SIMD/scalar loops. ARM64 NEON
   processes 2 doubles/cycle; x86 AVX processes 4 doubles/cycle.
4. **In-place mutation.** Optimizers and initializers mutate tensors directly, avoiding
   allocation in the inner loop.
5. **Composable scheduling.** LR schedulers are pure functions returning scalars,
   enabling arbitrary composition without callback registrations.

---

## Appendix: Complete Builtin Reference

**Activation Functions (14):**
`relu` `sigmoid` `softmax` `gelu` `leaky-relu` `silu` `elu` `selu`
`mish` `hard-swish` `hard-sigmoid` `softplus` `celu` `dropout`

**Loss Functions (14):**
`mse-loss` `mae-loss` `cross-entropy-loss` `bce-loss` `binary-cross-entropy-loss`
`huber-loss` `kl-div-loss` `hinge-loss` `smooth-l1-loss` `focal-loss`
`triplet-loss` `contrastive-loss` `label-smoothing-loss` `cosine-embedding-loss`

**Optimizers (5) + Gradient Utilities (3):**
`sgd-step!` `adam-step!` `adamw-step!` `rmsprop-step!` `adagrad-step!`
`zero-grad!` `clip-grad-norm!` `check-grad-health`

**Weight Initializers (5):**
`xavier-uniform!` `xavier-normal!` `kaiming-uniform!` `kaiming-normal!` `lecun-normal!`

**Learning Rate Schedulers (4):**
`linear-warmup-lr` `step-decay-lr` `exponential-decay-lr` `cosine-annealing-lr`

**CNN Operations (7):**
`conv1d` `conv2d` `conv3d` `max-pool2d` `avg-pool2d` `batch-norm` `layer-norm`

**Transformer Operations (8):**
`scaled-dot-attention` `multi-head-attention` `positional-encoding`
`rotary-embedding` `causal-mask` `padding-mask` `feed-forward` `embedding`

**Data Loading (6):**
`make-dataloader` `dataloader-next` `dataloader-reset!` `dataloader-length`
`dataloader-has-next?` `train-test-split`

**Linear Algebra (7):**
`tensor-lu` `tensor-det` `tensor-inverse` `tensor-solve`
`tensor-cholesky` `tensor-qr` `tensor-svd`

**Einstein Summation (1):**
`einsum`
