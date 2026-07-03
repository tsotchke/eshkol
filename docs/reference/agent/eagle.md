# `agent.eagle` — Native Linear-Head Training (EAGLE)

A native, dependency-free fully-connected linear layer with mean-squared-error
loss and plain SGD — no Python/PyTorch. Used for feature-distillation training
of a linear head. Introduced in PR #104.

```scheme
(require agent.eagle)
```

Source: `lib/agent/eagle.esk`. C implementation:
`lib/agent/c/agent_eagle_training.c`. C ABI symbols: `qllm_ffi_linear_*`.

The layer computes `y = W · x` (row-major weights, shape `out×in`, **no bias**),
MSE loss over the outputs, gradient `grad = (2/out_dim)·(pred − target)·xᵀ`, and
`W -= lr·grad` on an SGD step. A handle is an opaque pointer you must destroy.

## API

| Procedure | Signature | Description |
|-----------|-----------|-------------|
| `eagle-linear-create` | `(eagle-linear-create in-dim out-dim)` | New layer; returns handle |
| `eagle-linear-destroy` | `(eagle-linear-destroy h)` | Free the layer |
| `eagle-linear-set-weight!` | `(eagle-linear-set-weight! h out in value)` | Set `W[out][in]` |
| `eagle-linear-weight` | `(eagle-linear-weight h out in)` | Get `W[out][in]` |
| `eagle-linear-set-input!` | `(eagle-linear-set-input! h in value)` | Set input `x[in]` |
| `eagle-linear-set-target!` | `(eagle-linear-set-target! h out value)` | Set target `t[out]` |
| `eagle-linear-forward!` | `(eagle-linear-forward! h)` | Compute `pred = W·x` |
| `eagle-linear-pred` | `(eagle-linear-pred h out)` | Read prediction `pred[out]` |
| `eagle-linear-loss` | `(eagle-linear-loss h)` | MSE over outputs |
| `eagle-linear-backward!` | `(eagle-linear-backward! h)` | Compute gradients |
| `eagle-linear-grad` | `(eagle-linear-grad h out in)` | Read `grad[out][in]` |
| `eagle-linear-sgd-step!` | `(eagle-linear-sgd-step! h lr)` | `W -= lr·grad` |
| `eagle-linear-train-step!` | `(eagle-linear-train-step! h lr)` | forward + backward + sgd in one call |

The wrappers add capability checks and translate C return codes into errors
(`eagle-check`). Dimensions are validated (max `1<<20`, overflow-guarded).

## Underlying C ABI

| C symbol | Signature |
|----------|-----------|
| `qllm_ffi_linear_create` | `void*(int64 in_dim, int64 out_dim)` |
| `qllm_ffi_linear_destroy` | `void(void*)` |
| `qllm_ffi_linear_set_weight` | `int32(void*, int64 out, int64 in, double)` (0 ok / -1 OOB) |
| `qllm_ffi_linear_get_weight` | `double(void*, int64 out, int64 in)` |
| `qllm_ffi_linear_set_input` | `int32(void*, int64 in, double)` |
| `qllm_ffi_linear_set_target` | `int32(void*, int64 out, double)` |
| `qllm_ffi_linear_forward` | `int32(void*)` |
| `qllm_ffi_linear_pred` | `double(void*, int64 out)` |
| `qllm_ffi_linear_loss` | `double(void*)` |
| `qllm_ffi_linear_backward` | `int32(void*)` |
| `qllm_ffi_linear_grad` | `double(void*, int64 out, int64 in)` |
| `qllm_ffi_linear_sgd_step` | `int32(void*, double lr)` |
| `qllm_ffi_linear_train_step` | `int32(void*, double lr)` |

## Verified example

```scheme
(require agent.eagle)
(define h (eagle-linear-create 2 1))
(eagle-linear-set-weight! h 0 0 0.5)
(eagle-linear-set-weight! h 0 1 -0.5)
(eagle-linear-set-input! h 0 1.0)
(eagle-linear-set-input! h 1 2.0)
(eagle-linear-forward! h)
(display (eagle-linear-pred h 0)) (newline)   ;; => -0.5   (0.5*1 + -0.5*2)
(eagle-linear-destroy h)
```

A training loop repeatedly sets inputs/targets and calls `eagle-linear-train-step!`
with a learning rate until `eagle-linear-loss` converges.
