# Tensor application shares ordinary callable dispatch

`(tensor-apply tensor callable)` evaluates both operands once and invokes the
resolved callable on each scalar in row-major order. No function-name table or
identity substitution participates. A non-callable raises even for an empty
tensor. Output dimensions match the input; output numeric precision is f64,
with tagged dual storage when a callback returns forward-mode AD values.

In LLVM, `TensorCodegen` receives the existing `codegenClosureCall` callback
through `builtin_factory_codegen.cpp`. It is the dispatcher used by ordinary
lambda applications, callable variables, and callable-producing expressions.
Builtin references become their ordinary procedure wrappers, so lexical
bindings and user definitions determine behavior. Callable validation guards
base tags before reading object headers. `loadTensorScalar`, shared with tensor
indexing, retains reverse-mode node pointers and complete forward jets across
the call. Results are staged as tagged values, then packed to the tensor slot
representation after their carrier is known.

In the VM, `vm_enter_call` is the single entry for threaded `OP_CALL`, switch
`OP_CALL`, and native higher-order invocation. Closure arity, parameter
invocation, continuation transfer, frame admission, and ordinary bytecode
entry therefore follow the same route. A native caller uses the existing
sentinel return frame. `tensor-apply` preserves the complete `VmDual` carrier,
including Taylor coefficients and captured differentiable values. Full sum
and mean reductions retain this carrier. A unary collection Hessian receives
one vector, matching gradient's arity contract, and uses the existing Taylor
carrier for exact mixed partials; spread scalar losses retain their existing
hyper-dual representation.

Run the required engine matrix with:

```sh
python3 scripts/run_tensor_apply_callable_matrix.py --build build
```

The CTest name is `tensor_apply_callable_matrix`. Each of LLVM JIT at O0,
LLVM AOT at O2, VM source execution, and emitted ESKB execution must report all
31 ordered assertions. Missing engines, missing assertions, timeouts, nonzero
exit statuses, and numerical failures fail the gate. JIT caching is disabled
so that the JIT row executes the in-process LLVM engine directly. Tests cover
builtin and named procedures, overridden historical whitelist names, lambdas,
captures, variables, returned/conditional callables, variadic procedures,
continuation escape, evaluation counts, empty tensors, errors, gradients,
differentiable captures, and Hessians for every user procedure form.
