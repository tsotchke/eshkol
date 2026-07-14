# Eshkol API Reference

Generated from the Doxygen `/** ... */` comment blocks in the public headers under `inc/eshkol/**/*.h`. Do not edit files under `docs/api/` by hand — regenerate with:

```sh
python3 scripts/gen_api_docs.py
```

**Coverage:** 1959/3432 public symbols documented (57.1%), 1473 undocumented.

See also [INDEX.md](INDEX.md) for an alphabetical symbol table.

## Subsystems

### (root headers)

252/550 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`eshkol.h`](eshkol.md) | 309 | 142 |
| [`eshkol_ffi.h`](eshkol_ffi.md) | 44 | 32 |
| [`http_request_utils.h`](http_request_utils.md) | 6 | 3 |
| [`llvm_backend.h`](llvm_backend.md) | 79 | 2 |
| [`logger.h`](logger.md) | 37 | 21 |
| [`model_io.h`](model_io.md) | 8 | 5 |
| [`platform_runtime.h`](platform_runtime.md) | 20 | 19 |
| [`runtime_exports.h`](runtime_exports.md) | 47 | 28 |

### `backend/`

1121/1885 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`backend/arithmetic_codegen.h`](backend/arithmetic_codegen.md) | 48 | 41 |
| [`backend/autodiff_codegen.h`](backend/autodiff_codegen.md) | 160 | 99 |
| [`backend/binding_codegen.h`](backend/binding_codegen.md) | 54 | 21 |
| [`backend/blas_backend.h`](backend/blas_backend.md) | 23 | 23 |
| [`backend/builtin_declarations.h`](backend/builtin_declarations.md) | 16 | 6 |
| [`backend/call_apply_codegen.h`](backend/call_apply_codegen.md) | 49 | 22 |
| [`backend/codegen_context.h`](backend/codegen_context.md) | 171 | 41 |
| [`backend/collection_codegen.h`](backend/collection_codegen.md) | 32 | 20 |
| [`backend/complex_codegen.h`](backend/complex_codegen.md) | 33 | 20 |
| [`backend/control_flow_codegen.h`](backend/control_flow_codegen.md) | 32 | 12 |
| [`backend/cpu_features.h`](backend/cpu_features.md) | 49 | 22 |
| [`backend/function_cache.h`](backend/function_cache.md) | 30 | 11 |
| [`backend/function_codegen.h`](backend/function_codegen.md) | 20 | 8 |
| [`backend/hash_codegen.h`](backend/hash_codegen.md) | 36 | 9 |
| [`backend/homoiconic_codegen.h`](backend/homoiconic_codegen.md) | 16 | 11 |
| [`backend/logic_workspace_codegen.h`](backend/logic_workspace_codegen.md) | 42 | 25 |
| [`backend/map_codegen.h`](backend/map_codegen.md) | 53 | 22 |
| [`backend/memory_codegen.h`](backend/memory_codegen.md) | 87 | 38 |
| [`backend/parallel_codegen.h`](backend/parallel_codegen.md) | 60 | 19 |
| [`backend/string_io_codegen.h`](backend/string_io_codegen.md) | 71 | 54 |
| [`backend/system_codegen.h`](backend/system_codegen.md) | 255 | 243 |
| [`backend/tagged_value_codegen.h`](backend/tagged_value_codegen.md) | 49 | 42 |
| [`backend/tail_call_codegen.h`](backend/tail_call_codegen.md) | 30 | 16 |
| [`backend/tensor_backward.h`](backend/tensor_backward.md) | 17 | 17 |
| [`backend/tensor_codegen.h`](backend/tensor_codegen.md) | 190 | 173 |
| [`backend/thread_pool.h`](backend/thread_pool.md) | 59 | 48 |
| [`backend/type_system.h`](backend/type_system.md) | 76 | 21 |
| [`backend/vm.h`](backend/vm.md) | 37 | 0 |
| [`backend/vm_limits.h`](backend/vm_limits.md) | 10 | 0 |
| [`backend/work_stealing_deque.h`](backend/work_stealing_deque.md) | 80 | 37 |

### `backend/gpu/`

50/59 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`backend/gpu/gpu_memory.h`](backend/gpu/gpu_memory.md) | 59 | 50 |

### `backend/xla/`

117/178 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`backend/xla/stablehlo_emitter.h`](backend/xla/stablehlo_emitter.md) | 33 | 22 |
| [`backend/xla/xla_codegen.h`](backend/xla/xla_codegen.md) | 30 | 24 |
| [`backend/xla/xla_compiler.h`](backend/xla/xla_compiler.md) | 28 | 12 |
| [`backend/xla/xla_memory.h`](backend/xla/xla_memory.md) | 24 | 16 |
| [`backend/xla/xla_runtime.h`](backend/xla/xla_runtime.md) | 30 | 18 |
| [`backend/xla/xla_types.h`](backend/xla/xla_types.md) | 33 | 25 |

### `bridge/`

20/20 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`bridge/qllm_bridge.h`](bridge/qllm_bridge.md) | 20 | 20 |

### `core/`

208/393 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`core/bignum.h`](core/bignum.md) | 42 | 25 |
| [`core/config.h`](core/config.md) | 62 | 19 |
| [`core/eval_bridge.h`](core/eval_bridge.md) | 9 | 4 |
| [`core/execution_profile.h`](core/execution_profile.md) | 36 | 8 |
| [`core/image_io.h`](core/image_io.md) | 4 | 4 |
| [`core/inference.h`](core/inference.md) | 19 | 7 |
| [`core/introspection.h`](core/introspection.md) | 29 | 28 |
| [`core/logic.h`](core/logic.md) | 34 | 11 |
| [`core/rational.h`](core/rational.md) | 29 | 8 |
| [`core/resource_limits.h`](core/resource_limits.md) | 45 | 33 |
| [`core/runtime.h`](core/runtime.md) | 61 | 49 |
| [`core/sexp_to_ast.h`](core/sexp_to_ast.md) | 8 | 8 |
| [`core/workspace.h`](core/workspace.md) | 15 | 4 |

### `frontend/`

23/37 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`frontend/macro_expander.h`](frontend/macro_expander.md) | 37 | 23 |

### `pkg/`

5/5 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`pkg/subprocess.h`](pkg/subprocess.md) | 5 | 5 |

### `types/`

163/305 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`types/dependent.h`](types/dependent.md) | 63 | 29 |
| [`types/hott_types.h`](types/hott_types.md) | 108 | 60 |
| [`types/type_checker.h`](types/type_checker.md) | 134 | 74 |
