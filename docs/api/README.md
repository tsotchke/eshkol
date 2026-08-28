# Eshkol API Reference

Generated from the Doxygen `/** ... */` comment blocks in the public headers under `inc/eshkol/**/*.h`. Do not edit files under `docs/api/` by hand — regenerate with:

```sh
python3 scripts/gen_api_docs.py
```

**Coverage:** 2190/3955 public symbols documented (55.4%), 1765 undocumented.

See also [INDEX.md](INDEX.md) for an alphabetical symbol table.

## Subsystems

### (root headers)

315/768 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`abi_fingerprint.h`](abi_fingerprint.md) | 28 | 13 |
| [`agent_capabilities.h`](agent_capabilities.md) | 37 | 0 |
| [`agent_http.h`](agent_http.md) | 27 | 0 |
| [`eshkol.h`](eshkol.md) | 330 | 156 |
| [`eshkol_ffi.h`](eshkol_ffi.md) | 44 | 32 |
| [`exhaustive_dispatch.h`](exhaustive_dispatch.md) | 2 | 0 |
| [`http_request_utils.h`](http_request_utils.md) | 6 | 3 |
| [`llvm_backend.h`](llvm_backend.md) | 83 | 3 |
| [`logger.h`](logger.md) | 38 | 22 |
| [`memory_abi_v2.h`](memory_abi_v2.md) | 33 | 15 |
| [`model_io.h`](model_io.md) | 8 | 5 |
| [`platform_runtime.h`](platform_runtime.md) | 51 | 38 |
| [`runtime_exports.h`](runtime_exports.md) | 47 | 28 |
| [`tensorcore_adapter.h`](tensorcore_adapter.md) | 34 | 0 |

### `backend/`

1178/1990 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`backend/arithmetic_codegen.h`](backend/arithmetic_codegen.md) | 49 | 42 |
| [`backend/autodiff_codegen.h`](backend/autodiff_codegen.md) | 180 | 115 |
| [`backend/binding_codegen.h`](backend/binding_codegen.md) | 57 | 21 |
| [`backend/blas_backend.h`](backend/blas_backend.md) | 23 | 23 |
| [`backend/builtin_declarations.h`](backend/builtin_declarations.md) | 16 | 6 |
| [`backend/call_apply_codegen.h`](backend/call_apply_codegen.md) | 49 | 22 |
| [`backend/cblas_compat.h`](backend/cblas_compat.md) | 7 | 0 |
| [`backend/codegen_context.h`](backend/codegen_context.md) | 174 | 43 |
| [`backend/collection_codegen.h`](backend/collection_codegen.md) | 32 | 20 |
| [`backend/complex_codegen.h`](backend/complex_codegen.md) | 34 | 23 |
| [`backend/control_flow_codegen.h`](backend/control_flow_codegen.md) | 32 | 12 |
| [`backend/cpu_features.h`](backend/cpu_features.md) | 49 | 22 |
| [`backend/frechet_mean_core.h`](backend/frechet_mean_core.md) | 8 | 4 |
| [`backend/function_cache.h`](backend/function_cache.md) | 30 | 11 |
| [`backend/function_codegen.h`](backend/function_codegen.md) | 20 | 8 |
| [`backend/hash_codegen.h`](backend/hash_codegen.md) | 38 | 9 |
| [`backend/homoiconic_codegen.h`](backend/homoiconic_codegen.md) | 16 | 11 |
| [`backend/logic_workspace_codegen.h`](backend/logic_workspace_codegen.md) | 42 | 25 |
| [`backend/map_codegen.h`](backend/map_codegen.md) | 53 | 22 |
| [`backend/memory_codegen.h`](backend/memory_codegen.md) | 88 | 38 |
| [`backend/parallel_codegen.h`](backend/parallel_codegen.md) | 60 | 19 |
| [`backend/qllm_backward.h`](backend/qllm_backward.md) | 9 | 0 |
| [`backend/string_io_codegen.h`](backend/string_io_codegen.md) | 74 | 56 |
| [`backend/system_codegen.h`](backend/system_codegen.md) | 274 | 262 |
| [`backend/tagged_value_codegen.h`](backend/tagged_value_codegen.md) | 49 | 42 |
| [`backend/tail_call_codegen.h`](backend/tail_call_codegen.md) | 30 | 16 |
| [`backend/tensor_backward.h`](backend/tensor_backward.md) | 19 | 19 |
| [`backend/tensor_codegen.h`](backend/tensor_codegen.md) | 196 | 180 |
| [`backend/tensorcore_codegen.h`](backend/tensorcore_codegen.md) | 7 | 0 |
| [`backend/thread_pool.h`](backend/thread_pool.md) | 59 | 48 |
| [`backend/type_system.h`](backend/type_system.md) | 76 | 21 |
| [`backend/vm.h`](backend/vm.md) | 45 | 1 |
| [`backend/vm_limits.h`](backend/vm_limits.md) | 15 | 0 |
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

22/22 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`bridge/qllm_bridge.h`](bridge/qllm_bridge.md) | 22 | 22 |

### `core/`

281/535 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`core/bignum.h`](core/bignum.md) | 43 | 26 |
| [`core/complex_math.h`](core/complex_math.md) | 33 | 30 |
| [`core/config.h`](core/config.md) | 62 | 19 |
| [`core/dtoa_shortest.h`](core/dtoa_shortest.md) | 1 | 1 |
| [`core/eval_bridge.h`](core/eval_bridge.md) | 9 | 4 |
| [`core/event_loop.h`](core/event_loop.md) | 25 | 23 |
| [`core/execution_profile.h`](core/execution_profile.md) | 36 | 8 |
| [`core/i128.h`](core/i128.md) | 16 | 0 |
| [`core/i128_runtime.h`](core/i128_runtime.md) | 11 | 0 |
| [`core/image_io.h`](core/image_io.md) | 4 | 4 |
| [`core/inference.h`](core/inference.md) | 19 | 7 |
| [`core/introspection.h`](core/introspection.md) | 30 | 29 |
| [`core/linear_solve.h`](core/linear_solve.md) | 9 | 4 |
| [`core/logic.h`](core/logic.md) | 37 | 11 |
| [`core/rational.h`](core/rational.md) | 35 | 10 |
| [`core/resource_limits.h`](core/resource_limits.md) | 64 | 39 |
| [`core/runtime.h`](core/runtime.md) | 62 | 50 |
| [`core/sexp_to_ast.h`](core/sexp_to_ast.md) | 8 | 8 |
| [`core/symbol_syntax.h`](core/symbol_syntax.md) | 16 | 4 |
| [`core/workspace.h`](core/workspace.md) | 15 | 4 |

### `frontend/`

37/56 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`frontend/macro_expander.h`](frontend/macro_expander.md) | 42 | 23 |
| [`frontend/node_identity.h`](frontend/node_identity.md) | 14 | 14 |

### `pkg/`

5/5 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`pkg/subprocess.h`](pkg/subprocess.md) | 5 | 5 |

### `types/`

185/342 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`types/dependent.h`](types/dependent.md) | 63 | 29 |
| [`types/hott_types.h`](types/hott_types.md) | 121 | 68 |
| [`types/type_checker.h`](types/type_checker.md) | 158 | 88 |
