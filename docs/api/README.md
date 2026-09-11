# Eshkol API Reference

Generated from the Doxygen `/** ... */` comment blocks in the public headers under `inc/eshkol/**/*.h`. Do not edit files under `docs/api/` by hand — regenerate with:

```sh
python3 scripts/gen_api_docs.py
```

**Coverage:** 2304/5127 public symbols documented (44.9%), 2823 undocumented.

See also [INDEX.md](INDEX.md) for an alphabetical symbol table.

The reviewed DD-11 consumer-facing subset is tracked in [public_surface.md](public_surface.md); the coverage gate verifies every entry has a source, page anchor, and index link.

## Subsystems

### (root headers)

332/855 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`abi_fingerprint.h`](abi_fingerprint.md) | 58 | 14 |
| [`agent_capabilities.h`](agent_capabilities.md) | 37 | 0 |
| [`agent_http.h`](agent_http.md) | 27 | 0 |
| [`agent_platform.h`](agent_platform.md) | 2 | 0 |
| [`builtin_libraries.h`](builtin_libraries.md) | 3 | 1 |
| [`eshkol.h`](eshkol.md) | 365 | 167 |
| [`eshkol_ffi.h`](eshkol_ffi.md) | 44 | 32 |
| [`exhaustive_dispatch.h`](exhaustive_dispatch.md) | 2 | 0 |
| [`http_request_utils.h`](http_request_utils.md) | 6 | 3 |
| [`llvm_backend.h`](llvm_backend.md) | 83 | 3 |
| [`logger.h`](logger.md) | 41 | 25 |
| [`memory_abi_v2.h`](memory_abi_v2.md) | 33 | 15 |
| [`model_io.h`](model_io.md) | 8 | 5 |
| [`module_resolver.h`](module_resolver.md) | 1 | 0 |
| [`module_visibility.h`](module_visibility.md) | 1 | 0 |
| [`platform_runtime.h`](platform_runtime.md) | 51 | 38 |
| [`runtime_exports.h`](runtime_exports.md) | 48 | 29 |
| [`tensor_cross_entropy.h`](tensor_cross_entropy.md) | 4 | 0 |
| [`tensor_validation.h`](tensor_validation.md) | 4 | 0 |
| [`tensorcore_adapter.h`](tensorcore_adapter.md) | 37 | 0 |

### `backend/`

1257/2925 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`backend/arithmetic_codegen.h`](backend/arithmetic_codegen.md) | 54 | 47 |
| [`backend/autodiff_codegen.h`](backend/autodiff_codegen.md) | 185 | 118 |
| [`backend/binding_codegen.h`](backend/binding_codegen.md) | 71 | 22 |
| [`backend/blas_backend.h`](backend/blas_backend.md) | 23 | 23 |
| [`backend/builtin_declarations.h`](backend/builtin_declarations.md) | 16 | 6 |
| [`backend/call_apply_codegen.h`](backend/call_apply_codegen.md) | 54 | 22 |
| [`backend/cblas_compat.h`](backend/cblas_compat.md) | 7 | 0 |
| [`backend/codegen_context.h`](backend/codegen_context.md) | 178 | 43 |
| [`backend/collection_codegen.h`](backend/collection_codegen.md) | 32 | 20 |
| [`backend/complex_codegen.h`](backend/complex_codegen.md) | 34 | 23 |
| [`backend/control_flow_codegen.h`](backend/control_flow_codegen.md) | 32 | 12 |
| [`backend/cpu_features.h`](backend/cpu_features.md) | 49 | 22 |
| [`backend/differential_form_core.h`](backend/differential_form_core.md) | 17 | 12 |
| [`backend/frechet_mean_core.h`](backend/frechet_mean_core.md) | 8 | 4 |
| [`backend/function_cache.h`](backend/function_cache.md) | 30 | 11 |
| [`backend/function_codegen.h`](backend/function_codegen.md) | 20 | 8 |
| [`backend/hash_codegen.h`](backend/hash_codegen.md) | 38 | 9 |
| [`backend/homoiconic_codegen.h`](backend/homoiconic_codegen.md) | 16 | 11 |
| [`backend/ir_builder.h`](backend/ir_builder.md) | 10 | 0 |
| [`backend/link_probe.h`](backend/link_probe.md) | 1 | 1 |
| [`backend/llvm_codegen.h`](backend/llvm_codegen.md) | 736 | 17 |
| [`backend/llvm_compat.h`](backend/llvm_compat.md) | 5 | 0 |
| [`backend/logic_workspace_codegen.h`](backend/logic_workspace_codegen.md) | 42 | 25 |
| [`backend/map_codegen.h`](backend/map_codegen.md) | 54 | 23 |
| [`backend/memory_codegen.h`](backend/memory_codegen.md) | 94 | 41 |
| [`backend/mutation_observation.h`](backend/mutation_observation.md) | 10 | 0 |
| [`backend/parallel_codegen.h`](backend/parallel_codegen.md) | 61 | 19 |
| [`backend/qllm_backward.h`](backend/qllm_backward.md) | 9 | 0 |
| [`backend/riemannian_core.h`](backend/riemannian_core.md) | 87 | 26 |
| [`backend/string_io_codegen.h`](backend/string_io_codegen.md) | 74 | 56 |
| [`backend/system_codegen.h`](backend/system_codegen.md) | 277 | 265 |
| [`backend/tagged_value_codegen.h`](backend/tagged_value_codegen.md) | 50 | 42 |
| [`backend/tail_call_codegen.h`](backend/tail_call_codegen.md) | 30 | 16 |
| [`backend/tensor_backward.h`](backend/tensor_backward.md) | 22 | 22 |
| [`backend/tensor_codegen.h`](backend/tensor_codegen.md) | 204 | 184 |
| [`backend/tensorcore_codegen.h`](backend/tensorcore_codegen.md) | 7 | 0 |
| [`backend/thread_pool.h`](backend/thread_pool.md) | 59 | 48 |
| [`backend/type_system.h`](backend/type_system.md) | 78 | 21 |
| [`backend/vm.h`](backend/vm.md) | 47 | 1 |
| [`backend/vm_limits.h`](backend/vm_limits.md) | 24 | 0 |
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

34/34 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`bridge/qllm_bridge.h`](bridge/qllm_bridge.md) | 25 | 25 |
| [`bridge/space_form.h`](bridge/space_form.md) | 9 | 9 |

### `core/`

287/560 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`core/arity_contract.h`](core/arity_contract.md) | 3 | 3 |
| [`core/ast_routing.h`](core/ast_routing.md) | 3 | 0 |
| [`core/bignum.h`](core/bignum.md) | 43 | 26 |
| [`core/complex_math.h`](core/complex_math.md) | 33 | 30 |
| [`core/config.h`](core/config.md) | 62 | 19 |
| [`core/dtoa_shortest.h`](core/dtoa_shortest.md) | 1 | 1 |
| [`core/eval_bridge.h`](core/eval_bridge.md) | 9 | 4 |
| [`core/event_loop.h`](core/event_loop.md) | 25 | 23 |
| [`core/execution_profile.h`](core/execution_profile.md) | 36 | 8 |
| [`core/i128.h`](core/i128.md) | 18 | 0 |
| [`core/i128_runtime.h`](core/i128_runtime.md) | 12 | 0 |
| [`core/image_io.h`](core/image_io.md) | 4 | 4 |
| [`core/inference.h`](core/inference.md) | 19 | 7 |
| [`core/introspection.h`](core/introspection.md) | 30 | 29 |
| [`core/linear_solve.h`](core/linear_solve.md) | 9 | 4 |
| [`core/logic.h`](core/logic.md) | 37 | 11 |
| [`core/object_limits.h`](core/object_limits.md) | 4 | 0 |
| [`core/rational.h`](core/rational.md) | 36 | 10 |
| [`core/resource_limits.h`](core/resource_limits.md) | 65 | 40 |
| [`core/runtime.h`](core/runtime.md) | 64 | 52 |
| [`core/sexp_to_ast.h`](core/sexp_to_ast.md) | 8 | 8 |
| [`core/string_escape.h`](core/string_escape.md) | 3 | 0 |
| [`core/symbol_syntax.h`](core/symbol_syntax.md) | 16 | 4 |
| [`core/unicode.h`](core/unicode.md) | 5 | 0 |
| [`core/workspace.h`](core/workspace.md) | 15 | 4 |

### `frontend/`

37/163 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`frontend/binding_forms.h`](frontend/binding_forms.md) | 3 | 0 |
| [`frontend/diagnostic.h`](frontend/diagnostic.md) | 8 | 0 |
| [`frontend/macro_expander.h`](frontend/macro_expander.md) | 42 | 23 |
| [`frontend/node_identity.h`](frontend/node_identity.md) | 14 | 14 |
| [`frontend/semantic_identity.h`](frontend/semantic_identity.md) | 75 | 0 |
| [`frontend/workspace.h`](frontend/workspace.md) | 21 | 0 |

### `pkg/`

5/5 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`pkg/subprocess.h`](pkg/subprocess.md) | 5 | 5 |

### `types/`

185/344 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`types/dependent.h`](types/dependent.md) | 63 | 29 |
| [`types/hott_types.h`](types/hott_types.md) | 121 | 68 |
| [`types/type_checker.h`](types/type_checker.md) | 160 | 88 |

### `util/`

0/4 symbols documented.

| Header | Symbols | Documented |
|---|---:|---:|
| [`util/continuation_task.h`](util/continuation_task.md) | 4 | 0 |
