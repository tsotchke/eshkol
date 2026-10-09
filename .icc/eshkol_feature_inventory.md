# Feature Inventory: `eshkol`

Clusters: `1268`  Coverage: `{"adequate": 377, "deep": 451, "mentioned": 21, "none": 168, "shallow": 251}`

| Coverage | Cluster | Symbols | Refs | Docs | Home | Sample |
|---|---|---:|---:|---:|---|---|
| `none` | `deps/pjrt/PJRT` | 210 | 0 | 0 |  | `PJRT_Api`, `PJRT_API_MAJOR`, `PJRT_API_MINOR`, `PJRT_Api_Version`, `PJRT_AsyncHostToDeviceTransferManager`, `PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args`, `PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args`, `PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args` |
| `none` | `lib/backend/eskb` | 19 | 0 | 0 |  | `eskb_buf_ensure`, `eskb_buf_free`, `eskb_buf_init`, `eskb_buf_write`, `eskb_buf_write_f64`, `eskb_buf_write_i64`, `eskb_buf_write_leb128`, `eskb_buf_write_string` |
| `none` | `lib/backend/gpu/F32S` | 16 | 0 | 0 |  | `F32S_BK`, `F32S_BM`, `F32S_BN`, `F32S_THREADS`, `F32S_TM`, `F32S_TN`, `F32S_WM`, `F32S_WN` |
| `none` | `lib/backend/gpu/F32S128` | 16 | 0 | 0 |  | `F32S128_BK`, `F32S128_BM`, `F32S128_BN`, `F32S128_THREADS`, `F32S128_TM`, `F32S128_TN`, `F32S128_WM`, `F32S128_WN` |
| `none` | `lib/agent/c/eshkol` | 14 | 0 | 0 |  | `eshkol_agent_mutex_lock`, `eshkol_agent_mutex_unlock`, `eshkol_http_is_token_char`, `eshkol_http_valid_field_value`, `eshkol_http_valid_header_line`, `eshkol_http_valid_method`, `eshkol_http_valid_token`, `eshkol_sse_parser` |
| `none` | `lib/backend/gpu/cuda` | 13 | 0 | 0 |  | `cuda_batchnorm_backward_f64`, `cuda_conv2d_backward_input_f64`, `cuda_conv2d_backward_kernel_f64`, `cuda_launch_elementwise_f64`, `cuda_launch_normalize_f64`, `cuda_launch_reduce_axis_f64`, `cuda_launch_reduce_f64`, `cuda_launch_softmax_f64` |
| `none` | `inc/eshkol/backend/xla/ESHKOL` | 12 | 0 | 0 |  | `ESHKOL_STABLEHLO_EMITTER_H`, `ESHKOL_XLA_CODEGEN_H`, `ESHKOL_XLA_COMPILER_H`, `ESHKOL_XLA_MEMORY_H`, `ESHKOL_XLA_RUNTIME_H`, `ESHKOL_XLA_TYPES_H` |
| `none` | `lib/backend/gpu/DF64` | 12 | 0 | 0 |  | `DF64_BK`, `DF64_BM`, `DF64_BN`, `DF64_TG`, `DF64_THREADS`, `DF64_TT` |
| `none` | `lib/backend/gpu/FP53` | 12 | 0 | 0 |  | `FP53_BK`, `FP53_BM`, `FP53_BN`, `FP53_SB_STRIDE`, `FP53_THREADS`, `FP53_TT` |
| `none` | `lib/agent/c/ESHKOL` | 10 | 0 | 0 |  | `ESHKOL_AGENT_HTTP_INTERNAL_H`, `ESHKOL_AGENT_MUTEX_INITIALIZER`, `ESHKOL_AGENT_NATIVE_MUTEX_H`, `ESHKOL_AGENT_SSE_INTERNAL_H` |
| `none` | `lib/backend/gpu/FP` | 10 | 0 | 0 |  | `FP_BK`, `FP_BM`, `FP_BN`, `FP_THREADS`, `FP_TT` |
| `none` | `lib/repl/ESHKOL` | 10 | 0 | 0 |  | `ESHKOL_JIT_COFF_MEMORY_MANAGER_H`, `ESHKOL_JIT_TARGET_CONFIG_H`, `ESHKOL_JITLINK_BRANCH26_RANGE_EXTENSION_H`, `ESHKOL_REPL_JIT_H`, `ESHKOL_REPL_UTILS_H` |
| `none` | `tests/core` | 10 | 0 | 0 |  | `__asan_default_options`, `__wrap_arena_allocate_aligned`, `__wrap_arena_allocate_closure_with_header`, `__wrap_arena_allocate_cons_with_header`, `__wrap_arena_allocate_vector_with_header`, `__wrap_calloc`, `__wrap_malloc` |
| `none` | `inc/eshkol/types/ESHKOL` | 8 | 0 | 0 |  | `ESHKOL_TYPE_CHECKER_H`, `ESHKOL_TYPES_DEPENDENT_H`, `ESHKOL_TYPES_HOTT_TYPES_H`, `ESHKOL_TYPES_TYPE_RELATION_H` |
| `none` | `lib/core/SDNC` | 8 | 0 | 0 |  | `SDNC_D`, `SDNC_FFN_DIM`, `SDNC_HD`, `SDNC_N_LAYERS` |
| `none` | `lib/test/modules` | 8 | 0 | 0 |  | `test.modules.dd10_loaded_private`, `test.modules.dd10_loaded_private_ad`, `test.modules.dd10_privacy_scope`, `test.modules.dd10_public_hidden`, `test.modules.internal_forward_helper`, `test.modules.math_utils`, `test.modules.mod_a`, `test.modules.mod_b` |
| `none` | `lib/repl/bright` | 7 | 0 | 0 |  | `bright_black`, `bright_blue`, `bright_cyan`, `bright_green`, `bright_magenta`, `bright_red`, `bright_yellow` |
| `none` | `lib/repl/get` | 7 | 0 | 0 |  | `get_ast_type_string`, `get_builtin_symbols`, `get_function_docs`, `get_history_file_path`, `get_op_name`, `get_repl_commands`, `get_type_name` |
| `none` | `lib/backend/ESHKOL` | 6 | 0 | 0 |  | `ESHKOL_LIB_BACKEND_SDNC_ISA_H`, `ESHKOL_TENSOR_CONV_KERNEL_H`, `ESHKOL_VM_PRELUDE_SOURCE_H` |
| `none` | `lib/repl/print` | 6 | 0 | 0 |  | `print_doc`, `print_doc_topics`, `print_examples`, `print_info`, `print_success`, `print_welcome_banner` |
| `none` | `lib/test/modules/loaded` | 6 | 0 | 0 |  | `loaded-private`, `loaded-private-derivative-n`, `loaded-private-let-values`, `loaded-private-parameter`, `loaded-private-taylor-ok`, `loaded-private-the` |
| `none` | `lib/math/fixed_point/ESK` | 5 | 0 | 0 |  | `ESK_OF_SATURATE`, `ESK_OF_WRAP`, `ESK_ROUND_NEAREST_EVEN`, `ESK_ROUND_STOCHASTIC`, `ESK_ROUND_TRUNCATE` |
| `none` | `tests/core/constructor` | 5 | 0 | 0 |  | `constructor_test_arm`, `constructor_test_caught`, `constructor_test_finish`, `constructor_test_reused_list`, `constructor_test_step` |
| `none` | `inc/eshkol/backend/xla/REDUCE` | 4 | 0 | 0 |  | `REDUCE_MAX`, `REDUCE_MIN`, `REDUCE_PROD`, `REDUCE_SUM` |
| `none` | `lib/backend/gpu/convert` | 4 | 0 | 0 |  | `convert_df64_to_f64`, `convert_f32_to_f64`, `convert_f64_to_df64`, `convert_f64_to_f32` |
| `none` | `lib/core/debug` | 4 | 0 | 0 |  | `debug_print_ad_mode`, `debug_print_ptr` |
| `none` | `lib/frontend/ESHKOL` | 4 | 0 | 0 |  | `ESHKOL_FRONTEND_LIBRARY_REGISTRY_H`, `ESHKOL_FRONTEND_PARSER_TASK_H` |
| `none` | `lib/quantum/QUANTUM` | 4 | 0 | 0 |  | `QUANTUM_RNG_H`, `QUANTUM_RNG_WRAPPER_H` |
| `none` | `tests/fixed_point/CHECK` | 4 | 0 | 0 |  | `CHECK`, `CHECK_EQ_I128` |
| `none` | `tests/toolchain/fake_cuda/include/CUBLAS` | 4 | 0 | 0 |  | `CUBLAS_STATIC`, `CUBLAS_VER_MAJOR` |
| `none` | `tests/toolchain/fake_cuda/include/CUBLASWINAPI` | 4 | 0 | 0 |  | `CUBLASWINAPI` |
| `none` | `lib/backend/FlushLanguageCoverage` | 3 | 0 | 0 |  | `FlushLanguageCoverage` |
| `none` | `lib/backend/~FlushLanguageCoverage` | 3 | 0 | 0 |  | `~FlushLanguageCoverage` |
| `none` | `lib/core/sigaction` | 3 | 0 | 0 |  | `sigaction` |
| `none` | `tests/v1_2_edge_cases/fixtures` | 3 | 0 | 0 |  | `tests.v1_2_edge_cases.fixtures.named_let_repl_global_capture_dim`, `tests.v1_2_edge_cases.fixtures.named_let_repl_global_capture_use`, `tests.v1_2_edge_cases.fixtures.repl_loaded_macro_no_return` |
| `none` | `deps/pjrt/XLA` | 2 | 0 | 0 |  | `XLA_PJRT_C_PJRT_C_API_H_` |
| `none` | `inc/eshkol/pkg/ESHKOL` | 2 | 0 | 0 |  | `ESHKOL_PKG_SUBPROCESS_H` |
| `none` | `inc/eshkol/pkg/NOMINMAX` | 2 | 0 | 0 |  | `NOMINMAX` |
| `none` | `inc/eshkol/util/ESHKOL` | 2 | 0 | 0 |  | `ESHKOL_UTIL_CONTINUATION_TASK_H` |
| `none` | `lib/agent/c/WIN32` | 2 | 0 | 0 |  | `WIN32_LEAN_AND_MEAN` |
| `none` | `lib/backend/AD` | 2 | 0 | 0 |  | `AD_NODE_FIELDS` |
| `none` | `lib/backend/VmArenaBlock` | 2 | 0 | 0 |  | `VmArenaBlock` |
| `none` | `lib/backend/VmBignum` | 2 | 0 | 0 |  | `VmBignum` |
| `none` | `lib/backend/VmRegion` | 2 | 0 | 0 |  | `VmRegion` |
| `none` | `lib/backend/gpu/add64` | 2 | 0 | 0 |  | `add64`, `add64_carry` |
| `none` | `lib/backend/gpu/conv2d` | 2 | 0 | 0 |  | `conv2d_backward_input_sf64`, `conv2d_backward_kernel_sf64` |
| `none` | `lib/backend/gpu/f32` | 2 | 0 | 0 |  | `f32_to_native_f64`, `f32_to_sf64` |
| `none` | `lib/backend/gpu/ozfast` | 2 | 0 | 0 |  | `ozfast_df_add`, `ozfast_df_mul` |
| `none` | `lib/math/fixed_point/ESHKOL` | 2 | 0 | 0 |  | `ESHKOL_FIXED_POINT_H` |
| `none` | `tests/bridge/ESHKOL` | 2 | 0 | 0 |  | `ESHKOL_GEOMETRIC_BOUNDARY_GOLDEN_H` |
| `none` | `tests/fixed_point/ESHKOL` | 2 | 0 | 0 |  | `ESHKOL_FP_TEST_HARNESS_H` |
| `none` | `tests/tensor/module` | 2 | 0 | 0 |  | `module-embedding-lookup`, `module-embedding-row-sum` |
| `none` | `tests/toolchain/fake_cuda/include/CUBLASAPI` | 2 | 0 | 0 |  | `CUBLASAPI` |
| `none` | `tests/toolchain/fake_cuda/include/cublasGemmStridedBatchedEx` | 2 | 0 | 0 |  | `cublasGemmStridedBatchedEx` |
| `none` | `tests/v1_2_edge_cases` | 2 | 0 | 0 |  | `tests.v1_2_edge_cases.bug_ee_statement_parser_lib`, `tests.v1_2_edge_cases.cross_file_symbol_eq_module` |
| `none` | `tests/v1_2_edge_cases/check` | 2 | 0 | 0 |  | `check-eq-with-caller-symbol`, `check-sigma` |
| `none` | `tests/v1_2_edge_cases/make` | 2 | 0 | 0 |  | `make-episode-vec`, `make-sigma-literal` |
| `none` | `deps/pjrt` | 1 | 0 | 0 |  | `_PJRT_API_STRUCT_FIELD` |
| `none` | `inc/eshkol/backend/AVX512` | 1 | 0 | 0 |  | `AVX512` |
| `none` | `inc/eshkol/backend/FunctionContext` | 1 | 0 | 0 |  | `FunctionContext` |
| `none` | `inc/eshkol/backend/SharedLibraryExportAbi` | 1 | 0 | 0 |  | `SharedLibraryExportAbi` |
| `none` | `inc/eshkol/backend/xla/BROADCAST` | 1 | 0 | 0 |  | `BROADCAST` |
| `none` | `inc/eshkol/backend/xla/CONCATENATE` | 1 | 0 | 0 |  | `CONCATENATE` |
| `none` | `inc/eshkol/backend/xla/DEVICE` | 1 | 0 | 0 |  | `DEVICE_ALLOC` |
| `none` | `inc/eshkol/backend/xla/DIVIDE` | 1 | 0 | 0 |  | `DIVIDE` |
| `none` | `inc/eshkol/backend/xla/F16` | 1 | 0 | 0 |  | `F16` |
| `none` | `inc/eshkol/backend/xla/I16` | 1 | 0 | 0 |  | `I16` |
| `none` | `inc/eshkol/backend/xla/I32` | 1 | 0 | 0 |  | `I32` |
| `none` | `inc/eshkol/backend/xla/I64` | 1 | 0 | 0 |  | `I64` |
| `none` | `inc/eshkol/backend/xla/MULTIPLY` | 1 | 0 | 0 |  | `MULTIPLY` |
| `none` | `inc/eshkol/backend/xla/NEGATE` | 1 | 0 | 0 |  | `NEGATE` |
| `none` | `inc/eshkol/backend/xla/SLICE` | 1 | 0 | 0 |  | `SLICE` |
| `none` | `inc/eshkol/backend/xla/SUBTRACT` | 1 | 0 | 0 |  | `SUBTRACT` |
| `none` | `inc/eshkol/backend/xla/U16` | 1 | 0 | 0 |  | `U16` |
| `none` | `inc/eshkol/backend/xla/U32` | 1 | 0 | 0 |  | `U32` |
| `none` | `inc/eshkol/backend/xla/U64` | 1 | 0 | 0 |  | `U64` |
| `none` | `inc/eshkol/backend/xla/ZERO` | 1 | 0 | 0 |  | `ZERO_COPY` |
| `none` | `inc/eshkol/core/ast` | 1 | 0 | 0 |  | `ast_routing_detail` |
| `none` | `inc/eshkol/frontend/MacroBinding` | 1 | 0 | 0 |  | `MacroBinding` |
| `none` | `inc/eshkol/pkg/process` | 1 | 0 | 0 |  | `process_info` |
| `none` | `inc/eshkol/pkg/startup` | 1 | 0 | 0 |  | `startup_info` |
| `none` | `inc/eshkol/pkg/timespec` | 1 | 0 | 0 |  | `timespec` |
| `none` | `inc/eshkol/types/BorrowInfo` | 1 | 0 | 0 |  | `BorrowInfo` |
| `none` | `inc/eshkol/types/CompareResult` | 1 | 0 | 0 |  | `CompareResult` |
| `none` | `inc/eshkol/types/ISize` | 1 | 0 | 0 |  | `ISize` |
| `none` | `inc/eshkol/types/Int16` | 1 | 0 | 0 |  | `Int16` |
| `none` | `inc/eshkol/types/Int32` | 1 | 0 | 0 |  | `Int32` |
| `none` | `inc/eshkol/types/TypeFlags` | 1 | 0 | 0 |  | `TypeFlags` |
| `none` | `inc/eshkol/types/TypeU1` | 1 | 0 | 0 |  | `TypeU1` |
| `none` | `inc/eshkol/types/TypeU2` | 1 | 0 | 0 |  | `TypeU2` |
| `none` | `inc/eshkol/types/UInt16` | 1 | 0 | 0 |  | `UInt16` |
| `none` | `inc/eshkol/types/UInt32` | 1 | 0 | 0 |  | `UInt32` |
| `none` | `inc/eshkol/types/UInt64` | 1 | 0 | 0 |  | `UInt64` |
| `none` | `inc/eshkol/types/UInt8` | 1 | 0 | 0 |  | `UInt8` |
| `none` | `inc/eshkol/types/USize` | 1 | 0 | 0 |  | `USize` |
| `none` | `inc/eshkol/util/await` | 1 | 0 | 0 |  | `await_resume` |
| `none` | `inc/eshkol/util/get` | 1 | 0 | 0 |  | `get_return_object` |
| `none` | `inc/eshkol/util/promise` | 1 | 0 | 0 |  | `promise_type` |
| `none` | `inc/eshkol/util/return` | 1 | 0 | 0 |  | `return_value` |
| `none` | `inc/eshkol/util/unhandled` | 1 | 0 | 0 |  | `unhandled_exception` |
