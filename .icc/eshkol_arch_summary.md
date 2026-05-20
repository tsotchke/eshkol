# Architecture Summary

- repo root: repository checkout root
- files indexed: `1555`
- total lines indexed: `989775`
- languages: `{"eshkol": 777, "markdown": 302, "cpp": 107, "shell": 94, "c_header": 76, "text": 73, "c": 71, "html": 11, "cmake": 10, "json": 9, "python": 8, "dockerfile": 5, "jsonl": 3, "javascript": 3, "css": 2, "objcxx": 1, "ruby": 1, "powershell": 1, "typescript": 1}`

## Important Files

- `CMakeLists.txt`
- `CONTRIBUTING.md`
- `README.md`
- `docker/cuda/Dockerfile`
- `docker/debian/debug/Dockerfile`
- `docker/debian/release/Dockerfile`
- `docker/ubuntu/release/Dockerfile`
- `docker/xla/Dockerfile`
- `docs/architecture/README.md`
- `docs/breakdown/README.md`
- `docs/components/README.md`
- `docs/development/README.md`
- `docs/platform/README.md`
- `docs/private/README.md`
- `docs/tutorials/README.md`
- `docs/vision/README.md`
- `examples/README.md`
- `tests/fuzz/CMakeLists.txt`
- `tests/fuzz/README.md`
- `tests/stress/README.md`
- `tools/icc_extras/README.md`
- `tools/vscode-eshkol/package.json`

## Project Manifests

- none detected

## Likely Launch Surfaces (Optional)

- none detected

## Public Module Roots

- `lib/backend`
- `lib/core`
- `lib/agent`
- `tools`
- `lib/math`
- `lib/quantum`
- `lib/repl`
- `lib/ml`
- `lib/types`
- `lib/frontend`
- `lib`
- `lib/signal`
- `lib/web`
- `lib/bridge`
- `lib/ffi`
- `lib/random`
- `lib/tensor`
- `lib/tsotchke`

## Integration Surfaces

- `lib/agent/c/agent_http_client.c`
- `lib/agent/c/agent_http_server.c`
- `lib/agent/http_server.esk`
- `lib/web/http.esk`
- `exe/eshkol-server.cpp`
- `inc/eshkol/bridge/qllm_bridge.h`
- `inc/eshkol/core/eval_bridge.h`
- `lib/bridge/tensor_backward.cpp`
- `lib/core/eval_bridge.cpp`
- `lib/repl/eval_bridge_impl.cpp`
- `docs/API_REFERENCE.md`
- `docs/breakdown/WEB_PLATFORM.md`
- `docs/private/ESHKOL_QLLM_INTEGRATION_SPECIFICATION.md`
- `docs/private/IDEAL_MIGRATION_STRATEGY_ANALYSIS.md`
- `docs/private/LAMBDA_SEXPR_MIGRATION_PATH.md`
- `docs/private/PHASE_2_MIGRATION_STRATEGY_COMPLETE.md`
- `docs/private/PHASE_3_MIGRATION_STATUS.md`
- `docs/private/SESSION_005_FIX_STRATEGY.md`
- `docs/private/XLA_INTEGRATION_STRATEGY.md`
- `docs/private/hott_llvm_integration.md`
- `docs/tutorials/18_WEB_PLATFORM.md`
- `inc/eshkol/http_request_utils.h`
- `lib/agent/c/agent_sqlite.c`
- `lib/agent/c/agent_watch.c`
- `lib/agent/fs-watch.esk`

## Eshkol Module Graph

- modules: `777`
- local require edges: `377`
- dependency hubs:
  stdlib (260 inbound)
  core.list.transform (15 inbound)
  core.testing (8 inbound)
  core.list.search (7 inbound)
  core.strings (7 inbound)
  core.list.query (6 inbound)
  core.list.higher_order (5 inbound)
  core.threads (4 inbound)
  core.data.base64 (3 inbound)
  core.data.csv (3 inbound)
- unresolved/external requires:
  NOESIS_ROOT/src/core/self_modify/self_modify.esk (1)
  bug_BB_xfile_lib.esk (1)
- intentional design-time requires (in examples/, won't resolve):
  tsotchke/eshkol_stdlib/qllm_ffi.esk (2)
  tsotchke/eshkol_stdlib/system_ffi.esk (2)
  tsotchke/eshkol_stdlib/selene_ffi.esk (1)
- public surface: total exports=921, top exporters: web.http(97), web.web(97), math.constants(40), core.list.compound(38), agent.terminal(34)

## Test Roots

- `tests/lists`
- `tests/v1_2_edge_cases`
- `tests/vm`
- `tests/autodiff`
- `tests/ml`
- `tests/features`
- `tests/repl`
- `tests/system`
- `tests/types`
- `tests/xla`
- `tests/gpu`
- `tests/parser`
- `tests/stdlib`
- `tests/control_flow`
- `tests/toolchain`
- `tests/typesystem`
- `tests/error_handling`
- `tests/numeric`
- `tests/parallel`
- `tests/logic`

## Top Modules

- `docs/private`: 165 files, 120969 lines, symbols=7001, tests=0, languages={"markdown": 165}
- `tests/lists`: 129 files, 6719 lines, symbols=631, tests=129, languages={"eshkol": 129}
- `examples-dep`: 102 files, 4401 lines, symbols=403, tests=0, languages={"eshkol": 101, "c": 1}
- `lib/backend`: 94 files, 146963 lines, symbols=3614, tests=0, languages={"c": 45, "cpp": 43, "c_header": 6}
- `tests/v1_2_edge_cases`: 89 files, 7555 lines, symbols=638, tests=89, languages={"eshkol": 77, "shell": 10, "python": 2}
- `scripts`: 77 files, 14467 lines, symbols=199, tests=0, languages={"shell": 75, "python": 1, "powershell": 1}
- `tests/vm`: 57 files, 3750 lines, symbols=337, tests=57, languages={"eshkol": 56, "cpp": 1}
- `.`: 53 files, 429195 lines, symbols=1699, tests=0, languages={"text": 36, "markdown": 13, "cmake": 4}
- `tests/autodiff`: 52 files, 2053 lines, symbols=162, tests=52, languages={"eshkol": 52}
- `lib/core`: 50 files, 30236 lines, symbols=1699, tests=0, languages={"cpp": 22, "eshkol": 21, "c": 6, "c_header": 1}
- `tests/ml`: 37 files, 2613 lines, symbols=114, tests=37, languages={"eshkol": 37}
- `docs/breakdown`: 36 files, 21449 lines, symbols=1442, tests=0, languages={"markdown": 36}
- `docs/tutorials`: 30 files, 4151 lines, symbols=227, tests=0, languages={"markdown": 30}
- `inc/eshkol/backend`: 29 files, 8752 lines, symbols=2437, tests=0, languages={"c_header": 29}
- `tests/features`: 24 files, 3673 lines, symbols=397, tests=24, languages={"eshkol": 24}
- `tests/repl`: 23 files, 1080 lines, symbols=64, tests=23, languages={"eshkol": 20, "cpp": 2, "text": 1}
- `lib/agent/c`: 17 files, 6959 lines, symbols=358, tests=0, languages={"c": 17}
- `docs`: 15 files, 16544 lines, symbols=1121, tests=0, languages={"markdown": 15}
- `docs/platform`: 15 files, 2603 lines, symbols=332, tests=0, languages={"markdown": 15}
- `inc/eshkol/core`: 13 files, 2360 lines, symbols=393, tests=0, languages={"c_header": 13}
- `tests/types`: 13 files, 3800 lines, symbols=314, tests=13, languages={"eshkol": 11, "cpp": 2}
- `tests/xla`: 13 files, 919 lines, symbols=143, tests=13, languages={"eshkol": 12, "cpp": 1}
- `examples`: 13 files, 1741 lines, symbols=123, tests=0, languages={"eshkol": 12, "markdown": 1}
- `tests/system`: 13 files, 570 lines, symbols=40, tests=13, languages={"eshkol": 13}
- `lib/agent`: 12 files, 1351 lines, symbols=146, tests=0, languages={"eshkol": 12}
- `tests/gpu`: 12 files, 960 lines, symbols=130, tests=12, languages={"eshkol": 12}
- `tests/parser`: 12 files, 1366 lines, symbols=98, tests=12, languages={"eshkol": 12}
- `tests/stdlib`: 12 files, 748 lines, symbols=68, tests=12, languages={"eshkol": 11, "python": 1}
- `tests/control_flow`: 10 files, 1485 lines, symbols=84, tests=10, languages={"eshkol": 10}
- `tests/toolchain`: 9 files, 713 lines, symbols=53, tests=9, languages={"cpp": 6, "shell": 2, "eshkol": 1}
- `tools/icc_extras`: 8 files, 14738 lines, symbols=615, tests=0, languages={"json": 4, "markdown": 2, "python": 2}
- `inc/eshkol`: 8 files, 2996 lines, symbols=463, tests=0, languages={"c_header": 8}
- `docs/vision`: 8 files, 5091 lines, symbols=321, tests=0, languages={"markdown": 8}
- `lib/core/list`: 8 files, 469 lines, symbols=80, tests=0, languages={"eshkol": 8}
- `tests/typesystem`: 8 files, 98 lines, symbols=10, tests=8, languages={"eshkol": 8}
- `modules_test_outputs`: 8 files, 164 lines, symbols=0, tests=0, languages={"text": 8}
- `tests/parallel`: 7 files, 275 lines, symbols=36, tests=7, languages={"eshkol": 7}
- `tests/error_handling`: 7 files, 704 lines, symbols=28, tests=7, languages={"eshkol": 7}
- `tests/numeric`: 7 files, 580 lines, symbols=12, tests=7, languages={"eshkol": 7}
- `memory_test_outputs`: 7 files, 131 lines, symbols=0, tests=0, languages={"text": 7}
