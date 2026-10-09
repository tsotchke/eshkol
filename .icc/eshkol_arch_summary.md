# Architecture Summary

- repo root: repository checkout root
- files indexed: `4304`
- total lines indexed: `1043318`
- languages: `{"eshkol": 2462, "markdown": 505, "shell": 265, "cpp": 263, "python": 190, "c_header": 157, "c": 115, "text": 113, "html": 92, "json": 65, "javascript": 21, "document": 18, "cmake": 17, "dockerfile": 6, "typescript": 5, "css": 2, "makefile": 1, "wgsl": 1, "objc": 1, "objcxx": 1, "ruby": 1, "powershell": 1, "toml": 1, "jsonl": 1}`

## Important Files

- `CMakeLists.txt`
- `CONTRIBUTING.md`
- `Makefile`
- `README.md`
- `bench/README.md`
- `docker/cuda/Dockerfile`
- `docker/debian/debug/Dockerfile`
- `docker/debian/release/Dockerfile`
- `docker/ubuntu/release/Dockerfile`
- `docker/xla/Dockerfile`
- `docs/README.md`
- `docs/api/README.md`
- `docs/architecture/README.md`
- `docs/breakdown/README.md`
- `docs/components/README.md`
- `docs/design/adr/README.md`
- `docs/development/README.md`
- `docs/platform/README.md`
- `docs/reports/README.md`
- `docs/tutorials/README.md`
- `docs/vision/README.md`
- `examples/README.md`
- `examples/wgsl_artifact/README.md`
- `nix/jetson/README.md`
- `scripts/mesh/runner/Dockerfile`

## Project Manifests

- none detected

## Likely Launch Surfaces (Optional)

- none detected

## Public Module Roots

- `lib/core`
- `lib/backend`
- `lib/agent`
- `tools`
- `lib/frontend`
- `lib/repl`
- `lib/math`
- `lib/bridge`
- `lib/types`
- `lib/quantum`
- `lib`
- `lib/ml`
- `lib/signal`
- `lib/ffi`
- `lib/random`
- `lib/tensor`
- `lib/web`

## Integration Surfaces

- `docs/api/bridge/qllm_bridge.md`
- `docs/api/bridge/space_form.md`
- `docs/api/core/eval_bridge.md`
- `docs/reference/stdlib/http_server.md`
- `lib/agent/c/agent_http_client.c`
- `lib/agent/c/agent_http_client_apple.m`
- `lib/agent/c/agent_http_client_winhttp.c`
- `lib/agent/c/agent_http_server.c`
- `lib/agent/http_server.esk`
- `lib/core/http_server.esk`
- `docs/api/agent_http.md`
- `docs/api/http_request_utils.md`
- `exe/eshkol-server.cpp`
- `inc/eshkol/bridge/qllm_bridge.h`
- `inc/eshkol/bridge/space_form.h`
- `inc/eshkol/core/eval_bridge.h`
- `lib/bridge/qllm_bridge.cpp`
- `lib/bridge/qllm_interop.cpp`
- `lib/bridge/space_form_ad.cpp`
- `lib/bridge/tensor_backward.cpp`
- `lib/bridge/tensorcore_adapter.cpp`
- `lib/core/eval_bridge.cpp`
- `lib/repl/eval_bridge_impl.cpp`
- `lib/types/linear_check_bridge.cpp`
- `tools/erepl_client.py`

## Eshkol Module Graph

- modules: `2462`
- local require edges: `899`
- dependency hubs:
  stdlib (655 inbound)
  core.list.transform (18 inbound)
  core.strings (13 inbound)
  core.testing (13 inbound)
  agent.quantum (10 inbound)
  core.capabilities (9 inbound)
  core.list.search (9 inbound)
  core.list.query (8 inbound)
  core.threads (7 inbound)
  core.list.higher_order (6 inbound)
- unresolved/external requires:
  scheme (6)
  base (6)
  qllm_oracle_lib.esk (6)
  prefix (5)
  only (4)
  test (3)
  modules (3)
  rename (3)
  except (3)
  mod_b (2)
- public surface: total exports=1437, top exporters: web.web(97), core.distributed(68), core.ad.tape(49), core.blc(47), core.dbsp(40)

## Test Roots

- `tests/edge_matrix/generated`
- `tests/vm_parity/corpus`
- `tests/lists`
- `tests/typesystem`
- `tests/v1_2_edge_cases`
- `tests/toolchain`
- `tests/vm`
- `tests/ad`
- `tests/core`
- `tests/recursion_depth/generated`
- `tests/memory`
- `tests/autodiff`
- `tests/differential/corpus`
- `tests/ml`
- `tests/ad_oracle/generated`
- `tests/features`
- `tests/sicp`
- `tests/error_handling/guard_coverage`
- `tests/reference-diff/corpus`
- `tests/stress`

## Top Modules

- `tests/edge_matrix/generated`: 289 files, 14342 lines, symbols=3470, tests=289, languages={"eshkol": 288, "text": 1}
- `scripts`: 264 files, 76092 lines, symbols=2862, tests=0, languages={"shell": 148, "python": 105, "javascript": 4, "text": 3, "json": 3, "powershell": 1}
- `tests/vm_parity/corpus`: 181 files, 7770 lines, symbols=775, tests=181, languages={"eshkol": 180, "text": 1}
- `lib/core`: 135 files, 68561 lines, symbols=3257, tests=0, languages={"cpp": 69, "eshkol": 35, "c": 20, "c_header": 10, "text": 1}
- `tests/lists`: 129 files, 6716 lines, symbols=635, tests=129, languages={"eshkol": 129}
- `tests/typesystem`: 126 files, 2794 lines, symbols=380, tests=126, languages={"eshkol": 125, "shell": 1}
- `lib/backend`: 109 files, 215677 lines, symbols=4914, tests=0, languages={"c": 52, "cpp": 48, "c_header": 9}
- `tests/v1_2_edge_cases`: 109 files, 10227 lines, symbols=833, tests=109, languages={"eshkol": 88, "shell": 19, "python": 2}
- `site/static/content`: 89 files, 46610 lines, symbols=428, tests=0, languages={"html": 88, "json": 1}
- `tests/vm`: 83 files, 8750 lines, symbols=758, tests=83, languages={"eshkol": 82, "cpp": 1}
- `tests/toolchain`: 83 files, 14802 lines, symbols=665, tests=83, languages={"cpp": 32, "python": 22, "shell": 17, "eshkol": 8, "javascript": 4}
- `tests/ad`: 76 files, 10354 lines, symbols=1072, tests=76, languages={"eshkol": 72, "shell": 2, "python": 2}
- `docs/reference/stdlib`: 71 files, 11315 lines, symbols=1068, tests=0, languages={"markdown": 70, "text": 1}
- `tests/core`: 71 files, 8999 lines, symbols=632, tests=71, languages={"eshkol": 38, "cpp": 29, "c": 2, "python": 1, "shell": 1}
- `examples`: 67 files, 19589 lines, symbols=4068, tests=0, languages={"eshkol": 66, "markdown": 1}
- `tests/recursion_depth/generated`: 64 files, 1085 lines, symbols=250, tests=64, languages={"eshkol": 63, "text": 1}
- `tests/memory`: 60 files, 6426 lines, symbols=377, tests=60, languages={"eshkol": 43, "shell": 16, "python": 1}
- `tests/autodiff`: 59 files, 3191 lines, symbols=342, tests=59, languages={"eshkol": 59}
- `tests/differential/corpus`: 55 files, 1331 lines, symbols=197, tests=55, languages={"eshkol": 55}
- `tests/ml`: 48 files, 4161 lines, symbols=240, tests=48, languages={"eshkol": 48}
- `tests/ad_oracle/generated`: 45 files, 3819 lines, symbols=1531, tests=45, languages={"eshkol": 44, "text": 1}
- `docs/api/backend`: 44 files, 19947 lines, symbols=1434, tests=0, languages={"markdown": 44}
- `tests/sicp`: 44 files, 6166 lines, symbols=974, tests=44, languages={"eshkol": 44}
- `inc/eshkol/backend`: 44 files, 19888 lines, symbols=600, tests=0, languages={"c_header": 44}
- `tests/features`: 44 files, 6341 lines, symbols=547, tests=44, languages={"eshkol": 44}
- `tests/error_handling/guard_coverage`: 38 files, 917 lines, symbols=80, tests=38, languages={"text": 19, "eshkol": 18, "markdown": 1}
- `docs`: 37 files, 32401 lines, symbols=1827, tests=0, languages={"markdown": 37}
- `docs/breakdown`: 37 files, 28203 lines, symbols=1614, tests=0, languages={"markdown": 37}
- `tests/stress`: 36 files, 959 lines, symbols=139, tests=36, languages={"eshkol": 33, "markdown": 1, "text": 1, "shell": 1}
- `tests/reference-diff/corpus`: 36 files, 358 lines, symbols=0, tests=36, languages={"text": 36}
- `tests/tensor_collection_depth/generated/nested_list`: 32 files, 160 lines, symbols=32, tests=32, languages={"eshkol": 32}
- `tests/tensor_collection_depth/generated/nested_vector`: 32 files, 160 lines, symbols=32, tests=32, languages={"eshkol": 32}
- `docs/design/adr`: 31 files, 12492 lines, symbols=569, tests=0, languages={"markdown": 31}
- `tests/parser`: 31 files, 1934 lines, symbols=163, tests=31, languages={"eshkol": 31}
- `tests/vm_parity/resolved`: 31 files, 412 lines, symbols=36, tests=31, languages={"eshkol": 30, "markdown": 1}
- `docs/tutorials`: 30 files, 4394 lines, symbols=227, tests=0, languages={"markdown": 30}
- `tests/ad_adversarial/generated`: 29 files, 6457 lines, symbols=1488, tests=29, languages={"eshkol": 28, "text": 1}
- `tests/stdlib`: 27 files, 3238 lines, symbols=357, tests=27, languages={"eshkol": 26, "python": 1}
- `docs/reports`: 27 files, 2364 lines, symbols=195, tests=0, languages={"markdown": 26, "json": 1}
- `tests/repl`: 27 files, 1184 lines, symbols=65, tests=27, languages={"eshkol": 22, "cpp": 2, "python": 1, "shell": 1, "text": 1}
