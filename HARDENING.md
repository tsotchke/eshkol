# Eshkol Hardening Status

Per-module hardening audit + mitigation record. Updated when a task
from the `#178`–`#195` hardening epics closes. Sanitizer coverage for
each item is implied by "ASan-clean" / "UBSan-clean" etc.

## Legend

- **FIX** — hardened with code change + regression test.
- **DOC** — documented as risky; caller is responsible.
- **TODO** — known risk, no mitigation yet.

---

## v1.2-scale shipped

### Input validation (`#179`)

| Module                             | Status | Notes                                                         |
| ---------------------------------- | ------ | ------------------------------------------------------------- |
| `agent_subprocess.c` shell inject  | FIX    | `process-spawn-argv` / `run-argv` use `execvp` directly. `#190`. |
| `eshkol_module.cpp` AST inject     | FIX    | `validate_lambda_source` guards derivative/gradient. `#191`.  |
| `bindings/python` `eval_file` path | FIX    | rejects empty, NUL-containing, >4 KiB. `#195`.                |
| `agent/http.esk` URL/header CRLF   | FIX    | `http-safe-string?` check on URL + every header. `#195`.      |

### Memory safety (`#178`)

| Module                                  | Status | Notes                                                                  |
| --------------------------------------- | ------ | ---------------------------------------------------------------------- |
| `arena_memory.cpp` overflow             | FIX    | `data_size` capped at `SIZE_MAX - header - 8` and `UINT32_MAX`. `#192`. |
| `kb_persistence.cpp` arity overflow     | FIX    | arity ≤ 4096 + explicit multiply bound. `#192`.                        |
| `kb_persistence.cpp` string length bomb | FIX    | 16 MiB cap on serialized strings/symbols/bignums. `#192`.              |
| `image_io.c` dimension overflow         | FIX    | `w`,`h` ≤ 65535; `channels` ≤ 16; explicit product bounds. `#192`.      |
| `system_builtins.c` `path_normalize`    | FIX    | rejects inputs ≥ PATH_MAX; bounded concatenation. `#193`.              |
| `system_builtins.c` `file_copy` TOCTOU  | FIX    | `O_NOFOLLOW` + `O_CLOEXEC` on both fds. `#193`.                        |
| `agent_subprocess.c` Win cmdline buf    | FIX    | heap-allocate sized exactly, reject ≥ 32768-char cmdlines. `#193`.      |

### Denial of service (`#195`)

| Module                          | Status | Notes                                                                      |
| ------------------------------- | ------ | -------------------------------------------------------------------------- |
| `agent_regex.c` ReDoS           | FIX    | 10M-step match_limit + 100K depth_limit via process-global match_context. |
| `agent/sqlite.esk` injection    | DOC    | `sqlite-exec-safe` blocks `;`, `--`, `/*`; prefer bind-* for user input.   |

### Linker / build

| Module                   | Status | Notes                                                                       |
| ------------------------ | ------ | --------------------------------------------------------------------------- |
| Sanitizer CMake options  | FIX    | ASan / UBSan / TSan / MSan / LSan wired. `#188`. `scripts/build-sanitizer.sh`. |
| Eval bridge (compile)    | FIX    | Function-pointer indirection; user binaries link cleanly without REPL lib. `#134`. |
| Testing framework        | FIX    | `core.testing`: function-based, works JIT + compile. `#142`.                |

### R7RS compliance

| Module                              | Status | Notes                                                                    |
| ----------------------------------- | ------ | ------------------------------------------------------------------------ |
| `read-bytevector` k=0 semantics     | FIX    | Returns empty bytevector (not eof-object) per R7RS §6.13.2. `#144`.       |
| Symbol interning across modules     | FIX    | `eshkol_intern_symbol_lookup` in `symbol_intern.cpp`. `#196`.             |
| First-class codegen builtins        | FIX    | Sret wrapper registry for AD ops. `#197`.                                 |

### Architectural audit follow-ups (landed 2026-04-18)

| Item                                       | Status | Notes                                                                           |
| ------------------------------------------ | ------ | ------------------------------------------------------------------------------- |
| Missing tensor backward (6 ops)            | FIX    | TRANSPOSE, SUM, BROADCAST_{ADD,MUL}, EMBEDDING, ATTENTION — was silent gradient corruption. `tensor_backward.cpp`. |
| `sexp_to_ast` call-arg uninit slots        | FIX    | Matches the earlier `#194` pattern for sequence-op loops. Logs + marks ESHKOL_INVALID. |
| Symbol interning arena coupling            | FIX    | `symbol_intern.cpp` now uses process-lifetime malloc blocks. Dual-instance embedding safe. |
| `reset-tests!` leaks logic/predicate state | FIX    | `eshkol_logic_registry_reset` added, wired into `core/testing.esk`.              |
| Agent `shell-escape` copy-paste            | FIX    | fs-watch/glob/git-ffi/keychain alias `shell-quote` from stdlib. Single source of truth. |
| ESHKOL_PATH accepts garbage                | FIX    | Empty segments, nonexistent dirs, files-posing-as-dirs now rejected with debug log. |
| Precompiled stdlib detection silent-fail   | FIX    | `eshkol-run.cpp` errors clearly if stdlib.o is linked but `lib/*.esk` tree is missing. |
| tensor_backward NULL-dispatch warning      | FIX    | Default case now stderr-warns once per AD_NODE type instead of silently returning NULL. |
| Version string (was "1.1.0-accelerate")    | FIX    | Bumped to 1.2.0-scale to match the shipping branch.                              |
| Embedding constraints doc                  | DOC    | `SECURITY.md` § Embedding Constraints — now states what's safe, not what's banned. |

---

## Still pending

### HIGH

- `#194` — error-propagation audit: 36 silent-swallow sites.
  Each needs a triage: propagate via exception, return sentinel, or
  log+continue. No code change yet.
- `#187` — libfuzzer harnesses per ingest point. Prerequisite for
  finding the next wave of issues that targeted audits miss.
- `#186` — 24h stress test: long-running agent + parallel-at-scale
  under ASan; verifies the overflow guards hold under sustained
  load.

### MEDIUM

- `system_builtins.c` `getenv` (USER/HOME/PATH) unchecked before
  path construction — `#193` MEDIUM.
- `system_builtins.c` `path_join` does not normalize — `../` passes
  through unchanged.
- `system_builtins.c` `mkdir_recursive` / `rmdir_recursive` — `stat`
  then operation is a classic TOCTOU; pending.

### LOW

- `system_builtins.c:880` `fcntl` file-lock non-blocking fallback
  silently proceeds on failure. Should log or return distinct sentinel.
- `system_builtins.c:236` `snprintf` strlen sum can hypothetically
  exceed PATH_MAX on malformed input. Belt-and-suspenders guard.

---

## Sanitizer regression gate

Any change that touches code listed in this file MUST pass
`scripts/build-sanitizer.sh asan` + the v1.2 edge-case suites before
merge. The sanitizer infrastructure was added in `#188` precisely so
regressions in these guards are caught early.

```
bash scripts/build-sanitizer.sh asan
(cd build-asan && ./eshkol-run -r ../tests/v1_2_edge_cases/hardening_path_test.esk)
(cd build-asan && ./eshkol-run -r ../tests/v1_2_edge_cases/testing_framework_test.esk)
(cd build-asan && ./eshkol-run -r ../tests/v1_2_edge_cases/argparse_test.esk)
(cd build-asan && ./eshkol-run -r ../tests/v1_2_edge_cases/time_api_test.esk)
(cd build-asan && ./eshkol-run -r ../tests/v1_2_edge_cases/binary_io_test.esk)
```

All five suites must report `RESULT: OK` (with `testing_framework_
test.esk`'s META self-check printing the intentional-failure diff).
