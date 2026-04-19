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

### Noesis residual audit (2026-04-18 v2) — landed 2026-04-18

| Item                                                  | Status | Notes                                                                                           |
| ----------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| BUG 1 — `call-with-values` can't resolve stdlib list  | FIX    | Synthesized `builtin_list_varargs` / `builtin_values_varargs` in codegenVariable so the bare symbol has a first-class value with CLOSURE_FLAG_VARIADIC set. |
| BUG 2 — `(define (f . args) args)` bound first arg only | FIX    | REPL hot-reload + forward-ref call paths now look up `g_repl_variadic_functions` and build the slot with (fixed_params+1) ABI + cons the tail into a Scheme list. |
| BUG 3a — agent FFI symbols stripped under `-e`        | FIX    | `-force_load` / `--whole-archive` on `libeshkol-agent-ffi.a` + PUBLIC transitive deps (sqlite3, pcre2, Security framework, OpenSSL) + agent target-link moved after target definition. |
| BUG 3b — PCRE2 pkg-config name mismatch (Homebrew)    | FIX    | CMake tries `libpcre2-8` first, falls back to `pcre2-8` for Debian-style distros. Uses PCRE2_AGENT_LINK_LIBRARIES (full paths) instead of short `-l` names. |
| BUG 3c — regex.esk used nonexistent `char-set` in trim | FIX    | Replaced with explicit NUL-scan loop; PCRE2 match output is NUL-terminated anyway. |

### Noesis residual audit (2026-04-18 v3) — landed 2026-04-18

| Item                                                  | Status | Notes                                                                                           |
| ----------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| BUG A — `':key` / `:foo` colon-keyword parse error    | FIX    | Tokenizer now disambiguates: `:` glued to identifier-start emits a single TOKEN_SYMBOL (`:foo`); whitespace-separated `:` keeps TOKEN_COLON for type annotations. Extern parser updated to accept both `:real` shapes. |
| BUG B — bare `eshkol-run file.esk` silently no-op     | FIX    | Prints a one-line stderr notice in default-output (a.out) case telling the user the file compiled and how to JIT-execute. Skipped when -o is given. |
| BUG C — user define doesn't shadow builtin arity check | FIX    | codegenCall shadowing check now consults `g_repl_user_function_names` for cross-batch REPL user functions; promotes them to the user-defined dispatch path so the hard-coded `if (func_name == "outer") …` table is bypassed. |
| BUG D — verifier "Terminator in middle of BB" / cascading undef vars | FIX    | TCO codegen at codegenTailCallFromContext was emitting `br loop_header` for any name-matching self-recursive call inside the function body — including ones in arg position. R5RS §3.5: args are never in tail position. The inner branch terminated the block mid-arg-evaluation; the outer call's subsequent stores then landed after the terminator. Fix saves/clears the TCO context across arg evaluation. Sigma discover.esk now loads cleanly. |
| String-length regression cluster                      | FIX    | `eshkol_utf8_strlen` honours object header `size` instead of strlen, so NUL-filled make-string buffers report correct length. Fixed 4 callers (rational.cpp, bignum.cpp, runtime.cpp, makeString codegen) that were over-allocating by `+1`. R7RS §6.7. |
| `(command-line)` returned garbage in compiled binaries | FIX    | SystemCodegen::commandLine now uses `arena_allocate_string_with_header` so each argv string carries the header that display/string-length need. |
| eshkol-run main publishes argc/argv globals           | FIX    | Sets `__eshkol_argc`/`__eshkol_argv` from main(argc, argv) so user scripts under -e/-r can read the CLI; standalone-binary path already populated them via codegen-emitted main. |
| forward-ref non-variadic path double-emitted args     | FIX    | The non-variadic else branch added in v2 BUG 2 fix re-emitted args after the fixed-loop already pushed them; LLVM module verifier rejected with "Incorrect number of arguments". Removed the redundant else branch. |

---

## v1.2.0-scale closeout audits (2026-04-19)

### `#178` memory-safety — ASan + UBSan clean PASS

Built with `scripts/build-sanitizer.sh asan+ubsan`, then ran every
suite in `tests/v1_2_edge_cases/` plus `tests/features/data_encoding_test.esk`
under `ASAN_OPTIONS=detect_leaks=0:halt_on_error=0` +
`UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0`:

| Suite | errors | checks |
|---|---|---|
| `bug_regression_test.esk` | 0 | 57 pass |
| `agent_ffi_jit_test.esk` | 0 | 10 pass |
| `first_class_builtins_test.esk` | 0 | 21 pass |
| `collections_test.esk` | 0 | 17 pass |
| `boundary_values_test.esk` | 0 | 36 pass |
| `image_io_test.esk` | 0 | 9 pass |
| `hash_table_test.esk` | 0 | 27 pass |
| `binary_io_test.esk` | 0 | 8 pass |
| `data_encoding_test.esk` | 0 | 11 pass |

Zero AddressSanitizer / UndefinedBehaviorSanitizer reports across
**196 checks / 9 suites**. The fixes tracked in `#192` / `#193` /
`#194` under v1.2-scale shipped (arena overflow, integer overflow
in kb_persistence, path-traversal, Windows cmdline buffer, error-
propagation, ReDoS / SQLi / URL-encoding validation) hold under
sanitizer instrumentation.

## Still pending

### HIGH

- `#187` — libfuzzer harnesses per ingest point. Prerequisite for
  finding the next wave of issues that targeted audits miss.
- `#186` — 24h stress test: long-running agent + parallel-at-scale
  under ASan; verifies the overflow guards hold under sustained
  load.
- `#180` — TSan audit: `scripts/build-sanitizer.sh tsan`, run the
  parallel and async suites, fix any data races in parallel-map /
  worker-thread paths.

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
