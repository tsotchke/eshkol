# v1.3.6 native dotted case-lambda evidence

- Repro: `.scratch/case-lambda-native-dotted-extra-arity.esk`
- SHA256: `11d70074913513c23ffae8faef5ad2ed27b26a659b6b7e4ce4877f6023140689`
- `git diff --check`: PASS
- CMake target `eshkol-static`: PASS (`[100%] Built target eshkol-static`), compiling the changed parser.
- Native executable gate: BLOCKED. A full configure first tried to fetch tree-sitter-typescript and network DNS failed. Retried with `-DESHKOL_BUILD_AGENT_FFI=OFF`; `eshkol-static` built, but `eshkol-run` link failed on unresolved `_qllm_*` optional FFI symbols from `repl_jit.cpp` (see `/private/tmp/v136-native-case-lambda-dotted-build.log`). `BUILD_REPL=OFF` did not remove that link dependency.
- Consequently JIT/AOT runtime axes and VM parity were not executable in this worktree.
