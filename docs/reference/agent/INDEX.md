# Agent & FFI Reference

The `agent.*` modules are Eshkol's hosted, capability-gated surface for talking
to the outside world — HTTP, databases, subprocesses, cryptography, the terminal,
git, and native training. They are built on the [FFI](ffi.md) `extern`
mechanism and link native C runtimes (`qllm_*`, `eshkol_*`). Load one with
`(require agent.<name>)`.

## Foundations

- [FFI](ffi.md) — the `extern` / `:real` declaration syntax, the type-keyword →
  C ABI mapping, tagged-value boundary conversion, and how `requires_agent_ffi`
  drives **transitive** agent-FFI linking for AOT binaries (vs JIT resolution).
- [Capabilities](capabilities.md) — the process-local allow-list policy and which
  agent operation requires which capability.

## Modules

| Module | Doc | Summary |
|--------|-----|---------|
| `agent.http`, `agent.http-server` | [http](http.md) | HTTP client (GET/POST, SSE), HTTP/Unix-socket/WebSocket server |
| `agent.sqlite` | [sqlite](sqlite.md) | Embedded SQLite: connections, prepared statements, `with-db`/`with-statement` |
| `agent.subprocess` | [subprocess](subprocess.md) | Process spawning (shell + injection-safe argv), ownership/cleanup contract (#94) |
| `agent.crypto` | [crypto](crypto.md) | SHA-256, HMAC, random bytes/hex, UUIDv4, base64url |
| `agent.eagle` | [eagle](eagle.md) | Native linear-head training (`qllm_ffi_linear_*`, PR #104) |
| `core.memory`, `core.memory-store` | [memory faculty](memory-faculty.md) | Content-addressed, CRDT-merged event log |
| `agent.regex`, `agent.glob`, `agent.fs-watch`, `agent.keychain`, `agent.terminal`, `agent.git-ffi`, `agent.layout` | [platform utilities](platform-utilities.md) | Regex (PCRE2), globbing, file watching, secret store, terminal/TUI, git, layout |
| `agent.quantum` | [quantum](quantum.md) | Moonlab state-vector simulation, gates, measurement, CHSH Bell gate, molecular Hamiltonians, and a VQE energy differentiable through Eshkol's AD (opt-in `-DESHKOL_QUANTUM_ENABLED=ON`; `quantum-random` builtins in all builds) |
| `agent.pqc` | [pqc](pqc.md) | ML-KEM (FIPS 203) post-quantum KEM at 512/768/1024, QRNG-seeded, over the same opt-in Moonlab link target |

## Cross-cutting notes

- **AOT vs JIT**: always verify agent-FFI code under AOT, not just `-r` — the JIT
  resolves symbols from the host process, while AOT depends on the link-args scan
  firing. See [FFI § AOT linking](ffi.md).
- **Capabilities off by default**: with no policy installed everything is
  permitted; installing one is deny-by-default. See [capabilities](capabilities.md).

## See also

- [Runtime reference](../runtime/INDEX.md) — CLI, environment variables, memory,
  parallelism, JIT.
