# Eshkol Security Policy

## Reporting Vulnerabilities

Report security issues privately to **security@eshkol.ai** (or to the
current maintainer listed in `CODEOWNERS`). Please include:

- Affected version/commit
- Minimal reproducer
- Impact (crash, memory corruption, injection, information leak)
- Any proposed remediation

We target an initial response within 3 business days and a fix or
mitigation plan within 14 days for issues rated HIGH or CRITICAL.

**Do not open a public GitHub issue** for security vulnerabilities
before a coordinated disclosure window has been agreed.

## Supported Versions

| Version          | Security fixes                |
| ---------------- | ----------------------------- |
| v1.3.5 (current) | yes                           |
| v1.3.0 - v1.3.4  | yes                           |
| v1.2.x           | yes (until v1.4 GA)           |
| < v1.2           | no                            |

## Trust Boundaries

Eshkol programs cross several trust boundaries that the runtime and
stdlib harden:

- **User Scheme source → compiler**: parser / type checker treat the
  input as trusted code from the developer. Do not pass
  attacker-controlled `.esk` source to the compiler.
- **External data → runtime**: `kb-load` / `model-load` / `image-read` /
  regex / JSON / CSV / HTTP responses are treated as untrusted. Size
  caps, integer-overflow checks, and injection guards are in place
  (see `docs/HARDENING.md`).
- **User program → OS**: subprocess, filesystem, network calls are
  guarded against shell injection (argv-based spawn), path traversal
  (`O_NOFOLLOW` on file_copy), and CRLF injection (HTTP URL/header
  sanitization).
- **Python FFI → Eshkol runtime**: `derivative` / `gradient`
  `func_source` must be a lambda expression (no string literals,
  balanced parens, no trailing code); `eval_file` path must not
  contain NUL or exceed 4 KiB.

## Embedding Constraints

The Eshkol runtime uses several process-global singletons. Each is
designed so the common multi-surface embedding case (Python bindings +
in-process REPL JIT + compiled-to-binary user code) works correctly
without ceremony:

- **Symbol interning table** (`lib/core/symbol_intern.cpp`) —
  process-global `g_interned_symbols` map. Canonical symbol char*
  pointers live in dedicated malloc-backed blocks, NOT the main arena,
  so `eq?` on symbol literals across modules holds even across arena
  resets, REPL session recycles, and independent `EshkolContext`
  instances. The backing blocks are intentionally never freed
  (process-lifetime).
- **Logic-var / predicate registry** (`lib/core/logic.cpp`) —
  `g_var_names`, `g_pred_pool`, `g_pred_table` are shared across all
  callers in the process. Call `eshkol_logic_registry_reset()` (exposed
  to Scheme as part of `(reset-tests!)` in `core/testing.esk`) between
  independent test batches to clear stale logic-var IDs and predicate
  canonical pointers.
- **AD tape** (`lib/core/arena_memory.cpp`) — the reverse-mode tape
  stack is **thread-local** so parallel workers keep isolated tapes.
  The tape node storage itself lives in the main arena; if you reset
  the arena, outstanding tape references go with it. Finalize gradient
  computations before bulk-resetting the arena.

Practical implication: dual-instance embedding is now safe. A Python
process that imports `eshkol` and also spawns an in-process JIT REPL
will observe consistent symbol identity and can reset logic state per
test batch without cross-contamination. What is **not** safe: holding
a direct pointer into the arena (e.g. a tensor data buffer) across an
arena reset — the arena owns that lifetime, and per-instance embedders
need to coordinate resets.

## Known Risky Surfaces (use with care)

- `process-spawn` / `process-spawn-shell` / `run-command` /
  `run-command-capture` — accept a full shell string. Prefer the
  `-argv` variants (`process-spawn-argv`, `run-argv`,
  `run-argv-capture`) for any command built from user input.
- `sqlite-exec` — raw SQL. Prefer `sqlite-prepare` +
  `sqlite-bind-*` + `sqlite-step`, or use `sqlite-exec-safe` if the
  input is constrained.
- `eshkol_eval` / `EshkolContext.eval` — evaluates arbitrary Eshkol
  source. Do not pass attacker-controlled strings.

## Sanitizer / Fuzzing Coverage

- ASan / UBSan builds pass the full aggregate suite (46 suites), including
  the v1.2 edge-case coverage (testing framework, argparse, time API, binary
  I/O, hardening path, regex, JSON). See `scripts/build-sanitizer.sh`.
  `linux-x64-asan-ubsan` is a required, merge-blocking CI lane.
- Leak detection on that lane is proved armed before its silence is trusted
  (v1.3.5-evolve, #486). `scripts/check_leak_detection_selftest.sh` compiles
  two probes under the lane's exact `ASAN_OPTIONS`/`LSAN_OPTIONS`, including
  the checked-in suppression file: one that deliberately leaks and must be
  reported, and one that allocates and frees cleanly and must not be. Until
  this shipped, the lane ran with `detect_leaks=0` and a suppression file that
  could have grown broad enough to swallow a real leak with no visible effect.
- `tests/memory/leak_audit_gate.sh` then runs an AOT compile, the compiled
  program, the VM and the REPL under `detect_leaks=1`, failing on any leak
  `.icc/lsan-suppressions.txt` does not already name and justify, plus a slope
  check on the one retention the suppressions do hide, so a per-form growth
  regression cannot hide behind a suppression rule.
- Closed-enum dispatch is compiler-enforced (v1.3.5-evolve, #500). A `switch`
  over a closed enum may not carry a `default:` clause, so adding a tag, an
  opcode, a heap subtype or a port flag cannot silently fall through to a
  catch-all arm at any registered dispatch site. This is a memory-safety
  property as much as a correctness one: several historical defects here were
  a new subtype reaching a handler that treated it as the wrong shape.
  Enforcement is `-Werror=switch -Werror=switch-enum` plus the
  `ESHKOL_EXHAUSTIVE_SWITCH_BEGIN` macros in
  `inc/eshkol/exhaustive_dispatch.h`, with `scripts/gate_exhaustive_dispatch.py`
  re-deriving each enum's members from its own definition so a removed arming
  is reported rather than merely producing a build that no longer checks.
- ThreadSanitizer is run nightly against the parallel runtime (the
  `concurrency-tsan` job in `.github/workflows/adversarial-nightly.yml`): the
  v1.3.4-evolve `parallel-map` fix took the arena data-race count to zero.
- MSan / LSan are wired via the same CMake flags; their CI lanes are being
  added incrementally.
- Seeded differential fuzzing ships and runs today:
  `scripts/run_differential_fuzz.sh` compares `jit`, `jit-nocache`, `aot-o0`
  and `aot-o2` on generated programs and auto-shrinks any divergence to a
  minimal repro, and `scripts/run_generative_differential.py` drives the
  generative corpus. libFuzzer-based in-process harnesses remain tracked under
  #187.

## Threat Model (summary)

The runtime assumes:

- The host OS, stdlib code, and linked LLVM binaries are trusted.
- The developer writing Eshkol source is trusted.
- **Untrusted**: `.kb` / `.em` / image / JSON / CSV / regex pattern /
  regex subject / URL / HTTP response body inputs, and any
  subprocess command arguments built from these.

Under this model, the hardening priorities are, in order:

1. No memory corruption (ASan/UBSan clean).
2. No command injection via subprocess or HTTP.
3. No DoS from malformed inputs (ReDoS, multi-GB allocations).
4. No silent error swallowing — every ingest point logs or returns
   a specific failure value.

See `docs/HARDENING.md` for the per-module status.
