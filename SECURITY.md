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
| v1.2.x (current) | yes                           |
| v1.1.x           | yes (until v1.3 GA)           |
| < v1.1           | no                            |

## Trust Boundaries

Eshkol programs cross several trust boundaries that the runtime and
stdlib harden:

- **User Scheme source → compiler**: parser / type checker treat the
  input as trusted code from the developer. Do not pass
  attacker-controlled `.esk` source to the compiler.
- **External data → runtime**: `kb-load` / `model-load` / `image-read` /
  regex / JSON / CSV / HTTP responses are treated as untrusted. Size
  caps, integer-overflow checks, and injection guards are in place
  (see `HARDENING.md`).
- **User program → OS**: subprocess, filesystem, network calls are
  guarded against shell injection (argv-based spawn), path traversal
  (`O_NOFOLLOW` on file_copy), and CRLF injection (HTTP URL/header
  sanitization).
- **Python FFI → Eshkol runtime**: `derivative` / `gradient`
  `func_source` must be a lambda expression (no string literals,
  balanced parens, no trailing code); `eval_file` path must not
  contain NUL or exceed 4 KiB.

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

- ASan / UBSan builds pass the full v1.2 edge-case suite
  (testing framework, argparse, time API, binary I/O, hardening
  path, regex, JSON). See `scripts/build-sanitizer.sh`.
- TSan / MSan / LSan are wired via the same CMake flags; CI lanes
  are being added incrementally.
- Fuzzing harnesses (libfuzzer) are tracked under #187.

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

See `HARDENING.md` for the per-module status.
