---
kind: reference
status: current
owner-area: runtime
since: v1.4.0-connection
sources:
  - lib/core/tls.esk
  - lib/core/tls.c
  - inc/eshkol/core/tls.h
---

# TLS

`(require core.tls)` runs TLS 1.2 and 1.3 through the system OpenSSL library
over [core.net](net.md) TCP streams. Linux and macOS builds link OpenSSL when
the configure step finds version 1.1.1 or newer; `(tls-available?)` reports it.
Windows TLS through Schannel is Planned for v1.4.1. A build without TLS
returns error `-3` from every operation and never falls back to plaintext. The
bytecode VM refuses `core.tls` at compile time.

## Verification

Peer verification is on by default. A client context trusts the certificate
authorities in `ca-file`, or the platform's default trust store when
`ca-file` is `#f`. A client handshake sends the expected name as SNI and checks
the certificate against it: a DNS name against the certificate's DNS names, a
numeric address against its IP addresses. A refused handshake reports why:

| Code | Reason | Meaning |
|---|---|---|
| `-10` | `untrusted-certificate` | The certificate chain does not lead to a trusted authority, or is expired or malformed. |
| `-11` | `identity-mismatch` | The certificate is trusted but names another host or address. |

An unverified connection needs an explicit option:
`(tls-client-context/options ca-file '(insecure-no-verify))`. Any other option
list is refused with `-4`; `'()` is the verifying default. Server contexts load
a PEM certificate chain and an unencrypted PEM private key that must match it;
encrypted keys are refused rather than prompting for a password.

## Sessions on TCP streams

| Procedure | Result |
|---|---|
| `(tls-client-context ca-file)` | Verifying client context. |
| `(tls-client-context/options ca-file options)` | Client context with an option list. |
| `(tls-server-context cert-file key-file)` | Server context. |
| `(tls-client-start context tcp-stream server-name timeout-ms)` | Client session over an owned TCP stream. |
| `(tls-server-start context tcp-stream timeout-ms)` | Server session over an owned TCP stream. |
| `(tls-connect context address port server-name timeout-ms)` | Connect and start a client session; `server-name` `#f` verifies the address. |
| `(tls-listen context address port)`, `(tls-accept listener timeout-ms)` | TLS listener and server sessions. |

A session takes ownership of its TCP stream and holds that stream's lease
for its whole life: other uses of the stream handle return `-5` (busy), and
closing the session closes the stream. If the handshake fails the stream is
closed. A stream that is already in use, closed or of the wrong kind is
refused with `-5` or `-4` and stays with the caller. Each start, connect and
accept completes the TCP and TLS phases within one deadline.

The scoped forms close the session exactly once when `proc` returns, raises,
or escapes, and refuse re-entry, with the same rules as the
[core.net scoped forms](net.md#handles-and-their-kinds):
`(call-with-tls-client context tcp-stream server-name timeout-ms proc)`,
`(call-with-tls-server context tcp-stream timeout-ms proc)`,
`(call-with-tls-connection context address port server-name timeout-ms proc)`
and `(call-with-tls-accept listener timeout-ms proc)`.

## Data

| Procedure | Result |
|---|---|
| `(tls-read! session bv start count timeout-ms)` | Plaintext bytes received (1 to `count`, at most 1 MiB), the eof object after the peer's close-notify, or an error. |
| `(tls-write! session bv start count timeout-ms)` | `count` (at most 1 MiB per call). |
| `(tls-write-all! session bv start count timeout-ms)` | `count` for a range of any length, within one deadline. |
| `(tls-cancel! session)` | Cancels the session's waiting operation from any thread, or its next one (`-6`). |
| `(tls-read session capacity timeout-ms)`, `(tls-write session data timeout-ms)` | String forms. |

Close a session with `tls-session-close` (it sends close-notify when the
session is healthy), a listener with `tls-listener-close` and a context with
`tls-context-close`; active sessions keep their context alive. Errors use the
codes of [core.net](net.md#outcomes) plus the verification codes above, through
`tls-error?`, `tls-error-code`, `tls-error-operation` and `tls-error-reason`.
TLS operations require the `network` capability, and reading certificate or
trust files requires `file-read`.

## Evidence

The tests generate a local certificate authority, a server certificate for
`localhost` and `127.0.0.1`, and a certificate from an unrelated authority
under the build directory (`tls_test_ca`); they use loopback only.

- `tls_lease_test` (C) and `tls_net_core_jit`, `tls_net_core_aot`: client and
  server handshakes on owned TCP streams, the session's exclusive stream lease,
  a round trip of more than 16 MiB, refusal of an untrusted certificate and of
  a hostname mismatch, the explicit insecure option, cancellation, and
  guaranteed close.
- `tls_context_native`, `tls_public_scheme_jit`, `tls_public_scheme_aot` and the
  Python-peer suites: context validation, expired and wrong certificates,
  concurrent close, and interoperation with an independent TLS stack.
