# The compilation server — request handling

`exe/eshkol-server.cpp` is a small HTTP server that compiles Eshkol source to
WebAssembly for the browser REPL. It binds the loopback by default and a public
bind requires a compile token. This page is the reference for what it accepts
as a request: how a request is framed, what bounds it, and what each refusal
is answered with.

The rules live in `inc/eshkol/http_request_utils.h`, which the server drives
from its read loop and which `eshkol_server_request_handling_test` drives
directly, so the behaviour described here is the behaviour under test.

## Framing: one reading of the header section

A request's body length is decided once, from one reading of the header
section, and that one answer is what both the read loop and the request
handler use.

| Rule | Answer when it is not met |
|------|---------------------------|
| A field name counts only where a line begins. `X-Note: content-length: 999` names no framing field. | — (the field is simply not a framing field) |
| `Content-Length` appears at most once (RFC 9112 §6.3). Two occurrences are refused even when the values agree, and a list-valued `Content-Length` is the same message as a repeated one. | 400 |
| `Content-Length` is a single decimal length. | 400 |
| `Transfer-Encoding` and `Content-Length` do not both appear. | 400 |
| `Transfer-Encoding` names a framing this server implements. It implements none, so a request carrying the field is refused rather than read as though it had no body. | 400 |
| The request method is an RFC 9110 token. | 400 |
| The request target is one of the four RFC 9112 §3.2 forms. | 400 |
| A request that ends before its declared body has arrived is not answered as complete. | 400 |

## Accumulation is by byte count

Read bytes are appended with their exact length, never as a NUL-terminated
string, so a body containing NUL bytes arrives whole and the server's view of
the request is the view on the wire. The terminator of the header section is
the blank line and nothing else; a NUL inside a header line does not end it.

## Budgets

Three numbers bound what one connection can cost. Each has a documented
default and one environment variable that overrides it, read once per process.

| Budget | Default | Environment variable | Answer when passed |
|--------|---------|----------------------|--------------------|
| Header section (request line and fields, through the blank line) | 64 KiB | `ESHKOL_SERVER_MAX_HEADER_BYTES` | 431 Request Header Fields Too Large |
| Body | 10 MiB | `ESHKOL_SERVER_MAX_BODY_BYTES` | 413 Payload Too Large |
| Time for one request to arrive, measured from accept | 15 s | `ESHKOL_SERVER_REQUEST_DEADLINE_SECONDS` | 408 Request Timeout |

A fourth number bounds concurrency: `kMaxConcurrentConnections` (128) is the
number of connections the server will hold at once.

The header budget bounds a header section that never ends as well as a single
long field: the bound is on the bytes held, not on any one line.

The deadline is enforced against two clocks. Each `recv()` blocks for at most
5 seconds, which is well inside the request deadline, so the read loop returns
to re-check the deadline rather than waiting on the peer. A client that sends
one byte at a time therefore holds exactly its own connection for at most the
deadline, gets 408, and is closed; connections being served alongside it are
answered on their own time.

The reader's own input-size cap
(`ESHKOL_MAX_SOURCE_BYTES`, see [`eshkol-run`](eshkol-run.md)) still applies
downstream to the source a request carries: the server's body budget bounds
the request, the reader's cap bounds the text.

## Per-request isolation of a compilation

`POST /compile` compiles one request's source on that request's connection.
A source the compiler cannot find memory for is answered 413 on that
connection, and any other failure in the pipeline is answered 400 with the
reason; the process stays up and every other connection is unaffected.

## Session identifiers

A session identifier names a compilation a client will come back for, so it is
drawn from the operating system's cryptographic source — 16 bytes, rendered as
32 hexadecimal characters. The seeded generator behind Eshkol's `(random)` is a
reproducible stream for simulation and replay and has no part in this; see
[`core.crypto`](../agent/crypto.md). When the cryptographic source cannot be
read the request is answered 503 rather than issued a guessable identifier.

A client may supply its own `session_id`, so it is a value the server did not
choose. Like every other string in a response it is rendered by
`eshkol_json_string()`, the one serializer that knows what a JSON string may
contain. Quotes, backslashes, tabs, line breaks and NUL in a session
identifier come back escaped; they cannot close the string or add a member to
the document.

## Statuses

| Status | When |
|--------|------|
| 200 | the request was answered |
| 400 | the request could not be framed, or its line was not a request line |
| 401 | `POST /compile` without the configured compile token |
| 403 | a static path that resolves outside the served directory |
| 404 | no route |
| 408 | the request did not arrive within the per-request deadline |
| 413 | the body budget, or one compilation's memory, was passed |
| 415 | `POST /compile` without `Content-Type: application/json` |
| 431 | the header budget was passed |
| 503 | the cryptographic source for session identifiers is unavailable |

Every refusal body is a JSON document naming the rule that was not met, built
by the serializer; it never echoes the client's bytes.
