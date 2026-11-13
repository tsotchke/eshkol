# `agent.http` and `agent.http_server` — HTTP Client & Server

## Client — `agent.http`

```scheme
(require agent.http)
```

Source: `lib/agent/http.esk`. C symbols: `qllm_http_*` (libcurl-backed).

### Lifecycle

| Procedure | Signature |
|-----------|-----------|
| `http-init` | `(http-init)` → `#t` on success |
| `http-shutdown` | `(http-shutdown)` |
| `http-has-ssl?` | `(http-has-ssl?)` → `#t` if TLS available |

### Requests

| Procedure | Signature | Returns |
|-----------|-----------|---------|
| `http-get` | `(http-get url [timeout-ms])` | `(status . body-string)` or `#f` |
| `http-post` | `(http-post url headers body [timeout-ms])` | `(status . body-string)` or `#f` |
| `http-request` | `(http-request method url headers body [timeout-ms])` | dispatches GET/POST |

- `headers` is a list of `(name . value)` pairs; `body` is a string.
- Default timeout is `30000` ms.
- Every request calls `(capability-require! 'network url)` — a `network`
  capability policy (if active) is enforced here (see [capabilities](capabilities.md)).

### What a request must be

A request line and a header block are built from values a program supplies,
and a program's values are data. Eshkol decides what that data is allowed to
be before any of it reaches a socket, and the decision is one grammar, in
`inc/eshkol/core/http_syntax.h`, that every client path reads: this Scheme
wrapper, the core `(http-request ...)` builtin, the bytecode VM's own
implementation, and the native client on libcurl, NSURLSession and WinHTTP.
A method, a target or a header line one engine refuses is refused by all of
them.

| Field | Must be | Refusal |
|-------|---------|---------|
| method | an RFC 9110 §5.6.2 token: letters, digits and `!#$%&'*+-.^_`\|~` | an error naming `method` |
| URL | an absolute `http://` or `https://` URL, built of RFC 3986 target bytes, with no space and no control byte | an error naming the URL |
| request target | one of the four RFC 9112 §3.2 forms: origin (`/path?query`), absolute, authority (`host:port`), or `*` | an error naming `request-target` |
| header name | an RFC 9110 token | an error naming `header` |
| header value | visible ASCII, SP and the RFC 9110 `obs-text` range (0x80-0xFF) — Eshkol deliberately refuses HTAB and the rest of the C0 controls as well as CR, LF, NUL and DEL, because a proxy on the path may re-split a value on them | an error naming `header` |
| header name, reserved set | not `Content-Length`, `Transfer-Encoding`, `Host`, `Connection` or `Upgrade`, and on the WebSocket path not `Sec-WebSocket-*` | an error naming `framing header` |

A refusal is an ordinary Eshkol error naming the field that did not fit, so a
program can catch it with `with-exception-handler` or `guard` and decide what
to do. It is raised before a socket is opened, so nothing is sent and no
descriptor is created.

The header block a caller supplies is never forwarded as bytes. It is
re-rendered from the validated parse with canonical CRLF line endings, so the
request on the wire is exactly the lines the grammar admitted and nothing
else. Blank lines in the input are dropped rather than carried through.

Some fields a caller may not supply at all. Every client writes the framing
and connection-control fields itself from the request it was given:
`Content-Length` and `Transfer-Encoding` frame the body, `Host` comes from the
URL, and `Connection` and `Upgrade` belong to the client's own connection
handling. A caller copy would put a second framing header on the wire beside
the client's, and RFC 9112 §6.3 forbids a sender emitting both a
`Content-Length` and a `Transfer-Encoding`: a front end that honours one and a
back end that honours the other disagree about where the message ends, which
is request smuggling. The WebSocket client additionally reserves the
`Sec-WebSocket-*` set, whose values it derives per connection from the
cryptographic source. Naming any of them is refused with `framing header`.

The agent clients on libcurl, NSURLSession and WinHTTP require a full
absolute `http://` or `https://` URL. A URL carrying non-ASCII, `{`, `|`, `^`
or a fragment is refused there rather than handed to the transport, so the
same program text can be refused by those clients and accepted by the raw
socket client, which parses a narrower set of URLs of its own.

### SSE streaming

| Procedure | Signature |
|-----------|-----------|
| `http-stream-open` | `(http-stream-open url headers body [timeout-ms])` → stream or `#f` |
| `http-stream-next` | `(http-stream-next stream [timeout-ms])` → `(type . data)` or `#f` |
| `http-stream-done?` | `(http-stream-done? stream)` |
| `http-stream-close` | `(http-stream-close stream)` |
| `sse-event-type` / `sse-event-data` / `sse-event-free` | event accessors |

### What the response side does, and does not, do

The validation above is **request-side**. On the response side the two client
layers differ, and a program should know which one it is on.

- The agent clients on libcurl, NSURLSession and WinHTTP parse responses
  through those libraries: `Content-Length`, chunked transfer coding,
  duplicate and conflicting framing fields and truncation are the transport's
  business and are handled there.
- The raw socket client behind the core `(http-request ...)` builtin and the
  bytecode VM reads whatever arrives within a 64 KiB cap and returns it. It
  does not consult `Content-Length`, does not implement chunked decoding — a
  chunked response is returned with its chunk-size lines inside the body — and
  cannot distinguish a response truncated by a reset from a complete one. Its
  allocation is bounded, so this is a completeness limit rather than a safety
  one, but a program parsing those bytes must not assume a framed message.

Response-side grammar for the raw socket client is its own piece of work and
is not part of this client today.

### Implementation status

- Synchronous requests marshal the complete header list through the native
  length-aware HTTP ABI; custom headers are preserved for GET, POST, and the
  generic `http-request` path.
- SSE streaming is implemented end to end: open, next-event, done/error
  status, owned event type/data accessors, and close are all wired to the
  native client. `http-stream-next` returns the parsed event rather than a
  placeholder.

## Server — `agent.http_server`

```scheme
(require agent.http_server)
```

Source: `lib/agent/http_server.esk`. C symbols: `eshkol_http_server_*`,
`eshkol_unix_socket_*`, `eshkol_ws_*`. Handles are `i64`.

### HTTP server

| Procedure | Signature |
|-----------|-----------|
| `http-server-create` | `(http-server-create port)` → handle |
| `http-server-port` | `(http-server-port handle)` → bound port (useful with port 0) |
| `http-server-accept` | `(http-server-accept handle buffer-size timeout-ms)` |
| `http-server-respond` | `(http-server-respond handle status content-type body)` |
| `http-server-close` | `(http-server-close handle)` |

### Unix domain socket & WebSocket

| Procedure | Signature |
|-----------|-----------|
| `unix-socket-connect` | `(unix-socket-connect path)` → fd |
| `ws-wrap-fd` | `(ws-wrap-fd fd)` → ws handle |
| `ws-send-text` | `(ws-send-text handle data)` |
| `ws-send-binary` | `(ws-send-binary handle data)` |
| `ws-receive` | `(ws-receive handle buffer-size timeout-ms)` → `(frame-type . data)` |
| `ws-close` | `(ws-close handle)` |

### The WebSocket opening handshake

`(websocket-connect url headers)` performs the RFC 6455 §4.1 opening
handshake. The caller-supplied headers and the URL-derived target and
authority go through the request grammar above first; what follows is what
the client owes the protocol.

- **A fresh key per handshake.** `Sec-WebSocket-Key` is 16 bytes from the
  operating system's cryptographic source, base64-encoded — a new value for
  every connection, never a value compiled into the client. A key that
  repeated between handshakes would let a cached or replayed response pass for
  a live one. When the cryptographic source cannot be read the connection is
  not attempted and an error names the key.
- **The server's answer checked against that key.** The handshake completes
  only when the response carries an HTTP/1.x `101` status, an `Upgrade` field
  naming `websocket`, a `Connection` field naming the `Upgrade` token, and a
  `Sec-WebSocket-Accept` equal to `base64(SHA-1(key || GUID))` for the key
  this connection sent. A field name is matched only where a line begins, and
  a `101` inside a reason phrase is not a `101` status. Anything short of all
  four leaves the socket closed and returns `#f`.
- **A fresh mask per frame.** Every client frame carries a 4-byte masking key
  drawn for that frame from the same cryptographic source, with the payload
  masked by it, as RFC 6455 §5.3 requires. When the source cannot be read the
  frame is not sent.

SHA-1 appears in Eshkol for this one purpose: it is the function RFC 6455
names for the handshake value. It is not offered as a hash for anything else;
for digests use [`agent.crypto`](crypto.md).

### What a received frame must be

`ws-receive` judges a frame's header before it reads a byte of payload, and
any refusal **fails the connection**: a close frame carrying the RFC 6455 §7.4
code, then the socket closed and the handle marked closed. Returning with the
stream sitting mid-frame is what must not happen — a later read would parse
attacker-chosen payload bytes as a frame header, putting the frame type and
length under the peer's control.

| Rule | RFC 6455 | Close code |
|------|----------|------------|
| a frame from a server is never masked | §5.1 | 1002 |
| RSV1-3 are zero; this client negotiates no extension | §5.2 | 1002 |
| opcodes 3-7 and 11-15 are reserved | §5.2 | 1002 |
| a control frame carries at most 125 bytes and is never fragmented | §5.5 | 1002 |
| a continuation arrives only inside a fragment sequence, and a new data frame only outside one | §5.4 | 1002 |
| a payload, or a gathered message, within the frame budget | — | 1009 |
| a text message is well-formed UTF-8, rejecting overlong forms, surrogates and values above U+10FFFF | §8.1 | 1007 |

A fragmented message is gathered and delivered only when `FIN` arrives, so a
fragment is never surfaced as though it were a whole message; `ws-receive`
returns `#f` for the fragments in between. A `ping` is answered with the
`pong` the protocol requires and also reported to the program.

The frame budget is `ESHKOL_WS_MAX_FRAME_BYTES`, default `65536`, and it
bounds both one frame's payload and the total of a gathered message.

Frame-type constants, the `frame-type` half of the pair `ws-receive` returns
(the values are the RFC 6455 opcodes):

| Constant | Value | Frame |
|----------|-------|-------|
| `WS-FRAME-TEXT` | `1` | UTF-8 text message |
| `WS-FRAME-BINARY` | `2` | binary message |
| `WS-FRAME-CLOSE` | `8` | close handshake |
| `WS-FRAME-PING` | `9` | ping |
| `WS-FRAME-PONG` | `10` | pong |

```scheme
(require agent.http_server)
(display (list WS-FRAME-TEXT WS-FRAME-BINARY WS-FRAME-CLOSE WS-FRAME-PING WS-FRAME-PONG))
(newline)
;; => (1 2 8 9 10)
```

The server can be protected with a bearer token via the `ESHKOL_SERVER_TOKEN`
environment variable.

For the compilation server that serves the browser REPL — how it frames a
request, what bounds it, and what each refusal is answered with — see
[the compilation server reference](../runtime/compilation-server.md).
