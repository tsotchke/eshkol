# `agent.http` and `agent.http-server` — HTTP Client & Server

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
- URLs and header values are validated (`http-safe-string?`) to reject control
  characters / injection.

### SSE streaming

| Procedure | Signature |
|-----------|-----------|
| `http-stream-open` | `(http-stream-open url headers body [timeout-ms])` → stream or `#f` |
| `http-stream-next` | `(http-stream-next stream [timeout-ms])` → `(type . data)` or `#f` |
| `http-stream-done?` | `(http-stream-done? stream)` |
| `http-stream-close` | `(http-stream-close stream)` |
| `sse-event-type` / `sse-event-data` / `sse-event-free` | event accessors |

### Known limitations (report inline)

- `http-post` currently routes through the JSON convenience FFI
  (`qllm_http_post_json`) and only forwards the `Authorization` or `x-api-key`
  header; arbitrary custom headers are not yet marshaled through the full FFI.
- `http-stream-next` returns a placeholder `("message" . "")` pending full SSE
  event-struct field access. Treat SSE as experimental.

## Server — `agent.http-server`

```scheme
(require agent.http-server)
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

Frame-type constants from `ws-receive`: `WS-FRAME-TEXT`, `WS-FRAME-BINARY`,
`WS-FRAME-CLOSE`, `WS-FRAME-PING`, `WS-FRAME-PONG`.

The server can be protected with a bearer token via the `ESHKOL_SERVER_TOKEN`
environment variable.
