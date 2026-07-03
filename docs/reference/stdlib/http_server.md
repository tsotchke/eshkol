# `core.http_server` — pure-Scheme HTTP request parsing, responses, and routing

**Source**: [`lib/core/http_server.esk`](../../../lib/core/http_server.esk)
**Require**: `(require core.http_server)` — **NOT** auto-loaded via `stdlib`; require it explicitly. (It internally pulls in `core.strings` and `core.metrics`.)

The pure-Scheme HTTP layer: parse a raw request **string**, build/inspect response tuples, and route requests to handlers with built-in `/health`, `/ready`, `/metrics` fallbacks. The low-level socket server (`http-server-create`, `http-server-accept`, `http-server-respond`) is a set of **native generated builtins** — this module wraps them but the parsing/routing helpers here work on plain strings and need no live socket.

> Not to be confused with `lib/agent/http_server.esk` (the agent socket server). This document covers only the core pure-Scheme parser module.

A **response** is a 3-element list `(status content-type body)`. A **route** is a 3-element list `(method path handler)` where `handler` is `(lambda (request) …)` returning a response.

## Constants

### `http-max-request-body-size`
Maximum accepted request body in bytes. Value: `10485760` (10 MiB).

## Request parsing

All request accessors take the raw request **string**. They tolerate both `\r\n` and `\n` line endings and degrade gracefully (returning `""` / `#f`) on malformed input rather than erroring.

### `(http-request-line request)` — the first line verbatim.
### `(http-request-method request)` — the method token (e.g. `"GET"`), or `""`.
### `(http-request-target request)` — the request target incl. query (e.g. `"/a?x=1"`).
### `(http-request-path request)` — target with the query stripped.
### `(http-request-query request)` — the query string after `?`, or `""`.
### `(http-request-version request)` — e.g. `"HTTP/1.1"`, or `""`.
### `(http-request-header request name)` — header value by (case-insensitive) name, or `#f`. `name` may be a string or symbol.
### `(http-request-content-length request)` — numeric `Content-Length`, or `#f` if absent/invalid.
### `(http-request-body request)` — everything after the header/body separator, or `""`.
### `(http-request-body-too-large? request)` — `#t` if the declared or actual body exceeds `http-max-request-body-size`.

```scheme
;; http.esk
(require core.http_server)
(define req
  "GET /api/items?page=2&q=x HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\nhello")
(display (http-request-method req)) (newline)          ; GET
(display (http-request-target req)) (newline)          ; /api/items?page=2&q=x
(display (http-request-path req)) (newline)            ; /api/items
(display (http-request-query req)) (newline)           ; page=2&q=x
(display (http-request-version req)) (newline)         ; HTTP/1.1
(display (http-request-header req "host")) (newline)   ; localhost (case-insensitive)
(display (http-request-header req "missing")) (newline); #f
(display (http-request-content-length req)) (newline)  ; 5
(display (http-request-body req)) (newline)            ; hello
(display (http-request-body-too-large? req)) (newline) ; #f
```
```
GET
/api/items?page=2&q=x
/api/items
page=2&q=x
HTTP/1.1
localhost
#f
5
hello
#f
```

## Responses

### `(http-response status content-type body)` — build the `(status content-type body)` tuple.
### `(http-response? x)` — `#t` if `x` is a proper 3+ element list (the tuple shape).
### `(http-response-status r)` / `(http-response-content-type r)` / `(http-response-body r)` — accessors.

```scheme
;; http.esk
(require core.http_server)
(define r (http-response 200 "text/plain" "OK"))
(display (http-response? r)) (newline)               ; #t
(display (http-response? 5)) (newline)               ; #f
(display (http-response-status r)) (newline)         ; 200
(display (http-response-content-type r)) (newline)   ; text/plain
(display (http-response-body r)) (newline)           ; OK
```
```
#t
#f
200
text/plain
OK
```

## Routing

### `(http-route method path handler)`
Build a route tuple. `method`/`path` are matched exactly (string `=`); `handler` takes the request string and returns a response.

### `(http-route-request request routes [ready?])`
Route `request` through `routes` (a list of route tuples). Runs the first matching handler (wrapped so a raised error or non-response becomes a 500); if none match, falls back to `http-standard-response`. Malformed `Content-Length` short-circuits to 400 and an over-size body to 413. Optional `ready?` (default `#t`) controls the `/ready` fallback.

### `(http-standard-response request [ready?])`
Built-in responses for `/health` (`200 OK`), `/ready` (`200 READY` or `500 NOT READY`), and `/metrics` (`200`, body = `(metrics-render)`). Returns `400` for empty method/path, `405` for a non-GET on a standard path, and `404` otherwise.

```scheme
;; http.esk
(require core.http_server)
(define routes
  (list (http-route "GET" "/api/items"
                    (lambda (rq) (http-response 200 "application/json" "[]")))))
(define req "GET /api/items HTTP/1.1\r\nHost: x\r\n\r\n")
(display (http-route-request req routes)) (newline)
(display (http-standard-response "GET /health HTTP/1.1\r\nHost: x\r\n\r\n")) (newline)
(display (http-standard-response "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n")) (newline)
```
```
(200 application/json [])
(200 text/plain OK
)
(404 text/plain Not Found
)
```

(The `/metrics` standard response body is whatever `core.metrics`' `metrics-render` produces; it is empty when no metrics are registered.)

## Socket-bound helpers

These two wrap the native `http-server-respond` builtin and therefore require a live server handle from `http-server-create` / `http-server-accept`. They cannot be exercised without an actual socket, so no standalone run is shown.

### `(http-server-respond-response handle response)`
Send a response tuple over the given server `handle`. Returns `#t`.

### `(http-server-respond-standard handle request [ready?])`
Compute `(http-standard-response request ready?)` and send it over `handle`.

Edge cases: request accessors never raise on malformed input — a garbage string yields `""`/`#f` from the parsers. A handler that throws or returns a non-response is converted by `http-route-request` into `500 Internal Server Error`.
