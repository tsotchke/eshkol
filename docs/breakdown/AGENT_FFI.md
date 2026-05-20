# Eshkol Agent FFI Subsystem

**Status**: Stable — v1.2.1-scale
**Modules**: `agent.http`, `agent.http_server`, `agent.sqlite`,
`agent.subprocess`, `agent.regex`, `agent.crypto`, `agent.fs-watch`,
`agent.glob`, `agent.keychain`, `agent.git-ffi`
**C runtime root**: `lib/agent/c/`
**Eshkol bindings**: `lib/agent/*.esk`
**Cross-reference**: [`WEB_PLATFORM.md`](WEB_PLATFORM.md) (browser-side
counterpart for `--wasm` builds)

---

## 1. Overview and design philosophy

The Eshkol *agent FFI* is the set of native side-doors that an
agentic Eshkol program reaches for when re-implementing the service
in Eshkol itself would be wasteful, slow, or insecure. It covers
the surfaces a long-running autonomous program needs in practice —
HTTPS calls to model APIs, a structured local store, subprocesses
to drive external tools, a way to receive events from the
filesystem, regular-expression parsing, and cryptographic
primitives. Each surface is a thin Scheme wrapper over a small
C translation unit under `lib/agent/c/`. The split is deliberate:

- **C runtime** owns the system call, the locking, the error
  envelope, and the resource lifetime. Most modules expose work
  through an opaque `int64_t` handle indexing a fixed-size table
  in the runtime (see `lib/agent/c/agent_sqlite.c §alloc_db`,
  `agent_http_server.c §HttpServer`, `agent_watch.c §Watcher`).
  Subprocess returns a `void*` because the per-process struct is
  too large to pack into a single integer (`agent_subprocess.c
  §eshkol_subprocess_t`).
- **Eshkol wrapper** does input validation, builds the higher-
  level shape (alists, lists, pairs), and binds tagged result
  values back into the language. Validation lives in the
  wrapper, not the runtime, so the rejection happens *before*
  the call crosses the FFI boundary.

The security posture is **validated by default**. The argv-based
subprocess wrappers refuse tab and NUL in any argument
(`lib/agent/subprocess.esk §process-argv-check-args`); the HTTP
client refuses CR/LF/NUL/TAB/VT/FF/DEL in URLs and headers
(`lib/agent/http.esk §http-safe-string?`); the regex engine
refuses to backtrack past 10 million steps
(`lib/agent/c/agent_regex.c §get_match_context`); the SQLite
`-safe` wrapper refuses statement-batching characters
(`lib/agent/sqlite.esk §sqlite-exec-safe`).

The agent FFI is **optional**: the CMake option
`ESHKOL_BUILD_AGENT_FFI` defaults to `ON` on POSIX and `OFF` on
Windows (`CMakeLists.txt:1705`), and individual modules degrade
gracefully when their system library is absent. A build without
`libcurl` still links the agent stub set but `http-get` returns
`#f` at runtime because `qllm_http_get` is the weak unavailable
form. A build without `pcre2-8` simply omits `agent_regex.c`
from the agent archive entirely.

---

## 2. Build-time wiring

### 2.1 Static archive — `libeshkol-agent-ffi.a`

`CMakeLists.txt §ESHKOL_BUILD_AGENT_FFI` (line 1711) collects every
file under `lib/agent/c/` into a single static library named
`eshkol-agent-ffi`. Optional dependencies are probed via
`pkg_check_modules`:

```cmake
pkg_check_modules(PCRE2_AGENT libpcre2-8)   # fallback: pcre2-8
pkg_check_modules(SQLITE3_AGENT sqlite3)
pkg_check_modules(LIBCURL_AGENT libcurl)
```

Each `*_FOUND` flag gates inclusion of the corresponding C source
into `AGENT_FFI_SOURCES`; when libcurl is found, the build also
defines `ESHKOL_HAVE_LIBCURL=1` on the agent target so the body
of `agent_http_client.c` is compiled in (otherwise the entire
file is `#ifdef`'d out and the qllm_http_* symbols remain
satisfied by the existing weak unavailable stubs).

The Crypto module is unconditional: macOS picks up CommonCrypto
and SecRandom; Linux uses OpenSSL via `find_package(OpenSSL)`
(`agent_crypto.c §HAVE_COMMONCRYPTO`/`HAVE_OPENSSL`).

### 2.2 Force-load into eshkol-run for JIT discovery

`eshkol-run` itself links the agent archive PRIVATE with
`-force_load` on macOS (or `-Wl,--whole-archive` on Linux) so the
JIT in REPL mode can dlsym every symbol — none of these symbols
are referenced from eshkol-run's C++, only from `(extern …
:real …)` declarations inside `lib/agent/*.esk`
(`CMakeLists.txt:1970-1988`, see also v2 BUG 3 in HARDENING.md).
The same force-load wiring is applied to `eshkol-repl`
(`CMakeLists.txt:1990-2007`) so long-lived control sessions and
warm `--machine`-mode workers can resolve the agent symbols
identically to one-shot JIT runs.

### 2.3 AOT link arguments — `ESHKOL_HOST_AGENT_FFI_LINK_ARGS`

User binaries built via `eshkol-run file.esk -o out` need their
own link line. `CMakeLists.txt` assembles the right ordered list
of flags during configure and bakes it into `build_config.h` via
the substitution in `cmake/build_config.h.in:15`:

```c
#define ESHKOL_HOST_AGENT_FFI_LINK_ARGS "@ESHKOL_HOST_AGENT_FFI_LINK_ARGS@"
```

The order is:

1. The force-load fragment that prevents dead-stripping of the
   agent archive: `-Wl,-force_load,<path>` on macOS, or
   `-Wl,--whole-archive <path> -Wl,--no-whole-archive` on Linux
   (`CMakeLists.txt:1885-1891`).
2. Each found pkg-config dependency's link libraries (libcurl,
   sqlite3, pcre2-8) — full absolute paths when CMake populated
   `*_LINK_LIBRARIES`, otherwise short `-l<name>` plus a
   `-L<dir>` fallback (`CMakeLists.txt:1893-1930`).
3. ncurses (for `agent_terminal.c`), and the crypto vendor
   libraries: `-framework Security -framework CoreFoundation`
   on macOS, `-lssl -lcrypto` on Linux when OpenSSL is found
   (`CMakeLists.txt:1931-1942`).

The list is whitespace-joined into a single C string
(`CMakeLists.txt:1952`). Splitting on whitespace inside
eshkol-run is safe because the inputs are all system paths
from pkg-config; user paths with embedded spaces would be a
problem in principle but pkg-config does not produce them in
practice.

### 2.4 AST scan before `process_requires`

The critical sequencing in `exe/eshkol-run.cpp` is that the
agent-FFI usage flag must be captured *before* `process_requires`
inlines the user's `(require agent.…)` form into the AST tree.
After inlining, the top-level `ESHKOL_REQUIRE_OP` no longer
exists and the scanner would miss it (`eshkol-run.cpp:2952-2965`):

```cpp
// #248: capture agent-FFI usage BEFORE process_requires expands
// (require agent.…) into the inlined module ASTs, after which the
// top-level require op is gone and the AST scanner can no longer
// tell agent-using programs from plain ones.
bool needs_agent_ffi = requires_agent_ffi(asts);

for (const auto &source_file : source_files) {
    std::filesystem::path source_path(source_file);
    std::string base_dir = source_path.parent_path().string();
    if (base_dir.empty()) base_dir = ".";
    process_requires(asts, base_dir, debug_mode);
    process_imports(asts, base_dir, debug_mode);
}
```

`requires_agent_ffi` (`eshkol-run.cpp:1620-1633`) walks the top-
level AST and returns true whenever any module name on a
`(require …)` form begins with `agent.`. The downstream effect
is that the link block at `eshkol-run.cpp:3363-3376` splices
`ESHKOL_HOST_AGENT_FFI_LINK_ARGS` into the link command only
when `needs_agent_ffi` is true — programs that never touch
agent.* are linked without libcurl, sqlite3, or pcre2, paying
no cost.

### 2.5 JIT path — DynamicLibrarySearchGenerator + explicit registration

For REPL JIT runs, the LLJIT instance attaches a
`DynamicLibrarySearchGenerator::GetForCurrentProcess` generator
to the main JITDylib (`lib/repl/repl_jit.cpp:534`). Because
eshkol-run was linked with `-force_load`/`--whole-archive`
against `libeshkol-agent-ffi.a`, every agent symbol — `qllm_http_*`,
`eshkol_sqlite_*`, `qllm_process_*`, `eshkol_http_server_*`,
`eshkol_ws_*`, `eshkol_regex_*`, `eshkol_watch_*`, the agent
crypto family, etc. — is live in the process's symbol table.
`dlsym(RTLD_DEFAULT, "qllm_http_get")` therefore returns the
real function pointer at the first `(extern :real qllm_http_get)`
lookup the JIT performs.

The macros `ESHKOL_AGENT_FFI_SYMBOL(name)` and
`ADD_OPTIONAL_AGENT_FFI_SYMBOL(name)` (`lib/repl/repl_jit.cpp:290`
and `:578`) additionally make every optional agent symbol's
*address* visible to the linker so the static archive isn't
dropped under LTO; they declare each `name` with weak
linkage so the JIT can fall back to a null-check if a build
omitted the underlying library. The full list of agent symbols
registered by hand (rather than discovered via dlsym) spans
`repl_jit.cpp:170-407` — that is the authoritative inventory
of which native symbols every JIT and AOT run has visible.

---

## 3. HTTP client (`agent.http`)

### 3.1 C runtime — `lib/agent/c/agent_http_client.c`

The HTTP client wraps libcurl's *easy* interface. The whole
translation unit is gated by `#ifdef ESHKOL_HAVE_LIBCURL`
(`agent_http_client.c:33`) so the build degrades cleanly when
libcurl is absent. Each call creates and destroys its own
`CURL*` handle — libcurl easy handles are documented as per-
thread, which lets multiple Eshkol threads call the FFI
concurrently without synchronisation
(`agent_http_client.c §threading` comment).

Common-options application is centralised in `apply_common_opts`
(`agent_http_client.c:101-115`) so TLS verification and
redirect policy stay identical across `get`/`post`/`post_json`:

```c
curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,        timeout_ms);
curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 10000L);
curl_easy_setopt(curl, CURLOPT_USERAGENT, "eshkol-agent/1.2");
curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
```

`CURLOPT_NOSIGNAL=1L` is the salient detail: libcurl normally
installs SIGPIPE / SIGALRM handlers on the calling thread which
collide with Eshkol's own subprocess timeout handler. Setting
NOSIGNAL keeps the thread signal mask untouched.

TLS verification is **on by default and not toggleable through
the API surface**. A caller that needs to talk to a self-signed
dev origin has to do so via `subprocess` + `curl --insecure`
until an explicit insecure-mode flag is wired
(`agent_http_client.c:110-114`).

The body buffer grows geometrically in `write_cb`
(`agent_http_client.c:81-96`) — initial capacity 4096, doubled
each time it runs short — keeping amortised cost O(n). A NUL
terminator is always written one byte past the live region so
the caller can use the body as a C string, but the binary-safe
length is reported via `qllm_http_response_body_len`.

Global libcurl init is reference-counted under a mutex
(`agent_http_client.c §g_init_refs`, lines 145-168): the first
`qllm_http_init` calls `curl_global_init(CURL_GLOBAL_DEFAULT)`
and subsequent ones increment the refcount; only the last
`qllm_http_shutdown` actually tears libcurl down. `qllm_http_get`,
`qllm_http_post`, and `qllm_http_post_json` all lazy-init when
`g_init_refs == 0` so callers that forget the explicit init
don't crash (lines 182, 209, 247).

#### Public ABI

| Symbol | Signature | File:line |
|---|---|---|
| `qllm_http_init` | `int32_t (void)` | `agent_http_client.c:145` |
| `qllm_http_shutdown` | `void (void)` | `agent_http_client.c:159` |
| `qllm_http_has_ssl` | `int32_t (void)` | `agent_http_client.c:170` |
| `qllm_http_get` | `qllm_http_response_t* (const char* url, int32_t timeout_ms)` | `agent_http_client.c:178` |
| `qllm_http_post` | `qllm_http_response_t* (url, headers[], hdr_count, body, body_len, timeout_ms)` | `agent_http_client.c:202` |
| `qllm_http_post_json` | `qllm_http_response_t* (url, body, auth_header, timeout_ms)` | `agent_http_client.c:242` |
| `qllm_http_response_status` | `int32_t (qllm_http_response_t*)` | `agent_http_client.c:283` |
| `qllm_http_response_body` | `const char* (qllm_http_response_t*)` | `agent_http_client.c:287` |
| `qllm_http_response_body_len` | `int64_t (qllm_http_response_t*)` | `agent_http_client.c:291` |
| `qllm_http_response_free` | `void (qllm_http_response_t*)` | `agent_http_client.c:295` |
| `qllm_http_error_string` | `const char* (int32_t code)` | `agent_http_client.c:302` |

Resource ownership: every successful call returns a heap-allocated
`qllm_http_response_t`. The caller must invoke `qllm_http_response_free`
or the malloc'd body buffer leaks (`agent_http_client.c:25-31`,
`:295`). The Scheme wrapper handles this automatically.

### 3.2 Eshkol surface — `lib/agent/http.esk`

| Function | Arity | Return | Wraps |
|---|---|---|---|
| `(http-init)` | 0 | `#t`/`#f` | `qllm_http_init` |
| `(http-shutdown)` | 0 | unspecified | `qllm_http_shutdown` |
| `(http-has-ssl?)` | 0 | `#t`/`#f` | `qllm_http_has_ssl` |
| `(http-get url [timeout-ms])` | 1 + optional | `(status . body)` or `#f` | `qllm_http_get` |
| `(http-post url headers body [timeout-ms])` | 3 + optional | `(status . body)` or `#f` | `qllm_http_post_json` (currently — see below) |
| `(http-request method url headers body [timeout-ms])` | 4 + optional | `(status . body)` or `#f` | dispatch on method |

The wrapper's contract is `(cons status-code body-string)` for
success, `#f` for a libcurl transport-level error
(`http.esk:117-127`). HTTP error statuses (4xx / 5xx) still
produce a `cons`; only complete failures to receive a response
return `#f`.

#### Security guard — `http-safe-string?`

Before any URL or header reaches libcurl, the Scheme layer runs
`http-safe-string?` (`http.esk:92-101`):

```scheme
(define (http-safe-string? s)
  (and (string? s)
       (not (string-contains s "\r"))
       (not (string-contains s "\n"))
       (not (string-contains s "\t"))
       (not (string-contains s "\x00"))
       (not (string-contains s "\x0b"))   ;; VT
       (not (string-contains s "\x0c"))   ;; FF
       (not (string-contains s "\x7f"))   ;; DEL
       ))
```

`http-check-url` (`http.esk:103-105`) and `http-check-headers`
(`http.esk:107-115`) call this for the URL and every header
name/value, raising `error` on rejection. The threat model is
documented in the surrounding comment: CRLF injection lets an
attacker append a second request to the same TCP stream that
many older proxies happily forward, and a literal NUL truncates
the URL at the C boundary and can route the request to the
wrong origin. Tab and the rest of the C0 controls are blocked
under the *Audit M8* tightening because some HTTP parsers
re-split header values on horizontal tab.

#### SSE streaming — deliberately unimplemented

The `qllm_http_stream_*` family at `agent_http_client.c:320-344`
returns `NULL` / `-1` / closes immediately. The Eshkol wrapper
(`http.esk:161-187`) honours the stub return values and
surfaces them as `#f` to the caller. Streaming SSE for chat
APIs is therefore **not yet a first-class agent FFI**; callers
that need it today use `subprocess` to drive `curl -N` and
parse `data:` lines themselves. See `MEMORY.md` *native HTTP
client (#234)* — SSE is a follow-up explicitly deferred from
the v1.2 cut.

#### Examples

```scheme
(require agent.http)

(http-init)

;; Simple GET.
(define resp (http-get "https://api.example.com/v1/health" 5000))
(display "status: ") (display (car resp)) (newline)
(display "body:   ") (display (cdr resp)) (newline)

;; JSON POST with Authorization.
(define body "{\"prompt\":\"hello\",\"max_tokens\":32}")
(define result
  (http-post "https://api.anthropic.com/v1/messages"
             '(("Authorization" . "Bearer sk-…")
               ("anthropic-version" . "2023-06-01"))
             body
             60000))

(http-shutdown)
```

A header value containing `"foo\r\nHost: evil.com"` would be
rejected at `http-check-headers` before any byte hits the wire.

---

## 4. HTTP server, WebSocket, Unix domain sockets (`agent.http_server`)

### 4.1 C runtime — `lib/agent/c/agent_http_server.c`

The HTTP server is intentionally minimal: single-connection,
blocking, loopback-only. It exists for OAuth PKCE callbacks,
MCP authentication handshakes, and `/health` endpoints — *not*
production web serving (`agent_http_server.c:8-9`,
`http_server.esk:7-13`). The bind socket address is hard-coded
to `INADDR_LOOPBACK`:

```c
addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  /* Bind to localhost only */
addr.sin_port = htons((uint16_t)port);
```

(`agent_http_server.c:68-69`). Programs that need to expose
themselves on a public interface must do so behind a real
reverse proxy — this surface is deliberately not pluggable.

Server slots are a fixed `MAX_HTTP_SERVERS = 4` table
(`agent_http_server.c:37-45`); each entry holds the listening
fd plus the currently-accepted client fd. `eshkol_http_server_create(0)`
asks the kernel to allocate a free port and reads it back via
`getsockname` so callers can advertise the bound port to a
peer (`agent_http_server.c:82-89`). Accept is poll-driven with
a caller-supplied timeout and a fixed 5-second per-client
`SO_RCVTIMEO` so a slow client can't pin the server forever
(`agent_http_server.c:117-129`).

Response framing is one shot — `eshkol_http_server_respond`
writes the status line, headers (`Content-Type`, `Content-Length`,
`Connection: close`), the body, and closes the socket
(`agent_http_server.c:148-178`). Recognised reason phrases:
200 OK / 301 Moved / 400 Bad Request / 404 Not Found / 500
Internal Server Error. Any other status emits "OK" — callers
that need a richer reason set should send the raw bytes via
the lower-level Unix-socket entry point.

#### WebSocket client — RFC 6455

`agent_http_server.c` also exposes a WebSocket client that
wraps an existing TCP fd after the HTTP/1.1 upgrade has
completed. The implementation supports text (opcode `0x1`)
and binary (`0x2`) frames, close (`0x8`), ping (`0x9`),
pong (`0xA`), and auto-replies to ping with pong
(`agent_http_server.c:384-387`). It does *not* support:

- Permessage-deflate (no extension negotiation).
- Fragmented frames (every frame must arrive in one piece).
- Masking with non-zero keys on outbound frames (the all-
  zero mask is per-RFC valid but unusual; see comment at
  `agent_http_server.c:278-285`).

The handle table is fixed at `MAX_WS_HANDLES = 8`
(`agent_http_server.c:224`). The wrapper module
`http_server.esk` documents this as suitable for MCP-over-WS
and voice streaming; for production WS work, link libcurl
7.86+ or libwebsockets.

#### Unix domain socket — IDE IPC

`eshkol_unix_socket_connect(path)` (`agent_http_server.c:196-211`)
connects to a `SOCK_STREAM` AF_UNIX socket at `path` and
returns the raw fd. The only intended consumer today is the
VS Code `VSCODE_IPC_HOOK` IPC channel that the Eshkol
extension speaks to. Combine with `eshkol_ws_wrap_fd` for an
IPC channel that exchanges WebSocket-framed messages.

#### Public ABI

| Symbol | Signature | File:line |
|---|---|---|
| `eshkol_http_server_create` | `int64_t (int32_t port)` | `agent_http_server.c:52` |
| `eshkol_http_server_port` | `int32_t (int64_t handle)` | `agent_http_server.c:97` |
| `eshkol_http_server_accept` | `int32_t (handle, buf, buf_size, timeout_ms)` | `agent_http_server.c:110` |
| `eshkol_http_server_respond` | `void (handle, status, ct, body)` | `agent_http_server.c:148` |
| `eshkol_http_server_close` | `void (int64_t handle)` | `agent_http_server.c:183` |
| `eshkol_unix_socket_connect` | `int64_t (const char* path)` | `agent_http_server.c:196` |
| `eshkol_ws_wrap_fd` | `int64_t (int32_t fd)` | `agent_http_server.c:239` |
| `eshkol_ws_send_text` | `int32_t (handle, data, len)` | `agent_http_server.c:254` |
| `eshkol_ws_send_binary` | `int32_t (handle, data, len)` | `agent_http_server.c:290` |
| `eshkol_ws_receive` | `int32_t (handle, buf, buf_size, frame_type*, timeout_ms)` | `agent_http_server.c:320` |
| `eshkol_ws_close` | `void (int64_t handle)` | `agent_http_server.c:393` |

### 4.2 Eshkol surface — `lib/agent/http_server.esk`

| Function | Wraps |
|---|---|
| `(http-server-create port)` | `eshkol_http_server_create` |
| `(http-server-port handle)` | `eshkol_http_server_port` |
| `(http-server-accept handle buffer-size timeout-ms)` | `eshkol_http_server_accept` |
| `(http-server-respond handle status content-type body)` | `eshkol_http_server_respond` |
| `(http-server-close handle)` | `eshkol_http_server_close` |
| `(unix-socket-connect path)` | `eshkol_unix_socket_connect` |
| `(ws-wrap-fd fd)` | `eshkol_ws_wrap_fd` |
| `(ws-send-text handle data)` | `eshkol_ws_send_text` |
| `(ws-send-binary handle data)` | `eshkol_ws_send_binary` |
| `(ws-receive handle buffer-size timeout-ms)` | `eshkol_ws_receive` |
| `(ws-close handle)` | `eshkol_ws_close` |

`(http-server-accept …)` returns the raw request bytes as a
string with `"METHOD PATH\r\n…headers…\r\n\r\n…body"` shape
(`http_server.esk:67-80`). The Eshkol code is expected to
parse this — the runtime deliberately does not.

`(ws-receive …)` returns `(cons frame-type payload)` where
frame-type is one of the `WS-FRAME-*` constants. Ping frames
have already been answered by the C side before the call
returns; the caller may still observe and log them.

#### Example: OAuth PKCE callback

```scheme
(require agent.http_server)

(define server (http-server-create 0))     ;; ephemeral port
(define port   (http-server-port server))

;; Tell the browser where to hit us back.
(display (string-append "Open: http://127.0.0.1:"
                        (number->string port) "/callback"))
(newline)

;; Block up to 5 minutes waiting for one callback hit.
(let ((req (http-server-accept server 4096 300000)))
  (cond
    ((not req)       (display "accept error"))
    ((string=? req "") (display "timed out"))
    (else
      (http-server-respond server 200 "text/html"
        "<h1>You may close this window.</h1>")
      (display req))))

(http-server-close server)
```

---

## 5. SQLite (`agent.sqlite`)

### 5.1 C runtime — `lib/agent/c/agent_sqlite.c`

`agent_sqlite.c` is a thin handle-table over the `sqlite3`
public C API. Database handles are kept in a 64-slot table,
prepared statements in a 512-slot table
(`agent_sqlite.c:20-26`). The slot allocator scans forward
from `g_next_db`/`g_next_stmt`, wrapping around when the
first half is full, so handle IDs are stable for the lifetime
of the open object but reuse after close
(`agent_sqlite.c:28-46`).

Every database is opened with WAL journaling and a 5-second
busy timeout (`agent_sqlite.c:62-85`):

```c
sqlite3_open_v2(path, &db,
                SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                SQLITE_OPEN_FULLMUTEX, NULL);
sqlite3_exec(db, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
sqlite3_busy_timeout(db, 5000);
```

`SQLITE_OPEN_FULLMUTEX` lets concurrent Eshkol threads share
the same `sqlite3*` without external synchronisation; WAL
mode is essential for the pattern where one writer and many
readers operate against an agent's local store. There is no
connection pool: the agent is expected to keep a single
database handle alive for the duration of work.

#### The v1.2 dynamic column-text fix

The historical bug was that `eshkol_sqlite_column_text` had no
way to communicate the true payload size to the caller — the
old wrapper passed a fixed 8 KiB buffer and silently truncated
larger values, after which JSON parsing of the partial result
threw "invalid JSON" deep in the agent (`MEMORY.md §session
persistence unblockers (#236/#246/#247)`). The fix added a
companion function that returns the exact byte length without
copying:

```c
/* lib/agent/c/agent_sqlite.c:183 */
int64_t eshkol_sqlite_column_bytes(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    /* Note: SQLite docs require column_text() to be called before
     * column_bytes() if the underlying value isn't already TEXT,
     * because SQLite may need to convert the type. Calling text
     * first is harmless when the column is already TEXT. */
    sqlite3_column_text(stmt, index);
    return (int64_t)sqlite3_column_bytes(stmt, index);
}
```

The Scheme wrapper `(sqlite-column-text stmt idx)` then
queries `sqlite-column-bytes-raw` first, allocates a buffer
of exactly that size, and copies the payload in one shot —
2 MB session JSON round-trips losslessly
(`sqlite.esk:92-107`).

`eshkol_sqlite_column_text` itself also encodes an overflow
signal: if the caller's buffer was too small, the return is
`-(len + 1)` rather than the truncated count, so a wrapper
that hasn't migrated to the new sizing path can still detect
truncation and retry rather than silently drop bytes
(`agent_sqlite.c:213-224`).

#### Public ABI

| Symbol | Signature | File:line |
|---|---|---|
| `eshkol_sqlite_open` | `int64_t (const char* path)` | `agent_sqlite.c:62` |
| `eshkol_sqlite_close` | `void (int64_t handle)` | `agent_sqlite.c:87` |
| `eshkol_sqlite_exec` | `int (int64_t handle, const char* sql)` | `agent_sqlite.c:95` |
| `eshkol_sqlite_prepare` | `int64_t (int64_t db, const char* sql)` | `agent_sqlite.c:109` |
| `eshkol_sqlite_step` | `int (int64_t stmt)` | `agent_sqlite.c:125` |
| `eshkol_sqlite_reset` | `int (int64_t stmt)` | `agent_sqlite.c:131` |
| `eshkol_sqlite_finalize` | `void (int64_t stmt)` | `agent_sqlite.c:137` |
| `eshkol_sqlite_bind_text` | `int (stmt, idx, text)` | `agent_sqlite.c:149` |
| `eshkol_sqlite_bind_int` | `int (stmt, idx, int64_t value)` | `agent_sqlite.c:155` |
| `eshkol_sqlite_bind_double` | `int (stmt, idx, double value)` | `agent_sqlite.c:161` |
| `eshkol_sqlite_bind_null` | `int (stmt, idx)` | `agent_sqlite.c:167` |
| `eshkol_sqlite_column_bytes` | `int64_t (stmt, idx)` | `agent_sqlite.c:183` |
| `eshkol_sqlite_column_text` | `int (stmt, idx, buf, buf_size)` | `agent_sqlite.c:194` |
| `eshkol_sqlite_column_int` | `int64_t (stmt, idx)` | `agent_sqlite.c:227` |
| `eshkol_sqlite_column_double` | `double (stmt, idx)` | `agent_sqlite.c:233` |
| `eshkol_sqlite_column_count` | `int (stmt)` | `agent_sqlite.c:239` |
| `eshkol_sqlite_column_name` | `int (stmt, idx, buf, buf_size)` | `agent_sqlite.c:245` |
| `eshkol_sqlite_column_type` | `int (stmt, idx)` | `agent_sqlite.c:259` |
| `eshkol_sqlite_last_error` | `int (db, buf, buf_size)` | `agent_sqlite.c:269` |
| `eshkol_sqlite_last_insert_rowid` | `int64_t (db)` | `agent_sqlite.c:281` |
| `eshkol_sqlite_changes` | `int (db)` | `agent_sqlite.c:287` |

`sqlite3_bind_text` uses `SQLITE_TRANSIENT` so SQLite copies the
text immediately; the Eshkol-side string can be freed or rebound
before `sqlite_step` runs (`agent_sqlite.c:152`).

### 5.2 Eshkol surface — `lib/agent/sqlite.esk`

The wrapper exports both the low-level prepared-statement API
and two convenience macros for resource management:

```scheme
(with-db path
  (lambda (db)  …))         ;; close on exit, even on exception

(with-statement db sql
  (lambda (stmt) …))         ;; finalize on exit
```

Both use `dynamic-wind` (`sqlite.esk:129-143`) so the close /
finalize fires on normal return AND on every escape from the
body — including `error`, continuation jumps, and exceptions.

#### `sqlite-exec-safe` — the "blocks the dumbest injection" filter

`sqlite-exec` takes raw SQL with the explicit documented
expectation that the caller is providing trusted literal text
(`sqlite.esk:48-61`):

```
SECURITY (#195): sqlite-exec takes raw SQL — any caller that
builds the statement by concatenating user input is vulnerable
to SQL injection. Prefer sqlite-prepare + sqlite-bind-* +
sqlite-step for anything with user-controlled values.
```

`sqlite-exec-safe` (`sqlite.esk:63-77`) refuses statements
containing any of `;` (multi-statement), `--` (line comment),
or `/*` (block comment). The documented contract is that this
is **not a substitute for prepared statements** — it merely
blocks textbook payloads on legacy call sites that cannot
migrate to bind-* immediately. Future versions may further
restrict it to a whitelist of statement kinds.

#### Error model

Functions that take a handle return `-1` on bad-handle. Status
results from sqlite3_step are reported verbatim: `SQLITE_ROW`
(100) when a row was produced, `SQLITE_DONE` (101) at the end
of a query, anything else is an error (call `sqlite-last-error`
for the message string). The Eshkol wrappers do not raise on
SQLite-level errors — they return the integer code so the
caller can decide whether to abort or retry.

#### Example: insert + query

```scheme
(require agent.sqlite)

(with-db "/tmp/agent.db"
  (lambda (db)
    (sqlite-exec db
      "CREATE TABLE IF NOT EXISTS notes
       (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    ;; Bound parameters keep this injection-safe.
    (with-statement db "INSERT INTO notes(text) VALUES (?)"
      (lambda (stmt)
        (sqlite-bind-text stmt 1 "first note")
        (sqlite-step stmt)))
    (with-statement db "SELECT id, text FROM notes ORDER BY id"
      (lambda (stmt)
        (let loop ()
          (when (= (sqlite-step stmt) SQLITE_ROW)
            (display (list (sqlite-column-int  stmt 0)
                           (sqlite-column-text stmt 1)))
            (newline)
            (loop)))))))
```

---

## 6. Subprocess (`agent.subprocess`)

### 6.1 C runtime — `lib/agent/c/agent_subprocess.c`

Subprocess is the largest single agent FFI module (1524 lines).
It exists because driving external tools — `git`, `lean`,
`cmake`, `rustc`, model CLIs — is the single most common
agentic side-effect, and the canonical `system(3)` /
`popen(3)` interface is unsafe (shell injection) and slow
(`/bin/sh -c` startup is 3-5 ms per call on macOS).

#### The three spawn entrypoints

```c
qllm_process_spawn_shell  /* always /bin/sh -c (or cmd /c on Windows) */
qllm_process_spawn_argv   /* always execvp on a tab-packed argv list, no shell */
qllm_process_spawn        /* legacy compatibility — bypasses sh for safe inputs */
```

The boundary contract is documented at
`agent_subprocess.c:80-93`. Callers that interpolate
user-controlled values are required to use `argv`; callers
that need shell grammar (pipes, redirections, globs) use
`shell` and quote interpolated values themselves.

`qllm_process_spawn` is the historical command-string entry
that mediates between the two: if the command string contains
no shell metacharacters and no C0 control bytes other than
space/tab, it splits on whitespace and runs `execvp` directly
(`agent_subprocess.c §command_is_shell_safe`, lines 106-131).
The metacharacter list is `| & ; < > * ? $ ` \ ' " ~ ( ) { } [ ] ! #`,
plus a refusal of any C0 byte except space/tab (audit M5,
which closed VT/FF/ESC/DEL holes) and any 8th-bit byte. Skipping
`/bin/sh` saves 3-5 ms per call.

#### posix_spawn vs fork+exec

Inside `eshkol-run`, the host address space is ~200 MB plus a
worker thread pool. A naive `fork()` would copy-on-write the
page tables and snapshot every libsystem thread, costing
10-15 ms per spawn (`agent_subprocess.c:756-773`). The hot
path uses `posix_spawn` instead:

```c
spawn_errno = posix_spawnp(&pid, argv[0], &fa, attr_ptr,
                           argv, env_to_pass);
```

Darwin's `posix_spawn` is vfork-like and doesn't touch page
tables. The chdir branch (when cwd is non-empty and not `.`)
falls back to `fork+exec` because `posix_spawn_file_actions_addchdir_np`
is macOS 14.0+ only and the agent still ships on 13.x
(`agent_subprocess.c §no_chdir`, lines 427, 781-783). Treating
`cwd="."` as "no chdir needed" gives the fast path
automatically.

On macOS the spawn additionally uses `POSIX_SPAWN_CLOEXEC_DEFAULT`,
which makes every fd not explicitly preserved in the file
actions close-on-exec — saving six `addclose` entries per
spawn and an implicit fd-table walk in the child
(`agent_subprocess.c:441-461`).

#### Pipe leak fix — v1.2.1

The original spawn code used short-circuit evaluation:

```c
if (pipe(stdin_pipe) != 0 ||
    pipe(stdout_pipe) != 0 ||
    pipe(stderr_pipe) != 0) return NULL;
```

When the first call succeeded and the second failed, the
two stdin-pipe fds leaked — a classical partial-success
resource leak (HARDENING.md §`#182`). The current code gates
every `pipe()` separately so cleanup closes exactly the fds
that actually opened (`agent_subprocess.c:399-418` for shell
form, `:730-753` for argv form). The pattern is identical on
both paths.

#### Pipe drainer — single blocking waitpid + kqueue

The historical implementation spawned two pthread drainers per
`process-wait` (one for stdout, one for stderr) plus a SIGALRM
timer for the timeout. Inside eshkol-run that cost 5-10 ms per
spawn even when the child itself was sub-millisecond — entirely
pthread setup + signal-handler bookkeeping. The current
implementation on macOS / FreeBSD uses kqueue with
`EVFILT_PROC NOTE_EXIT` joined to `EVFILT_READ` on the two
pipe fds (`agent_subprocess.c:1300-1378`):

```c
EV_SET(&evs[nev++], proc->pid, EVFILT_PROC,
       EV_ADD | EV_ENABLE | EV_ONESHOT, NOTE_EXIT, 0, NULL);
EV_SET(&evs[nev++], proc->stdout_fd, EVFILT_READ,
       EV_ADD | EV_ENABLE, 0, 0, NULL);
EV_SET(&evs[nev++], proc->stderr_fd, EVFILT_READ,
       EV_ADD | EV_ENABLE, 0, 0, NULL);
kevent(kq, evs, nev, NULL, 0, NULL);
```

The parent thread blocks in one syscall and wakes for any
of: child exit, stdout-readable, stderr-readable. On Linux
the path falls back to the pthread drainer pattern at
`:1380-1450` because Linux pidfd_open arrived only in 5.3 —
modern Linux distros are already past that and the fallback
is the simpler-but-slower path.

Per-stream drain cap is 16 MiB
(`agent_subprocess.c:1268-1270`). Past the cap, the drainer
keeps reading from the pipe so the child doesn't deadlock on
write but drops the bytes — agents that capture sub-tool
output should size with this ceiling in mind.

#### Environment scrubbing — audit C6

Every spawn passes a filtered `environ` to the child. The
filter strips the dynamic-linker injection variables that
would otherwise let a compromised shell profile inject a
library into every git/lean/python/cmake the agent launches
(`agent_subprocess.c §eshkol_scrub_environ`, lines 317-382):

```
LD_PRELOAD, LD_AUDIT, LD_LIBRARY_PATH, LD_BIND_NOW, LD_BIND_NOT,
LD_DEBUG, LD_PROFILE, LD_USE_LOAD_BIAS,
DYLD_INSERT_LIBRARIES, DYLD_LIBRARY_PATH, DYLD_FALLBACK_LIBRARY_PATH,
DYLD_FRAMEWORK_PATH, DYLD_FALLBACK_FRAMEWORK_PATH,
DYLD_FORCE_FLAT_NAMESPACE, DYLD_PRINT_LIBRARIES,
DYLD_PRINT_LIBRARIES_POST_LAUNCH, DYLD_PRINT_ENV,
DYLD_PRINT_BINDINGS, DYLD_BIND_AT_LAUNCH, DYLD_IMAGE_SUFFIX
```

PATH / HOME / USER / LANG still pass through. The filter is
cached under a mutex (`g_env_cache`) so repeated spawns don't
redo the linear walk of `environ` each time. On the fork+exec
fallback path, `eshkol_unset_env_injection_vars` performs the
equivalent scrub in the child after fork but before exec
(`agent_subprocess.c:264-280`).

#### Resource limits — audit H7

`eshkol_apply_subproc_rlimits` (`agent_subprocess.c:204-258`)
applies four `setrlimit(2)` caps inside the fork+exec child
before `execvp`:

```
RLIMIT_AS     virtual memory    default 4 GiB  (ESHKOL_SUBPROC_MEM_MB)
RLIMIT_CPU    CPU seconds       default 300 s  (ESHKOL_SUBPROC_CPU_SEC)
RLIMIT_NOFILE file descriptors  default 1024   (ESHKOL_SUBPROC_NOFILE)
RLIMIT_NPROC  processes per UID default 512    (ESHKOL_SUBPROC_NPROC)
```

The coverage gap noted in the source comment is that
`posix_spawn` has no `posix_spawnattr` analogue for rlimits;
the hot posix_spawn path therefore skips them. Callers that
need guaranteed limits on every child should `setrlimit` the
eshkol-run process itself at startup so every child inherits
identical caps via the normal POSIX inheritance rule.

#### Windows cmdline buffer — audit #193

`CreateProcessA` mutates its lpCommandLine buffer in place.
The historical code passed a 4096-byte stack buffer and
silently `snprintf`-truncated longer commands, producing a
valid-looking but malformed command line (HARDENING.md `#193`
HIGH). The current code measures `"cmd /c " + command` ahead
of time, rejects ≥ 32768 (the Windows cmdline limit), and
heap-allocates the exact size — no silent truncation, no
overflow (`agent_subprocess.c:589-622`).

#### Public ABI (POSIX + Windows)

| Symbol | Returns | File:line |
|---|---|---|
| `qllm_process_spawn` | `eshkol_subprocess_t*` | `agent_subprocess.c:648` |
| `qllm_process_spawn_shell` | `eshkol_subprocess_t*` | `agent_subprocess.c:654` |
| `qllm_process_spawn_argv` | `eshkol_subprocess_t*` | `agent_subprocess.c:896` |
| `qllm_process_spawn_argv_flags` | `eshkol_subprocess_t*` | `agent_subprocess.c:712` |
| `qllm_process_write_stdin` | `int64_t bytes_written` | `agent_subprocess.c:905` |
| `qllm_process_close_stdin` | `void` | `agent_subprocess.c:929` |
| `qllm_process_read_stdout` | `int64_t bytes_read` | `agent_subprocess.c:942` |
| `qllm_process_read_stderr` | `int64_t bytes_read` | `agent_subprocess.c:956` |
| `qllm_process_read_all_stdout` | `char* (caller frees via qllm_process_free_buffer)` | `agent_subprocess.c:1037` |
| `qllm_process_read_all_stderr` | `char*` | `agent_subprocess.c:1049` |
| `qllm_process_wait` | `int32_t (0=exited, 1=timeout, -1=error)` | `agent_subprocess.c:1263` |
| `qllm_process_running` | `int32_t (1=running, 0=exited)` | `agent_subprocess.c:1461` |
| `qllm_process_exit_code` | `int32_t` | `agent_subprocess.c:1467` |
| `qllm_process_pid` | `int64_t` | `agent_subprocess.c:1496` |
| `qllm_process_kill` | `void (proc, signal)` | `agent_subprocess.c:1477` |
| `qllm_process_destroy` | `void` | `agent_subprocess.c:1501` |
| `qllm_process_free_buffer` | `void (char* buf)` | `agent_subprocess.c:660` |

`qllm_process_wait` returns 0/1/-1 not the exit code — the
caller reads the code separately. This is corrected from the
historical contract that returned `exit_code` directly: a
child exiting with status 1 (test failure, bad-proof, etc.)
collided with the "1=timeout" sentinel, so every legitimate
non-zero exit was misreported as a 124 timeout (Noesis v5
audit BUG A, see comment at `:1249-1259`).

#### Eshkol surface — `lib/agent/subprocess.esk`

| Function | Wraps | Notes |
|---|---|---|
| `(process-spawn command cwd)` | `qllm_process_spawn` | legacy command-string |
| `(process-spawn-nostdin command cwd)` | same with `flags=PROCESS_SPAWN_STDIN_NULL` | wires child's stdin to `/dev/null` |
| `(process-spawn-shell command cwd)` | `qllm_process_spawn_shell` | always `/bin/sh -c` |
| `(process-spawn-shell-nostdin command cwd)` | same + stdin-null | |
| `(process-spawn-argv argv cwd)` | `qllm_process_spawn_argv` | **safe** — no shell |
| `(process-spawn-argv-nostdin argv cwd)` | flags-variant | |
| `(process-write-stdin proc data)` | `qllm_process_write_stdin` | |
| `(process-close-stdin proc)` | `qllm_process_close_stdin` | |
| `(process-read-stdout proc max-bytes)` | `qllm_process_read_all_stdout` | returns string |
| `(process-read-stderr proc max-bytes)` | `qllm_process_read_all_stderr` | |
| `(process-wait proc timeout-ms)` | `qllm_process_wait` | 0 / 1 / -1 |
| `(process-running? proc)` | `qllm_process_running` | |
| `(process-exit-code proc)` | `qllm_process_exit_code` | |
| `(process-pid proc)` | `qllm_process_pid` | live PID for trace/observability |
| `(process-kill proc [signal])` | `qllm_process_kill` | default SIGTERM (15) |
| `(process-destroy proc)` | `qllm_process_destroy` | frees handle |
| `(run-command command [cwd] [timeout-ms])` | shell convenience | returns exit code |
| `(run-command-capture command [cwd] [timeout-ms] [max-output])` | shell convenience | returns alist |
| `(run-argv argv [cwd] [timeout-ms])` | argv convenience | returns exit code |
| `(run-argv-capture argv [cwd] [timeout-ms] [max-output])` | argv convenience | returns alist |

`run-command-capture` and `run-argv-capture` return an
association list of `((exit-code . N) (stdout . "…") (stderr . "…"))`.
On timeout the exit code is 124 (matches GNU coreutils
`timeout(1)` convention) and stderr is suffixed with
`"\n[Process timed out after Ns]"`
(`subprocess.esk:228-244`, `:283-294`).

`process-spawn-argv` enforces the argv invariants via
`process-argv-check-args` (`subprocess.esk:85-97`): every
element must be a string, contain no literal tab byte (which
would be re-split by the tab-separated wire format) and no
NUL byte (which would truncate at the C boundary).

#### Examples

```scheme
(require agent.subprocess)

;; Argv form — safe interpolation of a user-controlled filename.
(define result (run-argv-capture
                 (list "lean" "--quiet" user-filename)
                 (current-directory) 60000))
(display (cdr (assv 'exit-code result))) (newline)
(display (cdr (assv 'stdout    result)))

;; Streaming child — write commands, drain output as it arrives.
(define proc (process-spawn-argv (list "python3" "-u" "repl.py") "."))
(process-write-stdin proc "print(1+1)\n")
(process-close-stdin proc)
(process-wait proc 30000)
(display (process-read-stdout proc 65536))
(process-destroy proc)
```

---

## 7. Filesystem watch — two surfaces

The agent FFI exposes filesystem watching at **two distinct
layers**, and the choice matters.

### 7.1 Native kqueue / inotify — `lib/agent/c/agent_watch.c`

`agent_watch.c` is the proper native watcher. On macOS / FreeBSD
it uses kqueue with `EVFILT_VNODE` and the
`NOTE_WRITE|NOTE_DELETE|NOTE_RENAME|NOTE_ATTRIB` flags
(`agent_watch.c:74-92`); on Linux it uses inotify with
`IN_CREATE|IN_DELETE|IN_MODIFY|IN_MOVED_FROM|IN_MOVED_TO`
(`agent_watch.c:111-122`). Recursive mode walks the directory
tree at start time and adds a watch for each subdirectory
(`agent_watch.c:94-109`, `:124-139`); new directories created
after start are **not** auto-watched.

Events accumulate in an internal ring buffer with capacity 64
(`agent_watch.c:40-71`). The agent calls `eshkol_watch_poll`
in its event loop and receives one "type\tpath" string per
poll, or 0 when the buffer is empty and the OS has no new
events. Overflowing the ring drops oldest events
(`agent_watch.c:66-72`).

Public ABI:

| Symbol | Signature | File:line |
|---|---|---|
| `eshkol_watch_start` | `int64_t (const char* path, int32_t recursive)` | `agent_watch.c:148` |
| `eshkol_watch_poll` | `int32_t (handle, buf, buf_size)` | `agent_watch.c:189` |
| `eshkol_watch_stop` | `void (int64_t handle)` | `agent_watch.c:267` |

The JIT symbol table registers these at `repl_jit.cpp:914-916`,
so they are available under `eshkol-repl` and `eshkol-run -e`.
**No `.esk` extern wrapper currently exposes these symbols** —
they are reachable from Scheme only by writing a fresh
`(extern …)` declaration. This is an explicit limitation: the
high-level wrapper at `lib/agent/fs-watch.esk` was written
before the native runtime existed and still routes through
`fswatch` / `inotifywait` subprocesses.

### 7.2 Polling builtins — `fs-watch-native` / `fs-watch-recursive` / `fs-watch-poll`

Separately, `lib/core/system_builtins.c §eshkol_builtin_fs_watch_start_v`
(line 1911) implements a *poll-based* watcher that diffs
mtime + size of the watched path on each call. It is exposed
to user code via the codegen builtins `fs-watch-native`,
`fs-watch-recursive`, `fs-watch-poll`, and `fs-unwatch`
(`lib/backend/llvm_codegen.cpp:12197-12200` and the
function_return_types entries above them). This path does
*not* use kqueue / inotify and is independent of
`agent_watch.c`.

The watcher table is fixed-size; events are derived from
mtime/size diff between two consecutive `fs-watch-poll`
calls, emitted as `"create"`, `"change"`, or `"delete"`
(`system_builtins.c:1962-1969`). This is suitable for single
files or coarse-grained polling but not for high-frequency
directory monitoring — for that, use the native watcher
above.

### 7.3 Subprocess-based fallback — `lib/agent/fs-watch.esk`

`fs-watch.esk` provides `fs-watch-start`, `fs-watch-poll`,
`fs-watch-stop` as subprocess wrappers (`fswatch -1
--recursive` on macOS, `inotifywait -r -m` on Linux, find-newer
fallback). Quoting routes through `shell-quote` to escape the
directory path. This is the **legacy path** preserved for
existing agents; new code should prefer the native runtime
symbols.

---

## 8. Regex (`agent.regex`)

### 8.1 C runtime — `lib/agent/c/agent_regex.c`

PCRE2 with UTF-8 (`PCRE2_CODE_UNIT_WIDTH=8`) is the default
compile-time width (`agent_regex.c:10-11`,
`CMakeLists.txt:1819`). Compiled patterns are stored in a
256-slot handle table (`agent_regex.c:21-43`) with the same
wrap-around allocator the other agent modules use.

Flags surfaced to Eshkol:

```
1 PCRE2_CASELESS   2 PCRE2_MULTILINE   4 PCRE2_DOTALL
```

(`agent_regex.c:57-60`). UTF mode is always on
(`agent_regex.c:57`).

#### ReDoS hardening — audit `#195`

A pattern like `"(a+)+$"` matched against a long subject of
`a`s plus one unmatched trailing character triggers exponential
backtracking and pins the calling thread. PCRE2 supports
`pcre2_set_match_limit` and `pcre2_set_depth_limit` to bound
this, but they must be applied via a `pcre2_match_context` —
otherwise the engine has no upper bound.

`agent_regex.c §get_match_context` (line 91) builds a
process-global, lazily-allocated match context with conservative
caps:

```c
pcre2_set_match_limit(s_ctx, 10000000);    /* ≈ 10 ms on modern CPUs */
pcre2_set_depth_limit(s_ctx, 100000);
```

Every `pcre2_match` and `pcre2_substitute` in the file passes
this context (`agent_regex.c:115, 151, 234, 251`). The context
is allocated once and never freed — process-lifetime is fine
and reusing one context is documented as thread-safe (the
context is read-only during matching).

#### Public ABI

| Symbol | Signature | File:line |
|---|---|---|
| `eshkol_regex_compile` | `int64_t (const char* pattern, int flags)` | `agent_regex.c:54` |
| `eshkol_regex_match` | `int (handle, subject, match_buf, buf_size)` | `agent_regex.c:106` |
| `eshkol_regex_match_all` | `int (handle, subject, buf, buf_size, max)` | `agent_regex.c:135` |
| `eshkol_regex_replace` | `int (handle, subject, replacement, out, out_size)` | `agent_regex.c:174` |
| `eshkol_regex_free` | `void (int64_t handle)` | `agent_regex.c:200` |
| `eshkol_regex_match_groups_count` | `int (handle, subject)` | `agent_regex.c:227` |
| `eshkol_regex_match_groups` | `int (handle, subject, out_buf, buf_size)` | `agent_regex.c:243` |
| `eshkol_regex_named_group_number` | `int (handle, name)` | `agent_regex.c:287` |

`eshkol_regex_match_all` and `eshkol_regex_match_groups`
return their results as a flat NUL-separated buffer
(`"match0\0match1\0…"`) so the Scheme wrapper can split on
NUL bytes into a list of strings (`agent_regex.c:218-222`).
Unset optional groups (e.g. `(a(b)?c)` where the `b` was
absent) are emitted as empty strings, not skipped
(`agent_regex.c:268-271`).

### 8.2 Eshkol surface — `lib/agent/regex.esk`

| Function | Wraps |
|---|---|
| `(regex-compile pattern [flags])` | `eshkol_regex_compile` |
| `(regex-match handle subject)` | first match or `#f` |
| `(regex-match? handle subject)` | bool |
| `(regex-match-all handle subject [max])` | list of strings |
| `(regex-replace handle subject replacement)` | new string |
| `(regex-free handle)` | release |
| `(regex-match-groups handle subject)` | list of (full . groups) or `#f` |
| `(regex-group match-groups idx)` | indexed access |
| `(regex-named-group-number handle name)` | int (1-based) or -1 |

Flags constants:
`REGEX_CASELESS = 1`, `REGEX_MULTILINE = 2`, `REGEX_DOTALL = 4`.

`regex-match-groups` returns a list whose head is the full
match and whose tail elements are the captured subgroups in
order (`regex.esk:87-103`). The walk over the NUL-separated
buffer is bounded by both `count` and `buf-size` so a
malformed result (or a buffer overflow inside the C side)
returns whatever was readable rather than going past the
intended payload.

---

## 9. Crypto (`agent.crypto`)

### 9.1 C runtime — `lib/agent/c/agent_crypto.c`

Three primitives: HMAC-SHA256, SHA256, secure random bytes.
The vendor backend is selected at compile time:

```c
#ifdef __APPLE__
#include <CommonCrypto/CommonHMAC.h>
#include <Security/SecRandom.h>
#define HAVE_COMMONCRYPTO 1
#else
#include <openssl/hmac.h>
#include <openssl/rand.h>
#define HAVE_OPENSSL 1
#endif
```

(`agent_crypto.c:14-24`). Random bytes use `SecRandomCopyBytes`
on macOS or `RAND_bytes` on Linux — both cryptographically
secure CSPRNG interfaces. HMAC and SHA256 produce 32-byte
digests, hex-encoded into a 65-byte output buffer (64 hex
chars + NUL).

Public ABI:

| Symbol | Signature |
|---|---|
| `eshkol_hmac_sha256` | `int (key, key_len, data, data_len, out_hex, out_size)` |
| `eshkol_sha256` | `int (data, data_len, out_hex, out_size)` |
| `eshkol_random_bytes` | `int (char* buf, size_t len)` |
| `eshkol_random_hex` | `int (char* buf, size_t hex_len)` |

All return 0 on success, non-zero on failure.

### 9.2 Eshkol surface — `lib/agent/crypto.esk`

| Function | Notes |
|---|---|
| `(sha256 data)` | returns 64-char hex string or `#f` |
| `(hmac-sha256 key data)` | returns 64-char hex string or `#f` |
| `(random-bytes len)` | returns binary string |
| `(random-hex hex-len)` | returns hex string of `hex-len` chars |
| `(uuid-v4)` | spec-conformant UUID v4 string |
| `(base64url-encode data)` | RFC 4648 §5 — replaces `+/` with `-_`, strips `=` |
| `(base64url-decode url-str)` | inverse |

`uuid-v4` derives 16 random bytes via `random-hex`, then
patches the version (nibble 12 = `4`) and variant (nibble 16
in `{8,9,a,b}`) bits in place to comply with RFC 4122
(`crypto.esk:52-73`).

`base64url-encode` and `base64url-decode` route through
`base64-encode-string` / `base64-decode-string` from
`core.data.base64`. The MEMORY.md note *Bug HH —
base64url-encode wrong API entry* (`crypto.esk:118-126`,
`:131-138`) documents that the previous version called
`base64-encode` directly with a string, which expected a list
of bytes and aborted with `cdr: argument is not a pair` —
the entrypoints `*-string` exist precisely for the
string-in / string-out symmetry agent callers need.

---

## 10. Security model — every guard, with citation

The agent FFI uses defense in depth. Each guard is
implemented in a specific place; this section lists them by
threat class.

### 10.1 Shell injection — eliminated by argv

| Surface | Mitigation | Implementation |
|---|---|---|
| `process-spawn-argv` | direct `execvp`, no shell | `agent_subprocess.c §qllm_process_spawn_argv_flags` (line 712) |
| `process-spawn` (auto-detect) | drops shell when no metacharacters | `agent_subprocess.c §command_is_shell_safe` (line 106) |
| Tab in argv element | rejected at wrapper | `subprocess.esk §process-argv-check-args` (line 85) |
| NUL in argv element | rejected at wrapper | same, line 94 |

The argv form is the recommended path for any caller that
interpolates user-controlled values. The shell form is kept
for callers that genuinely need pipes / redirection / globbing,
on the explicit understanding that they own the quoting.

### 10.2 SQL injection — bound parameters or `safe` filter

| Surface | Mitigation | Implementation |
|---|---|---|
| `sqlite-prepare` + `sqlite-bind-*` | parameterised queries — SQLite binds the value, no interpolation | `agent_sqlite.c §eshkol_sqlite_bind_*` (lines 149-170) |
| `sqlite-exec-safe` | rejects `;`, `--`, `/*` | `sqlite.esk §sqlite-exec-safe` (line 63) |
| `sqlite-exec` (raw) | **DOCUMENTED as risky** | `sqlite.esk:48-61` — caller responsibility |

`sqlite-exec-safe` is explicitly *not* a substitute for prepared
statements — it blocks the dumbest injection attempts on call
sites that absolutely cannot migrate today. The HARDENING.md
table records this as DOC, not FIX.

### 10.3 CRLF injection — URL + header sanitisation

| Surface | Mitigation | Implementation |
|---|---|---|
| URL | reject CR / LF / NUL / TAB / VT / FF / DEL | `http.esk §http-safe-string?` (line 92), `http-check-url` (line 103) |
| Headers | same check on name AND value | `http.esk §http-check-headers` (line 107) |

The original CRLF audit was `#195`; the audit-M8 tightening
added tab and the rest of C0 because some HTTP parsers
re-split a header value on horizontal tab. The Scheme layer
raises `error` on rejection — the byte never reaches libcurl.

### 10.4 Path traversal — TOCTOU + O_NOFOLLOW

| Surface | Mitigation | Implementation |
|---|---|---|
| `file-copy` | `O_NOFOLLOW | O_CLOEXEC` on both fds | `system_builtins.c §eshkol_builtin_file_copy_v` (line 788) |
| `path-normalize` | reject inputs ≥ PATH_MAX | `system_builtins.c §eshkol_builtin_path_normalize_v` |

`O_NOFOLLOW` refuses to follow symlinks: a symlink-swap attack
between the open-of-src and open-of-dst calls cannot redirect
writes to a sensitive target. `O_CLOEXEC` ensures the fds are
closed across an exec — defensive against fd-passing
amplification in the rare case file-copy is called from a
about-to-exec context (HARDENING.md `#193`).

### 10.5 ReDoS — match-limit + depth-limit

| Surface | Mitigation | Implementation |
|---|---|---|
| every `pcre2_match` / `pcre2_substitute` | 10 M backtrack steps, 100 K depth | `agent_regex.c §get_match_context` (line 91) |

The numbers were chosen empirically: 10 M steps is ≈ a few
tens of milliseconds on a modern CPU — well below the
"noticeably slow" threshold but easily enough for any
legitimate match. A pathological pattern hits the limit and
returns a non-match instead of pinning the thread.

### 10.6 Windows command-line buffer overflow

| Surface | Mitigation | Implementation |
|---|---|---|
| `qllm_process_spawn_command_impl` (Windows branch) | heap-alloc exact size, reject ≥ 32768 | `agent_subprocess.c:589-622` |

`CreateProcessA` mutates lpCommandLine in place; the previous
4096-byte stack buffer was overflowable. The fix measures the
final length, rejects past the Windows hard limit, and
heap-allocates exactly the needed size.

### 10.7 Dynamic-linker env injection — child env scrubbing

| Surface | Mitigation | Implementation |
|---|---|---|
| posix_spawn child env | filter LD_* / DYLD_* injection vars | `agent_subprocess.c §eshkol_scrub_environ` (line 317) |
| fork+exec child | `unsetenv` same list pre-exec | `agent_subprocess.c §eshkol_unset_env_injection_vars` (line 264) |

A compromised shell profile that sets `DYLD_INSERT_LIBRARIES`
or `LD_PRELOAD` to a malicious library would otherwise inject
into every `git`/`lean`/`python` the agent launches. The
scrub eliminates that lateral-movement primitive at the
spawn boundary.

### 10.8 Pipe leak on partial spawn

| Surface | Mitigation | Implementation |
|---|---|---|
| `qllm_process_spawn*` | separate gate per pipe() call | `agent_subprocess.c:399-418`, `:730-753` (HARDENING.md `#182`) |

### 10.9 Resource limits — RLIMIT_AS / CPU / NOFILE / NPROC

| Surface | Mitigation | Implementation |
|---|---|---|
| fork+exec child | setrlimit caps inside child | `agent_subprocess.c §eshkol_apply_subproc_rlimits` (line 204) |

Coverage gap acknowledged: posix_spawn skips the rlimits for
performance. Callers needing guaranteed limits should
setrlimit the parent eshkol-run at startup so children
inherit via POSIX.

---

## 11. JIT vs AOT operational notes

| Scenario | What works | What the user must do |
|---|---|---|
| `eshkol-repl` interactive | All agent FFI surfaces resolved via dlsym + forced symbols | nothing — the build's `force_load` of `eshkol-agent-ffi` puts every symbol in the process |
| `eshkol-run script.esk -e` (JIT) | Same as REPL — agent symbols are live in the eshkol-run process | nothing |
| `eshkol-run script.esk -o out` (AOT) | `requires_agent_ffi(asts)` detects `(require agent.…)` BEFORE process_requires inlines it, then splices `ESHKOL_HOST_AGENT_FFI_LINK_ARGS` into the link line | nothing — but the build must have libcurl / sqlite3 / pcre2 installed at *configure* time for these to be linked. A binary built without libcurl will still run, but `http-get` returns `#f`. |
| Cross-machine deploy | The produced binary depends on the same `.dylib` / `.so` set the build host had | install matching libcurl / sqlite3 / pcre2 packages on the target, or build statically |
| Build host has no libcurl | `agent_http_client.c` is `#ifdef`'d out; `qllm_http_*` resolves to the weak fallback (returns NULL / "explicit unavailable") | install libcurl-dev and rebuild |

The bridge between AOT-time AST inspection and configure-time
link args is the indirection through `build_config.h.in:15`:

```c
#define ESHKOL_HOST_AGENT_FFI_LINK_ARGS "@ESHKOL_HOST_AGENT_FFI_LINK_ARGS@"
```

CMake substitutes the value during `configure_file`, the
agent archive is built with PUBLIC transitive deps so the
linker can resolve `_sqlite3_exec`, `_SecRandomCopyBytes`,
etc. when force-loading the agent archive into the user
binary.

---

## 12. Limitations and future work

Items called out explicitly in the source as deferred:

- **SSE / streaming HTTP** (`agent_http_client.c:317-344`).
  The `qllm_http_stream_*` functions are stubs; `http-stream-open`
  returns `#f` today. Real SSE needs the libcurl multi
  interface or chunked-transfer line parsing. `MEMORY.md
  §native HTTP client (#234)` lists this as a follow-up.

- **TLS insecure mode** (`agent_http_client.c:110-114`). No
  way through the API to disable peer/host verification.
  Callers needing self-signed dev origins shell out to
  `curl --insecure` via `subprocess`.

- **`qllm_http_request` / `qllm_http_post` with header
  arrays** (`http.esk:138-149`). The Eshkol wrapper currently
  routes everything via `qllm_http_post_json` and synthesises
  a single auth header from the alist; the full pass-through
  for arbitrary header lists is wired in the C ABI but not in
  the wrapper.

- **Subprocess spawn on Windows with argv form**
  (`agent_subprocess.c:886-890`). The argv path is POSIX-only;
  the Windows branch returns NULL. A Windows port needs to
  build the cmdline string per Windows quoting rules and call
  CreateProcessW (or accept the lossy CreateProcessA path).

- **Recursive watch — new directories** (`agent_watch.c:94-109`).
  Recursive mode enumerates subdirectories at start time and
  adds a watch per directory. Directories created *after*
  start are not auto-watched. Linux inotify supports
  IN_MOVE_SELF and IN_CREATE on parents that would let us
  detect and add new dirs; macOS kqueue would require an
  FSEvents migration.

- **WebSocket extensions** (`agent_http_server.c:218-222`).
  No permessage-deflate, no fragmentation handling. Production
  workloads should link libwebsockets or libcurl 7.86+.

- **HTTP server beyond loopback** (`agent_http_server.c:68-69`).
  The bind is hard-coded to `INADDR_LOOPBACK`. Programs
  needing public exposure must front with a proper proxy.

- **Match-callback regex substitution** (`agent_regex.c:295-297`).
  `pcre2_substitute_callback` is not part of PCRE2's stable
  API; the Scheme layer composes `regex-match-groups` +
  `regex-replace` for the equivalent.

- **posix_spawn rlimits coverage** (`agent_subprocess.c §coverage gap
  comment`, line 186). Only the fork+exec fallback applies
  `setrlimit`; the hot posix_spawn path skips it. Callers
  needing guaranteed caps set them at the parent process.

---

## 13. See also

- `lib/agent/c/agent_http_client.c` — libcurl HTTP/HTTPS, qllm_http_* ABI
- `lib/agent/c/agent_http_server.c` — minimal HTTP/1.1 + WebSocket client + AF_UNIX
- `lib/agent/c/agent_sqlite.c` — SQLite3 bindings, dynamic column-text sizing
- `lib/agent/c/agent_subprocess.c` — process spawn, pipe-drain, env scrub, rlimits
- `lib/agent/c/agent_regex.c` — PCRE2 with ReDoS limits and capture groups
- `lib/agent/c/agent_watch.c` — native kqueue / inotify watcher
- `lib/agent/c/agent_crypto.c` — HMAC-SHA256 / SHA256 / CSPRNG
- `lib/agent/*.esk` — Scheme wrappers for each subsystem
- `exe/eshkol-run.cpp §requires_agent_ffi` — AOT link-arg gating
- `cmake/build_config.h.in:15` — `ESHKOL_HOST_AGENT_FFI_LINK_ARGS`
- `CMakeLists.txt:1711-2007` — agent FFI build wiring and force-load
- `lib/repl/repl_jit.cpp:160-413` — JIT symbol registration for agent FFI
- `HARDENING.md` `#178`, `#182`, `#190`, `#191`, `#192`, `#193`, `#195` —
  the hardening commits that closed each guard documented above
- [`WEB_PLATFORM.md`](WEB_PLATFORM.md) — browser-side counterpart;
  the `--wasm` path emits the same kind of `(extern …)` bindings but
  with host JavaScript fulfilling the FFI rather than native C
- [`REPL_JIT.md`](REPL_JIT.md) — JIT machinery, including the
  DynamicLibrarySearchGenerator that resolves agent FFI symbols
  at REPL time
- [`COMPILATION_GUIDE.md`](COMPILATION_GUIDE.md) — AOT build pipeline
