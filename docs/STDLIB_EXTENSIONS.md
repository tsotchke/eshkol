# Eshkol Standard Library: Complete Extensions Requirements

**Origin**: restored from commit `e2eb942` (branch `internal/agent-infra`).
Previously tracked as `STDLIB_ADDITIONS.md`; renamed and re-tracked in
v1.2-scale as the authoritative agent-stdlib specification.

**Cross-reference**: items here appear in `docs/COMPILER_ROADMAP.md` under
their target release (e.g. B.10 concurrency → v1.4; B.19 LRU cache → v1.3;
E.1 IO multiplexing → v1.2.x). This doc is the expanded spec; the roadmap
is the schedule.

**Tracker**: epic task **#198** references this file as the source of truth
for agent-stdlib completeness.

Every function below is needed for 100% Claude Code parity AND to make our
agent vastly superior to Claude Code. Organized into three sections:

- **Part A**: Already in Eshkol but BROKEN/STUBBED — fix these first
- **Part B**: Truly missing — implement from scratch
- **Part C**: Beyond Claude Code — our unique advantages

---

## Architecture: Why We Target the VM

Eshkol is a **homoiconic, self-referential computation system**. Every Eshkol
expression is an S-expression — code IS data. The system forms an open loop:

```
Compiled Agent (LLVM, AOT)  ──writes S-expressions──►  JIT REPL (runtime)
        ▲                                                     │
        │                                               compiles to bytecode
    reads own code                                            │
    as data (homoiconic)                                      ▼
        │                                            Bytecode VM (interpreter)
        │                                                     │
        │                                         state vector = d_model=256
        │                                                     │
        │                                                     ▼
        │                                     Transformer Weights (6-layer, analytical)
        │                                          from weight_matrices.c
        │                                                     │
        │                                              forward pass =
        │                                         executing VM semantics
        │                                                     │
        └──────── agent reasons about ◄───────── qLLM / Selene
                  computation natively             reads code as weights,
                  (not as text tokens)             not text
```

### The State Vector IS the Model

`weight_matrices.c`: The VM's bounded execution state is a **256-dimensional float vector**
that simultaneously serves as:
- The VM's runtime state (PC, TOS, SOS, registers, arena cells, type tags, AD tape)
- The embedding dimension for a 6-layer transformer (d_model=256, n_heads=16, FFN_DIM=2304)

### Two Tiers of Execution

**Tier 1 — Weight-Encoded:** The bounded canonical VM/AD opcodes are implemented
as gated neuron pairs across the transformer layers. ADD, SUB, MUL, CONST, JUMP,
arena memory, closures, continuations, type predicates, and the verifier-covered
reverse-mode AD tape compute their results entirely in matrix multiplications.

**Tier 2 — Native Call Dispatch (IDs 300+):** `OP_NATIVE_CALL` (opcode 37) carries
the native function ID as its operand. It remains the explicit external boundary
for host services and high-level library calls. This is where
consciousness engine (500-527), factor graphs (520-539), workspace (540-549),
geometric manifolds (804-861), tensors (410-470), autodiff (370-409), and everything
in this document executes.

### Implementation Pattern Per Function

Every new function needs:

| Layer | What | How |
|-------|------|-----|
| **C implementation** | The actual logic | Write C function |
| **VM native case** | `case NNN:` in `vm_native.c` | Dispatched via `IS_NATIVE` → `exec_loop_postprocess` |
| **LLVM extern** | `.esk` `(extern ...)` declaration | Resolved at link time for compiled binary |
| **JIT accessible** | Available in REPL prelude | The agent can generate code that uses it at runtime |

New native IDs do NOT require changes to `weight_matrices.c` — `OP_NATIVE_CALL`
already routes through `IS_NATIVE` to C dispatch. Only new core VM opcodes
(beyond the 64-opcode interpreter core, or beyond the 83-opcode canonical
VM/AD ISA used by the SDNC artifact) need weight-matrix encoding.

---

# Part A: Originally Broken/Stale VM Builtins

These builtins were registered in `vm_native.c` but originally returned stubs
(empty list, nil, or hardcoded values). The status below reflects the current
standalone VM surface.

## A.1 `directory-entries` (builtin 601) — IMPLEMENTED

**Status**: The standalone VM now implements builtin 601 with
`opendir(3)` / `readdir(3)` / `closedir(3)` on non-WASM targets. It returns a
list of filename strings excluding `.` and `..`; the WASM build keeps the safe
`nil` fallback because browser-hosted VM execution has no POSIX directory API.

**Impact**: Eliminates `ls -1` shell-outs in 6+ files in eshkol-agent.

---

## A.2 `command-line` (builtin 602) — IMPLEMENTED

Standalone VM startup stores the script path plus user arguments and builtin
602 returns them as a list of strings. The regression fixture is
`tests/vm/command_line_args.esk`.

**Impact**: Needed for CLI argument parsing in `src/main.esk`.

---

## A.3 Parallel primitives (620-628) — WORKER VM PATH

`parallel-map`, `parallel-filter`, `parallel-fold`, `parallel-for-each`,
`parallel-execute`, `future`, `force`, `future-ready?`, `thread-pool-info`
have correct VM APIs. `parallel-map`, `parallel-filter`, and
`parallel-for-each` lazily initialize and use the VM thread pool for scheduling
when available. `thread-pool-info` / `thread-pool-size` report the standalone
VM pool size. `parallel-execute` accepts the source-level variadic thunk form
and schedules thunks through the same pool path. `future` now returns a
standalone VM future handle backed by the same pool; `force` joins that handle
(`force-future` remains as a compatibility alias) and `future-ready?` checks it
without forcing.

Read-only user closures now run in isolated worker VM contexts. Each worker
shares immutable bytecode/constants, clones the reachable heap graph into its
own arena at stable heap indexes, executes with private stack/frame/heap state,
and publishes returned worker-local cons/vector/string/numeric/tensor/bytevector
graphs back to the main heap under the runtime mutex. This preserves the VM's
arena allocation model without racing `Heap.next_free` or region-stack state.

Closures whose bytecode may mutate shared state, call arbitrary closures, perform
I/O, or return unsupported opaque objects deliberately fall back to the serialized
main-VM closure bridge. That fallback is required for correctness because the VM
does not have transactional shared-object semantics. `parallel-fold` also remains
an order-preserving sequential fallback for arbitrary non-associative fold
functions.

Key implementation:
```c
typedef struct {
    pthread_t* threads;
    int num_threads;
    // Lock-free work queue
    WorkItem* queue;
    int queue_head, queue_tail;
    pthread_mutex_t mutex;
    pthread_cond_t work_available;
    pthread_cond_t work_done;
} ThreadPool;
```

**Impact**: Tool execution in `query-loop.esk` can run tools in actual
parallel once closure execution no longer serializes through the main VM.

---

## A.4 `term-cursor-pos` (terminal.esk line 56) — IMPLEMENTED

The standalone VM exposes builtin 603, returning `(row . col)`. The agent
terminal FFI wrapper now calls scalar row/column helpers over the existing
DSR query path. Non-TTY contexts return `(0 . 0)` without writing terminal
escape sequences, which keeps CI and redirected output deterministic.

---

# Part B: Truly Missing — Implement From Scratch

These do NOT exist anywhere in Eshkol and must be added.

---

## B.1 Filesystem

### `(mkdir-recursive path) -> bool`
Create directory and all parents. `mkdir(2)` recursive.
Eliminates 12+ `run-command-capture "mkdir -p"` calls.

### `(file-rename old new) -> bool`
POSIX `rename(2)`. Eliminates `mv` shell-outs in `fs-write-atomic`.

### `(file-size path) -> integer`
`stat(2)` returning `st_size`. Eliminates `wc -c` + temp file hack.

### `(file-stat path) -> alist-or-false`
Full `lstat(2)`: `((size . N) (mtime . N) (ctime . N) (mode . N) (type . "file"|"directory"|"symlink") (uid . N) (gid . N))`.
Needed for: edit detection (mtime), type checks, permission checks.

### `(directory-walk path callback max-depth) -> void`
Recursive `opendir`/`readdir` calling `(callback path type depth)`.
Needed for: memory file discovery (`find` replacement), glob tool, grep tool.

### `(directory-delete-recursive path) -> bool`
Recursive delete with safety checks. Eliminates `rm -rf` shell-outs.
MUST refuse `/`, `/usr`, `/bin`, `/etc`, `/var`, `/home`, `/Users`, `/System`.

Implemented in the standalone VM as direct `opendir`/`lstat`/`unlink`/`rmdir`
recursion with root-path refusal; no shell command is invoked.

### `(file-copy src dst) -> bool`
Read/write in chunks. macOS can use `clonefile(2)` for CoW.

### `(file-chmod path mode) -> bool`
`chmod(2)`. Needed for keychain file security.

### `(symlink-create target link) -> bool`
`symlink(2)`. Needed for IDE lockfiles.

### `(symlink-read path) -> string-or-false`
`readlink(2)`.

### `(realpath path) -> string-or-false`
`realpath(3)`. Canonical path resolution.

### `(glob-match pattern path) -> bool`
`fnmatch(3)` with `**` and `{}` extensions. For permission path matching.

Implemented in the standalone VM as native ID 1757 using `fnmatch(3)` where
available.

### `(glob-expand pattern root) -> list`
Walk + match. Replaces the shell-based `agent.glob` module entirely.

The standalone VM exposes the one-argument compiled-runtime form
`(glob-expand pattern)` as native ID 1756, returning a newline-separated match
string.

### `(file-lock path) -> lock-or-false`
`flock(2)` with `LOCK_EX | LOCK_NB`. For session DB exclusive access.

The standalone VM follows the compiled-runtime low-level form
`(file-lock fd)` as native ID 1754.

### `(file-unlock lock) -> void`
`flock(2)` with `LOCK_UN`.

The standalone VM follows the compiled-runtime low-level form
`(file-unlock fd)` as native ID 1755.

### `(file-mmap path offset length) -> bytevector-or-false`
`mmap(2)` with `PROT_READ | MAP_PRIVATE`. For large file hashing.

Implemented in the standalone VM as native ID 1758. It maps the requested file
range with the host mapping API (`mmap` on POSIX, file mappings on Windows),
copies it into an arena-owned bytevector, then unmaps immediately so the result
obeys the VM's arena lifetime model. WASM builds return `#f` because they do
not expose the host file-mapping surface.

### `(file-munmap bv) -> void`

Implemented in the standalone VM as native ID 1759. Because standalone
`file-mmap` returns an arena-owned bytevector copy, this is a compatibility
no-op.

### `(file-mtime path) -> integer-or-false`

Implemented in the standalone VM as native ID 1752.

### `(file-atime path) -> integer-or-false`

Implemented in the standalone VM as native ID 1753.

---

## B.2 Path Manipulation

### `(path-join . components) -> string`
Join with `/`, normalize slashes, handle absolute reset.

### `(path-dirname path) -> string`
Directory component. Handle edge cases: root, trailing slash, empty.

### `(path-basename path) -> string`
Filename component.

### `(path-extname path) -> string`
Extension including dot. `".gitignore"` -> `""`, `"a.tar.gz"` -> `".gz"`.

### `(path-relative from to) -> string`
Compute relative path between two absolutes.

Implemented in the standalone VM as native ID 1727.

### `(path-resolve . components) -> string`
Resolve to absolute, prepending CWD if needed.

The standalone VM exposes the two-argument form `(path-resolve base rel)` as
native ID 1728.

### `(path-is-absolute? path) -> bool`
Starts with `/`.

### `(path-normalize path) -> string`
Resolve `.`, `..`, double slashes. No symlink resolution.

---

## B.3 Process Management

### `(process-pid) -> integer`
Return the current process ID. Implemented in the standalone VM as native ID
1784, matching the compiled runtime's current-process alias.

### `(process-spawn-pty command) -> proc`
`forkpty(3)`. Child sees a real terminal. Needed for: interactive commands,
Python/Node REPL, commands that detect `isatty`.

The standalone VM exposes native ID 1787 as a POSIX-only low-level form. It
returns a process handle `(pid . master-fd)`; on Windows and WASM it returns
`#f`.
`process-wait`, `process-kill`, `process-kill-tree`, `process-setpgid`,
`io-poll`, and `process-read-nonblocking` accept that handle directly.

### `(process-setpgid proc pgid) -> bool`
`setpgid(2)`. For process group management.

Implemented in the standalone VM as native ID 1785.

### `(process-kill-tree pid signal) -> bool`
Kill process + all descendants. Walk process tree via `pgrep -P` (macOS)
or `/proc/PID/children` (Linux). Send signal bottom-up.
Claude Code uses `tree-kill` npm package.

Implemented in the standalone VM as native ID 1786 for VM-spawned children:
`process-spawn` places each child in its own process group, and
`process-kill-tree` signals that group with a direct-PID fallback for external
processes.

### `(process-read-stdout-nonblocking proc max) -> string-or-false`
`fcntl(O_NONBLOCK)` + `read(2)`. For MCP message polling.

### `(process-read-stderr-nonblocking proc max) -> string-or-false`

The standalone VM exposes the underlying native ID 1788 as
`(process-read-nonblocking proc-or-fd max)`, returning a string when bytes are
available and `#f` on EOF, EAGAIN, or error.

### `(signal-install signum) -> bool`
Install a flag-only POSIX signal handler for `signum`. The standalone VM
exposes this as native ID 1794; Windows and WASM return `#f`.

### `(signal-check) -> integer`
Return the last delivered signal number and clear it, or `0` when no signal is
pending. The standalone VM exposes this as native ID 1795. Handlers only update
`sig_atomic_t` flags; Scheme code polls at safe points.

### `(signal-reset signum) -> bool`
Restore the default POSIX handler for `signum`. Standalone native ID 1796.

### `(signal-ignore signum) -> bool`
Ignore `signum` via `sigaction(SIG_IGN)`. Standalone native ID 1797.

### `(signal-count) -> integer`
Return the number of signals handled by the standalone VM process. Standalone
native ID 1798.

---

## B.4 HTTP and Networking

### `(http-request method url headers body timeout) -> (status headers body)`
All methods: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS.
Return response headers as alist. Currently only GET/POST, no response headers.

### `(http-set-proxy url) -> void`
`curl_easy_setopt(CURLOPT_PROXY, ...)`. For enterprise environments.

### `(http-set-tls-client-cert cert key ca) -> void`
mTLS via `CURLOPT_SSLCERT`/`CURLOPT_SSLKEY`/`CURLOPT_CAINFO`.

### `(http-server-create handler) -> server`
Bind TCP socket, parse HTTP/1.1, call `(handler method path query headers body)`.
Needed for OAuth PKCE callback and MCP auth flow.

### `(http-server-listen server port) -> integer`
Listen on port. Port 0 = random. Returns actual port.

### `(http-server-close server) -> void`

### `(websocket-connect url headers) -> ws`
WebSocket upgrade handshake. Using libcurl 7.86+ or raw TCP + TLS.
Needed for: voice STT streaming, remote sessions, MCP WebSocket transport.

### `(websocket-send ws data) -> bool`
Text frame.

### `(websocket-send-binary ws data) -> bool`
Binary frame.

### `(websocket-receive ws timeout) -> (type . data)-or-false`
`type`: `"text"`, `"binary"`, `"ping"`, `"close"`.

### `(websocket-close ws) -> void`

### `(unix-socket-connect path) -> socket-or-false`
For IDE IPC (VS Code's `VSCODE_IPC_HOOK`).

The standalone VM exposes this POSIX-only surface as native ID 1790. It
returns a raw socket file descriptor on success and `#f` on Windows, WASM, or
connection failure.

### `(socket-send socket string-or-bytevector) -> bytes-or-false`
Write a string or bytevector to a socket. The standalone VM exposes this as
native ID 1791 and returns the number of bytes written or `#f` on error.

### `(socket-recv socket max) -> string-or-false`
Read at most `max` bytes from a socket. The standalone VM exposes this as
native ID 1792, performs a non-blocking read, and returns an arena-owned string
when bytes are available or `#f` on EOF, EAGAIN, or error.

### `(socket-close socket) -> bool`
Close a socket file descriptor. The standalone VM exposes this as native ID
1793.

---

## B.5 Terminal Extensions

### `(term-set-scroll-region top bottom) -> void`
CSI DECSTBM. Fixed header/footer with scrollable content.

Standalone VM native ID 1930. It emits only when stdout is a TTY and returns
`#t` for valid regions.

### `(term-reset-scroll-region) -> void`
Standalone VM native ID 1931.

### `(term-enable-mouse) -> void`
SGR mouse tracking.

Standalone VM native ID 1932 enables X10 + SGR mouse modes when stdout is a
TTY.

### `(term-disable-mouse) -> void`
Standalone VM native ID 1933.

### `(term-read-mouse-event timeout) -> (button x y modifiers type)-or-false`
Standalone VM native ID 1934 parses SGR mouse events from stdin when attached
to a TTY and returns `#f` on timeout or non-TTY input.

### `(term-enable-alternate-screen) -> void`
`\033[?1049h`. Preserves shell history on exit.

Standalone VM native ID 1935.

### `(term-disable-alternate-screen) -> void`
Standalone VM native ID 1936.

### `(term-clipboard-write text) -> void`
OSC 52.

Standalone VM native ID 1937. Clipboard writes are TTY-gated.

### `(term-clipboard-read) -> string-or-false`
OSC 52.

Standalone VM native ID 1938 currently returns `#f` unless a terminal response
backend is available.

### `(term-hyperlink url text) -> string`
Return OSC 8 escape string for clickable links.

Standalone VM native ID 1939.

### `(term-detect-capabilities) -> alist`
Check `TERM`, `COLORTERM`, `TERM_PROGRAM`. Return `((color-depth . 24) (unicode . #t) ...)`.

Standalone VM native ID 1940 returns string-keyed alist entries for
`color-depth`, `unicode`, and `tty`.

### `(term-bell) -> void`
BEL character.

Standalone VM native ID 1941.

---

## B.6 Cryptography Extensions

### `(sha256-file path) -> string-or-false`
Stream hash without loading file into memory.

### `(base64url-encode data) -> string`
URL-safe base64 (`-_`, no padding). For JWT and PKCE.

### `(base64url-decode data) -> string-or-false`

### `(uuid-v4) -> string`
`"xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"`. For session/message/request IDs.

### `(constant-time-equal? a b) -> bool`
XOR-accumulate comparison. For HMAC verification.

---

## B.7 System Information

### `(os-type) -> string`
`"darwin"` or `"linux"`. Compile-time or `uname(2)`.
Eliminates `uname -s` shell-out in keychain.esk.

### `(os-arch) -> string`
`"x86_64"` or `"aarch64"`.

### `(home-directory) -> string`
`getpwuid(getuid())->pw_dir`. Fallback to `$HOME`.
Eliminates fragile `(or (getenv "HOME") "/tmp")` in 15+ files.

### `(current-directory) -> string`
`getcwd(3)`.

### `(set-current-directory! path) -> bool`
`chdir(2)`.

### `(setenv name value) -> void`
`setenv(3)`.

### `(unsetenv name) -> void`
`unsetenv(3)`.

### `(hostname) -> string`
`gethostname(3)`.

### `(username) -> string`
`getpwuid(getuid())->pw_name`.

### `(executable-exists? name) -> bool`
Split PATH on `:`, check `access(path, X_OK)`.
Eliminates `which` shell-outs.

### `(executable-path name) -> string-or-false`
Same but returns full path.

### `(current-time-ms) -> integer`
`clock_gettime(CLOCK_REALTIME)` in milliseconds.

### `(monotonic-time-ms) -> integer`
`clock_gettime(CLOCK_MONOTONIC)` in milliseconds. Not affected by NTP.

### `(temp-directory) -> string`
`$TMPDIR` or `/tmp`.

### `(prevent-sleep reason) -> handle`
macOS: `IOPMAssertionCreateWithName`. Linux: `systemd-inhibit`.

### `(allow-sleep handle) -> void`

---

## B.8 String Operations

### `(string-display-width str) -> integer`
Terminal column width accounting for CJK (2 cols), emoji (2 cols),
zero-width combining marks (0 cols), ANSI escapes (0 cols).
Uses Unicode East Asian Width (UAX #11).
**Critical for TUI correctness.**

Standalone VM native ID 1947 strips ANSI escape sequences from the measured
width, counts common CJK/emoji ranges as double-width, and treats combining
marks/variation selectors as zero-width.

### `(string-truncate-display str max suffix) -> string`
Truncate to `max` display columns. Never splits a wide char.

Standalone VM native ID 1948 uses the same width accounting as
`string-display-width`; suffix text is included only after the source string
needs truncation, and the returned string is capped to `max` display columns.

### `(url-encode str) -> string`
Percent-encoding. Unreserved chars: `A-Za-z0-9-_.~`.

### `(url-decode str) -> string`

### `(url-parse str) -> alist-or-false`
`((scheme . "https") (host . "...") (port . 443) (path . "...") (query . "..."))`.

### `(string-ends-with? str suffix) -> bool`

### `(string-index-of str substr start) -> integer-or-false`
Find with start offset. Unlike `string-contains` which returns bool.

### `(string-pad-left str width ch) -> string`

### `(string-pad-right str width ch) -> string`

### `(string-repeat str count) -> string`
`(string-repeat "-" 40)` -> `"----------------------------------------"`

### `(string-split str delim limit) -> list`
Split with max splits: `(string-split "a:b:c" ":" 1)` -> `("a" "b:c")`.

---

## B.9 Regex Extensions

### `(regex-match-groups handle str) -> list-or-false`
Return capture groups: `((full . "text") (groups . ("g1" "g2")) (start . 0) (end . 5))`.
The existing `regex-match` only returns the full match string.

### `(regex-split handle str) -> list`
Split string on pattern matches.

---

## B.10 Concurrency

### `(make-channel capacity) -> channel`
Bounded thread-safe message queue. 0 = synchronous.

### `(channel-send! ch value) -> void`
Blocks if full.

### `(channel-receive ch timeout) -> value-or-false`
Blocks until value or timeout.

### `(channel-try-receive ch) -> value-or-false`
Non-blocking.

### `(channel-close! ch) -> void`

### `(make-mutex) -> mutex`
pthread_mutex.

### `(mutex-lock! m) -> void`

### `(mutex-unlock! m) -> void`

### `(with-mutex m thunk) -> value`
Lock + dynamic-wind unlock.

### `(make-condition-variable) -> cv`
pthread_cond.

### `(condition-wait cv mutex) -> void`

### `(condition-signal cv) -> void`

### `(condition-broadcast cv) -> void`

### `(make-timer delay callback) -> timer`
Schedule callback after delay-ms.

### `(timer-cancel! timer) -> void`

### `(make-interval interval callback) -> interval`
Repeat callback every interval-ms.

### `(interval-cancel! interval) -> void`

---

## B.11 Date/Time

### `(format-iso8601 epoch) -> string`
`"2026-04-03T15:30:00Z"` via `gmtime_r` + `strftime`.

### `(parse-iso8601 str) -> integer-or-false`

### `(format-relative seconds-ago) -> string`
30 -> `"30s ago"`, 3600 -> `"1h ago"`.

### `(local-timezone-offset) -> integer`
Seconds east of UTC.

---

## B.12 JSON Extensions

### `(json-get-in obj path default) -> value`
Deep access: `(json-get-in resp '("usage" "input_tokens") 0)`.

### `(json-stringify-pretty obj indent) -> string`
Human-readable JSON with indentation.

### `(json-merge a b) -> object`
Deep merge. b overrides a. Arrays replaced (not merged).

---

## B.13 SQLite Extensions

### `(db-transaction db thunk) -> result`
BEGIN + thunk + COMMIT. ROLLBACK on error.

### `(db-busy-timeout db ms) -> void`
Retry on SQLITE_BUSY.

### `(db-last-insert-id db) -> integer`

### `(db-changes db) -> integer`
Rows affected by last statement.

---

## B.14 Compression

### `(deflate data) -> bytevector`
zlib compress.

### `(inflate data) -> bytevector-or-false`
zlib decompress.

### `(gzip data) -> bytevector`

### `(gunzip data) -> bytevector-or-false`

---

## B.15 Streams

### `(make-pipe) -> (read-fd . write-fd)`
`pipe(2)`.

### `(make-line-reader fd callback) -> reader`
Buffer reads, deliver complete lines. The `readline.createInterface` equivalent.

### `(line-reader-close reader) -> void`

---

## B.16 Diff Algorithm

### `(diff-lines old new) -> hunks`
Myers' O(ND) diff. Returns list of hunks with `=`, `+`, `-` lines.
Needed for: file edit preview, screen buffer optimization, compaction diffs.

---

## B.17 Fuzzy Search

### `(fuzzy-match pattern candidates key-fn max) -> list`
Score-ranked fuzzy matching. Bonus for consecutive matches, word boundaries,
camelCase. Returns `((score . candidate) ...)`.

---

## B.18 Semantic Versioning

### `(semver-parse str) -> alist-or-false`
### `(semver-compare a b) -> -1|0|1`
### `(semver-satisfies? version range) -> bool`

---

## B.19 LRU Cache

### `(make-lru-cache max-size) -> cache`
Hash table + doubly-linked list. O(1) get/set/evict.

### `(lru-get cache key) -> value-or-false`
### `(lru-set! cache key value) -> void`
### `(lru-has? cache key) -> bool`
### `(lru-delete! cache key) -> void`
### `(lru-clear! cache) -> void`
### `(lru-size cache) -> integer`

---

## B.20 Event Emitter

### `(make-event-emitter) -> emitter`
### `(emit! emitter event . args) -> void`
### `(on! emitter event handler) -> void`
### `(once! emitter event handler) -> void`
### `(off! emitter event handler) -> void`

---

## B.21 Layout Engine (Full Flexbox)

The existing `agent.layout` is a simple row/column divider. For Claude Code
parity we need real flexbox:

### `(yoga-node-create) -> node`
Bind Yoga C library. Properties: `width`, `height`, `flex-direction`,
`justify-content`, `align-items`, `flex-grow`, `flex-shrink`, `padding`,
`margin`, `overflow`, `position`, `border`, `gap`.

### `(yoga-node-set! node prop value) -> void`
### `(yoga-node-add-child! parent child) -> void`
### `(yoga-node-calculate! root width height) -> void`
### `(yoga-node-get-computed node prop) -> number`
Props: `"left"`, `"top"`, `"width"`, `"height"`.

### `(yoga-node-free! node) -> void`

---

## B.22 Shell Utilities

### `(shell-quote str) -> string`
Proper single-quote escaping. Eliminates 20+ inline `string-replace` hacks.

### `(shell-split str) -> list`
Parse shell command into argv respecting quotes and escapes.

---

## B.23 File Watching (Native)

The existing `agent.fs-watch` shells out to `fswatch`/`inotifywait` via
`system()`. Replace with native:

### `(fs-watch-native path callback) -> watcher`
Use `kqueue` (macOS) or `inotify` (Linux) directly via FFI.
`callback`: `(lambda (event-type filename) ...)`.
`event-type`: `"change"`, `"create"`, `"delete"`, `"rename"`.

Standalone VM native ID 1942 returns a watcher handle. The VM implementation
uses native `stat(2)` signatures and explicit polling so tests and agent loops
can avoid shelling out even when kqueue/inotify callbacks are unavailable.
The callback argument is accepted for API shape but not invoked by the
standalone VM.

### `(fs-watch-recursive path callback) -> watcher`
Watch entire tree. Auto-add new subdirectories.

Standalone VM native ID 1943 returns a watcher handle and records the recursive
intent. Current standalone polling tracks the root path signature.

### `(fs-watch-poll watcher) -> string-or-false`
Standalone VM native ID 1944 returns `"event\tpath"` for `change`, `create`,
or `delete`, or `#f` when no change is pending.

### `(fs-unwatch watcher) -> void`
Standalone VM native ID 1945 releases the watcher handle.

---

## B.24 Self-Compilation and Hot-Reload

The agent IS Eshkol — a homoiconic LISP where every expression is an S-expression.
She should be able to read her own source, modify it, recompile, and replace
herself while running. The compiled binary already contains an embedded JIT
compiler (`ReplJITContext` from `lib/repl/repl_jit.h`) accessible via `eval`.

### `(execv path argv) -> never-returns`
POSIX `execv(2)`. Replace the running process image with a new binary.
This is how the agent replaces herself after recompilation.

### `(fork) -> integer`
POSIX `fork(2)`. Returns 0 in child, PID in parent.
For safe self-replacement: fork, exec new binary in child, parent exits.

### `(read port) -> sexp`
R7RS `read`. Parse an S-expression from a port or string.
Already exists as a builtin — verify it works in the compiled binary.
This + `eval` = the agent can parse and execute arbitrary Eshkol at runtime.

### `(write sexp port) -> void`
R7RS `write`. Serialize an S-expression to a port or string.
This + homoiconicity = the agent can generate and persist new code.

### `(open-input-string str) -> port`
R7RS string port. `(read (open-input-string "(+ 1 2)"))` → S-expression.

### `(open-output-string) -> port` / `(get-output-string port) -> string`
R7RS string port for writing. Serialize S-expressions to strings.

### `(eval sexp) -> value`
Already implemented (`lib/core/introspection.cpp:624`): S-expression → AST →
JIT compile → execute. The compiled binary has a full LLVM JIT inside.
Verify: does `(eval (list '+ 1 2))` return 3 from the compiled agent binary?

### `(load path) -> void`
Load and evaluate a file. Already exists via `require`, but verify dynamic
loading works at runtime (not just compile-time module resolution).

**Self-compilation pattern:**
```scheme
;; Recompile the agent from modified source
(define (recompile-self! source-dir output-path)
  (run-command-capture
    (string-append "eshkol-run " source-dir "/src/main.esk -o " output-path)))

;; Hot-reload a single module via embedded JIT
(define (hot-reload! module-source)
  (eval (read (open-input-string module-source))))

;; Replace the running binary
(define (replace-self! new-binary args)
  (execv new-binary (cons new-binary args)))
```

---

# Part C: The Agent's Cognitive Architecture

These aren't "nice to have" features. They ARE the agent's mind.

The consciousness engine is how she reasons. The manifolds are the space she
thinks in. The autodiff is how she improves herself. The homoiconic eval is
how she writes and rewrites her own code. Every function below is a cognitive
primitive — a computation the agent can perform, compose, evaluate at runtime,
compile into new code, and reason about through transformer weights.

---

## C.1 Tensor-Powered Token Estimation

Claude Code's token estimator uses `len / 4` heuristics. We have a full
tensor engine. We should use it.

### `(token-estimate-ml text) -> integer`

Use a small trained model (character n-gram frequencies -> token count) stored
as tensors. Train on a corpus of text + actual tokenizer output. The tensor
ops already exist:

```scheme
;; Character frequency tensor
(define *char-freq-weights* (make-tensor '(256 16) 0.0))
;; Single layer: char-freqs -> token estimate
(define (token-estimate-ml text)
  (let* ((freq-vec (text->char-freq-tensor text))    ;; 256-dim
         (hidden (tensor-relu (matmul *char-freq-weights* freq-vec)))
         (estimate (tensor-sum hidden 0)))
    (max 1 (inexact->exact (round (tensor-ref estimate '(0)))))))
```

**Functions needed** (already exist as builtins 410-470):
- `make-tensor`, `tensor-ref`, `tensor-set!`, `matmul`, `tensor-relu`,
  `tensor-sum`, `tensor-sigmoid`, `tensor-softmax`

**Additional needed for training**:
### `(tensor-save tensor path) -> bool`
Serialize tensor to file (binary format).

### `(tensor-load path) -> tensor-or-false`
Load tensor from file.

---

## C.2 Autodiff-Optimized Algorithms

Eshkol has automatic differentiation (builtins 370-409). We can use it
to self-optimize:

### `(optimize-parameter f initial-value) -> value`

Use `gradient` (builtin 750) + gradient descent to tune parameters:

```scheme
;; Auto-tune the chars-per-token ratio for each content type
(define (auto-calibrate-token-ratio corpus actual-tokens)
  (let ((ratio (make-dual 3.5 1.0)))  ;; Start at 3.5 chars/token
    (gradient-descent
      (lambda (r)
        (let ((estimated (/ (string-length corpus) (dual-primal r))))
          (dual-pow (dual-sub (make-dual estimated 0.0)
                              (make-dual actual-tokens 0.0)) 2.0)))
      ratio 0.01 100 0.001)))
```

**Already exists**: `make-dual`, `dual-primal`, `dual-tangent`,
`dual-add/sub/mul/div`, `dual-sin/cos/exp/log/sqrt/pow`,
`dual-relu/sigmoid/tanh`, `gradient`, `jacobian`, `hessian`,
AD tape operations (390-409).

**Needed for integration**:
### `(gradient-descent f x0 step max-iters tol) -> value`
Already in `ml.optimization` but may need to be re-exported or linked.

### `(adam f x0 lr beta1 beta2 eps max-iters) -> value`
Adam optimizer from `ml.optimization`.

---

## C.3 Consciousness-Driven Tool Selection

Already partially used in `consciousness.esk`. Extend it:

### `(mind-predict-tool-success mind tool-name input-summary) -> float`

Use the factor graph to predict whether a tool call will succeed based on
history:

```scheme
(define (mind-predict-tool-success mind tool-name input-summary)
  (let* ((kb (hash-ref mind "kb"))
         (history (kb-query kb (make-fact 'tool-result tool-name '?result)))
         (successes (count (lambda (f) (string=? (fact-object f) "success")) history))
         (failures (count (lambda (f) (string=? (fact-object f) "failure")) history))
         (total (+ successes failures)))
    (if (= total 0) 0.5
        (/ successes total))))
```

**Already exists**: `make-kb`, `kb-assert!`, `kb-query`, `make-fact`,
`make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`,
`fg-observe!`, `free-energy`, `expected-free-energy`,
`make-workspace`, `ws-register!`, `ws-step!`, `ws-get-content`,
`ws-set-content!`, `ws-get-dim`, `ws-get-step-count`.

**Needed**:
### `(fg-marginal fg var-id) -> tensor`
Extract the marginal probability distribution for a variable after inference.
The inference (`fg-infer!`) runs loopy belief propagation but there's no
way to read the resulting beliefs out.

### `(fg-entropy fg) -> float`
Shannon entropy of the current belief state. For measuring uncertainty.

### `(kb-retract! kb fact) -> bool`
Remove a fact from the knowledge base. Needed for correcting stale beliefs.

### `(kb-count kb predicate) -> integer`
Count matching facts without materializing the list.

---

## C.4 Active Inference for Strategy Selection

Use Eshkol's free energy minimization (builtins 525-527) to select
strategies based on expected information gain, not just heuristics:

```scheme
(define (select-strategy-active-inference mind task-description)
  (let* ((fg (hash-ref mind "fg"))
         ;; Compute expected free energy for each strategy
         (strategies '("direct" "explore" "cautious" "systematic" "creative"))
         (scores (map (lambda (strat)
                        (let ((action-idx (strategy->index strat)))
                          (cons strat (expected-free-energy fg 0 action-idx))))
                      strategies))
         ;; Select strategy with lowest expected free energy
         (best (fold (lambda (s best)
                       (if (< (cdr s) (cdr best)) s best))
                     (car scores) (cdr scores))))
    (car best)))
```

**Already exists**: `free-energy`, `expected-free-energy`, `fg-observe!`.

---

## C.5 Complex Number Arithmetic

Eshkol has full complex number support (builtins 300-319). Uses:

- **FFT-based fuzzy search**: Use `fft` (from `signal.fft`) for fast
  string similarity via cross-correlation
- **Signal processing on token streams**: Detect periodicity in user
  request patterns
- **Phase analysis for timing**: Represent timing patterns as complex
  exponentials

**Already exists**: `make-rectangular`, `make-polar`, `real-part`,
`imag-part`, `magnitude`, `angle`, `conjugate`, `complex-add/sub/mul/div`,
`sqrt-complex`, `exp-complex`, `log-complex`, `sin-complex`, `cos-complex`,
`complex?`, `complex-expt`, `complex=?`.

---

## C.6 Bignum Exact Arithmetic

Eshkol has arbitrary-precision integers (builtins 350-369). Uses:

- **Exact cost tracking**: No floating-point rounding errors in
  cumulative cost calculations
- **Cryptographic operations**: Large prime generation, RSA if needed
- **Token counting**: Never overflow on large documents

**Already exists**: `bignum-from-int`, `bignum-to-int`, `bignum-to-float`,
`bignum-string-scan`, `bignum-string-emit`, `bignum-add/sub/mul/div/mod`,
`bignum-compare`, `bignum-pow`, `bignum-gcd`, `bignum-lcm`,
`bignum-bitwise-and/or/xor/not`, `bignum-shift`.

---

## C.7 Rational Exact Fractions

Eshkol has exact rational arithmetic (builtins 330-349). Uses:

- **Exact token/dollar ratios**: No floating-point drift
- **Layout calculations**: Exact column width division
- **Budget enforcement**: Never lose a fraction of a cent

**Already exists**: `make-rational`, `rational-add/sub/mul/div`,
`rational-numerator`, `rational-denominator`, `rational-floor`,
`rational-ceiling`, `rational-truncate`, `rational-round`,
`rational-compare`, `rational-gcd`, `rational-lcm`.

---

## C.8 Statistical Analysis

Use `math.statistics` for analytics:

```scheme
(require math.statistics)

;; Analyze response time distribution
(define (analyze-response-times times)
  (list (cons 'median (median times))
        (cons 'p95 (percentile times 95))
        (cons 'std-dev (std-dev times))
        (cons 'iqr (iqr times))))
```

**Already exists**: `median`, `percentile`, `quartiles`, `iqr`,
`variance`, `std-dev`, `correlation`, `covariance`, `histogram`,
`bin-data`, `zscore`, `describe`.

---

## C.9 CSV Data Processing

Use `core.data.csv` for structured data:

**Already exists**: `csv-parse`, `csv-parse-file`, `csv-stringify`,
`csv-write-file`.

---

## C.10 ODE Solvers for Adaptive Rate Limiting

Use `math.ode` to model API rate limit dynamics:

```scheme
(require math.ode)

;; Model token bucket as ODE: dy/dt = refill_rate - consumption_rate
(define (predict-rate-limit-recovery current-tokens refill-rate consumption-rate)
  (euler (lambda (t y) (- refill-rate consumption-rate))
         current-tokens 0.0 60.0 1.0))  ;; Predict 60 seconds ahead
```

**Already exists**: `euler`, `euler-step`, `euler-final`.

---

## C.11 FFT for Pattern Detection

Use `signal.fft` for analyzing user behavior patterns:

```scheme
(require signal.fft)

;; Detect periodicity in user request timing
(define (detect-request-periodicity timestamps)
  (let* ((intervals (map - (cdr timestamps) timestamps))
         (spectrum (fft intervals)))
    ;; Find dominant frequency
    (let loop ((i 1) (max-mag 0) (max-i 0))
      (if (>= i (/ (length spectrum) 2))
          (/ (length intervals) max-i)  ;; Period in requests
          (let ((mag (magnitude (list-ref spectrum i))))
            (if (> mag max-mag)
                (loop (+ i 1) mag i)
                (loop (+ i 1) max-mag max-i)))))))
```

**Already exists**: `fft`, `ifft`.

---

## C.12 Logic Programming for Permission Rules

Use Eshkol's logic/constraint system (builtins 500-512) for declarative
permission checking:

```scheme
;; Assert permission rules as facts
(kb-assert! rules-kb (make-fact 'allow "bash" "read-only"))
(kb-assert! rules-kb (make-fact 'deny "bash" "rm -rf"))
(kb-assert! rules-kb (make-fact 'allow "file-write" "*.esk"))

;; Query: is this tool + input allowed?
(define (check-permission tool-name input)
  (let ((denies (kb-query rules-kb (make-fact 'deny tool-name '?pattern))))
    (if (any (lambda (f) (glob-match (fact-object f) input)) denies)
        'denied
        'allowed)))
```

**Already exists**: `make-logic-var`, `logic-var?`, `unify`, `walk`,
`walk-deep`, `make-substitution`, `substitution?`, `make-fact`, `fact?`,
`make-kb`, `kb?`, `kb-assert!`, `kb-query`.

---

## C.13 Parameter Objects for Dynamic Configuration

Use Eshkol's parameter objects (builtins 700-704) instead of mutable globals:

```scheme
(define *current-model* (make-parameter "claude-sonnet-4-6" #f))
(define *current-max-tokens* (make-parameter 8192 #f))

;; Temporarily override for a scope
(parameterize ((*current-model* "claude-opus-4-6")
               (*current-max-tokens* 16384))
  (send-api-request ...))
;; Parameters automatically restored after scope
```

**Already exists**: `make-parameter`, `parameter-ref`,
`parameterize-push`, `parameterize-pop`, `parameter?`.

---

## C.14 Trampoline for Stack Safety

Use `core.control.trampoline` for deep recursion without stack overflow:

```scheme
(require core.control.trampoline)

;; Process deeply nested JSON without stack overflow
(define (deep-json-walk obj f)
  (trampoline
    (lambda ()
      (cond
        ((hash-table? obj)
         (for-each (lambda (k)
                     (bounce (lambda () (deep-json-walk (hash-ref obj k) f))))
                   (hash-keys obj)))
        ((list? obj) ...)
        (else (f obj) (done obj))))))
```

**Already exists**: `trampoline`, `bounce`, `done`.

---

## C.15 Multi-Value Returns

Use Eshkol's multi-value system (builtins 650-654) for efficient returns:

```scheme
;; Return multiple values without consing
(define (parse-command input)
  (values (extract-command input)
          (extract-args input)
          (extract-flags input)))

(call-with-values
  (lambda () (parse-command "git commit -m 'msg'"))
  (lambda (cmd args flags)
    ...))
```

**Already exists**: `values`, `multi-value-ref`, `multi-value-count`,
`multi-value?`, `call-with-values`.

---

## C.16 Bytevector Operations for Binary Protocols

Use builtins 680-689 for binary data:

```scheme
;; Parse binary protocol header
(define (parse-mcp-frame data)
  (let* ((bv (string->utf8 data))
         (version (bytevector-u8-ref bv 0))
         (length (+ (* (bytevector-u8-ref bv 1) 256)
                     (bytevector-u8-ref bv 2))))
    (list version length)))
```

**Already exists**: `make-bytevector`, `bytevector-length`,
`bytevector-u8-ref`, `bytevector-u8-set!`, `bytevector-append`,
`bytevector-copy!`, `bytevector?`, `bytevector-copy`,
`utf8->string`, `string->utf8`.

---

## C.17 Functional Combinators

Use `core.functional` for cleaner code:

```scheme
(require core.functional.compose)
(require core.functional.curry)

;; Compose validation pipeline
(define validate-input
  (compose3 check-length check-encoding check-safety))

;; Partial application for tool dispatch
(define run-bash (partial2 execute-tool "bash"))
```

**Already exists**: `compose`, `compose3`, `curry2`, `curry3`,
`uncurry2`, `partial`, `partial1`, `partial2`, `partial3`,
`flip`, `identity`, `constantly`.

---

# Summary

| Section | Category | Count | Status |
|---------|----------|-------|--------|
| **A** | Fix broken stubs | **4** | Mostly implemented; A.3 remains partial |
| **B.1** | Filesystem | **16** | Truly missing |
| **B.2** | Path | **8** | Truly missing |
| **B.3** | Process | **6** | Truly missing |
| **B.4** | HTTP/Network | **12** | Truly missing |
| **B.5** | Terminal | **11** | Truly missing |
| **B.6** | Crypto | **5** | Truly missing |
| **B.7** | System Info | **15** | Truly missing |
| **B.8** | String | **11** | Truly missing |
| **B.9** | Regex | **2** | Truly missing |
| **B.10** | Concurrency | **17** | Truly missing |
| **B.11** | Date/Time | **4** | Truly missing |
| **B.12** | JSON | **3** | Truly missing |
| **B.13** | SQLite | **4** | Truly missing |
| **B.14** | Compression | **4** | Truly missing |
| **B.15** | Streams | **3** | Truly missing |
| **B.16** | Diff | **1** | Truly missing |
| **B.17** | Fuzzy search | **1** | Truly missing |
| **B.18** | Semver | **3** | Truly missing |
| **B.19** | LRU Cache | **7** | Truly missing |
| **B.20** | Event Emitter | **5** | Truly missing |
| **B.21** | Layout (Yoga) | **6** | Truly missing |
| **B.22** | Shell utils | **2** | Truly missing |
| **B.23** | File watching | **3** | Truly missing |
| | **Part B total** | **148** | |
| **C** | Beyond Claude Code | **~30 integrations** | Already in Eshkol, not used |
| | **Grand total** | **~182 new functions** | |

## Already In Eshkol, Not Being Used by eshkol-agent

These exist and work. We just need to `(require ...)` them:

| Module | Functions | What we gain |
|--------|-----------|-------------|
| `core.data.base64` | `base64-encode`, `base64-decode` | Replace manual base64 in bridge.esk |
| builtins 557-558 | `string-upcase`, `string-downcase` | Case conversion |
| builtins 565, 567 | `string->list`, `list->string` | Character-level ops |
| builtins 562 | `string-ci=?` | Case-insensitive compare |
| builtins 570 | `string-hash` | Hash keys, dedup |
| builtins 663-670 | `hash-delete!`, `hash-has-key?`, `hash-values`, `hash-count`, `hash-table-copy`, `hash-clear!`, `hash-table?` | Full hash table API |
| builtins 680-689 | Full bytevector API | Binary protocol handling |
| builtins 700-704 | `make-parameter`, `parameterize` | Dynamic config scoping |
| builtins 650-654 | `values`, `call-with-values` | Multi-value returns |
| builtins 710-714 | `error`, `error-message`, `error-type`, `error?` | Structured errors |
| builtins 300-319 | Complex numbers | FFT, signal analysis |
| builtins 330-349 | Rationals | Exact arithmetic |
| builtins 350-369 | Bignums | Arbitrary precision |
| builtins 370-409 | Autodiff + AD tape | Self-optimizing algorithms |
| builtins 410-470 | Tensors | ML-based token estimation |
| builtins 500-527 | Logic, KB, factor graph | Already partially used |
| builtins 750-752 | `gradient`, `jacobian`, `hessian` | Optimization |
| `math.statistics` | median, percentile, std-dev, etc. | Analytics |
| `math.ode` | Euler solver | Rate limit modeling |
| `signal.fft` | FFT, IFFT | Pattern detection |
| `ml.optimization` | gradient-descent, Adam, L-BFGS, CG | Self-tuning |
| `ml.activations` | ReLU, sigmoid, tanh + derivatives | Neural net inference |
| `core.functional` | compose, curry, partial, flip | Cleaner code |
| `core.control.trampoline` | trampoline, bounce, done | Stack safety |
| `core.data.csv` | CSV parse/write | Data processing |
| `core.list.*` | sort, partition, zip, iota, range, etc. | Better list ops |
| `core.logic.*` | Predicates, boolean ops | Input validation |

## Implementation Priority

**P0 — Remaining work that unblocks everything:**
- A.3 Fix parallel primitives (makes tool execution truly concurrent)
- B.1 Filesystem: `mkdir-recursive`, `file-rename`, `file-size`, `directory-delete-recursive`
- B.7 System: `os-type`, `home-directory`, `executable-exists?`, `current-time-ms`
- B.22 `shell-quote`

**P1 — Core agent:**
- B.4 Full HTTP (all methods, headers, response headers)
- B.3 Process: `process-pid`, `process-kill-tree`
- B.10 Concurrency: channels, mutex (for real parallelism)
- B.8 `string-display-width` (TUI correctness)
- B.12 `json-get-in`
- B.6 `uuid-v4`, `base64url-encode/decode`
- Start using C.1 tensors for token estimation

**P2 — TUI parity:**
- B.5 Terminal: scroll regions, mouse, alternate screen, clipboard
- B.21 Yoga layout engine
- B.16 Diff algorithm

**P3 — Full parity + beyond:**
- B.4 WebSocket, HTTP server
- B.23 Native file watching
- B.14 Compression
- B.3 PTY support
- Everything in Part C

---

# Part A (continued): More Broken Stubs Found in Code

## A.5 Riemannian Adam (builtins 840, 860-861) — IMPLEMENTED

**File**: `lib/backend/vm_geometric.c`
```c
VmRiemannianAdamState* st = vm_default_riemannian_adam_state(vm, point);
vm_riemannian_adam_euclidean_step(vm, point, grad, st, lr, beta1, beta2);
```

**Status**: Adam now has per-parameter first and second moment vectors `m[]`
and `v[]` in an arena-backed `VmRiemannianAdamState`. The existing six-argument
`riemannian-adam-step` uses a VM-lifetime default state keyed by tensor shape for
backward compatibility. Callers that need independent optimizer streams can use
the explicit arena-backed state surface:

```scheme
(make-riemannian-adam-state point)                         ;; native 860
(riemannian-adam-step! state point grad lr beta1 beta2 curv) ;; native 861
```

**Impact**: Proper Adam is needed for learning curvature and embedding parameters.

---

## A.6 Christoffel Symbols (builtin 830) — IMPLEMENTED

**File**: `lib/backend/vm_geometric.c`
```c
case 830: { /* christoffel(manifold, point) — connection coefficients */
    vm_geometric_christoffel_tensor(vm);
    break;
```

**Status**: For constant-curvature spaces (which is all Eshkol supports), the
Christoffel symbols are expressible in closed form. For hyperbolic space with
curvature K, `Γ^k_{ij} = K * (δ_ij x^k - δ_jk x^i - δ_ik x^j)` at a point
`x`. The standalone VM now returns this as a `dim×dim×dim` tensor allocated in
the VM arena.

---

## A.7 Pullback (builtin 838) — IMPLEMENTED

**File**: `lib/backend/vm_geometric.c`
```c
case 838: { /* pullback(form, jacobian) — 2 args */
    vm_geometric_pullback_tensor(vm);
    break;
```

**Status**: `f*(omega)(v) = omega(J*v)` where `J` is the Jacobian matrix passed
as the second argument. The VM now computes one-form pullback as `J^T * omega`
and returns the pulled-back coefficient tensor.

---

## A.8 Reverse-Mode AD Tape VM Surface — IMPLEMENTED

**File**: `lib/backend/vm_autodiff.c` implements the tape operations and
`lib/backend/vm_native.c` now dispatches the standalone VM surface:
- 390: `ad-tape-new`
- 391: `ad-const`
- 392: `ad-var`
- 393: `derivative` / `diff` forward-mode dual bridge
- 394-397: `ad-add`, `ad-sub`, `ad-mul`, `ad-div`
- 398-407: `ad-sin`, `ad-cos`, `ad-exp`, `ad-log`, `ad-sqrt`, `ad-neg`,
  `ad-abs`, `ad-relu`, `ad-sigmoid`, `ad-tanh`
- 408: `ad-backward` (run backward pass, fill `.gradient` fields)
- 409: `ad-gradient` / `ad-gradient-of` (read gradient of node by index)
- 1841: `ad-tape-release` (standalone VM logical handle release; idempotent)
- 1842: `ad-node-value` / `ad-value` / `ad-value-of` (read forward value)
- 1843: `ad-tape-length`
- 1844: `ad-pow`

The standalone VM allocates AD tapes in its arena-backed region stack, so
`ad-tape-release` invalidates the VM tape handle and makes double release safe.
The LLVM/AOT runtime path still uses the owned main-arena scope release for
iterative training loops that need memory reclamation.

---

# Part B (continued): New Missing Items From Code Audit

## B.24 Geometric Manifold VM Surface

`lib/backend/vm_geometric.c` exposes geometric native calls in the standalone
VM at IDs 804-861. The source VM now registers these names directly. When
`ESHKOL_GEOMETRIC_ENABLED` is not linked, the standalone VM uses a portable
constant-curvature fallback with arena-backed manifold handles, tensor-returning
metric/connection/form operations, Euclidean approximations for map/transport
operations, SO3/SE3 quaternion helpers, and curvature adaptation primitives.

A higher-level `lib/agent/geometry.esk` convenience module is still useful, but
the raw builtin surface is no longer undefined:

### Manifold creation
```scheme
(make-euclidean-manifold dim)       ;; native 804
(make-hyperbolic-manifold dim curv) ;; native 805
(make-spherical-manifold dim)       ;; native 806
(make-product-manifold m1 m2)       ;; native 807
(manifold-curvature m)              ;; native 808
(manifold-type m)                   ;; native 857
(manifold-dim m)                    ;; native 858
(manifold-destroy! m)               ;; native 859
```

### Hyperbolic geometry
```scheme
(hyperbolic-exp-map base tangent curv)   ;; native 809
(hyperbolic-log-map base point curv)     ;; native 810
(geodesic-distance x y curv)            ;; native 811
(parallel-transport x y v curv)         ;; native 812
(manifold-project x curv)              ;; native 813
(mobius-add x y curv)                   ;; native 814
(mobius-scalar-mul r x curv)            ;; native 815
(poincare-distance x y curv)            ;; native 816
(frechet-mean points weights curv)      ;; native 817
```

### Spherical geometry
```scheme
(great-circle-distance x y)             ;; native 819
(slerp x y t)                           ;; native 820
(spherical-exp base tangent)            ;; native 821
(spherical-log base point)              ;; native 822
(spherical-project x)                   ;; native 823
```

### Lie groups (SO3/SE3)
```scheme
;; SO3: rotation as quaternion #(w x y z)
(so3-exp omega)              ;; native 824 — axis-angle [3] → quat [4]
(so3-log quat)               ;; native 825 — quat [4] → axis-angle [3]
;; SE3: rigid transform as #(qw qx qy qz tx ty tz)
(se3-exp twist)              ;; native 826 — twist [6] → pose [7]
(se3-log pose)               ;; native 827 — pose [7] → twist [6]
(quaternion-mul q1 q2)       ;; native 828 — Hamilton product
```

### Differential geometry
```scheme
(metric-tensor m)            ;; native 829 — fallback returns identity metric tensor
(christoffel m point)        ;; native 830
(riemann-curvature m)        ;; native 831
(ricci-scalar m)             ;; native 832
(sectional-curvature m u v)  ;; native 833
```

### Differential forms
```scheme
(wedge-product alpha beta)          ;; native 834
(exterior-derivative form)          ;; native 835
(hodge-star form metric)            ;; native 836
(interior-product vector form)      ;; native 837
(pullback form jacobian)            ;; native 838
```

### Riemannian optimization
```scheme
(riemannian-sgd-step point grad lr curv)           ;; native 839
(riemannian-adam-step pt grad lr b1 b2 curv)       ;; native 840 — default Adam state
(make-riemannian-adam-state point)                 ;; native 860
(riemannian-adam-step! state pt grad lr b1 b2 curv);; native 861 — explicit Adam state
(riemannian-grad euclidean-grad point curv)         ;; native 841
(retraction base tangent curv)                     ;; native 842
(vector-transport x y v curv)                      ;; native 843
```

### Geodesic attention
```scheme
(geodesic-attention-scores Q K curv)               ;; native 844
(geodesic-attention-values scores V curv)           ;; native 845 — Fréchet mean
(curvature-softmax scores curv)                    ;; native 846
(geodesic-attention-forward Q K V curv)            ;; native 847 — full attention
```

### Adaptive curvature
```scheme
(set-curvature! m new-c)              ;; native 850
(get-curvature m)                     ;; native 851
(curvature-gradient m loss-grad)      ;; native 852
(transition-geometry! m target rate)  ;; native 853
(manifold-interpolate m1 m2 t)        ;; native 854
(curvature-hessian m grad)            ;; native 855
(adaptive-curvature-step m grad)      ;; native 856
```

---

## B.25 Reverse-Mode AD Tape Eshkol Wrappers

A.8 is wired in the VM. A higher-level convenience wrapper module is still
useful; create `lib/agent/autodiff-tape.esk`:

```scheme
(define (with-tape f)
  "Call f with a fresh tape. Returns (cons tape result)."
  (let ((tape (ad-tape-new)))
    (cons tape (f tape))))

(define (tape-forward tape expr-fn . initial-values)
  "Record forward pass. initial-values become ad-var nodes."
  (let ((vars (map (lambda (v) (ad-var tape v)) initial-values)))
    (apply expr-fn vars)))

(define (tape-gradients tape output-node)
  "Run backward pass. Returns list of gradients for all vars."
  (ad-backward tape output-node)
  (map (lambda (i) (ad-gradient-of tape i))
       (iota (ad-tape-length tape))))
```

---

## B.26 Workspace Introspection

The standalone VM exposes the workspace introspection surface directly:

```scheme
(ws-step! ws)                ;; native 543 — run one cognitive cycle
(ws-get-content ws)          ;; native 544 — returns content tensor
(ws-set-content! ws tensor)  ;; native 545 — overwrite content
(ws-get-dim ws)              ;; native 546 — content dimension
(ws-get-step-count ws)       ;; native 547 — cognitive cycle counter
```

**Use case**: The agent can query `(ws-get-step-count mind)` to log how many
deliberation cycles it has performed per query, or `(ws-get-content mind)` to
snapshot the current workspace state.

---

## B.27 Factor Graph Marginals and Entropy

The standalone VM exposes the belief arrays and KB maintenance helpers:

### `(fg-marginal fg var-id) -> tensor` — native 1810
Return the posterior marginal P(X_var=k) for each state k as a 1D tensor.
Reads from `fg->beliefs[var_id][]`, exponentiates from log-space, normalizes,
and allocates the result tensor in the VM arena. This path now supports
variables with more than 64 states.

### `(fg-entropy fg var-id) -> float` — native 1811
Shannon entropy of one variable marginal: `H = -sum_s b(v,s) * log(b(v,s))`.

### `(fg-total-entropy fg) -> float` — native 1812
Sum of all variable marginal entropies:
`H = -sum_v sum_s b(v,s) * log(b(v,s))`.

### `(fg-observe! fg var-id state) -> bool` — native 527
Clamp a variable to an observed state before rerunning inference.

### `(kb-retract! kb fact) -> bool` — native 1801
Remove a fact from the knowledge base by structural equality of the stored
Scheme fact datum, then swap with the last fact and decrement the count.

### `(kb-count kb) -> integer` — native 1800
Count all facts in the knowledge base.

### `(kb-count-predicate kb predicate) -> integer` — native 1802
Count facts whose datum starts with the given predicate symbol.

The older LLVM/AOT names continue to route through their existing codegen
surface; this section describes the standalone VM native IDs.

---

# Part C (continued): Geometric Capabilities Beyond Claude Code

---

## C.18 Hyperbolic Semantic Memory

Use the Poincaré ball model for embedding agent knowledge. Hyperbolic space
is exponentially more efficient than Euclidean for tree-structured data (which
describes most code hierarchies, file trees, and conversation trees).

```scheme
(require agent.geometry)  ;; lib/agent/geometry.esk (new — B.24)

;; Embed concept hierarchy into Poincaré ball
(define *semantic-manifold* (make-hyperbolic-manifold 32 -1.0))
(define *concept-embeddings* (make-hash-table))  ;; string -> tensor

(define (embed-concept! name parent-name)
  "Place concept in hyperbolic space near parent."
  (let* ((parent-emb (if parent-name
                         (hash-ref *concept-embeddings* parent-name)
                         (zeros 32)))
         ;; Small random tangent from parent
         (tangent (tensor-scale (random-normal-tensor '(32)) 0.1))
         ;; Move along geodesic from parent
         (new-emb (hyperbolic-exp-map parent-emb tangent -1.0)))
    (hash-set! *concept-embeddings* name new-emb)))

(define (semantic-distance a b)
  "Hyperbolic distance between two concepts."
  (poincare-distance (hash-ref *concept-embeddings* a)
                     (hash-ref *concept-embeddings* b)
                     -1.0))
```

**Why this is better than Claude Code**: Claude Code stores no semantic memory
between turns. We can learn concept embeddings over the session and use geodesic
distance to find related concepts in O(1) time.

**Requires**: B.24 convenience wrapper plus VM builtins 804-817.

---

## C.19 Geodesic Attention in the Query Loop

Replace standard scaled dot-product attention in `query-loop.esk` with
geodesic attention. This is provably better for hierarchical data because
distances in hyperbolic space reflect tree structure.

```scheme
;; In query-loop.esk, when compressing context:
(define (geodesic-compress-context messages curv)
  "Compress context using hyperbolic attention. Returns top-k messages."
  (let* ((embeddings (map message->tensor messages))
         (E (list->tensor embeddings))   ;; N×dim matrix
         ;; Self-attention with hyperbolic distances
         (scores (geodesic-attention-scores E E curv))
         (weights (curvature-softmax (tensor-row-sum scores) curv))
         ;; Weight messages by attention
         (sorted (sort (zip weights messages) (lambda (a b) (> (car a) (car b)))))
         (top-k (take sorted (min *max-context-messages* (length sorted)))))
    (map cdr top-k)))
```

**Requires**: builtins 844-847 (geodesic attention forward pass).

---

## C.20 Riemannian Self-Optimization

Use Riemannian gradient descent to optimize agent parameters (embedding
curvature, temperature, retry delays) on the hyperbolic manifold:

```scheme
;; Optimize curvature of the semantic manifold based on retrieval quality
(define (adapt-curvature! manifold retrieval-loss)
  (let* ((curv (get-curvature manifold))
         (curv-grad (curvature-gradient manifold retrieval-loss))
         (lr 0.01))
    ;; Gradient step on curvature parameter
    (transition-geometry! manifold
                          (- curv (* lr curv-grad))
                          0.1)))  ;; Lerp rate

;; Optimize embedding of a concept using Riemannian SGD
(define (update-concept-embedding! name error-grad)
  (let* ((emb (hash-ref *concept-embeddings* name))
         ;; Convert Euclidean gradient to Riemannian gradient
         (riem-grad (riemannian-grad error-grad emb -1.0))
         ;; Retraction step (stays on manifold)
         (new-emb (riemannian-sgd-step emb riem-grad 0.01 -1.0)))
    (hash-set! *concept-embeddings* name new-emb)))
```

**Requires**: builtins 835-839, 850-854.

---

## C.21 Symbolic Autodiff for Compiled Gradient Code

`lib/backend/vm_symbolic_ad.c` implements:
- `sym_trace_bytecode`: Trace VM bytecode to symbolic expression graph
- `sym_differentiate`: Symbolic differentiation with full rules (sin, cos, exp, log, sqrt, pow, product rule, quotient rule, chain rule)
- `sym_simplify`: Algebraic simplification (constant folding, x+0=x, x*1=x, --x=x, exp(log(x))=x etc.)
- `sym_compile`: Compile symbolic expression back to optimized bytecode

This is **not exposed as VM builtins at all** — it exists only as internal C.

**What to expose** (new native IDs, e.g. 760-769):

```scheme
(sym-build-var tape var-id)              ;; Create symbolic variable
(sym-build-const tape value)             ;; Create symbolic constant
(sym-build-op tape op left right)        ;; Build expression tree node
(sym-differentiate tape output-node var-id)  ;; Symbolic differentiation
(sym-simplify tape node)                 ;; Algebraic simplification
(sym-eval tape node values)             ;; Evaluate with concrete values
```

**Use case in agent**: Pre-compile gradient computations for hot paths:
```scheme
;; Compile token estimation gradient once, use many times
(define *token-gradient-fn*
  (sym-simplify
    (sym-differentiate *token-cost-expr* var-token-count)))
```

---

## C.22 Quantum RNG for Secure Session IDs

`lib/random/random.esk` already provides `qrandom`, `qrandom-int`, `qrandom-bool`,
`qrandom-choice`. The underlying `quantum_rng_wrapper.c` uses `eshkol_qrng_*`
which simulates an 8-qubit quantum circuit.

**Currently used**: Nowhere in eshkol-agent.

**How to use**:
```scheme
(require random)  ;; loads lib/random/random.esk

;; Cryptographically strong session ID
(define (make-session-id)
  (string-append
    (number->string (qrandom-int 0 #xFFFFFFFF) 16)
    (number->string (qrandom-int 0 #xFFFFFFFF) 16)
    (number->string (qrandom-int 0 #xFFFFFFFF) 16)
    (number->string (qrandom-int 0 #xFFFFFFFF) 16)))

;; Quantum random exponential backoff (avoids thundering herd)
(define (quantum-backoff-delay attempt base-ms max-ms)
  (min max-ms (* base-ms (expt 2 attempt) (quniform 0.5 1.5))))
```

**VM status**: `quantum-random`, `quantum-random-int`, and
`quantum-random-range` are exposed in the standalone VM as native IDs
1860-1862. The VM implementation uses a local xorshift generator seeded from
time, PID, and process address entropy; LLVM/AOT still routes through the
existing quantum RNG runtime.

---

## C.23 Distributed Multi-Agent Gradient Sync

`lib/backend/qllm_distributed.c` implements:
- Ring-AllReduce for gradient averaging across workers
- Top-K sparsification (only send the largest gradients)
- Parameter server mode

Currently not exposed at all. For swarm/multi-agent mode (Step 17 of the plan),
agents that share a learned model need to synchronize:

```scheme
;; New native IDs (e.g. 900-909):
(dist-init world-size rank)              ;; 900 — initialize distributed state
(dist-allreduce! grads-tensor)           ;; 901 — average gradients in-place
(dist-broadcast! tensor src-rank)        ;; 902 — broadcast from one worker
(dist-barrier!)                          ;; 903 — synchronize all workers
(dist-destroy! state)                    ;; 904
```

**Use case**: Multiple agent instances collaboratively fine-tune shared
embeddings. Each agent processes different tools/queries, computes local
gradients, then AllReduce averages them.

---

## C.24 WASM Browser REPL

`lib/backend/vm_wasm_repl.c` compiles with Emscripten and exports:
- `repl_init()` — initialize global VM session
- `repl_eval(source)` — evaluate Eshkol source, return result string

This is **already complete**. We get a browser-based Eshkol REPL for free.

**What to do**: Create `src/browser.esk` that defines the agent's browser
interface. Users could run the agent entirely in a web browser.

---

## C.25 SO3/SE3 for Spatial Tool Integration

The Lie group operations (builtins 820-824) enable:

```scheme
;; Tool: rotate-3d — apply rotation to spatial data
(define (rotate-tool ctx input)
  (let* ((quat (so3-exp (tensor input 'axis-angle)))
         (points (tensor input 'points))
         ;; Apply rotation to each point via quaternion sandwich product
         (rotated (tensor-map
                    (lambda (p) (quaternion-rotate quat p))
                    points)))
    (ok (tensor->list rotated))))

;; Tool: transform-frame — rigid body pose composition
(define (compose-frames pose1 pose2)
  (se3-exp
    (tensor-add
      (se3-log pose1)
      (se3-log pose2))))
```

**Use case**: Computer use tool, 3D visualization, robot arm integration.

---

## C.26 Differential Forms for Code Topology

Exterior calculus operations (builtins 830-834) can analyze the
topological structure of code dependency graphs:

```scheme
;; Represent import graph as a differential form
(define (import-graph->1-form files)
  "Create a 1-form representing import relationships."
  (let ((n (length files)))
    (let ((adj (make-tensor n n)))
      (for-each (lambda (i)
                  (for-each (lambda (j)
                              (when (imports? (list-ref files i) (list-ref files j))
                                (tensor-set! adj i j 1.0)))
                            (iota n)))
                (iota n))
      adj)))

;; Exterior derivative = "how does the dependency pattern change"
(define (dependency-gradient import-form)
  (exterior-derivative import-form))

;; Wedge product = "combined dependency structure"
(define (combined-dependencies f1 f2)
  (wedge-product f1 f2))
```

---

## C.27 Live Curvature-Adaptive Attention

During long conversations, the geometry of the semantic space changes as
new concepts are introduced. Use adaptive curvature (builtins 850-854) to
continuously tune the manifold:

```scheme
(define (cognitive-loop mind)
  (let loop ((step 0))
    (ws-step! (mind-workspace mind))
    ;; Every 10 steps, adapt curvature based on retrieval quality
    (when (zero? (modulo step 10))
      (let* ((content (ws-get-content (mind-workspace mind)))
             (loss (compute-retrieval-loss content *recent-queries*))
             (manifold (mind-manifold mind)))
        (adapt-curvature! manifold loss)))
    (loop (+ step 1))))
```

---

# Updated Summary Table

| Section | Category | Count | Status |
|---------|----------|-------|--------|
| **A.1** | Fix `directory-entries` | 1 | Implemented |
| **A.2** | Fix `command-line` | 1 | Implemented |
| **A.3** | Fix parallel primitives (620-628) | 9 | Partial thread-pool path |
| **A.4** | Fix `term-cursor-pos` | 1 | Implemented |
| **A.5** | Fix Riemannian Adam (840) | 1 | Implemented |
| **A.6** | Fix Christoffel symbols (830) | 1 | Implemented |
| **A.7** | Fix pullback (838) | 1 | Implemented |
| **A.8** | Wire reverse-mode AD tape (390-409) | 20 | Implemented |
| **B.1-B.23** | Original missing functions | 148 | Truly missing |
| **B.24** | Geometric manifold wrappers | 40 | VM surface registered (804-861); portable fallback implemented; wrapper still useful |
| **B.25** | Reverse-mode AD tape wrappers | 5 | A.8 VM surface implemented |
| **B.26** | Workspace introspection (543-546) | 4 | C exists, no .esk |
| **B.27** | Factor graph marginals + KB extensions | 6 | Implemented |
| **C.1-C.17** | Original Beyond Claude Code | ~30 | Already in Eshkol |
| **C.18** | Hyperbolic semantic memory | integration | Uses 804-817 |
| **C.19** | Geodesic attention in query loop | integration | Uses 844-847 |
| **C.20** | Riemannian self-optimization | integration | Uses 839-854 |
| **C.21** | Symbolic autodiff compilation | 6 new natives | vm_symbolic_ad.c |
| **C.22** | Quantum RNG for security | integration | random.esk exists |
| **C.23** | Distributed multi-agent grad sync | 5 new natives | qllm_distributed.c |
| **C.24** | WASM browser REPL | integration | vm_wasm_repl.c exists |
| **C.25** | SO3/SE3 for spatial tools | integration | Uses 820-824 |
| **C.26** | Differential forms for code topology | integration | Uses 830-833 |
| **C.27** | Live curvature-adaptive attention | integration | Uses 850-854 |
| | **Grand total new functions** | **~260** | |

---

# Part D: Architecture & Ecosystem (Added via codebase audit)

## D.1 Dual-Backend Architecture

Eshkol has TWO backends that BOTH need these additions:

### LLVM Compiler — The Agent's Body
The agent itself (I/O, tools, TUI, HTTP, subprocess) compiles via LLVM to native
code. New capabilities use `(extern ret-type name args :real c_func)` FFI — no
compiler changes needed. The LLVM backend already has many B-section functions
(file-exists?, file-rename, file-size, directory-list, make-directory, etc.) in
`system_codegen.cpp`. These only need VM parity, not new LLVM work.

### VM Bytecode — Computable Transformer Weights
The VM is the core of the **computable transformer model** (`weight_matrices.c`).
The transformer's weights ARE a universal stack machine interpreter:
- Weights: analytically constructed float matrices (d_model=256, 6 layers, FFN_DIM=2304)
- Programs: bytecode embedded as attention values `pe[pos][S_OPCODE/S_OPERAND]`
- Execution: each forward pass = one VM step (attention fetch → gated FFN dispatch)
- Three-way verified: reference C ≡ simulated transformer ≡ matrix forward pass

The VM builtins (tensors, autodiff, KB, factor graphs) extend the 14-opcode base
so programs running THROUGH transformer weights can access higher-level operations.
This is why STDLIB_ADDITIONS.md correctly targets vm_native.c case numbers.

**Both backends need parity:**
- LLVM: agent tools at native speed (`extern` FFI path)
- VM: hypothesis testing, fold-regrow cycles, meta-learning via transformer weights

---

## D.2 Ecosystem: Already Implemented in semiclassical_qllm

`~/Desktop/semiclassical_qllm/src/agent/system_bridge.c` already provides:

| Function | What It Does | Eliminates |
|----------|-------------|-----------|
| `eshkol_mkdir_p(path)` | Recursive mkdir | B.1 `mkdir-recursive` for LLVM path |
| `eshkol_file_exists(path)` | File existence | (also in LLVM backend) |
| `eshkol_system_capture(cmd,buf,max)` | Shell + capture | Shell-out replacement |
| `eshkol_json_parse(str)` | JSON parse → handle | B.12 partial |
| `eshkol_json_get_string(obj,key)` | JSON field access | B.12 partial |
| `eshkol_json_get_number(obj,key)` | JSON number access | B.12 partial |
| `eshkol_json_free(obj)` | JSON cleanup | |
| `eshkol_socket_connect(path)` | Unix domain socket | B.4 unix socket |
| `eshkol_socket_send(sock,data,len)` | Socket write | B.4 |
| `eshkol_socket_recv(sock,buf,max)` | Socket read | B.4 |
| `eshkol_socket_close(sock)` | Socket cleanup | B.4 |
| `eshkol_signal_install(signum)` | Signal handler reg | NEW (was missing) |
| `eshkol_signal_check()` | Check if signaled | NEW (was missing) |

`agent_bridge.c` provides **54 Selene FFI functions** for consciousness,
geometric inference, training, GeoRefine, and soul persistence — all fully
implemented and tested.

Link via: `-lsemiclassical_qllm -L ~/Desktop/semiclassical_qllm/build/lib/`

---

## D.3 Attention/Tsotchke Ecosystem Requirements

`~/Desktop/attention/tsotchke/eshkol_stdlib/` contains 19 .esk files (3,293 LOC)
that define what the full Tsotchke Agent OS needs from Eshkol:

### Required (verified working in current Eshkol):
- Math: det, inv, solve, power-iteration, covariance, variance, std
- ML: gradient-descent, adam, gradient, hessian, jacobian
- Tensor: tensor-dot, tensor-norm, vec-add, vec-sub, vec-scale
- KB: make-kb, kb-assert!, kb-query
- Factor graph: make-factor-graph, fg-add-factor!, fg-infer!, fg-update-cpt!, expected-free-energy
- FFI: extern declarations with ptr/i32/i64/f64/void

### Additional requirements NOT in Parts A-C:

### `(fg-marginal fg var-id) -> tensor` — CRITICAL
Read posterior beliefs after inference. The beliefs exist in C (`fg->beliefs[]`)
but cannot be read from Eshkol. This blocks the entire consciousness activation
chain. **Native ID 528 in VM. Needs C runtime function for LLVM.**

### `(kb-save path) -> bool` and `(kb-load path) -> kb` — CRITICAL
Persist knowledge base across sessions. Without this, all learned facts reset on
every agent restart. Format: JSON array of `{"pred":"...","args":[...]}`.

### `(tensor-save tensor path) -> bool` and `(tensor-load path) -> tensor`
Persist learned embeddings and model weights across sessions.
Format: 4-byte magic "ESHT" | 4-byte ndim | ndim*8 shape | raw float64 data.

### 12-Dimensional Emotional State (pure Eshkol)
Vector of 12 floats: pleasure, curiosity, focus, confidence, frustration,
boredom, satisfaction, anxiety, excitement, calm, determination, wonder.
Gaussian fluctuation + decay to baseline + user mood contagion.
~50 lines of pure Eshkol, no C needed.

### Global Workspace Lateral Inhibition
`ws-step!` exists but needs: winner boost (+0.05), loser inhibition (-0.02),
broadcast history (last 100). Pure Eshkol wrapper over existing VM builtins.

---

# Part E: Missing Functions Found in Audit (not in Parts A-C)

## E.1 IO Multiplexing — P0 CRITICAL

`(poll fd-list timeout-ms) -> list-of-ready-fds`

Agent's MCP transport busy-waits with `(sleep 0.05)` in a loop. Cannot handle
SSE + subprocess + keyboard simultaneously.

**VM**: Implemented in the standalone VM as an alias of `io-poll` (native ID
1783). Wraps `poll(2)` and returns a list of ready file descriptors.
**LLVM**: `extern "C" int eshkol_poll(int* fds, int* events, int nfds, int timeout_ms, int* ready)`

## E.2 Process Spawn With Environment — P0

`(process-spawn-with-env command cwd env-alist) -> proc`

`subprocess.esk:41` passes `#f` for the env parameter to `qllm_process_spawn`.
The C function already accepts env — just stop ignoring it.

**VM**: The standalone VM exposes `process-spawn-with-env` as an alias of the
env-aware `process-spawn` native ID 1780. Its standalone signature is
`(process-spawn-with-env command argv-list env-alist)`.

## E.3 Signal Handling — P1

`(set-signal-handler! signal handler) -> old-handler`

Already in qLLM (`eshkol_signal_install`/`eshkol_signal_check`). Needs:
- `(extern ...)` declarations for LLVM path

**VM**: The standalone VM now exposes low-level polling primitives:
`signal-install`, `signal-check`, `signal-reset`, `signal-ignore`, and
`signal-count` (native IDs 1794-1798). These deliberately do not invoke Scheme
closures from async signal context; they mirror qLLM's flag-and-poll model.

## E.4 Atexit Handlers — P1

`(at-exit thunk) -> void`

Register cleanup closures for program exit (MCP shutdown, terminal restore,
session save).

## E.5 stderr / Error Port — P1

`(current-error-port) -> port` and `(display-error str) -> void`

All output currently goes to stdout. SDK/non-interactive mode can't distinguish
errors from results.

## E.6 getpid — P1

`(getpid) -> integer`

One-line C function. Needed for lockfiles, temp file uniqueness, MCP transport.
**Not** the same as B.3 `process-pid proc` (which gets subprocess PID).

## E.7 sleep-ms — P1

`(sleep-ms ms) -> void`

VM has NO sleep builtin. `eshkol_usleep` exists in LLVM's `platform_runtime.cpp`
but is not wired to the VM. Add case in vm_native.c.

## E.8 mkstemp / mkdtemp — P1

`(make-temp-file prefix suffix dir) -> path`
`(make-temp-dir prefix dir) -> path`

Race-free temp file creation. Eliminates `mkdir -p` shell-out in `temp.esk`.

## E.9 ANSI Escape Stripping — P1

`(ansi-strip str) -> string`

Pure Eshkol state machine. Needed for output capture and `string-display-width`.

Standalone VM native ID 1946 removes CSI/OSC and related ANSI escape sequences.

## E.10 Tree-sitter Integration — P1 (highest-impact single addition)

```scheme
(ts-parser-new language)    ; -> parser handle
(ts-parse parser source)    ; -> tree handle
(ts-node-type node)         ; -> string
(ts-node-text node source)  ; -> string
(ts-node-children node)     ; -> list
(ts-query-new lang pattern) ; -> query handle
(ts-query-matches query tree source) ; -> list of matches
```

10 C functions binding `libtree-sitter` + 10 language grammars. Enables
go-to-definition, find-references, structural rename WITHOUT external LSP.

## E.11 Dynamic Library Loading — P1

`(dlopen path) -> handle`, `(dlsym handle name) -> ptr`, `(dlclose handle)`

REPL JIT already uses dlsym internally. Expose for runtime binding to ANY C
library (tree-sitter, libgit2, etc.) without recompiling Eshkol.

## E.12 Format String Function — P2

`(format "~a: ~d items" name count) -> string`

Pure Eshkol. Eliminates 100+ nested `string-append` chains across the agent.

## E.13 number->string With Radix — P2

`(number->string n radix)` — radix 2-36

Implemented in standalone VM case 51 for integer radices 2-36; radix 10 keeps
the existing integer/float formatting path.

---

## Complete Priority List (Final)

**P0 — Unblocks everything (remaining items):**
- A.3 Fix parallel primitives (real concurrency)
- B.1 `mkdir-recursive`, `file-rename`, `file-size`, `directory-delete-recursive`
- B.7 `os-type`, `home-directory`, `executable-exists?`, `current-time-ms`
- B.22 `shell-quote`
- **E.1 IO multiplexing (poll)** ← CRITICAL NEW
- **E.2 process-spawn-with-env** ← CRITICAL NEW

**P1 — Core intelligence + production robustness (30+ items):**
- A.8 Wire reverse-mode AD tape (390-409)
- B.3 `process-pid`, `process-kill-tree`
- B.4 Full HTTP (all methods, response headers)
- B.6 `uuid-v4`, `base64url-encode/decode`
- B.8 `string-display-width`
- B.10 Channels, mutex
- B.12 `json-get-in`
- B.15 `make-pipe`, `make-line-reader`
- **B.21 Yoga layout engine** ← elevated from P2
- B.24 Geometric manifold wrappers
- B.26 Workspace introspection
- B.27 `fg-marginal`, `fg-entropy`, `kb-retract!`, `kb-count`
- **E.3 Signal handling** ← NEW
- **E.4 Atexit handlers** ← NEW
- **E.5 stderr/error-port** ← NEW
- **E.6 getpid** ← NEW
- **E.7 sleep-ms** ← NEW
- **E.8 mkstemp/mkdtemp** ← NEW
- **E.9 ANSI strip** ← NEW
- **E.10 Tree-sitter integration** ← NEW (highest-impact)
- **E.11 dlopen/dlsym** ← NEW
- **D.3 tensor-save/tensor-load** ← NEW (persistence)
- **D.3 kb-save/kb-load** ← NEW (persistence)
- **D.3 Consciousness activation** (wire fg-infer! → ws-step! → EFE)
- C.1 Tensor token estimation
- C.18 Hyperbolic semantic memory

**P2 — TUI parity + quality:**
- B.5 Terminal extensions (scroll, mouse, alt screen)
- B.16 Diff algorithm
- B.25 Reverse-mode AD tape wrappers
- **E.12 Format string function** ← NEW
- **E.13 number->string with radix** ← NEW
- A.5 Fix Riemannian Adam
- C.19 Geodesic attention
- C.20 Riemannian self-optimization

**P3 — Full parity + beyond:**
- A.6-A.7 Christoffel, pullback
- B.4 WebSocket, HTTP server
- B.14 Compression
- B.23 Native file watching
- C.21-C.27 Symbolic AD, quantum RNG, distributed sync, WASM, etc.
