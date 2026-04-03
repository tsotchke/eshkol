# Eshkol Standard Library Additions for Claude Code Parity

Every function listed below is required to achieve 100% behavioral parity with
Claude Code CLI. Each entry includes the signature, return type, behavior
specification, and the specific files in `eshkol-agent/` that need it.

**Existing builtins confirmed working** (NOT listed here): `getenv`, `read-file`,
`write-file`, `file-exists?`, `file-delete`, `append-file`, `current-seconds`,
`string-append`, `string-length`, `string-ref`, `substring`, `string=?`,
`string-contains`, `string-starts-with?`, `string-replace`, `string-trim`,
`string-split-ordered`, `string-join`, `string-last-index`, `json-parse`,
`json-stringify`, `json-get`, `make-hash-table`, `hash-ref`, `hash-set!`,
`hash-keys`, `make-vector`, `vector-ref`, `vector-set!`, `dynamic-wind`,
`guard`, `number->string`, `string->number`, `regex-compile`, `regex-match`,
`hmac-sha256`, `sha256`, `random-hex`, `process-spawn`, `process-wait`,
`process-kill`, `process-exit-code`, `process-read-all-stdout`,
`process-read-all-stderr`, `process-close-stdin`, `process-destroy`,
`process-write-stdin`, `process-read-stdout`, `process-read-stderr`,
`run-command-capture`, `http-init`, `http-get`, `http-post`,
`http-stream-open`, `http-stream-next`, `http-stream-done?`,
`http-stream-close`, `parallel-map`, `with-db`, `db-exec`,
`with-statement`, `stmt-bind`, `stmt-step`, `stmt-column-text`,
`term-init`, `term-raw-mode`, `term-cooked-mode`, `term-shutdown`,
`term-read-key-timeout`, `term-write`, `term-flush`, `term-move-to`,
`term-hide-cursor`, `term-show-cursor`, `term-clear`, `term-width`,
`term-height`, `term-resized?`, `make-kb`, `kb-assert!`, `kb-query`,
`make-fact`, `make-factor-graph`, `fg-add-factor!`, `fg-update-cpt!`,
`fg-infer!`, `make-workspace`, `ws-register!`, `ws-step!`.

---

## 1. Filesystem

### 1.1 `(mkdir-recursive path) -> bool`

Create a directory and all parent directories. Equivalent to POSIX
`mkdir -p`. Returns `#t` on success, `#f` on failure (e.g., permission
denied). If the directory already exists, returns `#t` (idempotent).

**Implementation**: Call `mkdir(2)` for each component. On `ENOENT`,
create the parent first (recursive). On `EEXIST`, succeed silently.

**Used by**: `lib/platform/fs.esk` (lines 102, 122, 143),
`lib/platform/temp.esk` (line 46), `lib/services/bridge/bridge.esk`
(line 52), `lib/agent/swarm/coordinator.esk` (lines 42, 78), and 15+
test files. Currently all shell out to `mkdir -p`.

---

### 1.2 `(file-rename old-path new-path) -> bool`

Atomically rename a file or directory. Equivalent to POSIX `rename(2)`.
Returns `#t` on success, `#f` on failure. If `old-path` and `new-path`
are on different filesystems, copies then deletes (not atomic in that
case). If `new-path` exists, it is replaced.

**Implementation**: Call `rename(2)`. On `EXDEV` (cross-device), fall
back to copy + unlink.

**Used by**: `lib/platform/fs.esk` (line 128) in `fs-write-atomic`
where we currently build `mv 'old' 'new'` shell commands. Every
atomic file save goes through this path.

---

### 1.3 `(file-size path) -> integer`

Return file size in bytes. Does NOT read the file contents into memory.
Returns the size as an exact integer, or `-1` on error (file not found,
permission denied).

**Implementation**: Call `stat(2)` and return `st_size`.

**Used by**: `lib/platform/fs.esk` (lines 169-183) where we currently
shell out to `wc -c`, write to a temp file, read the temp file, parse
the output, then delete the temp file. Six operations for one syscall.

---

### 1.4 `(file-stat path) -> alist-or-false`

Return file metadata as an association list. Returns `#f` if the file
does not exist or cannot be stat'd.

Return format:
```scheme
((size . 4096)
 (mtime . 1712150400)    ; modification time, Unix epoch seconds
 (ctime . 1712150000)    ; status change time
 (atime . 1712150400)    ; access time
 (mode . 33188)          ; raw mode bits (use with file-mode-* helpers)
 (type . "file")         ; "file", "directory", "symlink", "other"
 (uid . 501)
 (gid . 20)
 (inode . 12345678)
 (nlinks . 1))
```

**Implementation**: Call `lstat(2)`. Decode `st_mode` to determine type.

**Used by**: `lib/tools/file-edit.esk` (concurrent edit detection via
mtime comparison), `lib/tools/file-read.esk` (binary detection, type
checking), `lib/services/file-history.esk` (change detection).

---

### 1.5 `(file-lstat path) -> alist-or-false`

Same as `file-stat` but does NOT follow symlinks. Returns metadata
about the symlink itself. Used for symlink detection.

**Implementation**: Call `lstat(2)` (same as above, `file-stat` should
use `lstat` by default; provide `file-stat-follow` for following).

---

### 1.6 `(directory-list path) -> list-of-strings`

List the entries in a directory. Returns a list of filenames (NOT full
paths, just the entry names). Excludes `.` and `..`. Returns an empty
list if the directory is empty. Raises an error if `path` is not a
directory or cannot be read.

**Implementation**: Call `opendir(3)` + `readdir(3)` loop + `closedir(3)`.

**Used by**: `lib/tools/file-edit.esk` (line 549, similar file
suggestions), `lib/services/bridge/bridge.esk` (line 97, listing
message files), `lib/agent/swarm/coordinator.esk` (line 143, listing
agents), `lib/services/output-styles.esk` (line 111). All currently
shell out to `ls -1`.

---

### 1.7 `(directory-list-full path) -> list-of-alists`

List directory entries with full stat info. Each entry is:
```scheme
((name . "file.txt")
 (type . "file")
 (size . 4096)
 (mtime . 1712150400))
```

Sorted by name. More efficient than calling `directory-list` then
`file-stat` on each entry (one `getdents` + `fstatat` per entry instead
of multiple syscalls).

**Used by**: `lib/services/bridge/polling.esk` (line 146) where we
shell out to `ls -1t` to find the most recent message file.

---

### 1.8 `(directory-walk path callback max-depth) -> void`

Recursively walk a directory tree depth-first. Calls
`(callback entry-path entry-type depth)` for every entry. `entry-type`
is `"file"`, `"directory"`, or `"symlink"`. `max-depth` limits
recursion (`-1` for unlimited). Does NOT follow symlinks into
directories (prevents cycles).

**Implementation**: Recursive `opendir`/`readdir` with depth tracking.

**Used by**: `lib/services/memory/memory.esk` (line 83, discovering
memory files via `find`), `lib/tools/glob.esk` (recursive pattern
matching), `lib/tools/grep.esk` (recursive file search).

---

### 1.9 `(directory-delete-recursive path) -> bool`

Recursively delete a directory and all its contents. Returns `#t` on
success, `#f` on failure. MUST refuse to delete the following paths
(safety check): `/`, `/usr`, `/bin`, `/sbin`, `/etc`, `/var`, `/home`,
`/Users`, `/System`, `/Library`.

**Implementation**: Recursive `opendir`/`readdir`, `unlink` files,
`rmdir` empty directories bottom-up.

**Used by**: `lib/platform/temp.esk` (line 99, cleaning temp dirs),
`lib/services/bridge/bridge.esk` (line 68, cleaning message dirs),
`lib/agent/swarm/coordinator.esk` (line 65), and 15+ test cleanup
blocks. All currently shell out to `rm -rf`.

---

### 1.10 `(file-copy src dst) -> bool`

Copy a file from `src` to `dst`. Creates `dst` if it doesn't exist,
overwrites if it does. Preserves file permissions. Returns `#t` on
success.

**Implementation**: Open both files, read/write in chunks (e.g., 64KB),
close both. On macOS, can use `clonefile(2)` for CoW copy.

**Used by**: Needed for backup-before-edit in `lib/tools/file-edit.esk`,
session export in `lib/services/teleport/teleport.esk`.

---

### 1.11 `(file-chmod path mode) -> bool`

Set file permissions. `mode` is an integer (e.g., `#o755`, `#o644`).
Returns `#t` on success.

**Implementation**: Call `chmod(2)`.

**Used by**: Needed for `lib/services/keychain.esk` (securing keystore
files), making scripts executable after write, securing temp files.

---

### 1.12 `(symlink-create target link-path) -> bool`

Create a symbolic link at `link-path` pointing to `target`. Returns
`#t` on success.

**Implementation**: Call `symlink(2)`.

**Used by**: Needed for IDE lockfile management in
`lib/services/ide/ide.esk`, worktree management in
`lib/tools/worktree.esk`.

---

### 1.13 `(symlink-read path) -> string-or-false`

Read the target of a symbolic link. Returns the target path string, or
`#f` if `path` is not a symlink.

**Implementation**: Call `readlink(2)`.

**Used by**: Needed for resolving workspace paths in IDE integration.

---

### 1.14 `(realpath path) -> string-or-false`

Resolve a path to its canonical absolute form. Resolves all `.`, `..`,
and symlinks. Returns `#f` if the path does not exist.

**Implementation**: Call `realpath(3)`.

**Used by**: Needed for path canonicalization throughout the codebase,
ensuring safe-path checks work against resolved paths.

---

### 1.15 `(glob-match pattern path) -> bool`

Test if a filename matches a glob pattern. Patterns support:
- `*` — matches any sequence of non-`/` characters
- `**` — matches any sequence including `/` (recursive)
- `?` — matches exactly one non-`/` character
- `[abc]` — character class
- `[a-z]` — character range
- `{a,b,c}` — alternation
- `\x` — literal escape

Returns `#t` if `path` matches `pattern`.

**Implementation**: Can use `fnmatch(3)` for simple patterns, custom
implementation for `**` and `{}`.

**Used by**: `lib/tools/glob.esk` (the entire glob tool),
`lib/agent/permissions.esk` (path pattern matching for allow/deny rules),
`lib/services/settings.esk` (ignore patterns).

---

### 1.16 `(glob-expand pattern root-dir) -> list-of-paths`

Expand a glob pattern to all matching file paths under `root-dir`.
Returns a sorted list of absolute paths. Supports the full pattern
syntax of `glob-match` above.

**Implementation**: Walk directory tree, test each path against pattern.

**Used by**: `lib/tools/glob.esk` — would make the entire glob tool a
single native call instead of building and parsing `find` commands.

---

### 1.17 `(file-watch path callback) -> watcher`

Watch a file or directory for changes. Calls
`(callback event-type filename)` when a change is detected.
`event-type` is one of: `"change"`, `"rename"`, `"create"`, `"delete"`.
Returns a watcher handle.

**Implementation**: Use `kqueue` on macOS, `inotify` on Linux.

**Used by**: Needed for config hot-reload (`lib/services/settings.esk`),
team memory sync (Claude Code's `teamMemorySync/watcher.ts`), IDE
lockfile monitoring.

---

### 1.18 `(file-watch-recursive dir callback) -> watcher`

Recursively watch an entire directory tree. Same callback signature as
`file-watch`. Automatically watches new subdirectories as they appear.

**Implementation**: Recursive `kqueue`/`inotify` watches.

**Used by**: Workspace-level change detection for team sync.

---

### 1.19 `(file-unwatch watcher) -> void`

Stop watching. Frees kernel resources.

---

### 1.20 `(file-lock path) -> lock-or-false`

Acquire an exclusive advisory lock on a file. Returns a lock handle on
success, `#f` if the file is already locked. Non-blocking.

**Implementation**: Call `flock(2)` with `LOCK_EX | LOCK_NB`.

**Used by**: Needed for session database exclusive access
(`lib/services/session.esk`), settings concurrent write prevention,
IDE lockfiles.

---

### 1.21 `(file-unlock lock) -> void`

Release a file lock.

**Implementation**: Call `flock(2)` with `LOCK_UN`.

---

### 1.22 `(with-file-lock path thunk) -> value`

Execute `thunk` while holding an exclusive lock on `path`. Lock is
automatically released when `thunk` returns or throws (via
`dynamic-wind`). Returns whatever `thunk` returns.

---

### 1.23 `(file-mmap path offset length) -> bytevector-or-false`

Memory-map a region of a file. Returns a bytevector backed by the
mapped memory, or `#f` on failure. For read-only access to large files
without loading them entirely into heap memory.

**Implementation**: Call `mmap(2)` with `PROT_READ`, `MAP_PRIVATE`.

**Used by**: Needed for efficient large file hashing (`sha256-file`),
binary file inspection in `lib/tools/file-read.esk`.

---

### 1.24 `(file-munmap mapped-data) -> void`

Unmap a memory-mapped region.

---

---

## 2. Path Manipulation

### 2.1 `(path-join . components) -> string`

Join path components with the platform separator. Handles:
- Redundant slashes: `(path-join "/a/" "/b")` -> `"/a/b"`
- Empty components are skipped
- Absolute paths reset: `(path-join "a" "/b")` -> `"/b"`
- Trailing slash handling: `(path-join "a" "b/")` -> `"a/b/"`

**Implementation**: Iterate components, concatenate with `/`, normalize.

**Used by**: Would be used in virtually every file in the codebase.
Currently every file builds paths with `string-append`. This is the
single most-used path function in Claude Code (hundreds of call sites).

---

### 2.2 `(path-dirname path) -> string`

Return the directory component of a path.
- `(path-dirname "/a/b/c.txt")` -> `"/a/b"`
- `(path-dirname "/a")` -> `"/"`
- `(path-dirname "a")` -> `"."`
- `(path-dirname "/")` -> `"/"`
- `(path-dirname "")` -> `"."`

Currently partially implemented in `lib/platform/fs.esk` (line 201) as
a user-space function. Should be a builtin.

---

### 2.3 `(path-basename path) -> string`

Return the filename component.
- `(path-basename "/a/b/c.txt")` -> `"c.txt"`
- `(path-basename "/a/b/")` -> `"b"`
- `(path-basename "c.txt")` -> `"c.txt"`

**Used by**: Needed for file extension checking, display names.

---

### 2.4 `(path-extname path) -> string`

Return the file extension including the dot.
- `(path-extname "file.txt")` -> `".txt"`
- `(path-extname "file.tar.gz")` -> `".gz"`
- `(path-extname "Makefile")` -> `""`
- `(path-extname ".gitignore")` -> `""`

**Used by**: `lib/tools/file-read.esk` (content type detection),
`lib/tools/file-write.esk` (newline handling), syntax highlighting.

---

### 2.5 `(path-relative from to) -> string`

Compute the relative path from `from` to `to`.
- `(path-relative "/a/b" "/a/c/d")` -> `"../c/d"`
- `(path-relative "/a/b" "/a/b/c")` -> `"c"`

**Used by**: Needed for display paths in tool output (showing paths
relative to CWD).

---

### 2.6 `(path-resolve . components) -> string`

Resolve to an absolute path. Processes from right to left; the first
absolute component anchors the result. If no component is absolute,
prepends the current working directory.
- `(path-resolve "a" "b")` -> `"/cwd/a/b"`
- `(path-resolve "/a" "b" "../c")` -> `"/a/c"`

---

### 2.7 `(path-is-absolute? path) -> bool`

Check if a path starts with `/`.

---

### 2.8 `(path-normalize path) -> string`

Normalize a path by resolving `.` and `..` segments and collapsing
redundant slashes. Does NOT resolve symlinks (use `realpath` for that).
- `(path-normalize "/a/b/../c/./d")` -> `"/a/c/d"`
- `(path-normalize "a//b///c")` -> `"a/b/c"`

---

---

## 3. Process Management

### 3.1 `(process-pid proc) -> integer`

Get the PID of a spawned process handle.

**Used by**: `lib/tools/bash.esk` (reporting PIDs to user),
`lib/tools/agent.esk` (line 129, writing `.pid` files for background
tasks), `lib/tools/task-tools.esk` (line 234, background task tracking),
`lib/agent/swarm/coordinator.esk` (process group management).

---

### 3.2 `(process-running? proc) -> bool`

Check if a process is still running without blocking. Returns `#t` if
the process has not exited.

**Implementation**: Call `waitpid(2)` with `WNOHANG`. If returns 0,
still running.

**Used by**: Needed for polling background tasks, MCP server health
checks, REPL subprocess monitoring.

---

### 3.3 `(process-signal proc signal) -> bool`

Send a signal to a process. `signal` is an integer:
- `1` = SIGHUP
- `2` = SIGINT
- `9` = SIGKILL
- `15` = SIGTERM
- `18` = SIGCONT
- `19` = SIGSTOP
- `10` = SIGUSR1
- `12` = SIGUSR2

Returns `#t` if the signal was sent successfully.

Note: The existing `process-kill` may already do this. If so, ensure it
accepts an explicit signal number argument.

**Implementation**: Call `kill(2)`.

**Used by**: `lib/platform/process.esk` (currently calls `process-kill`
with implicit SIGTERM, then SIGKILL).

---

### 3.4 `(process-spawn-pty command cwd) -> proc`

Spawn a process with a pseudo-terminal (PTY) instead of pipes. The
child process sees a real terminal (`isatty(STDOUT_FILENO)` returns
true). This means:
- Commands output ANSI colors and progress bars
- `git log` uses pager mode
- Interactive prompts work
- `stty` settings apply

The returned handle supports the same read/write operations as a
pipe-based process.

**Implementation**: Call `forkpty(3)` (macOS) or `openpty(3)` +
`fork(2)` (Linux). Set up the PTY as the child's stdin/stdout/stderr.

**Used by**: Needed for `lib/tools/bash.esk` (running interactive
commands), `lib/tools/repl.esk` (Python/Node REPL).

---

### 3.5 `(process-setpgid proc pgid) -> bool`

Set the process group ID. If `pgid` is 0, sets the process group to
the process's own PID (making it a group leader).

**Implementation**: Call `setpgid(2)`.

**Used by**: Needed for `lib/tools/bash.esk` to manage process groups
so that killing a pipeline kills all processes in it.

---

### 3.6 `(process-kill-tree pid signal) -> bool`

Kill a process and ALL its descendants. Walks the process tree and
sends `signal` to every child, grandchild, etc.

**Implementation**: On macOS: `pgrep -P pid` recursively. On Linux:
read `/proc/PID/children`. Send signal bottom-up (children first).

**Used by**: `lib/tools/bash.esk` — when the user cancels a command,
the entire pipeline must die, not just the parent shell. Claude Code
uses the `tree-kill` npm package for this.

---

### 3.7 `(process-read-stdout-nonblocking proc max-bytes) -> string-or-false`

Non-blocking read from process stdout. Returns available data
immediately, or `#f` if no data is available (would block).

**Implementation**: `fcntl(fd, F_SETFL, O_NONBLOCK)` then `read(2)`.

**Used by**: `lib/services/mcp-transport.esk` — polling for JSON-RPC
messages without blocking the event loop.

---

### 3.8 `(process-read-stderr-nonblocking proc max-bytes) -> string-or-false`

Same as above for stderr.

---

### 3.9 `(process-env-set! proc name value) -> void`

Set an environment variable for a process BEFORE spawning it. This must
be called between creating the process configuration and actually
spawning. Alternative: accept an `env` alist parameter in
`process-spawn`.

**Used by**: Setting `PATH`, `HOME`, `TERM`, etc. for child processes
in the bash tool.

---

---

## 4. HTTP and Networking

### 4.1 `(http-request method url headers body timeout-ms) -> (status headers body)`

General HTTP request supporting ALL methods: `"GET"`, `"POST"`, `"PUT"`,
`"DELETE"`, `"PATCH"`, `"HEAD"`, `"OPTIONS"`. Currently only GET and
POST are available.

The return value must include response headers (as an alist of
`(name . value)` pairs), not just status and body.

**Implementation**: Extend the existing libcurl-based HTTP to support
all methods and return headers.

**Used by**: `lib/platform/http.esk` (line 104, currently returns error
for non-GET/POST). PUT needed for REST APIs, DELETE for cleanup
endpoints, PATCH for partial updates.

---

### 4.2 `(http-get-with-headers url headers timeout-ms) -> (status headers body)`

HTTP GET that accepts request headers. The existing `http-get` does NOT
accept headers. We must pass `Authorization`, `Content-Type`,
`X-Api-Key`, `anthropic-version`, and `anthropic-beta` headers.

**Implementation**: Extend existing `http-get` to accept a headers
alist.

**Used by**: `lib/platform/http.esk` (line 84), every API call to
Anthropic.

---

### 4.3 `(http-set-proxy proxy-url) -> void`

Configure an HTTP proxy for all subsequent requests. `proxy-url` is
like `"http://proxy.example.com:8080"` or `"socks5://..."`. Passing
`#f` disables the proxy.

**Implementation**: `curl_easy_setopt(CURLOPT_PROXY, ...)`.

**Used by**: Needed for enterprise environments. Claude Code supports
`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` env vars. The upstream proxy
system (`src/upstreamproxy/`) relies on this.

---

### 4.4 `(http-set-tls-client-cert cert-path key-path ca-path) -> void`

Configure mTLS (mutual TLS) for all subsequent HTTPS requests. Needed
for enterprise API endpoints that require client certificates.

**Implementation**: `curl_easy_setopt(CURLOPT_SSLCERT, ...)`,
`CURLOPT_SSLKEY`, `CURLOPT_CAINFO`.

**Used by**: Claude Code's `utils/mtls.ts` configures custom CAs and
client certs.

---

### 4.5 `(http-server-create handler) -> server`

Create an HTTP server. `handler` is
`(lambda (method path query headers body) -> (status headers body))`.

Example:
```scheme
(http-server-create
  (lambda (method path query headers body)
    (list 200 '(("Content-Type" . "text/html")) "<h1>OK</h1>")))
```

**Implementation**: Bind a TCP socket, accept connections, parse HTTP/1.1
requests, call handler, write response.

**Used by**: Needed for:
- OAuth PKCE callback receiver (`lib/services/oauth.esk` — Claude Code
  starts a local HTTP server on a random port to receive the auth code)
- MCP auth flow (`services/mcp/auth.ts`)
- IdP login redirect handler

---

### 4.6 `(http-server-listen server port) -> integer`

Start listening. If `port` is 0, picks a random available port. Returns
the actual port number.

---

### 4.7 `(http-server-close server) -> void`

Stop the HTTP server and free the socket.

---

### 4.8 `(websocket-connect url headers) -> ws-handle`

Open a WebSocket connection. Performs the HTTP upgrade handshake.
Returns a handle for sending/receiving frames.

**Implementation**: Can use libcurl WebSocket support (curl 7.86+) or
a lightweight custom implementation over raw TCP + TLS.

**Used by**: Needed for:
- Voice streaming STT (`services/voiceStreamSTT.ts` — WebSocket to
  transcription service)
- Remote session communication (`remote/SessionsWebSocket.ts` — with
  auto-reconnect and exponential backoff)
- MCP over WebSocket transport (`cli/transports/WebSocketTransport.ts`)

---

### 4.9 `(websocket-send ws data) -> bool`

Send a text frame. Returns `#t` on success.

---

### 4.10 `(websocket-send-binary ws data) -> bool`

Send a binary frame.

---

### 4.11 `(websocket-receive ws timeout-ms) -> (type . data)-or-false`

Receive a frame. `type` is `"text"`, `"binary"`, `"ping"`, `"close"`.
Returns `#f` on timeout.

---

### 4.12 `(websocket-close ws) -> void`

Send close frame and shut down the connection.

---

### 4.13 `(tcp-connect host port) -> socket-or-false`

Open a raw TCP connection. Returns a socket handle or `#f`.

**Implementation**: `socket(2)` + `connect(2)`.

**Used by**: Needed for Unix domain socket IPC to IDE extensions,
custom protocol implementations.

---

### 4.14 `(tcp-read socket max-bytes timeout-ms) -> string-or-false`

Read from a TCP socket with timeout.

---

### 4.15 `(tcp-write socket data) -> integer`

Write to a TCP socket. Returns bytes written.

---

### 4.16 `(tcp-close socket) -> void`

Close a TCP socket.

---

### 4.17 `(unix-socket-connect path) -> socket-or-false`

Connect to a Unix domain socket. Returns a socket handle.

**Used by**: Needed for IDE integration — VS Code and JetBrains expose
IPC handles via Unix domain sockets (`VSCODE_IPC_HOOK`, etc.).

---

---

## 5. Terminal

### 5.1 `(term-set-scroll-region top bottom) -> void`

Set the terminal scroll region using CSI DECSTBM (`\033[top;bottomr`).
Lines outside the region don't scroll. `top` and `bottom` are 1-based
line numbers.

**Used by**: Needed for TUI with fixed header/footer and scrollable
content area.

---

### 5.2 `(term-reset-scroll-region) -> void`

Reset scroll region to full terminal (`\033[r`).

---

### 5.3 `(term-enable-mouse) -> void`

Enable SGR mouse tracking (`\033[?1000h\033[?1006h`). Mouse events are
delivered as escape sequences readable via `term-read-key-timeout`.

**Used by**: Needed for click-to-position, scroll wheel, permission
dialog interaction.

---

### 5.4 `(term-disable-mouse) -> void`

Disable mouse tracking (`\033[?1000l\033[?1006l`).

---

### 5.5 `(term-read-mouse-event timeout-ms) -> event-or-false`

Read and parse a mouse event. Returns
`(button x y modifiers event-type)` or `#f`.
- `button`: 0=left, 1=middle, 2=right, 3=release, 64=scroll-up,
  65=scroll-down
- `x`, `y`: 1-based column and row
- `modifiers`: bitmask (4=shift, 8=alt, 16=ctrl)
- `event-type`: `"press"`, `"release"`, `"move"`, `"scroll"`

---

### 5.6 `(term-enable-alternate-screen) -> void`

Switch to alternate screen buffer (`\033[?1049h`). The primary screen
is preserved and restored when the alternate screen is disabled. The
TUI runs on the alternate screen so the user's shell history is intact
after exit.

---

### 5.7 `(term-disable-alternate-screen) -> void`

Switch back to primary screen buffer (`\033[?1049l`).

---

### 5.8 `(term-set-title title) -> void`

Set the terminal window/tab title via OSC 2 (`\033]2;title\033\\`).

**Used by**: Claude Code sets the title to show the current model and
working directory.

---

### 5.9 `(term-clipboard-write text) -> void`

Write text to system clipboard via OSC 52
(`\033]52;c;base64-data\033\\`).

**Used by**: Copy-to-clipboard in diff view and code blocks.

---

### 5.10 `(term-clipboard-read) -> string-or-false`

Read from system clipboard via OSC 52. Returns clipboard contents or
`#f` if unsupported by terminal.

---

### 5.11 `(term-hyperlink url text) -> string`

Generate an OSC 8 hyperlink escape sequence. Returns the ANSI string:
`\033]8;;url\033\\text\033]8;;\033\\`

Terminals that support this render `text` as a clickable link.

**Used by**: Needed for clickable file paths and URLs in output.

---

### 5.12 `(term-detect-capabilities) -> alist`

Detect terminal capabilities. Returns:
```scheme
((color-depth . 24)      ; 1, 4, 8, or 24
 (unicode . #t)
 (hyperlinks . #t)
 (mouse . #t)
 (images . #f)           ; sixel or kitty protocol
 (alternate-screen . #t)
 (title . #t)
 (clipboard . #f)        ; OSC 52 support
 (program . "iTerm2")    ; from TERM_PROGRAM
 (term . "xterm-256color"))
```

**Implementation**: Check `TERM`, `COLORTERM`, `TERM_PROGRAM`, and
optionally send device attribute queries.

---

### 5.13 `(term-bell) -> void`

Ring terminal bell (BEL character `\x07`).

**Used by**: Notification on task completion.

---

### 5.14 `(term-request-size) -> (width . height)`

Query the actual terminal size via `TIOCGWINSZ` ioctl. More reliable
than cached values from `term-width`/`term-height`.

---

---

## 6. Cryptography

### 6.1 `(sha256-hex data) -> string`

SHA-256 hash of a string. Returns 64-character lowercase hex string.

Note: `sha256` may already exist (used in `oauth.esk` and
`keychain.esk`). Confirm it's a builtin and document its signature.

**Used by**: `lib/services/oauth.esk` (line 80, PKCE challenge),
`lib/services/keychain.esk` (line 128, machine fingerprint), content
hashing, cache keys.

---

### 6.2 `(sha256-file path) -> string-or-false`

SHA-256 hash of a file WITHOUT reading it all into heap memory. Uses
streaming reads (e.g., 64KB chunks fed into the hash context).

**Implementation**: `SHA256_Init`, loop `read` + `SHA256_Update`,
`SHA256_Final`.

**Used by**: Large file fingerprinting, content-addressable caching.

---

### 6.3 `(base64-encode data) -> string`

Standard base64 encoding (RFC 4648 Section 4). Uses `A-Z`, `a-z`,
`0-9`, `+`, `/` with `=` padding.

**Implementation**: Can use OpenSSL `BIO_f_base64` or a simple
lookup-table implementation.

**Used by**: `lib/services/bridge/bridge.esk` (currently implemented
manually character-by-character), image content blocks in API messages,
OSC 52 clipboard, HTTP Basic auth.

---

### 6.4 `(base64-decode data) -> string-or-false`

Standard base64 decoding. Returns `#f` on invalid input.

---

### 6.5 `(base64url-encode data) -> string`

URL-safe base64 (RFC 4648 Section 5). Uses `-` and `_` instead of `+`
and `/`. NO padding.

**Used by**: `lib/services/bridge/bridge.esk` (JWT encoding),
`lib/services/oauth.esk` (PKCE code verifier/challenge).

---

### 6.6 `(base64url-decode data) -> string-or-false`

URL-safe base64 decoding.

---

### 6.7 `(random-bytes count) -> bytevector`

Generate `count` cryptographically secure random bytes. More general
than `random-hex` (which returns hex string). Returns a bytevector.

**Implementation**: Read from `/dev/urandom` or use
`CCRandomGenerateBytes` on macOS.

**Used by**: UUID generation, nonce generation, PKCE code verifier
(needs raw bytes for base64url encoding).

---

### 6.8 `(uuid-v4) -> string`

Generate a UUID v4 string: `"xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"`
where `y` is one of `8`, `9`, `a`, `b`.

**Implementation**: Generate 16 random bytes, set version and variant
bits, format as hex with dashes.

**Used by**: Session IDs, message IDs, request IDs, MCP JSON-RPC IDs.
Currently we construct IDs from `current-seconds` + counters, which
can collide.

---

### 6.9 `(constant-time-equal? a b) -> bool`

Constant-time string comparison. Compares every byte regardless of
early mismatches. Prevents timing side-channel attacks.

**Implementation**: XOR each byte pair, OR accumulate, return whether
accumulator is zero.

**Used by**: JWT signature verification, API key comparison, HMAC
verification.

---

---

## 7. System Information

### 7.1 `(os-type) -> string`

Returns `"darwin"`, `"linux"`, or `"windows"`.

**Implementation**: Compile-time constant or `uname(2)`.

**Used by**: `lib/services/keychain.esk` (line 36, currently shells
out to `uname -s`), platform-specific behavior everywhere (shell
selection, path formats, keychain backend).

---

### 7.2 `(os-arch) -> string`

Returns `"x86_64"`, `"aarch64"` / `"arm64"`, etc.

**Implementation**: `uname(2)` machine field.

**Used by**: Needed for native module selection, system prompt
generation.

---

### 7.3 `(home-directory) -> string`

Returns the user's home directory. More reliable than
`(getenv "HOME")` — uses `getpwuid(3)` as fallback.

**Used by**: 15+ files currently use `(or (getenv "HOME") "/tmp")`.
A proper builtin eliminates the fragile `/tmp` fallback.

---

### 7.4 `(current-directory) -> string`

Returns the current working directory.

**Implementation**: Call `getcwd(3)`.

**Used by**: Needed for path resolution, system prompt generation.

---

### 7.5 `(set-current-directory! path) -> bool`

Change the current working directory. Returns `#t` on success.

**Implementation**: Call `chdir(2)`.

**Used by**: `lib/tools/bash.esk` (the `cwd` parameter for commands).

---

### 7.6 `(setenv name value) -> void`

Set an environment variable for the current process and all future
child processes.

**Implementation**: Call `setenv(3)`.

**Used by**: Configuring child process environment, setting `TERM` for
subprocess PTY mode.

---

### 7.7 `(unsetenv name) -> void`

Remove an environment variable.

---

### 7.8 `(hostname) -> string`

Get the system hostname.

**Implementation**: Call `gethostname(3)`.

**Used by**: Session identification, bridge authentication, system
prompt generation.

---

### 7.9 `(username) -> string`

Get the current username.

**Implementation**: Call `getpwuid(getuid())` and return `pw_name`.

**Used by**: System prompt, keychain access. Currently uses
`(getenv "USER")` which may not be set in all environments.

---

### 7.10 `(executable-exists? name) -> bool`

Check if an executable exists in `PATH`. Returns `#t` if found.

**Implementation**: Split `PATH` on `:`, check each directory for the
executable with `access(2)` using `X_OK`.

**Used by**: `lib/tools/grep.esk` (checking for `rg`),
`lib/tools/glob.esk` (checking for `fd`),
`lib/tools/file-read.esk` (checking for `pdftotext`),
`lib/services/keychain.esk` (checking for `security` command).

---

### 7.11 `(executable-path name) -> string-or-false`

Find the full path of an executable in `PATH`. Returns the path or `#f`.

**Implementation**: Same as above but returns the found path.

---

### 7.12 `(current-time-ms) -> integer`

Millisecond-precision Unix timestamp.

**Implementation**: `gettimeofday(2)` or `clock_gettime(CLOCK_REALTIME)`.

**Used by**: Performance timing, rate limiting, token/second
calculation, retry delay calculation.

---

### 7.13 `(monotonic-time-ms) -> integer`

Monotonic clock in milliseconds. NOT affected by system clock changes
(NTP, manual adjustments). Suitable for measuring elapsed time.

**Implementation**: `clock_gettime(CLOCK_MONOTONIC)`.

**Used by**: Timeout tracking, performance measurement, debouncing.

---

### 7.14 `(prevent-sleep reason) -> handle-or-false`

Prevent the system from sleeping while long-running operations execute.
`reason` is a human-readable string for system logs.

**Implementation**: On macOS: `IOPMAssertionCreateWithName`. On Linux:
`systemd-inhibit` or D-Bus `Inhibit`.

**Used by**: Long-running agent operations (multi-file edits, large
code generation). Claude Code's `preventSleep.ts`.

---

### 7.15 `(allow-sleep handle) -> void`

Release the sleep prevention assertion.

---

### 7.16 `(temp-directory) -> string`

Return the system temp directory (e.g., `/tmp`, or `$TMPDIR`).

**Implementation**: Check `TMPDIR` env var, fall back to `/tmp`.

**Used by**: `lib/platform/temp.esk` (currently hardcodes `/tmp`).

---

---

## 8. String Operations

### 8.1 `(string-display-width str) -> integer`

Calculate the display width of a string in terminal columns:
- ASCII characters = 1 column
- East Asian wide characters (CJK Unified Ideographs, Katakana,
  Hangul, etc.) = 2 columns
- Emoji = 2 columns (most)
- Zero-width characters (combining marks, ZWJ, variation selectors) =
  0 columns
- ANSI escape sequences = 0 columns
- Tab = tabstop width (8 by default)

**Implementation**: Use Unicode East Asian Width property table (UAX
#11). Strip ANSI escapes first.

**Used by**: Critical for TUI rendering. Without this, CJK text and
emoji break column alignment in tables, diffs, status bar, markdown
rendering. Claude Code uses `get-east-asian-width`.

---

### 8.2 `(string-truncate-display str max-width suffix) -> string`

Truncate a string to fit in `max-width` terminal columns, respecting
wide characters. Appends `suffix` (e.g., `"..."`) if truncated. Never
splits a wide character.

---

### 8.3 `(url-encode str) -> string`

Percent-encode a string for URL use. Encodes all characters except
unreserved: `A-Z a-z 0-9 - _ . ~`. Spaces become `%20` (NOT `+`).

**Used by**: OAuth redirect URIs, web search queries, API URL construction.

---

### 8.4 `(url-decode str) -> string`

Decode percent-encoded string. Also decodes `+` as space (for
`application/x-www-form-urlencoded`).

---

### 8.5 `(url-parse str) -> alist-or-false`

Parse a URL into components:
```scheme
((scheme . "https")
 (host . "api.anthropic.com")
 (port . 443)
 (path . "/v1/messages")
 (query . "beta=true")
 (fragment . "section1")
 (userinfo . #f))
```

Returns `#f` if the URL is malformed.

**Used by**: `lib/services/deep-link/deep-link.esk` (URI parsing),
proxy configuration, API endpoint construction, MCP server URLs.

---

### 8.6 `(string-ends-with? str suffix) -> bool`

Check if string ends with suffix.

**Used by**: File extension checking, path matching.

---

### 8.7 `(string-index-of str substring start) -> integer-or-false`

Find first index of `substring` starting from position `start`.
Returns `#f` if not found. Unlike `string-contains` (which returns
bool), this returns the position.

---

### 8.8 `(string-last-index-of str substring) -> integer-or-false`

Find the last occurrence of `substring`. Confirm this is the same as
the existing `string-last-index`.

---

### 8.9 `(string-replace-all str old new) -> string`

Replace ALL occurrences of `old` with `new`. Confirm this exists
and handles edge cases (empty `old`, overlapping matches).

---

### 8.10 `(string-split str delimiter limit) -> list`

Split with optional `limit`. When `limit` is provided, splits at most
`limit` times, leaving the remainder in the last element.
- `(string-split "a:b:c:d" ":" 2)` -> `("a" "b" "c:d")`

---

### 8.11 `(string-pad-left str width char) -> string`

Left-pad a string to `width` with `char`.
- `(string-pad-left "42" 5 #\0)` -> `"00042"`

**Used by**: Line number formatting in file-read, table alignment.

---

### 8.12 `(string-pad-right str width char) -> string`

Right-pad for table column alignment.

---

### 8.13 `(string-repeat str count) -> string`

Repeat a string `count` times.
- `(string-repeat "ab" 3)` -> `"ababab"`

**Used by**: Drawing box borders, indentation levels, separator lines
in TUI.

---

### 8.14 `(string-trim-left str) -> string`

Trim whitespace from start only.

---

### 8.15 `(string-trim-right str) -> string`

Trim whitespace from end only.

---

### 8.16 `(string-upcase str) -> string`

Convert to uppercase.

---

### 8.17 `(string-downcase str) -> string`

Convert to lowercase.

**Used by**: Case-insensitive comparisons, command normalization.

---

### 8.18 `(char-alphabetic? ch) -> bool`

### 8.19 `(char-numeric? ch) -> bool`

### 8.20 `(char-whitespace? ch) -> bool`

### 8.21 `(char-upper-case? ch) -> bool`

### 8.22 `(char-lower-case? ch) -> bool`

Character classification predicates.

**Used by**: Token estimation content-type detection, input validation,
identifier parsing.

---

### 8.23 `(string->bytevector str encoding) -> bytevector`

Convert a string to a bytevector using the specified encoding
(`"utf-8"`, `"ascii"`, `"latin-1"`). Needed for binary protocol
operations and crypto.

---

### 8.24 `(bytevector->string bv encoding) -> string`

Convert a bytevector to a string.

---

### 8.25 `(bytevector->hex bv) -> string`

Convert a bytevector to a lowercase hex string.

---

### 8.26 `(hex->bytevector str) -> bytevector-or-false`

Parse a hex string into a bytevector.

---

---

## 9. Regex Extensions

### 9.1 `(regex-match-groups pattern str) -> list-or-false`

Match and return capture groups. Returns `#f` on no match, or a list:
```scheme
((full . "matched text")
 (groups . ("group1" "group2" ...))
 (start . 0)
 (end . 10))
```

**Implementation**: Use PCRE2 `pcre2_match` with ovector extraction.

**Used by**: `lib/tools/bash-security.esk` — extracting filenames from
redirect patterns, heredoc delimiters, brace expansion contents.

---

### 9.2 `(regex-match-all pattern str) -> list-of-matches`

Find ALL non-overlapping matches. Each match has the same structure as
`regex-match-groups`.

---

### 9.3 `(regex-replace pattern str replacement) -> string`

Replace the first match. `replacement` supports `$1`, `$2`
backreferences and `$0` for the full match.

---

### 9.4 `(regex-replace-all pattern str replacement) -> string`

Replace all matches with backreference support.

---

### 9.5 `(regex-split pattern str) -> list`

Split a string on regex pattern matches.

---

---

## 10. Concurrency and Coordination

### 10.1 `(make-channel capacity) -> channel`

Create a bounded message channel for thread-safe communication.
`capacity` is the buffer size (0 for unbuffered/synchronous).

**Used by**: Background task result delivery, MCP message routing,
swarm coordination, event bus between TUI and agent.

---

### 10.2 `(channel-send! ch value) -> void`

Send a value to a channel. Blocks if buffer is full.

---

### 10.3 `(channel-send-nonblocking ch value) -> bool`

Non-blocking send. Returns `#f` if buffer is full.

---

### 10.4 `(channel-receive ch timeout-ms) -> value-or-false`

Receive from channel with timeout. Returns `#f` on timeout.

---

### 10.5 `(channel-try-receive ch) -> value-or-false`

Non-blocking receive. Returns `#f` if empty.

---

### 10.6 `(channel-close! ch) -> void`

Close a channel. Further sends fail; receives drain remaining items.

---

### 10.7 `(make-mutex) -> mutex`

Create a mutual exclusion lock.

**Used by**: Session message list (concurrent tool results), tool
registry, cost tracker.

---

### 10.8 `(mutex-lock! m) -> void`

Acquire the mutex. Blocks if held by another thread.

---

### 10.9 `(mutex-unlock! m) -> void`

Release the mutex.

---

### 10.10 `(with-mutex m thunk) -> value`

Execute `thunk` while holding the mutex. Release guaranteed via
`dynamic-wind`.

---

### 10.11 `(make-condition-variable) -> cv`

Create a condition variable for thread signaling.

---

### 10.12 `(condition-wait cv mutex) -> void`

Atomically release `mutex` and wait on `cv`. Reacquires `mutex` on
return.

---

### 10.13 `(condition-signal cv) -> void`

Wake one thread waiting on `cv`.

---

### 10.14 `(condition-broadcast cv) -> void`

Wake all threads waiting on `cv`.

---

### 10.15 `(make-timer delay-ms callback) -> timer`

Schedule `callback` to run after `delay-ms` milliseconds.

**Used by**: Debouncing resize events, auto-save scheduling, idle
timeout, retry delay.

---

### 10.16 `(timer-cancel! timer) -> void`

Cancel a pending timer.

---

### 10.17 `(make-interval interval-ms callback) -> interval`

Repeat `callback` every `interval-ms` milliseconds.

**Used by**: Status line refresh, MCP health checks, session auto-save,
cost display.

---

### 10.18 `(interval-cancel! interval) -> void`

Stop a repeating interval.

---

### 10.19 `(atomic-ref-create initial-value) -> aref`

Create an atomic reference for lock-free shared state.

---

### 10.20 `(atomic-ref-get aref) -> value`

Get the current value atomically.

---

### 10.21 `(atomic-ref-set! aref value) -> void`

Set the value atomically.

---

### 10.22 `(atomic-ref-cas! aref expected new-value) -> bool`

Compare-and-swap. Set to `new-value` only if current value is
`expected`. Returns `#t` if swap occurred.

---

---

## 11. Date and Time

### 11.1 `(format-iso8601 epoch-seconds) -> string`

Format a Unix timestamp as ISO 8601 UTC: `"2026-04-03T15:30:00Z"`.

**Implementation**: `gmtime_r` + `strftime`.

**Used by**: Session timestamps, message IDs, log entries.

---

### 11.2 `(parse-iso8601 str) -> integer-or-false`

Parse an ISO 8601 string to Unix timestamp. Handles:
- `"2026-04-03T15:30:00Z"`
- `"2026-04-03T15:30:00+05:00"`
- `"2026-04-03"`

Returns `#f` on invalid input.

---

### 11.3 `(format-relative seconds-ago) -> string`

Format a duration as human-readable relative time:
- 30 -> `"30s ago"`
- 300 -> `"5m ago"`
- 7200 -> `"2h ago"`
- 86400 -> `"1d ago"`
- 604800 -> `"1w ago"`

**Used by**: Session list display, file modification times.

---

### 11.4 `(local-timezone-offset) -> integer`

Get local timezone offset in seconds east of UTC. E.g., EST = -18000.

**Implementation**: `localtime_r`, compute offset from `tm_gmtoff`.

---

---

## 12. JSON Extensions

### 12.1 `(json-get-in obj path default) -> value`

Deep nested access into parsed JSON:
```scheme
(json-get-in response '("usage" "input_tokens") 0)
```

`path` is a list of keys (strings) and indices (integers). Returns
`default` if any key is missing.

**Used by**: API response parsing throughout `lib/services/api-streaming.esk`
(currently chains multiple `json-get` calls).

---

### 12.2 `(json-stringify-pretty obj indent) -> string`

Pretty-print JSON with indentation. `indent` is the number of spaces
per level.

**Used by**: MCP protocol debugging, tool input/output display,
settings file writing (human-readable).

---

### 12.3 `(json-merge a b) -> object`

Deep merge two JSON objects. Values in `b` override values in `a`.
Nested objects are merged recursively. Arrays are NOT merged (b
replaces a).

**Used by**: Settings merge — project settings override global settings
in `lib/services/settings.esk`.

---

### 12.4 `(json-type obj) -> string`

Return the JSON type: `"object"`, `"array"`, `"string"`, `"number"`,
`"boolean"`, `"null"`.

**Used by**: Type checking in message parsing, API response validation.

---

---

## 13. SQLite Extensions

### 13.1 `(db-transaction db thunk) -> result`

Execute `thunk` inside a SQLite transaction. Calls `BEGIN` before
`thunk`, `COMMIT` on success, `ROLLBACK` on error. Returns whatever
`thunk` returns.

**Used by**: `lib/services/session.esk` (line 66, atomic session save).

---

### 13.2 `(db-busy-timeout db ms) -> void`

Set the SQLite busy timeout. When another process holds a lock, SQLite
will retry for up to `ms` milliseconds before returning `SQLITE_BUSY`.

**Used by**: Needed for concurrent session access from multiple agent
instances.

---

### 13.3 `(db-last-insert-id db) -> integer`

Get the rowid of the last INSERT statement.

**Used by**: Session and message creation.

---

### 13.4 `(db-changes db) -> integer`

Number of rows changed by the last INSERT/UPDATE/DELETE.

**Used by**: Verifying updates succeeded.

---

---

## 14. Compression

### 14.1 `(deflate data) -> bytevector`

Compress data using zlib deflate.

**Implementation**: Use `zlib` library (`deflate` function).

**Used by**: Token-efficient message encoding. Claude Code uses
`zlib.deflateSync` in `slowOperations.ts` for payload compression.

---

### 14.2 `(inflate compressed) -> bytevector-or-false`

Decompress zlib deflate data. Returns `#f` on invalid input.

---

### 14.3 `(gzip data) -> bytevector`

Gzip compression (deflate with gzip header).

**Used by**: HTTP content-encoding.

---

### 14.4 `(gunzip compressed) -> bytevector-or-false`

Gzip decompression.

---

---

## 15. Streams and Pipes

### 15.1 `(make-pipe) -> (read-end . write-end)`

Create an in-process pipe pair. Data written to `write-end` can be read
from `read-end`. Both ends are file-descriptor-backed.

**Implementation**: Call `pipe(2)`.

**Used by**: Routing subprocess output through transforms, internal
message routing.

---

### 15.2 `(make-line-reader fd-or-port callback) -> reader`

Read from a file descriptor or port line-by-line, calling
`(callback line)` for each complete `\n`-terminated line. Handles
partial reads (buffers incomplete lines until `\n` arrives).

**Implementation**: Read into buffer, scan for `\n`, deliver complete
lines.

**Used by**: `lib/services/mcp-transport.esk` — MCP JSON-RPC is one
JSON object per line. Claude Code uses `readline.createInterface` for
this.

---

### 15.3 `(line-reader-close reader) -> void`

Stop the line reader and free resources.

---

---

## 16. Diff Algorithm

### 16.1 `(diff-lines old-lines new-lines) -> list-of-hunks`

Compute the minimal edit script between two lists of strings using
Myers' diff algorithm. Returns a list of hunks:
```scheme
((old-start . 3)
 (old-count . 2)
 (new-start . 3)
 (new-count . 4)
 (lines . (("=" . "unchanged")
           ("-" . "removed line")
           ("+" . "added line 1")
           ("+" . "added line 2")
           ("=" . "unchanged"))))
```

Line types: `"="` (context), `"-"` (deletion), `"+"` (insertion).

**Implementation**: Myers' O(ND) algorithm or patience diff.

**Used by**: `lib/tui/diff-view.esk` (file edit preview),
`lib/tui/screen-buffer.esk` (optimized redraw), `lib/agent/compaction.esk`
(showing what changed).

---

---

## 17. Fuzzy Search

### 17.1 `(fuzzy-match pattern candidates key-fn max-results) -> list`

Fuzzy string matching with scoring and ranking. Returns up to
`max-results` candidates, sorted by match quality (best first).

`key-fn` is `(lambda (candidate) -> string)` — extracts the searchable
text from each candidate.

Each result is `(score . candidate)` where `score` is 0.0-1.0.

Match algorithm should handle:
- Character-by-character fuzzy matching
- Bonus for consecutive matches
- Bonus for matches at word boundaries (after `-`, `_`, `/`, `.`)
- Bonus for camelCase boundaries
- Penalty for distance between matches

**Implementation**: Fuse.js-compatible scoring or `fzf`-style algorithm.

**Used by**: `lib/tools/tool-search.esk` (find tools by approximate
name), `lib/tools/file-edit.esk` (similar file suggestions),
slash command matching, memory search.

---

---

## 18. Semantic Versioning

### 18.1 `(semver-parse str) -> alist-or-false`

Parse a semantic version string:
```scheme
(semver-parse "1.2.3-beta.1+build.42")
;; => ((major . 1) (minor . 2) (patch . 3)
;;     (prerelease . "beta.1") (build . "build.42"))
```

Returns `#f` if invalid.

---

### 18.2 `(semver-compare a b) -> integer`

Compare two version strings. Returns `-1`, `0`, or `1`.
Follows semver precedence rules (prerelease < release).

---

### 18.3 `(semver-satisfies? version range) -> bool`

Check if `version` satisfies a range expression:
- `">=1.0.0"` — greater than or equal
- `"<2.0.0"` — less than
- `">=1.0.0 <2.0.0"` — AND
- `"^1.2.3"` — compatible with (>=1.2.3 <2.0.0)
- `"~1.2.3"` — approximately (>=1.2.3 <1.3.0)

**Used by**: Plugin version requirements, model version checks, config
migration.

---

---

## 19. LRU Cache

### 19.1 `(make-lru-cache max-size) -> cache`

Create a least-recently-used cache with bounded size. When inserting
into a full cache, the least recently accessed entry is evicted.

**Implementation**: Hash table + doubly-linked list for O(1) get/set.

**Used by**: Token estimation cache, file content cache, API response
cache, compiled regex cache.

---

### 19.2 `(lru-get cache key) -> value-or-false`

Retrieve value. Marks entry as most recently used.

---

### 19.3 `(lru-set! cache key value) -> void`

Insert or update. Evicts LRU entry if at capacity.

---

### 19.4 `(lru-has? cache key) -> bool`

Check existence without updating recency.

---

### 19.5 `(lru-delete! cache key) -> void`

Remove an entry.

---

### 19.6 `(lru-clear! cache) -> void`

Remove all entries.

---

### 19.7 `(lru-size cache) -> integer`

Current number of entries.

---

---

## 20. Event Emitter

### 20.1 `(make-event-emitter) -> emitter`

Create a pub/sub event emitter for decoupled component communication.

---

### 20.2 `(emit! emitter event-name . args) -> void`

Fire an event. All registered handlers are called synchronously in
registration order with `args`.

---

### 20.3 `(on! emitter event-name handler) -> void`

Subscribe to an event. `handler` is `(lambda args ...)`.

**Used by**: TUI subscribing to agent events (tool-start, tool-complete,
text-delta, error) without the agent knowing about the TUI.

---

### 20.4 `(once! emitter event-name handler) -> void`

Subscribe for a single event only. Auto-unsubscribes after firing.

---

### 20.5 `(off! emitter event-name handler) -> void`

Unsubscribe a specific handler.

---

### 20.6 `(off-all! emitter event-name) -> void`

Remove all handlers for an event.

---

---

## 21. Layout Engine

### 21.1 `(make-layout-node) -> node`

Create a flexbox layout node. Claude Code's entire TUI is built on
Yoga (Facebook's C flexbox library) via React/Ink. Properties:

- `width`, `height` (absolute or percent)
- `min-width`, `max-width`, `min-height`, `max-height`
- `flex-direction`: `"row"` or `"column"`
- `justify-content`: `"flex-start"`, `"center"`, `"flex-end"`,
  `"space-between"`, `"space-around"`
- `align-items`: `"flex-start"`, `"center"`, `"flex-end"`, `"stretch"`
- `flex-grow`, `flex-shrink`, `flex-basis`
- `padding-top`, `padding-right`, `padding-bottom`, `padding-left`
- `margin-top`, `margin-right`, `margin-bottom`, `margin-left`
- `overflow`: `"visible"`, `"hidden"`, `"scroll"`
- `position`: `"relative"`, `"absolute"`
- `border-width` (top/right/bottom/left)
- `gap` (between children)

**Implementation**: Link Yoga C library via FFI, or implement a subset
(row/column layout with flex-grow).

---

### 21.2 `(layout-set! node property value) -> void`

Set a layout property on a node. `property` is a string matching the
names above.

---

### 21.3 `(layout-add-child! parent child) -> void`

Add a child node to a parent. Order matters (first child = first in
flex layout).

---

### 21.4 `(layout-remove-child! parent child) -> void`

Remove a child node.

---

### 21.5 `(layout-calculate! root available-width available-height) -> void`

Calculate layout for the entire tree starting from `root`. After this
call, every node has computed `left`, `top`, `width`, `height` values.

---

### 21.6 `(layout-get-computed node property) -> number`

Get a computed value after `layout-calculate!`:
- `"left"` — x position relative to parent
- `"top"` — y position relative to parent
- `"width"` — computed width
- `"height"` — computed height

---

---

## 22. Shell Utilities

### 22.1 `(shell-quote str) -> string`

Escape a string for safe use in shell commands. Uses single-quote
wrapping with proper escaping of embedded single quotes:
`hello'world` -> `'hello'"'"'world'`

**Used by**: Every file that constructs shell commands (currently done
inline in 20+ places with `string-replace`).

---

### 22.2 `(shell-split str) -> list-of-strings`

Split a shell command into arguments, respecting quotes and escapes:
```scheme
(shell-split "echo 'hello world' \"foo bar\"")
;; => ("echo" "hello world" "foo bar")
```

**Used by**: Needed for bash tool command analysis, MCP command parsing.

---

---

## Summary Table

| # | Category | Functions | Count |
|---|----------|-----------|-------|
| 1 | Filesystem | 1.1-1.24 | 24 |
| 2 | Path Manipulation | 2.1-2.8 | 8 |
| 3 | Process Management | 3.1-3.9 | 9 |
| 4 | HTTP and Networking | 4.1-4.17 | 17 |
| 5 | Terminal | 5.1-5.14 | 14 |
| 6 | Cryptography | 6.1-6.9 | 9 |
| 7 | System Information | 7.1-7.16 | 16 |
| 8 | String Operations | 8.1-8.26 | 26 |
| 9 | Regex Extensions | 9.1-9.5 | 5 |
| 10 | Concurrency | 10.1-10.22 | 22 |
| 11 | Date and Time | 11.1-11.4 | 4 |
| 12 | JSON Extensions | 12.1-12.4 | 4 |
| 13 | SQLite Extensions | 13.1-13.4 | 4 |
| 14 | Compression | 14.1-14.4 | 4 |
| 15 | Streams and Pipes | 15.1-15.3 | 3 |
| 16 | Diff Algorithm | 16.1 | 1 |
| 17 | Fuzzy Search | 17.1 | 1 |
| 18 | Semantic Versioning | 18.1-18.3 | 3 |
| 19 | LRU Cache | 19.1-19.7 | 7 |
| 20 | Event Emitter | 20.1-20.6 | 6 |
| 21 | Layout Engine | 21.1-21.6 | 6 |
| 22 | Shell Utilities | 22.1-22.2 | 2 |
| | **TOTAL** | | **188** |

## Implementation Priority

**P0 — Eliminates all shell-outs from platform layer (do first):**
1.1 `mkdir-recursive`, 1.2 `file-rename`, 1.3 `file-size`, 1.6
`directory-list`, 1.9 `directory-delete-recursive`, 7.1 `os-type`,
7.10 `executable-exists?`, 2.1 `path-join`, 2.2 `path-dirname`,
2.3 `path-basename`, 6.3 `base64-encode`, 6.4 `base64-decode`,
6.8 `uuid-v4`, 22.1 `shell-quote`

**P1 — Required for core agent functionality:**
4.1 full HTTP methods, 4.2 GET with headers, 1.4 `file-stat`,
1.8 `directory-walk`, 3.1 `process-pid`, 3.6 `process-kill-tree`,
7.12 `current-time-ms`, 10.1-10.6 channels, 10.7-10.10 mutex,
9.1 `regex-match-groups`, 12.1 `json-get-in`

**P2 — Required for TUI parity:**
5.1-5.14 all terminal extensions, 8.1 `string-display-width`,
16.1 `diff-lines`, 21.1-21.6 layout engine

**P3 — Required for full feature parity:**
Everything else: WebSocket, HTTP server, file watching, compression,
fuzzy search, semver, timers, PTY, mTLS, etc.
