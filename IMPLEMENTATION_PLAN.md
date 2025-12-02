# Eshkol Feature Implementation Plan

## Overview

This document outlines the implementation plan to extend Eshkol with features needed for AI, networking, text processing, HTTP, TCP/UDP, and general-purpose applications.

**Current State**: ~266 functions/operators, strong in functional programming, autodiff, tensors, lists
**Target**: Full-stack language for AI applications with networking, data formats, concurrency

---

## Phase 1: Enhanced String Processing (Foundation)

**Priority**: Critical - unlocks text processing, data parsing, and API work
**Estimated Functions**: 12 new functions

### 1.1 String Manipulation Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `string-split` | `(string-split str delimiter) -> list` | Split string by delimiter |
| `string-join` | `(string-join list delimiter) -> string` | Join list with delimiter |
| `string-trim` | `(string-trim str) -> string` | Remove leading/trailing whitespace |
| `string-trim-left` | `(string-trim-left str) -> string` | Remove leading whitespace |
| `string-trim-right` | `(string-trim-right str) -> string` | Remove trailing whitespace |
| `string-replace` | `(string-replace str old new) -> string` | Replace all occurrences |
| `string-contains?` | `(string-contains? str substr) -> bool` | Check if contains substring |
| `string-index` | `(string-index str substr) -> int/-1` | Find index of substring |
| `string-upcase` | `(string-upcase str) -> string` | Convert to uppercase |
| `string-downcase` | `(string-downcase str) -> string` | Convert to lowercase |
| `string-reverse` | `(string-reverse str) -> string` | Reverse string |
| `string-copy` | `(string-copy str) -> string` | Create copy of string |

### Implementation Approach

**Location**: `lib/backend/llvm_codegen.cpp`

1. Add dispatch entries in `codegenFunctionCall()` (~line 7119)
2. Implement codegen methods following existing pattern (`codegenStringAppend` as template)
3. Use C stdlib functions where possible: `strstr`, `toupper`, `tolower`
4. Use arena allocation for new strings

**Example Implementation Pattern**:
```cpp
Value* codegenStringSplit(const eshkol_operations_t* op) {
    // 1. Validate argument count
    // 2. Extract string pointer with extractStringPtr()
    // 3. Call C functions or implement inline
    // 4. Build result list using codegenTaggedArenaConsCell()
    // 5. Return packed tagged value
}
```

---

## Phase 2: Data Formats (JSON & CSV)

**Priority**: High - enables API integration and data processing
**Estimated Functions**: 10 new functions

### 2.1 JSON Support

| Function | Signature | Description |
|----------|-----------|-------------|
| `json-parse` | `(json-parse str) -> value` | Parse JSON to Eshkol value |
| `json-stringify` | `(json-stringify value) -> string` | Convert value to JSON |
| `json-get` | `(json-get obj key) -> value` | Get field from JSON object |
| `json-array-ref` | `(json-array-ref arr idx) -> value` | Get element from JSON array |

### 2.2 CSV Support

| Function | Signature | Description |
|----------|-----------|-------------|
| `csv-parse` | `(csv-parse str) -> list-of-lists` | Parse CSV to nested lists |
| `csv-stringify` | `(csv-stringify data) -> string` | Convert to CSV string |
| `csv-read-file` | `(csv-read-file path) -> list-of-lists` | Read CSV file |
| `csv-write-file` | `(csv-write-file path data) -> bool` | Write CSV file |

### 2.3 Encoding Support

| Function | Signature | Description |
|----------|-----------|-------------|
| `base64-encode` | `(base64-encode str) -> string` | Base64 encode |
| `base64-decode` | `(base64-decode str) -> string` | Base64 decode |

### Implementation Approach

**JSON**: Implement recursive descent parser in LLVM IR
- Objects map to association lists: `((key1 . val1) (key2 . val2))`
- Arrays map to Eshkol lists
- Numbers, strings, booleans, null map directly

**CSV**: Simpler parsing with string-split
- Rows as lists, file as list of rows
- Handle quoted fields, escape characters

---

## Phase 3: Hash Tables

**Priority**: High - O(1) lookup needed for efficient data structures
**Estimated Functions**: 10 new functions

### 3.1 Hash Table Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `make-hash-table` | `(make-hash-table) -> hash-table` | Create empty hash table |
| `hash-table?` | `(hash-table? obj) -> bool` | Type predicate |
| `hash-ref` | `(hash-ref ht key default) -> value` | Get value or default |
| `hash-set!` | `(hash-set! ht key value) -> void` | Set key-value pair |
| `hash-remove!` | `(hash-remove! ht key) -> void` | Remove key |
| `hash-has-key?` | `(hash-has-key? ht key) -> bool` | Check key exists |
| `hash-keys` | `(hash-keys ht) -> list` | Get all keys |
| `hash-values` | `(hash-values ht) -> list` | Get all values |
| `hash->alist` | `(hash->alist ht) -> list` | Convert to alist |
| `alist->hash` | `(alist->hash alist) -> hash-table` | Convert from alist |

### Implementation Approach

1. Add new value type: `ESHKOL_VALUE_HASH_PTR = 13`
2. Create hash table structure in `arena_memory.h`:
```c
typedef struct eshkol_hash_table {
    size_t capacity;
    size_t size;
    eshkol_hash_entry_t* buckets;
} eshkol_hash_table_t;

typedef struct eshkol_hash_entry {
    eshkol_tagged_value_t key;
    eshkol_tagged_value_t value;
    struct eshkol_hash_entry* next;
} eshkol_hash_entry_t;
```
3. Implement arena allocation functions in `arena_memory.cpp`
4. Add codegen functions in `llvm_codegen.cpp`

---

## Phase 4: File System Operations

**Priority**: Medium-High - needed for practical applications
**Estimated Functions**: 15 new functions

### 4.1 File Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `file-exists?` | `(file-exists? path) -> bool` | Check if file exists |
| `file-delete` | `(file-delete path) -> bool` | Delete file |
| `file-rename` | `(file-rename old new) -> bool` | Rename file |
| `file-size` | `(file-size path) -> int` | Get file size |
| `read-file` | `(read-file path) -> string` | Read entire file |
| `write-file` | `(write-file path content) -> bool` | Write entire file |
| `append-file` | `(append-file path content) -> bool` | Append to file |

### 4.2 Directory Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `directory-exists?` | `(directory-exists? path) -> bool` | Check if directory exists |
| `make-directory` | `(make-directory path) -> bool` | Create directory |
| `directory-list` | `(directory-list path) -> list` | List directory contents |
| `current-directory` | `(current-directory) -> string` | Get current directory |
| `set-current-directory!` | `(set-current-directory! path) -> bool` | Change directory |

### 4.3 Path Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `path-join` | `(path-join parts...) -> string` | Join path components |
| `path-dirname` | `(path-dirname path) -> string` | Get directory part |
| `path-basename` | `(path-basename path) -> string` | Get filename part |

### Implementation Approach

Use POSIX/C stdlib functions via extern:
- `stat()`, `remove()`, `rename()` for file operations
- `opendir()`, `readdir()`, `mkdir()` for directory operations
- `getcwd()`, `chdir()` for working directory

---

## Phase 5: Networking - TCP/UDP Sockets

**Priority**: High - core networking capability
**Estimated Functions**: 16 new functions

### 5.1 TCP Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `tcp-connect` | `(tcp-connect host port) -> socket` | Connect to server |
| `tcp-listen` | `(tcp-listen port backlog) -> socket` | Create listening socket |
| `tcp-accept` | `(tcp-accept socket) -> (client-socket . addr)` | Accept connection |
| `tcp-send` | `(tcp-send socket data) -> int` | Send data |
| `tcp-recv` | `(tcp-recv socket max-len) -> string` | Receive data |
| `tcp-close` | `(tcp-close socket) -> bool` | Close socket |

### 5.2 UDP Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `udp-socket` | `(udp-socket) -> socket` | Create UDP socket |
| `udp-bind` | `(udp-bind socket port) -> bool` | Bind to port |
| `udp-send-to` | `(udp-send-to socket data host port) -> int` | Send datagram |
| `udp-recv-from` | `(udp-recv-from socket max-len) -> (data . addr)` | Receive datagram |

### 5.3 Socket Options

| Function | Signature | Description |
|----------|-----------|-------------|
| `socket-set-option` | `(socket-set-option socket opt value) -> bool` | Set socket option |
| `socket-get-option` | `(socket-get-option socket opt) -> value` | Get socket option |
| `socket-set-timeout` | `(socket-set-timeout socket secs) -> bool` | Set timeout |
| `socket-set-nonblocking` | `(socket-set-nonblocking socket bool) -> bool` | Set non-blocking |

### 5.4 DNS/Address Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `resolve-hostname` | `(resolve-hostname name) -> ip-string` | DNS lookup |
| `get-local-address` | `(get-local-address socket) -> addr` | Get local address |

### Implementation Approach

1. Add new value type: `ESHKOL_VALUE_SOCKET_PTR = 14`
2. Use POSIX socket API via extern declarations:
   - `socket()`, `connect()`, `bind()`, `listen()`, `accept()`
   - `send()`, `recv()`, `sendto()`, `recvfrom()`
   - `close()`, `setsockopt()`, `getsockopt()`
   - `getaddrinfo()`, `gethostbyname()`
3. Create socket wrapper structure in arena memory
4. Implement codegen functions

---

## Phase 6: HTTP Client/Server

**Priority**: High - essential for web/API applications
**Estimated Functions**: 12 new functions

### 6.1 HTTP Client

| Function | Signature | Description |
|----------|-----------|-------------|
| `http-get` | `(http-get url) -> response` | Simple GET request |
| `http-post` | `(http-post url body) -> response` | Simple POST request |
| `http-request` | `(http-request method url headers body) -> response` | Full HTTP request |
| `http-response-status` | `(http-response-status resp) -> int` | Get status code |
| `http-response-body` | `(http-response-body resp) -> string` | Get response body |
| `http-response-headers` | `(http-response-headers resp) -> alist` | Get headers |

### 6.2 HTTP Server

| Function | Signature | Description |
|----------|-----------|-------------|
| `http-server` | `(http-server port handler) -> server` | Create HTTP server |
| `http-server-stop` | `(http-server-stop server) -> bool` | Stop server |
| `http-request-method` | `(http-request-method req) -> string` | Get request method |
| `http-request-path` | `(http-request-path req) -> string` | Get request path |
| `http-request-headers` | `(http-request-headers req) -> alist` | Get request headers |
| `http-request-body` | `(http-request-body req) -> string` | Get request body |

### 6.3 URL Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `url-parse` | `(url-parse url) -> alist` | Parse URL components |
| `url-encode` | `(url-encode str) -> string` | Percent-encode string |
| `url-decode` | `(url-decode str) -> string` | Percent-decode string |

### Implementation Approach

Build HTTP on top of TCP sockets:
1. Parse HTTP/1.1 protocol manually (simpler than external lib)
2. Implement request/response structures in arena memory
3. Use `tcp-connect` for client, `tcp-listen`/`tcp-accept` for server
4. Parse headers as association lists

---

## Phase 7: System & Environment

**Priority**: Medium - needed for complete applications
**Estimated Functions**: 10 new functions

### 7.1 Environment

| Function | Signature | Description |
|----------|-----------|-------------|
| `getenv` | `(getenv name) -> string/false` | Get environment variable |
| `setenv` | `(setenv name value) -> bool` | Set environment variable |
| `unsetenv` | `(unsetenv name) -> bool` | Remove environment variable |
| `environ` | `(environ) -> alist` | Get all environment variables |

### 7.2 Command Line

| Function | Signature | Description |
|----------|-----------|-------------|
| `command-line` | `(command-line) -> list` | Get command line arguments |
| `program-name` | `(program-name) -> string` | Get program name |

### 7.3 Process

| Function | Signature | Description |
|----------|-----------|-------------|
| `exit` | `(exit code) -> never` | Exit with code |
| `system` | `(system command) -> int` | Run shell command |
| `sleep` | `(sleep seconds) -> void` | Delay execution |
| `current-time` | `(current-time) -> int` | Get Unix timestamp |

### Implementation Approach

Use C stdlib functions:
- `getenv()`, `setenv()`, `unsetenv()` for environment
- Store argc/argv in globals at program start
- `system()`, `exit()`, `sleep()`, `time()` for process control

---

## Phase 8: Concurrency (Future)

**Priority**: Medium - needed for async I/O and parallel processing
**Estimated Functions**: 15 new functions

### 8.1 Threads

| Function | Signature | Description |
|----------|-----------|-------------|
| `thread-spawn` | `(thread-spawn thunk) -> thread` | Create new thread |
| `thread-join` | `(thread-join thread) -> value` | Wait for thread |
| `thread-current` | `(thread-current) -> thread` | Get current thread |
| `thread-yield` | `(thread-yield) -> void` | Yield execution |

### 8.2 Channels (Message Passing)

| Function | Signature | Description |
|----------|-----------|-------------|
| `make-channel` | `(make-channel) -> channel` | Create channel |
| `channel-send` | `(channel-send ch value) -> void` | Send value |
| `channel-recv` | `(channel-recv ch) -> value` | Receive value (blocking) |
| `channel-try-recv` | `(channel-try-recv ch) -> value/false` | Non-blocking receive |

### 8.3 Synchronization

| Function | Signature | Description |
|----------|-----------|-------------|
| `make-mutex` | `(make-mutex) -> mutex` | Create mutex |
| `mutex-lock` | `(mutex-lock mutex) -> void` | Acquire lock |
| `mutex-unlock` | `(mutex-unlock mutex) -> void` | Release lock |
| `mutex-try-lock` | `(mutex-try-lock mutex) -> bool` | Try to acquire |

### 8.4 Atomics

| Function | Signature | Description |
|----------|-----------|-------------|
| `atomic-ref` | `(atomic-ref atom) -> value` | Read atomic |
| `atomic-set!` | `(atomic-set! atom value) -> void` | Write atomic |
| `atomic-cas!` | `(atomic-cas! atom old new) -> bool` | Compare-and-swap |

### Implementation Approach

Use pthreads via extern:
- `pthread_create()`, `pthread_join()`, `pthread_self()`
- `pthread_mutex_init()`, `pthread_mutex_lock()`, `pthread_mutex_unlock()`
- Channels implemented as mutex-protected queues
- LLVM atomic instructions for atomic operations

**Note**: Concurrency requires careful consideration of:
- Arena memory thread safety (per-thread arenas or locking)
- GC implications (if ever added)
- Closure safety across threads

---

## Phase 9: Error Handling

**Priority**: Medium - needed for robust applications
**Estimated Functions**: 8 new functions

### 9.1 Exception Handling

| Function | Signature | Description |
|----------|-----------|-------------|
| `try` | `(try expr handler) -> value` | Try with handler |
| `raise` | `(raise condition) -> never` | Raise exception |
| `error` | `(error message args...) -> never` | Raise error |
| `with-exception-handler` | `(with-exception-handler handler thunk) -> value` | R7RS style |

### 9.2 Conditions

| Function | Signature | Description |
|----------|-----------|-------------|
| `make-condition` | `(make-condition type message) -> condition` | Create condition |
| `condition?` | `(condition? obj) -> bool` | Type predicate |
| `condition-type` | `(condition-type cond) -> symbol` | Get condition type |
| `condition-message` | `(condition-message cond) -> string` | Get message |

### Implementation Approach

Use setjmp/longjmp for non-local exits:
1. Add condition type to value system
2. Create exception handler stack (per-thread for concurrency)
3. Implement `try` using setjmp, `raise` using longjmp
4. Unwind arena scopes on exception

---

## Implementation Order & Dependencies

```
Phase 1: String Processing
    ↓
Phase 2: Data Formats (JSON/CSV) ← depends on string-split
    ↓
Phase 3: Hash Tables ← useful for JSON objects
    ↓
Phase 4: File System ← needed for practical apps
    ↓
Phase 5: TCP/UDP Sockets ← core networking
    ↓
Phase 6: HTTP Client/Server ← depends on sockets
    ↓
Phase 7: System/Environment ← useful utilities
    ↓
Phase 8: Concurrency ← for async/parallel
    ↓
Phase 9: Error Handling ← for robustness
```

---

## File Modifications Required

### Core Files

| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Add new value types (HASH_PTR, SOCKET_PTR, etc.) |
| `lib/core/arena_memory.h` | Add structures for hash tables, sockets |
| `lib/core/arena_memory.cpp` | Add allocation functions for new types |
| `lib/backend/llvm_codegen.cpp` | Add codegen functions (~100 new functions) |

### New Library Files (Pure Eshkol)

| File | Contents |
|------|----------|
| `lib/net.esk` | High-level networking utilities |
| `lib/http.esk` | HTTP request/response helpers |
| `lib/json.esk` | JSON utilities (if not in C) |
| `lib/io.esk` | File I/O utilities |

### Test Files

| Directory | Tests |
|-----------|-------|
| `tests/strings/` | String processing tests |
| `tests/json/` | JSON parsing tests |
| `tests/hash/` | Hash table tests |
| `tests/files/` | File system tests |
| `tests/net/` | Networking tests |
| `tests/http/` | HTTP client/server tests |

---

## Estimated Scope

| Phase | New Functions | Complexity | Est. LOC |
|-------|---------------|------------|----------|
| 1. Strings | 12 | Low | ~500 |
| 2. JSON/CSV | 10 | Medium | ~800 |
| 3. Hash Tables | 10 | Medium | ~600 |
| 4. File System | 15 | Low | ~400 |
| 5. Sockets | 16 | High | ~1000 |
| 6. HTTP | 12 | High | ~1200 |
| 7. System/Env | 10 | Low | ~300 |
| 8. Concurrency | 15 | Very High | ~1500 |
| 9. Error Handling | 8 | High | ~800 |
| **TOTAL** | **~108** | | **~7100** |

---

## Success Criteria

After implementation, Eshkol should be able to:

1. **Parse and generate JSON** for API integration
2. **Make HTTP requests** to external APIs
3. **Serve HTTP responses** as a web server
4. **Read/write files** and manipulate directories
5. **Process text** with split, join, trim, replace
6. **Use hash tables** for O(1) key-value storage
7. **Handle errors** gracefully with exceptions
8. **Run concurrent tasks** with threads and channels (Phase 8)

---

## Next Steps

1. **Approve this plan** or request modifications
2. **Start with Phase 1** (String Processing) - lowest risk, highest impact
3. **Create test cases** before implementation
4. **Implement incrementally** - one function at a time
5. **Test thoroughly** after each function
