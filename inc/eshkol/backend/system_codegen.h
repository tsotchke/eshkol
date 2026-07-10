/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * SystemCodegen - System, environment, and file system code generation
 *
 * This module handles:
 * - Environment variables (getenv, setenv, unsetenv)
 * - System operations (system, sleep, current-seconds)
 * - File system operations (file-exists?, read-file, write-file, etc.)
 * - Directory operations (directory-list, make-directory, etc.)
 */
#ifndef ESHKOL_BACKEND_SYSTEM_CODEGEN_H
#define ESHKOL_BACKEND_SYSTEM_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <unordered_map>

namespace eshkol {

/**
 * SystemCodegen handles system, environment, and file operations.
 */
class SystemCodegen {
public:
    /**
     * Construct SystemCodegen with context and helpers.
     */
    SystemCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem,
                  std::unordered_map<std::string, llvm::Function*>& function_table);

    // === Environment Operations ===

    /**
     * Get environment variable: (getenv "NAME")
     * @return String value or #f if not set
     */
    llvm::Value* getenv(const eshkol_operations_t* op);

    /**
     * Set environment variable: (setenv "NAME" "VALUE")
     * @return #t on success, #f on failure
     */
    llvm::Value* setenv(const eshkol_operations_t* op);

    /**
     * Unset environment variable: (unsetenv "NAME")
     * @return #t on success, #f on failure
     */
    llvm::Value* unsetenv(const eshkol_operations_t* op);

    // === System Operations ===

    /**
     * Execute shell command: (system "command")
     * @return Exit code as integer
     */
    llvm::Value* systemCall(const eshkol_operations_t* op);

    /**
     * Sleep for seconds: (sleep n)
     * @return null
     */
    llvm::Value* sleep(const eshkol_operations_t* op);

    /**
     * Get current Unix timestamp: (current-seconds)
     * @return Timestamp as integer
     */
    llvm::Value* currentSeconds(const eshkol_operations_t* op);

    /**
     * Get current time in seconds with microsecond precision: (current-time)
     * @return Time in seconds as double (e.g., 1709049600.123456)
     */
    llvm::Value* currentTime(const eshkol_operations_t* op);

    /**
     * Get current time in milliseconds: (current-time-ms)
     * @return Time in milliseconds as double
     */
    llvm::Value* currentTimeMs(const eshkol_operations_t* op);

    /**
     * Get current time in nanoseconds: (current-time-ns)
     * Uses clock_gettime with CLOCK_UPTIME_RAW (macOS) or CLOCK_MONOTONIC (Linux)
     * @return Time in nanoseconds as double
     */
    llvm::Value* currentTimeNs(const eshkol_operations_t* op);

    /**
     * Exit the program: (exit code)
     * @param code Exit code (integer)
     * @return Does not return
     */
    llvm::Value* exitProgram(const eshkol_operations_t* op);

    /**
     * Get command-line arguments: (command-line)
     * @return List of strings (argv)
     */
    llvm::Value* commandLine(const eshkol_operations_t* op);

    // === File Operations ===

    /**
     * Check if file exists: (file-exists? "path")
     * @return #t or #f
     */
    llvm::Value* fileExists(const eshkol_operations_t* op);

    /**
     * Check if file is readable: (file-readable? "path")
     * @return #t or #f
     */
    llvm::Value* fileReadable(const eshkol_operations_t* op);

    /**
     * Check if file is writable: (file-writable? "path")
     * @return #t or #f
     */
    llvm::Value* fileWritable(const eshkol_operations_t* op);

    /**
     * Delete a file: (file-delete "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* fileDelete(const eshkol_operations_t* op);

    /**
     * Rename/move a file: (file-rename "old" "new")
     * @return #t on success, #f on failure
     */
    llvm::Value* fileRename(const eshkol_operations_t* op);

    /**
     * Get file size: (file-size "path")
     * @return Size in bytes, or #f on error
     */
    llvm::Value* fileSize(const eshkol_operations_t* op);

    /**
     * Read entire file: (read-file "path")
     * @return File contents as string, or #f on error
     */
    llvm::Value* readFile(const eshkol_operations_t* op);

    /**
     * Write string to file: (write-file "path" "contents")
     * @return #t on success, #f on failure
     */
    llvm::Value* writeFile(const eshkol_operations_t* op);

    /**
     * Append string to file: (append-file "path" "contents")
     * @return #t on success, #f on failure
     */
    llvm::Value* appendFile(const eshkol_operations_t* op);

    // === Directory Operations ===

    /**
     * Check if directory exists: (directory-exists? "path")
     * @return #t or #f
     */
    llvm::Value* directoryExists(const eshkol_operations_t* op);

    /**
     * Create directory: (make-directory "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* makeDirectory(const eshkol_operations_t* op);

    /**
     * Delete directory: (delete-directory "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* deleteDirectory(const eshkol_operations_t* op);

    /**
     * List directory contents: (directory-list "path")
     * @return List of filenames
     */
    llvm::Value* directoryList(const eshkol_operations_t* op);

    /**
     * Get current working directory: (current-directory)
     * @return Path as string
     */
    llvm::Value* currentDirectory(const eshkol_operations_t* op);

    /**
     * Set current working directory: (set-current-directory! "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* setCurrentDirectory(const eshkol_operations_t* op);

    /* ── v1.2 system builtins (delegate to C runtime) ── */

    /** @brief (os-type) — operating system name string (e.g. "darwin", "linux"). */
    llvm::Value* osType(const eshkol_operations_t* op);
    /** @brief (os-arch) — CPU architecture string (e.g. "arm64", "x86_64"). */
    llvm::Value* osArch(const eshkol_operations_t* op);
    /** @brief (hostname) — the machine's hostname string. */
    llvm::Value* hostnameBuiltin(const eshkol_operations_t* op);
    /** @brief (username) — the current user's login name string. */
    llvm::Value* usernameBuiltin(const eshkol_operations_t* op);
    /** @brief (cpu-count) — number of logical CPUs as an integer. */
    llvm::Value* cpuCount(const eshkol_operations_t* op);
    /** @brief (getpid) — the current process id as an integer. */
    llvm::Value* getpidBuiltin(const eshkol_operations_t* op);
    /** @brief (home-directory) — the current user's home directory path. */
    llvm::Value* homeDirectory(const eshkol_operations_t* op);
    /** @brief (sleep-ms n) — sleep for n milliseconds. @return Unspecified value. */
    llvm::Value* sleepMs(const eshkol_operations_t* op);
    /** @brief (executable-exists? name) — check whether an executable is resolvable on PATH. */
    llvm::Value* executableExists(const eshkol_operations_t* op);
    /** @brief (executable-path) — absolute path to the running executable. */
    llvm::Value* executablePath(const eshkol_operations_t* op);
    /** @brief (monotonic-time-ms) — monotonic clock reading in milliseconds (double). */
    llvm::Value* monotonicTimeMs(const eshkol_operations_t* op);
    /** @brief (__arena-used) — bytes currently used in the global arena (ESH-0187 debug hook). */
    llvm::Value* arenaUsed(const eshkol_operations_t* op);  // ESH-0187 debug hook
    /** @brief (ad-reset-counters!) — zero all AD instrumentation counters. */
    llvm::Value* adResetCounters(const eshkol_operations_t* op);
    /** @brief (ad-primal-calls) — primal (user-function) evaluations since reset. */
    llvm::Value* adPrimalCalls(const eshkol_operations_t* op);
    /** @brief (ad-reverse-passes) — reverse (backward) sweeps since reset. */
    llvm::Value* adReversePasses(const eshkol_operations_t* op);
    /** @brief (ad-tape-allocations) — reverse-mode tapes allocated since reset. */
    llvm::Value* adTapeAllocations(const eshkol_operations_t* op);
    /** @brief (ad-finite-difference-evals) — finite-difference evaluations since reset. */
    llvm::Value* adFiniteDifferenceEvals(const eshkol_operations_t* op);
    /** @brief (ad-counters) — assoc list of all AD instrumentation counters. */
    llvm::Value* adCounters(const eshkol_operations_t* op);
    /** @brief (temp-directory) — the system temporary-files directory path. */
    llvm::Value* tempDirectory(const eshkol_operations_t* op);
    /** @brief (prevent-sleep) — inhibit system sleep/idle while running. @return Unspecified value. */
    llvm::Value* preventSleep(const eshkol_operations_t* op);
    /** @brief (allow-sleep) — undo a prior (prevent-sleep). @return Unspecified value. */
    llvm::Value* allowSleep(const eshkol_operations_t* op);
    /** @brief (path-join a b ...) — join path components with the OS separator. */
    llvm::Value* pathJoin(const eshkol_operations_t* op);
    /** @brief (path-dirname path) — the directory portion of a path. */
    llvm::Value* pathDirname(const eshkol_operations_t* op);
    /** @brief (path-basename path) — the final path component. */
    llvm::Value* pathBasename(const eshkol_operations_t* op);
    /** @brief (path-extname path) — the file extension (including the leading dot). */
    llvm::Value* pathExtname(const eshkol_operations_t* op);
    /** @brief (path-is-absolute? path) — whether path is an absolute path. */
    llvm::Value* pathIsAbsolute(const eshkol_operations_t* op);
    /** @brief (path-normalize path) — collapse "." / ".." segments and redundant separators. */
    llvm::Value* pathNormalize(const eshkol_operations_t* op);
    /** @brief (realpath path) — canonicalized absolute path, or #f if it doesn't resolve. */
    llvm::Value* realpathBuiltin(const eshkol_operations_t* op);
    /** @brief (file-stat path) — file metadata (size, mode, mtime, ...), or #f on error. */
    llvm::Value* fileStat(const eshkol_operations_t* op);
    /** @brief (file-copy src dst) — copy a file. @return #t on success, #f on failure. */
    llvm::Value* fileCopy(const eshkol_operations_t* op);
    /** @brief (mkdir-recursive path) — create a directory and any missing parents. @return #t/#f. */
    llvm::Value* mkdirRecursive(const eshkol_operations_t* op);
    /** @brief (mkdtemp template) — create a unique temporary directory from a template. */
    llvm::Value* mkdtempBuiltin(const eshkol_operations_t* op);
    /** @brief (make-temp-file ...) — create and open a fresh temporary file, returning its path/port. */
    llvm::Value* makeTempFile(const eshkol_operations_t* op);
    /** @brief (make-temp-dir ...) — create a fresh temporary directory, returning its path. */
    llvm::Value* makeTempDir(const eshkol_operations_t* op);
    /** @brief (directory-delete-recursive path) — remove a directory and its contents. @return #t/#f. */
    llvm::Value* directoryDeleteRecursive(const eshkol_operations_t* op);
    /** @brief (shell-quote str) — quote a string for safe inclusion in a shell command line. */
    llvm::Value* shellQuote(const eshkol_operations_t* op);
    /** @brief (fork) — POSIX fork(); returns the child pid in the parent, 0 in the child. */
    llvm::Value* forkBuiltin(const eshkol_operations_t* op);
    /** @brief (execv path args) — replace the process image; does not return on success. */
    llvm::Value* execvBuiltin(const eshkol_operations_t* op);
    /** @brief (process-spawn ...) — spawn a child process. @return A process handle. */
    llvm::Value* processSpawn(const eshkol_operations_t* op);
    /** @brief (process-wait handle) — block until a spawned process exits. @return Its exit status. */
    llvm::Value* processWait(const eshkol_operations_t* op);
    /** @brief (poll-fd fd ...) — poll a file descriptor for readiness. */
    llvm::Value* pollFd(const eshkol_operations_t* op);
    /** @brief (tensor-save tensor path) — serialize a tensor to disk. @return #t/#f. */
    llvm::Value* tensorSave(const eshkol_operations_t* op);
    /** @brief (tensor-load path) — deserialize a tensor from disk. @return The tensor, or #f on failure. */
    llvm::Value* tensorLoad(const eshkol_operations_t* op);

    /* v1.2 batch 2: VM-parity + new builtins */

    /** @brief (file-chmod path mode) — change file permissions. @return #t/#f. */
    llvm::Value* fileChmod(const eshkol_operations_t* op);
    /** @brief (symlink-create target linkpath) — create a symbolic link. @return #t/#f. */
    llvm::Value* symlinkCreate(const eshkol_operations_t* op);
    /** @brief (symlink-read path) — read the target of a symbolic link. */
    llvm::Value* symlinkRead(const eshkol_operations_t* op);
    /** @brief (directory-walk path) — recursively list directory entries. */
    llvm::Value* directoryWalk(const eshkol_operations_t* op);
    /** @brief (mkstemp template) — create and open a unique temporary file, returning a port/fd. */
    llvm::Value* mkstempBuiltin(const eshkol_operations_t* op);
    /** @brief (process-kill pid [signal]) — send a signal to a process. @return #t/#f. */
    llvm::Value* processKill(const eshkol_operations_t* op);
    /** @brief (file-mtime path) — file modification time, in seconds since epoch. */
    llvm::Value* fileMtime(const eshkol_operations_t* op);
    /** @brief (file-atime path) — file last-access time, in seconds since epoch. */
    llvm::Value* fileAtime(const eshkol_operations_t* op);
    /** @brief (file-lock port) — acquire an advisory lock on an open file. @return #t/#f. */
    llvm::Value* fileLock(const eshkol_operations_t* op);
    /** @brief (file-unlock port) — release an advisory lock. @return #t/#f. */
    llvm::Value* fileUnlock(const eshkol_operations_t* op);
    /** @brief (path-relative from to) — compute a relative path from one path to another. */
    llvm::Value* pathRelative(const eshkol_operations_t* op);
    /** @brief (path-resolve ...) — resolve path components against the current directory. */
    llvm::Value* pathResolve(const eshkol_operations_t* op);
    /** @brief (glob-expand pattern) — expand a glob pattern. @return List of matching paths. */
    llvm::Value* globExpand(const eshkol_operations_t* op);
    /** @brief (glob-match pattern path) — test a single path against a glob pattern. */
    llvm::Value* globMatch(const eshkol_operations_t* op);

    /* v1.2 batch 3: advanced process management */

    /** @brief (process-setpgid pid pgid) — set a process group id. @return #t/#f. */
    llvm::Value* processSetpgid(const eshkol_operations_t* op);
    /** @brief (process-kill-tree pid [signal]) — signal a process and all its descendants. @return #t/#f. */
    llvm::Value* processKillTree(const eshkol_operations_t* op);
    /** @brief (process-spawn-pty ...) — spawn a child process attached to a pseudo-terminal. */
    llvm::Value* processSpawnPty(const eshkol_operations_t* op);
    /** @brief (process-read-nonblocking handle) — non-blocking read of a spawned process's output. */
    llvm::Value* processReadNonblocking(const eshkol_operations_t* op);

    /* v1.2 time API (ISO8601, #168) */

    /** @brief (format-iso8601 seconds) — format a Unix timestamp as an ISO 8601 string. */
    llvm::Value* formatIso8601(const eshkol_operations_t* op);
    /** @brief (parse-iso8601 str) — parse an ISO 8601 string. @return Seconds since epoch. */
    llvm::Value* parseIso8601(const eshkol_operations_t* op);
    /** @brief (current-timestamp) — the current time as an ISO 8601 string. */
    llvm::Value* currentTimestamp(const eshkol_operations_t* op);
    /** @brief (format-relative seconds) — human-readable relative time (e.g. "3 minutes ago"). */
    llvm::Value* formatRelative(const eshkol_operations_t* op);
    /** @brief (local-timezone-offset) — local timezone's offset from UTC, in seconds. */
    llvm::Value* localTimezoneOffset(const eshkol_operations_t* op);

    /* v1.2 batch 4 */

    /** @brief (process-pid handle) — the pid of a process spawned via process-spawn. */
    llvm::Value* processPid(const eshkol_operations_t* op);
    /** @brief (file-mmap path ...) — memory-map a file. @return A mapping handle/pointer. */
    llvm::Value* fileMmap(const eshkol_operations_t* op);
    /** @brief (file-munmap handle) — unmap a previously mmap'd file. @return Unspecified value. */
    llvm::Value* fileMunmap(const eshkol_operations_t* op);
    /** @brief (unix-socket-connect path) — connect to a Unix domain socket. @return A socket handle. */
    llvm::Value* unixSocketConnect(const eshkol_operations_t* op);
    /** @brief (socket-send sock data) — send bytes on a socket. @return Number of bytes sent. */
    llvm::Value* socketSend(const eshkol_operations_t* op);
    /** @brief (socket-recv sock n) — receive up to n bytes from a socket. */
    llvm::Value* socketRecv(const eshkol_operations_t* op);
    /** @brief (socket-close sock) — close a socket. @return Unspecified value. */
    llvm::Value* socketClose(const eshkol_operations_t* op);
    /** @brief (term-set-scroll-region top bottom) — set the terminal's scrolling region. */
    llvm::Value* termSetScrollRegion(const eshkol_operations_t* op);
    /** @brief (term-reset-scroll-region) — restore the terminal's default scrolling region. */
    llvm::Value* termResetScrollRegion(const eshkol_operations_t* op);
    /** @brief (term-enable-mouse) — enable terminal mouse-tracking reports. */
    llvm::Value* termEnableMouse(const eshkol_operations_t* op);
    /** @brief (term-disable-mouse) — disable terminal mouse-tracking reports. */
    llvm::Value* termDisableMouse(const eshkol_operations_t* op);
    /** @brief (term-read-mouse-event) — read and decode a pending mouse event. */
    llvm::Value* termReadMouseEvent(const eshkol_operations_t* op);
    /** @brief (term-enable-alternate-screen) — switch to the terminal's alternate screen buffer. */
    llvm::Value* termEnableAlternateScreen(const eshkol_operations_t* op);
    /** @brief (term-disable-alternate-screen) — restore the terminal's primary screen buffer. */
    llvm::Value* termDisableAlternateScreen(const eshkol_operations_t* op);
    /** @brief (term-clipboard-write str) — write to the system clipboard via OSC52. */
    llvm::Value* termClipboardWrite(const eshkol_operations_t* op);
    /** @brief (term-clipboard-read) — read the system clipboard via OSC52. */
    llvm::Value* termClipboardRead(const eshkol_operations_t* op);
    /** @brief (term-hyperlink url text) — emit an OSC8 clickable terminal hyperlink. */
    llvm::Value* termHyperlink(const eshkol_operations_t* op);
    /** @brief (term-detect-capabilities) — probe and report terminal feature support. */
    llvm::Value* termDetectCapabilities(const eshkol_operations_t* op);
    /** @brief (term-bell) — ring the terminal bell. @return Unspecified value. */
    llvm::Value* termBell(const eshkol_operations_t* op);
    /** @brief (fs-watch-native path) — start a native (non-recursive) filesystem watch. @return A watch handle. */
    llvm::Value* fsWatchNative(const eshkol_operations_t* op);
    /** @brief (fs-watch-recursive path) — start a recursive filesystem watch. @return A watch handle. */
    llvm::Value* fsWatchRecursive(const eshkol_operations_t* op);
    /** @brief (fs-watch-poll handle) — poll a filesystem watch for pending change events. */
    llvm::Value* fsWatchPoll(const eshkol_operations_t* op);
    /** @brief (fs-unwatch handle) — stop a filesystem watch. @return Unspecified value. */
    llvm::Value* fsUnwatch(const eshkol_operations_t* op);
    /** @brief (ansi-strip str) — remove ANSI escape sequences from a string. */
    llvm::Value* ansiStrip(const eshkol_operations_t* op);
    /** @brief (string-display-width str) — terminal display width, accounting for wide/zero-width chars. */
    llvm::Value* stringDisplayWidth(const eshkol_operations_t* op);
    /** @brief (string-truncate-display str width) — truncate a string to a terminal display width. */
    llvm::Value* stringTruncateDisplay(const eshkol_operations_t* op);
    /** @brief (url-encode str) — percent-encode a string for use in a URL. */
    llvm::Value* urlEncode(const eshkol_operations_t* op);
    /** @brief (url-decode str) — decode a percent-encoded URL string. */
    llvm::Value* urlDecode(const eshkol_operations_t* op);
    /** @brief (url-parse str) — parse a URL into its components (scheme, host, path, query, ...). */
    llvm::Value* urlParse(const eshkol_operations_t* op);
    /** @brief (base64url-encode bytes) — base64url-encode a byte string. */
    llvm::Value* base64urlEncode(const eshkol_operations_t* op);
    /** @brief (base64url-decode str) — decode a base64url-encoded string. */
    llvm::Value* base64urlDecode(const eshkol_operations_t* op);
    /** @brief (uuid-v4) — generate a random (version 4) UUID string. */
    llvm::Value* uuidV4(const eshkol_operations_t* op);
    /** @brief (constant-time-equal? a b) — timing-safe byte-string comparison. */
    llvm::Value* constantTimeEqual(const eshkol_operations_t* op);
    /** @brief (sha256-file path) — SHA-256 digest of a file's contents, as a hex string. */
    llvm::Value* sha256File(const eshkol_operations_t* op);
    /** @brief (regex-compile pattern) — compile a regular expression. @return A regex handle. */
    llvm::Value* regexCompile(const eshkol_operations_t* op);
    /** @brief (regex-free handle) — release a compiled regex. @return Unspecified value. */
    llvm::Value* regexFree(const eshkol_operations_t* op);
    /** @brief (regex-match handle str) — match a compiled regex against a string. */
    llvm::Value* regexMatch(const eshkol_operations_t* op);
    /** @brief (regex-match? handle str) — whether a compiled regex matches a string. */
    llvm::Value* regexMatchPredicate(const eshkol_operations_t* op);
    /** @brief (regex-match-groups handle str) — captured groups from a regex match. */
    llvm::Value* regexMatchGroups(const eshkol_operations_t* op);
    /** @brief (regex-split handle str) — split a string on regex matches. */
    llvm::Value* regexSplit(const eshkol_operations_t* op);
    /** @brief (diff-lines a b) — compute a line-based diff between two strings. */
    llvm::Value* diffLines(const eshkol_operations_t* op);
    /** @brief (fuzzy-match pattern str) — fuzzy substring match score/result. */
    llvm::Value* fuzzyMatch(const eshkol_operations_t* op);
    /** @brief (semver-parse str) — parse a semantic version string into its components. */
    llvm::Value* semverParse(const eshkol_operations_t* op);
    /** @brief (semver-compare a b) — compare two semantic versions. @return -1, 0, or 1. */
    llvm::Value* semverCompare(const eshkol_operations_t* op);
    /** @brief (semver-satisfies? version range) — whether a version satisfies a semver range. */
    llvm::Value* semverSatisfies(const eshkol_operations_t* op);
    /** @brief (make-pipe) — create a pipe. @return A pair of read/write file descriptors. */
    llvm::Value* makePipe(const eshkol_operations_t* op);
    /** @brief (fd-write fd data) — write raw bytes to a file descriptor. @return Bytes written. */
    llvm::Value* fdWrite(const eshkol_operations_t* op);
    /** @brief (make-line-reader fd) — create a buffered line reader over a file descriptor. */
    llvm::Value* makeLineReader(const eshkol_operations_t* op);
    /** @brief (line-reader-poll reader) — read the next available line, or #f if none is ready. */
    llvm::Value* lineReaderPoll(const eshkol_operations_t* op);
    /** @brief (line-reader-close reader) — release a line reader. @return Unspecified value. */
    llvm::Value* lineReaderClose(const eshkol_operations_t* op);
    /** @brief (fd-close fd) — close a raw file descriptor. @return Unspecified value. */
    llvm::Value* fdClose(const eshkol_operations_t* op);
    /** @brief (make-lru-cache capacity) — create a fixed-capacity LRU cache. */
    llvm::Value* makeLruCache(const eshkol_operations_t* op);
    /** @brief (lru-get cache key) — look up a key in an LRU cache. @return The value, or #f. */
    llvm::Value* lruGet(const eshkol_operations_t* op);
    /** @brief (lru-set! cache key value) — insert/update an entry in an LRU cache. */
    llvm::Value* lruSet(const eshkol_operations_t* op);
    /** @brief (lru-has? cache key) — whether a key is present in an LRU cache. */
    llvm::Value* lruHas(const eshkol_operations_t* op);
    /** @brief (lru-delete! cache key) — remove an entry from an LRU cache. */
    llvm::Value* lruDelete(const eshkol_operations_t* op);
    /** @brief (lru-clear! cache) — remove all entries from an LRU cache. */
    llvm::Value* lruClear(const eshkol_operations_t* op);
    /** @brief (lru-size cache) — number of entries currently in an LRU cache. */
    llvm::Value* lruSize(const eshkol_operations_t* op);
    /** @brief (_format-list ...) — internal formatted-string helper backing the format/printf-style builtins. */
    llvm::Value* formatList(const eshkol_operations_t* op);
    /** @brief (http-server-create port) — start a minimal HTTP server. @return A server handle. */
    llvm::Value* httpServerCreate(const eshkol_operations_t* op);
    /** @brief (http-server-port server) — the port an HTTP server is bound to. */
    llvm::Value* httpServerPort(const eshkol_operations_t* op);
    /** @brief (http-server-accept server) — accept the next incoming HTTP connection. */
    llvm::Value* httpServerAccept(const eshkol_operations_t* op);
    /** @brief (http-server-respond conn ...) — send an HTTP response on an accepted connection. */
    llvm::Value* httpServerRespond(const eshkol_operations_t* op);
    /** @brief (http-server-close server) — shut down an HTTP server. @return Unspecified value. */
    llvm::Value* httpServerClose(const eshkol_operations_t* op);
    /** @brief (http-request ...) — issue an HTTP client request. @return The response. */
    llvm::Value* httpRequest(const eshkol_operations_t* op);
    /** @brief (websocket-connect url) — open a WebSocket client connection. */
    llvm::Value* websocketConnect(const eshkol_operations_t* op);
    /** @brief (websocket-send ws text) — send a text frame on a WebSocket. */
    llvm::Value* websocketSend(const eshkol_operations_t* op);
    /** @brief (websocket-send-binary ws bytes) — send a binary frame on a WebSocket. */
    llvm::Value* websocketSendBinary(const eshkol_operations_t* op);
    /** @brief (websocket-receive ws) — receive the next WebSocket message. */
    llvm::Value* websocketReceive(const eshkol_operations_t* op);
    /** @brief (websocket-close ws) — close a WebSocket connection. @return Unspecified value. */
    llvm::Value* websocketClose(const eshkol_operations_t* op);
    /** @brief (ts-parser-new lang) — create a tree-sitter parser for a language. */
    llvm::Value* tsParserNew(const eshkol_operations_t* op);
    /** @brief (ts-parser-free parser) — release a tree-sitter parser. @return Unspecified value. */
    llvm::Value* tsParserFree(const eshkol_operations_t* op);
    /** @brief (ts-parse parser source) — parse source text. @return A parse-tree handle. */
    llvm::Value* tsParse(const eshkol_operations_t* op);
    /** @brief (ts-tree-free tree) — release a tree-sitter parse tree. @return Unspecified value. */
    llvm::Value* tsTreeFree(const eshkol_operations_t* op);
    /** @brief (ts-node-type node) — a tree-sitter node's grammar type name. */
    llvm::Value* tsNodeType(const eshkol_operations_t* op);
    /** @brief (ts-node-text node source) — the source text spanned by a tree-sitter node. */
    llvm::Value* tsNodeText(const eshkol_operations_t* op);
    /** @brief (ts-node-children node) — a tree-sitter node's direct children. */
    llvm::Value* tsNodeChildren(const eshkol_operations_t* op);
    /** @brief (ts-query-new lang pattern) — compile a tree-sitter query. */
    llvm::Value* tsQueryNew(const eshkol_operations_t* op);
    /** @brief (ts-query-matches query tree) — run a compiled query against a parse tree. */
    llvm::Value* tsQueryMatches(const eshkol_operations_t* op);
    /** @brief (ts-query-free query) — release a compiled tree-sitter query. @return Unspecified value. */
    llvm::Value* tsQueryFree(const eshkol_operations_t* op);
    /** @brief (ts-available?) — whether tree-sitter support was compiled into this build. */
    llvm::Value* tsAvailable(const eshkol_operations_t* op);
    /** @brief (ts-tree-root tree) — the root node of a tree-sitter parse tree. */
    llvm::Value* tsTreeRoot(const eshkol_operations_t* op);
    /** @brief (http-set-proxy url) — configure the proxy used by the HTTP client. @return Unspecified value. */
    llvm::Value* httpSetProxy(const eshkol_operations_t* op);
    /** @brief (http-set-tls-client-cert cert key) — configure a TLS client certificate for the HTTP client. */
    llvm::Value* httpSetTlsClientCert(const eshkol_operations_t* op);
    /** @brief (display-error obj) — write an error/condition object to stderr. */
    llvm::Value* displayError(const eshkol_operations_t* op);
    /** @brief (string-ends-with? s suffix) — whether a string ends with a given suffix. */
    llvm::Value* stringEndsWith(const eshkol_operations_t* op);
    /** @brief (string-index-of s sub) — index of the first occurrence of sub in s, or #f. */
    llvm::Value* stringIndexOf(const eshkol_operations_t* op);
    /** @brief (string-pad-left s width [char]) — left-pad a string to a minimum width. */
    llvm::Value* stringPadLeft(const eshkol_operations_t* op);
    /** @brief (string-pad-right s width [char]) — right-pad a string to a minimum width. */
    llvm::Value* stringPadRight(const eshkol_operations_t* op);
    /** @brief (kb-save kb path) — serialize a knowledge base to disk. @return #t/#f. */
    llvm::Value* kbSave(const eshkol_operations_t* op);
    /** @brief (kb-load path) — deserialize a knowledge base from disk. */
    llvm::Value* kbLoad(const eshkol_operations_t* op);
    /** @brief (tensor-token-estimate tensor) — estimate an LLM token count for tensor contents. */
    llvm::Value* tensorTokenEstimate(const eshkol_operations_t* op);

    /* Noesis requirements */

    /** @brief (fg-marginal graph node) — marginal distribution for a factor-graph node. */
    llvm::Value* fgMarginal(const eshkol_operations_t* op);
    /** @brief (fg-entropy graph) — entropy of a factor graph's current beliefs. */
    llvm::Value* fgEntropy(const eshkol_operations_t* op);
    /** @brief (kb-retract! kb fact) — remove a fact from a knowledge base. @return Unspecified value. */
    llvm::Value* kbRetract(const eshkol_operations_t* op);

    /* Consciousness engine */

    /** @brief (make-substitution) — create an empty substitution. */
    llvm::Value* makeSubstitution(const eshkol_operations_t* op);
    /** @brief (unify a b subst) — unify two terms under a substitution. @return Extended substitution or #f. */
    llvm::Value* unifyBuiltin(const eshkol_operations_t* op);
    /** @brief (walk term subst) — resolve a term through a substitution. */
    llvm::Value* walkBuiltin(const eshkol_operations_t* op);
    /** @brief (make-fact ...) — build a knowledge-base fact. */
    llvm::Value* makeFactBuiltin(const eshkol_operations_t* op);
    /** @brief (make-kb) — create an empty knowledge base. */
    llvm::Value* makeKbBuiltin(const eshkol_operations_t* op);
    /** @brief (kb-assert! kb fact) — add a fact to a knowledge base. @return Unspecified value. */
    llvm::Value* kbAssertBuiltin(const eshkol_operations_t* op);
    /** @brief (kb-query kb pattern) — query facts matching a pattern. */
    llvm::Value* kbQueryBuiltin(const eshkol_operations_t* op);
    /** @brief (make-factor-graph) — create an empty factor graph. */
    llvm::Value* makeFactorGraphBuiltin(const eshkol_operations_t* op);
    /** @brief (fg-add-factor! graph ...) — add a factor/CPT to a factor graph. @return Unspecified value. */
    llvm::Value* fgAddFactorBuiltin(const eshkol_operations_t* op);
    /** @brief (fg-infer! graph) — run belief propagation / inference on a factor graph. */
    llvm::Value* fgInferBuiltin(const eshkol_operations_t* op);
    /** @brief (free-energy graph) — compute a factor graph's variational free energy. */
    llvm::Value* freeEnergyBuiltin(const eshkol_operations_t* op);
    /** @brief (expected-free-energy graph ...) — compute expected free energy for active inference. */
    llvm::Value* expectedFreeEnergyBuiltin(const eshkol_operations_t* op);
    /** @brief (make-workspace) — create the global workspace. */
    llvm::Value* makeWorkspaceBuiltin(const eshkol_operations_t* op);
    /** @brief (ws-register! workspace module) — register a processing module with the workspace. */
    llvm::Value* wsRegisterBuiltin(const eshkol_operations_t* op);
    /** @brief (ws-step! workspace) — run one global-workspace broadcast cycle. */
    llvm::Value* wsStepBuiltin(const eshkol_operations_t* op);

    /* Differentiable external memory (core.dnc) */

    /** @brief (dnc-make ...) — create a Differentiable Neural Computer memory handle. */
    llvm::Value* dncMakeBuiltin(const eshkol_operations_t* op);
    /** @brief (dnc-content-address ...) — content-based addressing weights over DNC memory. */
    llvm::Value* dncContentAddressBuiltin(const eshkol_operations_t* op);
    /** @brief (dnc-loc-address ...) — location-based addressing weights over DNC memory. */
    llvm::Value* dncLocAddressBuiltin(const eshkol_operations_t* op);
    /** @brief (dnc-read ...) — read a vector from DNC memory using addressing weights. */
    llvm::Value* dncReadBuiltin(const eshkol_operations_t* op);
    /** @brief (dnc-write! ...) — write to DNC memory using addressing weights. @return Unspecified value. */
    llvm::Value* dncWriteBuiltin(const eshkol_operations_t* op);
    /** @brief (dnc-alloc-weights ...) — compute DNC memory allocation weighting. */
    llvm::Value* dncAllocWeightsBuiltin(const eshkol_operations_t* op);
    /** @brief (dnc-read-grad ...) — gradient of a DNC read operation. */
    llvm::Value* dncReadGradBuiltin(const eshkol_operations_t* op);
    /** @brief (dnc? obj) — whether obj is a DNC memory handle. */
    llvm::Value* dncPredBuiltin(const eshkol_operations_t* op);

    /* SDNC weight-program (core.sdnc) */

    /** @brief (sdnc-program ...) — compile a stateful-DNC weight-program handle. */
    llvm::Value* sdncProgramBuiltin(const eshkol_operations_t* op);
    /** @brief (sdnc-run ...) — execute a weight-program. @return Its output. */
    llvm::Value* sdncRunBuiltin(const eshkol_operations_t* op);
    /** @brief (sdnc-weight-grad ...) — gradient of a weight-program's output with respect to its weights. */
    llvm::Value* sdncWeightGradBuiltin(const eshkol_operations_t* op);
    /** @brief (sdnc-params ...) — current parameter tensor of a weight-program. */
    llvm::Value* sdncParamsBuiltin(const eshkol_operations_t* op);
    /** @brief (sdnc-set-params! ...) — overwrite a weight-program's parameters. @return Unspecified value. */
    llvm::Value* sdncSetParamsBuiltin(const eshkol_operations_t* op);
    /** @brief (sdnc-improve! ...) — apply a gradient step to a weight-program's parameters. */
    llvm::Value* sdncImproveBuiltin(const eshkol_operations_t* op);
    /** @brief (sdnc? obj) — whether obj is a weight-program handle. */
    llvm::Value* sdncPredBuiltin(const eshkol_operations_t* op);

    /* Reverse-mode AD tape */

    /** @brief (ad-tape-new) — create a new reverse-mode AD tape. */
    llvm::Value* adTapeNew(const eshkol_operations_t* op);
    /** @brief (ad-tape-release tape) — release an AD tape. @return Unspecified value. */
    llvm::Value* adTapeRelease(const eshkol_operations_t* op);
    /** @brief (ad-const value) — wrap a constant as an AD node (zero gradient). */
    llvm::Value* adConst(const eshkol_operations_t* op);
    /** @brief (ad-var value) — wrap a value as a differentiable AD leaf node. */
    llvm::Value* adVar(const eshkol_operations_t* op);
    /**
     * @brief Shared implementation for binary reverse-mode AD ops (ad-add, ad-sub, ad-mul, ad-div).
     * @param op The operation AST node (two AD-node operands)
     * @param func_name Name of the runtime `eshkol_ad_*_sret` function to call
     * @return New AD node recording the operation on the tape
     */
    llvm::Value* adBinaryOp(const eshkol_operations_t* op, const char* func_name);
    /**
     * @brief Shared implementation for unary reverse-mode AD ops (ad-sin, ad-cos, ad-exp,
     * ad-log, ad-sqrt, ad-neg, ad-abs, ad-relu, ad-sigmoid, ad-tanh).
     * @param op The operation AST node (one AD-node operand)
     * @param func_name Name of the runtime `eshkol_ad_*_sret` function to call
     * @return New AD node recording the operation on the tape
     */
    llvm::Value* adUnaryOp(const eshkol_operations_t* op, const char* func_name);
    /** @brief (ad-backward node) — run backpropagation from an AD node. @return Unspecified value. */
    llvm::Value* adBackward(const eshkol_operations_t* op);
    /** @brief (ad-gradient node wrt) — the accumulated gradient of node with respect to wrt. */
    llvm::Value* adGradient(const eshkol_operations_t* op);
    /** @brief (ad-node-value node) / (ad-value node) — an AD node's forward (primal) value. */
    llvm::Value* adNodeValue(const eshkol_operations_t* op);
    /** @brief (onnx-export-tensor ...) — export a tensor/model in ONNX format. */
    llvm::Value* onnxExportTensor(const eshkol_operations_t* op);

    /* Type predicates */

    /** @brief (logic-var? obj) — whether obj is a logic variable. */
    llvm::Value* logicVarPred(const eshkol_operations_t* op);
    /** @brief (substitution? obj) — whether obj is a substitution. */
    llvm::Value* substitutionPred(const eshkol_operations_t* op);
    /** @brief (fact? obj) — whether obj is a knowledge-base fact. */
    llvm::Value* factPred(const eshkol_operations_t* op);
    /** @brief (kb? obj) — whether obj is a knowledge base. */
    llvm::Value* kbPred(const eshkol_operations_t* op);
    /** @brief (factor-graph? obj) — whether obj is a factor graph. */
    llvm::Value* factorGraphPred(const eshkol_operations_t* op);
    /** @brief (workspace? obj) — whether obj is a global workspace. */
    llvm::Value* workspacePred(const eshkol_operations_t* op);
    /** @brief (tensor? obj) — whether obj is a tensor. */
    llvm::Value* tensorPred(const eshkol_operations_t* op);
    /** @brief (dual? obj) — whether obj is a forward-mode AD dual number. */
    llvm::Value* dualPred(const eshkol_operations_t* op);
    /** @brief (fg-update-cpt! graph ...) — update a factor's conditional probability table (function-call dispatch path). */
    llvm::Value* fgUpdateCpt(const eshkol_operations_t* op);
    /** @brief (kb-count kb) — number of facts stored in a knowledge base. */
    llvm::Value* kbCount(const eshkol_operations_t* op);

    /* Image I/O */

    /** @brief (image-read path) — load an image file. @return A tensor of pixel data. */
    llvm::Value* imageRead(const eshkol_operations_t* op);
    /** @brief (image-write tensor path) — write a pixel tensor to an image file. @return #t/#f. */
    llvm::Value* imageWrite(const eshkol_operations_t* op);
    /** @brief (image-to-grayscale tensor) — convert a pixel tensor to grayscale. */
    llvm::Value* imageGrayscale(const eshkol_operations_t* op);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;
    std::unordered_map<std::string, llvm::Function*>& function_table_;

    // Callback for AST code generation
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    void* callback_context_ = nullptr;

    // Helper to extract string pointer from tagged value
    llvm::Value* extractStringPtr(llvm::Value* tagged_val);

public:
    /**
     * Set callbacks for AST code generation.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_SYSTEM_CODEGEN_H
