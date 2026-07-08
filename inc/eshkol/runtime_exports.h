/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Stable C ABI exported by the runtime for generated programs and the REPL JIT.
 * These wrappers hide platform-specific libc differences from LLVM codegen.
 */

#ifndef ESHKOL_RUNTIME_EXPORTS_H
#define ESHKOL_RUNTIME_EXPORTS_H

#include <cstdint>
#include <cstdio>

namespace eshkol::runtime {

/**
 * @brief Linker/JIT symbol names for the extern "C" runtime wrappers
 *        declared below.
 *
 * LLVM codegen uses these string constants (rather than hardcoding the
 * name at each call site) when emitting external function declarations or
 * looking up functions already inserted into a Module, so the C++ symbol
 * name and the name generated code links against stay in sync from a
 * single source of truth.
 */
inline constexpr const char* stdout_stream_symbol = "eshkol_stdout_stream";
inline constexpr const char* stdin_stream_symbol = "eshkol_stdin_stream";
inline constexpr const char* stderr_stream_symbol = "eshkol_stderr_stream";
inline constexpr const char* jmp_buf_size_symbol = "eshkol_jmp_buf_size";
inline constexpr const char* drand48_symbol = "eshkol_drand48";
inline constexpr const char* srand48_symbol = "eshkol_srand48";
inline constexpr const char* getenv_symbol = "eshkol_getenv";
inline constexpr const char* setenv_symbol = "eshkol_setenv";
inline constexpr const char* unsetenv_symbol = "eshkol_unsetenv";
inline constexpr const char* usleep_symbol = "eshkol_usleep";
inline constexpr const char* fopen_symbol = "eshkol_fopen";
inline constexpr const char* fputs_symbol = "eshkol_fputs";
inline constexpr const char* access_symbol = "eshkol_access";
inline constexpr const char* remove_symbol = "eshkol_remove";
inline constexpr const char* rename_symbol = "eshkol_rename";
inline constexpr const char* mkdir_symbol = "eshkol_mkdir";
inline constexpr const char* rmdir_symbol = "eshkol_rmdir";
inline constexpr const char* chdir_symbol = "eshkol_chdir";
inline constexpr const char* stat_symbol = "eshkol_stat";
inline constexpr const char* opendir_symbol = "eshkol_opendir";

} // namespace eshkol::runtime

extern "C" {

/**
 * @brief Returns the process's standard output stream.
 * @return The libc `stdout` FILE*. Never NULL; not owned by the caller.
 */
FILE* eshkol_stdout_stream();

/**
 * @brief Returns the process's standard input stream.
 * @return The libc `stdin` FILE*. Never NULL; not owned by the caller.
 */
FILE* eshkol_stdin_stream();

/**
 * @brief Returns the process's standard error stream.
 * @return The libc `stderr` FILE*. Never NULL; not owned by the caller.
 */
FILE* eshkol_stderr_stream();

/**
 * @brief Returns the platform's `sizeof(jmp_buf)` so generated code can
 *        allocate correctly sized buffers for setjmp/longjmp without
 *        embedding a platform-specific constant.
 * @return Size of `jmp_buf` in bytes on the current platform.
 */
std::uint64_t eshkol_jmp_buf_size();

/**
 * @brief Portable drand48()-equivalent pseudo-random number generator.
 *
 * Implements the same 48-bit linear congruential algorithm as POSIX
 * drand48(), providing consistent behavior on platforms (e.g. Windows)
 * that lack a native implementation.
 *
 * @return A pseudo-random double uniformly distributed in [0.0, 1.0).
 * @note Thread-safe; the generator state is protected by an internal
 *       mutex shared with eshkol_srand48().
 */
double eshkol_drand48();

/**
 * @brief Seeds the generator used by eshkol_drand48().
 * @param seed Seed value; combined with drand48's fixed low-order bits
 *        (0x330E) as in the POSIX algorithm.
 * @note Thread-safe.
 */
void eshkol_srand48(std::int64_t seed);

/**
 * @brief Reads an environment variable, gated by the "env-read" capability.
 *
 * @param name Name of the environment variable to look up. NULL or empty
 *        returns NULL without consulting the environment.
 * @return The variable's value on success, or NULL if name is NULL/empty,
 *         the "env-read" capability is not granted, or the variable is
 *         unset. The returned pointer is owned by the C library (as with
 *         plain getenv()); do not free it, and it may be invalidated by a
 *         subsequent setenv/putenv.
 */
char* eshkol_getenv(const char* name);

/**
 * @brief Sets an environment variable, gated by the "env-write" capability.
 *
 * @param name Variable name; NULL or empty is an error.
 * @param value New value; NULL is an error.
 * @param overwrite If 0 and the variable is already set, leaves it
 *        unchanged and returns success (0); if non-zero, always sets it.
 * @return 0 on success; -1 on error (invalid arguments, capability denied,
 *         or the underlying platform call failing), with errno set
 *         accordingly (EINVAL for invalid arguments).
 */
int eshkol_setenv(const char* name, const char* value, int overwrite);

/**
 * @brief Removes an environment variable, gated by the "env-write"
 *        capability.
 * @param name Variable name; NULL or empty is an error.
 * @return 0 on success; -1 on error (invalid argument or capability
 *         denied), with errno set to EINVAL for invalid arguments.
 */
int eshkol_unsetenv(const char* name);

/**
 * @brief Suspends the calling thread for at least the given duration.
 * @param usec Sleep duration in microseconds.
 * @return Always 0.
 */
int eshkol_usleep(std::uint32_t usec);

/**
 * @brief Opens a file, gated by the "file-read"/"file-write" capabilities
 *        implied by `mode`.
 *
 * On Windows, `/tmp`-rooted and `/<drive-letter>/...` style paths are
 * remapped to their native equivalents before opening.
 *
 * @param path Path to open; NULL is an error.
 * @param mode fopen()-style mode string (e.g. "r", "w", "a", with a
 *        trailing '+' requiring both read and write capability); NULL is
 *        an error.
 * @return An open FILE* owned by the caller (must be closed with fclose),
 *         or NULL if path/mode is NULL, the required capability was
 *         denied, or the underlying fopen() failed. errno is set to
 *         EINVAL for invalid arguments or otherwise left as set by fopen().
 */
FILE* eshkol_fopen(const char* path, const char* mode);

/**
 * @brief Writes a NUL-terminated string to a stream.
 * @param str String to write; NULL is an error.
 * @param stream Destination stream; NULL is an error.
 * @return A non-negative value on success, or EOF on error (including
 *         NULL arguments, which also set errno to EINVAL).
 */
int eshkol_fputs(const char* str, FILE* stream);

/**
 * @brief Checks file accessibility, gated by the "file-read"/"file-write"
 *        capabilities implied by `mode`.
 * @param path Path to check; NULL is an error.
 * @param mode POSIX access() mode bitmask (e.g. R_OK=4, W_OK=2, F_OK=0).
 * @return 0 if the requested access is permitted; -1 on error (invalid
 *         argument, capability denied, or the underlying access()/_access()
 *         call failing).
 */
int eshkol_access(const char* path, int mode);

/**
 * @brief Deletes a file, gated by the "file-write" capability.
 * @param path Path to remove; NULL is an error.
 * @return 0 on success; -1 on error (invalid argument, capability denied,
 *         or the underlying remove() call failing).
 */
int eshkol_remove(const char* path);

/**
 * @brief Renames/moves a file, gated by the "file-write" capability.
 * @param old_path Existing path; NULL is an error.
 * @param new_path Destination path; NULL is an error.
 * @return 0 on success; -1 on error (invalid argument, capability denied,
 *         or the underlying rename() call failing).
 */
int eshkol_rename(const char* old_path, const char* new_path);

/**
 * @brief Creates a directory, gated by the "file-write" capability.
 * @param path Directory path to create; NULL is an error.
 * @param mode POSIX permission bits for the new directory (ignored on
 *        Windows).
 * @return 0 on success; -1 on error (invalid argument, capability denied,
 *         or the underlying mkdir()/_mkdir() call failing).
 */
int eshkol_mkdir(const char* path, int mode);

/**
 * @brief Removes an empty directory, gated by the "file-write" capability.
 * @param path Directory path to remove; NULL is an error.
 * @return 0 on success; -1 on error (invalid argument, capability denied,
 *         or the underlying rmdir()/_rmdir() call failing).
 */
int eshkol_rmdir(const char* path);

/**
 * @brief Changes the process's current working directory, gated by the
 *        "file-read" capability.
 * @param path New working directory; NULL is an error.
 * @return 0 on success; -1 on error (invalid argument, capability denied,
 *         or the underlying chdir()/_chdir() call failing).
 */
int eshkol_chdir(const char* path);

/**
 * @brief Retrieves file status information, gated by the "file-read"
 *        capability.
 * @param path Path to stat; NULL is an error.
 * @param buf Output buffer, treated as a native `struct stat*`; NULL is
 *        an error. Caller-owned.
 * @return 0 on success; -1 on error (invalid argument, capability denied,
 *         or the underlying stat() call failing).
 */
int eshkol_stat(const char* path, void* buf);

/**
 * @brief Opens a directory stream for iteration, gated by the "file-read"
 *        capability.
 * @param path Directory path to open; NULL is an error.
 * @return An opaque directory handle (a `DIR*` on POSIX platforms, or an
 *         internal handle on Windows) owned by the caller and released via
 *         the platform's closedir()-equivalent, or NULL on error (invalid
 *         argument, capability denied, or the path could not be opened).
 */
void* eshkol_opendir(const char* path);

/**
 * @brief Resets the capability runtime to its default state: policy
 *        inactive and the allow-list cleared, so eshkol_capability_runtime_allows()
 *        permits everything.
 * @note Thread-safe.
 */
void eshkol_capability_runtime_clear();

/**
 * @brief Activates capability enforcement and clears any previously
 *        installed allow-list, in preparation for a fresh set of
 *        eshkol_capability_runtime_allow() calls.
 * @note Thread-safe.
 */
void eshkol_capability_runtime_begin_install();

/**
 * @brief Grants a capability, adding it to the active allow-list.
 *
 * Has no effect if the capability policy is not active (i.e. this must be
 * called after eshkol_capability_runtime_begin_install()).
 *
 * @param capability Capability name to grant (e.g. "file-read",
 *        "file-write", "env-read", "env-write"). NULL or empty is ignored.
 * @note Thread-safe.
 */
void eshkol_capability_runtime_allow(const char* capability);

/**
 * @brief Reports whether capability enforcement is currently active.
 * @return 1 if a capability policy has been installed via
 *         eshkol_capability_runtime_begin_install() and not since cleared;
 *         0 otherwise (in which case all capabilities are implicitly
 *         allowed).
 */
int eshkol_capability_runtime_is_active();

/**
 * @brief Checks whether a capability is currently permitted.
 * @param capability Capability name to check. NULL or empty is always
 *        denied.
 * @return 1 if the capability is allowed (either because no policy is
 *         active, or because it is present in the active allow-list); 0
 *         otherwise.
 */
int eshkol_capability_runtime_allows(const char* capability);

/**
 * @brief Checks whether an fopen()-style mode string's implied
 *        read/write requirements are currently permitted.
 *
 * A mode requires read for a leading 'r', write for a leading 'w' or 'a',
 * and both when '+' appears anywhere in the mode string.
 *
 * @param mode fopen()-style mode string (e.g. "r", "w+", "a"). NULL or
 *        empty is treated as requiring "file-read".
 * @return 1 if all capabilities implied by mode are currently allowed; 0
 *         otherwise.
 */
int eshkol_capability_runtime_allows_file_mode(const char* mode);

/**
 * @brief Records that a capability was denied, for diagnostics.
 *
 * Sets errno to EACCES and prints a one-time "capability denied: <name>"
 * message to stderr the first time a given capability name is denied
 * (subsequent denials of the same name are silent).
 *
 * @param capability Capability name that was denied; NULL or empty is
 *        reported as "unknown".
 * @note Thread-safe.
 */
void eshkol_capability_runtime_deny(const char* capability);

}

#endif // ESHKOL_RUNTIME_EXPORTS_H
