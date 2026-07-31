/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef ESHKOL_PLATFORM_RUNTIME_H
#define ESHKOL_PLATFORM_RUNTIME_H

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace eshkol::platform {

/**
 * @brief Get the absolute path to the currently running executable.
 * @return Absolute path resolved via the platform-native API
 *         (GetModuleFileNameW on Windows, _NSGetExecutablePath on macOS,
 *         /proc/self/exe on Linux), or an empty path if it cannot be
 *         determined (including on unsupported platforms).
 */
std::filesystem::path executable_path();

/**
 * @brief Get the directory containing the currently running executable.
 * @return Parent directory of executable_path(), or an empty path if
 *         executable_path() could not be determined.
 */
std::filesystem::path executable_directory();

/**
 * @brief Get the directory containing the currently running executable, with
 *        every symlink on the way resolved.
 *
 * `/proc/self/exe` is already symlink-free, but `_NSGetExecutablePath` and
 * `GetModuleFileNameW` report the path the process was *launched* through.
 * When that is a symlink into an install tree (the Homebrew
 * `<prefix>/bin/eshkol-run -> ../Cellar/eshkol/<v>/bin/eshkol-run` layout, or
 * any hand-made `~/bin` link), the launch path's parent is not the directory
 * the compiler's own runtime archive/module tree was installed into.
 * Artifact resolution must start from the real location, so it is this
 * function — not executable_directory() — that seeds the co-located search
 * tier.
 *
 * @return Symlink-resolved parent directory of the running executable, or an
 *         empty path if it cannot be determined.
 */
std::filesystem::path executable_real_directory();

/**
 * @brief Get the process's current working directory.
 * @return Absolute current working directory, or an empty path on error.
 */
std::filesystem::path current_directory();

/**
 * @brief Find the first candidate path that exists on disk.
 * @param candidates Ordered list of paths to probe.
 * @return The canonicalized (or, if canonicalization fails, absolute) string
 *         form of the first candidate that exists, or an empty string if
 *         none of the candidates exist.
 */
std::string find_first_existing(const std::vector<std::filesystem::path>& candidates);

// ---------------------------------------------------------------------------
// Install-artifact resolution (runtime archives, stdlib.o/.bc, module tree)
// ---------------------------------------------------------------------------
//
// Every artifact the toolchain resolves at run time — libeshkol-runtime.a /
// libeshkol-static.a, libeshkol-agent-*.a, stdlib.o, stdlib.bc, and the
// lib/**.esk module tree — is looked up through the ordered root list built
// by install_library_roots() / install_module_roots() and probed by
// resolve_install_artifact().
//
// The order is LOCATION-major: each root is fully probed (for every accepted
// file name) before the next root is considered.  A search that is instead
// name-major — all locations for name A, then all locations for name B —
// lets a system copy of A outrank the compiler's own co-located B, which is
// how a v1.3.4 compiler could link an eleven-day-old /usr/local/lib archive
// in preference to the archive installed beside it.  Resolution must never
// leave the install the compiler belongs to unless that install genuinely
// does not carry the artifact.

/** @brief Precedence tier an install artifact was resolved from, highest
 *         precedence first. */
enum class InstallOrigin {
    NotFound = 0,   ///< No candidate existed in any root.
    EnvOverride,    ///< $ESHKOL_LIB_DIR — the absolute escape hatch.
    ExplicitFlag,   ///< A `-L<dir>` given on the command line.
    CoLocated,      ///< Beside the running executable's real path.
    WorkingTree,    ///< cwd / cwd's build tree — developer convenience.
    SystemFallback, ///< A system prefix (/usr/local, /usr, /opt/homebrew, …).
};

/** @brief One directory to probe, tagged with the tier it came from. */
struct InstallSearchRoot {
    std::filesystem::path path;
    InstallOrigin origin = InstallOrigin::NotFound;
};

/** @brief Outcome of an install-artifact lookup. */
struct ResolvedInstallArtifact {
    std::string path;            ///< Canonical path; empty when unresolved.
    InstallOrigin origin = InstallOrigin::NotFound;
    std::filesystem::path root;  ///< Root the artifact was found under.

    bool found() const { return !path.empty(); }
};

/**
 * @brief Build the ordered root list for native install artifacts (static
 *        archives, stdlib.o, stdlib.bc).
 *
 * Tiers, in order: $ESHKOL_LIB_DIR (and its `eshkol/` subdirectory), each
 * @p explicit_dirs entry (`-L`), the real executable directory plus its
 * `../lib` and `../lib/eshkol`, the same for the launch-path directory when
 * that differs, the working directory and its build trees, and finally the
 * system prefixes.
 *
 * @param explicit_dirs Directories named by `-L` flags, in command-line order.
 * @return Roots to probe, highest precedence first, duplicates removed.
 */
std::vector<InstallSearchRoot> install_library_roots(
    const std::vector<std::string>& explicit_dirs = {});

/**
 * @brief Build the ordered root list for the Eshkol module source tree
 *        (`stdlib.esk` and the `core/`, `agent/`, … directories beside it).
 *
 * Same tier order as install_library_roots(), with the layouts that carry
 * module sources: `<dir>/lib`, `<dir>/../lib`, `<dir>/../share/eshkol/lib`
 * and `<prefix>/share/eshkol/lib`.
 *
 * @return Roots to probe, highest precedence first, duplicates removed.
 */
std::vector<InstallSearchRoot> install_module_roots();

/**
 * @brief Probe @p roots in order and return the first existing artifact.
 *
 * Within a single root every entry of @p leaf_names is tried in order, so a
 * lower-precedence root can never outrank a higher-precedence one merely by
 * carrying an earlier-named file.
 *
 * @param roots      Ordered roots, as built by install_library_roots() /
 *                   install_module_roots().
 * @param leaf_names Accepted file (or subdirectory) names, most preferred
 *                   first — e.g. {"libeshkol-runtime.a", "libeshkol-static.a"}.
 * @return The resolved artifact, or a ResolvedInstallArtifact whose found()
 *         is false when no root carries any of the names.
 */
ResolvedInstallArtifact resolve_install_artifact(
    const std::vector<InstallSearchRoot>& roots,
    const std::vector<std::string>& leaf_names);

/** @brief Human-readable name of a precedence tier, for diagnostics. */
const char* install_origin_name(InstallOrigin origin);

/**
 * @brief Read the Eshkol version an archive (or object file) was built from.
 *
 * Scans @p artifact for the build stamp that every translation unit of
 * lib/core/platform_runtime.cpp embeds, so it works for `libeshkol-runtime.a`
 * and `libeshkol-static.a` regardless of archive format and without an `ar`
 * reader.
 *
 * @param artifact File to scan.
 * @return The stamped version string (e.g. "1.3.4-evolve"), or an empty
 *         string when @p artifact carries no stamp — which is the case for
 *         every archive built before the stamp existed.
 */
std::string archive_build_version(const std::filesystem::path& artifact);

/** @brief A diagnostic about where an artifact came from. */
struct InstallArtifactNote {
    std::string text;     ///< Message to print; empty when nothing to report.
    bool severe = false;  ///< True when the artifact's version disagrees with this build.
};

/**
 * @brief Describe a resolution that the user should know about.
 *
 * Reports two situations and stays silent otherwise: an artifact taken from a
 * system prefix rather than from the compiler's own install, and an artifact
 * whose embedded build stamp disagrees with this compiler's
 * ESHKOL_VERSION_STRING (the silent-ABI-skew case, flagged as severe).
 *
 * @param label    What was resolved, e.g. "runtime archive".
 * @param artifact The resolution to describe.
 * @param verify_version Whether to read @p artifact's build stamp; pass false
 *        for artifacts that carry no stamp (stdlib.o, module trees).
 * @return The note to print, or a note whose text is empty.
 */
InstallArtifactNote describe_install_artifact(std::string_view label,
                                             const ResolvedInstallArtifact& artifact,
                                             bool verify_version = true);

// ---------------------------------------------------------------------------
// Module source resolution — `(require m)`, `(import "…")`, `(load "…")`
// ---------------------------------------------------------------------------
//
// ONE resolver, used by every execution path of the driver.  `eshkol-run -r`
// alone reaches the language through two engines — the persistent JIT run
// cache (which compiles ahead of time) and the in-process LLVM JIT (used
// whenever the cache is bypassed: `ESHKOL_JIT_CACHE=0`, `-d`, `--dump-ast`,
// `--dump-ir`, several inputs, or an active
// `$ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR`).  While each engine carried its own
// copy of the search order they disagreed about what a relative `(load "…")`
// means: the AOT copy resolved it against the requiring FILE's directory, the
// JIT copy against the process's working directory.  One command, two answers
// — so the resolver lives here, below both, and neither engine may keep a
// private copy.
//
// The order is the one documented in docs/reference/language/modules.md.

/**
 * @brief Locate the Eshkol module source tree (the directory that carries
 *        `stdlib.esk` beside the `core/`, `agent/`, … subtrees).
 *
 * Prefers the first install_module_roots() entry that actually carries
 * `stdlib.esk` — that check is what keeps a downstream project's unrelated
 * `lib/`, or a release package's archive-only `lib/`, from being mistaken for
 * the module tree.  Falling back, returns the first root that is merely a
 * directory, never `$ESHKOL_LIB_DIR` (it names a directory of native
 * archives; accepting it would point `(require …)` at a tree with no `.esk`
 * files in it).
 *
 * @return The resolved root; `origin == InstallOrigin::NotFound` and an empty
 *         path when no candidate exists.
 */
InstallSearchRoot module_source_root();

/**
 * @brief Resolve a module name or path literal to an existing `.esk` source
 *        file — the single authority behind `require`, `import` and `load`.
 *
 * @p module_name is either a dotted module name (`core.list.transform`, whose
 * dots become directory separators before `.esk` is appended) or a path
 * literal — a string that starts with `/`, `./` or `../`, contains a `/`, or
 * ends in `.esk`.  Path literals are taken verbatim, with no dot-to-slash
 * rewrite, which is what makes `(load "some/dir.v2/file.esk")` work on a
 * directory whose name contains a dot (macOS `$TMPDIR` is
 * `/var/folders/<hash>.<rand>/T`).  A path literal without the extension is
 * probed both as given and with `.esk` appended, in each tier, so the
 * extension is optional wherever the file actually lives rather than only in
 * the working directory.
 *
 * Search order (docs/reference/language/modules.md):
 *   1. @p base_dir — the directory of the file doing the require/import/load;
 *   2. the process working directory (the project root), so a project-rooted
 *      dotted name like `src.core.x` resolves against `./src/…`;
 *   3. `$ESHKOL_PATH` (the `-I` flags are merged into it), highest-priority
 *      user override, deliberately ahead of the install for the same reason
 *      `$ESHKOL_LIB_DIR` precedes every archive location;
 *   4. @p lib_dir, including directory-as-module entry points
 *      (`<lib>/web/web.esk`, then `<lib>/web/index.esk`);
 *   5. `lib/…` and `../lib/…` relative to the working directory — the
 *      build-tree fallback for a layout whose `lib_dir` resolved elsewhere.
 *
 * Empty, missing and non-directory `$ESHKOL_PATH` segments are skipped.
 *
 * @param module_name Dotted module name or path literal, exactly as written.
 * @param base_dir    Requiring file's directory; `"."` when there is none.
 * @param lib_dir     Module tree root, as returned by module_source_root();
 *                    may be empty.
 * @return Canonical path of the first match, or an empty string when the
 *         module is not on any tier.
 */
std::string resolve_module_source_path(const std::string& module_name,
                                       const std::string& base_dir,
                                       const std::string& lib_dir);

/**
 * @brief Declare, for the duration of the scope, which source file's forms
 *        are being compiled — tier 1 of resolve_module_source_path().
 *
 * Nested loads stack: while `a/main.esk` loads `a/helper.esk`, which loads
 * `"util.esk"`, the innermost scope is `a/helper.esk`, so `util.esk` is looked
 * for beside `helper.esk` — not beside the outermost file and not in the
 * working directory.  Both engines push the same scopes, which is what makes
 * their answers identical.
 *
 * A pseudo-path (`"<repl>"`, `""`) pushes nothing, leaving resolution rooted
 * at the working directory — the right answer for an expression typed at the
 * REPL or passed with `-e`.
 */
class ScopedRequiringFile {
public:
    explicit ScopedRequiringFile(const std::string& source_path);
    ~ScopedRequiringFile();

    ScopedRequiringFile(const ScopedRequiringFile&) = delete;
    ScopedRequiringFile& operator=(const ScopedRequiringFile&) = delete;

private:
    bool pushed_ = false;
};

/**
 * @brief Directory of the innermost active ScopedRequiringFile.
 * @return That directory, or `"."` when no source file is being compiled.
 */
std::string current_requiring_directory();

/**
 * @brief Get the current user's home directory.
 * @return Value of the `HOME` environment variable; on Windows, falls back
 *         to `USERPROFILE` then `APPDATA` if `HOME` is unset; if none of
 *         those are set, falls back to current_directory(); returns an
 *         empty string if nothing can be determined.
 */
std::string home_directory();

/**
 * @brief Check whether standard input is connected to an interactive
 *        terminal.
 * @return true if stdin is a TTY.
 */
bool stdin_isatty();

/**
 * @brief Check whether standard output is connected to an interactive
 *        terminal.
 * @return true if stdout is a TTY.
 */
bool stdout_isatty();

/**
 * @brief Configure the console for interactive UTF-8 input/output.
 *
 * On Windows, sets the console output (and, if stdin is a TTY, input) code
 * page to UTF-8 and enables ANSI virtual terminal processing on stdout when
 * available. No-op on other platforms.
 *
 * @return true if stdout ends up supporting UTF-8 after configuration
 *         (see stdout_supports_utf8()); false if stdout is not a TTY, or
 *         unconditionally false on non-Windows platforms.
 */
bool initialize_interactive_console();

/**
 * @brief Check whether stdout can currently render UTF-8 text.
 * @return On Windows, true only if stdout is a TTY and the console output
 *         code page is UTF-8; always true on other platforms.
 */
bool stdout_supports_utf8();

/**
 * @brief Generate a path to a not-currently-existing temporary file.
 *
 * Combines @p stem, a randomized numeric suffix, and @p extension inside the
 * system temporary directory (falling back to current_directory() if the
 * temp directory cannot be determined). No file is created by this call, so
 * the returned path is not guaranteed to remain unused afterward.
 *
 * @param stem      Filename prefix.
 * @param extension Filename suffix, including the leading dot (default
 *                  ".tmp").
 * @return Candidate temporary file path.
 */
std::filesystem::path make_temp_path(std::string_view stem, std::string_view extension = ".tmp");

/**
 * @brief Resolve the C++ compiler used to link generated programs.
 *
 * Resolution order is the runtime `ESHKOL_CXX_COMPILER` override, the
 * build-time compiler when it still exists, PATH, and platform-standard LLVM
 * installation roots. This makes installed packages independent of the build
 * machine's filesystem while retaining exact build-tree toolchains.
 *
 * @return Driver path or the platform's default clang++/c++ command name.
 */
std::string cxx_compiler();

/**
 * @brief Resolve the compiler-rt builtins archive belonging to the selected
 *        Windows C++ driver.
 *
 * ClangCL/MSVC builds use a GNU-compatible clang++ driver for generated
 * executables.  Unlike a complete compiler-driver link, that path does not
 * reliably inject compiler-rt when Eshkol's static runtime introduces
 * 128-bit division helpers.  The archive must therefore be selected from the
 * consumer toolchain at runtime rather than recorded as an absolute path from
 * the build host.
 *
 * @return Canonical path to clang_rt.builtins-x86_64.lib or
 *         clang_rt.builtins-aarch64.lib on native ClangCL/MSVC Windows;
 *         empty on non-Windows/MinGW or when no matching archive is present.
 */
std::string compiler_rt_builtins_library();

/**
 * @brief Normalize one library/linker argument for the configured host C++
 *        compiler driver.
 *
 * On Windows Eshkol invokes the GNU-compatible `clang++` driver for generated
 * programs. A bare MSVC library token such as `winhttp.lib` is interpreted by
 * that driver as an input file in the current directory; `-lwinhttp` is the
 * portable driver form that searches the Visual Studio/Windows SDK paths
 * discovered by Clang. Absolute and relative paths are retained. On other
 * platforms the argument is returned unchanged.
 *
 * @param argument Raw configured link argument.
 * @return Driver-ready argument.
 */
std::string cxx_driver_link_arg(std::string argument);

/**
 * @brief Get the path to the `llc` executable bundled with the LLVM
 *        toolchain used at build time.
 * @return Absolute path baked in at build time (ESHKOL_HOST_LLC_EXECUTABLE).
 */
std::string llc_executable();

/**
 * @brief Get the platform's native executable filename suffix.
 * @return Suffix such as ".exe" on Windows, or an empty string on platforms
 *         with no executable extension.
 */
std::string executable_suffix();

/**
 * @brief Build the platform's native static library filename for a given
 *        library stem.
 * @param stem Base component of the library name, without prefix or suffix
 *             (e.g. "eshkol").
 * @return Platform-specific filename, e.g. "libeshkol.a" on Unix-like
 *         systems or "eshkol.lib" on Windows.
 */
std::string static_library_name(std::string_view stem);

/**
 * @brief Resolve logical CUDA library names against the consumer toolkit.
 *
 * Build-host CUDA imported targets contain absolute SDK paths that are not
 * portable to generated AOT or persistent-cache links. This routine locates a
 * single consumer-side development-library directory from explicit CUDA root
 * variables, nvcc, and platform-standard layouts, then returns driver-ready
 * search/RUNPATH and `-l` arguments. If no directory is found, driver-search
 * names are retained (ABI-major-exact on Linux) so the compiler driver's
 * normal search remains authoritative and produces the final diagnostic.
 *
 * @param libraries Logical names such as `cudart`, `cublas`, and `cublasLt`.
 * @return Consumer-resolved C++ driver link arguments.
 */
std::vector<std::string> cuda_runtime_link_args(
    const std::vector<std::string>& libraries);

/**
 * @brief Get the linker arguments required to link against the host
 *        runtime libraries.
 * @return Arguments parsed from the build-time
 *         ESHKOL_HOST_RUNTIME_LINK_ARGS list (semicolon-separated), with
 *         forward-slash paths normalized to backslashes on Windows.
 */
std::vector<std::string> host_runtime_link_args();
// On macOS, the resolved SDK library directory (<sdk>/usr/lib) discovered at
// runtime via `xcrun --show-sdk-path`. Adding it to the linker search path lets
// a bare `-lobjc` resolve on any mac, not just the builder. Empty on non-macOS
// or when xcrun is unavailable.
std::string macos_sdk_lib_dir();

/**
 * @brief Ensure a path carries the platform's native executable suffix.
 * @param path Base path, which may or may not already have the suffix.
 * @return An empty path if @p path is empty; @p path unchanged if the
 *         platform's executable suffix is empty or @p path's extension
 *         already matches it; otherwise @p path with the suffix appended.
 */
std::filesystem::path with_executable_suffix(const std::filesystem::path& path);

/**
 * @brief Quote a single argument for safe inclusion in a platform shell
 *        command line.
 * @param argument Raw, unescaped argument text.
 * @return The argument quoted/escaped for the platform's shell (cmd.exe
 *         double-quote escaping on Windows, POSIX single-quote escaping
 *         elsewhere); returned unmodified if it needs no escaping.
 */
std::string shell_quote(std::string_view argument);

/**
 * @brief Run an external command synchronously and wait for it to finish.
 * @param arguments Argv-style argument vector; `arguments[0]` is the
 *                  program path or name to launch.
 * @return The child process's exit code on success; -1 if @p arguments is
 *         empty; on Windows, a `GetLastError()`-derived code (e.g.
 *         `ERROR_INVALID_PARAMETER`) if the process could not be launched.
 * @note On non-Windows platforms this shells out via `std::system()` with
 *       each argument passed through shell_quote().
 */
int run_command(const std::vector<std::string>& arguments);

/**
 * @brief Resolve a base output path to the executable file that actually
 *        exists on disk.
 * @param base_path Desired output path, with or without the platform
 *                  executable suffix.
 * @return An empty path if @p base_path is empty; otherwise the canonical
 *         path of whichever of {@p base_path + suffix, @p base_path} exists
 *         first, or @p base_path + suffix if neither exists yet.
 */
std::filesystem::path resolve_executable_output(const std::filesystem::path& base_path);

} // namespace eshkol::platform

#endif // ESHKOL_PLATFORM_RUNTIME_H
