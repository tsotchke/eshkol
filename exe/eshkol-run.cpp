/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>
#include <eshkol/core/runtime.h>
#include <eshkol/core/resource_limits.h>
#include <eshkol/core/execution_profile.h>
#include <eshkol/backend/vm.h>
#include <eshkol/build_config.h>
#include <eshkol/platform_runtime.h>
#include <eshkol/pkg/subprocess.h>

#include <eshkol/llvm_backend.h>
#include "../lib/repl/repl_jit.h"

#include <llvm/Config/llvm-config.h>
#include <llvm/Support/SHA256.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#ifndef _WIN32
#include <getopt.h>
#endif
#include <errno.h>
#include <filesystem>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <set>
#include <map>
#if defined(__APPLE__)
#include <mach-o/dyld.h>  // _dyld_image_count/_dyld_get_image_name — child lib search path
#endif
#include <algorithm>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <array>
#include <chrono>
#include <optional>

static constexpr char eshkol_path_separator =
#ifdef _WIN32
    ';';
#else
    ':';
#endif

static void append_host_runtime_link_args(std::vector<std::string>& link_args) {
#ifdef _WIN32
    std::stringstream stream(ESHKOL_HOST_RUNTIME_LINK_ARGS);
    std::string item;

    while (std::getline(stream, item, ';')) {
        if (item.empty()) {
            continue;
        }

        if (item.size() >= 3 &&
            std::isalpha(static_cast<unsigned char>(item[0])) &&
            item[1] == ':' &&
            item[2] == '/') {
            std::replace(item.begin(), item.end(), '/', '\\');
        }

        link_args.emplace_back(item);
    }

    const auto has_cudadevrt = std::any_of(link_args.begin(), link_args.end(), [](const std::string& arg) {
        return std::filesystem::path(arg).filename() == "cudadevrt.lib";
    });
    if (!has_cudadevrt) {
        for (const auto& arg : link_args) {
            const auto path = std::filesystem::path(arg);
            if (path.filename() == "cudart.lib") {
                const auto candidate = path.parent_path() / "cudadevrt.lib";
                std::error_code ec;
                if (std::filesystem::exists(candidate, ec)) {
                    link_args.emplace_back(candidate.string());
                }
                break;
            }
        }
    }
#else
    for (const auto& runtime_arg : eshkol::platform::host_runtime_link_args()) {
        link_args.emplace_back(runtime_arg);
    }
#endif
}

static void append_space_separated_link_args(const char* raw_args,
                                             std::vector<std::string>& link_args) {
    if (!raw_args || !*raw_args) {
        return;
    }
    std::string normalized(raw_args);
    std::replace(normalized.begin(), normalized.end(), ';', ' ');
    std::stringstream stream(normalized);
    std::string item;
    while (stream >> item) {
#ifdef __APPLE__
        if (item == "-lc++" || item == "-lc++abi") {
            continue;
        }
#endif
        link_args.emplace_back(item);
    }
}

// Noesis bug report #2 (2026-07-04), part (b): a standalone `eshkol-run -r`
// AOT link that races a *concurrent* rebuild of one of the static runtime
// archives (libeshkol-runtime.a / libeshkol-agent-ffi.a — e.g. a `cmake
// --build` running in another terminal against the same build tree) can see
// a partially-written `.a` and fail with plain "undefined symbols" errors
// that look exactly like a missing-symbol bug. We can't intercept or rewrite
// the linker's own stderr — run_subprocess() lets the child inherit the
// parent's stdio directly so live diagnostics/progress are never buffered or
// dropped — but we CAN cheaply check, after a link failure, whether any
// static archive on the link line was modified within the last few seconds.
// That's a strong signal of exactly this race, so we append an explanatory
// note (in addition to, not instead of, the linker's own error text already
// printed by the child) pointing the user at the concurrent-rebuild
// explanation before they go spelunking for a "missing" symbol that in fact
// exists in the finished archive.
static void warn_if_link_archive_recently_modified(const std::vector<std::string>& link_args) {
    constexpr long long kRecentSeconds = 30;
    const auto now_system = std::chrono::system_clock::now();
    for (const auto& arg : link_args) {
        if (arg.size() < 2 || arg.compare(arg.size() - 2, 2, ".a") != 0) {
            continue;
        }
        std::error_code ec;
        if (!std::filesystem::is_regular_file(arg, ec) || ec) {
            continue;
        }
        auto ftime = std::filesystem::last_write_time(arg, ec);
        if (ec) {
            continue;
        }
        // Portable file_time_type -> system_clock::time_point conversion
        // (no std::chrono::clock_cast pre-C++20).
        auto as_system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now() + now_system);
        auto age_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            now_system - as_system_time).count();
        if (age_seconds >= 0 && age_seconds < kRecentSeconds) {
            eshkol_error(
                "note: '%s' was modified %lld second(s) ago. If another build "
                "(e.g. `cmake --build build --target eshkol-run stdlib`) is "
                "rebuilding the Eshkol runtime/agent-FFI libraries "
                "concurrently in this tree, the archive above may have been "
                "linked while only partially written. Wait for that build to "
                "finish, then retry this command.",
                arg.c_str(), static_cast<long long>(age_seconds));
        }
    }
}

#if !defined(_WIN32)
// Run a fixed, trusted command and capture its trimmed stdout. Returns empty on
// any failure. Used only for runtime toolchain probes (xcrun, llvm-config) with
// no user-controlled input, so popen() is safe here.
static std::string captureCommandOutput(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return {};
    }
    std::string output;
    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    int status = pclose(pipe);
    if (status != 0) {
        return {};
    }
    while (!output.empty() &&
           (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
        output.pop_back();
    }
    return output;
}

// Resolve the LLVM library directory at RUNTIME rather than trusting the
// absolute Homebrew Cellar path baked into ESHKOL_HOST_LLVM_LINK_ARGS at build
// time. That baked path (e.g. /opt/homebrew/Cellar/llvm/21.1.7/lib) does not
// exist on a host whose LLVM is a different patch/prefix, so the link fails
// with "search path '...' not found" / "library 'LLVM-NN' not found" and the
// binary can't JIT off-builder (Noesis hit this copying atlas's binary to enki).
//
// We try `llvm-config --libdir` (and the versioned `llvm-config-NN`) from PATH,
// then common prefixes. Returns empty if nothing resolves, in which case the
// caller falls back to the baked path. The result is cached.
static const std::string& resolveLlvmLibDir() {
    static const std::string cached = [] {
        const std::string major = std::to_string(LLVM_VERSION_MAJOR);
        std::vector<std::string> probes = {
            "llvm-config-" + major + " --libdir 2>/dev/null",
            "llvm-config --libdir 2>/dev/null",
        };
        for (const auto& cmd : probes) {
            std::string dir = captureCommandOutput(cmd);
            std::error_code ec;
            if (!dir.empty() && std::filesystem::is_directory(dir, ec)) {
                return dir;
            }
        }
        // Known prefixes as a last resort before the baked path.
        std::vector<std::string> candidates = {
            "/opt/homebrew/opt/llvm/lib",
            "/opt/homebrew/opt/llvm@" + major + "/lib",
            "/usr/local/opt/llvm/lib",
            "/usr/local/opt/llvm@" + major + "/lib",
            "/usr/lib/llvm-" + major + "/lib",
        };
        for (const auto& dir : candidates) {
            std::error_code ec;
            if (std::filesystem::is_directory(dir, ec)) {
                return dir;
            }
        }
        return std::string{};
    }();
    return cached;
}
#endif

static void append_host_llvm_link_args(std::vector<std::string>& link_args) {
#if defined(_WIN32)
    append_space_separated_link_args(ESHKOL_HOST_LLVM_LINK_ARGS, link_args);
#else
    // Prepend a runtime-resolved LLVM -L (and rpath on ELF) so the libdir is
    // searched BEFORE the build-time absolute Cellar path baked into
    // ESHKOL_HOST_LLVM_LINK_ARGS. This makes a binary built on one host link
    // against the LLVM present on a different host (different patch/prefix).
    // The baked args are still spliced afterwards as a fallback.
    const std::string& runtime_libdir = resolveLlvmLibDir();
    if (!runtime_libdir.empty()) {
        link_args.emplace_back("-L" + runtime_libdir);
#  if !defined(__APPLE__)
        // ELF: a -L alone leaves no runtime search path; add an rpath so the
        // resulting binary can find libLLVM.so at run time (mirrors CMake).
        link_args.emplace_back("-Wl,-rpath," + runtime_libdir);
#  endif
    }
    append_space_separated_link_args(ESHKOL_HOST_LLVM_LINK_ARGS, link_args);
#endif
}

// Bounded wait for the native link / cached-compile subprocesses. A hung `ld`
// (e.g. a misconfigured search path on a non-builder host) must never wedge the
// process: on expiry the child is killed and the caller fails fast — to the
// interpreter for the `-r` JIT path, or to a clean link error for AOT. 0 means
// "unbounded"; ESHKOL_LINK_TIMEOUT_SECONDS overrides the default. The default is
// generous because a legitimate link of the 58 MB stdlib can take tens of
// seconds on a cold host.
static unsigned int link_subprocess_timeout_seconds() {
    if (const char* raw = std::getenv("ESHKOL_LINK_TIMEOUT_SECONDS")) {
        char* end = nullptr;
        long parsed = std::strtol(raw, &end, 10);
        if (end && *end == '\0' && parsed >= 0) {
            return static_cast<unsigned int>(parsed);
        }
    }
    return 300; // 5 minutes
}

// Digest of the FULL compilation unit: the entry file plus every source file
// reachable from it via (load "…") / (import "…") / (require module).  Defined
// far below (it needs resolve_module_path); forward-declared at file scope here
// so the anonymous-namespace makeJitRunCacheKey can key on it.  Previously the
// run-cache key hashed only the entry file's bytes, so editing a (load …)ed
// dependency left the key unchanged and `-r` silently re-executed a STALE
// cached binary (ESH-0183).
static std::string transitiveSourceDigest(
    const std::filesystem::path& entry_source,
    const std::vector<char*>& include_paths);

namespace {

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered != "0" && lowered != "false" && lowered != "off" && lowered != "no";
}

bool envFlagDisabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered == "0" || lowered == "false" || lowered == "off" || lowered == "no";
}

void jitCacheTrace(const char* status, const std::string& detail) {
    if (envFlagEnabled("ESHKOL_JIT_CACHE_TRACE")) {
        std::fprintf(stderr, "[jit-cache] %s %s\n", status, detail.c_str());
    }
}

std::string readFileBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void hashUpdate(llvm::SHA256& hash, const std::string& label, const std::string& value) {
    hash.update(llvm::StringRef(label.data(), label.size()));
    hash.update(llvm::StringRef("\0", 1));
    hash.update(llvm::StringRef(value.data(), value.size()));
    hash.update(llvm::StringRef("\0", 1));
}

std::string sha256Hex(llvm::SHA256& hash) {
    std::array<uint8_t, 32> digest = hash.final();
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(digest.size() * 2);
    for (uint8_t byte : digest) {
        out.push_back(hex[(byte >> 4) & 0x0f]);
        out.push_back(hex[byte & 0x0f]);
    }
    return out;
}

std::string fileMetadataFingerprint(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return "missing:" + path.string();
    }
    const auto size = std::filesystem::file_size(path, ec);
    const auto mtime = std::filesystem::last_write_time(path, ec);
    const auto mtime_count =
        static_cast<long long>(mtime.time_since_epoch().count());
    return path.string() + "|" + std::to_string(size) + "|" +
           std::to_string(mtime_count);
}

std::filesystem::path resolveSelfPath(const char* argv0) {
    std::error_code ec;
    if (argv0 && *argv0) {
        std::filesystem::path candidate(argv0);
        if (candidate.is_relative()) {
            candidate = std::filesystem::current_path(ec) / candidate;
        }
        auto canonical = std::filesystem::weakly_canonical(candidate, ec);
        if (!ec && !canonical.empty()) {
            return canonical;
        }
        return candidate;
    }
    return {};
}

std::filesystem::path findBuildArtifact(const std::string& name,
                                        const std::filesystem::path& self_path) {
    auto cwd = std::filesystem::current_path();
    auto exe_dir = self_path.empty() ? std::filesystem::path{} : self_path.parent_path();
    std::vector<std::filesystem::path> candidates;
    if (!exe_dir.empty()) {
        candidates.push_back(exe_dir / name);
        candidates.push_back(exe_dir / "../lib" / name);
        candidates.push_back(exe_dir / "../lib/eshkol" / name);
    }
    candidates.push_back(cwd / name);
    candidates.push_back(cwd / "build" / name);
    candidates.push_back(cwd / "build-verify" / name);
    candidates.push_back(cwd.parent_path() / "build" / name);
    candidates.push_back(cwd.parent_path() / "build-verify" / name);

#ifndef _WIN32
    candidates.emplace_back("/usr/local/lib/eshkol/" + name);
    candidates.emplace_back("/usr/lib/eshkol/" + name);
#endif

    std::error_code ec;
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate, ec)) {
            auto canonical = std::filesystem::weakly_canonical(candidate, ec);
            return ec ? candidate : canonical;
        }
    }
    return {};
}

std::filesystem::path jitCacheRoot() {
    if (const char* explicit_dir = std::getenv("ESHKOL_JIT_CACHE_DIR")) {
        if (*explicit_dir) {
            return std::filesystem::path(explicit_dir);
        }
    }
#ifdef _WIN32
    if (const char* local_app_data = std::getenv("LOCALAPPDATA")) {
        if (*local_app_data) {
            return std::filesystem::path(local_app_data) / "eshkol" / "jit";
        }
    }
#else
    if (const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg_cache_home) {
            return std::filesystem::path(xdg_cache_home) / "eshkol" / "jit";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home) {
            return std::filesystem::path(home) / ".cache" / "eshkol" / "jit";
        }
    }
#endif
    return std::filesystem::temp_directory_path() / "eshkol" / "jit";
}

void pruneJitCache(const std::filesystem::path& root) {
    static constexpr uintmax_t max_cache_bytes = 1024ull * 1024ull * 1024ull;
    static constexpr auto max_age = std::chrono::hours(24 * 30);

    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        return;
    }

    struct Entry {
        std::filesystem::path path;
        uintmax_t size;
        std::filesystem::file_time_type mtime;
    };

    std::vector<Entry> entries;
    uintmax_t total_size = 0;
    const auto now = std::filesystem::file_time_type::clock::now();

    for (const auto& dir_entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) break;
        if (!dir_entry.is_regular_file(ec)) {
            continue;
        }
        const auto path = dir_entry.path();
        const auto size = dir_entry.file_size(ec);
        const auto mtime = dir_entry.last_write_time(ec);
        if (ec) {
            ec.clear();
            continue;
        }
        if (now - mtime > max_age) {
            std::filesystem::remove(path, ec);
            ec.clear();
            continue;
        }
        total_size += size;
        entries.push_back({path, size, mtime});
    }

    if (total_size <= max_cache_bytes) {
        return;
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.mtime < b.mtime;
    });
    for (const auto& entry : entries) {
        if (total_size <= max_cache_bytes) {
            break;
        }
        if (std::filesystem::remove(entry.path, ec)) {
            total_size = entry.size > total_size ? 0 : total_size - entry.size;
        }
        ec.clear();
    }
}

std::string makeJitRunCacheKey(const std::filesystem::path& source_path,
                               const std::filesystem::path& self_path,
                               uint8_t no_stdlib,
                               uint8_t strict_types,
                               uint8_t unsafe_mode,
                               int opt_level,
                               const char* target_triple,
                               const std::vector<char*>& linked_libs,
                               const std::vector<char*>& lib_paths,
                               const std::vector<char*>& include_paths) {
    llvm::SHA256 hash;
    std::error_code ec;
    auto canonical_source = std::filesystem::weakly_canonical(source_path, ec);
    if (ec) {
        canonical_source = source_path;
    }

    hashUpdate(hash, "schema", "eshkol-run-cache-v1");
    hashUpdate(hash, "source-path", canonical_source.string());
    hashUpdate(hash, "source-bytes", readFileBytes(source_path));
    // Fold in every transitively (load …)ed / (require …)d / (import …)ed
    // source so editing a dependency — not just the entry file — invalidates
    // the persistent run-cache (ESH-0183).
    hashUpdate(hash, "transitive-deps",
               transitiveSourceDigest(source_path, include_paths));
    hashUpdate(hash, "eshkol-version", ESHKOL_VER);
    hashUpdate(hash, "llvm-version", LLVM_VERSION_STRING);
    hashUpdate(hash, "self", fileMetadataFingerprint(self_path));
    hashUpdate(hash, "no-stdlib", std::to_string(no_stdlib));
    hashUpdate(hash, "strict-types", std::to_string(strict_types));
    hashUpdate(hash, "unsafe", std::to_string(unsafe_mode));
    hashUpdate(hash, "opt-level", std::to_string(opt_level));
    hashUpdate(hash, "target-triple", target_triple ? target_triple : "");

    const auto stdlib_bc = findBuildArtifact("stdlib.bc", self_path);
    const auto stdlib_o = findBuildArtifact("stdlib.o", self_path);
    hashUpdate(hash, "stdlib.bc", stdlib_bc.empty() ? "missing" : fileMetadataFingerprint(stdlib_bc));
    hashUpdate(hash, "stdlib.o", stdlib_o.empty() ? "missing" : fileMetadataFingerprint(stdlib_o));

    for (char* path : include_paths) hashUpdate(hash, "include-path", path ? path : "");
    for (char* path : lib_paths) hashUpdate(hash, "lib-path", path ? path : "");
    for (char* lib : linked_libs) hashUpdate(hash, "linked-lib", lib ? lib : "");

    return sha256Hex(hash);
}

// `-r` runs a single file through the persistent AOT cache for speed. That is
// only semantically equivalent to true in-process JIT execution when the
// program does NOT need the live JIT at runtime. `eval`/`compile` resolve their
// argument by spinning up a ReplJITContext via the eval bridge, which is only
// linked into eshkol-run itself — never into a cached standalone AOT binary. So
// a program that calls eval/compile must bypass the cache and fall through to
// the in-process ReplJITContext path. Detect those builtins with a whole-word
// scan (false positives merely forfeit the cache; correctness is preserved).
bool sourceRequiresInProcessJit(const std::string& source) {
    auto is_ident_char = [](unsigned char c) -> bool {
        // R7RS/Eshkol identifier characters that can neighbour a builtin name.
        return std::isalnum(c) || std::strchr("-_?!*/+<>=.:$%&~^@", c) != nullptr;
    };
    static const char* kJitBuiltins[] = {"eval", "compile"};
    for (const char* needle : kJitBuiltins) {
        const size_t nlen = std::strlen(needle);
        size_t pos = 0;
        while ((pos = source.find(needle, pos)) != std::string::npos) {
            const bool left_ok =
                (pos == 0) || !is_ident_char((unsigned char)source[pos - 1]);
            const size_t after = pos + nlen;
            const bool right_ok =
                (after >= source.size()) || !is_ident_char((unsigned char)source[after]);
            if (left_ok && right_ok) {
                return true;
            }
            pos = after;
        }
    }
    return false;
}

// A persistent run-cache AOT binary is a normal dynamically-linked executable:
// it needs its whole shared-library closure (libLLVM, libstdc++/libc++, zlib,
// libffi, …) at exec time. On systems where those libraries live OUTSIDE the
// default loader search path — Nix (/nix/store), Homebrew, custom prefixes — the
// child dies with "libX.so.NN: cannot open shared object file". eshkol-run itself
// has ALREADY resolved that exact closure (it links the same way), so the robust,
// platform-independent fix is to hand the child the directories the PARENT
// actually loaded its libraries from, rather than guessing per-library.
//
// We seed from the build-time LLVM -L dirs, then add every directory the running
// process has a shared library mapped from (/proc/self/maps on Linux, the dyld
// image list on macOS). Prepended to the platform loader var for any child we
// spawn (the run-cache binary).
static void ensureChildLibrarySearchPath() {
    static bool done = false;
    if (done) return;
    done = true;

    std::vector<std::string> order;          // preserve insertion order
    std::set<std::string> seen;
    auto add_dir = [&](std::string d) {
        if (d.empty()) return;
        while (d.size() > 1 && d.back() == '/') d.pop_back();
        if (seen.insert(d).second) order.push_back(d);
    };

    // 1) LLVM lib dirs from the build-time link args.
    {
        const std::string link_args = ESHKOL_HOST_LLVM_LINK_ARGS;
        size_t pos = 0;
        while (pos <= link_args.size()) {
            size_t semi = link_args.find(';', pos);
            std::string tok = link_args.substr(
                pos, semi == std::string::npos ? std::string::npos : semi - pos);
            if (tok.rfind("-L", 0) == 0 && tok.size() > 2) add_dir(tok.substr(2));
            if (semi == std::string::npos) break;
            pos = semi + 1;
        }
    }

    // 2) Every directory the parent process actually loaded a library from.
#if defined(__linux__)
    {
        std::ifstream maps("/proc/self/maps");
        std::string line;
        while (std::getline(maps, line)) {
            size_t slash = line.find('/');
            if (slash == std::string::npos) continue;
            std::string path = line.substr(slash);
            if (path.find(".so") == std::string::npos) continue;
            size_t last = path.find_last_of('/');
            if (last != std::string::npos && last > 0) add_dir(path.substr(0, last));
        }
    }
#elif defined(__APPLE__)
    {
        uint32_t n = _dyld_image_count();
        for (uint32_t i = 0; i < n; i++) {
            const char* nm = _dyld_get_image_name(i);
            if (!nm) continue;
            std::string p(nm);
            if (p.find(".dylib") == std::string::npos) continue;
            size_t last = p.find_last_of('/');
            if (last != std::string::npos && last > 0) add_dir(p.substr(0, last));
        }
    }
#endif

    if (order.empty()) return;

#if defined(__APPLE__)
    const char* var = "DYLD_LIBRARY_PATH";
    const char sep = ':';
#elif defined(_WIN32)
    const char* var = "PATH";
    const char sep = ';';
#else
    const char* var = "LD_LIBRARY_PATH";
    const char sep = ':';
#endif
    std::string combined;
    for (const auto& d : order) {
        if (!combined.empty()) combined += sep;
        combined += d;
    }
    if (const char* existing = std::getenv(var)) {
        if (existing[0] != '\0') { combined += sep; combined += existing; }
    }
#if defined(_WIN32)
    _putenv_s(var, combined.c_str());
#else
    setenv(var, combined.c_str(), 1);
#endif
}

std::optional<int> tryRunFromPersistentJitCache(const char* argv0,
                                                const std::string& filepath,
                                                uint8_t no_stdlib,
                                                uint8_t strict_types,
                                                uint8_t unsafe_mode,
                                                int opt_level,
                                                const char* target_triple,
                                                const std::vector<char*>& linked_libs,
                                                const std::vector<char*>& lib_paths,
                                                const std::vector<char*>& include_paths) {
    if (envFlagDisabled("ESHKOL_JIT_CACHE")) {
        jitCacheTrace("bypass", "disabled");
        return std::nullopt;
    }
    // Ensure any cached/freshly-built run binary we exec can find libLLVM.
    ensureChildLibrarySearchPath();

    std::filesystem::path source_path(filepath);
    std::error_code ec;
    if (!std::filesystem::exists(source_path, ec) || !std::filesystem::is_regular_file(source_path, ec)) {
        return std::nullopt;
    }

    // Programs that use eval/compile need the in-process JIT bridge, which a
    // cached standalone AOT binary does not have. Bypass the cache so `-r`
    // honours its "interpret without compiling" contract for them.
    if (sourceRequiresInProcessJit(readFileBytes(source_path))) {
        jitCacheTrace("bypass", "needs-in-process-jit");
        return std::nullopt;
    }

    auto self_path = resolveSelfPath(argv0);
    if (self_path.empty() || !std::filesystem::exists(self_path, ec)) {
        jitCacheTrace("bypass", "self-not-found");
        return std::nullopt;
    }

    const auto cache_dir = jitCacheRoot();
    std::filesystem::create_directories(cache_dir, ec);
    if (ec) {
        jitCacheTrace("bypass", "mkdir-failed");
        return std::nullopt;
    }

    const std::string key = makeJitRunCacheKey(source_path, self_path, no_stdlib,
                                              strict_types, unsafe_mode, opt_level,
                                              target_triple, linked_libs, lib_paths,
                                              include_paths);
    const auto cached_binary =
        cache_dir / ("run-" + key + std::string(ESHKOL_HOST_EXECUTABLE_SUFFIX));

    if (std::filesystem::exists(cached_binary, ec)) {
        jitCacheTrace("hit", key);
        return eshkol::pkg::run_subprocess({cached_binary.string()});
    }

    jitCacheTrace("miss", key);
    pruneJitCache(cache_dir);

    const auto stamp = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto temp_binary = cache_dir / (cached_binary.filename().string() + ".tmp-" + stamp);

    std::vector<std::string> compile_args;
    compile_args.push_back(self_path.string());
    if (no_stdlib) compile_args.emplace_back("-n");
    if (strict_types) compile_args.emplace_back("--strict-types");
    if (unsafe_mode) compile_args.emplace_back("--unsafe");
    if (opt_level != 0) {
        compile_args.emplace_back("-O");
        compile_args.push_back(std::to_string(opt_level));
    }
    if (target_triple && *target_triple) {
        compile_args.emplace_back("--target");
        compile_args.emplace_back(target_triple);
    }
    for (char* path : include_paths) {
        compile_args.emplace_back("-I");
        compile_args.emplace_back(path ? path : "");
    }
    for (char* path : lib_paths) {
        compile_args.emplace_back("-L");
        compile_args.emplace_back(path ? path : "");
    }
    for (char* lib : linked_libs) {
        compile_args.emplace_back("-l");
        compile_args.emplace_back(lib ? lib : "");
    }
    compile_args.push_back(source_path.string());
    compile_args.emplace_back("-o");
    compile_args.push_back(temp_binary.string());

    // Bound the cached-binary build. The child runs the full AOT compile +
    // native link; if its linker hangs (the child bounds its own link too, but
    // belt-and-suspenders here covers any other stall), kill it and return
    // nullopt so the caller fails fast to the interpreter rather than wedging
    // the whole `-r` invocation. This is the architectural guarantee Noesis
    // asked for: a JIT-link problem must never hang, only fall back.
    int compile_status = eshkol::pkg::run_subprocess(
        compile_args, nullptr, link_subprocess_timeout_seconds());
    if (compile_status == eshkol::pkg::SUBPROCESS_TIMEOUT) {
        std::filesystem::remove(temp_binary, ec);
        jitCacheTrace("compile-timeout", std::to_string(link_subprocess_timeout_seconds()));
        return std::nullopt;
    }
    if (compile_status != 0) {
        std::filesystem::remove(temp_binary, ec);
        jitCacheTrace("compile-failed", std::to_string(compile_status));
        return std::nullopt;
    }

    std::filesystem::rename(temp_binary, cached_binary, ec);
    if (ec) {
        std::filesystem::remove(cached_binary, ec);
        ec.clear();
        std::filesystem::rename(temp_binary, cached_binary, ec);
    }
    if (ec) {
        std::filesystem::remove(temp_binary, ec);
        jitCacheTrace("store-failed", key);
        return std::nullopt;
    }

    jitCacheTrace("store", key);
    return eshkol::pkg::run_subprocess({cached_binary.string()});
}

} // namespace

#ifdef _WIN32
enum {
    no_argument = 0,
    required_argument = 1,
    optional_argument = 2
};

struct option {
    const char* name;
    int has_arg;
    int* flag;
    int val;
};

char* optarg = nullptr;
int optind = 1;
int opterr = 1;
int optopt = 0;

static const char* g_short_option_cursor = nullptr;
static int g_non_option_count = 0;

static int getopt_long(int argc, char** argv, const char* shortopts,
                       const struct option* longopts, int* longindex) {
    optarg = nullptr;

    if (optind <= 1 && !g_short_option_cursor) {
        g_non_option_count = 0;
    }

    if (g_short_option_cursor && *g_short_option_cursor == '\0') {
        g_short_option_cursor = nullptr;
    }

    while (!g_short_option_cursor) {
        if (optind >= argc - g_non_option_count) {
            return -1;
        }

        const char* current = argv[optind];
        if (std::strcmp(current, "--") == 0) {
            ++optind;
            return -1;
        }

        if (current[0] != '-' || current[1] == '\0') {
            char* positional = argv[optind];
            for (int i = optind; i < argc - 1; ++i) {
                argv[i] = argv[i + 1];
            }
            argv[argc - 1] = positional;
            ++g_non_option_count;
            continue;
        }

        if (current[1] == '-') {
            const char* option_name = current + 2;
            const char* option_value = std::strchr(option_name, '=');
            std::string name = option_value
                ? std::string(option_name, option_value - option_name)
                : std::string(option_name);

            for (int i = 0; longopts[i].name; ++i) {
                if (name == longopts[i].name) {
                    if (longindex) {
                        *longindex = i;
                    }

                    if (longopts[i].has_arg == required_argument) {
                        if (option_value) {
                            optarg = const_cast<char*>(option_value + 1);
                        } else if (optind + 1 < argc) {
                            optarg = argv[++optind];
                        } else {
                            if (opterr) {
                                std::fprintf(stderr, "Option '--%s' requires an argument\n", name.c_str());
                            }
                            ++optind;
                            return '?';
                        }
                    } else if (option_value) {
                        if (opterr) {
                            std::fprintf(stderr, "Option '--%s' does not take an argument\n", name.c_str());
                        }
                        ++optind;
                        return '?';
                    }

                    ++optind;
                    if (longopts[i].flag) {
                        *longopts[i].flag = longopts[i].val;
                        return 0;
                    }
                    return longopts[i].val;
                }
            }

            if (opterr) {
                std::fprintf(stderr, "Unknown option: --%s\n", name.c_str());
            }
            ++optind;
            return '?';
        }

        g_short_option_cursor = current + 1;
    }

    const char opt = *g_short_option_cursor++;
    const char* spec = std::strchr(shortopts, opt);
    if (!spec) {
        optopt = opt;
        if (*g_short_option_cursor == '\0') {
            g_short_option_cursor = nullptr;
            ++optind;
        }
        if (opterr) {
            std::fprintf(stderr, "Unknown option: -%c\n", opt);
        }
        return '?';
    }

    if (spec[1] == ':') {
        if (*g_short_option_cursor != '\0') {
            optarg = const_cast<char*>(g_short_option_cursor);
        } else if (optind + 1 < argc) {
            optarg = argv[++optind];
        } else {
            optopt = opt;
            if (opterr) {
                std::fprintf(stderr, "Option '-%c' requires an argument\n", opt);
            }
            g_short_option_cursor = nullptr;
            ++optind;
            return '?';
        }

        g_short_option_cursor = nullptr;
        ++optind;
        return opt;
    }

    if (*g_short_option_cursor == '\0') {
        g_short_option_cursor = nullptr;
        ++optind;
    }

    return opt;
}
#endif

static struct option long_options[] = {
    {"help", no_argument, nullptr, 'h'},
    {"debug", no_argument, nullptr, 'd'},
    {"dump-ast", no_argument, nullptr, 'a'},
    {"dump-ir", no_argument, nullptr, 'i'},
    {"output", required_argument, nullptr, 'o'},
    {"compile-only", no_argument, nullptr, 'c'},
    {"emit-object", no_argument, nullptr, 259},
    {"emit-depfile", required_argument, nullptr, 265},
    {"shared-lib", no_argument, nullptr, 's'},
    {"wasm", no_argument, nullptr, 'w'},
    {"lib", required_argument, nullptr, 'l'},
    {"lib-path", required_argument, nullptr, 'L'},
    {"no-stdlib", no_argument, nullptr, 'n'},
    {"eval", required_argument, nullptr, 'e'},
    {"run", no_argument, nullptr, 'r'},
    {"strict-types", no_argument, nullptr, 256},
    {"unsafe", no_argument, nullptr, 257},
    {"version", no_argument, nullptr, 258},
    {"debug-info", no_argument, nullptr, 'g'},
    {"optimize", required_argument, nullptr, 'O'},
    {"emit-eskb", required_argument, nullptr, 'B'},
    {"profile", required_argument, nullptr, 260},
    {"target", required_argument, nullptr, 261},
    {"require-vm-entry", required_argument, nullptr, 262},
    {"require-vm-entry-zero-arg", required_argument, nullptr, 263},
    {"features", no_argument, nullptr, 264},
    {0, 0, 0, 0}
};

// Set to track imported files (prevent circular imports)
static std::set<std::string> imported_files;

// Set to track pre-compiled modules (when linking with .o files like stdlib.o)
// When a module is in this set, process_requires() will skip loading it
static std::set<std::string> precompiled_modules;

// ===== MODULE DEPENDENCY RESOLVER =====
// Provides cycle detection and topological sorting for module loading

class ModuleDependencyResolver {
public:
    // Module states for cycle detection (DFS coloring)
    enum class ModuleState { UNVISITED, VISITING, VISITED };

    // Add a module and its dependencies
    void addModule(const std::string& module_name, const std::vector<std::string>& dependencies) {
        if (dependency_graph.find(module_name) == dependency_graph.end()) {
            dependency_graph[module_name] = dependencies;
            module_states[module_name] = ModuleState::UNVISITED;
        }
    }

    // Check for cycles and return topologically sorted module order
    // Returns empty vector if cycle detected
    std::vector<std::string> resolve() {
        std::vector<std::string> sorted_modules;
        std::vector<std::string> cycle_path;

        for (const auto& [module_name, _] : dependency_graph) {
            if (module_states[module_name] == ModuleState::UNVISITED) {
                if (!dfs(module_name, sorted_modules, cycle_path)) {
                    // Cycle detected
                    std::string cycle_str = formatCycle(cycle_path);
                    eshkol_error("Circular dependency detected: %s", cycle_str.c_str());
                    return {};
                }
            }
        }

        // Reverse for topological order (dependencies first)
        std::reverse(sorted_modules.begin(), sorted_modules.end());
        return sorted_modules;
    }

    // Clear the resolver state
    void clear() {
        dependency_graph.clear();
        module_states.clear();
    }

private:
    std::map<std::string, std::vector<std::string>> dependency_graph;
    std::map<std::string, ModuleState> module_states;

    // DFS with cycle detection
    bool dfs(const std::string& module, std::vector<std::string>& sorted,
             std::vector<std::string>& path) {
        module_states[module] = ModuleState::VISITING;
        path.push_back(module);

        auto it = dependency_graph.find(module);
        if (it != dependency_graph.end()) {
            for (const auto& dep : it->second) {
                // Ensure dependency is in the graph
                if (dependency_graph.find(dep) == dependency_graph.end()) {
                    dependency_graph[dep] = {};
                    module_states[dep] = ModuleState::UNVISITED;
                }

                if (module_states[dep] == ModuleState::VISITING) {
                    // Cycle detected - add the closing node
                    path.push_back(dep);
                    return false;
                }

                if (module_states[dep] == ModuleState::UNVISITED) {
                    if (!dfs(dep, sorted, path)) {
                        return false;
                    }
                }
            }
        }

        module_states[module] = ModuleState::VISITED;
        sorted.push_back(module);
        path.pop_back();
        return true;
    }

    std::string formatCycle(const std::vector<std::string>& path) {
        if (path.empty()) return "(empty)";

        std::string result;
        // Find where cycle starts (last element appears earlier)
        size_t cycle_start = 0;
        for (size_t i = 0; i < path.size() - 1; i++) {
            if (path[i] == path.back()) {
                cycle_start = i;
                break;
            }
        }

        for (size_t i = cycle_start; i < path.size(); i++) {
            if (i > cycle_start) result += " -> ";
            result += path[i];
        }
        return result;
    }
};

// Global module resolver instance
static ModuleDependencyResolver g_module_resolver;

// Set of modules currently being loaded (for cycle detection during recursive require)
static std::set<std::string> g_loading_modules;

// ===== END MODULE DEPENDENCY RESOLVER =====

// ===== MODULE SYMBOL TABLE =====
// Tracks symbol visibility (exports) per module

class ModuleSymbolTable {
public:
    // Register a module's exported symbols
    void registerModuleExports(const std::string& module_name, const std::set<std::string>& exports) {
        module_exports[module_name] = exports;
    }

    // Check if a symbol is exported by a module
    bool isExported(const std::string& module_name, const std::string& symbol) const {
        auto it = module_exports.find(module_name);
        if (it == module_exports.end()) {
            // No exports registered = everything is visible (backward compatibility)
            return true;
        }
        if (it->second.empty()) {
            // Empty export list = nothing explicitly provided = all visible
            return true;
        }
        return it->second.count(symbol) > 0;
    }

    std::set<std::string> exportsFor(const std::string& module_name) const {
        auto it = module_exports.find(module_name);
        if (it == module_exports.end()) {
            return {};
        }
        return it->second;
    }

    // Get the private (mangled) name for a symbol
    static std::string getPrivateName(const std::string& module_name, const std::string& symbol) {
        // Convert module name to safe identifier: test.modules.mod_a -> __test_modules_mod_a__
        std::string mangled = "__";
        for (char c : module_name) {
            mangled += (c == '.') ? '_' : c;
        }
        mangled += "__";
        mangled += symbol;
        return mangled;
    }

    // Clear all tracked modules
    void clear() {
        module_exports.clear();
    }

    // Debug: print all registered exports
    void dump() const {
        for (const auto& [module, exports] : module_exports) {
            std::string exp_str;
            for (const auto& e : exports) {
                if (!exp_str.empty()) exp_str += ", ";
                exp_str += e;
            }
            eshkol_debug("Module '%s' exports: [%s]", module.c_str(), exp_str.c_str());
        }
    }

private:
    std::map<std::string, std::set<std::string>> module_exports;
};

// Global symbol table instance
static ModuleSymbolTable g_symbol_table;

// ===== END MODULE SYMBOL TABLE =====

// ===== OWNERSHIP ANALYSIS =====
// Compile-time tracking of owned values for OALR memory management

class OwnershipAnalyzer {
public:
    // Ownership state for a variable
    enum class State {
        UNOWNED,    // Normal value, no ownership tracking
        OWNED,      // Owned value, must be consumed before scope exit
        MOVED,      // Has been moved, cannot be used
        BORROWED    // Currently borrowed by a borrow expression
    };

    struct VariableInfo {
        State state;
        std::string defined_at;  // Location info for error messages
        bool is_owned_binding;   // Was bound with (owned ...)
    };

    // Ownership scope (lexical block)
    struct Scope {
        std::map<std::string, VariableInfo> variables;
        std::string name;  // For debugging
        std::set<std::string> borrowed_vars;  // Variables currently borrowed in this scope
    };

    OwnershipAnalyzer() : has_errors_(false) {}

    // Run analysis on all ASTs
    bool analyze(const std::vector<eshkol_ast_t>& asts) {
        has_errors_ = false;
        errors_.clear();
        scope_stack_.clear();

        // Push global scope
        pushScope("global");

        for (const auto& ast : asts) {
            analyzeAST(&ast);
        }

        // Check global scope exit
        checkScopeExit();
        popScope();

        return !has_errors_;
    }

    // Get error messages
    const std::vector<std::string>& getErrors() const { return errors_; }
    bool hasErrors() const { return has_errors_; }

    // Print all errors
    void printErrors() const {
        for (const auto& err : errors_) {
            eshkol_error("%s", err.c_str());
        }
    }

private:
    std::vector<Scope> scope_stack_;
    std::vector<std::string> errors_;
    bool has_errors_;

    void pushScope(const std::string& name = "") {
        scope_stack_.push_back(Scope{{}, name, {}});
    }

    void popScope() {
        if (!scope_stack_.empty()) {
            scope_stack_.pop_back();
        }
    }

    Scope& currentScope() {
        return scope_stack_.back();
    }

    // Look up variable in all scopes
    VariableInfo* lookupVariable(const std::string& name) {
        for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
            auto var_it = it->variables.find(name);
            if (var_it != it->variables.end()) {
                return &var_it->second;
            }
        }
        return nullptr;
    }

    // Check if variable is currently borrowed anywhere
    bool isBorrowed(const std::string& name) {
        for (const auto& scope : scope_stack_) {
            if (scope.borrowed_vars.count(name) > 0) {
                return true;
            }
        }
        return false;
    }

    void reportError(const std::string& msg) {
        errors_.push_back(msg);
        has_errors_ = true;
    }

    // Check scope exit for unconsumed owned values
    void checkScopeExit() {
        if (scope_stack_.empty()) return;

        const auto& scope = currentScope();

        // Global scope owned values don't require consumption (they live until program exit)
        if (scope.name == "global") return;

        for (const auto& [name, info] : scope.variables) {
            if (info.state == State::OWNED && info.is_owned_binding) {
                reportError("Owned value '" + name + "' not consumed before scope exit" +
                           (info.defined_at.empty() ? "" : " (defined at " + info.defined_at + ")"));
            }
        }
    }

    // Get variable name from AST
    std::string getVarName(const eshkol_ast_t* ast) {
        if (ast && ast->type == ESHKOL_VAR && ast->variable.id) {
            return ast->variable.id;
        }
        return "";
    }

    void analyzeAST(const eshkol_ast_t* ast) {
        if (!ast) return;

        switch (ast->type) {
            case ESHKOL_VAR: {
                // Variable use - check if it's been moved
                std::string name = getVarName(ast);
                if (!name.empty()) {
                    VariableInfo* info = lookupVariable(name);
                    if (info && info->state == State::MOVED) {
                        reportError("Use of moved value '" + name + "'");
                    }
                }
                break;
            }

            case ESHKOL_OP:
                analyzeOperation(&ast->operation);
                break;

            case ESHKOL_CONS:
                analyzeAST(ast->cons_cell.car);
                analyzeAST(ast->cons_cell.cdr);
                break;

            default:
                // Literals etc - nothing to analyze
                break;
        }
    }

    void analyzeOperation(const eshkol_operations_t* op) {
        if (!op) return;

        switch (op->op) {
            case ESHKOL_DEFINE_OP: {
                // Track defined variable
                std::string name = op->define_op.name ? op->define_op.name : "";
                if (!name.empty()) {
                    // Check if the value transfers ownership (owned or move)
                    bool is_owned = transfersOwnership(op->define_op.value);
                    currentScope().variables[name] = {
                        is_owned ? State::OWNED : State::UNOWNED,
                        name,  // Use variable name as location identifier
                        is_owned
                    };
                    // Analyze the value (this handles the move marking)
                    analyzeAST(op->define_op.value);
                }
                break;
            }

            case ESHKOL_OWNED_OP: {
                // (owned expr) - the result should be tracked as owned
                // The actual tracking happens when bound to a variable
                analyzeAST(op->owned_op.value);
                break;
            }

            case ESHKOL_MOVE_OP: {
                // (move var) - transfers ownership, marks var as moved
                std::string var_name = getVarName(op->move_op.value);
                if (!var_name.empty()) {
                    VariableInfo* info = lookupVariable(var_name);
                    if (info) {
                        if (info->state == State::MOVED) {
                            reportError("Double move of '" + var_name + "' - value already moved");
                        } else if (isBorrowed(var_name)) {
                            reportError("Cannot move '" + var_name + "' while it is borrowed");
                        } else {
                            info->state = State::MOVED;
                        }
                    }
                } else {
                    // Moving a non-variable expression - just analyze it
                    analyzeAST(op->move_op.value);
                }
                break;
            }

            case ESHKOL_BORROW_OP: {
                // (borrow var body...) - marks var as borrowed during body
                std::string var_name = getVarName(op->borrow_op.value);
                if (!var_name.empty()) {
                    VariableInfo* info = lookupVariable(var_name);
                    if (info && info->state == State::MOVED) {
                        reportError("Cannot borrow moved value '" + var_name + "'");
                    } else {
                        // Mark as borrowed for this scope
                        currentScope().borrowed_vars.insert(var_name);
                    }
                }
                // Analyze body
                for (uint64_t i = 0; i < op->borrow_op.num_body_exprs; i++) {
                    analyzeAST(&op->borrow_op.body[i]);
                }
                // Unborrow at end
                if (!var_name.empty()) {
                    currentScope().borrowed_vars.erase(var_name);
                }
                break;
            }

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                pushScope("let");
                // Analyze bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        bool is_owned = transfersOwnership(binding->cons_cell.cdr);
                        if (!var_name.empty()) {
                            currentScope().variables[var_name] = {
                                is_owned ? State::OWNED : State::UNOWNED,
                                "",
                                is_owned
                            };
                        }
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }
                // Analyze body
                analyzeAST(op->let_op.body);
                // Check scope exit
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_LETREC_OP: {
                pushScope("letrec");
                // First pass: register all bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        bool is_owned = transfersOwnership(binding->cons_cell.cdr);
                        if (!var_name.empty()) {
                            currentScope().variables[var_name] = {
                                is_owned ? State::OWNED : State::UNOWNED,
                                "",
                                is_owned
                            };
                        }
                    }
                }
                // Second pass: analyze values
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }
                // Analyze body
                analyzeAST(op->let_op.body);
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_LAMBDA_OP: {
                pushScope("lambda");
                // Add parameters to scope
                for (uint64_t i = 0; i < op->lambda_op.num_params; i++) {
                    std::string param_name = getVarName(&op->lambda_op.parameters[i]);
                    if (!param_name.empty()) {
                        currentScope().variables[param_name] = {State::UNOWNED, "", false};
                    }
                }
                // Analyze body
                analyzeAST(op->lambda_op.body);
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_CALL_OP: {
                // Analyze function and arguments
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP: {
                // These use call_op structure
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_SEQUENCE_OP: {
                for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                    analyzeAST(&op->sequence_op.expressions[i]);
                }
                break;
            }

            case ESHKOL_WITH_REGION_OP: {
                pushScope("region");
                for (uint64_t i = 0; i < op->with_region_op.num_body_exprs; i++) {
                    analyzeAST(&op->with_region_op.body[i]);
                }
                checkScopeExit();
                popScope();
                break;
            }

            case ESHKOL_SHARED_OP:
                analyzeAST(op->shared_op.value);
                break;

            case ESHKOL_WEAK_REF_OP:
                analyzeAST(op->weak_ref_op.value);
                break;

            case ESHKOL_SET_OP:
                analyzeAST(op->set_op.value);
                break;

            case ESHKOL_QUOTE_OP:
                // Quoted data - no ownership analysis needed
                break;

            default:
                // Other ops - nothing special to do
                break;
        }
    }

    // Check if expression is (owned ...)
    bool isOwnedExpr(const eshkol_ast_t* ast) {
        if (!ast) return false;
        if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_OWNED_OP) {
            return true;
        }
        return false;
    }

    // Check if expression is (move ...)
    bool isMoveExpr(const eshkol_ast_t* ast) {
        if (!ast) return false;
        if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_MOVE_OP) {
            return true;
        }
        return false;
    }

    // Check if expression transfers ownership (owned or move)
    bool transfersOwnership(const eshkol_ast_t* ast) {
        return isOwnedExpr(ast) || isMoveExpr(ast);
    }
};

// Global ownership analyzer
static OwnershipAnalyzer g_ownership_analyzer;

// ===== END OWNERSHIP ANALYSIS =====

// ===== ESCAPE ANALYSIS =====
// Compile-time tracking of value flow for allocation decisions

class EscapeAnalyzer {
public:
    // Escape classification - determines allocation strategy
    enum class EscapeKind {
        NO_ESCAPE,      // Value stays in scope → Stack allocation
        RETURN_ESCAPE,  // Value returned from function → Caller's region
        CLOSURE_ESCAPE, // Value captured by closure → Shared (ref-counted)
        GLOBAL_ESCAPE   // Value stored in global/mutable → Shared (ref-counted)
    };

    // Allocation strategy based on escape analysis
    enum class AllocationStrategy {
        STACK,          // Fast bump-pointer, freed on scope exit
        REGION,         // Arena allocation, freed with region
        SHARED          // Reference-counted, freed when count hits zero
    };

    // Node in the escape graph representing a value
    struct EscapeNode {
        std::string name;           // Variable/value name (for debugging)
        EscapeKind escape_kind;     // Current escape classification
        AllocationStrategy strategy;// Determined allocation strategy
        std::set<std::string> flows_to;  // Values this flows into
        std::set<std::string> flows_from;// Values that flow into this
        bool is_return_value;       // Is this a return value?
        bool is_closure_captured;   // Captured by a closure?
        bool is_globally_stored;    // Stored in global/mutable location?
        int scope_depth;            // Lexical scope depth
    };

    // Scope for tracking values
    struct EscapeScope {
        std::string name;
        int depth;
        std::set<std::string> local_values;  // Values defined in this scope
        std::string return_target;           // What variable receives our return?
    };

    EscapeAnalyzer() : current_depth_(0), in_lambda_(false) {}

    // Run escape analysis on all ASTs
    bool analyze(const std::vector<eshkol_ast_t>& asts) {
        escape_graph_.clear();
        scope_stack_.clear();
        current_depth_ = 0;
        in_lambda_ = false;

        // Push global scope
        pushScope("global");

        for (const auto& ast : asts) {
            analyzeAST(&ast);
        }

        popScope();

        // Compute final escape classifications
        computeEscapeKinds();

        // Determine allocation strategies
        determineAllocationStrategies();

        return true;  // Escape analysis is informational, doesn't fail
    }

    // Get escape info for a variable
    const EscapeNode* getEscapeInfo(const std::string& name) const {
        auto it = escape_graph_.find(name);
        return (it != escape_graph_.end()) ? &it->second : nullptr;
    }

    // Get allocation strategy for a variable
    AllocationStrategy getAllocationStrategy(const std::string& name) const {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            return it->second.strategy;
        }
        return AllocationStrategy::STACK;  // Default to stack
    }

    // Print escape analysis results (for debugging)
    void printAnalysis() const {
        eshkol_debug("=== Escape Analysis Results ===");
        for (const auto& [name, node] : escape_graph_) {
            const char* escape_str = "unknown";
            switch (node.escape_kind) {
                case EscapeKind::NO_ESCAPE: escape_str = "no-escape"; break;
                case EscapeKind::RETURN_ESCAPE: escape_str = "return"; break;
                case EscapeKind::CLOSURE_ESCAPE: escape_str = "closure"; break;
                case EscapeKind::GLOBAL_ESCAPE: escape_str = "global"; break;
            }
            const char* alloc_str = "unknown";
            switch (node.strategy) {
                case AllocationStrategy::STACK: alloc_str = "stack"; break;
                case AllocationStrategy::REGION: alloc_str = "region"; break;
                case AllocationStrategy::SHARED: alloc_str = "shared"; break;
            }
            eshkol_debug("  %s: escape=%s, alloc=%s",
                        name.c_str(), escape_str, alloc_str);
        }
    }

private:
    std::map<std::string, EscapeNode> escape_graph_;
    std::vector<EscapeScope> scope_stack_;
    int current_depth_;
    bool in_lambda_;
    std::set<std::string> captured_by_current_lambda_;

    void pushScope(const std::string& name) {
        scope_stack_.push_back({name, current_depth_, {}, ""});
        current_depth_++;
    }

    void popScope() {
        if (!scope_stack_.empty()) {
            scope_stack_.pop_back();
            current_depth_--;
        }
    }

    EscapeScope& currentScope() {
        return scope_stack_.back();
    }

    // Register a new value in the escape graph
    void registerValue(const std::string& name, bool is_return = false) {
        if (escape_graph_.find(name) == escape_graph_.end()) {
            EscapeNode node;
            node.name = name;
            node.escape_kind = EscapeKind::NO_ESCAPE;
            node.strategy = AllocationStrategy::STACK;
            node.is_return_value = is_return;
            node.is_closure_captured = false;
            node.is_globally_stored = false;
            node.scope_depth = current_depth_;
            escape_graph_[name] = node;
        }
        currentScope().local_values.insert(name);
    }

    // Record that value 'from' flows into value 'to'
    void recordFlow(const std::string& from, const std::string& to) {
        if (escape_graph_.find(from) != escape_graph_.end()) {
            escape_graph_[from].flows_to.insert(to);
        }
        if (escape_graph_.find(to) != escape_graph_.end()) {
            escape_graph_[to].flows_from.insert(from);
        }
    }

    // Mark a value as captured by a closure
    void markClosureCaptured(const std::string& name) {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            it->second.is_closure_captured = true;
        }
    }

    // Mark a value as stored globally
    void markGloballyStored(const std::string& name) {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            it->second.is_globally_stored = true;
        }
    }

    // Mark a value as being returned
    void markAsReturn(const std::string& name) {
        auto it = escape_graph_.find(name);
        if (it != escape_graph_.end()) {
            it->second.is_return_value = true;
        }
    }

    // Get variable name from AST
    std::string getVarName(const eshkol_ast_t* ast) {
        if (ast && ast->type == ESHKOL_VAR && ast->variable.id) {
            return ast->variable.id;
        }
        return "";
    }

    // Generate unique name for anonymous values
    int anon_counter_ = 0;
    std::string genAnonName() {
        return "$anon" + std::to_string(anon_counter_++);
    }

    void analyzeAST(const eshkol_ast_t* ast) {
        if (!ast) return;

        switch (ast->type) {
            case ESHKOL_VAR: {
                std::string name = getVarName(ast);
                // Check if this variable is from an outer scope (closure capture)
                if (in_lambda_ && !name.empty()) {
                    // Check if it's defined in an outer scope
                    for (int i = scope_stack_.size() - 2; i >= 0; i--) {
                        if (scope_stack_[i].local_values.count(name) > 0) {
                            // This is a capture!
                            markClosureCaptured(name);
                            captured_by_current_lambda_.insert(name);
                            break;
                        }
                    }
                }
                break;
            }

            case ESHKOL_OP:
                analyzeOperation(&ast->operation);
                break;

            case ESHKOL_CONS:
                analyzeAST(ast->cons_cell.car);
                analyzeAST(ast->cons_cell.cdr);
                break;

            default:
                break;
        }
    }

    void analyzeOperation(const eshkol_operations_t* op) {
        if (!op) return;

        switch (op->op) {
            case ESHKOL_DEFINE_OP: {
                std::string name = op->define_op.name ? op->define_op.name : "";
                if (!name.empty()) {
                    registerValue(name);

                    // If at global scope, mark as globally stored
                    if (current_depth_ == 1) {  // depth 1 is global scope
                        markGloballyStored(name);
                    }

                    // Track flow from value to variable
                    std::string value_name = getVarName(op->define_op.value);
                    if (!value_name.empty()) {
                        recordFlow(value_name, name);
                    }

                    analyzeAST(op->define_op.value);
                }
                break;
            }

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                pushScope("let");

                // Analyze bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        if (!var_name.empty()) {
                            registerValue(var_name);

                            std::string value_name = getVarName(binding->cons_cell.cdr);
                            if (!value_name.empty()) {
                                recordFlow(value_name, var_name);
                            }
                        }
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }

                // Analyze body - the result escapes if this is the return value
                analyzeAST(op->let_op.body);

                // Check if body returns a local value (return escape)
                std::string body_name = getVarName(op->let_op.body);
                if (!body_name.empty() && currentScope().local_values.count(body_name) > 0) {
                    // This local value is returned from the let
                    markAsReturn(body_name);
                }

                popScope();
                break;
            }

            case ESHKOL_LETREC_OP: {
                pushScope("letrec");

                // First pass: register all bindings
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        if (!var_name.empty()) {
                            registerValue(var_name);
                        }
                    }
                }

                // Second pass: analyze values
                for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        analyzeAST(binding->cons_cell.cdr);
                    }
                }

                analyzeAST(op->let_op.body);
                popScope();
                break;
            }

            case ESHKOL_LAMBDA_OP: {
                bool was_in_lambda = in_lambda_;
                in_lambda_ = true;
                std::set<std::string> prev_captured = captured_by_current_lambda_;
                captured_by_current_lambda_.clear();

                pushScope("lambda");

                // Register parameters
                for (uint64_t i = 0; i < op->lambda_op.num_params; i++) {
                    std::string param_name = getVarName(&op->lambda_op.parameters[i]);
                    if (!param_name.empty()) {
                        registerValue(param_name);
                    }
                }

                // Analyze body
                analyzeAST(op->lambda_op.body);

                // Check if body returns a local value
                std::string body_name = getVarName(op->lambda_op.body);
                if (!body_name.empty() && currentScope().local_values.count(body_name) > 0) {
                    markAsReturn(body_name);
                }

                popScope();

                captured_by_current_lambda_ = prev_captured;
                in_lambda_ = was_in_lambda;
                break;
            }

            case ESHKOL_CALL_OP: {
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_SET_OP: {
                // set! to a variable - if the target is in an outer scope,
                // the value escapes
                std::string target = op->set_op.name ? op->set_op.name : "";

                if (!target.empty()) {
                    // Check if target is in a parent scope
                    bool in_outer_scope = false;
                    for (int i = scope_stack_.size() - 2; i >= 0; i--) {
                        if (scope_stack_[i].local_values.count(target) > 0) {
                            in_outer_scope = true;
                            break;
                        }
                    }

                    if (in_outer_scope || current_depth_ == 1) {
                        // Value escapes via mutation
                        std::string value_name = getVarName(op->set_op.value);
                        if (!value_name.empty()) {
                            markGloballyStored(value_name);
                        }
                    }

                    std::string value_name = getVarName(op->set_op.value);
                    if (!value_name.empty()) {
                        recordFlow(value_name, target);
                    }
                }

                analyzeAST(op->set_op.value);
                break;
            }

            case ESHKOL_SEQUENCE_OP: {
                for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                    analyzeAST(&op->sequence_op.expressions[i]);
                }
                break;
            }

            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP: {
                analyzeAST(op->call_op.func);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    analyzeAST(&op->call_op.variables[i]);
                }
                break;
            }

            case ESHKOL_WITH_REGION_OP: {
                pushScope("region");
                for (uint64_t i = 0; i < op->with_region_op.num_body_exprs; i++) {
                    analyzeAST(&op->with_region_op.body[i]);
                }
                popScope();
                break;
            }

            case ESHKOL_OWNED_OP:
                analyzeAST(op->owned_op.value);
                break;

            case ESHKOL_MOVE_OP:
                analyzeAST(op->move_op.value);
                break;

            case ESHKOL_BORROW_OP: {
                analyzeAST(op->borrow_op.value);
                for (uint64_t i = 0; i < op->borrow_op.num_body_exprs; i++) {
                    analyzeAST(&op->borrow_op.body[i]);
                }
                break;
            }

            case ESHKOL_SHARED_OP: {
                // (shared expr) - explicitly requests shared allocation
                std::string value_name = getVarName(op->shared_op.value);
                if (!value_name.empty()) {
                    // Mark as globally stored to force shared allocation
                    markGloballyStored(value_name);
                }
                analyzeAST(op->shared_op.value);
                break;
            }

            case ESHKOL_WEAK_REF_OP:
                analyzeAST(op->weak_ref_op.value);
                break;

            default:
                break;
        }
    }

    // Propagate escape information through the flow graph
    void computeEscapeKinds() {
        // Fixed-point iteration until no changes
        bool changed = true;
        int iterations = 0;
        const int max_iterations = 100;  // Safety limit

        while (changed && iterations < max_iterations) {
            changed = false;
            iterations++;

            for (auto& [name, node] : escape_graph_) {
                EscapeKind new_kind = node.escape_kind;

                // Check direct escape markers
                if (node.is_globally_stored) {
                    new_kind = EscapeKind::GLOBAL_ESCAPE;
                } else if (node.is_closure_captured) {
                    new_kind = std::max(new_kind, EscapeKind::CLOSURE_ESCAPE);
                } else if (node.is_return_value) {
                    new_kind = std::max(new_kind, EscapeKind::RETURN_ESCAPE);
                }

                // Propagate from flows_to (if we flow into an escaping value, we escape)
                for (const auto& target : node.flows_to) {
                    auto it = escape_graph_.find(target);
                    if (it != escape_graph_.end()) {
                        new_kind = std::max(new_kind, it->second.escape_kind);
                    }
                }

                if (new_kind != node.escape_kind) {
                    node.escape_kind = new_kind;
                    changed = true;
                }
            }
        }
    }

    // Determine allocation strategy based on escape kind
    void determineAllocationStrategies() {
        for (auto& [name, node] : escape_graph_) {
            switch (node.escape_kind) {
                case EscapeKind::NO_ESCAPE:
                    node.strategy = AllocationStrategy::STACK;
                    break;
                case EscapeKind::RETURN_ESCAPE:
                    node.strategy = AllocationStrategy::REGION;
                    break;
                case EscapeKind::CLOSURE_ESCAPE:
                case EscapeKind::GLOBAL_ESCAPE:
                    node.strategy = AllocationStrategy::SHARED;
                    break;
            }
        }
    }
};

// Global escape analyzer
static EscapeAnalyzer g_escape_analyzer;

// ===== END ESCAPE ANALYSIS =====

// Forward declarations
static void process_imports(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode);
static void load_file_asts(const std::string& filepath, std::vector<eshkol_ast_t>& asts, bool debug_mode);
static bool g_source_parse_failed = false;

static void print_help(int x = 0)
{
    printf(
        "Usage: eshkol-run [options] <input.esk|input.o> [input.esk|input.o]\n"
        "       eshkol-run -e '<expression>'   (JIT evaluate expression)\n"
        "       eshkol-run -r <file.esk>       (JIT run file)\n\n"
        "\t--help:[-h] = Print this help message.\n"
        "\t--debug:[-d] = Debugging information added inside the program.\n"
        "\t--dump-ast:[-a] = Dumps the AST into a .ast file.\n"
        "\t--dump-ir:[-i] = Dumps the IR into a .ll file.\n"
        "\t--output:[-o] = Outputs into a binary file.\n"
        "\t--compile-only:[-c] = Compiles into an intermediate object file.\n"
        "\t--emit-object = Alias for --compile-only.\n"
        "\t--emit-depfile PATH = With -o/--emit-object, write a Makefile-format\n"
        "\t    depfile listing the entry source plus every file transitively\n"
        "\t    reached via (load …)/(import …)/(require …), so a build system\n"
        "\t    (e.g. ninja DEPFILE) recompiles the object when any of them change.\n"
        "\t--shared-lib:[-s] = Compiles it into a shared library.\n"
        "\t-fPIC = Accepted for build-system compatibility.\n"
        "\t-I DIR = Add a source/module search path.\n"
        "\t-D NAME[=VALUE] = Accepted for build-system compatibility.\n"
        "\t--wasm:[-w] = Compiles to WebAssembly (.wasm) format.\n"
        "\t--profile NAME = Use an execution profile.\n"
        "\t    Profiles: hosted-native, hosted-wasm, hosted-vm, freestanding-kernel-native,\n"
        "\t              freestanding-mcu-native, freestanding-vm, embedded-vm.\n"
        "\t--target TRIPLE = Set the LLVM target triple.\n"
        "\t--require-vm-entry NAME = Require a named VM entry in emitted ESKB.\n"
        "\t--require-vm-entry-zero-arg NAME = Require a named zero-argument VM entry in emitted ESKB.\n"
        "\t--lib:[-l] = Links a shared library to the resulting executable.\n"
        "\t--lib-path:[-L] = Adds a directory to the library search path.\n"
        "\t--no-stdlib:[-n] = Do not auto-load the standard library.\n"
        "\t--eval:[-e] = JIT evaluate an expression; output is shown via (display ...).\n"
        "\t--run:[-r] = JIT run a file (interpret without compiling).\n"
        "\t--version = Print version information.\n"
        "\t--debug-info:[-g] = Emit DWARF debug info (enables lldb/gdb source-level debugging).\n"
        "\t--optimize:[-O] N = Set LLVM optimization level (0=none, 1=basic, 2=full, 3=aggressive).\n"
        "\t--strict-types = Type errors are fatal (default: gradual/warnings).\n"
        "\t--unsafe = Skip all type checks.\n\n"
        "Eshkol Compiler v%s\n",
        ESHKOL_VER
    );
    exit(x);
}

// Load ASTs from a file
static void load_file_asts(const std::string& filepath, std::vector<eshkol_ast_t>& asts, bool debug_mode)
{
    // Check file exists first (required before calling canonical)
    if (!std::filesystem::exists(filepath)) {
        eshkol_error("File not found: %s", filepath.c_str());
        return;
    }

    // Normalize path to prevent duplicate imports
    std::string normalized_path = std::filesystem::canonical(filepath).string();

    // Skip if already imported
    if (imported_files.count(normalized_path)) {
        if (debug_mode) {
            eshkol_debug("Skipping already imported file: %s", normalized_path.c_str());
        }
        return;
    }
    imported_files.insert(normalized_path);

    std::ifstream read_file(filepath);
    if (!read_file.is_open()) {
        eshkol_error("Failed to open file: %s", filepath.c_str());
        return;
    }

    if (debug_mode) {
        eshkol_info("Loading file: %s", filepath.c_str());
    }

    // Reset cumulative line/column counter so this file's first AST is at
    // line 1.  Without this, a file loaded after some other file would
    // start counting at the previous file's last line.
    eshkol_set_parse_source_context(filepath.c_str());
    eshkol_reset_parse_line_counter();

    eshkol_ast_t ast = eshkol_parse_next_ast(read_file);
    while (ast.type != ESHKOL_INVALID) {
        if (debug_mode) {
            printf("\n=== AST Debug Output ===\n");
            eshkol_ast_pretty_print(&ast, 0);
            printf("========================\n\n");
        }
        asts.push_back(ast);
        ast = eshkol_parse_next_ast(read_file);
    }
    /* ESHKOL_INVALID is also the parser's EOF sentinel.  If the stream has
     * not reached EOF, however, the invalid node is a reported syntax error
     * and compilation must not silently truncate the file and return success
     * (notably for the R7RS `syntax-error` form). */
    if (!read_file.eof()) g_source_parse_failed = true;

    read_file.close();
}

static void flatten_top_level_sequences(std::vector<eshkol_ast_t>& asts)
{
    std::vector<eshkol_ast_t> flattened;
    flattened.reserve(asts.size());

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_SEQUENCE_OP) {
            for (uint64_t i = 0; i < ast.operation.sequence_op.num_expressions; i++) {
                flattened.push_back(ast.operation.sequence_op.expressions[i]);
            }
        } else {
            flattened.push_back(ast);
        }
    }

    asts = std::move(flattened);
}

// Process import statements in ASTs and load referenced files
static void process_imports(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode)
{
    flatten_top_level_sequences(asts);

    std::vector<eshkol_ast_t> new_asts;
    std::vector<eshkol_ast_t> imported_asts;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_IMPORT_OP) {
            // This is an import statement
            std::string import_path = ast.operation.import_op.path;

            // Resolve relative paths
            std::filesystem::path resolved_path;
            if (import_path[0] == '/') {
                // Absolute path
                resolved_path = import_path;
            } else {
                // Relative to base directory
                resolved_path = std::filesystem::path(base_dir) / import_path;
            }

            if (!std::filesystem::exists(resolved_path)) {
                eshkol_error("Import file not found: %s", resolved_path.c_str());
                continue;
            }

            // Load the imported file
            std::vector<eshkol_ast_t> file_asts;
            load_file_asts(resolved_path.string(), file_asts, debug_mode);

            // Recursively process imports in the loaded file
            std::string import_dir = resolved_path.parent_path().string();
            process_imports(file_asts, import_dir, debug_mode);

            // Add imported ASTs (definitions from the imported file)
            for (auto& imported_ast : file_asts) {
                imported_asts.push_back(imported_ast);
            }

            // Don't add the import statement itself to new_asts
        } else {
            // Not an import, keep the AST
            new_asts.push_back(ast);
        }
    }

    // Prepend imported ASTs before the current file's ASTs
    asts.clear();
    for (auto& ast : imported_asts) {
        asts.push_back(ast);
    }
    for (auto& ast : new_asts) {
        asts.push_back(ast);
    }
}

// Find the pre-compiled stdlib.o file
static void append_library_candidates(std::vector<std::filesystem::path>& candidates,
                                      const std::vector<char*>& lib_paths,
                                      const std::filesystem::path& leaf_name) {
    for (const auto* lib_path : lib_paths) {
        if (!lib_path || lib_path[0] == '\0') {
            continue;
        }
        candidates.emplace_back(std::filesystem::path(lib_path) / leaf_name);
    }
}

static std::string find_stdlib_object(const std::vector<char*>& lib_paths)
{
    auto cwd = eshkol::platform::current_directory();
    auto exe_dir = eshkol::platform::executable_directory();

    std::vector<std::filesystem::path> candidates;
    append_library_candidates(candidates, lib_paths, "stdlib.o");

    candidates.insert(candidates.end(), {
        exe_dir / "stdlib.o",
        exe_dir / "../lib/stdlib.o",
        exe_dir / "../lib/eshkol/stdlib.o",
        cwd / "stdlib.o",
        cwd / "build/stdlib.o",
        cwd.parent_path() / "build/stdlib.o",
    });

#ifndef _WIN32
    candidates.emplace_back("/usr/local/lib/eshkol/stdlib.o");
    candidates.emplace_back("/usr/local/lib/stdlib.o");
    candidates.emplace_back("/usr/lib/eshkol/stdlib.o");
    candidates.emplace_back("/usr/lib/stdlib.o");
    candidates.emplace_back("/opt/homebrew/lib/eshkol/stdlib.o");
    candidates.emplace_back("/opt/homebrew/lib/stdlib.o");
#endif

    return eshkol::platform::find_first_existing(candidates);
}

// Find the runtime library. Prefer the split runtime archive; fall back to the
// historical aggregate archive for older build/install layouts.
static std::string find_runtime_library(const std::vector<char*>& lib_paths)
{
    auto cwd = eshkol::platform::current_directory();
    auto exe_dir = eshkol::platform::executable_directory();
    std::vector<std::filesystem::path> candidates;

    for (const char* logical_name : {"eshkol-runtime", "eshkol-static"}) {
        const auto library_name = eshkol::platform::static_library_name(logical_name);

        if (const char* env_lib = std::getenv("ESHKOL_LIB_DIR")) {
            candidates.emplace_back(std::filesystem::path(env_lib) / library_name);
        }

        append_library_candidates(candidates, lib_paths, library_name);

        candidates.insert(candidates.end(), {
            exe_dir / library_name,
            exe_dir / "../lib" / library_name,
            exe_dir / "../lib/eshkol" / library_name,
            cwd / library_name,
            cwd / "build" / library_name,
            cwd.parent_path() / "build" / library_name,
        });

#ifndef _WIN32
        candidates.emplace_back("/usr/local/lib" / std::filesystem::path(library_name));
        candidates.emplace_back("/usr/local/lib/eshkol" / std::filesystem::path(library_name));
        candidates.emplace_back("/usr/lib" / std::filesystem::path(library_name));
        candidates.emplace_back("/usr/lib/eshkol" / std::filesystem::path(library_name));
        candidates.emplace_back("/opt/homebrew/lib" / std::filesystem::path(library_name));
        candidates.emplace_back("/opt/homebrew/lib/eshkol" / std::filesystem::path(library_name));
#endif
    }

    return eshkol::platform::find_first_existing(candidates);
}

// Check if any AST contains a require for stdlib or core.* modules.
// Used to decide whether stdlib.o needs to be auto-linked: any core.*
// require pulls in symbols whose bodies live in stdlib.o, same as an
// explicit `(require stdlib)` would.
static bool requires_stdlib(const std::vector<eshkol_ast_t>& asts)
{
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string module_name = ast.operation.require_op.module_names[i];
                if (module_name == "stdlib" || module_name.find("core.") == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

// ESH-0220: Check if any AST contains an EXPLICIT `(require stdlib)` —
// as opposed to requires_stdlib() above, which also matches any dotted
// `core.*` require. This narrower check gates whether we still need to
// synthesize a top-of-module `(require stdlib)`.
//
// stdlib.esk itself defines top-level helpers (__keyword-arg,
// __keyword-args-validate, __keyword-member? — the desugaring targets
// for keyword-formal parameters, see parser.cpp wrap_keyword_formal_body)
// that are NOT re-exported by any individual core.* submodule. Loading
// a core.* module directly (e.g. `(require core.list.search)`) only
// pulls in THAT module's own provided declarations — it does not pull
// in stdlib.esk's own top-level definitions. Previously the synthetic
// `(require stdlib)` insertion was gated on requires_stdlib(asts), so
// any file that explicitly required a dotted core.* module (without
// ALSO explicitly requiring "stdlib") skipped the synthetic require
// entirely, silently losing __keyword-arg/__keyword-args-validate (and
// any other stdlib.esk-local helper) for the rest of the compilation
// unit — surfacing as "Unknown function: __keyword-arg" only when a
// keyword-arg-formal function happened to be defined in the same file.
static bool requires_stdlib_module_explicitly(const std::vector<eshkol_ast_t>& asts)
{
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string module_name = ast.operation.require_op.module_names[i];
                if (module_name == "stdlib") {
                    return true;
                }
            }
        }
    }
    return false;
}

// Check if any AST requires an agent.* module (#248). When true the
// AOT link step splices ESHKOL_HOST_AGENT_FFI_LINK_ARGS into the
// link command so symbols like qllm_http_get / eshkol_sqlite_* /
// qllm_process_* resolve in the produced binary. Programs that don't
// touch agent.* don't pay the libcurl/sqlite/pcre2 link cost.
// Defined after resolve_module_path / g_lib_dir (it now follows transitive
// requires through resolved module files, so it needs both).
static bool requires_agent_ffi(const std::vector<eshkol_ast_t>& asts,
                               const std::string& base_dir);

// Find the library base directory (lib/)
static std::string find_lib_dir()
{
    auto cwd = eshkol::platform::current_directory();
    auto exe_dir = eshkol::platform::executable_directory();

    // Executable-relative first: stdlib + core.* belong to the Eshkol install,
    // not to the cwd. A downstream project (run from its own cwd) may have its
    // OWN lib/ that would otherwise shadow Eshkol's — which breaks stdlib
    // submodule discovery (collect_all_submodules parses <lib>/stdlib.esk), so
    // e.g. (require core.manifold) is not recognised as precompiled and AOT
    // tries to source-resolve it and fails. That makes the -r run-cache fall
    // back to the in-process JIT every run.
    std::vector<std::filesystem::path> candidates = {
        exe_dir / "lib",
        exe_dir / "../lib",
        exe_dir / "../share/eshkol/lib",
        cwd / "lib",
        cwd.parent_path() / "lib",
        cwd / "share/eshkol/lib",
    };

#ifndef _WIN32
    candidates.emplace_back("/usr/local/share/eshkol/lib");
    candidates.emplace_back("/usr/share/eshkol/lib");
#endif

    // The Eshkol lib is the one that actually carries stdlib.esk. Prefer the
    // first candidate that has it (skips a downstream project's unrelated lib/).
    for (const auto& c : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(c / "stdlib.esk", ec)) {
            return c.string();
        }
    }

    return eshkol::platform::find_first_existing(candidates);
}

// Convert symbolic module name to file path
// e.g., "data.json" -> "lib/data/json.esk"
//       "core.strings" -> "lib/core/strings.esk"
//
// Path-literal exception: `(load "...")` strings are stored verbatim by
// the parser since 5992fdb-era fixes; they may legitimately contain
// dots in directory names (e.g. macOS $TMPDIR=/var/folders/<hash>.<r>/T,
// or any project cache dir like build.v2/).  Detect path-like strings
// and skip the dot-to-slash rewrite entirely.
static std::string resolve_module_path(const std::string& module_name, const std::string& base_dir, const std::string& lib_dir)
{
    bool is_path_literal =
        !module_name.empty() &&
        (module_name[0] == '/' ||
         module_name.rfind("./", 0) == 0 ||
         module_name.rfind("../", 0) == 0 ||
         module_name.find('/') != std::string::npos ||
         (module_name.size() > 4 &&
          module_name.compare(module_name.size() - 4, 4, ".esk") == 0));

    std::string path_part;
    if (is_path_literal) {
        path_part = module_name;
        if (path_part.size() < 4 ||
            path_part.compare(path_part.size() - 4, 4, ".esk") != 0) {
            if (!std::filesystem::exists(path_part)) {
                path_part += ".esk";
            }
        }
    } else {
        // Convert dots to path separators (dotted module name)
        path_part = module_name;
        for (char& c : path_part) {
            if (c == '.') c = '/';
        }
        path_part += ".esk";
    }

    // Search order:
    // 1. Current directory (relative to base_dir)
    // 2. Library path (lib/)
    // 3. Environment variable $ESHKOL_PATH (colon-separated)

    // Try current directory first
    std::filesystem::path current_path = std::filesystem::path(base_dir) / path_part;
    if (std::filesystem::exists(current_path)) {
        return std::filesystem::canonical(current_path).string();
    }

    // Try the cwd / project root. A dotted module like `src.core.encoder.x` is
    // rooted at the project, not at the requiring file's directory — so a file in
    // tests/gates/ that (require src.core...) must resolve against ./src/..., not
    // tests/gates/src/.... The in-process JIT resolver already searches cwd; match
    // it here so the AOT path (and thus the -r persistent run-cache) resolves the
    // SAME modules. Without this, any program whose required user module is found
    // by the JIT but not by AOT falls back to in-process JIT on every run — slow,
    // and on arm64 Linux/Windows it also hits the AArch64 Branch26 JIT limit.
    if (base_dir != ".") {
        std::filesystem::path cwd_path = std::filesystem::path(".") / path_part;
        if (std::filesystem::exists(cwd_path)) {
            return std::filesystem::canonical(cwd_path).string();
        }
    }

    // Try library directory
    if (!lib_dir.empty()) {
        std::filesystem::path lib_path = std::filesystem::path(lib_dir) / path_part;
        if (std::filesystem::exists(lib_path)) {
            return std::filesystem::canonical(lib_path).string();
        }
        // Directory-as-module: if lib/web.esk doesn't exist, try lib/web/web.esk
        // This allows (require web) to find lib/web/web.esk as the package entry point
        std::filesystem::path dir_module = std::filesystem::path(lib_dir) / module_name;
        if (std::filesystem::is_directory(dir_module)) {
            // Try same-name entry point: lib/web/web.esk
            std::filesystem::path entry = dir_module / (module_name + ".esk");
            if (std::filesystem::exists(entry)) {
                return std::filesystem::canonical(entry).string();
            }
            // Try index.esk: lib/web/index.esk
            std::filesystem::path index = dir_module / "index.esk";
            if (std::filesystem::exists(index)) {
                return std::filesystem::canonical(index).string();
            }
        }
    }

    // Try $ESHKOL_PATH. Empty segments, non-existent directories, and
    // files-posing-as-dirs are silently ignored (they can't hold
    // modules) but flagged in debug mode so "module not found" errors
    // are easy to trace to a misconfigured path. Known pitfalls:
    //   ESHKOL_PATH=":x"           → empty leading segment
    //   ESHKOL_PATH="/does/not/exist" → valid syntax, wrong content
    //   ESHKOL_PATH="/etc/passwd"  → file instead of directory
    const char* eshkol_path = std::getenv("ESHKOL_PATH");
    if (eshkol_path) {
        std::stringstream ss(eshkol_path);
        std::string search_dir;
        while (std::getline(ss, search_dir, eshkol_path_separator)) {
            if (search_dir.empty()) continue;
            std::error_code ec;
            std::filesystem::path dir_path(search_dir);
            if (!std::filesystem::exists(dir_path, ec)) {
                eshkol_debug("ESHKOL_PATH entry does not exist: %s", search_dir.c_str());
                continue;
            }
            if (!std::filesystem::is_directory(dir_path, ec)) {
                eshkol_debug("ESHKOL_PATH entry is not a directory: %s", search_dir.c_str());
                continue;
            }
            std::filesystem::path env_path = dir_path / path_part;
            if (std::filesystem::exists(env_path, ec)) {
                return std::filesystem::canonical(env_path, ec).string();
            }
        }
    }

    return "";  // Not found
}

// Global library directory (cached)
static std::string g_lib_dir;

// Is `module_name` an agent.* module whose native symbols actually live in
// the optional eshkol-agent-ffi archive (and its libcurl/sqlite3/pcre2/
// ncurses deps)? `agent.crypto` is a deliberate exception (2026-07,
// Noesis bug report #2): its four native symbols (eshkol_sha256/
// eshkol_hmac_sha256/eshkol_random_bytes/eshkol_random_hex) were moved to
// lib/core/crypto_primitives.c, part of the always-linked core eshkol-runtime
// archive, precisely so they no longer need this splice at all. If
// `(require agent.crypto)` still forced ESHKOL_HOST_AGENT_FFI_LINK_ARGS in
// (in particular the `-Wl,-force_load,libeshkol-agent-ffi.a` / whole-archive
// clause), a program that ONLY uses agent.crypto — directly or transitively
// via e.g. core.memory — would still race a concurrent rebuild of that much
// larger, more-frequently-rebuilt archive despite crypto no longer needing
// anything from it. Excluding it here closes that gap for real, rather than
// just moving where the race can happen.
static bool agent_module_needs_ffi_archive(const std::string& module_name)
{
    return module_name.rfind("agent.", 0) == 0 && module_name != "agent.crypto";
}

// Does a module source file (or anything it transitively `require`s) pull in an
// agent.* module? Text-scans `(require …)` clauses and recurses into resolved
// module files. This is how a faculty like core.memory — which uses sha256 via
// (require agent.crypto) INTERNALLY — gets the agent-FFI link args spliced into
// an AOT build of a program that only requires core.memory (not agent.* directly).
// A false positive merely links a few extra libs; correctness is preserved.
static bool file_requires_agent_ffi(const std::string& path,
                                    std::set<std::string>& visited)
{
    std::error_code ec;
    std::string canon = std::filesystem::weakly_canonical(path, ec).string();
    if (canon.empty()) canon = path;
    if (!visited.insert(canon).second) return false;

    std::string text = readFileBytes(std::filesystem::path(path));
    if (text.empty()) return false;
    std::string base_dir = std::filesystem::path(path).parent_path().string();
    if (base_dir.empty()) base_dir = ".";

    size_t pos = 0;
    while ((pos = text.find("(require", pos)) != std::string::npos) {
        pos += 8;  // past "(require"
        size_t close = text.find(')', pos);
        if (close == std::string::npos) break;
        std::string clause = text.substr(pos, close - pos);
        pos = close;
        // Tokenise the require clause on whitespace; strip stray quote/paren chars.
        std::stringstream ts(clause);
        std::string tok;
        while (ts >> tok) {
            while (!tok.empty() && (tok.front() == '(' || tok.front() == '\'' ||
                                    tok.front() == '"'))
                tok.erase(tok.begin());
            while (!tok.empty() && (tok.back() == ')' || tok.back() == '"'))
                tok.pop_back();
            if (tok.empty()) continue;
            if (agent_module_needs_ffi_archive(tok)) return true;
            std::string mp = resolve_module_path(tok, base_dir, g_lib_dir);
            if (!mp.empty() && file_requires_agent_ffi(mp, visited)) return true;
        }
    }
    return false;
}

static bool requires_agent_ffi(const std::vector<eshkol_ast_t>& asts,
                               const std::string& base_dir)
{
    if (g_lib_dir.empty()) g_lib_dir = find_lib_dir();
    std::set<std::string> visited;
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string module_name = ast.operation.require_op.module_names[i];
                if (agent_module_needs_ffi_archive(module_name)) return true;
                std::string mp = resolve_module_path(module_name, base_dir, g_lib_dir);
                if (!mp.empty() && file_requires_agent_ffi(mp, visited)) return true;
            }
        }
    }
    return false;
}

// ESH-0183: recursively collect the content of every source file the compile
// will read, so the persistent run-cache key invalidates on ANY source edit —
// not merely an edit to the entry file.  Reachability follows the same three
// forms the compiler splices at build time: (load "path"), (import "path"),
// and (require module).  The scan is intentionally GENEROUS: over-collection
// only forfeits a cache hit (a re-compile), whereas under-collection would
// resurrect the stale-output bug.  Results are keyed by canonical path in a
// std::map so the eventual digest is order-independent.
static void collectTransitiveSources(
    const std::filesystem::path& path,
    const std::vector<char*>& include_paths,
    std::map<std::string, std::string>& out)
{
    std::error_code ec;
    std::string canon = std::filesystem::weakly_canonical(path, ec).string();
    if (canon.empty()) canon = path.string();
    if (!out.emplace(canon, std::string()).second) {
        return;  // already visited
    }

    std::string text = readFileBytes(path);
    out[canon] = text;  // record bytes (empty string keeps "seen" marker)
    if (text.empty()) return;

    if (g_lib_dir.empty()) g_lib_dir = find_lib_dir();
    std::string base_dir = path.parent_path().string();
    if (base_dir.empty()) base_dir = ".";

    // Resolve a referenced module/path against the referring file's directory,
    // then the -I include paths, then resolve_module_path's own search order
    // (cwd, lib dir, $ESHKOL_PATH).
    auto resolve_ref = [&](const std::string& ref) -> std::string {
        std::string mp = resolve_module_path(ref, base_dir, g_lib_dir);
        if (!mp.empty()) return mp;
        for (char* inc : include_paths) {
            if (!inc || !*inc) continue;
            mp = resolve_module_path(ref, inc, g_lib_dir);
            if (!mp.empty()) return mp;
        }
        return "";
    };

    static const char* kForms[] = {"(load", "(import", "(require"};
    for (const char* form : kForms) {
        const size_t flen = std::strlen(form);
        size_t pos = 0;
        while ((pos = text.find(form, pos)) != std::string::npos) {
            pos += flen;
            const size_t close = text.find(')', pos);
            if (close == std::string::npos) break;
            std::string clause = text.substr(pos, close - pos);
            pos = close;
            // Tokenise; strip stray quote/paren/quasiquote chars (mirrors the
            // require scan used by file_requires_agent_ffi).
            std::stringstream ts(clause);
            std::string tok;
            while (ts >> tok) {
                while (!tok.empty() && (tok.front() == '(' || tok.front() == '\'' ||
                                        tok.front() == '"'))
                    tok.erase(tok.begin());
                while (!tok.empty() && (tok.back() == ')' || tok.back() == '"'))
                    tok.pop_back();
                if (tok.empty()) continue;
                std::string mp = resolve_ref(tok);
                if (!mp.empty())
                    collectTransitiveSources(mp, include_paths, out);
            }
        }
    }
}

static std::string transitiveSourceDigest(
    const std::filesystem::path& entry_source,
    const std::vector<char*>& include_paths)
{
    std::map<std::string, std::string> sources;  // canonical path -> bytes
    collectTransitiveSources(entry_source, include_paths, sources);
    llvm::SHA256 hash;
    for (const auto& entry : sources) {
        hashUpdate(hash, "dep-path", entry.first);
        hashUpdate(hash, "dep-bytes", entry.second);
    }
    return sha256Hex(hash);
}

// ESH-0215: collect the canonical path of the entry file plus every source
// file reachable from it via (load "…") / (import "…") / (require module),
// for --emit-depfile. This walks the SAME graph — through the same
// resolve_module_path search order (referring dir, -I include paths, cwd,
// lib dir, $ESHKOL_PATH) — as transitiveSourceDigest() (fix/aot-stale-output,
// ESH-0183) uses to invalidate the `-r` run-cache; the two exist for
// different consumers (a build-system depfile vs. a persistent-cache key) so
// only the path is retained here, not the file bytes. The scan is
// deliberately generous: over-collection only adds a harmless extra
// prerequisite, whereas under-collection would resurrect the stale-object
// bug (Noesis BUGS-2026-07-04 #3/#5).
static void collectTransitiveSourcePaths(
    const std::filesystem::path& path,
    const std::vector<char*>& include_paths,
    std::vector<std::string>& order,
    std::set<std::string>& seen)
{
    std::error_code ec;
    std::string canon = std::filesystem::weakly_canonical(path, ec).string();
    if (canon.empty()) canon = path.string();
    if (!seen.insert(canon).second) {
        return;  // already visited
    }
    order.push_back(canon);

    std::string text = readFileBytes(path);
    if (text.empty()) return;

    if (g_lib_dir.empty()) g_lib_dir = find_lib_dir();
    std::string base_dir = path.parent_path().string();
    if (base_dir.empty()) base_dir = ".";

    auto resolve_ref = [&](const std::string& ref) -> std::string {
        std::string mp = resolve_module_path(ref, base_dir, g_lib_dir);
        if (!mp.empty()) return mp;
        for (char* inc : include_paths) {
            if (!inc || !*inc) continue;
            mp = resolve_module_path(ref, inc, g_lib_dir);
            if (!mp.empty()) return mp;
        }
        return "";
    };

    static const char* kForms[] = {"(load", "(import", "(require"};
    for (const char* form : kForms) {
        const size_t flen = std::strlen(form);
        size_t pos = 0;
        while ((pos = text.find(form, pos)) != std::string::npos) {
            pos += flen;
            const size_t close = text.find(')', pos);
            if (close == std::string::npos) break;
            std::string clause = text.substr(pos, close - pos);
            pos = close;
            // Tokenise; strip stray quote/paren/quasiquote chars (mirrors the
            // require scan used by file_requires_agent_ffi / transitiveSourceDigest).
            std::stringstream ts(clause);
            std::string tok;
            while (ts >> tok) {
                while (!tok.empty() && (tok.front() == '(' || tok.front() == '\'' ||
                                        tok.front() == '"'))
                    tok.erase(tok.begin());
                while (!tok.empty() && (tok.back() == ')' || tok.back() == '"'))
                    tok.pop_back();
                if (tok.empty()) continue;
                std::string mp = resolve_ref(tok);
                if (!mp.empty())
                    collectTransitiveSourcePaths(mp, include_paths, order, seen);
            }
        }
    }
}

// Escape a path for a Makefile-format depfile: Make treats a bare space as a
// prerequisite separator and '#' as a comment start, so both must be
// backslash-escaped; '$' begins a variable reference, so it is doubled.
static std::string escapeDepfilePath(const std::string& path) {
    std::string out;
    out.reserve(path.size());
    for (char c : path) {
        switch (c) {
        case ' ':
        case '#':
            out.push_back('\\');
            out.push_back(c);
            break;
        case '$':
            out.push_back('$');
            out.push_back('$');
            break;
        default:
            out.push_back(c);
        }
    }
    return out;
}

// ESH-0215: write a Makefile-format depfile (`target: prereq1 prereq2 …`,
// consumable via ninja's `DEPFILE` build-edge attribute or `make`'s
// `-include foo.d`) so a build system recompiles the emitted object when the
// entry source OR any file it transitively (load …)s / (import …)s /
// (require …)s changes. Without this, editing a (load …)ed dependency left
// the object's only tracked prerequisite (the entry file) unchanged, so
// ninja treated the object as up to date — the exact staleness Noesis
// BUGS-2026-07-04 items #3/#5 reported ("must rm the object and run ninja
// 2-3x, or silently ship a stale binary").
static bool writeDepfile(const std::string& depfile_path,
                         const std::string& target_path,
                         const std::string& entry_source,
                         const std::vector<char*>& include_paths) {
    std::vector<std::string> order;
    std::set<std::string> seen;
    collectTransitiveSourcePaths(entry_source, include_paths, order, seen);

    std::ofstream out(depfile_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        eshkol_error("Failed to write depfile: %s", depfile_path.c_str());
        return false;
    }

    // The left-hand target spelling is conventional (ninja documents that it
    // ignores the depfile's own target field and attaches the listed
    // prerequisites to the OUTPUT of the owning build edge instead), so the
    // caller-provided path is used verbatim rather than canonicalized.
    out << escapeDepfilePath(target_path) << ":";
    for (const auto& dep : order) {
        out << " \\\n  " << escapeDepfilePath(dep);
    }
    out << "\n";
    if (!out.good()) {
        eshkol_error("Failed to write depfile: %s", depfile_path.c_str());
        return false;
    }
    return true;
}

// Recursively discover all modules that a library requires.
// Used to mark all sub-modules of a pre-compiled library as pre-compiled.
// NOTE: Does NOT use load_file_asts to avoid polluting imported_files.
static void collect_all_submodules(const std::string& module_name,
                                   std::set<std::string>& out,
                                   const std::string& lib_dir) {
    if (out.count(module_name)) return;  // already visited
    out.insert(module_name);

    // Find and parse the module source to discover its requires
    std::string module_path = resolve_module_path(module_name, ".", lib_dir);
    if (module_path.empty()) return;

    // Parse directly — do NOT use load_file_asts (it tracks imported_files
    // and would prevent process_requires from loading these modules later)
    std::ifstream file(module_path);
    if (!file.is_open()) return;

    eshkol_set_parse_source_context(module_path.c_str());
    eshkol_reset_parse_line_counter();
    eshkol_ast_t ast = eshkol_parse_next_ast(file);
    while (ast.type != ESHKOL_INVALID) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string sub = ast.operation.require_op.module_names[i];
                collect_all_submodules(sub, out, lib_dir);
            }
        }
        eshkol_set_parse_source_context(module_path.c_str());
        ast = eshkol_parse_next_ast(file);
    }
}

// ===== SYMBOL VISIBILITY HELPERS =====

// Collect exported symbols from provide declarations in ASTs
static std::set<std::string> collect_module_exports(const std::vector<eshkol_ast_t>& asts) {
    std::set<std::string> exports;
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_PROVIDE_OP) {
            for (uint64_t i = 0; i < ast.operation.provide_op.num_exports; i++) {
                exports.insert(ast.operation.provide_op.export_names[i]);
            }
        }
    }
    return exports;
}

// Collect all defined symbols from ASTs
static std::set<std::string> collect_defined_symbols(const std::vector<eshkol_ast_t>& asts) {
    std::set<std::string> defined;
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP) {
            if (ast.operation.define_op.name) {
                defined.insert(ast.operation.define_op.name);
            }
        }
    }
    return defined;
}

static char* copy_ast_cstr(const std::string& value) {
    char* out = new char[value.size() + 1];
    if (out) {
        memcpy(out, value.c_str(), value.size() + 1);
    }
    return out;
}

static eshkol_ast_t make_runtime_var_ast(const std::string& name,
                                         uint32_t line,
                                         uint32_t column) {
    eshkol_ast_t ast = {};
    ast.type = ESHKOL_VAR;
    ast.line = line;
    ast.column = column;
    ast.variable.id = copy_ast_cstr(name);
    ast.variable.data = nullptr;
    return ast;
}

static eshkol_ast_t make_runtime_define_alias_ast(const std::string& alias,
                                                  const std::string& source,
                                                  uint32_t line,
                                                  uint32_t column) {
    eshkol_ast_t ast = {};
    ast.type = ESHKOL_OP;
    ast.line = line;
    ast.column = column;
    ast.operation.op = ESHKOL_DEFINE_OP;
    ast.operation.define_op.name = copy_ast_cstr(alias);
    ast.operation.define_op.value = new eshkol_ast_t;
    *ast.operation.define_op.value = make_runtime_var_ast(source, line, column);
    return ast;
}

static void append_r7rs_prefix_aliases_for_module(const eshkol_ast_t& require_ast,
                                                  uint64_t module_index,
                                                  const std::set<std::string>& exports,
                                                  std::vector<eshkol_ast_t>& out) {
    const auto& require_op = require_ast.operation.require_op;
    if (!require_op.import_prefixes ||
        module_index >= require_op.num_modules ||
        !require_op.import_prefixes[module_index] ||
        require_op.import_prefixes[module_index][0] == '\0') {
        return;
    }

    std::set<std::string> excepts;
    if (require_op.import_except_names &&
        require_op.num_import_except_names &&
        require_op.import_except_names[module_index]) {
        for (uint64_t i = 0; i < require_op.num_import_except_names[module_index]; i++) {
            const char* name = require_op.import_except_names[module_index][i];
            if (name) {
                excepts.insert(name);
            }
        }
    }

    const std::string prefix = require_op.import_prefixes[module_index];
    for (const auto& exported : exports) {
        if (excepts.count(exported) > 0) continue;
        out.push_back(make_runtime_define_alias_ast(prefix + exported,
                                                    exported,
                                                    require_ast.line,
                                                    require_ast.column));
    }
}

// Forward declaration for recursive reference updating
static void update_ast_references(eshkol_ast_t* ast,
                                  const std::map<std::string, std::string>& rename_map);

// Update references in an AST node recursively
static void update_ast_references(eshkol_ast_t* ast,
                                  const std::map<std::string, std::string>& rename_map) {
    if (!ast) return;

    switch (ast->type) {
        case ESHKOL_VAR: {
            if (ast->variable.id) {
                auto it = rename_map.find(ast->variable.id);
                if (it != rename_map.end()) {
                    delete[] ast->variable.id;
                    ast->variable.id = strdup(it->second.c_str());
                }
            }
            break;
        }

        case ESHKOL_OP: {
            switch (ast->operation.op) {
                case ESHKOL_CALL_OP:
                case ESHKOL_COND_OP:  // cond uses call_op structure
                case ESHKOL_IF_OP:    // if uses call_op structure
                    if (ast->operation.call_op.func) {
                        update_ast_references(ast->operation.call_op.func, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        update_ast_references(&ast->operation.call_op.variables[i], rename_map);
                    }
                    break;

                case ESHKOL_AND_OP:
                case ESHKOL_OR_OP:
                    // NOTE: AND_OP/OR_OP use sequence_op structure, NOT call_op!
                    for (uint64_t i = 0; i < ast->operation.sequence_op.num_expressions; i++) {
                        update_ast_references(&ast->operation.sequence_op.expressions[i], rename_map);
                    }
                    break;

                case ESHKOL_DEFINE_OP:
                    // Don't rename the definition name itself - that's handled separately
                    // But do update references in the body
                    update_ast_references(ast->operation.define_op.value, rename_map);
                    break;

                case ESHKOL_LAMBDA_OP:
                    update_ast_references(ast->operation.lambda_op.body, rename_map);
                    break;

                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:  // R7RS letrec* - used for internal defines
                    // Each binding is a CONS cell: (var . value)
                    for (uint64_t i = 0; i < ast->operation.let_op.num_bindings; i++) {
                        eshkol_ast_t* binding = &ast->operation.let_op.bindings[i];
                        if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr) {
                            // Update references in the value part
                            update_ast_references(binding->cons_cell.cdr, rename_map);
                        } else {
                            // Fallback: treat entire binding as expression
                            update_ast_references(binding, rename_map);
                        }
                    }
                    if (ast->operation.let_op.body) {
                        update_ast_references(ast->operation.let_op.body, rename_map);
                    }
                    break;

                case ESHKOL_SEQUENCE_OP:
                    for (uint64_t i = 0; i < ast->operation.sequence_op.num_expressions; i++) {
                        update_ast_references(&ast->operation.sequence_op.expressions[i], rename_map);
                    }
                    break;

                case ESHKOL_SET_OP:
                    // Update the target variable name
                    if (ast->operation.set_op.name) {
                        auto it = rename_map.find(ast->operation.set_op.name);
                        if (it != rename_map.end()) {
                            delete[] ast->operation.set_op.name;
                            ast->operation.set_op.name = strdup(it->second.c_str());
                        }
                    }
                    update_ast_references(ast->operation.set_op.value, rename_map);
                    break;

                // Calculus operations - function + point (or expression)
                case ESHKOL_GRADIENT_OP:
                    if (ast->operation.gradient_op.function) {
                        update_ast_references(ast->operation.gradient_op.function, rename_map);
                    }
                    if (ast->operation.gradient_op.point) {
                        update_ast_references(ast->operation.gradient_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DERIVATIVE_OP:
                    if (ast->operation.derivative_op.function) {
                        update_ast_references(ast->operation.derivative_op.function, rename_map);
                    }
                    if (ast->operation.derivative_op.point) {
                        update_ast_references(ast->operation.derivative_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DIRECTIONAL_DERIV_OP:
                    if (ast->operation.directional_deriv_op.function) {
                        update_ast_references(ast->operation.directional_deriv_op.function, rename_map);
                    }
                    if (ast->operation.directional_deriv_op.point) {
                        update_ast_references(ast->operation.directional_deriv_op.point, rename_map);
                    }
                    if (ast->operation.directional_deriv_op.direction) {
                        update_ast_references(ast->operation.directional_deriv_op.direction, rename_map);
                    }
                    break;

                case ESHKOL_JACOBIAN_OP:
                    if (ast->operation.jacobian_op.function) {
                        update_ast_references(ast->operation.jacobian_op.function, rename_map);
                    }
                    if (ast->operation.jacobian_op.point) {
                        update_ast_references(ast->operation.jacobian_op.point, rename_map);
                    }
                    break;

                case ESHKOL_HESSIAN_OP:
                    if (ast->operation.hessian_op.function) {
                        update_ast_references(ast->operation.hessian_op.function, rename_map);
                    }
                    if (ast->operation.hessian_op.point) {
                        update_ast_references(ast->operation.hessian_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DIVERGENCE_OP:
                    if (ast->operation.divergence_op.function) {
                        update_ast_references(ast->operation.divergence_op.function, rename_map);
                    }
                    if (ast->operation.divergence_op.point) {
                        update_ast_references(ast->operation.divergence_op.point, rename_map);
                    }
                    break;

                case ESHKOL_CURL_OP:
                    if (ast->operation.curl_op.function) {
                        update_ast_references(ast->operation.curl_op.function, rename_map);
                    }
                    if (ast->operation.curl_op.point) {
                        update_ast_references(ast->operation.curl_op.point, rename_map);
                    }
                    break;

                case ESHKOL_LAPLACIAN_OP:
                    if (ast->operation.laplacian_op.function) {
                        update_ast_references(ast->operation.laplacian_op.function, rename_map);
                    }
                    if (ast->operation.laplacian_op.point) {
                        update_ast_references(ast->operation.laplacian_op.point, rename_map);
                    }
                    break;

                case ESHKOL_DIFF_OP:
                    if (ast->operation.diff_op.expression) {
                        update_ast_references(ast->operation.diff_op.expression, rename_map);
                    }
                    break;

                // call_op structure operations (same as CALL_OP handler)
                case ESHKOL_WHEN_OP:
                case ESHKOL_UNLESS_OP:
                case ESHKOL_DO_OP:
                case ESHKOL_CASE_OP:
                case ESHKOL_QUOTE_OP:
                case ESHKOL_QUASIQUOTE_OP:
                case ESHKOL_UNQUOTE_OP:
                case ESHKOL_UNQUOTE_SPLICING_OP:
                    if (ast->operation.call_op.func) {
                        update_ast_references(ast->operation.call_op.func, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        update_ast_references(&ast->operation.call_op.variables[i], rename_map);
                    }
                    break;

                // Control flow operations
                case ESHKOL_DYNAMIC_WIND_OP:
                    if (ast->operation.dynamic_wind_op.before) {
                        update_ast_references(ast->operation.dynamic_wind_op.before, rename_map);
                    }
                    if (ast->operation.dynamic_wind_op.thunk) {
                        update_ast_references(ast->operation.dynamic_wind_op.thunk, rename_map);
                    }
                    if (ast->operation.dynamic_wind_op.after) {
                        update_ast_references(ast->operation.dynamic_wind_op.after, rename_map);
                    }
                    break;

                case ESHKOL_CALL_CC_OP:
                    if (ast->operation.call_cc_op.proc) {
                        update_ast_references(ast->operation.call_cc_op.proc, rename_map);
                    }
                    break;

                case ESHKOL_GUARD_OP: {
                    for (uint64_t i = 0; i < ast->operation.guard_op.num_body_exprs; i++) {
                        update_ast_references(&ast->operation.guard_op.body[i], rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.guard_op.num_clauses; i++) {
                        update_ast_references(&ast->operation.guard_op.clauses[i], rename_map);
                    }
                    break;
                }

                case ESHKOL_RAISE_OP:
                    if (ast->operation.raise_op.exception) {
                        update_ast_references(ast->operation.raise_op.exception, rename_map);
                    }
                    break;

                case ESHKOL_VALUES_OP:
                    for (uint64_t i = 0; i < ast->operation.values_op.num_values; i++) {
                        update_ast_references(&ast->operation.values_op.expressions[i], rename_map);
                    }
                    break;

                case ESHKOL_CALL_WITH_VALUES_OP:
                    if (ast->operation.call_with_values_op.producer) {
                        update_ast_references(ast->operation.call_with_values_op.producer, rename_map);
                    }
                    if (ast->operation.call_with_values_op.consumer) {
                        update_ast_references(ast->operation.call_with_values_op.consumer, rename_map);
                    }
                    break;

                case ESHKOL_MATCH_OP: {
                    if (ast->operation.match_op.expr) {
                        update_ast_references(ast->operation.match_op.expr, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.match_op.num_clauses; i++) {
                        if (ast->operation.match_op.clauses[i].guard) {
                            update_ast_references(ast->operation.match_op.clauses[i].guard, rename_map);
                        }
                        if (ast->operation.match_op.clauses[i].body) {
                            update_ast_references(ast->operation.match_op.clauses[i].body, rename_map);
                        }
                    }
                    break;
                }

                // Memory management operations - FIXED: with_region_op.body is an array
                case ESHKOL_WITH_REGION_OP:
                    for (uint64_t i = 0; i < ast->operation.with_region_op.num_body_exprs; i++) {
                        update_ast_references(&ast->operation.with_region_op.body[i], rename_map);
                    }
                    break;

                case ESHKOL_OWNED_OP:
                    update_ast_references(ast->operation.owned_op.value, rename_map);
                    break;

                case ESHKOL_MOVE_OP:
                    update_ast_references(ast->operation.move_op.value, rename_map);
                    break;

                // FIXED: borrow_op.body is an array
                case ESHKOL_BORROW_OP:
                    if (ast->operation.borrow_op.value) {
                        update_ast_references(ast->operation.borrow_op.value, rename_map);
                    }
                    for (uint64_t i = 0; i < ast->operation.borrow_op.num_body_exprs; i++) {
                        update_ast_references(&ast->operation.borrow_op.body[i], rename_map);
                    }
                    break;

                case ESHKOL_SHARED_OP:
                    update_ast_references(ast->operation.shared_op.value, rename_map);
                    break;

                case ESHKOL_WEAK_REF_OP:
                    update_ast_references(ast->operation.weak_ref_op.value, rename_map);
                    break;

                // Logic/consciousness operations use call_op structure
                case ESHKOL_UNIFY_OP:
                case ESHKOL_MAKE_SUBST_OP:
                case ESHKOL_WALK_OP:
                case ESHKOL_MAKE_FACT_OP:
                case ESHKOL_MAKE_KB_OP:
                case ESHKOL_KB_ASSERT_OP:
                case ESHKOL_KB_QUERY_OP:
                case ESHKOL_LOGIC_VAR_PRED_OP:
                case ESHKOL_SUBSTITUTION_PRED_OP:
                case ESHKOL_KB_PRED_OP:
                case ESHKOL_FACT_PRED_OP:
                case ESHKOL_FACTOR_GRAPH_PRED_OP:
                case ESHKOL_WORKSPACE_PRED_OP:
                case ESHKOL_MAKE_FACTOR_GRAPH_OP:
                case ESHKOL_FG_ADD_FACTOR_OP:
                case ESHKOL_FG_INFER_OP:
                case ESHKOL_FG_UPDATE_CPT_OP:
                case ESHKOL_FREE_ENERGY_OP:
                case ESHKOL_EXPECTED_FREE_ENERGY_OP:
                case ESHKOL_MAKE_WORKSPACE_OP:
                case ESHKOL_WS_REGISTER_OP:
                case ESHKOL_WS_STEP_OP:
                case ESHKOL_DNC_MAKE_OP:
                case ESHKOL_DNC_CONTENT_ADDR_OP:
                case ESHKOL_DNC_LOC_ADDR_OP:
                case ESHKOL_DNC_READ_OP:
                case ESHKOL_DNC_WRITE_OP:
                case ESHKOL_DNC_ALLOC_WEIGHTS_OP:
                case ESHKOL_DNC_READ_GRAD_OP:
                case ESHKOL_DNC_PRED_OP:
                case ESHKOL_SDNC_PROGRAM_OP:
                case ESHKOL_SDNC_RUN_OP:
                case ESHKOL_SDNC_WEIGHT_GRAD_OP:
                case ESHKOL_SDNC_PARAMS_OP:
                case ESHKOL_SDNC_SET_PARAMS_OP:
                case ESHKOL_SDNC_IMPROVE_OP:
                case ESHKOL_SDNC_PRED_OP:
                    // Consciousness-engine ops use call_op semantics — they
                    // codegen as direct function calls with the same arg
                    // shape as ESHKOL_CALL_OP.
                    for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                        update_ast_references(&ast->operation.call_op.variables[i], rename_map);
                    }
                    break;

                case ESHKOL_EXTERN_OP:
                    // ESHKOL_EXTERN_OP uses `extern_op` in the union (name /
                    // real_name / return_type / parameters / num_params),
                    // NOT `call_op`. Reading call_op.num_vars here dereferenced
                    // extern_op.return_type (a char*) as a uint64_t length and
                    // walked off into uninitialised memory — SIGSEGV in every
                    // private-extern-bearing module under process_requires +
                    // rename_private_symbols.
                    for (uint64_t i = 0; i < ast->operation.extern_op.num_params; i++) {
                        update_ast_references(&ast->operation.extern_op.parameters[i], rename_map);
                    }
                    break;

                default:
                    eshkol_debug("update_ast_references: unhandled op type %d", ast->operation.op);
                    break;
            }
            break;
        }

        case ESHKOL_CONS:
            // Traverse cons cells
            if (ast->cons_cell.car) {
                update_ast_references(ast->cons_cell.car, rename_map);
            }
            if (ast->cons_cell.cdr) {
                update_ast_references(ast->cons_cell.cdr, rename_map);
            }
            break;

        default:
            // Literals, etc. - no update needed
            break;
    }
}

// Rename private (non-exported) symbols in module ASTs
static void rename_private_symbols(std::vector<eshkol_ast_t>& asts,
                                   const std::string& module_name,
                                   const std::set<std::string>& exports,
                                   bool debug_mode) {
    // Build rename map: private_name -> mangled_name
    std::map<std::string, std::string> rename_map;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP) {
            if (ast.operation.define_op.name) {
                std::string name = ast.operation.define_op.name;
                // Only rename if not exported
                if (exports.count(name) == 0) {
                    std::string mangled = ModuleSymbolTable::getPrivateName(module_name, name);
                    rename_map[name] = mangled;

                    if (debug_mode) {
                        eshkol_debug("  Renaming private symbol '%s' -> '%s'",
                                    name.c_str(), mangled.c_str());
                    }
                }
            }
        }
    }

    if (rename_map.empty()) {
        return;  // No private symbols to rename
    }

    // Apply renames: first rename definitions, then update all references
    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_OP) {
            if (ast.operation.define_op.name) {
                auto it = rename_map.find(ast.operation.define_op.name);
                if (it != rename_map.end()) {
                    delete[] ast.operation.define_op.name;
                    ast.operation.define_op.name = strdup(it->second.c_str());
                }
            }
        }
        // Update all references in this AST
        update_ast_references(&ast, rename_map);
    }
}

// ===== END SYMBOL VISIBILITY HELPERS =====

// Process require statements (new module system)
static void process_requires(std::vector<eshkol_ast_t>& asts, const std::string& base_dir, bool debug_mode)
{
    if (g_lib_dir.empty()) {
        g_lib_dir = find_lib_dir();
    }

    flatten_top_level_sequences(asts);

    std::vector<eshkol_ast_t> new_asts;
    std::vector<eshkol_ast_t> required_asts;

    for (auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_REQUIRE_OP) {
            // Process each required module
            for (uint64_t i = 0; i < ast.operation.require_op.num_modules; i++) {
                std::string module_name = ast.operation.require_op.module_names[i];

                // Check if this module is pre-compiled (e.g., from stdlib.o)
                bool is_precompiled = precompiled_modules.count(module_name) > 0;

                // All sub-modules of pre-compiled libraries are already in
                // precompiled_modules (populated by collect_all_submodules),
                // so the direct lookup above is sufficient.

                if (is_precompiled) {
                    // For pre-compiled modules, we parse to get function declarations
                    // but strip the bodies. The actual code comes from the .o file.
                    std::string module_path = resolve_module_path(module_name, base_dir, g_lib_dir);
                    if (module_path.empty()) {
                        if (debug_mode) {
                            eshkol_warn("Pre-compiled module %s source not found - skipping declarations", module_name.c_str());
                        }
                        continue;
                    }

                    if (debug_mode) {
                        eshkol_info("Module %s is pre-compiled - loading declarations from %s", module_name.c_str(), module_path.c_str());
                    }

                    // Load the module to get function declarations
                    std::vector<eshkol_ast_t> module_asts;
                    load_file_asts(module_path, module_asts, debug_mode);

                    if (module_asts.empty()) {
                        continue;
                    }

                    // Collect module exports BEFORE processing requires (process_requires may modify ASTs)
                    std::set<std::string> module_exports = collect_module_exports(module_asts);
                    std::set<std::string> module_defined_symbols = collect_defined_symbols(module_asts);
                    std::set<std::string> r7rs_alias_sources =
                        module_exports.empty() ? module_defined_symbols : module_exports;
                    // Register the importable surface for later duplicate import sets.
                    g_symbol_table.registerModuleExports(module_name, r7rs_alias_sources);
                    if (debug_mode) {
                        std::string exp_str;
                        for (const auto& e : module_exports) {
                            if (!exp_str.empty()) exp_str += ", ";
                            exp_str += e;
                        }
                        eshkol_debug("Module provides: %s (count=%zu)", exp_str.c_str(), module_exports.size());
                    }

                    // Recursively process requires in the module (they're also pre-compiled)
                    std::string module_dir = std::filesystem::path(module_path).parent_path().string();
                    process_requires(module_asts, module_dir, debug_mode);

                    // Extract function/variable definitions for external linkage
                    // After process_requires, module_asts contains:
                    //   - ASTs from sub-required modules (already marked external)
                    //   - This module's ASTs (need to check against module_exports)
                    // The actual code/data will come from stdlib.o at link time
                    for (auto& module_ast : module_asts) {
                        if (module_ast.type == ESHKOL_OP &&
                            module_ast.operation.op == ESHKOL_DEFINE_OP) {
                            const char* sym_name = module_ast.operation.define_op.name;
                            if (!sym_name) continue;

                            // Only export if symbol is public (in provides list) OR already external
                            // Already-external symbols come from sub-required modules
                            // Private symbols have been/will be renamed with module prefix
                            bool already_external = module_ast.operation.define_op.is_external;
                            bool is_public = already_external || module_exports.empty() || module_exports.count(sym_name) > 0;
                            if (debug_mode) {
                                eshkol_debug("  Check symbol %s: exports_empty=%d, in_exports=%d, is_public=%d",
                                            sym_name, (int)module_exports.empty(),
                                            (int)(module_exports.count(sym_name) > 0), (int)is_public);
                            }
                            if (!is_public) {
                                if (debug_mode) {
                                    eshkol_debug("  Skipping private symbol: %s", sym_name);
                                }
                                continue;
                            }

                            if (module_ast.operation.define_op.is_function) {
                                // Mark as external function (body will come from .o file)
                                module_ast.operation.define_op.is_external = 1;
                                required_asts.push_back(module_ast);
                                if (debug_mode) {
                                    eshkol_info("  External function: %s", sym_name);
                                }
                            } else {
                                // Export PUBLIC variable as external
                                // Only public variables (in provides) need external declarations
                                module_ast.operation.define_op.is_external = 1;
                                required_asts.push_back(module_ast);
                                if (debug_mode) {
                                    eshkol_info("  External variable: %s", sym_name);
                                }
                            }
                        }
                    }
                    append_r7rs_prefix_aliases_for_module(ast, i, r7rs_alias_sources, required_asts);
                    continue;
                }

                std::string module_path = resolve_module_path(module_name, base_dir, g_lib_dir);

                if (module_path.empty()) {
                    eshkol_error("Module '%s' not found", module_name.c_str());
                    eshkol_error("  Searched:");
                    eshkol_error("    - %s/%s.esk", base_dir.c_str(), module_name.c_str());
                    if (!g_lib_dir.empty()) {
                        // Convert dots to slashes for display
                        std::string path_display = module_name;
                        for (char& c : path_display) if (c == '.') c = '/';
                        eshkol_error("    - %s/%s.esk", g_lib_dir.c_str(), path_display.c_str());
                    }
                    eshkol_error("    - $ESHKOL_PATH entries");
                    continue;
                }

                if (debug_mode) {
                    eshkol_info("Requiring module '%s' from: %s", module_name.c_str(), module_path.c_str());
                }

                // Normalize path for cycle detection
                std::string norm_path = std::filesystem::canonical(module_path).string();

                // Circular dependency detection: if this module is currently being loaded
                // higher up the call stack AND hasn't been fully loaded yet, we have a cycle.
                // If it's already in imported_files, it's fully loaded — not a cycle.
                if (g_loading_modules.count(norm_path) && !imported_files.count(norm_path)) {
                    eshkol_error("Circular dependency detected: module '%s' requires itself (directly or indirectly)",
                                module_name.c_str());
                    continue;
                }
                g_loading_modules.insert(norm_path);

                // Load the module file
                std::vector<eshkol_ast_t> module_asts;
                load_file_asts(module_path, module_asts, debug_mode);

                // Skip if module was already loaded (detected by load_file_asts)
                if (module_asts.empty()) {
                    append_r7rs_prefix_aliases_for_module(
                        ast, i, g_symbol_table.exportsFor(module_name), required_asts);
                    g_loading_modules.erase(norm_path);
                    continue;
                }

                // Collect exports from this module BEFORE processing sub-modules
                std::set<std::string> exports = collect_module_exports(module_asts);
                std::set<std::string> defined_symbols = collect_defined_symbols(module_asts);
                std::set<std::string> r7rs_alias_sources =
                    exports.empty() ? defined_symbols : exports;

                // Register the importable surface for later duplicate import sets.
                g_symbol_table.registerModuleExports(module_name, r7rs_alias_sources);

                if (debug_mode && !exports.empty()) {
                    std::string exp_str;
                    for (const auto& e : exports) {
                        if (!exp_str.empty()) exp_str += ", ";
                        exp_str += e;
                    }
                    eshkol_info("Module '%s' exports: [%s]", module_name.c_str(), exp_str.c_str());
                }

                // Bug Z (Noesis 2026-04-30): `(provide ...)` is
                // documented and used (across 65 Noesis source files
                // and the Eshkol stdlib itself) as INFORMATIONAL, not
                // as a hard export boundary.  JIT mode treats it that
                // way; AOT was renaming non-exported names so calls
                // from other files failed with "Unknown function: X"
                // with a misleading source marker.  Match the
                // documented + JIT semantics here.  If a strict
                // export mode is wanted later, expose it via a
                // per-file pragma so existing code keeps compiling.
                //
                // Collision avoidance was the original motivation —
                // see git history of rename_private_symbols.  In
                // practice multi-module collisions show up at link
                // time as ODR / "duplicate symbol" errors, so we
                // surface them where they originate rather than
                // hiding them with a silent global mangle.  For now
                // we skip the rename entirely; rename_private_symbols
                // and the ESHKOL_PROVIDE_OP machinery stay in place
                // for the future strict-mode pragma.
                (void)rename_private_symbols;
                (void)exports;

                // Recursively process requires in the loaded module
                std::string module_dir = std::filesystem::path(module_path).parent_path().string();
                process_requires(module_asts, module_dir, debug_mode);

                // Also process legacy imports
                process_imports(module_asts, module_dir, debug_mode);

                // Add module ASTs
                for (auto& module_ast : module_asts) {
                    required_asts.push_back(module_ast);
                }
                append_r7rs_prefix_aliases_for_module(ast, i, r7rs_alias_sources, required_asts);

                // Remove from loading stack (module fully processed)
                g_loading_modules.erase(norm_path);
            }
            // Don't add the require statement itself to new_asts
        } else if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_PROVIDE_OP) {
            // For now, we just skip provide statements
            // Full symbol visibility will be implemented later
            if (debug_mode) {
                std::string exports;
                for (uint64_t i = 0; i < ast.operation.provide_op.num_exports; i++) {
                    if (i > 0) exports += ", ";
                    exports += ast.operation.provide_op.export_names[i];
                }
                eshkol_debug("Module provides: %s", exports.c_str());
            }
            // Don't add provide to final ASTs
        } else {
            new_asts.push_back(ast);
        }
    }

    // Prepend required modules before the current file's ASTs
    asts.clear();
    for (auto& ast : required_asts) {
        asts.push_back(ast);
    }
    for (auto& ast : new_asts) {
        asts.push_back(ast);
    }
}

/* `(command-line)` reads these globals from libeshkol-static.
 * They default to zero/null (weak linkage in arena_memory.cpp) since
 * the runtime supports being embedded with no real main. eshkol-run
 * is a real main, so we publish argc/argv here at startup so user
 * scripts under `-e`, `-r`, or compile-and-link see the actual
 * command line instead of an empty list. The full argv is published
 * (including argv[0] = the eshkol-run binary path); user-script args
 * begin at argv[optind+1] after argument parsing. */
extern "C" {
    extern int32_t __eshkol_argc;
    extern char**  __eshkol_argv;
}

/* LeakSanitizer policy for eshkol-run.
 *
 * eshkol-run is a one-shot batch compiler: parse → typecheck → codegen
 * → emit object → exit. AST nodes (every `new eshkol_ast_t[N]` and
 * `new char[N]` in lib/frontend/parser.cpp) are owned by the AST tree
 * and intentionally never freed — process exit reaps them. This is
 * the same convention clang, rustc, and gcc use for their internal
 * IRs, since walking and freeing a multi-hundred-thousand-node AST at
 * exit is pure busywork on the way to _exit().
 *
 * exitcode=0 tells LSan to PRINT exit-time leaks (so we still see them
 * in CI logs and can drive the long-tail cleanup tracked under the
 * resource-management hardening epic) but not fail the build on them.
 * ASan's other checks — heap-buffer-overflow, use-after-free, invalid
 * free, double-free, stack-use-after-scope — remain hard failures.
 * UBSan is unaffected. Long-running leaks during execution can still
 * be caught by calling __lsan_do_recoverable_leak_check() at well-
 * defined recovery points if needed.
 *
 * When the AST gains a proper destructor (epic #182, or the v1.3
 * arena-backed AST refactor), this override should be removed so that
 * any regression — a new alloc that *isn't* exit-time-only — fails CI. */
/* clang exposes __has_feature; gcc does not (and using it as
 * `defined(__has_feature)` works on clang but mis-parses on gcc).
 * Detect each compiler's sanitizer macro independently. */
#ifndef ESHKOL_HAS_ASAN
# if defined(__SANITIZE_ADDRESS__)
#  define ESHKOL_HAS_ASAN 1
# elif defined(__clang__)
#  if defined(__has_feature)
#   if __has_feature(address_sanitizer) || __has_feature(leak_sanitizer)
#    define ESHKOL_HAS_ASAN 1
#   endif
#  endif
# endif
#endif

#ifdef ESHKOL_HAS_ASAN
extern "C" const char* __lsan_default_options(void) {
    return "exitcode=0:print_suppressions=0:report_objects=1";
}
#endif

int main(int argc, char **argv)
{
    __eshkol_argc = (int32_t)argc;
    __eshkol_argv = argv;

    int ch = 0;

    uint8_t debug_mode = 0;
    uint8_t dump_ast = 0;
    uint8_t dump_ir = 0;
    uint8_t compile_only = 0;
    uint8_t shared_lib = 0;
    uint8_t wasm_output = 0;
    uint8_t no_stdlib = 0;
    uint8_t strict_types = 0;
    uint8_t unsafe_mode = 0;
    uint8_t debug_info = 0;  // -g flag: emit DWARF debug info for lldb/gdb
    int opt_level = 0;       // -O flag: LLVM optimization level (0-3)
    bool opt_level_explicit = false;  // true once the user passes an explicit -O

    std::vector<char*> source_files;
    std::vector<char*> compiled_files;
    std::vector<char*> linked_libs;
    std::vector<char*> lib_paths;
    std::vector<char*> include_paths;

    std::vector<eshkol_ast_t> asts;

    char *output = nullptr;
    char *eval_expr = nullptr;  // For -e/--eval flag
    uint8_t run_mode = 0;       // For -r/--run flag (JIT run file)
    const char* profile_name = nullptr;
    const char* target_triple = nullptr;
    bool freestanding_native_profile = false;
    bool vm_only_profile = false;
    bool embedded_vm_profile = false;
    std::vector<std::string> required_vm_entries;
    std::vector<std::string> required_zero_arg_vm_entries;
    const char* depfile_path = nullptr;  // --emit-depfile PATH (ESH-0215)

    if (argc == 1) print_help(1);

    const char* eskb_output_path = nullptr;
    while ((ch = getopt_long(argc, argv, "hdaio:cswl:L:ne:rgO:B:f:I:D:", long_options, nullptr)) != -1) {
        switch (ch) {
        case 'h':
            print_help(0);
            break;
        case 'd':
            debug_mode = 1;
            eshkol_set_logger_level(ESHKOL_DEBUG);
            break;
        case 'a':
            dump_ast = 1;
            break;
        case 'i':
            dump_ir = 1;
            break;
        case 'o':
            output = optarg;
            break;
        case 'c':
            compile_only = 1;
            break;
        case 's':
            shared_lib = 1;
            compile_only = 1;  // Library mode implies compile to object file
            break;
        case 'w':
            wasm_output = 1;
            break;
        case 'l':
            linked_libs.push_back(optarg);
            break;
        case 'L':
            lib_paths.push_back(optarg);
            break;
        case 'I':
            include_paths.push_back(optarg);
            break;
        case 'D':
            /* Accepted for CMake-style object builds; preprocessor defines are
             * not part of Eshkol source semantics yet. */
            break;
        case 'n':
            no_stdlib = 1;
            break;
        case 'e':
            eval_expr = optarg;
            break;
        case 'r':
            run_mode = 1;
            break;
        case 'g':
            debug_info = 1;
            break;
        case 'O':
            opt_level = atoi(optarg);
            if (opt_level < 0 || opt_level > 3) {
                fprintf(stderr, "Invalid optimization level: %s (must be 0-3)\n", optarg);
                return 1;
            }
            opt_level_explicit = true;
            break;
        case 256:
            strict_types = 1;
            break;
        case 257:
            unsafe_mode = 1;
            break;
        case 258:
            printf("Eshkol Compiler v%s\n", ESHKOL_VER);
            return 0;
        case 264: {
            /* Authoritative capability introspection (GPU-LLM brief §5).
             * Machine-parseable KEY=VALUE lines so a mesh deploy can assert
             * a build has the capabilities a workload needs BEFORE scheduling
             * it — the version-drift failure (old-donkey shipped a build
             * without gpu-matmul) becomes a one-line precondition check:
             *   eshkol-run --features | grep -q '^gpu=on'
             * Values are compile-time facts about THIS binary. */
            printf("version=%s\n", ESHKOL_VER);
#if defined(ESHKOL_LLVM_BACKEND_ENABLED)
            printf("llvm-backend=on\n");
#else
            printf("llvm-backend=off\n");
#endif
#if defined(ESHKOL_GPU_ENABLED)
            printf("gpu=on\n");
#else
            printf("gpu=off\n");
#endif
#if defined(ESHKOL_GPU_METAL_ENABLED)
            printf("gpu-metal=on\n");
#else
            printf("gpu-metal=off\n");
#endif
#if defined(ESHKOL_GPU_CUDA_ENABLED) || defined(ESHKOL_GPU_CUDA)
            printf("gpu-cuda=on\n");
#else
            printf("gpu-cuda=off\n");
#endif
#if defined(ESHKOL_BLAS_ENABLED)
            printf("blas=on\n");
#else
            printf("blas=off\n");
#endif
#if defined(ESHKOL_XLA_ENABLED)
            printf("xla=on\n");
#else
            printf("xla=off\n");
#endif
            /* Tensor element dtypes this build supports (brief §1, ESH-0020).
             * Storage is f64 bit patterns; dtype records logical precision and
             * tensor-cast applies precision reduction. Deploys can gate on
             * e.g. 'tensor-dtypes=.*f16'. */
            printf("tensor-dtypes=f64,f32,f16,bf16,i8\n");
            return 0;
        }
        case 259:
            compile_only = 1;
            break;
        case 'f':
            /* Accept -fPIC/-f PIC for build-system compatibility. LLVM object
             * emission already produces relocatable code for this path. */
            break;
        case 'B':
            eskb_output_path = optarg;
            break;
        case 260:
            profile_name = optarg;
            break;
        case 261:
            target_triple = optarg;
            break;
        case 262:
            if (!optarg || !*optarg) {
                fprintf(stderr, "--require-vm-entry requires a non-empty name\n");
                return 1;
            }
            required_vm_entries.emplace_back(optarg);
            break;
        case 263:
            if (!optarg || !*optarg) {
                fprintf(stderr, "--require-vm-entry-zero-arg requires a non-empty name\n");
                return 1;
            }
            required_zero_arg_vm_entries.emplace_back(optarg);
            break;
        case 265:
            if (!optarg || !*optarg) {
                fprintf(stderr, "--emit-depfile requires a non-empty path\n");
                return 1;
            }
            depfile_path = optarg;
            break;
        default:
            print_help(1);
        }
    }

    // Default optimization level for GENERATED code.
    //
    // A CMake Release build of the compiler does NOT imply optimized *emitted*
    // Eshkol code: the backend optimization plane is independent of how the
    // compiler itself was built (ADR 0007, "Generated code has an independent
    // optimization plane"). Historically this defaulted to O0, so even
    // `eshkol-run file.esk -o bin` produced unoptimized artifacts.
    //
    // Optimize the paths that PRODUCE a persisted artifact (-o AOT binary,
    // -c object, --shared-lib), where the user is building something to keep
    // and run repeatedly, so the "sleeper" O0 default no longer ships
    // unoptimized binaries. Leave the ephemeral/interactive paths (plain run,
    // -r JIT, -e eval, REPL) and debug builds (-g) at O0 for fast turnaround:
    // for those, whole-module optimization (which folds in referenced stdlib)
    // costs far more compile time than a single ephemeral execution saves.
    // An explicit -O<n> always wins on every path, including -O0 to opt out
    // and -O3 for a performance artifact.
    //
    // AD/gradient correctness is unaffected: the default pipeline enables no
    // fast-math or reassociation; only inlining/DCE/vectorization change.
    if (!opt_level_explicit) {
        const bool builds_artifact =
            (output != nullptr) || compile_only || shared_lib;
        opt_level = (builds_artifact && !debug_info) ? 2 : 0;
    }

    {
        eshkol::profile::Selection selection;

        if (profile_name) {
            const auto* profile = eshkol::profile::find(profile_name);
            if (!profile) {
                fprintf(stderr,
                        "Unknown execution profile: %s\nSupported profiles: %s\n",
                        profile_name,
                        eshkol::profile::supported_names().c_str());
                return 1;
            }
            selection.requested = profile->id;
            selection.explicit_request = true;
        } else if (wasm_output) {
            selection.requested = eshkol::profile::ExecutionProfile::HostedWasm;
        }

        selection.explicit_target_triple = target_triple;
        selection.compile_only = compile_only != 0;
        selection.shared_lib = shared_lib != 0;
        selection.wasm_flag = wasm_output != 0;
        selection.no_stdlib = no_stdlib != 0;
        selection.eval_mode = eval_expr != nullptr;
        selection.run_mode = run_mode != 0;
        selection.has_eskb_output = eskb_output_path != nullptr;
        selection.has_linked_libs = !linked_libs.empty();

        const auto resolved = eshkol::profile::resolve(selection);
        if (!resolved.error.empty()) {
            fprintf(stderr, "%s\n", resolved.error.c_str());
            return 1;
        }

        compile_only = resolved.compile_only ? 1 : 0;
        no_stdlib = resolved.no_stdlib ? 1 : 0;
        wasm_output = resolved.wasm_output ? 1 : 0;
        freestanding_native_profile =
            resolved.profile &&
            resolved.profile->freestanding &&
            resolved.profile->backend == eshkol::profile::Backend::Native;
        vm_only_profile = resolved.vm_only;
        embedded_vm_profile = resolved.embedded_vm;
        eshkol_set_target(resolved.target_triple);
        eshkol_set_freestanding_codegen(freestanding_native_profile ? 1 : 0);
    }

    if (!required_vm_entries.empty() && !vm_only_profile) {
        fprintf(stderr, "--require-vm-entry requires a VM profile with --emit-eskb\n");
        return 1;
    }
    if (!required_zero_arg_vm_entries.empty() && !vm_only_profile) {
        fprintf(stderr, "--require-vm-entry-zero-arg requires a VM profile with --emit-eskb\n");
        return 1;
    }

    if (!include_paths.empty()) {
        std::string merged;
        const char* existing_path = std::getenv("ESHKOL_PATH");
        if (existing_path && *existing_path) {
            merged = existing_path;
        }
        for (char* path : include_paths) {
            if (!path || !*path) continue;
            if (!merged.empty()) merged.push_back(eshkol_path_separator);
            merged += path;
        }
#ifdef _WIN32
        _putenv_s("ESHKOL_PATH", merged.c_str());
#else
        setenv("ESHKOL_PATH", merged.c_str(), 1);
#endif
    }

    // If we have an eval expression, use JIT mode
    if (eval_expr) {
        // Initialize runtime system
        if (eshkol_runtime_init() != 0) {
            eshkol_error("Failed to initialize runtime system");
            return 1;
        }
        eshkol_init_limits_from_env();

        // Create JIT context
        eshkol::ReplJITContext jit_ctx;

        // Load stdlib for eval mode
        if (!no_stdlib) {
            jit_ctx.loadStdlib();
        }

        // Parse the expression using istringstream (no temp file needed)
        std::string eval_input = std::string(eval_expr) + "\n";
        std::istringstream eval_stream(eval_input);

        // Parse and execute ALL expressions from the input (not just one)
        eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(eval_stream);
        bool parsed_any = false;

        while (ast.type != ESHKOL_INVALID) {
            parsed_any = true;

            if (debug_mode) {
                printf("=== AST ===\n");
                eshkol_ast_pretty_print(&ast, 0);
                printf("===========\n");
            }

            // Execute the AST (definitions don't print, expressions do via display)
            bool is_define = (ast.type == ESHKOL_OP &&
                (ast.operation.op == ESHKOL_DEFINE_OP ||
                 ast.operation.op == ESHKOL_REQUIRE_OP ||
                 ast.operation.op == ESHKOL_IMPORT_OP));

            // Check if it's a display/print call (don't wrap these)
            bool is_output_call = false;
            if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_CALL_OP &&
                ast.operation.call_op.func && ast.operation.call_op.func->type == ESHKOL_VAR) {
                const char* name = ast.operation.call_op.func->variable.id;
                if (name && (strcmp(name, "display") == 0 || strcmp(name, "newline") == 0 ||
                             strcmp(name, "print") == 0 || strcmp(name, "write") == 0)) {
                    is_output_call = true;
                }
            }

            // Execute directly (don't auto-wrap with display - let user control output)
            jit_ctx.execute(&ast);

            // Parse next expression
            ast = eshkol_parse_next_ast_from_stream(eval_stream);
        }

        if (!parsed_any) {
            eshkol_error("Failed to parse expression: %s", eval_expr);
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }

        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
        return 0;
    }

    // If run mode, JIT execute all expressions from file(s)
    if (run_mode) {
        if (optind == argc) {
            eshkol_error("No input file specified for --run mode");
            print_help(1);
        }

        const char* language_coverage_trace_dir =
            std::getenv("ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR");
        const bool language_coverage_tracing =
            language_coverage_trace_dir && *language_coverage_trace_dir;
        if (argc - optind == 1 && !debug_mode && !dump_ast && !dump_ir &&
            !language_coverage_tracing) {
            if (auto cached_status = tryRunFromPersistentJitCache(
                    argv[0], argv[optind], no_stdlib, strict_types, unsafe_mode,
                    opt_level, target_triple, linked_libs, lib_paths, include_paths)) {
                return *cached_status;
            }
        } else {
            jitCacheTrace("bypass", language_coverage_tracing
                                        ? "language-coverage-tracing"
                                        : "multi-file-or-debug");
        }

        // Initialize runtime system
        if (eshkol_runtime_init() != 0) {
            eshkol_error("Failed to initialize runtime system");
            return 1;
        }
        eshkol_init_limits_from_env();

        // Create JIT context
        eshkol::ReplJITContext jit_ctx;

        // Load stdlib for run mode
        if (!no_stdlib) {
            jit_ctx.loadStdlib();
        }

        // Process each input file
        for (int i = optind; i < argc; i++) {
            std::string filepath = argv[i];

            if (!std::filesystem::exists(filepath)) {
                eshkol_error("File not found: %s", filepath.c_str());
                continue;
            }

            std::ifstream file(filepath);
            if (!file.is_open()) {
                eshkol_error("Failed to open file: %s", filepath.c_str());
                continue;
            }

            if (debug_mode) {
                eshkol_info("JIT running: %s", filepath.c_str());
            }

            // Set source context so runtime type errors carry a
            // "file:line:col:" prefix (v1.3 source-span errors), matching
            // the AOT path. Read the file text once for diagnostics.
            {
                std::ifstream src_stream(filepath);
                if (src_stream.is_open()) {
                    std::string src_text(
                        (std::istreambuf_iterator<char>(src_stream)),
                        std::istreambuf_iterator<char>());
                    eshkol_set_source_context(filepath.c_str(), src_text.c_str());
                }
            }

            // Architectural fix for the per-form JIT module boundary class:
            //
            // Previously this loop compiled and executed each top-level form
            // as its own LLVM module, matching REPL interactive semantics.
            // That forced cross-module references for every define-then-use
            // pattern (e.g. (define hw ...) then (parallel-map hw ...)),
            // which tripped thread-pool/closure-capture state issues and
            // hung benches that work cleanly in compiled mode.
            //
            // For file execution the right architectural choice is a single
            // module containing every top-level form — identical to what
            // compiled mode does — then one executeBatch call. require/
            // import forms are still processed inline by the batch path
            // (they load submodules synchronously).
            std::vector<eshkol_ast_t> file_asts;
            file_asts.reserve(64);
            {
                eshkol_ast_t ast = eshkol_parse_next_ast(file);
                while (ast.type != ESHKOL_INVALID) {
                    if (debug_mode) {
                        printf("=== AST ===\n");
                        eshkol_ast_pretty_print(&ast, 0);
                        printf("===========\n");
                    }
                    file_asts.push_back(ast);
                    ast = eshkol_parse_next_ast(file);
                }
            }

            // Process require/import up front so their module contents are
            // available as externally-resolvable symbols before batch codegen.
            // Everything else gets batched into a single module.
            std::vector<eshkol_ast_t> batch;
            batch.reserve(file_asts.size());
            for (auto& ast : file_asts) {
                bool is_load = (ast.type == ESHKOL_OP &&
                    (ast.operation.op == ESHKOL_REQUIRE_OP ||
                     ast.operation.op == ESHKOL_IMPORT_OP));
                if (is_load) {
                    try { jit_ctx.execute(&ast); } catch (...) { /* continue */ }
                } else {
                    batch.push_back(ast);
                }
            }

            if (!batch.empty()) {
                try {
                    jit_ctx.executeBatch(batch, /*silent=*/false);
                } catch (const std::exception& e) {
                    eshkol_error("JIT batch execution failed: %s", e.what());
                    // Quirk #6 (2026-04-23): surface the failure in the
                    // exit status. The -r path used to return 0 even
                    // when batch codegen failed (LLVM verification
                    // error, module unwrap, etc.), which made compile
                    // failures look like "program ran but printed
                    // nothing" to the user. executeBatch now throws
                    // on codegen failure; signal the caller via non-
                    // zero exit so CI / bench / scripts see it.
                    file.close();
                    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
                    return 1;
                }
            }

            file.close();
        }

        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
        return 0;
    }

    if (optind == argc) print_help(1);

    // Initialize runtime system (signal handlers, shutdown hooks)
    if (eshkol_runtime_init() != 0) {
        eshkol_error("Failed to initialize runtime system");
        return 1;
    }

    // Initialize resource limits from environment variables
    // Supported: ESHKOL_MAX_HEAP, ESHKOL_TIMEOUT_MS, ESHKOL_MAX_STACK, etc.
    eshkol_init_limits_from_env();

    // Helper function to check string suffix (C++17 compatible)
    auto ends_with = [](const std::string& str, const std::string& suffix) {
        if (suffix.size() > str.size()) return false;
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    for (; optind < argc; ++optind) {
        std::string tmp = (const char*) argv[optind];
        if (ends_with(tmp, ".esk"))
            source_files.push_back(argv[optind]);
        else if (ends_with(tmp, ".o"))
            compiled_files.push_back(argv[optind]);
    }

    if (vm_only_profile) {
        if (source_files.empty()) {
            eshkol_error("VM profile requires an input .esk source file");
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }
        if (source_files.size() != 1 || !compiled_files.empty()) {
            eshkol_error("VM profiles currently support exactly one .esk source file and no .o inputs");
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }

        std::ifstream eskb_src(source_files[0]);
        if (!eskb_src.is_open()) {
            eshkol_error("Failed to open file: %s", source_files[0]);
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }

        std::string eskb_source((std::istreambuf_iterator<char>(eskb_src)),
                                std::istreambuf_iterator<char>());
        int eskb_result = embedded_vm_profile
                              ? eshkol_emit_eskb_embedded(eskb_source.c_str(), eskb_output_path)
                              : eshkol_emit_eskb(eskb_source.c_str(), eskb_output_path);
        if (eskb_result != 0) {
            eshkol_error("ESKB emission failed for profile %s",
                         profile_name ? profile_name : "hosted-vm");
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }

        const bool require_vm_admission =
            !required_vm_entries.empty() || !required_zero_arg_vm_entries.empty();
        if (require_vm_admission) {
            std::ifstream emitted_eskb(eskb_output_path, std::ios::binary);
            if (!emitted_eskb.is_open()) {
                eshkol_error("Failed to open emitted ESKB for admission: %s",
                             eskb_output_path);
                std::error_code remove_ec;
                std::filesystem::remove(eskb_output_path, remove_ec);
                eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
                return 1;
            }

            std::vector<uint8_t> emitted_bytes(
                (std::istreambuf_iterator<char>(emitted_eskb)),
                std::istreambuf_iterator<char>());
            if (emitted_bytes.empty()) {
                eshkol_error("ESKB admission failed: emitted bytecode is empty");
                std::error_code remove_ec;
                std::filesystem::remove(eskb_output_path, remove_ec);
                eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
                return 1;
            }

            EshkolVmLoadOptions admission_options;
            if (eshkol_vm_default_load_options(&admission_options) != 0) {
                eshkol_error("ESKB admission failed: could not initialize VM load options");
                std::error_code remove_ec;
                std::filesystem::remove(eskb_output_path, remove_ec);
                eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
                return 1;
            }
            if (embedded_vm_profile) {
                admission_options.native_policy = ESHKOL_VM_NATIVE_POLICY_HOST_ONLY;
                admission_options.reject_string_constants = 1;
                admission_options.reject_desktop_native_calls = 1;
            }

            std::vector<EshkolVmFunctionRequirement> required_zero_arg_entries;
            required_zero_arg_entries.reserve(required_zero_arg_vm_entries.size());
            for (const std::string& entry : required_zero_arg_vm_entries) {
                EshkolVmFunctionRequirement requirement;
                requirement.name = entry.c_str();
                requirement.n_params = 0;
                requirement.max_locals = -1;
                requirement.max_code_len = -1;
                requirement.require_no_upvalues = 1;
                required_zero_arg_entries.push_back(requirement);
            }
            if (!required_zero_arg_entries.empty()) {
                admission_options.required_function_metadata =
                    required_zero_arg_entries.data();
                admission_options.required_function_metadata_count =
                    static_cast<int>(required_zero_arg_entries.size());
            }

            EshkolVmHandle* admission_vm =
                eshkol_vm_load_chunk_with_options(emitted_bytes.data(),
                                                  emitted_bytes.size(),
                                                  &admission_options);
            if (!admission_vm) {
                eshkol_error("ESKB admission failed for profile %s",
                             profile_name ? profile_name : "hosted-vm");
                std::error_code remove_ec;
                std::filesystem::remove(eskb_output_path, remove_ec);
                eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
                return 1;
            }

            for (const std::string& entry : required_vm_entries) {
                if (eshkol_vm_has_function(admission_vm, entry.c_str()) != 1) {
                    eshkol_error("ESKB admission failed: missing required VM entry '%s'",
                                 entry.c_str());
                    eshkol_vm_destroy(admission_vm);
                    std::error_code remove_ec;
                    std::filesystem::remove(eskb_output_path, remove_ec);
                    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
                    return 1;
                }
            }
            eshkol_vm_destroy(admission_vm);
        }

        printf("[ESKB] Emitted bytecode to %s\n", eskb_output_path);
        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
        return 0;
    }

    // First pass: Load source files to check for stdlib requirements
    // (We'll process requires after potentially adding stdlib.o)
    eshkol_reset_parse_errors();
    for (const auto &source_file : source_files) {
        load_file_asts(source_file, asts, debug_mode);
    }
    if (g_source_parse_failed || eshkol_parse_had_error()) {
        eshkol_error("Source parsing failed; refusing to compile a truncated program");
        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
        return 1;
    }

    // Bug Y (Noesis 2026-04-30): stdlib auto-loading was previously
    // gated on `(require stdlib)` appearing in the source.  JIT mode
    // (`eshkol-run -r`) auto-loads stdlib regardless — the REPL JIT
    // discovers symbols from `stdlib.bc` at startup — so the same
    // file would print correct output under JIT and fail with
    // "Unknown function: length / reverse / append / assoc / filter
    // / for-each / …" under AOT.  Anyone running plain
    // `eshkol-run foo.esk` was getting the worst-of-both-worlds
    // surprise: JIT-flavoured semantics in the docs, AOT-flavoured
    // brittleness on the command line.
    //
    // Restore the documented behaviour: `--no-stdlib` opts OUT of
    // auto-load; the default IS auto-load.  `(require stdlib)`
    // remains valid (and idempotent — has_stdlib check below).
    //
    // ESH-0220: this gate MUST use requires_stdlib_module_explicitly(),
    // not requires_stdlib(). The latter also matches any dotted
    // `(require core.*)`, which meant a file that required a core.*
    // submodule directly (without ALSO explicitly requiring "stdlib")
    // never got the synthetic `(require stdlib)` below — silently
    // losing stdlib.esk's own top-level definitions (__keyword-arg,
    // __keyword-args-validate, __keyword-member?, …) for the rest of
    // the compilation unit. See requires_stdlib_module_explicitly()'s
    // doc comment for the full story.
    bool need_stdlib = !no_stdlib && (!shared_lib);
    if (need_stdlib && !requires_stdlib_module_explicitly(asts)) {
        // The source didn't `(require stdlib)` explicitly, but the
        // user expects stdlib to be available.  Synthesize a top-of-
        // module require so process_requires() handles it
        // identically to a user-written `(require stdlib)`.
        eshkol_ast_t req_ast = {};
        req_ast.type = ESHKOL_OP;
        req_ast.operation.op = ESHKOL_REQUIRE_OP;
        req_ast.operation.require_op.num_modules = 1;
        req_ast.operation.require_op.module_names = (char**)malloc(sizeof(char*));
        req_ast.operation.require_op.module_names[0] = strdup("stdlib");
        req_ast.operation.require_op.import_prefixes = nullptr;
        req_ast.operation.require_op.import_except_names = nullptr;
        req_ast.operation.require_op.num_import_except_names = nullptr;
        req_ast.line = 0;
        req_ast.column = 0;
        asts.insert(asts.begin(), req_ast);
    }

    // Auto-link stdlib.o if the source requires stdlib and we're not in library mode.
    //
    // WASM: never auto-link the native stdlib.o. It is a wasm-incompatible
    // native object, and treating it as "pre-compiled" turns every stdlib
    // function into an `env.*` import that no JS glue can satisfy — the module
    // then fails to instantiate the moment the program actually uses a stdlib
    // function. For wasm we instead inline the stdlib source (below) and let
    // the wasm dead-strip (internalize + globalDCE in the emit path) drop
    // whatever the program does not reach, yielding a small self-contained module.
    if (!shared_lib && !wasm_output && requires_stdlib(asts)) {
        // Check if stdlib.o is already in compiled_files
        bool has_stdlib = false;
        for (const auto& obj_file : compiled_files) {
            std::string filename = std::filesystem::path(obj_file).filename().string();
            if (filename == "stdlib.o" || filename == "libstdlib.o") {
                has_stdlib = true;
                break;
            }
        }

        if (!has_stdlib) {
            // Try to find stdlib.o automatically
            std::string stdlib_path = find_stdlib_object(lib_paths);
            if (!stdlib_path.empty()) {
                eshkol_info("Auto-linking pre-compiled stdlib: %s", stdlib_path.c_str());
                compiled_files.push_back(strdup(stdlib_path.c_str()));
            }
        }
    }

    // Detect pre-compiled libraries and discover ALL their sub-modules.
    // Every module recursively required by a pre-compiled library is also pre-compiled.
    // Skipped for wasm: see the auto-link comment above — wasm inlines stdlib
    // from source and dead-strips it rather than importing a native .o.
    if (g_lib_dir.empty()) g_lib_dir = find_lib_dir();
    for (const auto& obj_file : compiled_files) {
        if (wasm_output) break;
        std::string filename = std::filesystem::path(obj_file).filename().string();
        if (filename == "stdlib.o" || filename == "libstdlib.o") {
            eshkol_info("Detected pre-compiled stdlib: %s", obj_file);
            // Recursively discover all modules included in stdlib.
            // collect_all_submodules parses lib/stdlib.esk to build the
            // set — if that source file is missing or unreadable (e.g.
            // stdlib.o was installed without the accompanying .esk
            // tree), we end up with an empty precompiled set and
            // process_requires will then try to load core.* modules
            // from source instead, doubling up symbols at link time.
            // Fail loudly up-front instead of silently producing a bad
            // binary.
            size_t before = precompiled_modules.size();
            collect_all_submodules("stdlib", precompiled_modules, g_lib_dir);
            size_t added = precompiled_modules.size() - before;
            eshkol_info("Pre-compiled modules: %zu total (+%zu from stdlib)",
                        precompiled_modules.size(), added);
            if (added == 0) {
                eshkol_error(
                    "stdlib.o is linked but no .esk sources were found under "
                    "lib_dir=%s — precompiled-module detection is empty, which "
                    "will produce duplicate symbols at link time. Check that "
                    "the lib/ tree is installed alongside stdlib.o.",
                    g_lib_dir.c_str());
            }
            // Tell codegen that stdlib is being used (for homoiconic display support)
            eshkol_set_uses_stdlib(1);
        }
        // Future: detect other pre-compiled libraries by naming convention
    }

    // Second pass: Process require statements now that we know about precompiled modules
    // #248: capture agent-FFI usage BEFORE process_requires expands
    // (require agent.…) into the inlined module ASTs, after which the
    // top-level require op is gone and the AST scanner can no longer
    // tell agent-using programs from plain ones.
    std::string agent_scan_base_dir = ".";
    if (!source_files.empty()) {
        std::string p = std::filesystem::path(source_files[0]).parent_path().string();
        if (!p.empty()) agent_scan_base_dir = p;
    }
    bool needs_agent_ffi = requires_agent_ffi(asts, agent_scan_base_dir);

    for (const auto &source_file : source_files) {
        std::filesystem::path source_path(source_file);
        std::string base_dir = source_path.parent_path().string();
        if (base_dir.empty()) base_dir = ".";

        // Process require statements (new module system)
        process_requires(asts, base_dir, debug_mode);

        // Process imports in the loaded ASTs (legacy)
        process_imports(asts, base_dir, debug_mode);
    }

    // Handle AST dumping if requested
    if (dump_ast && !source_files.empty()) {
        std::string ast_filename;
        if (output) {
            ast_filename = std::string(output) + ".ast";
        } else {
            std::string first_source = source_files[0];
            size_t last_slash = first_source.find_last_of("/\\");
            size_t last_dot = first_source.find_last_of('.');
            std::string base_name;
            if (last_slash != std::string::npos) {
                if (last_dot != std::string::npos && last_dot > last_slash) {
                    base_name = first_source.substr(last_slash + 1, last_dot - last_slash - 1);
                } else {
                    base_name = first_source.substr(last_slash + 1);
                }
            } else {
                if (last_dot != std::string::npos) {
                    base_name = first_source.substr(0, last_dot);
                } else {
                    base_name = first_source;
                }
            }
            ast_filename = base_name + ".ast";
        }

        std::ofstream ast_file(ast_filename);
        if (ast_file.is_open()) {
            for (const auto& ast : asts) {
                ast_file << "=== AST Node ===\n";
                eshkol_ast_pretty_print(&ast, 0);
                ast_file << "=================\n\n";
            }
            ast_file.close();
            eshkol_info("AST dumped to: %s", ast_filename.c_str());
        } else {
            eshkol_error("Failed to open AST file: %s", ast_filename.c_str());
        }
    }

    // ESH-0103 phase timing (gated on ESHKOL_PHASE_TIME)
    const bool esh0103_phase_time = (std::getenv("ESHKOL_PHASE_TIME") != nullptr);
    auto esh0103_now = []() { return std::chrono::steady_clock::now(); };
    auto esh0103_report = [&](const char* name, std::chrono::steady_clock::time_point a,
                              std::chrono::steady_clock::time_point b) {
        if (esh0103_phase_time) {
            double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
            fprintf(stderr, "[PHASE] %-22s %10.2f ms\n", name, ms);
        }
    };
    auto esh0103_t_own0 = esh0103_now();

    // Run ownership analysis before code generation
    if (!asts.empty()) {
        eshkol_info("Running ownership analysis...");
        if (!g_ownership_analyzer.analyze(asts)) {
            eshkol_error("Ownership analysis failed:");
            g_ownership_analyzer.printErrors();
            return 1;
        }
        if (debug_mode) {
            eshkol_info("Ownership analysis passed");
        }
    }

    auto esh0103_t_own1 = esh0103_now();
    esh0103_report("ownership", esh0103_t_own0, esh0103_t_own1);

    // Run escape analysis for allocation decisions
    if (!asts.empty()) {
        eshkol_info("Running escape analysis...");
        g_escape_analyzer.analyze(asts);
        if (debug_mode) {
            g_escape_analyzer.printAnalysis();
        }
    }
    auto esh0103_t_escape1 = esh0103_now();
    esh0103_report("escape", esh0103_t_own1, esh0103_t_escape1);

    // Generate LLVM IR if we have ASTs and need compilation or IR output
    // Default behavior is to compile to executable unless only AST dump is requested
    if (!asts.empty()) {
        // Determine module name from first source file or use default
        std::string module_name = "eshkol_module";
        if (!source_files.empty()) {
            std::string source_file = source_files[0];
            size_t last_slash = source_file.find_last_of("/\\");
            size_t last_dot = source_file.find_last_of('.');
            if (last_slash != std::string::npos) {
                if (last_dot != std::string::npos && last_dot > last_slash) {
                    module_name = source_file.substr(last_slash + 1, last_dot - last_slash - 1);
                } else {
                    module_name = source_file.substr(last_slash + 1);
                }
            } else {
                if (last_dot != std::string::npos) {
                    module_name = source_file.substr(0, last_dot);
                } else {
                    module_name = source_file;
                }
            }
        }
        
        // Apply type system flags to global config
        if (strict_types || unsafe_mode) {
            eshkol_config_t cfg = *eshkol_config_get();
            cfg.strict_types = strict_types;
            cfg.unsafe_mode = unsafe_mode;
            eshkol_config_set(&cfg);
        }

        eshkol_info("Generating LLVM IR for module: %s", module_name.c_str());

        // LLVM OPTIMIZATION: Set optimization level before compilation
        if (opt_level > 0) {
            eshkol_set_optimization_level(opt_level);
        }

        // DWARF DEBUG INFO: Enable debug info before IR generation if -g flag was set
        if (debug_info && !source_files.empty()) {
            // Resolve the source file to an absolute path for DWARF
            std::filesystem::path abs_source = std::filesystem::absolute(source_files[0]);
            eshkol_enable_debug_info(abs_source.string().c_str());
        }

        // Set source context for structured error messages
        if (!source_files.empty()) {
            std::ifstream src_stream(source_files[0]);
            if (src_stream.is_open()) {
                std::string src_text((std::istreambuf_iterator<char>(src_stream)),
                                     std::istreambuf_iterator<char>());
                eshkol_set_source_context(source_files[0], src_text.c_str());
            }
        }

        // Generate LLVM IR (use library mode if --shared-lib flag is set)
        auto esh0103_t_irgen0 = esh0103_now();
        LLVMModuleRef llvm_module;
        const bool library_codegen = shared_lib || freestanding_native_profile;
        if (library_codegen) {
            eshkol_info("Using library mode (no main function)");
            llvm_module = eshkol_generate_llvm_ir_library(
                asts.data(),
                asts.size(),
                module_name.c_str()
            );
        } else {
            llvm_module = eshkol_generate_llvm_ir(
                asts.data(),
                asts.size(),
                module_name.c_str()
            );
        }
        
        auto esh0103_t_irgen1 = esh0103_now();
        esh0103_report("irgen", esh0103_t_irgen0, esh0103_t_irgen1);

        if (!llvm_module) {
            eshkol_error("Failed to generate LLVM IR");
            return 1;
        }
        
        // Handle different output modes
        if (dump_ir) {
            // Dump IR to file
            std::string ir_filename;
            if (output) {
                ir_filename = std::string(output) + ".ll";
            } else {
                ir_filename = module_name + ".ll";
            }
            
            eshkol_info("Dumping LLVM IR to: %s", ir_filename.c_str());
            if (eshkol_dump_llvm_ir_to_file(llvm_module, ir_filename.c_str()) != 0) {
                eshkol_error("Failed to dump LLVM IR to file");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        }
        
        if (debug_mode) {
            eshkol_info("Generated LLVM IR:");
            eshkol_print_llvm_ir(llvm_module);
        }
        
        /* Emit ESKB bytecode if requested */
        if (eskb_output_path) {
            /* Read source file for bytecode compilation */
            FILE* eskb_src_f = source_files.empty() ? NULL : fopen(source_files[0], "r");
            if (eskb_src_f) {
                fseek(eskb_src_f, 0, SEEK_END);
                long eskb_len = ftell(eskb_src_f);
                fseek(eskb_src_f, 0, SEEK_SET);
                /* P1: ftell returns -1 on a pipe/FIFO/error; malloc(-1+1)=malloc(0)
                   then fread(.., (size_t)-1, ..) is a massive heap overflow. Guard
                   the length and NUL-terminate at the actual bytes read. */
                if (eskb_len < 0) { fclose(eskb_src_f); eskb_src_f = NULL; }
                char* eskb_source = eskb_src_f ? (char*)malloc((size_t)eskb_len + 1) : NULL;
                if (eskb_source) {
                    size_t eskb_nread = fread(eskb_source, 1, (size_t)eskb_len, eskb_src_f);
                    eskb_source[eskb_nread] = 0;
                    fclose(eskb_src_f);
                    int eskb_result = eshkol_emit_eskb(eskb_source, eskb_output_path);
                    if (eskb_result == 0) {
                        printf("[ESKB] Emitted bytecode to %s\n", eskb_output_path);
                    } else {
                        fprintf(stderr, "WARNING: ESKB emission failed\n");
                    }
                    free(eskb_source);
                } else {
                    fclose(eskb_src_f);
                }
            }
        }

        if (compile_only) {
            // Compile to object file
            std::string obj_filename;
            if (output) {
                obj_filename = std::string(output);
                if (obj_filename.size() < 2 ||
                    obj_filename.substr(obj_filename.size() - 2) != ".o") {
                    obj_filename += ".o";
                }
            } else {
                obj_filename = module_name + ".o";
            }

            eshkol_info("Compiling to object file: %s", obj_filename.c_str());
            auto esh0103_t_obj0 = esh0103_now();
            if (eshkol_compile_llvm_ir_to_object(llvm_module, obj_filename.c_str()) != 0) {
                eshkol_error("Object file compilation failed");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
            esh0103_report("emit-object", esh0103_t_obj0, esh0103_now());

            // Also emit bitcode for REPL JIT loading (avoids ABI mismatch with addObjectFile)
            std::string bc_filename;
            if (output) {
                bc_filename = std::string(output) + ".bc";
            } else {
                bc_filename = module_name + ".bc";
            }
            eshkol_compile_llvm_ir_to_bitcode(llvm_module, bc_filename.c_str());

            // ESH-0215: --emit-depfile PATH — write a Makefile-format depfile
            // so a build system (ninja DEPFILE, make -include) recompiles this
            // object when the entry file or any transitively load/import/
            // require-reached dependency changes (Noesis BUGS-2026-07-04 #3/#5).
            if (depfile_path) {
                if (source_files.empty()) {
                    eshkol_error("--emit-depfile requires a source file");
                    eshkol_dispose_llvm_module(llvm_module);
                    return 1;
                }
                if (!writeDepfile(depfile_path, obj_filename, source_files[0], include_paths)) {
                    eshkol_dispose_llvm_module(llvm_module);
                    return 1;
                }
                eshkol_info("Wrote depfile: %s", depfile_path);
            }
        } else if (wasm_output) {
            // Compile to WebAssembly
            std::string wasm_filename;
            if (output) {
                wasm_filename = std::string(output);
                // Add .wasm extension if not present
                if (wasm_filename.size() < 5 || wasm_filename.substr(wasm_filename.size() - 5) != ".wasm") {
                    wasm_filename += ".wasm";
                }
            } else {
                wasm_filename = module_name + ".wasm";
            }

            eshkol_info("Compiling to WebAssembly: %s", wasm_filename.c_str());
            if (eshkol_compile_llvm_ir_to_wasm_file(llvm_module, wasm_filename.c_str()) != 0) {
                eshkol_error("WebAssembly compilation failed");
                eshkol_dispose_llvm_module(llvm_module);
                return 1;
            }
        } else if (!compile_only) {
            // Default behavior: compile to executable
            // If we have object files to link (like stdlib.o), compile to temp .o first
            // then link everything together
            std::string exe_name = eshkol::platform::with_executable_suffix(
                output ? std::filesystem::path(output) : std::filesystem::path("a.out")
            ).generic_string();

            if (!compiled_files.empty()) {
                // Compile main program to temp .o, then link with stdlib.o etc.
                std::string temp_obj = exe_name + ".tmp.o";
                eshkol_info("Compiling to temp object: %s", temp_obj.c_str());
                if (eshkol_compile_llvm_ir_to_object(llvm_module, temp_obj.c_str()) != 0) {
                    eshkol_error("Object file compilation failed");
                    eshkol_dispose_llvm_module(llvm_module);
                    return 1;
                }
                compiled_files.push_back(strdup(temp_obj.c_str()));
                // Normalize the final output path now so the later link step
                // uses the platform executable suffix on Windows as well.
                output = strdup(exe_name.c_str());
            } else {
                // No object files to link, compile directly to executable
                eshkol_info("Compiling to executable: %s", exe_name.c_str());

                // Prepare C-style arrays for library paths and libraries
                const char** lib_path_ptrs = nullptr;
                const char** linked_lib_ptrs = nullptr;

                if (!lib_paths.empty()) {
                    lib_path_ptrs = const_cast<const char**>(lib_paths.data());
                }
                if (!linked_libs.empty()) {
                    linked_lib_ptrs = const_cast<const char**>(linked_libs.data());
                }

                if (eshkol_compile_llvm_ir_to_executable(llvm_module, exe_name.c_str(),
                                                       lib_path_ptrs, lib_paths.size(),
                                                       linked_lib_ptrs, linked_libs.size()) != 0) {
                    eshkol_error("Executable compilation failed");
                    eshkol_dispose_llvm_module(llvm_module);
                    return 1;
                }

                // Bug X (Noesis 2026-04-30): the link-branch path below
                // already prints "[eshkol-run] compiled to a.out — run
                // it (./a.out) or use eshkol-run -r" so users with the
                // Lisp-shebang expectation aren't silently confused.
                // The single-file LLVM-direct path here used to skip
                // that notice entirely, which made `eshkol-run foo.esk`
                // look like it had silently produced no output.  Print
                // the same notice here when no -o was given.
                if (!output) {
                    const char* basename_of_exe = std::strrchr(exe_name.c_str(), '/');
                    basename_of_exe = basename_of_exe ? basename_of_exe + 1 : exe_name.c_str();
                    bool default_output_name = (std::strcmp(basename_of_exe, "a.out") == 0
#ifdef _WIN32
                                                || std::strcmp(basename_of_exe, "a.exe") == 0
#endif
                                                );
                    if (default_output_name) {
                        fprintf(stderr,
                                "[eshkol-run] compiled to '%s'. Run it (./%s) or use "
                                "`eshkol-run -r %s` to JIT-execute without producing "
                                "a binary.\n",
                                exe_name.c_str(), exe_name.c_str(),
                                source_files.empty() ? "<file>" : source_files[0]);
                    }
                }
            }
        }
        
        // Clean up
        eshkol_dispose_llvm_module(llvm_module);
    }

    // Process compiled object files if we have them and an output target.
    // Don't link in compile-only mode (-c flag) - user can link manually.
    // Don't link in --wasm mode either: the wasm_output branch above has
    // already produced the .wasm file via the LLVM in-memory codegen path
    // (the WebAssembly target emits a self-contained module).  Falling
    // through to native clang++ link would try to relink stdlib.o (an
    // arm64 / x86_64 native object auto-added at line ~2780) into the
    // wasm output, which fails with "_main … referenced from initial
    // undefines" because (a) the user's --wasm test files are typically
    // module-style libraries with no scheme_main, and (b) the host
    // platform linker can't consume native objects when targeting wasm.
    // Reproducer: tests/web/web_canvas_test.esk and web_extern_test.esk
    // succeeded in eshkol_compile_llvm_ir_to_wasm_file() but the run was
    // marked FAIL because the redundant native link below crashed.
    if (!compiled_files.empty() && output && !compile_only && !wasm_output) {
        std::vector<std::string> link_args;
        link_args.push_back(eshkol::platform::cxx_compiler());
#ifndef _WIN32
        link_args.emplace_back("-fPIE");
#endif

        // Add all object files
        for (const auto &compiled_file : compiled_files) {
            link_args.emplace_back(compiled_file);
        }

        // Add library search paths
        for (const auto &lib_path : lib_paths) {
            link_args.emplace_back("-L" + std::string(lib_path));
        }

        // Add libeshkol-runtime.a (or legacy libeshkol-static.a) for runtime functions.
        std::string runtime_lib = find_runtime_library(lib_paths);
        if (!runtime_lib.empty()) {
            link_args.emplace_back(runtime_lib);
        } else {
            eshkol_error("Could not find libeshkol-runtime.a or legacy libeshkol-static.a");
            eshkol_error("Searched: ./build/, /usr/local/lib/, /opt/homebrew/lib/, and relative to executable");
            eshkol_error("Please install Eshkol properly or build from source");
            return 1;
        }

        // Add linked libraries
        for (const auto &linked_lib : linked_libs) {
            link_args.emplace_back("-l" + std::string(linked_lib));
        }

        // Add BLAS framework/library (required for libeshkol-static.a BLAS functions)
#ifdef __APPLE__
        // Link with Accelerate framework for BLAS (Apple Silicon optimized)
        link_args.emplace_back("-framework");
        link_args.emplace_back("Accelerate");
#endif

        // Add GPU frameworks/libraries (for GPU-accelerated tensor operations)
        // Metal: System framework on macOS, always safe to link (like Accelerate)
        // CUDA: Added from build-config metadata captured by CMake.
#ifdef __APPLE__
        // Metal and MetalPerformanceShaders for GPU on macOS
        link_args.emplace_back("-framework");
        link_args.emplace_back("Metal");
        link_args.emplace_back("-framework");
        link_args.emplace_back("MetalPerformanceShaders");
        link_args.emplace_back("-framework");
        link_args.emplace_back("Foundation");
        link_args.emplace_back("-framework");
        link_args.emplace_back("ImageIO");
        link_args.emplace_back("-framework");
        link_args.emplace_back("CoreGraphics");
        link_args.emplace_back("-framework");
        link_args.emplace_back("CoreFoundation");
        // Add the macOS SDK lib dir so `-lobjc` resolves portably (libobjc.tbd
        // lives in <sdk>/usr/lib). Resolved at runtime, never hardcoded, so a
        // freshly-built binary links on any mac regardless of the builder host.
        {
            const std::string sdk_lib = eshkol::platform::macos_sdk_lib_dir();
            if (!sdk_lib.empty()) {
                link_args.emplace_back("-L" + sdk_lib);
            }
        }
        link_args.emplace_back("-lobjc");  // Objective-C runtime

        // Security framework: required by lib/core/crypto_primitives.c's
        // SecRandomCopyBytes() (eshkol_random_bytes/eshkol_random_hex).
        // CommonCrypto's HMAC/SHA256 calls (eshkol_hmac_sha256/eshkol_sha256)
        // resolve via libSystem with no extra framework needed. This is
        // always linked (not gated behind `(require agent.…)` detection)
        // because crypto_primitives.c is part of the always-linked
        // eshkol-runtime archive now — see Noesis bug report #2 (2026-07-04):
        // previously these symbols lived in the separate, optional
        // eshkol-agent-ffi archive and were only pulled in for programs that
        // required agent.*, which meant a link racing a concurrent rebuild of
        // that (much larger, more frequently rebuilt) archive could see a
        // partial/inconsistent .a and fail with spurious undefined-symbol
        // errors.
        link_args.emplace_back("-framework");
        link_args.emplace_back("Security");
#endif

        append_host_runtime_link_args(link_args);
        append_host_llvm_link_args(link_args);

        // #248: Splice agent-FFI link args when the user's source has
        // any (require agent.…). Empty when the build wasn't
        // configured with libcurl / sqlite3 / pcre2 — in that case
        // calls to qllm_http_get / etc. would still hit the existing
        // "explicit unavailable" stubs at runtime, same as before.
        // Splitting on whitespace is safe because we constructed
        // ESHKOL_HOST_AGENT_FFI_LINK_ARGS from pkg-config and CMake
        // path lookups; user paths with spaces would be a problem,
        // but pkg-config produces system-style absolute paths which
        // are never spaced in practice.
        if (needs_agent_ffi) {
            std::string raw = ESHKOL_HOST_AGENT_FFI_LINK_ARGS;
            if (!raw.empty()) {
                size_t pos = 0;
                while (pos < raw.size()) {
                    size_t end = raw.find(' ', pos);
                    if (end == std::string::npos) end = raw.size();
                    if (end > pos) {
                        link_args.emplace_back(raw.substr(pos, end - pos));
                    }
                    pos = end + 1;
                }
            }
        }

// Set 512 MB main-thread stack so deeply recursive Scheme code
// (e.g. nested letrec / non-tail-recursive helpers in
// tests/tco/nested_tco_test.esk) doesn't overflow the default 8 MB
// macOS / Linux thread stack.  Without this the binary's LC_MAIN
// shows `stacksize 0` (i.e. linker default) on Darwin and the
// recursion-depth check itself segfaults on its own frame push
// once the user stack is exhausted.  llvm_codegen.cpp's parallel
// link path already does this; the path here for pre-compiled
// .o inputs (the common `eshkol-run file.esk -o exe` flow) was
// missing the Darwin branch.
#ifdef _WIN32
        link_args.emplace_back("-fuse-ld=lld");
#ifdef __MINGW32__
        link_args.emplace_back("-Wl,--stack,536870912");
#else
        link_args.emplace_back("-Xlinker");
        link_args.emplace_back("/STACK:536870912");
#endif
#elif defined(__APPLE__)
        link_args.emplace_back("-Wl,-stack_size,0x20000000");
#elif defined(__linux__)
        link_args.emplace_back("-Wl,-z,stack-size=536870912");
        // --export-dynamic: see lib/backend/llvm_codegen.cpp — required
        // for parallel-worker dlsym() fallback.
        link_args.emplace_back("-Wl,--export-dynamic");
        // AArch64 Linux: see lib/backend/llvm_codegen.cpp for rationale —
        // GNU ld 2.38 has bugs handling large user binaries; switch to
        // LLVM's lld which has no such limits. Keep in sync with the
        // JIT link path.
#  if defined(__aarch64__) || defined(__arm64__)
        link_args.emplace_back("-fuse-ld=lld");
#  endif
#endif

        // Add output
        link_args.emplace_back("-o");
        link_args.emplace_back(output);
#if !defined(_WIN32) && !defined(__APPLE__)
        link_args.emplace_back("-lm");
        // libdl: see lib/backend/llvm_codegen.cpp for rationale
        // (dlsym fallback for parallel-worker registration on
        // platforms where the global ctor doesn't fire).
        link_args.emplace_back("-ldl");
#endif

        std::string link_cmd;
        for (const auto& arg : link_args) {
            if (!link_cmd.empty()) {
                link_cmd.push_back(' ');
            }
            link_cmd += eshkol::platform::shell_quote(arg);
        }

        eshkol_info("Linking object files: %s", link_cmd.c_str());
        // Use the shell-free, killable subprocess launcher with a bounded wait.
        // A hung linker (e.g. an unresolved -L search path on a non-builder
        // host) would otherwise block forever via std::system(); the timeout
        // turns that into a reported link failure instead of an infinite hang.
        int result = eshkol::pkg::run_subprocess(
            link_args, nullptr, link_subprocess_timeout_seconds());

        // Clean up temp object files (mkstemp-created files in TMPDIR)
        for (const auto &compiled_file : compiled_files) {
            std::string file(compiled_file);
            if (file.find("eshkol_") != std::string::npos && file.find(".o") != std::string::npos) {
                std::remove(file.c_str());
            }
        }

        if (result == eshkol::pkg::SUBPROCESS_TIMEOUT) {
            eshkol_error("Linking timed out after %u seconds and was aborted "
                         "(set ESHKOL_LINK_TIMEOUT_SECONDS to adjust)",
                         link_subprocess_timeout_seconds());
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }
        if (result != 0) {
            eshkol_error("Linking failed with exit code %d", result);
            // #212(b): distinguish "the library is genuinely missing a
            // symbol" from "the library is mid-rebuild" when the evidence
            // supports it — see warn_if_link_archive_recently_modified().
            warn_if_link_archive_recently_modified(link_args);
            eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
            return 1;
        }

        eshkol_info("Successfully created executable: %s", output);
        // BUG B (Noesis residual audit v3): bare `eshkol-run file.esk`
        // was silently producing a.out with no indication it had
        // compiled (rather than run) the script. Anyone expecting Lisp
        // shebang semantics — `(display …)` to actually display —
        // saw nothing. Print a one-line stderr notice in the default-
        // output case so the surprise is at most one help message
        // long. Skipped when -o was explicitly given (the user knows
        // exactly what they want) and silently absent in -e/-r/
        // --compile paths since they never reach this branch.
        const char* basename_of_output = std::strrchr(output, '/');
        basename_of_output = basename_of_output ? basename_of_output + 1 : output;
        bool default_output_name = (std::strcmp(basename_of_output, "a.out") == 0
#ifdef _WIN32
                                    || std::strcmp(basename_of_output, "a.exe") == 0
#endif
                                    );
        if (default_output_name) {
            fprintf(stderr,
                    "[eshkol-run] compiled to '%s'. Run it (./%s) or use "
                    "`eshkol-run -r %s` to JIT-execute without producing "
                    "a binary.\n",
                    output, output,
                    source_files.empty() ? "<file>" : source_files[0]);
        }
    } else if (!compiled_files.empty() && !compile_only && !wasm_output) {
        // Only warn about unused object files if we're not in
        // compile-only mode and we weren't asked for WASM (the WASM
        // path emits a self-contained module via LLVM in-memory codegen
        // and intentionally bypasses the native link step that would
        // otherwise consume those .o files).
        eshkol_warn("Object files provided but no output specified. Use -o to specify output executable.");
        eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
        return 1;
    }

    // Graceful shutdown - calls all registered shutdown hooks
    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_NONE);
    return 0;
}
