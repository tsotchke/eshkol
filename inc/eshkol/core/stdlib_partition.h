/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U3: Three-tier stdlib partition classification and profile
 * admission rules.
 *
 * Tier 1 — FreestandingPure:
 *   Zero-allocation, zero-OS, strictly `:no-alloc` pure modules safe for
 *   `freestanding-kernel-native` and `freestanding-mcu-native`.
 *
 * Tier 2 — OsUserspaceOalr:
 *   OALR-backed pure language, collection, math, and reasoning stdlib modules
 *   that require the OALR page-hook allocator but no hosted POSIX/OS syscalls.
 *   Rejected in freestanding kernel/MCU profiles to enforce absolute region
 *   isolation between the kernel and OALR.
 *
 * Tier 3 — HostedOs:
 *   Modules requiring hosted OS services (filesystem, sockets, TLS, HTTP,
 *   threads, audio/windowing, agent FFI, SQLite, or the full hosted stdlib).
 *   Rejected in freestanding kernel/MCU profiles.
 */

#ifndef ESHKOL_CORE_STDLIB_PARTITION_H
#define ESHKOL_CORE_STDLIB_PARTITION_H

#include <eshkol/core/execution_profile.h>

#include <string>
#include <string_view>

namespace eshkol::stdlib_partition {

enum class StdlibPartition {
    FreestandingPure,
    OsUserspaceOalr,
    HostedOs,
    UserModule,
};

inline const char* partition_name(StdlibPartition partition) noexcept {
    switch (partition) {
        case StdlibPartition::FreestandingPure: return "freestanding-pure";
        case StdlibPartition::OsUserspaceOalr:  return "os-userspace-oalr";
        case StdlibPartition::HostedOs:         return "hosted-os";
        case StdlibPartition::UserModule:       return "user-module";
    }
    return "unknown";
}

inline std::string normalize_module_key(std::string_view raw) {
    std::string key(raw);
    for (char& c : key) {
        if (c == '\\') c = '/';
    }
    while (key.size() >= 2 && key[0] == '.' && key[1] == '/') {
        key.erase(0, 2);
    }
    if (key.rfind("lib/", 0) == 0) {
        key.erase(0, 4);
    }
    if (key.size() >= 4 && key.compare(key.size() - 4, 4, ".esk") == 0) {
        key.erase(key.size() - 4);
    }
    for (char& c : key) {
        if (c == '/') c = '.';
    }
    return key;
}

inline bool matches_exact_or_prefix(const std::string& key, std::string_view root) {
    if (key == root) return true;
    return key.size() > root.size() &&
           key.compare(0, root.size(), root) == 0 &&
           key[root.size()] == '.';
}

inline StdlibPartition classify_stdlib_module(std::string_view module_name) {
    const std::string key = normalize_module_key(module_name);
    if (key.empty()) return StdlibPartition::UserModule;

    // Tier 1: explicitly verified freestanding-pure stdlib modules only.
    if (key == "core.freestanding_pure" ||
        key == "core.freestanding-pure" ||
        matches_exact_or_prefix(key, "core.freestanding")) {
        return StdlibPartition::FreestandingPure;
    }

    // Tier 3: hosted OS, networking, filesystem, threading, GUI/audio, agent FFI.
    if (key == "stdlib" ||
        matches_exact_or_prefix(key, "agent") ||
        matches_exact_or_prefix(key, "platform") ||
        matches_exact_or_prefix(key, "web") ||
        matches_exact_or_prefix(key, "core.io") ||
        matches_exact_or_prefix(key, "core.files") ||
        matches_exact_or_prefix(key, "core.net") ||
        matches_exact_or_prefix(key, "core.tls") ||
        matches_exact_or_prefix(key, "core.http_server") ||
        matches_exact_or_prefix(key, "core.url") ||
        matches_exact_or_prefix(key, "core.threads") ||
        matches_exact_or_prefix(key, "core.distributed") ||
        matches_exact_or_prefix(key, "core.logging") ||
        matches_exact_or_prefix(key, "core.argparse") ||
        matches_exact_or_prefix(key, "core.plot") ||
        matches_exact_or_prefix(key, "core.capabilities") ||
        key == "core.memory_store" ||
        key == "core.transcript" ||
        key == "core.dbsp.sqlite_changefeed" ||
        key == "core.quantum.moonlab") {
        return StdlibPartition::HostedOs;
    }

    // Tier 2: OALR-backed pure language, collections, math, ML, reasoning stdlib.
    if (matches_exact_or_prefix(key, "core") ||
        matches_exact_or_prefix(key, "math") ||
        matches_exact_or_prefix(key, "ml") ||
        matches_exact_or_prefix(key, "hott") ||
        matches_exact_or_prefix(key, "signal") ||
        matches_exact_or_prefix(key, "random") ||
        matches_exact_or_prefix(key, "tensor") ||
        key == "tensorcore") {
        return StdlibPartition::OsUserspaceOalr;
    }

    return StdlibPartition::UserModule;
}

inline bool partition_allowed_for_profile(
    StdlibPartition partition,
    eshkol::profile::ExecutionProfile profile) noexcept {
    switch (profile) {
        case eshkol::profile::ExecutionProfile::FreestandingKernelNative:
        case eshkol::profile::ExecutionProfile::FreestandingMcuNative:
            return partition == StdlibPartition::FreestandingPure ||
                   partition == StdlibPartition::UserModule;
        case eshkol::profile::ExecutionProfile::HostedNative:
        case eshkol::profile::ExecutionProfile::HostedWasm:
        case eshkol::profile::ExecutionProfile::HostedVm:
        case eshkol::profile::ExecutionProfile::FreestandingVm:
        case eshkol::profile::ExecutionProfile::EmbeddedVm:
            return true;
    }
    return false;
}

}  // namespace eshkol::stdlib_partition

#endif  // ESHKOL_CORE_STDLIB_PARTITION_H
