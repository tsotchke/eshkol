/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * The process-local capability policy: one store, consulted by every gated
 * runtime entry point.
 *
 * This translation unit holds the whole policy — the active flag, the
 * allow-list and the denial-report set — and nothing else. It is deliberately
 * free of host-specific dependencies (no filesystem, no exception machinery)
 * so that every engine links the same store: the native runtime used by JIT
 * and AOT programs, the bytecode VM, and the VM's WebAssembly bundle. A
 * second copy of this state in any engine would mean a policy that holds on
 * one and not the other, so there is exactly one.
 *
 * Policy semantics:
 *   - With no policy installed, every capability is permitted. Programs that
 *     never install one behave exactly as they did before.
 *   - Installing a policy (eshkol_capability_runtime_begin_install followed by
 *     eshkol_capability_runtime_allow for each granted name) makes the policy
 *     active and denies every capability it does not name.
 *   - A policy only ever narrows. Installing one while another is active keeps
 *     the capabilities both name and drops the rest, so the code a policy wraps
 *     cannot widen what it was given. Clearing is refused while a policy is
 *     active, for the same reason: a policy the governed code can switch off is
 *     not a policy. An embedder that owns the process — the host that installed
 *     the first policy, a test harness between cases — resets through
 *     eshkol_capability_runtime_host_reset, which is not reachable from a
 *     program's own code.
 *
 * The capability vocabulary is file-read, file-write, env-read, env-write,
 * subprocess, shell, network, ffi and keychain; see
 * docs/reference/language/capabilities.md.
 *
 * Raising a catchable Eshkol error on a denial is a separate concern and lives
 * with the engine's error machinery (eshkol_capability_enforce for native code,
 * the VM's own raise path for bytecode), because the two engines unwind
 * differently. This file only answers "is it allowed" and records the denial.
 */

#include <cerrno>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_set>

namespace {

std::mutex g_capability_mutex;
bool g_capability_policy_active = false;
std::unordered_set<std::string> g_capability_allow_list;
std::unordered_set<std::string> g_capability_denials_reported;

/* While a second install narrows an active policy, the capabilities the active
 * one granted — the ceiling the new list is intersected with. */
std::unordered_set<std::string> g_capability_previous_allow_list;
bool g_capability_narrowing = false;

bool capability_allows_locked(const char* capability) {
    if (capability == nullptr || capability[0] == '\0') {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    if (!g_capability_policy_active) {
        return true;
    }
    return g_capability_allow_list.find(capability) != g_capability_allow_list.end();
}

} // namespace

/**
 * @brief Deactivate the capability policy, reverting to "everything permitted".
 *
 * Refused while a policy is active: the code a policy governs does not get to
 * remove it. Succeeds when no policy is installed, where it is a no-op, so a
 * program that clears before installing behaves as before.
 *
 * @return 1 when the policy was cleared (or there was none), 0 when an active
 *         policy refused. Thread-safe.
 */
extern "C" int eshkol_capability_runtime_clear() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    if (g_capability_policy_active) return 0;
    g_capability_allow_list.clear();
    return 1;
}

/**
 * @brief Drop any active policy. For the host that owns the process, not for
 *        the code the policy governs.
 *
 * An embedder that installed a policy around a guest computation, or a test
 * harness moving between cases, needs a way back to the unrestricted state.
 * That way is this C entry point, which no Eshkol program can name: declaring
 * it is refused at compile time along with the rest of the policy's own entry
 * points. Thread-safe.
 */
extern "C" void eshkol_capability_runtime_host_reset() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    g_capability_policy_active = false;
    g_capability_allow_list.clear();
}

/**
 * @brief Begin installing a policy: activate it with nothing granted yet.
 *
 * After this call every capability is denied until one is permitted through
 * eshkol_capability_runtime_allow(). When a policy is already active the
 * capabilities it granted are remembered, so the names that follow are
 * intersected with them rather than replacing them — a second install can only
 * narrow what the first one gave. Thread-safe.
 */
extern "C" void eshkol_capability_runtime_begin_install() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    if (g_capability_policy_active) {
        g_capability_previous_allow_list = g_capability_allow_list;
        g_capability_narrowing = true;
    } else {
        g_capability_previous_allow_list.clear();
        g_capability_narrowing = false;
    }
    g_capability_policy_active = true;
    g_capability_allow_list.clear();
}

/**
 * @brief Grant a capability in the policy being installed.
 *
 * No-op if no policy is currently active (i.e. before the first
 * begin_install()) or if `capability` is NULL/empty. When the install is
 * narrowing an existing policy, a name the previous policy did not grant is
 * not granted now either: a policy cannot hand its code more than it holds.
 * Thread-safe.
 *
 * @param capability  Capability name to permit, e.g. "file-read".
 */
extern "C" void eshkol_capability_runtime_allow(const char* capability) {
    if (capability == nullptr || capability[0] == '\0') {
        return;
    }

    std::lock_guard<std::mutex> lock(g_capability_mutex);
    if (!g_capability_policy_active) return;
    if (g_capability_narrowing &&
        g_capability_previous_allow_list.find(capability) ==
            g_capability_previous_allow_list.end()) {
        return;
    }
    g_capability_allow_list.insert(capability);
}

/** @brief Whether a capability policy is currently installed (1) or all capabilities are unrestricted (0). Thread-safe. */
extern "C" int eshkol_capability_runtime_is_active() {
    std::lock_guard<std::mutex> lock(g_capability_mutex);
    return g_capability_policy_active ? 1 : 0;
}

/**
 * @brief Whether @p capability is permitted under the current policy.
 *
 * With no policy installed every capability is allowed; once a policy is
 * active only the names it granted are. NULL or empty is never allowed.
 * Thread-safe.
 */
extern "C" int eshkol_capability_runtime_allows(const char* capability) {
    return capability_allows_locked(capability) ? 1 : 0;
}

/**
 * @brief Name the first capability an fopen()-style mode string needs and
 *        does not have.
 *
 * Inspects the first mode character ('r' implies read; 'w'/'a' imply write)
 * and any '+' flag (implies both), then checks each implied requirement
 * against the active policy.
 *
 * @param mode  fopen()-style mode string (e.g. "r", "w+", "a"). NULL/empty
 *              is treated as requiring "file-read".
 * @return      "file-read" or "file-write" naming the first missing
 *              capability, or NULL if the mode is fully permitted.
 */
extern "C" const char* eshkol_capability_runtime_file_mode_missing(const char* mode) {
    if (mode == nullptr || mode[0] == '\0') {
        return eshkol_capability_runtime_allows("file-read") ? nullptr : "file-read";
    }

    bool needs_read = mode[0] == 'r';
    bool needs_write = mode[0] == 'w' || mode[0] == 'a';
    for (const char* p = mode; *p; ++p) {
        if (*p == '+') {
            needs_read = true;
            needs_write = true;
        }
    }

    if (needs_read && !capability_allows_locked("file-read")) {
        return "file-read";
    }
    if (needs_write && !capability_allows_locked("file-write")) {
        return "file-write";
    }
    return nullptr;
}

/** @brief Whether an fopen()-style mode string is fully permitted by the current policy. */
extern "C" int eshkol_capability_runtime_allows_file_mode(const char* mode) {
    return eshkol_capability_runtime_file_mode_missing(mode) == nullptr ? 1 : 0;
}

/**
 * @brief Set errno to EACCES and report a one-time "capability denied" line
 *        for @p capability on stderr.
 *
 * Deduplicates per capability name, so a denied call in a loop produces a
 * single line. Thread-safe.
 *
 * @param capability  Capability name that was denied; "unknown" is used if
 *                    NULL or empty.
 */
extern "C" void eshkol_capability_runtime_deny(const char* capability) {
    errno = EACCES;
    const char* name = (capability != nullptr && capability[0] != '\0')
        ? capability
        : "unknown";

    bool should_report = false;
    {
        std::lock_guard<std::mutex> lock(g_capability_mutex);
        should_report = g_capability_denials_reported.insert(name).second;
    }
    if (should_report) {
        std::fprintf(stderr, "capability denied: %s\n", name);
    }
}

/** @brief Allow-or-record-denial guard shared by every capability-gated builtin (see runtime_exports.h). */
extern "C" int eshkol_capability_require(const char* capability) {
    if (!capability) return 1;
    if (capability_allows_locked(capability)) return 1;
    eshkol_capability_runtime_deny(capability);
    return 0;
}
