/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted shutdown-hook registry and dispatch.
 */

#include "runtime_hosted_internal.h"

#include <eshkol/logger.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct ShutdownHook {
    uint32_t id;
    eshkol_shutdown_hook_t callback;
    void* context;
    std::string name;
};

std::mutex g_hooks_mutex;
std::vector<ShutdownHook> g_shutdown_hooks;
std::atomic<uint32_t> g_next_hook_id{1};

}  // namespace

extern "C" {

/**
 * @brief Register a callback to run when the runtime shuts down.
 *
 * Thread-safe: takes g_hooks_mutex while appending to the shared hook
 * registry. Assigns the hook a fresh monotonically increasing id (starting
 * at 1) so it can later be looked up by eshkol_unregister_shutdown_hook().
 *
 * @param hook     Callback to invoke at shutdown; registration is rejected (returns 0) if NULL.
 * @param context  Opaque pointer passed back to `hook` on invocation.
 * @param name     Optional human-readable name for logging; "(unnamed)" if NULL.
 * @return         The new hook's id, or 0 if `hook` is NULL.
 */
uint32_t eshkol_register_shutdown_hook(eshkol_shutdown_hook_t hook,
                                        void* context,
                                        const char* name) {
    if (!hook) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(g_hooks_mutex);

    uint32_t id = g_next_hook_id.fetch_add(1, std::memory_order_relaxed);

    g_shutdown_hooks.push_back({
        .id = id,
        .callback = hook,
        .context = context,
        .name = name ? name : "(unnamed)"
    });

    eshkol_debug("Registered shutdown hook %u: %s", id, name ? name : "(unnamed)");
    return id;
}

/**
 * @brief Remove a previously registered shutdown hook by id.
 *
 * Thread-safe: takes g_hooks_mutex while searching and erasing from the
 * shared hook registry.
 *
 * @param hook_id  Id previously returned by eshkol_register_shutdown_hook().
 * @return         True if a matching hook was found and removed; false if
 *                 `hook_id` is 0 or no such hook is registered.
 */
bool eshkol_unregister_shutdown_hook(uint32_t hook_id) {
    if (hook_id == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_hooks_mutex);

    auto it = std::find_if(g_shutdown_hooks.begin(), g_shutdown_hooks.end(),
                           [hook_id](const ShutdownHook& h) { return h.id == hook_id; });

    if (it != g_shutdown_hooks.end()) {
        eshkol_debug("Unregistered shutdown hook %u: %s", hook_id, it->name.c_str());
        g_shutdown_hooks.erase(it);
        return true;
    }

    return false;
}

}  // extern "C"

namespace eshkol::runtime_hosted {

/**
 * @brief Invoke all registered shutdown hooks in LIFO (most-recently-registered-first) order.
 *
 * Takes a snapshot copy of the hook registry under g_hooks_mutex (so hook
 * callbacks are free to register/unregister hooks without deadlocking or
 * racing on the shared vector), reverses it, then calls each hook's
 * callback with `reason`, logging a warning for any hook that returns a
 * non-zero (error) result.
 *
 * @param reason  Reason code passed through to every hook callback.
 */
void run_shutdown_hooks(eshkol_shutdown_reason_t reason) {
    std::vector<ShutdownHook> hooks_copy;
    {
        std::lock_guard<std::mutex> lock(g_hooks_mutex);
        hooks_copy = g_shutdown_hooks;
    }

    std::reverse(hooks_copy.begin(), hooks_copy.end());

    for (const auto& hook : hooks_copy) {
        eshkol_debug("Calling shutdown hook: %s", hook.name.c_str());
        int result = hook.callback(hook.context, reason);
        if (result != 0) {
            eshkol_warn("Shutdown hook '%s' returned error: %d", hook.name.c_str(), result);
        }
    }
}

}  // namespace eshkol::runtime_hosted
