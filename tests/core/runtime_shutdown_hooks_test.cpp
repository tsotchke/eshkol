/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/runtime.h>

#include <iostream>
#include <vector>

namespace {

std::vector<int> g_calls;

int record_hook(void* context, eshkol_shutdown_reason_t reason) {
    if (reason != ESHKOL_SHUTDOWN_REQUESTED) {
        return 100;
    }
    g_calls.push_back(*static_cast<int*>(context));
    return 0;
}

int failing_hook(void* context, eshkol_shutdown_reason_t reason) {
    if (reason != ESHKOL_SHUTDOWN_REQUESTED) {
        return 100;
    }
    g_calls.push_back(*static_cast<int*>(context));
    return 7;
}

int fail(const char* message) {
    std::cerr << "runtime_shutdown_hooks_test: " << message << '\n';
    return 1;
}

}  // namespace

int main() {
    if (eshkol_runtime_init() != 0) {
        return fail("runtime init failed");
    }

    int first = 1;
    int removed = 2;
    int third = 3;
    int fourth = 4;

    const uint32_t first_id = eshkol_register_shutdown_hook(record_hook, &first, "first");
    const uint32_t removed_id = eshkol_register_shutdown_hook(record_hook, &removed, "removed");
    const uint32_t third_id = eshkol_register_shutdown_hook(record_hook, &third, "third");
    const uint32_t fourth_id = eshkol_register_shutdown_hook(failing_hook, &fourth, "fourth");

    if (first_id == 0 || removed_id == 0 || third_id == 0 || fourth_id == 0) {
        return fail("hook registration returned zero");
    }
    if (!eshkol_unregister_shutdown_hook(removed_id)) {
        return fail("failed to unregister existing hook");
    }
    if (eshkol_unregister_shutdown_hook(removed_id)) {
        return fail("unregistered hook was removed twice");
    }

    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_REQUESTED);

    const std::vector<int> expected = {4, 3, 1};
    if (g_calls != expected) {
        std::cerr << "runtime_shutdown_hooks_test: unexpected hook order";
        std::cerr << " got:";
        for (int value : g_calls) {
            std::cerr << ' ' << value;
        }
        std::cerr << " expected:";
        for (int value : expected) {
            std::cerr << ' ' << value;
        }
        std::cerr << '\n';
        return 1;
    }

    if (eshkol_runtime_get_state() != ESHKOL_RUNTIME_TERMINATED) {
        return fail("runtime did not terminate");
    }

    return 0;
}
