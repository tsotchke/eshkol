/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/runtime.h>

#include <csignal>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << "runtime_signal_handlers_test: " << message << '\n';
    return 1;
}

}  // namespace

int main() {
    if (eshkol_runtime_init() != 0) {
        return fail("runtime init failed");
    }

    if (eshkol_runtime_interrupt_requested()) {
        return fail("interrupt was set before signal delivery");
    }

    if (std::raise(SIGTERM) != 0) {
        return fail("failed to raise SIGTERM");
    }

    if (!eshkol_runtime_interrupt_requested()) {
        return fail("SIGTERM handler did not set interrupt flag");
    }

    eshkol_runtime_clear_interrupt();
    if (eshkol_runtime_interrupt_requested()) {
        return fail("interrupt flag did not clear");
    }

    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_REQUESTED);
    if (eshkol_runtime_get_state() != ESHKOL_RUNTIME_TERMINATED) {
        return fail("runtime did not terminate");
    }

    return 0;
}
