/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted arena diagnostics policy.
 */

#include <cstdlib>

/**
 * @brief Report whether arena allocation poisoning is enabled for this process.
 *
 * Reads the ESHKOL_ARENA_POISON environment variable once (cached in a
 * function-local static) and enables poisoning unless it is unset, empty,
 * or "0".
 *
 * @return Non-zero if arena poisoning is enabled, zero otherwise.
 */
extern "C" int eshkol_arena_poison_enabled(void) {
    static int poison_enabled = -1;
    if (poison_enabled < 0) {
        const char* env = std::getenv("ESHKOL_ARENA_POISON");
        poison_enabled = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return poison_enabled;
}
