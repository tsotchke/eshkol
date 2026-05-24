/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted arena diagnostics policy.
 */

#include <cstdlib>

extern "C" int eshkol_arena_poison_enabled(void) {
    static int poison_enabled = -1;
    if (poison_enabled < 0) {
        const char* env = std::getenv("ESHKOL_ARENA_POISON");
        poison_enabled = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return poison_enabled;
}
