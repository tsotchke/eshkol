/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted process stack-limit setup.
 */

#include <eshkol/eshkol.h>

#include <cstdlib>

#ifndef _WIN32
#include <sys/resource.h>
#endif

extern "C" void eshkol_init_stack_size(void) {
#ifdef _WIN32
    // Windows thread stack sizing is handled at link/thread creation time.
    return;
#else
    const rlim_t default_stack = 512ULL * 1024 * 1024;  // 512MB
    rlim_t target = default_stack;

    const char* env_val = std::getenv("ESHKOL_STACK_SIZE");
    if (env_val) {
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(env_val, &end, 0);
        if (end != env_val && parsed >= 1024 * 1024) {
            target = (rlim_t)parsed;
        }
    }

    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0) {
        if (rl.rlim_cur < target) {
            rl.rlim_cur = target;
            if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < target) {
                rl.rlim_cur = rl.rlim_max;
            }
            setrlimit(RLIMIT_STACK, &rl);
        }
    }
#endif
}
