/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted fatal runtime reporting.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#include <cstdio>
#include <cstdlib>

extern "C" [[noreturn]] void eshkol_runtime_default_fatal_sink(
    const char* message) {
    const char* text = message ? message : "Fatal runtime error";
    std::fprintf(stderr, "%s\n", text);

    eshkol_exception_t* exc =
        eshkol_make_exception(ESHKOL_EXCEPTION_ERROR, text);
    if (exc) {
        eshkol_raise(exc);
    }
    std::exit(1);
}
