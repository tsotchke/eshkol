/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted runtime error helpers.
 *
 * This file owns the current stderr/logger/exit-backed error path. The symbol
 * names are still part of the generated-code runtime ABI, but the implementation
 * is hosted until the freestanding panic/error hook ABI is introduced.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

extern "C" {

void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    std::fprintf(stderr, "%s\n", buf);

    eshkol_exception_t* exc = eshkol_make_exception(type, buf);
    if (exc) {
        eshkol_raise(exc);
        // If eshkol_raise returns, do not let callers continue past a fatal
        // runtime condition.
    }
    std::exit(1);
}

void eshkol_type_error(const char* proc_name, const char* expected_type) {
    eshkol_error("Type error in %s: expected %s",
                 proc_name ? proc_name : "<unknown>",
                 expected_type ? expected_type : "<type>");

    eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                         "Type error in %s: expected %s",
                         proc_name ? proc_name : "<unknown>",
                         expected_type ? expected_type : "<type>");
}

void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                  const char* actual_type) {
    eshkol_error("Type error in %s: expected %s, got %s",
                 proc_name ? proc_name : "<unknown>",
                 expected_type ? expected_type : "<type>",
                 actual_type ? actual_type : "<unknown>");

    eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                         "Type error in %s: expected %s, got %s",
                         proc_name ? proc_name : "<unknown>",
                         expected_type ? expected_type : "<type>",
                         actual_type ? actual_type : "<unknown>");
}

}  // extern "C"
