/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted resource-limit configuration.
 *
 * Owns environment-driven limit loading while shared accounting, timer, and
 * validation state live in resource_limits_core.cpp.
 */

#include <eshkol/core/resource_limits.h>
#include <eshkol/core/runtime.h>

#include <cstdlib>
#include <cstring>

namespace {

size_t parse_size(const char* str) {
    if (!str || !*str) return 0;

    char* end = nullptr;
    double value = strtod(str, &end);
    if (end == str) return 0;

    if (end && *end) {
        switch (*end) {
            case 'K':
            case 'k':
                value *= 1024;
                break;
            case 'M':
            case 'm':
                value *= 1024 * 1024;
                break;
            case 'G':
            case 'g':
                value *= 1024 * 1024 * 1024;
                break;
        }
    }

    return static_cast<size_t>(value);
}

bool parse_bool(const char* str) {
    if (!str) return false;
    return (strcmp(str, "true") == 0 || strcmp(str, "TRUE") == 0 ||
            strcmp(str, "1") == 0 || strcmp(str, "yes") == 0 ||
            strcmp(str, "YES") == 0);
}

}  // namespace

extern "C" {

eshkol_resource_limits_t eshkol_init_limits_from_env(void) {
    eshkol_resource_limits_t limits = eshkol_get_default_limits();

    const char* max_heap = std::getenv("ESHKOL_MAX_HEAP");
    if (max_heap) {
        limits.max_heap_bytes = parse_size(max_heap);
        limits.heap_soft_limit_bytes =
            (limits.max_heap_bytes * ESHKOL_HEAP_SOFT_LIMIT_PERCENT) / 100;
        eshkol_runtime_debugf("Max heap from env: %zu bytes",
                              limits.max_heap_bytes);
    }

    const char* timeout = std::getenv("ESHKOL_TIMEOUT_MS");
    if (timeout) {
        limits.max_execution_time_ms = static_cast<uint64_t>(atoll(timeout));
        eshkol_runtime_debugf("Timeout from env: %llu ms",
                              (unsigned long long)limits.max_execution_time_ms);
    }

    const char* max_stack = std::getenv("ESHKOL_MAX_STACK");
    if (max_stack) {
        limits.max_stack_depth = static_cast<size_t>(atoll(max_stack));
        eshkol_runtime_debugf("Max stack from env: %zu",
                              limits.max_stack_depth);
    }

    const char* max_tensor = std::getenv("ESHKOL_MAX_TENSOR_ELEMS");
    if (max_tensor) {
        limits.max_tensor_elements = parse_size(max_tensor);
        eshkol_runtime_debugf("Max tensor elements from env: %zu",
                              limits.max_tensor_elements);
    }

    const char* max_string = std::getenv("ESHKOL_MAX_STRING_LEN");
    if (max_string) {
        limits.max_string_length = parse_size(max_string);
        eshkol_runtime_debugf("Max string length from env: %zu",
                              limits.max_string_length);
    }

    const char* enforce = std::getenv("ESHKOL_ENFORCE_LIMITS");
    if (enforce) {
        limits.enforce_hard_limits = parse_bool(enforce);
        eshkol_runtime_debugf("Enforce limits from env: %s",
                              limits.enforce_hard_limits ? "true" : "false");
    }

    const char* warnings = std::getenv("ESHKOL_LIMIT_WARNINGS");
    if (warnings) {
        limits.enable_warnings = parse_bool(warnings);
        eshkol_runtime_debugf("Limit warnings from env: %s",
                              limits.enable_warnings ? "true" : "false");
    }

    eshkol_set_limits(&limits);
    return limits;
}

}  // extern "C"
