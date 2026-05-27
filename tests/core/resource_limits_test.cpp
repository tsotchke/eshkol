/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/resource_limits.h>
#include <eshkol/core/runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

int fail(const char* message) {
    std::cerr << "resource_limits_test: " << message << '\n';
    return 1;
}

void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

void unset_env(const char* key) {
#ifdef _WIN32
    _putenv_s(key, "");
#else
    unsetenv(key);
#endif
}

}  // namespace

int main() {
    const eshkol_resource_limits_t defaults = eshkol_get_default_limits();
    const eshkol_resource_limits_t* active_defaults = eshkol_get_limits();
    if (active_defaults->max_heap_bytes != defaults.max_heap_bytes) {
        return fail("active limits were not default-initialized");
    }
    if (active_defaults->max_stack_depth != defaults.max_stack_depth) {
        return fail("active stack limit was not default-initialized");
    }

    set_env("ESHKOL_MAX_HEAP", "64KB");
    set_env("ESHKOL_TIMEOUT_MS", "25");
    set_env("ESHKOL_MAX_STACK", "3");
    set_env("ESHKOL_MAX_TENSOR_ELEMS", "7");
    set_env("ESHKOL_MAX_STRING_LEN", "9");
    set_env("ESHKOL_ENFORCE_LIMITS", "true");
    set_env("ESHKOL_LIMIT_WARNINGS", "false");

    eshkol_resource_limits_t env_limits = eshkol_init_limits_from_env();
    if (env_limits.max_heap_bytes != 64 * 1024) return fail("env heap parse mismatch");
    if (env_limits.heap_soft_limit_bytes != (64 * 1024 * ESHKOL_HEAP_SOFT_LIMIT_PERCENT) / 100) {
        return fail("env heap soft limit mismatch");
    }
    if (env_limits.max_execution_time_ms != 25) return fail("env timeout parse mismatch");
    if (env_limits.max_stack_depth != 3) return fail("env stack parse mismatch");
    if (env_limits.max_tensor_elements != 7) return fail("env tensor parse mismatch");
    if (env_limits.max_string_length != 9) return fail("env string parse mismatch");
    if (!env_limits.enforce_hard_limits) return fail("env enforce parse mismatch");
    if (env_limits.enable_warnings) return fail("env warnings parse mismatch");

    unset_env("ESHKOL_MAX_HEAP");
    unset_env("ESHKOL_TIMEOUT_MS");
    unset_env("ESHKOL_MAX_STACK");
    unset_env("ESHKOL_MAX_TENSOR_ELEMS");
    unset_env("ESHKOL_MAX_STRING_LEN");
    unset_env("ESHKOL_ENFORCE_LIMITS");
    unset_env("ESHKOL_LIMIT_WARNINGS");

    set_env("ESHKOL_MAX_HEAP", "64bad");
    set_env("ESHKOL_TIMEOUT_MS", "-25");
    set_env("ESHKOL_MAX_STACK", "");
    set_env("ESHKOL_MAX_TENSOR_ELEMS", "nan");
    set_env("ESHKOL_MAX_STRING_LEN", "9 trailing");
    set_env("ESHKOL_ENFORCE_LIMITS", "maybe");
    set_env("ESHKOL_LIMIT_WARNINGS", "sometimes");

    env_limits = eshkol_init_limits_from_env();
    if (env_limits.max_heap_bytes != defaults.max_heap_bytes) {
        return fail("invalid env heap did not preserve default");
    }
    if (env_limits.heap_soft_limit_bytes != defaults.heap_soft_limit_bytes) {
        return fail("invalid env heap soft limit did not preserve default");
    }
    if (env_limits.max_execution_time_ms != defaults.max_execution_time_ms) {
        return fail("invalid env timeout did not preserve default");
    }
    if (env_limits.max_stack_depth != defaults.max_stack_depth) {
        return fail("invalid env stack did not preserve default");
    }
    if (env_limits.max_tensor_elements != defaults.max_tensor_elements) {
        return fail("invalid env tensor limit did not preserve default");
    }
    if (env_limits.max_string_length != defaults.max_string_length) {
        return fail("invalid env string limit did not preserve default");
    }
    if (env_limits.enforce_hard_limits != defaults.enforce_hard_limits) {
        return fail("invalid env enforce flag did not preserve default");
    }
    if (env_limits.enable_warnings != defaults.enable_warnings) {
        return fail("invalid env warning flag did not preserve default");
    }

    unset_env("ESHKOL_MAX_HEAP");
    unset_env("ESHKOL_TIMEOUT_MS");
    unset_env("ESHKOL_MAX_STACK");
    unset_env("ESHKOL_MAX_TENSOR_ELEMS");
    unset_env("ESHKOL_MAX_STRING_LEN");
    unset_env("ESHKOL_ENFORCE_LIMITS");
    unset_env("ESHKOL_LIMIT_WARNINGS");

    eshkol_resource_limits_t limits = defaults;
    limits.max_heap_bytes = 16;
    limits.heap_soft_limit_bytes = 12;
    limits.enforce_hard_limits = false;
    limits.enable_warnings = false;
    eshkol_set_limits(&limits);
    eshkol_reset_resource_tracking();

    if (!eshkol_track_allocation(8)) return fail("first allocation was rejected");
    if (!eshkol_track_allocation(7)) return fail("second allocation was rejected");
    if (eshkol_get_heap_usage() != 15) return fail("heap usage mismatch after allocations");
    if (eshkol_get_peak_heap_usage() != 15) return fail("peak heap usage mismatch");
    if (!eshkol_is_near_memory_limit()) return fail("near-memory check did not trip");
    if (eshkol_track_allocation(2)) return fail("over-limit allocation was accepted");
    if (eshkol_get_heap_usage() != 15) return fail("rejected allocation changed heap usage");
    if (eshkol_get_last_limit_error() != ESHKOL_LIMIT_HEAP_HARD) {
        return fail("heap hard-limit error was not recorded");
    }
    eshkol_track_deallocation(6);
    if (eshkol_get_heap_usage() != 9) return fail("heap deallocation mismatch");
    eshkol_track_deallocation(99);
    if (eshkol_get_heap_usage() != 0) return fail("heap deallocation underflowed");

    limits = defaults;
    limits.max_stack_depth = 1;
    limits.enforce_hard_limits = false;
    eshkol_set_limits(&limits);
    eshkol_reset_resource_tracking();
    if (!eshkol_stack_push()) return fail("first stack push rejected");
    if (eshkol_stack_push()) return fail("over-limit stack push accepted");
    if (eshkol_get_stack_depth() != 1) return fail("failed stack push changed depth");
    eshkol_stack_pop();
    if (eshkol_get_stack_depth() != 0) return fail("stack pop mismatch");

    limits = defaults;
    limits.max_tensor_elements = 3;
    limits.max_string_length = 4;
    limits.enforce_hard_limits = false;
    eshkol_set_limits(&limits);
    eshkol_reset_resource_tracking();
    if (!eshkol_check_tensor_size(3)) return fail("valid tensor size rejected");
    if (eshkol_check_tensor_size(4)) return fail("over-limit tensor size accepted");
    if (eshkol_get_last_limit_error() != ESHKOL_LIMIT_TENSOR_SIZE) {
        return fail("tensor size error was not recorded");
    }
    if (!eshkol_check_string_length(4)) return fail("valid string length rejected");
    if (eshkol_check_string_length(5)) return fail("over-limit string length accepted");
    if (eshkol_get_last_limit_error() != ESHKOL_LIMIT_STRING_LENGTH) {
        return fail("string length error was not recorded");
    }

    limits = defaults;
    limits.max_execution_time_ms = 20;
    limits.enforce_hard_limits = false;
    eshkol_set_limits(&limits);
    eshkol_reset_resource_tracking();
    eshkol_start_timer(20);
    std::this_thread::sleep_for(std::chrono::milliseconds(35));
    if (!eshkol_is_timed_out()) return fail("polled timer did not time out");
    if (eshkol_get_last_limit_error() != ESHKOL_LIMIT_TIMEOUT) {
        return fail("timer timeout error was not recorded");
    }
    eshkol_stop_timer();

    limits = defaults;
    limits.max_execution_time_ms = 30;
    limits.enforce_hard_limits = true;
    limits.enable_warnings = false;
    eshkol_set_limits(&limits);
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    eshkol_start_timer(30);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    if (!eshkol_runtime_interrupt_requested()) {
        return fail("watchdog did not request runtime interrupt");
    }
    if (eshkol_runtime_get_shutdown_reason() != ESHKOL_SHUTDOWN_TIMEOUT) {
        return fail("watchdog shutdown reason mismatch");
    }
    if (eshkol_get_last_limit_error() != ESHKOL_LIMIT_TIMEOUT) {
        return fail("watchdog timeout error was not recorded");
    }
    eshkol_stop_timer();
    eshkol_runtime_clear_interrupt();

    return 0;
}
