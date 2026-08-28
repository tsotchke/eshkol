/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/config.h>
#include <eshkol/core/resource_limits.h>
#include <eshkol/core/runtime.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>

// The native engine's ESHKOL_ARENA_POISON accessor
// (lib/core/runtime_arena_diagnostics_hosted.cpp) is not declared in any
// installed header -- every caller forward-declares it (see e.g.
// tests/core/ad_tape_region_growth_test.cpp). This binary links eshkol-static,
// so it is the one place that can see both that accessor and the bytecode
// VM's vm_arena_poison_enabled() (a self-contained, dependency-free header --
// see tests/core/vm_arena_poison_test.cpp, which cannot link eshkol-static)
// in the same translation unit, to check they agree on every value.
extern "C" int eshkol_arena_poison_enabled(void);
extern "C" {
#include "../../lib/backend/vm_arena.h"
}

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
    // BI-12: ESHKOL_ARENA_POISON parity across the native engine's accessor
    // (eshkol_arena_poison_enabled(), lib/core/runtime_arena_diagnostics_hosted.cpp)
    // and the bytecode VM's (vm_arena_poison_enabled(), lib/backend/vm_arena.h;
    // the VM evacuator now calls this same function rather than re-reading the
    // variable itself, so checking it here also covers the evacuator). Both
    // cache their result on first call for the life of the process, so this
    // must run before anything else in this binary can have touched either --
    // hence first in main(), with a value ("01") that used to arm only the VM
    // arena because the native accessor tested just the first byte.
    {
        set_env("ESHKOL_ARENA_POISON", "01");
        const int native_armed = eshkol_arena_poison_enabled();
        const int vm_armed = vm_arena_poison_enabled();
        if (!native_armed) {
            return fail("ESHKOL_ARENA_POISON=01: native accessor did not arm (regression -- "
                        "it used to test only the first byte)");
        }
        if (!vm_armed) return fail("ESHKOL_ARENA_POISON=01: VM accessor did not arm");
        if ((native_armed != 0) != (vm_armed != 0)) {
            return fail("ESHKOL_ARENA_POISON=01: native and VM accessors disagree");
        }
    }

    // BI-10: the `[types]` config-file section (docs/breakdown/
    // RUNTIME_CONFIGURATION.md) is documented but was never wired into
    // apply_config_section() -- `strict` / `unsafe` were parsed nowhere and
    // silently ignored. Load the checked-in fixture through the real
    // production path (eshkol_config_load_file() -> parse_toml() ->
    // apply_config_section()) and confirm both fields move off their
    // defaults, exactly as --strict-types / --unsafe would set them.
    {
        const std::filesystem::path fixture =
            std::filesystem::path(__FILE__).parent_path() /
            "fixtures" / "config_types_section.toml";
        if (!std::filesystem::exists(fixture)) {
            return fail(("config fixture missing: " + fixture.string()).c_str());
        }

        eshkol_config_t config = eshkol_config_defaults();
        if (config.strict_types) return fail("config: strict_types default should be false");
        if (config.unsafe_mode) return fail("config: unsafe_mode default should be false");

        if (eshkol_config_load_file(&config, fixture.string().c_str()) != 0) {
            return fail("config: eshkol_config_load_file failed on the [types] fixture");
        }
        if (!config.strict_types) {
            return fail("config: [types] strict = true from the file was not applied "
                        "(the [types] section is not wired into apply_config_section)");
        }
        if (!config.unsafe_mode) {
            return fail("config: [types] unsafe = true from the file was not applied "
                        "(the [types] section is not wired into apply_config_section)");
        }
    }

    // eshkol_parse_size() is the shared parser behind every ESHKOL_* size
    // variable, including ESHKOL_STACK_SIZE (parsed in
    // runtime_stack_hosted.cpp, which does not otherwise expose a testable
    // seam). Exercise it directly: bare bytes, K/M/G and KiB/MiB/GiB
    // suffixes, garbage, and below-floor values (floor enforcement is the
    // caller's job, not the parser's -- eshkol_parse_size only reports
    // whether the string parsed).
    {
        size_t out = 0;

        if (!eshkol_parse_size("512", &out) || out != 512) {
            return fail("parse_size: bare byte count mismatch");
        }
        if (!eshkol_parse_size("512M", &out) || out != 512ULL * 1024 * 1024) {
            return fail("parse_size: 512M mismatch");
        }
        if (!eshkol_parse_size("1G", &out) || out != 1ULL * 1024 * 1024 * 1024) {
            return fail("parse_size: 1G mismatch");
        }
        if (!eshkol_parse_size("1GiB", &out) || out != 1ULL * 1024 * 1024 * 1024) {
            return fail("parse_size: 1GiB mismatch");
        }
        if (!eshkol_parse_size("64KB", &out) || out != 64ULL * 1024) {
            return fail("parse_size: 64KB mismatch");
        }
        if (!eshkol_parse_size("1 MiB", &out) || out != 1ULL * 1024 * 1024) {
            return fail("parse_size: '1 MiB' (space before suffix) mismatch");
        }
        // "512M" misread as 512 bytes was the bug this closes: it must parse
        // to 512 MiB, not 512 bytes.
        if (!eshkol_parse_size("512M", &out) || out == 512) {
            return fail("parse_size: 512M must not be misread as 512 bytes");
        }

        // Below a caller's floor still parses successfully -- floor
        // enforcement happens above eshkol_parse_size(), not inside it.
        if (!eshkol_parse_size("1", &out) || out != 1) {
            return fail("parse_size: below-floor value should still parse");
        }

        // Garbage / trailing garbage must fail outright, not silently parse
        // a numeric prefix.
        if (eshkol_parse_size("", &out)) return fail("parse_size: empty string should fail");
        if (eshkol_parse_size(nullptr, &out)) return fail("parse_size: null should fail");
        if (eshkol_parse_size("garbage", &out)) return fail("parse_size: pure garbage should fail");
        if (eshkol_parse_size("512X", &out)) return fail("parse_size: unrecognized suffix should fail");
        if (eshkol_parse_size("512M!", &out)) return fail("parse_size: trailing garbage after suffix should fail");
        if (eshkol_parse_size("512 trailing", &out)) {
            return fail("parse_size: trailing garbage after bare number should fail");
        }
        if (eshkol_parse_size("-5", &out)) return fail("parse_size: negative value should fail");
    }

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
