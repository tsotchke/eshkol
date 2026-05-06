#include <eshkol/core/arena.h>
#include <eshkol/core/resource_limits.h>
#include <eshkol/core/runtime.h>
#include <eshkol/eshkol.h>

#include <cstdio>
#include <cstdlib>
#include <csetjmp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

struct CapturedDiagnostic {
    eshkol_runtime_diagnostic_level_t level;
    std::string message;
};

struct ShutdownCapture {
    std::vector<std::string>* order;
    std::vector<eshkol_shutdown_reason_t>* reasons;
    const char* label;
};

struct OperationDrainCapture {
    std::vector<int>* timeouts;
    uint32_t* operation_to_end;
    size_t call_count;
};

struct MonotonicClockCapture {
    uint64_t next_time_ns;
    uint64_t step_ns;
    size_t call_count;
};

struct DelayCapture {
    std::vector<uint64_t>* durations_ns;
    bool result;
};

template <typename T>
bool expect_equal(const T& actual, const T& expected, const std::string& label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& label) {
    if (haystack.find(needle) != std::string::npos) {
        return true;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

bool expect_flag_set(uint32_t flags, uint32_t flag, bool expected,
                     const std::string& label) {
    const bool actual = (flags & flag) != 0u;
    return expect_equal(actual, expected, label);
}

bool expect_null_tagged_value(const eshkol_tagged_value_t& value,
                              const std::string& label) {
    if (value.type == ESHKOL_VALUE_NULL && value.data.raw_val == 0) {
        return true;
    }
    std::cerr << "FAIL: " << label << " (type="
              << static_cast<int>(value.type)
              << ", raw=" << value.data.raw_val << ")" << std::endl;
    return false;
}

extern "C" void capture_diagnostic_hook(
    eshkol_runtime_diagnostic_level_t level,
    const char* message,
    void* context) {
    auto* captures = static_cast<std::vector<CapturedDiagnostic>*>(context);
    if (!captures) {
        return;
    }

    captures->push_back(CapturedDiagnostic{
        level,
        message ? std::string(message) : std::string(),
    });
}

extern "C" int capture_shutdown_hook(void* context,
                                     eshkol_shutdown_reason_t reason) {
    auto* capture = static_cast<ShutdownCapture*>(context);
    if (!capture || !capture->order || !capture->reasons) {
        return 1;
    }

    capture->order->push_back(capture->label ? capture->label : "");
    capture->reasons->push_back(reason);
    return 0;
}

extern "C" bool capture_operation_drain_hook(int timeout_ms, void* context) {
    auto* capture = static_cast<OperationDrainCapture*>(context);
    if (!capture || !capture->timeouts) {
        return false;
    }

    capture->timeouts->push_back(timeout_ms);
    ++capture->call_count;
    if (capture->operation_to_end && *capture->operation_to_end != 0) {
        eshkol_runtime_end_operation(*capture->operation_to_end);
        *capture->operation_to_end = 0;
    }
    return true;
}

extern "C" bool capture_monotonic_clock_hook(uint64_t* out_time_ns,
                                             void* context) {
    auto* capture = static_cast<MonotonicClockCapture*>(context);
    if (!capture || !out_time_ns) {
        return false;
    }

    *out_time_ns = capture->next_time_ns;
    capture->next_time_ns += capture->step_ns;
    ++capture->call_count;
    return true;
}

extern "C" bool capture_delay_hook(uint64_t duration_ns, void* context) {
    auto* capture = static_cast<DelayCapture*>(context);
    if (!capture || !capture->durations_ns) {
        return false;
    }

    capture->durations_ns->push_back(duration_ns);
    return capture->result;
}

extern "C" void capture_fatal_hook(const char* message, void* context) {
    const char* output_path = static_cast<const char*>(context);
    if (!output_path) {
        return;
    }

    std::ofstream output(output_path, std::ios::trunc);
    output << (message ? message : "");
    output.flush();
}

bool test_runtime_freestanding_capability_registry_surface() {
    eshkol_runtime_clear_diagnostic_hook();
    eshkol_runtime_clear_fatal_hook();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_delay_hook();
    eshkol_runtime_clear_operation_drain_hook();

    eshkol_runtime_capability_descriptor_t descriptor{};
    bool ok =
        expect_equal(eshkol_runtime_get_capability_count(), size_t{5},
                     "freestanding runtime capability registry exposes the bounded hook family") &&
        expect_equal(eshkol_runtime_describe_capability(
                         ESHKOL_RUNTIME_CAPABILITY_DIAGNOSTIC_SINK,
                         &descriptor),
                     true,
                     "freestanding runtime capability registry can describe diagnostic sink") &&
        expect_equal(std::string(descriptor.name), std::string("diagnostic-sink"),
                     "freestanding runtime capability registry names the diagnostic sink") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, true,
                        "freestanding runtime keeps a default diagnostic sink capability") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLABLE, true,
                        "freestanding runtime diagnostic capability is hook-installable") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, false,
                        "freestanding runtime diagnostic capability starts without a custom hook") &&
        expect_equal(eshkol_runtime_describe_capability(
                         ESHKOL_RUNTIME_CAPABILITY_MONOTONIC_CLOCK,
                         &descriptor),
                     true,
                     "freestanding runtime capability registry can describe monotonic clock") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, false,
                        "freestanding runtime does not claim a default monotonic clock") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLABLE, true,
                        "freestanding monotonic clock capability remains hook-installable") &&
        expect_equal(eshkol_runtime_describe_capability(
                         ESHKOL_RUNTIME_CAPABILITY_DELAY, &descriptor),
                     true,
                     "freestanding runtime capability registry can describe delay") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, false,
                        "freestanding runtime does not claim a default non-zero delay primitive") &&
        expect_equal(eshkol_runtime_describe_capability(
                         ESHKOL_RUNTIME_CAPABILITY_OPERATION_DRAIN,
                         &descriptor),
                     true,
                     "freestanding runtime capability registry can describe operation drain") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, false,
                        "freestanding runtime does not claim an implicit operation-drain hook") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, false,
                        "freestanding operation-drain capability starts without a hook");

    std::vector<CapturedDiagnostic> captures;
    std::vector<uint64_t> delay_durations;
    std::vector<int> drain_timeouts;
    MonotonicClockCapture clock_capture{100, 25, 0};
    DelayCapture delay_capture{&delay_durations, true};
    OperationDrainCapture drain_capture{&drain_timeouts, nullptr, 0};
    eshkol_runtime_set_diagnostic_hook(capture_diagnostic_hook, &captures);
    eshkol_runtime_set_monotonic_clock_hook(capture_monotonic_clock_hook,
                                            &clock_capture);
    eshkol_runtime_set_delay_hook(capture_delay_hook, &delay_capture);
    eshkol_runtime_set_operation_drain_hook(capture_operation_drain_hook,
                                            &drain_capture);

    ok = ok &&
         expect_equal(eshkol_runtime_describe_capability(
                          ESHKOL_RUNTIME_CAPABILITY_DIAGNOSTIC_SINK,
                          &descriptor),
                      true,
                      "freestanding runtime capability registry re-reads diagnostic sink state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "freestanding diagnostic capability reflects an installed custom hook") &&
         expect_equal(eshkol_runtime_describe_capability(
                          ESHKOL_RUNTIME_CAPABILITY_MONOTONIC_CLOCK,
                          &descriptor),
                      true,
                      "freestanding runtime capability registry re-reads monotonic clock state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "freestanding monotonic clock capability reflects an installed BSP hook") &&
         expect_equal(eshkol_runtime_describe_capability(
                          ESHKOL_RUNTIME_CAPABILITY_DELAY, &descriptor),
                      true,
                      "freestanding runtime capability registry re-reads delay state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "freestanding delay capability reflects an installed BSP hook") &&
         expect_equal(eshkol_runtime_describe_capability(
                          ESHKOL_RUNTIME_CAPABILITY_OPERATION_DRAIN,
                          &descriptor),
                      true,
                      "freestanding runtime capability registry re-reads operation-drain state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "freestanding operation-drain capability reflects an installed BSP hook");

    eshkol_runtime_clear_operation_drain_hook();
    eshkol_runtime_clear_delay_hook();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();

    return ok &&
           expect_equal(eshkol_runtime_describe_capability(
                            ESHKOL_RUNTIME_CAPABILITY_MONOTONIC_CLOCK,
                            &descriptor),
                        true,
                        "freestanding runtime capability registry remains readable after hook clear") &&
           expect_flag_set(descriptor.flags,
                           ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, false,
                           "freestanding monotonic clock capability clears the installed-hook bit");
}

bool test_runtime_freestanding_diagnostic_hooks() {
    std::vector<CapturedDiagnostic> captures;

    eshkol_runtime_set_diagnostic_hook(capture_diagnostic_hook, &captures);
    eshkol_runtime_debugf("freestanding debug %d", 7);
    eshkol_runtime_infof("freestanding info %s", "limits");
    eshkol_runtime_warnf("freestanding warn %s", "uart");
    eshkol_runtime_errorf("freestanding error %d", 11);
    eshkol_runtime_clear_diagnostic_hook();

    const bool ok =
        expect_equal(captures.size(), size_t{4},
                     "freestanding diagnostic hook captures runtime diagnostics") &&
        expect_equal(captures[0].level, ESHKOL_RUNTIME_DIAGNOSTIC_DEBUG,
                     "freestanding diagnostic hook preserves debug level") &&
        expect_contains(captures[0].message, "freestanding debug 7",
                        "freestanding diagnostic hook formats debug messages") &&
        expect_equal(captures[1].level, ESHKOL_RUNTIME_DIAGNOSTIC_INFO,
                     "freestanding diagnostic hook preserves info level") &&
        expect_contains(captures[1].message, "freestanding info limits",
                        "freestanding diagnostic hook formats info messages") &&
        expect_equal(captures[2].level, ESHKOL_RUNTIME_DIAGNOSTIC_WARNING,
                     "freestanding diagnostic hook preserves warning level") &&
        expect_contains(captures[2].message, "freestanding warn uart",
                        "freestanding diagnostic hook formats warning messages") &&
        expect_equal(captures[3].level, ESHKOL_RUNTIME_DIAGNOSTIC_ERROR,
                     "freestanding diagnostic hook preserves error level") &&
        expect_contains(captures[3].message, "freestanding error 11",
                        "freestanding diagnostic hook formats error messages");

    eshkol_runtime_debugf("post-clear diagnostic");
    return ok &&
           expect_equal(captures.size(), size_t{4},
                        "freestanding default diagnostic sink is inert after hook clear");
}

bool test_runtime_freestanding_interrupt_surface() {
#ifdef _WIN32
    return true;
#else
    const pid_t pid = fork();
    if (pid == -1) {
        std::cerr << "FAIL: freestanding interrupt fork" << std::endl;
        return false;
    }

    if (pid == 0) {
        eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_MEMORY);
        if (!eshkol_runtime_interrupt_requested()) {
            _exit(101);
        }
        if (eshkol_runtime_get_shutdown_reason() != ESHKOL_SHUTDOWN_MEMORY) {
            _exit(102);
        }
        if (eshkol_runtime_get_state() != ESHKOL_RUNTIME_SHUTTING_DOWN) {
            _exit(103);
        }
        eshkol_runtime_clear_interrupt();
        if (eshkol_runtime_interrupt_requested()) {
            _exit(104);
        }
        if (eshkol_runtime_get_shutdown_reason() != ESHKOL_SHUTDOWN_NONE) {
            _exit(105);
        }
        _exit(0);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) == -1) {
        std::cerr << "FAIL: freestanding interrupt waitpid" << std::endl;
        return false;
    }

    return expect_equal(WIFEXITED(status) && WEXITSTATUS(status) == 0, true,
                        "freestanding interrupt request/clear surface works in an isolated process");
#endif
}

bool test_runtime_freestanding_timebase_surface() {
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_delay_hook();

    uint64_t default_time = 0;
    bool ok =
        expect_equal(eshkol_runtime_get_monotonic_time_ns(nullptr), false,
                     "freestanding monotonic time rejects null output storage") &&
        expect_equal(eshkol_runtime_get_monotonic_time_ns(&default_time), false,
                     "freestanding runtime has no default monotonic time source without a BSP hook") &&
        expect_equal(eshkol_runtime_delay_ns(0), true,
                     "freestanding zero-duration delay remains a successful no-op") &&
        expect_equal(eshkol_runtime_delay_ns(123), false,
                     "freestanding runtime has no default non-zero delay primitive without a BSP hook");

    std::vector<uint64_t> delay_durations;
    MonotonicClockCapture clock_capture{1000, 500, 0};
    DelayCapture delay_capture{&delay_durations, true};
    eshkol_runtime_set_monotonic_clock_hook(capture_monotonic_clock_hook,
                                            &clock_capture);
    eshkol_runtime_set_delay_hook(capture_delay_hook, &delay_capture);

    uint64_t hook_first = 0;
    uint64_t hook_second = 0;
    ok = ok &&
         expect_equal(eshkol_runtime_get_monotonic_time_ns(&hook_first), true,
                      "freestanding runtime can read monotonic time through a BSP hook") &&
         expect_equal(eshkol_runtime_get_monotonic_time_ns(&hook_second), true,
                      "freestanding runtime reuses the installed monotonic clock hook") &&
         expect_equal(hook_first, uint64_t{1000},
                      "freestanding monotonic clock hook controls the returned value") &&
         expect_equal(hook_second, uint64_t{1500},
                      "freestanding monotonic clock hook can advance under BSP control") &&
         expect_equal(clock_capture.call_count, size_t{2},
                      "freestanding monotonic clock hook records both reads") &&
         expect_equal(eshkol_runtime_delay_ns(777), true,
                      "freestanding runtime can delay through a BSP hook") &&
         expect_equal(delay_durations.size(), size_t{1},
                      "freestanding delay hook records the call") &&
         expect_equal(delay_durations[0], uint64_t{777},
                      "freestanding delay hook receives the requested duration");

    eshkol_runtime_clear_delay_hook();
    eshkol_runtime_clear_monotonic_clock_hook();

    uint64_t cleared_time = 0;
    return ok &&
           expect_equal(eshkol_runtime_get_monotonic_time_ns(&cleared_time), false,
                        "freestanding runtime falls back to unsupported monotonic time after hook clear") &&
           expect_equal(eshkol_runtime_delay_ns(123), false,
                        "freestanding runtime falls back to unsupported delay after hook clear");
}

bool test_runtime_freestanding_resource_limits_surface() {
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    eshkol_runtime_clear_monotonic_clock_hook();

    eshkol_resource_limits_t limits = eshkol_get_default_limits();
    limits.enforce_hard_limits = false;
    limits.enable_warnings = false;
    eshkol_set_limits(&limits);

    MonotonicClockCapture clock_capture{1000000000ULL, 1000000000ULL, 0};
    eshkol_runtime_set_monotonic_clock_hook(capture_monotonic_clock_hook,
                                            &clock_capture);

    eshkol_start_timer(2000);
    const bool ok =
        expect_equal(eshkol_track_allocation(1024), true,
                     "freestanding resource-limit core allows tracked allocations below the limit") &&
        expect_equal(eshkol_get_heap_usage(), size_t{1024},
                     "freestanding resource-limit core tracks current heap usage") &&
        expect_equal(eshkol_get_peak_heap_usage(), size_t{1024},
                     "freestanding resource-limit core tracks peak heap usage") &&
        expect_equal(eshkol_get_remaining_time_ms(), uint64_t{1000},
                     "freestanding resource-limit core uses the runtime monotonic hook for remaining time") &&
        expect_equal(eshkol_is_timed_out(), true,
                     "freestanding resource-limit core trips timeout through the runtime monotonic hook") &&
        expect_equal(eshkol_get_last_limit_error(), ESHKOL_LIMIT_TIMEOUT,
                     "freestanding resource-limit core records timeout as the last limit error") &&
        expect_equal(eshkol_runtime_get_shutdown_reason(), ESHKOL_SHUTDOWN_NONE,
                     "freestanding resource-limit core does not request shutdown when hard limits are disabled") &&
        expect_equal(clock_capture.call_count, size_t{3},
                     "freestanding resource-limit core consults the runtime monotonic hook for start and checks");

    eshkol_track_deallocation(1024);
    eshkol_stop_timer();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();

    return ok &&
           expect_equal(eshkol_get_heap_usage(), size_t{0},
                        "freestanding resource-limit core resets tracked heap usage cleanly");
}

bool test_runtime_freestanding_raise_surface() {
#ifdef _WIN32
    return true;
#else
    char fatal_path[] = "/tmp/eshkol-runtime-freestanding-fatal-XXXXXX";
    const int fatal_fd = mkstemp(fatal_path);
    if (fatal_fd == -1) {
        std::cerr << "FAIL: freestanding fatal temp path" << std::endl;
        return false;
    }
    close(fatal_fd);

    const pid_t pid = fork();
    if (pid == -1) {
        std::cerr << "FAIL: freestanding fatal fork" << std::endl;
        std::remove(fatal_path);
        return false;
    }

    if (pid == 0) {
        eshkol_runtime_set_fatal_hook(capture_fatal_hook, fatal_path);
        eshkol_raise(eshkol_make_exception(ESHKOL_EXCEPTION_USER_DEFINED,
                                           "freestanding fatal path"));
        _exit(101);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) == -1) {
        std::cerr << "FAIL: freestanding fatal waitpid" << std::endl;
        std::remove(fatal_path);
        return false;
    }

    std::ifstream input(fatal_path);
    std::string message;
    std::getline(input, message);
    std::remove(fatal_path);

    const bool terminated =
        WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0);

    return expect_equal(terminated, true,
                        "freestanding raise still terminates through the default fatal sink") &&
           expect_contains(message, "freestanding fatal path",
                           "freestanding raise routes the exception message through the fatal hook");
#endif
}

bool test_runtime_freestanding_exception_handler_surface() {
#ifdef _WIN32
    return true;
#else
    const pid_t pid = fork();
    if (pid == -1) {
        std::cerr << "FAIL: freestanding handler fork" << std::endl;
        return false;
    }

    if (pid == 0) {
        const size_t embedded_bytes = eshkol_arena_embedded_bytes(4096);
        std::vector<unsigned char> storage(embedded_bytes);
        eshkol_arena_t* arena =
            eshkol_arena_init_embedded(storage.data(), storage.size());
        if (!arena) {
            _exit(111);
        }
        if (!eshkol_arena_bind_runtime_global(arena)) {
            _exit(112);
        }

        std::vector<CapturedDiagnostic> captures;
        eshkol_runtime_set_diagnostic_hook(capture_diagnostic_hook, &captures);

        eshkol_tagged_value_t raised_value{};
        raised_value.type = ESHKOL_VALUE_INT64;
        raised_value.flags = 0;
        raised_value.reserved = 0;
        raised_value.data.int_val = 99;
        eshkol_set_raised_value(&raised_value);

        jmp_buf env;
        eshkol_push_exception_handler(&env);
        const int jumped = setjmp(env);
        if (jumped == 0) {
            eshkol_exception_t* exc = eshkol_make_exception_with_header(
                ESHKOL_EXCEPTION_USER_DEFINED, "handled freestanding exception");
            if (!exc) {
                _exit(113);
            }
            if (ESHKOL_GET_SUBTYPE(exc) != HEAP_SUBTYPE_EXCEPTION) {
                _exit(114);
            }
            eshkol_exception_set_location(exc, 12, 34, "boot.esk");
            eshkol_exception_add_irritant(exc, raised_value);
            eshkol_raise(exc);
            _exit(115);
        }

        eshkol_pop_exception_handler();

        eshkol_exception_t* current = eshkol_get_current_exception();
        if (!current) {
            _exit(116);
        }
        if (!eshkol_exception_type_matches(current, ESHKOL_EXCEPTION_USER_DEFINED)) {
            _exit(117);
        }
        if (!current->message ||
            std::string(current->message) != "handled freestanding exception") {
            _exit(118);
        }
        if (current->line != 12 || current->column != 34) {
            _exit(119);
        }
        if (!current->filename || std::string(current->filename) != "boot.esk") {
            _exit(120);
        }
        if (current->num_irritants != 1) {
            _exit(121);
        }

        eshkol_tagged_value_t recovered{};
        eshkol_get_raised_value(&recovered);
        if (recovered.type != ESHKOL_VALUE_INT64 ||
            recovered.data.int_val != 99) {
            _exit(122);
        }

        eshkol_display_exception(current);
        if (captures.empty() ||
            captures.back().level != ESHKOL_RUNTIME_DIAGNOSTIC_ERROR ||
            captures.back().message.find("handled freestanding exception") ==
                std::string::npos) {
            _exit(123);
        }

        eshkol_clear_current_exception();
        if (eshkol_get_current_exception() != nullptr) {
            _exit(124);
        }
        eshkol_tagged_value_t cleared{};
        eshkol_get_raised_value(&cleared);
        if (!expect_null_tagged_value(
                cleared,
                "freestanding clear_current_exception resets stale raised-value state")) {
            _exit(127);
        }

        eshkol_reset_recursion_depth();
        if (eshkol_check_recursion_depth() != 1) {
            _exit(125);
        }
        eshkol_decrement_recursion_depth();
        if (eshkol_check_recursion_depth() != 1) {
            _exit(126);
        }
        eshkol_reset_recursion_depth();
        _exit(0);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) == -1) {
        std::cerr << "FAIL: freestanding handler waitpid" << std::endl;
        return false;
    }

    return expect_equal(WIFEXITED(status) && WEXITSTATUS(status) == 0, true,
                        "freestanding exception handler path preserves raised values and helper state");
#endif
}

bool test_runtime_freestanding_lifecycle_surface() {
    eshkol_init_stack_size();

    const size_t embedded_bytes = eshkol_arena_embedded_bytes(2048);
    std::vector<unsigned char> storage(embedded_bytes);
    eshkol_arena_t* arena =
        eshkol_arena_init_embedded(storage.data(), storage.size());
    if (!arena) {
        std::cerr << "FAIL: freestanding embedded arena bootstrap" << std::endl;
        return false;
    }

    std::vector<std::string> order;
    std::vector<eshkol_shutdown_reason_t> reasons;
    std::vector<int> drain_timeouts;
    uint32_t operation_to_end = 0;
    ShutdownCapture first{&order, &reasons, "first"};
    ShutdownCapture second{&order, &reasons, "second"};
    OperationDrainCapture drain_capture{&drain_timeouts, &operation_to_end, 0};

    const uint32_t first_hook =
        eshkol_register_shutdown_hook(capture_shutdown_hook, &first, "first");
    const uint32_t second_hook =
        eshkol_register_shutdown_hook(capture_shutdown_hook, &second, "second");
    const uint32_t op = eshkol_runtime_begin_operation("boot");
    operation_to_end = op;

    bool ok = expect_equal(eshkol_runtime_get_state(), ESHKOL_RUNTIME_INITIALIZING,
                           "freestanding runtime starts in initializing state") &&
              expect_equal(eshkol_arena_bind_runtime_global(arena), true,
                           "freestanding runtime can bind an embedded global arena") &&
              expect_equal(first_hook != 0, true,
                           "freestanding runtime registers the first shutdown hook") &&
              expect_equal(second_hook != 0, true,
                           "freestanding runtime registers the second shutdown hook") &&
              expect_equal(op != 0, true,
                           "freestanding runtime begins tracked operations") &&
              expect_equal(eshkol_runtime_get_operation_count(), uint32_t{1},
                           "freestanding runtime counts active operations") &&
              expect_equal(eshkol_runtime_drain_operations(0), false,
                           "freestanding runtime reports pending operations before completion") &&
              expect_equal(eshkol_runtime_drain_operations(25), false,
                           "freestanding runtime cannot wait for operations without an installed drain hook") &&
              expect_equal(eshkol_runtime_get_operation_count(), uint32_t{1},
                           "freestanding runtime keeps operations pending when no drain hook exists") &&
              expect_equal(drain_timeouts.empty(), true,
                           "freestanding runtime does not synthesize drain callbacks before the hook runs") &&
              expect_equal(eshkol_runtime_init(), 0,
                           "freestanding runtime initializes cleanly") &&
              expect_equal(eshkol_runtime_get_state(), ESHKOL_RUNTIME_RUNNING,
                           "freestanding runtime enters running state");

    eshkol_runtime_set_operation_drain_hook(capture_operation_drain_hook,
                                            &drain_capture);
    ok = ok &&
         expect_equal(eshkol_runtime_drain_operations(25), true,
                      "freestanding runtime can cooperatively drain operations through the BSP hook") &&
         expect_equal(eshkol_runtime_get_operation_count(), uint32_t{0},
                      "freestanding runtime drain hook can complete the active operation") &&
         expect_equal(drain_capture.call_count, size_t{1},
                      "freestanding runtime invokes the drain hook once for the explicit drain request") &&
         expect_equal(drain_timeouts.size(), size_t{1},
                      "freestanding runtime records one explicit drain hook timeout") &&
         expect_equal(drain_timeouts[0], 25,
                      "freestanding runtime forwards explicit drain timeouts to the BSP hook");

    auto* exc =
        eshkol_make_exception(ESHKOL_EXCEPTION_DIVIDE_BY_ZERO, "divide by zero");
    ok = ok &&
         expect_equal(exc != nullptr, true,
                      "freestanding runtime produces a minimal exception object") &&
         expect_equal(eshkol_exception_type_matches(exc,
                                                   ESHKOL_EXCEPTION_DIVIDE_BY_ZERO),
                      1,
                      "freestanding runtime preserves exception type matching");

    const uint32_t shutdown_op = eshkol_runtime_begin_operation("shutdown");
    operation_to_end = shutdown_op;
    ok = ok &&
         expect_equal(shutdown_op != 0, true,
                      "freestanding runtime can begin another tracked operation before shutdown") &&
         expect_equal(eshkol_runtime_get_operation_count(), uint32_t{1},
                      "freestanding runtime counts the shutdown-time operation");

    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_ERROR);
    const bool result =
        ok &&
        expect_equal(eshkol_runtime_get_state(), ESHKOL_RUNTIME_TERMINATED,
                     "freestanding runtime enters terminated state after shutdown") &&
        expect_equal(eshkol_runtime_get_operation_count(), uint32_t{0},
                     "freestanding runtime shutdown drains the final operation through the BSP hook") &&
        expect_equal(eshkol_runtime_get_shutdown_reason(),
                     ESHKOL_SHUTDOWN_ERROR,
                     "freestanding runtime records the shutdown reason") &&
        expect_equal(drain_capture.call_count, size_t{2},
                     "freestanding runtime also uses the drain hook during shutdown") &&
        expect_equal(drain_timeouts.size(), size_t{2},
                     "freestanding runtime records the shutdown drain timeout") &&
        expect_equal(drain_timeouts[1], 5000,
                     "freestanding runtime shutdown uses the bounded drain timeout contract") &&
        expect_equal(order.size(), size_t{2},
                     "freestanding runtime calls both registered shutdown hooks") &&
        expect_equal(order[0], std::string("second"),
                     "freestanding runtime calls shutdown hooks in reverse registration order") &&
        expect_equal(order[1], std::string("first"),
                     "freestanding runtime preserves earlier shutdown hooks") &&
        expect_equal(reasons[0], ESHKOL_SHUTDOWN_ERROR,
                     "freestanding runtime passes the shutdown reason to the newest hook") &&
        expect_equal(reasons[1], ESHKOL_SHUTDOWN_ERROR,
                     "freestanding runtime passes the shutdown reason to the oldest hook");
    eshkol_runtime_clear_operation_drain_hook();
    return result;
}

}  // namespace

int main() {
    if (!test_runtime_freestanding_capability_registry_surface()) {
        return 1;
    }
    if (!test_runtime_freestanding_diagnostic_hooks()) {
        return 1;
    }
    if (!test_runtime_freestanding_timebase_surface()) {
        return 1;
    }
    if (!test_runtime_freestanding_resource_limits_surface()) {
        return 1;
    }
    if (!test_runtime_freestanding_interrupt_surface()) {
        return 1;
    }
    if (!test_runtime_freestanding_raise_surface()) {
        return 1;
    }
    if (!test_runtime_freestanding_exception_handler_surface()) {
        return 1;
    }
    if (!test_runtime_freestanding_lifecycle_surface()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
