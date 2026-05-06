#include <eshkol/core/resource_limits.h>
#include <eshkol/core/runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct MonotonicClockCapture {
    std::vector<uint64_t> timestamps_ns;
    size_t next_index;
    size_t call_count;
    bool should_fail;
};

struct DiagnosticCapture {
    std::vector<eshkol_runtime_diagnostic_level_t> levels;
    std::vector<std::string> messages;
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

extern "C" bool capture_monotonic_clock_hook(uint64_t* out_time_ns,
                                             void* context) {
    auto* capture = static_cast<MonotonicClockCapture*>(context);
    if (!capture || !out_time_ns) {
        return false;
    }

    ++capture->call_count;
    if (capture->should_fail || capture->timestamps_ns.empty()) {
        return false;
    }

    const size_t index = capture->next_index < capture->timestamps_ns.size()
                             ? capture->next_index
                             : capture->timestamps_ns.size() - 1;
    *out_time_ns = capture->timestamps_ns[index];
    if (capture->next_index < capture->timestamps_ns.size()) {
        ++capture->next_index;
    }
    return true;
}

extern "C" void capture_diagnostic_hook(
    eshkol_runtime_diagnostic_level_t level,
    const char* message,
    void* context) {
    auto* capture = static_cast<DiagnosticCapture*>(context);
    if (!capture) {
        return;
    }

    capture->levels.push_back(level);
    capture->messages.emplace_back(message ? message : "");
}

bool test_resource_limits_timer_uses_runtime_timebase() {
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();

    eshkol_resource_limits_t limits = eshkol_get_default_limits();
    limits.enforce_hard_limits = false;
    limits.enable_warnings = false;
    eshkol_set_limits(&limits);

    MonotonicClockCapture capture{
        {
            1000000000ULL,
            1500000000ULL,
            2900000000ULL,
            3100000000ULL,
            3100000000ULL,
        },
        0,
        0,
        false,
    };
    eshkol_runtime_set_monotonic_clock_hook(capture_monotonic_clock_hook,
                                            &capture);

    eshkol_start_timer(2000);
    const bool ok =
        expect_equal(eshkol_get_remaining_time_ms(), uint64_t{1500},
                     "resource limit timer uses runtime monotonic time for remaining time") &&
        expect_equal(eshkol_is_timed_out(), false,
                     "resource limit timer stays below timeout while hook time is in range") &&
        expect_equal(eshkol_is_timed_out(), true,
                     "resource limit timer trips once hook time crosses the limit") &&
        expect_equal(eshkol_get_remaining_time_ms(), uint64_t{0},
                     "resource limit timer reports zero remaining time after timeout") &&
        expect_equal(eshkol_get_last_limit_error(), ESHKOL_LIMIT_TIMEOUT,
                     "resource limit timer records timeout as the last limit error") &&
        expect_equal(eshkol_runtime_get_shutdown_reason(), ESHKOL_SHUTDOWN_NONE,
                     "resource limit timer does not request shutdown when hard limits are disabled") &&
        expect_equal(capture.call_count, size_t{5},
                     "resource limit timer consults the runtime monotonic hook for start and checks");

    eshkol_stop_timer();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    return ok;
}

bool test_resource_limits_timer_disables_when_timebase_fails() {
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();

    eshkol_resource_limits_t limits = eshkol_get_default_limits();
    limits.enforce_hard_limits = false;
    limits.enable_warnings = false;
    eshkol_set_limits(&limits);

    MonotonicClockCapture capture{{}, 0, 0, true};
    eshkol_runtime_set_monotonic_clock_hook(capture_monotonic_clock_hook,
                                            &capture);
    DiagnosticCapture diagnostics;
    eshkol_runtime_set_diagnostic_hook(capture_diagnostic_hook, &diagnostics);

    eshkol_start_timer(1000);
    const bool ok =
        expect_equal(eshkol_get_remaining_time_ms(), uint64_t{0},
                     "resource limit timer stays inactive when the runtime timebase hook fails") &&
        expect_equal(eshkol_is_timed_out(), false,
                     "resource limit timer does not synthesize a timeout without a working runtime timebase") &&
        expect_equal(diagnostics.levels.size(), size_t{1},
                     "resource limit timer reports one warning when timebase start fails") &&
        expect_equal(diagnostics.levels[0], ESHKOL_RUNTIME_DIAGNOSTIC_WARNING,
                     "resource limit timer timebase failure is a warning") &&
        expect_contains(diagnostics.messages[0], "monotonic time source unavailable",
                        "resource limit timer warning explains the missing runtime timebase") &&
        expect_equal(capture.call_count, size_t{1},
                     "resource limit timer attempts to read the runtime timebase during start");

    eshkol_stop_timer();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    return ok;
}

bool test_unlimited_resource_limits_timer_does_not_require_timebase() {
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();

    eshkol_resource_limits_t limits = eshkol_get_default_limits();
    limits.max_execution_time_ms = 0;
    limits.enforce_hard_limits = false;
    limits.enable_warnings = false;
    eshkol_set_limits(&limits);

    MonotonicClockCapture capture{{}, 0, 0, true};
    eshkol_runtime_set_monotonic_clock_hook(capture_monotonic_clock_hook,
                                            &capture);

    eshkol_start_timer(0);
    const bool ok =
        expect_equal(eshkol_get_remaining_time_ms(), UINT64_MAX,
                     "unlimited resource limit timer reports no timeout ceiling") &&
        expect_equal(eshkol_is_timed_out(), false,
                     "unlimited resource limit timer never reports timeout without a ceiling") &&
        expect_equal(eshkol_get_last_limit_error(), ESHKOL_LIMIT_OK,
                     "unlimited resource limit timer preserves a clean error state") &&
        expect_equal(capture.call_count, size_t{0},
                     "unlimited resource limit timer does not consult the runtime timebase");

    eshkol_stop_timer();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();
    eshkol_reset_resource_tracking();
    eshkol_runtime_clear_interrupt();
    return ok;
}

}  // namespace

int main() {
    if (!test_resource_limits_timer_uses_runtime_timebase()) {
        return 1;
    }
    if (!test_resource_limits_timer_disables_when_timebase_fails()) {
        return 1;
    }
    if (!test_unlimited_resource_limits_timer_does_not_require_timebase()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
