#include <eshkol/core/runtime.h>

#include <cstdio>
#include <cstdlib>
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

struct CapturedMonotonicClock {
    uint64_t next_time_ns;
    uint64_t step_ns;
    size_t call_count;
};

struct CapturedDelay {
    std::vector<uint64_t> durations_ns;
    bool result;
};

struct CapturedOperationDrain {
    std::vector<int> timeouts_ms;
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

extern "C" bool capture_monotonic_clock_hook(uint64_t* out_time_ns,
                                             void* context) {
    auto* capture = static_cast<CapturedMonotonicClock*>(context);
    if (!capture || !out_time_ns) {
        return false;
    }

    *out_time_ns = capture->next_time_ns;
    capture->next_time_ns += capture->step_ns;
    ++capture->call_count;
    return true;
}

extern "C" bool capture_delay_hook(uint64_t duration_ns, void* context) {
    auto* capture = static_cast<CapturedDelay*>(context);
    if (!capture) {
        return false;
    }

    capture->durations_ns.push_back(duration_ns);
    return capture->result;
}

extern "C" bool capture_operation_drain_hook(int timeout_ms, void* context) {
    auto* capture = static_cast<CapturedOperationDrain*>(context);
    if (!capture) {
        return false;
    }

    capture->timeouts_ms.push_back(timeout_ms);
    return capture->result;
}

bool test_runtime_capability_registry_surface() {
    eshkol_runtime_clear_diagnostic_hook();
    eshkol_runtime_clear_fatal_hook();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_delay_hook();
    eshkol_runtime_clear_operation_drain_hook();

    eshkol_runtime_capability_descriptor_t descriptor{};
    bool ok =
        expect_equal(eshkol_runtime_get_capability_count(), size_t{5},
                     "runtime capability registry exposes the bounded hook family") &&
        expect_equal(eshkol_runtime_describe_capability_at(
                         0, &descriptor),
                     true,
                     "runtime capability registry can describe the first capability") &&
        expect_equal(descriptor.kind,
                     ESHKOL_RUNTIME_CAPABILITY_DIAGNOSTIC_SINK,
                     "runtime capability registry keeps diagnostic sink first") &&
        expect_equal(std::string(descriptor.name), std::string("diagnostic-sink"),
                     "runtime capability registry names the diagnostic sink") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, true,
                        "runtime diagnostic capability reports a default sink") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLABLE, true,
                        "runtime diagnostic capability reports hook installation support") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, false,
                        "runtime diagnostic capability starts without a custom hook") &&
        expect_equal(eshkol_runtime_describe_capability(
                         ESHKOL_RUNTIME_CAPABILITY_MONOTONIC_CLOCK,
                         &descriptor),
                     true,
                     "runtime capability registry can describe monotonic clock") &&
        expect_equal(std::string(descriptor.name), std::string("monotonic-clock"),
                     "runtime capability registry names the monotonic clock capability") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, true,
                        "hosted runtime reports a default monotonic clock capability") &&
        expect_flag_set(descriptor.flags,
                        ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, false,
                        "hosted runtime monotonic capability starts without a custom hook") &&
        expect_equal(eshkol_runtime_describe_capability_at(
                         eshkol_runtime_get_capability_count(), &descriptor),
                     false,
                     "runtime capability registry rejects out-of-range enumeration") &&
        expect_equal(eshkol_runtime_describe_capability(
                         static_cast<eshkol_runtime_capability_kind_t>(99),
                         &descriptor),
                     false,
                     "runtime capability registry rejects unknown capability kinds");

    std::vector<CapturedDiagnostic> captures;
    CapturedMonotonicClock clock_capture{42, 1, 0};
    CapturedDelay delay_capture{{}, true};
    CapturedOperationDrain drain_capture{{}, true};
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
                      "runtime capability registry can re-read diagnostic state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "runtime diagnostic capability reflects an installed custom hook") &&
         expect_equal(eshkol_runtime_describe_capability(
                          ESHKOL_RUNTIME_CAPABILITY_MONOTONIC_CLOCK,
                          &descriptor),
                      true,
                      "runtime capability registry can re-read monotonic clock state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "runtime monotonic capability reflects an installed custom hook") &&
         expect_equal(eshkol_runtime_describe_capability(
                          ESHKOL_RUNTIME_CAPABILITY_DELAY, &descriptor),
                      true,
                      "runtime capability registry can describe delay state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, true,
                         "hosted runtime delay capability reports a default delay primitive") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "runtime delay capability reflects an installed custom hook") &&
         expect_equal(eshkol_runtime_describe_capability(
                          ESHKOL_RUNTIME_CAPABILITY_OPERATION_DRAIN,
                          &descriptor),
                      true,
                      "runtime capability registry can describe operation-drain state") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE, false,
                         "runtime operation-drain capability does not claim an implicit default hook") &&
         expect_flag_set(descriptor.flags,
                         ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, true,
                         "runtime operation-drain capability reflects an installed custom hook");

    eshkol_runtime_clear_operation_drain_hook();
    eshkol_runtime_clear_delay_hook();
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_diagnostic_hook();

    return ok &&
           expect_equal(eshkol_runtime_describe_capability(
                            ESHKOL_RUNTIME_CAPABILITY_OPERATION_DRAIN,
                            &descriptor),
                        true,
                        "runtime capability registry remains readable after hook clear") &&
           expect_flag_set(descriptor.flags,
                           ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED, false,
                           "runtime operation-drain capability clears the installed-hook bit");
}

bool test_runtime_diagnostic_hook_surface() {
    std::vector<CapturedDiagnostic> captures;

    eshkol_runtime_set_diagnostic_hook(capture_diagnostic_hook, &captures);
    eshkol_runtime_debugf("debug value %d", 7);
    eshkol_runtime_infof("info value %s", "limits");
    eshkol_runtime_warnf("warn value %s", "uart");
    eshkol_runtime_errorf("error value %d", 11);
    eshkol_runtime_clear_diagnostic_hook();

    const bool ok =
        expect_equal(captures.size(), size_t{4},
                     "diagnostic hook captures debug/info/warn/error calls") &&
        expect_equal(captures[0].level, ESHKOL_RUNTIME_DIAGNOSTIC_DEBUG,
                     "diagnostic hook preserves debug level") &&
        expect_contains(captures[0].message, "debug value 7",
                        "diagnostic hook formats debug messages") &&
        expect_equal(captures[1].level, ESHKOL_RUNTIME_DIAGNOSTIC_INFO,
                     "diagnostic hook preserves info level") &&
        expect_contains(captures[1].message, "info value limits",
                        "diagnostic hook formats info messages") &&
        expect_equal(captures[2].level, ESHKOL_RUNTIME_DIAGNOSTIC_WARNING,
                     "diagnostic hook preserves warning level") &&
        expect_contains(captures[2].message, "warn value uart",
                        "diagnostic hook formats warning messages") &&
        expect_equal(captures[3].level, ESHKOL_RUNTIME_DIAGNOSTIC_ERROR,
                     "diagnostic hook preserves error level") &&
        expect_contains(captures[3].message, "error value 11",
                        "diagnostic hook formats error messages");

    eshkol_runtime_debugf("post-clear diagnostic");
    return ok &&
           expect_equal(captures.size(), size_t{4},
                        "clearing the diagnostic hook stops custom capture");
}

bool test_runtime_timebase_hook_surface() {
    eshkol_runtime_clear_monotonic_clock_hook();
    eshkol_runtime_clear_delay_hook();

    uint64_t default_first = 0;
    uint64_t default_second = 0;
    const bool default_ok =
        expect_equal(eshkol_runtime_get_monotonic_time_ns(nullptr), false,
                     "monotonic time query rejects null output storage") &&
        expect_equal(eshkol_runtime_get_monotonic_time_ns(&default_first), true,
                     "hosted runtime exposes a default monotonic time source") &&
        expect_equal(eshkol_runtime_get_monotonic_time_ns(&default_second), true,
                     "hosted runtime can read the monotonic time source repeatedly") &&
        expect_equal(default_second >= default_first, true,
                     "hosted monotonic time does not move backwards") &&
        expect_equal(eshkol_runtime_delay_ns(0), true,
                     "zero-duration runtime delay is always a successful no-op");

    CapturedMonotonicClock clock_capture{1000, 250, 0};
    CapturedDelay delay_capture{{}, true};
    eshkol_runtime_set_monotonic_clock_hook(capture_monotonic_clock_hook,
                                            &clock_capture);
    eshkol_runtime_set_delay_hook(capture_delay_hook, &delay_capture);

    uint64_t hook_first = 0;
    uint64_t hook_second = 0;
    const bool hook_ok =
        expect_equal(eshkol_runtime_get_monotonic_time_ns(&hook_first), true,
                     "hosted runtime can override monotonic time through a hook") &&
        expect_equal(eshkol_runtime_get_monotonic_time_ns(&hook_second), true,
                     "hosted runtime reuses the installed monotonic time hook") &&
        expect_equal(hook_first, uint64_t{1000},
                     "hosted monotonic time hook controls the returned value") &&
        expect_equal(hook_second, uint64_t{1250},
                     "hosted monotonic time hook advances under caller control") &&
        expect_equal(clock_capture.call_count, size_t{2},
                     "hosted monotonic time hook records both reads") &&
        expect_equal(eshkol_runtime_delay_ns(777), true,
                     "hosted runtime can override delay through a hook") &&
        expect_equal(delay_capture.durations_ns.size(), size_t{1},
                     "hosted delay hook records the call") &&
        expect_equal(delay_capture.durations_ns[0], uint64_t{777},
                     "hosted delay hook receives the requested duration");

    eshkol_runtime_clear_delay_hook();
    eshkol_runtime_clear_monotonic_clock_hook();

    uint64_t cleared_time = 0;
    return default_ok && hook_ok &&
           expect_equal(eshkol_runtime_get_monotonic_time_ns(&cleared_time), true,
                        "hosted runtime falls back to the default monotonic time source after hook clear") &&
           expect_equal(eshkol_runtime_delay_ns(0), true,
                        "hosted runtime still accepts zero-duration delay after hook clear");
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

bool test_runtime_fatal_hook_surface() {
#ifdef _WIN32
    return true;
#else
    char fatal_path[] = "/tmp/eshkol-runtime-fatal-hook-XXXXXX";
    const int fatal_fd = mkstemp(fatal_path);
    if (fatal_fd == -1) {
        std::cerr << "FAIL: fatal hook temp path" << std::endl;
        return false;
    }
    close(fatal_fd);

    const pid_t pid = fork();
    if (pid == -1) {
        std::cerr << "FAIL: fatal hook fork" << std::endl;
        std::remove(fatal_path);
        return false;
    }

    if (pid == 0) {
        eshkol_runtime_set_fatal_hook(capture_fatal_hook, fatal_path);
        eshkol_runtime_fatalf("fatal hook %d", 42);
        _exit(101);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) == -1) {
        std::cerr << "FAIL: fatal hook waitpid" << std::endl;
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
                        "fatal hook still terminates through the default fatal sink") &&
           expect_contains(message, "fatal hook 42",
                           "fatal hook receives the formatted message before termination");
#endif
}

}  // namespace

int main() {
    if (!test_runtime_capability_registry_surface()) {
        return 1;
    }
    if (!test_runtime_diagnostic_hook_surface()) {
        return 1;
    }
    if (!test_runtime_timebase_hook_surface()) {
        return 1;
    }
    if (!test_runtime_fatal_hook_surface()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
