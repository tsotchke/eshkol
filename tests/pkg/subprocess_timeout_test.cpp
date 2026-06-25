// Regression test for the bounded-wait timeout in
// inc/eshkol/pkg/subprocess.h.
//
// Before this fix, run_subprocess() always waited unboundedly
// (waitpid(pid, &status, 0) on POSIX, WaitForSingleObject(..., INFINITE) on
// Windows). A hung child — e.g. a stuck `ld` on a host whose link search path
// is misconfigured — wedged the whole process, so the `-r` JIT path never
// reached its interpreter fallback (Noesis observed this as eshkol-run sitting
// at 0% CPU indefinitely). The fix adds an optional timeout: on expiry the
// child is killed and SUBPROCESS_TIMEOUT (124) is returned so callers fail
// fast.
//
// This test asserts:
//   1. A child that sleeps far longer than the timeout is killed promptly and
//      returns SUBPROCESS_TIMEOUT.
//   2. A child that finishes within the timeout returns its real exit code.
//   3. timeout==0 preserves the historical unbounded (here: prompt) wait.

#include <eshkol/pkg/subprocess.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

#ifdef _WIN32
const std::vector<std::string> kSleepLong = {"cmd", "/c", "ping", "127.0.0.1",
                                             "-n", "30", ">nul"};
const std::vector<std::string> kQuick = {"cmd", "/c", "exit", "0"};
#else
const std::vector<std::string> kSleepLong = {"/bin/sleep", "30"};
const std::vector<std::string> kQuick = {"/bin/sh", "-c", "exit 0"};
#endif

} // namespace

int main() {
    using namespace std::chrono;

    // 1. Hung child must be killed within the bound, not after 30s.
    {
        auto start = steady_clock::now();
        int rc = eshkol::pkg::run_subprocess(kSleepLong, nullptr, /*timeout=*/1);
        auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start).count();
        if (rc != eshkol::pkg::SUBPROCESS_TIMEOUT) {
            return fail("expected SUBPROCESS_TIMEOUT from a hung child, got " +
                        std::to_string(rc));
        }
        if (elapsed > 10000) {
            return fail("bounded wait took too long: " + std::to_string(elapsed) +
                        "ms (timeout not enforced?)");
        }
    }

    // 2. A child that completes within the bound returns its real exit code.
    {
        int rc = eshkol::pkg::run_subprocess(kQuick, nullptr, /*timeout=*/30);
        if (rc != 0) {
            return fail("expected exit 0 from a quick child, got " +
                        std::to_string(rc));
        }
    }

    // 3. timeout==0 keeps the historical behaviour for a quick child.
    {
        int rc = eshkol::pkg::run_subprocess(kQuick, nullptr, /*timeout=*/0);
        if (rc != 0) {
            return fail("expected exit 0 with unbounded wait, got " +
                        std::to_string(rc));
        }
    }

    std::cout << "PASS: subprocess timeout" << std::endl;
    return 0;
}
