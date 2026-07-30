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
//   4. A child that exits immediately is never misreported as a timeout.
//
// THE CHILD IS THIS TEST BINARY, re-executed with a mode argument. It used to
// be `/bin/sleep 30` (POSIX) and `cmd /c ping` (Windows), and that made the
// test a probe of the host's filesystem layout rather than of the timeout:
// `/bin/sleep` is not guaranteed by POSIX or by any Linux standard, and on a
// NixOS host `/bin` contains exactly one entry — `sh`. execvp() therefore
// returned ENOENT, run_subprocess() correctly reported 127 ("program not
// found"), and the test failed with `expected SUBPROCESS_TIMEOUT from a hung
// child, got 127` in 0.02s — never having started a child to time out at all.
// That reproduced on the mesh's aarch64 node and read as an architecture bug;
// it was the distribution's /bin, and it would reproduce identically on an x86
// NixOS host.
//
// Re-executing ourselves removes the external dependency completely: the one
// executable we are guaranteed to be able to launch is the one already running,
// named by argv[0] — which both back ends of run_subprocess resolve exactly the
// way this process was itself resolved, so no path constant is needed either.

#include <eshkol/pkg/subprocess.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

// Child modes. argv[1] selects one; with no argument the binary is the parent
// and runs the assertions below.
constexpr const char* kModeHang = "--child-hang";
constexpr const char* kModeQuick = "--child-quick";
constexpr const char* kModeBrief = "--child-brief";

// A child that does real but clearly sub-second work. Used to pin that the
// bound is a duration and not "until the wall clock's next whole second".
constexpr int kBriefMilliseconds = 150;

// How long the hung child sleeps if nobody kills it. Long enough that a
// working timeout is unambiguous, bounded so a broken run cannot wedge CTest.
constexpr int kHangSeconds = 30;

} // namespace

int main(int argc, char** argv) {
    using namespace std::chrono;

    // ---- child modes -----------------------------------------------------
    if (argc > 1) {
        const std::string mode = argv[1];
        if (mode == kModeHang) {
            std::this_thread::sleep_for(seconds(kHangSeconds));
            return 0; // only reached if the parent's timeout did NOT fire
        }
        if (mode == kModeQuick) {
            return 0;
        }
        if (mode == kModeBrief) {
            std::this_thread::sleep_for(milliseconds(kBriefMilliseconds));
            return 0;
        }
        return fail("unknown child mode: " + mode);
    }

    // ---- parent ----------------------------------------------------------
    // argv[0] is passed to the child launcher verbatim, and that resolves in
    // every way this process could itself have been launched: both back ends
    // of run_subprocess do a PATH search when the program has no separator
    // (POSIX execvp; CreateProcessW with a null lpApplicationName), and a
    // relative path still resolves because the child inherits our working
    // directory (cwd is null below). So no filesystem probing and no
    // build-time path constant are needed to name ourselves.
    const std::string self = (argc > 0 && argv[0] && *argv[0]) ? argv[0] : "";
    if (self.empty()) {
        return fail("argv[0] is empty — cannot name this executable to re-run "
                    "it as the child under test");
    }
    const std::vector<std::string> hang_long = {self, kModeHang};
    const std::vector<std::string> quick = {self, kModeQuick};
    const std::vector<std::string> brief = {self, kModeBrief};

    // 1. Hung child must be killed within the bound, not after kHangSeconds.
    {
        auto start = steady_clock::now();
        int rc = eshkol::pkg::run_subprocess(hang_long, nullptr, /*timeout=*/1);
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
        int rc = eshkol::pkg::run_subprocess(quick, nullptr, /*timeout=*/30);
        if (rc != 0) {
            return fail("expected exit 0 from a quick child, got " +
                        std::to_string(rc));
        }
    }

    // 3. timeout==0 keeps the historical behaviour for a quick child.
    {
        int rc = eshkol::pkg::run_subprocess(quick, nullptr, /*timeout=*/0);
        if (rc != 0) {
            return fail("expected exit 0 with unbounded wait, got " +
                        std::to_string(rc));
        }
    }

    // 4. `timeout` is a DURATION, not "until the wall clock's next whole
    //    second". The bounded wait used to compute its deadline as
    //    `std::time(nullptr) + timeout_seconds`, and integer-second truncation
    //    means the usable budget for a 1s bound was anywhere from ~0ms to
    //    1000ms depending on where the call happened to land inside the
    //    current second — so a perfectly healthy sub-second child could be
    //    SIGKILLed and misreported as SUBPROCESS_TIMEOUT. (The same expression
    //    is also not monotonic: a clock step forward kills a healthy child and
    //    a step backward defeats the bound entirely.)
    //
    //    Aligning to just before a second boundary makes that deterministic
    //    rather than a coin flip: with ~50ms left in the wall second, the old
    //    deadline expires 50ms after launch, while the child needs 150ms, so
    //    the old code times out every time. The monotonic deadline gives the
    //    child its full second and leaves ~850ms of margin, so this is not a
    //    tight race in the passing direction.
    {
        for (;;) {
            auto into_second = system_clock::now().time_since_epoch() % seconds(1);
            if (into_second >= milliseconds(950)) break;
            std::this_thread::sleep_for(milliseconds(2));
        }
        auto start = steady_clock::now();
        int rc = eshkol::pkg::run_subprocess(brief, nullptr, /*timeout=*/1);
        auto elapsed = duration_cast<milliseconds>(steady_clock::now() - start).count();
        if (rc == eshkol::pkg::SUBPROCESS_TIMEOUT) {
            return fail("a " + std::to_string(kBriefMilliseconds) +
                        "ms child under a 1s bound was reported as a timeout "
                        "after " + std::to_string(elapsed) +
                        "ms (deadline truncated to a wall-clock second?)");
        }
        if (rc != 0) {
            return fail("expected exit 0 from a brief child under a 1s bound, "
                        "got " + std::to_string(rc));
        }
    }

    std::cout << "PASS: subprocess timeout" << std::endl;
    return 0;
}
