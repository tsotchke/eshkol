#include "../../lib/backend/blas_peak_calibration.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    using eshkol::blas::detail::PeakCalibration;

    PeakCalibration calibration(1100.0);
    if (!expect(calibration.effective() == 1100.0, "fallback is effective before samples") ||
        !expect(calibration.record(405.0), "first valid sample is recorded") ||
        !expect(calibration.effective() == 405.0,
                "first sample below fallback replaces fallback") ||
        !expect(calibration.record(300.0), "lower valid sample is recorded") ||
        !expect(calibration.effective() == 405.0,
                "lower sample does not reduce high-water") ||
        !expect(calibration.record(750.0), "higher valid sample is recorded") ||
        !expect(calibration.effective() == 750.0,
                "higher sample raises high-water")) {
        return 1;
    }

    const unsigned valid_samples = calibration.samples();
    if (!expect(!calibration.record(std::numeric_limits<double>::quiet_NaN()),
                "NaN sample is rejected") ||
        !expect(!calibration.record(std::numeric_limits<double>::infinity()),
                "infinite sample is rejected") ||
        !expect(!calibration.record(0.0), "zero sample is rejected") ||
        !expect(!calibration.record(-1.0), "negative sample is rejected") ||
        !expect(calibration.samples() == valid_samples,
                "invalid samples do not change sample count")) {
        return 1;
    }

    calibration.override(42.0);
    if (!expect(!calibration.enabled(), "override disables calibration") ||
        !expect(calibration.effective() == 42.0,
                "override fallback is authoritative after samples") ||
        !expect(!calibration.record(1000.0), "samples after override are rejected") ||
        !expect(calibration.samples() == valid_samples,
                "rejected override sample does not change count")) {
        return 1;
    }

    constexpr int kWriterCount = 16;
    PeakCalibration concurrent(1.0);
    std::vector<std::thread> writers;
    writers.reserve(kWriterCount);
    for (int value = 1; value <= kWriterCount; ++value) {
        writers.emplace_back([&concurrent, value] {
            concurrent.record(static_cast<double>(value));
        });
    }
    for (auto& writer : writers) {
        writer.join();
    }

    if (!expect(concurrent.effective() == static_cast<double>(kWriterCount),
                "concurrent writers retain the exact maximum") ||
        !expect(concurrent.samples() == static_cast<unsigned>(kWriterCount),
                "concurrent valid writers retain the exact sample count")) {
        return 1;
    }

    return 0;
}
