#pragma once

#include <atomic>
#include <cmath>

namespace eshkol::blas::detail {

// Process-wide BLAS throughput calibration.  The configured peak is a
// deterministic fallback; successful observations form a separate monotonic
// high-water mark so the first real sample can be below (and replace) it.
class PeakCalibration {
public:
    explicit PeakCalibration(double fallback) : fallback_(fallback) {}

    void disable() {
        enabled_.store(false, std::memory_order_relaxed);
    }

    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }

    // An explicit operator override is authoritative for the process.
    void override(double fallback) {
        fallback_.store(fallback, std::memory_order_relaxed);
        disable();
    }

    bool record(double sample) {
        if (!enabled() || !std::isfinite(sample) || sample <= 0.0) {
            return false;
        }

        samples_.fetch_add(1, std::memory_order_relaxed);
        double current = measured_.load(std::memory_order_relaxed);
        while (sample > current &&
               !measured_.compare_exchange_weak(
                   current, sample, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
        return true;
    }

    double effective() const {
        const double measured = measured_.load(std::memory_order_relaxed);
        if (enabled() && measured > 0.0) {
            return measured;
        }
        return fallback_.load(std::memory_order_relaxed);
    }

    unsigned samples() const {
        return samples_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<double> fallback_;
    std::atomic<double> measured_{0.0};
    std::atomic<unsigned> samples_{0};
    std::atomic<bool> enabled_{true};
};

}  // namespace eshkol::blas::detail
