/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted in-flight operation tracking for graceful runtime shutdown.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/logger.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct InFlightOperation {
    uint32_t id;
    std::string name;
    std::chrono::steady_clock::time_point start_time;
};

std::mutex g_operations_mutex;
std::condition_variable g_operations_cv;
std::vector<InFlightOperation> g_in_flight_operations;
std::atomic<uint32_t> g_next_operation_id{1};

}  // namespace

extern "C" {

uint32_t eshkol_runtime_begin_operation(const char* name) {
    std::lock_guard<std::mutex> lock(g_operations_mutex);

    uint32_t id = g_next_operation_id.fetch_add(1, std::memory_order_relaxed);

    g_in_flight_operations.push_back({
        .id = id,
        .name = name ? name : "(unnamed)",
        .start_time = std::chrono::steady_clock::now()
    });

    eshkol_debug("Started operation %u: %s", id, name ? name : "(unnamed)");
    return id;
}

void eshkol_runtime_end_operation(uint32_t operation_id) {
    if (operation_id == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_operations_mutex);

    auto it = std::find_if(g_in_flight_operations.begin(), g_in_flight_operations.end(),
                           [operation_id](const InFlightOperation& op) {
                               return op.id == operation_id;
                           });

    if (it != g_in_flight_operations.end()) {
        auto duration = std::chrono::steady_clock::now() - it->start_time;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        eshkol_debug("Completed operation %u: %s (took %lld ms)",
                     operation_id, it->name.c_str(), (long long)ms);
        g_in_flight_operations.erase(it);
    }

    g_operations_cv.notify_all();
}

bool eshkol_runtime_drain_operations(int timeout_ms) {
    std::unique_lock<std::mutex> lock(g_operations_mutex);

    if (timeout_ms == 0) {
        return g_in_flight_operations.empty();
    }

    if (timeout_ms < 0) {
        g_operations_cv.wait(lock, []() {
            return g_in_flight_operations.empty();
        });
        return true;
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    return g_operations_cv.wait_until(lock, deadline, []() {
        return g_in_flight_operations.empty();
    });
}

uint32_t eshkol_runtime_get_operation_count(void) {
    std::lock_guard<std::mutex> lock(g_operations_mutex);
    return static_cast<uint32_t>(g_in_flight_operations.size());
}

}  // extern "C"
