#include <eshkol/frontend/diagnostic.h>

#include <atomic>
#include <mutex>

namespace {

std::mutex g_sink_mutex;
eshkol_diagnostic_sink_v1 g_sink = nullptr;
void* g_sink_userdata = nullptr;
std::atomic<uint64_t> g_count{0};

}

extern "C" void eshkol_diagnostic_set_sink_v1(eshkol_diagnostic_sink_v1 sink,
                                                void* userdata) {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    g_sink = sink;
    g_sink_userdata = userdata;
}

extern "C" void eshkol_diagnostic_emit_v1(eshkol_diagnostic_severity_t severity,
                                           eshkol_node_id_t node_id,
                                           const char* code,
                                           const char* message) {
    eshkol_diagnostic_v1_t diagnostic{};
    diagnostic.schema = ESHKOL_DIAGNOSTIC_V1_SCHEMA;
    diagnostic.severity = severity;
    diagnostic.node_id = node_id;
    diagnostic.code = code ? code : "";
    diagnostic.message = message ? message : "";
    (void)eshkol_node_span_lookup(node_id, &diagnostic.span);

    eshkol_diagnostic_sink_v1 sink = nullptr;
    void* userdata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_sink_mutex);
        sink = g_sink;
        userdata = g_sink_userdata;
    }
    g_count.fetch_add(1, std::memory_order_relaxed);
    if (sink) sink(&diagnostic, userdata);
}

extern "C" uint64_t eshkol_diagnostic_count_v1(void) {
    return g_count.load(std::memory_order_relaxed);
}

extern "C" void eshkol_diagnostic_reset_v1(void) {
    g_count.store(0, std::memory_order_relaxed);
}
