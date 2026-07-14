/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Opt-in, execution-time language-surface evidence.  Normal builds pay only
 * the codegen-time environment check: generated programs contain no coverage
 * calls unless ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR is set while compiling.
 */
#include <eshkol/core/runtime.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

int currentProcessId() {
#ifdef _WIN32
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

struct ExecutionSite {
    const char* source;
    const char* name;
    uint32_t line;
    uint32_t column;
    uint32_t operation;
    char kind;

    bool operator==(const ExecutionSite& other) const {
        return source == other.source && name == other.name &&
               line == other.line && column == other.column &&
               operation == other.operation && kind == other.kind;
    }
};

struct ExecutionSiteHash {
    size_t operator()(const ExecutionSite& site) const {
        size_t hash = reinterpret_cast<size_t>(site.source);
        hash ^= reinterpret_cast<size_t>(site.name) + 0x9e3779b97f4a7c15ULL +
                (hash << 6) + (hash >> 2);
        hash ^= static_cast<size_t>(site.line) * 0x85ebca6bU;
        hash ^= static_cast<size_t>(site.column) * 0xc2b2ae35U;
        hash ^= static_cast<size_t>(site.operation) << 1;
        hash ^= static_cast<unsigned char>(site.kind);
        return hash;
    }
};

bool firstExecutionAtSite(const ExecutionSite& site) {
    static thread_local std::unordered_set<ExecutionSite, ExecutionSiteHash> seen;
    return seen.insert(site).second;
}

struct CoverageTrace {
    std::mutex mutex;
    std::unordered_set<std::string> emitted;
    std::ofstream stream;
    bool enabled = false;
    size_t pending_records = 0;
    int process_id = currentProcessId();

    CoverageTrace() {
        const char* raw_dir = std::getenv("ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR");
        if (!raw_dir || !*raw_dir) {
            return;
        }

        std::error_code ec;
        const std::filesystem::path dir(raw_dir);
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            return;
        }

        const std::filesystem::path path =
            dir / ("language-coverage-" + std::to_string(process_id) + ".tsv");
        stream.open(path, std::ios::out | std::ios::app);
        enabled = stream.good();
    }

    static std::string clean(const char* value) {
        std::string result = value ? value : "<unknown>";
        for (char& ch : result) {
            if (ch == '\t' || ch == '\n' || ch == '\r') {
                ch = ' ';
            }
        }
        return result;
    }

    static bool knownSource(const char* value) {
        return value && *value && std::strcmp(value, "<unknown>") != 0 &&
               std::strcmp(value, "unknown") != 0;
    }

    void write(const std::string& record) {
        if (!enabled) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (!emitted.insert(record).second) {
            return;
        }
        stream << record << '\n';
        if (++pending_records >= 256) {
            stream.flush();
            pending_records = 0;
        }
    }

    void flush() {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        stream.flush();
        pending_records = 0;
    }
};

CoverageTrace& trace() {
    // A fork inherits C++ stream buffers and mutex state. Sharing the parent's
    // buffered stream lets parent/child records interleave and can deadlock if
    // another thread held the mutex at fork time. Detect the changed PID
    // before touching the inherited object and give the child its own stream.
    // The holder deletes the active process instance at normal exit so its
    // final partial batch is flushed; the inherited parent instance is
    // intentionally left untouched in the child.
    struct ProcessTraceHolder {
        CoverageTrace* active = new CoverageTrace();

        ~ProcessTraceHolder() { delete active; }

        CoverageTrace& get() {
            if (active->process_id != currentProcessId()) {
                active = new CoverageTrace();
            }
            return *active;
        }
    };
    static ProcessTraceHolder holder;
    return holder.get();
}

}  // namespace

extern "C" void eshkol_language_coverage_parse(const char* source,
                                                uint32_t line,
                                                uint32_t column,
                                                uint32_t operation,
                                                const char* name) {
    CoverageTrace& sink = trace();
    if (!sink.enabled || !CoverageTrace::knownSource(source)) return;
    std::ostringstream record;
    record << "P\t" << CoverageTrace::clean(source) << '\t' << line << '\t'
           << column << '\t' << operation << '\t'
           << CoverageTrace::clean(name);
    sink.write(record.str());
}

extern "C" void eshkol_language_coverage_codegen(const char* source,
                                                   uint32_t line,
                                                   uint32_t column,
                                                   uint32_t operation) {
    CoverageTrace& sink = trace();
    if (!sink.enabled || !CoverageTrace::knownSource(source)) return;
    std::ostringstream record;
    record << "G\t" << CoverageTrace::clean(source) << '\t' << line << '\t'
           << column << '\t' << operation;
    sink.write(record.str());
}

extern "C" void eshkol_language_coverage_accept(const char* source,
                                                  uint32_t line,
                                                  uint32_t column,
                                                  uint32_t operation) {
    CoverageTrace& sink = trace();
    if (!sink.enabled || !CoverageTrace::knownSource(source)) return;
    std::ostringstream record;
    record << "A\t" << CoverageTrace::clean(source) << '\t' << line << '\t'
           << column << '\t' << operation;
    sink.write(record.str());
}

extern "C" void eshkol_language_coverage_reject(const char* source,
                                                   uint32_t line,
                                                   uint32_t column,
                                                   uint32_t operation,
                                                   const char* name) {
    CoverageTrace& sink = trace();
    if (!sink.enabled || !CoverageTrace::knownSource(source) || !name || !*name) {
        return;
    }
    std::ostringstream record;
    record << "R\t" << CoverageTrace::clean(source) << '\t' << line << '\t'
           << column << '\t' << operation << '\t' << CoverageTrace::clean(name);
    sink.write(record.str());
    // Negative compilation may terminate before the normal RAII flush path.
    sink.flush();
}

extern "C" void eshkol_language_coverage_exec_op(const char* source,
                                                  uint32_t line,
                                                  uint32_t column,
                                                  uint32_t operation) {
    CoverageTrace& sink = trace();
    if (!sink.enabled || !CoverageTrace::knownSource(source)) return;
    if (!firstExecutionAtSite({source, nullptr, line, column, operation, 'O'})) {
        return;
    }
    std::ostringstream record;
    record << "O\t" << CoverageTrace::clean(source) << '\t' << line << '\t'
           << column << '\t' << operation;
    sink.write(record.str());
}

extern "C" void eshkol_language_coverage_exec_call(const char* source,
                                                    uint32_t line,
                                                    uint32_t column,
                                                    const char* name) {
    CoverageTrace& sink = trace();
    if (!sink.enabled) return;
    if (!firstExecutionAtSite({source, name, line, column, 0, 'C'})) return;
    std::ostringstream record;
    record << "C\t" << CoverageTrace::clean(source) << '\t' << line << '\t'
           << column << '\t' << CoverageTrace::clean(name);
    sink.write(record.str());
    // emergency-exit uses _exit and therefore bypasses C++ destructors.
    if (std::strcmp(name, "emergency-exit") == 0) sink.flush();
}

extern "C" void eshkol_language_coverage_vm_dispatch(const char* name,
                                                        uint32_t native_id) {
    CoverageTrace& sink = trace();
    if (!sink.enabled || !name || !*name) return;
    if (!firstExecutionAtSite({"<vm>", name, 0, 0, native_id, 'V'})) return;
    std::ostringstream record;
    record << "V\t<vm>\t0\t0\t" << native_id << '\t'
           << CoverageTrace::clean(name);
    sink.write(record.str());
}

extern "C" void eshkol_language_coverage_vm_call_hash(uint32_t name_hash) {
    CoverageTrace& sink = trace();
    if (!sink.enabled) return;
    const char* marker = "@call";
    if (!firstExecutionAtSite({"<vm>", marker, 0, 0, name_hash, 'V'})) return;
    std::ostringstream record;
    record << "V\t<vm>\t0\t0\t" << name_hash << "\t@call";
    sink.write(record.str());
}

extern "C" void eshkol_language_coverage_flush(void) {
    trace().flush();
}
