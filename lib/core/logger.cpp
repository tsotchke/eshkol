/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Logger Implementation
 *
 * Supports TEXT (human-readable) and JSON (structured) output formats.
 */
#include <eshkol/logger.h>

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <execinfo.h>
#include <cxxabi.h>
#endif

#ifdef __linux__
#include <execinfo.h>
#include <cxxabi.h>
#endif

#include <mutex>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>

// ============================================================================
// ANSI Color Codes
// ============================================================================

#define COLOR_PREFIX "\033["
#define BOLD COLOR_PREFIX "1m"
#define RED_COLOR BOLD COLOR_PREFIX "31m"
#define GREEN_COLOR BOLD COLOR_PREFIX "32m"
#define YELLOW_COLOR BOLD COLOR_PREFIX "33m"
#define BLUE_COLOR BOLD COLOR_PREFIX "34m"
#define MAGENTA_COLOR BOLD COLOR_PREFIX "35m"
#define CYAN_COLOR BOLD COLOR_PREFIX "36m"
#define WHITE_COLOR BOLD COLOR_PREFIX "37m"
#define RESET_COLOR COLOR_PREFIX "0m"

// ============================================================================
// Global State
// ============================================================================

static std::mutex g_log_mutex;
static eshkol_logger_t g_max_level = ESHKOL_NOTICE;
static eshkol_log_format_t g_log_format = ESHKOL_LOG_TEXT;
static FILE* g_log_file = nullptr;
static bool g_color_enabled = true;
static bool g_timestamps_enabled = true;
static bool g_owns_log_file = false;

static const char* g_log_names[] = {
    "FATAL",
    "ERROR",
    "WARNING",
    "NOTICE",
    "INFO",
    "DEBUG"
};

static const char* g_log_colors[] = {
    RED_COLOR,
    RED_COLOR,
    YELLOW_COLOR,
    BLUE_COLOR,
    WHITE_COLOR,
    GREEN_COLOR
};

// ============================================================================
// Helper Functions
// ============================================================================

static FILE* get_output() {
    return g_log_file ? g_log_file : stderr;
}

static std::string get_iso_timestamp() {
    time_t now = time(nullptr);
    struct tm* gmt = gmtime(&now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmt);
    return std::string(buf);
}

static std::string escape_json(const char* str) {
    if (!str) return "";

    std::string result;
    result.reserve(strlen(str) * 2);

    for (const char* p = str; *p; ++p) {
        switch (*p) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(*p) < 0x20) {
                    char hex[8];
                    snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned char>(*p));
                    result += hex;
                } else {
                    result += *p;
                }
                break;
        }
    }
    return result;
}

static void output_text(eshkol_logger_t level, const char* msg) {
    FILE* out = get_output();
    const char* prefix = g_log_names[level];
    const char* color = g_color_enabled ? g_log_colors[level] : "";
    const char* reset = g_color_enabled ? RESET_COLOR : "";

    if (g_timestamps_enabled) {
        time_t t = time(nullptr);
        char time_buf[64];
        strftime(time_buf, sizeof(time_buf), "%x (%X)", localtime(&t));
        fprintf(out, "%s%10s:%s ", color, prefix, reset);
    } else {
        fprintf(out, "%s%10s:%s ", color, prefix, reset);
    }

    fputs(msg, out);

    // Add newline if not present
    size_t len = strlen(msg);
    if (len == 0 || msg[len - 1] != '\n') {
        fputc('\n', out);
    }
}

static void output_json(eshkol_logger_t level, const char* msg,
                        const char* file, int line, const char* func) {
    FILE* out = get_output();

    std::string json = "{";
    json += "\"timestamp\":\"" + get_iso_timestamp() + "\"";
    json += ",\"level\":\"" + std::string(g_log_names[level]) + "\"";
    json += ",\"msg\":\"" + escape_json(msg) + "\"";

    if (file && file[0]) {
        json += ",\"file\":\"" + escape_json(file) + "\"";
    }
    if (line > 0) {
        json += ",\"line\":" + std::to_string(line);
    }
    if (func && func[0]) {
        json += ",\"func\":\"" + escape_json(func) + "\"";
    }

    json += "}\n";

    fputs(json.c_str(), out);
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

void eshkol_set_logger_level(eshkol_logger_t level) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_max_level = level;
}

eshkol_logger_t eshkol_get_logger_level(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_max_level;
}

void eshkol_set_log_format(eshkol_log_format_t format) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_format = format;
}

eshkol_log_format_t eshkol_get_log_format(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_log_format;
}

int eshkol_set_log_file(const char* path) {
    std::lock_guard<std::mutex> lock(g_log_mutex);

    // Close existing file if we own it
    if (g_log_file && g_owns_log_file) {
        fclose(g_log_file);
        g_log_file = nullptr;
        g_owns_log_file = false;
    }

    if (!path) {
        g_log_file = nullptr;
        return 0;
    }

    FILE* f = fopen(path, "a");
    if (!f) {
        return -1;
    }

    g_log_file = f;
    g_owns_log_file = true;
    return 0;
}

void eshkol_set_color_output(bool enabled) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_color_enabled = enabled;
}

void eshkol_set_timestamps(bool enabled) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_timestamps_enabled = enabled;
}

void eshkol_printf(eshkol_logger_t level, const char* msg, ...) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (level > g_max_level) return;
    }

    va_list ap;
    va_start(ap, msg);

    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), msg, ap);
    va_end(ap);

    {
        std::lock_guard<std::mutex> lock(g_log_mutex);

        if (g_log_format == ESHKOL_LOG_JSON) {
            output_json(level, buffer, nullptr, 0, nullptr);
        } else {
            output_text(level, buffer);
        }
    }

    if (level == ESHKOL_FATAL) {
        exit(1);
    }
}

void eshkol_log_with_location(eshkol_logger_t level,
                               const char* file,
                               int line,
                               const char* func,
                               const char* msg, ...) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (level > g_max_level) return;
    }

    va_list ap;
    va_start(ap, msg);

    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), msg, ap);
    va_end(ap);

    {
        std::lock_guard<std::mutex> lock(g_log_mutex);

        if (g_log_format == ESHKOL_LOG_JSON) {
            output_json(level, buffer, file, line, func);
        } else {
            // In text mode, append location info
            FILE* out = get_output();
            const char* prefix = g_log_names[level];
            const char* color = g_color_enabled ? g_log_colors[level] : "";
            const char* reset = g_color_enabled ? RESET_COLOR : "";

            fprintf(out, "%s%10s:%s ", color, prefix, reset);
            fputs(buffer, out);

            // Add location in debug mode
            if (file && line > 0) {
                fprintf(out, " [%s:%d]", file, line);
            }

            size_t len = strlen(buffer);
            if (len == 0 || buffer[len - 1] != '\n') {
                fputc('\n', out);
            }
        }
    }

    if (level == ESHKOL_FATAL) {
        exit(1);
    }
}

void eshkol_log_structured(eshkol_logger_t level, const char* msg, ...) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (level > g_max_level) return;
    }

    va_list ap;
    va_start(ap, msg);

    std::string json = "{";
    json += "\"timestamp\":\"" + get_iso_timestamp() + "\"";
    json += ",\"level\":\"" + std::string(g_log_names[level]) + "\"";
    json += ",\"msg\":\"" + escape_json(msg) + "\"";

    // Process key-value pairs until NULL
    while (true) {
        const char* key = va_arg(ap, const char*);
        if (!key) break;

        const char* value = va_arg(ap, const char*);
        if (!value) break;

        json += ",\"" + escape_json(key) + "\":\"" + escape_json(value) + "\"";
    }

    va_end(ap);

    json += "}\n";

    {
        std::lock_guard<std::mutex> lock(g_log_mutex);

        if (g_log_format == ESHKOL_LOG_JSON) {
            fputs(json.c_str(), get_output());
        } else {
            // In text mode, output as plain message with key-value pairs
            output_text(level, msg);
        }
    }

    if (level == ESHKOL_FATAL) {
        exit(1);
    }
}

const char* eshkol_log_level_name(eshkol_logger_t level) {
    if (level >= 0 && level <= ESHKOL_DEBUG) {
        return g_log_names[level];
    }
    return "UNKNOWN";
}

void eshkol_log_flush(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    FILE* out = get_output();
    fflush(out);
}

void eshkol_log_close(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_file && g_owns_log_file) {
        fclose(g_log_file);
        g_log_file = nullptr;
        g_owns_log_file = false;
    }
}

void eshkol_stacktrace(eshkol_logger_t level) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (level > g_max_level) return;
    }

#if defined(__APPLE__) || defined(__linux__)
    const uint64_t max_frames = 63;
    void* addrlist[max_frames + 1];
    int addrlen;
    char** symbollist;
    size_t func_name_size = 256;
    char* funcname = nullptr;

    std::lock_guard<std::mutex> lock(g_log_mutex);

    FILE* out = get_output();
    const char* prefix = g_log_names[level];
    const char* color = g_color_enabled ? g_log_colors[level] : "";
    const char* reset = g_color_enabled ? RESET_COLOR : "";

    addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));
    symbollist = backtrace_symbols(addrlist, addrlen);
    funcname = (char*)malloc(func_name_size);

    for (int i = 1; i < addrlen; ++i) {
        char* begin_name = nullptr;
        char* begin_offset = nullptr;
        char* end_offset = nullptr;

        for (char* p = symbollist[i]; *p; ++p) {
            if (*p == '(')
                begin_name = p;
            else if (*p == '+')
                begin_offset = p;
            else if (*p == ')' && begin_offset != nullptr) {
                end_offset = p;
                break;
            }
        }

        if (begin_name != nullptr &&
            begin_offset != nullptr &&
            end_offset != nullptr &&
            begin_name < begin_offset) {
            int status;
            char* ret = nullptr;

            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset++ = '\0';

            ret = abi::__cxa_demangle(begin_name, funcname, &func_name_size, &status);

            if (status == 0) {
                funcname = ret;
                fprintf(out, "%s%10s:%s  %s (%s+%s)\n",
                        color, prefix, reset,
                        symbollist[i], funcname, begin_offset);
            } else {
                fprintf(out, "%s%10s:%s  %s (%s+%s)\n",
                        color, prefix, reset,
                        symbollist[i], begin_name, begin_offset);
            }
        } else {
            fprintf(out, "%s%10s:%s  %s\n", color, prefix, reset, symbollist[i]);
        }
    }

    free(funcname);
    free(symbollist);
#else
    // Stacktrace not available on this platform
    eshkol_printf(level, "Stacktrace not available on this platform");
#endif

    if (level == ESHKOL_FATAL) {
        exit(1);
    }
}

} // extern "C"
