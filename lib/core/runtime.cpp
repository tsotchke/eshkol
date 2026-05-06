/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Core runtime helper implementation.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/eshkol.h>

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void eshkol_runtime_default_diagnostic_sink(
    eshkol_runtime_diagnostic_level_t level,
    const char* message);
extern "C" [[noreturn]] void eshkol_runtime_default_fatal_sink(
    const char* message);
extern "C" bool eshkol_runtime_default_monotonic_time_source(
    uint64_t* out_time_ns);
extern "C" bool eshkol_runtime_default_delay(uint64_t duration_ns);
extern "C" bool eshkol_runtime_profile_has_default_monotonic_clock(void);
extern "C" bool eshkol_runtime_profile_has_default_delay(void);
extern "C" bool eshkol_runtime_profile_has_operation_drain_hook_installed(void);

namespace {

constexpr size_t kEshkolRuntimeDiagnosticBufferSize = 1024;
constexpr size_t kEshkolRuntimeCapabilityCount = 5;

constexpr std::array<const char*, kEshkolRuntimeCapabilityCount>
    kEshkolRuntimeCapabilityNames = {{
        "diagnostic-sink",
        "fatal-sink",
        "monotonic-clock",
        "delay",
        "operation-drain",
    }};

std::atomic<eshkol_runtime_diagnostic_hook_t> g_runtime_diagnostic_hook{nullptr};
std::atomic<void*> g_runtime_diagnostic_hook_context{nullptr};
std::atomic<eshkol_runtime_fatal_hook_t> g_runtime_fatal_hook{nullptr};
std::atomic<void*> g_runtime_fatal_hook_context{nullptr};
std::atomic<eshkol_runtime_monotonic_clock_hook_t> g_runtime_monotonic_clock_hook{
    nullptr};
std::atomic<void*> g_runtime_monotonic_clock_hook_context{nullptr};
std::atomic<eshkol_runtime_delay_hook_t> g_runtime_delay_hook{nullptr};
std::atomic<void*> g_runtime_delay_hook_context{nullptr};

void format_runtime_message(char* buffer, size_t buffer_size,
                            const char* fallback_message,
                            const char* format, va_list args) {
    if (!buffer || buffer_size == 0) {
        return;
    }

    const int written = std::vsnprintf(
        buffer, buffer_size, format ? format : fallback_message, args);
    if (written < 0) {
        std::snprintf(buffer, buffer_size, "%s", fallback_message);
        return;
    }

    if (static_cast<size_t>(written) >= buffer_size) {
        buffer[buffer_size - 1] = '\0';
    }
}

void dispatch_runtime_diagnostic(eshkol_runtime_diagnostic_level_t level,
                                 const char* message) {
    const auto hook = g_runtime_diagnostic_hook.load(std::memory_order_acquire);
    if (hook) {
        void* context =
            g_runtime_diagnostic_hook_context.load(std::memory_order_relaxed);
        hook(level, message, context);
        return;
    }

    eshkol_runtime_default_diagnostic_sink(level, message);
}

bool describe_runtime_capability(
    eshkol_runtime_capability_kind_t kind,
    eshkol_runtime_capability_descriptor_t* out_descriptor) {
    if (!out_descriptor) {
        return false;
    }

    const auto index = static_cast<size_t>(kind);
    if (index >= kEshkolRuntimeCapabilityCount) {
        return false;
    }

    uint32_t flags = ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLABLE;
    switch (kind) {
        case ESHKOL_RUNTIME_CAPABILITY_DIAGNOSTIC_SINK:
            flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE;
            if (g_runtime_diagnostic_hook.load(std::memory_order_acquire)) {
                flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED;
            }
            break;
        case ESHKOL_RUNTIME_CAPABILITY_FATAL_SINK:
            flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE;
            if (g_runtime_fatal_hook.load(std::memory_order_acquire)) {
                flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED;
            }
            break;
        case ESHKOL_RUNTIME_CAPABILITY_MONOTONIC_CLOCK:
            if (eshkol_runtime_profile_has_default_monotonic_clock()) {
                flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE;
            }
            if (g_runtime_monotonic_clock_hook.load(std::memory_order_acquire)) {
                flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED;
            }
            break;
        case ESHKOL_RUNTIME_CAPABILITY_DELAY:
            if (eshkol_runtime_profile_has_default_delay()) {
                flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE;
            }
            if (g_runtime_delay_hook.load(std::memory_order_acquire)) {
                flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED;
            }
            break;
        case ESHKOL_RUNTIME_CAPABILITY_OPERATION_DRAIN:
            if (eshkol_runtime_profile_has_operation_drain_hook_installed()) {
                flags |= ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED;
            }
            break;
    }

    out_descriptor->kind = kind;
    out_descriptor->name = kEshkolRuntimeCapabilityNames[index];
    out_descriptor->flags = flags;
    return true;
}

}  // namespace

extern "C" {

void eshkol_runtime_set_diagnostic_hook(eshkol_runtime_diagnostic_hook_t hook,
                                        void* context) {
    g_runtime_diagnostic_hook_context.store(context, std::memory_order_relaxed);
    g_runtime_diagnostic_hook.store(hook, std::memory_order_release);
}

void eshkol_runtime_clear_diagnostic_hook(void) {
    g_runtime_diagnostic_hook.store(nullptr, std::memory_order_release);
    g_runtime_diagnostic_hook_context.store(nullptr, std::memory_order_relaxed);
}

void eshkol_runtime_debugf(const char* format, ...) {
    char buffer[kEshkolRuntimeDiagnosticBufferSize];

    va_list args;
    va_start(args, format);
    format_runtime_message(buffer, sizeof(buffer), "Runtime diagnostic", format,
                           args);
    va_end(args);

    dispatch_runtime_diagnostic(ESHKOL_RUNTIME_DIAGNOSTIC_DEBUG, buffer);
}

void eshkol_runtime_infof(const char* format, ...) {
    char buffer[kEshkolRuntimeDiagnosticBufferSize];

    va_list args;
    va_start(args, format);
    format_runtime_message(buffer, sizeof(buffer), "Runtime diagnostic", format,
                           args);
    va_end(args);

    dispatch_runtime_diagnostic(ESHKOL_RUNTIME_DIAGNOSTIC_INFO, buffer);
}

void eshkol_runtime_warnf(const char* format, ...) {
    char buffer[kEshkolRuntimeDiagnosticBufferSize];

    va_list args;
    va_start(args, format);
    format_runtime_message(buffer, sizeof(buffer), "Runtime diagnostic", format,
                           args);
    va_end(args);

    dispatch_runtime_diagnostic(ESHKOL_RUNTIME_DIAGNOSTIC_WARNING, buffer);
}

void eshkol_runtime_errorf(const char* format, ...) {
    char buffer[kEshkolRuntimeDiagnosticBufferSize];

    va_list args;
    va_start(args, format);
    format_runtime_message(buffer, sizeof(buffer), "Runtime diagnostic", format,
                           args);
    va_end(args);

    dispatch_runtime_diagnostic(ESHKOL_RUNTIME_DIAGNOSTIC_ERROR, buffer);
}

void eshkol_runtime_set_fatal_hook(eshkol_runtime_fatal_hook_t hook,
                                   void* context) {
    g_runtime_fatal_hook_context.store(context, std::memory_order_relaxed);
    g_runtime_fatal_hook.store(hook, std::memory_order_release);
}

void eshkol_runtime_clear_fatal_hook(void) {
    g_runtime_fatal_hook.store(nullptr, std::memory_order_release);
    g_runtime_fatal_hook_context.store(nullptr, std::memory_order_relaxed);
}

void eshkol_runtime_set_monotonic_clock_hook(
    eshkol_runtime_monotonic_clock_hook_t hook,
    void* context) {
    g_runtime_monotonic_clock_hook_context.store(context,
                                                 std::memory_order_relaxed);
    g_runtime_monotonic_clock_hook.store(hook, std::memory_order_release);
}

void eshkol_runtime_clear_monotonic_clock_hook(void) {
    g_runtime_monotonic_clock_hook.store(nullptr, std::memory_order_release);
    g_runtime_monotonic_clock_hook_context.store(nullptr,
                                                 std::memory_order_relaxed);
}

void eshkol_runtime_set_delay_hook(eshkol_runtime_delay_hook_t hook,
                                   void* context) {
    g_runtime_delay_hook_context.store(context, std::memory_order_relaxed);
    g_runtime_delay_hook.store(hook, std::memory_order_release);
}

void eshkol_runtime_clear_delay_hook(void) {
    g_runtime_delay_hook.store(nullptr, std::memory_order_release);
    g_runtime_delay_hook_context.store(nullptr, std::memory_order_relaxed);
}

bool eshkol_runtime_get_monotonic_time_ns(uint64_t* out_time_ns) {
    if (!out_time_ns) {
        return false;
    }

    const auto hook =
        g_runtime_monotonic_clock_hook.load(std::memory_order_acquire);
    if (hook) {
        void* context = g_runtime_monotonic_clock_hook_context.load(
            std::memory_order_relaxed);
        return hook(out_time_ns, context);
    }

    return eshkol_runtime_default_monotonic_time_source(out_time_ns);
}

bool eshkol_runtime_delay_ns(uint64_t duration_ns) {
    if (duration_ns == 0) {
        return true;
    }

    const auto hook = g_runtime_delay_hook.load(std::memory_order_acquire);
    if (hook) {
        void* context =
            g_runtime_delay_hook_context.load(std::memory_order_relaxed);
        return hook(duration_ns, context);
    }

    return eshkol_runtime_default_delay(duration_ns);
}

size_t eshkol_runtime_get_capability_count(void) {
    return kEshkolRuntimeCapabilityCount;
}

bool eshkol_runtime_describe_capability_at(
    size_t index,
    eshkol_runtime_capability_descriptor_t* out_descriptor) {
    if (index >= kEshkolRuntimeCapabilityCount) {
        return false;
    }
    return describe_runtime_capability(
        static_cast<eshkol_runtime_capability_kind_t>(index), out_descriptor);
}

bool eshkol_runtime_describe_capability(
    eshkol_runtime_capability_kind_t kind,
    eshkol_runtime_capability_descriptor_t* out_descriptor) {
    return describe_runtime_capability(kind, out_descriptor);
}

void eshkol_runtime_fatalf(const char* format, ...) {
    char buffer[kEshkolRuntimeDiagnosticBufferSize];

    va_list args;
    va_start(args, format);
    format_runtime_message(buffer, sizeof(buffer), "Fatal runtime error", format,
                           args);
    va_end(args);

    const auto hook = g_runtime_fatal_hook.load(std::memory_order_acquire);
    if (hook) {
        void* context =
            g_runtime_fatal_hook_context.load(std::memory_order_relaxed);
        hook(buffer, context);
    }

    eshkol_runtime_default_fatal_sink(buffer);
}

// ----------------------------------------------------------------------------
// Type Errors (R7RS Compliance)
// ----------------------------------------------------------------------------

void eshkol_type_error(const char* proc_name, const char* expected_type) {
    eshkol_runtime_fatalf("Type error in %s: expected %s",
                          proc_name ? proc_name : "<unknown>",
                          expected_type ? expected_type : "<type>");
}

void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                   const char* actual_type) {
    eshkol_runtime_fatalf("Type error in %s: expected %s, got %s",
                          proc_name ? proc_name : "<unknown>",
                          expected_type ? expected_type : "<type>",
                          actual_type ? actual_type : "<unknown>");
}

// ----------------------------------------------------------------------------
// Parameter Objects (R7RS make-parameter / parameterize)
// ----------------------------------------------------------------------------

// Forward declaration for arena allocation with GC-tracked header
extern void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                        uint8_t subtype, uint8_t flags);

#define HEAP_SUBTYPE_PARAMETER 20

// Parameter object: a stack of tagged values implementing dynamic binding.
// The bottom of the stack (index 0) holds the default value.
// Each parameterize pushes a new value; exiting parameterize pops it.
typedef struct {
    eshkol_tagged_value_t* stack;
    int top;
    int capacity;
} eshkol_param_t;

// Create a new parameter object with the given default value.
// The parameter struct is arena-allocated (GC-tracked); the internal
// value stack uses malloc since it may need to grow.
void* eshkol_make_parameter(void* arena, eshkol_tagged_value_t default_val) {
    eshkol_param_t* param = (eshkol_param_t*)arena_allocate_with_header(
        arena, sizeof(eshkol_param_t), HEAP_SUBTYPE_PARAMETER, 0);
    if (!param) {
        return nullptr;
    }

    const int initial_capacity = 8;
    param->stack = (eshkol_tagged_value_t*)std::malloc(
        initial_capacity * sizeof(eshkol_tagged_value_t));
    if (!param->stack) {
        // Allocation failure — return the param with no stack
        // (eshkol_parameter_ref will return null tagged value)
        param->top = -1;
        param->capacity = 0;
        return (void*)param;
    }

    param->capacity = initial_capacity;
    param->top = 0;
    param->stack[0] = default_val;
    return (void*)param;
}

// Push a new binding onto the parameter's value stack.
// Called at entry to a parameterize block.
void eshkol_parameter_push(void* param_ptr, eshkol_tagged_value_t val) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    // Grow stack if at capacity
    if (param->top + 1 >= param->capacity) {
        int new_capacity = param->capacity * 2;
        if (new_capacity < 8) new_capacity = 8;
        eshkol_tagged_value_t* new_stack = (eshkol_tagged_value_t*)std::realloc(
            param->stack, new_capacity * sizeof(eshkol_tagged_value_t));
        if (!new_stack) {
            // realloc failed — silently drop the push
            // (better than crashing; current binding remains)
            return;
        }
        param->stack = new_stack;
        param->capacity = new_capacity;
    }

    param->top++;
    param->stack[param->top] = val;
}

// Pop the most recent binding from the parameter's value stack.
// Called at exit from a parameterize block (including unwinding).
// Never pops below index 0 — the default value is always preserved.
void eshkol_parameter_pop(void* param_ptr) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    // Never pop below 1 — always keep the default value at index 0
    if (param->top > 0) {
        param->top--;
    }
}

// Return the current (topmost) binding of the parameter.
// If the parameter has no stack (allocation failure), returns ESHKOL_VALUE_NULL.
eshkol_tagged_value_t eshkol_parameter_ref(void* param_ptr) {
    if (!param_ptr) {
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    if (param->top < 0 || !param->stack) {
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }

    return param->stack[param->top];
}

// Pointer-based wrappers for LLVM codegen (avoids by-value struct ABI issues)
void* eshkol_make_parameter_ptr(void* arena, const eshkol_tagged_value_t* default_val) {
    if (!default_val) return nullptr;
    return eshkol_make_parameter(arena, *default_val);
}

void eshkol_parameter_push_ptr(void* param, const eshkol_tagged_value_t* val) {
    if (!val) return;
    eshkol_parameter_push(param, *val);
}

void eshkol_parameter_ref_ptr(void* param, eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = eshkol_parameter_ref(param);
}

// ----------------------------------------------------------------------------
// Bytevector Runtime (R7RS)
// ----------------------------------------------------------------------------
// Layout: ptr[-8] = header (subtype byte), ptr[0..7] = int64_t length, ptr[8..] = byte data
// HEAP_SUBTYPE_BYTEVECTOR = 8 (defined in eshkol/eshkol.h)

/* Forward declaration for arena allocation with header */
extern void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                        uint8_t subtype, uint8_t flags);

/* Create a new bytevector of given length, filled with fill_byte */
void* eshkol_make_bytevector(void* arena, int64_t len, int64_t fill_byte) {
    if (!arena || len < 0) {
        eshkol_runtime_fatalf("Error in make-bytevector: invalid arguments (len=%lld)",
                              (long long)len);
    }

    // Allocate: 8 bytes for length + len bytes for data
    uint64_t data_size = (uint64_t)(8 + len);
    void* ptr = arena_allocate_with_header(arena, data_size, 8 /* HEAP_SUBTYPE_BYTEVECTOR */, 0);
    if (!ptr) {
        eshkol_runtime_fatalf("Error in make-bytevector: out of memory (len=%lld)",
                              (long long)len);
    }

    // Store length at ptr[0..7]
    int64_t* length_ptr = (int64_t*)ptr;
    *length_ptr = len;

    // Fill byte data at ptr[8..]
    uint8_t* data = (uint8_t*)ptr + 8;
    uint8_t fill = (uint8_t)(fill_byte & 0xFF);
    memset(data, fill, (size_t)len);

    return ptr;
}

/* Get byte at index k (returns int64_t for tagged value compatibility) */
int64_t eshkol_bytevector_u8_ref(void* bv, int64_t k) {
    if (!bv) {
        eshkol_runtime_fatalf("Error in bytevector-u8-ref: null bytevector");
    }

    int64_t len = *((int64_t*)bv);
    if (k < 0 || k >= len) {
        eshkol_runtime_fatalf("Error in bytevector-u8-ref: index %lld out of range [0, %lld)",
                              (long long)k, (long long)len);
    }

    uint8_t* data = (uint8_t*)bv + 8;
    return (int64_t)data[k];
}

/* Set byte at index k */
void eshkol_bytevector_u8_set(void* bv, int64_t k, int64_t byte_val) {
    if (!bv) {
        eshkol_runtime_fatalf("Error in bytevector-u8-set!: null bytevector");
    }

    int64_t len = *((int64_t*)bv);
    if (k < 0 || k >= len) {
        eshkol_runtime_fatalf("Error in bytevector-u8-set!: index %lld out of range [0, %lld)",
                              (long long)k, (long long)len);
    }

    if (byte_val < 0 || byte_val > 255) {
        eshkol_runtime_fatalf("Error in bytevector-u8-set!: byte value %lld out of range [0, 255]",
                              (long long)byte_val);
    }

    uint8_t* data = (uint8_t*)bv + 8;
    data[k] = (uint8_t)byte_val;
}

/* Get length of bytevector */
int64_t eshkol_bytevector_length(void* bv) {
    if (!bv) {
        eshkol_runtime_fatalf("Error in bytevector-length: null bytevector");
    }

    return *((int64_t*)bv);
}

/* Copy bytevector (or sub-range) into a new arena-allocated bytevector */
void* eshkol_bytevector_copy(void* arena, void* bv, int64_t start, int64_t end) {
    if (!bv) {
        eshkol_runtime_fatalf("Error in bytevector-copy: null bytevector");
    }
    if (!arena) {
        eshkol_runtime_fatalf("Error in bytevector-copy: null arena");
    }

    int64_t len = *((int64_t*)bv);

    // Clamp end to length if -1 (sentinel for "copy to end")
    if (end < 0) end = len;

    if (start < 0 || start > len || end < start || end > len) {
        eshkol_runtime_fatalf("Error in bytevector-copy: range [%lld, %lld) out of bounds [0, %lld)",
                              (long long)start, (long long)end, (long long)len);
    }

    int64_t copy_len = end - start;

    // Allocate new bytevector: 8 bytes for length + copy_len bytes for data
    uint64_t data_size = (uint64_t)(8 + copy_len);
    void* new_bv = arena_allocate_with_header(arena, data_size, 8 /* HEAP_SUBTYPE_BYTEVECTOR */, 0);
    if (!new_bv) {
        eshkol_runtime_fatalf("Error in bytevector-copy: out of memory (len=%lld)",
                              (long long)copy_len);
    }

    // Store length
    *((int64_t*)new_bv) = copy_len;

    // Copy byte data
    if (copy_len > 0) {
        uint8_t* src_data = (uint8_t*)bv + 8 + start;
        uint8_t* dst_data = (uint8_t*)new_bv + 8;
        memcpy(dst_data, src_data, (size_t)copy_len);
    }

    return new_bv;
}

// ----------------------------------------------------------------------------
// UTF-8 Helpers
// ----------------------------------------------------------------------------

static int64_t eshkol_string_byte_length(const char* s) {
    if (!s) return 0;
    const eshkol_object_header_t* header =
        reinterpret_cast<const eshkol_object_header_t*>(
            reinterpret_cast<const uint8_t*>(s) - sizeof(eshkol_object_header_t));
    if (header->subtype == HEAP_SUBTYPE_STRING && header->size > 0) {
        return static_cast<int64_t>(header->size) - 1;
    }
    return static_cast<int64_t>(strlen(s));
}

int64_t eshkol_utf8_strlen(const char* s) {
    if (!s) return 0;
    const int64_t byte_len = eshkol_string_byte_length(s);
    int64_t count = 0;
    for (int64_t i = 0; i < byte_len; i++) {
        if ((s[i] & 0xC0) != 0x80) count++;
    }
    return count;
}

/* Decode a single UTF-8 codepoint at s, advance *s past it */
static int64_t decode_utf8_codepoint(const char** s) {
    const unsigned char* p = (const unsigned char*)*s;
    int64_t cp;
    if (*p < 0x80) { cp = *p; *s += 1; }
    else if ((*p & 0xE0) == 0xC0) { cp = (*p & 0x1F) << 6 | (p[1] & 0x3F); *s += 2; }
    else if ((*p & 0xF0) == 0xE0) { cp = (*p & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F); *s += 3; }
    else if ((*p & 0xF8) == 0xF0) { cp = (*p & 0x07) << 18 | (p[1] & 0x3F) << 12 | (p[2] & 0x3F) << 6 | (p[3] & 0x3F); *s += 4; }
    else { cp = 0xFFFD; *s += 1; } /* replacement char for invalid */
    return cp;
}

int64_t eshkol_utf8_ref(const char* s, int64_t k) {
    if (!s || k < 0) return -1;
    const int64_t byte_len = eshkol_string_byte_length(s);
    int64_t i = 0;
    int64_t cp_idx = 0;
    while (i < byte_len && cp_idx < k) {
        if ((s[i] & 0xC0) != 0x80) cp_idx++;
        i++;
    }
    if (i >= byte_len) return -1;
    const char* p = s + i;
    return decode_utf8_codepoint(&p);
}

char* eshkol_utf8_substring(const char* s, int64_t start, int64_t end, void* arena) {
    if (!s || !arena || start < 0 || end < start) return nullptr;
    extern void* arena_allocate_string_with_header(void*, uint64_t);

    const int64_t byte_len_total = eshkol_string_byte_length(s);
    auto advance_one_codepoint = [&](int64_t& i) {
        if (i >= byte_len_total) return;
        const unsigned char byte = static_cast<unsigned char>(s[i]);
        if ((byte & 0x80) == 0) {
            i += 1;
        } else if ((byte & 0xE0) == 0xC0) {
            i += 2;
        } else if ((byte & 0xF0) == 0xE0) {
            i += 3;
        } else if ((byte & 0xF8) == 0xF0) {
            i += 4;
        } else {
            i += 1;
        }
        if (i > byte_len_total) i = byte_len_total;
    };

    int64_t i = 0;
    int64_t cp_idx = 0;
    while (i < byte_len_total && cp_idx < start) {
        advance_one_codepoint(i);
        cp_idx++;
    }
    const int64_t start_off = i;
    while (i < byte_len_total && cp_idx < end) {
        advance_one_codepoint(i);
        cp_idx++;
    }

    const int64_t byte_len = i - start_off;
    char* buf = static_cast<char*>(
        arena_allocate_string_with_header(arena, static_cast<uint64_t>(byte_len)));
    if (buf) {
        memcpy(buf, s + start_off, static_cast<size_t>(byte_len));
        buf[byte_len] = '\0';
    }
    return buf;
}

extern eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(
    const struct arena_tagged_cons_cell* cell, bool is_cdr);

int64_t eshkol_unwrap_list_index(const eshkol_tagged_value_t* tv_in) {
    if (!tv_in) {
        return 0;
    }

    eshkol_tagged_value_t value = *tv_in;
    uint8_t base_type = value.type;
    if (base_type < 8) {
        base_type &= 0x0F;
    }

    if (base_type == ESHKOL_VALUE_HEAP_PTR && value.data.ptr_val != 0) {
        const auto* header =
            ESHKOL_GET_HEADER(reinterpret_cast<void*>(value.data.ptr_val));
        if (header && header->subtype == HEAP_SUBTYPE_CONS) {
            const auto* cell = reinterpret_cast<const struct arena_tagged_cons_cell*>(
                value.data.ptr_val);
            eshkol_tagged_value_t car =
                arena_tagged_cons_get_tagged_value(cell, false);
            uint8_t car_type = car.type;
            if (car_type < 8) {
                car_type &= 0x0F;
            }
            if (car_type == ESHKOL_VALUE_INT64) {
                return static_cast<int64_t>(car.data.int_val);
            }
            if (car_type == ESHKOL_VALUE_DOUBLE) {
                double d;
                std::memcpy(&d, &car.data, sizeof(double));
                return static_cast<int64_t>(d);
            }
            return static_cast<int64_t>(car.data.int_val);
        }
    }

    if (base_type == ESHKOL_VALUE_INT64) {
        return static_cast<int64_t>(value.data.int_val);
    }
    if (base_type == ESHKOL_VALUE_DOUBLE) {
        double d;
        std::memcpy(&d, &value.data, sizeof(double));
        return static_cast<int64_t>(d);
    }
    return static_cast<int64_t>(value.data.int_val);
}

} // extern "C"
