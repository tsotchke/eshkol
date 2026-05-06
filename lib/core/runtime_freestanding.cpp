/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Minimal freestanding runtime lifecycle and fallback sink stubs.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/eshkol.h>

#include "arena_memory.h"

#include <atomic>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

volatile sig_atomic_t g_eshkol_interrupt_flag = 0;

eshkol_exception_t* g_current_exception = nullptr;
eshkol_exception_handler_t* g_exception_handler_stack = nullptr;

namespace {

constexpr size_t kMaxFreestandingShutdownHooks = 32;
constexpr size_t kMaxFreestandingOperations = 64;
constexpr size_t kMaxFreestandingExceptionHandlers = 16;
constexpr int64_t kMaxFreestandingRecursionDepth = 100000;

struct ShutdownHookSlot {
    uint32_t id = 0;
    eshkol_shutdown_hook_t callback = nullptr;
    void* context = nullptr;
    const char* name = nullptr;
};

struct OperationSlot {
    uint32_t id = 0;
    bool active = false;
    const char* name = nullptr;
};

std::atomic<eshkol_runtime_state_t> g_runtime_state{
    ESHKOL_RUNTIME_INITIALIZING};
std::atomic<eshkol_shutdown_reason_t> g_shutdown_reason{
    ESHKOL_SHUTDOWN_NONE};
std::atomic<eshkol_runtime_operation_drain_hook_t> g_operation_drain_hook{
    nullptr};
std::atomic<void*> g_operation_drain_hook_context{nullptr};
std::atomic<uint32_t> g_next_hook_id{1};
std::atomic<uint32_t> g_next_operation_id{1};
std::atomic<uint32_t> g_operation_count{0};

ShutdownHookSlot g_shutdown_hooks[kMaxFreestandingShutdownHooks];
OperationSlot g_operations[kMaxFreestandingOperations];
eshkol_exception_t g_last_exception{};
eshkol_exception_handler_t
    g_exception_handlers[kMaxFreestandingExceptionHandlers];
eshkol_tagged_value_t g_raised_tagged_value = {0, 0, 0, {0}};
bool g_raised_value_set_by_user = false;
size_t g_exception_handler_count = 0;
int64_t g_recursion_depth = 0;

bool already_dispatched(const uint32_t* dispatched_ids, size_t dispatched_count,
                        uint32_t id) {
    for (size_t i = 0; i < dispatched_count; ++i) {
        if (dispatched_ids[i] == id) {
            return true;
        }
    }
    return false;
}

void clear_shutdown_slot(ShutdownHookSlot& slot) {
    slot.id = 0;
    slot.callback = nullptr;
    slot.context = nullptr;
    slot.name = nullptr;
}

void clear_operation_slot(OperationSlot& slot) {
    slot.id = 0;
    slot.active = false;
    slot.name = nullptr;
}

char* duplicate_exception_string(const char* value) {
    if (!value) {
        return nullptr;
    }

    if (__global_arena) {
        const size_t len = std::strlen(value) + 1;
        char* copy = static_cast<char*>(arena_allocate(__global_arena, len));
        if (!copy) {
            return nullptr;
        }
        std::memcpy(copy, value, len);
        return copy;
    }

    return const_cast<char*>(value);
}

eshkol_exception_t* initialize_exception_object(eshkol_exception_t* exc,
                                                eshkol_exception_type_t type,
                                                const char* message) {
    if (!exc) {
        return nullptr;
    }

    exc->type = type;
    exc->message = duplicate_exception_string(
        message ? message : "freestanding runtime exception");
    exc->irritants = nullptr;
    exc->num_irritants = 0;
    exc->line = 0;
    exc->column = 0;
    exc->filename = nullptr;
    return exc;
}

}  // namespace

namespace eshkol::runtime {

void destroy_arena_threading_state(arena_t* arena) { (void)arena; }

}  // namespace eshkol::runtime

extern "C" {

bool eshkol_runtime_default_monotonic_time_source(uint64_t* out_time_ns) {
    (void)out_time_ns;
    return false;
}

bool eshkol_runtime_default_delay(uint64_t duration_ns) {
    (void)duration_ns;
    return false;
}

bool eshkol_runtime_profile_has_default_monotonic_clock(void) {
    return false;
}

bool eshkol_runtime_profile_has_default_delay(void) { return false; }

void eshkol_init_stack_size(void) {}

void eshkol_runtime_default_diagnostic_sink(
    eshkol_runtime_diagnostic_level_t level,
    const char* message) {
    (void)level;
    (void)message;
}

[[noreturn]] void eshkol_runtime_default_fatal_sink(const char* message) {
    (void)message;
#if defined(__clang__) || defined(__GNUC__)
    __builtin_trap();
    __builtin_unreachable();
#else
    for (;;) {
    }
#endif
}

void eshkol_runtime_init_signals(void) {}

void eshkol_runtime_restore_signals(void) {}

void eshkol_runtime_request_interrupt(eshkol_shutdown_reason_t reason) {
    g_eshkol_interrupt_flag = 1;
    g_shutdown_reason.store(reason, std::memory_order_release);
    g_runtime_state.store(ESHKOL_RUNTIME_SHUTTING_DOWN,
                          std::memory_order_release);
}

void eshkol_runtime_clear_interrupt(void) {
    g_eshkol_interrupt_flag = 0;
    g_shutdown_reason.store(ESHKOL_SHUTDOWN_NONE, std::memory_order_release);
}

void eshkol_runtime_set_operation_drain_hook(
    eshkol_runtime_operation_drain_hook_t hook,
    void* context) {
    g_operation_drain_hook_context.store(context, std::memory_order_relaxed);
    g_operation_drain_hook.store(hook, std::memory_order_release);
}

void eshkol_runtime_clear_operation_drain_hook(void) {
    g_operation_drain_hook.store(nullptr, std::memory_order_release);
    g_operation_drain_hook_context.store(nullptr, std::memory_order_relaxed);
}

bool eshkol_runtime_profile_has_operation_drain_hook_installed(void) {
    return g_operation_drain_hook.load(std::memory_order_acquire) != nullptr;
}

eshkol_shutdown_reason_t eshkol_runtime_get_shutdown_reason(void) {
    return g_shutdown_reason.load(std::memory_order_acquire);
}

uint32_t eshkol_register_shutdown_hook(eshkol_shutdown_hook_t hook,
                                       void* context, const char* name) {
    if (!hook) {
        return 0;
    }

    for (ShutdownHookSlot& slot : g_shutdown_hooks) {
        if (slot.id == 0) {
            slot.id = g_next_hook_id.fetch_add(1, std::memory_order_relaxed);
            slot.callback = hook;
            slot.context = context;
            slot.name = name;
            return slot.id;
        }
    }

    eshkol_runtime_warnf(
        "Freestanding shutdown hook capacity exceeded (max=%zu)",
        kMaxFreestandingShutdownHooks);
    return 0;
}

bool eshkol_unregister_shutdown_hook(uint32_t hook_id) {
    if (hook_id == 0) {
        return false;
    }

    for (ShutdownHookSlot& slot : g_shutdown_hooks) {
        if (slot.id == hook_id) {
            clear_shutdown_slot(slot);
            return true;
        }
    }

    return false;
}

int eshkol_runtime_init(void) {
    eshkol_runtime_state_t expected = ESHKOL_RUNTIME_INITIALIZING;
    if (!g_runtime_state.compare_exchange_strong(expected,
                                                 ESHKOL_RUNTIME_RUNNING)) {
        return expected == ESHKOL_RUNTIME_RUNNING ? 0 : -1;
    }

    return 0;
}

void eshkol_runtime_shutdown(eshkol_shutdown_reason_t reason) {
    eshkol_runtime_state_t expected = ESHKOL_RUNTIME_RUNNING;
    if (!g_runtime_state.compare_exchange_strong(
            expected, ESHKOL_RUNTIME_SHUTTING_DOWN)) {
        if (expected == ESHKOL_RUNTIME_SHUTTING_DOWN ||
            expected == ESHKOL_RUNTIME_TERMINATED) {
            return;
        }
        g_runtime_state.store(ESHKOL_RUNTIME_SHUTTING_DOWN,
                              std::memory_order_release);
    }

    g_shutdown_reason.store(reason, std::memory_order_release);
    g_eshkol_interrupt_flag = 1;

    const uint32_t active_operations =
        g_operation_count.load(std::memory_order_acquire);
    if (active_operations > 0) {
        if (!eshkol_runtime_drain_operations(5000)) {
            eshkol_runtime_warnf(
                "Freestanding runtime shutdown is proceeding with %u undrained operations",
                g_operation_count.load(std::memory_order_acquire));
        }
    }

    uint32_t dispatched_ids[kMaxFreestandingShutdownHooks] = {};
    size_t dispatched_count = 0;
    while (true) {
        ShutdownHookSlot* next_slot = nullptr;
        uint32_t next_id = 0;

        for (ShutdownHookSlot& slot : g_shutdown_hooks) {
            if (slot.id == 0 || !slot.callback ||
                already_dispatched(dispatched_ids, dispatched_count, slot.id)) {
                continue;
            }
            if (!next_slot || slot.id > next_id) {
                next_slot = &slot;
                next_id = slot.id;
            }
        }

        if (!next_slot) {
            break;
        }

        dispatched_ids[dispatched_count++] = next_slot->id;
        const int hook_result = next_slot->callback(next_slot->context, reason);
        if (hook_result != 0) {
            eshkol_runtime_warnf(
                "Freestanding shutdown hook returned error: %d", hook_result);
        }
    }

    g_runtime_state.store(ESHKOL_RUNTIME_TERMINATED, std::memory_order_release);
}

eshkol_runtime_state_t eshkol_runtime_get_state(void) {
    return g_runtime_state.load(std::memory_order_acquire);
}

uint32_t eshkol_runtime_begin_operation(const char* name) {
    for (OperationSlot& slot : g_operations) {
        if (!slot.active) {
            slot.id = g_next_operation_id.fetch_add(1, std::memory_order_relaxed);
            slot.active = true;
            slot.name = name;
            g_operation_count.fetch_add(1, std::memory_order_relaxed);
            return slot.id;
        }
    }

    eshkol_runtime_warnf(
        "Freestanding operation capacity exceeded (max=%zu)",
        kMaxFreestandingOperations);
    return 0;
}

void eshkol_runtime_end_operation(uint32_t operation_id) {
    if (operation_id == 0) {
        return;
    }

    for (OperationSlot& slot : g_operations) {
        if (slot.active && slot.id == operation_id) {
            clear_operation_slot(slot);
            g_operation_count.fetch_sub(1, std::memory_order_relaxed);
            return;
        }
    }
}

bool eshkol_runtime_drain_operations(int timeout_ms) {
    if (g_operation_count.load(std::memory_order_acquire) == 0) {
        return true;
    }

    if (timeout_ms == 0) {
        return false;
    }

    const auto hook = g_operation_drain_hook.load(std::memory_order_acquire);
    if (!hook) {
        eshkol_runtime_warnf(
            "Freestanding runtime cannot drain operations without an installed operation-drain hook");
        return false;
    }

    void* context =
        g_operation_drain_hook_context.load(std::memory_order_relaxed);
    if (!hook(timeout_ms, context)) {
        return g_operation_count.load(std::memory_order_acquire) == 0;
    }

    return g_operation_count.load(std::memory_order_acquire) == 0;
}

uint32_t eshkol_runtime_get_operation_count(void) {
    return g_operation_count.load(std::memory_order_acquire);
}

arena_t* __global_arena = nullptr;

void arena_lock(arena_t* arena) { (void)arena; }

void arena_unlock(arena_t* arena) { (void)arena; }

arena_t* get_global_arena(void) { return __global_arena; }

eshkol_exception_t* eshkol_make_exception(eshkol_exception_type_t type,
                                          const char* message) {
    if (__global_arena) {
        eshkol_exception_t* exc = static_cast<eshkol_exception_t*>(
            arena_allocate(__global_arena, sizeof(eshkol_exception_t)));
        if (exc) {
            return initialize_exception_object(exc, type, message);
        }
    }

    return initialize_exception_object(&g_last_exception, type, message);
}

eshkol_exception_t* eshkol_make_exception_with_header(
    eshkol_exception_type_t type, const char* message) {
    if (!__global_arena) {
        return eshkol_make_exception(type, message);
    }

    const size_t data_size = sizeof(eshkol_exception_t);
    const size_t total_size =
        sizeof(eshkol_object_header_t) + data_size;
    auto* mem = static_cast<uint8_t*>(
        arena_allocate_aligned(__global_arena, total_size, alignof(uint64_t)));
    if (!mem) {
        return eshkol_make_exception(type, message);
    }

    auto* header = reinterpret_cast<eshkol_object_header_t*>(mem);
    header->subtype = HEAP_SUBTYPE_EXCEPTION;
    header->flags = 0;
    header->ref_count = 0;
    header->size = static_cast<uint32_t>(data_size);

    auto* exc = reinterpret_cast<eshkol_exception_t*>(
        mem + sizeof(eshkol_object_header_t));
    return initialize_exception_object(exc, type, message);
}

void eshkol_exception_add_irritant(eshkol_exception_t* exc,
                                   eshkol_tagged_value_t irritant) {
    if (!exc || !__global_arena) {
        return;
    }

    const uint32_t new_count = exc->num_irritants + 1;
    auto* new_irritants = static_cast<eshkol_tagged_value_t*>(arena_allocate(
        __global_arena, new_count * sizeof(eshkol_tagged_value_t)));
    if (!new_irritants) {
        return;
    }

    if (exc->irritants && exc->num_irritants > 0) {
        std::memcpy(new_irritants, exc->irritants,
                    exc->num_irritants * sizeof(eshkol_tagged_value_t));
    }

    new_irritants[exc->num_irritants] = irritant;
    exc->irritants = new_irritants;
    exc->num_irritants = new_count;
}

void eshkol_exception_set_location(eshkol_exception_t* exc, uint32_t line,
                                   uint32_t column, const char* filename) {
    if (!exc) {
        return;
    }

    exc->line = line;
    exc->column = column;
    exc->filename = duplicate_exception_string(filename);
}

void eshkol_set_raised_value(const eshkol_tagged_value_t* value) {
    if (!value) {
        return;
    }

    g_raised_tagged_value = *value;
    g_raised_value_set_by_user = true;
}

void eshkol_get_raised_value(eshkol_tagged_value_t* out) {
    if (!out) {
        return;
    }

    *out = g_raised_tagged_value;
}

eshkol_exception_t* eshkol_get_current_exception(void) {
    return g_current_exception;
}

void eshkol_clear_current_exception(void) {
    g_current_exception = nullptr;
    g_raised_tagged_value.type = ESHKOL_VALUE_NULL;
    g_raised_tagged_value.flags = 0;
    g_raised_tagged_value.reserved = 0;
    g_raised_tagged_value.data.raw_val = 0;
    g_raised_value_set_by_user = false;
}

void eshkol_raise(eshkol_exception_t* exception) {
    g_current_exception = exception;

    if (!g_raised_value_set_by_user) {
        g_raised_tagged_value.type = ESHKOL_VALUE_HEAP_PTR;
        g_raised_tagged_value.flags = 0;
        g_raised_tagged_value.reserved = 0;
        g_raised_tagged_value.data.ptr_val =
            reinterpret_cast<uint64_t>(exception);
    }
    g_raised_value_set_by_user = false;

    if (g_exception_handler_stack && g_exception_handler_stack->jmp_buf_ptr) {
        longjmp(*reinterpret_cast<jmp_buf*>(
                    g_exception_handler_stack->jmp_buf_ptr),
                1);
    }

    eshkol_runtime_fatalf("%s", exception && exception->message
                                    ? exception->message
                                    : "Unhandled freestanding exception");
}

void eshkol_push_exception_handler(void* jmp_buf_ptr) {
    if (g_exception_handler_count >= kMaxFreestandingExceptionHandlers) {
        eshkol_runtime_warnf(
            "Freestanding exception handler capacity exceeded (max=%zu)",
            kMaxFreestandingExceptionHandlers);
        return;
    }

    eshkol_exception_handler_t* handler =
        &g_exception_handlers[g_exception_handler_count++];
    handler->jmp_buf_ptr = jmp_buf_ptr;
    handler->prev = g_exception_handler_stack;
    g_exception_handler_stack = handler;
}

void eshkol_pop_exception_handler(void) {
    if (!g_exception_handler_stack) {
        return;
    }

    g_exception_handler_stack = g_exception_handler_stack->prev;
    if (g_exception_handler_count > 0) {
        --g_exception_handler_count;
    }
}

int eshkol_exception_type_matches(eshkol_exception_t* exc,
                                  eshkol_exception_type_t type) {
    return exc && exc->type == type;
}

void eshkol_display_exception(eshkol_exception_t* exc) {
    if (!exc) {
        eshkol_runtime_errorf("#<exception:null>");
        return;
    }

    const char* type_name = "unknown";
    switch (exc->type) {
        case ESHKOL_EXCEPTION_ERROR:
            type_name = "error";
            break;
        case ESHKOL_EXCEPTION_TYPE_ERROR:
            type_name = "type-error";
            break;
        case ESHKOL_EXCEPTION_FILE_ERROR:
            type_name = "file-error";
            break;
        case ESHKOL_EXCEPTION_READ_ERROR:
            type_name = "read-error";
            break;
        case ESHKOL_EXCEPTION_SYNTAX_ERROR:
            type_name = "syntax-error";
            break;
        case ESHKOL_EXCEPTION_RANGE_ERROR:
            type_name = "range-error";
            break;
        case ESHKOL_EXCEPTION_ARITY_ERROR:
            type_name = "arity-error";
            break;
        case ESHKOL_EXCEPTION_DIVIDE_BY_ZERO:
            type_name = "divide-by-zero";
            break;
        case ESHKOL_EXCEPTION_USER_DEFINED:
            type_name = "user-exception";
            break;
        default:
            break;
    }

    eshkol_runtime_errorf("#<%s: %s>", type_name,
                          exc->message ? exc->message : "");
}

int64_t eshkol_check_recursion_depth(void) {
    ++g_recursion_depth;
    if (g_recursion_depth > kMaxFreestandingRecursionDepth) {
        g_recursion_depth = 0;
        eshkol_exception_t* exc = eshkol_make_exception(
            ESHKOL_EXCEPTION_ERROR, "maximum recursion depth exceeded");
        eshkol_raise(exc);
    }
    return g_recursion_depth;
}

void eshkol_decrement_recursion_depth(void) {
    if (g_recursion_depth > 0) {
        --g_recursion_depth;
    }
}

void eshkol_reset_recursion_depth(void) { g_recursion_depth = 0; }

}  // extern "C"
