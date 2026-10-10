#include "../../lib/core/arena_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <setjmp.h>
#include <thread>
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_calloc(size_t, size_t);
extern "C" void* __real_arena_allocate(arena_t*, size_t);
extern "C" void* __real_arena_allocate_aligned(arena_t*, size_t, size_t);
extern "C" size_t eshkol_test_handler_pool_size();
extern "C" void eshkol_test_handler_pool_release();
extern "C" void eshkol_gradient_in_point_shape(arena_t*, const eshkol_tagged_value_t*,
                                                const eshkol_tagged_value_t*,
                                                eshkol_tagged_value_t*);
namespace {
int handler_calls = 0;
bool watch_handlers = false, refuse_aligned = false;
int fail_aligned_call_once = 0;
size_t fail_aligned_bytes_once = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
void arm() { handler_calls = 0; watch_handlers = true; }
void disarm() { watch_handlers = false; refuse_aligned = false; }
void release_pool() {
    check(!g_exception_handler_stack && eshkol_exception_handler_depth() == 0, "no active handler");
    eshkol_test_handler_pool_release();
}
void push_handlers(int depth) {
    if (!depth) return;
    jmp_buf buf;
    check(setjmp(buf) == 0, "handler push cannot fail");
    eshkol_push_exception_handler(&buf);
    check(g_exception_handler_stack->replay_values == nullptr &&
          g_exception_handler_stack->replay_capacity == 0, "fresh replay fields initialized");
    push_handlers(depth - 1);
    eshkol_pop_exception_handler();
}
void test_failed_push() {
    jmp_buf outer;
    if (!setjmp(outer)) {
        eshkol_push_exception_handler(&outer);
        arm();
        refuse_aligned = true; // persistent failure: raising may not allocate a condition
        jmp_buf rejected;
        if (setjmp(rejected)) check(false, "unpublished handler caught failure");
        eshkol_push_exception_handler(&rejected);
        check(false, "failed push must transfer");
    }
    disarm();
    check(handler_calls == 1 && eshkol_exception_handler_depth() == 1, "failed push preserves depth");
    check(g_current_exception && std::strstr(g_current_exception->message, "exception-handler push"),
          "failed push remains catchable with allocation refused");
    eshkol_pop_exception_handler();
    release_pool();
}
void test_replay() {
    release_pool();
    jmp_buf buf;
    check(setjmp(buf) == 0, "replay test does not raise");
    eshkol_push_exception_handler(&buf);
    check(g_exception_handler_stack->replay_values == nullptr &&
          g_exception_handler_stack->replay_capacity == 0, "fresh replay fields initialized");
    eshkol_tagged_value_t value{}; value.type = ESHKOL_VALUE_INT64; value.data.int_val = 42;
    eshkol_guard_replay_snapshot(&value, 1, 1);
    auto* replay = g_exception_handler_stack->replay_values;
    check(replay && g_exception_handler_stack->replay_capacity == 1, "snapshot on handler frame");
    eshkol_pop_exception_handler();
    arm();
    eshkol_push_exception_handler(&buf);
    check(g_exception_handler_stack->replay_values == replay &&
          g_exception_handler_stack->replay_capacity == 1 &&
          g_exception_handler_stack->replay_active == 0 &&
          g_exception_handler_stack->replay_count == 0, "reuse retains buffer and resets snapshot");
    eshkol_pop_exception_handler(); disarm(); release_pool();
}
void test_closure() {
    arena_t* arena = arena_create(4096); check(arena, "closure arena");
    check(!arena_allocate_closure_env(arena, SIZE_MAX), "closure capture size overflow rejected");
    // Wrapped allocator refuses precisely the environment after closure allocation.
    extern bool fail_environment;
    fail_environment = true;
    check(!arena_allocate_closure_with_header(arena, 1, CLOSURE_ENV_PACK(2, 0, 0), 0, 0, nullptr),
          "header closure never publishes missing environment");
    fail_environment = false;
    arena_destroy(arena);
}
bool test_gradient_shape_allocation_failure(int fail_call, size_t fail_bytes,
                                            const char* condition) {
    arena_t* inputs = arena_create(4096); check(inputs, "gradient input arena");
    arena_t* output = arena_create(4096); check(output, "gradient output arena");
    auto* point_tensor = arena_allocate_tensor_full(inputs, 2, 2);
    auto* grad_tensor = arena_allocate_tensor_full(inputs, 1, 2);
    check(point_tensor && grad_tensor, "gradient input descriptors");
    point_tensor->dimensions[0] = 1; point_tensor->dimensions[1] = 2;
    grad_tensor->dimensions[0] = 2;
    eshkol_tagged_value_t point{}, gradient{}, shaped{};
    point.type = gradient.type = ESHKOL_VALUE_HEAP_PTR;
    point.data.ptr_val = reinterpret_cast<uint64_t>(point_tensor);
    gradient.data.ptr_val = reinterpret_cast<uint64_t>(grad_tensor);

    jmp_buf handler;
    bool caught = false;
    if (setjmp(handler) == 0) {
        eshkol_push_exception_handler(&handler);
        fail_aligned_call_once = fail_call;
        fail_aligned_bytes_once = fail_bytes;
        eshkol_gradient_in_point_shape(output, &gradient, &point, &shaped);
    } else {
        caught = true;
    }
    fail_aligned_call_once = 0;
    fail_aligned_bytes_once = 0;
    const bool matched = caught && g_current_exception && g_current_exception->message &&
                         std::strstr(g_current_exception->message, condition);
    eshkol_pop_exception_handler();
    g_current_exception = nullptr;
    arena_destroy(output); arena_destroy(inputs);
    if (!matched) std::fprintf(stderr, "FAIL: gradient shape allocation did not raise %s\n", condition);
    return matched;
}
void test_gradient_shape_success() {
    arena_t* inputs = arena_create(4096); check(inputs, "gradient input arena");
    arena_t* output = arena_create(4096); check(output, "gradient output arena");
    auto* point_tensor = arena_allocate_tensor_full(inputs, 2, 2);
    auto* grad_tensor = arena_allocate_tensor_full(inputs, 1, 2);
    check(point_tensor && grad_tensor, "gradient input descriptors");
    point_tensor->dimensions[0] = 1; point_tensor->dimensions[1] = 2;
    grad_tensor->dimensions[0] = 2;
    eshkol_tagged_value_t point{}, gradient{}, shaped{};
    point.type = gradient.type = ESHKOL_VALUE_HEAP_PTR;
    point.data.ptr_val = reinterpret_cast<uint64_t>(point_tensor);
    gradient.data.ptr_val = reinterpret_cast<uint64_t>(grad_tensor);
    eshkol_gradient_in_point_shape(output, &gradient, &point, &shaped);
    auto* tensor = reinterpret_cast<eshkol_tensor_t*>(shaped.data.ptr_val);
    check(tensor && tensor->num_dimensions == 2 && tensor->total_elements == 2 &&
          tensor->dimensions[0] == 1 && tensor->dimensions[1] == 2 &&
          tensor->elements == grad_tensor->elements,
          "tensor gradient preserves point shape and gradient storage");
    arena_destroy(output); arena_destroy(inputs);
}
bool fail_environment = false;
}
extern "C" void* __wrap_malloc(size_t bytes) {
    if (watch_handlers && bytes == sizeof(eshkol_exception_handler_t)) {
        ++handler_calls;
        return nullptr;
    }
    return __real_malloc(bytes);
}
// Optimizers may fold malloc + zero-initialization into calloc.
extern "C" void* __wrap_calloc(size_t count, size_t bytes) {
    if (count && bytes <= SIZE_MAX / count && watch_handlers &&
        count * bytes == sizeof(eshkol_exception_handler_t)) {
        ++handler_calls;
        return nullptr;
    }
    return __real_calloc(count, bytes);
}
extern "C" void* __wrap_arena_allocate_aligned(arena_t* arena, size_t bytes, size_t alignment) {
    if (fail_aligned_call_once && --fail_aligned_call_once == 0) {
        return nullptr;
    }
    if (refuse_aligned || (fail_environment && bytes == sizeof(eshkol_closure_env_t) +
                                              2 * sizeof(eshkol_tagged_value_t))) return nullptr;
    return __real_arena_allocate_aligned(arena, bytes, alignment);
}
extern "C" void* __wrap_arena_allocate(arena_t* arena, size_t bytes) {
    if (fail_aligned_bytes_once && bytes == fail_aligned_bytes_once) {
        fail_aligned_bytes_once = 0;
        return nullptr;
    }
    return __real_arena_allocate(arena, bytes);
}
int main() {
    test_failed_push(); test_replay(); test_closure();
    const bool descriptor_failure =
        test_gradient_shape_allocation_failure(1, 0, "gradient shape descriptor");
    const bool dimensions_failure =
        test_gradient_shape_allocation_failure(0, 2 * sizeof(uint64_t),
                                               "gradient shape dimensions");
    push_handlers(1);
    std::thread worker([] {
        check(eshkol_test_handler_pool_size() == 0, "thread-local pool starts empty");
        // First push on this thread must prime the condition without a region
        // prelude. The subsequent failed push cannot allocate.
        test_failed_push();
        push_handlers(2); release_pool();
        g_current_exception = nullptr;
    });
    worker.join();
    check(eshkol_test_handler_pool_size() == 1, "worker leaves main pool intact");
    release_pool(); g_current_exception = nullptr;
    if (!descriptor_failure || !dimensions_failure) return 1;
    test_gradient_shape_success();
    std::puts("PASS allocation hardening: failure, retry, replay, reuse, closure and thread isolation");
}
