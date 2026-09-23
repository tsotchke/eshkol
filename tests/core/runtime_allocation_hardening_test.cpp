#include "../../lib/core/arena_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <setjmp.h>
#include <thread>
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_calloc(size_t, size_t);
extern "C" void* __real_arena_allocate_aligned(arena_t*, size_t, size_t);
extern "C" size_t eshkol_test_handler_pool_size();
extern "C" void eshkol_test_handler_pool_release();
namespace {
int remaining = -1, handler_calls = 0;
bool watch_handlers = false, refuse_aligned = false;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
void arm(int n) { remaining = n; handler_calls = 0; watch_handlers = true; }
void disarm() { watch_handlers = false; refuse_aligned = false; }
void release_pool() {
    check(!g_exception_handler_stack && eshkol_exception_handler_depth() == 0, "no active handler");
    eshkol_test_handler_pool_release();
}
void push_reserved(int depth) {
    if (!depth) return;
    jmp_buf buf;
    check(setjmp(buf) == 0, "reserved push cannot fail");
    eshkol_push_exception_handler(&buf);
    check(g_exception_handler_stack->replay_values == nullptr &&
          g_exception_handler_stack->replay_capacity == 0, "reserved replay fields initialized");
    push_reserved(depth - 1);
    eshkol_pop_exception_handler();
}
void test_reservation() {
    release_pool();
    arm(0);
    check(eshkol_runtime_reserve_exception_handlers_v1(0) == 0, "zero reserve");
    check(eshkol_runtime_reserve_exception_handlers_v1(-1) == -1, "negative rejected");
    check(eshkol_runtime_reserve_exception_handlers_v1(INT64_MAX) == -1, "overflow rejected");
    check(handler_calls == 0 && eshkol_test_handler_pool_size() == 0, "invalid reserve does not mutate");
    disarm();
    jmp_buf outer;
    if (!setjmp(outer)) {
        eshkol_push_exception_handler(&outer);
        arm(2);
        eshkol_runtime_reserve_exception_handlers_v1(5);
        check(false, "partial reserve must fail");
    }
    disarm();
    check(handler_calls == 3 && eshkol_test_handler_pool_size() == 2, "partial prefix reusable");
    check(g_current_exception && std::strstr(g_current_exception->message, "exception-handler reservation"),
          "reservation failure reached established handler");
    check(eshkol_exception_handler_depth() == 1, "failed reserve did not publish handler");
    eshkol_pop_exception_handler();
    arm(-1);
    check(eshkol_runtime_reserve_exception_handlers_v1(5) == 0 && handler_calls == 2,
          "retry allocates missing deficit only");
    disarm();
    for (int i=0; i<3; ++i) {
        arm(0);
        check(eshkol_runtime_reserve_exception_handlers_v1(5) == 0, "idempotent reserve");
        push_reserved(5);
        check(handler_calls == 0 && eshkol_test_handler_pool_size() == 5, "repeated pushes reuse pool");
        disarm();
    }
    release_pool();
}
void test_failed_push() {
    jmp_buf outer;
    if (!setjmp(outer)) {
        eshkol_push_exception_handler(&outer);
        arm(0);
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
    check(eshkol_runtime_reserve_exception_handlers_v1(1) == 0, "reserve replay frame");
    jmp_buf buf;
    check(setjmp(buf) == 0, "replay test does not raise");
    eshkol_push_exception_handler(&buf);
    eshkol_tagged_value_t value{}; value.type = ESHKOL_VALUE_INT64; value.data.int_val = 42;
    eshkol_guard_replay_snapshot(&value, 1, 1);
    auto* replay = g_exception_handler_stack->replay_values;
    check(replay && g_exception_handler_stack->replay_capacity == 1, "snapshot on reserved frame");
    eshkol_pop_exception_handler();
    arm(0);
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
bool fail_environment = false;
}
extern "C" void* __wrap_malloc(size_t bytes) {
    if (watch_handlers && bytes == sizeof(eshkol_exception_handler_t)) {
        ++handler_calls;
        if (remaining == 0) return nullptr;
        if (remaining > 0) --remaining;
    }
    return __real_malloc(bytes);
}
// Optimizers may fold malloc + zero-initialization into calloc.
extern "C" void* __wrap_calloc(size_t count, size_t bytes) {
    if (count && bytes <= SIZE_MAX / count && watch_handlers &&
        count * bytes == sizeof(eshkol_exception_handler_t)) {
        ++handler_calls;
        if (remaining == 0) return nullptr;
        if (remaining > 0) --remaining;
    }
    return __real_calloc(count, bytes);
}
extern "C" void* __wrap_arena_allocate_aligned(arena_t* arena, size_t bytes, size_t alignment) {
    if (refuse_aligned || (fail_environment && bytes == sizeof(eshkol_closure_env_t) +
                                              2 * sizeof(eshkol_tagged_value_t))) return nullptr;
    return __real_arena_allocate_aligned(arena, bytes, alignment);
}
int main() {
    test_reservation(); test_failed_push(); test_replay(); test_closure();
    check(eshkol_runtime_reserve_exception_handlers_v1(1) == 0, "main thread pool");
    std::thread worker([] {
        check(eshkol_test_handler_pool_size() == 0, "thread-local pool starts empty");
        // First push on this thread must prime the condition without a region
        // or explicit reservation. The subsequent failed push cannot allocate.
        test_failed_push();
        check(eshkol_runtime_reserve_exception_handlers_v1(2) == 0, "worker reserve");
        push_reserved(2); release_pool();
        g_current_exception = nullptr;
    });
    worker.join();
    check(eshkol_test_handler_pool_size() == 1, "worker leaves main pool intact");
    release_pool(); g_current_exception = nullptr;
    std::puts("PASS allocation hardening: failure, retry, replay, reuse, closure and thread isolation");
}
