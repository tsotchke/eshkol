#include "../../lib/core/arena_memory.h"
#include <eshkol/core/resource_limits.h>

#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define ARENA_TEST_ASAN 1
#  endif
#endif
#if !defined(ARENA_TEST_ASAN) && defined(__SANITIZE_ADDRESS__)
#  define ARENA_TEST_ASAN 1
#endif
#if defined(ARENA_TEST_ASAN)
#include <sanitizer/asan_interface.h>
#endif

extern "C" int eshkol_arena_poison_enabled(void);
extern "C" size_t eshkol_arena_block_pool_cap(void);

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

class ScopedEnvironment {
public:
    explicit ScopedEnvironment(const char* name) : name_(name) {
        const char* current = std::getenv(name_.c_str());
        was_set_ = current != nullptr;
        if (was_set_) previous_ = current;
    }

    bool set(const char* value) {
#if defined(_WIN32)
        return _putenv_s(name_.c_str(), value) == 0;
#else
        return setenv(name_.c_str(), value, 1) == 0;
#endif
    }

    ~ScopedEnvironment() {
#if defined(_WIN32)
        (void)_putenv_s(name_.c_str(), was_set_ ? previous_.c_str() : "");
#else
        if (was_set_) (void)setenv(name_.c_str(), previous_.c_str(), 1);
        else (void)unsetenv(name_.c_str());
#endif
    }

private:
    std::string name_;
    std::string previous_;
    bool was_set_ = false;
};

bool is_aligned(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

#if !defined(_WIN32)
int run_heap_limit_pool_process_regression() {
    int pipefd[2];
    if (pipe(pipefd) != 0) return fail("heap-limit regression pipe creation failed");

    const pid_t child = fork();
    if (child < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return fail("heap-limit regression fork failed");
    }
    if (child == 0) {
        close(pipefd[0]);
        if (dup2(pipefd[1], STDERR_FILENO) < 0) _exit(90);
        close(pipefd[1]);

        eshkol_resource_limits_t limits = eshkol_get_default_limits();
        limits.max_heap_bytes = (size_t)3 << 20;
        limits.heap_soft_limit_bytes = 0;
        limits.active_limits = ESHKOL_LIMIT_ACTIVE_HEAP;
        limits.enforce_hard_limits = true;
        limits.enable_warnings = false;
        eshkol_set_limits(&limits);
        eshkol_reset_resource_tracking();
        unsetenv("ESHKOL_ARENA_POISON");
        unsetenv("ESHKOL_ARENA_BLOCK_POOL_MB");
        setenv("ESHKOL_ARENA_BLOCK_POOL_MB", "16", 1);

        // Warm one 2 MiB pooled block, then hold 2 MiB in four 512 KiB blocks.
        arena_t* warm = arena_create((size_t)2 << 20);
        if (!warm) { dprintf(STDERR_FILENO, "TEST_SETUP_FAIL warm\n"); _exit(90); }
        arena_destroy(warm);
        arena_t* live[4] = {};
        for (arena_t*& a : live) {
            a = arena_create((size_t)512 << 10);
            if (!a) { dprintf(STDERR_FILENO, "TEST_SETUP_FAIL live\n"); _exit(90); }
        }
        if (eshkol_get_heap_usage() != ((size_t)2 << 20)) {
            dprintf(STDERR_FILENO, "TEST_SETUP_FAIL usage=%zu\n", eshkol_get_heap_usage());
            _exit(90);
        }

        // Corrected core terminates here with ESHKOL_EXIT_LIMIT_HEAP after the
        // pooled block's extra 1 MiB charge is rejected.
        dprintf(STDERR_FILENO, "TARGET_REQUEST\n");
        arena_t* target = arena_create((size_t)1 << 20);
        (void)target;
        dprintf(STDERR_FILENO, "TEST_TARGET_RETURNED\n");
        _exit(91);
    }

    close(pipefd[1]);
    std::string diagnostic;
    char buffer[256];
    for (;;) {
        const ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
        if (n == 0) break;
        if (n < 0) {
            if (errno == EINTR) continue;
            close(pipefd[0]);
            return fail("heap-limit regression diagnostic read failed");
        }
        diagnostic.append(buffer, static_cast<size_t>(n));
    }
    close(pipefd[0]);

    int status = 0;
    for (;;) {
        if (waitpid(child, &status, 0) >= 0) break;
        if (errno != EINTR) return fail("heap-limit regression waitpid failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != ESHKOL_EXIT_LIMIT_HEAP) {
        std::cerr << "FAIL: pooled-block enforcement status=" << status
                  << " diagnostic=" << diagnostic;
        return 1;
    }
    if (diagnostic.find("TEST_SETUP_FAIL") != std::string::npos ||
        diagnostic.find("TARGET_REQUEST") == std::string::npos ||
        diagnostic.find("TEST_TARGET_RETURNED") != std::string::npos ||
        diagnostic.find("Heap hard limit exceeded") == std::string::npos ||
        diagnostic.find("pooled arena block") == std::string::npos) {
        std::cerr << "FAIL: pooled-block enforcement diagnostic mismatch: " << diagnostic;
        return 1;
    }
    return 0;
}
#endif

#if defined(ARENA_TEST_ASAN)
int verify_asan_pool_addressability() {
    ScopedEnvironment pool_cap("ESHKOL_ARENA_BLOCK_POOL_MB");
    ScopedEnvironment arena_poison("ESHKOL_ARENA_POISON");
    if (!pool_cap.set("16") || !arena_poison.set("0")) {
        return fail("could not set test-local arena-pool environment");
    }
    if (eshkol_arena_poison_enabled() != 0) {
        return fail("arena poison mode remained enabled in the ASan pool test");
    }
    if (eshkol_arena_block_pool_cap() < ((size_t)2 << 20)) {
        return fail("ASan pool test has no capacity for its 2 MiB block");
    }

    arena_t* first = arena_create((size_t)2 << 20);
    if (!first) return fail("asan pool arena_create returned null");
    arena_block_t* pooled_header = first->current_block;
    auto* inside = static_cast<unsigned char*>(arena_allocate(first, 64));
    if (!inside) return fail("asan pool allocation returned null");
    if (__asan_address_is_poisoned(inside)) return fail("live arena payload is poisoned");
    std::memset(inside, 0x5a, 64);
    arena_destroy(first);
    if (__asan_address_is_poisoned(pooled_header)) {
        return fail("pooled arena block header is poisoned");
    }
    if (!__asan_address_is_poisoned(inside)) {
        return fail("a pooled payload is addressable while it sits in the pool");
    }

    arena_t* second = arena_create((size_t)2 << 20);
    if (!second) return fail("asan pool reuse arena_create returned null");
    if (__asan_address_is_poisoned(second) ||
        __asan_address_is_poisoned(second->current_block)) {
        return fail("reused arena or block header is poisoned");
    }
    if (second->current_block != pooled_header) {
        return fail("ASan pool test did not reuse the same block header");
    }
    auto* reused = static_cast<unsigned char*>(arena_allocate(second, 64));
    if (!reused) return fail("asan pool reuse allocation returned null");
    if (reused != inside) return fail("ASan pool test did not reuse the poisoned block");
    if (__asan_address_is_poisoned(reused)) {
        return fail("reused arena payload is still poisoned");
    }
    std::memset(reused, 0xa5, 64);
    if (reused[0] != 0xa5 || reused[63] != 0xa5) {
        return fail("reused arena payload was not writable");
    }
    arena_destroy(second);
    return 0;
}

#if !defined(_WIN32)
int run_asan_pool_addressability_process() {
    const pid_t child = fork();
    if (child < 0) return fail("ASan pool regression fork failed");
    if (child == 0) _exit(verify_asan_pool_addressability() == 0 ? 0 : 1);

    int status = 0;
    for (;;) {
        if (waitpid(child, &status, 0) >= 0) break;
        if (errno != EINTR) return fail("ASan pool regression waitpid failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return fail("ASan pooled-block addressability regression failed in child");
    }
    std::cout << "PASS: ASan pool payload poison, block-header accessibility, and same-block reuse\n" << std::flush;
    return 0;
}
#else
int run_asan_pool_addressability_process() {
    if (verify_asan_pool_addressability() != 0) return 1;
    std::cout << "PASS: ASan pool payload poison, block-header accessibility, and same-block reuse\n" << std::flush;
    return 0;
}
#endif
#endif

}  // namespace

int main() {
#if defined(ARENA_TEST_ASAN)
    // Use a fresh pool and initialize its cached settings in a child, leaving
    // the parent test process and its environment unchanged.
    if (run_asan_pool_addressability_process() != 0) return 1;
#endif
#if defined(_WIN32)
    // The process-isolated enforcement regression is POSIX-only; the rest of
    // this arena test remains portable.
#else
    if (run_heap_limit_pool_process_regression() != 0) return 1;
#endif

    if (arena_get_used_memory(nullptr) != 0) return fail("null used-memory query mismatch");
    if (arena_get_total_memory(nullptr) != 0) return fail("null total-memory query mismatch");
    if (arena_get_block_count(nullptr) != 0) return fail("null block-count query mismatch");

    arena_t* arena = arena_create(8);
    if (!arena) return fail("arena_create returned null");
    if (arena_get_total_memory(arena) != 1024) return fail("minimum block size mismatch");
    if (arena_get_block_count(arena) != 1) return fail("initial block count mismatch");
    if (arena_get_used_memory(arena) != 0) return fail("initial used memory mismatch");

    if (arena_allocate(nullptr, 8) != nullptr) return fail("null arena allocation succeeded");
    if (arena_allocate(arena, 0) != nullptr) return fail("zero-size allocation succeeded");

    void* first = arena_allocate(arena, 7);
    if (!first) return fail("basic arena allocation returned null");
    if (!is_aligned(first, 8)) return fail("default allocation was not 8-byte aligned");
    if (arena_get_used_memory(arena) < 7) return fail("used memory did not increase");

    const size_t used_before_invalid = arena_get_used_memory(arena);
    if (arena_allocate_aligned(arena, 8, 24) != nullptr) {
        return fail("non-power-of-two alignment allocation succeeded");
    }
    if (arena_get_used_memory(arena) != used_before_invalid) {
        return fail("invalid alignment changed arena state");
    }

    void* aligned = arena_allocate_aligned(arena, 1, 32);
    if (!aligned) return fail("over-aligned allocation returned null");
    if (!is_aligned(aligned, 32)) return fail("over-aligned allocation returned misaligned pointer");

    auto* zeroed = static_cast<unsigned char*>(arena_allocate_zeroed(arena, 32));
    if (!zeroed) return fail("zeroed allocation returned null");
    for (size_t i = 0; i < 32; ++i) {
        if (zeroed[i] != 0) return fail("zeroed allocation contained non-zero byte");
    }

    arena_cons_cell_t* cons = arena_allocate_cons_cell(arena);
    if (!cons) return fail("legacy cons allocation returned null");
    cons->car = 11;
    cons->cdr = 22;
    if (cons->car != 11 || cons->cdr != 22) return fail("legacy cons write/read mismatch");

    auto* nodes = static_cast<int64_t*>(arena_allocate_list_node(arena, sizeof(int64_t), 3));
    if (!nodes) return fail("list-node allocation returned null");
    nodes[0] = 1;
    nodes[1] = 2;
    nodes[2] = 3;
    if (nodes[2] != 3) return fail("list-node write/read mismatch");

    const size_t used_before_scope = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 2048)) return fail("scoped large allocation returned null");
    if (arena_get_block_count(arena) <= 1) return fail("large scoped allocation did not add block");
    arena_pop_scope(arena);
    if (arena_get_used_memory(arena) != used_before_scope) {
        return fail("scope pop did not restore used memory");
    }
    if (arena_get_block_count(arena) != 1) return fail("scope pop did not release extra block");

    if (!arena_allocate(arena, 4096)) return fail("large allocation returned null");
    if (arena_get_block_count(arena) <= 1) return fail("large allocation did not add block");
    arena_reset(arena);
    if (arena_get_used_memory(arena) != 0) return fail("reset did not clear used memory");
    if (arena_get_block_count(arena) != 1) return fail("reset did not restore one block");
    if (arena_get_total_memory(arena) != 1024) return fail("reset did not restore total memory");

    const size_t max = std::numeric_limits<size_t>::max();
    if (arena_allocate_aligned(arena, max, 8) != nullptr) {
        return fail("overflowing aligned allocation succeeded");
    }
    if (arena_allocate_list_node(arena, max / 2 + 1, 3) != nullptr) {
        return fail("overflowing list-node allocation succeeded");
    }

    // ── ESH-0214b: per-iteration loop scope primitives ──

    // arena_commit_scope keeps allocations but drops the scope record.
    const size_t used_before_commit = arena_get_used_memory(arena);
    arena_push_scope(arena);
    void* committed_alloc = arena_allocate(arena, 64);
    if (!committed_alloc) return fail("commit-scope test allocation returned null");
    arena_commit_scope(arena);
    if (arena_get_used_memory(arena) <= used_before_commit) {
        return fail("commit released memory it should have kept");
    }
    std::memset(committed_alloc, 0x5A, 64);  // must still be writable (ASAN lane)

    // arena_top_scope_contains: in-span vs pre-span vs foreign pointers.
    void* before_scope = arena_allocate(arena, 16);
    arena_push_scope(arena);
    void* in_scope = arena_allocate(arena, 16);
    int on_stack = 0;
    if (!arena_top_scope_contains(arena, in_scope)) {
        return fail("in-scope pointer not detected");
    }
    if (arena_top_scope_contains(arena, before_scope)) {
        return fail("pre-scope pointer misdetected as in-scope");
    }
    if (arena_top_scope_contains(arena, &on_stack)) {
        return fail("foreign (stack) pointer misdetected as in-scope");
    }
    // ...including across a block boundary added inside the scope.
    void* in_scope_big = arena_allocate(arena, 4096);
    if (!arena_top_scope_contains(arena, in_scope_big)) {
        return fail("in-scope pointer in overflow block not detected");
    }
    arena_pop_scope(arena);

    // eshkol_arena_iter_scope_end: POP path (no out-flowing heap values).
    const size_t used_before_iter = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 512)) return fail("iter-pop test allocation returned null");
    eshkol_tagged_value_t ints[2];
    std::memset(ints, 0, sizeof(ints));
    ints[0].type = ESHKOL_VALUE_INT64;  ints[0].data.int_val = 41;
    ints[1].type = ESHKOL_VALUE_DOUBLE; ints[1].data.double_val = 2.5;
    eshkol_arena_iter_scope_end(arena, ints, 2);
    if (arena_get_used_memory(arena) != used_before_iter) {
        return fail("iter-scope-end with immediates did not pop (reclaim)");
    }

    // POP path with a heap value that lies OUTSIDE the scope span (the
    // carried-port shape): reclamation must still happen.
    void* pre_alloc = arena_allocate(arena, 32);
    const size_t used_before_iter2 = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 256)) return fail("iter-pop2 test allocation returned null");
    eshkol_tagged_value_t carried;
    std::memset(&carried, 0, sizeof(carried));
    carried.type = ESHKOL_VALUE_HEAP_PTR;
    carried.data.ptr_val = (uint64_t)(uintptr_t)pre_alloc;
    eshkol_arena_iter_scope_end(arena, &carried, 1);
    if (arena_get_used_memory(arena) != used_before_iter2) {
        return fail("iter-scope-end with pre-scope heap value did not pop");
    }

    // COMMIT path: an out-flowing heap value allocated INSIDE the scope.
    arena_push_scope(arena);
    void* escaping = arena_allocate(arena, 128);
    if (!escaping) return fail("iter-commit test allocation returned null");
    const size_t used_at_commit = arena_get_used_memory(arena);
    eshkol_tagged_value_t esc;
    std::memset(&esc, 0, sizeof(esc));
    esc.type = ESHKOL_VALUE_HEAP_PTR;
    esc.data.ptr_val = (uint64_t)(uintptr_t)escaping;
    eshkol_arena_iter_scope_end(arena, &esc, 1);
    if (arena_get_used_memory(arena) != used_at_commit) {
        return fail("iter-scope-end with escaping heap value did not commit (keep memory)");
    }
    std::memset(escaping, 0x7E, 128);  // must still be writable (ASAN lane)

    // Conservative typing: an unknown/exotic type tag with a null pointer
    // must not block reclamation (eof-object shape), and the scope stack
    // must stay balanced through mixed pop/commit sequences.
    const size_t used_mixed = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 64)) return fail("mixed test allocation returned null");
    eshkol_tagged_value_t eof;
    std::memset(&eof, 0, sizeof(eof));
    eof.type = 0xFF;  // eof-object: data always 0
    eshkol_arena_iter_scope_end(arena, &eof, 1);
    if (arena_get_used_memory(arena) != used_mixed) {
        return fail("iter-scope-end with eof-object did not pop");
    }

    arena_destroy(arena);

    // Large-block pool accounting: a block released by a scope pop goes to the
    // pool, and a later smaller request can reuse it. The arena must charge the
    // block's real size, because every release subtracts block->size; charging
    // the requested size instead made total_allocated drift down (and wrap
    // below zero for a small arena) after each such reuse.
    {
        arena_t* pooled = arena_create(1024);
        if (!pooled) return fail("pool-accounting arena_create returned null");
        const size_t base_total = arena_get_total_memory(pooled);

        arena_push_scope(pooled);
        if (!arena_allocate(pooled, (size_t)1900 * 1024)) return fail("1.9 MiB allocation returned null");
        arena_pop_scope(pooled);
        if (arena_get_total_memory(pooled) != base_total) {
            return fail("scope pop did not release the 1.9 MiB block from the arena total");
        }

        arena_push_scope(pooled);  // may be served by the pooled 1.9 MiB block
        if (!arena_allocate(pooled, (size_t)1100 * 1024)) return fail("1.1 MiB allocation returned null");
        arena_pop_scope(pooled);
        if (arena_get_total_memory(pooled) != base_total) {
            return fail("arena total drifted after reusing a larger pooled block");
        }
        arena_destroy(pooled);
    }

    std::cout << "PASS\n";
    return 0;
}
