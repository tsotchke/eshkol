/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Stackful asymmetric coroutines: stacks, the machine context switch, the
 * execution-context switch, the handle registry and the builtin entry points.
 * The design and the reasons for separate stacks are in
 * inc/eshkol/core/coroutine.h.
 */

#include "eshkol/core/coroutine.h"
#include "eshkol/eshkol.h"
#include "eshkol/core/resource_limits.h"
#include "arena_memory.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  define ESHKOL_CORO_WINDOWS 1
#elif defined(__wasm__) || defined(__EMSCRIPTEN__)
#  define ESHKOL_CORO_UNAVAILABLE 1
#elif (defined(__x86_64__) || defined(__aarch64__)) && \
      (defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__))
#  include <sys/mman.h>
#  include <unistd.h>
#  define ESHKOL_CORO_ASM 1
#else
#  define ESHKOL_CORO_UNAVAILABLE 1
#endif

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define ESHKOL_CORO_ASAN 1
#  endif
#  if __has_feature(thread_sanitizer)
#    define ESHKOL_CORO_TSAN 1
#  endif
#endif
#if !defined(ESHKOL_CORO_ASAN) && defined(__SANITIZE_ADDRESS__)
#  define ESHKOL_CORO_ASAN 1
#endif
#if !defined(ESHKOL_CORO_TSAN) && defined(__SANITIZE_THREAD__)
#  define ESHKOL_CORO_TSAN 1
#endif

#if defined(ESHKOL_CORO_ASAN)
extern "C" void __sanitizer_start_switch_fiber(void** fake_stack_save,
                                               const void* bottom, size_t size);
extern "C" void __sanitizer_finish_switch_fiber(void* fake_stack_save,
                                                const void** bottom_old,
                                                size_t* size_old);
#endif
#if defined(ESHKOL_CORO_TSAN)
extern "C" void* __tsan_get_current_fiber(void);
extern "C" void* __tsan_create_fiber(unsigned flags);
extern "C" void __tsan_destroy_fiber(void* fiber);
extern "C" void __tsan_switch_to_fiber(void* fiber, unsigned flags);
#endif

/* Defined in lib/core/runtime_errors_hosted.cpp: raises a catchable condition
 * and does not return. */
extern "C" void eshkol_runtime_fatal(int type, const char* fmt, ...);
extern "C" void* eshkol_intern_symbol_lookup(const char* name);
extern "C" void eshkol_raise_object(const eshkol_tagged_value_t* obj);
extern "C" void eshkol_ad_state_capture(unsigned char* mode_active, uint64_t* tape_depth,
                                        void** current_tape, void** seed_node,
                                        uint64_t* mixed_record_count);

namespace {

/* ── Tagged-value helpers ─────────────────────────────────────────────── */

eshkol_tagged_value_t tv_null() {
    eshkol_tagged_value_t v;
    std::memset(&v, 0, sizeof(v));
    v.type = ESHKOL_VALUE_NULL;
    return v;
}

eshkol_tagged_value_t tv_bool(bool b) {
    eshkol_tagged_value_t v = tv_null();
    v.type = ESHKOL_VALUE_BOOL;
    v.data.int_val = b ? 1 : 0;
    return v;
}

eshkol_tagged_value_t tv_int(int64_t i) {
    eshkol_tagged_value_t v = tv_null();
    v.type = ESHKOL_VALUE_INT64;
    v.flags = ESHKOL_VALUE_EXACT_FLAG;
    v.data.int_val = i;
    return v;
}

eshkol_tagged_value_t tv_symbol(const char* name) {
    void* sym = eshkol_intern_symbol_lookup(name);
    if (!sym) return tv_bool(false);
    eshkol_tagged_value_t v = tv_null();
    v.type = ESHKOL_VALUE_HEAP_PTR;
    v.data.ptr_val = (uint64_t)(uintptr_t)sym;
    return v;
}

eshkol_tagged_value_t tv_pair(eshkol_tagged_value_t car, eshkol_tagged_value_t cdr) {
    arena_t* arena = get_global_arena();
    void* cell = arena ? arena_allocate_cons_with_header(arena) : nullptr;
    if (!cell) return tv_null();
    std::memcpy(cell, &car, sizeof(car));
    std::memcpy((char*)cell + sizeof(car), &cdr, sizeof(cdr));
    eshkol_tagged_value_t v = tv_null();
    v.type = ESHKOL_VALUE_HEAP_PTR;
    v.data.ptr_val = (uint64_t)(uintptr_t)cell;
    return v;
}

bool tv_is_int(const eshkol_tagged_value_t* v) {
    return v && v->type == ESHKOL_VALUE_INT64;
}

[[noreturn]] void raise_error(const char* fmt, const char* who) {
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, fmt, who);
    std::abort();   /* unreachable: eshkol_runtime_fatal raises */
}

/* ── Stacks ────────────────────────────────────────────────────────────── */

constexpr size_t kDefaultStackBytes = 1024u * 1024u;   /* reserved, committed lazily */
constexpr size_t kMinStackBytes = 128u * 1024u;
constexpr size_t kGuardBytes = 64u * 1024u;            /* PROT_NONE below the stack */
constexpr size_t kStackPoolMax = 64;                   /* parked stacks kept per thread */

size_t page_size() {
#if defined(ESHKOL_CORO_ASM)
    static const size_t p = (size_t)sysconf(_SC_PAGESIZE);
    return p ? p : 4096u;
#else
    return 4096u;
#endif
}

size_t round_up(size_t n, size_t to) { return (n + to - 1) / to * to; }

/* ESHKOL_FIBER_STACK_BYTES: reserved bytes per coroutine stack. Pages are
 * committed only when touched, so the reservation bounds recursion depth
 * inside a coroutine without costing resident memory. */
size_t configured_stack_bytes() {
    static size_t cached = 0;
    if (cached) return cached;
    size_t want = kDefaultStackBytes;
    if (const char* env = std::getenv("ESHKOL_FIBER_STACK_BYTES")) {
        size_t parsed = 0;
        if (eshkol_parse_size(env, &parsed) && parsed >= kMinStackBytes) want = parsed;
    }
    cached = round_up(want, page_size());
    return cached;
}

struct Stack {
    void* mapping = nullptr;     /* base of the whole mapping, guard included */
    size_t mapping_bytes = 0;
    uintptr_t low = 0;           /* lowest usable address (above the guard) */
    uintptr_t high = 0;          /* one past the highest usable address */
};

/* ── Contexts and coroutines ───────────────────────────────────────────── */

struct Ctx {
    eshkol_exec_context_t exec;
    void* sp = nullptr;                   /* saved machine stack pointer / Windows fiber */
#if defined(ESHKOL_CORO_ASAN)
    void* asan_fake = nullptr;
    const void* asan_bottom = nullptr;
    size_t asan_size = 0;
#endif
#if defined(ESHKOL_CORO_TSAN)
    void* tsan_fiber = nullptr;
#endif
    Ctx() { std::memset(&exec, 0, sizeof(exec)); }
};

struct Coroutine {
    Ctx ctx;
    int64_t handle = 0;
    uint64_t owner_thread = 0;
    eshkol_coroutine_status_t status = ESHKOL_COROUTINE_SUSPENDED;
    bool started = false;
    bool inject_raise = false;          /* the pending yield raises `transfer` */
    eshkol_tagged_value_t thunk;
    eshkol_tagged_value_t transfer;     /* value moving across the switch */
    eshkol_boundary_raise_t failure;    /* recorded raise when FAILED */
    Stack stack;
    Coroutine* resumer = nullptr;       /* nullptr = the thread's root context */
};

uint64_t next_thread_token() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

std::atomic<uint64_t> g_next_context_id{1};

struct ThreadState {
    Ctx root;
    Coroutine* current = nullptr;       /* nullptr while the root context runs */
    uint64_t token = next_thread_token();
    std::vector<Stack> pool;
    uint64_t live = 0;
    uint64_t peak_live = 0;
    uint64_t created = 0;
    uint64_t mapped = 0;
#if defined(ESHKOL_CORO_WINDOWS)
    bool converted = false;
#endif
    ~ThreadState();
};

thread_local ThreadState t_state;

/* ── Handle registry ───────────────────────────────────────────────────────
 *
 * Process-wide and locked, because a handle is a plain integer any thread can
 * hold; a coroutine itself is only ever run by the thread that created it
 * (checked on every resume). Slot 0 is reserved so a handle is never 0, and the
 * generation in the high bits makes a released handle fail closed after its
 * slot is reused. */
constexpr int kSlotBits = 24;
constexpr int64_t kSlotMask = (int64_t(1) << kSlotBits) - 1;

struct Slot {
    Coroutine* co = nullptr;
    int64_t generation = 0;
};

std::mutex g_registry_mutex;
std::vector<Slot> g_slots(1);
std::vector<int64_t> g_free_slots;
uint64_t g_handles_open = 0;

int64_t registry_add(Coroutine* co) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    int64_t slot;
    if (!g_free_slots.empty()) {
        slot = g_free_slots.back();
        g_free_slots.pop_back();
    } else {
        if ((int64_t)g_slots.size() > kSlotMask) return -1;
        g_slots.emplace_back();
        slot = (int64_t)g_slots.size() - 1;
    }
    Slot& s = g_slots[(size_t)slot];
    s.co = co;
    ++s.generation;
    ++g_handles_open;
    return (s.generation << kSlotBits) | slot;
}

Coroutine* registry_get(int64_t handle) {
    if (handle <= 0) return nullptr;
    const int64_t slot = handle & kSlotMask;
    const int64_t generation = handle >> kSlotBits;
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (slot <= 0 || slot >= (int64_t)g_slots.size()) return nullptr;
    const Slot& s = g_slots[(size_t)slot];
    return (s.co && s.generation == generation) ? s.co : nullptr;
}

void registry_remove(int64_t handle) {
    const int64_t slot = handle & kSlotMask;
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (slot <= 0 || slot >= (int64_t)g_slots.size()) return;
    g_slots[(size_t)slot].co = nullptr;
    g_free_slots.push_back(slot);
    if (g_handles_open) --g_handles_open;
}

}  // namespace

/* ── Machine context switch ───────────────────────────────────────────────
 *
 * eshkol_coro_switch(save, load) pushes the callee-saved registers on the
 * current stack, stores the stack pointer in *save, installs `load` and pops
 * the registers saved there. A fresh stack is laid out so that the pop lands
 * in eshkol_coro_trampoline with the coroutine in a callee-saved register and
 * the entry function in another, with the stack 16-byte aligned for the call.
 */
#if defined(ESHKOL_CORO_ASM)
extern "C" void eshkol_coro_switch(void** save_sp, void* load_sp);
extern "C" void eshkol_coro_trampoline(void);

#  if defined(__APPLE__)
#    define ESHKOL_CORO_SYM(name) "_" #name
#    define ESHKOL_CORO_FN_BEGIN(name) \
        ".text\n.globl " ESHKOL_CORO_SYM(name) "\n.private_extern " \
        ESHKOL_CORO_SYM(name) "\n.p2align 4\n" ESHKOL_CORO_SYM(name) ":\n"
#    define ESHKOL_CORO_FN_END(name) ""
#  else
#    define ESHKOL_CORO_SYM(name) #name
#    define ESHKOL_CORO_FN_BEGIN(name) \
        ".text\n.globl " #name "\n.hidden " #name "\n.type " #name \
        ",%function\n.p2align 4\n" #name ":\n"
#    define ESHKOL_CORO_FN_END(name) ".size " #name ", .-" #name "\n"
#  endif

#  if defined(__x86_64__)
/* SysV x86-64: rbx, rbp, r12-r15 are callee-saved, plus the MXCSR control
 * bits and the x87 control word. Frame (from the saved sp up): csr(8) r15 r14
 * r13 r12 rbx rbp ret. */
__asm__(
    ESHKOL_CORO_FN_BEGIN(eshkol_coro_switch)
    "    pushq %rbp\n"
    "    pushq %rbx\n"
    "    pushq %r12\n"
    "    pushq %r13\n"
    "    pushq %r14\n"
    "    pushq %r15\n"
    "    subq $8, %rsp\n"
    "    stmxcsr (%rsp)\n"
    "    fnstcw 4(%rsp)\n"
    "    movq %rsp, (%rdi)\n"
    "    movq %rsi, %rsp\n"
    "    ldmxcsr (%rsp)\n"
    "    fldcw 4(%rsp)\n"
    "    addq $8, %rsp\n"
    "    popq %r15\n"
    "    popq %r14\n"
    "    popq %r13\n"
    "    popq %r12\n"
    "    popq %rbx\n"
    "    popq %rbp\n"
    "    ret\n"
    ESHKOL_CORO_FN_END(eshkol_coro_switch)
    ESHKOL_CORO_FN_BEGIN(eshkol_coro_trampoline)
    "    movq %r12, %rdi\n"
    "    callq *%r13\n"
    "    ud2\n"
    ESHKOL_CORO_FN_END(eshkol_coro_trampoline)
);
constexpr size_t kInitialFrameBytes = 64;
#  elif defined(__aarch64__)
/* AAPCS64: x19-x28, the frame pointer x29, the link register x30 and the low
 * halves of d8-d15 are callee-saved. Frame: 176 bytes, 16-byte aligned. */
__asm__(
    ESHKOL_CORO_FN_BEGIN(eshkol_coro_switch)
    "    sub sp, sp, #176\n"
    "    stp x19, x20, [sp, #0]\n"
    "    stp x21, x22, [sp, #16]\n"
    "    stp x23, x24, [sp, #32]\n"
    "    stp x25, x26, [sp, #48]\n"
    "    stp x27, x28, [sp, #64]\n"
    "    stp x29, x30, [sp, #80]\n"
    "    stp d8, d9, [sp, #96]\n"
    "    stp d10, d11, [sp, #112]\n"
    "    stp d12, d13, [sp, #128]\n"
    "    stp d14, d15, [sp, #144]\n"
    "    mov x9, sp\n"
    "    str x9, [x0]\n"
    "    mov sp, x1\n"
    "    ldp x19, x20, [sp, #0]\n"
    "    ldp x21, x22, [sp, #16]\n"
    "    ldp x23, x24, [sp, #32]\n"
    "    ldp x25, x26, [sp, #48]\n"
    "    ldp x27, x28, [sp, #64]\n"
    "    ldp x29, x30, [sp, #80]\n"
    "    ldp d8, d9, [sp, #96]\n"
    "    ldp d10, d11, [sp, #112]\n"
    "    ldp d12, d13, [sp, #128]\n"
    "    ldp d14, d15, [sp, #144]\n"
    "    add sp, sp, #176\n"
    "    ret\n"
    ESHKOL_CORO_FN_END(eshkol_coro_switch)
    ESHKOL_CORO_FN_BEGIN(eshkol_coro_trampoline)
    "    mov x0, x19\n"
    "    blr x20\n"
    "    brk #1\n"
    ESHKOL_CORO_FN_END(eshkol_coro_trampoline)
);
constexpr size_t kInitialFrameBytes = 176;
#  endif
#endif  /* ESHKOL_CORO_ASM */

namespace {

#if !defined(ESHKOL_CORO_UNAVAILABLE)

void coroutine_run(Coroutine* co);

#  if defined(ESHKOL_CORO_ASM)
/* A fresh stack's first frame. coroutine_run's final switch never comes back
 * here: a stack is re-laid-out before it is used again. */
extern "C" [[noreturn]] void eshkol_coro_entry(void* arg) {
    coroutine_run(static_cast<Coroutine*>(arg));
    __builtin_trap();
}

bool stack_acquire(ThreadState& ts, Stack* out) {
    if (!ts.pool.empty()) {
        *out = ts.pool.back();
        ts.pool.pop_back();
        return true;
    }
    const size_t usable = configured_stack_bytes();
    const size_t guard = round_up(kGuardBytes, page_size());
    const size_t total = usable + guard;
    int flags = MAP_PRIVATE | MAP_ANON;
#    if defined(MAP_NORESERVE)
    flags |= MAP_NORESERVE;
#    endif
#    if defined(MAP_STACK) && !defined(__APPLE__)
    flags |= MAP_STACK;
#    endif
    void* mem = mmap(nullptr, total, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (mem == MAP_FAILED) return false;
    if (mprotect(mem, guard, PROT_NONE) != 0) {
        munmap(mem, total);
        return false;
    }
    out->mapping = mem;
    out->mapping_bytes = total;
    out->low = (uintptr_t)mem + guard;
    out->high = (uintptr_t)mem + total;
    ts.mapped++;
    return true;
}

void stack_release(ThreadState& ts, Stack& s) {
    if (!s.mapping) return;
    if (ts.pool.size() < kStackPoolMax) {
        ts.pool.push_back(s);
    } else {
        munmap(s.mapping, s.mapping_bytes);
    }
    s = Stack{};
}

/* Lay out the initial frame eshkol_coro_switch pops into the trampoline. */
void* stack_prepare(Coroutine* co) {
    uintptr_t top = co->stack.high & ~(uintptr_t)15;
    auto* frame = reinterpret_cast<uint64_t*>(top - kInitialFrameBytes);
    std::memset(frame, 0, kInitialFrameBytes);
#    if defined(__x86_64__)
    frame[0] = 0x037F00001F80ull;           /* fpucw 0x037F @+4, mxcsr 0x1F80 @+0 */
    frame[3] = (uint64_t)(uintptr_t)&eshkol_coro_entry;   /* r13 */
    frame[4] = (uint64_t)(uintptr_t)co;                   /* r12 */
    frame[6] = 0;                                         /* rbp */
    frame[7] = (uint64_t)(uintptr_t)&eshkol_coro_trampoline;  /* ret */
#    elif defined(__aarch64__)
    frame[0] = (uint64_t)(uintptr_t)co;                   /* x19 */
    frame[1] = (uint64_t)(uintptr_t)&eshkol_coro_entry;   /* x20 */
    frame[10] = 0;                                        /* x29 */
    frame[11] = (uint64_t)(uintptr_t)&eshkol_coro_trampoline;  /* x30 */
#    endif
    return frame;
}

#  elif defined(ESHKOL_CORO_WINDOWS)

/* Windows: the system Fibers API owns stack layout, TEB stack limits and
 * structured-exception chains, so it is the switch primitive there. A fiber
 * whose coroutine finished parks in the pool and is handed the next
 * coroutine instead of being deleted. */
struct WinFiberBox {
    Coroutine* co;
};

VOID CALLBACK windows_fiber_proc(LPVOID param) {
    auto* box = static_cast<WinFiberBox*>(param);
    /* A pooled fiber handed a new coroutine resumes right after the previous
     * coroutine's final switch, returns here, and runs the next one. */
    for (;;) coroutine_run(box->co);
}

/* The thread's own context must itself be a fiber before SwitchToFiber. */
void windows_adopt_thread(ThreadState& ts) {
    if (ts.converted) return;
    ts.root.sp = IsThreadAFiber() ? GetCurrentFiber() : ConvertThreadToFiber(nullptr);
    ts.converted = true;
}

bool stack_acquire(ThreadState& ts, Stack* out) {
    windows_adopt_thread(ts);
    if (!ts.root.sp) return false;
    if (!ts.pool.empty()) {
        *out = ts.pool.back();
        ts.pool.pop_back();
        return true;
    }
    auto* box = static_cast<WinFiberBox*>(std::calloc(1, sizeof(WinFiberBox)));
    if (!box) return false;
    LPVOID fiber = CreateFiberEx(64 * 1024, configured_stack_bytes(),
                                 FIBER_FLAG_FLOAT_SWITCH, windows_fiber_proc, box);
    if (!fiber) {
        std::free(box);
        return false;
    }
    out->mapping = fiber;
    out->mapping_bytes = (size_t)(uintptr_t)box;   /* the box rides along */
    out->low = 0;      /* learned on first entry (GetCurrentThreadStackLimits) */
    out->high = 0;
    ts.mapped++;
    return true;
}

void stack_release(ThreadState& ts, Stack& s) {
    if (!s.mapping) return;
    if (ts.pool.size() < kStackPoolMax) {
        ts.pool.push_back(s);
    } else {
        DeleteFiber(s.mapping);
        std::free((void*)(uintptr_t)s.mapping_bytes);
    }
    s = Stack{};
}

void* stack_prepare(Coroutine* co) {
    reinterpret_cast<WinFiberBox*>((uintptr_t)co->stack.mapping_bytes)->co = co;
    return co->stack.mapping;
}
#  endif

/* Save the running context into `from`, load `to`, and switch machines.
 * `from_finished` marks a coroutine leaving for the last time: its dynamic
 * state is already empty and its stack must not be recorded by the sanitizer.
 * Raises (before changing anything) if `from`'s region slots cannot grow. */
void switch_context(ThreadState& ts, Ctx* from, Ctx* to, Coroutine* to_co,
                    bool from_finished) {
    /* The two saves that can need memory run first, so a failure raises
     * before anything has moved. Parameter bindings are lifted after the
     * region save succeeded; lifting itself cannot fail once room exists. */
    if (!from_finished && !eshkol_region_context_save(&from->exec)) {
        eshkol_raise_allocation_failure("coroutine switch", 0);
    }
    if (!from_finished && !eshkol_parameter_context_save(&from->exec)) {
        eshkol_raise_allocation_failure("coroutine switch", 0);
    }
    eshkol_dynamic_wind_context_save(&from->exec);
    eshkol_exception_context_save(&from->exec);
    eshkol_parallel_boundary_context_save(&from->exec);
    eshkol_resource_limits_context_save(&from->exec);

    eshkol_region_context_load(&to->exec);
    eshkol_parameter_context_load(&to->exec);
    eshkol_dynamic_wind_context_load(&to->exec);
    eshkol_exception_context_load(&to->exec);
    eshkol_parallel_boundary_context_load(&to->exec);
    eshkol_resource_limits_context_load(&to->exec);
    eshkol_stack_context_load(&to->exec);
    ts.current = to_co;

#  if defined(ESHKOL_CORO_TSAN)
    /* The thread's own fiber is named the first time it is left; every
     * coroutine gets its own when its stack is set up. */
    if (!from->tsan_fiber) from->tsan_fiber = __tsan_get_current_fiber();
    __tsan_switch_to_fiber(to->tsan_fiber, 0);
#  endif
#  if defined(ESHKOL_CORO_ASAN)
    __sanitizer_start_switch_fiber(from_finished ? nullptr : &from->asan_fake,
                                   to->asan_bottom, to->asan_size);
#  endif

#  if defined(ESHKOL_CORO_ASM)
    eshkol_coro_switch(&from->sp, to->sp);
#  elif defined(ESHKOL_CORO_WINDOWS)
    (void)from;
    SwitchToFiber(to->sp);
#  endif

#  if defined(ESHKOL_CORO_ASAN)
    /* Back on `from`'s stack. The thread's own stack bounds were recorded by
     * the first coroutine it entered (coroutine_run). */
    __sanitizer_finish_switch_fiber(from->asan_fake, nullptr, nullptr);
#  endif
}

Ctx* ctx_of(ThreadState& ts, Coroutine* co) { return co ? &co->ctx : &ts.root; }

/* Runs on the coroutine's own stack. Its final switch hands the stack back to
 * the resumer, which releases it; control returns from that switch only when a
 * pooled Windows fiber is handed its next coroutine. */
void coroutine_run(Coroutine* co) {
    ThreadState& ts = t_state;
#  if defined(ESHKOL_CORO_ASAN)
    {
        const void* old_bottom = nullptr;
        size_t old_size = 0;
        __sanitizer_finish_switch_fiber(nullptr, &old_bottom, &old_size);
        Ctx* resumer = ctx_of(ts, co->resumer);
        if (!resumer->asan_bottom) {
            resumer->asan_bottom = old_bottom;
            resumer->asan_size = old_size;
        }
    }
#  endif
#  if defined(ESHKOL_CORO_WINDOWS)
    {
        ULONG_PTR low = 0, high = 0;
        GetCurrentThreadStackLimits(&low, &high);
        co->stack.low = (uintptr_t)low;
        co->stack.high = (uintptr_t)high;
        co->ctx.exec.stack_low = co->stack.low;
        co->ctx.exec.stack_high = co->stack.high;
        eshkol_stack_context_load(&co->ctx.exec);
    }
#  endif

    eshkol_tagged_value_t result = tv_null();
    eshkol_boundary_raise_t raised;
    std::memset(&raised, 0, sizeof(raised));
    /* make-coroutine checked that a thunk worker is registered. */
    (void)eshkol_boundary_call_thunk(&co->thunk, &result, &raised);
    if (raised.pending) {
        co->status = ESHKOL_COROUTINE_FAILED;
        co->failure = raised;
        co->transfer = tv_null();
    } else {
        co->status = ESHKOL_COROUTINE_DONE;
        eshkol_region_write_barrier_into(&co->transfer, co, &result);
    }
    switch_context(ts, &co->ctx, ctx_of(ts, co->resumer), co->resumer, true);
}

#endif  /* !ESHKOL_CORO_UNAVAILABLE */

Coroutine* require_coroutine(const eshkol_tagged_value_t* v, const char* who) {
    if (!tv_is_int(v)) raise_error("%s: expected a coroutine", who);
    Coroutine* co = registry_get(v->data.int_val);
    if (!co) raise_error("%s: not a live coroutine (released or never created)", who);
    return co;
}

void require_switch_allowed(const char* who) {
    unsigned char ad_mode = 0;
    uint64_t ad_depth = 0;
    eshkol_ad_state_capture(&ad_mode, &ad_depth, nullptr, nullptr, nullptr);
    if (ad_mode || ad_depth) {
        raise_error("%s: a coroutine cannot switch inside a differentiation pass", who);
    }
}

const char* status_name(eshkol_coroutine_status_t s) {
    switch (s) {
        case ESHKOL_COROUTINE_SUSPENDED: return "suspended";
        case ESHKOL_COROUTINE_RUNNING:   return "running";
        case ESHKOL_COROUTINE_NORMAL:    return "normal";
        case ESHKOL_COROUTINE_DONE:      return "done";
        case ESHKOL_COROUTINE_FAILED:    return "failed";
    }
    return "failed";
}

void release_stack(ThreadState& ts, Coroutine* co) {
#if !defined(ESHKOL_CORO_UNAVAILABLE)
    if (co->stack.mapping) {
#  if defined(ESHKOL_CORO_TSAN)
        if (co->ctx.tsan_fiber) {
            __tsan_destroy_fiber(co->ctx.tsan_fiber);
            co->ctx.tsan_fiber = nullptr;
        }
#  endif
        stack_release(ts, co->stack);
        if (ts.live) ts.live--;
    }
#else
    (void)ts;
    (void)co;
#endif
    eshkol_exec_context_dispose(&co->ctx.exec);
}

/* Resume `co` from the running context, delivering `value` (or raising it at
 * the suspension point when `inject` is set). Returns what the coroutine
 * yields or returns; re-delivers its raise when it fails. */
eshkol_tagged_value_t resume(Coroutine* co, const eshkol_tagged_value_t* value,
                             bool inject, const char* who) {
#if defined(ESHKOL_CORO_UNAVAILABLE)
    (void)co; (void)value; (void)inject;
    raise_error("%s: coroutines are not available on this platform", who);
#else
    ThreadState& ts = t_state;
    if (co->owner_thread != ts.token) {
        raise_error("%s: a coroutine runs only on the thread that created it", who);
    }
    switch (co->status) {
        case ESHKOL_COROUTINE_SUSPENDED: break;
        case ESHKOL_COROUTINE_RUNNING:
            raise_error("%s: the coroutine is already running", who);
        case ESHKOL_COROUTINE_NORMAL:
            raise_error("%s: the coroutine is waiting on a coroutine it resumed", who);
        case ESHKOL_COROUTINE_DONE:
        case ESHKOL_COROUTINE_FAILED:
            raise_error("%s: the coroutine has finished", who);
    }
    require_switch_allowed(who);

    eshkol_tagged_value_t v = value ? *value : tv_null();
    if (inject && !co->started) {
        /* Raising into a coroutine that never ran: it finishes without
         * running, and the raise is delivered here. */
        co->status = ESHKOL_COROUTINE_FAILED;
        std::memset(&co->failure, 0, sizeof(co->failure));
        release_stack(ts, co);
        eshkol_raise_object(&v);
    }
    if (!co->started) {
        if (!stack_acquire(ts, &co->stack)) {
            raise_error("%s: could not allocate a coroutine stack", who);
        }
        co->ctx.sp = stack_prepare(co);
        co->ctx.exec.stack_low = co->stack.low;
        co->ctx.exec.stack_high = co->stack.high;
        co->ctx.exec.context_id =
            g_next_context_id.fetch_add(1, std::memory_order_relaxed);
#  if defined(ESHKOL_CORO_ASAN)
        co->ctx.asan_bottom = (const void*)co->stack.low;
        co->ctx.asan_size = (size_t)(co->stack.high - co->stack.low);
#  endif
#  if defined(ESHKOL_CORO_TSAN)
        co->ctx.tsan_fiber = __tsan_create_fiber(0);
#  endif
        co->started = true;
        ts.live++;
        if (ts.live > ts.peak_live) ts.peak_live = ts.live;
    }

    eshkol_region_write_barrier_into(&co->transfer, co, &v);
    co->inject_raise = inject;
    Coroutine* resumer = ts.current;
    co->resumer = resumer;
    if (resumer) resumer->status = ESHKOL_COROUTINE_NORMAL;
    co->status = ESHKOL_COROUTINE_RUNNING;

    switch_context(ts, ctx_of(ts, resumer), &co->ctx, co, false);

    /* Back in the resumer: `co` yielded, returned or raised. */
    if (resumer) resumer->status = ESHKOL_COROUTINE_RUNNING;
    eshkol_tagged_value_t out = co->transfer;
    if (co->status == ESHKOL_COROUTINE_DONE) {
        release_stack(ts, co);
    } else if (co->status == ESHKOL_COROUTINE_FAILED) {
        release_stack(ts, co);
        eshkol_boundary_raise_t failure = co->failure;
        eshkol_boundary_redeliver(&failure);
    }
    return out;
#endif
}

}  // namespace

#if !defined(ESHKOL_CORO_UNAVAILABLE)
ThreadState::~ThreadState() {
    for (Stack& s : pool) {
#  if defined(ESHKOL_CORO_ASM)
        munmap(s.mapping, s.mapping_bytes);
#  elif defined(ESHKOL_CORO_WINDOWS)
        DeleteFiber(s.mapping);
        std::free((void*)(uintptr_t)s.mapping_bytes);
#  endif
    }
    pool.clear();
    eshkol_exec_context_dispose(&root.exec);
}
#else
ThreadState::~ThreadState() {}
#endif

/* ── C API ─────────────────────────────────────────────────────────────── */

extern "C" int eshkol_coroutines_available(void) {
#if defined(ESHKOL_CORO_UNAVAILABLE)
    return 0;
#else
    return 1;
#endif
}

extern "C" void eshkol_coroutine_stats(eshkol_coroutine_stats_t* out) {
    if (!out) return;
    ThreadState& ts = t_state;
    out->live = ts.live;
    out->peak_live = ts.peak_live;
    out->created = ts.created;
    out->stacks_mapped = ts.mapped;
    out->stacks_pooled = ts.pool.size();
    out->stack_bytes = configured_stack_bytes();
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    out->handles_open = g_handles_open;
}

/* ── Builtins ──────────────────────────────────────────────────────────── */

extern "C" void eshkol_builtin_make_coroutine(eshkol_tagged_value_t* out,
                                              const eshkol_tagged_value_t* thunk) {
    if (!thunk || thunk->type != ESHKOL_VALUE_CALLABLE) {
        raise_error("%s: expected a procedure of no arguments", "make-coroutine");
    }
#if defined(ESHKOL_CORO_UNAVAILABLE)
    raise_error("%s: coroutines are not available on this platform", "make-coroutine");
#endif
    if (!eshkol_boundary_thunks_available()) {
        raise_error("%s: this program has no compiled procedure worker registered",
                    "make-coroutine");
    }
    auto* co = new (std::nothrow) Coroutine();
    if (!co) eshkol_raise_allocation_failure("make-coroutine", sizeof(Coroutine));
    co->owner_thread = t_state.token;
    co->thunk = tv_null();
    co->transfer = tv_null();
    std::memset(&co->failure, 0, sizeof(co->failure));
    /* The procedure is held by a malloc'd record for as long as the coroutine
     * lives, so it is promoted out of every open region first. */
    eshkol_region_write_barrier_into(&co->thunk, co, thunk);
    const int64_t handle = registry_add(co);
    if (handle < 0) {
        delete co;
        raise_error("%s: too many live coroutines", "make-coroutine");
    }
    co->handle = handle;
    t_state.created++;
    *out = tv_int(handle);
}

extern "C" void eshkol_builtin_coroutine_resume(eshkol_tagged_value_t* out,
                                                const eshkol_tagged_value_t* co_val,
                                                const eshkol_tagged_value_t* value) {
    Coroutine* co = require_coroutine(co_val, "coroutine-resume");
    *out = resume(co, value, false, "coroutine-resume");
}

extern "C" void eshkol_builtin_coroutine_raise(eshkol_tagged_value_t* out,
                                               const eshkol_tagged_value_t* co_val,
                                               const eshkol_tagged_value_t* obj) {
    Coroutine* co = require_coroutine(co_val, "coroutine-raise");
    *out = resume(co, obj, true, "coroutine-raise");
}

extern "C" void eshkol_builtin_coroutine_yield(eshkol_tagged_value_t* out,
                                               const eshkol_tagged_value_t* value) {
#if defined(ESHKOL_CORO_UNAVAILABLE)
    (void)out; (void)value;
    raise_error("%s: not inside a coroutine", "coroutine-yield");
#else
    ThreadState& ts = t_state;
    Coroutine* co = ts.current;
    if (!co) raise_error("%s: not inside a coroutine", "coroutine-yield");
    require_switch_allowed("coroutine-yield");
    eshkol_tagged_value_t v = value ? *value : tv_null();
    /* The value outlives every region this coroutine has open. */
    eshkol_region_write_barrier_into(&co->transfer, co, &v);
    co->status = ESHKOL_COROUTINE_SUSPENDED;
    Coroutine* resumer = co->resumer;
    switch_context(ts, &co->ctx, ctx_of(ts, resumer), resumer, false);

    /* Resumed. */
    if (co->inject_raise) {
        co->inject_raise = false;
        eshkol_tagged_value_t obj = co->transfer;
        eshkol_raise_object(&obj);
    }
    *out = co->transfer;
#endif
}

extern "C" void eshkol_builtin_coroutine_status(eshkol_tagged_value_t* out,
                                                const eshkol_tagged_value_t* co_val) {
    Coroutine* co = require_coroutine(co_val, "coroutine-status");
    *out = tv_symbol(status_name(co->status));
}

extern "C" void eshkol_builtin_coroutine_release(eshkol_tagged_value_t* out,
                                                 const eshkol_tagged_value_t* co_val) {
    if (!tv_is_int(co_val)) raise_error("%s: expected a coroutine", "coroutine-release!");
    Coroutine* co = registry_get(co_val->data.int_val);
    if (!co) {
        /* Released already: observable, not fatal (the handle-lifecycle rule
         * event-loop-close follows). */
        *out = tv_bool(false);
        return;
    }
    if (co->status == ESHKOL_COROUTINE_RUNNING || co->status == ESHKOL_COROUTINE_NORMAL) {
        raise_error("%s: the coroutine is running", "coroutine-release!");
    }
    if (co->status == ESHKOL_COROUTINE_SUSPENDED && co->started) {
        /* Its stack still holds frames whose dynamic-wind after thunks have
         * not run. Releasing it would skip them; finishing it is the
         * owner's job (coroutine-raise delivers a cancellation). */
        raise_error("%s: the coroutine is suspended mid-body; finish it first "
                    "(coroutine-raise delivers a cancellation)",
                    "coroutine-release!");
    }
    if (co->owner_thread != t_state.token && co->stack.mapping) {
        raise_error("%s: a coroutine is released by the thread that created it",
                    "coroutine-release!");
    }
    registry_remove(co->handle);
    release_stack(t_state, co);
    delete co;
    *out = tv_bool(true);
}

extern "C" void eshkol_builtin_current_coroutine(eshkol_tagged_value_t* out) {
    Coroutine* co = t_state.current;
    *out = co ? tv_int(co->handle) : tv_bool(false);
}

extern "C" void eshkol_builtin_coroutine_p(eshkol_tagged_value_t* out,
                                           const eshkol_tagged_value_t* value) {
    *out = tv_bool(tv_is_int(value) && registry_get(value->data.int_val) != nullptr);
}

extern "C" void eshkol_builtin_coroutine_stats(eshkol_tagged_value_t* out) {
    eshkol_coroutine_stats_t s;
    eshkol_coroutine_stats(&s);
    const struct { const char* key; uint64_t value; } rows[] = {
        {"live", s.live},
        {"peak-live", s.peak_live},
        {"created", s.created},
        {"stacks-mapped", s.stacks_mapped},
        {"stacks-pooled", s.stacks_pooled},
        {"stack-bytes", s.stack_bytes},
        {"handles-open", s.handles_open},
    };
    eshkol_tagged_value_t list = tv_null();
    for (size_t i = sizeof(rows) / sizeof(rows[0]); i-- > 0;) {
        list = tv_pair(tv_pair(tv_symbol(rows[i].key), tv_int((int64_t)rows[i].value)), list);
    }
    *out = list;
}
