#include "eshkol/backend/vm_limits.h"

#ifndef ESHKOL_VM_NATIVE_POLICY_DESKTOP
#define ESHKOL_VM_NATIVE_POLICY_DESKTOP 0
#endif

/*******************************************************************************
 * Instruction Set
 ******************************************************************************/

typedef enum {
    /* Constants & Stack */
    OP_NOP = 0,
    OP_CONST = 1,       /* operand = constant pool index */
    OP_NIL = 2,
    OP_TRUE = 3,
    OP_FALSE = 4,
    OP_POP = 5,
    OP_DUP = 6,

    /* Arithmetic */
    OP_ADD = 7,
    OP_SUB = 8,
    OP_MUL = 9,
    OP_DIV = 10,
    OP_MOD = 11,
    OP_NEG = 12,
    OP_ABS = 13,

    /* Comparison (push boolean) */
    OP_EQ = 14,
    OP_LT = 15,
    OP_GT = 16,
    OP_LE = 17,
    OP_GE = 18,
    OP_NOT = 19,

    /* Variables */
    OP_GET_LOCAL = 20,   /* operand = slot offset from FP */
    OP_SET_LOCAL = 21,
    OP_GET_UPVALUE = 22, /* operand = upvalue index */
    OP_SET_UPVALUE = 23,

    /* Functions */
    OP_CLOSURE = 24,     /* operand = func constant index */
    OP_CALL = 25,        /* operand = argument count */
    OP_TAIL_CALL = 26,
    OP_RETURN = 27,

    /* Control Flow */
    OP_JUMP = 28,        /* operand = absolute target */
    OP_JUMP_IF_FALSE = 29,
    OP_LOOP = 30,        /* operand = backward target */

    /* Pairs & Lists */
    OP_CONS = 31,
    OP_CAR = 32,
    OP_CDR = 33,
    OP_NULL_P = 34,

    /* I/O */
    OP_PRINT = 35,
    OP_HALT = 36,
    OP_NATIVE_CALL = 37, /* operand = native function ID */

    OP_CLOSE_UPVALUE = 38,
    /* Vectors */
    OP_VEC_CREATE = 39,   /* operand = count; pops count values, creates vector */
    OP_VEC_REF = 40,      /* TOS=index, SOS=vector -> push vector[index] */
    OP_VEC_SET = 41,      /* TOS=value, SOS=index, TOS-2=vector -> set */
    OP_VEC_LEN = 42,      /* TOS=vector -> push length */
    /* Strings */
    OP_STR_REF = 43,      /* TOS=index, SOS=string -> push char */
    OP_STR_LEN = 44,      /* TOS=string -> push length */
    /* Type checks */
    OP_PAIR_P = 45,       /* TOS -> push (pair? TOS) */
    OP_NUM_P = 46,        /* TOS -> push (number? TOS) */
    OP_STR_P = 47,        /* TOS -> push (string? TOS) */
    OP_BOOL_P = 48,       /* TOS -> push (boolean? TOS) */
    OP_PROC_P = 49,       /* TOS -> push (procedure? TOS) */
    OP_VEC_P = 50,        /* TOS -> push (vector? TOS) */
    /* Set mutations */
    OP_SET_CAR = 51,      /* TOS=val, SOS=pair -> set car */
    OP_SET_CDR = 52,      /* TOS=val, SOS=pair -> set cdr */
    OP_POPN = 53,         /* operand=N: pop N values below TOS, keeping TOS */
    OP_OPEN_CLOSURE = 54,
    OP_CALLCC = 55,       /* call/cc: capture continuation, call TOS with it */
    OP_INVOKE_CC = 56,    /* invoke a captured continuation with a value */
    OP_PUSH_HANDLER = 57, /* operand=handler_pc: save continuation, push exception handler */
    OP_POP_HANDLER = 58,  /* remove topmost exception handler */
    OP_GET_EXN = 59,      /* push current exception value */
    OP_PACK_REST = 60,    /* operand=n_fixed: pack args into list */
    OP_WIND_PUSH = 61,    /* push after thunk onto wind stack */
    OP_WIND_POP = 62,     /* pop from wind stack */
    OP_VOID = 63,         /* push unspecified void value (return of display/newline) */

    /* Opt-in metadata immediately preceding OP_NATIVE_CALL. Its operand is a
     * BUILTINS[] index. The native-call handler validates that the named
     * builtin and native ID agree before emitting exact VM dispatch evidence.
     * Normal compilation never emits this opcode. */
    OP_LANGUAGE_COVERAGE = 64,
    /* Stable-hash marker immediately preceding a direct Scheme CALL/TCALL. */
    OP_LANGUAGE_COVERAGE_CALL = 65,

    /* operand = number of top-level binding slots established so far.
     * Emitted once after each top-level form by compile_and_run(). Raises
     * vm->global_top, the STORE/CONTROL boundary a continuation restore must
     * not roll back across (see vm_restore_continuation() in vm_run.c). The
     * VM binds every top-level define to a stack slot, so without this marker
     * a continuation's stack snapshot restores the *store* along with the
     * control state and `set!` mutations are silently undone (SW-52). */
    OP_GLOBAL_MARK = 66,

    OP_COUNT = 67
} OpCode;

typedef struct { uint8_t op; int32_t operand; } Instr;

/*******************************************************************************
 * Value Representation (tagged values)
 ******************************************************************************/

typedef enum {
    VAL_NIL = 0,
    VAL_INT = 1,
    VAL_FLOAT = 2,
    VAL_BOOL = 3,
    VAL_PAIR = 4,       /* heap pointer to cons cell */
    VAL_CLOSURE = 5,    /* heap pointer to closure */
    VAL_STRING = 6,     /* heap pointer to string */
    VAL_VECTOR = 7,     /* heap pointer to vector */
    VAL_CONTINUATION = 15, /* heap pointer to saved continuation */
} ValType;

typedef struct {
    ValType type;
    union {
        int64_t i;
        double  f;
        int     b;       /* boolean */
        int32_t ptr;     /* heap pointer (index into heap array) */
    } as;
} Value;

#define NIL_VAL    ((Value){.type = VAL_NIL})
#define INT_VAL(v) ((Value){.type = VAL_INT, .as.i = (v)})
#define FLOAT_VAL(v) ((Value){.type = VAL_FLOAT, .as.f = (v)})
#define BOOL_VAL(v) ((Value){.type = VAL_BOOL, .as.b = (v)})
#define PAIR_VAL(p) ((Value){.type = VAL_PAIR, .as.ptr = (p)})
#define CLOSURE_VAL(p) ((Value){.type = VAL_CLOSURE, .as.ptr = (p)})

/** @brief R7RS truthiness: only `#f` is false, everything else — including
 *         '(), 0 and "" — is truthy.
 *
 *  The `VAL_NIL` line below used to return 0, contradicting this very
 *  comment: `(if '() 'T 'F)` was `F` on the VM and `T` natively, and with it
 *  `not`/`and`/`or`/`cond`/`when`/`unless` and every `assq`/`memq`-style
 *  "did it find anything" idiom inverted.  R7RS 6.3.1 is unambiguous, and
 *  the native engine has always been right, so this is the VM's bug. */
static int is_truthy(Value v) {
    if (v.type == VAL_BOOL) return v.as.b;
    return 1;  /* everything that is not #f is true */
}

/** @brief Coerce a plain (non-heap-boxed) numeric Value to a double; 0.0
 *         for non-numeric types. See as_number_vm() for the heap-aware
 *         version that also handles rationals/bignums. */
static double as_number(Value v) {
    if (v.type == VAL_INT) return (double)v.as.i;
    if (v.type == VAL_FLOAT) return v.as.f;
    if (v.type == VAL_CHAR) return (double)v.as.i; /* codepoint (char->integer, char comparisons) */
    return 0.0;
}

/* as_number_vm defined after VM struct (needs heap access for rationals) */

/** @brief Wrap a double as an INT Value if it's an exact, small
 *         (< 1e15 in magnitude) integer, else as a FLOAT Value.
 *
 * This classifies by the result's VALUE SHAPE and therefore may only be used
 * where the operation is known to be in the EXACT domain (an exact-integer
 * fold, `inexact->exact`, an exact-integer builtin).  For anything derived
 * from operands whose exactness must be respected, use the tag-driven
 * number_val_contagious() / number_val_contagious1() below — see the comment
 * there for the class of silent wrong answers this distinction prevents. */
static Value number_val(double d) {
    /* A negative zero must stay INEXACT.  IEEE-754 makes (* -1.0 0.0) = -0.0,
     * but the integer collapse below treats -0.0 == 0 and turned it into the
     * exact integer 0, discarding the sign bit — so the VM printed 0 where
     * native prints -0, and a printed -0.0 could not round-trip. */
    if (d == 0.0 && signbit(d)) return FLOAT_VAL(d);
    if (d == (int64_t)d && fabs(d) < 1e15) return INT_VAL((int64_t)d);
    return FLOAT_VAL(d);
}

/** @brief Does @p v carry the INEXACT runtime tag?
 *
 * VAL_FLOAT is the VM's only inexact representation; VAL_INT, VAL_BIGNUM,
 * VAL_RATIONAL, VAL_I128 and VAL_CHAR are all exact.  Exactness is a property
 * of the operand's TAG, never of a result's value shape. */
static inline int vm_is_inexact_tag(Value v) { return v.type == VAL_FLOAT; }

/** @brief Wrap the double result of a BINARY numeric operation, preserving
 *         R7RS inexact contagion (R7RS 6.2.2: an operation with any inexact
 *         argument yields an inexact result).
 *
 * Classifying the result by value shape alone (bare number_val) collapsed an
 * integral-valued flonum result to the EXACT integer, which then re-entered
 * dispatch in the wrong numeric domain:
 *
 *     (- 2.0 1.0)              → exact 1  (must be inexact 1.0)
 *     (/ (- 2.0 1.0) (+ 2.0 1.0))
 *                              → 1/3      (must be 0.3333333333333333)
 *
 * The exactness of a flonum-domain result is decided HERE, by the operand
 * tags, so no downstream dispatch can mistake it for an exact integer. */
static inline Value number_val_contagious(Value a, Value b, double d) {
    if (vm_is_inexact_tag(a) || vm_is_inexact_tag(b)) return FLOAT_VAL(d);
    return number_val(d);
}

/** @brief Unary form of number_val_contagious(): (abs -2.0), (- 2.0),
 *         (floor 2.5) and (max 1.0 2.0) are inexact because their argument
 *         is, not because the value happens to be non-integral. */
static inline Value number_val_contagious1(Value a, double d) {
    if (vm_is_inexact_tag(a)) return FLOAT_VAL(d);
    return number_val(d);
}

/*******************************************************************************
 * Heap (arena-based, OALR)
 ******************************************************************************/

typedef enum {
    HEAP_CONS = 0,
    HEAP_CLOSURE = 1,
    HEAP_STRING = 2,
    HEAP_VECTOR = 3,
    HEAP_MULTI_VALUE = 4,
    HEAP_COMPLEX = 5,
    HEAP_RATIONAL = 6,
    HEAP_BIGNUM = 7,
    HEAP_DUAL = 8,
    HEAP_TENSOR = 9,
    HEAP_LOGIC_VAR = 10,
    HEAP_SUBST = 11,
    HEAP_FACT = 12,
    HEAP_KB = 13,
    HEAP_FACTOR_GRAPH = 14,
    HEAP_WORKSPACE = 15,
    HEAP_PORT = 16,
    HEAP_AD_TAPE = 17,
    HEAP_PROMISE = 18,
    HEAP_CONTINUATION = 19,
    HEAP_HASH = 20,
    HEAP_ERROR = 21,
    HEAP_BYTEVECTOR = 22,
    HEAP_PARAMETER = 23,
    HEAP_HYPER_DUAL = 24,
    HEAP_RIEMANNIAN_ADAM_STATE = 25,
    HEAP_FUTURE = 26,
    HEAP_I128 = 27,
} HeapType;

typedef struct {
    HeapType type;
    union {
        struct { Value car; Value cdr; } cons;
        struct {
            int32_t func_pc;
            /* Declared fixed-argument arity of the function, packed into the
             * high bits of the func-PC constant at compile time and unpacked by
             * OP_CLOSURE — so it survives ESKB serialization (the entry table's
             * offsets don't, since bodies are re-laid-out on load).  -1 means
             * unknown (an anonymous/synthesized closure); a variadic function
             * records 255.  Read via vm_closure_arity() so `gradient` can
             * expand a point to a callable's true signature. */
            int32_t arity;
            int32_t n_upvalues;
            /* Capacity MUST equal the compiler's MAX_UPVALUES (both are
             * ESHKOL_VM_MAX_CLOSURE_UPVALUES, see vm_limits.h) — a closure
             * whose upvalue count the compiler allowed but this array
             * couldn't hold is exactly the defect that let a large
             * procedure's OP_CLOSURE silently strand values on the operand
             * stack and corrupt whatever top-level `define` compiled next. */
            Value upvalues[ESHKOL_VM_MAX_CLOSURE_UPVALUES];
            /* -1 means closed/captured-by-value; otherwise this is an
             * absolute VM stack slot shared by every closure that captures
             * the same live top-level binding. */
            int32_t open_slots[ESHKOL_VM_MAX_CLOSURE_UPVALUES];
        } closure;
        struct { void* ptr; int subtype; } opaque;  /* for complex, rational, tensor, logic, etc. */
    };
} HeapObject;

/** @brief Per-open-region bookkeeping for the Stage-1 region evacuator.
 *
 * `slots` records, in allocation order, every heap-object index handed out
 * while this region was the innermost open one.  It is the ONLY authority on
 * region membership: `next_free` cannot be used as a base pointer once the
 * evacuator recycles index-space, because a later region can be handed an
 * index below its own start.
 *
 * `pinned` marks a region that must NOT be reclaimed — a region that allocated
 * an object of a subtype the evacuator does not deep-walk, or one that had a
 * continuation captured inside it.  A pinned region promotes wholesale into its
 * parent (exactly the pre-evacuator behaviour) instead of freeing anything, so
 * an uncovered case degrades to a leak and never to a dangling index.
 */
typedef struct {
    int32_t* slots;
    int32_t  n_slots;
    int32_t  cap_slots;
    int      pinned;
    const char* pin_reason;
} VmHeapRegionSlots;

typedef struct {
    VmRegionStack regions;
    HeapObject** objects;    /* array of pointers to arena-allocated objects */
    int32_t next_free;
    int32_t capacity;
    /* SW-14 growth watchdog. Before the Stage-1 evacuator the VM heap had no
     * reclamation of any kind and this watchdog was the whole story; it is kept
     * because `with-region` is opt-in and a workload that never opens a region
     * still grows monotonically. These two fields turn that growth into a NAMED
     * diagnostic at a budget. */
    uint32_t alloc_tick;         /* allocations since the last budget probe */
    int      budget_reported;    /* the budget diagnostic is emitted once */

    /* ── Stage-1 region evacuator state (SW-14 close) ────────────────────── */
    VmHeapRegionSlots region_slots[VM_ARENA_MAX_REGIONS];
    unsigned char* markbits;     /* 1 bit per object index; grown with `objects` */
    int32_t markbits_cap;        /* indices covered by `markbits` */
    int32_t* mark_stack;         /* worklist of object indices during a mark   */
    int32_t  mark_stack_n;
    int32_t  mark_stack_cap;
    int32_t* free_slots;         /* recycled object-table indices              */
    int32_t  n_free_slots;
    int32_t  cap_free_slots;
    /* Cumulative evacuator metrics, reported by `(vm-region-stats)`-style
     * diagnostics and asserted by tests/memory/vm_region_evacuator_test.sh. */
    uint64_t regions_reclaimed;
    uint64_t regions_pinned;
    uint64_t objects_reclaimed;
    uint64_t objects_promoted;
    uint64_t bytes_reclaimed;
    uint64_t bytes_promoted;
} Heap;

/** @return the VM arena budget in bytes past which the growth watchdog speaks,
 *          or 0 when the watchdog is disabled.
 *
 * `ESHKOL_VM_HEAP_BUDGET_MB` overrides the default; `0` disables the watchdog
 * entirely. The default is deliberately far above any test or ordinary
 * program, so the diagnostic means "this workload really is growing without
 * bound", never "this program is big".
 */
static size_t vm_heap_budget_bytes(void) {
    static size_t cached = (size_t)-1;
    if (cached != (size_t)-1) return cached;
    cached = (size_t)vm_host_env_long("ESHKOL_VM_HEAP_BUDGET_MB", 1024)
             * 1024u * 1024u;
    return cached;
}

/** @return 1 when crossing the budget must be fatal rather than advisory
 *          (`ESHKOL_VM_HEAP_BUDGET_FATAL=1`), so a CI lane can gate on it. */
static int vm_heap_budget_fatal(void) {
    return vm_host_env_flag("ESHKOL_VM_HEAP_BUDGET_FATAL");
}

/** @brief Initialize the VM heap: sets up its arena region stack and the
 *         object-pointer table (fixed capacity HEAP_SIZE). */
static void heap_init(Heap* h) {
    vm_region_stack_init(&h->regions);
    h->capacity = HEAP_SIZE;
    h->objects = (HeapObject**)calloc(h->capacity, sizeof(HeapObject*));
    h->next_free = 0;
    h->alloc_tick = 0;
    h->budget_reported = 0;
    memset(h->region_slots, 0, sizeof(h->region_slots));
    h->markbits = NULL;
    h->markbits_cap = 0;
    h->mark_stack = NULL;
    h->mark_stack_n = 0;
    h->mark_stack_cap = 0;
    h->free_slots = NULL;
    h->n_free_slots = 0;
    h->cap_free_slots = 0;
    h->regions_reclaimed = 0;
    h->regions_pinned = 0;
    h->objects_reclaimed = 0;
    h->objects_promoted = 0;
    h->bytes_reclaimed = 0;
    h->bytes_promoted = 0;
}

/** @return total bytes the VM's arenas hold: the global arena plus every open
 *          region's arena. This is the VM's real memory footprint: the Stage-1
 *          evacuator returns a popped region's dead blocks to the allocator,
 *          but the global arena itself is never released. */
static size_t heap_arena_bytes(const Heap* h) {
    size_t total = h->regions.global_arena.total_allocated;
    for (int i = 0; i < h->regions.depth; i++)
        if (h->regions.stack[i]) total += h->regions.stack[i]->arena.total_allocated;
    return total;
}

/**
 * @brief Growth watchdog: name the VM's unbounded heap growth once it crosses
 *        the configured budget, instead of letting it stay silent.
 *
 * This outlived the SW-14 close. `with-region` reclaims now, but OUTSIDE a
 * region the VM heap has no reclamation of any kind — no collector, no
 * per-loop nursery — so a workload that never opens a region still grows
 * monotonically, and should be told so.
 *
 * Sampled every 4096 allocations so the per-allocation cost is a counter
 * increment and a compare. The message is deliberately specific: it names the
 * mechanism that DOES reclaim on this substrate, so a reader who is growing
 * without bound is told what to reach for rather than left to rediscover it.
 */
static void heap_check_budget(Heap* h) {
    if (h->budget_reported) return;
    if (++h->alloc_tick < 4096u) return;
    h->alloc_tick = 0;
    size_t budget = vm_heap_budget_bytes();
    if (budget == 0) return;
    size_t used = heap_arena_bytes(h);
    if (used < budget) return;
    h->budget_reported = 1;
    fprintf(stderr,
            "eshkol-vm: heap budget exceeded — %.1f MB of arena allocated "
            "(budget %.0f MB, ESHKOL_VM_HEAP_BUDGET_MB).\n"
            "  Outside a region the bytecode VM does not reclaim heap memory: the "
            "global arena grows monotonically. `(with-region ...)` is the "
            "mechanism that returns memory here — it reclaims as of the Stage-1 "
            "evacuator (docs/reference/runtime/memory-model.md).\n"
            "  Wrap the allocating step in `(with-region ('step SIZE) ...)`, or "
            "set ESHKOL_VM_HEAP_BUDGET_MB=0 to silence this, or "
            "ESHKOL_VM_HEAP_BUDGET_FATAL=1 to make it fail closed.\n",
            (double)used / (1024.0 * 1024.0),
            (double)budget / (1024.0 * 1024.0));
    if (vm_heap_budget_fatal()) {
        fprintf(stderr, "eshkol-vm: ERROR: heap budget is fatal "
                        "(ESHKOL_VM_HEAP_BUDGET_FATAL=1); terminating.\n");
        exit(1);
    }
}

/**
 * @brief One-time notice that the region HANDLE surface on the bytecode VM
 *        reclaims nothing (Stage-2).
 *
 * `region-open` and `region-close` resolve and behave identically on both
 * substrates — same handle protocol, same validation, same error messages —
 * but on the VM a close reclaims no VM heap. `with-region` DOES reclaim here
 * as of the Stage-1 evacuator (lib/backend/vm_region_evac.c), which is why
 * this notice fires only on the handle surface: a user who reached for the
 * memory tool and silently got no memory back is the thing worth saying, and
 * saying it about `with-region` would now be false. Emitted at most once per
 * process, on stderr, and suppressed by `ESHKOL_VM_REGION_QUIET=1`.
 */
static void vm_region_reclaim_notice(void) {
    static int said = 0;
    if (said) return;
    said = 1;
    if (vm_host_env_flag("ESHKOL_VM_REGION_QUIET")) return;
    fprintf(stderr,
            "eshkol-vm: note: the region HANDLE surface (region-open / "
            "region-close) is bookkeeping-only on the bytecode VM — the handle "
            "protocol, its validation and its errors are identical to native, "
            "but a close reclaims no VM heap. `(with-region ...)` DOES reclaim "
            "here as of the Stage-1 evacuator; the handle surface is Stage-2 "
            "(docs/reference/runtime/memory-model.md). Set "
            "ESHKOL_VM_REGION_QUIET=1 to silence this note.\n");
}

/** @brief Record heap index @p slot as belonging to the innermost open region.
 *
 * Membership is recorded explicitly rather than inferred from an index range,
 * because index recycling means a region's slots are not contiguous and not
 * necessarily above the region's start. On allocation failure the region is
 * PINNED: an incomplete membership list must never let the evacuator conclude
 * that an object it forgot about is dead.
 */
static void heap_region_record_slot(Heap* h, int32_t slot) {
    int d = h->regions.depth - 1;
    if (d < 0 || d >= VM_ARENA_MAX_REGIONS) return;
    VmHeapRegionSlots* rs = &h->region_slots[d];
    if (rs->n_slots >= rs->cap_slots) {
        int32_t cap = rs->cap_slots > 0 ? rs->cap_slots * 2 : 256;
        int32_t* grown = (int32_t*)realloc(rs->slots, (size_t)cap * sizeof(int32_t));
        if (!grown) {
            rs->pinned = 1;
            rs->pin_reason = "region membership list could not grow";
            return;
        }
        rs->slots = grown;
        rs->cap_slots = cap;
    }
    rs->slots[rs->n_slots++] = slot;
}

/** @brief Allocate a new (zeroed) HeapObject slot from the arena and
 *         register it in the object table.
 * @return The new object's heap index, or -1 on capacity/allocation
 *         failure.
 */
static int32_t heap_alloc(Heap* h) {
    if (h->n_free_slots == 0 && h->next_free >= h->capacity) {
        /* The object table is a growable pointer array, not a fixed pool.
         * It used to be sized once at HEAP_SIZE, which turned any workload
         * whose live-object count exceeded that (e.g. an N-parameter
         * forward-mode gradient, which boxes N duals per pass for N passes)
         * into a hard "HEAP OVERFLOW" instead of a slower-but-correct run.
         * Grow geometrically up to ESHKOL_VM_HEAP_MAX_SIZE, which is the one
         * remaining bound and is reported by name when it is reached. */
        if (h->capacity >= ESHKOL_VM_HEAP_MAX_SIZE) {
            fprintf(stderr, "ERROR: VM heap object limit reached "
                            "(ESHKOL_VM_HEAP_MAX_SIZE=%d objects)\n",
                    (int)ESHKOL_VM_HEAP_MAX_SIZE);
            return -1;
        }
        int32_t new_cap = (h->capacity > 0) ? h->capacity * 2 : 1024;
        if (new_cap > ESHKOL_VM_HEAP_MAX_SIZE || new_cap < 0)
            new_cap = ESHKOL_VM_HEAP_MAX_SIZE;
        HeapObject** grown =
            (HeapObject**)realloc(h->objects, (size_t)new_cap * sizeof(HeapObject*));
        if (!grown) {
            fprintf(stderr, "ERROR: VM heap object table growth to %d objects failed\n",
                    (int)new_cap);
            return -1;
        }
        memset(grown + h->capacity, 0,
               (size_t)(new_cap - h->capacity) * sizeof(HeapObject*));
        h->objects = grown;
        h->capacity = new_cap;
    }
    HeapObject* obj = (HeapObject*)vm_alloc(&h->regions, sizeof(HeapObject));
    if (!obj) { fprintf(stderr, "ARENA OOM\n"); return -1; }
    memset(obj, 0, sizeof(HeapObject));

    /* Prefer a slot recycled by a previous region pop over growing the table;
     * the index space is a monotone counter otherwise, and a `with-region`
     * loop that reclaimed its arena but not its indices would still climb
     * (ledger SW-14, item 4 of scope_of_the_real_fix). Recycling is disabled
     * under ESHKOL_VM_REGION_RECYCLE=0 and under poison mode, where a freed
     * slot is deliberately left NULL forever so a missed reference faults
     * instead of aliasing a fresh object. */
    int32_t slot;
    if (h->n_free_slots > 0) {
        slot = h->free_slots[--h->n_free_slots];
    } else {
        slot = h->next_free++;
    }
    h->objects[slot] = obj;
    if (h->regions.depth > 0) heap_region_record_slot(h, slot);
    heap_check_budget(h);
    return slot;
}

/** @brief Push a new arena region scope onto the heap (see OALR — objects
 *         allocated after this call are freed in bulk by the matching
 *         heap_region_pop(), minus whatever escapes). */
static int heap_region_push(Heap* h, const char* name, size_t size_hint) {
    if (!vm_region_push(&h->regions, name, size_hint)) return 0;
    int d = h->regions.depth - 1;
    if (d >= 0 && d < VM_ARENA_MAX_REGIONS) {
        h->region_slots[d].n_slots = 0;
        h->region_slots[d].pinned = 0;
        h->region_slots[d].pin_reason = NULL;
    }
    return 1;
}

/** @brief Mark the innermost open region as un-reclaimable, naming why.
 *
 * A pinned region promotes wholesale into its parent on pop: nothing is freed
 * and no object slot is retired. Every case the evacuator is not certain it can
 * walk goes through here, so an uncovered case degrades to the pre-evacuator
 * leak instead of to a dangling heap index. */
static void heap_region_pin(Heap* h, const char* reason) {
    int d = h->regions.depth - 1;
    if (d < 0 || d >= VM_ARENA_MAX_REGIONS) return;
    if (h->region_slots[d].pinned) return;
    h->region_slots[d].pinned = 1;
    h->region_slots[d].pin_reason = reason;
}

/** @brief Pin every currently-open region (used when a captured continuation
 *         could resurrect a region body that a pop would otherwise free). */
static void heap_region_pin_all(Heap* h, const char* reason) {
    for (int d = 0; d < h->regions.depth && d < VM_ARENA_MAX_REGIONS; d++) {
        if (!h->region_slots[d].pinned) {
            h->region_slots[d].pinned = 1;
            h->region_slots[d].pin_reason = reason;
        }
    }
}

/* Defined in vm_region_evac.c (after every heap payload type is in scope).
 *
 * vm_region_evacuate_pop() pops the innermost region, promoting whatever is
 * still reachable and returning the rest to the allocator.
 *
 * vm_region_bracket_unwind_to() is the SINGLE teardown entry point, the VM
 * counterpart of native's eshkol_region_unwind_to(): normal `with-region`
 * exit, a raise crossing a region, and a continuation transfer all go through
 * it, so the structured and unstructured paths cannot drift apart. */
static void vm_region_evacuate_pop(VM* vm);
static void vm_region_bracket_unwind_to(VM* vm, int target_brackets);
static void vm_region_bracket_unwind_pinned(VM* vm, int target_brackets);
/* Fails the process if the evacuator's subtype coverage table has a hole. */
static void vm_evac_assert_table_total(void);

/** @brief Tear down the heap's arena region stack and free its object
 *         table. */
static void heap_destroy(Heap* h) {
    vm_region_stack_destroy(&h->regions);
    for (int i = 0; i < VM_ARENA_MAX_REGIONS; i++) free(h->region_slots[i].slots);
    memset(h->region_slots, 0, sizeof(h->region_slots));
    free(h->markbits);   h->markbits = NULL;   h->markbits_cap = 0;
    free(h->mark_stack); h->mark_stack = NULL; h->mark_stack_n = h->mark_stack_cap = 0;
    free(h->free_slots); h->free_slots = NULL; h->n_free_slots = h->cap_free_slots = 0;
    free(h->objects);
    h->objects = NULL;
}

/*******************************************************************************
 * Call Frame
 ******************************************************************************/

typedef struct {
    int32_t return_pc;
    int32_t return_fp;
    int32_t func_pc;     /* for debugging */
} CallFrame;

/*******************************************************************************
 * VM State
 ******************************************************************************/

typedef struct VM {
    /* Program */
    Instr* code;
    int code_len;
    /* Growable constant pool. It used to be a fixed `Value[MAX_CONSTS]` while
     * the copy-in loops clamped at MAX_CONSTS and still set n_constants to the
     * chunk's full count — so a program with more constants than the pool (a
     * multi-thousand-member literal is the easy way to get there) executed
     * OP_CONST against uninitialized slots. Capacity now grows with the
     * program, up to ESHKOL_VM_MAX_CONSTS_CEILING. */
    Value* constants;
    int n_constants;
    int const_cap;

    /* Execution state */
    int32_t pc;
    Value stack[STACK_SIZE];
    int32_t sp;           /* stack pointer (next free slot) */

    /* Call frames */
    CallFrame frames[MAX_FRAMES];
    int32_t fp;           /* frame pointer (base of current frame's locals) */
    int frame_count;

    /* Heap */
    Heap heap;

    /* Output */
    Value outputs[256];
    int n_outputs;

    /* Exception handling */
    struct {
        int pc;
        int sp;
        int fp;
        int frame_count;
        int n_winds;
        int n_parameter_bindings;
        Value promise_mark;
        /* #341: open-handle sequence mark. A raise retires every region handle
         * opened after the handler was installed, so handle liveness after a
         * caught exception reads identically on the VM and on native. */
        uint64_t region_handle_mark;
        /* Stage-1 evacuator: how many `with-region` brackets were open when
         * this handler was installed. A raise closes every region entered
         * since, promoting the raised value out of each on the way — the same
         * guarantee the native raise path gives (runtime_exceptions_hosted.cpp
         * calls eshkol_region_unwind_to with the in-flight value). Without it a
         * caught exception would leave the region stack deeper than the
         * program thinks it is, and the arenas would never be released. */
        int region_bracket_mark;
    } handler_stack[16];
    int n_handlers;
    Value current_exception;

    /* Dynamic-wind stack */
    struct { Value before; Value after; } wind_stack[32];
    int n_winds;

    /* STORE/CONTROL boundary: stack slots [0, global_top) hold top-level
     * bindings (the store), everything at or above is operand/frame state
     * (the control stack). Raised monotonically by OP_GLOBAL_MARK. A
     * continuation captures and restores only the control side. */
    int global_top;

    /* Runaway-instruction budget. This lives on the VM, not in a vm_run()
     * local, so it accumulates across nested vm_run() calls (native->closure
     * callbacks) and across the native-escape longjmp a continuation invoke
     * performs. As a vm_run() local it was reset to zero on every
     * continuation invocation, so an infinite continuation loop never tripped
     * the guard and hung silently instead of failing loudly. */
    uint64_t insns_executed;

    /* Dynamic parameter bindings parallel dynamic-wind for VM exception
     * unwinding.  Each entry names a VmParameter whose stack received an
     * actual native 702 push; normal 703 and exceptional exits pop LIFO. */
    Value parameter_bindings[64];
    int n_parameter_bindings;

    /* `with-region` brackets currently open, innermost last. Each entry is the
     * heap region depth the matching push established, or -1 for a bracket
     * whose push was refused (region-stack overflow) and which must therefore
     * pop nothing. Tracked separately from heap.regions.depth so a bracket can
     * never close a region it did not open. */
    int region_bracket_marks[VM_ARENA_MAX_REGIONS];
    int n_region_brackets;

    /* Status */
    int halted;
    int error;
    int native_policy;

    /* Promise evaluation is an intrusive chain stored in each evaluating
     * promise's cached slot.  Nonlocal control rolls this chain back to the
     * mark captured by its handler/continuation, keeping failed promises
     * retryable without allocating an auxiliary stack. */
    Value promise_eval_head;

    /* Scheme closures invoked by a C native recursively enter vm_run().
     * A handled raise or continuation transfer must escape every intervening
     * C helper frame and resume the owning interpreter loop at the restored
     * VM state, rather than letting the nested loop consume the handler. */
    int native_call_depth;
    int native_escape_ready;
    jmp_buf native_escape_jmp;

    uint32_t language_coverage_call_hash;
    int32_t language_coverage_call_pc;

    /* Reverse-mode AD tracing context.
     * When active_tape != NULL, arithmetic operations (+,-,*,/,sin,cos,...)
     * record on the tape. Each stack value that flows through tape-aware ops
     * gets tracked via ad_node_map: maps stack slot → tape node index.
     * This enables transparent reverse-mode gradient computation. */
    void* active_tape;                /* AdTape* or NULL */
    int   ad_node_map[STACK_SIZE];    /* stack slot → tape node index (-1 = not tracked) */

    /* Backend-local AD instrumentation.  These mirror the public native
     * `(ad-*-counters)` contract, but count the VM's own exact/finite-
     * difference work instead of reporting the LLVM runtime's globals. */
    uint64_t ad_primal_calls;
    uint64_t ad_reverse_passes;
    uint64_t ad_tape_allocations;
    uint64_t ad_tape_nodes;
    uint64_t ad_scalar_ad_nodes;
    uint64_t ad_tensor_ad_nodes;
    uint64_t ad_finite_difference_evals;

    /* VM-lifetime geometric optimizer state for compatibility builtins. */
    void* geometric_adam_states[16];   /* VmRiemannianAdamState* */

    /* VM-lifetime process handles.  A PTY process is exposed to Scheme as
     * (pid . master-fd), while these slots let native wait/kill/read accept
     * either that handle or the pid directly. */
    struct { int64_t pid; int fd; } pty_handles[64];
    int n_pty_handles;

    /* VM-lifetime file watchers.  The standalone VM uses stat-based polling
     * for deterministic, dependency-free watcher handles. */
    struct {
        int active;
        int recursive;
        int exists;
        int64_t mtime_ns;
        int64_t size;
        char path[1024];
    } fs_watchers[32];

    struct {
        int active;
        int64_t handle;
    } sleep_inhibitors[16];
    int64_t next_sleep_inhibitor;

    struct {
        int active;
        void* regex;
    } regex_handles[32];

    struct {
        int active;
        int fd;
        int len;
        char buffer[4096];
    } line_readers[32];

    struct {
        int active;
        int max_size;
        int size;
        int64_t tick;
        struct {
            int active;
            int64_t tick;
            Value key;
            Value value;
        } entries[64];
    } lru_caches[16];

    struct {
        int active;
        struct {
            int active;
            int once;
            Value event;
            Value handler;
        } listeners[64];
    } event_emitters[16];

    struct {
        int active;
        int capacity;
        int head;
        int tail;
        int count;
        int closed;
        Value buffer[64];
    } channels[16];

    struct {
        int active;
        int locked;
        int recursion;
    } mutexes[16];

    struct {
        int active;
        int signals;
    } condvars[16];

    struct {
        int allocated;
        int active;
        int repeating;
        int fired_count;
        int64_t next_due_ms;
        int64_t interval_ms;
        Value callback;
    } timers[32];
    int polling_timers;

    Value exit_handlers[32];
    int n_exit_handlers;
    int exit_handlers_drained;

    struct {
        int active;
        void* handle;
    } dynamic_libraries[32];

    struct {
        int active;
        int parent;
        int child_count;
    } yoga_nodes[512];

    struct {
        int active;
        int listen_fd;
        int client_fd;
        int port;
    } http_servers[8];

    struct {
        int active;
        int fd;
        int closed;
    } websocket_clients[16];

    char http_proxy_url[512];
    char http_tls_cert[512];
    char http_tls_key[512];
    char http_tls_ca[512];

    struct {
        int active;
    } ts_parsers[32];

    struct {
        int active;
    } ts_trees[64];

    struct {
        int active;
        int tree;
        uint32_t start;
        uint32_t end;
        int root;
        char type[128];
    } ts_nodes[4096];

    struct {
        int active;
    } ts_queries[32];
} VM;

/* Command-line arguments (set in main, read by native 602) */
static int g_vm_argc = 0;
static char** g_vm_argv = NULL;
/** @brief Stash the process's argc/argv for later retrieval by native call
 *         602 (`command-line`). */
static void vm_set_command_line(int argc, char** argv) { g_vm_argc = argc; g_vm_argv = argv; }

/** @brief Zero-initialize a VM instance: clears all state, initializes the
 *         heap, sets the default native policy, and marks the AD tape
 *         inactive with an empty node map. */
/** @brief Ensure the constant pool can hold at least @p need entries, growing
 *         it geometrically. Returns 0 (and reports the ceiling by name) when
 *         @p need exceeds ESHKOL_VM_MAX_CONSTS_CEILING or on allocation
 *         failure. */
static int vm_ensure_const_cap(VM* vm, int need) {
    if (!vm || need < 0) return 0;
    if (need <= vm->const_cap) return 1;
    if (need > ESHKOL_VM_MAX_CONSTS_CEILING) {
        fprintf(stderr, "ERROR: constant pool limit reached "
                        "(need %d, ESHKOL_VM_MAX_CONSTS_CEILING=%d)\n",
                need, (int)ESHKOL_VM_MAX_CONSTS_CEILING);
        return 0;
    }
    int cap = vm->const_cap > 0 ? vm->const_cap : MAX_CONSTS;
    while (cap < need) {
        if (cap > ESHKOL_VM_MAX_CONSTS_CEILING / 2) { cap = ESHKOL_VM_MAX_CONSTS_CEILING; break; }
        cap *= 2;
    }
    Value* grown = (Value*)realloc(vm->constants, (size_t)cap * sizeof(Value));
    if (!grown) {
        fprintf(stderr, "ERROR: constant pool growth to %d entries failed\n", cap);
        return 0;
    }
    memset(grown + vm->const_cap, 0, (size_t)(cap - vm->const_cap) * sizeof(Value));
    vm->constants = grown;
    vm->const_cap = cap;
    return 1;
}

static void vm_init(VM* vm) {
    memset(vm, 0, sizeof(VM));
    heap_init(&vm->heap);
    vm->constants = NULL;
    vm->const_cap = 0;
    (void)vm_ensure_const_cap(vm, MAX_CONSTS);
    vm->native_policy = ESHKOL_VM_NATIVE_POLICY_DESKTOP;
    vm->active_tape = NULL;
    memset(vm->ad_node_map, -1, sizeof(vm->ad_node_map));
}

/** @brief Bounds-check a heap object index against the live-object range
 *         [0, next_free). */
static inline int is_valid_heap_ptr(VM* vm, int32_t ptr) {
    /* The NULL-slot test is load-bearing since the Stage-1 region evacuator:
     * a reclaimed index keeps its place in the table but its pointer is
     * cleared, so a stale reference reads as an INVALID heap pointer (which
     * every caller already handles) instead of dereferencing freed arena
     * memory. Without it a missed reference would be a silent aliasing bug
     * rather than a loud type error — the exact failure mode SW-14's ruling
     * called strictly worse than the leak it replaces. */
    return ptr >= 0 && ptr < vm->heap.next_free && vm->heap.objects[ptr] != NULL;
}

/** @brief VM-aware coercion of a Value to a double, extending as_number()
 *         to unwrap heap-boxed rationals (num/denom), bignums and duals
 *         (primal component) via the VM's heap.
 *
 * Every numeric tag must be covered here: a tag this function does not know
 * reads the Value's .as union as an int64 heap index and answers 0.0, which
 * is how a heap-boxed operand reaching a plain-double path turns into a
 * silent zero rather than an error. */
static double as_number_vm(VM* vm, Value v) {
    if (v.type == VAL_INT) return (double)v.as.i;
    if (v.type == VAL_FLOAT) return v.as.f;
    if (v.type == VAL_CHAR) return (double)v.as.i; /* codepoint */
    if (v.type == VAL_RATIONAL && vm) {
        VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
        /* SW-18: a bignum-backed rational has num/denom = 0/1, so reading the
         * int64 halves unconditionally answered 0.0 for every big rational. */
        if (r && r->is_big) return vm_rational_to_double(r);
        if (r && r->denom != 0) return (double)r->num / (double)r->denom;
    }
    if (v.type == VAL_BIGNUM && vm) {
        VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr;
        if (b) return bignum_to_double(b);
    }
    if (v.type == VAL_DUAL && vm) {
        VmDual* d = (VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr;
        if (d) return d->primal;
    }
    return 0.0;
}

/** @brief Validate that @p v's heap pointer is in range AND its object
 *         header matches @p type. */
static inline int is_heap_type(VM* vm, Value v, HeapType type) {
    return v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
           vm->heap.objects[v.as.ptr]->type == type;
}

/** @brief Push @p v onto the VM's value stack, setting vm->error on
 *         overflow. */
static void vm_push(VM* vm, Value v) {
    if (vm->sp >= STACK_SIZE) { fprintf(stderr, "STACK OVERFLOW\n"); vm->error = 1; return; }
    vm->stack[vm->sp++] = v;
}

/** @brief Pop and return the top of the VM's value stack, setting
 *         vm->error and returning NIL on underflow. */
static Value vm_pop(VM* vm) {
    if (vm->sp <= 0) { fprintf(stderr, "STACK UNDERFLOW\n"); vm->error = 1; return NIL_VAL; }
    return vm->stack[--vm->sp];
}

/** @brief Read the value @p offset slots below the top of the VM's value
 *         stack without popping (offset 0 = TOS). */
static Value vm_peek(VM* vm, int offset) {
    int idx = vm->sp - 1 - offset;
    if (idx < 0 || idx >= vm->sp) { fprintf(stderr, "PEEK OUT OF BOUNDS\n"); return NIL_VAL; }
    return vm->stack[idx];
}

/** @brief Append @p v to the VM's constant pool, growing it as needed.
 * @return The new constant's index, or -1 if the pool cannot grow.
 */
static int add_constant(VM* vm, Value v) {
    if (!vm_ensure_const_cap(vm, vm->n_constants + 1)) return -1;
    vm->constants[vm->n_constants] = v;
    return vm->n_constants++;
}

/*******************************************************************************
 * Print value
 ******************************************************************************/

/* Forward declarations for print_value */
typedef struct { Value* items; int len; int cap; } VmVector;

static void print_value_mode(VM* vm, Value v, int write_syntax);
static void print_value(VM* vm, Value v);

/**
 * @brief ESH-0226: print an N-dimensional tensor as nested vector literal
 *        syntax (#(...) for 1D, #((...) (...)) for 2D, etc.), matching the
 *        native/LLVM runtime's display_tensor()/display_tensor_recursive()
 *        (lib/core/runtime_display_hosted.cpp) so `(display
 *        (tensor-matmul ...))` renders identically on both the bytecode VM
 *        and the native compiler.
 */
static void print_tensor_recursive(VM* vm, const VmTensor* t, int dim, int64_t offset) {
    int64_t dim_size = t->shape[dim];
    if (dim == t->n_dims - 1) {
        printf("(");
        for (int64_t i = 0; i < dim_size; i++) {
            if (i) printf(" ");
            print_value(vm, FLOAT_VAL(t->data[offset + i]));
        }
        printf(")");
        return;
    }
    int64_t stride = 1;
    for (int k = dim + 1; k < t->n_dims; k++) stride *= t->shape[k];
    printf("(");
    for (int64_t i = 0; i < dim_size; i++) {
        if (i) printf(" ");
        print_tensor_recursive(vm, t, dim + 1, offset + i * stride);
    }
    printf(")");
}

/**
 * @brief Recursively print a runtime Value for `display`/`write`: dispatches
 *        on @p v's type tag, unwrapping heap-boxed objects (pairs, strings,
 *        vectors, complex/rational/tensor/factor-graph/workspace/etc.) via
 *        the VM's heap. Most opaque heap types not yet given a full
 *        printer render as a `<type-name>` placeholder.
 */
static void print_value_mode(VM* vm, Value v, int write_syntax) {
    switch ((int)v.type) {
        case VAL_NIL:   printf("()"); break;
        case VAL_INT:   printf("%lld", (long long)v.as.i); break;
        case VAL_FLOAT: { char fbuf[48]; eshkol_dtoa_shortest(fbuf, sizeof(fbuf), v.as.f); fputs(fbuf, stdout); break; }
        case VAL_CHAR: {
            if (write_syntax) {
                if (v.as.i == ' ') { fputs("#\\space", stdout); break; }
                if (v.as.i == '\n') { fputs("#\\newline", stdout); break; }
                if (v.as.i == '\t') { fputs("#\\tab", stdout); break; }
                fputs("#\\", stdout);
            }
            char buf[4];
            int n = vm_utf8_encode((int)v.as.i, buf);
            if (n > 0) printf("%.*s", n, buf);
            break;
        }
        case VAL_BOOL:  printf("%s", v.as.b ? "#t" : "#f"); break;
        case VAL_PAIR: {
            /* A fact is a heap wrapper around its `(predicate arg ...)`
             * datum; print the datum, not the wrapper, so `(display
             * (make-fact 'p 'a 'b))` is `(p a b)` on both substrates
             * (native eshkol_display_fact prints the same). */
            if (v.as.ptr >= 0 && vm->heap.objects[v.as.ptr] &&
                vm->heap.objects[v.as.ptr]->type == HEAP_FACT) {
                print_value_mode(vm, vm->heap.objects[v.as.ptr]->cons.car, write_syntax);
                break;
            }
            printf("(");
            Value cur = v;
            int first = 1;
            while (cur.type == VAL_PAIR) {
                if (!first) printf(" ");
                first = 0;
                HeapObject* obj = vm->heap.objects[cur.as.ptr];
                print_value_mode(vm, obj->cons.car, write_syntax);
                cur = obj->cons.cdr;
            }
            if (cur.type != VAL_NIL) {
                printf(" . ");
                print_value_mode(vm, cur, write_syntax);
            }
            printf(")");
            break;
        }
        case VAL_STRING: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                VmString* s = (VmString*)obj->opaque.ptr;
                if (!write_syntax) {
                    printf("%.*s", s->byte_len, s->data);
                } else {
                    fputc('"', stdout);
                    for (int i = 0; i < s->byte_len; ++i) {
                        unsigned char ch = (unsigned char)s->data[i];
                        switch (ch) {
                            case '"': fputs("\\\"", stdout); break;
                            case '\\': fputs("\\\\", stdout); break;
                            case '\n': fputs("\\n", stdout); break;
                            case '\r': fputs("\\r", stdout); break;
                            case '\t': fputs("\\t", stdout); break;
                            default: fputc((int)ch, stdout); break;
                        }
                    }
                    fputc('"', stdout);
                }
            }
            break;
        }
        case VAL_SYMBOL: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                VmString* s = (VmString*)obj->opaque.ptr;
                printf("%.*s", s->byte_len, s->data);
            }
            break;
        }
        case VAL_VECTOR: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            printf("#(");
            if (obj && obj->opaque.ptr) {
                VmVector* vec = (VmVector*)obj->opaque.ptr;
                for (int i = 0; i < vec->len; i++) {
                    if (i) printf(" ");
                    print_value_mode(vm, vec->items[i], write_syntax);
                }
            }
            printf(")");
            break;
        }
        case VAL_CLOSURE: printf("<closure@%d>", v.as.ptr); break;
        case VAL_COMPLEX: {
            VmComplex* z = (VmComplex*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            if (z) {
                char rb[48], ib[48];
                eshkol_dtoa_shortest(rb, sizeof(rb), z->real);
                eshkol_dtoa_shortest(ib, sizeof(ib), z->imag);
                /* dtoa never emits a leading '+', so force an explicit sign on
                 * the imaginary part (R7RS complex external representation). */
                if (ib[0] == '+' || ib[0] == '-') printf("%s%si", rb, ib);
                else                               printf("%s+%si", rb, ib);
            }
            else printf("<complex>");
            break;
        }
        case VAL_RATIONAL: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                VmRational* r = (VmRational*)obj->opaque.ptr;
                if (r->is_big) {
                    /* SW-18: print the exact bignum halves, not the int64
                     * shadow (which is 0/1 on the big path). */
                    char* ns = bignum_to_string(&vm->heap.regions, r->big_num);
                    char* ds = bignum_to_string(&vm->heap.regions, r->big_den);
                    if (ns && ds) printf("%s/%s", ns, ds);
                    else printf("<rational>");
                }
                else if (r->denom == 1) printf("%lld", (long long)r->num);
                else printf("%lld/%lld", (long long)r->num, (long long)r->denom);
            } else printf("<rational>");
            break;
        }
        case VAL_BIGNUM: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                VmBignum* b = (VmBignum*)obj->opaque.ptr;
                char* s = bignum_to_string(&vm->heap.regions, b);
                if (s) printf("%s", s);
                else printf("<bignum>");
            } else printf("<bignum>");
            break;
        }
        case VAL_I128: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                char buf[ESHKOL_I128_STR_MAX];
                __int128 x = eshkol_i128_from_abi(*(eshkol_i128_abi*)obj->opaque.ptr);
                eshkol_i128_format(x, buf);
                printf("%s", buf);
            } else printf("<i128>");
            break;
        }
        case VAL_DUAL: printf("<dual>"); break;
        case VAL_TENSOR: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            VmTensor* t = (obj && obj->opaque.ptr) ? (VmTensor*)obj->opaque.ptr : NULL;
            if (!t || t->n_dims == 0 || t->total == 0) { printf("#()"); break; }
            printf("#");
            print_tensor_recursive(vm, t, 0, 0);
            break;
        }
        case VAL_FACTOR_GRAPH: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                VmFactorGraph* fg = (VmFactorGraph*)obj->opaque.ptr;
                printf("<factor-graph: %d vars, %d factors>",
                       fg->num_vars, fg->num_factors);
            } else printf("<factor-graph>");
            break;
        }
        case VAL_WORKSPACE: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                VmWorkspace* ws = (VmWorkspace*)obj->opaque.ptr;
                /* Same external form as the native runtime's
                 * eshkol_display_workspace. The VM printed a differently
                 * ordered, differently punctuated summary that also omitted
                 * step_count, so any program displaying a workspace read
                 * differently on the two engines. */
                printf("#<workspace: dim=%d, %d modules, step=%d>",
                       ws->dim, ws->n_modules, ws->step_count);
            } else printf("#<workspace: empty>");
            break;
        }
        case VAL_KB: {
            /* Same external form as the native runtime's eshkol_display_kb:
             * `#<knowledge-base: N facts>`. The VM printed a bare
             * `<knowledge-base>`, so every program that displayed a KB read
             * differently on the two engines. */
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            VmKnowledgeBase* kb = (obj && obj->opaque.ptr)
                ? (VmKnowledgeBase*)obj->opaque.ptr : NULL;
            if (kb) printf("#<knowledge-base: %d facts>", kb->n_facts);
            else    printf("#<knowledge-base: empty>");
            break;
        }
        case VAL_SUBST: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            vm_print_substitution(obj ? (const VmSubstitution*)obj->opaque.ptr : NULL);
            break;
        }
        case VAL_HASH:        printf("<hash-table>"); break;
        case VAL_BYTEVECTOR:  printf("<bytevector>"); break;
        case VAL_PARAMETER_OBJ: printf("<parameter>"); break;
        case VAL_AD_TAPE:     printf("<ad-tape>"); break;
        case VAL_ERROR_OBJ:   printf("<error-object>"); break;
        case VAL_MANIFOLD:    printf("<manifold>"); break;
        case VAL_RIEMANNIAN_ADAM_STATE: printf("<riemannian-adam-state>"); break;
        case VAL_PORT:        printf("<port>"); break;
        case VAL_FUTURE:      printf("<future>"); break;
        case VAL_EOF:         printf("#<eof>"); break;
        case VAL_VOID:        break; /* unspecified — produces no output */
        default: printf("<unknown>"); break;
    }
}

/** Display-style printer used by diagnostics and the `display` primitive. */
static void print_value(VM* vm, Value v) {
    print_value_mode(vm, v, 0);
}

/*******************************************************************************
 * Forward declaration for vm_run (needed by closure bridge below)
 ******************************************************************************/

static void vm_run(VM* vm);

/**
 * @brief Call a VM closure from native C code — the critical bridge that
 *        lets native functions (ws-step!, parallel-map,
 *        call-with-values, etc.) invoke user-defined closures.
 *
 * Protocol:
 *   1. Save entire VM state (pc, fp, sp, frame_count, halted)
 *   2. Push closure + args, set up frame with return_pc = -1 as sentinel
 *   3. Run vm_run — OP_RETURN detects sentinel, halts, pushes result
 *   4. Capture result, restore VM state, return it
 */
static Value vm_call_closure_from_native(VM* vm, Value closure, Value* args, int argc) {
    if (closure.type != VAL_CLOSURE || closure.as.ptr < 0) return NIL_VAL;
    HeapObject* cl = vm->heap.objects[closure.as.ptr];
    if (!cl) return NIL_VAL;

    /* Save VM state */
    int32_t saved_pc = vm->pc;
    int32_t saved_fp = vm->fp;
    int32_t saved_sp = vm->sp;
    int saved_frame_count = vm->frame_count;
    int saved_halted = vm->halted;
    int saved_error = vm->error;

    /* Push closure below args (calling convention: func at sp-argc-1) */
    vm_push(vm, closure);
    for (int i = 0; i < argc; i++) vm_push(vm, args[i]);

    /* Set up call frame with sentinel */
    if (vm->frame_count >= MAX_FRAMES) {
        vm->sp = saved_sp; /* restore */
        return NIL_VAL;
    }
    vm->frames[vm->frame_count].return_pc = -1; /* SENTINEL: return to native */
    vm->frames[vm->frame_count].return_fp = saved_fp;
    vm->frames[vm->frame_count].func_pc = cl->closure.func_pc;
    vm->frame_count++;
    vm->fp = vm->sp - argc;
    vm->pc = cl->closure.func_pc;
    vm->halted = 0;
    vm->error = 0;

    /* Run VM loop — will stop when OP_RETURN hits our sentinel frame */
    vm->native_call_depth++;
    vm_run(vm);
    vm->native_call_depth--;

    const int callee_error = vm->error;

    /* Capture result (should be on stack) */
    Value result = NIL_VAL;
    if (vm->sp > saved_sp) {
        result = vm->stack[vm->sp - 1];
    }

    /* Restore VM state */
    vm->pc = saved_pc;
    vm->fp = saved_fp;
    vm->sp = saved_sp;
    vm->frame_count = saved_frame_count;
    vm->halted = saved_halted;
    vm->error = saved_error || callee_error;

    return result;
}

/*******************************************************************************
 * Shape extraction helper: parse a Value into an int64_t shape array.
 * Handles both list (VAL_PAIR) and scalar (VAL_INT/VAL_FLOAT) shapes.
 * Returns number of dimensions filled (0 on error).
 ******************************************************************************/

/**
 * @brief Extract a tensor shape from a Scheme Value that may be either a
 *        proper list of dimension sizes or a single scalar dimension
 *        (treated as a 1-D shape), writing up to @p max_dims entries into
 *        @p shape.
 * @return The number of dimensions written.
 */
static int vm_extract_shape(VM* vm, Value shape_val, int64_t* shape, int max_dims) {
    int n_dims = 0;
    if (shape_val.type == VAL_PAIR) {
        Value cur = shape_val;
        while (cur.type == VAL_PAIR && n_dims < max_dims) {
            shape[n_dims++] = (int64_t)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
    } else {
        shape[0] = (int64_t)as_number(shape_val);
        n_dims = 1;
    }
    return n_dims;
}

/* Continuation: saved VM state for call/cc */
typedef struct {
    /* stack_base: the STORE/CONTROL boundary at capture time (vm->global_top).
     * saved_stack holds slots [stack_base, sp) only. Slots below it are
     * top-level bindings — the store — which R7RS `call/cc` does not capture
     * and a re-entry must therefore never roll back (SW-52). */
    int stack_base;
    int pc, fp, sp, frame_count, n_handlers, n_winds, n_parameter_bindings;
    /* `with-region` brackets open at capture. Invoking the continuation closes
     * every bracket entered since, exactly as native's
     * eshkol_region_unwind_for_continuation() does. */
    int n_region_brackets;
    Value promise_mark;
    Value* saved_stack;
    CallFrame* saved_frames;
    Value* saved_wind_befores;
    Value* saved_wind_afters;
    Value* saved_parameter_bindings;
    Value* saved_parameter_values;
} VmContinuation;

/* Simple vector for VEC_CREATE/VEC_REF/VEC_SET/VEC_LEN opcodes */
/* VmVector defined earlier (before print_value) */

/* Macro: allocate heap object, set type, set opaque ptr, push result */
#define VM_PUSH_HEAP_OPAQUE(vm, heap_type, val_type, ptr_val) do { \
    int32_t _hp = heap_alloc(&(vm)->heap); \
    if (_hp < 0) { (vm)->error = 1; break; } \
    (vm)->heap.objects[_hp]->type = (heap_type); \
    (vm)->heap.objects[_hp]->opaque.ptr = (ptr_val); \
    vm_push((vm), (Value){.type = (val_type), .as.ptr = _hp}); \
} while(0)

#define VM_PUSH_TENSOR(vm, tptr) VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_TENSOR, tptr)

/*******************************************************************************
 * vm_dispatch_native — ALL native function dispatch in one place.
 *
 * Both the computed-goto and switch paths call this, eliminating duplication.
 ******************************************************************************/
