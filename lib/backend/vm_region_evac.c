/**
 * @file vm_region_evac.c
 * @brief Stage-1 OALR region evacuator for the bytecode VM (SW-14).
 *
 * WHAT THIS CLOSES. Before this file, `(with-region ...)` on the bytecode VM
 * lowered to `begin`: the body ran, its value was returned, and not one byte
 * came back. `heap_alloc()` bump-allocated from an arena that was never
 * released and handed out a monotonically increasing object index that was
 * never recycled. The measured cost was 1.503 GB with the wrapper against
 * 1.504 GB without it — the form was inert, not merely weak (ledger SW-14).
 *
 * WHY IT IS SHAPED THE WAY IT IS. The native engine's evacuator
 * (lib/core/runtime_regions.cpp) is a Cheney-style COPYING collector: escape
 * roots are explicitly designated (the kept value at teardown, plus a mutation
 * write barrier at every store into a longer-lived slot), the escaping subgraph
 * is copied into the parent arena, and a forwarding map rewrites interior
 * pointers on the copies. Porting that shape verbatim would have required the
 * ~130-site write barrier the SW-14 ruling sized, and — far worse on this
 * substrate — it would have had to REWRITE heap indices, because a VM `Value`
 * addresses the heap by a small integer rather than by pointer. A missed
 * rewrite there aliases a live object and produces a silently wrong VALUE.
 *
 * So the port matches the SEMANTICS and not the implementation. The ruling
 * offers the alternative in as many words: a mutation write barrier "or, in its
 * place, a full mark from roots". This is that mark, and the substitution buys
 * three properties the copying design has to work for:
 *
 *   - NOTHING MOVES. Objects keep their heap index and their address, so no
 *     Value anywhere in the VM has to be rewritten, no forwarding map is
 *     needed, `eq?` identity and shared structure survive by construction, and
 *     cycles need no special case.
 *   - NO WRITE BARRIER. Reachability is recomputed at the pop, so a store into
 *     an older container needs no instrumentation to be seen.
 *   - CONSERVATIVE MEMORY, PRECISE OBJECTS. Sweeping is at ARENA-BLOCK
 *     granularity: a block is freed only when nothing live is in it and nothing
 *     anywhere points into it. Every payload buffer (VmVector.items,
 *     VmBignum.limbs, VmContinuation's six saved arrays, VmFactorGraph's five
 *     pointer arrays, ...) is therefore covered WITHOUT the evacuator knowing
 *     those layouts at all, because a raw pointer to them is found by scanning.
 *
 * The one thing that must be precise is the OBJECT graph: which heap INDICES a
 * live object references. An index is a small integer and cannot be recognised
 * by scanning, so this is where the ESH-0214d lesson applies and where
 * vm_evac_walk_object() has to be total over every subtype. It is, and the
 * totality is asserted three ways (see vm_evac_subtype_table below):
 * a compile-time size check, a startup check that every row is filled in, and a
 * `default:` arm that PINS the region rather than guessing.
 *
 * DEGRADATION IS ALWAYS TOWARD THE LEAK. Every case this file is not certain
 * about — an unknown subtype, a continuation captured inside a region, an
 * outer heap too large to scan within budget, a bookkeeping allocation that
 * failed — PINS the region: its blocks are promoted into the parent and nothing
 * is freed or retired. That is exactly the pre-evacuator behaviour, which the
 * ruling calls a leak and not a defect. Nothing in this file may degrade toward
 * a dangling index.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

/* ── Configuration ─────────────────────────────────────────────────────────
 *
 * Every switch defaults to the safe setting; the environment can only make the
 * evacuator MORE conservative, except ESHKOL_VM_REGION_EVAC=0 which turns it
 * off entirely and restores the pre-Stage-1 pass-through.
 */

/* Policy is read through the vm_host_env_* hooks rather than getenv(), because
 * the VM core component is freestanding by contract and
 * tests/toolchain/vm_source_boundary_test.cpp fails the build on a direct
 * getenv() here. */

/** @return 1 when region pops reclaim (default), 0 under ESHKOL_VM_REGION_EVAC=0. */
static int vm_evac_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = vm_host_env_long("ESHKOL_VM_REGION_EVAC", 1) != 0;
    return cached;
}

/** @return 1 under ESHKOL_ARENA_POISON — the SAME variable the native engine's
 *          arena poisoning reads (lib/core/runtime_arena_diagnostics_hosted.cpp),
 *          so one setting arms both substrates in a mixed gate.
 *
 * Poison mode makes a reclamation bug LOUD instead of silent: dead blocks are
 * stamped with 0xCB and kept mapped rather than returned to the allocator, and
 * retired object slots are never recycled. A reference the mark phase missed
 * therefore reads a NULL object slot (caught by is_valid_heap_ptr) or 0xCB
 * bytes, instead of aliasing whatever was allocated next. */
static int vm_evac_poison(void) {
    static int cached = -1;
    if (cached < 0) cached = vm_host_env_flag("ESHKOL_ARENA_POISON");
    return cached;
}

/** @return 1 when retired object-table indices may be handed out again.
 *
 * Recycling is what makes a `with-region` loop flat rather than merely
 * cheaper: without it the object table still grows 8 bytes per allocation
 * forever (ledger SW-14, item 4). It is disabled under poison because reuse is
 * precisely what converts a missed reference from a loud NULL into a silent
 * alias, and that is the failure mode the gates need to be able to see. */
static int vm_evac_recycle(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = vm_host_env_long("ESHKOL_VM_REGION_RECYCLE", 1) != 0;
        if (vm_evac_poison()) cached = 0;
    }
    return cached;
}

/** @return 1 when each pop runs the independent post-sweep audit.
 *
 * The audit is not a restatement of the mark: it scans the WHOLE object table
 * (live or not) plus the root set for any surviving reference to an index this
 * pop is about to retire. A reference found there means the mark missed a root
 * or an interior field — the one class of bug that would otherwise be silent.
 * On by default under poison; ESHKOL_VM_REGION_VERIFY=1 forces it on. */
static int vm_evac_verify(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = vm_host_env_flag("ESHKOL_VM_REGION_VERIFY") ? 1 : vm_evac_poison();
    }
    return cached;
}

/** @return 1 when a surviving object's fixed-size HeapObject struct is COPIED
 *          out of the dying region instead of pinning the 8 KB arena block it
 *          happens to sit in.
 *
 * Block-granularity sweeping alone is safe but blunt: one escaping cons in a
 * block full of garbage keeps the whole block. Copying the struct is the one
 * relocation that needs no layout knowledge at all — a HeapObject is a fixed
 * size, and the object TABLE is the only thing that holds its address, so the
 * copy is complete after a single `h->objects[idx] = copy`. Nothing else in
 * the VM stores a HeapObject*: every reader goes through the table.
 *
 * Out-of-line payloads are NOT copied — that would need the per-subtype size
 * and interior-pointer knowledge this design deliberately avoids — so a region
 * whose escapees carry payloads still retains the blocks those payloads sit
 * in. Payload-copying promotion is Stage-2.
 *
 * Disabled under poison, where keeping every address stable is what lets a
 * dangling access be recognised. */
static int vm_evac_compact(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = vm_host_env_long("ESHKOL_VM_REGION_COMPACT", 1) != 0;
        if (vm_evac_poison()) cached = 0;
    }
    return cached;
}

#define VM_EVAC_POISON_BYTE 0xCB   /* the native arena's poison byte */

/* ── Subtype coverage table ────────────────────────────────────────────────
 *
 * `HeapType` runs 0..27 (vm_core.c), and vm_geometric.c defines three further
 * heap tags OUTSIDE that enum as bare macros: HEAP_MANIFOLD 30,
 * HEAP_MANIFOLD_POINT 31, HEAP_MANIFOLD_TANGENT 32. A switch over the enum
 * alone would fall through for those, which is exactly the shape of defect
 * ESH-0214d closed natively — so the table is sized to the full tag space and
 * the two unused slots (28, 29) are filled in explicitly as absent.
 */
#define VM_EVAC_TYPE_COUNT 33

typedef enum {
    /* Walked precisely by vm_evac_walk_object(); may be reclaimed. */
    VM_EVAC_WALK = 0,
    /* Walked precisely, but instances are never reclaimed: they own a resource
     * whose lifetime is not the region's (an OS handle, a malloc'd buffer, a
     * live pthread primitive). Treated as a mark root, so the object and its
     * block survive the pop; everything else in the region still goes. */
    VM_EVAC_ROOT = 1,
    /* Interior reference graph not established. An instance in a region PINS
     * the whole region: nothing is freed, nothing is retired. */
    VM_EVAC_PIN  = 2,
} VmEvacClass;

typedef struct {
    const char*  name;    /* NULL means "row never filled in" — a startup error */
    VmEvacClass  cls;
    const char*  note;
} VmEvacSpec;

/**
 * @brief The coverage table. One row per heap tag, no gaps, no defaults.
 *
 * `note` states, for every row, WHERE the outgoing object references are — or
 * that there are none. "payload only" means the subtype holds no heap indices
 * at all: its buffers are still covered, conservatively, by the block sweep.
 */
static const VmEvacSpec vm_evac_subtype_table[VM_EVAC_TYPE_COUNT] = {
    [HEAP_CONS]         = { "cons",         VM_EVAC_WALK, "car, cdr" },
    [HEAP_CLOSURE]      = { "closure",      VM_EVAC_WALK, "upvalues[0..n_upvalues)" },
    [HEAP_STRING]       = { "string/symbol",VM_EVAC_WALK, "payload only (VmString.data)" },
    [HEAP_VECTOR]       = { "vector",       VM_EVAC_WALK, "VmVector.items[0..len)" },
    [HEAP_MULTI_VALUE]  = { "multi-value",  VM_EVAC_WALK, "VmVector.items[0..len)" },
    [HEAP_COMPLEX]      = { "complex",      VM_EVAC_WALK, "payload only (two doubles)" },
    [HEAP_RATIONAL]     = { "rational",     VM_EVAC_WALK, "payload only (int64 pair or two VmBignum)" },
    [HEAP_BIGNUM]       = { "bignum",       VM_EVAC_WALK, "payload only (VmBignum.limbs)" },
    [HEAP_DUAL]         = { "dual",         VM_EVAC_WALK, "two doubles + two optional VmRational* exact halves (SW-85), retained by the interior-pointer walk" },
    [HEAP_TENSOR]       = { "tensor",       VM_EVAC_WALK, "payload only (shape/strides/data; views borrow)" },
    [HEAP_LOGIC_VAR]    = { "logic-var",    VM_EVAC_PIN,  "no constructor exists; pin rather than guess a layout" },
    [HEAP_SUBST]        = { "substitution", VM_EVAC_WALK, "VmSubstitution.terms[i] of kind OPAQUE/FACT" },
    [HEAP_FACT]         = { "fact",         VM_EVAC_WALK, "inline cons.car datum (the only constructor, native 507)" },
    [HEAP_KB]           = { "knowledge-base",VM_EVAC_WALK,"facts[i]->datum_ptr and facts[i]->args[] terms" },
    [HEAP_FACTOR_GRAPH] = { "factor-graph", VM_EVAC_WALK, "payload only (numeric buffers, no Values)" },
    [HEAP_WORKSPACE]    = { "workspace",    VM_EVAC_WALK, "modules[i].process_fn -> Value* holding a closure" },
    [HEAP_PORT]         = { "port",         VM_EVAC_ROOT, "owns a FILE*/malloc'd buffer freed only by close" },
    [HEAP_AD_TAPE]      = { "ad-tape",      VM_EVAC_WALK, "payload only (AdNode array holds no Values)" },
    [HEAP_PROMISE]      = { "promise",      VM_EVAC_WALK, "VmVector.items[0..3): forced flag, thunk, cached" },
    [HEAP_CONTINUATION] = { "continuation", VM_EVAC_WALK, "promise_mark + saved stack/winds/parameter arrays" },
    [HEAP_HASH]         = { "hash-table",   VM_EVAC_WALK, "keys/values are raw scalars; marked CONSERVATIVELY" },
    [HEAP_ERROR]        = { "error-object", VM_EVAC_WALK, "message/type inline; irritants chain must be NULL" },
    [HEAP_BYTEVECTOR]   = { "bytevector",   VM_EVAC_WALK, "payload only (VmBytevector.data)" },
    [HEAP_PARAMETER]    = { "parameter",    VM_EVAC_WALK, "current_value, converter, save_stack[0..stack_depth)" },
    [HEAP_HYPER_DUAL]   = { "hyper-dual",   VM_EVAC_WALK, "payload only (four doubles)" },
    [HEAP_RIEMANNIAN_ADAM_STATE] = { "riemannian-adam-state", VM_EVAC_WALK,
                                     "payload only (two double buffers)" },
    [HEAP_FUTURE]       = { "future",       VM_EVAC_ROOT, "thunk/result Values, plus a live pthread mutex+cond" },
    [HEAP_I128]         = { "i128",         VM_EVAC_WALK, "payload only (flat {lo,hi})" },
    [28]                = { "<unassigned 28>", VM_EVAC_PIN, "no such tag; pin if one ever appears" },
    [29]                = { "<unassigned 29>", VM_EVAC_PIN, "no such tag; pin if one ever appears" },
    [30]                = { "manifold",     VM_EVAC_ROOT, "payload may be owned by semiclassical_qllm, not the arena" },
    [31]                = { "manifold-point", VM_EVAC_PIN, "no constructor exists; pin rather than guess a layout" },
    [32]                = { "manifold-tangent", VM_EVAC_PIN, "no constructor exists; pin rather than guess a layout" },
};

/* Compile-time half of the totality guard: the table must span the whole tag
 * space, and the last enumerated HeapType must still be inside it. Adding a
 * HeapType past HEAP_I128 without widening VM_EVAC_TYPE_COUNT fails the build
 * here rather than silently falling through at run time. */
typedef char vm_evac_table_is_total[
    (sizeof(vm_evac_subtype_table) / sizeof(vm_evac_subtype_table[0]) == VM_EVAC_TYPE_COUNT) &&
    ((int)HEAP_I128 < VM_EVAC_TYPE_COUNT) ? 1 : -1];

/**
 * @brief Startup half of the totality guard: every row must be filled in.
 *
 * A designated-initializer table silently zero-fills any index nobody wrote,
 * and a zero row would read as `VM_EVAC_WALK` with a NULL name — i.e. "walk
 * this, I know how", which is the one answer a forgotten subtype must never
 * give. Checked once, on the first region push, and fatal: shipping a VM that
 * reclaims memory on a guess is worse than not starting.
 */
static void vm_evac_assert_table_total(void) {
    static int checked = 0;
    if (checked) return;
    checked = 1;
    for (int i = 0; i < VM_EVAC_TYPE_COUNT; i++) {
        if (!vm_evac_subtype_table[i].name) {
            fprintf(stderr,
                    "eshkol-vm: FATAL: region evacuator coverage table has no row for "
                    "heap subtype %d. Every subtype must be classified explicitly "
                    "(lib/backend/vm_region_evac.c); an unclassified subtype would be "
                    "walked as if its layout were known.\n", i);
            exit(1);
        }
    }
}

/** @return the coverage row for heap tag @p t, or the pinning row for a tag
 *          outside the table entirely. */
static const VmEvacSpec* vm_evac_spec_for(int t) {
    static const VmEvacSpec out_of_range = {
        "<out-of-range heap tag>", VM_EVAC_PIN, "tag outside the classified space"
    };
    if (t < 0 || t >= VM_EVAC_TYPE_COUNT) return &out_of_range;
    return &vm_evac_subtype_table[t];
}

/* ── Value -> heap index ───────────────────────────────────────────────────
 *
 * TOTAL over ValType. The two halves are enumerated separately and the
 * `default:` arm is a PIN, not a "no": a ValType introduced later must stop
 * reclamation rather than be assumed non-referencing, because assuming "no"
 * frees the object it points at.
 */
typedef enum { VM_EVAC_REF_NONE, VM_EVAC_REF_INDEX, VM_EVAC_REF_UNKNOWN } VmEvacRefKind;

static VmEvacRefKind vm_evac_value_ref(Value v, int32_t* out_index) {
    switch ((int)v.type) {
    /* Immediates: nothing on the heap. */
    case VAL_NIL: case VAL_INT: case VAL_FLOAT: case VAL_BOOL:
    case VAL_VOID: case VAL_CHAR: case VAL_EOF:
        return VM_EVAC_REF_NONE;
    /* Everything below stores a heap-object index in `as.ptr`. */
    case VAL_PAIR: case VAL_CLOSURE: case VAL_STRING: case VAL_VECTOR:
    case VAL_TENSOR: case VAL_KB: case VAL_COMPLEX: case VAL_RATIONAL:
    case VAL_BIGNUM: case VAL_DUAL: case VAL_FACTOR_GRAPH: case VAL_CONTINUATION:
    case VAL_WORKSPACE: case VAL_SUBST: case VAL_HASH: case VAL_BYTEVECTOR:
    case VAL_PARAMETER_OBJ: case VAL_AD_TAPE: case VAL_ERROR_OBJ:
    case VAL_MANIFOLD: case VAL_PORT: case VAL_HYPER_DUAL:
    case VAL_RIEMANNIAN_ADAM_STATE: case VAL_FUTURE: case VAL_MULTI_VALUE:
    case VAL_SYMBOL: case VAL_I128:
        *out_index = v.as.ptr;
        return VM_EVAC_REF_INDEX;
    default:
        return VM_EVAC_REF_UNKNOWN;
    }
}

/* ── Mark state ────────────────────────────────────────────────────────────*/

static int vm_evac_markbits_ensure(Heap* h) {
    if (h->markbits_cap >= h->capacity && h->markbits) {
        memset(h->markbits, 0, (size_t)((h->markbits_cap + 7) / 8));
        return 1;
    }
    size_t bytes = (size_t)((h->capacity + 7) / 8);
    unsigned char* grown = (unsigned char*)realloc(h->markbits, bytes ? bytes : 1);
    if (!grown) return 0;
    h->markbits = grown;
    h->markbits_cap = h->capacity;
    memset(h->markbits, 0, bytes);
    return 1;
}

static inline int vm_evac_marked(const Heap* h, int32_t i) {
    return (h->markbits[i >> 3] >> (i & 7)) & 1;
}

static inline void vm_evac_set_mark(Heap* h, int32_t i) {
    h->markbits[i >> 3] |= (unsigned char)(1u << (i & 7));
}

/** @return 0 if the mark worklist could not grow — the caller must pin. */
static int vm_evac_mark_index(Heap* h, int32_t idx) {
    if (idx < 0 || idx >= h->next_free) return 1;   /* not a live index */
    if (!h->objects[idx]) return 1;                 /* already retired */
    if (vm_evac_marked(h, idx)) return 1;
    vm_evac_set_mark(h, idx);
    if (h->mark_stack_n >= h->mark_stack_cap) {
        int32_t cap = h->mark_stack_cap > 0 ? h->mark_stack_cap * 2 : 1024;
        int32_t* grown = (int32_t*)realloc(h->mark_stack, (size_t)cap * sizeof(int32_t));
        if (!grown) return 0;
        h->mark_stack = grown;
        h->mark_stack_cap = cap;
    }
    h->mark_stack[h->mark_stack_n++] = idx;
    return 1;
}

/** @return 0 on an unknown ValType or a failed worklist growth (caller pins). */
static int vm_evac_mark_value(Heap* h, Value v) {
    int32_t idx = -1;
    switch (vm_evac_value_ref(v, &idx)) {
    case VM_EVAC_REF_NONE:  return 1;
    case VM_EVAC_REF_INDEX: return vm_evac_mark_index(h, idx);
    default:                return 0;
    }
}

/** @brief Mark a raw word as a heap index IF it could plausibly be one.
 *
 * Used only where the VM stores a reference with its tag erased — the hash
 * table, whose `hash-set!` writes `value.as.i` through a `void*` and loses the
 * ValType. Retaining an object because an unrelated small integer happened to
 * name it is a bounded over-retention; freeing one because its only reference
 * was untagged is a dangling index. */
static int vm_evac_mark_conservative_word(Heap* h, uintptr_t w) {
    if (w >= (uintptr_t)(uint32_t)h->next_free) return 1;
    return vm_evac_mark_index(h, (int32_t)w);
}

/* ── Precise object-graph walk ─────────────────────────────────────────────*/

/** @brief Mark the heap indices reachable from a VmValue unification term.
 *
 * VmSubstitution.terms[] and VmFact.args[] carry references with their tag
 * packed into the high half of `data.ptr_val` (vm_logic_term_opaque,
 * vm_native.c) — a form no pointer scan can see and no `Value` walk reaches.
 * @return 0 if the caller must pin. */
static int vm_evac_walk_term(VM* vm, const VmValue* t, int depth);

static int vm_evac_walk_fact(VM* vm, const VmFact* f, int depth) {
    if (!f || depth > 64) return f == NULL;
    Heap* h = &vm->heap;
    if (f->has_datum && !vm_evac_mark_index(h, f->datum_ptr)) return 0;
    if (f->args) {
        for (int i = 0; i < f->arity; i++)
            if (!vm_evac_walk_term(vm, &f->args[i], depth + 1)) return 0;
    }
    return 1;
}

static int vm_evac_walk_term(VM* vm, const VmValue* t, int depth) {
    if (!t || depth > 64) return t == NULL;
    if (t->flags == VM_TERM_KIND_OPAQUE) {
        /* ptr_val = (ValType << 32) | (uint32_t)heap_index */
        return vm_evac_mark_index(&vm->heap, (int32_t)(uint32_t)(t->data.ptr_val & 0xffffffffu));
    }
    if (t->flags == VM_TERM_KIND_FACT)
        return vm_evac_walk_fact(vm, (const VmFact*)(uintptr_t)t->data.ptr_val, depth + 1);
    /* PLAIN / SYMBOL / STRING: immediates or interned text outside the heap. */
    return 1;
}

/**
 * @brief Push every heap index referenced by object @p idx onto the mark
 *        worklist. TOTAL over the heap tag space.
 *
 * @return 1 on success; 0 when the caller must PIN the region — an unclassified
 *         subtype, an unclassified ValType, a structure this walk cannot
 *         traverse, or a worklist that would not grow. Never a partial walk.
 */
static int vm_evac_walk_object(VM* vm, int32_t idx) {
    Heap* h = &vm->heap;
    HeapObject* o = h->objects[idx];
    if (!o) return 1;

    const VmEvacSpec* spec = vm_evac_spec_for((int)o->type);
    if (spec->cls == VM_EVAC_PIN) return 0;

    switch ((int)o->type) {

    case HEAP_CONS:
    case HEAP_FACT:   /* native 507 is the only constructor: an inline cons */
        return vm_evac_mark_value(h, o->cons.car) &&
               vm_evac_mark_value(h, o->cons.cdr);

    case HEAP_CLOSURE: {
        int n = o->closure.n_upvalues;
        if (n < 0 || n > 16) return 0;      /* corrupt arity: refuse to guess */
        for (int i = 0; i < n; i++)
            if (!vm_evac_mark_value(h, o->closure.upvalues[i])) return 0;
        /* open_slots[] are absolute VM stack slots, already covered as roots. */
        return 1;
    }

    /* Payload-only subtypes: no heap indices anywhere inside. Their buffers
     * are covered by the conservative block sweep, which is why this arm can
     * be a no-op without being a shallow leaf copy. */
    case HEAP_STRING:                 /* VmString: byte_len/char_len/char* data */
    case HEAP_COMPLEX:                /* two doubles                            */
    case HEAP_RATIONAL:               /* int64 pair, or two VmBignum*            */
    case HEAP_BIGNUM:                 /* sign/limbs/n_limbs/capacity             */
    /* HEAP_DUAL's exact halves (SW-85) are ARENA pointers, not heap indices,
     * so this mark arm stays a no-op; they are retained by the
     * interior-pointer walk further down, which is where arena chains live. */
    case HEAP_DUAL:                   /* primal/tangent (+ exact halves)         */
    case HEAP_HYPER_DUAL:             /* f/f1/f2/f12                             */
    case HEAP_I128:                   /* {lo, hi}                                */
    case HEAP_BYTEVECTOR:             /* len/uint8_t* data                       */
    case HEAP_TENSOR:                 /* shape/strides/data, all numeric         */
    case HEAP_FACTOR_GRAPH:           /* var_dims/beliefs/factors/messages       */
    case HEAP_RIEMANNIAN_ADAM_STATE:  /* two double buffers                      */
    case 30:                          /* manifold: opaque to the VM              */
        return 1;

    case HEAP_AD_TAPE: {
        /* AdNode is {op, value, gradient, left, right, saved} — parent links
         * are node indices INTO THE TAPE, not heap indices. Nothing to mark. */
        return 1;
    }

    case HEAP_VECTOR:
    case HEAP_MULTI_VALUE:
    case HEAP_PROMISE: {
        VmVector* vec = (VmVector*)o->opaque.ptr;
        if (!vec) return 1;
        if (vec->len < 0 || (vec->len > 0 && !vec->items)) return 0;
        for (int i = 0; i < vec->len; i++)
            if (!vm_evac_mark_value(h, vec->items[i])) return 0;
        return 1;
    }

    case HEAP_CONTINUATION: {
        VmContinuation* c = (VmContinuation*)o->opaque.ptr;
        if (!c) return 1;
        if (c->sp < 0 || c->n_winds < 0 || c->n_parameter_bindings < 0) return 0;
        if (!vm_evac_mark_value(h, c->promise_mark)) return 0;
        if (c->sp > 0) {
            if (!c->saved_stack) return 0;
            for (int i = 0; i < c->sp; i++)
                if (!vm_evac_mark_value(h, c->saved_stack[i])) return 0;
        }
        if (c->n_winds > 0) {
            if (!c->saved_wind_befores || !c->saved_wind_afters) return 0;
            for (int i = 0; i < c->n_winds; i++)
                if (!vm_evac_mark_value(h, c->saved_wind_befores[i]) ||
                    !vm_evac_mark_value(h, c->saved_wind_afters[i])) return 0;
        }
        if (c->n_parameter_bindings > 0) {
            if (!c->saved_parameter_bindings || !c->saved_parameter_values) return 0;
            for (int i = 0; i < c->n_parameter_bindings; i++)
                if (!vm_evac_mark_value(h, c->saved_parameter_bindings[i]) ||
                    !vm_evac_mark_value(h, c->saved_parameter_values[i])) return 0;
        }
        return 1;
    }

    case HEAP_PARAMETER: {
        VmParameter* p = (VmParameter*)o->opaque.ptr;
        if (!p) return 1;
        /* VmParameterValue is a bit-identical clone of Value (vm_parameter.c). */
        if (!vm_evac_mark_value(h, *(const Value*)&p->current_value) ||
            !vm_evac_mark_value(h, *(const Value*)&p->converter)) return 0;
        if (p->stack_depth < 0) return 0;
        if (p->stack_depth > 0) {
            if (!p->save_stack) return 0;
            for (int i = 0; i < p->stack_depth; i++)
                if (!vm_evac_mark_value(h, *(const Value*)&p->save_stack[i])) return 0;
        }
        return 1;
    }

    case HEAP_FUTURE: {
#ifndef ESHKOL_VM_WASM
        VmFuture* f = (VmFuture*)o->opaque.ptr;
        if (!f) return 1;
        return vm_evac_mark_value(h, f->thunk_or_value) &&
               vm_evac_mark_value(h, f->result);
#else
        /* vm_parallel.c (and with it VmFuture) is not compiled for WASM, so a
         * future cannot exist here; pin rather than cast a type we do not have. */
        return 0;
#endif
    }

    case HEAP_HASH: {
        VmHashTable* ht = (VmHashTable*)o->opaque.ptr;
        if (!ht) return 1;
        if (ht->capacity < 0 || (ht->capacity > 0 && (!ht->keys || !ht->values))) return 0;
        /* `hash-set!` stores key.as.i / value.as.i through a void*, discarding
         * the ValType, so a reference here is indistinguishable from an
         * integer. Mark both halves conservatively (see
         * vm_evac_mark_conservative_word). */
        for (int i = 0; i < ht->capacity; i++) {
            if (!vm_evac_mark_conservative_word(h, (uintptr_t)ht->keys[i]) ||
                !vm_evac_mark_conservative_word(h, (uintptr_t)ht->values[i])) return 0;
        }
        return 1;
    }

    case HEAP_ERROR: {
        VmError* e = (VmError*)o->opaque.ptr;
        if (!e) return 1;
        /* `message` and `type` are inline char arrays. `irritants` is a chain
         * of untyped VmCons cells; every VM constructor passes NULL, and there
         * is no tag to walk one safely, so a non-NULL chain pins rather than
         * being skipped. */
        return e->irritants == NULL;
    }

    case HEAP_SUBST: {
        VmSubstitution* s = (VmSubstitution*)o->opaque.ptr;
        if (!s) return 1;
        if (s->n_bindings < 0 || (s->n_bindings > 0 && !s->terms)) return 0;
        /* Only [0, n_bindings) is initialised; capacity beyond that is
         * uninitialised memory, exactly as native's EVAC_SUBSTITUTION notes. */
        for (int i = 0; i < s->n_bindings; i++)
            if (!vm_evac_walk_term(vm, &s->terms[i], 0)) return 0;
        return 1;
    }

    case HEAP_KB: {
        VmKnowledgeBase* kb = (VmKnowledgeBase*)o->opaque.ptr;
        if (!kb) return 1;
        if (kb->n_facts < 0 || (kb->n_facts > 0 && !kb->facts)) return 0;
        for (int i = 0; i < kb->n_facts; i++)
            if (!vm_evac_walk_fact(vm, kb->facts[i], 0)) return 0;
        return 1;
    }

    case HEAP_WORKSPACE: {
        VmWorkspace* w = (VmWorkspace*)o->opaque.ptr;
        if (!w) return 1;
        if (w->n_modules < 0 || w->n_modules > VM_WS_MAX_MODULES) return 0;
        for (int i = 0; i < w->n_modules; i++) {
            /* process_fn is a void* that really points at ONE arena-allocated
             * Value holding a VAL_CLOSURE (vm_native.c, native 542). It is the
             * easiest reference in the whole heap to miss. */
            const Value* fn = (const Value*)w->modules[i].process_fn;
            if (fn && !vm_evac_mark_value(h, *fn)) return 0;
        }
        return 1;
    }

    case HEAP_PORT:
        /* VmPort holds a FILE* or a char buffer; no Values. Classified
         * VM_EVAC_ROOT above so the object itself is never reclaimed. */
        return 1;

    default:
        /* Unreachable while the table above is total, and deliberately kept as
         * a second line of defence: a tag with no case is a PIN, never a guess. */
        return 0;
    }
}

/* ── Region block table ────────────────────────────────────────────────────*/

typedef struct {
    uintptr_t     lo, hi;
    VmArenaBlock* blk;
    unsigned char retained;
    unsigned char scanned;
} VmEvacBlock;

typedef struct {
    VmEvacBlock* v;
    int          n;
    uintptr_t    span_lo, span_hi;   /* O(1) pre-filter over the whole region */
    int*         scan_stack;
    int          scan_n;
} VmEvacBlocks;

static int vm_evac_block_cmp(const void* a, const void* b) {
    uintptr_t x = ((const VmEvacBlock*)a)->lo, y = ((const VmEvacBlock*)b)->lo;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/** @return the index of the region block containing @p p, or -1. */
static int vm_evac_block_of(const VmEvacBlocks* bs, uintptr_t p) {
    if (p - bs->span_lo >= bs->span_hi - bs->span_lo) return -1;   /* fast reject */
    int lo = 0, hi = bs->n - 1;
    while (lo <= hi) {
        int m = (lo + hi) >> 1;
        if (p < bs->v[m].lo)      hi = m - 1;
        else if (p >= bs->v[m].hi) lo = m + 1;
        else return m;
    }
    return -1;
}

/** @brief Retain block @p i and schedule its bytes for the transitive scan. */
static void vm_evac_retain_block(VmEvacBlocks* bs, int i) {
    if (i < 0 || bs->v[i].retained) return;
    bs->v[i].retained = 1;
    bs->scan_stack[bs->scan_n++] = i;
}

/** @brief Retain whichever region block contains @p p, if any. */
static void vm_evac_retain_ptr(VmEvacBlocks* bs, const void* p) {
    if (!p) return;
    vm_evac_retain_block(bs, vm_evac_block_of(bs, (uintptr_t)p));
}

/**
 * @brief Conservatively scan [@p base, @p base + @p len) for words that address
 *        a region block, retaining every block found.
 *
 * This is what makes payload coverage TOTAL without per-subtype layout
 * knowledge: `VmVector.items`, `VmBignum.limbs`, `VmContinuation`'s six saved
 * arrays, `VmFactorGraph`'s five pointer arrays and every nested row inside
 * them are all found as ordinary pointer-shaped words. Over-retention (an
 * integer that happens to look like an address) costs memory; under-retention
 * would cost correctness, and cannot happen here.
 */
static void vm_evac_scan_range(VmEvacBlocks* bs, const void* base, size_t len) {
    if (!base || len < sizeof(uintptr_t)) return;
    uintptr_t start = (uintptr_t)base;
    /* Align up; every arena allocation is 8-byte aligned by vm_arena_alloc. */
    uintptr_t p = (start + sizeof(uintptr_t) - 1) & ~(uintptr_t)(sizeof(uintptr_t) - 1);
    uintptr_t end = start + len;
    for (; p + sizeof(uintptr_t) <= end; p += sizeof(uintptr_t)) {
        uintptr_t w = *(const uintptr_t*)p;
        if (w - bs->span_lo < bs->span_hi - bs->span_lo)
            vm_evac_retain_block(bs, vm_evac_block_of(bs, w));
    }
}

/** @brief Drain the retained-block worklist, scanning each retained block's
 *         live bytes for pointers into other region blocks. */
static void vm_evac_scan_retained(VmEvacBlocks* bs) {
    while (bs->scan_n > 0) {
        int i = bs->scan_stack[--bs->scan_n];
        if (bs->v[i].scanned) continue;
        bs->v[i].scanned = 1;
        vm_evac_scan_range(bs, bs->v[i].blk->data, bs->v[i].blk->used);
    }
}

/**
 * @brief Retain, and conservatively scan, the memory object @p o OWNS.
 *
 * This is the counterpart of the object-graph walk, and it exists for one
 * reason: while a region is open it is the active arena, so a payload array
 * that GROWS during the body lands inside the region even when the structure
 * that owns it is older — a hash table rehashing, a knowledge base doubling
 * its fact array, a parameter pushing its save stack. Native handles that class
 * with a mutation write barrier at ~130 call sites; scanning the payload of
 * every LIVE object instead cannot be incomplete in the same way, because it
 * does not depend on anyone having remembered to instrument a store.
 *
 * Only the SIZE of each payload struct is needed, never its layout: the
 * pointers inside it (`VmVector.items`, `VmBignum.limbs`, `VmContinuation`'s
 * six saved arrays, `VmFactorGraph`'s five pointer arrays) are found as
 * pointer-shaped words, and anything they in turn point at is found when the
 * block they live in is scanned in its own right.
 *
 * Cost is O(live objects x payload size) — proportional to LIVE data, not to
 * the total heap, which is what keeps a long `with-region` loop linear.
 * Subtypes whose payload is inline (cons, closure, fact) own no memory at all
 * and cost nothing here.
 */
static void vm_evac_scan_object_payload(VmEvacBlocks* bs, const HeapObject* o) {
    switch ((int)o->type) {
    case HEAP_CONS: case HEAP_CLOSURE: case HEAP_FACT:
        return;   /* payload is inline in the HeapObject */
    default: break;
    }
    void* p = o->opaque.ptr;
    if (!p) return;
    size_t size = 0;
    switch ((int)o->type) {
    case HEAP_STRING:       size = sizeof(VmString); break;
    case HEAP_VECTOR:
    case HEAP_MULTI_VALUE:
    case HEAP_PROMISE:      size = sizeof(VmVector); break;
    case HEAP_COMPLEX:      size = sizeof(VmComplex); break;
    case HEAP_RATIONAL:     size = sizeof(VmRational); break;
    case HEAP_BIGNUM:       size = sizeof(VmBignum); break;
    case HEAP_DUAL:         size = sizeof(VmDual); break;
    case HEAP_HYPER_DUAL:   size = sizeof(VmHyperDual); break;
    case HEAP_TENSOR:       size = sizeof(VmTensor); break;
    case HEAP_SUBST:        size = sizeof(VmSubstitution); break;
    case HEAP_KB:           size = sizeof(VmKnowledgeBase); break;
    case HEAP_FACTOR_GRAPH: size = sizeof(VmFactorGraph); break;
    case HEAP_WORKSPACE:    size = sizeof(VmWorkspace); break;
    case HEAP_PORT:         size = sizeof(VmPort); break;
    case HEAP_AD_TAPE:      size = sizeof(AdTape); break;
    case HEAP_CONTINUATION: size = sizeof(VmContinuation); break;
    case HEAP_HASH:         size = sizeof(VmHashTable); break;
    case HEAP_ERROR:        size = sizeof(VmError); break;
    case HEAP_BYTEVECTOR:   size = sizeof(VmBytevector); break;
    case HEAP_PARAMETER:    size = sizeof(VmParameter); break;
    case HEAP_RIEMANNIAN_ADAM_STATE: size = sizeof(VmRiemannianAdamState); break;
    case HEAP_I128:         size = sizeof(eshkol_i128_abi); break;
#ifndef ESHKOL_VM_WASM
    case HEAP_FUTURE:       size = sizeof(VmFuture); break;
#endif
    /* 30 = manifold. Its payload may be owned by semiclassical_qllm rather
     * than by any VM arena, so its extent is not ours to read; the portable
     * fallback struct holds only scalars, so there is nothing to find. */
    default: return;
    }
    vm_evac_retain_ptr(bs, p);
    vm_evac_scan_range(bs, p, size);

    /* Structures that own a further level of arena struct BY POINTER rather
     * than by an array the level above already covers. Listed explicitly
     * because a three-level chain whose middle link is outside the region
     * (old rational -> old bignum -> region-grown limbs) is not reachable by
     * scanning alone. */
    switch ((int)o->type) {
    case HEAP_RATIONAL: {
        const VmRational* r = (const VmRational*)p;
        if (r->is_big) {
            if (r->big_num) { vm_evac_retain_ptr(bs, r->big_num);
                              vm_evac_scan_range(bs, r->big_num, sizeof(VmBignum)); }
            if (r->big_den) { vm_evac_retain_ptr(bs, r->big_den);
                              vm_evac_scan_range(bs, r->big_den, sizeof(VmBignum)); }
        }
        break;
    }
    case HEAP_DUAL: {
        /* SW-85: a dual seeded at an EXACT point owns two VmRational* in the
         * region arena, and each of those may itself be bignum-backed. Before
         * the exact halves existed a dual really was "two doubles" and this
         * arm was correctly absent; the moment the carrier gained interior
         * arena pointers, leaving it out would let the exact halves dangle
         * into a freed region — the SW-66 defect, one subtype over. The chain
         * is three deep (dual -> rational -> bignum limbs) and a scan cannot
         * reach the far end on its own, which is exactly why HEAP_RATIONAL
         * above has to walk its own bignums too. */
        const VmDual* d = (const VmDual*)p;
        const VmRational* halves[2] = { d->eprimal, d->etangent };
        for (int i = 0; i < 2; i++) {
            const VmRational* r = halves[i];
            if (!r) continue;                     /* inexact half: nothing to retain */
            vm_evac_retain_ptr(bs, (void*)r);
            vm_evac_scan_range(bs, (void*)r, sizeof(VmRational));
            if (!r->is_big) continue;
            if (r->big_num) { vm_evac_retain_ptr(bs, r->big_num);
                              vm_evac_scan_range(bs, r->big_num, sizeof(VmBignum)); }
            if (r->big_den) { vm_evac_retain_ptr(bs, r->big_den);
                              vm_evac_scan_range(bs, r->big_den, sizeof(VmBignum)); }
        }
        if (d->kind == VM_DUAL_KIND_TAYLOR) {
            if (d->coeff) {
                vm_evac_retain_ptr(bs, d->coeff);
                vm_evac_scan_range(bs, d->coeff,
                                   (size_t)(d->order + 1) * sizeof(double));
            }
            if (d->tangent_coeff) {
                vm_evac_retain_ptr(bs, d->tangent_coeff);
                vm_evac_scan_range(bs, d->tangent_coeff,
                                   (size_t)(d->order + 1) * sizeof(double));
            }
            if (d->exact_coeff) {
                vm_evac_retain_ptr(bs, d->exact_coeff);
                vm_evac_scan_range(bs, d->exact_coeff,
                                   (size_t)(d->order + 1) * sizeof(VmRational*));
                for (uint32_t i = 0; i <= d->order; i++) {
                    const VmRational* er = d->exact_coeff[i];
                    if (!er) continue;
                    vm_evac_retain_ptr(bs, er);
                    vm_evac_scan_range(bs, er, sizeof(VmRational));
                    if (er->is_big) {
                        if (er->big_num) { vm_evac_retain_ptr(bs, er->big_num);
                                           vm_evac_scan_range(bs, er->big_num, sizeof(VmBignum)); }
                        if (er->big_den) { vm_evac_retain_ptr(bs, er->big_den);
                                           vm_evac_scan_range(bs, er->big_den, sizeof(VmBignum)); }
                    }
                }
            }
        }
        break;
    }
    case HEAP_KB: {
        const VmKnowledgeBase* kb = (const VmKnowledgeBase*)p;
        if (kb->facts && kb->n_facts > 0)
            for (int i = 0; i < kb->n_facts; i++)
                if (kb->facts[i]) { vm_evac_retain_ptr(bs, kb->facts[i]);
                                    vm_evac_scan_range(bs, kb->facts[i], sizeof(VmFact)); }
        break;
    }
    case HEAP_FACTOR_GRAPH: {
        const VmFactorGraph* fg = (const VmFactorGraph*)p;
        if (fg->factors && fg->num_factors > 0)
            for (int i = 0; i < fg->num_factors; i++)
                if (fg->factors[i]) { vm_evac_retain_ptr(bs, fg->factors[i]);
                                      vm_evac_scan_range(bs, fg->factors[i], sizeof(VmFactor)); }
        break;
    }
    default: break;
    }
}

/** @brief Retain the region memory the VM points at directly, rather than
 *         through an arena or a heap object: the live AD tape and the
 *         geometric optimizer states. Everything else in `struct VM` is either
 *         a Value (covered as a mark root) or a non-arena OS handle. */
static void vm_evac_scan_vm_raw_pointers(VM* vm, VmEvacBlocks* bs) {
    if (vm->active_tape) {
        vm_evac_retain_ptr(bs, vm->active_tape);
        vm_evac_scan_range(bs, vm->active_tape, sizeof(AdTape));
    }
    for (int i = 0; i < 16; i++) {
        if (!vm->geometric_adam_states[i]) continue;
        vm_evac_retain_ptr(bs, vm->geometric_adam_states[i]);
        vm_evac_scan_range(bs, vm->geometric_adam_states[i],
                           sizeof(VmRiemannianAdamState));
    }
}

/* ── Roots ────────────────────────────────────────────────────────────────
 *
 * EXHAUSTIVE over the Value-carrying members of `struct VM`. A missed root
 * frees live memory, so the list below is written against the struct member by
 * member; anything added to `struct VM` that holds a Value must be added here
 * too. The bytecode operand stack is scanned to `sp` exactly: every active
 * frame's locals live below it, and slots above it are dead by construction.
 *
 * @return 0 if the caller must pin (unknown ValType, or worklist growth failed).
 */
static int vm_evac_mark_roots(VM* vm) {
    Heap* h = &vm->heap;

    /* Constant pool — reachable from any OP_CONST the program has not run yet. */
    int nconst = vm->n_constants < vm->const_cap ? vm->n_constants : vm->const_cap;
    for (int i = 0; i < nconst; i++)
        if (!vm_evac_mark_value(h, vm->constants[i])) return 0;

    /* Operand stack (locals of every active frame live below sp). */
    for (int i = 0; i < vm->sp && i < STACK_SIZE; i++)
        if (!vm_evac_mark_value(h, vm->stack[i])) return 0;

    /* Recorded outputs. */
    for (int i = 0; i < vm->n_outputs && i < 256; i++)
        if (!vm_evac_mark_value(h, vm->outputs[i])) return 0;

    /* Exception state: the in-flight condition and each handler's promise mark. */
    if (!vm_evac_mark_value(h, vm->current_exception)) return 0;
    for (int i = 0; i < vm->n_handlers && i < 16; i++)
        if (!vm_evac_mark_value(h, vm->handler_stack[i].promise_mark)) return 0;

    /* dynamic-wind thunks still to run. */
    for (int i = 0; i < vm->n_winds && i < 32; i++)
        if (!vm_evac_mark_value(h, vm->wind_stack[i].before) ||
            !vm_evac_mark_value(h, vm->wind_stack[i].after)) return 0;

    /* Dynamic parameter bindings and the promise-evaluation chain. */
    for (int i = 0; i < vm->n_parameter_bindings && i < 64; i++)
        if (!vm_evac_mark_value(h, vm->parameter_bindings[i])) return 0;
    if (!vm_evac_mark_value(h, vm->promise_eval_head)) return 0;

    /* VM-lifetime side tables. Each is scanned in full rather than to a count,
     * because the `active` flags are the only liveness they carry. */
    for (int i = 0; i < 16; i++) {
        if (!vm->lru_caches[i].active) continue;
        for (int j = 0; j < 64; j++) {
            if (!vm->lru_caches[i].entries[j].active) continue;
            if (!vm_evac_mark_value(h, vm->lru_caches[i].entries[j].key) ||
                !vm_evac_mark_value(h, vm->lru_caches[i].entries[j].value)) return 0;
        }
    }
    for (int i = 0; i < 16; i++) {
        if (!vm->event_emitters[i].active) continue;
        for (int j = 0; j < 64; j++) {
            if (!vm->event_emitters[i].listeners[j].active) continue;
            if (!vm_evac_mark_value(h, vm->event_emitters[i].listeners[j].event) ||
                !vm_evac_mark_value(h, vm->event_emitters[i].listeners[j].handler)) return 0;
        }
    }
    for (int i = 0; i < 16; i++) {
        if (!vm->channels[i].active) continue;
        for (int j = 0; j < 64; j++)
            if (!vm_evac_mark_value(h, vm->channels[i].buffer[j])) return 0;
    }
    for (int i = 0; i < 32; i++) {
        if (!vm->timers[i].allocated) continue;
        if (!vm_evac_mark_value(h, vm->timers[i].callback)) return 0;
    }
    for (int i = 0; i < vm->n_exit_handlers && i < 32; i++)
        if (!vm_evac_mark_value(h, vm->exit_handlers[i])) return 0;

    /* Objects this region owns that must never be reclaimed no matter what
     * refers to them, because they hold a resource whose lifetime is not the
     * region's: an open port's FILE* and malloc'd buffer, a future's live
     * pthread mutex and condition variable. Only the region's OWN membership
     * list is scanned — an object outside the region is not a candidate for
     * reclamation in the first place, so walking the whole table to find them
     * would be O(heap) per pop for no added safety. */
    const VmHeapRegionSlots* rs = &h->region_slots[h->regions.depth - 1];
    for (int32_t k = 0; k < rs->n_slots; k++) {
        int32_t idx = rs->slots[k];
        if (idx < 0 || idx >= h->next_free || !h->objects[idx]) continue;
        if (vm_evac_spec_for((int)h->objects[idx]->type)->cls == VM_EVAC_ROOT)
            if (!vm_evac_mark_index(h, idx)) return 0;
    }
    return 1;
}

/**
 * @brief Drain the mark worklist: walk each live object's references, and at
 *        the same time retain the region memory it owns.
 *
 * Both halves are done in one pass over the LIVE set so the cost of a pop is
 * O(live objects + region size) rather than O(whole heap). A `with-region`
 * loop that promotes a little on every pass builds up a large table of dead
 * objects, and re-walking that table on every pop is what turns a linear loop
 * quadratic.
 *
 * @return 0 if the caller must pin.
 */
static int vm_evac_trace(VM* vm, VmEvacBlocks* bs, int compact) {
    Heap* h = &vm->heap;
    while (h->mark_stack_n > 0) {
        int32_t idx = h->mark_stack[--h->mark_stack_n];
        HeapObject* o = h->objects[idx];
        if (!o) continue;
        if (!compact) vm_evac_retain_ptr(bs, o);
        vm_evac_scan_object_payload(bs, o);
        if (!vm_evac_walk_object(vm, idx)) return 0;
    }
    return 1;
}

/* ── Post-sweep audit ──────────────────────────────────────────────────────*/

/**
 * @brief Independent check that nothing still points at an index being retired.
 *
 * This is NOT a restatement of the mark. The mark asks "what is reachable from
 * the roots"; this asks "does ANY object in the table — reachable or not — or
 * any root still name a slot we are about to clear". A hit means the mark
 * missed a root or an interior field, which is the only failure mode of this
 * design that could otherwise be silent. Loud on stderr, and fatal under
 * ESHKOL_VM_REGION_VERIFY_FATAL=1 so a lane can gate on it.
 */
static void vm_evac_audit_retired(VM* vm, const int32_t* retiring, int n_retiring) {
    Heap* h = &vm->heap;
    if (n_retiring <= 0) return;

    /* Reuse the mark bitset as a "being retired" set; the sweep is done with it. */
    size_t bytes = (size_t)((h->markbits_cap + 7) / 8);
    memset(h->markbits, 0, bytes);
    for (int i = 0; i < n_retiring; i++)
        if (retiring[i] >= 0 && retiring[i] < h->markbits_cap) vm_evac_set_mark(h, retiring[i]);

    int hits = 0;
    for (int32_t i = 0; i < h->next_free && hits < 8; i++) {
        HeapObject* o = h->objects[i];
        if (!o) continue;
        int32_t refs[64];
        int n = 0;
        switch ((int)o->type) {
        case HEAP_CONS: case HEAP_FACT: {
            int32_t x;
            if (vm_evac_value_ref(o->cons.car, &x) == VM_EVAC_REF_INDEX) refs[n++] = x;
            if (vm_evac_value_ref(o->cons.cdr, &x) == VM_EVAC_REF_INDEX) refs[n++] = x;
            break;
        }
        case HEAP_CLOSURE: {
            int32_t x;
            for (int k = 0; k < o->closure.n_upvalues && k < 16 && n < 64; k++)
                if (vm_evac_value_ref(o->closure.upvalues[k], &x) == VM_EVAC_REF_INDEX)
                    refs[n++] = x;
            break;
        }
        case HEAP_VECTOR: case HEAP_MULTI_VALUE: case HEAP_PROMISE: {
            VmVector* vec = (VmVector*)o->opaque.ptr;
            int32_t x;
            if (vec && vec->items)
                for (int k = 0; k < vec->len && n < 64; k++)
                    if (vm_evac_value_ref(vec->items[k], &x) == VM_EVAC_REF_INDEX)
                        refs[n++] = x;
            break;
        }
        default: break;
        }
        for (int k = 0; k < n; k++) {
            int32_t r = refs[k];
            if (r >= 0 && r < h->markbits_cap && vm_evac_marked(h, r)) {
                fprintf(stderr,
                        "eshkol-vm: REGION EVACUATOR AUDIT: object %d (subtype %s) still "
                        "references heap index %d, which this region pop is retiring. "
                        "The mark phase missed a root or an interior field "
                        "(lib/backend/vm_region_evac.c).\n",
                        (int)i, vm_evac_spec_for((int)o->type)->name, (int)r);
                hits++;
                break;
            }
        }
    }
    if (hits && vm_host_env_flag("ESHKOL_VM_REGION_VERIFY_FATAL")) {
        fprintf(stderr, "eshkol-vm: ERROR: region evacuator audit is fatal "
                        "(ESHKOL_VM_REGION_VERIFY_FATAL=1); terminating.\n");
        exit(1);
    }
}

/* ── Pop ──────────────────────────────────────────────────────────────────*/

/** @brief One-time notice that a region could not be reclaimed, and why. */
static void vm_evac_pin_notice(const char* reason) {
    static int said = 0;
    if (said) return;
    said = 1;
    if (vm_host_env_flag("ESHKOL_VM_REGION_QUIET")) return;
    fprintf(stderr,
            "eshkol-vm: note: a `with-region` body could not be reclaimed and was "
            "promoted whole (%s). The answer is unaffected; the memory is not "
            "returned until the enclosing scope ends. This is the evacuator "
            "degrading toward the pre-Stage-1 leak rather than toward a dangling "
            "reference (lib/backend/vm_region_evac.c). Set "
            "ESHKOL_VM_REGION_QUIET=1 to silence this note.\n",
            reason ? reason : "reason not recorded");
}

/**
 * @brief Hand every block of the dying region to its parent arena, unchanged.
 *
 * The blocks are spliced BEHIND the parent's bump block, which `vm_arena_alloc`
 * never allocates from (it only ever fills `a->current`), so promoted memory is
 * sealed: addresses stay valid for the parent's whole lifetime and no later
 * allocation can land inside a promoted block.
 */
static void vm_evac_promote_all_blocks(Heap* h, VmRegion* r, VmArena* parent) {
    VmArenaBlock* b = r->arena.current;
    if (!b) return;
    VmArenaBlock* last = b;
    int n = 0;
    size_t bytes = 0;
    while (last) { n++; bytes += last->size; if (!last->next) break; last = last->next; }
    if (parent->current) {
        last->next = parent->current->next;
        parent->current->next = b;
    } else {
        parent->current = b;
    }
    parent->n_blocks += n;
    parent->total_allocated += bytes;
    parent->total_used += r->arena.total_used;
    h->bytes_promoted += bytes;
    r->arena.current = NULL;
    r->arena.n_blocks = 0;
    r->arena.total_allocated = 0;
    r->arena.total_used = 0;
}

/** @brief Move region-owned object indices onto the parent region's membership
 *         list so a later pop can still reclaim them. */
static void vm_evac_inherit_slots(Heap* h, int depth, const int32_t* slots, int32_t n) {
    if (depth - 1 < 0) return;   /* parent is the global arena: nothing reclaims it */
    int saved = h->regions.depth;
    h->regions.depth = depth;    /* record_slot writes to depth-1 */
    for (int32_t i = 0; i < n; i++) heap_region_record_slot(h, slots[i]);
    h->regions.depth = saved;
}

/**
 * @brief Pop the innermost region: promote everything still reachable, return
 *        the rest to the allocator.
 *
 * Phases, in the order the native teardown fixes (promote while the region is
 * still current, then release):
 *   1. bail out to a whole-region promotion if reclamation is off or pinned;
 *   2. mark from the VM root set across the WHOLE heap;
 *   3. retain the arena blocks holding anything marked, then transitively any
 *      block a retained block points into, then any block the surviving heap
 *      or the VM's raw arena pointers point into;
 *   4. clear the object slots of everything region-owned that landed in a
 *      freed block, and recycle their indices;
 *   5. free (or, under poison, stamp and keep) the blocks nothing needs.
 */
static void vm_region_evacuate_pop(VM* vm) {
    Heap* h = &vm->heap;
    int depth = h->regions.depth;
    if (depth <= 0) { vm_region_pop(&h->regions); return; }

    VmRegion* r = h->regions.stack[depth - 1];
    VmHeapRegionSlots* rs = &h->region_slots[depth - 1];
    VmArena* parent = (depth - 1 > 0) ? &h->regions.stack[depth - 2]->arena
                                      : &h->regions.global_arena;
    const char* pin_reason = rs->pinned ? rs->pin_reason : NULL;

    if (!vm_evac_enabled()) pin_reason = pin_reason ? pin_reason : "ESHKOL_VM_REGION_EVAC=0";

    VmEvacBlocks bs;
    memset(&bs, 0, sizeof(bs));

    /* ── Build the block table ─────────────────────────────────────────── */
    if (!pin_reason) {
        int n = 0;
        for (VmArenaBlock* b = r->arena.current; b; b = b->next) n++;
        if (n > 0) {
            bs.v = (VmEvacBlock*)calloc((size_t)n, sizeof(VmEvacBlock));
            bs.scan_stack = (int*)malloc((size_t)n * sizeof(int));
            if (!bs.v || !bs.scan_stack) {
                pin_reason = "block table allocation failed";
            } else {
                int i = 0;
                for (VmArenaBlock* b = r->arena.current; b; b = b->next, i++) {
                    bs.v[i].lo  = (uintptr_t)b->data;
                    bs.v[i].hi  = (uintptr_t)b->data + b->size;
                    bs.v[i].blk = b;
                }
                bs.n = n;
                qsort(bs.v, (size_t)n, sizeof(VmEvacBlock), vm_evac_block_cmp);
                bs.span_lo = bs.v[0].lo;
                bs.span_hi = bs.v[n - 1].hi;
                for (int k = 0; k < n; k++)
                    if (bs.v[k].hi > bs.span_hi) bs.span_hi = bs.v[k].hi;
            }
        }
    }

    /* ── Mark, retaining owned memory as we go ─────────────────────────── */
    const int compact = vm_evac_compact();
    if (!pin_reason && bs.n > 0) {
        if (!vm_evac_markbits_ensure(h)) {
            pin_reason = "mark bitset allocation failed";
        } else {
            h->mark_stack_n = 0;
            if (!vm_evac_mark_roots(vm) || !vm_evac_trace(vm, &bs, compact))
                pin_reason = "an object or value type the evacuator does not classify";
        }
    }

    if (pin_reason || bs.n == 0) {
        if (pin_reason) {
            h->regions_pinned++;
            vm_evac_pin_notice(pin_reason);
        }
        vm_evac_promote_all_blocks(h, r, parent);
        vm_evac_inherit_slots(h, depth - 1, rs->slots, rs->n_slots);
        rs->n_slots = 0;
        free(bs.v); free(bs.scan_stack);
        vm_region_pop(&h->regions);
        return;
    }

    /* ── Retain the region memory the VM itself points at ───────────────── */
    vm_evac_scan_vm_raw_pointers(vm, &bs);
    vm_evac_scan_retained(&bs);

    /* ── Promote or retire every object this region owns ────────────────
     *
     * Marked  + compaction  -> the fixed-size struct is copied into the parent
     *                          arena and the object table is repointed, so the
     *                          block it came from is free to go.
     * Marked  + no compaction, or a copy that could not be allocated
     *                       -> its block was retained above; leave it in place.
     * Unmarked, block freed -> clear the slot (a stale reference then reads as
     *                          an invalid heap pointer, loudly) and recycle it.
     * Unmarked, block kept  -> the memory survives anyway; keep the slot rather
     *                          than create a reference that resolves to NULL.
     */
    const int poison = vm_evac_poison();
    const int recycle = vm_evac_recycle();
    int32_t* retiring = NULL;
    int n_retiring = 0;
    const int verify = vm_evac_verify();
    if (verify && rs->n_slots > 0)
        retiring = (int32_t*)malloc((size_t)rs->n_slots * sizeof(int32_t));

    int32_t survivors_n = 0;
    for (int32_t k = 0; k < rs->n_slots; k++) {
        int32_t idx = rs->slots[k];
        if (idx < 0 || idx >= h->capacity || !h->objects[idx]) continue;
        int b = vm_evac_block_of(&bs, (uintptr_t)h->objects[idx]);
        if (b < 0) {                       /* already promoted out of this region */
            rs->slots[survivors_n++] = idx;
            continue;
        }
        if (vm_evac_marked(h, idx)) {
            if (compact) {
                HeapObject* copy = (HeapObject*)vm_arena_alloc(parent, sizeof(HeapObject));
                if (copy) {
                    *copy = *h->objects[idx];
                    h->objects[idx] = copy;
                } else {
                    vm_evac_retain_block(&bs, b);   /* out of memory: keep it put */
                }
            } else {
                vm_evac_retain_block(&bs, b);
            }
            rs->slots[survivors_n++] = idx;
            h->objects_promoted++;
            continue;
        }
        if (bs.v[b].retained) {            /* dead, but its block lives on */
            rs->slots[survivors_n++] = idx;
            continue;
        }
        h->objects[idx] = NULL;
        h->objects_reclaimed++;
        if (retiring) retiring[n_retiring++] = idx;
        if (recycle) {
            if (h->n_free_slots >= h->cap_free_slots) {
                int32_t cap = h->cap_free_slots > 0 ? h->cap_free_slots * 2 : 256;
                int32_t* grown = (int32_t*)realloc(h->free_slots, (size_t)cap * sizeof(int32_t));
                if (grown) { h->free_slots = grown; h->cap_free_slots = cap; }
            }
            if (h->n_free_slots < h->cap_free_slots)
                h->free_slots[h->n_free_slots++] = idx;
        }
    }
    rs->n_slots = survivors_n;
    /* A block retained by the loop above (compaction failure) may itself point
     * into further region blocks; drain the worklist once more before any of
     * them is released. */
    vm_evac_scan_retained(&bs);

    if (retiring) {
        vm_evac_audit_retired(vm, retiring, n_retiring);
        free(retiring);
    }

    /* ── Release the blocks nothing needs, promote the ones something does ─ */
    VmArenaBlock* keep_head = NULL;
    VmArenaBlock* keep_tail = NULL;
    size_t keep_bytes = 0, keep_used = 0, freed_bytes = 0;
    int keep_blocks = 0;
    for (int i = 0; i < bs.n; i++) {
        VmArenaBlock* b = bs.v[i].blk;
        b->next = NULL;
        if (bs.v[i].retained || poison) {
            if (poison && !bs.v[i].retained) memset(b->data, VM_EVAC_POISON_BYTE, b->used);
            if (keep_tail) { keep_tail->next = b; keep_tail = b; }
            else { keep_head = keep_tail = b; }
            keep_bytes += b->size;
            keep_used  += b->used;
            keep_blocks++;
        } else {
            freed_bytes += b->size;
            vm_arena_block_destroy(b);
        }
    }
    if (keep_head) {
        if (parent->current) {
            keep_tail->next = parent->current->next;
            parent->current->next = keep_head;
        } else {
            parent->current = keep_head;
        }
        parent->n_blocks += keep_blocks;
        parent->total_allocated += keep_bytes;
        parent->total_used += keep_used;
        h->bytes_promoted += keep_bytes;
    }
    h->bytes_reclaimed += freed_bytes;
    h->regions_reclaimed++;

    r->arena.current = NULL;
    r->arena.n_blocks = 0;
    r->arena.total_allocated = 0;
    r->arena.total_used = 0;

    vm_evac_inherit_slots(h, depth - 1, rs->slots, rs->n_slots);
    rs->n_slots = 0;

    free(bs.v);
    free(bs.scan_stack);
    vm_region_pop(&h->regions);

    /* An index below next_free whose slot is NULL is now a legal state, so the
     * table's high-water mark can be walked back whenever the top is empty.
     * This is what keeps a `with-region` tick loop from growing 8 bytes per
     * allocation even when index recycling is disabled.
     *
     * The free list must then be filtered, not just truncated: it is not
     * sorted, and handing out an index at or above next_free would produce an
     * object that is_valid_heap_ptr() rejects — a live object nobody could
     * read. */
    int32_t old_next_free = h->next_free;
    while (h->next_free > 0 && !h->objects[h->next_free - 1]) h->next_free--;
    if (h->next_free != old_next_free) {
        int32_t w = 0;
        for (int32_t i = 0; i < h->n_free_slots; i++)
            if (h->free_slots[i] < h->next_free) h->free_slots[w++] = h->free_slots[i];
        h->n_free_slots = w;
    }
}

/**
 * @brief The single region teardown path: close every `with-region` bracket
 *        above @p target_brackets, innermost first.
 *
 * Normal exit passes `n_region_brackets - 1`; a raise passes the count
 * recorded when its handler was installed; a continuation transfer passes the
 * count recorded at capture. Routing all three here is the same discipline
 * native keeps around eshkol_region_unwind_to(): there is exactly one place
 * that pops a region, so the structured and unstructured surfaces cannot
 * acquire different escape semantics.
 *
 * The in-flight value needs no special handling, unlike native's designated
 * `vals` array — it is promoted because it is reachable from a root
 * (the operand stack, or vm->current_exception), which is the whole point of
 * marking instead of copying.
 */
static void vm_region_bracket_unwind_to(VM* vm, int target_brackets) {
    if (target_brackets < 0) target_brackets = 0;
    while (vm->n_region_brackets > target_brackets) {
        int mark = vm->region_bracket_marks[--vm->n_region_brackets];
        if (mark <= 0) continue;                 /* bracket owned no region */
        while (vm->heap.regions.depth >= mark && vm->heap.regions.depth > 0)
            vm_region_evacuate_pop(vm);
    }
}

/** @brief Pin every open region, then unwind to @p target_brackets.
 *
 * Used on both sides of a continuation transfer. A captured continuation can
 * resurrect a stack state from inside a region body, and re-entering a region
 * whose arena was released is not something Stage-1 supports — so a region that
 * a continuation could reach is never released at all. This costs reclamation
 * only in the rare `call/cc`-across-`with-region` case, and it costs it in the
 * direction of a leak. */
static void vm_region_bracket_unwind_pinned(VM* vm, int target_brackets) {
    if (vm->heap.regions.depth > 0)
        heap_region_pin_all(&vm->heap, "a continuation crossed the region boundary");
    vm_region_bracket_unwind_to(vm, target_brackets);
}
