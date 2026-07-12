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

    OP_COUNT = 64
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

/** @brief R7RS truthiness: only `#f` is false, everything else (including
 *         '()) is truthy. */
static int is_truthy(Value v) {
    if (v.type == VAL_BOOL) return v.as.b;
    if (v.type == VAL_NIL) return 0;
    return 1;  /* everything else is truthy */
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
 *         (< 1e15 in magnitude) integer, else as a FLOAT Value. */
static Value number_val(double d) {
    if (d == (int64_t)d && fabs(d) < 1e15) return INT_VAL((int64_t)d);
    return FLOAT_VAL(d);
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
} HeapType;

typedef struct {
    HeapType type;
    union {
        struct { Value car; Value cdr; } cons;
        struct { int32_t func_pc; int32_t n_upvalues; Value upvalues[16]; } closure;
        struct { void* ptr; int subtype; } opaque;  /* for complex, rational, tensor, logic, etc. */
    };
} HeapObject;

typedef struct {
    VmRegionStack regions;
    HeapObject** objects;    /* array of pointers to arena-allocated objects */
    int32_t next_free;
    int32_t capacity;
} Heap;

/** @brief Initialize the VM heap: sets up its arena region stack and the
 *         object-pointer table (fixed capacity HEAP_SIZE). */
static void heap_init(Heap* h) {
    vm_region_stack_init(&h->regions);
    h->capacity = HEAP_SIZE;
    h->objects = (HeapObject**)calloc(h->capacity, sizeof(HeapObject*));
    h->next_free = 0;
}

/** @brief Allocate a new (zeroed) HeapObject slot from the arena and
 *         register it in the object table.
 * @return The new object's heap index, or -1 on capacity/allocation
 *         failure.
 */
static int32_t heap_alloc(Heap* h) {
    if (h->next_free >= h->capacity) {
        fprintf(stderr, "HEAP OVERFLOW (max %d objects)\n", h->capacity);
        return -1;
    }
    HeapObject* obj = (HeapObject*)vm_alloc(&h->regions, sizeof(HeapObject));
    if (!obj) { fprintf(stderr, "ARENA OOM\n"); return -1; }
    memset(obj, 0, sizeof(HeapObject));
    h->objects[h->next_free] = obj;
    return h->next_free++;
}

/** @brief Push a new arena region scope onto the heap (see OALR — objects
 *         allocated after this call are freed in bulk by the matching
 *         heap_region_pop()). */
static void heap_region_push(Heap* h) {
    vm_region_push(&h->regions, NULL, 0);
}

/** @brief Pop the most recent arena region scope, bulk-freeing everything
 *         allocated since the matching heap_region_push(). */
static void heap_region_pop(Heap* h) {
    vm_region_pop(&h->regions);
}

/** @brief Tear down the heap's arena region stack and free its object
 *         table. */
static void heap_destroy(Heap* h) {
    vm_region_stack_destroy(&h->regions);
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
    Value constants[MAX_CONSTS];
    int n_constants;

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
    } handler_stack[16];
    int n_handlers;
    Value current_exception;

    /* Dynamic-wind stack */
    struct { Value before; Value after; } wind_stack[32];
    int n_winds;

    /* Dynamic parameter bindings parallel dynamic-wind for VM exception
     * unwinding.  Each entry names a VmParameter whose stack received an
     * actual native 702 push; normal 703 and exceptional exits pop LIFO. */
    Value parameter_bindings[64];
    int n_parameter_bindings;

    /* Status */
    int halted;
    int error;
    int native_policy;

    /* Reverse-mode AD tracing context.
     * When active_tape != NULL, arithmetic operations (+,-,*,/,sin,cos,...)
     * record on the tape. Each stack value that flows through tape-aware ops
     * gets tracked via ad_node_map: maps stack slot → tape node index.
     * This enables transparent reverse-mode gradient computation. */
    void* active_tape;                /* AdTape* or NULL */
    int   ad_node_map[STACK_SIZE];    /* stack slot → tape node index (-1 = not tracked) */

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
        int children[16];
        int flex_direction; /* 0 = column, 1 = row */
        double width;
        double height;
        double flex_grow;
        double flex_shrink;
        double padding;
        double margin;
        double gap;
        double computed_left;
        double computed_top;
        double computed_width;
        double computed_height;
    } yoga_nodes[64];

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
        char language[32];
    } ts_parsers[16];

    struct {
        int active;
        int parser;
        int root_node;
        const char* source;
        int64_t source_len;
        char language[32];
    } ts_trees[16];

    struct {
        int active;
        int tree;
        int parent;
        int64_t start;
        int64_t end;
        char type[48];
    } ts_nodes[128];

    struct {
        int active;
        char language[32];
        char pattern[256];
        char capture[64];
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
static void vm_init(VM* vm) {
    memset(vm, 0, sizeof(VM));
    heap_init(&vm->heap);
    vm->native_policy = ESHKOL_VM_NATIVE_POLICY_DESKTOP;
    vm->active_tape = NULL;
    memset(vm->ad_node_map, -1, sizeof(vm->ad_node_map));
}

/** @brief Bounds-check a heap object index against the live-object range
 *         [0, next_free). */
static inline int is_valid_heap_ptr(VM* vm, int32_t ptr) {
    return ptr >= 0 && ptr < vm->heap.next_free;
}

/** @brief VM-aware coercion of a Value to a double, extending as_number()
 *         to unwrap heap-boxed rationals (num/denom) and duals (primal
 *         component) via the VM's heap. */
static double as_number_vm(VM* vm, Value v) {
    if (v.type == VAL_INT) return (double)v.as.i;
    if (v.type == VAL_FLOAT) return v.as.f;
    if (v.type == VAL_CHAR) return (double)v.as.i; /* codepoint */
    if (v.type == VAL_RATIONAL && vm) {
        VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
        if (r && r->denom != 0) return (double)r->num / (double)r->denom;
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

/** @brief Append @p v to the VM's constant pool.
 * @return The new constant's index, or -1 if MAX_CONSTS is exceeded.
 */
static int add_constant(VM* vm, Value v) {
    if (vm->n_constants >= MAX_CONSTS) return -1;
    vm->constants[vm->n_constants] = v;
    return vm->n_constants++;
}

/*******************************************************************************
 * Print value
 ******************************************************************************/

/* Forward declarations for print_value */
typedef struct { Value* items; int len; int cap; } VmVector;

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
static void print_value(VM* vm, Value v) {
    switch ((int)v.type) {
        case VAL_NIL:   printf("()"); break;
        case VAL_INT:   printf("%lld", (long long)v.as.i); break;
        case VAL_FLOAT: printf("%.6g", v.as.f); break;
        case VAL_CHAR: {
            /* `display` renders a character as its glyph (UTF-8). `write`
             * would use #\ syntax, but this VM shares one printer for both
             * (the write/display distinction is filed separately). */
            char buf[4];
            int n = vm_utf8_encode((int)v.as.i, buf);
            if (n > 0) printf("%.*s", n, buf);
            break;
        }
        case VAL_BOOL:  printf("%s", v.as.b ? "#t" : "#f"); break;
        case VAL_PAIR: {
            printf("(");
            Value cur = v;
            int first = 1;
            while (cur.type == VAL_PAIR) {
                if (!first) printf(" ");
                first = 0;
                HeapObject* obj = vm->heap.objects[cur.as.ptr];
                print_value(vm, obj->cons.car);
                cur = obj->cons.cdr;
            }
            if (cur.type != VAL_NIL) {
                printf(" . ");
                print_value(vm, cur);
            }
            printf(")");
            break;
        }
        case VAL_STRING: {
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
                    print_value(vm, vec->items[i]);
                }
            }
            printf(")");
            break;
        }
        case VAL_CLOSURE: printf("<closure@%d>", v.as.ptr); break;
        case VAL_COMPLEX: {
            VmComplex* z = (VmComplex*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            if (z) printf("%g%+gi", z->real, z->imag);
            else printf("<complex>");
            break;
        }
        case VAL_RATIONAL: {
            HeapObject* obj = vm->heap.objects[v.as.ptr];
            if (obj && obj->opaque.ptr) {
                VmRational* r = (VmRational*)obj->opaque.ptr;
                if (r->denom == 1) printf("%lld", (long long)r->num);
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
                printf("<workspace: %d modules, dim=%d>",
                       ws->n_modules, ws->dim);
            } else printf("<workspace>");
            break;
        }
        case VAL_KB:          printf("<knowledge-base>"); break;
        case VAL_SUBST:       printf("<substitution>"); break;
        case VAL_HASH:        printf("<hash-table>"); break;
        case VAL_BYTEVECTOR:  printf("<bytevector>"); break;
        case VAL_PARAMETER_OBJ: printf("<parameter>"); break;
        case VAL_AD_TAPE:     printf("<ad-tape>"); break;
        case VAL_ERROR_OBJ:   printf("<error-object>"); break;
        case VAL_MANIFOLD:    printf("<manifold>"); break;
        case VAL_RIEMANNIAN_ADAM_STATE: printf("<riemannian-adam-state>"); break;
        case VAL_PORT:        printf("<port>"); break;
        case VAL_FUTURE:      printf("<future>"); break;
        case VAL_VOID:        break; /* unspecified — produces no output */
        default: printf("<unknown>"); break;
    }
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

    /* Run VM loop — will stop when OP_RETURN hits our sentinel frame */
    vm_run(vm);

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
    vm->error = 0;

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
    int pc, fp, sp, frame_count, n_handlers, n_winds, n_parameter_bindings;
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
