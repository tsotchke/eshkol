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

    OP_COUNT = 63
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

static int is_truthy(Value v) {
    if (v.type == VAL_BOOL) return v.as.b;
    if (v.type == VAL_NIL) return 0;
    return 1;  /* everything else is truthy */
}

static double as_number(Value v) {
    if (v.type == VAL_INT) return (double)v.as.i;
    if (v.type == VAL_FLOAT) return v.as.f;
    return 0.0;
}

static Value number_val(double d) {
    if (d == (int64_t)d && fabs(d) < 1e15) return INT_VAL((int64_t)d);
    return FLOAT_VAL(d);
}

/*******************************************************************************
 * Heap (arena-based, OALR)
 ******************************************************************************/

#define HEAP_SIZE 65536

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

static void heap_init(Heap* h) {
    vm_region_stack_init(&h->regions);
    h->capacity = HEAP_SIZE;
    h->objects = (HeapObject**)calloc(h->capacity, sizeof(HeapObject*));
    h->next_free = 0;
}

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

static void heap_region_push(Heap* h) {
    vm_region_push(&h->regions, NULL, 0);
}

static void heap_region_pop(Heap* h) {
    vm_region_pop(&h->regions);
}

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

#define STACK_SIZE 4096
#define MAX_FRAMES 256
#define MAX_CONSTS 1024

typedef struct {
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

    /* Status */
    int halted;
    int error;
} VM;

static void vm_init(VM* vm) {
    memset(vm, 0, sizeof(VM));
    heap_init(&vm->heap);
}

/* Validate a heap pointer */
static inline int is_valid_heap_ptr(VM* vm, int32_t ptr) {
    return ptr >= 0 && ptr < vm->heap.next_free;
}

static void vm_push(VM* vm, Value v) {
    if (vm->sp >= STACK_SIZE) { printf("STACK OVERFLOW\n"); vm->error = 1; return; }
    vm->stack[vm->sp++] = v;
}

static Value vm_pop(VM* vm) {
    if (vm->sp <= 0) { printf("STACK UNDERFLOW\n"); vm->error = 1; return NIL_VAL; }
    return vm->stack[--vm->sp];
}

static Value vm_peek(VM* vm, int offset) {
    int idx = vm->sp - 1 - offset;
    if (idx < 0 || idx >= vm->sp) { printf("PEEK OUT OF BOUNDS\n"); return NIL_VAL; }
    return vm->stack[idx];
}

static int add_constant(VM* vm, Value v) {
    if (vm->n_constants >= MAX_CONSTS) return -1;
    vm->constants[vm->n_constants] = v;
    return vm->n_constants++;
}

/*******************************************************************************
 * Print value
 ******************************************************************************/

static void print_value(VM* vm, Value v) {
    switch ((int)v.type) {
        case VAL_NIL:   printf("()"); break;
        case VAL_INT:   printf("%lld", (long long)v.as.i); break;
        case VAL_FLOAT: printf("%.6g", v.as.f); break;
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
        case VAL_CLOSURE: printf("<closure@%d>", v.as.ptr); break;
        case VAL_COMPLEX: {
            VmComplex* z = (VmComplex*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            if (z) printf("%g%+gi", z->real, z->imag);
            else printf("<complex>");
            break;
        }
        case VAL_RATIONAL: printf("<rational>"); break;
        case VAL_BIGNUM: printf("<bignum>"); break;
        case VAL_DUAL: printf("<dual>"); break;
        default: printf("<unknown>"); break;
    }
}

/*******************************************************************************
 * Forward declaration for vm_run (needed by closure bridge below)
 ******************************************************************************/

static void vm_run(VM* vm);

/*******************************************************************************
 * vm_call_closure_from_native — call a VM closure from native C code.
 *
 * This is the critical bridge that enables native functions (ws-step!,
 * parallel-map, call-with-values, etc.) to invoke user-defined closures.
 *
 * Protocol:
 *   1. Save entire VM state (pc, fp, sp, frame_count, halted)
 *   2. Push closure + args, set up frame with return_pc = -1 as sentinel
 *   3. Run vm_run — OP_RETURN detects sentinel, halts, pushes result
 *   4. Capture result, restore VM state, return it
 ******************************************************************************/

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

/* Simple vector for VEC_CREATE/VEC_REF/VEC_SET/VEC_LEN opcodes */
typedef struct {
    Value* items;
    int len;
    int cap;
} VmVector;

/* Macro: allocate heap object, set type, set opaque ptr, push result */
#define VM_PUSH_HEAP_OPAQUE(vm, heap_type, val_type, ptr_val) do { \
    int32_t _hp = heap_alloc(&(vm)->heap); \
    if (_hp < 0) { (vm)->error = 1; break; } \
    (vm)->heap.objects[_hp]->type = (heap_type); \
    (vm)->heap.objects[_hp]->opaque.ptr = (ptr_val); \
    vm_push((vm), (Value){.type = (val_type), .as.ptr = _hp}); \
} while(0)

#define VM_PUSH_TENSOR(vm, tptr) VM_PUSH_HEAP_OPAQUE(vm, HEAP_TENSOR, VAL_INT, tptr)

/*******************************************************************************
 * vm_dispatch_native — ALL native function dispatch in one place.
 *
 * Both the computed-goto and switch paths call this, eliminating duplication.
 ******************************************************************************/

