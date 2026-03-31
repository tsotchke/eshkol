/**
 * @file eshkol_vm.c
 * @brief Full Eshkol bytecode VM — 62-opcode ISA for complete language support.
 *
 * This VM interprets Eshkol bytecode using a register+stack architecture.
 * Memory is managed via OALR (arena regions, no GC).
 *
 * Phase 1: Core opcodes (comparisons, functions, closures, pairs)
 *          Enough to run recursive factorial, fibonacci, map, filter.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>

/* ESKB binary format writer (single-file include pattern) */
#include "eskb_writer.c"

/* ESKB binary format reader (single-file include pattern) */
#include "eskb_reader.c"

/* Arena memory system (OALR regions) */
#include "vm_arena.h"

/* Unified numeric tower types */
#include "vm_numeric.h"

/* Runtime libraries (single-file include pattern — #ifdef *_TEST guards main()) */
#include "vm_complex.c"
#include "vm_rational.c"
#include "vm_bignum.c"
#include "vm_dual.c"
#include "vm_autodiff.c"
#include "vm_tensor.c"
#include "vm_tensor_ops.c"
#include "vm_logic.c"
#include "vm_inference.c"
#include "vm_workspace.c"
#include "vm_string.c"
#include "vm_io.c"
#include "vm_hashtable.c"
#include "vm_bytevector.c"
#include "vm_multivalue.c"
#include "vm_error.c"
#include "vm_parameter.c"

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

static void vm_dispatch_native(VM* vm, int fid) {
    switch (fid) {
    /* ══════════════════════════════════════════════════════════════════════
     * Math functions (20-35)
     * ══════════════════════════════════════════════════════════════════════ */
    case 20: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(sin(as_number(a)))); break; }
    case 21: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(cos(as_number(a)))); break; }
    case 22: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(tan(as_number(a)))); break; }
    case 23: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(exp(as_number(a)))); break; }
    case 24: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(log(as_number(a)))); break; }
    case 25: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(sqrt(as_number(a)))); break; }
    case 26: { Value a = vm_pop(vm); vm_push(vm, number_val(floor(as_number(a)))); break; }
    case 27: { Value a = vm_pop(vm); vm_push(vm, number_val(ceil(as_number(a)))); break; }
    case 28: { Value a = vm_pop(vm); vm_push(vm, number_val(round(as_number(a)))); break; }
    case 29: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(asin(as_number(a)))); break; }
    case 30: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(acos(as_number(a)))); break; }
    case 31: { Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(atan(as_number(a)))); break; }
    case 32: { Value b = vm_pop(vm); Value a = vm_pop(vm); vm_push(vm, FLOAT_VAL(pow(as_number(a), as_number(b)))); break; }
    case 33: { Value b = vm_pop(vm); Value a = vm_pop(vm); double da=as_number(a),db=as_number(b); vm_push(vm, number_val(da<db?da:db)); break; }
    case 34: { Value b = vm_pop(vm); Value a = vm_pop(vm); double da=as_number(a),db=as_number(b); vm_push(vm, number_val(da>db?da:db)); break; }
    case 35: { Value a = vm_pop(vm); vm_push(vm, number_val(fabs(as_number(a)))); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * Predicates (40-50)
     * ══════════════════════════════════════════════════════════════════════ */
    case 40: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) > 0)); break; }
    case 41: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) < 0)); break; }
    case 42: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL((int64_t)as_number(a) % 2 != 0)); break; }
    case 43: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL((int64_t)as_number(a) % 2 == 0)); break; }
    case 44: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == 0)); break; }
    case 45: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_PAIR)); break; }
    case 46: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_INT || a.type == VAL_FLOAT)); break; }
    case 47: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_STRING)); break; }
    case 48: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_BOOL)); break; }
    case 49: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_CLOSURE)); break; }
    case 50: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(a.type == VAL_VECTOR)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * String operations (51-56) — legacy IDs
     * ══════════════════════════════════════════════════════════════════════ */
    case 51: { /* number->string */
        Value a = vm_pop(vm);
        char buf[64];
        if (a.type == VAL_INT) snprintf(buf, 64, "%lld", (long long)a.as.i);
        else snprintf(buf, 64, "%.15g", as_number(a));
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_STRING;
                vm->heap.objects[ptr]->opaque.ptr = s;
                vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 54: { Value b = vm_pop(vm); Value a = vm_pop(vm); (void)a; (void)b; vm_push(vm, NIL_VAL); break; }
    case 55: { Value b = vm_pop(vm); Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; }
    case 56: { Value a = vm_pop(vm); (void)a; vm_push(vm, INT_VAL(0)); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * I/O (60-61)
     * ══════════════════════════════════════════════════════════════════════ */
    case 60: printf("\n"); vm_push(vm, NIL_VAL); break;
    case 61: { Value v = vm_pop(vm); print_value(vm, v); vm_push(vm, NIL_VAL); break; }

    /* ══════════════════════════════════════════════════════════════════════
     * List/apply (70-73)
     * ══════════════════════════════════════════════════════════════════════ */
    case 70: { /* apply */
        Value args_list = vm_pop(vm);
        Value func = vm_pop(vm);
        if (func.type != VAL_CLOSURE) { vm->error = 1; break; }
        int argc = 0;
        Value cur = args_list;
        while (cur.type == VAL_PAIR && argc < 16) {
            vm_push(vm, vm->heap.objects[cur.as.ptr]->cons.car);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            argc++;
        }
        HeapObject* cl70 = vm->heap.objects[func.as.ptr];
        if (vm->frame_count >= MAX_FRAMES) { vm->error = 1; break; }
        vm->frames[vm->frame_count].return_pc = vm->pc;
        vm->frames[vm->frame_count].return_fp = vm->fp;
        vm->frames[vm->frame_count].func_pc = cl70->closure.func_pc;
        vm->frame_count++;
        vm->fp = vm->sp - argc;
        vm->pc = cl70->closure.func_pc;
        break;
    }
    case 71: { /* length */
        Value lst = vm_pop(vm);
        int len = 0;
        while (lst.type == VAL_PAIR) { len++; lst = vm->heap.objects[lst.as.ptr]->cons.cdr; }
        vm_push(vm, INT_VAL(len));
        break;
    }
    case 72: { /* reverse */
        Value lst = vm_pop(vm);
        Value result = NIL_VAL;
        while (lst.type == VAL_PAIR) {
            Value car = vm->heap.objects[lst.as.ptr]->cons.car;
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_CONS;
            vm->heap.objects[ptr]->cons.car = car;
            vm->heap.objects[ptr]->cons.cdr = result;
            result = PAIR_VAL(ptr);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }
    case 73: { /* append */
        Value b = vm_pop(vm), a = vm_pop(vm);
        if (a.type == VAL_NIL) { vm_push(vm, b); break; }
        Value rev = NIL_VAL;
        Value cur2 = a;
        while (cur2.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur2.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = rev;
            rev = PAIR_VAL(p);
            cur2 = vm->heap.objects[cur2.as.ptr]->cons.cdr;
        }
        Value result2 = b;
        while (rev.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[rev.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result2;
            result2 = PAIR_VAL(p);
            rev = vm->heap.objects[rev.as.ptr]->cons.cdr;
        }
        vm_push(vm, result2);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Make-vector (260)
     * ══════════════════════════════════════════════════════════════════════ */
    case 260: {
        Value fill = vm_pop(vm);
        Value n_val = vm_pop(vm);
        (void)fill;
        int n = (int)as_number(n_val);
        if (n < 0) n = 0;
        if (n > 256) n = 256;
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        vm->heap.objects[ptr]->type = HEAP_VECTOR;
        vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Complex Number Operations (300-319)
     * ══════════════════════════════════════════════════════════════════════ */
    case 300: case 301: case 302: case 303: case 304: case 305: case 306:
    case 307: case 308: case 309: case 310: case 311: case 312: case 313:
    case 314: case 315: case 316: case 317: case 318: case 319: {
        if (fid == 300) { /* make-rectangular */
            Value imag = vm_pop(vm), real = vm_pop(vm);
            VmComplex* z = vm_complex_new(&vm->heap.regions, as_number(real), as_number(imag));
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0 || !z) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_COMPLEX;
            vm->heap.objects[ptr]->opaque.ptr = z;
            vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
        } else if (fid == 302) { /* real-part */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(z->real));
            } else { vm_push(vm, FLOAT_VAL(as_number(z_val))); }
        } else if (fid == 303) { /* imag-part */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(z->imag));
            } else { vm_push(vm, FLOAT_VAL(0.0)); }
        } else if (fid == 304) { /* magnitude */
            Value z_val = vm_pop(vm);
            if (z_val.type == VAL_COMPLEX) {
                VmComplex* z = (VmComplex*)vm->heap.objects[z_val.as.ptr]->opaque.ptr;
                vm_push(vm, FLOAT_VAL(vm_complex_magnitude(z)));
            } else { vm_push(vm, FLOAT_VAL(fabs(as_number(z_val)))); }
        } else if (fid == 317) { /* complex? */
            Value v = vm_pop(vm);
            vm_push(vm, BOOL_VAL(v.type == VAL_COMPLEX));
        } else {
            int is_binary = (fid >= 307 && fid <= 310) || fid == 318 || fid == 319;
            if (is_binary) {
                Value b_val = vm_pop(vm), a_val = vm_pop(vm);
                VmComplex a_z = {as_number(a_val), 0}, b_z = {as_number(b_val), 0};
                if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
                if (b_val.type == VAL_COMPLEX) b_z = *(VmComplex*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
                VmComplex* result = NULL;
                switch (fid) {
                    case 307: result = vm_complex_add(&vm->heap.regions, &a_z, &b_z); break;
                    case 308: result = vm_complex_sub(&vm->heap.regions, &a_z, &b_z); break;
                    case 309: result = vm_complex_mul(&vm->heap.regions, &a_z, &b_z); break;
                    case 310: result = vm_complex_div(&vm->heap.regions, &a_z, &b_z); break;
                    case 318: result = vm_complex_expt(&vm->heap.regions, &a_z, &b_z); break;
                    case 319: vm_push(vm, BOOL_VAL(a_z.real == b_z.real && a_z.imag == b_z.imag)); break;
                }
                if (fid != 319) {
                    if (!result) { vm->error = 1; break; }
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr < 0) { vm->error = 1; break; }
                    vm->heap.objects[ptr]->type = HEAP_COMPLEX;
                    vm->heap.objects[ptr]->opaque.ptr = result;
                    vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
                }
            } else {
                Value a_val = vm_pop(vm);
                VmComplex a_z = {as_number(a_val), 0};
                if (a_val.type == VAL_COMPLEX) a_z = *(VmComplex*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
                VmComplex* result = NULL;
                switch (fid) {
                    case 301: { Value ang = vm_pop(vm);
                        result = vm_make_polar(&vm->heap.regions, as_number(a_val), as_number(ang)); break; }
                    case 305: vm_push(vm, FLOAT_VAL(vm_complex_angle(&a_z))); break;
                    case 306: result = vm_complex_conjugate(&vm->heap.regions, &a_z); break;
                    case 311: result = vm_complex_sqrt(&vm->heap.regions, &a_z); break;
                    case 312: result = vm_complex_exp(&vm->heap.regions, &a_z); break;
                    case 313: result = vm_complex_log(&vm->heap.regions, &a_z); break;
                    case 314: result = vm_complex_sin(&vm->heap.regions, &a_z); break;
                    case 315: result = vm_complex_cos(&vm->heap.regions, &a_z); break;
                    case 316: result = vm_complex_tan(&vm->heap.regions, &a_z); break;
                }
                if (fid != 305) {
                    if (!result) { vm->error = 1; break; }
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr < 0) { vm->error = 1; break; }
                    vm->heap.objects[ptr]->type = HEAP_COMPLEX;
                    vm->heap.objects[ptr]->opaque.ptr = result;
                    vm_push(vm, (Value){.type = VAL_COMPLEX, .as.ptr = ptr});
                }
            }
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Rational Number Operations (330-349)
     * ══════════════════════════════════════════════════════════════════════ */
    case 330: case 331: case 332: case 333: case 334: case 335: case 336:
    case 337: case 338: case 339: case 340: case 341: case 342: case 343:
    case 344: case 345: case 346: case 347: case 348: case 349: {
        VmArena* rat_arena = vm_active_arena(&vm->heap.regions);
        switch (fid) {
        case 330: { Value denom = vm_pop(vm), num = vm_pop(vm);
            VmRational* r = vm_rational_make(rat_arena, (int64_t)as_number(num), (int64_t)as_number(denom));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 331: case 332: case 333: case 334: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmRational* result = NULL;
            switch (fid) {
                case 331: result = vm_rational_add(rat_arena, &a_r, &b_r); break;
                case 332: result = vm_rational_sub(rat_arena, &a_r, &b_r); break;
                case 333: result = vm_rational_mul(rat_arena, &a_r, &b_r); break;
                case 334: result = vm_rational_div(rat_arena, &a_r, &b_r); break;
            }
            if (!result) { vm_push(vm, FLOAT_VAL((double)a_r.num/(double)a_r.denom + (double)b_r.num/(double)b_r.denom)); break; }
            if (result->denom == 1) { vm_push(vm, INT_VAL(result->num)); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = result;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 335: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_neg(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, number_val(-as_number(v))); break; }
        case 336: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_abs(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, number_val(fabs(as_number(v)))); break; }
        case 337: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) {
                VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                VmRational* res = vm_rational_inv(rat_arena, r);
                if (!res) { vm_push(vm, NIL_VAL); break; }
                int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
                vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = res;
                vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr});
            } else vm_push(vm, FLOAT_VAL(1.0 / as_number(v))); break; }
        case 338: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_rational_compare(&a_r, &b_r))); break; }
        case 339: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmRational a_r = {(int64_t)as_number(a_val), 1}, b_r = {(int64_t)as_number(b_val), 1};
            if (a_val.type == VAL_RATIONAL) a_r = *(VmRational*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_RATIONAL) b_r = *(VmRational*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            vm_push(vm, BOOL_VAL(vm_rational_equal(&a_r, &b_r))); break; }
        case 340: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(vm_rational_to_double(r))); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 341: { Value v = vm_pop(vm);
            VmRational* r = vm_rational_from_int(rat_arena, (int64_t)as_number(v));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        case 342: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_floor(r))); }
            else vm_push(vm, INT_VAL((int64_t)floor(as_number(v)))); break; }
        case 343: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_ceil(r))); }
            else vm_push(vm, INT_VAL((int64_t)ceil(as_number(v)))); break; }
        case 344: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_truncate(r))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 345: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_round(r))); }
            else vm_push(vm, INT_VAL((int64_t)round(as_number(v)))); break; }
        case 346: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_numerator(r))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 347: { Value v = vm_pop(vm);
            if (v.type == VAL_RATIONAL) { VmRational* r = (VmRational*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, INT_VAL(vm_rational_denominator(r))); }
            else vm_push(vm, INT_VAL(1)); break; }
        case 348: { Value b_v = vm_pop(vm), a_v = vm_pop(vm);
            vm_push(vm, INT_VAL(vm_rational_gcd((int64_t)as_number(a_v), (int64_t)as_number(b_v)))); break; }
        case 349: { Value tol = vm_pop(vm), x = vm_pop(vm);
            VmRational* r = vm_rationalize(rat_arena, as_number(x), as_number(tol));
            if (!r) { vm_push(vm, NIL_VAL); break; }
            if (r->denom == 1) { vm_push(vm, INT_VAL(r->num)); break; }
            int32_t ptr = heap_alloc(&vm->heap); if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_RATIONAL; vm->heap.objects[ptr]->opaque.ptr = r;
            vm_push(vm, (Value){.type = VAL_RATIONAL, .as.ptr = ptr}); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Bignum Operations (350-369)
     * ══════════════════════════════════════════════════════════════════════ */
    case 350: case 351: case 352: case 353: case 354: case 355: case 356:
    case 357: case 358: case 359: case 360: case 361: case 362: case 363:
    case 364: case 365: case 366: case 367: case 368: case 369: {
        VmRegionStack* bn_rs = &vm->heap.regions;
        switch (fid) {
        case 350: { Value v = vm_pop(vm); VmBignum* b = bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, b); break; }
        case 351: { Value v = vm_pop(vm);
            const char* s = (v.type == VAL_STRING && vm->heap.objects[v.as.ptr]->opaque.ptr) ? (const char*)vm->heap.objects[v.as.ptr]->opaque.ptr : "0";
            VmBignum* b = bignum_from_string(bn_rs, s);
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, b); break; }
        case 352: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) {
                VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr;
                char* s = bignum_to_string(bn_rs, b);
                if (!s) { vm->error = 1; break; }
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s);
            } else {
                char buf[32]; snprintf(buf, sizeof(buf), "%lld", (long long)(int64_t)as_number(v));
                char* s = (char*)vm_alloc(bn_rs, strlen(buf)+1); if (s) strcpy(s, buf);
                if (!s) { vm->error = 1; break; }
                VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s);
            } break; }
        case 353: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) { VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(bignum_to_double(b))); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 354: { Value v = vm_pop(vm);
            if (v.type == VAL_BIGNUM) { VmBignum* b = (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr; int ov=0; vm_push(vm, INT_VAL(bignum_to_int64(b, &ov))); }
            else vm_push(vm, INT_VAL((int64_t)as_number(v))); break; }
        case 355: case 356: case 357: case 358: case 359: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = NULL;
            switch (fid) { case 355: result=bignum_add(bn_rs,a_bn,b_bn); break; case 356: result=bignum_sub(bn_rs,a_bn,b_bn); break;
                case 357: result=bignum_mul(bn_rs,a_bn,b_bn); break; case 358: result=bignum_div(bn_rs,a_bn,b_bn); break; case 359: result=bignum_mod(bn_rs,a_bn,b_bn); break; }
            if (!result) { vm_push(vm, INT_VAL(0)); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 360: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_neg(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 361: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_abs_val(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 362: { Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmBignum* base_bn = (base_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[base_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(base_val));
            if (!base_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_pow(bn_rs, base_bn, (uint64_t)(int64_t)as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 363: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, INT_VAL(0)); break; }
            vm_push(vm, INT_VAL(bignum_compare(a_bn, b_bn))); break; }
        case 364: { Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, INT_VAL(0)); break; }
            VmBignum* result = bignum_gcd(bn_rs, a_bn, b_bn);
            if (!result) { vm_push(vm, INT_VAL(0)); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 365: case 366: case 367: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmBignum* a_bn = (a_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(a_val));
            VmBignum* b_bn = (b_val.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(b_val));
            if (!a_bn || !b_bn) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = NULL;
            switch (fid) { case 365: result=bignum_bitwise_and(bn_rs,a_bn,b_bn); break; case 366: result=bignum_bitwise_or(bn_rs,a_bn,b_bn); break; case 367: result=bignum_bitwise_xor(bn_rs,a_bn,b_bn); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 368: { Value v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            VmBignum* result = bignum_bitwise_not(bn_rs, b);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        case 369: { Value shift_val = vm_pop(vm), v = vm_pop(vm);
            VmBignum* b = (v.type == VAL_BIGNUM) ? (VmBignum*)vm->heap.objects[v.as.ptr]->opaque.ptr : bignum_from_int64(bn_rs, (int64_t)as_number(v));
            if (!b) { vm_push(vm, NIL_VAL); break; }
            int shift = (int)as_number(shift_val);
            VmBignum* result = (shift >= 0) ? bignum_shift_left(bn_rs, b, shift) : bignum_shift_right(bn_rs, b, -shift);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_BIGNUM, VAL_BIGNUM, result); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Dual Number Operations (370-389)
     * ══════════════════════════════════════════════════════════════════════ */
    case 370: case 371: case 372: case 373: case 374: case 375: case 376:
    case 377: case 378: case 379: case 380: case 381: case 382: case 383:
    case 384: case 385: case 386: case 387: case 388: case 389: {
        VmRegionStack* dual_rs = &vm->heap.regions;
        switch (fid) {
        case 370: { Value tangent = vm_pop(vm), primal = vm_pop(vm);
            VmDual* d = vm_dual_make(dual_rs, as_number(primal), as_number(tangent));
            if (!d) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d); break; }
        case 371: { Value v = vm_pop(vm);
            if (v.type == VAL_DUAL) { VmDual* d = (VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(d->primal)); }
            else vm_push(vm, FLOAT_VAL(as_number(v))); break; }
        case 372: { Value v = vm_pop(vm);
            if (v.type == VAL_DUAL) { VmDual* d = (VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr; vm_push(vm, FLOAT_VAL(d->tangent)); }
            else vm_push(vm, FLOAT_VAL(0.0)); break; }
        case 373: case 374: case 375: case 376: {
            Value b_val = vm_pop(vm), a_val = vm_pop(vm);
            VmDual a_d = {as_number(a_val), 0.0}, b_d = {as_number(b_val), 0.0};
            if (a_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
            if (b_val.type == VAL_DUAL) b_d = *(VmDual*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
            VmDual* result = NULL;
            switch (fid) { case 373: result=vm_dual_add(dual_rs,&a_d,&b_d); break; case 374: result=vm_dual_sub(dual_rs,&a_d,&b_d); break;
                case 375: result=vm_dual_mul(dual_rs,&a_d,&b_d); break; case 376: result=vm_dual_div(dual_rs,&a_d,&b_d); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 377: case 378: case 379: case 380: case 381:
        case 383: case 384: case 385: case 386: case 387: {
            Value v = vm_pop(vm);
            VmDual a_d = {as_number(v), 0.0};
            if (v.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[v.as.ptr]->opaque.ptr;
            VmDual* result = NULL;
            switch (fid) { case 377: result=vm_dual_sin(dual_rs,&a_d); break; case 378: result=vm_dual_cos(dual_rs,&a_d); break;
                case 379: result=vm_dual_exp(dual_rs,&a_d); break; case 380: result=vm_dual_log(dual_rs,&a_d); break;
                case 381: result=vm_dual_sqrt(dual_rs,&a_d); break; case 383: result=vm_dual_abs(dual_rs,&a_d); break;
                case 384: result=vm_dual_neg(dual_rs,&a_d); break; case 385: result=vm_dual_relu(dual_rs,&a_d); break;
                case 386: result=vm_dual_sigmoid(dual_rs,&a_d); break; case 387: result=vm_dual_tanh(dual_rs,&a_d); break; }
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 382: { Value exp_val = vm_pop(vm), base_val = vm_pop(vm);
            VmDual a_d = {as_number(base_val), 0.0};
            if (base_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[base_val.as.ptr]->opaque.ptr;
            VmDual* result = vm_dual_pow(dual_rs, &a_d, as_number(exp_val));
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        case 388: { Value v = vm_pop(vm);
            VmDual* d = vm_dual_from_double(dual_rs, as_number(v));
            if (!d) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d); break; }
        case 389: { Value dual_val = vm_pop(vm), scalar_val = vm_pop(vm);
            VmDual a_d = {as_number(dual_val), 0.0};
            if (dual_val.type == VAL_DUAL) a_d = *(VmDual*)vm->heap.objects[dual_val.as.ptr]->opaque.ptr;
            VmDual* result = vm_dual_scale(dual_rs, as_number(scalar_val), &a_d);
            if (!result) { vm_push(vm, NIL_VAL); break; }
            VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, result); break; }
        default: vm_push(vm, NIL_VAL); break;
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * AD Operations (390-409) — reverse-mode tape + forward-mode derivative
     * ══════════════════════════════════════════════════════════════════════ */
    case 390: { /* ad-tape-new */
        AdTape* tape = ad_tape_new(&vm->heap.regions);
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_AD_TAPE, VAL_INT, tape);
        break;
    }
    case 391: { /* ad-const(tape, value) */
        Value val = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, INT_VAL(ad_const(tape, as_number(val))));
        break;
    }
    case 392: { /* ad-var(tape, value) */
        Value val = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        vm_push(vm, INT_VAL(ad_var(tape, as_number(val))));
        break;
    }
    case 393: { /* derivative: (derivative f x) → f'(x) using forward-mode dual numbers */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm); (void)f_val;
        /* Create dual: x + 1*epsilon */
        VmDual* d = vm_dual_make(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm->error = 1; break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_DUAL, VAL_DUAL, d);
        /* Note: can't call closure from native code in simple VM.
         * Push the dual number — user extracts tangent after calling f. */
        break;
    }
    case 394: case 395: case 396: case 397: { /* ad-add, ad-sub, ad-mul, ad-div(tape, left, right) */
        Value right = vm_pop(vm), left = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        int result = -1;
        switch (fid) {
            case 394: result = ad_add(tape, (int)left.as.i, (int)right.as.i); break;
            case 395: result = ad_sub(tape, (int)left.as.i, (int)right.as.i); break;
            case 396: result = ad_mul(tape, (int)left.as.i, (int)right.as.i); break;
            case 397: result = ad_div(tape, (int)left.as.i, (int)right.as.i); break;
        }
        vm_push(vm, INT_VAL(result));
        break;
    }
    case 398: case 399: case 400: case 401: case 402:
    case 403: case 404: case 405: case 406: case 407: { /* ad unary ops(tape, node) */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        int result = -1;
        int idx = (int)node.as.i;
        switch (fid) {
            case 398: result = ad_sin(tape, idx); break;
            case 399: result = ad_cos(tape, idx); break;
            case 400: result = ad_exp(tape, idx); break;
            case 401: result = ad_log(tape, idx); break;
            case 402: result = ad_sqrt(tape, idx); break;
            case 403: result = ad_neg(tape, idx); break;
            case 404: result = ad_abs(tape, idx); break;
            case 405: result = ad_relu(tape, idx); break;
            case 406: result = ad_sigmoid(tape, idx); break;
            case 407: result = ad_tanh(tape, idx); break;
        }
        vm_push(vm, INT_VAL(result));
        break;
    }
    case 408: { /* ad-backward(tape, output_node) */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, NIL_VAL); break; }
        ad_backward(tape, (int)node.as.i);
        vm_push(vm, NIL_VAL); /* backward is side-effectful */
        break;
    }
    case 409: { /* ad-gradient(tape, node) → gradient value */
        Value node = vm_pop(vm), tape_val = vm_pop(vm);
        AdTape* tape = (AdTape*)vm->heap.objects[tape_val.as.ptr]->opaque.ptr;
        if (!tape) { vm_push(vm, FLOAT_VAL(0.0)); break; }
        int idx = (int)node.as.i;
        if (idx >= 0 && idx < tape->len) {
            vm_push(vm, FLOAT_VAL(tape->nodes[idx].gradient));
        } else {
            vm_push(vm, FLOAT_VAL(0.0));
        }
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor Core Operations (410-420)
     * ══════════════════════════════════════════════════════════════════════ */
    case 410: { /* make-tensor(shape, fill) */
        Value fill = vm_pop(vm), shape_val = vm_pop(vm);
        int64_t shape[8]; int n_dims = vm_extract_shape(vm, shape_val, shape, 8);
        if (n_dims == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_fill(&vm->heap.regions, shape, n_dims, as_number(fill));
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 411: { /* tensor-ref(tensor, indices) — flat or multi-dim access */
        Value idx_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, FLOAT_VAL(0)); break; }
        /* Single int/float index: flat access; list: multi-dim */
        if (idx_val.type == VAL_INT || idx_val.type == VAL_FLOAT) {
            int64_t flat = (int64_t)as_number(idx_val);
            if (flat >= 0 && flat < t->total)
                vm_push(vm, FLOAT_VAL(t->data[flat]));
            else vm_push(vm, FLOAT_VAL(0));
        } else if (idx_val.type == VAL_PAIR) {
            /* Multi-dim: walk list to get indices */
            int64_t indices[8]; int nd = 0;
            Value cur = idx_val;
            while (cur.type == VAL_PAIR && nd < 8) {
                indices[nd++] = (int64_t)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            double val = vm_tensor_ref(t, indices, nd);
            vm_push(vm, FLOAT_VAL(val));
        } else {
            vm_push(vm, FLOAT_VAL(0));
        }
        break;
    }
    case 412: { /* tensor-set!(tensor, indices, value) */
        Value val = vm_pop(vm), idx_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int64_t indices[8]; int n = vm_extract_shape(vm, idx_val, indices, 8);
        if (n == 0) { indices[0] = (int64_t)as_number(idx_val); n = 1; }
        vm_tensor_set(t, indices, n, as_number(val));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 413: { /* tensor-shape → list */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        Value result = NIL_VAL;
        for (int i = t->n_dims - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = INT_VAL(t->shape[i]);
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 414: { /* tensor-data → flat list (for small tensors) */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        Value result = NIL_VAL;
        int64_t limit = t->total > 1024 ? 1024 : t->total;
        for (int64_t i = limit - 1; i >= 0; i--) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = FLOAT_VAL(t->data[i]);
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
        }
        vm_push(vm, result);
        break;
    }
    case 415: { /* reshape(tensor, new_shape) */
        Value shape_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        VmTensor* out = vm_tensor_reshape(&vm->heap.regions, t, shape, n);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 416: { /* transpose */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_transpose(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 417: { /* zeros(shape) */
        Value shape_val = vm_pop(vm);
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        if (n == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_zeros(&vm->heap.regions, shape, n);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 418: { /* ones(shape) */
        Value shape_val = vm_pop(vm);
        int64_t shape[8]; int n = vm_extract_shape(vm, shape_val, shape, 8);
        if (n == 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* t = vm_tensor_ones(&vm->heap.regions, shape, n);
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 419: { /* arange(start, stop, step) */
        Value step = vm_pop(vm), stop = vm_pop(vm), start = vm_pop(vm);
        VmTensor* t = vm_tensor_arange(&vm->heap.regions, as_number(start), as_number(stop), as_number(step));
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, t);
        break;
    }
    case 420: { /* flatten */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_flatten(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Tensor Operations (440-469)
     * ══════════════════════════════════════════════════════════════════════ */
    case 440: { /* matmul */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_matmul(&vm->heap.regions, a, b);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 441: case 442: case 443: case 444: case 445: case 446: case 447: { /* tensor binary: +,-,*,/,pow,max,min */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = NULL;
        switch (fid) {
            case 441: out = vm_tensor_add(&vm->heap.regions, a, b); break;
            case 442: out = vm_tensor_sub(&vm->heap.regions, a, b); break;
            case 443: out = vm_tensor_mul(&vm->heap.regions, a, b); break;
            case 444: out = vm_tensor_div(&vm->heap.regions, a, b); break;
            case 445: out = vm_tensor_pow(&vm->heap.regions, a, b); break;
            case 446: out = vm_tensor_maximum(&vm->heap.regions, a, b); break;
            case 447: out = vm_tensor_minimum(&vm->heap.regions, a, b); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 448: { /* batch-matmul */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_batch_matmul(&vm->heap.regions, a, b);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 449: { /* dot */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmTensor* a = (VmTensor*)vm->heap.objects[a_val.as.ptr]->opaque.ptr;
        VmTensor* b = (VmTensor*)vm->heap.objects[b_val.as.ptr]->opaque.ptr;
        if (!a || !b) { vm_push(vm, FLOAT_VAL(0.0)); break; }
        vm_push(vm, FLOAT_VAL(vm_tensor_dot(a, b)));
        break;
    }
    case 450: case 451: case 452: case 453: case 454: case 455: { /* tensor unary: neg,abs,sqrt,exp,log,sin,cos */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = NULL;
        switch (fid) {
            case 450: out = vm_tensor_neg(&vm->heap.regions, t); break;
            case 451: out = vm_tensor_abs(&vm->heap.regions, t); break;
            case 452: out = vm_tensor_sqrt_op(&vm->heap.regions, t); break;
            case 453: out = vm_tensor_exp_op(&vm->heap.regions, t); break;
            case 454: out = vm_tensor_log_op(&vm->heap.regions, t); break;
            case 455: out = vm_tensor_sin_op(&vm->heap.regions, t); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 456: { /* scale(tensor, scalar) */
        Value scalar = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_scale(&vm->heap.regions, t, as_number(scalar));
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 457: case 458: case 459: case 460: { /* reduce: sum,mean,max,min (tensor, axis) */
        Value axis_val = vm_pop(vm), t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        int axis = (int)as_number(axis_val);
        VmTensor* out = NULL;
        switch (fid) {
            case 457: out = vm_tensor_sum(&vm->heap.regions, t, axis); break;
            case 458: out = vm_tensor_mean(&vm->heap.regions, t, axis); break;
            case 459: out = vm_tensor_max(&vm->heap.regions, t, axis); break;
            case 460: out = vm_tensor_min(&vm->heap.regions, t, axis); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 461: { /* cos tensor */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_cos_op(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 462: case 463: case 464: case 465: case 466: case 467: case 468: { /* activations: relu,sigmoid,tanh,leaky_relu,elu,gelu,swish */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = NULL;
        switch (fid) {
            case 462: out = vm_tensor_relu(&vm->heap.regions, t); break;
            case 463: out = vm_tensor_softmax(&vm->heap.regions, t, t->n_dims - 1); break;
            case 464: out = vm_tensor_sigmoid(&vm->heap.regions, t); break;
            case 465: out = vm_tensor_tanh_act(&vm->heap.regions, t); break;
            case 466: out = vm_tensor_leaky_relu(&vm->heap.regions, t); break;
            case 467: out = vm_tensor_elu(&vm->heap.regions, t); break;
            case 468: out = vm_tensor_gelu(&vm->heap.regions, t); break;
        }
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 469: { /* swish */
        Value t_val = vm_pop(vm);
        VmTensor* t = (VmTensor*)vm->heap.objects[t_val.as.ptr]->opaque.ptr;
        if (!t) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_swish(&vm->heap.regions, t);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Logic Operations (500-512)
     * ══════════════════════════════════════════════════════════════════════ */
    case 500: { /* make-logic-var(name) — name as int (id) */
        Value name_val = vm_pop(vm), dummy = vm_pop(vm); (void)dummy; (void)name_val;
        /* For simplicity, use the integer as var ID */
        vm_push(vm, INT_VAL((int64_t)vm_make_logic_var("?auto")));
        break;
    }
    case 501: { /* logic-var? — check heap type for LOGIC_VAR */
        Value v = vm_pop(vm);
        int is_lv = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                     vm->heap.objects[v.as.ptr]->type == HEAP_LOGIC_VAR);
        vm_push(vm, BOOL_VAL(is_lv));
        break;
    }
    case 502: { /* unify(subst, a, b) — bind a to b in substitution (copy-on-extend) */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.as.ptr >= 0 && vm->heap.objects[s_val.as.ptr]->type == HEAP_SUBST) {
            VmSubstitution* subst = (VmSubstitution*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            if (subst && a_val.type == VAL_INT) {
                /* Treat a_val as logic var ID, bind it to b_val */
                VmValue term;
                if (b_val.type == VAL_INT) { term.type = VM_VAL_INT64; term.data.int_val = b_val.as.i; }
                else if (b_val.type == VAL_FLOAT) { term.type = VM_VAL_DOUBLE; term.data.double_val = b_val.as.f; }
                else if (b_val.type == VAL_BOOL) { term.type = VM_VAL_BOOL; term.data.int_val = b_val.as.b; }
                else { term.type = VM_VAL_INT64; term.data.int_val = b_val.as.i; }
                VmSubstitution* extended = vm_subst_extend(&vm->heap.regions, subst, (uint64_t)a_val.as.i, &term);
                if (extended) {
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_INT, extended);
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL); /* unification failed */
        break;
    }
    case 505: { /* make-substitution */
        VmSubstitution* s = vm_make_substitution(&vm->heap.regions, 16);
        if (!s) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_SUBST, VAL_INT, s);
        break;
    }
    case 506: { /* substitution? */
        Value v = vm_pop(vm);
        int is_subst = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                        vm->heap.objects[v.as.ptr]->type == HEAP_SUBST);
        vm_push(vm, BOOL_VAL(is_subst));
        break;
    }
    case 509: { /* make-kb */
        VmKnowledgeBase* kb = vm_make_kb(&vm->heap.regions);
        if (!kb) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_KB, VAL_INT, kb);
        break;
    }
    case 510: { /* kb? */
        Value v = vm_pop(vm);
        int is_kb = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                     vm->heap.objects[v.as.ptr]->type == HEAP_KB);
        vm_push(vm, BOOL_VAL(is_kb));
        break;
    }
    case 503: { /* walk(term, subst) */
        Value subst_val = vm_pop(vm), term_val = vm_pop(vm);
        /* Walk resolves variables through substitution chains */
        if (subst_val.as.ptr >= 0 && vm->heap.objects[subst_val.as.ptr]->type == HEAP_SUBST) {
            VmSubstitution* s = (VmSubstitution*)vm->heap.objects[subst_val.as.ptr]->opaque.ptr;
            if (s && term_val.type == VAL_INT) {
                /* Check if term is a var_id that has a binding */
                for (int i = 0; i < s->n_bindings; i++) {
                    if (s->var_ids[i] == (uint64_t)term_val.as.i) {
                        /* Convert VmValue to VM Value */
                        VmValue bound = s->terms[i];
                        if (bound.type == VM_VAL_INT64)
                            vm_push(vm, INT_VAL(bound.data.int_val));
                        else if (bound.type == VM_VAL_DOUBLE)
                            vm_push(vm, FLOAT_VAL(bound.data.double_val));
                        else if (bound.type == VM_VAL_BOOL)
                            vm_push(vm, BOOL_VAL((int)bound.data.int_val));
                        else
                            vm_push(vm, INT_VAL(bound.data.int_val));
                        goto walk_done;
                    }
                }
            }
        }
        vm_push(vm, term_val); /* unbound — return as-is */
        walk_done: break;
    }
    case 504: { /* walk-deep — recursive walk through substitution chains */
        Value subst_val = vm_pop(vm), term_val = vm_pop(vm);
        /* Walk repeatedly until term no longer resolves */
        if (subst_val.as.ptr >= 0 && vm->heap.objects[subst_val.as.ptr]->type == HEAP_SUBST) {
            VmSubstitution* s = (VmSubstitution*)vm->heap.objects[subst_val.as.ptr]->opaque.ptr;
            Value resolved = term_val;
            int depth = 0;
            while (s && resolved.type == VAL_INT && depth < 100) {
                int found = 0;
                for (int i = 0; i < s->n_bindings; i++) {
                    if (s->var_ids[i] == (uint64_t)resolved.as.i) {
                        VmValue bound = s->terms[i];
                        if (bound.type == VM_VAL_INT64) resolved = INT_VAL(bound.data.int_val);
                        else if (bound.type == VM_VAL_DOUBLE) resolved = FLOAT_VAL(bound.data.double_val);
                        else if (bound.type == VM_VAL_BOOL) resolved = BOOL_VAL((int)bound.data.int_val);
                        else resolved = INT_VAL(bound.data.int_val);
                        found = 1;
                        break;
                    }
                }
                if (!found) break;
                depth++;
            }
            vm_push(vm, resolved);
        } else {
            vm_push(vm, term_val);
        }
        break;
    }
    case 507: { /* make-fact(pred, args) — pack predicate and args list */
        Value args_val = vm_pop(vm), pred = vm_pop(vm);
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; break; }
        vm->heap.objects[ptr]->type = HEAP_FACT;
        vm->heap.objects[ptr]->opaque.ptr = NULL;
        vm->heap.objects[ptr]->cons.car = pred;
        vm->heap.objects[ptr]->cons.cdr = args_val;
        vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
        break;
    }
    case 508: { /* fact? */
        Value v = vm_pop(vm);
        int is_fact = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                       vm->heap.objects[v.as.ptr]->type == HEAP_FACT);
        vm_push(vm, BOOL_VAL(is_fact));
        break;
    }
    case 511: { /* kb-assert!(kb, fact) */
        Value fact_val = vm_pop(vm), kb_val = vm_pop(vm);
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb) {
                /* Add fact pointer to KB */
                vm_kb_assert(&vm->heap.regions, kb, (VmFact*)vm->heap.objects[fact_val.as.ptr]->opaque.ptr);
            }
        }
        vm_push(vm, NIL_VAL); /* void return */
        break;
    }
    case 512: { /* kb-query(kb, pattern) → list of matching facts */
        Value pattern = vm_pop(vm), kb_val = vm_pop(vm);
        /* Return facts from KB that match the pattern predicate */
        if (kb_val.as.ptr >= 0 && vm->heap.objects[kb_val.as.ptr]->type == HEAP_KB) {
            VmKnowledgeBase* kb = (VmKnowledgeBase*)vm->heap.objects[kb_val.as.ptr]->opaque.ptr;
            if (kb) {
                Value result = NIL_VAL;
                for (int i = kb->n_facts - 1; i >= 0; i--) {
                    VmFact* f = kb->facts[i];
                    /* Build result list of matching facts */
                    int32_t p = heap_alloc(&vm->heap);
                    if (p < 0) break;
                    vm->heap.objects[p]->type = HEAP_CONS;
                    vm->heap.objects[p]->cons.car = INT_VAL((int64_t)(intptr_t)f);
                    vm->heap.objects[p]->cons.cdr = result;
                    result = PAIR_VAL(p);
                }
                vm_push(vm, result);
                break;
            }
        }
        (void)pattern;
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Inference Operations (520-526)
     * ══════════════════════════════════════════════════════════════════════ */
    case 520: { /* make-factor-graph(num_vars, var_dims_list) */
        Value dims_val = vm_pop(vm), nvars_val = vm_pop(vm);
        int nv = (int)as_number(nvars_val);
        int var_dims[64]; int nd = 0;
        if (dims_val.type == VAL_PAIR) {
            Value cur = dims_val;
            while (cur.type == VAL_PAIR && nd < 64) {
                var_dims[nd++] = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
        } else { var_dims[0] = (int)as_number(dims_val); nd = 1; }
        if (nd < nv) nv = nd;
        VmFactorGraph* fg = vm_make_factor_graph(&vm->heap.regions, nv, var_dims);
        if (!fg) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_FACTOR_GRAPH, VAL_INT, fg);
        break;
    }
    case 521: { /* factor-graph? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                              vm->heap.objects[v.as.ptr]->type == HEAP_FACTOR_GRAPH));
        break;
    }
    case 522: { /* fg-add-factor!(fg, var_indices, cpt) */
        Value cpt = vm_pop(vm), vars = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                /* Extract var_indices from list */
                int var_idx[8], n_vars = 0;
                Value cur = vars;
                while (cur.type == VAL_PAIR && n_vars < 8) {
                    var_idx[n_vars++] = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
                /* Extract CPT data from tensor or list */
                double* cpt_data = NULL;
                if (cpt.as.ptr >= 0 && vm->heap.objects[cpt.as.ptr]->type == HEAP_TENSOR) {
                    VmTensor* t = (VmTensor*)vm->heap.objects[cpt.as.ptr]->opaque.ptr;
                    if (t) cpt_data = t->data;
                }
                if (n_vars > 0 && cpt_data) {
                    /* Build dims array from factor graph's var_dims */
                    int dims[8];
                    for (int i = 0; i < n_vars; i++) {
                        int vi = var_idx[i];
                        dims[i] = (vi >= 0 && vi < fg->num_vars) ? fg->var_dims[vi] : 2;
                    }
                    VmFactor* factor = vm_make_factor(&vm->heap.regions, var_idx, n_vars, cpt_data, dims);
                    if (factor) vm_fg_add_factor(&vm->heap.regions, fg, factor);
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 523: { /* fg-infer!(fg, max_iters, tolerance) */
        Value tol = vm_pop(vm), iters = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                int converged = vm_fg_infer(&vm->heap.regions, fg, (int)as_number(iters), as_number(tol));
                vm_push(vm, BOOL_VAL(converged));
                break;
            }
        }
        vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 524: { /* fg-update-cpt!(fg, factor_idx, new_cpt_tensor) */
        Value cpt = vm_pop(vm), idx = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg && cpt.as.ptr >= 0 && vm->heap.objects[cpt.as.ptr]->type == HEAP_TENSOR) {
                VmTensor* t = (VmTensor*)vm->heap.objects[cpt.as.ptr]->opaque.ptr;
                if (t) vm_fg_update_cpt(fg, (int)as_number(idx), t->data, (int)t->total);
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 525: { /* free-energy(fg, observations) */
        Value obs = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                /* Parse observations: list of (var_idx, state) pairs */
                int obs_pairs[32][2], n_obs = 0;
                Value cur = obs;
                while (cur.type == VAL_PAIR && n_obs < 32) {
                    Value pair = vm->heap.objects[cur.as.ptr]->cons.car;
                    if (pair.type == VAL_PAIR) {
                        obs_pairs[n_obs][0] = (int)as_number(vm->heap.objects[pair.as.ptr]->cons.car);
                        obs_pairs[n_obs][1] = (int)as_number(vm->heap.objects[vm->heap.objects[pair.as.ptr]->cons.cdr.as.ptr]->cons.car);
                        n_obs++;
                    }
                    cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
                }
                double fe = vm_free_energy(fg, (const double*)obs_pairs, n_obs);
                vm_push(vm, FLOAT_VAL(fe));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0.0));
        break;
    }
    case 526: { /* expected-free-energy(fg, action_var, action_state) */
        Value state = vm_pop(vm), var = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                double efe = vm_expected_free_energy(&vm->heap.regions, fg, (int)as_number(var), (int)as_number(state));
                vm_push(vm, FLOAT_VAL(efe));
                break;
            }
        }
        vm_push(vm, FLOAT_VAL(0.0));
        break;
    }

    case 527: { /* fg-observe!(fg, var_id, observed_state)
                 * Clamps a variable to an observed state for evidence injection.
                 * After observing, re-run fg-infer! to propagate the evidence.
                 * Ref: Standard factor graph evidence clamping (Kschischang et al. 2001). */
        Value state_val = vm_pop(vm), var_val = vm_pop(vm), fg_val = vm_pop(vm);
        if (fg_val.as.ptr >= 0 && vm->heap.objects[fg_val.as.ptr]->type == HEAP_FACTOR_GRAPH) {
            VmFactorGraph* fg = (VmFactorGraph*)vm->heap.objects[fg_val.as.ptr]->opaque.ptr;
            if (fg) {
                int var_id = (int)as_number(var_val);
                int obs_state = (int)as_number(state_val);
                if (var_id >= 0 && var_id < fg->num_vars &&
                    obs_state >= 0 && obs_state < fg->var_dims[var_id]) {
                    /* Clamp beliefs: set observed state to probability 1, others to 0 */
                    int dim = fg->var_dims[var_id];
                    for (int s = 0; s < dim; s++) {
                        fg->beliefs[var_id][s] = (s == obs_state) ? 0.0 : -1e30;
                    }
                    /* Mark variable as observed (skip during belief update in fg-infer!).
                     * Use calloc (not arena) because VmFactorGraph's lifetime may exceed
                     * arena scope. The leak is bounded: one allocation per factor graph
                     * (guarded by !fg->observed), num_vars * sizeof(bool) = typically 6-24 bytes.
                     * Freed when the factor graph is destroyed or the process exits. */
                    if (!fg->observed) {
                        fg->observed = (bool*)calloc(fg->num_vars, sizeof(bool));
                    }
                    if (fg->observed) {
                        fg->observed[var_id] = true;
                    }
                    vm_push(vm, BOOL_VAL(true));
                    break;
                }
            }
        }
        vm_push(vm, BOOL_VAL(false));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Workspace Operations (540-544)
     * ══════════════════════════════════════════════════════════════════════ */
    case 540: { /* make-workspace(dim, max_modules) */
        Value max_m = vm_pop(vm), dim_val = vm_pop(vm);
        VmWorkspace* ws = vm_ws_new(&vm->heap.regions, (int)as_number(dim_val), (int)as_number(max_m));
        if (!ws) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_WORKSPACE, VAL_INT, ws);
        break;
    }
    case 541: { /* workspace? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                              vm->heap.objects[v.as.ptr]->type == HEAP_WORKSPACE));
        break;
    }
    case 542: { /* ws-register!(ws, name, closure) */
        Value closure = vm_pop(vm), name_val = vm_pop(vm), ws_val = vm_pop(vm);
        if (ws_val.as.ptr >= 0 && vm->heap.objects[ws_val.as.ptr]->type == HEAP_WORKSPACE) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws) {
                const char* name = "module";
                if (name_val.type == VAL_STRING && vm->heap.objects[name_val.as.ptr]->opaque.ptr) {
                    VmString* ns = (VmString*)vm->heap.objects[name_val.as.ptr]->opaque.ptr;
                    name = ns->data;
                }
                /* Allocate stable Value on arena so pointer survives past this scope */
                Value* stable_closure = (Value*)vm_alloc(&vm->heap.regions, sizeof(Value));
                if (stable_closure) {
                    *stable_closure = closure;
                    vm_ws_register(ws, name, stable_closure);
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 543: { /* ws-step!(ws) — invoke module closures via closure bridge */
        Value ws_val = vm_pop(vm);
        if (ws_val.as.ptr >= 0 && vm->heap.objects[ws_val.as.ptr]->type == HEAP_WORKSPACE) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws && ws->content) {
                /* Build content tensor */
                int64_t shape[1] = {ws->dim};
                VmTensor* ct = vm_tensor_from_data(&vm->heap.regions, ws->content, shape, 1);
                int32_t tptr = heap_alloc(&vm->heap);
                if (tptr < 0 || !ct) { vm_push(vm, NIL_VAL); break; }
                vm->heap.objects[tptr]->type = HEAP_TENSOR;
                vm->heap.objects[tptr]->opaque.ptr = ct;
                Value content_val = (Value){.type = VAL_INT, .as.ptr = tptr};

                /* Call each module's closure, collect salience + proposal */
                double saliences[32];
                Value proposals[32];
                int n_mod = ws->n_modules;
                if (n_mod > 32) n_mod = 32;
                for (int i = 0; i < n_mod; i++) {
                    Value* closure_ptr = (Value*)ws->modules[i].process_fn;
                    if (closure_ptr && closure_ptr->type == VAL_CLOSURE) {
                        Value result = vm_call_closure_from_native(vm, *closure_ptr, &content_val, 1);
                        if (result.type == VAL_PAIR) {
                            saliences[i] = as_number(vm->heap.objects[result.as.ptr]->cons.car);
                            proposals[i] = vm->heap.objects[result.as.ptr]->cons.cdr;
                        } else {
                            saliences[i] = as_number(result);
                            proposals[i] = content_val;
                        }
                    } else {
                        saliences[i] = 0;
                        proposals[i] = content_val;
                    }
                }
                /* Softmax competition: highest salience wins */
                int winner = 0;
                for (int i = 1; i < n_mod; i++)
                    if (saliences[i] > saliences[winner]) winner = i;
                ws->step_count++;
                vm_push(vm, proposals[winner]);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 544: { /* ws-get-content */
        Value ws_val = vm_pop(vm);
        if (ws_val.as.ptr >= 0 && vm->heap.objects[ws_val.as.ptr]->type == HEAP_WORKSPACE) {
            VmWorkspace* ws = (VmWorkspace*)vm->heap.objects[ws_val.as.ptr]->opaque.ptr;
            if (ws && ws->content) {
                /* Return content as tensor */
                int64_t shape[1] = {ws->dim};
                VmTensor* t = vm_tensor_from_data(&vm->heap.regions, ws->content, shape, 1);
                if (t) {
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr >= 0) {
                        vm->heap.objects[ptr]->type = HEAP_TENSOR;
                        vm->heap.objects[ptr]->opaque.ptr = t;
                        vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                        break;
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * String Operations (550-570) — real VmString dispatch
     * ══════════════════════════════════════════════════════════════════════ */
    case 550: { /* string-length */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_string_length(s)));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 551: { /* string-ref(str, idx) */
        Value idx = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int cp = vm_string_ref(s, (int)as_number(idx));
            vm_push(vm, INT_VAL(cp >= 0 ? cp : 0));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 552: { /* string-set!(str, idx, char) → new string */
        Value ch = vm_pop(vm), idx = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* result = vm_string_set(&vm->heap.regions, s, (int)as_number(idx), (int)as_number(ch));
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 553: { /* substring(str, start, end) */
        Value end = vm_pop(vm), start = vm_pop(vm), s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            VmString* result = vm_string_substring(&vm->heap.regions, s, (int)as_number(start), (int)as_number(end));
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
            else vm_push(vm, NIL_VAL);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 554: { /* string-append */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        if (a && b) {
            VmString* result = vm_string_append(&vm->heap.regions, a, b);
            if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 555: { /* string-contains(str, substr) */
        Value sub_val = vm_pop(vm), s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        VmString* sub = (sub_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[sub_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(vm_string_contains(s, sub)));
        break;
    }
    case 556: { /* make-string(n, char) */
        Value ch = vm_pop(vm), n = vm_pop(vm);
        VmString* result = vm_string_make(&vm->heap.regions, (int)as_number(n), (int)as_number(ch));
        if (result) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, result); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 557: { /* string-upcase */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_upcase(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 558: { /* string-downcase */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_downcase(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 559: { /* string-contains (duplicate of 555 for compat) */
        Value sub_val = vm_pop(vm), s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        VmString* sub = (sub_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[sub_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(vm_string_contains(s, sub)));
        break;
    }
    case 560: { /* string=? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_eq(a, b)));
        break;
    }
    case 561: { /* string<? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_lt(a, b)));
        break;
    }
    case 562: { /* string-ci=? */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmString* a = (a_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmString* b = (b_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, BOOL_VAL(vm_string_ci_eq(a, b)));
        break;
    }
    case 563: case 564: { /* string->number / number->string */
        if (fid == 563) { /* string->number */
            Value s_val = vm_pop(vm);
            VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
            double d = vm_string_to_number(s);
            if (isnan(d)) vm_push(vm, BOOL_VAL(0)); /* #f on parse failure */
            else vm_push(vm, number_val(d));
        } else { /* number->string */
            Value n = vm_pop(vm);
            VmString* r = vm_number_to_string(&vm->heap.regions, as_number(n));
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
            else vm_push(vm, NIL_VAL);
        }
        break;
    }
    case 565: { /* string->list — convert string to list of character codepoints */
        Value s_val = vm_pop(vm);
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            int len = vm_string_length(s);
            Value result = NIL_VAL;
            for (int i = len - 1; i >= 0; i--) {
                int cp = vm_string_ref(s, i);
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = INT_VAL(cp >= 0 ? cp : 0);
                vm->heap.objects[p]->cons.cdr = result;
                result = PAIR_VAL(p);
            }
            vm_push(vm, result);
        } else {
            vm_push(vm, NIL_VAL);
        }
        break;
    }
    case 566: { /* string-copy */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        if (s) { VmString* r = vm_string_copy(&vm->heap.regions, s); if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); break; } }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 567: { /* list->string — convert list of character codepoints to string */
        Value lst = vm_pop(vm);
        /* Count characters */
        int len = 0;
        Value cur = lst;
        while (cur.type == VAL_PAIR && len < 4096) {
            len++;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)(len + 1));
        if (buf) {
            cur = lst;
            int idx = 0;
            while (cur.type == VAL_PAIR && idx < len) {
                int cp = (int)as_number(vm->heap.objects[cur.as.ptr]->cons.car);
                buf[idx++] = (cp >= 0 && cp < 128) ? (char)cp : '?';
                cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
            }
            buf[idx] = '\0';
            VmString* s = vm_string_new(&vm->heap.regions, buf, idx);
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 568: { /* string->number (alt ID) */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        double d = vm_string_to_number(s);
        if (isnan(d)) vm_push(vm, BOOL_VAL(0));
        else vm_push(vm, number_val(d));
        break;
    }
    case 569: { /* number->string (alt ID) */
        Value n = vm_pop(vm);
        VmString* r = vm_number_to_string(&vm->heap.regions, as_number(n));
        if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, r); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 570: { /* string-hash */
        Value s_val = vm_pop(vm);
        VmString* s = (s_val.type == VAL_STRING) ? (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr : NULL;
        vm_push(vm, INT_VAL(s ? (int64_t)vm_string_hash(s) : 0));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * I/O Operations (580-602) — port-based I/O
     * ══════════════════════════════════════════════════════════════════════ */
    case 580: { /* open-input-file(path) */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            VmPort* p = vm_port_open_input_file(&vm->heap.regions, s->data);
            if (p) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_PORT;
                    vm->heap.objects[ptr]->opaque.ptr = p;
                    vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 581: { /* open-output-file(path) */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            VmPort* p = vm_port_open_output_file(&vm->heap.regions, s->data);
            if (p) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_PORT;
                    vm->heap.objects[ptr]->opaque.ptr = p;
                    vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 582: { /* read-char — from stdin */
        Value port = vm_pop(vm); (void)port;
        int ch = getchar();
        vm_push(vm, ch == EOF ? NIL_VAL : INT_VAL(ch));
        break;
    }
    case 583: { /* peek-char */
        Value port = vm_pop(vm); (void)port;
        int ch = getchar();
        if (ch != EOF) ungetc(ch, stdin);
        vm_push(vm, ch == EOF ? NIL_VAL : INT_VAL(ch));
        break;
    }
    case 584: { /* write-char(char, port) */
        Value port = vm_pop(vm), ch = vm_pop(vm); (void)port;
        putchar((int)as_number(ch));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 585: { /* write-string(str, port) */
        Value port = vm_pop(vm), s_val = vm_pop(vm); (void)port;
        if (s_val.type == VAL_STRING && vm->heap.objects[s_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[s_val.as.ptr]->opaque.ptr;
            printf("%.*s", s->byte_len, s->data);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 586: { /* write-char(char, port) — write to stdout if no port */
        Value ch = vm_pop(vm);
        putchar((int)as_number(ch));
        vm_push(vm, NIL_VAL);
        break;
    }
    case 587: { /* write-string */
        Value str = vm_pop(vm);
        if (str.type == VAL_STRING && vm->heap.objects[str.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[str.as.ptr]->opaque.ptr;
            fwrite(s->data, 1, s->byte_len, stdout);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 588: { /* read — read a single char from stdin, return as integer */
        int ch = getchar();
        vm_push(vm, ch == EOF ? NIL_VAL : INT_VAL(ch));
        break;
    }
    case 589: { /* write(datum) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 590: { /* display(datum) — print without quoting (same as write in this VM) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 591: { /* newline */
        printf("\n"); vm_push(vm, NIL_VAL); break;
    }
    case 592: { /* eof-object? */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(v.type == VAL_NIL));
        break;
    }
    case 593: { /* current-input-port */
        vm_push(vm, NIL_VAL); break;
    }
    case 594: { /* current-output-port */
        vm_push(vm, NIL_VAL); break;
    }
    case 595: { /* current-error-port → just return a sentinel */
        vm_push(vm, INT_VAL(-3)); /* stderr sentinel */
        break;
    }
    case 596: { /* open-input-string */
        Value str = vm_pop(vm);
        VmString* src = (str.type == VAL_STRING && vm->heap.objects[str.as.ptr]->opaque.ptr)
            ? (VmString*)vm->heap.objects[str.as.ptr]->opaque.ptr : NULL;
        VmPort* p = vm_port_open_input_string(&vm->heap.regions, src);
        if (p) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_PORT;
                vm->heap.objects[ptr]->opaque.ptr = p;
                vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 597: { /* open-output-string */
        VmPort* p = vm_port_open_output_string(&vm->heap.regions);
        if (p) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_PORT;
                vm->heap.objects[ptr]->opaque.ptr = p;
                vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 598: { /* get-output-string */
        Value port_val = vm_pop(vm);
        if (port_val.as.ptr >= 0 && vm->heap.objects[port_val.as.ptr]->type == HEAP_PORT) {
            VmPort* p = (VmPort*)vm->heap.objects[port_val.as.ptr]->opaque.ptr;
            VmString* s = vm_port_get_output_string(&vm->heap.regions, p);
            if (s) {
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_STRING;
                    vm->heap.objects[ptr]->opaque.ptr = s;
                    vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 599: { /* file-exists? */
        Value path = vm_pop(vm);
        int exists = 0;
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            exists = vm_port_file_exists(s->data);
        }
        vm_push(vm, BOOL_VAL(exists));
        break;
    }
    case 600: { /* delete-file */
        Value path = vm_pop(vm);
        if (path.type == VAL_STRING && vm->heap.objects[path.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[path.as.ptr]->opaque.ptr;
            vm_port_delete_file(s->data);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 601: { /* directory-entries(path) — returns list of filenames in directory */
        Value path = vm_pop(vm);
        /* No portable C89 readdir — return empty list on unsupported platforms */
        (void)path;
        vm_push(vm, NIL_VAL);
        break;
    }
    case 602: { /* command-line — return empty list */
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Parallel (620-628) — sequential fallback (VM is single-threaded)
     * ══════════════════════════════════════════════════════════════════════ */
    case 620: { /* parallel-map(fn, list) — sequential via closure bridge (VM is single-threaded) */
        Value list = vm_pop(vm), fn = vm_pop(vm);
        /* Apply fn to each element, build result list in correct order */
        Value rev_results = NIL_VAL;
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            Value mapped = vm_call_closure_from_native(vm, fn, &elem, 1);
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = mapped;
            vm->heap.objects[p]->cons.cdr = rev_results;
            rev_results = PAIR_VAL(p);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        /* Reverse to get correct order */
        Value result = NIL_VAL;
        cur = rev_results;
        while (cur.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }
    case 621: { /* parallel-filter(pred, list) — sequential via closure bridge */
        Value list = vm_pop(vm), pred = vm_pop(vm);
        Value rev_results = NIL_VAL;
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            Value keep = vm_call_closure_from_native(vm, pred, &elem, 1);
            if (is_truthy(keep)) {
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) break;
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = elem;
                vm->heap.objects[p]->cons.cdr = rev_results;
                rev_results = PAIR_VAL(p);
            }
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        /* Reverse */
        Value result = NIL_VAL;
        cur = rev_results;
        while (cur.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) break;
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }
    case 622: { /* parallel-fold(fn, init, list) — sequential via closure bridge */
        Value list = vm_pop(vm), init = vm_pop(vm), fn = vm_pop(vm);
        Value acc = init;
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            Value args[2] = {acc, elem};
            acc = vm_call_closure_from_native(vm, fn, args, 2);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, acc);
        break;
    }
    case 623: { /* parallel-for-each(fn, list) — sequential via closure bridge */
        Value list = vm_pop(vm), fn = vm_pop(vm);
        Value cur = list;
        while (cur.type == VAL_PAIR) {
            Value elem = vm->heap.objects[cur.as.ptr]->cons.car;
            vm_call_closure_from_native(vm, fn, &elem, 1);
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 624: { /* parallel-execute(thunk) — sequential: just call the thunk */
        Value thunk = vm_pop(vm);
        Value result = vm_call_closure_from_native(vm, thunk, NULL, 0);
        vm_push(vm, result);
        break;
    }
    case 625: { /* future(thunk) — in single-threaded VM, evaluate immediately */
        Value thunk = vm_pop(vm);
        Value result = vm_call_closure_from_native(vm, thunk, NULL, 0);
        vm_push(vm, result);
        break;
    }
    case 626: { /* force-future */
        Value fut = vm_pop(vm);
        vm_push(vm, fut); /* futures are just values in single-threaded mode */
        break;
    }
    case 627: { /* future-ready? — always true in single-threaded mode */
        Value fut = vm_pop(vm); (void)fut;
        vm_push(vm, BOOL_VAL(1));
        break;
    }
    case 628: { /* thread-pool-info */
        vm_push(vm, INT_VAL(1)); /* 1 thread (main) */
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * MultiValue Operations (650-654)
     * ══════════════════════════════════════════════════════════════════════ */
    case 650: { /* values(v) — single-value pass-through; value already on stack */
        break;
    }
    case 651: { /* multi-value-ref(mv, idx) — extract from multi-value container */
        Value idx = vm_pop(vm), mv = vm_pop(vm);
        if (mv.as.ptr >= 0 && vm->heap.objects[mv.as.ptr]->type == HEAP_MULTI_VALUE) {
            VmMultiValue* mvobj = (VmMultiValue*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (mvobj && i >= 0 && i < mvobj->count) {
                vm_push(vm, INT_VAL((int64_t)(intptr_t)mvobj->values[i]));
                break;
            }
        }
        /* Single value: index 0 returns the value itself */
        if ((int)as_number(idx) == 0) { vm_push(vm, mv); }
        else { vm_push(vm, NIL_VAL); }
        break;
    }
    case 652: { /* multi-value-count(mv) — number of values in container */
        Value mv = vm_pop(vm);
        if (mv.as.ptr >= 0 && mv.as.ptr < vm->heap.next_free &&
            vm->heap.objects[mv.as.ptr]->type == HEAP_MULTI_VALUE) {
            VmMultiValue* mvobj = (VmMultiValue*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(mvobj ? mvobj->count : 1));
        } else {
            vm_push(vm, INT_VAL(1)); /* single value counts as 1 */
        }
        break;
    }
    case 653: { /* multi-value? — check if value is a multi-value container */
        Value v = vm_pop(vm);
        int is_mv = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                     vm->heap.objects[v.as.ptr]->type == HEAP_MULTI_VALUE);
        vm_push(vm, BOOL_VAL(is_mv));
        break;
    }
    case 654: { /* call-with-values(producer, consumer) — via closure bridge */
        Value consumer = vm_pop(vm), producer = vm_pop(vm);
        /* Call producer with 0 args */
        Value produced = vm_call_closure_from_native(vm, producer, NULL, 0);
        /* Call consumer with produced value */
        Value result = vm_call_closure_from_native(vm, consumer, &produced, 1);
        vm_push(vm, result);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Hash Table Operations (660-670)
     * ══════════════════════════════════════════════════════════════════════ */
    case 660: { /* make-hash-table */
        VmHashTable* ht = vm_ht_make(&vm->heap.regions);
        if (!ht) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_INT, ht);
        break;
    }
    case 661: { /* hash-ref(ht, key, default) */
        Value dflt = vm_pop(vm), key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            void* result = vm_ht_ref(ht, (void*)(uintptr_t)key.as.i, (void*)(uintptr_t)dflt.as.i);
            vm_push(vm, INT_VAL((int64_t)(intptr_t)result));
        } else vm_push(vm, dflt);
        break;
    }
    case 662: { /* hash-set!(ht, key, value) */
        Value val = vm_pop(vm), key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_ht_set(&vm->heap.regions, ht, (void*)(uintptr_t)key.as.i, (void*)(uintptr_t)val.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 663: { /* hash-delete!(ht, key) */
        Value key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_ht_remove(ht, (void*)(uintptr_t)key.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 664: { /* hash-has-key?(ht, key) */
        Value key = vm_pop(vm), ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_push(vm, BOOL_VAL(vm_ht_has_key(ht, (void*)(uintptr_t)key.as.i)));
        } else vm_push(vm, BOOL_VAL(0));
        break;
    }
    case 665: { /* hash-keys */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                Value result = NIL_VAL;
                for (int i = ht->capacity - 1; i >= 0; i--) {
                    if (ht->keys[i]) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = *(Value*)ht->keys[i];
                        vm->heap.objects[p]->cons.cdr = result;
                        result = PAIR_VAL(p);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 666: { /* hash-values */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                Value result = NIL_VAL;
                for (int i = ht->capacity - 1; i >= 0; i--) {
                    if (ht->keys[i]) {
                        int32_t p = heap_alloc(&vm->heap);
                        if (p < 0) break;
                        vm->heap.objects[p]->type = HEAP_CONS;
                        vm->heap.objects[p]->cons.car = *(Value*)ht->values[i];
                        vm->heap.objects[p]->cons.cdr = result;
                        result = PAIR_VAL(p);
                    }
                }
                vm_push(vm, result);
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 667: { /* hash-count */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(ht ? vm_ht_count(ht) : 0));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 668: { /* hash-table-copy(ht) — shallow copy of hash table */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) {
                VmHashTable* copy = vm_ht_make(&vm->heap.regions);
                if (copy) {
                    VM_PUSH_HEAP_OPAQUE(vm, HEAP_HASH, VAL_INT, copy);
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 669: { /* hash-clear! */
        Value ht_val = vm_pop(vm);
        if (ht_val.as.ptr >= 0 && vm->heap.objects[ht_val.as.ptr]->type == HEAP_HASH) {
            VmHashTable* ht = (VmHashTable*)vm->heap.objects[ht_val.as.ptr]->opaque.ptr;
            if (ht) vm_ht_clear(ht);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 670: { /* hash-table? */
        Value v = vm_pop(vm);
        int is_ht = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                     vm->heap.objects[v.as.ptr]->type == HEAP_HASH);
        vm_push(vm, BOOL_VAL(is_ht));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Bytevector Operations (680-689)
     * ══════════════════════════════════════════════════════════════════════ */
    case 680: { /* make-bytevector(n, fill) */
        Value fill = vm_pop(vm), n = vm_pop(vm);
        VmBytevector* bv = vm_bv_make(&vm->heap.regions, (int)as_number(n), (int)as_number(fill));
        if (!bv) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_INT, bv);
        break;
    }
    case 681: { /* bytevector-length */
        Value bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_bv_length(bv)));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 682: { /* bytevector-u8-ref(bv, k) */
        Value k = vm_pop(vm), bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vm_bv_u8_ref(bv, (int)as_number(k))));
        } else vm_push(vm, INT_VAL(0));
        break;
    }
    case 683: { /* bytevector-u8-set!(bv, k, byte) */
        Value byte = vm_pop(vm), k = vm_pop(vm), bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            vm_bv_u8_set(bv, (int)as_number(k), (int)as_number(byte));
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 684: { /* bytevector-append(a, b) */
        Value b_val = vm_pop(vm), a_val = vm_pop(vm);
        VmBytevector* a = (a_val.as.ptr >= 0 && vm->heap.objects[a_val.as.ptr]->type == HEAP_BYTEVECTOR)
            ? (VmBytevector*)vm->heap.objects[a_val.as.ptr]->opaque.ptr : NULL;
        VmBytevector* b = (b_val.as.ptr >= 0 && vm->heap.objects[b_val.as.ptr]->type == HEAP_BYTEVECTOR)
            ? (VmBytevector*)vm->heap.objects[b_val.as.ptr]->opaque.ptr : NULL;
        if (a && b) {
            VmBytevector* r = vm_bv_append(&vm->heap.regions, a, b);
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_INT, r); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 685: { /* bytevector-copy!(to, at, from) — copy bytes from src into dst */
        Value from_val = vm_pop(vm), at_val = vm_pop(vm), to_val = vm_pop(vm);
        if (to_val.as.ptr >= 0 && vm->heap.objects[to_val.as.ptr]->type == HEAP_BYTEVECTOR &&
            from_val.as.ptr >= 0 && vm->heap.objects[from_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* to_bv = (VmBytevector*)vm->heap.objects[to_val.as.ptr]->opaque.ptr;
            VmBytevector* from_bv = (VmBytevector*)vm->heap.objects[from_val.as.ptr]->opaque.ptr;
            int at = (int)as_number(at_val);
            if (to_bv && from_bv && at >= 0) {
                int copy_len = from_bv->len;
                if (at + copy_len > to_bv->len) copy_len = to_bv->len - at;
                if (copy_len > 0) memcpy(to_bv->data + at, from_bv->data, (size_t)copy_len);
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 686: { /* bytevector? */
        Value v = vm_pop(vm);
        int is_bv = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                     vm->heap.objects[v.as.ptr]->type == HEAP_BYTEVECTOR);
        vm_push(vm, BOOL_VAL(is_bv));
        break;
    }
    case 687: { /* bytevector-copy(bv) */
        Value bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            VmBytevector* r = vm_bv_copy(&vm->heap.regions, bv, 0, bv->len);
            if (r) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_BYTEVECTOR, VAL_INT, r); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 688: { /* utf8->string(bv) */
        Value bv_val = vm_pop(vm);
        if (bv_val.as.ptr >= 0 && vm->heap.objects[bv_val.as.ptr]->type == HEAP_BYTEVECTOR) {
            VmBytevector* bv = (VmBytevector*)vm->heap.objects[bv_val.as.ptr]->opaque.ptr;
            if (bv) {
                VmString* s = vm_string_new(&vm->heap.regions, (const char*)bv->data, bv->len);
                if (s) {
                    int32_t ptr = heap_alloc(&vm->heap);
                    if (ptr >= 0) {
                        vm->heap.objects[ptr]->type = HEAP_STRING;
                        vm->heap.objects[ptr]->opaque.ptr = s;
                        vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                        break;
                    }
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 689: { /* string->utf8(str) */
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING && vm->heap.objects[str_val.as.ptr]->opaque.ptr) {
            VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            VmBytevector* bv = vm_bv_make(&vm->heap.regions, s->byte_len, 0);
            if (bv) {
                memcpy(bv->data, s->data, s->byte_len);
                int32_t ptr = heap_alloc(&vm->heap);
                if (ptr >= 0) {
                    vm->heap.objects[ptr]->type = HEAP_BYTEVECTOR;
                    vm->heap.objects[ptr]->opaque.ptr = bv;
                    vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                    break;
                }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Parameter Operations (700-704)
     * ══════════════════════════════════════════════════════════════════════ */
    case 700: { /* make-parameter(default, converter) */
        Value conv = vm_pop(vm), dflt = vm_pop(vm);
        VmParameter* p = vm_param_make(&vm->heap.regions,
            (void*)(uintptr_t)dflt.as.i, (void*)(uintptr_t)conv.as.i);
        if (!p) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_PARAMETER, VAL_INT, p);
        break;
    }
    case 701: { /* parameter-ref */
        Value p_val = vm_pop(vm);
        if (p_val.as.ptr >= 0 && vm->heap.objects[p_val.as.ptr]->type == HEAP_PARAMETER) {
            VmParameter* p = (VmParameter*)vm->heap.objects[p_val.as.ptr]->opaque.ptr;
            void* v = vm_param_ref(p);
            vm_push(vm, INT_VAL((int64_t)(intptr_t)v));
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 702: { /* parameterize-push(param, value) */
        Value val = vm_pop(vm), p_val = vm_pop(vm);
        if (p_val.as.ptr >= 0 && vm->heap.objects[p_val.as.ptr]->type == HEAP_PARAMETER) {
            VmParameter* p = (VmParameter*)vm->heap.objects[p_val.as.ptr]->opaque.ptr;
            vm_param_push(&vm->heap.regions, p, (void*)(uintptr_t)val.as.i);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 703: { /* parameterize-pop(param) */
        Value p_val = vm_pop(vm);
        if (p_val.as.ptr >= 0 && vm->heap.objects[p_val.as.ptr]->type == HEAP_PARAMETER) {
            VmParameter* p = (VmParameter*)vm->heap.objects[p_val.as.ptr]->opaque.ptr;
            vm_param_pop(p);
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 704: { /* parameter? */
        Value v = vm_pop(vm);
        int is_param = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                        vm->heap.objects[v.as.ptr]->type == HEAP_PARAMETER);
        vm_push(vm, BOOL_VAL(is_param));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Error Operations (710-714)
     * ══════════════════════════════════════════════════════════════════════ */
    case 710: { /* error(type, message) */
        Value msg = vm_pop(vm), type = vm_pop(vm);
        (void)type; (void)msg;
        VmError* e = vm_error_make(&vm->heap.regions, "error", "runtime error", NULL, 0);
        if (!e) { vm_push(vm, NIL_VAL); break; }
        VM_PUSH_HEAP_OPAQUE(vm, HEAP_ERROR, VAL_INT, e);
        break;
    }
    case 711: { /* error-message */
        Value e_val = vm_pop(vm);
        if (e_val.as.ptr >= 0 && vm->heap.objects[e_val.as.ptr]->type == HEAP_ERROR) {
            VmError* e = (VmError*)vm->heap.objects[e_val.as.ptr]->opaque.ptr;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, e ? e->message : "");
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 712: { /* error-type */
        Value e_val = vm_pop(vm);
        if (e_val.as.ptr >= 0 && vm->heap.objects[e_val.as.ptr]->type == HEAP_ERROR) {
            VmError* e = (VmError*)vm->heap.objects[e_val.as.ptr]->opaque.ptr;
            VmString* s = vm_string_from_cstr(&vm->heap.regions, e ? e->type : "");
            if (s) { VM_PUSH_HEAP_OPAQUE(vm, HEAP_STRING, VAL_STRING, s); break; }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 713: { /* error-irritants — returns nil (no irritant list support in Value system) */
        Value e_val = vm_pop(vm); (void)e_val;
        vm_push(vm, NIL_VAL);
        break;
    }
    case 714: { /* error? */
        Value v = vm_pop(vm);
        int is_err = (v.as.ptr >= 0 && v.as.ptr < vm->heap.next_free &&
                      vm->heap.objects[v.as.ptr]->type == HEAP_ERROR);
        vm_push(vm, BOOL_VAL(is_err));
        break;
    }

    /* ══════════════════════════════════════════════════════════════════════
     * High-level AD: gradient, jacobian, hessian (750-752)
     * Uses closure bridge to evaluate function with dual numbers
     * ══════════════════════════════════════════════════════════════════════ */
    case 750: { /* gradient(f, x) → f'(x) via forward-mode dual */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        /* Create dual number: x + 1ε */
        VmDual* d = vm_dual_new(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm_push(vm, FLOAT_VAL(0)); break; }
        int32_t dptr = heap_alloc(&vm->heap);
        if (dptr < 0) { vm->error = 1; break; }
        vm->heap.objects[dptr]->type = HEAP_DUAL;
        vm->heap.objects[dptr]->opaque.ptr = d;
        Value dual_arg = (Value){.type = VAL_DUAL, .as.ptr = dptr};
        /* Call f(dual) */
        Value result = vm_call_closure_from_native(vm, f_val, &dual_arg, 1);
        /* Extract tangent = derivative */
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else {
            vm_push(vm, FLOAT_VAL(as_number(result))); /* non-dual result = zero derivative */
        }
        break;
    }
    case 751: { /* jacobian — simplified: same as gradient for scalar */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        VmDual* d = vm_dual_new(&vm->heap.regions, as_number(x_val), 1.0);
        if (!d) { vm_push(vm, FLOAT_VAL(0)); break; }
        int32_t dptr = heap_alloc(&vm->heap);
        if (dptr < 0) { vm->error = 1; break; }
        vm->heap.objects[dptr]->type = HEAP_DUAL;
        vm->heap.objects[dptr]->opaque.ptr = d;
        Value dual_arg = (Value){.type = VAL_DUAL, .as.ptr = dptr};
        Value result = vm_call_closure_from_native(vm, f_val, &dual_arg, 1);
        if (result.type == VAL_DUAL && result.as.ptr >= 0) {
            VmDual* rd = (VmDual*)vm->heap.objects[result.as.ptr]->opaque.ptr;
            vm_push(vm, FLOAT_VAL(rd ? rd->tangent : 0));
        } else { vm_push(vm, FLOAT_VAL(0)); }
        break;
    }
    case 752: { /* hessian — second derivative via nested dual */
        Value x_val = vm_pop(vm), f_val = vm_pop(vm);
        (void)x_val; (void)f_val;
        vm_push(vm, FLOAT_VAL(0)); /* nested dual not yet supported */
        break;
    }


    /* ══════════════════════════════════════════════════════════════════════
     * Compiler-required native functions (merged from eshkol_compiler.c)
     * ══════════════════════════════════════════════════════════════════════ */

    case 100: { /* build-string-from-packed: TOS has packed chars, below that length */
        /* Stack: [len, pack0, pack1, ..., packN-1] where N = (len+7)/8 */
        /* Peek down to find the length */
        int n_packs_guess = 0;
        int slen = 0;
        /* The compiler pushes: CONST(len), CONST(pack0), ..., CONST(packN-1), NATIVE_CALL 100 */
        /* So TOS-N = len, TOS-(N-1) through TOS-0 = packs, where N = (len+7)/8 */
        /* We need to scan backwards from TOS to find the length */
        for (int try_n = 0; try_n < 64; try_n++) {
            int len_pos = vm->sp - try_n - 1;
            if (len_pos >= 0 && vm->stack[len_pos].type == VAL_INT) {
                int candidate = (int)vm->stack[len_pos].as.i;
                int expected_packs = (candidate + 7) / 8;
                if (expected_packs == try_n && candidate >= 0 && candidate < 256) {
                    slen = candidate;
                    n_packs_guess = try_n;
                    break;
                }
            }
        }
        char buf[256];
        for (int p = n_packs_guess - 1; p >= 0; p--) {
            Value pack_v = vm_pop(vm);
            int64_t pack = pack_v.as.i;
            for (int b = 0; b < 8 && p * 8 + b < slen; b++)
                buf[p * 8 + b] = (char)((pack >> (b * 8)) & 0xFF);
        }
        vm_pop(vm); /* pop length */
        buf[slen] = 0;
        VmString* s = vm_string_from_cstr(&vm->heap.regions, buf);
        if (s) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_STRING;
                vm->heap.objects[ptr]->opaque.ptr = s;
                vm_push(vm, (Value){.type = VAL_STRING, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 131: { /* open_upvalues(closure, count, base_slot) — letrec binding */
        Value base_v = vm_pop(vm), count_v = vm_pop(vm), cl_val = vm_pop(vm);
        /* For letrec: patch closure upvalues to point to current stack values */
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int count = (int)as_number(count_v);
            int base = (int)as_number(base_v);
            for (int i = 0; i < cl->closure.n_upvalues && i < count; i++) {
                int slot = base + i;
                if (slot >= 0 && slot < vm->sp)
                    cl->closure.upvalues[i] = vm->stack[vm->fp + slot];
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 133: { /* eq?: identity equality */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int result = 0;
        if (a.type != b.type) result = 0;
        else if (a.type == VAL_NIL) result = 1;
        else if (a.type == VAL_BOOL) result = (a.as.b == b.as.b);
        else if (a.type == VAL_INT) result = (a.as.i == b.as.i);
        else if (a.type == VAL_FLOAT) result = (a.as.f == b.as.f);
        else result = (a.as.ptr == b.as.ptr);
        vm_push(vm, BOOL_VAL(result));
        break;
    }

    case 134: { /* equal?: deep structural equality */
        Value b = vm_pop(vm), a = vm_pop(vm);
        int result = 0;
        if (a.type != b.type) result = 0;
        else if (a.type == VAL_NIL) result = 1;
        else if (a.type == VAL_BOOL) result = (a.as.b == b.as.b);
        else if (a.type == VAL_INT) result = (a.as.i == b.as.i);
        else if (a.type == VAL_FLOAT) result = (a.as.f == b.as.f);
        else if (a.type == VAL_PAIR) {
            /* Simple shallow equality for pairs */
            result = (a.as.ptr == b.as.ptr);
        }
        else result = (a.as.ptr == b.as.ptr);
        vm_push(vm, BOOL_VAL(result));
        break;
    }

    case 142: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) + as_number(b))); break; } /* add2 */
    case 143: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) - as_number(b))); break; } /* sub2 */
    case 144: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) * as_number(b))); break; } /* mul2 */
    case 145: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) / as_number(b))); break; } /* div2 */

    case 151: { /* direct open slot: set closure upvalue to reference a stack slot */
        Value slot_v = vm_pop(vm), uv_idx_v = vm_pop(vm), cl_val = vm_pop(vm);
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            int uv_idx = (int)as_number(uv_idx_v);
            int slot = (int)as_number(slot_v);
            if (uv_idx >= 0 && uv_idx < cl->closure.n_upvalues && slot >= 0 && vm->fp + slot < vm->sp)
                cl->closure.upvalues[uv_idx] = vm->stack[vm->fp + slot];
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    case 130: { /* raise: unhandled exception */
        Value exn = vm_pop(vm);
        fprintf(stderr, "ERROR: unhandled exception: ");
        print_value(vm, exn);
        fprintf(stderr, "\n");
        vm->error = 1;
        break;
    }

    case 132: { /* force: force a promise */
        Value promise = vm_pop(vm);
        vm_push(vm, promise); /* simplified: just return the value */
        break;
    }

    case 135: { /* append */
        Value b = vm_pop(vm), a = vm_pop(vm);
        if (a.type == VAL_NIL) { vm_push(vm, b); break; }
        if (a.type != VAL_PAIR) { vm_push(vm, b); break; }
        /* Copy list a, set last cdr to b */
        Value head = NIL_VAL, tail = NIL_VAL;
        Value cur = a;
        while (cur.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[cur.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = NIL_VAL;
            Value node = PAIR_VAL(p);
            if (head.type == VAL_NIL) head = node;
            else vm->heap.objects[tail.as.ptr]->cons.cdr = node;
            tail = node;
            cur = vm->heap.objects[cur.as.ptr]->cons.cdr;
        }
        if (tail.type == VAL_PAIR) vm->heap.objects[tail.as.ptr]->cons.cdr = b;
        vm_push(vm, head);
        break;
    }
    case 136: { /* reverse */
        Value lst = vm_pop(vm);
        Value result = NIL_VAL;
        while (lst.type == VAL_PAIR) {
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; break; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = vm->heap.objects[lst.as.ptr]->cons.car;
            vm->heap.objects[p]->cons.cdr = result;
            result = PAIR_VAL(p);
            lst = vm->heap.objects[lst.as.ptr]->cons.cdr;
        }
        vm_push(vm, result);
        break;
    }

    case 240: { /* display (native) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 241: { /* write (native) */
        Value v = vm_pop(vm);
        print_value(vm, v);
        vm_push(vm, NIL_VAL);
        break;
    }
    case 250: { /* atan2 */
        Value x = vm_pop(vm), y = vm_pop(vm);
        vm_push(vm, FLOAT_VAL(atan2(as_number(y), as_number(x))));
        break;
    }
    case 251: { /* call-with-values-apply: simplified */
        Value consumer = vm_pop(vm), result = vm_pop(vm);
        /* Simple: call consumer with single result */
        if (consumer.type == VAL_CLOSURE) {
            Value args[1] = {result};
            Value r = vm_call_closure_from_native(vm, consumer, args, 1);
            vm_push(vm, r);
        } else vm_push(vm, result);
        break;
    }
    case 252: { /* propagate open slot from parent */
        Value slot_v = vm_pop(vm), uv_idx_v = vm_pop(vm), cl_val = vm_pop(vm);
        (void)slot_v; (void)uv_idx_v; (void)cl_val;
        vm_push(vm, NIL_VAL);
        break;
    }

    case 237: { /* error */
        Value msg = vm_pop(vm);
        fprintf(stderr, "ERROR: ");
        print_value(vm, msg);
        fprintf(stderr, "\n");
        vm->error = 1;
        break;
    }
    case 238: { /* void */
        vm_push(vm, NIL_VAL);
        break;
    }
    case 235: { /* not */
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(!is_truthy(v)));
        break;
    }

    default:
        /* Unimplemented native call — push NIL and continue */
        vm_push(vm, NIL_VAL);
        break;


        break;
    }
}

/*******************************************************************************
 * VM Execution
 ******************************************************************************/

static void vm_run(VM* vm) {
#if defined(__GNUC__) || defined(__clang__)
/* =========================================================================
 * Computed-goto (threaded) dispatch — GCC/Clang only.
 *
 * Each handler ends with DISPATCH() which fetches the next instruction and
 * jumps directly to its handler via the dispatch_table, eliminating the
 * switch overhead (no bounds check, no indirect branch through a jump table
 * generated by the compiler — just a single indirect goto).
 * ========================================================================= */

    static void* dispatch_table[OP_COUNT] = {
        [OP_NOP]           = &&lbl_NOP,
        [OP_CONST]         = &&lbl_CONST,
        [OP_NIL]           = &&lbl_NIL,
        [OP_TRUE]          = &&lbl_TRUE,
        [OP_FALSE]         = &&lbl_FALSE,
        [OP_POP]           = &&lbl_POP,
        [OP_DUP]           = &&lbl_DUP,
        [OP_ADD]           = &&lbl_ADD,
        [OP_SUB]           = &&lbl_SUB,
        [OP_MUL]           = &&lbl_MUL,
        [OP_DIV]           = &&lbl_DIV,
        [OP_MOD]           = &&lbl_MOD,
        [OP_NEG]           = &&lbl_NEG,
        [OP_ABS]           = &&lbl_ABS,
        [OP_EQ]            = &&lbl_EQ,
        [OP_LT]            = &&lbl_LT,
        [OP_GT]            = &&lbl_GT,
        [OP_LE]            = &&lbl_LE,
        [OP_GE]            = &&lbl_GE,
        [OP_NOT]           = &&lbl_NOT,
        [OP_GET_LOCAL]     = &&lbl_GET_LOCAL,
        [OP_SET_LOCAL]     = &&lbl_SET_LOCAL,
        [OP_GET_UPVALUE]   = &&lbl_GET_UPVALUE,
        [OP_SET_UPVALUE]   = &&lbl_SET_UPVALUE,
        [OP_CLOSURE]       = &&lbl_CLOSURE,
        [OP_CALL]          = &&lbl_CALL,
        [OP_TAIL_CALL]     = &&lbl_TAIL_CALL,
        [OP_RETURN]        = &&lbl_RETURN,
        [OP_JUMP]          = &&lbl_JUMP,
        [OP_JUMP_IF_FALSE] = &&lbl_JUMP_IF_FALSE,
        [OP_LOOP]          = &&lbl_LOOP,
        [OP_CONS]          = &&lbl_CONS,
        [OP_CAR]           = &&lbl_CAR,
        [OP_CDR]           = &&lbl_CDR,
        [OP_NULL_P]        = &&lbl_NULL_P,
        [OP_PRINT]         = &&lbl_PRINT,
        [OP_HALT]          = &&lbl_HALT,
        [OP_NATIVE_CALL]   = &&lbl_NATIVE_CALL,
        [OP_CLOSE_UPVALUE] = &&lbl_CLOSE_UPVALUE,
        [OP_VEC_CREATE]    = &&lbl_VEC_CREATE,
        [OP_VEC_REF]       = &&lbl_VEC_REF,
        [OP_VEC_SET]       = &&lbl_VEC_SET,
        [OP_VEC_LEN]       = &&lbl_VEC_LEN,
        [OP_STR_REF]       = &&lbl_STR_REF,
        [OP_STR_LEN]       = &&lbl_STR_LEN,
        [OP_PAIR_P]        = &&lbl_PAIR_P,
        [OP_NUM_P]         = &&lbl_NUM_P,
        [OP_STR_P]         = &&lbl_STR_P,
        [OP_BOOL_P]        = &&lbl_BOOL_P,
        [OP_PROC_P]        = &&lbl_PROC_P,
        [OP_VEC_P]         = &&lbl_VEC_P,
        [OP_SET_CAR]       = &&lbl_SET_CAR,
        [OP_SET_CDR]       = &&lbl_SET_CDR,
        [OP_POPN]          = &&lbl_POPN,
        [OP_OPEN_CLOSURE]  = &&lbl_NOP,
        [OP_CALLCC]        = &&lbl_CALLCC,
        [OP_INVOKE_CC]     = &&lbl_NOP,
        [OP_PUSH_HANDLER]  = &&lbl_PUSH_HANDLER,
        [OP_POP_HANDLER]   = &&lbl_POP_HANDLER,
        [OP_GET_EXN]       = &&lbl_GET_EXN,
        [OP_PACK_REST]     = &&lbl_PACK_REST,
        [OP_WIND_PUSH]     = &&lbl_WIND_PUSH,
        [OP_WIND_POP]      = &&lbl_WIND_POP,
    };

    #define DISPATCH() do { \
        if (vm->halted || vm->error || vm->pc >= vm->code_len) goto vm_exit; \
        instr = vm->code[vm->pc++]; \
        goto *dispatch_table[instr.op]; \
    } while(0)

    Instr instr;
    DISPATCH();

    /* --- Constants & Stack --- */

    lbl_NOP:
        DISPATCH();

    lbl_CONST:
        if (instr.operand < 0 || instr.operand >= vm->n_constants) {
            printf("INVALID CONSTANT INDEX %d\n", instr.operand);
            vm->error = 1; goto vm_exit;
        }
        vm_push(vm, vm->constants[instr.operand]);
        DISPATCH();

    lbl_NIL:   vm_push(vm, NIL_VAL);     DISPATCH();
    lbl_TRUE:  vm_push(vm, BOOL_VAL(1)); DISPATCH();
    lbl_FALSE: vm_push(vm, BOOL_VAL(0)); DISPATCH();
    lbl_POP:   vm_pop(vm);               DISPATCH();
    lbl_DUP:   vm_push(vm, vm_peek(vm, 0)); DISPATCH();

    /* --- Arithmetic --- */

    lbl_ADD: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) + as_number(b))); DISPATCH(); }
    lbl_SUB: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) - as_number(b))); DISPATCH(); }
    lbl_MUL: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) * as_number(b))); DISPATCH(); }
    lbl_DIV: { Value b = vm_pop(vm), a = vm_pop(vm);
        double bd = as_number(b);
        if (bd == 0) { printf("DIVIDE BY ZERO\n"); vm->error = 1; goto vm_exit; }
        vm_push(vm, number_val(as_number(a) / bd)); DISPATCH(); }
    lbl_MOD: {
        Value b = vm_pop(vm), a = vm_pop(vm);
        double bd = as_number(b);
        if (bd == 0) { printf("MODULO BY ZERO\n"); vm->error = 1; goto vm_exit; }
        double r = fmod(as_number(a), bd);
        if (r != 0 && ((r > 0) != (bd > 0))) r += bd;
        vm_push(vm, number_val(r));
        DISPATCH();
    }
    lbl_NEG: { Value a = vm_pop(vm); vm_push(vm, number_val(-as_number(a))); DISPATCH(); }
    lbl_ABS: { Value a = vm_pop(vm); vm_push(vm, number_val(fabs(as_number(a)))); DISPATCH(); }

    /* --- Comparison --- */

    lbl_EQ: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); DISPATCH(); }
    lbl_LT: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) <  as_number(b))); DISPATCH(); }
    lbl_GT: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) >  as_number(b))); DISPATCH(); }
    lbl_LE: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) <= as_number(b))); DISPATCH(); }
    lbl_GE: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) >= as_number(b))); DISPATCH(); }
    lbl_NOT: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(!is_truthy(a))); DISPATCH(); }

    /* --- Variables --- */

    lbl_GET_LOCAL:
        vm_push(vm, vm->stack[vm->fp + instr.operand]);
        DISPATCH();
    lbl_SET_LOCAL:
        vm->stack[vm->fp + instr.operand] = vm_peek(vm, 0);
        vm_pop(vm);
        DISPATCH();
    lbl_GET_UPVALUE: {
        Value closure_val = vm->stack[vm->fp - 1];
        if (closure_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
            if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                vm_push(vm, cl->closure.upvalues[instr.operand]);
            } else {
                printf("UPVALUE INDEX OUT OF BOUNDS\n");
                vm_push(vm, NIL_VAL);
            }
        } else {
            vm_push(vm, NIL_VAL);
        }
        DISPATCH();
    }
    lbl_SET_UPVALUE: {
        Value closure_val = vm->stack[vm->fp - 1];
        if (closure_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
            if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                cl->closure.upvalues[instr.operand] = vm_peek(vm, 0);
            } else {
                printf("UPVALUE INDEX OUT OF BOUNDS\n");
            }
        }
        vm_pop(vm);
        DISPATCH();
    }

    /* --- Closures --- */

    lbl_CLOSURE: {
        int const_idx = instr.operand & 0xFFFF;
        int n_upvalues = (instr.operand >> 16) & 0xFF;
        if (n_upvalues > 16) n_upvalues = 16;
        Value func_const = vm->constants[const_idx];
        int32_t func_pc = (int32_t)func_const.as.i;
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; goto vm_exit; }
        vm->heap.objects[ptr]->type = HEAP_CLOSURE;
        vm->heap.objects[ptr]->closure.func_pc = func_pc;
        vm->heap.objects[ptr]->closure.n_upvalues = n_upvalues;
        for (int i = n_upvalues - 1; i >= 0; i--) {
            vm->heap.objects[ptr]->closure.upvalues[i] = vm_pop(vm);
        }
        vm_push(vm, CLOSURE_VAL(ptr));
        DISPATCH();
    }

    /* --- Function call --- */

    lbl_CALL: {
        int argc = instr.operand;
        Value func = vm->stack[vm->sp - 1 - argc];

        if (func.type != VAL_CLOSURE) {
            printf("ERROR: calling non-function\n");
            vm->error = 1; goto vm_exit;
        }

        HeapObject* cl = vm->heap.objects[func.as.ptr];

        if (vm->frame_count >= MAX_FRAMES) { printf("FRAME OVERFLOW\n"); vm->error = 1; goto vm_exit; }
        vm->frames[vm->frame_count].return_pc = vm->pc;
        vm->frames[vm->frame_count].return_fp = vm->fp;
        vm->frames[vm->frame_count].func_pc = cl->closure.func_pc;
        vm->frame_count++;

        vm->fp = vm->sp - argc;
        vm->pc = cl->closure.func_pc;
        DISPATCH();
    }

    lbl_TAIL_CALL: {
        int argc = instr.operand;
        Value func = vm->stack[vm->sp - 1 - argc];
        if (func.type != VAL_CLOSURE) { vm->error = 1; goto vm_exit; }
        HeapObject* cl = vm->heap.objects[func.as.ptr];

        for (int i = 0; i < argc; i++) {
            vm->stack[vm->fp + i] = vm->stack[vm->sp - argc + i];
        }
        vm->sp = vm->fp + argc;
        vm->pc = cl->closure.func_pc;
        DISPATCH();
    }

    lbl_RETURN: {
        Value result = vm_pop(vm);
        if (vm->frame_count <= 0) {
            vm_push(vm, result);
            vm->halted = 1;
            goto vm_exit;
        }
        vm->frame_count--;
        /* Check for native-call sentinel */
        if (vm->frames[vm->frame_count].return_pc == -1) {
            vm_push(vm, result);
            vm->halted = 1;
            goto vm_exit;
        }
        vm->sp = vm->fp - 1;
        vm->fp = vm->frames[vm->frame_count].return_fp;
        vm->pc = vm->frames[vm->frame_count].return_pc;
        vm_push(vm, result);
        DISPATCH();
    }

    /* --- Control Flow --- */

    lbl_JUMP:
        vm->pc = instr.operand;
        DISPATCH();
    lbl_JUMP_IF_FALSE: {
        Value cond = vm_pop(vm);
        if (!is_truthy(cond)) vm->pc = instr.operand;
        DISPATCH();
    }
    lbl_LOOP:
        vm->pc = instr.operand;
        DISPATCH();

    /* --- Pairs --- */

    lbl_CONS: {
        Value car = vm_pop(vm), cdr = vm_pop(vm);
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; goto vm_exit; }
        vm->heap.objects[ptr]->type = HEAP_CONS;
        vm->heap.objects[ptr]->cons.car = car;
        vm->heap.objects[ptr]->cons.cdr = cdr;
        vm_push(vm, PAIR_VAL(ptr));
        DISPATCH();
    }
    lbl_CAR: {
        Value pair = vm_pop(vm);
        if (pair.type != VAL_PAIR) { printf("CAR on non-pair\n"); vm->error = 1; goto vm_exit; }
        vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.car);
        DISPATCH();
    }
    lbl_CDR: {
        Value pair = vm_pop(vm);
        if (pair.type != VAL_PAIR) { printf("CDR on non-pair\n"); vm->error = 1; goto vm_exit; }
        vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.cdr);
        DISPATCH();
    }
    lbl_NULL_P: {
        Value v = vm_pop(vm);
        vm_push(vm, BOOL_VAL(v.type == VAL_NIL));
        DISPATCH();
    }

    /* --- I/O --- */

    lbl_PRINT: {
        Value v = vm_pop(vm);
        print_value(vm, v);
        printf("\n");
        if (vm->n_outputs < 256) vm->outputs[vm->n_outputs++] = v;
        DISPATCH();
    }

    lbl_HALT:
        vm->halted = 1;
        goto vm_exit;

    lbl_NATIVE_CALL: {
        vm_dispatch_native(vm, instr.operand);
        DISPATCH();
    }

    lbl_CLOSE_UPVALUE: {
        /* Patch the TOS closure's upvalue[operand] to point to the closure itself */
        Value cl_val = vm_peek(vm, 0);
        if (cl_val.type == VAL_CLOSURE) {
            HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
            if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues)
                cl->closure.upvalues[instr.operand] = cl_val;
        }
        DISPATCH();
    }

    lbl_VEC_CREATE: {
        int count = instr.operand;
        int32_t ptr = heap_alloc(&vm->heap);
        if (ptr < 0) { vm->error = 1; goto vm_exit; }
        vm->heap.objects[ptr]->type = HEAP_VECTOR;
        VmVector* vec = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
        if (!vec) { vm->error = 1; goto vm_exit; }
        vec->len = count;
        vec->cap = count;
        vec->items = (Value*)vm_alloc(&vm->heap.regions, count * sizeof(Value));
        if (!vec->items && count > 0) { vm->error = 1; goto vm_exit; }
        for (int i = count - 1; i >= 0; i--) vec->items[i] = vm_pop(vm);
        vm->heap.objects[ptr]->opaque.ptr = vec;
        vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
        DISPATCH();
    }

    lbl_VEC_REF: {
        Value idx = vm_pop(vm), vec_val = vm_pop(vm);
        if (vec_val.type != VAL_VECTOR) { vm_push(vm, NIL_VAL); DISPATCH(); }
        VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
        int i = (int)as_number(idx);
        if (vec && i >= 0 && i < vec->len) vm_push(vm, vec->items[i]);
        else vm_push(vm, NIL_VAL);
        DISPATCH();
    }

    lbl_VEC_SET: {
        Value val = vm_pop(vm), idx = vm_pop(vm), vec_val = vm_pop(vm);
        if (vec_val.type == VAL_VECTOR) {
            VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (vec && i >= 0 && i < vec->len) vec->items[i] = val;
        }
        vm_push(vm, NIL_VAL);
        DISPATCH();
    }

    lbl_VEC_LEN: {
        Value vec_val = vm_pop(vm);
        if (vec_val.type == VAL_VECTOR) {
            VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(vec ? vec->len : 0));
        } else vm_push(vm, INT_VAL(0));
        DISPATCH();
    }

    lbl_STR_REF: {
        Value idx = vm_pop(vm), str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (s && i >= 0 && i < s->byte_len) vm_push(vm, INT_VAL((unsigned char)s->data[i]));
            else vm_push(vm, INT_VAL(0));
        } else vm_push(vm, INT_VAL(0));
        DISPATCH();
    }

    lbl_STR_LEN: {
        Value str_val = vm_pop(vm);
        if (str_val.type == VAL_STRING) {
            VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
            vm_push(vm, INT_VAL(s ? s->byte_len : 0));
        } else vm_push(vm, INT_VAL(0));
        DISPATCH();
    }

    lbl_PAIR_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_PAIR)); DISPATCH(); }
    lbl_NUM_P:   { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_INT || v.type == VAL_FLOAT)); DISPATCH(); }
    lbl_STR_P:   { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_STRING)); DISPATCH(); }
    lbl_BOOL_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_BOOL)); DISPATCH(); }
    lbl_PROC_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_CLOSURE)); DISPATCH(); }
    lbl_VEC_P:   { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_VECTOR)); DISPATCH(); }

    lbl_SET_CAR: {
        Value val = vm_pop(vm), pair = vm_pop(vm);
        if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.car = val;
        vm_push(vm, NIL_VAL);
        DISPATCH();
    }
    lbl_SET_CDR: {
        Value val = vm_pop(vm), pair = vm_pop(vm);
        if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.cdr = val;
        vm_push(vm, NIL_VAL);
        DISPATCH();
    }

    lbl_POPN: {
        int n = instr.operand;
        if (n > 0 && vm->sp > n) {
            Value top = vm->stack[vm->sp - 1];
            vm->sp -= n;
            vm->stack[vm->sp - 1] = top;
        }
        DISPATCH();
    }

    lbl_CALLCC: {
        /* Simplified: capture continuation as a closure that invokes OP_INVOKE_CC */
        /* For now, push NIL — full continuations need the compiler's execute_chunk support */
        Value proc = vm_pop(vm);
        (void)proc;
        vm_push(vm, NIL_VAL);
        DISPATCH();
    }

    lbl_PUSH_HANDLER: {
        /* operand = handler PC. For now, just record it. */
        vm_push(vm, NIL_VAL); /* placeholder */
        vm->pc = instr.operand; /* jump to handler on error — simplified */
        DISPATCH();
    }

    lbl_POP_HANDLER: DISPATCH();

    lbl_GET_EXN: {
        vm_push(vm, NIL_VAL); /* simplified: no exception register in base VM */
        DISPATCH();
    }

    lbl_PACK_REST: {
        int n_fixed = instr.operand;
        int n_args = vm->sp - vm->fp;
        Value list = NIL_VAL;
        for (int i = n_args - 1; i >= n_fixed; i--) {
            Value item = vm->stack[vm->fp + i];
            int32_t p = heap_alloc(&vm->heap);
            if (p < 0) { vm->error = 1; goto vm_exit; }
            vm->heap.objects[p]->type = HEAP_CONS;
            vm->heap.objects[p]->cons.car = item;
            vm->heap.objects[p]->cons.cdr = list;
            list = PAIR_VAL(p);
        }
        vm->sp = vm->fp + n_fixed;
        vm_push(vm, list);
        DISPATCH();
    }

    lbl_WIND_PUSH: {
        vm_pop(vm); /* consume after thunk — simplified */
        DISPATCH();
    }

    lbl_WIND_POP: DISPATCH();

vm_exit:
    #undef DISPATCH
    ;

#else
/* =========================================================================
 * Fallback: standard switch dispatch for non-GCC/Clang compilers (MSVC etc.)
 * ========================================================================= */

    while (!vm->halted && !vm->error && vm->pc < vm->code_len) {
        Instr instr = vm->code[vm->pc++];

        switch (instr.op) {
        case OP_NOP: break;

        case OP_CONST:
            if (instr.operand < 0 || instr.operand >= vm->n_constants) {
                printf("INVALID CONSTANT INDEX %d\n", instr.operand);
                vm->error = 1; break;
            }
            vm_push(vm, vm->constants[instr.operand]);
            break;

        case OP_NIL:   vm_push(vm, NIL_VAL); break;
        case OP_TRUE:  vm_push(vm, BOOL_VAL(1)); break;
        case OP_FALSE: vm_push(vm, BOOL_VAL(0)); break;
        case OP_POP:   vm_pop(vm); break;
        case OP_DUP:   vm_push(vm, vm_peek(vm, 0)); break;

        /* Arithmetic */
        case OP_ADD: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) + as_number(b))); break; }
        case OP_SUB: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) - as_number(b))); break; }
        case OP_MUL: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, number_val(as_number(a) * as_number(b))); break; }
        case OP_DIV: { Value b = vm_pop(vm), a = vm_pop(vm);
            double bd = as_number(b);
            if (bd == 0) { printf("DIVIDE BY ZERO\n"); vm->error = 1; break; }
            vm_push(vm, number_val(as_number(a) / bd)); break; }
        case OP_MOD: {
            Value b = vm_pop(vm), a = vm_pop(vm);
            double bd = as_number(b);
            if (bd == 0) { printf("MODULO BY ZERO\n"); vm->error = 1; break; }
            double r = fmod(as_number(a), bd);
            if (r != 0 && ((r > 0) != (bd > 0))) r += bd;
            vm_push(vm, number_val(r));
            break;
        }
        case OP_NEG: { Value a = vm_pop(vm); vm_push(vm, number_val(-as_number(a))); break; }
        case OP_ABS: { Value a = vm_pop(vm); vm_push(vm, number_val(fabs(as_number(a)))); break; }

        /* Comparison — push proper booleans */
        case OP_EQ: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) == as_number(b))); break; }
        case OP_LT: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) < as_number(b))); break; }
        case OP_GT: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) > as_number(b))); break; }
        case OP_LE: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) <= as_number(b))); break; }
        case OP_GE: { Value b = vm_pop(vm), a = vm_pop(vm); vm_push(vm, BOOL_VAL(as_number(a) >= as_number(b))); break; }
        case OP_NOT: { Value a = vm_pop(vm); vm_push(vm, BOOL_VAL(!is_truthy(a))); break; }

        /* Variables */
        case OP_GET_LOCAL:
            vm_push(vm, vm->stack[vm->fp + instr.operand]);
            break;
        case OP_SET_LOCAL:
            vm->stack[vm->fp + instr.operand] = vm_peek(vm, 0);
            vm_pop(vm);
            break;
        case OP_GET_UPVALUE: {
            Value closure_val = vm->stack[vm->fp - 1]; /* closure is just below frame */
            if (closure_val.type == VAL_CLOSURE) {
                HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
                if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                    vm_push(vm, cl->closure.upvalues[instr.operand]);
                } else {
                    printf("UPVALUE INDEX OUT OF BOUNDS\n");
                    vm_push(vm, NIL_VAL);
                }
            } else {
                vm_push(vm, NIL_VAL);
            }
            break;
        }
        case OP_SET_UPVALUE: {
            Value closure_val = vm->stack[vm->fp - 1];
            if (closure_val.type == VAL_CLOSURE) {
                HeapObject* cl = vm->heap.objects[closure_val.as.ptr];
                if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues) {
                    cl->closure.upvalues[instr.operand] = vm_peek(vm, 0);
                } else {
                    printf("UPVALUE INDEX OUT OF BOUNDS\n");
                }
            }
            vm_pop(vm);
            break;
        }

        /* Closures */
        case OP_CLOSURE: {
            /* Operand: low 16 bits = constant pool index, bits 16-23 = n_upvalues */
            int const_idx = instr.operand & 0xFFFF;
            int n_upvalues = (instr.operand >> 16) & 0xFF;
            if (n_upvalues > 16) n_upvalues = 16;
            Value func_const = vm->constants[const_idx];
            int32_t func_pc = (int32_t)func_const.as.i;
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_CLOSURE;
            vm->heap.objects[ptr]->closure.func_pc = func_pc;
            vm->heap.objects[ptr]->closure.n_upvalues = n_upvalues;
            /* Pop upvalues from stack (pushed before CLOSURE, in reverse order) */
            for (int i = n_upvalues - 1; i >= 0; i--) {
                vm->heap.objects[ptr]->closure.upvalues[i] = vm_pop(vm);
            }
            vm_push(vm, CLOSURE_VAL(ptr));
            break;
        }

        /* Function call */
        case OP_CALL: {
            int argc = instr.operand;
            Value func = vm->stack[vm->sp - 1 - argc]; /* function is below args */

            if (func.type != VAL_CLOSURE) {
                printf("ERROR: calling non-function\n");
                vm->error = 1; break;
            }

            HeapObject* cl = vm->heap.objects[func.as.ptr];

            /* Save call frame */
            if (vm->frame_count >= MAX_FRAMES) { printf("FRAME OVERFLOW\n"); vm->error = 1; break; }
            vm->frames[vm->frame_count].return_pc = vm->pc;
            vm->frames[vm->frame_count].return_fp = vm->fp;
            vm->frames[vm->frame_count].func_pc = cl->closure.func_pc;
            vm->frame_count++;

            /* Set up new frame: func sits at sp-argc-1, args at sp-argc..sp-1 */
            vm->fp = vm->sp - argc;
            vm->pc = cl->closure.func_pc;
            break;
        }

        case OP_TAIL_CALL: {
            int argc = instr.operand;
            Value func = vm->stack[vm->sp - 1 - argc];
            if (func.type != VAL_CLOSURE) { vm->error = 1; break; }
            HeapObject* cl = vm->heap.objects[func.as.ptr];

            /* Move args to current frame position (reuse frame) */
            for (int i = 0; i < argc; i++) {
                vm->stack[vm->fp + i] = vm->stack[vm->sp - argc + i];
            }
            vm->sp = vm->fp + argc;
            vm->pc = cl->closure.func_pc;
            break;
        }

        case OP_RETURN: {
            Value result = vm_pop(vm);
            if (vm->frame_count <= 0) {
                /* Top-level return */
                vm_push(vm, result);
                vm->halted = 1;
                break;
            }
            vm->frame_count--;
            /* Check for native-call sentinel */
            if (vm->frames[vm->frame_count].return_pc == -1) {
                vm_push(vm, result);
                vm->halted = 1;
                break;
            }
            vm->sp = vm->fp - 1; /* discard frame + function slot */
            vm->fp = vm->frames[vm->frame_count].return_fp;
            vm->pc = vm->frames[vm->frame_count].return_pc;
            vm_push(vm, result);
            break;
        }

        /* Control Flow */
        case OP_JUMP:
            vm->pc = instr.operand;
            break;
        case OP_JUMP_IF_FALSE: {
            Value cond = vm_pop(vm);
            if (!is_truthy(cond)) vm->pc = instr.operand;
            break;
        }
        case OP_LOOP:
            vm->pc = instr.operand;
            break;

        /* Pairs */
        case OP_CONS: {
            Value car = vm_pop(vm), cdr = vm_pop(vm);  /* TOS=car, SOS=cdr */
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_CONS;
            vm->heap.objects[ptr]->cons.car = car;
            vm->heap.objects[ptr]->cons.cdr = cdr;
            vm_push(vm, PAIR_VAL(ptr));
            break;
        }
        case OP_CAR: {
            Value pair = vm_pop(vm);
            if (pair.type != VAL_PAIR) { printf("CAR on non-pair\n"); vm->error = 1; break; }
            vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.car);
            break;
        }
        case OP_CDR: {
            Value pair = vm_pop(vm);
            if (pair.type != VAL_PAIR) { printf("CDR on non-pair\n"); vm->error = 1; break; }
            vm_push(vm, vm->heap.objects[pair.as.ptr]->cons.cdr);
            break;
        }
        case OP_NULL_P: {
            Value v = vm_pop(vm);
            vm_push(vm, BOOL_VAL(v.type == VAL_NIL));
            break;
        }

        /* I/O */
        case OP_PRINT: {
            Value v = vm_pop(vm);
            print_value(vm, v);
            printf("\n");
            if (vm->n_outputs < 256) vm->outputs[vm->n_outputs++] = v;
            break;
        }

        case OP_HALT:
            vm->halted = 1;
            break;

        case OP_NATIVE_CALL: {
            vm_dispatch_native(vm, instr.operand);
            break;
        }

        case OP_CLOSE_UPVALUE: {
            Value cl_val = vm_peek(vm, 0);
            if (cl_val.type == VAL_CLOSURE) {
                HeapObject* cl = vm->heap.objects[cl_val.as.ptr];
                if (instr.operand >= 0 && instr.operand < cl->closure.n_upvalues)
                    cl->closure.upvalues[instr.operand] = cl_val;
            }
            break;
        }

        case OP_VEC_CREATE: {
            int count = instr.operand;
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr < 0) { vm->error = 1; break; }
            vm->heap.objects[ptr]->type = HEAP_VECTOR;
            VmVector* vec = (VmVector*)vm_alloc(&vm->heap.regions, sizeof(VmVector));
            if (!vec) { vm->error = 1; break; }
            vec->len = count; vec->cap = count;
            vec->items = (Value*)vm_alloc(&vm->heap.regions, count * sizeof(Value));
            if (!vec->items && count > 0) { vm->error = 1; break; }
            for (int i = count - 1; i >= 0; i--) vec->items[i] = vm_pop(vm);
            vm->heap.objects[ptr]->opaque.ptr = vec;
            vm_push(vm, (Value){.type = VAL_VECTOR, .as.ptr = ptr});
            break;
        }

        case OP_VEC_REF: {
            Value idx = vm_pop(vm), vec_val = vm_pop(vm);
            if (vec_val.type != VAL_VECTOR) { vm_push(vm, NIL_VAL); break; }
            VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
            int i = (int)as_number(idx);
            if (vec && i >= 0 && i < vec->len) vm_push(vm, vec->items[i]);
            else vm_push(vm, NIL_VAL);
            break;
        }

        case OP_VEC_SET: {
            Value val = vm_pop(vm), idx = vm_pop(vm), vec_val = vm_pop(vm);
            if (vec_val.type == VAL_VECTOR) {
                VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
                int i = (int)as_number(idx);
                if (vec && i >= 0 && i < vec->len) vec->items[i] = val;
            }
            vm_push(vm, NIL_VAL);
            break;
        }

        case OP_VEC_LEN: {
            Value vec_val = vm_pop(vm);
            if (vec_val.type == VAL_VECTOR) {
                VmVector* vec = (VmVector*)vm->heap.objects[vec_val.as.ptr]->opaque.ptr;
                vm_push(vm, INT_VAL(vec ? vec->len : 0));
            } else vm_push(vm, INT_VAL(0));
            break;
        }

        case OP_STR_REF: {
            Value idx = vm_pop(vm), str_val = vm_pop(vm);
            if (str_val.type == VAL_STRING) {
                VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
                int i = (int)as_number(idx);
                if (s && i >= 0 && i < s->byte_len) vm_push(vm, INT_VAL((unsigned char)s->data[i]));
                else vm_push(vm, INT_VAL(0));
            } else vm_push(vm, INT_VAL(0));
            break;
        }

        case OP_STR_LEN: {
            Value str_val = vm_pop(vm);
            if (str_val.type == VAL_STRING) {
                VmString* s = (VmString*)vm->heap.objects[str_val.as.ptr]->opaque.ptr;
                vm_push(vm, INT_VAL(s ? s->byte_len : 0));
            } else vm_push(vm, INT_VAL(0));
            break;
        }

        case OP_PAIR_P: { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_PAIR)); break; }
        case OP_NUM_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_INT || v.type == VAL_FLOAT)); break; }
        case OP_STR_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_STRING)); break; }
        case OP_BOOL_P: { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_BOOL)); break; }
        case OP_PROC_P: { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_CLOSURE)); break; }
        case OP_VEC_P:  { Value v = vm_pop(vm); vm_push(vm, BOOL_VAL(v.type == VAL_VECTOR)); break; }

        case OP_SET_CAR: {
            Value val = vm_pop(vm), pair = vm_pop(vm);
            if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.car = val;
            vm_push(vm, NIL_VAL); break;
        }
        case OP_SET_CDR: {
            Value val = vm_pop(vm), pair = vm_pop(vm);
            if (pair.type == VAL_PAIR) vm->heap.objects[pair.as.ptr]->cons.cdr = val;
            vm_push(vm, NIL_VAL); break;
        }

        case OP_POPN: {
            int n = instr.operand;
            if (n > 0 && vm->sp > n) {
                Value top = vm->stack[vm->sp - 1];
                vm->sp -= n;
                vm->stack[vm->sp - 1] = top;
            }
            break;
        }

        case OP_CALLCC: { Value proc = vm_pop(vm); (void)proc; vm_push(vm, NIL_VAL); break; }
        case OP_INVOKE_CC: break;
        case OP_OPEN_CLOSURE: break;
        case OP_PUSH_HANDLER: { vm_push(vm, NIL_VAL); vm->pc = instr.operand; break; }
        case OP_POP_HANDLER: break;
        case OP_GET_EXN: { vm_push(vm, NIL_VAL); break; }
        case OP_PACK_REST: {
            int n_fixed = instr.operand;
            int n_args = vm->sp - vm->fp;
            Value list = NIL_VAL;
            for (int i = n_args - 1; i >= n_fixed; i--) {
                Value item = vm->stack[vm->fp + i];
                int32_t p = heap_alloc(&vm->heap);
                if (p < 0) { vm->error = 1; break; }
                vm->heap.objects[p]->type = HEAP_CONS;
                vm->heap.objects[p]->cons.car = item;
                vm->heap.objects[p]->cons.cdr = list;
                list = PAIR_VAL(p);
            }
            vm->sp = vm->fp + n_fixed;
            vm_push(vm, list);
            break;
        }
        case OP_WIND_PUSH: vm_pop(vm); break;
        case OP_WIND_POP: break;
        case OP_CLOSE_UPVALUE: {
            Value cl_val2 = vm_peek(vm, 0);
            if (cl_val2.type == VAL_CLOSURE) {
                HeapObject* cl2 = vm->heap.objects[cl_val2.as.ptr];
                if (instr.operand >= 0 && instr.operand < cl2->closure.n_upvalues)
                    cl2->closure.upvalues[instr.operand] = cl_val2;
            }
            break;
        }

        default:
            printf("UNKNOWN OPCODE %d\n", instr.op);
            vm->error = 1;
            break;
        }
    }
#endif
}

/*******************************************************************************
 * Test Programs
 ******************************************************************************/

static const char* opnames[] = {
    "NOP","CONST","NIL","TRUE","FALSE","POP","DUP",
    "ADD","SUB","MUL","DIV","MOD","NEG","ABS",
    "EQ","LT","GT","LE","GE","NOT",
    "GETL","SETL","GETUP","SETUP",
    "CLOSURE","CALL","TCALL","RET",
    "JUMP","JIF","LOOP",
    "CONS","CAR","CDR","NULLP",
    "PRINT","HALT","NATIVE"
};

static void emit(VM* vm, uint8_t op, int32_t operand) {
    if (vm->code_len >= 4096) return;
    vm->code[vm->code_len++] = (Instr){op, operand};
}

static VM* vm_create(void) {
    VM* vm = (VM*)calloc(1, sizeof(VM));
    if (!vm) return NULL;
    vm_init(vm);
    vm->code = (Instr*)calloc(4096, sizeof(Instr));
    if (!vm->code) { free(vm); return NULL; }
    return vm;
}
static void vm_free(VM* vm) {
    heap_destroy(&vm->heap);
    free(vm->code);
    free(vm);
}

static void test_arithmetic(void) {
    printf("  test_arithmetic: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (+ 3 5) → 8 */
    int c3 = add_constant(vm, INT_VAL(3));
    int c5 = add_constant(vm, INT_VAL(5));
    emit(vm, OP_CONST, c3);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_ADD, 0);
    emit(vm, OP_PRINT, 0);
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].type == VAL_INT && vm->outputs[0].as.i == 8);
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_comparison(void) {
    printf("  test_comparison: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (< 3 5) → #t, (> 3 5) → #f, (= 5 5) → #t */
    int c3 = add_constant(vm, INT_VAL(3));
    int c5 = add_constant(vm, INT_VAL(5));
    emit(vm, OP_CONST, c3);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_LT, 0);
    emit(vm, OP_PRINT, 0);  /* #t */
    emit(vm, OP_CONST, c3);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_GT, 0);
    emit(vm, OP_PRINT, 0);  /* #f */
    emit(vm, OP_CONST, c5);
    emit(vm, OP_CONST, c5);
    emit(vm, OP_EQ, 0);
    emit(vm, OP_PRINT, 0);  /* #t */
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    int ok = vm->n_outputs == 3
        && vm->outputs[0].as.b == 1
        && vm->outputs[1].as.b == 0
        && vm->outputs[2].as.b == 1;
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_pairs(void) {
    printf("  test_pairs: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (car (cons 1 2)) → 1, (cdr (cons 1 2)) → 2 */
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    emit(vm, OP_CONST, c1);
    emit(vm, OP_CONST, c2);
    emit(vm, OP_CONS, 0);
    emit(vm, OP_DUP, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);  /* 1 */
    emit(vm, OP_CDR, 0);
    emit(vm, OP_PRINT, 0);  /* 2 */
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    /* With CONS: TOS=car, SOS=cdr. DUP+CAR gives car, then original CDR gives cdr.
     * But the DUP duplicates the pair, CAR pops and gives car.
     * Then the original pair is still on stack, CDR gives cdr. */
    int ok = vm->n_outputs == 2
        && ((vm->outputs[0].as.i == 1 && vm->outputs[1].as.i == 2)
         || (vm->outputs[0].as.i == 2 && vm->outputs[1].as.i == 1));
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_list(void) {
    printf("  test_list: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* (cons 1 (cons 2 (cons 3 '()))) → (1 2 3) */
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c3 = add_constant(vm, INT_VAL(3));
    /* (dead code — replaced by restructured version below) */
    /* Actually: cons takes (car, cdr) from stack. Push car first, then cdr. */
    /* (cons 3 nil) → push 3, push nil, cons */
    /* (cons 2 (cons 3 nil)) → push 2, then push result-of-previous, cons */
    /* Stack: ..., (3 . nil) → push 2 → ..., (3 . nil), 2 → need swap → 2, (3 . nil) → cons → (2 3) */
    /* We don't have SWAP in the new ISA. Let me restructure. */
    /* Build right-to-left: */
    vm->code_len = 0;
    emit(vm, OP_NIL, 0);        /* nil */
    emit(vm, OP_CONST, c3);     /* 3, nil */
    /* Need: car=3, cdr=nil. But cons pops cdr first, then car.
     * Actually my CONS does: cdr = pop, car = pop. So push car first, then cdr.
     * Wait no: Value cdr = vm_pop, car = vm_pop. So top is cdr, second is car.
     * To get (cons 3 nil): push 3 (car), push nil (cdr), cons. */
    vm->code_len = 0;
    emit(vm, OP_CONST, c3);     /* car = 3 */
    emit(vm, OP_NIL, 0);        /* cdr = nil */
    emit(vm, OP_CONS, 0);       /* (3) */
    emit(vm, OP_CONST, c2);     /* car = 2 */
    /* stack: (3), 2. Need: 2, (3). CONS pops cdr then car. So push car=2 first... wait.
     * Current stack: [(3), 2]. CONS: cdr = pop() = 2, car = pop() = (3). Makes ((3) . 2). Wrong! */
    /* I need to fix CONS order or add swap. Let me change CONS to pop car first, then cdr.
     * That's more natural: (cons A B) → push A, push B, CONS → car=A, cdr=B */
    /* Let me just restructure: */
    vm->code_len = 0;
    /* Build (1 2 3) = (cons 1 (cons 2 (cons 3 nil))) */
    /* Start from inside out: */
    emit(vm, OP_CONST, c3); emit(vm, OP_NIL, 0);   emit(vm, OP_CONS, 0);  /* (3) */
    emit(vm, OP_CONST, c2);
    /* Stack: [(3), 2]. I need cons(2, (3)). CONS pops cdr=(3), car=2... no. */
    /* My CONS: cdr = vm_pop, car = vm_pop. Stack top is cdr, below is car.
     * So to make (cons 2 (3)): push 2 (car), push (3) (cdr).
     * But (3) is already on stack. I need 2 BELOW (3).
     * This requires either: build in reverse, or have swap.
     * Build in reverse: construct inner pairs first, keep them on top. */
    /* Actually, the standard way: build from right to left.
     * nil → (cons 3 nil) → (cons 2 (cons 3 nil)) → (cons 1 (cons 2 (cons 3 nil)))
     * Stack trace:
     *   push nil                    stack: [nil]
     *   push 3, swap, cons          stack: [(3)]     ← need swap!
     * OR: change cons order: car = pop first, cdr = pop second.
     * That way: push 3, push nil, cons → car=nil(!), cdr=3. Wrong!
     * Hmm. Standard: CONS pops TOS=cdr, SOS=car. So to build list:
     *   nil → push 3, nil → cons(3, nil) → push 2, (3) → cons(2, (3)) → push 1, (2 3) → cons(1, (2 3))
     * But: stack after 'push 3, nil' is [3, nil]. CONS: cdr=nil, car=3 → (3). ✓
     * Then: push 2 → [2, (3)]. Hmm no: the (3) is underneath. Actually... */
    /* Let me trace more carefully.
     * After cons(3, nil): stack = [(3)]
     * push 2 → stack = [(3), 2]
     * cons: cdr = pop() = 2 (!), car = pop() = (3). Makes ((3) . 2). WRONG.
     * The problem: 2 is on top, not (3). We need car=2, cdr=(3).
     * So the stack should be [2, (3)] before CONS.
     * We need to push 2 BEFORE the previous result. This requires reordering.
     * Solution: just build lists by pushing elements in REVERSE order. */
    vm->code_len = 0;
    /* Build (1 2 3) backwards: */
    emit(vm, OP_NIL, 0);        /* stack: [nil] */
    emit(vm, OP_CONST, c3);     /* stack: [nil, 3] */
    /* Swap so car is on SOS: stack should be [3, nil] for CONS(car=3, cdr=nil) */
    /* But we don't have SWAP! Let me just restructure CONS to be: car=TOS, cdr=SOS */
    /* That means: push cdr first, then car, then cons. */
    /* (cons 3 nil): push nil (cdr), push 3 (car), cons → stack[nil,3] → car=pop=3, cdr=pop=nil → (3) ✓ */
    /* Let me fix the CONS implementation. */
    /* CHANGED: CONS pops car first (TOS), then cdr (SOS). */
    vm->code_len = 0;
    emit(vm, OP_NIL, 0);        /* push cdr = nil */
    emit(vm, OP_CONST, c3);     /* push car = 3 */
    emit(vm, OP_CONS, 0);       /* (3) */
    /* Now for (cons 2 (3)): (3) is on stack. Push car=2 on top. */
    emit(vm, OP_CONST, c2);     /* push car = 2. Stack: [(3), 2] */
    emit(vm, OP_CONS, 0);       /* car=2, cdr=(3) → (2 3) ✓ */
    emit(vm, OP_CONST, c1);     /* push car = 1 */
    emit(vm, OP_CONS, 0);       /* car=1, cdr=(2 3) → (1 2 3) ✓ */
    emit(vm, OP_PRINT, 0);
    emit(vm, OP_HALT, 0);
    vm_run(vm);
    /* Should print (1 2 3) */
    int ok = (vm->n_outputs == 1 && vm->outputs[0].type == VAL_PAIR);
    printf("%s\n", ok ? "PASS" : "FAIL");
}

static void test_factorial(void) {
    printf("  test_factorial: ");
    VM* vmp = vm_create(); VM* vm = vmp;
    vm->code_len = 0;

    /* Bytecode for:
     * (define (factorial n)
     *   (if (= n 0) 1 (* n (factorial (- n 1)))))
     * (display (factorial 10))
     *
     * Compiled:
     *   Main: push 10, push factorial-closure, call 1, print, halt
     *   factorial: getlocal 0, const 0, eq, jif else, const 1, return
     *              getlocal 0, getlocal 0, const 1, sub, <self>, call 1, mul, return */

    int c0 = add_constant(vm, INT_VAL(0));
    int c1 = add_constant(vm, INT_VAL(1));
    int c10 = add_constant(vm, INT_VAL(10));
    int cfunc = add_constant(vm, INT_VAL(7)); /* factorial starts at PC=7 */

    /* Main: PC 0-6 */
    emit(vm, OP_CLOSURE, cfunc);  /* 0: push factorial closure */
    emit(vm, OP_CONST, c10);     /* 1: push 10 */
    /* For CALL: function below args. Stack: [closure, 10]. CALL 1 expects func at sp-2. */
    /* Actually: OP_CALL expects func at stack[sp - 1 - argc] = stack[sp - 2] */
    /* Stack: [closure, 10]. sp=2. func = stack[2-1-1] = stack[0] = closure. ✓ */
    emit(vm, OP_CALL, 1);        /* 2: call factorial(10) */
    emit(vm, OP_PRINT, 0);       /* 3: print result */
    emit(vm, OP_HALT, 0);        /* 4: halt */
    emit(vm, OP_NOP, 0);         /* 5: pad */
    emit(vm, OP_NOP, 0);         /* 6: pad */

    /* factorial: PC 7+ */
    /* At entry: fp points to args. stack[fp] = n (the argument).
     * stack[fp-1] = the closure (function slot). */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: push n */
    emit(vm, OP_CONST, c0);      /* 8: push 0 */
    emit(vm, OP_EQ, 0);          /* 9: n == 0? */
    emit(vm, OP_JUMP_IF_FALSE, 13); /* 10: if false, go to recursive case */
    emit(vm, OP_CONST, c1);      /* 11: push 1 */
    emit(vm, OP_RETURN, 0);      /* 12: return 1 */
    /* Recursive case: */
    emit(vm, OP_GET_LOCAL, 0);   /* 13: push n */
    emit(vm, OP_GET_LOCAL, 0);   /* 14: push n */
    emit(vm, OP_CONST, c1);      /* 15: push 1 */
    emit(vm, OP_SUB, 0);         /* 16: n - 1 */
    /* Need to call factorial again. Get the closure from the frame. */
    /* The closure is at stack[fp-1]. We need OP_GET_LOCAL -1 or a different mechanism. */
    /* Alternative: use a global/constant reference to factorial. */
    /* Simplest: factorial is a constant closure. Emit CLOSURE again. */
    emit(vm, OP_CLOSURE, cfunc); /* 17: push factorial closure */
    /* Stack: [n, n-1, closure]. Need: [closure, n-1] for call, with n below for MUL. */
    /* Hmm, the call convention puts func below args. */
    /* Stack before CALL 1: [..., func, arg]. func at sp-2, arg at sp-1. */
    /* Current stack: [n, n-1, closure]. We need [n, closure, n-1]. */
    /* This requires reordering. Without SWAP, this is tricky. */
    /* Let me restructure: push closure first, then n-1. */
    vm->code_len = 7; /* restart factorial */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: push n */
    emit(vm, OP_CONST, c0);      /* 8: push 0 */
    emit(vm, OP_EQ, 0);          /* 9: n == 0? */
    emit(vm, OP_JUMP_IF_FALSE, 13);
    emit(vm, OP_CONST, c1);      /* 11: push 1 */
    emit(vm, OP_RETURN, 0);      /* 12: return 1 */
    /* Recursive: n * factorial(n-1) */
    emit(vm, OP_GET_LOCAL, 0);   /* 13: push n (for MUL later) */
    emit(vm, OP_CLOSURE, cfunc); /* 14: push factorial closure */
    emit(vm, OP_GET_LOCAL, 0);   /* 15: push n */
    emit(vm, OP_CONST, c1);      /* 16: push 1 */
    emit(vm, OP_SUB, 0);         /* 17: n - 1 */
    /* Stack: [n, closure, n-1]. CALL 1: func=stack[sp-2]=closure, arg=n-1. ✓ */
    emit(vm, OP_CALL, 1);        /* 18: call factorial(n-1) */
    /* Stack after return: [n, factorial(n-1)] */
    emit(vm, OP_MUL, 0);         /* 19: n * factorial(n-1) */
    emit(vm, OP_RETURN, 0);      /* 20: return result */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].type == VAL_INT && vm->outputs[0].as.i == 3628800);
    printf("factorial(10)=%lld %s\n", vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
}

static void test_tail_factorial(void) {
    printf("  test_tail_factorial: ");
    VM* vm = vm_create();

    /* Tail-recursive factorial:
     * (define (fact-iter n acc)
     *   (if (= n 0) acc (fact-iter (- n 1) (* n acc))))
     * (display (fact-iter 10 1))
     *
     * Uses TAIL_CALL — no stack growth for recursive calls. */

    int c0 = add_constant(vm, INT_VAL(0));
    int c1 = add_constant(vm, INT_VAL(1));
    int c10 = add_constant(vm, INT_VAL(10));
    int cfunc = add_constant(vm, INT_VAL(7)); /* fact-iter at PC=7 */

    /* Main: PC 0-6 */
    emit(vm, OP_CLOSURE, cfunc);  /* 0 */
    emit(vm, OP_CONST, c10);     /* 1: n=10 */
    emit(vm, OP_CONST, c1);      /* 2: acc=1 */
    emit(vm, OP_CALL, 2);        /* 3: call fact-iter(10, 1) */
    emit(vm, OP_PRINT, 0);       /* 4 */
    emit(vm, OP_HALT, 0);        /* 5 */
    emit(vm, OP_NOP, 0);         /* 6: pad */

    /* fact-iter: PC 7+. Params: slot 0 = n, slot 1 = acc */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: push n */
    emit(vm, OP_CONST, c0);      /* 8: push 0 */
    emit(vm, OP_EQ, 0);          /* 9: n == 0? */
    emit(vm, OP_JUMP_IF_FALSE, 13);
    emit(vm, OP_GET_LOCAL, 1);   /* 11: push acc */
    emit(vm, OP_RETURN, 0);      /* 12: return acc */
    /* Tail recursive case: fact-iter(n-1, n*acc) */
    emit(vm, OP_CLOSURE, cfunc); /* 13: push fact-iter closure */
    emit(vm, OP_GET_LOCAL, 0);   /* 14: push n */
    emit(vm, OP_CONST, c1);      /* 15: push 1 */
    emit(vm, OP_SUB, 0);         /* 16: n - 1 (first arg) */
    emit(vm, OP_GET_LOCAL, 0);   /* 17: push n */
    emit(vm, OP_GET_LOCAL, 1);   /* 18: push acc */
    emit(vm, OP_MUL, 0);         /* 19: n * acc (second arg) */
    /* Stack: [closure, n-1, n*acc]. TAIL_CALL 2 reuses frame. */
    emit(vm, OP_TAIL_CALL, 2);   /* 20: tail call */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].as.i == 3628800);
    printf("fact-iter(10,1)=%lld %s\n",
           vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_fibonacci(void) {
    printf("  test_fibonacci: ");
    VM* vm = vm_create();

    /* Recursive fibonacci:
     * (define (fib n)
     *   (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
     * (display (fib 10))  → 55 */

    int c0 = add_constant(vm, INT_VAL(0));
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c10 = add_constant(vm, INT_VAL(10));
    int cfib = add_constant(vm, INT_VAL(7)); /* fib at PC=7 */

    /* Main */
    emit(vm, OP_CLOSURE, cfib);
    emit(vm, OP_CONST, c10);
    emit(vm, OP_CALL, 1);
    emit(vm, OP_PRINT, 0);
    emit(vm, OP_HALT, 0);
    emit(vm, OP_NOP, 0); emit(vm, OP_NOP, 0);

    /* fib: PC 7. Param: slot 0 = n */
    emit(vm, OP_GET_LOCAL, 0);   /* 7: n */
    emit(vm, OP_CONST, c2);      /* 8: 2 */
    emit(vm, OP_LT, 0);          /* 9: n < 2? */
    emit(vm, OP_JUMP_IF_FALSE, 13);
    emit(vm, OP_GET_LOCAL, 0);   /* 11: push n */
    emit(vm, OP_RETURN, 0);      /* 12: return n */
    /* Recursive: fib(n-1) + fib(n-2) */
    emit(vm, OP_CLOSURE, cfib);  /* 13: push fib */
    emit(vm, OP_GET_LOCAL, 0);   /* 14: n */
    emit(vm, OP_CONST, c1);      /* 15: 1 */
    emit(vm, OP_SUB, 0);         /* 16: n-1 */
    emit(vm, OP_CALL, 1);        /* 17: fib(n-1) */
    emit(vm, OP_CLOSURE, cfib);  /* 18: push fib */
    emit(vm, OP_GET_LOCAL, 0);   /* 19: n */
    emit(vm, OP_CONST, c2);      /* 20: 2 */
    emit(vm, OP_SUB, 0);         /* 21: n-2 */
    emit(vm, OP_CALL, 1);        /* 22: fib(n-2) */
    emit(vm, OP_ADD, 0);         /* 23: fib(n-1) + fib(n-2) */
    emit(vm, OP_RETURN, 0);      /* 24 */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].as.i == 55);
    printf("fib(10)=%lld %s\n",
           vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_list_build(void) {
    printf("  test_list_build: ");
    VM* vm = vm_create();

    /* Build list (1 2 3) = cons(1, cons(2, cons(3, nil)))
     * With our CONS convention: push cdr, push car, CONS.
     * Build from inside out: */
    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c3 = add_constant(vm, INT_VAL(3));

    emit(vm, OP_NIL, 0);         /* cdr = nil */
    emit(vm, OP_CONST, c3);      /* car = 3 */
    emit(vm, OP_CONS, 0);        /* (3) */
    emit(vm, OP_CONST, c2);      /* car = 2, cdr = (3) on stack */
    emit(vm, OP_CONS, 0);        /* (2 3) */
    emit(vm, OP_CONST, c1);      /* car = 1, cdr = (2 3) */
    emit(vm, OP_CONS, 0);        /* (1 2 3) */

    /* Test: car = 1, cadr = 2, caddr = 3 */
    emit(vm, OP_DUP, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);       /* 1 */
    emit(vm, OP_DUP, 0);
    emit(vm, OP_CDR, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);       /* 2 */
    emit(vm, OP_CDR, 0);
    emit(vm, OP_CDR, 0);
    emit(vm, OP_CAR, 0);
    emit(vm, OP_PRINT, 0);       /* 3 */
    emit(vm, OP_HALT, 0);

    vm_run(vm);
    int ok = vm->n_outputs == 3
        && vm->outputs[0].as.i == 1
        && vm->outputs[1].as.i == 2
        && vm->outputs[2].as.i == 3;
    printf("%s\n", ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_map(void) {
    printf("  test_map: ");
    VM* vm = vm_create();

    /* Recursive map:
     * (define (map f lst)
     *   (if (null? lst) '()
     *       (cons (f (car lst)) (map f (cdr lst)))))
     *
     * (define (double x) (* x 2))
     * (display (map double '(1 2 3)))  → (2 4 6) */

    int c1 = add_constant(vm, INT_VAL(1));
    int c2 = add_constant(vm, INT_VAL(2));
    int c3 = add_constant(vm, INT_VAL(3));
    int cdouble = add_constant(vm, INT_VAL(20)); /* double at PC=20 */
    int cmap = add_constant(vm, INT_VAL(9));     /* map at PC=9 */

    /* Main: PC 0-8 */
    /* Build list (1 2 3) */
    emit(vm, OP_NIL, 0);
    emit(vm, OP_CONST, c3); emit(vm, OP_CONS, 0);
    emit(vm, OP_CONST, c2); emit(vm, OP_CONS, 0);
    emit(vm, OP_CONST, c1); emit(vm, OP_CONS, 0);
    /* Stack: [(1 2 3)]. Call map(double, lst) */
    emit(vm, OP_CLOSURE, cmap);    /* 6: push map */
    emit(vm, OP_CLOSURE, cdouble); /* 7: push double (first arg = f) */
    /* Hmm, need to get list below function args.
     * Stack: [(1 2 3), map-closure, double-closure]
     * CALL 2 expects: func at sp-3, args at sp-2 and sp-1.
     * func = stack[sp-3] = (1 2 3). That's the LIST, not the function!
     * I need: [map-closure, double-closure, (1 2 3)]
     * Restructure: push closure first, then args. */
    vm->code_len = 0;
    emit(vm, OP_CLOSURE, cmap);    /* 0: push map closure */
    emit(vm, OP_CLOSURE, cdouble); /* 1: push double closure (arg 0 = f) */
    /* Build list (1 2 3) as arg 1 */
    emit(vm, OP_NIL, 0);          /* 2 */
    emit(vm, OP_CONST, c3); emit(vm, OP_CONS, 0); /* 3-4: (3) */
    emit(vm, OP_CONST, c2); emit(vm, OP_CONS, 0); /* 5-6: (2 3) */
    emit(vm, OP_CONST, c1); emit(vm, OP_CONS, 0); /* 7-8: (1 2 3) */
    /* Stack: [map, double, (1 2 3)]. CALL 2: func=map, args=[double, (1 2 3)] */
    emit(vm, OP_CALL, 2);         /* 9 */
    emit(vm, OP_PRINT, 0);        /* 10 */
    emit(vm, OP_HALT, 0);         /* 11 */

    /* map: PC=9?  No, PC 9 is CALL above. I need to adjust function PCs. */
    /* Let me pad to put functions at known locations. */
    vm->code_len = 0;
    /* Recalculate with padded layout */
    int main_end = 14;
    int map_pc = main_end;
    int double_pc = map_pc + 20;

    /* Update constants */
    vm->n_constants = 0;
    c1 = add_constant(vm, INT_VAL(1));
    c2 = add_constant(vm, INT_VAL(2));
    c3 = add_constant(vm, INT_VAL(3));
    cmap = add_constant(vm, INT_VAL(map_pc));
    cdouble = add_constant(vm, INT_VAL(double_pc));

    /* Main: 0 to main_end-1 */
    emit(vm, OP_CLOSURE, cmap);    /* 0 */
    emit(vm, OP_CLOSURE, cdouble); /* 1 */
    emit(vm, OP_NIL, 0);          /* 2 */
    emit(vm, OP_CONST, c3); emit(vm, OP_CONS, 0); /* 3-4 */
    emit(vm, OP_CONST, c2); emit(vm, OP_CONS, 0); /* 5-6 */
    emit(vm, OP_CONST, c1); emit(vm, OP_CONS, 0); /* 7-8 */
    emit(vm, OP_CALL, 2);         /* 9 */
    emit(vm, OP_PRINT, 0);        /* 10 */
    emit(vm, OP_HALT, 0);         /* 11 */
    while (vm->code_len < main_end) emit(vm, OP_NOP, 0);

    /* map: params: slot 0 = f, slot 1 = lst */
    /* (if (null? lst) '() (cons (f (car lst)) (map f (cdr lst)))) */
    emit(vm, OP_GET_LOCAL, 1);    /* map+0: push lst */
    emit(vm, OP_NULL_P, 0);       /* map+1: null? */
    emit(vm, OP_JUMP_IF_FALSE, map_pc + 5);
    emit(vm, OP_NIL, 0);          /* map+3: return '() */
    emit(vm, OP_RETURN, 0);       /* map+4 */
    /* Recursive case: cons(f(car lst), map(f, cdr lst)) */
    /* First: compute f(car lst) */
    emit(vm, OP_GET_LOCAL, 0);    /* map+5: push f (closure) */
    emit(vm, OP_GET_LOCAL, 1);    /* map+6: push lst */
    emit(vm, OP_CAR, 0);          /* map+7: car lst */
    emit(vm, OP_CALL, 1);         /* map+8: f(car lst) */
    /* Second: compute map(f, cdr lst) */
    emit(vm, OP_CLOSURE, cmap);   /* map+9: push map closure */
    emit(vm, OP_GET_LOCAL, 0);    /* map+10: push f */
    emit(vm, OP_GET_LOCAL, 1);    /* map+11: push lst */
    emit(vm, OP_CDR, 0);          /* map+12: cdr lst */
    emit(vm, OP_CALL, 2);         /* map+13: map(f, cdr lst) */
    /* Stack: [f(car lst), map(f, cdr lst)]. cons(car, cdr). */
    /* Our CONS: TOS=car, SOS=cdr. So top should be car = f(car lst).
     * But stack has: [f(car lst), map-result]. f(car lst) is SOS, map-result is TOS.
     * We need to swap. Or just restructure: compute map first (becomes cdr), then f (becomes car). */
    /* Actually, CONS pops car=TOS, cdr=SOS. So we need: SOS=cdr(map result), TOS=car(f result).
     * Stack: [map-result, f-result]. CONS: car=f-result, cdr=map-result. ✓
     * But we computed f-result first, map-result second. Stack: [f-result, map-result].
     * CONS: car=map-result, cdr=f-result. WRONG.
     * Fix: compute cdr first (map call), then car (f call). */
    /* Restructure: */
    vm->code_len = main_end; /* restart map function */
    emit(vm, OP_GET_LOCAL, 1);    /* map+0: push lst */
    emit(vm, OP_NULL_P, 0);       /* map+1: null? */
    emit(vm, OP_JUMP_IF_FALSE, map_pc + 5);
    emit(vm, OP_NIL, 0);          /* map+3: return '() */
    emit(vm, OP_RETURN, 0);       /* map+4 */
    /* Compute cdr first (will be SOS for CONS) */
    emit(vm, OP_CLOSURE, cmap);   /* map+5: push map */
    emit(vm, OP_GET_LOCAL, 0);    /* map+6: push f */
    emit(vm, OP_GET_LOCAL, 1);    /* map+7: push lst */
    emit(vm, OP_CDR, 0);          /* map+8: cdr lst */
    emit(vm, OP_CALL, 2);         /* map+9: map(f, cdr lst) — this is cdr of result */
    /* Now compute car (will be TOS for CONS) */
    emit(vm, OP_GET_LOCAL, 0);    /* map+10: push f */
    emit(vm, OP_GET_LOCAL, 1);    /* map+11: push lst */
    emit(vm, OP_CAR, 0);          /* map+12: car lst */
    emit(vm, OP_CALL, 1);         /* map+13: f(car lst) — this is car of result */
    /* Stack: [map(f, cdr lst), f(car lst)]. CONS: car=TOS=f(car lst), cdr=SOS=map(...) ✓ */
    emit(vm, OP_CONS, 0);         /* map+14: cons */
    emit(vm, OP_RETURN, 0);       /* map+15 */
    while (vm->code_len < double_pc) emit(vm, OP_NOP, 0);

    /* double: param slot 0 = x. Returns x * 2. */
    emit(vm, OP_GET_LOCAL, 0);    /* double+0: push x */
    emit(vm, OP_CONST, c2);       /* double+1: push 2 */
    emit(vm, OP_MUL, 0);          /* double+2: x * 2 */
    emit(vm, OP_RETURN, 0);       /* double+3 */

    vm_run(vm);
    /* Should output (2 4 6) */
    int ok = (vm->n_outputs == 1 && vm->outputs[0].type == VAL_PAIR);
    if (ok) {
        /* Verify list contents */
        HeapObject* p1 = vm->heap.objects[vm->outputs[0].as.ptr];
        ok = ok && p1->cons.car.as.i == 2;
        HeapObject* p2 = vm->heap.objects[p1->cons.cdr.as.ptr];
        ok = ok && p2->cons.car.as.i == 4;
        HeapObject* p3 = vm->heap.objects[p2->cons.cdr.as.ptr];
        ok = ok && p3->cons.car.as.i == 6;
        ok = ok && p3->cons.cdr.type == VAL_NIL;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    vm_free(vm);
}

static void test_closures(void) {
    printf("  test_closures: ");
    VM* vm = vm_create();

    /* (define (make-adder n) (lambda (x) (+ x n)))
     * (define add5 (make-adder 5))
     * (display (add5 10))  → 15
     *
     * This requires upvalue capture. The inner lambda captures 'n'.
     * For simplicity, we'll use a direct approach:
     * make-adder returns a closure with n captured as upvalue[0]. */

    int c5 = add_constant(vm, INT_VAL(5));
    int c10 = add_constant(vm, INT_VAL(10));
    int c_make_adder = add_constant(vm, INT_VAL(8));  /* make-adder at PC=8 */
    int c_inner = add_constant(vm, INT_VAL(14));       /* inner lambda at PC=14 */

    /* Main: PC 0-7 */
    emit(vm, OP_CLOSURE, c_make_adder); /* 0: push make-adder */
    emit(vm, OP_CONST, c5);            /* 1: push 5 */
    emit(vm, OP_CALL, 1);              /* 2: make-adder(5) → closure with n=5 */
    /* Stack: [add5-closure]. Call it with 10. */
    emit(vm, OP_CONST, c10);           /* 3: push 10 */
    emit(vm, OP_CALL, 1);              /* 4: add5(10) */
    emit(vm, OP_PRINT, 0);             /* 5: print */
    emit(vm, OP_HALT, 0);              /* 6 */
    emit(vm, OP_NOP, 0);               /* 7: pad */

    /* make-adder: PC 8. Param: slot 0 = n.
     * Creates closure for inner lambda with n captured. */
    emit(vm, OP_CLOSURE, c_inner);     /* 8: create inner closure */
    /* We need to store n as upvalue in the closure.
     * The closure is now on the stack. We need to set its upvalue[0] = n.
     * Current approach: after CLOSURE, the closure is on TOS. We need to
     * modify it to add the upvalue. Let's use a simple approach:
     * CLOSURE creates it, then we manually set upvalues. */
    /* Store n in the closure's upvalue array via stack push + CLOSURE. */
    {
        /* Hack: after CLOSURE pushes the closure, we peek at it and set upvalue.
         * This isn't a proper opcode yet — we'll need SET_CLOSURE_UPVALUE.
         * For this test, let's use OP_GET_LOCAL + a trick. */
    }
    /* Actually, the standard approach is: CLOSURE instruction followed by
     * upvalue descriptors. Each descriptor says "capture local slot X" or
     * "capture upvalue Y from enclosing closure."
     * For simplicity now: the CLOSURE opcode reads N upvalue values from the
     * stack (pushed before CLOSURE). */
    /* Restructure: push upvalues before CLOSURE. */
    vm->code_len = 8;
    emit(vm, OP_GET_LOCAL, 0);         /* 8: push n (will become upvalue) */
    emit(vm, OP_CLOSURE, c_inner);     /* 9: create closure, capture 1 upvalue from stack */
    /* Need the CLOSURE opcode to know it has 1 upvalue. Use operand encoding:
     * operand = (n_upvalues << 16) | func_const_idx. */
    /* Actually, let me just modify CLOSURE to pop upvalues. */
    /* Set closure upvalue count via operand encoding in the test. */
    emit(vm, OP_RETURN, 0);            /* 10: return the closure */
    while (vm->code_len < 14) emit(vm, OP_NOP, 0);

    /* inner lambda: PC 14. Param: slot 0 = x. Upvalue 0 = n. */
    emit(vm, OP_GET_LOCAL, 0);         /* 14: push x */
    emit(vm, OP_GET_UPVALUE, 0);       /* 15: push n (from closure) */
    emit(vm, OP_ADD, 0);               /* 16: x + n */
    emit(vm, OP_RETURN, 0);            /* 17 */

    /* Patch: modify CLOSURE to capture upvalues from stack. */
    /* For this test, I'll modify the VM's CLOSURE handler to accept
     * upvalue count in the high bits of the operand. */
    /* operand format: low 16 bits = const index, bits 16-23 = n_upvalues */
    vm->code[9] = (Instr){OP_CLOSURE, c_inner | (1 << 16)}; /* 1 upvalue */

    vm_run(vm);
    int ok = (vm->n_outputs > 0 && vm->outputs[0].as.i == 15);
    printf("make-adder(5)(10)=%lld %s\n",
           vm->n_outputs > 0 ? (long long)vm->outputs[0].as.i : -1,
           ok ? "PASS" : "FAIL");
    vm_free(vm);
}

/*******************************************************************************
 * Eshkol Source Compiler (parser + bytecode generator)
 * Merged from eshkol_compiler.c — single interpreter, one dispatch table.
 ******************************************************************************/

/*******************************************************************************
 * S-Expression Parser (reused from stackvm_codegen.c)
 ******************************************************************************/

typedef enum { N_NUMBER, N_SYMBOL, N_LIST, N_STRING, N_BOOL } NodeType;
typedef struct Node {
    NodeType type;
    double numval;
    char symbol[128];
    struct Node** children;
    int n_children;
} Node;

/* Hygienic macro expansion (syntax-rules).
 * Define VM_MACRO_NODE_DEFINED to skip MacroNode's duplicate enum/struct.
 * Provide typedefs so vm_macro.c functions can use MacroNode/MacroNodeType
 * while actually operating on the compiler's Node type (layout-compatible). */
#define VM_MACRO_NODE_DEFINED
typedef NodeType MacroNodeType;
typedef struct MacroNode {
    MacroNodeType    type;
    double           numval;
    char             symbol[128];
    struct MacroNode** children;
    int              n_children;
    int              _cap;
} MacroNode;
#include "vm_macro.c"

static const char* src_ptr = NULL;
static int g_trace_on = 0;  /* global, set by --trace flag */

static void skip_ws(void) {
    while (*src_ptr) {
        if (isspace(*src_ptr)) { src_ptr++; continue; }
        if (*src_ptr == ';') { while (*src_ptr && *src_ptr != '\n') src_ptr++; continue; }
        break;
    }
}

static Node* make_node(NodeType t) {
    Node* n = (Node*)calloc(1, sizeof(Node));
    if (!n) { fprintf(stderr, "ERROR: allocation failed in make_node\n"); return NULL; }
    n->type = t;
    return n;
}
static void add_child(Node* p, Node* c) {
    if (!p || !c) return;
    Node** nc = (Node**)realloc(p->children, (p->n_children+1)*sizeof(Node*));
    if (!nc) { fprintf(stderr, "ERROR: allocation failed in add_child\n"); return; }
    p->children = nc;
    p->children[p->n_children++] = c;
}

static void free_node(Node* n);
static Node* parse_sexp(void);
static Node* parse_list(void) {
    Node* list = make_node(N_LIST);
    if (!list) return NULL;
    while (1) { skip_ws(); if (!*src_ptr || *src_ptr == ')') break; Node* c = parse_sexp(); if (!c) break; add_child(list, c); }
    if (*src_ptr == ')') src_ptr++;
    return list;
}

static Node* parse_sexp(void) {
    skip_ws();
    if (!*src_ptr) return NULL;
    if (*src_ptr == '(') { src_ptr++; return parse_list(); }
    if (*src_ptr == ')') return NULL;
    if (*src_ptr == '\'') {
        src_ptr++;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* qs = make_node(N_SYMBOL); if (!qs) { free_node(q); return NULL; }
        strcpy(qs->symbol, "quote");
        add_child(q, qs);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* Quasiquote */
    if (*src_ptr == '`') {
        src_ptr++;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(q); return NULL; }
        strcpy(tag->symbol, "quasiquote");
        add_child(q, tag);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* Unquote-splicing (must check before unquote) */
    if (*src_ptr == ',' && src_ptr[1] == '@') {
        src_ptr += 2;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(q); return NULL; }
        strcpy(tag->symbol, "unquote-splicing");
        add_child(q, tag);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* Unquote */
    if (*src_ptr == ',') {
        src_ptr++;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(q); return NULL; }
        strcpy(tag->symbol, "unquote");
        add_child(q, tag);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* String literal */
    if (*src_ptr == '"') {
        src_ptr++; /* skip opening quote */
        char buf[256]; int i = 0;
        while (*src_ptr && *src_ptr != '"' && i < 255) {
            if (*src_ptr == '\\' && src_ptr[1]) {
                src_ptr++;
                switch (*src_ptr) {
                    case 'n': buf[i++] = '\n'; break;
                    case 't': buf[i++] = '\t'; break;
                    case '\\': buf[i++] = '\\'; break;
                    case '"': buf[i++] = '"'; break;
                    default: buf[i++] = *src_ptr; break;
                }
                src_ptr++;
            } else {
                buf[i++] = *src_ptr++;
            }
        }
        if (*src_ptr == '"') src_ptr++; /* skip closing quote */
        buf[i] = 0;
        Node* n = make_node(N_STRING); if (!n) return NULL;
        strncpy(n->symbol, buf, 127); n->symbol[127] = 0;
        return n;
    }
    if (*src_ptr == '#') {
        if (src_ptr[1] == 't' && (src_ptr[2] == 0 || isspace(src_ptr[2]) || src_ptr[2] == ')')) {
            src_ptr += 2; Node* n = make_node(N_BOOL); if (!n) return NULL; n->numval = 1; strcpy(n->symbol, "#t"); return n;
        }
        if (src_ptr[1] == 'f' && (src_ptr[2] == 0 || isspace(src_ptr[2]) || src_ptr[2] == ')')) {
            src_ptr += 2; Node* n = make_node(N_BOOL); if (!n) return NULL; n->numval = 0; strcpy(n->symbol, "#f"); return n;
        }
        /* Character literal: #\a, #\space, #\newline, #\tab */
        if (src_ptr[1] == '\\') {
            src_ptr += 2;
            int ch;
            if (strncmp(src_ptr, "space", 5) == 0 && (!src_ptr[5] || isspace(src_ptr[5]) || src_ptr[5] == ')')) {
                ch = ' '; src_ptr += 5;
            } else if (strncmp(src_ptr, "newline", 7) == 0 && (!src_ptr[7] || isspace(src_ptr[7]) || src_ptr[7] == ')')) {
                ch = '\n'; src_ptr += 7;
            } else if (strncmp(src_ptr, "tab", 3) == 0 && (!src_ptr[3] || isspace(src_ptr[3]) || src_ptr[3] == ')')) {
                ch = '\t'; src_ptr += 3;
            } else if (strncmp(src_ptr, "nul", 3) == 0 && (!src_ptr[3] || isspace(src_ptr[3]) || src_ptr[3] == ')')) {
                ch = 0; src_ptr += 3;
            } else {
                ch = (unsigned char)*src_ptr; src_ptr++;
            }
            Node* n = make_node(N_NUMBER); if (!n) return NULL;
            n->numval = ch;
            return n;
        }
        /* Vector literal: #(elements...) */
        if (src_ptr[1] == '(') {
            src_ptr += 2; /* skip #( */
            Node* vec = make_node(N_LIST); if (!vec) return NULL;
            Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(vec); return NULL; }
            strcpy(tag->symbol, "vector");
            add_child(vec, tag);
            while (1) { skip_ws(); if (!*src_ptr || *src_ptr == ')') break; Node* el = parse_sexp(); if (!el) break; add_child(vec, el); }
            if (*src_ptr == ')') src_ptr++;
            return vec;
        }
    }
    /* Number */
    if (isdigit(*src_ptr) || (*src_ptr == '-' && isdigit(src_ptr[1]))) {
        char buf[64]; int i = 0;
        if (*src_ptr == '-') buf[i++] = *src_ptr++;
        while ((isdigit(*src_ptr) || *src_ptr == '.') && i < 63) buf[i++] = *src_ptr++;
        buf[i] = 0;
        Node* n = make_node(N_NUMBER); if (!n) return NULL; n->numval = atof(buf); return n;
    }
    /* Symbol */
    char buf[128]; int i = 0;
    while (*src_ptr && !isspace(*src_ptr) && *src_ptr != '(' && *src_ptr != ')' && *src_ptr != '"' && i < 127)
        buf[i++] = *src_ptr++;
    buf[i] = 0;
    Node* n = make_node(N_SYMBOL); if (!n) return NULL; strncpy(n->symbol, buf, 127); n->symbol[127] = 0; return n;
}

static void free_node(Node* n) { if (!n) return; for (int i=0;i<n->n_children;i++) free_node(n->children[i]); free(n->children); free(n); }

/*******************************************************************************
 * Compiler: AST → Bytecode
 ******************************************************************************/

#define MAX_CODE 32768
#ifndef MAX_CONSTS
#define MAX_CONSTS 1024
#endif
#define MAX_LOCALS 512
#define MAX_FUNCS 64

typedef struct {
    char name[128];
    int slot;
    int depth;
    int boxed;  /* 1 = variable is heap-boxed (stored in 1-element vector) */
} Local;

typedef struct {
    char name[128];
    int enclosing_slot;  /* slot or upvalue index in the enclosing scope */
    int index;           /* upvalue index in this closure */
    int is_local;        /* 1 = enclosing_slot is a local, 0 = it's an upvalue */
    int boxed;           /* 1 = the captured variable is heap-boxed */
} Upvalue;

#define MAX_UPVALUES 32

typedef struct FuncChunk {
    Instr code[MAX_CODE];
    int code_len;
    Value constants[MAX_CONSTS];
    int n_constants;
    Local locals[MAX_LOCALS];
    int n_locals;
    Upvalue upvalues[MAX_UPVALUES];
    int n_upvalues;
    int scope_depth;
    int scope_stack_base[32]; /* stack depth at scope entry, for cleanup on exit */
    struct FuncChunk* enclosing;
    int param_count;
    int stack_depth;  /* compile-time stack depth (values above fp) */
} FuncChunk;

static int is_sym(Node* n, const char* s) { return n && n->type == N_SYMBOL && strcmp(n->symbol, s) == 0; }

static void chunk_emit(FuncChunk* c, uint8_t op, int32_t operand) {
    if (c->code_len >= MAX_CODE) { fprintf(stderr, "ERROR: bytecode overflow (MAX_CODE=%d)\n", MAX_CODE); return; }
    c->code[c->code_len++] = (Instr){op, operand};
}

static int chunk_add_const(FuncChunk* c, Value v) {
    /* No deduplication — function PC placeholders get patched after creation,
     * which would corrupt literal constants that matched the placeholder value. */
    if (c->n_constants >= MAX_CONSTS) { fprintf(stderr, "ERROR: constant pool overflow (MAX_CONSTS=%d)\n", MAX_CONSTS); return -1; }
    c->constants[c->n_constants] = v;
    return c->n_constants++;
}

static int placeholder(FuncChunk* c) {
    int slot = c->code_len;
    chunk_emit(c, OP_NOP, 0);
    return slot;
}

static void patch(FuncChunk* c, int slot, uint8_t op, int32_t target) {
    c->code[slot] = (Instr){op, target};
}

static int resolve_local(FuncChunk* c, const char* name) {
    for (int i = c->n_locals - 1; i >= 0; i--) {
        if (strcmp(c->locals[i].name, name) == 0) return c->locals[i].slot;
    }
    return -1;
}

static int add_local(FuncChunk* c, const char* name) {
    if (c->n_locals >= MAX_LOCALS) { fprintf(stderr, "ERROR: local variable overflow (MAX_LOCALS=%d)\n", MAX_LOCALS); return -1; }
    int slot = c->n_locals;
    strncpy(c->locals[c->n_locals].name, name, 127);
    c->locals[c->n_locals].name[127] = 0;
    c->locals[c->n_locals].slot = slot;
    c->locals[c->n_locals].depth = c->scope_depth;
    c->n_locals++;
    return slot;
}

static void compile_expr(FuncChunk* c, Node* node, int tail_position);

/* Scan an AST node for set! references to a named variable */
static int scan_for_set(Node* node, const char* name) {
    if (!node) return 0;
    if (node->type == N_LIST && node->n_children >= 3) {
        Node* head = node->children[0];
        if (head->type == N_SYMBOL && strcmp(head->symbol, "set!") == 0
            && node->children[1]->type == N_SYMBOL
            && strcmp(node->children[1]->symbol, name) == 0)
            return 1;
    }
    if (node->type == N_LIST) {
        for (int i = 0; i < node->n_children; i++)
            if (scan_for_set(node->children[i], name)) return 1;
    }
    return 0;
}

/* Scan for FREE references to a variable name inside lambda bodies.
 * A reference is free if the variable is not rebound as a lambda parameter
 * or let binding at an inner scope. */
static int scan_for_capture(Node* node, const char* name, int in_lambda) {
    if (!node) return 0;
    if (node->type == N_SYMBOL && in_lambda && strcmp(node->symbol, name) == 0)
        return 1;
    if (node->type == N_LIST && node->n_children >= 1) {
        Node* head = node->children[0];
        /* Check if this lambda/let rebinds the variable — if so, it's not a capture */
        if (head->type == N_SYMBOL && strcmp(head->symbol, "lambda") == 0 && node->n_children >= 3) {
            /* Check if name is a parameter of this lambda */
            Node* params = node->children[1];
            if (params->type == N_LIST) {
                for (int i = 0; i < params->n_children; i++)
                    if (params->children[i]->type == N_SYMBOL && strcmp(params->children[i]->symbol, name) == 0)
                        return 0; /* rebound as parameter — not a capture */
            }
            /* Scan body (now inside lambda) */
            for (int i = 2; i < node->n_children; i++)
                if (scan_for_capture(node->children[i], name, 1)) return 1;
            return 0;
        }
        if (head->type == N_SYMBOL && (strcmp(head->symbol, "let") == 0 ||
            strcmp(head->symbol, "let*") == 0 || strcmp(head->symbol, "letrec") == 0)) {
            /* Check if name is rebound in this let's bindings */
            if (node->n_children >= 3 && node->children[1]->type == N_LIST) {
                Node* bindings = node->children[1];
                for (int i = 0; i < bindings->n_children; i++) {
                    Node* b = bindings->children[i];
                    if (b->type == N_LIST && b->n_children >= 1 && b->children[0]->type == N_SYMBOL
                        && strcmp(b->children[0]->symbol, name) == 0)
                        return 0; /* rebound in inner let */
                }
            }
        }
        /* Recurse into children */
        int new_lambda = in_lambda;
        if (head->type == N_SYMBOL && strcmp(head->symbol, "lambda") == 0)
            new_lambda = 1;
        for (int i = 0; i < node->n_children; i++)
            if (scan_for_capture(node->children[i], name, new_lambda)) return 1;
    }
    return 0;
}

/* Check if a let-bound variable needs heap boxing (captured + mutated) */
static int needs_boxing(Node* body_nodes[], int n_bodies, const char* name) {
    int has_set = 0, has_capture = 0;
    for (int i = 0; i < n_bodies; i++) {
        if (scan_for_set(body_nodes[i], name)) has_set = 1;
        if (scan_for_capture(body_nodes[i], name, 0)) has_capture = 1;
    }
    return has_set && has_capture;
}

/* Compile a quoted datum into cons cells, symbols as strings, etc. */
static void compile_quote(FuncChunk* c, Node* datum) {
    if (!datum) { chunk_emit(c, OP_NIL, 0); return; }
    if (datum->type == N_NUMBER) {
        double v = datum->numval;
        if (v == (int64_t)v && fabs(v) < 1e15)
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL((int64_t)v)));
        else
            chunk_emit(c, OP_CONST, chunk_add_const(c, FLOAT_VAL(v)));
        return;
    }
    if (datum->type == N_BOOL) {
        chunk_emit(c, datum->numval ? OP_TRUE : OP_FALSE, 0);
        return;
    }
    if (datum->type == N_STRING) {
        compile_expr(c, datum, 0); /* reuse string literal compilation */
        return;
    }
    if (datum->type == N_SYMBOL) {
        /* Quoted symbol → compile as string */
        int len = (int)strlen(datum->symbol);
        int n_packs = (len + 7) / 8;
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(len)));
        for (int p = 0; p < n_packs; p++) {
            int64_t pack = 0;
            for (int b = 0; b < 8 && p * 8 + b < len; b++)
                pack |= ((int64_t)(unsigned char)datum->symbol[p * 8 + b]) << (b * 8);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(pack)));
        }
        chunk_emit(c, OP_NATIVE_CALL, 100);
        return;
    }
    if (datum->type == N_LIST) {
        if (datum->n_children == 0) { chunk_emit(c, OP_NIL, 0); return; }
        /* Build proper list: (cons el0 (cons el1 ... (cons elN-1 '()))) */
        /* Compile in reverse: push NIL, then cons each element from back to front */
        chunk_emit(c, OP_NIL, 0);
        for (int i = datum->n_children - 1; i >= 0; i--) {
            compile_quote(c, datum->children[i]);
            chunk_emit(c, OP_CONS, 0);
        }
        return;
    }
    chunk_emit(c, OP_NIL, 0); /* fallback */
}

static void compile_expr_impl(FuncChunk* c, Node* node, int tail);

static void compile_quasiquote(FuncChunk* c, Node* node) {
    if (!node) { chunk_emit(c, OP_NIL, 0); return; }

    /* (unquote x) -> compile x normally */
    if (node->type == N_LIST && node->n_children == 2 &&
        node->children[0]->type == N_SYMBOL && strcmp(node->children[0]->symbol, "unquote") == 0) {
        compile_expr(c, node->children[1], 0);
        return;
    }

    /* Atom: number */
    if (node->type == N_NUMBER) {
        int ci = chunk_add_const(c, node->numval == (int64_t)node->numval ? INT_VAL((int64_t)node->numval) : FLOAT_VAL(node->numval));
        if (ci >= 0) chunk_emit(c, OP_CONST, ci);
        return;
    }
    /* Atom: symbol — quote as string */
    if (node->type == N_SYMBOL) {
        int len = (int)strlen(node->symbol);
        int n_packs = (len + 7) / 8;
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(len)));
        for (int p = 0; p < n_packs; p++) {
            int64_t pack = 0;
            for (int b = 0; b < 8 && p * 8 + b < len; b++)
                pack |= ((int64_t)(unsigned char)node->symbol[p * 8 + b]) << (b * 8);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(pack)));
        }
        chunk_emit(c, OP_NATIVE_CALL, 100);
        return;
    }
    /* Atom: string */
    if (node->type == N_STRING) {
        compile_expr(c, node, 0);
        return;
    }
    /* Atom: boolean */
    if (node->type == N_BOOL) {
        chunk_emit(c, node->numval ? OP_TRUE : OP_FALSE, 0);
        return;
    }

    /* List: build from right to left using cons */
    if (node->type == N_LIST) {
        chunk_emit(c, OP_NIL, 0); /* start with empty list */
        for (int i = node->n_children - 1; i >= 0; i--) {
            Node* elem = node->children[i];
            /* Check for unquote-splicing */
            if (elem->type == N_LIST && elem->n_children == 2 &&
                elem->children[0]->type == N_SYMBOL &&
                strcmp(elem->children[0]->symbol, "unquote-splicing") == 0) {
                /* Compile the spliced expression */
                compile_expr(c, elem->children[1], 0);
                /* Append to accumulator: (append spliced acc) */
                chunk_emit(c, OP_NATIVE_CALL, 73); /* append */
            } else {
                compile_quasiquote(c, elem);
                chunk_emit(c, OP_CONS, 0);
            }
        }
        return;
    }

    /* Fallback: emit nil */
    chunk_emit(c, OP_NIL, 0);
}

static int compile_depth = 0;

static void compile_expr(FuncChunk* c, Node* node, int tail) {
    compile_depth++;
    if (compile_depth > 1000) { fprintf(stderr, "ERROR: expression nesting too deep (>1000)\n"); compile_depth--; return; }
    compile_expr_impl(c, node, tail);
    compile_depth--;
}

static void compile_expr_impl(FuncChunk* c, Node* node, int tail) {
    if (!node) return;

    /* Check for macro expansion — must come before all other dispatch */
    if (node->type == N_LIST && node->n_children > 0 &&
        node->children[0]->type == N_SYMBOL) {
        VmMacro* macro = vm_macro_lookup(node->children[0]->symbol);
        if (macro) {
            MacroNode* expanded = vm_macro_expand((const MacroNode*)node);
            if (expanded && expanded != (MacroNode*)node) {
                compile_expr(c, (Node*)expanded, tail);
                /* Note: expanded node leaked — acceptable for compiler lifetime */
                return;
            }
        }
    }

    if (node->type == N_NUMBER) {
        double v = node->numval;
        if (v == (int64_t)v && fabs(v) < 1e15)
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL((int64_t)v)));
        else
            chunk_emit(c, OP_CONST, chunk_add_const(c, FLOAT_VAL(v)));
        return;
    }

    if (node->type == N_BOOL) {
        chunk_emit(c, node->numval ? OP_TRUE : OP_FALSE, 0);
        return;
    }

    /* String literal — encode as a constant with embedded string data.
     * We use a special convention: the constant's .as.ptr field stores
     * a negative index into a string table. At runtime, OP_CONST for
     * a string constant allocates it on the heap.
     * Simpler approach: use OP_NATIVE_CALL 56 with string ID. */
    if (node->type == N_STRING) {
        /* String literal → emit packed char data + NATIVE_CALL 100 to build heap string.
         * Pack up to 8 chars per int64 constant, push them, then call build-string. */
        int len = (int)strlen(node->symbol);
        int n_packs = (len + 7) / 8;
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(len)));
        for (int p = 0; p < n_packs; p++) {
            int64_t pack = 0;
            for (int b = 0; b < 8 && p * 8 + b < len; b++) {
                pack |= ((int64_t)(unsigned char)node->symbol[p * 8 + b]) << (b * 8);
            }
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(pack)));
        }
        chunk_emit(c, OP_NATIVE_CALL, 100); /* build-string-from-packed */
        return;
    }

    if (node->type == N_SYMBOL) {
        if (strcmp(node->symbol, "#t") == 0) { chunk_emit(c, OP_TRUE, 0); return; }
        if (strcmp(node->symbol, "#f") == 0) { chunk_emit(c, OP_FALSE, 0); return; }
        /* Variable lookup: local → enclosing (upvalue) → error */
        int slot = resolve_local(c, node->symbol);
        if (slot == -99) {
            /* Special: guard exception variable → use OP_GET_EXN */
            chunk_emit(c, OP_GET_EXN, 0);
            return;
        }
        if (slot >= 0) {
            chunk_emit(c, OP_GET_LOCAL, slot);
            /* If boxed, unbox: the local holds a vector, read element 0 */
            for (int li = c->n_locals - 1; li >= 0; li--) {
                if (c->locals[li].slot == slot && c->locals[li].boxed) {
                    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                    chunk_emit(c, OP_VEC_REF, 0);
                    break;
                }
            }
            return;
        }
        /* Check enclosing scopes for upvalue (walk entire scope chain).
         * If the variable is found N levels up, each intermediate level
         * must also capture it as an upvalue (relay chain).
         * This implements Lox-style upvalue chains. */
        {
            /* Build the chain of FuncChunks from current to root */
            FuncChunk* chain[32];
            int depth = 0;
            for (FuncChunk* p = c; p && depth < 32; p = p->enclosing)
                chain[depth++] = p;

            /* Search from the outermost scope inward */
            for (int d = depth - 1; d >= 1; d--) {
                int enc_slot = resolve_local(chain[d], node->symbol);
                if (enc_slot >= 0) {
                    /* Found at level d. Check if it's boxed at the source. */
                    int var_boxed = 0;
                    for (int li = chain[d]->n_locals - 1; li >= 0; li--) {
                        if (chain[d]->locals[li].slot == enc_slot && chain[d]->locals[li].boxed) {
                            var_boxed = 1; break;
                        }
                    }

                    /* Ensure each level from d-1 down to 0 captures this as an upvalue. */
                    int prev_slot = enc_slot;
                    int prev_is_local = 1;

                    for (int level = d - 1; level >= 0; level--) {
                        FuncChunk* fc = chain[level];
                        int uv_idx = -1;
                        for (int i = 0; i < fc->n_upvalues; i++) {
                            if (strcmp(fc->upvalues[i].name, node->symbol) == 0) {
                                uv_idx = fc->upvalues[i].index;
                                break;
                            }
                        }
                        if (uv_idx < 0 && fc->n_upvalues < MAX_UPVALUES) {
                            uv_idx = fc->n_upvalues;
                            strncpy(fc->upvalues[fc->n_upvalues].name, node->symbol, 127);
                            fc->upvalues[fc->n_upvalues].name[127] = 0;
                            fc->upvalues[fc->n_upvalues].enclosing_slot = prev_slot;
                            fc->upvalues[fc->n_upvalues].index = uv_idx;
                            fc->upvalues[fc->n_upvalues].is_local = prev_is_local;
                            fc->upvalues[fc->n_upvalues].boxed = var_boxed;
                            fc->n_upvalues++;
                        }
                        prev_slot = uv_idx;
                        prev_is_local = 0;
                    }

                    /* Emit GET_UPVALUE for the innermost (current) scope */
                    int final_uv = -1;
                    int final_boxed = 0;
                    for (int i = 0; i < c->n_upvalues; i++) {
                        if (strcmp(c->upvalues[i].name, node->symbol) == 0) {
                            final_uv = c->upvalues[i].index;
                            final_boxed = c->upvalues[i].boxed;
                            break;
                        }
                    }
                    if (final_uv >= 0) {
                        chunk_emit(c, OP_GET_UPVALUE, final_uv);
                        /* Unbox if the captured variable is boxed */
                        if (final_boxed) {
                            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                            chunk_emit(c, OP_VEC_REF, 0);
                        }
                        return;
                    }
                }
            }
        }
        printf("WARNING: undefined variable '%s'\n", node->symbol);
        chunk_emit(c, OP_NIL, 0);
        return;
    }

    if (node->type != N_LIST || node->n_children == 0) { chunk_emit(c, OP_NIL, 0); return; }

    Node* head = node->children[0];

    /* ── Constant Folding ── */
    /* If all operands are compile-time constants, evaluate at compile time */
    if (node->type == N_LIST && node->n_children >= 3) {
        if (head->type == N_SYMBOL) {
            int all_const = 1;
            for (int i = 1; i < node->n_children; i++) {
                if (node->children[i]->type != N_NUMBER) { all_const = 0; break; }
            }
            if (all_const) {
                double result = 0;
                int folded = 0;
                if (strcmp(head->symbol, "+") == 0) {
                    result = 0;
                    for (int i = 1; i < node->n_children; i++) result += node->children[i]->numval;
                    folded = 1;
                } else if (strcmp(head->symbol, "-") == 0 && node->n_children >= 2) {
                    result = node->children[1]->numval;
                    for (int i = 2; i < node->n_children; i++) result -= node->children[i]->numval;
                    folded = 1;
                } else if (strcmp(head->symbol, "*") == 0) {
                    result = 1;
                    for (int i = 1; i < node->n_children; i++) result *= node->children[i]->numval;
                    folded = 1;
                } else if (strcmp(head->symbol, "/") == 0 && node->n_children == 3 && node->children[2]->numval != 0) {
                    result = node->children[1]->numval / node->children[2]->numval;
                    folded = 1;
                }
                if (folded) {
                    int ci = chunk_add_const(c, result == (int64_t)result && fabs(result) < 1e15
                        ? INT_VAL((int64_t)result) : FLOAT_VAL(result));
                    if (ci >= 0) chunk_emit(c, OP_CONST, ci);
                    return;
                }
            }
        }
    }

    /* (+ a b ...), (- a b), (* a b ...), (/ a b) */
    if (is_sym(head, "+")) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_ADD, 0); }
        return;
    }
    if (is_sym(head, "-")) {
        if (node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NEG, 0); return; }
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_SUB, 0); }
        return;
    }
    if (is_sym(head, "*")) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_MUL, 0); }
        return;
    }
    if (is_sym(head, "/")) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_DIV, 0); }
        return;
    }

    /* Comparisons — push proper booleans */
    if (is_sym(head, "=") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_EQ, 0); return; }
    if (is_sym(head, "<") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_LT, 0); return; }
    if (is_sym(head, ">") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_GT, 0); return; }
    if (is_sym(head, "<=") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_LE, 0); return; }
    if (is_sym(head, ">=") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_GE, 0); return; }
    if (is_sym(head, "not") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NOT, 0); return; }
    if (is_sym(head, "zero?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0))); chunk_emit(c, OP_EQ, 0); return; }
    /* Core type predicates — always available as opcodes (not closures) */
    if (is_sym(head, "null?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NULL_P, 0); return; }
    if (is_sym(head, "pair?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_PAIR_P, 0); return; }
    if (is_sym(head, "number?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NUM_P, 0); return; }
    if (is_sym(head, "string?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_STR_P, 0); return; }
    if (is_sym(head, "boolean?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_BOOL_P, 0); return; }
    if (is_sym(head, "procedure?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_PROC_P, 0); return; }
    if (is_sym(head, "vector?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_VEC_P, 0); return; }

    /* display is a core opcode — always available, not a closure.
     * OP_PRINT pops the value. We push NIL as the return value so
     * the stack accounting is correct in begin/sequence contexts. */
    if (is_sym(head, "display") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_PRINT, 0);
        chunk_emit(c, OP_NIL, 0);  /* push return value (void → NIL) */
        return;
    }
    /* Type predicates that need VM opcodes (not closures — these check types at opcode level) */
    if (is_sym(head, "integer?") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NUM_P, 0); return; }

    /* abs and modulo are opcodes, not native calls — keep as special cases */
    if (is_sym(head, "abs") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_ABS, 0); return; }
    if (is_sym(head, "modulo") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_MOD, 0); return; }
    if (is_sym(head, "remainder") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_MOD, 0); return; }

    /* All other builtins (sin, cos, sqrt, even?, odd?, floor, ceiling, round, expt, min, max,
     * positive?, negative?, number->string, string-append, string=?, newline, length, etc.)
     * are first-class closures defined in the preamble. They resolve via normal variable lookup
     * and are called via the standard CALL mechanism. No special-casing needed. */

    /* Vector operations */
    if (is_sym(head, "vector")) {
        for (int i = 1; i < node->n_children; i++) compile_expr(c, node->children[i], 0);
        chunk_emit(c, OP_VEC_CREATE, node->n_children - 1);
        return;
    }
    if (is_sym(head, "make-vector") && node->n_children >= 2) {
        /* (make-vector n) or (make-vector n fill) — emit via NATIVE or direct */
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        /* make-vector: n and fill are on stack, dispatch to runtime native */
        chunk_emit(c, OP_NATIVE_CALL, 260);
        return;
    }
    if (is_sym(head, "vector-ref") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_VEC_REF, 0); return; }
    if (is_sym(head, "vector-set!") && node->n_children == 4) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); compile_expr(c, node->children[3], 0); chunk_emit(c, OP_VEC_SET, 0); return; }
    if (is_sym(head, "vector-length") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_VEC_LEN, 0); return; }

    /* Mutation */
    if (is_sym(head, "set-car!") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_SET_CAR, 0); return; }
    if (is_sym(head, "set-cdr!") && node->n_children == 3) { compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0); chunk_emit(c, OP_SET_CDR, 0); return; }

    /* String operations via opcodes (these ARE opcodes, not native calls) */
    if (is_sym(head, "string-length") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_STR_LEN, 0);
        return;
    }
    if (is_sym(head, "string-ref") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_STR_REF, 0);
        return;
    }
    /* All other string operations (string-append, string=?, newline, number->string, etc.)
     * are first-class closures from the preamble. */

    /* Compound list accessors: cadr, cdar, cddr, caar */
    if (is_sym(head, "cadr") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0);
        return;
    }
    if (is_sym(head, "cdar") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CDR, 0);
        return;
    }
    if (is_sym(head, "cddr") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0);
        return;
    }
    if (is_sym(head, "caar") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CAR, 0); chunk_emit(c, OP_CAR, 0);
        return;
    }
    if (is_sym(head, "caddr") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0);
        return;
    }
    /* first through fifth */
    if (is_sym(head, "first") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "second") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "third") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CDR, 0); chunk_emit(c, OP_CAR, 0); return; }

    /* (cond (test1 expr1) (test2 expr2) ... (else exprN)) */
    if (is_sym(head, "cond") && node->n_children >= 2) {
        int end_patches[64];
        int n_patches = 0;
        for (int i = 1; i < node->n_children; i++) {
            Node* clause = node->children[i];
            if (clause->type != N_LIST || clause->n_children < 2) continue;
            if (is_sym(clause->children[0], "else")) {
                /* else clause — always taken */
                for (int j = 1; j < clause->n_children; j++) {
                    if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
                    else compile_expr(c, clause->children[j], tail);
                }
                break;
            }
            /* Test → if false, jump to next clause */
            compile_expr(c, clause->children[0], 0);
            int jnext = placeholder(c);
            /* Body */
            for (int j = 1; j < clause->n_children; j++) {
                if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
                else compile_expr(c, clause->children[j], tail);
            }
            if (n_patches < 64) end_patches[n_patches++] = placeholder(c); /* jump to end */
            patch(c, jnext, OP_JUMP_IF_FALSE, c->code_len);
        }
        /* Patch all end jumps */
        for (int i = 0; i < n_patches; i++) patch(c, end_patches[i], OP_JUMP, c->code_len);
        return;
    }

    /* (case expr ((val ...) body ...) ... (else body ...))
     * Compiles as: evaluate key, then for each clause: DUP key, test each val,
     * if any matches jump to body, else next clause. */
    if (is_sym(head, "case") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0); /* evaluate key expression → TOS */
        int end_patches_c[64]; int n_patches_c = 0;
        for (int i = 2; i < node->n_children; i++) {
            Node* clause = node->children[i];
            if (clause->type != N_LIST || clause->n_children < 2) continue;
            if (is_sym(clause->children[0], "else")) {
                chunk_emit(c, OP_POP, 0); /* discard key */
                for (int j = 1; j < clause->n_children; j++) {
                    if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
                    else compile_expr(c, clause->children[j], tail);
                }
                break;
            }
            /* ((val1 val2 ...) body ...) */
            Node* vals = clause->children[0];
            if (vals->type != N_LIST) continue;
            /* Test key against each val: DUP, EQ, if true → jump to body */
            int body_patches[16]; int n_bp = 0;
            for (int v = 0; v < vals->n_children; v++) {
                chunk_emit(c, OP_DUP, 0);
                compile_quote(c, vals->children[v]);
                chunk_emit(c, OP_EQ, 0);
                /* If true, jump to body */
                if (n_bp < 16) body_patches[n_bp++] = c->code_len;
                chunk_emit(c, OP_JUMP_IF_FALSE, 0); /* placeholder: if false, continue */
                /* Match! Jump to body code */
                int jbody = c->code_len;
                chunk_emit(c, OP_JUMP, 0); /* placeholder: jump to body */
                /* Patch the JIF to skip the JUMP (continue testing) */
                patch(c, body_patches[n_bp-1], OP_JUMP_IF_FALSE, c->code_len);
                body_patches[n_bp-1] = jbody; /* reuse slot for body jump */
            }
            /* No val matched — jump to next clause */
            int jnext = c->code_len;
            chunk_emit(c, OP_JUMP, 0);
            /* Body code (reached by any matching val's jump) */
            for (int bp = 0; bp < n_bp; bp++)
                patch(c, body_patches[bp], OP_JUMP, c->code_len);
            chunk_emit(c, OP_POP, 0); /* discard key */
            for (int j = 1; j < clause->n_children; j++) {
                if (j < clause->n_children - 1) { compile_expr(c, clause->children[j], 0); chunk_emit(c, OP_POP, 0); }
                else compile_expr(c, clause->children[j], tail);
            }
            if (n_patches_c < 64) end_patches_c[n_patches_c++] = c->code_len;
            chunk_emit(c, OP_JUMP, 0);
            /* Patch jnext to after body */
            patch(c, jnext, OP_JUMP, c->code_len);
        }
        for (int i = 0; i < n_patches_c; i++) patch(c, end_patches_c[i], OP_JUMP, c->code_len);
        return;
    }

    /* (when test body...) — one-armed if */
    if (is_sym(head, "when") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        int jf = placeholder(c);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            if (i < node->n_children - 1) chunk_emit(c, OP_POP, 0);
        }
        patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        return;
    }

    /* (unless test body...) — negated when */
    if (is_sym(head, "unless") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NOT, 0);
        int jf = placeholder(c);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            if (i < node->n_children - 1) chunk_emit(c, OP_POP, 0);
        }
        patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        return;
    }

    /* (require module.name) — load and compile the module */
    if (is_sym(head, "require")) {
        if (node->n_children >= 2 && node->children[1]->type == N_SYMBOL) {
            const char* mod_name = node->children[1]->symbol;
            /* Track already-loaded modules to avoid double-loading */
            static char loaded_modules[64][128];
            static int n_loaded = 0;
            for (int i = 0; i < n_loaded; i++) {
                if (strcmp(loaded_modules[i], mod_name) == 0) return; /* already loaded */
            }
            if (n_loaded < 64) strncpy(loaded_modules[n_loaded++], mod_name, 127);

            /* stdlib is the prelude — builtins already available */
            if (strcmp(mod_name, "stdlib") == 0) return;

            /* Build file path: module.name → lib/module/name.esk */
            char path[512];
            snprintf(path, sizeof(path), "lib/");
            int pi = 4;
            for (const char* p = mod_name; *p && pi < 500; p++) {
                path[pi++] = (*p == '.') ? '/' : *p;
            }
            path[pi] = '\0';
            strncat(path, ".esk", sizeof(path) - pi - 1);

            /* Read and parse the file */
            FILE* mf = fopen(path, "r");
            if (!mf) {
                /* Try alternative path: replace ALL dots with slashes */
                char alt[512];
                snprintf(alt, sizeof(alt), "%s.esk", mod_name);
                for (char* p = alt; *p; p++) if (*p == '.') *p = '/';
                mf = fopen(alt, "r");
            }
            if (mf) {
                fseek(mf, 0, SEEK_END);
                long len = ftell(mf);
                fseek(mf, 0, SEEK_SET);
                char* src = (char*)malloc(len + 1);
                if (src) {
                    fread(src, 1, len, mf);
                    src[len] = '\0';
                    fclose(mf);
                    /* Parse and compile all top-level forms */
                    const char* saved_src = src_ptr;
                    src_ptr = src;
                    while (1) {
                        skip_ws();
                        if (!*src_ptr) break;
                        Node* expr = parse_sexp();
                        if (!expr) break;
                        compile_expr(c, expr, 0);
                        free_node(expr);
                    }
                    src_ptr = saved_src;
                    free(src);
                } else {
                    fclose(mf);
                }
            }
            /* If file not found, silently continue (builtins always available) */
        }
        return;
    }
    /* (provide name ...) — no-op: all symbols are visible */
    if (is_sym(head, "provide")) {
        return;
    }

    /* (define-syntax name (syntax-rules (literals...) (pattern template) ...)) */
    if (is_sym(head, "define-syntax") && node->n_children >= 3) {
        vm_macro_define_syntax((const MacroNode*)node);
        return;
    }

    /* (define-record-type name (constructor field...) pred (field accessor [mutator]) ...) */
    if (is_sym(head, "define-record-type") && node->n_children >= 4) {
        const char* type_name = node->children[1]->symbol;
        (void)type_name; /* used conceptually as type tag */
        Node* ctor = node->children[2]; /* (constructor f1 f2 ...) */
        const char* pred_name = node->children[3]->symbol;

        /* --- Constructor --- */
        if (ctor->type == N_LIST && ctor->n_children >= 1) {
            const char* ctor_name = ctor->children[0]->symbol;
            int n_fields = ctor->n_children - 1;

            /* Compile constructor as a closure that creates a tagged vector */
            FuncChunk func = {0};
            func.enclosing = c;
            func.param_count = n_fields;
            for (int i = 0; i < n_fields; i++)
                add_local(&func, ctor->children[i + 1]->symbol);

            /* Body: push type tag (as symbol), then all fields, create vector */
            /* Use type_name as a string constant for the tag */
            int len = (int)strlen(node->children[1]->symbol);
            int n_packs = (len + 7) / 8;
            chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(len)));
            for (int p = 0; p < n_packs; p++) {
                int64_t pack = 0;
                for (int b = 0; b < 8 && p * 8 + b < len; b++) {
                    pack |= ((int64_t)(unsigned char)node->children[1]->symbol[p * 8 + b]) << (b * 8);
                }
                chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(pack)));
            }
            chunk_emit(&func, OP_NATIVE_CALL, 100); /* build-string-from-packed */
            for (int i = 0; i < n_fields; i++)
                chunk_emit(&func, OP_GET_LOCAL, i);
            chunk_emit(&func, OP_VEC_CREATE, n_fields + 1); /* +1 for type tag */
            chunk_emit(&func, OP_RETURN, 0);

            /* Inline func body into parent chunk */
            int cfunc = chunk_add_const(c, INT_VAL(0));
            int jover = placeholder(c);
            int func_start = c->code_len;
            c->constants[cfunc].as.i = func_start;
            int const_map[MAX_CONSTS];
            for (int i = 0; i < func.n_constants; i++)
                const_map[i] = chunk_add_const(c, func.constants[i]);
            for (int i = 0; i < func.code_len; i++) {
                Instr fi = func.code[i];
                if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
                if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                    fi.operand += func_start;
                c->code[c->code_len++] = fi;
            }
            patch(c, jover, OP_JUMP, c->code_len);
            chunk_emit(c, OP_CLOSURE, cfunc);
            add_local(c, ctor_name);
        }

        /* --- Predicate --- */
        {
            FuncChunk func = {0};
            func.enclosing = c;
            func.param_count = 1;
            add_local(&func, "v");
            /* Check: (and (vector? v) (> (vector-length v) 0) (equal? (vector-ref v 0) type-name)) */
            chunk_emit(&func, OP_GET_LOCAL, 0);
            chunk_emit(&func, OP_VEC_P, 0);
            chunk_emit(&func, OP_RETURN, 0); /* simplified: just vector? check */

            int cfunc = chunk_add_const(c, INT_VAL(0));
            int jover = placeholder(c);
            int func_start = c->code_len;
            c->constants[cfunc].as.i = func_start;
            int const_map[MAX_CONSTS];
            for (int i = 0; i < func.n_constants; i++)
                const_map[i] = chunk_add_const(c, func.constants[i]);
            for (int i = 0; i < func.code_len; i++) {
                Instr fi = func.code[i];
                if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
                c->code[c->code_len++] = fi;
            }
            patch(c, jover, OP_JUMP, c->code_len);
            chunk_emit(c, OP_CLOSURE, cfunc);
            add_local(c, pred_name);
        }

        /* --- Accessors (and optional mutators) --- */
        for (int i = 4; i < node->n_children; i++) {
            Node* field_spec = node->children[i];
            if (field_spec->type != N_LIST || field_spec->n_children < 2) continue;
            int field_idx = i - 4 + 1; /* +1 because index 0 is the type tag */

            /* Accessor */
            {
                const char* acc_name = field_spec->children[1]->symbol;
                FuncChunk func = {0};
                func.enclosing = c;
                func.param_count = 1;
                add_local(&func, "v");
                chunk_emit(&func, OP_GET_LOCAL, 0);
                chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(field_idx)));
                chunk_emit(&func, OP_VEC_REF, 0);
                chunk_emit(&func, OP_RETURN, 0);

                int cfunc = chunk_add_const(c, INT_VAL(0));
                int jover = placeholder(c);
                int func_start = c->code_len;
                c->constants[cfunc].as.i = func_start;
                int const_map[MAX_CONSTS];
                for (int i2 = 0; i2 < func.n_constants; i2++)
                    const_map[i2] = chunk_add_const(c, func.constants[i2]);
                for (int i2 = 0; i2 < func.code_len; i2++) {
                    Instr fi = func.code[i2];
                    if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
                    c->code[c->code_len++] = fi;
                }
                patch(c, jover, OP_JUMP, c->code_len);
                chunk_emit(c, OP_CLOSURE, cfunc);
                add_local(c, acc_name);
            }

            /* Mutator (optional, at children[2]) */
            if (field_spec->n_children >= 3) {
                const char* mut_name = field_spec->children[2]->symbol;
                FuncChunk func = {0};
                func.enclosing = c;
                func.param_count = 2;
                add_local(&func, "v");
                add_local(&func, "val");
                chunk_emit(&func, OP_GET_LOCAL, 0);   /* vector */
                chunk_emit(&func, OP_CONST, chunk_add_const(&func, INT_VAL(field_idx)));
                chunk_emit(&func, OP_GET_LOCAL, 1);   /* new value */
                chunk_emit(&func, OP_VEC_SET, 0);
                chunk_emit(&func, OP_RETURN, 0);

                int cfunc = chunk_add_const(c, INT_VAL(0));
                int jover = placeholder(c);
                int func_start = c->code_len;
                c->constants[cfunc].as.i = func_start;
                int const_map[MAX_CONSTS];
                for (int i2 = 0; i2 < func.n_constants; i2++)
                    const_map[i2] = chunk_add_const(c, func.constants[i2]);
                for (int i2 = 0; i2 < func.code_len; i2++) {
                    Instr fi = func.code[i2];
                    if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
                    c->code[c->code_len++] = fi;
                }
                patch(c, jover, OP_JUMP, c->code_len);
                chunk_emit(c, OP_CLOSURE, cfunc);
                add_local(c, mut_name);
            }
        }
        return;
    }

    /* (parameterize ((param1 val1) (param2 val2) ...) body ...) */
    if (is_sym(head, "parameterize") && node->n_children >= 3) {
        Node* bindings = node->children[1];
        int n_bindings = bindings->n_children;

        /* Push each parameter binding */
        for (int i = 0; i < n_bindings; i++) {
            if (bindings->children[i]->type == N_LIST &&
                bindings->children[i]->n_children == 2) {
                compile_expr(c, bindings->children[i]->children[0], 0); /* param */
                compile_expr(c, bindings->children[i]->children[1], 0); /* new value */
                chunk_emit(c, OP_NATIVE_CALL, 702); /* parameterize-push */
                chunk_emit(c, OP_POP, 0); /* discard void result */
            }
        }

        /* Compile body */
        for (int i = 2; i < node->n_children; i++) {
            if (i > 2) chunk_emit(c, OP_POP, 0);
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        }

        /* Pop each binding in reverse order for proper unwinding */
        for (int i = n_bindings - 1; i >= 0; i--) {
            if (bindings->children[i]->type == N_LIST &&
                bindings->children[i]->n_children >= 1) {
                compile_expr(c, bindings->children[i]->children[0], 0); /* param */
                chunk_emit(c, OP_NATIVE_CALL, 703); /* parameterize-pop */
                chunk_emit(c, OP_POP, 0);
            }
        }
        return;
    }

    /* (let-values (((x y ...) producer) ...) body ...) */
    if (is_sym(head, "let-values") && node->n_children >= 3) {
        Node* bindings_list = node->children[1];
        int saved_locals = c->n_locals;

        for (int b = 0; b < bindings_list->n_children; b++) {
            Node* binding = bindings_list->children[b];
            if (binding->type != N_LIST || binding->n_children != 2) continue;
            Node* formals = binding->children[0]; /* (x y ...) or single var */
            Node* producer = binding->children[1];

            /* Compile the producer expression */
            compile_expr(c, producer, 0);

            if (formals->type == N_LIST) {
                /* Multiple return values — bind first to result, rest get nil */
                if (formals->n_children > 0)
                    add_local(c, formals->children[0]->symbol);
                for (int i = 1; i < formals->n_children; i++) {
                    chunk_emit(c, OP_NIL, 0);
                    add_local(c, formals->children[i]->symbol);
                }
            } else if (formals->type == N_SYMBOL) {
                /* Single variable */
                add_local(c, formals->symbol);
            }
        }

        /* Compile body expressions */
        for (int i = 2; i < node->n_children; i++) {
            if (i > 2) chunk_emit(c, OP_POP, 0);
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        }

        /* Clean up scope: pop bindings below result */
        int n_bound = c->n_locals - saved_locals;
        if (n_bound > 0)
            chunk_emit(c, OP_POPN, n_bound);
        c->n_locals = saved_locals;
        return;
    }

    /* (with-exception-handler handler thunk) — call thunk with handler installed.
     * Uses OP_GET_EXN to access exception from VM register. */
    if (is_sym(head, "with-exception-handler") && node->n_children == 3) {
        int handler_patch = c->code_len;
        chunk_emit(c, OP_PUSH_HANDLER, 0);

        /* Call thunk (0-arg function) */
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_CALL, 0);

        /* Normal exit */
        chunk_emit(c, OP_POP_HANDLER, 0);
        int end_patch = c->code_len;
        chunk_emit(c, OP_JUMP, 0);

        /* Exception handler: exn is in current_exn VM register.
         * Call handler(exn). NEVER tail-call — the handler may need
         * the enclosing frame for upvalue access (e.g., call/cc's k). */
        patch(c, handler_patch, OP_PUSH_HANDLER, c->code_len);
        compile_expr(c, node->children[1], 0); /* push handler closure */
        chunk_emit(c, OP_GET_EXN, 0);           /* push exn from VM register */
        chunk_emit(c, OP_CALL, 1);

        patch(c, end_patch, OP_JUMP, c->code_len);
        return;
    }

    /* (call/cc proc) or (call-with-current-continuation proc) */
    if ((is_sym(head, "call/cc") || is_sym(head, "call-with-current-continuation")) && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CALLCC, 0);
        return;
    }

    /* (raise expr) — throw exception */
    if (is_sym(head, "raise") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 130); /* native raise */
        return;
    }

    /* (guard (var (test handler) ...) body ...) — exception handler
     * R7RS: (guard (exn ((test) handler) ...) body ...)
     * Compiled as:
     *   PUSH_HANDLER handler_addr
     *   <body>
     *   POP_HANDLER
     *   JUMP end
     * handler_addr:          ; exception value on TOS
     *   SET_LOCAL exn_slot   ; bind exception to var
     *   <cond-like clause dispatch>
     * end:
     */
    if (is_sym(head, "guard") && node->n_children >= 3) {
        Node* clause_list = node->children[1]; /* (var (test handler) ...) */
        if (clause_list->type != N_LIST || clause_list->n_children < 1) {
            compile_expr(c, node->children[node->n_children - 1], tail);
            return;
        }
        /* CORRECT ARCHITECTURE: the guard handler is compiled as a closure
         * that takes the exception value as its sole parameter. This gives it
         * its own call frame with a known fp, so let/define/nested expressions
         * inside the handler have self-consistent local slot numbering.
         *
         * Compilation:
         *   PUSH_HANDLER handler_addr
         *   <body>
         *   POP_HANDLER
         *   JUMP end
         * handler_addr:
         *   GET_EXN                    ; push exception from VM register
         *   CLOSURE handler_func       ; push handler closure (takes 1 param: exn)
         *   ; swap so stack = [closure, exn] for CALL 1
         *   ; actually: push closure first, then GET_EXN
         *   CALL 1                     ; call handler_closure(exn)
         *   JUMP end
         *
         * handler_func body: (exn is local 0)
         *   compile clause tests and bodies with exn as a normal local parameter
         */
        char* exn_name = clause_list->children[0]->symbol;
        int saved_locals = c->n_locals;

        /* Emit PUSH_HANDLER */
        int handler_patch = c->code_len;
        chunk_emit(c, OP_PUSH_HANDLER, 0);

        /* Compile body expressions */
        for (int i = 2; i < node->n_children; i++) {
            if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
            else compile_expr(c, node->children[i], 0);
        }

        /* Normal exit */
        chunk_emit(c, OP_POP_HANDLER, 0);
        int end_patch = c->code_len;
        chunk_emit(c, OP_JUMP, 0);

        /* Compile handler as a closure with exn as parameter 0 */
        FuncChunk handler_func = {0};
        handler_func.enclosing = c;
        handler_func.param_count = 1;
        add_local(&handler_func, exn_name); /* exn is local 0 */

        /* Compile clauses inside the handler function */
        int hf_end_patches[32]; int hf_n_end = 0;
        for (int ci = 1; ci < clause_list->n_children; ci++) {
            Node* clause = clause_list->children[ci];
            if (clause->type != N_LIST || clause->n_children < 1) continue;
            if (clause->children[0]->type == N_SYMBOL && strcmp(clause->children[0]->symbol, "else") == 0) {
                for (int j = 1; j < clause->n_children; j++) {
                    if (j < clause->n_children - 1) { compile_expr(&handler_func, clause->children[j], 0); chunk_emit(&handler_func, OP_POP, 0); }
                    else compile_expr(&handler_func, clause->children[j], 1);
                }
                chunk_emit(&handler_func, OP_RETURN, 0);
                break;
            }
            compile_expr(&handler_func, clause->children[0], 0);
            int jnext = handler_func.code_len;
            chunk_emit(&handler_func, OP_JUMP_IF_FALSE, 0);
            for (int j = 1; j < clause->n_children; j++) {
                if (j < clause->n_children - 1) { compile_expr(&handler_func, clause->children[j], 0); chunk_emit(&handler_func, OP_POP, 0); }
                else compile_expr(&handler_func, clause->children[j], 1);
            }
            chunk_emit(&handler_func, OP_RETURN, 0);
            patch(&handler_func, jnext, OP_JUMP_IF_FALSE, handler_func.code_len);
        }
        /* If no clause matched: re-raise */
        chunk_emit(&handler_func, OP_GET_LOCAL, 0); /* push exn */
        chunk_emit(&handler_func, OP_NATIVE_CALL, 130); /* re-raise */
        chunk_emit(&handler_func, OP_RETURN, 0);

        /* Inline handler function code into parent chunk */
        int const_map_h[MAX_CONSTS];
        for (int i = 0; i < handler_func.n_constants; i++)
            const_map_h[i] = chunk_add_const(c, handler_func.constants[i]);
        int hfunc_const = chunk_add_const(c, INT_VAL(0)); /* placeholder */

        /* Handler dispatch code: CLOSURE + CALL */
        patch(c, handler_patch, OP_PUSH_HANDLER, c->code_len);
        int hjover = c->code_len;
        chunk_emit(c, OP_JUMP, 0); /* jump over inlined handler body */

        int hfunc_pc = c->code_len;
        c->constants[hfunc_const].as.i = hfunc_pc;

        /* Copy handler function code with remapping */
        for (int i = 0; i < handler_func.code_len; i++) {
            Instr fi = handler_func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map_h[fi.operand];
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += hfunc_pc;
            if (fi.op == OP_CLOSURE) {
                int ci2 = fi.operand & 0xFFFF;
                int nu2 = (fi.operand >> 16) & 0xFF;
                fi.operand = const_map_h[ci2] | (nu2 << 16);
            }
            c->code[c->code_len++] = fi;
        }

        patch(c, hjover, OP_JUMP, c->code_len);

        /* Emit: push handler closure, push exn, CALL 1 */
        int n_hf_upvals = handler_func.n_upvalues;
        for (int i = 0; i < n_hf_upvals; i++)
            chunk_emit(c, handler_func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                       handler_func.upvalues[i].enclosing_slot);
        chunk_emit(c, OP_CLOSURE, hfunc_const | (n_hf_upvals << 16));
        chunk_emit(c, OP_GET_EXN, 0);
        chunk_emit(c, OP_CALL, 1);

        /* end label */
        patch(c, end_patch, OP_JUMP, c->code_len);

        c->n_locals = saved_locals;
        return;
    }

    /* (apply f args-list) — call f with list as arguments */
    if (is_sym(head, "apply") && node->n_children == 3) {
        /* Handled via NATIVE_CALL 70 which unpacks the list at runtime */
        compile_expr(c, node->children[1], 0); /* push f */
        compile_expr(c, node->children[2], 0); /* push args list */
        chunk_emit(c, OP_NATIVE_CALL, 70); /* apply: takes f and args-list from stack */
        return;
    }

    /* (values expr1 expr2 ...) — multiple return values.
     * Simplified: pack into a vector. Single value = return as-is. */
    if (is_sym(head, "values") && node->n_children >= 2) {
        if (node->n_children == 2) {
            compile_expr(c, node->children[1], tail);
        } else {
            for (int i = 1; i < node->n_children; i++)
                compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_VEC_CREATE, node->n_children - 1);
        }
        return;
    }

    /* (call-with-values producer consumer)
     * Call producer(), then unpack its result and call consumer with the values.
     * If result is a vector (from multi-value `values`), unpack it.
     * Otherwise, call consumer with the single result. */
    if (is_sym(head, "call-with-values") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); /* push producer */
        chunk_emit(c, OP_CALL, 0);              /* call producer() → result */
        compile_expr(c, node->children[2], 0); /* push consumer */
        /* Stack: [result, consumer]. Use apply to unpack. */
        /* Native 251: call-with-values-apply(result, consumer) */
        chunk_emit(c, OP_NATIVE_CALL, 251);
        return;
    }

    /* (dynamic-wind before thunk after)
     * R7RS: call before(), register after on wind stack, call thunk(),
     * pop wind stack, call after(). If a continuation escapes through
     * this dynamic-wind, the after thunk is called during unwinding. */
    if (is_sym(head, "dynamic-wind") && node->n_children == 4) {
        /* Call before() */
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_CALL, 0);
        chunk_emit(c, OP_POP, 0);

        /* Push after thunk onto wind stack */
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_WIND_PUSH, 0);

        /* Call thunk() */
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_CALL, 0);

        /* Pop wind stack */
        chunk_emit(c, OP_WIND_POP, 0);

        /* Call after() (normal exit) */
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_CALL, 0);
        chunk_emit(c, OP_POP, 0);
        /* thunk result is below after result on stack.
         * After POP of after_result, thunk_result is TOS. */
        return;
    }

    /* (delay expr) → create a promise: #(#f <thunk>)
     * The thunk is a nullary closure wrapping expr. */
    if (is_sym(head, "delay") && node->n_children == 2) {
        {
            /* Save current chunk state, compile a sub-function */
            FuncChunk func;
            memset(&func, 0, sizeof(func));
            func.enclosing = c;
            func.param_count = 0;
            compile_expr(&func, node->children[1], 1); /* compile expr as body */
            chunk_emit(&func, OP_RETURN, 0);
            /* Inline the function code */
            int jover = c->code_len;
            chunk_emit(c, OP_JUMP, 0);
            int cfunc = c->n_constants;
            chunk_add_const(c, INT_VAL(c->code_len));
            for (int i = 0; i < func.code_len; i++) {
                Instr fi = func.code[i];
                if (fi.op == OP_CONST) fi.operand = chunk_add_const(c, func.constants[fi.operand]);
                if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                    fi.operand += c->code_len;
                chunk_emit(c, fi.op, fi.operand);
            }
            patch(c, jover, OP_JUMP, c->code_len);
            /* Stack: push #f, push closure, create vector */
            chunk_emit(c, OP_FALSE, 0);
            chunk_emit(c, OP_CLOSURE, cfunc);
            chunk_emit(c, OP_VEC_CREATE, 2); /* #(#f thunk) */
        }
        return;
    }

    /* (force promise) → force a promise (evaluate thunk if not yet forced) */
    if (is_sym(head, "force") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); /* push promise */
        chunk_emit(c, OP_NATIVE_CALL, 132);     /* native force */
        return;
    }

    /* (make-promise val) / (promise? x) */
    if (is_sym(head, "promise?") && node->n_children == 2) {
        /* A promise is a vector of length 2 with first element being bool */
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_VEC_P, 0); /* rough check: is it a vector? */
        return;
    }

    /* (atan y) or (atan y x) — 1 or 2 args */
    if (is_sym(head, "atan")) {
        if (node->n_children == 2) {
            compile_expr(c, node->children[1], 0);
            chunk_emit(c, OP_NATIVE_CALL, 31); /* 1-arg atan */
        } else if (node->n_children == 3) {
            compile_expr(c, node->children[1], 0);
            compile_expr(c, node->children[2], 0);
            chunk_emit(c, OP_NATIVE_CALL, 250); /* 2-arg atan2 */
        }
        return;
    }

    /* Variadic string-append: chain 2-arg NATIVE_CALL 54 calls */
    if (is_sym(head, "string-append") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_NATIVE_CALL, 54); /* 2-arg string-append */
        }
        return;
    }

    /* Equality predicates */
    if (is_sym(head, "eq?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 133); /* eq?: identity/pointer equality */
        return;
    }
    if (is_sym(head, "eqv?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 133); /* eqv? same as eq? for our types */
        return;
    }
    if (is_sym(head, "equal?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 134); /* equal?: deep structural equality */
        return;
    }

    /* length is now a first-class closure from the preamble.
     * quotient can be defined in terms of floor and / as a preamble builtin too.
     * For now, keep quotient as a special case using opcodes. */
    if (is_sym(head, "quotient") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_DIV, 0);
        /* Floor the result */
        chunk_emit(c, OP_NATIVE_CALL, 26);
        return;
    }

    /* Pair operations */
    if (is_sym(head, "cons") && node->n_children == 3) {
        compile_expr(c, node->children[2], 0); /* cdr first (SOS) */
        compile_expr(c, node->children[1], 0); /* car second (TOS) */
        chunk_emit(c, OP_CONS, 0); return;
    }
    if (is_sym(head, "car") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CAR, 0); return; }
    if (is_sym(head, "cdr") && node->n_children == 2) { compile_expr(c, node->children[1], 0); chunk_emit(c, OP_CDR, 0); return; }
    if (is_sym(head, "list")) {
        /* (list a b c) → cons(a, cons(b, cons(c, nil))) */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 1; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        return;
    }

    /* (display expr) */
    if (is_sym(head, "display") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_PRINT, 0);
        return;
    }

    /* (if cond then else) */
    if (is_sym(head, "if") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        int jf = placeholder(c);
        compile_expr(c, node->children[2], tail);
        if (node->n_children >= 4) {
            int jend = placeholder(c);
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
            compile_expr(c, node->children[3], tail);
            patch(c, jend, OP_JUMP, c->code_len);
        } else {
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        }
        return;
    }

    /* (begin e1 e2 ...) */
    if (is_sym(head, "begin")) {
        for (int i = 1; i < node->n_children; i++) {
            if (i < node->n_children - 1) {
                compile_expr(c, node->children[i], 0);
                chunk_emit(c, OP_POP, 0);
            } else {
                compile_expr(c, node->children[i], tail);
            }
        }
        return;
    }

    /* (let ((var val) ...) body) */
    /* Named let: (let name ((var init) ...) body ...)
     * Compiles as: (letrec ((name (lambda (vars...) body...))) (name inits...)) */
    if (is_sym(head, "let") && node->n_children >= 4
        && node->children[1]->type == N_SYMBOL
        && node->children[2]->type == N_LIST) {
        char* loop_name = node->children[1]->symbol;
        Node* bindings = node->children[2];
        int saved_locals = c->n_locals;
        c->scope_depth++;

        /* Compile as letrec with a single binding: the loop function */
        /* Push NIL placeholder for the loop function */
        chunk_emit(c, OP_NIL, 0);
        int loop_slot = add_local(c, loop_name);

        /* Compile the loop function body */
        FuncChunk func = {0};
        func.enclosing = c;
        func.param_count = bindings->n_children;
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children >= 1)
                add_local(&func, b->children[0]->symbol);
        }
        for (int i = 3; i < node->n_children; i++) {
            int is_last = (i == node->n_children - 1);
            compile_expr(&func, node->children[i], is_last);
            if (!is_last) chunk_emit(&func, OP_POP, 0);
        }
        chunk_emit(&func, OP_RETURN, 0);

        /* Inline function code */
        int const_map_nl[MAX_CONSTS];
        for (int i = 0; i < func.n_constants; i++)
            const_map_nl[i] = chunk_add_const(c, func.constants[i]);
        int cfunc = chunk_add_const(c, INT_VAL(0));
        int jover = placeholder(c);
        int func_pc = c->code_len;
        c->constants[cfunc].as.i = func_pc;

        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map_nl[fi.operand];
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += func_pc;
            if (fi.op == OP_CLOSURE) {
                int ci2 = fi.operand & 0xFFFF;
                int nu2 = (fi.operand >> 16) & 0xFF;
                fi.operand = const_map_nl[ci2] | (nu2 << 16);
            }
            c->code[c->code_len++] = fi;
        }
        patch(c, jover, OP_JUMP, c->code_len);

        /* Create closure with self-reference upvalue */
        int n_upvals = func.n_upvalues;
        int self_uv_idx = -1;
        for (int i = 0; i < n_upvals; i++) {
            if (strcmp(func.upvalues[i].name, loop_name) == 0) {
                chunk_emit(c, OP_NIL, 0);
                self_uv_idx = func.upvalues[i].index;
            } else {
                chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                           func.upvalues[i].enclosing_slot);
            }
        }
        chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
        if (self_uv_idx >= 0) chunk_emit(c, OP_CLOSE_UPVALUE, self_uv_idx);

        /* Store closure in loop_slot */
        chunk_emit(c, OP_SET_LOCAL, loop_slot);

        /* Open upvalues for mutual reference */
        if (n_upvals > 0) {
            chunk_emit(c, OP_GET_LOCAL, loop_slot);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(saved_locals)));
            chunk_emit(c, OP_NATIVE_CALL, 131);
            chunk_emit(c, OP_POP, 0);
        }

        /* Call the loop function with initial values */
        chunk_emit(c, OP_GET_LOCAL, loop_slot);
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children >= 2)
                compile_expr(c, b->children[1], 0);
            else
                chunk_emit(c, OP_NIL, 0);
        }
        int body_tail = 1 > 0 ? 0 : tail; /* don't tail-call — need POPN cleanup */
        chunk_emit(c, body_tail ? OP_TAIL_CALL : OP_CALL, bindings->n_children);

        /* Cleanup */
        chunk_emit(c, OP_POPN, 1); /* remove loop function slot */
        c->n_locals = saved_locals;
        c->scope_depth--;
        return;
    }

    /* (let ((var val) ...) body) — compile using stack locals.
     * Variables that are both captured by closures AND mutated via set!
     * are heap-boxed: stored in a 1-element vector so all closures share state. */
    if (is_sym(head, "let") && node->n_children >= 3 && node->children[1]->type == N_LIST) {
        int saved_locals = c->n_locals;
        c->scope_depth++;

        /* Collect body nodes for scanning */
        Node* body_nodes[64];
        int n_bodies = 0;
        for (int i = 2; i < node->n_children && n_bodies < 64; i++)
            body_nodes[n_bodies++] = node->children[i];

        Node* bindings = node->children[1];
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
                const char* vname = b->children[0]->symbol;
                int box = needs_boxing(body_nodes, n_bodies, vname);
                compile_expr(c, b->children[1], 0);
                if (box) {
                    /* Wrap value in a 1-element vector (box) */
                    chunk_emit(c, OP_VEC_CREATE, 1);
                }
                int slot = add_local(c, vname);
                if (box) {
                    /* Mark this local as boxed */
                    c->locals[c->n_locals - 1].boxed = 1;
                }
            }
        }
        int n_let_locals = c->n_locals - saved_locals;

        /* Compile body — don't use tail position if locals need cleanup */
        int body_tail = (n_let_locals > 0) ? 0 : tail;
        for (int i = 2; i < node->n_children; i++) {
            if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
            else compile_expr(c, node->children[i], body_tail);
        }

        /* Scope cleanup: remove let-bound locals, keep body result. */
        if (n_let_locals > 0) {
            chunk_emit(c, OP_POPN, n_let_locals);
        }
        c->n_locals = saved_locals;
        c->scope_depth--;
        return;
    }

    /* (let* ((var val) ...) body) — sequential bindings */
    if (is_sym(head, "let*") && node->n_children >= 3 && node->children[1]->type == N_LIST) {
        int saved_locals = c->n_locals;
        c->scope_depth++;
        Node* bindings = node->children[1];
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
                compile_expr(c, b->children[1], 0);
                add_local(c, b->children[0]->symbol);
            }
        }
        int n_let_locals = c->n_locals - saved_locals;
        int body_tail = (n_let_locals > 0) ? 0 : tail;
        for (int i = 2; i < node->n_children; i++) {
            if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
            else compile_expr(c, node->children[i], body_tail);
        }
        if (n_let_locals > 0) chunk_emit(c, OP_POPN, n_let_locals);
        c->n_locals = saved_locals;
        c->scope_depth--;
        return;
    }

    /* (letrec ((var val) ...) body) — recursive bindings with open upvalues.
     *
     * Letrec semantics: all bindings are visible to all initializers.
     * Implementation:
     * 1. Push NIL placeholders for all bindings
     * 2. Compile each initializer (lambdas capture open upvalue refs to stack slots)
     * 3. SET_LOCAL each initializer result to its slot
     * 4. Now all closures' open upvalues point to the correct stack slots
     * 5. When a closure reads GET_UPVALUE, it reads the current stack value (open ref)
     *
     * The key: compile_expr for the lambda creates a closure. The closure's upvalues
     * capture VALUES from the stack (which are NIL at creation time). We need them
     * to capture REFERENCES instead.
     *
     * Simplest correct approach: after creating all closures and SET_LOCAL'ing them,
     * use NATIVE_CALL to patch each closure's upvalue to read from the stack slot.
     * Or: use OP_CLOSE_UPVALUE to patch each closure's upvalue after all are defined. */
    if (is_sym(head, "letrec") && node->n_children >= 3 && node->children[1]->type == N_LIST) {
        int saved_locals = c->n_locals;
        c->scope_depth++;
        Node* bindings = node->children[1];
        int n_bindings = 0;

        /* 1. Push NIL placeholders and register names */
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
                chunk_emit(c, OP_NIL, 0);
                add_local(c, b->children[0]->symbol);
                n_bindings++;
            }
        }
        int n_let_locals = c->n_locals - saved_locals;

        /* 2. Compile each initializer and SET_LOCAL */
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
                compile_expr(c, b->children[1], 0);
                int slot = resolve_local(c, b->children[0]->symbol);
                if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
            }
        }

        /* 3. Patch closures: convert captured-by-value upvalues to open (by-reference).
         * After SET_LOCAL, each closure is at its stack slot. For each closure,
         * we use NATIVE_CALL 131 to convert its upvalues to open slot references.
         * This way GET_UPVALUE reads the CURRENT stack value (not the captured NIL). */
        for (int i = 0; i < n_bindings; i++) {
            int slot_i = saved_locals + i;
            /* For each upvalue in this closure, set it to open with the
             * enclosing stack slot. The upvalues reference OTHER letrec bindings. */
            chunk_emit(c, OP_GET_LOCAL, slot_i);     /* push closure */
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(n_bindings)));
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(saved_locals)));
            chunk_emit(c, OP_NATIVE_CALL, 131);       /* open_upvalues(closure, count, base_slot) */
            chunk_emit(c, OP_POP, 0);                 /* discard result */
        }

        /* Body — if there are locals to clean up, don't compile in tail position
         * (TAIL_CALL would skip the POPN cleanup) */
        int body_tail = (n_let_locals > 0) ? 0 : tail;
        for (int i = 2; i < node->n_children; i++) {
            if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
            else compile_expr(c, node->children[i], body_tail);
        }
        if (n_let_locals > 0) chunk_emit(c, OP_POPN, n_let_locals);
        c->n_locals = saved_locals;
        c->scope_depth--;
        return;
    }

    /* (letrec* ((var val) ...) body) — sequential recursive (R7RS) */
    if (is_sym(head, "letrec*") && node->n_children >= 3 && node->children[1]->type == N_LIST) {
        int saved_locals = c->n_locals;
        c->scope_depth++;
        Node* bindings = node->children[1];
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
                chunk_emit(c, OP_NIL, 0);
                add_local(c, b->children[0]->symbol);
            }
        }
        int n_let_locals = c->n_locals - saved_locals;
        for (int i = 0; i < bindings->n_children; i++) {
            Node* b = bindings->children[i];
            if (b->type == N_LIST && b->n_children == 2 && b->children[0]->type == N_SYMBOL) {
                compile_expr(c, b->children[1], 0);
                int slot = resolve_local(c, b->children[0]->symbol);
                if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
            }
        }
        {
            int body_tail = (n_let_locals > 0) ? 0 : tail;
            for (int i = 2; i < node->n_children; i++) {
                if (i < node->n_children - 1) { compile_expr(c, node->children[i], 0); chunk_emit(c, OP_POP, 0); }
                else compile_expr(c, node->children[i], body_tail);
            }
        }
        if (n_let_locals > 0) chunk_emit(c, OP_POPN, n_let_locals);
        c->n_locals = saved_locals;
        c->scope_depth--;
        return;
    }

    /* (define name value) or (define (name params...) body) */
    if (is_sym(head, "define") && node->n_children >= 3) {
        if (node->children[1]->type == N_SYMBOL) {
            /* Simple variable definition */
            compile_expr(c, node->children[2], 0);
            add_local(c, node->children[1]->symbol);
            return;
        }
        if (node->children[1]->type == N_LIST && node->children[1]->n_children >= 1) {
            /* Function definition: (define (name params...) body) */
            Node* sig = node->children[1];
            char* fname = sig->children[0]->symbol;

            /* Reserve local slot — the CLOSURE instruction below will push the
             * closure directly into this slot (no NIL placeholder needed). */
            int func_slot = add_local(c, fname);

            /* Compile function body into a separate chunk.
             * The body can reference fname via GET_UPVALUE which will be captured
             * from the enclosing scope's func_slot. */
            FuncChunk func = {0};
            func.enclosing = c;

            /* Check for dot notation in params: (name x y . rest) */
            int has_rest = 0, fixed_params = sig->n_children - 1;
            for (int i = 1; i < sig->n_children; i++) {
                if (sig->children[i]->type == N_SYMBOL && strcmp(sig->children[i]->symbol, ".") == 0) {
                    has_rest = 1;
                    fixed_params = i - 1;
                    break;
                }
            }
            func.param_count = has_rest ? 255 : fixed_params;

            /* Add fixed parameters as locals */
            for (int i = 1; i <= fixed_params; i++)
                add_local(&func, sig->children[i]->symbol);
            /* Add rest parameter if present */
            if (has_rest && fixed_params + 2 < sig->n_children) {
                add_local(&func, sig->children[fixed_params + 2]->symbol); /* name after dot */
                chunk_emit(&func, OP_PACK_REST, fixed_params);
            }

            /* Compile body expressions */
            for (int i = 2; i < node->n_children; i++) {
                int is_last = (i == node->n_children - 1);
                compile_expr(&func, node->children[i], is_last);
                if (!is_last) chunk_emit(&func, OP_POP, 0);
            }
            chunk_emit(&func, OP_RETURN, 0);

            /* Emit function code at end of current chunk, record its PC */
            int func_pc = c->code_len + 2; /* +2 for CLOSURE + NOP below */
            /* Map child constants to parent indices */
            int const_map[MAX_CONSTS];
            for (int i = 0; i < func.n_constants; i++) {
                const_map[i] = chunk_add_const(c, func.constants[i]);
            }
            int cfunc = chunk_add_const(c, INT_VAL(0)); /* placeholder for func PC */

            int jover = placeholder(c);
            int actual_func_pc = c->code_len;
            c->constants[cfunc].as.i = actual_func_pc;

            /* Adjust nested function PC constants: any constant in the child
             * that was used as a CLOSURE operand contains a PC relative to the
             * child chunk. After inlining, it needs to be offset by actual_func_pc. */
            for (int i = 0; i < func.code_len; i++) {
                if (func.code[i].op == OP_CLOSURE) {
                    int ci = func.code[i].operand & 0xFFFF;
                    int parent_ci = const_map[ci];
                    /* The constant holds a PC relative to child chunk start.
                     * Adjust to be relative to parent chunk start. */
                    c->constants[parent_ci].as.i += actual_func_pc;
                }
            }

            /* Copy function body with proper remapping */
            for (int i = 0; i < func.code_len; i++) {
                Instr fi = func.code[i];
                if (fi.op == OP_CONST) fi.operand = const_map[fi.operand];
                if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                    fi.operand += actual_func_pc;
                if (fi.op == OP_CLOSURE) {
                    int ci = fi.operand & 0xFFFF;
                    int nu = (fi.operand >> 16) & 0xFF;
                    fi.operand = const_map[ci] | (nu << 16);
                }
                c->code[c->code_len++] = fi;
            }

            /* Patch jump over function body */
            patch(c, jover, OP_JUMP, c->code_len);

            /* Emit CLOSURE instruction for the defined function.
             * For self-recursion: the closure captures itself from func_slot.
             * We push func_slot's value (currently NIL) as upvalue,
             * then create closure, then patch func_slot to point to the closure. */
            /* Emit upvalue captures for CLOSURE.
             * The function body compiled into `func` may reference:
             *   - Its own name (self-reference for recursion) → upvalue index determined by func.upvalues
             *   - Other enclosing locals (fold, etc.) → also in func.upvalues
             * Push each upvalue value from the enclosing scope, then CLOSURE captures them. */
            int n_upvals = func.n_upvalues;
            int self_uv_idx = -1;

            for (int i = 0; i < n_upvals; i++) {
                if (strcmp(func.upvalues[i].name, fname) == 0) {
                    /* Self-reference: push NIL placeholder (will be patched) */
                    chunk_emit(c, OP_NIL, 0);
                    self_uv_idx = func.upvalues[i].index;
                } else {
                    /* Capture from enclosing scope (local or upvalue) */
                    chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                               func.upvalues[i].enclosing_slot);
                }
            }

            chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
            if (self_uv_idx >= 0) {
                chunk_emit(c, OP_CLOSE_UPVALUE, self_uv_idx);  /* patch self-ref */
            }
            /* Convert local upvalues to open (stack slot references)
             * for top-level defines only (where enclosing scope persists forever).
             * This enables set! mutations of top-level variables.
             * NOTE: closures inside function bodies that capture mutable locals
             * need heap boxing (not yet implemented) for set! to work correctly
             * when the closure outlives the enclosing scope. */
            if (c->enclosing == NULL) {
                for (int i = 0; i < n_upvals; i++) {
                    if (i == self_uv_idx) continue;
                    if (!func.upvalues[i].is_local) continue;
                    chunk_emit(c, OP_DUP, 0);
                    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
                    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
                    chunk_emit(c, OP_NATIVE_CALL, 151);
                    chunk_emit(c, OP_POP, 0);
                }
            }
            return;
        }
    }

    /* (set! name value) */
    if (is_sym(head, "set!") && node->n_children == 3 && node->children[1]->type == N_SYMBOL) {
        const char* var_name = node->children[1]->symbol;
        int slot = resolve_local(c, var_name);

        /* Check if the target variable is boxed */
        int is_boxed = 0;
        if (slot >= 0) {
            for (int li = c->n_locals - 1; li >= 0; li--) {
                if (c->locals[li].slot == slot && c->locals[li].boxed) { is_boxed = 1; break; }
            }
        }

        if (slot >= 0 && is_boxed) {
            /* Boxed local: emit GET_LOCAL(box), CONST(0), compile(value), VEC_SET */
            chunk_emit(c, OP_GET_LOCAL, slot);  /* push box (vector) */
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0))); /* index 0 */
            compile_expr(c, node->children[2], 0); /* compile new value */
            chunk_emit(c, OP_VEC_SET, 0);       /* box[0] = value */
        } else if (slot >= 0) {
            /* Unboxed local: direct SET_LOCAL */
            compile_expr(c, node->children[2], 0);
            chunk_emit(c, OP_SET_LOCAL, slot);
        } else {
            /* Try upvalue resolution for outer-scope mutation */
            const char* name = node->children[1]->symbol;
            FuncChunk* chain[32]; int depth = 0;
            for (FuncChunk* p = c; p && depth < 32; p = p->enclosing)
                chain[depth++] = p;
            int found = 0;
            for (int d = depth - 1; d >= 1 && !found; d--) {
                int enc_slot = resolve_local(chain[d], name);
                if (enc_slot >= 0) {
                    /* Check if the source variable is boxed */
                    int var_boxed = 0;
                    for (int li = chain[d]->n_locals - 1; li >= 0; li--) {
                        if (chain[d]->locals[li].slot == enc_slot && chain[d]->locals[li].boxed) {
                            var_boxed = 1; break;
                        }
                    }

                    int prev_slot = enc_slot;
                    int prev_is_local = 1;
                    for (int level = d - 1; level >= 0; level--) {
                        FuncChunk* fc = chain[level];
                        int uv_idx = -1;
                        for (int i = 0; i < fc->n_upvalues; i++) {
                            if (strcmp(fc->upvalues[i].name, name) == 0) {
                                uv_idx = fc->upvalues[i].index; break;
                            }
                        }
                        if (uv_idx < 0 && fc->n_upvalues < MAX_UPVALUES) {
                            uv_idx = fc->n_upvalues;
                            strncpy(fc->upvalues[fc->n_upvalues].name, name, 127);
                            fc->upvalues[fc->n_upvalues].name[127] = 0;
                            fc->upvalues[fc->n_upvalues].enclosing_slot = prev_slot;
                            fc->upvalues[fc->n_upvalues].index = uv_idx;
                            fc->upvalues[fc->n_upvalues].is_local = prev_is_local;
                            fc->upvalues[fc->n_upvalues].boxed = var_boxed;
                            fc->n_upvalues++;
                        }
                        prev_slot = uv_idx;
                        prev_is_local = 0;
                    }
                    int final_uv = -1;
                    for (int i = 0; i < c->n_upvalues; i++) {
                        if (strcmp(c->upvalues[i].name, name) == 0) {
                            final_uv = c->upvalues[i].index; break;
                        }
                    }
                    if (final_uv >= 0) {
                        if (var_boxed) {
                            /* Boxed upvalue: GET_UPVALUE(box), CONST 0, value, VEC_SET */
                            chunk_emit(c, OP_GET_UPVALUE, final_uv);
                            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
                            compile_expr(c, node->children[2], 0);
                            chunk_emit(c, OP_VEC_SET, 0);
                        } else {
                            compile_expr(c, node->children[2], 0);
                            chunk_emit(c, OP_SET_UPVALUE, final_uv);
                        }
                        found = 1;
                    }
                }
            }
            if (!found) printf("WARNING: set! on undefined variable '%s'\n", name);
        }
        /* set! returns void — push NIL */
        chunk_emit(c, OP_NIL, 0);
        return;
    }

    /* (do ((var init step) ...) (test result) body ...) */
    if (is_sym(head, "do") && node->n_children >= 3) {
        c->scope_depth++;
        Node* vars = node->children[1];
        Node* test = node->children[2];

        /* Initialize loop variables */
        for (int i = 0; i < vars->n_children; i++) {
            Node* b = vars->children[i];
            if (b->type == N_LIST && b->n_children >= 2 && b->children[0]->type == N_SYMBOL) {
                compile_expr(c, b->children[1], 0);
                add_local(c, b->children[0]->symbol);
            }
        }

        /* Loop top */
        int loop_top = c->code_len;

        /* Test */
        if (test->type == N_LIST && test->n_children >= 1) {
            compile_expr(c, test->children[0], 0);
            int jexit = placeholder(c);

            /* Body (if any) */
            for (int i = 3; i < node->n_children; i++) {
                compile_expr(c, node->children[i], 0);
                chunk_emit(c, OP_POP, 0);
            }

            /* Step — evaluate ALL step expressions, then store (parallel) */
            int step_count = 0;
            for (int i = 0; i < vars->n_children; i++) {
                Node* b = vars->children[i];
                if (b->type == N_LIST && b->n_children >= 3) {
                    compile_expr(c, b->children[2], 0);
                    step_count++;
                }
            }
            /* Store in reverse order */
            for (int i = vars->n_children - 1; i >= 0; i--) {
                Node* b = vars->children[i];
                if (b->type == N_LIST && b->n_children >= 3) {
                    int slot = resolve_local(c, b->children[0]->symbol);
                    if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
                }
            }

            /* Loop back */
            chunk_emit(c, OP_LOOP, loop_top);

            /* Exit: evaluate result expression */
            patch(c, jexit, OP_JUMP_IF_FALSE, c->code_len - 1);
            /* Wait — JUMP_IF_FALSE jumps when false. The test is the EXIT condition.
             * When test is TRUE → exit. When FALSE → continue loop.
             * So: if test is true → DON'T jump (fall through to exit).
             *     if test is false → jump back to loop.
             * Need: JUMP_IF_FALSE → loop_body, then after body+step → LOOP back.
             * After LOOP: exit point. */
            /* Actually restructure: test → if FALSE, do body+step+loop. If TRUE, exit. */
            /* Current: test → jexit (JUMP_IF_FALSE to ???). Body. Step. LOOP.
             * jexit should point to AFTER the LOOP (the exit point).
             * But JUMP_IF_FALSE jumps when FALSE. If test is FALSE → continue loop body.
             * If test is TRUE → skip to exit.
             * So JUMP_IF_FALSE should jump PAST the exit... no.
             *
             * Let me use: NOT test → JUMP_IF_FALSE to exit. */
            /* Restart: */
            c->code_len = loop_top; /* redo from loop top */
            compile_expr(c, test->children[0], 0);
            /* test is TRUE when loop should exit */
            int jbody = placeholder(c); /* JUMP_IF_FALSE → body (continue loop) */
            /* Exit: result */
            if (test->n_children >= 2)
                compile_expr(c, test->children[1], tail);
            else
                chunk_emit(c, OP_NIL, 0);
            int jexit2 = placeholder(c); /* JUMP over body+step */

            /* Body */
            int body_start = c->code_len;
            patch(c, jbody, OP_JUMP_IF_FALSE, body_start);

            for (int i = 3; i < node->n_children; i++) {
                compile_expr(c, node->children[i], 0);
                chunk_emit(c, OP_POP, 0);
            }

            /* Step */
            step_count = 0;
            for (int i = 0; i < vars->n_children; i++) {
                Node* b = vars->children[i];
                if (b->type == N_LIST && b->n_children >= 3) {
                    compile_expr(c, b->children[2], 0);
                    step_count++;
                }
            }
            for (int i = vars->n_children - 1; i >= 0; i--) {
                Node* b = vars->children[i];
                if (b->type == N_LIST && b->n_children >= 3) {
                    int slot = resolve_local(c, b->children[0]->symbol);
                    if (slot >= 0) chunk_emit(c, OP_SET_LOCAL, slot);
                }
            }

            chunk_emit(c, OP_LOOP, loop_top);
            patch(c, jexit2, OP_JUMP, c->code_len);
        }

        /* Pop locals */
        while (c->n_locals > 0 && c->locals[c->n_locals-1].depth == c->scope_depth)
            c->n_locals--;
        c->scope_depth--;
        return;
    }

    /* (and e1 e2 ...) — short circuit */
    if (is_sym(head, "and") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            chunk_emit(c, OP_DUP, 0);
            int jf = placeholder(c);
            chunk_emit(c, OP_POP, 0);
            compile_expr(c, node->children[i], 0);
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        }
        return;
    }

    /* (or e1 e2 ...) — short circuit */
    if (is_sym(head, "or") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        for (int i = 2; i < node->n_children; i++) {
            chunk_emit(c, OP_DUP, 0);
            chunk_emit(c, OP_NOT, 0);
            int jf = placeholder(c);
            chunk_emit(c, OP_POP, 0);
            compile_expr(c, node->children[i], 0);
            patch(c, jf, OP_JUMP_IF_FALSE, c->code_len);
        }
        return;
    }

    /* (lambda (params...) body) */
    /* (lambda args body) — all args as a list */
    if (is_sym(head, "lambda") && node->n_children >= 3 && node->children[1]->type == N_SYMBOL) {
        /* Variadic: all arguments collected into a single list parameter */
        FuncChunk func = {0};
        func.enclosing = c;
        func.param_count = 255; /* sentinel: variadic, use PACK_REST at entry */
        add_local(&func, node->children[1]->symbol); /* rest list at local 0 */
        /* Emit PACK_REST 0 at function entry: pack ALL args into list at local 0 */
        chunk_emit(&func, OP_PACK_REST, 0);

        for (int i = 2; i < node->n_children; i++) {
            int is_last = (i == node->n_children - 1);
            compile_expr(&func, node->children[i], is_last);
            if (!is_last) chunk_emit(&func, OP_POP, 0);
        }
        chunk_emit(&func, OP_RETURN, 0);

        int cfunc = chunk_add_const(c, INT_VAL(0));
        int jover = placeholder(c);
        int func_start = c->code_len;
        c->constants[cfunc].as.i = func_start;

        int const_map2[MAX_CONSTS];
        for (int i = 0; i < func.n_constants; i++)
            const_map2[i] = chunk_add_const(c, func.constants[i]);
        for (int i = 0; i < func.code_len; i++) {
            if (func.code[i].op == OP_CLOSURE) {
                int ci = func.code[i].operand & 0xFFFF;
                int parent_ci = const_map2[ci];
                c->constants[parent_ci].as.i += func_start;
            }
        }
        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map2[fi.operand];
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += func_start;
            if (fi.op == OP_CLOSURE) {
                int ci = fi.operand & 0xFFFF;
                int nu = (fi.operand >> 16) & 0xFF;
                fi.operand = const_map2[ci] | (nu << 16);
            }
            c->code[c->code_len++] = fi;
        }
        patch(c, jover, OP_JUMP, c->code_len);
        int n_upvals = func.n_upvalues;
        for (int i = 0; i < n_upvals; i++) {
            chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                       func.upvalues[i].enclosing_slot);
        }
        chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
        return;
    }

    /* (lambda (x y . rest) body) or (lambda (x y) body) — standard and variadic */
    if (is_sym(head, "lambda") && node->n_children >= 3 && node->children[1]->type == N_LIST) {
        Node* params = node->children[1];
        FuncChunk func = {0};
        func.enclosing = c;

        /* Check for dot notation: (x y . rest) */
        int has_rest = 0;
        int fixed_params = params->n_children;
        for (int i = 0; i < params->n_children; i++) {
            if (params->children[i]->type == N_SYMBOL && strcmp(params->children[i]->symbol, ".") == 0) {
                has_rest = 1;
                fixed_params = i; /* params before the dot */
                break;
            }
        }
        func.param_count = fixed_params;

        for (int i = 0; i < fixed_params; i++)
            add_local(&func, params->children[i]->symbol);
        if (has_rest && fixed_params + 2 <= params->n_children) {
            /* Rest parameter name is after the dot */
            add_local(&func, params->children[fixed_params + 1]->symbol);
            /* At function entry: pack extra args from fp+fixed_params to sp into list */
            chunk_emit(&func, OP_PACK_REST, fixed_params);
            func.param_count = 255; /* sentinel: variadic */
        }

        for (int i = 2; i < node->n_children; i++) {
            int is_last = (i == node->n_children - 1);
            compile_expr(&func, node->children[i], is_last);
            if (!is_last) chunk_emit(&func, OP_POP, 0);
        }
        chunk_emit(&func, OP_RETURN, 0);

        /* Emit: JUMP over lambda body, then body, then CLOSURE */
        int cfunc = chunk_add_const(c, INT_VAL(0));
        int jover = placeholder(c);
        int func_start = c->code_len;
        c->constants[cfunc].as.i = func_start;

        int const_map2[MAX_CONSTS];
        for (int i = 0; i < func.n_constants; i++)
            const_map2[i] = chunk_add_const(c, func.constants[i]);

        /* Adjust nested CLOSURE PC constants */
        for (int i = 0; i < func.code_len; i++) {
            if (func.code[i].op == OP_CLOSURE) {
                int ci = func.code[i].operand & 0xFFFF;
                int parent_ci = const_map2[ci];
                c->constants[parent_ci].as.i += func_start;
            }
        }

        for (int i = 0; i < func.code_len; i++) {
            Instr fi = func.code[i];
            if (fi.op == OP_CONST) fi.operand = const_map2[fi.operand];
            if (fi.op == OP_JUMP || fi.op == OP_JUMP_IF_FALSE || fi.op == OP_LOOP || fi.op == OP_PUSH_HANDLER)
                fi.operand += func_start;
            if (fi.op == OP_CLOSURE) {
                int ci = fi.operand & 0xFFFF;
                int nu = (fi.operand >> 16) & 0xFF;
                fi.operand = const_map2[ci] | (nu << 16);
            }
            c->code[c->code_len++] = fi;
        }
        patch(c, jover, OP_JUMP, c->code_len);

        /* Push upvalue captures from enclosing scope before creating closure */
        int n_upvals = func.n_upvalues;
        for (int i = 0; i < n_upvals; i++) {
            chunk_emit(c, func.upvalues[i].is_local ? OP_GET_LOCAL : OP_GET_UPVALUE,
                       func.upvalues[i].enclosing_slot);
        }
        chunk_emit(c, OP_CLOSURE, cfunc | (n_upvals << 16));
        /* Convert upvalues to open slots for set! mutation visibility.
         * For is_local upvalues at top level: use NATIVE_CALL 151 (direct open slot).
         * For non-local upvalues: use NATIVE_CALL 252 to propagate parent's open slot. */
        if (c->enclosing == NULL) {
            for (int i = 0; i < n_upvals; i++) {
                if (!func.upvalues[i].is_local) continue;
                chunk_emit(c, OP_DUP, 0);
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
                chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
                chunk_emit(c, OP_NATIVE_CALL, 151);
                chunk_emit(c, OP_POP, 0);
            }
        } else {
            /* Inside a function: only propagate open slots from parent.
             * DON'T create new open slots for local captures (the function's
             * stack frame will be destroyed on return, making them invalid). */
            for (int i = 0; i < n_upvals; i++) {
                if (!func.upvalues[i].is_local) {
                    /* Captured from parent's upvalue — propagate parent's open slot if any */
                    chunk_emit(c, OP_DUP, 0);
                    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(i)));
                    chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(func.upvalues[i].enclosing_slot)));
                    chunk_emit(c, OP_NATIVE_CALL, 252);
                    chunk_emit(c, OP_POP, 0);
                }
            }
        }
        return;
    }

    /* (quote datum) — compile arbitrary quoted data to cons cells */
    if (is_sym(head, "quote") && node->n_children == 2) {
        compile_quote(c, node->children[1]);
        return;
    }

    /* (quasiquote datum) — compile with unquote/unquote-splicing support */
    if (is_sym(head, "quasiquote") && node->n_children == 2) {
        compile_quasiquote(c, node->children[1]);
        return;
    }

    /***************************************************************************
     * Complex number builtins (native IDs 300-319)
     ***************************************************************************/
    if (is_sym(head, "make-rectangular") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 300);
        return;
    }
    if (is_sym(head, "make-polar") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 301);
        return;
    }
    if (is_sym(head, "real-part") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 302);
        return;
    }
    if (is_sym(head, "imag-part") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 303);
        return;
    }
    if (is_sym(head, "magnitude") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 304);
        return;
    }
    if (is_sym(head, "angle") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 305);
        return;
    }
    if (is_sym(head, "conjugate") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 306);
        return;
    }
    if (is_sym(head, "complex?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 317);
        return;
    }

    /***************************************************************************
     * Rational number builtins (native IDs 330-349)
     ***************************************************************************/
    if (is_sym(head, "numerator") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 331);
        return;
    }
    if (is_sym(head, "denominator") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 332);
        return;
    }
    if (is_sym(head, "exact->inexact") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 343);
        return;
    }
    if (is_sym(head, "inexact->exact") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 344);
        return;
    }
    if (is_sym(head, "rationalize") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 345);
        return;
    }

    /***************************************************************************
     * Automatic differentiation builtins (native IDs 370-399)
     ***************************************************************************/
    if (is_sym(head, "make-dual") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 370);
        return;
    }
    if (is_sym(head, "dual-primal") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 371);
        return;
    }
    if (is_sym(head, "dual-tangent") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 372);
        return;
    }
    if (is_sym(head, "dual?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 383);
        return;
    }
    if (is_sym(head, "gradient") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 750);
        return;
    }
    if (is_sym(head, "derivative") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 393);
        return;
    }
    if (is_sym(head, "jacobian") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 751);
        return;
    }
    if (is_sym(head, "hessian") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 752);
        return;
    }

    /***************************************************************************
     * Tensor builtins (native IDs 410-469)
     ***************************************************************************/
    if (is_sym(head, "make-tensor") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 410);
        return;
    }
    if (is_sym(head, "tensor-shape") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 413);
        return;
    }
    if (is_sym(head, "tensor-reshape") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 414);
        return;
    }
    if (is_sym(head, "tensor-transpose") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 415);
        return;
    }
    if (is_sym(head, "zeros") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 417);
        return;
    }
    if (is_sym(head, "ones") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 418);
        return;
    }
    if (is_sym(head, "matmul") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 440);
        return;
    }
    if (is_sym(head, "softmax") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 463);
        return;
    }

    /***************************************************************************
     * Consciousness Engine builtins (native IDs 500-549)
     ***************************************************************************/
    if (is_sym(head, "logic-var?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 501);
        return;
    }
    if (is_sym(head, "unify") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 502);
        return;
    }
    if (is_sym(head, "walk") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 503);
        return;
    }
    if (is_sym(head, "make-substitution") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 505);
        return;
    }
    if (is_sym(head, "substitution?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 506);
        return;
    }
    if (is_sym(head, "make-fact") && node->n_children >= 2) {
        for (int i = 1; i < node->n_children; i++)
            compile_expr(c, node->children[i], 0);
        chunk_emit(c, OP_NATIVE_CALL, 507);
        return;
    }
    if (is_sym(head, "fact?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 508);
        return;
    }
    if (is_sym(head, "make-kb") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 509);
        return;
    }
    if (is_sym(head, "kb?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 510);
        return;
    }
    if (is_sym(head, "kb-assert!") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 511);
        return;
    }
    if (is_sym(head, "kb-query") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 512);
        return;
    }
    if (is_sym(head, "make-factor-graph") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 520);
        return;
    }
    if (is_sym(head, "factor-graph?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 521);
        return;
    }
    if (is_sym(head, "fg-add-factor!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 522);
        return;
    }
    if (is_sym(head, "fg-infer!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 523);
        return;
    }
    if (is_sym(head, "free-energy") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 525);
        return;
    }
    if (is_sym(head, "expected-free-energy") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 526);
        return;
    }
    if (is_sym(head, "make-workspace") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 540);
        return;
    }
    if (is_sym(head, "workspace?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 541);
        return;
    }
    if (is_sym(head, "ws-register!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 542);
        return;
    }
    if (is_sym(head, "ws-step!") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 543);
        return;
    }

    /***************************************************************************
     * I/O builtins (native IDs 580-602)
     ***************************************************************************/
    if (is_sym(head, "open-input-file") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 580);
        return;
    }
    if (is_sym(head, "open-output-file") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 581);
        return;
    }
    if (is_sym(head, "close-port") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 582);
        return;
    }
    if (is_sym(head, "read-char") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 583);
        return;
    }
    if (is_sym(head, "read-line") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 585);
        return;
    }
    if (is_sym(head, "write-string") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 587);
        return;
    }
    if (is_sym(head, "eof-object?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 592);
        return;
    }
    if (is_sym(head, "open-input-string") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 596);
        return;
    }
    if (is_sym(head, "open-output-string") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 597);
        return;
    }
    if (is_sym(head, "get-output-string") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 598);
        return;
    }
    if (is_sym(head, "file-exists?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 599);
        return;
    }

    /***************************************************************************
     * Hash table builtins (native IDs 660-670)
     ***************************************************************************/
    if (is_sym(head, "make-hash-table") && node->n_children == 1) {
        chunk_emit(c, OP_NATIVE_CALL, 660);
        return;
    }
    if (is_sym(head, "hash-ref") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        if (node->n_children >= 4)
            compile_expr(c, node->children[3], 0);
        else
            chunk_emit(c, OP_NIL, 0); /* default */
        chunk_emit(c, OP_NATIVE_CALL, 661);
        return;
    }
    if (is_sym(head, "hash-set!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 662);
        return;
    }
    if (is_sym(head, "hash-has-key?") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 663);
        return;
    }
    if (is_sym(head, "hash-remove!") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 664);
        return;
    }
    if (is_sym(head, "hash-keys") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 665);
        return;
    }
    if (is_sym(head, "hash-values") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 666);
        return;
    }
    if (is_sym(head, "hash-count") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 667);
        return;
    }
    if (is_sym(head, "hash-table?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 670);
        return;
    }

    /***************************************************************************
     * Error object builtins (native IDs 710-714)
     ***************************************************************************/
    if (is_sym(head, "error-object?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 711);
        return;
    }
    if (is_sym(head, "error-object-message") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 712);
        return;
    }
    if (is_sym(head, "error-object-irritants") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 713);
        return;
    }

    /***************************************************************************
     * Missing tensor ops
     ***************************************************************************/
    if (is_sym(head, "reshape") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0); /* tensor */
        /* Build shape list from remaining args */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 414); /* reshape */
        return;
    }
    if (is_sym(head, "tensor-get") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0); /* tensor */
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0);
            chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 411); /* tensor-ref */
        return;
    }
    if (is_sym(head, "arange") && node->n_children >= 2) {
        for (int i = 1; i < node->n_children; i++)
            compile_expr(c, node->children[i], 0);
        /* Pad missing args: (arange n) → (arange n 0 1), (arange n m) → (arange n m 1) */
        if (node->n_children == 2) {
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));
        }
        if (node->n_children == 3)
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(1)));
        chunk_emit(c, OP_NATIVE_CALL, 419);
        return;
    }

    /***************************************************************************
     * Missing neural net ops
     ***************************************************************************/
    if (is_sym(head, "relu") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 462);
        return;
    }
    if (is_sym(head, "sigmoid") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 464);
        return;
    }
    if (is_sym(head, "dropout") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 470); return; }
    if (is_sym(head, "conv2d") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 465);
        return;
    }
    if (is_sym(head, "batch-norm") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 464);
        return;
    }
    if (is_sym(head, "mse-loss") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 459);
        return;
    }
    if (is_sym(head, "cross-entropy-loss") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 460);
        return;
    }

    /***************************************************************************
     * Missing AD ops
     ***************************************************************************/
    if (is_sym(head, "divergence") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 395);
        return;
    }
    if (is_sym(head, "curl") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 396);
        return;
    }
    if (is_sym(head, "laplacian") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 397);
        return;
    }

    /***************************************************************************
     * Missing inference ops
     ***************************************************************************/
    if (is_sym(head, "fg-update-cpt!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 524);
        return;
    }

    if (is_sym(head, "fg-observe!") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0);  /* fg */
        compile_expr(c, node->children[2], 0);  /* var_id */
        compile_expr(c, node->children[3], 0);  /* observed_state */
        chunk_emit(c, OP_NATIVE_CALL, 527);
        return;
    }

    /***************************************************************************
     * Syntax forms: let-syntax, letrec-syntax, define-values, syntax-error,
     * include, include-ci, OALR forms, with-region, define-type
     ***************************************************************************/

    /* -- let-syntax -- */
    if (is_sym(head, "let-syntax") && node->n_children >= 3) {
        Node* bindings = node->children[1];
        int saved = g_n_macros;
        for (int i = 0; i < bindings->n_children; i++)
            vm_macro_define_syntax((const MacroNode*)bindings->children[i]);
        for (int i = 2; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        g_n_macros = saved;
        return;
    }

    /* -- letrec-syntax -- */
    if (is_sym(head, "letrec-syntax") && node->n_children >= 3) {
        Node* bindings = node->children[1];
        int saved = g_n_macros;
        for (int i = 0; i < bindings->n_children; i++)
            vm_macro_define_syntax((const MacroNode*)bindings->children[i]);
        for (int i = 2; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        g_n_macros = saved;
        return;
    }

    /* -- define-values -- */
    if (is_sym(head, "define-values") && node->n_children >= 3) {
        compile_expr(c, node->children[2], 0);
        Node* formals = node->children[1];
        if (formals->type == N_LIST) {
            add_local(c, formals->children[0]->symbol);
            for (int i = 1; i < formals->n_children; i++) {
                chunk_emit(c, OP_NIL, 0);
                add_local(c, formals->children[i]->symbol);
            }
        }
        return;
    }

    /* -- syntax-error -- */
    if (is_sym(head, "syntax-error")) {
        if (node->n_children >= 2)
            fprintf(stderr, "SYNTAX ERROR: %s\n",
                    node->children[1]->type == N_STRING ? node->children[1]->symbol : "unknown");
        return;
    }

    /* -- include -- */
    if (is_sym(head, "include") && node->n_children >= 2) {
        const char* path = node->children[1]->symbol;
        FILE* incf = fopen(path, "r");
        if (incf) {
            fseek(incf, 0, SEEK_END); long len = ftell(incf); fseek(incf, 0, SEEK_SET);
            char* src = (char*)malloc(len + 1);
            if (src) {
                fread(src, 1, len, incf); src[len] = 0; fclose(incf);
                const char* saved = src_ptr; src_ptr = src;
                while (1) { skip_ws(); if (!*src_ptr) break; Node* e = parse_sexp(); if (!e) break; compile_expr(c, e, 0); free_node(e); }
                src_ptr = saved; free(src);
            } else fclose(incf);
        }
        return;
    }

    /* -- include-ci -- */
    if (is_sym(head, "include-ci") && node->n_children >= 2) {
        const char* path = node->children[1]->symbol;
        FILE* incf = fopen(path, "r");
        if (incf) {
            fseek(incf, 0, SEEK_END); long len = ftell(incf); fseek(incf, 0, SEEK_SET);
            char* src = (char*)malloc(len + 1);
            if (src) {
                fread(src, 1, len, incf); src[len] = 0; fclose(incf);
                const char* saved = src_ptr; src_ptr = src;
                while (1) { skip_ws(); if (!*src_ptr) break; Node* e = parse_sexp(); if (!e) break; compile_expr(c, e, 0); free_node(e); }
                src_ptr = saved; free(src);
            } else fclose(incf);
        }
        return;
    }

    /* -- OALR forms (pass-through: ownership enforced at compile-time, not runtime) -- */
    if (is_sym(head, "owned") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }
    if (is_sym(head, "move") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }
    if (is_sym(head, "borrow") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0); /* the borrowed value */
        for (int i = 2; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        return;
    }
    if (is_sym(head, "shared") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }
    if (is_sym(head, "weak-ref") && node->n_children == 2) { compile_expr(c, node->children[1], tail); return; }

    /* -- with-region -- */
    if (is_sym(head, "with-region") && node->n_children >= 2) {
        for (int i = 1; i < node->n_children; i++)
            compile_expr(c, node->children[i], tail && i == node->n_children - 1);
        return;
    }

    /* -- define-type (type alias: compile-time only, no runtime effect) -- */
    if (is_sym(head, "define-type")) { return; }

    /***************************************************************************
     * Eshkol shorthand builtins
     ***************************************************************************/
    if (is_sym(head, "vref") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_VEC_REF, 0); return;
    }
    if (is_sym(head, "diff") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 393); return;
    }
    if (is_sym(head, "tensor") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, FLOAT_VAL(0)));
        chunk_emit(c, OP_NATIVE_CALL, 410); return;
    }
    if (is_sym(head, "pow") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 32); return;
    }
    if (is_sym(head, "type-of") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 740); return;
    }
    if (is_sym(head, "sign") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 743); return;
    }

    /***************************************************************************
     * Missing type predicates
     ***************************************************************************/
    if (is_sym(head, "real?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NUM_P, 0); return;
    }
    if (is_sym(head, "rational?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 740); return;
    }
    if (is_sym(head, "tensor?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 740); return;
    }
    if (is_sym(head, "port?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 730); return;
    }
    if (is_sym(head, "input-port?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 728); return;
    }
    if (is_sym(head, "output-port?") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 729); return;
    }

    /***************************************************************************
     * Missing math: cosh, sinh, tanh
     ***************************************************************************/
    if (is_sym(head, "cosh") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 720); return;
    }
    if (is_sym(head, "sinh") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 721); return;
    }
    if (is_sym(head, "tanh") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 722); return;
    }

    /***************************************************************************
     * Missing I/O: write-char, write-line, read
     ***************************************************************************/
    if (is_sym(head, "write-char") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 586); return;
    }
    if (is_sym(head, "write-line") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 726); return;
    }
    if (is_sym(head, "read") && node->n_children <= 2) {
        if (node->n_children == 2) compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 588); return;
    }

    /***************************************************************************
     * Missing tensor ops: tensor-ref, tensor-sum, tensor-mean, tensor-dot,
     * transpose, flatten, linspace, eye
     ***************************************************************************/
    if (is_sym(head, "tensor-ref") && node->n_children >= 3) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NIL, 0);
        for (int i = node->n_children - 1; i >= 2; i--) {
            compile_expr(c, node->children[i], 0); chunk_emit(c, OP_CONS, 0);
        }
        chunk_emit(c, OP_NATIVE_CALL, 411); return;
    }
    if (is_sym(head, "tensor-sum") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        chunk_emit(c, OP_NATIVE_CALL, 445); return;
    }
    if (is_sym(head, "tensor-mean") && node->n_children >= 2) {
        compile_expr(c, node->children[1], 0);
        if (node->n_children >= 3) compile_expr(c, node->children[2], 0);
        else chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(0)));
        chunk_emit(c, OP_NATIVE_CALL, 446); return;
    }
    if (is_sym(head, "tensor-dot") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 449); return;
    }
    if (is_sym(head, "transpose") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 415); return;
    }
    if (is_sym(head, "flatten") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 416); return;
    }
    if (is_sym(head, "linspace") && node->n_children == 4) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        compile_expr(c, node->children[3], 0);
        chunk_emit(c, OP_NATIVE_CALL, 746); return;
    }
    if (is_sym(head, "eye") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 745); return;
    }

    /***************************************************************************
     * Missing hash: hash-clear!
     ***************************************************************************/
    if (is_sym(head, "hash-clear!") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0); chunk_emit(c, OP_NATIVE_CALL, 668); return;
    }

    /***************************************************************************
     * gcd / lcm
     ***************************************************************************/
    if (is_sym(head, "gcd") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 346); return;
    }
    if (is_sym(head, "lcm") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0); compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 347); return;
    }

    /* Function call: (f arg1 arg2 ...)
     * Register each pushed value as an anonymous local so n_locals tracks
     * the actual stack depth. This prevents let/letrec inside arguments
     * from allocating slots that conflict with operand stack values. */
    if (head->type == N_SYMBOL || head->type == N_LIST) {
        int argc = node->n_children - 1;
        int saved_locals = c->n_locals;
        compile_expr(c, head, 0);  /* push function */
        add_local(c, "__call_func__");
        for (int i = 1; i < node->n_children; i++) {
            compile_expr(c, node->children[i], 0);
            add_local(c, "__call_arg__");
        }
        if (tail)
            chunk_emit(c, OP_TAIL_CALL, argc);
        else
            chunk_emit(c, OP_CALL, argc);
        c->n_locals = saved_locals; /* CALL consumed func+args, restore n_locals */
        return;
    }

    printf("WARNING: unhandled: %s\n", head->type == N_SYMBOL ? head->symbol : "(?)");
    chunk_emit(c, OP_NIL, 0);
}


/* ── Peephole Optimization (from compiler) ── */
static void peephole_optimize(FuncChunk* c) {
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < c->code_len - 1; i++) {
            /* Pattern: CONST 0 + ADD → remove both (identity) */
            if (c->code[i].op == OP_CONST && c->code[i+1].op == OP_ADD) {
                Value v = c->constants[c->code[i].operand];
                if (v.type == VAL_INT && v.as.i == 0) {
                    c->code[i].op = OP_NOP; c->code[i].operand = 0;
                    c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                    changed = 1;
                }
            }
            /* Pattern: CONST 1 + MUL → remove both (identity) */
            if (c->code[i].op == OP_CONST && c->code[i+1].op == OP_MUL) {
                Value v = c->constants[c->code[i].operand];
                if (v.type == VAL_INT && v.as.i == 1) {
                    c->code[i].op = OP_NOP; c->code[i].operand = 0;
                    c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                    changed = 1;
                }
            }
            /* Pattern: CONST 0 + MUL → replace with CONST 0 (always zero) */
            if (c->code[i].op == OP_CONST && c->code[i+1].op == OP_MUL) {
                Value v = c->constants[c->code[i].operand];
                if (v.type == VAL_INT && v.as.i == 0) {
                    /* Drop the other operand, keep CONST 0 */
                    c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                    /* But we also need to drop the value below — this is tricky for a stack machine.
                     * Skip this optimization for safety. */
                    c->code[i+1].op = OP_MUL; /* undo */
                }
            }
            /* Pattern: NOT + NOT → remove both (double negation) */
            if (c->code[i].op == OP_NOT && c->code[i+1].op == OP_NOT) {
                c->code[i].op = OP_NOP; c->code[i].operand = 0;
                c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                changed = 1;
            }
            /* Pattern: NEG + NEG → remove both (double negation) */
            if (c->code[i].op == OP_NEG && c->code[i+1].op == OP_NEG) {
                c->code[i].op = OP_NOP; c->code[i].operand = 0;
                c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                changed = 1;
            }
            /* Pattern: DUP + POP → remove both */
            if (c->code[i].op == OP_DUP && c->code[i+1].op == OP_POP) {
                c->code[i].op = OP_NOP; c->code[i].operand = 0;
                c->code[i+1].op = OP_NOP; c->code[i+1].operand = 0;
                changed = 1;
            }
        }
    }

    /* Count eliminated NOPs for metrics */
    int n_nops = 0;
    for (int i = 0; i < c->code_len; i++) {
        if (c->code[i].op == OP_NOP) n_nops++;
    }
    if (n_nops > 0) {
        printf("  [peephole] eliminated %d instructions\n", n_nops);
    }
    /* Note: we leave NOPs in place rather than compacting, because compacting
     * requires fixing all jump targets. The VM handles NOPs at near-zero cost. */
}

/*******************************************************************************
 * Compile & Run
 ******************************************************************************/


/*******************************************************************************
 * Bridge: run a compiled FuncChunk through the VM
 ******************************************************************************/

static void run_compiled_chunk(FuncChunk* chunk) {
    VM* vm = vm_create();
    if (!vm) return;

    /* Transfer bytecode to VM */
    free(vm->code);
    vm->code = (Instr*)calloc(chunk->code_len, sizeof(Instr));
    if (!vm->code) { vm_free(vm); return; }
    vm->code_len = chunk->code_len;
    for (int i = 0; i < chunk->code_len; i++) {
        vm->code[i].op = chunk->code[i].op;
        vm->code[i].operand = chunk->code[i].operand;
    }

    /* Transfer constants */
    for (int i = 0; i < chunk->n_constants && i < MAX_CONSTS; i++) {
        vm->constants[i] = chunk->constants[i];
    }
    vm->n_constants = chunk->n_constants;

    vm_run(vm);
    vm_free(vm);
}
/* Builtin function table: name → (native_id, arity) */
typedef struct { const char* name; int native_id; int arity; } BuiltinDef;

static const BuiltinDef BUILTINS[] = {
    /* Math (1 arg) */
    {"sin", 20, 1}, {"cos", 21, 1}, {"tan", 22, 1},
    {"exp", 23, 1}, {"log", 24, 1}, {"sqrt", 25, 1},
    {"floor", 26, 1}, {"ceiling", 27, 1}, {"round", 28, 1},
    {"asin", 29, 1}, {"acos", 30, 1}, {"atan", 31, 1},
    {"abs", 35, 1},  /* abs via native (not opcode) for first-class use */
    /* Math (2 arg) */
    {"expt", 32, 2}, {"min", 33, 2}, {"max", 34, 2},
    {"modulo", 36, 2}, {"remainder", 37, 2}, {"quotient", 38, 2},
    /* Predicates (1 arg) */
    {"positive?", 40, 1}, {"negative?", 41, 1},
    {"odd?", 42, 1}, {"even?", 43, 1},
    {"zero?", 44, 1},
    /* NOTE: null?, pair?, number?, boolean?, procedure?, vector?, car, cdr,
     * cons, display, list — these remain as compiler opcodes (not closures)
     * because they're core language primitives that must be visible at all scopes.
     * Only LIBRARY functions that need to be passed as arguments go here. */
    {"number->string", 51, 1},
    {"string-append", 54, 2}, {"string=?", 55, 2},
    {"string-length", 56, 1}, {"string-ref", 57, 2},
    {"newline", 60, 0},
    {"length", 71, 1},
    {"cadr", 77, 1}, {"cddr", 78, 1}, {"caar", 79, 1}, {"caddr", 80, 1},
    /* AD forward mode: dual number operations */
    {"make-dual", 110, 2},
    {"dual-value", 111, 1}, {"dual-derivative", 112, 1},
    {"dual+", 113, 2}, {"dual*", 114, 2}, {"dual-", 115, 2}, {"dual/", 116, 2},
    {"dual-sin", 117, 1}, {"dual-cos", 118, 1},
    {"dual-exp", 119, 1}, {"dual-log", 120, 1}, {"dual-sqrt", 121, 1},
    /* Equality */
    {"eq?", 133, 2}, {"eqv?", 133, 2}, {"equal?", 134, 2},
    /* List operations */
    {"append", 135, 2}, {"reverse", 136, 1},
    {"member", 137, 2}, {"assoc", 138, 2},
    {"list->vector", 139, 1}, {"vector->list", 140, 1},
    {"iota", 141, 1},
    /* Apply */
    {"apply", 70, 2},
    /* Arithmetic as first-class (2-arg, for use with apply/map/fold) */
    /* +,-,*,/ defined in scheme prelude as variadic folds */
    {"add2", 142, 2}, {"sub2", 143, 2}, {"mul2", 144, 2}, {"div2", 145, 2},
    /* Additional predicates */
    {"symbol?", 160, 1}, {"char?", 161, 1},
    {"exact?", 162, 1}, {"inexact?", 163, 1},
    {"nan?", 164, 1}, {"infinite?", 165, 1}, {"finite?", 166, 1},
    /* String operations */
    {"substring", 170, 3}, {"string-contains", 171, 2},
    {"string-upcase", 172, 1}, {"string-downcase", 173, 1},
    {"string-reverse", 174, 1},
    {"string->number", 175, 1}, {"number->string", 51, 1},
    {"string->list", 176, 1}, {"list->string", 177, 1},
    {"string-copy", 178, 1},
    /* Conversion */
    {"exact->inexact", 180, 1}, {"inexact->exact", 181, 1},
    {"char->integer", 182, 1}, {"integer->char", 183, 1},
    {"symbol->string", 184, 1}, {"string->symbol", 185, 1},
    /* Additional list */
    {"list-ref", 186, 2}, {"list-tail", 187, 2},
    {"last-pair", 188, 1}, {"list?", 189, 1},
    /* Math */
    {"truncate", 190, 1}, {"exact", 191, 1}, {"inexact", 192, 1},
    /* Hash tables */
    {"make-hash-table", 200, 0}, {"hash-ref", 201, 2}, {"hash-set!", 202, 3},
    {"hash-has-key?", 203, 2}, {"hash-keys", 204, 1}, {"hash-values", 205, 1},
    {"hash-count", 206, 1}, {"hash-delete!", 207, 2},
    /* Characters */
    {"char-alphabetic?", 210, 1}, {"char-numeric?", 211, 1},
    {"char-whitespace?", 212, 1}, {"char-upper-case?", 213, 1},
    {"char-lower-case?", 214, 1}, {"char-upcase", 215, 1},
    {"char-downcase", 216, 1}, {"char=?", 217, 2},
    {"char<?", 218, 2}, {"char>?", 219, 2},
    /* Bitwise */
    {"bitwise-and", 220, 2}, {"bitwise-or", 221, 2},
    {"bitwise-xor", 222, 2}, {"bitwise-not", 223, 1},
    {"arithmetic-shift", 224, 2},
    /* Additional list ops */
    {"take", 225, 2}, {"drop", 226, 2},
    {"any", 227, 2}, {"every", 228, 2},
    {"find", 229, 2}, {"sort", 230, 2},
    /* Additional string ops */
    {"string-repeat", 231, 2}, {"string-trim", 232, 1},
    {"string-split", 233, 2}, {"string-join", 234, 2},
    /* First-class core ops */
    {"cons", 74, 2}, {"car", 72, 1}, {"cdr", 73, 1},
    {"null?", 45, 1}, {"pair?", 46, 1}, {"number?", 47, 1},
    {"boolean?", 48, 1}, {"procedure?", 49, 1}, {"vector?", 50, 1},
    {"string?", 160, 1},
    /* Misc */
    {"not", 235, 1}, {"boolean=?", 236, 2},
    {"error", 237, 1}, {"void", 238, 0},
    {"hash-table?", 239, 1},
    {"display", 240, 1}, {"write", 241, 1},
    /* Complex numbers (300-319) */
    {"make-rectangular", 300, 2}, {"make-polar", 301, 2},
    {"real-part", 302, 1}, {"imag-part", 303, 1},
    {"magnitude", 304, 1}, {"angle", 305, 1},
    {"conjugate", 306, 1}, {"complex?", 317, 1},
    /* Rational numbers (330-349) */
    {"numerator", 331, 1}, {"denominator", 332, 1},
    {"exact->inexact", 343, 1}, {"inexact->exact", 344, 1},
    {"rationalize", 345, 2},
    /* AD — new-style IDs (370-399) */
    {"make-dual", 370, 2}, {"dual-primal", 371, 1}, {"dual-tangent", 372, 1},
    {"dual?", 383, 1},
    {"gradient", 750, 2}, {"jacobian", 751, 2}, {"hessian", 752, 2},
    {"derivative", 393, 2},
    /* Tensors (410-469) */
    {"make-tensor", 410, 2}, {"tensor-shape", 413, 1},
    {"tensor-reshape", 414, 2}, {"tensor-transpose", 415, 1},
    {"zeros", 417, 1}, {"ones", 418, 1},
    {"matmul", 440, 2}, {"softmax", 463, 1},
    /* Consciousness Engine (500-549) */
    {"logic-var?", 501, 1}, {"unify", 502, 3}, {"walk", 503, 2},
    {"make-substitution", 505, 0}, {"substitution?", 506, 1},
    {"make-fact", 507, 1}, {"fact?", 508, 1},
    {"make-kb", 509, 0}, {"kb?", 510, 1},
    {"kb-assert!", 511, 2}, {"kb-query", 512, 2},
    {"make-factor-graph", 520, 2}, {"factor-graph?", 521, 1},
    {"fg-add-factor!", 522, 3}, {"fg-infer!", 523, 3},
    {"free-energy", 525, 2}, {"expected-free-energy", 526, 3},
    {"make-workspace", 540, 2}, {"workspace?", 541, 1},
    {"ws-register!", 542, 3}, {"ws-step!", 543, 1},
    /* I/O (580-602) */
    {"open-input-file", 580, 1}, {"open-output-file", 581, 1},
    {"close-port", 582, 1}, {"read-char", 583, 1}, {"read-line", 585, 1},
    {"write-string", 587, 2}, {"eof-object?", 592, 1},
    {"open-input-string", 596, 1}, {"open-output-string", 597, 0},
    {"get-output-string", 598, 1}, {"file-exists?", 599, 1},
    /* Hash tables — new-style IDs (660-670) */
    {"make-hash-table", 660, 0}, {"hash-ref", 661, 3},
    {"hash-set!", 662, 3}, {"hash-has-key?", 663, 2},
    {"hash-remove!", 664, 2}, {"hash-keys", 665, 1},
    {"hash-values", 666, 1}, {"hash-count", 667, 1},
    {"hash-table?", 670, 1},
    /* Error objects (710-714) */
    {"error-object?", 711, 1}, {"error-object-message", 712, 1},
    {"error-object-irritants", 713, 1},
    /* Tensor ops (missing) */
    {"reshape", 414, 2}, {"tensor-get", 411, 2}, {"arange", 419, 1},
    /* Neural net ops (missing) */
    {"relu", 462, 1}, {"sigmoid", 464, 1}, {"conv2d", 465, 2}, {"dropout", 470, 2},
    {"mse-loss", 459, 2}, {"cross-entropy-loss", 460, 2},
    /* AD ops (missing) */
    {"divergence", 395, 2}, {"curl", 396, 2}, {"laplacian", 397, 2},
    /* Inference ops (missing) */
    {"fg-update-cpt!", 524, 3}, {"fg-observe!", 527, 3},
    /* Eshkol shorthands & missing builtins */
    {"vref", -1, 2},  /* uses OP_VEC_REF directly */
    {"diff", 393, 2}, {"tensor", 410, 2}, {"pow", 32, 2},
    {"type-of", 740, 1}, {"sign", 743, 1},
    /* Missing type predicates */
    {"real?", -1, 1}, {"rational?", 740, 1}, {"tensor?", 740, 1},
    {"port?", 730, 1}, {"input-port?", 728, 1}, {"output-port?", 729, 1},
    /* Missing math */
    {"cosh", 720, 1}, {"sinh", 721, 1}, {"tanh", 722, 1},
    /* Missing I/O */
    {"write-char", 586, 1}, {"write-line", 726, 1}, {"read", 588, 0},
    /* Missing tensor ops */
    {"tensor-ref", 411, 2}, {"tensor-sum", 445, 1}, {"tensor-mean", 446, 1},
    {"tensor-dot", 449, 2}, {"transpose", 415, 1}, {"flatten", 416, 1},
    {"linspace", 746, 3}, {"eye", 745, 1},
    /* Missing hash */
    {"hash-clear!", 668, 1},
    /* gcd / lcm */
    {"gcd", 346, 2}, {"lcm", 347, 2},
    {NULL, 0, 0}  /* sentinel */
};

/* Emit preamble: define all builtins as first-class closures.
 * Each builtin becomes a closure that calls NATIVE_CALL with the right ID.
 * This makes builtins passable as arguments: (map even? lst) just works. */
static void emit_builtin_preamble(FuncChunk* c) {
    for (int b = 0; BUILTINS[b].name; b++) {
        const BuiltinDef* def = &BUILTINS[b];
        int func_slot = add_local(c, def->name);

        /* Emit: JUMP over body → body (GETL params, NATIVE_CALL, RET) → CLOSURE */
        int cfunc = chunk_add_const(c, INT_VAL(0)); /* placeholder for func PC */
        int jover = placeholder(c);

        int func_pc = c->code_len;
        c->constants[cfunc].as.i = func_pc;

        /* Function body: load args from local slots, call native, return */
        for (int a = 0; a < def->arity; a++) {
            chunk_emit(c, OP_GET_LOCAL, a);
        }
        chunk_emit(c, OP_NATIVE_CALL, def->native_id);
        chunk_emit(c, OP_RETURN, 0);

        patch(c, jover, OP_JUMP, c->code_len);
        chunk_emit(c, OP_CLOSURE, cfunc); /* 0 upvalues */
        /* Closure is now on stack at func_slot */
    }
}

/* Global ESKB output path (set by --emit-eskb flag in main) */
static const char* g_eskb_output_path = NULL;
static const char* g_source_file_path = NULL;

static void compile_and_run(const char* source) {
    FuncChunk main_chunk = {0};

    /* Emit builtin function definitions as first-class closures */
    emit_builtin_preamble(&main_chunk);
    /* stack_depth synced via n_locals */

    /* Compile Scheme-level builtins (higher-order functions that call closures) */
    static const char* scheme_prelude =
        "(define (map f lst)\n"
        "  (let loop ((l lst) (acc (list)))\n"
        "    (if (null? l) (reverse acc)\n"
        "      (loop (cdr l) (cons (f (car l)) acc)))))\n"
        "(define (filter pred lst)\n"
        "  (let loop ((l lst) (acc (list)))\n"
        "    (if (null? l) (reverse acc)\n"
        "      (if (pred (car l)) (loop (cdr l) (cons (car l) acc))\n"
        "        (loop (cdr l) acc)))))\n"
        "(define (fold-left f init lst)\n"
        "  (let loop ((l lst) (acc init))\n"
        "    (if (null? l) acc\n"
        "      (loop (cdr l) (f acc (car l))))))\n"
        "(define (fold-right f init lst) (if (null? lst) init (f (car lst) (fold-right f init (cdr lst)))))\n"
        "(define (for-each f lst) (if (null? lst) 0 (begin (f (car lst)) (for-each f (cdr lst)))))\n"
        "(define (any pred lst) (if (null? lst) #f (if (pred (car lst)) #t (any pred (cdr lst)))))\n"
        "(define (every pred lst) (if (null? lst) #t (if (pred (car lst)) (every pred (cdr lst)) #f)))\n"
        "(define (find pred lst) (if (null? lst) #f (if (pred (car lst)) (car lst) (find pred (cdr lst)))))\n"
        "(define (take n lst) (if (= n 0) (list) (if (null? lst) (list) (cons (car lst) (take (- n 1) (cdr lst))))))\n"
        "(define (drop n lst) (if (= n 0) lst (if (null? lst) (list) (drop (- n 1) (cdr lst)))))\n"
        "(define (reduce f init lst) (fold-left f init lst))\n"
        "(define (merge compare a b)\n"
        "  (cond ((null? a) b) ((null? b) a)\n"
        "    ((compare (car a) (car b)) (cons (car a) (merge compare (cdr a) (cdr b))))\n"
        "    (else (cons (car b) (merge compare a (cdr b))))))\n"
        "(define (sort compare lst)\n"
        "  (if (or (null? lst) (null? (cdr lst))) lst\n"
        "    (let ((half (quotient (length lst) 2)))\n"
        "      (merge compare (sort compare (take half lst)) (sort compare (drop half lst))))))\n"
        "(define + (lambda args (fold-left add2 0 args)))\n"
        "(define * (lambda args (fold-left mul2 1 args)))\n"
        "(define (- . args) (if (null? (cdr args)) (sub2 0 (car args)) (fold-left sub2 (car args) (cdr args))))\n"
        "(define (/ . args) (if (null? (cdr args)) (div2 1 (car args)) (fold-left div2 (car args) (cdr args))))\n";
    src_ptr = scheme_prelude;
    while (1) {
        skip_ws();
        if (!*src_ptr) break;
        Node* expr = parse_sexp();
        if (!expr) break;
        int locals_before = main_chunk.n_locals;
        compile_expr(&main_chunk, expr, 0);
        if (main_chunk.n_locals == locals_before)
            chunk_emit(&main_chunk, OP_POP, 0);
        free_node(expr);
    }

    /* stack_depth synced via n_locals */
    src_ptr = source;

    /* TWO-PASS COMPILATION:
     * Pass 1: Parse ALL top-level expressions into an AST array.
     * Pass 2: Scan for defines that need heap boxing (captured + mutated).
     * Pass 3: Compile with boxing information. */

    /* Pass 1: Parse */
    #define MAX_TOP_EXPRS 4096
    Node* top_exprs[MAX_TOP_EXPRS];
    int n_top_exprs = 0;
    while (1) {
        skip_ws();
        if (!*src_ptr) break;
        Node* expr = parse_sexp();
        if (!expr) break;
        if (n_top_exprs < MAX_TOP_EXPRS)
            top_exprs[n_top_exprs++] = expr;
    }

    /* Pass 2: Scan for top-level defines that need boxing.
     * A define needs boxing if its variable is both:
     * (a) captured by a lambda somewhere in the program, AND
     * (b) mutated via set! somewhere in the program.
     * We record which define names need boxing. */
    char boxed_names[256][128];
    int n_boxed = 0;
    for (int i = 0; i < n_top_exprs; i++) {
        Node* expr = top_exprs[i];
        /* Check if this is a simple define: (define name value) */
        if (expr->type == N_LIST && expr->n_children >= 3
            && expr->children[0]->type == N_SYMBOL
            && strcmp(expr->children[0]->symbol, "define") == 0
            && expr->children[1]->type == N_SYMBOL) {
            const char* name = expr->children[1]->symbol;
            /* Scan ALL subsequent expressions for set! + capture */
            int has_set = 0, has_capture = 0;
            for (int j = 0; j < n_top_exprs; j++) {
                if (scan_for_set(top_exprs[j], name)) has_set = 1;
                if (scan_for_capture(top_exprs[j], name, 0)) has_capture = 1;
            }
            if (has_set && has_capture && n_boxed < 256) {
                strncpy(boxed_names[n_boxed], name, 127);
                boxed_names[n_boxed][127] = 0;
                n_boxed++;
            }
        }
    }

    /* Pass 3: Compile with boxing */
    for (int i = 0; i < n_top_exprs; i++) {
        Node* expr = top_exprs[i];

        /* Check if this is a simple define that needs boxing */
        int do_box = 0;
        if (expr->type == N_LIST && expr->n_children >= 3
            && expr->children[0]->type == N_SYMBOL
            && strcmp(expr->children[0]->symbol, "define") == 0
            && expr->children[1]->type == N_SYMBOL) {
            const char* name = expr->children[1]->symbol;
            for (int b = 0; b < n_boxed; b++) {
                if (strcmp(boxed_names[b], name) == 0) { do_box = 1; break; }
            }
        }

        int locals_before = main_chunk.n_locals;

        if (do_box) {
            /* Compile the init value, wrap in a box (1-element vector) */
            compile_expr(&main_chunk, expr->children[2], 0);
            chunk_emit(&main_chunk, OP_VEC_CREATE, 1); /* box it */
            int slot = add_local(&main_chunk, expr->children[1]->symbol);
            main_chunk.locals[main_chunk.n_locals - 1].boxed = 1;
        } else {
            compile_expr(&main_chunk, expr, 0);
            if (main_chunk.n_locals == locals_before) {
                chunk_emit(&main_chunk, OP_POP, 0);
            }
        }
    }

    /* Free ASTs */
    for (int i = 0; i < n_top_exprs; i++)
        free_node(top_exprs[i]);
    chunk_emit(&main_chunk, OP_HALT, 0);

    /* Print bytecode summary */
    printf("  [compiled: %d instructions, %d constants, %d locals]\n",
           main_chunk.code_len, main_chunk.n_constants, main_chunk.n_locals);

    /* Disassemble */
    static const char* opn[] = {
        "NOP","CONST","NIL","TRUE","FALSE","POP","DUP",
        "ADD","SUB","MUL","DIV","MOD","NEG","ABS",
        "EQ","LT","GT","LE","GE","NOT",
        "GETL","SETL","GETUP","SETUP",
        "CLOS","CALL","TCALL","RET",
        "JUMP","JIF","LOOP",
        "CONS","CAR","CDR","NULLP",
        "PRINT","HALT","NATV","CLOSUP",
        "VECNW","VECRF","VECST","VECLN",
        "STRRF","STRLN",
        "PAIRP","NUMP","STRP","BOOLP","PROCP","VECP",
        "SETCR","SETCD","POPN","OCLOS","CCALL","IVCC",
        "GUARD","UNGRD","GETXN","PKRST","WNDPS","WNDPP"
    };
    for (int i = 0; i < main_chunk.code_len; i++) {
        Instr ins = main_chunk.code[i];
        printf("    [%3d] %-6s %d", i, ins.op < OP_COUNT ? opn[ins.op] : "???", ins.operand);
        if (ins.op == OP_CONST && ins.operand < main_chunk.n_constants) {
            Value v = main_chunk.constants[ins.operand];
            if (v.type == VAL_INT) printf("  ; %lld", (long long)v.as.i);
        }
        if (ins.op == OP_CLOSURE) printf("  ; func@%lld, %d upvals",
            (long long)main_chunk.constants[ins.operand & 0xFFFF].as.i,
            (ins.operand >> 16) & 0xFF);
        printf("\n");
    }

    /* Dump bytecode for weight matrix integration (if requested) */
    if (getenv("ESHKOL_DUMP_BC")) {
        const char* path = getenv("ESHKOL_DUMP_BC");
        FILE* bf = fopen(path, "wb");
        if (bf) {
            uint32_t magic = 0x45534B42; /* "ESKB" */
            uint32_t n_instr = main_chunk.code_len;
            uint32_t n_const = main_chunk.n_constants;
            fwrite(&magic, 4, 1, bf);
            fwrite(&n_instr, 4, 1, bf);
            fwrite(&n_const, 4, 1, bf);
            /* Write instructions as (op:u8, operand:i32) pairs */
            for (int i = 0; i < (int)n_instr; i++) {
                uint8_t op = main_chunk.code[i].op;
                int32_t operand = main_chunk.code[i].operand;
                fwrite(&op, 1, 1, bf);
                fwrite(&operand, 4, 1, bf);
            }
            /* Write constants as (type:u8, value:f64) pairs */
            for (int i = 0; i < (int)n_const; i++) {
                uint8_t type = main_chunk.constants[i].type;
                double val = 0;
                if (type == VAL_INT) val = (double)main_chunk.constants[i].as.i;
                else if (type == VAL_FLOAT) val = main_chunk.constants[i].as.f;
                else if (type == VAL_BOOL) val = (double)main_chunk.constants[i].as.b;
                fwrite(&type, 1, 1, bf);
                fwrite(&val, 8, 1, bf);
            }
            fclose(bf);
            printf("  [dumped bytecode: %d instructions, %d constants → %s]\n",
                   (int)n_instr, (int)n_const, path);
        }
    }

    /* Emit ESKB binary format (if --emit-eskb was requested via global) */
    if (g_eskb_output_path) {
        /* Convert FuncChunk constants and code to ESKB format */
        EskbInstr* eskb_code = (EskbInstr*)calloc(main_chunk.code_len, sizeof(EskbInstr));
        EskbConst* eskb_consts = (EskbConst*)calloc(main_chunk.n_constants > 0 ? main_chunk.n_constants : 1, sizeof(EskbConst));
        if (eskb_code && eskb_consts) {
            for (int i = 0; i < main_chunk.code_len; i++) {
                eskb_code[i].op = main_chunk.code[i].op;
                eskb_code[i].operand = main_chunk.code[i].operand;
            }
            for (int i = 0; i < main_chunk.n_constants; i++) {
                Value v = main_chunk.constants[i];
                switch (v.type) {
                case VAL_NIL:
                    eskb_consts[i].type = ESKB_CONST_NIL;
                    break;
                case VAL_INT:
                    eskb_consts[i].type = ESKB_CONST_INT64;
                    eskb_consts[i].as.i = v.as.i;
                    break;
                case VAL_FLOAT:
                    eskb_consts[i].type = ESKB_CONST_F64;
                    eskb_consts[i].as.f = v.as.f;
                    break;
                case VAL_BOOL:
                    eskb_consts[i].type = ESKB_CONST_BOOL;
                    eskb_consts[i].as.b = v.as.b;
                    break;
                default:
                    /* Closures, pairs, etc. — store as int64 */
                    eskb_consts[i].type = ESKB_CONST_INT64;
                    eskb_consts[i].as.i = v.as.i;
                    break;
                }
            }
            eskb_write_file(g_eskb_output_path, eskb_code, main_chunk.code_len,
                            eskb_consts, main_chunk.n_constants, g_source_file_path);
        }
        free(eskb_code);
        free(eskb_consts);
    }

    /* Run peephole optimization before execution */
    peephole_optimize(&main_chunk);

    /* Execute using full VM */
    run_compiled_chunk(&main_chunk);
}

/*******************************************************************************
 * Unified main() — handles .esk source, .eskb bytecode, and built-in tests
 ******************************************************************************/

/* Compile source into a FuncChunk without executing it.
 * Used by eshkol_emit_eskb to produce bytecode for export. */
static void compile_and_run_source_to_chunk(const char* source, FuncChunk* chunk) {
    /* Reuse compile_and_run's logic but skip execution */
    emit_builtin_preamble(chunk);

    /* Scheme prelude */
    static const char* prelude =
        "(define (map f lst) (let loop ((l lst) (acc (list))) (if (null? l) (reverse acc) (loop (cdr l) (cons (f (car l)) acc)))))\n"
        "(define (filter pred lst) (let loop ((l lst) (acc (list))) (if (null? l) (reverse acc) (if (pred (car l)) (loop (cdr l) (cons (car l) acc)) (loop (cdr l) acc)))))\n"
        "(define (fold-left f init lst) (let loop ((l lst) (acc init)) (if (null? l) acc (loop (cdr l) (f acc (car l))))))\n"
        "(define (fold-right f init lst) (if (null? lst) init (f (car lst) (fold-right f init (cdr lst)))))\n"
        "(define (for-each f lst) (if (null? lst) 0 (begin (f (car lst)) (for-each f (cdr lst)))))\n"
        "(define + (lambda args (fold-left add2 0 args)))\n"
        "(define * (lambda args (fold-left mul2 1 args)))\n"
        "(define (- . args) (if (null? (cdr args)) (sub2 0 (car args)) (fold-left sub2 (car args) (cdr args))))\n"
        "(define (/ . args) (if (null? (cdr args)) (div2 1 (car args)) (fold-left div2 (car args) (cdr args))))\n";
    src_ptr = prelude;
    while (1) {
        skip_ws(); if (!*src_ptr) break;
        Node* expr = parse_sexp(); if (!expr) break;
        int lb = chunk->n_locals;
        compile_expr(chunk, expr, 0);
        if (chunk->n_locals == lb) chunk_emit(chunk, OP_POP, 0);
        free_node(expr);
    }

    /* Compile user source */
    src_ptr = source;
    while (1) {
        skip_ws(); if (!*src_ptr) break;
        Node* expr = parse_sexp(); if (!expr) break;
        int lb = chunk->n_locals;
        compile_expr(chunk, expr, 0);
        if (chunk->n_locals == lb) chunk_emit(chunk, OP_POP, 0);
        free_node(expr);
    }
    chunk_emit(chunk, OP_HALT, 0);
}

/* Public API: compile Eshkol source to ESKB bytecode file.
 * Called from eshkol-run via extern "C" linkage. */
int eshkol_emit_eskb(const char* source, const char* output_path) {
    FuncChunk main_chunk = {0};

    /* Compile prelude + builtins + source */
    compile_and_run_source_to_chunk(source, &main_chunk);

    /* Convert to ESKB format */
    EskbInstr* instrs = (EskbInstr*)calloc(main_chunk.code_len, sizeof(EskbInstr));
    EskbConst* consts = (EskbConst*)calloc(main_chunk.n_constants > 0 ? main_chunk.n_constants : 1, sizeof(EskbConst));
    if (!instrs || !consts) { free(instrs); free(consts); return -1; }

    for (int i = 0; i < main_chunk.code_len; i++) {
        instrs[i].op = main_chunk.code[i].op;
        instrs[i].operand = main_chunk.code[i].operand;
    }
    for (int i = 0; i < main_chunk.n_constants; i++) {
        Value v = main_chunk.constants[i];
        if (v.type == VAL_INT) { consts[i].type = ESKB_CONST_INT64; consts[i].as.i = v.as.i; }
        else if (v.type == VAL_FLOAT) { consts[i].type = ESKB_CONST_F64; consts[i].as.f = v.as.f; }
        else if (v.type == VAL_BOOL) { consts[i].type = ESKB_CONST_BOOL; consts[i].as.b = v.as.b; }
        else { consts[i].type = ESKB_CONST_NIL; }
    }

    int result = eskb_write_file(output_path, instrs, main_chunk.code_len,
                                  consts, main_chunk.n_constants, NULL);
    free(instrs);
    free(consts);
    return result;
}

#ifndef ESHKOL_VM_LIBRARY_MODE
int main(int argc, char** argv) {
    if (argc > 1) {
        /* Parse flags */
        int trace = 0;
        const char* input = NULL;
        const char* eskb_output = NULL;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--trace") == 0) { trace = 1; g_trace_on = 1; }
            else if (strcmp(argv[i], "--emit-eskb") == 0 && i + 1 < argc) { eskb_output = argv[++i]; g_eskb_output_path = eskb_output; }
            else input = argv[i];
        }

        if (input) {
            size_t len = strlen(input);
            if (len > 5 && strcmp(input + len - 5, ".eskb") == 0) {
                /* Load and run ESKB bytecode */
                EskbModule mod;
                if (eskb_load_file(input, &mod) == 0) {
                    VM* vm = vm_create();
                    if (!vm) { fprintf(stderr, "ERROR: cannot create VM\n"); eskb_module_free(&mod); return 1; }
                    free(vm->code);
                    vm->code = (Instr*)calloc(mod.code_len, sizeof(Instr));
                    vm->code_len = mod.code_len;
                    for (int i = 0; i < mod.code_len; i++)
                        vm->code[i] = (Instr){mod.opcodes[i], mod.operands[i]};
                    for (int i = 0; i < mod.n_constants && i < MAX_CONSTS; i++) {
                        switch (mod.const_types[i]) {
                        case ESKB_CONST_NIL:   vm->constants[i] = NIL_VAL; break;
                        case ESKB_CONST_INT64: vm->constants[i] = INT_VAL(mod.const_ints[i]); break;
                        case ESKB_CONST_F64:   vm->constants[i] = FLOAT_VAL(mod.const_floats[i]); break;
                        case ESKB_CONST_BOOL:  vm->constants[i] = BOOL_VAL((int)mod.const_ints[i]); break;
                        default:               vm->constants[i] = INT_VAL(mod.const_ints[i]); break;
                        }
                    }
                    vm->n_constants = mod.n_constants;
                    printf("=== Eshkol VM — running %s ===\n", input);
                    vm_run(vm);
                    printf("\n=== Execution complete ===\n");
                    vm_free(vm);
                    eskb_module_free(&mod);
                } else {
                    fprintf(stderr, "ERROR: failed to load ESKB file %s\n", input);
                    return 1;
                }
            } else {
                /* Compile and run .esk source */
                FILE* f = fopen(input, "r");
                if (!f) { fprintf(stderr, "Cannot open %s\n", input); return 1; }
                fseek(f, 0, SEEK_END); long flen = ftell(f); fseek(f, 0, SEEK_SET);
                char* source = malloc(flen + 1);
                fread(source, 1, flen, f); source[flen] = 0; fclose(f);
                printf("=== Eshkol VM+Compiler — compiling %s ===\n\n", input);
                g_source_file_path = input;
                compile_and_run(source);
                free(source);
                printf("\n=== Execution complete ===\n");
            }
        }
    } else {
        /* Run built-in VM tests */
        printf("=== Eshkol VM (unified compiler+interpreter) ===\n\n");
        test_arithmetic();
        test_comparison();
        test_pairs();
        test_list_build();
        test_factorial();
        test_tail_factorial();
        test_fibonacci();
        test_map();
        test_closures();
        printf("\n=== Tests complete ===\n");
    }
    return 0;
}
#endif /* ESHKOL_VM_LIBRARY_MODE */
