/**
 * @file eshkol_vm.c
 * @brief Eshkol bytecode VM — unity build hub.
 *
 * This file #includes all VM components in the correct order.
 * The VM implements a 63-opcode ISA with arena-based memory (OALR),
 * closures, continuations, and a full numeric tower.
 *
 * Components:
 *   vm_core.c      — Types, heap, stack, value representation
 *   vm_native.c    — 550+ native function implementations
 *   vm_run.c       — 63-opcode interpreter dispatch loop
 *   vm_tests.c     — Built-in test suite
 *   vm_parser.c    — S-expression tokenizer and parser
 *   vm_compiler.c  — Bytecode compiler (source → FuncChunk)
 *   vm_peephole.c  — Peephole optimization pass
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>

/* ESKB binary format */
#include "eskb_writer.c"
#include "eskb_reader.c"

/* Arena memory system (OALR regions) */
#include "vm_arena.h"

/* Unified numeric tower types */
#include "vm_numeric.h"

/* Runtime type libraries */
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

/* VM core: types, heap, stack operations */
#include "vm_core.c"

/* GPU tensor dispatch (threshold-based routing to Metal/CUDA) */
#include "vm_gpu_dispatch.h"

/* Geometric manifold operations (Riemannian, geodesic, Lie groups) */
#include "vm_geometric.c"

/* Native function dispatch (550+ functions) */
#include "vm_native.c"

/* VM interpreter: 63-opcode dispatch loop */
#include "vm_run.c"

/* Built-in tests */
#include "vm_tests.c"

/* S-expression parser */
#include "vm_parser.c"

/* Hygienic macro expander (syntax-rules) — must precede compiler */
#include "vm_macro.c"

/* Bytecode compiler */
#include "vm_compiler.c"

/* Peephole optimizer */
#include "vm_peephole.c"

/* Symbolic automatic differentiation */
#include "vm_symbolic_ad.c"


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
    /* Comparison operators as first-class functions */
    {"<", 146, 2}, {">", 147, 2}, {"<=", 148, 2}, {">=", 149, 2}, {"=", 150, 2},
    /* Core operations as first-class closures (native IDs 200-220) */
    {"car", 200, 1}, {"cdr", 201, 1}, {"cons", 202, 2},
    {"null?", 203, 1}, {"pair?", 204, 1}, {"not", 205, 1},
    {"number?", 206, 1}, {"string?", 207, 1}, {"boolean?", 208, 1},
    {"procedure?", 209, 1}, {"vector?", 210, 1},
    {"display", 211, 1}, {"write", 212, 1},
    {"exact->inexact", 213, 1}, {"inexact->exact", 214, 1},
    {"string->number", 215, 1},
    {"char->integer", 216, 1}, {"integer->char", 217, 1},
    {"make-vector", 218, 2}, {"vector-ref", 219, 2}, {"vector-set!", 220, 3},
    {"vector-length", 221, 1},
    {"string->list", 222, 1}, {"list->string", 223, 1},
    {"gcd", 224, 2}, {"lcm", 225, 2},
    {"make-string", 226, 2},
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
    /* First-class core ops — handled by native IDs 200-226 (defined above) */
    /* Misc */
    {"boolean=?", 236, 2},
    {"error", 237, 1}, {"void", 238, 0},
    {"hash-table?", 239, 1},
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
    /* numerator / denominator (rational parts) */
    {"numerator", 346, 1}, {"denominator", 347, 1},
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

/* Global ESKB output path — aliased through CompilerContext */
#define g_eskb_output_path g_compiler_ctx.eskb_output
#define g_source_file_path g_compiler_ctx.source_path

static void compile_and_run(const char* source) {
    FuncChunk main_chunk; chunk_init_arrays(&main_chunk);

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
        "    ((compare (car a) (car b)) (cons (car a) (merge compare (cdr a) b)))\n"
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
                /* Last non-define expression: print result (REPL behavior).
                 * Non-last expressions: discard result. */
                int is_last_expr = (i == n_top_exprs - 1);
#ifdef ESHKOL_VM_NO_DISASM
                /* WASM REPL mode: auto-print last expression result */
                if (is_last_expr) {
                    chunk_emit(&main_chunk, OP_PRINT, 0);
                } else {
                    chunk_emit(&main_chunk, OP_POP, 0);
                }
#else
                chunk_emit(&main_chunk, OP_POP, 0);
#endif
            }
        }
    }

    /* Free ASTs */
    for (int i = 0; i < n_top_exprs; i++)
        free_node(top_exprs[i]);
    chunk_emit(&main_chunk, OP_HALT, 0);

    /* Print bytecode summary + disassemble (skip in WASM / quiet mode) */
#ifdef ESHKOL_VM_NO_DISASM
    goto skip_disasm;
#else
    if (getenv("ESHKOL_VM_NO_DISASM")) goto skip_disasm;
#endif
    printf("  [compiled: %d instructions, %d constants, %d locals]\n",
           main_chunk.code_len, main_chunk.n_constants, main_chunk.n_locals);
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

    skip_disasm:
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
    FuncChunk main_chunk; chunk_init_arrays(&main_chunk);

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

/*******************************************************************************
 * Persistent REPL API — compile incrementally into an existing VM
 ******************************************************************************/

typedef struct {
    VM* vm;
    FuncChunk chunk;
    int initialized;
} ReplSession;

/* Load prelude from cache if available (eliminates ~50ms recompilation) */
#ifdef ESHKOL_VM_NO_DISASM
#include "vm_prelude_cache.h"
#endif

static int repl_load_prelude_cache(FuncChunk* chunk) {
#ifdef ESHKOL_VM_NO_DISASM
    /* Load cached prelude bytecode directly — skip parse+compile */
    chunk_ensure_code_cap(chunk, prelude_code_len);
    for (int i = 0; i < prelude_code_len; i++) {
        chunk->code[i] = (Instr){prelude_ops[i], prelude_operands[i]};
    }
    chunk->code_len = prelude_code_len;

    for (int i = 0; i < prelude_n_constants; i++) {
        Value v;
        v.type = prelude_const_types[i];
        if (v.type == VAL_FLOAT) v.as.f = prelude_const_floats[i];
        else v.as.i = prelude_const_ints[i];
        chunk_add_const(chunk, v);
    }

    for (int i = 0; i < prelude_n_locals; i++) {
        add_local(chunk, prelude_local_names[i]);
    }
    return 1; /* cache loaded */
#else
    (void)chunk;
    return 0; /* no cache available */
#endif
}

/* Create a REPL session: compile prelude, create VM, run prelude */
static ReplSession* repl_session_create(void) {
    ReplSession* rs = (ReplSession*)calloc(1, sizeof(ReplSession));
    if (!rs) return NULL;

    chunk_init_arrays(&rs->chunk);

    /* Try loading prelude from cache first (skips ~50ms recompilation) */
    int cache_loaded = repl_load_prelude_cache(&rs->chunk);

    if (!cache_loaded) {
        /* No cache — compile builtins + prelude from source */
        emit_builtin_preamble(&rs->chunk);
    }

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
        "    ((compare (car a) (car b)) (cons (car a) (merge compare (cdr a) b)))\n"
        "    (else (cons (car b) (merge compare a (cdr b))))))\n"
        "(define (sort compare lst)\n"
        "  (if (or (null? lst) (null? (cdr lst))) lst\n"
        "    (let ((half (quotient (length lst) 2)))\n"
        "      (merge compare (sort compare (take half lst)) (sort compare (drop half lst))))))\n"
        "(define + (lambda args (fold-left add2 0 args)))\n"
        "(define * (lambda args (fold-left mul2 1 args)))\n"
        "(define (- . args) (if (null? (cdr args)) (sub2 0 (car args)) (fold-left sub2 (car args) (cdr args))))\n"
        "(define (/ . args) (if (null? (cdr args)) (div2 1 (car args)) (fold-left div2 (car args) (cdr args))))\n";
    if (!cache_loaded) {
        src_ptr = scheme_prelude;
        while (1) {
            skip_ws(); if (!*src_ptr) break;
            Node* expr = parse_sexp(); if (!expr) break;
            int lb = rs->chunk.n_locals;
            compile_expr(&rs->chunk, expr, 0);
            if (rs->chunk.n_locals == lb)
                chunk_emit(&rs->chunk, OP_POP, 0);
            free_node(expr);
        }
    }

    /* Add HALT so we can run the prelude */
    int halt_pos = rs->chunk.code_len;
    chunk_emit(&rs->chunk, OP_HALT, 0);

    /* Create VM and transfer prelude code */
    rs->vm = vm_create();
    if (!rs->vm) { free(rs); return NULL; }

    free(rs->vm->code);
    rs->vm->code = (Instr*)calloc(rs->chunk.code_len, sizeof(Instr));
    rs->vm->code_len = rs->chunk.code_len;
    for (int i = 0; i < rs->chunk.code_len; i++)
        rs->vm->code[i] = rs->chunk.code[i];
    for (int i = 0; i < rs->chunk.n_constants && i < MAX_CONSTS; i++)
        rs->vm->constants[i] = rs->chunk.constants[i];
    rs->vm->n_constants = rs->chunk.n_constants;

    /* Run prelude to populate stack with closures */
    vm_run(rs->vm);
    rs->vm->halted = 0;
    rs->vm->error = 0;

    /* Remove the HALT — chunk is now the living base for incremental compilation */
    rs->chunk.code_len = halt_pos;

    rs->initialized = 1;
    return rs;
}

/* Evaluate an expression in an existing REPL session.
 * Definitions persist across calls. */
static jmp_buf g_repl_jmp;
static int g_repl_jmp_active = 0;

static void repl_session_eval(ReplSession* rs, const char* source, int auto_print) {
    if (!rs || !rs->initialized) return;

    /* Save state for error recovery — if eval fails, roll back */
    int code_start = rs->chunk.code_len;
    int const_start = rs->chunk.n_constants;
    int locals_start = rs->chunk.n_locals;
    int saved_sp = rs->vm->sp;
    int saved_fp = rs->vm->fp;
    int saved_frame_count = rs->vm->frame_count;

    /* setjmp boundary — catches fatal errors during compilation/execution */
    g_repl_jmp_active = 1;
    if (setjmp(g_repl_jmp) != 0) {
        /* Error occurred — roll back */
        rs->chunk.code_len = code_start;
        rs->chunk.n_constants = const_start;
        for (int i = locals_start; i < rs->chunk.n_locals; i++)
            free(rs->chunk.locals[i].name);
        rs->chunk.n_locals = locals_start;
        rs->vm->sp = saved_sp;
        rs->vm->fp = saved_fp;
        rs->vm->frame_count = saved_frame_count;
        rs->vm->halted = 0;
        rs->vm->error = 0;
        g_repl_jmp_active = 0;
        printf("Error during evaluation\n");
        return;
    }

    /* Parse and compile user expression INTO the existing chunk */
    src_ptr = source;
    Node* top_exprs[256];
    int n_top = 0;
    while (1) {
        skip_ws(); if (!*src_ptr) break;
        Node* expr = parse_sexp(); if (!expr) break;
        if (n_top < 256) top_exprs[n_top++] = expr;
    }

    for (int i = 0; i < n_top; i++) {
        int lb = rs->chunk.n_locals;
        compile_expr(&rs->chunk, top_exprs[i], 0);
        if (rs->chunk.n_locals == lb) {
            if (auto_print && i == n_top - 1)
                chunk_emit(&rs->chunk, OP_PRINT, 0);
            else
                chunk_emit(&rs->chunk, OP_POP, 0);
        }
        free_node(top_exprs[i]);
    }
    chunk_emit(&rs->chunk, OP_HALT, 0);

    /* Update VM code (full rebuild — base + user) */
    free(rs->vm->code);
    rs->vm->code = (Instr*)calloc(rs->chunk.code_len, sizeof(Instr));
    rs->vm->code_len = rs->chunk.code_len;
    for (int i = 0; i < rs->chunk.code_len; i++)
        rs->vm->code[i] = rs->chunk.code[i];

    /* Update VM constants */
    for (int i = const_start; i < rs->chunk.n_constants && i < MAX_CONSTS; i++)
        rs->vm->constants[i] = rs->chunk.constants[i];
    rs->vm->n_constants = rs->chunk.n_constants;

    /* Run from where user code starts */
    rs->vm->pc = code_start;
    rs->vm->halted = 0;
    rs->vm->error = 0;
    vm_run(rs->vm);

    if (rs->vm->error) {
        /* Error occurred — roll back to pre-eval state.
         * Definitions from this eval are discarded. */
        rs->chunk.code_len = code_start;
        rs->chunk.n_constants = const_start;
        /* Free any new local names */
        for (int i = locals_start; i < rs->chunk.n_locals; i++)
            free(rs->chunk.locals[i].name);
        rs->chunk.n_locals = locals_start;
        /* Restore VM stack state */
        rs->vm->sp = saved_sp;
        rs->vm->fp = saved_fp;
        rs->vm->frame_count = saved_frame_count;
    } else {
        /* Success — remove just the HALT, keep new definitions */
        rs->chunk.code_len--;
    }

    rs->vm->halted = 0;
    rs->vm->error = 0;
    g_repl_jmp_active = 0;
}

static void repl_session_destroy(ReplSession* rs) {
    if (!rs) return;
    if (rs->vm) vm_free(rs->vm);
    chunk_free_arrays(&rs->chunk);
    free(rs);
}

#if !defined(ESHKOL_VM_LIBRARY_MODE) && !defined(GENERATE_PRELUDE_CACHE)
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
        run_source_tests();
    }
    return 0;
}
#endif /* ESHKOL_VM_LIBRARY_MODE */
