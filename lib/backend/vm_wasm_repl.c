/**
 * @file vm_wasm_repl.c
 * @brief WASM entry point for the Eshkol browser REPL.
 *
 * Provides a persistent REPL: definitions from one eval are available
 * in the next. The prelude is compiled once on first use.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define ESHKOL_VM_LIBRARY_MODE 1  /* Suppress eshkol_vm.c's main() */
#define ESHKOL_VM_NO_DISASM 1     /* Suppress bytecode dump in WASM builds */

#include "eshkol_vm.c"

#include <emscripten/emscripten.h>
#include <setjmp.h>

/* ── Persistent REPL State ── */

typedef struct {
    VM* vm;                  /* Persistent VM instance */
    FuncChunk base_chunk;    /* Compiled prelude (builtins + stdlib) */
    int base_n_locals;       /* Number of locals after prelude */
    int base_n_constants;    /* Number of constants after prelude */
    int initialized;
} ReplState;

static ReplState g_repl = {0};
static jmp_buf g_repl_jmp;

/* Initialize: compile prelude, create VM, run prelude to populate closures */
static void repl_setup(void) {
    chunk_init_arrays(&g_repl.base_chunk);

    /* Emit builtin preamble (native function closures) */
    emit_builtin_preamble(&g_repl.base_chunk);

    /* Compile Scheme prelude */
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
        skip_ws(); if (!*src_ptr) break;
        Node* expr = parse_sexp(); if (!expr) break;
        int lb = g_repl.base_chunk.n_locals;
        compile_expr(&g_repl.base_chunk, expr, 0);
        if (g_repl.base_chunk.n_locals == lb)
            chunk_emit(&g_repl.base_chunk, OP_POP, 0);
        free_node(expr);
    }

    /* Don't emit HALT — we'll append user code after this */
    g_repl.base_n_locals = g_repl.base_chunk.n_locals;
    g_repl.base_n_constants = g_repl.base_chunk.n_constants;

    /* Create persistent VM and run the prelude to populate closures on the stack */
    g_repl.vm = vm_create();
    if (!g_repl.vm) return;

    /* Transfer base chunk code to VM */
    free(g_repl.vm->code);
    g_repl.vm->code = (Instr*)calloc(g_repl.base_chunk.code_len + 1, sizeof(Instr));
    g_repl.vm->code_len = g_repl.base_chunk.code_len + 1;
    for (int i = 0; i < g_repl.base_chunk.code_len; i++)
        g_repl.vm->code[i] = g_repl.base_chunk.code[i];
    g_repl.vm->code[g_repl.base_chunk.code_len] = (Instr){OP_HALT, 0};

    /* Transfer constants */
    for (int i = 0; i < g_repl.base_chunk.n_constants && i < MAX_CONSTS; i++)
        g_repl.vm->constants[i] = g_repl.base_chunk.constants[i];
    g_repl.vm->n_constants = g_repl.base_chunk.n_constants;

    /* Run prelude — this populates the stack with builtin closures */
    vm_run(g_repl.vm);

    /* VM is now halted with all prelude closures on the stack.
     * Reset halted flag so we can continue running. */
    g_repl.vm->halted = 0;
    g_repl.vm->error = 0;

    g_repl.initialized = 1;
}

/* ── Public API ── */

EMSCRIPTEN_KEEPALIVE
void repl_init(void) {
    if (!g_repl.initialized) repl_setup();
}

EMSCRIPTEN_KEEPALIVE
const char* repl_eval(const char* source) {
    if (!g_repl.initialized) repl_setup();
    if (!g_repl.vm) return "ERROR: VM not initialized";

    /* Compile user expression into a temporary chunk that shares
     * the base chunk's locals/constants namespace */
    FuncChunk user_chunk; chunk_init_arrays(&user_chunk);

    /* Copy base locals so the compiler knows about prelude bindings */
    for (int i = 0; i < g_repl.base_chunk.n_locals && i < user_chunk.local_cap; i++) {
        if (i >= user_chunk.local_cap) {
            int new_cap = user_chunk.local_cap * 2;
            Local* nl = (Local*)realloc(user_chunk.locals, new_cap * sizeof(Local));
            if (!nl) break;
            user_chunk.locals = nl;
            user_chunk.local_cap = new_cap;
        }
        user_chunk.locals[i].name = g_repl.base_chunk.locals[i].name ? strdup(g_repl.base_chunk.locals[i].name) : NULL;
        user_chunk.locals[i].slot = g_repl.base_chunk.locals[i].slot;
        user_chunk.locals[i].depth = g_repl.base_chunk.locals[i].depth;
        user_chunk.locals[i].boxed = g_repl.base_chunk.locals[i].boxed;
    }
    user_chunk.n_locals = g_repl.base_chunk.n_locals;

    /* Copy base constants */
    for (int i = 0; i < g_repl.base_chunk.n_constants; i++)
        chunk_add_const(&user_chunk, g_repl.base_chunk.constants[i]);

    /* Parse and compile user source */
    src_ptr = source;

    /* Two-pass: parse all, then compile */
    Node* top_exprs[256];
    int n_top = 0;
    while (1) {
        skip_ws(); if (!*src_ptr) break;
        Node* expr = parse_sexp(); if (!expr) break;
        if (n_top < 256) top_exprs[n_top++] = expr;
    }

    for (int i = 0; i < n_top; i++) {
        int lb = user_chunk.n_locals;
        compile_expr(&user_chunk, top_exprs[i], 0);
        if (user_chunk.n_locals == lb) {
            /* Last expression: auto-print. Others: pop. */
            if (i == n_top - 1)
                chunk_emit(&user_chunk, OP_PRINT, 0);
            else
                chunk_emit(&user_chunk, OP_POP, 0);
        }
        free_node(top_exprs[i]);
    }
    chunk_emit(&user_chunk, OP_HALT, 0);

    /* Only run the NEW code (skip the base prelude instructions).
     * The VM's stack already has the prelude closures from the initial run. */
    int user_code_start = g_repl.base_chunk.code_len;
    int user_code_len = user_chunk.code_len - user_code_start;

    /* Rebuild VM code: base code + user code */
    int total_len = g_repl.base_chunk.code_len + user_code_len;
    free(g_repl.vm->code);
    g_repl.vm->code = (Instr*)calloc(total_len, sizeof(Instr));
    g_repl.vm->code_len = total_len;

    /* Copy base code */
    for (int i = 0; i < g_repl.base_chunk.code_len; i++)
        g_repl.vm->code[i] = g_repl.base_chunk.code[i];

    /* Copy user code (with constant index adjustment) */
    for (int i = user_code_start; i < user_chunk.code_len; i++)
        g_repl.vm->code[i] = user_chunk.code[i];

    /* Update VM constants (user code may have added new ones) */
    for (int i = g_repl.base_n_constants; i < user_chunk.n_constants && i < MAX_CONSTS; i++)
        g_repl.vm->constants[i] = user_chunk.constants[i];
    g_repl.vm->n_constants = user_chunk.n_constants;

    /* Set PC to start of user code, keep stack/fp intact */
    g_repl.vm->pc = user_code_start;
    g_repl.vm->halted = 0;
    g_repl.vm->error = 0;

    /* Run user code */
    vm_run(g_repl.vm);

    /* Update base chunk with new definitions (persist for next eval) */
    /* If user defined new locals, they're now on the VM's stack */
    if (user_chunk.n_locals > g_repl.base_chunk.n_locals) {
        /* Grow base locals to include new definitions */
        for (int i = g_repl.base_chunk.n_locals; i < user_chunk.n_locals; i++) {
            add_local(&g_repl.base_chunk, user_chunk.locals[i].name ? user_chunk.locals[i].name : "_");
        }
    }
    /* Persist new constants */
    for (int i = g_repl.base_n_constants; i < user_chunk.n_constants; i++)
        chunk_add_const(&g_repl.base_chunk, user_chunk.constants[i]);
    g_repl.base_n_constants = g_repl.base_chunk.n_constants;

    /* Persist new code (base chunk grows with each eval) */
    /* Remove the HALT at the end of user code — base should be continuous */
    int new_base_len = user_chunk.code_len - 1; /* exclude HALT */
    chunk_ensure_code_cap(&g_repl.base_chunk, new_base_len - g_repl.base_chunk.code_len);
    for (int i = g_repl.base_chunk.code_len; i < new_base_len; i++)
        g_repl.base_chunk.code[g_repl.base_chunk.code_len++] = user_chunk.code[i];

    /* Reset for next eval */
    g_repl.vm->halted = 0;
    g_repl.vm->error = 0;

    chunk_free_arrays(&user_chunk);
    return "";
}
