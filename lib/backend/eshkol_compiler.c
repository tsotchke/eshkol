/**
 * @file eshkol_compiler.c
 * @brief Eshkol source → bytecode compiler targeting eshkol_vm.c ISA.
 *
 * Compiles S-expression source to the 38-opcode bytecode format.
 * Supports: arithmetic, comparisons, let/define, if/cond, do loops,
 * function definitions, lambda, closures, cons/car/cdr, display.
 *
 * Usage: ./eshkol_compiler [file.esk]
 *        Reads .esk, compiles to bytecode, executes via eshkol_vm.
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

/* Opcodes (must match eshkol_vm.c) */
typedef enum {
    OP_NOP=0, OP_CONST=1, OP_NIL=2, OP_TRUE=3, OP_FALSE=4, OP_POP=5, OP_DUP=6,
    OP_ADD=7, OP_SUB=8, OP_MUL=9, OP_DIV=10, OP_MOD=11, OP_NEG=12, OP_ABS=13,
    OP_EQ=14, OP_LT=15, OP_GT=16, OP_LE=17, OP_GE=18, OP_NOT=19,
    OP_GET_LOCAL=20, OP_SET_LOCAL=21, OP_GET_UPVALUE=22, OP_SET_UPVALUE=23,
    OP_CLOSURE=24, OP_CALL=25, OP_TAIL_CALL=26, OP_RETURN=27,
    OP_JUMP=28, OP_JUMP_IF_FALSE=29, OP_LOOP=30,
    OP_CONS=31, OP_CAR=32, OP_CDR=33, OP_NULL_P=34,
    OP_PRINT=35, OP_HALT=36, OP_NATIVE_CALL=37,
    OP_CLOSE_UPVALUE=38,
    /* Vectors */
    OP_VEC_CREATE=39,   /* operand = count; pops count values, creates vector */
    OP_VEC_REF=40,      /* TOS=index, SOS=vector → push vector[index] */
    OP_VEC_SET=41,      /* TOS=value, SOS=index, TOS-2=vector → set */
    OP_VEC_LEN=42,      /* TOS=vector → push length */
    /* Strings */
    OP_STR_REF=43,      /* TOS=index, SOS=string → push char */
    OP_STR_LEN=44,      /* TOS=string → push length */
    /* Type checks */
    OP_PAIR_P=45,       /* TOS → push (pair? TOS) */
    OP_NUM_P=46,        /* TOS → push (number? TOS) */
    OP_STR_P=47,        /* TOS → push (string? TOS) */
    OP_BOOL_P=48,       /* TOS → push (boolean? TOS) */
    OP_PROC_P=49,       /* TOS → push (procedure? TOS) */
    OP_VEC_P=50,        /* TOS → push (vector? TOS) */
    /* Set mutations */
    OP_SET_CAR=51,      /* TOS=val, SOS=pair → set car */
    OP_SET_CDR=52,      /* TOS=val, SOS=pair → set cdr */
    OP_POPN=53,         /* operand=N: pop N values below TOS, keeping TOS (scope cleanup) */
    OP_OPEN_CLOSURE=54,
    OP_CALLCC=55,       /* call/cc: capture continuation, call TOS with it */
    OP_INVOKE_CC=56,    /* invoke a captured continuation with a value */
    OP_PUSH_HANDLER=57, /* operand=handler_pc: save continuation, push exception handler */
    OP_POP_HANDLER=58,  /* remove topmost exception handler (normal guard exit) */
    OP_GET_EXN=59,      /* push current exception value (set by raise) */
    OP_PACK_REST=60,    /* operand=n_fixed: pack args from fp+n_fixed..sp into list at fp+n_fixed */
    OP_WIND_PUSH=61,    /* push after thunk onto wind stack */
    OP_WIND_POP=62,     /* pop from wind stack */

    OP_COUNT=63
} OpCode;

typedef struct { uint8_t op; int32_t operand; } Instr;

/* Value types for constant pool */
typedef enum {
    VAL_NIL=0, VAL_INT=1, VAL_FLOAT=2, VAL_BOOL=3,
    VAL_PAIR=4, VAL_CLOSURE=5, VAL_STRING=6, VAL_VECTOR=7,
    VAL_CONTINUATION=8, VAL_HASH=9
} ValType;
typedef struct { ValType type; union { int64_t i; double f; int b; int32_t ptr; } as; } Value;
#define INT_VAL(v) ((Value){.type=VAL_INT, .as.i=(v)})
#define FLOAT_VAL(v) ((Value){.type=VAL_FLOAT, .as.f=(v)})

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
#define MAX_CONSTS 1024
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
    if (is_sym(head, "tensor-save") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 802);
        return;
    }
    if (is_sym(head, "tensor-load") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 803);
        return;
    }
    if (is_sym(head, "model-save") && node->n_children == 3) {
        compile_expr(c, node->children[1], 0);
        compile_expr(c, node->children[2], 0);
        chunk_emit(c, OP_NATIVE_CALL, 800);
        return;
    }
    if (is_sym(head, "model-load") && node->n_children == 2) {
        compile_expr(c, node->children[1], 0);
        chunk_emit(c, OP_NATIVE_CALL, 801);
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

/*******************************************************************************
 * Full VM Execution Engine
 ******************************************************************************/

#define HEAP_SIZE 4194304  /* 4M objects */
#define STACK_SIZE 4096
#define MAX_FRAMES 256

typedef enum { HEAP_CONS=0, HEAP_CLOSURE=1, HEAP_STRING=2, HEAP_VECTOR=3, HEAP_CONTINUATION=4, HEAP_HASH=5 } HeapType;

typedef struct HeapObjectTag {
    HeapType type;
    union {
        struct { Value car; Value cdr; } cons;
        struct {
            int32_t func_pc;
            int32_t n_upvalues;
            Value upvalues[16];          /* closed upvalues (captured by value) */
            int32_t open_slots[16];      /* stack slots for open upvalues (-1 = closed) */
        } closure;
        struct { char data[256]; int32_t len; } string;
        struct { Value items[64]; int32_t len; } vector;
        struct { int32_t saved_pc; int32_t saved_sp; int32_t saved_fp; int32_t saved_frame_count;
                 Value saved_stack[256]; int used; int saved_wind_depth; } continuation;
        struct { Value keys[32]; Value vals[32]; int32_t count; } hash;
    };
} HeapObject;

typedef struct {
    int32_t return_pc, return_fp;
    int32_t heap_mark;  /* heap pointer at time of CALL — for OALR region cleanup */
    int32_t force_promise_ptr; /* -1 if not a force call, else heap index of promise */
} CallFrame;

/* Execute a compiled chunk through the full VM */
/* Forward declarations for recursive printer */
typedef struct HeapObjectTag HeapObject;
/* mode: 0=display (human-readable), 1=write (machine-readable with quotes/escapes) */
static void print_value(Value v, HeapObject* heap, int depth, int mode);

static void print_value(Value v, HeapObject* heap, int depth, int mode) {
    if (depth > 50) { printf("..."); return; }
    switch (v.type) {
    case VAL_INT:   printf("%lld", (long long)v.as.i); break;
    case VAL_FLOAT: printf("%.6g", v.as.f); break;
    case VAL_BOOL:  printf("%s", v.as.b ? "#t" : "#f"); break;
    case VAL_NIL:   printf("()"); break;
    case VAL_CLOSURE: printf("<closure>"); break;
    case VAL_CONTINUATION: printf("<continuation>"); break;
    case VAL_HASH: printf("<hash-table:%d>", heap[v.as.ptr].hash.count); break;
    case VAL_STRING:
        if (mode == 1) {
            /* write mode: output with quotes and escape sequences */
            printf("\"");
            const char* s = heap[v.as.ptr].string.data;
            int len = heap[v.as.ptr].string.len;
            for (int i = 0; i < len; i++) {
                switch (s[i]) {
                case '"':  printf("\\\""); break;
                case '\\': printf("\\\\"); break;
                case '\n': printf("\\n"); break;
                case '\t': printf("\\t"); break;
                case '\r': printf("\\r"); break;
                default:   putchar(s[i]); break;
                }
            }
            printf("\"");
        } else {
            /* display mode: output raw string contents */
            printf("%.*s", heap[v.as.ptr].string.len, heap[v.as.ptr].string.data);
        }
        break;
    case VAL_PAIR: {
        printf("(");
        Value cur = v; int first = 1;
        while (cur.type == VAL_PAIR && depth < 50) {
            if (!first) printf(" "); first = 0;
            print_value(heap[cur.as.ptr].cons.car, heap, depth + 1, mode);
            cur = heap[cur.as.ptr].cons.cdr;
        }
        if (cur.type != VAL_NIL) { printf(" . "); print_value(cur, heap, depth + 1, mode); }
        printf(")");
        break;
    }
    case VAL_VECTOR: {
        printf("#(");
        for (int i = 0; i < heap[v.as.ptr].vector.len; i++) {
            if (i > 0) printf(" ");
            print_value(heap[v.as.ptr].vector.items[i], heap, depth + 1, mode);
        }
        printf(")");
        break;
    }
    default: printf("<unknown>"); break;
    }
}

/* ── Peephole Optimization ── */
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

static void execute_chunk(FuncChunk* chunk) {
    /* Allocate VM on heap (too large for stack) */
    Value* stack = (Value*)calloc(STACK_SIZE, sizeof(Value));
    HeapObject* heap = (HeapObject*)calloc(HEAP_SIZE, sizeof(HeapObject));
    CallFrame* frames = (CallFrame*)calloc(MAX_FRAMES, sizeof(CallFrame));
    if (!stack || !heap || !frames) {
        fprintf(stderr, "ERROR: VM allocation failed\n");
        free(stack); free(heap); free(frames);
        return;
    }
    int32_t sp = 0, pc = 0, fp = 0, heap_next = 0;
    int frame_count = 0, halted = 0, error = 0;

    #define PUSH(v) do { if(sp>=STACK_SIZE){fprintf(stderr,"STACK OVERFLOW at PC=%d\n",pc);error=1;}else stack[sp++]=(v); } while(0)
    #define POP() (sp > 0 ? stack[--sp] : (fprintf(stderr,"STACK UNDERFLOW at PC=%d\n",pc),error=1, (Value){.type=VAL_NIL}))
    #define PEEK(off) ((sp-1-(off)) >= 0 ? stack[sp-1-(off)] : (Value){.type=VAL_NIL})
    #define AS_NUM(v) ((v).type==VAL_INT?(double)(v).as.i:(v).as.f)
    #define NUM_VAL(r) ((r)==(int64_t)(r)&&fabs(r)<1e15 ? INT_VAL((int64_t)(r)) : FLOAT_VAL(r))
    /* R7RS: only #f is falsy. Empty list '() is truthy. */
    #define IS_FALSY(v) ((v).type==VAL_BOOL && !(v).as.b)
    #define HALLOC() (heap_next < HEAP_SIZE ? heap_next++ : (printf("HEAP OOM\n"),error=1,-1))

    /* Exception handler stack (for guard/raise) */
    #define MAX_HANDLERS 32
    struct { int32_t handler_pc; int32_t saved_sp; int32_t saved_fp;
             int32_t saved_frame_count; } exc_handlers[MAX_HANDLERS];
    int handler_count = 0;
    Value current_exn = {.type = VAL_NIL}; /* current exception value (set by raise) */

    /* Dynamic-wind stack: tracks active before/after thunks */
    #define MAX_WINDS 32
    struct { Value after; /* after thunk closure */ } wind_stack[MAX_WINDS];
    int wind_depth = 0;

    /* Pending continuation invocation (for wind unwinding) */
    int pending_cc = -1;       /* heap index of pending continuation, -1 = none */
    Value pending_cc_result = {.type = VAL_NIL};

    int64_t insn_count = 0;
    int max_depth = 0;
    int trace_on = g_trace_on; /* set by --trace flag */
    while (!halted && !error && pc < chunk->code_len) {
        if (frame_count > max_depth) max_depth = frame_count;
        { int64_t max_insn = 10000000LL;
          const char* env_max = getenv("ESHKOL_VM_MAX_INSN");
          if (env_max) max_insn = atoll(env_max);
          if (++insn_count > max_insn) { printf("RUNAWAY (%lld insns, depth=%d, heap=%d)\n", (long long)max_insn, max_depth, heap_next); error=1; break; }
        }
        if (trace_on && insn_count < 500) {
            printf("  [%04d] op=%2d sp=%d fp=%d", pc-1, chunk->code[pc-1].op, sp, fp);
            if (sp > 0) { Value t = stack[sp-1]; printf(" TOS="); if(t.type==VAL_INT)printf("%lld",(long long)t.as.i); else if(t.type==VAL_VECTOR)printf("#vec@%d",t.as.ptr); else if(t.type==VAL_CLOSURE)printf("<cl@%d>",t.as.ptr); else if(t.type==VAL_BOOL)printf(t.as.b?"#t":"#f"); else if(t.type==VAL_NIL)printf("nil"); else printf("?"); }
            printf("\n");
        }
        Instr ins = chunk->code[pc++];
        switch (ins.op) {
        case OP_NOP: break;
        case OP_CONST: PUSH(chunk->constants[ins.operand]); break;
        case OP_NIL: PUSH(((Value){.type=VAL_NIL})); break;
        case OP_TRUE: PUSH(((Value){.type=VAL_BOOL,.as.b=1})); break;
        case OP_FALSE: PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break;
        case OP_POP: sp--; break;
        case OP_DUP: { Value v = PEEK(0); PUSH(v); break; }

        case OP_ADD: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)+AS_NUM(b))); break; }
        case OP_SUB: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)-AS_NUM(b))); break; }
        case OP_MUL: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)*AS_NUM(b))); break; }
        case OP_DIV: {
            Value b=POP(),a=POP(); double bv=AS_NUM(b);
            if (bv == 0.0 && b.type == VAL_INT) {
                /* Integer division by zero: raise error */
                Value exn_msg = INT_VAL(0); /* "division by zero" */
                if (handler_count > 0) {
                    handler_count--;
                    sp = exc_handlers[handler_count].saved_sp;
                    fp = exc_handlers[handler_count].saved_fp;
                    frame_count = exc_handlers[handler_count].saved_frame_count;
                    pc = exc_handlers[handler_count].handler_pc;
                    current_exn = exn_msg;
                } else { printf("ERROR: division by zero\n"); error=1; }
            } else {
                PUSH(NUM_VAL(AS_NUM(a)/bv));
            }
            break;
        }
        case OP_MOD: {
            Value b=POP(),a=POP();
            double bv=AS_NUM(b);
            if (bv == 0.0 && b.type == VAL_INT) {
                if (handler_count > 0) {
                    handler_count--;
                    sp = exc_handlers[handler_count].saved_sp;
                    fp = exc_handlers[handler_count].saved_fp;
                    frame_count = exc_handlers[handler_count].saved_frame_count;
                    pc = exc_handlers[handler_count].handler_pc;
                    current_exn = INT_VAL(0);
                } else { printf("ERROR: modulo by zero\n"); error=1; }
            } else {
                double r = fmod(AS_NUM(a), bv);
                if (r != 0 && ((r > 0) != (bv > 0))) r += bv;
                PUSH(NUM_VAL(r));
            }
            break;
        }
        case OP_NEG: { Value a=POP(); PUSH(NUM_VAL(-AS_NUM(a))); break; }
        case OP_ABS: { Value a=POP(); PUSH(NUM_VAL(fabs(AS_NUM(a)))); break; }

        case OP_EQ: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)==AS_NUM(b))})); break; }
        case OP_LT: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)<AS_NUM(b))})); break; }
        case OP_GT: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)>AS_NUM(b))})); break; }
        case OP_LE: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)<=AS_NUM(b))})); break; }
        case OP_GE: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)>=AS_NUM(b))})); break; }
        case OP_NOT: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=IS_FALSY(a)})); break; }

        case OP_GET_LOCAL: {
            int idx = fp + ins.operand;
            if (idx < 0 || idx >= sp) { printf("GET_LOCAL out of bounds: fp=%d op=%d sp=%d\n", fp, ins.operand, sp); error=1; break; }
            PUSH(stack[idx]); break;
        }
        case OP_SET_LOCAL: {
            int idx = fp + ins.operand;
            if (idx < 0 || idx >= STACK_SIZE) { printf("SET_LOCAL out of bounds\n"); error=1; break; }
            stack[idx] = PEEK(0); sp--; break;
        }
        case OP_GET_UPVALUE: {
            Value cl = stack[fp - 1];
            if (cl.type == VAL_CLOSURE) {
                int uv = ins.operand;
                if (uv < 0 || uv >= 16) { printf("UPVALUE index out of bounds: %d\n", uv); error=1; break; }
                int32_t open_slot = heap[cl.as.ptr].closure.open_slots[uv];
                if (open_slot >= 0) {
                    /* Open upvalue: read from stack slot (sees latest value) */
                    PUSH(stack[open_slot]);
                } else {
                    /* Closed upvalue: read from captured value */
                    PUSH(heap[cl.as.ptr].closure.upvalues[uv]);
                }
            } else PUSH(((Value){.type=VAL_NIL}));
            break;
        }
        case OP_SET_UPVALUE: {
            Value cl = stack[fp - 1];
            if (cl.type == VAL_CLOSURE) {
                int uv = ins.operand;
                if (uv < 0 || uv >= 16) { printf("UPVALUE index out of bounds: %d\n", uv); error=1; break; }
                int32_t open_slot = heap[cl.as.ptr].closure.open_slots[uv];
                if (open_slot >= 0) {
                    stack[open_slot] = PEEK(0);  /* write through to stack */
                } else {
                    heap[cl.as.ptr].closure.upvalues[uv] = PEEK(0);
                }
            }
            sp--; break;
        }

        case OP_CLOSURE: {
            int ci = ins.operand & 0xFFFF;
            int nu = (ins.operand >> 16) & 0xFF;
            if (nu > 16) nu = 16;
            int32_t func_pc = (int32_t)chunk->constants[ci].as.i;
            int32_t ptr = HALLOC();
            if (ptr < 0) break;
            heap[ptr].type = HEAP_CLOSURE;
            heap[ptr].closure.func_pc = func_pc;
            heap[ptr].closure.n_upvalues = nu;
            for (int i = nu - 1; i >= 0; i--) {
                heap[ptr].closure.upvalues[i] = POP();
                heap[ptr].closure.open_slots[i] = -1; /* closed by default */
            }
            PUSH(((Value){.type=VAL_CLOSURE,.as.ptr=ptr}));
            break;
        }

        case OP_CALL: {
            int argc = ins.operand;
            Value func = stack[sp - 1 - argc];
            /* Handle continuation invocation */
            if (func.type == VAL_CONTINUATION) {
                Value result = (argc >= 1) ? stack[sp - 1] : ((Value){.type=VAL_NIL});
                int32_t cc_ptr = func.as.ptr;
                if (heap[cc_ptr].continuation.used) {
                    printf("ERROR: continuation invoked more than once (single-shot)\n");
                    error = 1; break;
                }
                heap[cc_ptr].continuation.used = 1;
                /* Check if we need to unwind dynamic-wind frames */
                int target_wind = heap[cc_ptr].continuation.saved_wind_depth;
                if (wind_depth > target_wind) {
                    /* Need to unwind: save continuation for later, call first after thunk */
                    pending_cc = cc_ptr;
                    pending_cc_result = result;
                    wind_depth--;
                    Value after = wind_stack[wind_depth].after;
                    if (after.type == VAL_CLOSURE) {
                        /* Call after thunk — when it returns, OP_RETURN will check pending_cc */
                        PUSH(after);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = -1;
                        frame_count++;
                        fp = sp;
                        pc = heap[after.as.ptr].closure.func_pc;
                    }
                    break;
                }
                /* No wind unwinding needed — restore directly */
                int save_n = heap[cc_ptr].continuation.saved_sp;
                if (save_n > 256) save_n = 256;
                memcpy(stack, heap[cc_ptr].continuation.saved_stack, save_n * sizeof(Value));
                pc = heap[cc_ptr].continuation.saved_pc;
                sp = heap[cc_ptr].continuation.saved_sp;
                fp = heap[cc_ptr].continuation.saved_fp;
                frame_count = heap[cc_ptr].continuation.saved_frame_count;
                PUSH(result);
                break;
            }
            if (func.type != VAL_CLOSURE) { printf("CALL non-function\n"); error=1; break; }
            if (frame_count >= MAX_FRAMES) { printf("FRAME OVERFLOW\n"); error=1; break; }
            frames[frame_count].return_pc = pc;
            frames[frame_count].return_fp = fp;
            frames[frame_count].heap_mark = heap_next;  /* OALR: save region boundary */
            frames[frame_count].force_promise_ptr = -1;
            frame_count++;
            fp = sp - argc;
            pc = heap[func.as.ptr].closure.func_pc;
            break;
        }

        case OP_TAIL_CALL: {
            int argc = ins.operand;
            Value func = stack[sp - 1 - argc];
            /* Handle continuation invocation (same as OP_CALL path) */
            if (func.type == VAL_CONTINUATION) {
                Value result = (argc >= 1) ? stack[sp - 1] : ((Value){.type=VAL_NIL});
                int32_t cc_ptr = func.as.ptr;
                if (heap[cc_ptr].continuation.used) {
                    printf("ERROR: continuation invoked more than once (single-shot)\n");
                    error = 1; break;
                }
                heap[cc_ptr].continuation.used = 1;
                int target_wind = heap[cc_ptr].continuation.saved_wind_depth;
                if (wind_depth > target_wind) {
                    pending_cc = cc_ptr;
                    pending_cc_result = result;
                    wind_depth--;
                    Value after = wind_stack[wind_depth].after;
                    if (after.type == VAL_CLOSURE) {
                        PUSH(after);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = -1;
                        frame_count++;
                        fp = sp;
                        pc = heap[after.as.ptr].closure.func_pc;
                    }
                    break;
                }
                int save_n = heap[cc_ptr].continuation.saved_sp;
                if (save_n > 256) save_n = 256;
                memcpy(stack, heap[cc_ptr].continuation.saved_stack, save_n * sizeof(Value));
                pc = heap[cc_ptr].continuation.saved_pc;
                sp = heap[cc_ptr].continuation.saved_sp;
                fp = heap[cc_ptr].continuation.saved_fp;
                frame_count = heap[cc_ptr].continuation.saved_frame_count;
                PUSH(result);
                break;
            }
            if (func.type != VAL_CLOSURE) { error=1; break; }
            /* Update the closure reference at fp-1 so GET_UPVALUE works
             * correctly for the callee (not the caller's closure) */
            stack[fp - 1] = func;
            for (int i = 0; i < argc; i++)
                stack[fp + i] = stack[sp - argc + i];
            sp = fp + argc;
            pc = heap[func.as.ptr].closure.func_pc;
            break;
        }

        case OP_RETURN: {
            Value result = POP();
            if (frame_count <= 0) { PUSH(result); halted = 1; break; }
            frame_count--;
            int32_t mark = frames[frame_count].heap_mark;

            /* OALR region cleanup disabled — it's unsafe when side-effecting
             * functions store heap values in persistent structures (hash tables,
             * vectors, set! targets). Full OALR requires escape analysis.
             * The heap is large enough (4M objects) that this isn't a problem. */
            (void)mark;

            /* Check if returning from a force — memoize the result */
            if (frames[frame_count].force_promise_ptr >= 0) {
                int32_t pp = frames[frame_count].force_promise_ptr;
                if (pp < HEAP_SIZE && heap[pp].type == HEAP_VECTOR) {
                    heap[pp].vector.items[0] = (Value){.type=VAL_BOOL,.as.b=1}; /* mark forced */
                    heap[pp].vector.items[1] = result; /* cache the value */
                }
                frames[frame_count].force_promise_ptr = -1;
            }

            sp = fp - 1;
            fp = frames[frame_count].return_fp;
            pc = frames[frame_count].return_pc;
            PUSH(result);

            /* Check for pending continuation (wind unwinding in progress) */
            if (pending_cc >= 0) {
                int target_wind = heap[pending_cc].continuation.saved_wind_depth;
                if (wind_depth > target_wind) {
                    /* More after thunks to call */
                    POP(); /* discard after thunk's return value */
                    wind_depth--;
                    Value after = wind_stack[wind_depth].after;
                    if (after.type == VAL_CLOSURE) {
                        PUSH(after);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = -1;
                        frame_count++;
                        fp = sp;
                        pc = heap[after.as.ptr].closure.func_pc;
                    }
                } else {
                    /* All after thunks done — invoke the pending continuation.
                     * Do NOT memcpy — preserve after thunk mutations.
                     * Only restore sp/fp/frame_count/pc. */
                    POP(); /* discard last after thunk's return value */
                    int32_t cc_ptr = pending_cc;
                    Value cc_result = pending_cc_result;
                    pending_cc = -1;
                    pc = heap[cc_ptr].continuation.saved_pc;
                    sp = heap[cc_ptr].continuation.saved_sp;
                    fp = heap[cc_ptr].continuation.saved_fp;
                    frame_count = heap[cc_ptr].continuation.saved_frame_count;
                    wind_depth = heap[cc_ptr].continuation.saved_wind_depth;
                    PUSH(cc_result);
                }
            }
            break;
        }

        case OP_JUMP:
            if (ins.operand < 0 || ins.operand >= chunk->code_len) { printf("JUMP target out of bounds: %d\n", ins.operand); error=1; break; }
            pc = ins.operand; break;
        case OP_JUMP_IF_FALSE: {
            Value v=POP();
            if(IS_FALSY(v)) {
                if (ins.operand < 0 || ins.operand >= chunk->code_len) { printf("JUMP_IF_FALSE target out of bounds: %d\n", ins.operand); error=1; break; }
                pc=ins.operand;
            }
            break;
        }
        case OP_LOOP:
            if (ins.operand < 0 || ins.operand >= chunk->code_len) { printf("LOOP target out of bounds: %d\n", ins.operand); error=1; break; }
            pc = ins.operand; break;

        case OP_CONS: {
            Value car=POP(), cdr=POP(); /* TOS=car, SOS=cdr */
            int32_t ptr = HALLOC(); if(ptr<0) break;
            heap[ptr].type = HEAP_CONS;
            heap[ptr].cons.car = car;
            heap[ptr].cons.cdr = cdr;
            PUSH(((Value){.type=VAL_PAIR,.as.ptr=ptr}));
            break;
        }
        case OP_CAR: { Value p=POP(); if(p.type!=VAL_PAIR){error=1;break;} PUSH(heap[p.as.ptr].cons.car); break; }
        case OP_CDR: { Value p=POP(); if(p.type!=VAL_PAIR){error=1;break;} PUSH(heap[p.as.ptr].cons.cdr); break; }
        case OP_NULL_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_NIL)})); break; }

        case OP_PRINT: {
            Value v = POP();
            printf("  → ");
            print_value(v, heap, 0, 1);
            printf("\n");
            break;
        }

        case OP_CLOSE_UPVALUE: {
            /* Patch closure's upvalue[operand] = the closure itself (self-reference) */
            Value cl = PEEK(0);
            if (cl.type == VAL_CLOSURE) {
                heap[cl.as.ptr].closure.upvalues[ins.operand] = cl;
            }
            break;
        }

        /* Vectors */
        case OP_VEC_CREATE: {
            int count = ins.operand;
            int32_t ptr = HALLOC(); if (ptr < 0) break;
            heap[ptr].type = HEAP_VECTOR;
            heap[ptr].vector.len = count;
            for (int i = count - 1; i >= 0; i--)
                heap[ptr].vector.items[i] = POP();
            PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
            break;
        }
        case OP_VEC_REF: {
            Value idx_v = POP(), vec_v = POP();
            if (vec_v.type != VAL_VECTOR) { error=1; break; }
            int idx = (int)AS_NUM(idx_v);
            if (idx < 0 || idx >= heap[vec_v.as.ptr].vector.len) { printf("VEC_REF out of bounds\n"); error=1; break; }
            PUSH(heap[vec_v.as.ptr].vector.items[idx]);
            break;
        }
        case OP_VEC_SET: {
            Value val = POP(), idx_v = POP(), vec_v = POP();
            if (vec_v.type != VAL_VECTOR) { error=1; break; }
            int idx = (int)AS_NUM(idx_v);
            if (idx >= 0 && idx < heap[vec_v.as.ptr].vector.len)
                heap[vec_v.as.ptr].vector.items[idx] = val;
            break;
        }
        case OP_VEC_LEN: {
            Value v = POP();
            if (v.type == VAL_VECTOR) PUSH(INT_VAL(heap[v.as.ptr].vector.len));
            else PUSH(INT_VAL(0));
            break;
        }

        /* Strings */
        case OP_STR_REF: {
            Value idx_v = POP(), str_v = POP();
            if (str_v.type != VAL_STRING) { error=1; break; }
            int idx = (int)AS_NUM(idx_v);
            if (idx >= 0 && idx < heap[str_v.as.ptr].string.len)
                PUSH(INT_VAL(heap[str_v.as.ptr].string.data[idx]));
            else PUSH(INT_VAL(0));
            break;
        }
        case OP_STR_LEN: {
            Value v = POP();
            if (v.type == VAL_STRING) PUSH(INT_VAL(heap[v.as.ptr].string.len));
            else PUSH(INT_VAL(0));
            break;
        }

        /* Type predicates */
        case OP_PAIR_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_PAIR)})); break; }
        case OP_NUM_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_INT||v.type==VAL_FLOAT)})); break; }
        case OP_STR_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_STRING)})); break; }
        case OP_BOOL_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_BOOL)})); break; }
        case OP_PROC_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_CLOSURE)})); break; }
        case OP_VEC_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_VECTOR)})); break; }

        /* Set mutations */
        case OP_SET_CAR: { Value val=POP(), p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.car=val; break; }
        case OP_SET_CDR: { Value val=POP(), p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.cdr=val; break; }

        /* call/cc: capture current continuation, call TOS with it */
        case OP_CALLCC: {
            Value proc = POP(); /* the procedure to call with the continuation */
            if (proc.type != VAL_CLOSURE) { printf("CALLCC non-procedure\n"); error=1; break; }
            /* Capture continuation: save pc, sp, fp, frame_count, and stack snapshot */
            int32_t cc_ptr = HALLOC(); if (cc_ptr < 0) break;
            heap[cc_ptr].type = HEAP_CONTINUATION;
            heap[cc_ptr].continuation.saved_pc = pc;
            heap[cc_ptr].continuation.saved_sp = sp;
            heap[cc_ptr].continuation.saved_fp = fp;
            heap[cc_ptr].continuation.saved_frame_count = frame_count;
            heap[cc_ptr].continuation.used = 0;
            heap[cc_ptr].continuation.saved_wind_depth = wind_depth;
            int save_n = sp < 256 ? sp : 256;
            memcpy(heap[cc_ptr].continuation.saved_stack, stack, save_n * sizeof(Value));
            /* Create a closure that invokes the continuation when called */
            Value cc_val = {.type = VAL_CONTINUATION, .as.ptr = cc_ptr};
            /* Call proc with the continuation as argument */
            PUSH(proc); /* function */
            PUSH(cc_val); /* argument = continuation */
            /* Inline CALL 1 */
            if (frame_count >= MAX_FRAMES) { error=1; break; }
            frames[frame_count].return_pc = pc;
            frames[frame_count].return_fp = fp;
            frames[frame_count].heap_mark = heap_next;
            frames[frame_count].force_promise_ptr = -1;
            frame_count++;
            fp = sp - 1;
            pc = heap[proc.as.ptr].closure.func_pc;
            break;
        }

        /* Invoke a continuation with a value */
        case OP_INVOKE_CC: {
            /* Not needed as opcode — continuations are invoked via CALL */
            break;
        }

        /* Exception handling: push handler (save unwind point) */
        case OP_PUSH_HANDLER: {
            if (handler_count >= MAX_HANDLERS) { printf("TOO MANY HANDLERS\n"); error=1; break; }
            exc_handlers[handler_count].handler_pc = ins.operand;
            exc_handlers[handler_count].saved_sp = sp;
            exc_handlers[handler_count].saved_fp = fp;
            exc_handlers[handler_count].saved_frame_count = frame_count;
            handler_count++;
            break;
        }

        /* Exception handling: pop handler (normal guard exit) */
        case OP_POP_HANDLER: {
            if (handler_count > 0) handler_count--;
            break;
        }

        /* Push current exception value (set by most recent raise) */
        case OP_GET_EXN: {
            PUSH(current_exn);
            break;
        }

        /* Wind stack push: store after thunk for dynamic-wind unwinding */
        case OP_WIND_PUSH: {
            Value after = POP();
            if (wind_depth < MAX_WINDS) {
                wind_stack[wind_depth].after = after;
                wind_depth++;
            }
            break;
        }

        /* Wind stack pop */
        case OP_WIND_POP: {
            if (wind_depth > 0) wind_depth--;
            break;
        }

        /* Pack rest arguments into a list.
         * operand = n_fixed params. Args from fp+n_fixed to sp-1 become a list.
         * After packing: sp = fp + n_fixed + 1, stack[fp+n_fixed] = rest list. */
        case OP_PACK_REST: {
            int n_fixed = ins.operand;
            int rest_start = fp + n_fixed;
            int rest_count = sp - rest_start;
            /* Build list from the rest args (right to left) */
            Value rest_list = {.type = VAL_NIL};
            for (int ri = sp - 1; ri >= rest_start; ri--) {
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_CONS;
                heap[ptr].cons.car = stack[ri];
                heap[ptr].cons.cdr = rest_list;
                rest_list = (Value){.type = VAL_PAIR, .as.ptr = ptr};
            }
            /* Store rest list at fp+n_fixed, set sp */
            stack[rest_start] = rest_list;
            sp = rest_start + 1;
            break;
        }

        /* Open closure: like CLOSURE but upvalues are stack slot references (not captured values).
         * The stack has [slot_idx_0, slot_idx_1, ...] pushed as INT values.
         * Each slot index tells GET_UPVALUE where to read from the ENCLOSING stack. */
        case OP_OPEN_CLOSURE: {
            int ci = ins.operand & 0xFFFF;
            int nu = (ins.operand >> 16) & 0xFF;
            int32_t func_pc = (int32_t)chunk->constants[ci].as.i;
            int32_t ptr = HALLOC();
            if (ptr < 0) break;
            heap[ptr].type = HEAP_CLOSURE;
            heap[ptr].closure.func_pc = func_pc;
            heap[ptr].closure.n_upvalues = nu;
            for (int i = nu - 1; i >= 0; i--) {
                Value slot_val = POP();
                int32_t slot_idx = (int32_t)slot_val.as.i;
                heap[ptr].closure.upvalues[i] = ((Value){.type=VAL_NIL}); /* placeholder */
                heap[ptr].closure.open_slots[i] = slot_idx; /* open reference to stack slot */
            }
            PUSH(((Value){.type=VAL_CLOSURE,.as.ptr=ptr}));
            break;
        }

        /* Scope cleanup: pop N values below TOS, keeping TOS */
        case OP_POPN: {
            int n = ins.operand;
            if (n > 0 && sp > n) {
                Value result = stack[sp - 1]; /* save TOS */
                sp -= n;                       /* remove N values */
                stack[sp - 1] = result;        /* put result back at new TOS */
            }
            break;
        }

        /* Native call registry */
        case OP_NATIVE_CALL: {
            int fid = ins.operand;
            switch (fid) {
            /* Math functions (single arg on stack) */
            case 20: { Value a=POP(); PUSH(FLOAT_VAL(sin(AS_NUM(a)))); break; }   /* sin */
            case 21: { Value a=POP(); PUSH(FLOAT_VAL(cos(AS_NUM(a)))); break; }   /* cos */
            case 22: { Value a=POP(); PUSH(FLOAT_VAL(tan(AS_NUM(a)))); break; }   /* tan */
            case 23: { Value a=POP(); PUSH(FLOAT_VAL(exp(AS_NUM(a)))); break; }   /* exp */
            case 24: { Value a=POP(); PUSH(FLOAT_VAL(log(AS_NUM(a)))); break; }   /* log */
            case 25: { Value a=POP(); PUSH(FLOAT_VAL(sqrt(AS_NUM(a)))); break; }  /* sqrt */
            case 26: { Value a=POP(); PUSH(NUM_VAL(floor(AS_NUM(a)))); break; }   /* floor */
            case 27: { Value a=POP(); PUSH(NUM_VAL(ceil(AS_NUM(a)))); break; }    /* ceiling */
            case 28: { Value a=POP(); PUSH(NUM_VAL(round(AS_NUM(a)))); break; }   /* round */
            case 29: { Value a=POP(); PUSH(FLOAT_VAL(asin(AS_NUM(a)))); break; }  /* asin */
            case 30: { Value a=POP(); PUSH(FLOAT_VAL(acos(AS_NUM(a)))); break; }  /* acos */
            case 31: { Value a=POP(); PUSH(FLOAT_VAL(atan(AS_NUM(a)))); break; }  /* atan */
            /* Two-arg math */
            case 32: { Value b=POP(),a=POP(); PUSH(FLOAT_VAL(pow(AS_NUM(a),AS_NUM(b)))); break; } /* expt */
            case 33: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmin(AS_NUM(a),AS_NUM(b)))); break; }  /* min */
            case 34: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmax(AS_NUM(a),AS_NUM(b)))); break; }  /* max */
            case 35: { Value a=POP(); PUSH(NUM_VAL(fabs(AS_NUM(a)))); break; }  /* abs */
            case 36: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmod(AS_NUM(a),AS_NUM(b)))); break; }  /* modulo */
            case 37: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmod(AS_NUM(a),AS_NUM(b)))); break; }  /* remainder */
            case 38: { Value b=POP(),a=POP(); PUSH(NUM_VAL(floor(AS_NUM(a)/AS_NUM(b)))); break; } /* quotient */
            /* Numeric predicates */
            case 40: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)>0)})); break; }  /* positive? */
            case 41: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)<0)})); break; }  /* negative? */
            case 42: { Value a=POP(); double v=AS_NUM(a); PUSH(((Value){.type=VAL_BOOL,.as.b=((int64_t)v%2!=0)})); break; } /* odd? */
            case 43: { Value a=POP(); double v=AS_NUM(a); PUSH(((Value){.type=VAL_BOOL,.as.b=((int64_t)v%2==0)})); break; } /* even? */
            case 44: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)==0)})); break; }  /* zero? */
            case 45: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_NIL)})); break; } /* null? */
            case 46: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_PAIR)})); break; } /* pair? */
            case 47: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_INT||a.type==VAL_FLOAT)})); break; } /* number? */
            case 48: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_BOOL)})); break; } /* boolean? */
            case 49: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_CLOSURE)})); break; } /* procedure? */
            case 50: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_VECTOR)})); break; } /* vector? */
            /* String operations */
            /* native 50 is vector? (defined above) */
            /* string-length is now native 56 */
            case 51: { /* number->string */
                Value v=POP();
                int32_t ptr = HALLOC(); if(ptr<0) break;
                heap[ptr].type = HEAP_STRING;
                if (v.type==VAL_INT) heap[ptr].string.len = snprintf(heap[ptr].string.data, 255, "%lld", (long long)v.as.i);
                else heap[ptr].string.len = snprintf(heap[ptr].string.data, 255, "%.6g", v.as.f);
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            /* Display with newline */
            case 53: { /* build-string: stack has [..., len, c0, c1, ..., cN-1]
                       * len was pushed first, then N char values (as numbers).
                       * TOS = cN-1, below that c0..cN-2, below those = len.
                       * We peek at len to know how many chars to pop. */
                /* Scan for the length value below the chars on the stack.
                 * len is at position sp - N - 1 where N = len value itself. */
                int slen = 0;
                for (int try_len = 0; try_len < 256; try_len++) {
                    int len_pos = sp - try_len - 1;
                    if (len_pos >= 0 && stack[len_pos].type == VAL_INT && stack[len_pos].as.i == try_len) {
                        slen = try_len;
                        break;
                    }
                }
                if (slen > 255) slen = 255;
                /* Pop chars in reverse (TOS is last char) */
                char buf[256];
                for (int i = slen - 1; i >= 0; i--) {
                    Value ch = POP();
                    buf[i] = (char)(int)AS_NUM(ch);
                }
                POP(); /* pop len */
                buf[slen] = 0;
                int32_t sptr = HALLOC(); if (sptr < 0) break;
                heap[sptr].type = HEAP_STRING;
                heap[sptr].string.len = slen;
                memcpy(heap[sptr].string.data, buf, slen + 1);
                PUSH(((Value){.type=VAL_STRING,.as.ptr=sptr}));
                break;
            }
            case 54: { /* string-append: pop 2 strings, concat */
                Value b = POP(), a = POP();
                int32_t sptr = HALLOC(); if (sptr < 0) break;
                heap[sptr].type = HEAP_STRING;
                int la = (a.type==VAL_STRING) ? heap[a.as.ptr].string.len : 0;
                int lb = (b.type==VAL_STRING) ? heap[b.as.ptr].string.len : 0;
                if (la + lb > 255) { la = (la > 255) ? 255 : la; lb = 255 - la; }
                heap[sptr].string.len = la + lb;
                if (a.type==VAL_STRING) memcpy(heap[sptr].string.data, heap[a.as.ptr].string.data, la);
                if (b.type==VAL_STRING) memcpy(heap[sptr].string.data + la, heap[b.as.ptr].string.data, lb);
                heap[sptr].string.data[la+lb] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=sptr}));
                break;
            }
            case 55: { /* string=? */
                Value b = POP(), a = POP();
                int eq = 0;
                if (a.type==VAL_STRING && b.type==VAL_STRING) {
                    eq = (heap[a.as.ptr].string.len == heap[b.as.ptr].string.len &&
                          memcmp(heap[a.as.ptr].string.data, heap[b.as.ptr].string.data, heap[a.as.ptr].string.len) == 0);
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=eq}));
                break;
            }
            case 60: { /* newline */
                printf("\n"); break;
            }
            case 70: { /* apply: f args-list → call f with unpacked args */
                Value args_list = POP(), func = POP();
                if (func.type != VAL_CLOSURE) { printf("APPLY non-function\n"); error=1; break; }
                /* Count args */
                int argc = 0;
                Value tmp = args_list;
                while (tmp.type == VAL_PAIR) { argc++; tmp = heap[tmp.as.ptr].cons.cdr; }
                /* Push function below args */
                PUSH(func);
                /* Unpack args onto stack */
                tmp = args_list;
                while (tmp.type == VAL_PAIR) {
                    PUSH(heap[tmp.as.ptr].cons.car);
                    tmp = heap[tmp.as.ptr].cons.cdr;
                }
                /* Now stack: [..., func, arg0, arg1, ...]. Call. */
                if (frame_count >= MAX_FRAMES) { error=1; break; }
                frames[frame_count].return_pc = pc;
                frames[frame_count].return_fp = fp;
                frames[frame_count].heap_mark = heap_next;
                frames[frame_count].force_promise_ptr = -1;
                frame_count++;
                fp = sp - argc;
                pc = heap[func.as.ptr].closure.func_pc;
                break;
            }
            case 56: { /* string-length */
                Value v=POP();
                if (v.type==VAL_STRING) PUSH(INT_VAL(heap[v.as.ptr].string.len));
                else PUSH(INT_VAL(0));
                break;
            }
            case 57: { /* string-ref */
                Value idx=POP(), s=POP();
                if (s.type==VAL_STRING) {
                    int i=(int)AS_NUM(idx);
                    if(i>=0&&i<heap[s.as.ptr].string.len) PUSH(INT_VAL(heap[s.as.ptr].string.data[i]));
                    else PUSH(INT_VAL(0));
                } else PUSH(INT_VAL(0));
                break;
            }
            case 71: { /* length: list → int */
                Value lst = POP();
                int len = 0;
                while (lst.type == VAL_PAIR) { len++; lst = heap[lst.as.ptr].cons.cdr; }
                PUSH(INT_VAL(len));
                break;
            }
            case 72: { /* car */
                Value p=POP();
                if(p.type==VAL_PAIR) PUSH(heap[p.as.ptr].cons.car);
                else { printf("CAR non-pair\n"); PUSH(((Value){.type=VAL_NIL})); }
                break;
            }
            case 73: { /* cdr */
                Value p=POP();
                if(p.type==VAL_PAIR) PUSH(heap[p.as.ptr].cons.cdr);
                else { printf("CDR non-pair\n"); PUSH(((Value){.type=VAL_NIL})); }
                break;
            }
            case 74: { /* cons */
                Value cdr_v=POP(), car_v=POP();
                int32_t ptr=HALLOC(); if(ptr<0) break;
                heap[ptr].type=HEAP_CONS;
                heap[ptr].cons.car=car_v; heap[ptr].cons.cdr=cdr_v;
                PUSH(((Value){.type=VAL_PAIR,.as.ptr=ptr}));
                break;
            }
            case 75: { /* set-car! */ Value v=POP(),p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.car=v; break; }
            case 76: { /* set-cdr! */ Value v=POP(),p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.cdr=v; break; }
            case 77: { /* cadr */ Value p=POP(); if(p.type==VAL_PAIR) { Value c=heap[p.as.ptr].cons.cdr; if(c.type==VAL_PAIR) PUSH(heap[c.as.ptr].cons.car); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 78: { /* cddr */ Value p=POP(); if(p.type==VAL_PAIR) { Value c=heap[p.as.ptr].cons.cdr; if(c.type==VAL_PAIR) PUSH(heap[c.as.ptr].cons.cdr); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 79: { /* caar */ Value p=POP(); if(p.type==VAL_PAIR) { Value c=heap[p.as.ptr].cons.car; if(c.type==VAL_PAIR) PUSH(heap[c.as.ptr].cons.car); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 80: { /* caddr */ Value p=POP(); if(p.type==VAL_PAIR) { Value c1=heap[p.as.ptr].cons.cdr; if(c1.type==VAL_PAIR) { Value c2=heap[c1.as.ptr].cons.cdr; if(c2.type==VAL_PAIR) PUSH(heap[c2.as.ptr].cons.car); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 81: { /* vector-ref */ Value idx=POP(),v=POP(); if(v.type==VAL_VECTOR){int i=(int)AS_NUM(idx); if(i>=0&&i<heap[v.as.ptr].vector.len) PUSH(heap[v.as.ptr].vector.items[i]); else PUSH(((Value){.type=VAL_NIL}));} else PUSH(((Value){.type=VAL_NIL})); break; }
            case 82: { /* vector-length */ Value v=POP(); if(v.type==VAL_VECTOR) PUSH(INT_VAL(heap[v.as.ptr].vector.len)); else PUSH(INT_VAL(0)); break; }
            case 83: { /* vector-set! */ Value val=POP(),idx=POP(),v=POP(); if(v.type==VAL_VECTOR){int i=(int)AS_NUM(idx); if(i>=0&&i<heap[v.as.ptr].vector.len) heap[v.as.ptr].vector.items[i]=val;} break; }
            case 90: /* display — same as PRINT but without "→ " prefix */
            case 91: { /* write */
                Value v = POP();
                if (v.type==VAL_INT) printf("%lld", (long long)v.as.i);
                else if (v.type==VAL_FLOAT) printf("%.6g", v.as.f);
                else if (v.type==VAL_BOOL) printf("%s", v.as.b?"#t":"#f");
                else if (v.type==VAL_NIL) printf("()");
                else if (v.type==VAL_PAIR) {
                    printf("("); Value cur=v; int f=1;
                    while(cur.type==VAL_PAIR){if(!f)printf(" ");f=0;
                    Value car=heap[cur.as.ptr].cons.car;
                    if(car.type==VAL_INT)printf("%lld",(long long)car.as.i);
                    else if(car.type==VAL_FLOAT)printf("%.6g",car.as.f);
                    else if(car.type==VAL_BOOL)printf("%s",car.as.b?"#t":"#f");
                    else if(car.type==VAL_NIL)printf("()");
                    else printf("<v>");
                    cur=heap[cur.as.ptr].cons.cdr;}
                    if(cur.type!=VAL_NIL){printf(" . ");if(cur.type==VAL_INT)printf("%lld",(long long)cur.as.i);else printf("<v>");}
                    printf(")");
                }
                else if (v.type==VAL_CLOSURE) printf("<procedure>");
                else if (v.type==VAL_STRING) printf("%.*s", heap[v.as.ptr].string.len, heap[v.as.ptr].string.data);
                else if (v.type==VAL_VECTOR) {
                    printf("#("); for(int vi=0;vi<heap[v.as.ptr].vector.len;vi++){if(vi)printf(" ");Value it=heap[v.as.ptr].vector.items[vi];if(it.type==VAL_INT)printf("%lld",(long long)it.as.i);else if(it.type==VAL_FLOAT)printf("%.6g",it.as.f);else printf("<v>");} printf(")");
                }
                printf("\n");
                PUSH(((Value){.type=VAL_NIL})); /* display returns void */
                break;
            }
            case 130: { /* raise: throw exception value to nearest handler */
                Value exn = POP();
                if (handler_count <= 0) {
                    printf("ERROR: unhandled exception: ");
                    if (exn.type == VAL_INT) printf("%lld", (long long)exn.as.i);
                    else if (exn.type == VAL_FLOAT) printf("%.6g", exn.as.f);
                    else if (exn.type == VAL_STRING) printf("%.*s", heap[exn.as.ptr].string.len, heap[exn.as.ptr].string.data);
                    else if (exn.type == VAL_BOOL) printf("%s", exn.as.b ? "#t" : "#f");
                    else printf("<value>");
                    printf("\n");
                    error = 1; break;
                }
                handler_count--;
                /* Unwind to handler point — do NOT restore stack values.
                 * This preserves set! side effects (R7RS semantics).
                 * Only restore sp/fp/frame_count/pc. */
                sp = exc_handlers[handler_count].saved_sp;
                fp = exc_handlers[handler_count].saved_fp;
                frame_count = exc_handlers[handler_count].saved_frame_count;
                pc = exc_handlers[handler_count].handler_pc;
                /* Store exception in VM register — handler accesses via OP_GET_EXN */
                current_exn = exn;
                break;
            }
            case 133: { /* eq?: identity equality (pointer for heap types, value for scalars) */
                Value b=POP(), a=POP();
                int result = 0;
                if (a.type != b.type) result = 0;
                else if (a.type == VAL_NIL) result = 1;
                else if (a.type == VAL_BOOL) result = (a.as.b == b.as.b);
                else if (a.type == VAL_INT) result = (a.as.i == b.as.i);
                else if (a.type == VAL_FLOAT) result = (a.as.f == b.as.f);
                else result = (a.as.ptr == b.as.ptr); /* pointer equality for heap types */
                PUSH(((Value){.type=VAL_BOOL,.as.b=result}));
                break;
            }
            case 134: { /* equal?: deep structural equality */
                Value b=POP(), a=POP();
                /* Iterative deep comparison using a work stack */
                Value work_a[64], work_b[64]; int wn = 0;
                work_a[wn] = a; work_b[wn] = b; wn++;
                int result = 1;
                while (wn > 0 && result) {
                    wn--;
                    Value x = work_a[wn], y = work_b[wn];
                    if (x.type != y.type) { result = 0; break; }
                    if (x.type == VAL_NIL) continue;
                    else if (x.type == VAL_BOOL) { if (x.as.b != y.as.b) result = 0; }
                    else if (x.type == VAL_INT) { if (x.as.i != y.as.i) result = 0; }
                    else if (x.type == VAL_FLOAT) { if (x.as.f != y.as.f) result = 0; }
                    else if (x.type == VAL_STRING) {
                        if (heap[x.as.ptr].string.len != heap[y.as.ptr].string.len) result = 0;
                        else if (memcmp(heap[x.as.ptr].string.data, heap[y.as.ptr].string.data,
                                        heap[x.as.ptr].string.len) != 0) result = 0;
                    }
                    else if (x.type == VAL_PAIR) {
                        if (wn + 2 > 64) { result = (x.as.ptr == y.as.ptr); } /* overflow: fallback */
                        else {
                            work_a[wn] = heap[x.as.ptr].cons.cdr; work_b[wn] = heap[y.as.ptr].cons.cdr; wn++;
                            work_a[wn] = heap[x.as.ptr].cons.car; work_b[wn] = heap[y.as.ptr].cons.car; wn++;
                        }
                    }
                    else if (x.type == VAL_VECTOR) {
                        int la = heap[x.as.ptr].vector.len, lb = heap[y.as.ptr].vector.len;
                        if (la != lb) result = 0;
                        else { for (int vi=0; vi < la && wn < 64; vi++) {
                            work_a[wn] = heap[x.as.ptr].vector.items[vi];
                            work_b[wn] = heap[y.as.ptr].vector.items[vi]; wn++;
                        }}
                    }
                    else result = (x.as.ptr == y.as.ptr);
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=result}));
                break;
            }
            case 135: { /* append: append two lists */
                Value b=POP(), a=POP();
                if (a.type == VAL_NIL) { PUSH(b); break; }
                if (a.type != VAL_PAIR) { PUSH(b); break; }
                /* Copy list a, set last cdr to b */
                Value head_v = {.type=VAL_NIL}, tail_v = {.type=VAL_NIL};
                Value cur = a;
                while (cur.type == VAL_PAIR) {
                    int32_t ptr = HALLOC(); if (ptr < 0) goto append_done;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = heap[cur.as.ptr].cons.car;
                    heap[ptr].cons.cdr = (Value){.type=VAL_NIL};
                    Value node_v = {.type=VAL_PAIR,.as.ptr=ptr};
                    if (head_v.type == VAL_NIL) head_v = node_v;
                    else heap[tail_v.as.ptr].cons.cdr = node_v;
                    tail_v = node_v;
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                if (tail_v.type == VAL_PAIR) heap[tail_v.as.ptr].cons.cdr = b;
                append_done:
                PUSH(head_v);
                break;
            }
            case 136: { /* reverse: reverse a list */
                Value lst = POP();
                Value result_v = {.type=VAL_NIL};
                while (lst.type == VAL_PAIR) {
                    int32_t ptr = HALLOC(); if (ptr < 0) break;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = heap[lst.as.ptr].cons.car;
                    heap[ptr].cons.cdr = result_v;
                    result_v = (Value){.type=VAL_PAIR,.as.ptr=ptr};
                    lst = heap[lst.as.ptr].cons.cdr;
                }
                PUSH(result_v);
                break;
            }
            case 137: { /* member: (member val lst) → sublist starting at val, or #f */
                Value lst=POP(), val=POP();
                Value cur = lst;
                while (cur.type == VAL_PAIR) {
                    Value car = heap[cur.as.ptr].cons.car;
                    /* Use equal? semantics: numeric or pointer equality */
                    int match = 0;
                    if (car.type == val.type) {
                        if (car.type == VAL_INT) match = (car.as.i == val.as.i);
                        else if (car.type == VAL_FLOAT) match = (car.as.f == val.as.f);
                        else if (car.type == VAL_BOOL) match = (car.as.b == val.as.b);
                        else if (car.type == VAL_NIL) match = 1;
                        else match = (car.as.ptr == val.as.ptr);
                    }
                    if (match) { PUSH(cur); goto member_done; }
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                member_done: break;
            }
            case 138: { /* assoc: (assoc key alist) → pair or #f */
                Value alist=POP(), key=POP();
                Value cur = alist;
                while (cur.type == VAL_PAIR) {
                    Value pair = heap[cur.as.ptr].cons.car;
                    if (pair.type == VAL_PAIR) {
                        Value k = heap[pair.as.ptr].cons.car;
                        int match = 0;
                        if (k.type == key.type) {
                            if (k.type == VAL_INT) match = (k.as.i == key.as.i);
                            else if (k.type == VAL_FLOAT) match = (k.as.f == key.as.f);
                            else match = (k.as.ptr == key.as.ptr);
                        }
                        if (match) { PUSH(pair); goto assoc_done; }
                    }
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                assoc_done: break;
            }
            case 139: { /* list->vector */
                Value lst=POP();
                int len = 0; Value tmp = lst;
                while (tmp.type == VAL_PAIR) { len++; tmp = heap[tmp.as.ptr].cons.cdr; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = len < 64 ? len : 64;
                tmp = lst;
                for (int vi=0; vi < len && vi < 64; vi++) {
                    heap[ptr].vector.items[vi] = heap[tmp.as.ptr].cons.car;
                    tmp = heap[tmp.as.ptr].cons.cdr;
                }
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 140: { /* vector->list */
                Value v=POP();
                if (v.type != VAL_VECTOR) { PUSH(((Value){.type=VAL_NIL})); break; }
                Value result_v = {.type=VAL_NIL};
                for (int vi = heap[v.as.ptr].vector.len - 1; vi >= 0; vi--) {
                    int32_t ptr = HALLOC(); if (ptr < 0) break;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = heap[v.as.ptr].vector.items[vi];
                    heap[ptr].cons.cdr = result_v;
                    result_v = (Value){.type=VAL_PAIR,.as.ptr=ptr};
                }
                PUSH(result_v);
                break;
            }
            case 142: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)+AS_NUM(b))); break; } /* + */
            case 143: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)-AS_NUM(b))); break; } /* - */
            case 144: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)*AS_NUM(b))); break; } /* * */
            case 145: { Value b=POP(),a=POP(); double bv=AS_NUM(b); PUSH(NUM_VAL(AS_NUM(a)/bv)); break; } /* / */
            case 141: { /* iota: (iota n) → (0 1 2 ... n-1) */
                Value n_v=POP();
                int n = (int)AS_NUM(n_v);
                Value result_v = {.type=VAL_NIL};
                for (int i = n-1; i >= 0; i--) {
                    int32_t ptr = HALLOC(); if (ptr < 0) break;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = INT_VAL(i);
                    heap[ptr].cons.cdr = result_v;
                    result_v = (Value){.type=VAL_PAIR,.as.ptr=ptr};
                }
                PUSH(result_v);
                break;
            }
            /* === Additional predicates === */
            case 160: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_STRING)})); break; } /* symbol? (strings are symbols in our VM) */
            case 161: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_INT && v.as.i >= 0 && v.as.i < 128)})); break; } /* char? */
            case 162: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_INT)})); break; } /* exact? */
            case 163: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_FLOAT)})); break; } /* inexact? */
            case 164: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_FLOAT && isnan(v.as.f))})); break; } /* nan? */
            case 165: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_FLOAT && isinf(v.as.f))})); break; } /* infinite? */
            case 166: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((v.type==VAL_INT) || (v.type==VAL_FLOAT && isfinite(v.as.f)))})); break; } /* finite? */

            /* === String operations === */
            case 170: { /* substring: str, start, end → new string */
                Value end_v=POP(), start_v=POP(), s=POP();
                int start = (int)AS_NUM(start_v), end = (int)AS_NUM(end_v);
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int slen = heap[s.as.ptr].string.len;
                if (start < 0) start = 0; if (end > slen) end = slen;
                int nlen = end - start; if (nlen < 0) nlen = 0; if (nlen > 255) nlen = 255;
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                heap[ptr].string.len = nlen;
                memcpy(heap[ptr].string.data, heap[s.as.ptr].string.data + start, nlen);
                heap[ptr].string.data[nlen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 171: { /* string-contains: str, substr → index or #f */
                Value sub=POP(), s=POP();
                if (s.type != VAL_STRING || sub.type != VAL_STRING) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                char* found = strstr(heap[s.as.ptr].string.data, heap[sub.as.ptr].string.data);
                if (found) PUSH(INT_VAL(found - heap[s.as.ptr].string.data));
                else PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                break;
            }
            case 172: { /* string-upcase */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int slen = heap[s.as.ptr].string.len; if (slen > 255) slen = 255;
                heap[ptr].string.len = slen;
                for (int i = 0; i < slen; i++) heap[ptr].string.data[i] = toupper(heap[s.as.ptr].string.data[i]);
                heap[ptr].string.data[slen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 173: { /* string-downcase */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int slen = heap[s.as.ptr].string.len; if (slen > 255) slen = 255;
                heap[ptr].string.len = slen;
                for (int i = 0; i < slen; i++) heap[ptr].string.data[i] = tolower(heap[s.as.ptr].string.data[i]);
                heap[ptr].string.data[slen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 174: { /* string-reverse */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int slen = heap[s.as.ptr].string.len; if (slen > 255) slen = 255;
                heap[ptr].string.len = slen;
                for (int i = 0; i < slen; i++) heap[ptr].string.data[i] = heap[s.as.ptr].string.data[slen - 1 - i];
                heap[ptr].string.data[slen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 175: { /* string->number */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                char* endp;
                double v = strtod(heap[s.as.ptr].string.data, &endp);
                if (endp == heap[s.as.ptr].string.data) PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                else PUSH(NUM_VAL(v));
                break;
            }
            case 176: { /* string->list: convert string to list of char codes */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(((Value){.type=VAL_NIL})); break; }
                Value result = {.type=VAL_NIL};
                for (int i = heap[s.as.ptr].string.len - 1; i >= 0; i--) {
                    int32_t p = HALLOC(); if (p < 0) break;
                    heap[p].type = HEAP_CONS;
                    heap[p].cons.car = INT_VAL((unsigned char)heap[s.as.ptr].string.data[i]);
                    heap[p].cons.cdr = result;
                    result = (Value){.type=VAL_PAIR,.as.ptr=p};
                }
                PUSH(result); break;
            }
            case 177: { /* list->string: convert list of char codes to string */
                Value lst=POP();
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int len = 0; Value cur = lst;
                while (cur.type == VAL_PAIR && len < 255) {
                    Value ch = heap[cur.as.ptr].cons.car;
                    heap[ptr].string.data[len++] = (char)(int)AS_NUM(ch);
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                heap[ptr].string.len = len;
                heap[ptr].string.data[len] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 178: { /* string-copy */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                heap[ptr].string.len = heap[s.as.ptr].string.len;
                memcpy(heap[ptr].string.data, heap[s.as.ptr].string.data, heap[s.as.ptr].string.len + 1);
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 180: { Value v=POP(); PUSH(FLOAT_VAL(AS_NUM(v))); break; } /* exact->inexact */
            case 181: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* inexact->exact */
            case 182: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* char->integer */
            case 183: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* integer->char */
            case 184: { /* symbol->string (symbols ARE strings in our VM, just return) */
                /* Top of stack is the symbol (string). Just leave it. */
                break;
            }
            case 185: { /* string->symbol: symbols ARE strings in this VM, passthrough */ break; }
            case 186: { /* list-ref: lst, index → element */
                Value idx_v=POP(), lst=POP();
                int idx = (int)AS_NUM(idx_v);
                while (lst.type == VAL_PAIR && idx > 0) { lst = heap[lst.as.ptr].cons.cdr; idx--; }
                if (lst.type == VAL_PAIR) PUSH(heap[lst.as.ptr].cons.car);
                else PUSH(((Value){.type=VAL_NIL}));
                break;
            }
            case 187: { /* list-tail: lst, k → sublist */
                Value k_v=POP(), lst=POP();
                int k = (int)AS_NUM(k_v);
                while (lst.type == VAL_PAIR && k > 0) { lst = heap[lst.as.ptr].cons.cdr; k--; }
                PUSH(lst); break;
            }
            case 188: { /* last-pair */
                Value lst=POP();
                if (lst.type != VAL_PAIR) { PUSH(lst); break; }
                while (heap[lst.as.ptr].cons.cdr.type == VAL_PAIR)
                    lst = heap[lst.as.ptr].cons.cdr;
                PUSH(lst); break;
            }
            case 189: { /* list? */
                Value lst=POP();
                int is_list = 1;
                Value cur = lst;
                int count = 0;
                while (cur.type == VAL_PAIR && count < 10000) { cur = heap[cur.as.ptr].cons.cdr; count++; }
                is_list = (cur.type == VAL_NIL);
                PUSH(((Value){.type=VAL_BOOL,.as.b=is_list})); break;
            }
            case 190: { Value v=POP(); PUSH(NUM_VAL(trunc(AS_NUM(v)))); break; } /* truncate */
            case 191: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* exact */
            case 192: { Value v=POP(); PUSH(FLOAT_VAL(AS_NUM(v))); break; } /* inexact */

            /* === Hash tables === */
            case 200: { /* make-hash-table */
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_HASH;
                heap[ptr].hash.count = 0;
                PUSH(((Value){.type=VAL_HASH,.as.ptr=ptr}));
                break;
            }
            case 201: { /* hash-ref: hash key → value or #f */
                Value key=POP(), h=POP();
                if (h.type != VAL_HASH) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                int found_it = 0;
                for (int hi = 0; hi < heap[h.as.ptr].hash.count; hi++) {
                    Value k = heap[h.as.ptr].hash.keys[hi];
                    int match = 0;
                    if (k.type == key.type) {
                        if (k.type == VAL_INT) match = (k.as.i == key.as.i);
                        else if (k.type == VAL_FLOAT) match = (k.as.f == key.as.f);
                        else if (k.type == VAL_STRING) match = (heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0);
                        else match = (k.as.ptr == key.as.ptr);
                    }
                    if (match) { PUSH(heap[h.as.ptr].hash.vals[hi]); found_it = 1; break; }
                }
                if (!found_it) PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                break;
            }
            case 202: { /* hash-set!: hash key value → void */
                Value val=POP(), key=POP(), h=POP();
                if (h.type != VAL_HASH) break;
                /* Check if key exists */
                int found_it = 0;
                for (int hi = 0; hi < heap[h.as.ptr].hash.count; hi++) {
                    Value k = heap[h.as.ptr].hash.keys[hi];
                    int match = 0;
                    if (k.type == key.type) {
                        if (k.type == VAL_INT) match = (k.as.i == key.as.i);
                        else if (k.type == VAL_FLOAT) match = (k.as.f == key.as.f);
                        else if (k.type == VAL_STRING) match = (heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0);
                        else match = (k.as.ptr == key.as.ptr);
                    }
                    if (match) { heap[h.as.ptr].hash.vals[hi] = val; found_it = 1; break; }
                }
                if (!found_it && heap[h.as.ptr].hash.count < 32) {
                    int idx = heap[h.as.ptr].hash.count++;
                    heap[h.as.ptr].hash.keys[idx] = key;
                    heap[h.as.ptr].hash.vals[idx] = val;
                }
                PUSH(((Value){.type=VAL_NIL}));
                break;
            }
            case 203: { /* hash-has-key? */
                Value key=POP(), h=POP();
                if (h.type != VAL_HASH) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                int found_it = 0;
                for (int hi = 0; hi < heap[h.as.ptr].hash.count && !found_it; hi++) {
                    Value k = heap[h.as.ptr].hash.keys[hi];
                    if (k.type == key.type) {
                        if (k.type == VAL_INT) found_it = (k.as.i == key.as.i);
                        else if (k.type == VAL_STRING) found_it = (heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0);
                        else found_it = (k.as.ptr == key.as.ptr);
                    }
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=found_it})); break;
            }
            case 204: { /* hash-keys → list of keys */
                Value h=POP();
                Value result = {.type=VAL_NIL};
                if (h.type == VAL_HASH) {
                    for (int hi = heap[h.as.ptr].hash.count - 1; hi >= 0; hi--) {
                        int32_t p = HALLOC(); if (p < 0) break;
                        heap[p].type = HEAP_CONS; heap[p].cons.car = heap[h.as.ptr].hash.keys[hi]; heap[p].cons.cdr = result;
                        result = (Value){.type=VAL_PAIR,.as.ptr=p};
                    }
                }
                PUSH(result); break;
            }
            case 205: { /* hash-values → list of values */
                Value h=POP();
                Value result = {.type=VAL_NIL};
                if (h.type == VAL_HASH) {
                    for (int hi = heap[h.as.ptr].hash.count - 1; hi >= 0; hi--) {
                        int32_t p = HALLOC(); if (p < 0) break;
                        heap[p].type = HEAP_CONS; heap[p].cons.car = heap[h.as.ptr].hash.vals[hi]; heap[p].cons.cdr = result;
                        result = (Value){.type=VAL_PAIR,.as.ptr=p};
                    }
                }
                PUSH(result); break;
            }
            case 206: { /* hash-count */
                Value h=POP();
                PUSH(INT_VAL(h.type == VAL_HASH ? heap[h.as.ptr].hash.count : 0)); break;
            }
            case 207: { /* hash-delete! */
                Value key=POP(), h=POP();
                if (h.type == VAL_HASH) {
                    for (int hi = 0; hi < heap[h.as.ptr].hash.count; hi++) {
                        Value k = heap[h.as.ptr].hash.keys[hi];
                        int match = (k.type == key.type && ((k.type == VAL_INT && k.as.i == key.as.i) || (k.type == VAL_STRING && heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0)));
                        if (match) {
                            if (heap[h.as.ptr].hash.count > 0) {
                                heap[h.as.ptr].hash.count--;
                                heap[h.as.ptr].hash.keys[hi] = heap[h.as.ptr].hash.keys[heap[h.as.ptr].hash.count];
                                heap[h.as.ptr].hash.vals[hi] = heap[h.as.ptr].hash.vals[heap[h.as.ptr].hash.count];
                            }
                            break;
                        }
                    }
                }
                PUSH(((Value){.type=VAL_NIL})); break;
            }

            /* === Characters === */
            case 210: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isalpha(c2)})); break; } /* char-alphabetic? */
            case 211: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isdigit(c2)})); break; } /* char-numeric? */
            case 212: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isspace(c2)})); break; } /* char-whitespace? */
            case 213: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isupper(c2)})); break; } /* char-upper-case? */
            case 214: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=islower(c2)})); break; } /* char-lower-case? */
            case 215: { Value v=POP(); PUSH(INT_VAL(toupper((int)AS_NUM(v)))); break; } /* char-upcase */
            case 216: { Value v=POP(); PUSH(INT_VAL(tolower((int)AS_NUM(v)))); break; } /* char-downcase */
            case 217: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((int)AS_NUM(a)==(int)AS_NUM(b))})); break; } /* char=? */
            case 218: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((int)AS_NUM(a)<(int)AS_NUM(b))})); break; } /* char<? */
            case 219: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((int)AS_NUM(a)>(int)AS_NUM(b))})); break; } /* char>? */

            /* === Bitwise === */
            case 220: { Value b=POP(),a=POP(); PUSH(INT_VAL((int64_t)AS_NUM(a) & (int64_t)AS_NUM(b))); break; } /* bitwise-and */
            case 221: { Value b=POP(),a=POP(); PUSH(INT_VAL((int64_t)AS_NUM(a) | (int64_t)AS_NUM(b))); break; } /* bitwise-or */
            case 222: { Value b=POP(),a=POP(); PUSH(INT_VAL((int64_t)AS_NUM(a) ^ (int64_t)AS_NUM(b))); break; } /* bitwise-xor */
            case 223: { Value a=POP(); PUSH(INT_VAL(~(int64_t)AS_NUM(a))); break; } /* bitwise-not */
            case 224: { Value b=POP(),a=POP(); int64_t n=(int64_t)AS_NUM(b); PUSH(INT_VAL(n>=0 ? ((int64_t)AS_NUM(a)<<n) : ((int64_t)AS_NUM(a)>>(-n)))); break; } /* arithmetic-shift */

            /* === Additional list ops (these call closures, implemented in Scheme prelude) */
            /* take(225), drop(226), any(227), every(228), find(229), sort(230) */
            /* These are handled by the Scheme prelude definitions above. */
            /* Native stubs kept as fallback — should not be reached. */
            case 225: case 226: case 227: case 228: case 229: case 230:
                fprintf(stderr, "NATIVE_CALL %d: use Scheme prelude version\n", fid);
                PUSH(((Value){.type=VAL_NIL})); break;

            /* === Additional string ops === */
            case 231: { /* string-repeat: str n → repeated string */
                Value n_v=POP(), s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int n = (int)AS_NUM(n_v);
                int slen = heap[s.as.ptr].string.len;
                int rlen = slen * n; if (rlen > 255) rlen = 255;
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING; heap[ptr].string.len = rlen;
                for (int i = 0; i < rlen; i++) heap[ptr].string.data[i] = heap[s.as.ptr].string.data[i % slen];
                heap[ptr].string.data[rlen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 232: { /* string-trim */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int slen = heap[s.as.ptr].string.len;
                int start = 0, end = slen;
                while (start < end && isspace(heap[s.as.ptr].string.data[start])) start++;
                while (end > start && isspace(heap[s.as.ptr].string.data[end-1])) end--;
                int nlen = end - start;
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING; heap[ptr].string.len = nlen;
                memcpy(heap[ptr].string.data, heap[s.as.ptr].string.data + start, nlen);
                heap[ptr].string.data[nlen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 233: { /* string-split: str delim → list of strings */
                Value delim=POP(), s=POP();
                if (s.type != VAL_STRING || delim.type != VAL_STRING) { PUSH(((Value){.type=VAL_NIL})); break; }
                Value result = {.type=VAL_NIL};
                char* str = heap[s.as.ptr].string.data;
                int dlen = heap[delim.as.ptr].string.len;
                char* d = heap[delim.as.ptr].string.data;
                /* Simple split */
                char* prev = str;
                for (char* p = str; *p; p++) {
                    if (dlen > 0 && strncmp(p, d, dlen) == 0) {
                        int seglen = (int)(p - prev);
                        int32_t sptr = HALLOC(); if (sptr < 0) break;
                        heap[sptr].type = HEAP_STRING; heap[sptr].string.len = seglen;
                        memcpy(heap[sptr].string.data, prev, seglen); heap[sptr].string.data[seglen] = 0;
                        int32_t cp = HALLOC(); if (cp < 0) break;
                        heap[cp].type = HEAP_CONS;
                        heap[cp].cons.car = (Value){.type=VAL_STRING,.as.ptr=sptr};
                        heap[cp].cons.cdr = result;
                        result = (Value){.type=VAL_PAIR,.as.ptr=cp};
                        p += dlen - 1; prev = p + 1;
                    }
                }
                /* Last segment */
                int seglen = (int)(str + heap[s.as.ptr].string.len - prev);
                int32_t sptr = HALLOC(); if (sptr < 0) { PUSH(result); break; }
                heap[sptr].type = HEAP_STRING; heap[sptr].string.len = seglen;
                memcpy(heap[sptr].string.data, prev, seglen); heap[sptr].string.data[seglen] = 0;
                int32_t cp = HALLOC(); if (cp < 0) { PUSH(result); break; }
                heap[cp].type = HEAP_CONS;
                heap[cp].cons.car = (Value){.type=VAL_STRING,.as.ptr=sptr};
                heap[cp].cons.cdr = result;
                result = (Value){.type=VAL_PAIR,.as.ptr=cp};
                /* Reverse (split built in reverse order) */
                Value rev = {.type=VAL_NIL};
                while (result.type == VAL_PAIR) {
                    int32_t rp = HALLOC(); if (rp < 0) break;
                    heap[rp].type = HEAP_CONS; heap[rp].cons.car = heap[result.as.ptr].cons.car; heap[rp].cons.cdr = rev;
                    rev = (Value){.type=VAL_PAIR,.as.ptr=rp};
                    result = heap[result.as.ptr].cons.cdr;
                }
                PUSH(rev); break;
            }
            case 234: { /* string-join: list-of-strings separator → string */
                Value sep=POP(), lst=POP();
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING; heap[ptr].string.len = 0; heap[ptr].string.data[0] = 0;
                int pos = 0; int first_j = 1;
                Value cur = lst;
                while (cur.type == VAL_PAIR && pos < 250) {
                    Value s = heap[cur.as.ptr].cons.car;
                    if (!first_j && sep.type == VAL_STRING) {
                        int sl = heap[sep.as.ptr].string.len;
                        if (pos + sl > 255) { sl = 255 - pos; if (sl <= 0) break; }
                        memcpy(heap[ptr].string.data + pos, heap[sep.as.ptr].string.data, sl); pos += sl;
                    }
                    first_j = 0;
                    if (s.type == VAL_STRING) {
                        int sl = heap[s.as.ptr].string.len;
                        if (pos + sl > 255) { sl = 255 - pos; if (sl <= 0) break; }
                        memcpy(heap[ptr].string.data + pos, heap[s.as.ptr].string.data, sl); pos += sl;
                    }
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                heap[ptr].string.len = pos; heap[ptr].string.data[pos] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }

            /* === Misc === */
            case 235: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=IS_FALSY(v)})); break; } /* not */
            case 236: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.as.b==b.as.b)})); break; } /* boolean=? */
            case 237: { /* error: raise with message */
                Value msg=POP(); PUSH(msg); /* push as exception value */
                /* Trigger raise logic */
                if (handler_count <= 0) {
                    printf("ERROR: "); if (msg.type==VAL_STRING) printf("%.*s",heap[msg.as.ptr].string.len,heap[msg.as.ptr].string.data); else printf("error"); printf("\n");
                    error=1; break;
                }
                handler_count--;
                sp = exc_handlers[handler_count].saved_sp;
                fp = exc_handlers[handler_count].saved_fp;
                frame_count = exc_handlers[handler_count].saved_frame_count;
                pc = exc_handlers[handler_count].handler_pc;
                current_exn = msg;
                break;
            }
            case 238: { PUSH(((Value){.type=VAL_NIL})); break; } /* void */
            case 239: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_HASH)})); break; } /* hash-table? */
            case 250: { Value x=POP(),y=POP(); PUSH(FLOAT_VAL(atan2(AS_NUM(y),AS_NUM(x)))); break; } /* atan2 */
            case 252: { /* propagate_open_slot: child_closure, child_uv_idx, parent_uv_idx → void
                         * If the parent closure (at stack[fp-1]) has an open slot for parent_uv_idx,
                         * copy that open slot to child_closure's child_uv_idx. */
                Value parent_uv = POP(), child_uv = POP(), child_cl = POP();
                int puv = (int)AS_NUM(parent_uv);
                int cuv = (int)AS_NUM(child_uv);
                Value parent_cl = stack[fp - 1]; /* current frame's closure */
                if (child_cl.type == VAL_CLOSURE && parent_cl.type == VAL_CLOSURE
                    && puv >= 0 && puv < heap[parent_cl.as.ptr].closure.n_upvalues
                    && cuv >= 0 && cuv < heap[child_cl.as.ptr].closure.n_upvalues) {
                    int32_t parent_open = heap[parent_cl.as.ptr].closure.open_slots[puv];
                    if (parent_open >= 0) {
                        /* Parent has an open slot — propagate to child */
                        heap[child_cl.as.ptr].closure.open_slots[cuv] = parent_open;
                    }
                }
                PUSH(((Value){.type=VAL_NIL})); break;
            }
            case 251: { /* call-with-values-apply: result consumer → call consumer with unpacked result */
                Value consumer=POP(), result=POP();
                if (consumer.type != VAL_CLOSURE) { PUSH(((Value){.type=VAL_NIL})); break; }
                if (result.type == VAL_VECTOR) {
                    /* Unpack vector values as arguments */
                    int argc = heap[result.as.ptr].vector.len;
                    PUSH(consumer);
                    for (int vi = 0; vi < argc; vi++)
                        PUSH(heap[result.as.ptr].vector.items[vi]);
                    if (frame_count >= MAX_FRAMES) { error=1; break; }
                    frames[frame_count].return_pc = pc;
                    frames[frame_count].return_fp = fp;
                    frames[frame_count].heap_mark = heap_next;
                    frames[frame_count].force_promise_ptr = -1;
                    frame_count++;
                    fp = sp - argc;
                    pc = heap[consumer.as.ptr].closure.func_pc;
                } else {
                    /* Single value — call consumer(result) */
                    PUSH(consumer);
                    PUSH(result);
                    if (frame_count >= MAX_FRAMES) { error=1; break; }
                    frames[frame_count].return_pc = pc;
                    frames[frame_count].return_fp = fp;
                    frames[frame_count].heap_mark = heap_next;
                    frames[frame_count].force_promise_ptr = -1;
                    frame_count++;
                    fp = sp - 1;
                    pc = heap[consumer.as.ptr].closure.func_pc;
                }
                break;
            }
            case 240: { /* display: human-readable output (no quotes on strings) */
                Value v=POP(); print_value(v, heap, 0, 0); PUSH(((Value){.type=VAL_NIL})); break;
            }
            case 241: { /* write: machine-readable output (strings quoted with escapes) */
                Value v=POP(); print_value(v, heap, 0, 1); PUSH(((Value){.type=VAL_NIL})); break;
            }

            case 151: { /* set_open_slot: closure, uv_idx, abs_slot → set one open upvalue slot */
                Value slot_v = POP(), idx_v = POP(), cl = POP();
                int idx = (int)AS_NUM(idx_v);
                int abs_slot = fp + (int)AS_NUM(slot_v); /* fp-relative → absolute */
                if (cl.type == VAL_CLOSURE && idx >= 0 && idx < heap[cl.as.ptr].closure.n_upvalues) {
                    heap[cl.as.ptr].closure.open_slots[idx] = abs_slot;
                }
                PUSH(((Value){.type=VAL_NIL}));
                break;
            }
            case 132: { /* force: force a promise #(forced? thunk-or-value) */
                Value prom = POP();
                if (prom.type != VAL_VECTOR || heap[prom.as.ptr].vector.len < 2) {
                    /* Not a promise — return as-is */
                    PUSH(prom);
                    break;
                }
                Value forced = heap[prom.as.ptr].vector.items[0];
                if (forced.type == VAL_BOOL && forced.as.b) {
                    /* Already forced — return cached value */
                    PUSH(heap[prom.as.ptr].vector.items[1]);
                } else {
                    /* Not yet forced — call the thunk */
                    Value thunk = heap[prom.as.ptr].vector.items[1];
                    if (thunk.type == VAL_CLOSURE) {
                        /* Call thunk inline: push closure, CALL 0 setup.
                         * Save promise index so we can memoize after return. */
                        PUSH(thunk);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = prom.as.ptr;
                        frame_count++;
                        fp = sp;
                        pc = heap[thunk.as.ptr].closure.func_pc;
                    } else {
                        /* Thunk is already a value — memoize and return it */
                        heap[prom.as.ptr].vector.items[0] = (Value){.type=VAL_BOOL,.as.b=1};
                        PUSH(thunk);
                    }
                }
                break;
            }
            case 131: { /* open_upvalues: convert closure's upvalues to open (stack refs)
                         * Stack: [closure, count, base_slot]
                         * Sets closure.open_slots[i] = fp + base_slot + i for i in 0..count-1
                         * base_slot is compile-time relative; fp adjusts to runtime position */
                Value base_v = POP(), count_v = POP(), cl = POP();
                int count = (int)AS_NUM(count_v);
                int base = fp + (int)AS_NUM(base_v); /* fp-relative → absolute */
                if (cl.type == VAL_CLOSURE) {
                    for (int i = 0; i < count && i < heap[cl.as.ptr].closure.n_upvalues; i++) {
                        heap[cl.as.ptr].closure.open_slots[i] = base + i;
                    }
                }
                PUSH(((Value){.type=VAL_NIL})); /* return nil */
                break;
            }

            case 100: { /* build-string-from-packed: stack has [len, pack0, pack1, ...] */
                /* Read how many packs from the length */
                /* Stack layout: ..., len, pack0, pack1, ..., packN-1 (top) */
                /* We need to pop packs in reverse, then pop len */
                /* But we need len to know how many packs... */
                /* Convention: len is deepest, packs are on top */
                /* Calculate n_packs from the len value below the packs */
                /* This is tricky — we don't know sp offset for len without scanning. */
                /* Alternative: use the instruction's embedded data.
                 * Actually: NATIVE_CALL 100 is preceded by CONST(len), CONST(pack0), etc.
                 * The len value is at sp - n_packs - 1, packs at sp - n_packs .. sp - 1. */
                /* We'll read backwards: first pop all packs into a buffer, then pop len. */
                /* But we don't know n_packs without len! */
                /* Fix: pop values until we find the len marker.
                 * Len is always a small positive integer. Packs are large (multi-byte).
                 * This is fragile. Better approach: use a fixed protocol. */
                /* REVISED: the caller pushes len FIRST, then packs.
                 * At this point stack = [..., len, p0, p1, ..., pN-1]
                 * We need to peek at len to know N = (len+7)/8.
                 * Then pop N packs, pop len. */
                {
                    /* First, peek at the length value (it's below the packs) */
                    /* We need to find it. Count backwards from sp. */
                    /* Try: assume the len is the value at sp - n_packs - 1 where
                     * n_packs is determined by the len value itself. Chicken-and-egg.
                     * Solution: just try small values of n_packs. */
                    int slen = 0;
                    int npacks = 0;
                    /* Scan stack for the length value (must be < 256 and at position sp - k - 1
                     * where k = (len+7)/8) */
                    for (int try_len = 0; try_len < 256; try_len++) {
                        int try_np = (try_len + 7) / 8;
                        int len_pos = sp - try_np - 1;
                        if (len_pos >= 0 && stack[len_pos].type == VAL_INT && stack[len_pos].as.i == try_len) {
                            slen = try_len;
                            npacks = try_np;
                            break;
                        }
                    }
                    /* Pop packs */
                    int64_t packs[32];
                    for (int i = npacks - 1; i >= 0; i--) {
                        Value v = POP();
                        packs[i] = v.as.i;
                    }
                    POP(); /* pop len */
                    /* Build string */
                    int32_t sptr = HALLOC(); if (sptr < 0) break;
                    heap[sptr].type = HEAP_STRING;
                    heap[sptr].string.len = slen;
                    for (int i = 0; i < slen && i < 255; i++) {
                        int pack_idx = i / 8;
                        int byte_idx = i % 8;
                        heap[sptr].string.data[i] = (char)((packs[pack_idx] >> (byte_idx * 8)) & 0xFF);
                    }
                    heap[sptr].string.data[slen] = 0;
                    PUSH(((Value){.type=VAL_STRING,.as.ptr=sptr}));
                }
                break;
            }

            /* AD forward mode: dual numbers */
            case 110: { /* make-dual: value, derivative → dual number (stored as 2-element vector) */
                Value deriv = POP(), val = POP();
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = val;
                heap[ptr].vector.items[1] = deriv;
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 111: { /* dual-value: dual → value part */
                Value d = POP();
                if (d.type == VAL_VECTOR && heap[d.as.ptr].vector.len >= 2)
                    PUSH(heap[d.as.ptr].vector.items[0]);
                else PUSH(d); /* non-dual: value is itself */
                break;
            }
            case 112: { /* dual-derivative: dual → derivative part */
                Value d = POP();
                if (d.type == VAL_VECTOR && heap[d.as.ptr].vector.len >= 2)
                    PUSH(heap[d.as.ptr].vector.items[1]);
                else PUSH(FLOAT_VAL(0.0)); /* non-dual: derivative is 0 */
                break;
            }
            case 113: { /* dual-add: add two dual numbers */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av + bv);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad + bd);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 114: { /* dual-mul: product rule (a,a')*(b,b') = (a*b, a*b'+a'*b) */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av * bv);
                heap[ptr].vector.items[1] = FLOAT_VAL(av * bd + ad * bv);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 115: { /* dual-sub: subtract dual numbers */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av - bv);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad - bd);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 116: { /* dual-div: quotient rule (a,a')/(b,b') = (a/b, (a'*b-a*b')/b²) */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av / bv);
                heap[ptr].vector.items[1] = FLOAT_VAL((ad * bv - av * bd) / (bv * bv));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 117: { /* dual-sin: sin(a,a') = (sin(a), a'*cos(a)) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(sin(av));
                heap[ptr].vector.items[1] = FLOAT_VAL(ad * cos(av));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 118: { /* dual-cos: cos(a,a') = (cos(a), -a'*sin(a)) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(cos(av));
                heap[ptr].vector.items[1] = FLOAT_VAL(-ad * sin(av));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 119: { /* dual-exp: exp(a,a') = (exp(a), a'*exp(a)) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                double ea = exp(av);
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(ea);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad * ea);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 120: { /* dual-log: log(a,a') = (log(a), a'/a) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(log(av));
                heap[ptr].vector.items[1] = FLOAT_VAL(ad / av);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 121: { /* dual-sqrt: sqrt(a,a') = (sqrt(a), a'/(2*sqrt(a))) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                double sa = sqrt(av);
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(sa);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad / (2.0 * sa));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }

            case 260: { /* make-vector: TOS=fill, SOS=n → create vector of n elements filled with fill */
                Value fill = POP();
                Value n_val = POP();
                int n = (int)AS_NUM(n_val);
                if (n < 0) n = 0;
                if (n > 256) n = 256; /* safety limit */
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = n;
                for (int i = 0; i < n; i++) {
                    heap[ptr].vector.items[i] = fill;
                }
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }

            default:
                fprintf(stderr, "ERROR: NATIVE_CALL %d not implemented\n", fid);
                error = 1;
                break;
            }
            break;
        }

        case OP_HALT: halted = 1; break;
        default: printf("  UNKNOWN OPCODE %d at PC=%d\n", ins.op, pc-1); error = 1; break;
        }
    }

    if (error) {
        fprintf(stderr, "=== VM ERROR at PC=%d ===\n", pc);
        fprintf(stderr, "  sp=%d fp=%d frame_count=%d heap=%d/%d\n",
                sp, fp, frame_count, heap_next, HEAP_SIZE);
        if (pc > 0 && pc <= chunk->code_len) {
            fprintf(stderr, "  last opcode: %d (operand=%d)\n",
                    chunk->code[pc-1].op, chunk->code[pc-1].operand);
        }
        fprintf(stderr, "  stack top 5:");
        for (int i = 0; i < 5 && i < sp; i++) {
            Value v = stack[sp - 1 - i];
            if (v.type == VAL_INT) fprintf(stderr, " %lld", (long long)v.as.i);
            else if (v.type == VAL_FLOAT) fprintf(stderr, " %.4g", v.as.f);
            else if (v.type == VAL_BOOL) fprintf(stderr, " %s", v.as.b ? "#t" : "#f");
            else if (v.type == VAL_NIL) fprintf(stderr, " nil");
            else fprintf(stderr, " <type=%d>", v.type);
        }
        fprintf(stderr, "\n");
    } else {
        printf("  [metrics] %lld insns, max_depth=%d, heap=%d/%d\n",
               (long long)insn_count, max_depth, heap_next, HEAP_SIZE);
    }

    #undef PUSH
    #undef POP
    #undef PEEK
    #undef AS_NUM
    #undef NUM_VAL
    #undef IS_FALSY
    #undef HALLOC

    free(stack); free(heap); free(frames);
}

/*******************************************************************************
 * Compile & Run
 ******************************************************************************/

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
    {"fg-update-cpt!", 524, 3},
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
    {"model-save", 800, 2}, {"model-load", 801, 1},
    {"tensor-save", 802, 2}, {"tensor-load", 803, 1},
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
    execute_chunk(&main_chunk);
}

int main(int argc, char** argv) {
    printf("=== Eshkol Compiler (targeting 38-opcode VM) ===\n\n");

    /* Check for --emit-eskb and --trace flags */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--emit-eskb") == 0 && i + 1 < argc) {
            g_eskb_output_path = argv[++i];
        }
        if (strcmp(argv[i], "--trace") == 0) g_trace_on = 1;
    }

    /* Find the input file (first arg that isn't a flag) */
    const char* input_file = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--emit-eskb") == 0) { i++; continue; }
        if (strcmp(argv[i], "--trace") == 0) continue;
        input_file = argv[i];
        break;
    }

    if (input_file) {
        /* Check if input is a .eskb file — load and execute directly */
        size_t flen = strlen(input_file);
        if (flen > 5 && strcmp(input_file + flen - 5, ".eskb") == 0) {
            EskbModule mod;
            if (eskb_load_file(input_file, &mod) == 0) {
                /* Convert to VM FuncChunk format and execute */
                printf("  [ESKB] Loaded %d instructions, %d constants from %s\n",
                       mod.code_len, mod.n_constants, input_file);
                if (mod.has_debug) printf("  [ESKB] Source: %s\n", mod.source_file);

                /* Build a FuncChunk from the loaded module */
                FuncChunk chunk = {0};
                for (int i = 0; i < mod.code_len && i < MAX_CODE; i++) {
                    chunk.code[i] = (Instr){mod.opcodes[i], mod.operands[i]};
                }
                chunk.code_len = mod.code_len;
                for (int i = 0; i < mod.n_constants && i < MAX_CONSTS; i++) {
                    switch (mod.const_types[i]) {
                    case ESKB_CONST_NIL:
                        chunk.constants[i] = (Value){.type=VAL_NIL, .as.i=0};
                        break;
                    case ESKB_CONST_INT64:
                        chunk.constants[i] = (Value){.type=VAL_INT, .as.i=mod.const_ints[i]};
                        break;
                    case ESKB_CONST_F64:
                        chunk.constants[i] = (Value){.type=VAL_FLOAT, .as.f=mod.const_floats[i]};
                        break;
                    case ESKB_CONST_BOOL:
                        chunk.constants[i] = (Value){.type=VAL_BOOL, .as.b=(int)mod.const_ints[i]};
                        break;
                    default:
                        chunk.constants[i] = (Value){.type=VAL_INT, .as.i=mod.const_ints[i]};
                        break;
                    }
                }
                chunk.n_constants = mod.n_constants;
                execute_chunk(&chunk);
                eskb_module_free(&mod);
            } else {
                printf("ERROR: failed to load ESKB file %s\n", input_file);
                return 1;
            }
        } else {
            /* Compile .esk source file */
            FILE* f = fopen(input_file, "r");
            if (!f) { printf("ERROR: cannot open %s\n", input_file); return 1; }
            fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
            char* src = (char*)malloc(len + 1);
            fread(src, 1, len, f); src[len] = 0; fclose(f);
            printf("  Source: %s\n", input_file);
            g_source_file_path = input_file;
            compile_and_run(src);
            free(src);
        }
    } else {
        /* Built-in tests */
        printf("  Test 1: (display (+ 3 5))\n");
        compile_and_run("(display (+ 3 5))");

        printf("\n  Test 2: (let ((x 10)) (display (if (> x 5) (* x 2) x)))\n");
        compile_and_run("(let ((x 10)) (display (if (> x 5) (* x 2) x)))");

        printf("\n  Test 3: factorial\n");
        compile_and_run(
            "(define (factorial n)\n"
            "  (if (= n 0) 1 (* n (factorial (- n 1)))))\n"
            "(display (factorial 10))\n");

        printf("\n  Test 4: fibonacci\n");
        compile_and_run(
            "(define (fib n)\n"
            "  (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))\n"
            "(display (fib 10))\n");

        printf("\n  Test 5: map with lambda\n");
        compile_and_run(
            "(define (map f lst)\n"
            "  (if (null? lst) '()\n"
            "      (cons (f (car lst)) (map f (cdr lst)))))\n"
            "(display (map (lambda (x) (* x x)) (list 1 2 3 4 5)))\n");

        /* ── Edge Case Tests ── */
        printf("\n  Edge case tests:\n");

        /* Modulo with negative numbers (R7RS floored modulo) */
        printf("\n  Test 6: modulo-neg\n");
        compile_and_run("(display (modulo -5 3))");  /* should output 1 */

        /* Nested let expressions */
        printf("\n  Test 7: nested-let\n");
        compile_and_run("(display (let ((x 10)) (let ((y (+ x 5))) (* x y))))");  /* should output 150 */

        /* Higher-order: compose */
        printf("\n  Test 8: compose\n");
        compile_and_run(
            "(define (compose f g) (lambda (x) (f (g x))))\n"
            "(define (add1 x) (+ x 1))\n"
            "(define (double x) (* x 2))\n"
            "(display ((compose double add1) 5))");  /* should output 12 */

        /* Tail recursion: should not stack overflow */
        printf("\n  Test 9: deep-tail-call\n");
        compile_and_run(
            "(define (count n) (if (= n 0) 0 (count (- n 1))))\n"
            "(display (count 100000))");  /* should output 0 */

        /* Boolean edge cases */
        printf("\n  Test 10: bool-edge\n");
        compile_and_run("(display (if '() 1 2))");  /* R7RS: '() is truthy; should output 1 */

        /* Constant folding verification */
        printf("\n  Test 11: const-fold\n");
        compile_and_run("(display (+ (* 3 4) (- 10 5)))");  /* should output 17, folded at compile time */

        printf("\n  All tests complete.\n");
    }

    printf("\n=== Compilation complete ===\n");
    return 0;
}
