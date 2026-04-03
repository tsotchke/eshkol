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

/* Compiler context — encapsulates all mutable state for reentrancy and REPL */
typedef struct {
    const char* src_ptr;       /* Current parse position */
    int trace_on;              /* Trace execution flag */
    const char* eskb_output;   /* ESKB output path (--emit-eskb) */
    const char* source_path;   /* Source file path */
    char loaded_modules[64][128]; /* Module cache for require */
    int n_loaded;
} CompilerContext;

static CompilerContext g_compiler_ctx = {0};

/* Convenience macros — allow existing code to use src_ptr directly */
#define src_ptr   g_compiler_ctx.src_ptr
#define g_trace_on g_compiler_ctx.trace_on

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
        strncpy(qs->symbol, "quote", 127); qs->symbol[127] = 0;
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
        strncpy(tag->symbol, "quasiquote", 127); tag->symbol[127] = 0;
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
        strncpy(tag->symbol, "unquote-splicing", 127); tag->symbol[127] = 0;
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
        strncpy(tag->symbol, "unquote", 127); tag->symbol[127] = 0;
        add_child(q, tag);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* String literal */
    if (*src_ptr == '"') {
        src_ptr++; /* skip opening quote */
        int buf_cap = 256;
        char* buf = (char*)malloc(buf_cap);
        int i = 0;
        while (*src_ptr && *src_ptr != '"') {
            if (i >= buf_cap - 2) {
                buf_cap *= 2;
                buf = (char*)realloc(buf, buf_cap);
            }
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
        Node* n = make_node(N_STRING); if (!n) { free(buf); return NULL; }
        strncpy(n->symbol, buf, 127); n->symbol[127] = 0;
        free(buf);
        return n;
    }
    if (*src_ptr == '#') {
        if (src_ptr[1] == 't' && (src_ptr[2] == 0 || isspace(src_ptr[2]) || src_ptr[2] == ')')) {
            src_ptr += 2; Node* n = make_node(N_BOOL); if (!n) return NULL; n->numval = 1; strncpy(n->symbol, "#t", 127); n->symbol[127] = 0; return n;
        }
        if (src_ptr[1] == 'f' && (src_ptr[2] == 0 || isspace(src_ptr[2]) || src_ptr[2] == ')')) {
            src_ptr += 2; Node* n = make_node(N_BOOL); if (!n) return NULL; n->numval = 0; strncpy(n->symbol, "#f", 127); n->symbol[127] = 0; return n;
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
            strncpy(tag->symbol, "vector", 127); tag->symbol[127] = 0;
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
    char* name;   /* heap-allocated via strdup() */
    int slot;
    int depth;
    int boxed;    /* 1 = variable is heap-boxed (stored in 1-element vector) */
} Local;

typedef struct {
    char* name;   /* heap-allocated via strdup() */
    int enclosing_slot;  /* slot or upvalue index in the enclosing scope */
    int index;           /* upvalue index in this closure */
    int is_local;        /* 1 = enclosing_slot is a local, 0 = it's an upvalue */
    int boxed;           /* 1 = the captured variable is heap-boxed */
} Upvalue;

#define MAX_UPVALUES 32
#define CHUNK_INIT_CODE 256
#define CHUNK_INIT_CONSTS 64
#define CHUNK_INIT_LOCALS 32

typedef struct FuncChunk {
    Instr* code;         int code_len;     int code_cap;
    Value* constants;    int n_constants;  int const_cap;
    Local* locals;       int n_locals;     int local_cap;
    Upvalue upvalues[MAX_UPVALUES];
    int n_upvalues;
    int scope_depth;
    int scope_stack_base[64]; /* stack depth at scope entry, for cleanup on exit */
    struct FuncChunk* enclosing;
    int param_count;
    int stack_depth;  /* compile-time stack depth (values above fp) */
} FuncChunk;

/* Initialize a FuncChunk's dynamic arrays (for stack-allocated chunks) */
static int chunk_init_arrays(FuncChunk* c) {
    memset(c, 0, sizeof(FuncChunk));
    c->code_cap = CHUNK_INIT_CODE;
    c->code = (Instr*)calloc(c->code_cap, sizeof(Instr));
    c->const_cap = CHUNK_INIT_CONSTS;
    c->constants = (Value*)calloc(c->const_cap, sizeof(Value));
    c->local_cap = CHUNK_INIT_LOCALS;
    c->locals = (Local*)calloc(c->local_cap, sizeof(Local));
    if (!c->code || !c->constants || !c->locals) {
        free(c->code); free(c->constants); free(c->locals);
        c->code = NULL; c->constants = NULL; c->locals = NULL;
        fprintf(stderr, "ERROR: cannot allocate FuncChunk arrays\n");
        return -1;
    }
    return 0;
}

/* Free a FuncChunk's dynamic arrays (for stack-allocated chunks) */
static void chunk_free_arrays(FuncChunk* c) {
    if (!c) return;
    for (int i = 0; i < c->n_locals; i++) free(c->locals[i].name);
    for (int i = 0; i < c->n_upvalues; i++) free(c->upvalues[i].name);
    free(c->code); free(c->constants); free(c->locals);
    c->code = NULL; c->constants = NULL; c->locals = NULL;
}

/* Heap-allocate a FuncChunk with dynamic arrays (~300 bytes vs. 354KB fixed) */
static FuncChunk* chunk_create(void) {
    FuncChunk* c = (FuncChunk*)calloc(1, sizeof(FuncChunk));
    if (!c) { fprintf(stderr, "ERROR: cannot allocate FuncChunk\n"); return NULL; }
    c->code_cap = CHUNK_INIT_CODE;
    c->code = (Instr*)calloc(c->code_cap, sizeof(Instr));
    c->const_cap = CHUNK_INIT_CONSTS;
    c->constants = (Value*)calloc(c->const_cap, sizeof(Value));
    c->local_cap = CHUNK_INIT_LOCALS;
    c->locals = (Local*)calloc(c->local_cap, sizeof(Local));
    if (!c->code || !c->constants || !c->locals) {
        free(c->code); free(c->constants); free(c->locals); free(c);
        fprintf(stderr, "ERROR: cannot allocate FuncChunk arrays\n");
        return NULL;
    }
    return c;
}
static void chunk_destroy(FuncChunk* c) {
    if (!c) return;
    for (int i = 0; i < c->n_locals; i++) free(c->locals[i].name);
    for (int i = 0; i < c->n_upvalues; i++) free(c->upvalues[i].name);
    free(c->code); free(c->constants); free(c->locals);
    free(c);
}

static int is_sym(Node* n, const char* s) { return n && n->type == N_SYMBOL && strcmp(n->symbol, s) == 0; }

static void chunk_ensure_code_cap(FuncChunk* c, int needed) {
    while (c->code_len + needed > c->code_cap) {
        int new_cap = c->code_cap * 2;
        Instr* new_code = (Instr*)realloc(c->code, new_cap * sizeof(Instr));
        if (!new_code) { fprintf(stderr, "ERROR: bytecode realloc failed\n"); return; }
        c->code = new_code;
        c->code_cap = new_cap;
    }
}

static void chunk_emit(FuncChunk* c, uint8_t op, int32_t operand) {
    chunk_ensure_code_cap(c, 1);
    c->code[c->code_len++] = (Instr){op, operand};
}

/* Copy an instruction directly (used when inlining function code) */
static void chunk_emit_instr(FuncChunk* c, Instr fi) {
    chunk_ensure_code_cap(c, 1);
    c->code[c->code_len++] = fi;
}


static int chunk_add_const(FuncChunk* c, Value v) {
    /* No deduplication — function PC placeholders get patched after creation,
     * which would corrupt literal constants that matched the placeholder value. */
    if (c->n_constants >= c->const_cap) {
        int new_cap = c->const_cap * 2;
        Value* new_consts = (Value*)realloc(c->constants, new_cap * sizeof(Value));
        if (!new_consts) { fprintf(stderr, "ERROR: constant pool realloc failed\n"); return -1; }
        c->constants = new_consts;
        c->const_cap = new_cap;
    }
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
    if (c->n_locals >= c->local_cap) {
        int new_cap = c->local_cap * 2;
        Local* new_locals = (Local*)realloc(c->locals, new_cap * sizeof(Local));
        if (!new_locals) { fprintf(stderr, "ERROR: local variable realloc failed\n"); return -1; }
        c->locals = new_locals;
        c->local_cap = new_cap;
    }
    int slot = c->n_locals;
    c->locals[c->n_locals].name = strdup(name);
    c->locals[c->n_locals].slot = slot;
    c->locals[c->n_locals].depth = c->scope_depth;
    c->locals[c->n_locals].boxed = 0;
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

