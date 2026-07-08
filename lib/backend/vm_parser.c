/*******************************************************************************
 * Eshkol Source Compiler (parser + bytecode generator)
 * Merged from eshkol_compiler.c — single interpreter, one dispatch table.
 ******************************************************************************/

#include "eshkol/backend/vm_limits.h"

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

static int append_char_buf(char** buf, int* len, int* cap, char ch) {
    if (!buf || !*buf || !len || !cap) return -1;
    if (*len >= *cap - 2) {
        int next_cap = *cap * 2;
        char* next = (char*)realloc(*buf, next_cap);
        if (!next) return -1;
        *buf = next;
        *cap = next_cap;
    }
    (*buf)[(*len)++] = ch;
    return 0;
}

static Node* make_symbol_node(const char* text) {
    Node* n = make_node(N_SYMBOL);
    if (!n) return NULL;
    strncpy(n->symbol, text ? text : "", 127);
    n->symbol[127] = 0;
    return n;
}

static Node* make_string_node(const char* text) {
    Node* n = make_node(N_STRING);
    if (!n) return NULL;
    strncpy(n->symbol, text ? text : "", 127);
    n->symbol[127] = 0;
    return n;
}

static Node* make_call_node(const char* name) {
    Node* call = make_node(N_LIST);
    if (!call) return NULL;
    Node* head = make_symbol_node(name);
    if (!head) { free_node(call); return NULL; }
    add_child(call, head);
    return call;
}

static Node* parse_sexp_from_string(const char* source) {
    const char* saved_src = src_ptr;
    src_ptr = source ? source : "";
    skip_ws();
    if (!*src_ptr) {
        fprintf(stderr, "ERROR: string interpolation expression cannot be empty\n");
        src_ptr = saved_src;
        return NULL;
    }
    Node* expr = parse_sexp();
    skip_ws();
    if (*src_ptr) {
        fprintf(stderr, "ERROR: string interpolation accepts exactly one expression\n");
        free_node(expr);
        expr = NULL;
    }
    src_ptr = saved_src;
    return expr;
}

static int add_part(Node*** parts, int* n_parts, int* cap_parts, Node* part) {
    if (!parts || !n_parts || !cap_parts || !part) return -1;
    if (*n_parts >= *cap_parts) {
        int next_cap = *cap_parts ? *cap_parts * 2 : 4;
        Node** next = (Node**)realloc(*parts, next_cap * sizeof(Node*));
        if (!next) return -1;
        *parts = next;
        *cap_parts = next_cap;
    }
    (*parts)[(*n_parts)++] = part;
    return 0;
}

static Node* make_format_display_node(Node* expr) {
    Node* call = make_call_node("format");
    if (!call) return NULL;
    add_child(call, make_string_node("~a"));
    add_child(call, expr);
    return call;
}

static Node* make_string_append_node(Node** parts, int n_parts) {
    if (n_parts <= 0) return make_string_node("");
    if (n_parts == 1) return parts[0];

    Node* call = make_call_node("string-append");
    if (!call) return NULL;
    for (int i = 0; i < n_parts; i++) add_child(call, parts[i]);
    return call;
}

static Node* parse_string_literal(void) {
    src_ptr++; /* skip opening quote */

    int buf_cap = 256;
    char* buf = (char*)malloc(buf_cap);
    if (!buf) return NULL;
    int len = 0;

    Node** parts = NULL;
    int n_parts = 0;
    int cap_parts = 0;
    int has_interpolation = 0;

    while (*src_ptr && *src_ptr != '"') {
        if (src_ptr[0] == '\\' && src_ptr[1]) {
            src_ptr++;
            char out = *src_ptr;
            switch (*src_ptr) {
                case 'n': out = '\n'; break;
                case 't': out = '\t'; break;
                case '\\': out = '\\'; break;
                case '"': out = '"'; break;
                default: break;
            }
            if (append_char_buf(&buf, &len, &buf_cap, out) != 0) {
                free(buf); free(parts); return NULL;
            }
            src_ptr++;
            continue;
        }

        if (src_ptr[0] == '~' && src_ptr[1] == '~' && src_ptr[2] == '{') {
            if (append_char_buf(&buf, &len, &buf_cap, '~') != 0 ||
                append_char_buf(&buf, &len, &buf_cap, '{') != 0) {
                free(buf); free(parts); return NULL;
            }
            src_ptr += 3;
            continue;
        }

        if (src_ptr[0] == '~' && src_ptr[1] == '{') {
            has_interpolation = 1;
            if (len > 0) {
                buf[len] = 0;
                if (add_part(&parts, &n_parts, &cap_parts, make_string_node(buf)) != 0) {
                    free(buf); free(parts); return NULL;
                }
                len = 0;
            }

            src_ptr += 2;
            int expr_cap = 128;
            char* expr_buf = (char*)malloc(expr_cap);
            if (!expr_buf) { free(buf); free(parts); return NULL; }
            int expr_len = 0;
            int in_expr_string = 0;
            int escaped = 0;
            int closed = 0;

            while (*src_ptr) {
                char ch = *src_ptr;
                if (!in_expr_string && ch == '}') {
                    closed = 1;
                    break;
                }

                char tracked = ch;
                int consumed = 1;
                if (ch == '\\' && src_ptr[1]) {
                    consumed = 2;
                    switch (src_ptr[1]) {
                        case 'n': tracked = '\n'; break;
                        case 't': tracked = '\t'; break;
                        case '\\': tracked = '\\'; break;
                        case '"': tracked = '"'; break;
                        default: tracked = src_ptr[1]; break;
                    }
                }

                if (append_char_buf(&expr_buf, &expr_len, &expr_cap, tracked) != 0) {
                    free(expr_buf); free(buf); free(parts); return NULL;
                }

                if (in_expr_string) {
                    if (escaped) {
                        escaped = 0;
                    } else if (tracked == '\\') {
                        escaped = 1;
                    } else if (tracked == '"') {
                        in_expr_string = 0;
                    }
                } else if (tracked == '"') {
                    in_expr_string = 1;
                }
                src_ptr += consumed;
            }

            if (!closed) {
                fprintf(stderr, "ERROR: unterminated string interpolation\n");
                free(expr_buf); free(buf); free(parts); return NULL;
            }

            expr_buf[expr_len] = 0;
            Node* expr = parse_sexp_from_string(expr_buf);
            free(expr_buf);
            if (!expr) { free(buf); free(parts); return NULL; }

            Node* formatted = make_format_display_node(expr);
            if (add_part(&parts, &n_parts, &cap_parts, formatted) != 0) {
                free_node(formatted); free(buf); free(parts); return NULL;
            }
            src_ptr++; /* skip closing interpolation brace */
            continue;
        }

        if (append_char_buf(&buf, &len, &buf_cap, *src_ptr) != 0) {
            free(buf); free(parts); return NULL;
        }
        src_ptr++;
    }

    if (*src_ptr == '"') src_ptr++; /* skip closing quote */
    buf[len] = 0;

    if (!has_interpolation) {
        Node* n = make_string_node(buf);
        free(buf);
        return n;
    }

    if (len > 0) {
        if (add_part(&parts, &n_parts, &cap_parts, make_string_node(buf)) != 0) {
            free(buf); free(parts); return NULL;
        }
    }
    free(buf);

    Node* result = make_string_append_node(parts, n_parts);
    free(parts);
    return result;
}

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
        return parse_string_literal();
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
    /* R7RS special float literals: +nan.0, +inf.0, -inf.0 */
    if ((src_ptr[0] == '+' || src_ptr[0] == '-') &&
        (strncmp(src_ptr + 1, "nan.0", 5) == 0 || strncmp(src_ptr + 1, "inf.0", 5) == 0)) {
        double val;
        if (strncmp(src_ptr + 1, "nan.0", 5) == 0) val = NAN;
        else if (src_ptr[0] == '+') val = INFINITY;
        else val = -INFINITY;
        src_ptr += 6; /* skip +nan.0 / +inf.0 / -inf.0 */
        Node* n = make_node(N_NUMBER); if (!n) return NULL;
        n->numval = val;
        return n;
    }
    /* Number (including rational literals like 1/3) */
    if (isdigit(*src_ptr) || (*src_ptr == '-' && isdigit(src_ptr[1]))) {
        char buf[64]; int i = 0;
        if (*src_ptr == '-') buf[i++] = *src_ptr++;
        while ((isdigit(*src_ptr) || *src_ptr == '.') && i < 63) buf[i++] = *src_ptr++;
        /* Scientific notation: e.g. 1e-6, 2.5E+10 */
        if (i < 62 && (*src_ptr == 'e' || *src_ptr == 'E')) {
            buf[i++] = *src_ptr++;
            if (i < 62 && (*src_ptr == '+' || *src_ptr == '-')) buf[i++] = *src_ptr++;
            while (isdigit(*src_ptr) && i < 63) buf[i++] = *src_ptr++;
        }
        /* Null-terminate the integer part BEFORE any atoll below. Without this
         * the rational-literal branch called atoll(buf) on a buffer whose tail
         * was still uninitialized stack memory, so atoll kept consuming any
         * garbage digit bytes after the real ones -> a corrupted numerator
         * (e.g. 1/3 parsed with num != 1). It only surfaced when the stack
         * garbage happened to be a digit, making it a memory-layout-dependent
         * heisenbug. The non-rational path re-terminates below (harmless). */
        buf[i] = 0;
        /* Check for rational literal: digits/digits */
        if (*src_ptr == '/' && isdigit(src_ptr[1])) {
            int64_t num = atoll(buf);
            src_ptr++; /* skip '/' */
            char den_buf[32]; int j = 0;
            while (isdigit(*src_ptr) && j < 31) den_buf[j++] = *src_ptr++;
            den_buf[j] = 0;
            int64_t denom = atoll(den_buf);
            if (denom == 0) denom = 1;
            /* Emit as (/ num denom) — a list node */
            Node* div_node = make_node(N_LIST); if (!div_node) return NULL;
            Node* op = make_node(N_SYMBOL); if (!op) return NULL;
            strncpy(op->symbol, "exact-rational", 127);
            Node* n_node = make_node(N_NUMBER); if (!n_node) return NULL; n_node->numval = (double)num;
            Node* d_node = make_node(N_NUMBER); if (!d_node) return NULL; d_node->numval = (double)denom;
            add_child(div_node, op); add_child(div_node, n_node); add_child(div_node, d_node);
            return div_node;
        }
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

#ifndef MAX_CODE
#define MAX_CODE 32768
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

typedef struct {
    char* name;   /* heap-allocated via strdup() */
    int n_params;
    int n_locals;
    int n_upvalues;
    int code_offset;
    int code_len;
} ChunkEntry;

#define MAX_UPVALUES 32
#define CHUNK_INIT_CODE 256
#define CHUNK_INIT_CONSTS 64
#define CHUNK_INIT_LOCALS 32
#define CHUNK_INIT_ENTRIES 8

typedef struct FuncChunk {
    Instr* code;         int code_len;     int code_cap;
    Value* constants;    int n_constants;  int const_cap;
    Local* locals;       int n_locals;     int local_cap;
    ChunkEntry* entries; int n_entries;    int entry_cap;
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
    c->entry_cap = CHUNK_INIT_ENTRIES;
    c->entries = (ChunkEntry*)calloc(c->entry_cap, sizeof(ChunkEntry));
    if (!c->code || !c->constants || !c->locals || !c->entries) {
        free(c->code); free(c->constants); free(c->locals); free(c->entries);
        c->code = NULL; c->constants = NULL; c->locals = NULL; c->entries = NULL;
        fprintf(stderr, "ERROR: cannot allocate FuncChunk arrays\n");
        return -1;
    }
    return 0;
}

/* Free a FuncChunk's dynamic arrays (for stack-allocated chunks) */
static void chunk_free_arrays(FuncChunk* c) {
    if (!c) return;
    for (int i = 0; i < c->n_locals; i++) free(c->locals[i].name);
    for (int i = 0; i < c->n_entries; i++) free(c->entries[i].name);
    for (int i = 0; i < c->n_upvalues; i++) free(c->upvalues[i].name);
    free(c->code); free(c->constants); free(c->locals); free(c->entries);
    c->code = NULL; c->constants = NULL; c->locals = NULL; c->entries = NULL;
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

static int chunk_add_entry(FuncChunk* c, const char* name, int n_params,
                           int n_locals, int n_upvalues, int code_offset,
                           int code_len) {
    if (!c || !name || !name[0] || code_offset < 0 || code_len <= 0) return -1;
    if (n_params < 0 || n_params > 255) return -1;
    if (n_locals < 0 || n_upvalues < 0 || n_upvalues > 255) return -1;
    if (c->n_entries >= c->entry_cap) {
        int new_cap = c->entry_cap * 2;
        ChunkEntry* new_entries =
            (ChunkEntry*)realloc(c->entries, new_cap * sizeof(ChunkEntry));
        if (!new_entries) {
            fprintf(stderr, "ERROR: entry table realloc failed\n");
            return -1;
        }
        memset(new_entries + c->entry_cap, 0,
               (size_t)(new_cap - c->entry_cap) * sizeof(ChunkEntry));
        c->entries = new_entries;
        c->entry_cap = new_cap;
    }
    ChunkEntry* entry = &c->entries[c->n_entries];
    memset(entry, 0, sizeof(*entry));
    entry->name = strdup(name);
    if (!entry->name) return -1;
    entry->n_params = n_params;
    entry->n_locals = n_locals;
    entry->n_upvalues = n_upvalues;
    entry->code_offset = code_offset;
    entry->code_len = code_len;
    c->n_entries++;
    return 0;
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
        /* (define (name ...) body) is an implicit lambda — check params and scan body */
        if (head->type == N_SYMBOL && strcmp(head->symbol, "define") == 0
            && node->n_children >= 3 && node->children[1]->type == N_LIST) {
            Node* sig = node->children[1];
            for (int i = 1; i < sig->n_children; i++)
                if (sig->children[i]->type == N_SYMBOL && strcmp(sig->children[i]->symbol, name) == 0)
                    return 0; /* rebound as parameter */
            for (int i = 2; i < node->n_children; i++)
                if (scan_for_capture(node->children[i], name, 1)) return 1;
            return 0;
        }
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
