/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase U6: Freestanding zero-LLVM, zero-POSIX, zero-malloc in-OS
 * self-hosting Eshkol compiler (ESKB v2 bytecode + standalone W^X ELF64 for
 * x86_64, AArch64, and RISC-V 64).
 *
 * External symbol closure: ZERO external symbols (pure freestanding C11).
 */

#include <eshkol/core/os_self_host.h>

#include <stddef.h>
#include <stdint.h>

/* ── ESKB v2 format constants (matching os_vm_shell.c / eskb_format.h) ── */
#define SH_ESKB_MAGIC            0x45534B42u
#define SH_ESKB_VERSION          2u
#define SH_ESKB_SECTION_CONST    0u
#define SH_ESKB_SECTION_CODE     1u
#define SH_ESKB_CONST_NIL        0u
#define SH_ESKB_CONST_INT64      1u
#define SH_ESKB_CONST_F64        2u
#define SH_ESKB_CONST_BOOL       3u
#define SH_ESKB_CONST_STRING     6u

/* ── Capability host-call IDs (matching os_vm_shell.h) ── */
#define SH_VM_HOST_NATIVE_BASE   100000
#define SH_VM_CALL_CAP_DERIVE    (SH_VM_HOST_NATIVE_BASE + 0)
#define SH_VM_CALL_CAP_REVOKE    (SH_VM_HOST_NATIVE_BASE + 1)
#define SH_VM_CALL_CAP_DROP      (SH_VM_HOST_NATIVE_BASE + 2)
#define SH_VM_CALL_IPC_SUBMIT    (SH_VM_HOST_NATIVE_BASE + 3)
#define SH_VM_CALL_IPC_COMPLETE  (SH_VM_HOST_NATIVE_BASE + 4)
#define SH_VM_CALL_IPC_WAIT      (SH_VM_HOST_NATIVE_BASE + 5)
#define SH_VM_CALL_PAGE_MAP      (SH_VM_HOST_NATIVE_BASE + 6)
#define SH_VM_CALL_PAGE_UNMAP    (SH_VM_HOST_NATIVE_BASE + 7)
#define SH_VM_CALL_TASK_YIELD    (SH_VM_HOST_NATIVE_BASE + 8)
#define SH_VM_CALL_TASK_EXIT     (SH_VM_HOST_NATIVE_BASE + 9)
#define SH_VM_CALL_CONSOLE_WRITE (SH_VM_HOST_NATIVE_BASE + 10)

/* ── ESKB v2 VM opcodes (matching os_vm_shell.c) ── */
enum {
    SH_OP_NOP           = 0,
    SH_OP_CONST         = 1,
    SH_OP_NIL           = 2,
    SH_OP_TRUE          = 3,
    SH_OP_FALSE         = 4,
    SH_OP_POP           = 5,
    SH_OP_DUP           = 6,
    SH_OP_ADD           = 7,
    SH_OP_SUB           = 8,
    SH_OP_MUL           = 9,
    SH_OP_DIV           = 10,
    SH_OP_MOD           = 11,
    SH_OP_NEG           = 12,
    SH_OP_ABS           = 13,
    SH_OP_EQ            = 14,
    SH_OP_LT            = 15,
    SH_OP_GT            = 16,
    SH_OP_LE            = 17,
    SH_OP_GE            = 18,
    SH_OP_NOT           = 19,
    SH_OP_GET_LOCAL     = 20,
    SH_OP_SET_LOCAL     = 21,
    SH_OP_CLOSURE       = 24,
    SH_OP_CALL          = 25,
    SH_OP_TAIL_CALL     = 26,
    SH_OP_RETURN        = 27,
    SH_OP_JUMP          = 28,
    SH_OP_JUMP_IF_FALSE = 29,
    SH_OP_LOOP          = 30,
    SH_OP_CONS          = 31,
    SH_OP_CAR           = 32,
    SH_OP_CDR           = 33,
    SH_OP_NULL_P        = 34,
    SH_OP_PRINT         = 35,
    SH_OP_HALT          = 36,
    SH_OP_NATIVE_CALL   = 37,
    SH_OP_PAIR_P        = 45,
    SH_OP_POPN          = 53,
    SH_OP_VOID          = 63
};

/* ── ELF64 constants ── */
#define SH_ELF_ET_EXEC           2u
#define SH_ELF_EM_X86_64         62u
#define SH_ELF_EM_AARCH64        183u
#define SH_ELF_EM_RISCV          243u
#define SH_ELF_PT_LOAD           1u
#define SH_ELF_PF_X              1u
#define SH_ELF_PF_W              2u
#define SH_ELF_PF_R              4u
#define SH_ELF_SHT_NULL          0u
#define SH_ELF_SHT_PROGBITS      1u
#define SH_ELF_SHT_STRTAB        3u
#define SH_ELF_SHF_WRITE         1ULL
#define SH_ELF_SHF_ALLOC         2ULL
#define SH_ELF_SHF_EXECINSTR     4ULL

/* ── Deterministic bounded table limits (fit strictly inside 64 KiB workspace) ── */
#define SH_ARENA_ALIGN           8u
#define SH_MAX_NODES             384u
#define SH_MAX_STRPOOL           3072u
#define SH_MAX_TOP_EXPRS         64u
#define SH_MAX_FUNCS             48u
#define SH_MAX_PARAMS            16u
#define SH_MAX_LOCALS            48u
#define SH_MAX_CONSTS            256u
#define SH_MAX_ESKB_INSTRS       2048u
#define SH_MAX_CONST_SEC_BYTES   4096u
#define SH_MAX_CODE_SEC_BYTES    12288u
#define SH_MAX_NATIVE_TEXT_BYTES 16384u
#define SH_MAX_NATIVE_PATCHES    256u

/* ── Freestanding byte/string helpers (volatile loops avoid libc lowering) ── */

static void sh_zero(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; i++) {
        p[i] = 0u;
    }
}

static void sh_copy(void* dst, const void* src, size_t n) {
    volatile uint8_t* d = (volatile uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0u; i < n; i++) {
        d[i] = s[i];
    }
}

static int sh_memeq(const char* a, uint32_t alen, const char* b, uint32_t blen) {
    if (alen != blen) return 0;
    for (uint32_t i = 0u; i < alen; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static int sh_streq_lit(const char* a, uint32_t alen, const char* lit) {
    uint32_t i = 0u;
    while (lit[i] != '\0') {
        if (i >= alen || a[i] != lit[i]) return 0;
        i++;
    }
    return i == alen;
}

static uint64_t sh_f64_to_bits(double d) {
    union {
        double d;
        uint64_t u;
    } conv;
    conv.d = d;
    return conv.u;
}

static uint32_t sh_crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0u; i < len; i++) {
        crc ^= (uint32_t)data[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = (crc & 1u) ? 0xEDB88320u : 0u;
            crc = (crc >> 1) ^ mask;
        }
    }
    return ~crc;
}

static void sh_put_u16_le(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static void sh_put_u32_le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void sh_put_u64_le(uint8_t* p, uint64_t v) {
    sh_put_u32_le(p, (uint32_t)(v & 0xFFFFFFFFULL));
    sh_put_u32_le(p + 4u, (uint32_t)((v >> 32) & 0xFFFFFFFFULL));
}

static size_t sh_align_up(size_t val, size_t align) {
    return (val + (align - 1u)) & ~(align - 1u);
}

/* ── Bounded caller-workspace bump arena ── */

typedef struct sh_arena {
    uint8_t* base;
    size_t capacity;
    size_t used;
    size_t peak;
} sh_arena_t;

static int sh_arena_init(sh_arena_t* a, void* workspace, size_t workspace_bytes) {
    if (!a || !workspace || workspace_bytes < ESHKOL_OS_SH_MIN_WORKSPACE_BYTES) {
        return -1;
    }
    uintptr_t raw = (uintptr_t)workspace;
    uintptr_t aligned = (raw + (uintptr_t)(SH_ARENA_ALIGN - 1u)) &
                        ~(uintptr_t)(SH_ARENA_ALIGN - 1u);
    size_t pad = (size_t)(aligned - raw);
    if (pad >= workspace_bytes) return -1;
    a->base = (uint8_t*)aligned;
    a->capacity = workspace_bytes - pad;
    a->used = 0u;
    a->peak = 0u;
    return 0;
}

static void* sh_arena_alloc(sh_arena_t* a, size_t size) {
    if (!a || !a->base) return NULL;
    if (size == 0u) size = SH_ARENA_ALIGN;
    if (size > ((size_t)-1) - (SH_ARENA_ALIGN - 1u)) return NULL;
    size_t aligned = sh_align_up(size, SH_ARENA_ALIGN);
    if (aligned > a->capacity - a->used) return NULL;
    void* ptr = a->base + a->used;
    a->used += aligned;
    if (a->used > a->peak) a->peak = a->used;
    sh_zero(ptr, aligned);
    return ptr;
}

static void sh_arena_rewind(sh_arena_t* a) {
    if (!a || !a->base) return;
    if (a->used > 0u) {
        sh_zero(a->base, a->used);
    }
    a->used = 0u;
}

/* ── AST & Program Representation ── */

typedef enum sh_node_kind {
    SH_NODE_INT    = 1,
    SH_NODE_FLOAT  = 2,
    SH_NODE_BOOL   = 3,
    SH_NODE_NIL    = 4,
    SH_NODE_STRING = 5,
    SH_NODE_SYMBOL = 6,
    SH_NODE_LIST   = 7
} sh_node_kind_t;

typedef struct sh_node {
    uint32_t kind;
    uint32_t line;
    uint32_t col;
    int32_t next_sibling;
    union {
        int64_t iv;
        double fv;
        int32_t bv;
        struct {
            uint32_t off;
            uint32_t len;
        } str;
        struct {
            int32_t first_child;
            uint32_t count;
        } list;
    } as;
} sh_node_t;

typedef struct sh_func_def {
    uint32_t name_off;
    uint32_t name_len;
    uint32_t param_off[SH_MAX_PARAMS];
    uint32_t param_len[SH_MAX_PARAMS];
    uint32_t n_params;
    int32_t body_first;
    uint32_t body_count;
    uint32_t is_main;
    uint32_t is_value_def;
    uint32_t code_offset;
    uint32_t code_len;
    uint32_t n_locals;
    int32_t closure_const_idx;
} sh_func_def_t;

typedef struct sh_prog {
    sh_arena_t arena;
    sh_node_t* nodes;
    uint32_t n_nodes;
    char* strpool;
    uint32_t strpool_used;
    sh_func_def_t* funcs;
    uint32_t n_funcs;
    int32_t* top_exprs;
    uint32_t n_top_exprs;
    uint32_t err_line;
    uint32_t err_col;
} sh_prog_t;

static void sh_set_err(sh_prog_t* p, uint32_t line, uint32_t col) {
    if (p && p->err_line == 0u) {
        p->err_line = line;
        p->err_col = col;
    }
}

static int32_t sh_add_str(sh_prog_t* p, const char* s, uint32_t len, uint32_t* out_off) {
    if (len + 1u > SH_MAX_STRPOOL - p->strpool_used) return -1;
    uint32_t off = p->strpool_used;
    sh_copy(p->strpool + off, s, len);
    p->strpool[off + len] = '\0';
    p->strpool_used += len + 1u;
    *out_off = off;
    return 0;
}

static int32_t sh_new_node(sh_prog_t* p, sh_node_kind_t kind, uint32_t line, uint32_t col) {
    if (p->n_nodes >= SH_MAX_NODES) return -1;
    int32_t idx = (int32_t)p->n_nodes++;
    sh_zero(&p->nodes[idx], sizeof(sh_node_t));
    p->nodes[idx].kind = (uint32_t)kind;
    p->nodes[idx].line = line;
    p->nodes[idx].col = col;
    p->nodes[idx].next_sibling = -1;
    return idx;
}

/* ── S-Expression Lexer & Bounded Recursive-Descent Parser ── */

typedef struct sh_lexer {
    const char* src;
    size_t len;
    size_t pos;
    uint32_t line;
    uint32_t col;
} sh_lexer_t;

static int sh_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static int sh_is_delim(char c) {
    return sh_is_space(c) || c == '(' || c == ')' || c == ';' || c == '"' || c == '\'';
}

static int sh_is_digit(char c) {
    return c >= '0' && c <= '9';
}

static int sh_hex_val(char c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'a' && c <= 'f') return 10 + (int)(c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (int)(c - 'A');
    return -1;
}

static char sh_lex_advance(sh_lexer_t* lx) {
    char c = lx->src[lx->pos++];
    if (c == '\n') {
        lx->line++;
        lx->col = 1u;
    } else {
        lx->col++;
    }
    return c;
}

static void sh_skip_ws(sh_lexer_t* lx) {
    while (lx->pos < lx->len) {
        char c = lx->src[lx->pos];
        if (sh_is_space(c)) {
            (void)sh_lex_advance(lx);
        } else if (c == ';') {
            while (lx->pos < lx->len && lx->src[lx->pos] != '\n') {
                (void)sh_lex_advance(lx);
            }
        } else {
            break;
        }
    }
}

static int sh_try_parse_number(const char* s,
                               uint32_t len,
                               int* is_float,
                               int64_t* out_i,
                               double* out_f) {
    if (len == 0u) return 0;
    uint32_t i = 0u;
    int neg = 0;
    if (s[i] == '-') {
        neg = 1;
        i++;
    } else if (s[i] == '+') {
        i++;
    }
    if (i >= len) return 0;

    /* Hex integer literal: 0x... / 0X... */
    if (i + 2u < len && s[i] == '0' && (s[i + 1u] == 'x' || s[i + 1u] == 'X')) {
        i += 2u;
        uint64_t acc = 0u;
        uint32_t digits = 0u;
        while (i < len) {
            int hv = sh_hex_val(s[i]);
            if (hv < 0) return 0;
            if (acc > (0xFFFFFFFFFFFFFFFFULL >> 4)) return 0;
            acc = (acc << 4) | (uint64_t)hv;
            digits++;
            i++;
        }
        if (digits == 0u) return 0;
        *is_float = 0;
        *out_i = neg ? -(int64_t)acc : (int64_t)acc;
        return 1;
    }

    uint64_t int_part = 0u;
    uint32_t int_digits = 0u;
    int int_overflow = 0;
    while (i < len && sh_is_digit(s[i])) {
        uint64_t d = (uint64_t)(s[i] - '0');
        if (int_part > (9223372036854775807ULL - d) / 10ULL) {
            int_overflow = 1;
        } else {
            int_part = int_part * 10ULL + d;
        }
        int_digits++;
        i++;
    }

    int has_dot = 0;
    double frac_val = 0.0;
    double frac_scale = 0.1;
    uint32_t frac_digits = 0u;
    if (i < len && s[i] == '.') {
        has_dot = 1;
        i++;
        while (i < len && sh_is_digit(s[i])) {
            frac_val += (double)(s[i] - '0') * frac_scale;
            frac_scale *= 0.1;
            frac_digits++;
            i++;
        }
    }

    if (int_digits == 0u && frac_digits == 0u) return 0;

    int has_exp = 0;
    int exp_neg = 0;
    int32_t exp_val = 0;
    if (i < len && (s[i] == 'e' || s[i] == 'E')) {
        has_exp = 1;
        i++;
        if (i < len && s[i] == '-') {
            exp_neg = 1;
            i++;
        } else if (i < len && s[i] == '+') {
            i++;
        }
        uint32_t exp_digits = 0u;
        while (i < len && sh_is_digit(s[i])) {
            if (exp_val < 1000) exp_val = exp_val * 10 + (int32_t)(s[i] - '0');
            exp_digits++;
            i++;
        }
        if (exp_digits == 0u) return 0;
    }

    if (i != len) return 0;

    if (!has_dot && !has_exp) {
        if (int_overflow) {
            /* Special case: INT64_MIN (-9223372036854775808) */
            if (neg && sh_streq_lit(s, len, "-9223372036854775808")) {
                *is_float = 0;
                *out_i = (-9223372036854775807LL - 1LL);
                return 1;
            }
            return 0;
        }
        *is_float = 0;
        *out_i = neg ? -(int64_t)int_part : (int64_t)int_part;
        return 1;
    }

    /* Recompute float value accurately from digits */
    double val = (double)int_part + frac_val;
    if (has_exp) {
        double p10 = 1.0;
        for (int32_t k = 0; k < exp_val; k++) p10 *= 10.0;
        val = exp_neg ? (val / p10) : (val * p10);
    }
    *is_float = 1;
    *out_f = neg ? -val : val;
    return 1;
}

static eshkol_os_self_host_status_t sh_parse_expr(sh_prog_t* p,
                                                  sh_lexer_t* lx,
                                                  uint32_t depth,
                                                  int32_t* out_node) {
    sh_skip_ws(lx);
    if (lx->pos >= lx->len) {
        sh_set_err(p, lx->line, lx->col);
        return ESHKOL_SH_ERR_SYNTAX;
    }
    if (depth > ESHKOL_OS_SH_MAX_PARSE_DEPTH) {
        sh_set_err(p, lx->line, lx->col);
        return ESHKOL_SH_ERR_DEPTH_EXCEEDED;
    }

    uint32_t start_line = lx->line;
    uint32_t start_col = lx->col;
    char c = lx->src[lx->pos];

    /* List: ( ... ) */
    if (c == '(') {
        (void)sh_lex_advance(lx);
        int32_t list_idx = sh_new_node(p, SH_NODE_LIST, start_line, start_col);
        if (list_idx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
        p->nodes[list_idx].as.list.first_child = -1;
        p->nodes[list_idx].as.list.count = 0u;

        int32_t last_child = -1;
        for (;;) {
            sh_skip_ws(lx);
            if (lx->pos >= lx->len) {
                sh_set_err(p, start_line, start_col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            if (lx->src[lx->pos] == ')') {
                (void)sh_lex_advance(lx);
                break;
            }
            int32_t child = -1;
            eshkol_os_self_host_status_t st = sh_parse_expr(p, lx, depth + 1u, &child);
            if (st != ESHKOL_SH_OK) return st;
            if (last_child < 0) {
                p->nodes[list_idx].as.list.first_child = child;
            } else {
                p->nodes[last_child].next_sibling = child;
            }
            last_child = child;
            p->nodes[list_idx].as.list.count++;
        }
        *out_node = list_idx;
        return ESHKOL_SH_OK;
    }

    if (c == ')') {
        sh_set_err(p, start_line, start_col);
        return ESHKOL_SH_ERR_SYNTAX;
    }

    /* Quoted empty list: '() */
    if (c == '\'') {
        (void)sh_lex_advance(lx);
        if (lx->pos + 1u < lx->len && lx->src[lx->pos] == '(' && lx->src[lx->pos + 1u] == ')') {
            (void)sh_lex_advance(lx);
            (void)sh_lex_advance(lx);
            int32_t nil_idx = sh_new_node(p, SH_NODE_NIL, start_line, start_col);
            if (nil_idx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
            *out_node = nil_idx;
            return ESHKOL_SH_OK;
        }
        sh_set_err(p, start_line, start_col);
        return ESHKOL_SH_ERR_UNSUPPORTED_FORM;
    }

    /* String literal: "..." */
    if (c == '"') {
        (void)sh_lex_advance(lx);
        char tmp[256];
        uint32_t tlen = 0u;
        int closed = 0;
        while (lx->pos < lx->len) {
            char ch = sh_lex_advance(lx);
            if (ch == '"') {
                closed = 1;
                break;
            }
            if (ch == '\\') {
                if (lx->pos >= lx->len) {
                    sh_set_err(p, lx->line, lx->col);
                    return ESHKOL_SH_ERR_SYNTAX;
                }
                char esc = sh_lex_advance(lx);
                if (esc == 'n') ch = '\n';
                else if (esc == 'r') ch = '\r';
                else if (esc == 't') ch = '\t';
                else if (esc == '\\') ch = '\\';
                else if (esc == '"') ch = '"';
                else ch = esc;
            }
            if (tlen >= sizeof(tmp) - 1u) {
                sh_set_err(p, start_line, start_col);
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            tmp[tlen++] = ch;
        }
        if (!closed) {
            sh_set_err(p, start_line, start_col);
            return ESHKOL_SH_ERR_SYNTAX;
        }
        uint32_t soff = 0u;
        if (sh_add_str(p, tmp, tlen, &soff) < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
        int32_t sidx = sh_new_node(p, SH_NODE_STRING, start_line, start_col);
        if (sidx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
        p->nodes[sidx].as.str.off = soff;
        p->nodes[sidx].as.str.len = tlen;
        *out_node = sidx;
        return ESHKOL_SH_OK;
    }

    /* Atom: boolean, number, or symbol */
    size_t tok_start = lx->pos;
    while (lx->pos < lx->len && !sh_is_delim(lx->src[lx->pos])) {
        (void)sh_lex_advance(lx);
    }
    uint32_t tok_len = (uint32_t)(lx->pos - tok_start);
    const char* tok = lx->src + tok_start;
    if (tok_len == 0u) {
        sh_set_err(p, start_line, start_col);
        return ESHKOL_SH_ERR_SYNTAX;
    }

    if (sh_streq_lit(tok, tok_len, "#t") || sh_streq_lit(tok, tok_len, "#true")) {
        int32_t bidx = sh_new_node(p, SH_NODE_BOOL, start_line, start_col);
        if (bidx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
        p->nodes[bidx].as.bv = 1;
        *out_node = bidx;
        return ESHKOL_SH_OK;
    }
    if (sh_streq_lit(tok, tok_len, "#f") || sh_streq_lit(tok, tok_len, "#false")) {
        int32_t bidx = sh_new_node(p, SH_NODE_BOOL, start_line, start_col);
        if (bidx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
        p->nodes[bidx].as.bv = 0;
        *out_node = bidx;
        return ESHKOL_SH_OK;
    }

    int is_float = 0;
    int64_t iv = 0;
    double fv = 0.0;
    if (sh_try_parse_number(tok, tok_len, &is_float, &iv, &fv)) {
        if (is_float) {
            int32_t fidx = sh_new_node(p, SH_NODE_FLOAT, start_line, start_col);
            if (fidx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
            p->nodes[fidx].as.fv = fv;
            *out_node = fidx;
        } else {
            int32_t iidx = sh_new_node(p, SH_NODE_INT, start_line, start_col);
            if (iidx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
            p->nodes[iidx].as.iv = iv;
            *out_node = iidx;
        }
        return ESHKOL_SH_OK;
    }

    if (tok_len > 255u) {
        sh_set_err(p, start_line, start_col);
        return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
    }
    uint32_t soff = 0u;
    if (sh_add_str(p, tok, tok_len, &soff) < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
    int32_t sym_idx = sh_new_node(p, SH_NODE_SYMBOL, start_line, start_col);
    if (sym_idx < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
    p->nodes[sym_idx].as.str.off = soff;
    p->nodes[sym_idx].as.str.len = tok_len;
    *out_node = sym_idx;
    return ESHKOL_SH_OK;
}

static int sh_find_func(const sh_prog_t* p, const char* name, uint32_t len) {
    for (uint32_t i = 0u; i < p->n_funcs; i++) {
        const char* fn = p->strpool + p->funcs[i].name_off;
        if (sh_memeq(fn, p->funcs[i].name_len, name, len)) {
            return (int)i;
        }
    }
    return -1;
}

static int sh_node_is_sym(const sh_prog_t* p, int32_t idx, const char* lit) {
    if (idx < 0 || p->nodes[idx].kind != SH_NODE_SYMBOL) return 0;
    const char* s = p->strpool + p->nodes[idx].as.str.off;
    return sh_streq_lit(s, p->nodes[idx].as.str.len, lit);
}

static int sh_lookup_syscall(const sh_prog_t* p,
                             int32_t node_idx,
                             uint32_t* out_sysno,
                             int32_t* out_vm_fid,
                             uint32_t* out_arity) {
    int64_t num = -1;
    if (p->nodes[node_idx].kind == SH_NODE_INT) {
        num = p->nodes[node_idx].as.iv;
    } else if (p->nodes[node_idx].kind == SH_NODE_SYMBOL) {
        const char* s = p->strpool + p->nodes[node_idx].as.str.off;
        uint32_t len = p->nodes[node_idx].as.str.len;
        if (sh_streq_lit(s, len, "cap-derive"))         num = ESHKOL_SYS_CAP_DERIVE;
        else if (sh_streq_lit(s, len, "cap-revoke"))    num = ESHKOL_SYS_CAP_REVOKE;
        else if (sh_streq_lit(s, len, "cap-drop"))      num = ESHKOL_SYS_CAP_DROP;
        else if (sh_streq_lit(s, len, "ipc-submit"))    num = ESHKOL_SYS_IPC_SUBMIT;
        else if (sh_streq_lit(s, len, "ipc-complete"))  num = ESHKOL_SYS_IPC_COMPLETE;
        else if (sh_streq_lit(s, len, "ipc-wait"))      num = ESHKOL_SYS_IPC_WAIT;
        else if (sh_streq_lit(s, len, "page-map"))      num = ESHKOL_SYS_PAGE_MAP;
        else if (sh_streq_lit(s, len, "page-unmap"))    num = ESHKOL_SYS_PAGE_UNMAP;
        else if (sh_streq_lit(s, len, "task-yield"))    num = ESHKOL_SYS_TASK_YIELD;
        else if (sh_streq_lit(s, len, "task-exit"))     num = ESHKOL_SYS_TASK_EXIT;
        else if (sh_streq_lit(s, len, "console-write")) num = ESHKOL_SYS_CONSOLE_WRITE;
    }

    switch (num) {
    case ESHKOL_SYS_CAP_DERIVE:
        *out_sysno = ESHKOL_SYS_CAP_DERIVE;
        *out_vm_fid = SH_VM_CALL_CAP_DERIVE;
        *out_arity = 3u;
        return 1;
    case ESHKOL_SYS_CAP_REVOKE:
        *out_sysno = ESHKOL_SYS_CAP_REVOKE;
        *out_vm_fid = SH_VM_CALL_CAP_REVOKE;
        *out_arity = 1u;
        return 1;
    case ESHKOL_SYS_CAP_DROP:
        *out_sysno = ESHKOL_SYS_CAP_DROP;
        *out_vm_fid = SH_VM_CALL_CAP_DROP;
        *out_arity = 1u;
        return 1;
    case ESHKOL_SYS_IPC_SUBMIT:
        *out_sysno = ESHKOL_SYS_IPC_SUBMIT;
        *out_vm_fid = SH_VM_CALL_IPC_SUBMIT;
        *out_arity = 4u;
        return 1;
    case ESHKOL_SYS_IPC_COMPLETE:
        *out_sysno = ESHKOL_SYS_IPC_COMPLETE;
        *out_vm_fid = SH_VM_CALL_IPC_COMPLETE;
        *out_arity = 4u;
        return 1;
    case ESHKOL_SYS_IPC_WAIT:
        *out_sysno = ESHKOL_SYS_IPC_WAIT;
        *out_vm_fid = SH_VM_CALL_IPC_WAIT;
        *out_arity = 3u;
        return 1;
    case ESHKOL_SYS_PAGE_MAP:
        *out_sysno = ESHKOL_SYS_PAGE_MAP;
        *out_vm_fid = SH_VM_CALL_PAGE_MAP;
        *out_arity = 4u;
        return 1;
    case ESHKOL_SYS_PAGE_UNMAP:
        *out_sysno = ESHKOL_SYS_PAGE_UNMAP;
        *out_vm_fid = SH_VM_CALL_PAGE_UNMAP;
        *out_arity = 3u;
        return 1;
    case ESHKOL_SYS_TASK_YIELD:
        *out_sysno = ESHKOL_SYS_TASK_YIELD;
        *out_vm_fid = SH_VM_CALL_TASK_YIELD;
        *out_arity = 0u;
        return 1;
    case ESHKOL_SYS_TASK_EXIT:
        *out_sysno = ESHKOL_SYS_TASK_EXIT;
        *out_vm_fid = SH_VM_CALL_TASK_EXIT;
        *out_arity = 1u;
        return 1;
    case ESHKOL_SYS_CONSOLE_WRITE:
        *out_sysno = ESHKOL_SYS_CONSOLE_WRITE;
        *out_vm_fid = SH_VM_CALL_CONSOLE_WRITE;
        *out_arity = 3u;
        return 1;
    default:
        return 0;
    }
}

static eshkol_os_self_host_status_t sh_parse_program(sh_prog_t* p,
                                                     const char* source,
                                                     size_t source_len,
                                                     void* workspace,
                                                     size_t workspace_bytes) {
    sh_zero(p, sizeof(*p));
    if (sh_arena_init(&p->arena, workspace, workspace_bytes) < 0) {
        return ESHKOL_SH_ERR_INVALID_ARG;
    }

    p->nodes = (sh_node_t*)sh_arena_alloc(&p->arena, SH_MAX_NODES * sizeof(sh_node_t));
    p->strpool = (char*)sh_arena_alloc(&p->arena, SH_MAX_STRPOOL);
    p->funcs = (sh_func_def_t*)sh_arena_alloc(&p->arena, SH_MAX_FUNCS * sizeof(sh_func_def_t));
    p->top_exprs = (int32_t*)sh_arena_alloc(&p->arena, SH_MAX_TOP_EXPRS * sizeof(int32_t));
    if (!p->nodes || !p->strpool || !p->funcs || !p->top_exprs) {
        return ESHKOL_SH_ERR_WORKSPACE_OOM;
    }

    /* Reserve slot 0 for "main" entry function */
    uint32_t main_off = 0u;
    if (sh_add_str(p, "main", 4u, &main_off) < 0) return ESHKOL_SH_ERR_WORKSPACE_OOM;
    p->funcs[0].name_off = main_off;
    p->funcs[0].name_len = 4u;
    p->funcs[0].is_main = 1u;
    p->funcs[0].body_first = -1;
    p->funcs[0].closure_const_idx = -1;
    p->n_funcs = 1u;

    sh_lexer_t lx;
    lx.src = source;
    lx.len = source_len;
    lx.pos = 0u;
    lx.line = 1u;
    lx.col = 1u;

    int has_explicit_main = 0;

    for (;;) {
        sh_skip_ws(&lx);
        if (lx.pos >= lx.len) break;

        int32_t form = -1;
        eshkol_os_self_host_status_t st = sh_parse_expr(p, &lx, 0u, &form);
        if (st != ESHKOL_SH_OK) return st;

        if (p->nodes[form].kind == SH_NODE_LIST &&
            p->nodes[form].as.list.count >= 1u &&
            sh_node_is_sym(p, p->nodes[form].as.list.first_child, "define")) {
            uint32_t fcnt = p->nodes[form].as.list.count;
            if (fcnt < 3u) {
                sh_set_err(p, p->nodes[form].line, p->nodes[form].col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            int32_t sig = p->nodes[p->nodes[form].as.list.first_child].next_sibling;
            int32_t body_first = p->nodes[sig].next_sibling;
            uint32_t body_count = fcnt - 2u;

            if (p->nodes[sig].kind == SH_NODE_LIST) {
                if (p->nodes[sig].as.list.count < 1u) {
                    sh_set_err(p, p->nodes[sig].line, p->nodes[sig].col);
                    return ESHKOL_SH_ERR_SYNTAX;
                }
                int32_t fname_node = p->nodes[sig].as.list.first_child;
                if (p->nodes[fname_node].kind != SH_NODE_SYMBOL) {
                    sh_set_err(p, p->nodes[fname_node].line, p->nodes[fname_node].col);
                    return ESHKOL_SH_ERR_SYNTAX;
                }
                const char* fn_str = p->strpool + p->nodes[fname_node].as.str.off;
                uint32_t fn_len = p->nodes[fname_node].as.str.len;
                uint32_t n_params = p->nodes[sig].as.list.count - 1u;
                if (n_params > SH_MAX_PARAMS) {
                    sh_set_err(p, p->nodes[sig].line, p->nodes[sig].col);
                    return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                }

                uint32_t target_idx = 0u;
                if (sh_streq_lit(fn_str, fn_len, "main")) {
                    if (has_explicit_main || n_params != 0u) {
                        sh_set_err(p, p->nodes[fname_node].line, p->nodes[fname_node].col);
                        return has_explicit_main ? ESHKOL_SH_ERR_DUPLICATE_SYMBOL
                                                 : ESHKOL_SH_ERR_ARITY;
                    }
                    has_explicit_main = 1;
                    target_idx = 0u;
                } else {
                    if (sh_find_func(p, fn_str, fn_len) >= 0) {
                        sh_set_err(p, p->nodes[fname_node].line, p->nodes[fname_node].col);
                        return ESHKOL_SH_ERR_DUPLICATE_SYMBOL;
                    }
                    if (p->n_funcs >= SH_MAX_FUNCS) {
                        sh_set_err(p, p->nodes[fname_node].line, p->nodes[fname_node].col);
                        return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                    }
                    target_idx = p->n_funcs++;
                    sh_zero(&p->funcs[target_idx], sizeof(sh_func_def_t));
                    p->funcs[target_idx].name_off = p->nodes[fname_node].as.str.off;
                    p->funcs[target_idx].name_len = fn_len;
                    p->funcs[target_idx].closure_const_idx = -1;
                }

                p->funcs[target_idx].n_params = n_params;
                p->funcs[target_idx].body_first = body_first;
                p->funcs[target_idx].body_count = body_count;
                int32_t param = p->nodes[fname_node].next_sibling;
                for (uint32_t pi = 0u; pi < n_params; pi++) {
                    if (param < 0 || p->nodes[param].kind != SH_NODE_SYMBOL) {
                        sh_set_err(p, p->nodes[sig].line, p->nodes[sig].col);
                        return ESHKOL_SH_ERR_SYNTAX;
                    }
                    p->funcs[target_idx].param_off[pi] = p->nodes[param].as.str.off;
                    p->funcs[target_idx].param_len[pi] = p->nodes[param].as.str.len;
                    param = p->nodes[param].next_sibling;
                }
            } else if (p->nodes[sig].kind == SH_NODE_SYMBOL) {
                if (body_count != 1u) {
                    sh_set_err(p, p->nodes[form].line, p->nodes[form].col);
                    return ESHKOL_SH_ERR_SYNTAX;
                }
                const char* vn_str = p->strpool + p->nodes[sig].as.str.off;
                uint32_t vn_len = p->nodes[sig].as.str.len;
                if (sh_find_func(p, vn_str, vn_len) >= 0) {
                    sh_set_err(p, p->nodes[sig].line, p->nodes[sig].col);
                    return ESHKOL_SH_ERR_DUPLICATE_SYMBOL;
                }
                if (p->n_funcs >= SH_MAX_FUNCS) {
                    sh_set_err(p, p->nodes[sig].line, p->nodes[sig].col);
                    return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                }
                uint32_t idx = p->n_funcs++;
                sh_zero(&p->funcs[idx], sizeof(sh_func_def_t));
                p->funcs[idx].name_off = p->nodes[sig].as.str.off;
                p->funcs[idx].name_len = vn_len;
                p->funcs[idx].n_params = 0u;
                p->funcs[idx].body_first = body_first;
                p->funcs[idx].body_count = 1u;
                p->funcs[idx].is_value_def = 1u;
                p->funcs[idx].closure_const_idx = -1;
            } else {
                sh_set_err(p, p->nodes[sig].line, p->nodes[sig].col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
        } else {
            if (p->n_top_exprs >= SH_MAX_TOP_EXPRS) {
                sh_set_err(p, p->nodes[form].line, p->nodes[form].col);
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            p->top_exprs[p->n_top_exprs++] = form;
        }
    }

    if (p->n_top_exprs == 0u && !has_explicit_main && p->n_funcs == 1u) {
        sh_set_err(p, 1u, 1u);
        return ESHKOL_SH_ERR_SYNTAX;
    }

    return ESHKOL_SH_OK;
}

/* ── ESKB v2 Bytecode Emitter ── */

typedef struct sh_const_entry {
    uint8_t type;
    uint8_t is_func_desc;
    uint8_t pad[2];
    uint32_t str_len;
    union {
        int64_t iv;
        uint64_t fbits;
        uint8_t bv;
        uint32_t str_off;
    } as;
} sh_const_entry_t;

typedef struct sh_eskb_instr {
    uint8_t op;
    uint8_t pad0;
    uint16_t pad1;
    int32_t operand;
} sh_eskb_instr_t;

typedef struct sh_local_var {
    uint32_t name_off;
    uint32_t name_len;
    uint32_t slot;
} sh_local_var_t;

typedef struct sh_eskb_ctx {
    sh_prog_t* prog;
    sh_const_entry_t* consts;
    uint32_t n_consts;
    sh_eskb_instr_t* code;
    uint32_t code_len;
    uint32_t cur_func_idx;
    uint32_t cur_func_start;
    sh_local_var_t locals[SH_MAX_LOCALS];
    uint32_t n_locals;
    uint32_t stack_depth;
    uint32_t max_locals;
    uint32_t lifted_counter;
} sh_eskb_ctx_t;

static int32_t sh_eskb_add_const_int(sh_eskb_ctx_t* c, int64_t val, int dedup) {
    if (dedup) {
        for (uint32_t i = 0u; i < c->n_consts; i++) {
            if (c->consts[i].type == SH_ESKB_CONST_INT64 &&
                !c->consts[i].is_func_desc &&
                c->consts[i].as.iv == val) {
                return (int32_t)i;
            }
        }
    }
    if (c->n_consts >= SH_MAX_CONSTS) return -1;
    uint32_t idx = c->n_consts++;
    sh_zero(&c->consts[idx], sizeof(sh_const_entry_t));
    c->consts[idx].type = SH_ESKB_CONST_INT64;
    c->consts[idx].is_func_desc = dedup ? 0u : 1u;
    c->consts[idx].as.iv = val;
    return (int32_t)idx;
}

static int32_t sh_eskb_add_const_f64(sh_eskb_ctx_t* c, double val) {
    uint64_t bits = sh_f64_to_bits(val);
    for (uint32_t i = 0u; i < c->n_consts; i++) {
        if (c->consts[i].type == SH_ESKB_CONST_F64 && c->consts[i].as.fbits == bits) {
            return (int32_t)i;
        }
    }
    if (c->n_consts >= SH_MAX_CONSTS) return -1;
    uint32_t idx = c->n_consts++;
    sh_zero(&c->consts[idx], sizeof(sh_const_entry_t));
    c->consts[idx].type = SH_ESKB_CONST_F64;
    c->consts[idx].as.fbits = bits;
    return (int32_t)idx;
}

static int32_t sh_eskb_add_const_str(sh_eskb_ctx_t* c, uint32_t off, uint32_t len) {
    for (uint32_t i = 0u; i < c->n_consts; i++) {
        if (c->consts[i].type == SH_ESKB_CONST_STRING &&
            c->consts[i].str_len == len &&
            sh_memeq(c->prog->strpool + c->consts[i].as.str_off, len,
                     c->prog->strpool + off, len)) {
            return (int32_t)i;
        }
    }
    if (c->n_consts >= SH_MAX_CONSTS) return -1;
    uint32_t idx = c->n_consts++;
    sh_zero(&c->consts[idx], sizeof(sh_const_entry_t));
    c->consts[idx].type = SH_ESKB_CONST_STRING;
    c->consts[idx].as.str_off = off;
    c->consts[idx].str_len = len;
    return (int32_t)idx;
}

static int32_t sh_eskb_emit(sh_eskb_ctx_t* c, uint8_t op, int32_t operand) {
    if (c->code_len >= SH_MAX_ESKB_INSTRS) return -1;
    uint32_t idx = c->code_len++;
    c->code[idx].op = op;
    c->code[idx].pad0 = 0u;
    c->code[idx].pad1 = 0u;
    c->code[idx].operand = operand;
    return (int32_t)(idx - c->cur_func_start);
}

static int sh_eskb_find_local(const sh_eskb_ctx_t* c, const char* name, uint32_t len) {
    for (int i = (int)c->n_locals - 1; i >= 0; i--) {
        const char* ln = c->prog->strpool + c->locals[i].name_off;
        if (sh_memeq(ln, c->locals[i].name_len, name, len)) {
            return (int)c->locals[i].slot;
        }
    }
    return -1;
}

static int32_t sh_eskb_func_const(sh_eskb_ctx_t* c, uint32_t fi) {
    if (c->prog->funcs[fi].closure_const_idx < 0) {
        /* Dedicated non-deduplicated constant slot patched once code offsets are final */
        int32_t ci = sh_eskb_add_const_int(c, 0, 0);
        if (ci < 0) return -1;
        c->prog->funcs[fi].closure_const_idx = ci;
    }
    return c->prog->funcs[fi].closure_const_idx;
}

static eshkol_os_self_host_status_t sh_eskb_compile_expr(sh_eskb_ctx_t* c,
                                                         int32_t node_idx,
                                                         int is_tail);

static eshkol_os_self_host_status_t sh_eskb_compile_seq(sh_eskb_ctx_t* c,
                                                        int32_t first,
                                                        uint32_t count,
                                                        int is_tail) {
    if (count == 0u) {
        if (sh_eskb_emit(c, SH_OP_VOID, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        c->stack_depth++;
        return ESHKOL_SH_OK;
    }
    int32_t cur = first;
    for (uint32_t i = 0u; i < count; i++) {
        int last = (i + 1u == count);
        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, last ? is_tail : 0);
        if (st != ESHKOL_SH_OK) return st;
        if (!last) {
            if (sh_eskb_emit(c, SH_OP_POP, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            c->stack_depth--;
        }
        cur = c->prog->nodes[cur].next_sibling;
    }
    return ESHKOL_SH_OK;
}

static eshkol_os_self_host_status_t sh_eskb_compile_let(sh_eskb_ctx_t* c,
                                                        int32_t list_idx,
                                                        int is_star,
                                                        int is_tail) {
    sh_prog_t* p = c->prog;
    int32_t head = p->nodes[list_idx].as.list.first_child;
    uint32_t total = p->nodes[list_idx].as.list.count;
    if (total < 3u) {
        sh_set_err(p, p->nodes[list_idx].line, p->nodes[list_idx].col);
        return ESHKOL_SH_ERR_SYNTAX;
    }
    int32_t second = p->nodes[head].next_sibling;

    /* Named let: (let loop ((x1 e1) ...) body...) */
    if (!is_star && p->nodes[second].kind == SH_NODE_SYMBOL) {
        if (total < 4u) {
            sh_set_err(p, p->nodes[list_idx].line, p->nodes[list_idx].col);
            return ESHKOL_SH_ERR_SYNTAX;
        }
        int32_t binds = p->nodes[second].next_sibling;
        if (p->nodes[binds].kind != SH_NODE_LIST) {
            sh_set_err(p, p->nodes[binds].line, p->nodes[binds].col);
            return ESHKOL_SH_ERR_SYNTAX;
        }
        uint32_t n_binds = p->nodes[binds].as.list.count;
        if (n_binds > SH_MAX_PARAMS) {
            sh_set_err(p, p->nodes[binds].line, p->nodes[binds].col);
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        if (p->n_funcs >= SH_MAX_FUNCS) {
            sh_set_err(p, p->nodes[list_idx].line, p->nodes[list_idx].col);
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        /* Create lifted helper function named by the named-let loop symbol */
        uint32_t fi = p->n_funcs++;
        sh_zero(&p->funcs[fi], sizeof(sh_func_def_t));
        p->funcs[fi].name_off = p->nodes[second].as.str.off;
        p->funcs[fi].name_len = p->nodes[second].as.str.len;
        p->funcs[fi].n_params = n_binds;
        p->funcs[fi].body_first = p->nodes[binds].next_sibling;
        p->funcs[fi].body_count = total - 3u;
        p->funcs[fi].closure_const_idx = -1;

        int32_t fci = sh_eskb_func_const(c, fi);
        if (fci < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        if (sh_eskb_emit(c, SH_OP_CLOSURE, fci) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        c->stack_depth++;

        int32_t bnode = p->nodes[binds].as.list.first_child;
        for (uint32_t i = 0u; i < n_binds; i++) {
            if (p->nodes[bnode].kind != SH_NODE_LIST || p->nodes[bnode].as.list.count != 2u) {
                sh_set_err(p, p->nodes[bnode].line, p->nodes[bnode].col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            int32_t bname = p->nodes[bnode].as.list.first_child;
            int32_t bval = p->nodes[bname].next_sibling;
            if (p->nodes[bname].kind != SH_NODE_SYMBOL) {
                sh_set_err(p, p->nodes[bname].line, p->nodes[bname].col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            p->funcs[fi].param_off[i] = p->nodes[bname].as.str.off;
            p->funcs[fi].param_len[i] = p->nodes[bname].as.str.len;
            eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, bval, 0);
            if (st != ESHKOL_SH_OK) return st;
            bnode = p->nodes[bnode].next_sibling;
        }
        if (sh_eskb_emit(c, is_tail ? SH_OP_TAIL_CALL : SH_OP_CALL, (int32_t)n_binds) < 0) {
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        c->stack_depth -= n_binds;
        return ESHKOL_SH_OK;
    }

    int32_t binds = second;
    if (p->nodes[binds].kind != SH_NODE_LIST) {
        sh_set_err(p, p->nodes[binds].line, p->nodes[binds].col);
        return ESHKOL_SH_ERR_SYNTAX;
    }
    uint32_t n_binds = p->nodes[binds].as.list.count;
    if (c->n_locals + n_binds > SH_MAX_LOCALS) {
        sh_set_err(p, p->nodes[binds].line, p->nodes[binds].col);
        return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
    }

    uint32_t saved_locals = c->n_locals;
    uint32_t base_slot = c->stack_depth;
    uint32_t pending_off[SH_MAX_LOCALS];
    uint32_t pending_len[SH_MAX_LOCALS];

    int32_t bnode = p->nodes[binds].as.list.first_child;
    for (uint32_t i = 0u; i < n_binds; i++) {
        if (p->nodes[bnode].kind != SH_NODE_LIST || p->nodes[bnode].as.list.count != 2u) {
            sh_set_err(p, p->nodes[bnode].line, p->nodes[bnode].col);
            return ESHKOL_SH_ERR_SYNTAX;
        }
        int32_t bname = p->nodes[bnode].as.list.first_child;
        int32_t bval = p->nodes[bname].next_sibling;
        if (p->nodes[bname].kind != SH_NODE_SYMBOL) {
            sh_set_err(p, p->nodes[bname].line, p->nodes[bname].col);
            return ESHKOL_SH_ERR_SYNTAX;
        }
        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, bval, 0);
        if (st != ESHKOL_SH_OK) return st;

        if (is_star) {
            c->locals[c->n_locals].name_off = p->nodes[bname].as.str.off;
            c->locals[c->n_locals].name_len = p->nodes[bname].as.str.len;
            c->locals[c->n_locals].slot = base_slot + i;
            c->n_locals++;
            if (c->n_locals > c->max_locals) c->max_locals = c->n_locals;
        } else {
            pending_off[i] = p->nodes[bname].as.str.off;
            pending_len[i] = p->nodes[bname].as.str.len;
        }
        bnode = p->nodes[bnode].next_sibling;
    }

    if (!is_star) {
        for (uint32_t i = 0u; i < n_binds; i++) {
            c->locals[c->n_locals].name_off = pending_off[i];
            c->locals[c->n_locals].name_len = pending_len[i];
            c->locals[c->n_locals].slot = base_slot + i;
            c->n_locals++;
            if (c->n_locals > c->max_locals) c->max_locals = c->n_locals;
        }
    }

    eshkol_os_self_host_status_t st = sh_eskb_compile_seq(
        c, p->nodes[binds].next_sibling, total - 2u, is_tail);
    c->n_locals = saved_locals;
    if (st != ESHKOL_SH_OK) return st;

    if (n_binds > 0u) {
        if (sh_eskb_emit(c, SH_OP_POPN, (int32_t)n_binds) < 0) {
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        c->stack_depth -= n_binds;
    }
    return ESHKOL_SH_OK;
}

static eshkol_os_self_host_status_t sh_eskb_compile_expr(sh_eskb_ctx_t* c,
                                                         int32_t node_idx,
                                                         int is_tail) {
    sh_prog_t* p = c->prog;
    const sh_node_t* n = &p->nodes[node_idx];

    switch (n->kind) {
    case SH_NODE_INT: {
        int32_t ci = sh_eskb_add_const_int(c, n->as.iv, 1);
        if (ci < 0 || sh_eskb_emit(c, SH_OP_CONST, ci) < 0) {
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        c->stack_depth++;
        return ESHKOL_SH_OK;
    }
    case SH_NODE_FLOAT: {
        int32_t ci = sh_eskb_add_const_f64(c, n->as.fv);
        if (ci < 0 || sh_eskb_emit(c, SH_OP_CONST, ci) < 0) {
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        c->stack_depth++;
        return ESHKOL_SH_OK;
    }
    case SH_NODE_BOOL:
        if (sh_eskb_emit(c, n->as.bv ? SH_OP_TRUE : SH_OP_FALSE, 0) < 0) {
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        c->stack_depth++;
        return ESHKOL_SH_OK;
    case SH_NODE_NIL:
        if (sh_eskb_emit(c, SH_OP_NIL, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        c->stack_depth++;
        return ESHKOL_SH_OK;
    case SH_NODE_STRING: {
        int32_t ci = sh_eskb_add_const_str(c, n->as.str.off, n->as.str.len);
        if (ci < 0 || sh_eskb_emit(c, SH_OP_CONST, ci) < 0) {
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        c->stack_depth++;
        return ESHKOL_SH_OK;
    }
    case SH_NODE_SYMBOL: {
        const char* s = p->strpool + n->as.str.off;
        uint32_t len = n->as.str.len;
        int slot = sh_eskb_find_local(c, s, len);
        if (slot >= 0) {
            if (sh_eskb_emit(c, SH_OP_GET_LOCAL, (int32_t)slot) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            c->stack_depth++;
            return ESHKOL_SH_OK;
        }
        int fi = sh_find_func(p, s, len);
        if (fi >= 0 && p->funcs[fi].is_value_def) {
            int32_t fci = sh_eskb_func_const(c, (uint32_t)fi);
            if (fci < 0 ||
                sh_eskb_emit(c, SH_OP_CLOSURE, fci) < 0 ||
                sh_eskb_emit(c, SH_OP_CALL, 0) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            c->stack_depth++;
            return ESHKOL_SH_OK;
        }
        sh_set_err(p, n->line, n->col);
        return ESHKOL_SH_ERR_UNBOUND_SYMBOL;
    }
    case SH_NODE_LIST:
        break;
    default:
        return ESHKOL_SH_ERR_SYNTAX;
    }

    /* List expression */
    if (n->as.list.count == 0u) {
        sh_set_err(p, n->line, n->col);
        return ESHKOL_SH_ERR_SYNTAX;
    }
    int32_t head = n->as.list.first_child;
    if (p->nodes[head].kind != SH_NODE_SYMBOL) {
        sh_set_err(p, p->nodes[head].line, p->nodes[head].col);
        return ESHKOL_SH_ERR_UNSUPPORTED_FORM;
    }
    const char* op_str = p->strpool + p->nodes[head].as.str.off;
    uint32_t op_len = p->nodes[head].as.str.len;
    uint32_t argc = n->as.list.count - 1u;
    int32_t first_arg = p->nodes[head].next_sibling;

    /* Special forms */
    if (sh_streq_lit(op_str, op_len, "if")) {
        if (argc != 2u && argc != 3u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t cond_n = first_arg;
        int32_t then_n = p->nodes[cond_n].next_sibling;
        int32_t else_n = (argc == 3u) ? p->nodes[then_n].next_sibling : -1;

        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cond_n, 0);
        if (st != ESHKOL_SH_OK) return st;
        int32_t jfalse_pc = sh_eskb_emit(c, SH_OP_JUMP_IF_FALSE, 0);
        if (jfalse_pc < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        c->stack_depth--;

        uint32_t depth_before = c->stack_depth;
        st = sh_eskb_compile_expr(c, then_n, is_tail);
        if (st != ESHKOL_SH_OK) return st;
        int32_t jend_pc = sh_eskb_emit(c, SH_OP_JUMP, 0);
        if (jend_pc < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;

        int32_t else_target = (int32_t)(c->code_len - c->cur_func_start);
        c->code[c->cur_func_start + (uint32_t)jfalse_pc].operand = else_target;

        c->stack_depth = depth_before;
        if (else_n >= 0) {
            st = sh_eskb_compile_expr(c, else_n, is_tail);
            if (st != ESHKOL_SH_OK) return st;
        } else {
            if (sh_eskb_emit(c, SH_OP_FALSE, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            c->stack_depth++;
        }
        int32_t end_target = (int32_t)(c->code_len - c->cur_func_start);
        c->code[c->cur_func_start + (uint32_t)jend_pc].operand = end_target;
        return ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "begin")) {
        return sh_eskb_compile_seq(c, first_arg, argc, is_tail);
    }

    if (sh_streq_lit(op_str, op_len, "let")) {
        return sh_eskb_compile_let(c, node_idx, 0, is_tail);
    }

    if (sh_streq_lit(op_str, op_len, "let*")) {
        return sh_eskb_compile_let(c, node_idx, 1, is_tail);
    }

    if (sh_streq_lit(op_str, op_len, "and")) {
        if (argc == 0u) {
            if (sh_eskb_emit(c, SH_OP_TRUE, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            c->stack_depth++;
            return ESHKOL_SH_OK;
        }
        int32_t jumps[32];
        uint32_t njumps = 0u;
        int32_t cur = first_arg;
        for (uint32_t i = 0u; i < argc; i++) {
            int last = (i + 1u == argc);
            eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, last ? is_tail : 0);
            if (st != ESHKOL_SH_OK) return st;
            if (!last) {
                if (njumps >= 32u) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                if (sh_eskb_emit(c, SH_OP_DUP, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                int32_t jf = sh_eskb_emit(c, SH_OP_JUMP_IF_FALSE, 0);
                if (jf < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                jumps[njumps++] = jf;
                if (sh_eskb_emit(c, SH_OP_POP, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                c->stack_depth--;
            }
            cur = p->nodes[cur].next_sibling;
        }
        int32_t end_pc = (int32_t)(c->code_len - c->cur_func_start);
        for (uint32_t i = 0u; i < njumps; i++) {
            c->code[c->cur_func_start + (uint32_t)jumps[i]].operand = end_pc;
        }
        return ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "or")) {
        if (argc == 0u) {
            if (sh_eskb_emit(c, SH_OP_FALSE, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            c->stack_depth++;
            return ESHKOL_SH_OK;
        }
        int32_t jumps[32];
        uint32_t njumps = 0u;
        int32_t cur = first_arg;
        for (uint32_t i = 0u; i < argc; i++) {
            int last = (i + 1u == argc);
            eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, last ? is_tail : 0);
            if (st != ESHKOL_SH_OK) return st;
            if (!last) {
                if (njumps >= 32u) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                if (sh_eskb_emit(c, SH_OP_DUP, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                int32_t jf = sh_eskb_emit(c, SH_OP_JUMP_IF_FALSE, 0);
                int32_t jdone = sh_eskb_emit(c, SH_OP_JUMP, 0);
                if (jf < 0 || jdone < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                jumps[njumps++] = jdone;
                c->code[c->cur_func_start + (uint32_t)jf].operand =
                    (int32_t)(c->code_len - c->cur_func_start);
                if (sh_eskb_emit(c, SH_OP_POP, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                c->stack_depth--;
            }
            cur = p->nodes[cur].next_sibling;
        }
        int32_t end_pc = (int32_t)(c->code_len - c->cur_func_start);
        for (uint32_t i = 0u; i < njumps; i++) {
            c->code[c->cur_func_start + (uint32_t)jumps[i]].operand = end_pc;
        }
        return ESHKOL_SH_OK;
    }

    /* Capability syscall form: (os-syscall num cap arg0 ...) */
    if (sh_streq_lit(op_str, op_len, "os-syscall")) {
        if (argc < 1u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        uint32_t sysno = 0u;
        int32_t vm_fid = 0;
        uint32_t expected_args = 0u;
        if (!sh_lookup_syscall(p, first_arg, &sysno, &vm_fid, &expected_args)) {
            sh_set_err(p, p->nodes[first_arg].line, p->nodes[first_arg].col);
            return ESHKOL_SH_ERR_INVALID_SYSCALL;
        }
        (void)sysno;
        if (argc - 1u != expected_args) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t cur = p->nodes[first_arg].next_sibling;
        for (uint32_t i = 0u; i < expected_args; i++) {
            eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, 0);
            if (st != ESHKOL_SH_OK) return st;
            cur = p->nodes[cur].next_sibling;
        }
        if (sh_eskb_emit(c, SH_OP_NATIVE_CALL, vm_fid) < 0) {
            return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        c->stack_depth = c->stack_depth - expected_args + 1u;
        return ESHKOL_SH_OK;
    }

    /* Arithmetic primitives */
    if (sh_streq_lit(op_str, op_len, "+") || sh_streq_lit(op_str, op_len, "*")) {
        int is_add = (op_str[0] == '+');
        if (argc == 0u) {
            int32_t ci = sh_eskb_add_const_int(c, is_add ? 0 : 1, 1);
            if (ci < 0 || sh_eskb_emit(c, SH_OP_CONST, ci) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            c->stack_depth++;
            return ESHKOL_SH_OK;
        }
        int32_t cur = first_arg;
        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, 0);
        if (st != ESHKOL_SH_OK) return st;
        for (uint32_t i = 1u; i < argc; i++) {
            cur = p->nodes[cur].next_sibling;
            st = sh_eskb_compile_expr(c, cur, 0);
            if (st != ESHKOL_SH_OK) return st;
            if (sh_eskb_emit(c, is_add ? SH_OP_ADD : SH_OP_MUL, 0) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            c->stack_depth--;
        }
        return ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "-") || sh_streq_lit(op_str, op_len, "/")) {
        int is_sub = (op_str[0] == '-');
        if (argc == 0u || (!is_sub && argc < 2u)) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t cur = first_arg;
        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, 0);
        if (st != ESHKOL_SH_OK) return st;
        if (is_sub && argc == 1u) {
            if (sh_eskb_emit(c, SH_OP_NEG, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            return ESHKOL_SH_OK;
        }
        for (uint32_t i = 1u; i < argc; i++) {
            cur = p->nodes[cur].next_sibling;
            st = sh_eskb_compile_expr(c, cur, 0);
            if (st != ESHKOL_SH_OK) return st;
            if (sh_eskb_emit(c, is_sub ? SH_OP_SUB : SH_OP_DIV, 0) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            c->stack_depth--;
        }
        return ESHKOL_SH_OK;
    }

    /* Binary primitives: modulo/remainder, comparisons, cons */
    uint8_t bin_op = 0u;
    if (sh_streq_lit(op_str, op_len, "modulo") || sh_streq_lit(op_str, op_len, "remainder")) bin_op = SH_OP_MOD;
    else if (sh_streq_lit(op_str, op_len, "="))    bin_op = SH_OP_EQ;
    else if (sh_streq_lit(op_str, op_len, "<"))    bin_op = SH_OP_LT;
    else if (sh_streq_lit(op_str, op_len, "<="))   bin_op = SH_OP_LE;
    else if (sh_streq_lit(op_str, op_len, ">"))    bin_op = SH_OP_GT;
    else if (sh_streq_lit(op_str, op_len, ">="))   bin_op = SH_OP_GE;
    else if (sh_streq_lit(op_str, op_len, "cons")) bin_op = SH_OP_CONS;
    if (bin_op != 0u) {
        if (argc != 2u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t a1 = first_arg;
        int32_t a2 = p->nodes[a1].next_sibling;
        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, a1, 0);
        if (st != ESHKOL_SH_OK) return st;
        st = sh_eskb_compile_expr(c, a2, 0);
        if (st != ESHKOL_SH_OK) return st;
        if (sh_eskb_emit(c, bin_op, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        c->stack_depth--;
        return ESHKOL_SH_OK;
    }

    /* Unary primitives */
    uint8_t un_op = 0u;
    if (sh_streq_lit(op_str, op_len, "not"))        un_op = SH_OP_NOT;
    else if (sh_streq_lit(op_str, op_len, "abs"))   un_op = SH_OP_ABS;
    else if (sh_streq_lit(op_str, op_len, "car"))   un_op = SH_OP_CAR;
    else if (sh_streq_lit(op_str, op_len, "cdr"))   un_op = SH_OP_CDR;
    else if (sh_streq_lit(op_str, op_len, "null?")) un_op = SH_OP_NULL_P;
    else if (sh_streq_lit(op_str, op_len, "pair?")) un_op = SH_OP_PAIR_P;
    if (un_op != 0u) {
        if (argc != 1u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, first_arg, 0);
        if (st != ESHKOL_SH_OK) return st;
        if (sh_eskb_emit(c, un_op, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        return ESHKOL_SH_OK;
    }

    /* User-defined function call */
    int fi = sh_find_func(p, op_str, op_len);
    if (fi < 0) {
        sh_set_err(p, p->nodes[head].line, p->nodes[head].col);
        return ESHKOL_SH_ERR_UNBOUND_SYMBOL;
    }
    if (argc != p->funcs[fi].n_params) {
        sh_set_err(p, n->line, n->col);
        return ESHKOL_SH_ERR_ARITY;
    }

    /* Direct self-tail-call lowering: zero-allocation loop via SET_LOCAL + OS_OP_LOOP */
    if (is_tail && (uint32_t)fi == c->cur_func_idx && !p->funcs[fi].is_main) {
        uint32_t n_params = p->funcs[fi].n_params;
        uint32_t extra_slots = c->stack_depth > n_params ? (c->stack_depth - n_params) : 0u;
        int32_t cur = first_arg;
        for (uint32_t i = 0u; i < n_params; i++) {
            eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, 0);
            if (st != ESHKOL_SH_OK) return st;
            cur = p->nodes[cur].next_sibling;
        }
        for (int32_t i = (int32_t)n_params - 1; i >= 0; i--) {
            if (sh_eskb_emit(c, SH_OP_SET_LOCAL, i) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            c->stack_depth--;
        }
        for (uint32_t i = 0u; i < extra_slots; i++) {
            if (sh_eskb_emit(c, SH_OP_POP, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        if (sh_eskb_emit(c, SH_OP_LOOP, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        /* Keep stack_depth accounting consistent for sibling branches */
        c->stack_depth++;
        return ESHKOL_SH_OK;
    }

    int32_t fci = sh_eskb_func_const(c, (uint32_t)fi);
    if (fci < 0 || sh_eskb_emit(c, SH_OP_CLOSURE, fci) < 0) {
        return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
    }
    c->stack_depth++;

    int32_t cur = first_arg;
    for (uint32_t i = 0u; i < argc; i++) {
        eshkol_os_self_host_status_t st = sh_eskb_compile_expr(c, cur, 0);
        if (st != ESHKOL_SH_OK) return st;
        cur = p->nodes[cur].next_sibling;
    }
    if (sh_eskb_emit(c, is_tail ? SH_OP_TAIL_CALL : SH_OP_CALL, (int32_t)argc) < 0) {
        return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
    }
    c->stack_depth -= argc;
    return ESHKOL_SH_OK;
}

/* ── ESKB v2 Serialization Helpers ── */

typedef struct sh_byte_buf {
    uint8_t* data;
    size_t cap;
    size_t len;
} sh_byte_buf_t;

static int sh_buf_u8(sh_byte_buf_t* b, uint8_t v) {
    if (b->len >= b->cap) return -1;
    b->data[b->len++] = v;
    return 0;
}

static int sh_buf_bytes(sh_byte_buf_t* b, const void* src, size_t n) {
    if (n > b->cap - b->len) return -1;
    sh_copy(b->data + b->len, src, n);
    b->len += n;
    return 0;
}

static int sh_buf_u64_le(sh_byte_buf_t* b, uint64_t v) {
    if (8u > b->cap - b->len) return -1;
    sh_put_u64_le(b->data + b->len, v);
    b->len += 8u;
    return 0;
}

static int sh_buf_leb(sh_byte_buf_t* b, uint64_t val) {
    do {
        uint8_t byte = (uint8_t)(val & 0x7Fu);
        val >>= 7;
        if (val != 0u) byte |= 0x80u;
        if (sh_buf_u8(b, byte) < 0) return -1;
    } while (val != 0u);
    return 0;
}

static uint64_t sh_zigzag_i32(int32_t v) {
    uint64_t u = (uint64_t)(int64_t)v;
    return (u << 1) ^ (uint64_t)(v < 0 ? ~0ULL : 0ULL);
}

/* ── Public API: Compile Eshkol Source to ESKB v2 ── */

eshkol_os_self_host_status_t eshkol_os_compile_to_eskb(
    const char* source,
    size_t source_len,
    void* workspace,
    size_t workspace_bytes,
    uint8_t* out_buf,
    size_t out_cap,
    eshkol_os_compile_result_t* out_result) {
    eshkol_os_compile_result_t res;
    sh_zero(&res, sizeof(res));

    if (!source || source_len == 0u || !workspace ||
        workspace_bytes < ESHKOL_OS_SH_MIN_WORKSPACE_BYTES ||
        !out_buf || out_cap < 32u) {
        res.status = ESHKOL_SH_ERR_INVALID_ARG;
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return res.status;
    }

    sh_prog_t prog;
    eshkol_os_self_host_status_t st = sh_parse_program(
        &prog, source, source_len, workspace, workspace_bytes);
    if (st != ESHKOL_SH_OK) {
        res.status = st;
        res.error_line = prog.err_line;
        res.error_col = prog.err_col;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return st;
    }

    sh_eskb_ctx_t ctx;
    sh_zero(&ctx, sizeof(ctx));
    ctx.prog = &prog;
    ctx.consts = (sh_const_entry_t*)sh_arena_alloc(
        &prog.arena, SH_MAX_CONSTS * sizeof(sh_const_entry_t));
    ctx.code = (sh_eskb_instr_t*)sh_arena_alloc(
        &prog.arena, SH_MAX_ESKB_INSTRS * sizeof(sh_eskb_instr_t));
    uint8_t* const_sec_mem = (uint8_t*)sh_arena_alloc(&prog.arena, SH_MAX_CONST_SEC_BYTES);
    uint8_t* code_sec_mem = (uint8_t*)sh_arena_alloc(&prog.arena, SH_MAX_CODE_SEC_BYTES);
    if (!ctx.consts || !ctx.code || !const_sec_mem || !code_sec_mem) {
        res.status = ESHKOL_SH_ERR_WORKSPACE_OOM;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return res.status;
    }

    /* Compile all functions (index 0 = main, plus any lifted named-let helpers) */
    for (uint32_t fi = 0u; fi < prog.n_funcs; fi++) {
        ctx.cur_func_idx = fi;
        ctx.cur_func_start = ctx.code_len;
        ctx.n_locals = 0u;
        ctx.stack_depth = 0u;
        ctx.max_locals = 0u;

        sh_func_def_t* f = &prog.funcs[fi];
        for (uint32_t pi = 0u; pi < f->n_params; pi++) {
            ctx.locals[ctx.n_locals].name_off = f->param_off[pi];
            ctx.locals[ctx.n_locals].name_len = f->param_len[pi];
            ctx.locals[ctx.n_locals].slot = pi;
            ctx.n_locals++;
        }
        ctx.stack_depth = f->n_params;
        ctx.max_locals = f->n_params;

        if (fi == 0u && prog.n_top_exprs > 0u) {
            for (uint32_t ti = 0u; ti < prog.n_top_exprs; ti++) {
                int last = (ti + 1u == prog.n_top_exprs);
                st = sh_eskb_compile_expr(&ctx, prog.top_exprs[ti], last ? 1 : 0);
                if (st != ESHKOL_SH_OK) break;
                if (!last) {
                    if (sh_eskb_emit(&ctx, SH_OP_POP, 0) < 0) {
                        st = ESHKOL_SH_ERR_LIMIT_EXCEEDED;
                        break;
                    }
                    ctx.stack_depth--;
                }
            }
        } else if (f->body_count > 0u) {
            st = sh_eskb_compile_seq(&ctx, f->body_first, f->body_count, 1);
        } else {
            int32_t zci = sh_eskb_add_const_int(&ctx, 0, 1);
            if (zci < 0 || sh_eskb_emit(&ctx, SH_OP_CONST, zci) < 0) {
                st = ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
        }

        if (st == ESHKOL_SH_OK) {
            if (sh_eskb_emit(&ctx, SH_OP_RETURN, 0) < 0) {
                st = ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
        }
        if (st != ESHKOL_SH_OK) {
            res.status = st;
            res.error_line = prog.err_line;
            res.error_col = prog.err_col;
            res.workspace_peak_bytes = prog.arena.peak;
            sh_arena_rewind(&prog.arena);
            if (out_result) sh_copy(out_result, &res, sizeof(res));
            return st;
        }

        f->code_offset = ctx.cur_func_start;
        f->code_len = ctx.code_len - ctx.cur_func_start;
        f->n_locals = ctx.max_locals;
    }

    /* Patch function closure constants with final global code offsets + arity */
    for (uint32_t fi = 0u; fi < prog.n_funcs; fi++) {
        int32_t ci = prog.funcs[fi].closure_const_idx;
        if (ci >= 0) {
            int64_t encoded = (int64_t)(uint64_t)prog.funcs[fi].code_offset |
                              (((int64_t)prog.funcs[fi].n_params) << 32) |
                              (1LL << 40);
            ctx.consts[ci].as.iv = encoded;
        }
    }

    /* Serialize CONST section */
    sh_byte_buf_t const_sec;
    const_sec.data = const_sec_mem;
    const_sec.cap = SH_MAX_CONST_SEC_BYTES;
    const_sec.len = 0u;
    if (sh_buf_leb(&const_sec, ctx.n_consts) < 0) {
        st = ESHKOL_SH_ERR_WORKSPACE_OOM;
    }
    for (uint32_t i = 0u; st == ESHKOL_SH_OK && i < ctx.n_consts; i++) {
        const sh_const_entry_t* ce = &ctx.consts[i];
        if (sh_buf_u8(&const_sec, ce->type) < 0) { st = ESHKOL_SH_ERR_WORKSPACE_OOM; break; }
        if (ce->type == SH_ESKB_CONST_INT64) {
            if (sh_buf_u64_le(&const_sec, (uint64_t)ce->as.iv) < 0) st = ESHKOL_SH_ERR_WORKSPACE_OOM;
        } else if (ce->type == SH_ESKB_CONST_F64) {
            if (sh_buf_u64_le(&const_sec, ce->as.fbits) < 0) st = ESHKOL_SH_ERR_WORKSPACE_OOM;
        } else if (ce->type == SH_ESKB_CONST_BOOL) {
            if (sh_buf_u8(&const_sec, ce->as.bv) < 0) st = ESHKOL_SH_ERR_WORKSPACE_OOM;
        } else if (ce->type == SH_ESKB_CONST_STRING) {
            if (sh_buf_leb(&const_sec, ce->str_len) < 0 ||
                sh_buf_bytes(&const_sec, prog.strpool + ce->as.str_off, ce->str_len) < 0) {
                st = ESHKOL_SH_ERR_WORKSPACE_OOM;
            }
        }
    }

    /* Serialize CODE section */
    sh_byte_buf_t code_sec;
    code_sec.data = code_sec_mem;
    code_sec.cap = SH_MAX_CODE_SEC_BYTES;
    code_sec.len = 0u;
    if (st == ESHKOL_SH_OK && sh_buf_leb(&code_sec, prog.n_funcs) < 0) {
        st = ESHKOL_SH_ERR_WORKSPACE_OOM;
    }
    for (uint32_t fi = 0u; st == ESHKOL_SH_OK && fi < prog.n_funcs; fi++) {
        const sh_func_def_t* f = &prog.funcs[fi];
        if (sh_buf_leb(&code_sec, f->name_len) < 0 ||
            sh_buf_bytes(&code_sec, prog.strpool + f->name_off, f->name_len) < 0 ||
            sh_buf_u8(&code_sec, (uint8_t)f->n_params) < 0 ||
            sh_buf_leb(&code_sec, f->n_locals) < 0 ||
            sh_buf_leb(&code_sec, 0u) < 0 ||
            sh_buf_leb(&code_sec, f->code_len) < 0) {
            st = ESHKOL_SH_ERR_WORKSPACE_OOM;
            break;
        }
        for (uint32_t ii = 0u; ii < f->code_len; ii++) {
            const sh_eskb_instr_t* ins = &ctx.code[f->code_offset + ii];
            if (sh_buf_u8(&code_sec, ins->op) < 0 ||
                sh_buf_leb(&code_sec, sh_zigzag_i32(ins->operand)) < 0) {
                st = ESHKOL_SH_ERR_WORKSPACE_OOM;
                break;
            }
        }
    }

    if (st != ESHKOL_SH_OK) {
        res.status = st;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return st;
    }

    /* Assemble final ESKB v2 container directly into out_buf */
    sh_byte_buf_t out;
    out.data = out_buf;
    out.cap = out_cap;
    out.len = 0u;
    sh_zero(out_buf, 16u);
    out.len = 16u; /* Reserve 16-byte header */

    if (sh_buf_leb(&out, 2u) < 0 ||
        sh_buf_u8(&out, SH_ESKB_SECTION_CONST) < 0 ||
        sh_buf_leb(&out, const_sec.len) < 0 ||
        sh_buf_u8(&out, SH_ESKB_SECTION_CODE) < 0 ||
        sh_buf_leb(&out, code_sec.len) < 0 ||
        sh_buf_bytes(&out, const_sec.data, const_sec.len) < 0 ||
        sh_buf_bytes(&out, code_sec.data, code_sec.len) < 0) {
        res.status = ESHKOL_SH_ERR_OUTPUT_TOO_SMALL;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return res.status;
    }

    uint32_t crc = sh_crc32(out_buf + 16u, out.len - 16u);
    sh_put_u32_le(out_buf + 0u, SH_ESKB_MAGIC);
    sh_put_u32_le(out_buf + 4u, SH_ESKB_VERSION);
    sh_put_u32_le(out_buf + 8u, 1u);
    sh_put_u32_le(out_buf + 12u, crc);

    res.status = ESHKOL_SH_OK;
    res.crc32 = crc;
    res.bytes_written = out.len;
    res.function_count = prog.n_funcs;
    res.instruction_count = ctx.code_len;
    res.constant_count = ctx.n_consts;
    res.workspace_peak_bytes = prog.arena.peak;

    sh_arena_rewind(&prog.arena);
    if (out_result) sh_copy(out_result, &res, sizeof(res));
    return ESHKOL_SH_OK;
}

/* ── Native Machine Code & Standalone W^X ELF64 Emitter ── */

typedef enum sh_patch_kind {
    SH_PATCH_CALL_FUNC = 1,
    SH_PATCH_JUMP      = 2,
    SH_PATCH_JZ        = 3
} sh_patch_kind_t;

typedef struct sh_native_patch {
    uint32_t kind;
    uint32_t offset;
    uint32_t target;
    uint32_t reserved;
} sh_native_patch_t;

typedef struct sh_native_ctx {
    sh_prog_t* prog;
    eshkol_os_target_isa_t target;
    uint8_t* text;
    uint32_t text_len;
    uint32_t text_cap;
    sh_native_patch_t* patches;
    uint32_t n_patches;
    uint32_t func_offsets[SH_MAX_FUNCS];
    sh_local_var_t locals[SH_MAX_LOCALS];
    uint32_t n_locals;
    uint32_t n_params;
    uint32_t let_depth;
    uint32_t instr_count;
} sh_native_ctx_t;

static int sh_nat_u8(sh_native_ctx_t* nc, uint8_t b) {
    if (nc->text_len >= nc->text_cap) return -1;
    nc->text[nc->text_len++] = b;
    return 0;
}

static int sh_nat_u32(sh_native_ctx_t* nc, uint32_t w) {
    if (nc->text_len + 4u > nc->text_cap) return -1;
    sh_put_u32_le(nc->text + nc->text_len, w);
    nc->text_len += 4u;
    nc->instr_count++;
    return 0;
}

static int sh_nat_u64(sh_native_ctx_t* nc, uint64_t q) {
    if (nc->text_len + 8u > nc->text_cap) return -1;
    sh_put_u64_le(nc->text + nc->text_len, q);
    nc->text_len += 8u;
    return 0;
}

static int sh_nat_bytes(sh_native_ctx_t* nc, const uint8_t* src, uint32_t n) {
    if (nc->text_len + n > nc->text_cap) return -1;
    sh_copy(nc->text + nc->text_len, src, n);
    nc->text_len += n;
    nc->instr_count++;
    return 0;
}

static int sh_nat_add_patch(sh_native_ctx_t* nc, sh_patch_kind_t kind, uint32_t off, uint32_t target) {
    if (nc->n_patches >= SH_MAX_NATIVE_PATCHES) return -1;
    uint32_t idx = nc->n_patches++;
    nc->patches[idx].kind = (uint32_t)kind;
    nc->patches[idx].offset = off;
    nc->patches[idx].target = target;
    nc->patches[idx].reserved = 0u;
    return 0;
}

/* ── RISC-V 64 Instruction Encoders ── */

static uint32_t rv_r(uint32_t f7, uint32_t rs2, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t op) {
    return ((f7 & 0x7Fu) << 25) | ((rs2 & 0x1Fu) << 20) | ((rs1 & 0x1Fu) << 15) |
           ((f3 & 0x7u) << 12) | ((rd & 0x1Fu) << 7) | (op & 0x7Fu);
}

static uint32_t rv_i(int32_t imm, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t op) {
    return (((uint32_t)imm & 0xFFFu) << 20) | ((rs1 & 0x1Fu) << 15) |
           ((f3 & 0x7u) << 12) | ((rd & 0x1Fu) << 7) | (op & 0x7Fu);
}

static uint32_t rv_s(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3, uint32_t op) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 5) & 0x7Fu) << 25) | ((rs2 & 0x1Fu) << 20) | ((rs1 & 0x1Fu) << 15) |
           ((f3 & 0x7u) << 12) | ((u & 0x1Fu) << 7) | (op & 0x7Fu);
}

static uint32_t rv_b(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 12) & 1u) << 31) | (((u >> 5) & 0x3Fu) << 25) |
           ((rs2 & 0x1Fu) << 20) | ((rs1 & 0x1Fu) << 15) | ((f3 & 0x7u) << 12) |
           (((u >> 1) & 0xFu) << 8) | (((u >> 11) & 1u) << 7) | 0x63u;
}

static uint32_t rv_j(int32_t imm, uint32_t rd) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 20) & 1u) << 31) | (((u >> 1) & 0x3FFu) << 21) |
           (((u >> 11) & 1u) << 20) | (((u >> 12) & 0xFFu) << 12) |
           ((rd & 0x1Fu) << 7) | 0x6Fu;
}

/* ── ISA-Unified Macro-Operations (x86_64, AArch64, RISC-V 64) ── */

static int sh_nat_load_imm_reg(sh_native_ctx_t* nc, uint32_t reg, int64_t val) {
    uint64_t u = (uint64_t)val;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        /* movabs r64, imm64 (supports rax=0, rdi=7, rsi=6, rdx=2, rcx=1, r8=8..r10=10) */
        uint8_t rex = (reg >= 8u) ? 0x49u : 0x48u;
        uint8_t op = (uint8_t)(0xB8u + (reg & 7u));
        if (sh_nat_u8(nc, rex) < 0 || sh_nat_u8(nc, op) < 0 || sh_nat_u64(nc, u) < 0) return -1;
        nc->instr_count++;
        return 0;
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        uint32_t w0 = (uint32_t)(u & 0xFFFFu);
        uint32_t w1 = (uint32_t)((u >> 16) & 0xFFFFu);
        uint32_t w2 = (uint32_t)((u >> 32) & 0xFFFFu);
        uint32_t w3 = (uint32_t)((u >> 48) & 0xFFFFu);
        if (sh_nat_u32(nc, 0xD2800000u | (w0 << 5) | (reg & 31u)) < 0) return -1;
        if (w1 != 0u && sh_nat_u32(nc, 0xF2A00000u | (w1 << 5) | (reg & 31u)) < 0) return -1;
        if (w2 != 0u && sh_nat_u32(nc, 0xF2C00000u | (w2 << 5) | (reg & 31u)) < 0) return -1;
        if (w3 != 0u && sh_nat_u32(nc, 0xF2E00000u | (w3 << 5) | (reg & 31u)) < 0) return -1;
        return 0;
    } else {
        /* RISC-V 64: 1 instruction for [-2048, 2047], else top-9 + 5x11-bit shifts */
        if (val >= -2048 && val <= 2047) {
            return sh_nat_u32(nc, rv_i((int32_t)val, 0u, 0u, reg, 0x13u));
        }
        int32_t top9 = (int32_t)(val >> 55);
        if (sh_nat_u32(nc, rv_i(top9, 0u, 0u, reg, 0x13u)) < 0) return -1;
        for (int shift = 44; shift >= 0; shift -= 11) {
            uint32_t chunk = (uint32_t)((u >> (unsigned)shift) & 0x7FFu);
            if (sh_nat_u32(nc, rv_i(11, reg, 1u, reg, 0x13u)) < 0) return -1; /* slli reg, reg, 11 */
            if (chunk != 0u) {
                if (sh_nat_u32(nc, rv_i((int32_t)chunk, reg, 6u, reg, 0x13u)) < 0) return -1; /* ori */
            }
        }
        return 0;
    }
}

static int sh_nat_push_acc(sh_native_ctx_t* nc) {
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        const uint8_t b[1] = {0x50u}; /* push rax */
        return sh_nat_bytes(nc, b, 1u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        return sh_nat_u32(nc, 0xF81F0FE0u); /* str x0, [sp, #-16]! */
    } else {
        if (sh_nat_u32(nc, rv_i(-8, 2u, 0u, 2u, 0x13u)) < 0) return -1; /* addi sp, sp, -8 */
        return sh_nat_u32(nc, rv_s(0, 10u, 2u, 3u, 0x23u));             /* sd a0, 0(sp) */
    }
}

static int sh_nat_pop_reg(sh_native_ctx_t* nc, uint32_t reg) {
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        if (reg >= 8u) {
            uint8_t b[2] = {0x41u, (uint8_t)(0x58u + (reg & 7u))};
            return sh_nat_bytes(nc, b, 2u);
        } else {
            uint8_t b[1] = {(uint8_t)(0x58u + (reg & 7u))};
            return sh_nat_bytes(nc, b, 1u);
        }
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        return sh_nat_u32(nc, 0xF84107E0u | (reg & 31u)); /* ldr xN, [sp], #16 */
    } else {
        if (sh_nat_u32(nc, rv_i(0, 2u, 3u, reg, 0x03u)) < 0) return -1; /* ld reg, 0(sp) */
        return sh_nat_u32(nc, rv_i(8, 2u, 0u, 2u, 0x13u));              /* addi sp, sp, 8 */
    }
}

static int sh_nat_pop_acc(sh_native_ctx_t* nc) {
    if (nc->target == ESHKOL_OS_TARGET_X86_64) return sh_nat_pop_reg(nc, 0u);  /* rax */
    if (nc->target == ESHKOL_OS_TARGET_AARCH64) return sh_nat_pop_reg(nc, 0u); /* x0 */
    return sh_nat_pop_reg(nc, 10u);                                             /* a0 */
}

static int sh_nat_push_imm(sh_native_ctx_t* nc, int64_t val) {
    uint32_t acc = (nc->target == ESHKOL_OS_TARGET_RISCV64) ? 10u : 0u;
    if (sh_nat_load_imm_reg(nc, acc, val) < 0) return -1;
    return sh_nat_push_acc(nc);
}

static int sh_nat_prologue(sh_native_ctx_t* nc) {
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        const uint8_t b[4] = {0x55u, 0x48u, 0x89u, 0xE5u}; /* push rbp; mov rbp, rsp */
        return sh_nat_bytes(nc, b, 4u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        if (sh_nat_u32(nc, 0xA9BF7BFDu) < 0) return -1; /* stp x29, x30, [sp, #-16]! */
        return sh_nat_u32(nc, 0x910003FDu);              /* mov x29, sp */
    } else {
        if (sh_nat_u32(nc, rv_i(-16, 2u, 0u, 2u, 0x13u)) < 0) return -1; /* addi sp, sp, -16 */
        if (sh_nat_u32(nc, rv_s(8, 1u, 2u, 3u, 0x23u)) < 0) return -1;   /* sd ra, 8(sp) */
        if (sh_nat_u32(nc, rv_s(0, 8u, 2u, 3u, 0x23u)) < 0) return -1;   /* sd s0, 0(sp) */
        return sh_nat_u32(nc, rv_i(0, 2u, 0u, 8u, 0x13u));               /* addi s0, sp, 0 */
    }
}

static int sh_nat_epilogue(sh_native_ctx_t* nc) {
    if (sh_nat_pop_acc(nc) < 0) return -1;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        const uint8_t b[5] = {0x48u, 0x89u, 0xECu, 0x5Du, 0xC3u}; /* mov rsp, rbp; pop rbp; ret */
        return sh_nat_bytes(nc, b, 5u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        if (sh_nat_u32(nc, 0x910003BFu) < 0) return -1; /* mov sp, x29 */
        if (sh_nat_u32(nc, 0xA8C17BFDu) < 0) return -1; /* ldp x29, x30, [sp], #16 */
        return sh_nat_u32(nc, 0xD65F03C0u);              /* ret */
    } else {
        if (sh_nat_u32(nc, rv_i(0, 8u, 0u, 2u, 0x13u)) < 0) return -1;   /* addi sp, s0, 0 */
        if (sh_nat_u32(nc, rv_i(0, 2u, 3u, 8u, 0x03u)) < 0) return -1;   /* ld s0, 0(sp) */
        if (sh_nat_u32(nc, rv_i(8, 2u, 3u, 1u, 0x03u)) < 0) return -1;   /* ld ra, 8(sp) */
        if (sh_nat_u32(nc, rv_i(16, 2u, 0u, 2u, 0x13u)) < 0) return -1;  /* addi sp, sp, 16 */
        return sh_nat_u32(nc, rv_i(0, 1u, 0u, 0u, 0x67u));               /* jalr zero, ra, 0 */
    }
}

static int sh_nat_load_fp_offset(sh_native_ctx_t* nc, int32_t disp) {
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        uint8_t b[7];
        b[0] = 0x48u; b[1] = 0x8Bu; b[2] = 0x85u; /* mov rax, [rbp + disp32] */
        sh_put_u32_le(b + 3u, (uint32_t)disp);
        if (sh_nat_bytes(nc, b, 7u) < 0) return -1;
        return sh_nat_push_acc(nc);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        if (sh_nat_load_imm_reg(nc, 9u, (int64_t)disp) < 0) return -1;
        if (sh_nat_u32(nc, 0x8B0903A9u) < 0) return -1; /* add x9, x29, x9 */
        if (sh_nat_u32(nc, 0xF9400120u) < 0) return -1; /* ldr x0, [x9] */
        return sh_nat_push_acc(nc);
    } else {
        if (disp >= -2048 && disp <= 2047) {
            if (sh_nat_u32(nc, rv_i(disp, 8u, 3u, 10u, 0x03u)) < 0) return -1; /* ld a0, disp(s0) */
        } else {
            if (sh_nat_load_imm_reg(nc, 5u, (int64_t)disp) < 0) return -1;
            if (sh_nat_u32(nc, rv_r(0u, 5u, 8u, 0u, 5u, 0x33u)) < 0) return -1; /* add t0, s0, t0 */
            if (sh_nat_u32(nc, rv_i(0, 5u, 3u, 10u, 0x03u)) < 0) return -1;     /* ld a0, 0(t0) */
        }
        return sh_nat_push_acc(nc);
    }
}

static int sh_nat_add_sp(sh_native_ctx_t* nc, uint32_t bytes) {
    if (bytes == 0u) return 0;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        uint8_t b[7];
        b[0] = 0x48u; b[1] = 0x81u; b[2] = 0xC4u; /* add rsp, imm32 */
        sh_put_u32_le(b + 3u, bytes);
        return sh_nat_bytes(nc, b, 7u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        while (bytes > 0u) {
            uint32_t chunk = bytes > 4080u ? 4080u : bytes;
            if (sh_nat_u32(nc, 0x910003FFu | ((chunk & 0xFFFu) << 10)) < 0) return -1;
            bytes -= chunk;
        }
        return 0;
    } else {
        while (bytes > 0u) {
            uint32_t chunk = bytes > 2040u ? 2040u : bytes;
            if (sh_nat_u32(nc, rv_i((int32_t)chunk, 2u, 0u, 2u, 0x13u)) < 0) return -1;
            bytes -= chunk;
        }
        return 0;
    }
}

static int sh_nat_emit_call(sh_native_ctx_t* nc, uint32_t target_fi) {
    uint32_t off = nc->text_len;
    if (sh_nat_add_patch(nc, SH_PATCH_CALL_FUNC, off, target_fi) < 0) return -1;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        const uint8_t b[5] = {0xE8u, 0u, 0u, 0u, 0u}; /* call rel32 */
        return sh_nat_bytes(nc, b, 5u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        return sh_nat_u32(nc, 0x94000000u); /* bl #0 */
    } else {
        return sh_nat_u32(nc, rv_j(0, 1u)); /* jal ra, 0 */
    }
}

static int sh_nat_emit_jz(sh_native_ctx_t* nc, uint32_t* out_patch_idx) {
    if (sh_nat_pop_acc(nc) < 0) return -1;
    if (nc->n_patches >= SH_MAX_NATIVE_PATCHES) return -1;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        const uint8_t tst[3] = {0x48u, 0x85u, 0xC0u}; /* test rax, rax */
        if (sh_nat_bytes(nc, tst, 3u) < 0) return -1;
        *out_patch_idx = nc->n_patches;
        if (sh_nat_add_patch(nc, SH_PATCH_JZ, nc->text_len, 0u) < 0) return -1;
        const uint8_t jz[6] = {0x0Fu, 0x84u, 0u, 0u, 0u, 0u}; /* jz rel32 */
        return sh_nat_bytes(nc, jz, 6u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        *out_patch_idx = nc->n_patches;
        if (sh_nat_add_patch(nc, SH_PATCH_JZ, nc->text_len, 0u) < 0) return -1;
        return sh_nat_u32(nc, 0xB4000000u); /* cbz x0, #0 */
    } else {
        /* bne a0, zero, +8 ; jal zero, target */
        if (sh_nat_u32(nc, rv_b(8, 0u, 10u, 1u)) < 0) return -1;
        *out_patch_idx = nc->n_patches;
        if (sh_nat_add_patch(nc, SH_PATCH_JUMP, nc->text_len, 0u) < 0) return -1;
        return sh_nat_u32(nc, rv_j(0, 0u));
    }
}

static int sh_nat_emit_jmp(sh_native_ctx_t* nc, uint32_t* out_patch_idx) {
    if (nc->n_patches >= SH_MAX_NATIVE_PATCHES) return -1;
    *out_patch_idx = nc->n_patches;
    if (sh_nat_add_patch(nc, SH_PATCH_JUMP, nc->text_len, 0u) < 0) return -1;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        const uint8_t j[5] = {0xE9u, 0u, 0u, 0u, 0u}; /* jmp rel32 */
        return sh_nat_bytes(nc, j, 5u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        return sh_nat_u32(nc, 0x14000000u); /* b #0 */
    } else {
        return sh_nat_u32(nc, rv_j(0, 0u)); /* jal zero, 0 */
    }
}

static int sh_nat_emit_binop(sh_native_ctx_t* nc, uint8_t vm_op) {
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        if (sh_nat_pop_reg(nc, 1u) < 0) return -1; /* pop rcx (rhs) */
        if (sh_nat_pop_reg(nc, 0u) < 0) return -1; /* pop rax (lhs) */
        switch (vm_op) {
        case SH_OP_ADD: { const uint8_t b[3] = {0x48u, 0x01u, 0xC8u}; if (sh_nat_bytes(nc, b, 3u) < 0) return -1; break; }
        case SH_OP_SUB: { const uint8_t b[3] = {0x48u, 0x29u, 0xC8u}; if (sh_nat_bytes(nc, b, 3u) < 0) return -1; break; }
        case SH_OP_MUL: { const uint8_t b[4] = {0x48u, 0x0Fu, 0xAFu, 0xC1u}; if (sh_nat_bytes(nc, b, 4u) < 0) return -1; break; }
        case SH_OP_DIV: { const uint8_t b[5] = {0x48u, 0x99u, 0x48u, 0xF7u, 0xF9u}; if (sh_nat_bytes(nc, b, 5u) < 0) return -1; break; }
        case SH_OP_MOD: { const uint8_t b[8] = {0x48u, 0x99u, 0x48u, 0xF7u, 0xF9u, 0x48u, 0x89u, 0xD0u}; if (sh_nat_bytes(nc, b, 8u) < 0) return -1; break; }
        case SH_OP_EQ:
        case SH_OP_LT:
        case SH_OP_LE:
        case SH_OP_GT:
        case SH_OP_GE: {
            uint8_t setcc = 0x94u;
            if (vm_op == SH_OP_LT) setcc = 0x9Cu;
            else if (vm_op == SH_OP_LE) setcc = 0x9Eu;
            else if (vm_op == SH_OP_GT) setcc = 0x9Fu;
            else if (vm_op == SH_OP_GE) setcc = 0x9Du;
            const uint8_t b[10] = {0x48u, 0x39u, 0xC8u, 0x0Fu, setcc, 0xC0u, 0x48u, 0x0Fu, 0xB6u, 0xC0u};
            if (sh_nat_bytes(nc, b, 10u) < 0) return -1;
            break;
        }
        default: return -1;
        }
        return sh_nat_push_acc(nc);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        if (sh_nat_pop_reg(nc, 1u) < 0) return -1; /* x1 = rhs */
        if (sh_nat_pop_reg(nc, 0u) < 0) return -1; /* x0 = lhs */
        switch (vm_op) {
        case SH_OP_ADD: if (sh_nat_u32(nc, 0x8B010000u) < 0) return -1; break;
        case SH_OP_SUB: if (sh_nat_u32(nc, 0xCB010000u) < 0) return -1; break;
        case SH_OP_MUL: if (sh_nat_u32(nc, 0x9B017C00u) < 0) return -1; break;
        case SH_OP_DIV: if (sh_nat_u32(nc, 0x9AC10C00u) < 0) return -1; break;
        case SH_OP_MOD:
            if (sh_nat_u32(nc, 0x9AC10C02u) < 0) return -1; /* sdiv x2, x0, x1 */
            if (sh_nat_u32(nc, 0x9B018040u) < 0) return -1; /* msub x0, x2, x1, x0 */
            break;
        case SH_OP_EQ:
        case SH_OP_LT:
        case SH_OP_LE:
        case SH_OP_GT:
        case SH_OP_GE: {
            uint32_t csinc = 0x9A9F17E0u;
            if (vm_op == SH_OP_LT) csinc = 0x9A9FA7E0u;
            else if (vm_op == SH_OP_LE) csinc = 0x9A9FC7E0u;
            else if (vm_op == SH_OP_GT) csinc = 0x9A9FD7E0u;
            else if (vm_op == SH_OP_GE) csinc = 0x9A9FB7E0u;
            if (sh_nat_u32(nc, 0xEB01001Fu) < 0) return -1; /* cmp x0, x1 */
            if (sh_nat_u32(nc, csinc) < 0) return -1;       /* cset x0, cond */
            break;
        }
        default: return -1;
        }
        return sh_nat_push_acc(nc);
    } else {
        if (sh_nat_pop_reg(nc, 6u) < 0) return -1;  /* t1 = rhs */
        if (sh_nat_pop_reg(nc, 10u) < 0) return -1; /* a0 = lhs */
        switch (vm_op) {
        case SH_OP_ADD: if (sh_nat_u32(nc, rv_r(0x00u, 6u, 10u, 0u, 10u, 0x33u)) < 0) return -1; break;
        case SH_OP_SUB: if (sh_nat_u32(nc, rv_r(0x20u, 6u, 10u, 0u, 10u, 0x33u)) < 0) return -1; break;
        case SH_OP_MUL: if (sh_nat_u32(nc, rv_r(0x01u, 6u, 10u, 0u, 10u, 0x33u)) < 0) return -1; break;
        case SH_OP_DIV: if (sh_nat_u32(nc, rv_r(0x01u, 6u, 10u, 4u, 10u, 0x33u)) < 0) return -1; break;
        case SH_OP_MOD: if (sh_nat_u32(nc, rv_r(0x01u, 6u, 10u, 6u, 10u, 0x33u)) < 0) return -1; break;
        case SH_OP_EQ:
            if (sh_nat_u32(nc, rv_r(0x20u, 6u, 10u, 0u, 10u, 0x33u)) < 0) return -1;
            if (sh_nat_u32(nc, rv_i(1, 10u, 3u, 10u, 0x13u)) < 0) return -1; /* sltiu a0, a0, 1 */
            break;
        case SH_OP_LT:
            if (sh_nat_u32(nc, rv_r(0x00u, 6u, 10u, 2u, 10u, 0x33u)) < 0) return -1; /* slt a0, a0, t1 */
            break;
        case SH_OP_GT:
            if (sh_nat_u32(nc, rv_r(0x00u, 10u, 6u, 2u, 10u, 0x33u)) < 0) return -1; /* slt a0, t1, a0 */
            break;
        case SH_OP_LE:
            if (sh_nat_u32(nc, rv_r(0x00u, 10u, 6u, 2u, 10u, 0x33u)) < 0) return -1; /* slt a0, t1, a0 */
            if (sh_nat_u32(nc, rv_i(1, 10u, 4u, 10u, 0x13u)) < 0) return -1;         /* xori a0, a0, 1 */
            break;
        case SH_OP_GE:
            if (sh_nat_u32(nc, rv_r(0x00u, 6u, 10u, 2u, 10u, 0x33u)) < 0) return -1; /* slt a0, a0, t1 */
            if (sh_nat_u32(nc, rv_i(1, 10u, 4u, 10u, 0x13u)) < 0) return -1;         /* xori a0, a0, 1 */
            break;
        default: return -1;
        }
        return sh_nat_push_acc(nc);
    }
}

static int sh_nat_emit_unop(sh_native_ctx_t* nc, uint8_t vm_op) {
    if (sh_nat_pop_acc(nc) < 0) return -1;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        if (vm_op == SH_OP_NEG) {
            const uint8_t b[3] = {0x48u, 0xF7u, 0xD8u}; /* neg rax */
            if (sh_nat_bytes(nc, b, 3u) < 0) return -1;
        } else if (vm_op == SH_OP_NOT) {
            const uint8_t b[10] = {0x48u, 0x85u, 0xC0u, 0x0Fu, 0x94u, 0xC0u, 0x48u, 0x0Fu, 0xB6u, 0xC0u};
            if (sh_nat_bytes(nc, b, 10u) < 0) return -1;
        } else if (vm_op == SH_OP_ABS) {
            const uint8_t b[13] = {
                0x48u, 0x89u, 0xC1u,       /* mov rcx, rax */
                0x48u, 0xC1u, 0xF9u, 0x3Fu,/* sar rcx, 63 */
                0x48u, 0x31u, 0xC8u,       /* xor rax, rcx */
                0x48u, 0x29u, 0xC8u        /* sub rax, rcx */
            };
            if (sh_nat_bytes(nc, b, 13u) < 0) return -1;
        } else {
            return -1;
        }
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        if (vm_op == SH_OP_NEG) {
            if (sh_nat_u32(nc, 0xCB0003E0u) < 0) return -1; /* neg x0, x0 */
        } else if (vm_op == SH_OP_NOT) {
            if (sh_nat_u32(nc, 0xF100001Fu) < 0) return -1; /* cmp x0, #0 */
            if (sh_nat_u32(nc, 0x9A9F17E0u) < 0) return -1; /* cset x0, eq */
        } else if (vm_op == SH_OP_ABS) {
            if (sh_nat_u32(nc, 0x937FFC01u) < 0) return -1; /* asr x1, x0, #63 */
            if (sh_nat_u32(nc, 0xCA010000u) < 0) return -1; /* eor x0, x0, x1 */
            if (sh_nat_u32(nc, 0xCB010000u) < 0) return -1; /* sub x0, x0, x1 */
        } else {
            return -1;
        }
    } else {
        if (vm_op == SH_OP_NEG) {
            if (sh_nat_u32(nc, rv_r(0x20u, 10u, 0u, 0u, 10u, 0x33u)) < 0) return -1; /* sub a0, zero, a0 */
        } else if (vm_op == SH_OP_NOT) {
            if (sh_nat_u32(nc, rv_i(1, 10u, 3u, 10u, 0x13u)) < 0) return -1; /* sltiu a0, a0, 1 */
        } else if (vm_op == SH_OP_ABS) {
            if (sh_nat_u32(nc, rv_i(0x43F, 10u, 5u, 6u, 0x13u)) < 0) return -1;      /* srai t1, a0, 63 */
            if (sh_nat_u32(nc, rv_r(0x00u, 6u, 10u, 4u, 10u, 0x33u)) < 0) return -1; /* xor a0, a0, t1 */
            if (sh_nat_u32(nc, rv_r(0x20u, 6u, 10u, 0u, 10u, 0x33u)) < 0) return -1; /* sub a0, a0, t1 */
        } else {
            return -1;
        }
    }
    return sh_nat_push_acc(nc);
}

static int sh_nat_emit_syscall(sh_native_ctx_t* nc, uint32_t sysno, uint32_t argc) {
    /* Pop arguments in reverse order into ISA syscall registers (os_syscall_abi.h) */
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        static const uint32_t x64_regs[6] = {7u, 6u, 2u, 10u, 8u, 9u}; /* rdi, rsi, rdx, r10, r8, r9 */
        for (int32_t i = (int32_t)argc - 1; i >= 0; i--) {
            if (sh_nat_pop_reg(nc, x64_regs[i]) < 0) return -1;
        }
        if (sh_nat_load_imm_reg(nc, 0u, (int64_t)sysno) < 0) return -1; /* rax = sysno */
        const uint8_t sc[2] = {0x0Fu, 0x05u};                           /* syscall */
        if (sh_nat_bytes(nc, sc, 2u) < 0) return -1;
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        for (int32_t i = (int32_t)argc - 1; i >= 0; i--) {
            if (sh_nat_pop_reg(nc, (uint32_t)i) < 0) return -1; /* x0..x5 */
        }
        if (sh_nat_load_imm_reg(nc, 8u, (int64_t)sysno) < 0) return -1; /* x8 = sysno */
        if (sh_nat_u32(nc, 0xD4000001u) < 0) return -1;                 /* svc #0 */
    } else {
        for (int32_t i = (int32_t)argc - 1; i >= 0; i--) {
            if (sh_nat_pop_reg(nc, 10u + (uint32_t)i) < 0) return -1; /* a0..a5 */
        }
        if (sh_nat_load_imm_reg(nc, 17u, (int64_t)sysno) < 0) return -1; /* a7 = sysno */
        if (sh_nat_u32(nc, 0x00000073u) < 0) return -1;                  /* ecall */
    }
    return sh_nat_push_acc(nc);
}

static int sh_nat_emit_start_stub(sh_native_ctx_t* nc) {
    /* _start: call main (func 0), then invoke ESHKOL_SYS_TASK_EXIT(ret) */
    if (sh_nat_emit_call(nc, 0u) < 0) return -1;
    if (nc->target == ESHKOL_OS_TARGET_X86_64) {
        const uint8_t mov_rdi_rax[3] = {0x48u, 0x89u, 0xC7u}; /* mov rdi, rax */
        if (sh_nat_bytes(nc, mov_rdi_rax, 3u) < 0) return -1;
        if (sh_nat_load_imm_reg(nc, 0u, (int64_t)ESHKOL_SYS_TASK_EXIT) < 0) return -1;
        const uint8_t sc_trap[4] = {0x0Fu, 0x05u, 0x0Fu, 0x0Bu}; /* syscall; ud2 */
        return sh_nat_bytes(nc, sc_trap, 4u);
    } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
        /* x0 already holds return value of main */
        if (sh_nat_load_imm_reg(nc, 8u, (int64_t)ESHKOL_SYS_TASK_EXIT) < 0) return -1;
        if (sh_nat_u32(nc, 0xD4000001u) < 0) return -1; /* svc #0 */
        return sh_nat_u32(nc, 0xD4200000u);              /* brk #0 */
    } else {
        /* a0 already holds return value of main */
        if (sh_nat_load_imm_reg(nc, 17u, (int64_t)ESHKOL_SYS_TASK_EXIT) < 0) return -1;
        if (sh_nat_u32(nc, 0x00000073u) < 0) return -1; /* ecall */
        return sh_nat_u32(nc, 0x00100073u);              /* ebreak */
    }
}

static int sh_nat_find_local_disp(const sh_native_ctx_t* nc, const char* name, uint32_t len, int32_t* out_disp) {
    int32_t slot_sz = (nc->target == ESHKOL_OS_TARGET_AARCH64) ? 16 : 8;
    for (int i = (int)nc->n_locals - 1; i >= 0; i--) {
        const char* ln = nc->prog->strpool + nc->locals[i].name_off;
        if (sh_memeq(ln, nc->locals[i].name_len, name, len)) {
            uint32_t slot = nc->locals[i].slot;
            if (slot < nc->n_params) {
                /* Param i sits above saved (FP, RA) pair at +16 + (n_params - 1 - i)*slot_sz */
                *out_disp = 16 + (int32_t)(nc->n_params - 1u - slot) * slot_sz;
            } else {
                /* Let local k (0-based above params) sits below FP at -(k + 1)*slot_sz */
                uint32_t k = slot - nc->n_params;
                *out_disp = -((int32_t)(k + 1u) * slot_sz);
            }
            return 1;
        }
    }
    return 0;
}

static eshkol_os_self_host_status_t sh_nat_compile_expr(sh_native_ctx_t* nc, int32_t node_idx);

static eshkol_os_self_host_status_t sh_nat_compile_seq(sh_native_ctx_t* nc,
                                                       int32_t first,
                                                       uint32_t count) {
    if (count == 0u) {
        return sh_nat_push_imm(nc, 0) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
    }
    int32_t cur = first;
    for (uint32_t i = 0u; i < count; i++) {
        eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, cur);
        if (st != ESHKOL_SH_OK) return st;
        if (i + 1u < count) {
            if (sh_nat_pop_acc(nc) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        cur = nc->prog->nodes[cur].next_sibling;
    }
    return ESHKOL_SH_OK;
}

static eshkol_os_self_host_status_t sh_nat_compile_expr(sh_native_ctx_t* nc, int32_t node_idx) {
    sh_prog_t* p = nc->prog;
    const sh_node_t* n = &p->nodes[node_idx];
    uint32_t slot_sz = (nc->target == ESHKOL_OS_TARGET_AARCH64) ? 16u : 8u;

    switch (n->kind) {
    case SH_NODE_INT:
        return sh_nat_push_imm(nc, n->as.iv) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
    case SH_NODE_BOOL:
        return sh_nat_push_imm(nc, n->as.bv ? 1 : 0) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
    case SH_NODE_NIL:
        return sh_nat_push_imm(nc, 0) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
    case SH_NODE_FLOAT:
        return sh_nat_push_imm(nc, (int64_t)n->as.fv) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
    case SH_NODE_SYMBOL: {
        const char* s = p->strpool + n->as.str.off;
        uint32_t len = n->as.str.len;
        int32_t disp = 0;
        if (sh_nat_find_local_disp(nc, s, len, &disp)) {
            return sh_nat_load_fp_offset(nc, disp) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
        }
        int fi = sh_find_func(p, s, len);
        if (fi >= 0 && p->funcs[fi].is_value_def) {
            if (sh_nat_emit_call(nc, (uint32_t)fi) < 0 || sh_nat_push_acc(nc) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            return ESHKOL_SH_OK;
        }
        sh_set_err(p, n->line, n->col);
        return ESHKOL_SH_ERR_UNBOUND_SYMBOL;
    }
    case SH_NODE_LIST:
        break;
    default:
        sh_set_err(p, n->line, n->col);
        return ESHKOL_SH_ERR_UNSUPPORTED_FORM;
    }

    if (n->as.list.count == 0u) {
        sh_set_err(p, n->line, n->col);
        return ESHKOL_SH_ERR_SYNTAX;
    }
    int32_t head = n->as.list.first_child;
    if (p->nodes[head].kind != SH_NODE_SYMBOL) {
        sh_set_err(p, p->nodes[head].line, p->nodes[head].col);
        return ESHKOL_SH_ERR_UNSUPPORTED_FORM;
    }
    const char* op_str = p->strpool + p->nodes[head].as.str.off;
    uint32_t op_len = p->nodes[head].as.str.len;
    uint32_t argc = n->as.list.count - 1u;
    int32_t first_arg = p->nodes[head].next_sibling;

    if (sh_streq_lit(op_str, op_len, "if")) {
        if (argc != 2u && argc != 3u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t cond_n = first_arg;
        int32_t then_n = p->nodes[cond_n].next_sibling;
        int32_t else_n = (argc == 3u) ? p->nodes[then_n].next_sibling : -1;

        eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, cond_n);
        if (st != ESHKOL_SH_OK) return st;
        uint32_t jz_patch = 0u;
        if (sh_nat_emit_jz(nc, &jz_patch) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;

        st = sh_nat_compile_expr(nc, then_n);
        if (st != ESHKOL_SH_OK) return st;
        uint32_t jmp_patch = 0u;
        if (sh_nat_emit_jmp(nc, &jmp_patch) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;

        nc->patches[jz_patch].target = nc->text_len;
        if (else_n >= 0) {
            st = sh_nat_compile_expr(nc, else_n);
            if (st != ESHKOL_SH_OK) return st;
        } else {
            if (sh_nat_push_imm(nc, 0) < 0) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
        nc->patches[jmp_patch].target = nc->text_len;
        return ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "begin")) {
        return sh_nat_compile_seq(nc, first_arg, argc);
    }

    if (sh_streq_lit(op_str, op_len, "let") || sh_streq_lit(op_str, op_len, "let*")) {
        int is_star = (op_len == 4u);
        if (argc < 2u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_SYNTAX;
        }
        int32_t second = first_arg;
        /* Named let */
        if (!is_star && p->nodes[second].kind == SH_NODE_SYMBOL) {
            if (argc < 3u) {
                sh_set_err(p, n->line, n->col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            int32_t binds = p->nodes[second].next_sibling;
            if (p->nodes[binds].kind != SH_NODE_LIST) {
                sh_set_err(p, p->nodes[binds].line, p->nodes[binds].col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            uint32_t n_binds = p->nodes[binds].as.list.count;
            if (n_binds > SH_MAX_PARAMS || p->n_funcs >= SH_MAX_FUNCS) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            uint32_t fi = p->n_funcs++;
            sh_zero(&p->funcs[fi], sizeof(sh_func_def_t));
            p->funcs[fi].name_off = p->nodes[second].as.str.off;
            p->funcs[fi].name_len = p->nodes[second].as.str.len;
            p->funcs[fi].n_params = n_binds;
            p->funcs[fi].body_first = p->nodes[binds].next_sibling;
            p->funcs[fi].body_count = argc - 2u;

            int32_t bnode = p->nodes[binds].as.list.first_child;
            for (uint32_t i = 0u; i < n_binds; i++) {
                if (p->nodes[bnode].kind != SH_NODE_LIST || p->nodes[bnode].as.list.count != 2u) {
                    sh_set_err(p, p->nodes[bnode].line, p->nodes[bnode].col);
                    return ESHKOL_SH_ERR_SYNTAX;
                }
                int32_t bname = p->nodes[bnode].as.list.first_child;
                int32_t bval = p->nodes[bname].next_sibling;
                if (p->nodes[bname].kind != SH_NODE_SYMBOL) {
                    sh_set_err(p, p->nodes[bname].line, p->nodes[bname].col);
                    return ESHKOL_SH_ERR_SYNTAX;
                }
                p->funcs[fi].param_off[i] = p->nodes[bname].as.str.off;
                p->funcs[fi].param_len[i] = p->nodes[bname].as.str.len;
                eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, bval);
                if (st != ESHKOL_SH_OK) return st;
                bnode = p->nodes[bnode].next_sibling;
            }
            if (sh_nat_emit_call(nc, fi) < 0 ||
                sh_nat_add_sp(nc, n_binds * slot_sz) < 0 ||
                sh_nat_push_acc(nc) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
            return ESHKOL_SH_OK;
        }

        int32_t binds = second;
        if (p->nodes[binds].kind != SH_NODE_LIST) {
            sh_set_err(p, p->nodes[binds].line, p->nodes[binds].col);
            return ESHKOL_SH_ERR_SYNTAX;
        }
        uint32_t n_binds = p->nodes[binds].as.list.count;
        if (nc->n_locals + n_binds > SH_MAX_LOCALS) return ESHKOL_SH_ERR_LIMIT_EXCEEDED;

        uint32_t saved_locals = nc->n_locals;
        uint32_t saved_let_depth = nc->let_depth;
        uint32_t pending_off[SH_MAX_LOCALS];
        uint32_t pending_len[SH_MAX_LOCALS];

        int32_t bnode = p->nodes[binds].as.list.first_child;
        for (uint32_t i = 0u; i < n_binds; i++) {
            if (p->nodes[bnode].kind != SH_NODE_LIST || p->nodes[bnode].as.list.count != 2u) {
                sh_set_err(p, p->nodes[bnode].line, p->nodes[bnode].col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            int32_t bname = p->nodes[bnode].as.list.first_child;
            int32_t bval = p->nodes[bname].next_sibling;
            if (p->nodes[bname].kind != SH_NODE_SYMBOL) {
                sh_set_err(p, p->nodes[bname].line, p->nodes[bname].col);
                return ESHKOL_SH_ERR_SYNTAX;
            }
            eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, bval);
            if (st != ESHKOL_SH_OK) return st;
            if (is_star) {
                nc->locals[nc->n_locals].name_off = p->nodes[bname].as.str.off;
                nc->locals[nc->n_locals].name_len = p->nodes[bname].as.str.len;
                nc->locals[nc->n_locals].slot = nc->n_params + nc->let_depth;
                nc->n_locals++;
                nc->let_depth++;
            } else {
                pending_off[i] = p->nodes[bname].as.str.off;
                pending_len[i] = p->nodes[bname].as.str.len;
            }
            bnode = p->nodes[bnode].next_sibling;
        }
        if (!is_star) {
            for (uint32_t i = 0u; i < n_binds; i++) {
                nc->locals[nc->n_locals].name_off = pending_off[i];
                nc->locals[nc->n_locals].name_len = pending_len[i];
                nc->locals[nc->n_locals].slot = nc->n_params + nc->let_depth;
                nc->n_locals++;
                nc->let_depth++;
            }
        }

        eshkol_os_self_host_status_t st = sh_nat_compile_seq(
            nc, p->nodes[binds].next_sibling, argc - 1u);
        nc->n_locals = saved_locals;
        nc->let_depth = saved_let_depth;
        if (st != ESHKOL_SH_OK) return st;

        if (n_binds > 0u) {
            if (sh_nat_pop_acc(nc) < 0 ||
                sh_nat_add_sp(nc, n_binds * slot_sz) < 0 ||
                sh_nat_push_acc(nc) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
        }
        return ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "os-syscall")) {
        if (argc < 1u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        uint32_t sysno = 0u;
        int32_t vm_fid = 0;
        uint32_t expected_args = 0u;
        if (!sh_lookup_syscall(p, first_arg, &sysno, &vm_fid, &expected_args)) {
            sh_set_err(p, p->nodes[first_arg].line, p->nodes[first_arg].col);
            return ESHKOL_SH_ERR_INVALID_SYSCALL;
        }
        (void)vm_fid;
        if (argc - 1u != expected_args) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t cur = p->nodes[first_arg].next_sibling;
        for (uint32_t i = 0u; i < expected_args; i++) {
            eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, cur);
            if (st != ESHKOL_SH_OK) return st;
            cur = p->nodes[cur].next_sibling;
        }
        return sh_nat_emit_syscall(nc, sysno, expected_args) < 0
                   ? ESHKOL_SH_ERR_LIMIT_EXCEEDED
                   : ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "+") || sh_streq_lit(op_str, op_len, "*")) {
        int is_add = (op_str[0] == '+');
        if (argc == 0u) {
            return sh_nat_push_imm(nc, is_add ? 0 : 1) < 0
                       ? ESHKOL_SH_ERR_LIMIT_EXCEEDED
                       : ESHKOL_SH_OK;
        }
        int32_t cur = first_arg;
        eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, cur);
        if (st != ESHKOL_SH_OK) return st;
        for (uint32_t i = 1u; i < argc; i++) {
            cur = p->nodes[cur].next_sibling;
            st = sh_nat_compile_expr(nc, cur);
            if (st != ESHKOL_SH_OK) return st;
            if (sh_nat_emit_binop(nc, is_add ? SH_OP_ADD : SH_OP_MUL) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
        }
        return ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "-") || sh_streq_lit(op_str, op_len, "/")) {
        int is_sub = (op_str[0] == '-');
        if (argc == 0u || (!is_sub && argc < 2u)) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t cur = first_arg;
        eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, cur);
        if (st != ESHKOL_SH_OK) return st;
        if (is_sub && argc == 1u) {
            return sh_nat_emit_unop(nc, SH_OP_NEG) < 0
                       ? ESHKOL_SH_ERR_LIMIT_EXCEEDED
                       : ESHKOL_SH_OK;
        }
        for (uint32_t i = 1u; i < argc; i++) {
            cur = p->nodes[cur].next_sibling;
            st = sh_nat_compile_expr(nc, cur);
            if (st != ESHKOL_SH_OK) return st;
            if (sh_nat_emit_binop(nc, is_sub ? SH_OP_SUB : SH_OP_DIV) < 0) {
                return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
            }
        }
        return ESHKOL_SH_OK;
    }

    uint8_t bin_op = 0u;
    if (sh_streq_lit(op_str, op_len, "modulo") || sh_streq_lit(op_str, op_len, "remainder")) bin_op = SH_OP_MOD;
    else if (sh_streq_lit(op_str, op_len, "="))  bin_op = SH_OP_EQ;
    else if (sh_streq_lit(op_str, op_len, "<"))  bin_op = SH_OP_LT;
    else if (sh_streq_lit(op_str, op_len, "<=")) bin_op = SH_OP_LE;
    else if (sh_streq_lit(op_str, op_len, ">"))  bin_op = SH_OP_GT;
    else if (sh_streq_lit(op_str, op_len, ">=")) bin_op = SH_OP_GE;
    if (bin_op != 0u) {
        if (argc != 2u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        int32_t a1 = first_arg;
        int32_t a2 = p->nodes[a1].next_sibling;
        eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, a1);
        if (st != ESHKOL_SH_OK) return st;
        st = sh_nat_compile_expr(nc, a2);
        if (st != ESHKOL_SH_OK) return st;
        return sh_nat_emit_binop(nc, bin_op) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
    }

    if (sh_streq_lit(op_str, op_len, "not") || sh_streq_lit(op_str, op_len, "abs")) {
        if (argc != 1u) {
            sh_set_err(p, n->line, n->col);
            return ESHKOL_SH_ERR_ARITY;
        }
        eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, first_arg);
        if (st != ESHKOL_SH_OK) return st;
        uint8_t uop = (op_str[0] == 'n') ? SH_OP_NOT : SH_OP_ABS;
        return sh_nat_emit_unop(nc, uop) < 0 ? ESHKOL_SH_ERR_LIMIT_EXCEEDED : ESHKOL_SH_OK;
    }

    int fi = sh_find_func(p, op_str, op_len);
    if (fi < 0) {
        sh_set_err(p, p->nodes[head].line, p->nodes[head].col);
        return ESHKOL_SH_ERR_UNBOUND_SYMBOL;
    }
    if (argc != p->funcs[fi].n_params) {
        sh_set_err(p, n->line, n->col);
        return ESHKOL_SH_ERR_ARITY;
    }
    int32_t cur = first_arg;
    for (uint32_t i = 0u; i < argc; i++) {
        eshkol_os_self_host_status_t st = sh_nat_compile_expr(nc, cur);
        if (st != ESHKOL_SH_OK) return st;
        cur = p->nodes[cur].next_sibling;
    }
    if (sh_nat_emit_call(nc, (uint32_t)fi) < 0 ||
        sh_nat_add_sp(nc, argc * slot_sz) < 0 ||
        sh_nat_push_acc(nc) < 0) {
        return ESHKOL_SH_ERR_LIMIT_EXCEEDED;
    }
    return ESHKOL_SH_OK;
}

static int sh_nat_apply_patches(sh_native_ctx_t* nc) {
    for (uint32_t i = 0u; i < nc->n_patches; i++) {
        const sh_native_patch_t* pt = &nc->patches[i];
        uint32_t target_off = (pt->kind == SH_PATCH_CALL_FUNC)
                                  ? nc->func_offsets[pt->target]
                                  : pt->target;
        if (nc->target == ESHKOL_OS_TARGET_X86_64) {
            uint32_t next_ip = (pt->kind == SH_PATCH_JZ) ? (pt->offset + 6u) : (pt->offset + 5u);
            int32_t rel = (int32_t)target_off - (int32_t)next_ip;
            uint32_t imm_off = (pt->kind == SH_PATCH_JZ) ? (pt->offset + 2u) : (pt->offset + 1u);
            sh_put_u32_le(nc->text + imm_off, (uint32_t)rel);
        } else if (nc->target == ESHKOL_OS_TARGET_AARCH64) {
            int32_t rel_bytes = (int32_t)target_off - (int32_t)pt->offset;
            int32_t rel_words = rel_bytes / 4;
            uint32_t ins = 0u;
            if (pt->kind == SH_PATCH_CALL_FUNC) {
                ins = 0x94000000u | ((uint32_t)rel_words & 0x03FFFFFFu); /* bl */
            } else if (pt->kind == SH_PATCH_JUMP) {
                ins = 0x14000000u | ((uint32_t)rel_words & 0x03FFFFFFu); /* b */
            } else {
                ins = 0xB4000000u | (((uint32_t)rel_words & 0x7FFFFu) << 5); /* cbz x0 */
            }
            sh_put_u32_le(nc->text + pt->offset, ins);
        } else {
            int32_t rel_bytes = (int32_t)target_off - (int32_t)pt->offset;
            uint32_t rd = (pt->kind == SH_PATCH_CALL_FUNC) ? 1u : 0u;
            sh_put_u32_le(nc->text + pt->offset, rv_j(rel_bytes, rd));
        }
    }
    return 0;
}

static void sh_write_elf64_phdr(uint8_t* dst,
                                uint32_t p_type,
                                uint32_t p_flags,
                                uint64_t p_offset,
                                uint64_t p_vaddr,
                                uint64_t p_filesz,
                                uint64_t p_memsz,
                                uint64_t p_align) {
    sh_put_u32_le(dst + 0u, p_type);
    sh_put_u32_le(dst + 4u, p_flags);
    sh_put_u64_le(dst + 8u, p_offset);
    sh_put_u64_le(dst + 16u, p_vaddr);
    sh_put_u64_le(dst + 24u, p_vaddr);
    sh_put_u64_le(dst + 32u, p_filesz);
    sh_put_u64_le(dst + 40u, p_memsz);
    sh_put_u64_le(dst + 48u, p_align);
}

static void sh_write_elf64_shdr(uint8_t* dst,
                                uint32_t sh_name,
                                uint32_t sh_type,
                                uint64_t sh_flags,
                                uint64_t sh_addr,
                                uint64_t sh_offset,
                                uint64_t sh_size,
                                uint64_t sh_addralign) {
    sh_zero(dst, 64u);
    sh_put_u32_le(dst + 0u, sh_name);
    sh_put_u32_le(dst + 4u, sh_type);
    sh_put_u64_le(dst + 8u, sh_flags);
    sh_put_u64_le(dst + 16u, sh_addr);
    sh_put_u64_le(dst + 24u, sh_offset);
    sh_put_u64_le(dst + 32u, sh_size);
    sh_put_u64_le(dst + 48u, sh_addralign);
}

/* ── Public API: Compile Eshkol Source to Standalone Static W^X ELF64 ── */

eshkol_os_self_host_status_t eshkol_os_compile_to_elf64(
    const char* source,
    size_t source_len,
    eshkol_os_target_isa_t target,
    void* workspace,
    size_t workspace_bytes,
    uint8_t* out_buf,
    size_t out_cap,
    eshkol_os_compile_result_t* out_result) {
    eshkol_os_compile_result_t res;
    sh_zero(&res, sizeof(res));

    uint16_t e_machine = 0u;
    if (target == ESHKOL_OS_TARGET_X86_64) e_machine = SH_ELF_EM_X86_64;
    else if (target == ESHKOL_OS_TARGET_AARCH64) e_machine = SH_ELF_EM_AARCH64;
    else if (target == ESHKOL_OS_TARGET_RISCV64) e_machine = SH_ELF_EM_RISCV;
    else {
        res.status = ESHKOL_SH_ERR_INVALID_TARGET;
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return res.status;
    }

    if (!source || source_len == 0u || !workspace ||
        workspace_bytes < ESHKOL_OS_SH_MIN_WORKSPACE_BYTES ||
        !out_buf || out_cap < 8192u) {
        res.status = ESHKOL_SH_ERR_INVALID_ARG;
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return res.status;
    }

    sh_prog_t prog;
    eshkol_os_self_host_status_t st = sh_parse_program(
        &prog, source, source_len, workspace, workspace_bytes);
    if (st != ESHKOL_SH_OK) {
        res.status = st;
        res.error_line = prog.err_line;
        res.error_col = prog.err_col;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return st;
    }

    sh_native_ctx_t nc;
    sh_zero(&nc, sizeof(nc));
    nc.prog = &prog;
    nc.target = target;
    nc.text = (uint8_t*)sh_arena_alloc(&prog.arena, SH_MAX_NATIVE_TEXT_BYTES);
    nc.text_cap = SH_MAX_NATIVE_TEXT_BYTES;
    nc.patches = (sh_native_patch_t*)sh_arena_alloc(
        &prog.arena, SH_MAX_NATIVE_PATCHES * sizeof(sh_native_patch_t));
    if (!nc.text || !nc.patches) {
        res.status = ESHKOL_SH_ERR_WORKSPACE_OOM;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return res.status;
    }

    /* Emit _start entry stub at offset 0 of .text (ESHKOL_OS_SH_ELF_TEXT_VADDR) */
    if (sh_nat_emit_start_stub(&nc) < 0) {
        st = ESHKOL_SH_ERR_LIMIT_EXCEEDED;
    }

    /* Emit every compiled function in deterministic index order */
    for (uint32_t fi = 0u; st == ESHKOL_SH_OK && fi < prog.n_funcs; fi++) {
        /* 4-byte align function entry points */
        while ((nc.text_len & 3u) != 0u) {
            if (sh_nat_u8(&nc, 0x90u) < 0) { st = ESHKOL_SH_ERR_LIMIT_EXCEEDED; break; }
        }
        if (st != ESHKOL_SH_OK) break;

        nc.func_offsets[fi] = nc.text_len;
        nc.n_locals = 0u;
        nc.let_depth = 0u;

        sh_func_def_t* f = &prog.funcs[fi];
        nc.n_params = f->n_params;
        for (uint32_t pi = 0u; pi < f->n_params; pi++) {
            nc.locals[nc.n_locals].name_off = f->param_off[pi];
            nc.locals[nc.n_locals].name_len = f->param_len[pi];
            nc.locals[nc.n_locals].slot = pi;
            nc.n_locals++;
        }

        if (sh_nat_prologue(&nc) < 0) { st = ESHKOL_SH_ERR_LIMIT_EXCEEDED; break; }

        if (fi == 0u && prog.n_top_exprs > 0u) {
            for (uint32_t ti = 0u; ti < prog.n_top_exprs; ti++) {
                st = sh_nat_compile_expr(&nc, prog.top_exprs[ti]);
                if (st != ESHKOL_SH_OK) break;
                if (ti + 1u < prog.n_top_exprs) {
                    if (sh_nat_pop_acc(&nc) < 0) { st = ESHKOL_SH_ERR_LIMIT_EXCEEDED; break; }
                }
            }
        } else if (f->body_count > 0u) {
            st = sh_nat_compile_seq(&nc, f->body_first, f->body_count);
        } else {
            if (sh_nat_push_imm(&nc, 0) < 0) st = ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }

        if (st == ESHKOL_SH_OK && sh_nat_epilogue(&nc) < 0) {
            st = ESHKOL_SH_ERR_LIMIT_EXCEEDED;
        }
    }

    if (st != ESHKOL_SH_OK) {
        res.status = st;
        res.error_line = prog.err_line;
        res.error_col = prog.err_col;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return st;
    }

    (void)sh_nat_apply_patches(&nc);

    /* Layout deterministic W^X ELF64 image:
     *   [0x0000 .. 0x003F] Elf64_Ehdr (64 bytes)
     *   [0x0040 .. 0x00AF] 2 x Elf64_Phdr (112 bytes):
     *                      PT_LOAD #0: .text (PF_R | PF_X) at 0x1000 / vaddr 0x401000
     *                      PT_LOAD #1: .data (PF_R | PF_W) at next 4KiB page
     *   [0x1000 ..       ] .text machine code
     *   [data_off ..     ] .data static metadata (16 bytes)
     *   [shstr_off ..    ] .shstrtab ("\0.text\0.data\0.shstrtab\0")
     *   [shdr_off ..     ] 4 x Elf64_Shdr (256 bytes)
     */
    static const uint8_t data_payload[16] = {
        'E', 'S', 'H', 'K', 'O', 'L', '-', 'O', 'S', '-', 'U', '6', 0u, 0u, 0u, 0u
    };
    static const char shstrtab[] = "\0.text\0.data\0.shstrtab";
    const size_t shstrtab_len = sizeof(shstrtab); /* 23 bytes incl. trailing NUL */

    const size_t text_off = (size_t)ESHKOL_OS_SH_ELF_PAGE_SIZE;
    const size_t text_size = (size_t)nc.text_len;
    const size_t data_off = sh_align_up(text_off + text_size, (size_t)ESHKOL_OS_SH_ELF_PAGE_SIZE);
    const size_t data_size = sizeof(data_payload);
    const size_t shstr_off = sh_align_up(data_off + data_size, 8u);
    const size_t shdr_off = sh_align_up(shstr_off + shstrtab_len, 8u);
    const size_t total_size = shdr_off + 4u * 64u;

    if (total_size > out_cap) {
        res.status = ESHKOL_SH_ERR_OUTPUT_TOO_SMALL;
        res.workspace_peak_bytes = prog.arena.peak;
        sh_arena_rewind(&prog.arena);
        if (out_result) sh_copy(out_result, &res, sizeof(res));
        return res.status;
    }

    const uint64_t text_vaddr = ESHKOL_OS_SH_ELF_TEXT_VADDR;
    const uint64_t data_vaddr = ESHKOL_OS_SH_ELF_BASE_VADDR + (uint64_t)data_off;

    sh_zero(out_buf, total_size);

    /* Elf64_Ehdr */
    out_buf[0] = 0x7Fu;
    out_buf[1] = 'E';
    out_buf[2] = 'L';
    out_buf[3] = 'F';
    out_buf[4] = 2u; /* ELFCLASS64 */
    out_buf[5] = 1u; /* ELFDATA2LSB */
    out_buf[6] = 1u; /* EV_CURRENT */
    out_buf[7] = 0u; /* ELFOSABI_NONE */
    sh_put_u16_le(out_buf + 16u, (uint16_t)SH_ELF_ET_EXEC);
    sh_put_u16_le(out_buf + 18u, e_machine);
    sh_put_u32_le(out_buf + 20u, 1u);               /* e_version */
    sh_put_u64_le(out_buf + 24u, text_vaddr);       /* e_entry */
    sh_put_u64_le(out_buf + 32u, 64ULL);            /* e_phoff */
    sh_put_u64_le(out_buf + 40u, (uint64_t)shdr_off);/* e_shoff */
    sh_put_u32_le(out_buf + 48u, 0u);               /* e_flags */
    sh_put_u16_le(out_buf + 52u, 64u);              /* e_ehsize */
    sh_put_u16_le(out_buf + 54u, 56u);              /* e_phentsize */
    sh_put_u16_le(out_buf + 56u, 2u);               /* e_phnum */
    sh_put_u16_le(out_buf + 58u, 64u);              /* e_shentsize */
    sh_put_u16_le(out_buf + 60u, 4u);               /* e_shnum */
    sh_put_u16_le(out_buf + 62u, 3u);               /* e_shstrndx */

    /* Program Header 0: .text (PT_LOAD, PF_R | PF_X ONLY) */
    sh_write_elf64_phdr(
        out_buf + 64u,
        SH_ELF_PT_LOAD,
        SH_ELF_PF_R | SH_ELF_PF_X,
        (uint64_t)text_off,
        text_vaddr,
        (uint64_t)text_size,
        (uint64_t)text_size,
        ESHKOL_OS_SH_ELF_PAGE_SIZE);

    /* Program Header 1: .data (PT_LOAD, PF_R | PF_W ONLY) */
    sh_write_elf64_phdr(
        out_buf + 120u,
        SH_ELF_PT_LOAD,
        SH_ELF_PF_R | SH_ELF_PF_W,
        (uint64_t)data_off,
        data_vaddr,
        (uint64_t)data_size,
        (uint64_t)data_size,
        ESHKOL_OS_SH_ELF_PAGE_SIZE);

    /* Payload sections */
    sh_copy(out_buf + text_off, nc.text, text_size);
    sh_copy(out_buf + data_off, data_payload, data_size);
    sh_copy(out_buf + shstr_off, shstrtab, shstrtab_len);

    /* Section Header Table */
    sh_write_elf64_shdr(out_buf + shdr_off + 0u * 64u, 0u, SH_ELF_SHT_NULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    sh_write_elf64_shdr(out_buf + shdr_off + 1u * 64u, 1u, SH_ELF_SHT_PROGBITS,
                        SH_ELF_SHF_ALLOC | SH_ELF_SHF_EXECINSTR,
                        text_vaddr, (uint64_t)text_off, (uint64_t)text_size, 16ULL);
    sh_write_elf64_shdr(out_buf + shdr_off + 2u * 64u, 7u, SH_ELF_SHT_PROGBITS,
                        SH_ELF_SHF_WRITE | SH_ELF_SHF_ALLOC,
                        data_vaddr, (uint64_t)data_off, (uint64_t)data_size, 8ULL);
    sh_write_elf64_shdr(out_buf + shdr_off + 3u * 64u, 13u, SH_ELF_SHT_STRTAB,
                        0ULL, 0ULL, (uint64_t)shstr_off, (uint64_t)shstrtab_len, 1ULL);

    res.status = ESHKOL_SH_OK;
    res.crc32 = sh_crc32(out_buf, total_size);
    res.bytes_written = total_size;
    res.function_count = prog.n_funcs;
    res.instruction_count = nc.instr_count;
    res.entry_vaddr = text_vaddr;
    res.text_vaddr = text_vaddr;
    res.text_bytes = (uint64_t)text_size;
    res.data_vaddr = data_vaddr;
    res.data_bytes = (uint64_t)data_size;
    res.workspace_peak_bytes = prog.arena.peak;

    sh_arena_rewind(&prog.arena);
    if (out_result) sh_copy(out_result, &res, sizeof(res));
    return ESHKOL_SH_OK;
}

const char* eshkol_os_self_host_status_name(eshkol_os_self_host_status_t status) {
    switch (status) {
    case ESHKOL_SH_OK:                   return "ESHKOL_SH_OK";
    case ESHKOL_SH_ERR_INVALID_ARG:      return "ESHKOL_SH_ERR_INVALID_ARG";
    case ESHKOL_SH_ERR_SYNTAX:           return "ESHKOL_SH_ERR_SYNTAX";
    case ESHKOL_SH_ERR_DEPTH_EXCEEDED:   return "ESHKOL_SH_ERR_DEPTH_EXCEEDED";
    case ESHKOL_SH_ERR_UNBOUND_SYMBOL:   return "ESHKOL_SH_ERR_UNBOUND_SYMBOL";
    case ESHKOL_SH_ERR_ARITY:            return "ESHKOL_SH_ERR_ARITY";
    case ESHKOL_SH_ERR_UNSUPPORTED_FORM: return "ESHKOL_SH_ERR_UNSUPPORTED_FORM";
    case ESHKOL_SH_ERR_WORKSPACE_OOM:    return "ESHKOL_SH_ERR_WORKSPACE_OOM";
    case ESHKOL_SH_ERR_OUTPUT_TOO_SMALL: return "ESHKOL_SH_ERR_OUTPUT_TOO_SMALL";
    case ESHKOL_SH_ERR_INVALID_TARGET:   return "ESHKOL_SH_ERR_INVALID_TARGET";
    case ESHKOL_SH_ERR_INVALID_SYSCALL:  return "ESHKOL_SH_ERR_INVALID_SYSCALL";
    case ESHKOL_SH_ERR_DUPLICATE_SYMBOL: return "ESHKOL_SH_ERR_DUPLICATE_SYMBOL";
    case ESHKOL_SH_ERR_LIMIT_EXCEEDED:   return "ESHKOL_SH_ERR_LIMIT_EXCEEDED";
    }
    return "ESHKOL_SH_UNKNOWN";
}

const char* eshkol_os_target_isa_name(eshkol_os_target_isa_t target) {
    switch (target) {
    case ESHKOL_OS_TARGET_X86_64:  return "x86_64";
    case ESHKOL_OS_TARGET_AARCH64: return "aarch64";
    case ESHKOL_OS_TARGET_RISCV64: return "riscv64";
    }
    return "unknown";
}
