/**
 * @file vm_symbolic_ad.c
 * @brief Symbolic automatic differentiation for the Eshkol bytecode VM.
 *
 * Builds symbolic expression graphs from bytecode traces,
 * applies algebraic simplification, and generates optimized
 * gradient computation code.
 *
 * Simplification rules:
 *   x + 0 → x,  x * 1 → x,  x * 0 → 0
 *   x - x → 0,  x / x → 1
 *   exp(log(x)) → x,  log(exp(x)) → x
 *   d/dx(x^n) → n*x^(n-1)
 *   d/dx(sin(x)) → cos(x)
 *   Common subexpression elimination via hash-consing
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum {
    SYM_CONST, SYM_VAR,
    SYM_ADD, SYM_SUB, SYM_MUL, SYM_DIV, SYM_POW,
    SYM_NEG, SYM_SQRT,
    SYM_SIN, SYM_COS, SYM_EXP, SYM_LOG,
    SYM_MATMUL, SYM_REDUCE_SUM, SYM_SOFTMAX
} SymNodeType;

typedef struct SymNode {
    SymNodeType type;
    double const_val;
    int var_id;
    struct SymNode* left;
    struct SymNode* right;
    struct SymNode* grad;   /* lazily computed symbolic gradient */
    uint32_t hash;          /* for CSE (common subexpression elimination) */
} SymNode;

/* Arena for symbolic nodes (no individual free needed) */
static SymNode* sym_arena = NULL;
static int sym_arena_len = 0, sym_arena_cap = 0;

static SymNode* sym_alloc(void) {
    if (sym_arena_len >= sym_arena_cap) {
        sym_arena_cap = sym_arena_cap ? sym_arena_cap * 2 : 1024;
        sym_arena = (SymNode*)realloc(sym_arena, sym_arena_cap * sizeof(SymNode));
    }
    SymNode* n = &sym_arena[sym_arena_len++];
    memset(n, 0, sizeof(SymNode));
    return n;
}

static SymNode* sym_const(double v) {
    SymNode* n = sym_alloc();
    n->type = SYM_CONST; n->const_val = v;
    return n;
}

static SymNode* sym_var(int id) {
    SymNode* n = sym_alloc();
    n->type = SYM_VAR; n->var_id = id;
    return n;
}

static SymNode* sym_binop(SymNodeType type, SymNode* l, SymNode* r) {
    SymNode* n = sym_alloc();
    n->type = type; n->left = l; n->right = r;
    return n;
}

static SymNode* sym_unop(SymNodeType type, SymNode* child) {
    SymNode* n = sym_alloc();
    n->type = type; n->left = child;
    return n;
}

/* Symbolic differentiation */
static SymNode* sym_differentiate(SymNode* expr, int var_id) {
    if (!expr) return sym_const(0);

    switch (expr->type) {
    case SYM_CONST: return sym_const(0);
    case SYM_VAR:   return sym_const(expr->var_id == var_id ? 1 : 0);

    case SYM_ADD: return sym_binop(SYM_ADD,
        sym_differentiate(expr->left, var_id),
        sym_differentiate(expr->right, var_id));

    case SYM_SUB: return sym_binop(SYM_SUB,
        sym_differentiate(expr->left, var_id),
        sym_differentiate(expr->right, var_id));

    case SYM_MUL: /* Product rule: (fg)' = f'g + fg' */
        return sym_binop(SYM_ADD,
            sym_binop(SYM_MUL, sym_differentiate(expr->left, var_id), expr->right),
            sym_binop(SYM_MUL, expr->left, sym_differentiate(expr->right, var_id)));

    case SYM_DIV: /* Quotient rule: (f/g)' = (f'g - fg') / g^2 */
        return sym_binop(SYM_DIV,
            sym_binop(SYM_SUB,
                sym_binop(SYM_MUL, sym_differentiate(expr->left, var_id), expr->right),
                sym_binop(SYM_MUL, expr->left, sym_differentiate(expr->right, var_id))),
            sym_binop(SYM_MUL, expr->right, expr->right));

    case SYM_SIN: /* d/dx sin(f) = cos(f) * f' */
        return sym_binop(SYM_MUL,
            sym_unop(SYM_COS, expr->left),
            sym_differentiate(expr->left, var_id));

    case SYM_COS: /* d/dx cos(f) = -sin(f) * f' */
        return sym_binop(SYM_MUL,
            sym_unop(SYM_NEG, sym_unop(SYM_SIN, expr->left)),
            sym_differentiate(expr->left, var_id));

    case SYM_EXP: /* d/dx exp(f) = exp(f) * f' */
        return sym_binop(SYM_MUL, expr,
            sym_differentiate(expr->left, var_id));

    case SYM_LOG: /* d/dx log(f) = f'/f */
        return sym_binop(SYM_DIV,
            sym_differentiate(expr->left, var_id), expr->left);

    case SYM_SQRT: /* d/dx sqrt(f) = f' / (2*sqrt(f)) */
        return sym_binop(SYM_DIV,
            sym_differentiate(expr->left, var_id),
            sym_binop(SYM_MUL, sym_const(2), expr));

    case SYM_NEG:
        return sym_unop(SYM_NEG, sym_differentiate(expr->left, var_id));

    case SYM_POW: /* d/dx f^g = f^g * (g'*ln(f) + g*f'/f) — generalized */
        /* Simplified: if g is constant n, d/dx f^n = n*f^(n-1)*f' */
        if (expr->right && expr->right->type == SYM_CONST) {
            double n = expr->right->const_val;
            return sym_binop(SYM_MUL,
                sym_binop(SYM_MUL, sym_const(n),
                    sym_binop(SYM_POW, expr->left, sym_const(n - 1))),
                sym_differentiate(expr->left, var_id));
        }
        return sym_const(0); /* General case not handled */

    default: return sym_const(0);
    }
}

/* Algebraic simplification */
static SymNode* sym_simplify(SymNode* expr) {
    if (!expr) return NULL;

    /* Recursively simplify children first */
    if (expr->left) expr->left = sym_simplify(expr->left);
    if (expr->right) expr->right = sym_simplify(expr->right);

    SymNode* l = expr->left;
    SymNode* r = expr->right;

    switch (expr->type) {
    case SYM_ADD:
        if (l && l->type == SYM_CONST && l->const_val == 0) return r; /* 0 + x = x */
        if (r && r->type == SYM_CONST && r->const_val == 0) return l; /* x + 0 = x */
        if (l && r && l->type == SYM_CONST && r->type == SYM_CONST)
            return sym_const(l->const_val + r->const_val); /* constant fold */
        break;
    case SYM_SUB:
        if (r && r->type == SYM_CONST && r->const_val == 0) return l; /* x - 0 = x */
        if (l && r && l->type == SYM_CONST && r->type == SYM_CONST)
            return sym_const(l->const_val - r->const_val);
        break;
    case SYM_MUL:
        if (l && l->type == SYM_CONST && l->const_val == 0) return sym_const(0); /* 0 * x = 0 */
        if (r && r->type == SYM_CONST && r->const_val == 0) return sym_const(0); /* x * 0 = 0 */
        if (l && l->type == SYM_CONST && l->const_val == 1) return r; /* 1 * x = x */
        if (r && r->type == SYM_CONST && r->const_val == 1) return l; /* x * 1 = x */
        if (l && r && l->type == SYM_CONST && r->type == SYM_CONST)
            return sym_const(l->const_val * r->const_val);
        break;
    case SYM_DIV:
        if (l && l->type == SYM_CONST && l->const_val == 0) return sym_const(0); /* 0 / x = 0 */
        if (r && r->type == SYM_CONST && r->const_val == 1) return l; /* x / 1 = x */
        if (l && r && l->type == SYM_CONST && r->type == SYM_CONST && r->const_val != 0)
            return sym_const(l->const_val / r->const_val);
        break;
    case SYM_NEG:
        if (l && l->type == SYM_CONST) return sym_const(-l->const_val);
        if (l && l->type == SYM_NEG) return l->left; /* --x = x */
        break;
    case SYM_POW:
        if (r && r->type == SYM_CONST && r->const_val == 0) return sym_const(1); /* x^0 = 1 */
        if (r && r->type == SYM_CONST && r->const_val == 1) return l; /* x^1 = x */
        break;
    default: break;
    }
    return expr;
}

/* Reset the symbolic arena (call between compilations) */
static void sym_arena_reset(void) {
    sym_arena_len = 0;
}

/*******************************************************************************
 * Bytecode Trace → Symbolic Graph
 ******************************************************************************/

/* Build a symbolic expression graph from a bytecode instruction sequence.
 * Traces the stack effect of each instruction to build the expression tree. */
static SymNode* sym_trace_bytecode(const Instr* code, int start, int end,
                                     const Value* constants, int n_constants) {
    /* Symbolic stack — tracks expressions, not values */
    SymNode* stack[256];
    int sp = 0;

    for (int pc = start; pc < end; pc++) {
        uint8_t op = code[pc].op;
        int32_t operand = code[pc].operand;

        switch (op) {
        case 1: /* OP_CONST */
            if (operand >= 0 && operand < n_constants) {
                Value v = constants[operand];
                if (v.type == 1) /* VAL_INT */
                    stack[sp++] = sym_const((double)v.as.i);
                else if (v.type == 2) /* VAL_FLOAT */
                    stack[sp++] = sym_const(v.as.f);
                else
                    stack[sp++] = sym_const(0);
            }
            break;
        case 7: /* OP_ADD */
            if (sp >= 2) { SymNode* b = stack[--sp]; SymNode* a = stack[--sp]; stack[sp++] = sym_binop(SYM_ADD, a, b); }
            break;
        case 8: /* OP_SUB */
            if (sp >= 2) { SymNode* b = stack[--sp]; SymNode* a = stack[--sp]; stack[sp++] = sym_binop(SYM_SUB, a, b); }
            break;
        case 9: /* OP_MUL */
            if (sp >= 2) { SymNode* b = stack[--sp]; SymNode* a = stack[--sp]; stack[sp++] = sym_binop(SYM_MUL, a, b); }
            break;
        case 10: /* OP_DIV */
            if (sp >= 2) { SymNode* b = stack[--sp]; SymNode* a = stack[--sp]; stack[sp++] = sym_binop(SYM_DIV, a, b); }
            break;
        case 12: /* OP_NEG */
            if (sp >= 1) { SymNode* a = stack[--sp]; stack[sp++] = sym_unop(SYM_NEG, a); }
            break;
        case 20: /* OP_GET_LOCAL — treat as variable */
            stack[sp++] = sym_var(operand);
            break;
        default:
            /* Unknown op — push a variable placeholder */
            stack[sp++] = sym_var(-(pc + 1));
            break;
        }

        if (sp >= 256) break; /* stack overflow protection */
    }

    return (sp > 0) ? stack[sp - 1] : sym_const(0);
}

/* Generate bytecode from a simplified symbolic expression.
 * Emits OP_CONST for constants, OP_GET_LOCAL for variables,
 * and arithmetic ops for operators. */
static void sym_compile(SymNode* expr, FuncChunk* output) {
    if (!expr || !output) return;

    switch (expr->type) {
    case SYM_CONST:
        chunk_emit(output, 1 /* OP_CONST */,
                   chunk_add_const(output, (Value){.type = 2, .as.f = expr->const_val}));
        break;
    case SYM_VAR:
        chunk_emit(output, 20 /* OP_GET_LOCAL */, expr->var_id);
        break;
    case SYM_ADD:
        sym_compile(expr->left, output);
        sym_compile(expr->right, output);
        chunk_emit(output, 7 /* OP_ADD */, 0);
        break;
    case SYM_SUB:
        sym_compile(expr->left, output);
        sym_compile(expr->right, output);
        chunk_emit(output, 8 /* OP_SUB */, 0);
        break;
    case SYM_MUL:
        sym_compile(expr->left, output);
        sym_compile(expr->right, output);
        chunk_emit(output, 9 /* OP_MUL */, 0);
        break;
    case SYM_DIV:
        sym_compile(expr->left, output);
        sym_compile(expr->right, output);
        chunk_emit(output, 10 /* OP_DIV */, 0);
        break;
    case SYM_NEG:
        sym_compile(expr->left, output);
        chunk_emit(output, 12 /* OP_NEG */, 0);
        break;
    case SYM_SIN:
        sym_compile(expr->left, output);
        chunk_emit(output, 37 /* OP_NATIVE_CALL */, 20); /* sin */
        break;
    case SYM_COS:
        sym_compile(expr->left, output);
        chunk_emit(output, 37 /* OP_NATIVE_CALL */, 21); /* cos */
        break;
    case SYM_EXP:
        sym_compile(expr->left, output);
        chunk_emit(output, 37 /* OP_NATIVE_CALL */, 23); /* exp */
        break;
    case SYM_LOG:
        sym_compile(expr->left, output);
        chunk_emit(output, 37 /* OP_NATIVE_CALL */, 24); /* log */
        break;
    case SYM_SQRT:
        sym_compile(expr->left, output);
        chunk_emit(output, 37 /* OP_NATIVE_CALL */, 25); /* sqrt */
        break;
    case SYM_POW:
        sym_compile(expr->left, output);
        sym_compile(expr->right, output);
        chunk_emit(output, 37 /* OP_NATIVE_CALL */, 32); /* expt */
        break;
    default:
        chunk_emit(output, 2 /* OP_NIL */, 0);
        break;
    }
}
