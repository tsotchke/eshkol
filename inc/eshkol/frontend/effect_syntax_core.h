/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_FRONTEND_EFFECT_SYNTAX_CORE_H
#define ESHKOL_FRONTEND_EFFECT_SYNTAX_CORE_H

/**
 * @file effect_syntax_core.h
 * @brief The one lowering of the algebraic-effect forms (ADR-0045).
 *
 * `define-effect`, `handle`, `handle-shallow` and `declare-effects` are
 * rewritten here, over the neutral syntax tree of syntax_rules_core.h, into
 * ordinary Scheme that calls the effect runtime (lib/core/control/effects.esk).
 * Both engines run this same source: the native parser converts its reader
 * syntax (SyntaxDatum) to `eshkol_syn` and back, and the bytecode VM converts
 * its reader nodes. Neither engine has its own copy of the rules.
 *
 *   (define-effect Name (op formals...) ...)
 *     => (begin (define Name (%effect-make 'Name))
 *               (define op (%effect-operation Name 'op '(formals...))) ...)
 *
 *   (handle body clause...)          deep handler
 *   (handle-shallow body clause...)  shallow handler
 *     clause ::= (return (x) e...)               applied to a normal result
 *              | ((op formals...) k e...)        k is the resumption
 *              | (fun (op formals...) e...)      resumes with the value of e...
 *     => (%effect-handle 'light|'full 'deep|'shallow <return>
 *                        (list (%effect-clause op '<kind> (lambda ...)) ...)
 *                        (lambda () body))
 *
 *   (declare-effects name (Label ...))  => (%effect-declare 'name '(Label ...))
 *
 * Every operation clause is classified by how its body uses the resumption k.
 * The classification selects the cheapest runtime strategy that has the
 * same meaning:
 *
 *   abort        k never occurs: the handled computation is abandoned and
 *                the clause's value is the value of the `handle` form. No
 *                continuation is captured at the operation.
 *   tail         every tail position of the body is `(k e)` and k occurs
 *                nowhere else: the clause always resumes, as its last act. It
 *                runs in place at the operation, and `(k e)` becomes `e`. No
 *                continuation is captured. (A `fun` clause is this kind.)
 *   general-tail k occurs only as the operator of a call in tail position,
 *                but some path does not resume: the continuation of the
 *                operation is captured; resuming does not capture the
 *                resumption site, because nothing in the clause follows it.
 *   general      anything else (k stored, passed, called more than once in
 *                sequence, or called in non-tail position): the resumption
 *                site is captured too, so `(k v)` returns the result of the
 *                resumed computation.
 *
 * A handler is `light` when every clause is `tail`: it captures nothing.
 *
 * The form is recognised only when it is shaped like one: a `handle` whose
 * trailing operands are not all clause-shaped is left as an ordinary call, so
 * a program's own procedure named `handle` keeps working.
 *
 * The lowering emits only symbols and lists (never a new atom), because each
 * engine rebuilds atoms from the reader token that produced them.
 *
 * Header-only and C/C++ compatible; every function is static inline.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eshkol/frontend/syntax_rules_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_EFFECT_FORM_NONE = 0,
    ESHKOL_EFFECT_FORM_DEFINE_EFFECT,
    ESHKOL_EFFECT_FORM_HANDLE,
    ESHKOL_EFFECT_FORM_HANDLE_SHALLOW,
    ESHKOL_EFFECT_FORM_DECLARE_EFFECTS
} eshkol_effect_form;

typedef enum {
    ESHKOL_EFFECT_CLAUSE_ABORT = 0,
    ESHKOL_EFFECT_CLAUSE_TAIL,
    ESHKOL_EFFECT_CLAUSE_GENERAL_TAIL,
    ESHKOL_EFFECT_CLAUSE_GENERAL
} eshkol_effect_clause_kind;

static inline const char* eshkol_effect_clause_kind_name(eshkol_effect_clause_kind k) {
    switch (k) {
        case ESHKOL_EFFECT_CLAUSE_ABORT: return "abort";
        case ESHKOL_EFFECT_CLAUSE_TAIL: return "tail";
        case ESHKOL_EFFECT_CLAUSE_GENERAL_TAIL: return "general-tail";
        case ESHKOL_EFFECT_CLAUSE_GENERAL: return "general";
    }
    return "general";
}

/* ── small tree helpers ─────────────────────────────────────────────────── */

static inline int eshkol_effect_is_sym(const eshkol_syn* n, const char* name) {
    return n && n->kind == ESHKOL_SYN_SYMBOL && eshkol_syntax_base_is(n->text, name);
}

static inline int eshkol_effect_is_proper_list(const eshkol_syn* n) {
    return n && n->kind == ESHKOL_SYN_LIST && !n->dotted;
}

static inline const char* eshkol_effect_head_name(const eshkol_syn* n) {
    if (!n || n->kind != ESHKOL_SYN_LIST || n->n_items == 0) return NULL;
    if (n->items[0]->kind != ESHKOL_SYN_SYMBOL) return NULL;
    return n->items[0]->text;
}

static inline eshkol_syn* eshkol_effect_sym(const char* name, const void* origin) {
    (void)origin;
    return eshkol_syn_new(ESHKOL_SYN_SYMBOL, name, NULL);
}

static inline eshkol_syn* eshkol_effect_list(void) {
    return eshkol_syn_new(ESHKOL_SYN_LIST, "(", NULL);
}

/* Push and report failure through *ok. Takes ownership of item. */
static inline void eshkol_effect_push(eshkol_syn* list, eshkol_syn* item, int* ok) {
    if (!item) { *ok = 0; return; }
    if (!eshkol_syn_push(list, item)) { eshkol_syn_free(item); *ok = 0; }
}

/** (quote datum) — datum is copied. */
static inline eshkol_syn* eshkol_effect_quote(const eshkol_syn* datum, int* ok) {
    eshkol_syn* q = eshkol_effect_list();
    if (!q) { *ok = 0; return NULL; }
    eshkol_effect_push(q, eshkol_effect_sym("quote", NULL), ok);
    eshkol_effect_push(q, eshkol_syn_copy(datum), ok);
    return q;
}

static inline eshkol_syn* eshkol_effect_quote_sym(const char* name, int* ok) {
    eshkol_syn* s = eshkol_effect_sym(name, NULL);
    if (!s) { *ok = 0; return NULL; }
    eshkol_syn* q = eshkol_effect_quote(s, ok);
    eshkol_syn_free(s);
    return q;
}

/** A formals list of symbols, possibly dotted with a symbol tail. */
static inline int eshkol_effect_valid_formals(const eshkol_syn* f) {
    if (!f || f->kind != ESHKOL_SYN_LIST) return 0;
    for (int i = 0; i < f->n_items; ++i)
        if (f->items[i]->kind != ESHKOL_SYN_SYMBOL) return 0;
    return 1;
}

/** Copy of an operation pattern's formals: (op a b . r) -> (a b . r). */
static inline eshkol_syn* eshkol_effect_pattern_formals(const eshkol_syn* pattern, int prepend_k,
                                                        const eshkol_syn* k, int* ok) {
    eshkol_syn* f = eshkol_effect_list();
    if (!f) { *ok = 0; return NULL; }
    if (prepend_k) eshkol_effect_push(f, eshkol_syn_copy(k), ok);
    for (int i = 1; i < pattern->n_items; ++i)
        eshkol_effect_push(f, eshkol_syn_copy(pattern->items[i]), ok);
    /* (op . r): the only formal is the dotted tail. */
    if (pattern->dotted) {
        if (pattern->n_items == 2 && !prepend_k) {
            /* (op . r) => formals are the bare symbol r */
            eshkol_syn* tail = eshkol_syn_copy(pattern->items[1]);
            eshkol_syn_free(f);
            return tail;
        }
        f->dotted = 1;
    }
    return f;
}

/* ── shape recognition ──────────────────────────────────────────────────── */

/** An operation pattern: (op formals...) with a symbol head. */
static inline int eshkol_effect_op_pattern(const eshkol_syn* p) {
    if (!p || p->kind != ESHKOL_SYN_LIST || p->n_items == 0) return 0;
    if (p->items[0]->kind != ESHKOL_SYN_SYMBOL) return 0;
    if (p->dotted && p->n_items < 2) return 0;
    return eshkol_effect_valid_formals(p);
}

static inline int eshkol_effect_clause_shaped(const eshkol_syn* c) {
    if (!eshkol_effect_is_proper_list(c) || c->n_items < 2) return 0;
    if (eshkol_effect_is_sym(c->items[0], "return") || eshkol_effect_is_sym(c->items[0], "fun"))
        return 1;
    return eshkol_effect_op_pattern(c->items[0]) && c->items[1]->kind == ESHKOL_SYN_SYMBOL;
}

/** Which effect form @p form is, or NONE. Only the head and the shape are read. */
static inline eshkol_effect_form eshkol_effect_form_of(const eshkol_syn* form) {
    const char* head = eshkol_effect_head_name(form);
    if (!head || form->dotted) return ESHKOL_EFFECT_FORM_NONE;
    if (eshkol_syntax_base_is(head, "define-effect")) {
        if (form->n_items < 2 || form->items[1]->kind != ESHKOL_SYN_SYMBOL)
            return ESHKOL_EFFECT_FORM_NONE;
        for (int i = 2; i < form->n_items; ++i)
            if (!eshkol_effect_op_pattern(form->items[i])) return ESHKOL_EFFECT_FORM_NONE;
        return ESHKOL_EFFECT_FORM_DEFINE_EFFECT;
    }
    if (eshkol_syntax_base_is(head, "handle") || eshkol_syntax_base_is(head, "handle-shallow")) {
        if (form->n_items < 3) return ESHKOL_EFFECT_FORM_NONE;
        for (int i = 2; i < form->n_items; ++i)
            if (!eshkol_effect_clause_shaped(form->items[i])) return ESHKOL_EFFECT_FORM_NONE;
        return eshkol_syntax_base_is(head, "handle") ? ESHKOL_EFFECT_FORM_HANDLE
                                                     : ESHKOL_EFFECT_FORM_HANDLE_SHALLOW;
    }
    if (eshkol_syntax_base_is(head, "declare-effects")) {
        if (form->n_items != 3 || form->items[1]->kind != ESHKOL_SYN_SYMBOL)
            return ESHKOL_EFFECT_FORM_NONE;
        const eshkol_syn* row = form->items[2];
        if (row->kind == ESHKOL_SYN_SYMBOL) return ESHKOL_EFFECT_FORM_DECLARE_EFFECTS;
        if (row->kind != ESHKOL_SYN_LIST) return ESHKOL_EFFECT_FORM_NONE;
        for (int i = 0; i < row->n_items; ++i)
            if (row->items[i]->kind != ESHKOL_SYN_SYMBOL) return ESHKOL_EFFECT_FORM_NONE;
        return ESHKOL_EFFECT_FORM_DECLARE_EFFECTS;
    }
    return ESHKOL_EFFECT_FORM_NONE;
}

/** True when @p name is one of the effect form keywords. */
static inline int eshkol_effect_keyword(const char* name) {
    return name && (eshkol_syntax_base_is(name, "define-effect") ||
                    eshkol_syntax_base_is(name, "handle") ||
                    eshkol_syntax_base_is(name, "handle-shallow") ||
                    eshkol_syntax_base_is(name, "declare-effects"));
}

/* ── resumption-usage analysis ──────────────────────────────────────────── */

typedef struct {
    const char* k;        /* the resumption's name */
    int tail_calls;       /* (k e) in tail position, k not inside e */
    int other_uses;       /* every other occurrence of k */
    int non_k_tail;       /* some tail position does not resume */
    int shadowed;         /* a binder inside the body rebinds k */
} eshkol_effect_k_usage;

static inline int eshkol_effect_is_k(const eshkol_syn* n, const char* k) {
    return n && n->kind == ESHKOL_SYN_SYMBOL && strcmp(n->text, k) == 0;
}

/* Occurrences of k anywhere in n, quoted data excluded. */
static inline int eshkol_effect_count_k(const eshkol_syn* n, const char* k) {
    if (!n) return 0;
    if (n->kind == ESHKOL_SYN_SYMBOL) return strcmp(n->text, k) == 0;
    if (n->kind == ESHKOL_SYN_ATOM) return 0;
    if (n->kind == ESHKOL_SYN_PREFIX) {
        if (strcmp(n->text, "'") == 0) return 0;
        int c = 0;
        for (int i = 0; i < n->n_items; ++i) c += eshkol_effect_count_k(n->items[i], k);
        return c;
    }
    if (n->kind == ESHKOL_SYN_LIST && n->n_items > 0 && eshkol_effect_is_sym(n->items[0], "quote"))
        return 0;
    int c = 0;
    for (int i = 0; i < n->n_items; ++i) c += eshkol_effect_count_k(n->items[i], k);
    return c;
}

/* Does a binding construct in n bind k? Conservative: any lambda/let-family
 * binder list or define naming k. */
static inline int eshkol_effect_binds_k(const eshkol_syn* n, const char* k) {
    if (!n || n->kind != ESHKOL_SYN_LIST) return 0;
    if (n->n_items > 0 && eshkol_effect_is_sym(n->items[0], "quote")) return 0;
    const char* h = eshkol_effect_head_name(n);
    if (h && n->n_items >= 2) {
        const eshkol_syn* b = n->items[1];
        if (eshkol_syntax_base_is(h, "lambda") || eshkol_syntax_base_is(h, "define")) {
            if (eshkol_effect_is_k(b, k)) return 1;
            if (b->kind == ESHKOL_SYN_LIST)
                for (int i = 0; i < b->n_items; ++i)
                    if (eshkol_effect_is_k(b->items[i], k)) return 1;
        } else if (eshkol_syntax_base_is(h, "let") || eshkol_syntax_base_is(h, "let*") ||
                   eshkol_syntax_base_is(h, "letrec") || eshkol_syntax_base_is(h, "letrec*") ||
                   eshkol_syntax_base_is(h, "do")) {
            if (eshkol_effect_is_k(b, k)) return 1;           /* named let */
            const eshkol_syn* binds = b->kind == ESHKOL_SYN_SYMBOL && n->n_items >= 3 ? n->items[2] : b;
            if (binds->kind == ESHKOL_SYN_LIST)
                for (int i = 0; i < binds->n_items; ++i) {
                    const eshkol_syn* bi = binds->items[i];
                    if (bi->kind == ESHKOL_SYN_LIST && bi->n_items > 0 &&
                        eshkol_effect_is_k(bi->items[0], k)) return 1;
                }
        }
    }
    for (int i = 0; i < n->n_items; ++i)
        if (eshkol_effect_binds_k(n->items[i], k)) return 1;
    return 0;
}

static inline void eshkol_effect_walk(const eshkol_syn* n, int tail, eshkol_effect_k_usage* u);

static inline void eshkol_effect_walk_body(const eshkol_syn* list, int from, int tail,
                                           eshkol_effect_k_usage* u) {
    if (from >= list->n_items) { if (tail) u->non_k_tail = 1; return; }
    for (int i = from; i < list->n_items; ++i)
        eshkol_effect_walk(list->items[i], tail && i == list->n_items - 1, u);
}

static inline void eshkol_effect_walk_nontail(const eshkol_syn* n, int tail, eshkol_effect_k_usage* u) {
    u->other_uses += eshkol_effect_count_k(n, u->k);
    if (tail) u->non_k_tail = 1;
}

/* cond / case clause bodies */
static inline void eshkol_effect_walk_clauses(const eshkol_syn* n, int first, int is_case,
                                              int tail, eshkol_effect_k_usage* u) {
    int has_else = 0;
    for (int i = first; i < n->n_items; ++i) {
        const eshkol_syn* c = n->items[i];
        if (!eshkol_effect_is_proper_list(c) || c->n_items == 0) {
            eshkol_effect_walk_nontail(c, tail, u);
            continue;
        }
        int is_else = eshkol_effect_is_sym(c->items[0], "else");
        if (is_else) has_else = 1;
        int arrow = c->n_items >= 2 && eshkol_effect_is_sym(c->items[1], "=>");
        if (arrow) { eshkol_effect_walk_nontail(c, tail, u); continue; }
        if (!is_else && !is_case) eshkol_effect_walk(c->items[0], 0, u);
        if (c->n_items == 1) {               /* (test): the test's value is the result */
            if (tail) u->non_k_tail = 1;
            continue;
        }
        eshkol_effect_walk_body(c, 1, tail, u);
    }
    if (!has_else && tail) u->non_k_tail = 1;
}

static inline void eshkol_effect_walk(const eshkol_syn* n, int tail, eshkol_effect_k_usage* u) {
    if (!n) return;
    if (n->kind == ESHKOL_SYN_SYMBOL) {
        if (strcmp(n->text, u->k) == 0) u->other_uses++;
        if (tail) u->non_k_tail = 1;
        return;
    }
    if (n->kind != ESHKOL_SYN_LIST || n->dotted || n->n_items == 0) {
        eshkol_effect_walk_nontail(n, tail, u);
        return;
    }
    const eshkol_syn* head = n->items[0];
    if (eshkol_effect_is_k(head, u->k)) {
        int inner = 0;
        for (int i = 1; i < n->n_items; ++i) inner += eshkol_effect_count_k(n->items[i], u->k);
        if (tail && n->n_items == 2 && inner == 0) u->tail_calls++;
        else u->other_uses += 1 + inner;
        return;
    }
    const char* h = head->kind == ESHKOL_SYN_SYMBOL ? head->text : NULL;
    if (h && eshkol_syntax_base_is(h, "quote")) { if (tail) u->non_k_tail = 1; return; }
    if (h && eshkol_syntax_base_is(h, "if") && (n->n_items == 3 || n->n_items == 4)) {
        eshkol_effect_walk(n->items[1], 0, u);
        eshkol_effect_walk(n->items[2], tail, u);
        if (n->n_items == 4) eshkol_effect_walk(n->items[3], tail, u);
        else if (tail) u->non_k_tail = 1;
        return;
    }
    if (h && eshkol_syntax_base_is(h, "begin")) { eshkol_effect_walk_body(n, 1, tail, u); return; }
    if (h && (eshkol_syntax_base_is(h, "let") || eshkol_syntax_base_is(h, "let*") ||
              eshkol_syntax_base_is(h, "letrec") || eshkol_syntax_base_is(h, "letrec*")) &&
        n->n_items >= 3 && n->items[1]->kind == ESHKOL_SYN_LIST) {
        const eshkol_syn* binds = n->items[1];
        for (int i = 0; i < binds->n_items; ++i) {
            const eshkol_syn* b = binds->items[i];
            if (b->kind == ESHKOL_SYN_LIST)
                for (int j = 1; j < b->n_items; ++j) eshkol_effect_walk(b->items[j], 0, u);
            else
                eshkol_effect_walk(b, 0, u);
        }
        eshkol_effect_walk_body(n, 2, tail, u);
        return;
    }
    if (h && eshkol_syntax_base_is(h, "cond")) { eshkol_effect_walk_clauses(n, 1, 0, tail, u); return; }
    if (h && eshkol_syntax_base_is(h, "case") && n->n_items >= 2) {
        eshkol_effect_walk(n->items[1], 0, u);
        eshkol_effect_walk_clauses(n, 2, 1, tail, u);
        return;
    }
    if (h && (eshkol_syntax_base_is(h, "when") || eshkol_syntax_base_is(h, "unless")) &&
        n->n_items >= 2) {
        eshkol_effect_walk(n->items[1], 0, u);
        eshkol_effect_walk_body(n, 2, tail, u);
        if (tail) u->non_k_tail = 1;           /* the test can fail */
        return;
    }
    eshkol_effect_walk_nontail(n, tail, u);
}

/** Classify one operation clause body (items [from, n) of @p clause). */
static inline eshkol_effect_clause_kind eshkol_effect_classify(const eshkol_syn* clause, int from,
                                                               const char* k) {
    eshkol_effect_k_usage u;
    memset(&u, 0, sizeof(u));
    u.k = k;
    for (int i = from; i < clause->n_items; ++i)
        if (eshkol_effect_binds_k(clause->items[i], k)) u.shadowed = 1;
    eshkol_effect_walk_body(clause, from, 1, &u);
    if (u.shadowed) return ESHKOL_EFFECT_CLAUSE_GENERAL;
    if (u.tail_calls == 0 && u.other_uses == 0) return ESHKOL_EFFECT_CLAUSE_ABORT;
    if (u.other_uses == 0 && !u.non_k_tail) return ESHKOL_EFFECT_CLAUSE_TAIL;
    if (u.other_uses == 0) return ESHKOL_EFFECT_CLAUSE_GENERAL_TAIL;
    return ESHKOL_EFFECT_CLAUSE_GENERAL;
}

/* Rewrite every tail `(k e)` of n to `e` (valid only for TAIL clauses). */
static inline eshkol_syn* eshkol_effect_strip_k(const eshkol_syn* n, const char* k, int* ok);

static inline eshkol_syn* eshkol_effect_strip_body(const eshkol_syn* n, int from, const char* k,
                                                   int* ok) {
    eshkol_syn* out = eshkol_syn_new(n->kind, n->text, n->origin);
    if (!out) { *ok = 0; return NULL; }
    for (int i = 0; i < n->n_items; ++i) {
        if (i >= from && i == n->n_items - 1)
            eshkol_effect_push(out, eshkol_effect_strip_k(n->items[i], k, ok), ok);
        else
            eshkol_effect_push(out, eshkol_syn_copy(n->items[i]), ok);
    }
    return out;
}

static inline eshkol_syn* eshkol_effect_strip_k(const eshkol_syn* n, const char* k, int* ok) {
    if (!n || n->kind != ESHKOL_SYN_LIST || n->n_items == 0) return eshkol_syn_copy(n);
    if (eshkol_effect_is_k(n->items[0], k) && n->n_items == 2) return eshkol_syn_copy(n->items[1]);
    const char* h = eshkol_effect_head_name(n);
    if (!h) return eshkol_syn_copy(n);
    if (eshkol_syntax_base_is(h, "if") && (n->n_items == 3 || n->n_items == 4)) {
        eshkol_syn* out = eshkol_syn_new(n->kind, n->text, n->origin);
        if (!out) { *ok = 0; return NULL; }
        eshkol_effect_push(out, eshkol_syn_copy(n->items[0]), ok);
        eshkol_effect_push(out, eshkol_syn_copy(n->items[1]), ok);
        for (int i = 2; i < n->n_items; ++i)
            eshkol_effect_push(out, eshkol_effect_strip_k(n->items[i], k, ok), ok);
        return out;
    }
    if (eshkol_syntax_base_is(h, "begin")) return eshkol_effect_strip_body(n, 1, k, ok);
    if (eshkol_syntax_base_is(h, "let") || eshkol_syntax_base_is(h, "let*") ||
        eshkol_syntax_base_is(h, "letrec") || eshkol_syntax_base_is(h, "letrec*"))
        return eshkol_effect_strip_body(n, 2, k, ok);
    if (eshkol_syntax_base_is(h, "cond") || eshkol_syntax_base_is(h, "case")) {
        int first = eshkol_syntax_base_is(h, "cond") ? 1 : 2;
        eshkol_syn* out = eshkol_syn_new(n->kind, n->text, n->origin);
        if (!out) { *ok = 0; return NULL; }
        for (int i = 0; i < n->n_items; ++i) {
            const eshkol_syn* c = n->items[i];
            if (i >= first && eshkol_effect_is_proper_list(c) && c->n_items >= 2)
                eshkol_effect_push(out, eshkol_effect_strip_body(c, 1, k, ok), ok);
            else
                eshkol_effect_push(out, eshkol_syn_copy(c), ok);
        }
        return out;
    }
    return eshkol_syn_copy(n);
}

/* ── lowering ───────────────────────────────────────────────────────────── */

static inline void eshkol_effect_set_error(char* err, size_t errlen, const char* msg,
                                           const char* detail) {
    if (!err || errlen == 0) return;
    if (detail) snprintf(err, errlen, "%s: %s", msg, detail);
    else snprintf(err, errlen, "%s", msg);
}

/* (begin ...) of a body slice, or the single expression. */
static inline eshkol_syn* eshkol_effect_body(const eshkol_syn* list, int from, int* ok) {
    if (list->n_items - from == 1) return eshkol_syn_copy(list->items[from]);
    eshkol_syn* b = eshkol_effect_list();
    if (!b) { *ok = 0; return NULL; }
    eshkol_effect_push(b, eshkol_effect_sym("begin", NULL), ok);
    for (int i = from; i < list->n_items; ++i) eshkol_effect_push(b, eshkol_syn_copy(list->items[i]), ok);
    return b;
}

/* (lambda formals body...) — formals owned by the caller, body slice copied */
static inline eshkol_syn* eshkol_effect_lambda(eshkol_syn* formals, const eshkol_syn* list, int from,
                                               int* ok) {
    eshkol_syn* l = eshkol_effect_list();
    if (!l) { eshkol_syn_free(formals); *ok = 0; return NULL; }
    eshkol_effect_push(l, eshkol_effect_sym("lambda", NULL), ok);
    eshkol_effect_push(l, formals, ok);
    for (int i = from; i < list->n_items; ++i) eshkol_effect_push(l, eshkol_syn_copy(list->items[i]), ok);
    return l;
}

static inline eshkol_syn* eshkol_effect_lower_define_effect(const eshkol_syn* form, int* ok) {
    const eshkol_syn* name = form->items[1];
    eshkol_syn* out = eshkol_effect_list();
    if (!out) { *ok = 0; return NULL; }
    eshkol_effect_push(out, eshkol_effect_sym("begin", NULL), ok);

    eshkol_syn* def = eshkol_effect_list();
    if (!def) { eshkol_syn_free(out); *ok = 0; return NULL; }
    eshkol_effect_push(def, eshkol_effect_sym("define", NULL), ok);
    eshkol_effect_push(def, eshkol_syn_copy(name), ok);
    eshkol_syn* mk = eshkol_effect_list();
    if (mk) {
        eshkol_effect_push(mk, eshkol_effect_sym("%effect-make", NULL), ok);
        eshkol_effect_push(mk, eshkol_effect_quote(name, ok), ok);
    }
    eshkol_effect_push(def, mk, ok);
    eshkol_effect_push(out, def, ok);

    for (int i = 2; i < form->n_items; ++i) {
        const eshkol_syn* pat = form->items[i];
        eshkol_syn* d = eshkol_effect_list();
        if (!d) { *ok = 0; break; }
        eshkol_effect_push(d, eshkol_effect_sym("define", NULL), ok);
        eshkol_effect_push(d, eshkol_syn_copy(pat->items[0]), ok);
        eshkol_syn* call = eshkol_effect_list();
        if (call) {
            eshkol_effect_push(call, eshkol_effect_sym("%effect-operation", NULL), ok);
            eshkol_effect_push(call, eshkol_syn_copy(name), ok);
            eshkol_effect_push(call, eshkol_effect_quote(pat->items[0], ok), ok);
            eshkol_syn* formals = eshkol_effect_pattern_formals(pat, 0, NULL, ok);
            if (formals) {
                eshkol_effect_push(call, eshkol_effect_quote(formals, ok), ok);
                eshkol_syn_free(formals);
            }
        }
        eshkol_effect_push(d, call, ok);
        eshkol_effect_push(out, d, ok);
    }
    /* The value of the form is the effect's name, so a REPL echoes something. */
    eshkol_effect_push(out, eshkol_effect_quote(name, ok), ok);
    return out;
}

static inline eshkol_syn* eshkol_effect_lower_handle(const eshkol_syn* form, int shallow,
                                                     char* err, size_t errlen, int* ok) {
    eshkol_syn* ret = NULL;
    eshkol_syn* clauses = eshkol_effect_list();
    int all_tail = 1;
    if (!clauses) { *ok = 0; return NULL; }
    eshkol_effect_push(clauses, eshkol_effect_sym("list", NULL), ok);

    for (int i = 2; i < form->n_items && *ok; ++i) {
        const eshkol_syn* c = form->items[i];
        if (eshkol_effect_is_sym(c->items[0], "return")) {
            const eshkol_syn* f = c->items[1];
            if (ret || !eshkol_effect_is_proper_list(f) || f->n_items != 1 ||
                f->items[0]->kind != ESHKOL_SYN_SYMBOL || c->n_items < 3) {
                eshkol_effect_set_error(err, errlen,
                    ret ? "handle has more than one return clause"
                        : "a return clause is (return (x) body...)", NULL);
                eshkol_syn_free(ret); eshkol_syn_free(clauses);
                return NULL;
            }
            ret = eshkol_effect_lambda(eshkol_syn_copy(f), c, 2, ok);
            continue;
        }
        const int is_fun = eshkol_effect_is_sym(c->items[0], "fun");
        const eshkol_syn* pat = is_fun ? c->items[1] : c->items[0];
        if (!eshkol_effect_op_pattern(pat) || (!is_fun && c->n_items < 3) || (is_fun && c->n_items < 3)) {
            eshkol_effect_set_error(err, errlen,
                "an operation clause is ((op args...) k body...) or (fun (op args...) body...)", NULL);
            eshkol_syn_free(ret); eshkol_syn_free(clauses);
            return NULL;
        }
        /* duplicate operation clauses */
        for (int j = 2; j < i; ++j) {
            const eshkol_syn* o = form->items[j];
            if (eshkol_effect_is_sym(o->items[0], "return")) continue;
            const eshkol_syn* op = eshkol_effect_is_sym(o->items[0], "fun") ? o->items[1] : o->items[0];
            if (strcmp(op->items[0]->text, pat->items[0]->text) == 0) {
                eshkol_effect_set_error(err, errlen, "handle has two clauses for operation",
                                        pat->items[0]->text);
                eshkol_syn_free(ret); eshkol_syn_free(clauses);
                return NULL;
            }
        }

        eshkol_effect_clause_kind kind;
        eshkol_syn* proc = NULL;
        if (is_fun) {
            if (!shallow) {
                kind = ESHKOL_EFFECT_CLAUSE_TAIL;
                proc = eshkol_effect_lambda(eshkol_effect_pattern_formals(pat, 0, NULL, ok), c, 2, ok);
            } else {
                /* a shallow handler leaves after one operation: (k (begin body...)) */
                kind = ESHKOL_EFFECT_CLAUSE_GENERAL_TAIL;
                eshkol_syn* k = eshkol_effect_sym("%effect-k", NULL);
                eshkol_syn* call = eshkol_effect_list();
                eshkol_syn* wrapper = eshkol_effect_list();
                if (!k || !call || !wrapper) { *ok = 0; eshkol_syn_free(k); eshkol_syn_free(call); eshkol_syn_free(wrapper); break; }
                eshkol_effect_push(call, eshkol_syn_copy(k), ok);
                eshkol_effect_push(call, eshkol_effect_body(c, 2, ok), ok);
                eshkol_effect_push(wrapper, eshkol_effect_sym("%effect-clause-body", NULL), ok);
                eshkol_effect_push(wrapper, call, ok);
                proc = eshkol_effect_lambda(eshkol_effect_pattern_formals(pat, 1, k, ok), wrapper, 1, ok);
                eshkol_syn_free(wrapper);
                eshkol_syn_free(k);
            }
        } else {
            const eshkol_syn* k = c->items[1];
            kind = eshkol_effect_classify(c, 2, k->text);
            if (shallow && kind == ESHKOL_EFFECT_CLAUSE_TAIL) kind = ESHKOL_EFFECT_CLAUSE_GENERAL_TAIL;
            if (kind == ESHKOL_EFFECT_CLAUSE_TAIL) {
                eshkol_syn* stripped = eshkol_effect_strip_body(c, 2, k->text, ok);
                proc = stripped ? eshkol_effect_lambda(eshkol_effect_pattern_formals(pat, 0, NULL, ok),
                                                       stripped, 2, ok) : NULL;
                eshkol_syn_free(stripped);
            } else if (kind == ESHKOL_EFFECT_CLAUSE_ABORT) {
                proc = eshkol_effect_lambda(eshkol_effect_pattern_formals(pat, 0, NULL, ok), c, 2, ok);
            } else {
                proc = eshkol_effect_lambda(eshkol_effect_pattern_formals(pat, 1, k, ok), c, 2, ok);
            }
        }
        if (kind != ESHKOL_EFFECT_CLAUSE_TAIL) all_tail = 0;

        eshkol_syn* cl = eshkol_effect_list();
        if (!cl) { eshkol_syn_free(proc); *ok = 0; break; }
        eshkol_effect_push(cl, eshkol_effect_sym("%effect-clause", NULL), ok);
        eshkol_effect_push(cl, eshkol_syn_copy(pat->items[0]), ok);
        eshkol_effect_push(cl, eshkol_effect_quote_sym(eshkol_effect_clause_kind_name(kind), ok), ok);
        eshkol_effect_push(cl, proc, ok);
        eshkol_effect_push(clauses, cl, ok);
    }
    if (!*ok) { eshkol_syn_free(ret); eshkol_syn_free(clauses); return NULL; }

    eshkol_syn* out = eshkol_effect_list();
    if (!out) { eshkol_syn_free(ret); eshkol_syn_free(clauses); *ok = 0; return NULL; }
    eshkol_effect_push(out, eshkol_effect_sym("%effect-handle", NULL), ok);
    eshkol_effect_push(out, eshkol_effect_quote_sym(all_tail ? "light" : "full", ok), ok);
    eshkol_effect_push(out, eshkol_effect_quote_sym(shallow ? "shallow" : "deep", ok), ok);
    eshkol_effect_push(out, ret ? ret : eshkol_effect_sym("%effect-identity", NULL), ok);
    eshkol_effect_push(out, clauses, ok);
    eshkol_syn* none = eshkol_effect_list();
    eshkol_effect_push(out, none ? eshkol_effect_lambda(none, form, 1, ok) : NULL, ok);
    /* the body is exactly one expression: trim the lambda to items[1] */
    if (*ok) {
        eshkol_syn* thunk = out->items[out->n_items - 1];
        while (thunk->n_items > 3) eshkol_syn_free(thunk->items[--thunk->n_items]);
    }
    return out;
}

static inline eshkol_syn* eshkol_effect_lower_declare(const eshkol_syn* form, int* ok) {
    eshkol_syn* out = eshkol_effect_list();
    if (!out) { *ok = 0; return NULL; }
    eshkol_effect_push(out, eshkol_effect_sym("%effect-declare", NULL), ok);
    eshkol_effect_push(out, eshkol_effect_quote(form->items[1], ok), ok);
    eshkol_effect_push(out, eshkol_effect_quote(form->items[2], ok), ok);
    return out;
}

/**
 * Lower one effect form. Returns a new tree (caller frees), or NULL with a
 * message in @p err when the form is shaped like an effect form but invalid,
 * or when memory ran out. Call only when eshkol_effect_form_of() != NONE.
 */
static inline eshkol_syn* eshkol_effect_lower(const eshkol_syn* form, char* err, size_t errlen) {
    int ok = 1;
    eshkol_syn* out = NULL;
    if (err && errlen) err[0] = '\0';
    switch (eshkol_effect_form_of(form)) {
        case ESHKOL_EFFECT_FORM_DEFINE_EFFECT:
            out = eshkol_effect_lower_define_effect(form, &ok);
            break;
        case ESHKOL_EFFECT_FORM_HANDLE:
            out = eshkol_effect_lower_handle(form, 0, err, errlen, &ok);
            break;
        case ESHKOL_EFFECT_FORM_HANDLE_SHALLOW:
            out = eshkol_effect_lower_handle(form, 1, err, errlen, &ok);
            break;
        case ESHKOL_EFFECT_FORM_DECLARE_EFFECTS:
            out = eshkol_effect_lower_declare(form, &ok);
            break;
        default:
            eshkol_effect_set_error(err, errlen, "not an effect form", NULL);
            return NULL;
    }
    if (!ok || !out) {
        if (err && errlen && !err[0]) eshkol_effect_set_error(err, errlen, "out of memory lowering an effect form", NULL);
        eshkol_syn_free(out);
        return NULL;
    }
    return out;
}

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_FRONTEND_EFFECT_SYNTAX_CORE_H */
