/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_FRONTEND_SYNTAX_RULES_CORE_H
#define ESHKOL_FRONTEND_SYNTAX_RULES_CORE_H

/**
 * @file syntax_rules_core.h
 * @brief The one `syntax-rules` engine (ADR-0026).
 *
 * R7RS 4.3.2 pattern matching and template instantiation over a neutral
 * syntax tree, shared verbatim by both engines:
 *
 *   - the native expander (lib/frontend/syntax_rules.cpp) converts the
 *     parser's reader syntax (SyntaxDatum) to and from `eshkol_syn`;
 *   - the bytecode VM (lib/backend/vm_macro.c) converts its reader nodes.
 *
 * The engine applies the one renaming rule of syntax_color.h: every
 * identifier a template introduces, outside quoted data, is emitted with the
 * expansion's color; pattern variables are replaced by the caller's syntax
 * unchanged. What a colored identifier then denotes is each engine's name
 * resolution, specified by that same rule.
 *
 * Supported pattern language: literals, `_`, a custom ellipsis, an ellipsis
 * anywhere in a list or vector followed by further patterns, dotted tail
 * patterns, nested ellipses, and atoms compared by spelling. Templates:
 * ellipsis-following elements (several ellipses flatten), `(... ...)`
 * escapes, dotted tails, and vectors.
 *
 * Header-only and C/C++ compatible so the VM stays a single C translation
 * unit (the browser and freestanding VM builds compile no C++). Every
 * function is static inline: each engine carries its own copy of ONE source.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eshkol/frontend/syntax_color.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_SYN_SYMBOL = 0,
    ESHKOL_SYN_ATOM   = 1,   /* compared by `text`; rebuilt from `origin` */
    ESHKOL_SYN_LIST   = 2,   /* items; the last one is the tail if `dotted` */
    ESHKOL_SYN_VECTOR = 3,
    ESHKOL_SYN_PREFIX = 4    /* reader prefix: text is ' ` , or ,@ ; items[0] */
} eshkol_syn_kind;

typedef struct eshkol_syn {
    eshkol_syn_kind kind;
    char* text;                 /* owned; symbol name, atom key, prefix marker */
    int dotted;
    struct eshkol_syn** items;  /* owned */
    int n_items;
    int cap_items;
    const void* origin;         /* borrowed: the engine user's source object */
} eshkol_syn;

static inline char* eshkol_syn_strdup(const char* text) {
    size_t n = text ? strlen(text) : 0;
    char* copy = (char*)malloc(n + 1);
    if (!copy) return NULL;
    if (n) memcpy(copy, text, n);
    copy[n] = '\0';
    return copy;
}

static inline eshkol_syn* eshkol_syn_new(eshkol_syn_kind kind, const char* text, const void* origin) {
    eshkol_syn* node = (eshkol_syn*)calloc(1, sizeof(eshkol_syn));
    if (!node) return NULL;
    node->kind = kind;
    node->text = eshkol_syn_strdup(text ? text : "");
    node->origin = origin;
    if (!node->text) { free(node); return NULL; }
    return node;
}

static inline int eshkol_syn_push(eshkol_syn* list, eshkol_syn* item) {
    if (!list || !item) return 0;
    if (list->n_items == list->cap_items) {
        int cap = list->cap_items ? list->cap_items * 2 : 4;
        eshkol_syn** grown = (eshkol_syn**)realloc(list->items, (size_t)cap * sizeof(eshkol_syn*));
        if (!grown) return 0;
        list->items = grown;
        list->cap_items = cap;
    }
    list->items[list->n_items++] = item;
    return 1;
}

static inline void eshkol_syn_free(eshkol_syn* node) {
    if (!node) return;
    for (int i = 0; i < node->n_items; ++i) eshkol_syn_free(node->items[i]);
    free(node->items);
    free(node->text);
    free(node);
}

static inline eshkol_syn* eshkol_syn_copy(const eshkol_syn* node) {
    if (!node) return NULL;
    eshkol_syn* copy = eshkol_syn_new(node->kind, node->text, node->origin);
    if (!copy) return NULL;
    copy->dotted = node->dotted;
    for (int i = 0; i < node->n_items; ++i) {
        if (!eshkol_syn_push(copy, eshkol_syn_copy(node->items[i]))) {
            eshkol_syn_free(copy);
            return NULL;
        }
    }
    return copy;
}

/** Number of proper elements of a list or vector (a dotted tail excluded). */
static inline int eshkol_syn_proper(const eshkol_syn* node) {
    return node->dotted && node->n_items > 0 ? node->n_items - 1 : node->n_items;
}

/* ── Pattern-variable environment ──────────────────────────────────────── */

typedef struct eshkol_syn_match {
    int depth;                          /* 0: `value`; N: one match per repetition */
    eshkol_syn* value;                  /* owned copy of the matched syntax */
    struct eshkol_syn_match* reps;
    int n_reps;
    int cap_reps;
} eshkol_syn_match;

typedef struct {
    char* name;
    eshkol_syn_match match;
} eshkol_syn_binding;

typedef struct eshkol_syn_env {
    eshkol_syn_binding* items;
    int n;
    int cap;
    /* A repetition step binds only the iterated variables, borrowing their
     * matches, and looks everything else up in its parent. */
    const struct eshkol_syn_env* parent;
    int borrowed;
} eshkol_syn_env;

static inline void eshkol_syn_match_free(eshkol_syn_match* m) {
    eshkol_syn_free(m->value);
    for (int i = 0; i < m->n_reps; ++i) eshkol_syn_match_free(&m->reps[i]);
    free(m->reps);
    memset(m, 0, sizeof(*m));
}

static inline void eshkol_syn_env_free(eshkol_syn_env* env) {
    for (int i = 0; i < env->n; ++i) {
        free(env->items[i].name);
        if (!env->borrowed) eshkol_syn_match_free(&env->items[i].match);
    }
    free(env->items);
    memset(env, 0, sizeof(*env));
}

static inline eshkol_syn_match* eshkol_syn_env_find(const eshkol_syn_env* env, const char* name) {
    for (; env; env = env->parent)
        for (int i = env->n - 1; i >= 0; --i)
            if (strcmp(env->items[i].name, name) == 0) return &env->items[i].match;
    return NULL;
}

/** Bind @p name, taking ownership of @p match. */
static inline int eshkol_syn_env_bind(eshkol_syn_env* env, const char* name, eshkol_syn_match match) {
    for (int i = 0; i < env->n; ++i) {
        if (strcmp(env->items[i].name, name) != 0) continue;
        if (!env->borrowed) eshkol_syn_match_free(&env->items[i].match);
        env->items[i].match = match;
        return 1;
    }
    if (env->n == env->cap) {
        int cap = env->cap ? env->cap * 2 : 8;
        eshkol_syn_binding* grown =
            (eshkol_syn_binding*)realloc(env->items, (size_t)cap * sizeof(eshkol_syn_binding));
        if (!grown) { if (!env->borrowed) eshkol_syn_match_free(&match); return 0; }
        env->items = grown;
        env->cap = cap;
    }
    env->items[env->n].name = eshkol_syn_strdup(name);
    env->items[env->n].match = match;
    env->n++;
    return 1;
}

static inline int eshkol_syn_match_push_rep(eshkol_syn_match* seq, eshkol_syn_match rep) {
    if (seq->n_reps == seq->cap_reps) {
        int cap = seq->cap_reps ? seq->cap_reps * 2 : 4;
        eshkol_syn_match* grown =
            (eshkol_syn_match*)realloc(seq->reps, (size_t)cap * sizeof(eshkol_syn_match));
        if (!grown) { eshkol_syn_match_free(&rep); return 0; }
        seq->reps = grown;
        seq->cap_reps = cap;
    }
    seq->reps[seq->n_reps++] = rep;
    return 1;
}

/* ── Transformer state ─────────────────────────────────────────────────── */

typedef const char* (*eshkol_syn_keyword_fn)(void* ctx, const char* name);

typedef struct {
    const char* ellipsis;
    const char* const* literals;
    int n_literals;
    unsigned color;
    eshkol_syn_keyword_fn keyword;
    void* keyword_ctx;
    char* error;
    size_t error_cap;
    int failed;
} eshkol_syn_transformer;

static inline void eshkol_syn_fail(eshkol_syn_transformer* t, const char* message, const char* name) {
    t->failed = 1;
    if (t->error && t->error_cap) {
        if (name) snprintf(t->error, t->error_cap, message, name);
        else snprintf(t->error, t->error_cap, "%s", message);
    }
}

static inline int eshkol_syn_is_ellipsis(const eshkol_syn_transformer* t, const eshkol_syn* node) {
    return node && node->kind == ESHKOL_SYN_SYMBOL && strcmp(node->text, t->ellipsis) == 0;
}

static inline int eshkol_syn_is_literal(const eshkol_syn_transformer* t, const char* name) {
    for (int i = 0; i < t->n_literals; ++i)
        if (strcmp(t->literals[i], name) == 0) return 1;
    return 0;
}

static inline const char* eshkol_syn_prefix_name(const char* marker) {
    if (strcmp(marker, "'") == 0) return "quote";
    if (strcmp(marker, "`") == 0) return "quasiquote";
    if (strcmp(marker, ",") == 0) return "unquote";
    if (strcmp(marker, ",@") == 0) return "unquote-splicing";
    return NULL;
}

/** `'x` as the list `(quote x)` (owned result). */
static inline eshkol_syn* eshkol_syn_prefix_as_list(const eshkol_syn* prefix) {
    const char* name = eshkol_syn_prefix_name(prefix->text);
    eshkol_syn* list = eshkol_syn_new(ESHKOL_SYN_LIST, "(", prefix->origin);
    if (!list) return NULL;
    eshkol_syn_push(list, eshkol_syn_new(ESHKOL_SYN_SYMBOL, name ? name : "quote", prefix->origin));
    if (prefix->n_items > 0) eshkol_syn_push(list, eshkol_syn_copy(prefix->items[0]));
    return list;
}

/* Pattern variables of @p pattern with their ellipsis depth relative to it. */
static inline void eshkol_syn_pattern_vars(const eshkol_syn_transformer* t, const eshkol_syn* pattern,
                                    int depth, eshkol_syn_env* out) {
    if (!pattern) return;
    if (pattern->kind == ESHKOL_SYN_SYMBOL) {
        if (strcmp(pattern->text, "_") != 0 && !eshkol_syn_is_ellipsis(t, pattern) &&
            !eshkol_syn_is_literal(t, pattern->text)) {
            eshkol_syn_match m;
            memset(&m, 0, sizeof(m));
            m.depth = depth;
            eshkol_syn_env_bind(out, pattern->text, m);
        }
        return;
    }
    for (int i = 0; i < pattern->n_items; ++i) {
        if (eshkol_syn_is_ellipsis(t, pattern->items[i])) continue;
        int repeated = pattern->kind != ESHKOL_SYN_PREFIX && i + 1 < pattern->n_items &&
                       eshkol_syn_is_ellipsis(t, pattern->items[i + 1]);
        eshkol_syn_pattern_vars(t, pattern->items[i], repeated ? depth + 1 : depth, out);
    }
}

static inline int eshkol_syn_match_sequence(eshkol_syn_transformer* t, const eshkol_syn* pattern,
                                     const eshkol_syn* input, eshkol_syn_env* env);

static inline int eshkol_syn_match_node(eshkol_syn_transformer* t, const eshkol_syn* pattern,
                                 const eshkol_syn* input, eshkol_syn_env* env) {
    switch (pattern->kind) {
        case ESHKOL_SYN_SYMBOL: {
            if (strcmp(pattern->text, "_") == 0) return 1;
            if (eshkol_syn_is_literal(t, pattern->text))
                return input->kind == ESHKOL_SYN_SYMBOL &&
                       eshkol_syntax_same_base(input->text, pattern->text);
            eshkol_syn_match m;
            memset(&m, 0, sizeof(m));
            m.value = eshkol_syn_copy(input);
            return eshkol_syn_env_bind(env, pattern->text, m);
        }
        case ESHKOL_SYN_ATOM:
            return input->kind == ESHKOL_SYN_ATOM && strcmp(input->text, pattern->text) == 0;
        case ESHKOL_SYN_PREFIX: {
            if (input->kind == ESHKOL_SYN_PREFIX && strcmp(input->text, pattern->text) == 0)
                return eshkol_syn_match_node(t, pattern->items[0], input->items[0], env);
            if (input->kind != ESHKOL_SYN_LIST) return 0;
            eshkol_syn* as_list = eshkol_syn_prefix_as_list(pattern);
            int ok = as_list && eshkol_syn_match_node(t, as_list, input, env);
            eshkol_syn_free(as_list);
            return ok;
        }
        case ESHKOL_SYN_LIST:
            if (input->kind == ESHKOL_SYN_PREFIX) {
                eshkol_syn* as_list = eshkol_syn_prefix_as_list(input);
                int ok = as_list && eshkol_syn_match_node(t, pattern, as_list, env);
                eshkol_syn_free(as_list);
                return ok;
            }
            if (input->kind != ESHKOL_SYN_LIST) return 0;
            return eshkol_syn_match_sequence(t, pattern, input, env);
        case ESHKOL_SYN_VECTOR:
            if (input->kind != ESHKOL_SYN_VECTOR) return 0;
            return eshkol_syn_match_sequence(t, pattern, input, env);
    }
    return 0;
}

/* The input elements from @p from onward, with the input's tail (owned). */
static inline eshkol_syn* eshkol_syn_rest(const eshkol_syn* input, int from) {
    const int count = eshkol_syn_proper(input);
    const eshkol_syn* tail = input->dotted ? input->items[input->n_items - 1] : NULL;
    if (from >= count && tail) return eshkol_syn_copy(tail);
    eshkol_syn* rest = eshkol_syn_new(ESHKOL_SYN_LIST, "(", input->origin);
    if (!rest) return NULL;
    for (int i = from; i < count; ++i) eshkol_syn_push(rest, eshkol_syn_copy(input->items[i]));
    if (tail) { eshkol_syn_push(rest, eshkol_syn_copy(tail)); rest->dotted = 1; }
    return rest;
}

static inline int eshkol_syn_match_sequence(eshkol_syn_transformer* t, const eshkol_syn* pattern,
                                     const eshkol_syn* input, eshkol_syn_env* env) {
    const int p_count = eshkol_syn_proper(pattern);
    const eshkol_syn* p_tail = pattern->dotted ? pattern->items[pattern->n_items - 1] : NULL;
    const int i_count = eshkol_syn_proper(input);
    const int i_dotted = input->dotted;

    int ellipsis_at = -1;
    for (int i = 0; i + 1 < p_count; ++i)
        if (eshkol_syn_is_ellipsis(t, pattern->items[i + 1])) { ellipsis_at = i; break; }

    if (ellipsis_at < 0) {
        if (p_tail) {
            if (i_count < p_count) return 0;
            for (int i = 0; i < p_count; ++i)
                if (!eshkol_syn_match_node(t, pattern->items[i], input->items[i], env)) return 0;
            eshkol_syn* rest = eshkol_syn_rest(input, p_count);
            int ok = rest && eshkol_syn_match_node(t, p_tail, rest, env);
            eshkol_syn_free(rest);
            return ok;
        }
        if (i_dotted || i_count != p_count) return 0;
        for (int i = 0; i < p_count; ++i)
            if (!eshkol_syn_match_node(t, pattern->items[i], input->items[i], env)) return 0;
        return 1;
    }

    const int before = ellipsis_at;
    const int after = p_count - ellipsis_at - 2;
    if (!p_tail && i_dotted) return 0;
    if (i_count < before + after) return 0;
    for (int i = 0; i < before; ++i)
        if (!eshkol_syn_match_node(t, pattern->items[i], input->items[i], env)) return 0;

    const int repeat_end = i_count - after;
    const eshkol_syn* repeated = pattern->items[ellipsis_at];
    eshkol_syn_env vars;
    memset(&vars, 0, sizeof(vars));
    eshkol_syn_pattern_vars(t, repeated, 0, &vars);
    /* One sequence per variable of the repeated pattern, even when it
     * matches zero times: a template may still iterate it (to nothing). */
    eshkol_syn_env sequences;
    memset(&sequences, 0, sizeof(sequences));
    for (int v = 0; v < vars.n; ++v) {
        eshkol_syn_match seq;
        memset(&seq, 0, sizeof(seq));
        seq.depth = vars.items[v].match.depth + 1;
        eshkol_syn_env_bind(&sequences, vars.items[v].name, seq);
    }
    int ok = 1;
    for (int i = before; ok && i < repeat_end; ++i) {
        eshkol_syn_env one;
        memset(&one, 0, sizeof(one));
        if (!eshkol_syn_match_node(t, repeated, input->items[i], &one)) {
            ok = 0;
        } else {
            for (int v = 0; v < vars.n; ++v) {
                eshkol_syn_match* got = eshkol_syn_env_find(&one, vars.items[v].name);
                eshkol_syn_match* seq = eshkol_syn_env_find(&sequences, vars.items[v].name);
                eshkol_syn_match rep;
                if (got) { rep = *got; memset(got, 0, sizeof(*got)); }   /* move */
                else { memset(&rep, 0, sizeof(rep)); rep.depth = vars.items[v].match.depth; }
                eshkol_syn_match_push_rep(seq, rep);
            }
        }
        eshkol_syn_env_free(&one);
    }
    eshkol_syn_env_free(&vars);
    if (ok) {
        for (int v = 0; v < sequences.n; ++v) {
            eshkol_syn_env_bind(env, sequences.items[v].name, sequences.items[v].match);
            memset(&sequences.items[v].match, 0, sizeof(eshkol_syn_match));
        }
    }
    eshkol_syn_env_free(&sequences);
    if (!ok) return 0;
    for (int i = 0; i < after; ++i)
        if (!eshkol_syn_match_node(t, pattern->items[ellipsis_at + 2 + i],
                                   input->items[repeat_end + i], env))
            return 0;
    if (p_tail) {
        eshkol_syn* rest = eshkol_syn_rest(input, i_count);
        int tail_ok = rest && eshkol_syn_match_node(t, p_tail, rest, env);
        eshkol_syn_free(rest);
        return tail_ok;
    }
    return 1;
}

/* ── Instantiation ─────────────────────────────────────────────────────── */

/* quote mode: 0 = code, 1 = quoted data, N >= 2 = quasiquote nesting N - 1. */
static inline int eshkol_syn_enter_mode(const char* marker, int mode) {
    if (mode == 1) return 1;
    if (strcmp(marker, "quote") == 0) return mode == 0 ? 1 : mode;
    if (strcmp(marker, "quasiquote") == 0) return mode == 0 ? 2 : mode + 1;
    if (strcmp(marker, "unquote") == 0 || strcmp(marker, "unquote-splicing") == 0)
        return mode >= 2 ? (mode == 2 ? 0 : mode - 1) : mode;
    return mode;
}

static inline int eshkol_syn_base_is_marker(const char* text, char* marker, size_t cap) {
    size_t len = eshkol_syntax_base_length(text);
    if (len + 1 > cap) return 0;
    memcpy(marker, text, len);
    marker[len] = '\0';
    return 1;
}

/* Collect template variables matched under at least one ellipsis. */
static inline void eshkol_syn_template_vars(const eshkol_syn* tmpl, const eshkol_syn_env* env,
                                     const char*** names, int* n, int* cap) {
    if (tmpl->kind == ESHKOL_SYN_SYMBOL) {
        eshkol_syn_match* m = eshkol_syn_env_find(env, tmpl->text);
        if (!m || m->depth == 0) return;
        for (int i = 0; i < *n; ++i) if (strcmp((*names)[i], tmpl->text) == 0) return;
        if (*n == *cap) {
            int grown_cap = *cap ? *cap * 2 : 4;
            const char** grown = (const char**)realloc((void*)*names, (size_t)grown_cap * sizeof(char*));
            if (!grown) return;
            *names = grown;
            *cap = grown_cap;
        }
        (*names)[(*n)++] = tmpl->text;
        return;
    }
    for (int i = 0; i < tmpl->n_items; ++i) eshkol_syn_template_vars(tmpl->items[i], env, names, n, cap);
}

static inline eshkol_syn* eshkol_syn_instantiate(eshkol_syn_transformer* t, const eshkol_syn* tmpl,
                                          const eshkol_syn_env* env, int mode, int escaped);

/* Instantiate @p item followed by @p ellipses ellipses, appending to @p out. */
static inline int eshkol_syn_repeat(eshkol_syn_transformer* t, const eshkol_syn* item,
                             const eshkol_syn_env* env, int ellipses, int mode, int escaped,
                             eshkol_syn* out) {
    if (ellipses == 0) {
        eshkol_syn* one = eshkol_syn_instantiate(t, item, env, mode, escaped);
        if (!one) return 0;
        return eshkol_syn_push(out, one);
    }
    const char** vars = NULL;
    int n_vars = 0, cap_vars = 0;
    eshkol_syn_template_vars(item, env, &vars, &n_vars, &cap_vars);
    if (n_vars == 0) {
        free((void*)vars);
        eshkol_syn_fail(t, "a template element followed by an ellipsis contains no pattern "
                           "variable matched under an ellipsis", NULL);
        return 0;
    }
    int count = eshkol_syn_env_find(env, vars[0])->n_reps;
    for (int v = 1; v < n_vars; ++v) {
        if (eshkol_syn_env_find(env, vars[v])->n_reps != count) {
            free((void*)vars);
            eshkol_syn_fail(t, "pattern variables under one ellipsis matched different lengths",
                            NULL);
            return 0;
        }
    }
    int ok = 1;
    for (int i = 0; ok && i < count; ++i) {
        /* The step environment: every iterated variable bound to its i-th
         * repetition, everything else as before. */
        eshkol_syn_env step;
        memset(&step, 0, sizeof(step));
        step.parent = env;
        step.borrowed = 1;
        for (int v = 0; v < n_vars; ++v) {
            const eshkol_syn_match* seq = eshkol_syn_env_find(env, vars[v]);
            eshkol_syn_env_bind(&step, vars[v], seq->reps[i]);
        }
        ok = eshkol_syn_repeat(t, item, &step, ellipses - 1, mode, escaped, out);
        eshkol_syn_env_free(&step);
    }
    free((void*)vars);
    return ok;
}

static inline eshkol_syn* eshkol_syn_instantiate(eshkol_syn_transformer* t, const eshkol_syn* tmpl,
                                          const eshkol_syn_env* env, int mode, int escaped) {
    switch (tmpl->kind) {
        case ESHKOL_SYN_SYMBOL: {
            eshkol_syn_match* bound = eshkol_syn_env_find(env, tmpl->text);
            if (bound) {
                if (bound->depth != 0) {
                    eshkol_syn_fail(t, "pattern variable '%s' is used with fewer ellipses "
                                       "than it was matched with", tmpl->text);
                    return NULL;
                }
                return eshkol_syn_copy(bound->value);
            }
            if (mode != 0 || eshkol_syntax_is_auxiliary(tmpl->text) ||
                eshkol_syn_is_ellipsis(t, tmpl))
                return eshkol_syn_copy(tmpl);                 /* data or pattern marker */
            const char* alias = t->keyword ? t->keyword(t->keyword_ctx, tmpl->text) : NULL;
            if (alias && *alias) return eshkol_syn_new(ESHKOL_SYN_SYMBOL, alias, tmpl->origin);
            size_t cap = strlen(tmpl->text) + 16;
            char* colored = (char*)malloc(cap);
            if (!colored) return NULL;
            eshkol_syntax_color_name(colored, cap, tmpl->text, t->color);
            eshkol_syn* node = eshkol_syn_new(ESHKOL_SYN_SYMBOL, colored, tmpl->origin);
            free(colored);
            return node;
        }
        case ESHKOL_SYN_ATOM:
            return eshkol_syn_copy(tmpl);
        case ESHKOL_SYN_PREFIX: {
            eshkol_syn* out = eshkol_syn_new(ESHKOL_SYN_PREFIX, tmpl->text, tmpl->origin);
            const char* name = eshkol_syn_prefix_name(tmpl->text);
            int inner = name ? eshkol_syn_enter_mode(name, mode) : mode;
            eshkol_syn* child = tmpl->n_items > 0
                ? eshkol_syn_instantiate(t, tmpl->items[0], env, inner, escaped) : NULL;
            if (!out || !child) { eshkol_syn_free(out); eshkol_syn_free(child); return NULL; }
            eshkol_syn_push(out, child);
            return out;
        }
        case ESHKOL_SYN_LIST:
        case ESHKOL_SYN_VECTOR: {
            /* (... template): the ellipsis is literal inside `template`. */
            if (!escaped && tmpl->kind == ESHKOL_SYN_LIST && !tmpl->dotted && tmpl->n_items == 2 &&
                eshkol_syn_is_ellipsis(t, tmpl->items[0]))
                return eshkol_syn_instantiate(t, tmpl->items[1], env, mode, 1);
            eshkol_syn* out = eshkol_syn_new(tmpl->kind, tmpl->text, tmpl->origin);
            if (!out) return NULL;
            int rest_mode = mode;
            if (tmpl->kind == ESHKOL_SYN_LIST && tmpl->n_items > 0 &&
                tmpl->items[0]->kind == ESHKOL_SYN_SYMBOL &&
                !eshkol_syn_env_find(env, tmpl->items[0]->text)) {
                char head[32];
                if (eshkol_syn_base_is_marker(tmpl->items[0]->text, head, sizeof(head)))
                    rest_mode = eshkol_syn_enter_mode(head, mode);
            }
            const int count = eshkol_syn_proper(tmpl);
            for (int i = 0; i < count;) {
                const int item_mode = i == 0 ? mode : rest_mode;
                int ellipses = 0;
                while (!escaped && i + 1 + ellipses < count &&
                       eshkol_syn_is_ellipsis(t, tmpl->items[i + 1 + ellipses]))
                    ++ellipses;
                if (!eshkol_syn_repeat(t, tmpl->items[i], env, ellipses, item_mode, escaped, out)) {
                    eshkol_syn_free(out);
                    return NULL;
                }
                i += 1 + ellipses;
            }
            if (tmpl->dotted) {
                eshkol_syn* tail = eshkol_syn_instantiate(t, tmpl->items[tmpl->n_items - 1], env,
                                                          rest_mode, escaped);
                if (!tail) { eshkol_syn_free(out); return NULL; }
                if (tail->kind == ESHKOL_SYN_LIST) {
                    for (int i = 0; i < tail->n_items; ++i) {
                        eshkol_syn_push(out, tail->items[i]);
                        tail->items[i] = NULL;
                    }
                    out->dotted = tail->dotted;
                    tail->n_items = 0;
                    eshkol_syn_free(tail);
                } else {
                    eshkol_syn_push(out, tail);
                    out->dotted = 1;
                }
            }
            return out;
        }
    }
    return NULL;
}

typedef enum {
    ESHKOL_SYN_EXPANDED = 0,
    ESHKOL_SYN_NO_MATCH = 1,
    ESHKOL_SYN_ERROR    = 2
} eshkol_syn_outcome;

/**
 * @brief Apply a transformer to a macro use.
 *
 * @param patterns,templates the rules, in order (each pattern is the whole
 *        `(keyword pattern ...)` list; the keyword position is ignored).
 * @param use the whole use `(keyword operand ...)`.
 * @param color the expansion's color (syntax_color.h).
 * @param keyword optional: maps a template identifier to the spelling that
 *        denotes the same macro keyword at the use site, or NULL.
 * @param expansion receives the instantiated template (caller frees).
 */
static inline eshkol_syn_outcome eshkol_syntax_rules_apply(
    const char* ellipsis, const char* const* literals, int n_literals,
    eshkol_syn* const* patterns, eshkol_syn* const* templates, int n_rules,
    const eshkol_syn* use, unsigned color,
    eshkol_syn_keyword_fn keyword, void* keyword_ctx,
    eshkol_syn** expansion, char* error, size_t error_cap) {
    eshkol_syn_transformer t;
    t.ellipsis = ellipsis ? ellipsis : "...";
    t.literals = literals;
    t.n_literals = n_literals;
    t.color = color;
    t.keyword = keyword;
    t.keyword_ctx = keyword_ctx;
    t.error = error;
    t.error_cap = error_cap;
    t.failed = 0;
    *expansion = NULL;
    if (!use || use->kind != ESHKOL_SYN_LIST || use->n_items == 0) return ESHKOL_SYN_NO_MATCH;
    /* Operands: the use without its keyword. */
    eshkol_syn* operands = eshkol_syn_rest(use, 1);
    if (!operands) return ESHKOL_SYN_ERROR;
    if (operands->kind != ESHKOL_SYN_LIST) {       /* (keyword . x) */
        eshkol_syn* wrapped = eshkol_syn_new(ESHKOL_SYN_LIST, "(", use->origin);
        eshkol_syn_push(wrapped, operands);
        wrapped->dotted = 1;
        operands = wrapped;
    }
    for (int r = 0; r < n_rules; ++r) {
        const eshkol_syn* pattern = patterns[r];
        if (!pattern || pattern->kind != ESHKOL_SYN_LIST || pattern->n_items == 0) continue;
        eshkol_syn* formals = eshkol_syn_rest(pattern, 1);
        if (!formals) continue;
        if (formals->kind != ESHKOL_SYN_LIST) {
            eshkol_syn* wrapped = eshkol_syn_new(ESHKOL_SYN_LIST, "(", pattern->origin);
            eshkol_syn_push(wrapped, formals);
            wrapped->dotted = 1;
            formals = wrapped;
        }
        eshkol_syn_env env;
        memset(&env, 0, sizeof(env));
        int matched = eshkol_syn_match_sequence(&t, formals, operands, &env);
        eshkol_syn_free(formals);
        if (!matched) { eshkol_syn_env_free(&env); continue; }
        *expansion = eshkol_syn_instantiate(&t, templates[r], &env, 0, 0);
        eshkol_syn_env_free(&env);
        eshkol_syn_free(operands);
        if (!*expansion) {
            if (!t.failed) eshkol_syn_fail(&t, "out of memory instantiating a template", NULL);
            return ESHKOL_SYN_ERROR;
        }
        return ESHKOL_SYN_EXPANDED;
    }
    eshkol_syn_free(operands);
    return ESHKOL_SYN_NO_MATCH;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ESHKOL_FRONTEND_SYNTAX_RULES_CORE_H */
