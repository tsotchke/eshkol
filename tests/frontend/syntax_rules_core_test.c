/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * The shared syntax-rules engine (ADR-0026) compiled as plain C, the way the
 * bytecode VM compiles it, and checked against hand-computed expansions.
 */
#include "eshkol/frontend/syntax_rules_core.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

/* ── A tiny reader for test syntax: lists, dotted tails, #( vectors, symbols,
 *    and atoms written as digits. ─────────────────────────────────────────── */
static const char* cursor;

static eshkol_syn* read_datum(void);

static void skip(void) { while (*cursor == ' ') ++cursor; }

static eshkol_syn* read_datum(void) {
    skip();
    if (*cursor == '(' || (cursor[0] == '#' && cursor[1] == '(')) {
        int vector = *cursor == '#';
        cursor += vector ? 2 : 1;
        eshkol_syn* list = eshkol_syn_new(vector ? ESHKOL_SYN_VECTOR : ESHKOL_SYN_LIST,
                                          vector ? "#(" : "(", NULL);
        for (;;) {
            skip();
            if (*cursor == ')') { ++cursor; return list; }
            if (cursor[0] == '.' && cursor[1] == ' ') {
                cursor += 2;
                eshkol_syn_push(list, read_datum());
                list->dotted = 1;
                skip();
                ++cursor;   /* ) */
                return list;
            }
            eshkol_syn_push(list, read_datum());
        }
    }
    char word[64];
    int n = 0;
    while (*cursor && *cursor != ' ' && *cursor != '(' && *cursor != ')' && n < 63)
        word[n++] = *cursor++;
    word[n] = '\0';
    int numeric = n > 0 && word[0] >= '0' && word[0] <= '9';
    return eshkol_syn_new(numeric ? ESHKOL_SYN_ATOM : ESHKOL_SYN_SYMBOL, word, NULL);
}

static eshkol_syn* parse(const char* text) {
    cursor = text;
    return read_datum();
}

/* Print with colors shown as <name>#<color>. */
static void print(const eshkol_syn* s, char* out, size_t cap) {
    size_t len = strlen(out);
    if (s->kind == ESHKOL_SYN_SYMBOL || s->kind == ESHKOL_SYN_ATOM) {
        for (const char* p = s->text; *p && len + 2 < cap; ++p)
            out[len++] = *p == ESHKOL_SYNTAX_COLOR_MARK ? '#' : *p;
        out[len] = '\0';
        return;
    }
    snprintf(out + len, cap - len, "%s", s->kind == ESHKOL_SYN_VECTOR ? "#(" : "(");
    for (int i = 0; i < s->n_items; ++i) {
        if (i) strncat(out, " ", cap - strlen(out) - 1);
        if (s->dotted && i == s->n_items - 1) strncat(out, ". ", cap - strlen(out) - 1);
        print(s->items[i], out, cap);
    }
    strncat(out, ")", cap - strlen(out) - 1);
}

static void check(const char* label, const char* const* literals, int n_literals,
                  const char* ellipsis, const char* const* rules, int n_rules,
                  const char* use, const char* expected) {
    eshkol_syn* patterns[8];
    eshkol_syn* templates[8];
    for (int i = 0; i < n_rules; ++i) {
        patterns[i] = parse(rules[2 * i]);
        templates[i] = parse(rules[2 * i + 1]);
    }
    eshkol_syn* use_syn = parse(use);
    eshkol_syn* result = NULL;
    char error[256] = {0};
    eshkol_syn_outcome outcome = eshkol_syntax_rules_apply(
        ellipsis, literals, n_literals, patterns, templates, n_rules, use_syn, 7,
        NULL, NULL, &result, error, sizeof(error));
    char got[512] = {0};
    if (outcome == ESHKOL_SYN_EXPANDED) print(result, got, sizeof(got));
    else if (outcome == ESHKOL_SYN_NO_MATCH) snprintf(got, sizeof(got), "NO-MATCH");
    else snprintf(got, sizeof(got), "ERROR: %s", error);
    if (strcmp(got, expected) != 0) {
        printf("FAIL %s: expected %s got %s\n", label, expected, got);
        ++failures;
    } else {
        printf("PASS %s\n", label);
    }
    eshkol_syn_free(result);
    eshkol_syn_free(use_syn);
    for (int i = 0; i < n_rules; ++i) {
        eshkol_syn_free(patterns[i]);
        eshkol_syn_free(templates[i]);
    }
}

int main(void) {
    const char* my_or[] = {
        "(_)", "#f",
        "(_ e)", "e",
        "(_ e r ...)", "(let ((t e)) (if t t (my-or r ...)))",
    };
    check("binders and free identifiers are colored, operands are not",
          NULL, 0, "...", my_or, 3, "(my-or a t)",
          "(let#7 ((t#7 a)) (if#7 t#7 t#7 (my-or#7 t)))");

    const char* rev[] = {
        "(_ () acc)", "(quote acc)",
        "(_ (x . xs) acc)", "(rev xs (x . acc))",
    };
    check("dotted pattern and template tail", NULL, 0, "...", rev, 2,
          "(rev (1 2) ())", "(rev#7 (2) (1))");
    check("quoted data is never colored", NULL, 0, "...", rev, 2,
          "(rev () (3 2 1))", "(quote#7 (3 2 1))");

    const char* flat[] = { "(_ (a ...) ...)", "(list a ... ...)" };
    check("nested ellipsis flattens", NULL, 0, "...", flat, 1,
          "(flat (1 2) (3))", "(list#7 1 2 3)");

    const char* last[] = { "(_ x ... y)", "y" };
    check("ellipsis followed by a pattern", NULL, 0, "...", last, 1,
          "(last a b c)", "c");

    const char* pairs[] = { "(_ (k v) ...)", "(list (cons k v) ...)" };
    check("structured repetition", NULL, 0, "...", pairs, 1,
          "(m (a 1) (b 2))", "(list#7 (cons#7 a 1) (cons#7 b 2))");

    const char* lits[] = { "=>" };
    const char* arrow[] = { "(_ a => b)", "(b a)", "(_ a b)", "(none)" };
    check("literal matches by spelling", lits, 1, "...", arrow, 2,
          "(m 1 => f)", "(f 1)");
    check("literal is not a pattern variable", lits, 1, "...", arrow, 2,
          "(m 1 2)", "(none#7)");

    const char* vec[] = { "(_ #(a b))", "b" };
    check("vector pattern", NULL, 0, "...", vec, 1, "(m #(1 2))", "2");

    const char* custom[] = { "(_ x etc)", "(list x etc)" };
    check("custom ellipsis", NULL, 0, "etc", custom, 1, "(m 1 2 3)", "(list#7 1 2 3)");

    const char* escape[] = { "(_ x)", "(quote (x (... ...)))" };
    check("escaped ellipsis", NULL, 0, "...", escape, 1, "(m 5)", "(quote#7 (5 ...))");

    const char* wild[] = { "(_ _ x)", "x" };
    check("wildcard binds nothing", NULL, 0, "...", wild, 1, "(m 1 2)", "2");

    const char* one[] = { "(_ x)", "x" };
    check("no rule matches", NULL, 0, "...", one, 1, "(m)", "NO-MATCH");

    const char* bad[] = { "(_ x ...)", "x" };
    check("too few ellipses is an error", NULL, 0, "...", bad, 1, "(m 1 2)",
          "ERROR: pattern variable 'x' is used with fewer ellipses than it was matched with");

    printf("syntax-rules core: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
