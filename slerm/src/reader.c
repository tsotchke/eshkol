#define _GNU_SOURCE
#include "reader.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == ';') {                 /* line comment to end of line */
        while (*p && *p != '\n') p++;
        return skip_ws(p);
    }
    return p;
}

/* Parse a real number starting at p (original buffer). On success set *out
   and *end (end points just past the consumed characters) and return 1. */
static int parse_number(const char *p, double *out, const char **end) {
    const char *s = p;
    if (*s == '+' || *s == '-') s++;
    int digs = 0, dots = 0;
    const char *q = s;
    while (isdigit((unsigned char)*q) || *q == '.') {
        if (*q == '.') dots++; else digs++;
        q++;
    }
    if (digs == 0 || dots > 1) return 0;
    /* optional exponent */
    if (*q == 'e' || *q == 'E') {
        const char *r = q + 1;
        if (*r == '+' || *r == '-') r++;
        if (!isdigit((unsigned char)*r)) return 0;
        while (isdigit((unsigned char)*r)) r++;
        q = r;
    }
    /* manual parse (avoids locale/errno issues) */
    double v = 0.0;
    const char *t = p;
    int neg = 0;
    if (*t == '+') t++;
    else if (*t == '-') { neg = 1; t++; }
    int seen_dot = 0; double frac = 1.0;
    while (isdigit((unsigned char)*t) || *t == '.') {
        if (*t == '.') { seen_dot = 1; t++; continue; }
        if (seen_dot) { frac *= 10.0; v += (*t - '0') / frac; }
        else v = v * 10.0 + (*t - '0');
        t++;
    }
    if (neg) v = -v;
    if (*t == 'e' || *t == 'E') {
        t++;
        int eneg = 0;
        if (*t == '+') t++; else if (*t == '-') { eneg = 1; t++; }
        int exp = 0;
        while (isdigit((unsigned char)*t)) { exp = exp * 10 + (*t - '0'); t++; }
        double f = 1.0;
        if (eneg) { for (int i = 0; i < exp; i++) f /= 10.0; }
        else      { for (int i = 0; i < exp; i++) f *= 10.0; }
        v *= f;
    }
    *out = v;
    *end = t;
    return 1;
}

static Value *read_atom(const char *p, const char **rest, char **err) {
    const char *start = p;
    while (*p && !isspace((unsigned char)*p) && *p != '(' && *p != ')' &&
           *p != ';' && *p != '"') p++;
    size_t len = (size_t)(p - start);
    if (len == 0) { if (err) *err = strdup("empty atom"); return NULL; }

    /* try as a number directly on the original buffer */
    double num;
    const char *num_end;
    if (parse_number(start, &num, &num_end) && (size_t)(num_end - start) == len) {
        *rest = p;
        return v_num(num);
    }
    /* boolean literals */
    if (len == 2 && strncmp(start, "#t", 2) == 0) { *rest = p; return v_bool(1); }
    if (len == 2 && strncmp(start, "#f", 2) == 0) { *rest = p; return v_bool(0); }
    /* symbol */
    char *tok = malloc(len + 1);
    memcpy(tok, start, len); tok[len] = 0;
    Value *v = v_sym(tok);
    free(tok);
    *rest = p;
    return v;
}

Value *read_sexpr(const char *src, const char **rest, char **err) {
    const char *p = skip_ws(src);
    if (*p == '\0') { if (err) *err = strdup("unexpected end of input"); return NULL; }

    if (*p == '(') {
        Value *head = NULL, *tail = NULL;
        p++;
        for (;;) {
            p = skip_ws(p);
            if (*p == '\0') { if (err) *err = strdup("unterminated list"); return NULL; }
            if (*p == ')') { p++; break; }
            if (*p == '.') {
                const char *q = p + 1;
                q = skip_ws(q);
                int is_dot = !isalnum((unsigned char)*q) && *q != '_' && *q != '-' &&
                             *q != '+' && *q != '*' && *q != '/' && *q != '?' &&
                             *q != '<' && *q != '>' && *q != '=' && *q != '!' &&
                             *q != '@' && *q != '%' && *q != '&' && *q != '~';
                if (is_dot) {
                    Value *tailv = read_sexpr(q, &q, err);
                    if (!tailv) return NULL;
                    p = skip_ws(q);
                    if (*p != ')') { if (err) *err = strdup("expected ) after dotted tail"); return NULL; }
                    p++;
                    if (!head) head = tailv;
                    else {
                        Value *t = head;
                        while (t->type == V_PAIR && t->as.pair.cdr->type == V_PAIR)
                            t = t->as.pair.cdr;
                        if (t->type == V_PAIR) t->as.pair.cdr = tailv;
                        else head = tailv;
                    }
                    *rest = p;
                    return head;
                }
            }
            Value *elem = read_sexpr(p, &p, err);
            if (!elem) return NULL;
            Value *cell = v_pair(elem, v_nil());
            if (!head) { head = cell; tail = cell; }
            else { tail->as.pair.cdr = cell; tail = cell; }
        }
        *rest = p;
        return head;
    }

    if (*p == '"') {
        p++;
        size_t cap = 16, len = 0;
        char *buf = malloc(cap);
        while (*p && *p != '"') {
            if (*p == '\\') {
                p++;
                char c = *p;
                if (c == 'n') c = '\n'; else if (c == 't') c = '\t';
                else if (c == 'r') c = '\r'; else if (c == '"') c = '"';
                else if (c == '\\') c = '\\';
                buf[len++] = c; p++;
            } else buf[len++] = *p++;
            if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
        }
        if (*p != '"') { free(buf); if (err) *err = strdup("unterminated string"); return NULL; }
        p++;
        buf[len] = 0;
        Value *s = v_str(buf);
        free(buf);
        *rest = p;
        return s;
    }

    if (*p == '\'') {
        Value *q = read_sexpr(p + 1, &p, err);
        if (!q) return NULL;
        Value *cell = v_pair(q, v_nil());
        *rest = p;
        return v_pair(v_sym("quote"), cell);
    }
    if (*p == '`') {
        Value *q = read_sexpr(p + 1, &p, err);
        if (!q) return NULL;
        Value *cell = v_pair(q, v_nil());
        *rest = p;
        return v_pair(v_sym("quasiquote"), cell);
    }

    return read_atom(p, rest, err);
}

Value *read_all(const char *src, char **err) {
    Value *head = NULL, *tail = NULL;
    const char *p = src;
    for (;;) {
        p = skip_ws(p);
        if (*p == '\0') break;
        const char *rest;
        Value *e = read_sexpr(p, &rest, err);
        if (!e) { if (head) v_decref(head); return NULL; }
        p = rest;
        Value *cell = v_pair(e, v_nil());
        if (!head) { head = cell; tail = cell; }
        else { tail->as.pair.cdr = cell; tail = cell; }
    }
    return head;
}
