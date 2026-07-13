#define _GNU_SOURCE
#include "eval.h"
#include "ad.h"
#include "reader.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

Value *g_env;

/* forward decls */
Value *v_str_aprintf(const char *fmt, ...);
Value *apply_value(Value *fn, Value *eargs);

/* ---- list helpers ---- */
static int list_len(Value *v) {
    int n = 0;
    while (v && v->type == V_PAIR) { n++; v = v->as.pair.cdr; }
    return n;
}
static Value *list_ref(Value *v, int i) {
    while (i-- > 0 && v->type == V_PAIR) v = v->as.pair.cdr;
    return (v->type == V_PAIR) ? v->as.pair.car : NULL;
}

Value *eval_list(Value *args, Value *env) {
    /* returns a NEW list of evaluated values (+1 each), or error string */
    Value *head = NULL, *tail = NULL;
    while (args && args->type == V_PAIR) {
        Value *ev = slermes_eval(args->as.pair.car, env);
        if (ev && ev->type == V_STR) { v_decref(head); return ev; }
        Value *cell = v_pair(ev, v_nil());
        if (!head) { head = cell; tail = cell; }
        else { tail->as.pair.cdr = cell; tail = cell; }
        args = args->as.pair.cdr;
    }
    return head;
}

static int is_symbol(Value *v, const char *s) {
    return v && v->type == V_SYM && strcmp(v->as.str, s) == 0;
}

/* ---- builtins ---- */
static Value *bi_add(Value *args, int argc) {
    double s = 0.0; Value *acc = NULL;
    /* if any tower, do tower add */
    int any_tower = 0;
    Value *a = args;
    while (a && a->type == V_PAIR) { if (ad_is_tower(a->as.pair.car)) any_tower = 1; a = a->as.pair.cdr; }
    if (any_tower) {
        Value *r = v_num(0.0);
        a = args;
        while (a && a->type == V_PAIR) {
            Value *nx = ad_add(r, a->as.pair.car);
            v_decref(r); r = nx; a = a->as.pair.cdr;
        }
        return r;
    }
    a = args;
    while (a && a->type == V_PAIR) { s += a->as.pair.car->as.num; a = a->as.pair.cdr; }
    return v_num(s);
}
static Value *bi_sub(Value *args, int argc) {
    if (argc == 1) return ad_neg(args->as.pair.car);
    int any_tower = 0; Value *a = args;
    while (a && a->type == V_PAIR) { if (ad_is_tower(a->as.pair.car)) any_tower = 1; a = a->as.pair.cdr; }
    if (any_tower) {
        Value *r = args->as.pair.car; v_incref(r);
        a = args->as.pair.cdr;
        while (a && a->type == V_PAIR) {
            Value *nx = ad_sub(r, a->as.pair.car); v_decref(r); r = nx; a = a->as.pair.cdr;
        }
        return r;
    }
    double s = args->as.pair.car->as.num;
    Value *r = args->as.pair.cdr;
    while (r && r->type == V_PAIR) { s -= r->as.pair.car->as.num; r = r->as.pair.cdr; }
    return v_num(s);
}
static Value *bi_mul(Value *args, int argc) {
    int any_tower = 0; Value *a = args;
    while (a && a->type == V_PAIR) { if (ad_is_tower(a->as.pair.car)) any_tower = 1; a = a->as.pair.cdr; }
    if (any_tower) {
        Value *r = v_num(1.0);
        a = args;
        while (a && a->type == V_PAIR) {
            Value *nx = ad_mul(r, a->as.pair.car); v_decref(r); r = nx; a = a->as.pair.cdr;
        }
        return r;
    }
    double p = 1.0; a = args;
    while (a && a->type == V_PAIR) { p *= a->as.pair.car->as.num; a = a->as.pair.cdr; }
    return v_num(p);
}
static Value *bi_div(Value *args, int argc) {
    int any_tower = 0; Value *a = args;
    while (a && a->type == V_PAIR) { if (ad_is_tower(a->as.pair.car)) any_tower = 1; a = a->as.pair.cdr; }
    if (any_tower) {
        Value *r = args->as.pair.car; v_incref(r);
        a = args->as.pair.cdr;
        while (a && a->type == V_PAIR) {
            Value *nx = ad_div(r, a->as.pair.car); v_decref(r); r = nx; a = a->as.pair.cdr;
        }
        return r;
    }
    if (argc == 1) return ad_div(v_num(1.0), args->as.pair.car);
    double s = args->as.pair.car->as.num;
    Value *r = args->as.pair.cdr;
    while (r && r->type == V_PAIR) { s /= r->as.pair.car->as.num; r = r->as.pair.cdr; }
    return v_num(s);
}
static Value *bi_eq(Value *args, int argc) {
    double x = args->as.pair.car->as.num;
    Value *r = args->as.pair.cdr;
    while (r && r->type == V_PAIR) {
        if (r->as.pair.car->as.num != x) return v_bool(0);
        r = r->as.pair.cdr;
    }
    return v_bool(1);
}
static Value *bi_lt(Value *args, int argc) {
    double prev = args->as.pair.car->as.num;
    Value *r = args->as.pair.cdr;
    while (r && r->type == V_PAIR) {
        double cur = r->as.pair.car->as.num;
        if (!(prev < cur)) return v_bool(0);
        prev = cur; r = r->as.pair.cdr;
    }
    return v_bool(1);
}
static Value *bi_gt(Value *args, int argc) {
    double prev = args->as.pair.car->as.num;
    Value *r = args->as.pair.cdr;
    while (r && r->type == V_PAIR) {
        double cur = r->as.pair.car->as.num;
        if (!(prev > cur)) return v_bool(0);
        prev = cur; r = r->as.pair.cdr;
    }
    return v_bool(1);
}
static Value *bi_cons(Value *args, int argc) { return v_pair(args->as.pair.car, list_ref(args,1)); }
static Value *bi_car(Value *args, int argc) {
    Value *p = args->as.pair.car;
    if (!p || p->type == V_NIL) return v_nil();
    return p->as.pair.car;
}
static Value *bi_cdr(Value *args, int argc) {
    Value *p = args->as.pair.car;
    if (!p || p->type == V_NIL) return v_nil();
    return p->as.pair.cdr;
}
static Value *bi_list(Value *args, int argc) {
    /* args already evaluated list -> return it as-is */
    v_incref(args); return args;
}
static Value *bi_display(Value *args, int argc) {
    v_print(args->as.pair.car); fflush(stdout);
    return v_nil();
}
static Value *bi_newline(Value *args, int argc) {
    printf("\n"); fflush(stdout);
    return v_nil();
}
static Value *bi_sqrt(Value *args, int argc) {
    Value *a = args->as.pair.car;
    return ad_sqrt(a);
}
static Value *bi_sin(Value *args, int argc) { return ad_sin(args->as.pair.car); }
static Value *bi_cos(Value *args, int argc) { return ad_cos(args->as.pair.car); }
static Value *bi_exp(Value *args, int argc) { return ad_exp(args->as.pair.car); }
static Value *bi_log(Value *args, int argc) { return ad_log(args->as.pair.car); }
static Value *bi_expt(Value *args, int argc) {
    Value *b = args->as.pair.car;
    Value *e = list_ref(args, 1);
    if (e->type == V_NUM && b->type != V_TOWER) return v_num(pow(b->as.num, e->as.num));
    /* tower ^ constant */
    return ad_pow_const(b, e->as.num);
}
static Value *bi_abs(Value *args, int argc) {
    double x = args->as.pair.car->as.num; return v_num(fabs(x));
}
static Value *bi_numberp(Value *args, int argc) {
    Value *a = args->as.pair.car; return v_bool(a && a->type == V_NUM);
}
static Value *bi_symbolp(Value *args, int argc) {
    Value *a = args->as.pair.car; return v_bool(a && a->type == V_SYM);
}
static Value *bi_listp(Value *args, int argc) {
    Value *a = args->as.pair.car; return v_bool(a && (a->type == V_PAIR || a->type == V_NIL));
}
static Value *bi_nullp(Value *args, int argc) {
    Value *a = args->as.pair.car; return v_bool(a && a->type == V_NIL);
}
/* cadr-family */
static Value *bi_cadr(Value *args, int argc) {
    Value *p = args->as.pair.car;
    return (p->type==V_PAIR && p->as.pair.cdr->type==V_PAIR) ? p->as.pair.cdr->as.pair.car : v_nil();
}
static Value *bi_caddr(Value *args, int argc) {
    Value *p = args->as.pair.car;
    for (int i=0;i<2 && p && p->type==V_PAIR;i++) p=p->as.pair.cdr;
    return (p && p->type==V_PAIR) ? p->as.pair.car : v_nil();
}
static Value *bi_cddr(Value *args, int argc) {
    Value *p = args->as.pair.car;
    for (int i=0;i<2 && p && p->type==V_PAIR;i++) p=p->as.pair.cdr;
    return p ? p : v_nil();
}
static Value *bi_cadddr(Value *args, int argc) {
    Value *p = args->as.pair.car;
    for (int i=0;i<3 && p && p->type==V_PAIR;i++) p=p->as.pair.cdr;
    return (p && p->type==V_PAIR) ? p->as.pair.car : v_nil();
}
static Value *bi_cdddr(Value *args, int argc) {
    Value *p = args->as.pair.car;
    for (int i=0;i<3 && p && p->type==V_PAIR;i++) p=p->as.pair.cdr;
    return p ? p : v_nil();
}
/* (fold-left proc init lst) */
static Value *bi_fold_left(Value *args, int argc) {
    Value *proc = args->as.pair.car;
    Value *acc = list_ref(args, 1);
    Value *lst = list_ref(args, 2);
    if (!proc || (proc->type != V_BUILTIN && proc->type != V_CLOSURE))
        return v_str("fold-left: proc must be a function");
    while (lst && lst->type == V_PAIR) {
        /* Build the call as a raw (already-evaluated) arg list and apply
           it directly via apply_value -- do NOT route through slermes_eval,
           which would re-evaluate the data element (a pair) as a function. */
        Value *call_args = v_pair(acc, v_pair(lst->as.pair.car, v_nil()));
        Value *r = apply_value(proc, call_args);
        v_decref(call_args);
        if (r && r->type == V_STR) return r;
        v_decref(acc);
        acc = r;
        lst = lst->as.pair.cdr;
    }
    return acc;
}

/* ---- AD primitives (eshkol headline features) ----
 *
 * Strategy: SYMBOLIC differentiation.  (derivative f x0) builds the
 * derivative AST of f's body with respect to its parameter, then evaluates
 * that AST at x0.  This composes correctly through nested derivatives
 * (e.g. (derivative (lambda (x) (derivative (lambda (y) y^5) x)) 2.0) -> 160),
 * matching the upstream eshkol behaviour.  A numeric forward-mode tower path
 * is kept as a fallback for bodies that use operators sym_deriv cannot handle.
 */

static Value *S_num(double x)  { return v_num(x); }
static Value *S_sym(const char *s) { return v_sym(s); }
static Value *S_call1(const char *op, Value *a) {
    return v_pair(v_sym(op), v_pair(a, v_nil()));
}
static Value *S_call2(const char *op, Value *a, Value *b) {
    return v_pair(v_sym(op), v_pair(a, v_pair(b, v_nil())));
}

/* Rename every free occurrence of symbol `from` to `to` in AST e. */
static Value *alpha_rename(Value *e, Value *from, Value *to) {
    if (!e) return e;
    if (e->type == V_SYM) {
        if (strcmp(e->as.str, from->as.str) == 0) return v_sym(to->as.str);
        return v_sym(e->as.str);
    }
    if (e->type == V_PAIR) {
        return v_pair(alpha_rename(e->as.pair.car, from, to),
                      alpha_rename(e->as.pair.cdr, from, to));
    }
    return e; /* num / other: unchanged */
}

/* Symbolic derivative of expression e w.r.t. variable var.
 * Returns a new AST, or a V_STR error if an operator is unsupported. */
static Value *sym_deriv(Value *e, Value *var) {
    if (!e) return v_nil();
    switch (e->type) {
    case V_NUM:  return S_num(0.0);
    case V_SYM:  return S_num(strcmp(e->as.str, var->as.str) == 0 ? 1.0 : 0.0);
    case V_PAIR: break;
    default:     return v_str("sym_deriv: unsupported expression");
    }
    /* A lambda body is stored wrapped as (( expr )); unwrap to the bare expr. */
    if (e->as.pair.car->type == V_PAIR && e->as.pair.cdr->type == V_NIL)
        return sym_deriv(e->as.pair.car, var);
    Value *head = e->as.pair.car;
    if (head->type != V_SYM)
        return v_str("sym_deriv: operator must be a symbol");
    const char *op = head->as.str;
    Value *a = list_ref(e, 1);
    Value *b = list_ref(e, 2);

    /* Nested derivative / derivative-n: differentiate the inner body, rename
     * its parameter to the inner variable, then differentiate w.r.t var. */
    if (strcmp(op, "derivative") == 0 || strcmp(op, "derivative-n") == 0) {
        /* a is either a V_CLOSURE or a raw (lambda (params) body...) S-expr
         * (the latter when the lambda appears unevaluated inside a body). */
        Value *inner_param = NULL, *inner_body = NULL;
        if (a && a->type == V_CLOSURE) {
            inner_param = a->as.closure.params->as.pair.car;
            inner_body  = a->as.closure.body;
        } else if (a && a->type == V_PAIR && a->as.pair.car->type == V_SYM &&
                   strcmp(a->as.pair.car->as.str, "lambda") == 0) {
            inner_param = list_ref(a, 1)->as.pair.car; /* first param symbol */
            inner_body  = list_ref(a, 2);              /* body forms */
        } else {
            static char ebuf[64];
            snprintf(ebuf, sizeof ebuf, "sym_deriv: bad derivative arg (type=%d)", a ? (int)a->type : -1);
            return v_str(ebuf);
        }
        int k = (strcmp(op, "derivative-n") == 0 && b && b->type == V_NUM)
                    ? (int)b->as.num : 1;
        Value *iv = (strcmp(op, "derivative-n") == 0) ? list_ref(e, 2) : b;
        Value *inner = inner_body;
        Value *d = sym_deriv(inner, inner_param);
        if (d->type == V_STR) return d;
        d = alpha_rename(d, inner_param, iv);   /* express in terms of iv */
        return sym_deriv(d, var);                 /* d/d(var) of that */
    }

    if (strcmp(op, "+") == 0) {
        /* (+ (d a0) (d a1) ...) */
        Value *sum = v_nil();
        Value *args = e->as.pair.cdr;
        while (args && args->type == V_PAIR) {
            Value *term = sym_deriv(args->as.pair.car, var);
            if (term->type == V_STR) return term;
            sum = (sum->type == V_NIL) ? S_call1("+", term)
                                       : S_call2("+", sum, term);
            args = args->as.pair.cdr;
        }
        return sum;
    }
    if (strcmp(op, "-") == 0) {
        if (!b) return S_call1("-", sym_deriv(a, var));
        return S_call2("-", sym_deriv(a, var), sym_deriv(b, var));
    }
    if (strcmp(op, "*") == 0) {
        /* product rule over all factors: sum_i ( d(ai) * PROD_{j!=i} aj ) */
        Value *sum = v_nil();
        Value *factors = e->as.pair.cdr;
        int nf = 0;
        while (factors && factors->type == V_PAIR) { factors = factors->as.pair.cdr; nf++; }
        factors = e->as.pair.cdr;
        for (int i = 0; i < nf; i++) {
            Value *prod_expr = v_nil();
            Value *rest = e->as.pair.cdr;
            for (int j = 0; j < nf; j++) {
                Value *fj = rest->as.pair.car;
                if (j != i) prod_expr = v_pair(fj, prod_expr); /* prepend: flat list */
                rest = rest->as.pair.cdr;
            }
            /* prod_expr is a proper (reversed) list of the other factors */
            if (prod_expr->type == V_NIL) prod_expr = S_num(1.0);
            else if (prod_expr->as.pair.cdr->type == V_NIL) prod_expr = prod_expr->as.pair.car;
            else {
                Value *p = prod_expr;
                Value *acc = p->as.pair.car; p = p->as.pair.cdr;
                while (p && p->type == V_PAIR) { acc = S_call2("*", acc, p->as.pair.car); p = p->as.pair.cdr; }
                prod_expr = acc;
            }
            Value *term = sym_deriv(factors->as.pair.car, var);
            if (term->type == V_STR) return term;
            Value *factor = S_call2("*", term, prod_expr);
            sum = (sum->type == V_NIL) ? S_call1("+", factor)
                                       : S_call2("+", sum, factor);
            factors = factors->as.pair.cdr;
        }
        return sum;
    }
    if (strcmp(op, "/") == 0) {
        /* (f/g)' = (f' g - f g') / g^2 */
        Value *fp = sym_deriv(a, var), *gp = sym_deriv(b, var);
        if (fp->type == V_STR) return fp;
        if (gp->type == V_STR) return gp;
        Value *num = S_call2("-", S_call2("*", fp, b), S_call2("*", a, gp));
        Value *den = S_call2("*", b, b);
        return S_call2("/", num, den);
    }
    if (strcmp(op, "sin") == 0) return S_call2("*", S_call1("cos", a), sym_deriv(a, var));
    if (strcmp(op, "cos") == 0) return S_call2("*", sym_deriv(a, var), S_call1("-", S_call1("sin", a)));
    if (strcmp(op, "exp") == 0) return S_call2("*", S_call1("exp", a), sym_deriv(a, var));
    if (strcmp(op, "log") == 0) return S_call2("/", sym_deriv(a, var), a);
    if (strcmp(op, "sqrt") == 0)
        return S_call2("/", sym_deriv(a, var), S_call2("*", S_num(2.0), S_call1("sqrt", a)));
    if (strcmp(op, "expt") == 0) {
        /* (a^b)' = b a^{b-1} a' + a^b ln(a) b' */
        Value *ap = sym_deriv(a, var), *bp = sym_deriv(b, var);
        if (ap->type == V_STR) return ap;
        if (bp->type == V_STR) return bp;
        Value *t1 = S_call2("*", b, S_call2("*", S_call2("expt", a, S_call2("-", b, S_num(1.0))), ap));
        Value *t2 = S_call2("*", S_call2("expt", a, b), S_call2("*", S_call1("log", a), bp));
        return S_call2("+", t1, t2);
    }
    if (strcmp(op, "abs") == 0) {
        /* d/dx |x| = sign(x); for AD of smooth points treat as identity deriv */
        return S_call2("*", S_call1("abs", a), sym_deriv(a, var));
    }
    return v_str("sym_deriv: unsupported operator");
}

/* Evaluate f's k-th derivative at x0.
 * order==1: try SYMBOLIC differentiation (supports nested derivatives and
 *           stays small); fall back to numeric towers if sym_deriv fails.
 * order >1: use NUMERIC forward-mode towers directly (bounded, exact, and
 *           avoids the AST explosion that symbolic high-order deriv. would
 *           cause on eval). */
static Value *deriv_at(Value *f, double x0, int order) {
    if (!f || f->type != V_CLOSURE)
        return v_str("derivative: f must be a lambda");
    Value *param = f->as.closure.params->as.pair.car;
    Value *body = f->as.closure.body;
    Value *d = NULL;
    if (order == 1) {
        d = sym_deriv(body, param);
        if (d->type == V_STR) d = NULL; /* fall through to numeric */
    }
    if (!d) {
        /* numeric forward-mode */
        AD_ORDER = order;
        Value *seed = ad_seed(x0, 1);
        Value *bind_env = v_env(f->as.closure.env);
        env_define(bind_env, param, seed);
        Value *res = NULL, *b = f->as.closure.body;
        while (b && b->type == V_PAIR) {
            if (res) v_decref(res);
            res = slermes_eval(b->as.pair.car, bind_env);
            b = b->as.pair.cdr;
        }
        v_decref(bind_env); v_decref(seed);
        if (res && res->type == V_TOWER) {
            double c = res->as.tower.coeff[order];
            double fact = 1.0; for (int i = 2; i <= order; i++) fact *= i;
            Value *out = v_num(c * fact); v_decref(res); return out;
        }
        if (res && res->type == V_NUM)
            return v_num(order == 0 ? res->as.num : 0.0);
        v_decref(res);
        return v_str("derivative: could not differentiate body");
    }
    /* evaluate the symbolic derivative AST at x0 */
    Value *bind_env = v_env(f->as.closure.env);
    env_define(bind_env, param, v_num(x0));
    Value *out = slermes_eval(d, bind_env);
    v_decref(bind_env);
    /* ad_ functions (exp/sin/...) return towers even for plain numbers; the
     * value lives in coeff[0]. */
    if (out && out->type == V_TOWER) {
        Value *v = v_num(out->as.tower.coeff[0]);
        v_decref(out);
        return v;
    }
    return out;
}

static Value *bi_derivative(Value *args, int argc) {
    /* (derivative f x0) -> f'(x0) */
    Value *f = args->as.pair.car;
    Value *xv = list_ref(args, 1);
    double x0 = (xv && xv->type == V_TOWER) ? xv->as.tower.coeff[0] : xv->as.num;
    return deriv_at(f, x0, 1);
}
static Value *bi_derivative_n(Value *args, int argc) {
    /* (derivative-n f x0 k) -> f^{(k)}(x0) */
    Value *f = args->as.pair.car;
    double x0 = list_ref(args, 1)->as.num;
    Value *kval = list_ref(args, 2);
    int k = (kval && kval->type == V_NUM) ? (int)kval->as.num : 1;
    if (k < 1) k = 1;
    return deriv_at(f, x0, k);
}
static Value *bi_taylor(Value *args, int argc) {
    /* (taylor f x0 k) -> list of k+1 Taylor coefficients a_m = f^{(m)}(x0)/m! */
    Value *f = args->as.pair.car;
    double x0 = list_ref(args, 1)->as.num;
    Value *kval = list_ref(args, 2);
    int k = (kval && kval->type == V_NUM) ? (int)kval->as.num : 1;
    if (k < 1) k = 1;
    Value *head = NULL, *tail = NULL;
    for (int m = 0; m <= k; m++) {
        Value *dm = deriv_at(f, x0, m);
        if (dm->type == V_STR) { v_decref(dm); return v_str("taylor: could not differentiate"); }
        double fact = 1.0; for (int i = 2; i <= m; i++) fact *= i;
        Value *coeff = v_num(dm->as.num / fact);
        v_decref(dm);
        Value *cell = v_pair(coeff, v_nil());
        if (!head) { head = cell; tail = cell; }
        else { tail->as.pair.cdr = cell; tail = cell; }
    }
    return head;
}

/* ---- special forms handled in eval, but a few as builtins via macros ---- */
/* We implement special forms in slermes_eval directly. */

/* ------------------------------------------------------------------ */
/* eval                                                                */
/* ------------------------------------------------------------------ */
Value *slermes_eval(Value *expr, Value *env) {
    if (!expr) return v_nil();
    switch (expr->type) {
    case V_NUM: case V_STR: case V_BOOL: case V_NIL:
    case V_BUILTIN: case V_CLOSURE: case V_TOWER:
        v_incref(expr); return expr;
    case V_SYM: {
        Value *v = env_lookup(env, expr);
        if (!v) return v_str_aprintf("unbound symbol: %s", expr->as.str);
        v_incref(v); return v;
    }
    case V_PAIR: {
        Value *op = expr->as.pair.car;
        /* special forms */
        if (op->type == V_SYM) {
            const char *s = op->as.str;
            Value *args = expr->as.pair.cdr;
            if (strcmp(s, "quote") == 0) {
                v_incref(args->as.pair.car); return args->as.pair.car;
            }
            if (strcmp(s, "quasiquote") == 0) {
                v_incref(args->as.pair.car); return args->as.pair.car;
            }
            if (strcmp(s, "if") == 0) {
                Value *test = slermes_eval(args->as.pair.car, env);
                if (test && test->type == V_STR) return test;
                int t = v_is_truthy(test); v_decref(test);
                if (t) return slermes_eval(list_ref(args,1), env);
                /* else branch optional */
                Value *els = list_ref(args, 2);
                if (els) return slermes_eval(els, env);
                return v_nil();
            }
            if (strcmp(s, "define") == 0) {
                Value *name = args->as.pair.car;
                if (name->type == V_PAIR) {
                    /* (define (f x y) body...) => lambda */
                    Value *fname = name->as.pair.car;
                    Value *params = name->as.pair.cdr;
                    Value *body = args->as.pair.cdr;
                    Value *lam = v_closure(params, body, env);
                    env_define(env, fname, lam);
                    v_incref(fname); return fname;
                }
                Value *val = slermes_eval(list_ref(args,1), env);
                if (val && val->type == V_STR) return val;
                env_define(env, name, val);
                v_incref(name); return name;
            }
            if (strcmp(s, "set!") == 0) {
                Value *name = args->as.pair.car;
                Value *val = slermes_eval(list_ref(args,1), env);
                if (val && val->type == V_STR) return val;
                env_set(env, name, val);
                v_incref(name); return name;
            }
            if (strcmp(s, "lambda") == 0) {
                return v_closure(args->as.pair.car, args->as.pair.cdr, env);
            }
            if (strcmp(s, "let") == 0) {
                Value *bindings = args->as.pair.car;
                Value *let_env = v_env(env);
                while (bindings && bindings->type == V_PAIR) {
                    Value *b = bindings->as.pair.car;
                    Value *bname = b->as.pair.car;
                    Value *bval = slermes_eval(b->as.pair.cdr->as.pair.car, env);
                    if (bval && bval->type == V_STR) { v_decref(let_env); return bval; }
                    env_define(let_env, bname, bval);
                    v_decref(bval);
                    bindings = bindings->as.pair.cdr;
                }
                Value *body = args->as.pair.cdr;
                Value *r = NULL;
                while (body && body->type == V_PAIR) {
                    if (r) v_decref(r);
                    r = slermes_eval(body->as.pair.car, let_env);
                    body = body->as.pair.cdr;
                }
                v_decref(let_env);
                return r ? r : v_nil();
            }
            if (strcmp(s, "begin") == 0) {
                Value *body = args; Value *r = NULL;
                while (body && body->type == V_PAIR) {
                    if (r) v_decref(r);
                    r = slermes_eval(body->as.pair.car, env);
                    body = body->as.pair.cdr;
                }
                return r ? r : v_nil();
            }
            if (strcmp(s, "cond") == 0) {
                Value *clauses = args;
                while (clauses && clauses->type == V_PAIR) {
                    Value *cl = clauses->as.pair.car;
                    Value *c_test = cl->as.pair.car;
                    if (is_symbol(c_test, "else")) {
                        Value *b = cl->as.pair.cdr; Value *r = NULL;
                        while (b && b->type == V_PAIR) { if (r) v_decref(r); r = slermes_eval(b->as.pair.car, env); b = b->as.pair.cdr; }
                        return r ? r : v_nil();
                    }
                    Value *tv = slermes_eval(c_test, env);
                    if (tv && tv->type == V_STR) return tv;
                    int t = v_is_truthy(tv); v_decref(tv);
                    if (t) {
                        Value *b = cl->as.pair.cdr; Value *r = NULL;
                        while (b && b->type == V_PAIR) { if (r) v_decref(r); r = slermes_eval(b->as.pair.car, env); b = b->as.pair.cdr; }
                        return r ? r : v_nil();
                    }
                    clauses = clauses->as.pair.cdr;
                }
                return v_nil();
            }
        }
        /* function application */
        Value *fn = slermes_eval(op, env);
        if (!fn) return v_nil();
        if (fn->type == V_STR) return fn;
        Value *eargs = eval_list(expr->as.pair.cdr, env);
        if (eargs && eargs->type == V_STR) { v_decref(fn); return eargs; }
        Value *r = apply_value(fn, eargs);
        v_decref(fn); v_decref(eargs);
        return r;
    }
    default:
        v_incref(expr); return expr;
    }
}

/* Apply a (already-evaluated) function value to an (already-evaluated)
   argument list. Used by the application branch above AND by builtins
   such as fold-left that build calls with raw (non-re-evaluated) data. */
Value *apply_value(Value *fn, Value *eargs) {
    if (!fn) return v_nil();
    if (fn->type == V_BUILTIN) {
        return fn->as.builtin.fn(eargs, list_len(eargs));
    }
    if (fn->type == V_CLOSURE) {
        Value *params = fn->as.closure.params;
        Value *call_env = v_env(fn->as.closure.env);
        Value *p = params, *a = eargs;
        while (p && p->type == V_PAIR && a && a->type == V_PAIR) {
            env_define(call_env, p->as.pair.car, a->as.pair.car);
            p = p->as.pair.cdr; a = a->as.pair.cdr;
        }
        Value *body = fn->as.closure.body;
        Value *r = NULL;
        while (body && body->type == V_PAIR) {
            if (r) v_decref(r);
            r = slermes_eval(body->as.pair.car, call_env);
            body = body->as.pair.cdr;
        }
        v_decref(call_env);
        return r ? r : v_nil();
    }
    char *es = v_to_string(fn);
    fprintf(stderr, "[apply-bug] not a function: %s\n", es ? es : "?");
    free(es);
    return v_str("not a function");
}

/* helper: make an error string with format */
#include <stdarg.h>
Value *v_str_aprintf(const char *fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return v_str(buf);
}

void eval_init(void) {
    g_env = v_env(NULL);
    struct { const char *name; BuiltinFn fn; } tbl[] = {
        {"+", bi_add}, {"-", bi_sub}, {"*", bi_mul}, {"/", bi_div},
        {"=", bi_eq}, {"<", bi_lt}, {">", bi_gt},
        {"cons", bi_cons}, {"car", bi_car}, {"cdr", bi_cdr},
        {"list", bi_list}, {"display", bi_display}, {"newline", bi_newline},
        {"sqrt", bi_sqrt}, {"sin", bi_sin}, {"cos", bi_cos},
        {"exp", bi_exp}, {"log", bi_log}, {"expt", bi_expt}, {"abs", bi_abs},
        {"number?", bi_numberp}, {"symbol?", bi_symbolp},
        {"list?", bi_listp}, {"null?", bi_nullp},
        {"derivative", bi_derivative},
        {"derivative-n", bi_derivative_n},
        {"taylor", bi_taylor},
        {"fold-left", bi_fold_left},
        {"cadr", bi_cadr}, {"caddr", bi_caddr}, {"cddr", bi_cddr},
        {"cadddr", bi_cadddr}, {"cdddr", bi_cdddr},
        {NULL, NULL}
    };
    for (int i = 0; tbl[i].name; i++)
        env_define(g_env, v_sym(tbl[i].name), v_builtin(tbl[i].fn, tbl[i].name));
}
