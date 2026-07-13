#define _GNU_SOURCE
#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* interned symbol table (linear probe hash)                          */
/* ------------------------------------------------------------------ */
typedef struct {
    char **names;
    Value **syms;
    size_t cap;
    size_t used;
} SymTab;

static SymTab g_symtab = {0};

static unsigned long sym_hash(const char *s) {
    unsigned long h = 1469598103934665603UL;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211UL; }
    return h;
}

static Value *symtab_intern(const char *s) {
    if (g_symtab.cap == 0) {
        g_symtab.cap = 1024;
        g_symtab.names = calloc(g_symtab.cap, sizeof(char*));
        g_symtab.syms  = calloc(g_symtab.cap, sizeof(Value*));
    }
    for (;;) {
        unsigned long h = sym_hash(s) & (g_symtab.cap - 1);
        size_t i = h;
        while (g_symtab.names[i]) {
            if (strcmp(g_symtab.names[i], s) == 0) return g_symtab.syms[i];
            i = (i + 1) & (g_symtab.cap - 1);
        }
        /* found empty slot */
        g_symtab.names[i] = strdup(s);
        Value *v = malloc(sizeof(Value));
        v->type = V_SYM; v->refcount = 1; v->as.str = g_symtab.names[i];
        g_symtab.syms[i] = v;
        g_symtab.used++;
        if (g_symtab.used * 2 > g_symtab.cap) {
            /* grow (rare) */
            size_t oldcap = g_symtab.cap;
            char **on = g_symtab.names; Value **os = g_symtab.syms;
            g_symtab.cap *= 2;
            g_symtab.names = calloc(g_symtab.cap, sizeof(char*));
            g_symtab.syms  = calloc(g_symtab.cap, sizeof(Value*));
            g_symtab.used = 0;
            for (size_t k = 0; k < oldcap; k++) {
                if (on[k]) {
                    unsigned long hh = sym_hash(on[k]) & (g_symtab.cap - 1);
                    size_t j = hh;
                    while (g_symtab.names[j]) j = (j + 1) & (g_symtab.cap - 1);
                    g_symtab.names[j] = on[k];
                    g_symtab.syms[j]  = os[k];
                    g_symtab.used++;
                }
            }
            free(on); free(os);
        }
        return v;
    }
}

/* ------------------------------------------------------------------ */
/* constructors                                                        */
/* ------------------------------------------------------------------ */
Value *v_nil(void)   { static Value n = {V_NIL, 1}; n.refcount = 1; return &n; }
Value *v_bool(int b){ static Value t = {V_BOOL,1}; static Value f={V_BOOL,1};
                     t.as.boolean=1; f.as.boolean=0; return b ? &t : &f; }

Value *v_num(double d) {
    Value *v = malloc(sizeof(Value));
    v->type = V_NUM; v->refcount = 1; v->as.num = d; return v;
}
Value *v_sym(const char *s) { return symtab_intern(s); }
Value *v_str(const char *s) {
    Value *v = malloc(sizeof(Value));
    v->type = V_STR; v->refcount = 1; v->as.str = strdup(s); return v;
}
Value *v_pair(Value *car, Value *cdr) {
    Value *v = malloc(sizeof(Value));
    v->type = V_PAIR; v->refcount = 1;
    v->as.pair.car = car; v->as.pair.cdr = cdr;
    v_incref(car); v_incref(cdr);
    return v;
}
Value *v_builtin(BuiltinFn fn, const char *name) {
    Value *v = malloc(sizeof(Value));
    v->type = V_BUILTIN; v->refcount = 1;
    v->as.builtin.fn = fn; v->as.builtin.name = name; return v;
}
Value *v_closure(Value *params, Value *body, Value *env) {
    Value *v = malloc(sizeof(Value));
    v->type = V_CLOSURE; v->refcount = 1;
    v->as.closure.params = params; v->as.closure.body = body; v->as.closure.env = env;
    v_incref(params); v_incref(body); v_incref(env);
    return v;
}
Value *v_env(Value *parent) {
    Value *v = malloc(sizeof(Value));
    v->type = V_ENV; v->refcount = 1;
    v->as.env.vars = NULL; v->as.env.vals = NULL;
    v->as.env.count = 0; v->as.env.cap = 0; v->as.env.parent = parent;
    v_incref(parent);
    return v;
}
Value *v_tower(int order, const double *coeff) {
    Value *v = malloc(sizeof(Value));
    v->type = V_TOWER; v->refcount = 1;
    v->as.tower.order = order;
    v->as.tower.coeff = malloc(sizeof(double) * (order + 1));
    for (int i = 0; i <= order; i++) v->as.tower.coeff[i] = coeff ? coeff[i] : 0.0;
    return v;
}

/* ------------------------------------------------------------------ */
/* refcount                                                            */
/* ------------------------------------------------------------------ */
void v_incref(Value *v) {
    if (!v) return;
    if (v->type == V_NIL || v->type == V_BOOL) return; /* immortal singletons */
    v->refcount++;
}
static void free_value(Value *v);
/* Values are arena-immortal: we never free individual compound/number
   values. Rationale: closures and `quote` capture pointers into the
   parsed source tree; freeing the source tree after eval would leave
   dangling references (use-after-free). eshkol advertises "no garbage
   collector"; we use whole-program lifetime allocation instead. The OS
   reclaims all memory on process exit. (Strings/symbols are likewise
   never freed.) */
void v_decref(Value *v) {
    (void)v;
    return;
}
static void free_value(Value *v) { (void)v; }

/* ------------------------------------------------------------------ */
/* accessors / printing                                                */
/* ------------------------------------------------------------------ */
int v_is_truthy(Value *v) {
    if (!v) return 0;
    if (v->type == V_BOOL) return v->as.boolean;
    if (v->type == V_NIL)  return 0;
    return 1; /* everything else is truthy */
}
const char *v_type_name(Value *v) {
    switch (v->type) {
    case V_NIL: return "nil"; case V_BOOL: return "bool";
    case V_NUM: return "number"; case V_SYM: return "symbol";
    case V_STR: return "string"; case V_PAIR: return "pair";
    case V_BUILTIN: return "builtin"; case V_CLOSURE: return "closure";
    case V_TOWER: return "tower"; case V_ENV: return "env";
    }
    return "?";
}

static void print_impl(Value *v) {
    switch (v->type) {
    case V_NIL:  fputs("()", stdout); break;
    case V_BOOL: fputs(v->as.boolean ? "#t" : "#f", stdout); break;
    case V_NUM: {
        double d = v->as.num;
        if (d == (long long)d && fabs(d) < 1e15) printf("%.0f", d);
        else printf("%.17g", d);
        break;
    }
    case V_SYM:  fputs(v->as.str, stdout); break;
    case V_STR:  printf("\"%s\"", v->as.str); break;
    case V_PAIR: {
        fputc('(', stdout);
        print_impl(v->as.pair.car);
        Value *rest = v->as.pair.cdr;
        while (rest && rest->type == V_PAIR) {
            fputc(' ', stdout);
            print_impl(rest->as.pair.car);
            rest = rest->as.pair.cdr;
        }
        if (rest && rest->type != V_NIL) {
            fputs(" . ", stdout); print_impl(rest);
        }
        fputc(')', stdout);
        break;
    }
    case V_BUILTIN: printf("<builtin:%s>", v->as.builtin.name); break;
    case V_CLOSURE: fputs("<closure>", stdout); break;
    case V_TOWER: {
        printf("<tower[0..%d]:", v->as.tower.order);
        for (int i = 0; i <= v->as.tower.order; i++) printf(" %.*g", 6, v->as.tower.coeff[i]);
        fputc('>', stdout); break;
    }
    case V_ENV: fputs("<env>", stdout); break;
    }
}
void v_print(Value *v) { if (v) print_impl(v); }

char *v_to_string(Value *v) {
    /* Render via a local recursive serializer into a growable buffer. */
    size_t cap = 1 << 16; size_t len = 0;
    char *buf = malloc(cap);
    void ser(Value *x) {
        char num[64];
        switch (x->type) {
        case V_NIL: memcpy(buf+len,"()",2); len+=2; break;
        case V_BOOL: memcpy(buf+len, x->as.boolean?"#t":"#f",2); len+=2; break;
        case V_NUM: {
            double d=x->as.num;
            int n = (d==(long long)d&&fabs(d)<1e15)? snprintf(num,sizeof num,"%.0f",d)
                                                  : snprintf(num,sizeof num,"%.17g",d);
            memcpy(buf+len,num,n); len+=n; break;
        }
        case V_SYM: { size_t l=strlen(x->as.str); memcpy(buf+len,x->as.str,l); len+=l; break; }
        case V_STR: { size_t l=strlen(x->as.str); int n=snprintf(buf+len,cap-l,"\"%s\"",x->as.str); len+=n; break; }
        case V_PAIR: {
            buf[len++]='('; ser(x->as.pair.car);
            Value *r=x->as.pair.cdr;
            while (r&&r->type==V_PAIR){ buf[len++]=' '; ser(r->as.pair.car); r=r->as.pair.cdr; }
            if (r&&r->type!=V_NIL){ buf[len++]=' '; buf[len++]='.'; buf[len++]=' '; ser(r); }
            buf[len++]=')'; break;
        }
        default: { const char *s=v_type_name(x); size_t l=strlen(s); memcpy(buf+len,s,l); len+=l; }
        }
    }
    ser(v);
    buf[len]=0;
    return buf;
}

/* ------------------------------------------------------------------ */
/* environment                                                         */
/* ------------------------------------------------------------------ */
void env_define(Value *env, Value *sym, Value *val) {
    if (env->type != V_ENV) return;
    /* grow */
    if (env->as.env.count == env->as.env.cap) {
        int nc = env->as.env.cap ? env->as.env.cap * 2 : 8;
        env->as.env.vars = realloc(env->as.env.vars, nc * sizeof(Value*));
        env->as.env.vals = realloc(env->as.env.vals, nc * sizeof(Value*));
        env->as.env.cap = nc;
    }
    env->as.env.vars[env->as.env.count] = sym; v_incref(sym);
    env->as.env.vals[env->as.env.count] = val; v_incref(val);
    env->as.env.count++;
}
static int env_find_local(Value *env, Value *sym) {
    for (int i = 0; i < env->as.env.count; i++)
        if (env->as.env.vars[i] == sym) return i;
    return -1;
}
Value *env_lookup(Value *env, Value *sym) {
    for (Value *e = env; e && e->type == V_ENV; e = e->as.env.parent) {
        int i = env_find_local(e, sym);
        if (i >= 0) return e->as.env.vals[i];
    }
    return NULL;
}
void env_set(Value *env, Value *sym, Value *val) {
    for (Value *e = env; e && e->type == V_ENV; e = e->as.env.parent) {
        int i = env_find_local(e, sym);
        if (i >= 0) {
            v_decref(e->as.env.vals[i]);
            e->as.env.vals[i] = val; v_incref(val);
            return;
        }
    }
    /* not found: define in global */
    env_define(env, sym, val);
}
