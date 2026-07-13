#ifndef SLERMES_VALUE_H
#define SLERMES_VALUE_H

#include <stddef.h>

/* Value kinds. Numbers and booleans are stored inline (no heap).
   Heap kinds carry a refcount for a simple ownership discipline. */
typedef enum {
    V_NIL = 0,
    V_BOOL,
    V_NUM,      /* inline double */
    V_SYM,      /* interned symbol (heap str) */
    V_STR,      /* heap string */
    V_PAIR,     /* cons cell (heap) */
    V_BUILTIN,  /* C function pointer (heap) */
    V_CLOSURE,  /* user lambda (heap) */
    V_TOWER,    /* Taylor tower for AD (heap) */
    V_ENV       /* environment frame (heap) */
} ValueType;

typedef struct Value Value;

/* C builtin signature: (args list, argc). Returns a new (or borrowed) Value. */
typedef Value *(*BuiltinFn)(Value *args, int argc);

struct Value {
    ValueType type;
    int refcount;
    union {
        double num;          /* V_NUM */
        int boolean;         /* V_BOOL */
        char *str;           /* V_SYM, V_STR */
        struct {             /* V_PAIR */
            Value *car;
            Value *cdr;
        } pair;
        struct {             /* V_BUILTIN */
            BuiltinFn fn;
            const char *name;
        } builtin;
        struct {             /* V_CLOSURE */
            Value *params;   /* list of symbols */
            Value *body;     /* list of expressions */
            Value *env;      /* closed environment (V_ENV) */
        } closure;
        struct {             /* V_TOWER */
            double *coeff;   /* coeff[0..order] */
            int order;       /* highest coefficient index */
        } tower;
        struct {             /* V_ENV */
            Value **vars;    /* symbol Values */
            Value **vals;    /* Values */
            int count;
            int cap;
            Value *parent;   /* V_ENV or NULL */
        } env;
    } as;
};

/* ---- constructors (return +1 refcount Value*) ---- */
Value *v_nil(void);
Value *v_bool(int b);
Value *v_num(double d);
Value *v_sym(const char *s);   /* interns */
Value *v_str(const char *s);
Value *v_pair(Value *car, Value *cdr);
Value *v_builtin(BuiltinFn fn, const char *name);
Value *v_closure(Value *params, Value *body, Value *env);
Value *v_env(Value *parent);
Value *v_tower(int order, const double *coeff);

/* ---- refcount ---- */
void v_incref(Value *v);
void v_decref(Value *v);

/* ---- accessors ---- */
int        v_is_truthy(Value *v);
const char *v_type_name(Value *v);
void        v_print(Value *v);          /* print to stdout (no newline) */
char       *v_to_string(Value *v);      /* malloc'd, caller frees */

/* environment */
void   env_define(Value *env, Value *sym, Value *val);
Value *env_lookup(Value *env, Value *sym);   /* NULL if unbound */
void   env_set(Value *env, Value *sym, Value *val); /* update existing binding */

#endif /* SLERMES_VALUE_H */
