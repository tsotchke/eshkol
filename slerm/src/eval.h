#ifndef SLERMES_EVAL_H
#define SLERMES_EVAL_H

#include "value.h"

/* Evaluate expr in env. Returns +1 Value*. On error returns a V_STR error
   message (caller should check v_is_error). */
Value *slermes_eval(Value *expr, Value *env);

/* global environment (set up by eval_init) */
extern Value *g_env;

void eval_init(void);

/* helper used by builtins: evaluate a list of args in env */
Value *eval_list(Value *args, Value *env);

#endif /* SLERMES_EVAL_H */
