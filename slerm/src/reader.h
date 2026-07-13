#ifndef SLERMES_READER_H
#define SLERMES_READER_H

#include "value.h"

/* Read one S-expression from a string. On success returns a Value* (+1)
   and sets *rest to point just past the consumed token. On error returns
   NULL and sets *err to a malloc'd message. */
Value *read_sexpr(const char *src, const char **rest, char **err);

/* Read ALL top-level S-exprs in src; returns a list (V_PAIR chain) of them. */
Value *read_all(const char *src, char **err);

#endif /* SLERMES_READER_H */
