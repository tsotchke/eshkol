/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * taylor_opcodes.h -- spelling -> op-code lookup over taylor_recurrences.def.
 *
 * The code generator routes a Taylor carrier reaching a numeric builtin to
 * eshkol_taylor_unary_tagged / eshkol_taylor_binary_tagged with the op-code
 * this lookup returns. Both the op-codes and the spellings come from the one
 * table, so a builtin gains a Taylor route exactly when it gains a row there,
 * and the runtime kernel and the code generator cannot disagree on a number.
 */
#ifndef ESHKOL_CORE_TAYLOR_OPCODES_H
#define ESHKOL_CORE_TAYLOR_OPCODES_H

#include <string.h>

/** Op-code of the unary Taylor recurrence for the builtin spelled `name`
 *  (a row's own spelling or a TAYLOR_ALIAS lowering spelling), or -1. */
static inline int eshkol_taylor_unary_opcode(const char* name) {
    static const struct { const char* sexpr; int opcode; } rows[] = {
#define TAYLOR_UN(nm, opcode, sexpr, testfn, x0) { sexpr, (opcode) },
#include "taylor_recurrences.def"
    };
    static const struct { const char* alias; const char* sexpr; } aliases[] = {
#define TAYLOR_ALIAS(alias, nm) { alias, #nm },
#include "taylor_recurrences.def"
    };
    if (!name) return -1;
    for (size_t i = 0; i < sizeof(aliases) / sizeof(aliases[0]); i++)
        if (strcmp(aliases[i].alias, name) == 0) { name = aliases[i].sexpr; break; }
    /* "-" names both the binary subtraction and the unary negation row; a
     * math builtin is never spelled that way. */
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++)
        if (strcmp(rows[i].sexpr, name) == 0 && strcmp(name, "-") != 0) return rows[i].opcode;
    return -1;
}

/** Op-code of the binary Taylor recurrence spelled `name`, or -1. */
static inline int eshkol_taylor_binary_opcode(const char* name) {
    static const struct { const char* sexpr; int opcode; } rows[] = {
#define TAYLOR_BIN(nm, opcode, sexpr) { sexpr, (opcode) },
#include "taylor_recurrences.def"
    };
    if (!name) return -1;
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++)
        if (strcmp(rows[i].sexpr, name) == 0) return rows[i].opcode;
    return -1;
}

#endif /* ESHKOL_CORE_TAYLOR_OPCODES_H */
