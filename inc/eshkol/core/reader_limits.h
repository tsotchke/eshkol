/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Reader limits — the one input-size cap every Eshkol reader shares, and the
 * one traversal-depth budget shared by the compiler passes that walk a datum
 * tree with native recursion.
 *
 * NESTING DEPTH IS BOUNDED BY MEMORY, NOT BY A NUMBER.  Both readers — the
 * native front end (lib/frontend/parser.cpp) and the bytecode VM's reader
 * (lib/backend/vm_parser.c) — carry their nesting on an explicit heap stack,
 * so a datum nests as deeply as the machine has memory for.  There is no
 * depth limit on reading and none is configurable.
 *
 * Two budgets do exist, and both are documented and configurable:
 *
 *   1. INPUT SIZE.  A reader accepts a source text up to
 *      eshkol_reader_max_source_bytes().  A larger text is refused before a
 *      byte of it is read, with an ordinary Eshkol error that names the cap.
 *      Default ESHKOL_READER_DEFAULT_MAX_SOURCE_BYTES (64 MiB); set by the
 *      ESHKOL_MAX_SOURCE_BYTES environment variable or the
 *      `--max-source-bytes=N` command-line option.  0 means "no cap".
 *
 *   2. EXPRESSION-NESTING BUDGET for the compiler passes that walk a form
 *      with native recursion.  Reading a 10^6-deep datum always succeeds
 *      within memory, and so do the paths that build, print, compare and
 *      collect it; a pass that compiles a 10^6-deep CODE form instead names
 *      this budget and refuses, with an ordinary error.  Default
 *      ESHKOL_READER_DEFAULT_MAX_PASS_DEPTH; set by ESHKOL_MAX_PASS_DEPTH or
 *      `--max-pass-depth=N`.  0 means "no count, the native stack guard
 *      decides" — and that guard (inc/eshkol/core/stack_guard.h) is consulted
 *      whatever the count is, so raising this number cannot turn into a
 *      signal: the pass still refuses, one frame before the bound.
 *
 * Both accessors are safe to call from any reader at any time: the values are
 * resolved from the environment once per process on first use, and a setter
 * (the command-line option) overrides that resolution for the rest of the run.
 */
#ifndef ESHKOL_CORE_READER_LIMITS_H
#define ESHKOL_CORE_READER_LIMITS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 64 MiB of source text.  Generous: the largest file in the shipped standard
 * library and test corpus is three orders of magnitude smaller, and a 64 MiB
 * datum still reads on a machine with a few hundred MiB free. */
#define ESHKOL_READER_DEFAULT_MAX_SOURCE_BYTES ((size_t)64u * 1024u * 1024u)

/* 1,000 levels of CODE nesting.  Data nesting has no limit: the readers, the
 * release of a tree, effect lowering, quoted-datum emission, printing,
 * comparison and collection all carry their own stack, so a 10^6-deep quoted
 * datum goes through.  A form nested a thousand deep is a different thing —
 * no human writes one — and the passes that compile it stay recursive, so
 * this is the budget they name when they refuse. */
#define ESHKOL_READER_DEFAULT_MAX_PASS_DEPTH ((size_t)1000u)

/** @brief The input-size cap in bytes; 0 means no cap. */
size_t eshkol_reader_max_source_bytes(void);

/** @brief Override the input-size cap for the rest of the process
 *         (the `--max-source-bytes=N` option).  0 removes the cap. */
void eshkol_reader_set_max_source_bytes(size_t bytes);

/** @brief The expression-nesting budget; 0 means no count (the native
 *         stack guard alone decides). */
size_t eshkol_reader_max_pass_depth(void);

/** @brief Override the expression-nesting budget (the
 *         `--max-pass-depth=N` option).  0 removes the count. */
void eshkol_reader_set_max_pass_depth(size_t depth);

/**
 * @brief Tests a source text's length against the input-size cap.
 *
 * @param bytes   Length of the source text in bytes.
 * @param message Buffer that receives the user-facing refusal text when the
 *                text is over the cap.  May be NULL.
 * @param cap     Size of @p message.
 * @return 1 when @p bytes is over the cap (and @p message is filled), else 0.
 */
int eshkol_reader_source_over_cap(size_t bytes, char* message, size_t cap);

/**
 * @brief Fills @p message with the user-facing text for a pass that reached
 *        its expression-nesting budget, or its native stack bound.
 *
 * @param pass_name The pass naming itself, e.g. "quoted-datum code generation".
 * @param depth     The budget that was reached.
 */
void eshkol_reader_depth_message(const char* pass_name, size_t depth,
                                 char* message, size_t cap);

/**
 * @brief Parses a `--max-source-bytes=N` / `--max-pass-depth=N` option.
 *
 * Accepts a plain decimal byte count and the suffixes K, M and G (1024-based),
 * so `--max-source-bytes=128M` is accepted as written.
 *
 * @return 1 when @p text parsed and *@p out holds the value, else 0.
 */
int eshkol_reader_parse_size(const char* text, size_t* out);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_READER_LIMITS_H */
