/**
 * @file abi_fingerprint.h
 * @brief Link-time fingerprint for the heap object ABI.
 *
 * Every heap-allocated Eshkol object is a payload preceded by an
 * ::eshkol_object_header_t, and the pointer that circulates through compiled
 * code, the runtime, and the embedding API points at the *payload*. The header
 * is reached by subtracting its size. The `subtype` field — the only thing that
 * says what kind of object this is — lives inside that header.
 *
 * That arrangement has a consequence worth stating plainly, because it governs
 * how the ABI may be changed: **an object carries no discriminator for its own
 * layout.** Given a payload pointer there is no test that distinguishes an
 * object laid out one way from an object laid out another. You must already
 * know. So two halves of a toolchain that disagree about the layout do not
 * fail to understand each other — they agree confidently and read the wrong
 * bytes. Nothing crashes. The program returns wrong answers.
 *
 * This header removes that failure mode by moving the disagreement to link
 * time, where it is loud.
 *
 * ## Mechanism
 *
 * The layout is described by four numbers that fully determine whether two
 * separately-compiled halves can safely exchange objects: the ABI version, the
 * header size, the offset of `subtype` within the header, and the payload
 * alignment. Those numbers are pasted by the preprocessor into the *name* of a
 * guard symbol, and every participant emits a reference to it:
 *
 *   - every C/C++ translation unit that includes this header, via
 *     ::ESHKOL_ABI_FINGERPRINT_ANCHOR;
 *   - every LLVM module the compiler emits, via `MemoryCodegen`, which plants
 *     the same reference in generated code.
 *
 * The runtime library defines exactly one such symbol. If any half is built
 * against a different layout, its reference names a symbol that does not exist,
 * and the link fails with an undefined-symbol error naming the layout it wanted.
 * A stale object file, a stale cached JIT artifact, a stale installed runtime,
 * a `--shared-lib` artifact built by yesterday's compiler: each becomes a link
 * error instead of a wrong answer.
 *
 * The guard is deliberately a *data* symbol rather than a function, so the
 * reference survives any level of optimization and inlining, and costs one
 * pointer-sized word per participating image.
 *
 * ## The two failure directions, and what each looks like
 *
 * | Situation | Before | With the guard |
 * |---|---|---|
 * | Old compiler output, new runtime | links, silent wrong data | undefined symbol at link |
 * | New compiler output, old runtime | links, silent wrong data | undefined symbol at link |
 * | Cached artifact from the other layout | reused, silent wrong data | undefined symbol at link |
 * | Layout changed but the numbers below were not | silent wrong data | compile error (static assert) |
 *
 * The last row matters as much as the others: the static assertions at the
 * bottom of this file bind each number to the real `offsetof`, so the
 * fingerprint cannot drift away from the layout it claims to describe. Change
 * the struct and the build stops; change the numbers and the symbol renames.
 * There is no path that changes the layout quietly.
 *
 * ## Changing the ABI
 *
 * Bump ::ESHKOL_OBJECT_ABI_VERSION and set the other numbers to the new layout.
 * The symbol renames itself, and everything built against the old one refuses to
 * link until it is rebuilt. That is the intended experience.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef ESHKOL_ABI_FINGERPRINT_H
#define ESHKOL_ABI_FINGERPRINT_H

#include <stddef.h>
#include <stdint.h>

/* Self-contained on purpose, and this include is load-bearing.
 *
 * The layout numbers below are selected by ESHKOL_MEMORY_ABI_ACTIVE, which
 * memory_abi_v2.h defines and eshkol.h pulls in. A translation unit that
 * included this header first would have found that macro undefined, fallen to
 * the v1 branch, and produced a v1 guard inside a v2 build — the two halves of a
 * single build disagreeing, which is the failure this header exists to prevent,
 * arriving through the header meant to prevent it. Depending on include order
 * for a correctness property is not a risk worth carrying, so the dependency is
 * taken here rather than assumed of every caller. */
#include <eshkol/eshkol.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * The four numbers that define exchange compatibility.
 *
 * These are the layout facts that a caller must agree with a callee about in
 * order to hand it an object. Fields other than `subtype` are checked by the
 * static assertions below but do not enter the symbol name: moving `flags`
 * within a header of unchanged size is still an incompatible change, and is
 * caught at compile time by the assertions, which every participant compiles.
 *
 * They are selected by ESHKOL_MEMORY_ABI_ACTIVE — the switch memory_abi_v2.h
 * already exposes — rather than being maintained separately. That is the point
 * of the coupling: configuring with `-DESHKOL_MEMORY_ABI_V2=ON` renames the
 * guard symbol on its own, so the migration flip itself is protected without
 * anyone having to remember to protect it. A build half-flipped between the two
 * layouts cannot link.
 *
 * They must be spelled as integer literals, not as `sizeof`, because the
 * preprocessor pastes them into the symbol's name.
 * ───────────────────────────────────────────────────────────────────────── */

#if !defined(ESHKOL_MEMORY_ABI_ACTIVE)
#error "eshkol/abi_fingerprint.h: ESHKOL_MEMORY_ABI_ACTIVE is not defined. \
The include of <eshkol/eshkol.h> above should have supplied it; silently \
assuming an ABI here would let two halves of one build disagree."
#endif

#if ESHKOL_MEMORY_ABI_ACTIVE == ESHKOL_MEMORY_ABI_V2

/** @brief Object ABI generation. Selected by ESHKOL_MEMORY_ABI_V2_ENABLED. */
#define ESHKOL_OBJECT_ABI_VERSION      2
/** @brief Bytes between the header's first byte and the payload's first byte. */
#define ESHKOL_OBJECT_ABI_HEADER_SIZE  32
/** @brief Offset of the subtype byte within the header. */
#define ESHKOL_OBJECT_ABI_SUBTYPE_OFF  6
/** @brief Alignment guaranteed for the payload pointer. */
#define ESHKOL_OBJECT_ABI_PAYLOAD_ALIGN 16

#else

/** @brief Object ABI generation. Selected by ESHKOL_MEMORY_ABI_V2_ENABLED. */
#define ESHKOL_OBJECT_ABI_VERSION      1
/** @brief Bytes between the header's first byte and the payload's first byte. */
#define ESHKOL_OBJECT_ABI_HEADER_SIZE  8
/** @brief Offset of the subtype byte within the header. */
#define ESHKOL_OBJECT_ABI_SUBTYPE_OFF  0
/** @brief Alignment guaranteed for the payload pointer. */
#define ESHKOL_OBJECT_ABI_PAYLOAD_ALIGN 8

#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Symbol-name construction.
 *
 * Two-level token pasting is required: the inner macro pastes, the outer one
 * forces its arguments to be expanded first, so ESHKOL_OBJECT_ABI_VERSION is
 * spelled as its value and not as its name.
 * ───────────────────────────────────────────────────────────────────────── */

#define ESHKOL_ABI_CAT_(a, b) a##b
#define ESHKOL_ABI_CAT(a, b)  ESHKOL_ABI_CAT_(a, b)

/* The inner macro does the pasting; the outer one exists so its arguments are
 * macro-expanded to their values before they reach it. Without the indirection
 * the symbol would be named after the macros rather than after the layout, and
 * would therefore never change when the layout did. */
#define ESHKOL_ABI_NAME_(v, h, s, a) eshkol_object_abi_v##v##_h##h##_s##s##_a##a
#define ESHKOL_ABI_NAME(v, h, s, a)  ESHKOL_ABI_NAME_(v, h, s, a)

/**
 * @brief The guard symbol's identifier, spelled from the layout numbers.
 *
 * Reads as, for the current layout: `eshkol_object_abi_v1_h8_s0_a8`.
 */
#define ESHKOL_ABI_FINGERPRINT_SYMBOL                                          \
    ESHKOL_ABI_NAME(ESHKOL_OBJECT_ABI_VERSION,                                 \
                    ESHKOL_OBJECT_ABI_HEADER_SIZE,                             \
                    ESHKOL_OBJECT_ABI_SUBTYPE_OFF,                             \
                    ESHKOL_OBJECT_ABI_PAYLOAD_ALIGN)

/* The same name as a string, for the code generator (which must plant the
 * reference into emitted LLVM IR, where it has no preprocessor) and for
 * diagnostics. Kept in step with the identifier by
 * eshkol_abi_fingerprint_name(), which is defined in the same translation unit
 * that defines the symbol and returns the stringified identifier — so the two
 * spellings cannot drift apart unnoticed. */
#define ESHKOL_ABI_STR_(x) #x
#define ESHKOL_ABI_STR(x)  ESHKOL_ABI_STR_(x)
#define ESHKOL_ABI_FINGERPRINT_NAME ESHKOL_ABI_STR(ESHKOL_ABI_FINGERPRINT_SYMBOL)

/**
 * @brief The guard symbol. Defined exactly once, by lib/core/abi_fingerprint.c.
 *
 * Its value is the header size, which makes it useful to read as well as to
 * require; its *name* is what does the work.
 */
extern const size_t ESHKOL_ABI_FINGERPRINT_SYMBOL;

/**
 * @brief The guard symbol's name as the runtime library was built to spell it.
 *
 * A consumer that dlopen()s the runtime rather than linking against it does not
 * get the link-time check, and can use this to perform the same check at load
 * time: compare against ::ESHKOL_ABI_FINGERPRINT_NAME.
 */
const char *eshkol_abi_fingerprint_name(void);

/**
 * @brief Header size the runtime library itself was compiled with.
 *
 * Distinct from ::ESHKOL_OBJECT_ABI_HEADER_SIZE, which is what the *caller* was
 * compiled with. Equal in any correctly linked program; comparing them is how a
 * dlopen()ing consumer checks what a static link would have checked for it.
 */
size_t eshkol_abi_runtime_header_size(void);

/**
 * @brief Force this translation unit to reference the guard symbol.
 *
 * Place once at file scope in any translation unit that must participate. The
 * `used` attribute keeps the reference through optimization; without it the
 * unreferenced static is discarded and the check evaporates.
 *
 * Including this header does not by itself create the reference — a header
 * cannot, without emitting a definition into every includer. Participation is
 * therefore explicit, and lib/core/abi_fingerprint.c anchors the runtime.
 */
#if defined(__GNUC__) || defined(__clang__)
#  define ESHKOL_ABI_FINGERPRINT_ANCHOR                                        \
      static const size_t *const ESHKOL_ABI_CAT(eshkol_abi_anchor_, __LINE__)  \
          __attribute__((used)) = &ESHKOL_ABI_FINGERPRINT_SYMBOL
#else
#  define ESHKOL_ABI_FINGERPRINT_ANCHOR                                        \
      extern const size_t *const ESHKOL_ABI_CAT(eshkol_abi_anchor_, __LINE__); \
      const size_t *const ESHKOL_ABI_CAT(eshkol_abi_anchor_, __LINE__) =       \
          &ESHKOL_ABI_FINGERPRINT_SYMBOL
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Binding the fingerprint to the real layout.
 *
 * Without these, the numbers above are a claim. With them, they are a
 * description that the compiler checks on every build.
 *
 * Unconditional: the include at the top of this file guarantees the header
 * types are in scope, so there is no configuration in which these assertions
 * are skipped. An assertion that can be skipped by include order is an
 * assertion that is not there on the day it matters.
 * ───────────────────────────────────────────────────────────────────────── */

#if defined(__cplusplus)
#  define ESHKOL_ABI_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define ESHKOL_ABI_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#  define ESHKOL_ABI_ASSERT(cond, msg) /* no compile-time assertions available */
#endif

/* Against the ACTIVE header, so the assertions follow the flag. Under v1 this
 * is eshkol_object_header_t; under v2, eshkol_object_header_v2_t. */
ESHKOL_ABI_ASSERT(sizeof(eshkol_object_header_active_t) == ESHKOL_OBJECT_ABI_HEADER_SIZE,
                  "ESHKOL_OBJECT_ABI_HEADER_SIZE no longer describes the active "
                  "object header. Update inc/eshkol/abi_fingerprint.h and bump "
                  "ESHKOL_OBJECT_ABI_VERSION so stale objects fail to link.");

ESHKOL_ABI_ASSERT(offsetof(eshkol_object_header_active_t, subtype) == ESHKOL_OBJECT_ABI_SUBTYPE_OFF,
                  "subtype moved within the active object header; the ABI fingerprint is stale.");

ESHKOL_ABI_ASSERT(ESHKOL_OBJECT_PAYLOAD_ALIGN == ESHKOL_OBJECT_ABI_PAYLOAD_ALIGN,
                  "payload alignment disagrees with the ABI fingerprint; a mixed "
                  "link would not be detected.");

/* The v1 layout is pinned field by field regardless of which ABI is active,
 * because generated code emits -8/-7/-6/-4 as separate literals: a field that
 * moves inside a header of unchanged size breaks that code while every
 * size assertion still passes. memory_abi_v2.h pins subtype, flags and size;
 * ref_count is pinned here so the set is complete. */
ESHKOL_ABI_ASSERT(offsetof(eshkol_object_header_t, ref_count) == 2,
                  "ref_count moved within the v1 object header; generated code "
                  "reads it at payload-6.");



#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_ABI_FINGERPRINT_H */
