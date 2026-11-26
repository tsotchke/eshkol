/**
 * @file vm_char_string_ext.c
 * @brief Extended R7RS character and string operations for the Eshkol
 *        bytecode VM (ADR-0064 Wave 1B), plus the SRFI-69 `hash-table-ref`
 *        lookup spelling.
 *
 * Reserved native call ID range: 2300-2349.
 *
 * ARCHITECTURE — two layers, deliberately separable:
 *
 *   Layer A (pure)  operates only on `VmString` / `VmRegionStack` from
 *                   vm_string.c and on bare codepoints. It has no knowledge
 *                   of `Value`, the heap table, or the interpreter, which is
 *                   what lets tests/vm_parity/test_wave1b_asan.c exercise
 *                   every UTF-8 boundary directly under ASan + UBSan.
 *
 *   Layer B (VM)    binds Layer A to the VM calling convention. It is
 *                   compiled out when VM_CHAR_STRING_EXT_PURE_HELPERS_ONLY
 *                   is defined, which is how the pure harness builds.
 *
 * INVARIANTS (ADR-0064 gates):
 *   - Pure C11. No compiler extensions, no VLAs, no anonymous unions beyond
 *     what the included VM headers already use.
 *   - ZERO malloc/calloc/realloc/free. Every allocation goes through the
 *     OALR arena (`vm_alloc` on `vm->heap.regions`).
 *   - ZERO mutable global or static state.
 *   - The `Value` representation is untouched: a character stays an
 *     immediate codepoint in `.as.i` under tag VAL_CHAR.
 *
 * CASE MAPPING — this file applies simple ASCII case folding to the FULL
 * decoded codepoint, which is the same convention vm_string_ascii_case()
 * already established for `string-upcase` / `string-downcase`. It agrees
 * with the native lowering across the whole ASCII range and, unlike a
 * truncation to one byte, never aliases a codepoint above U+007F onto an
 * ASCII answer.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef VM_CHAR_STRING_EXT_C_INCLUDED
#define VM_CHAR_STRING_EXT_C_INCLUDED

#include <limits.h>
#include <string.h>

/* ════════════════════════════════════════════════════════════════════════
 * Layer A — pure codepoint and VmString helpers (no VM dependency)
 * ════════════════════════════════════════════════════════════════════════ */

/** @brief R7RS `char-foldcase` on a bare codepoint: simple ASCII folding,
 *         identity everywhere else. */
static int vm_cse_char_foldcase(int cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    return cp;
}

/** @brief R7RS `digit-value` on a bare codepoint.
 * @return The decimal value 0-9, or -1 when @p cp is not a decimal digit. */
static int vm_cse_digit_value(int cp) {
    if (cp >= '0' && cp <= '9') return cp - '0';
    return -1;
}

/** @brief Three-way case-folded codepoint comparison. */
static int vm_cse_char_ci_cmp(int a, int b) {
    int fa = vm_cse_char_foldcase(a);
    int fb = vm_cse_char_foldcase(b);
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

/**
 * @brief Three-way lexicographic comparison of two strings by codepoint,
 *        optionally case-folding each codepoint first.
 *
 * A NULL operand sorts before a non-NULL one, matching how vm_string_lt()
 * and vm_string_eq() already treat a missing payload, so a malformed operand
 * can never steer the comparison into undefined behaviour.
 *
 * @param fold Non-zero to apply simple ASCII case folding (the `-ci` family).
 * @return -1, 0 or 1.
 */
static int vm_cse_string_cmp(const VmString* a, const VmString* b, int fold) {
    if (!a || !b) {
        if (a == b) return 0;
        return a ? 1 : -1;
    }
    int pa = 0, pb = 0;
    while (pa < a->byte_len && pb < b->byte_len) {
        int ca = vm_utf8_decode(a->data, a->byte_len, &pa);
        int cb = vm_utf8_decode(b->data, b->byte_len, &pb);
        if (fold) { ca = vm_cse_char_foldcase(ca); cb = vm_cse_char_foldcase(cb); }
        if (ca < cb) return -1;
        if (ca > cb) return 1;
    }
    int rest_a = (pa < a->byte_len);
    int rest_b = (pb < b->byte_len);
    if (rest_a == rest_b) return 0;   /* both exhausted: equal */
    return rest_a ? 1 : -1;           /* a proper prefix is the lesser string */
}

/**
 * @brief R7RS `string-foldcase`: a NEW arena string with every codepoint
 *        simple-case-folded.
 *
 * The output extent is counted in a first pass before anything is written,
 * so a fold that changes a codepoint's encoded width can never run past the
 * allocation.
 */
static VmString* vm_cse_string_foldcase(VmRegionStack* rs, const VmString* s) {
    if (!s || s->byte_len < 0 || (s->byte_len && !s->data)) return NULL;

    int pos = 0, required = 0;
    while (pos < s->byte_len) {
        int cp = vm_cse_char_foldcase(vm_utf8_decode(s->data, s->byte_len, &pos));
        char enc[4];
        int count = vm_utf8_encode(cp, enc);
        if (required > INT_MAX - count) return NULL;
        required += count;
    }

    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, (size_t)required + 1);
    if (!result->data) return NULL;

    pos = 0;
    int out = 0, chars = 0;
    while (pos < s->byte_len) {
        int cp = vm_cse_char_foldcase(vm_utf8_decode(s->data, s->byte_len, &pos));
        char enc[4];
        int count = vm_utf8_encode(cp, enc);
        if (count > required - out) return NULL;
        memcpy(result->data + out, enc, (size_t)count);
        out += count;
        ++chars;
    }
    result->data[out] = '\0';
    result->byte_len = out;
    result->char_len = chars;
    return result;
}

/**
 * @brief `(string-set! s k char)` as a genuine in-place mutation.
 *
 * `VmString` keeps its payload behind an indirection, so replacing the
 * payload and updating the header mutates the object every holder of the
 * string already references — which is what R7RS asks for, and what the
 * returning form at native call 552 cannot provide.
 *
 * The replacement codepoint may encode to a different number of bytes than
 * the one it replaces, so a fresh payload of the exact new extent is taken
 * from the arena rather than written over the old one. The old payload is
 * left to the region; nothing is freed.
 *
 * @param k Codepoint index, required to satisfy 0 <= k < char_len.
 * @return 1 on success, 0 when the operands are out of range or the arena
 *         is exhausted — in which case @p s is left completely untouched.
 */
static int vm_cse_string_set(VmRegionStack* rs, VmString* s, int k, int cp) {
    if (!s || !s->data || k < 0 || k >= s->char_len) return 0;

    int start_byte = vm_utf8_byte_offset(s->data, s->byte_len, k);
    if (start_byte < 0 || start_byte >= s->byte_len) return 0;
    int end_byte = start_byte;
    vm_utf8_decode(s->data, s->byte_len, &end_byte);
    if (end_byte < start_byte || end_byte > s->byte_len) return 0;

    char enc[4];
    int enc_len = vm_utf8_encode(cp, enc);

    int old_bytes = end_byte - start_byte;
    if (s->byte_len - old_bytes > INT_MAX - enc_len) return 0;
    int new_byte_len = s->byte_len - old_bytes + enc_len;

    char* buf = (char*)vm_alloc(rs, (size_t)new_byte_len + 1);
    if (!buf) return 0;

    memcpy(buf, s->data, (size_t)start_byte);
    memcpy(buf + start_byte, enc, (size_t)enc_len);
    memcpy(buf + start_byte + enc_len, s->data + end_byte,
           (size_t)(s->byte_len - end_byte));
    buf[new_byte_len] = '\0';

    /* Publish the new payload only once it is fully built. */
    s->data = buf;
    s->byte_len = new_byte_len;
    /* The codepoint count is unchanged: exactly one codepoint was replaced. */
    return 1;
}

/**
 * @brief `(string-copy! to at from start end)` per R7RS 6.7, indexed by
 *        codepoint and UTF-8 aware.
 *
 * The copied codepoints may differ in encoded width from the ones they
 * displace, so the result is assembled whole in one fresh arena payload and
 * then published onto @p to. Building the result out-of-place also makes the
 * overlapping case (`to == from`) correct by construction: every byte read
 * comes from the ORIGINAL payload, which the assembly never writes to.
 *
 * @param at    Destination codepoint index in @p to.
 * @param start Inclusive source codepoint index in @p from.
 * @param end   Exclusive source codepoint index in @p from.
 * @return 1 on success, 0 on any range violation or arena exhaustion, with
 *         @p to left untouched.
 */
static int vm_cse_string_copy_bang(VmRegionStack* rs, VmString* to, int at,
                                   const VmString* from, int start, int end) {
    if (!to || !to->data || !from || !from->data) return 0;
    if (at < 0 || at > to->char_len) return 0;
    if (start < 0 || end < start || end > from->char_len) return 0;

    int n = end - start;
    /* R7RS: the copied run must fit inside the destination. */
    if (n > to->char_len - at) return 0;
    if (n == 0) return 1;

    int dst_lo = vm_utf8_byte_offset(to->data, to->byte_len, at);
    int dst_hi = vm_utf8_byte_offset(to->data, to->byte_len, at + n);
    int src_lo = vm_utf8_byte_offset(from->data, from->byte_len, start);
    int src_hi = vm_utf8_byte_offset(from->data, from->byte_len, end);
    if (dst_lo < 0 || dst_hi < dst_lo || dst_hi > to->byte_len) return 0;
    if (src_lo < 0 || src_hi < src_lo || src_hi > from->byte_len) return 0;

    int src_bytes = src_hi - src_lo;
    int dst_bytes = dst_hi - dst_lo;
    if (to->byte_len - dst_bytes > INT_MAX - src_bytes) return 0;
    int new_byte_len = to->byte_len - dst_bytes + src_bytes;

    char* buf = (char*)vm_alloc(rs, (size_t)new_byte_len + 1);
    if (!buf) return 0;

    memcpy(buf, to->data, (size_t)dst_lo);
    memcpy(buf + dst_lo, from->data + src_lo, (size_t)src_bytes);
    memcpy(buf + dst_lo + src_bytes, to->data + dst_hi,
           (size_t)(to->byte_len - dst_hi));
    buf[new_byte_len] = '\0';

    to->data = buf;
    to->byte_len = new_byte_len;
    /* Codepoint count is preserved: n codepoints replaced n codepoints. */
    return 1;
}

#ifndef VM_CHAR_STRING_EXT_PURE_HELPERS_ONLY

/* ════════════════════════════════════════════════════════════════════════
 * Layer B — VM calling convention
 * ════════════════════════════════════════════════════════════════════════ */

/** @brief Reserved Wave 1B native call ID range (ADR-0064). */
#define VM_CSE_FID_BASE  2300
#define VM_CSE_FID_LIMIT 2350

/* Character family */
#define VM_CSE_CHAR_CI_EQ      2300
#define VM_CSE_CHAR_CI_LT      2301
#define VM_CSE_CHAR_CI_GT      2302
#define VM_CSE_CHAR_CI_LE      2303
#define VM_CSE_CHAR_CI_GE      2304
#define VM_CSE_CHAR_FOLDCASE   2305
#define VM_CSE_DIGIT_VALUE     2306
#define VM_CSE_CHAR_READY      2307
/* String comparison family */
#define VM_CSE_STRING_LE       2310
#define VM_CSE_STRING_GE       2311
#define VM_CSE_STRING_GT       2312
#define VM_CSE_STRING_CI_LT    2313
#define VM_CSE_STRING_CI_LE    2314
#define VM_CSE_STRING_CI_GT    2315
#define VM_CSE_STRING_CI_GE    2316
/* String transformation / search / mutation family */
#define VM_CSE_STRING_FOLDCASE 2317
#define VM_CSE_STRING_CONTAINS 2318
#define VM_CSE_STRING_INDEX    2319
#define VM_CSE_STRING_SET      2320
#define VM_CSE_STRING_COPY     2321
#define VM_CSE_STRING_PORT_P   2322
/* Hash table lookup */
#define VM_CSE_HASH_TABLE_REF  2348

/** @brief Codepoint carried by a character operand.
 * @return The codepoint, or -1 when @p v is not a character-like value.
 *         VAL_INT is accepted alongside VAL_CHAR because the existing
 *         character comparisons at native calls 1687-1689 do, so a program
 *         that already passes a raw codepoint keeps its answer. */
static int vm_cse_char_cp(Value v) {
    if (v.type == VAL_CHAR || v.type == VAL_INT) {
        if (v.as.i < 0 || v.as.i > 0x10FFFF) return -1;
        return (int)v.as.i;
    }
    return -1;
}

/** @brief Wrap an arena `VmString` as a fresh heap-backed VAL_STRING.
 *         This is the non-pushing counterpart of VM_PUSH_HEAP_OPAQUE. */
static Value vm_cse_string_value(VM* vm, VmString* s) {
    if (!s) return NIL_VAL;
    int32_t hp = heap_alloc(&vm->heap);
    if (hp < 0) { vm->error = 1; return NIL_VAL; }
    vm->heap.objects[hp]->type = HEAP_STRING;
    vm->heap.objects[hp]->opaque.ptr = s;
    return (Value){ .type = VAL_STRING, .as.ptr = hp };
}

/** @brief Mutable `VmString` behind a VAL_STRING operand, or NULL. */
static VmString* vm_cse_mutable_string(VM* vm, Value v) {
    if (v.type != VAL_STRING) return NULL;
    return vm_value_as_string(vm, v);
}

/** @brief Integral operand of a string index, defaulting an absent one. */
static int vm_cse_index_arg(Value v, int dflt) {
    if (vm_native_absent(v)) return dflt;
    if (v.type == VAL_INT || v.type == VAL_CHAR) return (int)v.as.i;
    return (int)as_number(v);
}

/**
 * @brief Operand count each Wave 1B native call expects on the value stack.
 *
 * This mirrors the `arity` column of the BUILTINS[] rows that bind these
 * IDs: the SHAPE of the call the op reads, with an omitted optional operand
 * arriving as the VAL_VOID absent marker. Returning 0 for an unclaimed ID is
 * what keeps the dispatcher's "not mine" answer unambiguous.
 */
static int vm_cse_operand_count(int fid) {
    switch (fid) {
    case VM_CSE_CHAR_FOLDCASE:
    case VM_CSE_DIGIT_VALUE:
    case VM_CSE_CHAR_READY:
    case VM_CSE_STRING_FOLDCASE:
    case VM_CSE_STRING_PORT_P:
        return 1;
    case VM_CSE_CHAR_CI_EQ:
    case VM_CSE_CHAR_CI_LT:
    case VM_CSE_CHAR_CI_GT:
    case VM_CSE_CHAR_CI_LE:
    case VM_CSE_CHAR_CI_GE:
    case VM_CSE_STRING_LE:
    case VM_CSE_STRING_GE:
    case VM_CSE_STRING_GT:
    case VM_CSE_STRING_CI_LT:
    case VM_CSE_STRING_CI_LE:
    case VM_CSE_STRING_CI_GT:
    case VM_CSE_STRING_CI_GE:
    case VM_CSE_STRING_CONTAINS:
    case VM_CSE_STRING_INDEX:
        return 2;
    case VM_CSE_STRING_SET:
    case VM_CSE_HASH_TABLE_REF:
        return 3;
    case VM_CSE_STRING_COPY:
        return 5;
    default:
        return 0;
    }
}

/**
 * @brief Pure Wave 1B native handler: reads @p args, writes @p out, touches
 *        no value stack.
 *
 * Keeping the handler body free of vm_pop()/vm_push() is what makes every
 * case here reachable from a unit harness, and it keeps the stack discipline
 * in exactly one place (the adapter below).
 *
 * @param argc Number of operands supplied in @p args; an operand the caller
 *             omitted is present as the VAL_VOID absent marker.
 * @return 1 when @p fid belongs to this module and @p out was written, 0
 *         when the ID is not in the reserved range.
 */
static int vm_char_string_ext_dispatch(VM* vm, int fid, int argc, Value* args, Value* out) {
    if (!vm || !out) return 0;
    if (fid < VM_CSE_FID_BASE || fid >= VM_CSE_FID_LIMIT) return 0;
    if (vm_cse_operand_count(fid) == 0) return 0;
    if (argc < vm_cse_operand_count(fid) || !args) {
        /* A short call cannot be served; answer #f rather than read past the
         * operands the caller actually supplied. */
        *out = BOOL_VAL(0);
        return 1;
    }

    VmRegionStack* rs = &vm->heap.regions;

    switch (fid) {

    /* ── Character comparison, case-folded (R7RS 6.6) ────────────────── */
    case VM_CSE_CHAR_CI_EQ:
    case VM_CSE_CHAR_CI_LT:
    case VM_CSE_CHAR_CI_GT:
    case VM_CSE_CHAR_CI_LE:
    case VM_CSE_CHAR_CI_GE: {
        int a = vm_cse_char_cp(args[0]);
        int b = vm_cse_char_cp(args[1]);
        if (a < 0 || b < 0) { *out = BOOL_VAL(0); break; }
        int c = vm_cse_char_ci_cmp(a, b);
        int r;
        switch (fid) {
        case VM_CSE_CHAR_CI_EQ: r = (c == 0); break;
        case VM_CSE_CHAR_CI_LT: r = (c <  0); break;
        case VM_CSE_CHAR_CI_GT: r = (c >  0); break;
        case VM_CSE_CHAR_CI_LE: r = (c <= 0); break;
        default:                r = (c >= 0); break;
        }
        *out = BOOL_VAL(r);
        break;
    }

    /* ── char-foldcase: VAL_CHAR in, VAL_CHAR out ───────────────────── */
    case VM_CSE_CHAR_FOLDCASE: {
        int cp = vm_cse_char_cp(args[0]);
        if (cp < 0) { *out = BOOL_VAL(0); break; }
        *out = (Value){ .type = VAL_CHAR, .as.i = vm_cse_char_foldcase(cp) };
        break;
    }

    /* ── digit-value: VAL_INT 0..9, or #f (R7RS 6.6) ─────────────────── */
    case VM_CSE_DIGIT_VALUE: {
        int cp = vm_cse_char_cp(args[0]);
        int d = (cp < 0) ? -1 : vm_cse_digit_value(cp);
        *out = (d < 0) ? BOOL_VAL(0) : INT_VAL(d);
        break;
    }

    /* ── char-ready?: #t, as the native lowering answers for every port.
     *    The optional port operand is accepted and ignored, so the
     *    zero-argument call and the one-argument call agree, and no port
     *    payload is dereferenced on a host without one. ─────────────── */
    case VM_CSE_CHAR_READY:
        *out = BOOL_VAL(1);
        break;

    /* ── String comparison, exact and case-folded ───────────────────── */
    case VM_CSE_STRING_LE:
    case VM_CSE_STRING_GE:
    case VM_CSE_STRING_GT:
    case VM_CSE_STRING_CI_LT:
    case VM_CSE_STRING_CI_LE:
    case VM_CSE_STRING_CI_GT:
    case VM_CSE_STRING_CI_GE: {
        VmString* a = vm_value_as_string(vm, args[0]);
        VmString* b = vm_value_as_string(vm, args[1]);
        int fold = (fid >= VM_CSE_STRING_CI_LT && fid <= VM_CSE_STRING_CI_GE);
        int c = vm_cse_string_cmp(a, b, fold);
        int r;
        switch (fid) {
        case VM_CSE_STRING_LE:    r = (c <= 0); break;
        case VM_CSE_STRING_GE:    r = (c >= 0); break;
        case VM_CSE_STRING_GT:    r = (c >  0); break;
        case VM_CSE_STRING_CI_LT: r = (c <  0); break;
        case VM_CSE_STRING_CI_LE: r = (c <= 0); break;
        case VM_CSE_STRING_CI_GT: r = (c >  0); break;
        default:                  r = (c >= 0); break;
        }
        *out = BOOL_VAL(r);
        break;
    }

    /* ── string-foldcase ────────────────────────────────────────────── */
    case VM_CSE_STRING_FOLDCASE: {
        VmString* s = vm_value_as_string(vm, args[0]);
        VmString* r = s ? vm_cse_string_foldcase(rs, s) : NULL;
        *out = r ? vm_cse_string_value(vm, r) : NIL_VAL;
        break;
    }

    /* ── string-contains? — the predicate spelling: #t / #f ─────────── */
    case VM_CSE_STRING_CONTAINS: {
        VmString* s   = vm_value_as_string(vm, args[0]);
        VmString* sub = vm_value_as_string(vm, args[1]);
        *out = BOOL_VAL(s && sub && vm_string_contains(s, sub) >= 0);
        break;
    }

    /* ── string-index — codepoint index of a substring, or -1.
     *    The -1 miss answer is the native one; the integer-or-#f answer at
     *    native call 555 belongs to the other spelling, `string-contains`,
     *    and is left alone. ──────────────────────────────────────────── */
    case VM_CSE_STRING_INDEX: {
        VmString* s   = vm_value_as_string(vm, args[0]);
        VmString* sub = vm_value_as_string(vm, args[1]);
        *out = INT_VAL((s && sub) ? vm_string_contains(s, sub) : -1);
        break;
    }

    /* ── string-set! — in-place, UTF-8 width-changing ────────────────── */
    case VM_CSE_STRING_SET: {
        VmString* s = vm_cse_mutable_string(vm, args[0]);
        int k  = vm_cse_index_arg(args[1], -1);
        int cp = vm_cse_char_cp(args[2]);
        if (s && cp >= 0) (void)vm_cse_string_set(rs, s, k, cp);
        *out = (Value){ .type = VAL_VOID };
        break;
    }

    /* ── string-copy! — `(string-copy! to at from [start [end]])` ────── */
    case VM_CSE_STRING_COPY: {
        VmString* to   = vm_cse_mutable_string(vm, args[0]);
        VmString* from = vm_value_as_string(vm, args[2]);
        if (to && from) {
            int at    = vm_cse_index_arg(args[1], 0);
            int start = vm_cse_index_arg(args[3], 0);
            int end   = vm_cse_index_arg(args[4], from->char_len);
            (void)vm_cse_string_copy_bang(rs, to, at, from, start, end);
        }
        *out = NIL_VAL;   /* the native lowering answers with the null value */
        break;
    }

    /* ── string-port? ───────────────────────────────────────────────── */
    case VM_CSE_STRING_PORT_P: {
        int is_string_port = 0;
        if (args[0].type == VAL_PORT && is_heap_type(vm, args[0], HEAP_PORT)) {
            VmPort* p = (VmPort*)vm->heap.objects[args[0].as.ptr]->opaque.ptr;
            if (p) is_string_port = (p->kind == VM_PORT_STRING);
        }
        *out = BOOL_VAL(is_string_port);
        break;
    }

    /* ── hash-table-ref — `(hash-table-ref ht key [default])`.
     *    The third operand is a DEFAULT VALUE, not a failure thunk: that is
     *    what HashCodegen::hashRef() lowers, and an absent operand defaults
     *    to #f there too. ────────────────────────────────────────────── */
    case VM_CSE_HASH_TABLE_REF: {
        Value dflt = args[2];
        if (vm_native_absent(dflt)) dflt = BOOL_VAL(0);
        if (is_heap_type(vm, args[0], HEAP_HASH)) {
            VmHashTable* ht =
                (VmHashTable*)vm->heap.objects[args[0].as.ptr]->opaque.ptr;
            Value key = args[1];
            Value* found = ht ? (Value*)vm_ht_ref(ht, &key, NULL) : NULL;
            *out = found ? *found : dflt;
        } else {
            *out = dflt;
        }
        break;
    }

    default:
        return 0;
    }

    return 1;
}

/** @brief Largest operand count any Wave 1B native call reads. */
#define VM_CSE_MAX_OPERANDS 5

/**
 * @brief Stack adapter: the form vm_dispatch_native() calls.
 *
 * Pops the declared operand count in call order, runs the pure handler, and
 * pushes its single result. Every Wave 1B op has the same shape, so the
 * stack discipline lives here once instead of in each case.
 *
 * @return 1 when @p fid was handled, 0 when it is not a Wave 1B ID — in
 *         which case nothing was popped or pushed.
 */
static int vm_char_string_ext_dispatch_stack(VM* vm, int fid) {
    int argc = vm_cse_operand_count(fid);
    if (fid < VM_CSE_FID_BASE || fid >= VM_CSE_FID_LIMIT || argc == 0) return 0;

    Value args[VM_CSE_MAX_OPERANDS];
    for (int i = 0; i < VM_CSE_MAX_OPERANDS; i++) args[i] = (Value){ .type = VAL_VOID };
    /* Operands were pushed left to right, so they pop right to left. */
    for (int i = argc - 1; i >= 0; i--) args[i] = vm_pop(vm);

    Value out = NIL_VAL;
    if (!vm_char_string_ext_dispatch(vm, fid, argc, args, &out)) {
        vm_push(vm, NIL_VAL);
        return 1;
    }
    vm_push(vm, out);
    return 1;
}

#endif /* !VM_CHAR_STRING_EXT_PURE_HELPERS_ONLY */

#endif /* VM_CHAR_STRING_EXT_C_INCLUDED */
