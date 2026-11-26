/**
 * @file test_wave1b_asan.c
 * @brief Standalone memory- and alignment-verification harness for the
 *        ADR-0064 Wave 1B character / string / hash-table handlers.
 *
 * Build and run (ASan + UBSan, warnings as errors):
 *
 *   clang -std=c11 -Wall -Wextra -Werror -fsanitize=address,undefined -g \
 *         -Iinc -Ilib/backend tests/vm_parity/test_wave1b_asan.c \
 *         -o <outdir>/test_wave1b_asan && <outdir>/test_wave1b_asan
 *
 * The harness builds Layer A of lib/backend/vm_char_string_ext.c — the pure
 * codepoint and `VmString` layer — against the real OALR arena and the real
 * UTF-8 codec, with no interpreter in the picture. That is the whole point:
 * every byte these handlers read or write is reachable from here, so ASan
 * sees the actual allocations and UBSan sees the actual index arithmetic.
 *
 * Coverage: ASCII, 2-, 3- and 4-byte UTF-8, the empty string, lone
 * surrogates and other invalid codepoints, out-of-range indices, the
 * overlapping `string-copy!`, `string-set!` across every encoded-width
 * change, and the `hash-table-ref` lookup contract including its default.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Layer A only: no Value, no heap table, no interpreter. */
#define VM_CHAR_STRING_EXT_PURE_HELPERS_ONLY 1

/* vm_string.c and vm_hashtable.c are inclusion units built for the VM's one
 * translation unit, so each carries helpers this harness does not call. */
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "vm_string.c"
#include "vm_hashtable.c"
#include "vm_char_string_ext.c"
#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

static int g_checks;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

/* Codepoints exercising each UTF-8 encoded width. */
#define CP_2BYTE 0x00E9    /* e with acute  */
#define CP_3BYTE 0x4E2D    /* CJK ideograph */
#define CP_4BYTE 0x1F600   /* emoji         */

/** @brief Byte-exact comparison of a string payload against a literal. */
static int payload_is(const VmString* s, const char* bytes, int byte_len, int char_len) {
    if (!s || s->byte_len != byte_len || s->char_len != char_len) return 0;
    if (memcmp(s->data, bytes, (size_t)byte_len) != 0) return 0;
    return s->data[byte_len] == '\0';   /* the NUL terminator must survive */
}

/* ── char-foldcase / digit-value / char-ci comparison ─────────────────── */

static int test_char_ops(void) {
    CHECK(vm_cse_char_foldcase('A') == 'a');
    CHECK(vm_cse_char_foldcase('Z') == 'z');
    CHECK(vm_cse_char_foldcase('a') == 'a');
    CHECK(vm_cse_char_foldcase('@') == '@');          /* just below 'A' */
    CHECK(vm_cse_char_foldcase('[') == '[');          /* just above 'Z' */
    CHECK(vm_cse_char_foldcase('0') == '0');
    /* A codepoint above ASCII folds to itself: no truncation aliasing. */
    CHECK(vm_cse_char_foldcase(CP_2BYTE) == CP_2BYTE);
    CHECK(vm_cse_char_foldcase(0x0141) == 0x0141);    /* low byte is 'A' */
    CHECK(vm_cse_char_foldcase(CP_4BYTE) == CP_4BYTE);
    CHECK(vm_cse_char_foldcase(0) == 0);
    CHECK(vm_cse_char_foldcase(0x10FFFF) == 0x10FFFF);

    CHECK(vm_cse_digit_value('0') == 0);
    CHECK(vm_cse_digit_value('9') == 9);
    CHECK(vm_cse_digit_value('5') == 5);
    CHECK(vm_cse_digit_value('/') < 0);               /* just below '0' */
    CHECK(vm_cse_digit_value(':') < 0);               /* just above '9' */
    CHECK(vm_cse_digit_value('a') < 0);
    CHECK(vm_cse_digit_value(0xFF10) < 0);            /* low byte is 0x10 */
    CHECK(vm_cse_digit_value(CP_4BYTE) < 0);

    CHECK(vm_cse_char_ci_cmp('A', 'a') == 0);
    CHECK(vm_cse_char_ci_cmp('a', 'B') < 0);
    CHECK(vm_cse_char_ci_cmp('B', 'a') > 0);
    CHECK(vm_cse_char_ci_cmp(CP_3BYTE, CP_3BYTE) == 0);
    CHECK(vm_cse_char_ci_cmp('a', CP_2BYTE) < 0);
    return 0;
}

/* ── string comparison, exact and case-folded ─────────────────────────── */

static int test_string_cmp(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* abc   = vm_string_from_cstr(&rs, "abc");
    VmString* abd   = vm_string_from_cstr(&rs, "abd");
    VmString* ab    = vm_string_from_cstr(&rs, "ab");
    VmString* ABC   = vm_string_from_cstr(&rs, "ABC");
    VmString* empty = vm_string_from_cstr(&rs, "");
    VmString* cjk   = vm_string_from_cstr(&rs, "\xE4\xB8\xAD");       /* U+4E2D */
    VmString* emoji = vm_string_from_cstr(&rs, "\xF0\x9F\x98\x80");   /* U+1F600 */
    CHECK(abc && abd && ab && ABC && empty && cjk && emoji);

    CHECK(vm_cse_string_cmp(abc, abc, 0) == 0);
    CHECK(vm_cse_string_cmp(abc, abd, 0) < 0);
    CHECK(vm_cse_string_cmp(abd, abc, 0) > 0);
    CHECK(vm_cse_string_cmp(ab,  abc, 0) < 0);        /* proper prefix */
    CHECK(vm_cse_string_cmp(abc, ab,  0) > 0);
    CHECK(vm_cse_string_cmp(empty, abc, 0) < 0);
    CHECK(vm_cse_string_cmp(empty, empty, 0) == 0);

    /* Exact comparison keeps case apart; the folded one does not. */
    CHECK(vm_cse_string_cmp(ABC, abc, 0) < 0);
    CHECK(vm_cse_string_cmp(ABC, abc, 1) == 0);
    CHECK(vm_cse_string_cmp(ABC, abd, 1) < 0);
    CHECK(vm_cse_string_cmp(abd, ABC, 1) > 0);

    /* Multi-byte codepoints order by codepoint, not by byte width. */
    CHECK(vm_cse_string_cmp(abc, cjk, 0) < 0);
    CHECK(vm_cse_string_cmp(cjk, emoji, 0) < 0);
    CHECK(vm_cse_string_cmp(emoji, cjk, 0) > 0);
    CHECK(vm_cse_string_cmp(cjk, cjk, 1) == 0);

    /* A missing payload is ordered, never dereferenced. */
    CHECK(vm_cse_string_cmp(NULL, NULL, 0) == 0);
    CHECK(vm_cse_string_cmp(NULL, abc, 0) < 0);
    CHECK(vm_cse_string_cmp(abc, NULL, 1) > 0);

    vm_region_stack_destroy(&rs);
    return 0;
}

/* ── string-foldcase across every encoded width ───────────────────────── */

static int test_string_foldcase(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* mixed = vm_string_from_cstr(&rs, "AbC-Z9");
    VmString* r = vm_cse_string_foldcase(&rs, mixed);
    CHECK(payload_is(r, "abc-z9", 6, 6));

    VmString* empty = vm_string_from_cstr(&rs, "");
    VmString* re = vm_cse_string_foldcase(&rs, empty);
    CHECK(payload_is(re, "", 0, 0));

    /* ASCII + 2-byte + 3-byte + 4-byte in one string: the non-ASCII
     * codepoints pass through byte-for-byte and the widths stay correct. */
    VmString* wide = vm_string_from_cstr(&rs,
        "A\xC3\xA9" "B\xE4\xB8\xAD" "C\xF0\x9F\x98\x80");
    VmString* rw = vm_cse_string_foldcase(&rs, wide);
    CHECK(payload_is(rw, "a\xC3\xA9" "b\xE4\xB8\xAD" "c\xF0\x9F\x98\x80", 12, 6));

    /* An invalid encoding decodes to U+FFFD and re-encodes as 3 bytes, which
     * grows the output: the counted first pass must have reserved for it. */
    VmString* bad = vm_string_new(&rs, "A\xFF" "B", 3);
    VmString* rb = vm_cse_string_foldcase(&rs, bad);
    CHECK(payload_is(rb, "a\xEF\xBF\xBD" "b", 5, 3));

    CHECK(vm_cse_string_foldcase(&rs, NULL) == NULL);

    vm_region_stack_destroy(&rs);
    return 0;
}

/* ── string-set!: in-place, every encoded-width change ────────────────── */

static int test_string_set(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* Same width: ASCII over ASCII. */
    VmString* s = vm_string_from_cstr(&rs, "abc");
    CHECK(vm_cse_string_set(&rs, s, 1, 'X') == 1);
    CHECK(payload_is(s, "aXc", 3, 3));

    /* Growth: 1 byte becomes 4. The codepoint count is unchanged. */
    CHECK(vm_cse_string_set(&rs, s, 2, CP_4BYTE) == 1);
    CHECK(payload_is(s, "aX\xF0\x9F\x98\x80", 6, 3));

    /* Shrink: 4 bytes become 1. */
    CHECK(vm_cse_string_set(&rs, s, 2, 'z') == 1);
    CHECK(payload_is(s, "aXz", 3, 3));

    /* Growth at index 0 and at the last index, over multi-byte neighbours. */
    VmString* m = vm_string_from_cstr(&rs, "\xC3\xA9\x41\xE4\xB8\xAD");  /* e' A CJK */
    CHECK(m->char_len == 3 && m->byte_len == 6);
    CHECK(vm_cse_string_set(&rs, m, 0, 'q') == 1);
    CHECK(payload_is(m, "q\x41\xE4\xB8\xAD", 5, 3));
    CHECK(vm_cse_string_set(&rs, m, 2, CP_2BYTE) == 1);
    CHECK(payload_is(m, "q\x41\xC3\xA9", 4, 3));

    /* A lone surrogate and an over-range codepoint become U+FFFD, exactly
     * as the shared codec defines, and never a malformed byte sequence. */
    VmString* u = vm_string_from_cstr(&rs, "ab");
    CHECK(vm_cse_string_set(&rs, u, 0, 0xD800) == 1);
    CHECK(payload_is(u, "\xEF\xBF\xBD" "b", 4, 2));
    CHECK(vm_cse_string_set(&rs, u, 1, 0x110000) == 1);
    CHECK(payload_is(u, "\xEF\xBF\xBD\xEF\xBF\xBD", 6, 2));

    /* Out-of-range indices are refused and change nothing. */
    VmString* g = vm_string_from_cstr(&rs, "abc");
    CHECK(vm_cse_string_set(&rs, g, -1, 'z') == 0);
    CHECK(vm_cse_string_set(&rs, g, 3, 'z') == 0);
    CHECK(vm_cse_string_set(&rs, g, 1 << 30, 'z') == 0);
    CHECK(payload_is(g, "abc", 3, 3));

    /* The empty string has no settable index at all. */
    VmString* e = vm_string_from_cstr(&rs, "");
    CHECK(vm_cse_string_set(&rs, e, 0, 'a') == 0);
    CHECK(payload_is(e, "", 0, 0));

    CHECK(vm_cse_string_set(&rs, NULL, 0, 'a') == 0);

    vm_region_stack_destroy(&rs);
    return 0;
}

/* ── string-copy!: bounds, widths, and the overlapping case ───────────── */

static int test_string_copy_bang(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* Whole-source copy. */
    VmString* to   = vm_string_from_cstr(&rs, "abcde");
    VmString* from = vm_string_from_cstr(&rs, "XY");
    CHECK(vm_cse_string_copy_bang(&rs, to, 1, from, 0, from->char_len) == 1);
    CHECK(payload_is(to, "aXYde", 5, 5));

    /* A sub-range of the source. */
    VmString* src = vm_string_from_cstr(&rs, "0123456789");
    CHECK(vm_cse_string_copy_bang(&rs, to, 0, src, 3, 6) == 1);
    CHECK(payload_is(to, "345de", 5, 5));

    /* Zero-length copies are legal no-ops at every legal `at`. */
    CHECK(vm_cse_string_copy_bang(&rs, to, 0, src, 4, 4) == 1);
    CHECK(vm_cse_string_copy_bang(&rs, to, to->char_len, src, 0, 0) == 1);
    CHECK(payload_is(to, "345de", 5, 5));

    /* Multi-byte source into an ASCII destination: the byte extent grows
     * while the codepoint count is preserved. */
    VmString* wide_src = vm_string_from_cstr(&rs, "\xE4\xB8\xAD\xF0\x9F\x98\x80");
    VmString* ascii_to = vm_string_from_cstr(&rs, "abcd");
    CHECK(vm_cse_string_copy_bang(&rs, ascii_to, 1, wide_src, 0, 2) == 1);
    CHECK(payload_is(ascii_to, "a\xE4\xB8\xAD\xF0\x9F\x98\x80" "d", 9, 4));

    /* And back the other way: multi-byte destination overwritten by ASCII. */
    VmString* narrow = vm_string_from_cstr(&rs, "zz");
    CHECK(vm_cse_string_copy_bang(&rs, ascii_to, 1, narrow, 0, 2) == 1);
    CHECK(payload_is(ascii_to, "azz" "d", 4, 4));

    /* Overlapping: to == from, forward and backward, including the
     * multi-byte case where the moved run changes the byte layout. */
    VmString* ov = vm_string_from_cstr(&rs, "abcdef");
    CHECK(vm_cse_string_copy_bang(&rs, ov, 0, ov, 2, 5) == 1);
    CHECK(payload_is(ov, "cdedef", 6, 6));

    VmString* ov2 = vm_string_from_cstr(&rs, "abcdef");
    CHECK(vm_cse_string_copy_bang(&rs, ov2, 3, ov2, 0, 3) == 1);
    CHECK(payload_is(ov2, "abcabc", 6, 6));

    VmString* ov3 = vm_string_from_cstr(&rs, "\xC3\xA9" "ab\xE4\xB8\xAD");  /* e' a b CJK */
    CHECK(ov3->char_len == 4);
    CHECK(vm_cse_string_copy_bang(&rs, ov3, 1, ov3, 0, 2) == 1);
    CHECK(payload_is(ov3, "\xC3\xA9\xC3\xA9" "a\xE4\xB8\xAD", 8, 4));

    /* Copying a string onto itself at the same place is the identity. */
    VmString* ov4 = vm_string_from_cstr(&rs, "\xF0\x9F\x98\x80" "q");
    CHECK(vm_cse_string_copy_bang(&rs, ov4, 0, ov4, 0, 2) == 1);
    CHECK(payload_is(ov4, "\xF0\x9F\x98\x80" "q", 5, 2));

    /* Every range violation is refused with the destination untouched. */
    VmString* keep = vm_string_from_cstr(&rs, "abcde");
    CHECK(vm_cse_string_copy_bang(&rs, keep, -1, src, 0, 1) == 0);
    CHECK(vm_cse_string_copy_bang(&rs, keep, 6, src, 0, 1) == 0);
    CHECK(vm_cse_string_copy_bang(&rs, keep, 0, src, -1, 2) == 0);
    CHECK(vm_cse_string_copy_bang(&rs, keep, 0, src, 2, 1) == 0);   /* end < start */
    CHECK(vm_cse_string_copy_bang(&rs, keep, 0, src, 0, 11) == 0);  /* end > source */
    CHECK(vm_cse_string_copy_bang(&rs, keep, 3, src, 0, 3) == 0);   /* run overruns */
    CHECK(vm_cse_string_copy_bang(&rs, keep, 0, src, 0, 1 << 30) == 0);
    CHECK(vm_cse_string_copy_bang(&rs, NULL, 0, src, 0, 1) == 0);
    CHECK(vm_cse_string_copy_bang(&rs, keep, 0, NULL, 0, 1) == 0);
    CHECK(payload_is(keep, "abcde", 5, 5));

    /* The empty destination accepts only the empty copy. */
    VmString* ze = vm_string_from_cstr(&rs, "");
    CHECK(vm_cse_string_copy_bang(&rs, ze, 0, src, 0, 0) == 1);
    CHECK(vm_cse_string_copy_bang(&rs, ze, 0, src, 0, 1) == 0);
    CHECK(payload_is(ze, "", 0, 0));

    vm_region_stack_destroy(&rs);
    return 0;
}

/* ── string-contains? / string-index ──────────────────────────────────── */

static int test_string_search(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* hay = vm_string_from_cstr(&rs, "a\xE4\xB8\xAD" "bc");  /* a CJK b c */
    CHECK(hay->char_len == 4 && hay->byte_len == 6);

    VmString* cjk   = vm_string_from_cstr(&rs, "\xE4\xB8\xAD");
    VmString* bc    = vm_string_from_cstr(&rs, "bc");
    VmString* miss  = vm_string_from_cstr(&rs, "zz");
    VmString* empty = vm_string_from_cstr(&rs, "");

    /* The answer is a CODEPOINT index, not a byte offset. */
    CHECK(vm_string_contains(hay, cjk) == 1);
    CHECK(vm_string_contains(hay, bc) == 2);
    CHECK(vm_string_contains(hay, miss) < 0);
    CHECK(vm_string_contains(hay, empty) == 0);
    CHECK(vm_string_contains(empty, hay) < 0);
    CHECK(vm_string_contains(hay, NULL) < 0);
    CHECK(vm_string_contains(NULL, hay) < 0);

    vm_region_stack_destroy(&rs);
    return 0;
}

/* ── hash-table-ref: the lookup contract and its default ──────────────── */

static int test_hash_table_ref(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmHashTable* ht = vm_ht_make(&rs);
    CHECK(ht != NULL);

    /* Keys and values are arena-owned boxes, as the VM's Scheme tables hold
     * them; the table itself only ever stores the opaque pointers. */
    enum { N = 40 };
    int* keys = (int*)vm_alloc(&rs, sizeof(int) * N);
    int* vals = (int*)vm_alloc(&rs, sizeof(int) * N);
    CHECK(keys != NULL && vals != NULL);
    for (int i = 0; i < N; i++) {
        keys[i] = i;
        vals[i] = i * 7;
        vm_ht_set(&rs, ht, &keys[i], &vals[i]);
    }

    /* Every stored key resolves to its own value, across the rehash the
     * 75% load factor forces partway through the insertions above. */
    for (int i = 0; i < N; i++) {
        void* got = vm_ht_ref(ht, &keys[i], NULL);
        CHECK(got == &vals[i]);
        CHECK(*(int*)got == i * 7);
    }

    /* A missing key answers the supplied default, and the two-argument
     * form's NULL "no default" is distinguishable from a stored value. */
    int absent_key = 12345;
    int fallback = -1;
    CHECK(vm_ht_ref(ht, &absent_key, NULL) == NULL);
    CHECK(vm_ht_ref(ht, &absent_key, &fallback) == &fallback);

    /* A deleted key reverts to the default without disturbing its probe
     * neighbours. */
    vm_ht_remove(ht, &keys[5]);
    CHECK(vm_ht_ref(ht, &keys[5], &fallback) == &fallback);
    CHECK(vm_ht_ref(ht, &keys[6], NULL) == &vals[6]);

    /* A missing table is answered, never dereferenced. */
    CHECK(vm_ht_ref(NULL, &keys[0], &fallback) == &fallback);

    vm_region_stack_destroy(&rs);
    return 0;
}

int main(void) {
    printf("Wave 1B char/string/hash-table memory verification\n");

    struct { const char* name; int (*fn)(void); } suite[] = {
        { "char-foldcase / digit-value / char-ci",  test_char_ops },
        { "string comparison (exact and folded)",   test_string_cmp },
        { "string-foldcase",                        test_string_foldcase },
        { "string-set!",                            test_string_set },
        { "string-copy!",                           test_string_copy_bang },
        { "string-contains? / string-index",        test_string_search },
        { "hash-table-ref",                         test_hash_table_ref },
    };
    const int count = (int)(sizeof(suite) / sizeof(suite[0]));

    for (int i = 0; i < count; i++) {
        int before = g_checks;
        if (suite[i].fn() != 0) {
            printf("  [%d/%d] %s: FAILED\n", i + 1, count, suite[i].name);
            return 1;
        }
        printf("  [%d/%d] %s: %d assertions passed\n",
               i + 1, count, suite[i].name, g_checks - before);
    }

    printf("ALL %d SUITES PASSED (%d assertions)\n", count, g_checks);
    return 0;
}
