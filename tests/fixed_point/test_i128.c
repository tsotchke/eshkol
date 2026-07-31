/* test_i128.c — Phase E1.1 / E1.3: i128 arithmetic, shifts, compares,
 * marshalling, printing/parsing, and overflow edges. */
#include "eshkol_fixed_point.h"
#include "test_harness.h"
#include <string.h>

static esk_i128 mkbig(const char *s) { esk_i128 v; bool o; esk_i128_from_string(s, &v, &o); return v; }

int main(void) {
    /* --- construction / marshalling round-trip --- */
    {
        esk_i128 v = esk_i128_from_parts(0x0123456789ABCDEFull, 0xFEDCBA9876543210ull);
        esk_i128_abi a = esk_i128_to_abi(v);
        CHECK(a.hi == 0x0123456789ABCDEFull && a.lo == 0xFEDCBA9876543210ull, "i128 -> abi limb order (lo,hi)");
        CHECK_EQ_I128(esk_i128_from_abi(a), v, "abi round-trip identity");
    }
    /* negative marshalling */
    {
        esk_i128 v = esk_i128_from_i64(-1);
        esk_i128_abi a = esk_i128_to_abi(v);
        CHECK(a.hi == 0xFFFFFFFFFFFFFFFFull && a.lo == 0xFFFFFFFFFFFFFFFFull, "-1 marshals to all-ones");
        CHECK_EQ_I128(esk_i128_from_abi(a), v, "-1 abi round-trip");
    }

    /* --- arithmetic --- */
    CHECK_EQ_I128(esk_i128_add(mkbig("170141183460469231731687303715884105726"), esk_i128_from_i64(1)),
                  ESK_I128_MAX, "(MAX-1)+1 == MAX");
    CHECK_EQ_I128(esk_i128_mul(esk_i128_from_i64(1000000000000LL), esk_i128_from_i64(1000000000000LL)),
                  mkbig("1000000000000000000000000"), "1e12 * 1e12 == 1e24 (exceeds i64)");
    CHECK_EQ_I128(esk_i128_sub(esk_i128_from_i64(5), esk_i128_from_i64(8)), esk_i128_from_i64(-3), "5-8 == -3");
    CHECK_EQ_I128(esk_i128_neg(ESK_I128_MAX), esk_i128_add(ESK_I128_MIN, esk_i128_from_i64(1)), "-MAX == MIN+1");

    /* --- widening multiply exactness --- */
    {
        int64_t a = 9223372036854775807LL;    /* i64 max */
        esk_i128 w = esk_i128_widen_mul_i64(a, a);
        CHECK_EQ_I128(w, mkbig("85070591730234615847396907784232501249"), "widen_mul(i64max, i64max) exact");
        CHECK_EQ_I128(esk_i128_widen_mul_i64(-a, a), esk_i128_neg(w), "widen_mul sign symmetry");
    }

    /* --- shifts --- */
    CHECK_EQ_I128(esk_i128_shl(esk_i128_from_i64(1), 100), mkbig("1267650600228229401496703205376"), "1 << 100 == 2^100");
    CHECK_EQ_I128(esk_i128_asr(esk_i128_from_i64(-8), 1), esk_i128_from_i64(-4), "asr(-8,1) == -4 (sign-preserving)");
    CHECK_EQ_I128(esk_i128_lsr(esk_i128_from_i64(-1), 127), esk_i128_from_i64(1), "lsr(-1,127) == 1 (logical)");

    /* --- compares --- */
    CHECK(esk_i128_cmp(ESK_I128_MIN, ESK_I128_MAX) < 0, "MIN < MAX");
    CHECK(esk_i128_cmp(ESK_I128_MAX, ESK_I128_MAX) == 0, "MAX == MAX");
    CHECK(esk_i128_cmp(esk_i128_from_i64(1), esk_i128_from_i64(-1)) > 0, "1 > -1");

    /* --- overflow edges --- */
    {
        esk_i128 out; bool o;
        o = esk_i128_add_overflow(ESK_I128_MAX, esk_i128_from_i64(1), &out);
        CHECK(o, "MAX+1 flags overflow");
        CHECK_EQ_I128(out, ESK_I128_MIN, "MAX+1 wraps to MIN");
        o = esk_i128_sub_overflow(ESK_I128_MIN, esk_i128_from_i64(1), &out);
        CHECK(o, "MIN-1 flags overflow");
        CHECK_EQ_I128(out, ESK_I128_MAX, "MIN-1 wraps to MAX");
        o = esk_i128_add_overflow(esk_i128_from_i64(100), esk_i128_from_i64(200), &out);
        CHECK(!o && out == esk_i128_from_i64(300), "100+200 no overflow");
        o = esk_i128_mul_overflow(ESK_I128_MIN, esk_i128_from_i64(-1), &out);
        CHECK(o, "MIN * -1 flags overflow (no +MIN representable)");
        o = esk_i128_mul_overflow(mkbig("18446744073709551616"), mkbig("18446744073709551616"), &out); /* 2^64 * 2^64 = 2^128 */
        CHECK(o, "2^64 * 2^64 overflows i128");
    }

    /* --- printing / parsing round-trip --- */
    {
        const char *cases[] = {
            "0", "1", "-1", "9223372036854775807", "-9223372036854775808",
            "170141183460469231731687303715884105727",   /* i128 max */
            "-170141183460469231731687303715884105728",  /* i128 min */
            "123456789012345678901234567890123456789"
        };
        for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
            esk_i128 v; bool o; bool ok = esk_i128_from_string(cases[i], &v, &o);
            char buf[64]; esk_i128_to_string(v, buf, sizeof buf);
            char lbl[160]; snprintf(lbl, sizeof lbl, "parse/print round-trip %s", cases[i]);
            CHECK(ok && !o && strcmp(buf, cases[i]) == 0, lbl);
        }
        /* overflow on parse */
        esk_i128 v; bool o;
        esk_i128_from_string("170141183460469231731687303715884105728", &v, &o); /* max+1 */
        CHECK(o, "parse i128max+1 flags overflow");
        esk_i128_from_string("999999999999999999999999999999999999999999", &v, &o);
        CHECK(o, "parse huge number flags overflow");
    }

    return harness_summary("test_i128");
}
