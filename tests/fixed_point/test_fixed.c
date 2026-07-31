/* test_fixed.c — Phase E1.2 / E1.3: parametric fixed<W,F> property tests.
 *   - exact add/sub, add ASSOCIATIVITY under ring (wrap) arithmetic
 *   - widening-multiply identities
 *   - explicit rounding-mode conformance (truncate | nearest-even | stochastic)
 *   - saturating vs wrapping overflow
 *   - conversions to/from int/double with exactness flags
 */
#include "eshkol_fixed_point.h"
#include "test_harness.h"
#include <string.h>

static esk_fixed FX(esk_i128 raw, uint8_t W, uint8_t F) { esk_fixed f = {raw, W, F}; return f; }

int main(void) {
    esk_rng rng; esk_rng_seed(&rng, 0xC0FFEEu);

    /* ===================== exact add / sub ===================== */
    {
        esk_fixed a = FX(3 << 4, 32, 4);   /* 3.0  in Q4  */
        esk_fixed b = FX(5 << 4, 32, 4);   /* 5.0        */
        bool o;
        CHECK(esk_fixed_add(a, b, ESK_OF_WRAP, &o).raw == (8 << 4) && !o, "3.0 + 5.0 == 8.0 (Q4)");
        CHECK(esk_fixed_sub(a, b, ESK_OF_WRAP, &o).raw == -(2 << 4) && !o, "3.0 - 5.0 == -2.0 (Q4)");
    }

    /* ============ add associativity (THE determinism payoff) ============ */
    /* Ring (wrap) addition is associative for ANY inputs — bit-for-bit. */
    {
        const uint8_t Ws[3] = {32, 64, 128};
        const uint8_t Fs[3] = {8, 16, 32};
        int fails = 0;
        for (int wi = 0; wi < 3; wi++) {
            for (int trial = 0; trial < 20000; trial++) {
                uint8_t W = Ws[wi], F = Fs[wi];
                esk_i128 lo = esk_fixed_wmin(W), hi = esk_fixed_wmax(W);
                esk_u128 span = (esk_u128)(hi - lo);
                esk_i128 ra = lo + (esk_i128)(esk_rng_next_u64(&rng) % (span ? span : 1));
                esk_i128 rb = lo + (esk_i128)(esk_rng_next_u64(&rng) % (span ? span : 1));
                esk_i128 rc = lo + (esk_i128)(esk_rng_next_u64(&rng) % (span ? span : 1));
                esk_fixed a = FX(ra, W, F), b = FX(rb, W, F), c = FX(rc, W, F);
                bool o;
                esk_fixed lft = esk_fixed_add(esk_fixed_add(a, b, ESK_OF_WRAP, &o), c, ESK_OF_WRAP, &o);
                esk_fixed rgt = esk_fixed_add(a, esk_fixed_add(b, c, ESK_OF_WRAP, &o), ESK_OF_WRAP, &o);
                if (lft.raw != rgt.raw) fails++;
            }
        }
        CHECK(fails == 0, "add associativity (a+b)+c == a+(b+c) over 60000 random triples, all W");
    }

    /* ===================== widening-multiply identities ===================== */
    {
        esk_fixed one = esk_fixed_from_i64(1, 32, 8, ESK_OF_WRAP, NULL);   /* 1.0 in Q8 */
        bool o;
        int idfail = 0, comfail = 0;
        for (int t = 0; t < 5000; t++) {
            esk_i128 r = (esk_i128)(int32_t)(esk_rng_next_u64(&rng));       /* fits Q8/32 range mostly */
            r = r % (esk_fixed_wmax(32));
            esk_fixed x = FX(r, 32, 8);
            esk_fixed y = FX((esk_i128)(int16_t)esk_rng_next_u64(&rng), 32, 8);
            esk_fixed x1 = esk_fixed_mul(x, one, ESK_ROUND_TRUNCATE, ESK_OF_WRAP, NULL, &o);
            if (x1.raw != x.raw) idfail++;
            esk_fixed xy = esk_fixed_mul(x, y, ESK_ROUND_NEAREST_EVEN, ESK_OF_SATURATE, NULL, &o);
            esk_fixed yx = esk_fixed_mul(y, x, ESK_ROUND_NEAREST_EVEN, ESK_OF_SATURATE, NULL, &o);
            if (xy.raw != yx.raw) comfail++;
        }
        CHECK(idfail == 0, "widening-mul identity  x * 1.0 == x  (5000 random x)");
        CHECK(comfail == 0, "widening-mul commutativity  x*y == y*x  (5000 random pairs)");
    }
    /* fixed<128,F> multiply uses the full 256-bit product path — spot-check exactness */
    {
        bool o;
        esk_fixed a = esk_fixed_from_i64(1000000, 128, 32, ESK_OF_WRAP, NULL);  /* 1e6 in Q32 */
        esk_fixed b = esk_fixed_from_i64(1000000, 128, 32, ESK_OF_WRAP, NULL);
        esk_fixed p = esk_fixed_mul(a, b, ESK_ROUND_TRUNCATE, ESK_OF_SATURATE, NULL, &o);
        /* 1e6 * 1e6 = 1e12, stored in Q32 -> raw = 1e12 * 2^32 */
        esk_i128 want = (esk_i128)1000000000000LL << 32;
        CHECK(p.raw == want && !o, "fixed<128,32> 1e6*1e6 == 1e12 (256-bit product path, exact)");
    }

    /* ===================== rounding-mode conformance ===================== */
    /* Q4 (F=4, 2^F=16, half=8). raw product = a.raw*b.raw, then >>4 with rounding. */
    {
        /* product raw = 9 -> 9>>4 = 0 rem 9 (>half) : truncate 0, nearest 1, stoch in {0,1} */
        esk_fixed a = FX(3, 32, 4), b = FX(3, 32, 4);  /* 3*3 = 9 */
        bool o;
        CHECK(esk_fixed_mul(a, b, ESK_ROUND_TRUNCATE,     ESK_OF_WRAP, NULL, &o).raw == 0, "round: rem>half truncate -> 0");
        CHECK(esk_fixed_mul(a, b, ESK_ROUND_NEAREST_EVEN, ESK_OF_WRAP, NULL, &o).raw == 1, "round: rem>half nearest  -> 1");
    }
    {
        /* ties-to-even: product raw = 8 -> q=0 rem 8 (==half), q even -> 0 */
        esk_fixed a = FX(8, 32, 4), b = FX(1, 32, 4);  bool o;
        CHECK(esk_fixed_mul(a, b, ESK_ROUND_NEAREST_EVEN, ESK_OF_WRAP, NULL, &o).raw == 0, "round: tie, q even (0) -> 0");
        /* product raw = 24 -> q=1 rem 8 (==half), q odd -> 2 */
        esk_fixed c = FX(6, 32, 4), d = FX(4, 32, 4);
        CHECK(esk_fixed_mul(c, d, ESK_ROUND_NEAREST_EVEN, ESK_OF_WRAP, NULL, &o).raw == 2, "round: tie, q odd (1) -> 2 (to even)");
        CHECK(esk_fixed_mul(c, d, ESK_ROUND_TRUNCATE,     ESK_OF_WRAP, NULL, &o).raw == 1, "round: tie truncate -> 1");
    }
    /* stochastic reproducibility + bounds + mean */
    {
        esk_fixed a = FX(3, 32, 4), b = FX(3, 32, 4);  /* frac = 9/16 */
        esk_rng r1, r2; esk_rng_seed(&r1, 42); esk_rng_seed(&r2, 42);
        bool o; int repro = 1, inbounds = 1; long sum = 0; const int N = 100000;
        for (int i = 0; i < N; i++) {
            esk_i128 v1 = esk_fixed_mul(a, b, ESK_ROUND_STOCHASTIC, ESK_OF_WRAP, &r1, &o).raw;
            esk_i128 v2 = esk_fixed_mul(a, b, ESK_ROUND_STOCHASTIC, ESK_OF_WRAP, &r2, &o).raw;
            if (v1 != v2) repro = 0;
            if (v1 != 0 && v1 != 1) inbounds = 0;
            sum += (long)v1;
        }
        CHECK(repro, "stochastic rounding: same seed -> byte-identical stream");
        CHECK(inbounds, "stochastic rounding stays within {floor, floor+1}");
        double mean = (double)sum / N;   /* should approach 9/16 = 0.5625 */
        CHECK(mean > 0.55 && mean < 0.575, "stochastic rounding mean ~= true frac 0.5625 (unbiased)");
    }

    /* ===================== saturate vs wrap ===================== */
    {
        esk_fixed mx = FX(esk_fixed_wmax(32), 32, 0);
        esk_fixed one = FX(1, 32, 0);
        bool o;
        esk_fixed sat = esk_fixed_add(mx, one, ESK_OF_SATURATE, &o);
        CHECK(sat.raw == esk_fixed_wmax(32) && o, "saturate: wmax + 1 clamps to wmax (overflow flagged)");
        esk_fixed wrp = esk_fixed_add(mx, one, ESK_OF_WRAP, &o);
        CHECK(wrp.raw == esk_fixed_wmin(32) && o, "wrap: wmax + 1 wraps to wmin");
    }
    {
        /* W=128 saturation */
        esk_fixed mx = FX(esk_fixed_wmax(128), 128, 0);
        bool o;
        esk_fixed sat = esk_fixed_add(mx, FX(1,128,0), ESK_OF_SATURATE, &o);
        CHECK(sat.raw == ESK_I128_MAX && o, "saturate W=128: I128_MAX + 1 clamps");
    }

    /* ===================== conversions + exactness ===================== */
    {
        bool ex;
        esk_fixed f = esk_fixed_from_double(0.5, 32, 8, ESK_ROUND_NEAREST_EVEN, ESK_OF_SATURATE, NULL, &ex);
        CHECK(f.raw == (1 << 7) && ex, "from_double 0.5 -> Q8 raw 128, exact");
        (void)esk_fixed_from_double(0.1, 32, 8, ESK_ROUND_NEAREST_EVEN, ESK_OF_SATURATE, NULL, &ex);
        CHECK(!ex, "from_double 0.1 -> Q8 inexact (0.1 not dyadic)");
        double back = esk_fixed_to_double(f, &ex);
        CHECK(back == 0.5 && ex, "to_double(0.5 fixed) == 0.5 exact");
        bool exi; int64_t iv = esk_fixed_to_i64(esk_fixed_from_i64(-7, 64, 16, ESK_OF_WRAP, NULL), &exi);
        CHECK(iv == -7 && exi, "i64 round-trip -7 exact");
        int64_t iv2 = esk_fixed_to_i64(f, &exi);
        CHECK(iv2 == 0 && !exi, "to_i64(0.5) truncates to 0, inexact");
    }
    /* requantize Q16 -> Q8 with rounding */
    {
        bool ex;
        esk_fixed q16 = esk_fixed_from_double(1.5, 32, 16, ESK_ROUND_NEAREST_EVEN, ESK_OF_SATURATE, NULL, &ex);
        esk_fixed q8  = esk_fixed_convert(q16, 32, 8, ESK_ROUND_NEAREST_EVEN, ESK_OF_SATURATE, NULL, &ex);
        CHECK(q8.raw == (esk_i128)(1.5 * 256) && ex, "convert Q16->Q8 of 1.5 exact (raw 384)");
    }

    /* ===================== division (documented rare path) ===================== */
    {
        bool o; esk_fixed q;
        esk_fixed six = esk_fixed_from_i64(6, 32, 8, ESK_OF_WRAP, NULL);
        esk_fixed two = esk_fixed_from_i64(2, 32, 8, ESK_OF_WRAP, NULL);
        esk_fixed sev = esk_fixed_from_i64(7, 32, 8, ESK_OF_WRAP, NULL);
        CHECK(esk_fixed_div(six, two, ESK_ROUND_NEAREST_EVEN, ESK_OF_WRAP, &q, &o) && q.raw == (3 << 8), "div 6.0/2.0 == 3.0 (Q8)");
        CHECK(esk_fixed_div(sev, two, ESK_ROUND_NEAREST_EVEN, ESK_OF_WRAP, &q, &o) && q.raw == (esk_i128)(3.5 * 256), "div 7.0/2.0 == 3.5 (Q8)");
        CHECK(!esk_fixed_div(six, esk_fixed_from_i64(0, 32, 8, ESK_OF_WRAP, NULL), ESK_ROUND_TRUNCATE, ESK_OF_WRAP, &q, &o), "div by zero returns false");
    }

    /* ===================== printing ===================== */
    {
        char buf[80];
        esk_fixed f = esk_fixed_from_double(-3.25, 64, 16, ESK_ROUND_NEAREST_EVEN, ESK_OF_SATURATE, NULL, NULL);
        esk_fixed_to_string(f, buf, sizeof buf);
        CHECK(strcmp(buf, "-3.25") == 0, "fixed_to_string(-3.25) exact decimal");
        esk_fixed g = esk_fixed_from_i64(42, 32, 8, ESK_OF_WRAP, NULL);
        esk_fixed_to_string(g, buf, sizeof buf);
        CHECK(strcmp(buf, "42") == 0, "fixed_to_string(42) integer");
    }

    return harness_summary("test_fixed");
}
