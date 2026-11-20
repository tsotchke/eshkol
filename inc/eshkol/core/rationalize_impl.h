/* Shared by the native and portable VM numeric towers. Each supplies its
 * arena, bignum type, and RZ_OP(name) arithmetic operations. The two inputs
 * are reduced fractions with positive denominators. */
static bool simplest_rational(RzArena* arena, RzInteger* x, RzInteger* xd,
                              RzInteger* tol, RzInteger* td,
                              RzInteger** numerator, RzInteger** denominator) {
    RzInteger* zero = RZ_OP(from_int64)(arena, 0);
    RzInteger* one = RZ_OP(from_int64)(arena, 1);
    if (!x || !xd || !tol || !td || !zero || !one) return false;
    if (RZ_OP(is_zero)(tol)) {
        *numerator = x;
        *denominator = xd;
        return true;
    }
    bool negative = RZ_OP(compare)(x, zero) < 0;
    if (negative) x = RZ_OP(neg)(arena, x);
    if (RZ_OP(compare)(tol, zero) < 0) tol = RZ_OP(neg)(arena, tol);
    if (!x || !tol) return false;
    RzInteger* center = RZ_OP(mul)(arena, x, td);
    RzInteger* radius = RZ_OP(mul)(arena, tol, xd);
    if (!center || !radius) return false;
    if (RZ_OP(compare)(center, radius) <= 0) {
        *numerator = zero;
        *denominator = one;
        return true;
    }

    RzInteger* low = RZ_OP(sub)(arena, center, radius);
    RzInteger* high = RZ_OP(add)(arena, center, radius);
    RzInteger* ld = RZ_OP(mul)(arena, xd, td);
    RzInteger* hd = ld;
    // Map the current continued-fraction tail t back to (a*t+b)/(c*t+d).
    RzInteger *a = one, *b = zero, *c = zero, *d = one;
    while (low && high && ld && hd) {
        RzInteger* q = RZ_OP(div)(arena, low, ld);
        RzInteger* rem = RZ_OP(mod)(arena, low, ld);
        RzInteger* hq = RZ_OP(div)(arena, high, hd);
        if (!q || !rem || !hq) return false;
        bool done = RZ_OP(is_zero)(rem);
        if (!done && RZ_OP(compare)(q, hq) < 0) {
            // ceil(low/ld) is the smallest integer in this closed interval.
            q = RZ_OP(add)(arena, q, one);
            if (!q) return false;
            done = true;
        }
        RzInteger* aq = RZ_OP(mul)(arena, a, q);
        RzInteger* cq = RZ_OP(mul)(arena, c, q);
        if (!aq || !cq) return false;
        RzInteger* next_a = RZ_OP(add)(arena, aq, b);
        RzInteger* next_c = RZ_OP(add)(arena, cq, d);
        if (!next_a || !next_c) return false;
        if (done) {
            *numerator = negative ? RZ_OP(neg)(arena, next_a) : next_a;
            *denominator = next_c;
            return *numerator != NULL;
        }
        // Both endpoints share q. Reciprocating the fractional parts
        // reverses their order and strictly decreases the denominators.
        RzInteger* hi_rem = RZ_OP(mod)(arena, high, hd);
        low = hd;
        high = ld;
        ld = hi_rem;
        hd = rem;
        b = a;
        d = c;
        a = next_a;
        c = next_c;
    }
    return false;
}
#undef RZ_OP
