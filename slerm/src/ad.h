#ifndef SLERMES_AD_H
#define SLERMES_AD_H

#include "value.h"

/* Forward-mode automatic differentiation via truncated Taylor towers.
 *
 * A "tower" is coeff[0..order] where coeff[0] is the value and coeff[k]
 * is the k-th derivative coefficient (coeff[k]/k! scaled by the mode).
 * We store the raw Taylor coefficients a_k of f(x0+h) = sum a_k h^k.
 * Then:
 *   f(x0)            = a_0
 *   f'(x0)           = a_1
 *   f^{(k)}(x0)      = a_k * k!
 *
 * All arithmetic primitives are tower-aware: if any operand is a tower,
 * the result is a tower (Cauchy products / recurrences); otherwise plain
 * doubles are returned. This makes AD transparent inside normal eval. */

extern int AD_ORDER;   /* set by derivative-n / taylor before evaluating */

/* promote a number or tower to a tower truncated/extended to AD_ORDER */
Value *ad_promote(Value *v);

/* arithmetic (tower-aware). Returns +1 Value*. */
Value *ad_add(Value *a, Value *b);
Value *ad_sub(Value *a, Value *b);
Value *ad_mul(Value *a, Value *b);
Value *ad_div(Value *a, Value *b);
Value *ad_neg(Value *a);

/* elementary functions (tower-aware) */
Value *ad_sin(Value *a);
Value *ad_cos(Value *a);
Value *ad_exp(Value *a);
Value *ad_log(Value *a);
Value *ad_sqrt(Value *a);
/* power with a CONSTANT real exponent c: a^c */
Value *ad_pow_const(Value *a, double c);

/* read coefficient k (raw Taylor coeff) of a tower/number */
double ad_coeff(Value *v, int k);

/* make a tower seeded at value x0 with 1 in slot `seed` (0 = just value) */
Value *ad_seed(double x0, int seed);

/* is this a tower (or would promote to one)? */
int ad_is_tower(Value *v);

#endif /* SLERMES_AD_H */
