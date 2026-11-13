#ifndef ESHKOL_VM_INTEGER_ARITHMETIC_H
#define ESHKOL_VM_INTEGER_ARITHMETIC_H

#include <stdint.h>

/* The caller must reject a zero divisor. C's signed remainder is undefined
 * for INT64_MIN / -1 even though the mathematical remainder is zero. Keep
 * that boundary shared by both interpreter loops and first-class natives. */
static inline int64_t vm_i64_remainder_nonzero(int64_t dividend, int64_t divisor) {
    return divisor == -1 ? 0 : dividend % divisor;
}

#endif
