/*
 * merkle.c — fast non-crypto hash primitives for content addressing.
 *
 * The C inner loop predates the bignum-modulo fix in #221; pure-Eshkol
 * FNV-1a is now correct (and a regression test pins it), but the C
 * version stays because it's ~30× faster on long inputs and avoids
 * heap-allocating a bignum per byte.  No external dependencies — 64-bit
 * modular arithmetic is just uint64_t overflow.
 *
 * For cryptographic content addressing (signing, distributed
 * adversarial storage), use lib/agent/crypto.esk's sha256 instead —
 * SHA256 is the right tool for that and it's already wrapped.
 */

#include "eshkol/core/merkle_hash.h"

int64_t eshkol_fnv1a_64(const char *data, int64_t len) {
    return eshkol_merkle_fnv1a(data, len);
}

int64_t eshkol_u64_to_hex16(int64_t value, char *out, int64_t size) {
    return eshkol_merkle_hex16(value, out, size);
}
