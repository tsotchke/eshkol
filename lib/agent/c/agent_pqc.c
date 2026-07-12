/*******************************************************************************
 * Moonlab ML-KEM Bindings for Eshkol (Stage S4)
 *
 * A narrow FIPS 203 bridge over Moonlab's ML-KEM-512, ML-KEM-768, and
 * ML-KEM-1024 APIs.  Scheme owns every key, ciphertext, and shared-secret as
 * an Eshkol bytevector; this shim only receives the bytevector payloads after
 * validating their heap subtype and exact parameter-set length.
 *
 * The real implementation is compiled only with ESHKOL_HAVE_MOONLAB, which
 * CMake defines when -DESHKOL_QUANTUM_ENABLED=ON successfully provides
 * Moonlab's quantumsim target.  The unconditional stub branch deliberately
 * keeps default builds linkable and reports an honest unavailable error.
 ******************************************************************************/

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "../../../inc/eshkol/eshkol.h"

static char g_mlkem_last_error[256] = {0};

static void mlkem_set_last_error(const char* message) {
    if (!message) {
        g_mlkem_last_error[0] = '\0';
        return;
    }
    size_t length = strlen(message);
    if (length >= sizeof(g_mlkem_last_error)) {
        length = sizeof(g_mlkem_last_error) - 1;
    }
    memcpy(g_mlkem_last_error, message, length);
    g_mlkem_last_error[length] = '\0';
}

int32_t eshkol_mlkem_last_error(char* buffer, int64_t buffer_size) {
    if (!buffer || buffer_size <= 0) return 0;
    size_t capacity = (size_t)buffer_size;
    size_t length = strlen(g_mlkem_last_error);
    if (length >= capacity) length = capacity - 1;
    memcpy(buffer, g_mlkem_last_error, length);
    buffer[length] = '\0';
    return (int32_t)length;
}

#ifdef ESHKOL_HAVE_MOONLAB

#include "crypto/mlkem/mlkem.h"
#include "crypto/sha3/sha3.h"
#include "crypto/drbg/ctr_drbg.h"

#define ESHKOL_MLKEM_SHARED_SECRET_BYTES 32u

typedef struct {
    size_t public_key_bytes;
    size_t secret_key_bytes;
    size_t ciphertext_bytes;
} eshkol_mlkem_sizes_t;

static int mlkem_sizes_for_level(int32_t level, eshkol_mlkem_sizes_t* sizes) {
    if (!sizes) {
        mlkem_set_last_error("ML-KEM internal error: missing size output");
        return -1;
    }
    switch (level) {
        case 512:
            sizes->public_key_bytes = MLKEM512_PUBLICKEYBYTES;
            sizes->secret_key_bytes = MLKEM512_SECRETKEYBYTES;
            sizes->ciphertext_bytes = MLKEM512_CIPHERTEXTBYTES;
            return 0;
        case 768:
            sizes->public_key_bytes = MLKEM768_PUBLICKEYBYTES;
            sizes->secret_key_bytes = MLKEM768_SECRETKEYBYTES;
            sizes->ciphertext_bytes = MLKEM768_CIPHERTEXTBYTES;
            return 0;
        case 1024:
            sizes->public_key_bytes = MLKEM1024_PUBLICKEYBYTES;
            sizes->secret_key_bytes = MLKEM1024_SECRETKEYBYTES;
            sizes->ciphertext_bytes = MLKEM1024_CIPHERTEXTBYTES;
            return 0;
        default:
            mlkem_set_last_error("ML-KEM level must be 512, 768, or 1024");
            return -1;
    }
}

/* An extern `ptr` receives the Eshkol heap payload, whose first eight bytes
 * are the bytevector length and whose bytes begin at +8.  High-level Scheme
 * wrappers check bytevector? before the FFI call; repeat the exact subtype and
 * length checks here so accidental ABI misuse cannot reach Moonlab. */
static int checked_bytevector(void* raw, size_t expected_length,
                              const char* role, uint8_t** bytes_out) {
    if (!raw || ESHKOL_GET_SUBTYPE(raw) != HEAP_SUBTYPE_BYTEVECTOR) {
        char message[160];
        snprintf(message, sizeof(message),
                 "ML-KEM %s must be an Eshkol bytevector", role);
        mlkem_set_last_error(message);
        return -1;
    }

    int64_t actual_length = -1;
    memcpy(&actual_length, raw, sizeof(actual_length));
    if (actual_length < 0 || (uint64_t)actual_length != (uint64_t)expected_length) {
        char message[192];
        snprintf(message, sizeof(message),
                 "ML-KEM %s must be exactly %zu bytes", role, expected_length);
        mlkem_set_last_error(message);
        return -1;
    }

    *bytes_out = (uint8_t*)raw + sizeof(int64_t);
    return 0;
}

static int checked_const_bytevector(const void* raw, size_t expected_length,
                                    const char* role, const uint8_t** bytes_out) {
    uint8_t* bytes = NULL;
    if (checked_bytevector((void*)raw, expected_length, role, &bytes) != 0) {
        return -1;
    }
    *bytes_out = bytes;
    return 0;
}

int32_t eshkol_mlkem_keygen(int32_t level, void* public_key, void* secret_key) {
    eshkol_mlkem_sizes_t sizes;
    uint8_t* ek = NULL;
    uint8_t* dk = NULL;
    if (mlkem_sizes_for_level(level, &sizes) != 0 ||
        checked_bytevector(public_key, sizes.public_key_bytes, "public key", &ek) != 0 ||
        checked_bytevector(secret_key, sizes.secret_key_bytes, "secret key", &dk) != 0) {
        return -1;
    }

    int rc = -1;
    switch (level) {
        case 512:  rc = moonlab_mlkem512_keygen_qrng(ek, dk); break;
        case 768:  rc = moonlab_mlkem768_keygen_qrng(ek, dk); break;
        case 1024: rc = moonlab_mlkem1024_keygen_qrng(ek, dk); break;
        default: return -1; /* Already rejected by mlkem_sizes_for_level. */
    }
    if (rc != 0) {
        mlkem_set_last_error("Moonlab ML-KEM key generation failed: quantum RNG unavailable");
        return -1;
    }
    mlkem_set_last_error(NULL);
    return 0;
}

int32_t eshkol_mlkem_encaps(int32_t level, void* ciphertext,
                            void* shared_secret, const void* public_key) {
    eshkol_mlkem_sizes_t sizes;
    uint8_t* c = NULL;
    uint8_t* K = NULL;
    const uint8_t* ek = NULL;
    if (mlkem_sizes_for_level(level, &sizes) != 0 ||
        checked_bytevector(ciphertext, sizes.ciphertext_bytes, "ciphertext", &c) != 0 ||
        checked_bytevector(shared_secret, ESHKOL_MLKEM_SHARED_SECRET_BYTES,
                           "shared secret", &K) != 0 ||
        checked_const_bytevector(public_key, sizes.public_key_bytes, "public key", &ek) != 0) {
        return -1;
    }

    int rc = -1;
    switch (level) {
        case 512:  rc = moonlab_mlkem512_encaps_qrng(c, K, ek); break;
        case 768:  rc = moonlab_mlkem768_encaps_qrng(c, K, ek); break;
        case 1024: rc = moonlab_mlkem1024_encaps_qrng(c, K, ek); break;
        default: return -1;
    }
    if (rc != 0) {
        mlkem_set_last_error("Moonlab ML-KEM encapsulation failed: quantum RNG unavailable");
        return -1;
    }
    mlkem_set_last_error(NULL);
    return 0;
}

int32_t eshkol_mlkem_decaps(int32_t level, void* shared_secret,
                            const void* ciphertext, const void* secret_key) {
    eshkol_mlkem_sizes_t sizes;
    uint8_t* K = NULL;
    const uint8_t* c = NULL;
    const uint8_t* dk = NULL;
    if (mlkem_sizes_for_level(level, &sizes) != 0 ||
        checked_bytevector(shared_secret, ESHKOL_MLKEM_SHARED_SECRET_BYTES,
                           "shared secret", &K) != 0 ||
        checked_const_bytevector(ciphertext, sizes.ciphertext_bytes, "ciphertext", &c) != 0 ||
        checked_const_bytevector(secret_key, sizes.secret_key_bytes, "secret key", &dk) != 0) {
        return -1;
    }

    /* FIPS 203 decapsulation has implicit rejection: malformed but correctly
     * sized ciphertexts produce a pseudorandom 32-byte secret, not an error. */
    switch (level) {
        case 512:  moonlab_mlkem512_decaps(K, c, dk); break;
        case 768:  moonlab_mlkem768_decaps(K, c, dk); break;
        case 1024: moonlab_mlkem1024_decaps(K, c, dk); break;
        default: return -1;
    }
    mlkem_set_last_error(NULL);
    return 0;
}

/* Moonlab ships a NIST-seeded deterministic KAT harness, but not the raw .rsp
 * artifacts.  Reproduce its count=0 SP 800-90A seed flow here so the Eshkol
 * integration tests verify the bridge against the same fixed FIPS 203 output
 * fingerprints while production APIs remain QRNG-only. */
typedef struct {
    const char* public_key_hash;
    const char* secret_key_hash;
    const char* ciphertext_hash;
    const char* shared_secret_hash;
} mlkem_kat_hashes_t;

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int sha3_256_matches_hex(const uint8_t* bytes, size_t length,
                                 const char* expected_hex) {
    uint8_t digest[32];
    sha3_256(bytes, length, digest);
    for (size_t i = 0; i < sizeof(digest); ++i) {
        int high = hex_nibble(expected_hex[2 * i]);
        int low = hex_nibble(expected_hex[2 * i + 1]);
        if (high < 0 || low < 0 || digest[i] != (uint8_t)((high << 4) | low)) {
            return 0;
        }
    }
    return expected_hex[64] == '\0';
}

static void secure_clear(void* ptr, size_t length) {
    volatile uint8_t* bytes = (volatile uint8_t*)ptr;
    while (length-- > 0) *bytes++ = 0;
}

int32_t eshkol_mlkem_nist_kat(int32_t level) {
    static const char seed_hex[] =
        "061550234D158C5EC95595FE04EF7A25767F2E24CC2BC479D09D86DC9ABCFDE7"
        "056A8C266F9EF97ED08541DBD2E1FFA1";
    static const mlkem_kat_hashes_t kat512 = {
        "50c8dd152a4531aab560d2fc7ca9a40ad8af25ad1dd08c6d79afe4dd4d1eee5a",
        "2e08d2c82a07f5f67878b54e06848c924ee7a0929cb6440d06fffd9622687b48",
        "ed14369380d501d2bb28861a26ad092b1bbd6764c083244551c436ccf98dd9f8",
        "ea6dfd89a819935fa7b9072a0cd4b495e751e5620d2cb173fc62843255a959d8"
    };
    static const mlkem_kat_hashes_t kat768 = {
        "f57262661358cde8d3ebf990e5fd1d5b896c992ccfaadb5256b68bbf5943b132",
        "7deef44965b03d76de543ad6ef9e74a2772fa5a9fa0e761120dac767cf0152ef",
        "6e777e2cf8054659136a971d9e70252f301226930c19c470ee0688163a63c15b",
        "732325d1305c70fd98df41716b42041eee95feba84bbc59a68175aee00f03b1e"
    };
    static const mlkem_kat_hashes_t kat1024 = {
        "ebbe41cd4dea489dedd00e76ae0bcf54aa8550202920eb64d5892ad02b13f2e5",
        "ecfa369ec7e834204291fccb4d59c3fd1557b0ec3ab2ed7d50c98c01f292612e",
        "507dcdae0432f558189a09c5e79fd69896e2f830e37ae4d598b00566e55f20f5",
        "2184138ce3b4d73ccc2f1c8b6c14b2c52df6ffc8d34dce162af386c6fad2d941"
    };

    eshkol_mlkem_sizes_t sizes;
    const mlkem_kat_hashes_t* expected = NULL;
    if (mlkem_sizes_for_level(level, &sizes) != 0) return -1;
    switch (level) {
        case 512: expected = &kat512; break;
        case 768: expected = &kat768; break;
        case 1024: expected = &kat1024; break;
        default: return -1;
    }

    uint8_t seed[48];
    uint8_t d[32], z[32], m[32], K[ESHKOL_MLKEM_SHARED_SECRET_BYTES];
    uint8_t public_key[MLKEM1024_PUBLICKEYBYTES];
    uint8_t secret_key[MLKEM1024_SECRETKEYBYTES];
    uint8_t ciphertext[MLKEM1024_CIPHERTEXTBYTES];
    for (size_t i = 0; i < sizeof(seed); ++i) {
        int high = hex_nibble(seed_hex[2 * i]);
        int low = hex_nibble(seed_hex[2 * i + 1]);
        if (high < 0 || low < 0) {
            mlkem_set_last_error("ML-KEM KAT has an invalid NIST seed constant");
            return -1;
        }
        seed[i] = (uint8_t)((high << 4) | low);
    }

    ctr_drbg_ctx_t drbg;
    ctr_drbg_init(&drbg, seed);
    ctr_drbg_generate(&drbg, d, sizeof(d));
    ctr_drbg_generate(&drbg, z, sizeof(z));
    ctr_drbg_generate(&drbg, m, sizeof(m));

    switch (level) {
        case 512:
            moonlab_mlkem512_keygen(public_key, secret_key, d, z);
            moonlab_mlkem512_encaps(ciphertext, K, public_key, m);
            break;
        case 768:
            moonlab_mlkem768_keygen(public_key, secret_key, d, z);
            moonlab_mlkem768_encaps(ciphertext, K, public_key, m);
            break;
        case 1024:
            moonlab_mlkem1024_keygen(public_key, secret_key, d, z);
            moonlab_mlkem1024_encaps(ciphertext, K, public_key, m);
            break;
        default:
            return -1;
    }

    int valid = sha3_256_matches_hex(public_key, sizes.public_key_bytes,
                                     expected->public_key_hash) &&
                sha3_256_matches_hex(secret_key, sizes.secret_key_bytes,
                                     expected->secret_key_hash) &&
                sha3_256_matches_hex(ciphertext, sizes.ciphertext_bytes,
                                     expected->ciphertext_hash) &&
                sha3_256_matches_hex(K, sizeof(K), expected->shared_secret_hash);

    secure_clear(&drbg, sizeof(drbg));
    secure_clear(seed, sizeof(seed));
    secure_clear(d, sizeof(d));
    secure_clear(z, sizeof(z));
    secure_clear(m, sizeof(m));
    secure_clear(K, sizeof(K));
    secure_clear(secret_key, sizeof(secret_key));

    if (!valid) {
        mlkem_set_last_error("Moonlab ML-KEM NIST deterministic KAT fingerprint mismatch");
        return -1;
    }
    mlkem_set_last_error(NULL);
    return 0;
}

#else

static int32_t mlkem_unavailable(const char* operation) {
    char message[256];
    snprintf(message, sizeof(message),
             "ML-KEM %s unavailable: rebuild with -DESHKOL_QUANTUM_ENABLED=ON",
             operation);
    mlkem_set_last_error(message);
    return -1;
}

int32_t eshkol_mlkem_keygen(int32_t level, void* public_key, void* secret_key) {
    (void)level;
    (void)public_key;
    (void)secret_key;
    return mlkem_unavailable("key generation");
}

int32_t eshkol_mlkem_encaps(int32_t level, void* ciphertext,
                            void* shared_secret, const void* public_key) {
    (void)level;
    (void)ciphertext;
    (void)shared_secret;
    (void)public_key;
    return mlkem_unavailable("encapsulation");
}

int32_t eshkol_mlkem_decaps(int32_t level, void* shared_secret,
                            const void* ciphertext, const void* secret_key) {
    (void)level;
    (void)shared_secret;
    (void)ciphertext;
    (void)secret_key;
    return mlkem_unavailable("decapsulation");
}

int32_t eshkol_mlkem_nist_kat(int32_t level) {
    (void)level;
    return mlkem_unavailable("NIST KAT");
}

#endif
