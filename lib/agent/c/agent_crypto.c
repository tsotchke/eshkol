/*******************************************************************************
 * Cryptographic Primitives for Eshkol Agent
 *
 * HMAC-SHA256, SHA256, and secure random generation via OpenSSL.
 *
 * Copyright (c) 2025 Eshkol Project
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef __APPLE__
#include <CommonCrypto/CommonHMAC.h>
#include <CommonCrypto/CommonDigest.h>
#include <Security/SecRandom.h>
#define HAVE_COMMONCRYPTO 1
#else
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#define HAVE_OPENSSL 1
#endif

/*******************************************************************************
 * Hex Encoding
 ******************************************************************************/

static void bytes_to_hex(const unsigned char* bytes, size_t len,
                          char* hex, size_t hex_size) {
    static const char hexchars[] = "0123456789abcdef";
    size_t i;
    for (i = 0; i < len && (i * 2 + 1) < hex_size; i++) {
        hex[i * 2]     = hexchars[(bytes[i] >> 4) & 0x0f];
        hex[i * 2 + 1] = hexchars[bytes[i] & 0x0f];
    }
    hex[i * 2] = '\0';
}

/*******************************************************************************
 * HMAC-SHA256
 ******************************************************************************/

int eshkol_hmac_sha256(const char* key, size_t key_len,
                        const char* data, size_t data_len,
                        char* hex_output, size_t output_size) {
    if (!key || !data || !hex_output || output_size < 65) return -1;

    unsigned char digest[32];

#ifdef HAVE_COMMONCRYPTO
    CCHmac(kCCHmacAlgSHA256,
           key, key_len,
           data, data_len,
           digest);
#elif defined(HAVE_OPENSSL)
    unsigned int digest_len = 32;
    HMAC(EVP_sha256(),
         key, (int)key_len,
         (const unsigned char*)data, data_len,
         digest, &digest_len);
#endif

    bytes_to_hex(digest, 32, hex_output, output_size);
    return 0;
}

/*******************************************************************************
 * SHA256
 ******************************************************************************/

int eshkol_sha256(const char* data, size_t data_len,
                   char* hex_output, size_t output_size) {
    if (!data || !hex_output || output_size < 65) return -1;

    unsigned char digest[32];

#ifdef HAVE_COMMONCRYPTO
    CC_SHA256(data, (CC_LONG)data_len, digest);
#elif defined(HAVE_OPENSSL)
    SHA256((const unsigned char*)data, data_len, digest);
#endif

    bytes_to_hex(digest, 32, hex_output, output_size);
    return 0;
}

/*******************************************************************************
 * Secure Random
 ******************************************************************************/

int eshkol_random_bytes(char* buf, size_t len) {
    if (!buf || len == 0) return -1;

#ifdef HAVE_COMMONCRYPTO
    return SecRandomCopyBytes(kSecRandomDefault, len, (uint8_t*)buf) == errSecSuccess ? 0 : -1;
#elif defined(HAVE_OPENSSL)
    return RAND_bytes((unsigned char*)buf, (int)len) == 1 ? 0 : -1;
#else
    /* Fallback to /dev/urandom */
    FILE* f = fopen("/dev/urandom", "rb");
    if (!f) return -1;
    size_t n = fread(buf, 1, len, f);
    fclose(f);
    return n == len ? 0 : -1;
#endif
}

int eshkol_random_hex(char* buf, size_t hex_len) {
    if (!buf || hex_len < 2) return -1;

    size_t byte_len = hex_len / 2;
    unsigned char* bytes = malloc(byte_len);
    if (!bytes) return -1;

    int rc = eshkol_random_bytes((char*)bytes, byte_len);
    if (rc != 0) {
        free(bytes);
        return -1;
    }

    bytes_to_hex(bytes, byte_len, buf, hex_len + 1);
    free(bytes);
    return 0;
}
