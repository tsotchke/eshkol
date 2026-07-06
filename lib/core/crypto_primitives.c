/*******************************************************************************
 * Core Cryptographic Primitives (SHA-256 / HMAC-SHA256 / secure random)
 *
 * These symbols back lib/agent/crypto.esk's `hmac-sha256` / `sha256` /
 * `random-bytes` / `random-hex` builtins, and are also used transitively by
 * faculties such as core.memory (content hashing / id generation) via
 * `(require agent.crypto)`.
 *
 * Historically these lived in lib/agent/c/agent_crypto.c and were only linked
 * into a binary when the AOT link step detected `(require agent.…)` in the
 * user's source (see requires_agent_ffi() / ESHKOL_HOST_AGENT_FFI_LINK_ARGS in
 * exe/eshkol-run.cpp) — pulling in the whole eshkol-agent-ffi static archive
 * (libcurl/sqlite3/pcre2/ncurses and all). That made a script's crypto
 * dependency fragile: an AOT link racing a concurrent rebuild of
 * libeshkol-agent-ffi.a (a much larger, more frequently rebuilt archive due to
 * its many optional sub-modules) could see a partially-written archive and
 * fail with "undefined symbols" for eshkol_sha256/eshkol_hmac_sha256/etc, even
 * though the runtime otherwise built fine (Noesis bug report #2, 2026-07-04).
 *
 * These four primitives have no optional/external dependency beyond the
 * platform's own system crypto (CommonCrypto+Security on Apple, BCrypt on
 * Windows, OpenSSL on everything else — the same OpenSSL dependency
 * lib/agent/c/agent_platform.c's file-hashing helper already assumes is
 * present on non-Apple/non-Windows hosts). There is no reason to gate them
 * behind the optional agent-FFI bundle, so they now live in the core runtime
 * (lib/core, part of the always-built/always-linked eshkol-runtime archive)
 * and are unconditionally present in every eshkol-run AOT binary and every
 * `-r`/JIT invocation, regardless of whether the source happens to
 * `(require agent.crypto)` directly or only transitively, and independent of
 * whatever state the separate, optional eshkol-agent-ffi archive is in.
 *
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
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
#elif defined(_WIN32)
#include <Windows.h>
#include <bcrypt.h>
#define HAVE_BCRYPT 1
#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS ((NTSTATUS)0x00000000L)
#endif
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

#ifdef HAVE_BCRYPT
/* One-shot BCryptHash needs a fresh algorithm-provider handle per call
 * (opening it once with BCRYPT_HASH_REUSABLE_FLAG and caching it would avoid
 * the repeated open/close, but these primitives are called at human/agent
 * cadence, not in a hot loop, so simplicity wins here). Returns 0 on success. */
static int bcrypt_digest(LPCWSTR alg_id, ULONG open_flags,
                          const unsigned char* secret, size_t secret_len,
                          const unsigned char* data, size_t data_len,
                          unsigned char* digest, ULONG digest_len) {
    BCRYPT_ALG_HANDLE hAlg = NULL;
    NTSTATUS status = BCryptOpenAlgorithmProvider(&hAlg, alg_id, NULL, open_flags);
    if (status != STATUS_SUCCESS || hAlg == NULL) return -1;

    status = BCryptHash(hAlg,
                         (PUCHAR)secret, (ULONG)secret_len,
                         (PUCHAR)data, (ULONG)data_len,
                         digest, digest_len);

    BCryptCloseAlgorithmProvider(hAlg, 0);
    return status == STATUS_SUCCESS ? 0 : -1;
}
#endif

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
#elif defined(HAVE_BCRYPT)
    if (bcrypt_digest(BCRYPT_SHA256_ALGORITHM, BCRYPT_ALG_HANDLE_HMAC_FLAG,
                       (const unsigned char*)key, key_len,
                       (const unsigned char*)data, data_len,
                       digest, sizeof(digest)) != 0) {
        return -1;
    }
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
#elif defined(HAVE_BCRYPT)
    if (bcrypt_digest(BCRYPT_SHA256_ALGORITHM, 0,
                       NULL, 0,
                       (const unsigned char*)data, data_len,
                       digest, sizeof(digest)) != 0) {
        return -1;
    }
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
#elif defined(HAVE_BCRYPT)
    NTSTATUS status = BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)len,
                                       BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return status == STATUS_SUCCESS ? 0 : -1;
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
