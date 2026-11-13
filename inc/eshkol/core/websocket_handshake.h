/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * websocket_handshake.h — the RFC 6455 opening handshake and client framing
 * values, shared by every Eshkol engine that speaks WebSocket.
 *
 * What a client owes the protocol, and what this header provides:
 *
 *   - A FRESH KEY PER HANDSHAKE. `Sec-WebSocket-Key` is 16 bytes from the
 *     operating system's cryptographic source, base64-encoded
 *     (eshkol_ws_generate_key). A key that repeats between handshakes lets a
 *     cached or replayed response pass for a live one, so each connection
 *     generates its own.
 *   - THE SERVER'S ANSWER CHECKED AGAINST THAT KEY. RFC 6455 section 4.1
 *     defines `Sec-WebSocket-Accept` as base64(SHA-1(key || GUID)); a client
 *     computes it and compares (eshkol_ws_handshake_refusal), together with
 *     the 101 status and the Upgrade and Connection fields. A peer that does
 *     not produce the value derived from this connection's key is not a
 *     WebSocket endpoint answering this handshake.
 *   - A FRESH MASK PER FRAME. Every client frame carries a 4-byte masking key
 *     from the same cryptographic source (eshkol_ws_generate_mask) and its
 *     payload masked with it, which is what keeps intermediaries from being
 *     steered by attacker-chosen payload bytes.
 *
 * SHA-1 appears here for exactly one purpose: it is the function RFC 6455
 * names for the handshake value. It is not offered as, and must not be used
 * as, a hash for anything else.
 *
 * Header-only and dependency-free apart from <stddef.h>/<stdint.h> and the
 * runtime's cryptographic source, so the native runtime, the bytecode VM and
 * the test binaries all compile the same handshake.
 */
#ifndef ESHKOL_CORE_WEBSOCKET_HANDSHAKE_H
#define ESHKOL_CORE_WEBSOCKET_HANDSHAKE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The platform cryptographic source (lib/core/crypto_primitives.c):
 * SecRandomCopyBytes, BCryptGenRandom, RAND_bytes or /dev/urandom.
 * Returns 0 on success.
 *
 * A browser bundle does not carry that source, and the WebSocket client is
 * not built there either. ESHKOL_WS_HAVE_CRYPTO_SOURCE says which case this
 * translation unit is in; where there is no source, the key and mask
 * functions below report failure, which is a refusal rather than a weaker
 * value, and nothing references the symbol. */
#if defined(__EMSCRIPTEN__)
#define ESHKOL_WS_HAVE_CRYPTO_SOURCE 0
#else
#define ESHKOL_WS_HAVE_CRYPTO_SOURCE 1
int eshkol_random_bytes(char* buf, size_t len);
#endif

/** The RFC 6455 section 1.3 handshake GUID. */
#define ESHKOL_WS_GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

/** Buffer sizes, including room for the NUL: a base64 16-byte key is 24
 *  characters, a base64 20-byte SHA-1 digest is 28. */
#define ESHKOL_WS_KEY_CAPACITY 25
#define ESHKOL_WS_ACCEPT_CAPACITY 29

/* ─── SHA-1, for the handshake value and nothing else ─────────────────────── */

typedef struct {
    uint32_t state[5];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_used;
} eshkol_ws_sha1_ctx;

static inline uint32_t eshkol_ws_rotl32(uint32_t v, int n) {
    return (uint32_t)((v << n) | (v >> (32 - n)));
}

static inline void eshkol_ws_sha1_compress(eshkol_ws_sha1_ctx* ctx,
                                           const unsigned char* block) {
    uint32_t w[80];
    uint32_t a, b, c, d, e;
    int i;
    for (i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }
    for (i = 16; i < 80; ++i) {
        w[i] = eshkol_ws_rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2];
    d = ctx->state[3]; e = ctx->state[4];
    for (i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20)      { f = (b & c) | ((~b) & d);        k = 0x5A827999u; }
        else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
        else             { f = b ^ c ^ d;                   k = 0xCA62C1D6u; }
        {
            uint32_t temp = eshkol_ws_rotl32(a, 5) + f + e + k + w[i];
            e = d; d = c; c = eshkol_ws_rotl32(b, 30); b = a; a = temp;
        }
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c;
    ctx->state[3] += d; ctx->state[4] += e;
}

static inline void eshkol_ws_sha1_init(eshkol_ws_sha1_ctx* ctx) {
    ctx->state[0] = 0x67452301u;
    ctx->state[1] = 0xEFCDAB89u;
    ctx->state[2] = 0x98BADCFEu;
    ctx->state[3] = 0x10325476u;
    ctx->state[4] = 0xC3D2E1F0u;
    ctx->bit_count = 0;
    ctx->block_used = 0;
}

static inline void eshkol_ws_sha1_update(eshkol_ws_sha1_ctx* ctx,
                                         const void* data, size_t len) {
    const unsigned char* p = (const unsigned char*)data;
    size_t i;
    ctx->bit_count += (uint64_t)len * 8u;
    for (i = 0; i < len; ++i) {
        ctx->block[ctx->block_used++] = p[i];
        if (ctx->block_used == 64) {
            eshkol_ws_sha1_compress(ctx, ctx->block);
            ctx->block_used = 0;
        }
    }
}

static inline void eshkol_ws_sha1_final(eshkol_ws_sha1_ctx* ctx,
                                        unsigned char out[20]) {
    unsigned char length_be[8];
    unsigned char pad = 0x80;
    int i;
    for (i = 0; i < 8; ++i)
        length_be[i] = (unsigned char)((ctx->bit_count >> (56 - 8 * i)) & 0xffu);
    eshkol_ws_sha1_update(ctx, &pad, 1);
    ctx->bit_count -= 8; /* the padding byte is not message content */
    while (ctx->block_used != 56) {
        unsigned char zero = 0;
        eshkol_ws_sha1_update(ctx, &zero, 1);
        ctx->bit_count -= 8;
    }
    {
        size_t j;
        for (j = 0; j < 8; ++j) ctx->block[56 + j] = length_be[j];
    }
    eshkol_ws_sha1_compress(ctx, ctx->block);
    ctx->block_used = 0;
    for (i = 0; i < 5; ++i) {
        out[i * 4]     = (unsigned char)((ctx->state[i] >> 24) & 0xffu);
        out[i * 4 + 1] = (unsigned char)((ctx->state[i] >> 16) & 0xffu);
        out[i * 4 + 2] = (unsigned char)((ctx->state[i] >> 8) & 0xffu);
        out[i * 4 + 3] = (unsigned char)(ctx->state[i] & 0xffu);
    }
}

/** @brief SHA-1 of `len` bytes at `data` into `out`. The handshake value of
 *         RFC 6455 section 4.1 is its only use in Eshkol. */
static inline void eshkol_ws_sha1(const void* data, size_t len,
                                  unsigned char out[20]) {
    eshkol_ws_sha1_ctx ctx;
    eshkol_ws_sha1_init(&ctx);
    eshkol_ws_sha1_update(&ctx, data, len);
    eshkol_ws_sha1_final(&ctx, out);
}

/* ─── base64, RFC 4648 with padding ───────────────────────────────────────── */

/** @brief Standard base64-encodes `len` bytes at `in` into `out`.
 *  @return 1 on success; 0 when `out_cap` cannot hold the encoding and a NUL. */
static inline int eshkol_ws_base64(const unsigned char* in, size_t len,
                                   char* out, size_t out_cap) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t needed = ((len + 2) / 3) * 4;
    size_t i = 0, o = 0;
    if (!in || !out || out_cap <= needed) return 0;
    while (i < len) {
        uint32_t n = (uint32_t)in[i] << 16;
        size_t remaining = len - i;
        if (remaining > 1) n |= (uint32_t)in[i + 1] << 8;
        if (remaining > 2) n |= (uint32_t)in[i + 2];
        out[o++] = table[(n >> 18) & 0x3f];
        out[o++] = table[(n >> 12) & 0x3f];
        out[o++] = remaining > 1 ? table[(n >> 6) & 0x3f] : '=';
        out[o++] = remaining > 2 ? table[n & 0x3f] : '=';
        i += 3;
    }
    out[o] = '\0';
    return 1;
}

/* ─── Handshake values ────────────────────────────────────────────────────── */

/**
 * @brief Generates one connection's `Sec-WebSocket-Key`.
 *
 * Draws 16 bytes from the operating system's cryptographic source and
 * base64-encodes them, so every handshake carries a value of its own.
 *
 * @param out     Receives the 24-character key and a NUL.
 * @param out_cap Capacity of `out`; must be at least ESHKOL_WS_KEY_CAPACITY.
 * @return 1 on success; 0 when the cryptographic source is unavailable or the
 *         buffer is too small. A caller that gets 0 must not open the
 *         connection: there is no acceptable substitute for the key.
 */
static inline int eshkol_ws_generate_key(char* out, size_t out_cap) {
#if ESHKOL_WS_HAVE_CRYPTO_SOURCE
    unsigned char raw[16];
    if (!out || out_cap < ESHKOL_WS_KEY_CAPACITY) return 0;
    if (eshkol_random_bytes((char*)raw, sizeof(raw)) != 0) return 0;
    return eshkol_ws_base64(raw, sizeof(raw), out, out_cap);
#else
    (void)out; (void)out_cap;
    return 0;
#endif
}

/**
 * @brief Generates one frame's 4-byte masking key.
 *
 * @return 1 on success; 0 when the cryptographic source is unavailable, in
 *         which case the frame must not be sent.
 */
static inline int eshkol_ws_generate_mask(unsigned char out[4]) {
    if (!out) return 0;
#if ESHKOL_WS_HAVE_CRYPTO_SOURCE
    return eshkol_random_bytes((char*)out, 4) == 0;
#else
    return 0;
#endif
}

/**
 * @brief Computes the `Sec-WebSocket-Accept` value a server must return for
 *        `key`, per RFC 6455 section 4.1: base64(SHA-1(key || GUID)).
 *
 * @param key     The `Sec-WebSocket-Key` this client sent.
 * @param out     Receives the 28-character value and a NUL.
 * @param out_cap Capacity of `out`; at least ESHKOL_WS_ACCEPT_CAPACITY.
 * @return 1 on success, 0 on a NULL key or an undersized buffer.
 */
static inline int eshkol_ws_expected_accept(const char* key,
                                            char* out, size_t out_cap) {
    eshkol_ws_sha1_ctx ctx;
    unsigned char digest[20];
    size_t key_len = 0;
    if (!key || !out || out_cap < ESHKOL_WS_ACCEPT_CAPACITY) return 0;
    while (key[key_len]) key_len++;
    eshkol_ws_sha1_init(&ctx);
    eshkol_ws_sha1_update(&ctx, key, key_len);
    eshkol_ws_sha1_update(&ctx, ESHKOL_WS_GUID, sizeof(ESHKOL_WS_GUID) - 1);
    eshkol_ws_sha1_final(&ctx, digest);
    return eshkol_ws_base64(digest, sizeof(digest), out, out_cap);
}

/* ─── Reading the server's answer ─────────────────────────────────────────── */

static inline int eshkol_ws_ascii_lower(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c;
}

/** @brief Case-insensitive comparison of `len` bytes at `a` against the
 *         NUL-terminated `b`, which must already be lowercase. */
static inline int eshkol_ws_iequal(const char* a, size_t len, const char* b) {
    size_t i;
    for (i = 0; i < len; ++i) {
        if (!b[i]) return 0;
        if (eshkol_ws_ascii_lower((unsigned char)a[i]) != (unsigned char)b[i])
            return 0;
    }
    return b[len] == '\0';
}

/**
 * @brief Finds a header field in a raw response head and reports its value.
 *
 * Matches the field name only at the start of a line, so a name appearing
 * inside another field's value cannot be mistaken for the field.
 *
 * @param head      Response bytes, starting at the status line.
 * @param head_len  Length of `head` in bytes.
 * @param name      Lowercase field name to find.
 * @param value_out Receives a pointer into `head` at the first value byte.
 * @param value_len Receives the value length, trailing whitespace trimmed.
 * @return 1 when the field is present, 0 otherwise.
 */
static inline int eshkol_ws_find_header(const char* head, size_t head_len,
                                        const char* name,
                                        const char** value_out,
                                        size_t* value_len) {
    size_t line_start = 0;
    size_t name_len = 0;
    if (!head || !name) return 0;
    while (name[name_len]) name_len++;
    /* Skip the status line; fields begin after the first line terminator. */
    while (line_start < head_len && head[line_start] != '\n') line_start++;
    if (line_start < head_len) line_start++;
    while (line_start < head_len) {
        size_t line_end = line_start;
        while (line_end < head_len && head[line_end] != '\n') line_end++;
        {
            size_t line_len = line_end - line_start;
            if (line_len > 0 && head[line_start + line_len - 1] == '\r')
                line_len--;
            if (line_len == 0) return 0; /* end of the header section */
            if (line_len > name_len &&
                head[line_start + name_len] == ':' &&
                eshkol_ws_iequal(head + line_start, name_len, name)) {
                size_t value_start = line_start + name_len + 1;
                size_t value_end = line_start + line_len;
                while (value_start < value_end &&
                       (head[value_start] == ' ' || head[value_start] == '\t'))
                    value_start++;
                while (value_end > value_start &&
                       (head[value_end - 1] == ' ' || head[value_end - 1] == '\t'))
                    value_end--;
                if (value_out) *value_out = head + value_start;
                if (value_len) *value_len = value_end - value_start;
                return 1;
            }
        }
        line_start = line_end < head_len ? line_end + 1 : head_len;
    }
    return 0;
}

/** @brief True when a comma-separated field value contains `token`
 *         (case-insensitively) as one of its elements. */
static inline int eshkol_ws_value_has_token(const char* value, size_t len,
                                            const char* token) {
    size_t i = 0;
    while (i < len) {
        size_t start = i;
        size_t end;
        while (i < len && value[i] != ',') i++;
        end = i;
        while (start < end && (value[start] == ' ' || value[start] == '\t')) start++;
        while (end > start && (value[end - 1] == ' ' || value[end - 1] == '\t')) end--;
        if (eshkol_ws_iequal(value + start, end - start, token)) return 1;
        if (i < len) i++;
    }
    return 0;
}

/**
 * @brief Decides whether a server's opening-handshake response completes the
 *        handshake this client started with `key`.
 *
 * Checks, in order: an HTTP/1.x status line carrying 101; `Upgrade` naming
 * websocket; `Connection` naming the Upgrade token; and
 * `Sec-WebSocket-Accept` equal to base64(SHA-1(key || GUID)).
 *
 * @param response Response bytes as received (the status line and fields; a
 *                 body, if any, is ignored).
 * @param length   Length of `response` in bytes.
 * @param key      The key this client sent in the request.
 * @return NULL when the handshake is complete; otherwise a static string
 *         naming what did not match, suitable for an error a program catches.
 */
static inline const char* eshkol_ws_handshake_refusal(const char* response,
                                                      size_t length,
                                                      const char* key) {
    const char* value = NULL;
    size_t value_len = 0;
    char expected[ESHKOL_WS_ACCEPT_CAPACITY];
    if (!response || length == 0) return "no response";
    /* Status line: HTTP/1.x SP 101 */
    {
        size_t i = 0;
        size_t line_len = 0;
        while (line_len < length && response[line_len] != '\n') line_len++;
        if (line_len < 12) return "status line";
        if (!(response[0] == 'H' && response[1] == 'T' && response[2] == 'T' &&
              response[3] == 'P' && response[4] == '/' && response[5] == '1' &&
              response[6] == '.')) return "status line";
        i = 8;
        while (i < line_len && response[i] == ' ') i++;
        if (i + 2 >= line_len) return "status line";
        if (!(response[i] == '1' && response[i + 1] == '0' &&
              response[i + 2] == '1')) return "status 101";
        if (i + 3 < line_len && response[i + 3] != ' ' &&
            response[i + 3] != '\r') return "status 101";
    }
    if (!eshkol_ws_find_header(response, length, "upgrade", &value, &value_len) ||
        !eshkol_ws_value_has_token(value, value_len, "websocket"))
        return "Upgrade header";
    if (!eshkol_ws_find_header(response, length, "connection", &value, &value_len) ||
        !eshkol_ws_value_has_token(value, value_len, "upgrade"))
        return "Connection header";
    if (!eshkol_ws_find_header(response, length, "sec-websocket-accept",
                               &value, &value_len))
        return "Sec-WebSocket-Accept header";
    if (!eshkol_ws_expected_accept(key, expected, sizeof(expected)))
        return "Sec-WebSocket-Accept header";
    {
        size_t expected_len = 0;
        size_t i;
        unsigned char diff = 0;
        while (expected[expected_len]) expected_len++;
        if (value_len != expected_len) return "Sec-WebSocket-Accept value";
        for (i = 0; i < expected_len; ++i)
            diff |= (unsigned char)(value[i] ^ expected[i]);
        if (diff != 0) return "Sec-WebSocket-Accept value";
    }
    return NULL;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif


/* ─── Received frames (RFC 6455 section 5) ────────────────────────────────── */

/**
 * @brief The largest frame payload this client will accept, in bytes.
 *
 * A budget, not a constant: `ESHKOL_WS_MAX_FRAME_BYTES` overrides it, the way
 * the server's three budgets are overridden. The default is 64 KiB. A frame
 * declaring more is refused with close code 1009 (message too big) and the
 * connection is closed, because a client that returned without consuming the
 * payload would leave the stream desynchronised: every later read would parse
 * attacker-chosen payload bytes as a frame header.
 */
static inline size_t eshkol_ws_max_frame_bytes(void) {
    const char* raw = getenv("ESHKOL_WS_MAX_FRAME_BYTES");
    if (raw && *raw) {
        char* end = NULL;
        unsigned long long parsed = strtoull(raw, &end, 10);
        if (end && *end == '\0' && parsed > 0) return (size_t)parsed;
    }
    return 65536u;
}

/** @brief What a received frame header is, once judged. */
typedef enum {
    ESHKOL_WS_FRAME_OK = 0,
    /** RFC 6455 section 5.1: a client must fail the connection on a masked
     *  frame from a server. */
    ESHKOL_WS_FRAME_MASKED_FROM_SERVER,
    /** Section 5.2: RSV1-3 are zero unless an extension negotiated them, and
     *  this client negotiates none. */
    ESHKOL_WS_FRAME_RESERVED_BITS_SET,
    /** Section 5.5: a control frame carries at most 125 bytes. */
    ESHKOL_WS_FRAME_CONTROL_TOO_LARGE,
    /** Section 5.5: a control frame is never fragmented. */
    ESHKOL_WS_FRAME_CONTROL_FRAGMENTED,
    /** Section 5.4: a continuation outside a fragment sequence, or a new data
     *  frame inside one. */
    ESHKOL_WS_FRAME_BAD_CONTINUATION,
    /** Section 5.2: opcodes 3-7 and 11-15 are reserved. */
    ESHKOL_WS_FRAME_RESERVED_OPCODE,
    /** The payload exceeds eshkol_ws_max_frame_bytes(). */
    ESHKOL_WS_FRAME_TOO_LARGE
} eshkol_ws_frame_verdict;

/**
 * @brief Judges a received frame header against RFC 6455 section 5.
 *
 * @param first_byte  FIN, RSV1-3 and the opcode.
 * @param second_byte MASK and the 7-bit length.
 * @param length      Payload length, after any 16- or 64-bit extension.
 * @param in_fragment Non-zero when a fragment sequence is open.
 * @return ESHKOL_WS_FRAME_OK, or the rule the frame breaks.
 */
static inline eshkol_ws_frame_verdict eshkol_ws_examine_frame(
    unsigned char first_byte, unsigned char second_byte, uint64_t length,
    int in_fragment) {
    const int fin = (first_byte & 0x80) != 0;
    const int opcode = first_byte & 0x0f;
    const int control = (opcode & 0x08) != 0;
    if (second_byte & 0x80) return ESHKOL_WS_FRAME_MASKED_FROM_SERVER;
    if (first_byte & 0x70) return ESHKOL_WS_FRAME_RESERVED_BITS_SET;
    if ((opcode >= 0x03 && opcode <= 0x07) || opcode >= 0x0b)
        return ESHKOL_WS_FRAME_RESERVED_OPCODE;
    if (control) {
        if (length > 125) return ESHKOL_WS_FRAME_CONTROL_TOO_LARGE;
        if (!fin) return ESHKOL_WS_FRAME_CONTROL_FRAGMENTED;
        return ESHKOL_WS_FRAME_OK;
    }
    if (opcode == 0x00 && !in_fragment) return ESHKOL_WS_FRAME_BAD_CONTINUATION;
    if (opcode != 0x00 && in_fragment) return ESHKOL_WS_FRAME_BAD_CONTINUATION;
    if (length > (uint64_t)eshkol_ws_max_frame_bytes())
        return ESHKOL_WS_FRAME_TOO_LARGE;
    return ESHKOL_WS_FRAME_OK;
}

/** @brief The RFC 6455 section 7.4 close code a verdict is reported with. */
static inline int eshkol_ws_frame_close_code(eshkol_ws_frame_verdict verdict) {
    switch (verdict) {
        case ESHKOL_WS_FRAME_OK: return 0;
        case ESHKOL_WS_FRAME_TOO_LARGE: return 1009;   /* message too big */
        default: return 1002;                          /* protocol error */
    }
}

/**
 * @brief True when `length` bytes at `text` are well-formed UTF-8.
 *
 * RFC 6455 section 8.1 requires a client to fail the connection on a text
 * frame whose payload is not valid UTF-8. Overlong encodings, surrogates and
 * values above U+10FFFF are refused, so the check is the strict one the RFC
 * points at rather than a length-only scan.
 */
static inline int eshkol_ws_valid_utf8(const char* text, size_t length) {
    size_t i = 0;
    if (!text) return length == 0;
    while (i < length) {
        const unsigned char c = (unsigned char)text[i];
        size_t extra;
        uint32_t code;
        if (c < 0x80) { i++; continue; }
        if ((c & 0xe0) == 0xc0) { extra = 1; code = (uint32_t)(c & 0x1f); }
        else if ((c & 0xf0) == 0xe0) { extra = 2; code = (uint32_t)(c & 0x0f); }
        else if ((c & 0xf8) == 0xf0) { extra = 3; code = (uint32_t)(c & 0x07); }
        else return 0;
        if (i + extra >= length) return 0;
        {
            size_t k;
            for (k = 1; k <= extra; ++k) {
                const unsigned char cc = (unsigned char)text[i + k];
                if ((cc & 0xc0) != 0x80) return 0;
                code = (code << 6) | (uint32_t)(cc & 0x3f);
            }
        }
        if (extra == 1 && code < 0x80) return 0;
        if (extra == 2 && code < 0x800) return 0;
        if (extra == 3 && code < 0x10000) return 0;
        if (code > 0x10ffff) return 0;
        if (code >= 0xd800 && code <= 0xdfff) return 0;
        i += extra + 1;
    }
    return 1;
}

#endif /* ESHKOL_CORE_WEBSOCKET_HANDSHAKE_H */
