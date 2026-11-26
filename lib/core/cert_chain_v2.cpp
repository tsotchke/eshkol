/*
 * Chained eshkol.certificate.v2 Derivation Verifier & Lean 4.12.0 Exporter
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/cert_chain_v2.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

/* ============================================================================
 * Self-Contained FIPS 180-4 SHA-256
 * ============================================================================ */

constexpr uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

struct Sha256Ctx {
    uint32_t h[8];
    uint8_t block[64];
    size_t block_len;
    uint64_t total_bits;

    void init() {
        h[0] = 0x6a09e667u; h[1] = 0xbb67ae85u;
        h[2] = 0x3c6ef372u; h[3] = 0xa54ff53au;
        h[4] = 0x510e527fu; h[5] = 0x9b05688cu;
        h[6] = 0x1f83d9abu; h[7] = 0x5be0cd19u;
        block_len = 0;
        total_bits = 0;
    }

    void compress(const uint8_t b[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(b[4 * i]) << 24) |
                   (static_cast<uint32_t>(b[4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(b[4 * i + 2]) << 8) |
                   static_cast<uint32_t>(b[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0], bb = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = hh + S1 + ch + kSha256K[i] + w[i];
            uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & bb) ^ (a & c) ^ (bb & c);
            uint32_t temp2 = S0 + maj;

            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = bb;
            bb = a;
            a = temp1 + temp2;
        }

        h[0] += a; h[1] += bb; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f;  h[6] += g; h[7] += hh;
    }

    void update(const uint8_t* data, size_t len) {
        if (!data || len == 0) return;
        total_bits += static_cast<uint64_t>(len) * 8ULL;
        size_t pos = 0;
        if (block_len > 0) {
            size_t take = 64 - block_len;
            if (take > len) take = len;
            std::memcpy(block + block_len, data, take);
            block_len += take;
            pos += take;
            if (block_len == 64) {
                compress(block);
                block_len = 0;
            }
        }
        while (pos + 64 <= len) {
            compress(data + pos);
            pos += 64;
        }
        if (pos < len) {
            block_len = len - pos;
            std::memcpy(block, data + pos, block_len);
        }
    }

    void final(uint8_t out[32]) {
        block[block_len++] = 0x80u;
        if (block_len > 56) {
            while (block_len < 64) block[block_len++] = 0;
            compress(block);
            block_len = 0;
        }
        while (block_len < 56) block[block_len++] = 0;
        for (int i = 7; i >= 0; --i) {
            block[56 + (7 - i)] = static_cast<uint8_t>((total_bits >> (8 * i)) & 0xFFu);
        }
        compress(block);
        for (int i = 0; i < 8; ++i) {
            out[4 * i]     = static_cast<uint8_t>((h[i] >> 24) & 0xFFu);
            out[4 * i + 1] = static_cast<uint8_t>((h[i] >> 16) & 0xFFu);
            out[4 * i + 2] = static_cast<uint8_t>((h[i] >> 8) & 0xFFu);
            out[4 * i + 3] = static_cast<uint8_t>(h[i] & 0xFFu);
        }
    }
};

/* ============================================================================
 * Canonical Little-Endian Serialization Helpers
 * ============================================================================ */

void append_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void append_u64_le(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void append_f64_le(std::vector<uint8_t>& buf, double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(double));
    append_u64_le(buf, bits);
}

void append_bytes(std::vector<uint8_t>& buf, const void* ptr, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    buf.insert(buf.end(), p, p + len);
}

void compute_link1_hash(const eshkol_cert_chain_v2_t* chain, uint8_t out[32]) {
    std::vector<uint8_t> buf;
    buf.reserve(256);
    const char tag[] = "LINK1_PROGRAM_V2|";
    append_bytes(buf, tag, sizeof(tag) - 1);
    append_bytes(buf, chain->context_id, ESHKOL_CERT_V2_CONTEXT_MAX);
    append_bytes(buf, chain->program_link.program_digest, ESHKOL_CERT_V2_HASH_BYTES);
    append_u32_le(buf, chain->program_link.input_dim);
    append_u32_le(buf, chain->program_link.output_dim);
    for (uint32_t i = 0; i < chain->program_link.input_dim; ++i) {
        append_f64_le(buf, chain->program_link.input_box[i].lo);
        append_f64_le(buf, chain->program_link.input_box[i].hi);
    }
    eshkol_cert_sha256(buf.data(), buf.size(), out);
}

void compute_link2_hash(const eshkol_cert_chain_v2_t* chain, uint8_t out[32]) {
    std::vector<uint8_t> buf;
    buf.reserve(256);
    const char tag[] = "LINK2_TAYLOR_V2|";
    append_bytes(buf, tag, sizeof(tag) - 1);
    append_bytes(buf, chain->taylor_link.prev_hash, ESHKOL_CERT_V2_HASH_BYTES);
    for (uint32_t i = 0; i < chain->program_link.output_dim; ++i) {
        append_f64_le(buf, chain->taylor_link.certified_bounds[i].lo);
        append_f64_le(buf, chain->taylor_link.certified_bounds[i].hi);
    }
    eshkol_cert_sha256(buf.data(), buf.size(), out);
}

void compute_link3_hash(const eshkol_cert_chain_v2_t* chain, uint8_t out[32]) {
    std::vector<uint8_t> buf;
    buf.reserve(256);
    const char tag[] = "LINK3_SAFETY_V2|";
    append_bytes(buf, tag, sizeof(tag) - 1);
    append_bytes(buf, chain->safety_link.prev_hash, ESHKOL_CERT_V2_HASH_BYTES);
    for (uint32_t i = 0; i < chain->program_link.output_dim; ++i) {
        append_f64_le(buf, chain->safety_link.safe_bounds[i].lo);
        append_f64_le(buf, chain->safety_link.safe_bounds[i].hi);
    }
    eshkol_cert_sha256(buf.data(), buf.size(), out);
}

void compute_envelope_hmac(const eshkol_cert_chain_v2_t* chain,
                           const uint8_t* signing_key,
                           size_t key_len,
                           uint8_t out_mac[32]) {
    std::vector<uint8_t> buf;
    buf.reserve(256);
    const char tag[] = "CERT_V2_ENV|";
    append_bytes(buf, tag, sizeof(tag) - 1);
    append_bytes(buf, chain->context_id, ESHKOL_CERT_V2_CONTEXT_MAX);
    append_u64_le(buf, chain->issued_epoch);
    append_u64_le(buf, chain->valid_until_epoch);
    append_bytes(buf, chain->program_link.link_hash, ESHKOL_CERT_V2_HASH_BYTES);
    append_bytes(buf, chain->taylor_link.link_hash, ESHKOL_CERT_V2_HASH_BYTES);
    append_bytes(buf, chain->safety_link.link_hash, ESHKOL_CERT_V2_HASH_BYTES);
    eshkol_cert_hmac_sha256(signing_key, key_len, buf.data(), buf.size(), out_mac);
}

bool ct_memeq(const uint8_t* a, const uint8_t* b, size_t n) {
    uint8_t diff = 0;
    for (size_t i = 0; i < n; ++i) diff |= static_cast<uint8_t>(a[i] ^ b[i]);
    return diff == 0;
}

bool check_context_str_valid(const char* s) {
    if (!s || s[0] == '\0') return false;
    size_t len = 0;
    while (len < ESHKOL_CERT_V2_CONTEXT_MAX && s[len] != '\0') ++len;
    return len > 0 && len < ESHKOL_CERT_V2_CONTEXT_MAX;
}

std::string hex_encode(const uint8_t* data, size_t len) {
    static const char kHex[] = "0123456789abcdef";
    std::string s;
    s.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        s.push_back(kHex[(data[i] >> 4) & 0x0Fu]);
        s.push_back(kHex[data[i] & 0x0Fu]);
    }
    return s;
}

std::string format_lean_int(int64_t v) {
    char tmp[64];
    if (v < 0) {
        std::snprintf(tmp, sizeof(tmp), "(%lld : Int)", (long long)v);
    } else {
        std::snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
    }
    return std::string(tmp);
}

} /* namespace */

extern "C" {

void eshkol_cert_sha256(const uint8_t* data, size_t len, uint8_t out_hash[32]) {
    if (!out_hash) return;
    Sha256Ctx ctx;
    ctx.init();
    if (data && len > 0) ctx.update(data, len);
    ctx.final(out_hash);
}

void eshkol_cert_hmac_sha256(const uint8_t* key, size_t key_len,
                             const uint8_t* data, size_t data_len,
                             uint8_t out_mac[32]) {
    if (!out_mac) return;
    uint8_t k_block[64];
    std::memset(k_block, 0, sizeof(k_block));
    if (key && key_len > 0) {
        if (key_len > 64) {
            eshkol_cert_sha256(key, key_len, k_block);
        } else {
            std::memcpy(k_block, key, key_len);
        }
    }

    uint8_t ipad[64];
    uint8_t opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = static_cast<uint8_t>(k_block[i] ^ 0x36u);
        opad[i] = static_cast<uint8_t>(k_block[i] ^ 0x5Cu);
    }

    uint8_t inner_hash[32];
    Sha256Ctx inner;
    inner.init();
    inner.update(ipad, 64);
    if (data && data_len > 0) inner.update(data, data_len);
    inner.final(inner_hash);

    Sha256Ctx outer;
    outer.init();
    outer.update(opad, 64);
    outer.update(inner_hash, 32);
    outer.final(out_mac);
}

eshkol_cert_v2_status_t eshkol_cert_chain_v2_build(
    const uint8_t* program_bytes,
    size_t program_len,
    const eshkol_tm_world_model_t* model,
    const eshkol_tm_interval_t* input_box,
    const eshkol_tm_interval_t* safe_bounds,
    const char* context_id,
    uint64_t issued_epoch,
    uint64_t valid_until_epoch,
    const uint8_t* signing_key,
    size_t key_len,
    eshkol_cert_chain_v2_t* out_chain) {
    if (!program_bytes || program_len == 0 || !model || !input_box || !safe_bounds ||
        !check_context_str_valid(context_id) || !signing_key || key_len == 0 || !out_chain) {
        return ESHKOL_CERT_ERR_MALFORMED;
    }

    const uint32_t in_dim = eshkol_tm_model_input_dim(model);
    const uint32_t out_dim = eshkol_tm_model_output_dim(model);
    if (in_dim == 0 || in_dim > ESHKOL_CERT_V2_MAX_DIMS ||
        out_dim == 0 || out_dim > ESHKOL_CERT_V2_MAX_DIMS) {
        return ESHKOL_CERT_ERR_MALFORMED;
    }

    std::memset(out_chain, 0, sizeof(*out_chain));
    out_chain->magic = ESHKOL_CERT_V2_MAGIC;
    out_chain->version = ESHKOL_CERT_V2_VERSION;
    std::strncpy(out_chain->context_id, context_id, ESHKOL_CERT_V2_CONTEXT_MAX - 1);
    out_chain->issued_epoch = issued_epoch;
    out_chain->valid_until_epoch = valid_until_epoch;

    /* Link 1: ProgramLink */
    std::vector<uint8_t> prog_payload;
    const char prog_tag[] = "PROGRAM_V2|";
    append_bytes(prog_payload, prog_tag, sizeof(prog_tag) - 1);
    append_bytes(prog_payload, program_bytes, program_len);
    eshkol_cert_sha256(prog_payload.data(), prog_payload.size(),
                       out_chain->program_link.program_digest);

    out_chain->program_link.input_dim = in_dim;
    out_chain->program_link.output_dim = out_dim;
    for (uint32_t i = 0; i < in_dim; ++i) {
        if (!std::isfinite(input_box[i].lo) || !std::isfinite(input_box[i].hi) ||
            input_box[i].lo > input_box[i].hi) {
            return ESHKOL_CERT_ERR_MALFORMED;
        }
        out_chain->program_link.input_box[i] = input_box[i];
    }
    compute_link1_hash(out_chain, out_chain->program_link.link_hash);

    /* Link 2: TaylorEnclosureLink via CE-8 order-2 Interval-Taylor certification */
    std::memcpy(out_chain->taylor_link.prev_hash,
                out_chain->program_link.link_hash,
                ESHKOL_CERT_V2_HASH_BYTES);
    eshkol_tm_status_t tm_st = eshkol_tm_model_certify(
        model, input_box, out_chain->taylor_link.certified_bounds);
    if (tm_st != ESHKOL_TM_OK) {
        return ESHKOL_CERT_ERR_POSTCONDITION_VIOLATION;
    }
    compute_link2_hash(out_chain, out_chain->taylor_link.link_hash);

    /* Link 3: SafetyPostconditionLink */
    std::memcpy(out_chain->safety_link.prev_hash,
                out_chain->taylor_link.link_hash,
                ESHKOL_CERT_V2_HASH_BYTES);
    for (uint32_t i = 0; i < out_dim; ++i) {
        if (!std::isfinite(safe_bounds[i].lo) || !std::isfinite(safe_bounds[i].hi) ||
            safe_bounds[i].lo > safe_bounds[i].hi) {
            return ESHKOL_CERT_ERR_MALFORMED;
        }
        out_chain->safety_link.safe_bounds[i] = safe_bounds[i];
    }
    compute_link3_hash(out_chain, out_chain->safety_link.link_hash);

    compute_envelope_hmac(out_chain, signing_key, key_len, out_chain->signature_hmac);
    return ESHKOL_CERT_OK;
}

eshkol_cert_v2_status_t eshkol_cert_chain_v2_resign(
    eshkol_cert_chain_v2_t* chain,
    const char* new_context_id,
    uint64_t new_issued_epoch,
    uint64_t new_valid_until_epoch,
    const uint8_t* signing_key,
    size_t key_len) {
    if (!chain || chain->magic != ESHKOL_CERT_V2_MAGIC ||
        chain->version != ESHKOL_CERT_V2_VERSION ||
        !check_context_str_valid(new_context_id) ||
        !signing_key || key_len == 0) {
        return ESHKOL_CERT_ERR_MALFORMED;
    }
    std::memset(chain->context_id, 0, ESHKOL_CERT_V2_CONTEXT_MAX);
    std::strncpy(chain->context_id, new_context_id, ESHKOL_CERT_V2_CONTEXT_MAX - 1);
    chain->issued_epoch = new_issued_epoch;
    chain->valid_until_epoch = new_valid_until_epoch;

    compute_link1_hash(chain, chain->program_link.link_hash);
    std::memcpy(chain->taylor_link.prev_hash, chain->program_link.link_hash, ESHKOL_CERT_V2_HASH_BYTES);
    compute_link2_hash(chain, chain->taylor_link.link_hash);
    std::memcpy(chain->safety_link.prev_hash, chain->taylor_link.link_hash, ESHKOL_CERT_V2_HASH_BYTES);
    compute_link3_hash(chain, chain->safety_link.link_hash);
    compute_envelope_hmac(chain, signing_key, key_len, chain->signature_hmac);
    return ESHKOL_CERT_OK;
}

eshkol_cert_v2_status_t eshkol_cert_chain_v2_verify(
    const eshkol_cert_chain_v2_t* chain,
    const char* expected_context_id,
    uint64_t current_epoch,
    const uint8_t* signing_key,
    size_t key_len) {
    /* 1. Structural validity */
    if (!chain || !expected_context_id || !signing_key || key_len == 0) {
        return ESHKOL_CERT_ERR_MALFORMED;
    }
    if (chain->magic != ESHKOL_CERT_V2_MAGIC || chain->version != ESHKOL_CERT_V2_VERSION) {
        return ESHKOL_CERT_ERR_MALFORMED;
    }
    const uint32_t in_dim = chain->program_link.input_dim;
    const uint32_t out_dim = chain->program_link.output_dim;
    if (in_dim == 0 || in_dim > ESHKOL_CERT_V2_MAX_DIMS ||
        out_dim == 0 || out_dim > ESHKOL_CERT_V2_MAX_DIMS) {
        return ESHKOL_CERT_ERR_MALFORMED;
    }
    bool ctx_terminated = false;
    for (size_t i = 0; i < ESHKOL_CERT_V2_CONTEXT_MAX; ++i) {
        if (chain->context_id[i] == '\0') { ctx_terminated = true; break; }
    }
    if (!ctx_terminated || chain->context_id[0] == '\0') {
        return ESHKOL_CERT_ERR_MALFORMED;
    }
    for (uint32_t i = 0; i < in_dim; ++i) {
        if (!std::isfinite(chain->program_link.input_box[i].lo) ||
            !std::isfinite(chain->program_link.input_box[i].hi) ||
            chain->program_link.input_box[i].lo > chain->program_link.input_box[i].hi) {
            return ESHKOL_CERT_ERR_MALFORMED;
        }
    }
    for (uint32_t i = 0; i < out_dim; ++i) {
        if (!std::isfinite(chain->taylor_link.certified_bounds[i].lo) ||
            !std::isfinite(chain->taylor_link.certified_bounds[i].hi) ||
            !std::isfinite(chain->safety_link.safe_bounds[i].lo) ||
            !std::isfinite(chain->safety_link.safe_bounds[i].hi) ||
            chain->taylor_link.certified_bounds[i].lo > chain->taylor_link.certified_bounds[i].hi ||
            chain->safety_link.safe_bounds[i].lo > chain->safety_link.safe_bounds[i].hi) {
            return ESHKOL_CERT_ERR_MALFORMED;
        }
    }

    /* 2. Cryptographic SHA-256 link chain + HMAC-SHA256 integrity */
    uint8_t h1[32], h2[32], h3[32], mac[32];
    compute_link1_hash(chain, h1);
    if (!ct_memeq(h1, chain->program_link.link_hash, 32)) {
        return ESHKOL_CERT_ERR_HASH_CHAIN_BREAK;
    }
    if (!ct_memeq(chain->taylor_link.prev_hash, chain->program_link.link_hash, 32)) {
        return ESHKOL_CERT_ERR_HASH_CHAIN_BREAK;
    }
    compute_link2_hash(chain, h2);
    if (!ct_memeq(h2, chain->taylor_link.link_hash, 32)) {
        return ESHKOL_CERT_ERR_HASH_CHAIN_BREAK;
    }
    if (!ct_memeq(chain->safety_link.prev_hash, chain->taylor_link.link_hash, 32)) {
        return ESHKOL_CERT_ERR_HASH_CHAIN_BREAK;
    }
    compute_link3_hash(chain, h3);
    if (!ct_memeq(h3, chain->safety_link.link_hash, 32)) {
        return ESHKOL_CERT_ERR_HASH_CHAIN_BREAK;
    }
    compute_envelope_hmac(chain, signing_key, key_len, mac);
    if (!ct_memeq(mac, chain->signature_hmac, 32)) {
        return ESHKOL_CERT_ERR_HASH_CHAIN_BREAK;
    }

    /* 3. Context binding */
    if (std::strncmp(chain->context_id, expected_context_id, ESHKOL_CERT_V2_CONTEXT_MAX) != 0) {
        return ESHKOL_CERT_ERR_CONTEXT_MISMATCH;
    }

    /* 4. Key validity epoch window */
    if (chain->valid_until_epoch < chain->issued_epoch ||
        current_epoch < chain->issued_epoch ||
        current_epoch > chain->valid_until_epoch) {
        return ESHKOL_CERT_ERR_EXPIRED_KEY;
    }

    /* 5. Safety postcondition enclosure check */
    for (uint32_t i = 0; i < out_dim; ++i) {
        if (chain->taylor_link.certified_bounds[i].lo < chain->safety_link.safe_bounds[i].lo ||
            chain->taylor_link.certified_bounds[i].hi > chain->safety_link.safe_bounds[i].hi) {
            return ESHKOL_CERT_ERR_POSTCONDITION_VIOLATION;
        }
    }

    return ESHKOL_CERT_OK;
}

bool eshkol_cert_chain_export_lean4(
    const eshkol_cert_chain_v2_t* chain,
    const char* expected_context_id,
    uint64_t eval_epoch,
    const char* module_namespace,
    char* out_buf,
    size_t out_cap,
    size_t* out_len) {
    if (!chain || !expected_context_id || !module_namespace || !out_buf || out_cap == 0) {
        return false;
    }
    const uint32_t out_dim = chain->program_link.output_dim;
    if (out_dim == 0 || out_dim > ESHKOL_CERT_V2_MAX_DIMS) return false;

    const double kScale = 1e9;
    std::string lean;
    lean.reserve(4096);

    lean += "/-\n";
    lean += "  Eshkol CE-9 Chained eshkol.certificate.v2 Derivation Proof\n";
    lean += "  Verified by Lean 4.12.0 kernel via decidable reflection (self-contained).\n";
    lean += "  Scaled-integer intervals use conservative outward rounding (10^9 scale).\n";
    lean += "-/\n\n";
    lean += "namespace ";
    lean += module_namespace;
    lean += "\n\n";

    lean += "structure IntInterval where\n";
    lean += "  lo : Int\n";
    lean += "  hi : Int\n";
    lean += "  deriving DecidableEq, Repr\n\n";

    lean += "structure ProgramLink where\n";
    lean += "  programDigest : String\n";
    lean += "  linkHash      : String\n";
    lean += "  deriving DecidableEq, Repr\n\n";

    lean += "structure TaylorEnclosureLink where\n";
    lean += "  prevHash        : String\n";
    lean += "  certifiedBounds : List IntInterval\n";
    lean += "  linkHash        : String\n";
    lean += "  deriving DecidableEq, Repr\n\n";

    lean += "structure SafetyPostconditionLink where\n";
    lean += "  prevHash   : String\n";
    lean += "  safeBounds : List IntInterval\n";
    lean += "  linkHash   : String\n";
    lean += "  deriving DecidableEq, Repr\n\n";

    lean += "structure CertChainV2 where\n";
    lean += "  version         : Nat\n";
    lean += "  contextId       : String\n";
    lean += "  issuedEpoch     : Nat\n";
    lean += "  validUntilEpoch : Nat\n";
    lean += "  link1           : ProgramLink\n";
    lean += "  link2           : TaylorEnclosureLink\n";
    lean += "  link3           : SafetyPostconditionLink\n";
    lean += "  signatureHmac   : String\n";
    lean += "  deriving DecidableEq, Repr\n\n";

    lean += "def intervalEnclosed (cert safe : IntInterval) : Bool :=\n";
    lean += "  decide (safe.lo ≤ cert.lo ∧ cert.lo ≤ cert.hi ∧ cert.hi ≤ safe.hi)\n\n";

    lean += "def allBoundsSafe : List IntInterval → List IntInterval → Bool\n";
    lean += "  | [], [] => true\n";
    lean += "  | c :: cs, s :: ss => intervalEnclosed c s && allBoundsSafe cs ss\n";
    lean += "  | _, _ => false\n\n";

    lean += "def verifyCertChainV2 (c : CertChainV2) (expectedCtx : String) (evalEpoch : Nat) : Bool :=\n";
    lean += "  (c.version == 2) &&\n";
    lean += "  (c.link2.prevHash == c.link1.linkHash) &&\n";
    lean += "  (c.link3.prevHash == c.link2.linkHash) &&\n";
    lean += "  (c.contextId == expectedCtx) &&\n";
    lean += "  decide (c.issuedEpoch ≤ evalEpoch ∧ evalEpoch ≤ c.validUntilEpoch) &&\n";
    lean += "  decide (c.link2.certifiedBounds.length > 0) &&\n";
    lean += "  allBoundsSafe c.link2.certifiedBounds c.link3.safeBounds\n\n";

    lean += "def expectedContext : String := \"";
    lean += expected_context_id;
    lean += "\"\n\n";

    char num_buf[64];
    std::snprintf(num_buf, sizeof(num_buf), "%llu", (unsigned long long)eval_epoch);
    lean += "def evalEpoch : Nat := ";
    lean += num_buf;
    lean += "\n\n";

    lean += "def exportedCertChain : CertChainV2 := {\n";
    lean += "  version := 2,\n";
    lean += "  contextId := \"";
    lean += chain->context_id;
    lean += "\",\n";
    std::snprintf(num_buf, sizeof(num_buf), "%llu", (unsigned long long)chain->issued_epoch);
    lean += "  issuedEpoch := ";
    lean += num_buf;
    lean += ",\n";
    std::snprintf(num_buf, sizeof(num_buf), "%llu", (unsigned long long)chain->valid_until_epoch);
    lean += "  validUntilEpoch := ";
    lean += num_buf;
    lean += ",\n";

    lean += "  link1 := {\n";
    lean += "    programDigest := \"" + hex_encode(chain->program_link.program_digest, 32) + "\",\n";
    lean += "    linkHash := \"" + hex_encode(chain->program_link.link_hash, 32) + "\"\n";
    lean += "  },\n";

    lean += "  link2 := {\n";
    lean += "    prevHash := \"" + hex_encode(chain->taylor_link.prev_hash, 32) + "\",\n";
    lean += "    certifiedBounds := [\n";
    for (uint32_t i = 0; i < out_dim; ++i) {
        /* Conservative outward rounding for certified bounds: floor(lo), ceil(hi). */
        int64_t lo_i = static_cast<int64_t>(std::floor(chain->taylor_link.certified_bounds[i].lo * kScale));
        int64_t hi_i = static_cast<int64_t>(std::ceil(chain->taylor_link.certified_bounds[i].hi * kScale));
        lean += "      { lo := " + format_lean_int(lo_i) + ", hi := " + format_lean_int(hi_i) + " }";
        if (i + 1 < out_dim) lean += ",";
        lean += "\n";
    }
    lean += "    ],\n";
    lean += "    linkHash := \"" + hex_encode(chain->taylor_link.link_hash, 32) + "\"\n";
    lean += "  },\n";

    lean += "  link3 := {\n";
    lean += "    prevHash := \"" + hex_encode(chain->safety_link.prev_hash, 32) + "\",\n";
    lean += "    safeBounds := [\n";
    for (uint32_t i = 0; i < out_dim; ++i) {
        /* Conservative inward rounding for allowed safety box: ceil(lo), floor(hi). */
        int64_t lo_i = static_cast<int64_t>(std::ceil(chain->safety_link.safe_bounds[i].lo * kScale));
        int64_t hi_i = static_cast<int64_t>(std::floor(chain->safety_link.safe_bounds[i].hi * kScale));
        lean += "      { lo := " + format_lean_int(lo_i) + ", hi := " + format_lean_int(hi_i) + " }";
        if (i + 1 < out_dim) lean += ",";
        lean += "\n";
    }
    lean += "    ],\n";
    lean += "    linkHash := \"" + hex_encode(chain->safety_link.link_hash, 32) + "\"\n";
    lean += "  },\n";
    lean += "  signatureHmac := \"" + hex_encode(chain->signature_hmac, 32) + "\"\n";
    lean += "}\n\n";

    lean += "theorem cert_chain_v2_verified :\n";
    lean += "    verifyCertChainV2 exportedCertChain expectedContext evalEpoch = true := by\n";
    lean += "  decide\n\n";

    lean += "theorem cert_chain_hash_linkage :\n";
    lean += "    exportedCertChain.link2.prevHash = exportedCertChain.link1.linkHash ∧\n";
    lean += "    exportedCertChain.link3.prevHash = exportedCertChain.link2.linkHash := by\n";
    lean += "  decide\n\n";

    lean += "theorem cert_chain_safety_enclosure :\n";
    lean += "    allBoundsSafe exportedCertChain.link2.certifiedBounds exportedCertChain.link3.safeBounds = true := by\n";
    lean += "  decide\n\n";

    lean += "end ";
    lean += module_namespace;
    lean += "\n";

    if (out_len) *out_len = lean.size();
    if (lean.size() + 1 > out_cap) return false;
    std::memcpy(out_buf, lean.data(), lean.size() + 1);
    return true;
}

} /* extern "C" */
