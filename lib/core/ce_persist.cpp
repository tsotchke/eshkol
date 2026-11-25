/*
 * Fail-Closed Versioned Binary Persistence for CE-2 PROOF_DAG & CE-3 E-Graphs
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/ce_persist.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace {

constexpr uint8_t kMagic[4] = {'E', 'C', 'E', '3'};
constexpr uint32_t kKindProofDag = 1u;
constexpr uint32_t kKindEGraph   = 2u;

uint32_t fnv1a_32(const uint8_t* data, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= static_cast<uint32_t>(data[i]);
        h *= 16777619u;
    }
    return h;
}

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

void put_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        out.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFFu));
    }
}

void put_f64(std::vector<uint8_t>& out, double d) {
    uint64_t bits = 0;
    std::memcpy(&bits, &d, sizeof(bits));
    put_u64(out, bits);
}

struct ByteReader {
    const uint8_t* ptr = nullptr;
    size_t len = 0;
    size_t pos = 0;

    bool read_u32(uint32_t& out) {
        if (pos + 4u > len) return false;
        out = static_cast<uint32_t>(ptr[pos]) |
              (static_cast<uint32_t>(ptr[pos + 1u]) << 8) |
              (static_cast<uint32_t>(ptr[pos + 2u]) << 16) |
              (static_cast<uint32_t>(ptr[pos + 3u]) << 24);
        pos += 4u;
        return true;
    }

    bool read_u64(uint64_t& out) {
        if (pos + 8u > len) return false;
        out = 0;
        for (int i = 0; i < 8; i++) {
            out |= (static_cast<uint64_t>(ptr[pos + static_cast<size_t>(i)]) << (i * 8));
        }
        pos += 8u;
        return true;
    }

    bool read_f64(double& out) {
        uint64_t bits = 0;
        if (!read_u64(bits)) return false;
        std::memcpy(&out, &bits, sizeof(out));
        return true;
    }

    bool read_bytes(std::string& out, uint32_t n) {
        if (pos + n > len) return false;
        out.assign(reinterpret_cast<const char*>(ptr + pos), n);
        pos += n;
        return true;
    }

    bool at_end() const { return pos == len; }
};

eshkol_persist_status_t wrap_envelope(uint32_t payload_kind,
                                      const std::vector<uint8_t>& payload,
                                      uint8_t** out_buf,
                                      size_t* out_len) {
    if (!out_buf || !out_len) return ESHKOL_PERSIST_ERR_INVALID_ARG;
    if (payload.size() > UINT32_MAX - ESHKOL_CE_PERSIST_HEADER_BYTES) {
        return ESHKOL_PERSIST_ERR_INVALID_ARG;
    }

    std::vector<uint8_t> frame;
    frame.reserve(ESHKOL_CE_PERSIST_HEADER_BYTES + payload.size());
    frame.insert(frame.end(), kMagic, kMagic + 4);
    put_u32(frame, ESHKOL_CE_PERSIST_SCHEMA_V1);
    put_u32(frame, payload_kind);
    put_u32(frame, static_cast<uint32_t>(payload.size()));

    /* Compute CRC-32 over header[0..15] || payload */
    std::vector<uint8_t> crc_input;
    crc_input.reserve(16u + payload.size());
    crc_input.insert(crc_input.end(), frame.begin(), frame.begin() + 16);
    crc_input.insert(crc_input.end(), payload.begin(), payload.end());
    uint32_t crc = eshkol_ce_crc32(crc_input.data(), crc_input.size());
    uint32_t fnv = fnv1a_32(payload.empty() ? nullptr : payload.data(), payload.size());

    put_u32(frame, crc);
    put_u32(frame, fnv);
    frame.insert(frame.end(), payload.begin(), payload.end());

    auto* raw = static_cast<uint8_t*>(std::malloc(frame.size()));
    if (!raw) return ESHKOL_PERSIST_ERR_IO;
    std::memcpy(raw, frame.data(), frame.size());
    *out_buf = raw;
    *out_len = frame.size();
    return ESHKOL_PERSIST_OK;
}

eshkol_persist_status_t validate_envelope(const uint8_t* buf,
                                          size_t len,
                                          uint32_t expected_kind,
                                          ByteReader& payload_reader) {
    if (!buf) return ESHKOL_PERSIST_ERR_INVALID_ARG;
    if (len < ESHKOL_CE_PERSIST_HEADER_BYTES) return ESHKOL_PERSIST_ERR_TRUNCATED;

    if (std::memcmp(buf, kMagic, 4) != 0) return ESHKOL_PERSIST_ERR_BAD_MAGIC;

    ByteReader hdr{buf + 4, 20u, 0};
    uint32_t version = 0, kind = 0, payload_bytes = 0, stored_crc = 0, stored_fnv = 0;
    hdr.read_u32(version);
    hdr.read_u32(kind);
    hdr.read_u32(payload_bytes);
    hdr.read_u32(stored_crc);
    hdr.read_u32(stored_fnv);

    if (version != ESHKOL_CE_PERSIST_SCHEMA_V1) return ESHKOL_PERSIST_ERR_BAD_VERSION;
    if (kind != expected_kind) return ESHKOL_PERSIST_ERR_BAD_KIND;
    if (static_cast<size_t>(payload_bytes) + ESHKOL_CE_PERSIST_HEADER_BYTES != len) {
        return ESHKOL_PERSIST_ERR_TRUNCATED;
    }

    const uint8_t* payload_ptr = buf + ESHKOL_CE_PERSIST_HEADER_BYTES;
    std::vector<uint8_t> crc_input;
    crc_input.reserve(16u + payload_bytes);
    crc_input.insert(crc_input.end(), buf, buf + 16);
    crc_input.insert(crc_input.end(), payload_ptr, payload_ptr + payload_bytes);
    uint32_t calc_crc = eshkol_ce_crc32(crc_input.data(), crc_input.size());
    uint32_t calc_fnv = fnv1a_32(payload_bytes > 0 ? payload_ptr : nullptr, payload_bytes);
    if (calc_crc != stored_crc || calc_fnv != stored_fnv) {
        return ESHKOL_PERSIST_ERR_CHECKSUM;
    }

    payload_reader = {payload_ptr, payload_bytes, 0};
    return ESHKOL_PERSIST_OK;
}

} /* anonymous namespace */

extern "C" uint32_t eshkol_ce_crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int bit = 0; bit < 8; bit++) {
            uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

extern "C" void eshkol_ce_buffer_free(uint8_t* buf) {
    std::free(buf);
}

extern "C" eshkol_persist_status_t eshkol_ce_serialize_proof_dag(const eshkol_proof_dag_t* dag,
                                                                 uint8_t** out_buf,
                                                                 size_t* out_len) {
    if (!dag || !out_buf || !out_len) return ESHKOL_PERSIST_ERR_INVALID_ARG;
    if (!eshkol_proof_dag_verify_replay(dag)) return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;

    std::vector<uint8_t> payload;
    const uint32_t n = eshkol_proof_dag_num_nodes(dag);
    put_u32(payload, n);

    for (uint32_t i = 0; i < n; i++) {
        eshkol_proof_node_kind_t kind{};
        const char* label = nullptr;
        const uint32_t* parents = nullptr;
        uint32_t num_parents = 0;
        eshkol_semiring_val_t val{};
        uint64_t hash = 0;
        if (!eshkol_proof_dag_inspect_node(dag, i, &kind, &label, &parents,
                                           &num_parents, &val, &hash)) {
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
        uint32_t lab_len = label ? static_cast<uint32_t>(std::strlen(label)) : 0u;
        put_u32(payload, static_cast<uint32_t>(kind));
        put_u32(payload, lab_len);
        if (lab_len > 0) {
            payload.insert(payload.end(),
                           reinterpret_cast<const uint8_t*>(label),
                           reinterpret_cast<const uint8_t*>(label) + lab_len);
        }
        put_u32(payload, static_cast<uint32_t>(val.kind));
        put_f64(payload, val.a);
        put_f64(payload, val.b);
        put_u32(payload, val.disc);
        put_u32(payload, num_parents);
        for (uint32_t p = 0; p < num_parents; p++) put_u32(payload, parents[p]);
        put_u64(payload, hash);
    }

    return wrap_envelope(kKindProofDag, payload, out_buf, out_len);
}

extern "C" eshkol_persist_status_t eshkol_ce_deserialize_proof_dag(const uint8_t* buf,
                                                                   size_t len,
                                                                   eshkol_proof_dag_t** out_dag) {
    if (!out_dag) return ESHKOL_PERSIST_ERR_INVALID_ARG;
    *out_dag = nullptr;

    ByteReader r{};
    eshkol_persist_status_t st = validate_envelope(buf, len, kKindProofDag, r);
    if (st != ESHKOL_PERSIST_OK) return st;

    uint32_t n = 0;
    if (!r.read_u32(n)) return ESHKOL_PERSIST_ERR_TRUNCATED;

    eshkol_proof_dag_t* dag = eshkol_proof_dag_create();
    if (!dag) return ESHKOL_PERSIST_ERR_IO;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t kind_u = 0, lab_len = 0, sr_u = 0, disc = 0, num_parents = 0;
        double a = 0.0, b = 0.0;
        uint64_t expected_hash = 0;
        std::string label;

        if (!r.read_u32(kind_u) || !r.read_u32(lab_len) ||
            !r.read_bytes(label, lab_len) ||
            !r.read_u32(sr_u) || !r.read_f64(a) || !r.read_f64(b) ||
            !r.read_u32(disc) || !r.read_u32(num_parents)) {
            eshkol_proof_dag_destroy(dag);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
        if (kind_u > static_cast<uint32_t>(ESHKOL_PROOF_DERIVED) ||
            sr_u > static_cast<uint32_t>(ESHKOL_SR_CERTIFIED_INTERVAL)) {
            eshkol_proof_dag_destroy(dag);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }

        std::vector<uint32_t> parents(num_parents);
        for (uint32_t p = 0; p < num_parents; p++) {
            if (!r.read_u32(parents[p]) || parents[p] >= i) {
                eshkol_proof_dag_destroy(dag);
                return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
            }
        }
        if (!r.read_u64(expected_hash)) {
            eshkol_proof_dag_destroy(dag);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }

        eshkol_semiring_val_t val{static_cast<eshkol_semiring_kind_t>(sr_u), a, b, disc};
        uint32_t nid = eshkol_proof_dag_add_node(
            dag, static_cast<eshkol_proof_node_kind_t>(kind_u), label.c_str(),
            parents.empty() ? nullptr : parents.data(), num_parents, val);
        if (nid != i || eshkol_proof_dag_node_hash(dag, nid) != expected_hash) {
            eshkol_proof_dag_destroy(dag);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
    }

    if (!r.at_end() || !eshkol_proof_dag_verify_replay(dag)) {
        eshkol_proof_dag_destroy(dag);
        return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
    }

    *out_dag = dag;
    return ESHKOL_PERSIST_OK;
}

extern "C" eshkol_persist_status_t eshkol_ce_serialize_egraph(const eshkol_egraph_t* eg,
                                                              uint8_t** out_buf,
                                                              size_t* out_len) {
    if (!eg || !out_buf || !out_len) return ESHKOL_PERSIST_ERR_INVALID_ARG;

    std::vector<uint8_t> payload;
    const uint32_t num_colors = eshkol_egraph_num_colors(eg);
    const uint32_t num_nodes  = eshkol_egraph_num_nodes(eg);
    const uint32_t num_merges = eshkol_egraph_num_merges(eg);

    put_u32(payload, num_colors);
    for (uint32_t c = 1; c < num_colors; c++) {
        put_u32(payload, eshkol_egraph_color_parent(eg, c));
    }

    put_u32(payload, num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
        uint32_t op_id = 0, nch = 0;
        const uint32_t* ch = nullptr;
        double cost = 0.0;
        if (!eshkol_egraph_inspect_node(eg, i, &op_id, &ch, &nch, &cost)) {
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
        put_u32(payload, op_id);
        put_f64(payload, cost);
        put_u32(payload, nch);
        for (uint32_t k = 0; k < nch; k++) put_u32(payload, ch[k]);
    }

    put_u32(payload, num_merges);
    for (uint32_t m = 0; m < num_merges; m++) {
        uint32_t a = 0, b = 0, col = 0;
        if (!eshkol_egraph_inspect_merge(eg, m, &a, &b, &col)) {
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
        put_u32(payload, a);
        put_u32(payload, b);
        put_u32(payload, col);
    }

    return wrap_envelope(kKindEGraph, payload, out_buf, out_len);
}

extern "C" eshkol_persist_status_t eshkol_ce_deserialize_egraph(const uint8_t* buf,
                                                                size_t len,
                                                                eshkol_egraph_t** out_eg) {
    if (!out_eg) return ESHKOL_PERSIST_ERR_INVALID_ARG;
    *out_eg = nullptr;

    ByteReader r{};
    eshkol_persist_status_t st = validate_envelope(buf, len, kKindEGraph, r);
    if (st != ESHKOL_PERSIST_OK) return st;

    uint32_t num_colors = 0;
    if (!r.read_u32(num_colors) || num_colors == 0) return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;

    eshkol_egraph_t* eg = eshkol_egraph_create();
    if (!eg) return ESHKOL_PERSIST_ERR_IO;

    for (uint32_t c = 1; c < num_colors; c++) {
        uint32_t par = 0;
        if (!r.read_u32(par) || eshkol_egraph_create_color(eg, par) != c) {
            eshkol_egraph_destroy(eg);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
    }

    uint32_t num_nodes = 0;
    if (!r.read_u32(num_nodes)) {
        eshkol_egraph_destroy(eg);
        return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
    }

    for (uint32_t i = 0; i < num_nodes; i++) {
        uint32_t op_id = 0, nch = 0;
        double cost = 0.0;
        if (!r.read_u32(op_id) || !r.read_f64(cost) || !r.read_u32(nch)) {
            eshkol_egraph_destroy(eg);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
        std::vector<uint32_t> ch(nch);
        for (uint32_t k = 0; k < nch; k++) {
            if (!r.read_u32(ch[k]) || ch[k] >= i) {
                eshkol_egraph_destroy(eg);
                return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
            }
        }
        uint32_t nid = eshkol_egraph_add_node(eg, op_id, ch.empty() ? nullptr : ch.data(), nch, cost);
        if (nid == UINT32_MAX) {
            eshkol_egraph_destroy(eg);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
    }

    uint32_t num_merges = 0;
    if (!r.read_u32(num_merges)) {
        eshkol_egraph_destroy(eg);
        return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
    }

    for (uint32_t m = 0; m < num_merges; m++) {
        uint32_t a = 0, b = 0, col = 0;
        if (!r.read_u32(a) || !r.read_u32(b) || !r.read_u32(col) ||
            !eshkol_egraph_union(eg, a, b, col)) {
            eshkol_egraph_destroy(eg);
            return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
        }
    }

    if (!r.at_end() || !eshkol_egraph_rebuild(eg)) {
        eshkol_egraph_destroy(eg);
        return ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD;
    }

    *out_eg = eg;
    return ESHKOL_PERSIST_OK;
}

extern "C" eshkol_persist_status_t eshkol_ce_write_file_atomic(const char* path,
                                                               const uint8_t* data,
                                                               size_t len) {
    if (!path || (len > 0 && !data)) return ESHKOL_PERSIST_ERR_INVALID_ARG;
    std::string tmp = std::string(path) + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return ESHKOL_PERSIST_ERR_IO;
    if (len > 0 && std::fwrite(data, 1, len, f) != len) {
        std::fclose(f);
        std::remove(tmp.c_str());
        return ESHKOL_PERSIST_ERR_IO;
    }
    if (std::fflush(f) != 0) {
        std::fclose(f);
        std::remove(tmp.c_str());
        return ESHKOL_PERSIST_ERR_IO;
    }
#if defined(__unix__) || defined(__APPLE__)
    (void)fsync(fileno(f));
#endif
    if (std::fclose(f) != 0) {
        std::remove(tmp.c_str());
        return ESHKOL_PERSIST_ERR_IO;
    }
    if (std::rename(tmp.c_str(), path) != 0) {
        std::remove(tmp.c_str());
        return ESHKOL_PERSIST_ERR_IO;
    }
    return ESHKOL_PERSIST_OK;
}

extern "C" eshkol_persist_status_t eshkol_ce_read_file(const char* path,
                                                       uint8_t** out_buf,
                                                       size_t* out_len) {
    if (!path || !out_buf || !out_len) return ESHKOL_PERSIST_ERR_INVALID_ARG;
    *out_buf = nullptr;
    *out_len = 0;

    FILE* f = std::fopen(path, "rb");
    if (!f) return ESHKOL_PERSIST_ERR_IO;
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return ESHKOL_PERSIST_ERR_IO;
    }
    long sz = std::ftell(f);
    if (sz < 0) {
        std::fclose(f);
        return ESHKOL_PERSIST_ERR_IO;
    }
    std::rewind(f);

    size_t ulen = static_cast<size_t>(sz);
    auto* raw = static_cast<uint8_t*>(std::malloc(ulen > 0 ? ulen : 1u));
    if (!raw) {
        std::fclose(f);
        return ESHKOL_PERSIST_ERR_IO;
    }
    if (ulen > 0 && std::fread(raw, 1, ulen, f) != ulen) {
        std::free(raw);
        std::fclose(f);
        return ESHKOL_PERSIST_ERR_IO;
    }
    std::fclose(f);
    *out_buf = raw;
    *out_len = ulen;
    return ESHKOL_PERSIST_OK;
}
