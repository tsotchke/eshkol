/**
 * @file training_checkpoint.cpp
 * @brief Implementation of the S9 training checkpoint: an ESKM tensor-list
 *        payload plus a text manifest sidecar, both published atomically.
 *
 * See inc/eshkol/ml/training_checkpoint.h for the format rationale.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/ml/training_checkpoint.h"

#include "../core/model_io_atomic.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char kMagic[4] = {'E', 'S', 'K', 'M'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint8_t kDTypeFloat64 = 0;

/** One named tensor view into a caller-owned buffer -- no copy. */
struct NamedTensor {
    const char* name;
    const double* data;
    int64_t elements;
};

/** CRC-32 (IEEE 802.3), matching lib/core/model_io.cpp's checksum exactly so
 *  a training checkpoint is byte-for-byte the same container a model-load
 *  reader already validates. */
uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

/** Thin atomic-write wrapper over model_io_atomic.h, tracking a running CRC
 *  exactly like model_io.cpp's FileWriter (duplicated rather than shared
 *  because that class is private to model_io.cpp's anonymous namespace; both
 *  copies write the identical wire format). */
class Writer {
public:
    explicit Writer(const char* path) { ok_ = eshkol_atomic_checkpoint_begin(&file_, path) != 0; }
    ~Writer() { eshkol_atomic_checkpoint_abort(&file_); }

    bool good() const { return file_.stream != nullptr && ok_; }
    uint32_t crc() const { return crc_; }

    bool bytes(const void* data, size_t size, bool in_crc = true) {
        if (!good()) return false;
        if (size != 0 && eshkol_atomic_checkpoint_write(&file_, data, size) != size) {
            ok_ = false;
            return false;
        }
        if (in_crc && size != 0) crc_ = crc32_update(crc_, static_cast<const uint8_t*>(data), size);
        return true;
    }
    bool u8(uint8_t v, bool in_crc = true) { return bytes(&v, 1, in_crc); }
    bool u32(uint32_t v, bool in_crc = true) {
        uint8_t b[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
        return bytes(b, 4, in_crc);
    }
    bool u64(uint64_t v, bool in_crc = true) {
        uint8_t b[8];
        for (int i = 0; i < 8; ++i) b[i] = uint8_t(v >> (8 * i));
        return bytes(b, 8, in_crc);
    }
    bool commit() {
        if (!good()) return false;
        ok_ = eshkol_atomic_checkpoint_commit(&file_) != 0;
        return ok_;
    }

private:
    eshkol_atomic_checkpoint_file_t file_{};
    bool ok_ = false;
    uint32_t crc_ = 0;
};

bool read_file(const char* path, std::vector<uint8_t>* out) {
    if (!path || !out) return false;
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return false; }
    long size = std::ftell(f);
    if (size < 0 || std::fseek(f, 0, SEEK_SET) != 0) { std::fclose(f); return false; }
    out->assign(static_cast<size_t>(size), 0);
    if (size > 0 && std::fread(out->data(), 1, out->size(), f) != out->size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

/** The eight named tensors that make up the training state, in a fixed
 *  order both save and load agree on. Names match the shapes documented in
 *  mixed_curvature_step.h so the payload reads sensibly through the generic
 *  `(model-load path)` builtin too. */
std::vector<NamedTensor> build_records(const EshkolMixedCurvatureShape& s,
                                       const EshkolMixedCurvatureParams* p,
                                       const EshkolMixedCurvatureMoments* m,
                                       const double* step_slot,
                                       const double* curvature_slot,
                                       const double* seed_slot) {
    int64_t w_n = eshkol_mixed_curvature_w_elements(s);
    int64_t p_n = eshkol_mixed_curvature_p_elements(s);
    return {
        {"w", p->w, w_n},
        {"p_hyp", p->p_hyp, p_n},
        {"p_sph", p->p_sph, p_n},
        {"p_euc", p->p_euc, p_n},
        {"m_w", m->m_w, w_n},     {"v_w", m->v_w, w_n},
        {"m_hyp", m->m_hyp, p_n}, {"v_hyp", m->v_hyp, p_n},
        {"m_sph", m->m_sph, p_n}, {"v_sph", m->v_sph, p_n},
        {"m_euc", m->m_euc, p_n}, {"v_euc", m->v_euc, p_n},
        {"step", step_slot, 1},
        {"curvature", curvature_slot, 1},
        {"seed", seed_slot, 1},
    };
}

/** Writes the ESKM payload; returns the payload's own trailing CRC-32 (0 on
 *  any failure, which the caller distinguishes from an actual all-zero CRC
 *  by checking the writer's success separately). */
bool write_payload(const char* path, const std::vector<NamedTensor>& records, uint32_t* out_crc) {
    Writer w(path);
    if (!w.good()) return false;
    if (!w.bytes(kMagic, 4)) return false;
    if (!w.u32(kFormatVersion)) return false;
    if (!w.u32(static_cast<uint32_t>(records.size()))) return false;
    if (!w.u32(0)) return false; /* flags, reserved -- unused, matches model_io.cpp */

    for (const NamedTensor& r : records) {
        if (!r.data && r.elements != 0) return false;
        size_t name_len = std::strlen(r.name);
        if (!w.u32(static_cast<uint32_t>(name_len))) return false;
        if (!w.bytes(r.name, name_len)) return false;
        if (!w.u32(1)) return false; /* ndims: every record here is a flat 1-D vector */
        if (!w.u64(static_cast<uint64_t>(r.elements))) return false;
        if (!w.u8(kDTypeFloat64)) return false;
        for (int64_t i = 0; i < r.elements; ++i) {
            uint64_t bits;
            std::memcpy(&bits, &r.data[i], sizeof(bits));
            if (!w.u64(bits)) return false;
        }
    }

    if (!w.u32(w.crc(), false)) return false;
    if (out_crc) *out_crc = w.crc();
    return w.commit();
}

struct ParsedRecord {
    std::string name;
    std::vector<double> values;
};

/** Parses (and CRC-validates) an ESKM payload written by write_payload().
 *  Accepts any record shape (not just the 1-D vectors this module writes) so
 *  it stays compatible with anything model_io.cpp itself produced. */
bool parse_payload(const char* path, std::vector<ParsedRecord>* out, uint32_t* out_stored_crc) {
    std::vector<uint8_t> bytes;
    if (!read_file(path, &bytes) || bytes.size() < 16) return false;

    size_t payload_size = bytes.size() - 4;
    uint32_t stored_crc = uint32_t(bytes[payload_size]) | (uint32_t(bytes[payload_size + 1]) << 8) |
                          (uint32_t(bytes[payload_size + 2]) << 16) | (uint32_t(bytes[payload_size + 3]) << 24);
    uint32_t computed_crc = crc32_update(0, bytes.data(), payload_size);
    if (stored_crc != computed_crc) return false;
    if (out_stored_crc) *out_stored_crc = stored_crc;

    size_t off = 0;
    auto need = [&](size_t n) { return off + n <= payload_size; };
    if (!need(4) || std::memcmp(bytes.data(), kMagic, 4) != 0) return false;
    off += 4;
    auto rd_u32 = [&]() -> uint32_t {
        uint32_t v = uint32_t(bytes[off]) | (uint32_t(bytes[off + 1]) << 8) |
                    (uint32_t(bytes[off + 2]) << 16) | (uint32_t(bytes[off + 3]) << 24);
        off += 4;
        return v;
    };
    auto rd_u64 = [&]() -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint64_t(bytes[off + i]) << (8 * i);
        off += 8;
        return v;
    };

    if (!need(12)) return false;
    uint32_t version = rd_u32();
    uint32_t count = rd_u32();
    (void)rd_u32(); /* flags */
    if (version != kFormatVersion) return false;

    out->clear();
    out->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!need(4)) return false;
        uint32_t name_len = rd_u32();
        if (!need(name_len)) return false;
        ParsedRecord rec;
        rec.name.assign(reinterpret_cast<const char*>(bytes.data() + off), name_len);
        off += name_len;
        if (!need(4)) return false;
        uint32_t ndims = rd_u32();
        if (!need(static_cast<size_t>(ndims) * 8)) return false;
        std::vector<uint64_t> dims(ndims);
        uint64_t total = 1;
        for (uint32_t d = 0; d < ndims; ++d) {
            dims[d] = rd_u64();
            if (dims[d] == 0) { total = 0; }
            else if (total != 0) { total *= dims[d]; }
        }
        if (!need(1)) return false;
        uint8_t dtype = bytes[off++];
        if (dtype != kDTypeFloat64) return false;
        if (!need(total * 8)) return false;
        rec.values.resize(static_cast<size_t>(total));
        for (uint64_t e = 0; e < total; ++e) {
            uint64_t bits = rd_u64();
            std::memcpy(&rec.values[static_cast<size_t>(e)], &bits, sizeof(double));
        }
        out->push_back(std::move(rec));
    }
    return off == payload_size;
}

const ParsedRecord* find_record(const std::vector<ParsedRecord>& records, const char* name) {
    for (const auto& r : records) {
        if (r.name == name) return &r;
    }
    return nullptr;
}

bool copy_record(const std::vector<ParsedRecord>& records, const char* name, int64_t expected, double* dst) {
    const ParsedRecord* r = find_record(records, name);
    if (!r || static_cast<int64_t>(r->values.size()) != expected) return false;
    if (expected > 0) std::memcpy(dst, r->values.data(), static_cast<size_t>(expected) * sizeof(double));
    return true;
}

/** Manifest text: one `key=value` per line, `#`-prefixed header line. Kept as
 *  plain text (not JSON) so writing it needs no dependency beyond snprintf --
 *  the same reason model_io.cpp's own container is a fixed binary layout
 *  rather than a generic serialization format. */
std::string build_manifest(const EshkolTrainingCheckpointMeta& meta, uint32_t content_crc32, int64_t step_value) {
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "eshkol-training-checkpoint-manifest v1\n"
                  "shape_n=%" PRId64 "\n"
                  "shape_d=%" PRId64 "\n"
                  "shape_c=%" PRId64 "\n"
                  "dtype_policy=%s\n"
                  "curvature=%.17g\n"
                  "step=%" PRId64 "\n"
                  "seed=%" PRIu64 "\n"
                  "content_crc32=%08x\n",
                  meta.shape.n, meta.shape.d, meta.shape.c,
                  meta.dtype_policy == ESHKOL_TRAINING_DTYPE_BF16_MIXED ? "bf16-mixed" : "f32",
                  meta.hyper.curvature, step_value, meta.seed, content_crc32);
    return std::string(buf);
}

struct ParsedManifest {
    int64_t n = 0, d = 0, c = 0;
    EshkolTrainingDtypePolicy dtype_policy = ESHKOL_TRAINING_DTYPE_F32;
    double curvature = 0;
    int64_t step = 0;
    uint64_t seed = 0;
    uint32_t content_crc32 = 0;
    bool valid = false;
};

ParsedManifest parse_manifest(const char* path) {
    ParsedManifest m;
    FILE* f = std::fopen(path, "rb");
    if (!f) return m;
    char line[768];
    bool saw_header = false;
    int fields = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (!saw_header) {
            saw_header = (std::strncmp(line, "eshkol-training-checkpoint-manifest",
                                       std::strlen("eshkol-training-checkpoint-manifest")) == 0);
            if (!saw_header) { std::fclose(f); return m; }
            continue;
        }
        long long ll;
        unsigned long long ull;
        unsigned int u;
        char sbuf[64];
        if (std::sscanf(line, "shape_n=%lld", &ll) == 1) { m.n = ll; fields++; }
        else if (std::sscanf(line, "shape_d=%lld", &ll) == 1) { m.d = ll; fields++; }
        else if (std::sscanf(line, "shape_c=%lld", &ll) == 1) { m.c = ll; fields++; }
        else if (std::sscanf(line, "dtype_policy=%63s", sbuf) == 1) {
            m.dtype_policy = (std::strcmp(sbuf, "bf16-mixed") == 0) ? ESHKOL_TRAINING_DTYPE_BF16_MIXED
                                                                     : ESHKOL_TRAINING_DTYPE_F32;
            fields++;
        } else if (std::sscanf(line, "curvature=%lg", &m.curvature) == 1) { fields++; }
        else if (std::sscanf(line, "step=%lld", &ll) == 1) { m.step = ll; fields++; }
        else if (std::sscanf(line, "seed=%llu", &ull) == 1) { m.seed = ull; fields++; }
        else if (std::sscanf(line, "content_crc32=%x", &u) == 1) { m.content_crc32 = u; fields++; }
    }
    std::fclose(f);
    m.valid = saw_header && fields == 8;
    return m;
}

std::string manifest_path(const char* path) { return std::string(path) + ".manifest"; }

} // namespace

bool eshkol_training_checkpoint_save(const char* path,
                                     const EshkolTrainingCheckpointMeta* meta,
                                     const EshkolMixedCurvatureParams* params,
                                     const EshkolMixedCurvatureMoments* moments) {
    if (!path || !meta || !params || !moments) return false;
    if (!params->w || !params->p_hyp || !params->p_sph || !params->p_euc) return false;
    if (!moments->m_w || !moments->v_w) return false;

    double step_slot = static_cast<double>(moments->step);
    double curvature_slot = meta->hyper.curvature;
    double seed_slot;
    std::memcpy(&seed_slot, &meta->seed, sizeof(seed_slot)); /* bit pattern, not a numeric cast */

    std::vector<NamedTensor> records =
        build_records(meta->shape, params, moments, &step_slot, &curvature_slot, &seed_slot);

    uint32_t content_crc32 = 0;
    if (!write_payload(path, records, &content_crc32)) return false;

    std::string manifest_text = build_manifest(*meta, content_crc32, moments->step);
    std::string mpath = manifest_path(path);
    Writer mw(mpath.c_str());
    if (!mw.good()) return false;
    if (!mw.bytes(manifest_text.data(), manifest_text.size(), false)) return false;
    return mw.commit();
}

namespace {

bool validate_and_parse(const char* path, const EshkolMixedCurvatureShape* expected_shape,
                        ParsedManifest* out_manifest, std::vector<ParsedRecord>* out_records) {
    if (!path) return false;
    ParsedManifest manifest = parse_manifest(manifest_path(path).c_str());
    if (!manifest.valid) return false;
    if (expected_shape &&
        (manifest.n != expected_shape->n || manifest.d != expected_shape->d || manifest.c != expected_shape->c)) {
        return false;
    }

    uint32_t stored_crc = 0;
    std::vector<ParsedRecord> records;
    if (!parse_payload(path, &records, &stored_crc)) return false;
    if (stored_crc != manifest.content_crc32) return false; /* manifest and payload disagree: refuse */

    if (out_manifest) *out_manifest = manifest;
    if (out_records) *out_records = std::move(records);
    return true;
}

} // namespace

bool eshkol_training_checkpoint_is_valid(const char* path, const EshkolMixedCurvatureShape* expected_shape) {
    return validate_and_parse(path, expected_shape, nullptr, nullptr);
}

bool eshkol_training_checkpoint_load(const char* path,
                                     const EshkolMixedCurvatureShape* expected_shape,
                                     EshkolTrainingCheckpointMeta* out_meta,
                                     EshkolMixedCurvatureParams* params,
                                     EshkolMixedCurvatureMoments* moments) {
    if (!path || !params || !moments) return false;

    ParsedManifest manifest;
    std::vector<ParsedRecord> records;
    if (!validate_and_parse(path, expected_shape, &manifest, &records)) return false;

    EshkolMixedCurvatureShape shape{manifest.n, manifest.d, manifest.c};
    int64_t w_n = eshkol_mixed_curvature_w_elements(shape);
    int64_t p_n = eshkol_mixed_curvature_p_elements(shape);

    bool ok = copy_record(records, "w", w_n, params->w) &&
             copy_record(records, "p_hyp", p_n, params->p_hyp) &&
             copy_record(records, "p_sph", p_n, params->p_sph) &&
             copy_record(records, "p_euc", p_n, params->p_euc) &&
             copy_record(records, "m_w", w_n, moments->m_w) &&
             copy_record(records, "v_w", w_n, moments->v_w) &&
             copy_record(records, "m_hyp", p_n, moments->m_hyp) &&
             copy_record(records, "v_hyp", p_n, moments->v_hyp) &&
             copy_record(records, "m_sph", p_n, moments->m_sph) &&
             copy_record(records, "v_sph", p_n, moments->v_sph) &&
             copy_record(records, "m_euc", p_n, moments->m_euc) &&
             copy_record(records, "v_euc", p_n, moments->v_euc);
    if (!ok) return false;

    const ParsedRecord* step_rec = find_record(records, "step");
    const ParsedRecord* curvature_rec = find_record(records, "curvature");
    const ParsedRecord* seed_rec = find_record(records, "seed");
    if (!step_rec || step_rec->values.size() != 1) return false;
    if (!curvature_rec || curvature_rec->values.size() != 1) return false;
    if (!seed_rec || seed_rec->values.size() != 1) return false;

    moments->step = static_cast<int64_t>(step_rec->values[0]);
    if (moments->step != manifest.step) return false; /* payload/manifest step disagree: refuse */

    uint64_t seed_bits;
    std::memcpy(&seed_bits, &seed_rec->values[0], sizeof(seed_bits));
    if (seed_bits != manifest.seed) return false;

    if (out_meta) {
        out_meta->shape = shape;
        eshkol_mixed_curvature_default_hyper(&out_meta->hyper);
        out_meta->hyper.curvature = curvature_rec->values[0];
        out_meta->seed = seed_bits;
        out_meta->dtype_policy = manifest.dtype_policy;
    }
    return true;
}
