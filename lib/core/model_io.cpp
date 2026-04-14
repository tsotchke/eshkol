#include <eshkol/model_io.h>

#include "arena_memory.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<std::uint8_t, 4> kMagic{{'E', 'S', 'K', 'M'}};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint8_t kDTypeFloat64 = 0;

struct TensorRecordView {
    std::string name;
    const eshkol_tensor_t* tensor;
};

struct ParsedTensorRecord {
    std::string name;
    std::uint32_t ndims;
    std::vector<std::uint64_t> dims;
    std::vector<std::uint64_t> element_bits;
};

eshkol_tagged_value_t make_null() {
    eshkol_tagged_value_t result{};
    result.type = ESHKOL_VALUE_NULL;
    return result;
}

eshkol_tagged_value_t make_bool(bool value) {
    eshkol_tagged_value_t result{};
    result.type = ESHKOL_VALUE_BOOL;
    result.data.int_val = value ? 1 : 0;
    return result;
}

eshkol_tagged_value_t make_heap_ptr(void* ptr) {
    eshkol_tagged_value_t result{};
    result.type = ESHKOL_VALUE_HEAP_PTR;
    result.data.ptr_val = reinterpret_cast<std::uint64_t>(ptr);
    return result;
}

bool is_pair(const eshkol_tagged_value_t& value) {
    return ESHKOL_IS_CONS_COMPAT(value);
}

const arena_tagged_cons_cell_t* as_pair(const eshkol_tagged_value_t& value) {
    if (!is_pair(value)) return nullptr;
    return reinterpret_cast<const arena_tagged_cons_cell_t*>(value.data.ptr_val);
}

eshkol_tagged_value_t pair_car(const eshkol_tagged_value_t& value) {
    const arena_tagged_cons_cell_t* pair = as_pair(value);
    return pair ? pair->car : make_null();
}

eshkol_tagged_value_t pair_cdr(const eshkol_tagged_value_t& value) {
    const arena_tagged_cons_cell_t* pair = as_pair(value);
    return pair ? pair->cdr : make_null();
}

const char* tagged_c_string(const eshkol_tagged_value_t* value) {
    if (!value || value->type != ESHKOL_VALUE_HEAP_PTR || value->data.ptr_val == 0) return nullptr;
    const auto* ptr = reinterpret_cast<const void*>(value->data.ptr_val);
    const auto subtype = ESHKOL_GET_SUBTYPE(ptr);
    if (subtype != HEAP_SUBTYPE_STRING && subtype != HEAP_SUBTYPE_SYMBOL) return nullptr;
    return reinterpret_cast<const char*>(ptr);
}

bool tagged_is_tensor(const eshkol_tagged_value_t* value) {
    return value && ESHKOL_IS_TENSOR_COMPAT(*value);
}

std::uint32_t crc32_update(std::uint32_t crc, const std::uint8_t* data, std::size_t len) {
    crc = ~crc;
    for (std::size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

class FileWriter {
public:
    explicit FileWriter(const char* path) : file_(std::fopen(path, "wb")) {}
    ~FileWriter() {
        if (file_) std::fclose(file_);
    }

    bool good() const { return file_ != nullptr && ok_; }
    std::uint32_t crc() const { return crc_; }

    bool write_bytes(const void* data, std::size_t size, bool include_in_crc = true) {
        if (!good()) return false;
        if (size != 0 && std::fwrite(data, 1, size, file_) != size) {
            ok_ = false;
            return false;
        }
        if (include_in_crc && size != 0) {
            crc_ = crc32_update(crc_, static_cast<const std::uint8_t*>(data), size);
        }
        return true;
    }

    bool write_u8(std::uint8_t value, bool include_in_crc = true) {
        return write_bytes(&value, sizeof(value), include_in_crc);
    }

    bool write_u32(std::uint32_t value, bool include_in_crc = true) {
        std::array<std::uint8_t, 4> bytes{{
            static_cast<std::uint8_t>(value & 0xFFu),
            static_cast<std::uint8_t>((value >> 8) & 0xFFu),
            static_cast<std::uint8_t>((value >> 16) & 0xFFu),
            static_cast<std::uint8_t>((value >> 24) & 0xFFu),
        }};
        return write_bytes(bytes.data(), bytes.size(), include_in_crc);
    }

    bool write_u64(std::uint64_t value, bool include_in_crc = true) {
        std::array<std::uint8_t, 8> bytes{{
            static_cast<std::uint8_t>(value & 0xFFu),
            static_cast<std::uint8_t>((value >> 8) & 0xFFu),
            static_cast<std::uint8_t>((value >> 16) & 0xFFu),
            static_cast<std::uint8_t>((value >> 24) & 0xFFu),
            static_cast<std::uint8_t>((value >> 32) & 0xFFu),
            static_cast<std::uint8_t>((value >> 40) & 0xFFu),
            static_cast<std::uint8_t>((value >> 48) & 0xFFu),
            static_cast<std::uint8_t>((value >> 56) & 0xFFu),
        }};
        return write_bytes(bytes.data(), bytes.size(), include_in_crc);
    }

private:
    FILE* file_ = nullptr;
    bool ok_ = true;
    std::uint32_t crc_ = 0;
};

bool read_file(const char* path, std::vector<std::uint8_t>* bytes) {
    if (!path || !bytes) return false;
    FILE* file = std::fopen(path, "rb");
    if (!file) return false;

    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return false;
    }
    long size = std::ftell(file);
    if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return false;
    }

    bytes->assign(static_cast<std::size_t>(size), 0);
    if (size > 0 && std::fread(bytes->data(), 1, bytes->size(), file) != bytes->size()) {
        std::fclose(file);
        return false;
    }
    std::fclose(file);
    return true;
}

struct BufferReader {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t offset = 0;

    bool read_u8(std::uint8_t* out) {
        if (!out || offset + 1 > size) return false;
        *out = data[offset++];
        return true;
    }

    bool read_u32(std::uint32_t* out) {
        if (!out || offset + 4 > size) return false;
        *out = static_cast<std::uint32_t>(data[offset]) |
               (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
               (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
               (static_cast<std::uint32_t>(data[offset + 3]) << 24);
        offset += 4;
        return true;
    }

    bool read_u64(std::uint64_t* out) {
        if (!out || offset + 8 > size) return false;
        *out = static_cast<std::uint64_t>(data[offset]) |
               (static_cast<std::uint64_t>(data[offset + 1]) << 8) |
               (static_cast<std::uint64_t>(data[offset + 2]) << 16) |
               (static_cast<std::uint64_t>(data[offset + 3]) << 24) |
               (static_cast<std::uint64_t>(data[offset + 4]) << 32) |
               (static_cast<std::uint64_t>(data[offset + 5]) << 40) |
               (static_cast<std::uint64_t>(data[offset + 6]) << 48) |
               (static_cast<std::uint64_t>(data[offset + 7]) << 56);
        offset += 8;
        return true;
    }

    bool read_string(std::uint32_t len, std::string* out) {
        if (!out || offset + len > size) return false;
        out->assign(reinterpret_cast<const char*>(data + offset), len);
        offset += len;
        return true;
    }
};

bool compute_total_elements(const std::vector<std::uint64_t>& dims, std::uint64_t* total) {
    if (!total) return false;
    std::uint64_t value = 1;
    for (std::uint64_t dim : dims) {
        if (dim == 0) {
            value = 0;
            break;
        }
        if (value > UINT64_MAX / dim) return false;
        value *= dim;
    }
    *total = value;
    return true;
}

bool write_checkpoint(const char* path, const std::vector<TensorRecordView>& records) {
    if (!path) return false;

    FileWriter writer(path);
    if (!writer.good()) return false;

    if (!writer.write_bytes(kMagic.data(), kMagic.size())) return false;
    if (!writer.write_u32(kFormatVersion)) return false;
    if (!writer.write_u32(static_cast<std::uint32_t>(records.size()))) return false;
    if (!writer.write_u32(0)) return false;

    for (const TensorRecordView& record : records) {
        if (!record.tensor) return false;
        if (record.name.size() > UINT32_MAX) return false;

        if (!writer.write_u32(static_cast<std::uint32_t>(record.name.size()))) return false;
        if (!writer.write_bytes(record.name.data(), record.name.size())) return false;
        if (record.tensor->num_dimensions > UINT32_MAX) return false;
        if (!writer.write_u32(static_cast<std::uint32_t>(record.tensor->num_dimensions))) return false;
        for (std::uint64_t i = 0; i < record.tensor->num_dimensions; ++i) {
            if (!writer.write_u64(record.tensor->dimensions ? record.tensor->dimensions[i] : 0)) return false;
        }
        if (!writer.write_u8(kDTypeFloat64)) return false;
        for (std::uint64_t i = 0; i < record.tensor->total_elements; ++i) {
            const std::uint64_t bits = std::bit_cast<std::uint64_t>(record.tensor->elements[i]);
            if (!writer.write_u64(bits)) return false;
        }
    }

    return writer.write_u32(writer.crc(), false);
}

bool parse_checkpoint(const char* path, std::vector<ParsedTensorRecord>* records) {
    if (!path || !records) return false;

    std::vector<std::uint8_t> bytes;
    if (!read_file(path, &bytes) || bytes.size() < 16) return false;

    const std::size_t payload_size = bytes.size() - 4;
    BufferReader footer{bytes.data() + payload_size, 4, 0};
    std::uint32_t stored_crc = 0;
    if (!footer.read_u32(&stored_crc)) return false;

    const std::uint32_t computed_crc = crc32_update(0, bytes.data(), payload_size);
    if (stored_crc != computed_crc) return false;

    BufferReader reader{bytes.data(), payload_size, 0};
    std::array<std::uint8_t, 4> magic{};
    for (std::uint8_t& byte : magic) {
        if (!reader.read_u8(&byte)) return false;
    }
    if (magic != kMagic) return false;

    std::uint32_t version = 0;
    std::uint32_t tensor_count = 0;
    std::uint32_t flags = 0;
    if (!reader.read_u32(&version) || !reader.read_u32(&tensor_count) || !reader.read_u32(&flags)) {
        return false;
    }
    (void)flags;
    if (version != kFormatVersion) return false;

    records->clear();
    records->reserve(tensor_count);
    for (std::uint32_t i = 0; i < tensor_count; ++i) {
        ParsedTensorRecord record;
        std::uint32_t name_len = 0;
        if (!reader.read_u32(&name_len) || !reader.read_string(name_len, &record.name)) return false;
        if (!reader.read_u32(&record.ndims)) return false;

        record.dims.resize(record.ndims);
        for (std::uint32_t dim = 0; dim < record.ndims; ++dim) {
            if (!reader.read_u64(&record.dims[dim])) return false;
        }

        std::uint8_t dtype = 0;
        if (!reader.read_u8(&dtype) || dtype != kDTypeFloat64) return false;

        std::uint64_t total_elements = 0;
        if (!compute_total_elements(record.dims, &total_elements)) return false;
        if (total_elements > SIZE_MAX / sizeof(std::uint64_t)) return false;
        record.element_bits.resize(static_cast<std::size_t>(total_elements));
        for (std::uint64_t elem = 0; elem < total_elements; ++elem) {
            if (!reader.read_u64(&record.element_bits[static_cast<std::size_t>(elem)])) return false;
        }

        records->push_back(std::move(record));
    }

    return reader.offset == reader.size;
}

bool tensor_from_record(arena_t* arena, const ParsedTensorRecord& record, eshkol_tensor_t** out) {
    if (!arena || !out) return false;
    std::uint64_t total_elements = 0;
    if (!compute_total_elements(record.dims, &total_elements)) return false;

    eshkol_tensor_t* tensor = arena_allocate_tensor_with_header(arena);
    if (!tensor) return false;
    tensor->num_dimensions = record.ndims;
    tensor->total_elements = total_elements;
    if (record.ndims > 0) {
        tensor->dimensions = static_cast<std::uint64_t*>(
            arena_allocate_aligned(arena, static_cast<std::size_t>(record.ndims) * sizeof(std::uint64_t), 64));
        if (!tensor->dimensions) return false;
    }
    if (total_elements > 0) {
        if (total_elements > SIZE_MAX / sizeof(std::int64_t)) return false;
        tensor->elements = static_cast<std::int64_t*>(
            arena_allocate_aligned(arena, total_elements * sizeof(std::int64_t), 64));
        if (!tensor->elements) return false;
    }
    for (std::uint32_t i = 0; i < record.ndims; ++i) {
        tensor->dimensions[i] = record.dims[i];
    }
    for (std::size_t i = 0; i < record.element_bits.size(); ++i) {
        tensor->elements[i] = std::bit_cast<std::int64_t>(record.element_bits[i]);
    }
    *out = tensor;
    return true;
}

bool extract_model_entries(const eshkol_tagged_value_t* list_value, std::vector<TensorRecordView>* out) {
    if (!list_value || !out) return false;
    out->clear();

    eshkol_tagged_value_t current = *list_value;
    while (is_pair(current)) {
        const eshkol_tagged_value_t entry_value = pair_car(current);
        if (!is_pair(entry_value)) return false;

        const eshkol_tagged_value_t name_value = pair_car(entry_value);
        const eshkol_tagged_value_t tensor_value = pair_cdr(entry_value);
        const char* name = tagged_c_string(&name_value);
        if (!name || !tagged_is_tensor(&tensor_value)) return false;

        out->push_back(TensorRecordView{
            .name = name,
            .tensor = reinterpret_cast<const eshkol_tensor_t*>(tensor_value.data.ptr_val),
        });
        current = pair_cdr(current);
    }

    return current.type == ESHKOL_VALUE_NULL;
}

bool make_string_value(arena_t* arena, std::string_view text, eshkol_tagged_value_t* out) {
    if (!arena || !out) return false;
    char* buffer = arena_allocate_string_with_header(arena, text.size());
    if (!buffer) return false;
    if (!text.empty()) std::memcpy(buffer, text.data(), text.size());
    buffer[text.size()] = '\0';
    *out = make_heap_ptr(buffer);
    return true;
}

bool prepend_list_node(arena_t* arena,
                       const eshkol_tagged_value_t& car,
                       const eshkol_tagged_value_t& cdr,
                       eshkol_tagged_value_t* out) {
    if (!arena || !out) return false;
    arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
    if (!cell) return false;
    arena_tagged_cons_set_tagged_value(cell, false, &car);
    arena_tagged_cons_set_tagged_value(cell, true, &cdr);
    *out = make_heap_ptr(cell);
    return true;
}

bool build_model_list(arena_t* arena,
                      const std::vector<ParsedTensorRecord>& records,
                      eshkol_tagged_value_t* result) {
    if (!arena || !result) return false;

    eshkol_tagged_value_t list = make_null();
    for (auto it = records.rbegin(); it != records.rend(); ++it) {
        eshkol_tensor_t* tensor = nullptr;
        if (!tensor_from_record(arena, *it, &tensor)) return false;

        eshkol_tagged_value_t name_value;
        if (!make_string_value(arena, it->name, &name_value)) return false;
        const eshkol_tagged_value_t tensor_value = make_heap_ptr(tensor);

        eshkol_tagged_value_t pair_value;
        if (!prepend_list_node(arena, name_value, tensor_value, &pair_value)) return false;
        if (!prepend_list_node(arena, pair_value, list, &list)) return false;
    }

    *result = list;
    return true;
}

} // namespace

extern "C" void eshkol_tensor_save_tagged(arena_t* arena,
                                           const eshkol_tagged_value_t* path_tv,
                                           const eshkol_tagged_value_t* tensor_tv,
                                           eshkol_tagged_value_t* result) {
    (void)arena;
    if (!result) return;
    *result = make_bool(false);

    const char* path = tagged_c_string(path_tv);
    if (!path || !tagged_is_tensor(tensor_tv)) return;

    const auto* tensor = reinterpret_cast<const eshkol_tensor_t*>(tensor_tv->data.ptr_val);
    *result = make_bool(write_checkpoint(path, {{"", tensor}}));
}

extern "C" void eshkol_tensor_load_tagged(arena_t* arena,
                                           const eshkol_tagged_value_t* path_tv,
                                           eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = make_null();

    const char* path = tagged_c_string(path_tv);
    if (!arena || !path) return;

    std::vector<ParsedTensorRecord> records;
    if (!parse_checkpoint(path, &records) || records.size() != 1) return;

    eshkol_tensor_t* tensor = nullptr;
    if (!tensor_from_record(arena, records.front(), &tensor)) return;
    *result = make_heap_ptr(tensor);
}

extern "C" void eshkol_model_save_tagged(arena_t* arena,
                                          const eshkol_tagged_value_t* path_tv,
                                          const eshkol_tagged_value_t* entries_tv,
                                          eshkol_tagged_value_t* result) {
    (void)arena;
    if (!result) return;
    *result = make_bool(false);

    const char* path = tagged_c_string(path_tv);
    if (!path) return;

    std::vector<TensorRecordView> entries;
    if (!extract_model_entries(entries_tv, &entries)) return;
    *result = make_bool(write_checkpoint(path, entries));
}

extern "C" void eshkol_model_load_tagged(arena_t* arena,
                                          const eshkol_tagged_value_t* path_tv,
                                          eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = make_null();

    const char* path = tagged_c_string(path_tv);
    if (!arena || !path) return;

    std::vector<ParsedTensorRecord> records;
    if (!parse_checkpoint(path, &records)) return;
    if (!build_model_list(arena, records, result)) {
        *result = make_null();
    }
}
