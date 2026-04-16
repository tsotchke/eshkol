#include <eshkol/model_io.h>

#include "../../lib/core/arena_memory.h"

#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

eshkol_tagged_value_t make_null() {
    eshkol_tagged_value_t value{};
    value.type = ESHKOL_VALUE_NULL;
    return value;
}

eshkol_tagged_value_t make_heap_ptr(void* ptr) {
    eshkol_tagged_value_t value{};
    value.type = ESHKOL_VALUE_HEAP_PTR;
    value.data.ptr_val = reinterpret_cast<std::uint64_t>(ptr);
    return value;
}

eshkol_tagged_value_t make_string(arena_t* arena, const std::string& text) {
    char* buffer = arena_allocate_string_with_header(arena, text.size());
    std::memcpy(buffer, text.data(), text.size());
    buffer[text.size()] = '\0';
    return make_heap_ptr(buffer);
}

void set_tensor_value(eshkol_tensor_t* tensor, std::size_t index, double value) {
    tensor->elements[index] = std::bit_cast<std::int64_t>(value);
}

double get_tensor_value(const eshkol_tensor_t* tensor, std::size_t index) {
    return std::bit_cast<double>(tensor->elements[index]);
}

eshkol_tensor_t* make_tensor(arena_t* arena,
                             const std::vector<std::uint64_t>& dims,
                             const std::vector<double>& values) {
    std::uint64_t total = 1;
    for (std::uint64_t dim : dims) {
        total *= dim;
    }
    eshkol_tensor_t* tensor = arena_allocate_tensor_full(arena, dims.size(), total);
    for (std::size_t i = 0; i < dims.size(); ++i) tensor->dimensions[i] = dims[i];
    for (std::size_t i = 0; i < values.size(); ++i) set_tensor_value(tensor, i, values[i]);
    return tensor;
}

eshkol_tensor_t* make_empty_tensor(arena_t* arena, const std::vector<std::uint64_t>& dims) {
    eshkol_tensor_t* tensor = arena_allocate_tensor_full(arena, dims.size(), 0);
    for (std::size_t i = 0; i < dims.size(); ++i) tensor->dimensions[i] = dims[i];
    return tensor;
}

bool tensor_equals(const eshkol_tensor_t* tensor,
                   const std::vector<std::uint64_t>& dims,
                   const std::vector<double>& values) {
    if (!tensor || tensor->num_dimensions != dims.size() || tensor->total_elements != values.size()) return false;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (tensor->dimensions[i] != dims[i]) return false;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (get_tensor_value(tensor, i) != values[i]) return false;
    }
    return true;
}

eshkol_tensor_t* make_pattern_tensor(arena_t* arena, const std::vector<std::uint64_t>& dims) {
    std::uint64_t total = 1;
    for (std::uint64_t dim : dims) total *= dim;
    eshkol_tensor_t* tensor = arena_allocate_tensor_full(arena, dims.size(), total);
    for (std::size_t i = 0; i < dims.size(); ++i) tensor->dimensions[i] = dims[i];
    for (std::uint64_t i = 0; i < total; ++i) {
        set_tensor_value(tensor, i, static_cast<double>((i % 1024) - 512) / 8.0);
    }
    return tensor;
}

bool tensor_matches_pattern(const eshkol_tensor_t* tensor, const std::vector<std::uint64_t>& dims) {
    if (!tensor || tensor->num_dimensions != dims.size()) return false;
    std::uint64_t total = 1;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (tensor->dimensions[i] != dims[i]) return false;
        total *= dims[i];
    }
    if (tensor->total_elements != total) return false;
    for (std::uint64_t i = 0; i < total; ++i) {
        const double expected = static_cast<double>((i % 1024) - 512) / 8.0;
        if (get_tensor_value(tensor, i) != expected) return false;
    }
    return true;
}

bool is_pair(const eshkol_tagged_value_t& value) {
    return ESHKOL_IS_CONS_COMPAT(value);
}

const arena_tagged_cons_cell_t* as_pair(const eshkol_tagged_value_t& value) {
    return is_pair(value) ? reinterpret_cast<const arena_tagged_cons_cell_t*>(value.data.ptr_val) : nullptr;
}

eshkol_tagged_value_t pair_car(const eshkol_tagged_value_t& value) {
    const arena_tagged_cons_cell_t* pair = as_pair(value);
    return pair ? pair->car : make_null();
}

eshkol_tagged_value_t pair_cdr(const eshkol_tagged_value_t& value) {
    const arena_tagged_cons_cell_t* pair = as_pair(value);
    return pair ? pair->cdr : make_null();
}

eshkol_tagged_value_t cons(arena_t* arena,
                           const eshkol_tagged_value_t& car,
                           const eshkol_tagged_value_t& cdr) {
    arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
    arena_tagged_cons_set_tagged_value(cell, false, &car);
    arena_tagged_cons_set_tagged_value(cell, true, &cdr);
    return make_heap_ptr(cell);
}

} // namespace

int main() {
    arena_t* arena = arena_create(1 << 26);
    if (!arena) return fail("failed to create arena");

    const fs::path temp_root = fs::temp_directory_path() / "eshkol-model-io-test";
    std::error_code ec;
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root, ec);

    const fs::path tensor_path = temp_root / "tensor.em";
    const fs::path model_path = temp_root / "model.em";
    const eshkol_tagged_value_t tensor_path_value = make_string(arena, tensor_path.string());
    const eshkol_tagged_value_t model_path_value = make_string(arena, model_path.string());

    eshkol_tensor_t* tensor = make_tensor(arena, {2, 2}, {1.0, 2.0, 3.0, 4.0});
    const eshkol_tagged_value_t tensor_value = make_heap_ptr(tensor);
    eshkol_tagged_value_t save_result{};
    eshkol_tensor_save_tagged(arena, &tensor_path_value, &tensor_value, &save_result);
    if (save_result.type != ESHKOL_VALUE_BOOL || save_result.data.int_val != 1) {
        return fail("tensor-save should succeed");
    }

    eshkol_tagged_value_t load_result{};
    eshkol_tensor_load_tagged(arena, &tensor_path_value, &load_result);
    if (!ESHKOL_IS_TENSOR_COMPAT(load_result) ||
        !tensor_equals(reinterpret_cast<eshkol_tensor_t*>(load_result.data.ptr_val), {2, 2}, {1.0, 2.0, 3.0, 4.0})) {
        return fail("tensor-load should round-trip tensor contents");
    }

    eshkol_tensor_t* scalar = arena_allocate_tensor_full(arena, 0, 1);
    set_tensor_value(scalar, 0, 42.5);
    const eshkol_tagged_value_t scalar_path_value = make_string(arena, (temp_root / "scalar.em").string());
    const eshkol_tagged_value_t scalar_value = make_heap_ptr(scalar);
    eshkol_tensor_save_tagged(arena, &scalar_path_value, &scalar_value, &save_result);
    eshkol_tensor_load_tagged(arena, &scalar_path_value, &load_result);
    auto* loaded_scalar = reinterpret_cast<eshkol_tensor_t*>(load_result.data.ptr_val);
    if (!loaded_scalar || loaded_scalar->num_dimensions != 0 || loaded_scalar->total_elements != 1 ||
        get_tensor_value(loaded_scalar, 0) != 42.5) {
        return fail("scalar tensor should round-trip");
    }

    eshkol_tensor_t* empty = make_empty_tensor(arena, {0, 3});
    const eshkol_tagged_value_t empty_path_value = make_string(arena, (temp_root / "empty.em").string());
    const eshkol_tagged_value_t empty_value = make_heap_ptr(empty);
    eshkol_tensor_save_tagged(arena, &empty_path_value, &empty_value, &save_result);
    eshkol_tensor_load_tagged(arena, &empty_path_value, &load_result);
    auto* loaded_empty = reinterpret_cast<eshkol_tensor_t*>(load_result.data.ptr_val);
    if (!loaded_empty || loaded_empty->num_dimensions != 2 || loaded_empty->dimensions[0] != 0 ||
        loaded_empty->dimensions[1] != 3 || loaded_empty->total_elements != 0) {
        return fail("empty tensor should round-trip");
    }

    const std::vector<std::uint64_t> large_dims{1024, 256};
    eshkol_tensor_t* large = make_pattern_tensor(arena, large_dims);
    const eshkol_tagged_value_t large_path_value = make_string(arena, (temp_root / "large.em").string());
    const eshkol_tagged_value_t large_value = make_heap_ptr(large);
    eshkol_tensor_save_tagged(arena, &large_path_value, &large_value, &save_result);
    eshkol_tensor_load_tagged(arena, &large_path_value, &load_result);
    if (!ESHKOL_IS_TENSOR_COMPAT(load_result) ||
        !tensor_matches_pattern(reinterpret_cast<eshkol_tensor_t*>(load_result.data.ptr_val), large_dims)) {
        return fail("large tensor should round-trip");
    }

    eshkol_tensor_t* w1 = make_tensor(arena, {2}, {5.0, 6.0});
    eshkol_tensor_t* b1 = make_tensor(arena, {1}, {7.0});
    eshkol_tagged_value_t model_entries = cons(
        arena,
        cons(arena, make_string(arena, "W1"), make_heap_ptr(w1)),
        cons(arena, cons(arena, make_string(arena, "b1"), make_heap_ptr(b1)), make_null()));

    eshkol_model_save_tagged(arena, &model_path_value, &model_entries, &save_result);
    if (save_result.type != ESHKOL_VALUE_BOOL || save_result.data.int_val != 1) {
        return fail("model-save should succeed");
    }

    eshkol_model_load_tagged(arena, &model_path_value, &load_result);
    if (!is_pair(load_result)) return fail("model-load should return a list");

    const eshkol_tagged_value_t first_entry = pair_car(load_result);
    const eshkol_tagged_value_t second_entry = pair_car(pair_cdr(load_result));
    if (std::string(reinterpret_cast<const char*>(pair_car(first_entry).data.ptr_val)) != "W1") {
        return fail("first model entry name should match");
    }
    if (!tensor_equals(reinterpret_cast<eshkol_tensor_t*>(pair_cdr(first_entry).data.ptr_val), {2}, {5.0, 6.0})) {
        return fail("first model tensor should match");
    }
    if (std::string(reinterpret_cast<const char*>(pair_car(second_entry).data.ptr_val)) != "b1") {
        return fail("second model entry name should match");
    }
    if (!tensor_equals(reinterpret_cast<eshkol_tensor_t*>(pair_cdr(second_entry).data.ptr_val), {1}, {7.0})) {
        return fail("second model tensor should match");
    }

    std::fstream corrupt(model_path, std::ios::in | std::ios::out | std::ios::binary);
    corrupt.seekp(0);
    corrupt.write("BAD!", 4);
    corrupt.close();
    eshkol_model_load_tagged(arena, &model_path_value, &load_result);
    if (load_result.type != ESHKOL_VALUE_NULL) {
        return fail("wrong magic should be rejected");
    }

    eshkol_model_save_tagged(arena, &model_path_value, &model_entries, &save_result);
    std::vector<std::uint8_t> bytes;
    {
        std::ifstream input(model_path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    bytes[4] = 2;
    bytes[5] = bytes[6] = bytes[7] = 0;
    {
        std::ofstream output(model_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    eshkol_model_load_tagged(arena, &model_path_value, &load_result);
    if (load_result.type != ESHKOL_VALUE_NULL) {
        return fail("unsupported version should be rejected");
    }

    eshkol_model_save_tagged(arena, &model_path_value, &model_entries, &save_result);
    bytes.clear();
    {
        std::ifstream input(model_path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    bytes.back() ^= 0xFF;
    {
        std::ofstream output(model_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    eshkol_model_load_tagged(arena, &model_path_value, &load_result);
    if (load_result.type != ESHKOL_VALUE_NULL) {
        return fail("crc mismatch should be rejected");
    }

    eshkol_model_save_tagged(arena, &model_path_value, &model_entries, &save_result);
    bytes.clear();
    {
        std::ifstream input(model_path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    bytes.resize(bytes.size() - 3);
    {
        std::ofstream output(model_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    eshkol_model_load_tagged(arena, &model_path_value, &load_result);
    if (load_result.type != ESHKOL_VALUE_NULL) {
        return fail("truncated file should be rejected");
    }

    std::cout << "PASS" << std::endl;
    fs::remove_all(temp_root, ec);
    arena_destroy(arena);
    return 0;
}
