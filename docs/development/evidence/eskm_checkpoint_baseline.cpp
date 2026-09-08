// Valid-input microbenchmark of the native ESKM codec; see ../ESKM_HANDOFF.md.
// Link with section GC to omit the unused public runtime/arena entry points.
#include "../../../lib/core/model_io.cpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2 || !std::filesystem::is_directory(argv[1])) return 2;
    const auto path = std::filesystem::path(argv[1]) / "baseline.eskm";
    if (std::filesystem::exists(path)) return 2;
    std::cout << "elements,file_bytes,iteration,write_us,parse_us\n";
    for (const std::size_t count : {512U, 32768U, 131072U}) {
        std::vector<std::int64_t> elements(count);
        std::vector<std::uint64_t> expected(count);
        for (std::size_t i = 0; i < count; ++i) {
            const double value = static_cast<double>(static_cast<int>(i % 1024) - 512) / 8;
            elements[i] = std::bit_cast<std::int64_t>(value);
            expected[i] = std::bit_cast<std::uint64_t>(value);
        }
        std::uint64_t dimension = count;
        eshkol_tensor_t tensor{};
        tensor.num_dimensions = 1;
        tensor.total_elements = count;
        tensor.dimensions = &dimension;
        tensor.elements = elements.data();
        const std::vector<TensorRecordView> input{{"weights", &tensor}};
        for (int iteration = -3; iteration < 15; ++iteration) {
            std::vector<ParsedTensorRecord> parsed;
            const auto begin = std::chrono::steady_clock::now();
            if (!write_checkpoint(path.c_str(), input)) return 1;
            const auto written = std::chrono::steady_clock::now();
            if (!parse_checkpoint(path.c_str(), &parsed)) return 1;
            const auto end = std::chrono::steady_clock::now();
            if (parsed.size() != 1 || parsed[0].name != "weights" ||
                parsed[0].ndims != 1 || parsed[0].dims != std::vector<std::uint64_t>{count} ||
                parsed[0].element_bits != expected ||
                std::filesystem::file_size(path) != 44 + 8 * count) return 1;
            if (iteration >= 0) {
                std::cout << count << ',' << std::filesystem::file_size(path) << ','
                          << iteration << ','
                          << std::chrono::duration<double, std::micro>(written - begin).count()
                          << ',' << std::chrono::duration<double, std::micro>(end - written).count()
                          << '\n';
            }
        }
    }
    std::filesystem::remove(path);
}
