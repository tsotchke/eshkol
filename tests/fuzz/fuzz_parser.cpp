/*
 * fuzz_parser.cpp — libFuzzer entry point for the Eshkol parser.
 *
 * Build and run:
 *   cd build-fuzz && cmake .. -DESHKOL_ENABLE_FUZZ=ON
 *   make fuzz_parser
 *   ./fuzz_parser corpus/
 *
 * Input: arbitrary byte buffer. Converted to a std::istringstream and
 * piped through `eshkol_parse_next_ast_from_stream` in a loop until
 * EOF or an invalid AST is reported. The harness must not crash or
 * leak regardless of input bytes — any segfault, UBSan hit, or ASan
 * report is a bug.
 *
 * #187 — fuzzing harness, landed under v1.2-scale.
 */

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "eshkol/eshkol.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    /* Reject absurdly large inputs so the fuzzer corpus doesn't
     * swamp tmpfs; real parse inputs top out in the low MB. */
    if (size > (1u << 20)) return 0;

    std::string src(reinterpret_cast<const char*>(data), size);
    std::istringstream stream(src);
    for (int i = 0; i < 256; i++) {
        eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(stream);
        /* ESHKOL_INVALID signals a parse error or EOF; stop the
         * loop either way. The point of the harness is to crash the
         * parser if the input triggers a bug, not to validate
         * syntax. */
        if (ast.type == ESHKOL_INVALID) break;
        /* Free any heap allocations the AST carries before the next
         * iteration overwrites the local. String/tensor payloads use
         * `new[]` so eshkol_ast_clean's delete[] is correct. */
        eshkol_ast_clean(&ast);
    }
    return 0;
}
