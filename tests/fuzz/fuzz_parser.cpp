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
#include "eshkol/frontend/ast_strings.h"

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
        /* Free the tensor arrays the AST carries before the next
         * iteration overwrites the local. String payloads belong to the
         * AST string owner (ADR-0021) and are released below. */
        eshkol_ast_clean(&ast);
    }
    /* Each input is a complete compilation: release its AST strings so a
     * long campaign stays inside libFuzzer's RSS limit, and so a stale AST
     * pointer from a previous input is a use-after-free, not a silent read. */
    eshkol_ast_strings_teardown();
    return 0;
}
