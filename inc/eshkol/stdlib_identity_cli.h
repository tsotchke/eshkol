#ifndef ESHKOL_STDLIB_IDENTITY_CLI_H
#define ESHKOL_STDLIB_IDENTITY_CLI_H

#include <eshkol/stdlib_identity.h>
#include <llvm/IR/LLVMContext.h>

#include <cstdio>
#include <string>

namespace eshkol::stdlib_identity {

// Metadata only: these commands never load a stdlib or evaluate Eshkol code.
inline bool print_expected_identity(FILE* output) {
    return std::fprintf(output,
        "schema=eshkol.stdlib-identity/v1\n"
        "release=%s\nabi=%s\nsource=%s\nsymbol=%s\n",
        ESHKOL_STDLIB_IDENTITY_RELEASE, ESHKOL_STDLIB_IDENTITY_ABI,
        ESHKOL_STDLIB_IDENTITY_SOURCE, ESHKOL_STDLIB_IDENTITY_SYMBOL) >= 0;
}

inline bool matches_file(const char* path, std::string& found) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
        found = "unreadable artifact: " + buffer.getError().message();
        return false;
    }
    const auto* begin = reinterpret_cast<const unsigned char*>(
        (*buffer)->getBufferStart());
    const auto* end = reinterpret_cast<const unsigned char*>(
        (*buffer)->getBufferEnd());
    if (llvm::isBitcode(begin, end)) {
        llvm::LLVMContext context;
        auto module = llvm::parseBitcodeFile((*buffer)->getMemBufferRef(), context);
        if (!module) {
            found = "unreadable bitcode: " + llvm::toString(module.takeError());
            return false;
        }
        return matches_module(**module, found);
    }
    return matches_object((*buffer)->getMemBufferRef(), found);
}

inline int check_file_identity(const char* path, FILE* output, FILE* error) {
    std::string found;
    if (!matches_file(path, found)) {
        std::fprintf(error, "stdlib identity check failed for %s: %s; expected %s\n",
                     path, found.c_str(), ESHKOL_STDLIB_IDENTITY_SYMBOL);
        return 1;
    }
    if (!print_expected_identity(output)) return 1;
    return std::fprintf(output, "status=PASS\n") >= 0 ? 0 : 1;
}

} // namespace eshkol::stdlib_identity
#endif
