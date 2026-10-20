#ifndef ESHKOL_STDLIB_IDENTITY_H
#define ESHKOL_STDLIB_IDENTITY_H

#include <eshkol/stdlib_identity_config.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <string>

namespace eshkol::stdlib_identity {

inline constexpr llvm::StringLiteral kSymbolPrefix("eshkol_stdlib_identity_");

/**
 * @brief Checks that an object contains exactly one defined stdlib identity
 *        symbol and that its name matches this build.
 *
 * Mach-O's leading underscore is normalized before comparison. A declaration,
 * foreign marker, duplicate marker, unreadable symbol name, or unreadable
 * marker flags rejects the object. On failure, @p found receives a concise
 * reason or the discovered foreign symbol name.
 *
 * @param buffer Object bytes to inspect before loading or linking the artifact.
 * @param found Receives the mismatch reason or foreign identity name.
 * @return true only for one defined expected identity marker.
 */
inline bool matches_object(llvm::MemoryBufferRef buffer, std::string& found) {
    auto object_or_error = llvm::object::ObjectFile::createObjectFile(buffer);
    if (!object_or_error) {
        found = "unreadable object";
        llvm::consumeError(object_or_error.takeError());
        return false;
    }

    size_t marker_count = 0;
    bool expected_marker = false;
    for (const llvm::object::SymbolRef& symbol : (*object_or_error)->symbols()) {
        auto name_or_error = symbol.getName();
        if (!name_or_error) {
            found = "unreadable object symbol name";
            llvm::consumeError(name_or_error.takeError());
            return false;
        }
        llvm::StringRef name = *name_or_error;
        name.consume_front("_"); // Mach-O external-symbol spelling.
        if (!name.starts_with(kSymbolPrefix)) continue;

        auto flags_or_error = symbol.getFlags();
        if (!flags_or_error) {
            found = "unreadable identity symbol flags";
            llvm::consumeError(flags_or_error.takeError());
            return false;
        }
        if (*flags_or_error & llvm::object::BasicSymbolRef::SF_Undefined) {
            found = "undefined identity marker: " + name.str();
            return false;
        }
        if (++marker_count != 1) {
            found = "multiple embedded identity markers";
            return false;
        }
        expected_marker = (name == ESHKOL_STDLIB_IDENTITY_SYMBOL);
        if (!expected_marker) found = name.str();
    }

    if (marker_count == 0) {
        found = "missing embedded identity";
        return false;
    }
    return expected_marker;
}

/**
 * @brief Checks that bitcode contains exactly one defined stdlib identity
 *        global and that its name matches this build.
 *
 * A declaration-only global, foreign marker, duplicate marker, or missing
 * marker rejects the module. On failure, @p found receives a concise reason
 * or the discovered foreign symbol name.
 *
 * @param module Parsed LLVM module to inspect before adding it to the JIT.
 * @param found Receives the mismatch reason or foreign identity name.
 * @return true only for one initialized expected identity global.
 */
inline bool matches_module(const llvm::Module& module, std::string& found) {
    size_t marker_count = 0;
    bool expected_marker = false;
    for (const llvm::GlobalVariable& global : module.globals()) {
        const llvm::StringRef name = global.getName();
        if (!name.starts_with(kSymbolPrefix)) continue;
        if (!global.hasInitializer()) {
            found = "declaration-only identity marker: " + name.str();
            return false;
        }
        if (++marker_count != 1) {
            found = "multiple embedded identity markers";
            return false;
        }
        expected_marker = (name == ESHKOL_STDLIB_IDENTITY_SYMBOL);
        if (!expected_marker) found = name.str();
    }

    if (marker_count == 0) {
        found = "missing embedded identity";
        return false;
    }
    return expected_marker;
}

} // namespace eshkol::stdlib_identity

#endif
