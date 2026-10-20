#ifndef ESHKOL_STDLIB_IDENTITY_H
#define ESHKOL_STDLIB_IDENTITY_H

#include <eshkol/stdlib_identity_config.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <string>

namespace eshkol::stdlib_identity {

inline constexpr llvm::StringLiteral kSymbolPrefix("eshkol_stdlib_identity_");

inline bool matches_object(llvm::MemoryBufferRef buffer, std::string& found) {
    auto object_or_error = llvm::object::ObjectFile::createObjectFile(buffer);
    if (!object_or_error) {
        found = "unreadable object";
        llvm::consumeError(object_or_error.takeError());
        return false;
    }

    for (const llvm::object::SymbolRef& symbol : (*object_or_error)->symbols()) {
        auto name_or_error = symbol.getName();
        if (!name_or_error) {
            llvm::consumeError(name_or_error.takeError());
            continue;
        }
        llvm::StringRef name = *name_or_error;
        name.consume_front("_"); // Mach-O external-symbol spelling.
        if (!name.starts_with(kSymbolPrefix)) continue;
        if (name == ESHKOL_STDLIB_IDENTITY_SYMBOL) return true;
        found = name.str();
        return false;
    }

    found = "missing embedded identity";
    return false;
}

inline bool matches_module(const llvm::Module& module, std::string& found) {
    for (const llvm::GlobalVariable& global : module.globals()) {
        const llvm::StringRef name = global.getName();
        if (!name.starts_with(kSymbolPrefix)) continue;
        if (name == ESHKOL_STDLIB_IDENTITY_SYMBOL) return true;
        found = name.str();
        return false;
    }

    found = "missing embedded identity";
    return false;
}

} // namespace eshkol::stdlib_identity

#endif
