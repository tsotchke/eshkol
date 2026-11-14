#ifndef ESHKOL_STDLIB_IDENTITY_H
#define ESHKOL_STDLIB_IDENTITY_H

#include <eshkol/logger.h>
#include <eshkol/platform_runtime.h>
#include <eshkol/stdlib_identity_config.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <mutex>
#include <set>
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

/**
 * @brief Checks whether a stdlib artifact on disk carries this build's
 *        identity, reading as little of it as the answer needs.
 *
 * `stdlib.bc` is opened lazily: the identity global lives at module level, so
 * no function body is materialized. Any other name is read as an object file.
 *
 * @param path Path of a `stdlib.bc` or `stdlib.o` candidate.
 * @return true only when the file is readable and carries the expected marker.
 */
inline bool file_belongs_to_this_build(const std::string& path,
                                       std::string* reason = nullptr) {
    std::string local;
    std::string& found = reason ? *reason : local;
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
        found = "unreadable file";
        return false;
    }
    if (llvm::StringRef(path).ends_with(".bc")) {
        llvm::LLVMContext context;
        auto module = llvm::getLazyBitcodeModule(
            (*buffer)->getMemBufferRef(), context,
            /*ShouldLazyLoadMetadata=*/true);
        if (!module) {
            llvm::consumeError(module.takeError());
            found = "unreadable bitcode";
            return false;
        }
        return matches_module(**module, found);
    }
    return matches_object((*buffer)->getMemBufferRef(), found);
}

/**
 * @brief Resolve a precompiled stdlib artifact (`stdlib.bc` or `stdlib.o`)
 *        for the running compiler.
 *
 * Every loader and the driver resolve through this one function, so they all
 * agree on which copy is this compiler's. A copy in a tier someone chose
 * (env, `-L`, beside the compiler, the working tree) is returned as found and
 * the caller's identity validation accepts or refuses it. A copy that sits
 * only in a system prefix and was produced by another build is not a
 * candidate at all: the compiler proceeds as it does on a machine with no
 * system install, loading the standard library from source.
 *
 * @param roots Ordered roots from platform::install_library_roots().
 * @param leaf  `"stdlib.bc"` or `"stdlib.o"`.
 */
inline platform::ResolvedInstallArtifact resolve_artifact(
    const std::vector<platform::InstallSearchRoot>& roots,
    const std::string& leaf) {
    return platform::resolve_install_artifact(
        roots, {leaf}, [](const std::string& path) {
            std::string reason;
            if (file_belongs_to_this_build(path, &reason)) return true;
            // Say it once per file: an operator with an older install in a
            // system prefix can see why it is not being used.
            static std::mutex reported_guard;
            static std::set<std::string> reported;
            bool first_report = false;
            {
                std::lock_guard<std::mutex> lock(reported_guard);
                first_report = reported.insert(path).second;
            }
            if (first_report) {
                eshkol_info(
                    "Passing over system stdlib %s: %s; this compiler expects %s. "
                    "The standard library is taken from the next location or from source",
                    path.c_str(), reason.empty() ? "another build's identity" : reason.c_str(),
                    static_cast<const char*>(ESHKOL_STDLIB_IDENTITY_SYMBOL));
            }
            return false;
        });
}

} // namespace eshkol::stdlib_identity

#endif
