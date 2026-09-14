/**
 * @file static_callee_binding.h
 * @brief When a variable may be resolved to an llvm::Function at compile time.
 *
 * Binding a variable to a lambda records a static alias, `<name>_func`, in
 * the symbol tables (inside a function also `<function>.<name>_func`). Every
 * static fast path reads those aliases instead of the variable's runtime
 * value: direct calls, apply, map, reduce, remove, and the differentiation
 * operators. An alias is only a correct stand-in for the variable while two
 * things hold, and this header is where both are decided:
 *
 *  1. The binding keeps the value it was created with. A binding that is the
 *     target of a set! anywhere in its scope, or a top-level name that is
 *     defined more than once, gets no alias; calls through it dispatch on the
 *     runtime value. bindStaticCallee() applies that rule for every binding
 *     form.
 *
 *  2. The alias belongs to the binding the name denotes at the use site. An
 *     alias outlives its binding form (the let family keeps scoped aliases
 *     after the body), so a later binding of the same name in the same
 *     function (a sibling let, a parameter, a loop variable) would otherwise
 *     be resolved to an unrelated lambda. Each alias records the storage of
 *     the binding that created it, and staticCalleeHiddenByRuntimeBinding()
 *     accepts an alias only when that storage is the one the name currently
 *     denotes.
 */

#ifndef ESHKOL_BACKEND_STATIC_CALLEE_BINDING_H
#define ESHKOL_BACKEND_STATIC_CALLEE_BINDING_H

#include <llvm/IR/Argument.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>

#include <string>
#include <mutex>
#include <unordered_map>

namespace eshkol {

using StaticCalleeSymbolTable = std::unordered_map<std::string, llvm::Value*>;

/** @brief Unscoped alias key for @p name: `<name>_func`. */
inline std::string staticCalleeKey(const std::string& name) {
    return name + "_func";
}

/** @brief Alias key for @p name scoped to @p fn: `<fn>.<name>_func`. */
inline std::string scopedStaticCalleeKey(const llvm::Function* fn, const std::string& name) {
    return fn ? fn->getName().str() + "." + staticCalleeKey(name) : staticCalleeKey(name);
}

/** @brief Binding facts keyed by the existing alias key and symbol table. */
using StaticCalleeOwnerTables = std::unordered_map<
    const StaticCalleeSymbolTable*, std::unordered_map<std::string, llvm::Value*>>;

inline std::mutex& staticCalleeOwnerMutex() {
    static std::mutex mutex;
    return mutex;
}

inline StaticCalleeOwnerTables& staticCalleeOwnerTables() {
    static StaticCalleeOwnerTables tables;
    return tables;
}

inline llvm::Value* staticCalleeOwner(const StaticCalleeSymbolTable* table,
                                      const std::string& alias_key) {
    if (!table) return nullptr;
    std::lock_guard<std::mutex> lock(staticCalleeOwnerMutex());
    auto table_it = staticCalleeOwnerTables().find(table);
    if (table_it == staticCalleeOwnerTables().end()) return nullptr;
    auto owner_it = table_it->second.find(alias_key);
    return owner_it == table_it->second.end() ? nullptr : owner_it->second;
}

inline void eraseStaticCallee(StaticCalleeSymbolTable* table,
                              const std::string& alias_key) {
    if (!table) return;
    table->erase(alias_key);
    std::lock_guard<std::mutex> lock(staticCalleeOwnerMutex());
    auto table_it = staticCalleeOwnerTables().find(table);
    if (table_it != staticCalleeOwnerTables().end()) {
        table_it->second.erase(alias_key);
        if (table_it->second.empty()) staticCalleeOwnerTables().erase(table_it);
    }
}

inline void recordStaticCallee(StaticCalleeSymbolTable* table,
                               const std::string& alias_key,
                               llvm::Function* fn,
                               llvm::Value* storage) {
    if (!table) return;
    (*table)[alias_key] = fn;
    std::lock_guard<std::mutex> lock(staticCalleeOwnerMutex());
    staticCalleeOwnerTables()[table][alias_key] = storage;
}

/** Carry a known lambda alias through a lexical closure capture.
 *
 * The captured storage value is a new LLVM instruction/argument in the
 * nested function, but it represents the same source binding as
 * @p source_storage. Record the nested scoped alias against that local
 * capture representation so shadowing checks keep recognizing this binding.
 */
inline bool inheritStaticCalleeCapture(StaticCalleeSymbolTable* local,
                                       StaticCalleeSymbolTable* global,
                                       const llvm::Function* source_function,
                                       const llvm::Function* current_function,
                                       const std::string& name,
                                       llvm::Value* source_storage,
                                       llvm::Value* captured_storage) {
    if (!local || !current_function || !source_storage || !captured_storage) return false;

    const std::string source_key = source_function
        ? scopedStaticCalleeKey(source_function, name)
        : staticCalleeKey(name);
    llvm::Value* alias = nullptr;
    auto local_it = local->find(source_key);
    if (local_it != local->end()) alias = local_it->second;
    if ((!alias || !llvm::isa<llvm::Function>(alias)) && global) {
        auto global_it = global->find(source_key);
        if (global_it != global->end()) alias = global_it->second;
    }
    auto* fn = llvm::dyn_cast_or_null<llvm::Function>(alias);
    if (!fn) return false;

    llvm::Value* owner = staticCalleeOwner(local, source_key);
    if (!owner) owner = staticCalleeOwner(global, source_key);
    if (owner != source_storage) return false;

    const std::string current_key = scopedStaticCalleeKey(current_function, name);
    recordStaticCallee(local, current_key, fn, captured_storage);
    recordStaticCallee(global, current_key, fn, captured_storage);
    return true;
}

/**
 * @brief Record, or refuse, the static alias of a variable bound to @p fn.
 *
 * @param local      the symbol table of the scope being emitted
 * @param global     the module-wide symbol table
 * @param current    the function being emitted, or nullptr at module level
 * @param name       the variable
 * @param fn         the lambda's function
 * @param storage    the binding's storage (alloca, cell or global), recorded
 *                   as the alias's owner; may be nullptr
 * @param reassigned true when the binding is the target of a set! in its
 *                   scope or is a redefined top-level name
 * @return true when an alias was recorded
 *
 * A reassigned binding gets no alias, and the unscoped entry of the current
 * scope is removed so an alias of an enclosing binding of the same name
 * cannot stand in for it. Scoped aliases of enclosing bindings are rejected
 * at the use site by staticCalleeHiddenByRuntimeBinding(), because their
 * recorded storage differs.
 */
inline bool bindStaticCallee(StaticCalleeSymbolTable* local,
                             StaticCalleeSymbolTable* global,
                             const llvm::Function* current,
                             const std::string& name,
                             llvm::Function* fn,
                             llvm::Value* storage,
                             bool reassigned) {
    const std::string key = staticCalleeKey(name);
    if (reassigned || !fn || !storage) {
        if (local) {
            eraseStaticCallee(local, key);
            if (current) eraseStaticCallee(local, scopedStaticCalleeKey(current, name));
        }
        return false;
    }

    if (current) {
        // Inside a function: the unscoped key serves direct calls in this
        // scope; only the scoped key goes to the module table, where an
        // unscoped key would collide with same-named bindings elsewhere.
        const std::string scoped = scopedStaticCalleeKey(current, name);
        if (local) {
            recordStaticCallee(local, key, fn, storage);
            recordStaticCallee(local, scoped, fn, storage);
        }
        recordStaticCallee(global, scoped, fn, storage);
    } else {
        recordStaticCallee(local, key, fn, storage);
        recordStaticCallee(global, key, fn, storage);
    }
    return true;
}

/**
 * @brief True when @p name denotes a runtime value of @p current that no
 *        static alias describes, so every alias of the name must be ignored.
 *
 * A parameter or a local storage cell of the function being emitted
 * lexically hides the aliases of same-named bindings elsewhere: unscoped
 * top-level aliases, and scoped aliases left behind by an earlier binding in
 * the same function. The one exception is a scoped alias created by the
 * binding the name denotes now, recognised by its recorded storage.
 */
inline bool staticCalleeHiddenByRuntimeBinding(const StaticCalleeSymbolTable* local,
                                               const StaticCalleeSymbolTable* global,
                                               const llvm::Function* current,
                                               const std::string& name) {
    if (!current || !local) return false;
    auto it = local->find(name);
    if (it == local->end() || !it->second) return false;
    const llvm::Value* binding = it->second;

    bool owned_by_current = false;
    if (const auto* arg = llvm::dyn_cast<llvm::Argument>(binding)) {
        owned_by_current = arg->getParent() == current;
    } else if (const auto* inst = llvm::dyn_cast<llvm::Instruction>(binding)) {
        owned_by_current = inst->getFunction() == current;
    }
    if (!owned_by_current) return false;

    const std::string scoped = scopedStaticCalleeKey(current, name);
    llvm::Value* owner = staticCalleeOwner(local, scoped);
    if (!owner) owner = staticCalleeOwner(global, scoped);
    return owner != binding;
}

/** True when a top-level mutable binding should override any static route,
 * except where a current lexical binding has an alias owned by its storage. */
inline bool staticCalleeTopLevelBindingIsDynamic(const StaticCalleeSymbolTable* local,
                                                 const StaticCalleeSymbolTable* global,
                                                 const llvm::Function* current,
                                                 const std::string& name,
                                                 bool top_level_reassigned) {
    if (!top_level_reassigned) return false;
    if (current && local) {
        auto binding = local->find(name);
        if (binding != local->end() && binding->second) {
            const std::string scoped = scopedStaticCalleeKey(current, name);
            llvm::Value* owner = staticCalleeOwner(local, scoped);
            if (!owner) owner = staticCalleeOwner(global, scoped);
            if (owner == binding->second) return false;
        }
    }
    return true;
}

} // namespace eshkol

#endif // ESHKOL_BACKEND_STATIC_CALLEE_BINDING_H
