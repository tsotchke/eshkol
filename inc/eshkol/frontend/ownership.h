#pragma once

#include <eshkol/eshkol.h>
#include <eshkol/strict_resource_contracts.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace eshkol {

// The module loader supplies contracts authenticated against its source inputs.
struct ResourceContracts {
    std::set<std::string> untrusted_modules;
    std::map<const char*, const strict_resources::Contract*> definitions;
    std::map<const char*, const strict_resources::Contract*> stdlib_contracts;
    std::set<const char*> scalar_primitives;
};

enum class OwnershipState { UNOWNED, OWNED, MOVED, BORROWED, MAYBE_MOVED };
using OwnershipBindings = std::map<std::string, OwnershipState>;

// Code generation uses a provisional environment until the JIT admits its module.
class OwnershipTransaction {
public:
    explicit OwnershipTransaction(OwnershipBindings& session);
    ~OwnershipTransaction();
    OwnershipTransaction(const OwnershipTransaction&) = delete;
    OwnershipTransaction& operator=(const OwnershipTransaction&) = delete;
    void commit();
    static OwnershipBindings* current();

private:
    OwnershipBindings& session_;
    OwnershipBindings pending_;
    OwnershipBindings* previous_;
};

// An interactive caller supplies the accepted global bindings for its session.
// Successful analysis advances that environment. Refusal preserves its input.
std::vector<std::string> checkOwnership(const std::vector<eshkol_ast_t>& forms,
                                        const ResourceContracts* strict = nullptr,
                                        OwnershipBindings* globals = nullptr);

}
