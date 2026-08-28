#ifndef ESHKOL_FRONTEND_SEMANTIC_IDENTITY_H
#define ESHKOL_FRONTEND_SEMANTIC_IDENTITY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
#include <map>
#include <memory>
#include <string>
#include <vector>
#endif

#include <eshkol/eshkol.h>
#include <eshkol/frontend/node_identity.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_BINDING_ID_NONE ((uint32_t)0)
#define ESHKOL_SYMBOL_ID_NONE ((uint32_t)0)
#define ESHKOL_MODULE_ID_NONE ((uint32_t)0)
#define ESHKOL_NOMINAL_TYPE_ID_NONE ((uint32_t)0)

typedef uint32_t eshkol_binding_id_t;
typedef uint32_t eshkol_symbol_id_t;
typedef uint32_t eshkol_module_id_t;
typedef uint32_t eshkol_nominal_type_id_t;
typedef uint32_t eshkol_index_ref_t;
typedef uint32_t eshkol_effect_row_ref_t;

typedef enum eshkol_type_term_kind {
    ESHKOL_TYPE_DYN = 1,
    ESHKOL_TYPE_VALUE = 2,
    ESHKOL_TYPE_ANY = 3,
    ESHKOL_TYPE_NOMINAL = 4,
    ESHKOL_TYPE_FUNCTION = 5
} eshkol_type_term_kind_t;

typedef struct eshkol_type_ref {
    eshkol_type_term_kind_t kind;
    uint32_t id;
} eshkol_type_ref_t;

typedef struct eshkol_typed_expr_info {
    eshkol_type_ref_t type;
    eshkol_index_ref_t index;
    eshkol_effect_row_ref_t effects;
    uint8_t checked;
} eshkol_typed_expr_info_t;

typedef enum eshkol_binding_kind {
    ESHKOL_BINDING_VALUE = 0,
    ESHKOL_BINDING_SYNTAX = 1,
    ESHKOL_BINDING_BUILTIN = 2,
    ESHKOL_BINDING_IMPORTED = 3
} eshkol_binding_kind_t;

typedef struct eshkol_binding_info {
    eshkol_binding_id_t id;
    eshkol_module_id_t module;
    uint32_t ordinal;
    eshkol_binding_kind_t kind;
    uint8_t mutable_binding;
    eshkol_node_id_t declaration;
} eshkol_binding_info_t;

eshkol_binding_id_t eshkol_binding_id_for_node(eshkol_node_id_t node_id);
bool eshkol_binding_info(eshkol_binding_id_t id, eshkol_binding_info_t* out);
bool eshkol_typed_expr_info(eshkol_node_id_t node_id, eshkol_typed_expr_info_t* out);
eshkol_nominal_type_id_t eshkol_nominal_type_intern(const char* name);
eshkol_type_ref_t eshkol_type_dyn(void);
eshkol_type_ref_t eshkol_type_value(void);
eshkol_type_ref_t eshkol_type_nominal(eshkol_nominal_type_id_t id);

#ifdef __cplusplus
}

namespace eshkol::frontend {

using BindingId = eshkol_binding_id_t;
using SymbolId = eshkol_symbol_id_t;
using ModuleId = eshkol_module_id_t;
using NominalTypeId = eshkol_nominal_type_id_t;
using IndexRef = eshkol_index_ref_t;
using EffectRowRef = eshkol_effect_row_ref_t;
using TypeRef = eshkol_type_ref_t;

constexpr TypeRef Dyn{ESHKOL_TYPE_DYN, 1};
constexpr TypeRef Value{ESHKOL_TYPE_VALUE, 1};

struct LibraryInterface {
    eshkol_module_id_t module = ESHKOL_MODULE_ID_NONE;
    std::string name;
    std::map<std::string, eshkol_binding_id_t> exports;
};

struct ImportSet {
    enum class Kind { Library, Only, Except, Prefix, Rename };
    Kind kind = Kind::Library;
    std::string library;
    std::shared_ptr<const ImportSet> base;
    std::vector<std::string> names;
    std::vector<std::pair<std::string, std::string>> renames;
    std::string prefix;

    static ImportSet library_named(std::string name);
    static ImportSet only(ImportSet base, std::vector<std::string> names);
    static ImportSet except(ImportSet base, std::vector<std::string> names);
    static ImportSet prefixing(ImportSet base, std::string prefix);
    static ImportSet renaming(ImportSet base,
                              std::vector<std::pair<std::string, std::string>> renames);
};

class ImportResolver {
public:
    void define(LibraryInterface interface);
    bool evaluate(const ImportSet& set, std::map<std::string, eshkol_binding_id_t>* out,
                  std::string* error = nullptr) const;
    bool merge(const std::map<std::string, eshkol_binding_id_t>& bindings,
               std::map<std::string, eshkol_binding_id_t>* into,
               std::string* error = nullptr) const;

private:
    std::map<std::string, LibraryInterface> interfaces_;
    bool evaluate_impl(const ImportSet& set, std::map<std::string, eshkol_binding_id_t>* out,
                       std::string* error) const;
};

struct ResolutionDiagnostic {
    eshkol_node_id_t node_id = ESHKOL_NODE_ID_NONE;
    std::string message;
};

struct ResolutionResult {
    size_t identifiers = 0;
    size_t resolved = 0;
    std::vector<ResolutionDiagnostic> diagnostics;
    bool ok() const { return diagnostics.empty(); }
};

class BindingResolver {
public:
    explicit BindingResolver(eshkol_module_id_t module = ESHKOL_MODULE_ID_NONE);
    ResolutionResult resolve(const std::vector<eshkol_ast_t>& forms);

private:
    eshkol_module_id_t module_;
    uint32_t next_ordinal_ = 1;
    std::vector<std::map<std::string, eshkol_binding_id_t>> scopes_;
    std::vector<ResolutionDiagnostic>* diagnostics_ = nullptr;

    eshkol_binding_id_t declare(const std::string& name, eshkol_binding_kind_t kind,
                                eshkol_node_id_t declaration, bool mutable_binding = false);
    eshkol_binding_id_t lookup(const std::string& name) const;
    void visit(const eshkol_ast_t* ast);
    void visit_body(const eshkol_ast_t* ast);
    void visit_call(const eshkol_operations_t* op);
    void visit_definition(const eshkol_operations_t* op);
    void visit_let(const eshkol_operations_t* op);
    void bind_node(const eshkol_ast_t* ast, eshkol_binding_id_t id);
    void type_node(const eshkol_ast_t* ast, eshkol_type_ref_t type);
};

}  // namespace eshkol::frontend
#endif

#endif
