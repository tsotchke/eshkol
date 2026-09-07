#include <eshkol/frontend/semantic_identity.h>

#include <atomic>
#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct BindingRecord {
    eshkol_binding_info_t info{};
    std::string name;
};

std::mutex g_identity_mutex;
std::unordered_map<eshkol_node_id_t, eshkol_binding_id_t> g_node_bindings;
std::unordered_map<eshkol_binding_id_t, BindingRecord> g_bindings;
std::unordered_map<eshkol_node_id_t, eshkol_typed_expr_info_t> g_typed_nodes;
std::unordered_map<std::string, eshkol_nominal_type_id_t> g_nominal_types;
std::unordered_map<std::string, eshkol_binding_id_t> g_builtins;
std::atomic<uint32_t> g_next_binding{1};
std::atomic<uint32_t> g_next_nominal{1};

eshkol_binding_id_t make_binding(const std::string& name,
                                 eshkol_module_id_t module,
                                 uint32_t ordinal,
                                 eshkol_binding_kind_t kind,
                                 eshkol_node_id_t declaration,
                                 bool mutable_binding) {
    const eshkol_binding_id_t id = g_next_binding.fetch_add(1, std::memory_order_relaxed);
    BindingRecord record;
    record.info.id = id;
    record.info.module = module;
    record.info.ordinal = ordinal;
    record.info.kind = kind;
    record.info.mutable_binding = mutable_binding ? 1 : 0;
    record.info.declaration = declaration;
    record.name = name;
    g_bindings.emplace(id, std::move(record));
    return id;
}

void record_binding(eshkol_node_id_t node_id, eshkol_binding_id_t id) {
    if (node_id != ESHKOL_NODE_ID_NONE) g_node_bindings[node_id] = id;
}

void record_type(eshkol_node_id_t node_id, eshkol_type_ref_t type) {
    if (node_id == ESHKOL_NODE_ID_NONE) return;
    g_typed_nodes[node_id] = {type, 0, 0, 1};
}

eshkol_type_ref_t literal_type(const eshkol_ast_t* ast) {
    if (!ast) return eshkol_type_dyn();
    switch (ast->type) {
        case ESHKOL_INT64:
        case ESHKOL_UINT64:
        case ESHKOL_INT32:
        case ESHKOL_UINT32:
            return eshkol_type_nominal(eshkol_nominal_type_intern("Int"));
        case ESHKOL_DOUBLE:
            return eshkol_type_nominal(eshkol_nominal_type_intern("Float"));
        case ESHKOL_STRING:
            return eshkol_type_nominal(eshkol_nominal_type_intern("String"));
        case ESHKOL_BOOL:
            return eshkol_type_nominal(eshkol_nominal_type_intern("Bool"));
        default:
            return eshkol_type_dyn();
    }
}

}  // namespace

extern "C" eshkol_binding_id_t eshkol_binding_id_for_node(eshkol_node_id_t node_id) {
    std::lock_guard<std::mutex> lock(g_identity_mutex);
    auto it = g_node_bindings.find(node_id);
    return it == g_node_bindings.end() ? ESHKOL_BINDING_ID_NONE : it->second;
}

extern "C" bool eshkol_binding_info(eshkol_binding_id_t id,
                                     eshkol_binding_info_t* out) {
    std::lock_guard<std::mutex> lock(g_identity_mutex);
    auto it = g_bindings.find(id);
    if (it == g_bindings.end()) return false;
    if (out) *out = it->second.info;
    return true;
}

extern "C" bool eshkol_typed_expr_info(eshkol_node_id_t node_id,
                                        eshkol_typed_expr_info_t* out) {
    std::lock_guard<std::mutex> lock(g_identity_mutex);
    auto it = g_typed_nodes.find(node_id);
    if (it == g_typed_nodes.end()) return false;
    if (out) *out = it->second;
    return true;
}

extern "C" eshkol_nominal_type_id_t eshkol_nominal_type_intern(const char* name) {
    if (!name || !name[0]) return ESHKOL_NOMINAL_TYPE_ID_NONE;
    std::lock_guard<std::mutex> lock(g_identity_mutex);
    auto it = g_nominal_types.find(name);
    if (it != g_nominal_types.end()) return it->second;
    const eshkol_nominal_type_id_t id = g_next_nominal.fetch_add(1, std::memory_order_relaxed);
    g_nominal_types.emplace(name, id);
    return id;
}

extern "C" eshkol_type_ref_t eshkol_type_dyn(void) {
    return {ESHKOL_TYPE_DYN, 1};
}

extern "C" eshkol_type_ref_t eshkol_type_value(void) {
    return {ESHKOL_TYPE_VALUE, 1};
}

extern "C" eshkol_type_ref_t eshkol_type_nominal(eshkol_nominal_type_id_t id) {
    return {ESHKOL_TYPE_NOMINAL, id};
}

namespace eshkol::frontend {

ImportSet ImportSet::library_named(std::string name) {
    ImportSet set;
    set.kind = Kind::Library;
    set.library = std::move(name);
    return set;
}

ImportSet ImportSet::only(ImportSet base_set, std::vector<std::string> selected) {
    ImportSet set;
    set.kind = Kind::Only;
    set.base = std::make_shared<const ImportSet>(std::move(base_set));
    set.names = std::move(selected);
    return set;
}

ImportSet ImportSet::except(ImportSet base_set, std::vector<std::string> excluded) {
    ImportSet set;
    set.kind = Kind::Except;
    set.base = std::make_shared<const ImportSet>(std::move(base_set));
    set.names = std::move(excluded);
    return set;
}

ImportSet ImportSet::prefixing(ImportSet base_set, std::string value) {
    ImportSet set;
    set.kind = Kind::Prefix;
    set.base = std::make_shared<const ImportSet>(std::move(base_set));
    set.prefix = std::move(value);
    return set;
}

ImportSet ImportSet::renaming(
    ImportSet base_set, std::vector<std::pair<std::string, std::string>> values) {
    ImportSet set;
    set.kind = Kind::Rename;
    set.base = std::make_shared<const ImportSet>(std::move(base_set));
    set.renames = std::move(values);
    return set;
}

void ImportResolver::define(LibraryInterface interface) {
    const std::string key = interface.name.empty()
        ? std::to_string(interface.module) : interface.name;
    interfaces_[key] = std::move(interface);
}

bool ImportResolver::evaluate(const ImportSet& set,
                              std::map<std::string, eshkol_binding_id_t>* out,
                              std::string* error) const {
    if (!out) return false;
    out->clear();
    return evaluate_impl(set, out, error);
}

bool ImportResolver::evaluate_impl(const ImportSet& set,
                                   std::map<std::string, eshkol_binding_id_t>* out,
                                   std::string* error) const {
    if (set.kind == ImportSet::Kind::Library) {
        auto it = interfaces_.find(set.library);
        if (it == interfaces_.end()) {
            if (error) *error = "unknown library: " + set.library;
            return false;
        }
        *out = it->second.exports;
        return true;
    }
    if (!set.base || !evaluate_impl(*set.base, out, error)) return false;

    if (set.kind == ImportSet::Kind::Only || set.kind == ImportSet::Kind::Except) {
        std::map<std::string, eshkol_binding_id_t> transformed;
        for (const auto& name : set.names) {
            auto it = out->find(name);
            if (set.kind == ImportSet::Kind::Only && it == out->end()) {
                if (error) *error = "import modifier names a non-exported binding: " + name;
                return false;
            }
            if (set.kind == ImportSet::Kind::Only) transformed.emplace(*it);
        }
        if (set.kind == ImportSet::Kind::Except) {
            transformed = *out;
            for (const auto& name : set.names) {
                if (out->find(name) == out->end()) {
                    if (error) *error = "except names a non-exported binding: " + name;
                    return false;
                }
                transformed.erase(name);
            }
        }
        *out = std::move(transformed);
        return true;
    }
    if (set.kind == ImportSet::Kind::Prefix) {
        std::map<std::string, eshkol_binding_id_t> transformed;
        for (const auto& item : *out)
            transformed.emplace(set.prefix + item.first, item.second);
        *out = std::move(transformed);
        return true;
    }

    std::map<std::string, eshkol_binding_id_t> transformed = *out;
    std::map<std::string, eshkol_binding_id_t> renamed;
    for (const auto& item : transformed) {
        auto rename = std::find_if(set.renames.begin(), set.renames.end(),
                                   [&](const auto& pair) { return pair.first == item.first; });
        const std::string& key = rename == set.renames.end() ? item.first : rename->second;
        auto [it, inserted] = renamed.emplace(key, item.second);
        if (!inserted) {
            if (error) *error = "import rename creates an ambiguous binding: " + key;
            return false;
        }
    }
    for (const auto& pair : set.renames) {
        if (transformed.find(pair.first) == transformed.end()) {
            if (error) *error = "rename names a non-exported binding: " + pair.first;
            return false;
        }
    }
    *out = std::move(renamed);
    return true;
}

bool ImportResolver::merge(const std::map<std::string, eshkol_binding_id_t>& bindings,
                           std::map<std::string, eshkol_binding_id_t>* into,
                           std::string* error) const {
    if (!into) return false;
    for (const auto& item : bindings) {
        auto [it, inserted] = into->emplace(item.first, item.second);
        if (!inserted && it->second != item.second) {
            if (error) *error = "ambiguous imported binding: " + item.first;
            return false;
        }
    }
    return true;
}

BindingResolver::BindingResolver(eshkol_module_id_t module) : module_(module) {
    scopes_.emplace_back();
}

eshkol_binding_id_t BindingResolver::declare(const std::string& name,
                                             eshkol_binding_kind_t kind,
                                             eshkol_node_id_t declaration,
                                             bool mutable_binding) {
    if (name.empty()) return ESHKOL_BINDING_ID_NONE;
    auto& scope = scopes_.back();
    auto existing = scope.find(name);
    if (existing != scope.end()) return existing->second;

    std::lock_guard<std::mutex> lock(g_identity_mutex);
    const eshkol_binding_id_t id = make_binding(name, module_, next_ordinal_++, kind,
                                                declaration, mutable_binding);
    scope.emplace(name, id);
    return id;
}

eshkol_binding_id_t BindingResolver::lookup(const std::string& name) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) return found->second;
    }
    return ESHKOL_BINDING_ID_NONE;
}

void BindingResolver::bind_node(const eshkol_ast_t* ast, eshkol_binding_id_t id) {
    if (!ast || id == ESHKOL_BINDING_ID_NONE) return;
    std::lock_guard<std::mutex> lock(g_identity_mutex);
    record_binding(ast->node_id, id);
}

void BindingResolver::type_node(const eshkol_ast_t* ast, eshkol_type_ref_t type) {
    if (!ast) return;
    std::lock_guard<std::mutex> lock(g_identity_mutex);
    record_type(ast->node_id, type);
}

void BindingResolver::visit_call(const eshkol_operations_t* op) {
    if (!op) return;
    visit(op->call_op.func);
    for (uint64_t i = 0; i < op->call_op.num_vars; ++i) visit(&op->call_op.variables[i]);
}

void BindingResolver::visit_definition(const eshkol_operations_t* op) {
    if (!op) return;
    if (op->define_op.value) {
        if (op->define_op.is_function) {
            scopes_.emplace_back();
            for (uint64_t i = 0; i < op->define_op.num_params; ++i) {
                const eshkol_ast_t* parameter = &op->define_op.parameters[i];
                if (parameter->type == ESHKOL_VAR && parameter->variable.id) {
                    const auto id = declare(parameter->variable.id, ESHKOL_BINDING_VALUE,
                                            parameter->node_id);
                    bind_node(parameter, id);
                }
            }
            visit(op->define_op.value);
            scopes_.pop_back();
        } else {
            visit(op->define_op.value);
        }
    }
}

void BindingResolver::visit_let(const eshkol_operations_t* op) {
    if (!op) return;
    const bool recursive = op->op == ESHKOL_LETREC_OP || op->op == ESHKOL_LETREC_STAR_OP;
    scopes_.emplace_back();
    if (recursive) {
        for (uint64_t i = 0; i < op->let_op.num_bindings; ++i) {
            const eshkol_ast_t* binding = &op->let_op.bindings[i];
            if (binding->type == ESHKOL_CONS && binding->cons_cell.car &&
                binding->cons_cell.car->type == ESHKOL_VAR && binding->cons_cell.car->variable.id) {
                const auto id = declare(binding->cons_cell.car->variable.id, ESHKOL_BINDING_VALUE,
                                        binding->cons_cell.car->node_id);
                bind_node(binding->cons_cell.car, id);
            }
        }
    }
    for (uint64_t i = 0; i < op->let_op.num_bindings; ++i) {
        const eshkol_ast_t* binding = &op->let_op.bindings[i];
        if (binding->type != ESHKOL_CONS || !binding->cons_cell.car || !binding->cons_cell.cdr)
            continue;
        if (!recursive) visit(binding->cons_cell.cdr);
        if (!recursive && binding->cons_cell.car->type == ESHKOL_VAR &&
            binding->cons_cell.car->variable.id) {
            const auto id = declare(binding->cons_cell.car->variable.id, ESHKOL_BINDING_VALUE,
                                    binding->cons_cell.car->node_id);
            bind_node(binding->cons_cell.car, id);
        }
        if (recursive) visit(binding->cons_cell.cdr);
    }
    visit(op->let_op.body);
    scopes_.pop_back();
}

void BindingResolver::visit_body(const eshkol_ast_t* ast) {
    if (!ast) return;
    if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_SEQUENCE_OP) {
        for (uint64_t i = 0; i < ast->operation.sequence_op.num_expressions; ++i)
            visit(&ast->operation.sequence_op.expressions[i]);
        return;
    }
    visit(ast);
}

void BindingResolver::visit(const eshkol_ast_t* ast) {
    if (!ast) return;
    type_node(ast, literal_type(ast));
    if (ast->type == ESHKOL_VAR) {
        const std::string name = ast->variable.id ? ast->variable.id : "";
        eshkol_binding_id_t id = lookup(name);
        if (id == ESHKOL_BINDING_ID_NONE) {
            std::lock_guard<std::mutex> lock(g_identity_mutex);
            auto builtin = g_builtins.find(name);
            if (builtin != g_builtins.end()) id = builtin->second;
            else {
                id = make_binding(name, ESHKOL_MODULE_ID_NONE, 0,
                                  ESHKOL_BINDING_BUILTIN, ESHKOL_NODE_ID_NONE, false);
                g_builtins.emplace(name, id);
            }
        }
        bind_node(ast, id);
        return;
    }
    if (ast->type == ESHKOL_CONS) {
        visit(ast->cons_cell.car);
        visit(ast->cons_cell.cdr);
        return;
    }
    if (ast->type != ESHKOL_OP) return;

    const auto* op = &ast->operation;
    switch (op->op) {
        case ESHKOL_DEFINE_OP: {
            if (op->define_op.name) {
                const auto id = lookup(op->define_op.name);
                bind_node(ast, id);
            }
            visit_definition(op);
            break;
        }
        case ESHKOL_LAMBDA_OP:
            scopes_.emplace_back();
            for (uint64_t i = 0; i < op->lambda_op.num_params; ++i) {
                const eshkol_ast_t* parameter = &op->lambda_op.parameters[i];
                if (parameter->type == ESHKOL_VAR && parameter->variable.id) {
                    const auto id = declare(parameter->variable.id, ESHKOL_BINDING_VALUE,
                                            parameter->node_id);
                    bind_node(parameter, id);
                }
            }
            visit(op->lambda_op.body);
            scopes_.pop_back();
            break;
        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            visit_let(op);
            break;
        case ESHKOL_CALL_OP:
        case ESHKOL_IF_OP:
        case ESHKOL_WHEN_OP:
        case ESHKOL_UNLESS_OP:
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
        case ESHKOL_COND_OP:
            visit_call(op);
            break;
        case ESHKOL_SEQUENCE_OP:
            visit_body(ast);
            break;
        case ESHKOL_SET_OP:
            if (op->set_op.name) {
                const auto id = lookup(op->set_op.name);
                if (id == ESHKOL_BINDING_ID_NONE && diagnostics_)
                    diagnostics_->push_back({ast->node_id, "set! references an unbound binding"});
            }
            visit(op->set_op.value);
            break;
        default:
            /* Operators with a distinct union payload are handled by their
             * owning semantic pass. Never interpret that payload as call_op:
             * doing so would turn an untyped compile-time form into a pointer
             * walk and make name resolution itself unsafe. */
            break;
    }
}

ResolutionResult BindingResolver::resolve(const std::vector<eshkol_ast_t>& forms) {
    ResolutionResult result;
    diagnostics_ = &result.diagnostics;

    for (const auto& form : forms) {
        if (form.type == ESHKOL_OP && form.operation.op == ESHKOL_DEFINE_OP &&
            form.operation.define_op.name) {
            const auto id = declare(form.operation.define_op.name, ESHKOL_BINDING_VALUE,
                                    form.node_id);
            bind_node(&form, id);
        }
    }
    for (const auto& form : forms) visit(&form);

    for (const auto& form : forms) {
        (void)form;
    }
    result.identifiers = g_node_bindings.size();
    result.resolved = result.identifiers;
    diagnostics_ = nullptr;
    return result;
}

}  // namespace eshkol::frontend
