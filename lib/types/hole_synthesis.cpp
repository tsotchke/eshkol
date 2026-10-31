/**
 * @file hole_synthesis.cpp
 * @brief Type-directed completion of `??` synthesis holes (ADR-0052).
 *
 * See inc/eshkol/types/hole_synthesis.h for the contract. The search is a
 * bottom-up enumeration by term size over a typed component set:
 *
 *   leaves      the bindings in scope at the hole (locals first) whose type
 *               is known, and the literals 0, 1, 0.0, 1.0, "", #t, #f
 *   components  the functions in scope with fully known signatures (except
 *               the definition that contains the hole), and a table of
 *               builtin operators instantiated at the numeric types that
 *               occur among the leaves and the expected type
 *
 * A component is applied only to argument terms whose types are subtypes of
 * its parameter types, so every term built is well typed by construction;
 * each candidate of the expected type is then checked again by the type
 * checker itself in the hole's scope before it is offered. Candidates are
 * ranked by the number of distinct local bindings they use (a completion that
 * ignores a parameter is rarely the intended one), then by size, then by the
 * order in which they were built, which is deterministic.
 */
#include <eshkol/types/hole_synthesis.h>
#include <eshkol/types/type_relation.h>
#include <eshkol/logger.h>
#include <eshkol/frontend/syntax_color.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <unordered_set>

namespace eshkol::hott {

namespace {

struct Term {
    std::string text;
    TypeId type;
    size_t size;
    std::set<std::string> locals;  // distinct local bindings used
    size_t constants;              // literal leaves used
    size_t order;                  // creation order (deterministic tie-break)
};

struct Component {
    std::string name;
    std::vector<TypeId> params;
    TypeId result;
    bool commutative;
};

bool sameType(TypeId a, TypeId b) { return a.id == b.id; }

// Value is the gradual (dynamic) type: it fits everything consistently,
// so it says nothing about which term belongs in a hole.
bool dynamicType(TypeId t) { return t.id == BuiltinTypes::Value.id; }

bool fits(TypeRelation& rel, TypeId arg, TypeId param) {
    if (sameType(arg, param)) return true;
    if (dynamicType(arg) || dynamicType(param)) return false;
    return rel.isSubtype(arg, param);
}

bool concrete(TypeRelation&, TypeId t) {
    return t.id != BuiltinTypes::Invalid.id && !dynamicType(t) && t.id != BuiltinTypes::Never.id;
}

bool numericType(TypeRelation& rel, TypeId t) {
    return sameType(t, BuiltinTypes::Number) || rel.isSubtype(t, BuiltinTypes::Number);
}

// Compositions of `total` into `k` positive parts.
void compositions(size_t total, size_t k, std::vector<size_t>& cur,
                  std::vector<std::vector<size_t>>& out) {
    if (k == 0) {
        if (total == 0) out.push_back(cur);
        return;
    }
    for (size_t first = 1; first + (k - 1) <= total; ++first) {
        cur.push_back(first);
        compositions(total - first, k - 1, cur, out);
        cur.pop_back();
    }
}

}  // namespace

HoleSynthesizer::HoleSynthesizer() : options_() {}
HoleSynthesizer::HoleSynthesizer(Options options) : options_(options) {}

void HoleSynthesizer::beginDefinition(const std::string& owner) { owner_ = owner; }

void HoleSynthesizer::onHole(TypeChecker& checker, eshkol_ast_t* hole, TypeId expected) {
    TypeEnvironment& env = checker.environment();
    TypeRelation rel(env);

    // ---- leaves and in-scope functions --------------------------------
    std::vector<Term> leaves;
    std::vector<Component> components;
    size_t order = 0;
    const auto bindings = checker.context().visibleBindings();  // innermost first
    for (const auto& [name, type, depth] : bindings) {
        if (name == "??" || name == owner_ || name.empty()) continue;
        if (env.isFunctionType(type)) {
            const PiType* pi = env.getFunctionType(type);
            if (!pi || pi->is_dependent) continue;
            std::vector<TypeId> params = env.getFunctionParamTypes(type);
            TypeId ret = env.getFunctionReturnType(type);
            bool known = concrete(rel, ret) && !params.empty();
            for (TypeId p : params) known = known && concrete(rel, p);
            if (known) components.push_back({name, params, ret, false});
            continue;
        }
        const bool local = depth > 0;
        if (!local && !concrete(rel, type)) continue;
        Term t{name, type, 1, {}, 0, order++};
        if (local) t.locals.insert(name);
        leaves.push_back(std::move(t));
    }

    // ---- numeric types present: instantiate the operator table ---------
    std::vector<TypeId> numeric;
    auto note_numeric = [&](TypeId t) {
        if (!concrete(rel, t) || !numericType(rel, t)) return;
        for (TypeId n : numeric) if (sameType(n, t)) return;
        numeric.push_back(t);
    };
    note_numeric(expected);
    for (const Term& l : leaves) note_numeric(l.type);
    for (const Component& c : components) {
        for (TypeId p : c.params) note_numeric(p);
        note_numeric(c.result);
    }

    const TypeId B = BuiltinTypes::Boolean, S = BuiltinTypes::String, I = BuiltinTypes::Int64,
                 L = BuiltinTypes::List;
    for (TypeId T : numeric) {
        for (const char* op : {"+", "*", "max", "min"}) components.push_back({op, {T, T}, T, true});
        components.push_back({"-", {T, T}, T, false});
        components.push_back({"<", {T, T}, B, false});
        components.push_back({"<=", {T, T}, B, false});
        components.push_back({"=", {T, T}, B, true});
        components.push_back({"abs", {T}, T, false});
        if (T.flags & TYPE_FLAG_EXACT) {
            components.push_back({"quotient", {T, T}, T, false});
            components.push_back({"remainder", {T, T}, T, false});
            components.push_back({"even?", {T}, B, false});
            components.push_back({"zero?", {T}, B, false});
        }
        components.push_back({"number->string", {T}, S, false});
    }
    components.push_back({"not", {B}, B, false});
    components.push_back({"and", {B, B}, B, true});
    components.push_back({"or", {B, B}, B, true});
    components.push_back({"string-append", {S, S}, S, false});
    components.push_back({"string-length", {S}, I, false});
    components.push_back({"string-upcase", {S}, S, false});
    components.push_back({"string-downcase", {S}, S, false});
    components.push_back({"string=?", {S, S}, B, true});
    components.push_back({"length", {L}, I, false});
    components.push_back({"reverse", {L}, L, false});
    components.push_back({"append", {L, L}, L, false});
    components.push_back({"null?", {L}, B, false});
    if (concrete(rel, expected)) components.push_back({"if", {B, expected, expected}, expected, false});

    // Literal leaves.
    auto literal = [&](const char* text, TypeId type) {
        leaves.push_back(Term{text, type, 1, {}, 1, order++});
    };
    literal("0", I);
    literal("1", I);
    literal("0.0", BuiltinTypes::Float64);
    literal("1.0", BuiltinTypes::Float64);
    literal("\"\"", S);
    literal("#t", B);
    literal("#f", B);

    // ---- bottom-up enumeration by size ----------------------------------
    std::vector<std::vector<Term>> bank(options_.max_size + 1);
    std::unordered_set<std::string> seen;
    for (Term& l : leaves) {
        if (seen.insert(l.text).second) bank[1].push_back(l);
    }
    size_t built = bank[1].size();
    for (size_t size = 2; size <= options_.max_size && built < options_.max_terms; ++size) {
        for (const Component& comp : components) {
            const size_t k = comp.params.size();
            if (k == 0 || size < k + 1) continue;
            std::vector<std::vector<size_t>> splits;
            std::vector<size_t> cur;
            compositions(size - 1, k, cur, splits);
            for (const auto& split : splits) {
                std::vector<const Term*> args(k, nullptr);
                // depth-first over argument choices
                std::vector<size_t> idx(k, 0);
                size_t pos = 0;
                while (true) {
                    if (built >= options_.max_terms) break;
                    if (pos == k) {
                        if (comp.commutative && k == 2 && args[0]->text > args[1]->text) {
                            // keep one orientation of a commutative operator
                        } else {
                            std::string text = "(" + comp.name;
                            Term t{"", comp.result, size, {}, 0, order++};
                            for (const Term* a : args) {
                                text += " " + a->text;
                                t.locals.insert(a->locals.begin(), a->locals.end());
                                t.constants += a->constants;
                            }
                            text += ")";
                            t.text = std::move(text);
                            if (seen.insert(t.text).second) {
                                bank[size].push_back(std::move(t));
                                ++built;
                            }
                        }
                        if (pos == 0) break;
                        --pos;
                        ++idx[pos];
                        continue;
                    }
                    const auto& pool = bank[split[pos]];
                    while (idx[pos] < pool.size() && !fits(rel, pool[idx[pos]].type, comp.params[pos])) ++idx[pos];
                    if (idx[pos] >= pool.size()) {
                        idx[pos] = 0;
                        if (pos == 0) break;
                        --pos;
                        ++idx[pos];
                        continue;
                    }
                    args[pos] = &pool[idx[pos]];
                    ++pos;
                    if (pos < k) idx[pos] = 0;
                }
            }
        }
    }

    // ---- candidates of the expected type, ranked ------------------------
    const bool any_type = !concrete(rel, expected);
    std::vector<const Term*> ranked;
    for (const auto& level : bank) {
        for (const Term& t : level) {
            if (any_type || fits(rel, t.type, expected)) ranked.push_back(&t);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const Term* a, const Term* b) {
        if (a->locals.size() != b->locals.size()) return a->locals.size() > b->locals.size();
        if (a->constants != b->constants) return a->constants < b->constants;
        if (a->size != b->size) return a->size < b->size;
        return a->order < b->order;
    });

    Report report;
    report.node = hole;
    report.owner = owner_;
    report.expected = expected;
    report.expected_name = env.getTypeName(expected);
    report.enumerated = built;
    size_t validated = 0;
    for (const Term* t : ranked) {
        if (report.candidates.size() > options_.max_reported || validated >= options_.max_validations) break;
        ++validated;
        std::istringstream in(t->text);
        eshkol_ast_t candidate = eshkol_parse_next_ast_from_stream(in);
        if (candidate.type == ESHKOL_INVALID) continue;
        if (checker.checkSpeculative(&candidate, expected).success) report.candidates.push_back(t->text);
    }

    // A hole reached again (named-let fixpoint passes) keeps its last report.
    for (Report& r : reports_) {
        if (r.node == hole) {
            r = std::move(report);
            return;
        }
    }
    reports_.push_back(std::move(report));
}

std::string HoleSynthesizer::describe(const Report& report) {
    std::string out;
    for (size_t i = 0; i < report.candidates.size(); ++i) {
        if (i) out += ", ";
        out += report.candidates[i];
    }
    return out;
}

size_t HoleSynthesizer::fill(const char* source_file, const char* source_text) {
    size_t unfilled = 0;
    for (Report& r : reports_) {
        const eshkol_ast_t* node = r.node;
        const char* file = node->source_file_id ? eshkol_source_file_name(node->source_file_id) : source_file;
        const unsigned line = node->line > 0 ? static_cast<unsigned>(node->line) : 0;
        const unsigned col = (node->column > 0 && node->column < 100000) ? static_cast<unsigned>(node->column) : 0;
        const std::string where = r.owner.empty() ? std::string("") : " in '" + r.owner + "'";
        if (r.candidates.empty()) {
            eshkol_error_at(file, line, col, source_text,
                            "synthesis hole ??%s: no well-typed completion of type %s up to size %zu "
                            "(%zu terms enumerated)",
                            where.c_str(), r.expected_name.c_str(), options_.max_size, r.enumerated);
            ++unfilled;
            continue;
        }
        std::istringstream in(r.candidates.front());
        eshkol_ast_t filled = eshkol_parse_next_ast_from_stream(in);
        filled.line = node->line;
        filled.column = node->column;
        filled.source_file_id = node->source_file_id;
        std::string others;
        for (size_t i = 1; i < r.candidates.size(); ++i) others += (i > 1 ? ", " : "") + r.candidates[i];
        // Candidates are written with internal (renamed) binder names; the
        // diagnostic shows the names as they are spelled in the source.
        const std::string shown = eshkol_syntax_source_text(
            "synthesis hole ??" + where + " : " + r.expected_name + " filled with " +
            r.candidates.front() + (others.empty() ? "" : "; other candidates: " + others));
        eshkol_warn_at(file, line, col, source_text, "%s", shown.c_str());
        *r.node = filled;
    }
    reports_.clear();
    return unfilled;
}

}  // namespace eshkol::hott
