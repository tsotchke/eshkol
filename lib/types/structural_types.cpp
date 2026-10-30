/**
 * @file structural_types.cpp
 * @brief Structural type terms and their solver (ADR-0046).
 *
 * See structural_types.h for the design. The solver is deliberately small:
 * every rule below is one clause of the published algorithm it names, and the
 * side conditions (occurs check, levels, skolem escape, row lacks sets,
 * predicativity, usage restriction) all live in bindMeta(), the single place
 * a metavariable is ever solved.
 */
#include "eshkol/types/structural_types.h"

#include <algorithm>
#include <climits>
#include <functional>
#include <sstream>
#include <unordered_set>

namespace eshkol::hott::st {

namespace {

/** Level for skolems that no metavariable may ever capture. */
constexpr int kRigidLevel = INT_MAX / 2;

/** Depth guard for pathological terms (each call nests at most this deep). */
constexpr int kMaxDepth = 20000;

struct DepthGuard {
    int& d;
    explicit DepthGuard(int& depth) : d(depth) { ++d; }
    ~DepthGuard() { --d; }
};

}  // namespace

// ============================================================================
// TypeStore
// ============================================================================

TypeStore::TypeStore() {
    dyn_ = make(Tag::Dyn);
    end_ = make(Tag::SEnd, Kind::Session);
    number_ = con("number");
    string_ = con("string");
    boolean_ = con("boolean");
    symbol_ = con("symbol");
    char_ = con("char");
    null_ = con("null");
}

TyRef TypeStore::make(Tag tag, Kind kind) {
    nodes_.emplace_back();
    Type& t = nodes_.back();
    t.tag = tag;
    t.kind = kind;
    return &t;
}

TyRef TypeStore::meta(Kind kind, int level, Kind field_kind) {
    TyRef t = make(Tag::Meta, kind);
    t->id = freshId();
    t->level = level;
    t->field_kind = field_kind;
    return t;
}

TyRef TypeStore::skolem(const std::string& name, Kind kind, int level, Kind field_kind,
                        const std::set<std::string>& lacks) {
    TyRef t = make(Tag::Skolem, kind);
    t->id = freshId();
    t->name = name;
    t->level = level;
    t->field_kind = field_kind;
    t->lacks = lacks;
    return t;
}

TyRef TypeStore::bound(const Binder& binder) {
    TyRef t = make(Tag::Bound, binder.kind);
    t->id = binder.id;
    t->name = binder.name;
    t->field_kind = binder.field_kind;
    return t;
}

TyRef TypeStore::con(const std::string& name, std::vector<TyRef> args) {
    TyRef t = make(Tag::Con);
    t->name = name;
    t->args = std::move(args);
    return t;
}

TyRef TypeStore::arrow(std::vector<TyRef> params, TyRef result, Mode mode, TyRef rest) {
    TyRef t = make(Tag::Arrow);
    t->args = std::move(params);
    t->a = result;
    t->b = rest;
    t->mode = mode;
    return t;
}

TyRef TypeStore::forall(std::vector<Binder> binders, TyRef body) {
    if (binders.empty()) return body;
    TyRef t = make(Tag::Forall, body ? body->kind : Kind::Type);
    t->binders = std::move(binders);
    t->a = body;
    return t;
}

TyRef TypeStore::record(TyRef row) {
    TyRef t = make(Tag::Record);
    t->a = row;
    return t;
}

TyRef TypeStore::rowEmpty(Kind field_kind) {
    TyRef t = make(Tag::RowEmpty, Kind::Row);
    t->field_kind = field_kind;
    return t;
}

TyRef TypeStore::rowExtend(const std::string& label, TyRef field, TyRef rest) {
    TyRef t = make(Tag::RowExtend, Kind::Row);
    t->name = label;
    t->a = field;
    t->b = rest;
    t->field_kind = field ? field->kind : Kind::Type;
    return t;
}

TyRef TypeStore::chan(TyRef protocol) {
    TyRef t = make(Tag::Chan);
    t->a = protocol;
    return t;
}

TyRef TypeStore::send(TyRef payload, TyRef cont) {
    TyRef t = make(Tag::SSend, Kind::Session);
    t->a = payload;
    t->b = cont;
    return t;
}

TyRef TypeStore::recv(TyRef payload, TyRef cont) {
    TyRef t = make(Tag::SRecv, Kind::Session);
    t->a = payload;
    t->b = cont;
    return t;
}

TyRef TypeStore::choose(TyRef row) {
    TyRef t = make(Tag::SChoose, Kind::Session);
    t->a = row;
    return t;
}

TyRef TypeStore::offer(TyRef row) {
    TyRef t = make(Tag::SOffer, Kind::Session);
    t->a = row;
    return t;
}

TyRef TypeStore::mu(int id, const std::string& name, TyRef body) {
    TyRef t = make(Tag::SMu, Kind::Session);
    t->id = id;
    t->name = name;
    t->a = body;
    return t;
}

TyRef TypeStore::recVar(int id, const std::string& name) {
    TyRef t = make(Tag::SRecVar, Kind::Session);
    t->id = id;
    t->name = name;
    return t;
}

TyRef TypeStore::sdual(TyRef operand) {
    TyRef t = make(Tag::SDual, operand ? operand->kind : Kind::Session);
    t->a = operand;
    t->field_kind = operand ? operand->field_kind : Kind::Session;
    return t;
}

TyRef TypeStore::values(std::vector<TyRef> elems) {
    TyRef t = make(Tag::Values);
    t->args = std::move(elems);
    return t;
}

TyRef TypeStore::prune(TyRef t) {
    if (!t) return t;
    TyRef root = t;
    while (root->tag == Tag::Meta && root->solution) root = root->solution;
    // Path compression.
    while (t->tag == Tag::Meta && t->solution && t->solution != root) {
        TyRef next = t->solution;
        t->solution = root;
        t = next;
    }
    return root;
}

namespace {

/**
 * Generic structural map: rebuilds a node only when a child changed.
 * @p leaf decides replacements for leaves (returns nullptr to keep).
 */
TyRef mapType(TypeStore& s, TyRef t, const std::function<TyRef(TyRef)>& leaf,
              int& depth) {
    if (!t) return t;
    DepthGuard guard(depth);
    if (depth > kMaxDepth) return t;
    t = TypeStore::prune(t);
    if (TyRef r = leaf(t)) return r;
    auto sub = [&](TyRef c) { return mapType(s, c, leaf, depth); };
    switch (t->tag) {
        case Tag::Con: {
            std::vector<TyRef> args;
            bool changed = false;
            for (TyRef c : t->args) {
                TyRef n = sub(c);
                changed |= (n != c);
                args.push_back(n);
            }
            return changed ? s.con(t->name, std::move(args)) : t;
        }
        case Tag::Arrow: {
            std::vector<TyRef> params;
            bool changed = false;
            for (TyRef c : t->args) {
                TyRef n = sub(c);
                changed |= (n != c);
                params.push_back(n);
            }
            TyRef res = sub(t->a);
            TyRef rest = sub(t->b);
            if (!changed && res == t->a && rest == t->b) return t;
            return s.arrow(std::move(params), res, t->mode, rest);
        }
        case Tag::Values: {
            std::vector<TyRef> elems;
            bool changed = false;
            for (TyRef c : t->args) {
                TyRef n = sub(c);
                changed |= (n != c);
                elems.push_back(n);
            }
            return changed ? s.values(std::move(elems)) : t;
        }
        case Tag::Forall: {
            TyRef body = sub(t->a);
            return body == t->a ? t : s.forall(t->binders, body);
        }
        case Tag::Record: {
            TyRef row = sub(t->a);
            return row == t->a ? t : s.record(row);
        }
        case Tag::RowExtend: {
            TyRef f = sub(t->a);
            TyRef r = sub(t->b);
            return (f == t->a && r == t->b) ? t : s.rowExtend(t->name, f, r);
        }
        case Tag::Chan: {
            TyRef p = sub(t->a);
            return p == t->a ? t : s.chan(p);
        }
        case Tag::SSend:
        case Tag::SRecv: {
            TyRef p = sub(t->a);
            TyRef c = sub(t->b);
            if (p == t->a && c == t->b) return t;
            return t->tag == Tag::SSend ? s.send(p, c) : s.recv(p, c);
        }
        case Tag::SChoose:
        case Tag::SOffer: {
            TyRef r = sub(t->a);
            if (r == t->a) return t;
            return t->tag == Tag::SChoose ? s.choose(r) : s.offer(r);
        }
        case Tag::SMu: {
            TyRef body = sub(t->a);
            return body == t->a ? t : s.mu(t->id, t->name, body);
        }
        case Tag::SDual: {
            TyRef inner = sub(t->a);
            if (inner == t->a) return t;
            return s.dual(inner);
        }
        default:
            return t;
    }
}

}  // namespace

TyRef TypeStore::zonk(TyRef t) {
    int depth = 0;
    return mapType(*this, t, [](TyRef) -> TyRef { return nullptr; }, depth);
}

TyRef TypeStore::substBound(TyRef t, const std::map<int, TyRef>& subst) {
    if (subst.empty()) return t;
    int depth = 0;
    return mapType(*this, t, [&](TyRef n) -> TyRef {
        if (n->tag == Tag::Bound) {
            auto it = subst.find(n->id);
            if (it != subst.end()) return it->second;
        }
        return nullptr;
    }, depth);
}

TyRef TypeStore::unfold(TyRef m) {
    m = prune(m);
    if (!m || m->tag != Tag::SMu) return m;
    const int id = m->id;
    int depth = 0;
    return mapType(*this, m->a, [&](TyRef n) -> TyRef {
        if (n->tag == Tag::SRecVar && n->id == id) return m;
        // An inner rec that rebinds the same id shadows it.
        if (n->tag == Tag::SMu && n->id == id) return n;
        return nullptr;
    }, depth);
}

TyRef TypeStore::dual(TyRef t) {
    t = prune(t);
    if (!t) return t;
    switch (t->tag) {
        case Tag::SEnd:
        case Tag::Dyn:
        case Tag::SRecVar:
        case Tag::RowEmpty:
            return t;
        case Tag::SSend:
            return recv(t->a, dual(t->b));
        case Tag::SRecv:
            return send(t->a, dual(t->b));
        case Tag::SChoose:
            return offer(dual(t->a));
        case Tag::SOffer:
            return choose(dual(t->a));
        case Tag::SMu:
            return mu(t->id, t->name, dual(t->a));
        case Tag::RowExtend:
            return rowExtend(t->name, dual(t->a), dual(t->b));
        case Tag::SDual:
            return prune(t->a);
        default:
            // An unsolved protocol variable, a rigid one, or a bound one:
            // its dual is not known yet and stays symbolic.
            return sdual(t);
    }
}

TyRef TypeStore::rowTail(TyRef row) {
    row = prune(row);
    while (row && row->tag == Tag::RowExtend) row = prune(row->b);
    return row;
}

std::vector<std::pair<std::string, TyRef>> TypeStore::rowFields(TyRef row) {
    std::vector<std::pair<std::string, TyRef>> out;
    row = prune(row);
    while (row && row->tag == Tag::RowExtend) {
        out.emplace_back(row->name, row->a);
        row = prune(row->b);
    }
    return out;
}

bool TypeStore::isLinear(TyRef t) {
    t = prune(t);
    if (!t) return false;
    switch (t->tag) {
        case Tag::Chan:
            return true;
        case Tag::Arrow:
            return t->mode == Mode::OneShot;
        case Tag::Con:
        case Tag::Values:
            for (TyRef c : t->args) {
                if (isLinear(c)) return true;
            }
            return false;
        case Tag::Record:
            for (const auto& f : rowFields(t->a)) {
                if (isLinear(f.second)) return true;
            }
            return false;
        default:
            return false;
    }
}

bool TypeStore::hasForall(TyRef t) {
    t = prune(t);
    if (!t) return false;
    switch (t->tag) {
        case Tag::Forall:
            return true;
        case Tag::Con:
        case Tag::Values:
            for (TyRef c : t->args) {
                if (hasForall(c)) return true;
            }
            return false;
        case Tag::Arrow:
            for (TyRef c : t->args) {
                if (hasForall(c)) return true;
            }
            return hasForall(t->a) || hasForall(t->b);
        case Tag::Record:
        case Tag::Chan:
        case Tag::SChoose:
        case Tag::SOffer:
        case Tag::SMu:
        case Tag::SDual:
            return hasForall(t->a);
        case Tag::RowExtend:
        case Tag::SSend:
        case Tag::SRecv:
            return hasForall(t->a) || hasForall(t->b);
        default:
            return false;
    }
}

void TypeStore::freeMetas(TyRef t, std::vector<TyRef>& out) {
    t = prune(t);
    if (!t) return;
    switch (t->tag) {
        case Tag::Meta:
            if (std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
            return;
        case Tag::Con:
        case Tag::Values:
            for (TyRef c : t->args) freeMetas(c, out);
            return;
        case Tag::Arrow:
            for (TyRef c : t->args) freeMetas(c, out);
            freeMetas(t->b, out);
            freeMetas(t->a, out);
            return;
        case Tag::Forall:
        case Tag::Record:
        case Tag::Chan:
        case Tag::SChoose:
        case Tag::SOffer:
        case Tag::SMu:
        case Tag::SDual:
            freeMetas(t->a, out);
            return;
        case Tag::RowExtend:
        case Tag::SSend:
        case Tag::SRecv:
            freeMetas(t->a, out);
            freeMetas(t->b, out);
            return;
        default:
            return;
    }
}

TyRef TypeStore::replaceMetas(TyRef t, const std::map<TyRef, TyRef>& subst) {
    if (subst.empty()) return t;
    int depth = 0;
    return mapType(*this, t, [&](TyRef n) -> TyRef {
        auto it = subst.find(n);
        return it == subst.end() ? nullptr : it->second;
    }, depth);
}

TyRef TypeStore::instantiate(TyRef sigma, int level) {
    sigma = prune(sigma);
    if (!sigma) return sigma;
    if (sigma->tag == Tag::Forall) {
        std::map<int, TyRef> subst;
        for (const Binder& b : sigma->binders) {
            TyRef m = meta(b.kind, level, b.field_kind);
            m->lacks = b.lacks;
            m->unrestricted = b.unrestricted;
            m->name = b.name + std::to_string(m->id);
            subst[b.id] = m;
        }
        return instantiate(substBound(sigma->a, subst), level);
    }
    if (sigma->tag == Tag::Arrow) {
        TyRef res = prune(sigma->a);
        if (res && (res->tag == Tag::Forall || res->tag == Tag::Arrow)) {
            TyRef inst = instantiate(res, level);
            if (inst != res) return arrow(sigma->args, inst, sigma->mode, sigma->b);
        }
    }
    return sigma;
}

TyRef TypeStore::skolemise(TyRef sigma, int level, std::vector<TyRef>& skolems) {
    sigma = prune(sigma);
    if (!sigma) return sigma;
    if (sigma->tag == Tag::Forall) {
        std::map<int, TyRef> subst;
        for (const Binder& b : sigma->binders) {
            TyRef k = skolem(b.name, b.kind, level, b.field_kind, b.lacks);
            skolems.push_back(k);
            subst[b.id] = k;
        }
        return skolemise(substBound(sigma->a, subst), level, skolems);
    }
    if (sigma->tag == Tag::Arrow) {
        TyRef res = prune(sigma->a);
        if (res && (res->tag == Tag::Forall || res->tag == Tag::Arrow)) {
            TyRef sk = skolemise(res, level, skolems);
            if (sk != res) return arrow(sigma->args, sk, sigma->mode, sigma->b);
        }
    }
    return sigma;
}

TyRef TypeStore::generalize(TyRef t, int level) {
    t = zonk(t);
    std::vector<TyRef> metas;
    freeMetas(t, metas);
    std::vector<Binder> binders;
    std::map<TyRef, TyRef> subst;
    int n_type = 0, n_row = 0, n_sess = 0;
    for (TyRef m : metas) {
        if (m->level <= level) continue;
        Binder b;
        b.id = freshId();
        b.kind = m->kind;
        b.field_kind = m->field_kind;
        b.lacks = m->lacks;
        b.unrestricted = (m->kind == Kind::Type);
        if (m->kind == Kind::Type) {
            static const char* names = "abcdefghijklmnopq";
            b.name = n_type < 17 ? std::string(1, names[n_type]) : "t" + std::to_string(n_type);
            ++n_type;
        } else if (m->kind == Kind::Row) {
            b.name = n_row == 0 ? "r" : "r" + std::to_string(n_row);
            ++n_row;
        } else {
            b.name = n_sess == 0 ? "p" : "p" + std::to_string(n_sess);
            ++n_sess;
        }
        subst[m] = bound(b);
        binders.push_back(b);
    }
    if (binders.empty()) return t;
    return forall(std::move(binders), replaceMetas(t, subst));
}

std::string TypeStore::show(TyRef t) {
    t = prune(t);
    if (!t) return "?";
    std::ostringstream o;
    switch (t->tag) {
        case Tag::Meta:
            o << "?" << (t->name.empty() ? std::to_string(t->id) : t->name);
            break;
        case Tag::Skolem:
        case Tag::Bound:
        case Tag::SRecVar:
            o << t->name;
            break;
        case Tag::Dyn:
            o << "any";
            break;
        case Tag::Con:
            if (t->args.empty()) {
                o << t->name;
            } else {
                o << "(" << t->name;
                for (TyRef c : t->args) o << " " << show(c);
                o << ")";
            }
            break;
        case Tag::Arrow:
            o << (t->mode == Mode::OneShot ? "(-o" : "(->");
            for (TyRef c : t->args) o << " " << show(c);
            if (t->b) o << " (... " << show(t->b) << ")";
            o << " " << show(t->a) << ")";
            break;
        case Tag::Forall:
            o << "(forall (";
            for (size_t i = 0; i < t->binders.size(); ++i) {
                if (i) o << " ";
                o << t->binders[i].name;
            }
            o << ") " << show(t->a) << ")";
            break;
        case Tag::Record: {
            o << "(record";
            for (const auto& f : rowFields(t->a)) o << " (" << f.first << " " << show(f.second) << ")";
            TyRef tail = rowTail(t->a);
            if (tail && tail->tag != Tag::RowEmpty) o << " " << show(tail);
            o << ")";
            break;
        }
        case Tag::RowEmpty:
            o << "(row)";
            break;
        case Tag::RowExtend: {
            o << "(row";
            for (const auto& f : rowFields(t)) o << " (" << f.first << " " << show(f.second) << ")";
            TyRef tail = rowTail(t);
            if (tail && tail->tag != Tag::RowEmpty) o << " " << show(tail);
            o << ")";
            break;
        }
        case Tag::Chan:
            o << "(chan " << show(t->a) << ")";
            break;
        case Tag::SEnd:
            o << "end";
            break;
        case Tag::SSend:
            o << "(! " << show(t->a) << " " << show(t->b) << ")";
            break;
        case Tag::SRecv:
            o << "(? " << show(t->a) << " " << show(t->b) << ")";
            break;
        case Tag::SChoose:
        case Tag::SOffer: {
            o << (t->tag == Tag::SChoose ? "(choose" : "(offer");
            for (const auto& f : rowFields(t->a)) o << " (" << f.first << " " << show(f.second) << ")";
            TyRef tail = rowTail(t->a);
            if (tail && tail->tag != Tag::RowEmpty) o << " " << show(tail);
            o << ")";
            break;
        }
        case Tag::SMu:
            o << "(rec " << t->name << " " << show(t->a) << ")";
            break;
        case Tag::SDual:
            o << "(dual " << show(t->a) << ")";
            break;
        case Tag::Values:
            o << "(values";
            for (TyRef c : t->args) o << " " << show(c);
            o << ")";
            break;
    }
    return o.str();
}

// ============================================================================
// Solver
// ============================================================================

bool Solver::fail(const std::string& msg) {
    if (error_.empty()) error_ = msg;
    return false;
}

bool Solver::unify(TyRef a, TyRef b) {
    if (depth_ == 0) assumed_.clear();
    DepthGuard guard(depth_);
    if (depth_ > kMaxDepth) return fail("type is too deeply nested to compare");

    a = TypeStore::prune(a);
    b = TypeStore::prune(b);
    if (!a || !b) return fail("missing type");
    if (a == b) return true;

    if (a->tag == Tag::Meta) return bindMeta(a, b);
    if (b->tag == Tag::Meta) return bindMeta(b, a);

    if (a->tag == Tag::Dyn || b->tag == Tag::Dyn) {
        if (dyn_meets) dyn_meets->push_back(a->tag == Tag::Dyn ? b : a);
        return true;
    }

    if (a->tag == Tag::SDual) return unifyDual(a, b);
    if (b->tag == Tag::SDual) return unifyDual(b, a);

    if (a->tag == Tag::SMu || b->tag == Tag::SMu) {
        // Equi-recursive protocols: assume the pair while comparing one
        // unfolding (Amadio and Cardelli; Gay and Hole).
        auto key = std::make_pair(a, b);
        if (assumed_.count(key)) return true;
        assumed_.insert(key);
        return unify(a->tag == Tag::SMu ? s_.unfold(a) : a,
                     b->tag == Tag::SMu ? s_.unfold(b) : b);
    }

    if (a->kind == Kind::Row || b->kind == Kind::Row) {
        if (a->kind != b->kind) {
            return fail("cannot match the row `" + s_.show(a) + "` with `" + s_.show(b) + "`");
        }
        return unifyRows(a, b);
    }

    if (a->tag != b->tag) {
        return fail("cannot match `" + s_.show(a) + "` with `" + s_.show(b) + "`");
    }

    switch (a->tag) {
        case Tag::Con:
            if (a->name != b->name || a->args.size() != b->args.size()) {
                return fail("cannot match `" + s_.show(a) + "` with `" + s_.show(b) + "`");
            }
            for (size_t i = 0; i < a->args.size(); ++i) {
                if (!unify(a->args[i], b->args[i])) return false;
            }
            return true;

        case Tag::Arrow:
            if (a->args.size() != b->args.size() || (a->b == nullptr) != (b->b == nullptr)) {
                return fail("function `" + s_.show(a) + "` and function `" + s_.show(b) +
                            "` take different numbers of arguments");
            }
            if (a->mode != b->mode) {
                return fail("cannot match the one-shot function type with the unrestricted one in `" +
                            s_.show(a) + "` and `" + s_.show(b) + "`");
            }
            for (size_t i = 0; i < a->args.size(); ++i) {
                if (!unify(a->args[i], b->args[i])) return false;
            }
            if (a->b && !unify(a->b, b->b)) return false;
            return unify(a->a, b->a);

        case Tag::Forall: {
            // Two polytypes are equal when their bodies are equal under the
            // same rigid variables (alpha-equivalence; predicative).
            if (a->binders.size() != b->binders.size()) {
                return fail("polymorphic types `" + s_.show(a) + "` and `" + s_.show(b) +
                            "` quantify different numbers of variables");
            }
            std::map<int, TyRef> sa, sb;
            for (size_t i = 0; i < a->binders.size(); ++i) {
                const Binder& ba = a->binders[i];
                const Binder& bb = b->binders[i];
                if (ba.kind != bb.kind) return fail("quantified variables of different kinds");
                std::set<std::string> lacks = ba.lacks;
                lacks.insert(bb.lacks.begin(), bb.lacks.end());
                TyRef sk = s_.skolem(ba.name, ba.kind, kRigidLevel, ba.field_kind, lacks);
                sa[ba.id] = sk;
                sb[bb.id] = sk;
            }
            return unify(s_.substBound(a->a, sa), s_.substBound(b->a, sb));
        }

        case Tag::Record:
        case Tag::Chan:
        case Tag::SChoose:
        case Tag::SOffer:
            return unify(a->a, b->a);

        case Tag::SEnd:
            return true;

        case Tag::SSend:
        case Tag::SRecv:
            return unify(a->a, b->a) && unify(a->b, b->b);

        case Tag::SRecVar:
        case Tag::Bound:
            if (a->id == b->id) return true;
            return fail("cannot match `" + s_.show(a) + "` with `" + s_.show(b) + "`");

        case Tag::Values:
            if (a->args.size() != b->args.size()) {
                return fail("`" + s_.show(a) + "` and `" + s_.show(b) +
                            "` produce different numbers of values");
            }
            for (size_t i = 0; i < a->args.size(); ++i) {
                if (!unify(a->args[i], b->args[i])) return false;
            }
            return true;

        case Tag::Skolem:
            return fail("the rigid type variable `" + s_.show(a) + "` cannot be `" +
                        s_.show(b) + "`");

        default:
            return fail("cannot match `" + s_.show(a) + "` with `" + s_.show(b) + "`");
    }
}

bool Solver::unifyDual(TyRef d, TyRef other) {
    TyRef x = TypeStore::prune(d->a);
    other = TypeStore::prune(other);
    if (x->tag != Tag::Meta && x->tag != Tag::Skolem && x->tag != Tag::Bound) {
        return unify(s_.dual(x), other);
    }
    if (other->tag == Tag::SDual) return unify(x, other->a);
    return unify(x, s_.dual(other));
}

bool Solver::unifyRows(TyRef a, TyRef b) {
    a = TypeStore::prune(a);
    b = TypeStore::prune(b);
    if (a == b) return true;
    if (a->tag == Tag::Meta) return bindMeta(a, b);
    if (b->tag == Tag::Meta) return bindMeta(b, a);
    if (a->tag == Tag::Dyn || b->tag == Tag::Dyn) return true;
    if (a->tag == Tag::SDual) return unifyDual(a, b);
    if (b->tag == Tag::SDual) return unifyDual(b, a);
    if (a->tag == Tag::RowEmpty && b->tag == Tag::RowEmpty) return true;
    if (a->tag != Tag::RowExtend && b->tag == Tag::RowExtend) std::swap(a, b);
    if (a->tag == Tag::RowExtend) {
        TyRef tail_before = TypeStore::rowTail(a->b);
        TyRef field = nullptr, rest = nullptr;
        if (!rewriteRow(b, a->name, field, rest)) return false;
        // Leijen's side condition: rewriting must not have solved a's own
        // tail, or the two rows describe an infinite record.
        if (tail_before && tail_before->tag == Tag::Meta && tail_before->solution) {
            return fail("the rows `" + s_.show(a) + "` and `" + s_.show(b) +
                        "` would describe an infinite row");
        }
        return unify(a->a, field) && unify(a->b, rest);
    }
    if (a->tag == Tag::Skolem || b->tag == Tag::Skolem) {
        return fail("the rigid row `" + s_.show(a->tag == Tag::Skolem ? a : b) +
                    "` cannot be `" + s_.show(a->tag == Tag::Skolem ? b : a) + "`");
    }
    return fail("cannot match the row `" + s_.show(a) + "` with `" + s_.show(b) + "`");
}

bool Solver::rewriteRow(TyRef row, const std::string& label, TyRef& field, TyRef& rest) {
    row = TypeStore::prune(row);
    if (!row) return fail("missing row");
    switch (row->tag) {
        case Tag::RowExtend: {
            if (row->name == label) {
                field = row->a;
                rest = row->b;
                return true;
            }
            TyRef f = nullptr, r = nullptr;
            if (!rewriteRow(row->b, label, f, r)) return false;
            field = f;
            rest = s_.rowExtend(row->name, row->a, r);
            return true;
        }
        case Tag::Meta: {
            if (row->lacks.count(label)) {
                return fail("the label `" + label + "` is already present or must be absent here");
            }
            TyRef f = s_.meta(row->field_kind, row->level);
            f->unrestricted = row->unrestricted && row->field_kind == Kind::Type;
            TyRef r = s_.meta(Kind::Row, row->level, row->field_kind);
            r->lacks = row->lacks;
            r->lacks.insert(label);
            r->unrestricted = row->unrestricted;
            row->solution = s_.rowExtend(label, f, r);
            field = f;
            rest = r;
            return true;
        }
        case Tag::Dyn:
            field = s_.dyn();
            rest = s_.dyn();
            return true;
        case Tag::RowEmpty:
            return fail("no field or label `" + label + "`");
        case Tag::Skolem:
            return fail("the rigid row `" + row->name + "` has no label `" + label + "`");
        case Tag::SDual: {
            TyRef x = TypeStore::prune(row->a);
            if (x->tag == Tag::Meta) {
                // dual(x) must contain `label`: give x that label, then dualise.
                TyRef f = nullptr, r = nullptr;
                if (!rewriteRow(x, label, f, r)) return false;
                field = s_.dual(f);
                rest = s_.dual(r);
                return true;
            }
            return rewriteRow(s_.dual(x), label, field, rest);
        }
        default:
            return fail("`" + s_.show(row) + "` is not a row");
    }
}

bool Solver::rowInclusion(TyRef sub, TyRef sup) {
    sub = TypeStore::prune(sub);
    TyRef remaining = sup;
    while (sub && sub->tag == Tag::RowExtend) {
        TyRef field = nullptr, rest = nullptr;
        if (!rewriteRow(remaining, sub->name, field, rest)) return false;
        if (!unify(sub->a, field)) return false;
        remaining = rest;
        sub = TypeStore::prune(sub->b);
    }
    if (!sub || sub->tag == Tag::RowEmpty || sub->tag == Tag::Dyn) return true;
    return unify(sub, remaining);
}

bool Solver::occursAndAdjust(TyRef m, TyRef t) {
    t = TypeStore::prune(t);
    if (!t) return true;
    if (t == m) {
        return fail("a type variable would have to contain itself (infinite type)");
    }
    switch (t->tag) {
        case Tag::Meta:
            t->level = std::min(t->level, m->level);
            if (m->unrestricted && t->kind == Kind::Type) t->unrestricted = true;
            return true;
        case Tag::Skolem:
            if (t->level > m->level) {
                return fail("the type variable `" + t->name + "` would escape its scope");
            }
            return true;
        case Tag::Con:
        case Tag::Values:
            for (TyRef c : t->args) {
                if (!occursAndAdjust(m, c)) return false;
            }
            return true;
        case Tag::Arrow: {
            // A function value's own usage is its mode; its parameters may be
            // linear without making the function linear, so the restriction
            // flag is not pushed into them.
            const bool saved = m->unrestricted;
            m->unrestricted = false;
            bool ok = true;
            for (TyRef c : t->args) ok = ok && occursAndAdjust(m, c);
            ok = ok && occursAndAdjust(m, t->b) && occursAndAdjust(m, t->a);
            m->unrestricted = saved;
            return ok;
        }
        case Tag::Forall:
        case Tag::Record:
        case Tag::SChoose:
        case Tag::SOffer:
        case Tag::SMu:
        case Tag::SDual:
            return occursAndAdjust(m, t->a);
        case Tag::Chan: {
            const bool saved = m->unrestricted;
            m->unrestricted = false;
            bool ok = occursAndAdjust(m, t->a);
            m->unrestricted = saved;
            return ok;
        }
        case Tag::RowExtend:
        case Tag::SSend:
        case Tag::SRecv:
            return occursAndAdjust(m, t->a) && occursAndAdjust(m, t->b);
        default:
            return true;
    }
}

bool Solver::bindMeta(TyRef m, TyRef t) {
    t = TypeStore::prune(t);
    if (t == m) return true;

    if (t->tag == Tag::Meta) {
        if (t->kind != m->kind) {
            return fail("a type variable of one kind cannot stand for one of another kind");
        }
        if (m->kind == Kind::Row && t->field_kind != m->field_kind) {
            return fail("rows of different field kinds cannot be matched");
        }
        t->level = std::min(t->level, m->level);
        t->lacks.insert(m->lacks.begin(), m->lacks.end());
        t->unrestricted = t->unrestricted || m->unrestricted;
        m->solution = t;
        return true;
    }

    if (t->tag == Tag::Dyn) {
        // Consistency, not equality: the variable stays open, and the meeting
        // is recorded so a linear solution found later is still rejected.
        if (dyn_meets) dyn_meets->push_back(m);
        return true;
    }

    if (t->kind != m->kind) {
        auto kind_name = [](Kind k) {
            return k == Kind::Type ? "a value type" : k == Kind::Row ? "a row" : "a protocol";
        };
        return fail(std::string("`") + s_.show(t) + "` is " + kind_name(t->kind) +
                    ", but " + kind_name(m->kind) + " is expected here");
    }

    if (TypeStore::hasForall(t)) {
        return fail("a type variable cannot stand for the polymorphic type `" + s_.show(t) +
                    "` (instantiation is predicative; annotate the binding with its "
                    "polymorphic type)");
    }

    if (!occursAndAdjust(m, t)) return false;

    if (m->kind == Kind::Row) {
        for (const auto& f : TypeStore::rowFields(t)) {
            if (m->lacks.count(f.first)) {
                return fail("the label `" + f.first + "` would occur twice in one row");
            }
        }
        TyRef tail = TypeStore::rowTail(t);
        if (tail && tail->tag == Tag::Meta) {
            tail->lacks.insert(m->lacks.begin(), m->lacks.end());
            if (m->unrestricted) tail->unrestricted = true;
        } else if (tail && tail->tag == Tag::Skolem) {
            for (const auto& l : m->lacks) {
                if (!tail->lacks.count(l)) {
                    return fail("the rigid row `" + tail->name + "` may contain the label `" +
                                l + "`, which must be absent here");
                }
            }
        }
    }

    if (m->unrestricted && TypeStore::isLinear(t)) {
        return fail("the linear type `" + s_.show(t) +
                    "` cannot instantiate a type variable whose values may be copied or "
                    "dropped");
    }

    m->solution = t;
    return true;
}

}  // namespace eshkol::hott::st
