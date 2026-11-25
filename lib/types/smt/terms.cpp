/**
 * @file terms.cpp
 * @brief Hash-consed terms, SMT-LIB2 rendering and exact model evaluation
 *        for the refinement decision procedure (ADR-0044).
 */
#include "eshkol/types/smt.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <variant>

namespace eshkol::smt {

const char* sortName(Sort s) {
    switch (s) {
        case Sort::Bool: return "Bool";
        case Sort::Int: return "Int";
        case Sort::Real: return "Real";
        case Sort::Obj: return "Obj";
    }
    return "?";
}

const char* resultName(Result r) {
    switch (r) {
        case Result::Sat: return "sat";
        case Result::Unsat: return "unsat";
        case Result::Unknown: return "unknown";
    }
    return "?";
}

// Interpreted symbols that the search treats as uninterpreted (an
// over-approximation, so Unsat stays sound) and that model validation
// evaluates exactly. Scheme semantics: quotient truncates, modulo takes the
// divisor's sign, remainder the dividend's.
bool isOverApproximatedSymbol(const std::string& fn) {
    return fn == "*" || fn == "quotient" || fn == "remainder" || fn == "modulo" ||
           fn == "/";
}

static bool isNumeric(Sort s) { return s == Sort::Int || s == Sort::Real; }

static Node makeNode(Kind k, Sort s) {
    Node n;
    n.kind = k;
    n.sort = s;
    return n;
}

static Sort joinNumeric(Sort a, Sort b) {
    return (a == Sort::Int && b == Sort::Int) ? Sort::Int : Sort::Real;
}

// ---------------------------------------------------------------------------
// TermManager
// ---------------------------------------------------------------------------

TermManager::TermManager() = default;
TermManager::~TermManager() = default;

Term TermManager::intern(Node&& n) {
    std::ostringstream key;
    key << static_cast<int>(n.kind) << ':' << static_cast<int>(n.sort) << ':' << n.bval << ':'
        << n.value.str() << ':' << n.name << ':';
    for (Term a : n.args) key << a->id << ',';
    std::string k = key.str();
    auto it = table_.find(k);
    if (it != table_.end()) return it->second;
    auto owned = std::make_unique<Node>(std::move(n));
    owned->id = static_cast<uint32_t>(nodes_.size() + 1);
    Term t = owned.get();
    nodes_.push_back(std::move(owned));
    table_.emplace(std::move(k), t);
    return t;
}

Term TermManager::mkTrue() {
    Node n = makeNode(Kind::BoolConst, Sort::Bool);
    n.bval = true;
    return intern(std::move(n));
}

Term TermManager::mkFalse() {
    Node n = makeNode(Kind::BoolConst, Sort::Bool);
    n.bval = false;
    return intern(std::move(n));
}

Term TermManager::mkNumeral(const Rational& v, Sort s) {
    Node n = makeNode(Kind::Numeral, (s == Sort::Int && v.isInteger()) ? Sort::Int : Sort::Real);
    n.value = v;
    return intern(std::move(n));
}

Term TermManager::mkInt(const Integer& v) { return mkNumeral(Rational(v), Sort::Int); }
Term TermManager::mkReal(const Rational& v) { return mkNumeral(v, Sort::Real); }

Term TermManager::mkVar(const std::string& name, Sort s) {
    std::string actual = name;
    auto it = var_sorts_.find(name);
    if (it != var_sorts_.end() && it->second != s) {
        actual = name + "!" + sortName(s);
    }
    Node n = makeNode(Kind::Var, s);
    n.name = actual;
    size_t before = nodes_.size();
    Term t = intern(std::move(n));
    if (nodes_.size() != before) {
        vars_.push_back(t);
        var_sorts_.emplace(actual, s);
    }
    return t;
}

std::string TermManager::freshName(const std::string& stem) {
    for (;;) {
        std::string candidate = "%" + stem + "!" + std::to_string(++fresh_counter_);
        if (!var_sorts_.count(candidate)) return candidate;
    }
}

Term TermManager::mkApp(const std::string& fn, Sort s, const std::vector<Term>& args) {
    std::string actual = fn;
    std::vector<Sort> arg_sorts;
    for (Term a : args) arg_sorts.push_back(a->sort);
    auto it = signatures_.find(fn);
    if (it == signatures_.end()) {
        signatures_.emplace(fn, Signature{arg_sorts, s});
    } else if (it->second.args != arg_sorts || it->second.result != s) {
        // The same symbol at another signature is another function.
        actual = fn + "!";
        for (Sort as : arg_sorts) actual += sortName(as);
        actual += std::string("!") + sortName(s);
        signatures_.emplace(actual, Signature{arg_sorts, s});
    }
    Node n = makeNode(Kind::App, s);
    n.name = actual;
    n.args = args;
    return intern(std::move(n));
}

Term TermManager::mkScale(const Rational& c, Term t) {
    Sort s = (t->sort == Sort::Int && c.isInteger()) ? Sort::Int : Sort::Real;
    if (c.isZero()) return mkNumeral(Rational(0), s);
    if (c == Rational(1)) return t;
    if (t->kind == Kind::Numeral) return mkNumeral(c * t->value, s);
    if (t->kind == Kind::Scale) return mkScale(c * t->value, t->args[0]);
    if (t->kind == Kind::Add) {
        std::vector<Term> parts;
        for (Term a : t->args) parts.push_back(mkScale(c, a));
        return mkAdd(parts);
    }
    Node n = makeNode(Kind::Scale, s);
    n.value = c;
    n.args = {t};
    return intern(std::move(n));
}

Term TermManager::mkAdd(const std::vector<Term>& args) {
    std::vector<Term> flat;
    Rational constant(0);
    Sort s = Sort::Int;
    std::vector<Term> work(args.rbegin(), args.rend());
    while (!work.empty()) {
        Term a = work.back();
        work.pop_back();
        s = joinNumeric(s, a->sort);
        if (a->kind == Kind::Add) {
            for (auto it = a->args.rbegin(); it != a->args.rend(); ++it) work.push_back(*it);
        } else if (a->kind == Kind::Numeral) {
            constant += a->value;
        } else {
            flat.push_back(a);
        }
    }
    if (!constant.isZero()) flat.push_back(mkNumeral(constant, s));
    if (flat.empty()) return mkNumeral(Rational(0), s);
    if (flat.size() == 1) {
        if (flat[0]->kind == Kind::Numeral) return mkNumeral(flat[0]->value, s);
        return flat[0];
    }
    Node n = makeNode(Kind::Add, s);
    n.args = std::move(flat);
    return intern(std::move(n));
}

Term TermManager::mkNeg(Term a) { return mkScale(Rational(-1), a); }

Term TermManager::mkSub(Term a, Term b) { return mkAdd(a, mkNeg(b)); }

Term TermManager::mkMul(Term a, Term b) {
    if (a->kind == Kind::Numeral) return mkScale(a->value, b);
    if (b->kind == Kind::Numeral) return mkScale(b->value, a);
    if (b->id < a->id) std::swap(a, b);
    return mkApp("*", joinNumeric(a->sort, b->sort), {a, b});
}

Term TermManager::mkIte(Term c, Term t, Term e) {
    if (c->kind == Kind::BoolConst) return c->bval ? t : e;
    if (t == e) return t;
    Sort s = t->sort;
    if (isNumeric(t->sort) && isNumeric(e->sort)) s = joinNumeric(t->sort, e->sort);
    if (s == Sort::Bool) return mkOr(mkAnd(c, t), mkAnd(mkNot(c), e));
    Node n = makeNode(Kind::Ite, s);
    n.args = {c, t, e};
    return intern(std::move(n));
}

Term TermManager::mkEq(Term a, Term b) {
    if (a == b) return mkTrue();
    if (a->kind == Kind::Numeral && b->kind == Kind::Numeral) return mkBool(a->value == b->value);
    if (a->kind == Kind::BoolConst && b->kind == Kind::BoolConst) return mkBool(a->bval == b->bval);
    if (a->sort == Sort::Bool) {
        if (a->kind == Kind::BoolConst) return a->bval ? b : mkNot(b);
        if (b->kind == Kind::BoolConst) return b->bval ? a : mkNot(a);
    }
    if (b->id < a->id) std::swap(a, b);
    Node n = makeNode(Kind::Eq, Sort::Bool);
    n.args = {a, b};
    return intern(std::move(n));
}

Term TermManager::mkLe(Term a, Term b) {
    if (a == b) return mkTrue();
    if (a->kind == Kind::Numeral && b->kind == Kind::Numeral) return mkBool(a->value <= b->value);
    Node n = makeNode(Kind::Le, Sort::Bool);
    n.args = {a, b};
    return intern(std::move(n));
}

Term TermManager::mkLt(Term a, Term b) {
    if (a == b) return mkFalse();
    if (a->kind == Kind::Numeral && b->kind == Kind::Numeral) return mkBool(a->value < b->value);
    Node n = makeNode(Kind::Lt, Sort::Bool);
    n.args = {a, b};
    return intern(std::move(n));
}

Term TermManager::mkNot(Term a) {
    if (a->kind == Kind::BoolConst) return mkBool(!a->bval);
    if (a->kind == Kind::Not) return a->args[0];
    Node n = makeNode(Kind::Not, Sort::Bool);
    n.args = {a};
    return intern(std::move(n));
}

Term TermManager::mkAnd(const std::vector<Term>& args) {
    std::vector<Term> flat;
    std::set<uint32_t> seen;
    std::vector<Term> work(args.rbegin(), args.rend());
    while (!work.empty()) {
        Term a = work.back();
        work.pop_back();
        if (a->kind == Kind::BoolConst) {
            if (!a->bval) return mkFalse();
            continue;
        }
        if (a->kind == Kind::And) {
            for (auto it = a->args.rbegin(); it != a->args.rend(); ++it) work.push_back(*it);
            continue;
        }
        if (seen.insert(a->id).second) flat.push_back(a);
    }
    for (Term a : flat) {
        if (a->kind == Kind::Not && seen.count(a->args[0]->id)) return mkFalse();
    }
    if (flat.empty()) return mkTrue();
    if (flat.size() == 1) return flat[0];
    Node n = makeNode(Kind::And, Sort::Bool);
    n.args = std::move(flat);
    return intern(std::move(n));
}

Term TermManager::mkOr(const std::vector<Term>& args) {
    std::vector<Term> flat;
    std::set<uint32_t> seen;
    std::vector<Term> work(args.rbegin(), args.rend());
    while (!work.empty()) {
        Term a = work.back();
        work.pop_back();
        if (a->kind == Kind::BoolConst) {
            if (a->bval) return mkTrue();
            continue;
        }
        if (a->kind == Kind::Or) {
            for (auto it = a->args.rbegin(); it != a->args.rend(); ++it) work.push_back(*it);
            continue;
        }
        if (seen.insert(a->id).second) flat.push_back(a);
    }
    for (Term a : flat) {
        if (a->kind == Kind::Not && seen.count(a->args[0]->id)) return mkTrue();
    }
    if (flat.empty()) return mkFalse();
    if (flat.size() == 1) return flat[0];
    Node n = makeNode(Kind::Or, Sort::Bool);
    n.args = std::move(flat);
    return intern(std::move(n));
}

Term TermManager::mkImplies(Term a, Term b) {
    if (a->kind == Kind::BoolConst) return a->bval ? b : mkTrue();
    if (b->kind == Kind::BoolConst && b->bval) return mkTrue();
    if (a == b) return mkTrue();
    Node n = makeNode(Kind::Implies, Sort::Bool);
    n.args = {a, b};
    return intern(std::move(n));
}

// ---------------------------------------------------------------------------
// SMT-LIB2 rendering
// ---------------------------------------------------------------------------

static std::string smtSymbol(const std::string& name) {
    static const std::string extra = "~!@$%^&*_-+=<>.?/";
    bool simple = !name.empty() && !(name[0] >= '0' && name[0] <= '9');
    for (char c : name) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  extra.find(c) != std::string::npos;
        if (!ok) simple = false;
    }
    if (simple) return name;
    std::string q = "|";
    for (char c : name) q += (c == '|' || c == '\\') ? '_' : c;
    return q + "|";
}

static std::string smtNumeral(const Rational& v, Sort s) {
    auto lit = [&](const Integer& i) {
        std::string t = i.abs().str();
        return s == Sort::Int ? t : t + ".0";
    };
    std::string body;
    if (v.isInteger()) {
        body = lit(v.num());
    } else {
        body = "(/ " + v.num().abs().str() + ".0 " + v.den().str() + ".0)";
    }
    return v.sign() < 0 ? "(- " + body + ")" : body;
}

static std::string render(Term t, Sort want);

static std::string renderAs(Term t, Sort want) {
    if (want == Sort::Real && t->sort == Sort::Int) {
        if (t->kind == Kind::Numeral) return smtNumeral(t->value, Sort::Real);
        return "(to_real " + render(t, Sort::Int) + ")";
    }
    return render(t, t->sort);
}

static std::string render(Term t, Sort /*want*/) {
    std::ostringstream o;
    switch (t->kind) {
        case Kind::BoolConst:
            return t->bval ? "true" : "false";
        case Kind::Numeral:
            return smtNumeral(t->value, t->sort);
        case Kind::Var:
            return smtSymbol(t->name);
        case Kind::App: {
            if (t->args.empty()) return smtSymbol(t->name);
            o << "(" << smtSymbol(t->name);
            for (Term a : t->args) {
                Sort as = a->sort;
                if (t->name == "*" || t->name == "/") as = t->sort;
                o << " " << renderAs(a, as);
            }
            o << ")";
            return o.str();
        }
        case Kind::Add:
            o << "(+";
            for (Term a : t->args) o << " " << renderAs(a, t->sort);
            o << ")";
            return o.str();
        case Kind::Scale:
            o << "(* " << smtNumeral(t->value, t->sort) << " " << renderAs(t->args[0], t->sort)
              << ")";
            return o.str();
        case Kind::Ite:
            o << "(ite " << render(t->args[0], Sort::Bool) << " " << renderAs(t->args[1], t->sort)
              << " " << renderAs(t->args[2], t->sort) << ")";
            return o.str();
        case Kind::Eq:
        case Kind::Le:
        case Kind::Lt: {
            const char* op = t->kind == Kind::Eq ? "=" : (t->kind == Kind::Le ? "<=" : "<");
            Sort s = t->args[0]->sort;
            if (isNumeric(s) && isNumeric(t->args[1]->sort)) s = joinNumeric(s, t->args[1]->sort);
            o << "(" << op << " " << renderAs(t->args[0], s) << " " << renderAs(t->args[1], s)
              << ")";
            return o.str();
        }
        case Kind::Not:
            return "(not " + render(t->args[0], Sort::Bool) + ")";
        case Kind::And:
        case Kind::Or:
            o << (t->kind == Kind::And ? "(and" : "(or");
            for (Term a : t->args) o << " " << render(a, Sort::Bool);
            o << ")";
            return o.str();
        case Kind::Implies:
            return "(=> " + render(t->args[0], Sort::Bool) + " " + render(t->args[1], Sort::Bool) +
                   ")";
    }
    return "?";
}

std::string toSmtLib2(Term t) { return render(t, t->sort); }

// Exact definitions of the over-approximated Scheme symbols, so an external
// solver reads the same semantics model validation uses.
static const char* kSchemeArithmeticPrelude =
    "(define-fun quotient ((a Int) (b Int)) Int\n"
    "  (ite (= (< a 0) (< b 0)) (div (abs a) (abs b)) (- (div (abs a) (abs b)))))\n"
    "(define-fun remainder ((a Int) (b Int)) Int (- a (* b (quotient a b))))\n"
    "(define-fun modulo ((a Int) (b Int)) Int\n"
    "  (ite (> b 0) (mod a b) (- (mod (- a) (- b)))))\n";

static void collect(Term t, std::set<Term>& seen, std::vector<Term>& vars,
                    std::set<std::string>& fns) {
    if (!seen.insert(t).second) return;
    if (t->kind == Kind::Var) vars.push_back(t);
    if (t->kind == Kind::App) fns.insert(t->name);
    for (Term a : t->args) collect(a, seen, vars, fns);
}

std::string toSmtLib2Script(const TermManager& tm, const std::vector<Term>& assertions,
                            bool with_check_sat) {
    std::set<Term> seen;
    std::vector<Term> vars;
    std::set<std::string> fns;
    for (Term a : assertions) collect(a, seen, vars, fns);
    std::sort(vars.begin(), vars.end(), [](Term a, Term b) { return a->id < b->id; });

    std::ostringstream o;
    o << "(set-logic ALL)\n";
    bool uses_obj = false;
    for (Term v : vars) uses_obj |= v->sort == Sort::Obj;
    for (const auto& f : fns) {
        auto it = tm.signatures().find(f);
        if (it == tm.signatures().end()) continue;
        uses_obj |= it->second.result == Sort::Obj;
        for (Sort s : it->second.args) uses_obj |= s == Sort::Obj;
    }
    if (uses_obj) o << "(declare-sort Obj 0)\n";
    bool scheme_arith = fns.count("quotient") || fns.count("remainder") || fns.count("modulo");
    if (scheme_arith) o << kSchemeArithmeticPrelude;
    for (Term v : vars) {
        o << "(declare-fun " << smtSymbol(v->name) << " () " << sortName(v->sort) << ")\n";
    }
    for (const auto& f : fns) {
        if (f == "*" || f == "/" || f == "quotient" || f == "remainder" || f == "modulo") continue;
        auto it = tm.signatures().find(f);
        if (it == tm.signatures().end()) continue;
        o << "(declare-fun " << smtSymbol(f) << " (";
        for (size_t i = 0; i < it->second.args.size(); ++i) {
            o << (i ? " " : "") << sortName(it->second.args[i]);
        }
        o << ") " << sortName(it->second.result) << ")\n";
    }
    for (Term a : assertions) o << "(assert " << toSmtLib2(a) << ")\n";
    if (with_check_sat) o << "(check-sat)\n";
    return o.str();
}

// ---------------------------------------------------------------------------
// Exact evaluation
// ---------------------------------------------------------------------------

namespace {

struct Value {
    enum class Tag { Bool, Num, Obj } tag = Tag::Num;
    bool b = false;
    Rational n;
    int obj = 0;
    std::string key() const {
        switch (tag) {
            case Tag::Bool: return b ? "true" : "false";
            case Tag::Num: return n.str();
            case Tag::Obj: return "@" + std::to_string(obj);
        }
        return "?";
    }
};

std::optional<Value> eval(Term t, const Model& m);

std::optional<Value> evalApp(Term t, const Model& m) {
    std::vector<Value> args;
    for (Term a : t->args) {
        auto v = eval(a, m);
        if (!v) return std::nullopt;
        args.push_back(*v);
    }
    Value r;
    r.tag = Value::Tag::Num;
    const std::string& f = t->name;
    if (f == "*" && args.size() == 2) {
        r.n = args[0].n * args[1].n;
        return r;
    }
    if (f == "/" && args.size() == 2) {
        if (args[1].n.isZero()) return std::nullopt;
        r.n = args[0].n / args[1].n;
        return r;
    }
    if ((f == "quotient" || f == "remainder" || f == "modulo") && args.size() == 2) {
        if (!args[0].n.isInteger() || !args[1].n.isInteger() || args[1].n.isZero()) {
            return std::nullopt;
        }
        Integer a = args[0].n.num(), b = args[1].n.num();
        Integer q = Integer::truncDiv(a, b);
        if (f == "quotient") r.n = Rational(q);
        else if (f == "remainder") r.n = Rational(a - b * q);
        else r.n = Rational(Integer::floorMod(a, b));
        return r;
    }
    std::string key = f + "(";
    for (size_t i = 0; i < args.size(); ++i) key += (i ? "," : "") + args[i].key();
    key += ")";
    auto it = m.applications.find(key);
    if (it == m.applications.end()) return std::nullopt;
    const std::string& s = it->second;
    if (s == "true" || s == "false") {
        r.tag = Value::Tag::Bool;
        r.b = s == "true";
    } else if (!s.empty() && s[0] == '@') {
        r.tag = Value::Tag::Obj;
        r.obj = std::stoi(s.substr(1));
    } else {
        auto q = Rational::parse(s);
        if (!q) return std::nullopt;
        r.n = *q;
    }
    return r;
}

std::optional<Value> eval(Term t, const Model& m) {
    Value r;
    switch (t->kind) {
        case Kind::BoolConst:
            r.tag = Value::Tag::Bool;
            r.b = t->bval;
            return r;
        case Kind::Numeral:
            r.n = t->value;
            return r;
        case Kind::Var: {
            if (t->sort == Sort::Bool) {
                auto it = m.booleans.find(t->name);
                if (it == m.booleans.end()) return std::nullopt;
                r.tag = Value::Tag::Bool;
                r.b = it->second;
            } else if (t->sort == Sort::Obj) {
                auto it = m.objects.find(t->name);
                if (it == m.objects.end()) return std::nullopt;
                r.tag = Value::Tag::Obj;
                r.obj = it->second;
            } else {
                auto it = m.numbers.find(t->name);
                if (it == m.numbers.end()) return std::nullopt;
                if (t->sort == Sort::Int && !it->second.isInteger()) return std::nullopt;
                r.n = it->second;
            }
            return r;
        }
        case Kind::App:
            return evalApp(t, m);
        case Kind::Add: {
            Rational s(0);
            for (Term a : t->args) {
                auto v = eval(a, m);
                if (!v) return std::nullopt;
                s += v->n;
            }
            r.n = s;
            return r;
        }
        case Kind::Scale: {
            auto v = eval(t->args[0], m);
            if (!v) return std::nullopt;
            r.n = t->value * v->n;
            return r;
        }
        case Kind::Ite: {
            auto c = eval(t->args[0], m);
            if (!c) return std::nullopt;
            return eval(c->b ? t->args[1] : t->args[2], m);
        }
        case Kind::Eq: {
            auto a = eval(t->args[0], m), b = eval(t->args[1], m);
            if (!a || !b) return std::nullopt;
            r.tag = Value::Tag::Bool;
            r.b = a->key() == b->key();
            return r;
        }
        case Kind::Le:
        case Kind::Lt: {
            auto a = eval(t->args[0], m), b = eval(t->args[1], m);
            if (!a || !b) return std::nullopt;
            r.tag = Value::Tag::Bool;
            r.b = t->kind == Kind::Le ? a->n <= b->n : a->n < b->n;
            return r;
        }
        case Kind::Not: {
            auto a = eval(t->args[0], m);
            if (!a) return std::nullopt;
            r.tag = Value::Tag::Bool;
            r.b = !a->b;
            return r;
        }
        case Kind::And:
        case Kind::Or: {
            bool is_and = t->kind == Kind::And;
            bool acc = is_and;
            for (Term a : t->args) {
                auto v = eval(a, m);
                if (!v) return std::nullopt;
                acc = is_and ? (acc && v->b) : (acc || v->b);
            }
            r.tag = Value::Tag::Bool;
            r.b = acc;
            return r;
        }
        case Kind::Implies: {
            auto a = eval(t->args[0], m), b = eval(t->args[1], m);
            if (!a || !b) return std::nullopt;
            r.tag = Value::Tag::Bool;
            r.b = !a->b || b->b;
            return r;
        }
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::string> evaluateToString(Term t, const Model& m) {
    auto v = eval(t, m);
    if (!v) return std::nullopt;
    return v->key();
}

std::optional<bool> evaluate(Term formula, const Model& m) {
    auto v = eval(formula, m);
    if (!v || v->tag != Value::Tag::Bool) return std::nullopt;
    return v->b;
}

}  // namespace eshkol::smt
