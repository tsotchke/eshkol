/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_TYPES_CUBICAL_INTERNAL_H
#define ESHKOL_TYPES_CUBICAL_INTERNAL_H

/**
 * @file cubical_internal.h
 * @brief Representation shared by the cubical kernel's translation units.
 *
 * Layers (ADR-0043):
 *   - the interval: De Morgan formulas over dimension atoms, and faces
 *     (cofibrations) in disjunctive normal form;
 *   - universe levels: max(c, v1 + k1, ...) over level variables;
 *   - core terms, produced by the elaborator from S-expressions, with named
 *     binders resolved through environments;
 *   - values, in weak head normal form. Term binders become closures;
 *     dimension binders are nominal (a fresh atom and a value body), so a
 *     value can be restricted to a face or have a dimension substituted, and
 *     every stuck form re-reduces when a substitution unblocks it.
 */

#include "eshkol/types/hott_cubical.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eshkol::hott::cubical {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** A kernel rejection: type error, scope error or malformed syntax. */
struct KernelError : std::runtime_error {
    uint32_t line = 0, column = 0;
    explicit KernelError(const std::string& m, uint32_t l = 0, uint32_t c = 0)
        : std::runtime_error(m), line(l), column(c) {}
};

// ---------------------------------------------------------------------------
// The interval
// ---------------------------------------------------------------------------

using Dim = uint32_t;
Dim freshDim();

struct Lit {
    Dim d;
    bool pos;  ///< true: d, false: ~d
    bool operator<(const Lit& o) const { return d != o.d ? d < o.d : pos < o.pos; }
    bool operator==(const Lit& o) const { return d == o.d && pos == o.pos; }
};
using Conj = std::vector<Lit>;  ///< sorted, duplicate-free

/** An element of the free De Morgan algebra: a minimal DNF over literals. */
struct Formula {
    std::vector<Conj> dnf;  ///< sorted antichain; empty = 0, {{}} = 1

    static Formula zero() { return Formula{}; }
    static Formula one() { Formula f; f.dnf.push_back({}); return f; }
    static Formula atom(Dim d, bool pos = true) { Formula f; f.dnf.push_back({Lit{d, pos}}); return f; }
    static Formula endpoint(bool b) { return b ? one() : zero(); }

    bool isZero() const { return dnf.empty(); }
    bool isOne() const { return dnf.size() == 1 && dnf[0].empty(); }
    bool mentions(Dim d) const;
    void collect(std::set<Dim>& out) const;
    Formula neg() const;
    Formula meet(const Formula& o) const;
    Formula join(const Formula& o) const;
    Formula subst(Dim d, const Formula& r) const;
    bool operator==(const Formula& o) const;
    bool operator!=(const Formula& o) const { return !(*this == o); }
};

/** A face: a consistent conjunction of constraints d = 0 / d = 1 (sorted by d). */
using Face = std::vector<std::pair<Dim, bool>>;

/** The faces on which `r = 1` holds, as a normalized antichain (empty list = never). */
std::vector<Face> facesOf(const Formula& r);
std::optional<Face> faceMeet(const Face& a, const Face& b);
Formula faceFormula(const Face& f);
Formula facesFormula(const std::vector<Face>& fs);
/** Face lists compared as cofibrations. */
bool sameFaces(std::vector<Face> a, std::vector<Face> b);
bool faceMentions(const Face& f, Dim d);
/** Substitute d := r inside a face; the result is a list of faces. */
std::vector<Face> faceSubst(const Face& f, Dim d, const Formula& r);
std::vector<Face> normalizeFaces(std::vector<Face> fs);

// ---------------------------------------------------------------------------
// Universe levels
// ---------------------------------------------------------------------------

struct Level {
    uint32_t c = 0;                          ///< constant part
    std::map<std::string, uint32_t> vars;    ///< level variable -> offset
    static Level constant(uint32_t n) { Level l; l.c = n; return l; }
    static Level var(const std::string& v) { Level l; l.vars[v] = 0; return l; }
    Level suc() const;
    Level max(const Level& o) const;
    bool leq(const Level& o) const;
    bool operator==(const Level& o) const;
    std::string str() const;
    void normalize();
};

// ---------------------------------------------------------------------------
// Core terms
// ---------------------------------------------------------------------------

struct IExpr;
using IExprP = std::shared_ptr<const IExpr>;
struct IExpr {
    enum class K : uint8_t { Zero, One, Var, Neg, Min, Max } k = K::Zero;
    std::string name;
    IExprP a, b;
};

struct LExpr;
using LExprP = std::shared_ptr<const LExpr>;
struct LExpr {
    enum class K : uint8_t { Const, Var, Suc, Max } k = K::Const;
    uint32_t n = 0;
    std::string name;
    LExprP a, b;
};

struct Term;
using TermP = std::shared_ptr<const Term>;
struct DataDecl;

struct TSysClause {
    IExprP face;
    TermP t;   ///< the clause term (a line for hcomp/comp, T for Glue, t for glue)
    TermP t2;  ///< Glue only: the equivalence
};

struct SplitClause {
    int con = -1;
    std::vector<std::string> args;
    std::vector<std::string> dims;
    TermP body;
};

struct Term {
    enum class K : uint8_t {
        Var, Global, U, Pi, Lam, App, Sigma, Pair, Fst, Snd, Let,
        PathP, PLam, PApp, Data, Con, Split, Transp, HComp, Comp,
        Glue, GlueElem, Unglue
    } k = K::Var;
    std::string name;               ///< Var/Global name; binder of Pi/Lam/Sigma/Let/PLam/Transp/HComp/Comp
    std::vector<LExprP> levels;     ///< Global/Data/Con: level arguments
    LExprP level;                   ///< U
    TermP a, b, c, d;               ///< operands (see eval.cpp for each kind)
    IExprP r;                       ///< PApp argument; Transp constancy formula
    std::vector<TSysClause> sys;    ///< HComp/Comp/Glue/GlueElem/Unglue systems
    std::vector<TermP> params, args;
    std::vector<IExprP> dims;
    const DataDecl* data = nullptr;
    int con = -1;
    std::vector<SplitClause> clauses;  ///< Split
    std::string motiveVar;             ///< Split: x in (x : D) -> B; B is `b`
    uint32_t line = 0, column = 0;
};

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

struct Val;
using V = std::shared_ptr<const Val>;
struct Env;
using EnvP = std::shared_ptr<const Env>;

struct Env {
    enum class Kind : uint8_t { Value, Dim, Level } kind = Kind::Value;
    std::string name;
    V v;
    Formula f;
    Level l;
    EnvP next;
};

struct Clo {
    EnvP env;
    std::string x;
    TermP body;
};

struct VClause {
    Face face;
    V v;   ///< line (hcomp), T (Glue), t (glue)
    V v2;  ///< Glue: the equivalence
};
using VSys = std::vector<VClause>;

struct Val {
    enum class K : uint8_t {
        // canonical
        U, Pi, Lam, Sigma, Pair, PathP, PLam, Data, Con, Split, Glue, GlueElem,
        // Kan operations kept as values: they reduce when eliminated (at Pi
        // types), are the hcomp constructor (at inductive types), or are
        // stuck (at neutral types)
        HComp, Transp,
        // neutral
        Var, App, Fst, Snd, PApp, SplitApp, Unglue
    } k = K::Var;
    Level level;                    ///< U
    std::string name;               ///< binder / variable name hint
    V a, b, c;                      ///< see the constructors in eval.cpp
    Clo clo;                        ///< Pi/Sigma codomain, Lam body, Split clauses
    Dim dim = 0;                    ///< PLam binder
    Formula f;                      ///< PApp argument, Transp constancy face formula
    VSys sys;
    const DataDecl* data = nullptr;
    int con = -1;
    std::vector<Level> levels;      ///< Data/Con
    std::vector<V> params, args;    ///< Data/Con
    std::vector<Formula> dims;      ///< Con
    uint64_t id = 0;                ///< Var
    TermP split;                    ///< Split
};

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

struct ConDecl {
    std::string name;
    std::vector<std::pair<std::string, TermP>> args;   ///< telescope over levels, params, earlier args
    std::vector<bool> recursive;                       ///< arg type is the data type itself
    std::vector<std::string> dims;
    std::vector<std::pair<IExprP, TermP>> boundary;    ///< faces over dims; terms over params, args, dims
    /// Every boundary term is a constructor expression over the constructor's
    /// own arguments, so transport commutes with the constructor (CHM 2018).
    bool structuralBoundary = true;
};

struct DataDecl {
    std::string name;
    std::vector<std::string> levelParams;
    std::vector<std::pair<std::string, TermP>> params;
    LExprP universe;
    std::vector<ConDecl> cons;
    bool higher = false;  ///< has a path constructor
    int conIndex(const std::string& n) const {
        for (size_t i = 0; i < cons.size(); ++i) if (cons[i].name == n) return (int)i;
        return -1;
    }
};

struct GlobalDef {
    std::string name;
    std::vector<std::string> levelParams;
    TermP type;
    TermP body;        ///< null while the definition is being checked
    V opaque;          ///< the neutral that stands for it while it is being checked
    std::map<std::string, V> cache;  ///< value per level instantiation
};

/** The global signature the evaluator reads. */
struct Signature {
    std::map<std::string, std::shared_ptr<GlobalDef>> globals;
    std::map<std::string, std::shared_ptr<DataDecl>> datas;
    std::map<std::string, std::pair<const DataDecl*, int>> constructors;
    const DataDecl* natural = nullptr;   ///< (builtin natural ...): numerals
    int natZero = -1, natSuc = -1;
    // Kernel builtins, defined by the kernel prelude.
    std::string equivName = "Equiv";
    std::string pathToEquivName = "path-to-equiv";
};

/** The signature the evaluator is running against (set by the kernel). */
const Signature& sig();
struct SignatureScope {
    const Signature* prev;
    explicit SignatureScope(const Signature* s);
    ~SignatureScope();
};

// ---------------------------------------------------------------------------
// Evaluation (eval.cpp)
// ---------------------------------------------------------------------------

EnvP envValue(const EnvP& e, const std::string& n, V v);
EnvP envDim(const EnvP& e, const std::string& n, Formula f);
EnvP envLevel(const EnvP& e, const std::string& n, Level l);
Level evalLevel(const EnvP& env, const LExprP& l);
Formula evalI(const EnvP& env, const IExprP& r);

V eval(const EnvP& env, const TermP& t);
V apply(const Clo& c, const V& arg);
V app(const V& f, const V& arg);
V vfst(const V& p);
V vsnd(const V& p);
/** Path application; a0/a1 are the endpoints the type gives. */
V papp(const V& p, const Formula& r, const V& a0, const V& a1);
/** Apply a line (a PLam value) to a formula. */
V lineAt(const V& line, const Formula& r);
V mkPLam(Dim i, V body);
/** Build the line `i |-> f(i)` for a fresh i. */
V mkLine(const std::function<V(const Formula&)>& f);

V mkU(Level l);
V mkVar(const std::string& name, V type);
V mkPair(V a, V b);
V mkData(const DataDecl* d, std::vector<Level> ls, std::vector<V> ps);
V mkCon(const DataDecl* d, std::vector<Level> ls, std::vector<V> ps, int con,
        std::vector<V> args, std::vector<Formula> dims);
V mkGlue(V A, VSys sys);
V mkGlueElem(VSys sys, V a);
V unglue(V x, V A, VSys sys);

/** transp along a line of types, constant on phi. */
V transp(const V& line, const Formula& phi, const V& u0);
/** hcomp in type A; sys clauses are lines. */
V hcomp(const V& A, VSys sys, const V& u0);
/** Heterogeneous composition along a line of types, constant on psi. */
V comp(const V& line, const Formula& psi, VSys sys, const V& u0);
/** hfill: the line i |-> hcomp A [sys(i /\ j), i = 0 |-> u0] u0. */
V hfill(const V& A, const VSys& sys, const V& u0);

/** Substitute dimension d := r in a value. */
V act(const V& v, Dim d, const Formula& r);
V restrict(const V& v, const Face& f);
VSys actSys(const VSys& s, Dim d, const Formula& r);
bool occurs(const V& v, Dim d);
/** The clause whose face is true, if any. */
const VClause* trueClause(const VSys& s);
std::vector<Face> sysFaces(const VSys& s);

/** The domain/codomain/type data of canonical type values. */
V piDom(const V& pi);
V piCod(const V& pi, const V& x);
/** Types of the arguments of constructor `con` at levels/params, given earlier args. */
V conArgType(const DataDecl* d, const std::vector<Level>& ls, const std::vector<V>& ps,
             int con, const std::vector<V>& earlier);
V paramType(const DataDecl* d, const std::vector<Level>& ls, const std::vector<V>& earlier);
/** The universe level of a data type's instances. */
Level dataLevel(const DataDecl* d, const std::vector<Level>& ls);
/** Value of the global at levels. */
V globalValue(const std::string& name, const std::vector<Level>& ls);

// ---------------------------------------------------------------------------
// Conversion, typing of neutrals and printing (conv.cpp)
// ---------------------------------------------------------------------------

/** Definitional equality of a and b at type T. */
bool conv(const V& T, const V& a, const V& b);
/** Definitional equality of two types. */
bool convType(const V& A, const V& B);
/** Cumulative subtyping: U l <= U m when l <= m, covariant in Pi codomains. */
bool subtype(const V& A, const V& B);
/** The type of a stuck value. */
V typeOfStuck(const V& v);
bool isStuck(const V& v);
/** Print a value as an S-expression (normal form). */
Sx quote(const V& v);
std::string show(const V& v);

}  // namespace eshkol::hott::cubical

#endif  // ESHKOL_TYPES_CUBICAL_INTERNAL_H
