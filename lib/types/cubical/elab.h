/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_TYPES_CUBICAL_ELAB_H
#define ESHKOL_TYPES_CUBICAL_ELAB_H

#include "cubical_internal.h"

#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace eshkol::hott::cubical {

TermP readback(const V& v);
LExprP levelTerm(const Level& l);
IExprP formulaTerm(const Formula& f);

/** Elaborates S-expressions against one signature (see elab.cpp). */
class Elab {
public:
    explicit Elab(Signature& s);

    struct Entry {
        enum class Kind : uint8_t { Value, Dim, Level } kind = Kind::Value;
        std::string surface;
        std::string internal;
        V type;
        V value;
        Formula f;
    };

    // terms
    TermP check(const Sx& sx, const V& T);
    std::pair<TermP, V> infer(const Sx& sx);
    std::pair<TermP, Level> inferType(const Sx& sx);
    TermP checkTypeAt(const Sx& sx, const Level& l);
    V evalT(const TermP& t);
    EnvP env() const;

    // declarations (decl.cpp)
    void declare(const Sx& decl);

    // context
    V bindVar(const std::string& surface, const V& type);
    Dim bindDim(const std::string& surface);
    void bindLevel(const std::string& surface);
    void pop();

    std::function<void(const std::string&)> importModule;

private:
    Signature& S;
    std::vector<Entry> ctx;
    std::vector<EnvP> envs;

    // structural recursion of the definition being checked
    std::string selfName;
    int selfArgPos = -1;
    bool topSpine = false;
    int spineArgs = 0;
    std::set<std::string> smaller;
    std::vector<std::function<void()>> deferred;

    // the data type being declared (for recursive occurrences)
    const DataDecl* dataInProgress = nullptr;
    std::vector<std::string> dataParamInternals;

    void push(Entry e);
    const Entry* find(const std::string& name) const;
    void underFace(const Face& f, const std::function<void()>& body);

    LExprP level(const Sx& sx);
    bool isLevelExpr(const Sx& sx) const;
    IExprP interval(const Sx& sx);
    IExprP face(const Sx& sx);

    std::vector<std::pair<std::string, Sx>> binders(const Sx& sx);
    TermP binderType(bool pi, const std::vector<std::pair<std::string, Sx>>& bs, size_t k,
                     const Sx& body, Level& lvl);
    void expect(const Sx& at, const V& T, const V& got, const V& want, const char* what);

    std::pair<TermP, V> inferSymbol(const Sx& sx);
    std::pair<TermP, V> inferLet(const Sx& sx, const V* expected);
    std::pair<TermP, V> applyArgs(const Sx& sx, TermP f, V T, size_t from);
    std::pair<TermP, V> inferDataApp(const Sx& sx, const DataDecl& d);
    std::pair<TermP, V> inferCon(const Sx& sx, const DataDecl* d, int c);
    std::pair<TermP, V> inferPathP(const Sx& sx);
    std::pair<TermP, V> inferTransp(const Sx& sx);
    std::pair<TermP, V> inferHComp(const Sx& sx);
    std::pair<TermP, V> inferComp(const Sx& sx);
    std::pair<TermP, V> inferGlue(const Sx& sx);
    std::pair<TermP, V> inferSelfCall(const Sx& sx);
    TermP checkCon(const Sx& sx, const V& D, const DataDecl* d, int c);
    TermP checkLambda(const Sx& sx, const V& T);
    TermP checkSplit(const Sx& sx, const V& T);
    TermP checkGlue(const Sx& sx, const V& T);
    TermP typeLine(const Sx& binder, const Sx& A, std::string& internal, Level& l);
    std::vector<TSysClause> lineSystem(const Sx& clauses, const std::string& binderSurface,
                                       const std::string& binderInternal,
                                       const std::function<V(const Formula& i)>& lineType,
                                       const V& u0, const V& baseType);
    std::vector<TSysClause> readbackGlueSys(const VSys& s);
    V equivType(const V& T, const V& A, const Level& l);
    V globalType(const GlobalDef& g, const std::vector<LExprP>& ls);
    void runDeferred();

    // decl.cpp
    void declData(const Sx& sx);
    void declDefine(const Sx& sx, bool example);
    void declBuiltin(const Sx& sx);
    void checkNameFree(const Sx& at, const std::string& name);
};

}  // namespace eshkol::hott::cubical

#endif  // ESHKOL_TYPES_CUBICAL_ELAB_H
