/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * The interval of the cubical kernel (ADR-0043): the free De Morgan algebra on
 * dimension atoms, faces (the cofibrations of CCHM), universe levels, and the
 * S-expression reader/printer the kernel consumes.
 *
 * The free De Morgan algebra on a set X is the free bounded distributive
 * lattice on X + ~X, so a minimal disjunctive normal form over literals is a
 * canonical representative: equality of formulas is equality of normal
 * forms. A conjunction containing both i and ~i is a legitimate element of
 * the algebra (i /\ ~i is not 0), but as a face it is unsatisfiable: faces
 * are evaluated in the face lattice, where i = 0 /\ i = 1 is false.
 */

#include "cubical_internal.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <sstream>

namespace eshkol::hott::cubical {

Dim freshDim() {
    static std::atomic<uint32_t> next{1};
    return next.fetch_add(1);
}

namespace {

bool subsetConj(const Conj& a, const Conj& b) {  // a ⊆ b
    return std::includes(b.begin(), b.end(), a.begin(), a.end());
}

std::vector<Conj> normalizeDnf(std::vector<Conj> cs) {
    for (auto& c : cs) {
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
    }
    std::sort(cs.begin(), cs.end(), [](const Conj& a, const Conj& b) {
        if (a.size() != b.size()) return a.size() < b.size();
        return a < b;
    });
    cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
    std::vector<Conj> out;
    for (auto& c : cs) {
        bool absorbed = false;
        for (auto& o : out)
            if (subsetConj(o, c)) { absorbed = true; break; }
        if (!absorbed) out.push_back(c);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

bool Formula::mentions(Dim d) const {
    for (auto& c : dnf)
        for (auto& l : c)
            if (l.d == d) return true;
    return false;
}

void Formula::collect(std::set<Dim>& out) const {
    for (auto& c : dnf)
        for (auto& l : c) out.insert(l.d);
}

Formula Formula::join(const Formula& o) const {
    Formula r;
    r.dnf = dnf;
    r.dnf.insert(r.dnf.end(), o.dnf.begin(), o.dnf.end());
    r.dnf = normalizeDnf(std::move(r.dnf));
    return r;
}

Formula Formula::meet(const Formula& o) const {
    Formula r;
    for (auto& a : dnf)
        for (auto& b : o.dnf) {
            Conj c = a;
            c.insert(c.end(), b.begin(), b.end());
            r.dnf.push_back(std::move(c));
        }
    r.dnf = normalizeDnf(std::move(r.dnf));
    return r;
}

Formula Formula::neg() const {
    // ~(\/_k /\_l x_kl) = /\_k \/_l ~x_kl
    Formula acc = Formula::one();
    for (auto& c : dnf) {
        Formula disj;  // zero
        for (auto& l : c) disj = disj.join(Formula::atom(l.d, !l.pos));
        acc = acc.meet(disj);
    }
    return acc;
}

Formula Formula::subst(Dim d, const Formula& r) const {
    if (!mentions(d)) return *this;
    Formula nr = r.neg();
    Formula out;
    for (auto& c : dnf) {
        Formula term = Formula::one();
        for (auto& l : c) {
            if (l.d == d) term = term.meet(l.pos ? r : nr);
            else term = term.meet(Formula::atom(l.d, l.pos));
        }
        out = out.join(term);
    }
    return out;
}

bool Formula::operator==(const Formula& o) const { return dnf == o.dnf; }

// ---------------------------------------------------------------------------
// Faces
// ---------------------------------------------------------------------------

std::vector<Face> normalizeFaces(std::vector<Face> fs) {
    for (auto& f : fs) std::sort(f.begin(), f.end());
    std::sort(fs.begin(), fs.end(), [](const Face& a, const Face& b) {
        if (a.size() != b.size()) return a.size() < b.size();
        return a < b;
    });
    fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
    std::vector<Face> out;
    for (auto& f : fs) {
        bool absorbed = false;
        for (auto& o : out)
            if (std::includes(f.begin(), f.end(), o.begin(), o.end())) { absorbed = true; break; }
        if (!absorbed) out.push_back(f);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<Face> facesOf(const Formula& r) {
    std::vector<Face> out;
    for (auto& c : r.dnf) {
        Face f;
        bool consistent = true;
        for (auto& l : c) {
            auto it = std::find_if(f.begin(), f.end(), [&](auto& p) { return p.first == l.d; });
            if (it != f.end()) {
                if (it->second != l.pos) { consistent = false; break; }
            } else {
                f.push_back({l.d, l.pos});
            }
        }
        if (consistent) out.push_back(std::move(f));
    }
    return normalizeFaces(std::move(out));
}

std::optional<Face> faceMeet(const Face& a, const Face& b) {
    Face out = a;
    for (auto& p : b) {
        auto it = std::find_if(out.begin(), out.end(), [&](auto& q) { return q.first == p.first; });
        if (it != out.end()) {
            if (it->second != p.second) return std::nullopt;
        } else {
            out.push_back(p);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

Formula faceFormula(const Face& f) {
    Formula r = Formula::one();
    for (auto& p : f) r = r.meet(Formula::atom(p.first, p.second));
    return r;
}

Formula facesFormula(const std::vector<Face>& fs) {
    Formula r = Formula::zero();
    for (auto& f : fs) r = r.join(faceFormula(f));
    return r;
}

bool sameFaces(std::vector<Face> a, std::vector<Face> b) {
    return normalizeFaces(std::move(a)) == normalizeFaces(std::move(b));
}

bool faceMentions(const Face& f, Dim d) {
    for (auto& p : f)
        if (p.first == d) return true;
    return false;
}

std::vector<Face> faceSubst(const Face& f, Dim d, const Formula& r) {
    Formula acc = Formula::one();
    for (auto& p : f) {
        if (p.first == d) acc = acc.meet(p.second ? r : r.neg());
        else acc = acc.meet(Formula::atom(p.first, p.second));
    }
    return facesOf(acc);
}

// ---------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------

void Level::normalize() {
    for (auto& [v, k] : vars)
        if (c <= k) { c = 0; break; }
}

Level Level::suc() const {
    Level l;
    l.c = c + 1;
    for (auto& [v, k] : vars) l.vars[v] = k + 1;
    l.normalize();
    return l;
}

Level Level::max(const Level& o) const {
    Level l = *this;
    l.c = std::max(c, o.c);
    for (auto& [v, k] : o.vars) {
        auto it = l.vars.find(v);
        if (it == l.vars.end()) l.vars[v] = k;
        else it->second = std::max(it->second, k);
    }
    l.normalize();
    return l;
}

bool Level::leq(const Level& o) const {
    // Every component of this must be bounded by a component of o, for all
    // assignments of the variables: a constant c is bounded by o.c or by any
    // v + k with k >= c; v + k is bounded only by v + k' with k <= k'.
    uint32_t omax = o.c;
    for (auto& [v, k] : o.vars) omax = std::max(omax, k);
    if (c > omax) return false;
    for (auto& [v, k] : vars) {
        auto it = o.vars.find(v);
        if (it == o.vars.end() || it->second < k) return false;
    }
    return true;
}

bool Level::operator==(const Level& o) const { return leq(o) && o.leq(*this); }

std::string Level::str() const {
    std::vector<std::string> parts;
    if (c > 0 || vars.empty()) parts.push_back(std::to_string(c));
    for (auto& [v, k] : vars) {
        std::string s = v;
        for (uint32_t i = 0; i < k; ++i) s = "(lsuc " + s + ")";
        parts.push_back(s);
    }
    if (parts.size() == 1) return parts[0];
    std::string out = "(lmax";
    for (auto& p : parts) out += " " + p;
    return out + ")";
}

// ---------------------------------------------------------------------------
// S-expressions
// ---------------------------------------------------------------------------

namespace {

struct Reader {
    const std::string& s;
    size_t pos = 0;
    uint32_t line = 1, col = 1;

    void adv() {
        if (s[pos] == '\n') { ++line; col = 1; } else { ++col; }
        ++pos;
    }
    void skip() {
        while (pos < s.size()) {
            char ch = s[pos];
            if (std::isspace((unsigned char)ch)) { adv(); continue; }
            if (ch == ';') { while (pos < s.size() && s[pos] != '\n') adv(); continue; }
            if (ch == '#' && pos + 1 < s.size() && s[pos + 1] == '|') {
                int depth = 0;
                while (pos < s.size()) {
                    if (s[pos] == '#' && pos + 1 < s.size() && s[pos + 1] == '|') { ++depth; adv(); adv(); continue; }
                    if (s[pos] == '|' && pos + 1 < s.size() && s[pos + 1] == '#') {
                        --depth; adv(); adv();
                        if (depth == 0) break;
                        continue;
                    }
                    adv();
                }
                continue;
            }
            if (ch == '#' && pos + 1 < s.size() && s[pos + 1] == ';') {
                adv(); adv();
                Sx ignored;
                std::string err;
                if (!read(ignored, err)) return;
                continue;
            }
            break;
        }
    }
    bool read(Sx& out, std::string& err) {
        skip();
        if (pos >= s.size()) { err = "unexpected end of input"; return false; }
        out = Sx{};
        out.line = line;
        out.column = col;
        char ch = s[pos];
        if (ch == '(' || ch == '[') {
            char close = ch == '(' ? ')' : ']';
            adv();
            out.kind = Sx::Kind::List;
            while (true) {
                skip();
                if (pos >= s.size()) { err = "unclosed list"; return false; }
                if (s[pos] == ')' || s[pos] == ']') {
                    if (s[pos] != close) { err = "mismatched bracket"; return false; }
                    adv();
                    return true;
                }
                Sx item;
                if (!read(item, err)) return false;
                out.items.push_back(std::move(item));
            }
        }
        if (ch == ')' || ch == ']') { err = "unexpected close bracket"; return false; }
        if (ch == '"') {
            adv();
            out.kind = Sx::Kind::String;
            while (pos < s.size() && s[pos] != '"') {
                if (s[pos] == '\\' && pos + 1 < s.size()) adv();
                out.text.push_back(s[pos]);
                adv();
            }
            if (pos >= s.size()) { err = "unclosed string"; return false; }
            adv();
            return true;
        }
        if (ch == '\'' || ch == '`' || ch == ',') { err = "quotation is not part of the HoTT syntax"; return false; }
        std::string tok;
        while (pos < s.size()) {
            char c = s[pos];
            if (std::isspace((unsigned char)c) || c == '(' || c == ')' || c == '[' || c == ']' ||
                c == '"' || c == ';')
                break;
            tok.push_back(c);
            adv();
        }
        bool numeric = !tok.empty();
        for (char c : tok)
            if (!std::isdigit((unsigned char)c)) numeric = false;
        out.kind = numeric ? Sx::Kind::Number : Sx::Kind::Symbol;
        out.text = tok;
        return true;
    }
};

}  // namespace

bool readSx(const std::string& text, std::vector<Sx>& out, std::string& error) {
    Reader r{text};
    while (true) {
        r.skip();
        if (r.pos >= text.size()) return true;
        Sx sx;
        if (!r.read(sx, error)) {
            error = "line " + std::to_string(r.line) + ": " + error;
            return false;
        }
        out.push_back(std::move(sx));
    }
}

std::string printSx(const Sx& sx) {
    switch (sx.kind) {
        case Sx::Kind::Symbol:
        case Sx::Kind::Number:
            return sx.text;
        case Sx::Kind::String:
            return "\"" + sx.text + "\"";
        case Sx::Kind::List: {
            std::string out = "(";
            for (size_t i = 0; i < sx.items.size(); ++i) {
                if (i) out += " ";
                out += printSx(sx.items[i]);
            }
            return out + ")";
        }
    }
    return "";
}

}  // namespace eshkol::hott::cubical
