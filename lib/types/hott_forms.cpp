/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Front-end side of the HoTT forms (ADR-0043): converts the reader datum the
 * parser recorded for a `(hott ...)` / `(hott-eval ...)` node into kernel
 * S-expressions, runs the kernel, and replaces an evaluated form by its
 * normal form. One kernel per TypeEnvironment, so declarations persist for
 * the whole compilation (and across REPL inputs).
 */

#include "eshkol/types/hott_forms.h"
#include "eshkol/types/hott_cubical.h"
#include "eshkol/frontend/syntax_datum.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>


namespace eshkol::hott {

namespace {

bool toSx(const SyntaxDatum& d, cubical::Sx& out, std::string& err) {
    using K = SyntaxDatum::Kind;
    out = cubical::Sx{};
    out.line = d.line;
    out.column = d.column;
    switch (d.kind) {
        case K::Symbol:
            out.kind = cubical::Sx::Kind::Symbol;
            out.text = d.text;
            return true;
        case K::Atom: {
            bool digits = !d.text.empty();
            for (char c : d.text)
                if (c < '0' || c > '9') digits = false;
            if (!digits) {
                err = "`" + d.text + "` is not part of the HoTT syntax";
                return false;
            }
            out.kind = cubical::Sx::Kind::Number;
            out.text = d.text;
            return true;
        }
        case K::List:
        case K::Vector:
            if (d.dotted) { err = "dotted lists are not part of the HoTT syntax"; return false; }
            out.kind = cubical::Sx::Kind::List;
            for (auto& item : d.items) {
                cubical::Sx s;
                if (!toSx(item, s, err)) return false;
                out.items.push_back(std::move(s));
            }
            return true;
        case K::Prefix:
            err = "quotation is not part of the HoTT syntax";
            return false;
    }
    return false;
}

std::shared_ptr<const SyntaxDatum> datumOf(const eshkol_ast_t* ast) {
    if (!ast || ast->type != ESHKOL_OP || ast->operation.op != ESHKOL_HOTT_OP) return nullptr;
    return syntax_use(ast->node_id);
}

}  // namespace

bool hottFormIsEval(const eshkol_ast_t* ast) {
    auto d = datumOf(ast);
    return d && d->kind == SyntaxDatum::Kind::List && !d->items.empty() &&
           d->items[0].text == "hott-eval";
}

HottFormOutcome elaborateHottForm(cubical::Kernel& kernel, eshkol_ast_t* ast) {
    HottFormOutcome out;
    out.line = ast->line;
    out.column = ast->column;
    auto d = datumOf(ast);
    cubical::Sx form;
    std::string err;
    if (!d || !toSx(*d, form, err) || !form.isList() || form.items.empty()) {
        out.ok = false;
        out.message = err.empty() ? "malformed HoTT form" : err;
        return out;
    }
    std::vector<cubical::Sx> rest(form.items.begin() + 1, form.items.end());
    if (form.items[0].isSymbol("hott")) {
        cubical::KernelResult r = kernel.declare(rest);
        if (!r.ok) {
            out.ok = false;
            out.message = r.message;
            if (r.line) { out.line = r.line; out.column = r.column; }
        }
        return out;
    }
    if (rest.size() != 1) {
        out.ok = false;
        out.message = "(hott-eval term) takes one term";
        return out;
    }
    cubical::KernelResult r = kernel.evaluate(rest[0]);
    if (!r.ok) {
        out.ok = false;
        out.message = r.message;
        if (r.line) { out.line = r.line; out.column = r.column; }
        return out;
    }
    // The normal form, as a quoted datum read by the same front end.
    std::istringstream in("(quote " + cubical::printSx(r.value) + ")");
    eshkol_ast_t q = eshkol_parse_next_ast_from_stream(in);
    if (q.type == ESHKOL_INVALID) {
        out.ok = false;
        out.message = "the normal form could not be read back: " + cubical::printSx(r.value);
        return out;
    }
    q.line = ast->line;
    q.column = ast->column;
    *ast = q;
    out.rewritten = true;
    return out;
}

}  // namespace eshkol::hott

namespace {
std::unique_ptr<eshkol::hott::cubical::Kernel>& vmKernel() {
    static std::unique_ptr<eshkol::hott::cubical::Kernel> k;
    return k;
}
char* dupString(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}
}  // namespace

extern "C" char* eshkol_hott_form_source(const char* form_text, int* ok) {
    using namespace eshkol::hott::cubical;
    if (ok) *ok = 1;
    if (!form_text) {
        vmKernel() = std::make_unique<Kernel>();
        return dupString("");
    }
    if (!vmKernel()) vmKernel() = std::make_unique<Kernel>();
    std::vector<Sx> forms;
    std::string err;
    if (!readSx(form_text, forms, err) || forms.size() != 1 || !forms[0].isList() || forms[0].items.empty()) {
        if (ok) *ok = 0;
        return dupString(err.empty() ? "malformed HoTT form" : err);
    }
    const Sx& f = forms[0];
    std::vector<Sx> rest(f.items.begin() + 1, f.items.end());
    KernelResult r;
    if (f.items[0].isSymbol("hott")) r = vmKernel()->declare(rest);
    else if (rest.size() == 1) r = vmKernel()->evaluate(rest[0]);
    else { r.ok = false; r.message = "(hott-eval term) takes one term"; }
    if (!r.ok) {
        if (ok) *ok = 0;
        return dupString(r.message);
    }
    return dupString(f.items[0].isSymbol("hott") ? "" : printSx(r.value));
}
