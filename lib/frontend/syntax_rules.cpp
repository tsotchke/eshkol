/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Native adapter for the one syntax-rules engine (ADR-0026): converts the
 * parser's reader syntax to the engine's neutral tree and back. The engine
 * itself (matching, instantiation, coloring) is syntax_rules_core.h, the same
 * source the bytecode VM compiles.
 */
#include <eshkol/frontend/syntax_rules.h>
#include <eshkol/frontend/syntax_rules_core.h>

#include <memory>
#include <string>
#include <vector>

namespace eshkol {
namespace {

struct SynDeleter {
    void operator()(eshkol_syn* node) const { eshkol_syn_free(node); }
};
using SynPtr = std::unique_ptr<eshkol_syn, SynDeleter>;

eshkol_syn_kind kind_of(SyntaxDatum::Kind kind) {
    switch (kind) {
        case SyntaxDatum::Kind::Symbol: return ESHKOL_SYN_SYMBOL;
        case SyntaxDatum::Kind::Atom:   return ESHKOL_SYN_ATOM;
        case SyntaxDatum::Kind::List:   return ESHKOL_SYN_LIST;
        case SyntaxDatum::Kind::Vector: return ESHKOL_SYN_VECTOR;
        case SyntaxDatum::Kind::Prefix: return ESHKOL_SYN_PREFIX;
    }
    return ESHKOL_SYN_ATOM;
}

/** Convert; `origin` keeps the datum so atoms and positions come back intact.
 *  An atom's key includes its token type, so `1` never matches `"1"`. */
eshkol_syn* to_syn(const SyntaxDatum& datum) {
    std::string text = datum.text;
    if (datum.kind == SyntaxDatum::Kind::Atom)
        text = std::to_string(datum.token_type) + ":" + datum.text;
    eshkol_syn* node = eshkol_syn_new(kind_of(datum.kind), text.c_str(), &datum);
    if (!node) return nullptr;
    node->dotted = datum.dotted ? 1 : 0;
    for (const auto& item : datum.items) {
        if (!eshkol_syn_push(node, to_syn(item))) {
            eshkol_syn_free(node);
            return nullptr;
        }
    }
    return node;
}

SyntaxDatum from_syn(const eshkol_syn* node) {
    const auto* origin = static_cast<const SyntaxDatum*>(node->origin);
    SyntaxDatum datum;
    if (origin) {
        datum.token_type = origin->token_type;
        datum.verbatim = origin->verbatim;
        datum.line = origin->line;
        datum.column = origin->column;
        datum.text = origin->text;
    }
    switch (node->kind) {
        case ESHKOL_SYN_SYMBOL:
            datum.kind = SyntaxDatum::Kind::Symbol;
            datum.text = node->text;
            // An engine-made symbol (the head of `(quote x)` read from 'x)
            // borrows its origin from a non-symbol; it is not verbatim.
            if (origin && origin->kind != SyntaxDatum::Kind::Symbol) datum.verbatim = false;
            break;
        case ESHKOL_SYN_ATOM:
            datum.kind = SyntaxDatum::Kind::Atom;          // text/type from origin
            break;
        case ESHKOL_SYN_PREFIX:
            datum.kind = SyntaxDatum::Kind::Prefix;
            datum.text = node->text;
            break;
        case ESHKOL_SYN_LIST:
            datum.kind = SyntaxDatum::Kind::List;
            datum.text = "(";
            break;
        case ESHKOL_SYN_VECTOR:
            datum.kind = SyntaxDatum::Kind::Vector;        // opener from origin
            break;
    }
    datum.dotted = node->dotted != 0;
    datum.items.reserve(static_cast<size_t>(node->n_items));
    for (int i = 0; i < node->n_items; ++i) datum.items.push_back(from_syn(node->items[i]));
    return datum;
}

struct KeywordContext {
    const SyntaxKeywordResolver* resolver;
    std::string result;
};

const char* resolve_keyword(void* ctx, const char* name) {
    auto* context = static_cast<KeywordContext*>(ctx);
    if (!context->resolver || !*context->resolver) return nullptr;
    context->result = (*context->resolver)(name);
    return context->result.empty() ? nullptr : context->result.c_str();
}

} // namespace

SyntaxRulesOutcome syntax_rules_apply(const MacroSyntax& transformer,
                                      const SyntaxDatum& use,
                                      unsigned color,
                                      const SyntaxKeywordResolver& keyword,
                                      SyntaxDatum& expansion,
                                      std::string& error) {
    std::vector<SynPtr> owned;
    std::vector<eshkol_syn*> patterns, templates;
    for (const auto& rule : transformer.rules) {
        owned.emplace_back(to_syn(rule.first));
        patterns.push_back(owned.back().get());
        owned.emplace_back(to_syn(rule.second));
        templates.push_back(owned.back().get());
    }
    std::vector<const char*> literals;
    for (const auto& literal : transformer.literals) literals.push_back(literal.c_str());
    SynPtr use_syn(to_syn(use));

    KeywordContext context{&keyword, {}};
    eshkol_syn* result = nullptr;
    char message[256] = {0};
    const auto outcome = eshkol_syntax_rules_apply(
        transformer.ellipsis.c_str(), literals.data(), static_cast<int>(literals.size()),
        patterns.data(), templates.data(), static_cast<int>(patterns.size()),
        use_syn.get(), color, resolve_keyword, &context, &result, message, sizeof(message));
    SynPtr expanded(result);
    switch (outcome) {
        case ESHKOL_SYN_EXPANDED:
            expansion = from_syn(expanded.get());
            return SyntaxRulesOutcome::Expanded;
        case ESHKOL_SYN_NO_MATCH:
            return SyntaxRulesOutcome::NoMatch;
        case ESHKOL_SYN_ERROR:
            break;
    }
    error = message;
    return SyntaxRulesOutcome::Error;
}

} // namespace eshkol
