/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_FRONTEND_SYNTAX_DATUM_H
#define ESHKOL_FRONTEND_SYNTAX_DATUM_H

/**
 * @file syntax_datum.h
 * @brief Syntax as the reader produced it: the object `syntax-rules`
 *        matches and instantiates (ADR-0026).
 *
 * R7RS 4.3 defines macro transformers over *syntax* -- the datum the reader
 * produced -- not over the parser's lowered AST. The native expander used to
 * match patterns against lowered ASTs and substitute into a template that had
 * already been parsed as code, so every AST payload shape (a `do` scaffold, a
 * `let-values` formals array, a nested `define-syntax`) needed its own
 * substitution and renaming case, and each missing case was a silent miscompile.
 *
 * Now the parser keeps the datum of every macro use and of every
 * `syntax-rules` transformer in the side tables below, keyed the way
 * ADR-0008 prescribes (a `NodeId` for a use, the definition record for a
 * transformer). The expander rewrites datums and hands the result back to
 * the parser, so a template means exactly what the same text means anywhere
 * else in a program.
 *
 * Every atom keeps the token it was read from (type, spelling, position), so
 * re-parsing an expansion reproduces the original literal exactly -- an
 * exact rational stays exact, a string keeps its escapes decoded once.
 */

#include <eshkol/eshkol.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace eshkol {

/** One node of reader syntax. */
struct SyntaxDatum {
    enum class Kind : uint8_t {
        Symbol,  ///< identifier; `text` is its name (possibly colored)
        Atom,    ///< any other token (number, string, char, boolean, ...)
        List,    ///< `( items... )`, or `( items... . tail )` when `dotted`
        Vector,  ///< `#( items... )`; `text`/`token_type` name the opener
        Prefix   ///< reader prefix `'` `` ` `` `,` `,@` applied to items[0]
    };

    Kind kind = Kind::Atom;
    int token_type = 0;        ///< parser token type (opaque to the expander)
    std::string text;          ///< symbol name / token spelling / prefix text
    bool verbatim = false;     ///< symbol spelled with |vertical bars|
    bool dotted = false;       ///< List only: items.back() is the dotted tail
    uint32_t line = 0;
    uint32_t column = 0;
    std::vector<SyntaxDatum> items;

    bool isSymbol() const { return kind == Kind::Symbol; }
    bool isList() const { return kind == Kind::List; }
    /** Number of proper elements (excludes a dotted tail). */
    size_t properCount() const {
        return dotted && !items.empty() ? items.size() - 1 : items.size();
    }
};

/** A `syntax-rules` transformer as written. */
struct MacroSyntax {
    std::string ellipsis = "...";                 ///< R7RS 4.3.2 custom ellipsis
    std::vector<std::string> literals;
    std::vector<std::pair<SyntaxDatum, SyntaxDatum>> rules;  ///< (pattern, template)
};

/**
 * The reader syntax of calls.
 *
 * A call whose head the parser already knew to be a macro keyword is
 * recorded with its datum. Every other symbol-headed call is recorded as a
 * reference into its tokenizer's tape (the tokens in reading order), so its
 * datum can be recovered when the expander finds that the head is a macro
 * keyword after all: a forward reference, a macro exported by another
 * module, or a macro another expansion defined. Nothing is materialized
 * until it is asked for.
 */
void syntax_record_use(eshkol_node_id_t id, SyntaxDatum use);
void syntax_record_use_tape(eshkol_node_id_t id, std::shared_ptr<const void> tape,
                            uint32_t start);
/** True if the parser read call @p id as a macro use: its operands were
 *  never parsed as expressions. */
bool syntax_use_unparsed(eshkol_node_id_t id);
/** The datum of call @p id, or null if the parser recorded none. */
std::shared_ptr<const SyntaxDatum> syntax_use(eshkol_node_id_t id);
/** Rename identifiers of call @p id when it is materialized (module privacy). */
void syntax_add_use_renames(eshkol_node_id_t id,
                            std::shared_ptr<const std::map<std::string, std::string>> names);
/** Materialize the datum starting at @p start on a parser tape (parser.cpp). */
SyntaxDatum syntax_read_tape(const void* tape, uint32_t start);

/** Record the transformer of @p macro. */
void syntax_record_macro(const eshkol_macro_def_t* macro, MacroSyntax syntax);
/** The transformer recorded for @p macro, or null. */
std::shared_ptr<const MacroSyntax> syntax_macro(const eshkol_macro_def_t* macro);
/** Replace the transformer of @p macro (module-private renaming). */
void syntax_replace_macro(const eshkol_macro_def_t* macro, MacroSyntax syntax);

/**
 * Parse one datum as an Eshkol expression, exactly as the parser would parse
 * the same text. @p macro_names are the macro keywords visible where the
 * datum is spliced; a list headed by one of them is recorded as a macro use.
 * Implemented by the parser (lib/frontend/parser.cpp).
 */
eshkol_ast_t parse_syntax_datum(const SyntaxDatum& datum,
                                const std::set<std::string>& macro_names);

/**
 * While alive, the next form parsed is a top-level form of a program
 * (R7RS 5.1): a `begin` there splices its forms, definitions included, into
 * the top level instead of scoping them, and passes the same status to its own
 * forms, as a top-level `with-region` does. The parser's top-level entry
 * points and the macro expander (for the expansion of a top-level macro use)
 * open one. Implemented by the parser (lib/frontend/parser.cpp).
 */
class ToplevelFormParseScope {
public:
    ToplevelFormParseScope();
    ~ToplevelFormParseScope();
    ToplevelFormParseScope(const ToplevelFormParseScope&) = delete;
    ToplevelFormParseScope& operator=(const ToplevelFormParseScope&) = delete;
private:
    bool previous_;
};

/**
 * Splice top-level sequences into the program, recursively (R7RS 5.1): every
 * ESHKOL_SEQUENCE_OP in @p forms -- a top-level `begin`, or a parser
 * expansion such as define-record-type -- is replaced by its forms, in order.
 * The one splice rule for every consumer of a top-level form list (the
 * driver, the code generator after macro expansion, the REPL).
 */
void splice_toplevel_forms(std::vector<eshkol_ast_t>& forms);

/**
 * Apply @p rename to every identifier occurrence that is code rather than
 * data: symbols under `quote`, and under `quasiquote` outside its `unquote`
 * escapes, are left alone. Used for module-private renaming.
 */
void syntax_rename_identifiers(SyntaxDatum& datum,
                               const std::function<void(std::string&)>& rename);

} // namespace eshkol

#endif // ESHKOL_FRONTEND_SYNTAX_DATUM_H
