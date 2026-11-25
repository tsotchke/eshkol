/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_TYPES_HOTT_CUBICAL_H
#define ESHKOL_TYPES_HOTT_CUBICAL_H

/**
 * @file hott_cubical.h
 * @brief The cubical kernel of Eshkol's HoTT layer (ADR-0043).
 *
 * A dependent type theory with a De Morgan interval, path types, Kan
 * operations (transp and hcomp), Glue types, univalence and higher inductive
 * types, in the presentation of Cohen-Coquand-Huber-Mortberg (CCHM 2018),
 * Coquand-Huber-Mortberg on higher inductive types (CHM 2018) and Cubical
 * Agda (Vezzosi-Mortberg-Abel 2021). Every construction computes: transport
 * along a univalence path applies the equivalence, the circle's eliminator
 * computes on `loop` and on compositions of loops, and closed terms of
 * inductive types normalize to constructors (canonicity).
 *
 * The kernel reads S-expressions. In Eshkol source they appear inside the
 * `(hott decl...)` and `(hott-eval term)` forms; the front end hands the
 * reader datum of those forms to this kernel. The surface language is
 * documented in docs/reference/language/hott.md and docs/guide/HOTT.md.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eshkol::hott::cubical {

/** One S-expression, as the kernel reads it. */
struct Sx {
    enum class Kind : uint8_t { Symbol, Number, String, List };
    Kind kind = Kind::Symbol;
    std::string text;            ///< symbol name, number spelling or string body
    std::vector<Sx> items;       ///< List only
    uint32_t line = 0;
    uint32_t column = 0;

    bool isSymbol() const { return kind == Kind::Symbol; }
    bool isSymbol(const char* name) const { return kind == Kind::Symbol && text == name; }
    bool isList() const { return kind == Kind::List; }
    bool isNumber() const { return kind == Kind::Number; }

    static Sx symbol(std::string name) {
        Sx s; s.kind = Kind::Symbol; s.text = std::move(name); return s;
    }
    static Sx number(std::string spelling) {
        Sx s; s.kind = Kind::Number; s.text = std::move(spelling); return s;
    }
    static Sx list(std::vector<Sx> items) {
        Sx s; s.kind = Kind::List; s.items = std::move(items); return s;
    }
};

/** Read every datum of @p text. Returns false with @p error set on a syntax error. */
bool readSx(const std::string& text, std::vector<Sx>& out, std::string& error);
/** Print @p sx as Eshkol S-expression text. */
std::string printSx(const Sx& sx);

/** Outcome of one kernel request. */
struct KernelResult {
    bool ok = true;
    std::string message;   ///< error text when !ok
    uint32_t line = 0;     ///< position of the offending datum, when known
    uint32_t column = 0;
    Sx value;              ///< evaluate(): the normal form, as a datum
    std::string type;      ///< evaluate(): the type of the term, printed
};

/**
 * One kernel context: the declarations of one compilation (or one REPL
 * session). Declarations accumulate; a failed declaration leaves the context
 * as it was before that declaration.
 */
class Kernel {
public:
    Kernel();
    ~Kernel();
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    /** Check and add the declarations of one `(hott decl...)` form (its items after the head). */
    KernelResult declare(const std::vector<Sx>& decls);
    /** Elaborate a closed term, normalize it and return its normal form as a datum. */
    KernelResult evaluate(const Sx& term);
    /** Names of the HoTT library modules available to `(import name)`. */
    static std::vector<std::string> libraryModules();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eshkol::hott::cubical

#endif  // ESHKOL_TYPES_HOTT_CUBICAL_H
