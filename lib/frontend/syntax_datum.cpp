/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Side tables that keep reader syntax for macro uses and syntax-rules
 * transformers (ADR-0026). See inc/eshkol/frontend/syntax_datum.h.
 */
#include <eshkol/frontend/syntax_datum.h>

#include <map>
#include <mutex>
#include <unordered_map>

namespace eshkol {
namespace {

std::mutex& table_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct UseEntry {
    std::shared_ptr<const void> tape;
    uint32_t start = 0;
    std::shared_ptr<const SyntaxDatum> datum;
    std::vector<std::shared_ptr<const std::map<std::string, std::string>>> renames;
    bool unparsed = false;
};

std::unordered_map<eshkol_node_id_t, UseEntry>& use_table() {
    static auto* table = new std::unordered_map<eshkol_node_id_t, UseEntry>();
    return *table;
}

std::map<const eshkol_macro_def_t*, std::shared_ptr<const MacroSyntax>>& macro_table() {
    static auto* table =
        new std::map<const eshkol_macro_def_t*, std::shared_ptr<const MacroSyntax>>();
    return *table;
}

// mode: 0 = code, 1 = quoted data, N >= 2 = quasiquote nesting N - 1.
void rename_in(SyntaxDatum& datum, const std::function<void(std::string&)>& rename,
               unsigned mode) {
    using Kind = SyntaxDatum::Kind;
    auto enter = [&](const std::string& marker, unsigned current) -> unsigned {
        if (current == 1) return 1;
        if (marker == "'" || marker == "quote") return current == 0 ? 1 : current;
        if (marker == "`" || marker == "quasiquote") return current == 0 ? 2 : current + 1;
        if (marker == "," || marker == ",@" || marker == "unquote" ||
            marker == "unquote-splicing")
            return current >= 2 ? (current == 2 ? 0 : current - 1) : current;
        return current;
    };
    switch (datum.kind) {
        case Kind::Symbol:
            if (mode == 0) rename(datum.text);
            return;
        case Kind::Atom:
            return;
        case Kind::Prefix: {
            const unsigned inner = enter(datum.text, mode);
            for (auto& item : datum.items) rename_in(item, rename, inner);
            return;
        }
        case Kind::Vector:
            for (auto& item : datum.items) rename_in(item, rename, mode);
            return;
        case Kind::List: {
            unsigned inner = mode;
            if (!datum.items.empty() && datum.items[0].isSymbol())
                inner = enter(datum.items[0].text, mode);
            for (size_t i = 0; i < datum.items.size(); ++i)
                rename_in(datum.items[i], rename, i == 0 ? mode : inner);
            return;
        }
    }
}

} // namespace

void syntax_record_use(eshkol_node_id_t id, SyntaxDatum use) {
    auto stored = std::make_shared<const SyntaxDatum>(std::move(use));
    std::lock_guard<std::mutex> lock(table_mutex());
    UseEntry entry;
    entry.datum = std::move(stored);
    entry.unparsed = true;
    use_table()[id] = std::move(entry);
}

bool syntax_use_unparsed(eshkol_node_id_t id) {
    std::lock_guard<std::mutex> lock(table_mutex());
    auto found = use_table().find(id);
    return found != use_table().end() && found->second.unparsed;
}

void syntax_record_use_tape(eshkol_node_id_t id, std::shared_ptr<const void> tape,
                            uint32_t start) {
    std::lock_guard<std::mutex> lock(table_mutex());
    UseEntry entry;
    entry.tape = std::move(tape);
    entry.start = start;
    use_table()[id] = std::move(entry);
}

std::shared_ptr<const SyntaxDatum> syntax_use(eshkol_node_id_t id) {
    UseEntry entry;
    {
        std::lock_guard<std::mutex> lock(table_mutex());
        auto found = use_table().find(id);
        if (found == use_table().end()) return nullptr;
        if (found->second.datum && found->second.renames.empty()) return found->second.datum;
        entry = found->second;
    }
    SyntaxDatum datum = entry.datum ? *entry.datum : syntax_read_tape(entry.tape.get(), entry.start);
    for (const auto& names : entry.renames) {
        syntax_rename_identifiers(datum, [&](std::string& name) {
            auto renamed = names->find(name);
            if (renamed != names->end()) name = renamed->second;
        });
    }
    auto stored = std::make_shared<const SyntaxDatum>(std::move(datum));
    std::lock_guard<std::mutex> lock(table_mutex());
    auto& slot = use_table()[id];
    slot.datum = stored;
    slot.renames.clear();
    slot.tape.reset();
    return stored;
}

void syntax_add_use_renames(eshkol_node_id_t id,
                            std::shared_ptr<const std::map<std::string, std::string>> names) {
    if (!names || names->empty()) return;
    std::lock_guard<std::mutex> lock(table_mutex());
    auto found = use_table().find(id);
    if (found != use_table().end()) found->second.renames.push_back(std::move(names));
}

void syntax_record_macro(const eshkol_macro_def_t* macro, MacroSyntax syntax) {
    auto stored = std::make_shared<const MacroSyntax>(std::move(syntax));
    std::lock_guard<std::mutex> lock(table_mutex());
    macro_table()[macro] = std::move(stored);
}

std::shared_ptr<const MacroSyntax> syntax_macro(const eshkol_macro_def_t* macro) {
    std::lock_guard<std::mutex> lock(table_mutex());
    auto found = macro_table().find(macro);
    return found == macro_table().end() ? nullptr : found->second;
}

void syntax_replace_macro(const eshkol_macro_def_t* macro, MacroSyntax syntax) {
    syntax_record_macro(macro, std::move(syntax));
}

void syntax_rename_identifiers(SyntaxDatum& datum,
                               const std::function<void(std::string&)>& rename) {
    rename_in(datum, rename, 0);
}

} // namespace eshkol
