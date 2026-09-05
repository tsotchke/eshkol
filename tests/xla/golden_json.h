/*
 * A minimal JSON reader, for reading tests/qllm_oracle/golden/*.json at run
 * time.
 *
 * WHY READ THE FILES RATHER THAN TRANSCRIBE THEM.
 *
 * tests/xla/gradient_parity_test.cpp's sphere_project golden row transcribes
 * two cases from sphere_project.json into a C++ literal. That was defensible
 * for two cases and it does not scale to the nine files and roughly eighty
 * cases the geometric sweep grades against: a transcription is a second copy
 * of the reference, it can be typed wrong, and it does not change when the
 * corpus is regenerated. The corpus README states that regenerating and then
 * diffing golden/ IS the drift check — a check a transcription defeats.
 *
 * So the harness parses the committed JSON. That also lets a row CITE its file
 * and case id, which is what makes "graded against the golden corpus" a
 * verifiable claim rather than an assertion.
 *
 * SCOPE. This reads the subset of JSON the corpus uses: objects, arrays,
 * strings without escapes beyond the standard two-character ones, numbers in
 * the form printf("%.17g") produces, true/false, and null. `null` is
 * meaningful in the corpus — it is how a non-finite gradient entry is written,
 * since JSON has no NaN literal — so it is represented, not skipped.
 *
 * It is deliberately not a general JSON library: a parser that silently
 * accepted malformed input would turn a corrupt corpus into a passing row.
 * Every parse failure carries the byte offset.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_TESTS_XLA_GOLDEN_JSON_H
#define ESHKOL_TESTS_XLA_GOLDEN_JSON_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace eshkol_golden {

struct Json;
using JsonPtr = std::shared_ptr<Json>;

struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<JsonPtr> items;
    std::map<std::string, JsonPtr> fields;

    bool isNull() const { return kind == Kind::Null; }

    /** @brief Member of an object, or nullptr. Never throws. */
    const Json* get(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        auto it = fields.find(key);
        return it == fields.end() ? nullptr : it->second.get();
    }

    /** @brief Element of an array, or nullptr. */
    const Json* at(size_t i) const {
        if (kind != Kind::Array || i >= items.size()) return nullptr;
        return items[i].get();
    }

    size_t size() const { return kind == Kind::Array ? items.size() : 0; }

    /** @brief Numeric value, or @p fallback for anything that is not a number. */
    double num(double fallback = 0.0) const {
        return kind == Kind::Number ? number : fallback;
    }

    /**
     * @brief This array as a vector of doubles; a `null` entry becomes NaN.
     *
     * NaN is the right image of `null` here: the corpus writes null exactly
     * where the mathematically correct gradient entry is non-finite, and the
     * comparator in tests/xla/parity_compare.h already treats NaN against NaN
     * as agreement and NaN against a number as disagreement.
     */
    std::vector<double> doubles() const {
        std::vector<double> out;
        if (kind != Kind::Array) return out;
        out.reserve(items.size());
        for (const auto& v : items) {
            if (!v) { out.push_back(NAN); continue; }
            if (v->kind == Kind::Number) out.push_back(v->number);
            else out.push_back(NAN);
        }
        return out;
    }

    /** @brief This array-of-arrays as a row-major matrix, flattened per row. */
    std::vector<std::vector<double>> matrix() const {
        std::vector<std::vector<double>> out;
        if (kind != Kind::Array) return out;
        for (const auto& row : items) {
            if (!row) { out.push_back({}); continue; }
            out.push_back(row->doubles());
        }
        return out;
    }
};

namespace detail {

struct Parser {
    const std::string& s;
    size_t i = 0;
    std::string error;

    explicit Parser(const std::string& text) : s(text) {}

    void skip() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }

    bool fail(const char* what) {
        if (error.empty()) {
            error = std::string(what) + " at byte " + std::to_string(i);
        }
        return false;
    }

    bool literal(const char* lit) {
        const size_t n = std::string(lit).size();
        if (s.compare(i, n, lit) != 0) return false;
        i += n;
        return true;
    }

    JsonPtr parseValue() {
        skip();
        if (i >= s.size()) { fail("unexpected end of input"); return nullptr; }
        const char c = s[i];
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') {
            auto j = std::make_shared<Json>();
            j->kind = Json::Kind::String;
            if (!parseString(&j->text)) return nullptr;
            return j;
        }
        if (literal("true")) {
            auto j = std::make_shared<Json>();
            j->kind = Json::Kind::Bool;
            j->boolean = true;
            return j;
        }
        if (literal("false")) {
            auto j = std::make_shared<Json>();
            j->kind = Json::Kind::Bool;
            j->boolean = false;
            return j;
        }
        if (literal("null")) {
            auto j = std::make_shared<Json>();
            j->kind = Json::Kind::Null;
            return j;
        }
        return parseNumber();
    }

    bool parseString(std::string* out) {
        if (i >= s.size() || s[i] != '"') { fail("expected a string"); return false; }
        ++i;
        out->clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                if (i >= s.size()) { fail("unterminated escape"); return false; }
                switch (s[i]) {
                    case 'n': out->push_back('\n'); break;
                    case 't': out->push_back('\t'); break;
                    case 'r': out->push_back('\r'); break;
                    case 'b': out->push_back('\b'); break;
                    case 'f': out->push_back('\f'); break;
                    case '/': out->push_back('/'); break;
                    case '\\': out->push_back('\\'); break;
                    case '"': out->push_back('"'); break;
                    case 'u':
                        // The corpus is ASCII. Refuse rather than mangle.
                        fail("\\u escapes are not supported by this reader");
                        return false;
                    default:
                        fail("unknown escape");
                        return false;
                }
                ++i;
                continue;
            }
            out->push_back(s[i++]);
        }
        if (i >= s.size()) { fail("unterminated string"); return false; }
        ++i;  // closing quote
        return true;
    }

    JsonPtr parseNumber() {
        const size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' ||
                                s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) {
            ++i;
        }
        if (i == start) { fail("expected a value"); return nullptr; }
        auto j = std::make_shared<Json>();
        j->kind = Json::Kind::Number;
        j->number = std::strtod(s.substr(start, i - start).c_str(), nullptr);
        return j;
    }

    JsonPtr parseArray() {
        auto j = std::make_shared<Json>();
        j->kind = Json::Kind::Array;
        ++i;  // '['
        skip();
        if (i < s.size() && s[i] == ']') { ++i; return j; }
        for (;;) {
            JsonPtr v = parseValue();
            if (!v) return nullptr;
            j->items.push_back(v);
            skip();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == ']') { ++i; return j; }
            fail("expected ',' or ']'");
            return nullptr;
        }
    }

    JsonPtr parseObject() {
        auto j = std::make_shared<Json>();
        j->kind = Json::Kind::Object;
        ++i;  // '{'
        skip();
        if (i < s.size() && s[i] == '}') { ++i; return j; }
        for (;;) {
            skip();
            std::string key;
            if (!parseString(&key)) return nullptr;
            skip();
            if (i >= s.size() || s[i] != ':') { fail("expected ':'"); return nullptr; }
            ++i;
            JsonPtr v = parseValue();
            if (!v) return nullptr;
            j->fields[key] = v;
            skip();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == '}') { ++i; return j; }
            fail("expected ',' or '}'");
            return nullptr;
        }
    }
};

}  // namespace detail

/** @brief Parse @p text. Returns nullptr and sets @p error on any malformation. */
inline JsonPtr parse(const std::string& text, std::string* error) {
    detail::Parser p(text);
    JsonPtr v = p.parseValue();
    if (!v) { if (error) *error = p.error; return nullptr; }
    p.skip();
    if (p.i != text.size()) {
        if (error) *error = "trailing content at byte " + std::to_string(p.i);
        return nullptr;
    }
    return v;
}

/** @brief Read and parse a file. Returns nullptr and sets @p error on failure. */
inline JsonPtr parseFile(const std::string& path, std::string* error) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (error) *error = "could not open " + path;
        return nullptr;
    }
    std::string text;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    JsonPtr v = parse(text, error);
    if (!v && error) *error = path + ": " + *error;
    return v;
}

}  // namespace eshkol_golden

#endif  // ESHKOL_TESTS_XLA_GOLDEN_JSON_H
