/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Kernel-level driver for the HoTT corpus (ADR-0043). Reads an Eshkol source
 * file, sends every top-level `(hott decl...)` form to one cubical kernel and
 * prints the normal form of every top-level `(display (hott-eval t))` /
 * `(hott-eval t)`, one per line, exactly as the compiled program prints them.
 *
 *   hott_kernel_driver FILE            exit 0 when every form is accepted
 *   hott_kernel_driver --reject FILE   exit 0 when some form is rejected; the
 *                                      diagnostic must contain the text of the
 *                                      file's `;; expect-error: ...` line
 */

#include "eshkol/types/hott_cubical.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace eshkol::hott::cubical;

static void evalForms(Kernel& k, const Sx& f, bool& ok, std::string& diag) {
    if (!ok || !f.isList() || f.items.empty()) return;
    if (f.items[0].isSymbol("hott")) {
        std::vector<Sx> decls(f.items.begin() + 1, f.items.end());
        KernelResult r = k.declare(decls);
        if (!r.ok) {
            ok = false;
            diag = std::to_string(r.line) + ":" + std::to_string(r.column) + ": " + r.message;
        }
        return;
    }
    if (f.items[0].isSymbol("hott-eval") && f.items.size() == 2) {
        KernelResult r = k.evaluate(f.items[1]);
        if (!r.ok) {
            ok = false;
            diag = std::to_string(r.line) + ":" + std::to_string(r.column) + ": " + r.message;
            return;
        }
        std::cout << printSx(r.value) << "\n";
        return;
    }
    for (auto& item : f.items) evalForms(k, item, ok, diag);
}

int main(int argc, char** argv) {
    bool reject = false;
    int a = 1;
    if (a < argc && std::string(argv[a]) == "--reject") { reject = true; ++a; }
    if (a >= argc) {
        std::cerr << "usage: hott_kernel_driver [--reject] FILE\n";
        return 2;
    }
    std::ifstream in(argv[a]);
    if (!in) { std::cerr << "cannot read " << argv[a] << "\n"; return 2; }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    std::string expect;
    {
        std::istringstream lines(text);
        std::string line;
        const std::string tag = ";; expect-error:";
        while (std::getline(lines, line)) {
            auto p = line.find(tag);
            if (p != std::string::npos) {
                expect = line.substr(p + tag.size());
                while (!expect.empty() && expect.front() == ' ') expect.erase(expect.begin());
            }
        }
    }
    std::vector<Sx> forms;
    std::string err;
    if (!readSx(text, forms, err)) {
        std::cerr << argv[a] << ": " << err << "\n";
        return reject ? 0 : 1;
    }
    Kernel k;
    bool ok = true;
    std::string diag;
    for (auto& f : forms) evalForms(k, f, ok, diag);
    if (reject) {
        if (ok) { std::cerr << argv[a] << ": expected a rejection, but every form was accepted\n"; return 1; }
        std::cerr << argv[a] << ": rejected as expected: " << diag << "\n";
        if (!expect.empty() && diag.find(expect) == std::string::npos) {
            std::cerr << argv[a] << ": diagnostic does not contain `" << expect << "`\n";
            return 1;
        }
        return 0;
    }
    if (!ok) { std::cerr << argv[a] << ":" << diag << "\n"; return 1; }
    return 0;
}
