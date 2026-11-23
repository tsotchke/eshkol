/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_VM_SOURCE_SYNTAX_H
#define ESHKOL_VM_SOURCE_SYNTAX_H

namespace eshkol::frontend {

// Select bytecode source syntax throughout parsing and macro expansion.
class VmSourceSyntaxScope {
public:
    VmSourceSyntaxScope() noexcept;
    ~VmSourceSyntaxScope() noexcept;
    VmSourceSyntaxScope(const VmSourceSyntaxScope&) = delete;
    VmSourceSyntaxScope& operator=(const VmSourceSyntaxScope&) = delete;
};

}  // namespace eshkol::frontend
#endif
