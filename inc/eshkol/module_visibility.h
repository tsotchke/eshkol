#ifndef ESHKOL_MODULE_VISIBILITY_H
#define ESHKOL_MODULE_VISIBILITY_H

#include <eshkol/eshkol.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace eshkol {

void rename_private_symbols(std::vector<eshkol_ast_t>& asts,
                            const std::string& module_name,
                            const std::set<std::string>& exports);

}  // namespace eshkol

#endif
