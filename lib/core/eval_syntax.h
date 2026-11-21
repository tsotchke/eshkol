#ifndef ESHKOL_CORE_EVAL_SYNTAX_H
#define ESHKOL_CORE_EVAL_SYNTAX_H

#include <eshkol/eshkol.h>
#include <set>
#include <string>

namespace eshkol {

// The hosted JIT supplies the macro keywords of its own evaluation context.
using EvalMacroNames = std::set<std::string> (*)(void* jit);
void register_eval_macro_names(EvalMacroNames provider);
eshkol_ast_t* eval_datum_to_ast(eshkol_tagged_value_t datum, void* jit);

}
#endif
