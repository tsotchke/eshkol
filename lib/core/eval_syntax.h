#ifndef ESHKOL_CORE_EVAL_SYNTAX_H
#define ESHKOL_CORE_EVAL_SYNTAX_H

#include <eshkol/eshkol.h>
#include <set>
#include <string>

namespace eshkol {

// The hosted JIT supplies the macro keywords of its own evaluation context.
using EvalMacroNames = std::set<std::string> (*)(void* jit);
void register_eval_macro_names(EvalMacroNames provider);
// Why a datum did not become an expression: the condition type to raise and
// its message. Plain storage, so a caller can raise after every C++ object of
// the conversion has been released.
struct EvalConversionError {
    eshkol_exception_type_t type = ESHKOL_EXCEPTION_ERROR;
    char message[384] = {0};
};

// Convert a runtime datum to an expression tree, exactly as the parser reads
// the same datum written as text, with the macro keywords of @p jit's context
// (none when @p jit is null). Every eval, eval-with-environment and compile
// entry point uses this one conversion. Nesting depth has no limit; the datum
// is weighed against the one input-size cap. A cyclic datum raises its
// condition here; any other failure returns null with @p error filled.
eshkol_ast_t* eval_datum_to_ast(eshkol_tagged_value_t datum, void* jit,
                                EvalConversionError& error);

}
#endif
