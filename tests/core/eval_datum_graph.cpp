#include "../../lib/core/datum_cycles.h"
#include <csetjmp>
#include <cstring>

static eshkol_tagged_value_t pair(arena_tagged_cons_cell_t& cell) {
    eshkol_tagged_value_t value{};
    value.type = ESHKOL_VALUE_CONS_PTR;
    value.data.ptr_val = reinterpret_cast<uintptr_t>(&cell);
    return value;
}

static bool raises_cycle_condition(eshkol_tagged_value_t datum) {
    std::jmp_buf landing;
    eshkol_push_exception_handler(&landing);
    if (setjmp(landing) == 0) {
        eshkol::validate_eval_datum(datum);
        eshkol_pop_exception_handler();
        return false;
    }
    eshkol_pop_exception_handler();
    return g_current_exception && g_current_exception->type == ESHKOL_EXCEPTION_TYPE_ERROR &&
           g_current_exception->message && std::strstr(g_current_exception->message, "cyclic datum");
}

int main() {
    arena_tagged_cons_cell_t top{}, child{};
    top.car = top.cdr = pair(child);
    eshkol::validate_eval_datum(pair(top));
    child.car = pair(top);
    if (!raises_cycle_condition(pair(top))) return 2;
    child.car = {};
    child.cdr = pair(child);
    return raises_cycle_condition(pair(top)) ? 0 : 3;
}
