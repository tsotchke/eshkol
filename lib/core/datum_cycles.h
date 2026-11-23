#ifndef ESHKOL_CORE_DATUM_CYCLES_H
#define ESHKOL_CORE_DATUM_CYCLES_H

#include "arena_memory.h"
#include <eshkol/core/runtime.h>
#include <unordered_map>
#include <vector>

namespace eshkol {

/** Test the car/cdr graph before eval traverses it as syntax. A pair reached
 * on the active path is cyclic. A finished pair may be shared by many forms. */
inline bool datum_has_cycle(eshkol_tagged_value_t datum) {
    struct Visit {
        const arena_tagged_cons_cell_t* pair;
        bool leaving;
    };
    enum class State { active, finished };
    std::unordered_map<const arena_tagged_cons_cell_t*, State> states;
    std::vector<Visit> pending;
    auto enter = [&](eshkol_tagged_value_t value) {
        if (ESHKOL_IS_CONS_COMPAT(value) && value.data.ptr_val) {
            pending.push_back({reinterpret_cast<const arena_tagged_cons_cell_t*>(
                                   value.data.ptr_val), false});
        }
    };
    enter(datum);
    while (!pending.empty()) {
        auto visit = pending.back();
        pending.pop_back();
        if (visit.leaving) {
            states.at(visit.pair) = State::finished;
            continue;
        }
        auto [entry, inserted] = states.emplace(visit.pair, State::active);
        if (!inserted) {
            if (entry->second == State::active) return true;
            continue;
        }
        pending.push_back({visit.pair, true});
        enter(visit.pair->cdr);
        enter(visit.pair->car);
    }
    return false;
}

/** Raise only after the graph walk has released its scratch storage. */
inline void validate_eval_datum(eshkol_tagged_value_t datum) {
    if (datum_has_cycle(datum))
        eshkol_type_error("eval (cyclic datum)", "an acyclic expression");
}

}
#endif
