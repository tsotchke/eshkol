/**
 * @file memory_abi_v2.cpp
 * @brief Runtime side of the OALR object/layout ABI version report (ADR-0001 §3).
 *
 * Phase A carries exactly one runtime symbol: the ABI version the runtime
 * archive was compiled with. A generated program, a JIT module, the
 * precompiled stdlib and this runtime must all agree on the object header
 * layout — a mismatch is a miscompile that produces garbage rather than a
 * link error, because the header prefix is invisible in every symbol
 * signature. Making the runtime's own answer observable is the precondition
 * for Phase B turning a mismatch into a hard startup failure.
 *
 * See docs/design/ABI_V2_MIGRATION_INVENTORY.md.
 */
#include "eshkol/eshkol.h"

extern "C" uint32_t eshkol_memory_abi_active(void) {
    return (uint32_t)ESHKOL_MEMORY_ABI_ACTIVE;
}
