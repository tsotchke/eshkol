/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * The single definition of the object-ABI guard symbol.
 *
 * This file belongs to ESHKOL_RUNTIME_CORE_SRC — the archive that is linked
 * into every generated program — because that is precisely the boundary the
 * guard defends. Compiled Eshkol code and the runtime it calls must agree about
 * where an object's header sits; the guard makes disagreement a link error.
 *
 * See inc/eshkol/abi_fingerprint.h for the mechanism and the reasoning.
 */

#include <stddef.h>   /* size_t, offsetof — eshkol.h expects them already present */
#include <stdint.h>

#include <eshkol/eshkol.h>
#include <eshkol/abi_fingerprint.h>

/*
 * The definition. Its name carries the layout; its value is the header size, so
 * a debugger or a `nm` on a shipped binary answers "which ABI is this?" without
 * needing the sources.
 *
 * There is exactly one of these in the tree. If a second appears — a second
 * runtime archive, a vendored copy — the duplicate-symbol error at link time is
 * the correct outcome, not an inconvenience: two runtimes in one process is
 * already the bug.
 */
const size_t ESHKOL_ABI_FINGERPRINT_SYMBOL = (size_t)ESHKOL_OBJECT_ABI_HEADER_SIZE;

/*
 * Anchor the runtime itself. Without this the runtime would *define* the guard
 * but never *require* it, which is fine for the runtime but leaves the anchor
 * macro itself untested in the build that matters most.
 */
ESHKOL_ABI_FINGERPRINT_ANCHOR;

const char *eshkol_abi_fingerprint_name(void)
{
    /* Stringified from the same macro that spells the identifier above, in the
     * same translation unit, so the string form and the symbol form cannot
     * drift apart. */
    return ESHKOL_ABI_FINGERPRINT_NAME;
}

size_t eshkol_abi_runtime_header_size(void)
{
    /* The ACTIVE header, not the v1 one: under ESHKOL_MEMORY_ABI_V2 this is the
     * 32-byte layout, and reporting v1's size would defeat the check. */
    return ESHKOL_OBJECT_HEADER_SIZE;
}
