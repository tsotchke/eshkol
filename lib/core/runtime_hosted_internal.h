/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Private hosted runtime coordination helpers.
 */
#ifndef ESHKOL_CORE_RUNTIME_HOSTED_INTERNAL_H
#define ESHKOL_CORE_RUNTIME_HOSTED_INTERNAL_H

#include <eshkol/core/runtime.h>

namespace eshkol::runtime_hosted {

void run_shutdown_hooks(eshkol_shutdown_reason_t reason);

}  // namespace eshkol::runtime_hosted

#endif  // ESHKOL_CORE_RUNTIME_HOSTED_INTERNAL_H
