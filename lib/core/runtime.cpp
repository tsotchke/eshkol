/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime interrupt flag definition.
 */

#include <eshkol/core/runtime.h>

volatile sig_atomic_t g_eshkol_interrupt_flag = 0;
