/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Compatibility translation unit for embedders that still compile
 * lib/core/resource_limits.cpp directly.
 *
 * CMake targets compile the split owners instead:
 * - resource_limits_core.cpp for allocation, stack, timer, and validation state
 * - resource_limits_hosted.cpp for environment-driven hosted configuration
 *
 * Keep this wrapper out of CMake's runtime source sets to avoid duplicate
 * definitions while preserving the older direct-source integration path.
 */

#include "resource_limits_core.cpp"

#if !defined(__STDC_HOSTED__) || (__STDC_HOSTED__ == 1)
#include "resource_limits_hosted.cpp"
#endif
