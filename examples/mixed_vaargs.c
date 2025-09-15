/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <stdio.h>
#include <stdarg.h>

const char *example_string = "Goodbye World!";

int example_int = 20;

void example_printf(const char *msg, ...)
{
  va_list ap;

  va_start(ap, msg);
  vprintf(msg, ap);
  va_end(ap);
}
