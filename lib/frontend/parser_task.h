/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_FRONTEND_PARSER_TASK_H
#define ESHKOL_FRONTEND_PARSER_TASK_H
#include <eshkol/util/continuation_task.h>
template<class T> using ParserTask = ContinuationTask<T>;
#endif
