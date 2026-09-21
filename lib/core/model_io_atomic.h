#ifndef ESHKOL_MODEL_IO_ATOMIC_H
#define ESHKOL_MODEL_IO_ATOMIC_H

#include <stdio.h>

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <signal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FILE* stream;
    char* destination_path;
    char* temporary_path;
    unsigned long write_calls;
#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
    sigset_t previous_signal_mask;
    int signals_blocked;
#endif
} eshkol_atomic_checkpoint_file_t;

int eshkol_atomic_checkpoint_begin(eshkol_atomic_checkpoint_file_t* file,
                                   const char* destination_path);
size_t eshkol_atomic_checkpoint_write(eshkol_atomic_checkpoint_file_t* file,
                                      const void* data,
                                      size_t size);
int eshkol_atomic_checkpoint_commit(eshkol_atomic_checkpoint_file_t* file);
void eshkol_atomic_checkpoint_abort(eshkol_atomic_checkpoint_file_t* file);

#ifdef __cplusplus
}
#endif

#endif
