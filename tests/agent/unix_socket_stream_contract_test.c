/* SPDX-License-Identifier: MIT */
#include "eshkol/core/unix_socket_stream.h"
#include "eshkol/core/event_loop.h"
#include <stdio.h>
#include <string.h>
int eshkol_capability_runtime_allows(const char *name) { return !strcmp(name, "network"); }
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL: socket argument contract line %d\n",__LINE__);return 1;} } while(0)
int main(void) {
    char byte = 0;
    CHECK(eshkol_unix_stream_socket_listen(0,"/tmp/socket") == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_connect_begin(0,"/tmp/socket",0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_accept(0,0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_connect_finish(0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_read(0,&byte,1) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_read(1,&byte,0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_write(1,NULL,1) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_poll(1,4,0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_cancel(0) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_close(0) == ESHKOL_IO_INVALID);
    int64_t loop = eshkol_event_loop_open_handle(8); CHECK(loop > 0);
    CHECK(eshkol_unix_stream_socket_listen(loop,"relative") == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_connect_begin(loop,"relative",1) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_cancel(loop) == ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_close(loop) == ESHKOL_IO_INVALID);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    puts("PASS: socket argument and cross-kind contract"); return 0;
}
