/* SPDX-License-Identifier: MIT */
/* Host-compiled source-branch contract, not Windows/VM/WASM execution. */
#include <eshkol/core/unix_socket_stream.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL unsupported socket:%d: %s\n",__LINE__,#x);exit(1); } } while (0)
static unsigned cleanup_calls;
static int refuse_cleanup(int64_t id,eshkol_handle_kind_t kind,uint32_t mask) {
    CHECK(id==123 && kind==ESHKOL_HANDLE_KIND_SOCKET && mask!=0);
    ++cleanup_calls;return ESHKOL_IO_UNSUPPORTED;
}
/* The unsupported branch delegates these owned-resource requests to the generic
 * layer. These link controls do not supply a socket implementation. */
int eshkol_resource_cancel(int64_t id,eshkol_handle_kind_t kind,uint32_t mask) {
    return refuse_cleanup(id,kind,mask);
}
int eshkol_resource_close(int64_t id,eshkol_handle_kind_t kind,uint32_t mask) {
    return refuse_cleanup(id,kind,mask);
}
int64_t eshkol_resource_close_request(int64_t id,eshkol_handle_kind_t kind,uint32_t mask) {
    return refuse_cleanup(id,kind,mask);
}
int main(void) {
    char byte=0;
    CHECK(eshkol_unix_stream_socket_listen(123,"/unused/stream.sock")==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_connect_begin(123,"/unused/stream.sock",10)==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_connect_finish(123)==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_accept(123,10)==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_read(123,&byte,1)==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_write(123,&byte,1)==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_poll(123,1,10)==ESHKOL_IO_UNSUPPORTED);
    CHECK(cleanup_calls==0);
    CHECK(eshkol_unix_stream_socket_cancel(123)==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_close(123)==ESHKOL_IO_UNSUPPORTED);
    CHECK(eshkol_unix_stream_socket_close_request(123)==ESHKOL_IO_UNSUPPORTED);
    CHECK(cleanup_calls==3);
    CHECK(eshkol_unix_stream_socket_last_os_error()==0);
    CHECK(eshkol_unix_stream_socket_last_path_error()==0);
    puts("PASS: HOST source-branch socket refusal; not target-platform execution");
    return 0;
}
