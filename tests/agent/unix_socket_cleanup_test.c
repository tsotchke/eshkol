/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include "eshkol/core/unix_socket_stream.h"
#include "eshkol/core/event_loop.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL cleanup line %d: %s errno=%d\n",__LINE__,#x,errno);exit(1);} }while(0)
int eshkol_capability_runtime_allows(const char *name) {return !strcmp(name,"network");}
static int64_t worker_id, worker_ticket;
static void *request_close(void *unused) {
    (void)unused; worker_ticket=eshkol_unix_stream_socket_close_request(worker_id);
    CHECK(worker_ticket>0);
    eshkol_resource_cleanup_result_t r;
    CHECK(eshkol_resource_cleanup_result(worker_ticket,&r)==0);
    CHECK(r.state==ESHKOL_CLEANUP_PENDING);
    return NULL;
}
int main(void) {
    char directory[]="/tmp/eshkol-cleanup-XXXXXX",path[108];
    CHECK(mkdtemp(directory));snprintf(path,sizeof(path),"%s/s.sock",directory);
    int64_t loop=eshkol_event_loop_open_handle(16);CHECK(loop>0);
    int64_t listener=eshkol_unix_stream_socket_listen(loop,path);CHECK(listener>0);
    eshkol_resource_lease_t lease;
    CHECK(eshkol_resource_borrow(listener,ESHKOL_HANDLE_KIND_SOCKET,0,&lease)==0);
    int64_t ticket=eshkol_unix_stream_socket_close_request(listener);CHECK(ticket>0 && ticket!=listener);
    CHECK(eshkol_unix_stream_socket_accept(listener,0)==ESHKOL_IO_INVALID);
    eshkol_resource_cleanup_result_t result;
    CHECK(eshkol_resource_cleanup_result(ticket,&result)==0 && result.state==ESHKOL_CLEANUP_PENDING);
    CHECK(eshkol_event_loop_close_handle(loop)==ESHKOL_EVENT_LOOP_EBUSY);
    eshkol_resource_release(&lease);
    CHECK(eshkol_resource_cleanup_result(ticket,&result)==0 && result.state==ESHKOL_CLEANUP_COMPLETE);
    CHECK(access(path,F_OK)<0 && errno==ENOENT);
    CHECK(eshkol_resource_cleanup_release(ticket)==0);
    CHECK(eshkol_resource_cleanup_result(ticket,&result)==ESHKOL_IO_INVALID);
    worker_id=eshkol_unix_stream_socket_listen(loop,path);CHECK(worker_id>0);
    pthread_t worker;CHECK(pthread_create(&worker,NULL,request_close,NULL)==0);
    CHECK(pthread_join(worker,NULL)==0);
    struct timespec before,after; CHECK(clock_gettime(CLOCK_MONOTONIC,&before)==0);
    const eshkol_event_t *events; int count;
    CHECK(eshkol_event_loop_poll(eshkol_event_loop_from_handle(loop),1000,&events,&count)==0);
    CHECK(clock_gettime(CLOCK_MONOTONIC,&after)==0);
    CHECK((after.tv_sec-before.tv_sec)*1000+(after.tv_nsec-before.tv_nsec)/1000000 < 500);
    CHECK(eshkol_resource_cleanup_result(worker_ticket,&result)==0 && result.state==ESHKOL_CLEANUP_COMPLETE);
    CHECK(access(path,F_OK)<0 && errno==ENOENT);
    CHECK(eshkol_resource_cleanup_release(worker_ticket)==0);
    listener=eshkol_unix_stream_socket_listen(loop,path);CHECK(listener>0);
    CHECK(chmod(directory,0500)==0);
    ticket=eshkol_unix_stream_socket_close_request(listener);CHECK(ticket>0);
    CHECK(eshkol_resource_cleanup_result(ticket,&result)==0);
    CHECK(result.state==ESHKOL_CLEANUP_FAILED && result.stage==ESHKOL_CLEANUP_PATH && result.retryable);
    CHECK(result.os_error==EACCES || result.os_error==EPERM);
    CHECK(eshkol_event_loop_close_handle(loop)==ESHKOL_EVENT_LOOP_EBUSY);
    int reused=open("/dev/null",O_RDONLY);CHECK(reused>=0);
    CHECK(chmod(directory,0700)==0);
    CHECK(eshkol_resource_cleanup_retry(ticket)==0);
    CHECK(eshkol_resource_cleanup_result(ticket,&result)==0 && result.state==ESHKOL_CLEANUP_COMPLETE);
    CHECK(fcntl(reused,F_GETFD)>=0);CHECK(close(reused)==0);
    CHECK(access(path,F_OK)<0 && errno==ENOENT);
    CHECK(eshkol_resource_cleanup_release(ticket)==0);
    CHECK(eshkol_event_loop_close_handle(loop)==0);CHECK(rmdir(directory)==0);
    puts("PASS: socket borrowed/queued close and retryable path cleanup; descriptor reused safely");return 0;
}
