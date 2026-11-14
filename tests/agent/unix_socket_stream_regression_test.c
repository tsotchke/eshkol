/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE 1
#include "eshkol/core/unix_socket_stream.h"
#include "eshkol/core/event_loop.h"
#include "unix_socket_test_directory.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL socket regression:%d: %s errno=%d\n",__LINE__,#x,errno);exit(1); } } while (0)
int eshkol_capability_runtime_allows(const char *name) { return !strcmp(name,"network"); }
/* Linker controls exercise interruption and failed unpublished cleanup while
 * keeping all descriptor creation, real I/O and exhaustion in the OS. */
static int interrupt_read, interrupt_write, fail_mode, fail_unlink;
static int atomic_sockets, atomic_accepts, connect_calls;
static int fail_close_once, failed_close_fd = -1, failed_close_calls;
int __real_close(int);
int __wrap_close(int fd) {
    if (fd == failed_close_fd) ++failed_close_calls;
    if (fail_close_once) {
        fail_close_once = 0;
        failed_close_fd = fd; failed_close_calls = 1;
        /* Model an uncertain close: the OS descriptor is gone even though
         * the caller receives an error, as may happen on an interrupted close. */
        CHECK(__real_close(fd) == 0);
        errno = EIO; return -1;
    }
    return __real_close(fd);
}
int __real_connect(int,const struct sockaddr*,socklen_t);
int __wrap_connect(int fd,const struct sockaddr *address,socklen_t size) {
    ++connect_calls; return __real_connect(fd,address,size);
}
int __real_socket(int,int,int);
int __wrap_socket(int domain,int type,int protocol) {
    if(domain==AF_UNIX) {
        CHECK((type&(SOCK_CLOEXEC|SOCK_NONBLOCK))==(SOCK_CLOEXEC|SOCK_NONBLOCK));
        ++atomic_sockets;
    }
    return __real_socket(domain,type,protocol);
}
int __real_accept4(int,struct sockaddr*,socklen_t*,int);
int __wrap_accept4(int fd,struct sockaddr *address,socklen_t *length,int flags) {
    CHECK((flags&(SOCK_CLOEXEC|SOCK_NONBLOCK))==(SOCK_CLOEXEC|SOCK_NONBLOCK));
    ++atomic_accepts; return __real_accept4(fd,address,length,flags);
}
ssize_t __real_recv(int,void*,size_t,int);
ssize_t __wrap_recv(int fd,void *buffer,size_t size,int flags) {
    if(interrupt_read) { interrupt_read=0;errno=EINTR;return -1; }
    return __real_recv(fd,buffer,size,flags);
}
ssize_t __real_send(int,const void*,size_t,int);
ssize_t __wrap_send(int fd,const void *buffer,size_t size,int flags) {
    if(interrupt_write) { interrupt_write=0;errno=EINTR;return -1; }
    return __real_send(fd,buffer,size,flags);
}
int __real_fchmodat(int,const char*,mode_t,int);
int __wrap_fchmodat(int fd,const char *path,mode_t mode,int flags) {
    if(fail_mode) { errno=EIO;return -1; }
    return __real_fchmodat(fd,path,mode,flags);
}
int __real_unlinkat(int,const char*,int);
int __wrap_unlinkat(int fd,const char *path,int flags) {
    if(fail_unlink) { errno=EACCES;return -1; }
    return __real_unlinkat(fd,path,flags);
}
static void combined_cleanup_failure(const char *path, int repair) {
    int64_t loop = eshkol_event_loop_open_handle(8); CHECK(loop > 0);
    int64_t listener = eshkol_unix_stream_socket_listen(loop, path); CHECK(listener > 0);
    fail_close_once = fail_unlink = 1;
    int64_t ticket = eshkol_unix_stream_socket_close_request(listener); CHECK(ticket > 0);
    eshkol_resource_cleanup_result_t result;
    CHECK(eshkol_resource_cleanup_result(ticket, &result) == 0);
    CHECK(result.state == ESHKOL_CLEANUP_FAILED &&
          result.stage == ESHKOL_CLEANUP_DESCRIPTOR && result.os_error == EIO && result.retryable);
    CHECK(failed_close_calls == 1);
    CHECK(eshkol_resource_cleanup_release(ticket) == ESHKOL_IO_BUSY);
    CHECK(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_EBUSY);
    int reused = open("/dev/null", O_RDONLY | O_CLOEXEC);
    CHECK(reused >= 0 && reused == failed_close_fd);
    if (repair) fail_unlink = 0;
    CHECK(eshkol_resource_cleanup_retry(ticket) == 0);
    CHECK(eshkol_resource_cleanup_result(ticket, &result) == 0);
    CHECK(result.state == ESHKOL_CLEANUP_FAILED &&
          result.stage == ESHKOL_CLEANUP_DESCRIPTOR && result.os_error == EIO && !result.retryable);
    CHECK(failed_close_calls == 1 && fcntl(reused, F_GETFD) >= 0);
    CHECK(eshkol_resource_cleanup_retry(ticket) == ESHKOL_IO_INVALID);
    CHECK(eshkol_resource_cleanup_release(ticket) == 0);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    CHECK(fcntl(reused, F_GETFD) >= 0);
    failed_close_fd = -1;
    CHECK(close(reused) == 0);
    fail_unlink = 0;
    if (repair) CHECK(access(path, F_OK) < 0 && errno == ENOENT);
    else CHECK(unlink(path) == 0);
}
static int64_t milliseconds(clockid_t clock) {
    struct timespec t;CHECK(clock_gettime(clock,&t)==0);
    return (int64_t)t.tv_sec*1000+t.tv_nsec/1000000;
}
static void bounded_connect_backoff(int64_t loop,const char *path) {
    int server=socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK,0);CHECK(server>=0);
    struct sockaddr_un address={.sun_family=AF_UNIX};strcpy(address.sun_path,path);
    CHECK(bind(server,(struct sockaddr*)&address,sizeof(address))==0 && listen(server,1)==0);
    int held[16],count=0,full=0;
    while(count<16) {
        int fd=socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK,0);CHECK(fd>=0);
        held[count++]=fd;
        if(connect(fd,(struct sockaddr*)&address,sizeof(address))<0) {
            CHECK(errno==EAGAIN || errno==EWOULDBLOCK);full=1;break;
        }
    }
    CHECK(full);
    int64_t id=eshkol_unix_stream_socket_connect_begin(loop,path,180);CHECK(id>0);
    connect_calls=0;
    int64_t cpu=milliseconds(CLOCK_THREAD_CPUTIME_ID),start=milliseconds(CLOCK_MONOTONIC);
    CHECK(eshkol_unix_stream_socket_connect_finish(id)==ESHKOL_IO_TIMEOUT);
    int64_t elapsed=milliseconds(CLOCK_MONOTONIC)-start,spent=milliseconds(CLOCK_THREAD_CPUTIME_ID)-cpu;
    CHECK(elapsed>=140 && elapsed<1500 && connect_calls<80 && spent<70);
    fprintf(stderr,"MEASURE: connect_backoff_elapsed_ms=%lld cpu_ms=%lld attempts=%d\n",(long long)elapsed,(long long)spent,connect_calls);
    CHECK(eshkol_unix_stream_socket_close(id)==0);
    while(count)CHECK(close(held[--count])==0);
    CHECK(close(server)==0 && unlink(path)==0);
}
static void exhaustion_child(const char *path) {
    int64_t loop=eshkol_event_loop_open_handle(16);CHECK(loop>0);
    int64_t listener=eshkol_unix_stream_socket_listen(loop,path);CHECK(listener>0);
    int64_t client=eshkol_unix_stream_socket_connect_begin(loop,path,1000);CHECK(client>0);
    CHECK(eshkol_unix_stream_socket_connect_finish(client)==0);
    struct rlimit limit;CHECK(getrlimit(RLIMIT_NOFILE,&limit)==0);
    if(limit.rlim_cur>64) limit.rlim_cur=64;
    CHECK(setrlimit(RLIMIT_NOFILE,&limit)==0);
    int held[64],count=0,fd;
    while(count<64 && (fd=open("/dev/null",O_RDONLY|O_CLOEXEC))>=0) held[count++]=fd;
    CHECK(count>0 && errno==EMFILE);
    CHECK(eshkol_unix_stream_socket_accept(listener,0)==ESHKOL_IO_SYSTEM);
    CHECK(eshkol_unix_stream_socket_last_os_error()==EMFILE);
    CHECK(close(held[--count])==0);
    int64_t peer=eshkol_unix_stream_socket_accept(listener,1000);CHECK(peer>0);
    char value='x',got=0;
    CHECK(eshkol_unix_stream_socket_write(client,&value,1)==1);
    CHECK(eshkol_unix_stream_socket_read(peer,&got,1)==1 && got==value);
    while(count) CHECK(close(held[--count])==0);
    CHECK(eshkol_unix_stream_socket_close(peer)==0);
    CHECK(eshkol_unix_stream_socket_close(client)==0);
    CHECK(eshkol_unix_stream_socket_close(listener)==0);
    CHECK(eshkol_event_loop_close_handle(loop)==0);
    _exit(0);
}
int main(void) {
    char directory[80],path[108];
    CHECK(eshkol_test_socket_directory(directory,sizeof(directory),"eshkol-socket-regression-"));
    CHECK(snprintf(path,sizeof(path),"%s/s.sock",directory)<(int)sizeof(path));
    fprintf(stderr,"SCRATCH: %s\n",directory);
    int64_t loop=eshkol_event_loop_open_handle(16);CHECK(loop>0);
    CHECK(eshkol_unix_stream_socket_listen(loop,"@abstract")==ESHKOL_IO_INVALID);
    CHECK(eshkol_unix_stream_socket_connect_begin(loop,"@abstract",0)==ESHKOL_IO_INVALID);
    const char abstract[]={'\0','a',0};
    CHECK(eshkol_unix_stream_socket_listen(loop,abstract)==ESHKOL_IO_INVALID);
    fail_mode=fail_unlink=fail_close_once=1;
    CHECK(eshkol_unix_stream_socket_listen(loop,path)==ESHKOL_IO_SYSTEM);
    CHECK(eshkol_unix_stream_socket_last_os_error()==EIO);
    CHECK(eshkol_unix_stream_socket_last_path_error()==EACCES);
    CHECK(failed_close_calls==1);failed_close_fd=-1;
    struct stat entry;CHECK(lstat(path,&entry)==0 && S_ISSOCK(entry.st_mode));
    fail_mode=fail_unlink=0;CHECK(unlink(path)==0);
    int64_t listener=eshkol_unix_stream_socket_listen(loop,path);CHECK(listener>0);
    CHECK(eshkol_unix_stream_socket_last_path_error()==0);
    int64_t client=eshkol_unix_stream_socket_connect_begin(loop,path,1000);CHECK(client>0);
    int64_t peer=eshkol_unix_stream_socket_accept(listener,1000);CHECK(peer>0);
    CHECK(eshkol_unix_stream_socket_connect_finish(client)==0);
    char value='r',got=0;
    interrupt_write=1;
    CHECK(eshkol_unix_stream_socket_write(client,&value,1)==ESHKOL_IO_TIMEOUT);
    CHECK(eshkol_unix_stream_socket_last_os_error()==EINTR);
    CHECK(eshkol_unix_stream_socket_write(client,&value,1)==1);
    interrupt_read=1;
    CHECK(eshkol_unix_stream_socket_read(peer,&got,1)==ESHKOL_IO_TIMEOUT);
    CHECK(eshkol_unix_stream_socket_last_os_error()==EINTR);
    CHECK(eshkol_unix_stream_socket_read(peer,&got,1)==1 && got==value);
    CHECK(atomic_sockets>=3 && atomic_accepts>=1);
    CHECK(eshkol_unix_stream_socket_close(peer)==0);
    CHECK(eshkol_unix_stream_socket_close(client)==0);
    CHECK(eshkol_unix_stream_socket_close(listener)==0);
    combined_cleanup_failure(path,1);
    combined_cleanup_failure(path,0);
    bounded_connect_backoff(loop,path);
    CHECK(eshkol_event_loop_close_handle(loop)==0);
    pid_t child=fork();CHECK(child>=0);
    if(!child) exhaustion_child(path);
    int status;CHECK(waitpid(child,&status,0)==child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0);
    CHECK(rmdir(directory)==0);
    puts("PASS: atomic socket/accept flags, EINTR retry, rollback path status, abstract refusal, bounded connect backoff, descriptor/path error priority, EMFILE recovery");
    return 0;
}
