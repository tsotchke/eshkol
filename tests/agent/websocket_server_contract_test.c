#define _DEFAULT_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

extern int64_t eshkol_http_server_create(int32_t);
extern int32_t eshkol_http_server_port(int64_t);
extern int32_t eshkol_http_server_accept(int64_t, char*, int32_t, int32_t);
extern void eshkol_http_server_respond(int64_t, int32_t, const char*, const char*);
extern int64_t eshkol_ws_server_upgrade(int64_t, const char*, int32_t);
extern int64_t eshkol_ws_wrap_fd(int64_t);
extern int32_t eshkol_ws_receive(int64_t, char*, int32_t, int32_t*, int32_t);
extern int32_t eshkol_ws_send_text(int64_t, const char*, int32_t);
extern int32_t eshkol_ws_send_binary(int64_t, const char*, int32_t);
extern void eshkol_ws_close(int64_t);
extern void eshkol_http_server_close(int64_t);

static int failures;
static void check(int ok, const char* what) {
    if (ok) printf("PASS: %s\n", what);
    else { fprintf(stderr,"FAIL: %s\n",what); ++failures; }
}

typedef struct { int64_t server; int ok; } server_context_t;
static void* server_main(void* opaque) {
    server_context_t* c=(server_context_t*)opaque;
    char request[32768], payload[256]; int32_t type=0;
    int32_t n=eshkol_http_server_accept(c->server,request,sizeof(request),5000);
    int64_t ws=n>0?eshkol_ws_server_upgrade(c->server,request,n):-1;
    if (ws<1) return NULL;
    n=eshkol_ws_receive(ws,payload,sizeof(payload),&type,3000);
    if (n!=2 || type!=1 || memcmp(payload,"hi",2) || eshkol_ws_send_text(ws,payload,n)) goto done;
    n=eshkol_ws_receive(ws,payload,sizeof(payload),&type,3000);
    if (n!=0 || type!=1 || eshkol_ws_send_text(ws,payload,0)) goto done;
    n=eshkol_ws_receive(ws,payload,sizeof(payload),&type,3000);
    if (n!=1 || type!=2 || (unsigned char)payload[0]!=0x7f || eshkol_ws_send_binary(ws,payload,n)) goto done;
    n=eshkol_ws_receive(ws,payload,sizeof(payload),&type,3000);
    if (n!=2 || type!=9 || memcmp(payload,"ok",2)) goto done;
    c->ok=1;
done:
    eshkol_ws_close(ws);
    return NULL;
}

static int write_all(int fd,const void* bytes,size_t length) {
    const char* p=(const char*)bytes; size_t done=0;
    while(done<length) { ssize_t n=send(fd,p+done,length-done,0); if(n<=0)return 0; done+=(size_t)n; }
    return 1;
}
static int read_all(int fd,void* bytes,size_t length) {
    char* p=(char*)bytes; size_t done=0;
    while(done<length) { ssize_t n=recv(fd,p+done,length-done,0); if(n<=0)return 0; done+=(size_t)n; }
    return 1;
}
static int connect_local(int port) {
    int fd=socket(AF_INET,SOCK_STREAM,0); if(fd<0)return -1;
    struct sockaddr_in addr={0}; addr.sin_family=AF_INET; addr.sin_port=htons((uint16_t)port);
    addr.sin_addr.s_addr=htonl(0x7f000001u);
    if(connect(fd,(struct sockaddr*)&addr,sizeof(addr))!=0){close(fd);return -1;} return fd;
}
static int frame_read(int fd,unsigned char* opcode,unsigned char* payload,size_t cap,size_t* length) {
    unsigned char h[2]; if(!read_all(fd,h,2))return 0;
    if(h[1]&0x80)return 0;
    size_t n=h[1]&0x7f; if(n==126){unsigned char x[2];if(!read_all(fd,x,2))return 0;n=((size_t)x[0]<<8)|x[1];}
    else if(n==127){unsigned char x[8];if(!read_all(fd,x,8))return 0;n=0;for(int i=0;i<8;++i)n=(n<<8)|x[i];}
    if(n>cap)return 0; if(!read_all(fd,payload,n))return 0;
    *opcode=h[0]&15;*length=n;return 1;
}

static int send_masked_binary(int fd,size_t length) {
    unsigned char header[14],mask[4]={1,2,3,4}; size_t hlen=0;
    header[hlen++]=0x82;
    if(length<126) header[hlen++]=(unsigned char)(0x80|length);
    else if(length<65536) {
        header[hlen++]=0xfe; header[hlen++]=(unsigned char)(length>>8);
        header[hlen++]=(unsigned char)length;
    } else {
        header[hlen++]=0xff;
        uint64_t n=length; for(int i=7;i>=0;--i)header[hlen++]=(unsigned char)(n>>(i*8));
    }
    memcpy(header+hlen,mask,4); hlen+=4;
    unsigned char* packet=(unsigned char*)malloc(hlen+length);
    if(!packet)return 0;
    memcpy(packet,header,hlen);
    for(size_t i=0;i<length;++i)packet[hlen+i]=(unsigned char)(i*7)^mask[i&3];
    int ok=write_all(fd,packet,hlen+length); free(packet); return ok;
}
static void test_valid_roundtrip(void) {
    int64_t server=eshkol_http_server_create(0); int port=eshkol_http_server_port(server);
    server_context_t context={server,0}; pthread_t thread;
    check(server>0 && pthread_create(&thread,NULL,server_main,&context)==0,"server and worker start");
    if(server<1)return;
    int fd=connect_local(port); check(fd>=0,"client connects to loopback"); if(fd<0)return;
    static const char hs[]="GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: keep-alive, Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
    /* Coalesce the first complete masked frame with the handshake to exercise
     * the accepted-buffer overread path. */
    unsigned char packet[sizeof(hs)+8]; memcpy(packet,hs,sizeof(hs)-1);
    unsigned char first[]={0x81,0x82,1,2,3,4,'h'^1,'i'^2};
    memcpy(packet+sizeof(hs)-1,first,sizeof(first));
    check(write_all(fd,packet,sizeof(hs)-1+sizeof(first)),"handshake and first frame sent together");
    char response[512]={0}; size_t used=0;
    while(used<sizeof(response)-1 && !strstr(response,"\r\n\r\n")) { ssize_t n=recv(fd,response+used,1,0); if(n<=0)break;used+=(size_t)n;response[used]=0; }
    check(strstr(response,"101 Switching Protocols")!=NULL,"server returns HTTP 101");
    check(strstr(response,"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")!=NULL,"RFC 6455 accept hash matches known vector");
    unsigned char op,payload[32]; size_t n=0;
    check(frame_read(fd,&op,payload,sizeof(payload),&n)&&op==1&&n==2&&!memcmp(payload,"hi",2),"text echo is unmasked and exact");
    unsigned char empty[]={0x81,0x80,5,6,7,8};
    write_all(fd,empty,sizeof(empty));
    check(frame_read(fd,&op,payload,sizeof(payload),&n)&&op==1&&n==0,"empty text frame round-trips");
    unsigned char binary[]={0x82,0x81,9,8,7,6,(unsigned char)(0x7f^9)};
    write_all(fd,binary,sizeof(binary));
    check(frame_read(fd,&op,payload,sizeof(payload),&n)&&op==2&&n==1&&payload[0]==0x7f,"binary echo is exact");
    unsigned char ping[]={0x89,0x82,1,3,5,7,'o'^1,'k'^3};
    write_all(fd,ping,sizeof(ping));
    check(frame_read(fd,&op,payload,sizeof(payload),&n)&&op==10&&n==2&&!memcmp(payload,"ok",2),"ping receives payload-preserving pong");
    close(fd); pthread_join(thread,NULL);
    check(context.ok,"server accepted masked frames and completed control flow");
    eshkol_http_server_close(server);
}

static void* refusal_server_main(void* opaque) {
    server_context_t* x=(server_context_t*)opaque; char req[4096];
    int n=eshkol_http_server_accept(x->server,req,sizeof(req),5000);
    x->ok=n>0 && eshkol_ws_server_upgrade(x->server,req,n)<0;
    return NULL;
}

static void* protocol_server_main(void* opaque) {
    server_context_t* x=(server_context_t*)opaque; char req[4096], buf[16]; int32_t type=0;
    int n=eshkol_http_server_accept(x->server,req,sizeof(req),5000);
    int64_t ws=n>0?eshkol_ws_server_upgrade(x->server,req,n):-1;
    if (ws<1) return NULL;
    x->ok=eshkol_ws_receive(ws,buf,sizeof(buf),&type,3000)<0;
    eshkol_ws_close(ws); return NULL;
}

typedef struct { int64_t server; int result; uint64_t elapsed_ms; } backpressure_context_t;
static uint64_t test_now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000+(uint64_t)t.tv_nsec/1000000;
}
static void* backpressure_server_main(void* opaque) {
    backpressure_context_t* x=(backpressure_context_t*)opaque;
    char req[4096]; int n=eshkol_http_server_accept(x->server,req,sizeof(req),5000);
    int64_t ws=n>0?eshkol_ws_server_upgrade(x->server,req,n):-1;
    if(ws<1)return NULL;
    char* payload=(char*)malloc(1024u*1024u);
    if(!payload){x->result=-2;eshkol_ws_close(ws);return NULL;}
    memset(payload,'A',1024u*1024u);
    uint64_t start=test_now_ms();
    x->result=eshkol_ws_send_text(ws,payload,1024*1024);
    x->elapsed_ms=test_now_ms()-start;
    free(payload); eshkol_ws_close(ws); return NULL;
}

static int send_handshake(int fd) {
    static const char hs[]="GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
    if (!write_all(fd,hs,sizeof(hs)-1)) return 0;
    char response[256]={0}; size_t used=0;
    while(used<sizeof(response)-1 && !strstr(response,"\r\n\r\n")) {
        ssize_t n=recv(fd,response+used,1,0); if(n<=0)return 0; used+=(size_t)n; response[used]=0;
    }
    return strstr(response,"101 Switching Protocols")!=NULL;
}

static void test_bad_key(void) {
    int64_t server=eshkol_http_server_create(0); int port=eshkol_http_server_port(server);
    pthread_t thread; server_context_t c={server,0};
    pthread_create(&thread,NULL,refusal_server_main,&c);
    int fd=connect_local(port);
    if(fd>=0) {
        static const char bad[]="GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: invalid\r\n\r\n";
        write_all(fd,bad,sizeof(bad)-1); char response[256]={0}; ssize_t n=recv(fd,response,sizeof(response)-1,0);
        check(n>0 && strstr(response,"400 Bad Request"),"malformed WebSocket key is refused with HTTP 400"); close(fd);
    } else check(0,"bad-key client connects");
    pthread_join(thread,NULL); check(c.ok,"malformed key transfers no socket ownership"); eshkol_http_server_close(server);
}

static void test_frame_rejections(void) {
    for(int oversized=0;oversized<3;++oversized) {
        int64_t server=eshkol_http_server_create(0); int port=eshkol_http_server_port(server);
        pthread_t thread; server_context_t c={server,0};
        pthread_create(&thread,NULL,protocol_server_main,&c);
        int fd=connect_local(port);
        if(fd>=0 && send_handshake(fd)) {
            if(oversized==1) { unsigned char h[]={0x81,0x9e}; write_all(fd,h,sizeof(h)); }
            else if(oversized==2) { unsigned char h[]={0x81,0x81,1,2}; write_all(fd,h,sizeof(h)); shutdown(fd,SHUT_WR); }
            else { unsigned char h[]={0x81,0x02,'h','i'}; write_all(fd,h,sizeof(h)); }
        } else check(0,"protocol rejection client handshakes");
        if(fd>=0)close(fd); pthread_join(thread,NULL);
        check(c.ok,oversized==1?"oversized frame is rejected without truncation":oversized==2?"truncated frame fails instead of returning success":"unmasked client frame is rejected");
        eshkol_http_server_close(server);
    }
}

static void test_backpressure_deadline(void) {
    int64_t server=eshkol_http_server_create(0); int port=eshkol_http_server_port(server);
    backpressure_context_t c={server,0,0}; pthread_t thread;
    pthread_create(&thread,NULL,backpressure_server_main,&c);
    int fd=connect_local(port);
    if(fd>=0) {
        int receive_buffer=4096;
        setsockopt(fd,SOL_SOCKET,SO_RCVBUF,(const char*)&receive_buffer,sizeof(receive_buffer));
        check(send_handshake(fd),"backpressure client completes handshake");
        pthread_join(thread,NULL);
        check(c.result<0 && c.elapsed_ms<4500,"stalled peer hits finite WebSocket write deadline");
        check(c.elapsed_ms>=1000,"stalled peer remains blocked until bounded deadline");
        close(fd);
    } else { check(0,"backpressure client connects"); pthread_join(thread,NULL); }
    eshkol_http_server_close(server);
}

static void* extended_echo_server_main(void* opaque) {
    server_context_t* c=(server_context_t*)opaque; char req[4096],payload[65537]; int32_t type=0;
    int n=eshkol_http_server_accept(c->server,req,sizeof(req),5000);
    int64_t ws=n>0?eshkol_ws_server_upgrade(c->server,req,n):-1;
    if(ws<1)return NULL;
    const int lengths[]={65535,65536}; c->ok=1;
    for(size_t j=0;j<2;++j) {
        n=eshkol_ws_receive(ws,payload,sizeof(payload),&type,5000);
        if(n!=lengths[j] || type!=2) { c->ok=0; break; }
        for(int i=0;i<n;++i)if((unsigned char)payload[i]!=(unsigned char)(i*7)){c->ok=0;break;}
        if(!c->ok || eshkol_ws_send_binary(ws,payload,n)!=0) { c->ok=0; break; }
    }
    eshkol_ws_close(ws); return NULL;
}

static void test_extended_frame_lengths(void) {
    int64_t server=eshkol_http_server_create(0); int port=eshkol_http_server_port(server);
    pthread_t thread; server_context_t c={server,0};
    if(pthread_create(&thread,NULL,extended_echo_server_main,&c)!=0) {
        check(0,"extended frame server thread starts"); eshkol_http_server_close(server); return;
    }
    int fd=connect_local(port);
    if(fd<0 || !send_handshake(fd)) {
        check(0,"extended frame client handshakes"); if(fd>=0)close(fd);
        pthread_join(thread,NULL); eshkol_http_server_close(server); return;
    }
    unsigned char* echoed=(unsigned char*)malloc(65536); unsigned char opcode=0; size_t length=0;
    int ok=echoed!=NULL;
    const size_t lengths[]={65535,65536};
    for(size_t j=0;j<2 && ok;++j) {
        ok=send_masked_binary(fd,lengths[j]) && frame_read(fd,&opcode,echoed,65536,&length) &&
           opcode==2 && length==lengths[j];
        for(size_t i=0;i<length && ok;++i)if(echoed[i]!=(unsigned char)(i*7))ok=0;
    }
    free(echoed); close(fd); pthread_join(thread,NULL);
    check(ok && c.ok,"16-bit and 64-bit frame lengths round-trip exactly");
    eshkol_http_server_close(server);
}

static void test_refusal(void) {
    int64_t server=eshkol_http_server_create(0); int port=eshkol_http_server_port(server);
    pthread_t thread; server_context_t c={server,0};
    check(pthread_create(&thread,NULL,refusal_server_main,&c)==0,"refusal worker starts");
    int fd=connect_local(port); if(fd>=0) {
        static const char bad[]="GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 12\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
        write_all(fd,bad,sizeof(bad)-1); char response[256]={0}; ssize_t n=recv(fd,response,sizeof(response)-1,0);
        check(n>0 && strstr(response,"400 Bad Request"),"bad WebSocket version is refused with HTTP 400"); close(fd);
    } else check(0,"refusal client connects");
    pthread_join(thread,NULL); check(c.ok,"invalid handshake transfers no socket ownership"); eshkol_http_server_close(server);
}

static void test_stale_handle_protection(void) {
    int64_t old_server=eshkol_http_server_create(0);
    eshkol_http_server_close(old_server);
    int64_t current_server=eshkol_http_server_create(0);
    int current_port=eshkol_http_server_port(current_server);
    eshkol_http_server_close(old_server);
    check(old_server>0 && current_server>0 && old_server!=current_server &&
          eshkol_http_server_port(current_server)==current_port,
          "stale HTTP server handle cannot close a reused server slot");
    eshkol_http_server_close(current_server);

    int first[2], second[2];
    if(socketpair(AF_UNIX,SOCK_STREAM,0,first)!=0) {
        check(0,"first WebSocket socket pair opens"); return;
    }
    check(1,"first WebSocket socket pair opens");
    int64_t stale=eshkol_ws_wrap_fd(first[0]);
    eshkol_ws_close(stale); close(first[1]);
    if(socketpair(AF_UNIX,SOCK_STREAM,0,second)!=0) {
        check(0,"second WebSocket socket pair opens"); return;
    }
    check(1,"second WebSocket socket pair opens");
    int64_t current=eshkol_ws_wrap_fd(second[0]);
    char payload[32]; int32_t type=0;
    int stale_send=eshkol_ws_send_text(stale,"stale",5);
    int stale_receive=eshkol_ws_receive(stale,payload,sizeof(payload),&type,0);
    eshkol_ws_close(stale);
    int current_send=eshkol_ws_send_text(current,"fresh",5);
    unsigned char frame[64]; ssize_t frame_len=recv(second[1],frame,sizeof(frame),0);
    check(stale>0 && current>0 && stale!=current && stale_send<0 &&
          stale_receive<0 && current_send==0 && frame_len>0,
          "stale WebSocket send/receive/close cannot affect a reused slot");
    eshkol_ws_close(current); close(second[1]);
}

static void test_pending_http_socket_ownership(void) {
    int64_t server=eshkol_http_server_create(0); int port=eshkol_http_server_port(server);
    int first=connect_local(port), second=-1; char request[2048];
    static const char message[]="GET /pending HTTP/1.1\r\nHost: localhost\r\n\r\n";
    int sent=first>=0 && write_all(first,message,sizeof(message)-1);
    int accepted=sent?eshkol_http_server_accept(server,request,sizeof(request),3000):-1;
    second=connect_local(port);
    int second_accept=eshkol_http_server_accept(server,request,sizeof(request),0);
    eshkol_http_server_respond(server,200,"text/plain","owned");
    char response[256]={0}; size_t used=0;
    while(first>=0 && used<sizeof(response)-1 && !strstr(response,"owned")) {
        ssize_t n=recv(first,response+used,sizeof(response)-1-used,0);
        if(n<=0)break; used+=(size_t)n; response[used]='\0';
    }
    check(accepted>0 && second>=0 && second_accept<0 && strstr(response,"owned")!=NULL,
          "pending HTTP socket is preserved when another accept is refused");
    if(first>=0)close(first);
    if(second>=0)close(second);
    eshkol_http_server_close(server);
}

int main(void) { test_valid_roundtrip(); test_refusal(); test_bad_key(); test_frame_rejections(); test_backpressure_deadline(); test_extended_frame_lengths(); test_stale_handle_protection(); test_pending_http_socket_ownership(); printf("WebSocket server checks: %s (%d failures)\n",failures?"FAIL":"PASS",failures); return failures?1:0; }
