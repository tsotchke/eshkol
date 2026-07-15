#import <Foundation/Foundation.h>

#include "eshkol/agent_http.h"
#include "agent_http_internal.h"
#include "agent_sse_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ESHKOL_HTTP_MAX_RESPONSE_BYTES ((NSUInteger)256 * 1024 * 1024)

struct qllm_http_response {
    int32_t status;
    char* body;
    int64_t body_len;
    char* error;
};

static char* apple_copy_bytes(const void* bytes, size_t length) {
    char* result = (char*)malloc(length + 1);
    if (!result) return NULL;
    if (length) memcpy(result, bytes, length);
    result[length] = '\0';
    return result;
}

static BOOL apple_apply_header_lines(NSMutableURLRequest* request,
                                     const char* lines) {
    if (!lines || !*lines) return YES;
    const char* cursor = lines;
    while (*cursor) {
        const char* end = strchr(cursor, '\n');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length == 0) {
            cursor = end ? end + 1 : cursor + length;
            continue;
        }
        const char* colon = memchr(cursor, ':', length);
        if (!colon) return NO;
        size_t name_length = (size_t)(colon - cursor);
        const char* value = colon + 1;
        size_t value_length = length - name_length - 1;
        while (value_length && *value == ' ') {
            value++;
            value_length--;
        }
        if (!eshkol_http_valid_token(cursor, name_length) ||
            !eshkol_http_valid_field_value(value, value_length)) return NO;
        NSString* name = [[NSString alloc] initWithBytes:cursor
                                                  length:name_length
                                                encoding:NSUTF8StringEncoding];
        NSString* fieldValue = [[NSString alloc] initWithBytes:value
                                                        length:value_length
                                                      encoding:NSUTF8StringEncoding];
        if (!name || !fieldValue) return NO;
        [request setValue:fieldValue forHTTPHeaderField:name];
        cursor = end ? end + 1 : cursor + length;
    }
    return YES;
}

static NSMutableURLRequest* apple_request(const char* method,
                                          const char* url,
                                          const char* header_lines,
                                          const char* body,
                                          int64_t body_len,
                                          int32_t timeout_ms) {
    if (!eshkol_http_valid_method(method) || !url || body_len < 0 ||
        (!body && body_len != 0) || (uint64_t)body_len > NSUIntegerMax) return nil;
    NSString* urlText = [NSString stringWithUTF8String:url];
    NSURL* target = urlText ? [NSURL URLWithString:urlText] : nil;
    if (!target || !target.scheme || !target.host ||
        !([target.scheme.lowercaseString isEqualToString:@"http"] ||
          [target.scheme.lowercaseString isEqualToString:@"https"])) return nil;
    NSTimeInterval timeout = (timeout_ms > 0 ? timeout_ms : 30000) / 1000.0;
    NSMutableURLRequest* request =
        [NSMutableURLRequest requestWithURL:target
                               cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                           timeoutInterval:timeout];
    request.HTTPMethod = [NSString stringWithUTF8String:method];
    [request setValue:@"eshkol-agent/1.3" forHTTPHeaderField:@"User-Agent"];
    if (!apple_apply_header_lines(request, header_lines)) return nil;
    if (body_len > 0)
        request.HTTPBody = [NSData dataWithBytes:body length:(NSUInteger)body_len];
    return request;
}

static NSURLSessionConfiguration* apple_session_configuration(int32_t timeout_ms) {
    NSURLSessionConfiguration* configuration =
        [NSURLSessionConfiguration ephemeralSessionConfiguration];
    NSTimeInterval timeout = (timeout_ms > 0 ? timeout_ms : 30000) / 1000.0;
    configuration.timeoutIntervalForRequest = timeout;
    configuration.timeoutIntervalForResource = timeout;
    configuration.HTTPShouldSetCookies = NO;
    configuration.requestCachePolicy = NSURLRequestReloadIgnoringLocalCacheData;
    return configuration;
}

@interface EshkolHTTPDelegate : NSObject <NSURLSessionDataDelegate>
@property(nonatomic, readonly) NSCondition* condition;
@property(nonatomic, readonly) NSMutableData* data;
@property(nonatomic) NSInteger status;
@property(nonatomic, strong) NSError* error;
@property(nonatomic) BOOL done;
@end

@implementation EshkolHTTPDelegate
- (instancetype)init {
    self = [super init];
    if (self) {
        _condition = [[NSCondition alloc] init];
        _data = [[NSMutableData alloc] init];
    }
    return self;
}
- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task
 didReceiveResponse:(NSURLResponse*)response
  completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler {
    (void)session;
    (void)task;
    [_condition lock];
    if ([response isKindOfClass:[NSHTTPURLResponse class]])
        _status = ((NSHTTPURLResponse*)response).statusCode;
    [_condition unlock];
    completionHandler(NSURLSessionResponseAllow);
}
- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task
    didReceiveData:(NSData*)data {
    (void)session;
    [_condition lock];
    if (_data.length > ESHKOL_HTTP_MAX_RESPONSE_BYTES - data.length) {
        _error = [NSError errorWithDomain:NSURLErrorDomain
                                     code:NSURLErrorDataLengthExceedsMaximum
                                 userInfo:@{NSLocalizedDescriptionKey:
                                                @"HTTP response exceeds 256 MiB limit"}];
        [task cancel];
    } else {
        [_data appendData:data];
    }
    [_condition unlock];
}
- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task
 didCompleteWithError:(NSError*)error {
    (void)session;
    (void)task;
    [_condition lock];
    if (!_error) _error = error;
    _done = YES;
    [_condition broadcast];
    [_condition unlock];
}
@end

static qllm_http_response_t* apple_perform(NSMutableURLRequest* request,
                                           int32_t timeout_ms) {
    if (!request) return NULL;
    EshkolHTTPDelegate* delegate = [[EshkolHTTPDelegate alloc] init];
    NSOperationQueue* queue = [[NSOperationQueue alloc] init];
    queue.maxConcurrentOperationCount = 1;
    NSURLSession* session = [NSURLSession sessionWithConfiguration:
                                 apple_session_configuration(timeout_ms)
                                                     delegate:delegate
                                                delegateQueue:queue];
    NSURLSessionDataTask* task = [session dataTaskWithRequest:request];
    [task resume];

    NSTimeInterval seconds = (timeout_ms > 0 ? timeout_ms : 30000) / 1000.0;
    NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:seconds];
    [delegate.condition lock];
    while (!delegate.done && [delegate.condition waitUntilDate:deadline]) {}
    BOOL timedOut = !delegate.done;
    if (timedOut) [task cancel];
    NSData* data = [delegate.data copy];
    NSInteger status = delegate.status;
    NSError* error = delegate.error;
    [delegate.condition unlock];
    [session finishTasksAndInvalidate];

    qllm_http_response_t* response =
        (qllm_http_response_t*)calloc(1, sizeof(*response));
    if (!response) return NULL;
    response->status = (int32_t)status;
    response->body = apple_copy_bytes(data.bytes, data.length);
    response->body_len = (int64_t)data.length;
    if (timedOut) {
        response->error = apple_copy_bytes("HTTP request timed out", 22);
    } else if (error) {
        const char* message = error.localizedDescription.UTF8String;
        response->error = apple_copy_bytes(message, strlen(message));
    }
    if (!response->body) response->body = apple_copy_bytes("", 0);
    return response;
}

int32_t qllm_http_init(void) { return 1; }
void qllm_http_shutdown(void) {}
int32_t qllm_http_has_ssl(void) { return 1; }

qllm_http_response_t* eshkol_http_request(const char* method,
                                          const char* url,
                                          const char* header_lines,
                                          const char* body,
                                          int32_t timeout_ms) {
    @autoreleasepool {
        return apple_perform(apple_request(method, url, header_lines, body,
                                           body ? (int64_t)strlen(body) : 0,
                                           timeout_ms),
                             timeout_ms);
    }
}

qllm_http_response_t* eshkol_http_request_bytes(const char* method,
                                                const char* url,
                                                const char* header_lines,
                                                const char* body,
                                                int64_t body_len,
                                                int32_t timeout_ms) {
    @autoreleasepool {
        return apple_perform(apple_request(method, url, header_lines, body,
                                           body_len, timeout_ms), timeout_ms);
    }
}

qllm_http_response_t* qllm_http_get(const char* url, int32_t timeout_ms) {
    return eshkol_http_request("GET", url, NULL, NULL, timeout_ms);
}

qllm_http_response_t* qllm_http_post(const char* url,
                                     const char** headers,
                                     int64_t header_count,
                                     const char* body,
                                     int64_t body_len,
                                     int32_t timeout_ms) {
    @autoreleasepool {
        NSMutableString* lines = [[NSMutableString alloc] init];
        for (int64_t i = 0; headers && i < header_count; ++i) {
            if (headers[i]) {
                if (!eshkol_http_valid_header_line(headers[i])) return NULL;
                [lines appendFormat:@"%s\n", headers[i]];
            }
        }
        return eshkol_http_request_bytes("POST", url, lines.UTF8String, body,
                                         body_len, timeout_ms);
    }
}

qllm_http_response_t* qllm_http_post_json(const char* url,
                                          const char* body,
                                          const char* auth_header,
                                          int32_t timeout_ms) {
    @autoreleasepool {
        NSMutableString* headers =
            [NSMutableString stringWithString:@"Content-Type: application/json\nAccept: application/json\n"];
        if (auth_header && *auth_header) {
            if (!eshkol_http_valid_header_line(auth_header)) return NULL;
            [headers appendFormat:@"%s\n", auth_header];
        }
        return eshkol_http_request("POST", url, headers.UTF8String, body, timeout_ms);
    }
}

int32_t qllm_http_response_status(qllm_http_response_t* response) {
    return response ? response->status : 0;
}
const char* qllm_http_response_body(qllm_http_response_t* response) {
    return response ? response->body : NULL;
}
int64_t qllm_http_response_body_len(qllm_http_response_t* response) {
    return response ? response->body_len : 0;
}
const char* qllm_http_response_error(qllm_http_response_t* response) {
    return response ? response->error : NULL;
}
void qllm_http_response_free(qllm_http_response_t* response) {
    if (!response) return;
    free(response->body);
    free(response->error);
    free(response);
}
const char* qllm_http_error_string(int32_t code) {
    @autoreleasepool {
        static _Thread_local char buffer[512];
        NSString* message = [NSHTTPURLResponse localizedStringForStatusCode:code];
        snprintf(buffer, sizeof(buffer), "%s", message.UTF8String ?: "HTTP error");
        return buffer;
    }
}

@interface EshkolSSEDelegate : NSObject <NSURLSessionDataDelegate>
@property(nonatomic, readonly) NSCondition* condition;
@property(nonatomic, readonly) eshkol_sse_parser_t* parser;
@property(nonatomic, strong) NSError* error;
@property(nonatomic) NSInteger status;
@property(nonatomic) BOOL headersReceived;
@property(nonatomic) BOOL done;
@end

@implementation EshkolSSEDelegate
- (instancetype)init {
    self = [super init];
    if (self) {
        _condition = [[NSCondition alloc] init];
        _parser = eshkol_sse_parser_create();
        if (!_parser) return nil;
    }
    return self;
}
- (void)dealloc { eshkol_sse_parser_destroy(_parser); }
- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task
 didReceiveResponse:(NSURLResponse*)response
  completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler {
    (void)session;
    (void)task;
    [_condition lock];
    if ([response isKindOfClass:[NSHTTPURLResponse class]])
        _status = ((NSHTTPURLResponse*)response).statusCode;
    _headersReceived = YES;
    BOOL accepted = _status >= 200 && _status < 300;
    if (!accepted) {
        _error = [NSError errorWithDomain:NSURLErrorDomain
                                     code:NSURLErrorBadServerResponse
                                 userInfo:@{NSLocalizedDescriptionKey:
                                                @"SSE endpoint returned a non-2xx response"}];
        _done = YES;
    }
    [_condition broadcast];
    [_condition unlock];
    completionHandler(accepted ? NSURLSessionResponseAllow : NSURLSessionResponseCancel);
}
- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task
    didReceiveData:(NSData*)data {
    (void)session;
    [_condition lock];
    if (!eshkol_sse_parser_feed(_parser, data.bytes, data.length)) {
        _error = [NSError errorWithDomain:NSURLErrorDomain
                                     code:NSURLErrorDataLengthExceedsMaximum
                                 userInfo:@{NSLocalizedDescriptionKey:
                                                @"SSE event buffer exceeds 8 MiB limit"}];
        _done = YES;
        [task cancel];
    }
    [_condition broadcast];
    [_condition unlock];
}
- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task
 didCompleteWithError:(NSError*)error {
    (void)session;
    (void)task;
    [_condition lock];
    if (!_error && error && error.code != NSURLErrorCancelled) _error = error;
    _done = YES;
    [_condition broadcast];
    [_condition unlock];
}
@end

typedef struct eshkol_apple_stream {
    const void* delegate_ref;
    const void* session_ref;
    const void* task_ref;
    char error[512];
} eshkol_apple_stream_t;

static EshkolSSEDelegate* stream_delegate(eshkol_apple_stream_t* stream) {
    return (__bridge EshkolSSEDelegate*)stream->delegate_ref;
}

void* eshkol_http_stream_open(const char* method,
                              const char* url,
                              const char* header_lines,
                              const char* body,
                              int32_t timeout_ms) {
    return eshkol_http_stream_open_bytes(method, url, header_lines, body,
                                         body ? (int64_t)strlen(body) : 0,
                                         timeout_ms);
}

void* eshkol_http_stream_open_bytes(const char* method,
                                    const char* url,
                                    const char* header_lines,
                                    const char* body,
                                    int64_t body_len,
                                    int32_t timeout_ms) {
    @autoreleasepool {
        NSMutableURLRequest* request = apple_request(method, url, header_lines,
                                                     body, body_len, timeout_ms);
        if (!request) return NULL;
        [request setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
        [request setValue:@"no-cache" forHTTPHeaderField:@"Cache-Control"];
        EshkolSSEDelegate* delegate = [[EshkolSSEDelegate alloc] init];
        if (!delegate) return NULL;
        NSOperationQueue* queue = [[NSOperationQueue alloc] init];
        queue.maxConcurrentOperationCount = 1;
        NSURLSessionConfiguration* configuration = apple_session_configuration(timeout_ms);
        configuration.timeoutIntervalForResource = 7 * 24 * 60 * 60;
        NSURLSession* session = [NSURLSession sessionWithConfiguration:configuration
                                                               delegate:delegate
                                                          delegateQueue:queue];
        NSURLSessionDataTask* task = [session dataTaskWithRequest:request];
        eshkol_apple_stream_t* stream =
            (eshkol_apple_stream_t*)calloc(1, sizeof(*stream));
        if (!stream) {
            [session invalidateAndCancel];
            return NULL;
        }
        stream->delegate_ref = CFBridgingRetain(delegate);
        stream->session_ref = CFBridgingRetain(session);
        stream->task_ref = CFBridgingRetain(task);
        [task resume];

        NSTimeInterval seconds = (timeout_ms > 0 ? timeout_ms : 30000) / 1000.0;
        NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:seconds];
        [delegate.condition lock];
        while (!delegate.headersReceived && !delegate.done &&
               [delegate.condition waitUntilDate:deadline]) {}
        BOOL established = delegate.headersReceived && !delegate.error;
        [delegate.condition unlock];
        if (!established) {
            eshkol_http_stream_close(stream);
            return NULL;
        }
        return stream;
    }
}

eshkol_sse_event_t* eshkol_http_stream_next(void* opaque, int32_t timeout_ms) {
    @autoreleasepool {
        eshkol_apple_stream_t* stream = (eshkol_apple_stream_t*)opaque;
        if (!stream) return NULL;
        EshkolSSEDelegate* delegate = stream_delegate(stream);
        NSTimeInterval seconds = (timeout_ms > 0 ? timeout_ms : 30000) / 1000.0;
        NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:seconds];
        [delegate.condition lock];
        eshkol_sse_event_t* event = eshkol_sse_parser_next(delegate.parser);
        if (!event && eshkol_sse_parser_failed(delegate.parser)) {
            delegate.error = [NSError errorWithDomain:NSURLErrorDomain
                                                  code:NSURLErrorCannotDecodeContentData
                                              userInfo:@{NSLocalizedDescriptionKey:
                                                             @"Invalid or oversized SSE event stream"}];
            delegate.done = YES;
        }
        while (!event && !delegate.done &&
               [delegate.condition waitUntilDate:deadline]) {
            event = eshkol_sse_parser_next(delegate.parser);
            if (!event && eshkol_sse_parser_failed(delegate.parser)) {
                delegate.error = [NSError errorWithDomain:NSURLErrorDomain
                                                      code:NSURLErrorCannotDecodeContentData
                                                  userInfo:@{NSLocalizedDescriptionKey:
                                                                 @"Invalid or oversized SSE event stream"}];
                delegate.done = YES;
            }
        }
        [delegate.condition unlock];
        return event;
    }
}

int32_t eshkol_http_stream_done(void* opaque) {
    @autoreleasepool {
        eshkol_apple_stream_t* stream = (eshkol_apple_stream_t*)opaque;
        if (!stream) return 1;
        EshkolSSEDelegate* delegate = stream_delegate(stream);
        [delegate.condition lock];
        BOOL done = delegate.done &&
                    !eshkol_sse_parser_has_complete_event(delegate.parser);
        [delegate.condition unlock];
        return done ? 1 : 0;
    }
}

const char* eshkol_http_stream_error(void* opaque) {
    @autoreleasepool {
        eshkol_apple_stream_t* stream = (eshkol_apple_stream_t*)opaque;
        if (!stream) return "invalid stream";
        EshkolSSEDelegate* delegate = stream_delegate(stream);
        [delegate.condition lock];
        const char* message = delegate.error.localizedDescription.UTF8String;
        if (message) snprintf(stream->error, sizeof(stream->error), "%s", message);
        [delegate.condition unlock];
        return stream->error[0] ? stream->error : NULL;
    }
}

void eshkol_http_stream_close(void* opaque) {
    @autoreleasepool {
        eshkol_apple_stream_t* stream = (eshkol_apple_stream_t*)opaque;
        if (!stream) return;
        NSURLSessionDataTask* task = (__bridge NSURLSessionDataTask*)stream->task_ref;
        NSURLSession* session = (__bridge NSURLSession*)stream->session_ref;
        [task cancel];
        [session invalidateAndCancel];
        if (stream->task_ref) CFRelease(stream->task_ref);
        if (stream->session_ref) CFRelease(stream->session_ref);
        if (stream->delegate_ref) CFRelease(stream->delegate_ref);
        free(stream);
    }
}
