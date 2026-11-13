/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * eshkol_server_request_handling_test.cpp — how the server frames, bounds and
 * answers a request.
 *
 * The request-handling rules live in <eshkol/http_request_utils.h>, which
 * eshkol-server drives from its read loop. This test drives the same objects:
 * first directly, over the byte sequences that make framing ambiguous, and
 * then over real loopback sockets, where one connection drips a request a
 * byte at a time while another sends a whole one and both get the answer they
 * should.
 *
 * Every listener this test starts is bound to the loopback, serves a counted
 * number of connections, and is joined before the test returns.
 */

#include <eshkol/http_request_utils.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using accumulator = eshkol_http_request_accumulator;
using state = accumulator::state;

int g_failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << std::endl;
        ++g_failures;
    }
}

void expect_equal(long long actual, long long expected, const std::string& label) {
    if (actual != expected) {
        std::cerr << "FAIL: " << label << "\n  expected: " << expected
                  << "\n  actual:   " << actual << std::endl;
        ++g_failures;
    }
}

// Feed a whole request in one go.
accumulator feed(const std::string& request) {
    accumulator a;
    a.append(request.data(), request.size());
    return a;
}

// Feed a request one byte at a time, which is how a request arrives when the
// network, or the peer, hands it over slowly.
accumulator feed_by_byte(const std::string& request) {
    accumulator a;
    for (size_t i = 0; i < request.size(); ++i) {
        if (a.current() != state::need_more) break;
        a.append(request.data() + i, 1);
    }
    return a;
}

// ─── A loopback server built from the same read path ────────────────────────

struct served_request {
    int status = 0;
    std::string body;
    size_t body_length = 0;
};

// Reads one request with the accumulator and reports what it decided. This is
// the server's read loop, with the same two clocks.
served_request serve_one(int client, int deadline_seconds) {
    served_request result;
    accumulator a;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(deadline_seconds);
    struct timeval receive_timeout;
    receive_timeout.tv_sec = 1;
    receive_timeout.tv_usec = 0;
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
               sizeof(receive_timeout));

    char buffer[4096];
    while (a.current() == state::need_more) {
        if (std::chrono::steady_clock::now() >= deadline) {
            result.status = 408;
            return result;
        }
        const ssize_t read_bytes = recv(client, buffer, sizeof(buffer), 0);
        if (read_bytes > 0) {
            a.append(buffer, static_cast<size_t>(read_bytes));
            continue;
        }
        if (read_bytes == 0) {
            a.close();
            break;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
        result.status = 400;
        return result;
    }
    if (a.current() == state::refused) {
        result.status = a.refusal_status();
        result.body = a.refusal_message();
        return result;
    }
    result.status = 200;
    result.body_length = a.declared_body_length();
    result.body = a.data().substr(a.header_end(), result.body_length);
    return result;
}

struct loopback_server {
    int listener = -1;
    int port = 0;
    std::thread worker;
    std::atomic<int> served{0};

    bool start(int connections, int deadline_seconds) {
        listener = socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0) return false;
        int reuse = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(listener, reinterpret_cast<sockaddr*>(&address),
                 sizeof(address)) != 0)
            return false;
        socklen_t length = sizeof(address);
        if (getsockname(listener, reinterpret_cast<sockaddr*>(&address),
                        &length) != 0)
            return false;
        port = ntohs(address.sin_port);
        if (listen(listener, 16) != 0) return false;
        worker = std::thread([this, connections, deadline_seconds]() {
            std::vector<std::thread> handlers;
            for (int i = 0; i < connections; ++i) {
                const int client = accept(listener, nullptr, nullptr);
                if (client < 0) break;
                handlers.emplace_back([this, client, deadline_seconds]() {
                    const served_request answer = serve_one(client, deadline_seconds);
                    const std::string response =
                        "HTTP/1.1 " + std::to_string(answer.status) +
                        " x\r\nContent-Length: " +
                        std::to_string(answer.body.size()) + "\r\n\r\n" +
                        answer.body;
                    (void)send(client, response.c_str(), response.size(), 0);
                    close(client);
                    served.fetch_add(1);
                });
            }
            for (std::thread& handler : handlers) handler.join();
        });
        return true;
    }

    void stop() {
        if (worker.joinable()) worker.join();
        if (listener >= 0) close(listener);
        listener = -1;
    }
};

int connect_loopback(int port) {
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    if (client < 0) return -1;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (connect(client, reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) != 0) {
        close(client);
        return -1;
    }
    return client;
}

int read_status(int client) {
    char buffer[4096];
    std::string response;
    while (true) {
        const ssize_t read_bytes = recv(client, buffer, sizeof(buffer), 0);
        if (read_bytes <= 0) break;
        response.append(buffer, static_cast<size_t>(read_bytes));
        if (response.find("\r\n\r\n") != std::string::npos) break;
    }
    if (response.size() < 12) return 0;
    return std::atoi(response.c_str() + 9);
}

}  // namespace

int main() {
    const std::string line = "POST /compile HTTP/1.1\r\nHost: 127.0.0.1\r\n";

    // ─── One Content-Length, read once ──────────────────────────────────────
    {
        accumulator a = feed(line + "Content-Length: 3\r\n\r\nabc");
        expect(a.current() == state::complete, "a declared body completes");
        expect_equal(static_cast<long long>(a.declared_body_length()), 3,
                     "the declared length is the one that was read");
        expect(a.data().substr(a.header_end()) == "abc", "the body is the body");
    }
    {
        accumulator a = feed(line + "\r\n");
        expect(a.current() == state::complete,
               "a request with no declared body completes");
        expect_equal(static_cast<long long>(a.declared_body_length()), 0,
                     "no Content-Length means no body");
    }

    // The audit's desync pair, in both orders. Agreement between the values is
    // not the rule; appearing once is.
    for (const std::string& head :
         {std::string("Content-Length: 0\r\nContent-Length: 10485760\r\n"),
          std::string("Content-Length: 10485760\r\nContent-Length: 0\r\n"),
          std::string("Content-Length: 3\r\nContent-Length: 3\r\n"),
          std::string("content-length: 3\r\nCONTENT-LENGTH: 3\r\n")}) {
        accumulator a = feed(line + head + "\r\nabc");
        expect(a.current() == state::refused,
               "a repeated Content-Length is refused: " + head);
        expect_equal(a.refusal_status(), 400,
                     "a repeated Content-Length answers 400");
    }
    {
        accumulator a = feed(line + "Content-Length: 3, 3\r\n\r\nabc");
        expect(a.current() == state::refused,
               "a list-valued Content-Length is refused");
    }

    // A field name is a field name only where a line begins.
    {
        accumulator a = feed(line + "X-Junk: content-length: 999\r\n\r\n");
        expect(a.current() == state::complete,
               "a field name inside another field's value frames nothing");
        expect_equal(static_cast<long long>(a.declared_body_length()), 0,
                     "the junk field declares no body");
    }
    {
        accumulator a = feed(line + "X-Junk: transfer-encoding: chunked\r\n\r\n");
        expect(a.current() == state::complete,
               "a Transfer-Encoding inside a value frames nothing");
    }

    // Transfer-Encoding and Content-Length name two framings.
    {
        accumulator a = feed(line +
                             "Content-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\nabc");
        expect(a.current() == state::refused,
               "Transfer-Encoding beside Content-Length is refused");
        expect_equal(a.refusal_status(), 400,
                     "Transfer-Encoding beside Content-Length answers 400");
    }
    {
        accumulator a = feed(line + "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
        expect(a.current() == state::refused,
               "a Transfer-Encoding this server does not implement is refused");
    }

    // A Content-Length that is not a length.
    for (const std::string& value : {std::string("abc"), std::string("-1"),
                                     std::string("0x10"), std::string("")}) {
        accumulator a = feed(line + "Content-Length: " + value + "\r\n\r\n");
        expect(a.current() == state::refused,
               "a Content-Length of \"" + value + "\" is refused");
        expect_equal(a.refusal_status(), 400,
                     "an unreadable Content-Length answers 400");
    }

    // ─── A body is bytes, including NUL ─────────────────────────────────────
    {
        std::string body("ab\0cd", 5);
        body += std::string("\0\0", 2);
        const std::string request =
            line + "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        accumulator a = feed(request);
        expect(a.current() == state::complete, "a NUL-carrying body completes");
        expect_equal(static_cast<long long>(a.data().size() - a.header_end()),
                     static_cast<long long>(body.size()),
                     "every byte of a NUL-carrying body is accumulated");
        expect(a.data().substr(a.header_end()) == body,
               "a NUL-carrying body arrives unchanged");

        accumulator dripped = feed_by_byte(request);
        expect(dripped.current() == state::complete,
               "a NUL-carrying body completes when dripped");
        expect(dripped.data().substr(dripped.header_end()) == body,
               "a NUL-carrying body arrives unchanged when dripped");
    }
    {
        // A NUL inside the header section does not end the header section
        // either: the terminator is the blank line.
        std::string request = line;
        request += std::string("X-Note: a\0b\r\n", 13);
        request += "Content-Length: 1\r\n\r\nz";
        accumulator a = feed(request);
        expect(a.current() == state::complete,
               "a NUL in a header line does not end the request");
        expect(a.data().substr(a.header_end()) == "z",
               "the body after a NUL-carrying header is read");
    }

    // ─── Budgets ────────────────────────────────────────────────────────────
    {
        const std::string oversized_field =
            "X-Pad: " + std::string(eshkol_server_max_header_bytes() + 16, 'p') + "\r\n";
        accumulator a = feed(line + oversized_field + "\r\n");
        expect(a.current() == state::refused, "an oversized header section is refused");
        expect_equal(a.refusal_status(), 431,
                     "an oversized header section answers 431");
    }
    {
        accumulator a = feed(line + "Content-Length: " +
                             std::to_string(eshkol_server_max_body_bytes() + 1) +
                             "\r\n\r\n");
        expect(a.current() == state::refused, "an oversized declared body is refused");
        expect_equal(a.refusal_status(), 413, "an oversized body answers 413");
    }
    {
        accumulator a = feed(line + "Content-Length: " +
                             std::to_string(eshkol_server_max_body_bytes()) +
                             "\r\n\r\n");
        expect(a.current() == state::need_more,
               "a body exactly at the limit is accepted and awaited");
    }
    {
        // A header section that never ends is bounded even though no field is
        // individually long.
        accumulator a;
        const std::string field = "X-Pad: pad\r\n";
        a.append(line.data(), line.size());
        while (a.current() == state::need_more &&
               a.data().size() < eshkol_server_max_header_bytes() * 2)
            a.append(field.data(), field.size());
        expect(a.current() == state::refused,
               "an endless header section is refused");
        expect_equal(a.refusal_status(), 431,
                     "an endless header section answers 431");
    }

    // ─── A peer that stops mid-request ──────────────────────────────────────
    {
        accumulator a;
        const std::string partial = line + "Content-Length: 10\r\n\r\nabc";
        a.append(partial.data(), partial.size());
        expect(a.current() == state::need_more, "a short body is awaited");
        expect(a.close() == state::refused,
               "a request that ends early is not answered as complete");
        expect_equal(a.refusal_status(), 400, "a truncated request answers 400");
    }
    {
        accumulator a;
        expect(a.close() == state::complete,
               "a connection that sent nothing is not an error");
        expect(a.data().empty(), "nothing was sent and nothing was accumulated");
    }

    // ─── Over real sockets: a drip client does not hold the others ──────────
    {
        const int drip_clients = 16;
        loopback_server server;
        expect(server.start(drip_clients + 1, 2),
               "the loopback server starts");
        if (server.listener >= 0) {
            // Open the drip connections first and send one byte each, so they
            // are all mid-request when the honest client arrives.
            std::vector<int> drips;
            for (int i = 0; i < drip_clients; ++i) {
                const int client = connect_loopback(server.port);
                if (client < 0) continue;
                (void)send(client, "G", 1, 0);
                drips.push_back(client);
            }

            const int honest = connect_loopback(server.port);
            expect(honest >= 0, "an honest client connects while the drips are open");
            const std::string request =
                "POST /compile HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Content-Length: 5\r\n\r\nhello";
            const auto started = std::chrono::steady_clock::now();
            (void)send(honest, request.c_str(), request.size(), 0);
            const int honest_status = read_status(honest);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            close(honest);
            expect_equal(honest_status, 200,
                         "the honest client is answered while the drips are open");
            expect(elapsed < std::chrono::seconds(2),
                   "the honest client is answered without waiting for the drips");

            // Each drip is answered 408 on its own deadline and closed.
            int timed_out = 0;
            for (const int client : drips) {
                if (read_status(client) == 408) ++timed_out;
                close(client);
            }
            expect_equal(timed_out, static_cast<long long>(drips.size()),
                         "every drip connection is answered 408 and dropped");
            server.stop();
            expect_equal(server.served.load(),
                         static_cast<long long>(drips.size()) + 1,
                         "every connection was accounted for");
        }
    }

    // ─── Over real sockets: a desync attempt and a NUL body ─────────────────
    {
        loopback_server server;
        expect(server.start(2, 5), "the loopback server starts for framing cases");
        if (server.listener >= 0) {
            const int desync = connect_loopback(server.port);
            const std::string smuggle =
                "POST /compile HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                "Content-Length: 0\r\nContent-Length: 10\r\n\r\n0123456789";
            (void)send(desync, smuggle.c_str(), smuggle.size(), 0);
            expect_equal(read_status(desync), 400,
                         "a desync attempt over a socket is answered 400");
            close(desync);

            const int nul_body = connect_loopback(server.port);
            std::string body("a\0b\0c", 5);
            std::string request =
                "POST /compile HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + body;
            (void)send(nul_body, request.data(), request.size(), 0);
            expect_equal(read_status(nul_body), 200,
                         "a NUL-carrying body over a socket is answered 200");
            close(nul_body);
            server.stop();
        }
    }

    // ─── Request lines ──────────────────────────────────────────────────────
    {
        expect(eshkol_http_valid_method("POST"), "POST is a method");
        expect(!eshkol_http_valid_request_target("/a b"),
               "the server refuses a target with a space");
    }

    // ─── JSON strings ───────────────────────────────────────────────────────
    {
        // A session identifier is a value the client may have chosen, so what
        // it contains cannot change the shape of the document it goes into.
        const std::string hostile = "a\"b\\c\nd\te\rf";
        const std::string rendered = eshkol_json_string(hostile);
        expect(rendered == "\"a\\\"b\\\\c\\nd\\te\\rf\"",
               "quotes, backslashes and control characters are escaped");
        expect(rendered.find('\n') == std::string::npos,
               "no raw newline survives into the document");
        expect(rendered.find('\r') == std::string::npos,
               "no raw carriage return survives into the document");

        const std::string injection = "\",\"success\":false,\"x\":\"";
        const std::string escaped = eshkol_json_string(injection);
        expect(escaped.find("\",\"success\"") == std::string::npos,
               "a value cannot close its own string and add a member");

        std::string with_nul("a\0b", 3);
        expect(eshkol_json_string(with_nul) == "\"a\\u0000b\"",
               "a NUL in a value is escaped, not dropped");
        expect(eshkol_json_string("\x01\x1f") == "\"\\u0001\\u001f\"",
               "the remaining control characters are escaped as \\u00XX");
        expect(eshkol_json_string("\b\f") == "\"\\b\\f\"",
               "backspace and form feed use their short escapes");
        expect(eshkol_json_string("plain") == "\"plain\"",
               "an ordinary value is rendered as itself");
    }

    if (g_failures == 0) {
        std::cout << "eshkol_server_request_handling_test: all cases pass"
                  << std::endl;
        return 0;
    }
    std::cerr << "eshkol_server_request_handling_test: " << g_failures
              << " case(s) failed" << std::endl;
    return 1;
}
