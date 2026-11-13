/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * eshkol_server_handle_client_test.cpp — the server binary answering real
 * connections.
 *
 * The request-handling test drives the accumulator and its own loopback
 * server, which is the right place to state the framing rules but does not
 * execute exe/eshkol-server.cpp. This one starts the shipped `eshkol-server`
 * binary on a loopback port and speaks HTTP to it, so handle_client's own
 * read loop, its refusal branches and its budgets are what answer.
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << std::endl;
        ++g_failures;
    }
}

void expect_status(int actual, int expected, const std::string& label) {
    if (actual != expected) {
        std::cerr << "FAIL: " << label << "\n  expected status: " << expected
                  << "\n  actual status:   " << actual << std::endl;
        ++g_failures;
    }
}

int connect_to(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* Sends exactly these bytes (NUL included) and returns the status line's
 * code, or 0 when the server answered nothing. */
int ask(int port, const std::string& request) {
    int fd = connect_to(port);
    if (fd < 0) return -1;
    size_t sent = 0;
    while (sent < request.size()) {
        ssize_t n = send(fd, request.data() + sent, request.size() - sent, 0);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
    std::string response;
    char buffer[4096];
    for (;;) {
        ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) break;
        response.append(buffer, static_cast<size_t>(n));
        if (response.size() > 64 * 1024) break;
    }
    close(fd);
    if (response.size() < 12 || response.compare(0, 5, "HTTP/") != 0) return 0;
    return std::atoi(response.c_str() + 9);
}

/* A port the kernel just gave us and we closed again: racy in principle,
 * fine for a test that owns the machine for a moment. */
int free_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return 0;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        close(fd);
        return 0;
    }
    int port = ntohs(addr.sin_port);
    close(fd);
    return port;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: eshkol_server_handle_client_test <server-binary>"
                  << std::endl;
        return 2;
    }
    const char* server = argv[1];
    const int port = free_port();
    if (port == 0) {
        std::cerr << "FAIL: could not reserve a port" << std::endl;
        return 1;
    }

    const pid_t child = fork();
    if (child < 0) {
        std::cerr << "FAIL: fork" << std::endl;
        return 1;
    }
    if (child == 0) {
        const std::string port_text = std::to_string(port);
        execl(server, server, "--host", "127.0.0.1", "--port",
              port_text.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    /* Wait for it to listen rather than sleeping a guessed interval. */
    bool up = false;
    for (int attempt = 0; attempt < 200 && !up; ++attempt) {
        int fd = connect_to(port);
        if (fd >= 0) {
            close(fd);
            up = true;
            break;
        }
        usleep(50 * 1000);
    }
    if (!up) {
        kill(child, SIGTERM);
        waitpid(child, nullptr, 0);
        std::cerr << "FAIL: the server did not start listening" << std::endl;
        return 1;
    }

    const std::string host = "Host: 127.0.0.1\r\n";

    // A request the server answers, so the cases below are refusals and not
    // a server that refuses everything.
    expect(ask(port, "GET / HTTP/1.1\r\n" + host + "\r\n") > 0,
           "the server answers an ordinary request");

    // ─── Framing, through the real read loop ────────────────────────────────
    expect_status(ask(port, "POST /compile HTTP/1.1\r\n" + host +
                                "Content-Length: 0\r\nContent-Length: 5\r\n\r\nhello"),
                  400, "a repeated Content-Length is refused");
    expect_status(ask(port, "POST /compile HTTP/1.1\r\n" + host +
                                "Content-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\nhello"),
                  400, "Transfer-Encoding beside Content-Length is refused");
    expect_status(ask(port, "POST /compile HTTP/1.1\r\n" + host +
                                "Content-Length : 5\r\n\r\nhello"),
                  400, "a space before the colon is refused");
    expect_status(ask(port, "POST /compile HTTP/1.1\r\n" + host +
                                "X: a\r\n content-length: 5\r\n\r\nhello"),
                  400, "an obs-fold continuation is refused");

    // ─── Host, through the real read loop ───────────────────────────────────
    expect_status(ask(port, "GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n"),
                  400, "two Host fields are refused");
    expect_status(ask(port, "GET / HTTP/1.1\r\n\r\n"),
                  400, "no Host field is refused");

    // ─── The request line is checked by length, not to the first NUL ───────
    {
        std::string nul_method = "G";
        nul_method.push_back('\0');
        nul_method += "ET / HTTP/1.1\r\n" + host + "\r\n";
        expect_status(ask(port, nul_method), 400,
                      "a NUL inside the method is refused");
    }

    // ─── The budgets answer on the connection ───────────────────────────────
    {
        std::string big = "GET / HTTP/1.1\r\n" + host;
        while (big.size() < 80 * 1024) big += "X-Pad: 0123456789abcdef\r\n";
        big += "\r\n";
        expect_status(ask(port, big), 431,
                      "a header section over the budget answers 431");
    }
    {
        const std::string over = std::to_string(11u * 1024u * 1024u);
        expect_status(ask(port, "POST /compile HTTP/1.1\r\n" + host +
                                    "Content-Length: " + over + "\r\n\r\n"),
                      413, "a body over the budget answers 413");
    }

    kill(child, SIGTERM);
    int status = 0;
    waitpid(child, &status, 0);

    if (g_failures == 0) {
        std::cout << "eshkol_server_handle_client_test: all cases pass"
                  << std::endl;
        return 0;
    }
    std::cerr << "eshkol_server_handle_client_test: " << g_failures
              << " case(s) failed" << std::endl;
    return 1;
}
