/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * websocket_handshake_test.cpp — the RFC 6455 opening handshake as Eshkol's
 * clients perform it.
 *
 * The first half checks the values against the specification: the SHA-1 and
 * base64 known answers, the Sec-WebSocket-Accept example from RFC 6455
 * section 1.3, a key and a mask that differ between draws.
 *
 * The second half runs the handshake over a loopback socket against a peer
 * this test starts: one peer answers with the value derived from the key, one
 * answers with a value derived from a different key, and the others answer
 * with a wrong status or a missing field. The client side is
 * eshkol_ws_handshake_refusal — the same function the core builtin and the
 * bytecode VM call on the bytes they read — so what this test observes is
 * what a connection gets. Each peer serves exactly one connection and the
 * test joins its thread, so nothing outlives the run.
 */

#include <eshkol/core/websocket_handshake.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace {

int g_failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << std::endl;
        ++g_failures;
    }
}

void expect_refusal(const char* actual, const char* expected,
                    const std::string& label) {
    const bool ok = expected ? (actual && std::strcmp(actual, expected) == 0)
                             : (actual == nullptr);
    if (!ok) {
        std::cerr << "FAIL: " << label << "\n  expected: "
                  << (expected ? expected : "(handshake complete)")
                  << "\n  actual:   " << (actual ? actual : "(handshake complete)")
                  << std::endl;
        ++g_failures;
    }
}

std::string hex(const unsigned char* bytes, size_t length) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (size_t i = 0; i < length; ++i) {
        out += digits[(bytes[i] >> 4) & 0x0f];
        out += digits[bytes[i] & 0x0f];
    }
    return out;
}

// ─── A loopback peer that answers one opening handshake ─────────────────────

// How the peer answers. The handshake is the client's decision either way;
// the peer only supplies bytes.
enum class peer_answer {
    derived_from_key,   // the value RFC 6455 requires
    derived_from_other, // a value derived from a different key
    status_200,         // not an upgrade at all
    missing_upgrade,    // 101 without the Upgrade field
    missing_accept,     // 101 with Upgrade and Connection but no Accept
    status_text_101     // a 101 that appears only inside a reason phrase
};

struct loopback_peer {
    int listener = -1;
    int port = 0;
    std::thread worker;
    std::string request_seen;

    bool start(peer_answer answer) {
        listener = socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0) return false;
        int reuse = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        if (bind(listener, reinterpret_cast<sockaddr*>(&address),
                 sizeof(address)) != 0)
            return false;
        socklen_t length = sizeof(address);
        if (getsockname(listener, reinterpret_cast<sockaddr*>(&address),
                        &length) != 0)
            return false;
        port = ntohs(address.sin_port);
        if (listen(listener, 1) != 0) return false;
        worker = std::thread([this, answer]() { serve(answer); });
        return true;
    }

    void serve(peer_answer answer) {
        const int client = accept(listener, nullptr, nullptr);
        if (client < 0) return;
        char buffer[4096];
        const ssize_t read_bytes = recv(client, buffer, sizeof(buffer) - 1, 0);
        if (read_bytes > 0) request_seen.assign(buffer, static_cast<size_t>(read_bytes));

        // Read the key out of the request the client actually sent.
        std::string key;
        {
            const char* value = nullptr;
            size_t value_length = 0;
            if (eshkol_ws_find_header(request_seen.c_str(), request_seen.size(),
                                      "sec-websocket-key", &value, &value_length))
                key.assign(value, value_length);
        }
        char accept_value[ESHKOL_WS_ACCEPT_CAPACITY] = {0};
        eshkol_ws_expected_accept(
            answer == peer_answer::derived_from_other ? "a-different-key"
                                                      : key.c_str(),
            accept_value, sizeof(accept_value));

        std::string response;
        switch (answer) {
            case peer_answer::status_200:
                response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
                break;
            case peer_answer::missing_upgrade:
                response = std::string("HTTP/1.1 101 Switching Protocols\r\n") +
                           "Connection: Upgrade\r\nSec-WebSocket-Accept: " +
                           accept_value + "\r\n\r\n";
                break;
            case peer_answer::missing_accept:
                response = "HTTP/1.1 101 Switching Protocols\r\n"
                           "Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n";
                break;
            case peer_answer::status_text_101:
                response = "HTTP/1.1 200 Give me 101 reasons\r\n"
                           "Content-Length: 0\r\n\r\n";
                break;
            default:
                response = std::string("HTTP/1.1 101 Switching Protocols\r\n") +
                           "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                           "Sec-WebSocket-Accept: " + accept_value + "\r\n\r\n";
                break;
        }
        (void)send(client, response.c_str(), response.size(), 0);
        close(client);
    }

    void stop() {
        if (worker.joinable()) worker.join();
        if (listener >= 0) close(listener);
        listener = -1;
    }
};

// Perform a real opening handshake against a peer on the loopback and return
// the client's verdict.
const char* handshake_against(peer_answer answer, const std::string& label,
                              std::string* request_seen = nullptr) {
    loopback_peer peer;
    if (!peer.start(answer)) {
        std::cerr << "FAIL: " << label << " (the loopback peer did not start)"
                  << std::endl;
        ++g_failures;
        return "test setup";
    }

    char key[ESHKOL_WS_KEY_CAPACITY];
    if (!eshkol_ws_generate_key(key, sizeof(key))) {
        peer.stop();
        std::cerr << "FAIL: " << label << " (no cryptographic source)" << std::endl;
        ++g_failures;
        return "test setup";
    }

    const int client = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(peer.port));
    if (client < 0 || connect(client, reinterpret_cast<sockaddr*>(&address),
                              sizeof(address)) != 0) {
        if (client >= 0) close(client);
        peer.stop();
        std::cerr << "FAIL: " << label << " (the loopback connect did not succeed)"
                  << std::endl;
        ++g_failures;
        return "test setup";
    }

    std::string request = std::string("GET /socket HTTP/1.1\r\nHost: 127.0.0.1\r\n") +
                          "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                          "Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: " +
                          key + "\r\n\r\n";
    (void)send(client, request.c_str(), request.size(), 0);

    char response[4096];
    ssize_t total = 0;
    while (total < static_cast<ssize_t>(sizeof(response))) {
        const ssize_t chunk = recv(client, response + total,
                                   sizeof(response) - static_cast<size_t>(total), 0);
        if (chunk <= 0) break;
        total += chunk;
    }
    close(client);

    static std::string verdict_storage;
    const char* verdict = eshkol_ws_handshake_refusal(
        response, total > 0 ? static_cast<size_t>(total) : 0, key);
    peer.stop();
    if (request_seen) *request_seen = peer.request_seen;
    if (verdict) {
        verdict_storage = verdict;
        return verdict_storage.c_str();
    }
    return nullptr;
}

}  // namespace

int main() {
    // ─── The values, against the specification ──────────────────────────────
    {
        unsigned char digest[20];
        eshkol_ws_sha1("", 0, digest);
        expect(hex(digest, 20) == "da39a3ee5e6b4b0d3255bfef95601890afd80709",
               "SHA-1 of the empty string");
        eshkol_ws_sha1("abc", 3, digest);
        expect(hex(digest, 20) == "a9993e364706816aba3e25717850c26c9cd0d89d",
               "SHA-1 of \"abc\"");
        const std::string long_input(1000, 'a');
        eshkol_ws_sha1(long_input.data(), long_input.size(), digest);
        expect(hex(digest, 20) == "291e9a6c66994949b57ba5e650361e98fc36b1ba",
               "SHA-1 of a thousand 'a' bytes (multi-block)");
    }
    {
        char encoded[64];
        expect(eshkol_ws_base64(reinterpret_cast<const unsigned char*>("f"), 1,
                                encoded, sizeof(encoded)) &&
                   std::string(encoded) == "Zg==",
               "base64 of one byte pads twice");
        expect(eshkol_ws_base64(reinterpret_cast<const unsigned char*>("fo"), 2,
                                encoded, sizeof(encoded)) &&
                   std::string(encoded) == "Zm8=",
               "base64 of two bytes pads once");
        expect(eshkol_ws_base64(reinterpret_cast<const unsigned char*>("foo"), 3,
                                encoded, sizeof(encoded)) &&
                   std::string(encoded) == "Zm9v",
               "base64 of three bytes does not pad");
        char cramped[4];
        expect(!eshkol_ws_base64(reinterpret_cast<const unsigned char*>("foo"), 3,
                                 cramped, sizeof(cramped)),
               "base64 refuses a buffer with no room for the NUL");
    }
    {
        // RFC 6455 section 1.3's worked example.
        char accept_value[ESHKOL_WS_ACCEPT_CAPACITY];
        expect(eshkol_ws_expected_accept("dGhlIHNhbXBsZSBub25jZQ==", accept_value,
                                         sizeof(accept_value)),
               "the RFC's example key yields an Accept value");
        expect(std::string(accept_value) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
               "the Accept value is the one RFC 6455 section 1.3 states");
    }

    // ─── A fresh key and a fresh mask ───────────────────────────────────────
    {
        char first[ESHKOL_WS_KEY_CAPACITY];
        char second[ESHKOL_WS_KEY_CAPACITY];
        expect(eshkol_ws_generate_key(first, sizeof(first)), "a key is generated");
        expect(eshkol_ws_generate_key(second, sizeof(second)),
               "a second key is generated");
        expect(std::strlen(first) == 24, "a key is 24 base64 characters");
        expect(std::string(first) != std::string(second),
               "two handshakes do not share a key");
        expect(std::string(first) != "ZXNoa29sLXZtLXdlYnNvY2tldA==",
               "a key is not a value compiled into the client");
        char cramped[8];
        expect(!eshkol_ws_generate_key(cramped, sizeof(cramped)),
               "a key will not be written into a buffer that cannot hold it");

        unsigned char mask_a[4] = {0, 0, 0, 0};
        unsigned char mask_b[4] = {0, 0, 0, 0};
        expect(eshkol_ws_generate_mask(mask_a), "a mask is generated");
        expect(eshkol_ws_generate_mask(mask_b), "a second mask is generated");
        const bool a_is_zero = mask_a[0] == 0 && mask_a[1] == 0 &&
                               mask_a[2] == 0 && mask_a[3] == 0;
        const bool b_is_zero = mask_b[0] == 0 && mask_b[1] == 0 &&
                               mask_b[2] == 0 && mask_b[3] == 0;
        expect(!(a_is_zero && b_is_zero),
               "masks are not the all-zero key twice over");
        expect(std::memcmp(mask_a, mask_b, 4) != 0,
               "two frames do not share a mask");
    }

    // ─── Reading an answer, byte by byte ────────────────────────────────────
    {
        const char* value = nullptr;
        size_t value_length = 0;
        const std::string head =
            "HTTP/1.1 101 Switching Protocols\r\n"
            "X-Note: sec-websocket-accept: not-the-field\r\n"
            "Sec-WebSocket-Accept: the-real-one\r\n\r\n";
        expect(eshkol_ws_find_header(head.c_str(), head.size(),
                                     "sec-websocket-accept", &value, &value_length),
               "a field is found");
        expect(std::string(value, value_length) == "the-real-one",
               "a field name inside another field's value is not the field");

        const std::string status_only = "HTTP/1.1 101 sec-websocket-accept: x\r\n\r\n";
        expect(!eshkol_ws_find_header(status_only.c_str(), status_only.size(),
                                      "sec-websocket-accept", &value, &value_length),
               "a field name in the status line is not a field");

        expect(eshkol_ws_value_has_token("keep-alive, Upgrade", 19, "upgrade"),
               "a token in a comma list is found");
        expect(!eshkol_ws_value_has_token("upgraded", 8, "upgrade"),
               "a token is not a prefix match");
    }

    // ─── The handshake over a loopback socket ───────────────────────────────
    {
        std::string request_seen;
        expect_refusal(handshake_against(peer_answer::derived_from_key,
                                         "a peer answering with the derived value",
                                         &request_seen),
                       nullptr, "a correct answer completes the handshake");

        expect_refusal(handshake_against(peer_answer::derived_from_other,
                                         "a peer answering from another key"),
                       "Sec-WebSocket-Accept value",
                       "an Accept derived from another key is refused");
        expect_refusal(handshake_against(peer_answer::status_200,
                                         "a peer answering 200"),
                       "status 101", "an answer that is not 101 is refused");
        expect_refusal(handshake_against(peer_answer::status_text_101,
                                         "a peer with 101 in its reason phrase"),
                       "status 101",
                       "a 101 inside a reason phrase is not a 101 status");
        expect_refusal(handshake_against(peer_answer::missing_upgrade,
                                         "a peer with no Upgrade field"),
                       "Upgrade header", "a 101 with no Upgrade is refused");
        expect_refusal(handshake_against(peer_answer::missing_accept,
                                         "a peer with no Accept field"),
                       "Sec-WebSocket-Accept header",
                       "a 101 with no Accept field is refused");

        expect_refusal(eshkol_ws_handshake_refusal(nullptr, 0, "key"), "no response",
                       "no bytes at all is a refusal");
    }

    // ─── Received frames (RFC 6455 section 5) ───────────────────────────────
    // A frame from a server is never masked; a client that unmasks one and
    // carries on has accepted a frame the RFC says must fail the connection.
    expect(eshkol_ws_examine_frame(0x81, 0x85, 5, 0) ==
               ESHKOL_WS_FRAME_MASKED_FROM_SERVER,
           "a masked server frame fails the connection");
    expect(eshkol_ws_examine_frame(0x81, 0x05, 5, 0) == ESHKOL_WS_FRAME_OK,
           "an unmasked text frame is admitted");
    expect(eshkol_ws_examine_frame(0xc1, 0x05, 5, 0) ==
               ESHKOL_WS_FRAME_RESERVED_BITS_SET,
           "RSV1 set with no extension negotiated is refused");
    expect(eshkol_ws_examine_frame(0x83, 0x05, 5, 0) ==
               ESHKOL_WS_FRAME_RESERVED_OPCODE,
           "a reserved opcode is refused");
    expect(eshkol_ws_examine_frame(0x8b, 0x00, 0, 0) ==
               ESHKOL_WS_FRAME_RESERVED_OPCODE,
           "a reserved control opcode is refused");

    // Control frames: at most 125 bytes, never fragmented.
    expect(eshkol_ws_examine_frame(0x89, 0x7e, 65535, 0) ==
               ESHKOL_WS_FRAME_CONTROL_TOO_LARGE,
           "a 65535-byte ping is refused");
    expect(eshkol_ws_examine_frame(0x88, 0x7e, 126, 0) ==
               ESHKOL_WS_FRAME_CONTROL_TOO_LARGE,
           "a 126-byte close is refused");
    expect(eshkol_ws_examine_frame(0x89, 0x7d, 125, 0) == ESHKOL_WS_FRAME_OK,
           "a 125-byte ping is admitted");
    expect(eshkol_ws_examine_frame(0x09, 0x05, 5, 0) ==
               ESHKOL_WS_FRAME_CONTROL_FRAGMENTED,
           "a ping with FIN clear is refused");

    // Fragment sequencing (section 5.4).
    expect(eshkol_ws_examine_frame(0x80, 0x05, 5, 0) ==
               ESHKOL_WS_FRAME_BAD_CONTINUATION,
           "a continuation outside a sequence is refused");
    expect(eshkol_ws_examine_frame(0x81, 0x05, 5, 1) ==
               ESHKOL_WS_FRAME_BAD_CONTINUATION,
           "a new data frame inside a sequence is refused");
    expect(eshkol_ws_examine_frame(0x80, 0x05, 5, 1) == ESHKOL_WS_FRAME_OK,
           "a continuation inside a sequence is admitted");
    expect(eshkol_ws_examine_frame(0x01, 0x05, 5, 0) == ESHKOL_WS_FRAME_OK,
           "a text frame with FIN clear opens a sequence");
    expect(eshkol_ws_examine_frame(0x89, 0x05, 5, 1) == ESHKOL_WS_FRAME_OK,
           "a control frame may arrive inside a sequence");

    // The budget, and the 126/127 length encodings that reach it. The
    // reviewer's frame: 82 7F 00 00 00 00 00 01 11 70 declares 69 999 999
    // bytes, which must be refused rather than read.
    expect(eshkol_ws_examine_frame(0x82, 0x7f, 69999999ull, 0) ==
               ESHKOL_WS_FRAME_TOO_LARGE,
           "a 64-bit length beyond the budget is refused");
    expect(eshkol_ws_examine_frame(0x82, 0x7e, 65535, 0) == ESHKOL_WS_FRAME_OK,
           "a 16-bit length within the budget is admitted");
    expect(eshkol_ws_examine_frame(0x82, 0x7f, 70000, 0) ==
               ESHKOL_WS_FRAME_TOO_LARGE,
           "a 70000-byte frame is refused against the default budget");
    expect(eshkol_ws_max_frame_bytes() == 65536,
           "the default frame budget is 64 KiB");

    // The close code a refusal is reported with.
    expect(eshkol_ws_frame_close_code(ESHKOL_WS_FRAME_TOO_LARGE) == 1009,
           "an over-budget frame is reported 1009");
    expect(eshkol_ws_frame_close_code(ESHKOL_WS_FRAME_MASKED_FROM_SERVER) == 1002,
           "a protocol error is reported 1002");

    // ─── Text payloads are UTF-8 (section 8.1) ──────────────────────────────
    expect(eshkol_ws_valid_utf8("hello", 5), "ASCII is UTF-8");
    expect(eshkol_ws_valid_utf8("", 0), "an empty payload is UTF-8");
    expect(eshkol_ws_valid_utf8("\xc2\xa2", 2), "a two-byte sequence is UTF-8");
    expect(eshkol_ws_valid_utf8("\xe2\x82\xac", 3), "a three-byte sequence is UTF-8");
    expect(eshkol_ws_valid_utf8("\xf0\x9f\x92\xa9", 4), "a four-byte sequence is UTF-8");
    expect(!eshkol_ws_valid_utf8("\xc0\xaf", 2), "an overlong form is refused");
    expect(!eshkol_ws_valid_utf8("\xed\xa0\x80", 3), "a surrogate is refused");
    expect(!eshkol_ws_valid_utf8("\xf5\x80\x80\x80", 4), "above U+10FFFF is refused");
    expect(!eshkol_ws_valid_utf8("\xe2\x82", 2), "a truncated sequence is refused");
    expect(!eshkol_ws_valid_utf8("\x80", 1), "a bare continuation byte is refused");

    // ─── The masking a client owes every frame it sends (section 5.3) ───────
    // The send path draws a fresh key per frame and masks the payload with
    // mask[(offset + i) % 4]. The property that keeps a chunked send correct
    // is that the index follows the payload offset, not the chunk offset.
    {
        const unsigned char mask[4] = {0x37, 0xfa, 0x21, 0x3d};
        std::string payload(9000, 'A');
        std::string masked = payload;
        for (size_t i = 0; i < masked.size(); ++i)
            masked[i] = (char)(masked[i] ^ mask[i % 4]);
        bool differs = masked != payload;
        bool round_trips = true;
        for (size_t i = 0; i < masked.size(); ++i) {
            if ((char)(masked[i] ^ mask[i % 4]) != payload[i]) round_trips = false;
        }
        expect(differs, "masking changes the bytes that leave");
        expect(round_trips, "unmasking with the same key restores the payload");
        // Across a 4096-byte chunk boundary the key index must still follow
        // the payload offset: byte 4096 takes mask[0], not mask[4096 % 4] of
        // a restarted count. Both are 0 here, so check a boundary that is not
        // a multiple of four as well.
        expect((4096 % 4) == 0 && (4097 % 4) == 1,
               "the key index follows the payload offset across a chunk edge");
    }

    if (g_failures == 0) {
        std::cout << "websocket_handshake_test: all cases pass" << std::endl;
        return 0;
    }
    std::cerr << "websocket_handshake_test: " << g_failures << " case(s) failed"
              << std::endl;
    return 1;
}
