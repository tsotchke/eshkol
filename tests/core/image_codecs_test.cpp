/**
 * @file image_codecs_test.cpp
 * @brief Gate for the runtime's built-in image codecs.
 *
 * The golden values below were recorded from the platform codec libraries
 * (libpng 1.6, libjpeg-turbo 2.1, libwebp 1.5) that image I/O used before its
 * codecs moved into the runtime:
 *   - decode goldens: dimensions/channels and an FNV-1a hash of the 8-bit
 *     samples image-read produced for each fixture in tests/core/image_fixtures;
 *   - encode goldens: size and FNV-1a hash of the file image-write produced
 *     for deterministic gradient and block images (PNG and JPEG).
 * Matching them shows the built-in codecs keep pixels and file bytes
 * identical.  The test also round-trips every format and channel count
 * (WebP losslessly), exercises the zlib encoder/decoder pair, and runs
 * a deterministic mutation pass over the fixtures (every input must return).
 *
 * Usage: image_codecs_test <fixture-dir> <scratch-dir>
 */

#include "../../inc/eshkol/core/image_io.h"
#include "../../lib/core/image_codecs.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL: " << what << "\n";
    }
}

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_all(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> to_bytes(const double* d, size_t n) {
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(std::lround(d[i] * 255.0));
    return out;
}

struct DecodeGolden {
    const char* file;
    int w, h, c;
    uint64_t hash;
};

const DecodeGolden kDecode[] = {
    {"x_png_ct3_d4_17x13_i1_t1.png", 17, 13, 4, 0xd76c1f14caf6c7d1ull},
    {"x_png_ct6_d16_64x33_i1_t0.png", 64, 33, 4, 0xf8f28e963fc0361eull},
    {"x_png_ct0_d2_17x13_i0_t1.png", 17, 13, 4, 0x8d3e93d2efd2ccc5ull},
    {"x_png_ct2_d16_17x13_i0_t1.png", 17, 13, 4, 0x3def6c4d0382885aull},
    {"x_png_ct4_d8_64x33_i0_t0.png", 64, 33, 4, 0x9daf0840eb755e16ull},
    {"x_png_ct0_d1_129x257_i1_t0.png", 129, 257, 3, 0x9ac0db51606476f5ull},
    {"x_jpg_33x31_f2_p1_q90_r0_o0.jpg", 33, 31, 3, 0xe414a0606dfc2558ull},
    {"x_jpg_100x75_f0_p0_q90_r3_o0.jpg", 100, 75, 3, 0xcb2a3772848362cbull},
    {"x_jpg_17x9_f1_p1_q90_r0_o0.jpg", 17, 9, 3, 0x3574889fe3465d4full},
    {"x_jpg_17x9_f3_p0_q50_r0_o1.jpg", 17, 9, 3, 0x2bb12aee90fbce97ull},
    {"x_jpg_100x75_f4_p1_q100_r3_o1.jpg", 100, 75, 3, 0x4c3e879b89b9bd9bull},
    {"x_jpg_17x9_f6_p1_q50_r3_o1.jpg", 17, 9, 3, 0x23d301435433fcd9ull},
    {"x_jpg_gray_17x9_p1.jpg", 17, 9, 3, 0x32b02d1c5ffc7346ull},
    {"x_jpg_rgbspace_33x31_p0.jpg", 33, 31, 3, 0x68023ad7319c7ef0ull},
    {"x_jpg_nojfif_33x31_p1.jpg", 33, 31, 3, 0xb2b2fa6902a66057ull},
    {"bad_trunc_5.jpg", 200, 150, 3, 0xd4088ba4cec71282ull},
    {"bad_trunc_0.png", 0, 0, 0, 0ull},
    {"bad_flip_0.png", 0, 0, 0, 0ull},
    {"webp_g0010_1x1_k1_a1_l0_q100_m4.webp", 1, 1, 4, 0xab619dcc29b8979eull},
    {"webp_g0170_64x48_k0_a2_l0_q75_m5.webp", 64, 48, 4, 0xbba075da5931ba0cull},
    {"webp_g0177_64x48_k1_a0_l0_q75_m6.webp", 64, 48, 3, 0x84b443c7dedb8f5dull},
    {"webp_g0191_64x48_k3_a2_l0_q75_m1.webp", 64, 48, 4, 0x3b734aea338f10a1ull},
    {"webp_g0174_64x48_k0_a2_l1_q100_m1.webp", 64, 48, 4, 0xcdee68c84ee86963ull},
    {"webp_g0186_64x48_k2_a2_l1_q0_m1.webp", 64, 48, 4, 0x03e5be7814922db5ull},
    {"webp_g0195_64x48_k3_a2_l1_q100_m3.webp", 64, 48, 4, 0xa40e02c8d94ddafaull},
    {"webp_g0246_257x129_k3_a0_l0_q40_m1.webp", 257, 129, 3, 0x0d65a479b7e9831aull},
    {"webp_pil_rgb_icc_1.webp", 5, 9, 3, 0xcf22ab01c7adc89bull},
    {"webp_pil_rgba_lossy_2.webp", 64, 64, 4, 0x8064ea9e9259051bull},
    {"webp_pil_webp_lossless_13x7.webp", 13, 7, 3, 0x07556fe15328f737ull},
    {"webp_bad_m00.webp", 0, 0, 0, 0ull},
};

struct EncodeGolden {
    const char* kind;
    int w, h, c;
    const char* fmt;
    size_t size;
    uint64_t hash;
};

const EncodeGolden kEncode[] = {
    {"grad", 1, 1, 1, "png", 67u, 0x7ae17f10cc03a510ull},
    {"grad", 1, 1, 1, "jpeg", 333u, 0xa1877d06a363d4a5ull},
    {"blocks", 1, 1, 1, "png", 67u, 0x2e4b4a19eeea6680ull},
    {"blocks", 1, 1, 1, "jpeg", 333u, 0x1c6c108e46970fa3ull},
    {"grad", 1, 1, 3, "png", 69u, 0x2cb7e8eb4dba9e9cull},
    {"grad", 1, 1, 3, "jpeg", 634u, 0xe03d16119f56159eull},
    {"blocks", 1, 1, 3, "png", 69u, 0x7ab8ad04fe99af9full},
    {"blocks", 1, 1, 3, "jpeg", 634u, 0x59d6f7f3ae3740f8ull},
    {"grad", 1, 1, 4, "png", 70u, 0x970f9eeb54fe0ed6ull},
    {"grad", 1, 1, 4, "jpeg", 634u, 0xe03d16119f56159eull},
    {"blocks", 1, 1, 4, "png", 70u, 0xe9a48e2023e3cbe7ull},
    {"blocks", 1, 1, 4, "jpeg", 634u, 0x59d6f7f3ae3740f8ull},
    {"grad", 5, 1, 1, "png", 71u, 0x6aa9eba9c06dd62aull},
    {"grad", 5, 1, 1, "jpeg", 337u, 0x2f8751952c0f70d5ull},
    {"blocks", 5, 1, 1, "png", 71u, 0x2ea57cfe261135edull},
    {"blocks", 5, 1, 1, "jpeg", 348u, 0x3942dd9e5d90c07eull},
    {"grad", 5, 1, 3, "png", 71u, 0x65651dbb99b2b8c0ull},
    {"grad", 5, 1, 3, "jpeg", 638u, 0x65729314c7af8349ull},
    {"blocks", 5, 1, 3, "png", 75u, 0x1b58a2f7ed89b8e4ull},
    {"blocks", 5, 1, 3, "jpeg", 668u, 0xda9dad8f63f27c85ull},
    {"grad", 5, 1, 4, "png", 72u, 0x19616278e5461f17ull},
    {"grad", 5, 1, 4, "jpeg", 638u, 0x65729314c7af8349ull},
    {"blocks", 5, 1, 4, "png", 81u, 0xbd38e95224e693faull},
    {"blocks", 5, 1, 4, "jpeg", 668u, 0xda9dad8f63f27c85ull},
    {"grad", 7, 5, 1, "png", 79u, 0xd4434145711aff60ull},
    {"grad", 7, 5, 1, "jpeg", 340u, 0x3f0c6a9e9f737410ull},
    {"blocks", 7, 5, 1, "png", 89u, 0x8d35d7e58b473b16ull},
    {"blocks", 7, 5, 1, "jpeg", 352u, 0x7617a5100e80880cull},
    {"grad", 7, 5, 3, "png", 78u, 0xa37b77d45aafc06bull},
    {"grad", 7, 5, 3, "jpeg", 641u, 0xf53ef3846aee26c7ull},
    {"blocks", 7, 5, 3, "png", 107u, 0xeaeddc1aaee685ceull},
    {"blocks", 7, 5, 3, "jpeg", 673u, 0x68f0b0ff66412816ull},
    {"grad", 7, 5, 4, "png", 80u, 0x1ccd3e3abcc37d8aull},
    {"grad", 7, 5, 4, "jpeg", 641u, 0xf53ef3846aee26c7ull},
    {"blocks", 7, 5, 4, "png", 121u, 0xa07cbd58d75d0b00ull},
    {"blocks", 7, 5, 4, "jpeg", 673u, 0x68f0b0ff66412816ull},
    {"grad", 17, 17, 1, "png", 78u, 0xa767ffecdf1ba8baull},
    {"grad", 17, 17, 1, "jpeg", 392u, 0x9b1db7c66343a61bull},
    {"blocks", 17, 17, 1, "png", 210u, 0x88210122a54dbc2dull},
    {"blocks", 17, 17, 1, "jpeg", 599u, 0x2da486e911e480b3ull},
    {"grad", 17, 17, 3, "png", 84u, 0x6a2988407fa3642eull},
    {"grad", 17, 17, 3, "jpeg", 700u, 0xb8c2a6debe3ef753ull},
    {"blocks", 17, 17, 3, "png", 385u, 0x0ba5e67e69e9dc9full},
    {"blocks", 17, 17, 3, "jpeg", 976u, 0xee5041c70b9a5b09ull},
    {"grad", 17, 17, 4, "png", 128u, 0xd63f6aac699e2335ull},
    {"grad", 17, 17, 4, "jpeg", 700u, 0xb8c2a6debe3ef753ull},
    {"blocks", 17, 17, 4, "png", 476u, 0xb31df0f756bf8bcaull},
    {"blocks", 17, 17, 4, "jpeg", 976u, 0xee5041c70b9a5b09ull},
    {"grad", 33, 9, 1, "png", 77u, 0x8efd6d7106aac9ddull},
    {"grad", 33, 9, 1, "jpeg", 394u, 0x7d951839c4dd1a40ull},
    {"blocks", 33, 9, 1, "png", 195u, 0xbf4d0b70f7866b0dull},
    {"blocks", 33, 9, 1, "jpeg", 600u, 0xb27893161432937cull},
    {"grad", 33, 9, 3, "png", 84u, 0x959bd2f0b5cb6c85ull},
    {"grad", 33, 9, 3, "jpeg", 718u, 0xf4cfaf454f1e7223ull},
    {"blocks", 33, 9, 3, "png", 318u, 0x902aeab455989d37ull},
    {"blocks", 33, 9, 3, "jpeg", 978u, 0xb97029214ddca3c0ull},
    {"grad", 33, 9, 4, "png", 120u, 0x22550a1b4c1a133cull},
    {"grad", 33, 9, 4, "jpeg", 718u, 0xf4cfaf454f1e7223ull},
    {"blocks", 33, 9, 4, "png", 405u, 0x747acdc416275367ull},
    {"blocks", 33, 9, 4, "jpeg", 978u, 0xb97029214ddca3c0ull},
    {"grad", 255, 3, 1, "png", 86u, 0xc38f05b93be05f2full},
    {"grad", 255, 3, 1, "jpeg", 706u, 0x726484391c0d47a3ull},
    {"blocks", 255, 3, 1, "png", 236u, 0x0aafddfd5b4de3fbull},
    {"blocks", 255, 3, 1, "jpeg", 1581u, 0x355b04f6ceea1dd8ull},
    {"grad", 255, 3, 3, "png", 109u, 0x05b39910bb95277bull},
    {"grad", 255, 3, 3, "jpeg", 1486u, 0x553c394192d41c0bull},
    {"blocks", 255, 3, 3, "png", 532u, 0xc73392de4132c02eull},
    {"blocks", 255, 3, 3, "jpeg", 2155u, 0xb951c2efbb39455bull},
    {"grad", 255, 3, 4, "png", 122u, 0x7509cf0e3916e65full},
    {"grad", 255, 3, 4, "jpeg", 1486u, 0x553c394192d41c0bull},
    {"blocks", 255, 3, 4, "png", 682u, 0x47edd881ee4c762bull},
    {"blocks", 255, 3, 4, "jpeg", 2155u, 0xb951c2efbb39455bull},
    {"grad", 90, 300, 1, "png", 769u, 0x0859129a72b4eb64ull},
    {"grad", 90, 300, 1, "jpeg", 7915u, 0x097bec9f1312921bull},
    {"blocks", 90, 300, 1, "png", 11211u, 0x14425a5dcaeb8ae7ull},
    {"blocks", 90, 300, 1, "jpeg", 22800u, 0xe29d2391899db3b5ull},
    {"grad", 90, 300, 3, "png", 1562u, 0xb6afeec065f48605ull},
    {"grad", 90, 300, 3, "jpeg", 14398u, 0x9518914f30003952ull},
    {"blocks", 90, 300, 3, "png", 28189u, 0x441b8501f2f99cafull},
    {"blocks", 90, 300, 3, "jpeg", 27007u, 0xe0cdde472701e8fbull},
    {"grad", 90, 300, 4, "png", 1914u, 0xe2d40863b821b008ull},
    {"grad", 90, 300, 4, "jpeg", 14398u, 0x9518914f30003952ull},
    {"blocks", 90, 300, 4, "png", 36696u, 0x773d27b53e5b94faull},
    {"blocks", 90, 300, 4, "jpeg", 27007u, 0xe0cdde472701e8fbull},
    {"grad", 640, 480, 1, "png", 4541u, 0x697579f0007d1a6bull},
    {"grad", 640, 480, 1, "jpeg", 82796u, 0x24f77580ddba35d3ull},
    {"blocks", 640, 480, 1, "png", 144135u, 0x6c9327e03c974102ull},
    {"blocks", 640, 480, 1, "jpeg", 244724u, 0x60e08e89d07f3c46ull},
    {"grad", 640, 480, 3, "png", 9771u, 0xf872a8139b3cd42eull},
    {"grad", 640, 480, 3, "jpeg", 149723u, 0xc3b19e5c5f1c7213ull},
    {"blocks", 640, 480, 3, "png", 320738u, 0xe16ddf05aa67711dull},
    {"blocks", 640, 480, 3, "jpeg", 283186u, 0x3ac4fe6b822dedacull},
    {"grad", 640, 480, 4, "png", 13990u, 0xfd2a910c5db79594ull},
    {"grad", 640, 480, 4, "jpeg", 149723u, 0xc3b19e5c5f1c7213ull},
    {"blocks", 640, 480, 4, "png", 442789u, 0xf8c47aabcf74aba9ull},
    {"blocks", 640, 480, 4, "jpeg", 283186u, 0x3ac4fe6b822dedacull},
};

uint8_t sample(const char* kind, int x, int y, int k) {
    if (std::strcmp(kind, "grad") == 0) return static_cast<uint8_t>((x * 3 + y * 5 + k * 60) & 255);
    return static_cast<uint8_t>(((x / 3 + y / 5 + k) % 3 == 0) ? 255 : ((x * y) & 255));
}

std::vector<double> make_image(const char* kind, int w, int h, int c) {
    std::vector<double> d(static_cast<size_t>(w) * h * c);
    size_t i = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int k = 0; k < c; ++k) d[i++] = sample(kind, x, y, k) / 255.0;
    return d;
}

void test_decode_goldens(const std::string& fixtures) {
    for (const DecodeGolden& g : kDecode) {
        int w = 0, h = 0, c = 0;
        const std::string path = fixtures + "/" + g.file;
        double* d = eshkol_image_read(path.c_str(), &w, &h, &c);
        if (g.c == 0) {
            check(d == nullptr, std::string("malformed fixture is refused: ") + g.file);
            continue;
        }
        check(d != nullptr, std::string("decodes: ") + g.file);
        if (!d) continue;
        check(w == g.w && h == g.h && c == g.c, std::string("shape matches reference: ") + g.file);
        const std::vector<uint8_t> px = to_bytes(d, static_cast<size_t>(w) * h * c);
        check(fnv1a(px.data(), px.size()) == g.hash, std::string("pixels match reference: ") + g.file);
    }
}

void test_encode_goldens(const std::string& scratch) {
    for (const EncodeGolden& g : kEncode) {
        const std::vector<double> img = make_image(g.kind, g.w, g.h, g.c);
        const std::string path = scratch + "/golden." + g.fmt;
        const int rc = eshkol_image_write(path.c_str(), img.data(), g.w, g.h, g.c, g.fmt);
        const std::string label = std::string(g.kind) + " " + std::to_string(g.w) + "x" + std::to_string(g.h) +
                                  "x" + std::to_string(g.c) + " " + g.fmt;
        check(rc == 0, "encodes: " + label);
        const std::vector<uint8_t> bytes = read_all(path);
        check(bytes.size() == g.size && fnv1a(bytes.data(), bytes.size()) == g.hash,
              "file bytes match reference: " + label);
        std::remove(path.c_str());
    }
}

void test_round_trips(const std::string& scratch) {
    const int sizes[][2] = {{1, 1}, {3, 2}, {31, 17}};
    for (const auto& s : sizes) {
        for (int c : {1, 3, 4}) {
            const std::vector<double> img = make_image("blocks", s[0], s[1], c);
            for (const char* fmt : {"png", "bmp", "webp", "jpeg"}) {
                const std::string path = scratch + "/rt." + fmt;
                const std::string label = std::string(fmt) + " " + std::to_string(s[0]) + "x" +
                                          std::to_string(s[1]) + "x" + std::to_string(c);
                check(eshkol_image_write(path.c_str(), img.data(), s[0], s[1], c, fmt) == 0, "writes " + label);
                int w = 0, h = 0, oc = 0;
                double* d = eshkol_image_read(path.c_str(), &w, &h, &oc);
                check(d && w == s[0] && h == s[1], "reads back " + label);
                const bool lossless = std::strcmp(fmt, "jpeg") != 0;
                // WebP, like libwebp, keeps an alpha channel only when some
                // sample is transparent; the "blocks" alpha is.
                const int expect_c = c == 4 && lossless ? 4 : 3;
                check(oc == expect_c, "channel count " + label);
                if (d && lossless && oc == expect_c) {
                    bool same = true;
                    for (int i = 0; i < w * h && same; ++i) {
                        for (int k = 0; k < oc; ++k) {
                            const int src_k = c == 1 ? 0 : k;
                            const double want = img[static_cast<size_t>(i) * c + src_k];
                            same = same && std::fabs(d[static_cast<size_t>(i) * oc + k] - want) < 1e-12;
                        }
                    }
                    check(same, "lossless samples " + label);
                }
                std::remove(path.c_str());
            }
        }
    }
}

void test_zlib() {
    uint32_t state = 99;
    auto next = [&]() {
        state = state * 1664525u + 1013904223u;
        return state >> 24;
    };
    for (size_t size : {size_t{0}, size_t{1}, size_t{300}, size_t{70000}, size_t{200000}}) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            std::vector<uint8_t> data(size);
            for (size_t i = 0; i < size; ++i) {
                data[i] = pattern == 0 ? static_cast<uint8_t>(next())
                          : pattern == 1 ? static_cast<uint8_t>(i % 251)
                                         : static_cast<uint8_t>((i / 1000) & 3);
            }
            for (int strategy : {0, 1}) {
                eshkol::image::DeflateEncoder enc(15, strategy);
                size_t off = 0;
                while (off < size) {
                    const size_t chunk = std::min<size_t>(size - off, 1 + next() * 37);
                    enc.write(data.data() + off, chunk);
                    off += chunk;
                }
                enc.finish();
                std::vector<uint8_t> back;
                const std::vector<uint8_t>& z = enc.output();
                const bool ok = eshkol::image::zlib_inflate_exact(z.data(), z.size(), size, &back);
                check(ok && back == data, "zlib round trip size=" + std::to_string(size) +
                                              " pattern=" + std::to_string(pattern) +
                                              " strategy=" + std::to_string(strategy));
                if (size > 0) {
                    const uint32_t adler = eshkol::image::adler32_update(1, data.data(), data.size());
                    const size_t n = z.size();
                    const uint32_t stored = (uint32_t{z[n - 4]} << 24) | (uint32_t{z[n - 3]} << 16) |
                                            (uint32_t{z[n - 2]} << 8) | z[n - 1];
                    check(adler == stored, "zlib trailer carries Adler-32");
                }
            }
        }
    }
}

void test_mutations(const std::string& fixtures, const std::string& scratch) {
    uint32_t state = 7;
    auto next = [&]() {
        state = state * 1103515245u + 12345u;
        return (state >> 8) & 0xFFFFFFu;
    };
    int decoded = 0;
    for (const DecodeGolden& g : kDecode) {
        const std::vector<uint8_t> original = read_all(fixtures + "/" + g.file);
        if (original.empty()) continue;
        for (int round = 0; round < 40; ++round) {
            std::vector<uint8_t> bytes = original;
            const int edits = 1 + static_cast<int>(next() % 4);
            for (int e = 0; e < edits; ++e) bytes[next() % bytes.size()] = static_cast<uint8_t>(next());
            if (round % 5 == 0) bytes.resize(next() % bytes.size() + 1);
            const std::string path = scratch + "/mutated.bin";
            write_all(path, bytes);
            int w = 0, h = 0, c = 0;
            if (eshkol_image_read(path.c_str(), &w, &h, &c)) {
                ++decoded;
                check(w > 0 && h > 0 && (c == 3 || c == 4), "mutated input decodes to a valid shape");
            }
            std::remove(path.c_str());
        }
    }
    std::cout << "mutation pass: " << decoded << " of mutated inputs still decoded, all returned\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: image_codecs_test <fixture-dir> <scratch-dir>\n";
        return 2;
    }
    const std::string fixtures = argv[1];
    const std::string scratch = argv[2];
    test_decode_goldens(fixtures);
    test_encode_goldens(scratch);
    test_round_trips(scratch);
    test_zlib();
    test_mutations(fixtures, scratch);
    if (failures == 0) {
        std::cout << "PASS: image codecs (" << std::size(kDecode) << " decode goldens, " << std::size(kEncode)
                  << " encode goldens)\n";
        return 0;
    }
    std::cout << failures << " check(s) failed\n";
    return 1;
}
