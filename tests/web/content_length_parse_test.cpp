#include <iostream>
#include <limits>
#include <string>

#include "eshkol/http_request_utils.h"

namespace {

int expect_true(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    size_t value = 0;

    if (expect_true(eshkol_parse_content_length("0", value) && value == 0,
                    "should parse zero")) return 1;
    if (expect_true(eshkol_parse_content_length(" 42 ", value) && value == 42,
                    "should trim surrounding whitespace")) return 1;
    if (expect_true(!eshkol_parse_content_length("", value),
                    "should reject empty header")) return 1;
    if (expect_true(!eshkol_parse_content_length("abc", value),
                    "should reject non-numeric header")) return 1;
    if (expect_true(!eshkol_parse_content_length("12x", value),
                    "should reject trailing garbage")) return 1;
    if (expect_true(!eshkol_parse_content_length("-1", value),
                    "should reject negative values")) return 1;

    const std::string overflow = std::to_string(std::numeric_limits<unsigned long long>::max()) + "0";
    if (expect_true(!eshkol_parse_content_length(overflow, value),
                    "should reject overflowing values")) return 1;

    std::cout << "PASS" << std::endl;
    return 0;
}
