/* Output bytes and their terminator must fit the actual arena allocation.
 * Checking only string contents misses writes into unused bump-arena space. */
#include "../../lib/backend/vm_string.c"

static int cases = 0;

static int check_case(const unsigned char* input, int input_len,
                      const unsigned char* expected, int expected_len,
                      int expected_chars, int upper) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);
    VmString* source = vm_string_new(&rs, (const char*)input, input_len);
    VmString* result = upper ? vm_string_upcase(&rs, source)
                             : vm_string_downcase(&rs, source);
    int failed = 0;
    size_t available = 0;
    if (result) {
        const uintptr_t address = (uintptr_t)result->data;
        for (VmArenaBlock* block = vm_active_arena(&rs)->current;
             block; block = block->next) {
            const uintptr_t start = (uintptr_t)block->data;
            if (address >= start && address - start <= block->used) {
                available = block->used - (size_t)(address - start);
                break;
            }
        }
    }
    if (!result || result->byte_len < 0 ||
        (size_t)result->byte_len + 1 > available) {
        fprintf(stderr, "FAIL: %s input=%d encoded output exceeds allocated arena extent (%d bytes, %zu available)\n",
                upper ? "upcase" : "downcase", input_len,
                result ? result->byte_len : -1, available);
        failed = 1;
    } else if (result->byte_len != expected_len ||
               result->char_len != expected_chars ||
               memcmp(result->data, expected, (size_t)expected_len) != 0 ||
               result->data[expected_len] != '\0') {
        fprintf(stderr, "FAIL: %s input=%d returned incorrect bytes/length/terminator\n",
                upper ? "upcase" : "downcase", input_len);
        failed = 1;
    }
    vm_region_stack_destroy(&rs);
    if (!failed) ++cases;
    return failed;
}

int main(void) {
    const unsigned char replacement[] = {0xef, 0xbf, 0xbd};
    for (int byte = 0; byte < 256; ++byte) {
        unsigned char input = (unsigned char)byte;
        for (int upper = 0; upper < 2; ++upper) {
            unsigned char ascii = input;
            if (upper && ascii >= 'a' && ascii <= 'z') ascii -= 'a' - 'A';
            if (!upper && ascii >= 'A' && ascii <= 'Z') ascii += 'a' - 'A';
            if (check_case(&input, 1, byte < 128 ? &ascii : replacement,
                           byte < 128 ? 1 : 3, 1, upper)) return 1;
        }
    }
    unsigned char invalid[64], replacements[192];
    memset(invalid, 0xff, sizeof(invalid));
    for (size_t i = 0; i < sizeof(invalid); ++i)
        memcpy(replacements + 3 * i, replacement, 3);
    for (int upper = 0; upper < 2; ++upper)
        if (check_case(invalid, 64, replacements, 192, 64, upper)) return 1;

    const unsigned char mixed[] = {'a', 0xff, 'Z', 0xc3, 0xa9, 0};
    const unsigned char mixed_upper[] = {'A', 0xef, 0xbf, 0xbd, 'Z', 0xc3, 0xa9, 0};
    const unsigned char mixed_lower[] = {'a', 0xef, 0xbf, 0xbd, 'z', 0xc3, 0xa9, 0};
    if (check_case(mixed, 6, mixed_upper, 8, 5, 1) ||
        check_case(mixed, 6, mixed_lower, 8, 5, 0)) return 1;
    const unsigned char valid[] = {'a', 0xc3, 0xa9, 0xf0, 0x9f, 0x98, 0x80, 'Z'};
    const unsigned char valid_upper[] = {'A', 0xc3, 0xa9, 0xf0, 0x9f, 0x98, 0x80, 'Z'};
    const unsigned char valid_lower[] = {'a', 0xc3, 0xa9, 0xf0, 0x9f, 0x98, 0x80, 'z'};
    if (check_case(valid, 8, valid_upper, 8, 4, 1) ||
        check_case(valid, 8, valid_lower, 8, 4, 0)) return 1;
    const unsigned char empty[] = {0};
    const unsigned char truncated[] = {0xe2, 0x82};
    const unsigned char overlong[] = {0xc0, 0xaf};
    const unsigned char twice[] = {0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd};
    for (int upper = 0; upper < 2; ++upper)
        if (check_case(empty, 0, empty, 0, 0, upper) ||
            check_case(truncated, 2, replacement, 3, 1, upper) ||
            check_case(overlong, 2, twice, 6, 2, upper)) return 1;
    printf("PASS: UTF-8 case capacity and byte oracle (%d cases)\n", cases);
    return 0;
}
