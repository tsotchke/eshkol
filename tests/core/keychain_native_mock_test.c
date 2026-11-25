#include "../../lib/core/keychain_native.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecItem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", \
                    __FILE__, __LINE__, #condition); \
            abort(); \
        } \
    } while (0)

/* Constants normally exported by Security.framework. The bridge test links
 * CoreFoundation only and supplies these values plus mocked SecItem calls. */
const CFStringRef kSecClass = CFSTR("class");
const CFStringRef kSecClassGenericPassword = CFSTR("generic-password");
const CFStringRef kSecAttrService = CFSTR("service");
const CFStringRef kSecAttrAccount = CFSTR("account");
const CFStringRef kSecValueData = CFSTR("value-data");
const CFStringRef kSecReturnData = CFSTR("return-data");
const CFStringRef kSecReturnAttributes = CFSTR("return-attributes");
const CFStringRef kSecMatchLimit = CFSTR("match-limit");
const CFStringRef kSecMatchLimitOne = CFSTR("match-one");
const CFStringRef kSecUseAuthenticationUI = CFSTR("auth-ui");
const CFStringRef kSecUseAuthenticationUIFail = CFSTR("auth-ui-fail");

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static CFStringRef mock_auth_ui_key(void) { return kSecUseAuthenticationUI; }
static CFStringRef mock_auth_ui_fail(void) { return kSecUseAuthenticationUIFail; }
#pragma clang diagnostic pop

static int allow_keychain = 1;
static int copy_calls, add_calls, update_calls, delete_calls;
static OSStatus copy_status = errSecSuccess;
static OSStatus update_status = errSecSuccess;
static OSStatus add_status = errSecSuccess;
static OSStatus delete_status = errSecSuccess;
static int invalid_copy_type;
static const UInt8 *stored_bytes;
static CFIndex stored_length;
static CFTypeRef last_copy_result;
static const UInt8 *expected_account;
static size_t expected_account_len;
static const UInt8 *expected_value;
static size_t expected_value_len;

int eshkol_capability_require(const char *name) {
    CHECK(strcmp(name, "keychain") == 0);
    return allow_keychain;
}

static void check_common_query(CFDictionaryRef query) {
    CHECK(CFEqual(CFDictionaryGetValue(query, kSecClass),
                   kSecClassGenericPassword));
    CHECK(CFEqual(CFDictionaryGetValue(query, kSecAttrService),
                   CFSTR("eshkol-agent")));
    CFStringRef actual_account = CFDictionaryGetValue(query, kSecAttrAccount);
    CHECK(actual_account != NULL);
    CFIndex character_count = CFStringGetLength(actual_account);
    UInt8 account_bytes[16];
    CFIndex used_bytes = 0;
    CHECK(character_count <= (CFIndex)sizeof(account_bytes));
    CHECK(CFStringGetBytes(actual_account, CFRangeMake(0, character_count),
                           kCFStringEncodingUTF8, 0, false, account_bytes,
                           sizeof(account_bytes), &used_bytes) == character_count);
    CHECK(used_bytes == (CFIndex)expected_account_len);
    CHECK(memcmp(account_bytes, expected_account, expected_account_len) == 0);
    CHECK(CFEqual(CFDictionaryGetValue(query, mock_auth_ui_key()),
                  mock_auth_ui_fail()));
}

static void check_value_data(CFTypeRef value) {
    CHECK(value != NULL);
    CHECK(CFGetTypeID(value) == CFDataGetTypeID());
    CFDataRef data = (CFDataRef)value;
    CHECK(CFDataGetLength(data) == (CFIndex)expected_value_len);
    if (expected_value_len > 0)
        CHECK(memcmp(CFDataGetBytePtr(data), expected_value,
                     expected_value_len) == 0);
}

OSStatus SecItemCopyMatching(CFDictionaryRef query, CFTypeRef *result) {
    copy_calls++;
    check_common_query(query);
    CHECK(CFEqual(CFDictionaryGetValue(query, kSecMatchLimit),
                   kSecMatchLimitOne));
    if (copy_status != errSecSuccess) return copy_status;
    if (CFDictionaryGetValue(query, kSecReturnData) == kCFBooleanTrue) {
        CHECK(CFDictionaryGetValue(query, kSecReturnAttributes) == NULL);
        if (invalid_copy_type)
            *result = CFStringCreateWithCString(kCFAllocatorDefault,
                                                "wrong type",
                                                kCFStringEncodingUTF8);
        else
            *result = CFDataCreate(kCFAllocatorDefault, stored_bytes,
                                   stored_length);
    } else {
        CHECK(CFDictionaryGetValue(query, kSecReturnAttributes) == kCFBooleanTrue);
        CHECK(CFDictionaryGetValue(query, kSecReturnData) == NULL);
        *result = CFDictionaryCreate(kCFAllocatorDefault, NULL, NULL, 0,
                                     &kCFTypeDictionaryKeyCallBacks,
                                     &kCFTypeDictionaryValueCallBacks);
    }
    if (last_copy_result) CFRelease(last_copy_result);
    last_copy_result = CFRetain(*result);
    return errSecSuccess;
}

OSStatus SecItemUpdate(CFDictionaryRef query, CFDictionaryRef attributes) {
    update_calls++;
    check_common_query(query);
    CHECK(CFDictionaryGetValue(query, kSecValueData) == NULL);
    check_value_data(CFDictionaryGetValue(attributes, kSecValueData));
    if (add_status == errSecDuplicateItem && update_calls == 4)
        return errSecSuccess; /* update after the duplicate-add race */
    return update_status;
}

OSStatus SecItemAdd(CFDictionaryRef attributes, CFTypeRef *result) {
    (void)result;
    add_calls++;
    check_common_query(attributes);
    check_value_data(CFDictionaryGetValue(attributes, kSecValueData));
    return add_status;
}

OSStatus SecItemDelete(CFDictionaryRef query) {
    delete_calls++;
    check_common_query(query);
    CHECK(CFDictionaryGetValue(query, kSecValueData) == NULL);
    return delete_status;
}

static void reset_counts(void) {
    copy_calls = add_calls = update_calls = delete_calls = 0;
}

static void check_and_release_last_copy_result(void) {
    CHECK(last_copy_result != NULL);
    CHECK(CFGetRetainCount(last_copy_result) == 1);
    CFRelease(last_copy_result);
    last_copy_result = NULL;
}

int main(void) {
    static const UInt8 secret[] = {' ', '\t', 'A', 0, '\n', ' '};
    static const UInt8 account[] = {'a', 0, 'c'};
    expected_account = account;
    expected_account_len = sizeof(account);
    expected_value = secret;
    expected_value_len = sizeof(secret);
    UInt8 output[16] = {0};
    stored_bytes = secret;
    stored_length = (CFIndex)sizeof(secret);

    allow_keychain = 0;
    reset_counts();
    CHECK(eshkol_keychain_size(account, sizeof(account)) < 0);
    CHECK(eshkol_keychain_get_into(account, sizeof(account), output,
                                    sizeof(secret)) < 0);
    CHECK(!eshkol_keychain_set(account, sizeof(account), secret,
                                sizeof(secret)));
    CHECK(!eshkol_keychain_delete(account, sizeof(account)));
    CHECK(!eshkol_keychain_has(account, sizeof(account)));
    CHECK(copy_calls + add_calls + update_calls + delete_calls == 0);
    allow_keychain = 1;

    reset_counts();
    CHECK(eshkol_keychain_size(account, sizeof(account)) == sizeof(secret));
    check_and_release_last_copy_result();
    CHECK(eshkol_keychain_get_into(account, sizeof(account), output,
                                    sizeof(secret)) == sizeof(secret));
    CHECK(memcmp(output, secret, sizeof(secret)) == 0);
    CHECK(copy_calls == 2);

    int64_t probed = eshkol_keychain_size(account, sizeof(account));
    stored_length = sizeof(secret) - 1;
    memset(output, 0x5a, sizeof(output));
    CHECK(eshkol_keychain_get_into(account, sizeof(account), output,
                                    probed) < 0);
    for (size_t i = 0; i < sizeof(output); ++i) CHECK(output[i] == 0x5a);
    stored_length = sizeof(secret);

    stored_length = 0;
    CHECK(eshkol_keychain_size(account, sizeof(account)) == 0);
    CHECK(eshkol_keychain_get_into(account, sizeof(account), NULL, 0) == 0);
    stored_bytes = secret;
    stored_length = sizeof(secret);
    static const UInt8 oversized[1024 * 1024 + 1] = {0};
    stored_bytes = oversized;
    stored_length = sizeof(oversized);
    CHECK(eshkol_keychain_size(account, sizeof(account)) < 0);
    stored_bytes = secret;
    stored_length = sizeof(secret);

    CHECK(eshkol_keychain_get_into(account, sizeof(account), output,
                                    sizeof(secret) - 1) < 0);
    CHECK(eshkol_keychain_size(NULL, sizeof(account)) < 0);
    CHECK(eshkol_keychain_size(account, -1) < 0);
    CHECK(eshkol_keychain_size(account, 1024 * 1024 + 1) < 0);
    CHECK(!eshkol_keychain_set(account, sizeof(account), NULL, 0));
    CHECK(!eshkol_keychain_set(account, sizeof(account), secret,
                                1024 * 1024 + 1));

    invalid_copy_type = 1;
    CHECK(eshkol_keychain_size(account, sizeof(account)) < 0);
    memset(output, 0x3c, sizeof(output));
    CHECK(eshkol_keychain_get_into(account, sizeof(account), output,
                                    sizeof(secret)) < 0);
    for (size_t i = 0; i < sizeof(output); ++i) CHECK(output[i] == 0x3c);
    invalid_copy_type = 0;
    copy_status = errSecItemNotFound;
    CHECK(eshkol_keychain_size(account, sizeof(account)) < 0);
    copy_status = errSecInteractionNotAllowed;
    CHECK(eshkol_keychain_size(account, sizeof(account)) < 0);
    copy_status = errSecSuccess;

    update_status = errSecSuccess;
    CHECK(eshkol_keychain_set(account, sizeof(account), secret,
                               sizeof(secret)));
    CHECK(update_calls == 1 && add_calls == 0);
    update_status = errSecItemNotFound;
    add_status = errSecSuccess;
    CHECK(eshkol_keychain_set(account, sizeof(account), secret,
                               sizeof(secret)));
    CHECK(add_calls == 1 && update_calls == 2);
    add_status = errSecDuplicateItem;
    CHECK(eshkol_keychain_set(account, sizeof(account), secret,
                               sizeof(secret)));
    CHECK(add_calls == 2 && update_calls == 4);
    add_status = errSecParam;
    CHECK(!eshkol_keychain_set(account, sizeof(account), secret,
                                sizeof(secret)));
    add_status = errSecSuccess;
    expected_value_len = 0;
    CHECK(eshkol_keychain_set(account, sizeof(account), secret, 0));
    expected_value_len = sizeof(secret);

    delete_status = errSecSuccess;
    CHECK(eshkol_keychain_delete(account, sizeof(account)));
    delete_status = errSecItemNotFound;
    CHECK(!eshkol_keychain_delete(account, sizeof(account)));
    delete_status = errSecSuccess;
    CHECK(eshkol_keychain_has(account, sizeof(account)));
    CHECK(!eshkol_keychain_has(NULL, sizeof(account)));

    check_and_release_last_copy_result();

    puts("keychain native mock tests passed");
    return 0;
}
