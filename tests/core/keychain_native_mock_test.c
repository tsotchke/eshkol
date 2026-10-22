#include "../../lib/core/keychain_native.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecItem.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

int eshkol_capability_require(const char *name) {
    assert(strcmp(name, "keychain") == 0);
    return allow_keychain;
}

static void check_common_query(CFDictionaryRef query) {
    assert(CFEqual(CFDictionaryGetValue(query, kSecClass),
                   kSecClassGenericPassword));
    assert(CFEqual(CFDictionaryGetValue(query, kSecAttrService),
                   CFSTR("eshkol-agent")));
    assert(CFDictionaryGetValue(query, kSecAttrAccount) != NULL);
    assert(CFEqual(CFDictionaryGetValue(query, kSecUseAuthenticationUI),
                   kSecUseAuthenticationUIFail));
}

OSStatus SecItemCopyMatching(CFDictionaryRef query, CFTypeRef *result) {
    copy_calls++;
    check_common_query(query);
    assert(CFEqual(CFDictionaryGetValue(query, kSecMatchLimit),
                   kSecMatchLimitOne));
    if (copy_status != errSecSuccess) return copy_status;
    if (CFDictionaryGetValue(query, kSecReturnData) == kCFBooleanTrue) {
        assert(CFDictionaryGetValue(query, kSecReturnAttributes) == NULL);
        if (invalid_copy_type)
            *result = CFStringCreateWithCString(kCFAllocatorDefault,
                                                "wrong type",
                                                kCFStringEncodingUTF8);
        else
            *result = CFDataCreate(kCFAllocatorDefault, stored_bytes,
                                   stored_length);
    } else {
        assert(CFDictionaryGetValue(query, kSecReturnAttributes) == kCFBooleanTrue);
        assert(CFDictionaryGetValue(query, kSecReturnData) == NULL);
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
    assert(CFDictionaryGetValue(query, kSecValueData) == NULL);
    assert(CFDictionaryGetValue(attributes, kSecValueData) != NULL);
    if (add_status == errSecDuplicateItem && update_calls == 4)
        return errSecSuccess; /* update after the duplicate-add race */
    return update_status;
}

OSStatus SecItemAdd(CFDictionaryRef attributes, CFTypeRef *result) {
    (void)result;
    add_calls++;
    check_common_query(attributes);
    assert(CFDictionaryGetValue(attributes, kSecValueData) != NULL);
    return add_status;
}

OSStatus SecItemDelete(CFDictionaryRef query) {
    delete_calls++;
    check_common_query(query);
    assert(CFDictionaryGetValue(query, kSecValueData) == NULL);
    return delete_status;
}

static void reset_counts(void) {
    copy_calls = add_calls = update_calls = delete_calls = 0;
}

static void assert_and_release_last_copy_result(void) {
    assert(last_copy_result != NULL);
    assert(CFGetRetainCount(last_copy_result) == 1);
    CFRelease(last_copy_result);
    last_copy_result = NULL;
}

int main(void) {
    static const UInt8 secret[] = {' ', '\t', 'A', 0, '\n', ' '};
    static const UInt8 account[] = {'a', 0, 'c'};
    UInt8 output[16] = {0};
    stored_bytes = secret;
    stored_length = (CFIndex)sizeof(secret);

    allow_keychain = 0;
    reset_counts();
    assert(eshkol_keychain_size(account, sizeof(account)) < 0);
    assert(eshkol_keychain_get_into(account, sizeof(account), output,
                                    sizeof(secret)) < 0);
    assert(!eshkol_keychain_set(account, sizeof(account), secret,
                                sizeof(secret)));
    assert(!eshkol_keychain_delete(account, sizeof(account)));
    assert(!eshkol_keychain_has(account, sizeof(account)));
    assert(copy_calls + add_calls + update_calls + delete_calls == 0);
    allow_keychain = 1;

    reset_counts();
    assert(eshkol_keychain_size(account, sizeof(account)) == sizeof(secret));
    assert_and_release_last_copy_result();
    assert(eshkol_keychain_get_into(account, sizeof(account), output,
                                    sizeof(secret)) == sizeof(secret));
    assert(memcmp(output, secret, sizeof(secret)) == 0);
    assert(copy_calls == 2);

    int64_t probed = eshkol_keychain_size(account, sizeof(account));
    stored_length = sizeof(secret) - 1;
    assert(eshkol_keychain_get_into(account, sizeof(account), output,
                                    probed) < 0);
    stored_length = sizeof(secret);

    stored_length = 0;
    assert(eshkol_keychain_size(account, sizeof(account)) == 0);
    assert(eshkol_keychain_get_into(account, sizeof(account), NULL, 0) == 0);
    stored_bytes = secret;
    stored_length = sizeof(secret);
    static const UInt8 oversized[1024 * 1024 + 1] = {0};
    stored_bytes = oversized;
    stored_length = sizeof(oversized);
    assert(eshkol_keychain_size(account, sizeof(account)) < 0);
    stored_bytes = secret;
    stored_length = sizeof(secret);

    assert(eshkol_keychain_get_into(account, sizeof(account), output,
                                    sizeof(secret) - 1) < 0);
    assert(eshkol_keychain_size(NULL, sizeof(account)) < 0);
    assert(eshkol_keychain_size(account, -1) < 0);
    assert(eshkol_keychain_size(account, 1024 * 1024 + 1) < 0);
    assert(!eshkol_keychain_set(account, sizeof(account), NULL, 0));
    assert(!eshkol_keychain_set(account, sizeof(account), secret,
                                1024 * 1024 + 1));

    invalid_copy_type = 1;
    assert(eshkol_keychain_size(account, sizeof(account)) < 0);
    invalid_copy_type = 0;
    copy_status = errSecItemNotFound;
    assert(eshkol_keychain_size(account, sizeof(account)) < 0);
    copy_status = errSecInteractionNotAllowed;
    assert(eshkol_keychain_size(account, sizeof(account)) < 0);
    copy_status = errSecSuccess;

    update_status = errSecSuccess;
    assert(eshkol_keychain_set(account, sizeof(account), secret,
                               sizeof(secret)));
    assert(update_calls == 1 && add_calls == 0);
    update_status = errSecItemNotFound;
    add_status = errSecSuccess;
    assert(eshkol_keychain_set(account, sizeof(account), secret,
                               sizeof(secret)));
    assert(add_calls == 1 && update_calls == 2);
    add_status = errSecDuplicateItem;
    assert(eshkol_keychain_set(account, sizeof(account), secret,
                               sizeof(secret)));
    assert(add_calls == 2 && update_calls == 4);
    add_status = errSecParam;
    assert(!eshkol_keychain_set(account, sizeof(account), secret,
                                sizeof(secret)));
    add_status = errSecSuccess;

    delete_status = errSecSuccess;
    assert(eshkol_keychain_delete(account, sizeof(account)));
    delete_status = errSecItemNotFound;
    assert(!eshkol_keychain_delete(account, sizeof(account)));
    delete_status = errSecSuccess;
    assert(eshkol_keychain_has(account, sizeof(account)));
    assert(!eshkol_keychain_has(NULL, sizeof(account)));

    assert_and_release_last_copy_result();

    puts("keychain native mock tests passed");
    return 0;
}
