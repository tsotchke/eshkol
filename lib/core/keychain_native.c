/* Native, bounded bridge for the agent Keychain API. Secret values are copied
 * only between caller-owned memory and CoreFoundation data objects. */
#include "keychain_native.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Kept as a C declaration because runtime_exports.h is a C++ interface. */
extern int eshkol_capability_require(const char *capability);

#define ESHKOL_KEYCHAIN_MAX_BYTES ((int64_t)1024 * 1024)
#define ESHKOL_KEYCHAIN_FAILURE ((int64_t)-1)
#define ESHKOL_KEYCHAIN_UNSUPPORTED ((int64_t)-2)

static int keychain_valid_span(const void *ptr, int64_t len) {
    return ptr != NULL && len >= 0 && len <= ESHKOL_KEYCHAIN_MAX_BYTES;
}

static int keychain_allowed(void) {
    return eshkol_capability_require("keychain") != 0;
}

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecItem.h>

/* This deprecated constant is the SDK-supported way to make every SecItem
 * operation fail instead of presenting authentication UI. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static CFStringRef keychain_no_ui(void) {
    return kSecUseAuthenticationUIFail;
}
#pragma clang diagnostic pop

static CFStringRef keychain_account_string(const void *account,
                                           int64_t account_len) {
    if (!keychain_valid_span(account, account_len)) return NULL;
    return CFStringCreateWithBytes(kCFAllocatorDefault,
                                   (const UInt8 *)account,
                                   (CFIndex)account_len,
                                   kCFStringEncodingUTF8, false);
}

static CFMutableDictionaryRef keychain_query(CFStringRef account) {
    if (!account) return NULL;
    CFMutableDictionaryRef query = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    if (!query) return NULL;
    CFDictionarySetValue(query, kSecClass, kSecClassGenericPassword);
    CFDictionarySetValue(query, kSecAttrService, CFSTR("eshkol-agent"));
    CFDictionarySetValue(query, kSecAttrAccount, account);
    CFDictionarySetValue(query, kSecUseAuthenticationUI, keychain_no_ui());
    return query;
}

static int keychain_copy_data(const void *account, int64_t account_len,
                              CFDataRef *data_out) {
    *data_out = NULL;
    CFStringRef account_string = keychain_account_string(account, account_len);
    if (!account_string) return 0;
    CFMutableDictionaryRef query = keychain_query(account_string);
    CFRelease(account_string);
    if (!query) return 0;
    CFDictionarySetValue(query, kSecReturnData, kCFBooleanTrue);
    CFDictionarySetValue(query, kSecMatchLimit, kSecMatchLimitOne);

    CFTypeRef result = NULL;
    OSStatus status = SecItemCopyMatching(query, &result);
    CFRelease(query);
    if (status != errSecSuccess || !result) {
        if (result) CFRelease(result);
        return 0;
    }
    if (CFGetTypeID(result) != CFDataGetTypeID()) {
        CFRelease(result);
        return 0;
    }
    *data_out = (CFDataRef)result;
    return 1;
}

int64_t eshkol_keychain_size(const void *account, int64_t account_len) {
    if (!keychain_allowed() || !keychain_valid_span(account, account_len))
        return ESHKOL_KEYCHAIN_FAILURE;
    CFDataRef data = NULL;
    if (!keychain_copy_data(account, account_len, &data))
        return ESHKOL_KEYCHAIN_FAILURE;
    CFIndex size = CFDataGetLength(data);
    CFRelease(data);
    if (size < 0 || (int64_t)size > ESHKOL_KEYCHAIN_MAX_BYTES)
        return ESHKOL_KEYCHAIN_FAILURE;
    return (int64_t)size;
}

int64_t eshkol_keychain_get_into(const void *account, int64_t account_len,
                                 void *output, int64_t output_capacity) {
    if (!keychain_allowed() || !keychain_valid_span(account, account_len) ||
        output_capacity < 0 || output_capacity > ESHKOL_KEYCHAIN_MAX_BYTES ||
        (!output && output_capacity != 0))
        return ESHKOL_KEYCHAIN_FAILURE;
    CFDataRef data = NULL;
    if (!keychain_copy_data(account, account_len, &data))
        return ESHKOL_KEYCHAIN_FAILURE;
    CFIndex size = CFDataGetLength(data);
    if (size < 0 || (int64_t)size > ESHKOL_KEYCHAIN_MAX_BYTES ||
        (int64_t)size != output_capacity) {
        CFRelease(data);
        return ESHKOL_KEYCHAIN_FAILURE;
    }
    if (size > 0) memcpy(output, CFDataGetBytePtr(data), (size_t)size);
    CFRelease(data);
    return (int64_t)size;
}

int eshkol_keychain_set(const void *account, int64_t account_len,
                        const void *value, int64_t value_len) {
    if (!keychain_allowed() || !keychain_valid_span(account, account_len) ||
        !keychain_valid_span(value, value_len)) return 0;
    CFStringRef account_string = keychain_account_string(account, account_len);
    if (!account_string) return 0;
    CFMutableDictionaryRef query = keychain_query(account_string);
    CFRelease(account_string);
    if (!query) return 0;
    CFDataRef value_data = CFDataCreate(kCFAllocatorDefault,
                                        (const UInt8 *)value,
                                        (CFIndex)value_len);
    if (!value_data) { CFRelease(query); return 0; }
    CFMutableDictionaryRef attributes = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    if (!attributes) {
        CFRelease(value_data);
        CFRelease(query);
        return 0;
    }
    CFDictionarySetValue(attributes, kSecValueData, value_data);
    CFRelease(value_data);

    OSStatus status = SecItemUpdate(query, attributes);
    if (status == errSecItemNotFound) {
        CFMutableDictionaryRef add_query = CFDictionaryCreateMutableCopy(
            kCFAllocatorDefault, 0, query);
        if (!add_query) {
            CFRelease(attributes);
            CFRelease(query);
            return 0;
        }
        CFDictionarySetValue(add_query, kSecValueData,
                             CFDictionaryGetValue(attributes, kSecValueData));
        status = SecItemAdd(add_query, NULL);
        CFRelease(add_query);
        if (status == errSecDuplicateItem) {
            status = SecItemUpdate(query, attributes);
        }
    }
    CFRelease(attributes);
    CFRelease(query);
    return status == errSecSuccess;
}

int eshkol_keychain_delete(const void *account, int64_t account_len) {
    if (!keychain_allowed() || !keychain_valid_span(account, account_len))
        return 0;
    CFStringRef account_string = keychain_account_string(account, account_len);
    if (!account_string) return 0;
    CFMutableDictionaryRef query = keychain_query(account_string);
    CFRelease(account_string);
    if (!query) return 0;
    OSStatus status = SecItemDelete(query);
    CFRelease(query);
    return status == errSecSuccess;
}

int eshkol_keychain_has(const void *account, int64_t account_len) {
    if (!keychain_allowed() || !keychain_valid_span(account, account_len))
        return 0;
    CFStringRef account_string = keychain_account_string(account, account_len);
    if (!account_string) return 0;
    CFMutableDictionaryRef query = keychain_query(account_string);
    CFRelease(account_string);
    if (!query) return 0;
    CFDictionarySetValue(query, kSecReturnAttributes, kCFBooleanTrue);
    CFDictionarySetValue(query, kSecMatchLimit, kSecMatchLimitOne);
    CFTypeRef result = NULL;
    OSStatus status = SecItemCopyMatching(query, &result);
    CFRelease(query);
    if (result) CFRelease(result);
    return status == errSecSuccess;
}

#else
int64_t eshkol_keychain_size(const void *account, int64_t account_len) {
    (void)account; (void)account_len;
    return keychain_allowed() ? ESHKOL_KEYCHAIN_UNSUPPORTED : ESHKOL_KEYCHAIN_FAILURE;
}
int64_t eshkol_keychain_get_into(const void *account, int64_t account_len,
                                 void *output, int64_t output_capacity) {
    (void)account; (void)account_len; (void)output; (void)output_capacity;
    return keychain_allowed() ? ESHKOL_KEYCHAIN_UNSUPPORTED : ESHKOL_KEYCHAIN_FAILURE;
}
int eshkol_keychain_set(const void *account, int64_t account_len,
                        const void *value, int64_t value_len) {
    if (!keychain_allowed()) return 0;
    (void)account; (void)account_len; (void)value; (void)value_len;
    return 0;
}
int eshkol_keychain_delete(const void *account, int64_t account_len) {
    if (!keychain_allowed()) return 0;
    (void)account; (void)account_len;
    return 0;
}
int eshkol_keychain_has(const void *account, int64_t account_len) {
    if (!keychain_allowed()) return 0;
    (void)account; (void)account_len;
    return 0;
}
#endif
