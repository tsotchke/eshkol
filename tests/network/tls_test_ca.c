/* Generate a local test PKI under a build directory; no network access.
 *
 *   ca.pem                     trusted test certificate authority
 *   server-cert.pem/-key.pem   leaf for DNS:localhost and IP:127.0.0.1, issued by ca.pem
 *   rogue-cert.pem/-key.pem    leaf for the same names, issued by an unrelated authority
 *
 * Usage: tls_test_ca <directory> */
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

static void die(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    ERR_print_errors_fp(stderr);
    exit(1);
}
static EVP_PKEY *make_key(void) {
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, NULL);
    EVP_PKEY *key = NULL;
    if (!ctx || EVP_PKEY_keygen_init(ctx) <= 0 ||
        EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, NID_X9_62_prime256v1) <= 0 ||
        EVP_PKEY_keygen(ctx, &key) <= 0) die("generate key");
    EVP_PKEY_CTX_free(ctx);
    return key;
}
static void add_ext(X509 *cert, X509 *issuer, int nid, const char *value) {
    X509V3_CTX context;
    X509V3_set_ctx(&context, issuer, cert, NULL, NULL, 0);
    X509_EXTENSION *extension = X509V3_EXT_conf_nid(NULL, &context, nid, value);
    if (!extension || X509_add_ext(cert, extension, -1) != 1) die("add extension");
    X509_EXTENSION_free(extension);
}
/* issuer == NULL makes a self-signed authority. */
static X509 *make_cert(EVP_PKEY *key, const char *common_name, long serial,
                       X509 *issuer, EVP_PKEY *issuer_key) {
    X509 *cert = X509_new();
    if (!cert || X509_set_version(cert, 2) != 1 ||
        ASN1_INTEGER_set(X509_get_serialNumber(cert), serial) != 1 ||
        !X509_gmtime_adj(X509_getm_notBefore(cert), -300) ||
        !X509_gmtime_adj(X509_getm_notAfter(cert), 86400) ||
        X509_set_pubkey(cert, key) != 1) die("build certificate");
    X509_NAME *name = X509_get_subject_name(cert);
    if (X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                                   (const unsigned char *)common_name, -1, -1, 0) != 1)
        die("subject name");
    X509 *signer = issuer ? issuer : cert;
    if (X509_set_issuer_name(cert, X509_get_subject_name(signer)) != 1) die("issuer name");
    if (issuer) {
        add_ext(cert, signer, NID_basic_constraints, "critical,CA:FALSE");
        add_ext(cert, signer, NID_key_usage, "critical,digitalSignature");
        add_ext(cert, signer, NID_ext_key_usage, "serverAuth");
        add_ext(cert, signer, NID_subject_alt_name, "DNS:localhost,IP:127.0.0.1");
    } else {
        add_ext(cert, signer, NID_basic_constraints, "critical,CA:TRUE");
        add_ext(cert, signer, NID_key_usage, "critical,keyCertSign,cRLSign");
    }
    add_ext(cert, signer, NID_subject_key_identifier, "hash");
    if (X509_sign(cert, issuer_key ? issuer_key : key, EVP_sha256()) <= 0) die("sign");
    return cert;
}
static void write_cert(const char *dir, const char *file, X509 *cert) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *out = fopen(path, "wb");
    if (!out || PEM_write_X509(out, cert) != 1 || fclose(out) != 0) die(path);
}
static void write_key(const char *dir, const char *file, EVP_PKEY *key) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *out = fopen(path, "wb");
    if (!out || PEM_write_PrivateKey(out, key, NULL, NULL, 0, NULL, NULL) != 1 ||
        fclose(out) != 0) die(path);
}
int main(int argc, char **argv) {
    if (argc != 2 || !argv[1][0]) { fprintf(stderr, "usage: tls_test_ca <directory>\n"); return 2; }
    const char *dir = argv[1];
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) die("create directory");
    EVP_PKEY *ca_key = make_key(), *rogue_ca_key = make_key();
    EVP_PKEY *server_key = make_key(), *rogue_key = make_key();
    X509 *ca = make_cert(ca_key, "Eshkol Test Authority", 1, NULL, NULL);
    X509 *rogue_ca = make_cert(rogue_ca_key, "Unrelated Authority", 2, NULL, NULL);
    X509 *server = make_cert(server_key, "localhost", 3, ca, ca_key);
    X509 *rogue = make_cert(rogue_key, "localhost", 4, rogue_ca, rogue_ca_key);
    write_cert(dir, "ca.pem", ca);
    write_cert(dir, "server-cert.pem", server);
    write_key(dir, "server-key.pem", server_key);
    write_cert(dir, "rogue-cert.pem", rogue);
    write_key(dir, "rogue-key.pem", rogue_key);
    X509_free(ca); X509_free(rogue_ca); X509_free(server); X509_free(rogue);
    EVP_PKEY_free(ca_key); EVP_PKEY_free(rogue_ca_key);
    EVP_PKEY_free(server_key); EVP_PKEY_free(rogue_key);
    printf("PASS: test PKI written to %s\n", dir);
    return 0;
}
