mindspore_add_pkg(openssl
  VER 1.1.1
  LIBS ssl crypto
  URL https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1k.tar.gz
  MD5 bdd51a68ad74618dd2519da8e0bcc759
  CONFIGURE_COMMAND ./config no-zlib no-shared)
include_directories(${openssl_INC})