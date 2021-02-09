set(grpc_USE_STATIC_LIBS ON)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -fvisibility=hidden \
            -D_FORTIFY_SOURCE=2 -O2")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fvisibility=hidden \
            -D_FORTIFY_SOURCE=2 -O2")
else()
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fvisibility=hidden \
            -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2")
endif()

set(grpc_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")


if(EXISTS ${protobuf_ROOT}/lib64)
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib64/cmake/protobuf")
else()
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib/cmake/protobuf")
endif()
message("grpc using Protobuf_DIR : " ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})

if(EXISTS ${absl_ROOT}/lib64)
    set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib64/cmake/absl")
else()
    set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib/cmake/absl")
endif()
message("grpc using absl_DIR : " ${_FINDPACKAGE_ABSL_CONFIG_DIR})

set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if(OPENSSL_ROOT_DIR)
    set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

mindspore_add_pkg(grpc
        VER 1.27.3
        LIBS grpc++ grpc gpr upb address_sorting
        EXE grpc_cpp_plugin
        URL https://github.com/grpc/grpc/archive/v1.27.3.tar.gz
        MD5 0c6c3fc8682d4262dd0e5e6fabe1a7e2
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DgRPC_INSTALL:BOOL=ON
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:PATH=${zlib_ROOT}
        -DgRPC_ABSL_PROVIDER:STRING=package
        -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR}
        -DgRPC_CARES_PROVIDER:STRING=package
        -Dc-ares_DIR:PATH=${c-ares_ROOT}/lib/cmake/c-ares
        -DgRPC_SSL_PROVIDER:STRING=package
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        )

include_directories(${grpc_INC})

add_library(mindspore_serving::grpc++ ALIAS grpc::grpc++)

# link other grpc libs
target_link_libraries(grpc::grpc++ INTERFACE grpc::grpc grpc::gpr grpc::upb grpc::address_sorting)

# link built dependencies
target_link_libraries(grpc::grpc++ INTERFACE mindspore_serving::z)
target_link_libraries(grpc::grpc++ INTERFACE mindspore_serving::cares)
target_link_libraries(grpc::grpc++ INTERFACE mindspore_serving::absl_strings mindspore_serving::absl_throw_delegate
        mindspore_serving::absl_raw_logging_internal mindspore_serving::absl_int128
        mindspore_serving::absl_bad_optional_access)

# link system openssl
find_package(OpenSSL REQUIRED)
target_link_libraries(grpc::grpc++ INTERFACE OpenSSL::SSL OpenSSL::Crypto)
