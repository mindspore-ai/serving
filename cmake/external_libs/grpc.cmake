set(grpc_USE_STATIC_LIBS OFF)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 -O2")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
else()
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2 \
  -Dgrpc=mindspore_serving_grpc -Dgrpc_impl=mindspore_serving_grpc_impl -Dgrpc_core=mindspore_serving_grpc_core")
    if(NOT ENABLE_GLIBCXX)
        set(grpc_CXXFLAGS "${grpc_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
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

if(EXISTS ${re2_ROOT}/lib64)
    set(_FINDPACKAGE_RE2_CONFIG_DIR "${re2_ROOT}/lib64/cmake/re2")
else()
    set(_FINDPACKAGE_RE2_CONFIG_DIR "${re2_ROOT}/lib/cmake/re2")
endif()
message("grpc using re2_DIR : " ${_FINDPACKAGE_RE2_CONFIG_DIR})

if(EXISTS ${openssl_ROOT})
    set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${openssl_ROOT}")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/grpc/repository/archive/v1.36.1.tar.gz")
    set(MD5 "90c93203e95e89af5f46738588217057")
else()
    set(REQ_URL "https://github.com/grpc/grpc/archive/v1.36.1.tar.gz")
    set(MD5 "90c93203e95e89af5f46738588217057")
endif()

mindspore_add_pkg(grpc
        VER 1.36.1
        LIBS mindspore_serving_grpc++_reflection mindspore_serving_grpc++ mindspore_serving_grpc mindspore_serving_gpr
        mindspore_serving_upb mindspore_serving_address_sorting
        EXE grpc_cpp_plugin
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/grpc/grpc.patch001
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DBUILD_SHARED_LIBS=ON
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
        -DgRPC_RE2_PROVIDER:STRING=package
        -Dre2_DIR:PATH=${_FINDPACKAGE_RE2_CONFIG_DIR}
        )

include_directories(${grpc_INC})

add_library(mindspore_serving::grpc++ ALIAS grpc::mindspore_serving_grpc++)
add_library(mindspore_serving::grpc++_reflection ALIAS grpc::mindspore_serving_grpc++_reflection)

# link other grpc libs
target_link_libraries(grpc::mindspore_serving_grpc++ INTERFACE grpc::mindspore_serving_grpc grpc::mindspore_serving_gpr
  grpc::mindspore_serving_upb grpc::mindspore_serving_address_sorting)
target_link_libraries(grpc::mindspore_serving_grpc++_reflection INTERFACE grpc::mindspore_serving_grpc++
  grpc::mindspore_serving_grpc grpc::mindspore_serving_gpr grpc::mindspore_serving_upb
  grpc::mindspore_serving_address_sorting)

# modify mindspore macro define
add_compile_definitions(grpc=mindspore_serving_grpc)
add_compile_definitions(grpc_impl=mindspore_serving_grpc_impl)
add_compile_definitions(grpc_core=mindspore_serving_grpc_core)