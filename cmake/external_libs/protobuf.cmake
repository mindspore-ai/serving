set(protobuf_USE_STATIC_LIBS ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
        -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
        -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
else()
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
        -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    if(NOT ENABLE_GLIBCXX)
        set(protobuf_CXXFLAGS "${protobuf_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()
set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.13.0.tar.gz")
    set(SHA256 "ab9b39e7053a6fb06b01bf75fb6ec6a71a1ada5a5f8e2446f927336e97b9e7bb")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz")
    set(SHA256 "9b4ee22c250fe31b16f1a24d61467e40780a3fbb9b91c3b65be2a376ed913a1a")
endif()

set(PROTOBUF_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/protobuf)

mindspore_add_pkg(protobuf
        VER 3.13.0
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_PATH cmake/
        CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
        PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2021-22570.patch
        PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2022-1941.patch)

include_directories(${protobuf_INC})
include_directories(${CMAKE_BINARY_DIR}/proto_py)
add_library(mindspore_serving::protobuf ALIAS protobuf::protobuf)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
# recover original value
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()
