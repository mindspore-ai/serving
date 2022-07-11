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

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.13.0.tar.gz")
    set(MD5 "53ab10736257b3c61749de9800b8ce97")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz")
    set(MD5 "1a6274bc4a65b55a6fa70e264d796490")
endif()

set(PROTOBUF_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/protobuf)

mindspore_add_pkg(protobuf
        VER 3.13.0
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_PATH cmake/
        CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
        PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2021-22570.patch)

include_directories(${protobuf_INC})
add_library(mindspore_serving::protobuf ALIAS protobuf::protobuf)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()
