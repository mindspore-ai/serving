set(protobuf_USE_STATIC_LIBS ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
else()
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC  \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2")
endif()

set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

mindspore_add_pkg(protobuf
        VER 3.8.0
        LIBS protobuf
        EXE protoc
        URL https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz
        MD5 3d9e32700639618a4d2d342c99d4507a
        CMAKE_PATH cmake/
        CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF)

include_directories(${protobuf_INC})
add_library(mindspore_serving::protobuf ALIAS protobuf::protobuf)
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
