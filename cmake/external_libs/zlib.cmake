if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/zlib/repository/archive/v1.2.11.tar.gz")
    set(SHA256 "f21b3885cc7732f0ab93dbe06ff1ec58069bb58657b3fda89531d1562d8ad708")
else()
    set(REQ_URL "https://github.com/madler/zlib/archive/v1.2.11.tar.gz")
    set(SHA256 "629380c90a77b964d896ed37163f5c3a34f6e6d897311f1df2a7016355c45eff")
endif()

mindspore_add_pkg(zlib
        VER 1.2.11
        LIBS z
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/zlib/CVE-2018-25032.patch
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/zlib/CVE-2022-37434.patch)

include_directories(${zlib_INC})
add_library(mindspore_serving::z ALIAS zlib::z)
