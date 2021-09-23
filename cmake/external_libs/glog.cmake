set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/glog/repository/archive/v0.4.0.tar.gz")
    set(MD5 "1e63d5e35a607bade6380a61ae61bceb")
else()
    set(REQ_URL "https://github.com/google/glog/archive/v0.4.0.tar.gz")
    set(MD5 "0daea8785e6df922d7887755c3d100d0")
endif()
mindspore_add_pkg(glog
        VER 0.4.0
        LIBS glog
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/glog/glog.patch001
        CMAKE_OPTION -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=OFF)
include_directories(${glog_INC})
add_library(mindspore_serving::glog ALIAS glog::glog)
