set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS} -Dgoogle=mindspore_serving_private")
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(glog_patch ${CMAKE_SOURCE_DIR}/third_party/patch/glog/glog.patch001)
set(glog_lib mindspore_serving_glog)

if(NOT ENABLE_GLIBCXX)
    set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/glog/repository/archive/v0.4.0.tar.gz")
    set(SHA256 "e17cd4bb7c06951a12fc9db5130ec63a9f090b84340b8556fa0d530f73c6b634")
else()
    set(REQ_URL "https://github.com/google/glog/archive/v0.4.0.tar.gz")
    set(SHA256 "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c")
endif()

set(glog_option -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=OFF
        -DCMAKE_BUILD_TYPE=Release)

if(WIN32 AND NOT MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(glog_option ${glog_option} -DHAVE_DBGHELP=ON)
    endif()
endif()

mindspore_add_pkg(glog
        VER 0.4.0
        LIBS ${glog_lib}
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${glog_patch}
        CMAKE_OPTION ${glog_option})
include_directories(${glog_INC})
add_library(mindspore_serving::glog ALIAS glog::${glog_lib})
