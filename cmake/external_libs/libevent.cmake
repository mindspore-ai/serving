set(libevent_CFLAGS "-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
set(libevent_LDFLAGS "-Wl,-z,now")
mindspore_add_pkg(libevent
        VER 2.1.12
        LIBS event event_pthreads event_core
        URL https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz
        MD5 b5333f021f880fe76490d8a799cd79f4
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_TESTING=OFF)

include_directories(${libevent_INC}) # 将指定目录添加到编译器的头文件搜索路径之下，指定的目录被解释成当前源码路径的相对路径。

add_library(mindspore_serving::event ALIAS libevent::event)
add_library(mindspore_serving::event_pthreads ALIAS libevent::event_pthreads)
add_library(mindspore_serving::event_core ALIAS libevent::event_core)
