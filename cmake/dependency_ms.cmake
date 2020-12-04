# Compile MindSpore

message(STATUS "**********begin to compile MindSpore**********")
set(MS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/mindspore)
message(STATUS "MindSpore dir: ${MS_SOURCE_DIR}")
message(STATUS "MindSpore compile method: -e${MS_BACKEND}")
message(STATUS "MindSpore compile thread num: -j${THREAD_NUM}")
message(STATUS "MindSpore version: -V${MS_VERSION}")

if(MS_VERSION)
set(MS_VERSION_OPTION -V${MS_VERSION})
endif()

execute_process(
        COMMAND bash ${MS_SOURCE_DIR}/build.sh -e${MS_BACKEND} ${MS_VERSION_OPTION} -j${THREAD_NUM}
        WORKING_DIRECTORY ${MS_SOURCE_DIR}
)
message(STATUS "**********end to compile MindSpore**********")