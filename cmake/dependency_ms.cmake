# Compile MindSpore

message(STATUS "**********begin to compile MindSpore**********")
set(MS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/mindspore)
message(STATUS "MindSpore dir: ${MS_SOURCE_DIR}")
message(STATUS "MindSpore compile method: -e${MS_BACKEND}")
message(STATUS "MindSpore compile thread num: -j${THREAD_NUM}")

execute_process(
        COMMAND bash ${MS_SOURCE_DIR}/build.sh -e${MS_BACKEND} -j${THREAD_NUM}
        WORKING_DIRECTORY ${MS_SOURCE_DIR}
)
message(STATUS "**********end to compile MindSpore**********")