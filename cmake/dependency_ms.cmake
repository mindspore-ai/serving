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

set(EXEC_COMMAND bash ${MS_SOURCE_DIR}/build.sh -e${MS_BACKEND} ${MS_VERSION_OPTION} -j${THREAD_NUM})
execute_process(
        COMMAND ${EXEC_COMMAND}
        WORKING_DIRECTORY ${MS_SOURCE_DIR}
        RESULT_VARIABLE RESULT
)
if(NOT RESULT EQUAL "0")
    message(FATAL_ERROR "error! when ${EXEC_COMMAND} in ${MS_SOURCE_DIR}")
endif()

message(STATUS "**********end to compile MindSpore**********")