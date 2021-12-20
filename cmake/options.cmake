option(DEBUG_MODE "Debug mode, default off" OFF)
option(ENABLE_COVERAGE "Enable code coverage report" OFF)
option(ENABLE_PYTHON "Enable python" ON)
option(ENABLE_ASAN "Enable Google Sanitizer to find memory bugs")
option(MS_WHL_LIB_PATH "MindSpore lib path")
option(MS_BACKEND "Compile MindSpore")
option(RUN_TESTCASES "Compile UT")

if(MS_WHL_LIB_PATH)
    message("MindSpore whl lib path:" ${MS_WHL_LIB_PATH})
elseif(MS_BACKEND)
    message("MindSpore backend method:" ${MS_BACKEND})
elseif(MS_BACKEND_HEADER)
    message("MindSpore backend method:" ${MS_BACKEND_HEADER})
elseif(RUN_TESTCASES)
    message("MindSpore Serving Compile UT:" ${RUN_TESTCASES})
elseif()
    message(FATAL_ERROR "Please confirm how to use MindSpore.")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND Linux)
    set(OPTION_CXX_FLAGS "${OPTION_CXX_FLAGS} -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
endif()

if(ENABLE_COVERAGE)
    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -ftest-coverage")
    set(OPTION_CXX_FLAGS "${OPTION_CXX_FLAGS} ${COVERAGE_COMPILER_FLAGS}")
endif()

if(ENABLE_ASAN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(OPTION_CXX_FLAGS "${OPTION_CXX_FLAGS} -fsanitize=address -fsanitize-recover=address \
                              -fno-omit-frame-pointer -fsanitize=undefined")
    else()
        set(OPTION_CXX_FLAGS "${OPTION_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -static-libsan \
                              -fsanitize=undefined")
    endif()
endif()

if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE "Debug")
    add_compile_definitions(MEM_REUSE_DEBUG)
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

if((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR (CMAKE_BUILD_TYPE STREQUAL Release))
    set(PYBIND11_LTO_CXX_FLAGS FALSE)
endif()

if(NOT BUILD_PATH)
    set(BUILD_PATH "${CMAKE_SOURCE_DIR}/build")
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(MS_BUILD_GRPC ON)
endif()

add_compile_definitions(USE_GLOG)
