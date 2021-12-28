# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib -Wl,--no-as-needed")

if(ENABLE_COVERAGE)
    add_compile_options(-coverage)
    add_link_options(-lgcov --coverage)
endif()

# Proto file
# Generated sources
file(GLOB_RECURSE PROTO_FILE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./proto/*.proto)
ms_grpc_generate(PROTO_SRC_LIST PROTO_HDR_LIST ${PROTO_FILE_LIST})
add_library(PROTO_SRC_LIB STATIC ${PROTO_SRC_LIST})
target_compile_options(PROTO_SRC_LIB PRIVATE "-Wno-array-bounds")

include_directories("${CMAKE_BINARY_DIR}/mindspore_serving" ${CMAKE_BINARY_DIR}) # for proto header file
include_directories("ccsrc")

# serving_common for c++ server and python interface
file(GLOB_RECURSE SERVING_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ccsrc/master/*.cc" "ccsrc/common/*.cc" "ccsrc/worker/*.cc")

file(GLOB_RECURSE SERVING_ASCEND_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ccsrc/worker/inference/mindspore_model_wrap.cc")

list(REMOVE_ITEM SERVING_SRC ${SERVING_ASCEND_SRC})
add_library(serving_common SHARED ${SERVING_SRC})
add_library(serving_ascend SHARED ${SERVING_ASCEND_SRC})
target_link_libraries(serving_ascend PRIVATE serving_common)
target_link_libraries(serving_ascend PRIVATE ${SECUREC_LIBRARY})

include(CheckPIESupported)
check_pie_supported()
set_property(TARGET serving_common PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_property(TARGET serving_ascend PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(serving_common PRIVATE PROTO_SRC_LIB)
target_link_libraries(serving_common PRIVATE mindspore_serving::grpc++)
target_link_libraries(serving_common PRIVATE mindspore_serving::protobuf pthread rt)
target_link_libraries(serving_common PRIVATE mindspore_serving::event mindspore_serving::event_pthreads)
target_link_libraries(serving_common PRIVATE mindspore_serving::event_core)
target_link_libraries(serving_common PRIVATE mindspore_serving::event_openssl)
target_link_libraries(serving_common PRIVATE mindspore_serving::glog)
target_link_libraries(serving_common PRIVATE mindspore_serving::eigen)
target_link_libraries(serving_common PRIVATE ${SECUREC_LIBRARY})

set_target_properties(serving_common PROPERTIES SKIP_BUILD_RPATH TRUE)

# python
add_compile_definitions(ENABLE_PYTHON)
file(GLOB_RECURSE PY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ccsrc/python/*.cc")

find_package(Python3 3.7 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()

include_directories(${PYTHON_INCLUDE_DIRS})
pybind11_add_module(_mindspore_serving ${PY_SRC_LIST})
target_link_libraries(_mindspore_serving PRIVATE "${PYTHON_LIBRARIES}")
target_include_directories(_mindspore_serving PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(_mindspore_serving PRIVATE serving_common)
set_property(TARGET _mindspore_serving PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_options(serving_common PRIVATE -Wl,-init,mindspore_serving_log_init)

# user set path
if(ENABLE_TESTCASES)
    include_directories(${CMAKE_SOURCE_DIR}/tests/ut/stub)
    target_link_libraries(serving_ascend PRIVATE mindspore)
elseif(MS_WHL_LIB_PATH)
    include_directories(${MS_WHL_LIB_PATH}/../)
elseif(MS_BACKEND_HEADER)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/mindspore)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/mindspore/mindspore/core)
else()
    message(FATAL_ERROR "Please check MindSpore path.")
endif()
