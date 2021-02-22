# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")

# python ut
add_compile_definitions(ENABLE_PYTHON)
file(GLOB_RECURSE PY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "../../../mindspore_serving/ccsrc/python/*.cc")

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
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/mindspore_serving)
pybind11_add_module(_mindspore_serving ${PY_SRC_LIST})
target_link_libraries(_mindspore_serving PRIVATE serving_common)
target_link_libraries(_mindspore_serving PRIVATE "${PYTHON_LIBRARIES}")
target_include_directories(_mindspore_serving PRIVATE ${pybind11_INCLUDE_DIRS})

target_link_options(_mindspore_serving PRIVATE -Wl,-init,mindspore_serving_log_init)

# disable auto rpath
set_target_properties(_mindspore_serving PROPERTIES SKIP_BUILD_RPATH TRUE)
