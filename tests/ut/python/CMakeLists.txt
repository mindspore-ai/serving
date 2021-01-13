# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")

add_library(protobuf::libprotobuf ALIAS protobuf::protobuf)
add_executable(protobuf::libprotoc ALIAS protobuf::protoc)

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
if (CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
else ()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif ()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
if (EXISTS ${grpc_ROOT}/lib64)
    set(gRPC_DIR "${grpc_ROOT}/lib64/cmake/grpc")
else ()
    set(gRPC_DIR "${grpc_ROOT}/lib/cmake/grpc")
endif ()
message("serving using grpc_DIR : " ${gPRC_DIR})

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
set(_REFLECTION gRPC::grpc++_reflection)

if (CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
    find_program(_GRPC_PYTHON_PLUGIN_EXECUTABLE grpc_python_plugin)
else ()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    set(_GRPC_PYTHON_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_python_plugin>)
endif ()

# Proto file

# Generated sources
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/proto/")
file(GLOB_RECURSE PROTO_FILE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/mindspore_serving/proto/ FOLLOW_SYMLINKS *.proto)
foreach (proto_file ${PROTO_FILE_LIST})
    message(------proto file: ${proto_file})
    get_filename_component(proto_I_DIR "./" ABSOLUTE)
    get_filename_component(proto_file_absolute "mindspore_serving/proto/${proto_file}" ABSOLUTE)
    set(proto_file_relative "mindspore_serving/proto/${proto_file}")
    string(REGEX REPLACE .proto "" proto_file_prefix ${proto_file})

    set(protoc_output_prefix ${CMAKE_CURRENT_BINARY_DIR}/mindspore_serving/proto)
    set(hw_proto_srcs "${protoc_output_prefix}/${proto_file_prefix}.pb.cc")
    set(hw_proto_hdrs "${protoc_output_prefix}/${proto_file_prefix}.pb.h")
    set(hw_grpc_srcs "${protoc_output_prefix}/${proto_file_prefix}.grpc.pb.cc")
    set(hw_grpc_hdrs "${protoc_output_prefix}/${proto_file_prefix}.grpc.pb.h")
    set(hw_py_pb2 "${protoc_output_prefix}/${proto_file_prefix}_pb2.py")
    set(hw_py_pb2_grpc "${protoc_output_prefix}/${proto_file_prefix}_pb2_grpc.py")
    add_custom_command(
            OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}" "${hw_py_pb2}" "${hw_py_pb2_grpc}"
            WORKING_DIRECTORY ${proto_I_DIR}
            COMMAND ${_PROTOBUF_PROTOC}
            ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
            --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
            -I "${proto_I_DIR}"
            --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
            "${proto_file_relative}"
            COMMAND ${_PROTOBUF_PROTOC}
            ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
            --python_out "${CMAKE_CURRENT_BINARY_DIR}"
            -I "${proto_I_DIR}"
            --plugin=protoc-gen-grpc="${_GRPC_PYTHON_PLUGIN_EXECUTABLE}"
            "${proto_file_relative}"
            DEPENDS "${proto_file_absolute}")

    list(APPEND PROTO_SRC_LIST ${hw_proto_srcs} ${hw_grpc_srcs})
endforeach (proto_file)

include_directories("${CMAKE_CURRENT_BINARY_DIR}/mindspore_serving") # for proto header file
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# serving_common for c++ server and python interface
file(GLOB_RECURSE UT_SERVING_CORE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore_serving/ccsrc/master/*.cc"
        "../../../mindspore_serving/ccsrc/common/*.cc"
        "../../../mindspore_serving/ccsrc/worker/*.cc")

file(GLOB_RECURSE UT_SERVING_STUB RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "../cpp/stub/*.cc")
set(UT_SERVING_COMMON ${UT_SERVING_CORE_SRC} ${UT_SERVING_STUB} ${PROTO_SRC_LIST})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/stub)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/stub/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore_serving/ccsrc)

link_directories(${CMKAE_BINARY_DIR}/securec/src)

# python ut
add_compile_definitions(ENABLE_PYTHON)
file(GLOB_RECURSE PY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "../../../mindspore_serving/ccsrc/python/*.cc")

find_package(Python3 3.7 COMPONENTS Interpreter Development)
if (Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
else ()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif ()

include_directories(${PYTHON_INCLUDE_DIRS})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/mindspore_serving)
pybind11_add_module(_mindspore_serving ${PY_SRC_LIST} ${UT_SERVING_COMMON})
target_link_libraries(_mindspore_serving PRIVATE "${PYTHON_LIBRARIES}")
target_include_directories(_mindspore_serving PRIVATE ${pybind11_INCLUDE_DIRS})

target_link_libraries(_mindspore_serving PRIVATE mindspore_serving::event mindspore_serving::event_pthreads)
target_link_libraries(_mindspore_serving PRIVATE ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} pthread mindspore::glog)
target_link_libraries(_mindspore_serving PRIVATE ${SECUREC_LIBRARY})

# disable auto rpath
set_target_properties(_mindspore_serving PROPERTIES SKIP_BUILD_RPATH TRUE RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# copy libevent lib
file(GLOB_RECURSE LIBEVENT_LIB_LIST ${libevent_LIBPATH}/libevent* libevent_LIBPATH}/libevent_pthreads*)
file(COPY ${LIBEVENT_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# copy glog lib
file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libglog*)
file(COPY ${GLOG_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
