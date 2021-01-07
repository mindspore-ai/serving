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

add_library(UT_PROTO_SRC_LIB STATIC ${PROTO_SRC_LIST})

include_directories("${CMAKE_CURRENT_BINARY_DIR}/mindspore_serving") # for proto header file
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# serving_common for c++ server and python interface
file(GLOB_RECURSE UT_CORE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore_serving/ccsrc/master/*.cc"
        "../../../mindspore_serving/ccsrc/common/*.cc"
        "../../../mindspore_serving/ccsrc/worker/*.cc")

list(APPEND FINAL_SERVING_SRC ${UT_CORE_SRC})
#add_library(serving_common SHARED ${SERVING_SRC})

include(CheckPIESupported)
check_pie_supported()
# set_property(TARGET serving_common PROPERTY POSITION_INDEPENDENT_CODE TRUE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore_serving/ccsrc)
link_directories(${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
link_directories(${CMKAE_BINARY_DIR}/securec/src)

file(GLOB_RECURSE UT_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(serving_ut ${UT_LIST} ${FINAL_SERVING_SRC})
target_link_libraries(serving_ut mindspore_serving::gtest)
target_link_libraries(serving_ut UT_PROTO_SRC_LIB mindspore_serving::event mindspore_serving::event_pthreads)
target_link_libraries(serving_ut ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} pthread mindspore::glog)
target_link_libraries(serving_ut ${SECUREC_LIBRARY})

