
add_library(protobuf::libprotobuf ALIAS protobuf::protobuf)
add_executable(protobuf::libprotoc ALIAS protobuf::protoc)

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
if(EXISTS ${grpc_ROOT}/lib64)
    set(gRPC_DIR "${grpc_ROOT}/lib64/cmake/grpc")
else()
    set(gRPC_DIR "${grpc_ROOT}/lib/cmake/grpc")
endif()
message("serving using grpc_DIR : " ${gPRC_DIR})

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
set(_REFLECTION gRPC::grpc++_reflection)

set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
set(_GRPC_PYTHON_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_python_plugin>)

# Proto file
# Generated sources
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/proto/")
file(GLOB_RECURSE PROTO_FILE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../mindspore_serving/proto/*.proto)
foreach(proto_file_with_path ${PROTO_FILE_LIST})
    message(------proto file: ${proto_file_with_path})
    get_filename_component(proto_I_DIR "../../" ABSOLUTE)
    get_filename_component(proto_file_absolute "${proto_file_with_path}" ABSOLUTE)
    string(REGEX REPLACE .*/proto/ "" proto_file ${proto_file_with_path})
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
            OUTPUT ${hw_proto_srcs} ${hw_proto_hdrs} ${hw_grpc_srcs} ${hw_grpc_hdrs} ${hw_py_pb2} ${hw_py_pb2_grpc}
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
endforeach()

add_library(PROTO_SRC_LIB STATIC ${PROTO_SRC_LIST})

include_directories("${CMAKE_CURRENT_BINARY_DIR}/mindspore_serving") # for proto header file
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/mindspore_serving")

# serving_common for c++ server and python interface
file(GLOB_RECURSE UT_SERVING_CORE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../mindspore_serving/ccsrc/common/*.cc"
        "../../mindspore_serving/ccsrc/master/*.cc"
        "../../mindspore_serving/ccsrc/worker/*.cc")

file(GLOB_RECURSE UT_SERVING_RMV_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../mindspore_serving/ccsrc/worker/inference/inference.cc")
list(REMOVE_ITEM UT_SERVING_CORE_SRC ${UT_SERVING_RMV_SRC})

file(GLOB_RECURSE UT_SERVING_STUB RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "stub/*.cc")
set(UT_SERVING_COMMON ${UT_SERVING_CORE_SRC} ${UT_SERVING_STUB})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(stub)
include_directories(stub/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../mindspore_serving/ccsrc)

link_directories(${CMKAE_BINARY_DIR}/securec/src)

add_library(serving_common SHARED ${UT_SERVING_COMMON})

target_link_options(serving_common PRIVATE -Wl,-init,mindspore_serving_log_init)

target_link_libraries(serving_common PRIVATE PROTO_SRC_LIB)
target_link_libraries(serving_common PRIVATE ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} pthread)
target_link_libraries(serving_common PRIVATE mindspore_serving::event mindspore_serving::event_pthreads)
target_link_libraries(serving_common PRIVATE mindspore_serving::glog)
target_link_libraries(serving_common PRIVATE mindspore_serving::eigen)
target_link_libraries(serving_common PRIVATE ${SECUREC_LIBRARY})
# copy libevent lib
file(GLOB_RECURSE LIBEVENT_LIB_LIST ${libevent_LIBPATH}/libevent* libevent_LIBPATH}/libevent_pthreads*)
file(COPY ${LIBEVENT_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)
# copy glog lib
file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libglog*)
file(COPY ${GLOG_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)
# copy grpc lib
file(GLOB_RECURSE GPRC_LIB_LIST ${grpc_LIBPATH}/lib*)
file(COPY ${GPRC_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)

# for cpp/serving_ut
set(CPP_UT_SERVING_CORE ${UT_SERVING_COMMON} ${UT_SERVING_ASCEND})
add_library(cpp_serving_common STATIC ${CPP_UT_SERVING_CORE})
target_link_libraries(cpp_serving_common PRIVATE PROTO_SRC_LIB)
target_link_libraries(cpp_serving_common PRIVATE ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} pthread)
target_link_libraries(cpp_serving_common PRIVATE mindspore_serving::event mindspore_serving::event_pthreads)
target_link_libraries(cpp_serving_common PRIVATE pthread mindspore_serving::glog)
target_link_libraries(cpp_serving_common PRIVATE mindspore_serving::eigen)
target_link_libraries(cpp_serving_common PRIVATE ${SECUREC_LIBRARY})

add_subdirectory("cpp")
add_subdirectory("python")
