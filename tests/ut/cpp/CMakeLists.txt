# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")

# Proto file
# Generated sources
file(GLOB_RECURSE PROTO_FILE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../../mindspore_serving/proto/*.proto)
ms_grpc_generate(PROTO_SRC_LIST PROTO_HDR_LIST ${PROTO_FILE_LIST})

include_directories("${CMAKE_BINARY_DIR}/mindspore_serving" ${CMAKE_BINARY_DIR}) # for proto header file
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/mindspore_serving")

# serving_common for c++ server and python interface
file(GLOB_RECURSE UT_SERVING_CORE_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore_serving/ccsrc/common/*.cc"
        "../../../mindspore_serving/ccsrc/master/*.cc"
        "../../../mindspore_serving/ccsrc/worker/*.cc")

file(GLOB_RECURSE UT_SERVING_RMV_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore_serving/ccsrc/worker/inference/inference.cc")
list(REMOVE_ITEM UT_SERVING_CORE_SRC ${UT_SERVING_RMV_SRC})

file(GLOB_RECURSE UT_SERVING_STUB RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "../stub/*.cc")
set(UT_SERVING_COMMON ${UT_SERVING_CORE_SRC} ${UT_SERVING_STUB})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../)
include_directories(../stub)
include_directories(../stub/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../mindspore_serving/ccsrc)

link_directories(${CMKAE_BINARY_DIR}/securec/src)
# copy libevent lib
file(GLOB_RECURSE LIBEVENT_LIB_LIST ${libevent_LIBPATH}/libevent* libevent_LIBPATH}/libevent_pthreads*)
file(COPY ${LIBEVENT_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# copy glog lib
file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libmindspore_serving_glog*)
file(COPY ${GLOG_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# copy grpc lib
file(GLOB_RECURSE GPRC_LIB_LIST ${grpc_LIBPATH}/lib*)
file(COPY ${GPRC_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# for cpp/serving_ut
set(CPP_UT_SERVING_CORE ${UT_SERVING_COMMON} ${UT_SERVING_ASCEND})
add_library(cpp_serving_common STATIC ${CPP_UT_SERVING_CORE} ${PROTO_SRC_LIST})
target_link_libraries(cpp_serving_common PRIVATE mindspore_serving::grpc++)
target_link_libraries(cpp_serving_common PRIVATE mindspore_serving::protobuf pthread rt)
target_link_libraries(cpp_serving_common PRIVATE mindspore_serving::event mindspore_serving::event_pthreads)
target_link_libraries(cpp_serving_common PRIVATE mindspore_serving::event_openssl)
target_link_libraries(cpp_serving_common PRIVATE pthread mindspore_serving::glog)
target_link_libraries(cpp_serving_common PRIVATE mindspore_serving::eigen)
target_link_libraries(cpp_serving_common PRIVATE ${SECUREC_LIBRARY})

# for test
link_directories(${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
file(GLOB_RECURSE UT_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "common/*.cc" "tests/*.cc")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

add_executable(serving_ut ${UT_LIST})
target_link_libraries(serving_ut PRIVATE mindspore_serving::gtest)
target_link_libraries(serving_ut PRIVATE -Wl,--whole-archive cpp_serving_common -Wl,--no-whole-archive)

# disable auto rpath
set_target_properties(serving_ut PROPERTIES SKIP_BUILD_RPATH TRUE)
# copy gtest lib
file(GLOB_RECURSE GTEST_LIB_LIST ${gtest_LIBPATH}/libgtest*)
file(COPY ${GTEST_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
