set(STUB_DIR ../stub)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

file(GLOB_RECURSE UT_SERVING_STUB RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${STUB_DIR}/cxx_api/*.cc"
        "${STUB_DIR}/graph_impl_stub.cc" "${STUB_DIR}/include/utils/*.cc")

add_library(mindspore SHARED ${UT_SERVING_STUB})

set(UT_SERVING_COMMON ${UT_SERVING_CORE_SRC} ${UT_SERVING_STUB})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${STUB_DIR}/..)
include_directories(${STUB_DIR})
include_directories(${STUB_DIR}/include)

include_directories(${ROOT_DIR}/third_party)
link_directories(${CMKAE_BINARY_DIR}/securec/src)

target_link_libraries(mindspore PRIVATE ${SECUREC_LIBRARY} pthread)
target_link_libraries(mindspore PRIVATE mindspore_serving::glog)

set(LIBRARY_OUTPUT_PATH ${ROOT_DIR}/build/package/tests/mindspore/lib/)

# copy mindspore include
file(COPY ${STUB_DIR}/include/api DESTINATION ${ROOT_DIR}/build/package/tests/mindspore/include)
