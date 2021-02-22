# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")

link_directories(${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
file(GLOB_RECURSE UT_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "common/*.cc" "tests/*.cc")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

add_executable(serving_ut ${UT_LIST})
target_link_libraries(serving_ut PRIVATE mindspore_serving::gtest)
target_link_libraries(serving_ut PRIVATE -Wl,--whole-archive cpp_serving_common -Wl,--no-whole-archive)

# target_link_options(serving_ut PRIVATE -Wl,-init,mindspore_serving_log_init)

# disable auto rpath
set_target_properties(serving_ut PROPERTIES SKIP_BUILD_RPATH TRUE)
# copy gtest lib
file(GLOB_RECURSE GTEST_LIB_LIST ${gtest_LIBPATH}/libgtest*)
file(COPY ${GTEST_LIB_LIST} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
