set(SECURE_CXX_FLAGS "")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(SECURE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
endif()
set(_ms_tmp_CMAKE_CXX_FLAGS_F ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

# define third party library download function
include(cmake/utils.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/eigen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/dependency_securec.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/json.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/protobuf.cmake)

# build dependencies of gRPC
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/absl.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/c-ares.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/zlib.cmake)
# build gRPC
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/grpc.cmake)
# build event
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/libevent.cmake)

include(${CMAKE_SOURCE_DIR}/cmake/external_libs/pybind11.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/gtest.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/glog.cmake)

set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS_F})

if(MS_BACKEND)
    include(${CMAKE_SOURCE_DIR}/cmake/dependency_ms.cmake)
endif()
