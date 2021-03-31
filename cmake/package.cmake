# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# set package information
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindspore_serving)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindspore_serving)

set(CPACK_MS_PACKAGE_NAME "mindspore_serving")
include(CPack)

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_LIB_DIR "lib")

file(GLOB_RECURSE LIBEVENT_LIB_LIST
        ${libevent_LIBPATH}/libevent*
        ${libevent_LIBPATH}/libevent_pthreads*
        )
install(
        FILES ${LIBEVENT_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore_serving
)

file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libglog*)
install(
        FILES ${GLOG_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore_serving
)

# set python files
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/mindspore_serving/*.py)
install(
        FILES ${MS_PY_LIST}
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore_serving
)

install(
        TARGETS _mindspore_serving
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT mindspore_serving
)
install(
        TARGETS serving_common
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore_serving
)
install(
        TARGETS serving_ascend
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore_serving
)
install(
        DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindspore_serving/master
        ${CMAKE_SOURCE_DIR}/mindspore_serving/worker
        ${CMAKE_SOURCE_DIR}/mindspore_serving/common
        ${CMAKE_SOURCE_DIR}/mindspore_serving/client
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore_serving
)
install(
        FILES ${CMAKE_SOURCE_DIR}/build/mindspore_serving/mindspore_serving/mindspore_serving/proto/ms_service_pb2.py
        ${CMAKE_SOURCE_DIR}/build/mindspore_serving/mindspore_serving/mindspore_serving/proto/ms_service_pb2_grpc.py
        DESTINATION ${INSTALL_PY_DIR}/proto
        COMPONENT mindspore_serving
)
