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

# libevent
install(FILES ${libevent_LIBPATH}/libevent-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent-2.1.so.7 COMPONENT mindspore_serving)
install(FILES ${libevent_LIBPATH}/libevent_core-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_core-2.1.so.7 COMPONENT mindspore)
install(FILES ${libevent_LIBPATH}/libevent_openssl-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_openssl-2.1.so.7 COMPONENT mindspore_serving)
install(FILES ${libevent_LIBPATH}/libevent_pthreads-2.1.so.7.0.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_pthreads-2.1.so.7 COMPONENT mindspore_serving)

# grpc
install(FILES ${grpc_LIBPATH}/libmindspore_serving_grpc++.so.1.36.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_serving_grpc++.so.1 COMPONENT mindspore_serving)
install(FILES ${grpc_LIBPATH}/libmindspore_serving_grpc.so.15.0.0
  DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_serving_grpc.so.15 COMPONENT mindspore_serving)
install(FILES ${grpc_LIBPATH}/libmindspore_serving_gpr.so.15.0.0
  DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_serving_gpr.so.15 COMPONENT mindspore_serving)
install(FILES ${grpc_LIBPATH}/libmindspore_serving_upb.so.15.0.0
  DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_serving_upb.so.15 COMPONENT mindspore_serving)
install(FILES ${grpc_LIBPATH}/libmindspore_serving_address_sorting.so.15.0.0
  DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_serving_address_sorting.so.15 COMPONENT mindspore_serving)

# glog
install(FILES ${glog_LIBPATH}/libmindspore_serving_glog.so.0.4.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_serving_glog.so.0 COMPONENT mindspore)

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
        ${CMAKE_SOURCE_DIR}/mindspore_serving/server
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
