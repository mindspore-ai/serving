set(grpc_USE_STATIC_LIBS OFF)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 -O2 \
        -Dgrpc=mindspore_serving_grpc -Dgrpc_impl=mindspore_serving_grpc_impl -Dgrpc_core=mindspore_serving_grpc_core")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
else()
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2 \
        -Dgrpc=mindspore_serving_grpc -Dgrpc_impl=mindspore_serving_grpc_impl -Dgrpc_core=mindspore_serving_grpc_core")
    set(grpc_CFLAGS "-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
    if(NOT ENABLE_GLIBCXX)
        set(grpc_CXXFLAGS "${grpc_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(grpc_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

if(EXISTS ${protobuf_ROOT}/lib64)
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib64/cmake/protobuf")
else()
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib/cmake/protobuf")
endif()
message("grpc using Protobuf_DIR : " ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})

if(EXISTS ${absl_ROOT}/lib64)
    set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib64/cmake/absl")
else()
    set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib/cmake/absl")
endif()
message("grpc using absl_DIR : " ${_FINDPACKAGE_ABSL_CONFIG_DIR})

if(EXISTS ${re2_ROOT}/lib64)
    set(_FINDPACKAGE_RE2_CONFIG_DIR "${re2_ROOT}/lib64/cmake/re2")
else()
    set(_FINDPACKAGE_RE2_CONFIG_DIR "${re2_ROOT}/lib/cmake/re2")
endif()
message("grpc using re2_DIR : " ${_FINDPACKAGE_RE2_CONFIG_DIR})

if(EXISTS ${openssl_ROOT})
    set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${openssl_ROOT}")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/grpc/repository/archive/v1.36.1.tar.gz")
    set(MD5 "71252ebcd8e9e32a818a907dfd4b63cc")
else()
    set(REQ_URL "https://github.com/grpc/grpc/archive/v1.36.1.tar.gz")
    set(MD5 "90c93203e95e89af5f46738588217057")
endif()

mindspore_add_pkg(grpc
        VER 1.36.1
        LIBS mindspore_serving_grpc++ mindspore_serving_grpc mindspore_serving_gpr mindspore_serving_upb
        mindspore_serving_address_sorting
        EXE grpc_cpp_plugin grpc_python_plugin
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/grpc/grpc.patch001
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DBUILD_SHARED_LIBS=ON
        -DgRPC_INSTALL:BOOL=ON
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:PATH=${zlib_ROOT}
        -DgRPC_ABSL_PROVIDER:STRING=package
        -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR}
        -DgRPC_CARES_PROVIDER:STRING=package
        -Dc-ares_DIR:PATH=${c-ares_ROOT}/lib/cmake/c-ares
        -DgRPC_SSL_PROVIDER:STRING=package
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DgRPC_RE2_PROVIDER:STRING=package
        -Dre2_DIR:PATH=${_FINDPACKAGE_RE2_CONFIG_DIR}
        )

include_directories(${grpc_INC})

add_library(mindspore_serving::grpc++ ALIAS grpc::mindspore_serving_grpc++)

# link other grpc libs
target_link_libraries(grpc::mindspore_serving_grpc++ INTERFACE grpc::mindspore_serving_grpc grpc::mindspore_serving_gpr
  grpc::mindspore_serving_upb grpc::mindspore_serving_address_sorting)

# modify mindspore macro define
add_compile_definitions(grpc=mindspore_serving_grpc)
add_compile_definitions(grpc_impl=mindspore_serving_grpc_impl)
add_compile_definitions(grpc_core=mindspore_serving_grpc_core)

function(ms_grpc_generate c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_grpc_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})

    foreach(proto_file_with_path ${ARGN})
        message(proto_file_with_path: ${proto_file_with_path})
        get_filename_component(proto_file_absolute "${proto_file_with_path}" ABSOLUTE)
        message(proto_file_absolute: ${proto_file_absolute})
        get_filename_component(file_dir ${proto_file_absolute} DIRECTORY)
        get_filename_component(proto_I_DIR "${file_dir}/../../" ABSOLUTE)
        get_filename_component(proto_file ${proto_file_absolute} NAME)
        get_filename_component(proto_file_prefix ${proto_file_absolute} NAME_WE)
        set(proto_file_relative "mindspore_serving/proto/${proto_file}")

        set(protoc_output_prefix ${CMAKE_BINARY_DIR}/mindspore_serving/proto)
        set(hw_proto_srcs "${protoc_output_prefix}/${proto_file_prefix}.pb.cc")
        set(hw_proto_hdrs "${protoc_output_prefix}/${proto_file_prefix}.pb.h")
        set(hw_grpc_srcs "${protoc_output_prefix}/${proto_file_prefix}.grpc.pb.cc")
        set(hw_grpc_hdrs "${protoc_output_prefix}/${proto_file_prefix}.grpc.pb.h")
        set(hw_py_pb2 "${protoc_output_prefix}/${proto_file_prefix}_pb2.py")
        set(hw_py_pb2_grpc "${protoc_output_prefix}/${proto_file_prefix}_pb2_grpc.py")
        add_custom_command(
                OUTPUT ${hw_proto_srcs} ${hw_proto_hdrs} ${hw_grpc_srcs} ${hw_grpc_hdrs} ${hw_py_pb2} ${hw_py_pb2_grpc}
                WORKING_DIRECTORY ${proto_I_DIR}
                COMMAND $<TARGET_FILE:protobuf::protoc>
                ARGS --grpc_out "${CMAKE_BINARY_DIR}"
                --cpp_out "${CMAKE_BINARY_DIR}"
                -I "${proto_I_DIR}"
                --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc::grpc_cpp_plugin>
                "${proto_file_relative}"
                COMMAND $<TARGET_FILE:protobuf::protoc>
                ARGS --grpc_out "${CMAKE_BINARY_DIR}"
                --python_out "${CMAKE_BINARY_DIR}"
                -I "${proto_I_DIR}"
                --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc::grpc_python_plugin>
                "${proto_file_relative}"
                DEPENDS "${proto_file_absolute}")

        list(APPEND ${c_var} ${hw_proto_srcs} ${hw_grpc_srcs})
        list(APPEND ${h_var} ${hw_proto_hdrs} ${hw_grpc_hdrs})
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()
