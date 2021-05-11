## define customized find functions, print customized error messages
function(find_required_package pkg_name)
    find_package(${pkg_name})
    if(NOT ${pkg_name}_FOUND)
        message(FATAL_ERROR "Required package ${pkg_name} not found, please install the package and try"
                " building mindspore_serving again.")
    endif()
endfunction()

## find python, quit if the found python is static
set(Python3_USE_STATIC_LIBS FALSE)
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    message("Python3 found, version: ${Python3_VERSION}")
    message("Python3 library path: ${Python3_LIBRARY}")
    message("Python3 interpreter: ${Python3_EXECUTABLE}")
elseif(Python3_LIBRARY AND Python3_EXECUTABLE AND
        ${Python3_VERSION} VERSION_GREATER_EQUAL "3.7.0" AND ${Python3_VERSION} VERSION_LESS "3.10.0")
    message(WARNING "Maybe python3 environment is broken.")
    message("Python3 library path: ${Python3_LIBRARY}")
    message("Python3 interpreter: ${Python3_EXECUTABLE}")
else()
    message(FATAL_ERROR "Python3 not found, please install Python>=3.7.5, and set --enable-shared "
            "if you are building Python locally")
endif()

## packages used both on windows and linux
if(DEFINED ENV{MS_PATCH_PATH})
    find_program(Patch_EXECUTABLE patch PATHS $ENV{MS_PATCH_PATH})
    set(Patch_FOUND ${Patch_EXECUTABLE})
else()
    find_package(Patch)
endif()
if(NOT Patch_FOUND)
    message(FATAL_ERROR "Patch not found, please set environment variable MS_PATCH_PATH to path where Patch is located,"
            " usually found in GIT_PATH/usr/bin on Windows")
endif()
message(PATCH_EXECUTABLE = ${Patch_EXECUTABLE})

find_required_package(Threads)
