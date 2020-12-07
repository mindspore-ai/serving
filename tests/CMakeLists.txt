#add flags
message("================START BUILD TESTCASES=================")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

add_subdirectory("ut")
