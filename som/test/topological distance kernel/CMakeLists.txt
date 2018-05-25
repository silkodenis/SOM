cmake_minimum_required(VERSION 2.8)

project(tests)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(../../../som/include)

set(TEST_SOURCE main.cpp)
set(TEST_NAME "Test_topological_distance_kernel")

add_executable(test_topological_distance_kernel ${TEST_SOURCE})

target_link_libraries(test_topological_distance_kernel ${OpenCL_LIBRARY})
target_link_libraries(test_topological_distance_kernel som)	

add_test(NAME ${TEST_NAME} COMMAND test_topological_distance_kernel)