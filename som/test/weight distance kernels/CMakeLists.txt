cmake_minimum_required(VERSION 2.8)

project(tests)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(../../../som/include)

set(TEST_SOURCE main.cpp)
set(TEST_NAME "Test_weight_distance_kernels")

add_executable(test_weight_distance_kernels ${TEST_SOURCE})

target_link_libraries(test_weight_distance_kernels ${OpenCL_LIBRARY})
target_link_libraries(test_weight_distance_kernels som)	

add_test(NAME ${TEST_NAME} COMMAND test_weight_distance_kernels)