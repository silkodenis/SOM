cmake_minimum_required(VERSION 2.8)

project(tests)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

set(TEST_SOURCE main.cpp)
set(TEST_NAME "Test_OpenCL_Host")

add_executable(test_opencl_host ${TEST_SOURCE})

target_link_libraries(test_opencl_host ${OpenCL_LIBRARY})

add_test(NAME ${TEST_NAME} COMMAND test_opencl_host)