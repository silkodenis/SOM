cmake_minimum_required(VERSION 2.8)

project(examples)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(../../view/include)
include_directories(../../som/include/public)

set(SOURCE_EXE main.cpp)

add_executable(data_approximation ${SOURCE_EXE})

target_link_libraries(data_approximation ${OpenCL_LIBRARY})
target_link_libraries(data_approximation som_view)	
