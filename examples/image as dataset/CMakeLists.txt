cmake_minimum_required(VERSION 2.8)

project(examples)

add_definitions(-DTARGET_IMAGE_PATH="${PROJECT_BINARY_DIR}/images/target_image.png")
add_definitions(-DTRAIN_IMAGE_PATH="${PROJECT_BINARY_DIR}/images/train_image.png")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(../../view)
include_directories(../../som/include/public)

set(SOURCE_EXE main.cpp)

add_executable(image_as_dataset ${SOURCE_EXE})

target_link_libraries(image_as_dataset ${OpenCL_LIBRARY})
target_link_libraries(image_as_dataset som_view)	

file(GLOB IMAGES images/*.png)
file(COPY ${IMAGES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/images)
