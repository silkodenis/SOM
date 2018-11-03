cmake_minimum_required(VERSION 2.8)

project(view)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../som/include/public)
include_directories(include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(SOURCE_LIB 
src/cv_view.cpp
src/pseudocolor_schemes.cpp)

set(HEADERS_LIB 
include/cv_view.hpp
include/pseudocolor_schemes.hpp)

if(NOT CMAKE_GENERATOR STREQUAL Xcode)
	file(COPY ${HEADERS_LIB} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()


if(BUILD_SHARED_LIBS)
set(CMAKE_MACOSX_RPATH 1)

add_library(som_view SHARED ${SOURCE_LIB} ${HEADERS_LIB})
target_link_libraries(som_view som)
target_link_libraries(som_view ${OpenCV_LIBS})

set_target_properties(som_view PROPERTIES PUBLIC_HEADER "${HEADERS_LIB}")
install(TARGETS som_view 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/som_view
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/som_view
)

else()
add_library(som_view STATIC ${SOURCE_LIB} ${HEADERS_LIB})
target_link_libraries(som_view som)
target_link_libraries(som_view ${OpenCV_LIBS})

set_target_properties(som_view PROPERTIES PUBLIC_HEADER "${HEADERS_LIB}")
install(TARGETS som_view 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/som_view
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/som_view
)

endif()
