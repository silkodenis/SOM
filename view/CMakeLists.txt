cmake_minimum_required(VERSION 2.8)

project(som)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_MACOSX_RPATH 1)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../som/include/public)

set(SOURCE_LIB som_cv_view.cpp)
set(HEADERS_LIB som_cv_view.hpp)

if(NOT CMAKE_GENERATOR STREQUAL Xcode)
	file(COPY ${HEADERS_LIB} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()


if(BUILD_SHARED_LIBS)
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